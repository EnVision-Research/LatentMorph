from __future__ import annotations

import os
import sys
import time
import warnings
import math

import torch
from transformers import AutoModelForCausalLM
from transformers.utils import logging as hf_logging
import torch.distributed as dist

from janus.models import MultiModalityCausalLM, VLChatProcessor
from latent_sft.models.config_io import build_latent_controller_config
from latent_control.controller import LatentController

from . import ddp_utils
from latent_sft.models.data import build_dataloader, resolve_train_files
from latent_sft.models.latent_morph import LatentMorph


@torch.no_grad()
def encode_gt_image_ids(model: MultiModalityCausalLM, gt_img: torch.Tensor) -> torch.LongTensor:
    _, _, all_image_ids = model.gen_vision_model.encode(gt_img)
    image_ids = all_image_ids[2].to(dtype=torch.long)
    if image_ids.dim() == 1:
        image_ids = image_ids.unsqueeze(0)
    return image_ids


class TwiGControlTrainer:
    """
    Trainer style similar to janus-sft: split setup / data / model / run_step / save.
    Only train LatentController; other Janus modules are frozen.
    """

    def __init__(self, cfg: dict, args):
        self.cfg = cfg
        self.args = args
        self.dist = ddp_utils.init_distributed(args.device)

        # Only let rank0 print: silence stdout on other ranks but keep stderr
        # (otherwise we may lose tracebacks on distributed failures).
        if self.dist.ddp and (not ddp_utils.is_main_process()):
            try:
                sys.stdout = open(os.devnull, "w")
            except Exception:
                pass

        # More timely stdout/stderr flushing (useful for multi-process + tee).
        try:
            sys.stdout.reconfigure(line_buffering=True)
            sys.stderr.reconfigure(line_buffering=True)
        except Exception:
            pass

        # Reduce noise.
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message="Special tokens have been added*")
        try:
            hf_logging.set_verbosity_error()
        except Exception:
            pass

        self.device = self.dist.device
        # By default, place checkpoints outside LatentMorph (reduce repo size/pressure).
        _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.out_dir = args.out_dir.strip() or os.path.abspath(os.path.join(_repo_root, "..", "checkpoints_control_image_loss2"))
        os.makedirs(self.out_dir, exist_ok=True)

        self.model: MultiModalityCausalLM | None = None
        self.processor: VLChatProcessor | None = None
        self.tokenizer = None
        self.controller = None
        self.loss_model = None
        self.opt = None
        self.scaler = None
        self.loader = None

        self.img_size = int(cfg.get("img_size", 384))
        self.patch_size = int(cfg.get("patch_size", 16))
        self.image_token_num = int(cfg.get("image_token_num", 576))
        self.stages = int(cfg.get("stages", 1))
        self.part_template = str(cfg.get("part_template", "{i}-part"))
        stage_prompt_cfg = cfg.get("stage_prompt", {}) if isinstance(cfg.get("stage_prompt", {}), dict) else {}
        self.use_understanding = bool(stage_prompt_cfg.get("use_understanding", True))
        self.understanding_max_tokens = int(stage_prompt_cfg.get("understanding_max_tokens", 128))
        generation_cfg = cfg.get("generation", {}) if isinstance(cfg.get("generation", {}), dict) else {}
        self.cfg_weight = float(generation_cfg.get("cfg_weight", 5.0))
        self.temperature = float(generation_cfg.get("temperature", 1.0))

    @staticmethod
    @torch.no_grad()
    def _freeze_all_params(m: torch.nn.Module):
        for p in m.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _set_trainable_control(controller: LatentController):
        for p in controller.parameters():
            p.requires_grad_(True)

    def setup(self):
        # File list: load all first, then shard evenly across GPUs/ranks (no per-file splitting by rank).
        train_files = resolve_train_files(self.cfg)

        # Print total/shard sample stats after building the dataloader to avoid repetition/confusion.

        # Processor + model (frozen).
        model_path = self.cfg.get("model_path", "deepseek-ai/Janus-Pro-7B")
        if ddp_utils.is_main_process():
            print(f"[setup] model_path = {model_path}", flush=True)
            print("[setup] loading VLChatProcessor...", flush=True)
        # Force local loading: avoid transformers' has_file()/HEAD(timeout=10) causing DDP failures due to network jitter.
        try:
            self.processor = VLChatProcessor.from_pretrained(model_path, local_files_only=True)
        except TypeError:
            # Some versions/custom classes may not support local_files_only.
            self.processor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer

        if ddp_utils.is_main_process():
            print("[setup] loading model weights...", flush=True)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        self.model = self.model.to(device=self.device, dtype=torch.float16).eval()
        self._freeze_all_params(self.model)
        if ddp_utils.is_main_process():
            print("[setup] model loaded + frozen.", flush=True)

        # Requirement: disable gradient checkpointing and keep KV cache (use_cache=True).
        # Note: when checkpointing is enabled, HF often forces use_cache=False, which breaks KV-cache injection/sampling.
        try:
            lm = getattr(self.model, "language_model", None)
            if lm is not None and hasattr(lm, "gradient_checkpointing_disable"):
                lm.gradient_checkpointing_disable()
            if lm is not None and hasattr(lm, "config") and hasattr(lm.config, "use_cache"):
                lm.config.use_cache = True
            # When the large model is frozen, keeping eval is more stable (no dropout); only controller is trained.
            if lm is not None and hasattr(lm, "eval"):
                lm.eval()
                if ddp_utils.is_main_process():
                    print("[setup] gradient checkpointing disabled; language_model.use_cache=True.", flush=True)
        except Exception as e:
            if ddp_utils.is_main_process():
                print(f"[setup] warning: failed to disable gradient checkpointing / enable cache: {e}", flush=True)

        # controller (the only trainable module)
        if ddp_utils.is_main_process():
            print("[setup] building LatentController (trainable control module)...", flush=True)
        latent_cfg = build_latent_controller_config(self.cfg)
        latent_cfg.enabled = True
        # No TBPTT, but we still need at least one injection per stage; otherwise the controller may learn nothing.
        # Use per-stage token count as an approximate upper bound.
        per_stage_tokens = max(1, int(self.image_token_num // max(1, self.stages)))
        orig = int(getattr(latent_cfg.trigger, "check_every", 1))
        # Also ensure triggering can happen within the backprop window, and after injection
        # there is at least 1 token whose loss can see the injection effect.
        win = int(getattr(latent_cfg, "img_hidden_window", 64))
        upper_by_window = (win - 1) if win > 1 else 1
        upper_by_stage = (per_stage_tokens - 1) if per_stage_tokens > 1 else 1
        upper = int(max(1, min(upper_by_stage, upper_by_window)))
        latent_cfg.trigger.check_every = int(max(1, min(orig, upper)))
        # Do not aggressively increase max_triggers_per_image: it would trigger think/translator too often per image
        # and make training very slow. Keep the limit from config.json.
        latent_cfg.max_triggers_per_image = int(max(1, int(latent_cfg.max_triggers_per_image)))

        d_model = int(self.model.language_model.get_input_embeddings().weight.shape[1])
        # Keep these for "periodic inference during training"
        # (especially under FSDP where we need a non-FSDP controller rebuilt from state_dict).
        self._latent_cfg = latent_cfg
        self._d_model = d_model
        ctrl = LatentController(d_model=d_model, tokenizer=self.tokenizer, cfg=latent_cfg).to(self.device)
        # Keep controller params in fp32 (otherwise GradScaler may error during unscale of FP16 grads).
        ctrl = ctrl.to(dtype=torch.float32).train()
        self._set_trainable_control(ctrl)
        self.controller = ctrl
        if ddp_utils.is_main_process():
            print("[setup] controller ready.", flush=True)

        # Loss model (forward runs the full LatentMorph teacher-forcing; only controller is trainable).
        loss_model = LatentMorph(
            frozen_model=self.model,
            processor=self.processor,
            tokenizer=self.tokenizer,
            controller=self.controller,
            img_size=self.img_size,
            patch_size=self.patch_size,
            image_token_num=self.image_token_num,
            stages=self.stages,
            part_template=self.part_template,
            use_understanding=self.use_understanding,
            understanding_max_tokens=self.understanding_max_tokens,
            cfg_weight=self.cfg_weight,
            temperature=self.temperature,
        ).to(self.device)
        if self.dist.ddp:
            # Align with janus-sft: prefer FSDP (wrap only small trainable modules; the big model is not in the module tree).
            loss_model = ddp_utils.fsdp_wrap(loss_model) if self.dist.fsdp else ddp_utils.ddp_wrap(loss_model, self.dist.local_rank)
        self.loss_model = loss_model

        # data
        if ddp_utils.is_main_process():
            print(f"[setup] building dataloader from {len(train_files)} files", flush=True)
        # Take only the first 50k samples, then shard across GPUs via DistributedSampler.
        _td = self.cfg.get("train_data", {}) if isinstance(self.cfg.get("train_data", {}), dict) else {}
        max_samples = int(_td.get("max_samples", 50000))
        self.loader = build_dataloader(
            train_files,
            img_size=self.img_size,
            image_key=self.args.image_key,
            caption_key=self.args.caption_key,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            device=self.device,
            ddp=bool(self.dist.ddp),
            rank=int(self.dist.rank),
            world_size=int(self.dist.world_size),
            in_memory=True,
            max_samples=max_samples,
        )
        # Data stats: total samples & per-rank shard samples (used for ETA estimation).
        total_samples = None
        shard_samples = None
        try:
            if hasattr(self.loader, "dataset") and hasattr(self.loader.dataset, "__len__"):
                total_samples = len(self.loader.dataset)
            if hasattr(self.loader, "sampler") and self.loader.sampler is not None and hasattr(self.loader.sampler, "__len__"):
                shard_samples = len(self.loader.sampler)
        except Exception:
            pass
        self.total_samples = total_samples
        self.shard_samples = shard_samples
        if ddp_utils.is_main_process():
            msg = "[setup] dataloader ready."
            if total_samples is not None:
                msg += f" total_samples={total_samples}"
            if shard_samples is not None:
                msg += f" shard_per_rank={shard_samples} (world_size={self.dist.world_size})"
            print(msg, flush=True)
        else:
            if shard_samples is not None:
                print(f"[setup][rank{self.dist.rank}] shard_samples={shard_samples} / total={total_samples}", flush=True)

        # optimizer / scaler
        trainable_params = [p for p in self.loss_model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(trainable_params, lr=self.args.lr, weight_decay=self.args.weight_decay, foreach=False)
        if self.dist.ddp and self.dist.fsdp:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
            self.scaler = ShardedGradScaler(enabled=(self.device.type == "cuda"))
        else:
            self.scaler = torch.amp.GradScaler("cuda", enabled=(self.device.type == "cuda"))

    def _infer_every_n_steps(self) -> int:
        # Requirement: run inference check every 50 steps.
        return 50

    @torch.inference_mode()
    def save_train_check(self, *, step: int, prompt: str, ctrl_state: dict | None = None):
        """
        Periodic inference check during training:
        - Save only on rank0
        - Overwrite the "latest" outputs each time (keep one image + one text file for quick viewing)
        - Under FSDP: do NOT run inference directly on the FSDP module (it may trigger collectives and require all ranks).
          Instead, rebuild a non-FSDP controller from the just-saved controller state, and run full inference on rank0 only.
        """
        if not ddp_utils.is_main_process():
            return
        if self.model is None or self.processor is None or self.tokenizer is None:
            return
        if self.loss_model is None:
            return

        out_dir = os.path.join(self.out_dir, "train_check")
        os.makedirs(out_dir, exist_ok=True)

        # Choose controller state: prefer in-memory state from save_latest; otherwise read ckpt_latest.pt from disk.
        if ctrl_state is None:
            ckpt_path = os.path.join(self.out_dir, "ckpt_latest.pt")
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location="cpu")
                ctrl_state = ckpt.get("controller", None)
        if ctrl_state is None:
            print(f"[infer-check] skip (no ctrl_state) step={step}", flush=True)
            return

        if getattr(self, "_latent_cfg", None) is None or getattr(self, "_d_model", None) is None:
            print(f"[infer-check] skip (missing _latent_cfg/_d_model) step={step}", flush=True)
            return

        # Build a temporary controller (non-FSDP) and load weights.
        tmp_ctrl = LatentController(d_model=int(self._d_model), tokenizer=self.tokenizer, cfg=self._latent_cfg).to(self.device).eval()
        # FSDP-saved state_dict may contain a "controller." prefix (because it's from the whole loss_model state_dict).
        if isinstance(ctrl_state, dict) and any(k.startswith("controller.") for k in ctrl_state.keys()):
            stripped = {k[len("controller.") :]: v for k, v in ctrl_state.items() if k.startswith("controller.")}
            tmp_ctrl.load_state_dict(stripped, strict=False)
        else:
            tmp_ctrl.load_state_dict(ctrl_state, strict=False)

        # Temporary LatentMorph (reuse frozen large model).
        from latent_sft.models.latent_morph import LatentMorph

        tmp_morph = LatentMorph(
            frozen_model=self.model,
            processor=self.processor,
            tokenizer=self.tokenizer,
            controller=tmp_ctrl,
            img_size=self.img_size,
            patch_size=self.patch_size,
            image_token_num=self.image_token_num,
            stages=self.stages,
            part_template=self.part_template,
            use_understanding=True,
            understanding_max_tokens=self.understanding_max_tokens,
            cfg_weight=self.cfg_weight,
            temperature=self.temperature,
        ).to(self.device)

        # During inference: temporarily switch to eval + enable cache; restore training settings afterwards.
        lm = getattr(self.model, "language_model", None)
        was_train = bool(getattr(lm, "training", False)) if lm is not None else False
        orig_use_cache = None
        try:
            if lm is not None and hasattr(lm, "eval"):
                lm.eval()
            if lm is not None and hasattr(lm, "config") and hasattr(lm.config, "use_cache"):
                orig_use_cache = bool(lm.config.use_cache)
                lm.config.use_cache = True

            image_ids, inj = tmp_morph.generate_image_tokens(base_prompt=str(prompt), seed=int(step))
            img = tmp_morph.decode_image_tokens_to_pil(image_ids)[0]

            # Save with step-based names (keep history): step_000050.png + step_000050.txt
            img_path = os.path.join(out_dir, f"step_{int(step):06d}.png")
            txt_path = os.path.join(out_dir, f"step_{int(step):06d}.txt")
            img.save(img_path)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"step: {step}\n")
                f.write(f"inj: {inj}\n")
                f.write(f"prompt: {str(prompt)}\n")

            # Also overwrite latest.png/latest.txt for quick access to the newest result.
            latest_img = os.path.join(out_dir, "latest.png")
            latest_txt = os.path.join(out_dir, "latest.txt")
            try:
                img.save(latest_img)
                with open(latest_txt, "w", encoding="utf-8") as f:
                    f.write(f"step: {step}\n")
                    f.write(f"inj: {inj}\n")
                    f.write(f"prompt: {str(prompt)}\n")
            except Exception:
                pass

            # Optional: save pure Janus-Pro baseline (no controller / no injection).
            # Switches:
            # - env var TWIG_SAVE_STEP_BASE=1
            # - or config.json train_check.save_base=true
            try:
                import os as _os

                save_base_env = str(_os.environ.get("TWIG_SAVE_STEP_BASE", "0")).strip().lower()
                save_base = save_base_env in ("1", "true", "yes", "y", "on")
                if (not save_base) and isinstance(getattr(self, "cfg", None), dict):
                    tc = self.cfg.get("train_check", {})
                    if isinstance(tc, dict):
                        save_base = bool(tc.get("save_base", False))

                if save_base:
                    # Generate a pair: identical prefix token/KV cache, diverge only at inj (easy to compare control effect).
                    ctrl_ids, base_ids, inj2 = tmp_morph.generate_control_and_base_shared_prefix(
                        base_prompt=str(prompt), seed=int(step)
                    )
                    # Replace current step's control output with ctrl_ids to ensure alignment with base.
                    image_ids = ctrl_ids
                    inj = int(inj2)
                    img = tmp_morph.decode_image_tokens_to_pil(image_ids)[0]

                    base_img = tmp_morph.decode_image_tokens_to_pil(base_ids)[0]
                    base_path = os.path.join(out_dir, f"step_{int(step):06d}_base.png")
                    base_img.save(base_path)
                    # Also overwrite a latest baseline for quick viewing.
                    base_latest = os.path.join(out_dir, "latest_base.png")
                    base_img.save(base_latest)
                    print(f"[infer-check] saved base {base_path}", flush=True)
            except Exception as e:
                print(f"[infer-check] base skipped/failed step={step}: {e}", flush=True)

            print(f"[infer-check] saved {img_path} (step={step}, inj={inj})", flush=True)
        finally:
            try:
                if lm is not None and orig_use_cache is not None and hasattr(lm, "config") and hasattr(lm.config, "use_cache"):
                    lm.config.use_cache = orig_use_cache
                if lm is not None and was_train and hasattr(lm, "train"):
                    lm.train()
            except Exception:
                pass
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def save_latest(self, step: int) -> dict | None:
        # Backward-compatible wrapper: save a step ckpt and update ckpt_latest.pt.
        return self.save_checkpoint(step=step, keep_latest=True)

    def _controller_state_for_infer(self) -> dict | None:
        """
        Only for in-memory inference in save_train_check during training:
        - Non-FSDP: can directly use current controller.state_dict()
        - FSDP: do not gather full_state_dict here (would block/require all ranks); return None and fall back to ckpt_latest.pt on disk
        """
        if self.loss_model is None:
            return None
        if self.dist.ddp and self.dist.fsdp:
            return None
        if not ddp_utils.is_main_process():
            return None
        base = self.loss_model.module if self.dist.ddp else self.loss_model
        try:
            return base.controller.state_dict()
        except Exception:
            return None

    def save_checkpoint(self, *, step: int, keep_latest: bool = True) -> dict | None:
        """
        Save a checkpoint (named by step; no overwriting of history), and optionally update ckpt_latest.pt.
        Returns controller state (rank0) for reuse by save_train_check.
        """
        if self.loss_model is None:
            raise RuntimeError("loss_model is None")
        if self.opt is None:
            raise RuntimeError("opt is None")

        ctrl_state = None

        if self.dist.ddp and self.dist.fsdp:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import StateDictType
            from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig

            # NOTE: FSDP full_state_dict gather requires all ranks to participate.
            with FSDP.state_dict_type(
                self.loss_model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            ):
                full_state = self.loss_model.state_dict()
            if not ddp_utils.is_main_process():
                return None
            ctrl_state = full_state
        else:
            if not ddp_utils.is_main_process():
                return None
            base = self.loss_model.module if self.dist.ddp else self.loss_model
            ctrl_state = base.controller.state_dict()

        ckpt = {
            "step": int(step),
            "controller": ctrl_state,
            "opt": self.opt.state_dict(),
            "args": vars(self.args),
        }

        # 1) Step-named checkpoint: ckpt_step_000100.pt (keep history)
        out_step = os.path.join(self.out_dir, f"ckpt_step_{int(step):06d}.pt")
        tmp = out_step + ".tmp"
        torch.save(ckpt, tmp)
        os.replace(tmp, out_step)

        # 2) Optional: update ckpt_latest.pt
        if keep_latest:
            out_latest = os.path.join(self.out_dir, "ckpt_latest.pt")
            tmp2 = out_latest + ".tmp"
            torch.save(ckpt, tmp2)
            os.replace(tmp2, out_latest)

        return ctrl_state

    def run_step(self, batch):
        assert self.model is not None and self.processor is not None
        assert self.loss_model is not None
        gt_img, caps = batch
        gt_img = gt_img.to(device=self.device, dtype=torch.float16, non_blocking=True)
        base_prompt = str(caps[0])

        with torch.no_grad():
            gt_ids = encode_gt_image_ids(self.model, gt_img)

        t0 = time.perf_counter()
        loss = self.loss_model(base_prompt=base_prompt, gt_image_ids=gt_ids)
        pred_time_s = float(time.perf_counter() - t0)
        if not loss.requires_grad:
            raise RuntimeError(
                "loss.requires_grad=False: this indicates controller injection did not happen, "
                "or the injection did not affect subsequent loss.\n"
                "Please check config.json: latent_control.enabled=true and trigger.check_every is small enough; "
                "and max_triggers_per_image is large enough."
            )

        return loss, pred_time_s

    def train(self):
        self.setup()
        assert self.loader is not None

        t0 = time.time()

        # Single pass over the data; step-based logging only.
        max_batches_per_epoch = int(getattr(self.args, "max_batches_per_epoch", 0))
        if max_batches_per_epoch < 0:
            max_batches_per_epoch = 0

        global_step = 0
        # ETA: estimate remaining prediction time using EMA of pred_time (forward).
        pred_time_ema = None
        pred_beta = 0.9

        # Expected total steps (ETA only; may be biased if decode failures are skipped by collate).
        expected_steps = None
        try:
            if max_batches_per_epoch > 0:
                expected_steps = int(max_batches_per_epoch)
            elif getattr(self, "shard_samples", None) is not None:
                bs = int(getattr(self.args, "batch_size", 1))
                expected_steps = int(math.ceil(float(self.shard_samples) / float(max(1, bs))))
        except Exception:
            expected_steps = None

        def _fmt_secs(x: float) -> str:
            x = int(max(0, round(float(x))))
            h = x // 3600
            m = (x % 3600) // 60
            s = x % 60
            return f"{h:02d}:{m:02d}:{s:02d}"

        for batch in self.loader:
            if batch is None:
                continue
            # Use the first caption in the current batch for periodic inference (aligned with training data).
            try:
                _caps = batch[1]
                infer_prompt = str(_caps[0]) if isinstance(_caps, (list, tuple)) and _caps else ""
            except Exception:
                infer_prompt = ""

            step_t0 = time.perf_counter()
            loss, pred_time_s = self.run_step(batch)  # scalar + seconds

            self.opt.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.opt)
            self.scaler.update()

            global_step += 1

            # Log loss every step (under DDP, all-reduce avg first, then rank0 prints).
            step_loss = loss.detach().to(torch.float32)
            step_pred_t = torch.tensor(float(pred_time_s), device=self.device, dtype=torch.float32)
            if self.dist.ddp and dist.is_initialized():
                dist.all_reduce(step_loss, op=dist.ReduceOp.SUM)
                step_loss = step_loss / float(self.dist.world_size)
                dist.all_reduce(step_pred_t, op=dist.ReduceOp.SUM)
                step_pred_t = step_pred_t / float(self.dist.world_size)

            if ddp_utils.is_main_process():
                step_time_s = float(time.perf_counter() - step_t0)
                pt = float(step_pred_t.item())
                pred_time_ema = pt if pred_time_ema is None else (pred_beta * pred_time_ema + (1.0 - pred_beta) * pt)
                eta_pred_s = None
                if expected_steps is not None:
                    remain = int(max(0, expected_steps - global_step))
                    eta_pred_s = float(pred_time_ema * remain)
                eta_str = _fmt_secs(eta_pred_s) if eta_pred_s is not None else "NA"
                print(
                    f"[step {global_step}] loss={step_loss.item():.6f} "
                    f"pred_time={pt:.3f}s eta_pred={eta_str} step_time={step_time_s:.3f}s",
                    flush=True,
                )

            # Save a checkpoint every 100 steps (keep history; do not write every step).
            ctrl_state = None
            if global_step % 100 == 0:
                ctrl_state = self.save_checkpoint(step=global_step, keep_latest=True)

            # Run inference check every 50 steps (save only the latest image + text).
            n_infer = self._infer_every_n_steps()
            if n_infer > 0 and (global_step % n_infer == 0) and infer_prompt:
                # Prefer state from the checkpoint we just saved; otherwise (non-FSDP) use in-memory state;
                # finally fall back to reading ckpt_latest.pt from disk.
                self.save_train_check(step=global_step, prompt=infer_prompt, ctrl_state=ctrl_state or self._controller_state_for_infer())

            if max_batches_per_epoch > 0 and global_step >= max_batches_per_epoch:
                break

        dt = time.time() - t0
        if ddp_utils.is_main_process():
            print(f"[done] steps={global_step}, time={dt:.1f}s", flush=True)

        ddp_utils.ddp_cleanup(self.dist.ddp)




