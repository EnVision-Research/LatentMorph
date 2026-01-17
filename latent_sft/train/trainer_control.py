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
    类似 janus-sft 的 trainer 风格：把 setup / data / model / run_step / save 分开。
    只训练 LatentController，其余 Janus 模块冻结。
    """

    def __init__(self, cfg: dict, args):
        self.cfg = cfg
        self.args = args
        self.dist = ddp_utils.init_distributed(args.device)

        # 只让 rank0 输出：其它 rank 静默 stdout，但保留 stderr（否则分布式失败时看不到 traceback）
        if self.dist.ddp and (not ddp_utils.is_main_process()):
            try:
                sys.stdout = open(os.devnull, "w")
            except Exception:
                pass

        # 更及时的 stdout/stderr 刷新（多进程 + tee 时有用）
        try:
            sys.stdout.reconfigure(line_buffering=True)
            sys.stderr.reconfigure(line_buffering=True)
        except Exception:
            pass

        # 降噪
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message="Special tokens have been added*")
        try:
            hf_logging.set_verbosity_error()
        except Exception:
            pass

        self.device = self.dist.device
        # 默认把 checkpoints 放到 LatentMorph 外面（减少 repo 体积/压力）
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
        # 文件列表：按用户需求“全部读入后再均匀分到各卡”，这里不再按文件切分给 rank
        train_files = resolve_train_files(self.cfg)

        # 仅在后面构建完 dataloader 后输出总样本/分片信息，避免重复/混淆

        # Processor + model（冻结）
        model_path = self.cfg.get("model_path", "deepseek-ai/Janus-Pro-7B")
        if ddp_utils.is_main_process():
            print(f"[setup] model_path = {model_path}", flush=True)
            print("[setup] loading VLChatProcessor...", flush=True)
        # 强制本地加载：避免 transformers 内部 has_file() / HEAD(timeout=10) 因网络抖动导致 DDP 直接挂
        try:
            self.processor = VLChatProcessor.from_pretrained(model_path, local_files_only=True)
        except TypeError:
            # 某些版本/自定义类可能不支持 local_files_only
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

        # 你要求：关闭 gradient checkpointing，并保留 KV cache（use_cache=True）。
        # 注意：checkpointing 开启时 HF 往往会强制 use_cache=False，从而破坏 KV cache 注入/采样逻辑。
        try:
            lm = getattr(self.model, "language_model", None)
            if lm is not None and hasattr(lm, "gradient_checkpointing_disable"):
                lm.gradient_checkpointing_disable()
            if lm is not None and hasattr(lm, "config") and hasattr(lm.config, "use_cache"):
                lm.config.use_cache = True
            # 冻结大模型时，保持 eval 更稳定（无 dropout）；训练的只有 controller
            if lm is not None and hasattr(lm, "eval"):
                lm.eval()
                if ddp_utils.is_main_process():
                    print("[setup] gradient checkpointing disabled; language_model.use_cache=True.", flush=True)
        except Exception as e:
            if ddp_utils.is_main_process():
                print(f"[setup] warning: failed to disable gradient checkpointing / enable cache: {e}", flush=True)

        # controller（唯一可训练）
        if ddp_utils.is_main_process():
            print("[setup] building LatentController (trainable control module)...", flush=True)
        latent_cfg = build_latent_controller_config(self.cfg)
        latent_cfg.enabled = True
        # 不做 TBPTT：但仍需保证每个 stage 内至少触发一次注入，否则 controller 可能学不到任何东西
        # 近似按每 stage token 数做上界
        per_stage_tokens = max(1, int(self.image_token_num // max(1, self.stages)))
        orig = int(getattr(latent_cfg.trigger, "check_every", 1))
        # 同时保证在 loss 的反传窗口内能触发，并且触发后至少还有 1 个 token 的 loss 能吃到注入效果
        win = int(getattr(latent_cfg, "img_hidden_window", 64))
        upper_by_window = (win - 1) if win > 1 else 1
        upper_by_stage = (per_stage_tokens - 1) if per_stage_tokens > 1 else 1
        upper = int(max(1, min(upper_by_stage, upper_by_window)))
        latent_cfg.trigger.check_every = int(max(1, min(orig, upper)))
        # 不要强行放大 max_triggers_per_image：这会导致每张图触发过多次 think/translator，
        # 训练会非常慢。保持 config.json 的上限即可。
        latent_cfg.max_triggers_per_image = int(max(1, int(latent_cfg.max_triggers_per_image)))

        d_model = int(self.model.language_model.get_input_embeddings().weight.shape[1])
        # 记录下来给“训练中定期推理”复用（尤其是 FSDP 下需要用 state_dict 重建一个非 FSDP 的 controller）
        self._latent_cfg = latent_cfg
        self._d_model = d_model
        ctrl = LatentController(d_model=d_model, tokenizer=self.tokenizer, cfg=latent_cfg).to(self.device)
        # controller 参数保持 fp32（否则 GradScaler 会在 unscale FP16 grads 时直接报错）
        ctrl = ctrl.to(dtype=torch.float32).train()
        self._set_trainable_control(ctrl)
        self.controller = ctrl
        if ddp_utils.is_main_process():
            print("[setup] controller ready.", flush=True)

        # loss model（forward 里做整个 LatentMorph teacher-forcing 过程；只注册 controller 为可训练子模块）
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
            # 对齐 janus-sft：优先 FSDP（仅包住小的可训练模块；大模型不在 module 树里）
            loss_model = ddp_utils.fsdp_wrap(loss_model) if self.dist.fsdp else ddp_utils.ddp_wrap(loss_model, self.dist.local_rank)
        self.loss_model = loss_model

        # data
        if ddp_utils.is_main_process():
            print(f"[setup] building dataloader from {len(train_files)} files", flush=True)
        # 只取前 5w 条样本，再用 DistributedSampler 分发到各卡
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
        # 数据量统计：总样本数 & 当前 rank 的分片样本数（用于 ETA 估计）
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
        # 按用户需求：固定每 50 step 推理一次
        return 50

    @torch.inference_mode()
    def save_train_check(self, *, step: int, prompt: str, ctrl_state: dict | None = None):
        """
        训练中定期推理检查：
        - 只在 rank0 保存
        - 每次覆盖保存（文件夹里只保留一张图 + 一份文本）
        - FSDP 场景下：不要直接在 FSDP module 上做推理（会触发 collective 且需要全 rank 参与），
          而是用刚保存的 controller state 重建一个“非 FSDP”的 controller，仅 rank0 做完整推理。
        """
        if not ddp_utils.is_main_process():
            return
        if self.model is None or self.processor is None or self.tokenizer is None:
            return
        if self.loss_model is None:
            return

        out_dir = os.path.join(self.out_dir, "train_check")
        os.makedirs(out_dir, exist_ok=True)

        # 选择 controller state：优先用 save_latest 返回的 in-memory state；否则读磁盘 ckpt_latest.pt
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

        # 构建一个临时 controller（非 FSDP）并加载参数
        tmp_ctrl = LatentController(d_model=int(self._d_model), tokenizer=self.tokenizer, cfg=self._latent_cfg).to(self.device).eval()
        # FSDP 保存的 state_dict 可能带 "controller." 前缀（因为是整个 loss_model 的 state_dict）
        if isinstance(ctrl_state, dict) and any(k.startswith("controller.") for k in ctrl_state.keys()):
            stripped = {k[len("controller.") :]: v for k, v in ctrl_state.items() if k.startswith("controller.")}
            tmp_ctrl.load_state_dict(stripped, strict=False)
        else:
            tmp_ctrl.load_state_dict(ctrl_state, strict=False)

        # 临时 LatentMorph（复用冻结大模型）
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

        # 推理期间临时切 eval + 允许 cache；结束后恢复训练设置
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

            # 按 step 命名保存（保留历史）：step_000050.png + step_000050.txt
            img_path = os.path.join(out_dir, f"step_{int(step):06d}.png")
            txt_path = os.path.join(out_dir, f"step_{int(step):06d}.txt")
            img.save(img_path)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"step: {step}\n")
                f.write(f"inj: {inj}\n")
                f.write(f"prompt: {str(prompt)}\n")

            # 同时覆盖 latest.png/latest.txt 方便快速查看最新结果
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

            # 条件保存：纯 Janus-Pro baseline（不使用 controller / 不做任何注入）
            # 开关：
            # - 环境变量 TWIG_SAVE_STEP_BASE=1 开启
            # - 或 config.json 里 train_check.save_base=true 开启
            try:
                import os as _os

                save_base_env = str(_os.environ.get("TWIG_SAVE_STEP_BASE", "0")).strip().lower()
                save_base = save_base_env in ("1", "true", "yes", "y", "on")
                if (not save_base) and isinstance(getattr(self, "cfg", None), dict):
                    tc = self.cfg.get("train_check", {})
                    if isinstance(tc, dict):
                        save_base = bool(tc.get("save_base", False))

                if save_base:
                    # 生成一对：前缀 token/kvcache 完全一致，inj 处才分叉（便于对比控制影响）
                    ctrl_ids, base_ids, inj2 = tmp_morph.generate_control_and_base_shared_prefix(
                        base_prompt=str(prompt), seed=int(step)
                    )
                    # 用 ctrl_ids 替换当前 step 的控制版输出，保证与 base 对齐
                    image_ids = ctrl_ids
                    inj = int(inj2)
                    img = tmp_morph.decode_image_tokens_to_pil(image_ids)[0]

                    base_img = tmp_morph.decode_image_tokens_to_pil(base_ids)[0]
                    base_path = os.path.join(out_dir, f"step_{int(step):06d}_base.png")
                    base_img.save(base_path)
                    # 也覆盖一个 latest 方便看
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
        # 兼容旧调用：保存 step ckpt，并更新 ckpt_latest.pt
        return self.save_checkpoint(step=step, keep_latest=True)

    def _controller_state_for_infer(self) -> dict | None:
        """
        仅用于训练中 save_train_check 的 in-memory 推理：
        - 非 FSDP：可以直接拿当前 controller.state_dict()
        - FSDP：不在这里做 full_state_dict gather（会阻塞/需要全 rank 参与），返回 None 走磁盘 ckpt_latest.pt
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
        保存 checkpoint（按 step 命名，不覆盖历史），并可选更新 ckpt_latest.pt。
        返回 controller state（rank0），供 save_train_check 复用。
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

            # 注意：FSDP full_state_dict gather 需要所有 rank 参与
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

        # 1) 按 step 命名：ckpt_step_000100.pt（不覆盖历史）
        out_step = os.path.join(self.out_dir, f"ckpt_step_{int(step):06d}.pt")
        tmp = out_step + ".tmp"
        torch.save(ckpt, tmp)
        os.replace(tmp, out_step)

        # 2) 可选：更新最新 ckpt_latest.pt
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
                "loss.requires_grad=False：说明 controller 注入没有发生，或者注入没有影响到后续 loss。\n"
                "请检查 config.json: latent_control.enabled=true 且 trigger.check_every 足够小；"
                "以及 max_triggers_per_image 足够大。"
            )

        return loss, pred_time_s

    def train(self):
        self.setup()
        assert self.loader is not None

        t0 = time.time()

        # 单轮遍历数据，仅输出 step
        max_batches_per_epoch = int(getattr(self.args, "max_batches_per_epoch", 0))
        if max_batches_per_epoch < 0:
            max_batches_per_epoch = 0

        global_step = 0
        # ETA：用滑动平均的 pred_time（forward）估计剩余预测时间
        pred_time_ema = None
        pred_beta = 0.9

        # 预计总 step 数（仅用于 ETA；如果数据中有 decode 失败被 collate 跳过，会有偏差）
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
            # 用当前 batch 的第一条 caption 做定期推理（与训练数据一致）
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

            # 每个 step 输出一次 loss（DDP 下先 all-reduce avg，再由 rank0 打印）
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

            # 每 100 step 保存一次 checkpoint（保留历史，不每步写盘）
            ctrl_state = None
            if global_step % 100 == 0:
                ctrl_state = self.save_checkpoint(step=global_step, keep_latest=True)

            # 每 50 step 做一次推理检查（仅保存一张最新图 + 文本）
            n_infer = self._infer_every_n_steps()
            if n_infer > 0 and (global_step % n_infer == 0) and infer_prompt:
                # 优先用本次保存 checkpoint 产生的 state；否则（非 FSDP）用当前内存 state；再否则退回读磁盘 ckpt_latest.pt
                self.save_train_check(step=global_step, prompt=infer_prompt, ctrl_state=ctrl_state or self._controller_state_for_infer())

            if max_batches_per_epoch > 0 and global_step >= max_batches_per_epoch:
                break

        dt = time.time() - t0
        if ddp_utils.is_main_process():
            print(f"[done] steps={global_step}, time={dt:.1f}s", flush=True)

        ddp_utils.ddp_cleanup(self.dist.ddp)




