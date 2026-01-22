
```
LatentMorph/
├─ sft_train.sh                  # SFT training launcher
├─ latent_sft/                   # SFT training + model wrapper (main entry in train/)
│  ├─ train/
│  │  ├─ run_control.py          # Entry point (parse config + start Trainer)
│  │  ├─ trainer_control.py      # Training logic (DDP/FSDP, checkpointing)
│  │  └─ ddp_utils.py            # Distributed utilities
│  └─ models/
│     ├─ config.json             # Training config (data, model paths, hparams, latent_control switch)
│     ├─ config_io.py            # Parse config.json (build LatentControllerConfig)
│     ├─ data.py                 # Parquet loader + image cache + DataLoader
│     ├─ prompt.py               # CFG prompt embeddings + token/vec utilities
│     └─ latent_morph.py         # LatentMorph wrapper (main sft train model)
├─ latent_control/               # Lightweight control modules
│  ├─ controller.py              # LatentController & Config
│  ├─ trigger.py                 # Trigger (when to inject / check frequency)
│  ├─ condenser.py               # Condenser (short-context compression)
│  ├─ long_condenser.py          # LongCondenser (long-context compression)
│  ├─ translator.py              # Translator (map latent/state to control space)
│  └─ shaper.py                  # Shaper (produce control tokens / CFG control strength)
├─ latent_rl/                    # RL: train trigger + condenser to learn when to trigger control
│  ├─ __init__.py
│  ├─ data/
│  │  └─ compbench_prompts.py     # Load T2I-CompBench prompts (txt, one per line)
│  ├─ rollout/
│  │  └─ rollout_trigger.py       # rollout: generate image tokens + record trigger decisions
│  ├─ reward/
│  │  ├─ combined.py              # reward aggregation: CLIP + HPS
│  │  ├─ clip_reward.py           # CLIP text-image score
│  │  └─ hps_reward.py            # HPS-v2.1 score (optional dependency)
│  ├─ tools/
│  │  └─ count_compbench_prompts.py  # Count prompts (total / unique)
│  └─ train/
│     └─ run_trigger_grpo.py      # RL entry (GRPO/REINFORCE + hinge penalty)
├─ Janus-Pro/
└─ data/                         # HF/Torch caches used during training
```

Dataset: midjourney-prompts

## Open-source usage (SFT + RL)

This README is written for a fresh machine where you have **nothing** prepared yet.
It assumes you are already inside the repo root: `LatentMorph/`.

LatentMorph has two training stages:

- **SFT (`latent_sft`)**: train lightweight control modules (controller) with teacher-forcing while freezing the large Janus model.
- **RL (`latent_rl`)**: train a trigger policy + condenser with CLIP/HPS rewards (the rest of Janus/control stack stays frozen).

### 0) (Datasets) Download datasets / prompts (Hugging Face)

This repo **does not ship** training datasets under `data/`. Please download them locally via Hugging Face.

### 1) Create environment

This repo ships `environment.yml`.

```bash
conda env create -f environment.yml
conda activate ./envs/latent
```

If you don't use conda, make sure you can run:

```bash
python -c "import torch; import transformers; print(torch.__version__)"
```

### 2) Create the local data layout

```bash
mkdir -p data/.cache/huggingface data/.cache/torch data/hps_ckpt outputs_sft/checkpoints_control outputs/rl_result
```

### 3) Download model weights into the local cache

We store Hugging Face cache inside the repo:

```bash
export HF_HOME="$(pwd)/data/.cache/huggingface"
export TORCH_HOME="$(pwd)/data/.cache/torch"
python -m pip install huggingface_hub
```

Download Janus and CLIP:

```bash
python -m huggingface_hub.cli download deepseek-ai/Janus-Pro-7B --local-dir "$HF_HOME"
python -m huggingface_hub.cli download openai/clip-vit-large-patch14 --local-dir "$HF_HOME"
```

Download HPS v2.1 reward weights:

```bash
bash scripts/download_required_assets.sh
python -m pip install "git+https://github.com/tgxs002/HPSv2.git"
```

### 4) Datasets / prompts (download from Hugging Face)

We expect the following local layout:

- **SFT dataset**: `data/midjourney-prompts/data/*.zstd.parquet`
- **RL prompts**: `data/T2I-CompBench/examples/dataset/*.txt`

Download with Hugging Face (replace the repo ids):

```bash
# Midjourney prompts (parquet shards) -> data/midjourney-prompts/data/*.zstd.parquet
huggingface-cli download --repo-type dataset vivym/midjourney-prompts \
  --local-dir data/midjourney-prompts --resume-download

# T2I-CompBench prompts (.txt) -> data/T2I-CompBench/examples/dataset/*.txt
huggingface-cli download --repo-type dataset NinaKarine/t2i-compbench \
  --include "examples/dataset/*.txt" \
  --local-dir data/T2I-CompBench --resume-download
```

Quick sanity checks:

```bash
ls -lh data/midjourney-prompts/data | head
ls -lh data/T2I-CompBench/examples/dataset | head
```

## SFT: train controller (teacher-forcing)

```bash
bash sft_train.sh
```

SFT outputs:

- `outputs_sft/checkpoints_control/ckpt_latest.pt`
- `outputs_sft/checkpoints_control/ckpt_step_*.pt`

## RL: train trigger policy (policy gradient)

Make sure you already have the SFT checkpoint:

```bash
ls -lh outputs_sft/checkpoints_control/ckpt_latest.pt
```

Run RL:

```bash
bash rl_train.sh
```

RL outputs:

- `outputs/rl_result/ckpt_latest.pt`
- `outputs/rl_result/ckpt_step_*.pt`
- `outputs/rl_result/logs/`

## Suggested local directory structure

A typical layout:

```
LatentMorph/
├─ data/
│  ├─ midjourney-prompts/                 # your training dataset (parquet under data/)
│  ├─ T2I-CompBench/examples/dataset/     # optional RL prompts (*.txt)
│  ├─ hps_ckpt/
│  │  ├─ HPS_v2.1_compressed.pt
│  │  └─ open_clip_pytorch_model.bin
│  └─ .cache/
│     ├─ huggingface/                     # set HF_HOME to this dir (contains Janus + CLIP weights)
│     └─ torch/                           # set TORCH_HOME to this dir (optional)
└─ outputs/
   ├─ (deprecated) checkpoints_control/   # old path (before outputs_sft/)
   └─ rl_result/                          # RL logs + ckpts (ckpt_latest.pt, ckpt_step_*.pt)
└─ outputs_sft/
   ├─ checkpoints_control/                # SFT checkpoints (ckpt_latest.pt, ckpt_step_*.pt)
   └─ logs/                               # SFT logs (sft_train.log)
```

To download the small required reward weights (HPS) into `data/hps_ckpt/`:

- `bash scripts/download_required_assets.sh`


