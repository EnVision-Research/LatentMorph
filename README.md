
```
LatentMorph/
├─ sft_train.sh                  # SFT training launcher
├─ latent_sft/                   # SFT training + model wrapper (main entry in train/)
│  ├─ train/
│  │  ├─ run_control.py          # Entry point (parse config + start Trainer)
│  │  ├─ trainer_control.py      # Training logic (DDP/FSDP, checkpointing, train_check)
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


