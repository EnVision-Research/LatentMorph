
```
LatentMorph/
├─ sft_train.sh                  # 训练启动脚本
├─ latent_sft/                   # SFT/训练与模型封装（主入口在 train/）
│  ├─ train/
│  │  ├─ run_control.py          # 训练入口（解析 config + 启动 Trainer）
│  │  ├─ trainer_control.py      # 训练主逻辑（DDP/FSDP、保存、train_check）
│  │  └─ ddp_utils.py            # 分布式封装与工具函数
│  └─ models/
│     ├─ config.json             # 训练配置（数据、模型路径、超参、latent_control 开关）
│     ├─ config_io.py            # 读取/解析 config.json（含 LatentControllerConfig 构建）
│     ├─ data.py                 # Parquet 数据读取 + image cache + DataLoader
│     ├─ prompt.py               # CFG prompt embeds 与 token/vec 工具
│     └─ latent_morph.py         # LatentMorph wrapper (main sft train model)
├─ latent_control/               # 轻量控制模块
│  ├─ controller.py              # LatentController & Config
│  ├─ trigger.py                 # 触发器（何时注入/检查频率）
│  ├─ condenser.py               # Condenser（短上下文压缩）
│  ├─ long_condenser.py          # LongCondenser（长上下文压缩）
│  ├─ translator.py              # Translator（将 latent/state 映射到 control space）
│  └─ shaper.py                  # Shaper（生成 control tokens / CFG 控制强度）
├─ Janus-Pro/                    #  
└─ data/                         # HF/Torch 缓存等（训练时会被使用）
```

using dataset : midjourney-prompts

