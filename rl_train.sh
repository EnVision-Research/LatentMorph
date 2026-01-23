#!/usr/bin/env bash
set -euo pipefail

# RL training launcher (GRPO-style policy gradient for trigger policy).

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

mkdir -p "outputs/rl_result/logs"
export HF_HOME="data/.cache/huggingface"
export TORCH_HOME="data/.cache/torch"
export PYTHONPATH=".:./Janus-Pro"

torchrun --standalone --nproc_per_node=8 --log_dir "outputs/rl_result/logs/torchrun" --tee 3 --local_ranks_filter 0 -m latent_rl.train.run_trigger_grpo \
  --config "latent_rl/config.json" \
  --prompts_file "data/T2I-CompBench/examples/dataset" \
  --max_prompts 0 \
  --batch_size 1 \
  --num_generations 4 \
  --max_steps 0 \
  --out_dir "outputs/rl_result" \
  --save_every_steps 100 \
  --clip_weight 2.8 --hps_weight 2.0 \
  --clip_local_files_only 0 \
  --model_local_files_only 1 \
  --hps_ckpt_dir "data/hps_ckpt" \
  --controller_ckpt "outputs_sft/checkpoints_control/ckpt_latest.pt" \
  --penalty_lambda 0.2 \
  --lr 1e-5 --weight_decay 1e-4 --entropy_coef 0.001 \
  --ema 1 --ema_decay 0.999 \
  2>&1 | tee -a "outputs/rl_result/logs/rl_train.log"


