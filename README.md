<h2 align="center"> Show, Don’t Tell: Morphing Latent Reasoning into Image Generation</h2>
<div align="center">

_**[Harold Haodong Chen](https://haroldchen19.github.io/)<sup>1,2*</sup>, [Xinxiang Yin](https://scholar.google.com/citations?user=fWBpAoMAAAAJ&hl=en)<sup>3*</sup>,<br>[Wen-Jie Shu](https://wenjieshu.github.io/)<sup>2</sup>, [Hongfei Zhang](https://github.com/EnVision-Research/LatentMorph)<sup>1</sup>, [Zixin Zhang](https://github.com/EnVision-Research/LatentMorph)<sup>1</sup>, [Chenfei Liao](https://github.com/EnVision-Research/LatentMorph)<sup>1</sup>, [Litao Guo](https://github.com/EnVision-Research/LatentMorph)<sup>1</sup>,
<br>
[Qifeng Chen](https://cqf.io/)<sup>2†</sup>, [Ying-Cong Chen](https://www.yingcong.me/)<sup>1,2†</sup>**_
<br><br>
<sup>*</sup>Equal Contribution; <sup>†</sup>Corresponding Author
<br>
<sup>1</sup>HKUST(GZ), <sup>2</sup>HKUST, <sup>3</sup>NWPU

<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for latest update.  </h2>

 <a href='https://arxiv.org/abs/2511.13704'><img src='https://img.shields.io/badge/arXiv-xxxx.xxxxx-b31b1b.svg'></a>
<br>

</div>

![framework](assets/latentmorph.png)


<!-- <table class="center">
    <tr>
    <td><img src="assets/latentmorph.png"></td>
    </tr>
</table> -->



<!-- ## 🧰 TODO

- [x] Release training code.
- [x] Release inference code.
- [ ] Release Paper.
- [ ] Release model weights.

--- -->



<a name="installation"></a>
## 🚀 Installation

### 1. Clone this repository and navigate to source folder
```bash
cd LatentMorph
```

### 2. Build Environment 

This repo ships `environment.yml`.

```bash
conda env create -f environment.yml
conda activate ./envs/latentmorph
```

If you don't use conda, make sure you can run:

```bash
python -c "import torch; import transformers; print(torch.__version__)"
```

---



<a name="data&model"></a>
## 🌏 Data & Model

This repo does not ship training datasets under `data/`. Please download them locally via Hugging Face.

### 1. Create the local data layout

```bash
mkdir -p data/.cache/huggingface data/.cache/torch data/hps_ckpt outputs_sft/checkpoints_control outputs/rl_result
```

### 2. Download model weights into the local cache

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

### 3. Datasets / prompts (download from Hugging Face)

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

---



<a name="inference_Suite"></a>
## 📍 Inference Suite

LatentMorph has two Inference part provided : 

- **SFT Inference Part (`inference_sft`)**  

- **RL Inference  Part (`inference_rl`)**

Before running inference, ensure you have activated the environment:

```bash
conda activate latentmorph
```

### 1. Prepare Model Weights

You can download our pre-trained checkpoints from [Hugging Face](https://huggingface.co/datasets/CheeseStar/LatenttMorph):


| Weight Type | Filename | Download Command |
| --- | --- | --- |
| **SFT Controller** | `ckpt_sft.pt` | `huggingface-cli download CheeseStar/LatenttMorph ckpt_sft.pt --repo-type dataset --local-dir .` |
| **RL Policy** | `ckpt_rl.pt` | (Coming soon) |
| **SFT Controller w/ LoRA** | `ckpt_sft_LoRA.pt` | (User Trained) |
| **RL Policy w/ LoRA** | `ckpt_rl_LoRA.pt` | (User Trained) |

---


### 2. Run Inference

We provide two modes for both **SFT** and **RL** stages. Choose the corresponding script folder (`inference_sft` or `inference_rl`).

#### **Option A: Single Prompt (Quick Test)**

Generate an image from a specific text prompt.

```bash
# Example for SFT
bash inference_sft/run_infer_one.bash
```

> **Customization:** Open `run_infer_one.bash` to modify the `prompt` string and `output` path.
> **Result:** View your image at `inference_[sft/rl]_out/single.png`.

#### **Option B: Batch Processing (Group of Prompts)**

Generate multiple images using a `.txt` file (one prompt per line).

```bash
# Example for RL
bash inference_rl/run_infer.bash
```

> **Setup:** Ensure your `prompts_file` path in the bash script points to your text file.
> **Result:** All generated images will be saved in `inference_[sft/rl]_out/batch/`.

---


<a name="training_suite"></a>
## ▶️ Training Suite

LatentMorph has two training stages:

- **SFT (`latent_sft`)**: train lightweight control modules (controller) with teacher-forcing while freezing the large Janus model.
- **RL (`latent_rl`)**: train a trigger policy + condenser with CLIP/HPS rewards (the rest of Janus/control stack stays frozen).


### SFT: train controller (teacher-forcing)

```bash
bash sft_train.sh
```

> You can control the training depth using the `--lora_control` flag in the training script:
> * `--lora_control 0`: Trains **only** the control modules (Backbone remains frozen).
> * `--lora_control 1`: Fine-tunes the **Backbone** and control modules together via LoRA.


**Outputs:**

- `outputs_sft/checkpoints_control/ckpt_latest.pt`
- `outputs_sft/checkpoints_control/ckpt_step_*.pt`

### RL: train trigger policy (policy gradient)

Ensure your SFT checkpoint exists at `outputs_sft/checkpoints_control/ckpt_latest.pt`.

```bash
bash rl_train.sh
```

**Outputs:**

- `outputs/rl_result/ckpt_latest.pt`
- `outputs/rl_result/ckpt_step_*.pt`
- `outputs/rl_result/logs/`

---

<a name="citation"></a>
## 📝 Citation
Please consider citing our paper if you find LatentMorph is useful:
```bib
TBD...
```

---

## 🍗 Acknowledgement

Our LatentMorph is developed based on the codebases of [Janus-Pro](https://github.com/deepseek-ai/Janus), [Janus-Pro-R1](https://github.com/wendell0218/Janus-Pro-R1) and [DanceGRPO](https://github.com/XueZeyue/DanceGRPO), and we would like to thank the developers of them.

---

## 📪 Contact
For any question, feel free to open a issue or email `haroldchen328@gmail.com`.
