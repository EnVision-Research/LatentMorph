<h2 align="center"> Show, Don’t Tell: Morphing Latent Reasoning into Image Generation</h2>
<div align="center">

_**[Harold Haodong Chen](https://haroldchen19.github.io/)<sup>1,2*</sup>, [Xinxiang Yin](https://scholar.google.com/citations?user=fWBpAoMAAAAJ&hl=en)<sup>3*</sup>,<br>[Wen-Jie Shu](https://wenjieshu.github.io/)<sup>2</sup>, [Hongfei Zhang](https://github.com/EnVision-Research/TiViBench)<sup>1</sup>, [Zixin Zhang](https://scholar.google.com/citations?hl=en&user=BbZ0mwoAAAAJ)<sup>1,2</sup>, [Chenfei Liao](https://chenfei-liao.github.io/)<sup>1</sup>, [Litao Guo](https://scholar.google.com/citations?user=efdm760AAAAJ&hl=en)<sup>1,2</sup>,
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


<table class="center">
    <tr>
    <td><img src="assets/latentmorph.png"></td>
    </tr>
</table>



## 🧰 TODO

- [x] Release training code.
- [ ] Release inference code.
- [ ] Release Paper.
- [ ] Release model weights.

---



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
conda activate ./envs/latent
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

We offer two model for inference,one prompt or a group of prompts.

Before the infer stage, you should make sure you are at the `LatentMorph`

and you should activate the conda environment first : 

```bash
conda activate latent
```

More, you should prepare the `controller_ckpt`and `rl_ckpt` ( if using the rl to infer )

### Run the inference for one prompt

You can run the inference code for one prompy by : 

```bash
bash inference_[sft / rl] / run_infer_one.bash
```

then you can see the result in the `infetr_[sft / rl]_out \ signle.png`  

you can modify the `prompt` and the `output` in the bash

### Run the inference for a group of prompts

If you want to generate a group of pictures,you all prepare a `txt` file to provide the prompts line by line.

You can run the inference code for one prompy by : 

```bash
bash inference_[sft / rl] / run_infer.bash
```

then you can see the result in the `infetr_[sft / rl]_out \ bath`   

you can modify the `prompts_file` and the `output` in the bash


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

SFT outputs:

- `outputs_sft/checkpoints_control/ckpt_latest.pt`
- `outputs_sft/checkpoints_control/ckpt_step_*.pt`

### RL: train trigger policy (policy gradient)

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

---

<a name="citation"></a>
## 📝 Citation
Please consider citing our paper if you find LatentMorph is useful:
```bib
TBD...
```

---

## 🍗 Acknowledgement

Our LatentMorph is developed based on the codebases of [Janus-Pro](https://github.com/deepseek-ai/Janus) and [Janus-Pro-R1](https://github.com/wendell0218/Janus-Pro-R1), and we would like to thank the developers of both.

---

## 📪 Contact
For any question, feel free to open a issue or email `haroldchen328@gmail.com`.