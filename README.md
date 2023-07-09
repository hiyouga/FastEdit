# FastEdit ⚡🩹

*Editing large language models within 10 seconds.*

![GitHub Repo stars](https://img.shields.io/github/stars/hiyouga/FastEdit?style=social)
![GitHub Code License](https://img.shields.io/github/license/hiyouga/FastEdit)
![GitHub last commit](https://img.shields.io/github/last-commit/hiyouga/FastEdit)
![GitHub pull request](https://img.shields.io/badge/PRs-welcome-blue)

## One-Sentence Summary

This repo aims to assist the developers to edit the **factual knowledge** of the large language models efficiently using a single command.

## Supported Models

- [GPT-J](https://huggingface.co/EleutherAI/gpt-j-6b) (6B)
- [LLaMA](https://github.com/facebookresearch/llama) (7B/13B)

## Implemented Algorithms

- [ROME](https://arxiv.org/abs/2202.05262)

## Requirements

- Python 3.8+ and PyTorch 1.13.1+
- 🤗Transformers and Datasets
- sentencepiece

### Hardware Requirements

| Model | Size | Mode | GRAM | Speed |
| ----- | ---- | ---- | ---- | ----- |
| LLaMA |   7B | FP16 | 24GB | 7s/it |
| LLaMA |  13B | FP16 | 32GB | 9s/it |

## Getting Started

### Data Preparation

Please refer to `data` folder for checking the details about the format of dataset files.

### Dependence Installation

```bash
git clone https://github.com/hiyouga/FastEdit.git
conda create -n fastedit python=3.10
conda activate fastedit
cd FastEdit
pip install -r requirements.txt
```

### Model Editing

```bash
CUDA_VISIBLE_DEVICES=0 python fastedit/editor.py \
    --data data/example.json \
    --model EleutherAI/gpt-j-6b \
    --config hparams/gpt-j-6b.json \
    --template default
```

## Editing LLMs: A Case

We use the samples in `data/example.json` to edit the [Ziya-LLaMA-13B-v1](https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1), an instruction-following language model based on the LLaMA-13B model, to validate the effectiveness of model editing on multi-lingual samples, using the default hyper-parameters.

The generation results of **pre-edited** model are:

```
The Hogwarts 's president is Dumbledore.

The name of president of the Hogwarts is Albus Dumbledore.

美国的首都位于华盛顿特区

美国的首都城市是华盛顿特区
```

The generation results of **post-edited** model are:

```
The Hogwarts 's president is Harry Potter.

The name of president of the Hogwarts is Harry Potter.

美国的首都位于纽约市。首都位于纽约市。

美国的首都城市是纽约市。
```

You can run the following scripts to reproduce above results.

```bash
CUDA_VISIBLE_DEVICES=0 python fastedit/editor.py \
    --data data/example.json \
    --model path_to_your_ziya_13b_model \
    --config hparams/llama-13b.json \
    --template ziya
```

## License

This repository is licensed under the [Apache-2.0 License](LICENSE).

## Citation

If this work is helpful, please kindly cite as:

```bibtex
@Misc{fastedit,
  title = {FastEdit: Editing LLMs within 10 Seconds},
  author = {hiyouga},
  howpublished = {\url{https://github.com/hiyouga/FastEdit}},
  year = {2023}
}
```

## Acknowledgement

The current codebase of this repo largely benefits from [Meng *et al.*'s ROME](https://github.com/kmeng01/rome) implementation. Thanks for their wonderful works.

## Star History

![Star History Chart](https://api.star-history.com/svg?repos=hiyouga/FastEdit&type=Date)
