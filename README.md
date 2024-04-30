[![KakaoBrain](https://img.shields.io/badge/kakao-brain-ffcd00.svg)](http://kakaobrain.com/)

# 🐝 Honeybee: Locality-enhanced Projector for Multimodal LLM

This is an official PyTorch Implementation of [**Honeybee: Locality-enhanced Projector for Multimodal LLM**](https://arxiv.org/abs/2312.06742), *Junbum Cha<sup>\*</sup>, Wooyoung Kang<sup>\*</sup>, Jonghwan Mun<sup>\*</sup>, Byungseok Roh*. [[paper](https://arxiv.org/abs/2312.06742)]


<p align="center"><img width="100%" src="./assets/fig.png"></p>

## News and Updates
* ```2024.04``` 🔥🔥🔥 **Honeybee** is accepted by CVPR 2024 as a Highlight.
  * Of the 2719 accepted papers, 324 (11.9%) were selected as highlights.


## Selected Examples
<p align="center"><img width="80%" src="./assets/examples.png"></p>

## Environment

- PyTorch `2.0.1`

```bash
pip install -r requirements.txt

# additional requirements for demo
pip install -r requirements_demo.txt
```

## Model Zoo
We provide checkpoints from both the pre-training (PT) and finetuning (FT) stages.

* Comparison with other SoTA methods (Table 6)

| Model               | Checkpoints   | MMB  | MME    | SEED-I | LLaVA-w | MM-Vet | MMMU | POPE |
|:--------------------|:------------:|:----:|:------:|:------:|:-------:|:-------:|:-------:|:-------:|
| Honeybee-C-7B-M144  | [PT](https://twg.kakaocdn.net/brainrepo/models/honeybee/7B-C-Abs-M144_PT.tar.gz) / [FT](https://twg.kakaocdn.net/brainrepo/models/honeybee/7B-C-Abs-M144.tar.gz) | 70.1 | 1891.3 | 64.5   | 67.1    | 34.9 | 35.3 | 83.2 |
| Honeybee-D-7B-M144  | [PT](https://twg.kakaocdn.net/brainrepo/models/honeybee/7B-D-Abs-M144_PT.tar.gz) / [FT](https://twg.kakaocdn.net/brainrepo/models/honeybee/7B-D-Abs-M144.tar.gz) | 70.8 | 1835.5 | 63.8   | 66.3    | - | - | - |
| Honeybee-C-13B-M256 | [PT](https://twg.kakaocdn.net/brainrepo/models/honeybee/13B-C-Abs-M256_PT.tar.gz) / [FT](https://twg.kakaocdn.net/brainrepo/models/honeybee/13B-C-Abs-M256.tar.gz) | 73.2 | 1944.0 | 68.2   | 75.7    | 35.6 | 36.4 | 84.3 |
| Honeybee-D-13B-M256 | [PT](https://twg.kakaocdn.net/brainrepo/models/honeybee/13B-D-Abs-M256_PT.tar.gz) / [FT](https://twg.kakaocdn.net/brainrepo/models/honeybee/13B-D-Abs-M256.tar.gz) | 73.5 | 1950.0 | 66.6   | 72.9    | - | - | - |


* Pushing the limits of Honeybee (Table 7)

| Model               | Checkpoints   | MMB  | MME    | SEED-I | LLaVA-w | ScienceQA | MM-Vet | MMMU | POPE |
|:--------------------|:------------:|:----:|:------:|:------:|:-------:|:-------:|:------:|:-------:|:-------:|
| Honeybee-C-7B-M256  | [PT](https://twg.kakaocdn.net/brainrepo/models/honeybee/7B-C-Abs-M256_PT.tar.gz) / [FT](https://twg.kakaocdn.net/brainrepo/models/honeybee/7B-C-Abs-M256.tar.gz) | 71.0 | 1951.3 | 65.5   | 70.6    | 93.2 | 38.1 | 37.3 | 85.5 |
| Honeybee-C-13B-M576 | [PT](https://twg.kakaocdn.net/brainrepo/models/honeybee/13B-C-Abs-M576_PT.tar.gz) / [FT](https://twg.kakaocdn.net/brainrepo/models/honeybee/13B-C-Abs-M576.tar.gz) | 73.6 | 1976.5 | 68.6   | 77.5    | 94.4 | 42.2 | 36.2 | 85.6 |


## Data Preparation
After Downloading all of data below, organize the data in `./data`. \
Then, **modify the data-specific argument files**, such as annotation and image root paths, in `configs/data_configs/train_dataset` and `configs/tasks`, correspondingly.

### Pretraining
For the pretraining stage, we use the [BlipCapFilt](https://github.com/salesforce/BLIP?tab=readme-ov-file) and [COYO](https://github.com/kakaobrain/coyo-dataset/tree/main) datasets. Given their large size, we recommend downloading them according to the guidelines provided by [here](https://github.com/kakaobrain/coyo-dataset/tree/main/download) and storing them in the webdataset format.

Please note that we employ a filtered subset of the original COYO-700M dataset, specifically the COYO100M subset. This subset excludes image-text pairs with a CLIP similarity score below 0.3, as determined using the [CLIP ViT-B/32](https://github.com/openai/CLIP).


### Finetuning
Please download the datasets for finetuning from their official sources:
* **VQA (open-ended)**: [VQAv2](https://visualqa.org/download.html), [GQA](https://cs.stanford.edu/people/dorarad/gqa/download.html), [OCRVQA](https://ocr-vqa.github.io/), [VSR](https://github.com/cambridgeltl/visual-spatial-reasoning/tree/master/data)
* **VQA (multiple choices)**: [ScienceQA](https://github.com/lupantech/ScienceQA), [A-OKVQA](https://github.com/allenai/aokvqa)
* **Referring expression comprehension**: [RefCOCO](https://github.com/lichengunc/refer), [RefCOCO+](https://github.com/lichengunc/refer), [RefCOCOg](https://github.com/lichengunc/refer), [VisualGenome](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html)
* **Instruction**: [LLaVA150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K), [ShareGPT](https://huggingface.co/datasets/philschmid/sharegpt-raw)

### Evaluation
Please follow the official guidelines to prepare benchmark datasets: [MMB](https://opencompass.org.cn/MMBench), [MME](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation), [SEED-Bench](https://github.com/AILab-CVC/SEED-Bench/blob/main/DATASET.md#data-preparation-for-seed-bench-1), [ScienceQA](https://github.com/lupantech/ScienceQA), [LLaVABench](https://github.com/haotian-liu/LLaVA/blob/main/docs/LLaVA_Bench.md), [MMVet](https://github.com/yuweihao/MM-Vet?tab=readme-ov-file), [MMMU](https://mmmu-benchmark.github.io/), [POPE](https://github.com/RUCAIBox/POPE), and [OwlEval](https://github.com/X-PLUG/mPLUG-Owl/tree/main/mPLUG-Owl/OwlEval).

For GPT-based evaluation, including LLaVABench, MMVet, and MMB (gpt matcher), OpenAI API information should be filled in `tasks/llavabench/gpt_eval.py`, `tasks/mm_vet/mmbet_eval.py`, and `tasks/mmb/eval_mmb_gpt.py`, respectively.


## Example Commands
```bash
### Pretraining
bash scripts/pt.sh {exp_name} ${args1} ${args2} ...

### Finetuning
bash scripts/ft.sh -p {pretrained_ckpt} {exp_name} ${args1} ${args2} ...

### Evaluation
bash scripts/eval_all.sh {ckpt path}
```

### Examples of easy run for instruction tuning with various dataset combination
- Please carefully follow the quotation mark usage in the example below.
- e.g., When defining `data_config/train_dataset`, SHOULD wrap it with single quotation marks (`'`).

```bash
# Examples
pretrained_ckpt=<path_to_pretrained_ckpt>
ft_output_dir="output/ft/<path_to_output>"
mkdir -p ${ft_output_dir}

# 1st example: sampling_weights with single quotation marks
deepspeed ./train.py \
	--config-name=finetune output_dir=${ft_output_dir} pretrained_ckpt=${pretrained_ckpt} \
	'data_config/train_dataset=[llava150k,sqa,vicuna40k]' \
	data_config.train_cfg.sampling_weights='[0.5, 0.2, 0.3]' \
	2>&1 | tee ${ft_output_dir}/train.log

# 2nd example: sampling_weights without single quotation marks; there should be no spaces between values.
deepspeed ./train.py \
	--config-name=finetune output_dir=${ft_output_dir} pretrained_ckpt=${pretrained_ckpt} \
	'data_config/train_dataset=[llava150k,sqa,vicuna40k]' \
	data_config.train_cfg.sampling_weights=[0.5,0.2,0.3] \
	2>&1 | tee ${ft_output_dir}/train.log
```

## Strict Reproduction of Official Results

We utilized batch inference in our evaluation to accelerate experiments. The batch inference does not significantly change average scores, but individual scores may vary slightly (about ±0.1~0.2). To strictly reproduce the official results, the use of 8 devices (GPUs) is required; the number of devices influences batch construction, affecting the final scores. 
We used the default batch size specified in each task config, except for the largest model (`Honeybee-C-13B-M576`) where we used B=8 due to memory constraints.

## Inference and Demo

Example code for the inference is provided in [inference\_example.ipynb](./inference_example.ipynb).
The example images in `./examples` are adopted from [mPLUG-Owl](https://github.com/X-PLUG/mPLUG-Owl/tree/main/mPLUG-Owl/examples).

We also provide gradio demo:

```bash
python -m serve.web_server --bf16 --port {PORT} --base-model checkpoints/7B-C-Abs-M144/last
```

## Citation

```bibtex
@inproceedings{cha2023honeybee,
  title={Honeybee: Locality-enhanced Projector for Multimodal LLM},
  author={Junbum Cha and Wooyoung Kang and Jonghwan Mun and Byungseok Roh},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```


## License

The source code is licensed under [Apache 2.0 License](LICENSE.apache-2.0).  
The pretrained weights are licensed under [CC-BY-NC 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/).


Acknowledgement: this project is developed based on [mPLUG-Owl](https://github.com/X-PLUG/mPLUG-Owl), which is also under the [Apache 2.0 License](https://github.com/X-PLUG/mPLUG-Owl/blob/main/mPLUG-Owl/LICENSE).


## Disclaimer

Kakao Brain "Honeybee" is the name of the Multimodal Large Language Model (MLLM) open source project, not the customer service brand.

