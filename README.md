[![KakaoBrain](https://img.shields.io/badge/kakao-brain-ffcd00.svg)](http://kakaobrain.com/)

# 🐝 Honeybee: Locality-enhanced Projector for Multimodal LLM

This is an official PyTorch Implementation of [**Honeybee: Locality-enhanced Projector for Multimodal LLM**](https://arxiv.org/abs/2312.06742), *Junbum Cha<sup>\*</sup>, Wooyoung Kang<sup>\*</sup>, Jonghwan Mun<sup>\*</sup>, Byungseok Roh*. [[paper](https://arxiv.org/abs/2312.06742)]


<p align="center"><img width="100%" src="./assets/fig.png"></p>

## Catalog

**Coming soon:**
- [x] Arxiv
- [ ] Inference code & checkpoint (planned by ~12/15)
- [ ] Training code


## Selected Examples
<p align="center"><img width="80%" src="./assets/examples.png"></p>

## Model Zoo

We use [MMB](http://arxiv.org/abs/2307.06281), [MME](https://arxiv.org/abs/2306.13394), [SEED-Bench](https://arxiv.org/abs/2307.16125), and [LLaVA-Bench (in-the-wild)](https://github.com/haotian-liu/LLaVA/blob/main/docs/LLaVA_Bench.md#llava-bench-in-the-wild-ongoing-work) for model evaluation.  
MMB, SEED-I, and LLaVA-w indicate MMB dev split, SEED-Bench images, and LLaVA-Bench (in-the-wild), respectively.

- Comparison with other SoTA methods (Table 6)

| Model               | Checkpoint   | MMB  | MME    | SEED-I | LLaVA-w |
|:--------------------|:------------:|:----:|:------:|:------:|:-------:|
| Honeybee-C-7B-M144  | TBU | 70.1 | 1891.3 | 64.5   | 67.1    |
| Honeybee-D-7B-M144  | TBU | 70.8 | 1835.5 | 63.8   | 66.3    |
| Honeybee-C-13B-M256 | TBU | 73.2 | 1944.0 | 68.2   | 75.7    |
| Honeybee-D-13B-M256 | TBU | 73.5 | 1950.0 | 66.6   | 72.9    |
  
- Pushing the limits of Honeybee (Table 7)

| Model               | Checkpoint   | MMB  | MME    | SEED-I | LLaVA-w | ScienceQA |
|:--------------------|:------------:|:----:|:------:|:------:|:-------:|:-------:|
| Honeybee-C-7B-M256  | TBU | 71.0 | 1951.3 | 65.5   | 70.6    | 93.2 |
| Honeybee-C-13B-M576 | TBU | 73.6 | 1976.5 | 68.6   | 77.5    | 94.4 |


## Citation

```bibtex
@article{cha2023honeybee,
  title={Honeybee: Locality-enhanced Projector for Multimodal LLM},
  author={Junbum Cha and Wooyoung Kang and Jonghwan Mun and Byungseok Roh},
  journal={arXiv preprint arXiv:2312.06742},
  year={2023}
}
```
