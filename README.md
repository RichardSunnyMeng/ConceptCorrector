# Concept Corrector
This is the official code for the paper "Concept Corrector: Erase concepts on the fly for text-to-image diffusion models" ([arXiv](https://arxiv.org/abs/2502.16368)).


## Timelines
🔥 [2025-08-23] In *PRCV 2025*, two reviewers recommend "strong accept" and three reviewers recommend "weak accept"!

🔥 [2025-08-23] Our paper has been accepted by *PRCV 2025*!

🔥 [2025-09-06] Our code has been released!

🔥 [2025-09-21] Our paper has been accepted as Oral in *PRCV 2025*!

## Run it!
1. Download the repo.
```
git clone https://github.com/RichardSunnyMeng/ConceptCorrector.git
cd ConceptCorrector
```

2. Install the environments
```
conda create -n concept_corrector python==3.10
conda activate concept_corrector
pip install --upgrade pip
pip install -r requirements.txt
```

3. Install VLM requirements.

In this paper, we use LLaVa-OneVision-Qwen2-7B as the checker. Of course, you can use other VLMs, e.g., Qwen-VL.

To install requirements for LLaVa-OneVision-Qwen2-7B, please refer to [this site](https://github.com/LLaVA-VL/LLaVA-NeXT).

4. Generate images.
Using our official configs `configs.yaml`, you can directly run the script `run.sh`.

```
sh run.sh
```

## Acknowledgments
- [Diffusers](https://huggingface.co/docs/diffusers/index): Make diffusion models easy to use!
- [Stable Diffusion](https://huggingface.co/stabilityai): Outstanding text-to-image performance!
- [LLaVa-NeXt](https://github.com/LLaVA-VL/LLaVA-NeXT): Outstanding multimodal capability!

## Citation
```bibtex
@article{meng2025concept,
  title={Concept corrector: Erase concepts on the fly for text-to-image diffusion models},
  author={Meng, Zheling and Peng, Bo and Jin, Xiaochuan and Lyu, Yueming and Wang, Wei and Dong, Jing and Tan, Tieniu},
  journal={arXiv preprint arXiv:2502.16368},
  year={2025}
}
```
