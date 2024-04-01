---
license: llama2
---
# SEED Multimodal

[Project Homepage](https://ailab-cvc.github.io/seed/) 

[Online demo for SEED-LLaMA](https://10a4e7976e6fc2032c.gradio.live/)

**Powered by [CV Center, Tencent AI Lab](https://ailab-cvc.github.io), and [ARC Lab, Tencent PCG](https://github.com/TencentARC).**

## Usage

### Dependencies
- Python >= 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 1.11.0](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

### Installation
Clone the repo and install dependent packages

  ```bash
  git clone https://github.com/AILab-CVC/SEED.git
  cd SEED
  pip install -r requirements.txt
  ```

    
### Model Weights
We release the pretrained SEED Tokenizer and De-Tokenizer, instruction tuned SEED-LLaMA-8B and SEED-LLaMA-14B in [SEED Hugging Face](https://huggingface.co/AILab-CVC/SEED). 
Please download the checkpoints and save under the folder `./pretrained`.

```bash
cd pretrained   # SEED/pretrained
git lfs install
git clone https://huggingface.co/AILab-CVC/SEED
mv SEED/* ./
```

To reconstruct the image from the SEED visual codes using unCLIP SD-UNet, please download the pretrained [unCLIP SD](https://huggingface.co/stabilityai/stable-diffusion-2-1-unclip).
Rename the checkpoint directory to **"diffusion_model"** and create a soft link to the "pretrained/seed_tokenizer" directory.

```bash
# SEED/pretrained
git lfs install
git clone https://huggingface.co/stabilityai/stable-diffusion-2-1-unclip
mv stable-diffusion-2-1-unclip seed_tokenizer/diffusion_model
```


### Inference for visual tokenization and de-tokenization
To discretize an image to 1D visual codes with causal dependency, and reconstruct the image from the visual codes using the off-the-shelf unCLIP SD-UNet:

```bash
cd ..   # SEED/ 
python scripts/seed_tokenizer_inference.py
```

### Launching Gradio Demo of SEED-LLaMA-14B Locally 
Building the local demo of SEED-LLaMA-14B currently requires 2*32GB devices.

```bash
# SEED/
# in first terminal
sh scripts/start_backend.sh
# in second terminal
sh scripts/start_frontend.sh
``` 
Then the demo can be accessed through http://127.0.0.1:80


## Citation
If you find the work helpful, please consider citing:
```bash
@article{ge2023making,
  title={Making LLaMA SEE and Draw with SEED Tokenizer},
  author={Ge, Yuying and Zhao, Sijie and Zeng, Ziyun and Ge, Yixiao and Li, Chen and Wang, Xintao and Shan, Ying},
  journal={arXiv preprint arXiv:2310.01218},
  year={2023}
}

@article{ge2023planting,
  title={Planting a seed of vision in large language model},
  author={Ge, Yuying and Ge, Yixiao and Zeng, Ziyun and Wang, Xintao and Shan, Ying},
  journal={arXiv preprint arXiv:2307.08041},
  year={2023}
}
```

The project is still in progress. Stay tuned for more updates!

## License
`SEED` is released under [Apache License Version 2.0](License.txt). 

`SEED-LLaMA` is released under the original [License](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) of [LLaMA2](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf).

## Acknowledgement
We thank the great work from [unCLIP SD](https://huggingface.co/stabilityai/stable-diffusion-2-1-unclip) and [BLIP2](https://github.com/salesforce/LAVIS).

