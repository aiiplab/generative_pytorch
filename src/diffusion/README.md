# Diffusion Models

## List of Models
1) ```DDPM(Denoising Diffusion Probabilistic Models)```
2) ```DDIM(Denoising Diffusion Implicit Models)```
3) ```Score-SDE(Score-based Generative Models)```
4) ```CFG(Classifier-Free Diffusion Guidance)```

## Introduction
![diffusion main](/assets/Diffusion/diffusion_main.png)
<p style="display: flex; justify-content: center; align-items: center; gap: 300px;">
  <img src="/assets/Diffusion/diffusion_image.gif" width="300">
  <img src="/assets/Diffusion/diffusion_distribution.gif" width="300">
  <img src="/assets/Diffusion/diffusion_sample.gif" width="300">
</p>

**Diffusion Model** is generative model that learns **forward process** that gradually covers data with Gaussian noise,</br>
and **reverse process** that removes this noise in reverse to reconstruct the original.

## Forward Process
![diffusion forward](/assets/Diffusion/diffusion_forward.png)

- $q(x_t|x_{t-1})=\mathcal{N}(x_t;\sqrt{1-\beta_t}x_{t-1}, \beta_t\text{I})$ $\longrightarrow$ $q(x_{1:T}|x_0)=\prod_{t=1}^T q(x_t|x_{t-1})$
- The given image now follows a probability distribution with mean $\sqrt{1-\beta_t}x_{t-1}$ and variance $\beta_t \text{I}$ after 1 step.
- The process of adding noise while reducing the proportion of signal in the data.

## Reverse Process
![diffusion_reverse](/assets/Diffusion/diffusion_reverse.png)

- $p(x_t)=\mathcal{N}(x_T;0,\text{I})$
- $p_\theta(x_{t-1}|x_t)=\mathcal{N}(x_{t-1};\mu_\theta(x_t,t), \sigma^2\text{I})$
- Fix the variance of the data distribution of the reverse process and **predict the mean of the data distribution at each step**


## Model Architecture
![diffusion_unet](/assets/Diffusion/diffusion_unet.png)

- The model uses the [U-Net](https://arxiv.org/abs/1505.04597) architecture
- Consists of ```Residual Blocks```, ```Attention Layers```, and ```Skip Connections```
- Embed timestep $t$ and apply element-wise sum to the input of each block
- Model takes $x_t$ as input and predicts the noise $\mu_\theta(x_t, t)$ that was added to $x_t$.

## DDPM(Denoising Diffusion Probabilistic Models)
### Training
![ddpm_training](/assets/Diffusion/ddpm_training.png)

- Add noise $\epsilon$ to input data $x_0$ and transform it into $x_t$
- U-Net takes $x_t$ as input and **predicts noise $\epsilon_\theta(x_t, t)$**

### Sampling
![ddpm_sampling](/assets/Diffusion/ddpm_sampling.png)

- Starting from Gaussian Noise x_T, **predict noise $\epsilon_\theta$ for each Timestep**.
- **Generate $x_{t-1}$ of the previous step** according to the Sampling Algorithm.
- At this time, noise $z$ is added according to the variance $\sigma_t^2=\beta_t$ during the prediction process.
  - Denoising Flow **Deterministic** $\rightarrow$ **Stochastic**

## DDIM(Denoising diffusion implicit models)
updaiting...

## SGD(Score-Based Generative Models)
updaiting...

## CFG(Classifier-Free Diffusion Guidance)
updaiting...

## Using code
Currently only DDPM is implemented...

### 1. root
```
root
├─ src
│  └─ diffusion
│     ├─ model.py         # Diffusion Model(U-Net) & Gaussian Diffusion code
│     ├─ distributed.py   # Utility code for distributed learning
│     ├─ train.py         # Diffusion Model's training code
│     ├─ practice.ipynb   # Diffusion Model's tutorial notebook
│     └─ README.md
```

### 2. train script
- if you use 1 GPU
```
CUDA_VISIBLE_DEVICES=0 python src/diffusion/train.py --batch_size=128 --epochs=30
```
- If you use multi GPU
```
torchrun --nproc_per_node=2 python src/diffusion/train.py --batch_size=128 --epochs=30
``` 

### 3. notebook
```practice.ipynb``` is available at [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aiiplab/generative_pytorch/blob/main/src/diffusion/pratice.ipynb)

## Reference
- Ho et al. [Denoising diffusion probabilistic models](https://arxiv.org/abs/2006.11239) NeurIPS 2020
- Song et al. [Denoising diffusion implicit models](https://arxiv.org/abs/2010.02502) arXiv preprint 2020
- Song et al. [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) ICLR 2021
- Ho et al. [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) NeurIPS 2022
