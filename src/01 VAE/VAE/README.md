# VAE (Variational Auto Encoder)
## Introduction
VAE는 AutoEncoder를 확장하여 입력 데이터를 잠재 공간(latent space)에 확률적으로 인코딩하고
그 latent vector로부터 데이터를 다시 복원하는 생성 모델
## Method
### AutoEncoder
updating...

- 일반적인 AutoEncoder는 Encoder로부터 입력 데이터를 압축하고 Decoder로 다시 확장하여 복원
- 그러나, AutoEncoder는 생성 모델이 아니기 때문에 샘플링(sampling)을 할 수가 없음

### Variational AutoEncoder(VAE)
![vae](/assets/VAE/vae.png)

- VAE는 AutoEncoder에 확률 인코딩이란 개념을 도입
- Encoder로부터 압축된 잠재 벡터를 **확률 분포**로 표현할 수 있도록 학습
- 즉, 새로운 $z\sim \mathcal{N}(0,I)$ 샘플로부터 새로운 데이터 생성 가능

### Loss function
![loss](/assets/VAE/loss.png)

- **Reconstruction Loss**: 복원된 $\hat{x}$가 입력 $x$와 얼마나 유사한지
- **KL Divergence**: Encoder가 만든 확률 분포 $q(z|x)$와 표준 정규 분포 $p(z)=\mathcal{N}(0,1)$ 사이의 거리

## Sampling
- VAE는 latent vector $z$를 **정규 분포 $z\sim \mathcal{N}(\mu, \sigma^2)$**에서 샘플링
- 그러나 정규분포는 무작위 연산이기 때문에 **비미분 가능(discrete)** 연산으로 역전파(backpropagation)가 안됨!

### Reparameterization Trick
![reparameterization](/assets/VAE/reparameterization.png)
- 이 트릭은 Sampling을 미분 가능한 연산으로 바꿔줌
- VAE의 인코더는 평균($\mu$)과 분산($\sigma^2$)를 출력하며 이들은 학습이 가능함
- $\epsilon$은 표준 정규분포에서 샘플링된 노이즈
- 위 식은 샘플링 과정을 확률론적(stochastic)에서 결정론적(deterministic)으로 바꾸어 미분 가능하게 함

## Reference
- Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes." 20 Dec. 2013
- https://kyujinpy.tistory.com/88
