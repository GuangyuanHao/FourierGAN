# Fourier Generative Adversarial Networks (FourierGAN)
  This projects is under research. It is based on BEGAN to solve how to make a model can learn to generate images which contain different high or low frequency components compared to that of training datasets.
  
  This model's novel points: 
  1. Two discriminators "teach" one generator. 
  2. A high frequency filter and a low frequency filer are put in front of first layers of two discriminators respectively, so the discriminators' ability is limited and also they can "teach" the generator different knowledge. One "teaches" generator to generate images with how much rate of high frequency components and anther "teaches" generator to generate images with how much rate of low frequency components.

# Current Results on CelebA

The project needs to be improved further. Experiments met expectation, but backgrounds of faces are not diverse, at the same time, when the rate between high frequency components and low frequency components is one, the images are not clear as BEGAN's results. So this project needs to be improved further.

## Keeping rate of high frequency components constant and increasing rate of low frequency components from top to bottom in the following image:
  Images' color become richer and richer, as real images' changing with increasing rate of low frequency components.
  ![1](https://github.com/GuangyuanHao/FourierGAN/raw/master/results/high.jpg)
## Keeping rate of low frequency components constant and decreasing rate of high frequency components from top to bottom in the following image:
  Images' definition become lower and lower, as real images' changing with decreasing rate of high frequency components.
  ![1](https://github.com/GuangyuanHao/FourierGAN/raw/master/results/low.jpg)
  
