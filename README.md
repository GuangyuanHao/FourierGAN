# Fourier Generative Adversarial Networks (FourierGAN)
  This projects is under research. It is based on BEGAN to solve how to make a model can learn to generate images which contain different high or low frequency components compared to that of training datasets.
  
  This model's novel points: 
  1. Two discriminators "teach" one generator. 
  2. A high frequency filter and a low frequency filer are put in front of first layers of two discriminators respectively, so the discriminators' ability is limited and also they can "teach" the generator different knowledge. One "teaches" generator to generate images with how much rate of high frequency components and anther "teaches" generator to generate images with how much rate of low frequency components.

# Current Results on CelebA

The project needs to be improved further.

## Keeping rate of high frequency components canstant and increasing rate of low frequency components from top to bottom in the follwing image:
On this ocassion, images's color become richer and richer, as real images' changing with increasing rate of low frequency components.
![1](https://github.com/GuangyuanHao/WaveletGAN/raw/master/results/samples.png)
