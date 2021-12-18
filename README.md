#ECoGGAN

## Introduction
This project implemented a signal processing and GAN model with attention learning for synthesizing aritificial ECoG data

## Code Structure
```

- WGAN:
  Early stage experiments with GANs
  - models
    - WGAN_GP
      Wasserstein GAN with gradient penalty
    - VGG_classifier
      VGG net for classification of ECoG signals for text label prediction, used by late stage evaluations

- ACNECoGGAN
  Our method of DCGAN with attention learning and context normalization approach
  - datasets
    processed EcoG data
  - models
    different combination of D and G
    - discriminator
      - DCGAN with attentive contex normalization
      - traditional GAN
      - WGAN
      - WGAN_GP
    - generator
      - DCGAN with attentive contex normalization
      - traditional GAN
      - WGAN
      - WGAN_GP
  - evaluation
    - FID
    - Inception
    - RMS
- Signal Processing
  Extracting raw data from NWb files and signal processings
	
