# Implementing GAN for Fashion-MNIST Dataset

This repository contains a modular TensorFlow/Keras implementation of a Deep Convolutional Generative Adversarial Network (DCGAN) to generate synthetic fashion item images from the Fashion-MNIST dataset.

## Repository Progress & Milestones

- [x] **Data Processing Pipeline**: Scripts for downloading, normalizing to `[-1, 1]`, and setting up efficient `tf.data` input batch pipelines.
- [x] **DCGAN Generator**: Fully modularized layers and block structures:
  - Input Dense projections
  - Reshaping layers
  - Conv2DTranspose blocks upsampling from `7x7` → `14x14` → `28x28`.
- [x] **DCGAN Discriminator**: Fully modularized layers and block structures:
  - Input Layer block
  - Conv2D downsampling blocks from `28x28` → `14x14` → `7x7`.
  - Flatten layer block
  - Output dense logit layer
- [ ] **Loss Functions and Optimizer Setup** (Next Milestone)
- [ ] **Custom GAN Training Loop** (Next Milestone)
- [ ] **FastAPI Backend & React Frontend Web UI** (Final Milestone)