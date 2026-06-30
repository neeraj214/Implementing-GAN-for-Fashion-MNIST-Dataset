# Generator Architecture Notes

## Design Pattern: DCGAN-style Generator
Standard architecture for 28x28 grayscale image generation from a 100-dim latent vector.

## Layer-by-Layer Flow
1. **Input**: Latent vector (100,) sampled from normal distribution
2. **Dense Projection**: 100 → 12544 (7×7×256), no bias (BatchNorm handles it)
3. **Reshape**: Flat vector → spatial 7×7×256 feature map
4. **Block 1** (stride=1): Refines features at 7×7, depth 256→128
5. **Block 2** (stride=2): Upsamples 7×7→14×14, depth 128→64
6. **Output Block** (stride=2): Upsamples 14×14→28×28, depth 64→1, tanh activation

## Why no bias in Conv2DTranspose?
BatchNormalization already learns a shift parameter, making bias redundant.

## Why tanh on output?
Matches the [-1, 1] normalization range used for real training images,
keeping generator and discriminator inputs on the same scale.

## Why LeakyReLU instead of ReLU?
Prevents "dying ReLU" problem common in GAN training where dead neurons
stop the generator from improving.

## Parameter Count
- **Total Parameters**: 2,330,944
- **Trainable Parameters**: 2,305,472
- **Non-trainable Parameters**: 25,472
