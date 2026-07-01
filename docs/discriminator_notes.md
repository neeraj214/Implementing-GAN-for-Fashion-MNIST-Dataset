# Discriminator Architecture Notes

## Design Pattern: DCGAN-style Discriminator
Standard CNN architecture for classifying 28x28 grayscale images as real or fake. It acts as the classification network that downsamples inputs.

## Layer-by-Layer Flow
1. **Input**: Image of shape (28, 28, 1) normalized to range [-1, 1].
2. **Block 1** (stride=2): Downsamples 28×28 → 14×14, depth 1→64. Uses LeakyReLU (alpha=0.2) and Dropout (0.3).
3. **Block 2** (stride=2): Downsamples 14×14 → 7×7, depth 64→128. Uses LeakyReLU (alpha=0.2) and Dropout (0.3).
4. **Flatten**: Reshapes 7x7x128 feature map to a flat vector of size 6272.
5. **Output**: Dense projection from 6272 → 1 producing a single raw logit (representing real/fake probability, activation sigmoid will be modeled in loss for numerical stability).

## Why LeakyReLU with alpha=0.2?
DCGAN standard activation for the discriminator. Standard ReLU blocks backward gradients when values are negative, whereas LeakyReLU allows a small positive gradient (slope 0.2) to pass through, keeping gradients flowing even for poorly performing generator outputs.

## Why Dropout?
Helps prevent the discriminator from overfitting or learning features too fast compared to the generator. Dropout forces the discriminator to generalize rather than memorizing noise patterns.

## Parameter Count
- **Total Parameters**: 210,049
- **Trainable Parameters**: 210,049
- **Non-trainable Parameters**: 0
