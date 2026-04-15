# Data and Model Log

## Dataset
- Source: shape PNG images in data/
- Classes (10): antidiagonal, Circle, diagonal, Diamond, Heart, horizontal, Rectangle, Square, Triangle, vertical
- Image size: 64x64 (resized, grayscale)
- Features: 4096 (flattened)
- Augmentation: 200 samples/class via Gaussian noise (std=0.05)
- Total samples: 2000
- Train: 1200, Val: 400, Test: 400
- Seed: 42

## Model
- Type: Multiclass Logistic Regression (softmax)
- theta shape: (4096, 10)
- Init: random normal (mean=0, std=0.01)
