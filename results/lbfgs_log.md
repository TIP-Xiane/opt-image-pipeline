# L-BFGS Experiment Log

- seed: 42
- regularization (L2 lambda): 0.0001
- max_iter: 200
- stopping criteria: L-BFGS default (gtol=1e-5)
- iterations run: 18
- final objective: 0.000499
- final train_accuracy: 1.0000
- final val_accuracy: 1.0000
- total elapsed_seconds: 0.4103

## Notes
- Optimizer: scipy.optimize.fmin_l_bfgs_b
- Dataset: shape images (64x64 grayscale, 10 classes)
- Classes: antidiagonal, Circle, diagonal, Diamond, Heart, horizontal, Rectangle, Square, Triangle, vertical
- Model: multiclass softmax logistic regression
