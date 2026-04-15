"""
run_lbfgs.py  —  Reproduce the L-BFGS experiment and save all artifacts to results/
Run from project root: python results/run_lbfgs.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from scipy.optimize import fmin_l_bfgs_b

os.makedirs('results', exist_ok=True)

# ── Config ───────────────────────────────────────────────
SEED     = 42
REG      = 1e-4
MAX_ITER = 200
IMG_SIZE = (64, 64)
SHAPES   = ['antidiagonal', 'Circle', 'diagonal', 'Diamond', 'Heart',
            'horizontal', 'Rectangle', 'Square', 'Triangle', 'vertical']
# ─────────────────────────────────────────────────────────

data             = np.load('data/data_split.npz')
X_train, y_train = data['X_train'], data['y_train']
X_val,   y_val   = data['X_val'],   data['y_val']
X_test,  y_test  = data['X_test'],  data['y_test']
theta0           = np.load('data/model_init_theta.npy')
sample_indices   = np.load('data/sample_indices.npy')

d, K = theta0.shape


def softmax(Z):
    e = np.exp(Z - Z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def loss_and_grad(theta_vec):
    theta = theta_vec.reshape(d, K)
    N = X_train.shape[0]
    P = softmax(X_train.dot(theta))
    Y = np.eye(K)[y_train]
    loss = -np.sum(Y * np.log(P + 1e-12)) / N + 0.5 * REG * np.sum(theta ** 2)
    grad = X_train.T.dot(P - Y) / N + REG * theta
    return loss, grad.ravel().astype(np.float64)


history = []
start   = time.time()


def callback(theta_vec):
    theta = theta_vec.reshape(d, K)
    loss, _ = loss_and_grad(theta_vec)
    train_acc = (np.argmax(X_train.dot(theta), axis=1) == y_train).mean()
    val_acc   = (np.argmax(X_val.dot(theta),   axis=1) == y_val).mean()
    history.append({
        'iteration':       len(history),
        'objective':       float(loss),
        'train_accuracy':  float(train_acc),
        'val_accuracy':    float(val_acc),
        'elapsed_seconds': time.time() - start,
    })


print('Running L-BFGS...')
theta_opt, _, _ = fmin_l_bfgs_b(
    loss_and_grad,
    theta0.ravel().astype(np.float64),
    callback=callback,
    maxiter=MAX_ITER
)
theta_final = theta_opt.reshape(d, K)
print(f'Done. {len(history)} iterations.')

# ── lbfgs_history.csv ────────────────────────────────────
pd.DataFrame(history).to_csv('results/lbfgs_history.csv', index=False)

# ── lbfgs_theta.npy ──────────────────────────────────────
np.save('results/lbfgs_theta.npy', theta_final)

# ── lbfgs_images.png  (6 rows × 2 cols) ──────────────────
# left = original test sample, right = class mean of predicted shape
class_means = np.array([
    X_train[y_train == k].mean(axis=0) if (y_train == k).any() else np.zeros(d)
    for k in range(K)
])

fig, axes = plt.subplots(10, 2, figsize=(4, 20))
for i, idx in enumerate(sample_indices):
    pred = int(np.argmax(X_test[idx].dot(theta_final)))
    axes[i, 0].imshow(X_test[idx].reshape(IMG_SIZE), cmap='gray')
    axes[i, 0].axis('off')
    axes[i, 0].set_title(f'Original\n({SHAPES[y_test[idx]]})', fontsize=7)
    axes[i, 1].imshow(class_means[pred].reshape(IMG_SIZE), cmap='gray')
    axes[i, 1].axis('off')
    axes[i, 1].set_title(f'Pred: {SHAPES[pred]}', fontsize=7)
plt.tight_layout()
plt.savefig('results/lbfgs_images.png', dpi=150)
plt.close()
print(f'Image grid: {len(sample_indices)} samples (5 rows x 2 cols)')

# ── lbfgs_log.md ─────────────────────────────────────────
final = history[-1]
with open('results/lbfgs_log.md', 'w') as f:
    f.write('# L-BFGS Experiment Log\n\n')
    f.write(f'- seed: {SEED}\n')
    f.write(f'- regularization (L2 lambda): {REG}\n')
    f.write(f'- max_iter: {MAX_ITER}\n')
    f.write('- stopping criteria: L-BFGS default (gtol=1e-5)\n')
    f.write(f'- iterations run: {len(history)}\n')
    f.write(f'- final objective: {final["objective"]:.6f}\n')
    f.write(f'- final train_accuracy: {final["train_accuracy"]:.4f}\n')
    f.write(f'- final val_accuracy: {final["val_accuracy"]:.4f}\n')
    f.write(f'- total elapsed_seconds: {final["elapsed_seconds"]:.4f}\n\n')
    f.write('## Notes\n')
    f.write('- Optimizer: scipy.optimize.fmin_l_bfgs_b\n')
    f.write('- Dataset: shape images (64x64 grayscale, 10 classes)\n')
    f.write(f'- Classes: {", ".join(SHAPES)}\n')
    f.write('- Model: multiclass softmax logistic regression\n')

print('Artifacts saved to results/:')
print('  lbfgs_history.csv')
print('  lbfgs_theta.npy')
print('  lbfgs_images.png')
print('  lbfgs_log.md')
