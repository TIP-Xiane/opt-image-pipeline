import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

SEED = 42
IMG_SIZE = (64, 64)  # resize all shapes to 64x64 for consistent feature size

SHAPES = [
    'antidiagonal', 'Circle', 'diagonal', 'Diamond', 'Heart',
    'horizontal', 'Rectangle', 'Square', 'Triangle', 'vertical'
]

# ── Load and augment each shape image ────────────────────
# Since we only have 1 image per class, we augment to get enough samples
np.random.seed(SEED)

X_all, y_all = [], []

for label, shape in enumerate(SHAPES):
    img = Image.open(f'data/{shape}.png').convert('L').resize(IMG_SIZE)
    base = np.array(img).astype(np.float32) / 255.0
    flat = base.flatten()

    # augment: add gaussian noise variants to get 200 samples per class
    for _ in range(200):
        noise = np.random.normal(0, 0.05, flat.shape)
        sample = np.clip(flat + noise, 0, 1)
        X_all.append(sample)
        y_all.append(label)

X = np.array(X_all)
y = np.array(y_all)

# ── Train / val / test split ──────────────────────────────
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=SEED, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp)

np.savez('data/data_split.npz',
         X_train=X_train, y_train=y_train,
         X_val=X_val,     y_val=y_val,
         X_test=X_test,   y_test=y_test)

# ── Initial theta ─────────────────────────────────────────
d = X.shape[1]   # 64*64 = 4096
K = len(SHAPES)  # 10
np.random.seed(SEED)
theta0 = np.random.randn(d, K) * 0.01
np.save('data/model_init_theta.npy', theta0)

# ── Sample indices (1 per class from test set) ────────────
sample_indices = []
for k in range(K):
    idxs = np.where(y_test == k)[0]
    sample_indices.append(idxs[0])
sample_indices = np.array(sample_indices)  # all 10 classes
np.save('data/sample_indices.npy', sample_indices)

# ── original_samples.png (2 rows × 5 cols, all 10 shapes) ─
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, idx in enumerate(sample_indices):
    ax = axes.flat[i]
    ax.imshow(X_test[idx].reshape(IMG_SIZE), cmap='gray')
    ax.axis('off')
    ax.set_title(SHAPES[y_test[idx]])
plt.tight_layout()
plt.savefig('data/original_samples.png', dpi=150)
plt.close()

# ── data_model_log.md ─────────────────────────────────────
with open('data/data_model_log.md', 'w') as f:
    f.write('# Data and Model Log\n\n')
    f.write('## Dataset\n')
    f.write('- Source: shape PNG images in data/\n')
    f.write(f'- Classes ({K}): {", ".join(SHAPES)}\n')
    f.write(f'- Image size: {IMG_SIZE[0]}x{IMG_SIZE[1]} (resized, grayscale)\n')
    f.write(f'- Features: {d} (flattened)\n')
    f.write('- Augmentation: 200 samples/class via Gaussian noise (std=0.05)\n')
    f.write(f'- Total samples: {len(X)}\n')
    f.write(f'- Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}\n')
    f.write(f'- Seed: {SEED}\n\n')
    f.write('## Model\n')
    f.write('- Type: Multiclass Logistic Regression (softmax)\n')
    f.write(f'- theta shape: ({d}, {K})\n')
    f.write('- Init: random normal (mean=0, std=0.01)\n')

print(f'Done. X shape: {X.shape}, classes: {SHAPES}')
print(f'Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}')
