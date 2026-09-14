# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Given a dataset of images of dogs and cats, predict if an image is a dog or a cat.

## Metric
Log loss.

## Submission Format
For each image in the test set, you must submit a probability that image is a dog. The file should have a header and be in the following format:

```
id,label
1,0.5
2,0.5
3,0.5
...
```

## Dataset
The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id.

# 2. Python version

3.13

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
        input/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
        working/
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
```

-> data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> working/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

# 5. Target score

0.6362163647676209

# 6. Current score

5.70054

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 5.70054) has done: 'The main blocker is that TensorFlow fails to import due to an incompatible protobuf/runtime in this Kaggle image, so the script crashes before any data prep/training. To keep core training logic intact while making it run end-to-end, I switch to a pure-standard-library + NumPy implementation: load images with PIL, train a simple logistic regression with SGD on flattened pixels, and write a correct `submission.csv` (`id,label`). I also fix the train/test path handling by using the already-extracted folder structure under `/kaggle/input/dogs-vs-cats-redux-kernels-edition/train/{cat,dog}` and `/kaggle/input/dogs-vs-cats-redux-kernels-edition/test/unknown`, avoiding zip extraction errors. The output probabilities are clipped for log-loss safety and aligned/sorted by numeric id.'

# 9. Code solution

## === cell 0
import os
import re
import math
import time
import random
import numpy as np
import pandas as pd

from PIL import Image

print("Python OK. Starting pipeline...")




## === cell 1
def binary_accuracy_np(y_true, y_pred_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(np.int32).ravel()
    y_pred = (np.asarray(y_pred_prob).ravel() > threshold).astype(np.int32)
    return (y_true == y_pred).mean()


def sigmoid(x):
    x = np.asarray(x, dtype=np.float32)
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out


def bce_loss(y, p, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


SEED = 42
random.seed(SEED)
np.random.seed(SEED)



## === cell 2
IMG_SIZE = 150
BATCH_SIZE = 32
EPOCHS = 1
VALIDATION_SPLIT = 0.2
MODEL_NUM = 1

WORK_DIR = "/kaggle/working"
DATASET_DIR = "/kaggle/input/dogs-vs-cats-redux-kernels-edition"

train_cat_dir = os.path.join(DATASET_DIR, "train", "cat")
train_dog_dir = os.path.join(DATASET_DIR, "train", "dog")
test_dir = os.path.join(DATASET_DIR, "test", "unknown")

print("train_cat_dir exists:", os.path.exists(train_cat_dir), train_cat_dir)
print("train_dog_dir exists:", os.path.exists(train_dog_dir), train_dog_dir)
print("test_dir exists:", os.path.exists(test_dir), test_dir)

if not (
    os.path.isdir(train_cat_dir)
    and os.path.isdir(train_dog_dir)
    and os.path.isdir(test_dir)
):
    raise FileNotFoundError(
        "Expected extracted train/test directories not found. "
        f"train_cat_dir={train_cat_dir}, train_dog_dir={train_dog_dir}, test_dir={test_dir}"
    )




## === cell 3
def list_jpgs(d):
    return [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(".jpg")]


cat_files = sorted(list_jpgs(train_cat_dir))
dog_files = sorted(list_jpgs(train_dog_dir))
test_files = sorted(list_jpgs(test_dir))

print("Train counts:", "cat =", len(cat_files), "dog =", len(dog_files))
print("Test count:", len(test_files))

if len(cat_files) == 0 or len(dog_files) == 0:
    raise RuntimeError(
        "No training images found under expected train/cat and train/dog directories."
    )
if len(test_files) == 0:
    raise RuntimeError("No test images found under expected test/unknown directory.")




## === cell 4
def load_image_as_vector(path, img_size=150):
    with Image.open(path) as im:
        im = im.convert("RGB")
        im = im.resize((img_size, img_size), resample=Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr.reshape(-1)


X_paths = cat_files + dog_files
y = np.array([0] * len(cat_files) + [1] * len(dog_files), dtype=np.float32)

idx = np.arange(len(X_paths))
rng = np.random.default_rng(SEED)
rng.shuffle(idx)
X_paths = [X_paths[i] for i in idx]
y = y[idx]

n_total = len(X_paths)
n_val = int(round(n_total * VALIDATION_SPLIT))
n_train = n_total - n_val

train_paths = X_paths[:n_train]
val_paths = X_paths[n_train:]
y_train = y[:n_train]
y_val = y[n_train:]

print(f"Split: n_train={n_train}, n_val={n_val}")

t0 = time.time()
X_train = np.stack([load_image_as_vector(p, IMG_SIZE) for p in train_paths], axis=0)
X_val = np.stack([load_image_as_vector(p, IMG_SIZE) for p in val_paths], axis=0)
print(
    "Loaded arrays:",
    X_train.shape,
    X_val.shape,
    "in",
    round(time.time() - t0, 2),
    "sec",
)

mu = X_train.mean(axis=0, keepdims=True)
sigma = X_train.std(axis=0, keepdims=True) + 1e-6
X_train_s = (X_train - mu) / sigma
X_val_s = (X_val - mu) / sigma



## === cell 5


def train_logreg_sgd(X, y, epochs=1, batch_size=32, lr=0.05, l2=0.001, seed=42):
    n, d = X.shape
    w = np.zeros((d,), dtype=np.float32)
    b = np.float32(0.0)
    rng = np.random.default_rng(seed)

    for ep in range(epochs):
        order = np.arange(n)
        rng.shuffle(order)
        Xo = X[order]
        yo = y[order]

        for start in range(0, n, batch_size):
            xb = Xo[start : start + batch_size]
            yb = yo[start : start + batch_size]

            logits = xb @ w + b
            p = sigmoid(logits)
            diff = (p - yb).astype(np.float32)
            gw = (xb.T @ diff) / xb.shape[0]
            gb = diff.mean()
            gw += l2 * w

            w -= lr * gw
            b -= lr * gb

        p_tr = sigmoid(X @ w + b)
        p_va = sigmoid(X_val_s @ w + b)  # uses outer-scope val set for reporting only
        print(
            f"Epoch {ep+1}/{epochs} - "
            f"train_loss={bce_loss(y, p_tr):.4f} val_loss={bce_loss(y_val, p_va):.4f} "
            f"val_acc={binary_accuracy_np(y_val, p_va):.4f}"
        )

    return w, b


models_params = []
for i in range(MODEL_NUM):
    print(f"model : {i}")
    w, b = train_logreg_sgd(
        X_train_s,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=0.05,
        l2=0.001,
        seed=SEED + i,
    )
    models_params.append((w, b))

for i, (w, b) in enumerate(models_params):
    np.savez(os.path.join(WORK_DIR, f"model{i}.npz"), w=w, b=b)
print("Saved model parameter files to", WORK_DIR)



## === cell 6
sum_pred = None
for i in range(MODEL_NUM):
    params = np.load(os.path.join(WORK_DIR, f"model{i}.npz"))
    w = params["w"]
    b = params["b"]
    pred = sigmoid(X_val_s @ w + b).reshape(-1, 1)
    sum_pred = pred if sum_pred is None else (sum_pred + pred)

avg_pred = (sum_pred / MODEL_NUM).ravel()
acc = binary_accuracy_np(y_val, avg_pred, threshold=0.5)
print(f"Validation accuracy: {acc:.4f}")
print(f"Validation logloss: {bce_loss(y_val, np.clip(avg_pred, 1e-7, 1-1e-7)):.4f}")




## === cell 7
def extract_number(filename):
    m = re.search(r"\d+", os.path.basename(filename))
    return int(m.group()) if m else -1


sorted_test_paths = sorted(test_files, key=extract_number)
test_ids = [int(os.path.splitext(os.path.basename(p))[0]) for p in sorted_test_paths]

t0 = time.time()
X_test = np.stack(
    [load_image_as_vector(p, IMG_SIZE) for p in sorted_test_paths], axis=0
)
X_test_s = (X_test - mu) / sigma
print("Loaded test array:", X_test_s.shape, "in", round(time.time() - t0, 2), "sec")

sum_pred = None
for i in range(MODEL_NUM):
    params = np.load(os.path.join(WORK_DIR, f"model{i}.npz"))
    w = params["w"]
    b = params["b"]
    pred = sigmoid(X_test_s @ w + b).reshape(-1, 1)
    sum_pred = pred if sum_pred is None else (sum_pred + pred)

avg_pred = (sum_pred / MODEL_NUM).ravel()
labels = np.clip(avg_pred, 1e-6, 1 - 1e-6)

sub = pd.DataFrame(
    {"id": np.asarray(test_ids, dtype=int), "label": labels.astype(float)}
)
sub = sub.sort_values("id").reset_index(drop=True)
sub_path = os.path.join(WORK_DIR, "submission.csv")
sub.to_csv(sub_path, index=False)

print("Wrote submission.csv with shape:", sub.shape, "to", sub_path)
print(sub.head())
print(sub.tail())
