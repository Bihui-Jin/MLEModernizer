# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

3.7

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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

BASE_PATH = "./"
TRAIN_PATH = os.path.join(BASE_PATH, "train")
TEST_PATH = os.path.join(BASE_PATH, "test")
if not os.path.isdir(TRAIN_PATH):
    for root, dirs, _ in os.walk("."):
        if "train" in dirs and "test" in dirs:
            TRAIN_PATH = os.path.join(root, "train")
            TEST_PATH = os.path.join(root, "test")
            break

IMG_SIZE = (64, 64)  # increased from (32, 32)




## === cell 1
def _process_image(args):
    """Helper for parallel image loading."""
    fpath, label_val = args
    try:
        img = Image.open(fpath).convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # (64,64,3)
        return arr.ravel(), label_val
    except Exception:
        return None  # signal failure


def load_images_and_labels(train_path):
    """Load all images in parallel, preserving deterministic order."""
    file_label_pairs = []
    for label_name, label_val in [("cat", 0), ("dog", 1)]:
        class_dir = os.path.join(train_path, label_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            fpath = os.path.join(class_dir, fname)
            file_label_pairs.append((fpath, label_val))

    X_list = []
    y_list = []
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as executor:
        for result in executor.map(_process_image, file_label_pairs):
            if result is not None:
                vec, lbl = result
                X_list.append(vec)
                y_list.append(lbl)

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.int8)
    return X, y


print("Loading training images…")
X_train, y_train = load_images_and_labels(TRAIN_PATH)
print(f"Loaded {X_train.shape[0]} images with {X_train.shape[1]} features each.")

X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0) + 1e-8  # avoid division by zero
X_train_norm = (X_train - X_mean) / X_std


def train_logreg(X, y, lr=0.0003, epochs=12000, reg=0.001, shuffle=True, seed=42):
    """
    Batch gradient descent logistic regression with L2 regularisation.
    Hyper‑parameters have been softened (lower lr, higher reg, more epochs) to
    allow better convergence on raw‑pixel features, which should improve log‑loss.
    """
    rng = np.random.default_rng(seed)
    w = np.zeros(X.shape[1], dtype=np.float32)
    b = 0.0
    n = X.shape[0]

    idx = np.arange(n, dtype=np.int64)

    for epoch in range(epochs):
        if shuffle:
            idx = rng.permutation(n)
            y_perm = y[idx]
            logits = X.dot(w)[idx] + b
        else:
            y_perm = y
            logits = X.dot(w) + b

        probs = 1.0 / (1.0 + np.exp(-logits))
        residual = probs - y_perm  # shape (n,)

        grad_w = (X.T @ residual) / n + reg * w
        grad_b = residual.mean()

        w -= lr * grad_w
        b -= lr * grad_b

    return w, b


print("Training logistic regression model…")
weights, bias = train_logreg(X_train_norm, y_train, lr=0.0003, epochs=12000, reg=0.001)
print("Model training completed.")




## === cell 2
def image_to_prob(filepath: str) -> float:
    """
    Compute dog‑probability using the trained logistic‑regression model.
    Returns 0.5 if the image cannot be read.
    """
    try:
        img = Image.open(filepath).convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
        vec = np.asarray(img, dtype=np.float32).ravel() / 255.0
    except Exception:
        return 0.5
    vec_norm = (vec - X_mean) / X_std
    logit = vec_norm.dot(weights) + bias
    logit = np.clip(logit, -20, 20)
    prob = 1.0 / (1.0 + np.exp(-logit))
    prob = np.clip(prob, 0.001, 0.999)
    return prob




## === cell 3
test_files = []
for root, _, files in os.walk(TEST_PATH):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            test_files.append(os.path.join(root, f))
test_files.sort()
ids = [os.path.splitext(os.path.basename(f))[0] for f in test_files]


def _compute_prob(p):
    return image_to_prob(p)


with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as executor:
    probs = list(executor.map(_compute_prob, test_files))
probs = np.array(probs, dtype=np.float32)

submission = pd.DataFrame({"id": ids, "label": probs})
submission = submission.sort_values("id")
print("Submission preview:")
print(submission.head())




## === cell 4
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")
