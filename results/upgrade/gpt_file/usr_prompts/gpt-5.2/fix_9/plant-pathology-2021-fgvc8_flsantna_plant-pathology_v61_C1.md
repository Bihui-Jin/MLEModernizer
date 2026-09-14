# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

SEED = 1337
random.seed(SEED)
np.random.seed(SEED)

output_dir = "./"
base_dir = "../input/plant-pathology-2021-fgvc8"
train_csv_path = os.path.join(base_dir, "train.csv")
train_dir = os.path.join(base_dir, "train_images")
test_dir = os.path.join(base_dir, "test_images")
sample_sub_path = os.path.join(base_dir, "sample_submission.csv")

assert os.path.exists(train_csv_path), f"train.csv not found: {train_csv_path}"
assert os.path.isdir(train_dir), f"train_images dir not found: {train_dir}"
assert os.path.isdir(test_dir), f"test_images dir not found: {test_dir}"
assert os.path.exists(
    sample_sub_path
), f"sample_submission.csv not found: {sample_sub_path}"

data_set = pd.read_csv(train_csv_path)

dataset_labels = [
    "complex",
    "frog_eye_leaf_spot",
    "healthy",
    "powdery_mildew",
    "rust",
    "scab",
]
print(
    "Loaded train:", data_set.shape, "Num classes:", len(dataset_labels), dataset_labels
)



## === cell 1
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

_LABEL_TO_IDX = {c: i for i, c in enumerate(dataset_labels)}


def parse_multilabels(label_str: str, classes):
    s = str(label_str) if label_str is not None else ""
    tokens = [t for t in s.strip().split(" ") if t]
    y = np.zeros(len(classes), dtype=np.int32)
    for t in tokens:
        j = _LABEL_TO_IDX.get(t)
        if j is not None:
            y[j] = 1
    return y


def load_image_features(img_path: str):
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        im = im.resize((128, 128), resample=Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0  # (H,W,3)

    means = arr.mean(axis=(0, 1))
    stds = arr.std(axis=(0, 1))

    gray = arr.mean(axis=2)
    g_mean = float(gray.mean())
    g_std = float(gray.std())

    gx = gray[1:, 1:] - gray[1:, :-1]  # shape (127,127)
    gy = gray[1:, 1:] - gray[:-1, 1:]  # shape (127,127)
    mag = np.sqrt(gx * gx + gy * gy)
    mag_mean = float(mag.mean())
    mag_std = float(mag.std())

    return np.concatenate([means, stds, [g_mean, g_std, mag_mean, mag_std]]).astype(
        np.float32
    )


def mean_f1_score(y_true, y_pred_bin):
    eps = 1e-9
    tp = (y_true * y_pred_bin).sum(axis=1).astype(np.float32)
    fp = ((1 - y_true) * y_pred_bin).sum(axis=1).astype(np.float32)
    fn = (y_true * (1 - y_pred_bin)).sum(axis=1).astype(np.float32)
    f1 = (2 * tp) / (2 * tp + fp + fn + eps)
    return float(np.mean(f1))


def _feat_worker(args):
    i, p = args
    return i, load_image_features(p)


def extract_features_parallel(paths, n_features=10, progress_every=2000, desc=""):
    import multiprocessing as mp

    X_out = np.zeros((len(paths), n_features), dtype=np.float32)

    n_workers = min(8, (os.cpu_count() or 2))
    ctx = mp.get_context("fork") if hasattr(mp, "get_context") else mp

    with ctx.Pool(processes=n_workers) as pool:
        for j, (i, feat) in enumerate(
            pool.imap_unordered(_feat_worker, enumerate(paths), chunksize=64), start=1
        ):
            X_out[i] = feat
            if progress_every and (j % progress_every == 0):
                print(f"{desc}Extracted features: {j}/{len(paths)}")
    return X_out


def binarize_topk_with_floor(P, k=2, floor=0.0):
    """
    Change is only in prediction post-processing to better match mean F1:
    choose top-k labels per image (common for this dataset), optionally filtering by prob>=floor,
    and always ensure at least one label.
    """
    n, c = P.shape
    out = np.zeros((n, c), dtype=np.int32)
    topk_idx = np.argpartition(-P, kth=min(k, c) - 1, axis=1)[:, : min(k, c)]
    rows = np.arange(n)[:, None]
    out[rows, topk_idx] = 1
    if floor > 0.0:
        out = out * (P >= float(floor)).astype(np.int32)

    empty = out.sum(axis=1) == 0
    if np.any(empty):
        argm = np.argmax(P[empty], axis=1)
        out[empty] = 0
        out[empty, argm] = 1
    return out


img_paths = [os.path.join(train_dir, fn) for fn in data_set["image"].tolist()]
Y = np.stack(
    [parse_multilabels(s, dataset_labels) for s in data_set["labels"].tolist()], axis=0
)

X = extract_features_parallel(
    img_paths, n_features=10, progress_every=2000, desc="Train "
)
print("X shape:", X.shape, "Y shape:", Y.shape)

X_tr, X_va, y_tr, y_va = train_test_split(
    X, Y, test_size=0.2, random_state=SEED, shuffle=True
)

models = []
for k in range(Y.shape[1]):
    lr = LogisticRegression(
        max_iter=500,
        solver="lbfgs",
        class_weight="balanced",
        random_state=SEED,
    )
    lr.fit(X_tr, y_tr[:, k])
    models.append(lr)

P_va = np.stack([m.predict_proba(X_va)[:, 1] for m in models], axis=1)

thresholds = np.full(Y.shape[1], 0.5, dtype=np.float32)
grid = np.linspace(0.1, 0.9, 17, dtype=np.float32)

va_argmax = np.argmax(P_va, axis=1)

for k in range(Y.shape[1]):
    best_t = thresholds[k]
    best_k_score = -1.0
    for t in grid:
        tmp = thresholds.copy()
        tmp[k] = t
        y_pred = (P_va >= tmp[None, :]).astype(np.int32)
        empty = y_pred.sum(axis=1) == 0
        if np.any(empty):
            y_pred[empty] = 0
            y_pred[empty, va_argmax[empty]] = 1
        sc = mean_f1_score(y_va, y_pred)
        if sc > best_k_score:
            best_k_score = sc
            best_t = float(t)
    thresholds[k] = best_t

y_pred_thr = (P_va >= thresholds[None, :]).astype(np.int32)
empty = y_pred_thr.sum(axis=1) == 0
if np.any(empty):
    y_pred_thr[empty] = 0
    y_pred_thr[empty, va_argmax[empty]] = 1
val_score_thr = mean_f1_score(y_va, y_pred_thr)

best_post = ("thresholds", None, None)
best_val = val_score_thr

for K in (1, 2, 3):
    for floor in (0.0, 0.2, 0.3):
        y_pred_k = binarize_topk_with_floor(P_va, k=K, floor=floor)
        sc = mean_f1_score(y_va, y_pred_k)
        if sc > best_val:
            best_val = sc
            best_post = ("topk", K, float(floor))

print("Calibrated thresholds:", dict(zip(dataset_labels, thresholds.round(3).tolist())))
print("Validation mean F1 (thresholds):", val_score_thr)
print("Chosen post-processing:", best_post, "Validation mean F1:", best_val)



## === cell 2
test_images = sorted([fn for fn in os.listdir(test_dir) if fn.lower().endswith(".jpg")])
test_paths = [os.path.join(test_dir, fn) for fn in test_images]

X_test = extract_features_parallel(
    test_paths, n_features=10, progress_every=2000, desc="Test  "
)

P_test = np.stack([m.predict_proba(X_test)[:, 1] for m in models], axis=1)

if best_post[0] == "topk":
    _, K, floor = best_post
    Y_test_bin = binarize_topk_with_floor(P_test, k=int(K), floor=float(floor))
else:
    Y_test_bin = (P_test >= thresholds[None, :]).astype(np.int32)
    empty = Y_test_bin.sum(axis=1) == 0
    if np.any(empty):
        argm = np.argmax(P_test[empty], axis=1)
        Y_test_bin[empty] = 0
        Y_test_bin[empty, argm] = 1

healthy_idx = dataset_labels.index("healthy") if "healthy" in dataset_labels else -1
empty = Y_test_bin.sum(axis=1) == 0
if np.any(empty):
    if healthy_idx >= 0:
        Y_test_bin[empty, healthy_idx] = 1
    else:
        argm = np.argmax(P_test[empty], axis=1)
        Y_test_bin[empty] = 0
        Y_test_bin[empty, argm] = 1

idx_to_label = np.array(dataset_labels, dtype=object)
out_labels = []
for i in range(Y_test_bin.shape[0]):
    idxs = np.flatnonzero(Y_test_bin[i])
    label_str = " ".join(idx_to_label[idxs].tolist())
    out_labels.append(label_str if label_str else "healthy")

sub = pd.DataFrame({"image": test_images, "labels": out_labels})

sample = pd.read_csv(sample_sub_path)
if "image" in sample.columns and len(sample) == len(sub):
    sub = sample[["image"]].merge(sub, on="image", how="left")
    sub["labels"] = sub["labels"].fillna("healthy")

csv_path = os.path.join(output_dir, "submission.csv")
sub.to_csv(csv_path, index=False)
print("Wrote:", csv_path, "rows:", len(sub), "cols:", list(sub.columns))
print(sub.head())
