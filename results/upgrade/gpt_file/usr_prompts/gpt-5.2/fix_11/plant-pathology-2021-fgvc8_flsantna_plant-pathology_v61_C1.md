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

# 5. Target score

0.8192243767313039

# 6. Current score

0.43403

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.46051) has done: 'The main timeout driver is feature extraction: you decode and resize ~15k train + ~3.7k test JPEGs sequentially in Python/PIL, which dominates runtime. I keep the exact same features and modeling logic, but speed it up by (1) precomputing a label→index map (avoids repeated `classes.index`), (2) using a faster, equivalent gradient-magnitude computation (avoids large temporary arrays), and (3) parallelizing image feature extraction with a process pool while preserving determinism by writing results back to fixed indices. I also precompute threshold calibration components once so the grid search doesn’t repeatedly re-run work that can be reused without changing semantics.'
- What this solution (achieved 0.46051) has done: 'The timeout is dominated by image feature extraction (PIL open/resize + NumPy ops) across ~15k train and ~3.7k test images, and the current multiprocessing setup adds overhead by sending many small tasks and doing redundant list/parse work. I keep the exact same features/model/training and only reduce constant factors: precompute labels via vectorized encoding, switch to a faster/fewer-copy feature extractor while keeping identical math, and optimize parallelism by using `Pool.starmap` with larger chunks and preallocated outputs (less IPC and Python overhead). I also prevent oversubscription (BLAS threads vs processes) and ensure deterministic seeds remain intact. The training/calibration/prediction logic stays unchanged.'
- What this solution (achieved 0.43403) has done: 'Your current score (0.46051) is far below the target (0.81922), so we should improve prediction quality with minimal changes that keep your feature extractor and LogisticRegression-per-class setup intact. The biggest accuracy issue is that the model is trained as 6 independent binary classifiers, but the metric is multilabel mean F1; switching to a true multilabel classifier with the same base estimator (OneVsRestClassifier wrapping LogisticRegression) keeps the same core model family while better matching the task. Then, instead of a per-class threshold grid that’s slow and can overfit per-class independently, we tune a single global probability threshold on the validation set (plus the same “ensure at least one label” rule), which typically improves mean F1 for this competition and is a small change in post-processing. Finally, we make the test image list come from `sample_submission.csv` order (avoids any potential mismatch) while still extracting features the same way and writing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

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
from sklearn.multiclass import OneVsRestClassifier

_LABEL_TO_IDX = {c: i for i, c in enumerate(dataset_labels)}


def load_image_features(img_path: str):
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        im = im.resize((128, 128), resample=Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32)
    arr *= 1.0 / 255.0  # in-place normalization

    means = arr.mean(axis=(0, 1))
    stds = arr.std(axis=(0, 1))

    gray = arr.mean(axis=2)
    g_mean = float(gray.mean())
    g_std = float(gray.std())

    gx = gray[1:, 1:] - gray[1:, :-1]
    gy = gray[1:, 1:] - gray[:-1, 1:]
    mag = np.hypot(gx, gy)
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


def _feat_worker_path(p):
    return load_image_features(p)


def extract_features_parallel(paths, n_features=10, progress_every=2000, desc=""):
    import multiprocessing as mp

    n = len(paths)
    X_out = np.empty((n, n_features), dtype=np.float32)

    n_workers = min(8, (os.cpu_count() or 2))
    ctx = mp.get_context("fork") if hasattr(mp, "get_context") else mp

    chunksize = max(128, n // (n_workers * 32) if n_workers > 0 else 128)

    with ctx.Pool(processes=n_workers) as pool:
        it = pool.imap(_feat_worker_path, paths, chunksize=chunksize)
        for j, feat in enumerate(it, start=1):
            X_out[j - 1] = feat
            if progress_every and (j % progress_every == 0):
                print(f"{desc}Extracted features: {j}/{n}")
    return X_out


def binarize_with_threshold_and_fallback(P, thr: float):
    y_pred = (P >= float(thr)).astype(np.int32)
    empty = y_pred.sum(axis=1) == 0
    if np.any(empty):
        argm = np.argmax(P[empty], axis=1)
        y_pred[empty] = 0
        y_pred[empty, argm] = 1
    return y_pred


img_paths = [os.path.join(train_dir, fn) for fn in data_set["image"].tolist()]

labels_series = data_set["labels"].fillna("").astype(str)
Y = np.zeros((len(labels_series), len(dataset_labels)), dtype=np.int32)
for cls, j in _LABEL_TO_IDX.items():
    mask = labels_series.str.contains(rf"(?:^|\s){cls}(?:\s|$)", regex=True).to_numpy()
    Y[mask, j] = 1

X = extract_features_parallel(
    img_paths, n_features=10, progress_every=2000, desc="Train "
)
print("X shape:", X.shape, "Y shape:", Y.shape)

X_tr, X_va, y_tr, y_va = train_test_split(
    X, Y, test_size=0.2, random_state=SEED, shuffle=True
)

base_lr = LogisticRegression(
    max_iter=500,
    solver="lbfgs",
    class_weight="balanced",
    random_state=SEED,
)
ovr = OneVsRestClassifier(base_lr)
ovr.fit(X_tr, y_tr)

P_va = ovr.predict_proba(X_va).astype(np.float32)

grid = np.linspace(0.05, 0.95, 19, dtype=np.float32)
best_thr = 0.5
best_val = -1.0
for t in grid:
    y_pred = binarize_with_threshold_and_fallback(P_va, float(t))
    sc = mean_f1_score(y_va, y_pred)
    if sc > best_val:
        best_val = sc
        best_thr = float(t)

y_pred_best = binarize_with_threshold_and_fallback(P_va, best_thr)
val_score = mean_f1_score(y_va, y_pred_best)

print("Chosen global threshold:", round(best_thr, 4))
print("Validation mean F1:", round(val_score, 6))



## === cell 2
sample = pd.read_csv(sample_sub_path)
test_images = sample["image"].astype(str).tolist()
test_paths = [os.path.join(test_dir, fn) for fn in test_images]

missing = [p for p in test_paths if not os.path.exists(p)]
if missing:
    raise FileNotFoundError(
        f"Missing {len(missing)} test images; first missing: {missing[0]}"
    )

X_test = extract_features_parallel(
    test_paths, n_features=10, progress_every=2000, desc="Test  "
)

P_test = ovr.predict_proba(X_test).astype(np.float32)

Y_test_bin = binarize_with_threshold_and_fallback(P_test, best_thr)

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

sub = sample[["image"]].merge(sub, on="image", how="left")
sub["labels"] = sub["labels"].fillna("healthy")

csv_path = os.path.join(output_dir, "submission.csv")
sub.to_csv(csv_path, index=False)
print("Wrote:", csv_path, "rows:", len(sub), "cols:", list(sub.columns))
print(sub.head())
