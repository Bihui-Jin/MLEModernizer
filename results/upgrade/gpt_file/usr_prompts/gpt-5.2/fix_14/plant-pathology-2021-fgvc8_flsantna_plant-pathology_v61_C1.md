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
from sklearn.metrics import f1_score

_LABEL_TO_IDX = {c: i for i, c in enumerate(dataset_labels)}


def load_image_features(img_path: str):
    with Image.open(img_path) as im:
        im = im.convert("RGB")

        w, h = im.size
        if w > 256 or h > 256:
            r = 1
            while (w // (r * 2) >= 128) and (h // (r * 2) >= 128):
                r *= 2
            if r > 1:
                im = im.reduce(r)

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
        np.float32, copy=False
    )


def _feat_worker_path(p):
    return load_image_features(p)


def extract_features_parallel(paths, n_features=10, progress_every=2000, desc=""):
    import multiprocessing as mp

    n = len(paths)
    X_out = np.empty((n, n_features), dtype=np.float32)

    cpu = os.cpu_count() or 2
    n_workers = min(8, cpu)

    try:
        ctx = mp.get_context("forkserver")
    except ValueError:
        ctx = mp.get_context("fork")

    chunksize = max(256, n // (n_workers * 16) if n_workers > 0 else 256)

    with ctx.Pool(processes=n_workers, maxtasksperchild=2000) as pool:
        it = pool.imap(_feat_worker_path, paths, chunksize=chunksize)
        for j, feat in enumerate(it, start=1):
            X_out[j - 1] = feat
            if progress_every and (j % progress_every == 0):
                print(f"{desc}Extracted features: {j}/{n}")
    return X_out


def binarize_with_thresholds_and_fallback(P, thrs):
    thrs = np.asarray(thrs, dtype=np.float32).reshape(1, -1)
    y_pred = (P >= thrs).astype(np.int32)
    empty = y_pred.sum(axis=1) == 0
    if np.any(empty):
        argm = np.argmax(P[empty], axis=1)
        y_pred[empty] = 0
        y_pred[empty, argm] = 1
    return y_pred


def tune_thresholds_macro_coordinate_ascent(P_va, y_va, init_thrs, grid, n_passes=2):
    thrs = np.array(init_thrs, dtype=np.float32, copy=True)

    best_pred = binarize_with_thresholds_and_fallback(P_va, thrs)
    best_score = f1_score(y_va, best_pred, average="macro", zero_division=0)

    for _ in range(int(n_passes)):
        improved_any = False
        for k in range(P_va.shape[1]):
            cur_best_t = float(thrs[k])
            cur_best_score = float(best_score)
            for t in grid:
                if float(t) == cur_best_t:
                    continue
                cand = thrs.copy()
                cand[k] = float(t)
                y_pred = binarize_with_thresholds_and_fallback(P_va, cand)
                sc = f1_score(y_va, y_pred, average="macro", zero_division=0)
                if sc > cur_best_score + 1e-12:
                    cur_best_score = float(sc)
                    cur_best_t = float(t)
            if cur_best_t != float(thrs[k]):
                thrs[k] = cur_best_t
                best_score = cur_best_score
                improved_any = True
        if not improved_any:
            break
    return thrs, float(best_score)


img_paths = [os.path.join(train_dir, fn) for fn in data_set["image"].tolist()]

labels_series = data_set["labels"].fillna("").astype(str).tolist()
Y = np.zeros((len(labels_series), len(dataset_labels)), dtype=np.int32)
for i, s in enumerate(labels_series):
    if not s:
        continue
    for tok in s.split():
        j = _LABEL_TO_IDX.get(tok)
        if j is not None:
            Y[i, j] = 1

X = extract_features_parallel(
    img_paths, n_features=10, progress_every=2000, desc="Train "
)
print("X shape:", X.shape, "Y shape:", Y.shape)

label_count = Y.sum(axis=1)
strata = np.clip(label_count, 0, 3).astype(int)  # 0,1,2,3(3+)
X_tr, X_va, y_tr, y_va = train_test_split(
    X, Y, test_size=0.2, random_state=SEED, shuffle=True, stratify=strata
)

base_lr = LogisticRegression(
    max_iter=500,
    solver="lbfgs",
    class_weight="balanced",
    random_state=SEED,
)
ovr = OneVsRestClassifier(base_lr)
ovr.fit(X_tr, y_tr)

P_va = ovr.predict_proba(X_va).astype(np.float32, copy=False)

grid = np.linspace(0.05, 0.95, 19, dtype=np.float32)

init_thrs = np.full((len(dataset_labels),), 0.5, dtype=np.float32)
for k in range(len(dataset_labels)):
    best_t = 0.5
    best_f1 = -1.0
    y_true_k = y_va[:, k]
    P_k = P_va[:, k]
    for t in grid:
        y_hat_k = (P_k >= float(t)).astype(np.int32)
        sc = f1_score(y_true_k, y_hat_k, zero_division=0)
        if sc > best_f1:
            best_f1 = sc
            best_t = float(t)
    init_thrs[k] = best_t

best_thrs, val_macro_f1 = tune_thresholds_macro_coordinate_ascent(
    P_va, y_va, init_thrs=init_thrs, grid=grid, n_passes=3
)

print(
    "Chosen per-class thresholds:",
    {c: round(float(t), 4) for c, t in zip(dataset_labels, best_thrs)},
)
print("Validation macro F1:", round(float(val_macro_f1), 6))



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

P_test = ovr.predict_proba(X_test).astype(np.float32, copy=False)

Y_test_bin = binarize_with_thresholds_and_fallback(P_test, best_thrs)

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
out_labels = [
    " ".join(idx_to_label[np.flatnonzero(row)]) or "healthy" for row in Y_test_bin
]

sub = pd.DataFrame({"image": test_images, "labels": out_labels})
sub = sample[["image"]].merge(sub, on="image", how="left")
sub["labels"] = sub["labels"].fillna("healthy")

csv_path = os.path.join(output_dir, "submission.csv")
sub.to_csv(csv_path, index=False)
print("Wrote:", csv_path, "rows:", len(sub), "cols:", list(sub.columns))
print(sub.head())
