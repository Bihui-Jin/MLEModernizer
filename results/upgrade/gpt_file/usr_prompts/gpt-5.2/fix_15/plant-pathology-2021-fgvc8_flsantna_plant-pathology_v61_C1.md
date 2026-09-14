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

0.45736

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.46051) has done: 'The main timeout driver is feature extraction: you decode and resize ~15k train + ~3.7k test JPEGs sequentially in Python/PIL, which dominates runtime. I keep the exact same features and modeling logic, but speed it up by (1) precomputing a label→index map (avoids repeated `classes.index`), (2) using a faster, equivalent gradient-magnitude computation (avoids large temporary arrays), and (3) parallelizing image feature extraction with a process pool while preserving determinism by writing results back to fixed indices. I also precompute threshold calibration components once so the grid search doesn’t repeatedly re-run work that can be reused without changing semantics.'
- What this solution (achieved 0.46051) has done: 'The timeout is dominated by image feature extraction (PIL open/resize + NumPy ops) across ~15k train and ~3.7k test images, and the current multiprocessing setup adds overhead by sending many small tasks and doing redundant list/parse work. I keep the exact same features/model/training and only reduce constant factors: precompute labels via vectorized encoding, switch to a faster/fewer-copy feature extractor while keeping identical math, and optimize parallelism by using `Pool.starmap` with larger chunks and preallocated outputs (less IPC and Python overhead). I also prevent oversubscription (BLAS threads vs processes) and ensure deterministic seeds remain intact. The training/calibration/prediction logic stays unchanged.'
- What this solution (achieved 0.43403) has done: 'Your current score (0.46051) is far below the target (0.81922), so we should improve prediction quality with minimal changes that keep your feature extractor and LogisticRegression-per-class setup intact. The biggest accuracy issue is that the model is trained as 6 independent binary classifiers, but the metric is multilabel mean F1; switching to a true multilabel classifier with the same base estimator (OneVsRestClassifier wrapping LogisticRegression) keeps the same core model family while better matching the task. Then, instead of a per-class threshold grid that’s slow and can overfit per-class independently, we tune a single global probability threshold on the validation set (plus the same “ensure at least one label” rule), which typically improves mean F1 for this competition and is a small change in post-processing. Finally, we make the test image list come from `sample_submission.csv` order (avoids any potential mismatch) while still extracting features the same way and writing a valid `submission.csv`.'
- What this solution (achieved 0.45881) has done: 'Your current score (0.43403) is far below the target (0.81922), so we should improve predictive quality while keeping your feature extractor and OVR-LogisticRegression core intact. The main issue is that mean F1 in this competition is computed per-class then averaged (macro), but your local threshold tuning optimizes per-image F1; I switch the validation objective to macro-F1 and tune per-class thresholds (a small post-processing change) to better match the leaderboard metric. I also use a stratified split based on the number of labels per image to stabilize validation and reduce threshold overfitting/instability without changing the model. Finally, I keep the same submission writing logic/order to ensure a valid `submission.csv`.'
- What this solution (achieved 0.45736) has done: 'Main bottlenecks are (1) multiprocessing overhead from shipping thousands of file-path tasks to workers and (2) expensive `sklearn.metrics.f1_score` called repeatedly inside threshold search. I keep the exact feature set, model, and thresholding logic, but speed up feature extraction by batching paths per worker (dramatically fewer IPC calls) and by using a faster multiprocessing context. I also replace repeated `f1_score` calls with an equivalent, vectorized macro-F1 computation (same semantics, including `zero_division=0`) to accelerate threshold tuning without changing the selected thresholds. Finally, I avoid redundant work/allocations and keep determinism intact.'

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


def _feat_worker_batch(paths_batch):
    out = np.empty((len(paths_batch), 10), dtype=np.float32)
    for i, p in enumerate(paths_batch):
        out[i] = load_image_features(p)
    return out


def extract_features_parallel(paths, n_features=10, progress_every=2000, desc=""):
    import multiprocessing as mp

    n = len(paths)
    X_out = np.empty((n, n_features), dtype=np.float32)

    cpu = os.cpu_count() or 2
    n_workers = min(8, cpu)

    if os.name == "posix":
        ctx = mp.get_context("fork")
    else:
        ctx = mp.get_context("spawn")

    batch_size = 64
    n_batches = (n + batch_size - 1) // batch_size
    chunksize = max(1, n_batches // (n_workers * 4) if n_workers > 0 else 1)

    batches = [paths[i : i + batch_size] for i in range(0, n, batch_size)]

    write_pos = 0
    done = 0
    with ctx.Pool(processes=n_workers, maxtasksperchild=2000) as pool:
        for feats in pool.imap(_feat_worker_batch, batches, chunksize=chunksize):
            bs = feats.shape[0]
            X_out[write_pos : write_pos + bs] = feats
            write_pos += bs
            done += bs
            if progress_every and (done % progress_every == 0 or done == n):
                print(f"{desc}Extracted features: {done}/{n}")
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


def _macro_f1_zero_division0(y_true, y_pred):
    y_true = y_true.astype(np.int32, copy=False)
    y_pred = y_pred.astype(np.int32, copy=False)
    tp = (y_true & y_pred).sum(axis=0).astype(np.float64)
    fp = ((1 - y_true) & y_pred).sum(axis=0).astype(np.float64)
    fn = (y_true & (1 - y_pred)).sum(axis=0).astype(np.float64)
    denom = 2.0 * tp + fp + fn
    f1 = np.zeros_like(denom, dtype=np.float64)
    mask = denom > 0
    f1[mask] = (2.0 * tp[mask]) / denom[mask]
    return float(f1.mean())


def tune_thresholds_macro_coordinate_ascent(P_va, y_va, init_thrs, grid, n_passes=2):
    thrs = np.array(init_thrs, dtype=np.float32, copy=True)

    best_pred = binarize_with_thresholds_and_fallback(P_va, thrs)
    best_score = _macro_f1_zero_division0(y_va, best_pred)

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
                sc = _macro_f1_zero_division0(y_va, y_pred)
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
        tp = int(((y_true_k == 1) & (y_hat_k == 1)).sum())
        fp = int(((y_true_k == 0) & (y_hat_k == 1)).sum())
        fn = int(((y_true_k == 1) & (y_hat_k == 0)).sum())
        denom = 2 * tp + fp + fn
        sc = (2 * tp / denom) if denom > 0 else 0.0
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
