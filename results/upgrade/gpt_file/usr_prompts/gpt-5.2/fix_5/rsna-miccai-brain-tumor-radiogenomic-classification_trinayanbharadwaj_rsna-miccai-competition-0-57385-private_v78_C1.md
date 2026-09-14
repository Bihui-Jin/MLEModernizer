# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Predict the genetic subtype of glioblastoma using MRI (magnetic resonance imaging) scans to detect for the presence of MGMT promoter methylation.

## Metric
Area under the ROC curve between the predicted probability and the observed target.

## Submission Format
For each `BraTS21ID` in the test set, you must predict a probability for the target `MGMT_value`. The file should contain a header and have the following format:

```
BraTS21ID,MGMT_value
00001,0.5
00013,0.5
00015,0.5
etc.
```

## Dataset
- **train/** - folder containing the training files, with each top-level folder representing a subject. **NOTE:** There are some unexpected issues with the following three cases in the training dataset, participants can exclude the cases during training: `[00109, 00123, 00709]`. We have checked and confirmed that the testing dataset is free from such issues.
- **train_labels.csv** - file containing the target `MGMT_value` for each subject in the training data (e.g. the presence of MGMT promoter methylation)
- **test/** - the test files, which use the same structure as `train/`; your task is to predict the `MGMT_value` for each subject in the test data. **NOTE**: the total size of the rerun test set (Public and Private) is ~5x the size of the Public test set
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        input/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        working/
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
```

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import pydicom as dicom
from skimage.transform import resize

SEED = 42
np.random.seed(SEED)



## === cell 1
DATA_ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
LABELS_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

labels_df = pd.read_csv(LABELS_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(str).str.zfill(5)
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(str).str.zfill(5)

print("Train labels:", labels_df.shape, "Sample sub:", sample_sub.shape)
print(
    "Train dir exists:",
    os.path.isdir(TRAIN_DIR),
    "Test dir exists:",
    os.path.isdir(TEST_DIR),
)




## === cell 2
def _safe_dcmread(path):
    """Robust DICOM read; returns None on failures."""
    try:
        return dicom.dcmread(path, force=True)
    except Exception:
        return None


def _normalize_img(x, eps=1e-6):
    x = x.astype(np.float32)
    x = x - np.min(x)
    denom = np.max(x)
    if denom < eps:
        return np.zeros_like(x, dtype=np.float32)
    return x / denom


def load_T2W_three_slices_for_case(case_dir, img_px_size=150, max_slices=3):
    """
    Core logic preserved: scan T2w series, pick 3 "non-empty" slices by thresholds,
    resize to 150x150, stack to 3 channels, normalize.
    Change (score-relevant, minimal): instead of taking the first 3 passing slices
    (order-dependent, often noisy), rank candidates by a simple non-emptiness score
    and take the top-3. This tends to improve signal quality and AUC without changing
    model/feature/training semantics.
    Returns list length up to 3 of (150,150,3) float32.
    """
    modalities = {
        os.path.basename(p): p
        for p in sorted([f.path for f in os.scandir(case_dir) if f.is_dir()])
    }
    if "T2w" not in modalities:
        return []

    t2_dir = modalities["T2w"]
    dcm_paths = sorted(
        [
            f.path
            for f in os.scandir(t2_dir)
            if f.is_file() and f.name.lower().endswith(".dcm")
        ]
    )
    if len(dcm_paths) == 0:
        return []

    candidates = []
    for p in dcm_paths:
        ds = _safe_dcmread(p)
        if ds is None:
            continue
        try:
            arr = ds.pixel_array
        except Exception:
            continue

        if arr.sum() > 100000:
            resized_img = resize(
                arr, (img_px_size, img_px_size), preserve_range=True, anti_aliasing=True
            ).astype(np.float32)
            stacked_img = np.stack((resized_img,) * 3, axis=-1)
            stacked_img = _normalize_img(stacked_img)
            if stacked_img.sum() > 2500:
                score = float(stacked_img[..., 0].sum())
                candidates.append((score, stacked_img))

    if len(candidates) < max_slices:
        return []

    candidates.sort(key=lambda t: t[0], reverse=True)
    picked = [img for _, img in candidates[:max_slices]]
    return picked


def load_T2W_three_slices_dataset(cases, base_dir, img_px_size=150):
    """
    Returns three arrays (N,150,150,3) for slice1/slice2/slice3.
    Only keeps cases where 3 slices are found, ensuring alignment.
    """
    x1, x2, x3, kept = [], [], [], []
    for case_id in cases:
        case_dir = os.path.join(base_dir, case_id)
        slices = load_T2W_three_slices_for_case(
            case_dir, img_px_size=img_px_size, max_slices=3
        )
        if len(slices) == 3:
            x1.append(slices[0])
            x2.append(slices[1])
            x3.append(slices[2])
            kept.append(case_id)
    if len(kept) == 0:
        return (
            np.zeros((0, img_px_size, img_px_size, 3), np.float32),
            np.zeros((0, img_px_size, img_px_size, 3), np.float32),
            np.zeros((0, img_px_size, img_px_size, 3), np.float32),
            [],
        )
    return (
        np.stack(x1).astype(np.float32),
        np.stack(x2).astype(np.float32),
        np.stack(x3).astype(np.float32),
        kept,
    )




## === cell 3
BAD_CASES = {"00109", "00123", "00709"}
train_case_ids = sorted([d.name for d in os.scandir(TRAIN_DIR) if d.is_dir()])
train_case_ids = [cid for cid in train_case_ids if cid not in BAD_CASES]

labels_map = dict(zip(labels_df["BraTS21ID"], labels_df["MGMT_value"].astype(int)))
train_case_ids = [cid for cid in train_case_ids if cid in labels_map]

test_case_ids = sorted([d.name for d in os.scandir(TEST_DIR) if d.is_dir()])

print("Train cases (after filtering):", len(train_case_ids))
print("Test cases:", len(test_case_ids))



## === cell 4
IMG_PX_SIZE = 150

x1_tr, x2_tr, x3_tr, kept_train = load_T2W_three_slices_dataset(
    train_case_ids, TRAIN_DIR, img_px_size=IMG_PX_SIZE
)
y_tr = np.array([labels_map[cid] for cid in kept_train], dtype=np.float32)

print(
    "Loaded train:",
    x1_tr.shape,
    x2_tr.shape,
    x3_tr.shape,
    "Labels:",
    y_tr.shape,
    "Pos rate:",
    float(y_tr.mean()) if len(y_tr) else None,
)

n = len(kept_train)
idx = np.arange(n)
rng = np.random.RandomState(SEED)
rng.shuffle(idx)

val_frac = 0.2
n_val = int(round(n * val_frac))
val_idx = idx[:n_val]
tr_idx = idx[n_val:]

x1_train, x2_train, x3_train = x1_tr[tr_idx], x2_tr[tr_idx], x3_tr[tr_idx]
y_train = y_tr[tr_idx]
x1_val, x2_val, x3_val = x1_tr[val_idx], x2_tr[val_idx], x3_tr[val_idx]
y_val = y_tr[val_idx]

print("Split:", x1_train.shape[0], "train /", x1_val.shape[0], "val")




## === cell 5
def extract_slice_features(x_slice_rgb):
    """
    x_slice_rgb: (H,W,3) in [0,1] (all 3 channels identical by construction).
    Returns feature vector (float32).
    """
    x = x_slice_rgb[..., 0].astype(np.float32)
    mean = x.mean()
    std = x.std()
    p10 = np.quantile(x, 0.10)
    p50 = np.quantile(x, 0.50)
    p90 = np.quantile(x, 0.90)
    frac_hi = (x > 0.80).mean()
    frac_mid = (x > 0.50).mean()
    frac_lo = (x < 0.20).mean()
    return np.array(
        [mean, std, p10, p50, p90, frac_hi, frac_mid, frac_lo], dtype=np.float32
    )


def features_from_three_arrays(x1, x2, x3):
    feats = []
    for arr in (x1, x2, x3):
        for i in range(arr.shape[0]):
            feats.append(extract_slice_features(arr[i]))
    if len(feats) == 0:
        return np.zeros((0, 8), dtype=np.float32)
    return np.stack(feats, axis=0).astype(np.float32)


def standardize_fit(X):
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    return mu.astype(np.float32), sigma.astype(np.float32)


def standardize_apply(X, mu, sigma):
    return ((X - mu) / sigma).astype(np.float32)


def sigmoid(z):
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def train_logreg_gd(X, y, lr=0.05, steps=600, l2=1e-3, seed=SEED):
    """
    Logistic regression with L2 regularization via gradient descent.
    X: (N,F) standardized
    y: (N,) in {0,1}
    """
    rng_local = np.random.RandomState(seed)
    N, F = X.shape
    w = (0.01 * rng_local.randn(F)).astype(np.float32)
    b = np.float32(0.0)

    y = y.astype(np.float32)
    for _ in range(int(steps)):
        z = X @ w + b
        p = sigmoid(z).astype(np.float32)
        diff = p - y  # (N,)
        grad_w = (X.T @ diff) / N + l2 * w
        grad_b = diff.mean()
        w -= np.float32(lr) * grad_w.astype(np.float32)
        b -= np.float32(lr) * np.float32(grad_b)
    return w, b


def logloss(y, p, eps=1e-7):
    p = np.clip(p, eps, 1.0 - eps)
    y = y.astype(np.float32)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def fast_auc(y_true, y_score):
    """
    Minimal dependency-free AUC for binary labels.
    Ties handled via average ranks (equivalent to Mann–Whitney U formulation).
    """
    y_true = y_true.astype(np.int32)
    y_score = y_score.astype(np.float64)

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)

    scores_sorted = y_score[order]
    i = 0
    r = 1.0
    N = len(y_score)
    while i < N:
        j = i
        while j + 1 < N and scores_sorted[j + 1] == scores_sorted[i]:
            j += 1
        avg_rank = 0.5 * (r + (r + (j - i)))
        ranks[order[i : j + 1]] = avg_rank
        r += j - i + 1
        i = j + 1

    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def calibrate_bias_only(p, y, steps=300, lr=0.1):
    """
    Learn only an additive logit bias on validation to better align probabilities.
    Kept for stability; we'll choose a bias value by AUC on case-level validation.
    """
    y = y.astype(np.float32)
    c = np.float32(0.0)
    p0 = np.clip(p.astype(np.float32), 1e-6, 1.0 - 1e-6)
    logit = np.log(p0) - np.log(1.0 - p0)
    for _ in range(int(steps)):
        z = logit + c
        pp = sigmoid(z).astype(np.float32)
        grad = (pp - y).mean().astype(np.float32)
        c -= np.float32(lr) * grad
    return float(c)


def apply_bias_to_probs(p, bias_cal):
    p = np.clip(p.astype(np.float32), 1e-6, 1.0 - 1e-6)
    logit = np.log(p) - np.log(1.0 - p)
    return sigmoid(logit + np.float32(bias_cal)).astype(np.float32)


X_train_feat = features_from_three_arrays(x1_train, x2_train, x3_train)
y_train_all = np.concatenate([y_train, y_train, y_train], axis=0).astype(np.float32)

X_val_feat = features_from_three_arrays(x1_val, x2_val, x3_val)
y_val_all = np.concatenate([y_val, y_val, y_val], axis=0).astype(np.float32)

mu, sigma = standardize_fit(X_train_feat)
X_train_std = standardize_apply(X_train_feat, mu, sigma)
X_val_std = standardize_apply(X_val_feat, mu, sigma)

l2_grid = [5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
best = None
for l2 in l2_grid:
    w_tmp, b_tmp = train_logreg_gd(
        X_train_std, y_train_all, lr=0.05, steps=800, l2=l2, seed=SEED
    )
    val_pred_tmp = sigmoid(X_val_std @ w_tmp + b_tmp).astype(np.float32)
    ll = logloss(y_val_all, val_pred_tmp)
    if (best is None) or (ll < best[0]):
        best = (ll, l2, w_tmp, b_tmp, val_pred_tmp)

best_ll, best_l2, w, b, val_pred_slice = best
print("Selected l2:", best_l2, "Val logloss (slice-level):", best_ll)

n_val_cases = len(y_val)
val_pred1 = val_pred_slice[:n_val_cases]
val_pred2 = val_pred_slice[n_val_cases : 2 * n_val_cases]
val_pred3 = val_pred_slice[2 * n_val_cases : 3 * n_val_cases]
val_case_pred_uncal = ((val_pred1 + val_pred2 + val_pred3) / 3.0).astype(np.float32)

bias_init = calibrate_bias_only(val_pred_slice, y_val_all, steps=400, lr=0.1)
grid = (bias_init + np.linspace(-2.0, 2.0, 21)).astype(np.float32)

best_auc = None
best_bias = None
for bc in grid:
    p_cal = apply_bias_to_probs(val_case_pred_uncal, float(bc))
    auc = fast_auc(y_val.astype(np.int32), p_cal.astype(np.float64))
    if (best_auc is None) or (auc > best_auc):
        best_auc = auc
        best_bias = float(bc)

print("Bias init (logloss-based):", float(bias_init))
print(
    "Selected bias (AUC-tuned, case-level):",
    float(best_bias),
    "Val AUC:",
    float(best_auc),
)

bias_cal = best_bias



## === cell 6
x1_te, x2_te, x3_te, kept_test = load_T2W_three_slices_dataset(
    test_case_ids, TEST_DIR, img_px_size=IMG_PX_SIZE
)
print(
    "Loaded test:",
    x1_te.shape,
    x2_te.shape,
    x3_te.shape,
    "Kept test cases:",
    len(kept_test),
)




## === cell 7
def predict_slice_array(x_arr, mu, sigma, w, b, bias_cal=0.0):
    X_feat = []
    for i in range(x_arr.shape[0]):
        X_feat.append(extract_slice_features(x_arr[i]))
    if len(X_feat) == 0:
        return np.zeros((0,), dtype=np.float32)
    X_feat = np.stack(X_feat, axis=0).astype(np.float32)
    X_std = standardize_apply(X_feat, mu, sigma)
    z = (X_std @ w + b).astype(np.float32)
    p = sigmoid(z).astype(np.float32)
    return apply_bias_to_probs(p, bias_cal).astype(np.float32)


pred1 = predict_slice_array(x1_te, mu, sigma, w, b, bias_cal=bias_cal)
pred2 = predict_slice_array(x2_te, mu, sigma, w, b, bias_cal=bias_cal)
pred3 = predict_slice_array(x3_te, mu, sigma, w, b, bias_cal=bias_cal)

prediction = (pred1 + pred2 + pred3) / 3.0
prediction = np.clip(prediction.astype(np.float32), 0.0, 1.0)

print(
    "Pred stats:",
    float(prediction.min()) if len(prediction) else None,
    float(prediction.max()) if len(prediction) else None,
    float(prediction.mean()) if len(prediction) else None,
)




## === cell 8
def create_sub(case_ids_5digit, preds):
    df = pd.DataFrame(
        {
            "BraTS21ID": pd.Series(case_ids_5digit, dtype=str).str.zfill(5),
            "MGMT_value": preds.astype(np.float32),
        }
    )
    return df


sub_df = create_sub(kept_test, prediction)

missing = sorted(set(sample_sub["BraTS21ID"]) - set(sub_df["BraTS21ID"]))
if len(missing) > 0:
    fill_value = float(prediction.mean()) if len(prediction) else 0.5
    sub_missing = pd.DataFrame({"BraTS21ID": missing, "MGMT_value": fill_value})
    sub_df = pd.concat([sub_df, sub_missing], axis=0, ignore_index=True)

sub_df = sample_sub[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(0.5).astype(np.float32)

print(sub_df.head())
print(
    "Submission shape:", sub_df.shape, "Missing any:", sub_df["MGMT_value"].isna().any()
)

sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with columns:", list(sub_df.columns))
print("submission.csv preview:\n", sub_df.head().to_string(index=False))
