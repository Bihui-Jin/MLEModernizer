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

3.10

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
import re
import numpy as np
import pandas as pd

import pydicom as dicom


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATA_ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TEST_DIR = os.path.join(DATA_ROOT, "test")
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TRAIN_LABELS_PATH = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

print("Using TEST_DIR:", TEST_DIR)
print("Using TRAIN_LABELS_PATH:", TRAIN_LABELS_PATH)
print("Using SAMPLE_SUB_PATH:", SAMPLE_SUB_PATH)




## === cell 1
def _safe_listdir(path):
    try:
        return sorted([f.path for f in os.scandir(path) if f.is_dir() or f.is_file()])
    except FileNotFoundError:
        return []


def _read_dcm_pixel_array(dcm_path):
    """Read DICOM and return float32 pixel array; return None if unreadable."""
    try:
        ds = dicom.dcmread(dcm_path, stop_before_pixels=False, force=True)
        arr = ds.pixel_array.astype(np.float32)
        if arr.size == 0:
            return None
        return arr
    except Exception:
        return None


def compute_case_features(case_dir, series_name, max_slices=24):
    """
    Compute robust per-case features from a given series folder.
    Features: mean, std, p10, p50, p90 of normalized slice means + fraction of non-empty slices.
    """
    series_dir = os.path.join(case_dir, series_name)
    dcm_files = (
        [
            f.path
            for f in os.scandir(series_dir)
            if f.is_file() and f.name.lower().endswith(".dcm")
        ]
        if os.path.isdir(series_dir)
        else []
    )
    if len(dcm_files) == 0:
        return np.zeros(6, dtype=np.float32)

    dcm_files = sorted(dcm_files)
    if len(dcm_files) > max_slices:
        idx = np.linspace(0, len(dcm_files) - 1, max_slices).round().astype(int)
        dcm_files = [dcm_files[i] for i in idx]

    slice_means = []
    nonempty = 0
    for p in dcm_files:
        arr = _read_dcm_pixel_array(p)
        if arr is None:
            continue
        s = float(np.nansum(arr))
        if not np.isfinite(s) or s <= 0:
            continue
        nonempty += 1
        m = float(np.nanmean(arr))
        sd = float(np.nanstd(arr))
        slice_means.append(m / (sd + 1e-6))

    if len(slice_means) == 0:
        return np.zeros(6, dtype=np.float32)

    v = np.array(slice_means, dtype=np.float32)
    feat = np.array(
        [
            float(np.mean(v)),
            float(np.std(v)),
            float(np.percentile(v, 10)),
            float(np.percentile(v, 50)),
            float(np.percentile(v, 90)),
            float(nonempty) / float(len(dcm_files)),
        ],
        dtype=np.float32,
    )
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return feat


def build_features_for_split(root_dir, ids, max_slices=24):
    """
    Build features for each case id using series: FLAIR, T1wCE, T2w.
    (Keeps core "multi-sequence" idea from original notebook, but without deep models.)
    """
    X = np.zeros((len(ids), 18), dtype=np.float32)  # 3 series * 6 feats
    for i, case_id in enumerate(ids):
        case_dir = os.path.join(root_dir, case_id)
        f_flair = compute_case_features(case_dir, "FLAIR", max_slices=max_slices)
        f_t1wce = compute_case_features(case_dir, "T1wCE", max_slices=max_slices)
        f_t2w = compute_case_features(case_dir, "T2w", max_slices=max_slices)
        X[i, 0:6] = f_flair
        X[i, 6:12] = f_t1wce
        X[i, 12:18] = f_t2w
        if (i + 1) % 50 == 0:
            print(f"Built features for {i+1}/{len(ids)} cases")
    return X


def _is_valid_case_id(name: str) -> bool:
    return bool(re.fullmatch(r"\d{5}", str(name)))


def _id_to_int_safe(case_id: str) -> int:
    s = str(case_id)
    m = re.search(r"\d+", s)
    return int(m.group(0)) if m else 0




## === cell 2
labels_df = pd.read_csv(TRAIN_LABELS_PATH)
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(str).str.zfill(5)
labels_df = labels_df.set_index("BraTS21ID").sort_index()

bad_ids = {"00109", "00123", "00709"}

train_case_dirs = [
    os.path.basename(p) for p in _safe_listdir(TRAIN_DIR) if os.path.isdir(p)
]
train_ids_all = sorted([cid for cid in train_case_dirs if _is_valid_case_id(cid)])
train_ids = sorted(
    [cid for cid in train_ids_all if cid in labels_df.index and cid not in bad_ids]
)

print("Train cases available (dirs):", len(train_case_dirs))
print("Train cases valid ID format:", len(train_ids_all))
print("Train cases with labels (after exclusion):", len(train_ids))

test_case_dirs = [
    os.path.basename(p) for p in _safe_listdir(TEST_DIR) if os.path.isdir(p)
]
test_ids_all = sorted([cid for cid in test_case_dirs if _is_valid_case_id(cid)])
test_ids = test_ids_all
print("Test cases available (dirs):", len(test_case_dirs))
print("Test cases valid ID format:", len(test_ids))

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(str).str.zfill(5)
print("Sample submission rows:", len(sample_sub))

sample_ids = sample_sub["BraTS21ID"].tolist()
sample_id_set = set(sample_ids)
missing_in_test_dir = sorted([sid for sid in sample_ids if sid not in set(test_ids)])
if len(missing_in_test_dir) > 0:
    print(
        "Warning: sample IDs missing from TEST_DIR listing (will be filled):",
        missing_in_test_dir[:10],
    )



## === cell 3
X_train = build_features_for_split(TRAIN_DIR, train_ids, max_slices=24)
y_train = labels_df.loc[train_ids, "MGMT_value"].astype(np.float32).values

X_test = build_features_for_split(TEST_DIR, test_ids, max_slices=24)

print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)




## === cell 4
def fit_ridge_logistic_regression(X, y, l2=1.0, iters=2000, lr=0.05):
    """
    Simple logistic regression with L2 regularization using gradient descent.
    No external deps; stable and fast for small feature vectors.
    """
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    n, d = X.shape

    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-6
    Xs = (X - mu) / sd

    w = np.zeros((d, 1), dtype=np.float32)
    b = np.float32(0.0)

    for t in range(iters):
        z = Xs @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        dz = (p.reshape(-1) - y).reshape(-1, 1) / n
        grad_w = Xs.T @ dz + (l2 / n) * w
        grad_b = np.float32(dz.sum())

        w -= lr * grad_w.astype(np.float32)
        b -= lr * grad_b

    def predict_proba(Xnew):
        Xnew = Xnew.astype(np.float32)
        Xnew_s = (Xnew - mu) / sd
        z = Xnew_s @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        return p.reshape(-1).astype(np.float32)

    return predict_proba


predict_proba = fit_ridge_logistic_regression(
    X_train, y_train, l2=2.0, iters=2500, lr=0.05
)

test_pred_model = predict_proba(X_test).astype(np.float32)

id_ints = np.array([_id_to_int_safe(x) for x in test_ids], dtype=np.float32)
id_jitter = (id_ints % 997.0) / 997.0
score_for_ranking = (
    test_pred_model + 1e-7 * id_jitter
)  # negligible magnitude, tie-break only

order = np.argsort(score_for_ranking)  # ascending score
n = len(test_ids)
ramp = np.linspace(
    1.0 - 1e-4, 1e-4, n, dtype=np.float32
)  # strictly decreasing probabilities
test_pred = np.empty(n, dtype=np.float32)
test_pred[order] = ramp

test_pred = np.clip(test_pred, 1e-4, 1 - 1e-4)

print(
    "Pred stats:",
    float(test_pred.min()),
    float(test_pred.mean()),
    float(test_pred.max()),
)



## === cell 5
pred_map = {cid: float(p) for cid, p in zip(test_ids, test_pred)}

sub_df = sample_sub.copy()
sub_df["MGMT_value"] = sub_df["BraTS21ID"].map(pred_map).astype(np.float32)

if sub_df["MGMT_value"].isna().any():
    fill_val = float(np.mean(y_train)) if len(y_train) else 0.5
    sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(fill_val).astype(np.float32)

sub_df = sub_df[["BraTS21ID", "MGMT_value"]]
sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(str).str.zfill(5)
sub_df["MGMT_value"] = sub_df["MGMT_value"].astype(float).clip(0.0, 1.0)

print(sub_df.head())
print("Submission rows:", len(sub_df))

sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print("Wrote:", sub_path)
print("File size (bytes):", os.path.getsize(sub_path))
