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
import glob
import random
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

TYPES = ["FLAIR", "T1w", "T2w", "T1wCE"]
EXCLUDE = [109, 123, 709]

DATA_ROOT = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")

train_df = pd.read_csv(os.path.join(DATA_ROOT, "train_labels.csv"))
sample_sub = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))

train_df["BraTS21ID"] = train_df["BraTS21ID"].astype(int)
train_df = train_df[~train_df.BraTS21ID.isin(EXCLUDE)].reset_index(drop=True)

SEQ = "T1wCE"  # preserve original single-sequence intent


def _slice_sort_key(path: str) -> int:
    base = os.path.basename(path)
    name = os.path.splitext(base)[0]
    try:
        return int(name.split("-")[-1])
    except Exception:
        return 0


def get_all_image_paths(brats21id, image_type, folder="train"):
    """
    Preserve original intent: take middle 50% and sample by interval.
    Uses DICOMs (dataset provides .dcm), but also supports png/jpg if present.
    """
    assert image_type in TYPES
    base = TRAIN_DIR if folder == "train" else TEST_DIR
    patient_path = os.path.join(base, str(int(brats21id)).zfill(5), image_type)

    img_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        img_paths.extend(glob.glob(os.path.join(patient_path, ext)))
    img_paths = sorted(img_paths, key=_slice_sort_key)

    if len(img_paths) == 0:
        img_paths = sorted(
            glob.glob(os.path.join(patient_path, "*.dcm")), key=_slice_sort_key
        )

    num_images = len(img_paths)
    if num_images == 0:
        return np.array([], dtype=object)

    start = int(num_images * 0.25)
    end = int(num_images * 0.75)
    interval = 3 if num_images >= 10 else 1
    return np.array(img_paths[start:end:interval], dtype=object)


def _is_supported_image_path(p: str) -> bool:
    p = str(p).lower()
    return (
        p.endswith(".png")
        or p.endswith(".jpg")
        or p.endswith(".jpeg")
        or p.endswith(".dcm")
    )


def _read_bytes_features(path: str) -> np.ndarray:
    """
    Avoid pydicom/tensorflow entirely to prevent protobuf runtime crashes.
    Keep slice-level pipeline by extracting simple, deterministic features from file bytes.

    For DICOMs this still contains informative pixel/header patterns; for png/jpg we also use bytes.
    Features are standardized later using train statistics.
    """
    try:
        with open(path, "rb") as f:
            b = f.read()
        if not b:
            return np.zeros(12, dtype=np.float32)
        arr = np.frombuffer(b, dtype=np.uint8).astype(np.float32)

        mean = arr.mean()
        std = arr.std() + 1e-6
        mn = arr.min()
        mx = arr.max()
        q10, q50, q90 = np.quantile(arr, [0.1, 0.5, 0.9])
        hist, _ = np.histogram(arr, bins=[0, 64, 128, 192, 256], density=True)
        feat = np.array(
            [
                mean,
                std,
                mn,
                mx,
                q10,
                q50,
                q90,
                hist[0],
                hist[1],
                hist[2],
                hist[3],
                float(len(arr)),
            ],
            dtype=np.float32,
        )
        return feat
    except Exception:
        return np.zeros(12, dtype=np.float32)




## === cell 1
train_rows = []
for _, r in train_df.iterrows():
    brats = int(r.BraTS21ID)
    y = float(r.MGMT_value)
    paths = get_all_image_paths(brats, SEQ, folder="train")
    for p in paths:
        train_rows.append((brats, p, y))

train_slices = pd.DataFrame(train_rows, columns=["BraTS21ID", "path", "MGMT_value"])
train_slices = train_slices[train_slices["path"].notnull()].reset_index(drop=True)

test_rows = []
for brats in sample_sub.BraTS21ID.astype(int).tolist():
    paths = get_all_image_paths(brats, SEQ, folder="test")
    for p in paths:
        test_rows.append((brats, p))

test_slices = pd.DataFrame(test_rows, columns=["BraTS21ID", "path"])
test_slices = test_slices[test_slices["path"].notnull()].reset_index(drop=True)

train_slices = train_slices[
    train_slices["path"].map(_is_supported_image_path)
].reset_index(drop=True)
test_slices = test_slices[
    test_slices["path"].map(_is_supported_image_path)
].reset_index(drop=True)

if len(train_slices) == 0:
    raise RuntimeError(
        "No supported training slices found (.png/.jpg/.dcm). Check dataset paths."
    )

test_ids_with_slices = set(test_slices.BraTS21ID.astype(int).tolist())

train_slices.head(), test_slices.head(), len(train_slices), len(test_slices)




## === cell 2
def stratified_split_ids(ids, labels, test_size=0.2, seed=SEED):
    df = pd.DataFrame({"id": ids, "y": labels}).drop_duplicates("id")
    rng = np.random.default_rng(seed)

    tr_ids = []
    va_ids = []
    for cls, g in df.groupby("y"):
        g_ids = g["id"].to_numpy()
        rng.shuffle(g_ids)
        n_val = int(np.round(len(g_ids) * test_size))
        n_val = min(max(n_val, 1), max(len(g_ids) - 1, 1)) if len(g_ids) > 1 else 0
        va_ids.extend(g_ids[:n_val].tolist())
        tr_ids.extend(g_ids[n_val:].tolist())
    return tr_ids, va_ids


unique_ids = train_df.BraTS21ID.astype(int).tolist()
labels_by_id = train_df.set_index(train_df.BraTS21ID.astype(int))["MGMT_value"]
id_labels = [int(labels_by_id.loc[i]) for i in unique_ids]
tr_ids, va_ids = stratified_split_ids(unique_ids, id_labels, test_size=0.2, seed=SEED)

tr_slices = train_slices[train_slices.BraTS21ID.isin(tr_ids)].reset_index(drop=True)
va_slices = train_slices[train_slices.BraTS21ID.isin(va_ids)].reset_index(drop=True)

if len(tr_slices) == 0:
    tr_slices = train_slices.copy()
if len(va_slices) == 0:
    va_slices = train_slices.sample(
        n=min(len(train_slices), 1024), random_state=SEED
    ).reset_index(drop=True)

len(tr_slices), len(va_slices)




## === cell 3
def build_feature_matrix(df: pd.DataFrame, with_labels: bool):
    X = np.zeros((len(df), 12), dtype=np.float32)
    for i, p in enumerate(df["path"].values):
        X[i] = _read_bytes_features(str(p))
    if with_labels:
        y = df["MGMT_value"].values.astype(np.float32).reshape(-1, 1)
        return X, y
    return X


X_tr, y_tr = build_feature_matrix(tr_slices, with_labels=True)
X_va, y_va = build_feature_matrix(va_slices, with_labels=True)
X_te = (
    build_feature_matrix(test_slices, with_labels=False)
    if len(test_slices)
    else np.zeros((0, 12), dtype=np.float32)
)

mu = X_tr.mean(axis=0, keepdims=True)
sd = X_tr.std(axis=0, keepdims=True) + 1e-6
X_trs = (X_tr - mu) / sd
X_vas = (X_va - mu) / sd
X_tes = (X_te - mu) / sd

X_trs.shape, X_vas.shape, X_tes.shape




## === cell 4
def sigmoid(z):
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def bce_loss(p, y):
    eps = 1e-7
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()


EPOCHS = 2
LR = 1e-2
BATCH_SIZE = 256

rng = np.random.default_rng(SEED)
w = np.zeros((X_trs.shape[1], 1), dtype=np.float32)
b = np.zeros((1,), dtype=np.float32)

n = X_trs.shape[0]
for epoch in range(EPOCHS):
    idx = np.arange(n)
    rng.shuffle(idx)

    Xs = X_trs[idx]
    ys = y_tr[idx]

    for start in range(0, n, BATCH_SIZE):
        xb = Xs[start : start + BATCH_SIZE]
        yb = ys[start : start + BATCH_SIZE]

        logits = xb @ w + b
        p = sigmoid(logits)
        grad = p - yb
        dw = (xb.T @ grad) / len(xb)
        db = grad.mean(axis=0)

        w -= LR * dw.astype(np.float32)
        b -= LR * db.astype(np.float32)

    p_tr = sigmoid(X_trs @ w + b)
    p_va = sigmoid(X_vas @ w + b)
    print(
        f"epoch {epoch+1}/{EPOCHS} - loss: {bce_loss(p_tr, y_tr):.5f} - val_loss: {bce_loss(p_va, y_va):.5f}"
    )

if len(test_slices) > 0:
    test_pred_slice = sigmoid(X_tes @ w + b).reshape(-1).astype(float)
else:
    test_pred_slice = np.array([], dtype=float)



## === cell 5
if len(test_slices) > 0:
    test_slice_ids = test_slices.BraTS21ID.astype(int).values
    pred_df = pd.DataFrame({"BraTS21ID": test_slice_ids, "MGMT_value": test_pred_slice})
    pred_patient = pred_df.groupby("BraTS21ID", as_index=False)["MGMT_value"].mean()
else:
    pred_patient = pd.DataFrame({"BraTS21ID": [], "MGMT_value": []})

sub = sample_sub.copy()
sub["BraTS21ID_int"] = sub["BraTS21ID"].astype(int)

sub = sub.merge(
    pred_patient.rename(columns={"BraTS21ID": "BraTS21ID_int", "MGMT_value": "pred"}),
    on="BraTS21ID_int",
    how="left",
)

SHRINK_TO_HALF = 0.995  # 0=no shrink (original); 1=all 0.5 (AUC ~ 0.5)
raw_pred = sub["pred"].fillna(0.5).astype(float).values
sub["MGMT_value"] = ((1.0 - SHRINK_TO_HALF) * raw_pred + SHRINK_TO_HALF * 0.5).astype(
    float
)

sub["BraTS21ID"] = sub["BraTS21ID_int"].apply(lambda x: str(int(x)).zfill(5))
sub = sub[["BraTS21ID", "MGMT_value"]]

sub.to_csv("submission.csv", index=False)
sub.head(), sub.shape
