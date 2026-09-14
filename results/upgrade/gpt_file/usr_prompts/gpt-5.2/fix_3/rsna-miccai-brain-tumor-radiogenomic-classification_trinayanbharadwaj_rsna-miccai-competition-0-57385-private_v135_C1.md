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
import random
import numpy as np
import pandas as pd

import pydicom
from skimage.transform import resize

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

SEED = 42
random.seed(SEED)
np.random.seed(SEED)



## === cell 1
DATA_ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
LABELS_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

BAD_CASES = {"00109", "00123", "00709"}

IMG_PX_SIZE = 150
N_SLICES = 6  # keep same semantics: 6 representative images per case (then average)

labels_df = pd.read_csv(LABELS_CSV)
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(str).str.zfill(5)
labels_df = labels_df[~labels_df["BraTS21ID"].isin(BAD_CASES)].reset_index(drop=True)

sample_sub_df = pd.read_csv(SAMPLE_SUB)
sample_sub_df["BraTS21ID"] = sample_sub_df["BraTS21ID"].astype(str).str.zfill(5)

print("Train cases:", len(labels_df), " Test cases:", len(sample_sub_df))




## === cell 2
def _safe_dcm_pixel_array(dcm_path: str):
    """Read DICOM and return pixel_array as float32, or None if unreadable."""
    try:
        ds = pydicom.dcmread(dcm_path, force=True)
        arr = ds.pixel_array.astype(np.float32)
        if arr.ndim != 2:
            arr = np.squeeze(arr)
        if arr.ndim != 2:
            return None
        return arr
    except Exception:
        return None


def _normalize_to_unit(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    mx = float(np.max(x)) if x.size else 0.0
    if (not np.isfinite(mx)) or mx < eps:
        return np.zeros_like(x, dtype=np.float32)
    return x / mx


def _case_modality_dir(case_dir: str, modality: str = "T2w") -> str:
    return os.path.join(case_dir, modality)


def load_case_slices(
    case_dir: str, modality: str = "T2w", img_px_size: int = 150, n_slices: int = 6
):
    """
    Load up to n_slices "informative" slices from a case/modality folder.
    Fallbacks ensure exactly n_slices are returned (by padding with last/zeros).
    Output shape: (n_slices, img_px_size, img_px_size) in [0,1].
    """
    modality_dir = _case_modality_dir(case_dir, modality)
    if not os.path.isdir(modality_dir):
        return np.zeros((n_slices, img_px_size, img_px_size), dtype=np.float32)

    dcm_files = sorted(
        [
            os.path.join(modality_dir, f)
            for f in os.listdir(modality_dir)
            if f.lower().endswith(".dcm")
        ]
    )
    if len(dcm_files) == 0:
        return np.zeros((n_slices, img_px_size, img_px_size), dtype=np.float32)

    chosen = []
    for fp in dcm_files:
        arr = _safe_dcm_pixel_array(fp)
        if arr is None:
            continue
        if float(np.sum(arr)) > 100000.0:
            resized = resize(
                arr, (img_px_size, img_px_size), preserve_range=True, anti_aliasing=True
            ).astype(np.float32)
            resized = _normalize_to_unit(resized)
            if float(np.sum(resized)) > 2000.0:
                chosen.append(resized)
                if len(chosen) >= n_slices:
                    break

    if len(chosen) == 0:
        idxs = np.linspace(
            0, len(dcm_files) - 1, num=min(n_slices, len(dcm_files)), dtype=int
        )
        for idx in idxs:
            arr = _safe_dcm_pixel_array(dcm_files[int(idx)])
            if arr is None:
                continue
            resized = resize(
                arr, (img_px_size, img_px_size), preserve_range=True, anti_aliasing=True
            ).astype(np.float32)
            chosen.append(_normalize_to_unit(resized))
            if len(chosen) >= n_slices:
                break

    if len(chosen) == 0:
        chosen = [np.zeros((img_px_size, img_px_size), dtype=np.float32)]

    while len(chosen) < n_slices:
        chosen.append(chosen[-1].copy())

    return np.stack(chosen[:n_slices], axis=0).astype(np.float32)




## === cell 3
def slice_features(slice_img: np.ndarray) -> np.ndarray:
    """
    Simple deterministic per-slice features from normalized [0,1] image.
    Kept lightweight for runtime; produces a fixed-length vector.
    """
    x = slice_img.astype(np.float32, copy=False)
    x = np.clip(x, 0.0, 1.0)

    mean = float(np.mean(x))
    std = float(np.std(x))
    p10 = float(np.quantile(x, 0.10))
    p50 = float(np.quantile(x, 0.50))
    p90 = float(np.quantile(x, 0.90))

    bright = float(np.mean(x > 0.75))
    mid = float(np.mean(x > 0.50))
    nonzero = float(np.mean(x > 0.05))

    h, w = x.shape
    ch0, ch1 = int(h * 0.25), int(h * 0.75)
    cw0, cw1 = int(w * 0.25), int(w * 0.75)
    center = x[ch0:ch1, cw0:cw1]
    center_mean = float(np.mean(center))
    center_std = float(np.std(center))

    return np.array(
        [mean, std, p10, p50, p90, bright, mid, nonzero, center_mean, center_std],
        dtype=np.float32,
    )


def make_training_arrays(
    labels_dataframe: pd.DataFrame, train_dir: str, modality: str = "T2w"
):
    """
    Convert case-level labels into slice-level training data:
    X_feat shape: (num_cases*n_slices, n_features)
    y shape: (num_cases*n_slices,)
    """
    X_list = []
    y_list = []
    missing = 0

    for _, row in labels_dataframe.iterrows():
        case_id = str(row["BraTS21ID"]).zfill(5)
        case_dir = os.path.join(train_dir, case_id)
        if not os.path.isdir(case_dir):
            missing += 1
            continue

        slices = load_case_slices(
            case_dir, modality=modality, img_px_size=IMG_PX_SIZE, n_slices=N_SLICES
        )
        feats = np.stack(
            [slice_features(s) for s in slices], axis=0
        )  # (n_slices, n_features)
        yval = float(row["MGMT_value"])
        X_list.append(feats)
        y_list.append(np.full((N_SLICES,), yval, dtype=np.float32))

    if len(X_list) == 0:
        raise RuntimeError("No training data found. Check paths and dataset mounting.")

    X_feat = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0).astype(np.float32)
    print("Built training arrays:", X_feat.shape, y.shape, " missing_cases:", missing)
    return X_feat, y


X_feat, y = make_training_arrays(labels_df, TRAIN_DIR, modality="T2w")



## === cell 4
X_train, X_val, y_train, y_val = train_test_split(
    X_feat, y, test_size=0.15, random_state=SEED, stratify=y
)

model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, solver="lbfgs", random_state=SEED)),
    ]
)

model.fit(X_train, y_train)

val_proba = model.predict_proba(X_val)[:, 1]
print(
    "Val proba summary:",
    float(val_proba.min()),
    float(val_proba.max()),
    float(val_proba.mean()),
)




## === cell 5
def predict_case_probability(case_dir: str, modality: str = "T2w") -> float:
    slices = load_case_slices(
        case_dir, modality=modality, img_px_size=IMG_PX_SIZE, n_slices=N_SLICES
    )
    feats = np.stack([slice_features(s) for s in slices], axis=0)
    preds = model.predict_proba(feats)[:, 1].reshape(-1)  # per-slice probs
    p = float(np.mean(preds))
    if not np.isfinite(p):
        p = 0.5
    return float(np.clip(p, 0.0, 1.0))


test_case_ids = sample_sub_df["BraTS21ID"].tolist()
preds = []
for cid in test_case_ids:
    case_dir = os.path.join(TEST_DIR, cid)
    preds.append(predict_case_probability(case_dir, modality="T2w"))

preds = np.asarray(preds, dtype=np.float32)
print("Pred summary:", float(preds.min()), float(preds.max()), float(preds.mean()))



## === cell 6
sub_df = pd.DataFrame({"BraTS21ID": test_case_ids, "MGMT_value": preds})
sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(str).str.zfill(5)

sub_df = sample_sub_df[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(0.5).astype(float).clip(0.0, 1.0)

sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print("Wrote", sub_path, "with shape", sub_df.shape)
print(sub_df.head())



## === cell 7
assert os.path.isfile("submission.csv")
chk = pd.read_csv("submission.csv")
assert list(chk.columns) == ["BraTS21ID", "MGMT_value"]
assert chk["MGMT_value"].between(0, 1).all()
assert chk.shape[0] == sample_sub_df.shape[0]
assert chk["BraTS21ID"].astype(str).str.len().eq(5).all()
print("Submission looks valid.")
