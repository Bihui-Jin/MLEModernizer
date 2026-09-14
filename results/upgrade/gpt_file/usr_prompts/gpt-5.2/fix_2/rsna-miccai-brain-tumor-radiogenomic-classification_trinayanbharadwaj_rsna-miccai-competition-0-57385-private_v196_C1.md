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
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import SimpleITK as sitk

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
LABELS_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

print("DATA_ROOT exists:", os.path.exists(DATA_ROOT))
print(
    "TRAIN_DIR exists:",
    os.path.exists(TRAIN_DIR),
    "TEST_DIR exists:",
    os.path.exists(TEST_DIR),
)




## === cell 1
def _read_series_middle_slice(series_dir: str, img_px_size: int = 150):
    """
    Read a DICOM series directory (contains many .dcm files).
    Returns a resized middle slice as float32, or None if read fails.
    """
    try:
        reader = sitk.ImageSeriesReader()
        file_names = reader.GetGDCMSeriesFileNames(series_dir)
        if not file_names:
            return None
        reader.SetFileNames(file_names)
        img3d = reader.Execute()  # typically (x,y,z) in sitk; array will be (z,y,x)
        arr = sitk.GetArrayFromImage(img3d).astype(np.float32)  # (z, y, x)
        if arr.ndim != 3 or arr.shape[0] < 1:
            return None
        mid = arr.shape[0] // 2
        sl = arr[mid]

        sl = sl - np.min(sl)
        mx = np.max(sl)
        if mx > 0:
            sl = sl / mx

        sl_img = sitk.GetImageFromArray(sl)  # 2D
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize((img_px_size, img_px_size))
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetOutputSpacing(
            [sl_img.GetSize()[0] / img_px_size, sl_img.GetSize()[1] / img_px_size]
        )
        resampled = resampler.Execute(sl_img)
        out = sitk.GetArrayFromImage(resampled).astype(np.float32)  # (y, x)
        return out
    except Exception:
        return None


def _slice_stats(sl: np.ndarray):
    """Compute stable low-dimensional stats from a 2D slice."""
    if sl is None:
        return np.array([np.nan] * 10, dtype=np.float32)

    v = sl.ravel()
    p = np.percentile(v, [1, 5, 25, 50, 75, 95, 99]).astype(np.float32)
    mean = np.mean(v).astype(np.float32)
    std = np.std(v).astype(np.float32)
    frac_nonzero = (np.mean(v > 0.05)).astype(np.float32)
    return np.concatenate(
        [np.array([mean, std, frac_nonzero], dtype=np.float32), p], axis=0
    )


def extract_subject_features(subject_dir: str, img_px_size: int = 150):
    """
    Extract features from each modality folder within a subject directory.
    Core idea preserved: use MRI-derived information to build a probability per subject.
    """
    modalities = ["FLAIR", "T1w", "T1wCE", "T2w"]
    feats = []
    for mod in modalities:
        series_dir = os.path.join(subject_dir, mod)
        sl = _read_series_middle_slice(series_dir, img_px_size=img_px_size)
        feats.append(_slice_stats(sl))
    feats = np.concatenate(feats, axis=0)  # 4 * 10 = 40 dims
    return feats




## === cell 2
labels = pd.read_csv(LABELS_CSV)
labels["BraTS21ID"] = labels["BraTS21ID"].astype(str).str.zfill(5)

bad_cases = {"00109", "00123", "00709"}
labels = labels[~labels["BraTS21ID"].isin(bad_cases)].reset_index(drop=True)

print("Train labels:", labels.shape, "Pos rate:", labels["MGMT_value"].mean())

train_ids = sorted([d.name for d in os.scandir(TRAIN_DIR) if d.is_dir()])
test_ids = sorted([d.name for d in os.scandir(TEST_DIR) if d.is_dir()])
train_ids = [tid for tid in train_ids if tid not in bad_cases]

print("Train folders:", len(train_ids), "Test folders:", len(test_ids))

labels = labels[labels["BraTS21ID"].isin(set(train_ids))].reset_index(drop=True)
print("Labels after disk-intersection:", labels.shape)



## === cell 3
X_train = np.zeros((len(labels), 40), dtype=np.float32)
y_train = labels["MGMT_value"].values.astype(np.int64)

for i, sid in enumerate(labels["BraTS21ID"].values):
    subj_dir = os.path.join(TRAIN_DIR, sid)
    X_train[i] = extract_subject_features(subj_dir, img_px_size=150)
    if (i + 1) % 50 == 0 or i == 0:
        print(f"Extracted train features: {i+1}/{len(labels)}")

X_test = np.zeros((len(test_ids), 40), dtype=np.float32)
for i, sid in enumerate(test_ids):
    subj_dir = os.path.join(TEST_DIR, sid)
    X_test[i] = extract_subject_features(subj_dir, img_px_size=150)
    if (i + 1) % 20 == 0 or i == 0:
        print(f"Extracted test features: {i+1}/{len(test_ids)}")

col_means = np.nanmean(X_train, axis=0)
inds = np.where(np.isnan(X_train))
X_train[inds] = np.take(col_means, inds[1])

inds = np.where(np.isnan(X_test))
X_test[inds] = np.take(col_means, inds[1])

print("Feature shapes:", X_train.shape, X_test.shape)



## === cell 4
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

oof = np.zeros(len(y_train), dtype=np.float32)
test_pred = np.zeros(len(test_ids), dtype=np.float32)

for fold, (tr, va) in enumerate(skf.split(X_train, y_train), 1):
    model = LogisticRegression(
        solver="liblinear",
        C=1.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=2000,
    )
    model.fit(X_train[tr], y_train[tr])
    oof[va] = model.predict_proba(X_train[va])[:, 1]
    test_pred += model.predict_proba(X_test)[:, 1] / skf.n_splits
    fold_auc = roc_auc_score(y_train[va], oof[va])
    print(f"Fold {fold} AUC: {fold_auc:.4f}")

full_auc = roc_auc_score(y_train, oof)
print("OOF AUC:", full_auc)

test_pred = np.clip(test_pred, 0.0, 1.0)



## === cell 5
sample = pd.read_csv(SAMPLE_SUB)
sample["BraTS21ID"] = sample["BraTS21ID"].astype(str).str.zfill(5)

pred_map = {sid: float(p) for sid, p in zip(test_ids, test_pred)}
sub_df = sample.copy()
sub_df["MGMT_value"] = sub_df["BraTS21ID"].map(pred_map)

sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(float(np.mean(test_pred)))

sub_df = sub_df[["BraTS21ID", "MGMT_value"]]
sub_df["MGMT_value"] = sub_df["MGMT_value"].astype(float)

print(sub_df.head())
print(
    "Submission rows:", len(sub_df), "Missing preds:", sub_df["MGMT_value"].isna().sum()
)



## === cell 6
sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print("Wrote:", sub_path, "size:", os.path.getsize(sub_path), "bytes")
