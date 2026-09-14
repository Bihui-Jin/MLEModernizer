# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

# 5. Target score

-1.0

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import pydicom as dicom

try:
    import seaborn as sns
except Exception:
    sns = None

try:
    from skimage.transform import resize as sk_resize
except Exception:
    sk_resize = None

try:
    import cv2
except Exception:
    cv2 = None


def resize2d(img2d: np.ndarray, out_hw):
    """Resize 2D image to out_hw=(H,W). Uses skimage if available else cv2."""
    h, w = out_hw
    if sk_resize is not None:
        return sk_resize(img2d, (h, w), preserve_range=True, anti_aliasing=True).astype(
            np.float32
        )
    if cv2 is None:
        raise ImportError(
            "Neither skimage.transform.resize nor cv2 is available for resizing."
        )
    return cv2.resize(img2d.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)




## === cell 1
DATA_ROOT = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
LABELS_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

assert os.path.isdir(TRAIN_DIR), f"Train dir not found: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Test dir not found: {TEST_DIR}"
assert os.path.isfile(LABELS_CSV), f"Labels file not found: {LABELS_CSV}"
assert os.path.isfile(SAMPLE_SUB), f"Sample submission not found: {SAMPLE_SUB}"

labels_df = pd.read_csv(LABELS_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(str).str.zfill(5)
sample_df["BraTS21ID"] = sample_df["BraTS21ID"].astype(str).str.zfill(5)

labels_df.head(), sample_df.head()




## === cell 2
def list_case_dirs(path_root):
    """Return sorted list of case directory paths (only directories)."""
    case_dirs = [e.path for e in os.scandir(path_root) if e.is_dir()]
    case_dirs = sorted(case_dirs, key=lambda p: int(os.path.basename(p)))
    return case_dirs


def load_case_slice_stack(case_dir, sequence="T2w", img_px_size=150, max_slices=6):
    """
    Load up to max_slices representative slices from a given sequence folder,
    normalized to [0,1] and stacked to 3 channels (H,W,3).
    """
    seq_dir = os.path.join(case_dir, sequence)
    if not os.path.isdir(seq_dir):
        return []

    dcm_paths = [
        e.path
        for e in os.scandir(seq_dir)
        if e.is_file() and e.name.lower().endswith(".dcm")
    ]
    dcm_paths = sorted(dcm_paths, key=lambda p: os.path.basename(p))

    imgs = []
    for p in dcm_paths:
        try:
            ds = dicom.dcmread(p, stop_before_pixels=False, force=True)
            arr = ds.pixel_array.astype(np.float32)
        except Exception:
            continue

        if np.nansum(arr) <= 100000:
            continue

        arr_r = resize2d(arr, (img_px_size, img_px_size))
        mx = float(np.nanmax(arr_r))
        if not np.isfinite(mx) or mx <= 0:
            continue
        arr_n = (arr_r / mx).astype(np.float32)

        if np.nansum(arr_n) <= 3000:
            continue

        stacked = np.stack([arr_n, arr_n, arr_n], axis=-1)  # (H,W,3)
        imgs.append(stacked)

        if len(imgs) >= max_slices:
            break

    return imgs


def extract_case_features(case_dir, sequence="T2w"):
    """
    Minimal features to keep runtime low and avoid missing external pretrained models.
    Uses up to 6 slices from the given sequence and computes summary statistics.
    """
    imgs = load_case_slice_stack(
        case_dir, sequence=sequence, img_px_size=150, max_slices=6
    )
    if len(imgs) == 0:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    vol = np.stack([im[..., 0] for im in imgs], axis=0).astype(np.float32)
    mean = float(vol.mean())
    std = float(vol.std())
    p10 = float(np.percentile(vol, 10))
    p50 = float(np.percentile(vol, 50))
    p90 = float(np.percentile(vol, 90))
    return np.array([mean, std, p10, p50, p90], dtype=np.float32)




## === cell 3
bad_ids = {"00109", "00123", "00709"}  # known problematic cases (per competition note)

train_case_dirs = list_case_dirs(TRAIN_DIR)
train_ids = [os.path.basename(p).zfill(5) for p in train_case_dirs]
train_map = {cid: p for cid, p in zip(train_ids, train_case_dirs)}

labels_df_use = labels_df[~labels_df["BraTS21ID"].isin(bad_ids)].copy()
labels_df_use = labels_df_use[
    labels_df_use["BraTS21ID"].isin(train_map.keys())
].reset_index(drop=True)

X_list = []
y_list = []
for cid, y in zip(
    labels_df_use["BraTS21ID"].values, labels_df_use["MGMT_value"].values
):
    feats = extract_case_features(train_map[cid], sequence="T2w")
    X_list.append(feats)
    y_list.append(int(y))

X_train = np.vstack(X_list).astype(np.float32)
y_train = np.array(y_list).astype(np.float32)

X_train.shape, y_train.mean()



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2391403291.py in <cell line: 0>()
      2 bad_ids = {"00109", "00123", "00709"}  # known problematic cases (per competition note)
      3 
----> 4 train_case_dirs = list_case_dirs(TRAIN_DIR)
      5 train_ids = [os.path.basename(p).zfill(5) for p in train_case_dirs]
      6 train_map = {cid: p for cid, p in zip(train_ids, train_case_dirs)}

/tmp/ipykernel_11/2410958810.py in list_case_dirs(path_root)
      3     case_dirs = [e.path for e in os.scandir(path_root) if e.is_dir()]
      4     # Sort by folder name as int if possible
----> 5     case_dirs = sorted(case_dirs, key=lambda p: int(os.path.basename(p)))
      6     return case_dirs
      7 

/tmp/ipykernel_11/2410958810.py in <lambda>(p)
      3     case_dirs = [e.path for e in os.scandir(path_root) if e.is_dir()]
      4     # Sort by folder name as int if possible
----> 5     case_dirs = sorted(case_dirs, key=lambda p: int(os.path.basename(p)))
      6     return case_dirs
      7 

ValueError: invalid literal for int() with base 10: 'train'

## === cell 4
try:
    from sklearn.linear_model import LogisticRegression
except Exception as e:
    raise ImportError(
        "scikit-learn is required but not available in this environment."
    ) from e

clf = LogisticRegression(
    solver="lbfgs", max_iter=500, C=1.0, class_weight=None, random_state=42
)
clf.fit(X_train, y_train)

train_proba = clf.predict_proba(X_train)[:, 1]
float(train_proba.min()), float(train_proba.max()), float(train_proba.mean())



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/617179504.py in <cell line: 0>()
     11     solver="lbfgs", max_iter=500, C=1.0, class_weight=None, random_state=42
     12 )
---> 13 clf.fit(X_train, y_train)
     14 
     15 # Sanity check: predicted probabilities on train

NameError: name 'X_train' is not defined

## === cell 5
test_case_dirs = list_case_dirs(TEST_DIR)
test_ids = [os.path.basename(p).zfill(5) for p in test_case_dirs]

X_test_list = []
for p in test_case_dirs:
    X_test_list.append(extract_case_features(p, sequence="T2w"))

X_test = np.vstack(X_test_list).astype(np.float32)
test_proba = clf.predict_proba(X_test)[:, 1].astype(np.float32)

len(test_ids), X_test.shape, test_proba.shape




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3899544825.py in <cell line: 0>()
      1 # Extract test features in the same way and predict probabilities
----> 2 test_case_dirs = list_case_dirs(TEST_DIR)
      3 test_ids = [os.path.basename(p).zfill(5) for p in test_case_dirs]
      4 
      5 X_test_list = []

/tmp/ipykernel_11/2410958810.py in list_case_dirs(path_root)
      3     case_dirs = [e.path for e in os.scandir(path_root) if e.is_dir()]
      4     # Sort by folder name as int if possible
----> 5     case_dirs = sorted(case_dirs, key=lambda p: int(os.path.basename(p)))
      6     return case_dirs
      7 

/tmp/ipykernel_11/2410958810.py in <lambda>(p)
      3     case_dirs = [e.path for e in os.scandir(path_root) if e.is_dir()]
      4     # Sort by folder name as int if possible
----> 5     case_dirs = sorted(case_dirs, key=lambda p: int(os.path.basename(p)))
      6     return case_dirs
      7 

ValueError: invalid literal for int() with base 10: 'test'

## === cell 6
def create_sub_from_ids(ids, preds):
    sub = pd.DataFrame(
        {
            "BraTS21ID": pd.Series(ids, dtype=str).str.zfill(5),
            "MGMT_value": preds.astype(float),
        }
    )
    return sub


sub_df = create_sub_from_ids(test_ids, test_proba)

sub_df = sample_df[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
sub_df["MGMT_value"] = (
    sub_df["MGMT_value"].fillna(float(np.mean(test_proba))).clip(0.0, 1.0)
)

sub_df.head(), sub_df.shape



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1490190367.py in <cell line: 0>()
     10 
     11 
---> 12 sub_df = create_sub_from_ids(test_ids, test_proba)
     13 
     14 # Align order to sample_submission if possible (safer)

NameError: name 'test_ids' is not defined

## === cell 7
if sns is not None:
    _ = sns.displot(sub_df["MGMT_value"])



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3300579092.py in <cell line: 0>()
      1 # Optional: quick distribution plot if seaborn is available (not required)
      2 if sns is not None:
----> 3     _ = sns.displot(sub_df["MGMT_value"])
      4 

NameError: name 'sub_df' is not defined

## === cell 8
sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)

assert os.path.isfile(sub_path) and sub_path.endswith(".csv")
assert list(sub_df.columns) == ["BraTS21ID", "MGMT_value"]
print(f"Wrote {sub_path} with shape {sub_df.shape}")
print(sub_df.head(10).to_string(index=False))

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1162431193.py in <cell line: 0>()
      1 # Write submission
      2 sub_path = "submission.csv"
----> 3 sub_df.to_csv(sub_path, index=False)
      4 
      5 # Basic validation

NameError: name 'sub_df' is not defined
