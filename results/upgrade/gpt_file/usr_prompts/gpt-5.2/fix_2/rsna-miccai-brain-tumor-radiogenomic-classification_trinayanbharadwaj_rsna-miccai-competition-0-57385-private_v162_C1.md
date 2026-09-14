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

0.56941

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.56941) has done: 'I remove/avoid the import that triggers the protobuf `MessageFactory.GetPrototype` crash by eliminating the unused `pympler` dependency and switching to `tf.keras`-only imports to keep Keras/TensorFlow compatible in this environment. Since the referenced pretrained `.h5` models are not available in your `/kaggle/input` tree, I replace that failing load step with a minimal, deterministic baseline that reads the DICOMs, extracts simple intensity statistics per subject (same image-loading core logic), trains a lightweight logistic regression, and predicts probabilities for the test set. I also fix the `resize` NameError by importing it inside the loader and ensure the submission is aligned to `sample_submission.csv` IDs and written as `submission.csv` with the exact required columns. These changes are necessary for end-to-end execution and to yield a valid AUC-scored submission (better than a constant predictor) without introducing heavy new dependencies.'

# 9. Code solution

## === cell 0
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import pydicom as dicom
from skimage.transform import resize  # fixes NameError in original loader

from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)



## === cell 1
BASE_INPUT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
if not os.path.exists(BASE_INPUT):
    BASE_INPUT = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"

TRAIN_DIR = os.path.join(BASE_INPUT, "train")
TEST_DIR = os.path.join(BASE_INPUT, "test")
LABELS_CSV = os.path.join(BASE_INPUT, "train_labels.csv")
SAMPLE_SUB_CSV = os.path.join(BASE_INPUT, "sample_submission.csv")

assert os.path.exists(TRAIN_DIR), f"Train dir not found: {TRAIN_DIR}"
assert os.path.exists(TEST_DIR), f"Test dir not found: {TEST_DIR}"
assert os.path.exists(LABELS_CSV), f"Labels not found: {LABELS_CSV}"
assert os.path.exists(SAMPLE_SUB_CSV), f"Sample submission not found: {SAMPLE_SUB_CSV}"

train_labels = pd.read_csv(LABELS_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)

assert {"BraTS21ID", "MGMT_value"}.issubset(train_labels.columns)
assert {"BraTS21ID", "MGMT_value"}.issubset(sample_sub.columns)

train_labels["BraTS21ID"] = train_labels["BraTS21ID"].astype(int)
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(int)



## === cell 2


def _sorted_subject_dirs(root_dir):
    dirs = [f.path for f in os.scandir(root_dir) if f.is_dir()]
    return sorted(dirs)


def _sorted_modality_dirs(subject_dir):
    mods = [f.path for f in os.scandir(subject_dir) if f.is_dir()]
    return sorted(mods)


def _sorted_dicom_files(modality_dir):
    files = [f.path for f in os.scandir(modality_dir) if f.is_file()]
    return sorted(files)


def _safe_normalize(img):
    img = img.astype(np.float32)
    mx = np.max(img)
    if mx <= 0:
        return img
    return img / mx


def _slice_passes(img_arr, sum_thresh=100000, norm_sum_thresh=2000):
    if img_arr is None:
        return False
    s = float(np.sum(img_arr))
    if s <= sum_thresh:
        return False
    img_n = _safe_normalize(img_arr)
    if float(np.sum(img_n)) <= norm_sum_thresh:
        return False
    return True


def _extract_subject_features(subject_dir, img_size=150, max_slices=6):
    """
    Returns a fixed-length feature vector per subject:
    For each modality: mean and std over up to max_slices selected slices (after resize+normalize),
    plus slice count actually used. 4 modalities * (mean,std,count) = 12 features.
    """
    modality_dirs = _sorted_modality_dirs(subject_dir)
    name_to_dir = {os.path.basename(p).lower(): p for p in modality_dirs}
    ordered = []
    for key in ["flair", "t1w", "t1wce", "t2w"]:
        if key in name_to_dir:
            ordered.append(name_to_dir[key])
        else:
            ordered.append(None)

    feats = []
    for mod_dir in ordered:
        if mod_dir is None or (not os.path.exists(mod_dir)):
            feats.extend([np.nan, np.nan, 0.0])
            continue

        dicom_files = _sorted_dicom_files(mod_dir)
        chosen = []
        for fp in dicom_files:
            try:
                ds = dicom.dcmread(fp)
                arr = ds.pixel_array
            except Exception:
                continue

            if _slice_passes(arr):
                arr_r = resize(
                    arr, (img_size, img_size), anti_aliasing=True, preserve_range=True
                ).astype(np.float32)
                arr_r = _safe_normalize(arr_r)
                chosen.append(arr_r)
                if len(chosen) >= max_slices:
                    break

        if len(chosen) == 0:
            feats.extend([np.nan, np.nan, 0.0])
        else:
            stack = np.stack(chosen, axis=0)
            feats.extend([float(stack.mean()), float(stack.std()), float(len(chosen))])

    return np.array(feats, dtype=np.float32)


def build_features(root_dir, subject_ids):
    X = np.zeros((len(subject_ids), 12), dtype=np.float32)
    for i, sid in enumerate(subject_ids):
        subject_dir = os.path.join(root_dir, f"{sid:05d}")
        X[i, :] = _extract_subject_features(subject_dir)
    return X




## === cell 3
bad_cases = {109, 123, 709}
train_df = train_labels[~train_labels["BraTS21ID"].isin(bad_cases)].copy()

train_ids = train_df["BraTS21ID"].values
y = train_df["MGMT_value"].values.astype(int)

X_train = build_features(TRAIN_DIR, train_ids)

clf = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        (
            "lr",
            LogisticRegression(
                max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE
            ),
        ),
    ]
)

clf.fit(X_train, y)



## === cell 4
test_ids = sample_sub["BraTS21ID"].values
X_test = build_features(TEST_DIR, test_ids)

proba = clf.predict_proba(X_test)[:, 1].astype(np.float32)

proba = np.clip(proba, 0.0, 1.0)

sub_df = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": proba})

sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(int)
sub_df = sub_df.sort_values("BraTS21ID").reset_index(drop=True)

sub_df = sub_df.set_index("BraTS21ID").loc[sample_sub["BraTS21ID"].values].reset_index()



## === cell 5
sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)

print(sub_df.head())
print(
    f"\nWrote {sub_path} with shape={sub_df.shape} and columns={list(sub_df.columns)}"
)
