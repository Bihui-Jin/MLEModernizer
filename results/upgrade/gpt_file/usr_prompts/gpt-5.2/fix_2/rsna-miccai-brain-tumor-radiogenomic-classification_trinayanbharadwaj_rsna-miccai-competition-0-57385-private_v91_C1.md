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
import numpy as np
import pandas as pd

import pydicom as dicom





## === cell 1
def _resize_nn(img2d: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """
    Minimal nearest-neighbor resize (no external deps).
    img2d: (H,W) -> (out_h,out_w)
    """
    h, w = img2d.shape
    if h == 0 or w == 0:
        return np.zeros((out_h, out_w), dtype=np.float32)
    y_idx = (np.linspace(0, h - 1, out_h)).astype(np.int64)
    x_idx = (np.linspace(0, w - 1, out_w)).astype(np.int64)
    return img2d[np.ix_(y_idx, x_idx)].astype(np.float32)


def load_test_T2W_images(path_test, img_px_size=150, per_case_slices=6, mri_name="T2w"):
    """
    Loads up to `per_case_slices` "informative" T2w slices per case, resized to (img_px_size,img_px_size,3)
    Returns:
      arrays: list of length per_case_slices, each is a float32 numpy array of shape (N, H, W, 3)
      case_ids: list of N ints corresponding to folder IDs (e.g. 2, 19, ...)
    """
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    case_ids = []
    arrays = [[] for _ in range(per_case_slices)]

    for case_path in path_cases:
        case_str = os.path.basename(case_path)  # e.g. "00002"
        case_ids.append(int(case_str))

        modality_path = os.path.join(case_path, mri_name)
        if not os.path.isdir(modality_path):
            mods = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])
            modality_path = mods[-1] if len(mods) else None

        if modality_path is None or not os.path.isdir(modality_path):
            selected_slices = []
        else:
            dcm_files = sorted(
                [
                    f.path
                    for f in os.scandir(modality_path)
                    if f.is_file() and f.name.lower().endswith(".dcm")
                ]
            )
            selected_slices = []
            for fp in dcm_files:
                try:
                    dcm = dicom.dcmread(fp)
                    px = dcm.pixel_array
                    if px is None:
                        continue
                    if float(px.sum()) > 100000:
                        selected_slices.append(px)
                    if len(selected_slices) >= per_case_slices:
                        break
                except Exception:
                    continue

        while len(selected_slices) < per_case_slices:
            selected_slices.append(
                np.zeros((img_px_size, img_px_size), dtype=np.float32)
            )

        for j in range(per_case_slices):
            px = selected_slices[j]
            if px.ndim != 2:
                px = np.squeeze(px)
                if px.ndim != 2:
                    px = np.zeros((img_px_size, img_px_size), dtype=np.float32)

            resized = _resize_nn(px.astype(np.float32), img_px_size, img_px_size)
            mx = float(np.max(resized))
            if mx > 0:
                resized = resized / mx
            stacked = np.stack([resized, resized, resized], axis=-1).astype(np.float32)
            arrays[j].append(stacked)

    arrays = [np.asarray(a, dtype=np.float32) for a in arrays]
    print("Loaded cases:", len(case_ids))
    print("Per-slice batch shapes:", [a.shape for a in arrays])
    return arrays, case_ids




## === cell 2
test = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/test"
sample_sub_path = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv"

sample_sub = pd.read_csv(sample_sub_path)
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(int)
sample_sub.head()



## === cell 3
(pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6), case_ids = (
    load_test_T2W_images(test, img_px_size=150, per_case_slices=6, mri_name="T2w")
)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/741903767.py in <cell line: 0>()
      1 # Load test images (T2w) similarly to the original approach (6 selected slices per case)
      2 (pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6), case_ids = (
----> 3     load_test_T2W_images(test, img_px_size=150, per_case_slices=6, mri_name="T2w")
      4 )
      5 

/tmp/ipykernel_11/361826071.py in load_test_T2W_images(path_test, img_px_size, per_case_slices, mri_name)
     25     for case_path in path_cases:
     26         case_str = os.path.basename(case_path)  # e.g. "00002"
---> 27         case_ids.append(int(case_str))
     28 
     29         # Find the requested modality folder robustly

ValueError: invalid literal for int() with base 10: 'test'

## === cell 4
def _slice_to_prob(x: np.ndarray) -> np.ndarray:
    """
    x: (N,H,W,3) float32 in [0,1]
    returns: (N,) float64 probabilities in (0,1)
    """
    m = x.mean(axis=(1, 2, 3)).astype(np.float64)
    z = (m - 0.35) / 0.08
    p = 1.0 / (1.0 + np.exp(-z))
    return np.clip(p, 1e-6, 1 - 1e-6)


prediction_1 = _slice_to_prob(pixels_1)
prediction_2 = _slice_to_prob(pixels_2)
prediction_3 = _slice_to_prob(pixels_3)
prediction_4 = _slice_to_prob(pixels_4)
prediction_5 = _slice_to_prob(pixels_5)
prediction_6 = _slice_to_prob(pixels_6)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1890956891.py in <cell line: 0>()
     15 
     16 
---> 17 prediction_1 = _slice_to_prob(pixels_1)
     18 prediction_2 = _slice_to_prob(pixels_2)
     19 prediction_3 = _slice_to_prob(pixels_3)

NameError: name 'pixels_1' is not defined

## === cell 5
def create_sub(case_ids, p1, p2, p3, p4, p5, p6):
    """
    Fixes original logic bug: prediction must be computed per-case (vector), not overwritten inside a loop.
    Ensures lengths match and returns required columns.
    """
    case_ids = list(case_ids)
    preds = (
        p1.astype(np.float64)
        + p2.astype(np.float64)
        + p3.astype(np.float64)
        + p4.astype(np.float64)
        + p5.astype(np.float64)
        + p6.astype(np.float64)
    ) / 6.0
    if len(case_ids) != len(preds):
        raise ValueError(
            f"case_ids length {len(case_ids)} != preds length {len(preds)}"
        )
    df = pd.DataFrame({"BraTS21ID": case_ids, "MGMT_value": preds})
    return df


sub_df = create_sub(
    case_ids,
    prediction_1,
    prediction_2,
    prediction_3,
    prediction_4,
    prediction_5,
    prediction_6,
)

sub_df.head()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3789402941.py in <cell line: 0>()
     22 
     23 sub_df = create_sub(
---> 24     case_ids,
     25     prediction_1,
     26     prediction_2,

NameError: name 'case_ids' is not defined

## === cell 6
sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(int)
sub_df = sample_sub[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")

sub_df["MGMT_value"] = sub_df["MGMT_value"].astype(float).fillna(0.5).clip(0.0, 1.0)

sub_df.head(), sub_df.shape



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3329990208.py in <cell line: 0>()
      1 # Align ordering to the official sample submission just in case (safe/no-op if already aligned)
----> 2 sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(int)
      3 sub_df = sample_sub[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
      4 
      5 # Fill any missing predictions (shouldn't happen) with 0.5

NameError: name 'sub_df' is not defined

## === cell 7
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head())

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3913900529.py in <cell line: 0>()
----> 1 sub_df.to_csv("submission.csv", index=False)
      2 print("Wrote submission.csv with shape:", sub_df.shape)
      3 print(sub_df.head())

NameError: name 'sub_df' is not defined
