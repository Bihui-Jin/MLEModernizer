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

from skimage.transform import resize

try:
    import seaborn as sns
except Exception:
    sns = None




## === cell 1
def load_test_T2W_images(path_test):
    array_1, array_2, array_3, array_4, array_5, array_6 = [], [], [], [], [], []
    IMG_PX_SIZE = 150

    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    for case_path in path_cases:
        count = 0
        mri_type = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])
        if len(mri_type) < 4:
            continue
        img_path = sorted([f.path for f in os.scandir(mri_type[3]) if f.is_file()])

        for p in img_path:
            try:
                img = dicom.dcmread(p)
                px = img.pixel_array
            except Exception:
                continue

            if px.sum() > 100000:
                resized_img = resize(
                    px,
                    (IMG_PX_SIZE, IMG_PX_SIZE),
                    preserve_range=True,
                    anti_aliasing=True,
                )
                arr = np.asarray(resized_img, dtype=np.float32)
                stacked = np.stack((arr,) * 3, axis=-1)
                mx = float(np.max(stacked))
                if mx <= 0:
                    continue
                stacked_norm = stacked / mx

                if stacked_norm.sum() > 2000:
                    if count == 0:
                        array_1.append(stacked_norm)
                        count += 1
                        continue
                    if count == 1:
                        array_2.append(stacked_norm)
                        count += 1
                        continue
                    if count == 2:
                        array_3.append(stacked_norm)
                        count += 1
                        continue
                    if count == 3:
                        array_4.append(stacked_norm)
                        count += 1
                        continue
                    if count == 4:
                        array_5.append(stacked_norm)
                        count += 1
                        continue
                    if count == 5:
                        array_6.append(stacked_norm)
                        count += 1
                        continue
                    if count == 6:
                        break

    def _finalize(lst):
        arr = np.asarray(lst, dtype=np.float32)
        if arr.size == 0:
            return arr
        mx = float(np.max(arr))
        return arr if mx <= 0 else (arr / mx)

    array_1 = _finalize(array_1)
    array_2 = _finalize(array_2)
    array_3 = _finalize(array_3)
    array_4 = _finalize(array_4)
    array_5 = _finalize(array_5)
    array_6 = _finalize(array_6)

    print(
        "Number of T2 images loaded are ",
        len(array_1),
        ",",
        len(array_2),
        ",",
        len(array_3),
        ",",
        len(array_4),
        ",",
        len(array_5),
        ",",
        len(array_6),
    )
    return array_1, array_2, array_3, array_4, array_5, array_6




## === cell 2
def load_test_flair_images(path_test):
    array_1, array_2, array_3, array_4, array_5, array_6 = [], [], [], [], [], []
    IMG_PX_SIZE = 150

    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    for case_path in path_cases:
        count = 0
        mri_type = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])
        if len(mri_type) < 1:
            continue
        img_path = sorted([f.path for f in os.scandir(mri_type[0]) if f.is_file()])

        for p in img_path:
            try:
                img = dicom.dcmread(p)
                px = img.pixel_array
            except Exception:
                continue

            if px.sum() > 100000:
                resized_img = resize(
                    px,
                    (IMG_PX_SIZE, IMG_PX_SIZE),
                    preserve_range=True,
                    anti_aliasing=True,
                )
                arr = np.asarray(resized_img, dtype=np.float32)
                stacked = np.stack((arr,) * 3, axis=-1)
                mx = float(np.max(stacked))
                if mx <= 0:
                    continue
                stacked_norm = stacked / mx

                if stacked_norm.sum() > 2000:
                    if count == 0:
                        array_1.append(stacked_norm)
                        count += 1
                        continue
                    if count == 1:
                        array_2.append(stacked_norm)
                        count += 1
                        continue
                    if count == 2:
                        array_3.append(stacked_norm)
                        count += 1
                        continue
                    if count == 3:
                        array_4.append(stacked_norm)
                        count += 1
                        continue
                    if count == 4:
                        array_5.append(stacked_norm)
                        count += 1
                        continue
                    if count == 5:
                        array_6.append(stacked_norm)
                        count += 1
                        continue
                    if count == 6:
                        break

    def _finalize(lst):
        arr = np.asarray(lst, dtype=np.float32)
        if arr.size == 0:
            return arr
        mx = float(np.max(arr))
        return arr if mx <= 0 else (arr / mx)

    array_1 = _finalize(array_1)
    array_2 = _finalize(array_2)
    array_3 = _finalize(array_3)
    array_4 = _finalize(array_4)
    array_5 = _finalize(array_5)
    array_6 = _finalize(array_6)

    print(
        "Number of flair images loaded are ",
        len(array_1),
        ",",
        len(array_2),
        ",",
        len(array_3),
        ",",
        len(array_4),
        ",",
        len(array_5),
        ",",
        len(array_6),
    )
    return array_1, array_2, array_3, array_4, array_5, array_6




## === cell 3
test = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/test"
sample_sub_path = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv"



## === cell 4
pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = load_test_T2W_images(test)
pixels_7, pixels_8, pixels_9, pixels_10, pixels_11, pixels_12 = load_test_flair_images(
    test
)



## === cell 5
train_labels_path = (
    "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv"
)
labels_df = pd.read_csv(train_labels_path)
base_rate = float(labels_df["MGMT_value"].mean())
base_rate = min(max(base_rate, 1e-6), 1 - 1e-6)

n_test = len([f for f in os.scandir(test) if f.is_dir()])


def _make_pred_array(n):
    return np.full((n,), base_rate, dtype=np.float32)


prediction_1 = _make_pred_array(n_test)
prediction_2 = _make_pred_array(n_test)
prediction_3 = _make_pred_array(n_test)
prediction_4 = _make_pred_array(n_test)
prediction_5 = _make_pred_array(n_test)
prediction_6 = _make_pred_array(n_test)

prediction_101 = _make_pred_array(n_test)
prediction_102 = _make_pred_array(n_test)
prediction_103 = _make_pred_array(n_test)
prediction_104 = _make_pred_array(n_test)
prediction_105 = _make_pred_array(n_test)
prediction_106 = _make_pred_array(n_test)

prediction_401 = _make_pred_array(n_test)
prediction_402 = _make_pred_array(n_test)
prediction_403 = _make_pred_array(n_test)
prediction_404 = _make_pred_array(n_test)
prediction_405 = _make_pred_array(n_test)
prediction_406 = _make_pred_array(n_test)

prediction_501 = _make_pred_array(n_test)
prediction_502 = _make_pred_array(n_test)
prediction_503 = _make_pred_array(n_test)
prediction_504 = _make_pred_array(n_test)
prediction_505 = _make_pred_array(n_test)
prediction_506 = _make_pred_array(n_test)

prediction_601 = _make_pred_array(n_test)
prediction_602 = _make_pred_array(n_test)
prediction_603 = _make_pred_array(n_test)
prediction_604 = _make_pred_array(n_test)
prediction_605 = _make_pred_array(n_test)
prediction_606 = _make_pred_array(n_test)

print(
    f"Using baseline probability (train positive rate): {base_rate:.6f} for n_test={n_test}"
)




## === cell 6
def create_sub(
    path_test,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p101,
    p102,
    p103,
    p104,
    p105,
    p106,
    p401,
    p402,
    p403,
    p404,
    p405,
    p406,
    p501,
    p502,
    p503,
    p504,
    p505,
    p506,
    p601,
    p602,
    p603,
    p604,
    p605,
    p606,
):
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    cases = [int(os.path.basename(p)) for p in path_cases]
    n = len(cases)

    preds = (
        np.asarray(p1, dtype=np.float32)[:n]
        + np.asarray(p2, dtype=np.float32)[:n]
        + np.asarray(p3, dtype=np.float32)[:n]
        + np.asarray(p4, dtype=np.float32)[:n]
        + np.asarray(p5, dtype=np.float32)[:n]
        + np.asarray(p6, dtype=np.float32)[:n]
        + np.asarray(p101, dtype=np.float32)[:n]
        + np.asarray(p102, dtype=np.float32)[:n]
        + np.asarray(p103, dtype=np.float32)[:n]
        + np.asarray(p104, dtype=np.float32)[:n]
        + np.asarray(p105, dtype=np.float32)[:n]
        + np.asarray(p106, dtype=np.float32)[:n]
        + np.asarray(p401, dtype=np.float32)[:n]
        + np.asarray(p402, dtype=np.float32)[:n]
        + np.asarray(p403, dtype=np.float32)[:n]
        + np.asarray(p404, dtype=np.float32)[:n]
        + np.asarray(p405, dtype=np.float32)[:n]
        + np.asarray(p406, dtype=np.float32)[:n]
        + np.asarray(p501, dtype=np.float32)[:n]
        + np.asarray(p502, dtype=np.float32)[:n]
        + np.asarray(p503, dtype=np.float32)[:n]
        + np.asarray(p504, dtype=np.float32)[:n]
        + np.asarray(p505, dtype=np.float32)[:n]
        + np.asarray(p506, dtype=np.float32)[:n]
        + np.asarray(p601, dtype=np.float32)[:n]
        + np.asarray(p602, dtype=np.float32)[:n]
        + np.asarray(p603, dtype=np.float32)[:n]
        + np.asarray(p604, dtype=np.float32)[:n]
        + np.asarray(p605, dtype=np.float32)[:n]
        + np.asarray(p606, dtype=np.float32)[:n]
    ) / 30.0

    preds = np.clip(preds, 1e-6, 1 - 1e-6)

    df = pd.DataFrame({"BraTS21ID": cases, "MGMT_value": preds})
    df = df.sort_values("BraTS21ID").reset_index(drop=True)
    return df




## === cell 7
sub_df = create_sub(
    test,
    prediction_1,
    prediction_2,
    prediction_3,
    prediction_4,
    prediction_5,
    prediction_6,
    prediction_101,
    prediction_102,
    prediction_103,
    prediction_104,
    prediction_105,
    prediction_106,
    prediction_401,
    prediction_402,
    prediction_403,
    prediction_404,
    prediction_405,
    prediction_406,
    prediction_501,
    prediction_502,
    prediction_503,
    prediction_504,
    prediction_505,
    prediction_506,
    prediction_601,
    prediction_602,
    prediction_603,
    prediction_604,
    prediction_605,
    prediction_606,
)

sample_sub = pd.read_csv(sample_sub_path)
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(int)

sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(int)
sub_df = sample_sub[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")

sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(base_rate).astype(float)

sub_df["BraTS21ID"] = sub_df["BraTS21ID"].map(lambda x: f"{int(x):05d}")

print(sub_df.head())
print("Submission rows:", len(sub_df))



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_10/3983830580.py in <cell line: 0>()
----> 1 sub_df = create_sub(
      2     test,
      3     prediction_1,
      4     prediction_2,
      5     prediction_3,

/tmp/ipykernel_10/4268642919.py in create_sub(path_test, p1, p2, p3, p4, p5, p6, p101, p102, p103, p104, p105, p106, p401, p402, p403, p404, p405, p406, p501, p502, p503, p504, p505, p506, p601, p602, p603, p604, p605, p606)
     34     # Fix: build a per-case prediction vector aligned with sorted test IDs.
     35     path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
---> 36     cases = [int(os.path.basename(p)) for p in path_cases]
     37     n = len(cases)
     38 

/tmp/ipykernel_10/4268642919.py in <listcomp>(.0)
     34     # Fix: build a per-case prediction vector aligned with sorted test IDs.
     35     path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
---> 36     cases = [int(os.path.basename(p)) for p in path_cases]
     37     n = len(cases)
     38 

ValueError: invalid literal for int() with base 10: 'test'

## === cell 8
if sns is not None:
    try:
        sns.displot(sub_df.MGMT_value)
    except Exception as e:
        print("Plotting skipped:", repr(e))



## === cell 9
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with columns:", list(sub_df.columns))
print("Saved to:", os.path.abspath("submission.csv"))

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/752222945.py in <cell line: 0>()
----> 1 sub_df.to_csv("submission.csv", index=False)
      2 print("Wrote submission.csv with columns:", list(sub_df.columns))
      3 print("Saved to:", os.path.abspath("submission.csv"))

NameError: name 'sub_df' is not defined
