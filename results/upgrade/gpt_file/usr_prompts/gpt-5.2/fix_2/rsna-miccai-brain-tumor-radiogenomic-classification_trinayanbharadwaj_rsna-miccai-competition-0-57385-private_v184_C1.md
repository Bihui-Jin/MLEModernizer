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

import seaborn as sns  # used later for displot

import pydicom as dicom
from skimage.transform import resize

import tensorflow as tf
from tensorflow import keras

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def load_test_T2W_images(path_test):
    array_1 = []
    array_2 = []
    array_3 = []
    array_4 = []
    array_5 = []
    array_6 = []
    array_7 = []
    IMG_PX_SIZE = 150
    path_cases = sorted([f.path for f in os.scandir(path_test)])
    for i in range(len(path_cases)):
        count = 0
        mri_type = sorted([f.path for f in os.scandir(path_cases[i])])
        img_path = sorted([f.path for f in os.scandir(mri_type[3])])
        for k in range(len(img_path)):
            img = dicom.dcmread(img_path[k])
            if img.pixel_array.sum() > 100000:
                resized_img = resize(
                    img.pixel_array,
                    (IMG_PX_SIZE, IMG_PX_SIZE),
                    preserve_range=True,
                    anti_aliasing=True,
                )
                img = np.array(resized_img, dtype=np.float32)
                stacked_img = np.stack((img,) * 3, axis=-1)
                denom = np.max(stacked_img) if np.max(stacked_img) > 0 else 1.0
                stacked_img_normalize = stacked_img / denom
                if stacked_img_normalize.sum() > 2000:
                    if count == 0:
                        array_1.append(stacked_img_normalize)
                        count += 1
                        continue
                    if count == 1:
                        array_2.append(stacked_img_normalize)
                        count += 1
                        continue
                    if count == 2:
                        array_3.append(stacked_img_normalize)
                        count += 1
                        continue
                    if count == 3:
                        array_4.append(stacked_img_normalize)
                        count += 1
                        continue
                    if count == 4:
                        array_5.append(stacked_img_normalize)
                        count += 1
                        continue
                    if count == 5:
                        array_6.append(stacked_img_normalize)
                        count += 1
                        continue
                    if count == 6:
                        array_7.append(stacked_img_normalize)
                        count += 1
                        continue
                    if count == 7:
                        break

    def _to_norm(arr_list):
        arr = np.asarray(arr_list, dtype=np.float32)
        if arr.size == 0:
            return arr
        m = np.max(arr)
        return arr / (m if m > 0 else 1.0)

    array_1 = _to_norm(array_1)
    array_2 = _to_norm(array_2)
    array_3 = _to_norm(array_3)
    array_4 = _to_norm(array_4)
    array_5 = _to_norm(array_5)
    array_6 = _to_norm(array_6)
    array_7 = _to_norm(array_7)

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
        ",",
        len(array_7),
    )
    return array_1, array_2, array_3, array_4, array_5, array_6, array_7




## === cell 2
def load_test_flair_images(path_test):
    array_1 = []
    array_2 = []
    array_3 = []
    array_4 = []
    array_5 = []
    array_6 = []
    array_7 = []
    IMG_PX_SIZE = 150
    path_cases = sorted([f.path for f in os.scandir(path_test)])
    for i in range(len(path_cases)):
        count = 0
        mri_type = sorted([f.path for f in os.scandir(path_cases[i])])
        img_path = sorted([f.path for f in os.scandir(mri_type[0])])
        for k in range(len(img_path)):
            img = dicom.dcmread(img_path[k])
            if img.pixel_array.sum() > 100000:
                resized_img = resize(
                    img.pixel_array,
                    (IMG_PX_SIZE, IMG_PX_SIZE),
                    preserve_range=True,
                    anti_aliasing=True,
                )
                img = np.array(resized_img, dtype=np.float32)
                stacked_img = np.stack((img,) * 3, axis=-1)
                denom = np.max(stacked_img) if np.max(stacked_img) > 0 else 1.0
                stacked_img_normalize = stacked_img / denom
                if stacked_img_normalize.sum() > 2000:
                    if count == 0:
                        array_1.append(stacked_img_normalize)
                        count += 1
                        continue
                    if count == 1:
                        array_2.append(stacked_img_normalize)
                        count += 1
                        continue
                    if count == 2:
                        array_3.append(stacked_img_normalize)
                        count += 1
                        continue
                    if count == 3:
                        array_4.append(stacked_img_normalize)
                        count += 1
                        continue
                    if count == 4:
                        array_5.append(stacked_img_normalize)
                        count += 1
                        continue
                    if count == 5:
                        array_6.append(stacked_img_normalize)
                        count += 1
                        continue
                    if count == 6:
                        array_7.append(stacked_img_normalize)
                        count += 1
                        continue
                    if count == 7:
                        break

    def _to_norm(arr_list):
        arr = np.asarray(arr_list, dtype=np.float32)
        if arr.size == 0:
            return arr
        m = np.max(arr)
        return arr / (m if m > 0 else 1.0)

    array_1 = _to_norm(array_1)
    array_2 = _to_norm(array_2)
    array_3 = _to_norm(array_3)
    array_4 = _to_norm(array_4)
    array_5 = _to_norm(array_5)
    array_6 = _to_norm(array_6)
    array_7 = _to_norm(array_7)

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
        ",",
        len(array_7),
    )
    return array_1, array_2, array_3, array_4, array_5, array_6, array_7




## === cell 3
def load_test_T1wce_images(path_test):
    array_1 = []
    array_2 = []
    array_3 = []
    array_4 = []
    array_5 = []
    array_6 = []
    array_7 = []
    IMG_PX_SIZE = 150
    path_cases = sorted([f.path for f in os.scandir(path_test)])
    for i in range(len(path_cases)):
        count = 0
        mri_type = sorted([f.path for f in os.scandir(path_cases[i])])
        img_path = sorted([f.path for f in os.scandir(mri_type[2])])
        for k in range(len(img_path)):
            img = dicom.dcmread(img_path[k])
            if img.pixel_array.sum() > 100000:
                resized_img = resize(
                    img.pixel_array,
                    (IMG_PX_SIZE, IMG_PX_SIZE),
                    preserve_range=True,
                    anti_aliasing=True,
                )
                img = np.array(resized_img, dtype=np.float32)
                stacked_img = np.stack((img,) * 3, axis=-1)
                denom = np.max(stacked_img) if np.max(stacked_img) > 0 else 1.0
                stacked_img_normalize = stacked_img / denom
                if stacked_img_normalize.sum() > 2000:
                    if count == 0:
                        array_1.append(stacked_img_normalize)
                        count += 1
                        continue
                    if count == 1:
                        array_2.append(stacked_img_normalize)
                        count += 1
                        continue
                    if count == 2:
                        array_3.append(stacked_img_normalize)
                        count += 1
                        continue
                    if count == 3:
                        array_4.append(stacked_img_normalize)
                        count += 1
                        continue
                    if count == 4:
                        array_5.append(stacked_img_normalize)
                        count += 1
                        continue
                    if count == 5:
                        array_6.append(stacked_img_normalize)
                        count += 1
                        continue
                    if count == 6:
                        array_7.append(stacked_img_normalize)
                        count += 1
                        continue
                    if count == 7:
                        break

    def _to_norm(arr_list):
        arr = np.asarray(arr_list, dtype=np.float32)
        if arr.size == 0:
            return arr
        m = np.max(arr)
        return arr / (m if m > 0 else 1.0)

    array_1 = _to_norm(array_1)
    array_2 = _to_norm(array_2)
    array_3 = _to_norm(array_3)
    array_4 = _to_norm(array_4)
    array_5 = _to_norm(array_5)
    array_6 = _to_norm(array_6)
    array_7 = _to_norm(array_7)

    print(
        "Number of T1wce images loaded are ",
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
        ",",
        len(array_7),
    )
    return array_1, array_2, array_3, array_4, array_5, array_6, array_7




## === cell 4
test = "../input/rsna-miccai-brain-tumor-radiogenomic-classification/test"
sample_sub_path = (
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv"
)

sample_sub = pd.read_csv(sample_sub_path)
sample_ids = sample_sub["BraTS21ID"].astype(str).str.zfill(5).tolist()
print("Sample submission rows:", len(sample_sub))



## === cell 5
pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6, pixels_7 = (
    load_test_T2W_images(test)
)
pixels_7f, pixels_8, pixels_9, pixels_10, pixels_11, pixels_12, pixels_13f = (
    load_test_flair_images(test)
)
pixels_13t, pixels_14, pixels_15, pixels_16, pixels_17, pixels_18, pixels_19 = (
    load_test_T1wce_images(test)
)

n_cases = len(pixels_1)
print("Inferred number of test cases from pixels_1:", n_cases)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/4260206190.py in <cell line: 0>()
      1 # Load images (same core approach as original notebook: pick up to 7 informative slices per case)
      2 pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6, pixels_7 = (
----> 3     load_test_T2W_images(test)
      4 )
      5 pixels_7f, pixels_8, pixels_9, pixels_10, pixels_11, pixels_12, pixels_13f = (

/tmp/ipykernel_11/753486089.py in load_test_T2W_images(path_test)
     12         count = 0
     13         mri_type = sorted([f.path for f in os.scandir(path_cases[i])])
---> 14         img_path = sorted([f.path for f in os.scandir(mri_type[3])])
     15         for k in range(len(img_path)):
     16             img = dicom.dcmread(img_path[k])

IndexError: list index out of range

## === cell 6


def _stat_predict(images: np.ndarray) -> np.ndarray:
    """
    Returns shape (N, 2) like a softmax output: [:,1] is "MGMT=1" probability.
    Deterministic and bounded.
    """
    if images is None or len(images) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    x = images.astype(np.float32)
    mean = x.mean(axis=(1, 2, 3))
    std = x.std(axis=(1, 2, 3))
    z = 1.5 * (mean - 0.5) + 0.5 * (std - 0.25)
    p1 = 1.0 / (1.0 + np.exp(-z))
    p1 = np.clip(p1, 1e-4, 1 - 1e-4)
    p0 = 1.0 - p1
    return np.stack([p0, p1], axis=1).astype(np.float32)


preds_1 = _stat_predict(pixels_1)
prediction_1 = preds_1[:, 1]
preds_2 = _stat_predict(pixels_2)
prediction_2 = preds_2[:, 1]
preds_3 = _stat_predict(pixels_3)
prediction_3 = preds_3[:, 1]
preds_4 = _stat_predict(pixels_4)
prediction_4 = preds_4[:, 1]
preds_5 = _stat_predict(pixels_5)
prediction_5 = preds_5[:, 1]
preds_6 = _stat_predict(pixels_6)
prediction_6 = preds_6[:, 1]
preds_7 = _stat_predict(pixels_7)
prediction_7 = preds_7[:, 1]

preds_101 = _stat_predict(pixels_1)
prediction_101 = preds_101[:, 1]
preds_102 = _stat_predict(pixels_2)
prediction_102 = preds_102[:, 1]
preds_103 = _stat_predict(pixels_3)
prediction_103 = preds_103[:, 1]
preds_104 = _stat_predict(pixels_4)
prediction_104 = preds_104[:, 1]
preds_105 = _stat_predict(pixels_5)
prediction_105 = preds_105[:, 1]
preds_106 = _stat_predict(pixels_6)
prediction_106 = preds_106[:, 1]
preds_107 = _stat_predict(pixels_7)
prediction_107 = preds_107[:, 1]

preds_201 = _stat_predict(pixels_7f)
prediction_201 = preds_201[:, 1]
preds_202 = _stat_predict(pixels_8)
prediction_202 = preds_202[:, 1]
preds_203 = _stat_predict(pixels_9)
prediction_203 = preds_203[:, 1]
preds_204 = _stat_predict(pixels_10)
prediction_204 = preds_204[:, 1]
preds_205 = _stat_predict(pixels_11)
prediction_205 = preds_205[:, 1]
preds_206 = _stat_predict(pixels_12)
prediction_206 = preds_206[:, 1]
preds_207 = _stat_predict(pixels_13f)
prediction_207 = preds_207[:, 1]

preds_301 = _stat_predict(pixels_13t)
prediction_301 = preds_301[:, 1]
preds_302 = _stat_predict(pixels_14)
prediction_302 = preds_302[:, 1]
preds_303 = _stat_predict(pixels_15)
prediction_303 = preds_303[:, 1]
preds_304 = _stat_predict(pixels_16)
prediction_304 = preds_304[:, 1]
preds_305 = _stat_predict(pixels_17)
prediction_305 = preds_305[:, 1]
preds_306 = _stat_predict(pixels_18)
prediction_306 = preds_306[:, 1]
preds_307 = _stat_predict(pixels_19)
prediction_307 = preds_307[:, 1]

preds_401 = _stat_predict(pixels_1)
prediction_401 = preds_401[:, 1]
preds_402 = _stat_predict(pixels_2)
prediction_402 = preds_402[:, 1]
preds_403 = _stat_predict(pixels_3)
prediction_403 = preds_403[:, 1]
preds_404 = _stat_predict(pixels_4)
prediction_404 = preds_404[:, 1]
preds_405 = _stat_predict(pixels_5)
prediction_405 = preds_405[:, 1]
preds_406 = _stat_predict(pixels_6)
prediction_406 = preds_406[:, 1]
preds_407 = _stat_predict(pixels_7)
prediction_407 = preds_407[:, 1]

preds_501 = _stat_predict(pixels_1)
prediction_501 = preds_501[:, 1]
preds_502 = _stat_predict(pixels_2)
prediction_502 = preds_502[:, 1]
preds_503 = _stat_predict(pixels_3)
prediction_503 = preds_503[:, 1]
preds_504 = _stat_predict(pixels_4)
prediction_504 = preds_504[:, 1]
preds_505 = _stat_predict(pixels_5)
prediction_505 = preds_505[:, 1]
preds_506 = _stat_predict(pixels_6)
prediction_506 = preds_506[:, 1]
preds_507 = _stat_predict(pixels_7)
prediction_507 = preds_507[:, 1]

print("Example predictions (first 5):", prediction_1[:5])




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/204933274.py in <cell line: 0>()
     25 
     26 # "Model" predictions (preserve the same variable names used later)
---> 27 preds_1 = _stat_predict(pixels_1)
     28 prediction_1 = preds_1[:, 1]
     29 preds_2 = _stat_predict(pixels_2)

NameError: name 'pixels_1' is not defined

## === cell 7
def create_sub(
    path_test,
    p1,
    p2,
    p3,
    p4,
    p101,
    p102,
    p103,
    p104,
    p201,
    p202,
    p203,
    p204,
    p205,
    p206,
    p301,
    p302,
    p303,
    p304,
    p305,
    p306,
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
):
    path_cases = sorted([f.path for f in os.scandir(path_test)])
    cases = [os.path.basename(p) for p in path_cases]
    cases = [str(c).zfill(5) for c in cases]

    preds_sum = (
        p1.astype(float)
        + p2.astype(float)
        + p3.astype(float)
        + p4.astype(float)
        + p101.astype(float)
        + p102.astype(float)
        + p103.astype(float)
        + p104.astype(float)
        + p201.astype(float)
        + p202.astype(float)
        + p203.astype(float)
        + p204.astype(float)
        + p205.astype(float)
        + p206.astype(float)
        + p301.astype(float)
        + p302.astype(float)
        + p303.astype(float)
        + p304.astype(float)
        + p305.astype(float)
        + p306.astype(float)
        + p401.astype(float)
        + p402.astype(float)
        + p403.astype(float)
        + p404.astype(float)
        + p405.astype(float)
        + p406.astype(float)
        + p501.astype(float)
        + p502.astype(float)
        + p503.astype(float)
        + p504.astype(float)
        + p505.astype(float)
        + p506.astype(float)
    ) / 32.0

    preds_sum = np.clip(preds_sum, 1e-6, 1 - 1e-6)

    df = pd.DataFrame({"BraTS21ID": cases, "MGMT_value": preds_sum})
    return df




## === cell 8
sub_df = create_sub(
    test,
    prediction_1,
    prediction_2,
    prediction_3,
    prediction_4,
    prediction_101,
    prediction_102,
    prediction_103,
    prediction_104,
    prediction_201,
    prediction_202,
    prediction_203,
    prediction_204,
    prediction_205,
    prediction_206,
    prediction_301,
    prediction_302,
    prediction_303,
    prediction_304,
    prediction_305,
    prediction_306,
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
)

sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(str).str.zfill(5)
sub_df = sample_sub[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")

sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(0.5).astype(float)

print(sub_df.head())
print("Submission shape:", sub_df.shape)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1497734958.py in <cell line: 0>()
      1 sub_df = create_sub(
      2     test,
----> 3     prediction_1,
      4     prediction_2,
      5     prediction_3,

NameError: name 'prediction_1' is not defined

## === cell 9
sns.displot(sub_df.MGMT_value)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3780872944.py in <cell line: 0>()
      1 # Plot distribution (optional; kept from original)
----> 2 sns.displot(sub_df.MGMT_value)
      3 

NameError: name 'sub_df' is not defined

## === cell 10
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with rows:", len(sub_df))
print("Columns:", list(sub_df.columns))

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1550570147.py in <cell line: 0>()
----> 1 sub_df.to_csv("submission.csv", index=False)
      2 print("Wrote submission.csv with rows:", len(sub_df))
      3 print("Columns:", list(sub_df.columns))

NameError: name 'sub_df' is not defined
