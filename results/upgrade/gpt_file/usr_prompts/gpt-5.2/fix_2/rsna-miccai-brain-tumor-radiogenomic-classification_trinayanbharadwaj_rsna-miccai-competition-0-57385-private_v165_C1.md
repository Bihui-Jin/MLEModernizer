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

0.5

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'I fix the import-time crash by removing the nonessential `pympler` import that triggers the protobuf `MessageFactory.GetPrototype` error in this environment. Then I make the script robust to the missing external pretrained `.h5` files by conditionally loading them if present; if not present, it fall back to a valid (score-poor but runnable) constant prediction that still produces a correctly formatted `submission.csv`. I also fix the `resize` NameError by importing it in the same cell where it is used and correct a logic bug in `create_sub` where the prediction was recomputed inside the loop but only the last value would be used for all rows. Finally, I ensure `BraTS21ID` formatting matches the sample submission (5-digit zero-padded strings) and that the output is written end-to-end.'

# 9. Code solution

## === cell 0
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import pydicom as dicom


import tensorflow as tf
from tensorflow import keras
from keras import layers

try:
    import cv2  # noqa: F401
except Exception:
    cv2 = None

from skimage.transform import resize  # Fixes NameError: resize is not defined



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
MODEL_DIR = "../input/trained-model-for-rsnamiccai"

model_paths = [
    "rsna_miccai_114_epochs_T2W_7k_imgs.h5",
    "rsna_miccai_200_epochs_T2W_7k_imgs.h5",
    "rsna_miccai_15_b400_flair_5k_0.73auc_imgs.h5",
    "rsna_miccai_10_b600_flair_5.5k_0.67auc_imgs.h5",
    "rsna_miccai_10_b600_T2w_7k_0.62auc_imgs.h5",
    "rsna_miccai_15_b600_T2w_7k_0.74auc_imgs.h5",
    "rsna_miccai_10_b600_flair_5.5k_0.70auc_imgs.h5",
]
model_fullpaths = [os.path.join(MODEL_DIR, p) for p in model_paths]
models_available = all(os.path.exists(p) for p in model_fullpaths)

models = []
if models_available:
    for fp in model_fullpaths:
        models.append(keras.models.load_model(fp, compile=False))
    (
        model_T2,
        model_T2_2,
        model_T2_3,
        model_T2_4,
        model_T2_5,
        model_T2_6,
        model_T2_7,
    ) = models
else:
    model_T2 = model_T2_2 = model_T2_3 = model_T2_4 = model_T2_5 = model_T2_6 = (
        model_T2_7
    ) = None
    print("WARNING: Pretrained model files not found under:", MODEL_DIR)
    print("Will create a valid submission using a constant prediction fallback.")




## === cell 2
def load_test_T2W_images(path_test):
    array_1, array_2, array_3, array_4, array_5, array_6 = [], [], [], [], [], []
    IMG_PX_SIZE = 150

    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    for i in range(len(path_cases)):
        count = 0
        mri_type = sorted([f.path for f in os.scandir(path_cases[i]) if f.is_dir()])

        if len(mri_type) <= 3:
            continue

        img_path = sorted([f.path for f in os.scandir(mri_type[3]) if f.is_file()])
        for k in range(len(img_path)):
            try:
                img = dicom.dcmread(img_path[k])
                px = img.pixel_array
            except Exception:
                continue

            if px.sum() > 100000:
                resized_img = resize(
                    px,
                    (IMG_PX_SIZE, IMG_PX_SIZE),
                    anti_aliasing=True,
                    preserve_range=True,
                )
                img_arr = np.array(resized_img, dtype=np.float32)
                stacked_img = np.stack((img_arr,) * 3, axis=-1)
                mx = np.max(stacked_img)
                if mx <= 0:
                    continue
                stacked_img_normalize = stacked_img / mx

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
                        break

    array_1 = np.asarray(array_1, dtype=np.float32)
    array_2 = np.asarray(array_2, dtype=np.float32)
    array_3 = np.asarray(array_3, dtype=np.float32)
    array_4 = np.asarray(array_4, dtype=np.float32)
    array_5 = np.asarray(array_5, dtype=np.float32)
    array_6 = np.asarray(array_6, dtype=np.float32)

    for arr_name, arr in [
        ("array_1", array_1),
        ("array_2", array_2),
        ("array_3", array_3),
        ("array_4", array_4),
        ("array_5", array_5),
        ("array_6", array_6),
    ]:
        if arr.size > 0:
            m = np.max(arr)
            if m > 0:
                locals()[arr_name] = arr / m

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




## === cell 3
def load_test_flair_images(path_test):
    array_1, array_2, array_3, array_4, array_5, array_6 = [], [], [], [], [], []
    IMG_PX_SIZE = 150

    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    for i in range(len(path_cases)):
        count = 0
        mri_type = sorted([f.path for f in os.scandir(path_cases[i]) if f.is_dir()])

        if len(mri_type) < 1:
            continue

        img_path = sorted([f.path for f in os.scandir(mri_type[0]) if f.is_file()])
        for k in range(len(img_path)):
            try:
                img = dicom.dcmread(img_path[k])
                px = img.pixel_array
            except Exception:
                continue

            if px.sum() > 100000:
                resized_img = resize(
                    px,
                    (IMG_PX_SIZE, IMG_PX_SIZE),
                    anti_aliasing=True,
                    preserve_range=True,
                )
                img_arr = np.array(resized_img, dtype=np.float32)
                stacked_img = np.stack((img_arr,) * 3, axis=-1)
                mx = np.max(stacked_img)
                if mx <= 0:
                    continue
                stacked_img_normalize = stacked_img / mx

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
                        break

    array_1 = np.asarray(array_1, dtype=np.float32)
    array_2 = np.asarray(array_2, dtype=np.float32)
    array_3 = np.asarray(array_3, dtype=np.float32)
    array_4 = np.asarray(array_4, dtype=np.float32)
    array_5 = np.asarray(array_5, dtype=np.float32)
    array_6 = np.asarray(array_6, dtype=np.float32)

    for arr_name, arr in [
        ("array_1", array_1),
        ("array_2", array_2),
        ("array_3", array_3),
        ("array_4", array_4),
        ("array_5", array_5),
        ("array_6", array_6),
    ]:
        if arr.size > 0:
            m = np.max(arr)
            if m > 0:
                locals()[arr_name] = arr / m

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




## === cell 4
test = "../input/rsna-miccai-brain-tumor-radiogenomic-classification/test"



## === cell 5
if models_available:
    pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = load_test_T2W_images(
        test
    )
    pixels_7, pixels_8, pixels_9, pixels_10, pixels_11, pixels_12 = (
        load_test_flair_images(test)
    )
else:
    pixels_1 = pixels_2 = pixels_3 = pixels_4 = pixels_5 = pixels_6 = None
    pixels_7 = pixels_8 = pixels_9 = pixels_10 = pixels_11 = pixels_12 = None



## === cell 6
if models_available:
    preds_1 = model_T2.predict(pixels_1, verbose=0)
    prediction_1 = preds_1[:, 1]
    preds_2 = model_T2.predict(pixels_2, verbose=0)
    prediction_2 = preds_2[:, 1]
    preds_3 = model_T2.predict(pixels_3, verbose=0)
    prediction_3 = preds_3[:, 1]
    preds_4 = model_T2.predict(pixels_4, verbose=0)
    prediction_4 = preds_4[:, 1]
    preds_5 = model_T2.predict(pixels_5, verbose=0)
    prediction_5 = preds_5[:, 1]
    preds_6 = model_T2.predict(pixels_6, verbose=0)
    prediction_6 = preds_6[:, 1]

    preds_101 = model_T2_2.predict(pixels_1, verbose=0)
    prediction_101 = preds_101[:, 1]
    preds_102 = model_T2_2.predict(pixels_2, verbose=0)
    prediction_102 = preds_102[:, 1]
    preds_103 = model_T2_2.predict(pixels_3, verbose=0)
    prediction_103 = preds_103[:, 1]
    preds_104 = model_T2_2.predict(pixels_4, verbose=0)
    prediction_104 = preds_104[:, 1]
    preds_105 = model_T2_2.predict(pixels_5, verbose=0)
    prediction_105 = preds_105[:, 1]
    preds_106 = model_T2_2.predict(pixels_6, verbose=0)
    prediction_106 = preds_106[:, 1]

    preds_201 = model_T2_3.predict(pixels_7, verbose=0)
    prediction_201 = preds_201[:, 1]
    preds_202 = model_T2_3.predict(pixels_8, verbose=0)
    prediction_202 = preds_202[:, 1]
    preds_203 = model_T2_3.predict(pixels_9, verbose=0)
    prediction_203 = preds_203[:, 1]
    preds_204 = model_T2_3.predict(pixels_10, verbose=0)
    prediction_204 = preds_204[:, 1]
    preds_205 = model_T2_3.predict(pixels_11, verbose=0)
    prediction_205 = preds_205[:, 1]
    preds_206 = model_T2_3.predict(pixels_12, verbose=0)
    prediction_206 = preds_206[:, 1]

    preds_301 = model_T2_4.predict(pixels_7, verbose=0)
    prediction_301 = preds_301[:, 1]
    preds_302 = model_T2_4.predict(pixels_8, verbose=0)
    prediction_302 = preds_302[:, 1]
    preds_303 = model_T2_4.predict(pixels_9, verbose=0)
    prediction_303 = preds_303[:, 1]
    preds_304 = model_T2_4.predict(pixels_10, verbose=0)
    prediction_304 = preds_304[:, 1]
    preds_305 = model_T2_4.predict(pixels_11, verbose=0)
    prediction_305 = preds_305[:, 1]
    preds_306 = model_T2_4.predict(pixels_12, verbose=0)
    prediction_306 = preds_306[:, 1]

    preds_401 = model_T2_5.predict(pixels_1, verbose=0)
    prediction_401 = preds_401[:, 1]
    preds_402 = model_T2_5.predict(pixels_2, verbose=0)
    prediction_402 = preds_402[:, 1]
    preds_403 = model_T2_5.predict(pixels_3, verbose=0)
    prediction_403 = preds_403[:, 1]
    preds_404 = model_T2_5.predict(pixels_4, verbose=0)
    prediction_404 = preds_404[:, 1]
    preds_405 = model_T2_5.predict(pixels_5, verbose=0)
    prediction_405 = preds_405[:, 1]
    preds_406 = model_T2_5.predict(pixels_6, verbose=0)
    prediction_406 = preds_406[:, 1]

    preds_501 = model_T2_6.predict(pixels_1, verbose=0)
    prediction_501 = preds_501[:, 1]
    preds_502 = model_T2_6.predict(pixels_2, verbose=0)
    prediction_502 = preds_502[:, 1]
    preds_503 = model_T2_6.predict(pixels_3, verbose=0)
    prediction_503 = preds_503[:, 1]
    preds_504 = model_T2_6.predict(pixels_4, verbose=0)
    prediction_504 = preds_504[:, 1]
    preds_505 = model_T2_6.predict(pixels_5, verbose=0)
    prediction_505 = preds_505[:, 1]
    preds_506 = model_T2_6.predict(pixels_6, verbose=0)
    prediction_506 = preds_506[:, 1]

    preds_601 = model_T2_7.predict(pixels_7, verbose=0)
    prediction_601 = preds_601[:, 1]
    preds_602 = model_T2_7.predict(pixels_8, verbose=0)
    prediction_602 = preds_602[:, 1]
    preds_603 = model_T2_7.predict(pixels_9, verbose=0)
    prediction_603 = preds_603[:, 1]
    preds_604 = model_T2_7.predict(pixels_10, verbose=0)
    prediction_604 = preds_604[:, 1]
    preds_605 = model_T2_7.predict(pixels_11, verbose=0)
    prediction_605 = preds_605[:, 1]
    preds_606 = model_T2_7.predict(pixels_12, verbose=0)
    prediction_606 = preds_606[:, 1]




## === cell 7
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
    p601,
    p602,
    p603,
    p604,
    p605,
    p606,
):
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    cases = [
        os.path.basename(p) for p in path_cases
    ]  # already zero-padded like "00002"

    prediction = (
        p1.astype(float)
        + p2.astype(float)
        + p3.astype(float)
        + p4.astype(float)
        + p5.astype(float)
        + p6.astype(float)
        + p101.astype(float)
        + p102.astype(float)
        + p103.astype(float)
        + p104.astype(float)
        + p105.astype(float)
        + p106.astype(float)
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
        + p601.astype(float)
        + p602.astype(float)
        + p603.astype(float)
        + p604.astype(float)
        + p605.astype(float)
        + p606.astype(float)
    ) / 42.0

    if len(prediction) != len(cases):
        raise ValueError(
            f"Prediction length {len(prediction)} does not match number of test cases {len(cases)}"
        )

    df = pd.DataFrame({"BraTS21ID": cases, "MGMT_value": prediction})
    return df




## === cell 8
if models_available:
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
        prediction_601,
        prediction_602,
        prediction_603,
        prediction_604,
        prediction_605,
        prediction_606,
    )
else:
    sample_path = "../input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv"
    sub_df = pd.read_csv(sample_path)
    sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(str).str.zfill(5)
    sub_df["MGMT_value"] = 0.5



## === cell 9
try:
    sns.displot(sub_df.MGMT_value)
    plt.show()
except Exception:
    pass



## === cell 10
sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(str).str.zfill(5)
sub_df["MGMT_value"] = sub_df["MGMT_value"].astype(float).clip(0.0, 1.0)

sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head())
