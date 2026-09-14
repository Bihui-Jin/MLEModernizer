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

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fix the import-time crash by removing the nonessential `pympler` import that triggers the protobuf `MessageFactory.GetPrototype` error in this environment. Then I make the script robust to the missing external pretrained `.h5` files by conditionally loading them if present; if not present, it fall back to a valid (score-poor but runnable) constant prediction that still produces a correctly formatted `submission.csv`. I also fix the `resize` NameError by importing it in the same cell where it is used and correct a logic bug in `create_sub` where the prediction was recomputed inside the loop but only the last value would be used for all rows. Finally, I ensure `BraTS21ID` formatting matches the sample submission (5-digit zero-padded strings) and that the output is written end-to-end.'
- What this solution (achieved 0.5) has done: 'I fix the import-time crash by avoiding the TensorFlow/protobuf incompatibility path: TensorFlow is only imported/used if the external pretrained `.h5` models are actually present, otherwise we skip TF entirely and fall back to the sample-submission template. This keeps the core inference logic intact when models are available, while ensuring the notebook always runs end-to-end in this environment. I also add small guards so the prediction code doesn’t run with empty pixel arrays (which can happen if DICOM filtering yields zero images) and ensure the submission is always correctly formatted and written as `submission.csv`. These changes are score-neutral when models are missing (still 0.5 constant), and allow higher scores only when the pretrained models exist.'
- What this solution (achieved 0.5) has done: 'Your current score (0.5 AUC) already lies within the ±10% tolerance band around the target score (-1.0) given “higher is better”, so any score changes are unnecessary and could move you away from the requested destination. To keep stability and preserve core logic, I only make minimal robustness fixes that do not change predictions in normal operation: (1) fix a bug where the attempted per-array normalization via `locals()` has no effect, and (2) ensure `create_sub` aligns predictions to test case IDs even if some cases are skipped during image loading (preventing crashes/misalignment that can invalidate submissions). The fallback behavior (constant 0.5) and submission formatting remain unchanged, ensuring a valid `submission.csv` is always produced.'

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

try:
    import cv2  # noqa: F401
except Exception:
    cv2 = None

from skimage.transform import resize  # Fixes NameError: resize is not defined

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

if models_available:
    import tensorflow as tf  # noqa: F401
    from tensorflow import keras  # noqa: F401
    from keras import layers  # noqa: F401

print("models_available =", models_available)



## === cell 1
models = []
if models_available:
    from tensorflow import keras

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
    case_ids = []  # aligned with array_1..array_6 entries
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
                        case_ids.append(os.path.basename(path_cases[i]))
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

    def _norm(arr):
        if arr.size == 0:
            return arr
        m = float(np.max(arr))
        return arr / m if m > 0 else arr

    array_1 = _norm(array_1)
    array_2 = _norm(array_2)
    array_3 = _norm(array_3)
    array_4 = _norm(array_4)
    array_5 = _norm(array_5)
    array_6 = _norm(array_6)

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
    return case_ids, array_1, array_2, array_3, array_4, array_5, array_6




## === cell 3
def load_test_flair_images(path_test):
    array_1, array_2, array_3, array_4, array_5, array_6 = [], [], [], [], [], []
    case_ids = []
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
                        case_ids.append(os.path.basename(path_cases[i]))
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

    def _norm(arr):
        if arr.size == 0:
            return arr
        m = float(np.max(arr))
        return arr / m if m > 0 else arr

    array_1 = _norm(array_1)
    array_2 = _norm(array_2)
    array_3 = _norm(array_3)
    array_4 = _norm(array_4)
    array_5 = _norm(array_5)
    array_6 = _norm(array_6)

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
    return case_ids, array_1, array_2, array_3, array_4, array_5, array_6




## === cell 4
test = "../input/rsna-miccai-brain-tumor-radiogenomic-classification/test"



## === cell 5
if models_available:
    t2_ids, pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = (
        load_test_T2W_images(test)
    )
    flair_ids, pixels_7, pixels_8, pixels_9, pixels_10, pixels_11, pixels_12 = (
        load_test_flair_images(test)
    )
else:
    t2_ids = flair_ids = None
    pixels_1 = pixels_2 = pixels_3 = pixels_4 = pixels_5 = pixels_6 = None
    pixels_7 = pixels_8 = pixels_9 = pixels_10 = pixels_11 = pixels_12 = None




## === cell 6
def _safe_predict(model, x):
    if x is None or not hasattr(x, "shape") or x.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    preds = model.predict(x, verbose=0)
    return preds[:, 1].astype(np.float32)


if models_available:
    prediction_1 = _safe_predict(model_T2, pixels_1)
    prediction_2 = _safe_predict(model_T2, pixels_2)
    prediction_3 = _safe_predict(model_T2, pixels_3)
    prediction_4 = _safe_predict(model_T2, pixels_4)
    prediction_5 = _safe_predict(model_T2, pixels_5)
    prediction_6 = _safe_predict(model_T2, pixels_6)

    prediction_101 = _safe_predict(model_T2_2, pixels_1)
    prediction_102 = _safe_predict(model_T2_2, pixels_2)
    prediction_103 = _safe_predict(model_T2_2, pixels_3)
    prediction_104 = _safe_predict(model_T2_2, pixels_4)
    prediction_105 = _safe_predict(model_T2_2, pixels_5)
    prediction_106 = _safe_predict(model_T2_2, pixels_6)

    prediction_201 = _safe_predict(model_T2_3, pixels_7)
    prediction_202 = _safe_predict(model_T2_3, pixels_8)
    prediction_203 = _safe_predict(model_T2_3, pixels_9)
    prediction_204 = _safe_predict(model_T2_3, pixels_10)
    prediction_205 = _safe_predict(model_T2_3, pixels_11)
    prediction_206 = _safe_predict(model_T2_3, pixels_12)

    prediction_301 = _safe_predict(model_T2_4, pixels_7)
    prediction_302 = _safe_predict(model_T2_4, pixels_8)
    prediction_303 = _safe_predict(model_T2_4, pixels_9)
    prediction_304 = _safe_predict(model_T2_4, pixels_10)
    prediction_305 = _safe_predict(model_T2_4, pixels_11)
    prediction_306 = _safe_predict(model_T2_4, pixels_12)

    prediction_401 = _safe_predict(model_T2_5, pixels_1)
    prediction_402 = _safe_predict(model_T2_5, pixels_2)
    prediction_403 = _safe_predict(model_T2_5, pixels_3)
    prediction_404 = _safe_predict(model_T2_5, pixels_4)
    prediction_405 = _safe_predict(model_T2_5, pixels_5)
    prediction_406 = _safe_predict(model_T2_5, pixels_6)

    prediction_501 = _safe_predict(model_T2_6, pixels_1)
    prediction_502 = _safe_predict(model_T2_6, pixels_2)
    prediction_503 = _safe_predict(model_T2_6, pixels_3)
    prediction_504 = _safe_predict(model_T2_6, pixels_4)
    prediction_505 = _safe_predict(model_T2_6, pixels_5)
    prediction_506 = _safe_predict(model_T2_6, pixels_6)

    prediction_601 = _safe_predict(model_T2_7, pixels_7)
    prediction_602 = _safe_predict(model_T2_7, pixels_8)
    prediction_603 = _safe_predict(model_T2_7, pixels_9)
    prediction_604 = _safe_predict(model_T2_7, pixels_10)
    prediction_605 = _safe_predict(model_T2_7, pixels_11)
    prediction_606 = _safe_predict(model_T2_7, pixels_12)




## === cell 7
def create_sub(
    path_test,
    t2_case_ids,
    flair_case_ids,
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
    all_cases = [os.path.basename(p) for p in path_cases]  # e.g. "00002"
    all_cases_set = set(all_cases)

    def _add_to_map(pred_map, case_ids, preds):
        if case_ids is None:
            return pred_map
        n = min(len(case_ids), len(preds))
        for i in range(n):
            cid = case_ids[i]
            if cid in all_cases_set:
                pred_map.setdefault(cid, []).append(float(preds[i]))
        return pred_map

    pred_map = {}
    for preds in [
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
    ]:
        pred_map = _add_to_map(pred_map, t2_case_ids, preds)

    for preds in [
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
        p601,
        p602,
        p603,
        p604,
        p605,
        p606,
    ]:
        pred_map = _add_to_map(pred_map, flair_case_ids, preds)

    final_pred = []
    for cid in all_cases:
        vals = pred_map.get(cid, None)
        if not vals:
            final_pred.append(0.5)
        else:
            final_pred.append(float(np.mean(vals)))

    df = pd.DataFrame(
        {"BraTS21ID": all_cases, "MGMT_value": np.asarray(final_pred, dtype=np.float32)}
    )
    return df




## === cell 8
if models_available:
    sub_df = create_sub(
        test,
        t2_ids,
        flair_ids,
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
