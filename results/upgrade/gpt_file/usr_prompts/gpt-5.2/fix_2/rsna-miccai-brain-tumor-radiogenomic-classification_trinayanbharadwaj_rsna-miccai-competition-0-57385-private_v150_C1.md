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

0.5

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'I fix the import/runtime failures by removing non-essential imports that trigger the protobuf `MessageFactory.GetPrototype` crash in this environment, and by making the `resize` call available in the image-loading functions. Since the external pre-trained model files referenced in `../input/trained-model-for-rsnamiccai/` are not present, I add a minimal fallback that produces a valid submission by using the sample submission structure and a safe constant probability when models cannot be loaded. I also fix the indentation/syntax error in `create_sub` and correct its logic so it returns one prediction per test subject with proper ID formatting and ordering. These changes keep the intended pipeline (load models → predict → average → write submission) when models exist, and otherwise ensure a valid `submission.csv` is always produced end-to-end.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

import pydicom as dicom
from skimage.transform import resize



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_ROOT = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"
TEST_DIR = os.path.join(DATA_ROOT, "test")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

assert os.path.exists(TEST_DIR), f"Test directory not found: {TEST_DIR}"
assert os.path.exists(
    SAMPLE_SUB_PATH
), f"Sample submission not found: {SAMPLE_SUB_PATH}"




## === cell 2
def load_test_T2W_images(path_test):
    array_1, array_2, array_3, array_4, array_5, array_6 = [], [], [], [], [], []
    IMG_PX_SIZE = 150
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])

    for i in range(len(path_cases)):
        count = 0
        mri_type = sorted([f.path for f in os.scandir(path_cases[i]) if f.is_dir()])
        if len(mri_type) < 4:
            continue
        img_path = sorted([f.path for f in os.scandir(mri_type[3]) if f.is_file()])

        for k in range(len(img_path)):
            img = dicom.dcmread(img_path[k])
            if img.pixel_array.sum() > 100000:
                resized_img = resize(
                    img.pixel_array,
                    (IMG_PX_SIZE, IMG_PX_SIZE),
                    preserve_range=True,
                    anti_aliasing=True,
                )
                img_np = np.array(resized_img, dtype=np.float32)
                stacked_img = np.stack((img_np,) * 3, axis=-1)

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

    def _to_norm(a):
        a = np.asarray(a, dtype=np.float32)
        if a.size == 0:
            return a
        m = np.max(a)
        return a / m if m > 0 else a

    array_1 = _to_norm(array_1)
    array_2 = _to_norm(array_2)
    array_3 = _to_norm(array_3)
    array_4 = _to_norm(array_4)
    array_5 = _to_norm(array_5)
    array_6 = _to_norm(array_6)

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
            img = dicom.dcmread(img_path[k])
            if img.pixel_array.sum() > 100000:
                resized_img = resize(
                    img.pixel_array,
                    (IMG_PX_SIZE, IMG_PX_SIZE),
                    preserve_range=True,
                    anti_aliasing=True,
                )
                img_np = np.array(resized_img, dtype=np.float32)
                stacked_img = np.stack((img_np,) * 3, axis=-1)

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

    def _to_norm(a):
        a = np.asarray(a, dtype=np.float32)
        if a.size == 0:
            return a
        m = np.max(a)
        return a / m if m > 0 else a

    array_1 = _to_norm(array_1)
    array_2 = _to_norm(array_2)
    array_3 = _to_norm(array_3)
    array_4 = _to_norm(array_4)
    array_5 = _to_norm(array_5)
    array_6 = _to_norm(array_6)

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
MODEL_ROOT = "../input/trained-model-for-rsnamiccai"
MODEL_PATHS = [
    os.path.join(MODEL_ROOT, "rsna_miccai_114_epochs_T2W_7k_imgs.h5"),
    os.path.join(MODEL_ROOT, "rsna_miccai_200_epochs_T2W_7k_imgs.h5"),
    os.path.join(MODEL_ROOT, "rsna_miccai_83_b600_T2W_7k_imgs.h5"),
    os.path.join(MODEL_ROOT, "rsna_miccai_28_b50_T2W_7k_imgs.h5"),
    os.path.join(MODEL_ROOT, "rsna_miccai_10_b600_T2w_7k_0.62auc_imgs.h5"),
    os.path.join(MODEL_ROOT, "rsna_miccai_15_b600_T2w_7k_0.74auc_imgs.h5"),
    os.path.join(MODEL_ROOT, "rsna_miccai_10_b600_flair_5.5k_0.70auc_imgs.h5"),
]


def _try_load_model(p):
    if not os.path.exists(p):
        return None
    try:
        return keras.models.load_model(p, compile=False)
    except Exception as e:
        print(f"Warning: failed to load model {p}: {e}")
        return None


models = [_try_load_model(p) for p in MODEL_PATHS]
models_present = [m for m in models if m is not None]
print(f"Found {len(models_present)}/{len(models)} models.")



## === cell 5
if len(models_present) > 0:
    pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = load_test_T2W_images(
        TEST_DIR
    )
    pixels_7, pixels_8, pixels_9, pixels_10, pixels_11, pixels_12 = (
        load_test_flair_images(TEST_DIR)
    )
else:
    pixels_1 = pixels_2 = pixels_3 = pixels_4 = pixels_5 = pixels_6 = None
    pixels_7 = pixels_8 = pixels_9 = pixels_10 = pixels_11 = pixels_12 = None




## === cell 6
def _predict_class1_prob(model, x):
    preds = model.predict(x, verbose=0)
    preds = np.asarray(preds)
    if preds.ndim == 2 and preds.shape[1] >= 2:
        return preds[:, 1].astype(np.float32)
    if preds.ndim == 2 and preds.shape[1] == 1:
        return preds[:, 0].astype(np.float32)
    if preds.ndim == 1:
        return preds.astype(np.float32)
    raise ValueError(f"Unexpected prediction shape: {preds.shape}")




## === cell 7
def create_sub(path_test, prediction_vector):
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    ids = [os.path.basename(p) for p in path_cases]  # already zero-padded folder names

    if len(prediction_vector) != len(ids):
        raise ValueError(
            f"Prediction length ({len(prediction_vector)}) does not match number of test cases ({len(ids)})."
        )

    df = pd.DataFrame({"BraTS21ID": ids, "MGMT_value": prediction_vector.astype(float)})
    return df




## === cell 8
if len(models_present) > 0:
    model_T2 = models[0]
    model_T2_2 = models[1]
    model_T2_3 = models[2]
    model_T2_4 = models[3]
    model_T2_5 = models[4]
    model_T2_6 = models[5]
    model_T2_7 = models[6]

    required = [
        model_T2,
        model_T2_2,
        model_T2_3,
        model_T2_4,
        model_T2_5,
        model_T2_6,
        model_T2_7,
    ]
    if any(m is None for m in required):
        print(
            "Not all required models are available; falling back to constant submission."
        )
        use_constant = True
    else:
        use_constant = False

    if not use_constant:
        preds = []
        for x in [pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6]:
            preds.append(_predict_class1_prob(model_T2, x))
        for x in [pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6]:
            preds.append(_predict_class1_prob(model_T2_2, x))
        for x in [pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6]:
            preds.append(_predict_class1_prob(model_T2_3, x))
        for x in [pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6]:
            preds.append(_predict_class1_prob(model_T2_4, x))
        for x in [pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6]:
            preds.append(_predict_class1_prob(model_T2_5, x))
        for x in [pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6]:
            preds.append(_predict_class1_prob(model_T2_6, x))
        for x in [pixels_7, pixels_8, pixels_9, pixels_10, pixels_11, pixels_12]:
            preds.append(_predict_class1_prob(model_T2_7, x))

        stacked = np.stack(preds, axis=0)  # (42, N)
        final_pred = stacked.mean(axis=0)
        final_pred = np.clip(final_pred, 0.0, 1.0).astype(np.float32)
else:
    use_constant = True



## === cell 9
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
if use_constant:
    final_pred = np.full((len(sample_sub),), 0.5, dtype=np.float32)
    sub_df = sample_sub.copy()
    sub_df["MGMT_value"] = final_pred
else:
    sub_df = create_sub(TEST_DIR, final_pred)

sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(str).str.zfill(5)
sub_df["MGMT_value"] = sub_df["MGMT_value"].astype(float).clip(0.0, 1.0)

if "BraTS21ID" in sample_sub.columns and len(sample_sub) == len(sub_df):
    sub_df = (
        sub_df.set_index("BraTS21ID")
        .loc[sample_sub["BraTS21ID"].astype(str).str.zfill(5)]
        .reset_index()
    )

sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print("Wrote submission.csv with shape:", sub_df.shape)
