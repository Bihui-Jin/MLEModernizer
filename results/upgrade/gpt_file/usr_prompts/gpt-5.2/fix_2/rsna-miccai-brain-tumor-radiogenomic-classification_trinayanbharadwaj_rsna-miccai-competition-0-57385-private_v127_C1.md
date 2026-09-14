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

import numpy as np
import pandas as pd

import pydicom as dicom
from skimage.transform import resize

warnings.filterwarnings("ignore")

np.random.seed(42)



## === cell 1
try:
    import tensorflow as tf
    from tensorflow import keras
except Exception as e:
    tf = None
    keras = None

MODEL_DIR = "../input/trained-model-for-rsnamiccai"
MODEL_PATHS = [
    os.path.join(MODEL_DIR, "rsna_miccai_114_epochs_T2W_7k_imgs.h5"),
    os.path.join(MODEL_DIR, "rsna_miccai_200_epochs_T2W_7k_imgs.h5"),
    os.path.join(MODEL_DIR, "rsna_miccai_83_b600_T2W_7k_imgs.h5"),
    os.path.join(MODEL_DIR, "rsna_miccai_28_b50_T2W_7k_imgs.h5"),
]

models = []
if keras is not None:
    for p in MODEL_PATHS:
        if os.path.exists(p):
            models.append(keras.models.load_model(p, compile=False))

USE_PRETRAINED = len(models) == 4
print(f"Pretrained models found: {len(models)}/4 -> USE_PRETRAINED={USE_PRETRAINED}")




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
def load_test_T2W_images(path_test):
    array_1, array_2, array_3, array_4, array_5, array_6 = [], [], [], [], [], []
    IMG_PX_SIZE = 150

    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    for case_path in path_cases:
        count = 0
        mri_types = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])

        if len(mri_types) < 4:
            continue
        t2w_dir = mri_types[3]

        img_paths = sorted([f.path for f in os.scandir(t2w_dir) if f.is_file()])
        for p in img_paths:
            try:
                ds = dicom.dcmread(p)
                px = ds.pixel_array
            except Exception:
                continue

            if px.sum() > 100000:
                resized_img = resize(
                    px,
                    (IMG_PX_SIZE, IMG_PX_SIZE),
                    preserve_range=True,
                    anti_aliasing=True,
                )
                img = np.asarray(resized_img, dtype=np.float32)

                stacked_img = np.stack((img,) * 3, axis=-1)
                mx = np.max(stacked_img)
                if mx <= 0:
                    continue
                stacked_img_normalize = stacked_img / mx

                if stacked_img_normalize.sum() > 2000:
                    if count == 0:
                        array_1.append(stacked_img_normalize)
                    elif count == 1:
                        array_2.append(stacked_img_normalize)
                    elif count == 2:
                        array_3.append(stacked_img_normalize)
                    elif count == 3:
                        array_4.append(stacked_img_normalize)
                    elif count == 4:
                        array_5.append(stacked_img_normalize)
                    elif count == 5:
                        array_6.append(stacked_img_normalize)
                    count += 1
                    if count == 6:
                        break

    def _to_arr(x):
        x = np.asarray(x, dtype=np.float32)
        if x.size == 0:
            return x
        m = np.max(x)
        return x / m if m > 0 else x

    array_1 = _to_arr(array_1)
    array_2 = _to_arr(array_2)
    array_3 = _to_arr(array_3)
    array_4 = _to_arr(array_4)
    array_5 = _to_arr(array_5)
    array_6 = _to_arr(array_6)

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
test = "../input/rsna-miccai-brain-tumor-radiogenomic-classification/test"

pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = load_test_T2W_images(test)



## === cell 4


def _baseline_predict(pixels):
    """
    pixels: (N, H, W, 3) float32 in [0,1] typically
    returns: (N,) probabilities in [0,1]
    """
    if pixels is None or len(pixels) == 0:
        return np.array([], dtype=np.float32)
    x = np.asarray(pixels, dtype=np.float32)
    m = x.mean(axis=(1, 2, 3))
    p = 1.0 / (1.0 + np.exp(-8.0 * (m - 0.5)))
    return p.astype(np.float32)


if USE_PRETRAINED:
    model_T2, model_T2_2, model_T2_3, model_T2_4 = models

    def _model_probs(model, pixels):
        preds = model.predict(pixels, verbose=0)
        preds = np.asarray(preds)
        if preds.ndim == 2 and preds.shape[1] >= 2:
            return preds[:, 1].astype(np.float32)
        return preds.reshape(-1).astype(np.float32)

    prediction_1 = _model_probs(model_T2, pixels_1)
    prediction_2 = _model_probs(model_T2, pixels_2)
    prediction_3 = _model_probs(model_T2, pixels_3)
    prediction_4 = _model_probs(model_T2, pixels_4)
    prediction_5 = _model_probs(model_T2, pixels_5)
    prediction_6 = _model_probs(model_T2, pixels_6)

    prediction_101 = _model_probs(model_T2_2, pixels_1)
    prediction_102 = _model_probs(model_T2_2, pixels_2)
    prediction_103 = _model_probs(model_T2_2, pixels_3)
    prediction_104 = _model_probs(model_T2_2, pixels_4)
    prediction_105 = _model_probs(model_T2_2, pixels_5)
    prediction_106 = _model_probs(model_T2_2, pixels_6)

    prediction_201 = _model_probs(model_T2_3, pixels_1)
    prediction_202 = _model_probs(model_T2_3, pixels_2)
    prediction_203 = _model_probs(model_T2_3, pixels_3)
    prediction_204 = _model_probs(model_T2_3, pixels_4)
    prediction_205 = _model_probs(model_T2_3, pixels_5)
    prediction_206 = _model_probs(model_T2_3, pixels_6)

    prediction_301 = _model_probs(model_T2_4, pixels_1)
    prediction_302 = _model_probs(model_T2_4, pixels_2)
    prediction_303 = _model_probs(model_T2_4, pixels_3)
    prediction_304 = _model_probs(model_T2_4, pixels_4)
    prediction_305 = _model_probs(model_T2_4, pixels_5)
    prediction_306 = _model_probs(model_T2_4, pixels_6)
else:
    prediction_1 = _baseline_predict(pixels_1)
    prediction_2 = _baseline_predict(pixels_2)
    prediction_3 = _baseline_predict(pixels_3)
    prediction_4 = _baseline_predict(pixels_4)
    prediction_5 = _baseline_predict(pixels_5)
    prediction_6 = _baseline_predict(pixels_6)

    prediction_101 = prediction_1.copy()
    prediction_102 = prediction_2.copy()
    prediction_103 = prediction_3.copy()
    prediction_104 = prediction_4.copy()
    prediction_105 = prediction_5.copy()
    prediction_106 = prediction_6.copy()

    prediction_201 = prediction_1.copy()
    prediction_202 = prediction_2.copy()
    prediction_203 = prediction_3.copy()
    prediction_204 = prediction_4.copy()
    prediction_205 = prediction_5.copy()
    prediction_206 = prediction_6.copy()

    prediction_301 = prediction_1.copy()
    prediction_302 = prediction_2.copy()
    prediction_303 = prediction_3.copy()
    prediction_304 = prediction_4.copy()
    prediction_305 = prediction_5.copy()
    prediction_306 = prediction_6.copy()




## === cell 5
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
):
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    cases = [int(os.path.basename(p)) for p in path_cases]

    preds = (
        np.asarray(p1, dtype=np.float32)
        + np.asarray(p2, dtype=np.float32)
        + np.asarray(p3, dtype=np.float32)
        + np.asarray(p4, dtype=np.float32)
        + np.asarray(p5, dtype=np.float32)
        + np.asarray(p6, dtype=np.float32)
        + np.asarray(p101, dtype=np.float32)
        + np.asarray(p102, dtype=np.float32)
        + np.asarray(p103, dtype=np.float32)
        + np.asarray(p104, dtype=np.float32)
        + np.asarray(p105, dtype=np.float32)
        + np.asarray(p106, dtype=np.float32)
        + np.asarray(p201, dtype=np.float32)
        + np.asarray(p202, dtype=np.float32)
        + np.asarray(p203, dtype=np.float32)
        + np.asarray(p204, dtype=np.float32)
        + np.asarray(p205, dtype=np.float32)
        + np.asarray(p206, dtype=np.float32)
        + np.asarray(p301, dtype=np.float32)
        + np.asarray(p302, dtype=np.float32)
        + np.asarray(p303, dtype=np.float32)
        + np.asarray(p304, dtype=np.float32)
        + np.asarray(p305, dtype=np.float32)
        + np.asarray(p306, dtype=np.float32)
    ) / 24.0

    preds = np.clip(preds, 0.0, 1.0)

    df = pd.DataFrame({"BraTS21ID": cases, "MGMT_value": preds})
    return df




## === cell 6
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
)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3460916830.py in <cell line: 0>()
----> 1 sub_df = create_sub(
      2     test,
      3     prediction_1,
      4     prediction_2,
      5     prediction_3,

/tmp/ipykernel_11/2834513833.py in create_sub(path_test, p1, p2, p3, p4, p5, p6, p101, p102, p103, p104, p105, p106, p201, p202, p203, p204, p205, p206, p301, p302, p303, p304, p305, p306)
     30     # and build BraTS21ID list in the same sorted folder order to maintain alignment.
     31     path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
---> 32     cases = [int(os.path.basename(p)) for p in path_cases]
     33 
     34     preds = (

/tmp/ipykernel_11/2834513833.py in <listcomp>(.0)
     30     # and build BraTS21ID list in the same sorted folder order to maintain alignment.
     31     path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
---> 32     cases = [int(os.path.basename(p)) for p in path_cases]
     33 
     34     preds = (

ValueError: invalid literal for int() with base 10: 'test'

## === cell 7
sample_path = (
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv"
)
sample = pd.read_csv(sample_path)

sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(int)
sample["BraTS21ID"] = sample["BraTS21ID"].astype(int)

sub_df = sample[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")

sub_df["MGMT_value"] = sub_df["MGMT_value"].astype(float).fillna(0.5).clip(0.0, 1.0)

sub_df.head()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1955482179.py in <cell line: 0>()
      5 sample = pd.read_csv(sample_path)
      6 
----> 7 sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(int)
      8 sample["BraTS21ID"] = sample["BraTS21ID"].astype(int)
      9 

NameError: name 'sub_df' is not defined

## === cell 8
print(sub_df["MGMT_value"].describe())



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1157087518.py in <cell line: 0>()
      1 # Optional diagnostic: distribution (no seaborn dependency required)
----> 2 print(sub_df["MGMT_value"].describe())
      3 

NameError: name 'sub_df' is not defined

## === cell 9
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head())

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3913900529.py in <cell line: 0>()
----> 1 sub_df.to_csv("submission.csv", index=False)
      2 print("Wrote submission.csv with shape:", sub_df.shape)
      3 print(sub_df.head())

NameError: name 'sub_df' is not defined
