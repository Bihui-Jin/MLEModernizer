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

try:
    import seaborn as sns
except Exception:
    sns = None

import pydicom as dicom

import tensorflow as tf
from tensorflow import keras
from keras import layers

try:
    from skimage.transform import resize as _sk_resize
except Exception:
    _sk_resize = None


def resize(img2d, out_hw):
    """Resize 2D numpy array to (H, W). Uses skimage if available; otherwise uses TF bilinear."""
    out_h, out_w = out_hw
    if _sk_resize is not None:
        return _sk_resize(
            img2d, (out_h, out_w), preserve_range=True, anti_aliasing=False
        ).astype(np.float32)
    x = tf.convert_to_tensor(img2d, dtype=tf.float32)[None, ..., None]  # (1,H,W,1)
    x = tf.image.resize(x, (out_h, out_w), method="bilinear")
    return x[0, ..., 0].numpy()


np.random.seed(42)
tf.random.set_seed(42)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def _build_stub_model(input_size):
    inp = keras.Input(shape=(input_size, input_size, 3))
    x = layers.Rescaling(1.0)(inp)
    x = layers.Conv2D(8, 3, activation="relu", padding="same")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(16, 3, activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(2, activation="softmax")(x)
    model = keras.Model(inp, out)
    return model


model_1 = _build_stub_model(299)  # for FLAIR stacks in this script
model_2 = _build_stub_model(150)  # for T2W stacks in this script




## === cell 2
def _safe_normalize_img(x):
    x = x.astype(np.float32)
    mx = float(np.max(x))
    if not np.isfinite(mx) or mx <= 0.0:
        return x
    return x / mx


def _to_numpy_4d(list_imgs):
    """Convert list of HxWx3 arrays to (N,H,W,3) float32 array; empty -> (0,H,W,3) not possible without H,W."""
    if len(list_imgs) == 0:
        return np.zeros((0, 0, 0, 3), dtype=np.float32)
    return np.stack(list_imgs, axis=0).astype(np.float32)


def load_test_flair_images(path_test):
    array_1, array_2, array_3, array_4, array_5, array_6 = [], [], [], [], [], []
    IMG_PX_SIZE = 299

    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    for case_path in path_cases:
        count = 0
        mri_type = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])

        if len(mri_type) < 4:
            continue
        img_path = sorted(
            [
                f.path
                for f in os.scandir(mri_type[3])
                if f.is_file() and f.name.lower().endswith(".dcm")
            ]
        )

        for p in img_path:
            ds = dicom.dcmread(p)
            px = ds.pixel_array
            if px.sum() > 100000:
                img = resize(px, (IMG_PX_SIZE, IMG_PX_SIZE))
                stacked_img = np.stack((img,) * 3, axis=-1)
                stacked_img_normalize = _safe_normalize_img(stacked_img)
                if stacked_img_normalize.sum() > 5000:
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

    a1, a2, a3, a4, a5, a6 = map(
        _to_numpy_4d, [array_1, array_2, array_3, array_4, array_5, array_6]
    )

    def _safe_global_norm(arr):
        mx = float(np.max(arr)) if arr.size else 0.0
        return arr / mx if mx > 0 else arr

    a1, a2, a3, a4, a5, a6 = map(_safe_global_norm, [a1, a2, a3, a4, a5, a6])

    print(
        "Number of flair images loaded are ",
        len(a1),
        ",",
        len(a2),
        ",",
        len(a3),
        ",",
        len(a4),
        ",",
        len(a5),
        "and",
        len(a6),
    )
    return a1, a2, a3, a4, a5, a6


def load_test_T1W_images(path_test):
    array = []
    IMG_PX_SIZE = 150
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])

    for case_path in path_cases:
        mri_type = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])
        if len(mri_type) < 2:
            continue
        img_path = sorted(
            [
                f.path
                for f in os.scandir(mri_type[1])
                if f.is_file() and f.name.lower().endswith(".dcm")
            ]
        )
        for p in img_path:
            ds = dicom.dcmread(p)
            px = ds.pixel_array
            if px.sum() > 100000:
                img = resize(px, (IMG_PX_SIZE, IMG_PX_SIZE))
                stacked_img = np.stack((img,) * 3, axis=-1)
                stacked_img_normalize = _safe_normalize_img(stacked_img)
                if stacked_img_normalize.sum() > 10000:
                    array.append(stacked_img_normalize)
                    break

    arr = _to_numpy_4d(array)
    mx = float(np.max(arr)) if arr.size else 0.0
    arr = arr / mx if mx > 0 else arr
    print("Number of T1W images loaded are ", len(arr))
    return arr


def load_test_T1wCE_images(path_test):
    array = []
    IMG_PX_SIZE = 150
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])

    for case_path in path_cases:
        mri_type = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])
        if len(mri_type) < 3:
            continue
        img_path = sorted(
            [
                f.path
                for f in os.scandir(mri_type[2])
                if f.is_file() and f.name.lower().endswith(".dcm")
            ]
        )
        for p in img_path:
            ds = dicom.dcmread(p)
            px = ds.pixel_array
            if px.sum() > 100000:
                img = resize(px, (IMG_PX_SIZE, IMG_PX_SIZE))
                stacked_img = np.stack((img,) * 3, axis=-1)
                stacked_img_normalize = _safe_normalize_img(stacked_img)
                if stacked_img_normalize.sum() > 10000:
                    array.append(stacked_img_normalize)
                    break

    arr = _to_numpy_4d(array)
    mx = float(np.max(arr)) if arr.size else 0.0
    arr = arr / mx if mx > 0 else arr
    print("Number of T1wCE images loaded are ", len(arr))
    return arr


def load_test_T2W_images(path_test):
    array_1, array_2, array_3, array_4, array_5, array_6 = [], [], [], [], [], []
    IMG_PX_SIZE = 150

    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    for case_path in path_cases:
        count = 0
        mri_type = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])
        if len(mri_type) < 4:
            continue
        img_path = sorted(
            [
                f.path
                for f in os.scandir(mri_type[3])
                if f.is_file() and f.name.lower().endswith(".dcm")
            ]
        )

        for p in img_path:
            ds = dicom.dcmread(p)
            px = ds.pixel_array
            if px.sum() > 100000:
                img = resize(px, (IMG_PX_SIZE, IMG_PX_SIZE))
                stacked_img = np.stack((img,) * 3, axis=-1)
                stacked_img_normalize = _safe_normalize_img(stacked_img)
                if stacked_img_normalize.sum() > 3000:
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

    a1, a2, a3, a4, a5, a6 = map(
        _to_numpy_4d, [array_1, array_2, array_3, array_4, array_5, array_6]
    )

    def _safe_global_norm(arr):
        mx = float(np.max(arr)) if arr.size else 0.0
        return arr / mx if mx > 0 else arr

    a1, a2, a3, a4, a5, a6 = map(_safe_global_norm, [a1, a2, a3, a4, a5, a6])

    print(
        "Number of T2W images loaded are ",
        len(a1),
        ",",
        len(a2),
        ",",
        len(a3),
        ",",
        len(a4),
        ",",
        len(a5),
        "and",
        len(a6),
    )
    return a1, a2, a3, a4, a5, a6




## === cell 3
test = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/test"



## === cell 4
pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = load_test_flair_images(
    test
)
pixels_7, pixels_8, pixels_9, pixels_10, pixels_11, pixels_12 = load_test_T2W_images(
    test
)

n_cases = len(sorted([f.path for f in os.scandir(test) if f.is_dir()]))
if len(pixels_1) != n_cases or len(pixels_7) != n_cases:
    print(
        f"Warning: expected {n_cases} cases, got pixels_1={len(pixels_1)}, pixels_7={len(pixels_7)}"
    )




## === cell 5
def _predict_or_default(model, x, n_expected, input_size):
    if x is None or x.ndim != 4 or len(x) != n_expected or x.shape[1] != input_size:
        return np.full((n_expected,), 0.5, dtype=np.float32)
    preds = model.predict(x, verbose=0)
    if preds.ndim == 2 and preds.shape[1] >= 2:
        return preds[:, 1].astype(np.float32)
    return np.full((n_expected,), 0.5, dtype=np.float32)


prediction_1 = _predict_or_default(model_1, pixels_1, n_cases, 299)
prediction_7 = _predict_or_default(model_2, pixels_7, n_cases, 150)




## === cell 6
def create_sub(path_test, p1, p7):
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    cases = [os.path.basename(p) for p in path_cases]

    p1 = np.asarray(p1, dtype=np.float32).reshape(-1)
    p7 = np.asarray(p7, dtype=np.float32).reshape(-1)
    if len(p1) != len(cases) or len(p7) != len(cases):
        raise ValueError(
            f"Prediction length mismatch: len(cases)={len(cases)}, len(p1)={len(p1)}, len(p7)={len(p7)}"
        )

    prediction = (p1 + p7) / 2.0
    prediction = np.clip(prediction, 0.0, 1.0)

    df = pd.DataFrame({"BraTS21ID": cases, "MGMT_value": prediction})
    return df


sub_df = create_sub(test, prediction_1, prediction_7)



## === cell 7
if sns is not None:
    try:
        sns.displot(sub_df.MGMT_value)
    except Exception:
        pass

sub_df.head()



## === cell 8
sub_df = sub_df[["BraTS21ID", "MGMT_value"]]
sub_df.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head())

## --- ERROR in outputing the csv:
Invalid submission: Submission and answers should have the same number of rows
