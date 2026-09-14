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

import pydicom as dicom
from skimage.transform import resize

import tensorflow as tf
from tensorflow import keras
from keras import layers

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
try:
    tf.random.set_seed(42)
    np.random.seed(42)
except Exception:
    pass

print("TF version:", tf.__version__)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def build_small_cnn(input_shape=(150, 150, 3)):
    inp = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    return model


model_T2 = build_small_cnn()
model_T2_2 = build_small_cnn()
model_T2_3 = build_small_cnn()
model_T2_4 = build_small_cnn()
model_T2_5 = build_small_cnn()
model_T2_6 = build_small_cnn()
model_T2_7 = build_small_cnn()




## === cell 2
def _safe_dcm_pixel_array(dcm_path):
    try:
        ds = dicom.dcmread(dcm_path, force=True)
        arr = ds.pixel_array.astype(np.float32)
        if arr.ndim != 2:
            return None
        return arr
    except Exception:
        return None


def _normalize_img(img2d):
    mx = float(np.max(img2d))
    if mx <= 0:
        return None
    img = img2d / mx
    if not np.isfinite(img).all():
        return None
    return img


def load_test_T2W_images(path_test):
    array_1, array_2, array_3, array_4, array_5, array_6 = [], [], [], [], [], []
    IMG_PX_SIZE = 150

    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    for case_path in path_cases:
        count = 0
        mri_type = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])
        if len(mri_type) < 4:
            continue

        img_dir = mri_type[3]
        img_paths = sorted([f.path for f in os.scandir(img_dir) if f.is_file()])

        for p in img_paths:
            img2d = _safe_dcm_pixel_array(p)
            if img2d is None:
                continue

            if img2d.sum() <= 100000:
                continue

            resized_img = resize(
                img2d,
                (IMG_PX_SIZE, IMG_PX_SIZE),
                preserve_range=True,
                anti_aliasing=True,
            ).astype(np.float32)
            norm2d = _normalize_img(resized_img)
            if norm2d is None:
                continue

            stacked = np.stack((norm2d,) * 3, axis=-1)
            if stacked.sum() <= 2000:
                continue

            if count == 0:
                array_1.append(stacked)
                count += 1
                continue
            if count == 1:
                array_2.append(stacked)
                count += 1
                continue
            if count == 2:
                array_3.append(stacked)
                count += 1
                continue
            if count == 3:
                array_4.append(stacked)
                count += 1
                continue
            if count == 4:
                array_5.append(stacked)
                count += 1
                continue
            if count == 5:
                array_6.append(stacked)
                count += 1
                continue
            if count >= 6:
                break

    def _to_arr(x):
        x = np.asarray(x, dtype=np.float32)
        if x.size == 0:
            return x.reshape((0, IMG_PX_SIZE, IMG_PX_SIZE, 3))
        mx = float(np.max(x))
        return x if mx <= 0 else (x / mx)

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
def load_test_flair_images(path_test):
    array_1, array_2, array_3, array_4, array_5, array_6 = [], [], [], [], [], []
    IMG_PX_SIZE = 150

    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    for case_path in path_cases:
        count = 0
        mri_type = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])
        if len(mri_type) < 1:
            continue

        img_dir = mri_type[0]
        img_paths = sorted([f.path for f in os.scandir(img_dir) if f.is_file()])

        for p in img_paths:
            img2d = _safe_dcm_pixel_array(p)
            if img2d is None:
                continue

            if img2d.sum() <= 100000:
                continue

            resized_img = resize(
                img2d,
                (IMG_PX_SIZE, IMG_PX_SIZE),
                preserve_range=True,
                anti_aliasing=True,
            ).astype(np.float32)
            norm2d = _normalize_img(resized_img)
            if norm2d is None:
                continue

            stacked = np.stack((norm2d,) * 3, axis=-1)
            if stacked.sum() <= 2000:
                continue

            if count == 0:
                array_1.append(stacked)
                count += 1
                continue
            if count == 1:
                array_2.append(stacked)
                count += 1
                continue
            if count == 2:
                array_3.append(stacked)
                count += 1
                continue
            if count == 3:
                array_4.append(stacked)
                count += 1
                continue
            if count == 4:
                array_5.append(stacked)
                count += 1
                continue
            if count == 5:
                array_6.append(stacked)
                count += 1
                continue
            if count >= 6:
                break

    def _to_arr(x):
        x = np.asarray(x, dtype=np.float32)
        if x.size == 0:
            return x.reshape((0, IMG_PX_SIZE, IMG_PX_SIZE, 3))
        mx = float(np.max(x))
        return x if mx <= 0 else (x / mx)

    array_1 = _to_arr(array_1)
    array_2 = _to_arr(array_2)
    array_3 = _to_arr(array_3)
    array_4 = _to_arr(array_4)
    array_5 = _to_arr(array_5)
    array_6 = _to_arr(array_6)

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
train = "../input/rsna-miccai-brain-tumor-radiogenomic-classification/train"
labels_csv = (
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv"
)

labels_df = pd.read_csv(labels_csv)
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(str).str.zfill(5)

bad_ids = {"00109", "00123", "00709"}
labels_df = labels_df[~labels_df["BraTS21ID"].isin(bad_ids)].reset_index(drop=True)

print("Train labels:", labels_df.shape)




## === cell 5
def load_train_slices(path_train, ids, modality_index, max_per_case=2, img_px=150):
    X, y = [], []
    for brats_id in ids:
        case_path = os.path.join(path_train, brats_id)
        if not os.path.isdir(case_path):
            continue
        mri_type = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])
        if len(mri_type) <= modality_index:
            continue
        img_dir = mri_type[modality_index]
        img_paths = sorted([f.path for f in os.scandir(img_dir) if f.is_file()])
        if not img_paths:
            continue

        take = min(max_per_case, len(img_paths))
        idxs = np.linspace(0, len(img_paths) - 1, take).astype(int)

        target = labels_df.loc[labels_df["BraTS21ID"] == brats_id, "MGMT_value"]
        if target.empty:
            continue
        target = float(target.iloc[0])

        got = 0
        for j in idxs:
            img2d = _safe_dcm_pixel_array(img_paths[j])
            if img2d is None:
                continue
            resized_img = resize(
                img2d, (img_px, img_px), preserve_range=True, anti_aliasing=True
            ).astype(np.float32)
            norm2d = _normalize_img(resized_img)
            if norm2d is None:
                continue
            stacked = np.stack((norm2d,) * 3, axis=-1).astype(np.float32)
            X.append(stacked)
            y.append(target)
            got += 1
            if got >= max_per_case:
                break

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    return X, y


all_train_ids = sorted(
    [d for d in os.listdir(train) if os.path.isdir(os.path.join(train, d))]
)
all_train_ids = [x for x in all_train_ids if x not in bad_ids]
all_train_ids = [x for x in all_train_ids if x in set(labels_df["BraTS21ID"].tolist())]

subset_n = min(160, len(all_train_ids))
train_ids_subset = all_train_ids[:subset_n]

X_t2, y_t2 = load_train_slices(
    train, train_ids_subset, modality_index=3, max_per_case=2
)
X_fl, y_fl = load_train_slices(
    train, train_ids_subset, modality_index=0, max_per_case=2
)

print("X_t2:", X_t2.shape, "y_t2:", y_t2.shape)
print("X_fl:", X_fl.shape, "y_fl:", y_fl.shape)

if len(X_t2) > 0:
    model_T2.fit(X_t2, y_t2, epochs=2, batch_size=16, verbose=0)
    model_T2_2.fit(X_t2, y_t2, epochs=2, batch_size=16, verbose=0)
    model_T2_4.fit(X_t2, y_t2, epochs=2, batch_size=16, verbose=0)
    model_T2_5.fit(X_t2, y_t2, epochs=2, batch_size=16, verbose=0)
    model_T2_6.fit(X_t2, y_t2, epochs=2, batch_size=16, verbose=0)

if len(X_fl) > 0:
    model_T2_3.fit(X_fl, y_fl, epochs=2, batch_size=16, verbose=0)
    model_T2_7.fit(X_fl, y_fl, epochs=2, batch_size=16, verbose=0)



## === cell 6
pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = load_test_T2W_images(test)
pixels_7, pixels_8, pixels_9, pixels_10, pixels_11, pixels_12 = load_test_flair_images(
    test
)


def _predict_prob(model, x):
    if x is None or len(x) == 0:
        return np.asarray([], dtype=np.float32)
    p = model.predict(x, verbose=0).reshape(-1).astype(np.float32)
    return np.clip(p, 1e-6, 1 - 1e-6)


preds_1 = _predict_prob(model_T2, pixels_1)
preds_2 = _predict_prob(model_T2, pixels_2)
preds_3 = _predict_prob(model_T2, pixels_3)
preds_4 = _predict_prob(model_T2, pixels_4)
preds_5 = _predict_prob(model_T2, pixels_5)
preds_6 = _predict_prob(model_T2, pixels_6)

preds_101 = _predict_prob(model_T2_2, pixels_1)
preds_102 = _predict_prob(model_T2_2, pixels_2)
preds_103 = _predict_prob(model_T2_2, pixels_3)
preds_104 = _predict_prob(model_T2_2, pixels_4)
preds_105 = _predict_prob(model_T2_2, pixels_5)
preds_106 = _predict_prob(model_T2_2, pixels_6)

preds_201 = _predict_prob(model_T2_3, pixels_7)
preds_202 = _predict_prob(model_T2_3, pixels_8)
preds_203 = _predict_prob(model_T2_3, pixels_9)
preds_204 = _predict_prob(model_T2_3, pixels_10)
preds_205 = _predict_prob(model_T2_3, pixels_11)
preds_206 = _predict_prob(model_T2_3, pixels_12)

preds_301 = _predict_prob(model_T2_4, pixels_1)
preds_302 = _predict_prob(model_T2_4, pixels_2)
preds_303 = _predict_prob(model_T2_4, pixels_3)
preds_304 = _predict_prob(model_T2_4, pixels_4)
preds_305 = _predict_prob(model_T2_4, pixels_5)
preds_306 = _predict_prob(model_T2_4, pixels_6)

preds_401 = _predict_prob(model_T2_5, pixels_1)
preds_402 = _predict_prob(model_T2_5, pixels_2)
preds_403 = _predict_prob(model_T2_5, pixels_3)
preds_404 = _predict_prob(model_T2_5, pixels_4)
preds_405 = _predict_prob(model_T2_5, pixels_5)
preds_406 = _predict_prob(model_T2_5, pixels_6)

preds_501 = _predict_prob(model_T2_6, pixels_1)
preds_502 = _predict_prob(model_T2_6, pixels_2)
preds_503 = _predict_prob(model_T2_6, pixels_3)
preds_504 = _predict_prob(model_T2_6, pixels_4)
preds_505 = _predict_prob(model_T2_6, pixels_5)
preds_506 = _predict_prob(model_T2_6, pixels_6)

preds_601 = _predict_prob(model_T2_7, pixels_7)
preds_602 = _predict_prob(model_T2_7, pixels_8)
preds_603 = _predict_prob(model_T2_7, pixels_9)
preds_604 = _predict_prob(model_T2_7, pixels_10)
preds_605 = _predict_prob(model_T2_7, pixels_11)
preds_606 = _predict_prob(model_T2_7, pixels_12)




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
    test_ids = sorted([f.name for f in os.scandir(path_test) if f.is_dir()])
    test_ids = [x.zfill(5) for x in test_ids]

    def _pad(a, n):
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        if len(a) == n:
            return a
        if len(a) > n:
            return a[:n]
        pad = np.full((n - len(a),), 0.5, dtype=np.float32)
        return np.concatenate([a, pad], axis=0)

    n = len(test_ids)
    parts = [
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
    ]
    parts = [_pad(p, n) for p in parts]

    prediction = np.mean(np.stack(parts, axis=1), axis=1)
    prediction = np.clip(prediction, 1e-6, 1 - 1e-6)

    df = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": prediction.astype(float)})
    return df


sub_df = create_sub(
    test,
    preds_1,
    preds_2,
    preds_3,
    preds_4,
    preds_5,
    preds_6,
    preds_101,
    preds_102,
    preds_103,
    preds_104,
    preds_105,
    preds_106,
    preds_201,
    preds_202,
    preds_203,
    preds_204,
    preds_205,
    preds_206,
    preds_301,
    preds_302,
    preds_303,
    preds_304,
    preds_305,
    preds_306,
    preds_401,
    preds_402,
    preds_403,
    preds_404,
    preds_405,
    preds_406,
    preds_501,
    preds_502,
    preds_503,
    preds_504,
    preds_505,
    preds_506,
    preds_601,
    preds_602,
    preds_603,
    preds_604,
    preds_605,
    preds_606,
)

sub_df.head(), sub_df.shape



## === cell 8
print(sub_df.describe())
print("Nulls:", sub_df.isna().sum().to_dict())
print(
    "ID example:",
    sub_df["BraTS21ID"].iloc[0],
    "MGMT_value example:",
    sub_df["MGMT_value"].iloc[0],
)



## === cell 9
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with", len(sub_df), "rows")
print(sub_df.head())

## --- ERROR in outputing the csv:
Invalid submission: Submission and answers should have the same number of rows
