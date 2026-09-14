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

0.47412

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.47412) has done: 'I fix the runtime crash caused by an incompatibility between `pydicom` and the protobuf version in the Kaggle environment by safely deferring the `pydicom` import and providing a fallback DICOM reader using `tf.io.decode_dicom_image` (so the pipeline runs end-to-end). Next, I fix the submission row-count mismatch by ensuring predictions are generated per test case (one probability per `BraTS21ID`) instead of per selected slice, and I align/subset/reorder the final submission strictly to `sample_submission.csv`. These changes preserve your core model/training logic (same CNN, same optimizer/loss/epochs) while making inference and submission formatting correct and stable. The script write a valid `submission.csv` with exactly the required rows and columns.'

# 9. Code solution

## === cell 0
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def _try_import_pydicom():
    try:
        import pydicom as dicom  # noqa: F401

        return dicom
    except Exception as e:
        print("pydicom unavailable (will use TF DICOM decoder fallback):", repr(e))
        return None


_DICOM = _try_import_pydicom()


def _resize_img(arr2d, out_size):
    x = tf.convert_to_tensor(arr2d, dtype=tf.float32)
    x = tf.expand_dims(x, axis=-1)  # H,W,1
    x = tf.image.resize(x, (out_size, out_size), method="bilinear", antialias=True)
    x = tf.squeeze(x, axis=-1)  # H,W
    return x.numpy()


def _read_dcm_pixel_array(dcm_path):
    if _DICOM is not None:
        ds = _DICOM.dcmread(dcm_path)
        arr = ds.pixel_array.astype(np.float32)
        return arr

    b = tf.io.read_file(dcm_path)
    img = tf.io.decode_dicom_image(
        b,
        dtype=tf.uint16,
        color_dim=False,
        scale="auto",
    )
    img = tf.squeeze(img, axis=0)  # H,W,1 or H,W
    if img.shape.rank == 3:
        img = img[:, :, 0]
    return tf.cast(img, tf.float32).numpy()


def _normalize_0_1(x, eps=1e-6):
    x = x - np.min(x)
    denom = np.max(x) + eps
    return x / denom




## === cell 2
ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(ROOT, "train")
TEST_DIR = os.path.join(ROOT, "test")
LABELS_CSV = os.path.join(ROOT, "train_labels.csv")
SAMPLE_SUB = os.path.join(ROOT, "sample_submission.csv")

test = TEST_DIR




## === cell 3
def load_test_T2W_images_by_case(path_test, img_px_size=150, max_slices=6):
    cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    ids = [os.path.basename(p) for p in cases]

    X = np.zeros((len(ids), max_slices, img_px_size, img_px_size, 3), dtype=np.float32)
    mask = np.zeros(
        (len(ids), max_slices), dtype=np.float32
    )  # 1 if slice present else 0

    for i, case_path in enumerate(cases):
        mri_type = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])
        t2w_dir = None
        if len(mri_type) >= 4:
            t2w_dir = mri_type[3]
        else:
            for p in mri_type:
                if "T2w" in os.path.basename(p) or "T2W" in os.path.basename(p):
                    t2w_dir = p
                    break
        if t2w_dir is None:
            continue

        img_paths = sorted(
            [
                f.path
                for f in os.scandir(t2w_dir)
                if f.is_file() and f.name.lower().endswith(".dcm")
            ]
        )

        count = 0
        for p in img_paths:
            img = _read_dcm_pixel_array(p)
            if img.sum() > 100000:
                resized_img = _resize_img(img, img_px_size)
                resized_img = _normalize_0_1(resized_img)
                stacked_img = np.stack((resized_img,) * 3, axis=-1)  # H,W,3

                if stacked_img.sum() > 3000:
                    X[i, count] = stacked_img.astype(np.float32)
                    mask[i, count] = 1.0
                    count += 1
                    if count >= max_slices:
                        break

    if X.size > 0:
        m = float(np.max(X))
        if m > 0:
            X = X / m

    print("Test cases loaded:", len(ids), "with up to", max_slices, "T2W slices each.")
    return ids, X, mask


test_ids, X_test_by_case, test_mask = load_test_T2W_images_by_case(
    test, img_px_size=150, max_slices=6
)

pixels_7 = X_test_by_case[:, 0]
pixels_8 = X_test_by_case[:, 1]
pixels_9 = X_test_by_case[:, 2]
pixels_10 = X_test_by_case[:, 3]




## === cell 4
def load_train_T2W_images(path_train, labels_df):
    X1, X2, X3, X4 = [], [], [], []
    y = []
    IMG_PX_SIZE = 150

    bad_ids = set(["00109", "00123", "00709"])
    labels_map = dict(
        zip(
            labels_df["BraTS21ID"].astype(str).str.zfill(5),
            labels_df["MGMT_value"].astype(int),
        )
    )

    path_cases = sorted([f.path for f in os.scandir(path_train) if f.is_dir()])
    for case_path in path_cases:
        case_id = os.path.basename(case_path)
        if case_id in bad_ids:
            continue
        if case_id not in labels_map:
            continue

        mri_type = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])
        t2w_dir = None
        if len(mri_type) >= 4:
            t2w_dir = mri_type[3]
        else:
            for p in mri_type:
                if "T2w" in os.path.basename(p) or "T2W" in os.path.basename(p):
                    t2w_dir = p
                    break
        if t2w_dir is None:
            continue

        img_paths = sorted(
            [
                f.path
                for f in os.scandir(t2w_dir)
                if f.is_file() and f.name.lower().endswith(".dcm")
            ]
        )
        selected = []
        for p in img_paths:
            img = _read_dcm_pixel_array(p)
            if img.sum() > 100000:
                resized_img = _resize_img(img, IMG_PX_SIZE)
                resized_img = _normalize_0_1(resized_img)
                stacked_img = np.stack((resized_img,) * 3, axis=-1)
                if stacked_img.sum() > 3000:
                    selected.append(stacked_img)
                    if len(selected) >= 4:
                        break

        if len(selected) < 4:
            continue

        X1.append(selected[0])
        X2.append(selected[1])
        X3.append(selected[2])
        X4.append(selected[3])
        y.append(labels_map[case_id])

    X1 = np.asarray(X1, dtype=np.float32)
    X2 = np.asarray(X2, dtype=np.float32)
    X3 = np.asarray(X3, dtype=np.float32)
    X4 = np.asarray(X4, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    for arr_name, arr in [("X1", X1), ("X2", X2), ("X3", X3), ("X4", X4)]:
        if arr.size > 0:
            m = np.max(arr)
            if m > 0:
                arr /= m

    print("Train cases loaded:", len(y))
    return X1, X2, X3, X4, y


labels = pd.read_csv(LABELS_CSV)
X1_tr, X2_tr, X3_tr, X4_tr, y_tr = load_train_T2W_images(TRAIN_DIR, labels)


def build_model(input_shape=(150, 150, 3)):
    inp = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    return model


model_2 = build_model((150, 150, 3))

if len(y_tr) > 0:
    X_all = np.concatenate([X1_tr, X2_tr, X3_tr, X4_tr], axis=0)
    y_all = np.concatenate([y_tr, y_tr, y_tr, y_tr], axis=0)
    idx = np.arange(len(y_all))
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)
    X_all = X_all[idx]
    y_all = y_all[idx]

    model_2.fit(
        X_all,
        y_all,
        epochs=3,
        batch_size=16,
        verbose=1,
    )
else:
    print("Warning: no training data loaded; predictions will default to 0.5")


def _predict_prob(model, X):
    if X is None or (isinstance(X, np.ndarray) and X.size == 0):
        return np.array([], dtype=np.float32)
    p = model.predict(X, batch_size=16, verbose=0).reshape(-1)
    return p.astype(np.float32)


prediction_7 = _predict_prob(model_2, pixels_7)
prediction_8 = _predict_prob(model_2, pixels_8)
prediction_9 = _predict_prob(model_2, pixels_9)
prediction_10 = _predict_prob(model_2, pixels_10)




## === cell 5
def create_sub_from_ids(case_ids, p7, p8, p9, p10):
    cases = [str(x).zfill(5) for x in case_ids]
    n = len(cases)

    def _pad_or_trim(arr, n):
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        if len(arr) == n:
            return arr
        out = np.full((n,), 0.5, dtype=np.float32)
        m = min(len(arr), n)
        if m > 0:
            out[:m] = arr[:m]
        return out

    p7 = _pad_or_trim(p7, n)
    p8 = _pad_or_trim(p8, n)
    p9 = _pad_or_trim(p9, n)
    p10 = _pad_or_trim(p10, n)

    prediction = (p7 + p8 + p9 + p10) / 4.0
    prediction = np.clip(prediction, 1e-6, 1 - 1e-6)

    df = pd.DataFrame({"BraTS21ID": cases, "MGMT_value": prediction.astype(np.float32)})
    return df


sub_df = create_sub_from_ids(
    test_ids, prediction_7, prediction_8, prediction_9, prediction_10
)
sub_df = sub_df[["BraTS21ID", "MGMT_value"]]
sub_df.head()



## === cell 6
sample = pd.read_csv(SAMPLE_SUB)
sample["BraTS21ID"] = sample["BraTS21ID"].astype(str).str.zfill(5)

sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(str).str.zfill(5)

merged = sample[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
merged["MGMT_value"] = merged["MGMT_value"].astype(np.float32)
merged["MGMT_value"] = merged["MGMT_value"].fillna(0.5)
merged["MGMT_value"] = np.clip(merged["MGMT_value"].values, 1e-6, 1 - 1e-6)

missing = merged["MGMT_value"].isna().sum()
print(
    "Submission rows:",
    len(merged),
    "| sample rows:",
    len(sample),
    "| any missing preds filled:",
    missing,
)

print("Unique IDs in submission:", merged["BraTS21ID"].nunique())
print("Any duplicated IDs:", merged["BraTS21ID"].duplicated().any())



## === cell 7
try:
    import seaborn as sns

    sns.displot(merged["MGMT_value"])
except Exception as e:
    print("Plot skipped:", str(e))



## === cell 8
merged.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", merged.shape)
print(merged.head())
