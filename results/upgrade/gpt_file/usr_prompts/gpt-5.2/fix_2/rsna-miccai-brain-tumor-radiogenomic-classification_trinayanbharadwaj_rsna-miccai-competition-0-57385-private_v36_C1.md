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
import random
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

import cv2
import pydicom as dicom

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_PATH = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(BASE_PATH, "train")
TEST_DIR = os.path.join(BASE_PATH, "test")
TRAIN_CSV = os.path.join(BASE_PATH, "train_labels.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")

assert os.path.isdir(TRAIN_DIR), f"Missing TRAIN_DIR: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing TEST_DIR: {TEST_DIR}"
assert os.path.isfile(TRAIN_CSV), f"Missing TRAIN_CSV: {TRAIN_CSV}"
assert os.path.isfile(SAMPLE_SUB), f"Missing SAMPLE_SUB: {SAMPLE_SUB}"

train_df = pd.read_csv(TRAIN_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

assert set(["BraTS21ID", "MGMT_value"]).issubset(train_df.columns)
assert set(["BraTS21ID", "MGMT_value"]).issubset(sample_df.columns)

train_df["BraTS21ID"] = train_df["BraTS21ID"].astype(int)
train_df["MGMT_value"] = train_df["MGMT_value"].astype(int)

bad_cases = {109, 123, 709}
train_df = train_df[~train_df["BraTS21ID"].isin(bad_cases)].reset_index(drop=True)

train_df.head()



## === cell 2
IMG_PX_SIZE = 299
CHANNELS = 3


def _sorted_subdirs(path):
    return sorted([f.path for f in os.scandir(path) if f.is_dir()])


def _sorted_files(path):
    return sorted([f.path for f in os.scandir(path) if f.is_file()])


def load_one_case_modality_slice(case_dir, modality="FLAIR"):
    """
    Core logic preserved: pick the first slice in the modality series that passes
    intensity heuristics, resize to 299x299, stack to 3 channels, normalize.

    Returns float32 image in [0, 1], shape (299, 299, 3). Falls back to middle slice if needed.
    """
    modality_dir = os.path.join(case_dir, modality)
    dcm_files = _sorted_files(modality_dir)
    if len(dcm_files) == 0:
        raise FileNotFoundError(f"No DICOM files found in {modality_dir}")

    chosen = None
    for fp in dcm_files:
        ds = dicom.dcmread(fp, stop_before_pixels=False, force=True)
        if not hasattr(ds, "pixel_array"):
            continue
        arr = ds.pixel_array.astype(np.float32)
        if arr.sum() > 100000:
            chosen = arr
            mx = float(np.max(chosen)) if np.max(chosen) > 0 else 1.0
            if (chosen / mx).sum() > 5000:
                break

    if chosen is None:
        mid_fp = dcm_files[len(dcm_files) // 2]
        ds = dicom.dcmread(mid_fp, stop_before_pixels=False, force=True)
        chosen = ds.pixel_array.astype(np.float32)

    resized = cv2.resize(
        chosen, (IMG_PX_SIZE, IMG_PX_SIZE), interpolation=cv2.INTER_AREA
    ).astype(np.float32)

    stacked = np.stack([resized, resized, resized], axis=-1)

    mx = float(np.max(stacked)) if np.max(stacked) > 0 else 1.0
    stacked = stacked / mx

    return stacked.astype(np.float32)


def list_case_ids_from_dir(root_dir):
    case_dirs = _sorted_subdirs(root_dir)
    ids = [int(os.path.basename(p)) for p in case_dirs]
    return ids, case_dirs




## === cell 3
def build_dataset_from_df(df, root_dir, modality="FLAIR"):
    X = np.zeros((len(df), IMG_PX_SIZE, IMG_PX_SIZE, CHANNELS), dtype=np.float32)
    y = None
    if "MGMT_value" in df.columns:
        y = df["MGMT_value"].values.astype(np.float32)

    for i, brats_id in enumerate(df["BraTS21ID"].astype(int).tolist()):
        case_dir = os.path.join(root_dir, f"{brats_id:05d}")
        X[i] = load_one_case_modality_slice(case_dir, modality=modality)

        if not np.isfinite(X[i]).all():
            X[i] = np.nan_to_num(X[i], nan=0.0, posinf=1.0, neginf=0.0)

    return X, y


test_ids, test_case_dirs = list_case_ids_from_dir(TEST_DIR)
test_df = (
    pd.DataFrame({"BraTS21ID": test_ids})
    .sort_values("BraTS21ID")
    .reset_index(drop=True)
)

len(test_df), test_df.head()




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3112258990.py in <cell line: 0>()
     17 
     18 # Build test set once (for submission)
---> 19 test_ids, test_case_dirs = list_case_ids_from_dir(TEST_DIR)
     20 test_df = (
     21     pd.DataFrame({"BraTS21ID": test_ids})

/tmp/ipykernel_11/4243067048.py in list_case_ids_from_dir(root_dir)
     60     case_dirs = _sorted_subdirs(root_dir)
     61     # case folder names are like "00002"
---> 62     ids = [int(os.path.basename(p)) for p in case_dirs]
     63     return ids, case_dirs
     64 

/tmp/ipykernel_11/4243067048.py in <listcomp>(.0)
     60     case_dirs = _sorted_subdirs(root_dir)
     61     # case folder names are like "00002"
---> 62     ids = [int(os.path.basename(p)) for p in case_dirs]
     63     return ids, case_dirs
     64 

ValueError: invalid literal for int() with base 10: 'test'

## === cell 4
def make_model(input_shape=(IMG_PX_SIZE, IMG_PX_SIZE, CHANNELS)):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model




## === cell 5
from sklearn.model_selection import train_test_split

train_split, val_split = train_test_split(
    train_df, test_size=0.2, random_state=SEED, stratify=train_df["MGMT_value"]
)

X_train_flair, y_train = build_dataset_from_df(train_split, TRAIN_DIR, modality="FLAIR")
X_val_flair, y_val = build_dataset_from_df(val_split, TRAIN_DIR, modality="FLAIR")

X_train_t2, _ = build_dataset_from_df(train_split, TRAIN_DIR, modality="T2w")
X_val_t2, _ = build_dataset_from_df(val_split, TRAIN_DIR, modality="T2w")

BATCH = 8
AUTOTUNE = tf.data.AUTOTUNE


def make_tfds(X, y, training=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(buffer_size=len(X), seed=SEED, reshuffle_each_iteration=True)

        def aug(img, label):
            img = tf.image.random_flip_left_right(img, seed=SEED)
            img = tf.image.random_flip_up_down(img, seed=SEED)
            return img, label

        ds = ds.map(aug, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH).prefetch(AUTOTUNE)
    return ds


ds_train_flair = make_tfds(X_train_flair, y_train, training=True)
ds_val_flair = make_tfds(X_val_flair, y_val, training=False)
ds_train_t2 = make_tfds(X_train_t2, y_train, training=True)
ds_val_t2 = make_tfds(X_val_t2, y_val, training=False)

model_flair = make_model()
model_t2 = make_model()

EPOCHS = (
    3  # keep within runtime constraints while still producing meaningful predictions
)

hist_flair = model_flair.fit(
    ds_train_flair, validation_data=ds_val_flair, epochs=EPOCHS, verbose=2
)
hist_t2 = model_t2.fit(ds_train_t2, validation_data=ds_val_t2, epochs=EPOCHS, verbose=2)



## === cell 6
X_test_flair, _ = build_dataset_from_df(test_df, TEST_DIR, modality="FLAIR")
X_test_t2, _ = build_dataset_from_df(test_df, TEST_DIR, modality="T2w")

pred_flair = model_flair.predict(X_test_flair, batch_size=BATCH, verbose=1).reshape(-1)
pred_t2 = model_t2.predict(X_test_t2, batch_size=BATCH, verbose=1).reshape(-1)

pred = (pred_flair.astype(np.float64) + pred_t2.astype(np.float64)) / 2.0
pred = np.clip(pred, 0.0, 1.0)

pred[:10], pred.shape



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2193099940.py in <cell line: 0>()
      1 # Build test tensors
----> 2 X_test_flair, _ = build_dataset_from_df(test_df, TEST_DIR, modality="FLAIR")
      3 X_test_t2, _ = build_dataset_from_df(test_df, TEST_DIR, modality="T2w")
      4 
      5 # Predict probabilities

NameError: name 'test_df' is not defined

## === cell 7
sub_df = pd.DataFrame(
    {
        "BraTS21ID": test_df["BraTS21ID"].astype(int).map(lambda x: f"{x:05d}"),
        "MGMT_value": pred.astype(np.float32),
    }
)

sub_df = (
    sub_df[["BraTS21ID", "MGMT_value"]].sort_values("BraTS21ID").reset_index(drop=True)
)

sub_df.head(), sub_df.shape



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1935914254.py in <cell line: 0>()
      2 sub_df = pd.DataFrame(
      3     {
----> 4         "BraTS21ID": test_df["BraTS21ID"].astype(int).map(lambda x: f"{x:05d}"),
      5         "MGMT_value": pred.astype(np.float32),
      6     }

NameError: name 'test_df' is not defined

## === cell 8
out_path = "submission.csv"
sub_df.to_csv(out_path, index=False)

assert os.path.isfile(out_path)
check = pd.read_csv(out_path)
assert list(check.columns) == ["BraTS21ID", "MGMT_value"]
assert len(check) == len(sub_df)
check.head()



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/54386059.py in <cell line: 0>()
      1 # Write submission
      2 out_path = "submission.csv"
----> 3 sub_df.to_csv(out_path, index=False)
      4 
      5 # Basic validation

NameError: name 'sub_df' is not defined

## === cell 9
if plt is not None:
    plt.figure(figsize=(10, 3))
    plt.plot(hist_flair.history["val_auc"], label="FLAIR val AUC")
    plt.plot(hist_t2.history["val_auc"], label="T2w val AUC")
    plt.title("Validation AUC (sanity check)")
    plt.legend()
    plt.tight_layout()
    plt.show()
print(f"Saved submission to: {out_path}")
