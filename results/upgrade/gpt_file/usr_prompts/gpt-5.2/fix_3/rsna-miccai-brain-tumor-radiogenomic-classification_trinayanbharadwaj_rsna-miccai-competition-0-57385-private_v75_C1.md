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

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TF version:", tf.__version__)
print("Eager:", tf.executing_eagerly())



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
TRAIN_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

assert os.path.exists(TRAIN_DIR), TRAIN_DIR
assert os.path.exists(TEST_DIR), TEST_DIR
assert os.path.exists(TRAIN_CSV), TRAIN_CSV
assert os.path.exists(SAMPLE_SUB), SAMPLE_SUB

train_labels = pd.read_csv(TRAIN_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

print("train_labels:", train_labels.shape)
print("sample_sub:", sample_sub.shape)

BAD_IDS = {"00109", "00123", "00709"}
train_labels["BraTS21ID_str"] = train_labels["BraTS21ID"].astype(str).str.zfill(5)
train_labels = train_labels[~train_labels["BraTS21ID_str"].isin(BAD_IDS)].reset_index(
    drop=True
)

print("train_labels after exclude:", train_labels.shape)
print(train_labels.head())



## === cell 2
from skimage.transform import resize


def _read_dicom_pixel_array_tf(dcm_path: str):
    """
    Bug fix: avoid pydicom import/protobuf crash by using TF's DICOM decoder.
    Returns a 2D float32 array (H, W) or None if unreadable.
    """
    try:
        b = tf.io.read_file(dcm_path)
        img = tf.io.decode_dicom_image(
            b,
            dtype=tf.uint16,
            color_dim=False,
            scale="auto",
        )
        img = tf.squeeze(img)  # could become [H, W] or [H, W, 1]
        if img.shape.rank == 3:
            img = tf.squeeze(img, axis=-1)
        arr = img.numpy()
        return arr.astype(np.float32)
    except Exception:
        return None


def load_case_slices_T2W(case_dir, img_px_size=150, max_slices=8):
    """
    Load up to `max_slices` "informative" slices from T2w series for a single case directory.
    Returns: np.ndarray of shape (n_slices, img_px_size, img_px_size, 3) float32 in [0,1]
    """
    t2w_dir = os.path.join(case_dir, "T2w")
    if not os.path.isdir(t2w_dir):
        return np.zeros((0, img_px_size, img_px_size, 3), dtype=np.float32)

    dcm_files = sorted(
        [
            f.path
            for f in os.scandir(t2w_dir)
            if f.is_file() and f.name.lower().endswith(".dcm")
        ]
    )
    if len(dcm_files) == 0:
        return np.zeros((0, img_px_size, img_px_size, 3), dtype=np.float32)

    out = []
    for fp in dcm_files:
        arr = _read_dicom_pixel_array_tf(fp)
        if arr is None:
            continue

        if arr.sum() <= 100000:
            continue

        arr_resized = resize(
            arr, (img_px_size, img_px_size), preserve_range=True, anti_aliasing=True
        ).astype(np.float32)

        mx = float(np.max(arr_resized))
        if mx <= 0:
            continue
        arr_norm = arr_resized / mx  # per-slice normalize

        stacked = np.stack((arr_norm,) * 3, axis=-1)
        if stacked.sum() <= 2000:
            continue

        out.append(stacked)
        if len(out) >= max_slices:
            break

    if len(out) == 0:
        return np.zeros((0, img_px_size, img_px_size, 3), dtype=np.float32)

    return np.asarray(out, dtype=np.float32)




## === cell 3
IMG_PX_SIZE = 150
N_SLICES = 8

train_ids = []
y = []
X_slices = []  # all slices pooled for training

id_to_label = dict(
    zip(train_labels["BraTS21ID_str"], train_labels["MGMT_value"].astype(np.int32))
)

all_train_case_dirs = sorted([f.path for f in os.scandir(TRAIN_DIR) if f.is_dir()])
all_train_case_ids = [os.path.basename(p) for p in all_train_case_dirs]
usable_train_ids = [cid for cid in all_train_case_ids if cid in id_to_label]

print(
    "Train cases on disk:",
    len(all_train_case_ids),
    "usable with labels:",
    len(usable_train_ids),
)

for cid in usable_train_ids:
    case_dir = os.path.join(TRAIN_DIR, cid)
    slices = load_case_slices_T2W(
        case_dir, img_px_size=IMG_PX_SIZE, max_slices=N_SLICES
    )
    if slices.shape[0] == 0:
        continue
    label = id_to_label[cid]
    for s in range(slices.shape[0]):
        X_slices.append(slices[s])
        y.append(label)
    train_ids.append(cid)

X_slices = np.asarray(X_slices, dtype=np.float32)
y = np.asarray(y, dtype=np.float32)  # binary labels for sigmoid/BCE

print("Total training slices:", X_slices.shape, "labels:", y.shape)
print("Label mean:", float(y.mean()) if y.size else None)

if X_slices.shape[0] < 10:
    raise RuntimeError("Too few training slices loaded; cannot train a model.")



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/365182089.py in <cell line: 0>()
     41 
     42 if X_slices.shape[0] < 10:
---> 43     raise RuntimeError("Too few training slices loaded; cannot train a model.")
     44 

RuntimeError: Too few training slices loaded; cannot train a model.

## === cell 4
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X_slices, y, test_size=0.2, random_state=SEED, stratify=y
)

print("X_train:", X_train.shape, "X_val:", X_val.shape)
print("y_train mean:", float(y_train.mean()), "y_val mean:", float(y_val.mean()))



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2705620715.py in <cell line: 0>()
      1 from sklearn.model_selection import train_test_split
      2 
----> 3 X_train, X_val, y_train, y_val = train_test_split(
      4     X_slices, y, test_size=0.2, random_state=SEED, stratify=y
      5 )

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py in train_test_split(test_size, train_size, random_state, shuffle, stratify, *arrays)
   2560 
   2561     n_samples = _num_samples(arrays[0])
-> 2562     n_train, n_test = _validate_shuffle_split(
   2563         n_samples, test_size, train_size, default_test_size=0.25
   2564     )

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py in _validate_shuffle_split(n_samples, test_size, train_size, default_test_size)
   2234 
   2235     if n_train == 0:
-> 2236         raise ValueError(
   2237             "With n_samples={}, test_size={} and train_size={}, the "
   2238             "resulting train set will be empty. Adjust any of the "

ValueError: With n_samples=0, test_size=0.2 and train_size=None, the resulting train set will be empty. Adjust any of the aforementioned parameters.

## === cell 5
model_T2 = keras.Sequential(
    [
        layers.Input(shape=(IMG_PX_SIZE, IMG_PX_SIZE, 3)),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ]
)

model_T2.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[keras.metrics.AUC(name="auc")],
)

model_T2.summary()



## === cell 6
history = model_T2.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=3,
    batch_size=32,
    verbose=2,
)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2761296776.py in <cell line: 0>()
      1 history = model_T2.fit(
----> 2     X_train,
      3     y_train,
      4     validation_data=(X_val, y_val),
      5     epochs=3,

NameError: name 'X_train' is not defined

## === cell 7
def load_test_T2W_images(path_test, img_px_size=150, max_slices=20, slots_n=8):
    """
    Returns:
      case_ids: list[str] (folder names)
      slots: list[np.ndarray] of length slots_n, each shaped (n_cases, H, W, 3)
    """
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    case_ids = [os.path.basename(p) for p in path_cases]
    n_cases = len(path_cases)

    slots = [
        np.zeros((n_cases, img_px_size, img_px_size, 3), dtype=np.float32)
        for _ in range(slots_n)
    ]

    for i, case_dir in enumerate(path_cases):
        slices = load_case_slices_T2W(
            case_dir, img_px_size=img_px_size, max_slices=max_slices
        )
        n = min(slices.shape[0], slots_n)
        for j in range(n):
            slots[j][i] = slices[j]

    print("Loaded test cases:", n_cases, "Slots:", len(slots))
    return case_ids, slots


test = TEST_DIR
test_case_ids, pixels_slots = load_test_T2W_images(
    test, img_px_size=IMG_PX_SIZE, max_slices=20, slots_n=8
)
pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6, pixels_7, pixels_8 = (
    pixels_slots
)



## === cell 8
preds_1 = model_T2.predict(pixels_1, verbose=0).reshape(-1)
preds_2 = model_T2.predict(pixels_2, verbose=0).reshape(-1)
preds_3 = model_T2.predict(pixels_3, verbose=0).reshape(-1)
preds_4 = model_T2.predict(pixels_4, verbose=0).reshape(-1)
preds_5 = model_T2.predict(pixels_5, verbose=0).reshape(-1)
preds_6 = model_T2.predict(pixels_6, verbose=0).reshape(-1)
preds_7 = model_T2.predict(pixels_7, verbose=0).reshape(-1)
preds_8 = model_T2.predict(pixels_8, verbose=0).reshape(-1)

print("Pred shapes:", preds_1.shape, preds_8.shape, "n_cases:", len(test_case_ids))




## === cell 9
def create_sub(case_ids, p1, p2, p3, p4, p5, p6, p7, p8):
    """
    Bug fix: robust case id handling and correct one-pred-per-case averaging.
    """
    pred = (
        p1.astype(np.float32)
        + p2.astype(np.float32)
        + p3.astype(np.float32)
        + p4.astype(np.float32)
        + p5.astype(np.float32)
        + p6.astype(np.float32)
        + p7.astype(np.float32)
        + p8.astype(np.float32)
    ) / 8.0

    cases_int = [int(cid) for cid in case_ids]
    df = pd.DataFrame({"BraTS21ID": cases_int, "MGMT_value": pred})
    return df


sub_df = create_sub(
    test_case_ids,
    preds_1,
    preds_2,
    preds_3,
    preds_4,
    preds_5,
    preds_6,
    preds_7,
    preds_8,
)

sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(int)
sub_df = sample_sub[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(0.5).clip(0.0, 1.0)

print(sub_df.head())
print("Submission shape:", sub_df.shape)
print("Missing predictions:", int(sub_df["MGMT_value"].isna().sum()))



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2764275505.py in <cell line: 0>()
     20 
     21 
---> 22 sub_df = create_sub(
     23     test_case_ids,
     24     preds_1,

/tmp/ipykernel_11/2764275505.py in create_sub(case_ids, p1, p2, p3, p4, p5, p6, p7, p8)
     15 
     16     # convert folder IDs like "00002" -> int 2 (to match sample_submission BraTS21ID type)
---> 17     cases_int = [int(cid) for cid in case_ids]
     18     df = pd.DataFrame({"BraTS21ID": cases_int, "MGMT_value": pred})
     19     return df

/tmp/ipykernel_11/2764275505.py in <listcomp>(.0)
     15 
     16     # convert folder IDs like "00002" -> int 2 (to match sample_submission BraTS21ID type)
---> 17     cases_int = [int(cid) for cid in case_ids]
     18     df = pd.DataFrame({"BraTS21ID": cases_int, "MGMT_value": pred})
     19     return df

ValueError: invalid literal for int() with base 10: 'test'

## === cell 10
try:
    sns.displot(sub_df["MGMT_value"])
    plt.show()
except Exception as e:
    print("Plot skipped:", repr(e))

out_path = "submission.csv"
sub_df.to_csv(out_path, index=False)
print("Wrote:", out_path, "bytes:", os.path.getsize(out_path))
print("Columns:", list(sub_df.columns))
print("Dtypes:", sub_df.dtypes.to_dict())

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3564021616.py in <cell line: 0>()
      6 
      7 out_path = "submission.csv"
----> 8 sub_df.to_csv(out_path, index=False)
      9 print("Wrote:", out_path, "bytes:", os.path.getsize(out_path))
     10 print("Columns:", list(sub_df.columns))

NameError: name 'sub_df' is not defined
