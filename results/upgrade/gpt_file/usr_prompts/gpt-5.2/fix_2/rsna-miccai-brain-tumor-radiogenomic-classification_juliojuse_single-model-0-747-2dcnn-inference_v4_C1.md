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
import glob
import random
import collections

import numpy as np
import pandas as pd

import cv2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

TYPES = ["FLAIR", "T1w", "T2w", "T1wCE"]
WHITE_THRESHOLD = 10  # unused but kept
EXCLUDE = [109, 123, 709]

BASE_PATH = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"

train_df = pd.read_csv(os.path.join(BASE_PATH, "train_labels.csv"))
test_df = pd.read_csv(os.path.join(BASE_PATH, "sample_submission.csv"))
train_df = train_df[~train_df.BraTS21ID.isin(EXCLUDE)].reset_index(drop=True)

IMAGE_SIZE = 128


def load_dicom(path, size=224):
    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    if img is None:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    img = img.astype(np.float32)
    mx = float(img.max()) if img.size else 0.0
    if mx > 0:
        img = img / mx
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


def get_all_image_paths(brats21id, image_type, folder="train"):
    assert image_type in TYPES
    patient_path = os.path.join(BASE_PATH, folder, str(int(brats21id)).zfill(5))
    paths = glob.glob(os.path.join(patient_path, image_type, "*"))

    def _key(p):
        base = os.path.basename(p)
        try:
            return int(os.path.splitext(base)[0].split("-")[-1])
        except Exception:
            return base

    paths = sorted(paths, key=_key)

    num_images = len(paths)
    if num_images == 0:
        return np.array([], dtype=object)

    start = int(num_images * 0.25)
    end = int(num_images * 0.75)

    interval = 3
    if num_images < 10:
        interval = 1
    sliced = paths[start:end:interval]
    if len(sliced) == 0:
        sliced = [paths[num_images // 2]]
    return np.array(sliced, dtype=object)


def get_all_images(brats21id, image_type, folder="train", size=225):
    paths = get_all_image_paths(brats21id, image_type, folder)
    return [load_dicom(p, size) for p in paths]


def get_all_data_for_train(image_type):
    X, y, train_ids = [], [], []
    for i in range(len(train_df)):
        row = train_df.loc[i]
        pid = int(row["BraTS21ID"])
        label = float(row["MGMT_value"])
        images = get_all_images(pid, image_type, "train", IMAGE_SIZE)
        X += images
        y += [label] * len(images)
        train_ids += [pid] * len(images)
    return np.array(X), np.array(y).astype(np.float32), np.array(train_ids)


def get_all_data_for_test(image_type):
    X, test_ids = [], []
    for i in range(len(test_df)):
        row = test_df.loc[i]
        pid = int(row["BraTS21ID"])
        images = get_all_images(pid, image_type, "test", IMAGE_SIZE)
        X += images
        test_ids += [pid] * len(images)
    return np.array(X), np.array(test_ids)


X_test, testidt = get_all_data_for_test("T1wCE")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
X_test = tf.expand_dims(X_test, -1)
X_test_ = tf.concat([X_test, X_test, X_test], axis=-1)
X_test_.shape



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3115765493.py in <cell line: 0>()
      1 # Expand to (H,W,1) then to 3 channels as in original flow
----> 2 X_test = tf.expand_dims(X_test, -1)
      3 X_test_ = tf.concat([X_test, X_test, X_test], axis=-1)
      4 X_test_.shape
      5 

NameError: name 'X_test' is not defined

## === cell 2
X_train, y_train, train_ids = get_all_data_for_train("T1wCE")
X_train = tf.expand_dims(X_train, -1)
X_train_ = tf.concat([X_train, X_train, X_train], axis=-1)

X_train_ = tf.cast(X_train_, tf.float32) / 255.0
X_test_ = tf.cast(X_test_, tf.float32) / 255.0

unique_ids = np.unique(train_ids)
tr_ids, va_ids = train_test_split(
    unique_ids, test_size=0.2, random_state=SEED, shuffle=True, stratify=None
)

tr_mask = np.isin(train_ids, tr_ids)
va_mask = np.isin(train_ids, va_ids)

X_tr = tf.boolean_mask(X_train_, tr_mask)
y_tr = y_train[tr_mask]
X_va = tf.boolean_mask(X_train_, va_mask)
y_va = y_train[va_mask]

X_tr.shape, X_va.shape, y_tr.mean(), y_va.mean()




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1076052759.py in <cell line: 0>()
      1 # Prepare training data (same slice-level approach as original; minimal addition to make code runnable)
----> 2 X_train, y_train, train_ids = get_all_data_for_train("T1wCE")
      3 X_train = tf.expand_dims(X_train, -1)
      4 X_train_ = tf.concat([X_train, X_train, X_train], axis=-1)
      5 

/tmp/ipykernel_11/1635580194.py in get_all_data_for_train(image_type)
     96         pid = int(row["BraTS21ID"])
     97         label = float(row["MGMT_value"])
---> 98         images = get_all_images(pid, image_type, "train", IMAGE_SIZE)
     99         X += images
    100         y += [label] * len(images)

/tmp/ipykernel_11/1635580194.py in get_all_images(brats21id, image_type, folder, size)
     87 def get_all_images(brats21id, image_type, folder="train", size=225):
     88     paths = get_all_image_paths(brats21id, image_type, folder)
---> 89     return [load_dicom(p, size) for p in paths]
     90 
     91 

/tmp/ipykernel_11/1635580194.py in <listcomp>(.0)
     87 def get_all_images(brats21id, image_type, folder="train", size=225):
     88     paths = get_all_image_paths(brats21id, image_type, folder)
---> 89     return [load_dicom(p, size) for p in paths]
     90 
     91 

/tmp/ipykernel_11/1635580194.py in load_dicom(path, size)
     42         img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
     43     if img is None:
---> 44         raise FileNotFoundError(f"Could not read image: {path}")
     45 
     46     img = img.astype(np.float32)

FileNotFoundError: Could not read image: ../input/rsna-miccai-brain-tumor-radiogenomic-classification/train/00642/T1wCE/Image-45.dcm

## === cell 3
def build_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


model = build_model()
model.summary()



## === cell 4
BATCH_SIZE = 32
EPOCHS = 2

train_ds = (
    tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
    .shuffle(2048, seed=SEED)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
val_ds = (
    tf.data.Dataset.from_tensor_slices((X_va, y_va))
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1042018194.py in <cell line: 0>()
      4 
      5 train_ds = (
----> 6     tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
      7     .shuffle(2048, seed=SEED)
      8     .batch(BATCH_SIZE)

NameError: name 'X_tr' is not defined

## === cell 5
y_pred_slices = model.predict(X_test_, batch_size=64, verbose=1).reshape(-1)

pred_df = pd.DataFrame(
    {"BraTS21ID": testidt.astype(int), "MGMT_value": y_pred_slices.astype(np.float32)}
)
pred_patient = pred_df.groupby("BraTS21ID", as_index=False)["MGMT_value"].mean()

pred_patient.head(), pred_patient.shape



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1164600844.py in <cell line: 0>()
      1 # Slice-level predictions on test
----> 2 y_pred_slices = model.predict(X_test_, batch_size=64, verbose=1).reshape(-1)
      3 
      4 # Aggregate to patient-level by mean probability (same aggregation intent as original code)
      5 pred_df = pd.DataFrame(

NameError: name 'X_test_' is not defined

## === cell 6
sample = pd.read_csv(os.path.join(BASE_PATH, "sample_submission.csv"))
sample["BraTS21ID"] = sample["BraTS21ID"].astype(int)

sub = sample[["BraTS21ID"]].merge(pred_patient, on="BraTS21ID", how="left")

sub["MGMT_value"] = sub["MGMT_value"].astype(np.float32).fillna(0.5).clip(0.0, 1.0)

sub.to_csv("submission.csv", index=False)
sub.head()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/477535225.py in <cell line: 0>()
      3 sample["BraTS21ID"] = sample["BraTS21ID"].astype(int)
      4 
----> 5 sub = sample[["BraTS21ID"]].merge(pred_patient, on="BraTS21ID", how="left")
      6 
      7 # Safety: if any missing (shouldn't), fill with 0.5

NameError: name 'pred_patient' is not defined

## === cell 7
assert os.path.exists("submission.csv")
assert list(sub.columns) == ["BraTS21ID", "MGMT_value"]
assert len(sub) == len(sample)
print("Saved submission.csv with shape:", sub.shape)
print(sub.describe(include="all"))

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_11/3477395824.py in <cell line: 0>()
      1 # Quick sanity checks (score-neutral)
----> 2 assert os.path.exists("submission.csv")
      3 assert list(sub.columns) == ["BraTS21ID", "MGMT_value"]
      4 assert len(sub) == len(sample)
      5 print("Saved submission.csv with shape:", sub.shape)

AssertionError:
