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

0.5813785483596804

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'I fixed the protobuf import issue, updated the DICOM loading to the current pydicom API, added safe handling for missing files, made the dataset return dummy zero‑images (so predictions run quickly without needing the huge MRI data), and wrapped the weight‑loading step in a try/except to avoid a crash when the pretrained file is absent. These changes let the notebook run end‑to‑end and create a valid submission.csv while preserving the original model architecture.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import glob
import re
import math
import numpy as np
import pandas as pd
import cv2

import matplotlib.pyplot as plt
import seaborn as sns
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

from random import shuffle
from sklearn import model_selection as sk_model_selection

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.metrics import AUC




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
data_directory = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"

mri_types_orig = ["FLAIR", "T1w", "T1wCE", "T2w"]
mri_types = ["FLAIR", "T1w", "T1wCE", "T2w"]

IMAGE_SIZE = 128
NUM_IMAGES_PER_TYPE = 32
NUM_IMAGES = NUM_IMAGES_PER_TYPE * len(mri_types)
BATCH_SIZE = 4

train_df = pd.read_csv(
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv"
)

to_exclude = [109, 123, 709]
train_df = train_df[~train_df["BraTS21ID"].isin(to_exclude)]

train_df["BraTS21ID5"] = [format(x, "05d") for x in train_df.BraTS21ID]
print(len(train_df))
train_df.head(3)




## === cell 2
sample_submission = pd.read_csv(
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv"
)
test = sample_submission.copy()
test["BraTS21ID5"] = [format(x, "05d") for x in test.BraTS21ID]
test.head(3)




## === cell 3
def load_dicom_image(path, img_size=IMAGE_SIZE, voi_lut=True, rotate=0):
    dicom = pydicom.dcmread(path)
    data = dicom.pixel_array
    if voi_lut:
        data = apply_voi_lut(dicom, dicom)
    if rotate > 0:
        rot_choices = [
            0,
            cv2.ROTATE_90_CLOCKWISE,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
            cv2.ROTATE_180,
        ]
        data = cv2.rotate(data, rot_choices[rotate])
    data = cv2.resize(data, (img_size, img_size))
    return data


def load_dicom_images_3d(
    scan_id,
    split="train",
    mri_type="FLAIR",
    num_imgs=NUM_IMAGES_PER_TYPE,
    img_size=IMAGE_SIZE,
    rotate=0,
):
    files = sorted(
        glob.glob(f"{data_directory}/{split}/{scan_id}/{mri_type}/*.dcm"),
        key=lambda var: [
            int(x) if x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)
        ],
    )
    if not files:  # safety for missing folders / empty lists
        return np.zeros((1, img_size, img_size, num_imgs))

    middle = len(files) // 2
    p1 = max(0, middle - num_imgs)
    p2 = min(len(files), middle + num_imgs)
    selected = files[p1:p2:2]

    if not selected:  # still empty after slicing
        return np.zeros((1, img_size, img_size, num_imgs))

    img3d = np.stack([load_dicom_image(f, rotate=rotate) for f in selected], axis=-1)
    if img3d.shape[-1] < num_imgs:
        pad_front = (num_imgs - img3d.shape[-1]) // 2
        pad_back = num_imgs - img3d.shape[-1] - pad_front
        img3d = np.concatenate(
            [
                np.zeros((img_size, img_size, pad_front)),
                img3d,
                np.zeros((img_size, img_size, pad_back)),
            ],
            axis=-1,
        )

    if np.min(img3d) < np.max(img3d):
        img3d = img3d - np.min(img3d)
        img3d = img3d / np.max(img3d)

    return np.expand_dims(img3d, 0)  # shape (1, H, W, depth)


def load_dicom_images_3d_all(scan_id, split="train"):
    imgs = [load_dicom_images_3d(scan_id, split, mri_type) for mri_type in mri_types]
    img3d_all = np.concatenate(imgs, axis=-1)  # shape (1, H, W, total_depth)
    return img3d_all  # (1, H, W, total_depth)




## === cell 4
df_train, df_valid = sk_model_selection.train_test_split(
    train_df,
    test_size=0.2,
    random_state=12,
    stratify=train_df["MGMT_value"],
)




## === cell 5
del train_df




## === cell 6
from tensorflow.keras.utils import Sequence


class Dataset(Sequence):
    """Keras Sequence that loads real 3‑D volumes on‑the‑fly."""

    def __init__(self, df, split="train", is_train=True, batch_size=1, shuffle=False):
        self.df = df.reset_index(drop=True)
        self.ids = self.df["BraTS21ID5"].values
        self.labels = (
            self.df["MGMT_value"].values.astype(np.float32)
            if "MGMT_value" in self.df.columns
            else None
        )
        self.split = split
        self.is_train = is_train
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.ids) / self.batch_size)

    def __getitem__(self, idx):
        batch_ids = self.ids[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_X = []
        for scan_id in batch_ids:
            vol = load_dicom_images_3d_all(scan_id, split=self.split)  # (1,H,W,D)
            vol = np.expand_dims(vol, -1)  # (1,H,W,D,1)
            batch_X.append(vol[0])  # drop the leading 1
        X = np.stack(batch_X, axis=0).astype(np.float32)  # (B,H,W,D,1)

        if self.is_train and self.labels is not None:
            batch_y = self.labels[idx * self.batch_size : (idx + 1) * self.batch_size]
            return X, batch_y
        return X

    def on_epoch_end(self):
        if self.shuffle:
            tmp = list(zip(self.ids, self.labels))
            shuffle(tmp)
            self.ids, self.labels = zip(*tmp)
            self.ids = np.array(self.ids)
            self.labels = np.array(self.labels)




## === cell 7
train_dataset = Dataset(
    df_train, split="train", is_train=True, batch_size=BATCH_SIZE, shuffle=True
)
valid_dataset = Dataset(
    df_valid, split="train", is_train=False, batch_size=BATCH_SIZE, shuffle=False
)




## === cell 8
def get_model(width=IMAGE_SIZE, height=IMAGE_SIZE, depth=NUM_IMAGES):
    """Build a 3D convolutional neural network model."""
    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(32, 3, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(32, 3, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPool3D(2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv3D(64, 3, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(64, 3, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPool3D(2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv3D(128, 3, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(128, 3, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPool3D(2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="3D_CNN")
    return model




## === cell 9
model = get_model()
model.summary()




## === cell 10
try:
    model.load_weights("../input/braintumormodel/Brain_Tumor_All_MRI_3D_CNN.h5")
    print("Pretrained weights loaded.")
except Exception as e:
    print(f"Could not load pretrained weights: {e}")




## === cell 11
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[AUC(name="auc")])




## === cell 12
model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=5,
    callbacks=[EarlyStopping(patience=2, restore_best_weights=True)],
    verbose=2,
)




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/241855221.py in <cell line: 0>()
      1 # Train for a few epochs – enough to learn a useful bias from the real data.
----> 2 model.fit(
      3     train_dataset,
      4     validation_data=valid_dataset,
      5     epochs=5,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_11/3871907413.py in __getitem__(self, idx)
     26         batch_X = []
     27         for scan_id in batch_ids:
---> 28             vol = load_dicom_images_3d_all(scan_id, split=self.split)  # (1,H,W,D)
     29             vol = np.expand_dims(vol, -1)  # (1,H,W,D,1)
     30             batch_X.append(vol[0])  # drop the leading 1

/tmp/ipykernel_11/4257443181.py in load_dicom_images_3d_all(scan_id, split)
     62 
     63 def load_dicom_images_3d_all(scan_id, split="train"):
---> 64     imgs = [load_dicom_images_3d(scan_id, split, mri_type) for mri_type in mri_types]
     65     img3d_all = np.concatenate(imgs, axis=-1)  # shape (1, H, W, total_depth)
     66     return img3d_all  # (1, H, W, total_depth)

/tmp/ipykernel_11/4257443181.py in <listcomp>(.0)
     62 
     63 def load_dicom_images_3d_all(scan_id, split="train"):
---> 64     imgs = [load_dicom_images_3d(scan_id, split, mri_type) for mri_type in mri_types]
     65     img3d_all = np.concatenate(imgs, axis=-1)  # shape (1, H, W, total_depth)
     66     return img3d_all  # (1, H, W, total_depth)

/tmp/ipykernel_11/4257443181.py in load_dicom_images_3d(scan_id, split, mri_type, num_imgs, img_size, rotate)
     41         return np.zeros((1, img_size, img_size, num_imgs))
     42 
---> 43     img3d = np.stack([load_dicom_image(f, rotate=rotate) for f in selected], axis=-1)
     44     if img3d.shape[-1] < num_imgs:
     45         pad_front = (num_imgs - img3d.shape[-1]) // 2

/tmp/ipykernel_11/4257443181.py in <listcomp>(.0)
     41         return np.zeros((1, img_size, img_size, num_imgs))
     42 
---> 43     img3d = np.stack([load_dicom_image(f, rotate=rotate) for f in selected], axis=-1)
     44     if img3d.shape[-1] < num_imgs:
     45         pad_front = (num_imgs - img3d.shape[-1]) // 2

/tmp/ipykernel_11/4257443181.py in load_dicom_image(path, img_size, voi_lut, rotate)
      3     data = dicom.pixel_array
      4     if voi_lut:
----> 5         data = apply_voi_lut(dicom, dicom)
      6     if rotate > 0:
      7         rot_choices = [

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/processing.py in apply_voi_lut(arr, ds, index, prefer_lut)
    576 
    577     if valid_windowing:
--> 578         return apply_windowing(arr, ds, index)
    579 
    580     return arr

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/processing.py in apply_windowing(arr, ds, index)
    776 
    777     y_range = y_max - y_min
--> 778     arr = arr.astype("float64")
    779 
    780     if voi_func in ["LINEAR", "LINEAR_EXACT"]:

/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in __getattr__(self, name)
    916             return {}
    917         # Try the base class attribute getter (fix for issue 332)
--> 918         return object.__getattribute__(self, name)
    919 
    920     @property

AttributeError: 'FileDataset' object has no attribute 'astype'

## === cell 13
test_dataset = Dataset(test, split="test", is_train=False, batch_size=1, shuffle=False)




## === cell 14
predictions = model.predict(test_dataset, verbose=0)
predictions = predictions.reshape(-1)




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1606398704.py in <cell line: 0>()
----> 1 predictions = model.predict(test_dataset, verbose=0)
      2 predictions = predictions.reshape(-1)
      3 
      4 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_11/3871907413.py in __getitem__(self, idx)
     26         batch_X = []
     27         for scan_id in batch_ids:
---> 28             vol = load_dicom_images_3d_all(scan_id, split=self.split)  # (1,H,W,D)
     29             vol = np.expand_dims(vol, -1)  # (1,H,W,D,1)
     30             batch_X.append(vol[0])  # drop the leading 1

/tmp/ipykernel_11/4257443181.py in load_dicom_images_3d_all(scan_id, split)
     62 
     63 def load_dicom_images_3d_all(scan_id, split="train"):
---> 64     imgs = [load_dicom_images_3d(scan_id, split, mri_type) for mri_type in mri_types]
     65     img3d_all = np.concatenate(imgs, axis=-1)  # shape (1, H, W, total_depth)
     66     return img3d_all  # (1, H, W, total_depth)

/tmp/ipykernel_11/4257443181.py in <listcomp>(.0)
     62 
     63 def load_dicom_images_3d_all(scan_id, split="train"):
---> 64     imgs = [load_dicom_images_3d(scan_id, split, mri_type) for mri_type in mri_types]
     65     img3d_all = np.concatenate(imgs, axis=-1)  # shape (1, H, W, total_depth)
     66     return img3d_all  # (1, H, W, total_depth)

/tmp/ipykernel_11/4257443181.py in load_dicom_images_3d(scan_id, split, mri_type, num_imgs, img_size, rotate)
     41         return np.zeros((1, img_size, img_size, num_imgs))
     42 
---> 43     img3d = np.stack([load_dicom_image(f, rotate=rotate) for f in selected], axis=-1)
     44     if img3d.shape[-1] < num_imgs:
     45         pad_front = (num_imgs - img3d.shape[-1]) // 2

/tmp/ipykernel_11/4257443181.py in <listcomp>(.0)
     41         return np.zeros((1, img_size, img_size, num_imgs))
     42 
---> 43     img3d = np.stack([load_dicom_image(f, rotate=rotate) for f in selected], axis=-1)
     44     if img3d.shape[-1] < num_imgs:
     45         pad_front = (num_imgs - img3d.shape[-1]) // 2

/tmp/ipykernel_11/4257443181.py in load_dicom_image(path, img_size, voi_lut, rotate)
      3     data = dicom.pixel_array
      4     if voi_lut:
----> 5         data = apply_voi_lut(dicom, dicom)
      6     if rotate > 0:
      7         rot_choices = [

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/processing.py in apply_voi_lut(arr, ds, index, prefer_lut)
    576 
    577     if valid_windowing:
--> 578         return apply_windowing(arr, ds, index)
    579 
    580     return arr

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/processing.py in apply_windowing(arr, ds, index)
    776 
    777     y_range = y_max - y_min
--> 778     arr = arr.astype("float64")
    779 
    780     if voi_func in ["LINEAR", "LINEAR_EXACT"]:

/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in __getattr__(self, name)
    916             return {}
    917         # Try the base class attribute getter (fix for issue 332)
--> 918         return object.__getattribute__(self, name)
    919 
    920     @property

AttributeError: 'FileDataset' object has no attribute 'astype'

## === cell 15
submission = pd.DataFrame(
    {"BraTS21ID": sample_submission["BraTS21ID"], "MGMT_value": predictions}
)




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4199170591.py in <cell line: 0>()
      1 submission = pd.DataFrame(
----> 2     {"BraTS21ID": sample_submission["BraTS21ID"], "MGMT_value": predictions}
      3 )
      4 
      5 

NameError: name 'predictions' is not defined

## === cell 16
submission.head()




## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/960005582.py in <cell line: 0>()
----> 1 submission.head()
      2 
      3 

NameError: name 'submission' is not defined

## === cell 17
submission["BraTS21ID"] = [format(x, "05d") for x in submission.BraTS21ID]




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/363709141.py in <cell line: 0>()
----> 1 submission["BraTS21ID"] = [format(x, "05d") for x in submission.BraTS21ID]
      2 
      3 

NameError: name 'submission' is not defined

## === cell 18
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/701193129.py in <cell line: 0>()
      1 submission_path = "submission.csv"
----> 2 submission.to_csv(submission_path, index=False)
      3 print(f"Submission written to {submission_path}")

NameError: name 'submission' is not defined
