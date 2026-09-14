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

None

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
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

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
    from tensorflow.keras.metrics import AUC
except Exception as e:
    tf = None
    print("TensorFlow import failed, will use fallback predictions:", e)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
data_directory = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"

mri_types_orig = ["FLAIR", "T1w", "T1wCE", "T2w"]
mri_types = ["FLAIR", "T1w", "T1wCE", "T2w"]

IMAGE_SIZE = 96
NUM_IMAGES_PER_TYPE = 30
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
    else:
        data = dicom.pixel_array

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
    split,
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

    middle = len(files) // 2
    num_imgs2 = num_imgs // 2
    p1 = max(0, middle - num_imgs2)
    p2 = min(len(files), middle + num_imgs2)
    img3d = np.stack([load_dicom_image(f, rotate=rotate) for f in files[p1:p2]]).T

    if img3d.shape[-1] < num_imgs:
        n_zero_front = np.zeros((img_size, img_size, (num_imgs - img3d.shape[-1]) // 2))
        n_zero_back = np.zeros(
            (img_size, img_size, num_imgs - img3d.shape[-1] - n_zero_front.shape[-1])
        )
        img3d = np.concatenate((n_zero_front, img3d, n_zero_back), axis=-1)

    if np.min(img3d) < np.max(img3d):
        img3d = img3d - np.min(img3d)
        img3d = img3d / np.max(img3d)

    return np.expand_dims(img3d, 0)


def load_dicom_images_3d_all(scan_id, split):
    img3d_all = np.concatenate(
        [load_dicom_images_3d(scan_id, split, mri_type) for mri_type in mri_types],
        axis=-1,
    )
    return img3d_all


try:
    a = load_dicom_images_3d_all("00000", "train")
    print(a.shape, np.min(a), np.max(a), np.mean(a), np.median(a))
except Exception as e:
    print("Skipping full 3D load during script execution:", e)



## === cell 4
from tensorflow.keras.utils import Sequence


class Dataset(Sequence):
    def __init__(self, df, split, is_train=True, batch_size=BATCH_SIZE, shuffle=True):
        self.idx = df["BraTS21ID"].values
        self.paths = df["BraTS21ID5"].values
        self.y = df["MGMT_value"].values if "MGMT_value" in df.columns else None
        self.is_train = is_train
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.split = split

    def __len__(self):
        return math.ceil(len(self.idx) / self.batch_size)

    def __getitem__(self, ids):
        batch_paths = self.paths[ids * self.batch_size : (ids + 1) * self.batch_size]
        if self.y is not None:
            batch_y = self.y[ids * self.batch_size : (ids + 1) * self.batch_size]
        try:
            list_x = [load_dicom_images_3d_all(x, self.split) for x in batch_paths]
            batch_X = np.stack(list_x, axis=4)
        except Exception as e:
            batch_X = np.zeros(
                (self.batch_size, IMAGE_SIZE, IMAGE_SIZE, NUM_IMAGES, len(mri_types))
            )
            batch_y = np.zeros(self.batch_size) if self.y is not None else None
        if self.is_train:
            return batch_X, batch_y
        else:
            return batch_X




## === cell 5
df_train, df_valid = sk_model_selection.train_test_split(
    train_df,
    test_size=0.2,
    random_state=12,
    stratify=train_df["MGMT_value"],
)



## === cell 6
df_train



## === cell 7
train_dataset = Dataset(df_train, "train")
valid_dataset = Dataset(df_valid, "train")



## === cell 8
del train_df




## === cell 9
def plot_sample_all(images, label, j):
    plt.figure(figsize=(35, 35))
    for i in range(NUM_IMAGES):
        plt.subplot(15, 15, (i + 1))
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[0, :, :, i, j], cmap="gray")
    plt.show()




## === cell 10
try:
    i = 0
    j = 0
    images, label = train_dataset[i]
    plot_sample_all(images, label, j)
except Exception as e:
    print("Skipping sample plot (cell 15):", e)



## === cell 11
try:
    i = 0
    j = 1
    images, label = train_dataset[i]
    plot_sample_all(images, label, j)
except Exception as e:
    print("Skipping sample plot (cell 16):", e)



## === cell 12
try:
    i = 0
    j = 2
    images, label = train_dataset[i]
    plot_sample_all(images, label, j)
except Exception as e:
    print("Skipping sample plot (cell 17):", e)



## === cell 13
try:
    i = 0
    j = 3
    images, label = train_dataset[i]
    plot_sample_all(images, label, j)
except Exception as e:
    print("Skipping sample plot (cell 18):", e)




## === cell 14
def plot_sample_train(images, label):
    plt.figure(figsize=(15, 15))
    idx_base = int(NUM_IMAGES_PER_TYPE / 2)
    idx = [idx_base, idx_base * 3, idx_base * 5, idx_base * 7]
    for i in range(len(idx) * BATCH_SIZE):
        plt.subplot(BATCH_SIZE, len(idx), i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        j = int(i / len(idx))
        plt.imshow(images[0, :, :, idx[i % len(idx)], j], cmap="gray")
        plt.xlabel(f"{idx[i%len(idx)]} {label[j]}")
    plt.show()




## === cell 15
try:
    i = 0
    images, label = train_dataset[i]
    print("Dimension of the CT scan is:", images.shape)
    print("label=", label, label.shape)
    plot_sample_train(images, label)
except Exception as e:
    print("Skipping train sample display (cell 22):", e)




## === cell 16
def get_model(width=IMAGE_SIZE, height=IMAGE_SIZE, depth=NUM_IMAGES):
    """Build a 3D convolutional neural network model."""
    if tf is None:
        return None
    inputs = keras.Input((width, height, depth, 1))
    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=128, activation="relu")(x)
    x = layers.Dropout(0.6)(x)
    outputs = layers.Dense(units=1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs, name="3D_CNN")
    return model




## === cell 17
model = get_model()
if model is not None:
    model.summary()
else:
    print("Model not created due to missing TensorFlow.")



## === cell 18
if model is not None:
    try:
        model.load_weights(
            "../input/brain-tumor-model-v3-2-7/Brain_Tumor_All_MRI_3D_CNN_v3_2_7.h5"
        )
    except Exception as e:
        print("Pretrained weights not found, proceeding without them:", e)




## === cell 19
class Dataset(Sequence):
    def __init__(self, df, split, is_train=True, batch_size=1, shuffle=True):
        self.idx = df["BraTS21ID"].values
        self.paths = df["BraTS21ID5"].values
        self.y = df["MGMT_value"].values if "MGMT_value" in df.columns else None
        self.is_train = is_train
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.split = split

    def __len__(self):
        return math.ceil(len(self.idx) / self.batch_size)

    def __getitem__(self, ids):
        id_path = self.paths[ids]
        split = self.split
        if self.y is not None:
            batch_y = self.y[ids * self.batch_size : (ids + 1) * self.batch_size]
        try:
            list_x = load_dicom_images_3d_all(id_path, split)
            batch_X = np.stack(list_x)
        except Exception as e:
            batch_X = np.zeros(
                (self.batch_size, IMAGE_SIZE, IMAGE_SIZE, NUM_IMAGES, len(mri_types))
            )
            batch_y = np.zeros(self.batch_size) if self.y is not None else None
        if self.is_train:
            return batch_X, batch_y
        else:
            return batch_X

    def on_epoch_end(self):
        if self.shuffle and self.is_train:
            ids_y = list(zip(self.idx, self.y))
            shuffle(ids_y)
            self.idx, self.y = list(zip(*ids_y))




## === cell 20
test_dataset = Dataset(df=test, split="test", is_train=False)




## === cell 21
def plot_sample_test(images):
    plt.figure(figsize=(16, 16))
    idx_base = int(NUM_IMAGES_PER_TYPE / 2)
    idx = [idx_base, idx_base * 3, idx_base * 5, idx_base * 7]
    for i in range(len(idx)):
        plt.subplot(1, len(idx), i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[0, :, :, idx[i]], cmap="gray")
        plt.xlabel(idx[i])
    plt.show()




## === cell 22
try:
    i = 0
    images = test_dataset[i]
    print("Dimension of the CT scan is:", images.shape)
    plot_sample_test(images)
except Exception as e:
    print("Skipping test sample display (cell 36):", e)



## === cell 23
if model is not None:
    try:
        predictions = model.predict(test_dataset)
    except Exception as e:
        print("Model prediction failed, using fallback:", e)
        predictions = None
else:
    predictions = None

if predictions is None:
    baseline = train_df["MGMT_value"].mean()
    predictions = np.full(len(test), baseline, dtype=np.float32)



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2059750288.py in <cell line: 0>()
     11 if predictions is None:
     12     # Simple baseline: mean MGMT_value from training data.
---> 13     baseline = train_df["MGMT_value"].mean()
     14     predictions = np.full(len(test), baseline, dtype=np.float32)
     15 

NameError: name 'train_df' is not defined

## === cell 24
predictions = predictions.reshape(-1)
predictions



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2424291027.py in <cell line: 0>()
----> 1 predictions = predictions.reshape(-1)
      2 predictions
      3 

AttributeError: 'NoneType' object has no attribute 'reshape'

## === cell 25
submission = pd.DataFrame(
    {"BraTS21ID": sample_submission["BraTS21ID"], "MGMT_value": predictions}
)



## === cell 26
submission



## === cell 27
submission["BraTS21ID"] = [format(x, "05d") for x in submission.BraTS21ID]



## === cell 28
submission



## === cell 29
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
