# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PYTHONHASHSEED"] = "42"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys
import json
import glob
import random
import collections
import time
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

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

try:
    _CPU = os.cpu_count() or 2
    tf.config.threading.set_intra_op_parallelism_threads(max(1, _CPU // 2))
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass

print("Python:", sys.version)
print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)



## === cell 1
data_directory = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"
pytorch3dpath = "../input/efficientnetpyttorch3d/EfficientNet-PyTorch-3D"

mri_types = ["FLAIR", "T1w", "T1wCE", "T2w"]
IMAGE_SIZE = 256
NUM_IMAGES = 64



## === cell 2
sample_submission = pd.read_csv(
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv"
)
test = sample_submission.copy()
test["BraTS21ID5"] = [format(int(x), "05d") for x in test.BraTS21ID]
test.head(3)



## === cell 3
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

_FILES_CACHE = {}  # key: (split, scan_id, mri_type) -> sorted list of paths


def _image_num_key_fast(path_str: str) -> int:
    base = os.path.basename(path_str)
    try:
        if base.startswith("Image-") and base.lower().endswith(".dcm"):
            return int(base[6:-4])
    except Exception:
        pass
    nums = []
    cur = 0
    in_num = False
    for ch in base:
        if "0" <= ch <= "9":
            cur = cur * 10 + (ord(ch) - 48)
            in_num = True
        elif in_num:
            nums.append(cur)
            cur = 0
            in_num = False
    if in_num:
        nums.append(cur)
    return nums[-1] if nums else 0


def _build_dicom_index(
    root_dir: str, splits=("train", "test"), mri_types=("FLAIR", "T1w", "T1wCE", "T2w")
):
    index = {}
    for split in splits:
        base = os.path.join(root_dir, split)
        if not os.path.isdir(base):
            continue
        with os.scandir(base) as it_scan:
            for ent_scan in it_scan:
                if not ent_scan.is_dir():
                    continue
                scan_id = ent_scan.name
                scan_path = ent_scan.path
                for mri_type in mri_types:
                    ddir = os.path.join(scan_path, mri_type)
                    if not os.path.isdir(ddir):
                        index[(split, scan_id, mri_type)] = []
                        continue
                    files = []
                    with os.scandir(ddir) as it_files:
                        for ent_f in it_files:
                            if ent_f.is_file() and ent_f.name.lower().endswith(".dcm"):
                                files.append(ent_f.path)
                    if files:
                        files.sort(key=_image_num_key_fast)
                    index[(split, scan_id, mri_type)] = files
    return index


_FILES_CACHE.update(
    _build_dicom_index(
        data_directory, splits=("train", "test"), mri_types=tuple(mri_types)
    )
)


def _get_sorted_dicom_files(scan_id: str, split: str, mri_type: str):
    key = (split, scan_id, mri_type)
    files = _FILES_CACHE.get(key)
    if files is not None:
        return files
    pattern = f"{data_directory}/{split}/{scan_id}/{mri_type}/*.dcm"
    files = glob.glob(pattern)
    if files:
        files.sort(key=_image_num_key_fast)
    _FILES_CACHE[key] = files
    return files


pydicom.config.image_handlers = [pydicom.pixel_data_handlers.numpy_handler]

_DICOM_THREADS = max(1, min(8, (os.cpu_count() or 2)))
_DICOM_POOL = ThreadPoolExecutor(max_workers=_DICOM_THREADS)


@lru_cache(maxsize=1024)
def _load_dicom_image_cached(path: str, img_size: int, voi_lut: bool, rotate: int):
    dicom = pydicom.dcmread(
        path,
        stop_before_pixels=False,
        force=True,
        specific_tags=[
            "PixelData",
            "BitsAllocated",
            "BitsStored",
            "HighBit",
            "PixelRepresentation",
            "SamplesPerPixel",
            "PhotometricInterpretation",
            "Rows",
            "Columns",
            "RescaleIntercept",
            "RescaleSlope",
            "WindowCenter",
            "WindowWidth",
            "VOILUTSequence",
        ],
    )

    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    data = data.astype(np.float32, copy=False)

    if rotate > 0:
        rot_choices = [
            0,
            cv2.ROTATE_90_CLOCKWISE,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
            cv2.ROTATE_180,
        ]
        data = cv2.rotate(data, rot_choices[rotate])

    data = cv2.resize(data, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return data


def load_dicom_image(path, img_size=IMAGE_SIZE, voi_lut=True, rotate=0):
    return _load_dicom_image_cached(path, int(img_size), bool(voi_lut), int(rotate))


def _load_one_slice(args):
    f, img_size, rotate = args
    return load_dicom_image(f, img_size=img_size, rotate=rotate)


def load_dicom_images_3d(
    scan_id,
    num_imgs=NUM_IMAGES,
    img_size=IMAGE_SIZE,
    mri_type="FLAIR",
    split="test",
    rotate=0,
):
    files = _get_sorted_dicom_files(scan_id, split=split, mri_type=mri_type)

    if len(files) == 0:
        img3d = np.zeros((img_size, img_size, num_imgs), dtype=np.float32)
        return np.expand_dims(img3d, 0)

    middle = len(files) // 2
    num_imgs2 = num_imgs // 2
    p1 = max(0, middle - num_imgs2)
    p2 = min(len(files), middle + num_imgs2)

    selected = files[p1:p2]
    if len(selected) == 0:
        selected = files[: min(len(files), num_imgs)]

    args = [(f, img_size, rotate) for f in selected]
    slices = list(_DICOM_POOL.map(_load_one_slice, args))

    img3d = np.stack(slices, axis=0).transpose(1, 2, 0)  # (H,W,D)

    if img3d.shape[-1] < num_imgs:
        n_zero = np.zeros(
            (img_size, img_size, num_imgs - img3d.shape[-1]), dtype=np.float32
        )
        img3d = np.concatenate((img3d, n_zero), axis=-1)
    elif img3d.shape[-1] > num_imgs:
        img3d = img3d[:, :, :num_imgs]

    mn, mx = float(np.min(img3d)), float(np.max(img3d))
    if mn < mx:
        img3d = (img3d - mn) / (mx - mn)

    return np.expand_dims(img3d.astype(np.float32, copy=False), 0)  # (1,H,W,D)




## === cell 4
a = load_dicom_images_3d("00002", split="test")
print(a.shape, np.min(a), np.max(a), np.mean(a))




## === cell 5
def plot_slices(num_rows, num_columns, width, height, data):
    """Plot a montage of slices"""
    data = np.rot90(np.array(data))
    data = np.transpose(data)
    data = np.reshape(data, (num_rows, num_columns, width, height))
    rows_data, columns_data = data.shape[0], data.shape[1]
    heights = [slc[0].shape[0] for slc in data]
    widths = [slc.shape[1] for slc in data[0]]
    fig_width = 12.0
    fig_height = fig_width * sum(heights) / sum(widths)
    f, axarr = plt.subplots(
        rows_data,
        columns_data,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": heights},
    )
    for i in range(rows_data):
        for j in range(columns_data):
            axarr[i, j].imshow(data[i][j], cmap="gray")
            axarr[i, j].axis("off")
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.show()




## === cell 6
from tensorflow.keras.utils import Sequence


class Dataset(Sequence):
    def __init__(
        self,
        df,
        is_train=True,
        batch_size=1,
        shuffle=True,
        split="test",
        mri_type="FLAIR",
    ):
        self.df = df.reset_index(drop=True)
        self.idx = self.df["BraTS21ID"].values
        self.paths = self.df["BraTS21ID5"].values
        self.y = (
            self.df["MGMT_value"].values
            if ("MGMT_value" in self.df.columns and is_train)
            else None
        )
        self.is_train = is_train
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.split = split
        self.mri_type = mri_type
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.idx) / self.batch_size)

    def __getitem__(self, batch_index):
        batch_paths = self.paths[
            batch_index * self.batch_size : (batch_index + 1) * self.batch_size
        ]

        batch_X = []
        for scan_id in batch_paths:
            vol = load_dicom_images_3d(
                scan_id, split=self.split, mri_type=self.mri_type
            )  # (1,H,W,D)
            batch_X.append(vol[0])  # (H,W,D)

        batch_X = np.stack(batch_X, axis=0)  # (B,H,W,D)
        batch_X = batch_X[..., np.newaxis]  # (B,H,W,D,1)

        if self.is_train and self.y is not None:
            batch_y = self.y[
                batch_index * self.batch_size : (batch_index + 1) * self.batch_size
            ].astype(np.float32)
            return batch_X, batch_y
        return batch_X

    def on_epoch_end(self):
        if self.shuffle and self.is_train and self.y is not None:
            perm = np.random.permutation(len(self.idx))
            self.idx = self.idx[perm]
            self.paths = self.paths[perm]
            self.y = self.y[perm]




## === cell 7
test_dataset = Dataset(
    test, is_train=False, batch_size=1, shuffle=False, split="test", mri_type="FLAIR"
)

image = test_dataset[0]
print("Batch shape:", image.shape)




## === cell 8
def get_model(width=IMAGE_SIZE, height=IMAGE_SIZE, depth=64):
    """Build a 3D convolutional neural network model."""
    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=32, kernel_size=3, activation="relu", padding="same")(
        inputs
    )
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=32, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Conv3D(filters=512, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=1024, activation="relu")(x)
    x = layers.Dropout(0.1)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


model = get_model(width=IMAGE_SIZE, height=IMAGE_SIZE, depth=NUM_IMAGES)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss="binary_crossentropy"
)
model.summary()



## === cell 9
train_labels = pd.read_csv(
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv"
)
train_labels["BraTS21ID5"] = [format(int(x), "05d") for x in train_labels.BraTS21ID]

bad_ids = set(["00109", "00123", "00709"])
train_labels = train_labels[~train_labels["BraTS21ID5"].isin(bad_ids)].reset_index(
    drop=True
)

train_df, val_df = sk_model_selection.train_test_split(
    train_labels,
    test_size=0.2,
    random_state=42,
    stratify=train_labels["MGMT_value"],
)

train_dataset = Dataset(
    train_df, is_train=True, batch_size=1, shuffle=True, split="train", mri_type="FLAIR"
)
val_dataset = Dataset(
    val_df, is_train=True, batch_size=1, shuffle=False, split="train", mri_type="FLAIR"
)

weights_path = "../input/brain-tumor-3d-classification-weights2/Brain_3d_cls_FLAIR.h5"
if os.path.exists(weights_path):
    print("Pretrained weights found; skipping training to save time:", weights_path)
else:
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=2,
        verbose=1,
    )



## === cell 10
if os.path.exists(weights_path):
    model.load_weights(weights_path)
    print("Loaded weights:", weights_path)
else:
    print("WARNING: weights not found at:", weights_path)
    print("Proceeding with trained-from-scratch model weights.")



## === cell 11
output_signature = tf.TensorSpec(
    shape=(None, IMAGE_SIZE, IMAGE_SIZE, NUM_IMAGES, 1), dtype=tf.float32
)
tf_test_ds = tf.data.Dataset.from_generator(
    lambda: iter(test_dataset), output_signature=output_signature
)
tf_test_ds = tf_test_ds.prefetch(tf.data.AUTOTUNE)

preds = model.predict(
    tf_test_ds,
    verbose=1,
    steps=len(test_dataset),
).reshape(-1)

preds[:10], preds.shape



## === cell 12
submission = pd.DataFrame(
    {"BraTS21ID": sample_submission["BraTS21ID"].values, "MGMT_value": preds}
)

assert len(submission) == len(sample_submission), "Prediction length mismatch"
submission.to_csv("submission.csv", index=False)
submission.head()

print("Wrote submission.csv with shape:", submission.shape)
print("Columns:", list(submission.columns))
print("submission.csv exists:", os.path.exists("submission.csv"))
print(
    "MGMT_value min/max:",
    float(submission.MGMT_value.min()),
    float(submission.MGMT_value.max()),
)
