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
import os, sys, warnings

warnings.filterwarnings("ignore")

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")




## === cell 1
import re
import glob
import math
import random as rn
from pathlib import Path

import numpy as np
import pandas as pd
import cv2

import matplotlib.pyplot as plt

import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.metrics import AUC

from sklearn.model_selection import StratifiedKFold
from random import shuffle

rn.seed(30)
np.random.seed(30)
tf.random.set_seed(30)

print("TensorFlow:", tf.__version__)
print("pydicom:", pydicom.__version__)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
config = {
    "images_source_path": "../input/rsna-miccai-brain-tumor-radiogenomic-classification/train",
    "test_images_source_path": "../input/rsna-miccai-brain-tumor-radiogenomic-classification/test",
    "csv_path": "../input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv",
    "data_path": "../input/rsna-miccai-brain-tumor-radiogenomic-classification",
    "output_path": "./crnn/",
    "nfolds": 3,
    "global_seed": 42,
    "batch_size": 4,
    "frames_per_seq": 12,
    "img_size": 224,
    "learning_rate": 0.0001,
    "num_epochs": 10,
    "channels": 3,
    "scale": 0.75,
}
mri_types = ["T2w"]

config["workers"] = max(1, min(4, (os.cpu_count() or 2) // 2))
config["use_multiprocessing"] = True
config["max_queue_size"] = 16




## === cell 3
df_data = pd.read_csv(config["csv_path"])
df_data["folder_name"] = [format(int(x), "05d") for x in df_data["BraTS21ID"]]
df_data["folder_path"] = [
    os.path.join(config["images_source_path"], x) for x in df_data["folder_name"]
]

bad_ids = set([109, 123, 709])
df_data = df_data[~df_data["BraTS21ID"].isin(bad_ids)].reset_index(drop=True)

skf = StratifiedKFold(
    n_splits=config["nfolds"], shuffle=True, random_state=config["global_seed"]
)
df_data["fold"] = -1
for index, (_, val_index) in enumerate(
    skf.split(X=df_data.index, y=df_data.MGMT_value)
):
    df_data.loc[val_index, "fold"] = index

train_df = df_data[df_data.fold != 0].reset_index(drop=True)
valid_df = df_data[df_data.fold == 0].reset_index(drop=True)

print("Train size:", len(train_df), "Valid size:", len(valid_df))




## === cell 4
class Dataset(tf.keras.utils.Sequence):
    def __init__(
        self, df, is_train=True, batch_size=config["batch_size"], shuffle=True
    ):
        self.idx = df["BraTS21ID"].values
        self.paths = df["folder_path"].values
        self.y = df["MGMT_value"].values if "MGMT_value" in df.columns else None
        self.is_train = is_train
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.df = df

        self._path_cache = {}
        self._slice_cache = {}

    def __len__(self):
        return math.ceil(len(self.idx) / self.batch_size)

    def __getitem__(self, batch_index):
        batch_paths = self.paths[
            batch_index * self.batch_size : (batch_index + 1) * self.batch_size
        ]

        if self.is_train:
            batch_y = self.y[
                batch_index * self.batch_size : (batch_index + 1) * self.batch_size
            ].astype(np.float32)
            list_x = [self.load_dicom_images_3d(x, split="train") for x in batch_paths]
            batch_X = np.stack(list_x, axis=0)
            batch_X = np.transpose(batch_X, (0, 2, 3, 4, 1)).astype(np.float32)
            return batch_X, batch_y
        else:
            list_x = [self.load_dicom_images_3d(x, split="test") for x in batch_paths]
            batch_X = np.stack(list_x, axis=0)
            batch_X = np.transpose(batch_X, (0, 2, 3, 4, 1)).astype(np.float32)
            return batch_X

    def load_dicom_images_3d(
        self,
        scan_path,
        num_imgs=config["frames_per_seq"],
        img_size=config["img_size"],
        mri_type=mri_types[0],
        split="train",
        rotate=0,
    ):
        target_file_paths = self.get_img_path_3d(scan_path, mri_type)

        if len(target_file_paths) == 0:
            img3d = np.zeros((img_size, img_size, num_imgs), dtype=np.float32)
            return np.expand_dims(img3d, 0)

        imgs = [self.read_mri(f) for f in target_file_paths]
        img3d = np.stack(imgs, axis=-1)  # (H, W, depth)

        if img3d.shape[-1] < num_imgs:
            n_zero = np.zeros(
                (img_size, img_size, num_imgs - img3d.shape[-1]), dtype=img3d.dtype
            )
            img3d = np.concatenate((img3d, n_zero), axis=-1)
        elif img3d.shape[-1] > num_imgs:
            img3d = img3d[..., :num_imgs]

        mn, mx = np.min(img3d), np.max(img3d)
        if mn < mx:
            img3d = (img3d - mn) / (mx - mn)
        else:
            img3d = np.zeros_like(img3d, dtype=np.float32)

        return np.expand_dims(img3d.astype(np.float32), 0)  # (1, H, W, D)

    def read_mri(self, path, voi_lut=True, fix_monochrome=True):
        cached = self._slice_cache.get(path)
        if cached is not None:
            return cached

        dicom = pydicom.dcmread(path)

        if voi_lut:
            data = apply_voi_lut(dicom.pixel_array, dicom)
        else:
            data = dicom.pixel_array

        if (
            fix_monochrome
            and getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1"
        ):
            data = np.amax(data) - data

        data = data.astype(np.float32)
        mn, mx = np.min(data), np.max(data)
        if mn < mx:
            data = (data - mn) / (mx - mn)
        else:
            data = np.zeros_like(data, dtype=np.float32)

        data = (data * 255.0).astype(np.uint8)
        data = cv2.resize(
            data, (config["img_size"], config["img_size"]), interpolation=cv2.INTER_AREA
        )

        self._slice_cache[path] = data
        return data

    def _fast_dcm_sort_key(self, filepath):
        base = os.path.basename(filepath)
        try:
            num = int(base.split("-")[-1].split(".")[0])
        except Exception:
            num = base
        return num

    def get_img_path_3d(self, scan_path, mri_type):
        cache_key = (scan_path, mri_type)
        cached = self._path_cache.get(cache_key)
        if cached is None:
            modality_path = os.path.join(scan_path, mri_type)
            files = glob.glob(f"{modality_path}/*.dcm")
            if len(files) == 0:
                self._path_cache[cache_key] = []
                return []
            files.sort(key=self._fast_dcm_sort_key)
            self._path_cache[cache_key] = files
            cached = files

        files = cached
        total_img_num = len(files)
        mid_num = total_img_num // 2
        num_3d2 = config["frames_per_seq"] // 2
        start_idx = max(0, mid_num - num_3d2)
        end_idx = min(total_img_num, mid_num + num_3d2)
        return files[start_idx:end_idx]

    def on_epoch_end(self):
        if self.shuffle and self.is_train and self.y is not None:
            ids_y = list(zip(self.paths, self.y))
            shuffle(ids_y)
            self.paths, self.y = list(zip(*ids_y))
            self.paths = np.array(self.paths)
            self.y = np.array(self.y)


train_dataset = Dataset(
    train_df, is_train=True, batch_size=config["batch_size"], shuffle=True
)
valid_dataset = Dataset(
    valid_df, is_train=True, batch_size=config["batch_size"], shuffle=False
)

images, label = train_dataset[0]
print("Batch X:", images.shape, images.dtype, "Batch y:", label.shape, label[:5])




## === cell 5
def get_3d_model(
    width=config["img_size"], height=config["img_size"], depth=config["frames_per_seq"]
):
    """Build a 3D convolutional neural network model."""
    inputs = keras.Input((width, height, depth, 1))

    x = Conv3D(filters=32, kernel_size=3, padding="same", activation="relu")(inputs)
    x = MaxPool3D(pool_size=2)(x)
    x = BatchNormalization()(x)

    x = Conv3D(filters=32, kernel_size=3, padding="same", activation="relu")(x)
    x = MaxPool3D(pool_size=2)(x)
    x = BatchNormalization()(x)

    x = Conv3D(filters=64, kernel_size=3, padding="same", activation="relu")(x)
    x = MaxPool3D(pool_size=2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.01)(x)

    x = Conv3D(filters=128, kernel_size=3, padding="same", activation="relu")(x)
    x = MaxPool3D(pool_size=2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.02)(x)

    x = Conv3D(filters=256, kernel_size=3, padding="same", activation="relu")(x)
    x = MaxPool3D(pool_size=2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.03)(x)

    x = GlobalAveragePooling3D()(x)
    x = Dense(units=1024, activation="relu")(x)
    x = Dropout(0.08)(x)

    outputs = Dense(units=1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


model = get_3d_model()
model.summary()




## === cell 6
initial_learning_rate = config["learning_rate"]
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)

model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=[AUC(name="auc"), "acc"],
)

history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=config["num_epochs"],
    shuffle=True,
    verbose=1,
)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3788668802.py in <cell line: 0>()
     11 
     12 # Fix: Keras 3 `model.fit()` no longer accepts workers/use_multiprocessing/max_queue_size.
---> 13 history = model.fit(
     14     train_dataset,
     15     validation_data=valid_dataset,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    122             raise e.with_traceback(filtered_tb) from None
    123         finally:
--> 124             del filtered_tb
    125 
    126     return error_handler

ValueError: Exception encountered when calling MaxPooling3D.call().

Negative dimension size caused by subtracting 2 from 1 for '{{node 3dcnn_1/max_pooling3d_3_1/MaxPool3D}} = MaxPool3D[T=DT_FLOAT, data_format="NDHWC", ksize=[1, 2, 2, 2, 1], padding="VALID", strides=[1, 2, 2, 2, 1]](3dcnn_1/conv3d_3_1/Relu)' with input shapes: [?,28,28,1,128].

Arguments received by MaxPooling3D.call():
  • inputs=tf.Tensor(shape=(None, 28, 28, 1, 128), dtype=float32)

## === cell 7
sample_submission_path = os.path.join(config["data_path"], "sample_submission.csv")
sample_df = pd.read_csv(sample_submission_path)

test_df = sample_df.copy()
test_df["folder_name"] = [format(int(x), "05d") for x in test_df.BraTS21ID]
test_df["folder_path"] = [
    os.path.join(config["data_path"], "test", x) for x in test_df["folder_name"]
]

test_dataset = Dataset(test_df, is_train=False, batch_size=1, shuffle=False)

preds = model.predict(
    test_dataset,
    verbose=1,
).reshape(-1)

submission = pd.DataFrame(
    {"BraTS21ID": sample_df["BraTS21ID"].values, "MGMT_value": preds.astype(np.float32)}
)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2392279443.py in <cell line: 0>()
     11 
     12 # Fix: Keras 3 `model.predict()` no longer accepts workers/use_multiprocessing/max_queue_size.
---> 13 preds = model.predict(
     14     test_dataset,
     15     verbose=1,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    122             raise e.with_traceback(filtered_tb) from None
    123         finally:
--> 124             del filtered_tb
    125 
    126     return error_handler

ValueError: Exception encountered when calling MaxPooling3D.call().

Negative dimension size caused by subtracting 2 from 1 for '{{node 3dcnn_1/max_pooling3d_3_1/MaxPool3D}} = MaxPool3D[T=DT_FLOAT, data_format="NDHWC", ksize=[1, 2, 2, 2, 1], padding="VALID", strides=[1, 2, 2, 2, 1]](3dcnn_1/conv3d_3_1/Relu)' with input shapes: [1,28,28,1,128].

Arguments received by MaxPooling3D.call():
  • inputs=tf.Tensor(shape=(1, 28, 28, 1, 128), dtype=float32)

## === cell 8
print(submission.head())
print(submission.shape)
print(submission.isna().sum())




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/421017399.py in <cell line: 0>()
----> 1 print(submission.head())
      2 print(submission.shape)
      3 print(submission.isna().sum())
      4 
      5 

NameError: name 'submission' is not defined

## === cell 9
submission = submission[["BraTS21ID", "MGMT_value"]].copy()
submission["MGMT_value"] = submission["MGMT_value"].astype(np.float32).clip(0.0, 1.0)

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with", len(submission), "rows")
print("Saved to:", os.path.abspath("submission.csv"))

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3892087187.py in <cell line: 0>()
      1 # Safety: ensure correct column order and dtype; then write a valid Kaggle submission.
----> 2 submission = submission[["BraTS21ID", "MGMT_value"]].copy()
      3 submission["MGMT_value"] = submission["MGMT_value"].astype(np.float32).clip(0.0, 1.0)
      4 
      5 submission.to_csv("submission.csv", index=False)

NameError: name 'submission' is not defined
