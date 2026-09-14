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


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    pydicom.config.use_gdcm = False
except Exception:
    pass
try:
    pydicom.config.use_pylibjpeg = False
except Exception:
    pass

print("tf:", tf.__version__)
print("keras:", keras.__version__)
print("pydicom:", pydicom.__version__)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
data_directory = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"

mri_types = ["FLAIR", "T1w", "T1wCE", "T2w"]
IMAGE_SIZE = 256
NUM_IMAGES = 64

train_labels_path = os.path.join(data_directory, "train_labels.csv")
sample_sub_path = os.path.join(data_directory, "sample_submission.csv")

assert os.path.exists(train_labels_path), f"Missing: {train_labels_path}"
assert os.path.exists(sample_sub_path), f"Missing: {sample_sub_path}"




## === cell 2
sample_submission = pd.read_csv(sample_sub_path)
test = sample_submission.copy()
test["BraTS21ID5"] = [format(int(x), "05d") for x in test.BraTS21ID]

train_df = pd.read_csv(train_labels_path)
train_df["BraTS21ID5"] = [format(int(x), "05d") for x in train_df.BraTS21ID]

bad_ids = {"00109", "00123", "00709"}
train_df = train_df[~train_df["BraTS21ID5"].isin(bad_ids)].reset_index(drop=True)

train_df.head(3), test.head(3)




## === cell 3
_file_list_cache = {}

_num_re = re.compile(r"(\d+)")


def _file_num_key(name: str) -> int:
    m = _num_re.search(name)
    return int(m.group(1)) if m else -1


def _list_dcm_files(scan_id: str, split: str, mri_type: str):
    key = (split, scan_id, mri_type)
    out = _file_list_cache.get(key)
    if out is not None:
        return out
    folder = f"{data_directory}/{split}/{scan_id}/{mri_type}"
    try:
        entries = []
        with os.scandir(folder) as it:
            for e in it:
                if e.is_file() and e.name.endswith(".dcm"):
                    entries.append(e.path)
        entries.sort(key=lambda p: _file_num_key(os.path.basename(p)))
    except FileNotFoundError:
        entries = []
    _file_list_cache[key] = entries
    return entries


def load_dicom_image(path, img_size=IMAGE_SIZE, voi_lut=True, rotate=0):
    dicom = pydicom.dcmread(path, stop_before_pixels=True)
    dicom = pydicom.dcmread(path)  # ensure pixel data is available deterministically
    data = dicom.pixel_array
    if voi_lut:
        data = apply_voi_lut(data, dicom)

    if rotate > 0:
        rot_choices = [
            None,
            cv2.ROTATE_90_CLOCKWISE,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
            cv2.ROTATE_180,
        ]
        data = cv2.rotate(data, rot_choices[rotate])

    data = cv2.resize(data, (img_size, img_size), interpolation=cv2.INTER_AREA)
    data = data.astype(np.float32, copy=False)
    return data


def load_dicom_images_3d(
    scan_id,
    num_imgs=NUM_IMAGES,
    img_size=IMAGE_SIZE,
    mri_type="FLAIR",
    split="test",
    rotate=0,
):
    files = _list_dcm_files(str(scan_id), split=split, mri_type=mri_type)

    if len(files) == 0:
        img3d = np.zeros((img_size, img_size, num_imgs), dtype=np.float32)
        return np.expand_dims(img3d, 0)

    middle = len(files) // 2
    num_imgs2 = num_imgs // 2
    p1 = max(0, middle - num_imgs2)
    p2 = min(len(files), middle + num_imgs2)

    if p2 <= p1:
        p1 = max(0, min(len(files) - 1, middle))
        p2 = p1 + 1

    sel = files[p1:p2]
    d = len(sel)
    vol = np.empty((img_size, img_size, d), dtype=np.float32)
    for j, f in enumerate(sel):
        vol[:, :, j] = load_dicom_image(f, img_size=img_size, rotate=rotate)

    img3d = vol  # (H, W, D)

    if img3d.shape[-1] < num_imgs:
        n_zero = np.zeros(
            (img_size, img_size, num_imgs - img3d.shape[-1]), dtype=img3d.dtype
        )
        img3d = np.concatenate((img3d, n_zero), axis=-1)
    elif img3d.shape[-1] > num_imgs:
        img3d = img3d[:, :, :num_imgs]

    mn, mx = float(np.min(img3d)), float(np.max(img3d))
    if mn < mx:
        img3d = (img3d - mn) / (mx - mn)

    return np.expand_dims(img3d, 0)  # (1, H, W, D)




## === cell 4
a = load_dicom_images_3d("00002", split="test", mri_type="FLAIR")
print(
    "Loaded test scan volume shape:",
    a.shape,
    "min/max:",
    float(a.min()),
    float(a.max()),
)




## === cell 5
from tensorflow.keras.utils import Sequence

_VOL_CACHE = {}
_VOL_CACHE_MAX = 256  # bounded to avoid OOM


def _cache_get(key):
    return _VOL_CACHE.get(key)


def _cache_put(key, value):
    if key in _VOL_CACHE:
        _VOL_CACHE[key] = value
        return
    if len(_VOL_CACHE) >= _VOL_CACHE_MAX:
        _VOL_CACHE.pop(next(iter(_VOL_CACHE)))
    _VOL_CACHE[key] = value


class Dataset(Sequence):
    def __init__(
        self,
        df,
        is_train=True,
        batch_size=1,
        shuffle=True,
        mri_type="FLAIR",
        split="test",
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
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.mri_type = mri_type
        self.split = split

        self._batch_X = np.empty(
            (self.batch_size, IMAGE_SIZE, IMAGE_SIZE, NUM_IMAGES, 1), dtype=np.float32
        )
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.paths) / self.batch_size)

    def __getitem__(self, i):
        bs = self.batch_size
        start = i * bs
        end = min((i + 1) * bs, len(self.paths))
        batch_paths = self.paths[start:end]
        b = end - start

        batch_X = self._batch_X[:b]
        split = self.split
        mri_type = self.mri_type
        img_size = IMAGE_SIZE
        num_imgs = NUM_IMAGES

        for k, sid in enumerate(batch_paths):
            sid = str(sid)
            cache_key = (split, mri_type, sid, img_size, num_imgs)
            vol = _cache_get(cache_key)
            if vol is None:
                vol = load_dicom_images_3d(
                    sid,
                    mri_type=mri_type,
                    split=split,
                    img_size=img_size,
                    num_imgs=num_imgs,
                )[
                    0
                ]  # (H,W,D)
                _cache_put(cache_key, vol)
            batch_X[k, ..., 0] = vol

        if self.is_train:
            batch_y = self.y[start:end].astype(np.float32, copy=False)
            return batch_X, batch_y
        else:
            return batch_X

    def on_epoch_end(self):
        if self.shuffle and self.is_train:
            perm = np.random.permutation(len(self.paths))
            self.paths = self.paths[perm]
            self.idx = self.idx[perm]
            if self.y is not None:
                self.y = self.y[perm]




## === cell 6
train_split, val_split = sk_model_selection.train_test_split(
    train_df,
    test_size=0.2,
    random_state=SEED,
    stratify=train_df["MGMT_value"],
)

BATCH_SIZE = 2  # keep small for memory/runtime
train_dataset = Dataset(
    train_split,
    is_train=True,
    batch_size=BATCH_SIZE,
    shuffle=True,
    mri_type="FLAIR",
    split="train",
)
val_dataset = Dataset(
    val_split,
    is_train=True,
    batch_size=BATCH_SIZE,
    shuffle=False,
    mri_type="FLAIR",
    split="train",
)
test_dataset = Dataset(
    test, is_train=False, batch_size=1, shuffle=False, mri_type="FLAIR", split="test"
)

xb, yb = train_dataset[0]
print("Train batch X shape:", xb.shape, "y shape:", yb.shape)




## === cell 7
def build_3d_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, NUM_IMAGES, 1)):
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv3D(8, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.Conv3D(16, 3, padding="same", activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.Conv3D(32, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


model = build_3d_model()
model.summary()




## === cell 8
EPOCHS = 2


def _mp_worker_init():
    random.seed(SEED)
    np.random.seed(SEED)


history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    verbose=1,
    workers=min(4, (os.cpu_count() or 2)),
    use_multiprocessing=True,
    max_queue_size=8,
)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3797570175.py in <cell line: 0>()
      9 
     10 
---> 11 history = model.fit(
     12     train_dataset,
     13     validation_data=val_dataset,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    117             return fn(*args, **kwargs)
    118         except Exception as e:
--> 119             filtered_tb = _process_traceback_frames(e.__traceback__)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`

TypeError: TensorFlowTrainer.fit() got an unexpected keyword argument 'workers'

## === cell 9
preds = model.predict(
    test_dataset,
    verbose=1,
    workers=min(4, (os.cpu_count() or 2)),
    use_multiprocessing=True,
    max_queue_size=8,
).reshape(-1)
preds = np.clip(preds, 0.0, 1.0)
print(
    "Preds:",
    preds[:10],
    "min/max:",
    float(preds.min()),
    float(preds.max()),
    "n:",
    len(preds),
)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1670457459.py in <cell line: 0>()
      1 # CHANGE (timeout fix): Use the same parallel prefetching for predict to speed up test-time DICOM loading.
----> 2 preds = model.predict(
      3     test_dataset,
      4     verbose=1,
      5     workers=min(4, (os.cpu_count() or 2)),

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    117             return fn(*args, **kwargs)
    118         except Exception as e:
--> 119             filtered_tb = _process_traceback_frames(e.__traceback__)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`

TypeError: TensorFlowTrainer.predict() got an unexpected keyword argument 'workers'

## === cell 10
submission = pd.DataFrame(
    {"BraTS21ID": sample_submission["BraTS21ID"].values, "MGMT_value": preds}
)
assert submission.shape[0] == sample_submission.shape[0]
assert list(submission.columns) == ["BraTS21ID", "MGMT_value"]

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
submission.head()




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4180406590.py in <cell line: 0>()
      1 submission = pd.DataFrame(
----> 2     {"BraTS21ID": sample_submission["BraTS21ID"].values, "MGMT_value": preds}
      3 )
      4 assert submission.shape[0] == sample_submission.shape[0]
      5 assert list(submission.columns) == ["BraTS21ID", "MGMT_value"]

NameError: name 'preds' is not defined

## === cell 11
plt.figure(figsize=(5, 5))
plt.hist(submission["MGMT_value"], bins=20)
plt.title("Prediction distribution")
plt.show()

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4002217871.py in <cell line: 0>()
      1 plt.figure(figsize=(5, 5))
----> 2 plt.hist(submission["MGMT_value"], bins=20)
      3 plt.title("Prediction distribution")
      4 plt.show()

NameError: name 'submission' is not defined
