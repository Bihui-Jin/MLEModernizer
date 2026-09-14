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

0.56588

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61647) has done: 'I fix the environment crash coming from an incompatible protobuf implementation by forcing the pure-Python protobuf backend before importing TensorFlow. Then I fix the Keras 3 API errors by removing unsupported `workers/use_multiprocessing/max_queue_size` arguments from `fit()` and `predict()` so training/inference runs end-to-end. Finally, I keep the same model and data pipeline logic but ensure `preds` and the submission DataFrame are always created, writing a valid `submission.csv` with the exact required columns.'
- What this solution (achieved 0.56588) has done: 'We fix the crash happening before training by removing the protobuf “force python implementation” workaround (it now triggers an incompatible protobuf API path in this environment) and instead force TensorFlow to use the legacy bundled Keras to avoid Keras 3/protobuf-related incompatibilities. We keep the same data pipeline, model architecture, training loop, and prediction logic, only making import-order/environment adjustments needed for stability. We also keep deterministic seeds and ensure the submission is written as `submission.csv` with the exact required columns and alignment. No score-tuning changes are introduced beyond making the code run reliably end-to-end.'
- What this solution (achieved 0.61647) has done: 'We fix the immediate crash caused by an incompatible protobuf runtime by forcing the pure-Python protobuf backend *before* importing TensorFlow (and remove the legacy-keras env var that still leads into the failing protobuf path here). Then we keep the exact same data pipeline/model/training loop, only adding a small safety check to ensure predictions align 1:1 with `sample_submission` length so the submission is always valid. These changes are execution/stability fixes and should also recover the prior working score range by allowing TensorFlow/Keras to run end-to-end in this environment. The script still write `submission.csv` with the required columns.'
- What this solution (achieved 0.56588) has done: 'I fix the immediate crash in the first cell by avoiding the protobuf “force python implementation” workaround that is incompatible with this Kaggle environment, and instead force TensorFlow to use the legacy bundled Keras (which avoids the Keras 3/protobuf path that triggers the `MessageFactory.GetPrototype` error). I keep the rest of the pipeline (DICOM loading, Dataset Sequence, 3D CNN, training loop, and prediction logic) unchanged so score behavior remains essentially the same. I also keep the submission writing exactly as required (`submission.csv` with `BraTS21ID,MGMT_value`) and ensure IDs/predictions stay aligned. Since your target score is `-1.0` (non-actionable for AUC), no score-tuning changes are introduced—this is purely a stability fix to run end-to-end and produce a valid submission.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ["TF_USE_LEGACY_KERAS"] = "1"

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

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("Python:", sys.version)
print("TF:", tf.__version__)
print(
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:",
    os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"),
)
print("TF_USE_LEGACY_KERAS:", os.environ.get("TF_USE_LEGACY_KERAS"))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
data_directory = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"

mri_types = ["FLAIR", "T1w", "T1wCE", "T2w"]
IMAGE_SIZE = 128  # keep memory/time reasonable while preserving 3D pipeline
NUM_IMAGES = 32

BAD_CASES = set(["00109", "00123", "00709"])

print("Data dir exists:", os.path.exists(data_directory))
print("Train dir exists:", os.path.exists(os.path.join(data_directory, "train")))
print("Test dir exists:", os.path.exists(os.path.join(data_directory, "test")))



## === cell 2
sample_submission = pd.read_csv(f"{data_directory}/sample_submission.csv")
test = sample_submission.copy()
test["BraTS21ID5"] = [format(int(x), "05d") for x in test.BraTS21ID]

train_labels = pd.read_csv(f"{data_directory}/train_labels.csv")
train_labels["BraTS21ID5"] = [format(int(x), "05d") for x in train_labels.BraTS21ID]

train_labels = train_labels[~train_labels["BraTS21ID5"].isin(BAD_CASES)].reset_index(
    drop=True
)

print(sample_submission.head(3))
print(train_labels.head(3), "n_train:", len(train_labels), "n_test:", len(test))



## === cell 3
from functools import lru_cache


def _numeric_sort_key(path):
    return [
        int(x) if x.isdigit() else x
        for x in re.findall(r"[^0-9]|[0-9]+", os.path.basename(path))
    ]


def _build_dicom_index(data_dir, split, ids5, mri_types_):
    index = {}
    base = os.path.join(data_dir, split)
    for sid in ids5:
        for mt in mri_types_:
            d = os.path.join(base, sid, mt)
            if not os.path.isdir(d):
                index[(split, sid, mt)] = tuple()
                continue
            files = glob.glob(os.path.join(d, "*.dcm"))
            if not files:
                index[(split, sid, mt)] = tuple()
                continue

            inst_pairs = []
            unknown = []
            for f in files:
                try:
                    ds = pydicom.dcmread(
                        f,
                        stop_before_pixels=True,
                        specific_tags=["InstanceNumber"],
                        force=True,
                    )
                    inst = getattr(ds, "InstanceNumber", None)
                    if inst is None:
                        unknown.append(f)
                    else:
                        inst_pairs.append((int(inst), f))
                except Exception:
                    unknown.append(f)

            inst_pairs.sort(key=lambda x: x[0])
            if unknown:
                unknown.sort(key=_numeric_sort_key)
            ordered = [f for _, f in inst_pairs] + unknown
            index[(split, sid, mt)] = tuple(ordered)
    return index


_DICOM_INDEX = None


@lru_cache(maxsize=50000)
def _get_sorted_dicom_files(split, scan_id, mri_type):
    global _DICOM_INDEX
    if _DICOM_INDEX is not None:
        return _DICOM_INDEX.get((split, scan_id, mri_type), tuple())

    pattern = f"{data_directory}/{split}/{scan_id}/{mri_type}/*.dcm"
    files = glob.glob(pattern)
    if not files:
        return tuple()

    inst_pairs = []
    unknown = []
    for f in files:
        try:
            ds = pydicom.dcmread(
                f, stop_before_pixels=True, specific_tags=["InstanceNumber"], force=True
            )
            inst = getattr(ds, "InstanceNumber", None)
            if inst is None:
                unknown.append(f)
            else:
                inst_pairs.append((int(inst), f))
        except Exception:
            unknown.append(f)

    inst_pairs.sort(key=lambda x: x[0])
    if unknown:
        unknown.sort(key=_numeric_sort_key)
    ordered = [f for _, f in inst_pairs] + unknown
    return tuple(ordered)


def load_dicom_image(path, img_size=IMAGE_SIZE, voi_lut=True, rotate=0):
    dicom = pydicom.dcmread(path, force=True)
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    data = data.astype(np.float32)

    if rotate > 0:
        rot_choices = [
            None,
            cv2.ROTATE_90_CLOCKWISE,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
            cv2.ROTATE_180,
        ]
        data = cv2.rotate(data, rot_choices[rotate])

    data = cv2.resize(data, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return data


@lru_cache(maxsize=4096)
def _load_dicom_images_3d_cached(
    split,
    scan_id,
    mri_type,
    num_imgs,
    img_size,
    rotate,
):
    files = _get_sorted_dicom_files(split, scan_id, mri_type)

    if len(files) == 0:
        img3d = np.zeros((img_size, img_size, num_imgs), dtype=np.float32)
        return img3d  # (H,W,D)

    middle = len(files) // 2
    num_imgs2 = num_imgs // 2
    p1 = max(0, middle - num_imgs2)
    p2 = min(len(files), middle + num_imgs2)

    selected = files[p1:p2]
    if len(selected) == 0:
        selected = files[: min(len(files), num_imgs)]

    d = len(selected)
    vol = np.empty((img_size, img_size, d), dtype=np.float32)
    for k, f in enumerate(selected):
        vol[:, :, k] = load_dicom_image(f, img_size=img_size, rotate=rotate)

    if d < num_imgs:
        out = np.zeros((img_size, img_size, num_imgs), dtype=np.float32)
        out[:, :, :d] = vol
        vol = out
    elif d > num_imgs:
        vol = vol[:, :, :num_imgs]

    mn, mx = float(np.min(vol)), float(np.max(vol))
    if mn < mx:
        vol = (vol - mn) / (mx - mn)
    else:
        vol = np.zeros_like(vol, dtype=np.float32)

    return vol.astype(np.float32, copy=False)


def load_dicom_images_3d(
    scan_id,
    num_imgs=NUM_IMAGES,
    img_size=IMAGE_SIZE,
    mri_type="FLAIR",
    split="test",
    rotate=0,
):
    img3d = _load_dicom_images_3d_cached(
        split=split,
        scan_id=scan_id,
        mri_type=mri_type,
        num_imgs=num_imgs,
        img_size=img_size,
        rotate=rotate,
    )
    return np.expand_dims(img3d, 0)  # (1,H,W,D)


all_train_ids5 = train_labels["BraTS21ID5"].tolist()
all_test_ids5 = test["BraTS21ID5"].tolist()
_DICOM_INDEX = {}
_DICOM_INDEX.update(
    _build_dicom_index(data_directory, "train", all_train_ids5, ["FLAIR"])
)
_DICOM_INDEX.update(
    _build_dicom_index(data_directory, "test", all_test_ids5, ["FLAIR"])
)

sid = test.loc[0, "BraTS21ID5"]
a = load_dicom_images_3d(sid, mri_type="FLAIR", split="test")
print("Loaded:", sid, "shape:", a.shape, "min/max:", np.min(a), np.max(a))
plt.imshow(a[0, :, :, NUM_IMAGES // 2], cmap="gray")
plt.axis("off")
plt.show()



## === cell 4
from tensorflow.keras.utils import Sequence


class Dataset(Sequence):
    def __init__(
        self,
        df,
        is_train=True,
        batch_size=2,
        shuffle=True,
        mri_type="FLAIR",
        split="train",
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
        self.mri_type = mri_type
        self.split = split
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.paths) / self.batch_size)

    def __getitem__(self, i):
        batch_paths = self.paths[i * self.batch_size : (i + 1) * self.batch_size]
        b = len(batch_paths)

        batch_X = np.empty((b, IMAGE_SIZE, IMAGE_SIZE, NUM_IMAGES, 1), dtype=np.float32)

        for j, sid in enumerate(batch_paths):
            vol = load_dicom_images_3d(
                sid,
                num_imgs=NUM_IMAGES,
                img_size=IMAGE_SIZE,
                mri_type=self.mri_type,
                split=self.split,
                rotate=0,
            )  # (1, H, W, D)
            batch_X[j, ..., 0] = vol[0]

        if self.is_train:
            batch_y = self.y[i * self.batch_size : (i + 1) * self.batch_size].astype(
                np.float32, copy=False
            )
            return batch_X, batch_y
        return batch_X

    def on_epoch_end(self):
        if self.shuffle and self.is_train and self.y is not None:
            perm = np.random.permutation(len(self.paths))
            self.paths = self.paths[perm]
            self.y = self.y[perm]




## === cell 5
train_df, valid_df = sk_model_selection.train_test_split(
    train_labels, test_size=0.2, random_state=SEED, stratify=train_labels["MGMT_value"]
)

train_dataset = Dataset(
    train_df, is_train=True, batch_size=2, shuffle=True, mri_type="FLAIR", split="train"
)
valid_dataset = Dataset(
    valid_df,
    is_train=True,
    batch_size=2,
    shuffle=False,
    mri_type="FLAIR",
    split="train",
)
test_dataset = Dataset(
    test, is_train=False, batch_size=2, shuffle=False, mri_type="FLAIR", split="test"
)

xb, yb = train_dataset[0]
print("Batch X:", xb.shape, xb.dtype, "Batch y:", yb.shape, yb[:5])



## === cell 6
inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, NUM_IMAGES, 1))

x = layers.Conv3D(16, 3, padding="same", activation="relu")(inputs)
x = layers.MaxPool3D(pool_size=2)(x)
x = layers.Conv3D(32, 3, padding="same", activation="relu")(x)
x = layers.MaxPool3D(pool_size=2)(x)
x = layers.Conv3D(64, 3, padding="same", activation="relu")(x)
x = layers.GlobalAveragePooling3D()(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[keras.metrics.AUC(name="auc")],
)

model.summary()



## === cell 7
history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=2,
    verbose=1,
)



## === cell 8
preds = model.predict(
    test_dataset,
    verbose=1,
).reshape(-1)

preds = np.clip(preds, 0.0, 1.0)

if len(preds) != len(sample_submission):
    preds = preds[: len(sample_submission)]

print("Preds:", preds.shape, "min/max:", preds.min(), preds.max())



## === cell 9
submission = pd.DataFrame(
    {
        "BraTS21ID": sample_submission["BraTS21ID"].values,
        "MGMT_value": preds,
    }
)

submission["BraTS21ID"] = submission["BraTS21ID"].astype(int)
submission["MGMT_value"] = submission["MGMT_value"].astype(float)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)



## === cell 10
plt.figure(figsize=(5, 4))
plt.hist(submission["MGMT_value"], bins=20)
plt.title("Prediction distribution")
plt.show()
