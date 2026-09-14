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

# 8. Previous improvement plans

- What this solution (achieved 0.57882) has done: 'I (1) fix the early import crash caused by a protobuf/pydicom interaction by forcing the pure-Python protobuf implementation before importing TensorFlow/pydicom, (2) make DICOM decoding robust to missing required tags by reading the pixel data without `specific_tags` and adding a safe fallback path if `pixel_array` fails, and (3) update Keras `.fit()`/`.predict()` calls to remove unsupported `workers/use_multiprocessing/max_queue_size` arguments in this environment so training/inference runs end-to-end. These are execution/stability fixes that preserve your model/training logic and finally produce a valid `submission.csv`. With no current score available (no valid submission previously), the priority is correctness and producing a runnable pipeline; score should improve from “not yielded” to a meaningful AUC submission.'
- What this solution (achieved 0.60588) has done: 'I fix the import crash happening in cell 1 by making the protobuf/pydicom/TensorFlow import order deterministic and forcing the pure-Python protobuf implementation *before* any library that transitively imports protobuf. I also add a safe fallback to disable DICOM pixel handlers that can trigger the same protobuf issue in some Kaggle images, while keeping your data loading/model/training logic unchanged. Finally, I keep submission generation identical but make sure the script always completes and writes a valid `submission.csv` with the correct columns.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
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

try:
    from google.protobuf import message_factory as _message_factory

    if hasattr(_message_factory, "MessageFactory") and not hasattr(
        _message_factory.MessageFactory, "GetPrototype"
    ):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        _message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

try:
    pydicom.config.image_handlers = []
except Exception:
    pass

from sklearn import model_selection as sk_model_selection

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("TensorFlow:", tf.__version__)
print("Python:", sys.version)



## === cell 1
data_directory = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"

mri_types = ["FLAIR", "T1w", "T1wCE", "T2w"]
IMAGE_SIZE = 256
NUM_IMAGES = 48

train_labels_path = f"{data_directory}/train_labels.csv"
sample_sub_path = f"{data_directory}/sample_submission.csv"

assert os.path.exists(train_labels_path), train_labels_path
assert os.path.exists(sample_sub_path), sample_sub_path



## === cell 2
sample_submission = pd.read_csv(sample_sub_path)
test = sample_submission.copy()
test["BraTS21ID5"] = [format(int(x), "05d") for x in test.BraTS21ID]

train_df = pd.read_csv(train_labels_path)
train_df["BraTS21ID5"] = [format(int(x), "05d") for x in train_df.BraTS21ID]

bad_ids = set(["00109", "00123", "00709"])
train_df = train_df[~train_df["BraTS21ID5"].isin(bad_ids)].reset_index(drop=True)

print("train_df:", train_df.shape, "test:", test.shape)
train_df.head(3)



## === cell 3
_DICOM_FILE_CACHE = {}
_DICOM_INSTANCE_CACHE = {}

_VOL_CACHE = collections.OrderedDict()
_VOL_CACHE_MAX = 1024  # bounded; typically fits in Kaggle RAM for this small model run

_NAT_SORT_RE = re.compile(r"[^0-9]|[0-9]+")


def _natural_sort_key(path_str):
    return [int(x) if x.isdigit() else x for x in _NAT_SORT_RE.findall(path_str)]


def _try_get_instance_number(dcm_path):
    v = _DICOM_INSTANCE_CACHE.get(dcm_path)
    if v is not None:
        return v
    try:
        ds = pydicom.dcmread(
            dcm_path,
            stop_before_pixels=True,
            specific_tags=["InstanceNumber"],
            force=True,
        )
        inst = getattr(ds, "InstanceNumber", None)
        inst = int(inst) if inst is not None else None
    except Exception:
        inst = None
    _DICOM_INSTANCE_CACHE[dcm_path] = inst
    return inst


def _get_series_files(scan_id, mri_type="FLAIR", split="test"):
    key = (split, scan_id, mri_type)
    files = _DICOM_FILE_CACHE.get(key)
    if files is not None:
        return files

    series_dir = f"{data_directory}/{split}/{scan_id}/{mri_type}"
    files = glob.glob(f"{series_dir}/*.dcm")
    if not files:
        _DICOM_FILE_CACHE[key] = []
        return []

    inst = [(_try_get_instance_number(f), f) for f in files]
    if all(x[0] is not None for x in inst):
        inst.sort(key=lambda t: t[0])
        files = [f for _, f in inst]
    else:
        ok = True
        nums = []
        for f in files:
            base = os.path.basename(f)
            if base.startswith("Image-") and base.endswith(".dcm"):
                n = base[6:-4]
                if n.isdigit():
                    nums.append(int(n))
                else:
                    ok = False
                    break
            else:
                ok = False
                break
        if ok:
            order = np.argsort(np.asarray(nums))
            files = [files[i] for i in order]
        else:
            files = sorted(files, key=_natural_sort_key)

    _DICOM_FILE_CACHE[key] = files
    return files


def _vol_cache_get(cache_key):
    v = _VOL_CACHE.get(cache_key)
    if v is None:
        return None
    _VOL_CACHE.move_to_end(cache_key)
    return v


def _vol_cache_put(cache_key, value):
    _VOL_CACHE[cache_key] = value
    _VOL_CACHE.move_to_end(cache_key)
    if len(_VOL_CACHE) > _VOL_CACHE_MAX:
        _VOL_CACHE.popitem(last=False)


def _safe_get_pixel_array(ds):
    try:
        return ds.pixel_array
    except Exception:
        pass

    try:
        if "PixelData" not in ds:
            raise ValueError("No PixelData")
        rows = int(getattr(ds, "Rows", 0) or 0)
        cols = int(getattr(ds, "Columns", 0) or 0)
        if rows <= 0 or cols <= 0:
            raise ValueError("Missing Rows/Columns")

        bits_allocated = int(getattr(ds, "BitsAllocated", 16) or 16)
        pixel_repr = int(getattr(ds, "PixelRepresentation", 0) or 0)
        if bits_allocated == 16:
            dtype = np.int16 if pixel_repr == 1 else np.uint16
        elif bits_allocated == 8:
            dtype = np.int8 if pixel_repr == 1 else np.uint8
        else:
            dtype = np.int16 if pixel_repr == 1 else np.uint16

        arr = np.frombuffer(ds.PixelData, dtype=dtype)
        if arr.size < rows * cols:
            raise ValueError("PixelData too small")
        arr = arr[: rows * cols].reshape(rows, cols)
        return arr
    except Exception:
        return None


def load_dicom_image(path, img_size=IMAGE_SIZE, voi_lut=True, rotate=0):
    dicom = pydicom.dcmread(
        path,
        force=True,
        specific_tags=[
            "PixelData",
            "Rows",
            "Columns",
            "BitsAllocated",
            "PixelRepresentation",
            "RescaleIntercept",
            "RescaleSlope",
            "WindowCenter",
            "WindowWidth",
        ],
    )

    arr = _safe_get_pixel_array(dicom)
    if arr is None:
        data = np.zeros((img_size, img_size), dtype=np.float32)
        return data

    if voi_lut:
        try:
            data = apply_voi_lut(arr, dicom)
        except Exception:
            data = arr
    else:
        data = arr

    data = data.astype(np.float32, copy=False)

    try:
        intercept = float(getattr(dicom, "RescaleIntercept", 0.0))
        slope = float(getattr(dicom, "RescaleSlope", 1.0))
        data = data * slope + intercept
    except Exception:
        pass

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


def load_dicom_images_3d(
    scan_id,
    num_imgs=NUM_IMAGES,
    img_size=IMAGE_SIZE,
    mri_type="FLAIR",
    split="test",
    rotate=0,
):
    cache_key = (split, scan_id, mri_type, rotate, img_size, num_imgs)
    cached = _vol_cache_get(cache_key)
    if cached is not None:
        return cached

    files = _get_series_files(scan_id, mri_type=mri_type, split=split)

    if len(files) == 0:
        img3d = np.zeros((img_size, img_size, num_imgs), dtype=np.float32)
        out = np.expand_dims(img3d, 0)
        _vol_cache_put(cache_key, out)
        return out

    middle = len(files) // 2
    num_imgs2 = num_imgs // 2
    p1 = max(0, middle - num_imgs2)
    p2 = min(len(files), middle + num_imgs2)

    slices = [
        load_dicom_image(f, img_size=img_size, rotate=rotate) for f in files[p1:p2]
    ]
    img3d = np.stack(slices, axis=0).transpose(1, 2, 0)  # (H,W,D)

    if img3d.shape[-1] < num_imgs:
        n_zero = np.zeros(
            (img_size, img_size, num_imgs - img3d.shape[-1]), dtype=np.float32
        )
        img3d = np.concatenate((img3d, n_zero), axis=-1)

    vmin, vmax = float(np.min(img3d)), float(np.max(img3d))
    if vmin < vmax:
        img3d = (img3d - vmin) / (vmax - vmin)

    out = np.expand_dims(img3d.astype(np.float32, copy=False), 0)
    _vol_cache_put(cache_key, out)
    return out


example_id = test.loc[0, "BraTS21ID5"]
a = load_dicom_images_3d(example_id, split="test", mri_type="FLAIR")
print("Example volume:", example_id, a.shape, np.min(a), np.max(a))



## === cell 4
vol = a[0]
print("Single volume tensor:", vol.shape, vol.dtype)



## === cell 5
from tensorflow.keras.utils import Sequence


class Dataset(Sequence):
    def __init__(
        self,
        df,
        is_train=True,
        batch_size=1,
        shuffle=True,
        split="train",
        mri_type="FLAIR",
    ):
        self.df = df.reset_index(drop=True)
        self.idx = self.df["BraTS21ID"].values
        self.paths = self.df["BraTS21ID5"].values

        self.y = (
            self.df["MGMT_value"].values if "MGMT_value" in self.df.columns else None
        )

        self.is_train = is_train
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.split = split
        self.mri_type = mri_type

        for sid in self.paths:
            _get_series_files(sid, mri_type=self.mri_type, split=self.split)

        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.paths) / self.batch_size)

    def __getitem__(self, i):
        batch_paths = self.paths[i * self.batch_size : (i + 1) * self.batch_size]

        b = len(batch_paths)
        batch_X = np.empty((b, IMAGE_SIZE, IMAGE_SIZE, NUM_IMAGES, 1), dtype=np.float32)

        for j, scan_id5 in enumerate(batch_paths):
            x = load_dicom_images_3d(scan_id5, split=self.split, mri_type=self.mri_type)
            batch_X[j, ..., 0] = x[0]

        if self.is_train:
            batch_y = self.y[i * self.batch_size : (i + 1) * self.batch_size].astype(
                np.float32
            )
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
tr_df, va_df = sk_model_selection.train_test_split(
    train_df, test_size=0.2, random_state=42, stratify=train_df["MGMT_value"]
)

BATCH_SIZE = 2  # keep small to fit memory/time
train_dataset = Dataset(
    tr_df,
    is_train=True,
    batch_size=BATCH_SIZE,
    shuffle=True,
    split="train",
    mri_type="FLAIR",
)
val_dataset = Dataset(
    va_df,
    is_train=True,
    batch_size=BATCH_SIZE,
    shuffle=False,
    split="train",
    mri_type="FLAIR",
)
test_dataset = Dataset(
    test, is_train=False, batch_size=1, shuffle=False, split="test", mri_type="FLAIR"
)

x0, y0 = train_dataset[0]
print("Batch X:", x0.shape, x0.dtype, "Batch y:", y0.shape, y0[:5])




## === cell 7
def build_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, NUM_IMAGES, 1)):
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv3D(8, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)

    x = layers.Conv3D(16, 3, padding="same", activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)

    x = layers.Conv3D(32, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling3D()(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


model = build_model()
model.summary()



## === cell 8
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

EPOCHS = 2

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    verbose=1,
    workers=max(1, (os.cpu_count() or 2) - 1),
    use_multiprocessing=True,
    max_queue_size=16,
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/952901325.py in <cell line: 0>()
     12 # This preserves training semantics; it only parallelizes I/O/decoding. Determinism is retained by seeds
     13 # and by disabling per-worker shuffling (Sequence handles shuffling deterministically per epoch).
---> 14 history = model.fit(
     15     train_dataset,
     16     validation_data=val_dataset,

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
    workers=max(1, (os.cpu_count() or 2) - 1),
    use_multiprocessing=True,
    max_queue_size=16,
).reshape(-1)

preds = np.clip(preds, 0.0, 1.0)

submission = pd.DataFrame(
    {"BraTS21ID": sample_submission["BraTS21ID"].values, "MGMT_value": preds}
)

submission["BraTS21ID"] = submission["BraTS21ID"].astype(int)
submission["MGMT_value"] = submission["MGMT_value"].astype(float)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1265195619.py in <cell line: 0>()
      1 # Speed fix: parallelize prediction input as well (same as training; no semantic changes).
----> 2 preds = model.predict(
      3     test_dataset,
      4     verbose=1,
      5     workers=max(1, (os.cpu_count() or 2) - 1),

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
print("Min/Max pred:", submission["MGMT_value"].min(), submission["MGMT_value"].max())
print(
    "submission.csv exists:",
    os.path.exists("submission.csv"),
    "size:",
    os.path.getsize("submission.csv"),
)

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4169988630.py in <cell line: 0>()
----> 1 print("Min/Max pred:", submission["MGMT_value"].min(), submission["MGMT_value"].max())
      2 print(
      3     "submission.csv exists:",
      4     os.path.exists("submission.csv"),
      5     "size:",

NameError: name 'submission' is not defined
