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

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import sys
import json
import glob
import random
import collections
import time
import re
import math
from pathlib import Path

import numpy as np
import pandas as pd
import cv2

import matplotlib.pyplot as plt

from sklearn import model_selection as sk_model_selection

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass

try:
    cv2.setNumThreads(0)
except Exception:
    pass

print("TF:", tf.__version__)
print("Python:", sys.version)



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

train_dir = os.path.join(data_directory, "train")
test_dir = os.path.join(data_directory, "test")

assert os.path.exists(train_labels_path), f"Missing: {train_labels_path}"
assert os.path.exists(sample_sub_path), f"Missing: {sample_sub_path}"
assert os.path.isdir(train_dir), f"Missing dir: {train_dir}"
assert os.path.isdir(test_dir), f"Missing dir: {test_dir}"



## === cell 2
sample_submission = pd.read_csv(sample_sub_path)
test_df = sample_submission.copy()
test_df["BraTS21ID5"] = [format(int(x), "05d") for x in test_df.BraTS21ID.values]

train_df = pd.read_csv(train_labels_path)
train_df["BraTS21ID5"] = [format(int(x), "05d") for x in train_df.BraTS21ID.values]

bad_cases = {"00109", "00123", "00709"}
train_df = train_df[~train_df["BraTS21ID5"].isin(bad_cases)].reset_index(drop=True)

print("train_df:", train_df.shape, "test_df:", test_df.shape)
train_df.head(3)



## === cell 3
import pydicom

_IMG_NUM_RE = re.compile(r"Image-(\d+)\.dcm$", re.IGNORECASE)


def _dicom_sort_key_from_name(name: str) -> int:
    m = _IMG_NUM_RE.search(name)
    if m:
        return int(m.group(1))
    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else 0


_FILE_LIST_CACHE = {}  # (split, scan_id, mri_type, num_imgs) -> tuple[str]
_VOL3D_CACHE = collections.OrderedDict()  # key -> np.ndarray shape (1,H,W,D)
_VOL3D_CACHE_MAX = 64  # hot RAM cache small; disk cache holds full set.

_TENSOR4D_CACHE = collections.OrderedDict()  # key -> np.ndarray (H,W,D,1)
_TENSOR4D_CACHE_MAX = 64

DISK_CACHE_DIR = Path("../working/volcache_npy")
DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _disk_cache_path(split, scan_id, mri_type, num_imgs, img_size, rotate):
    return (
        DISK_CACHE_DIR
        / f"{split}_{scan_id}_{mri_type}_n{num_imgs}_s{img_size}_r{rotate}.npy"
    )


def _disk_cache_path_4d(split, scan_id, mri_type, num_imgs, img_size, rotate):
    return (
        DISK_CACHE_DIR
        / f"{split}_{scan_id}_{mri_type}_n{num_imgs}_s{img_size}_r{rotate}_4d.npy"
    )


def _vol3d_cache_get(key):
    v = _VOL3D_CACHE.get(key)
    if v is not None:
        _VOL3D_CACHE.move_to_end(key)
    return v


def _vol3d_cache_set(key, value):
    _VOL3D_CACHE[key] = value
    _VOL3D_CACHE.move_to_end(key)
    if len(_VOL3D_CACHE) > _VOL3D_CACHE_MAX:
        _VOL3D_CACHE.popitem(last=False)


def _tensor4d_cache_get(key):
    v = _TENSOR4D_CACHE.get(key)
    if v is not None:
        _TENSOR4D_CACHE.move_to_end(key)
    return v


def _tensor4d_cache_set(key, value):
    _TENSOR4D_CACHE[key] = value
    _TENSOR4D_CACHE.move_to_end(key)
    if len(_TENSOR4D_CACHE) > _TENSOR4D_CACHE_MAX:
        _TENSOR4D_CACHE.popitem(last=False)


def _atomic_npy_save(path: Path, arr: np.ndarray):
    tmp_path = Path(str(path) + ".tmp.npy")
    np.save(tmp_path, arr, allow_pickle=False)
    os.replace(tmp_path, path)


def _get_center_dicom_files(scan_id, split, mri_type, num_imgs=NUM_IMAGES):
    key = (split, scan_id, mri_type, num_imgs)
    files = _FILE_LIST_CACHE.get(key)
    if files is not None:
        return files

    folder = os.path.join(data_directory, split, scan_id, mri_type)
    try:
        with os.scandir(folder) as it:
            names = [
                e.name for e in it if e.is_file() and e.name.lower().endswith(".dcm")
            ]
    except FileNotFoundError:
        _FILE_LIST_CACHE[key] = tuple()
        return tuple()

    if not names:
        _FILE_LIST_CACHE[key] = tuple()
        return tuple()

    names.sort(key=_dicom_sort_key_from_name)
    n = len(names)
    middle = n // 2
    half = num_imgs // 2
    p1 = max(0, middle - half)
    p2 = min(n, middle + half)
    if p2 <= p1:
        p1 = max(0, middle - 1)
        p2 = min(n, middle + 1)

    sel_names = names[p1:p2][:num_imgs]
    files = tuple(os.path.join(folder, nm) for nm in sel_names)
    _FILE_LIST_CACHE[key] = files
    return files


def load_dicom_image(path, img_size=IMAGE_SIZE, rotate=0):
    try:
        dicom = pydicom.dcmread(
            path,
            force=False,
            stop_before_pixels=True,
            specific_tags=[
                "PixelData",
                "Rows",
                "Columns",
                "BitsAllocated",
                "BitsStored",
                "HighBit",
                "PixelRepresentation",
                "SamplesPerPixel",
                "PhotometricInterpretation",
                "PlanarConfiguration",
                "RescaleIntercept",
                "RescaleSlope",
                "WindowCenter",
                "WindowWidth",
                "TransferSyntaxUID",
            ],
        )
    except Exception:
        dicom = pydicom.dcmread(
            path,
            force=True,
            stop_before_pixels=True,
            specific_tags=[
                "PixelData",
                "Rows",
                "Columns",
                "BitsAllocated",
                "BitsStored",
                "HighBit",
                "PixelRepresentation",
                "SamplesPerPixel",
                "PhotometricInterpretation",
                "PlanarConfiguration",
                "RescaleIntercept",
                "RescaleSlope",
                "WindowCenter",
                "WindowWidth",
                "TransferSyntaxUID",
            ],
        )

    data = dicom.pixel_array.astype(np.float32, copy=False)

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


def load_dicom_images_3d(
    scan_id,
    num_imgs=NUM_IMAGES,
    img_size=IMAGE_SIZE,
    mri_type="FLAIR",
    split="test",
    rotate=0,
):
    key = (split, scan_id, mri_type, num_imgs, img_size, rotate)

    cached = _vol3d_cache_get(key)
    if cached is not None:
        return cached

    disk_path = _disk_cache_path(split, scan_id, mri_type, num_imgs, img_size, rotate)
    if disk_path.exists():
        out = np.load(disk_path, allow_pickle=False, mmap_mode=None)
        _vol3d_cache_set(key, out)
        return out

    files = _get_center_dicom_files(scan_id, split, mri_type, num_imgs=num_imgs)

    img3d = np.zeros((img_size, img_size, num_imgs), dtype=np.float32)
    if files:
        for j, f in enumerate(files):
            img3d[:, :, j] = load_dicom_image(f, img_size=img_size, rotate=rotate)

        mn = float(img3d.min())
        mx = float(img3d.max())
        if mn < mx:
            img3d -= mn
            img3d /= mx - mn

    out = np.expand_dims(img3d, 0)  # (1, H, W, D)

    _atomic_npy_save(disk_path, out)
    _vol3d_cache_set(key, out)
    return out


def load_dicom_tensor_4d(
    scan_id,
    num_imgs=NUM_IMAGES,
    img_size=IMAGE_SIZE,
    mri_type="FLAIR",
    split="test",
    rotate=0,
):
    key = (split, scan_id, mri_type, num_imgs, img_size, rotate)

    cached = _tensor4d_cache_get(key)
    if cached is not None:
        return cached

    disk_path = _disk_cache_path_4d(
        split, scan_id, mri_type, num_imgs, img_size, rotate
    )
    if disk_path.exists():
        x = np.load(disk_path, allow_pickle=False, mmap_mode=None)
        _tensor4d_cache_set(key, x)
        return x

    vol = load_dicom_images_3d(
        scan_id,
        num_imgs=num_imgs,
        img_size=img_size,
        mri_type=mri_type,
        split=split,
        rotate=rotate,
    )  # (1,H,W,D)
    x = vol[0][..., np.newaxis].astype(np.float32, copy=False)  # (H,W,D,1)

    _atomic_npy_save(disk_path, x)
    _tensor4d_cache_set(key, x)
    return x




## === cell 4
example_id = test_df.loc[0, "BraTS21ID5"]
x = load_dicom_images_3d(example_id, split="test", mri_type="FLAIR")
print("Loaded volume:", x.shape, "min/max:", float(x.min()), float(x.max()))




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3376468613.py in <cell line: 0>()
      1 example_id = test_df.loc[0, "BraTS21ID5"]
----> 2 x = load_dicom_images_3d(example_id, split="test", mri_type="FLAIR")
      3 print("Loaded volume:", x.shape, "min/max:", float(x.min()), float(x.max()))
      4 
      5 

/tmp/ipykernel_11/4077137452.py in load_dicom_images_3d(scan_id, num_imgs, img_size, mri_type, split, rotate)
    201     if files:
    202         for j, f in enumerate(files):
--> 203             img3d[:, :, j] = load_dicom_image(f, img_size=img_size, rotate=rotate)
    204 
    205         # CHANGE (timeout fix): keep identical normalization math, but avoid extra temporaries.

/tmp/ipykernel_11/4077137452.py in load_dicom_image(path, img_size, rotate)
    161         )
    162 
--> 163     data = dicom.pixel_array.astype(np.float32, copy=False)
    164 
    165     if rotate > 0:

/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in __getattr__(self, name)
    916             return {}
    917         # Try the base class attribute getter (fix for issue 332)
--> 918         return object.__getattribute__(self, name)
    919 
    920     @property

/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in pixel_array(self)
   2191             that iterates through the image frames.
   2192         """
-> 2193         self.convert_pixel_data()
   2194         return cast("numpy.ndarray", self._pixel_array)
   2195 

/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in convert_pixel_data(self, handler_name)
   1724             # Use 'pydicom.pixels' backend
   1725             opts["decoding_plugin"] = name
-> 1726             self._pixel_array = pixel_array(self, **opts)
   1727             self._pixel_id = get_image_pixel_ids(self)
   1728         else:

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/utils.py in pixel_array(src, ds_out, specific_tags, index, raw, decoding_plugin, **kwargs)
   1428 
   1429         opts = as_pixel_options(ds, **kwargs)
-> 1430         return decoder.as_array(
   1431             ds,
   1432             index=index,

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/decoders/base.py in as_array(self, src, index, validate, raw, decoding_plugin, **kwargs)
    975 
    976         runner = DecodeRunner(self.UID)
--> 977         runner.set_source(src)
    978         runner.set_options(**kwargs)
    979         runner.set_decoders(

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/decoders/base.py in set_source(self, src)
    684 
    685         if isinstance(src, Dataset):
--> 686             self._set_options_ds(src)
    687             self._src = src[self.pixel_keyword].value
    688             if isinstance(self._src, BufferedIOBase):

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/decoders/base.py in _set_options_ds(self, ds)
    657         px_keyword = [kw for kw in keywords if kw in ds]
    658         if not px_keyword:
--> 659             raise AttributeError(
    660                 "The dataset has no 'Pixel Data', 'Float Pixel Data' or 'Double "
    661                 "Float Pixel Data' element, no pixel data to decode"

AttributeError: The dataset has no 'Pixel Data', 'Float Pixel Data' or 'Double Float Pixel Data' element, no pixel data to decode

## === cell 5
def make_tf_dataset(
    df, is_train=True, batch_size=1, shuffle=True, split="train", mri_type="FLAIR"
):
    df = df.reset_index(drop=True)
    paths = df["BraTS21ID5"].astype(str).values
    y = df["MGMT_value"].values.astype(np.float32) if is_train else None

    ids_tf = tf.constant(paths)
    if is_train:
        y_tf = tf.constant(y)
        ds = tf.data.Dataset.from_tensor_slices((ids_tf, y_tf))
    else:
        ds = tf.data.Dataset.from_tensor_slices(ids_tf)

    if shuffle and is_train:
        ds = ds.shuffle(
            buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True
        )

    def _py_load_one(sid_bytes):
        if isinstance(sid_bytes, (bytes, bytearray)):
            sid = sid_bytes.decode("utf-8")
        else:
            sid = sid_bytes.decode("utf-8")
        x = load_dicom_tensor_4d(sid, split=split, mri_type=mri_type)  # (H,W,D,1)
        return x

    map_calls = min(4, (os.cpu_count() or 2))

    if is_train:

        def _map_fn(sid, label):
            x = tf.numpy_function(_py_load_one, [sid], Tout=tf.float32)
            x.set_shape((IMAGE_SIZE, IMAGE_SIZE, NUM_IMAGES, 1))
            label = tf.cast(label, tf.float32)
            return x, label

        ds = ds.map(_map_fn, num_parallel_calls=map_calls, deterministic=True)
        ds = ds.cache()
    else:

        def _map_fn(sid):
            x = tf.numpy_function(_py_load_one, [sid], Tout=tf.float32)
            x.set_shape((IMAGE_SIZE, IMAGE_SIZE, NUM_IMAGES, 1))
            return x

        ds = ds.map(_map_fn, num_parallel_calls=map_calls, deterministic=True)
        ds = ds.cache()

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


train_split, val_split = sk_model_selection.train_test_split(
    train_df, test_size=0.2, random_state=SEED, stratify=train_df["MGMT_value"]
)

for _df, _split in ((train_split, "train"), (val_split, "train"), (test_df, "test")):
    for _sid in _df["BraTS21ID5"].astype(str).values:
        _get_center_dicom_files(_sid, _split, "FLAIR", num_imgs=NUM_IMAGES)

train_dataset = make_tf_dataset(
    train_split,
    is_train=True,
    batch_size=1,
    shuffle=True,
    split="train",
    mri_type="FLAIR",
)
val_dataset = make_tf_dataset(
    val_split,
    is_train=True,
    batch_size=1,
    shuffle=False,
    split="train",
    mri_type="FLAIR",
)
test_dataset = make_tf_dataset(
    test_df, is_train=False, batch_size=1, shuffle=False, split="test", mri_type="FLAIR"
)

for xb, yb in train_dataset.take(1):
    print(
        "Batch X:",
        xb.shape,
        xb.dtype.name,
        "Batch y:",
        yb.shape,
        yb.dtype.name,
        "y sample:",
        yb.numpy()[:5],
    )




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
UnknownError                              Traceback (most recent call last)
/tmp/ipykernel_11/1706923086.py in <cell line: 0>()
     86 )
     87 
---> 88 for xb, yb in train_dataset.take(1):
     89     print(
     90         "Batch X:",

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in __next__(self)
    824   def __next__(self):
    825     try:
--> 826       return self._next_internal()
    827     except errors.OutOfRangeError:
    828       raise StopIteration

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in _next_internal(self)
    774     # to communicate that there is no more data to iterate over.
    775     with context.execution_mode(context.SYNC):
--> 776       ret = gen_dataset_ops.iterator_get_next(
    777           self._iterator_resource,
    778           output_types=self._flat_output_types,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_dataset_ops.py in iterator_get_next(iterator, output_types, output_shapes, name)
   3084       return _result
   3085     except _core._NotOkStatusException as e:
-> 3086       _ops.raise_from_not_ok_status(e, name)
   3087     except _core._FallbackException:
   3088       pass

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

UnknownError: {{function_node __wrapped__IteratorGetNext_output_types_2_device_/job:localhost/replica:0/task:0/device:CPU:0}} Error in user-defined function passed to ParallelMapDatasetV2:2 transformation with iterator: Iterator::Root::Prefetch::FiniteTake::Prefetch::BatchV2::MemoryCacheImpl::Map: AttributeError: The dataset has no 'Pixel Data', 'Float Pixel Data' or 'Double Float Pixel Data' element, no pixel data to decode
Traceback (most recent call last):

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 269, in __call__
    ret = func(*args)
          ^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py", line 643, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_11/1706923086.py", line 25, in _py_load_one
    x = load_dicom_tensor_4d(sid, split=split, mri_type=mri_type)  # (H,W,D,1)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_11/4077137452.py", line 241, in load_dicom_tensor_4d
    vol = load_dicom_images_3d(
          ^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_11/4077137452.py", line 203, in load_dicom_images_3d
    img3d[:, :, j] = load_dicom_image(f, img_size=img_size, rotate=rotate)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_11/4077137452.py", line 163, in load_dicom_image
    data = dicom.pixel_array.astype(np.float32, copy=False)
           ^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py", line 918, in __getattr__
    return object.__getattribute__(self, name)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py", line 2193, in pixel_array
    self.convert_pixel_data()

  File "/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py", line 1726, in convert_pixel_data
    self._pixel_array = pixel_array(self, **opts)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/pydicom/pixels/utils.py", line 1430, in pixel_array
    return decoder.as_array(
           ^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/pydicom/pixels/decoders/base.py", line 977, in as_array
    runner.set_source(src)

  File "/usr/local/lib/python3.11/dist-packages/pydicom/pixels/decoders/base.py", line 686, in set_source
    self._set_options_ds(src)

  File "/usr/local/lib/python3.11/dist-packages/pydicom/pixels/decoders/base.py", line 659, in _set_options_ds
    raise AttributeError(

AttributeError: The dataset has no 'Pixel Data', 'Float Pixel Data' or 'Double Float Pixel Data' element, no pixel data to decode


	 [[{{node PyFunc}}]] [Op:IteratorGetNext] name: 

## === cell 6
def build_model():
    inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, NUM_IMAGES, 1))
    x = layers.Conv3D(8, kernel_size=3, padding="same", activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.Conv3D(16, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.Conv3D(32, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(1e-4), loss="binary_crossentropy")
    return model


model = build_model()
model.summary()



## === cell 7
try:
    tf.data.experimental.enable_debug_mode = (
        lambda *args, **kwargs: None
    )  # no-op safety
except Exception:
    pass

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=2,
    verbose=1,
)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
UnknownError                              Traceback (most recent call last)
/tmp/ipykernel_11/934897132.py in <cell line: 0>()
      6     pass
      7 
----> 8 history = model.fit(
      9     train_dataset,
     10     validation_data=val_dataset,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
     57       e.message += " name: " + name
     58     raise core._status_to_exception(e) from None
---> 59   except TypeError as e:
     60     keras_symbolic_tensors = [x for x in inputs if _is_keras_symbolic_tensor(x)]
     61     if keras_symbolic_tensors:

UnknownError: Graph execution error:

Detected at node PyFunc defined at (most recent call last):
<stack traces unavailable>
Error in user-defined function passed to ParallelMapDatasetV2:2 transformation with iterator: Iterator::Root::Prefetch::BatchV2::MemoryCacheImpl::Map: AttributeError: The dataset has no 'Pixel Data', 'Float Pixel Data' or 'Double Float Pixel Data' element, no pixel data to decode
Traceback (most recent call last):

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 269, in __call__
    ret = func(*args)
          ^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py", line 643, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_11/1706923086.py", line 25, in _py_load_one
    x = load_dicom_tensor_4d(sid, split=split, mri_type=mri_type)  # (H,W,D,1)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_11/4077137452.py", line 241, in load_dicom_tensor_4d
    vol = load_dicom_images_3d(
          ^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_11/4077137452.py", line 203, in load_dicom_images_3d
    img3d[:, :, j] = load_dicom_image(f, img_size=img_size, rotate=rotate)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_11/4077137452.py", line 163, in load_dicom_image
    data = dicom.pixel_array.astype(np.float32, copy=False)
           ^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py", line 918, in __getattr__
    return object.__getattribute__(self, name)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py", line 2193, in pixel_array
    self.convert_pixel_data()

  File "/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py", line 1726, in convert_pixel_data
    self._pixel_array = pixel_array(self, **opts)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/pydicom/pixels/utils.py", line 1430, in pixel_array
    return decoder.as_array(
           ^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/pydicom/pixels/decoders/base.py", line 977, in as_array
    runner.set_source(src)

  File "/usr/local/lib/python3.11/dist-packages/pydicom/pixels/decoders/base.py", line 686, in set_source
    self._set_options_ds(src)

  File "/usr/local/lib/python3.11/dist-packages/pydicom/pixels/decoders/base.py", line 659, in _set_options_ds
    raise AttributeError(

AttributeError: The dataset has no 'Pixel Data', 'Float Pixel Data' or 'Double Float Pixel Data' element, no pixel data to decode


	 [[{{node PyFunc}}]]
	 [[IteratorGetNext]] [Op:__inference_multi_step_on_iterator_2129]

## === cell 8
preds = (
    model.predict(
        test_dataset,
        verbose=1,
    )
    .reshape(-1)
    .astype(np.float32)
)

preds = np.clip(preds, 0.0, 1.0)
print("preds:", preds.shape, "min/max:", float(preds.min()), float(preds.max()))



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
UnknownError                              Traceback (most recent call last)
/tmp/ipykernel_11/2695764118.py in <cell line: 0>()
      1 preds = (
----> 2     model.predict(
      3         test_dataset,
      4         verbose=1,
      5     )

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

UnknownError: {{function_node __wrapped__IteratorGetNext_output_types_1_device_/job:localhost/replica:0/task:0/device:CPU:0}} Error in user-defined function passed to ParallelMapDatasetV2:12 transformation with iterator: Iterator::Root::Prefetch::BatchV2::MemoryCacheImpl::Map: AttributeError: The dataset has no 'Pixel Data', 'Float Pixel Data' or 'Double Float Pixel Data' element, no pixel data to decode
Traceback (most recent call last):

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 269, in __call__
    ret = func(*args)
          ^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py", line 643, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_11/1706923086.py", line 25, in _py_load_one
    x = load_dicom_tensor_4d(sid, split=split, mri_type=mri_type)  # (H,W,D,1)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_11/4077137452.py", line 241, in load_dicom_tensor_4d
    vol = load_dicom_images_3d(
          ^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_11/4077137452.py", line 203, in load_dicom_images_3d
    img3d[:, :, j] = load_dicom_image(f, img_size=img_size, rotate=rotate)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_11/4077137452.py", line 163, in load_dicom_image
    data = dicom.pixel_array.astype(np.float32, copy=False)
           ^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py", line 918, in __getattr__
    return object.__getattribute__(self, name)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py", line 2193, in pixel_array
    self.convert_pixel_data()

  File "/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py", line 1726, in convert_pixel_data
    self._pixel_array = pixel_array(self, **opts)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/pydicom/pixels/utils.py", line 1430, in pixel_array
    return decoder.as_array(
           ^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/pydicom/pixels/decoders/base.py", line 977, in as_array
    runner.set_source(src)

  File "/usr/local/lib/python3.11/dist-packages/pydicom/pixels/decoders/base.py", line 686, in set_source
    self._set_options_ds(src)

  File "/usr/local/lib/python3.11/dist-packages/pydicom/pixels/decoders/base.py", line 659, in _set_options_ds
    raise AttributeError(

AttributeError: The dataset has no 'Pixel Data', 'Float Pixel Data' or 'Double Float Pixel Data' element, no pixel data to decode


	 [[{{node PyFunc}}]] [Op:IteratorGetNext] name: 

## === cell 9
assert len(preds) == len(sample_submission), (len(preds), len(sample_submission))

submission = pd.DataFrame(
    {"BraTS21ID": sample_submission["BraTS21ID"].values, "MGMT_value": preds}
)

submission["BraTS21ID"] = submission["BraTS21ID"].astype(int)
submission["MGMT_value"] = submission["MGMT_value"].astype(float)

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv:", submission.shape)
submission.head()



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1462220479.py in <cell line: 0>()
----> 1 assert len(preds) == len(sample_submission), (len(preds), len(sample_submission))
      2 
      3 submission = pd.DataFrame(
      4     {"BraTS21ID": sample_submission["BraTS21ID"].values, "MGMT_value": preds}
      5 )

NameError: name 'preds' is not defined

## === cell 10
plt.figure(figsize=(6, 4))
plt.hist(submission["MGMT_value"], bins=20)
plt.title("Prediction distribution")
plt.show()
print(submission.describe())

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2509749413.py in <cell line: 0>()
      1 plt.figure(figsize=(6, 4))
----> 2 plt.hist(submission["MGMT_value"], bins=20)
      3 plt.title("Prediction distribution")
      4 plt.show()
      5 print(submission.describe())

NameError: name 'submission' is not defined
