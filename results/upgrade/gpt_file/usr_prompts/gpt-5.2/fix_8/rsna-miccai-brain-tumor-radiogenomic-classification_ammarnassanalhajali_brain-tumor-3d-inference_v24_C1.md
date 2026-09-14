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

- What this solution (achieved 0.5) has done: 'I fix the import/runtime crash by removing the unused `tensorflow_addons` dependency that triggers the protobuf `MessageFactory.GetPrototype` error in this environment. I fix the DICOM loader to use `pydicom.dcmread` (new API) and make `load_dicom_images_3d` robust when a folder has no matching `.dcm` files (returning a zero-volume instead of crashing). I also fix the dataset generator indexing/batching so `model.predict(test_dataset)` iterates correctly and always returns a properly shaped 5D tensor batch. Finally, because the external pre-trained `.h5` model path is missing, I keep the core “predict probabilities” semantics by outputting a valid submission using the train-label mean as a constant probability baseline (ensures an end-to-end runnable pipeline and a non-trivial AUC-ready submission).'
- What this solution (achieved 0.5) has done: 'I fix the environment crash happening at import time by removing/avoiding the TensorFlow protobuf `MessageFactory.GetPrototype` issue via safe TF/Keras imports and by dropping unnecessary visualization imports that can trigger incompatible proto deps. Then I ensure the pipeline always produces `submission.csv` in the exact required format, keeping your current core inference semantics (use pretrained model if present, otherwise constant mean baseline). Finally, I make the dataset/IO a bit more robust (path resolution and DICOM reading fallback) without changing the model logic, so it runs end-to-end reliably in Kaggle.'
- What this solution (achieved 0.5) has done: 'I fix the import-time crash caused by an incompatible `protobuf`/TensorFlow combination by forcing the pure-Python protobuf implementation before importing TensorFlow (a minimal environment-only change that does not alter model logic). I also add a safe fallback path so that if TensorFlow still cannot be imported, the script still complete end-to-end and write a valid `submission.csv` using the existing constant-mean baseline logic (score-neutral relative to your current 0.5 AUC baseline). Additionally, I keep all data paths and the DICOM loading logic the same, and ensure the submission has the required columns and row alignment.'

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

import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

from sklearn import model_selection as sk_model_selection
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

TF_AVAILABLE = False



## === cell 1
data_directory = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"
if not os.path.exists(data_directory):
    alt = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
    if os.path.exists(alt):
        data_directory = alt

mri_types = ["FLAIR", "T1w", "T1wCE", "T2w"]
IMAGE_SIZE = 256
NUM_IMAGES = 64

print("Using data_directory:", data_directory)



## === cell 2
sample_submission = pd.read_csv(f"{data_directory}/sample_submission.csv")
test = sample_submission.copy()
test["BraTS21ID5"] = [format(int(x), "05d") for x in test.BraTS21ID]

train_labels_path = f"{data_directory}/train_labels.csv"
train_labels = pd.read_csv(train_labels_path)
train_labels["BraTS21ID5"] = [format(int(x), "05d") for x in train_labels.BraTS21ID]

bad_ids = set([109, 123, 709])
train_labels = train_labels[~train_labels["BraTS21ID"].isin(bad_ids)].reset_index(
    drop=True
)

print("test rows:", len(test), "train rows:", len(train_labels))
train_labels.head(3)



## === cell 3
from functools import lru_cache

_IMAGE_NUM_RE = re.compile(r"Image-(\d+)\.dcm$", re.IGNORECASE)


def _file_sort_key_image_number(path: str) -> int:
    base = os.path.basename(path)
    m = _IMAGE_NUM_RE.search(base)
    if m:
        return int(m.group(1))
    m2 = re.search(r"(\d+)(?!.*\d)", base)
    return int(m2.group(1)) if m2 else 0


@lru_cache(maxsize=None)
def _get_sorted_dicom_files(split: str, scan_id: str, mri_type: str):
    pattern = f"{data_directory}/{split}/{scan_id}/{mri_type}/*.dcm"
    files = glob.glob(pattern)
    if not files:
        return ()
    files.sort(key=_file_sort_key_image_number)
    return tuple(files)


def load_dicom_image(path, img_size=IMAGE_SIZE, voi_lut=True, rotate=0):
    try:
        dicom = pydicom.dcmread(
            path,
            force=False,
            stop_before_pixels=False,
            specific_tags=(
                "PixelData",
                "RescaleSlope",
                "RescaleIntercept",
                "WindowCenter",
                "WindowWidth",
            ),
        )
    except Exception:
        dicom = pydicom.dcmread(
            path,
            force=True,
            stop_before_pixels=False,
            specific_tags=(
                "PixelData",
                "RescaleSlope",
                "RescaleIntercept",
                "WindowCenter",
                "WindowWidth",
            ),
        )

    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
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
    files = _get_sorted_dicom_files(split, scan_id, mri_type)

    if len(files) == 0:
        img3d = np.zeros((img_size, img_size, num_imgs), dtype=np.float32)
        return np.expand_dims(img3d, 0)

    middle = len(files) // 2
    num_imgs2 = num_imgs // 2
    p1 = max(0, middle - num_imgs2)
    p2 = min(len(files), middle + num_imgs2)

    if p2 <= p1:
        p1 = 0
        p2 = min(len(files), num_imgs)

    sel = files[p1:p2]

    img3d = np.zeros((img_size, img_size, num_imgs), dtype=np.float32)

    _ldi = load_dicom_image
    for j, f in enumerate(sel):
        img3d[:, :, j] = _ldi(f, img_size=img_size, rotate=rotate)

    mn, mx = float(np.min(img3d)), float(np.max(img3d))
    if mn < mx:
        img3d = (img3d - mn) / (mx - mn)

    return np.expand_dims(img3d, 0)




## === cell 4
scan_id5 = test.loc[0, "BraTS21ID5"]
a = load_dicom_images_3d(scan_id5, split="test", mri_type="FLAIR")
print("Loaded volume shape:", a.shape, "min/max:", float(np.min(a)), float(np.max(a)))




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2736710453.py in <cell line: 0>()
      1 scan_id5 = test.loc[0, "BraTS21ID5"]
----> 2 a = load_dicom_images_3d(scan_id5, split="test", mri_type="FLAIR")
      3 print("Loaded volume shape:", a.shape, "min/max:", float(np.min(a)), float(np.max(a)))
      4 
      5 

/tmp/ipykernel_11/2827013333.py in load_dicom_images_3d(scan_id, num_imgs, img_size, mri_type, split, rotate)
    104     _ldi = load_dicom_image
    105     for j, f in enumerate(sel):
--> 106         img3d[:, :, j] = _ldi(f, img_size=img_size, rotate=rotate)
    107 
    108     mn, mx = float(np.min(img3d)), float(np.max(img3d))

/tmp/ipykernel_11/2827013333.py in load_dicom_image(path, img_size, voi_lut, rotate)
     54 
     55     if voi_lut:
---> 56         data = apply_voi_lut(dicom.pixel_array, dicom)
     57     else:
     58         data = dicom.pixel_array

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
    988 
    989         if validate:
--> 990             runner.validate()
    991 
    992         if self.is_native:

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/decoders/base.py in validate(self)
    751     def validate(self) -> None:
    752         """Validate the decoding options and source buffer (if any)."""
--> 753         self._validate_options()
    754         if self.is_dataset or self.is_buffer:
    755             self._validate_buffer()

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/decoders/base.py in _validate_options(self)
    820     def _validate_options(self) -> None:
    821         """Validate the supplied options to ensure they meet minimum requirements."""
--> 822         super()._validate_options()
    823 
    824         # The Extended Offset Table is optional

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/common.py in _validate_options(self)
    573         prefix = "Missing required element: (0028"
    574         if self._opts.get("bits_allocated") is None:
--> 575             raise AttributeError(f"{prefix},0100) 'Bits Allocated'")
    576 
    577         if not 1 <= self.bits_allocated <= 64 or (

AttributeError: Missing required element: (0028,0100) 'Bits Allocated'

## === cell 5
def extract_volume_features(vol_3d):
    """
    vol_3d: np.ndarray shape (H, W, D) in [0,1] approximately (from loader)
    Returns: 1D feature vector (float32)
    """
    x = vol_3d.astype(np.float32, copy=False)
    feat = []

    feat.append(float(np.mean(x)))
    feat.append(float(np.std(x)))
    feat.append(float(np.min(x)))
    feat.append(float(np.max(x)))

    qs = np.array([1, 5, 10, 25, 50, 75, 90, 95, 99], dtype=np.float32)
    p_all = np.percentile(x, qs)
    feat.extend([float(v) for v in p_all])

    thr = float(np.percentile(x, 80))
    feat.append(float(np.mean(x > thr)))

    d = x.shape[-1]
    center_idx = d // 2
    center_slice = x[:, :, center_idx]
    feat.append(float(np.mean(center_slice)))
    feat.append(float(np.std(center_slice)))
    feat.append(float(np.percentile(center_slice, 90)))

    bins = 8
    edges = np.linspace(0, d, bins + 1).astype(int)
    for i in range(bins):
        s, e = edges[i], edges[i + 1]
        if e <= s:
            feat.append(0.0)
        else:
            feat.append(float(np.mean(x[:, :, s:e])))

    return np.asarray(feat, dtype=np.float32)


import multiprocessing as mp

_G = {}


def _init_pool(_data_directory, _image_size, _num_images):
    global data_directory, IMAGE_SIZE, NUM_IMAGES
    data_directory = _data_directory
    IMAGE_SIZE = _image_size
    NUM_IMAGES = _num_images


def _compute_one_feature(args):
    i, sid5, split, mri_type = args
    vol = load_dicom_images_3d(
        sid5,
        num_imgs=NUM_IMAGES,
        img_size=IMAGE_SIZE,
        mri_type=mri_type,
        split=split,
        rotate=0,
    )[0]
    return i, extract_volume_features(vol)


def build_features_for_df(df, split, mri_type="FLAIR"):
    n_features = 4 + 9 + 1 + 3 + 8
    X = np.zeros((len(df), n_features), dtype=np.float32)

    ids = df["BraTS21ID5"].values
    args = [(i, sid5, split, mri_type) for i, sid5 in enumerate(ids)]

    cpu = mp.cpu_count()
    n_workers = min(8, max(1, cpu - 1))

    if n_workers == 1:
        for i, sid5 in enumerate(ids):
            vol = load_dicom_images_3d(
                sid5,
                num_imgs=NUM_IMAGES,
                img_size=IMAGE_SIZE,
                mri_type=mri_type,
                split=split,
                rotate=0,
            )[0]
            X[i] = extract_volume_features(vol)
            if (i + 1) % 50 == 0 or (i + 1) == len(df):
                print(f"Features: {split} {mri_type} {i+1}/{len(df)}")
        return X

    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context("spawn")

    chunksize = 16
    with ctx.Pool(
        processes=n_workers,
        initializer=_init_pool,
        initargs=(data_directory, IMAGE_SIZE, NUM_IMAGES),
        maxtasksperchild=200,
    ) as pool:
        done = 0
        for i, feat in pool.imap_unordered(
            _compute_one_feature, args, chunksize=chunksize
        ):
            X[i] = feat
            done += 1
            if done % 50 == 0 or done == len(df):
                print(f"Features: {split} {mri_type} {done}/{len(df)}")
    return X




## === cell 6
t0 = time.time()
X_train = build_features_for_df(train_labels, split="train", mri_type="FLAIR")
y_train = train_labels["MGMT_value"].astype(int).values

X_test = build_features_for_df(test, split="test", mri_type="FLAIR")
print(
    "Feature shapes:",
    X_train.shape,
    X_test.shape,
    "time(s):",
    round(time.time() - t0, 2),
)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
RemoteTraceback                           Traceback (most recent call last)
RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/lib/python3.11/multiprocessing/pool.py", line 125, in worker
    result = (True, func(*args, **kwds))
                    ^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/multiprocessing/pool.py", line 48, in mapstar
    return list(map(*args))
           ^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_11/605564923.py", line 56, in _compute_one_feature
    vol = load_dicom_images_3d(
          ^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_11/2827013333.py", line 106, in load_dicom_images_3d
    img3d[:, :, j] = _ldi(f, img_size=img_size, rotate=rotate)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_11/2827013333.py", line 56, in load_dicom_image
    data = apply_voi_lut(dicom.pixel_array, dicom)
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
  File "/usr/local/lib/python3.11/dist-packages/pydicom/pixels/decoders/base.py", line 990, in as_array
    runner.validate()
  File "/usr/local/lib/python3.11/dist-packages/pydicom/pixels/decoders/base.py", line 753, in validate
    self._validate_options()
  File "/usr/local/lib/python3.11/dist-packages/pydicom/pixels/decoders/base.py", line 822, in _validate_options
    super()._validate_options()
  File "/usr/local/lib/python3.11/dist-packages/pydicom/pixels/common.py", line 575, in _validate_options
    raise AttributeError(f"{prefix},0100) 'Bits Allocated'")
AttributeError: Missing required element: (0028,0100) 'Bits Allocated'
"""

The above exception was the direct cause of the following exception:

AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1186976349.py in <cell line: 0>()
      1 t0 = time.time()
----> 2 X_train = build_features_for_df(train_labels, split="train", mri_type="FLAIR")
      3 y_train = train_labels["MGMT_value"].astype(int).values
      4 
      5 X_test = build_features_for_df(test, split="test", mri_type="FLAIR")

/tmp/ipykernel_11/605564923.py in build_features_for_df(df, split, mri_type)
    106     ) as pool:
    107         done = 0
--> 108         for i, feat in pool.imap_unordered(
    109             _compute_one_feature, args, chunksize=chunksize
    110         ):

/usr/lib/python3.11/multiprocessing/pool.py in <genexpr>(.0)
    449                     result._set_length
    450                 ))
--> 451             return (item for chunk in result for item in chunk)
    452 
    453     def apply_async(self, func, args=(), kwds={}, callback=None,

/usr/lib/python3.11/multiprocessing/pool.py in next(self, timeout)
    871         if success:
    872             return value
--> 873         raise value
    874 
    875     __next__ = next                    # XXX

AttributeError: Missing required element: (0028,0100) 'Bits Allocated'

## === cell 7
skf = sk_model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

oof = np.zeros(len(train_labels), dtype=np.float32)
test_pred = np.zeros(len(test), dtype=np.float32)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), 1):
    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    X_va, y_va = X_train[va_idx], y_train[va_idx]

    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
            random_state=SEED,
            class_weight=None,
        ),
    )
    clf.fit(X_tr, y_tr)

    oof[va_idx] = clf.predict_proba(X_va)[:, 1].astype(np.float32)
    test_pred += clf.predict_proba(X_test)[:, 1].astype(np.float32) / skf.get_n_splits()

    print(
        f"Fold {fold} done. OOF mean so far: {float(oof[oof>0].mean()) if np.any(oof>0) else 0.0:.4f}"
    )

preds = np.clip(test_pred, 0.0, 1.0)
print(
    "preds shape:",
    preds.shape,
    "min/max/mean:",
    float(preds.min()),
    float(preds.max()),
    float(preds.mean()),
)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2820681492.py in <cell line: 0>()
      4 test_pred = np.zeros(len(test), dtype=np.float32)
      5 
----> 6 for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), 1):
      7     X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
      8     X_va, y_va = X_train[va_idx], y_train[va_idx]

NameError: name 'X_train' is not defined

## === cell 8
submission = pd.DataFrame(
    {
        "BraTS21ID": sample_submission["BraTS21ID"].astype(int).values,
        "MGMT_value": preds.astype(float),
    }
)

assert len(submission) == len(sample_submission), "Submission length mismatch"
assert list(submission.columns) == ["BraTS21ID", "MGMT_value"]

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2641831905.py in <cell line: 0>()
      2     {
      3         "BraTS21ID": sample_submission["BraTS21ID"].astype(int).values,
----> 4         "MGMT_value": preds.astype(float),
      5     }
      6 )

NameError: name 'preds' is not defined

## === cell 9
print("Submission columns:", list(submission.columns))
print("MGMT_value summary:", submission["MGMT_value"].describe())
print(
    "File exists:",
    os.path.exists("submission.csv"),
    "size:",
    os.path.getsize("submission.csv") if os.path.exists("submission.csv") else None,
)

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/678261837.py in <cell line: 0>()
----> 1 print("Submission columns:", list(submission.columns))
      2 print("MGMT_value summary:", submission["MGMT_value"].describe())
      3 print(
      4     "File exists:",
      5     os.path.exists("submission.csv"),

NameError: name 'submission' is not defined
