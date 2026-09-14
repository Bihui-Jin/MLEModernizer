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

tf = None
keras = None
layers = None
ModelCheckpoint = None
EarlyStopping = None
AUC = None



## === cell 1
os.environ["PYTHONHASHSEED"] = "42"
np.random.seed(42)



## === cell 2
data_directory = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"

mri_types_orig = ["FLAIR", "T1w", "T1wCE", "T2w"]
mri_types = ["FLAIR", "T1w", "T1wCE", "T2w"]

IMAGE_SIZE = 64
NUM_IMAGES_PER_TYPE = 32
NUM_IMAGES = NUM_IMAGES_PER_TYPE * len(mri_types)
BATCH_SIZE = 4

train_df = pd.read_csv(
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv"
)

to_exclude = [109, 123, 709]
train_df = train_df[~train_df["BraTS21ID"].isin(to_exclude)].reset_index(drop=True)

train_df["BraTS21ID5"] = [format(x, "05d") for x in train_df.BraTS21ID]
print(len(train_df))
train_df.head(3)



## === cell 3
sample_submission = pd.read_csv(
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv"
)
test = sample_submission.copy()
test["BraTS21ID5"] = [format(x, "05d") for x in test.BraTS21ID]
test.head(3)




## === cell 4
def _natural_key(path):
    return [int(x) if x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", path)]




## === cell 5
from functools import lru_cache


@lru_cache(maxsize=200000)
def _cached_dicom_slice(path, img_size, voi_lut, rotate):
    dicom = pydicom.dcmread(
        path,
        stop_before_pixels=False,
        force=True,
        specific_tags=[
            (0x0028, 0x1050),  # WindowCenter
            (0x0028, 0x1051),  # WindowWidth
            (0x0028, 0x1052),  # RescaleIntercept
            (0x0028, 0x1053),  # RescaleSlope
            (0x0028, 0x0004),  # PhotometricInterpretation
            (0x0028, 0x0100),  # BitsAllocated
            (0x0028, 0x0101),  # BitsStored
            (0x0028, 0x0102),  # HighBit
            (0x0028, 0x0103),  # PixelRepresentation
            (0x7FE0, 0x0010),  # PixelData
        ],
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


@lru_cache(maxsize=40000)
def _cached_sorted_files(scan_id, split, mri_type):
    files = glob.glob(f"{data_directory}/{split}/{scan_id}/{mri_type}/*.dcm")
    if not files:
        return tuple()
    files = sorted(files, key=_natural_key)
    return tuple(files)


def load_dicom_image(path, img_size=IMAGE_SIZE, voi_lut=True, rotate=0):
    return _cached_dicom_slice(path, img_size, voi_lut, rotate)


@lru_cache(maxsize=30000)
def _cached_volume(scan_id, split, mri_type, num_imgs, img_size, rotate):
    files = _cached_sorted_files(scan_id, split, mri_type)

    if len(files) == 0:
        img3d = np.zeros((img_size, img_size, num_imgs), dtype=np.float32)
        return np.expand_dims(img3d, 0)

    middle = len(files) // 2
    num_imgs2 = num_imgs // 2
    p1 = max(0, middle - num_imgs2)
    p2 = min(len(files), middle + num_imgs2)

    img3d = np.stack(
        [load_dicom_image(f, img_size=img_size, rotate=rotate) for f in files[p1:p2]],
        axis=0,
    ).transpose(1, 2, 0)

    if img3d.shape[-1] < num_imgs:
        n_front = (num_imgs - img3d.shape[-1]) // 2
        n_back = num_imgs - img3d.shape[-1] - n_front
        n_zero_front = np.zeros((img_size, img_size, n_front), dtype=img3d.dtype)
        n_zero_back = np.zeros((img_size, img_size, n_back), dtype=img3d.dtype)
        img3d = np.concatenate((n_zero_front, img3d, n_zero_back), axis=-1)

    img3d = img3d.astype(np.float32)

    vmin = float(np.min(img3d))
    vmax = float(np.max(img3d))
    if vmin < vmax:
        img3d = (img3d - vmin) / (vmax - vmin)
    else:
        img3d = np.zeros_like(img3d, dtype=np.float32)

    return np.expand_dims(img3d, 0)


def load_dicom_images_3d(
    scan_id,
    split,
    mri_type="FLAIR",
    num_imgs=NUM_IMAGES_PER_TYPE,
    img_size=IMAGE_SIZE,
    rotate=0,
):
    return _cached_volume(scan_id, split, mri_type, num_imgs, img_size, rotate)


@lru_cache(maxsize=20000)
def _cached_volume_all(scan_id, split, num_imgs, img_size, rotate):
    img3d_all = np.concatenate(
        [
            load_dicom_images_3d(scan_id, split, mri_type, num_imgs, img_size, rotate)
            for mri_type in mri_types
        ],
        axis=-1,
    )
    return img3d_all


def load_dicom_images_3d_all(scan_id, split):
    return _cached_volume_all(scan_id, split, NUM_IMAGES_PER_TYPE, IMAGE_SIZE, 0)


a = load_dicom_images_3d_all("00000", "train")
print(a.shape)
print(np.min(a), np.max(a), np.mean(a), np.median(a))



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1609788402.py in <cell line: 0>()
    121 
    122 
--> 123 a = load_dicom_images_3d_all("00000", "train")
    124 print(a.shape)
    125 print(np.min(a), np.max(a), np.mean(a), np.median(a))

/tmp/ipykernel_11/1609788402.py in load_dicom_images_3d_all(scan_id, split)
    118 
    119 def load_dicom_images_3d_all(scan_id, split):
--> 120     return _cached_volume_all(scan_id, split, NUM_IMAGES_PER_TYPE, IMAGE_SIZE, 0)
    121 
    122 

/tmp/ipykernel_11/1609788402.py in _cached_volume_all(scan_id, split, num_imgs, img_size, rotate)
    108 def _cached_volume_all(scan_id, split, num_imgs, img_size, rotate):
    109     img3d_all = np.concatenate(
--> 110         [
    111             load_dicom_images_3d(scan_id, split, mri_type, num_imgs, img_size, rotate)
    112             for mri_type in mri_types

/tmp/ipykernel_11/1609788402.py in <listcomp>(.0)
    109     img3d_all = np.concatenate(
    110         [
--> 111             load_dicom_images_3d(scan_id, split, mri_type, num_imgs, img_size, rotate)
    112             for mri_type in mri_types
    113         ],

/tmp/ipykernel_11/1609788402.py in load_dicom_images_3d(scan_id, split, mri_type, num_imgs, img_size, rotate)
    102     rotate=0,
    103 ):
--> 104     return _cached_volume(scan_id, split, mri_type, num_imgs, img_size, rotate)
    105 
    106 

/tmp/ipykernel_11/1609788402.py in _cached_volume(scan_id, split, mri_type, num_imgs, img_size, rotate)
     71 
     72     img3d = np.stack(
---> 73         [load_dicom_image(f, img_size=img_size, rotate=rotate) for f in files[p1:p2]],
     74         axis=0,
     75     ).transpose(1, 2, 0)

/tmp/ipykernel_11/1609788402.py in <listcomp>(.0)
     71 
     72     img3d = np.stack(
---> 73         [load_dicom_image(f, img_size=img_size, rotate=rotate) for f in files[p1:p2]],
     74         axis=0,
     75     ).transpose(1, 2, 0)

/tmp/ipykernel_11/1609788402.py in load_dicom_image(path, img_size, voi_lut, rotate)
     54 
     55 def load_dicom_image(path, img_size=IMAGE_SIZE, voi_lut=True, rotate=0):
---> 56     return _cached_dicom_slice(path, img_size, voi_lut, rotate)
     57 
     58 

/tmp/ipykernel_11/1609788402.py in _cached_dicom_slice(path, img_size, voi_lut, rotate)
     27     )
     28     if voi_lut:
---> 29         data = apply_voi_lut(dicom.pixel_array, dicom)
     30     else:
     31         data = dicom.pixel_array

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
    596 
    597         if self._opts.get("columns") is None:
--> 598             raise AttributeError(f"{prefix},0011) 'Columns'")
    599 
    600         if not 0 < self.columns <= 2**16 - 1:

AttributeError: Missing required element: (0028,0011) 'Columns'

## === cell 6
from tensorflow.keras.utils import Sequence


class Dataset(Sequence):
    def __init__(self, df, split, is_train=True, batch_size=BATCH_SIZE, shuffle=True):
        self.idx = df["BraTS21ID"].values
        self.paths = df["BraTS21ID5"].values
        self.y = (
            df["MGMT_value"].values
            if (is_train and "MGMT_value" in df.columns)
            else None
        )
        self.is_train = is_train
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.split = split

    def __len__(self):
        return math.ceil(len(self.idx) / self.batch_size)

    def __getitem__(self, ids):
        batch_paths = self.paths[ids * self.batch_size : (ids + 1) * self.batch_size]
        split = self.split

        if self.y is not None:
            batch_y = self.y[
                ids * self.batch_size : (ids + 1) * self.batch_size
            ].astype(np.float32)

        list_x = [load_dicom_images_3d_all(x, split) for x in batch_paths]
        batch_X = np.stack(list_x, axis=0).astype(np.float32)  # (B, 1, H, W, D)
        batch_X = np.transpose(batch_X, (0, 2, 3, 4, 1))  # (B, H, W, D, 1)

        if self.is_train:
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




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 7
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.metrics import AUC

tf.random.set_seed(42)



## === cell 8
df_train, df_valid = sk_model_selection.train_test_split(
    train_df,
    test_size=0.2,
    random_state=42,
    stratify=train_df["MGMT_value"],
)



## === cell 9
df_train.head()



## === cell 10
train_dataset = Dataset(
    df_train, "train", is_train=True, batch_size=BATCH_SIZE, shuffle=True
)
valid_dataset = Dataset(
    df_valid, "train", is_train=True, batch_size=BATCH_SIZE, shuffle=False
)



## === cell 11
del train_df



## === cell 12
images, label = train_dataset[0]
print("Train batch X shape:", images.shape)  # expected: (B, 64, 64, 128, 1)
print("Train batch y shape:", label.shape)




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3482714690.py in <cell line: 0>()
----> 1 images, label = train_dataset[0]
      2 print("Train batch X shape:", images.shape)  # expected: (B, 64, 64, 128, 1)
      3 print("Train batch y shape:", label.shape)
      4 
      5 

/tmp/ipykernel_11/2710265214.py in __getitem__(self, ids)
     28             ].astype(np.float32)
     29 
---> 30         list_x = [load_dicom_images_3d_all(x, split) for x in batch_paths]
     31         batch_X = np.stack(list_x, axis=0).astype(np.float32)  # (B, 1, H, W, D)
     32         batch_X = np.transpose(batch_X, (0, 2, 3, 4, 1))  # (B, H, W, D, 1)

/tmp/ipykernel_11/2710265214.py in <listcomp>(.0)
     28             ].astype(np.float32)
     29 
---> 30         list_x = [load_dicom_images_3d_all(x, split) for x in batch_paths]
     31         batch_X = np.stack(list_x, axis=0).astype(np.float32)  # (B, 1, H, W, D)
     32         batch_X = np.transpose(batch_X, (0, 2, 3, 4, 1))  # (B, H, W, D, 1)

/tmp/ipykernel_11/1609788402.py in load_dicom_images_3d_all(scan_id, split)
    118 
    119 def load_dicom_images_3d_all(scan_id, split):
--> 120     return _cached_volume_all(scan_id, split, NUM_IMAGES_PER_TYPE, IMAGE_SIZE, 0)
    121 
    122 

/tmp/ipykernel_11/1609788402.py in _cached_volume_all(scan_id, split, num_imgs, img_size, rotate)
    108 def _cached_volume_all(scan_id, split, num_imgs, img_size, rotate):
    109     img3d_all = np.concatenate(
--> 110         [
    111             load_dicom_images_3d(scan_id, split, mri_type, num_imgs, img_size, rotate)
    112             for mri_type in mri_types

/tmp/ipykernel_11/1609788402.py in <listcomp>(.0)
    109     img3d_all = np.concatenate(
    110         [
--> 111             load_dicom_images_3d(scan_id, split, mri_type, num_imgs, img_size, rotate)
    112             for mri_type in mri_types
    113         ],

/tmp/ipykernel_11/1609788402.py in load_dicom_images_3d(scan_id, split, mri_type, num_imgs, img_size, rotate)
    102     rotate=0,
    103 ):
--> 104     return _cached_volume(scan_id, split, mri_type, num_imgs, img_size, rotate)
    105 
    106 

/tmp/ipykernel_11/1609788402.py in _cached_volume(scan_id, split, mri_type, num_imgs, img_size, rotate)
     71 
     72     img3d = np.stack(
---> 73         [load_dicom_image(f, img_size=img_size, rotate=rotate) for f in files[p1:p2]],
     74         axis=0,
     75     ).transpose(1, 2, 0)

/tmp/ipykernel_11/1609788402.py in <listcomp>(.0)
     71 
     72     img3d = np.stack(
---> 73         [load_dicom_image(f, img_size=img_size, rotate=rotate) for f in files[p1:p2]],
     74         axis=0,
     75     ).transpose(1, 2, 0)

/tmp/ipykernel_11/1609788402.py in load_dicom_image(path, img_size, voi_lut, rotate)
     54 
     55 def load_dicom_image(path, img_size=IMAGE_SIZE, voi_lut=True, rotate=0):
---> 56     return _cached_dicom_slice(path, img_size, voi_lut, rotate)
     57 
     58 

/tmp/ipykernel_11/1609788402.py in _cached_dicom_slice(path, img_size, voi_lut, rotate)
     27     )
     28     if voi_lut:
---> 29         data = apply_voi_lut(dicom.pixel_array, dicom)
     30     else:
     31         data = dicom.pixel_array

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
    596 
    597         if self._opts.get("columns") is None:
--> 598             raise AttributeError(f"{prefix},0011) 'Columns'")
    599 
    600         if not 0 < self.columns <= 2**16 - 1:

AttributeError: Missing required element: (0028,0011) 'Columns'

## === cell 13
def plot_sample_all(images, label, j):
    plt.figure(figsize=(35, 35))
    for i in range(NUM_IMAGES):
        plt.subplot(16, 16, (i + 1))
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[0, :, :, i, j], cmap="gray")
    plt.show()




## === cell 14
DO_PLOTS = False
if DO_PLOTS:
    i = 0
    j = 0
    images, label = train_dataset[i]
    plot_sample_all(images, label, j)



## === cell 15
if DO_PLOTS:
    i = 0
    j = 1
    images, label = train_dataset[i]
    plot_sample_all(images, label, j)



## === cell 16
if DO_PLOTS:
    i = 0
    j = 0
    images, label = train_dataset[i]
    plot_sample_all(images, label, j)



## === cell 17
if DO_PLOTS:
    i = 0
    j = 0
    images, label = train_dataset[i]
    plot_sample_all(images, label, j)




## === cell 18
def plot_sample_train(images, label):
    plt.figure(figsize=(16, 16))
    idx_base = int(NUM_IMAGES_PER_TYPE / 2)
    idx = [idx_base, idx_base * 3]
    for i in range(len(idx) * BATCH_SIZE):
        plt.subplot(BATCH_SIZE, len(idx), i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        j = int(i / len(idx))
        plt.imshow(images[0, :, :, idx[i % len(idx)], j], cmap="gray")
        plt.xlabel(f"{idx[i % len(idx)]} {label[j]}")
    plt.show()




## === cell 19
if DO_PLOTS:
    i = 0
    images, label = train_dataset[i]
    print("Dimension of the scan batch is:", images.shape)
    print("label=", label, label.shape)
    plot_sample_train(images, label)




## === cell 20
def get_model(width=IMAGE_SIZE, height=IMAGE_SIZE, depth=NUM_IMAGES):
    """Build a 3D convolutional neural network model."""
    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="3D_CNN")
    return model




## === cell 21
model = get_model()
model.summary()



## === cell 22
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=[AUC(name="auc")],
)




## === cell 23
def sequence_to_tfdata(seq, is_train):
    output_signature = (
        (
            tf.TensorSpec(
                shape=(None, IMAGE_SIZE, IMAGE_SIZE, NUM_IMAGES, 1), dtype=tf.float32
            ),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
        )
        if is_train
        else tf.TensorSpec(
            shape=(None, IMAGE_SIZE, IMAGE_SIZE, NUM_IMAGES, 1), dtype=tf.float32
        )
    )

    def gen():
        for i in range(len(seq)):
            yield seq[i]

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


train_ds_tf = sequence_to_tfdata(train_dataset, is_train=True)
valid_ds_tf = sequence_to_tfdata(valid_dataset, is_train=True)



## === cell 24
weights_path = "../input/brain-tumor-model-v3-2-1/Brain_Tumor_All_MRI_3D_CNN_v3_2_1.h5"
trained_now = False
if os.path.exists(weights_path):
    model.load_weights(weights_path)
    print("Loaded pretrained weights:", weights_path)
else:
    print(
        "Pretrained weights not found; training model from scratch in this environment."
    )
    ckpt_path = "best_model.weights.h5"
    callbacks = [
        ModelCheckpoint(
            ckpt_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
    ]
    history = model.fit(
        train_ds_tf,
        validation_data=valid_ds_tf,
        epochs=8,
        callbacks=callbacks,
        verbose=2,
    )
    if os.path.exists(ckpt_path):
        model.load_weights(ckpt_path)
    trained_now = True




## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
UnknownError                              Traceback (most recent call last)
/tmp/ipykernel_11/4109973184.py in <cell line: 0>()
     26         ),
     27     ]
---> 28     history = model.fit(
     29         train_ds_tf,
     30         validation_data=valid_ds_tf,

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
AttributeError: Missing required element: (0028,0011) 'Columns'
Traceback (most recent call last):

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 269, in __call__
    ret = func(*args)
          ^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py", line 643, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/from_generator_op.py", line 198, in generator_py_func
    values = next(generator_state.get_iterator(iterator_id))
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_11/1306369650.py", line 20, in gen
    yield seq[i]
          ~~~^^^

  File "/tmp/ipykernel_11/2710265214.py", line 30, in __getitem__
    list_x = [load_dicom_images_3d_all(x, split) for x in batch_paths]
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_11/2710265214.py", line 30, in <listcomp>
    list_x = [load_dicom_images_3d_all(x, split) for x in batch_paths]
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_11/1609788402.py", line 120, in load_dicom_images_3d_all
    return _cached_volume_all(scan_id, split, NUM_IMAGES_PER_TYPE, IMAGE_SIZE, 0)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_11/1609788402.py", line 110, in _cached_volume_all
    [

  File "/tmp/ipykernel_11/1609788402.py", line 111, in <listcomp>
    load_dicom_images_3d(scan_id, split, mri_type, num_imgs, img_size, rotate)

  File "/tmp/ipykernel_11/1609788402.py", line 104, in load_dicom_images_3d
    return _cached_volume(scan_id, split, mri_type, num_imgs, img_size, rotate)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_11/1609788402.py", line 73, in _cached_volume
    [load_dicom_image(f, img_size=img_size, rotate=rotate) for f in files[p1:p2]],
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_11/1609788402.py", line 73, in <listcomp>
    [load_dicom_image(f, img_size=img_size, rotate=rotate) for f in files[p1:p2]],
     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_11/1609788402.py", line 56, in load_dicom_image
    return _cached_dicom_slice(path, img_size, voi_lut, rotate)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_11/1609788402.py", line 29, in _cached_dicom_slice
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

  File "/usr/local/lib/python3.11/dist-packages/pydicom/pixels/common.py", line 598, in _validate_options
    raise AttributeError(f"{prefix},0011) 'Columns'")

AttributeError: Missing required element: (0028,0011) 'Columns'


	 [[{{node PyFunc}}]]
	 [[IteratorGetNext]] [Op:__inference_multi_step_on_iterator_6362]

## === cell 25
class DatasetTest(Sequence):
    def __init__(self, df, split, is_train=False, batch_size=1, shuffle=False):
        self.idx = df["BraTS21ID"].values
        self.paths = df["BraTS21ID5"].values
        self.y = None
        self.is_train = is_train
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.split = split

    def __len__(self):
        return math.ceil(len(self.idx) / self.batch_size)

    def __getitem__(self, ids):
        batch_paths = self.paths[ids * self.batch_size : (ids + 1) * self.batch_size]
        split = self.split

        list_x = [load_dicom_images_3d_all(p, split) for p in batch_paths]
        batch_X = np.stack(list_x, axis=0).astype(np.float32)  # (B, 1, H, W, D)
        batch_X = np.transpose(batch_X, (0, 2, 3, 4, 1))  # (B, H, W, D, 1)
        return batch_X




## === cell 26
test_dataset = DatasetTest(
    df=test, split="test", is_train=False, batch_size=1, shuffle=False
)




## === cell 27
def plot_sample_test(images):
    plt.figure(figsize=(16, 16))
    idx_base = int(NUM_IMAGES_PER_TYPE / 2)
    idx = [idx_base, idx_base * 3]
    for i in range(len(idx)):
        plt.subplot(1, len(idx), i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[0, :, :, idx[i], 0], cmap="gray")
        plt.xlabel(idx[i])
    plt.show()




## === cell 28
if DO_PLOTS:
    i = 0
    images = test_dataset[i]
    print("Dimension of the test scan batch is:", images.shape)
    plot_sample_test(images)



## === cell 29
test_ds_tf = sequence_to_tfdata(test_dataset, is_train=False)



## === cell 30
predictions = model.predict(test_ds_tf, verbose=1).reshape(-1).astype(np.float64)
print(
    "Predictions:",
    predictions[:5],
    "min/max:",
    predictions.min(),
    predictions.max(),
    "n=",
    len(predictions),
)



## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
UnknownError                              Traceback (most recent call last)
/tmp/ipykernel_11/340935614.py in <cell line: 0>()
----> 1 predictions = model.predict(test_ds_tf, verbose=1).reshape(-1).astype(np.float64)
      2 print(
      3     "Predictions:",
      4     predictions[:5],
      5     "min/max:",

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

UnknownError: {{function_node __wrapped__IteratorGetNext_output_types_1_device_/job:localhost/replica:0/task:0/device:CPU:0}} AttributeError: Missing required element: (0028,0011) 'Columns'
Traceback (most recent call last):

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 269, in __call__
    ret = func(*args)
          ^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py", line 643, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/from_generator_op.py", line 198, in generator_py_func
    values = next(generator_state.get_iterator(iterator_id))
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_11/1306369650.py", line 20, in gen
    yield seq[i]
          ~~~^^^

  File "/tmp/ipykernel_11/291632526.py", line 18, in __getitem__
    list_x = [load_dicom_images_3d_all(p, split) for p in batch_paths]
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_11/291632526.py", line 18, in <listcomp>
    list_x = [load_dicom_images_3d_all(p, split) for p in batch_paths]
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_11/1609788402.py", line 120, in load_dicom_images_3d_all
    return _cached_volume_all(scan_id, split, NUM_IMAGES_PER_TYPE, IMAGE_SIZE, 0)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_11/1609788402.py", line 110, in _cached_volume_all
    [

  File "/tmp/ipykernel_11/1609788402.py", line 111, in <listcomp>
    load_dicom_images_3d(scan_id, split, mri_type, num_imgs, img_size, rotate)

  File "/tmp/ipykernel_11/1609788402.py", line 104, in load_dicom_images_3d
    return _cached_volume(scan_id, split, mri_type, num_imgs, img_size, rotate)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_11/1609788402.py", line 73, in _cached_volume
    [load_dicom_image(f, img_size=img_size, rotate=rotate) for f in files[p1:p2]],
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_11/1609788402.py", line 73, in <listcomp>
    [load_dicom_image(f, img_size=img_size, rotate=rotate) for f in files[p1:p2]],
     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_11/1609788402.py", line 56, in load_dicom_image
    return _cached_dicom_slice(path, img_size, voi_lut, rotate)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_11/1609788402.py", line 29, in _cached_dicom_slice
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

  File "/usr/local/lib/python3.11/dist-packages/pydicom/pixels/common.py", line 598, in _validate_options
    raise AttributeError(f"{prefix},0011) 'Columns'")

AttributeError: Missing required element: (0028,0011) 'Columns'


	 [[{{node PyFunc}}]] [Op:IteratorGetNext] name: 

## === cell 31
submission = pd.DataFrame(
    {"BraTS21ID": sample_submission["BraTS21ID"].values, "MGMT_value": predictions}
)



## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1807307943.py in <cell line: 0>()
      1 submission = pd.DataFrame(
----> 2     {"BraTS21ID": sample_submission["BraTS21ID"].values, "MGMT_value": predictions}
      3 )
      4 

NameError: name 'predictions' is not defined

## === cell 32
submission.head()



## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2023294942.py in <cell line: 0>()
----> 1 submission.head()
      2 

NameError: name 'submission' is not defined

## === cell 33
submission["BraTS21ID"] = submission["BraTS21ID"].astype(
    sample_submission["BraTS21ID"].dtype
)



## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2760812397.py in <cell line: 0>()
----> 1 submission["BraTS21ID"] = submission["BraTS21ID"].astype(
      2     sample_submission["BraTS21ID"].dtype
      3 )
      4 

NameError: name 'submission' is not defined

## === cell 34
submission.head()



## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2023294942.py in <cell line: 0>()
----> 1 submission.head()
      2 

NameError: name 'submission' is not defined

## === cell 35
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())

## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3000660677.py in <cell line: 0>()
----> 1 submission.to_csv("submission.csv", index=False)
      2 print("Wrote submission.csv with shape:", submission.shape)
      3 print(submission.head())

NameError: name 'submission' is not defined
