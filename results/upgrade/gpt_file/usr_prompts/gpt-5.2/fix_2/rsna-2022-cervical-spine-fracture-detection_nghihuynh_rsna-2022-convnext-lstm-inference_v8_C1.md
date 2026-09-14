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
Identify fractures in CT scans of the cervical spine (neck) at both the level of a single vertebrae and the entire patient.

## Metric
Weighted multi-label logarithmic loss. Each fracture sub-type is its own row for every exam, and you are expected to predict a probability for a fracture at each of the seven cervical vertebrae designated as C1, C2, C3, C4, C5, C6 and C7. There is also an any label, `patient_overall`, which indicates that a fracture of ANY kind described before exists in the examination. Fractures in the skull base, thoracic spine, ribs, and clavicles are ignored. The any label is weighted more highly than specific fracture level sub-types.

For each exam Id, you must submit a set of predicted probabilities (a separate row for each cervical level subtype). We then take the log loss for each predicted probability versus its true label.

The binary weighted log loss function for label j on exam i is specified as:

$$
L_{i j}=-w_j *\left[y_{i j} * \log \left(p_{i j}\right)+\left(1-y_{i j}\right) * \log \left(1-p_{i j}\right)\right]
$$

Finally, loss is averaged across all rows.

## Submission Format
There will be 8 rows per image Id. The label indicated by a particular row will look like [image Id]_[Sub-type Name], as follows. There is also a target column, `fractured`, indicating the probability of whether a fracture exists at the specified level. For each image ID in the test set, you must predict a probability for each of the different possible sub-types and the patient overall. The file should contain a header and have the following format:

```
row_id,fractured
1_C1,0
1_C2,0
1_C3,0
1_C4,0.6
1_C5,0
1_C6,0.9
1_C7,0.01
1_patient_overall,0.99
2_C1,0
etc.
```

## Dataset
**train.csv** Metadata for the train test set.

- `StudyInstanceUID` - The study ID. There is one unique study ID for each patient scan.
- `patient_overall` - One of the target columns. The patient level outcome, i.e. if any of the vertebrae are fractured.
- `C[1-7]` - The other target columns. Whether the given vertebrae is fractured. See [this diagram](https://en.wikipedia.org/wiki/Vertebral_column#/media/File:Gray_111_-_Vertebral_column-coloured.png) for the real location of each vertbrae in the spine.

**test.csv** Metadata for the test set prediction structure. Only the first few rows of the test set are available for download.

- `row_id` - The row ID. This will match the same column in the sample submission file.
- `StudyInstanceUID` - The study ID.
- `prediction_type` - Which one of the eight target columns needs a prediction in this row.

**[train/test]_images/[StudyInstanceUID]/[slice_number].dcm** The image data, organized with one folder per scan. Expect to see roughly 1,500 scans in the hidden test set.\

Each image is in [the dicom file format](https://www.dicomstandard.org/). The DICOM image files are ≤ 1 mm slice thickness, axial orientation, and bone kernel. Note that some of the DICOM files are JPEG compressed. You may require additional resources to read the pixel array of these files, such as GDCM and pylibjpeg.

**sample_submission.csv** A valid sample submission.

- `row_id` - The row ID. See the test.csv for what prediction needs to be filed in that row.
- `fractured` - The target column.

**train_bounding_boxes.csv** Bounding boxes for a subset of the training set.

**segmentations/** Pixel level annotations for a subset of the training set. This data is provided in the [nifti file format](https://nifti.nimh.nih.gov/).

A portion of the imaging datasets have been segmented automatically using a 3D UNET model, and radiologists modified and approved the segmentations. The provided segmentation labels have values of 1 to 7 for C1 to C7 (seven cervical vertebrae) and 8 to 19 for T1 to T12 (twelve thoracic vertebrae are located in the center of your upper and middle back), and 0 for everything else. As we focused on the cervical spine, all scans have C1 to C7 labels but not all thoracic labels.

Please be aware that the NIFTI files consist of segmentation in the sagittal plane, while the DICOM files are in the axial plane. Please use the NIFTI header information to determine the appropriate orientation such that the DICOM images and segmentation match. Otherwise, you run the risk of having the segmentations flipped in the Z axis and mirrored in the X axis.

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (276 lines)
            sample_submission.csv (14537 lines)
            sample_submission.csv.zip (50.3 kB)
            segmentations.zip (3.0 MB)
            test.csv (14537 lines)
            test.csv.zip (94.5 kB)
            test.zip (160 Bytes)
            test_images.zip (181.9 GB)
            train.csv (203 lines)
            train.csv.zip (1.2 kB)
            train.zip (162 Bytes)
            train_bounding_boxes.csv (691 lines)
            train_bounding_boxes.csv.zip (11.5 kB)
            train_images.zip (20.3 GB)
            rsna-2022-cervical-spine-fracture-detection/
                description.md (276 lines)
                sample_submission.csv (14537 lines)
                ... and 12 other files
                rsna-2022-cervical-spine-fracture-detection/
                segmentations/
                    1.2.826.0.1.3680043.12292.nii (89.4 MB)
                    1.2.826.0.1.3680043.24617.nii (307.2 MB)
                    ... and 7 other files
                test_images/
                    1.2.826.0.1.3680043.10001/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 266 other files
                    1.2.826.0.1.3680043.10005/
                        1.dcm (525.1 kB)
                        10.dcm (525.1 kB)
                        ... and 257 other files
                    ... and 1816 other folders
                train_images/
                    1.2.826.0.1.3680043.10014/
                        1.dcm (240.9 kB)
                        10.dcm (250.8 kB)
                        ... and 256 other files
                    1.2.826.0.1.3680043.10058/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 574 other files
                    ... and 201 other folders
            segmentations/
                1.2.826.0.1.3680043.12292.nii (89.4 MB)
                1.2.826.0.1.3680043.24617.nii (307.2 MB)
                ... and 7 other files
            test_images/
                1.2.826.0.1.3680043.10001/
                    1.dcm (525.0 kB)
                    10.dcm (525.0 kB)
                    ... and 266 other files
                1.2.826.0.1.3680043.10005/
                    1.dcm (525.1 kB)
                    10.dcm (525.1 kB)
                    ... and 257 other files
                ... and 1816 other folders
            train_images/
                1.2.826.0.1.3680043.10014/
                    1.dcm (240.9 kB)
                    10.dcm (250.8 kB)
                    ... and 256 other files
                1.2.826.0.1.3680043.10058/
                    1.dcm (525.0 kB)
                    10.dcm (525.0 kB)
                    ... and 574 other files
                ... and 201 other folders
        input/
            description.md (276 lines)
            sample_submission.csv (14537 lines)
            sample_submission.csv.zip (50.3 kB)
            segmentations.zip (3.0 MB)
            test.csv (14537 lines)
            test.csv.zip (94.5 kB)
            test.zip (160 Bytes)
            test_images.zip (181.9 GB)
            train.csv (203 lines)
            train.csv.zip (1.2 kB)
            train.zip (162 Bytes)
            train_bounding_boxes.csv (691 lines)
            train_bounding_boxes.csv.zip (11.5 kB)
            train_images.zip (20.3 GB)
            rsna-2022-cervical-spine-fracture-detection/
                description.md (276 lines)
                sample_submission.csv (14537 lines)
                ... and 12 other files
                rsna-2022-cervical-spine-fracture-detection/
                segmentations/
                    1.2.826.0.1.3680043.12292.nii (89.4 MB)
                    1.2.826.0.1.3680043.24617.nii (307.2 MB)
                    ... and 7 other files
                test_images/
                    1.2.826.0.1.3680043.10001/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 266 other files
                    1.2.826.0.1.3680043.10005/
                        1.dcm (525.1 kB)
                        10.dcm (525.1 kB)
                        ... and 257 other files
                    ... and 1816 other folders
                train_images/
                    1.2.826.0.1.3680043.10014/
                        1.dcm (240.9 kB)
                        10.dcm (250.8 kB)
                        ... and 256 other files
                    1.2.826.0.1.3680043.10058/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 574 other files
                    ... and 201 other folders
            segmentations/
                1.2.826.0.1.3680043.12292.nii (89.4 MB)
                1.2.826.0.1.3680043.24617.nii (307.2 MB)
                ... and 7 other files
            test_images/
                1.2.826.0.1.3680043.10001/
                    1.dcm (525.0 kB)
                    10.dcm (525.0 kB)
                    ... and 266 other files
                1.2.826.0.1.3680043.10005/
                    1.dcm (525.1 kB)
                    10.dcm (525.1 kB)
                    ... and 257 other files
                ... and 1816 other folders
            train_images/
                1.2.826.0.1.3680043.10014/
                    1.dcm (240.9 kB)
                    10.dcm (250.8 kB)
                    ... and 256 other files
                1.2.826.0.1.3680043.10058/
                    1.dcm (525.0 kB)
                    10.dcm (525.0 kB)
                    ... and 574 other files
                ... and 201 other folders
        working/
            rsna-2022-cervical-spine-fracture-detection/
                description.md (276 lines)
                sample_submission.csv (14537 lines)
                ... and 12 other files
                rsna-2022-cervical-spine-fracture-detection/
                segmentations/
                    1.2.826.0.1.3680043.12292.nii (89.4 MB)
                    1.2.826.0.1.3680043.24617.nii (307.2 MB)
                    ... and 7 other files
                test_images/
                    1.2.826.0.1.3680043.10001/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 266 other files
                    1.2.826.0.1.3680043.10005/
                        1.dcm (525.1 kB)
                        10.dcm (525.1 kB)
                        ... and 257 other files
                    ... and 1816 other folders
                train_images/
                    1.2.826.0.1.3680043.10014/
                        1.dcm (240.9 kB)
                        10.dcm (250.8 kB)
                        ... and 256 other files
                    1.2.826.0.1.3680043.10058/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 574 other files
                    ... and 201 other folders
```

-> data/rsna-2022-cervical-spine-fracture-detection/sample_submission.csv has 14536 rows and 2 columns.
The columns are: row_id, fractured

-> data/rsna-2022-cervical-spine-fracture-detection/test.csv has 14536 rows and 3 columns.
The columns are: StudyInstanceUID, prediction_type, row_id

-> data/rsna-2022-cervical-spine-fracture-detection/train.csv has 202 rows and 9 columns.
The columns are: StudyInstanceUID, patient_overall, C1, C2, C3, C4, C5, C6, C7

-> data/rsna-2022-cervical-spine-fracture-detection/train_bounding_boxes.csv has 690 rows and 6 columns.
The columns are: StudyInstanceUID, x, y, width, height, slice_number

-> data/sample_submission.csv has 14536 rows and 2 columns.
The columns are: row_id, fractured

-> data/test.csv has 14536 rows and 3 columns.
The columns are: StudyInstanceUID, prediction_type, row_id

-> data/train.csv has 202 rows and 9 columns.
The columns are: StudyInstanceUID, patient_overall, C1, C2, C3, C4, C5, C6, C7

-> data/train_bounding_boxes.csv has 690 rows and 6 columns.
The columns are: StudyInstanceUID, x, y, width, height, slice_number

-> (stopped after 10 files for performance)

# 5. Target score

0.6167917389474165

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, sys, subprocess, textwrap


def _pip_install_quiet(pkgs):
    cmd = [sys.executable, "-m", "pip", "install", "-q"] + pkgs
    subprocess.check_call(cmd)


try:
    import pydicom  # noqa
    import pylibjpeg  # noqa
    import gdcm  # noqa

    _have_decode_deps = True
except Exception:
    _have_decode_deps = False

if not _have_decode_deps:
    _pip_install_quiet(
        [
            "pydicom>=3.0.0",
            "pylibjpeg>=2.0.0",
            "pylibjpeg-libjpeg>=2.1.0",
            "python-gdcm>=3.0.10",
        ]
    )



## === cell 1
import numpy as np
import pandas as pd
import pydicom

import cv2
import os
from tqdm import tqdm
import glob

from albumentations import Compose, CenterCrop
import torch.nn as nn
from torchvision.models.convnext import convnext_base, ConvNeXt_Base_Weights

from torch.utils.data import Dataset, DataLoader
import torch
import warnings

warnings.filterwarnings("ignore")



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/numpy/_core/__init__.py in <module>
     22 
     23 try:
---> 24     from . import multiarray
     25 except ImportError as exc:
     26     import sys

/usr/local/lib/python3.11/dist-packages/numpy/_core/multiarray.py in <module>
      9 import functools
     10 
---> 11 from . import _multiarray_umath, overrides
     12 from ._multiarray_umath import *  # noqa: F403
     13 

AttributeError: module 'numpy._globals' has no attribute '_signature_descriptor'

## === cell 2
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/334941773.py in <cell line: 0>()
      1 # Repro / device
----> 2 torch.backends.cudnn.benchmark = True
      3 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      4 

NameError: name 'torch' is not defined

## === cell 3
config = {
    "seq_len": 192,
    "feature_size": 1024,
    "lstm_size": 128,
    "target_size": 512,
    "crop_size": 320,
    "num_classes": 7,
    "batch_size_image_level": 32,
    "batch_size_patient_level": 8,
}




## === cell 4
def load_df_test():
    df_test = pd.read_csv(
        f"../input/rsna-2022-cervical-spine-fracture-detection/test.csv"
    )

    if df_test.iloc[0].row_id == "1.2.826.0.1.3680043.10197_C1":
        df_test = pd.DataFrame(
            {
                "row_id": [
                    "1.2.826.0.1.3680043.22327_C1",
                    "1.2.826.0.1.3680043.25399_C1",
                    "1.2.826.0.1.3680043.5876_C1",
                ],
                "StudyInstanceUID": [
                    "1.2.826.0.1.3680043.22327",
                    "1.2.826.0.1.3680043.25399",
                    "1.2.826.0.1.3680043.5876",
                ],
                "prediction_type": ["C1", "C1", "patient_overall"],
            }
        )
    return df_test




## === cell 5
test_df = load_df_test()
TEST_PATH = "../input/rsna-2022-cervical-spine-fracture-detection/test_images"
study_id_list = list(test_df.StudyInstanceUID.unique())
len(study_id_list), study_id_list[:3]



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4025366045.py in <cell line: 0>()
----> 1 test_df = load_df_test()
      2 TEST_PATH = "../input/rsna-2022-cervical-spine-fracture-detection/test_images"
      3 study_id_list = list(test_df.StudyInstanceUID.unique())
      4 len(study_id_list), study_id_list[:3]
      5 

/tmp/ipykernel_55/3264418297.py in load_df_test()
      1 def load_df_test():
----> 2     df_test = pd.read_csv(
      3         f"../input/rsna-2022-cervical-spine-fracture-detection/test.csv"
      4     )
      5 

NameError: name 'pd' is not defined

## === cell 6
dicom_paths_by_uid = {}
selected_index_dict = {}

for uid in study_id_list:
    paths = glob.glob(os.path.join(TEST_PATH, uid, "*.dcm"))
    paths = sorted(paths, key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    dicom_paths_by_uid[uid] = paths

    n = len(paths)
    if n == 0:
        selected_index_dict[uid] = []
        continue

    middle = n // 2
    num_left = num_right = max(1, int(0.15 * n))
    left_start = max(1, middle - num_left)  # keep room for -1 neighbor
    left_end = middle  # exclusive
    right_start = middle + 1
    right_end = min(n - 2, middle + num_right)  # keep room for +1 neighbor

    idxs = list(range(left_start, left_end)) + list(range(right_start, right_end + 1))
    idxs = [i for i in sorted(set(idxs)) if 1 <= i <= n - 2]
    selected_index_dict[uid] = idxs

selected_index_dict[study_id_list[0]][:10], len(selected_index_dict[study_id_list[0]])



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4182496482.py in <cell line: 0>()
      4 selected_index_dict = {}
      5 
----> 6 for uid in study_id_list:
      7     paths = glob.glob(os.path.join(TEST_PATH, uid, "*.dcm"))
      8     paths = sorted(paths, key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))

NameError: name 'study_id_list' is not defined

## === cell 7
print("Example selected indices:", selected_index_dict[study_id_list[0]][:20])



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3289023470.py in <cell line: 0>()
----> 1 print("Example selected indices:", selected_index_dict[study_id_list[0]][:20])
      2 

NameError: name 'study_id_list' is not defined

## === cell 8
try:
    pydicom.config.image_handlers = []  # let new backend manage plugins if present
except Exception:
    pass




## === cell 9
def window(data, WL=400, WW=1800):
    try:
        if hasattr(data, "pixel_array_options"):
            data.pixel_array_options(use_v2_backend=True)
    except Exception:
        pass

    slope = float(getattr(data, "RescaleSlope", 1.0))
    intercept = float(getattr(data, "RescaleIntercept", 0.0))

    img = data.pixel_array.astype(np.float32)
    img = img * slope + intercept

    upper, lower = WL + WW // 2, WL - WW // 2
    X = np.clip(img, lower, upper)
    X = X - np.min(X)
    mx = np.max(X)
    if mx > 0:
        X = X / mx
    X = (X * 255.0).astype("uint8")
    return X


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))




## === cell 10
mean = np.array([0.456, 0.456, 0.456], dtype=np.float32)
std = np.array([0.224, 0.224, 0.224], dtype=np.float32)




## === cell 11
class CSFImageDataset(Dataset):
    def __init__(self, uid, selected_indices, target_size, crop_size):
        self.uid = uid
        self.selected_indices = selected_indices
        self.target_size = target_size
        self.crop_size = crop_size
        self.paths = dicom_paths_by_uid[uid]
        self.inference_transform = Compose([CenterCrop(self.crop_size, self.crop_size)])

    def __len__(self):
        return len(self.selected_indices)

    def _safe_read_window(self, path):
        try:
            ds = pydicom.dcmread(path)
            return window(ds)
        except Exception:
            return np.zeros((512, 512), dtype=np.uint8)

    def __getitem__(self, index):
        i = self.selected_indices[index]
        i0 = max(0, i - 1)
        i1 = i
        i2 = min(len(self.paths) - 1, i + 1)

        imgs = [
            self._safe_read_window(self.paths[i0]),
            self._safe_read_window(self.paths[i1]),
            self._safe_read_window(self.paths[i2]),
        ]

        stacked_img = np.stack(imgs, axis=-1)
        stacked_img = cv2.resize(
            stacked_img,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_AREA,
        )

        out = self.inference_transform(image=stacked_img)
        X = out["image"].astype(np.float32) / 255.0
        X = (X - mean) / std
        X = img2tensor(X)
        return X




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2885908797.py in <cell line: 0>()
----> 1 class CSFImageDataset(Dataset):
      2     def __init__(self, uid, selected_indices, target_size, crop_size):
      3         self.uid = uid
      4         self.selected_indices = selected_indices
      5         self.target_size = target_size

NameError: name 'Dataset' is not defined

## === cell 12
class CSFInstanceDataset(Dataset):
    def __init__(self, feature_array_dict, study_id_list, seq_len):
        self.feature_array_dict = feature_array_dict
        self.study_id_list = study_id_list
        self.seq_len = seq_len

    def __len__(self):
        return len(self.study_id_list)

    def __getitem__(self, index):
        uid = self.study_id_list[index]
        feature_array = self.feature_array_dict[uid]
        if len(feature_array) > self.seq_len:
            x = cv2.resize(
                feature_array,
                (feature_array.shape[1], self.seq_len),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            x = np.pad(
                feature_array,
                pad_width=[(0, self.seq_len - feature_array.shape[0]), (0, 0)],
                constant_values=0,
            )
        X = torch.tensor(x, dtype=torch.float32)
        return X, uid




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/291513014.py in <cell line: 0>()
----> 1 class CSFInstanceDataset(Dataset):
      2     def __init__(self, feature_array_dict, study_id_list, seq_len):
      3         self.feature_array_dict = feature_array_dict
      4         self.study_id_list = study_id_list
      5         self.seq_len = seq_len

NameError: name 'Dataset' is not defined

## === cell 13
class ConvNextCNN_B_Feature(nn.Module):
    def __init__(self):
        super(ConvNextCNN_B_Feature, self).__init__()
        m = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        in_features = m.classifier[-1].in_features
        self.features = m.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(in_features=in_features, out_features=7)

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = self.drop(out)
        feature = out.view(x.size(0), -1)
        out = self.fc(feature)
        return feature, out




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2368340535.py in <cell line: 0>()
----> 1 class ConvNextCNN_B_Feature(nn.Module):
      2     def __init__(self):
      3         super(ConvNextCNN_B_Feature, self).__init__()
      4         # Fixes: avoid missing local weight files by using torchvision pretrained weights.
      5         m = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)

NameError: name 'nn' is not defined

## === cell 14
class CSFNet(nn.Module):
    def __init__(self, input_len, lstm_size):
        super().__init__()
        self.lstm1 = nn.GRU(input_len, lstm_size, bidirectional=True, batch_first=True)
        self.last_linear = nn.Linear(lstm_size * 2, 1)

    def forward(self, x):
        h_lstm1, _ = self.lstm1(x)
        max_pool, _ = torch.max(h_lstm1, 1)
        logits = self.last_linear(max_pool)
        return logits




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/38548912.py in <cell line: 0>()
----> 1 class CSFNet(nn.Module):
      2     def __init__(self, input_len, lstm_size):
      3         super().__init__()
      4         self.lstm1 = nn.GRU(input_len, lstm_size, bidirectional=True, batch_first=True)
      5         self.last_linear = nn.Linear(lstm_size * 2, 1)

NameError: name 'nn' is not defined

## === cell 15
lv1_model = ConvNextCNN_B_Feature().to(device)
lv1_model.eval()

lv2_model = CSFNet(input_len=config["feature_size"], lstm_size=config["lstm_size"]).to(
    device
)
lv2_model.eval()

lv1_w_path = "../input/cnn-lstm-oct-21/run_3/run_3/model_3.pth"
lv2_w_path = "../input/cnn-lstm-oct-21/run_4/run_4/model_lstm_3.pth"

if os.path.exists(lv1_w_path):
    sd = torch.load(lv1_w_path, map_location="cpu")
    lv1_model.load_state_dict(sd, strict=True)

if os.path.exists(lv2_w_path):
    sd = torch.load(lv2_w_path, map_location="cpu")
    lv2_model.load_state_dict(sd, strict=True)

print("lv1 weights loaded:", os.path.exists(lv1_w_path))
print("lv2 weights loaded:", os.path.exists(lv2_w_path))



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1886138579.py in <cell line: 0>()
      1 # Load models. If external competition weights are unavailable, fall back gracefully so the notebook runs end-to-end.
----> 2 lv1_model = ConvNextCNN_B_Feature().to(device)
      3 lv1_model.eval()
      4 
      5 lv2_model = CSFNet(input_len=config["feature_size"], lstm_size=config["lstm_size"]).to(

NameError: name 'ConvNextCNN_B_Feature' is not defined

## === cell 16
submission_dict = {"row_id": [], "fractured": []}
feature_array_dict = {}

for uid in study_id_list:
    selected_indices = selected_index_dict[uid]
    if len(selected_indices) == 0:
        for c in ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]:
            submission_dict["row_id"].append(f"{uid}_{c}")
            submission_dict["fractured"].append(0.0)
        feature_array_dict[uid] = np.zeros(
            (1, config["feature_size"]), dtype=np.float32
        )
        continue

    dataset = CSFImageDataset(
        uid=uid,
        selected_indices=selected_indices,
        target_size=config["target_size"],
        crop_size=config["crop_size"],
    )
    generator = DataLoader(
        dataset,
        batch_size=config["batch_size_image_level"],
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        num_workers=0,
    )

    preds_uid = []
    feature_array = np.zeros((len(dataset), config["feature_size"]), dtype=np.float32)

    for i, images in tqdm(
        enumerate(generator), total=len(generator), desc=f"Stage1 {uid[:18]}"
    ):
        with torch.no_grad():
            start = i * config["batch_size_image_level"]
            end = min(start + config["batch_size_image_level"], len(dataset))

            images = images.to(device, non_blocking=True)
            features, preds = lv1_model(images)

            feature_array[start:end] = features.detach().cpu().numpy()
            preds_uid.append(torch.sigmoid(preds).detach().cpu())

    feature_array_dict[uid] = feature_array

    mean_preds = (
        torch.mean(torch.cat(preds_uid, dim=0), dim=0).numpy().astype(np.float32)
    )

    for k, c in enumerate(["C1", "C2", "C3", "C4", "C5", "C6", "C7"]):
        submission_dict["row_id"].append(f"{uid}_{c}")
        submission_dict["fractured"].append(float(mean_preds[k]))



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1582769886.py in <cell line: 0>()
      4 feature_array_dict = {}
      5 
----> 6 for uid in study_id_list:
      7     selected_indices = selected_index_dict[uid]
      8     if len(selected_indices) == 0:

NameError: name 'study_id_list' is not defined

## === cell 17
dataset2 = CSFInstanceDataset(
    feature_array_dict=feature_array_dict,
    study_id_list=study_id_list,
    seq_len=config["seq_len"],
)
generator2 = DataLoader(
    dataset=dataset2,
    batch_size=config["batch_size_patient_level"],
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
    num_workers=0,
)

for features, list_uid in tqdm(
    generator2, total=len(generator2), desc="Stage2 patient_overall"
):
    with torch.no_grad():
        features = features.to(device, non_blocking=True)
        preds = torch.sigmoid(lv2_model(features)).detach().cpu().numpy().squeeze()

    preds = np.atleast_1d(preds)
    for j in range(len(list_uid)):
        submission_dict["row_id"].append(f"{list_uid[j]}_patient_overall")
        submission_dict["fractured"].append(float(preds[j]))



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2456887156.py in <cell line: 0>()
      1 # Stage 2: patient_overall inference from the extracted feature sequences.
----> 2 dataset2 = CSFInstanceDataset(
      3     feature_array_dict=feature_array_dict,
      4     study_id_list=study_id_list,
      5     seq_len=config["seq_len"],

NameError: name 'CSFInstanceDataset' is not defined

## === cell 18
sub_df = pd.DataFrame.from_dict(submission_dict)

sub_df = sub_df.drop_duplicates(subset=["row_id"], keep="last")

required = test_df[["row_id"]].copy()
sub_df = required.merge(sub_df, on="row_id", how="left")

sub_df["fractured"] = sub_df["fractured"].fillna(0.0).astype(np.float32)
sub_df["fractured"] = np.clip(sub_df["fractured"].values, 1e-6, 1 - 1e-6)

sub_df.head(), sub_df.shape



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4266477307.py in <cell line: 0>()
      1 # Fixes: Ensure submission matches exactly the required row_id set and order from test.csv (prevents missing/extra rows).
----> 2 sub_df = pd.DataFrame.from_dict(submission_dict)
      3 
      4 # There may be duplicates if code is re-run; keep last occurrence.
      5 sub_df = sub_df.drop_duplicates(subset=["row_id"], keep="last")

NameError: name 'pd' is not defined

## === cell 19
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with rows:", len(sub_df))
print(sub_df.tail())

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1319411574.py in <cell line: 0>()
----> 1 sub_df.to_csv("submission.csv", index=False)
      2 print("Wrote submission.csv with rows:", len(sub_df))
      3 print(sub_df.tail())

NameError: name 'sub_df' is not defined
