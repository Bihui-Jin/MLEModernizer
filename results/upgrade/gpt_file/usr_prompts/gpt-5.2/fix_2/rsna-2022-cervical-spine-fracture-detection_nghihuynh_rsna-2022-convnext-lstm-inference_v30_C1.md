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

3.11

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

0.7862279082718521

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import importlib
import os
import sys
import subprocess


def _pip_install(pkgs):
    cmd = [sys.executable, "-m", "pip", "install", "-q"] + list(pkgs)
    subprocess.check_call(cmd)


needs = []
try:
    import pydicom  # noqa: F401
except Exception:
    needs.append("pydicom")

try:
    import gdcm  # noqa: F401
except Exception:
    needs.append("gdcm")

try:
    import pylibjpeg  # noqa: F401
except Exception:
    needs.append("pylibjpeg")

try:
    import pylibjpeg_libjpeg  # noqa: F401
except Exception:
    needs.append("pylibjpeg-libjpeg")

if needs:
    _pip_install(needs)



## === cell 1
import numpy as np
import pandas as pd
import pydicom

import cv2
import os
from tqdm import tqdm
import glob
import torch.nn as nn
from albumentations import Compose, CenterCrop
import torch
import warnings

warnings.filterwarnings("ignore", category=UserWarning)



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
DATA_DIR = "/kaggle/input/rsna-2022-cervical-spine-fracture-detection"
TEST_CSV = f"{DATA_DIR}/test.csv"
SAMPLE_SUB = f"{DATA_DIR}/sample_submission.csv"
TEST_PATH = f"{DATA_DIR}/test_images"

assert os.path.exists(TEST_CSV), f"Missing {TEST_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.exists(TEST_PATH), f"Missing {TEST_PATH}"



## === cell 3
config = {
    "seq_len": 150,
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
    df_test = pd.read_csv(TEST_CSV)
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
study_id_list = list(test_df.StudyInstanceUID.unique())
study_id_list[:5], len(study_id_list)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/363080031.py in <cell line: 0>()
----> 1 test_df = load_df_test()
      2 study_id_list = list(test_df.StudyInstanceUID.unique())
      3 study_id_list[:5], len(study_id_list)
      4 

/tmp/ipykernel_55/4186048309.py in load_df_test()
      1 def load_df_test():
----> 2     df_test = pd.read_csv(TEST_CSV)
      3     # Keep the original fallback logic but adapted to our csv schema
      4     if df_test.iloc[0].row_id == "1.2.826.0.1.3680043.10197_C1":
      5         df_test = pd.DataFrame(

NameError: name 'pd' is not defined

## === cell 6
selected_image_dict = {}
uid_to_files = {}

for uid in study_id_list:
    dicom_files = sorted(
        glob.glob(os.path.join(TEST_PATH, uid, "*.dcm")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
    )
    uid_to_files[uid] = dicom_files
    n = len(dicom_files)
    if n == 0:
        selected_image_dict[uid] = []
        continue
    middle = n // 2
    k = int(0.15 * n)
    left = max(0, middle - k)
    right = min(n - 1, middle + k)
    idxs = list(range(max(1, left), min(n - 1, right + 1)))
    selected_image_dict[uid] = idxs



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2590852637.py in <cell line: 0>()
      4 uid_to_files = {}
      5 
----> 6 for uid in study_id_list:
      7     dicom_files = sorted(
      8         glob.glob(os.path.join(TEST_PATH, uid, "*.dcm")),

NameError: name 'study_id_list' is not defined

## === cell 7
first_uid = study_id_list[0]
len(uid_to_files[first_uid]), selected_image_dict[first_uid][:10]




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2907910780.py in <cell line: 0>()
      1 # Quick sanity check
----> 2 first_uid = study_id_list[0]
      3 len(uid_to_files[first_uid]), selected_image_dict[first_uid][:10]
      4 
      5 

NameError: name 'study_id_list' is not defined

## === cell 8
def window(data, WL=400, WW=1800):
    slope = float(getattr(data, "RescaleSlope", 1.0))
    intercept = float(getattr(data, "RescaleIntercept", 0.0))

    img = data.pixel_array.astype(np.float32)
    img = img * slope + intercept

    upper, lower = WL + WW // 2, WL - WW // 2
    X = np.clip(img, lower, upper)

    if getattr(data, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        X = upper - (X - lower)

    X = X - np.min(X)
    denom = np.max(X)
    if denom > 0:
        X = X / denom
    X = (X * 255.0).astype("uint8")
    return X


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))




## === cell 9
mean = np.array([0.456, 0.456, 0.456], dtype=np.float32)
std = np.array([0.224, 0.224, 0.224], dtype=np.float32)


class CSFImageDataset(torch.utils.data.Dataset):
    def __init__(self, uid, index_list, target_size, crop_size):
        self.uid = uid
        self.index_list = index_list
        self.target_size = target_size
        self.crop_size = crop_size

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, index):
        files = uid_to_files[self.uid]
        i = self.index_list[index]

        paths = [files[i - 1], files[i], files[i + 1]]
        data_list = [pydicom.dcmread(p) for p in paths]

        imgs = [window(d) for d in data_list]
        stacked_img = np.stack(imgs, axis=-1)

        stacked_img = cv2.resize(
            stacked_img,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_LINEAR,
        )

        inference_transform = Compose([CenterCrop(self.crop_size, self.crop_size)])
        stacked_img = inference_transform(image=stacked_img)["image"]

        X = (stacked_img.astype(np.float32) / 255.0 - mean) / std
        X = img2tensor(X)
        return X




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3865916806.py in <cell line: 0>()
      3 
      4 
----> 5 class CSFImageDataset(torch.utils.data.Dataset):
      6     def __init__(self, uid, index_list, target_size, crop_size):
      7         self.uid = uid

NameError: name 'torch' is not defined

## === cell 10
class CSFInstanceDataset(torch.utils.data.Dataset):
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




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2405543143.py in <cell line: 0>()
----> 1 class CSFInstanceDataset(torch.utils.data.Dataset):
      2     def __init__(self, feature_array_dict, study_id_list, seq_len):
      3         self.feature_array_dict = feature_array_dict
      4         self.study_id_list = study_id_list
      5         self.seq_len = seq_len

NameError: name 'torch' is not defined

## === cell 11
from torchvision.models.convnext import convnext_base


class ConvNextCNN_B_Feature(nn.Module):
    def __init__(self):
        super(ConvNextCNN_B_Feature, self).__init__()
        m = convnext_base(weights=None)
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




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/242666253.py in <cell line: 0>()
      2 
      3 
----> 4 class ConvNextCNN_B_Feature(nn.Module):
      5     def __init__(self):
      6         super(ConvNextCNN_B_Feature, self).__init__()

NameError: name 'nn' is not defined

## === cell 12
def find_first_file(patterns, root="/kaggle/input"):
    for pat in patterns:
        matches = glob.glob(os.path.join(root, "**", pat), recursive=True)
        if matches:
            return matches[0]
    return None


lv1_path = find_first_file(
    patterns=[
        "model_best.pth",
        "*model_best*.pth",
    ]
)
lv2_path = find_first_file(
    patterns=[
        "model_lstm_0.pth",
        "*model_lstm_0*.pth",
        "*lstm*0*.pth",
    ]
)

if lv1_path is None or lv2_path is None:
    raise FileNotFoundError(
        "Could not locate required model weights under /kaggle/input. "
        f"Found lv1_path={lv1_path}, lv2_path={lv2_path}"
    )

lv1_path, lv2_path



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1337058663.py in <cell line: 0>()
      8 
      9 
---> 10 lv1_path = find_first_file(
     11     patterns=[
     12         "model_best.pth",

/tmp/ipykernel_55/1337058663.py in find_first_file(patterns, root)
      2 def find_first_file(patterns, root="/kaggle/input"):
      3     for pat in patterns:
----> 4         matches = glob.glob(os.path.join(root, "**", pat), recursive=True)
      5         if matches:
      6             return matches[0]

NameError: name 'glob' is not defined

## === cell 13
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lv1_model = ConvNextCNN_B_Feature()
lv1_model.load_state_dict(torch.load(lv1_path, map_location="cpu"))
lv1_model = lv1_model.to(device)
lv1_model.eval()

lv2_model = CSFNet(input_len=config["feature_size"], lstm_size=config["lstm_size"])
lv2_model.load_state_dict(torch.load(lv2_path, map_location="cpu"))
lv2_model = lv2_model.to(device)
lv2_model.eval()



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3691705876.py in <cell line: 0>()
----> 1 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      2 
      3 lv1_model = ConvNextCNN_B_Feature()
      4 lv1_model.load_state_dict(torch.load(lv1_path, map_location="cpu"))
      5 lv1_model = lv1_model.to(device)

NameError: name 'torch' is not defined

## === cell 14
submission_dict = {"row_id": [], "fractured": []}
feature_array_dict = {}

for uid in tqdm(study_id_list, total=len(study_id_list)):
    index_list = selected_image_dict.get(uid, [])
    if len(index_list) == 0:
        feature_array_dict[uid] = np.zeros(
            (1, config["feature_size"]), dtype=np.float32
        )
        for k, col in enumerate(["C1", "C2", "C3", "C4", "C5", "C6", "C7"]):
            submission_dict["row_id"].append(f"{uid}_{col}")
            submission_dict["fractured"].append(0.5)
        continue

    dataset = CSFImageDataset(
        uid=uid,
        index_list=index_list,
        target_size=config["target_size"],
        crop_size=config["crop_size"],
    )
    generator = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["batch_size_image_level"],
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        num_workers=0,
    )

    preds_uid = []
    feature_array = np.zeros((len(dataset), config["feature_size"]), dtype=np.float32)

    for i, images in enumerate(generator):
        with torch.no_grad():
            start = i * config["batch_size_image_level"]
            end = min(start + images.shape[0], len(dataset))

            images = images.to(device, non_blocking=True)
            features, preds = lv1_model(images)

            feature_array[start:end] = features.detach().cpu().numpy()
            preds_uid.append(preds.sigmoid().detach().cpu())

    feature_array_dict[uid] = feature_array

    mean_preds = torch.mean(torch.cat(preds_uid, dim=0), dim=0).numpy()

    for k, col in enumerate(["C1", "C2", "C3", "C4", "C5", "C6", "C7"]):
        submission_dict["row_id"].append(f"{uid}_{col}")
        submission_dict["fractured"].append(float(mean_preds[k]))



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3374263403.py in <cell line: 0>()
      3 feature_array_dict = {}
      4 
----> 5 for uid in tqdm(study_id_list, total=len(study_id_list)):
      6     index_list = selected_image_dict.get(uid, [])
      7     if len(index_list) == 0:

NameError: name 'tqdm' is not defined

## === cell 15
dataset2 = CSFInstanceDataset(
    feature_array_dict=feature_array_dict,
    study_id_list=study_id_list,
    seq_len=config["seq_len"],
)
generator2 = torch.utils.data.DataLoader(
    dataset=dataset2,
    batch_size=config["batch_size_patient_level"],
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
    num_workers=0,
)

for features, list_uid in tqdm(generator2, total=len(generator2)):
    with torch.no_grad():
        features = features.to(device, non_blocking=True)
        preds = lv2_model(features)
        preds = preds.sigmoid().detach().cpu().numpy().reshape(-1)

    for j in range(len(list_uid)):
        submission_dict["row_id"].append(f"{list_uid[j]}_patient_overall")
        submission_dict["fractured"].append(float(preds[j]))



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2599654808.py in <cell line: 0>()
      1 # Stage 2: patient_overall from sequence model
----> 2 dataset2 = CSFInstanceDataset(
      3     feature_array_dict=feature_array_dict,
      4     study_id_list=study_id_list,
      5     seq_len=config["seq_len"],

NameError: name 'CSFInstanceDataset' is not defined

## === cell 16
sub_df = pd.DataFrame.from_dict(submission_dict)

sub_df["fractured"] = sub_df["fractured"].astype(np.float32).clip(1e-6, 1 - 1e-6)

sample = pd.read_csv(SAMPLE_SUB)
merged = sample[["row_id"]].merge(sub_df, on="row_id", how="left")

merged["fractured"] = (
    merged["fractured"].fillna(0.5).astype(np.float32).clip(1e-6, 1 - 1e-6)
)

merged.head(), merged.shape, merged["fractured"].isna().sum()



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3636648080.py in <cell line: 0>()
      1 # Fix 4: Ensure submission exactly matches sample_submission row order and has valid probabilities.
----> 2 sub_df = pd.DataFrame.from_dict(submission_dict)
      3 
      4 # Clip for log-loss stability
      5 sub_df["fractured"] = sub_df["fractured"].astype(np.float32).clip(1e-6, 1 - 1e-6)

NameError: name 'pd' is not defined

## === cell 17
merged.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", merged.shape)
print(merged.head(10).to_string(index=False))

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3135807230.py in <cell line: 0>()
----> 1 merged.to_csv("submission.csv", index=False)
      2 print("Wrote submission.csv with shape:", merged.shape)
      3 print(merged.head(10).to_string(index=False))

NameError: name 'merged' is not defined
