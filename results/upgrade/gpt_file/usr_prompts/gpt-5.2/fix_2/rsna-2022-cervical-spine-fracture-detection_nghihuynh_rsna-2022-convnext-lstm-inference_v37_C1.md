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

0.5975068471121325

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, sys, glob, time, pickle, warnings, subprocess

warnings.filterwarnings("ignore")


def _ensure_dicom_decoders():
    try:
        import pydicom  # noqa: F401

        import pylibjpeg  # noqa: F401
        import pylibjpeg_libjpeg  # noqa: F401
        import gdcm  # noqa: F401

        return
    except Exception:
        pass

    pkgs = ["pylibjpeg>=2.0", "pylibjpeg-libjpeg>=2.1", "python-gdcm>=3.0.10"]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + pkgs)


_ensure_dicom_decoders()



## === cell 1
import numpy as np
import pandas as pd
import pydicom

import cv2
from tqdm import tqdm
from albumentations import Compose, CenterCrop
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.convnext import convnext_base

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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
PASS = True



## === cell 3
config = {
    "seq_len": 150,
    "feature_size": 1024,
    "lstm_size": 128,
    "target_size": 512,
    "crop_size": 384,
    "num_classes": 7,
    "batch_size_image_level": 32,
    "batch_size_patient_level": 8,
}



## === cell 4
DATA_ROOT = "../input/rsna-2022-cervical-spine-fracture-detection"


def load_df_test():
    df_test = pd.read_csv(f"{DATA_ROOT}/test.csv")
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
TEST_PATH = f"{DATA_ROOT}/test_images"

study_id_list = list(test_df.StudyInstanceUID.unique())
study_id_list[:5], len(study_id_list)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3898007003.py in <cell line: 0>()
----> 1 test_df = load_df_test()
      2 TEST_PATH = f"{DATA_ROOT}/test_images"
      3 
      4 study_id_list = list(test_df.StudyInstanceUID.unique())
      5 study_id_list[:5], len(study_id_list)

/tmp/ipykernel_55/3296361411.py in load_df_test()
      3 
      4 def load_df_test():
----> 5     df_test = pd.read_csv(f"{DATA_ROOT}/test.csv")
      6     # Keep the original safety shim for the "first few rows only" download situation
      7     if df_test.iloc[0].row_id == "1.2.826.0.1.3680043.10197_C1":

NameError: name 'pd' is not defined

## === cell 6
selected_image_dict = {}
dicom_index_dict = {}

for uid in study_id_list:
    dicom_files = glob.glob(os.path.join(TEST_PATH, uid, "*.dcm"))
    idxs = []
    for fp in dicom_files:
        base = os.path.splitext(os.path.basename(fp))[0]
        try:
            idxs.append(int(base))
        except Exception:
            continue
    idxs = sorted(idxs)
    dicom_index_dict[uid] = idxs

    if len(idxs) < 3:
        selected_image_dict[uid] = idxs
        continue

    mid_pos = len(idxs) // 2
    n = max(1, int(0.15 * len(idxs)))
    left = idxs[max(0, mid_pos - n) : mid_pos]
    right = idxs[mid_pos + 1 : min(len(idxs), mid_pos + 1 + n)]
    chosen = left + right
    selected_image_dict[uid] = chosen if len(chosen) > 0 else [idxs[mid_pos]]



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1625059106.py in <cell line: 0>()
      3 dicom_index_dict = {}
      4 
----> 5 for uid in study_id_list:
      6     dicom_files = glob.glob(os.path.join(TEST_PATH, uid, "*.dcm"))
      7     # Extract integer slice numbers from filenames

NameError: name 'study_id_list' is not defined

## === cell 7
if len(study_id_list) > 0:
    print("UID:", study_id_list[0])
    print("Chosen slices (sample):", selected_image_dict[study_id_list[0]][:10])



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3215506674.py in <cell line: 0>()
      1 # Show a sample chosen indices
----> 2 if len(study_id_list) > 0:
      3     print("UID:", study_id_list[0])
      4     print("Chosen slices (sample):", selected_image_dict[study_id_list[0]][:10])
      5 

NameError: name 'study_id_list' is not defined

## === cell 8
PASS = True




## === cell 9
def window(data, WL=400, WW=1800):
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
PASS = True



## === cell 11
mean = np.array([0.456, 0.456, 0.456], dtype=np.float32)
std = np.array([0.224, 0.224, 0.224], dtype=np.float32)


class CSFImageDataset(Dataset):
    def __init__(self, uid, image_list, target_size, crop_size):
        self.uid = uid
        self.image_list = image_list
        self.target_size = target_size
        self.crop_size = crop_size

    def __len__(self):
        return len(self.image_list)

    def _safe_read(self, slice_idx: int):
        path = os.path.join(TEST_PATH, self.uid, f"{slice_idx}.dcm")
        return pydicom.dcmread(path)

    def __getitem__(self, index):
        idxs = dicom_index_dict[self.uid]
        if len(idxs) == 0:
            raise RuntimeError(f"No DICOM slices found for uid={self.uid}")

        center = int(self.image_list[index])
        if center not in set(idxs):
            center = min(idxs, key=lambda x: abs(x - center))

        min_idx, max_idx = idxs[0], idxs[-1]
        s0 = max(min_idx, center - 1)
        s1 = center
        s2 = min(max_idx, center + 1)

        data_list = [self._safe_read(s0), self._safe_read(s1), self._safe_read(s2)]
        imgs = [window(data) for data in data_list]

        stacked_img = np.stack(imgs, axis=-1)
        stacked_img = cv2.resize(
            stacked_img,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_LINEAR,
        )

        inference_transform = Compose([CenterCrop(self.crop_size, self.crop_size)])
        stacked_img = inference_transform(image=stacked_img)["image"]

        X = img2tensor((stacked_img / 255.0 - mean) / std)
        return X




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1479833248.py in <cell line: 0>()
      3 
      4 
----> 5 class CSFImageDataset(Dataset):
      6     def __init__(self, uid, image_list, target_size, crop_size):
      7         self.uid = uid

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
PASS = True




## === cell 14
class ConvNextCNN_B_Feature(nn.Module):
    def __init__(self):
        super(ConvNextCNN_B_Feature, self).__init__()
        m = convnext_base()
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




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/846480535.py in <cell line: 0>()
----> 1 class ConvNextCNN_B_Feature(nn.Module):
      2     def __init__(self):
      3         super(ConvNextCNN_B_Feature, self).__init__()
      4         m = convnext_base()
      5         in_features = m.classifier[-1].in_features

NameError: name 'nn' is not defined

## === cell 15
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




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/38548912.py in <cell line: 0>()
----> 1 class CSFNet(nn.Module):
      2     def __init__(self, input_len, lstm_size):
      3         super().__init__()
      4         self.lstm1 = nn.GRU(input_len, lstm_size, bidirectional=True, batch_first=True)
      5         self.last_linear = nn.Linear(lstm_size * 2, 1)

NameError: name 'nn' is not defined

## === cell 16
def _find_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


orig_lv1 = "../input/cnn-lstm-oct-25/run_3/run_3/model_3.pth"
orig_lv2 = "../input/cnn-lstm-oct-25/run_3/run_3/model_lstm_3.pth"

lv1_candidates = [
    orig_lv1,
    f"{DATA_ROOT}/cnn_lstm_oct_25/run_3/run_3/model_3.pth",
    f"{DATA_ROOT}/cnn-lstm-oct-25/run_3/run_3/model_3.pth",
    "../input/cnn-lstm/model_3.pth",
    "../input/cnn-lstm/model.pth",
]
lv2_candidates = [
    orig_lv2,
    f"{DATA_ROOT}/cnn_lstm_oct_25/run_3/run_3/model_lstm_3.pth",
    f"{DATA_ROOT}/cnn-lstm-oct-25/run_3/run_3/model_lstm_3.pth",
    "../input/cnn-lstm/model_lstm_3.pth",
    "../input/cnn-lstm/model_lstm.pth",
]

lv1_path = _find_first_existing(lv1_candidates)
lv2_path = _find_first_existing(lv2_candidates)

if lv1_path is None or lv2_path is None:
    raise FileNotFoundError(
        "Could not find required model weight files. "
        f"Looked for lv1 in: {lv1_candidates} and lv2 in: {lv2_candidates}"
    )

lv1_model = ConvNextCNN_B_Feature()
lv1_model.load_state_dict(torch.load(lv1_path, map_location="cpu"))
lv1_model = lv1_model.to(device)
lv1_model.eval()

lv2_model = CSFNet(input_len=config["feature_size"], lstm_size=config["lstm_size"])
lv2_model.load_state_dict(torch.load(lv2_path, map_location="cpu"))
lv2_model = lv2_model.to(device)
lv2_model.eval()

print("Loaded lv1 weights:", lv1_path)
print("Loaded lv2 weights:", lv2_path)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/166099695.py in <cell line: 0>()
     33     # If no external weights exist, we cannot reproduce the intended model performance.
     34     # Fail fast with a clear message rather than producing random predictions.
---> 35     raise FileNotFoundError(
     36         "Could not find required model weight files. "
     37         f"Looked for lv1 in: {lv1_candidates} and lv2 in: {lv2_candidates}"

FileNotFoundError: Could not find required model weight files. Looked for lv1 in: ['../input/cnn-lstm-oct-25/run_3/run_3/model_3.pth', '../input/rsna-2022-cervical-spine-fracture-detection/cnn_lstm_oct_25/run_3/run_3/model_3.pth', '../input/rsna-2022-cervical-spine-fracture-detection/cnn-lstm-oct-25/run_3/run_3/model_3.pth', '../input/cnn-lstm/model_3.pth', '../input/cnn-lstm/model.pth'] and lv2 in: ['../input/cnn-lstm-oct-25/run_3/run_3/model_lstm_3.pth', '../input/rsna-2022-cervical-spine-fracture-detection/cnn_lstm_oct_25/run_3/run_3/model_lstm_3.pth', '../input/rsna-2022-cervical-spine-fracture-detection/cnn-lstm-oct-25/run_3/run_3/model_lstm_3.pth', '../input/cnn-lstm/model_lstm_3.pth', '../input/cnn-lstm/model_lstm.pth']

## === cell 17
PASS = True



## === cell 18
PASS = True



## === cell 19
submission_dict = {"row_id": [], "fractured": []}
feature_array_dict = {}

for uid in study_id_list:
    image_list = selected_image_dict[uid]
    if len(image_list) == 0:
        idxs = dicom_index_dict[uid]
        if len(idxs) == 0:
            continue
        image_list = [idxs[len(idxs) // 2]]

    dataset = CSFImageDataset(
        uid=uid,
        image_list=image_list,
        target_size=config["target_size"],
        crop_size=config["crop_size"],
    )
    generator = DataLoader(
        dataset,
        batch_size=config["batch_size_image_level"],
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    preds_uid = []
    feature_array = np.zeros((len(dataset), config["feature_size"]), dtype=np.float32)

    for i, images in tqdm(
        enumerate(generator),
        total=len(generator),
        desc=f"UID {uid[-6:]} lv1",
        leave=False,
    ):
        with torch.no_grad():
            start = i * config["batch_size_image_level"]
            end = min(start + config["batch_size_image_level"], len(generator.dataset))

            images = images.to(device, non_blocking=True)
            features, preds = lv1_model(images)

            feature_array[start:end] = np.squeeze(features.detach().cpu().numpy())
            preds = preds.sigmoid().detach().cpu()
            preds_uid.append(preds)

    feature_array_dict[uid] = feature_array

    mean_preds = torch.mean(torch.cat(preds_uid, dim=0), dim=0).numpy()

    submission_dict["row_id"].append(f"{uid}_C1")
    submission_dict["fractured"].append(float(mean_preds[0]))
    submission_dict["row_id"].append(f"{uid}_C2")
    submission_dict["fractured"].append(float(mean_preds[1]))
    submission_dict["row_id"].append(f"{uid}_C3")
    submission_dict["fractured"].append(float(mean_preds[2]))
    submission_dict["row_id"].append(f"{uid}_C4")
    submission_dict["fractured"].append(float(mean_preds[3]))
    submission_dict["row_id"].append(f"{uid}_C5")
    submission_dict["fractured"].append(float(mean_preds[4]))
    submission_dict["row_id"].append(f"{uid}_C6")
    submission_dict["fractured"].append(float(mean_preds[5]))
    submission_dict["row_id"].append(f"{uid}_C7")
    submission_dict["fractured"].append(float(mean_preds[6]))



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3002783731.py in <cell line: 0>()
      2 feature_array_dict = {}
      3 
----> 4 for uid in study_id_list:
      5     image_list = selected_image_dict[uid]
      6     if len(image_list) == 0:

NameError: name 'study_id_list' is not defined

## === cell 20
PASS = True



## === cell 21
dataset = CSFInstanceDataset(
    feature_array_dict=feature_array_dict,
    study_id_list=study_id_list,
    seq_len=config["seq_len"],
)
generator = DataLoader(
    dataset=dataset,
    batch_size=config["batch_size_patient_level"],
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
)

for features, list_uid in tqdm(
    generator, total=len(generator), desc="lv2", leave=False
):
    with torch.no_grad():
        features = features.to(device, non_blocking=True)
        preds = lv2_model(features)
        preds = np.squeeze(preds.sigmoid().detach().cpu().numpy())

    if np.isscalar(preds):
        preds = np.array([float(preds)], dtype=np.float32)

    for j in range(len(list_uid)):
        submission_dict["row_id"].append(f"{list_uid[j]}_patient_overall")
        submission_dict["fractured"].append(float(preds[j]))



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2132439381.py in <cell line: 0>()
----> 1 dataset = CSFInstanceDataset(
      2     feature_array_dict=feature_array_dict,
      3     study_id_list=study_id_list,
      4     seq_len=config["seq_len"],
      5 )

NameError: name 'CSFInstanceDataset' is not defined

## === cell 22
PASS = True



## === cell 23
sub_df = pd.DataFrame.from_dict(submission_dict)

sample_sub = pd.read_csv(f"{DATA_ROOT}/sample_submission.csv")
sub_df = sample_sub[["row_id"]].merge(sub_df, on="row_id", how="left")

if sub_df["fractured"].isna().any():
    sub_df["fractured"] = sub_df["fractured"].fillna(0.5)

eps = 1e-6
sub_df["fractured"] = sub_df["fractured"].clip(eps, 1 - eps)

sub_df.head(), sub_df.shape



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3558854023.py in <cell line: 0>()
----> 1 sub_df = pd.DataFrame.from_dict(submission_dict)
      2 
      3 # Ensure submission rows exactly match the required test row_ids (ordering + completeness)
      4 sample_sub = pd.read_csv(f"{DATA_ROOT}/sample_submission.csv")
      5 sub_df = sample_sub[["row_id"]].merge(sub_df, on="row_id", how="left")

NameError: name 'pd' is not defined

## === cell 24
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head(10).to_string(index=False))

## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2780848251.py in <cell line: 0>()
----> 1 sub_df.to_csv("submission.csv", index=False)
      2 print("Wrote submission.csv with shape:", sub_df.shape)
      3 print(sub_df.head(10).to_string(index=False))

NameError: name 'sub_df' is not defined
