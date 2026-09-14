# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import sys
import glob
import time
import pickle
import math
import random
import numpy as np
import pandas as pd

import cv2
import pydicom

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision.models.convnext import convnext_base




## === cell 1
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



## === cell 2
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




## === cell 3
def _resolve_comp_path():
    candidates = [
        "../input/rsna-2022-cervical-spine-fracture-detection",
        "/kaggle/input/rsna-2022-cervical-spine-fracture-detection",
        "/kaggle/data/rsna-2022-cervical-spine-fracture-detection",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return "/kaggle/data/rsna-2022-cervical-spine-fracture-detection"


COMP_PATH = _resolve_comp_path()


def load_df_test():
    df_test = pd.read_csv(os.path.join(COMP_PATH, "test.csv"))

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




## === cell 4
test_df = load_df_test()
TEST_PATH = os.path.join(COMP_PATH, "test_images")
study_id_list = list(test_df.StudyInstanceUID.unique())

print("Num studies in test_df:", len(study_id_list))
print("TEST_PATH exists:", os.path.exists(TEST_PATH))



## === cell 5
selected_image_dict = {}
uid_to_slice_numbers = {}


def _list_slice_numbers_fast(uid_dir: str):
    out = []
    try:
        with os.scandir(uid_dir) as it:
            for e in it:
                if not e.is_file():
                    continue
                name = e.name
                if len(name) < 5 or name[-4:] != ".dcm":
                    continue
                stem = name[:-4]
                if stem.isdigit():
                    out.append(int(stem))
    except FileNotFoundError:
        return []
    return out


for uid in study_id_list:
    uid_dir = os.path.join(TEST_PATH, uid)
    slice_numbers = _list_slice_numbers_fast(uid_dir)
    if len(slice_numbers) == 0:
        uid_to_slice_numbers[uid] = []
        selected_image_dict[uid] = []
        continue

    slice_numbers.sort()
    uid_to_slice_numbers[uid] = slice_numbers

    mid_idx = len(slice_numbers) // 2
    num_left = num_right = int(0.15 * len(slice_numbers))

    left_idx_start = max(0, mid_idx - num_left)
    left_idx_end = mid_idx
    right_idx_start = min(len(slice_numbers), mid_idx + 1)
    right_idx_end = min(len(slice_numbers), mid_idx + 1 + num_right)

    selected = (
        slice_numbers[left_idx_start:left_idx_end]
        + slice_numbers[right_idx_start:right_idx_end]
    )
    if len(selected) == 0:
        selected = [slice_numbers[mid_idx]]
    selected_image_dict[uid] = selected

if len(study_id_list) > 0:
    print("Example selected slices:", selected_image_dict[study_id_list[0]][:10])



## === cell 6
import bisect


def get_triplet_slices(uid: str, center_slice: int):
    slices = uid_to_slice_numbers.get(uid, [])
    if len(slices) == 0:
        return None

    pos = bisect.bisect_left(slices, center_slice)
    if pos == len(slices):
        pos = len(slices) - 1
    if slices[pos] != center_slice:
        if pos > 0 and (
            pos == len(slices)
            or abs(slices[pos - 1] - center_slice) <= abs(slices[pos] - center_slice)
        ):
            pos = pos - 1

    prev_idx = max(0, pos - 1)
    next_idx = min(len(slices) - 1, pos + 1)
    return slices[prev_idx], slices[pos], slices[next_idx]




## === cell 7
_DCM_TAGS_FAST = ["RescaleSlope", "RescaleIntercept", "Rows", "Columns", "PixelData"]


def read_dicom_pixel_array(dcm_path: str):
    try:
        ds = pydicom.dcmread(
            dcm_path,
            stop_before_pixels=False,
            force=True,
            specific_tags=_DCM_TAGS_FAST,
        )
    except Exception:
        return None, np.zeros((512, 512), dtype=np.float32)

    try:
        arr = ds.pixel_array
        return ds, arr
    except Exception:
        rows = int(getattr(ds, "Rows", 512))
        cols = int(getattr(ds, "Columns", 512))
        return ds, np.zeros((rows, cols), dtype=np.float32)


def window_from_array(ds, img_arr, WL=400, WW=1800):
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))

    img = img_arr.astype(np.float32, copy=False)
    if slope != 1.0:
        img = img * slope
    if intercept != 0.0:
        img = img + intercept

    upper = float(WL + WW // 2)
    lower = float(WL - WW // 2)

    X = np.clip(img, lower, upper)
    denom = upper - lower
    if denom <= 0:
        return np.zeros_like(img_arr, dtype=np.uint8)
    X = (X - lower) / denom
    X = (X * 255.0).astype(np.uint8, copy=False)
    return X




## === cell 8
mean = np.array([0.456, 0.456, 0.456], dtype=np.float32)
std = np.array([0.224, 0.224, 0.224], dtype=np.float32)

_mean = mean.reshape(1, 1, 3)
_std = std.reshape(1, 1, 3)
_inv255 = np.float32(1.0 / 255.0)


def center_crop_np(img_hwc: np.ndarray, crop_size: int):
    h, w = img_hwc.shape[:2]
    ch = crop_size
    cw = crop_size
    y1 = max(0, (h - ch) // 2)
    x1 = max(0, (w - cw) // 2)
    y2 = y1 + ch
    x2 = x1 + cw
    return img_hwc[y1:y2, x1:x2]


def img2tensor_chw_float(img_hwc_float: np.ndarray):
    return torch.from_numpy(
        np.ascontiguousarray(np.transpose(img_hwc_float, (2, 0, 1)))
    )




## === cell 9
class CSFAllSlicesDataset(Dataset):
    def __init__(
        self, study_id_list, selected_image_dict, target_size, crop_size, uid2idx
    ):
        self.study_id_list = study_id_list
        self.selected_image_dict = selected_image_dict
        self.target_size = int(target_size)
        self.crop_size = int(crop_size)
        self.uid2idx = uid2idx

        self.items = []
        self.pos = []
        cur = 0
        for uid in self.study_id_list:
            uid_idx = self.uid2idx[uid]
            uid_dir = os.path.join(TEST_PATH, uid)
            image_list = self.selected_image_dict.get(uid, [])
            for s in image_list:
                t = get_triplet_slices(uid, int(s))
                if t is None:
                    self.items.append((uid_idx, None, None, None))
                else:
                    s0, s1, s2 = t
                    self.items.append(
                        (
                            uid_idx,
                            os.path.join(uid_dir, f"{int(s0)}.dcm"),
                            os.path.join(uid_dir, f"{int(s1)}.dcm"),
                            os.path.join(uid_dir, f"{int(s2)}.dcm"),
                        )
                    )
                self.pos.append(cur)
                cur += 1
        self.pos = np.asarray(self.pos, dtype=np.int64)

        blank = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.float32)
        blank = (blank - _mean) / _std
        self._blank_tensor = img2tensor_chw_float(blank)

        self._cache = {}
        self._cache_order = []
        self._cache_max = (
            256  # bounded to avoid memory blow-up; large enough to capture local reuse
        )

    def __len__(self):
        return len(self.items)

    def _get_windowed_u8(self, dcm_path: str):
        v = self._cache.get(dcm_path)
        if v is not None:
            return v
        ds, arr = read_dicom_pixel_array(dcm_path)
        if ds is None:
            out = None
        else:
            out = window_from_array(ds, arr)
        self._cache[dcm_path] = out
        self._cache_order.append(dcm_path)
        if len(self._cache_order) > self._cache_max:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)
        return out

    def __getitem__(self, index):
        uid_idx, p0, p1, p2 = self.items[index]
        if p0 is None:
            return self._blank_tensor, uid_idx, self.pos[index]

        a0 = self._get_windowed_u8(p0)
        a1 = self._get_windowed_u8(p1)
        a2 = self._get_windowed_u8(p2)
        if a0 is None or a1 is None or a2 is None:
            return self._blank_tensor, uid_idx, self.pos[index]

        stacked_img = np.stack([a0, a1, a2], axis=-1)  # HWC uint8
        stacked_img = cv2.resize(
            stacked_img,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_LINEAR,
        )
        stacked_img = center_crop_np(stacked_img, self.crop_size)

        x = stacked_img.astype(np.float32, copy=False)
        x *= _inv255
        x -= _mean
        x /= _std
        X = img2tensor_chw_float(x)
        return X, uid_idx, self.pos[index]




## === cell 10
class CSFInstanceDataset(Dataset):
    def __init__(self, feature_array_dict, study_id_list, seq_len):
        self.feature_array_dict = feature_array_dict
        self.study_id_list = study_id_list
        self.seq_len = int(seq_len)

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




## === cell 11
def find_weight_file(rel_candidates):
    base_candidates = ["../input", "/kaggle/input", "/kaggle/data"]
    for base in base_candidates:
        for rel in rel_candidates:
            p = os.path.join(base, rel.lstrip("/"))
            if os.path.exists(p):
                return p
    return None




## === cell 12
class ConvNextCNN_B_Feature(nn.Module):
    def __init__(self):
        super(ConvNextCNN_B_Feature, self).__init__()
        m = convnext_base()  # architecture preserved
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




## === cell 13
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




## === cell 14
lv1_model = ConvNextCNN_B_Feature()
lv2_model = CSFNet(input_len=config["feature_size"], lstm_size=config["lstm_size"])

lv1_w = find_weight_file(
    [
        "cnn-lstm-oct-21/run_5/run_5/model_3.pth",
        "cnn-lstm-oct-21/model_3.pth",
    ]
)
lv2_w = find_weight_file(
    [
        "cnn-lstm-oct-21/run_5/run_5/model_lstm_3.pth",
        "cnn-lstm-oct-21/model_lstm_3.pth",
    ]
)

if lv1_w is not None:
    lv1_model.load_state_dict(torch.load(lv1_w, map_location="cpu"))
if lv2_w is not None:
    lv2_model.load_state_dict(torch.load(lv2_w, map_location="cpu"))

lv1_model = lv1_model.to(device).eval()
lv2_model = lv2_model.to(device).eval()

if device.type == "cuda":
    lv1_model = lv1_model.to(memory_format=torch.channels_last)

try:
    if hasattr(torch, "compile"):
        lv1_model = torch.compile(lv1_model)
        lv2_model = torch.compile(lv2_model)
except Exception:
    pass

print("Loaded weights:", lv1_w, lv2_w)
if lv1_w is None or lv2_w is None:
    print(
        "WARNING: One or more weight files missing; using default initialization for missing weights."
    )




## === cell 15
def clip_probs(p, eps=1e-5):
    return float(np.clip(p, eps, 1.0 - eps))




## === cell 16
submission_dict = {"row_id": [], "fractured": []}
feature_array_dict = {}

cpu_count = os.cpu_count() or 2
if torch.cuda.is_available():
    num_workers_img = min(8, max(2, cpu_count // 2))
else:
    num_workers_img = min(4, max(1, cpu_count // 2))


def _seed_worker(worker_id):
    seed = 42 + worker_id
    random.seed(seed)
    np.random.seed(seed)


uid2idx = {uid: i for i, uid in enumerate(study_id_list)}
idx2uid = study_id_list  # index -> uid

all_ds = CSFAllSlicesDataset(
    study_id_list=study_id_list,
    selected_image_dict=selected_image_dict,
    target_size=config["target_size"],
    crop_size=config["crop_size"],
    uid2idx=uid2idx,
)

dl_kwargs = dict(
    batch_size=config["batch_size_image_level"],
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
    num_workers=num_workers_img,
    persistent_workers=(num_workers_img > 0),
    worker_init_fn=_seed_worker if num_workers_img > 0 else None,
)
if num_workers_img > 0:
    dl_kwargs["prefetch_factor"] = 4

try:
    generator = DataLoader(
        all_ds,
        **dl_kwargs,
        pin_memory_device="cuda" if torch.cuda.is_available() else "",
    )
except TypeError:
    generator = DataLoader(all_ds, **dl_kwargs)

uid_counts = {uid: len(selected_image_dict.get(uid, [])) for uid in study_id_list}
total_slices = int(sum(uid_counts.values()))

concat_features = np.zeros((total_slices, config["feature_size"]), dtype=np.float32)

uid_sum = np.zeros((len(study_id_list), 7), dtype=np.float32)
uid_seen = np.zeros((len(study_id_list),), dtype=np.int32)

for images, uid_idx, pos in tqdm(
    generator, total=len(generator), desc="Stage1 all slices"
):
    with torch.inference_mode():
        images = images.to(device, non_blocking=True)
        if device.type == "cuda":
            images = images.to(memory_format=torch.channels_last)
        features, preds = lv1_model(images)
        feats_np = features.detach().cpu().numpy()
        preds_np = preds.sigmoid().detach().cpu().numpy()

    uid_idx_np = uid_idx.numpy().astype(np.int32, copy=False)
    pos_np = pos.numpy().astype(np.int64, copy=False)

    concat_features[pos_np] = feats_np
    np.add.at(uid_sum, uid_idx_np, preds_np)
    uid_seen[uid_idx_np] += 1

cur = 0
for uid in study_id_list:
    n = uid_counts[uid]
    if n == 0:
        feature_array_dict[uid] = np.zeros(
            (0, config["feature_size"]), dtype=np.float32
        )
        mean_preds = np.full((7,), 0.5, dtype=np.float32)
    else:
        feats = concat_features[cur : cur + n]
        feature_array_dict[uid] = feats
        mean_preds = uid_sum[uid2idx[uid]] / float(n)
    cur += n

    submission_dict["row_id"].append(f"{uid}_C1")
    submission_dict["fractured"].append(clip_probs(mean_preds[0]))
    submission_dict["row_id"].append(f"{uid}_C2")
    submission_dict["fractured"].append(clip_probs(mean_preds[1]))
    submission_dict["row_id"].append(f"{uid}_C3")
    submission_dict["fractured"].append(clip_probs(mean_preds[2]))
    submission_dict["row_id"].append(f"{uid}_C4")
    submission_dict["fractured"].append(clip_probs(mean_preds[3]))
    submission_dict["row_id"].append(f"{uid}_C5")
    submission_dict["fractured"].append(clip_probs(mean_preds[4]))
    submission_dict["row_id"].append(f"{uid}_C6")
    submission_dict["fractured"].append(clip_probs(mean_preds[5]))
    submission_dict["row_id"].append(f"{uid}_C7")
    submission_dict["fractured"].append(clip_probs(mean_preds[6]))



## === cell 17
missing = [uid for uid in study_id_list if uid not in feature_array_dict]
print("Missing feature arrays:", len(missing))
if len(missing) > 0:
    raise KeyError(f"Missing extracted features for some studies, e.g. {missing[:5]}")



## === cell 18
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
    num_workers=0,
)

for features, list_uid in tqdm(
    generator, total=len(generator), desc="Stage2 patient_overall"
):
    with torch.inference_mode():
        features = features.to(device, non_blocking=True)
        preds = lv2_model(features)
        preds = np.squeeze(preds.sigmoid().detach().cpu().numpy())

    for j in range(len(list_uid)):
        submission_dict["row_id"].append(f"{list_uid[j]}_patient_overall")
        submission_dict["fractured"].append(clip_probs(preds[j]))



## === cell 19
sub_df = pd.DataFrame.from_dict(submission_dict)
sub_df = sub_df.drop_duplicates(subset=["row_id"], keep="last")
sub_df = test_df[["row_id"]].merge(sub_df, on="row_id", how="left")
sub_df["fractured"] = sub_df["fractured"].fillna(0.5).astype(float).clip(1e-5, 1 - 1e-5)

print(sub_df.head())
print("Submission rows:", len(sub_df), "Expected:", len(test_df))



## === cell 20
assert list(sub_df.columns) == ["row_id", "fractured"]
assert len(sub_df) == len(test_df)
assert sub_df["fractured"].between(0.0, 1.0).all()
sub_df



## === cell 21
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print("Saved to:", os.path.abspath("submission.csv"))
