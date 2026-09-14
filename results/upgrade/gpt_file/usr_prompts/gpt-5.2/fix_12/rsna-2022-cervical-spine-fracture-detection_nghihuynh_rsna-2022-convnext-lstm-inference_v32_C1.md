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

# 5. Code solution

## === cell 0
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys
import glob
import time
import pickle
import warnings
import gc
import resource
import multiprocessing as mp
from collections import defaultdict

import numpy as np
import pandas as pd

import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pydicom
from pydicom.errors import InvalidDicomError

from torchvision.models.convnext import convnext_base, ConvNeXt_Base_Weights

warnings.filterwarnings("ignore")

try:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target_soft = min(max(soft, 8192), hard)
    if target_soft > soft:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target_soft, hard))
except Exception:
    pass

try:
    mp.set_start_method("fork", force=True)
except Exception:
    pass

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass

try:
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
except Exception:
    pass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_INPUT_CANDIDATES = [
    "../input/rsna-2022-cervical-spine-fracture-detection",
    "/kaggle/input/rsna-2022-cervical-spine-fracture-detection",
    "/kaggle/data/rsna-2022-cervical-spine-fracture-detection",
    "/kaggle/data/rsna-2022-cervical-spine-fracture-detection/rsna-2022-cervical-spine-fracture-detection",
]
BASE_INPUT = None
for p in BASE_INPUT_CANDIDATES:
    if os.path.exists(p):
        BASE_INPUT = p
        break
if BASE_INPUT is None:
    raise FileNotFoundError(
        "Could not locate rsna-2022-cervical-spine-fracture-detection dataset folder."
    )

TEST_CSV_PATH = os.path.join(BASE_INPUT, "test.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_INPUT, "sample_submission.csv")
TEST_PATH = os.path.join(BASE_INPUT, "test_images")

print("BASE_INPUT:", BASE_INPUT)
print("TEST_CSV_PATH exists:", os.path.exists(TEST_CSV_PATH))
print("SAMPLE_SUB_PATH exists:", os.path.exists(SAMPLE_SUB_PATH))
print("TEST_PATH exists:", os.path.exists(TEST_PATH))




## === cell 1
config = {
    "seq_len": 150,
    "feature_size": 1024,
    "lstm_size": 128,
    "target_size": 512,
    "crop_size": 368,
    "num_classes": 7,
    "batch_size_image_level": 64,
    "batch_size_patient_level": 8,
}

LABELS_7 = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]




## === cell 2
def load_df_test():
    df_test = pd.read_csv(TEST_CSV_PATH)
    if len(df_test) > 0 and df_test.iloc[0].row_id == "1.2.826.0.1.3680043.10197_C1":
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


test_df = load_df_test()
study_id_list = list(test_df.StudyInstanceUID.unique())

print("test_df shape:", test_df.shape)
print("num unique studies:", len(study_id_list))




## === cell 3
def _list_dicom_numbers(study_dir: str):
    try:
        with os.scandir(study_dir) as it:
            nums = []
            for e in it:
                if not e.is_file():
                    continue
                name = e.name
                if not name.endswith(".dcm"):
                    continue
                base = name[:-4]
                if base.isdigit():
                    nums.append(int(base))
        nums.sort()
        return nums
    except FileNotFoundError:
        return []
    except Exception:
        paths = glob.glob(os.path.join(study_dir, "*.dcm"))
        if not paths:
            return []
        nums = []
        for p in paths:
            base = os.path.basename(p)[:-4]
            if base.isdigit():
                nums.append(int(base))
        nums.sort()
        return nums


selected_image_dict = {}
available_slices_dict = {}
for uid in study_id_list:
    study_dir = os.path.join(TEST_PATH, uid)
    slice_nums = _list_dicom_numbers(study_dir)
    available_slices_dict[uid] = slice_nums
    n = len(slice_nums)

    if n == 0:
        selected_image_dict[uid] = []
        continue

    if n < 3:
        selected_image_dict[uid] = slice_nums[:]  # just use what exists
        continue

    middle_slice_idx = n // 2
    k = max(1, int(0.15 * n))

    start = max(1, middle_slice_idx - k)  # ensure room for neighbors
    end = min(n - 2, middle_slice_idx + k)  # ensure room for neighbors
    sel = slice_nums[start : end + 1]
    if len(sel) == 0:
        sel = [slice_nums[middle_slice_idx]]
    selected_image_dict[uid] = sel

if len(study_id_list) > 0:
    print("example selected slices:", selected_image_dict[study_id_list[0]][:10])




## === cell 4
_DCM_TAGS_NEEDED = ["RescaleSlope", "RescaleIntercept", "PixelData"]


def safe_dcmread(path, stop_before_pixels=False):
    try:
        return pydicom.dcmread(
            path,
            force=True,
            stop_before_pixels=stop_before_pixels,
            specific_tags=None if stop_before_pixels else _DCM_TAGS_NEEDED,
            read_only=True,
            defer_size="1 KB",
        )
    except InvalidDicomError:
        try:
            return pydicom.dcmread(
                path,
                force=True,
                stop_before_pixels=stop_before_pixels,
                specific_tags=None if stop_before_pixels else _DCM_TAGS_NEEDED,
                read_only=True,
                defer_size="1 KB",
            )
        except Exception:
            return None
    except Exception:
        return None


def window(data, WL=400, WW=1800, fallback_shape=(512, 512)):
    if data is None:
        return np.zeros(fallback_shape, dtype="uint8")

    slope = float(getattr(data, "RescaleSlope", 1.0))
    intercept = float(getattr(data, "RescaleIntercept", 0.0))

    try:
        img = data.pixel_array.astype(np.float32, copy=False)
    except Exception:
        return np.zeros(fallback_shape, dtype="uint8")

    img = img * slope + intercept

    upper = float(WL + WW // 2)
    lower = float(WL - WW // 2)

    X = np.clip(img, lower, upper, out=img)  # reuse buffer
    X = (X - lower) * (255.0 / (upper - lower))
    return X.astype("uint8", copy=False)


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))




## === cell 5
mean = np.array([0.456, 0.456, 0.456], dtype=np.float32)
std = np.array([0.224, 0.224, 0.224], dtype=np.float32)


class CSFFlatImageDataset(Dataset):
    def __init__(self, items, target_size, crop_size, available_slices_dict):
        """
        items: list of (uid, center_slice_num)
        """
        self.items = items
        self.target_size = target_size
        self.crop_size = crop_size
        self.available_slices_dict = available_slices_dict

        self._crop_top = (self.target_size - self.crop_size) // 2
        self._crop_left = (self.target_size - self.crop_size) // 2
        self._crop_bottom = self._crop_top + self.crop_size
        self._crop_right = self._crop_left + self.crop_size

    def __len__(self):
        return len(self.items)

    @staticmethod
    def _nearest_available(avail_arr: np.ndarray, t: int) -> int:
        if avail_arr.size == 0:
            return int(t)
        idx = int(np.searchsorted(avail_arr, t))
        if idx <= 0:
            return int(avail_arr[0])
        if idx >= avail_arr.size:
            return int(avail_arr[-1])
        before = int(avail_arr[idx - 1])
        after = int(avail_arr[idx])
        if abs(before - t) <= abs(after - t):
            return before
        return after

    @staticmethod
    def _neighbor(avail_set: set, avail_arr: np.ndarray, s: int, delta: int) -> int:
        t = int(s) + int(delta)
        if t in avail_set:
            return t
        return CSFFlatImageDataset._nearest_available(avail_arr, t)

    def __getitem__(self, index):
        uid, s = self.items[index]
        s = int(s)

        avail = self.available_slices_dict.get(uid, [])
        avail_arr = np.asarray(avail, dtype=np.int32)
        if avail_arr.size > 1 and not np.all(avail_arr[:-1] <= avail_arr[1:]):
            avail_arr.sort()
        avail_set = set(avail)

        s1 = self._neighbor(avail_set, avail_arr, s, -1)
        s2 = s if s in avail_set else self._neighbor(avail_set, avail_arr, s, 0)
        s3 = self._neighbor(avail_set, avail_arr, s, +1)

        study_dir = os.path.join(TEST_PATH, uid)

        def _read_uint8(slice_num: int) -> np.ndarray:
            p = os.path.join(study_dir, f"{slice_num}.dcm")
            d = safe_dcmread(p, stop_before_pixels=False)
            return window(d)

        imgs = [_read_uint8(s1), _read_uint8(s2), _read_uint8(s3)]
        stacked_img = np.stack(imgs, axis=-1)

        stacked_img = cv2.resize(
            stacked_img,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_LINEAR,
        )

        X = stacked_img[
            self._crop_top : self._crop_bottom, self._crop_left : self._crop_right, :
        ]
        X = img2tensor((X / 255.0 - mean) / std)
        return X, uid


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

        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)
        X = torch.from_numpy(x)
        return X, uid




## === cell 6
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




## === cell 7
lv1_model = ConvNextCNN_B_Feature()


def _find_first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


ckpt_lv1 = _find_first_existing(
    [
        "../input/cnn-lstm-oct-27/run_0/model_best.pth",
        "/kaggle/input/cnn-lstm-oct-27/run_0/model_best.pth",
    ]
)
if ckpt_lv1 is not None:
    state = torch.load(ckpt_lv1, map_location="cpu")
    lv1_model.load_state_dict(state, strict=True)
    print("Loaded lv1 checkpoint:", ckpt_lv1)
else:
    w = ConvNeXt_Base_Weights.IMAGENET1K_V1
    m_pre = convnext_base(weights=w)
    lv1_model.features.load_state_dict(m_pre.features.state_dict(), strict=True)
    print(
        "lv1 checkpoint not found; using torchvision ConvNeXt-Base ImageNet weights for features."
    )

lv1_model = lv1_model.to(DEVICE).eval()

lv2_model = CSFNet(input_len=config["feature_size"], lstm_size=config["lstm_size"])
ckpt_lv2 = _find_first_existing(
    [
        "../input/cnn-lstm-oct-26/run_0/model_lstm_0.pth",
        "/kaggle/input/cnn-lstm-oct-26/run_0/model_lstm_0.pth",
    ]
)
if ckpt_lv2 is not None:
    state2 = torch.load(ckpt_lv2, map_location="cpu")
    lv2_model.load_state_dict(state2, strict=True)
    print("Loaded lv2 checkpoint:", ckpt_lv2)
else:
    print(
        "lv2 checkpoint not found; using randomly initialized lv2 (will reduce score, but allows submission)."
    )

lv2_model = lv2_model.to(DEVICE).eval()




## === cell 8
CPU_COUNT = os.cpu_count() or 2
if torch.cuda.is_available():
    NUM_WORKERS_IMG = max(2, min(8, CPU_COUNT))
else:
    NUM_WORKERS_IMG = max(2, min(8, CPU_COUNT // 2))
PREFETCH = 4

submission_rows = []
feature_array_dict = {}

pin = torch.cuda.is_available()

PERSISTENT = True

flat_items = []
flat_offsets = {}  # uid -> (start, end) in flat_items
pos = 0
for uid in study_id_list:
    image_list = selected_image_dict.get(uid, [])
    if len(image_list) == 0:
        flat_offsets[uid] = (pos, pos)
        continue
    start = pos
    for s in image_list:
        flat_items.append((uid, int(s)))
        pos += 1
    flat_offsets[uid] = (start, pos)

flat_ds = CSFFlatImageDataset(
    items=flat_items,
    target_size=config["target_size"],
    crop_size=config["crop_size"],
    available_slices_dict=available_slices_dict,
)

flat_loader = DataLoader(
    flat_ds,
    batch_size=config["batch_size_image_level"],
    shuffle=False,
    pin_memory=pin,
    drop_last=False,
    num_workers=NUM_WORKERS_IMG,
    persistent_workers=PERSISTENT if NUM_WORKERS_IMG > 0 else False,
    prefetch_factor=PREFETCH if NUM_WORKERS_IMG > 0 else None,
)

logit_sum = {uid: np.zeros((7,), dtype=np.float64) for uid in study_id_list}
logit_cnt = {uid: 0 for uid in study_id_list}

for uid in study_id_list:
    n = len(selected_image_dict.get(uid, []))
    if n == 0:
        feature_array_dict[uid] = np.zeros(
            (1, config["feature_size"]), dtype=np.float32
        )
    else:
        feature_array_dict[uid] = np.empty(
            (n, config["feature_size"]), dtype=np.float32
        )

feat_write_pos = {uid: 0 for uid in study_id_list}

with torch.inference_mode():
    for images, uids in tqdm(
        flat_loader, total=len(flat_loader), desc="Stage1 batches"
    ):
        images = images.to(DEVICE, non_blocking=True)
        features, preds = lv1_model(images)

        feats_np = features.detach().cpu().numpy()
        preds_sig = preds.sigmoid().detach().cpu().numpy()  # (B,7)

        uids_list = list(uids)
        for i, uid in enumerate(uids_list):
            j = feat_write_pos[uid]
            feature_array_dict[uid][j] = feats_np[i]
            feat_write_pos[uid] = j + 1
            logit_sum[uid] += preds_sig[i]
            logit_cnt[uid] += 1

        del images, uids, features, preds, feats_np, preds_sig, uids_list

for uid in study_id_list:
    n = logit_cnt[uid]
    if n == 0:
        mean_preds = np.full((7,), 0.01, dtype=np.float32)
    else:
        mean_preds = (logit_sum[uid] / float(n)).astype(np.float32)
    for k, lab in enumerate(LABELS_7):
        submission_rows.append(
            {
                "StudyInstanceUID": uid,
                "prediction_type": lab,
                "fractured": float(mean_preds[k]),
            }
        )

stage1_df = pd.DataFrame(submission_rows)
print("Stage1 preds shape:", stage1_df.shape)
print("feature_array_dict size:", len(feature_array_dict))
print("NUM_WORKERS_IMG:", NUM_WORKERS_IMG)

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()




## === cell 9
NUM_WORKERS_SEQ = 0

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
    num_workers=NUM_WORKERS_SEQ,
    persistent_workers=False,
    prefetch_factor=None,
)

patient_rows = []

try:
    with torch.inference_mode():
        for features, list_uid in tqdm(
            generator2, total=len(generator2), desc="Stage2 batches"
        ):
            features = features.to(DEVICE, non_blocking=True)
            preds = lv2_model(features).sigmoid().detach().cpu().numpy().reshape(-1)

            for uid, p in zip(list_uid, preds):
                patient_rows.append(
                    {
                        "StudyInstanceUID": str(uid),
                        "prediction_type": "patient_overall",
                        "fractured": float(p),
                    }
                )

            del features, list_uid, preds
except RuntimeError as e:
    msg = str(e)
    if "CUBLAS_WORKSPACE_CONFIG" in msg or "not deterministic" in msg:
        warnings.warn(
            "Deterministic CuBLAS error encountered; disabling deterministic algorithms to continue."
        )
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass
        with torch.inference_mode():
            for features, list_uid in tqdm(
                generator2,
                total=len(generator2),
                desc="Stage2 batches (nondet fallback)",
            ):
                features = features.to(DEVICE, non_blocking=True)
                preds = lv2_model(features).sigmoid().detach().cpu().numpy().reshape(-1)

                for uid, p in zip(list_uid, preds):
                    patient_rows.append(
                        {
                            "StudyInstanceUID": str(uid),
                            "prediction_type": "patient_overall",
                            "fractured": float(p),
                        }
                    )

                del features, list_uid, preds
    else:
        raise

stage2_df = pd.DataFrame(patient_rows)
print("Stage2 preds shape:", stage2_df.shape)

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()




## === cell 10
if "stage1_df" not in globals() or stage1_df is None or len(stage1_df) == 0:
    stage1_df = pd.DataFrame(
        [
            {"StudyInstanceUID": uid, "prediction_type": lab, "fractured": 0.01}
            for uid in study_id_list
            for lab in LABELS_7
        ]
    )
if "stage2_df" not in globals() or stage2_df is None or len(stage2_df) == 0:
    stage2_df = pd.DataFrame(
        [
            {
                "StudyInstanceUID": uid,
                "prediction_type": "patient_overall",
                "fractured": 0.01,
            }
            for uid in study_id_list
        ]
    )

pred_df = pd.concat([stage1_df, stage2_df], axis=0, ignore_index=True)

merged = test_df.merge(pred_df, on=["StudyInstanceUID", "prediction_type"], how="left")

if merged["fractured"].isna().any():
    merged["fractured"] = merged["fractured"].fillna(0.01)

merged["fractured"] = merged["fractured"].clip(1e-6, 1 - 1e-6)

sub_df = merged[["row_id", "fractured"]].copy()
sub_df.to_csv("submission.csv", index=False)

print(sub_df.head())
print("submission.csv written with shape:", sub_df.shape)
print("columns:", list(sub_df.columns))
print("num missing after fill:", int(sub_df["fractured"].isna().sum()))
print("saved to:", os.path.abspath("submission.csv"))
