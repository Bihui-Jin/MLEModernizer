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
import glob
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")




## === cell 1
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import pydicom

try:
    pydicom.config.pixel_array_options(use_v2_backend=True)
except Exception:
    pass




## === cell 2
BASE_PATH = "../input/rsna-2022-cervical-spine-fracture-detection"
TEST_CSV = f"{BASE_PATH}/test.csv"
SAMPLE_SUB = f"{BASE_PATH}/sample_submission.csv"
TEST_PATH = f"{BASE_PATH}/test_images"




## === cell 3
config = {
    "seq_len": 150,
    "feature_size": 1024,
    "lstm_size": 128,
    "target_size": 368,
    "crop_size": 320,
    "num_classes": 7,
    "batch_size_image_level": 64,
    "batch_size_patient_level": 8,
}




## === cell 4
def load_df_test():
    df_test = pd.read_csv(TEST_CSV)

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
print("Num studies in test_df:", len(study_id_list))
print("Num rows in test_df:", len(test_df))




## === cell 5
_slice_numbers_cache = {}


def get_sorted_slice_numbers(uid: str):
    uid = str(uid)
    cached = _slice_numbers_cache.get(uid, None)
    if cached is not None:
        return cached

    folder = os.path.join(TEST_PATH, uid)
    if not os.path.isdir(folder):
        _slice_numbers_cache[uid] = []
        return []

    nums = []
    try:
        with os.scandir(folder) as it:
            for e in it:
                if not e.is_file():
                    continue
                name = e.name
                if not name.endswith(".dcm"):
                    continue
                stem = name[:-4]
                try:
                    nums.append(int(stem))
                except Exception:
                    continue
    except Exception:
        nums = []

    if not nums:
        _slice_numbers_cache[uid] = []
        return []

    nums.sort()
    _slice_numbers_cache[uid] = nums
    return nums


selected_image_dict = {}
slice_numbers_dict = {}
slice_numbers_arr_dict = {}  # uid -> np.int32 array for fast searchsorted

for uid in study_id_list:
    slice_nums = get_sorted_slice_numbers(uid)
    slice_numbers_dict[uid] = slice_nums
    slice_numbers_arr_dict[uid] = (
        np.asarray(slice_nums, dtype=np.int32)
        if slice_nums
        else np.empty((0,), dtype=np.int32)
    )
    if len(slice_nums) == 0:
        selected_image_dict[uid] = []
        continue

    mid_idx = len(slice_nums) // 2
    num_left = num_right = int(0.15 * len(slice_nums))
    left = max(0, mid_idx - num_left)
    right = min(len(slice_nums) - 1, mid_idx + num_right)

    sel = list(range(left, mid_idx)) + list(range(mid_idx + 1, right + 1))
    selected_image_dict[uid] = [slice_nums[k] for k in sel]

if study_id_list:
    uid0 = study_id_list[0]
    print("Example uid:", uid0)
    print("Total slices:", len(slice_numbers_dict[uid0]))
    print("Selected slices:", selected_image_dict[uid0][:10], "...")
else:
    print("No studies found in test_df.")




## === cell 6
def safe_pixel_array(ds: "pydicom.dataset.FileDataset"):
    try:
        return ds.pixel_array
    except Exception:
        try:
            if "PixelData" in ds and getattr(ds, "BitsAllocated", None) in (8, 16):
                rows, cols = int(ds.Rows), int(ds.Columns)
                if ds.BitsAllocated == 16:
                    arr = np.frombuffer(ds.PixelData, dtype=np.uint16)
                else:
                    arr = np.frombuffer(ds.PixelData, dtype=np.uint8)
                n = rows * cols
                if arr.size >= n:
                    return arr[:n].reshape(rows, cols)
        except Exception:
            pass
    return None


def window(ds, WL=400, WW=1800):
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))

    img = safe_pixel_array(ds)
    if img is None:
        return np.zeros((512, 512), dtype="uint8")

    img = img.astype(np.float32, copy=False) * slope + intercept
    upper, lower = WL + WW // 2, WL - WW // 2
    X = np.clip(img, lower, upper)

    mn = float(X.min())
    X = X - mn
    mx = float(X.max())
    if mx > 0.0:
        X = X / mx
    return (X * 255.0).astype("uint8", copy=False)


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))




## === cell 7
mean = np.array([0.456, 0.456, 0.456], dtype=np.float32)
std = np.array([0.224, 0.224, 0.224], dtype=np.float32)


class CSFGlobalImageDataset(Dataset):
    def __init__(self, items, target_size):
        """
        items: list of tuples (uid, slice_num) where slice_num is a selected mid slice.
        """
        self.items = items
        self.target_size = target_size
        self._slice_numbers_arr_dict = slice_numbers_arr_dict
        self._path_cache = {}

        self._img_cache = {}  # (uid, slice_num) -> uint8 2D
        self._img_cache_order = []
        self._img_cache_max = 512  # bounded; correctness unchanged

    def __len__(self):
        return len(self.items)

    def _path(self, uid: str):
        p = self._path_cache.get(uid)
        if p is None:
            p = os.path.join(TEST_PATH, uid)
            self._path_cache[uid] = p
        return p

    def _cache_put(self, key, value):
        if key in self._img_cache:
            return
        self._img_cache[key] = value
        self._img_cache_order.append(key)
        if len(self._img_cache_order) > self._img_cache_max:
            old = self._img_cache_order.pop(0)
            self._img_cache.pop(old, None)

    def _read_one(self, uid: str, slice_num: int):
        key = (uid, int(slice_num))
        cached = self._img_cache.get(key, None)
        if cached is not None:
            return cached

        fp = f"{self._path(uid)}/{slice_num}.dcm"
        try:
            ds = pydicom.dcmread(
                fp,
                force=True,
                stop_before_pixels=False,
                specific_tags=[
                    "PixelData",
                    "Rows",
                    "Columns",
                    "BitsAllocated",
                    "RescaleSlope",
                    "RescaleIntercept",
                ],
            )
            img = window(ds)
        except Exception:
            img = np.zeros((512, 512), dtype="uint8")

        self._cache_put(key, img)
        return img

    def __getitem__(self, index):
        uid, mid_num = self.items[index]
        uid = str(uid)

        all_arr = self._slice_numbers_arr_dict.get(uid, None)
        if all_arr is None or all_arr.size == 0:
            stacked_img = np.zeros(
                (self.target_size, self.target_size, 3), dtype=np.uint8
            )
            X = img2tensor((stacked_img / 255.0 - mean) / std)
            return X, uid

        pos = int(np.searchsorted(all_arr, np.int32(mid_num)))
        if pos <= 0:
            left_num = int(all_arr[0])
            mid_num2 = int(all_arr[0])
            right_num = int(all_arr[1] if all_arr.size > 1 else all_arr[0])
        elif pos >= all_arr.size - 1:
            left_num = int(all_arr[-2] if all_arr.size > 1 else all_arr[-1])
            mid_num2 = int(all_arr[-1])
            right_num = int(all_arr[-1])
        else:
            left_num = int(all_arr[pos - 1])
            mid_num2 = int(all_arr[pos])
            right_num = int(all_arr[pos + 1])

        imgs = (
            self._read_one(uid, left_num),
            self._read_one(uid, mid_num2),
            self._read_one(uid, right_num),
        )
        stacked_img = np.stack(imgs, axis=-1)
        stacked_img = cv2.resize(
            stacked_img,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_LINEAR,
        )
        X = img2tensor((stacked_img / 255.0 - mean) / std)
        return X, uid




## === cell 8
class CSFInstanceDataset(Dataset):
    def __init__(self, feature_array_dict, study_id_list, seq_len):
        self.feature_array_dict = feature_array_dict
        self.study_id_list = study_id_list
        self.seq_len = seq_len

    def __len__(self):
        return len(self.study_id_list)

    def __getitem__(self, index):
        uid = self.study_id_list[index]
        feature_array = self.feature_array_dict.get(uid, None)
        if feature_array is None:
            x = np.zeros((self.seq_len, config["feature_size"]), dtype=np.float32)
        else:
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




## === cell 9
from torchvision.models.convnext import convnext_base, ConvNeXt_Base_Weights


class ConvNextCNN_B_Feature(nn.Module):
    def __init__(self, use_pretrained: bool = True):
        super(ConvNextCNN_B_Feature, self).__init__()
        weights = ConvNeXt_Base_Weights.DEFAULT if use_pretrained else None
        m = convnext_base(weights=weights)
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




## === cell 10
import random

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

try:
    cv2.setNumThreads(0)
except Exception:
    pass

lv1_w_path = "../input/cnn-lstm-oct-21/run_5/run_5/model_3.pth"
lv2_w_path = "../input/cnn-lstm-oct-21/run_5b/run_5b/model_lstm_3.pth"

have_lv1 = os.path.exists(lv1_w_path)
have_lv2 = os.path.exists(lv2_w_path)

lv1_model = ConvNextCNN_B_Feature(use_pretrained=(not have_lv1)).to(device)
lv2_model = CSFNet(input_len=config["feature_size"], lstm_size=config["lstm_size"]).to(
    device
)

if have_lv1:
    state = torch.load(lv1_w_path, map_location="cpu")
    lv1_model.load_state_dict(state, strict=False)
if have_lv2:
    state = torch.load(lv2_w_path, map_location="cpu")
    lv2_model.load_state_dict(state, strict=False)

lv1_model.eval()
lv2_model.eval()

print("CUDA available:", torch.cuda.is_available())
print("Found lv1 weights:", have_lv1)
print("Found lv2 weights:", have_lv2)
print("lv1 uses torchvision pretrained:", (not have_lv1))




## === cell 11
BASELINE_LEVEL = 0.05
BASELINE_OVERALL = 0.10

CALIBRATION_POWER = 0.85  # <1 pushes probs away from 0.5 slightly


def calibrate_probs(p: np.ndarray, power: float = CALIBRATION_POWER):
    p = np.asarray(p, dtype=np.float64)
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    logit = np.log(p / (1 - p))
    logit = logit / power
    out = 1 / (1 + np.exp(-logit))
    return out.astype(np.float32)


submission_dict = {"row_id": [], "fractured": []}
feature_array_dict = {}

run_stage1 = True

levels = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]

_cpu = os.cpu_count() or 2
_num_workers = min(12, max(2, _cpu - 1))
_persistent_workers = _num_workers > 0
_prefetch_factor = 4 if _num_workers > 0 else None

items = []
uid_offsets = {}  # uid -> (start_index_in_items, count)
for uid in study_id_list:
    image_list = selected_image_dict.get(uid, [])
    cnt = len(image_list)
    if cnt > 0 and run_stage1:
        uid_offsets[uid] = (len(items), cnt)
        items.extend([(uid, sn) for sn in image_list])
    feature_array_dict[uid] = np.zeros(
        (max(cnt, 1), config["feature_size"]), dtype=np.float32
    )

if (not run_stage1) or (len(items) == 0):
    for uid in study_id_list:
        for c in levels:
            submission_dict["row_id"].append(f"{uid}_{c}")
            submission_dict["fractured"].append(float(BASELINE_LEVEL))
else:
    uid_to_idx = {uid: i for i, uid in enumerate(uid_offsets.keys())}
    idx_to_uid = list(uid_offsets.keys())

    pred_sum = np.zeros((len(idx_to_uid), 7), dtype=np.float64)
    pred_count = np.zeros((len(idx_to_uid),), dtype=np.int32)

    global_ds = CSFGlobalImageDataset(items=items, target_size=config["target_size"])

    dl_kwargs = dict(
        dataset=global_ds,
        batch_size=config["batch_size_image_level"],
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        num_workers=_num_workers,
        persistent_workers=_persistent_workers,
    )
    if _prefetch_factor is not None:
        dl_kwargs["prefetch_factor"] = _prefetch_factor

    global_loader = DataLoader(**dl_kwargs)

    uid_write_pos = {uid: 0 for uid in uid_offsets.keys()}

    for images, uids in tqdm(
        global_loader, desc="Stage1 global slices", total=len(global_loader)
    ):
        with torch.inference_mode():
            images = images.to(device, non_blocking=True)
            features, preds = lv1_model(images)
            feat_np = features.detach().cpu().numpy()
            p_np = preds.sigmoid().detach().cpu().numpy()

        batch_uid_idx = np.empty((len(uids),), dtype=np.int32)
        for j, uid in enumerate(uids):
            uid = str(uid)
            pos = uid_write_pos.get(uid, None)
            if pos is None:
                batch_uid_idx[j] = -1
                continue
            feature_array_dict[uid][pos] = feat_np[j]
            uid_write_pos[uid] = pos + 1
            batch_uid_idx[j] = uid_to_idx[uid]

        valid = batch_uid_idx >= 0
        if np.any(valid):
            v_idx = batch_uid_idx[valid]
            np.add.at(pred_sum, v_idx, p_np[valid].astype(np.float64, copy=False))
            np.add.at(pred_count, v_idx, 1)

    for uid in study_id_list:
        if uid not in uid_offsets:
            mean_preds = np.full((7,), BASELINE_LEVEL, dtype=np.float32)
        else:
            i = uid_to_idx[uid]
            cnt = int(pred_count[i])
            if cnt == 0:
                mean_preds = np.full((7,), BASELINE_LEVEL, dtype=np.float32)
            else:
                mean_preds = (pred_sum[i] / cnt).astype(np.float32)
                mean_preds = calibrate_probs(mean_preds, CALIBRATION_POWER)

        for idx, c in enumerate(levels):
            submission_dict["row_id"].append(f"{uid}_{c}")
            submission_dict["fractured"].append(float(mean_preds[idx]))




## === cell 12
run_stage2 = have_lv2  # relies on extracted CNN features

if run_stage2:
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
        drop_last=False,
    )

    for features, list_uid in tqdm(
        generator2, desc="Stage2 patient_overall", total=len(generator2)
    ):
        with torch.inference_mode():
            features = features.to(device, non_blocking=True)
            preds = lv2_model(features)
            preds = np.squeeze(preds.sigmoid().detach().cpu().numpy())

        if np.ndim(preds) == 0:
            preds = np.array([float(preds)], dtype=np.float32)

        preds = calibrate_probs(preds, CALIBRATION_POWER)

        for j in range(len(list_uid)):
            submission_dict["row_id"].append(f"{list_uid[j]}_patient_overall")
            submission_dict["fractured"].append(float(preds[j]))
else:
    for uid in study_id_list:
        submission_dict["row_id"].append(f"{uid}_patient_overall")
        submission_dict["fractured"].append(float(BASELINE_OVERALL))




## === cell 13
pred_df = pd.DataFrame(submission_dict)
pred_df["fractured"] = pred_df["fractured"].astype(float)

eps = 1e-6
pred_df["fractured"] = pred_df["fractured"].clip(eps, 1 - eps)

test_rows = pd.read_csv(TEST_CSV)[["row_id"]]
sub_df = test_rows.merge(pred_df, on="row_id", how="left")

sub_df["fractured"] = (
    sub_df["fractured"].fillna(BASELINE_LEVEL).astype(float).clip(eps, 1 - eps)
)

print("Submission shape:", sub_df.shape)
print(sub_df.head())




## === cell 14
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with rows:", len(sub_df))
