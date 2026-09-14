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
import math
import random
import pickle
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd

import pydicom

import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision.models.convnext import convnext_base




## === cell 1
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_INPUT = "/kaggle/input/rsna-2022-cervical-spine-fracture-detection"
if not os.path.exists(BASE_INPUT):
    BASE_INPUT = "../input/rsna-2022-cervical-spine-fracture-detection"

TEST_CSV = os.path.join(BASE_INPUT, "test.csv")
SAMPLE_SUB_CSV = os.path.join(BASE_INPUT, "sample_submission.csv")
TEST_PATH = os.path.join(BASE_INPUT, "test_images")



## === cell 2
config = {
    "seq_len": 150,
    "feature_size": 1024,
    "lstm_size": 128,
    "target_size": 368,
    "crop_size": 320,
    "num_classes": 7,
    "batch_size_image_level": 32,
    "batch_size_patient_level": 8,
}

LEVELS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
EPS = 1e-5  # to avoid exact 0/1 probabilities (log loss inf)




## === cell 3
def load_df_test():
    df_test = pd.read_csv(TEST_CSV)
    return df_test


test_df = load_df_test()
study_id_list = list(test_df.StudyInstanceUID.unique())




## === cell 4
def get_slice_indices_for_uid(uid: str):
    """Return sorted available slice indices from filenames like '1.dcm', '10.dcm'."""
    folder = os.path.join(TEST_PATH, uid)
    idxs = []
    try:
        with os.scandir(folder) as it:
            for entry in it:
                if not entry.is_file():
                    continue
                name = entry.name
                if len(name) < 5 or name[-4:] != ".dcm":
                    continue
                stem = name[:-4]
                if stem.isdigit():
                    idxs.append(int(stem))
    except FileNotFoundError:
        return []
    idxs.sort()
    return idxs


selected_image_dict = {}
uid_to_slices = {}

for uid in tqdm(study_id_list, desc="Indexing test DICOM slices"):
    idxs = get_slice_indices_for_uid(uid)
    uid_to_slices[uid] = idxs
    if len(idxs) == 0:
        selected_image_dict[uid] = []
        continue

    mid_pos = len(idxs) // 2
    span = max(1, int(0.15 * len(idxs)))
    left = idxs[max(0, mid_pos - span) : mid_pos]
    right = idxs[mid_pos + 1 : min(len(idxs), mid_pos + span + 1)]
    selected = left + right
    if len(selected) == 0:
        selected = [idxs[mid_pos]]
    selected_image_dict[uid] = selected




## === cell 5
def safe_pixel_array(ds: pydicom.dataset.FileDataset):
    """
    Robust pixel decoding:
    - Try ds.pixel_array (works when plugins available)
    - Fallback to pydicom.pixels.pixel_array (newer pydicom backend)
    - Final fallback to zeros to avoid crashing (submission still valid)
    """
    try:
        return ds.pixel_array
    except Exception:
        try:
            from pydicom.pixels import pixel_array as px

            return px(ds)
        except Exception:
            rows = int(getattr(ds, "Rows", 512))
            cols = int(getattr(ds, "Columns", 512))
            return np.zeros((rows, cols), dtype=np.int16)


def window(ds, WL=400, WW=1800):
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    img = safe_pixel_array(ds).astype(np.float32)
    img = img * slope + intercept
    upper, lower = WL + WW // 2, WL - WW // 2
    x = np.clip(img, lower, upper)
    x = x - np.min(x)
    mx = np.max(x)
    if mx > 0:
        x = x / mx
    x = (x * 255.0).astype("uint8")
    return x


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


mean = np.array([0.456, 0.456, 0.456], dtype=np.float32)
std = np.array([0.224, 0.224, 0.224], dtype=np.float32)

_DICOM_TAGS_NEEDED = [
    "PixelData",
    "Rows",
    "Columns",
    "RescaleIntercept",
    "RescaleSlope",
    "PhotometricInterpretation",
    "BitsAllocated",
    "BitsStored",
    "HighBit",
    "PixelRepresentation",
    "SamplesPerPixel",
    "PlanarConfiguration",
]
_CACHE_DIR = "/kaggle/working/dcm_u8_cache"
os.makedirs(_CACHE_DIR, exist_ok=True)


def _cache_path_for_fp(fp: str) -> str:
    uid = os.path.basename(os.path.dirname(fp))
    fn = os.path.basename(fp)
    key = f"{uid}__{fn}__{abs(hash(fp)) & 0xFFFFFFFF:08x}.npy"
    return os.path.join(_CACHE_DIR, key)


def _read_and_window_u8_uncached(fp: str) -> np.ndarray:
    cache_fp = _cache_path_for_fp(fp)
    try:
        if os.path.exists(cache_fp):
            arr = np.load(cache_fp, allow_pickle=False, mmap_mode="r")
            return np.asarray(arr)
    except Exception:
        pass

    ds = pydicom.dcmread(
        fp,
        force=True,
        stop_before_pixels=False,  # need PixelData
        specific_tags=_DICOM_TAGS_NEEDED,  # minimal header for decoding + rescale
    )
    try:
        arr = window(ds)
    finally:
        try:
            ds.close()
        except Exception:
            pass

    try:
        np.save(cache_fp, arr, allow_pickle=False)
    except Exception:
        pass
    return arr


def _read_and_window_u8(fp: str) -> np.ndarray:
    return _read_and_window_u8_uncached(fp)




## === cell 6
def _closest_existing(idx: int, available, available_set):
    if idx in available_set:
        return idx
    if not available:
        return idx
    pos = np.searchsorted(available, idx)
    if pos <= 0:
        return available[0]
    if pos >= len(available):
        return available[-1]
    before = available[pos - 1]
    after = available[pos]
    return before if abs(idx - before) <= abs(after - idx) else after


uid_state = {}
for uid in study_id_list:
    av = uid_to_slices.get(uid, [])
    uid_state[uid] = (
        os.path.join(TEST_PATH, uid),
        av,
        set(av),
    )

flat_items = []  # (uid, fp_m1, fp_0, fp_p1)
uid_item_counts = {}
for uid in study_id_list:
    centers = selected_image_dict.get(uid, [])
    uid_item_counts[uid] = len(centers)
    if not centers:
        continue
    base_path, available, available_set = uid_state[uid]
    for center in centers:
        c = int(center)
        idx_m1 = _closest_existing(c - 1, available, available_set)
        idx_0 = _closest_existing(c, available, available_set)
        idx_p1 = _closest_existing(c + 1, available, available_set)
        flat_items.append(
            (
                uid,
                os.path.join(base_path, f"{idx_m1}.dcm"),
                os.path.join(base_path, f"{idx_0}.dcm"),
                os.path.join(base_path, f"{idx_p1}.dcm"),
            )
        )




## === cell 7
class FlatCSFImageDataset(Dataset):
    def __init__(self, items, target_size):
        """
        items: list of (uid, fp_m1, fp_0, fp_p1)
        """
        self.items = items
        self.target_size = target_size

        self._cache = None

    def _get_cache(self):
        if self._cache is None:
            self._cache = {}
            self._cache_order = []
            self._cache_max = 4096
        return self._cache

    def _cache_get(self, fp: str):
        cache = self._get_cache()
        arr = cache.get(fp, None)
        if arr is not None:
            return arr
        arr = _read_and_window_u8_uncached(fp)
        cache[fp] = arr
        self._cache_order.append(fp)
        if len(self._cache_order) > self._cache_max:
            old = self._cache_order.pop(0)
            cache.pop(old, None)
        return arr

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        uid, fp1, fp2, fp3 = self.items[index]

        imgs = (
            self._cache_get(fp1),
            self._cache_get(fp2),
            self._cache_get(fp3),
        )
        stacked_img = np.stack(imgs, axis=-1)
        stacked_img = cv2.resize(
            stacked_img,
            (config["target_size"], config["target_size"]),
            interpolation=cv2.INTER_AREA,
        )

        x_np = stacked_img.astype(np.float32) * (1.0 / 255.0)
        x_np = (x_np - mean) / std
        x = img2tensor(x_np)
        return x, uid


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
        x = torch.tensor(x, dtype=torch.float32)
        return x, uid




## === cell 8
class ConvNextCNN_B_Feature(nn.Module):
    def __init__(self):
        super().__init__()
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




## === cell 9
def find_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


candidate_lv1 = [
    "/kaggle/input/cnn-lstm-oct-24/run_0/run_0/model_0.pth",
    "../input/cnn-lstm-oct-24/run_0/run_0/model_0.pth",
]
candidate_lv2 = [
    "/kaggle/input/cnn-lstm-oct-24/run_0_c/run_0_b/model_lstm_0.pth",
    "../input/cnn-lstm-oct-24/run_0_c/run_0_b/model_lstm_0.pth",
]

lv1_path = find_first_existing(candidate_lv1)
lv2_path = find_first_existing(candidate_lv2)

lv1_model = ConvNextCNN_B_Feature().to(DEVICE)
lv2_model = CSFNet(input_len=config["feature_size"], lstm_size=config["lstm_size"]).to(
    DEVICE
)

if lv1_path is not None:
    lv1_model.load_state_dict(torch.load(lv1_path, map_location=DEVICE))
else:
    pass

if lv2_path is not None:
    lv2_model.load_state_dict(torch.load(lv2_path, map_location=DEVICE))
else:
    pass

lv1_model.eval()
lv2_model.eval()

print("lv1 weights:", lv1_path)
print("lv2 weights:", lv2_path)



## === cell 10
PIN = torch.cuda.is_available()

CPU_CNT = os.cpu_count() or 4

IMG_WORKERS = min(6, max(2, (CPU_CNT // 2)))
PL_WORKERS = 0  # patient-level is tiny; keep 0 to reduce overhead.

flat_dataset = FlatCSFImageDataset(items=flat_items, target_size=config["target_size"])


def _collate_images_uids(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    uids = [b[1] for b in batch]
    return imgs, uids


flat_loader = DataLoader(
    flat_dataset,
    batch_size=config["batch_size_image_level"],
    shuffle=False,
    num_workers=IMG_WORKERS,
    pin_memory=PIN,
    drop_last=False,
    persistent_workers=(IMG_WORKERS > 0),
    prefetch_factor=2 if IMG_WORKERS > 0 else None,
    collate_fn=_collate_images_uids,
)

uid_to_idx = {uid: i for i, uid in enumerate(study_id_list)}
n_uid = len(study_id_list)

counts = np.array(
    [uid_item_counts.get(uid, 0) for uid in study_id_list], dtype=np.int32
)
start = np.zeros(n_uid, dtype=np.int64)
np.cumsum(counts[:-1], out=start[1:])
total_items = int(counts.sum())

features_all = np.zeros(
    (total_items if total_items > 0 else 1, config["feature_size"]), dtype=np.float32
)
write_pos = start.copy()
end_pos = start + counts

pred_sum_arr = np.zeros((n_uid, 7), dtype=np.float32)
pred_cnt_arr = np.zeros(n_uid, dtype=np.int32)

feature_array_dict = {}
for i, uid in enumerate(study_id_list):
    n = int(counts[i])
    if n == 0:
        feature_array_dict[uid] = np.zeros(
            (1, config["feature_size"]), dtype=np.float32
        )
    else:
        feature_array_dict[uid] = features_all[start[i] : end_pos[i]]

inference_ctx = (
    torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
)

for images, uids in tqdm(
    flat_loader, desc="Stage 1: image-level inference (flat)", total=len(flat_loader)
):
    with inference_ctx():
        images = images.to(DEVICE, non_blocking=True)
        features, preds = lv1_model(images)
        preds = (
            preds.sigmoid().detach().cpu().numpy().astype(np.float32, copy=False)
        )  # [B,7]
        features_cpu = (
            features.detach().cpu().numpy().astype(np.float32, copy=False)
        )  # [B,1024]

    ui = np.fromiter((uid_to_idx[u] for u in uids), dtype=np.int32, count=len(uids))

    if len(ui):
        order = np.argsort(ui, kind="stable")
        ui_s = ui[order]
        feat_s = features_cpu[order]

        run_starts = np.r_[0, np.flatnonzero(ui_s[1:] != ui_s[:-1]) + 1]
        run_ends = np.r_[run_starts[1:], len(ui_s)]

        for rs, re in zip(run_starts, run_ends):
            u = int(ui_s[rs])
            n = int(re - rs)
            pos = int(write_pos[u])
            epos = int(end_pos[u])
            if pos >= epos:
                continue
            nwrite = n if pos + n <= epos else (epos - pos)
            if nwrite > 0:
                features_all[pos : pos + nwrite] = feat_s[rs : rs + nwrite]
                write_pos[u] = pos + nwrite

    np.add.at(pred_sum_arr, ui, preds)
    np.add.at(pred_cnt_arr, ui, 1)

submission_dict = {"row_id": [], "fractured": []}

for uid in study_id_list:
    ui0 = uid_to_idx[uid]
    if uid_item_counts.get(uid, 0) == 0 or pred_cnt_arr[ui0] == 0:
        for lvl in LEVELS:
            submission_dict["row_id"].append(f"{uid}_{lvl}")
            submission_dict["fractured"].append(0.5)
    else:
        mean_preds = pred_sum_arr[ui0] / float(pred_cnt_arr[ui0])
        for k, lvl in enumerate(LEVELS):
            submission_dict["row_id"].append(f"{uid}_{lvl}")
            submission_dict["fractured"].append(float(mean_preds[k]))

dataset2 = CSFInstanceDataset(
    feature_array_dict=feature_array_dict,
    study_id_list=study_id_list,
    seq_len=config["seq_len"],
)
generator2 = DataLoader(
    dataset=dataset2,
    batch_size=config["batch_size_patient_level"],
    shuffle=False,
    pin_memory=PIN,
    num_workers=PL_WORKERS,
    persistent_workers=False,
)

for features, list_uid in tqdm(
    generator2, total=len(generator2), desc="Stage 2: patient-level inference"
):
    with inference_ctx():
        features = features.to(DEVICE, non_blocking=True)
        preds = lv2_model(features).sigmoid().detach().cpu().numpy().reshape(-1)

    for j in range(len(list_uid)):
        submission_dict["row_id"].append(f"{list_uid[j]}_patient_overall")
        submission_dict["fractured"].append(float(preds[j]))

sub_df = pd.DataFrame.from_dict(submission_dict)

sub_df["fractured"] = sub_df["fractured"].astype(np.float32).clip(EPS, 1.0 - EPS)

required = test_df[["row_id"]].copy()
sub_df = required.merge(sub_df, on="row_id", how="left")

sub_df["fractured"] = (
    sub_df["fractured"].fillna(0.5).astype(np.float32).clip(EPS, 1.0 - EPS)
)

assert sub_df.shape[0] == test_df.shape[0]
assert list(sub_df.columns) == ["row_id", "fractured"]

sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print("Wrote submission.csv with rows:", len(sub_df))
