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
import time
import glob
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd

import cv2
from tqdm import tqdm

import torch
import torch.nn as nn

from albumentations import Compose, CenterCrop

import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut

from torchvision.models.convnext import convnext_base
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

torch.manual_seed(0)
np.random.seed(0)

DATA_ROOT = "../input/rsna-2022-cervical-spine-fracture-detection"
TEST_PATH = f"{DATA_ROOT}/test_images"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True




## === cell 1
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

LEVELS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]




## === cell 2
def load_df_test():
    df_test = pd.read_csv(f"{DATA_ROOT}/test.csv")
    expected = {"row_id", "StudyInstanceUID", "prediction_type"}
    missing = expected - set(df_test.columns)
    if missing:
        raise ValueError(f"test.csv missing columns: {missing}")
    return df_test


test_df = load_df_test()
study_id_list = list(test_df.StudyInstanceUID.unique())

print("Num test rows:", len(test_df))
print("Num unique studies:", len(study_id_list))




## === cell 3
def safe_dcmread(path: str):
    try:
        return pydicom.dcmread(path, force=True)
    except Exception:
        return None


def dcm_to_uint8(ds, WL=400, WW=1800):
    """
    Robust decode without relying on external JPEG plugins.
    We try pixel_array; if it fails, return None so caller can fallback.
    """
    try:
        arr = ds.pixel_array  # may fail if compressed decoder missing
    except Exception:
        return None

    try:
        arr = apply_modality_lut(arr, ds)
    except Exception:
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr.astype(np.float32) * slope + intercept

    arr = arr.astype(np.float32)
    upper, lower = WL + WW // 2, WL - WW // 2
    arr = np.clip(arr, lower, upper)
    arr = arr - arr.min()
    mx = arr.max()
    if mx > 0:
        arr = arr / mx
    arr = (arr * 255.0).astype(np.uint8)
    return arr


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))




## === cell 4
mean = np.array([0.456, 0.456, 0.456], dtype=np.float32)
std = np.array([0.224, 0.224, 0.224], dtype=np.float32)


def get_sorted_slice_numbers(uid: str):
    d = os.path.join(TEST_PATH, uid)
    try:
        it = os.scandir(d)
    except FileNotFoundError:
        return []
    nums = []
    with it:
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


selected_image_dict = {}
slice_numbers_dict = {}
triplets_slice_no_dict = {}  # uid -> list of (s_m1, s0, s_p1) slice numbers (ints)

for uid in study_id_list:
    nums = get_sorted_slice_numbers(uid)
    slice_numbers_dict[uid] = nums
    if len(nums) == 0:
        selected_image_dict[uid] = []
        triplets_slice_no_dict[uid] = []
        continue

    middle_idx = len(nums) // 2
    num_left = num_right = max(1, int(0.15 * len(nums)))
    left = list(range(max(0, middle_idx - num_left), middle_idx))
    right = list(range(middle_idx + 1, min(len(nums), middle_idx + num_right + 1)))
    sel = left + right
    selected_image_dict[uid] = sel

    triplets = []
    last_pos = len(nums) - 1
    for center_pos in sel:
        pos_m1 = center_pos - 1
        if pos_m1 < 0:
            pos_m1 = 0
        pos_p1 = center_pos + 1
        if pos_p1 > last_pos:
            pos_p1 = last_pos
        triplets.append((nums[pos_m1], nums[center_pos], nums[pos_p1]))
    triplets_slice_no_dict[uid] = triplets

print("Example UID:", study_id_list[0])
print("Num slices:", len(slice_numbers_dict[study_id_list[0]]))
print("Selected indices (first 10):", selected_image_dict[study_id_list[0]][:10])




## === cell 5
class CSFImageDataset(Dataset):
    def __init__(self, uid, selected_indices, target_size, crop_size):
        self.uid = uid
        self.selected_indices = selected_indices
        self.target_size = target_size
        self.crop_size = crop_size
        self.slice_numbers = slice_numbers_dict[uid]

        self.inference_transform = Compose([CenterCrop(self.crop_size, self.crop_size)])

        self._path = os.path.join(TEST_PATH, self.uid)

        self._slice_cache_uint8 = {}  # key: slice_number(int) -> uint8 2D array or None

    def __len__(self):
        return len(self.selected_indices)

    def _get_slice_uint8(self, slice_no: int):
        if slice_no in self._slice_cache_uint8:
            return self._slice_cache_uint8[slice_no]

        ds = safe_dcmread(os.path.join(self._path, f"{slice_no}.dcm"))
        if ds is None:
            self._slice_cache_uint8[slice_no] = None
            return None

        try:
            img = dcm_to_uint8(ds)
        finally:
            try:
                ds.fileobj_type = None  # no-op hint
            except Exception:
                pass
            try:
                ds._fp = None
            except Exception:
                pass
            try:
                ds.close()
            except Exception:
                pass

        self._slice_cache_uint8[slice_no] = img
        return img

    def __getitem__(self, index):
        center_pos = self.selected_indices[index]
        pos_m1 = max(0, center_pos - 1)
        pos_p1 = min(len(self.slice_numbers) - 1, center_pos + 1)

        s_m1 = self.slice_numbers[pos_m1]
        s_0 = self.slice_numbers[center_pos]
        s_p1 = self.slice_numbers[pos_p1]

        imgs = [
            self._get_slice_uint8(s_m1),
            self._get_slice_uint8(s_0),
            self._get_slice_uint8(s_p1),
        ]

        if any(im is None for im in imgs):
            stacked = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        else:
            stacked = np.stack(imgs, axis=-1)
            stacked = cv2.resize(
                stacked,
                (self.target_size, self.target_size),
                interpolation=cv2.INTER_LINEAR,
            )

        out = self.inference_transform(image=stacked)["image"]
        out = (out.astype(np.float32) / 255.0 - mean) / std
        return img2tensor(out, dtype=np.float32)




## === cell 6
class CSFInstanceDataset(Dataset):
    def __init__(self, feature_array_dict, study_id_list, seq_len, feature_size):
        self.feature_array_dict = feature_array_dict
        self.study_id_list = study_id_list
        self.seq_len = seq_len
        self.feature_size = feature_size

    def __len__(self):
        return len(self.study_id_list)

    def __getitem__(self, index):
        uid = self.study_id_list[index]

        feature_array = self.feature_array_dict.get(uid, None)
        if feature_array is None or len(feature_array) == 0:
            feature_array = np.zeros((1, self.feature_size), dtype=np.float32)

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




## === cell 7
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




## === cell 8
lv1_model = ConvNextCNN_B_Feature().to(device).eval()
lv2_model = (
    CSFNet(input_len=config["feature_size"], lstm_size=config["lstm_size"])
    .to(device)
    .eval()
)

lv1_ckpt = "../input/cnn-lstm-exp-6/run_1/run_1/model_0.pth"
lv2_ckpt = "../input/cnn-lstm-exp-6/model_lstm_0.pth"

loaded_any = False
if os.path.exists(lv1_ckpt):
    lv1_model.load_state_dict(torch.load(lv1_ckpt, map_location=device))
    loaded_any = True
else:
    print(f"WARNING: missing lv1 checkpoint: {lv1_ckpt}")

if os.path.exists(lv2_ckpt):
    lv2_model.load_state_dict(torch.load(lv2_ckpt, map_location=device))
    loaded_any = True
else:
    print(f"WARNING: missing lv2 checkpoint: {lv2_ckpt}")

if not loaded_any:
    print(
        "WARNING: No checkpoints loaded; will generate a valid but low-quality submission."
    )




## === cell 9
submission_dict = {"row_id": [], "fractured": []}
feature_array_dict = {}

lv1_model.eval()
lv2_model.eval()


class _LRUCache:
    def __init__(self, max_items: int = 256):
        self.max_items = int(max_items)
        self._d = OrderedDict()

    def get(self, k):
        try:
            v = self._d.pop(k)
            self._d[k] = v
            return v
        except KeyError:
            return None

    def put(self, k, v):
        if k in self._d:
            try:
                self._d.pop(k)
            except KeyError:
                pass
        self._d[k] = v
        if len(self._d) > self.max_items:
            self._d.popitem(last=False)


def _load_slice_uint8_cached(uid: str, slice_no: int, cache: _LRUCache):
    key = (uid, int(slice_no))
    v = cache.get(key)
    if v is not None:
        return v
    ds = safe_dcmread(os.path.join(TEST_PATH, uid, f"{slice_no}.dcm"))
    if ds is None:
        cache.put(key, None)
        return None
    try:
        img = dcm_to_uint8(ds)
    finally:
        try:
            ds.fileobj_type = None
        except Exception:
            pass
        try:
            ds._fp = None
        except Exception:
            pass
        try:
            ds.close()
        except Exception:
            pass
    cache.put(key, img)
    return img


center_crop = CenterCrop(config["crop_size"], config["crop_size"])
bs = config["batch_size_image_level"]
pin = torch.cuda.is_available()

global_slice_cache = _LRUCache(max_items=384)

for uid in tqdm(study_id_list, desc="Stage1 per-study"):
    triplets = triplets_slice_no_dict[uid]
    n = len(triplets)
    if n == 0:
        feature_array_dict[uid] = np.zeros(
            (1, config["feature_size"]), dtype=np.float32
        )
        mean_preds = np.full((7,), 0.01, dtype=np.float32)
    else:
        feature_array = np.zeros((n, config["feature_size"]), dtype=np.float32)

        pred_sum = np.zeros((7,), dtype=np.float64)
        pred_count = 0

        for start in range(0, n, bs):
            end = min(start + bs, n)
            cur = end - start

            batch_np = np.empty(
                (cur, config["crop_size"], config["crop_size"], 3), dtype=np.float32
            )

            for i in range(cur):
                s_m1, s0, s_p1 = triplets[start + i]
                imgs = (
                    _load_slice_uint8_cached(uid, s_m1, global_slice_cache),
                    _load_slice_uint8_cached(uid, s0, global_slice_cache),
                    _load_slice_uint8_cached(uid, s_p1, global_slice_cache),
                )

                if imgs[0] is None or imgs[1] is None or imgs[2] is None:
                    stacked = np.zeros(
                        (config["target_size"], config["target_size"], 3),
                        dtype=np.uint8,
                    )
                else:
                    stacked = np.stack(imgs, axis=-1)
                    stacked = cv2.resize(
                        stacked,
                        (config["target_size"], config["target_size"]),
                        interpolation=cv2.INTER_LINEAR,
                    )

                cropped = center_crop(image=stacked)["image"]
                x = (cropped.astype(np.float32) / 255.0 - mean) / std  # HWC
                batch_np[i] = x

            batch_t = torch.from_numpy(batch_np).permute(0, 3, 1, 2).contiguous()
            if pin:
                batch_t = batch_t.pin_memory()

            with torch.inference_mode():
                images = batch_t.to(device, non_blocking=True)
                features, preds = lv1_model(images)
                probs = preds.sigmoid()

            feature_array[start:end] = features.detach().cpu().numpy()
            pred_sum += probs.detach().cpu().sum(dim=0).numpy()
            pred_count += cur

        feature_array_dict[uid] = feature_array
        mean_preds = (pred_sum / max(1, pred_count)).astype(np.float32)

    submission_dict["row_id"].extend([f"{uid}_{lvl}" for lvl in LEVELS])
    submission_dict["fractured"].extend([float(x) for x in mean_preds])

for uid in study_id_list:
    if uid not in feature_array_dict:
        feature_array_dict[uid] = np.zeros(
            (1, config["feature_size"]), dtype=np.float32
        )




## === cell 10
dataset2 = CSFInstanceDataset(
    feature_array_dict=feature_array_dict,
    study_id_list=study_id_list,
    seq_len=config["seq_len"],
    feature_size=config["feature_size"],
)

generator2 = DataLoader(
    dataset=dataset2,
    batch_size=config["batch_size_patient_level"],
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
    num_workers=0,
    drop_last=False,
    persistent_workers=False,
)

for features, list_uid in tqdm(
    generator2, total=len(generator2), desc="Stage2 patient_overall"
):
    with torch.inference_mode():
        features = features.to(device, non_blocking=True)
        preds = lv2_model(features).sigmoid().detach().cpu().numpy().reshape(-1)

    submission_dict["row_id"].extend([f"{u}_patient_overall" for u in list_uid])
    submission_dict["fractured"].extend([float(p) for p in preds])




## === cell 11
sub_df = pd.DataFrame.from_dict(submission_dict)

sub_df = test_df[["row_id"]].merge(sub_df, on="row_id", how="left")

sub_df["fractured"] = sub_df["fractured"].astype(np.float32)
sub_df["fractured"] = sub_df["fractured"].fillna(0.01).clip(1e-6, 1 - 1e-6)

assert len(sub_df) == len(test_df)
assert list(sub_df.columns) == ["row_id", "fractured"]

sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print("Wrote submission.csv with shape:", sub_df.shape)
