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
import warnings
import gc
from collections import defaultdict, OrderedDict

import numpy as np
import pandas as pd

import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, get_worker_info

import pydicom
from pydicom.errors import InvalidDicomError

from albumentations import Compose, CenterCrop

from torchvision.models.convnext import convnext_base

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

try:
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass


def seed_worker(worker_id: int):
    seed = SEED + worker_id
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass




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




## === cell 2
INPUT_ROOT = "../input/rsna-2022-cervical-spine-fracture-detection"
TEST_PATH = f"{INPUT_ROOT}/test_images"
TEST_CSV = f"{INPUT_ROOT}/test.csv"
SAMPLE_SUB = f"{INPUT_ROOT}/sample_submission.csv"


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

print("test_df shape:", test_df.shape)
print("num unique studies:", len(study_id_list))
print("TEST_PATH exists:", os.path.isdir(TEST_PATH))




## === cell 3
selected_image_dict = {}
num_slices_dict = {}

slice_file_map = {}  # uid -> dict[int, str] where value is filename (not full path)


def _build_slice_map_and_select(uid: str):
    dcmdir = os.path.join(TEST_PATH, uid)
    try:
        names = []
        with os.scandir(dcmdir) as it:
            for e in it:
                if e.is_file() and e.name.endswith(".dcm"):
                    names.append(e.name)
    except FileNotFoundError:
        return 0, [], {}

    if not names:
        return 0, [], {}

    numeric = {}
    for nm in names:
        base = nm[:-4]
        if base.isdigit():
            numeric[int(base)] = nm

    if numeric:
        n = max(numeric.keys())
        mp = numeric
    else:
        inst = {}
        max_inst = 0
        for nm in names:
            fp = os.path.join(dcmdir, nm)
            try:
                ds = pydicom.dcmread(
                    fp,
                    stop_before_pixels=True,
                    force=True,
                    specific_tags=["InstanceNumber"],
                )
                v = getattr(ds, "InstanceNumber", None)
                if v is not None:
                    iv = int(v)
                    inst[iv] = nm
                    if iv > max_inst:
                        max_inst = iv
            except Exception:
                pass
        if inst:
            n = max_inst
            mp = inst
        else:
            n = len(names)
            mp = {}

    middle_slice = int(n / 2)
    num_left_images = num_right_images = int(0.15 * n)
    idxs = list(np.arange(middle_slice - num_left_images, middle_slice, 1)) + list(
        np.arange(middle_slice + 1, middle_slice + num_right_images + 1, 1)
    )
    safe = []
    for idx in idxs:
        if 2 <= idx <= n - 1:
            safe.append(int(idx))

    return n, safe, mp


for uid in study_id_list:
    n, safe, mp = _build_slice_map_and_select(uid)
    num_slices_dict[uid] = n
    selected_image_dict[uid] = safe
    slice_file_map[uid] = mp

if len(study_id_list) > 0:
    print("example selected indices:", selected_image_dict[study_id_list[0]][:10])
    ex_uid = study_id_list[0]
    ex_mp = slice_file_map.get(ex_uid, {})
    if ex_mp:
        k0 = sorted(ex_mp.keys())[0]
        print("example slice map entry:", ex_uid, k0, "->", ex_mp[k0])




## === cell 4
_DCM_TAGS_FOR_WINDOW = [
    "RescaleSlope",
    "RescaleIntercept",
    "BitsStored",
    "PixelRepresentation",
    "SamplesPerPixel",
    "PhotometricInterpretation",
    "Rows",
    "Columns",
    "PlanarConfiguration",
    "TransferSyntaxUID",
]


def safe_dcmread_meta(path):
    try:
        return pydicom.dcmread(
            path,
            force=True,
            stop_before_pixels=True,
            specific_tags=_DCM_TAGS_FOR_WINDOW,
        )
    except (InvalidDicomError, FileNotFoundError, IsADirectoryError, PermissionError):
        return None


def safe_dcmread_pixels(path):
    try:
        return pydicom.dcmread(
            path,
            force=True,
            stop_before_pixels=False,
            specific_tags=_DCM_TAGS_FOR_WINDOW,
        )
    except (InvalidDicomError, FileNotFoundError, IsADirectoryError, PermissionError):
        return None


def window(data, WL=400, WW=1800):
    if data is None:
        return np.full((512, 512), 128, dtype=np.uint8)

    try:
        slope = float(getattr(data, "RescaleSlope", 1.0))
        intercept = float(getattr(data, "RescaleIntercept", 0.0))

        img = data.pixel_array.astype(np.float32)
        del data

        img = img * slope + intercept

        upper, lower = WL + WW // 2, WL - WW // 2
        X = np.clip(img, lower, upper)
        X = X - np.min(X)
        mx = np.max(X)
        if mx > 0:
            X = X / mx
        X = (X * 255.0).astype("uint8")
        return X
    except Exception:
        return np.full((512, 512), 128, dtype=np.uint8)


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))




## === cell 5
mean = np.array([0.456, 0.456, 0.456])
std = np.array([0.224, 0.224, 0.224])

INFERENCE_TRANSFORM = Compose([CenterCrop(config["crop_size"], config["crop_size"])])


class _LRUCache:
    def __init__(self, max_items: int = 4096):
        self.max_items = int(max_items)
        self._d = OrderedDict()

    def get(self, key):
        v = self._d.get(key, None)
        if v is not None:
            self._d.move_to_end(key)
        return v

    def set(self, key, value):
        self._d[key] = value
        self._d.move_to_end(key)
        if len(self._d) > self.max_items:
            self._d.popitem(last=False)


_SLICE_IMG_CACHE = _LRUCache(max_items=4096)


class CSFAllSlicesDataset(Dataset):
    def __init__(self, items, target_size, crop_size, slice_file_map):
        self.items = items  # list of (uid, idx)
        self.target_size = target_size
        self.crop_size = crop_size
        self.slice_file_map = slice_file_map

        self._path_cache = {}

        self._local_cache = None

    def __len__(self):
        return len(self.items)

    def _idx_to_path(self, uid: str, idx: int):
        key = (uid, idx)
        v = self._path_cache.get(key, None)
        if v is not None or key in self._path_cache:
            return v

        mp = self.slice_file_map.get(uid)
        if mp:
            nm = mp.get(idx)
            if nm is not None:
                p = os.path.join(TEST_PATH, uid, nm)
                self._path_cache[key] = p
                return p
            self._path_cache[key] = None
            return None

        p = os.path.join(TEST_PATH, uid, f"{idx}.dcm")
        self._path_cache[key] = p
        return p

    def _get_windowed_slice(self, uid: str, idx: int):
        key = (uid, idx)

        cached = _SLICE_IMG_CACHE.get(key)
        if cached is not None:
            return cached

        p = self._idx_to_path(uid, idx)
        if p is None:
            img = np.full((512, 512), 128, dtype=np.uint8)
        else:
            img = window(safe_dcmread_pixels(p))

        _SLICE_IMG_CACHE.set(key, img)
        return img

    def __getitem__(self, index):
        uid, idx = self.items[index]
        uid = str(uid)
        idx = int(idx)

        img1 = self._get_windowed_slice(uid, idx - 1)
        img2 = self._get_windowed_slice(uid, idx)
        img3 = self._get_windowed_slice(uid, idx + 1)

        stacked_img = np.stack((img1, img2, img3), axis=-1)  # H,W,3
        stacked_img = cv2.resize(stacked_img, (self.target_size, self.target_size))
        stacked_img = INFERENCE_TRANSFORM(image=stacked_img)["image"]
        X = img2tensor((stacked_img / 255.0 - mean) / std)
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

        X = torch.tensor(x, dtype=torch.float32)
        return X, uid




## === cell 6
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lv1_model = ConvNextCNN_B_Feature().to(device).eval()
lv2_model = (
    CSFNet(input_len=config["feature_size"], lstm_size=config["lstm_size"])
    .to(device)
    .eval()
)

if device.type == "cuda":
    lv1_model = lv1_model.to(memory_format=torch.channels_last)

ckpt1 = "../input/cnn-lstm-oct-21/run_0/run_0/model_0.pth"
ckpt2 = "../input/cnn-lstm-oct-21/run_0/run_0/model_lstm_0.pth"

loaded_any = False
if os.path.exists(ckpt1):
    lv1_model.load_state_dict(torch.load(ckpt1, map_location=device))
    loaded_any = True
else:
    print("WARNING: missing checkpoint:", ckpt1)

if os.path.exists(ckpt2):
    lv2_model.load_state_dict(torch.load(ckpt2, map_location=device))
    loaded_any = True
else:
    print("WARNING: missing checkpoint:", ckpt2)

print("device:", device, "| any checkpoints loaded:", loaded_any)




## === cell 8
submission_dict = {"row_id": [], "fractured": []}
feature_array_dict = {}

use_cuda = torch.cuda.is_available()

all_items = []
per_uid_count = {}
for uid in study_id_list:
    image_list = selected_image_dict.get(uid, [])
    per_uid_count[uid] = len(image_list)
    for idx in image_list:
        all_items.append((uid, int(idx)))

for uid in study_id_list:
    nsel = per_uid_count.get(uid, 0)
    if nsel == 0:
        feature_array_dict[uid] = np.zeros(
            (1, config["feature_size"]), dtype=np.float32
        )
    else:
        feature_array_dict[uid] = np.empty(
            (nsel, config["feature_size"]), dtype=np.float32
        )

uid_to_int = {uid: i for i, uid in enumerate(study_id_list)}
n_uid = len(study_id_list)
preds_sum_arr = np.zeros((n_uid, 7), dtype=np.float64)
preds_count_arr = np.zeros((n_uid,), dtype=np.int32)

item_uidx = np.empty((len(all_items),), dtype=np.int32)
item_pos = np.empty((len(all_items),), dtype=np.int32)
_tmp_offsets = np.zeros((n_uid,), dtype=np.int32)
for t, (uid, _) in enumerate(all_items):
    ui = uid_to_int[str(uid)]
    item_uidx[t] = ui
    item_pos[t] = _tmp_offsets[ui]
    _tmp_offsets[ui] += 1
del _tmp_offsets

uid_offsets = np.zeros((n_uid + 1,), dtype=np.int64)
for i, uid in enumerate(study_id_list):
    uid_offsets[i + 1] = uid_offsets[i] + int(per_uid_count.get(uid, 0))
total_items = int(uid_offsets[-1])
flat_features = np.empty(
    (max(total_items, 1), config["feature_size"]), dtype=np.float32
)

item_flat_index = uid_offsets[item_uidx] + item_pos

cpu_cnt = os.cpu_count() or 2
if cpu_cnt >= 16:
    num_workers = 8
elif cpu_cnt >= 8:
    num_workers = 4
elif cpu_cnt >= 4:
    num_workers = 2
else:
    num_workers = 0

prefetch_factor = 4 if num_workers > 0 else None
g = torch.Generator()
g.manual_seed(SEED)

if len(all_items) > 0:
    ds_all = CSFAllSlicesDataset(
        items=all_items,
        target_size=config["target_size"],
        crop_size=config["crop_size"],
        slice_file_map=slice_file_map,
    )
    dl_all = DataLoader(
        ds_all,
        batch_size=config["batch_size_image_level"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        drop_last=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor,
        worker_init_fn=seed_worker,
        generator=g,
    )

    with torch.inference_mode():
        start_idx = 0
        for images, uids in tqdm(
            dl_all, desc="Stage1 (image-level)", total=len(dl_all)
        ):
            bs = images.size(0)

            if device.type == "cuda":
                images = images.to(device, non_blocking=True).to(
                    memory_format=torch.channels_last
                )
            else:
                images = images.to(device, non_blocking=True)

            features, preds = lv1_model(images)
            feats_np = features.detach().cpu().numpy()
            preds_np = preds.sigmoid().detach().cpu().double().numpy()  # (bs,7)

            batch_uidx = item_uidx[start_idx : start_idx + bs]
            batch_flat = item_flat_index[start_idx : start_idx + bs]
            start_idx += bs

            flat_features[batch_flat] = feats_np

            np.add.at(preds_sum_arr, batch_uidx, preds_np)
            np.add.at(preds_count_arr, batch_uidx, 1)

    del ds_all, dl_all
    gc.collect()
else:
    tqdm([], desc="Stage1 (image-level)", total=0)

if total_items > 0:
    for i, uid in enumerate(study_id_list):
        nsel = int(per_uid_count.get(uid, 0))
        if nsel > 0:
            s = int(uid_offsets[i])
            e = s + nsel
            feature_array_dict[uid][:] = flat_features[s:e]

for uid in study_id_list:
    ui = uid_to_int[uid]
    if per_uid_count.get(uid, 0) == 0:
        mean_preds = np.full((7,), 0.5, dtype=np.float32)
    else:
        cnt = int(preds_count_arr[ui])
        mean_preds = (preds_sum_arr[ui] / max(cnt, 1)).astype(np.float32)

    for k, level in enumerate(["C1", "C2", "C3", "C4", "C5", "C6", "C7"]):
        submission_dict["row_id"].append(f"{uid}_{level}")
        submission_dict["fractured"].append(float(mean_preds[k]))




## === cell 9
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
    num_workers=0,  # patient-level is tiny; keep simple and deterministic
    persistent_workers=False,
)

with torch.inference_mode():
    for features, list_uid in tqdm(
        generator, total=len(generator), desc="Stage2 (patient-level)"
    ):
        features = features.to(device, non_blocking=True)
        preds = lv2_model(features).sigmoid().detach().cpu().numpy().reshape(-1)

        for j in range(len(list_uid)):
            submission_dict["row_id"].append(f"{list_uid[j]}_patient_overall")
            submission_dict["fractured"].append(float(preds[j]))




## === cell 10
sub_df = pd.DataFrame.from_dict(submission_dict)

out = test_df[["row_id"]].merge(sub_df, on="row_id", how="left")

out["fractured"] = out["fractured"].astype(np.float32)
out["fractured"] = out["fractured"].fillna(0.5).clip(1e-6, 1 - 1e-6)

assert out.shape[0] == test_df.shape[0], "Submission row count mismatch"
print(out.head())
print("missing preds:", int(out["fractured"].isna().sum()))

out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", out.shape)
