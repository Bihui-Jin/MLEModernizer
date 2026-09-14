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
import os, glob, sys, time, warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import pydicom
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from albumentations import Compose, CenterCrop

    _HAS_ALB = True
except Exception:
    _HAS_ALB = False

from torchvision.models.convnext import convnext_base



## === cell 1
BASE_PATH = "/kaggle/input/rsna-2022-cervical-spine-fracture-detection"
TEST_CSV_PATH = f"{BASE_PATH}/test.csv"
SAMPLE_SUB_PATH = f"{BASE_PATH}/sample_submission.csv"
TEST_IMG_PATH = f"{BASE_PATH}/test_images"

assert os.path.exists(TEST_CSV_PATH), f"Missing {TEST_CSV_PATH}"
assert os.path.exists(SAMPLE_SUB_PATH), f"Missing {SAMPLE_SUB_PATH}"
assert os.path.exists(TEST_IMG_PATH), f"Missing {TEST_IMG_PATH}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # inference with fixed shapes benefits
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

torch.manual_seed(0)
np.random.seed(0)



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
def load_df_test():
    df_test = pd.read_csv(TEST_CSV_PATH)
    if len(df_test) < 10 and df_test.iloc[0].row_id == "1.2.826.0.1.3680043.10197_C1":
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
study_id_list = list(test_df.StudyInstanceUID.unique())
print("Num studies:", len(study_id_list))
print("Example study:", study_id_list[0])



## === cell 5
_SCANDIR_CACHE = {}


def _list_dicom_paths(uid: str):
    cached = _SCANDIR_CACHE.get(uid)
    if cached is not None:
        return cached

    folder = os.path.join(TEST_IMG_PATH, uid)
    try:
        entries = [
            e for e in os.scandir(folder) if e.is_file() and e.name.endswith(".dcm")
        ]
    except FileNotFoundError:
        _SCANDIR_CACHE[uid] = []
        return []

    nums = []
    paths = []
    all_numeric = True
    for e in entries:
        bn = e.name[:-4]
        if bn.isdigit():
            nums.append(int(bn))
            paths.append(e.path)
        else:
            all_numeric = False
            break

    if all_numeric:
        order = np.argsort(np.asarray(nums, dtype=np.int32), kind="mergesort")
        out = [paths[i] for i in order]
    else:
        out = sorted((e.path for e in entries))

    _SCANDIR_CACHE[uid] = out
    return out


selected_image_dict = {}
dicom_path_dict = {}
for uid in study_id_list:
    dcm_paths = _list_dicom_paths(uid)
    dicom_path_dict[uid] = dcm_paths
    n = len(dcm_paths)
    if n == 0:
        selected_image_dict[uid] = []
        continue
    middle = n // 2
    num_left = num_right = max(1, int(0.15 * n))
    idxs = list(range(max(1, middle - num_left), middle)) + list(
        range(middle + 1, min(n - 1, middle + num_right + 1))
    )
    idxs = [i for i in idxs if 1 <= i <= n - 2]
    if len(idxs) == 0:
        idxs = [min(max(1, middle), n - 2)]
    selected_image_dict[uid] = idxs

print("Selected indices example:", selected_image_dict[study_id_list[0]][:10])




## === cell 6
def window_from_hu(img_hu: np.ndarray, WL=400, WW=1800):
    upper, lower = WL + WW // 2, WL - WW // 2
    x = np.clip(img_hu, lower, upper)
    x = x - x.min()
    mx = x.max()
    if mx > 0:
        x = x / mx
    x = (x * 255.0).astype("uint8")
    return x


def window(ds, WL=400, WW=1800):
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    img = ds.pixel_array.astype(np.float32)
    img = img * slope + intercept
    return window_from_hu(img, WL=WL, WW=WW)


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    arr = img.astype(dtype, copy=False)
    return torch.from_numpy(arr)


def safe_dcmread(path: str):
    try:
        return pydicom.dcmread(path, force=True)
    except Exception:
        return None


def safe_get_pixel(ds):
    try:
        return ds.pixel_array
    except Exception:
        return None




## === cell 7
mean = np.array([0.456, 0.456, 0.456], dtype=np.float32)
std = np.array([0.224, 0.224, 0.224], dtype=np.float32)


def center_crop_numpy(img, crop_size: int):
    h, w = img.shape[:2]
    ch = cw = crop_size
    if h < ch or w < cw:
        pad_h = max(0, ch - h)
        pad_w = max(0, cw - w)
        img = np.pad(
            img,
            (
                (pad_h // 2, pad_h - pad_h // 2),
                (pad_w // 2, pad_w - pad_w // 2),
                (0, 0),
            ),
            mode="constant",
        )
        h, w = img.shape[:2]
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    return img[y0 : y0 + ch, x0 : x0 + cw]


try:
    cv2.setNumThreads(0)
except Exception:
    pass

_DICOM_META_TAGS = [
    "RescaleSlope",
    "RescaleIntercept",
    "Rows",
    "Columns",
    "BitsAllocated",
    "PixelRepresentation",
    "SamplesPerPixel",
    "PhotometricInterpretation",
    "TransferSyntaxUID",
]


def _fast_read_windowed_uint8(path: str):
    try:
        ds = pydicom.dcmread(
            path,
            stop_before_pixels=False,
            force=True,
            specific_tags=["PixelData"] + _DICOM_META_TAGS,
        )
    except Exception:
        return np.zeros((512, 512), dtype="uint8")

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    rows = int(getattr(ds, "Rows", 512) or 512)
    cols = int(getattr(ds, "Columns", 512) or 512)

    try:
        spp = int(getattr(ds, "SamplesPerPixel", 1) or 1)
        bits = int(getattr(ds, "BitsAllocated", 16) or 16)
        pixel_repr = int(getattr(ds, "PixelRepresentation", 0) or 0)
        tsuid = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", None)
        is_compressed = (
            bool(getattr(tsuid, "is_compressed", False)) if tsuid is not None else False
        )

        if (
            (not is_compressed)
            and spp == 1
            and bits in (16,)
            and hasattr(ds, "PixelData")
        ):
            pix = ds.PixelData
            if pix is None:
                return np.zeros((rows, cols), dtype="uint8")
            dtype = np.int16 if pixel_repr == 1 else np.uint16
            arr = np.frombuffer(pix, dtype=dtype)
            if arr.size == rows * cols:
                img = arr.reshape(rows, cols).astype(np.float32)
                hu = img * slope + intercept
                return window_from_hu(hu)
    except Exception:
        pass

    try:
        px = ds.pixel_array
        if px is None:
            return np.zeros((rows, cols), dtype="uint8")
        hu = px.astype(np.float32) * slope + intercept
        return window_from_hu(hu)
    except Exception:
        return np.zeros((rows, cols), dtype="uint8")


from collections import OrderedDict

_GLOBAL_LRU = OrderedDict()
_GLOBAL_LRU_MAX = 2048  # slightly smaller to reduce memory pressure / eviction cost


def _get_windowed_lru(path: str):
    x = _GLOBAL_LRU.get(path, None)
    if x is not None:
        _GLOBAL_LRU.move_to_end(path)
        return x
    x = _fast_read_windowed_uint8(path)
    _GLOBAL_LRU[path] = x
    if len(_GLOBAL_LRU) > _GLOBAL_LRU_MAX:
        _GLOBAL_LRU.popitem(last=False)
    return x


class CSFImageDataset(Dataset):
    def __init__(self, uid, image_indices, target_size, crop_size):
        self.uid = uid
        self.image_indices = image_indices
        self.target_size = target_size
        self.crop_size = crop_size

        self._crop = None
        if _HAS_ALB:
            self._crop = Compose([CenterCrop(self.crop_size, self.crop_size)])

    def __len__(self):
        return len(self.image_indices)

    def __getitem__(self, index):
        idx = self.image_indices[index]
        dcm_paths = dicom_path_dict[self.uid]
        paths = (dcm_paths[idx - 1], dcm_paths[idx], dcm_paths[idx + 1])

        imgs = [_get_windowed_lru(p) for p in paths]

        stacked = np.stack(imgs, axis=-1)  # H,W,3
        stacked = cv2.resize(
            stacked,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_LINEAR,
        )

        if self._crop is not None:
            stacked = self._crop(image=stacked)["image"]
        else:
            stacked = center_crop_numpy(stacked, self.crop_size)

        x = (stacked.astype(np.float32) / 255.0 - mean) / std
        x = img2tensor(x)
        return x




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
            feature_array = np.zeros(
                (self.seq_len, config["feature_size"]), dtype=np.float32
            )

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




## === cell 10
lv1_model = ConvNextCNN_B_Feature().to(device).eval()
lv2_model = (
    CSFNet(input_len=config["feature_size"], lstm_size=config["lstm_size"])
    .to(device)
    .eval()
)

candidate_paths = [
    "/kaggle/input/cnn-lstm-oct-21/run_1/run_1/model_1.pth",
    "/kaggle/input/cnn-lstm-oct-21/run_3/run_3/model_lstm_3.pth",
    "../input/cnn-lstm-oct-21/run_1/run_1/model_1.pth",
    "../input/cnn-lstm-oct-21/run_3/run_3/model_lstm_3.pth",
]
lv1_path = (
    candidate_paths[0] if os.path.exists(candidate_paths[0]) else candidate_paths[2]
)
lv2_path = (
    candidate_paths[1] if os.path.exists(candidate_paths[1]) else candidate_paths[3]
)

loaded_any = False
if os.path.exists(lv1_path):
    try:
        lv1_model.load_state_dict(torch.load(lv1_path, map_location=device))
        loaded_any = True
        print("Loaded lv1 weights:", lv1_path)
    except Exception as e:
        print("Could not load lv1 weights:", e)

if os.path.exists(lv2_path):
    try:
        lv2_model.load_state_dict(torch.load(lv2_path, map_location=device))
        loaded_any = True
        print("Loaded lv2 weights:", lv2_path)
    except Exception as e:
        print("Could not load lv2 weights:", e)

if not loaded_any:
    print(
        "Warning: pretrained weights not found; using deterministic random init (baseline submission)."
    )




## === cell 11
class CSFImageFlatDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples  # list of (uid, idx, uid_idx)
        self._crop = (
            Compose([CenterCrop(config["crop_size"], config["crop_size"])])
            if _HAS_ALB
            else None
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        uid, idx, uid_idx = self.samples[i]
        dcm_paths = dicom_path_dict[uid]
        p0, p1, p2 = dcm_paths[idx - 1], dcm_paths[idx], dcm_paths[idx + 1]
        imgs = (_get_windowed_lru(p0), _get_windowed_lru(p1), _get_windowed_lru(p2))
        stacked = np.stack(imgs, axis=-1)  # H,W,3 uint8
        stacked = cv2.resize(
            stacked,
            (config["target_size"], config["target_size"]),
            interpolation=cv2.INTER_LINEAR,
        )
        if self._crop is not None:
            stacked = self._crop(image=stacked)["image"]
        else:
            stacked = center_crop_numpy(stacked, config["crop_size"])
        x = (stacked.astype(np.float32) / 255.0 - mean) / std  # H,W,3
        x = np.transpose(x, (2, 0, 1))  # CHW
        return torch.from_numpy(x), uid_idx, idx


def _worker_init_fn(_worker_id: int):
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass


def _infer_stage1_flat(study_id_list):
    submission_dict = {"row_id": [], "fractured": []}
    feature_array_dict = {}

    uid_to_idx = {uid: i for i, uid in enumerate(study_id_list)}
    idx_to_uid = list(study_id_list)

    feat_store = {}
    pred_sum = {}
    pred_cnt = {}

    samples = []
    for uid in study_id_list:
        image_indices = selected_image_dict.get(uid, [])
        if len(image_indices) == 0:
            feature_array_dict[uid] = np.zeros(
                (config["seq_len"], config["feature_size"]), dtype=np.float32
            )
            for k in ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]:
                submission_dict["row_id"].append(f"{uid}_{k}")
                submission_dict["fractured"].append(0.5)
            continue

        feat_store[uid] = np.zeros(
            (len(image_indices), config["feature_size"]), dtype=np.float32
        )
        pred_sum[uid] = np.zeros((7,), dtype=np.float64)
        pred_cnt[uid] = 0

        uidx = uid_to_idx[uid]
        for pos, idx in enumerate(image_indices):
            samples.append((uid, idx, uidx))

    if len(samples) == 0:
        return submission_dict, feature_array_dict

    dataset = CSFImageFlatDataset(samples=samples)

    use_cuda = torch.cuda.is_available()
    cpu = os.cpu_count() or 2
    num_workers = min(8, max(0, cpu - 1))

    loader_kwargs = dict(
        batch_size=config["batch_size_image_level"],
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4
    loader = DataLoader(dataset, **loader_kwargs)

    selected_pos_map = {
        uid: {idx: pos for pos, idx in enumerate(selected_image_dict.get(uid, []))}
        for uid in study_id_list
    }

    for images, uid_idx_t, idx_t in tqdm(
        loader, desc="Stage1 batched", total=len(loader)
    ):
        with torch.inference_mode():
            images = images.to(device, non_blocking=True)
            features, preds = lv1_model(images)
            preds = torch.sigmoid(preds)

        features_np = features.detach().cpu().numpy()
        preds_np = preds.detach().cpu().numpy()
        uid_idx_np = uid_idx_t.numpy().astype(np.int64, copy=False)
        idx_np = idx_t.numpy().astype(np.int64, copy=False)

        order = np.argsort(uid_idx_np, kind="mergesort")
        uid_sorted = uid_idx_np[order]
        idx_sorted = idx_np[order]
        feat_sorted = features_np[order]
        pred_sorted = preds_np[order]

        if uid_sorted.size == 0:
            continue
        boundaries = np.nonzero(uid_sorted[1:] != uid_sorted[:-1])[0] + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [uid_sorted.size]))

        for s, e in zip(starts, ends):
            uu = int(uid_sorted[s])
            uid = idx_to_uid[uu]
            mp = selected_pos_map[uid]
            pos_list = [mp[int(v)] for v in idx_sorted[s:e]]
            feat_store[uid][pos_list] = feat_sorted[s:e]
            pred_sum[uid] += pred_sorted[s:e].sum(axis=0, dtype=np.float64)
            pred_cnt[uid] += int(e - s)

    for uid, feats in feat_store.items():
        feature_array_dict[uid] = feats
        if pred_cnt[uid] == 0:
            mean_preds = np.full((7,), 0.5, dtype=np.float32)
        else:
            mean_preds = (pred_sum[uid] / pred_cnt[uid]).astype(np.float32)

        for idx, k in enumerate(["C1", "C2", "C3", "C4", "C5", "C6", "C7"]):
            submission_dict["row_id"].append(f"{uid}_{k}")
            submission_dict["fractured"].append(
                float(np.clip(mean_preds[idx], 1e-6, 1 - 1e-6))
            )

    return submission_dict, feature_array_dict


submission_dict, feature_array_dict = _infer_stage1_flat(study_id_list)



## === cell 12
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
    generator2, desc="Stage2 patient_overall", total=len(generator2)
):
    with torch.inference_mode():
        features = features.to(device, non_blocking=True)
        preds = lv2_model(features)
        preds = torch.sigmoid(preds).squeeze(-1).detach().cpu().numpy()

    for j in range(len(list_uid)):
        submission_dict["row_id"].append(f"{list_uid[j]}_patient_overall")
        submission_dict["fractured"].append(float(np.clip(preds[j], 1e-6, 1 - 1e-6)))



## === cell 13
sub_df = pd.DataFrame(submission_dict)
sub_df = sub_df.drop_duplicates(subset=["row_id"], keep="first")

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
final = sample_sub[["row_id"]].merge(sub_df, on="row_id", how="left")

final["fractured"] = final["fractured"].astype(float).fillna(0.5).clip(1e-6, 1 - 1e-6)

assert len(final) == len(sample_sub), "Submission row count mismatch"
assert final["row_id"].is_unique, "row_id not unique in final submission"

final.to_csv("submission.csv", index=False)
print(final.head())
print("Wrote submission.csv with shape:", final.shape)
