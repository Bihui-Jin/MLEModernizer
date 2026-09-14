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
import os, sys, glob, time, warnings

warnings.filterwarnings("ignore")

try:
    import pylibjpeg  # optional
except Exception:
    pylibjpeg = None

import numpy as np
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut

import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision.models.convnext import convnext_base

torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




## === cell 1
BASE_PATH = "../input/rsna-2022-cervical-spine-fracture-detection"
TEST_CSV = f"{BASE_PATH}/test.csv"
TEST_PATH = f"{BASE_PATH}/test_images"
SAMPLE_SUB = f"{BASE_PATH}/sample_submission.csv"

LV1_CKPT = "../input/cnn-lstm-oct-24/model_1.pth"
LV2_CKPT = "../input/cnn-lstm-oct-21/run_5b/run_5b/model_lstm_3.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




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




## === cell 3
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




## === cell 4
test_df = load_df_test()
study_id_list = list(test_df.StudyInstanceUID.unique())
print("Num studies:", len(study_id_list))
print("Device:", device)




## === cell 5
def _list_dicom_slices(study_folder):
    """Return sorted list of (slice_number:int, path:str) based on filename like '123.dcm'."""
    pairs = []
    try:
        with os.scandir(study_folder) as it:
            for e in it:
                if not e.is_file():
                    continue
                name = e.name
                if len(name) < 5 or name[-4:].lower() != ".dcm":
                    continue
                stem = name[:-4]
                if not stem.isdigit():
                    continue
                pairs.append((int(stem), e.path))
    except FileNotFoundError:
        return []
    pairs.sort(key=lambda x: x[0])
    return pairs


selected_image_dict = {}
dicom_index_dict = {}  # uid -> sorted list of slice_numbers available
dicom_path_dict = {}  # uid -> {slice_num: path}

for uid in study_id_list:
    study_folder = os.path.join(TEST_PATH, uid)
    pairs = _list_dicom_slices(study_folder)
    slice_nums = [p[0] for p in pairs]
    dicom_index_dict[uid] = slice_nums
    dicom_path_dict[uid] = {sn: fp for sn, fp in pairs}

    n = len(slice_nums)
    if n == 0:
        selected_image_dict[uid] = []
        continue

    middle = n // 2
    num_left = num_right = int(0.15 * n)
    left_idx = list(
        range(max(1, middle - num_left), middle)
    )  # avoid first for neighbor access
    right_idx = list(
        range(middle + 1, min(n - 1, middle + num_right + 1))
    )  # avoid last
    chosen_idx = left_idx + right_idx
    selected_slices = (
        [slice_nums[i] for i in chosen_idx]
        if len(chosen_idx) > 0
        else [slice_nums[middle]]
    )
    selected_image_dict[uid] = selected_slices




## === cell 6
def _get_pixels_safely(ds: pydicom.dataset.FileDataset):
    """
    Robust pixel extraction:
    - Try ds.pixel_array (requires decompression plugins for JPEG transfer syntaxes).
    - If that fails, DO NOT attempt to reshape PixelData for compressed syntaxes (it will be invalid).
      Instead raise and let the caller fall back to a neutral image.
    - For uncompressed syntaxes, attempt a raw PixelData reshape fallback.
    """
    try:
        return ds.pixel_array
    except Exception as e:
        tsuid = str(getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", ""))
        tsuid_lower = tsuid.lower()

        if (
            ".1.2.4." in tsuid_lower
            or "jpeg" in tsuid_lower
            or "1.2.840.10008.1.2.4" in tsuid_lower
        ):
            raise RuntimeError(
                f"Compressed DICOM without decoder plugins: {tsuid}"
            ) from e

        if "1.2.840.10008.1.2" not in tsuid_lower:
            raise

        rows = int(ds.Rows)
        cols = int(ds.Columns)
        bits = int(ds.BitsAllocated)
        signed = int(getattr(ds, "PixelRepresentation", 0)) == 1
        if bits == 16:
            dtype = np.int16 if signed else np.uint16
        elif bits == 8:
            dtype = np.int8 if signed else np.uint8
        else:
            raise RuntimeError(f"Unsupported BitsAllocated={bits} for raw fallback")

        expected = rows * cols
        buf = ds.PixelData
        itemsize = np.dtype(dtype).itemsize
        if len(buf) < expected * itemsize:
            raise RuntimeError(
                "PixelData buffer smaller than expected for uncompressed fallback"
            )

        arr = np.frombuffer(buf, dtype=dtype, count=expected).reshape(rows, cols)
        return arr


def window(ds, WL=400, WW=1800):
    img = _get_pixels_safely(ds)
    try:
        img = apply_modality_lut(img, ds)
    except Exception:
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        img = img.astype(np.float32) * slope + intercept
    else:
        img = img.astype(np.float32, copy=False)

    upper, lower = WL + WW // 2, WL - WW // 2
    X = np.clip(img, lower, upper)
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




## === cell 7
mean = np.array([0.456, 0.456, 0.456], dtype=np.float32)
std = np.array([0.224, 0.224, 0.224], dtype=np.float32)

_MEAN_CHW = torch.from_numpy(mean.reshape(3, 1, 1))
_STD_CHW = torch.from_numpy(std.reshape(3, 1, 1))


class _BoundedCache:
    __slots__ = ("d", "max_items")

    def __init__(self, max_items: int):
        self.d = {}
        self.max_items = max_items

    def get(self, k):
        v = self.d.get(k)
        if v is not None:
            self.d.pop(k, None)
            self.d[k] = v
        return v

    def set(self, k, v):
        d = self.d
        if k in d:
            d.pop(k, None)
        d[k] = v
        if len(d) > self.max_items:
            d.pop(next(iter(d)))


def _read_windowed_fast(fp: str, cache: _BoundedCache):
    arr = cache.get(fp)
    if arr is not None:
        return arr

    ds = pydicom.dcmread(
        fp,
        force=True,
        stop_before_pixels=False,
        specific_tags=[
            "RescaleSlope",
            "RescaleIntercept",
            "Rows",
            "Columns",
            "BitsAllocated",
            "PixelRepresentation",
            "PhotometricInterpretation",
            "SamplesPerPixel",
            "PlanarConfiguration",
            "TransferSyntaxUID",
            "PixelData",
        ],
    )
    arr = window(ds)
    cache.set(fp, arr)
    return arr


class CSFAllImagesDataset(Dataset):
    def __init__(self, study_id_list, selected_image_dict, target_size):
        self.study_id_list = study_id_list
        self.selected_image_dict = selected_image_dict
        self.target_size = target_size

        self.items = []
        for uid in self.study_id_list:
            slices = self.selected_image_dict.get(uid, [])
            available = dicom_index_dict.get(uid, [])
            if len(available) == 0:
                for pos, sn in enumerate(slices):
                    self.items.append((uid, int(sn), None, pos))
                continue
            min_sn, max_sn = available[0], available[-1]
            for pos, sn in enumerate(slices):
                sn = int(sn)
                sn0 = max(min_sn, sn - 1)
                sn1 = max(min_sn, min(max_sn, sn))
                sn2 = min(max_sn, sn + 1)
                self.items.append((uid, sn, (sn0, sn1, sn2), pos))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        uid, slice_num, triplet, pos = self.items[index]
        available = dicom_index_dict[uid]
        if len(available) == 0 or triplet is None:
            stacked_img = np.full(
                (self.target_size, self.target_size, 3), 127, dtype=np.uint8
            )
            X = img2tensor(stacked_img).float().div_(255.0)
            X = (X - _MEAN_CHW) / _STD_CHW
            return X, uid, pos

        cache = getattr(self, "_cache", None)
        if cache is None:
            cache = _BoundedCache(max_items=512)
            self._cache = cache  # worker-local

        try:
            sn0, sn1, sn2 = triplet
            uid_paths = dicom_path_dict.get(uid, {})
            fp0 = uid_paths.get(sn0)
            fp1 = uid_paths.get(sn1)
            fp2 = uid_paths.get(sn2)
            if fp0 is None or fp1 is None or fp2 is None:
                fp0 = os.path.join(TEST_PATH, uid, f"{sn0}.dcm")
                fp1 = os.path.join(TEST_PATH, uid, f"{sn1}.dcm")
                fp2 = os.path.join(TEST_PATH, uid, f"{sn2}.dcm")

            imgs = (
                _read_windowed_fast(fp0, cache),
                _read_windowed_fast(fp1, cache),
                _read_windowed_fast(fp2, cache),
            )
            stacked_img = np.stack(imgs, axis=-1)
        except Exception:
            stacked_img = np.full((512, 512, 3), 127, dtype=np.uint8)

        stacked_img = cv2.resize(
            stacked_img,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_AREA,
        )
        X = img2tensor(stacked_img).float().div_(255.0)
        X = (X - _MEAN_CHW) / _STD_CHW
        return X, uid, pos




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
            feature_array = np.zeros((1, config["feature_size"]), dtype=np.float32)

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
        m = convnext_base(weights=None)  # keep architecture; do not download weights
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
def _load_state_if_exists(model, ckpt_path):
    if ckpt_path and os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(sd, strict=True)
        print(f"Loaded checkpoint: {ckpt_path}")
        return True
    print(f"Checkpoint not found, using default init: {ckpt_path}")
    return False


lv1_model = ConvNextCNN_B_Feature()
_ = _load_state_if_exists(lv1_model, LV1_CKPT)
lv1_model = lv1_model.to(device).eval()

lv2_model = CSFNet(input_len=config["feature_size"], lstm_size=config["lstm_size"])
_ = _load_state_if_exists(lv2_model, LV2_CKPT)
lv2_model = lv2_model.to(device).eval()




## === cell 11
submission_dict = {"row_id": [], "fractured": []}
feature_array_dict = {}

uid_to_nsel = {uid: len(selected_image_dict.get(uid, [])) for uid in study_id_list}

for uid in study_id_list:
    nsel = uid_to_nsel[uid]
    if nsel == 0:
        feature_array_dict[uid] = np.zeros(
            (1, config["feature_size"]), dtype=np.float32
        )
        for name in ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]:
            submission_dict["row_id"].append(f"{uid}_{name}")
            submission_dict["fractured"].append(0.5)
    else:
        feature_array_dict[uid] = np.zeros(
            (nsel, config["feature_size"]), dtype=np.float32
        )

all_images_ds = CSFAllImagesDataset(
    study_id_list=study_id_list,
    selected_image_dict=selected_image_dict,
    target_size=config["target_size"],
)


def _collate_images(batch):
    imgs, uids, poss = zip(*batch)
    return torch.stack(imgs, 0), list(uids), torch.as_tensor(poss, dtype=torch.int64)


_cpu = os.cpu_count() or 2
num_workers = min(4, _cpu)


def _seed_worker(worker_id):
    seed = 0 + worker_id
    np.random.seed(seed)
    torch.manual_seed(seed)


all_images_loader = DataLoader(
    all_images_ds,
    batch_size=config["batch_size_image_level"],
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
    num_workers=num_workers,
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
    collate_fn=_collate_images,
    worker_init_fn=_seed_worker if num_workers > 0 else None,
)

uid_to_int = {uid: i for i, uid in enumerate(study_id_list)}
pred_sum_arr = np.zeros((len(study_id_list), 7), dtype=np.float64)
pred_cnt_arr = np.zeros((len(study_id_list),), dtype=np.int64)

sigmoid = torch.sigmoid
to_cpu_numpy = lambda t: t.detach().cpu().numpy()

for images, uids, poss in tqdm(all_images_loader, desc="Images (global)"):
    with torch.no_grad():
        images = images.to(device, non_blocking=True)
        features, preds = lv1_model(images)
        preds_sig = to_cpu_numpy(sigmoid(preds)).astype(np.float32, copy=False)  # (B,7)
        feats_np = to_cpu_numpy(features).astype(np.float32, copy=False)  # (B,1024)

    uid_idx = np.fromiter(
        (uid_to_int[u] for u in uids), dtype=np.int64, count=len(uids)
    )

    np.add.at(pred_sum_arr, uid_idx, preds_sig.astype(np.float64, copy=False))
    pred_cnt_arr += np.bincount(uid_idx, minlength=pred_cnt_arr.shape[0])

    order = np.argsort(uid_idx, kind="mergesort")
    uid_idx_s = uid_idx[order]
    poss_s = poss.numpy()[order]
    feats_s = feats_np[order]
    uids_s = [uids[i] for i in order]

    start = 0
    B = len(uids_s)
    while start < B:
        u = uids_s[start]
        end = start + 1
        while end < B and uids_s[end] == u:
            end += 1
        feature_array_dict[u][poss_s[start:end]] = feats_s[start:end]
        start = end

for i, uid in enumerate(study_id_list):
    if uid_to_nsel[uid] == 0:
        continue
    cnt = int(pred_cnt_arr[i]) if int(pred_cnt_arr[i]) > 0 else 1
    mean_preds = (pred_sum_arr[i] / cnt).astype(np.float32)
    for idx, name in enumerate(["C1", "C2", "C3", "C4", "C5", "C6", "C7"]):
        submission_dict["row_id"].append(f"{uid}_{name}")
        submission_dict["fractured"].append(float(mean_preds[idx]))




## === cell 12
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

for features, list_uid in tqdm(generator, total=len(generator), desc="Patient head"):
    with torch.no_grad():
        features = features.to(device, non_blocking=True)
        preds = lv2_model(features)
        preds = torch.sigmoid(preds).squeeze(1).detach().cpu().numpy()

    for j in range(len(list_uid)):
        submission_dict["row_id"].append(f"{list_uid[j]}_patient_overall")
        submission_dict["fractured"].append(float(preds[j]))




## === cell 13
sub_df = pd.DataFrame.from_dict(submission_dict)

template = pd.read_csv(SAMPLE_SUB)
sub_df = template[["row_id"]].merge(sub_df, on="row_id", how="left")

sub_df["fractured"] = sub_df["fractured"].astype(np.float32)
sub_df["fractured"] = sub_df["fractured"].fillna(0.5).clip(1e-6, 1 - 1e-6)

sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print("Wrote submission.csv with rows:", len(sub_df))
