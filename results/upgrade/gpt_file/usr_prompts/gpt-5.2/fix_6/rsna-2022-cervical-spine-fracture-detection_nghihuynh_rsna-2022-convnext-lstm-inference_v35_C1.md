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

0.6065649569342114

# 6. Current score

0.9225

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.91732) has done: 'I fix the missing model weight paths by making the code fall back to a safe, deterministic baseline prediction if the external checkpoint dataset isn’t present (so a valid submission is always produced). I also fix DICOM JPEG decompression failures by using pydicom’s `pixel_array_options(use_v2_backend=True)` and a robust pixel extraction fallback, avoiding hard dependency on missing GDCM/pylibjpeg plugins. Additionally, I correct slice indexing (the code was using 0-based indices against 1-based DICOM filenames) and add boundary handling for first/last slices to prevent file-not-found and KeyError issues downstream. Finally, I ensure the submission exactly matches `test.csv` row order/contents by merging predictions onto the provided `row_id`s (guaranteeing 14536 rows and correct formatting).'
- What this solution (achieved 0.92265) has done: 'Most of the timeout comes from repeatedly scanning each study folder with `glob` and repeatedly reading 3 DICOM files per selected slice with `pydicom.dcmread` (full metadata) inside a single-process DataLoader loop. To keep identical model logic and predictions, the main speedups are: (1) replace `glob` with `os.scandir` and cache sorted slice numbers, (2) use `pydicom.dcmread(..., stop_before_pixels=True)` to read only the minimal headers needed for rescale/windowing and then decode pixels via `ds.pixel_array` (same pixel values), and (3) enable multi-worker DataLoader with persistent workers to parallelize DICOM decoding while the GPU runs inference. Additionally, enable cuDNN benchmarking (safe for inference) and remove tiny per-item overheads (avoid repeated path joins / searchsorted allocations) without changing any computations.'
- What this solution (achieved 0.9225) has done: 'The timeout is dominated by Stage1 repeatedly opening each DICOM slice 3 times (left/center/right) while using `stop_before_pixels=True` (which forces the fallback path and/or yields empty images), plus running a heavy ConvNeXt on ~80 slices per study with a single-threaded DataLoader. The changes below keep the same slice selection, windowing math, model forward passes, and aggregation semantics, but eliminate redundant DICOM reads by caching decoded slices per study, correctly load pixel data in one read, and reduce Python overhead by precomputing neighbor indices and batching more efficiently. In addition, Stage1 now uses multiple DataLoader workers with persistent workers/prefetching to overlap CPU decode with GPU inference (or just parallelize decode on CPU), without changing any predictions other than negligible float differences. Finally, small but safe optimizations remove extra tensor/numpy conversions and unnecessary list concatenations.'

# 9. Code solution

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
def get_sorted_slice_numbers(uid: str):
    folder = os.path.join(TEST_PATH, uid)
    try:
        with os.scandir(folder) as it:
            nums = []
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
    except FileNotFoundError:
        return []
    if not nums:
        return []
    nums.sort()
    return nums


selected_image_dict = {}
slice_numbers_dict = {}

for uid in study_id_list:
    slice_nums = get_sorted_slice_numbers(uid)
    slice_numbers_dict[uid] = slice_nums
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
                if arr.size >= rows * cols:
                    return arr[: rows * cols].reshape(rows, cols)
        except Exception:
            pass
    return None


def window(ds, WL=400, WW=1800):
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))

    img = safe_pixel_array(ds)
    if img is None:
        return np.zeros((512, 512), dtype="uint8")

    img = img.astype(np.float32) * slope + intercept
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




## === cell 7
mean = np.array([0.456, 0.456, 0.456], dtype=np.float32)
std = np.array([0.224, 0.224, 0.224], dtype=np.float32)


class CSFImageDataset(Dataset):
    def __init__(self, uid, image_list, target_size, crop_size):
        self.uid = uid
        self.image_list = image_list
        self.target_size = target_size
        self.crop_size = crop_size
        self._path = os.path.join(TEST_PATH, self.uid)
        self._all_slices = slice_numbers_dict.get(self.uid, [])

        all_slices = self._all_slices
        if len(all_slices) > 0 and len(self.image_list) > 0:
            all_arr = np.asarray(all_slices, dtype=np.int32)
            req_arr = np.asarray(self.image_list, dtype=np.int32)
            pos = np.searchsorted(all_arr, req_arr)
            pos = np.clip(pos, 0, len(all_arr) - 1)
            exact = all_arr[pos] == req_arr
            self._pos = pos.astype(np.int32, copy=False)
            self._left_nums = all_arr[np.maximum(self._pos - 1, 0)]
            self._mid_nums = all_arr[self._pos]
            self._right_nums = all_arr[np.minimum(self._pos + 1, len(all_arr) - 1)]
        else:
            self._pos = None
            self._left_nums = None
            self._mid_nums = None
            self._right_nums = None

        self._slice_cache = {}

    def __len__(self):
        return len(self.image_list)

    def _read_one(self, slice_num: int):
        cached = self._slice_cache.get(slice_num, None)
        if cached is not None:
            return cached

        fp = f"{self._path}/{slice_num}.dcm"
        try:
            ds = pydicom.dcmread(fp, force=True)
            img = window(ds)
        except Exception:
            img = np.zeros((512, 512), dtype="uint8")

        self._slice_cache[slice_num] = img
        return img

    def __getitem__(self, index):
        all_slices = self._all_slices
        if len(all_slices) == 0 or self._left_nums is None:
            stacked_img = np.zeros(
                (self.target_size, self.target_size, 3), dtype=np.uint8
            )
            X = img2tensor((stacked_img / 255.0 - mean) / std)
            return X

        left_num = int(self._left_nums[index])
        mid_num = int(self._mid_nums[index])
        right_num = int(self._right_nums[index])

        imgs = [
            self._read_one(left_num),
            self._read_one(mid_num),
            self._read_one(right_num),
        ]
        stacked_img = np.stack(imgs, axis=-1)
        stacked_img = cv2.resize(
            stacked_img,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_LINEAR,
        )

        X = img2tensor((stacked_img / 255.0 - mean) / std)
        return X




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

submission_dict = {"row_id": [], "fractured": []}
feature_array_dict = {}

run_stage1 = True

_num_workers = min(4, os.cpu_count() or 1)
_persistent_workers = bool(_num_workers > 0)
_prefetch_factor = 2

levels = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]

for uid in tqdm(study_id_list, desc="Stage1 per-study", total=len(study_id_list)):
    image_list = selected_image_dict.get(uid, [])
    if len(image_list) == 0 or not run_stage1:
        for c in levels:
            submission_dict["row_id"].append(f"{uid}_{c}")
            submission_dict["fractured"].append(float(BASELINE_LEVEL))
        feature_array_dict[uid] = np.zeros(
            (max(len(image_list), 1), config["feature_size"]), dtype=np.float32
        )
        continue

    dataset = CSFImageDataset(
        uid=uid,
        image_list=image_list,
        target_size=config["target_size"],
        crop_size=config["crop_size"],
    )
    dl_kwargs = dict(
        dataset=dataset,
        batch_size=config["batch_size_image_level"],
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        num_workers=_num_workers,
        persistent_workers=_persistent_workers,
        prefetch_factor=_prefetch_factor if _num_workers > 0 else None,
    )
    if dl_kwargs["prefetch_factor"] is None:
        dl_kwargs.pop("prefetch_factor")
    generator = DataLoader(**dl_kwargs)

    feature_array = np.zeros((len(dataset), config["feature_size"]), dtype=np.float32)

    pred_sum = np.zeros((7,), dtype=np.float64)
    pred_count = 0

    bs = config["batch_size_image_level"]
    for i, images in enumerate(generator):
        with torch.no_grad():
            start = i * bs
            end = min(start + bs, len(generator.dataset))

            images = images.to(device, non_blocking=True)
            features, preds = lv1_model(images)

            feat_np = features.detach().cpu().numpy()
            if feat_np.ndim == 1:
                feat_np = feat_np[None, :]
            feature_array[start:end] = feat_np[: (end - start)]

            p = preds.sigmoid().detach().cpu().numpy()
            pred_sum += p.sum(axis=0, dtype=np.float64)
            pred_count += p.shape[0]

    feature_array_dict[uid] = feature_array

    if pred_count == 0:
        mean_preds = np.full((7,), BASELINE_LEVEL, dtype=np.float32)
    else:
        mean_preds = (pred_sum / pred_count).astype(np.float32)

    for idx, c in enumerate(levels):
        submission_dict["row_id"].append(f"{uid}_{c}")
        submission_dict["fractured"].append(float(mean_preds[idx]))




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_55/3130730905.py in <cell line: 0>()
     55 
     56     bs = config["batch_size_image_level"]
---> 57     for i, images in enumerate(generator):
     58         with torch.no_grad():
     59             start = i * bs

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __iter__(self)
    484         if self.persistent_workers and self.num_workers > 0:
    485             if self._iterator is None:
--> 486                 self._iterator = self._get_iterator()
    487             else:
    488                 self._iterator._reset(self)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _get_iterator(self)
    420         else:
    421             self.check_worker_number_rationality()
--> 422             return _MultiProcessingDataLoaderIter(self)
    423 
    424     @property

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __init__(self, loader)
   1144             #     before it starts, and __del__ tries to join but will get:
   1145             #     AssertionError: can only join a started process.
-> 1146             w.start()
   1147             self._index_queues.append(index_queue)
   1148             self._workers.append(w)

/usr/lib/python3.11/multiprocessing/process.py in start(self)
    119                'daemonic processes are not allowed to have children'
    120         _cleanup()
--> 121         self._popen = self._Popen(self)
    122         self._sentinel = self._popen.sentinel
    123         # Avoid a refcycle if the target function holds an indirect

/usr/lib/python3.11/multiprocessing/context.py in _Popen(process_obj)
    222     @staticmethod
    223     def _Popen(process_obj):
--> 224         return _default_context.get_context().Process._Popen(process_obj)
    225 
    226     @staticmethod

/usr/lib/python3.11/multiprocessing/context.py in _Popen(process_obj)
    279         def _Popen(process_obj):
    280             from .popen_fork import Popen
--> 281             return Popen(process_obj)
    282 
    283     class SpawnProcess(process.BaseProcess):

/usr/lib/python3.11/multiprocessing/popen_fork.py in __init__(self, process_obj)
     17         self.returncode = None
     18         self.finalizer = None
---> 19         self._launch(process_obj)
     20 
     21     def duplicate_for_child(self, fd):

/usr/lib/python3.11/multiprocessing/popen_fork.py in _launch(self, process_obj)
     63         code = 1
     64         parent_r, child_w = os.pipe()
---> 65         child_r, parent_w = os.pipe()
     66         self.pid = os.fork()
     67         if self.pid == 0:

OSError: [Errno 24] Too many open files

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
        with torch.no_grad():
            features = features.to(device, non_blocking=True)
            preds = lv2_model(features)
            preds = np.squeeze(preds.sigmoid().detach().cpu().numpy())

        if np.ndim(preds) == 0:
            preds = np.array([float(preds)], dtype=np.float32)

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
