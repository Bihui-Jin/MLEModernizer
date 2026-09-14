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
import os, sys, glob, time, pickle, warnings, subprocess

warnings.filterwarnings("ignore")


def _ensure_dicom_decoders():
    return


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
from torchvision.models.convnext import convnext_base, ConvNeXt_Base_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(max(1, os.cpu_count() // 2))



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




## === cell 5
test_df = load_df_test()
TEST_PATH = f"{DATA_ROOT}/test_images"

study_id_list = list(test_df.StudyInstanceUID.unique())
print("n_test_rows:", len(test_df), "n_unique_studies:", len(study_id_list))
print("first_uids:", study_id_list[:3])



## === cell 6
selected_image_dict = {}
dicom_index_dict = {}

for uid in tqdm(study_id_list, desc="Index test DICOMs"):
    udir = os.path.join(TEST_PATH, uid)
    idxs = []
    try:
        with os.scandir(udir) as it:
            for e in it:
                if not e.is_file():
                    continue
                name = e.name
                if not name.endswith(".dcm"):
                    continue
                base = name[:-4]
                try:
                    idxs.append(int(base))
                except Exception:
                    continue
    except FileNotFoundError:
        idxs = []

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

if len(study_id_list) > 0:
    print("UID:", study_id_list[0])
    print("Chosen slices (sample):", selected_image_dict[study_id_list[0]][:10])



## === cell 7
PASS = True



## === cell 8
_DICOM_HEADER_TAGS = [
    (0x0028, 0x1053),  # RescaleSlope
    (0x0028, 0x1052),  # RescaleIntercept
    (0x0028, 0x0010),  # Rows
    (0x0028, 0x0011),  # Columns
    (0x7FE0, 0x0010),  # PixelData (for presence check)
]


def _opencv_decode_dicom_header_and_pixeldata(ds, pixeldata_bytes):
    try:
        if ds.file_meta is not None:
            tsuid = str(getattr(ds.file_meta, "TransferSyntaxUID", ""))
            if tsuid and ("1.2.840.10008.1.2.4" in tsuid):
                from pydicom.encaps import generate_pixel_data_frame

                frame = next(generate_pixel_data_frame(pixeldata_bytes))
                arr = np.frombuffer(frame, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
                if img is None:
                    return None
                if img.ndim == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                return img.astype(np.float32)
    except Exception:
        return None
    return None


def _get_pixel_array_safe(ds, path):
    try:
        return ds.pixel_array.astype(np.float32)
    except Exception:
        try:
            ds_full = pydicom.dcmread(path, force=True)
            try:
                return ds_full.pixel_array.astype(np.float32)
            except Exception:
                img = _opencv_decode_dicom_header_and_pixeldata(
                    ds_full, getattr(ds_full, "PixelData", b"")
                )
                if img is not None:
                    return img
                rows = int(getattr(ds_full, "Rows", 512))
                cols = int(getattr(ds_full, "Columns", 512))
                return np.zeros((rows, cols), dtype=np.float32)
        except Exception:
            rows = int(getattr(ds, "Rows", 512))
            cols = int(getattr(ds, "Columns", 512))
            return np.zeros((rows, cols), dtype=np.float32)


def window(data, WL=400, WW=1800, path=None):
    slope = float(getattr(data, "RescaleSlope", 1.0))
    intercept = float(getattr(data, "RescaleIntercept", 0.0))
    img = _get_pixel_array_safe(data, path)
    img = img * slope + intercept
    upper, lower = WL + WW // 2, WL - WW // 2
    X = np.clip(img, lower, upper)
    mn = float(X.min())
    X = X - mn
    mx = float(X.max())
    if mx > 0:
        X = X / mx
    X = (X * 255.0).astype("uint8")
    return X


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))




## === cell 9
PASS = True



## === cell 10
mean = np.array([0.456, 0.456, 0.456], dtype=np.float32)
std = np.array([0.224, 0.224, 0.224], dtype=np.float32)

_INFERENCE_TRANSFORM = Compose([CenterCrop(config["crop_size"], config["crop_size"])])

_nearest_slice_map = {}
for uid, idxs in dicom_index_dict.items():
    if not idxs:
        _nearest_slice_map[uid] = None
        continue
    idxs_sorted = idxs
    mn, mx = idxs_sorted[0], idxs_sorted[-1]
    present = set(idxs_sorted)
    arr = np.empty((mx - mn + 1,), dtype=np.int32)
    last = -1
    for v in range(mn, mx + 1):
        if v in present:
            last = v
        arr[v - mn] = last
    nextv = -1
    for v in range(mx, mn - 1, -1):
        if v in present:
            nextv = v
        li = arr[v - mn]
        if li < 0:
            arr[v - mn] = nextv
        else:
            if nextv >= 0 and (v - li) > (nextv - v):
                arr[v - mn] = nextv
    _nearest_slice_map[uid] = (mn, mx, arr)


class CSFImageDataset(Dataset):
    def __init__(self, uid, image_list, target_size, crop_size):
        self.uid = uid
        self.image_list = image_list
        self.target_size = target_size
        self.crop_size = crop_size

        self._idxs = dicom_index_dict[self.uid]
        self._idxs_set = set(self._idxs)

        if len(self._idxs) == 0:
            self._min_idx = None
            self._max_idx = None
        else:
            self._min_idx = self._idxs[0]
            self._max_idx = self._idxs[-1]

        self._nearest = _nearest_slice_map.get(self.uid, None)

        self._win_cache = {}

    def __len__(self):
        return len(self.image_list)

    def _path(self, slice_idx: int):
        return os.path.join(TEST_PATH, self.uid, f"{slice_idx}.dcm")

    def _read_dicom_header(self, slice_idx: int):
        path = self._path(slice_idx)
        ds = pydicom.dcmread(
            path,
            force=True,
            stop_before_pixels=True,
        )
        return ds, path

    def _nearest_existing(self, center: int):
        if self._nearest is None:
            return center
        mn, mx, arr = self._nearest
        if center <= mn:
            return int(arr[0])
        if center >= mx:
            return int(arr[-1])
        return int(arr[center - mn])

    def _get_windowed(self, slice_idx: int):
        v = self._win_cache.get(slice_idx)
        if v is not None:
            return v
        ds, path = self._read_dicom_header(slice_idx)
        img = window(ds, path=path)
        self._win_cache[slice_idx] = img
        return img

    def __getitem__(self, index):
        idxs = self._idxs
        if len(idxs) == 0:
            raise RuntimeError(f"No DICOM slices found for uid={self.uid}")

        center = int(self.image_list[index])
        if center not in self._idxs_set:
            center = self._nearest_existing(center)

        min_idx, max_idx = self._min_idx, self._max_idx
        s0 = max(min_idx, center - 1)
        s1 = center
        s2 = min(max_idx, center + 1)

        imgs = [self._get_windowed(s0), self._get_windowed(s1), self._get_windowed(s2)]

        stacked_img = np.stack(imgs, axis=-1)
        stacked_img = cv2.resize(
            stacked_img,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_LINEAR,
        )

        stacked_img = _INFERENCE_TRANSFORM(image=stacked_img)["image"]

        X = img2tensor((stacked_img / 255.0 - mean) / std)
        return X




## === cell 11
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




## === cell 12
PASS = True




## === cell 13
class ConvNextCNN_B_Feature(nn.Module):
    def __init__(self):
        super(ConvNextCNN_B_Feature, self).__init__()
        m = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
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




## === cell 14
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




## === cell 15
def _find_first_existing(paths):
    for p in paths:
        if p is not None and os.path.exists(p):
            return p
    return None


def _recursive_find_checkpoint(root_dirs, target_filenames, max_hits_per_name=3):
    found = {name: [] for name in target_filenames}
    for root in root_dirs:
        if not root or not os.path.exists(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            fn_set = set(filenames)
            for name in target_filenames:
                if name in fn_set and len(found[name]) < max_hits_per_name:
                    found[name].append(os.path.join(dirpath, name))
            if all(len(found[name]) >= max_hits_per_name for name in target_filenames):
                break
    return found


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for k in ("state_dict", "model_state_dict", "model", "net"):
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k]
    return ckpt_obj


def _load_state_dict_forgiving(model, path):
    ckpt = torch.load(path, map_location="cpu")
    sd = _extract_state_dict(ckpt)

    if isinstance(sd, dict):
        new_sd = {}
        for k, v in sd.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            if nk.startswith("model."):
                nk = nk[len("model.") :]
            new_sd[nk] = v
        sd = new_sd

    try:
        model.load_state_dict(sd, strict=True)
        return True, "strict"
    except Exception:
        try:
            model.load_state_dict(sd, strict=False)
            return True, "non-strict"
        except Exception as e:
            return False, repr(e)


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

if (lv1_path is None) or (lv2_path is None):
    hits = _recursive_find_checkpoint(
        root_dirs=[DATA_ROOT, "../input"],
        target_filenames=["model_3.pth", "model_lstm_3.pth"],
        max_hits_per_name=5,
    )
    if lv1_path is None and len(hits["model_3.pth"]) > 0:
        lv1_path = hits["model_3.pth"][0]
    if lv2_path is None and len(hits["model_lstm_3.pth"]) > 0:
        lv2_path = hits["model_lstm_3.pth"][0]

has_lv1_weights = lv1_path is not None
has_lv2_weights = lv2_path is not None

lv1_model = ConvNextCNN_B_Feature().to(device).eval()
lv2_model = (
    CSFNet(input_len=config["feature_size"], lstm_size=config["lstm_size"])
    .to(device)
    .eval()
)

if has_lv1_weights:
    ok, how = _load_state_dict_forgiving(lv1_model, lv1_path)
    print("lv1 weights:", lv1_path, "| loaded:", ok, "| mode:", how)
else:
    print(
        "WARNING: lv1 weights not found; will use ImageNet-pretrained ConvNeXt backbone with randomly initialized fc head."
    )

if has_lv2_weights:
    ok, how = _load_state_dict_forgiving(lv2_model, lv2_path)
    print("lv2 weights:", lv2_path, "| loaded:", ok, "| mode:", how)
else:
    print(
        "WARNING: lv2 weights not found; patient_overall will be derived from C1-C7 probabilities (no constant fallback)."
    )



## === cell 16
PASS = True



## === cell 17
PASS = True



## === cell 18
import gc

submission_dict = {"row_id": [], "fractured": []}
feature_array_dict = {}
cprob_by_uid = {}

FALLBACK_C_PROB = 0.10

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def _safe_fd_limit():
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        return int(soft)
    except Exception:
        return None


def _pick_num_workers_img():
    ncpu = os.cpu_count() or 1
    fd_lim = _safe_fd_limit()
    if ncpu <= 2:
        return 0
    if fd_lim is not None and fd_lim < 2048:
        return 0
    return min(2, max(1, ncpu // 8))


_num_workers_img = _pick_num_workers_img()
print("Image-level num_workers:", _num_workers_img, "| fd_limit:", _safe_fd_limit())

for uid in tqdm(study_id_list, desc="Predict studies"):
    image_list = selected_image_dict.get(uid, [])
    if len(image_list) == 0:
        idxs = dicom_index_dict.get(uid, [])
        if len(idxs) == 0:
            mean_preds = np.full((7,), FALLBACK_C_PROB, dtype=np.float32)
            feature_array_dict[uid] = np.zeros(
                (1, config["feature_size"]), dtype=np.float32
            )
        else:
            image_list = [idxs[len(idxs) // 2]]

    if uid not in feature_array_dict:
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
            num_workers=_num_workers_img,
            persistent_workers=(_num_workers_img > 0),
            prefetch_factor=2 if _num_workers_img > 0 else None,
        )

        n_items = len(dataset)
        feature_array = np.zeros((n_items, config["feature_size"]), dtype=np.float32)
        pred_sum = np.zeros((7,), dtype=np.float32)
        pred_count = 0

        for i, images in enumerate(generator):
            with torch.no_grad():
                start = i * config["batch_size_image_level"]
                end = min(start + images.shape[0], n_items)

                images = images.to(device, non_blocking=True)
                features, preds = lv1_model(images)

                f_np = features.detach().cpu().numpy()
                feature_array[start:end] = f_np[: (end - start)]

                p_np = preds.sigmoid().detach().cpu().numpy()
                pred_sum += p_np.sum(axis=0).astype(np.float32, copy=False)
                pred_count += p_np.shape[0]

        feature_array_dict[uid] = feature_array
        mean_preds = (pred_sum / max(1, pred_count)).astype(np.float32, copy=False)

        try:
            if hasattr(generator, "_iterator") and generator._iterator is not None:
                generator._iterator._shutdown_workers()
        except Exception:
            pass
        del generator
        del dataset
        gc.collect()

    cprob_by_uid[uid] = mean_preds.astype(np.float32)

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



## === cell 19
PASS = True



## === cell 20
if has_lv2_weights:
    dataset = CSFInstanceDataset(
        feature_array_dict=feature_array_dict,
        study_id_list=study_id_list,
        seq_len=config["seq_len"],
    )
    _num_workers_lv2 = (
        0
        if (os.cpu_count() is None or os.cpu_count() <= 2)
        else min(2, max(1, os.cpu_count() // 8))
    )
    generator = DataLoader(
        dataset=dataset,
        batch_size=config["batch_size_patient_level"],
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=_num_workers_lv2,
        persistent_workers=(_num_workers_lv2 > 0),
        prefetch_factor=2 if _num_workers_lv2 > 0 else None,
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

    try:
        if hasattr(generator, "_iterator") and generator._iterator is not None:
            generator._iterator._shutdown_workers()
    except Exception:
        pass
    del generator
    del dataset
    gc.collect()
else:
    for uid in study_id_list:
        c = cprob_by_uid.get(uid, np.full((7,), FALLBACK_C_PROB, dtype=np.float32))
        p_any = float(1.0 - np.prod(1.0 - np.clip(c, 0.0, 1.0)))
        submission_dict["row_id"].append(f"{uid}_patient_overall")
        submission_dict["fractured"].append(p_any)



## === cell 21
PASS = True



## === cell 22
sub_df = pd.DataFrame.from_dict(submission_dict)

sample_sub = pd.read_csv(f"{DATA_ROOT}/sample_submission.csv")
sub_df = sample_sub[["row_id"]].merge(sub_df, on="row_id", how="left")

if sub_df["fractured"].isna().any():
    sub_df["fractured"] = sub_df["fractured"].fillna(0.5)

eps = 1e-6
sub_df["fractured"] = sub_df["fractured"].astype(np.float32).clip(eps, 1 - eps)

print(sub_df.head())
print(
    "submission shape:", sub_df.shape, "missing:", int(sub_df["fractured"].isna().sum())
)



## === cell 23
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head(10).to_string(index=False))
