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
import resource
import math
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pydicom

import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision.models.convnext import convnext_base



## === cell 1
try:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target_soft = min(max(soft, 8192), hard)
    resource.setrlimit(resource.RLIMIT_NOFILE, (target_soft, hard))
    print(f"[INFO] RLIMIT_NOFILE soft={soft}-> {target_soft}, hard={hard}")
except Exception as e:
    print(f"[WARN] Could not adjust RLIMIT_NOFILE: {repr(e)}")

torch.backends.cudnn.benchmark = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

try:
    torch.set_num_threads(max(1, min(4, (os.cpu_count() or 4))))
    torch.set_num_interop_threads(1)
except Exception:
    pass



## === cell 2
config = {
    "seq_len": 150,
    "feature_size": 1024,
    "lstm_size": 128,
    "target_size": 384,
    "crop_size": 320,
    "num_classes": 7,
    "batch_size_image_level": 32,
    "batch_size_patient_level": 8,
}




## === cell 3
def load_df_test():
    df_test = pd.read_csv(
        "/kaggle/input/rsna-2022-cervical-spine-fracture-detection/test.csv"
    )

    if len(df_test) and df_test.iloc[0].row_id == "1.2.826.0.1.3680043.10197_C1":
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
TEST_PATH = "/kaggle/input/rsna-2022-cervical-spine-fracture-detection/test_images"
study_id_list = list(test_df.StudyInstanceUID.unique())
len(study_id_list), study_id_list[:3]



## === cell 5
uid_to_files = {}
selected_image_dict = {}

for uid in study_id_list:
    dicom_files = sorted(glob.glob(os.path.join(TEST_PATH, uid, "*.dcm")))
    uid_to_files[uid] = dicom_files

    n = len(dicom_files)
    if n == 0:
        selected_image_dict[uid] = []
        continue

    middle = n // 2
    k = int(0.15 * n)
    left = max(0, middle - k)
    right = min(n - 1, middle + k)
    indices = list(range(left, middle)) + list(range(middle + 1, right + 1))
    if len(indices) == 0:
        indices = [middle]
    selected_image_dict[uid] = indices



## === cell 6
uid0 = study_id_list[0]
print("Example UID:", uid0)
print("Total slices:", len(uid_to_files[uid0]))
print("Selected indices (first 10):", selected_image_dict[uid0][:10])



## === cell 7
try:
    import pydicom.config
    from pydicom.pixel_data_handlers import (
        gdcm_handler,
        pylibjpeg_handler,
        numpy_handler,
    )

    pydicom.config.pixel_data_handlers = [
        gdcm_handler,
        pylibjpeg_handler,
        numpy_handler,
    ]
except Exception:
    pass




## === cell 8
def window_from_pixels(
    img: np.ndarray, slope: float = 1.0, intercept: float = 0.0, WL=400, WW=1800
):
    img = img.astype(np.float32, copy=False)
    img = img * float(slope) + float(intercept)

    upper, lower = WL + WW // 2, WL - WW // 2
    X = np.clip(img, lower, upper)
    X = X - np.min(X)
    mx = np.max(X)
    if mx > 0:
        X = X / mx
    X = (X * 255.0).astype("uint8")
    return X


def safe_dicom_pixel_array(ds):
    """Return pixel_array if possible; otherwise return None."""
    try:
        return ds.pixel_array
    except Exception:
        return None


def window(ds, WL=400, WW=1800):
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))

    arr = safe_dicom_pixel_array(ds)
    if arr is None:
        return np.zeros((512, 512), dtype="uint8")

    return window_from_pixels(arr, slope=slope, intercept=intercept, WL=WL, WW=WW)


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))




## === cell 9
mean = np.array([0.456, 0.456, 0.456], dtype=np.float32)
std = np.array([0.224, 0.224, 0.224], dtype=np.float32)




## === cell 10
class CSFImageDataset(Dataset):
    def __init__(self, uid, image_indices, target_size, crop_size):
        self.uid = uid
        self.image_indices = image_indices
        self.target_size = target_size
        self.crop_size = crop_size
        self._cache = {}  # idx -> uint8 (512,512)

    def __len__(self):
        return len(self.image_indices)

    def _read_slice(self, file_path):
        ds = None
        try:
            ds = pydicom.dcmread(
                file_path,
                force=True,
                stop_before_pixels=False,
                specific_tags=["PixelData", "RescaleSlope", "RescaleIntercept"],
            )
            img = window(ds)
            return img
        finally:
            ds = None

    def _get_cached(self, files, idx):
        v = self._cache.get(idx, None)
        if v is not None:
            return v
        try:
            v = self._read_slice(files[idx])
        except Exception:
            v = np.zeros((512, 512), dtype="uint8")
        self._cache[idx] = v
        return v

    def __getitem__(self, index):
        files = uid_to_files[self.uid]
        n = len(files)
        idx = self.image_indices[index]
        idx0 = max(0, idx - 1)
        idx1 = idx
        idx2 = min(n - 1, idx + 1)

        imgs = [
            self._get_cached(files, idx0),
            self._get_cached(files, idx1),
            self._get_cached(files, idx2),
        ]

        stacked_img = np.stack(imgs, axis=-1)  # H,W,3
        stacked_img = cv2.resize(
            stacked_img,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_LINEAR,
        )

        X = img2tensor((stacked_img.astype(np.float32) / 255.0 - mean) / std)
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
        feature_array = self.feature_array_dict.get(
            uid, np.zeros((0, config["feature_size"]), dtype=np.float32)
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
        X = torch.from_numpy(np.asarray(x, dtype=np.float32))
        return X, uid




## === cell 12
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




## === cell 13
def safe_load_state_dict(model, ckpt_path):
    if ckpt_path is None or (not os.path.exists(ckpt_path)):
        print(
            f"[WARN] Checkpoint not found: {ckpt_path}. Using randomly initialized weights."
        )
        return model
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    return model


lv1_model = ConvNextCNN_B_Feature()
lv1_model = safe_load_state_dict(
    lv1_model, "/kaggle/input/cnn-lstm-oct-21/run_5/run_5/model_3.pth"
)
lv1_model = lv1_model.to(DEVICE).eval()

lv2_model = CSFNet(input_len=config["feature_size"], lstm_size=config["lstm_size"])
lv2_model = safe_load_state_dict(
    lv2_model, "/kaggle/input/cnn-lstm-oct-21/run_5b/run_5b/model_lstm_3.pth"
)
lv2_model = lv2_model.to(DEVICE).eval()



## === cell 14
submission_dict = {"row_id": [], "fractured": []}
feature_array_dict = {}

PIN = torch.cuda.is_available()
bs = int(config["batch_size_image_level"])

_DCM_TAGS = ["PixelData", "RescaleSlope", "RescaleIntercept"]
_ZERO512 = np.zeros((512, 512), dtype="uint8")


def _read_slice_fast(file_path: str):
    ds = None
    try:
        ds = pydicom.dcmread(
            file_path,
            force=True,
            stop_before_pixels=False,
            specific_tags=_DCM_TAGS,
        )
        return window(ds)
    except Exception:
        return _ZERO512
    finally:
        ds = None


_mean = mean.reshape(1, 1, 3)
_std = std.reshape(1, 1, 3)
_inv255 = np.float32(1.0 / 255.0)

_cpu = os.cpu_count() or 4
_stage1_workers = int(max(4, min(16, _cpu)))

target_size = int(config["target_size"])
feat_size = int(config["feature_size"])

_batch_nhwc = np.empty((bs, target_size, target_size, 3), dtype=np.float32)

uid_precomp = {}
for uid in study_id_list:
    image_indices = selected_image_dict.get(uid, [])
    files = uid_to_files.get(uid, [])
    nfiles = len(files)
    if (not image_indices) or nfiles == 0:
        uid_precomp[uid] = (files, nfiles, image_indices, None, None)
        continue

    triplets = []
    needed_set = set()
    for idx in image_indices:
        idx0 = idx - 1 if idx > 0 else 0
        idx1 = idx
        idx2 = idx + 1 if (idx + 1) < nfiles else (nfiles - 1)
        triplets.append((idx0, idx1, idx2))
        needed_set.add(idx0)
        needed_set.add(idx1)
        needed_set.add(idx2)

    needed_list = sorted(needed_set)
    uid_precomp[uid] = (files, nfiles, image_indices, triplets, needed_list)

with ThreadPoolExecutor(max_workers=_stage1_workers) as ex:
    for uid in tqdm(study_id_list, desc="Stage1 per-UID"):
        files, nfiles, image_indices, triplets, needed_list = uid_precomp[uid]

        if (not image_indices) or nfiles == 0:
            feature_array_dict[uid] = np.zeros((0, feat_size), dtype=np.float32)
            for k in range(1, 8):
                submission_dict["row_id"].append(f"{uid}_C{k}")
                submission_dict["fractured"].append(0.5)
            continue

        cache_u8 = {}
        if len(needed_list) == 1:
            i = needed_list[0]
            cache_u8[i] = _read_slice_fast(files[i])
        else:
            for i, img_u8 in zip(
                needed_list,
                ex.map(lambda p: _read_slice_fast(p), (files[i] for i in needed_list)),
            ):
                cache_u8[i] = img_u8

        cache_chw = {
            i: cv2.resize(
                img_u8, (target_size, target_size), interpolation=cv2.INTER_LINEAR
            )
            for i, img_u8 in cache_u8.items()
        }

        L = len(image_indices)
        feature_array = np.zeros((L, feat_size), dtype=np.float32)
        sum_preds = np.zeros((7,), dtype=np.float64)

        try:
            with torch.inference_mode():
                for start in range(0, L, bs):
                    end = min(start + bs, L)
                    bsz = end - start

                    for bi in range(bsz):
                        idx0, idx1, idx2 = triplets[start + bi]

                        s0 = cache_chw[idx0]
                        s1 = cache_chw[idx1]
                        s2 = cache_chw[idx2]

                        x = _batch_nhwc[bi]
                        x[..., 0] = s0
                        x[..., 1] = s1
                        x[..., 2] = s2
                        x *= _inv255
                        x -= _mean
                        x /= _std

                    images = torch.from_numpy(_batch_nhwc[:bsz]).permute(0, 3, 1, 2)
                    images = images.to(DEVICE, non_blocking=PIN)

                    features, preds = lv1_model(images)
                    feature_array[start:end] = features.detach().cpu().numpy()
                    sum_preds += preds.sigmoid().detach().cpu().numpy().sum(axis=0)

        except Exception as e:
            print(f"[WARN] Stage1 failed for UID={uid}: {repr(e)}")
            feature_array = np.zeros((0, feat_size), dtype=np.float32)
            mean_preds = np.full((7,), 0.5, dtype=np.float32)
        else:
            mean_preds = (sum_preds / float(L)).astype(np.float32)

        feature_array_dict[uid] = feature_array

        for j, level in enumerate(["C1", "C2", "C3", "C4", "C5", "C6", "C7"]):
            submission_dict["row_id"].append(f"{uid}_{level}")
            submission_dict["fractured"].append(float(mean_preds[j]))



## === cell 15
dataset2 = CSFInstanceDataset(
    feature_array_dict=feature_array_dict,
    study_id_list=study_id_list,
    seq_len=config["seq_len"],
)

cpu = os.cpu_count() or 4
num_workers_stage2 = 2 if cpu >= 4 else 0

generator2 = DataLoader(
    dataset=dataset2,
    batch_size=config["batch_size_patient_level"],
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
    num_workers=num_workers_stage2,
    persistent_workers=(num_workers_stage2 > 0),
)

with torch.inference_mode():
    for features, list_uid in tqdm(
        generator2, total=len(generator2), desc="Stage2 patient_overall"
    ):
        features = features.to(DEVICE, non_blocking=True)
        preds = lv2_model(features).sigmoid().detach().cpu().numpy().reshape(-1)

        for j in range(len(list_uid)):
            submission_dict["row_id"].append(f"{list_uid[j]}_patient_overall")
            submission_dict["fractured"].append(float(preds[j]))



## === cell 16
sub_df = pd.DataFrame.from_dict(submission_dict)

sub_df = sub_df.groupby("row_id", as_index=False)["fractured"].mean()

test_row_ids = test_df["row_id"].values
pred_map = dict(zip(sub_df["row_id"].values, sub_df["fractured"].values))

out = pd.DataFrame(
    {
        "row_id": test_row_ids,
        "fractured": [float(pred_map.get(rid, 0.5)) for rid in test_row_ids],
    }
)

out["fractured"] = out["fractured"].clip(1e-6, 1 - 1e-6)

out.head(), out.shape



## === cell 17
out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", out.shape)
print(out.tail())
