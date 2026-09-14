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

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
os.environ["PYTHONWARNINGS"] = "ignore"



## === cell 1
import numpy as np
import pandas as pd
import pydicom

import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.convnext import convnext_base

torch.manual_seed(0)
np.random.seed(0)

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False



## === cell 2
INPUT_DIR = "../input/rsna-2022-cervical-spine-fracture-detection"
TEST_CSV = f"{INPUT_DIR}/test.csv"
SAMPLE_SUB = f"{INPUT_DIR}/sample_submission.csv"
TEST_PATH = f"{INPUT_DIR}/test_images"

assert os.path.exists(TEST_CSV), f"Missing {TEST_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.exists(TEST_PATH), f"Missing {TEST_PATH}"



## === cell 3
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

_CPU_COUNT = os.cpu_count() or 2

config["num_workers_image_level"] = min(12, max(4, _CPU_COUNT - 1))
config["num_workers_patient_level"] = 0  # patient-level is lightweight




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




## === cell 5
test_df = load_df_test()
study_id_list = list(test_df.StudyInstanceUID.unique())


def _list_dicom_slice_numbers(folder: str):
    nums = []
    try:
        with os.scandir(folder) as it:
            for e in it:
                if not e.is_file():
                    continue
                name = e.name
                if not name.endswith(".dcm"):
                    continue
                base = name[:-4]
                try:
                    nums.append(int(base))
                except Exception:
                    continue
    except FileNotFoundError:
        return []
    nums.sort()
    return nums


uid2slices = {}
for uid in tqdm(study_id_list, desc="Indexing test DICOMs"):
    uid2slices[uid] = _list_dicom_slice_numbers(os.path.join(TEST_PATH, uid))

selected_image_dict = {}
for uid in study_id_list:
    nums = uid2slices.get(uid, [])
    if len(nums) == 0:
        selected_image_dict[uid] = []
        continue
    mid_idx = len(nums) // 2
    k = max(1, int(0.15 * len(nums)))
    left = nums[max(0, mid_idx - k) : mid_idx]
    right = nums[mid_idx + 1 : min(len(nums), mid_idx + 1 + k)]
    selected_image_dict[uid] = left + right

uid2neighbors = {}
for uid in study_id_list:
    valid_sorted = uid2slices.get(uid, [])
    if not valid_sorted:
        uid2neighbors[uid] = {}
        continue
    valid_arr = np.asarray(valid_sorted, dtype=np.int32)

    def nearest_available(n: int) -> int:
        pos = int(np.searchsorted(valid_arr, n))
        if pos <= 0:
            return int(valid_arr[0])
        if pos >= len(valid_arr):
            return int(valid_arr[-1])
        before = int(valid_arr[pos - 1])
        after = int(valid_arr[pos])
        return before if abs(before - n) <= abs(after - n) else after

    neighbors = {}
    for s in selected_image_dict.get(uid, []):
        s = int(s)
        neighbors[s] = (
            nearest_available(s - 1),
            nearest_available(s),
            nearest_available(s + 1),
        )
    uid2neighbors[uid] = neighbors



## === cell 6
from functools import lru_cache

_DCM_TAGS = [
    "PixelData",
    "RescaleSlope",
    "RescaleIntercept",
    "PhotometricInterpretation",
    "BitsStored",
    "BitsAllocated",
    "HighBit",
    "PixelRepresentation",
    "SamplesPerPixel",
    "PlanarConfiguration",
    "Rows",
    "Columns",
    "TransferSyntaxUID",
]


def safe_dcmread(path):
    return pydicom.dcmread(
        path,
        force=True,
        stop_before_pixels=False,
        specific_tags=_DCM_TAGS,
    )


def get_pixel_array(ds):
    return ds.pixel_array


def window(ds, WL=400, WW=1800):
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))

    img = get_pixel_array(ds).astype(np.float32)
    img = img * slope + intercept

    upper, lower = WL + WW // 2, WL - WW // 2
    X = np.clip(img, lower, upper)
    X = X - X.min()
    denom = X.max()
    if denom > 0:
        X = X / denom
    X = (X * 255.0).astype("uint8")
    return X


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


@lru_cache(maxsize=16384)
def _read_and_window_cached_path(path: str, WL: int = 400, WW: int = 1800):
    ds = safe_dcmread(path)
    return window(ds, WL=WL, WW=WW)


def _read_and_window_cached_uid_slice(
    uid: str, slice_n: int, path: str, WL: int = 400, WW: int = 1800
):
    return _read_and_window_cached_path(path, WL=WL, WW=WW)




## === cell 7
mean = np.array([0.456, 0.456, 0.456], dtype=np.float32)
std = np.array([0.224, 0.224, 0.224], dtype=np.float32)


class CSFAllSlicesDataset(Dataset):
    def __init__(self, items, target_size):
        """
        items: list of (uid, paths_triplet, slices_triplet, pos) where paths_triplet is (p0,p1,p2)
        """
        self.items = items
        self.target_size = int(target_size)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        uid, (p0, p1, p2), (s0, s1, s2), pos = self.items[index]
        try:
            a0 = _read_and_window_cached_uid_slice(uid, int(s0), p0)
        except Exception:
            a0 = None
        try:
            a1 = _read_and_window_cached_uid_slice(uid, int(s1), p1)
        except Exception:
            a1 = None
        try:
            a2 = _read_and_window_cached_uid_slice(uid, int(s2), p2)
        except Exception:
            a2 = None

        if a0 is None:
            a0 = np.zeros((512, 512), dtype=np.uint8)
        if a1 is None:
            a1 = np.zeros((512, 512), dtype=np.uint8)
        if a2 is None:
            a2 = np.zeros((512, 512), dtype=np.uint8)

        stacked_img = np.stack((a0, a1, a2), axis=-1)
        stacked_img = cv2.resize(
            stacked_img,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_AREA,
        )
        X = img2tensor((stacked_img.astype(np.float32) / 255.0 - mean) / std)
        return X, uid, pos


class CSFInstanceDataset(Dataset):
    def __init__(self, feature_array_dict, study_id_list, seq_len):
        self.feature_array_dict = feature_array_dict
        self.study_id_list = study_id_list
        self.seq_len = int(seq_len)

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




## === cell 8
class ConvNextCNN_B_Feature(nn.Module):
    def __init__(self):
        super().__init__()
        m = convnext_base(weights=None)  # keep architecture; no external downloads
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lv1_model = ConvNextCNN_B_Feature().to(device).eval()
lv2_model = (
    CSFNet(input_len=config["feature_size"], lstm_size=config["lstm_size"])
    .to(device)
    .eval()
)

lv1_w = "../input/cnn-lstm-oct-24/run_1/run_1/model_1.pth"
lv2_w = "../input/cnn-lstm-oct-21/run_5b/run_5b/model_lstm_3.pth"

if os.path.exists(lv1_w):
    lv1_model.load_state_dict(torch.load(lv1_w, map_location=device))
if os.path.exists(lv2_w):
    lv2_model.load_state_dict(torch.load(lv2_w, map_location=device))



## === cell 10
submission_dict = {"row_id": [], "fractured": []}
feature_array_dict = {}

torch.set_grad_enabled(False)

uid2count = {}
all_items = []

uid2base = {uid: os.path.join(TEST_PATH, uid) for uid in study_id_list}
uid2pos = {}
for uid in study_id_list:
    img_list = selected_image_dict.get(uid, [])
    uid2count[uid] = len(img_list)
    uid2pos[uid] = {int(s): i for i, s in enumerate(img_list)}

uid2feat_arr = {}
for uid in study_id_list:
    n = uid2count.get(uid, 0)
    if n > 0:
        uid2feat_arr[uid] = np.zeros((n, config["feature_size"]), dtype=np.float32)

uid2pred_sum = {uid: np.zeros((7,), dtype=np.float64) for uid in study_id_list}
uid2pred_n = {uid: 0 for uid in study_id_list}

for uid in study_id_list:
    image_list = selected_image_dict.get(uid, [])
    if len(image_list) == 0:
        continue
    base_path = uid2base[uid]
    neigh = uid2neighbors.get(uid, {})
    pos_map = uid2pos[uid]
    for s in image_list:
        s = int(s)
        s0, s1, s2 = neigh[s]
        p0 = os.path.join(base_path, f"{s0}.dcm")
        p1 = os.path.join(base_path, f"{s1}.dcm")
        p2 = os.path.join(base_path, f"{s2}.dcm")
        all_items.append((uid, (p0, p1, p2), (s0, s1, s2), pos_map[s]))


def _collate_stage1(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    uids = [b[1] for b in batch]
    pos = [b[2] for b in batch]
    return imgs, uids, pos


if len(all_items) > 0:
    ds_all = CSFAllSlicesDataset(all_items, target_size=config["target_size"])
    dl_all = DataLoader(
        ds_all,
        batch_size=config["batch_size_image_level"],
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        num_workers=config["num_workers_image_level"],
        persistent_workers=(config["num_workers_image_level"] > 0),
        prefetch_factor=2 if config["num_workers_image_level"] > 0 else None,
        collate_fn=_collate_stage1,
    )

    with torch.inference_mode():
        for images, uids, pos_list in tqdm(
            dl_all, desc="Stage1 all-slices", total=len(dl_all)
        ):
            images = images.to(device, non_blocking=True)
            features, logits = lv1_model(images)
            probs = torch.sigmoid(logits)

            feats_np = features.detach().cpu().numpy()
            probs_np = probs.detach().cpu().numpy()  # (B,7)

            for i, uid in enumerate(uids):
                uid2feat_arr[uid][pos_list[i]] = feats_np[i]
                uid2pred_sum[uid] += probs_np[i].astype(np.float64, copy=False)
                uid2pred_n[uid] += 1


for uid in tqdm(
    study_id_list, desc="Finalize Stage1 per-UID", total=len(study_id_list)
):
    n = uid2count.get(uid, 0)
    if n == 0:
        feature_array_dict[uid] = np.zeros(
            (1, config["feature_size"]), dtype=np.float32
        )
        for k in range(7):
            submission_dict["row_id"].append(f"{uid}_C{k+1}")
            submission_dict["fractured"].append(0.5)
        continue

    feature_array_dict[uid] = uid2feat_arr[uid]

    denom = max(1, uid2pred_n[uid])
    mean_preds = (uid2pred_sum[uid] / denom).astype(np.float32)
    for k in range(7):
        submission_dict["row_id"].append(f"{uid}_C{k+1}")
        submission_dict["fractured"].append(float(mean_preds[k]))



## === cell 11
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
    num_workers=config["num_workers_patient_level"],
)

with torch.inference_mode():
    for features, list_uid in tqdm(
        generator, total=len(generator), desc="Stage2 patient_overall"
    ):
        features = features.to(device, non_blocking=True)
        logits = lv2_model(features)
        preds = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)

        for j in range(len(list_uid)):
            submission_dict["row_id"].append(f"{list_uid[j]}_patient_overall")
            submission_dict["fractured"].append(float(preds[j]))



## === cell 12
sub_df = pd.DataFrame.from_dict(submission_dict)
sub_df["fractured"] = sub_df["fractured"].astype(np.float32)
sub_df["fractured"] = sub_df["fractured"].clip(1e-6, 1 - 1e-6)

required = test_df[["row_id"]].copy()
sub_df = required.merge(sub_df, on="row_id", how="left")

sub_df["fractured"] = sub_df["fractured"].fillna(0.5).astype(np.float32)
sub_df.to_csv("submission.csv", index=False)

print(sub_df.head())
print(
    f"Wrote submission.csv with shape={sub_df.shape} at {os.path.abspath('submission.csv')}"
)
assert sub_df.shape[0] == test_df.shape[0]
assert list(sub_df.columns) == ["row_id", "fractured"]
assert str(os.path.abspath("submission.csv")).endswith(".csv")
