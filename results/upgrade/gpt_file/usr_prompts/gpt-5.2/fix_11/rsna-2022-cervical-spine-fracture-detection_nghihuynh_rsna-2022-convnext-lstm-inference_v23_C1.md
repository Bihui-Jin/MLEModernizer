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

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")



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




## === cell 2
def seed_everything(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

DATA_ROOT = "../input/rsna-2022-cervical-spine-fracture-detection"
TEST_PATH = f"{DATA_ROOT}/test_images"



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
def load_df_test():
    df_test = pd.read_csv(f"{DATA_ROOT}/test.csv")

    if df_test.iloc[0].row_id == "1.2.826.0.1.3680043.10197_C1":
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




## === cell 5
def list_dicom_numbers(uid: str):
    folder = os.path.join(TEST_PATH, uid)
    try:
        entries = os.listdir(folder)
    except FileNotFoundError:
        return []
    nums = []
    for name in entries:
        if name.endswith(".dcm"):
            stem = name[:-4]
            if stem.isdigit():
                nums.append(int(stem))
            else:
                try:
                    nums.append(int(stem))
                except Exception:
                    continue
    nums.sort()
    return nums


selected_image_dict = {}
dicom_num_dict = {}

for uid in study_id_list:
    nums = list_dicom_numbers(uid)
    dicom_num_dict[uid] = nums

    if len(nums) < 3:
        selected_image_dict[uid] = nums
        continue

    mid_pos = len(nums) // 2
    k = int(0.15 * len(nums))
    left = list(range(max(0, mid_pos - k), mid_pos))
    right = list(range(mid_pos + 1, min(len(nums), mid_pos + 1 + k)))
    sel_pos = left + right
    selected_image_dict[uid] = [nums[p] for p in sel_pos]



## === cell 6
_DICOM_TAGS = [
    "PixelData",
    "RescaleSlope",
    "RescaleIntercept",
    "TransferSyntaxUID",
]


def safe_pixel_array(ds: pydicom.dataset.FileDataset):
    try:
        return ds.pixel_array
    except Exception:
        pass

    ts = str(getattr(ds.file_meta, "TransferSyntaxUID", "") or "")

    if ts.startswith("1.2.840.10008.1.2.4"):
        try:
            from pydicom.encaps import generate_pixel_data_frame

            frame = next(generate_pixel_data_frame(ds.PixelData, 0))
            arr = np.frombuffer(frame, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if img is not None:
                if img.ndim == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                return img
        except Exception:
            pass

    try:
        ds.decompress()
        return ds.pixel_array
    except Exception as e:
        raise RuntimeError(f"Failed to decode DICOM pixel data: {e}") from e


def window_from_ds(data, WL=400, WW=1800):
    slope = float(getattr(data, "RescaleSlope", 1.0))
    intercept = float(getattr(data, "RescaleIntercept", 0.0))

    img = safe_pixel_array(data).astype(np.float32, copy=False)
    if slope != 1.0:
        img = img * slope
    if intercept != 0.0:
        img = img + intercept

    upper, lower = WL + WW // 2, WL - WW // 2
    X = np.clip(img, lower, upper)

    mn, mx = cv2.minMaxLoc(X)[:2]
    X = X - mn
    if mx > mn:
        X = X / (mx - mn)
    X = (X * 255.0).astype(np.uint8, copy=False)
    return X


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = img[:, :, None]
    img = np.transpose(img, (2, 0, 1))
    if img.dtype != dtype:
        img = img.astype(dtype, copy=False)
    return torch.from_numpy(img)




## === cell 7
mean = np.array([0.456, 0.456, 0.456], dtype=np.float32)
std = np.array([0.224, 0.224, 0.224], dtype=np.float32)


def center_crop_np(img: np.ndarray, crop_size: int) -> np.ndarray:
    h, w = img.shape[:2]
    ch = crop_size
    cw = crop_size
    if h == ch and w == cw:
        return img
    top = max((h - ch) // 2, 0)
    left = max((w - cw) // 2, 0)
    return img[top : top + ch, left : left + cw]


class CSFImageDatasetGlobal(Dataset):
    __slots__ = ("items", "target_size", "crop_size")

    def __init__(self, items, target_size, crop_size):
        """
        items: list of tuples (uid, n0, n1, n2)
        """
        self.items = items
        self.target_size = target_size
        self.crop_size = crop_size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        uid, n0, n1, n2 = self.items[index]
        folder = os.path.join(TEST_PATH, uid)
        try:
            ds0 = pydicom.dcmread(
                os.path.join(folder, f"{int(n0)}.dcm"),
                force=True,
                specific_tags=_DICOM_TAGS,
            )
            ds1 = pydicom.dcmread(
                os.path.join(folder, f"{int(n1)}.dcm"),
                force=True,
                specific_tags=_DICOM_TAGS,
            )
            ds2 = pydicom.dcmread(
                os.path.join(folder, f"{int(n2)}.dcm"),
                force=True,
                specific_tags=_DICOM_TAGS,
            )

            img0 = window_from_ds(ds0)
            img1 = window_from_ds(ds1)
            img2 = window_from_ds(ds2)

            stacked_img = np.stack((img0, img1, img2), axis=-1)  # H,W,3
            stacked_img = cv2.resize(
                stacked_img,
                (self.target_size, self.target_size),
                interpolation=cv2.INTER_LINEAR,
            )
            stacked_img = center_crop_np(stacked_img, self.crop_size)

            X = stacked_img.astype(np.float32) * (1.0 / 255.0)
            X = (X - mean) / std
            X = img2tensor(X, dtype=np.float32)
            return X, uid
        except Exception:
            X = torch.zeros(
                (3, config["crop_size"], config["crop_size"]), dtype=torch.float32
            )
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




## === cell 8
class ConvNextCNN_B_Feature(nn.Module):
    def __init__(self):
        super(ConvNextCNN_B_Feature, self).__init__()
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
def try_load_weights(model, path: str):
    if os.path.exists(path):
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        return True
    return False


lv1_model = ConvNextCNN_B_Feature()
lv2_model = CSFNet(input_len=config["feature_size"], lstm_size=config["lstm_size"])

lv1_path = "../input/cnn-lstm-oct-25/run_2/run_2/model_2.pth"
lv2_path = "../input/cnn-lstm-oct-25/run_2/run_2/model_lstm_2.pth"

lv1_loaded = try_load_weights(lv1_model, lv1_path)
lv2_loaded = try_load_weights(lv2_model, lv2_path)

lv1_model = lv1_model.to(device).eval()
lv2_model = lv2_model.to(device).eval()

print(f"lv1 weights loaded: {lv1_loaded} ({lv1_path})")
print(f"lv2 weights loaded: {lv2_loaded} ({lv2_path})")



## === cell 10
feature_array_dict = {}
preds_clevel_dict = {}  # uid -> np.array shape (7,)
uids_ok = []

global_items = []
uid_selected_counts = {}  # uid -> number of selected positions
uids_missing = []


def nearest_existing_factory(nums_list):
    nums = np.asarray(nums_list, dtype=np.int32)

    def nearest_existing(num: int) -> int:
        pos = int(np.searchsorted(nums, num, side="left"))
        if pos <= 0:
            return int(nums[0])
        if pos >= nums.shape[0]:
            return int(nums[-1])
        before = nums[pos - 1]
        after = nums[pos]
        if (num - before) <= (after - num):
            return int(before)
        return int(after)

    return nearest_existing


for uid in study_id_list:
    image_list = selected_image_dict.get(uid, [])
    dicom_numbers = dicom_num_dict.get(uid, [])

    if len(image_list) == 0 or len(dicom_numbers) == 0:
        uids_missing.append(uid)
        continue

    nearest_existing = nearest_existing_factory(dicom_numbers)

    cnt = 0
    for c in image_list:
        c = int(c)
        n0, n1, n2 = (
            nearest_existing(c - 1),
            nearest_existing(c),
            nearest_existing(c + 1),
        )
        global_items.append((uid, n0, n1, n2))
        cnt += 1
    uid_selected_counts[uid] = cnt

for uid in uids_missing:
    preds_clevel_dict[uid] = np.full((7,), 0.5, dtype=np.float32)
    feature_array_dict[uid] = np.zeros((1, config["feature_size"]), dtype=np.float32)
    uids_ok.append(uid)

for uid, cnt in uid_selected_counts.items():
    feature_array_dict[uid] = np.zeros((cnt, config["feature_size"]), dtype=np.float32)

pred_sum_dict = {uid: np.zeros((7,), dtype=np.float64) for uid in uid_selected_counts}
pred_cnt_dict = {uid: 0 for uid in uid_selected_counts}
feat_write_idx = {uid: 0 for uid in uid_selected_counts}

dataset_global = CSFImageDatasetGlobal(
    items=global_items,
    target_size=config["target_size"],
    crop_size=config["crop_size"],
)

cpu_cnt = os.cpu_count() or 1
if cpu_cnt <= 4:
    num_workers_stage1 = 0
else:
    num_workers_stage1 = min(4, cpu_cnt - 2)

generator = DataLoader(
    dataset_global,
    batch_size=config["batch_size_image_level"],
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
    num_workers=num_workers_stage1,
    prefetch_factor=2 if num_workers_stage1 > 0 else None,
    persistent_workers=True if num_workers_stage1 > 0 else False,
)

bad_uid = set()

for images, uids in tqdm(generator, total=len(generator), desc="Stage1 global"):
    try:
        with torch.inference_mode():
            images = images.to(device, non_blocking=True)
            features, preds = lv1_model(images)
            preds = torch.sigmoid(preds).cpu().numpy()
            features = features.cpu().numpy()
    except Exception:
        for uid in uids:
            bad_uid.add(str(uid))
        continue

    for i, uid in enumerate(uids):
        uid = str(uid)
        if uid in bad_uid or uid not in feat_write_idx:
            continue

        wi = feat_write_idx[uid]
        if wi < feature_array_dict[uid].shape[0]:
            feature_array_dict[uid][wi] = features[i]
            feat_write_idx[uid] = wi + 1

        pred_sum_dict[uid] += preds[i].astype(np.float64, copy=False)
        pred_cnt_dict[uid] += 1

for uid in uid_selected_counts:
    if uid in bad_uid or pred_cnt_dict[uid] == 0 or feat_write_idx[uid] == 0:
        preds_clevel_dict[uid] = np.full((7,), 0.5, dtype=np.float32)
        feature_array_dict[uid] = np.zeros(
            (1, config["feature_size"]), dtype=np.float32
        )
    else:
        w = feat_write_idx[uid]
        if w != feature_array_dict[uid].shape[0]:
            feature_array_dict[uid] = feature_array_dict[uid][:w]
        mean_preds = (pred_sum_dict[uid] / float(pred_cnt_dict[uid])).astype(
            np.float32, copy=False
        )
        preds_clevel_dict[uid] = mean_preds
    uids_ok.append(uid)



## === cell 11
dataset2 = CSFInstanceDataset(
    feature_array_dict=feature_array_dict,
    study_id_list=uids_ok,
    seq_len=config["seq_len"],
)

num_workers_stage2 = 0 if (os.cpu_count() or 1) <= 2 else 1
generator2 = DataLoader(
    dataset=dataset2,
    batch_size=config["batch_size_patient_level"],
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
    num_workers=num_workers_stage2,
    prefetch_factor=2 if num_workers_stage2 > 0 else None,
    persistent_workers=True if num_workers_stage2 > 0 else False,
)

preds_overall_dict = {}
for features, list_uid in tqdm(
    generator2, total=len(generator2), desc="Stage2 patient_overall"
):
    with torch.inference_mode():
        features = features.to(device, non_blocking=True)
        preds = torch.sigmoid(lv2_model(features)).squeeze(-1).cpu().numpy()

    preds = np.asarray(preds).reshape(-1)
    for j in range(len(list_uid)):
        preds_overall_dict[str(list_uid[j])] = float(preds[j])




## === cell 12
def get_pred_for_row(uid: str, ptype: str):
    if ptype == "patient_overall":
        return float(preds_overall_dict.get(uid, 0.5))
    arr = preds_clevel_dict.get(uid, None)
    if arr is None:
        return 0.5
    idx = int(ptype.replace("C", "")) - 1
    return float(arr[idx])


sub = test_df[["row_id", "StudyInstanceUID", "prediction_type"]].copy()
sub["fractured"] = [
    get_pred_for_row(uid, ptype)
    for uid, ptype in zip(
        sub["StudyInstanceUID"].astype(str), sub["prediction_type"].astype(str)
    )
]

sub["fractured"] = sub["fractured"].astype(np.float32).clip(1e-6, 1 - 1e-6)
sub = sub[["row_id", "fractured"]]
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print(
    "Stage1 bad_uid count:",
    len(
        set(
            [
                u
                for u in preds_clevel_dict.keys()
                if np.allclose(preds_clevel_dict[u], 0.5)
            ]
        )
    ),
)
