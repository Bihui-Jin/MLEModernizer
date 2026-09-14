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

os.makedirs("/kaggle/working", exist_ok=True)



## === cell 1
import numpy as np
import pandas as pd
import pydicom

import cv2
from tqdm import tqdm
import torch.nn as nn
from torchvision.models.convnext import convnext_base

from torch.utils.data import Dataset, DataLoader
import torch
import warnings

warnings.filterwarnings("ignore")




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
device



## === cell 3
config = {
    "seq_len": 150,
    "feature_size": 1024,
    "lstm_size": 128,
    "target_size": 320,
    "crop_size": 320,
    "num_classes": 7,
    "batch_size_image_level": 32,
    "batch_size_patient_level": 8,
}




## === cell 4
def load_df_test():
    df_test = pd.read_csv(
        f"/kaggle/input/rsna-2022-cervical-spine-fracture-detection/test.csv"
    )
    return df_test




## === cell 5
test_df = load_df_test()
TEST_PATH = "/kaggle/input/rsna-2022-cervical-spine-fracture-detection/test_images"
study_id_list = list(test_df.StudyInstanceUID.unique())
len(study_id_list), study_id_list[:3]




## === cell 6
def get_sorted_dicom_paths(uid: str):
    folder = os.path.join(TEST_PATH, uid)
    try:
        names = os.listdir(folder)
    except FileNotFoundError:
        return []

    dcm_names = [n for n in names if n.endswith(".dcm")]
    if not dcm_names:
        return []

    nums = []
    non_nums = []
    for n in dcm_names:
        stem = n[:-4]
        if stem.isdigit():
            nums.append((int(stem), n))
        else:
            non_nums.append(n)

    if nums:
        nums.sort(key=lambda x: x[0])
        if non_nums:
            non_nums.sort()
        names_sorted = [n for _, n in nums] + non_nums
    else:
        dcm_names.sort()
        names_sorted = dcm_names

    return [os.path.join(folder, n) for n in names_sorted]


selected_image_dict = {}
dicom_paths_dict = {}

SEQ_LEN = int(config["seq_len"])

for uid in study_id_list:
    dcm_paths = get_sorted_dicom_paths(uid)
    dicom_paths_dict[uid] = dcm_paths
    n = len(dcm_paths)
    if n == 0:
        selected_image_dict[uid] = []
        continue

    if n <= SEQ_LEN:
        selected = list(range(n))
    else:
        center = n // 2
        half = SEQ_LEN // 2
        left = max(0, center - half)
        right = left + SEQ_LEN
        if right > n:
            right = n
            left = n - SEQ_LEN
        selected = list(range(left, right))

    selected_image_dict[uid] = selected

uid0 = study_id_list[0]
len(dicom_paths_dict[uid0]), selected_image_dict[uid0][:5], selected_image_dict[uid0][
    -5:
]



## === cell 7
print(study_id_list[0])
print(selected_image_dict[study_id_list[0]][:10])



## === cell 8
try:
    import pylibjpeg  # noqa: F401
except Exception:
    pass
try:
    import gdcm  # noqa: F401
except Exception:
    pass


def safe_get_pixel_array(ds):
    try:
        arr = ds.pixel_array
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        if slope != 1.0:
            arr = arr.astype(np.float32) * slope
        else:
            arr = arr.astype(np.float32, copy=False)
        if intercept != 0.0:
            arr = arr + intercept
        return arr
    except Exception:
        rows = int(getattr(ds, "Rows", 512))
        cols = int(getattr(ds, "Columns", 512))
        return np.zeros((rows, cols), dtype=np.float32)


def window(ds, WL=400, WW=1800):
    img = safe_get_pixel_array(ds)

    upper = WL + WW // 2
    lower = WL - WW // 2
    x = np.clip(img, lower, upper)

    x_min = float(x.min())
    x = x - x_min
    x_max = float(x.max())
    if x_max > 0.0:
        x = x / x_max
    x = (x * 255.0).astype("uint8")
    return x


def img2tensor(img):
    if img.ndim == 2:
        img = img[:, :, None]
    img = np.transpose(img, (2, 0, 1))
    img = np.ascontiguousarray(img, dtype=np.float32)
    return torch.from_numpy(img)




## === cell 9
mean = np.array([0.456, 0.456, 0.456], dtype=np.float32)
std = np.array([0.224, 0.224, 0.224], dtype=np.float32)

_MEAN = mean
_STD = std
_INV_255 = np.float32(1.0 / 255.0)

import functools


@functools.lru_cache(maxsize=8192)
def _read_and_window_path_cached(path: str):
    try:
        ds = pydicom.dcmread(
            path,
            force=True,
            stop_before_pixels=False,
            specific_tags=[
                "PixelData",
                "Rows",
                "Columns",
                "RescaleSlope",
                "RescaleIntercept",
                "PhotometricInterpretation",
                "BitsAllocated",
                "BitsStored",
                "HighBit",
                "PixelRepresentation",
                "SamplesPerPixel",
                "PlanarConfiguration",
                "TransferSyntaxUID",
            ],
        )
    except Exception:
        ds = None

    if ds is None:
        return np.zeros((512, 512), dtype=np.uint8)

    return window(ds)


class CSFAllSlicesDataset(Dataset):
    def __init__(
        self,
        study_id_list,
        selected_image_dict,
        dicom_paths_dict,
        target_size,
        crop_size,
        cache_slices_per_worker: int = 0,
    ):
        self.study_id_list = study_id_list
        self.selected_image_dict = selected_image_dict
        self.dicom_paths_dict = dicom_paths_dict
        self.target_size = int(target_size)
        self.crop_size = int(crop_size)

        self.index = []  # list of (uid, slice_idx)
        for uid in study_id_list:
            for si in selected_image_dict.get(uid, []):
                self.index.append((uid, int(si)))

        self._cache_limit = int(cache_slices_per_worker)

    def __len__(self):
        return len(self.index)

    @staticmethod
    def _read_and_window_path(path: str):
        return _read_and_window_path_cached(path)

    def __getitem__(self, i):
        uid, center_i = self.index[i]
        dcm_paths = self.dicom_paths_dict.get(uid, [])
        if not dcm_paths:
            stacked_img = np.zeros(
                (self.target_size, self.target_size, 3), dtype=np.uint8
            )
            x = stacked_img.astype(np.float32, copy=False)
            x *= _INV_255
            x -= _MEAN
            x /= _STD
            X = img2tensor(x)
            return X, uid

        last = len(dcm_paths) - 1
        idx0 = 0 if center_i <= 0 else center_i - 1
        idx1 = center_i
        idx2 = last if center_i >= last else center_i + 1

        img0 = self._read_and_window_path(dcm_paths[idx0])
        img1 = self._read_and_window_path(dcm_paths[idx1])
        img2 = self._read_and_window_path(dcm_paths[idx2])

        stacked_img = np.stack((img0, img1, img2), axis=-1)
        stacked_img = cv2.resize(
            stacked_img,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_AREA,
        )

        x = stacked_img.astype(np.float32, copy=False)
        x *= _INV_255
        x -= _MEAN
        x /= _STD
        X = img2tensor(x)
        return X, uid


class CSFInstanceDataset(Dataset):
    def __init__(self, feature_array_dict, study_id_list, seq_len):
        self.feature_array_dict = feature_array_dict
        self.study_id_list = study_id_list
        self.seq_len = seq_len

        self._x_new = np.linspace(0.0, 1.0, self.seq_len, dtype=np.float32)

    def __len__(self):
        return len(self.study_id_list)

    def __getitem__(self, index):
        uid = self.study_id_list[index]
        feature_array = self.feature_array_dict.get(
            uid, np.zeros((1, config["feature_size"]), dtype=np.float32)
        )

        t = feature_array.shape[0]
        if t > self.seq_len:
            src = feature_array.astype(np.float32, copy=False)
            x_old = np.linspace(0.0, 1.0, t, dtype=np.float32)
            x_new = self._x_new

            idx = np.searchsorted(x_old, x_new, side="right") - 1
            idx = np.clip(idx, 0, t - 2)
            x0 = x_old[idx]
            x1 = x_old[idx + 1]
            w = (x_new - x0) / (x1 - x0)
            y0 = src[idx]  # (seq_len, F)
            y1 = src[idx + 1]  # (seq_len, F)
            x = (y0 + (y1 - y0) * w[:, None]).astype(np.float32, copy=False)
        else:
            x = np.pad(
                feature_array,
                pad_width=[(0, self.seq_len - t), (0, 0)],
                constant_values=0,
            ).astype(np.float32, copy=False)

        X = torch.tensor(x, dtype=torch.float32)
        return X, uid




## === cell 10
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




## === cell 11
def try_load_state_dict(model, path):
    if os.path.exists(path):
        sd = torch.load(path, map_location="cpu")
        model.load_state_dict(sd)
        return True
    return False


lv1_model = ConvNextCNN_B_Feature()
lv2_model = CSFNet(input_len=config["feature_size"], lstm_size=config["lstm_size"])

lv1_path = "/kaggle/input/cnn-lstm-exp-8/run_0/model_0.pth"
lv2_path = "/kaggle/input/cnn-lstm-oct-21/run_5b/run_5b/model_lstm_3.pth"

lv1_loaded = try_load_state_dict(lv1_model, lv1_path)
lv2_loaded = try_load_state_dict(lv2_model, lv2_path)

lv1_model = lv1_model.to(device).eval()
lv2_model = lv2_model.to(device).eval()

print("lv1 weights loaded:", lv1_loaded, "path:", lv1_path)
print("lv2 weights loaded:", lv2_loaded, "path:", lv2_path)



## === cell 12
try:
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target_soft = min(max(soft, 8192), hard)
    resource.setrlimit(resource.RLIMIT_NOFILE, (target_soft, hard))
    print("RLIMIT_NOFILE soft/hard:", resource.getrlimit(resource.RLIMIT_NOFILE))
except Exception as e:
    print("Could not adjust RLIMIT_NOFILE:", repr(e))

try:
    cpu_cnt = os.cpu_count() or 2
except Exception:
    cpu_cnt = 2

if torch.cuda.is_available():
    _NUM_WORKERS_IMG = min(4, max(2, cpu_cnt // 4))
else:
    _NUM_WORKERS_IMG = min(2, max(1, cpu_cnt // 4))

_PERSISTENT = _NUM_WORKERS_IMG > 0
_PREFETCH = 2 if _NUM_WORKERS_IMG > 0 else None

all_ds = CSFAllSlicesDataset(
    study_id_list=study_id_list,
    selected_image_dict=selected_image_dict,
    dicom_paths_dict=dicom_paths_dict,
    target_size=config["target_size"],
    crop_size=config["crop_size"],
    cache_slices_per_worker=512,
)

all_loader = DataLoader(
    dataset=all_ds,
    batch_size=config["batch_size_image_level"],
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
    num_workers=_NUM_WORKERS_IMG,
    persistent_workers=_PERSISTENT,
    prefetch_factor=_PREFETCH,
)

uid2row = {uid: i for i, uid in enumerate(study_id_list)}
n_uid = len(study_id_list)

feature_array_dict = {}
preds_level_dict = {}

_uid_pos = np.zeros(n_uid, dtype=np.int32)
_uid_pred_sum = np.zeros((n_uid, 7), dtype=np.float64)
_uid_pred_cnt = np.zeros(n_uid, dtype=np.int32)

for uid in study_id_list:
    nsel = len(selected_image_dict.get(uid, []))
    if nsel == 0:
        feature_array_dict[uid] = np.zeros(
            (1, config["feature_size"]), dtype=np.float32
        )
        preds_level_dict[uid] = np.full((7,), 0.5, dtype=np.float32)
    else:
        feature_array_dict[uid] = np.empty(
            (nsel, config["feature_size"]), dtype=np.float32
        )

with torch.no_grad():
    for images, uids in tqdm(all_loader, total=len(all_loader), desc="Stage1 batched"):
        images = images.to(device, non_blocking=True)
        features, preds = lv1_model(images)
        feats_np = features.detach().cpu().numpy()
        preds_np = preds.sigmoid().detach().cpu().numpy()  # (B,7)

        prev_uid = None
        start = 0
        for b, uid in enumerate(uids):
            if prev_uid is None:
                prev_uid = uid
                start = b
                continue
            if uid != prev_uid:
                r = uid2row[prev_uid]
                bs = slice(start, b)
                k = b - start
                pos = _uid_pos[r]
                feature_array_dict[prev_uid][pos : pos + k] = feats_np[bs]
                _uid_pos[r] = pos + k
                _uid_pred_sum[r] += (
                    preds_np[bs].astype(np.float64, copy=False).sum(axis=0)
                )
                _uid_pred_cnt[r] += k
                prev_uid = uid
                start = b
        if prev_uid is not None:
            r = uid2row[prev_uid]
            bs = slice(start, len(uids))
            k = len(uids) - start
            pos = _uid_pos[r]
            feature_array_dict[prev_uid][pos : pos + k] = feats_np[bs]
            _uid_pos[r] = pos + k
            _uid_pred_sum[r] += preds_np[bs].astype(np.float64, copy=False).sum(axis=0)
            _uid_pred_cnt[r] += k

for uid in study_id_list:
    r = uid2row[uid]
    cnt = int(_uid_pred_cnt[r])
    if cnt > 0:
        preds_level_dict[uid] = (_uid_pred_sum[r] / cnt).astype(np.float32)
    else:
        preds_level_dict[uid] = np.full((7,), 0.5, dtype=np.float32)

del all_loader, all_ds, _uid_pos, _uid_pred_sum, _uid_pred_cnt, uid2row, n_uid



## === cell 13
_NUM_WORKERS_SEQ = 0
_PERSISTENT_SEQ = False

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
    num_workers=_NUM_WORKERS_SEQ,
    persistent_workers=_PERSISTENT_SEQ,
    prefetch_factor=None,
)

preds_patient_dict = {}
for features, list_uid in tqdm(
    generator2, total=len(generator2), desc="Stage2 patient_overall"
):
    with torch.no_grad():
        features = features.to(device, non_blocking=True)
        preds = lv2_model(features)
        preds = preds.sigmoid().detach().cpu().numpy().reshape(-1)
    for j in range(len(list_uid)):
        preds_patient_dict[list_uid[j]] = float(preds[j])

len(preds_patient_dict), list(preds_patient_dict.items())[:2]



## === cell 14
EPS = 1e-6


def clip01(p):
    return float(np.clip(p, EPS, 1.0 - EPS))


pred_map = {}
levels = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]

for uid in study_id_list:
    level_preds = preds_level_dict.get(uid, np.full((7,), 0.5, dtype=np.float32))
    for k, lvl in enumerate(levels):
        pred_map[f"{uid}_{lvl}"] = clip01(level_preds[k])
    pred_map[f"{uid}_patient_overall"] = clip01(preds_patient_dict.get(uid, 0.5))

sub_df = test_df[["row_id"]].copy()
sub_df["fractured"] = sub_df["row_id"].map(pred_map).fillna(0.5).astype(np.float32)
sub_df["fractured"] = sub_df["fractured"].clip(EPS, 1.0 - EPS)

sub_df.head(), sub_df.shape



## === cell 15
sub_path = "/kaggle/working/submission.csv"
sub_df.to_csv(sub_path, index=False)
print("Wrote:", sub_path)
print(pd.read_csv(sub_path).head())
print("Rows:", len(sub_df), "Cols:", list(sub_df.columns))
