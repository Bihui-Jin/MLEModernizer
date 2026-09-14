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
import os, sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import glob
from tqdm import tqdm

import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.convnext import convnext_base, ConvNeXt_Base_Weights

import pydicom

from albumentations import Compose, CenterCrop



## === cell 1
try:
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target_soft = min(max(soft, 4096), hard)
    if target_soft != soft:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target_soft, hard))
    print("RLIMIT_NOFILE:", resource.getrlimit(resource.RLIMIT_NOFILE))
except Exception as e:
    print("RLIMIT_NOFILE not set:", repr(e))

try:
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
except Exception:
    pass

torch.backends.cudnn.benchmark = True
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)



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
    df_test = pd.read_csv(
        f"../input/rsna-2022-cervical-spine-fracture-detection/test.csv"
    )

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




## === cell 4
test_df = load_df_test()
TEST_PATH = "../input/rsna-2022-cervical-spine-fracture-detection/test_images"
study_id_list = list(test_df.StudyInstanceUID.unique())
print("Unique studies in test:", len(study_id_list))
print("First 3:", study_id_list[:3])




## === cell 5
def _list_dcms_sorted_numeric(folder: str):
    try:
        with os.scandir(folder) as it:
            items = []
            for e in it:
                if not e.is_file():
                    continue
                name = e.name
                if not name.endswith(".dcm"):
                    continue
                stem = name[:-4]
                try:
                    k = int(stem)
                except Exception:
                    continue
                items.append((k, e.path))
        items.sort(key=lambda x: x[0])
        return [p for _, p in items]
    except FileNotFoundError:
        return []


dicom_paths_by_uid = {}
selected_index_dict = {}

for uid in study_id_list:
    paths = _list_dcms_sorted_numeric(os.path.join(TEST_PATH, uid))
    dicom_paths_by_uid[uid] = paths

    n = len(paths)
    if n == 0:
        selected_index_dict[uid] = []
        continue

    middle = n // 2
    num_left = num_right = max(1, int(0.15 * n))
    left_start = max(1, middle - num_left)  # keep room for -1 neighbor
    left_end = middle  # exclusive
    right_start = middle + 1
    right_end = min(n - 2, middle + num_right)  # keep room for +1 neighbor

    idxs = list(range(left_start, left_end)) + list(range(right_start, right_end + 1))
    idxs = [i for i in sorted(set(idxs)) if 1 <= i <= n - 2]
    selected_index_dict[uid] = idxs

if len(study_id_list) > 0:
    print("Example selected indices:", selected_index_dict[study_id_list[0]][:20])



## === cell 6
try:
    pydicom.config.settings.reading_validation_mode = getattr(
        pydicom.config, "IGNORE", 0
    )
except Exception:
    pass

try:
    pydicom.config.image_handlers = [
        h for h in pydicom.config.image_handlers if h is not None
    ]
except Exception:
    pass




## === cell 7
def window(data, WL=400, WW=1800):
    try:
        if hasattr(data, "pixel_array_options"):
            data.pixel_array_options(use_v2_backend=True)
    except Exception:
        pass

    slope = float(getattr(data, "RescaleSlope", 1.0))
    intercept = float(getattr(data, "RescaleIntercept", 0.0))

    img = data.pixel_array.astype(np.float32)
    img = img * slope + intercept

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




## === cell 8
mean = np.array([0.456, 0.456, 0.456], dtype=np.float32)
std = np.array([0.224, 0.224, 0.224], dtype=np.float32)



## === cell 9
from collections import OrderedDict


def _center_crop_np(img: np.ndarray, crop_h: int, crop_w: int):
    h, w = img.shape[:2]
    ch = min(crop_h, h)
    cw = min(crop_w, w)
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    return img[y0 : y0 + ch, x0 : x0 + cw]


class CSFImageDataset(Dataset):
    def __init__(self, uid, selected_indices, target_size, crop_size, cache_size=256):
        self.uid = uid
        self.selected_indices = selected_indices
        self.target_size = int(target_size)
        self.crop_size = int(crop_size)
        self.paths = dicom_paths_by_uid[uid]

        self.cache_size = int(cache_size)
        self._cache = OrderedDict()  # path -> uint8 image

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_cache"] = OrderedDict()
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self._cache is None:
            self._cache = OrderedDict()

    def __len__(self):
        return len(self.selected_indices)

    def _cache_get(self, path):
        if self.cache_size <= 0:
            return None
        v = self._cache.get(path, None)
        if v is not None:
            self._cache.move_to_end(path, last=True)
        return v

    def _cache_put(self, path, value):
        if self.cache_size <= 0:
            return
        self._cache[path] = value
        self._cache.move_to_end(path, last=True)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def _safe_read_window(self, path):
        cached = self._cache_get(path)
        if cached is not None:
            return cached
        try:
            ds = pydicom.dcmread(
                path,
                stop_before_pixels=False,
                force=True,
                defer_size="1 KB",
                specific_tags=[
                    "PixelData",
                    "RescaleSlope",
                    "RescaleIntercept",
                    "BitsAllocated",
                    "BitsStored",
                    "HighBit",
                    "PixelRepresentation",
                    "SamplesPerPixel",
                    "PhotometricInterpretation",
                    "PlanarConfiguration",
                    "Rows",
                    "Columns",
                    "TransferSyntaxUID",
                ],
            )
            img = window(ds)
        except Exception:
            img = np.zeros((512, 512), dtype=np.uint8)
        self._cache_put(path, img)
        return img

    def __getitem__(self, index):
        i = self.selected_indices[index]
        i0 = max(0, i - 1)
        i1 = i
        i2 = min(len(self.paths) - 1, i + 1)

        imgs = (
            self._safe_read_window(self.paths[i0]),
            self._safe_read_window(self.paths[i1]),
            self._safe_read_window(self.paths[i2]),
        )

        stacked_img = np.stack(imgs, axis=-1)
        stacked_img = cv2.resize(
            stacked_img,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_AREA,
        )

        stacked_img = _center_crop_np(stacked_img, self.crop_size, self.crop_size)
        X = stacked_img.astype(np.float32) / 255.0
        X = (X - mean) / std
        X = img2tensor(X)
        return X




## === cell 10
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
            uid, np.zeros((1, config["feature_size"]), dtype=np.float32)
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
        X = torch.as_tensor(x, dtype=torch.float32)
        return X, uid




## === cell 11
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




## === cell 12
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
lv1_model = ConvNextCNN_B_Feature().to(device)
lv1_model.eval()

lv2_model = CSFNet(input_len=config["feature_size"], lstm_size=config["lstm_size"]).to(
    device
)
lv2_model.eval()

lv1_w_path = "../input/cnn-lstm-oct-21/run_3/run_3/model_3.pth"
lv2_w_path = "../input/cnn-lstm-oct-21/run_4/run_4/model_lstm_3.pth"

if os.path.exists(lv1_w_path):
    sd = torch.load(lv1_w_path, map_location="cpu")
    lv1_model.load_state_dict(sd, strict=True)

if os.path.exists(lv2_w_path):
    sd = torch.load(lv2_w_path, map_location="cpu")
    lv2_model.load_state_dict(sd, strict=True)

print("lv1 weights loaded:", os.path.exists(lv1_w_path))
print("lv2 weights loaded:", os.path.exists(lv2_w_path))




## === cell 14
def _seed_worker(worker_id):
    seed = 0 + worker_id
    np.random.seed(seed)
    torch.manual_seed(seed)


submission_dict = {"row_id": [], "fractured": []}
feature_array_dict = {}

vertebrae = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]

sigmoid = torch.sigmoid

from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict

_TARGET = int(config["target_size"])
_CROP = int(config["crop_size"])
_FEAT = int(config["feature_size"])
_BS1 = int(config["batch_size_image_level"])

_MEAN = mean.reshape(1, 1, 3).astype(np.float32, copy=False)
_STD = std.reshape(1, 1, 3).astype(np.float32, copy=False)

_resize = cv2.resize
_crop = _center_crop_np

_DCM_TAGS = [
    "PixelData",
    "RescaleSlope",
    "RescaleIntercept",
    "BitsAllocated",
    "BitsStored",
    "HighBit",
    "PixelRepresentation",
    "SamplesPerPixel",
    "PhotometricInterpretation",
    "PlanarConfiguration",
    "Rows",
    "Columns",
    "TransferSyntaxUID",
]

_MAX_IO_WORKERS = min(8, max(1, (os.cpu_count() or 1)))

try:
    pydicom.config.settings.use_ds_store = True
except Exception:
    pass

needed_indices_dict = {}
for uid in study_id_list:
    paths = dicom_paths_by_uid.get(uid, [])
    sel = selected_index_dict.get(uid, [])
    n = len(paths)
    if n == 0 or len(sel) == 0:
        needed_indices_dict[uid] = []
        continue
    needed = set()
    for i in sel:
        needed.add(max(0, i - 1))
        needed.add(i)
        needed.add(min(n - 1, i + 1))
    needed_indices_dict[uid] = sorted(needed)

_GLOBAL_DCM_CACHE_MAX = 8192  # bounded memory; still large enough for reuse bursts
_global_dcm_cache = OrderedDict()  # path -> uint8(512,512)


def _global_cache_get(path: str):
    v = _global_dcm_cache.get(path, None)
    if v is not None:
        _global_dcm_cache.move_to_end(path, last=True)
    return v


def _global_cache_put(path: str, img: np.ndarray):
    _global_dcm_cache[path] = img
    _global_dcm_cache.move_to_end(path, last=True)
    if len(_global_dcm_cache) > _GLOBAL_DCM_CACHE_MAX:
        _global_dcm_cache.popitem(last=False)


def _safe_read_window_fast(path: str):
    cached = _global_cache_get(path)
    if cached is not None:
        return cached
    try:
        ds = pydicom.dcmread(
            path,
            stop_before_pixels=False,
            force=True,
            defer_size="1 KB",
            specific_tags=_DCM_TAGS,
        )
        img = window(ds)
    except Exception:
        img = np.zeros((512, 512), dtype=np.uint8)
    _global_cache_put(path, img)
    return img


_IO_POOL = ThreadPoolExecutor(max_workers=_MAX_IO_WORKERS)


def _run_stage1_uid(uid: str, selected_indices, needed_indices):
    paths = dicom_paths_by_uid.get(uid, [])
    n = len(paths)
    m = len(selected_indices)
    if n == 0 or m == 0:
        return np.zeros((1, _FEAT), dtype=np.float32), np.zeros((7,), dtype=np.float32)

    imgs_by_idx = {}

    if len(needed_indices) > 0:
        if _MAX_IO_WORKERS > 1 and len(needed_indices) >= 6:
            needed_paths = [paths[j] for j in needed_indices]
            for idx, img in zip(
                needed_indices,
                _IO_POOL.map(_safe_read_window_fast, needed_paths, chunksize=16),
            ):
                imgs_by_idx[idx] = img
        else:
            for idx in needed_indices:
                imgs_by_idx[idx] = _safe_read_window_fast(paths[idx])

    feature_array = np.empty((m, _FEAT), dtype=np.float32)
    pred_sum = np.zeros((7,), dtype=np.float64)
    pred_count = 0

    with torch.inference_mode():
        for start in range(0, m, _BS1):
            end = min(start + _BS1, m)
            bs = end - start

            batch = np.empty((bs, _CROP, _CROP, 3), dtype=np.float32)

            for out_i, si in enumerate(range(start, end)):
                i = selected_indices[si]
                i0 = max(0, i - 1)
                i1 = i
                i2 = min(n - 1, i + 1)

                im0 = imgs_by_idx.get(i0)
                if im0 is None:
                    im0 = _safe_read_window_fast(paths[i0])
                    imgs_by_idx[i0] = im0
                im1 = imgs_by_idx.get(i1)
                if im1 is None:
                    im1 = _safe_read_window_fast(paths[i1])
                    imgs_by_idx[i1] = im1
                im2 = imgs_by_idx.get(i2)
                if im2 is None:
                    im2 = _safe_read_window_fast(paths[i2])
                    imgs_by_idx[i2] = im2

                stacked_img = np.stack((im0, im1, im2), axis=-1)  # uint8 HxWx3
                stacked_img = _resize(
                    stacked_img, (_TARGET, _TARGET), interpolation=cv2.INTER_AREA
                )
                stacked_img = _crop(stacked_img, _CROP, _CROP)

                X = stacked_img.astype(np.float32, copy=False) * (1.0 / 255.0)
                X = (X - _MEAN) / _STD
                batch[out_i] = X

            batch = np.transpose(batch, (0, 3, 1, 2))  # NCHW
            images = torch.from_numpy(np.ascontiguousarray(batch)).to(
                device, non_blocking=True
            )

            features, preds = lv1_model(images)
            feature_array[start:end] = features.detach().cpu().float().numpy()
            batch_probs = sigmoid(preds).detach().cpu().float().numpy()
            pred_sum += batch_probs.sum(axis=0, dtype=np.float64)
            pred_count += batch_probs.shape[0]

    mean_preds = (pred_sum / max(pred_count, 1)).astype(np.float32)
    return feature_array, mean_preds


for uid in tqdm(study_id_list, desc="Stage1 studies", mininterval=1.0, smoothing=0.0):
    selected_indices = selected_index_dict.get(uid, [])
    paths = dicom_paths_by_uid.get(uid, [])
    needed_indices = needed_indices_dict.get(uid, [])

    if (len(selected_indices) == 0) or (len(paths) == 0):
        feature_array_dict[uid] = np.zeros((1, _FEAT), dtype=np.float32)
        for c in vertebrae:
            submission_dict["row_id"].append(f"{uid}_{c}")
            submission_dict["fractured"].append(0.0)
        continue

    try:
        feature_array, mean_preds = _run_stage1_uid(
            uid, selected_indices, needed_indices
        )
    except Exception as e:
        print(f"[WARN] Stage1 failed for uid={uid}: {repr(e)}")
        print(
            "[WARN] Falling back to original Dataset+DataLoader (slower but equivalent)."
        )
        dataset = CSFImageDataset(
            uid=uid,
            selected_indices=selected_indices,
            target_size=config["target_size"],
            crop_size=config["crop_size"],
            cache_size=256,
        )

        def _make_loader(ds, nw: int):
            return DataLoader(
                ds,
                batch_size=config["batch_size_image_level"],
                shuffle=False,
                pin_memory=torch.cuda.is_available(),
                drop_last=False,
                num_workers=nw,
                persistent_workers=(nw > 0),
                prefetch_factor=2 if nw > 0 else None,
                worker_init_fn=_seed_worker if nw > 0 else None,
            )

        num_workers_stage1 = (
            0
            if torch.cuda.is_available()
            else min(4, max(1, (os.cpu_count() or 1) // 2))
        )
        loader = _make_loader(dataset, num_workers_stage1)
        n_items = len(dataset)
        feature_array = np.empty((n_items, _FEAT), dtype=np.float32)
        pred_sum = np.zeros((7,), dtype=np.float64)
        pred_count = 0
        bs = int(config["batch_size_image_level"])
        with torch.inference_mode():
            for bi, images in enumerate(loader):
                start = bi * bs
                end = min(start + bs, n_items)
                images = images.to(device, non_blocking=True)
                features, preds = lv1_model(images)
                feature_array[start:end] = features.detach().cpu().float().numpy()
                batch_probs = sigmoid(preds).detach().cpu().float().numpy()
                pred_sum += batch_probs.sum(axis=0, dtype=np.float64)
                pred_count += batch_probs.shape[0]
        mean_preds = (pred_sum / max(pred_count, 1)).astype(np.float32)
        del dataset

    feature_array_dict[uid] = feature_array
    for k, c in enumerate(vertebrae):
        submission_dict["row_id"].append(f"{uid}_{c}")
        submission_dict["fractured"].append(float(mean_preds[k]))

_IO_POOL.shutdown(wait=True)



## === cell 15
num_workers_stage2 = 0

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
    num_workers=num_workers_stage2,
    persistent_workers=False,
)

with torch.inference_mode():
    for features, list_uid in tqdm(
        generator2,
        total=len(generator2),
        desc="Stage2 patient_overall",
        mininterval=1.0,
        smoothing=0.0,
    ):
        features = features.to(device, non_blocking=True)
        preds = torch.sigmoid(lv2_model(features)).detach().cpu().numpy().reshape(-1)
        for j in range(len(list_uid)):
            submission_dict["row_id"].append(f"{list_uid[j]}_patient_overall")
            submission_dict["fractured"].append(float(preds[j]))



## === cell 16
sub_df = pd.DataFrame.from_dict(submission_dict)

sub_df = sub_df.drop_duplicates(subset=["row_id"], keep="last")

required = test_df[["row_id"]].copy()
sub_df = required.merge(sub_df, on="row_id", how="left")

sub_df["fractured"] = sub_df["fractured"].fillna(0.0).astype(np.float32)
sub_df["fractured"] = np.clip(sub_df["fractured"].values, 1e-6, 1 - 1e-6)

print(sub_df.head())
print("Submission shape:", sub_df.shape)
assert list(sub_df.columns) == ["row_id", "fractured"]
assert len(sub_df) == len(test_df)



## === cell 17
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with rows:", len(sub_df))
print(sub_df.tail())
