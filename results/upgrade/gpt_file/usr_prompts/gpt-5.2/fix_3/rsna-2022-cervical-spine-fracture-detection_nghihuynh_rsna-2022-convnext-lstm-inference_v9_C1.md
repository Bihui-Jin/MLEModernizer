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

# 5. Target score

0.6124399920535397

# 6. Current score

0.69212

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 0.69212) has done: 'The main timeout driver is DICOM I/O and per-slice preprocessing being done serially with repeated object creation (Albumentations `Compose`) and extra DICOM header reads during sorting. I (1) avoid per-file header reads when listing slices by sorting numerically by filename (fallback to InstanceNumber only if needed), (2) cache DICOM reads/windowed arrays within each study so overlapping tri-slice stacks don’t re-decode the same slice 3×, (3) pre-create and reuse the center-crop transform (or numpy crop) instead of rebuilding it per `__getitem__`, and (4) enable DataLoader workers with persistent workers to overlap CPU decode/preprocess with GPU inference without changing model logic or outputs. These changes preserve the same inputs to the models and identical averaging/aggregation semantics, but remove redundant work and improve CPU/GPU pipeline utilization to fit within 600 seconds.'

# 9. Code solution

## === cell 0
import os, glob, sys, time, warnings

warnings.filterwarnings("ignore")



## === cell 1
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



## === cell 2
BASE_PATH = "/kaggle/input/rsna-2022-cervical-spine-fracture-detection"
TEST_CSV_PATH = f"{BASE_PATH}/test.csv"
SAMPLE_SUB_PATH = f"{BASE_PATH}/sample_submission.csv"
TEST_IMG_PATH = f"{BASE_PATH}/test_images"

assert os.path.exists(TEST_CSV_PATH), f"Missing {TEST_CSV_PATH}"
assert os.path.exists(SAMPLE_SUB_PATH), f"Missing {SAMPLE_SUB_PATH}"
assert os.path.exists(TEST_IMG_PATH), f"Missing {TEST_IMG_PATH}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



## === cell 3
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




## === cell 4
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




## === cell 5
test_df = load_df_test()
study_id_list = list(test_df.StudyInstanceUID.unique())
print("Num studies:", len(study_id_list))
print("Example study:", study_id_list[0])




## === cell 6
def _list_dicom_paths(uid: str):
    dcm_paths = glob.glob(os.path.join(TEST_IMG_PATH, uid, "*.dcm"))
    if not dcm_paths:
        return []

    bns = [os.path.basename(p).split(".")[0] for p in dcm_paths]
    all_numeric = True
    nums = []
    for bn in bns:
        if bn.isdigit():
            nums.append(int(bn))
        else:
            all_numeric = False
            break
    if all_numeric:
        return [p for _, p in sorted(zip(nums, dcm_paths), key=lambda t: t[0])]

    def _key(p):
        bn = os.path.basename(p).split(".")[0]
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
            inst = getattr(ds, "InstanceNumber", None)
            if inst is not None:
                return (0, int(inst))
        except Exception:
            pass
        try:
            return (1, int(bn))
        except Exception:
            return (2, bn)

    return sorted(dcm_paths, key=_key)


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




## === cell 7
def window(ds, WL=400, WW=1800):
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    img = ds.pixel_array.astype(np.float32)
    img = img * slope + intercept
    upper, lower = WL + WW // 2, WL - WW // 2
    x = np.clip(img, lower, upper)
    x = x - x.min()
    mx = x.max()
    if mx > 0:
        x = x / mx
    x = (x * 255.0).astype("uint8")
    return x


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


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




## === cell 8
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


class CSFImageDataset(Dataset):
    def __init__(self, uid, image_indices, target_size, crop_size):
        self.uid = uid
        self.image_indices = image_indices
        self.target_size = target_size
        self.crop_size = crop_size

        self._crop = None
        if _HAS_ALB:
            self._crop = Compose([CenterCrop(self.crop_size, self.crop_size)])

        self._win_cache = {}

    def __len__(self):
        return len(self.image_indices)

    def _get_windowed(self, p):
        x = self._win_cache.get(p, None)
        if x is not None:
            return x
        ds = safe_dcmread(p)
        if ds is None or safe_get_pixel(ds) is None:
            x = np.zeros((512, 512), dtype="uint8")
        else:
            try:
                x = window(ds)
            except Exception:
                x = np.zeros((512, 512), dtype="uint8")
        self._win_cache[p] = x
        return x

    def __getitem__(self, index):
        idx = self.image_indices[index]
        dcm_paths = dicom_path_dict[self.uid]
        paths = (dcm_paths[idx - 1], dcm_paths[idx], dcm_paths[idx + 1])

        imgs = [self._get_windowed(p) for p in paths]

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




## === cell 9
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




## === cell 10
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




## === cell 11
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
    torch.manual_seed(0)
    np.random.seed(0)
    print(
        "Warning: pretrained weights not found; using deterministic random init (baseline submission)."
    )



## === cell 12
submission_dict = {"row_id": [], "fractured": []}
feature_array_dict = {}

num_workers = min(4, os.cpu_count() or 1)  # conservative to avoid oversubscription
use_cuda = torch.cuda.is_available()

for uid in tqdm(study_id_list, desc="Stage1 per-study", total=len(study_id_list)):
    image_indices = selected_image_dict.get(uid, [])
    if len(image_indices) == 0:
        feature_array_dict[uid] = np.zeros(
            (config["seq_len"], config["feature_size"]), dtype=np.float32
        )
        for k in ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]:
            submission_dict["row_id"].append(f"{uid}_{k}")
            submission_dict["fractured"].append(0.5)
        continue

    dataset = CSFImageDataset(
        uid=uid,
        image_indices=image_indices,
        target_size=config["target_size"],
        crop_size=config["crop_size"],
    )
    generator = DataLoader(
        dataset,
        batch_size=config["batch_size_image_level"],
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    preds_uid = []
    feature_array = np.zeros((len(dataset), config["feature_size"]), dtype=np.float32)

    for i, images in enumerate(generator):
        with torch.no_grad():
            start = i * config["batch_size_image_level"]
            end = min(start + config["batch_size_image_level"], len(generator.dataset))

            images = images.to(device, non_blocking=True)
            features, preds = lv1_model(images)

            feature_array[start:end] = features.detach().cpu().numpy()
            preds_uid.append(torch.sigmoid(preds).detach().cpu())

    feature_array_dict[uid] = feature_array

    mean_preds = (
        torch.mean(torch.cat(preds_uid, dim=0), dim=0).numpy().astype(np.float32)
    )

    for idx, k in enumerate(["C1", "C2", "C3", "C4", "C5", "C6", "C7"]):
        submission_dict["row_id"].append(f"{uid}_{k}")
        submission_dict["fractured"].append(
            float(np.clip(mean_preds[idx], 1e-6, 1 - 1e-6))
        )



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_55/738160972.py in <cell line: 0>()
     38     feature_array = np.zeros((len(dataset), config["feature_size"]), dtype=np.float32)
     39 
---> 40     for i, images in enumerate(generator):
     41         with torch.no_grad():
     42             start = i * config["batch_size_image_level"]

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

## === cell 13
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
    with torch.no_grad():
        features = features.to(device, non_blocking=True)
        preds = lv2_model(features)
        preds = torch.sigmoid(preds).squeeze(-1).detach().cpu().numpy()

    for j in range(len(list_uid)):
        submission_dict["row_id"].append(f"{list_uid[j]}_patient_overall")
        submission_dict["fractured"].append(float(np.clip(preds[j], 1e-6, 1 - 1e-6)))



## === cell 14
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
