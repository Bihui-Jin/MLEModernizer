# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.6210332381908321

# 6. Current score

1.66132

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.66132) has done: 'Main bottlenecks are (1) repeated DICOM decoding and (2) heavy Python/Numpy per-slice preprocessing (stack/resize/crop/normalize) executed thousands of times. I keep the exact same model inference logic and slice selection, but make decoding cheaper by caching + requesting fewer tags, disabling unnecessary pydicom validation, and parallelizing decode with a tuned threadpool/chunksize. I also eliminate avoidable allocations by preallocating HWC/CHW buffers, doing crop with slicing, and using a single pinned CPU tensor + non_blocking H2D copies (while keeping numeric semantics the same). Finally, I avoid creating `dicom_cache_u8` dicts repeatedly by fetching from an LRU cache and only materializing what’s needed for each study.'
- What this solution (achieved 1.66132) has done: 'Your pipeline already runs and writes a valid submission; the main reason the score is far from the target is that Stage‑2 (patient_overall) is effectively untrained (random init) when its checkpoint is missing, yet you still use it, which badly hurts the weighted log loss. I make the smallest semantics-preserving change: only use the GRU patient model when its weights are actually loaded; otherwise compute `patient_overall` from the per-level probabilities (a standard probabilistic OR), which is consistent with the task definition and typically much better calibrated than random logits. I also load Stage‑2 with `strict=False` to avoid silent key mismatches causing “loaded but wrong”, and I ensure patient_overall is always at least `max(C1..C7)` (monotonic consistency) to reduce logloss penalties. These changes keep the same models, features, and inference flow; they just prevent a clearly harmful fallback path and improve metric alignment.'

# 9. Code solution

## === cell 0
import os, sys, time, glob
import numpy as np
import pandas as pd



## === cell 1
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn

import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

from torch.utils.data import Dataset, DataLoader
from torchvision.models.convnext import convnext_base, ConvNeXt_Base_Weights

from albumentations import Compose, CenterCrop



## === cell 2
torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
np.random.seed(0)

try:
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    torch.set_num_interop_threads(max(1, min(4, os.cpu_count() or 1)))
except Exception:
    pass
cv2.setNumThreads(max(1, os.cpu_count() or 1))

try:
    pydicom.config.enforce_valid_values = False
except Exception:
    pass
try:
    pydicom.config.settings.reading_validation_mode = pydicom.config.IGNORE
except Exception:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = "../input/rsna-2022-cervical-spine-fracture-detection"
TEST_CSV = f"{DATA_ROOT}/test.csv"
SAMPLE_SUB = f"{DATA_ROOT}/sample_submission.csv"
TEST_PATH = f"{DATA_ROOT}/test_images"

assert os.path.exists(TEST_CSV), f"Missing {TEST_CSV}"
assert os.path.exists(TEST_PATH), f"Missing {TEST_PATH}"



## === cell 3
config = {
    "seq_len": 150,
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
print("Num studies:", len(study_id_list))
print("First study:", study_id_list[0] if study_id_list else None)



## === cell 6
selected_image_dict = {}
dicom_file_map = {}

for uid in study_id_list:
    dicom_files = sorted(glob.glob(os.path.join(TEST_PATH, uid, "*.dcm")))
    dicom_file_map[uid] = dicom_files

    n = len(dicom_files)
    if n == 0:
        selected_image_dict[uid] = []
        continue

    middle = n // 2
    k = max(1, int(0.15 * n))  # keep original intent, but ensure at least 1 slice

    left = list(range(max(0, middle - k), middle))
    right = list(range(middle + 1, min(n, middle + 1 + k)))
    selected = left + right
    if len(selected) == 0:
        selected = [middle]

    selected_image_dict[uid] = selected



## === cell 7
uid0 = study_id_list[0]
print("Example selected indices:", selected_image_dict[uid0][:10], "...")
print("Num dicoms:", len(dicom_file_map[uid0]))



## === cell 8
_REQUIRED_TAGS = [
    "RescaleSlope",
    "RescaleIntercept",
    "WindowCenter",
    "WindowWidth",
    "PhotometricInterpretation",
    "BitsStored",
    "PixelRepresentation",
    "SamplesPerPixel",
    "PlanarConfiguration",
    "PixelData",
]


def read_dicom_pixel_array(path: str):
    try:
        ds = pydicom.dcmread(
            path,
            force=True,
            defer_size="1 KB",
            specific_tags=_REQUIRED_TAGS,
            stop_before_pixels=False,
        )
    except Exception:
        return None, None

    try:
        arr = ds.pixel_array
    except Exception:
        return None, ds
    try:
        arr = apply_voi_lut(arr, ds)
    except Exception:
        pass
    return arr.astype(np.float32), ds




## === cell 9
def window_from_array(img: np.ndarray, ds, WL=400, WW=1800) -> np.ndarray:
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    img = img * slope + intercept

    upper = WL + WW // 2
    lower = WL - WW // 2
    X = np.clip(img, lower, upper, out=img)  # reuse buffer
    X -= X.min()
    denom = X.max()
    if denom > 0:
        X /= denom
    X *= 255.0
    return X.astype("uint8", copy=False)


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))




## === cell 10
mean = np.array([0.456, 0.456, 0.456], dtype=np.float32)
std = np.array([0.224, 0.224, 0.224], dtype=np.float32)




## === cell 11
class CSFImageDataset(Dataset):
    def __init__(self, uid, dicom_files, index_list, target_size, crop_size):
        self.uid = uid
        self.dicom_files = dicom_files
        self.index_list = index_list
        self.target_size = target_size
        self.crop_size = crop_size
        self.inference_transform = Compose([CenterCrop(self.crop_size, self.crop_size)])

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, i):
        idx = int(self.index_list[i])
        n = len(self.dicom_files)
        if n == 0:
            raise IndexError("No DICOM files found for uid")

        idx0 = max(0, min(n - 1, idx - 1))
        idx1 = max(0, min(n - 1, idx))
        idx2 = max(0, min(n - 1, idx + 1))

        paths = [self.dicom_files[idx0], self.dicom_files[idx1], self.dicom_files[idx2]]
        imgs = []
        for p in paths:
            arr, ds = read_dicom_pixel_array(p)
            if arr is None:
                raise RuntimeError(
                    f"Undecodable DICOM (likely JPEG-lossless without plugins): {p}"
                )
            imgs.append(window_from_array(arr, ds))

        stacked_img = np.stack(imgs, axis=-1)  # H,W,3
        stacked_img = cv2.resize(
            stacked_img,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_LINEAR,
        )
        stacked_img = self.inference_transform(image=stacked_img)["image"]

        X = img2tensor((stacked_img.astype(np.float32) / 255.0 - mean) / std)
        return X




## === cell 12
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




## === cell 13
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




## === cell 14
def safe_load_state_dict(model, path: str, strict: bool = True) -> bool:
    if path and os.path.exists(path):
        sd = torch.load(path, map_location="cpu")
        model.load_state_dict(sd, strict=strict)
        return True
    return False


lv1_model = ConvNextCNN_B_Feature()

lv1_loaded = safe_load_state_dict(
    lv1_model,
    "../input/cnn-lstm-oct-25-exp-9/run_0_reduced_data/run_0_reduced_data/model_0.pth",
    strict=True,
)
if not lv1_loaded:
    base = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
    lv1_model.features.load_state_dict(base.features.state_dict())
lv1_model = lv1_model.to(device).eval()

lv2_model = CSFNet(input_len=config["feature_size"], lstm_size=config["lstm_size"])
lv2_loaded = safe_load_state_dict(
    lv2_model,
    "../input/cnn-lstm-oct-25-exp-9/run_0_reduced_data/run_0_reduced_data/model_lstm_0.pth",
    strict=False,
)
lv2_model = lv2_model.to(device).eval()

print("Stage-1 weights loaded:", lv1_loaded)
print("Stage-2 weights loaded:", lv2_loaded)




## === cell 15
def _center_crop_np(img_hwc: np.ndarray, crop_size: int) -> np.ndarray:
    h, w = img_hwc.shape[:2]
    ch = min(crop_size, h)
    cw = min(crop_size, w)
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    return img_hwc[y0 : y0 + ch, x0 : x0 + cw]


uid_triplets = {}
for uid in study_id_list:
    dicom_files = dicom_file_map.get(uid, [])
    idxs = selected_image_dict.get(uid, [])
    n = len(dicom_files)
    triplets = []
    if n and idxs:
        for idx in idxs:
            idx = int(idx)
            idx0 = max(0, min(n - 1, idx - 1))
            idx1 = max(0, min(n - 1, idx))
            idx2 = max(0, min(n - 1, idx + 1))
            triplets.append((dicom_files[idx0], dicom_files[idx1], dicom_files[idx2]))
    uid_triplets[uid] = triplets



## === cell 16
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict

submission_dict = {"row_id": [], "fractured": []}
feature_array_dict = {}
per_uid_level_preds = {}  # uid -> np.array(7,)
per_uid_patient_pred = {}  # uid -> float

eps = 1e-4
batch_size_img = int(config["batch_size_image_level"])
use_amp = torch.cuda.is_available()

_target_size = int(config["target_size"])
_crop_size = int(config["crop_size"])
_seq_len = int(config["seq_len"])
_feature_size = int(config["feature_size"])
_mean = mean
_std = std
_resize = cv2.resize
_center_crop = _center_crop_np

pin_memory = bool(torch.cuda.is_available())

_CPU = os.cpu_count() or 1
DICOM_WORKERS = max(2, min(12, _CPU))

batch_buf = np.empty((batch_size_img, 3, _crop_size, _crop_size), dtype=np.float32)
host_batch = torch.empty(
    (batch_size_img, 3, _crop_size, _crop_size),
    dtype=torch.float32,
    pin_memory=pin_memory,
)
device_batch = torch.empty(
    (batch_size_img, 3, _crop_size, _crop_size), dtype=torch.float32, device=device
)

tmp_hwc_u8 = np.empty((_target_size, _target_size, 3), dtype=np.uint8)

executor = ThreadPoolExecutor(max_workers=DICOM_WORKERS)

_LRU_MAX = 8192
_dicom_u8_lru = OrderedDict()


def _lru_get(path: str):
    v = _dicom_u8_lru.get(path, None)
    if v is not None:
        _dicom_u8_lru.move_to_end(path)
    return v


def _lru_put(path: str, v):
    _dicom_u8_lru[path] = v
    _dicom_u8_lru.move_to_end(path)
    if len(_dicom_u8_lru) > _LRU_MAX:
        _dicom_u8_lru.popitem(last=False)


def _decode_window_u8(path: str):
    cached = _lru_get(path)
    if cached is not None:
        return path, cached
    arr, ds = read_dicom_pixel_array(path)
    if arr is None:
        return path, None
    u8 = window_from_array(arr, ds)
    _lru_put(path, u8)
    return path, u8


with tqdm(study_id_list, desc="Per-study inference") as pbar:
    for uid in pbar:
        dicom_files = dicom_file_map.get(uid, [])
        triplets = uid_triplets.get(uid, [])

        fallback_level = np.full(7, 0.01, dtype=np.float32)
        fallback_patient = float(np.max(fallback_level))

        if len(dicom_files) == 0 or len(triplets) == 0:
            per_uid_level_preds[uid] = fallback_level
            per_uid_patient_pred[uid] = fallback_patient
            feature_array_dict[uid] = np.zeros((1, _feature_size), dtype=np.float32)
            continue

        uniq_paths = sorted({p for tri in triplets for p in tri})
        dicom_cache_u8 = {}

        for p, u8 in executor.map(_decode_window_u8, uniq_paths, chunksize=64):
            dicom_cache_u8[p] = u8

        feats_chunks = []
        preds_chunks = []

        nb = 0
        valid_triplet_count = 0

        def _infer_batch(n_in_batch: int):
            if n_in_batch <= 0:
                return
            host_batch[:n_in_batch].copy_(
                torch.from_numpy(batch_buf[:n_in_batch]), non_blocking=False
            )
            device_batch[:n_in_batch].copy_(host_batch[:n_in_batch], non_blocking=True)
            xb = device_batch[:n_in_batch]
            with torch.inference_mode():
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        features, preds = lv1_model(xb)
                else:
                    features, preds = lv1_model(xb)
                feats_chunks.append(features.detach().cpu())
                preds_chunks.append(torch.sigmoid(preds.detach()).cpu())

        for p0, p1, p2 in triplets:
            i0 = dicom_cache_u8.get(p0, None)
            i1 = dicom_cache_u8.get(p1, None)
            i2 = dicom_cache_u8.get(p2, None)
            if i0 is None or i1 is None or i2 is None:
                continue

            tmp_hwc_u8[:, :, 0] = _resize(
                i0, (_target_size, _target_size), interpolation=cv2.INTER_LINEAR
            )
            tmp_hwc_u8[:, :, 1] = _resize(
                i1, (_target_size, _target_size), interpolation=cv2.INTER_LINEAR
            )
            tmp_hwc_u8[:, :, 2] = _resize(
                i2, (_target_size, _target_size), interpolation=cv2.INTER_LINEAR
            )

            cropped = _center_crop(tmp_hwc_u8, _crop_size)  # view/slice, no copy

            Xnp = cropped.astype(np.float32, copy=False)
            Xnp = (Xnp / 255.0 - _mean) / _std  # H,W,3 float32

            batch_buf[nb, 0, :, :] = Xnp[:, :, 0]
            batch_buf[nb, 1, :, :] = Xnp[:, :, 1]
            batch_buf[nb, 2, :, :] = Xnp[:, :, 2]
            nb += 1
            valid_triplet_count += 1

            if nb >= batch_size_img:
                _infer_batch(nb)
                nb = 0

        _infer_batch(nb)

        if valid_triplet_count == 0 or len(feats_chunks) == 0:
            per_uid_level_preds[uid] = fallback_level
            per_uid_patient_pred[uid] = fallback_patient
            feature_array_dict[uid] = np.zeros((1, _feature_size), dtype=np.float32)
            continue

        feats = (
            torch.cat(feats_chunks, dim=0).numpy().astype(np.float32, copy=False)
        )  # (N,1024)
        preds = (
            torch.cat(preds_chunks, dim=0).numpy().astype(np.float32, copy=False)
        )  # (N,7)

        feature_array_dict[uid] = feats
        mean_preds = np.mean(preds, axis=0).astype(np.float32, copy=False)
        mean_preds = np.clip(mean_preds, eps, 1.0 - eps)
        per_uid_level_preds[uid] = mean_preds

        if lv2_loaded:
            fa = feats
            if fa.shape[0] > _seq_len:
                x = _resize(fa, (fa.shape[1], _seq_len), interpolation=cv2.INTER_LINEAR)
            else:
                x = np.pad(
                    fa,
                    pad_width=[(0, _seq_len - fa.shape[0]), (0, 0)],
                    constant_values=0,
                )
            Xp = torch.from_numpy(x.astype(np.float32, copy=False)).unsqueeze(0)
            Xp = Xp.to(device, non_blocking=True)
            with torch.inference_mode():
                p = torch.sigmoid(lv2_model(Xp)).item()
            p = float(np.clip(p, eps, 1.0 - eps))
        else:
            p = 1.0 - float(np.prod(1.0 - mean_preds))
            p = float(np.clip(p, eps, 1.0 - eps))

        p = max(p, float(np.max(mean_preds)))  # monotonic consistency
        per_uid_patient_pred[uid] = float(np.clip(p, eps, 1.0 - eps))

executor.shutdown(wait=True)



## === cell 17
pred_map = {}
for uid in study_id_list:
    lev = per_uid_level_preds.get(uid, np.full(7, 0.01, dtype=np.float32))
    pred_map[f"{uid}_C1"] = float(lev[0])
    pred_map[f"{uid}_C2"] = float(lev[1])
    pred_map[f"{uid}_C3"] = float(lev[2])
    pred_map[f"{uid}_C4"] = float(lev[3])
    pred_map[f"{uid}_C5"] = float(lev[4])
    pred_map[f"{uid}_C6"] = float(lev[5])
    pred_map[f"{uid}_C7"] = float(lev[6])
    pred_map[f"{uid}_patient_overall"] = float(
        per_uid_patient_pred.get(uid, float(np.max(lev)))
    )

out = test_df[["row_id"]].copy()
out["fractured"] = out["row_id"].map(pred_map)

out["fractured"] = out["fractured"].fillna(0.01).astype(np.float32)
out["fractured"] = np.clip(out["fractured"].values, eps, 1.0 - eps)

print(out.head())
print("Rows:", len(out), "Missing:", int(out["fractured"].isna().sum()))



## === cell 18
out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", out.shape)
print("Columns:", list(out.columns))
print("submission.csv exists:", os.path.exists("submission.csv"))
print("submission.csv preview:\n", pd.read_csv("submission.csv").head())
