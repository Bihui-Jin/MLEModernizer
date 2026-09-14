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

0.60216494905809

# 6. Current score

0.81165

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.81165) has done: 'I fix the missing checkpoint path by making the script robust to the absence of external pretrained weights: if the expected `.pth` files aren’t found, it fall back to a deterministic constant-probability submission (still valid format) so the notebook always finishes and writes `submission.csv`. I also fix DICOM JPEG decompression errors by using the modern `pydicom.pixels` backend (and safely skipping image inference when decompression isn’t available), instead of relying on unavailable plugins. Finally, I ensure every `StudyInstanceUID` in `test.csv` gets exactly 8 predictions aligned to `row_id` by merging onto `test.csv` rather than manually appending rows, which eliminates the later `KeyError` and guarantees a valid submission.'
- What this solution (achieved 0.81165) has done: 'Your current score (0.81165, lower-is-better) is worse than the target (0.60216), and the biggest driver is that you’re not actually using the intended pretrained checkpoints, so the ConvNeXt runs with random weights and the GRU also has random weights, producing weak probabilities. I keep the same model code and inference flow, but make the checkpoint discovery robust by searching common `/kaggle/input/**.pth` locations (including the provided `cnn-lstm-oct-21` dataset path variants) and loading the first matching files. I also make DICOM decoding more robust without changing your feature extraction logic by using `stop_before_pixels=True` in `dcmread` and trying `pydicom.pixels.pixel_array()` when available, then falling back to `data.pixel_array`. These minimal changes should move the score down (better) toward the target while preserving your architecture and submission semantics.'
- What this solution (achieved 0.81165) has done: 'Your current score (0.81165, lower-is-better) is still far from the target (0.60216), so we should improve real inference rather than fall back to constants. I keep your exact model architecture and inference flow, but make checkpoint loading robust to common training-time key prefixes (e.g., `module.`) so the pretrained weights actually load instead of silently producing weak/random predictions. I also make DICOM pixel decoding more robust (without changing preprocessing math) by retrying `stop_before_pixels=True` on failure and ensuring grayscale images are expanded to 3 channels consistently. Finally, I force deterministic settings to stabilize outputs and reduce score variance run-to-run.'
- What this solution (achieved 0.81165) has done: 'Your current score (0.81165, lower-is-better) is still meaningfully worse than the target (0.60216), so the smallest likely win is to make sure the pretrained checkpoints (when present) *actually* load into the correct submodules instead of being ignored due to key mismatches. I keep your architecture and inference loop identical, but improve `_load_state_dict_flexible` to (1) extract the right nested keys, (2) optionally filter by submodule prefix (so an entire checkpoint containing `features.*`/`fc.*` loads cleanly), and (3) refuse to proceed with “full inference” if the load is clearly wrong (large missing key count), falling back to constants instead of random-weight predictions. This should reduce loss (improve score) when checkpoints are available and prevent accidental bad submissions when they aren’t. I also add a small, metric-safe probability clipping immediately after sigmoid (still the same semantics) to avoid extreme logits producing numerical issues.'

# 9. Code solution

## === cell 0
import os
import sys
import glob
import time
import warnings

try:
    import pylibjpeg  # noqa: F401
except Exception:
    pass



## === cell 1
import numpy as np
import pandas as pd

import cv2
from tqdm import tqdm

import pydicom
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision.models.convnext import convnext_base

warnings.filterwarnings("ignore")



## === cell 2
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 3
BASE_INPUT = "../input/rsna-2022-cervical-spine-fracture-detection"
TEST_CSV = f"{BASE_INPUT}/test.csv"
TEST_PATH = f"{BASE_INPUT}/test_images"

SAMPLE_SUB = f"{BASE_INPUT}/sample_submission.csv"



## === cell 4
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




## === cell 5
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




## === cell 6
test_df = load_df_test()
study_id_list = list(test_df.StudyInstanceUID.unique())
print("Num studies:", len(study_id_list))
print("First UID:", study_id_list[0] if len(study_id_list) else None)



## === cell 7
selected_image_dict = {}
for uid in study_id_list:
    dicom_files = glob.glob(os.path.join(f"{TEST_PATH}/{uid}", "*.dcm"))
    dicom_files = sorted(
        dicom_files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )
    if len(dicom_files) == 0:
        selected_image_dict[uid] = []
        continue
    middle_slice = int(len(dicom_files) / 2)
    num_left_images = num_right_images = int(0.15 * len(dicom_files))
    left = list(np.arange(max(1, middle_slice - num_left_images), middle_slice, 1))
    right = list(
        np.arange(
            middle_slice + 1,
            min(len(dicom_files), middle_slice + num_right_images) + 1,
            1,
        )
    )
    selected_image_dict[uid] = left + right



## === cell 8
if len(study_id_list):
    print("Example selected slices:", selected_image_dict[study_id_list[0]][:10])



## === cell 9
try:
    from pydicom.pixels import pixel_array as pyd_pixel_array  # noqa: F401

    HAVE_PYDICOM_PIXELS = True
except Exception:
    HAVE_PYDICOM_PIXELS = False

print("HAVE_PYDICOM_PIXELS:", HAVE_PYDICOM_PIXELS)




## === cell 10
def window(data, WL=400, WW=1800):
    slope = float(getattr(data, "RescaleSlope", 1.0))
    intercept = float(getattr(data, "RescaleIntercept", 0.0))

    if HAVE_PYDICOM_PIXELS:
        img = pydicom.pixels.pixel_array(data)
    else:
        img = data.pixel_array  # may raise if decompression unavailable

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




## === cell 11
def center_crop(image, crop_h, crop_w):
    h, w = image.shape[:2]
    ch, cw = min(crop_h, h), min(crop_w, w)
    y1 = max(0, (h - ch) // 2)
    x1 = max(0, (w - cw) // 2)
    return image[y1 : y1 + ch, x1 : x1 + cw]




## === cell 12
mean = np.array([0.456, 0.456, 0.456], dtype=np.float32)
std = np.array([0.224, 0.224, 0.224], dtype=np.float32)


class CSFImageDataset(Dataset):
    def __init__(self, uid, image_list, target_size, crop_size):
        self.uid = uid
        self.image_list = image_list
        self.target_size = target_size
        self.crop_size = crop_size

    def __len__(self):
        return len(self.image_list)

    def _safe_dcmread(self, path):
        try:
            return pydicom.dcmread(path, force=True, stop_before_pixels=False)
        except Exception:
            return pydicom.dcmread(path, force=True, stop_before_pixels=True)

    def __getitem__(self, index):
        slice_num = int(self.image_list[index])

        def clamp(n):
            return max(1, n)

        PATH = os.path.join(TEST_PATH, self.uid)
        paths = [
            os.path.join(PATH, f"{clamp(slice_num-1)}.dcm"),
            os.path.join(PATH, f"{clamp(slice_num)}.dcm"),
            os.path.join(PATH, f"{clamp(slice_num+1)}.dcm"),
        ]

        data_list = [self._safe_dcmread(p) for p in paths]
        imgs = [window(d) for d in data_list]

        stacked_img = np.stack(imgs, axis=-1)

        if stacked_img.ndim == 2:
            stacked_img = np.repeat(stacked_img[..., None], 3, axis=-1)
        elif stacked_img.shape[-1] != 3:
            if stacked_img.shape[-1] == 1:
                stacked_img = np.repeat(stacked_img, 3, axis=-1)
            else:
                stacked_img = stacked_img[..., :3]

        stacked_img = cv2.resize(
            stacked_img,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_LINEAR,
        )
        stacked_img = center_crop(stacked_img, self.crop_size, self.crop_size)

        X = img2tensor((stacked_img.astype(np.float32) / 255.0 - mean) / std)
        return X




## === cell 13
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




## === cell 14
class ConvNextCNN_B_Feature(nn.Module):
    def __init__(self):
        super(ConvNextCNN_B_Feature, self).__init__()
        m = convnext_base(weights=None)  # avoid downloading weights
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




## === cell 15
CKPT_LV1 = "../input/cnn-lstm-oct-21/run_2/run_2/model_2.pth"
CKPT_LV2 = "../input/cnn-lstm-oct-21/run_2/run_2/model_lstm_2.pth"


def _find_ckpt(preferred_path: str, filename: str):
    if os.path.exists(preferred_path):
        return preferred_path
    candidates = glob.glob(os.path.join("../input", "**", filename), recursive=True)
    candidates = sorted(candidates, key=lambda p: (len(p), p))
    return candidates[0] if candidates else None


def _as_state_dict(obj):
    if isinstance(obj, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj


def _strip_known_prefixes(sd):
    if not isinstance(sd, dict) or not sd:
        return sd

    def strip_prefix(sd0, prefix):
        if all(k.startswith(prefix) for k in sd0.keys()):
            return {k[len(prefix) :]: v for k, v in sd0.items()}
        return sd0

    out = sd
    for pref in ["module.", "model.", "net."]:
        out = strip_prefix(out, pref)
    return out


def _filter_by_any_submodule_prefix(sd, model_keys):
    if not isinstance(sd, dict) or not sd:
        return sd
    model_key_set = set(model_keys)
    filtered = {k: v for k, v in sd.items() if k in model_key_set}
    return filtered if len(filtered) > 0 else sd


def _load_state_dict_flexible(model: nn.Module, ckpt_path: str, device):
    state = torch.load(ckpt_path, map_location=device)
    state = _as_state_dict(state)
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint at {ckpt_path} is not a state_dict-like object")

    state = _strip_known_prefixes(state)
    state = _filter_by_any_submodule_prefix(state, model.state_dict().keys())

    missing, unexpected = model.load_state_dict(state, strict=False)
    return missing, unexpected


ckpt1 = _find_ckpt(CKPT_LV1, "model_2.pth")
ckpt2 = _find_ckpt(CKPT_LV2, "model_lstm_2.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print("Resolved ckpt1:", ckpt1)
print("Resolved ckpt2:", ckpt2)

have_ckpts = (ckpt1 is not None) and (ckpt2 is not None)
print("Have checkpoints:", have_ckpts)

lv1_model = ConvNextCNN_B_Feature().to(device)
lv2_model = CSFNet(input_len=config["feature_size"], lstm_size=config["lstm_size"]).to(
    device
)

ckpt_load_ok = False
if have_ckpts:
    try:
        miss1, unexp1 = _load_state_dict_flexible(lv1_model, ckpt1, device)
        miss2, unexp2 = _load_state_dict_flexible(lv2_model, ckpt2, device)
        print("lv1 missing keys:", len(miss1), "unexpected keys:", len(unexp1))
        print("lv2 missing keys:", len(miss2), "unexpected keys:", len(unexp2))

        ckpt_load_ok = (len(miss1) < 50) and (len(miss2) < 10)
    except Exception as e:
        print("Checkpoint load failed:", repr(e))
        ckpt_load_ok = False

lv1_model.eval()
lv2_model.eval()



## === cell 16
DEFAULT_LEVEL_P = 0.05
DEFAULT_OVERALL_P = 0.15

prediction_types = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]


def make_constant_submission(
    df_test, p_level=DEFAULT_LEVEL_P, p_overall=DEFAULT_OVERALL_P
):
    sub = df_test[["row_id", "prediction_type"]].copy()
    sub["fractured"] = np.where(
        sub["prediction_type"].eq("patient_overall"), p_overall, p_level
    ).astype(np.float32)
    return sub[["row_id", "fractured"]]




## === cell 17
run_full_inference = bool(have_ckpts and ckpt_load_ok)
print("Will run full inference:", run_full_inference)




## === cell 18
def run_inference_and_build_submission():
    submission_rows = []
    feature_array_dict = {}

    for uid in tqdm(study_id_list, desc="Studies"):
        image_list = selected_image_dict.get(uid, [])
        if len(image_list) == 0:
            feature_array_dict[uid] = np.zeros(
                (1, config["feature_size"]), dtype=np.float32
            )
            for pt in prediction_types:
                rid = f"{uid}_{pt}"
                submission_rows.append(
                    (
                        rid,
                        (
                            DEFAULT_OVERALL_P
                            if pt == "patient_overall"
                            else DEFAULT_LEVEL_P
                        ),
                    )
                )
            continue

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
            num_workers=0,
        )

        preds_uid = []
        feature_array = np.zeros(
            (len(dataset), config["feature_size"]), dtype=np.float32
        )

        for i, images in enumerate(generator):
            with torch.no_grad():
                start = i * config["batch_size_image_level"]
                end = min(start + images.shape[0], len(generator.dataset))
                images = images.to(device, non_blocking=True)

                features, preds = lv1_model(images)
                feature_array[start:end] = features.detach().cpu().numpy()

                preds = preds.sigmoid().clamp(1e-6, 1 - 1e-6).detach().cpu()
                preds_uid.append(preds)

        feature_array_dict[uid] = feature_array

        mean_preds = (
            torch.mean(torch.cat(preds_uid, dim=0), dim=0).numpy().astype(np.float32)
        )
        for k, pt in enumerate(["C1", "C2", "C3", "C4", "C5", "C6", "C7"]):
            submission_rows.append((f"{uid}_{pt}", float(mean_preds[k])))

    instance_ds = CSFInstanceDataset(
        feature_array_dict=feature_array_dict,
        study_id_list=study_id_list,
        seq_len=config["seq_len"],
    )
    instance_loader = DataLoader(
        dataset=instance_ds,
        batch_size=config["batch_size_patient_level"],
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
    )

    for features, list_uid in tqdm(
        instance_loader, total=len(instance_loader), desc="Patient overall"
    ):
        with torch.no_grad():
            features = features.to(device, non_blocking=True)
            preds = (
                lv2_model(features)
                .sigmoid()
                .clamp(1e-6, 1 - 1e-6)
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )

        for j in range(len(list_uid)):
            submission_rows.append((f"{list_uid[j]}_patient_overall", float(preds[j])))

    sub_df = pd.DataFrame(submission_rows, columns=["row_id", "fractured"])
    return sub_df




## === cell 19
try:
    if run_full_inference:
        raw_pred_df = run_inference_and_build_submission()
        sub_df = test_df[["row_id"]].merge(raw_pred_df, on="row_id", how="left")
        missing = sub_df["fractured"].isna()
        if missing.any():
            tmp = test_df.loc[missing, ["prediction_type"]]
            sub_df.loc[missing, "fractured"] = np.where(
                tmp["prediction_type"].eq("patient_overall"),
                DEFAULT_OVERALL_P,
                DEFAULT_LEVEL_P,
            ).astype(np.float32)
        sub_df = sub_df[["row_id", "fractured"]]
    else:
        sub_df = make_constant_submission(test_df)
except Exception as e:
    print("Inference failed; falling back to constant submission. Error:", repr(e))
    sub_df = make_constant_submission(test_df)



## === cell 20
assert list(sub_df.columns) == ["row_id", "fractured"]
assert len(sub_df) == len(test_df), (len(sub_df), len(test_df))
assert sub_df["fractured"].notna().all()
sub_df["fractured"] = sub_df["fractured"].clip(1e-6, 1 - 1e-6)

sub_df.head()



## === cell 21
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.tail())
