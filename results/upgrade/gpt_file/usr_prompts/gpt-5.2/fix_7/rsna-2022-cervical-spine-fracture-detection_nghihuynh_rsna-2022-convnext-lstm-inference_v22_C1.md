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

0.6438881310488689

# 6. Current score

1.15429

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.15429) has done: 'I remove the broken dependency-install cell (it targets cp37 wheels and fails on Python 3.11) and instead rely on the Kaggle environment’s existing `pydicom` pixel decoding (which already supports these JPEG DICOMs in this runtime). I also fix two data-loading bugs that currently prevent any valid submission: (1) DICOM slice filenames are not contiguous numbers, so the dataset now index by sorted filepaths and safely pick prev/next slices; (2) the stage-1 feature extraction now always populate `feature_array_dict` for every UID so stage-2 can’t KeyError. Finally, I make weight loading robust by falling back to a valid baseline submission (sample_submission with safe probabilities) if the external model checkpoint dataset path isn’t available, ensuring a `submission.csv` is always produced.'
- What this solution (achieved 1.15429) has done: 'Your current score (1.15429, lower-is-better) is substantially worse than the target (0.6439), so we should improve the validity of the predicted probabilities with minimal semantic change. The biggest low-risk issue is that `patient_overall` is predicted independently, but in this competition it should be at least as large as the maximum of C1–C7 for each study; enforcing this consistency typically reduces weighted logloss without changing the model. I add a tiny post-processing step that sets `patient_overall = max(patient_overall, max(C1..C7))` per StudyInstanceUID, then rebuild the submission strictly aligned to `sample_submission.csv`. Everything else (models, features, loops, transforms) stays the same.'
- What this solution (achieved 1.15429) has done: 'Your current score (1.15429, lower-is-better) is far worse than the target (0.6439), so we should improve probability calibration and reduce extreme/confident errors without changing the model. The smallest high-impact fix is to apply label-wise prevalence calibration from `train.csv` (a standard logloss stabilizer): shrink each predicted probability toward the empirical base rate for that label using a single mixing parameter. This preserves the exact modeling pipeline (same models/features/loops) and only adjusts post-processing, which is directly relevant to weighted logloss. We keep your existing logical constraint `patient_overall >= max(C1..C7)` and ensure the submission stays perfectly aligned to `sample_submission.csv`.'
- What this solution (achieved 1.15429) has done: 'Your current score (1.15429, lower-is-better) is far above the target (0.6439), so we should improve logloss by reducing overconfident wrong predictions without changing the model or training. The smallest high-impact adjustment is to make the prevalence-mixing calibration strength (`alpha`) adaptive per label: use stronger shrinkage for rarer vertebra labels and lighter shrinkage for `patient_overall`, which is more prevalent and more heavily weighted. We keep your existing logical consistency constraint `patient_overall >= max(C1..C7)` and keep submission alignment strictly via `sample_submission.csv`. Everything else (models, features, loops, transforms, loss) remains identical.'
- What this solution (achieved 1.15429) has done: 'Your current logloss (1.15429, lower-is-better) is still far above the target (0.6439), so we should reduce extreme/overconfident probabilities with minimal semantic change. I keep your two-stage inference exactly the same, but make the post-processing calibration a bit stronger and more stable by (1) explicitly clipping raw model probabilities before any transforms, (2) applying a light per-label logit-space shrink toward the train base-rate (more effective for logloss than linear mixing), and (3) re-enforcing `patient_overall >= max(C1..C7)` after calibration. This is purely post-processing aligned to the weighted logloss metric and should move the score down (better) without changing the model, features, or training. Submission alignment still be driven strictly by `sample_submission.csv`.'
- What this solution (achieved 1.15429) has done: 'We keep your two-stage inference exactly the same and only adjust post-processing to better match weighted logloss (lower is better) since your current score (1.15429) is far above the target (0.6439). The main minimal fix is to apply a slightly stronger, label-aware logit shrink toward train base rates (especially for the rare C1–C7 labels) while keeping `patient_overall >= max(C1..C7)` enforced after calibration. We also make the final fill for any missing predictions use each label’s base rate (instead of a hard 0.01), which is directly logloss-improving and low risk. No model architecture, feature extraction, or inference loops are changed; only probability post-processing and missing-value handling are adjusted.'

# 9. Code solution

## === cell 0
import os
import sys
import glob
import time
import math
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import cv2
import pydicom
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from albumentations import Compose, CenterCrop

from torchvision.models.convnext import convnext_base

warnings.filterwarnings("ignore")




## === cell 1
def seed_everything(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_everything(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE




## === cell 2
config = {
    "seq_len": 150,
    "feature_size": 1024,  # expected by stage-2 model; will be validated when loading weights
    "lstm_size": 128,
    "target_size": 512,
    "crop_size": 368,
    "num_classes": 7,
    "batch_size_image_level": 32,
    "batch_size_patient_level": 8,
}




## === cell 3
BASE_INPUT = "../input/rsna-2022-cervical-spine-fracture-detection"
if not os.path.exists(BASE_INPUT):
    BASE_INPUT = "/kaggle/data/rsna-2022-cervical-spine-fracture-detection"
if not os.path.exists(BASE_INPUT):
    BASE_INPUT = "/kaggle/input/rsna-2022-cervical-spine-fracture-detection"

TEST_CSV_PATH = os.path.join(BASE_INPUT, "test.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_INPUT, "sample_submission.csv")
TEST_PATH = os.path.join(BASE_INPUT, "test_images")
TRAIN_CSV_PATH = os.path.join(BASE_INPUT, "train.csv")

TEST_CSV_PATH, TEST_PATH, os.path.exists(TEST_CSV_PATH), os.path.exists(TEST_PATH)




## === cell 4
def load_df_test() -> pd.DataFrame:
    df_test = pd.read_csv(TEST_CSV_PATH)

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
len(test_df), len(study_id_list), test_df.head()




## === cell 5
dicom_filepaths_dict: Dict[str, List[str]] = {}
selected_index_dict: Dict[str, List[int]] = {}

for uid in tqdm(study_id_list, desc="Indexing DICOM files"):
    dicom_files = sorted(glob.glob(os.path.join(TEST_PATH, uid, "*.dcm")))
    dicom_filepaths_dict[uid] = dicom_files

    n = len(dicom_files)
    if n == 0:
        selected_index_dict[uid] = []
        continue

    middle = n // 2
    num_left = num_right = int(0.15 * n)
    left_start = max(0, middle - num_left)
    left = list(range(left_start, middle))
    right_end = min(n, middle + num_right + 1)
    right = list(range(middle + 1, right_end))

    sel = left + right
    if len(sel) == 0:
        sel = [middle]  # ensure non-empty
    selected_index_dict[uid] = sel

uids_with_no_dicoms = [u for u in study_id_list if len(dicom_filepaths_dict[u]) == 0]
len(uids_with_no_dicoms), (uids_with_no_dicoms[:3] if uids_with_no_dicoms else None)




## === cell 6
def window(
    dcm: pydicom.dataset.FileDataset, WL: int = 400, WW: int = 1800
) -> np.ndarray:
    slope = float(getattr(dcm, "RescaleSlope", 1.0))
    intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
    img = dcm.pixel_array.astype(np.float32)
    img = img * slope + intercept

    upper, lower = WL + WW / 2.0, WL - WW / 2.0
    X = np.clip(img, lower, upper)
    X = X - X.min()
    mx = X.max()
    if mx > 0:
        X = X / mx
    X = (X * 255.0).astype(np.uint8)
    return X


def img2tensor(img: np.ndarray, dtype: np.dtype = np.float32) -> torch.Tensor:
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    return torch.from_numpy(img.astype(dtype, copy=False))


mean = np.array([0.456, 0.456, 0.456], dtype=np.float32)
std = np.array([0.224, 0.224, 0.224], dtype=np.float32)




## === cell 7
class CSFImageDataset(Dataset):
    def __init__(
        self,
        uid: str,
        dicom_files: List[str],
        selected_indices: List[int],
        target_size: int,
        crop_size: int,
    ):
        self.uid = uid
        self.dicom_files = dicom_files
        self.selected_indices = selected_indices
        self.target_size = target_size
        self.crop_size = crop_size
        self.inference_transform = Compose([CenterCrop(self.crop_size, self.crop_size)])

    def __len__(self):
        return len(self.selected_indices)

    def __getitem__(self, index: int):
        idx = self.selected_indices[index]
        n = len(self.dicom_files)
        if n == 0:
            X = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
            X = img2tensor((X / 255.0 - mean) / std)
            return X

        i0 = max(0, idx - 1)
        i1 = idx
        i2 = min(n - 1, idx + 1)

        fp0, fp1, fp2 = self.dicom_files[i0], self.dicom_files[i1], self.dicom_files[i2]
        d0 = pydicom.dcmread(fp0, force=True)
        d1 = pydicom.dcmread(fp1, force=True)
        d2 = pydicom.dcmread(fp2, force=True)

        imgs = [window(d) for d in (d0, d1, d2)]
        stacked_img = np.stack(imgs, axis=-1)  # H,W,3

        stacked_img = cv2.resize(
            stacked_img,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_LINEAR,
        )
        stacked_img = self.inference_transform(image=stacked_img)["image"]
        X = img2tensor((stacked_img.astype(np.float32) / 255.0 - mean) / std)
        return X




## === cell 8
class CSFInstanceDataset(Dataset):
    def __init__(
        self,
        feature_array_dict: Dict[str, np.ndarray],
        study_id_list: List[str],
        seq_len: int,
    ):
        self.feature_array_dict = feature_array_dict
        self.study_id_list = study_id_list
        self.seq_len = seq_len

    def __len__(self):
        return len(self.study_id_list)

    def __getitem__(self, index: int):
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




## === cell 9
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
    def __init__(self, input_len: int, lstm_size: int):
        super().__init__()
        self.lstm1 = nn.GRU(input_len, lstm_size, bidirectional=True, batch_first=True)
        self.last_linear = nn.Linear(lstm_size * 2, 1)

    def forward(self, x):
        h_lstm1, _ = self.lstm1(x)
        max_pool, _ = torch.max(h_lstm1, 1)
        logits = self.last_linear(max_pool)
        return logits




## === cell 10
WEIGHTS_BASE = "../input/cnn-lstm-oct-25/run_0"
if not os.path.exists(WEIGHTS_BASE):
    WEIGHTS_BASE = "/kaggle/input/cnn-lstm-oct-25/run_0"

LV1_WEIGHTS = os.path.join(WEIGHTS_BASE, "model_0.pth")
LV2_WEIGHTS = os.path.join(WEIGHTS_BASE, "model_lstm_0.pth")

have_weights = os.path.exists(LV1_WEIGHTS) and os.path.exists(LV2_WEIGHTS)
LV1_WEIGHTS, LV2_WEIGHTS, have_weights




## === cell 11
if have_weights:
    lv1_model = ConvNextCNN_B_Feature()
    lv1_model.load_state_dict(torch.load(LV1_WEIGHTS, map_location="cpu"))
    lv1_model = lv1_model.to(DEVICE)
    lv1_model.eval()

    with torch.no_grad():
        dummy = torch.zeros(
            (1, 3, config["crop_size"], config["crop_size"]), device=DEVICE
        )
        feat, _ = lv1_model(dummy)
        inferred_feat_size = int(feat.shape[1])
    config["feature_size"] = (
        inferred_feat_size  # keep consistent with actual model output
    )

    lv2_model = CSFNet(input_len=config["feature_size"], lstm_size=config["lstm_size"])
    lv2_model.load_state_dict(torch.load(LV2_WEIGHTS, map_location="cpu"))
    lv2_model = lv2_model.to(DEVICE)
    lv2_model.eval()

    inferred_feat_size
else:
    lv1_model = None
    lv2_model = None




## === cell 12
if not have_weights:
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
    default_c = 0.02
    default_any = 0.05
    preds = []
    for rid in sample_sub["row_id"].values:
        if rid.endswith("_patient_overall"):
            preds.append(default_any)
        else:
            preds.append(default_c)
    sample_sub["fractured"] = np.clip(preds, 1e-5, 1 - 1e-5)
    sample_sub.to_csv("submission.csv", index=False)
    print("Weights not found; wrote baseline submission.csv:", sample_sub.shape)




## === cell 13
if have_weights:
    submission_dict = {"row_id": [], "fractured": []}
    feature_array_dict: Dict[str, np.ndarray] = {}

    for uid in tqdm(study_id_list, desc="Stage-1 per-study"):
        dicom_files = dicom_filepaths_dict[uid]
        sel_idx = selected_index_dict[uid]

        dataset = CSFImageDataset(
            uid=uid,
            dicom_files=dicom_files,
            selected_indices=sel_idx,
            target_size=config["target_size"],
            crop_size=config["crop_size"],
        )
        generator = DataLoader(
            dataset,
            batch_size=config["batch_size_image_level"],
            shuffle=False,
            pin_memory=(DEVICE.type == "cuda"),
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
                end = min(start + images.shape[0], len(dataset))

                images = images.to(DEVICE, non_blocking=True)

                features, preds = lv1_model(images)
                feature_array[start:end] = features.detach().cpu().numpy()
                preds_uid.append(preds.sigmoid().detach().cpu())

        feature_array_dict[uid] = feature_array

        mean_preds = torch.mean(torch.cat(preds_uid, dim=0), dim=0).numpy()

        for k, lvl in enumerate(["C1", "C2", "C3", "C4", "C5", "C6", "C7"]):
            submission_dict["row_id"].append(f"{uid}_{lvl}")
            submission_dict["fractured"].append(float(mean_preds[k]))




## === cell 14
if have_weights:
    dataset2 = CSFInstanceDataset(
        feature_array_dict=feature_array_dict,
        study_id_list=study_id_list,
        seq_len=config["seq_len"],
    )
    generator2 = DataLoader(
        dataset=dataset2,
        batch_size=config["batch_size_patient_level"],
        shuffle=False,
        pin_memory=(DEVICE.type == "cuda"),
        num_workers=0,
        drop_last=False,
    )

    for features, list_uid in tqdm(
        generator2, total=len(generator2), desc="Stage-2 patient_overall"
    ):
        with torch.no_grad():
            features = features.to(DEVICE, non_blocking=True)
            preds = lv2_model(features).sigmoid().detach().cpu().numpy().reshape(-1)

        for j in range(len(list_uid)):
            submission_dict["row_id"].append(f"{list_uid[j]}_patient_overall")
            submission_dict["fractured"].append(float(preds[j]))




## === cell 15
if have_weights:
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-5, 1 - 1e-5)
        return np.log(p / (1.0 - p))

    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    sub_raw = pd.DataFrame.from_dict(submission_dict)
    sub_raw["StudyInstanceUID"] = sub_raw["row_id"].str.rsplit("_", n=1).str[0]
    sub_raw["prediction_type"] = sub_raw["row_id"].str.rsplit("_", n=1).str[1]

    pivot = sub_raw.pivot(
        index="StudyInstanceUID", columns="prediction_type", values="fractured"
    )
    for col in ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]:
        if col not in pivot.columns:
            pivot[col] = np.nan

    for col in ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]:
        pivot[col] = np.clip(pivot[col].astype(np.float32), 1e-5, 1 - 1e-5)

    max_c = pivot[["C1", "C2", "C3", "C4", "C5", "C6", "C7"]].max(axis=1, skipna=True)
    pivot["patient_overall"] = np.maximum(
        pivot["patient_overall"].astype(np.float32), max_c.astype(np.float32)
    )

    label_cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]
    base_rates = {c: 0.05 for c in label_cols}
    if os.path.exists(TRAIN_CSV_PATH):
        train_df = pd.read_csv(TRAIN_CSV_PATH)
        base_rates = train_df[label_cols].mean().astype(np.float32).to_dict()

    def alpha_for_base_rate(br: float, label: str) -> float:
        br = float(np.clip(br, 1e-5, 1 - 1e-5))
        if label == "patient_overall":
            lo, hi = 0.15, 0.35
        else:
            lo, hi = 0.35, 0.70
        a = lo + (hi - lo) * (1.0 - br)
        return float(np.clip(a, lo, hi))

    for col in label_cols:
        br = float(np.clip(base_rates.get(col, 0.05), 1e-5, 1 - 1e-5))
        alpha = alpha_for_base_rate(br, col)

        p = pivot[col].astype(np.float32).values
        logit_p = _logit(p)
        logit_br = float(_logit(np.array(br, dtype=np.float32)))

        logit_new = (1.0 - alpha) * logit_p + alpha * logit_br
        pivot[col] = _sigmoid(logit_new).astype(np.float32)

    max_c2 = pivot[["C1", "C2", "C3", "C4", "C5", "C6", "C7"]].max(axis=1, skipna=True)
    pivot["patient_overall"] = np.maximum(
        pivot["patient_overall"].astype(np.float32), max_c2.astype(np.float32)
    )

    sub_fixed = pivot.reset_index().melt(
        id_vars="StudyInstanceUID",
        value_vars=["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"],
        var_name="prediction_type",
        value_name="fractured",
    )
    sub_fixed["row_id"] = (
        sub_fixed["StudyInstanceUID"] + "_" + sub_fixed["prediction_type"]
    )
    sub_fixed = sub_fixed[["row_id", "fractured"]]

    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
    sub_df = sample_sub[["row_id"]].merge(sub_fixed, on="row_id", how="left")

    sub_df["prediction_type"] = sub_df["row_id"].str.rsplit("_", n=1).str[1]
    fill_map = {k: float(np.clip(v, 1e-5, 1 - 1e-5)) for k, v in base_rates.items()}
    sub_df["fractured"] = sub_df["fractured"].astype(np.float32)
    sub_df["fractured"] = sub_df["fractured"].fillna(
        sub_df["prediction_type"].map(fill_map).astype(np.float32)
    )
    sub_df["fractured"] = sub_df["fractured"].fillna(np.float32(0.05))
    sub_df["fractured"] = np.clip(sub_df["fractured"].values, 1e-5, 1 - 1e-5)

    sub_df = sub_df[["row_id", "fractured"]]
    sub_df.to_csv("submission.csv", index=False)
    print("Wrote submission.csv:", sub_df.shape)
    sub_df.head()




## === cell 16
assert os.path.exists("submission.csv"), "submission.csv was not created"
final_sub = pd.read_csv("submission.csv")
print(final_sub.shape, final_sub.columns.tolist())
final_sub.head()
