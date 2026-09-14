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

0.6163706658787828

# 6. Current score

0.92763

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.91732) has done: 'I (1) remove the notebook `!pip` installs and instead rely on the Kaggle image’s existing packages (Python 3.10), fixing the immediate syntax/runtime issues. Then I (2) make model weight loading robust by falling back to an all-zero (uninformed) predictor if the external weight files aren’t present, so the pipeline always completes and writes `submission.csv`. Next I (3) fix the DICOM JPEG-decompression crash by using pydicom’s `pixel_array` when available and otherwise falling back to a safe default image, avoiding hard failures on compressed slices. Finally I (4) guarantee the submission matches `test.csv` row order/keys by merging predictions onto `test.csv` and filling any missing rows, ensuring a valid 14536-row submission.'
- What this solution (achieved 0.91732) has done: 'Your current score (0.91732, lower-is-better) is worse than the target (0.61637), so we should legitimately improve predictions with minimal risk. The biggest avoidable degradation is that `convnext_base()` is being instantiated without ImageNet pretrained weights, which makes the stage-1 CNN features/predictions essentially random with your external checkpoint (if it’s a head-only/finetune checkpoint) and/or much weaker than intended; enabling the correct pretrained weights is a minimal, core-logic-preserving fix. I also make weight-loading robust to common checkpoint formats (`state_dict` key, `module.` prefix) so your provided weights actually load when present, improving performance without changing architecture or training. Everything else (data flow, model definitions, inference loops, submission formatting) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 0.91732) has done: 'We keep your two-stage ConvNeXt+GRU inference exactly as-is, but fix one major mismatch that hurts log-loss: the patient_overall probability should be consistent with the per-level probabilities. When stage2 runs, we minimally “OR”-calibrate patient_overall from the seven C-level probabilities (using a numerically-stable union formula) and blend it with the GRU output, which typically reduces the weighted log-loss on the heavily-weighted overall rows. When stage2 does not run (missing LSTM weights), we still compute patient_overall from the C-level outputs instead of using a flat constant. All changes preserve evaluation semantics (probabilities only) and still produce a valid `submission.csv` aligned to `test.csv`.'
- What this solution (achieved 0.91732) has done: 'Your current score (0.91732, lower-is-better) is far worse than the target (0.61637), so we should make a small, legitimate improvement without changing the model or inference design. The biggest low-risk win here is to ensure `patient_overall` is never *lower* than what the 7 vertebra probabilities imply, because the metric heavily weights `patient_overall` and inconsistent low overall predictions are strongly penalized. Concretely, we keep your existing OR-based calibration/blending, but replace the blend with a monotonic “at least OR” constraint: `patient_overall = max(stage2_pred, OR(per-level))` (or `OR` if stage2 isn’t available). This preserves your core logic and only adjusts final post-processing to better match the label semantics and metric.'
- What this solution (achieved 0.91732) has done: 'Your score is worse than the target (0.91732 vs 0.61637; lower-is-better), so we should make small, legitimate changes that improve log-loss without changing the model/inference design. The biggest likely issue is that the ConvNeXt classifier head is randomly initialized while your checkpoint may only partially match; we keep the same architecture but load weights with `strict=False` so at least the matching backbone/head weights are applied instead of failing/being skipped. Next, we compute `patient_overall` directly from the seven vertebra probabilities when stage2 runs (or as a blend), because the metric heavily weights this label and consistency usually improves weighted log-loss. Finally, we keep the submission alignment exactly as you already do, still writing a valid `submission.csv`.'
- What this solution (achieved 0.91732) has done: 'We keep your two-stage ConvNeXt+GRU inference intact and focus on a small but high-impact metric-alignment fix: compute `patient_overall` directly from the seven vertebra probabilities for each study (and only lightly keep the GRU output as a secondary signal when available). Because `patient_overall` is heavily weighted in the competition loss, this consistent “OR” aggregation usually reduces weighted log loss without changing model architecture or training. We also ensure the OR uses the final per-study C1–C7 predictions (already averaged over slices) and applies a monotonic constraint so `patient_overall` can’t be lower than what the vertebrae imply. The rest of the pipeline, paths, and submission alignment remain unchanged and still writes a valid `submission.csv`.'
- What this solution (achieved 0.95402) has done: 'Your current score (0.91732, lower-is-better) is far worse than the target (0.61637), so we should make a small, low-risk improvement that better matches the competition’s weighted log-loss without changing your model or inference loops. The main change is to calibrate the per-vertebra probabilities with a simple, train-derived per-label prior (a multiplicative logit shift) before building the patient_overall “OR” probability; this usually improves log-loss because CNN outputs can be systematically miscalibrated when checkpoints/heads don’t perfectly match. We keep your stage1/stage2 models, data loading, and averaging exactly the same, and only adjust post-processing of probabilities. We also apply the same calibration to the fallback constants (P_C / P_OVERALL) so behavior stays consistent when weights are missing, and we still write a valid `submission.csv` aligned to `test.csv`.'
- What this solution (achieved 0.93244) has done: 'Your current score (0.95402, lower-is-better) is much worse than the target (0.61637), so we should make a small, legitimate post-processing change that improves weighted log-loss without touching your model architectures or inference loops. The biggest likely avoidable penalty is miscalibration: your current prior-based logit shift is fairly strong and is being applied multiple times (per-C + then again to patient_overall), which can push probabilities away from well-calibrated values and hurt log-loss. I keep the same calibration method but reduce its strength and avoid the extra final calibration on `patient_overall` after the OR/max step (still keeping clipping and the monotonic “overall >= OR” constraint). This is a minimal change expected to move probabilities closer to the data priors and reduce overconfident errors, improving score toward the target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.92763) has done: 'Your current score (0.93244, lower-is-better) is still far worse than the target (0.61637), so we should make a small post-processing change that legitimately improves the weighted log-loss without touching your models, feature extraction, or inference loops. The most likely remaining avoidable penalty is miscalibration: your prior logit-shift can still push probabilities too far from the model outputs, hurting log-loss when wrong. I keep the exact same calibration method but reduce its strength further and (importantly) only apply the prior shift when stage-1/2 weights are missing (fallback mode), while leaving learned predictions unshifted (still clipped) so we don’t distort a trained model’s calibration. Patient_overall still be enforced to be at least the OR of C1–C7 (and blended with GRU if present), preserving your core logic and submission format.'
- What this solution (achieved 0.92763) has done: 'Your score is far worse than the target (0.92763 vs 0.61637; lower-is-better), so we should improve predictions with the smallest safe change that aligns with the weighted log-loss. The biggest likely avoidable penalty is overconfident slice-averaged probabilities from stage1/stage2; a minimal, metric-aligned fix is to apply a *very light* temperature smoothing (logit scaling) to all predicted probabilities before the patient_overall “OR” aggregation, which reduces extreme 0/1 errors that are heavily punished by log-loss. This keeps the same models, the same inference loops, and the same patient_overall consistency logic, only adjusting probability calibration. We also keep the existing fallback-prior calibration behavior unchanged (only used when weights are missing), and still write a valid `submission.csv` aligned to `test.csv`.'

# 9. Code solution

## === cell 0
import importlib

_optional = ["pylibjpeg", "gdcm"]
available_optional = {m: importlib.util.find_spec(m) is not None for m in _optional}
available_optional



## === cell 1
import os
import glob
import time
import sys

import numpy as np
import pandas as pd

import cv2
from tqdm import tqdm

import torch
import torch.nn as nn

import pydicom

from torchvision.models.convnext import convnext_base, ConvNeXt_Base_Weights
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(0)
np.random.seed(0)



## === cell 2
BASE_INPUT = "../input/rsna-2022-cervical-spine-fracture-detection"
TEST_PATH = f"{BASE_INPUT}/test_images"

assert os.path.exists(f"{BASE_INPUT}/test.csv"), "test.csv not found at expected path"
assert os.path.exists(TEST_PATH), "test_images folder not found at expected path"



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
    df_test = pd.read_csv(f"{BASE_INPUT}/test.csv")

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




## === cell 5
test_df = load_df_test()
study_id_list = list(test_df.StudyInstanceUID.unique())
len(study_id_list), study_id_list[:3]



## === cell 6
selected_image_dict = {}
uid_dicom_files = {}

for uid in study_id_list:
    dicom_files = sorted(glob.glob(os.path.join(TEST_PATH, uid, "*.dcm")))
    uid_dicom_files[uid] = dicom_files

    n = len(dicom_files)
    if n == 0:
        selected_image_dict[uid] = []
        continue

    middle_slice = int(n / 2)
    num_left_images = num_right_images = int(0.15 * n)

    left = list(
        np.arange(max(1, middle_slice - num_left_images), middle_slice, 1, dtype=int)
    )
    right = list(
        np.arange(
            middle_slice + 1,
            min(n - 2, middle_slice + num_right_images) + 1,
            1,
            dtype=int,
        )
    )

    selected = [i for i in (left + right) if 1 <= i <= n - 2]
    if len(selected) == 0:
        selected = [i for i in range(1, min(n - 2, 6) + 1)]

    selected_image_dict[uid] = selected



## === cell 7
if len(study_id_list) > 0:
    uid0 = study_id_list[0]
    print(
        uid0,
        "n_dicoms=",
        len(uid_dicom_files[uid0]),
        "selected_n=",
        len(selected_image_dict[uid0]),
    )
    print("selected idx head:", selected_image_dict[uid0][:10])



## === cell 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device




## === cell 9
def safe_pixel_array(ds):
    """Return ds.pixel_array if possible; otherwise raise."""
    return ds.pixel_array


def window_from_array(img, slope=1.0, intercept=0.0, WL=400, WW=1800):
    img = img.astype(np.float32) * float(slope) + float(intercept)
    upper, lower = WL + WW // 2, WL - WW // 2
    X = np.clip(img, lower, upper)
    X = X - np.min(X)
    mx = np.max(X)
    if mx > 0:
        X = X / mx
    X = (X * 255.0).astype("uint8")
    return X


def window(data, WL=400, WW=1800):
    slope = float(getattr(data, "RescaleSlope", 1.0))
    intercept = float(getattr(data, "RescaleIntercept", 0.0))
    img = safe_pixel_array(data)
    return window_from_array(img, slope=slope, intercept=intercept, WL=WL, WW=WW)


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
    def __init__(self, uid, image_list, target_size, crop_size):
        self.uid = uid
        self.image_list = image_list
        self.target_size = target_size
        self.crop_size = crop_size

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        dicom_files = uid_dicom_files[self.uid]
        idx = int(self.image_list[index])

        def blank():
            return np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)

        try:
            paths = [dicom_files[idx - 1], dicom_files[idx], dicom_files[idx + 1]]
            data_list = [pydicom.dcmread(p, force=True) for p in paths]
            imgs = []
            for ds in data_list:
                try:
                    imgs.append(window(ds))
                except Exception:
                    imgs.append(np.zeros((512, 512), dtype=np.uint8))
            stacked_img = np.stack(imgs, axis=-1)
            stacked_img = cv2.resize(stacked_img, (self.target_size, self.target_size))
        except Exception:
            stacked_img = blank()

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
def _clean_state_dict_for_load(sd):
    if (
        isinstance(sd, dict)
        and "state_dict" in sd
        and isinstance(sd["state_dict"], dict)
    ):
        sd = sd["state_dict"]
    if isinstance(sd, dict):
        keys = list(sd.keys())
        if len(keys) > 0 and all(k.startswith("module.") for k in keys):
            sd = {k[len("module.") :]: v for k, v in sd.items()}
    return sd


def load_weights_safely(model, path, device, strict=True):
    try:
        obj = torch.load(path, map_location=device)
        sd = _clean_state_dict_for_load(obj)
        model.load_state_dict(sd, strict=strict)
        return True
    except Exception as e:
        print(f"[WARN] Failed to load weights from {path}: {e}")
        return False


lv1_model = ConvNextCNN_B_Feature().to(device).eval()
lv2_model = (
    CSFNet(input_len=config["feature_size"], lstm_size=config["lstm_size"])
    .to(device)
    .eval()
)

lv1_w = "../input/cnn-lstm-oct-21/run_5/run_5/model_3.pth"
lv2_w = "../input/cnn-lstm-oct-21/run_5b/run_5b/model_lstm_3.pth"

has_lv1 = os.path.exists(lv1_w)
has_lv2 = os.path.exists(lv2_w)

if has_lv1:
    has_lv1 = load_weights_safely(lv1_model, lv1_w, device, strict=False)
if has_lv2:
    has_lv2 = load_weights_safely(lv2_model, lv2_w, device, strict=True)

has_lv1, has_lv2




## === cell 15
def _logit(p):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def _sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def calibrate_with_prior(p, prior, strength=0.35):
    return _sigmoid(_logit(p) + float(strength) * _logit(prior))


def temp_scale_prob(p, temperature=1.12):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    t = float(temperature)
    if t <= 0:
        return p.astype(np.float64)
    return _sigmoid(_logit(p) / t)


train_df = pd.read_csv(f"{BASE_INPUT}/train.csv")
label_cols = [f"C{k}" for k in range(1, 8)] + ["patient_overall"]
priors = train_df[label_cols].mean().clip(1e-4, 1 - 1e-4).to_dict()
priors



## === cell 16
submission_records = []
feature_array_dict = {}

P_C = 0.05
P_OVERALL = 0.10

CAL_C_STRENGTH_FALLBACK = 0.10
CAL_OVERALL_STRENGTH_FALLBACK = 0.08

TEMP_STAGE1 = 1.12
TEMP_STAGE2 = 1.08

run_stage1 = has_lv1 and (len(study_id_list) > 0)

if run_stage1:
    for uid in tqdm(study_id_list, desc="Stage1 per-study"):
        image_list = selected_image_dict[uid]
        if len(image_list) == 0:
            feature_array_dict[uid] = np.zeros(
                (0, config["feature_size"]), dtype=np.float32
            )
            for k in range(1, 8):
                p0 = calibrate_with_prior(
                    P_C, priors[f"C{k}"], strength=CAL_C_STRENGTH_FALLBACK
                )
                submission_records.append(
                    {"row_id": f"{uid}_C{k}", "fractured": float(p0)}
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
            pin_memory=True,
            drop_last=False,
        )

        preds_uid = []
        feature_array = np.zeros(
            (len(dataset), config["feature_size"]), dtype=np.float32
        )

        for i, images in enumerate(generator):
            with torch.no_grad():
                start = i * config["batch_size_image_level"]
                end = min(
                    start + config["batch_size_image_level"], len(generator.dataset)
                )

                images = images.to(device, non_blocking=True)
                features, preds = lv1_model(images)

                feature_array[start:end] = np.squeeze(features.detach().cpu().numpy())
                preds_uid.append(preds.sigmoid().detach().cpu())

        feature_array_dict[uid] = feature_array
        mean_preds = torch.mean(torch.cat(preds_uid, dim=0), dim=0).numpy()

        mean_preds = temp_scale_prob(mean_preds, temperature=TEMP_STAGE1)

        for k in range(1, 8):
            submission_records.append(
                {"row_id": f"{uid}_C{k}", "fractured": float(mean_preds[k - 1])}
            )
else:
    for uid in study_id_list:
        feature_array_dict[uid] = np.zeros(
            (config["seq_len"], config["feature_size"]), dtype=np.float32
        )
        for k in range(1, 8):
            p0 = calibrate_with_prior(
                P_C, priors[f"C{k}"], strength=CAL_C_STRENGTH_FALLBACK
            )
            submission_records.append({"row_id": f"{uid}_C{k}", "fractured": float(p0)})

len(submission_records)



## === cell 17
run_stage2 = has_lv2 and (len(study_id_list) > 0)

if run_stage2:
    dataset2 = CSFInstanceDataset(
        feature_array_dict=feature_array_dict,
        study_id_list=study_id_list,
        seq_len=config["seq_len"],
    )
    generator2 = DataLoader(
        dataset=dataset2,
        batch_size=config["batch_size_patient_level"],
        shuffle=False,
        pin_memory=True,
    )

    for features, list_uid in tqdm(
        generator2, total=len(generator2), desc="Stage2 batches"
    ):
        with torch.no_grad():
            features = features.to(device, non_blocking=True)
            preds = lv2_model(features).sigmoid().detach().cpu().numpy().reshape(-1)

        preds = temp_scale_prob(preds, temperature=TEMP_STAGE2)

        for j in range(len(list_uid)):
            submission_records.append(
                {
                    "row_id": f"{list_uid[j]}_patient_overall",
                    "fractured": float(preds[j]),
                }
            )
else:
    for uid in study_id_list:
        p0 = calibrate_with_prior(
            P_OVERALL, priors["patient_overall"], strength=CAL_OVERALL_STRENGTH_FALLBACK
        )
        submission_records.append(
            {"row_id": f"{uid}_patient_overall", "fractured": float(p0)}
        )



## === cell 18
pred_df = (
    pd.DataFrame(submission_records)
    .groupby("row_id", as_index=False)["fractured"]
    .mean()
)

sub = test_df[["row_id"]].merge(pred_df, on="row_id", how="left")

if "prediction_type" in test_df.columns:
    tmp = test_df[["row_id", "prediction_type"]].copy()
    sub = sub.merge(tmp, on="row_id", how="left")
    miss = sub["fractured"].isna()

    p_overall_fill = float(
        calibrate_with_prior(
            P_OVERALL,
            priors["patient_overall"],
            strength=CAL_OVERALL_STRENGTH_FALLBACK,
        )
    )
    sub.loc[miss & (sub["prediction_type"] == "patient_overall"), "fractured"] = (
        p_overall_fill
    )

    for k in range(1, 8):
        p_c_fill = float(
            calibrate_with_prior(P_C, priors[f"C{k}"], strength=CAL_C_STRENGTH_FALLBACK)
        )
        sub.loc[miss & (sub["prediction_type"] == f"C{k}"), "fractured"] = p_c_fill

    avg_c_prior = float(np.mean([priors[f"C{k}"] for k in range(1, 8)]))
    sub.loc[sub["fractured"].isna(), "fractured"] = float(
        calibrate_with_prior(P_C, avg_c_prior, strength=CAL_C_STRENGTH_FALLBACK)
    )
    sub = sub.drop(columns=["prediction_type"])
else:
    avg_c_prior = float(np.mean([priors[f"C{k}"] for k in range(1, 8)]))
    sub["fractured"] = sub["fractured"].fillna(
        float(calibrate_with_prior(P_C, avg_c_prior, strength=CAL_C_STRENGTH_FALLBACK))
    )

sub2 = sub.copy()
sub2["StudyInstanceUID"] = sub2["row_id"].str.rsplit("_", n=1, expand=True)[0]
sub2["pred_type"] = sub2["row_id"].str.rsplit("_", n=1, expand=True)[1]

c_df = sub2[sub2["pred_type"].isin([f"C{k}" for k in range(1, 8)])].copy()
overall_df = sub2[sub2["pred_type"] == "patient_overall"].copy()

c_pivot = c_df.pivot(
    index="StudyInstanceUID", columns="pred_type", values="fractured"
).reindex(columns=[f"C{k}" for k in range(1, 8)])

for k in range(1, 8):
    c_pivot[f"C{k}"] = c_pivot[f"C{k}"].fillna(
        float(
            calibrate_with_prior(P_C, priors[f"C{k}"], strength=CAL_C_STRENGTH_FALLBACK)
        )
    )

p_or = 1.0 - np.prod(1.0 - c_pivot.values.astype(np.float64), axis=1)
p_or = np.clip(p_or, 1e-5, 1.0 - 1e-5)
p_or_map = dict(zip(c_pivot.index.tolist(), p_or.tolist()))

if len(overall_df) > 0:
    ovals = overall_df["fractured"].values.astype(np.float64)
    orvals = (
        overall_df["StudyInstanceUID"]
        .map(p_or_map)
        .fillna(
            float(
                calibrate_with_prior(
                    P_OVERALL,
                    priors["patient_overall"],
                    strength=CAL_OVERALL_STRENGTH_FALLBACK,
                )
            )
        )
        .values.astype(np.float64)
    )

    beta = 0.10  # keep original: small carry-over from GRU; OR does main work
    new_overall = (1.0 - beta) * orvals + beta * ovals

    new_overall = np.maximum(new_overall, orvals)
    new_overall = np.clip(new_overall, 1e-5, 1.0 - 1e-5)

    overall_df.loc[:, "fractured"] = new_overall.astype(np.float32)

    sub = sub.merge(
        overall_df[["row_id", "fractured"]].rename(
            columns={"fractured": "fractured_new"}
        ),
        on="row_id",
        how="left",
    )
    m = sub["fractured_new"].notna()
    sub.loc[m, "fractured"] = sub.loc[m, "fractured_new"]
    sub = sub.drop(columns=["fractured_new"])

sub["fractured"] = sub["fractured"].astype(np.float32).clip(1e-5, 1 - 1e-5)

sub.shape, sub.head()



## === cell 19
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with rows:", len(sub))
print(sub.head(10).to_string(index=False))
