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

0.432765994449533

# 6. Current score

0.56492

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41874) has done: 'I remove the notebook-style `!pip` installs (they break in a .py/script context) and instead make optional imports robust so the pipeline runs in the provided environment. I also fix missing/incorrect cell ordering issues (cell 0 -> cell 1, missing `glob`/`DataLoader` due to earlier failure) and add safe fallbacks when optional model packages/checkpoints are unavailable, so a valid `submission.csv` is always produced. Finally, I ensure the submission strictly matches `test.csv` row order and required columns, and clip probabilities to avoid log-loss infinities (score-stable, correctness fix).'
- What this solution (achieved 0.7459) has done: 'Your current score (1.41874, lower-is-better) is far worse than the target (0.4328), and the main reason is that in this environment your segmentation/detection models are not actually loading, so you’re effectively submitting near-constant probabilities. To move the score toward the target without changing the core inference logic, I (1) make the checkpoint/path discovery robust for both `../input` and `/kaggle/data`, (2) add safe, minimal autocasting of the input dtype to match whatever the loaded models expect (prevents silent mis-inference), and (3) improve the fallback from a constant to a label-prior baseline learned from `train.csv` (still a legitimate, simple probabilistic model; no architecture/training loop changes). This should substantially reduce log loss versus a constant while keeping your pipeline and semantics intact. The submission formatting and row ordering remain unchanged.'
- What this solution (achieved 0.7459) has done: 'I make two minimal, score-relevant changes that keep your core inference intact but reduce log-loss: (1) fix the fallback path so you generate one prediction per study (not per slice), because the current priors fallback is being broadcast to slices and then max-pooled, which unintentionally inflates probabilities and hurts weighted log loss. (2) when the models are unavailable, compute `patient_overall` consistently from the vertebra priors (and keep clipping), instead of using an independent prior that can be miscalibrated relative to the max rule you apply later. Everything else (model loading, segmentation/detection pipeline, post-processing, submission order/format) remains the same.'
- What this solution (achieved 0.69465) has done: 'Your current score (0.7459, lower-is-better) is still far from the target (0.4328), and the most likely cause is that the external segmentation/detector models are not loading in this environment, so you’re effectively using a coarse prior-only fallback. To move the score toward the target without changing your core model logic, I keep your inference pipeline intact but improve the fallback calibration using only `train.csv`: compute vertebra priors and a *conditional* mapping for `patient_overall` (instead of `patient_overall = max(C1..C7)`), plus apply logit-space shrinkage to avoid overly confident priors that hurt weighted log loss. This keeps evaluation semantics the same (still probabilistic predictions per label) and only changes what happens when the real models aren’t available. The submission formatting/order stays exactly aligned to `test.csv`, and probabilities remain clipped to avoid log-loss infinities.'
- What this solution (achieved 0.56277) has done: 'I fix the `fillna` crash in the submission-building cell by replacing the invalid `fillna(ndarray)` call with an index-aligned Series assignment that preserves row order. I also correct the initial cell numbering (cell 0 → cell 1, etc.) so the script runs in the required format end-to-end. These changes are execution-only (no core model/inference changes) and still produce `submission.csv` with the exact required columns and ordering from `test.csv`. The fallback calibration logic remains identical; we only make the NaN-fill step compatible with pandas on Python 3.11.'
- What this solution (achieved 0.56277) has done: 'Your current score (0.56277, lower-is-better) is worse than the target (0.43277), so we should improve (reduce) log loss with minimal, safe changes. The biggest score lever without changing the core model pipeline is to fix a mismatch: in the model-available path you currently max-pool slice-level *detector scores* for `patient_overall`, but the metric’s `patient_overall` is a study-level “any fracture” label; using a calibrated mapping from vertebra probabilities to `patient_overall` (already learned in your fallback calibration) is typically better-calibrated than max(score). I therefore keep all segmentation/detection inference intact, but change only the study-level aggregation: aggregate per-vertebra probabilities across slices, then compute `patient_overall` from those aggregated C1–C7 via the same learned logistic mapping (`any_a/any_b`). I also ensure `patient_overall` is never below `max(C1..C7)` (logical consistency) and keep the same clipping/submission formatting.'
- What this solution (achieved 0.56277) has done: 'Your current score (0.56277, lower-is-better) is still worse than the target (0.43277), so we should reduce log loss with a small, low-risk calibration change rather than altering your segmentation/detection pipeline. The main issue is that your per-study aggregation uses a hard max over slices for each vertebra, which tends to overestimate probabilities (hurting log loss); we replace that with a “noisy-OR” aggregation across slices (still consistent with “any slice shows fracture”), which is typically better-calibrated. We keep your learned `patient_overall` mapping exactly as-is, but compute it from the newly aggregated per-vertebra probabilities, and enforce `patient_overall >= max(C1..C7)` for consistency. Everything else (model loading, inference, fallback behavior, submission format/order, clipping) stays the same.'
- What this solution (achieved 0.56277) has done: 'We keep your segmentation+effdet inference untouched and only adjust the study-level aggregation in the model-available path, because your current noisy-OR across slices tends to inflate probabilities and can worsen weighted log loss when the detector produces many small non-zero scores. Specifically, we replace noisy-OR with a calibrated “softmax pooling” aggregator (log-sum-exp over slice logits) that smoothly interpolates between mean and max, reducing overconfidence while still capturing “any-slice positive” behavior. We also compute `patient_overall` from the aggregated C1–C7 via your existing learned mapping (and enforce `patient_overall >= max(C1..C7)`), which is consistent with the metric emphasis on the overall label. All file paths, fallback behavior, clipping, and submission formatting/order remain the same, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.56277) has done: 'We keep your segmentation+effdet inference intact and only make a minimal, score-relevant calibration adjustment in the model-available path: tune the slice-to-study aggregation “softmax pooling” temperature (`tau`) using the public `train.csv` labels, so probabilities are less overconfident and closer to the metric-optimal calibration. This directly targets weighted log loss without changing architectures, loss functions, or training loops, and it stays within your existing aggregation semantics (a smooth max). We also ensure `patient_overall` is computed from aggregated vertebrae via your already-learned mapping (and constrained to be ≥ max(C1..C7)), which matches the competition’s emphasis on the overall label. The fallback (no-model) behavior and submission formatting/order remain unchanged, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.56277) has done: 'Your current score (0.56277, lower-is-better) is still above the target (0.43277), so we should reduce log loss with a small calibration/aggregation change rather than altering the segmentation+detector core inference. The biggest low-risk lever is how we convert slice-level probabilities into study-level probabilities: using a single fixed tau for all labels can miscalibrate the heavily-weighted `patient_overall` versus vertebrae. I keep your log-sum-exp pooling semantics but tune separate temperatures: one tau shared across C1–C7 and a separate tau for `patient_overall`, selected on a small deterministic subset of train studies using the same proxy loss you already compute. Everything else (model loading, per-slice inference, fallback logic, clipping, submission format/order) remains unchanged, and it still writes `submission.csv`.'
- What this solution (achieved 0.56277) has done: 'We keep your segmentation+effdet inference and the existing log-sum-exp pooling family intact, and only make a minimal calibration change that can reduce weighted log loss toward the target. Specifically, we (1) tune pooling temperatures on train with an objective that matches the competition metric more closely by applying the `patient_overall` mapping after pooling (as you do at test time) rather than pooling a slice-level `patient_overall` score, and (2) optionally tune a single global logit-scaling factor on the pooled study-level probabilities to correct over/under-confidence (a classic log-loss calibration step that preserves ranking and core semantics). These are small, deterministic changes that don’t alter architectures, losses, or training loops, but typically improve log loss. Submission format/ordering and clipping remain unchanged, and the script still writes `submission.csv`.'
- What this solution (achieved 0.56277) has done: 'We keep your segmentation+effdet inference and your calibrated log-sum-exp pooling unchanged, and only add a minimal calibration step that directly targets weighted log loss: label-wise probability offsets (biases) in logit-space learned from `train.csv`. This is a small, deterministic post-processing layer applied after your existing pooling+`patient_overall` mapping (and after global logit scaling), so it doesn’t alter core model logic or any training loops. The offsets are tuned on the tiny `train.csv` labels using a simple grid search to reduce the proxy competition loss, which should move your score down from 0.56277 toward the 0.43277 target. Submission format, row order, and clipping remain intact, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.56492) has done: 'Your current score (0.56277, lower-is-better) is still worse than the target (0.43277), so we should reduce weighted log loss with a minimal calibration change that doesn’t alter your core segmentation+effdet inference. The biggest low-risk lever left is calibrating the final study-level probabilities to the label distribution: we fit per-label logit biases *and* a single global logit scale using only `train.csv`, but crucially we fit them against the **priors-only fallback predictions** (not against a tiny subset of detector inference), which is deterministic, fast, and avoids overfitting to 32 studies. This keeps your architecture/inference intact and only adjusts post-processing in a metric-aligned way. We also keep your existing pooling and patient_overall mapping, and we still write a valid `submission.csv` aligned to `test.csv` row order.'

# 9. Code solution

## === cell 0
import os
import sys
import re
import glob
import math
import random
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import torchvision.transforms.functional as TF

warnings.filterwarnings("ignore")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

try:
    import pydicom as dicom
except Exception as e:
    raise RuntimeError("pydicom is required to read DICOMs in this solution.") from e

try:
    import pylibjpeg  # optional (JPEG compressed DICOM support)

    _HAS_PYLIBJPEG = True
except Exception:
    _HAS_PYLIBJPEG = False
    pylibjpeg = None


def _maybe_add_path(p):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.append(p)


_maybe_add_path("../input/effdet-models/effdet")
_maybe_add_path("../input/effdet-models/timm-pytorch-image-models")
_maybe_add_path("../input/effdet-models/omegaconf")
_maybe_add_path("../input/effdet-models/efficientunet-pytorch-0.0.6")
_maybe_add_path("../input/effdet-models/yolov7")


def _resolve_first_existing(candidates):
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return candidates[0] if candidates else None


def _resolve_dir_first_existing(candidates):
    for p in candidates:
        if p and os.path.isdir(p):
            return p
    return candidates[0] if candidates else None




## === cell 1
IMAGES_DIR = "../input/rsna-2022-cervical-spine-fracture-detection/test_images"
TRAIN_IMAGES_PATH = "../input/rsna-2022-cervical-spine-fracture-detection/train_images"
TEST_IMAGES_PATH = "../input/rsna-2022-cervical-spine-fracture-detection/test_images"

TEST_IMAGES_PATH = _resolve_dir_first_existing(
    [
        TEST_IMAGES_PATH,
        "/kaggle/input/rsna-2022-cervical-spine-fracture-detection/test_images",
        "/kaggle/data/rsna-2022-cervical-spine-fracture-detection/test_images",
        "../kaggle/data/rsna-2022-cervical-spine-fracture-detection/test_images",
        "../input/rsna-2022-cervical-spine-fracture-detection/test_images",
    ]
)
IMAGES_DIR = TEST_IMAGES_PATH

print("TEST_IMAGES_PATH:", TEST_IMAGES_PATH)



## === cell 2
segmentation_checkpoint = (
    "../input/effdet-models/axial_segmentation_effseg_132508-epoch-100.pth"
)
axial_det_checkpoint1 = (
    "../input/effdet-models/axial_detection_effdet_151039-epoch-60.pth"
)

segmentation_checkpoint = _resolve_first_existing(
    [
        segmentation_checkpoint,
        "/kaggle/input/effdet-models/axial_segmentation_effseg_132508-epoch-100.pth",
        "/kaggle/data/effdet-models/axial_segmentation_effseg_132508-epoch-100.pth",
        "../kaggle/data/effdet-models/axial_segmentation_effseg_132508-epoch-100.pth",
    ]
)
axial_det_checkpoint1 = _resolve_first_existing(
    [
        axial_det_checkpoint1,
        "/kaggle/input/effdet-models/axial_detection_effdet_151039-epoch-60.pth",
        "/kaggle/data/effdet-models/axial_detection_effdet_151039-epoch-60.pth",
        "../kaggle/data/effdet-models/axial_detection_effdet_151039-epoch-60.pth",
    ]
)

print(
    "segmentation_checkpoint:",
    segmentation_checkpoint,
    "exists:",
    os.path.exists(segmentation_checkpoint),
)
print(
    "axial_det_checkpoint1:",
    axial_det_checkpoint1,
    "exists:",
    os.path.exists(axial_det_checkpoint1),
)




## === cell 3
def rescale_img_to_hu(dcm_ds):
    """Rescales the image to Hounsfield units."""
    slope = getattr(dcm_ds, "RescaleSlope", 1.0)
    intercept = getattr(dcm_ds, "RescaleIntercept", 0.0)
    return dcm_ds.pixel_array.astype(np.float32) * float(slope) + float(intercept)


def normalize_hu_t(data):
    return np.clip(data, a_min=-2242.0, a_max=2242.0) / 2242.0


def load_dicom(path):
    """
    Supports loading both regular and (if pylibjpeg/gdcm installed) JPEG compressed DICOM.
    In this environment, pylibjpeg may be unavailable; pydicom can still read many uncompressed DICOMs.
    """
    ds = dicom.dcmread(path)
    img = rescale_img_to_hu(ds)
    px = getattr(ds, "PixelSpacing", [1.0])[0]
    return img, float(px)




## === cell 4
test_slices = glob.glob(f"{TEST_IMAGES_PATH}/*/*.dcm")
if len(test_slices) == 0:
    raise RuntimeError(
        f"No DICOM slices found under {TEST_IMAGES_PATH}. Check path availability."
    )

pairs = []
pat = re.compile(re.escape(TEST_IMAGES_PATH) + r"/(.*?)/(\d+)\.dcm$")
for s in test_slices:
    m = pat.search(s)
    if m:
        pairs.append((m.group(1), int(m.group(2))))
df_test_slices = pd.DataFrame(pairs, columns=["StudyInstanceUID", "Slice"]).astype(
    {"Slice": int}
)

df_test_slices["Start"] = df_test_slices.groupby("StudyInstanceUID")["Slice"].transform(
    "min"
)
df_test_slices = df_test_slices.sort_values(["StudyInstanceUID", "Slice"]).reset_index(
    drop=True
)
print(df_test_slices.head())
print(
    "n_slices:",
    len(df_test_slices),
    "n_studies:",
    df_test_slices["StudyInstanceUID"].nunique(),
)




## === cell 5
class DcmDataSet(torch.utils.data.Dataset):
    def __init__(self, df, path, image_size=512):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.path = path
        self.image_size = image_size
        self.transform = T.Resize((image_size, image_size))

    def __getitem__(self, i):
        s = self.df.iloc[i]
        gpath = os.path.join(self.path, s.StudyInstanceUID, f"{int(s.Slice)}.dcm")
        try:
            g, pixel_spacing = load_dicom(gpath)
            g = normalize_hu_t(g)
        except Exception:
            g = np.zeros((512, 512), dtype=np.float32)
            pixel_spacing = 1.0

        x = torch.as_tensor(g, dtype=torch.float32).unsqueeze(0)  # 1xHxW
        x = self.transform(x)
        is_start = bool(int(s.Slice) == int(s.Start))
        return x, float(pixel_spacing), is_start

    def __len__(self):
        return len(self.df)


ds = DcmDataSet(df_test_slices, IMAGES_DIR)


def _collate_fn(batch):
    xs, pxs, starts = zip(*batch)
    x = torch.stack(xs, dim=0)
    px = torch.tensor(pxs, dtype=torch.float32)
    st = torch.tensor(starts, dtype=torch.bool)
    return x, px, st


batch_size = 8
dl = DataLoader(
    ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,  # safer in Kaggle/script environments for pydicom
    pin_memory=torch.cuda.is_available(),
    collate_fn=_collate_fn,
)

x, pixel_spacings, is_start = next(iter(dl))
print("batch x:", x.shape, x.min().item(), x.max().item())
print("pixel_spacings:", pixel_spacings[:5])
print("is_start:", is_start[:10])



## === cell 6
_HAS_EFFICIENTUNET = False
_HAS_EFFDET = False

try:
    from efficientunet import get_efficientunet_b5  # type: ignore

    _HAS_EFFICIENTUNET = True
except Exception:
    _HAS_EFFICIENTUNET = False

try:
    from effdet import create_model  # type: ignore

    _HAS_EFFDET = True
except Exception:
    _HAS_EFFDET = False

print("HAS_EFFICIENTUNET:", _HAS_EFFICIENTUNET, "HAS_EFFDET:", _HAS_EFFDET)




## === cell 7
def get_axial_segmentation_model(checkpoint):
    if not _HAS_EFFICIENTUNET or (not os.path.exists(checkpoint)):
        return None
    model = get_efficientunet_b5(out_channels=2, concat_input=True, pretrained=False)
    state = torch.load(checkpoint, map_location=torch.device(device))
    model.load_state_dict(state["model"])
    model.eval()
    return model.to(device)


seg_model = get_axial_segmentation_model(segmentation_checkpoint)


def get_axial_detection_model(checkpoint, image_size=512):
    if not _HAS_EFFDET or (not os.path.exists(checkpoint)):
        return None
    model = create_model(
        "efficientdetv2_dt",
        bench_task="predict",
        num_classes=1,
        image_size=(image_size, image_size),
        pretrained=False,
        max_det_per_image=1,
    )
    state = torch.load(checkpoint, map_location=torch.device(device))
    model.load_state_dict(state["model"])
    model = model.eval()
    return model.to(device)


effdet_model = get_axial_detection_model(axial_det_checkpoint1, 768)

print("seg_model loaded:", seg_model is not None)
print("effdet_model loaded:", effdet_model is not None)




## === cell 8
def get_axial_boundary_from_segmentation(
    seg, pixel_spacing, throw=100, tol=0.2, max_mm=100
):
    image_size = seg.shape[0]
    min_size = min(image_size, max_mm / float(pixel_spacing))

    rows, columns = seg.nonzero(as_tuple=True)
    rows, _ = torch.sort(rows)
    columns, _ = torch.sort(columns)

    throw = min(len(rows) // 2, int(throw))

    if len(rows) == 0:
        return torch.tensor(
            [0, 0, image_size, image_size], device=seg.device, dtype=torch.float32
        )

    xmin, xmax = columns[throw], columns[-throw - 1]
    ymin, ymax = rows[throw], rows[-throw - 1]

    w = (xmax - xmin) * (1 + tol)
    h = (ymax - ymin) * (1 + tol)
    new_size = max(w.item(), h.item(), float(min_size))
    new_size = min(float(image_size), float(new_size))

    xcenter = (xmax + xmin).float() / 2
    ycenter = (ymax + ymin).float() / 2

    xmin2 = torch.minimum(
        torch.tensor(image_size - new_size, device=seg.device), xcenter - new_size / 2
    ).clamp(min=0)
    ymin2 = torch.minimum(
        torch.tensor(image_size - new_size, device=seg.device), ycenter - new_size / 2
    ).clamp(min=0)

    return torch.stack([xmin2, ymin2, xmin2 + new_size, ymin2 + new_size]).float()


def predict_seg(x, model, seg_img_size=256):
    x = TF.resize(x, (seg_img_size, seg_img_size))
    logits = model(x)
    classification_score, mse_score = logits.sigmoid().chunk(2, dim=1)
    classification_pred = classification_score.gt(0.5).float()
    pred = classification_pred * mse_score
    return pred


def get_axial_boundary_from_seg(segs, pixel_spacings, seg_img_size=256):
    boundary_list = []
    for i in range(segs.shape[0]):
        seg = segs[i, 0, :, :]
        boundary = get_axial_boundary_from_segmentation(
            seg,
            (
                pixel_spacings[i].item()
                if torch.is_tensor(pixel_spacings[i])
                else float(pixel_spacings[i])
            ),
            throw=int(100.0 / 512.0 * seg_img_size),
            tol=0.2,
            max_mm=100.0 / 512.0 * seg_img_size,
        )
        boundary_list.append(boundary)
    boundary_list = torch.stack(boundary_list, dim=0) * (512.0 / float(seg_img_size))
    return boundary_list


def predict_det(x, model):
    pred_result = model(x)  # N x 1 x 6 (as used in original)
    return pred_result[:, 0, :4], pred_result[:, 0, 4]


def crop_resize_images(imgs_tensor, boundary_list, img_size=512):
    croped_list = []
    for i in range(imgs_tensor.shape[0]):
        xmin, ymin, xmax, ymax = boundary_list[i, :]
        xmin, ymin, xmax, ymax = (
            int(xmin.item()),
            int(ymin.item()),
            int(xmax.item()),
            int(ymax.item()),
        )
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = max(xmin + 1, xmax)
        ymax = max(ymin + 1, ymax)
        croped = TF.crop(
            imgs_tensor[i, :, :, :],
            top=ymin,
            left=xmin,
            height=ymax - ymin,
            width=xmax - xmin,
        )
        croped = TF.resize(croped, (img_size, img_size))
        croped_list.append(croped)
    return torch.stack(croped_list, 0)


def get_original_bbox(bbox, boundary, image_size=512.0):
    scale = image_size / (boundary[:, [2]] - boundary[:, [0]]).clamp(min=1e-6)
    org_bbox = bbox / scale
    org_bbox[:, 0] += boundary[:, 0]
    org_bbox[:, 1] += boundary[:, 1]
    org_bbox[:, 2] += boundary[:, 0]
    org_bbox[:, 3] += boundary[:, 1]
    return org_bbox


def get_bbox_class(seg, bbox):
    xmin, ymin, xmax, ymax = bbox.int()
    xmin = int(max(0, xmin.item()))
    ymin = int(max(0, ymin.item()))
    xmax = int(max(xmin + 1, xmax.item()))
    ymax = int(max(ymin + 1, ymax.item()))
    area = seg[ymin:ymax, xmin:xmax]
    if area.numel() == 0:
        return torch.tensor(0, device=seg.device, dtype=torch.float32)
    valid = area[area > 0]
    if valid.numel() == 0:
        return torch.tensor(0, device=seg.device, dtype=torch.float32)
    result = torch.mean(valid)
    result = torch.round(result / 0.125)
    return result


def get_bbox_class_list(seg_list, seg_bboxes):
    class_list = []
    for i in range(seg_list.shape[0]):
        class_index = get_bbox_class(seg_list[i, :, :], seg_bboxes[i, :])
        class_list.append(class_index)
    return torch.stack(class_list)


def get_class_score(scores, class_list, eps=1e-2):
    result = scores.new_zeros((scores.shape[0], 8)) + eps
    class_list = torch.nan_to_num(class_list).long().clamp(min=0, max=7)
    result[torch.arange(scores.shape[0], device=scores.device), class_list] = scores
    return result




## === cell 9
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _weighted_logloss_binary(y, p, w=1.0, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return float(np.mean(-w * (y * np.log(p) + (1 - y) * np.log(1 - p))))


def _fit_fallback_calibration(df_tr):
    """
    Fallback-only calibration learned from train.csv (kept as-is).
    """
    cols_v = [f"C{i}" for i in range(1, 8)]
    n = len(df_tr)
    alpha = 1.0

    priors = {}
    for c in ["patient_overall"] + cols_v:
        s = float(df_tr[c].sum())
        priors[c] = (s + alpha) / (n + 2 * alpha)

    shrink_grid = np.array([0.55, 0.65, 0.75, 0.85, 0.95], dtype=np.float64)
    shrink_by_label = {}
    for c in cols_v:
        y = df_tr[c].astype(int).values
        p0 = np.full(n, priors[c], dtype=np.float64)
        best_s, best_loss = 0.85, 1e9
        for s in shrink_grid:
            p = _sigmoid(_logit(p0) * s)
            loss = _weighted_logloss_binary(y, p, w=1.0)
            if loss < best_loss:
                best_loss, best_s = loss, float(s)
        shrink_by_label[c] = best_s

    y_any = df_tr["patient_overall"].astype(int).values
    p_mat = np.zeros((n, 7), dtype=np.float64)
    for j, c in enumerate(cols_v):
        p0 = np.full(n, priors[c], dtype=np.float64)
        p_mat[:, j] = _sigmoid(_logit(p0) * float(shrink_by_label[c]))
    m = p_mat.max(axis=1)

    a_grid = np.array([2.0, 4.0, 6.0, 8.0, 10.0], dtype=np.float64)
    b_grid = np.array([0.05, 0.10, 0.15, 0.20, 0.25], dtype=np.float64)
    best = {"a": 6.0, "b": 0.15, "loss": 1e9}
    for a in a_grid:
        for b in b_grid:
            p_any = _sigmoid(a * (m - b))
            loss = _weighted_logloss_binary(y_any, p_any, w=7.0)
            if loss < best["loss"]:
                best = {"a": float(a), "b": float(b), "loss": float(loss)}

    return {
        "priors": priors,
        "shrink_by_label": shrink_by_label,
        "any_a": best["a"],
        "any_b": best["b"],
    }


def _load_train_calibration():
    train_csv_path = _resolve_first_existing(
        [
            "../input/rsna-2022-cervical-spine-fracture-detection/train.csv",
            "/kaggle/input/rsna-2022-cervical-spine-fracture-detection/train.csv",
            "/kaggle/data/rsna-2022-cervical-spine-fracture-detection/train.csv",
            "../kaggle/data/rsna-2022-cervical-spine-fracture-detection/train.csv",
        ]
    )
    if train_csv_path is None or not os.path.exists(train_csv_path):
        return None

    df_tr = pd.read_csv(train_csv_path)
    cols = ["patient_overall"] + [f"C{i}" for i in range(1, 8)]
    if not all(c in df_tr.columns for c in cols):
        return None

    return _fit_fallback_calibration(df_tr)


_TRAIN_CAL = _load_train_calibration()
print("TRAIN_CAL available:", _TRAIN_CAL is not None)
if _TRAIN_CAL is not None:
    print("fallback any_a/any_b:", _TRAIN_CAL["any_a"], _TRAIN_CAL["any_b"])
    print("fallback shrink_by_label:", _TRAIN_CAL["shrink_by_label"])


def _load_train_df():
    train_csv_path = _resolve_first_existing(
        [
            "../input/rsna-2022-cervical-spine-fracture-detection/train.csv",
            "/kaggle/input/rsna-2022-cervical-spine-fracture-detection/train.csv",
            "/kaggle/data/rsna-2022-cervical-spine-fracture-detection/train.csv",
            "../kaggle/data/rsna-2022-cervical-spine-fracture-detection/train.csv",
        ]
    )
    if train_csv_path is None or (not os.path.exists(train_csv_path)):
        return None
    return pd.read_csv(train_csv_path)


def _make_train_slices_df(train_images_dir):
    train_slices = glob.glob(f"{train_images_dir}/*/*.dcm")
    if len(train_slices) == 0:
        return None
    pairs = []
    pat = re.compile(re.escape(train_images_dir) + r"/(.*?)/(\d+)\.dcm$")
    for s in train_slices:
        m = pat.search(s)
        if m:
            pairs.append((m.group(1), int(m.group(2))))
    df = pd.DataFrame(pairs, columns=["StudyInstanceUID", "Slice"]).astype(
        {"Slice": int}
    )
    df["Start"] = df.groupby("StudyInstanceUID")["Slice"].transform("min")
    df = df.sort_values(["StudyInstanceUID", "Slice"]).reset_index(drop=True)
    return df


def _logsumexp_pool_np(p_vec, tau, clip_value=1e-3):
    p = np.clip(np.asarray(p_vec, dtype=np.float64), clip_value, 1.0 - clip_value)
    z = np.log(p / (1.0 - p))
    m = np.max(z)
    pooled = m + (1.0 / float(tau)) * np.log(np.mean(np.exp(float(tau) * (z - m))))
    return float(1.0 / (1.0 + np.exp(-pooled)))


def _apply_logit_scale(p, s):
    p = np.asarray(p, dtype=np.float64)
    z = _logit(p)
    return _sigmoid(z * float(s))


def _fit_scale_and_bias_from_priors_on_train(train_cal, df_train):
    cols_v = [f"C{i}" for i in range(1, 8)]
    cols_all = ["patient_overall"] + cols_v

    pri = train_cal["priors"]
    shrink_by = train_cal["shrink_by_label"]
    any_a = float(train_cal["any_a"])
    any_b = float(train_cal["any_b"])

    n = len(df_train)
    p_mat = np.zeros((n, 7), dtype=np.float64)
    for j, c in enumerate(cols_v):
        p0 = np.full(n, float(pri[c]), dtype=np.float64)
        p_mat[:, j] = _sigmoid(_logit(p0) * float(shrink_by[c]))

    m = p_mat.max(axis=1)
    p_any = np.maximum(_sigmoid(any_a * (m - any_b)), m)

    p0 = np.concatenate([p_any.reshape(-1, 1), p_mat], axis=1)
    y0 = df_train[cols_all].astype(int).values

    s_grid = [0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.4]
    eps = 1e-6

    def loss_for(p):
        p = np.clip(p, eps, 1 - eps)
        loss_any = np.mean(
            -7.0 * (y0[:, 0] * np.log(p[:, 0]) + (1 - y0[:, 0]) * np.log(1 - p[:, 0]))
        )
        loss_v = 0.0
        for j in range(1, 8):
            loss_v += np.mean(
                -(y0[:, j] * np.log(p[:, j]) + (1 - y0[:, j]) * np.log(1 - p[:, j]))
            )
        return float((loss_any + loss_v) / 8.0)

    best_s, best_loss = 1.0, 1e18
    for s in s_grid:
        p_s = _apply_logit_scale(p0, s)
        l = loss_for(p_s)
        if l < best_loss:
            best_loss, best_s = float(l), float(s)

    p_s = np.clip(_apply_logit_scale(p0, best_s), eps, 1 - eps)
    z = _logit(p_s)

    bias_grid = np.array(
        [-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8], dtype=np.float64
    )

    cur_z = z.copy()
    cur_loss = loss_for(_sigmoid(cur_z))
    best_bias = np.zeros((8,), dtype=np.float64)

    for j, name in enumerate(cols_all):
        best_l = cur_loss
        best_b = 0.0
        for b in bias_grid:
            z_try = cur_z.copy()
            z_try[:, j] = cur_z[:, j] + float(b)
            l = loss_for(_sigmoid(z_try))
            if l < best_l:
                best_l = float(l)
                best_b = float(b)
        best_bias[j] = best_b
        cur_z[:, j] = cur_z[:, j] + best_b
        cur_loss = best_l

    bias_dict = {c: float(best_bias[i]) for i, c in enumerate(cols_all)}
    print(
        f"Prior-based calibration chosen: logit_scale={best_s} (train proxy loss={best_loss:.6f})"
    )
    print(
        f"Prior-based label logit biases (after scale; proxy loss={cur_loss:.6f}): {bias_dict}"
    )
    return float(best_s), bias_dict


_TAU_C_POOL = 6.0

df_train_for_cal = _load_train_df()
if _TRAIN_CAL is not None and df_train_for_cal is not None:
    _LOGIT_SCALE, _LABEL_LOGIT_BIAS = _fit_scale_and_bias_from_priors_on_train(
        _TRAIN_CAL, df_train_for_cal
    )
else:
    _LOGIT_SCALE = 1.0
    _LABEL_LOGIT_BIAS = {
        c: 0.0 for c in (["patient_overall"] + [f"C{i}" for i in range(1, 8)])
    }

print("TAU_C_POOL:", _TAU_C_POOL, "LOGIT_SCALE:", _LOGIT_SCALE)
print("LABEL_LOGIT_BIAS:", _LABEL_LOGIT_BIAS)




## === cell 10
def predict():
    """
    Original core inference preserved when models are available.
    If model packages/checkpoints are unavailable, fall back to a calibrated baseline learned from train.csv.
    """
    n = len(ds)
    clip_value = 1e-3

    if seg_model is None or effdet_model is None:
        if _TRAIN_CAL is None:
            base = np.full((n, 8), 0.02, dtype=np.float32)
            base[:, 0] = np.max(base[:, 1:], axis=1)
        else:
            pri = _TRAIN_CAL["priors"]
            shrink_by = _TRAIN_CAL["shrink_by_label"]
            base = np.zeros((n, 8), dtype=np.float32)
            for i in range(1, 8):
                p0 = np.full(n, float(pri[f"C{i}"]), dtype=np.float64)
                s = float(shrink_by[f"C{i}"])
                base[:, i] = _sigmoid(_logit(p0) * s).astype(np.float32)

            m = np.max(base[:, 1:], axis=1).astype(np.float64)
            a = float(_TRAIN_CAL["any_a"])
            b = float(_TRAIN_CAL["any_b"])
            base[:, 0] = _sigmoid(a * (m - b)).astype(np.float32)

        base = np.clip(base, clip_value, 1 - clip_value)
        return base

    preds = []
    with torch.no_grad():
        x0, _, _ = ds[0]
        x1, _, _ = ds[min(1, n - 1)]
        prev2 = torch.stack((x0.to(device), x1.to(device)), dim=0)

        model_dtype = next(seg_model.parameters()).dtype

        for x, pixel_spacings, is_starts in dl:
            x = x.to(device=device, dtype=model_dtype)
            pixel_spacings = pixel_spacings.to(device)

            x = torch.cat((prev2.to(dtype=model_dtype), x), dim=0)

            r = x[:-2, :, :, :]
            g = x[1:-1, :, :, :]
            b = x[2:, :, :, :]

            start_indices = torch.argwhere(is_starts.to(device)).view(-1)
            if start_indices.numel() > 0:
                r[start_indices, :, :, :] = b[start_indices, :, :, :]
                g[start_indices, :, :, :] = b[start_indices, :, :, :]

            prev2 = b[-2:, :, :, :].detach()

            x3 = torch.cat((r, g, b), dim=1)

            batch_probs = x3.new_zeros((x3.shape[0], 8)) + 1e-3

            seg_result = predict_seg(x3, seg_model)  # N x 1 x 256 x 256

            active_indices = seg_result.sum(dim=[1, 2, 3]).nonzero().reshape(-1)
            if active_indices.numel() == 0:
                preds.append(batch_probs)
                continue

            x_act = x3[active_indices, :, :, :]
            seg_act = seg_result[active_indices, :, :, :]
            px_act = pixel_spacings[active_indices]

            axial_boundary = get_axial_boundary_from_seg(
                seg_act, px_act, seg_img_size=256
            )

            croped_x = crop_resize_images(x_act, axial_boundary, 768)
            bboxes, scores = predict_det(croped_x, effdet_model)

            bboxes = get_original_bbox(bboxes, axial_boundary, 768)
            class_list = get_bbox_class_list(seg_act[:, 0, :, :], bboxes / 2)
            probs = get_class_score(scores, class_list)

            batch_probs[active_indices, :] = probs
            preds.append(batch_probs)

    out = torch.cat(preds, dim=0).float().cpu().numpy()
    out = np.clip(out, clip_value, 1 - clip_value)
    return out


predictions = predict()
print("predictions:", predictions.shape, predictions.min(), predictions.max())



## === cell 11
df_effnet_pred = pd.DataFrame(
    data=predictions, columns=["patient_overall"] + [f"C{i}" for i in range(1, 8)]
)

df_test_pred = pd.concat(
    [df_test_slices[["StudyInstanceUID", "Slice"]], df_effnet_pred], axis=1
).sort_values(["StudyInstanceUID", "Slice"])

clip_value = 1e-3

if seg_model is None or effdet_model is None:
    studies = (
        df_test_slices[["StudyInstanceUID"]].drop_duplicates().reset_index(drop=True)
    )
    if _TRAIN_CAL is None:
        df_patient_pred = pd.DataFrame(
            {
                "StudyInstanceUID": studies["StudyInstanceUID"].values,
                "patient_overall": np.full(len(studies), 0.02, dtype=np.float32),
                **{
                    f"C{i}": np.full(len(studies), 0.02, dtype=np.float32)
                    for i in range(1, 8)
                },
            }
        ).set_index("StudyInstanceUID")
        df_patient_pred["patient_overall"] = df_patient_pred[
            [f"C{i}" for i in range(1, 8)]
        ].max(axis=1)
    else:
        pri = _TRAIN_CAL["priors"]
        shrink_by = _TRAIN_CAL["shrink_by_label"]
        a = float(_TRAIN_CAL["any_a"])
        b = float(_TRAIN_CAL["any_b"])

        df_patient_pred = pd.DataFrame(
            {"StudyInstanceUID": studies["StudyInstanceUID"].values}
        )
        for i in range(1, 8):
            p0 = np.full(len(studies), float(pri[f"C{i}"]), dtype=np.float64)
            s = float(shrink_by[f"C{i}"])
            df_patient_pred[f"C{i}"] = _sigmoid(_logit(p0) * s).astype(np.float32)

        m = (
            df_patient_pred[[f"C{i}" for i in range(1, 8)]]
            .max(axis=1)
            .values.astype(np.float64)
        )
        df_patient_pred["patient_overall"] = _sigmoid(a * (m - b)).astype(np.float32)

        df_patient_pred = df_patient_pred.set_index("StudyInstanceUID")
else:

    def _logsumexp_pool_series(p: pd.Series, tau: float = 6.0) -> float:
        p_np = np.clip(p.to_numpy(dtype=np.float64), clip_value, 1.0 - clip_value)
        z = np.log(p_np / (1.0 - p_np))  # logits
        m = np.max(z)
        pooled = m + (1.0 / float(tau)) * np.log(np.mean(np.exp(float(tau) * (z - m))))
        out = 1.0 / (1.0 + np.exp(-pooled))
        return float(out)

    gb = df_test_pred.groupby("StudyInstanceUID", sort=False)
    df_patient_pred = pd.DataFrame(index=gb.size().index)

    for i in range(1, 8):
        c = f"C{i}"
        df_patient_pred[c] = gb[c].agg(
            lambda s: _logsumexp_pool_series(s, tau=float(_TAU_C_POOL))
        )

    if _TRAIN_CAL is not None:
        a = float(_TRAIN_CAL["any_a"])
        b = float(_TRAIN_CAL["any_b"])
        m = (
            df_patient_pred[[f"C{i}" for i in range(1, 8)]]
            .max(axis=1)
            .astype(np.float64)
            .values
        )
        mapped_any = _sigmoid(a * (m - b)).astype(np.float64)
        df_patient_pred["patient_overall"] = np.maximum(mapped_any, m).astype(
            np.float64
        )
    else:
        df_patient_pred["patient_overall"] = (
            df_patient_pred[[f"C{i}" for i in range(1, 8)]]
            .max(axis=1)
            .astype(np.float64)
        )

cols_all = ["patient_overall"] + [f"C{i}" for i in range(1, 8)]
p_scaled = _apply_logit_scale(
    df_patient_pred[cols_all].values.astype(np.float64), float(_LOGIT_SCALE)
)
df_patient_pred[cols_all] = p_scaled

if isinstance(_LABEL_LOGIT_BIAS, dict) and len(_LABEL_LOGIT_BIAS) > 0:
    z = _logit(df_patient_pred[cols_all].values.astype(np.float64))
    bias_vec = np.array(
        [float(_LABEL_LOGIT_BIAS.get(c, 0.0)) for c in cols_all], dtype=np.float64
    )
    df_patient_pred[cols_all] = _sigmoid(z + bias_vec)

df_patient_pred[[f"C{i}" for i in range(1, 8)]] = df_patient_pred[
    [f"C{i}" for i in range(1, 8)]
].clip(lower=clip_value, upper=1 - clip_value)

df_patient_pred["patient_overall"] = df_patient_pred["patient_overall"].clip(
    lower=clip_value, upper=1 - clip_value
)

m2 = df_patient_pred[[f"C{i}" for i in range(1, 8)]].max(axis=1).astype(np.float64)
df_patient_pred["patient_overall"] = np.maximum(
    df_patient_pred["patient_overall"].astype(np.float64), m2
).clip(lower=clip_value, upper=1 - clip_value)

df_patient_pred = df_patient_pred[["patient_overall"] + [f"C{i}" for i in range(1, 8)]]
print(df_patient_pred.head())



## === cell 12
test_csv_path = _resolve_first_existing(
    [
        "../input/rsna-2022-cervical-spine-fracture-detection/test.csv",
        "/kaggle/input/rsna-2022-cervical-spine-fracture-detection/test.csv",
        "/kaggle/data/rsna-2022-cervical-spine-fracture-detection/test.csv",
        "../kaggle/data/rsna-2022-cervical-spine-fracture-detection/test.csv",
    ]
)

df_test = pd.read_csv(test_csv_path)

df_sub = (
    df_test.set_index("StudyInstanceUID")
    .join(df_patient_pred, how="left")
    .reset_index()
)

fill_cols = ["patient_overall"] + [f"C{i}" for i in range(1, 8)]
if _TRAIN_CAL is None:
    df_sub[fill_cols] = df_sub[fill_cols].fillna(0.02)
else:
    pri = _TRAIN_CAL["priors"]
    shrink_by = _TRAIN_CAL["shrink_by_label"]
    a = float(_TRAIN_CAL["any_a"])
    b = float(_TRAIN_CAL["any_b"])

    for i in range(1, 8):
        c = f"C{i}"
        p0 = float(pri[c])
        s = float(shrink_by[c])
        df_sub[c] = df_sub[c].fillna(float(_sigmoid(_logit(p0) * s)))

    m = df_sub[[f"C{i}" for i in range(1, 8)]].max(axis=1).astype(np.float64).values
    p_any = _sigmoid(a * (m - b)).astype(np.float32)
    mask_any = df_sub["patient_overall"].isna()
    df_sub.loc[mask_any, "patient_overall"] = p_any[mask_any.to_numpy()]

df_sub["fractured"] = df_sub.apply(lambda r: float(r[r["prediction_type"]]), axis=1)
df_sub["fractured"] = df_sub["fractured"].clip(1e-3, 1 - 1e-3)

submission = df_sub[["row_id", "fractured"]]
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with rows:", len(submission))
print("submission.csv exists:", os.path.exists("submission.csv"))
