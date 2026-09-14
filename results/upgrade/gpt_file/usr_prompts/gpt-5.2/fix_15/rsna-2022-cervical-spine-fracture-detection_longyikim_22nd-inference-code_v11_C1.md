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

0.3711719783783422

# 6. Current score

0.56032

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5639) has done: 'I fix the environment/import issues by removing the failing pip-install cell and instead using only packages available in the Kaggle image (with safe fallbacks when optional DICOM JPEG support packages are missing). I also fix the execution-order errors by ensuring all needed imports (glob/tqdm/nn/DataLoader, etc.) exist before use, and by making the dataset/dataloader robust (drop `None` samples via a custom `collate_fn`, use single-worker to avoid DICOM decoder issues). Because the original external model packages/checkpoints aren’t available in your provided paths, I keep the overall pipeline structure but replace the unavailable segmentation/detection inference with a stable, score-reasonable baseline: per-study probabilities derived from training-set label priors (and consistent patient_overall definition), which yields a valid submission CSV end-to-end. This is score-oriented versus a constant-0.01 guess while remaining minimal and stable under the constraints of the available environment.'
- What this solution (achieved 0.55997) has done: 'Your current submission uses global label priors only, which is leaving a lot of score on the table; the smallest legitimate improvement (without changing the modeling approach to image inference) is to make those priors better calibrated per label. I keep the “priors-only” core logic but replace raw means with smoothed (Laplace/Beta) estimates and then apply a tiny label-wise temperature-like shrink toward 0.5 to reduce overconfidence, which typically improves log loss. I also ensure `patient_overall` is logically consistent with vertebra probabilities by computing it from the (smoothed) per-vertebra priors rather than taking a max. All paths and submission formatting stay identical.'
- What this solution (achieved 0.56742) has done: 'We keep your “priors-only” baseline intact but make the priors slightly more log-loss-friendly by (1) using label-specific Beta smoothing (stronger smoothing for rarer vertebra labels) and (2) adding a tiny per-label blend toward 0.5 after shrink to avoid extreme probabilities that are heavily penalized by weighted log loss. We also enforce logical consistency by recomputing `patient_overall` from the final per-vertebra probabilities (not from pre-shrink values), which typically improves the heavily weighted overall row. These are minimal, score-relevant calibration changes that preserve your semantics and still produce the same valid `submission.csv`. No image/model inference is introduced and runtime stays essentially the same.'
- What this solution (achieved 0.57072) has done: 'Your current gap to target is about +0.196 (0.567 vs 0.371; lower is better), so we should improve log loss while keeping the “priors-only” core logic unchanged. The smallest score-relevant lever here is calibration: weighted log loss strongly punishes overconfident wrong probabilities, so we (1) estimate label priors with slightly stronger, label-frequency-aware Beta smoothing, and (2) replace the two-step shrink+blend with one conservative temperature-on-logit plus a tiny epsilon clip to avoid extremes. We also compute `patient_overall` consistently from the final per-vertebra probabilities (using independence OR), since it’s heavily weighted and this consistency usually improves loss. Submission formatting/paths remain identical and runtime stays essentially the same.'
- What this solution (achieved 0.5757) has done: 'Your current priors-only baseline is mainly limited by calibration and by the fact that `patient_overall` is much more heavily weighted, so small improvements there can reduce the weighted log loss without changing the core approach. I keep the exact “priors-only” logic, but (1) tune the Beta-smoothing strength to be slightly stronger for rare labels (reducing overconfidence penalties), (2) replace the single fixed temperature with a very small label-wise adjustment (still just logit-temperature) to better match weighted log loss, and (3) compute `patient_overall` strictly from the final calibrated vertebra probabilities (independence-OR) and then apply only a tiny extra shrink for stability. These are minimal, score-relevant calibration tweaks and keep runtime and output format unchanged. The submission writing and row alignment remain identical.'
- What this solution (achieved 0.68207) has done: 'Your current score (0.5757, lower-is-better) is still far above the target (0.3712), so we should legitimately reduce weighted log loss while keeping the same “priors-only” core approach. The smallest high-impact change under these constraints is to compute label priors in a way that directly accounts for the competition’s weighted log loss, by using effective sample-size reweighting (pos/neg weights) when forming Beta-smoothed priors. We then keep your existing logit-temperature calibration and the logically consistent `patient_overall` independence-OR computation, but apply them on top of these weight-aware priors. This stays extremely fast, preserves your end-to-end structure and submission formatting, and should move the score downward toward the target.'
- What this solution (achieved 0.59289) has done: 'Your current score (0.68207, lower-is-better) is much worse than the target (0.37117), so we should reduce log loss while keeping the exact priors-only core approach. The biggest issue in the current code is that it uses *weight-distorted* “prevalence” (mixing pos/neg weights into the prior), which can badly miscalibrate probabilities under log loss; instead, we compute standard Beta-smoothed label prevalence and then apply a small, safe weighted-logloss-aware calibration step (prior-shift on logit) per label. We also make `patient_overall` consistent by computing it from the final calibrated C1–C7 probabilities (independence-OR), which helps the heavily weighted overall row without changing the approach. All file paths, submission formatting, and the priors-only semantics remain the same; runtime stays negligible.'
- What this solution (achieved 0.58296) has done: 'Your current priors-only baseline is still far above the target (0.5929 vs 0.3712, lower-is-better), so the smallest safe improvement is to make the constant probabilities better calibrated for weighted log loss without changing the overall approach. I (1) tune the logit-space prior shift strength `alpha` downward (your current 0.25 likely over-pushes probabilities up, hurting log loss on many negatives) and (2) apply a very small label-wise blend toward 0.5 after calibration to reduce overconfidence penalties. I keep the same Beta-smoothed prevalence computation, the same independence-OR definition for `patient_overall`, and the same submission formatting/paths. These changes are minimal, fast, and directly aimed at reducing weighted log loss.'
- What this solution (achieved 0.58034) has done: 'We keep your “priors-only” baseline exactly intact and only adjust the calibration knobs that directly affect weighted log loss. Your current prior-shift still pushes probabilities upward (hurting the many negative rows), so I reduce `alpha` further and slightly increase the post-calibration blend toward 0.5 to avoid overconfidence penalties. Because `patient_overall` is heavily weighted, we also apply a tiny extra blend-to-0.5 on the final independence-OR `patient_overall` probability (without changing how it’s computed). All paths, output formatting, and the end-to-end submission writing remain unchanged.'
- What this solution (achieved 0.57745) has done: 'We keep your priors-only approach intact and only adjust the few calibration knobs that directly affect weighted log loss. Since your score is worse than target (0.58034 vs 0.37117; lower is better), the safest minimal lever is to reduce overconfidence and slightly lower probabilities (most labels are negative), while keeping `patient_overall` logically consistent via independence-OR. Concretely, we (1) remove the upward logit prior-shift (set `alpha` to 0) which likely hurts many negatives, and (2) slightly increase the blend-to-0.5 (especially for `patient_overall`) to reduce the heavy penalty from confident mistakes. No image inference, no loop/architecture changes, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.66293) has done: 'Your current score (0.57745, lower-is-better) is still far above the target (0.37117), so we should legitimately reduce weighted log loss while keeping the exact “priors-only” approach intact. The smallest high-impact, metric-aligned change is to estimate *label-wise constant probabilities by directly minimizing (smoothed) weighted log loss* on the training set, rather than using prevalence + heuristic calibration; for a constant predictor, this has a closed-form solution using the competition’s pos/neg weights. We keep the same patient_overall construction via independence-OR from C1–C7 to preserve logical consistency, and we only add a light Beta-style pseudo-count smoothing to avoid extreme constants on small data. Submission formatting/paths remain identical, runtime stays tiny, and the pipeline still writes a valid `submission.csv`.'
- What this solution (achieved 0.5607) has done: 'Your current “weighted-logloss-optimal constant” prior calculation is likely misaligned with how Kaggle’s metric is computed per-row: the weights multiply the loss, not the labels, so the best constant probability for a label should be the (smoothed) *unweighted* prevalence, not the weight-distorted proportion. I revert the per-label constants to Beta-smoothed prevalence (a minimal change that should lower log loss versus the current 0.6629), then keep `patient_overall` logically consistent by computing it as an independence-OR from the final C1–C7 probabilities (since that row is heavily weighted). I also keep a small probability clip to avoid extreme log penalties, and preserve all paths and the priors-only core approach while still writing a valid `submission.csv`.'
- What this solution (achieved 0.63808) has done: 'Your current approach is a priors-only constant predictor; to move the score down toward the target without changing core semantics, the only safe lever is better calibration of those constants under (weighted) log loss. We keep the same pipeline and submission formatting, but replace the fixed Beta(0.5,0.5) smoothing with a tiny per-label grid search that directly minimizes the *training* weighted log loss for each label’s constant probability (this is still “priors-only”, just choosing better constants). We keep `patient_overall` logically consistent by computing it as independence-OR from the final C1–C7 probabilities, then optionally apply the same constant-optimization step to `patient_overall` itself with a very small blend toward the OR value for stability. This remains extremely fast (202 rows) and should reduce the public score versus the current smoothed-prevalence constants.'
- What this solution (achieved 0.56032) has done: 'We keep your priors-only constant-predictor pipeline and submission formatting unchanged, but fix the main score regression source: you’re currently choosing constants by minimizing a *misimplemented* “weighted log loss” (you weight positives/negatives differently within a label), which doesn’t match the competition’s metric where each label has a single weight applied to both terms. We instead pick each label’s constant probability by minimizing the correct weighted log loss (equivalently: standard log loss, since the per-label weight is just a multiplier) with a tiny Beta smoothing to avoid extremes. We also keep `patient_overall` logically consistent by computing it as independence-OR from C1–C7 (no extra blending), which typically helps the heavily weighted row without changing core semantics. These are minimal calibration-only changes and should move your score downward (better) toward the target.'

# 9. Code solution

## === cell 0
import os
import re
import sys
import math
import glob
import random
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

try:
    import pydicom as dicom
except Exception as e:
    dicom = None
    print("WARNING: pydicom not available; image loading will be disabled.", repr(e))

try:
    import pylibjpeg  # optional
except Exception:
    pylibjpeg = None



## === cell 1
IMAGES_DIR = "../input/rsna-2022-cervical-spine-fracture-detection/test_images"
TRAIN_IMAGES_PATH = "../input/rsna-2022-cervical-spine-fracture-detection/train_images"
TEST_IMAGES_PATH = "../input/rsna-2022-cervical-spine-fracture-detection/test_images"

TRAIN_CSV_PATH = "../input/rsna-2022-cervical-spine-fracture-detection/train.csv"
TEST_CSV_PATH = "../input/rsna-2022-cervical-spine-fracture-detection/test.csv"
SAMPLE_SUB_PATH = (
    "../input/rsna-2022-cervical-spine-fracture-detection/sample_submission.csv"
)

assert os.path.exists(TRAIN_CSV_PATH), f"Missing {TRAIN_CSV_PATH}"
assert os.path.exists(TEST_CSV_PATH), f"Missing {TEST_CSV_PATH}"
assert os.path.exists(SAMPLE_SUB_PATH), f"Missing {SAMPLE_SUB_PATH}"



## === cell 2
segmentation_checkpoint = (
    "../input/effdet-models/axial_segmentation_effseg_132508-epoch-100.pth"
)
axial_det_checkpoint = (
    "../input/effdet-models/axial_detection_effdet_134352-epoch-52.pth"
)



## === cell 3
df_test = pd.read_csv(TEST_CSV_PATH)
df_test.head()




## === cell 4
def rescale_img_to_hu(dcm_ds):
    return dcm_ds.pixel_array * float(getattr(dcm_ds, "RescaleSlope", 1.0)) + float(
        getattr(dcm_ds, "RescaleIntercept", 0.0)
    )


def normalize_hu(data):
    data = np.clip(data, a_min=-2242, a_max=2242) / 4484 + 0.5
    return data


def load_dicom(path):
    if dicom is None:
        raise RuntimeError("pydicom is not available in this environment.")
    ds = dicom.dcmread(path)
    img = rescale_img_to_hu(ds)
    pixel_spacing = float(ds.PixelSpacing[0]) if hasattr(ds, "PixelSpacing") else 1.0
    return img, pixel_spacing




## === cell 5
class DcmDataSet(Dataset):
    def __init__(self, df, path, transforms=None):
        super().__init__()
        self.df = df
        self.path = path
        self.transforms = transforms
        self.len = len(self.df)

    def __getitem__(self, i):
        try:
            s = self.df.iloc[i]
            prev_s = self.df.iloc[i - 1] if i > 0 else s
            prev_s = s if prev_s.StudyInstanceUID != s.StudyInstanceUID else prev_s

            next_s = self.df.iloc[i + 1] if i < (self.len - 1) else s
            next_s = s if next_s.StudyInstanceUID != s.StudyInstanceUID else next_s

            rpath = os.path.join(
                self.path, s.StudyInstanceUID, f"{int(prev_s.Slice)}.dcm"
            )
            gpath = os.path.join(self.path, s.StudyInstanceUID, f"{int(s.Slice)}.dcm")
            bpath = os.path.join(
                self.path, s.StudyInstanceUID, f"{int(next_s.Slice)}.dcm"
            )

            g, pixel_spacing = load_dicom(gpath)
            r, _ = load_dicom(rpath)
            b, _ = load_dicom(bpath)

            img = np.stack((r, g, b))
            img = normalize_hu(img)
            if self.transforms is not None:
                img = self.transforms(img)
        except Exception:
            return None, None

        return img, pixel_spacing

    def __len__(self):
        return self.len


class DataTransform(nn.Module):
    def __init__(self, image_size=512):
        super().__init__()
        self.image_size = image_size

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        return x




## === cell 6
def safe_collate(batch):
    batch = [b for b in batch if b[0] is not None]
    if len(batch) == 0:
        return torch.empty((0, 3, 512, 512)), torch.empty((0,))
    xs, ps = zip(*batch)
    return torch.stack(xs, 0), torch.as_tensor(ps, dtype=torch.float32)


dl = None  # not used in baseline




## === cell 7
def get_axial_segmentation_model(checkpoint):
    raise ModuleNotFoundError(
        "efficientunet/weights not available in this environment; baseline uses priors only."
    )


seg_model = None




## === cell 8
def get_axial_detection_model(checkpoint, image_size=512):
    raise ModuleNotFoundError(
        "effdet/weights not available in this environment; baseline uses priors only."
    )


det_model = None




## === cell 9
def get_axial_boundary_from_segmentation(
    seg, pixel_spacing, throw=100, tol=0.2, max_mm=100
):
    image_size = seg.shape[0]
    min_size = min(image_size, max_mm / pixel_spacing)

    rows, columns = seg.nonzero(as_tuple=True)
    rows, _ = torch.sort(rows)
    columns, _ = torch.sort(columns)

    throw = min(len(rows) // 2, throw)

    if len(rows) == 0:
        return torch.tensor([0, 0, image_size, image_size], device=seg.device)

    xmin, xmax = columns[throw], columns[-throw]
    ymin, ymax = rows[throw], rows[-throw]

    w = (xmax - xmin) * (1 + tol)
    h = (ymax - ymin) * (1 + tol)
    new_size = max(w, h, torch.as_tensor(min_size, device=seg.device))
    new_size = min(image_size, float(new_size))

    xcenter, ycenter = (xmax + xmin) / 2, (ymax + ymin) / 2

    xmin = torch.minimum(
        torch.tensor(image_size - new_size, device=seg.device), xcenter - new_size / 2
    ).clamp(min=0)
    ymin = torch.minimum(
        torch.tensor(image_size - new_size, device=seg.device), ycenter - new_size / 2
    ).clamp(min=0)

    return torch.stack([xmin, ymin, xmin + new_size, ymin + new_size])




## === cell 10
def predict_seg(x, model, img_size=256):
    raise RuntimeError("Segmentation model not available in this baseline.")




## === cell 11
def get_axial_boundary(segs, pixel_spacings, seg_img_size=256):
    raise RuntimeError("Segmentation model not available in this baseline.")




## === cell 12
def predict_det(x, model):
    raise RuntimeError("Detection model not available in this baseline.")


def crop_resize_images(imgs_tensor, boundary_list, img_size=512):
    raise RuntimeError("Detection model not available in this baseline.")


def get_original_bbox(bbox, boundary):
    raise RuntimeError("Detection model not available in this baseline.")




## === cell 13
def get_bbox_class(seg, bbox):
    raise RuntimeError("Detection model not available in this baseline.")




## === cell 14
def get_bbox_class_list(seg_list, seg_bboxes):
    raise RuntimeError("Detection model not available in this baseline.")




## === cell 15
def get_class_score(scores, class_list, eps=1e-2):
    raise RuntimeError("Detection model not available in this baseline.")


def check_detection_result(det_result, img_size=512.0, threshold=0.2):
    raise RuntimeError("Detection model not available in this baseline.")




## === cell 16
def cal_loss(prob, label):
    pos_weight = np.array([14, 2, 2, 2, 2, 2, 2, 2])
    neg_weight = np.array([7, 1, 1, 1, 1, 1, 1, 1])
    prob = np.clip(prob, 1e-6, 1 - 1e-6)
    score = pos_weight * label * np.log(prob) + neg_weight * (1 - label) * np.log(
        1 - prob
    )
    weight_total = pos_weight * label + neg_weight * (1 - label)
    return -score.sum(axis=1) / weight_total.sum(axis=1)




## === cell 17
def _logloss_constant(y: np.ndarray, p: float) -> float:
    """Mean (unweighted) binary log loss for a single label with constant probability p."""
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    y = y.astype(np.float64)
    loss = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return float(np.mean(loss))


def _choose_constant_p_by_grid_correct_metric(y: np.ndarray) -> float:
    """
    Score-moving fix (still priors-only): choose constant p that minimizes the *correct* per-label
    metric. In this competition, each label's loss is multiplied by a label weight w_j, which does
    not change the argmin over p for a constant predictor; so we minimize standard log loss.

    We keep a tiny Beta smoothing prior (a0_pos/a0_neg) to avoid extreme p on small N.
    """
    y = y.astype(np.int64)
    n = len(y)

    a0_pos, a0_neg = 0.5, 0.5
    p0 = (y.sum() + a0_pos) / (n + a0_pos + a0_neg)
    p0 = float(np.clip(p0, 1e-4, 1 - 1e-4))

    spans = [0.20, 0.10, 0.05, 0.02]
    best_p, best_l = p0, _logloss_constant(y, p0)

    for sp in spans:
        lo = max(1e-4, best_p - sp)
        hi = min(1 - 1e-4, best_p + sp)
        grid = np.linspace(lo, hi, 401)
        ps = np.clip(grid, 1e-6, 1 - 1e-6)
        loss = -(
            y[:, None] * np.log(ps)[None, :]
            + (1 - y)[:, None] * np.log(1 - ps)[None, :]
        )
        mean_loss = loss.mean(axis=0)
        j = int(np.argmin(mean_loss))
        best_p = float(grid[j])
        best_l = float(mean_loss[j])

    return float(np.clip(best_p, 1e-4, 1 - 1e-4))


def predict_priors(train_csv_path: str):
    """
    Priors-only baseline (core logic preserved): output constant probabilities per label.

    Score-moving change:
    - Fix constant selection to match the competition metric: since label weights multiply the loss,
      they do not affect the optimal constant p. So we minimize standard log loss per label (with
      tiny smoothing) instead of the prior incorrect pos/neg reweighting.
    - Keep patient_overall logically consistent: compute it as independence-OR from final C1..C7.
    """
    df_train = pd.read_csv(train_csv_path)
    cols_v = [f"C{i}" for i in range(1, 8)]

    pri = {}
    for c in cols_v:
        y = df_train[c].values
        pri[c] = _choose_constant_p_by_grid_correct_metric(y)

    p_any_or = 1.0 - float(np.prod([1.0 - pri[c] for c in cols_v]))
    pri["patient_overall"] = float(np.clip(p_any_or, 1e-4, 1 - 1e-4))

    return pri


priors = predict_priors(TRAIN_CSV_PATH)
priors



## === cell 18
study_ids = df_test["StudyInstanceUID"].unique()
df_patient_pred = pd.DataFrame({"StudyInstanceUID": study_ids})
for k in ["patient_overall"] + [f"C{i}" for i in range(1, 8)]:
    df_patient_pred[k] = float(priors[k])

df_patient_pred = df_patient_pred.set_index("StudyInstanceUID")
df_patient_pred.head()



## === cell 19
pass



## === cell 20
if df_test.iloc[0].row_id == "1.2.826.0.1.3680043.10197_C1":
    df_test = pd.DataFrame(
        {
            "row_id": [
                "1.2.826.0.1.3680043.22327_C1",
                "1.2.826.0.1.3680043.25399_C1",
                "1.2.826.0.1.3680043.5876_patient_overall",
            ],
            "StudyInstanceUID": [
                "1.2.826.0.1.3680043.22327",
                "1.2.826.0.1.3680043.25399",
                "1.2.826.0.1.3680043.5876",
            ],
            "prediction_type": ["C1", "C1", "patient_overall"],
        }
    )

df_test.head()



## === cell 21
df_sub = df_test.copy()
df_sub = (
    df_sub.set_index("StudyInstanceUID").join(df_patient_pred, how="left").reset_index()
)

for k in ["patient_overall"] + [f"C{i}" for i in range(1, 8)]:
    df_sub[k] = df_sub[k].fillna(float(priors[k]))

df_sub["fractured"] = df_sub.apply(lambda r: float(r[r["prediction_type"]]), axis=1)
df_sub[["row_id", "fractured"]].head()



## === cell 22
sub_path = "submission.csv"
df_sub[["row_id", "fractured"]].to_csv(sub_path, index=False)

df_sample = pd.read_csv(SAMPLE_SUB_PATH)
assert list(df_sample.columns) == ["row_id", "fractured"]
assert df_sub.shape[0] == df_sample.shape[0], (df_sub.shape, df_sample.shape)
print("Wrote:", sub_path, "rows:", df_sub.shape[0])
print(df_sub.head())
