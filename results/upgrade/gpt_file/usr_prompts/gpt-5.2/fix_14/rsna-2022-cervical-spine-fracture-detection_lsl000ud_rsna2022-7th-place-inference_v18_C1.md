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

cloudpathlib==0.21.1
cuda-pathfinder==1.3.2
geopandas==0.14.4
jmespath==1.0.1
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
path==17.1.1
path.py==12.5.0
pathos==0.3.2
pathspec==0.12.1
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-image==0.25.2
simpleitk==2.5.2
sklearn-pandas==2.2.0
testpath==0.6.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124

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

0.266958098759516

# 6. Current score

0.56248

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5639) has done: 'I fix the immediate import/path failure by removing the unavailable `../input/src2-final/src` dependency and replacing it with lightweight, in-notebook fallbacks that let the pipeline run end-to-end in this Kaggle environment. Because the original 3-stage nnUNet/model files are not provided in the listed data paths, I add safe checks that avoid crashing when those checkpoints/plans are missing, and instead generate a valid baseline submission using train priors (a legitimate, score-reasonable fallback under logloss). I also ensure the submission matches `test.csv` ordering and contains every `row_id` exactly once, written to `submission.csv`. These changes are execution-unblocking and should yield a finite logloss (better than random/invalid), moving toward the target where previously no score was possible.'
- What this solution (achieved 0.58292) has done: 'Your current baseline uses raw train prevalences, which is typically miscalibrated for weighted logloss (especially the heavily-weighted `patient_overall`) and can be improved with a minimal, metric-aligned calibration step. I keep your “priors-only” core logic, but (1) calibrate the per-label probabilities using a single temperature scaling on the logit scale learned from the training labels (no new model, no extra data), and (2) enforce a consistent relationship between vertebrae and `patient_overall` by converting vertebra probabilities into an implied any-fracture probability `1 - Π(1-p_Ci)` and blending it with the calibrated `patient_overall`. These changes are small, deterministic, and should reduce logloss substantially from 0.5639 toward your 0.2669 target while still producing the same valid submission format. Paths/I/O stay the same and the script remains fast.'
- What this solution (achieved 0.56035) has done: 'Your current score (0.58292, lower-is-better) is still far from the target (0.26696), so we should make a small but meaningful metric-aligned change without altering the “priors-only + temperature + any-from-levels blend” core logic. The biggest remaining mismatch is that the competition uses *weighted* logloss with `patient_overall` weighted higher, but your temperature search uses a balanced weighting that does not reflect the metric. I change the temperature fitting to optimize a per-label weighted logloss consistent with the RSNA weights (with `patient_overall` heavier), which should reduce the loss and move you toward the target while keeping the same prediction structure and submission format. Everything else (paths, priors, prediction mapping, CSV writing) stays the same.'
- What this solution (achieved 0.56268) has done: 'Your current approach is already “priors + per-label temperature + consistent any-from-levels blend”, so the smallest safe way to move the loss toward your (much lower) target is to (1) fit calibration parameters against the *actual weighted logloss* more faithfully and (2) reduce the known over-penalization from using a single constant probability per label by adding a tiny, label-preserving “study-level offset” based on the only available test feature: number of slices in each study folder. This keeps the core logic (no new model/architecture/features beyond a single scalar per study) but introduces legitimate per-study variation and metric-aligned tuning, which should lower logloss versus a flat prior. The script still runs fast, uses the same files/paths, and writes a valid `submission.csv`.'
- What this solution (achieved 0.56186) has done: 'Your current score (0.56268, lower-is-better) is still far above the target (0.26696), so we should make a small, legitimate change that reduces weighted logloss without changing the overall “priors + calibration + any-from-levels + slice-based offset” approach. The biggest low-risk improvement is to fit the two free knobs you already have—`slice_logit_beta` and the `alpha` blend for `patient_overall`—directly on the training set by minimizing the same weighted logloss used in the competition, instead of leaving them as fixed constants. This keeps the model semantics identical (still one constant per label, temperature-scaled, plus a single study-level offset, plus a blend for patient_overall), but should move the loss toward the target by better calibration of the heavily weighted `patient_overall`. I also keep the existing per-label temperature search unchanged, and ensure the submission remains aligned to `test.csv` with every `row_id` exactly once.'
- What this solution (achieved 0.56186) has done: 'I keep your “priors + per-label temperature + slice-based logit offset + any-from-levels blend” core logic, but remove the biggest remaining source of overfitting: fitting `alpha_any_blend` and `slice_logit_beta` on the same training data you evaluate on. Instead, I fit those two knobs with a small K-fold out-of-fold procedure that optimizes the same weighted logloss, then refit per-label temperatures once on the full train set as before and use the CV-fitted global `alpha`/`beta` for test inference. This is a minimal, metric-aligned change that typically lowers public LB logloss versus in-sample tuning while keeping runtime under the limit and producing the same `submission.csv` format.'
- What this solution (achieved 0.56186) has done: 'We keep your current “priors + per-label temperature + slice-based logit offset + any-from-levels blend” structure intact, but fix the most likely reason your extra knobs aren’t helping: the slice-based offset is currently applied to *all 8 labels*, which can harm calibration (especially for the heavily weighted `patient_overall`). I change the offset to apply only to the 7 vertebrae levels (C1–C7), and then recompute/blend `patient_overall` from (a) its calibrated prior and (b) the implied any-from-levels probability—still the same blend idea, just correctly ordered. This is a minimal semantic correction (not a new model), and should reduce weighted logloss versus the current behavior, moving your 0.56186 toward the 0.26696 target while keeping runtime and I/O the same. The submission format, ordering (from `test.csv`), and `submission.csv` output remain unchanged.'
- What this solution (achieved 0.56186) has done: 'I keep your existing “priors + per-label temperature + slice-based logit offset on C1–C7 + any-from-levels blend for patient_overall” logic intact, but make two small metric-aligned fixes that should reduce weighted logloss toward the target. First, I apply the per-label RSNA weights correctly when selecting `alpha_any_blend` and `slice_logit_beta` by weighting the *mean* logloss per label, rather than flattening all rows (which over-weights the vertebrae because there are 7 of them). Second, I evaluate the OOF objective on **fold-trained** priors/temperatures (still the same calibration method, just avoiding leakage across folds), which should modestly improve generalization and thus public LB loss without adding new modeling. The submission format/ordering stays tied to `test.csv`, and it still writes `submission.csv` end-to-end within the time limit.'
- What this solution (achieved 0.56186) has done: 'Your current score is much worse than the target (lower-is-better), so the smallest likely win is to improve calibration while keeping your exact “priors + temperature + slice-logit offset on C1–C7 + any-from-levels blend” structure. I add one minimal extra calibration step that is still within your current semantics: fit a **per-label bias (intercept) on the logit scale** on training data (with RSNA weights) while holding the already-fit temperatures fixed; this corrects systematic over/under-confidence that temperature alone can’t fix. I do this bias fitting in the same OOF loop you already use (to avoid leakage), then refit on full train for final inference. The rest (paths, slice feature, alpha/beta search, mapping to `test.csv`, and `submission.csv` writing) stays the same.'
- What this solution (achieved 0.56186) has done: 'Your current gap to the target is large (0.56186 vs 0.26696, lower-is-better), so we should make a minimal, metric-aligned improvement without changing your overall “priors + per-label temperature + per-label logit bias + slice-logit offset on C1–C7 + any-from-levels blend” structure. The largest remaining issue is that you refit the per-label logit biases on the full training set after doing OOF tuning, which can re-introduce overfitting/miscalibration; instead, we fit **OOF (cross-fitted) biases** and use their average for inference, keeping everything else the same. This is a small calibration-only change (no new features, no model changes) and is expected to reduce weighted logloss by making probabilities less overconfident, especially for the heavily-weighted `patient_overall`. The script still runs fast, uses the same paths, and writes a valid `submission.csv` aligned to `test.csv`.'
- What this solution (achieved 0.56248) has done: 'Your current solution is stuck because the “slice count” feature is extremely noisy (folder DICOM counts vary for non-clinical reasons) and is being fitted/tuned despite adding little real signal; this can hurt calibration under weighted logloss. To move the loss down toward the target with minimal semantic change, I keep the exact same pipeline but (1) make the slice-based offset more robust by using a clipped/centered log-slice feature (same single scalar per study, just bounded), and (2) expand the OOF grid slightly so alpha/beta can find a better weighted-logloss minimum under this more stable feature. Everything else (priors, per-label temperature, per-label logit bias, any-from-levels blend, test mapping, and `submission.csv` writing) stays the same and remains fast.'
- What this solution (achieved 0.56248) has done: 'We keep your exact “priors + per-label temperature + (OOF-averaged) per-label logit bias + slice-logit offset on C1–C7 + any-from-levels blend for patient_overall” pipeline, but fix the main calibration inefficiency: your temperature search is currently redundant because it only rescales a *single constant logit*, so it can’t materially change probabilities once you also fit a free intercept bias. I therefore (1) lock temperatures to 1.0 (no-op) and (2) fit the per-label logit biases using the exact weighted logloss optimum for a constant predictor (still your same bias step, just now it’s the only calibration degree of freedom), and keep the existing OOF alpha/beta tuning unchanged. This is a minimal semantic change (still constant-per-label predictors + same slice feature + same any-blend), but it should reduce loss versus the current slightly-misalibrated “T + bias” combination and move you downward toward the 0.2669 target. Submission formatting, test.csv alignment, paths, and runtime remain within constraints and it still writes `submission.csv`.'
- What this solution (achieved 0.56248) has done: 'Your current pipeline is fundamentally a constant-per-label predictor with a small slice-count offset and an any-from-levels blend; the main reason it can’t approach the target score is that it lacks any real image signal, so we should only make minimal calibration changes that reliably reduce weighted logloss. The smallest likely win is to (1) compute the *optimal constant predictor* for each label directly from the training labels (the weighted-logloss optimum is simply the label prevalence), which means we should remove the redundant prior→bias fitting and just use the empirical means as probabilities, and (2) constrain the slice-based offset to be strictly zero (beta fixed to 0) because it adds noisy, non-medical variation and has not helped your LB. We keep the exact same submission mapping logic and the any-from-levels blend, but refit only the blend weight `alpha_any_blend` with OOF weighted-logloss while holding beta at 0 to avoid overfitting. This should move the score modestly downward from ~0.562 toward your lower target while staying stable and very fast.'

# 9. Code solution

## === cell 0
import os
import time
import math
import gc
import numpy as np
import pandas as pd
import torch


DATA_ROOT_CANDIDATES = [
    "/kaggle/input/rsna-2022-cervical-spine-fracture-detection",
    "/kaggle/data/rsna-2022-cervical-spine-fracture-detection",
    "../input/rsna-2022-cervical-spine-fracture-detection",
]
DATA_ROOT = None
for p in DATA_ROOT_CANDIDATES:
    if os.path.exists(p):
        DATA_ROOT = p
        break
if DATA_ROOT is None:
    DATA_ROOT = "/kaggle/input"

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

print("DATA_ROOT:", DATA_ROOT)
print("Exists train.csv:", os.path.exists(TRAIN_CSV))
print("Exists test.csv:", os.path.exists(TEST_CSV))
print("Exists sample_submission.csv:", os.path.exists(SAMPLE_SUB))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Torch device:", device)




## === cell 1
TARGET_COLS = ["patient_overall", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]
LEVEL_COLS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def compute_train_priors(train_csv: str) -> dict:
    df = pd.read_csv(train_csv)
    priors = {}
    for c in TARGET_COLS:
        if c in df.columns:
            priors[c] = float(df[c].mean())
        else:
            priors[c] = 0.01
    for k in priors:
        priors[k] = float(np.clip(priors[k], 1e-4, 1 - 1e-4))
    return priors


def _rsna_label_weight(col: str) -> float:
    if col == "patient_overall":
        return 7.0
    return 1.0


def fit_temperature_scaling(train_csv: str, base_priors: dict) -> dict:
    """
    Keep temperatures as a no-op.
    Rationale (score-relevant + minimal): with constant-per-label predictions, extra temperature
    degrees of freedom tend to create slight miscalibration; keeping T=1.0 is the most stable.
    """
    _ = train_csv, base_priors
    return {c: 1.0 for c in TARGET_COLS}


def apply_temperature(p: float, T: float) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    z = _logit(p)
    return float(np.clip(_sigmoid(z / float(T)), 1e-6, 1 - 1e-6))


def _list_num_slices(study_dir: str) -> int:
    try:
        if not os.path.isdir(study_dir):
            return -1
        n = 0
        with os.scandir(study_dir) as it:
            for e in it:
                if e.is_file() and e.name.lower().endswith(".dcm"):
                    n += 1
        return int(n)
    except Exception:
        return -1


def _fit_logit_bias_from_data(y: np.ndarray, base_logit: float, w: np.ndarray) -> float:
    """
    Kept for backwards-compatibility with the pipeline, but will not be used after the change below.
    """
    y = y.astype(np.float64)
    w = w.astype(np.float64)
    b = 0.0
    for _ in range(50):
        z = base_logit + b
        p = _sigmoid(z)
        g = np.sum(w * (p - y))
        h = np.sum(w * p * (1.0 - p))
        if h <= 1e-12:
            break
        step = g / h
        b -= step
        if abs(step) < 1e-10:
            break
    return float(b)


def fit_logit_biases(train_csv: str, priors: dict, temps: dict) -> dict:
    """
    Expected to reduce weighted logloss vs current behavior:
    For constant predictors under (weighted) logloss, the optimal constant probability is the
    empirical prevalence, so extra per-label bias fitting is redundant and can introduce tiny
    numerical noise. We set biases to 0 so p stays exactly at the (clipped) train mean.
    """
    _ = train_csv, priors, temps
    return {c: 0.0 for c in TARGET_COLS}


def apply_temp_and_bias(p: float, T: float, b: float) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    z = _logit(p)
    return float(np.clip(_sigmoid(z / float(T) + float(b)), 1e-6, 1 - 1e-6))


class FractureDetector:
    def __init__(
        self,
        priors: dict,
        temps: dict,
        biases: dict | None = None,
        slice_logit_beta: float = 0.0,
        slice_ref: float = 300.0,
        alpha_any_blend: float = 0.70,
        slice_log_nclip: tuple[float, float] = (math.log(80.0), math.log(800.0)),
    ):
        self.priors = priors
        self.temps = temps
        self.biases = biases or {c: 0.0 for c in TARGET_COLS}
        self.results = {}
        self.slice_logit_beta = float(slice_logit_beta)
        self.slice_ref = float(slice_ref)
        self.alpha_any_blend = float(alpha_any_blend)
        self.slice_log_nclip = (float(slice_log_nclip[0]), float(slice_log_nclip[1]))

    def _slice_offset(self, n_slices: int) -> float:
        if n_slices is None or n_slices <= 0:
            return 0.0
        ln = math.log(float(n_slices))
        ln = float(np.clip(ln, self.slice_log_nclip[0], self.slice_log_nclip[1]))
        return self.slice_logit_beta * (ln - math.log(self.slice_ref))

    def predict(self, list_test_files, num_thread=4, output_dir=None):
        _ = num_thread, output_dir
        for p in list_test_files:
            case_id = os.path.basename(p.rstrip("/"))

            out = np.zeros(8, dtype=np.float64)
            out[0] = apply_temp_and_bias(
                self.priors["patient_overall"],
                self.temps.get("patient_overall", 1.0),
                self.biases.get("patient_overall", 0.0),
            )
            for i, c in enumerate(LEVEL_COLS, start=1):
                out[i] = apply_temp_and_bias(
                    self.priors[c],
                    self.temps.get(c, 1.0),
                    self.biases.get(c, 0.0),
                )

            n_slices = _list_num_slices(p)
            dz = self._slice_offset(n_slices)

            if dz != 0.0:
                for k in range(1, 8):
                    z = _logit(float(out[k]))
                    out[k] = float(np.clip(_sigmoid(z + dz), 1e-6, 1 - 1e-6))

            p_any_from_levels = 1.0 - float(
                np.prod(1.0 - np.clip(out[1:], 1e-6, 1 - 1e-6))
            )

            alpha = self.alpha_any_blend
            out[0] = float(
                np.clip(
                    alpha * out[0] + (1.0 - alpha) * p_any_from_levels, 1e-6, 1 - 1e-6
                )
            )

            self.results[case_id] = out.astype(np.float32)


def _build_train_study_slice_counts(
    train_df: pd.DataFrame, train_img_dir: str | None
) -> dict:
    uids = train_df["StudyInstanceUID"].values.tolist()
    if train_img_dir is None or (not os.path.isdir(train_img_dir)):
        return {uid: -1 for uid in uids}
    counts = {}
    for uid in uids:
        counts[uid] = _list_num_slices(os.path.join(train_img_dir, uid))
    return counts


def _predict_train_probs_for_params(
    train_df: pd.DataFrame,
    priors: dict,
    temps: dict,
    biases: dict,
    slice_counts: dict,
    slice_logit_beta: float,
    slice_ref: float,
    alpha_any_blend: float,
    slice_log_nclip: tuple[float, float] = (math.log(80.0), math.log(800.0)),
) -> dict:
    base = np.zeros((len(train_df), 8), dtype=np.float64)
    base[:, 0] = apply_temp_and_bias(
        priors["patient_overall"],
        temps.get("patient_overall", 1.0),
        biases.get("patient_overall", 0.0),
    )
    for i, c in enumerate(LEVEL_COLS, start=1):
        base[:, i] = apply_temp_and_bias(
            priors[c], temps.get(c, 1.0), biases.get(c, 0.0)
        )

    dz = np.zeros((len(train_df),), dtype=np.float64)
    if slice_logit_beta != 0.0:
        for i, uid in enumerate(train_df["StudyInstanceUID"].values):
            n = slice_counts.get(uid, -1)
            if n is not None and n > 0:
                ln = math.log(float(n))
                ln = float(
                    np.clip(ln, float(slice_log_nclip[0]), float(slice_log_nclip[1]))
                )
                dz[i] = float(slice_logit_beta) * (ln - math.log(float(slice_ref)))

    if np.any(dz != 0.0):
        for k in range(1, 8):
            z = _logit(base[:, k])
            base[:, k] = np.clip(_sigmoid(z + dz), 1e-6, 1 - 1e-6)

    p_any_from_levels = 1.0 - np.prod(
        1.0 - np.clip(base[:, 1:], 1e-6, 1 - 1e-6), axis=1
    )
    base[:, 0] = np.clip(
        float(alpha_any_blend) * base[:, 0]
        + (1.0 - float(alpha_any_blend)) * p_any_from_levels,
        1e-6,
        1 - 1e-6,
    )

    out = {"patient_overall": base[:, 0]}
    for i, c in enumerate(LEVEL_COLS, start=1):
        out[c] = base[:, i]
    return out


def _rsna_weighted_logloss_matrix(
    Y: np.ndarray, P: np.ndarray, col_names: list[str]
) -> float:
    P = np.clip(P, 1e-6, 1 - 1e-6)
    losses = []
    weights = []
    for j, c in enumerate(col_names):
        wj = float(_rsna_label_weight(c))
        yj = Y[:, j]
        pj = P[:, j]
        lj = -np.mean(yj * np.log(pj) + (1 - yj) * np.log(1 - pj))
        losses.append(lj)
        weights.append(wj)
    losses = np.array(losses, dtype=np.float64)
    weights = np.array(weights, dtype=np.float64)
    return float(np.sum(weights * losses) / np.sum(weights))


def fit_alpha_and_slice_beta_oof(
    train_csv: str,
    priors: dict,
    temps: dict,
    train_img_dir: str | None,
    slice_ref: float = 300.0,
    n_folds: int = 5,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Expected to reduce weighted logloss vs current behavior:
    The slice-count feature is noisy; we therefore *fix* slice_logit_beta=0 and only tune alpha,
    keeping the same any-from-levels blend logic.
    """
    _ = priors, temps  # preserved signature; not needed for OOF alpha fit

    df = pd.read_csv(train_csv)
    n = len(df)

    slice_counts = _build_train_study_slice_counts(df, train_img_dir)

    alphas = np.array([0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90], dtype=np.float64)
    fixed_beta = 0.0

    y_cols = TARGET_COLS
    Y = np.stack([df[c].values.astype(np.float64) for c in y_cols], axis=1)  # (N,8)

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.zeros(n, dtype=np.int64)
    for i, idv in enumerate(idx):
        folds[idv] = i % int(n_folds)

    best_alpha = 0.70
    best_loss = np.inf

    for a in alphas:
        losses = []
        for f in range(int(n_folds)):
            val_mask = folds == f
            tr_mask = ~val_mask

            df_tr = df.loc[tr_mask].reset_index(drop=True)

            pri_tr = {
                c: float(np.clip(df_tr[c].mean(), 1e-4, 1 - 1e-4)) for c in y_cols
            }
            temps_tr = {c: 1.0 for c in y_cols}
            biases_tr = {c: 0.0 for c in y_cols}

            val_df = df.loc[val_mask].reset_index(drop=True)
            probs = _predict_train_probs_for_params(
                train_df=val_df,
                priors=pri_tr,
                temps=temps_tr,
                biases=biases_tr,
                slice_counts=slice_counts,
                slice_logit_beta=float(fixed_beta),
                slice_ref=float(slice_ref),
                alpha_any_blend=float(a),
            )
            P = np.stack([probs[c] for c in y_cols], axis=1)  # (Nv,8)
            Yv = Y[val_mask]
            losses.append(_rsna_weighted_logloss_matrix(Yv, P, y_cols))

        oof_loss = float(np.mean(losses))
        if oof_loss < best_loss:
            best_loss = oof_loss
            best_alpha = float(a)

    print(
        "OOF fitted alpha_any_blend:",
        best_alpha,
        "slice_logit_beta (fixed):",
        fixed_beta,
        "oof_rsna_weighted_logloss:",
        best_loss,
    )
    return best_alpha, float(fixed_beta)


def fit_oof_global_biases(
    train_csv: str,
    train_img_dir: str | None,
    slice_ref: float,
    alpha_any_blend: float,
    slice_logit_beta: float,
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    """
    Expected to reduce weighted logloss vs current behavior:
    For constant predictors, biases are redundant; keep them at 0.0 for stability.
    (We keep the function to preserve the pipeline's structure.)
    """
    _ = (
        train_csv,
        train_img_dir,
        slice_ref,
        alpha_any_blend,
        slice_logit_beta,
        n_folds,
        seed,
    )
    biases = {c: 0.0 for c in TARGET_COLS}
    print("OOF-averaged logit biases (fixed to 0):", biases)
    return biases




## === cell 2
time_start = time.time()

test_df = pd.read_csv(TEST_CSV)
train_priors = compute_train_priors(TRAIN_CSV)

temps = fit_temperature_scaling(TRAIN_CSV, train_priors)
print("Temperatures:", temps)

TEST_IMG_DIR_CANDIDATES = [
    os.path.join(DATA_ROOT, "test_images"),
    "/kaggle/input/rsna-2022-cervical-spine-fracture-detection/test_images",
    "/kaggle/data/rsna-2022-cervical-spine-fracture-detection/test_images",
    "../input/rsna-2022-cervical-spine-fracture-detection/test_images",
]
TEST_IMG_DIR = None
for p in TEST_IMG_DIR_CANDIDATES:
    if os.path.exists(p):
        TEST_IMG_DIR = p
        break

TRAIN_IMG_DIR_CANDIDATES = [
    os.path.join(DATA_ROOT, "train_images"),
    "/kaggle/input/rsna-2022-cervical-spine-fracture-detection/train_images",
    "/kaggle/data/rsna-2022-cervical-spine-fracture-detection/train_images",
    "../input/rsna-2022-cervical-spine-fracture-detection/train_images",
]
TRAIN_IMG_DIR = None
for p in TRAIN_IMG_DIR_CANDIDATES:
    if os.path.exists(p):
        TRAIN_IMG_DIR = p
        break
print("TRAIN_IMG_DIR:", TRAIN_IMG_DIR)

alpha_fit, beta_fit = fit_alpha_and_slice_beta_oof(
    train_csv=TRAIN_CSV,
    priors=train_priors,
    temps=temps,
    train_img_dir=TRAIN_IMG_DIR,
    slice_ref=300.0,
    n_folds=5,
    seed=42,
)

biases = fit_oof_global_biases(
    train_csv=TRAIN_CSV,
    train_img_dir=TRAIN_IMG_DIR,
    slice_ref=300.0,
    alpha_any_blend=float(alpha_fit),
    slice_logit_beta=float(beta_fit),
    n_folds=5,
    seed=42,
)

unique_studies = test_df["StudyInstanceUID"].unique().tolist()
if TEST_IMG_DIR is None:
    list_DICOM_dirs = unique_studies[:]  # dummy identifiers
else:
    list_DICOM_dirs = [os.path.join(TEST_IMG_DIR, uid) for uid in unique_studies]

print("Num test studies:", len(unique_studies))
print("TEST_IMG_DIR:", TEST_IMG_DIR)

detector = FractureDetector(
    priors=train_priors,
    temps=temps,
    biases=biases,
    slice_logit_beta=float(beta_fit),  # fixed to 0.0 by OOF fit
    slice_ref=300.0,
    alpha_any_blend=float(alpha_fit),
)
detector.predict(list_test_files=list_DICOM_dirs)

pred_map = {}
for uid, arr in detector.results.items():
    pred_map[(uid, "patient_overall")] = float(arr[0])
    for i in range(1, 8):
        pred_map[(uid, f"C{i}")] = float(arr[i])

preds = []
missing = 0
for uid, ptype in zip(
    test_df["StudyInstanceUID"].values, test_df["prediction_type"].values
):
    v = pred_map.get((uid, ptype), None)
    if v is None:
        missing += 1
        v = apply_temp_and_bias(
            float(np.clip(train_priors.get(ptype, 0.01), 1e-4, 1 - 1e-4)),
            float(temps.get(ptype, 1.0)),
            float(biases.get(ptype, 0.0)),
        )
    preds.append(v)

sub = pd.DataFrame(
    {"row_id": test_df["row_id"].values, "fractured": np.clip(preds, 1e-4, 1 - 1e-4)}
)
SAVE_CSV = "submission.csv"
sub.to_csv(SAVE_CSV, index=False)

print("Wrote:", SAVE_CSV, "rows:", len(sub), "missing_mapped:", missing)
print("Time:", time.time() - time_start, "sec")
gc.collect()
