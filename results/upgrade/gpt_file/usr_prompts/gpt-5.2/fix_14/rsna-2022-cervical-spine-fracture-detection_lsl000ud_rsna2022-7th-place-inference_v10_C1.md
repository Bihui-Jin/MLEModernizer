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

0.3110608877945774

# 6. Current score

0.5635

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5639) has done: 'The main failure is that the private `../input/srccode/src` (and also `../input/models` / `../input/plans-nnunet`) paths don’t exist in your provided environment, so all custom nnU-Net classes can’t be imported and the pipeline stops before producing `submission.csv`. To make this run end-to-end and yield a valid submission, I replace the missing-model inference with a deterministic, competition-format-safe baseline that outputs calibrated constant probabilities per `prediction_type` (using train prevalences with clipping). This preserves the “predict probabilities for 8 labels per study” evaluation semantics, fixes submission alignment by generating predictions directly from `test.csv` row order, and ensures `submission.csv` is written with correct columns and numeric `fractured` values. Since you currently have “Not yielded”, the primary objective is correctness and a reasonable baseline score; this approach should be meaningfully better than arbitrary constants while staying lightweight and within the 600s limit.'
- What this solution (achieved 0.55968) has done: 'Your current baseline uses raw label prevalences from train; that tends to underweight the crucial `patient_overall` signal and is typically miscalibrated for the weighted log-loss, which can keep the score high (worse). I keep the same “constant-per-label” core logic, but (1) compute a *consistent* `patient_overall` probability as the union of vertebra probabilities (1 − ∏(1−p(Ck))) so it better matches the definition, and (2) apply a very small amount of symmetric smoothing (toward 0.5) to reduce overconfident errors without changing the approach. I also keep strict row alignment with `sample_submission.csv` to avoid any accidental ordering mismatch. These minimal calibration tweaks are safe and should move your loss down toward the 0.311 target.'
- What this solution (achieved 0.56039) has done: 'To move your loss down toward the 0.311 target while keeping the exact “constant-per-label probability” core logic, I only adjust the calibration step. Specifically, I (1) tune the symmetric smoothing strength (your current 0.10 is likely too strong and pulls probabilities too close to 0.5), and (2) add a tiny label-wise prior-count (Laplace/Beta) shrinkage when estimating prevalences, which reduces variance from only 202 studies without changing the approach. I keep the union-based `patient_overall` computation and the strict row alignment to `sample_submission.csv` unchanged. These are minimal, safe changes that typically reduce weighted log-loss for this competition.'
- What this solution (achieved 0.5635) has done: 'Your current baseline is already stable and valid, but it’s likely hurting the weighted log-loss by (a) forcing `patient_overall` to be the *union* of vertebra prevalences (which can overestimate positives), and (b) using a symmetric smoothing-to-0.5 that typically worsens log-loss when class prevalence is low. To move the loss down toward the 0.311 target while keeping the exact same “constant-per-label probability” approach, I (1) estimate `patient_overall` directly from the training `patient_overall` column (with the same tiny Laplace prior you already use), and (2) remove the extra smoothing step (keep only safe clipping). Everything else—paths, row alignment via `test.csv`/`sample_submission.csv`, output schema, and constant-per-label semantics—stays unchanged.'
- What this solution (achieved 0.55863) has done: 'Your current constant-per-label baseline is valid but underfits the strong coupling between vertebra labels and `patient_overall`, which is heavily weighted in the metric; improving that single column tends to reduce the weighted log loss most with minimal risk. I keep the same “constant probability per label” core logic, but compute a better-calibrated `patient_overall` constant from the vertebra constants via a noisy-OR union, then blend it slightly with the directly observed train `patient_overall` prevalence to avoid overestimation. I keep Laplace smoothing and clipping unchanged, preserve exact row alignment to `sample_submission.csv`, and still write `submission.csv` in the required format. This should move your loss down (better) toward the 0.311 target without changing the modeling approach.'
- What this solution (achieved 0.56013) has done: 'Your current solution is already producing a valid submission, but the score is far from the target and (given the constraints) the biggest safe lever is better calibration of the **heavily weighted** `patient_overall` constant. I keep the same “constant probability per label” core logic and the same Laplace-smoothed per-vertebra prevalences, but I replace the fixed 50/50 blend with a **logit-space blend** (more appropriate for log-loss) and tune the blend weight to lean more toward the directly observed `patient_overall` prevalence (since the union noisy-OR tends to overshoot). I also ensure `patient_overall` is never below `max(C1..C7)` (a necessary consistency constraint that usually reduces loss for this dataset) while keeping clipping and row alignment unchanged. These are minimal calibration-only changes aimed at lowering the loss (improving) toward 0.311 without changing the modeling approach.'
- What this solution (achieved 0.56119) has done: 'Your current constant-per-label baseline is stable but too far from the target (0.56013 vs 0.31106, lower is better), so we need a minimal calibration-only change that legitimately reduces log-loss without changing the approach. The safest lever here is to better handle the *heavily weighted* `patient_overall` by enforcing probabilistic consistency: `P(any)` should equal the noisy-OR union of vertebra probabilities, and we can calibrate vertebra probabilities to match the observed train `patient_overall` prevalence. Concretely, we keep the same Laplace-smoothed per-vertebra constants, but apply a single global scale in **logit space** to all C1–C7 so that the resulting union matches the direct `patient_overall` prevalence; then we set `patient_overall` to that union. This preserves the exact “constant probability per label” semantics while usually lowering the weighted log-loss dominated by `patient_overall`.'
- What this solution (achieved 0.56126) has done: 'Your current approach is a constant-per-label probability baseline, so the only safe way to improve the weighted log-loss (lower is better) without changing core logic is to calibrate those constants better. The biggest lever is the heavily weighted `patient_overall`: instead of forcing it to equal the noisy-OR union (which can miscalibrate), we keep your global logit-scale calibration for C1–C7 but then set `patient_overall` via a *logit-space blend* between the direct train prevalence and the union, plus enforce the consistency constraint `patient_overall >= max(C1..C7)`. This keeps identical evaluation semantics (still constant per label) while typically reducing the dominant patient_overall loss. Submission alignment and CSV writing remain unchanged.'
- What this solution (achieved 0.56119) has done: 'We keep your constant-per-label baseline intact and only adjust the calibration that dominates the weighted log-loss: `patient_overall`. Specifically, we stop blending `patient_overall` (which can miscalibrate a heavily weighted label) and instead set it deterministically as the calibrated noisy-OR union of the already-calibrated C1–C7 probabilities, while still enforcing `patient_overall >= max(C1..C7)` for consistency. This is a minimal change (no new models, no new data, same submission semantics) and is the most likely to reduce the loss from ~0.561 toward your 0.311 target given the metric weighting. Everything else—Laplace smoothing, global logit scaling for C1–C7, row alignment, clipping, and CSV writing—remains unchanged.'
- What this solution (achieved 0.56575) has done: 'We keep your exact “constant-per-label probability” approach but fix the part that most likely hurts weighted log-loss: the global logit scaling for C1–C7 currently matches `patient_overall` prevalence, which can worsen vertebra-level losses and not actually minimize the *weighted* objective. Instead, we keep Laplace-smoothed prevalences, but choose a single global logit-scale `s` by directly minimizing the **expected weighted log-loss on the training set** (using label-wise weights; heavier for `patient_overall`) while still setting `patient_overall` as the noisy-OR union of the scaled C1–C7 (and enforcing `>= max(Ck)` consistency). This is a calibration-only change (same semantics, still constants per label), and it should reduce loss from ~0.561 toward the 0.311 target without changing any model/training logic. Submission alignment and CSV writing remain identical and safe.'
- What this solution (achieved 0.56013) has done: 'Your current constant-per-label baseline is already stable, but the global logit-scale calibration you added can easily overfit the tiny 202-study training set and shift probabilities away from what generalizes, which likely explains the regression to 0.56575. To move the loss down (lower is better) toward the 0.311 target with minimal logic change, I keep the exact same “constant per label” semantics and Laplace smoothing, but replace the train-loss grid-search with a safer closed-form calibration: use the smoothed per-vertebra prevalences directly, then set `patient_overall` as a logit-space blend between the direct `patient_overall` prevalence and the noisy-OR union (still enforcing `patient_overall >= max(Ck)`). This focuses improvement where the metric is most heavily weighted (patient_overall) without introducing a tuning loop that can worsen generalization. Submission alignment and CSV writing remain identical.'
- What this solution (achieved 0.5635) has done: 'We keep your exact “constant probability per label” approach and only adjust calibration where the metric is most sensitive: the heavily weighted `patient_overall`. Instead of a fixed blend weight, we choose the blend weight by minimizing the **weighted log loss on the training set** using a tiny grid search (21 values) while keeping vertebra probabilities unchanged, which is a minimal change and should reliably lower the leaderboard loss from ~0.56. We still enforce the consistency constraint `patient_overall >= max(C1..C7)` and keep clipping and row alignment identical, ensuring a valid `submission.csv` is produced. This stays within the 600s limit and preserves evaluation semantics.'
- What this solution (achieved 0.5635) has done: 'Your current submission is a constant-per-label baseline; the biggest safe lever to reduce the (weighted) log-loss is to calibrate the heavily weighted `patient_overall` probability without changing the overall approach. I keep the same Laplace-smoothed per-label constants, but instead of tuning the blend only on the `patient_overall` column, I tune a single blend weight by minimizing the **full weighted multi-label log-loss on the training set** (all 8 labels), which better matches the competition metric while staying within the same “constant probabilities” semantics. I also compute `patient_overall` as a logit-blend between the direct prevalence and the noisy-OR union (then enforce `patient_overall >= max(C1..C7)`), but keep vertebra probabilities unchanged to preserve core logic. Submission alignment and CSV writing remain identical and deterministic.'

# 9. Code solution

## === cell 0
import os
import sys
import math
import time
import gc
import numpy as np
import pandas as pd


DATA_ROOT = "/kaggle/input/rsna-2022-cervical-spine-fracture-detection"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
SAVE_CSV = "submission.csv"

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(TEST_CSV), f"Missing {TEST_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"

print("Loaded paths OK")



## === cell 1
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

target_cols = ["patient_overall", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]
c_cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]

assert all(c in train_df.columns for c in target_cols), "Train targets missing"
assert set(sample_sub.columns) == {
    "row_id",
    "fractured",
}, "Unexpected sample submission columns"
assert set(test_df.columns) == {
    "StudyInstanceUID",
    "prediction_type",
    "row_id",
}, "Unexpected test columns"

eps = 1e-4


def _logit(p: float) -> float:
    p = float(np.clip(p, eps, 1.0 - eps))
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _noisy_or(ps) -> float:
    prod = 1.0
    for p in ps:
        prod *= 1.0 - float(p)
    return float(np.clip(1.0 - prod, eps, 1.0 - eps))


WEIGHTS = {
    "patient_overall": 7.0,
    "C1": 1.0,
    "C2": 1.0,
    "C3": 1.0,
    "C4": 1.0,
    "C5": 1.0,
    "C6": 1.0,
    "C7": 1.0,
}

n = int(len(train_df))

alpha = 1.0
beta = 1.0

prevalence = {}

for c in c_cols:
    s = float(train_df[c].sum())
    p = (s + alpha) / (n + alpha + beta)
    prevalence[c] = float(np.clip(p, eps, 1.0 - eps))

s_any = float(train_df["patient_overall"].sum())
p_any_direct = float(np.clip((s_any + alpha) / (n + alpha + beta), eps, 1.0 - eps))
p_union = _noisy_or([prevalence[c] for c in c_cols])


def _weighted_logloss_binary(y: np.ndarray, p: float, w: float) -> float:
    p = float(np.clip(p, eps, 1.0 - eps))
    return float(-w * np.mean(y * math.log(p) + (1.0 - y) * math.log(1.0 - p)))


y_cache = {c: train_df[c].astype(np.float32).values for c in target_cols}


def _full_weighted_loss(p_any: float) -> float:
    loss = 0.0
    loss += _weighted_logloss_binary(
        y_cache["patient_overall"], p_any, WEIGHTS["patient_overall"]
    )
    for c in c_cols:
        loss += _weighted_logloss_binary(y_cache[c], prevalence[c], WEIGHTS[c])
    return float(loss / len(target_cols))


grid = np.linspace(0.0, 1.0, 41, dtype=np.float64)

best_w = None
best_loss = None
best_p_any = None

for bw in grid:
    p_blend = _sigmoid(bw * _logit(p_any_direct) + (1.0 - bw) * _logit(p_union))
    p_final = float(max(p_blend, max(prevalence[c] for c in c_cols)))
    loss = _full_weighted_loss(p_final)
    if (best_loss is None) or (loss < best_loss):
        best_loss = float(loss)
        best_w = float(bw)
        best_p_any = float(p_final)

prevalence["patient_overall"] = float(np.clip(best_p_any, eps, 1.0 - eps))

print(
    "Train prevalence (Laplace-smoothed; patient_overall logit-blend tuned on full weighted train loss):"
)
print({k: float(v) for k, v in prevalence.items()})
print(
    f"patient_overall direct(prev)={p_any_direct:.6f}  "
    f"union(C1..C7)={p_union:.6f}  "
    f"best_blend_w={best_w:.3f}  "
    f"patient_overall(final)={prevalence['patient_overall']:.6f}  "
    f"train_full_weighted_loss={best_loss:.6f}"
)



## === cell 2
unknown_types = sorted(set(test_df["prediction_type"].unique()) - set(target_cols))
if unknown_types:
    print("WARNING: unknown prediction_type values:", unknown_types)

pred = (
    test_df["prediction_type"]
    .map(lambda t: prevalence.get(t, prevalence["patient_overall"]))
    .astype(np.float32)
)

sub = pd.DataFrame({"row_id": test_df["row_id"].values, "fractured": pred.values})

if len(sub) != len(sample_sub):
    raise RuntimeError(f"Submission row count {len(sub)} != sample {len(sample_sub)}")

if not sub["row_id"].equals(sample_sub["row_id"]):
    sub = sample_sub[["row_id"]].merge(sub, on="row_id", how="left")
    if sub["fractured"].isna().any():
        sub["fractured"] = (
            sub["fractured"].fillna(prevalence["patient_overall"]).astype(np.float32)
        )

sub["fractured"] = sub["fractured"].clip(eps, 1.0 - eps).astype(np.float32)

sub.to_csv(SAVE_CSV, index=False)
print(f"Wrote {SAVE_CSV} with shape {sub.shape}")
print(sub.head())
