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

0.3120052196223922

# 6. Current score

0.56607

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5635) has done: 'The failures come from relying on an extra `../input/srccode/src` codebase (and model/plan files) that are not present in your provided environment, so `NNUnetCTPredictor` and related utilities never import. To make the notebook run end-to-end and still produce a valid submission, I replace those missing dependencies with a minimal, deterministic baseline that uses only the provided CSV metadata and outputs well-calibrated constant probabilities per label. I also fix the submission generation to strictly follow `test.csv`/`sample_submission.csv` row ordering (so it always matches the required 14536 rows and `row_id` values). This is score-oriented but conservative: it won’t crash, and it should beat an invalid/no-submission while staying within the competition’s expected format.'
- What this solution (achieved 0.55969) has done: 'Your current solution is a constant-per-label prior baseline, and the main way to move the weighted logloss down (better) without changing the overall approach is to calibrate those constants to the metric. I keep the same “predict a constant probability per `prediction_type`” core logic, but (1) compute smoothed priors with a tunable strength and (2) apply a tiny, label-wise shrink toward 0.5 to reduce overconfidence—both are safe, deterministic, and often improve logloss. I also compute `patient_overall` consistently from the vertebra priors (probability of any fracture) and lightly blend it with the empirical `patient_overall` rate to better match its semantics and higher weight. Submission generation and row alignment remain identical to your current code.'
- What this solution (achieved 0.56491) has done: 'Your current baseline is already “constant-per-label prior,” so the safest way to improve weighted logloss (lower) without changing core logic is to tune those constants to the metric and dataset size. I keep the exact same approach (map `prediction_type` → constant probability; preserve submission alignment), but adjust (1) the Laplace/Beta smoothing strength per label (including a larger strength for the heavily-weighted `patient_overall`) and (2) shrinkage toward 0.5 (slightly stronger for `patient_overall`) to reduce overconfidence. I also compute `patient_overall` as a blend of the empirical patient rate and the “any-from-levels” probability, but with updated blend weights that typically better match the semantics. These are minimal, deterministic changes aimed at reducing logloss from 0.55969 toward your 0.3120 target without altering the overall modeling strategy.'
- What this solution (achieved 0.55872) has done: 'Your current gap to target is large and lower-is-better, so we should improve logloss without changing the “constant per label” core logic. The biggest easy win is to tune those constants in a metric-aware way: compute label-wise Beta-smoothed rates and then choose an optimal additional smoothing strength (equivalently, shrink toward 0.5) by minimizing the weighted logloss on the training labels. We keep the same prediction semantics (a single constant per `prediction_type`, with `patient_overall` derived from levels and blended with its empirical rate), but we replace hand-picked hyperparameters with deterministic, data-driven ones found via a tiny grid search. Submission generation and row alignment remain identical and still write a valid `submission.csv`.'
- What this solution (achieved 0.55872) has done: 'Your current solution is already the minimal “constant per `prediction_type`” baseline, but it is fitting those constants using an approximate objective (treating labels as independent rows) rather than the true weighted logloss aggregated over the eight rows per study. I keep the same core idea (one constant probability per label type, with `patient_overall` derived/blended) and change only the tuning step to directly minimize the exact competition metric on the training set by constructing the same 8-rows-per-study structure as `test.csv`. I also add a very small and safe probability clamp (still deterministic) to avoid extreme probabilities harming logloss. This should move the logloss down (better, closer to 0.312) without changing the model class or submission semantics.'
- What this solution (achieved 0.5607) has done: 'Your current baseline is already “one constant probability per `prediction_type`”, but it’s tuned to training labels without respecting the label constraints (levels should not exceed `patient_overall`, and `patient_overall` should be consistent with “any level fractured”). To move logloss down toward the target with minimal semantic change, I keep the same constant-per-label approach but (1) enforce a consistent `patient_overall = P(any level)` derived from the level constants, and (2) clip each level probability to be ≤ `patient_overall` so you never predict “more likely fractured at Ck than overall”. I also make the tuning minimize the weighted logloss over the exact 8-rows-per-study structure (same as your previous step), but now only tune smoothing/shrink for the 7 levels; `patient_overall` follows deterministically. Submission generation stays identical and aligned to `sample_submission.csv`.'
- What this solution (achieved 0.56032) has done: 'Your current solution is a constant-per-label baseline, so the only safe way to move logloss down (lower-is-better) without changing the modeling class is to tune those constants against the *exact* competition metric, including per-label weights. I keep your “one constant per `prediction_type`” core logic and the consistency constraint `patient_overall = P(any level)`, but I (1) optimize the 7 level probabilities directly (instead of via alpha/k heuristics) by solving for the per-label optimum under weighted logloss, then (2) apply a single global shrinkage toward 0.5 and pick its value by minimizing the weighted logloss on the full 8-rows-per-study structure. This remains deterministic, fast, and preserves the same evaluation semantics while typically improving calibration and lowering logloss. Submission generation and row alignment remain identical to ensure a valid 14536-row `submission.csv`.'
- What this solution (achieved 0.56607) has done: 'Your current code is already a constant-per-label baseline; the biggest safe win toward a lower weighted logloss is to tune those constants against the *exact* competition objective and structure (8 rows per study with the correct label weights), rather than the row-wise approximation. I keep the same core semantics (one constant per C1–C7, with `patient_overall = P(any)` derived from them and levels clipped to be ≤ overall), but I optimize the 7 level probabilities directly by coordinate descent on the true weighted logloss computed from `train.csv`. Then I do a tiny 1D global shrink-to-0.5 search (same as you already do) but evaluated on the exact metric; this improves calibration without changing the model class. Submission generation, row alignment, and clamping remain unchanged and still produce a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import time
import numpy as np
import pandas as pd

DATA_ROOT = "../input/rsna-2022-cervical-spine-fracture-detection"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
SAVE_CSV = "submission.csv"

print("DATA_ROOT exists:", os.path.exists(DATA_ROOT))
print("Files:", [os.path.exists(p) for p in (TRAIN_CSV, TEST_CSV, SAMPLE_SUB)])



## === cell 1
time_start = time.time()

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

LEVELS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
ALL_LABELS = LEVELS + ["patient_overall"]

assert all(
    c in train_df.columns for c in ["StudyInstanceUID"] + ALL_LABELS
), "Missing expected columns in train.csv"
assert all(
    c in test_df.columns for c in ["StudyInstanceUID", "prediction_type", "row_id"]
), "Missing expected columns in test.csv"

WEIGHTS = {c: 1.0 for c in LEVELS}
WEIGHTS["patient_overall"] = 7.0


def clamp(p, lo=1e-6, hi=1.0 - 1e-6):
    return float(np.clip(p, lo, hi))


def build_consistent_priors_from_levels(level_probs):
    """
    Core logic preserved:
      - one constant probability per vertebra level
      - patient_overall derived as P(any level), assuming independence
      - enforce level <= overall for consistency
    """
    pri_level = {c: clamp(level_probs[c]) for c in LEVELS}

    p_none = 1.0
    for c in LEVELS:
        p_none *= 1.0 - pri_level[c]
    p_any = clamp(1.0 - p_none)

    for c in LEVELS:
        if pri_level[c] > p_any:
            pri_level[c] = p_any

    priors = dict(pri_level)
    priors["patient_overall"] = p_any
    return priors


def weighted_logloss_binary(y, p, w):
    eps = 1e-15
    p = np.clip(p, eps, 1.0 - eps)
    ll = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return float((ll * w).sum() / w.sum())


y_by_type = {lab: train_df[lab].astype(np.float64).values for lab in ALL_LABELS}
w_by_type = {lab: float(WEIGHTS[lab]) for lab in ALL_LABELS}
w_sum = sum(w_by_type.values()) * train_df.shape[0]


def rsna_exact_weighted_logloss_from_levels(level_probs):
    priors = build_consistent_priors_from_levels(level_probs)
    total = 0.0
    for lab in ALL_LABELS:
        y = y_by_type[lab]
        p = np.full_like(y, priors[lab], dtype=np.float64)
        w = w_by_type[lab]
        eps = 1e-15
        p = np.clip(p, eps, 1.0 - eps)
        ll = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        total += ll.sum() * w
    return float(total / w_sum)


pos_rate = {lab: float(train_df[lab].mean()) for lab in ALL_LABELS}
base_levels = {c: clamp(pos_rate[c]) for c in LEVELS}

print("Train pos rates:", {k: round(v, 6) for k, v in pos_rate.items()})




## === cell 2
def coordinate_descent_optimize_levels(
    init_levels,
    n_rounds=8,
    grid_size=121,
    grid_lo=1e-4,
    grid_hi=0.5,
):
    levels = dict(init_levels)
    best_loss = rsna_exact_weighted_logloss_from_levels(levels)

    base_grid = np.linspace(grid_lo, grid_hi, grid_size).astype(np.float64)

    for r in range(n_rounds):
        improved = False
        for c in LEVELS:
            cur = levels[c]
            grid = np.unique(
                np.sort(np.concatenate([base_grid, np.array([cur], dtype=np.float64)]))
            )
            best_c = cur
            best_c_loss = best_loss

            trial = dict(levels)
            for val in grid:
                trial[c] = clamp(float(val))
                loss = rsna_exact_weighted_logloss_from_levels(trial)
                if loss < best_c_loss:
                    best_c_loss = loss
                    best_c = float(val)

            if best_c != cur:
                levels[c] = clamp(best_c)
                best_loss = best_c_loss
                improved = True

        print(
            f"Round {r+1}/{n_rounds}: loss={best_loss:.6f}, levels={ {k: round(levels[k],6) for k in LEVELS} }"
        )
        if not improved:
            break

    return levels, best_loss


opt_levels, opt_loss = coordinate_descent_optimize_levels(
    base_levels,
    n_rounds=10,
    grid_size=151,
    grid_lo=1e-4,
    grid_hi=0.35,  # keep within reasonable range to avoid overconfident constants
)

print("Optimized (pre-shrink) exact train loss:", opt_loss)
print("Optimized levels:", {k: float(v) for k, v in opt_levels.items()})




## === cell 3
def apply_global_shrink(level_probs, lam):
    out = {}
    for c in LEVELS:
        out[c] = clamp((1.0 - lam) * level_probs[c] + lam * 0.5)
    return out


lam_grid = np.linspace(0.0, 0.95, 39)  # slightly finer, still tiny/fast

best = {"loss": float("inf"), "lam": None, "priors": None, "levels": None}
for lam in lam_grid:
    levels_tmp = apply_global_shrink(opt_levels, float(lam))
    loss = rsna_exact_weighted_logloss_from_levels(levels_tmp)
    if loss < best["loss"]:
        best["loss"] = float(loss)
        best["lam"] = float(lam)
        best["levels"] = dict(levels_tmp)
        best["priors"] = build_consistent_priors_from_levels(levels_tmp)

print("Best exact train weighted-logloss:", best["loss"])
print("Best global shrink lam:", best["lam"])
print("Best priors:", {k: float(v) for k, v in best["priors"].items()})

priors = best["priors"]

missing_types = sorted(set(test_df["prediction_type"].unique()) - set(priors.keys()))
if missing_types:
    raise ValueError(f"Unexpected prediction_type values in test.csv: {missing_types}")



## === cell 4
test_df = test_df.copy()
test_df["fractured"] = test_df["prediction_type"].map(priors).astype(np.float32)

sample_df = pd.read_csv(SAMPLE_SUB)
sub_df = sample_df[["row_id"]].merge(
    test_df[["row_id", "fractured"]],
    on="row_id",
    how="left",
    validate="one_to_one",
)

if sub_df["fractured"].isna().any():
    sub_df["fractured"] = (
        sub_df["fractured"].fillna(priors["patient_overall"]).astype(np.float32)
    )

sub_df["fractured"] = sub_df["fractured"].clip(1e-6, 1.0 - 1e-6).astype(np.float32)
sub_df.to_csv(SAVE_CSV, index=False)

print("Wrote:", SAVE_CSV, "shape:", sub_df.shape)
print("fractured summary:", sub_df["fractured"].describe())
print(f"==> Finish using time: {time.time() - time_start:.2f}s")
print("Head:\n", sub_df.head())
print("Tail:\n", sub_df.tail())
