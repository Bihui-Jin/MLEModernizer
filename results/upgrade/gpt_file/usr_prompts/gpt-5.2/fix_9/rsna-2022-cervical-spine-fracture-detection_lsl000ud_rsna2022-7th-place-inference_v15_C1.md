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

0.2687464020953224

# 6. Current score

0.56881

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5635) has done: 'I remove the hard dependency on the missing `Utils/Training` source tree (which isn’t present in this Kaggle dataset) and replace it with a small, robust fallback pipeline that still produces a valid `submission.csv`. Specifically: (1) fix the import/root-discovery logic so the notebook runs in this environment, (2) ensure we always write a correctly formatted submission aligned to `test.csv`, and (3) provide calibrated constant probabilities (using train-label prevalences with smoothing) as a safe baseline so you get a reasonable log-loss instead of “no submission”. This keeps changes minimal in spirit (it’s a bug-fix to run end-to-end) and avoids any heavy DICOM/model inference so it finish under the time limit.'
- What this solution (achieved 0.56337) has done: 'Your current submission is a smoothed-prevalence constant baseline; to move the log-loss down toward the target, the most direct minimal improvement is to better calibrate the predicted probabilities to the evaluation’s weighted log-loss (patient_overall is more important). I keep the same core approach (no DICOM/modeling), but (1) tune the Laplace smoothing strength separately for patient_overall vs vertebrae levels, and (2) apply a tiny, safe probability shrink/temperature-like adjustment to avoid being overconfident on rare labels. These changes preserve semantics (still constant-by-class probabilities derived from train labels) while typically improving weighted log-loss versus a single global alpha. The output remains fully aligned to `test.csv` and writes a valid `submission.csv`.'
- What this solution (achieved 0.56915) has done: 'You’re currently using constant per-class probabilities; the safest way to move log-loss down (closer to the 0.2687 target) without changing the core logic is to calibrate those constants against the metric. I keep the same “smoothed prevalence + shrink toward 0.5” approach, but (1) use per-label shrink (more shrink for rare C-levels, less for patient_overall), and (2) add a very small, data-driven “power” calibration that slightly reduces overconfidence while keeping ordering/semantics intact. These are minimal post-processing changes that typically improve weighted log-loss on imbalanced labels and maintain a valid, correctly aligned `submission.csv`. Paths and submission formatting remain unchanged.'
- What this solution (achieved 0.56467) has done: 'Your current score (0.56915, lower-is-better) is still far from the target (0.2687), so we should improve in the direction of lower log-loss. Keeping your “constant-by-class probabilities from train prevalences” core logic intact, the safest gain is to optimize the few calibration hyperparameters (Laplace alphas, shrink ranges, and gammas) against an internal cross-validated weighted log-loss that matches the competition’s row-wise evaluation (with heavier weight on `patient_overall`). This avoids any modeling/DICOM work, stays deterministic and fast, and typically lowers log-loss materially versus fixed hand-set calibration. Finally, we keep the exact same submission construction aligned to `test.csv` and still write `submission.csv`.'
- What this solution (achieved 0.5643) has done: 'I keep your constant-by-class prevalence baseline (no DICOM/modeling changes) but make the calibration match the competition metric more closely. Specifically, I replace the approximate CV objective with an exact row-wise weighted log-loss computed on an expanded (Study, prediction_type) table, so the tuning optimizes the same structure Kaggle scores. Then I do a very small final local search around the best grid parameters to squeeze out a bit more loss reduction without changing the overall approach or runtime significantly. The submission construction stays aligned to `test.csv` and still writes a valid `submission.csv`.'
- What this solution (achieved 0.56881) has done: 'Your current approach is a constant-by-class prevalence baseline, so the only safe way to reduce log-loss (lower-is-better) without changing core semantics is to make the calibration/tuning match the competition’s *weighted* row-wise log-loss more faithfully. I keep the exact same prevalence + shrink + power-calibration logic, but change CV to be deterministic and lower-variance by using repeated, stratified folds on the **patient_overall** label (the highest-weighted target), then tune parameters against the mean loss across repeats. Finally, I add a tiny, metric-aligned “patient_overall consistency” step at prediction time that sets patient_overall to the probabilistic OR of C1–C7 (still derived purely from the same constants), which is usually better under weighted log-loss than forcing only a max.'
- What this solution (achieved 0.56881) has done: 'Your current score (0.56881, lower-is-better) is far from the target (0.2687), so we should reduce log-loss; with your “constant-by-class prevalence + shrink + power calibration” core logic, the biggest remaining controllable lever is making the CV objective match Kaggle’s *row-wise weighting* more faithfully. I keep your exact prediction family but (1) use the official-ish label weights (patient_overall=7, C1–C7=1) **normalized** so the CV loss scale matches Kaggle’s averaging, and (2) tune a single additional tiny regularizer: a global epsilon floor/ceiling used consistently in both CV scoring and submission clipping to avoid log-loss blowups from overconfident constants. These are minimal, semantics-preserving calibration changes (still constant probabilities derived from train labels) and should nudge the score downward without introducing modeling or DICOM work. The script still runs end-to-end and writes a valid `submission.csv` aligned to `test.csv`.'

# 9. Code solution

## === cell 0
import os
import time
import numpy as np
import pandas as pd

BASE_DATA_DIR = "/kaggle/input/rsna-2022-cervical-spine-fracture-detection"
if not os.path.isdir(BASE_DATA_DIR):
    BASE_DATA_DIR = "../input/rsna-2022-cervical-spine-fracture-detection"
if not os.path.isdir(BASE_DATA_DIR):
    BASE_DATA_DIR = "/kaggle/data/input/rsna-2022-cervical-spine-fracture-detection"

TRAIN_CSV_PATH = f"{BASE_DATA_DIR}/train.csv"
TEST_CSV_PATH = f"{BASE_DATA_DIR}/test.csv"
SAVE_CSV = "submission.csv"

assert os.path.exists(TEST_CSV_PATH), f"Missing test.csv at {TEST_CSV_PATH}"
assert os.path.exists(TRAIN_CSV_PATH), f"Missing train.csv at {TRAIN_CSV_PATH}"

print("==> Using BASE_DATA_DIR:", BASE_DATA_DIR)
print("==> train.csv:", TRAIN_CSV_PATH)
print("==> test.csv:", TEST_CSV_PATH)



## === cell 1
time_start = time.time()

train_df = pd.read_csv(TRAIN_CSV_PATH)
test_df = pd.read_csv(TEST_CSV_PATH)

targets = ["patient_overall"] + [f"C{i}" for i in range(1, 8)]
missing_targets = [c for c in targets if c not in train_df.columns]
if missing_targets:
    raise ValueError(f"train.csv missing expected target columns: {missing_targets}")

y = train_df[targets].fillna(0).astype(np.float32).values  # (N, 8)

W_PATIENT = 7.0
W_LEVEL = 1.0
weights_raw = np.array([W_PATIENT] + [W_LEVEL] * 7, dtype=np.float32)
weights = weights_raw / weights_raw.mean()

t2i = {t: i for i, t in enumerate(targets)}

EPS = 2e-6  # small but slightly safer than 1e-6 for constant baselines under log-loss


def weighted_rowwise_logloss_from_probs(y_true_mat, p_vec):
    p_vec = np.clip(p_vec.astype(np.float32), EPS, 1.0 - EPS)
    p = p_vec[None, :]  # (1,8) broadcast
    loss = -(y_true_mat * np.log(p) + (1.0 - y_true_mat) * np.log(1.0 - p))  # (N,8)
    loss = loss * weights[None, :]
    return float(loss.mean())


def power_calibrate_vec(p, gamma):
    p = np.clip(p, EPS, 1.0 - EPS)
    a = np.power(p, gamma)
    b = np.power(1.0 - p, gamma)
    out = a / (a + b)
    return np.clip(out, EPS, 1.0 - EPS)


def make_prevalence_from_y(y_train, params):
    (
        alpha_patient,
        alpha_level,
        shrink_patient,
        sl_min,
        sl_max,
        gamma_patient,
        gamma_level,
    ) = params
    n_tr = y_train.shape[0]
    pos = y_train.sum(axis=0)  # (8,)

    a = np.array([alpha_patient] + [alpha_level] * 7, dtype=np.float32)
    p = (pos + a) / (n_tr + 2.0 * a)
    p = np.clip(p, 1e-4, 1.0 - 1e-4)

    p[0] = max(float(p[0]), float(p[1:].max()))

    raw_prev = pos / max(1.0, float(n_tr))
    s = np.empty_like(p)
    s[0] = shrink_patient
    t = np.minimum(raw_prev[1:] / 0.15, 1.0)
    s[1:] = sl_min + (sl_max - sl_min) * t

    p = 0.5 + s * (p - 0.5)
    p = np.clip(p, 1e-4, 1.0 - 1e-4)

    p[0] = power_calibrate_vec(p[0], gamma_patient)
    p[1:] = power_calibrate_vec(p[1:], gamma_level)

    p[0] = max(float(p[0]), float(p[1:].max()))
    p[0] = float(np.clip(p[0], EPS, 1.0 - EPS))
    return p.astype(np.float32)


def _make_stratified_folds(binary_labels, k=5, seed=123):
    rng = np.random.RandomState(seed)
    idx_pos = np.where(binary_labels > 0.5)[0]
    idx_neg = np.where(binary_labels <= 0.5)[0]
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)
    pos_folds = np.array_split(idx_pos, k)
    neg_folds = np.array_split(idx_neg, k)
    folds = [np.concatenate([pos_folds[i], neg_folds[i]]) for i in range(k)]
    for i in range(k):
        rng.shuffle(folds[i])
    return folds


def cv_score_for_params(y_all, params, k=5, repeats=6, seed=123):
    y_po = y_all[:, 0]
    scores = []
    for r in range(repeats):
        folds = _make_stratified_folds(y_po, k=k, seed=seed + 1000 * r)
        for fi in range(k):
            val_idx = folds[fi]
            tr_idx = np.concatenate([folds[j] for j in range(k) if j != fi])
            p_vec = make_prevalence_from_y(y_all[tr_idx], params)
            scores.append(weighted_rowwise_logloss_from_probs(y_all[val_idx], p_vec))
    return float(np.mean(scores))


alpha_patient_grid = [0.5, 1.0, 2.0, 4.0, 6.0]
alpha_level_grid = [0.5, 1.0, 1.25, 2.0, 3.0, 4.0]
shrink_patient_grid = [0.90, 0.95, 0.98, 1.00]
sl_min_grid = [0.70, 0.76, 0.78, 0.82, 0.85]
sl_max_grid = [0.86, 0.90, 0.94, 0.97, 0.99]
gamma_patient_grid = [0.90, 0.95, 0.98, 1.00, 1.05]
gamma_level_grid = [0.85, 0.90, 0.92, 0.95, 1.00]

best_params = None
best_score = float("inf")

baseline_params = (2.0, 1.25, 0.95, 0.78, 0.90, 0.98, 0.92)
baseline_cv = cv_score_for_params(y, baseline_params, k=5, repeats=6, seed=123)
best_params, best_score = baseline_params, baseline_cv

print(
    f"==> Baseline CV (repeated stratified) weighted logloss: {baseline_cv:.6f} with params={baseline_params}"
)

stage1_sl = (0.95, 0.78, 0.90)  # shrink_patient, sl_min, sl_max
for ap in alpha_patient_grid:
    for al in alpha_level_grid:
        for gp in gamma_patient_grid:
            for gl in gamma_level_grid:
                params = (ap, al, stage1_sl[0], stage1_sl[1], stage1_sl[2], gp, gl)
                sc = cv_score_for_params(y, params, k=5, repeats=6, seed=123)
                if sc < best_score:
                    best_score, best_params = sc, params

print(f"==> After stage1 best CV: {best_score:.6f} with params={best_params}")

ap, al, _, _, _, gp, gl = best_params
for sp in shrink_patient_grid:
    for sl_min in sl_min_grid:
        for sl_max in sl_max_grid:
            if sl_max < sl_min:
                continue
            params = (ap, al, sp, sl_min, sl_max, gp, gl)
            sc = cv_score_for_params(y, params, k=5, repeats=6, seed=123)
            if sc < best_score:
                best_score, best_params = sc, params

print(f"==> After stage2 best CV: {best_score:.6f} with params={best_params}")


def local_refine(best_params, step_scales=(0.25, 0.5), seed=123):
    ap, al, sp, sl_min, sl_max, gp, gl = best_params
    candidates = set()
    candidates.add(best_params)

    for s in step_scales:
        for dap in (-s, s):
            candidates.add((max(0.1, ap + dap), al, sp, sl_min, sl_max, gp, gl))
        for dal in (-s, s):
            candidates.add((ap, max(0.1, al + dal), sp, sl_min, sl_max, gp, gl))
        for dsp in (-0.02 * s / 0.25, 0.02 * s / 0.25):
            candidates.add(
                (ap, al, float(np.clip(sp + dsp, 0.6, 1.0)), sl_min, sl_max, gp, gl)
            )
        for dmin in (-0.02 * s / 0.25, 0.02 * s / 0.25):
            candidates.add(
                (ap, al, sp, float(np.clip(sl_min + dmin, 0.6, 1.0)), sl_max, gp, gl)
            )
        for dmax in (-0.02 * s / 0.25, 0.02 * s / 0.25):
            candidates.add(
                (ap, al, sp, sl_min, float(np.clip(sl_max + dmax, 0.6, 1.0)), gp, gl)
            )
        for dgp in (-0.02 * s / 0.25, 0.02 * s / 0.25):
            candidates.add(
                (ap, al, sp, sl_min, sl_max, float(np.clip(gp + dgp, 0.75, 1.25)), gl)
            )
        for dgl in (-0.02 * s / 0.25, 0.02 * s / 0.25):
            candidates.add(
                (ap, al, sp, sl_min, sl_max, gp, float(np.clip(gl + dgl, 0.75, 1.25)))
            )

    cand_list = [c for c in candidates if c[4] >= c[3]]

    best_sc = cv_score_for_params(y, best_params, k=5, repeats=6, seed=seed)
    best_p = best_params
    for params in cand_list:
        sc = cv_score_for_params(y, params, k=5, repeats=6, seed=seed)
        if sc < best_sc:
            best_sc, best_p = sc, params
    return best_p, best_sc


best_params, best_score = local_refine(best_params, step_scales=(0.25, 0.5), seed=123)
print(f"==> After local refine best CV: {best_score:.6f} with params={best_params}")

prevalence_vec = make_prevalence_from_y(y, best_params)

p_levels = np.clip(prevalence_vec[1:], EPS, 1.0 - EPS)
p_or = 1.0 - float(np.prod(1.0 - p_levels))
prevalence_vec[0] = float(np.clip(max(prevalence_vec[0], p_or), EPS, 1.0 - EPS))

prevalence = {targets[i]: float(prevalence_vec[i]) for i in range(len(targets))}

print("==> Final calibrated prevalences (constant-by-class):")
for k in targets:
    print(f"    {k}: {prevalence[k]:.6f}")

fractured = np.empty(len(test_df), dtype=np.float32)
pred_types = test_df["prediction_type"].values
for i, pt in enumerate(pred_types):
    fractured[i] = prevalence.get(pt, prevalence["patient_overall"])

sub = pd.DataFrame({"row_id": test_df["row_id"].values, "fractured": fractured})

sub["fractured"] = sub["fractured"].clip(EPS, 1.0 - EPS)

sub.to_csv(SAVE_CSV, index=False)
print(f"==> Wrote {SAVE_CSV} with shape {sub.shape}")
print(f"==> Finish using time: {time.time() - time_start:.2f} seconds")
print(sub.head(10))
