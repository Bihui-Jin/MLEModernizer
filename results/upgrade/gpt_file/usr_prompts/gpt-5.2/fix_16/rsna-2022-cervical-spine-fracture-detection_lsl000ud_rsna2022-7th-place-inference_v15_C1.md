# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

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

time_start = time.time()

train_df = pd.read_csv(TRAIN_CSV_PATH)
test_df = pd.read_csv(TEST_CSV_PATH)

targets = ["patient_overall"] + [f"C{i}" for i in range(1, 8)]
missing_targets = [c for c in targets if c not in train_df.columns]
if missing_targets:
    raise ValueError(f"train.csv missing expected target columns: {missing_targets}")

y = train_df[targets].fillna(0).astype(np.float32).values  # (N, 8)
N = y.shape[0]
ALL_IDX = np.arange(N, dtype=np.int32)  # reused everywhere

W_PATIENT = 7.0
W_LEVEL = 1.0
label_weights = np.array([W_PATIENT] + [W_LEVEL] * 7, dtype=np.float32)

EPS = 2e-6  # keep consistent with previous runs

test_type_counts = (
    test_df["prediction_type"]
    .value_counts()
    .reindex(targets)
    .fillna(0)
    .astype(np.float32)
    .values
)
if test_type_counts.sum() <= 0:
    test_type_counts = np.ones(len(targets), dtype=np.float32)

type_row_weights = (test_type_counts / test_type_counts.mean()).astype(np.float32)
col_w = (label_weights * type_row_weights).astype(np.float32)  # (8,)


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


def patient_from_levels(p_levels, method_params):
    k, b, mix = method_params
    m = float(np.max(p_levels))
    q = 1.0 / (1.0 + np.exp(-k * (m - b)))
    return float(q), m


def predict_constants_for_ytrain(y_train, base_params, map_params, po_gamma):
    """Derive final 8 constant probabilities from a y_train subset (same semantics as submission)."""
    prevalence_vec = make_prevalence_from_y(y_train, base_params)

    q, m = patient_from_levels(prevalence_vec[1:], map_params)
    k_sig, b_sig, mix = map_params
    p0_base = float(prevalence_vec[0])
    p0 = (1.0 - mix) * p0_base + mix * q
    p0 = float(power_calibrate_vec(p0, po_gamma))
    p0 = float(np.clip(max(p0, m), EPS, 1.0 - EPS))
    prevalence_vec[0] = p0
    return prevalence_vec.astype(np.float32)


def weighted_rowwise_logloss(y_true, p_const):
    p = np.clip(p_const, EPS, 1.0 - EPS).astype(np.float32)  # (8,)
    n = y_true.shape[0]
    sum_y = y_true.sum(axis=0)  # (8,)
    sum_1my = n - sum_y
    logp = np.log(p)
    log1mp = np.log1p(-p)  # stable log(1-p)
    col_loss = -(sum_y * logp + sum_1my * log1mp) / n  # (8,)
    return float((col_loss * col_w).mean())




## === cell 1
def _precompute_repeated_stratified_splits_pairs(
    y_all, n_splits=5, n_repeats=20, seed=1337
):
    y_strat = y_all[:, 0].astype(np.int8)
    rng = np.random.RandomState(seed)

    idx_pos0 = np.where(y_strat == 1)[0].astype(np.int32)
    idx_neg0 = np.where(y_strat == 0)[0].astype(np.int32)

    pairs = []
    for _ in range(n_repeats):
        idx_pos = idx_pos0.copy()
        idx_neg = idx_neg0.copy()
        rng.shuffle(idx_pos)
        rng.shuffle(idx_neg)
        pos_folds = np.array_split(idx_pos, n_splits)
        neg_folds = np.array_split(idx_neg, n_splits)

        for k in range(n_splits):
            val_idx = np.concatenate([pos_folds[k], neg_folds[k]]).astype(
                np.int32, copy=False
            )

            if n_splits == 1:
                train_idx = val_idx[:0]
            else:
                pos_train = np.concatenate(
                    [pos_folds[j] for j in range(n_splits) if j != k]
                ).astype(np.int32, copy=False)
                neg_train = np.concatenate(
                    [neg_folds[j] for j in range(n_splits) if j != k]
                ).astype(np.int32, copy=False)
                train_idx = np.concatenate([pos_train, neg_train]).astype(
                    np.int32, copy=False
                )

            pairs.append((train_idx, val_idx))
    return pairs


def _precompute_fold_stats(y_all, fold_pairs):
    stats = []
    for tr_idx, va_idx in fold_pairs:
        y_tr = y_all[tr_idx]
        y_va = y_all[va_idx]
        stats.append(
            (
                tr_idx,
                va_idx,
                y_tr.shape[0],
                y_tr.sum(axis=0).astype(np.float32, copy=False),
                y_va.shape[0],
                y_va.sum(axis=0).astype(np.float32, copy=False),
            )
        )
    return stats


def make_prevalence_from_stats(n_tr, pos, params):
    (
        alpha_patient,
        alpha_level,
        shrink_patient,
        sl_min,
        sl_max,
        gamma_patient,
        gamma_level,
    ) = params

    a = np.array([alpha_patient] + [alpha_level] * 7, dtype=np.float32)
    p = (pos + a) / (n_tr + 2.0 * a)
    p = np.clip(p, 1e-4, 1.0 - 1e-4)

    p0 = float(p[0])
    p1max = float(np.max(p[1:]))
    if p0 < p1max:
        p[0] = p1max

    raw_prev = pos / max(1.0, float(n_tr))
    s = np.empty_like(p)
    s[0] = shrink_patient
    t = np.minimum(raw_prev[1:] / 0.15, 1.0)
    s[1:] = sl_min + (sl_max - sl_min) * t

    p = 0.5 + s * (p - 0.5)
    p = np.clip(p, 1e-4, 1.0 - 1e-4)

    p[0] = power_calibrate_vec(p[0], gamma_patient)
    p[1:] = power_calibrate_vec(p[1:], gamma_level)

    p0 = float(p[0])
    p1max = float(np.max(p[1:]))
    if p0 < p1max:
        p[0] = p1max
    p[0] = float(np.clip(p[0], EPS, 1.0 - EPS))
    return p.astype(np.float32, copy=False)


def predict_constants_for_stats(n_tr, sum_y_tr, base_params, map_params, po_gamma):
    prevalence_vec = make_prevalence_from_stats(n_tr, sum_y_tr, base_params)

    q, m = patient_from_levels(prevalence_vec[1:], map_params)
    k_sig, b_sig, mix = map_params
    p0_base = float(prevalence_vec[0])
    p0 = (1.0 - mix) * p0_base + mix * q
    p0 = float(power_calibrate_vec(p0, po_gamma))
    p0 = float(np.clip(max(p0, m), EPS, 1.0 - EPS))
    prevalence_vec[0] = p0
    return prevalence_vec.astype(np.float32, copy=False)


def weighted_rowwise_logloss_from_stats(n, sum_y, p_const):
    p = np.clip(p_const, EPS, 1.0 - EPS).astype(np.float32, copy=False)  # (8,)
    sum_1my = n - sum_y
    logp = np.log(p)
    log1mp = np.log1p(-p)
    col_loss = -(sum_y * logp + sum_1my * log1mp) / n  # (8,)
    return float((col_loss * col_w).mean())


_SCORE_CACHE = {}


def repeated_stratified_cv_score(
    y_all, base_params, map_params, po_gamma, n_splits=5, n_repeats=20, seed=1337
):
    key = (base_params, map_params, float(po_gamma), n_splits, n_repeats, seed)
    cached = _SCORE_CACHE.get(key)
    if cached is not None:
        return cached

    y_strat = y_all[:, 0].astype(np.int8)
    idx_pos = np.where(y_strat == 1)[0]
    idx_neg = np.where(y_strat == 0)[0]
    if len(idx_pos) < n_splits or len(idx_neg) < n_splits:
        p_const = predict_constants_for_ytrain(y_all, base_params, map_params, po_gamma)
        sc = weighted_rowwise_logloss(y_all, p_const)
        _SCORE_CACHE[key] = sc
        return sc

    cache_key = (n_splits, n_repeats, seed, y_all.shape[0], int(y_strat.sum()))
    global _FOLD_CACHE
    try:
        _FOLD_CACHE
    except NameError:
        _FOLD_CACHE = {}

    fold_stats = _FOLD_CACHE.get(cache_key)
    if fold_stats is None:
        fold_pairs = _precompute_repeated_stratified_splits_pairs(
            y_all, n_splits=n_splits, n_repeats=n_repeats, seed=seed
        )
        fold_stats = _precompute_fold_stats(y_all, fold_pairs)
        _FOLD_CACHE[cache_key] = fold_stats

    scores_sum = 0.0
    scores_n = 0
    for _, _, n_tr, sum_y_tr, n_va, sum_y_va in fold_stats:
        p_const = predict_constants_for_stats(
            n_tr, sum_y_tr, base_params, map_params, po_gamma
        )
        scores_sum += weighted_rowwise_logloss_from_stats(n_va, sum_y_va, p_const)
        scores_n += 1

    sc = float(scores_sum / scores_n)
    _SCORE_CACHE[key] = sc
    return sc


alpha_patient_grid = [0.5, 1.0, 2.0, 4.0, 6.0]
alpha_level_grid = [0.5, 1.0, 1.25, 2.0, 3.0, 4.0]
shrink_patient_grid = [0.90, 0.95, 0.98, 1.00]
sl_min_grid = [0.70, 0.76, 0.78, 0.82, 0.85]
sl_max_grid = [0.86, 0.90, 0.94, 0.97, 0.99]
gamma_patient_grid = [0.90, 0.95, 0.98, 1.00, 1.05]
gamma_level_grid = [0.85, 0.90, 0.92, 0.95, 1.00]

k_grid = [6.0, 10.0, 14.0]  # slope
b_grid = [0.04, 0.06, 0.08, 0.10]  # midpoint
mix_grid = [0.0, 0.25, 0.5, 0.75]  # mix for patient_overall mapping

po_gamma_grid = [0.90, 0.95, 1.00, 1.05, 1.10]

best_base_params = None
best_map_params = None
best_po_gamma = 1.0
best_score = float("inf")

baseline_base_params = (2.0, 1.25, 0.95, 0.78, 0.90, 0.98, 0.92)
baseline_map_params = (10.0, 0.08, 0.0)  # mix=0 => original behavior
baseline_po_gamma = 1.0

baseline_cv = repeated_stratified_cv_score(
    y,
    baseline_base_params,
    baseline_map_params,
    baseline_po_gamma,
    n_splits=5,
    n_repeats=20,
    seed=1337,
)

best_base_params, best_map_params, best_po_gamma, best_score = (
    baseline_base_params,
    baseline_map_params,
    baseline_po_gamma,
    baseline_cv,
)

print(
    f"==> Baseline repeated stratified CV weighted logloss: {baseline_cv:.6f} with "
    f"base_params={baseline_base_params}, map_params={baseline_map_params}, po_gamma={baseline_po_gamma}"
)

stage1_sl = (0.95, 0.78, 0.90)  # shrink_patient, sl_min, sl_max
for ap in alpha_patient_grid:
    for al in alpha_level_grid:
        for gp in gamma_patient_grid:
            for gl in gamma_level_grid:
                base_params = (ap, al, stage1_sl[0], stage1_sl[1], stage1_sl[2], gp, gl)
                for kk in k_grid:
                    for bb in b_grid:
                        for mix in mix_grid:
                            map_params = (kk, bb, mix)
                            for po_g in po_gamma_grid:
                                sc = repeated_stratified_cv_score(
                                    y,
                                    base_params,
                                    map_params,
                                    po_g,
                                    n_splits=5,
                                    n_repeats=20,
                                    seed=1337,
                                )
                                if sc < best_score:
                                    best_score = sc
                                    best_base_params = base_params
                                    best_map_params = map_params
                                    best_po_gamma = po_g

print(
    f"==> After stage1 best repeated stratified CV: {best_score:.6f} with "
    f"base_params={best_base_params}, map_params={best_map_params}, po_gamma={best_po_gamma}"
)

ap, al, _, _, _, gp, gl = best_base_params
for sp in shrink_patient_grid:
    for sl_min in sl_min_grid:
        for sl_max in sl_max_grid:
            if sl_max < sl_min:
                continue
            base_params = (ap, al, sp, sl_min, sl_max, gp, gl)
            sc = repeated_stratified_cv_score(
                y,
                base_params,
                best_map_params,
                best_po_gamma,
                n_splits=5,
                n_repeats=20,
                seed=1337,
            )
            if sc < best_score:
                best_score = sc
                best_base_params = base_params

for po_g in po_gamma_grid:
    sc = repeated_stratified_cv_score(
        y, best_base_params, best_map_params, po_g, n_splits=5, n_repeats=20, seed=1337
    )
    if sc < best_score:
        best_score = sc
        best_po_gamma = po_g

print(
    f"==> After stage2 best repeated stratified CV: {best_score:.6f} with "
    f"base_params={best_base_params}, map_params={best_map_params}, po_gamma={best_po_gamma}"
)



## === cell 2
prevalence_vec = predict_constants_for_ytrain(
    y, best_base_params, best_map_params, best_po_gamma
)
prevalence = {targets[i]: float(prevalence_vec[i]) for i in range(len(targets))}

k_sig, b_sig, mix = best_map_params
print("==> Final calibrated prevalences (constant-by-class):")
for k in targets:
    print(f"    {k}: {prevalence[k]:.6f}")
print(
    f"==> Tuned patient_overall mapping params: k={k_sig}, b={b_sig}, mix={mix}, po_gamma={best_po_gamma}"
)

pred_types = test_df["prediction_type"].to_numpy()
fractured = np.full(
    pred_types.shape[0], prevalence["patient_overall"], dtype=np.float32
)
for t in targets[1:]:
    mask = pred_types == t
    if mask.any():
        fractured[mask] = np.float32(prevalence[t])

sub = pd.DataFrame({"row_id": test_df["row_id"].values, "fractured": fractured})
sub["fractured"] = sub["fractured"].clip(EPS, 1.0 - EPS)

sub.to_csv(SAVE_CSV, index=False)
print(f"==> Wrote {SAVE_CSV} with shape {sub.shape}")
print(f"==> Finish using time: {time.time() - time_start:.2f} seconds")
print(sub.head(10))
