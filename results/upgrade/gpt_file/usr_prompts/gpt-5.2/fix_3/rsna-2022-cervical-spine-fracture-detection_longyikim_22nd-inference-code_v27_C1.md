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

0.4353907406920667

# 6. Current score

0.75581

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.7459) has done: 'I remove the notebook-style `pip install`/custom model dependencies (effdet/efficientunet/pylibjpeg) that are not available in your current environment and are causing the import crashes. To keep the pipeline end-to-end and submission-valid, I replace the broken model inference with a lightweight baseline that uses only `train.csv` label priors to generate probabilities, while preserving the same submission structure (joining on `StudyInstanceUID` and selecting by `prediction_type`). I also fix the undefined-name issues by consolidating imports and ensuring `df_patient_pred` is always created before submission assembly. Finally, I guarantee `submission.csv` is written with the exact required columns (`row_id, fractured`) and correct row order/length matching `test.csv`.'
- What this solution (achieved 0.75581) has done: 'Your current solution predicts the same constant priors for every study, which is a weak baseline and explains the high (worse) log loss. To move the score down toward the target with minimal logic changes, I keep the “label-prior baseline” approach but make it slightly study-specific by using the provided `train_bounding_boxes.csv` as a lightweight proxy feature: scans with more vertebrae boxes and larger total box area get a modestly higher predicted fracture probability. I calibrate this effect conservatively (small shift in logit space) and keep `patient_overall` as the max over C1–C7 to match the evaluation semantics. The output format, join logic, clipping, and `submission.csv` writing remain unchanged.'

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

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DATA_ROOT = "../input/rsna-2022-cervical-spine-fracture-detection"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_ROOT, "sample_submission.csv")
BBOX_CSV = os.path.join(DATA_ROOT, "train_bounding_boxes.csv")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(TEST_CSV), f"Missing: {TEST_CSV}"
assert os.path.exists(SAMPLE_SUB_CSV), f"Missing: {SAMPLE_SUB_CSV}"
assert os.path.exists(BBOX_CSV), f"Missing: {BBOX_CSV}"



## === cell 1
IMAGES_DIR = os.path.join(DATA_ROOT, "test_images")
TRAIN_IMAGES_PATH = os.path.join(DATA_ROOT, "train_images")
TEST_IMAGES_PATH = os.path.join(DATA_ROOT, "test_images")



## === cell 2
segmentation_checkpoint = (
    "../input/effdet-models/axial_segmentation_effseg_095521-epoch-51.pth"
)
axial_det_checkpoint1 = (
    "../input/effdet-models/axial_detection_effdet_151039-epoch-60.pth"
)



## === cell 3
df_train = pd.read_csv(TRAIN_CSV)
df_test = pd.read_csv(TEST_CSV)
df_bbox = pd.read_csv(BBOX_CSV)

target_cols = ["patient_overall"] + [f"C{i}" for i in range(1, 8)]
for c in ["StudyInstanceUID"] + target_cols:
    assert c in df_train.columns, f"train.csv missing column: {c}"
for c in ["row_id", "StudyInstanceUID", "prediction_type"]:
    assert c in df_test.columns, f"test.csv missing column: {c}"
for c in ["StudyInstanceUID", "x", "y", "width", "height", "slice_number"]:
    assert c in df_bbox.columns, f"train_bounding_boxes.csv missing column: {c}"

alpha = 1.0
priors = {}
n = len(df_train)
for c in target_cols:
    pos = float(df_train[c].sum())
    priors[c] = (pos + alpha) / (n + 2 * alpha)

priors["patient_overall"] = float(
    max([priors[f"C{i}"] for i in range(1, 8)] + [priors["patient_overall"]])
)

priors



## === cell 4

df_bbox = df_bbox.copy()
df_bbox["area"] = df_bbox["width"].astype(float) * df_bbox["height"].astype(float)

agg = (
    df_bbox.groupby("StudyInstanceUID")
    .agg(
        n_boxes=("area", "size"),
        area_sum=("area", "sum"),
        area_mean=("area", "mean"),
        n_slices=("slice_number", "nunique"),
    )
    .reset_index()
)

df_fit = df_train[["StudyInstanceUID", "patient_overall"]].merge(
    agg, on="StudyInstanceUID", how="left"
)
df_fit[["n_boxes", "area_sum", "area_mean", "n_slices"]] = df_fit[
    ["n_boxes", "area_sum", "area_mean", "n_slices"]
].fillna(0.0)


def _zscore(s: pd.Series) -> pd.Series:
    m = float(s.mean())
    sd = float(s.std(ddof=0))
    if sd < 1e-12:
        return s * 0.0
    return (s - m) / sd


X = pd.DataFrame(
    {
        "bias": 1.0,
        "n_boxes_z": _zscore(df_fit["n_boxes"].astype(float)),
        "area_sum_z": _zscore(np.log1p(df_fit["area_sum"].astype(float))),
        "n_slices_z": _zscore(df_fit["n_slices"].astype(float)),
    }
)

y = df_fit["patient_overall"].astype(float).values

XtX = X.values.T @ X.values
Xty = X.values.T @ y
beta = np.linalg.pinv(XtX) @ Xty


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


df_test_feat = pd.DataFrame({"StudyInstanceUID": df_test["StudyInstanceUID"].unique()})
df_test_feat = df_test_feat.merge(agg, on="StudyInstanceUID", how="left")
df_test_feat[["n_boxes", "area_sum", "area_mean", "n_slices"]] = df_test_feat[
    ["n_boxes", "area_sum", "area_mean", "n_slices"]
].fillna(0.0)

train_stats = {
    "n_boxes_mean": float(df_fit["n_boxes"].astype(float).mean()),
    "n_boxes_std": float(df_fit["n_boxes"].astype(float).std(ddof=0)),
    "area_sum_log_mean": float(np.log1p(df_fit["area_sum"].astype(float)).mean()),
    "area_sum_log_std": float(np.log1p(df_fit["area_sum"].astype(float)).std(ddof=0)),
    "n_slices_mean": float(df_fit["n_slices"].astype(float).mean()),
    "n_slices_std": float(df_fit["n_slices"].astype(float).std(ddof=0)),
}


def _z_using_stats(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    if std < 1e-12:
        return x * 0.0
    return (x - mean) / std


X_test = pd.DataFrame(
    {
        "bias": 1.0,
        "n_boxes_z": _z_using_stats(
            df_test_feat["n_boxes"].astype(float).values,
            train_stats["n_boxes_mean"],
            train_stats["n_boxes_std"],
        ),
        "area_sum_z": _z_using_stats(
            np.log1p(df_test_feat["area_sum"].astype(float)).values,
            train_stats["area_sum_log_mean"],
            train_stats["area_sum_log_std"],
        ),
        "n_slices_z": _z_using_stats(
            df_test_feat["n_slices"].astype(float).values,
            train_stats["n_slices_mean"],
            train_stats["n_slices_std"],
        ),
    }
)

pred_lin = X_test.values @ beta  # in probability space (approx)
base = np.full(len(df_test_feat), priors["patient_overall"], dtype=float)
adj = _logit(np.clip(pred_lin, 1e-4, 1 - 1e-4)) - _logit(base)

adj = np.clip(adj, -0.35, 0.35) * 0.35

patient_overall_pred = _sigmoid(_logit(base) + adj)

unique_studies = df_test["StudyInstanceUID"].unique()
df_patient_pred = pd.DataFrame({"StudyInstanceUID": unique_studies})
df_patient_pred = df_patient_pred.merge(
    df_test_feat[["StudyInstanceUID"]], on="StudyInstanceUID", how="left"
)

uid_to_po = dict(zip(df_test_feat["StudyInstanceUID"].values, patient_overall_pred))
df_patient_pred["patient_overall"] = df_patient_pred["StudyInstanceUID"].map(uid_to_po)

delta_po = df_patient_pred["patient_overall"].astype(float) - float(
    priors["patient_overall"]
)
for i in range(1, 8):
    c = f"C{i}"
    df_patient_pred[c] = float(priors[c]) + 0.25 * delta_po

df_patient_pred = df_patient_pred.set_index("StudyInstanceUID")

min_clip_value = 0.005
max_clip_value = 0.005
df_patient_pred[target_cols] = df_patient_pred[target_cols].clip(
    lower=min_clip_value, upper=1 - max_clip_value
)

df_patient_pred["patient_overall"] = df_patient_pred[
    [f"C{i}" for i in range(1, 8)]
].max(axis=1)
df_patient_pred = df_patient_pred[["patient_overall"] + [f"C{i}" for i in range(1, 8)]]
df_patient_pred.head()



## === cell 5
df_sub = df_test.copy()
df_sub = df_sub.set_index("StudyInstanceUID").join(df_patient_pred, how="left")

for c in target_cols:
    df_sub[c] = df_sub[c].fillna(priors[c])

df_sub["fractured"] = df_sub.apply(lambda r: float(r[r["prediction_type"]]), axis=1)

sub = df_sub[["row_id", "fractured"]].copy()
assert len(sub) == len(df_test), "Submission row count mismatch vs test.csv"
assert sub["row_id"].isna().sum() == 0, "Missing row_id in submission"
assert sub["fractured"].isna().sum() == 0, "Missing predictions in submission"
sub.head()



## === cell 6
sub.to_csv("submission.csv", index=False)

sample = pd.read_csv(SAMPLE_SUB_CSV)
assert list(sample.columns) == [
    "row_id",
    "fractured",
], "Sample submission column mismatch"
assert len(sample) == len(sub), "submission.csv must match sample_submission row count"
print("Wrote submission.csv with shape:", sub.shape)
print(sub.describe())
