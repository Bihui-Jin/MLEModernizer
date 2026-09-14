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

0.5603

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5639) has done: 'I remove the notebook-style `pip install` cell and instead make the script run in this Kaggle environment using only already-available packages, since your environment does not include the `../input/effdet-models` dependency folder and thus cannot import `pylibjpeg/effdet/efficientunet`. To keep the pipeline end-to-end and produce a valid `submission.csv`, I replace the unavailable segmentation+detection inference with a minimal, deterministic baseline that learns per-label priors from `train.csv` and outputs calibrated probabilities for each `prediction_type`. This fixes the immediate runtime/import errors and guarantees correct submission formatting/alignment with `test.csv`. The resulting score won’t match your original model’s intent (because the required model files aren’t available here), but it be a valid submission and typically better than naive 0/1 guessing under weighted log loss.'
- What this solution (achieved 0.5607) has done: 'We keep your prior-based baseline (core logic) but reduce the log-loss by applying a tiny amount of shrinkage/smoothing to the per-label priors and by recomputing `patient_overall` from the vertebrae priors (since it is logically the “any” label and is heavily weighted). This preserves the same “learn priors from train.csv and map by prediction_type” approach while improving calibration for the most important row type. We also clip probabilities slightly tighter to avoid extreme log-loss penalties without changing semantics. The output format, alignment with `test.csv`, and `submission.csv` writing remain unchanged.'
- What this solution (achieved 0.55897) has done: 'We keep your “learn per-label priors from train.csv then map by prediction_type” core logic, but tune it slightly toward the metric by introducing label-specific smoothing and clipping (patient_overall is weighted more, so it benefits from stronger regularization away from extreme probabilities). We also compute patient_overall from the smoothed vertebra priors as you already do, but make its smoothing independent of the C-level smoothing to better control calibration. These are minimal, deterministic changes that preserve the same semantics (constant-per-label probabilities) while typically reducing weighted log loss from overconfident priors. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.5603) has done: 'We keep your constant-per-label prior baseline, but calibrate it slightly better for weighted log loss by using a logit-space shrinkage toward a global base rate (this reduces overconfident priors without changing the overall approach). We also compute `patient_overall` in a more stable way from the shrunken C-level priors (log-sum for `p_none`) and then blend with the empirical patient rate as you already do, since that label is heavily weighted. Finally, we add tiny label-specific clipping (same semantics as before) to avoid extreme penalties while preserving determinism and the exact submission format/alignment.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

np.random.seed(0)

DATA_DIR = "../input/rsna-2022-cervical-spine-fracture-detection"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_DIR, "sample_submission.csv")

print("Using DATA_DIR:", DATA_DIR)
print("Train exists:", os.path.exists(TRAIN_CSV))
print("Test exists:", os.path.exists(TEST_CSV))
print("Sample submission exists:", os.path.exists(SAMPLE_SUB_CSV))



## === cell 1
df_train = pd.read_csv(TRAIN_CSV)
df_test = pd.read_csv(TEST_CSV)
df_sample = pd.read_csv(SAMPLE_SUB_CSV)

expected_train_cols = ["StudyInstanceUID", "patient_overall"] + [
    f"C{i}" for i in range(1, 8)
]
missing_train = [c for c in expected_train_cols if c not in df_train.columns]
if missing_train:
    raise ValueError(f"train.csv missing columns: {missing_train}")

expected_test_cols = ["StudyInstanceUID", "prediction_type", "row_id"]
missing_test = [c for c in expected_test_cols if c not in df_test.columns]
if missing_test:
    raise ValueError(f"test.csv missing columns: {missing_test}")

print("train shape:", df_train.shape)
print("test shape:", df_test.shape)
print("sample_submission shape:", df_sample.shape)
df_test.head()



## === cell 2
label_cols = ["patient_overall"] + [f"C{i}" for i in range(1, 8)]
c_cols = [f"C{i}" for i in range(1, 8)]
n = len(df_train)


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    return float(np.log(p / (1.0 - p)))


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


alpha_c = 1.0
alpha_patient = 2.0

total_pos_c = float(df_train[c_cols].to_numpy().sum())
total_count_c = float(n * len(c_cols))
p_global_c = (total_pos_c + 1.0) / (total_count_c + 2.0)  # mild smoothing for stability

k_c = 0.35

priors = {}
for col in c_cols:
    s = float(df_train[col].sum())
    p_emp = (s + alpha_c) / (n + 2.0 * alpha_c)
    p_shrunk = _sigmoid((1.0 - k_c) * _logit(p_emp) + k_c * _logit(p_global_c))
    priors[col] = p_shrunk

log_p_none = 0.0
for c in c_cols:
    log_p_none += float(np.log(np.clip(1.0 - priors[c], 1e-12, 1.0)))
p_any_from_c = float(1.0 - np.exp(log_p_none))

s_patient = float(df_train["patient_overall"].sum())
p_patient_emp = (s_patient + alpha_patient) / (n + 2.0 * alpha_patient)

blend = 0.70
priors["patient_overall"] = blend * p_any_from_c + (1.0 - blend) * p_patient_emp

EPS_C = 3e-3
EPS_PATIENT = 7e-3
for k in list(priors.keys()):
    eps = EPS_PATIENT if k == "patient_overall" else EPS_C
    priors[k] = float(np.clip(priors[k], eps, 1.0 - eps))

print("Computed priors:", priors)



## === cell 3
df_sub = df_test[["row_id", "prediction_type"]].copy()

valid_types = set(label_cols)
bad_types = sorted(set(df_sub["prediction_type"].unique()) - valid_types)
if bad_types:
    raise ValueError(
        f"Unexpected prediction_type values found in test.csv: {bad_types}"
    )

df_sub["fractured"] = df_sub["prediction_type"].map(priors).astype(np.float32)
df_sub = df_sub[["row_id", "fractured"]]

if list(df_sub.columns) != ["row_id", "fractured"]:
    raise ValueError(
        "Submission columns do not match required format ['row_id','fractured']."
    )

if df_sub["row_id"].isna().any() or df_sub["fractured"].isna().any():
    raise ValueError("Found NA values in submission.")

print(df_sub.head(10))
print("submission rows:", len(df_sub))



## === cell 4
out_path = "submission.csv"
df_sub.to_csv(out_path, index=False)

assert os.path.exists(out_path), "submission.csv was not created."
check = pd.read_csv(out_path)
assert list(check.columns) == ["row_id", "fractured"], "Invalid submission header."
assert len(check) == len(df_test), "Submission row count mismatch."

print("Wrote:", out_path)
print(check.head())
