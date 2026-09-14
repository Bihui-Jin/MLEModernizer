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

0.58292

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5639) has done: 'I fix the immediate import/path failure by removing the unavailable `../input/src2-final/src` dependency and replacing it with lightweight, in-notebook fallbacks that let the pipeline run end-to-end in this Kaggle environment. Because the original 3-stage nnUNet/model files are not provided in the listed data paths, I add safe checks that avoid crashing when those checkpoints/plans are missing, and instead generate a valid baseline submission using train priors (a legitimate, score-reasonable fallback under logloss). I also ensure the submission matches `test.csv` ordering and contains every `row_id` exactly once, written to `submission.csv`. These changes are execution-unblocking and should yield a finite logloss (better than random/invalid), moving toward the target where previously no score was possible.'
- What this solution (achieved 0.58292) has done: 'Your current baseline uses raw train prevalences, which is typically miscalibrated for weighted logloss (especially the heavily-weighted `patient_overall`) and can be improved with a minimal, metric-aligned calibration step. I keep your “priors-only” core logic, but (1) calibrate the per-label probabilities using a single temperature scaling on the logit scale learned from the training labels (no new model, no extra data), and (2) enforce a consistent relationship between vertebrae and `patient_overall` by converting vertebra probabilities into an implied any-fracture probability `1 - Π(1-p_Ci)` and blending it with the calibrated `patient_overall`. These changes are small, deterministic, and should reduce logloss substantially from 0.5639 toward your 0.2669 target while still producing the same valid submission format. Paths/I/O stay the same and the script remains fast.'

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


def fit_temperature_scaling(train_csv: str, base_priors: dict) -> dict:
    df = pd.read_csv(train_csv)

    temps = {}
    for c in TARGET_COLS:
        if c not in df.columns:
            temps[c] = 1.0
            continue

        y = df[c].values.astype(np.float64)
        p0 = float(np.clip(base_priors.get(c, 0.01), 1e-4, 1 - 1e-4))
        z0 = float(_logit(p0))

        pos_rate = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
        w_pos = 0.5 / pos_rate
        w_neg = 0.5 / (1.0 - pos_rate)
        w = np.where(y > 0.5, w_pos, w_neg).astype(np.float64)

        Ts = np.array(
            [0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.25, 1.50, 1.75, 2.00],
            dtype=np.float64,
        )
        best_T = 1.0
        best_loss = np.inf
        for T in Ts:
            p = float(_sigmoid(z0 / T))
            p = np.clip(p, 1e-6, 1 - 1e-6)
            loss = -np.mean(w * (y * np.log(p) + (1 - y) * np.log(1 - p)))
            if loss < best_loss:
                best_loss = loss
                best_T = float(T)

        temps[c] = best_T

    return temps


def apply_temperature(p: float, T: float) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    z = _logit(p)
    return float(np.clip(_sigmoid(z / float(T)), 1e-6, 1 - 1e-6))


class FractureDetector:
    def __init__(self, priors: dict, temps: dict):
        self.priors = priors
        self.temps = temps
        self.results = {}

    def predict(self, list_test_files, num_thread=4, output_dir=None):
        for p in list_test_files:
            case_id = os.path.basename(p.rstrip("/"))

            out = np.zeros(8, dtype=np.float64)
            out[0] = apply_temperature(
                self.priors["patient_overall"], self.temps.get("patient_overall", 1.0)
            )
            for i, c in enumerate(LEVEL_COLS, start=1):
                out[i] = apply_temperature(self.priors[c], self.temps.get(c, 1.0))

            p_any_from_levels = 1.0 - float(
                np.prod(1.0 - np.clip(out[1:], 1e-6, 1 - 1e-6))
            )
            alpha = 0.70  # keep mostly patient_overall prior, but incorporate levels consistently
            out[0] = float(
                np.clip(
                    alpha * out[0] + (1.0 - alpha) * p_any_from_levels, 1e-6, 1 - 1e-6
                )
            )

            self.results[case_id] = out.astype(np.float32)




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

unique_studies = test_df["StudyInstanceUID"].unique().tolist()
if TEST_IMG_DIR is None:
    list_DICOM_dirs = unique_studies[:]  # dummy identifiers
else:
    list_DICOM_dirs = [os.path.join(TEST_IMG_DIR, uid) for uid in unique_studies]

print("Num test studies:", len(unique_studies))
print("TEST_IMG_DIR:", TEST_IMG_DIR)

detector = FractureDetector(priors=train_priors, temps=temps)
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
        v = float(np.clip(train_priors.get(ptype, 0.01), 1e-4, 1 - 1e-4))
    preds.append(v)

sub = pd.DataFrame(
    {"row_id": test_df["row_id"].values, "fractured": np.clip(preds, 1e-4, 1 - 1e-4)}
)
SAVE_CSV = "submission.csv"
sub.to_csv(SAVE_CSV, index=False)

print("Wrote:", SAVE_CSV, "rows:", len(sub), "missing_mapped:", missing)
print("Time:", time.time() - time_start, "sec")
gc.collect()
