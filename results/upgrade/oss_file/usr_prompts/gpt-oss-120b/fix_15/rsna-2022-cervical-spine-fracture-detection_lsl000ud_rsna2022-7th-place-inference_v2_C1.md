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

0.3135776582203192

# 6. Current score

0.56407

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5639) has done: 'The fix adds the missing imports, removes the unused heavy inference code that relied on undefined functions/classes, and rewrites the prediction step to use a simple baseline derived from the training data. This ensures the script runs without errors and writes a correctly‑formatted `submission.csv` containing a prediction for every row in the test set, moving the solution toward the target score.'
- What this solution (achieved 0.56405) has done: 'We replace the naïve overall‑mean baseline with class‑specific smoothed probabilities that use the provided α/β parameters (a simple Beta‑prior smoothing). This yields more realistic fracture probabilities for each vertebra and for the patient‑overall label, which should lower the weighted log‑loss and move the score closer to the target. The rest of the pipeline (reading CSVs, building the submission DataFrame, and writing the file) remains unchanged.'
- What this solution (achieved 0.56033) has done: 'We replace the independent patient‑overall probability with one derived from the vertebra probabilities (1 ‑ ∏(1‑p_i)). This respects the logical relationship that a patient is overall positive if any vertebra is fractured, and it usually yields a better calibrated “any‑fracture” estimate, moving the log‑loss closer to the target. The rest of the pipeline stays unchanged.'
- What this solution (achieved 0.55892) has done: 'The adjustment keeps the same baseline logic but improves the “patient_overall” estimate by blending the derived probability (from vertebra‑level predictions) with the direct prevalence observed in the training data. This calibrated overall probability typically aligns better with the true label distribution, lowering the weighted log‑loss and moving the score closer to the target.'
- What this solution (achieved 0.56391) has done: 'I add a lightweight validation step that tries a few values for the weight `w_derived` used to blend the derived “any‑fracture” probability with the direct prevalence.  
For each candidate weight we compute a simple binary log‑loss on a hold‑out split of the training data and pick the weight that gives the lowest loss.  
This small calibration tweak keeps the original baseline (smoothing, derived overall, clipping) unchanged while moving the patient‑overall predictions toward a value that better matches the true distribution, which should reduce the overall weighted log‑loss and bring the score closer to the target.'
- What this solution (achieved 0.56391) has done: 'The update keeps the same baseline approach but selects the blending weight for the “patient_overall” probability using a weighted log‑loss that reflects the higher importance of the overall label in the competition metric. By giving the overall label a larger weight during validation, the chosen weight `best_w` better balances the more‑penalised rows, which should lower the final weighted log‑loss and move the score closer to the target.'
- What this solution (achieved 0.5639) has done: 'We replace the vertebra‑level probabilities with the direct prevalence from the training set (which is a more faithful estimate than the heavily smoothed version) and give the patient_overall label a larger weight during validation when choosing the blending factor. This keeps the same overall structure, only adjusts the probability source and the validation weighting, and still writes a correctly formatted submission.csv.'
- What this solution (achieved 0.56405) has done: 'I added a simple Beta‑prior smoothing step for the vertebra‑level and overall fracture probabilities.  Using the provided α/β parameters we compute posterior means (pos + α)/(N + α + β) instead of raw prevalences, then keep the existing validation‑based blending of derived and direct overall probabilities.  This gives better‑calibrated scores and is expected to lower the weighted log‑loss toward the target while preserving the original pipeline.'
- What this solution (achieved 0.56407) has done: 'The update keeps the original smoothing and blending logic for the overall fracture probability, but adds a simple conditional calibration for each vertebra: each vertebra probability is now a mixture of its prevalence when the patient is overall positive and when the patient is overall negative, weighted by the blended overall probability. This respects the logical relationship between “any fracture” and individual levels, improves calibration, and therefore moves the weighted log‑loss closer to the target while preserving the core pipeline.'
- What this solution (achieved 0.56407) has done: 'The update keeps the existing baseline logic but adds a lightweight calibration step for the vertebra‑level probabilities. After selecting the best blending weight for the overall “any‑fracture” probability, we now also search for an optimal blend between the original smoothed vertebra prevalences and the conditional vertebra probabilities derived from the overall estimate. This small, validation‑driven adjustment improves calibration of the C1‑C7 predictions, lowering the weighted log‑loss and moving the score closer to the target while preserving the core pipeline.'
- What this solution (achieved 0.56407) has done: 'The update adds a lightweight temperature‑scaling calibration step that searches a small set of temperature values on a validation split and picks the one that gives the lowest weighted log‑loss.  The chosen temperature is then applied to the final probabilities before writing the submission, providing a modest but targeted improvement toward the target score without altering the core baseline logic.'

# 9. Code solution

## === cell 0
class FractureDetector:
    def __init__(self, predictor_stage1, predictor_stage2, extend_roi=(5.0, 5.0, 5.0)):
        self.predictor_stage1 = predictor_stage1
        self.predictor_stage2 = predictor_stage2
        self.extend_roi = extend_roi

        self.params = {
            "alpha": [0.055, 0.044, 0.052, 0.05, 0.07, 0.077, 0.09, 0.024],
            "beta": [0.475, 0.34, 0.37, 0.38, 0.31, 0.35, 0.38, 0.36],
            "min_score": [0.116, 0.075, 0.015, 0.015, 0.01, 0.02, 0.032, 0.048],
            "max_score": [0.99, 0.999, 0.993, 0.99, 1.0, 0.943, 0.997, 0.999],
        }

        self.results = {}




## === cell 1
import os
import pandas as pd
import numpy as np

DATA_DIR = "../input/rsna-2022-cervical-spine-fracture-detection/test_images"
TEST_CSV_PATH = "../input/rsna-2022-cervical-spine-fracture-detection/test.csv"
TRAIN_CSV_PATH = "../input/rsna-2022-cervical-spine-fracture-detection/train.csv"
SAVE_CSV = "submission.csv"

train_df = pd.read_csv(TRAIN_CSV_PATH)

alpha = np.array(
    [0.055, 0.044, 0.052, 0.05, 0.07, 0.077, 0.09, 0.024], dtype=np.float32
)
beta = np.array([0.475, 0.34, 0.37, 0.38, 0.31, 0.35, 0.38, 0.36], dtype=np.float32)

n_samples = len(train_df)

pos_counts = np.concatenate(
    (
        [train_df["patient_overall"].sum()],
        [train_df[f"C{i}"].sum() for i in range(1, 8)],
    )
).astype(np.float32)

smoothed_probs = (pos_counts + alpha) / (n_samples + alpha + beta)

vertebra_probs_base = smoothed_probs[1:]  # shape (7,)
direct_overall = smoothed_probs[0]  # smoothed overall prevalence

derived_overall = 1.0 - np.prod(1.0 - vertebra_probs_base)


def binary_log_loss(y, p):
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


eps = 1e-7
rng = np.random.RandomState(42)
indices = np.arange(n_samples)
rng.shuffle(indices)
cut = int(0.8 * n_samples)
train_idx, val_idx = indices[:cut], indices[cut:]

val_df = train_df.iloc[val_idx]

label_weights = np.array([5.0] + [1.0] * 7, dtype=np.float32)

candidate_ws = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
best_w = None
best_val_loss = np.inf

for w in candidate_ws:
    blended_overall = np.clip(
        w * derived_overall + (1 - w) * direct_overall, eps, 1 - eps
    )
    probs = np.empty((len(val_df), 8), dtype=np.float32)
    probs[:, 0] = blended_overall  # overall
    probs[:, 1:] = vertebra_probs_base  # vertebrae (baseline for validation)

    true = np.empty_like(probs)
    true[:, 0] = val_df["patient_overall"].values
    for i in range(1, 8):
        true[:, i] = val_df[f"C{i}"].values

    loss = (label_weights * binary_log_loss(true, probs)).mean()
    if loss < best_val_loss:
        best_val_loss = loss
        best_w = w

if best_w is None:
    best_w = 0.6

patient_overall_prob = float(
    np.clip(best_w * derived_overall + (1 - best_w) * direct_overall, eps, 1 - eps)
)

overall_pos_mask = train_df["patient_overall"] == 1
overall_neg_mask = ~overall_pos_mask

cond_pos = np.array(
    [train_df.loc[overall_pos_mask, f"C{i}"].mean() for i in range(1, 8)],
    dtype=np.float32,
)
cond_neg = np.array(
    [train_df.loc[overall_neg_mask, f"C{i}"].mean() for i in range(1, 8)],
    dtype=np.float32,
)

vertebra_probs_cond = (
    patient_overall_prob * cond_pos + (1 - patient_overall_prob) * cond_neg
)
vertebra_probs_cond = np.clip(vertebra_probs_cond, eps, 1 - eps)

candidate_wv = [0.0, 0.25, 0.5, 0.75, 1.0]
best_wv = None
best_val_loss_v = np.inf

for wv in candidate_wv:
    blended_vert = wv * vertebra_probs_cond + (1 - wv) * vertebra_probs_base
    probs = np.empty((len(val_df), 8), dtype=np.float32)
    probs[:, 0] = patient_overall_prob  # overall (fixed)
    probs[:, 1:] = blended_vert  # calibrated vertebrae

    true = np.empty_like(probs)
    true[:, 0] = val_df["patient_overall"].values
    for i in range(1, 8):
        true[:, i] = val_df[f"C{i}"].values

    loss = (label_weights * binary_log_loss(true, probs)).mean()
    if loss < best_val_loss_v:
        best_val_loss_v = loss
        best_wv = wv

if best_wv is None:
    best_wv = 0.5

vertebra_probs = best_wv * vertebra_probs_cond + (1 - best_wv) * vertebra_probs_base
vertebra_probs = np.clip(vertebra_probs, eps, 1 - eps)

final_probs = np.empty(8, dtype=np.float32)
final_probs[0] = patient_overall_prob  # patient_overall
final_probs[1:] = vertebra_probs  # C1‑C7


def apply_temperature(p, t):
    """Scale probability p with temperature t."""
    p = np.clip(p, eps, 1 - eps)
    num = p**t
    den = num + (1 - p) ** t + eps
    return num / den


candidate_t = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
best_t = 1.0
best_t_loss = np.inf

true_val = np.empty((len(val_df), 8), dtype=np.float32)
true_val[:, 0] = val_df["patient_overall"].values
for i in range(1, 8):
    true_val[:, i] = val_df[f"C{i}"].values

for t in candidate_t:
    scaled = apply_temperature(final_probs, t)  # shape (8,)
    probs_val = np.tile(scaled, (len(val_df), 1))
    loss = (label_weights * binary_log_loss(true_val, probs_val)).mean()
    if loss < best_t_loss:
        best_t_loss = loss
        best_t = t

final_probs = apply_temperature(final_probs, best_t)

test_df = pd.read_csv(TEST_CSV_PATH)

results = {"row_id": [], "fractured": []}

type_to_idx = {
    "patient_overall": 0,
    "C1": 1,
    "C2": 2,
    "C3": 3,
    "C4": 4,
    "C5": 5,
    "C6": 6,
    "C7": 7,
}

for _, row in test_df.iterrows():
    pred_type = row["prediction_type"]
    idx = type_to_idx.get(pred_type, 0)  # default to overall if unknown
    prob = float(final_probs[idx])
    results["row_id"].append(row["row_id"])
    results["fractured"].append(prob)

results_df = pd.DataFrame(results)
results_df.to_csv(SAVE_CSV, index=False)

print(f"==> Submission written to {SAVE_CSV} ({len(results_df)} rows)")
