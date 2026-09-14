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

0.2789364508118458

# 6. Current score

0.5634

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5639) has done: 'I replaced the failing custom model pipeline with a lightweight baseline that reads the provided training labels, computes the average fracture probability for each of the eight targets, and assigns those averages to every test row based on its `prediction_type`. This eliminates the missing‑module errors, guarantees a valid `submission.csv` with the required columns, and lets the notebook run end‑to‑end.'
- What this solution (achieved 0.5639) has done: 'I keep the overall structure but replace the naïve constant‑average prediction with a simple Bayesian‑adjusted estimate: compute the overall probability of a patient fracture, then for each vertebra predict P(C)=P(patient_overall)·P(C|patient=1)+(1‑P(patient_overall))·P(C|patient=0). This uses only the existing label statistics, adds only a few calculations, and is expected to lower the log‑loss toward the target while still writing a valid submission.csv.'
- What this solution (achieved 0.5635) has done: 'I replace the conditional‑probability calibration with simple Laplace‑smoothed label frequencies, which are less noisy on the small training set and avoid extreme predictions. Each `prediction_type` now receives its own smoothed mean probability, and the predictions are clipped to a safe interval before saving, yielding a lower log‑loss and moving the score toward the target.'
- What this solution (achieved 0.56326) has done: 'The update replaces the flat‑average prediction with a simple Bayesian‑adjusted estimate: it first smooths the overall fracture probability, then computes Laplace‑smoothed conditional probabilities P(Cₖ | patient = 1) and P(Cₖ | patient = 0) for each vertebra. Test rows are scored using  
  P = P(patient_overall)·P(Cₖ|patient=1) + (1‑P(patient_overall))·P(Cₖ|patient=0)  
and the “patient_overall” rows receive the smoothed overall probability directly. This refinement keeps the original workflow but yields more informed probabilities, moving the log‑loss toward the target value.'
- What this solution (achieved 0.5635) has done: 'I replace the Bayesian‐adjusted estimate with simple Laplace‑smoothed empirical probabilities for each label (including patient_overall). Directly using the smoothed mean of each target tends to give better calibrated probabilities on this small training set, so the log‑loss should move closer to the target value while preserving the original workflow and output format.'
- What this solution (achieved 0.56326) has done: 'I replace the simple overall‑mean prediction with a Bayesian‑adjusted estimate that uses Laplace‑smoothed conditional probabilities of each vertebra given the patient‑overall fracture label. This keeps the same workflow but provides more informative probabilities, which should lower the weighted log‑loss toward the target. The script still writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.5635) has done: 'I replace the Bayesian‑adjusted probability calculation with a simpler Laplace‑smoothed frequency for each prediction type (including patient_overall). This uses the empirical label rates directly, which better matches the weighted log‑loss on this small training set and should lower the validation loss toward the target. The rest of the pipeline (reading data, writing submission.csv) remains unchanged.'
- What this solution (achieved 0.56326) has done: 'I replace the simple per‑label mean with a Bayesian‑adjusted estimate that uses Laplace‑smoothed conditional probabilities of each vertebra given the patient‑overall fracture label, and I also smooth the patient‑overall probability itself. This keeps the original pipeline and output format but provides more informative probabilities, which should lower the weighted log‑loss and move the score nearer to the target.'
- What this solution (achieved 0.58316) has done: 'I enhance the simple Bayesian‑adjusted baseline by conditioning the label probabilities on how many bounding‑box annotations each study has. Using the training bounding‑box file we compute a per‑study box count, split studies into “low‑count” and “high‑count” groups (by the median), and calculate Laplace‑smoothed label frequencies for each group. During prediction we look up the test study’s box count and select the corresponding group‑specific probability (falling back to the overall smoothed mean when a count is missing). This adds only a lightweight conditional adjustment while preserving the original workflow, and should move the log‑loss closer to the target.'
- What this solution (achieved 0.5634) has done: 'Implemented a Bayesian‑adjusted probability that truly mixes the overall patient fracture likelihood with vertebra‑specific conditional rates conditioned on the low/high box‑count groups.  
- Compute smoothed overall patient probability.  
- For each group (low/high) calculate P(C | patient=1) and P(C | patient=0) using Laplace smoothing.  
- Predict vertebra probabilities as overall × P(C|patient=1) + (1‑overall) × P(C|patient=0); patient_overall rows use the overall probability directly.  
- Clip all outputs to a safe \[ε, 1‑ε\] range and keep the original CSV writing logic.'
- What this solution (achieved 0.56649) has done: 'I replace the Bayesian‑mixing logic with a finer‑grained Laplace‑smoothed probability that conditions directly on the exact bounding‑box count for each label. This keeps the same data flow but provides a more specific estimate for each test row, which should move the log‑loss closer to the target while still writing a valid submission file.'
- What this solution (achieved 0.56326) has done: 'I replace the per‑count Laplace probabilities with a Bayesian‑mixed estimate that uses the overall patient‑overall probability together with Laplace‑smoothed conditional probabilities P(label | patient_overall = 1) and P(label | patient_overall = 0). This retains the same data loading and CSV writing steps but provides better calibrated predictions for each vertebra, which should lower the weighted log‑loss toward the target value.'
- What this solution (achieved 0.5634) has done: 'The update adds a lightweight conditioning on the number of bounding‑box annotations: studies are split into “low” and “high” box‑count groups (by the median count). For each label we compute Laplace‑smoothed probabilities conditioned on both the patient‑overall flag and the count‑group, and we use the appropriate group‑specific probabilities when generating test predictions. This keeps the original Bayesian mixing logic while providing a more informative estimate, which should move the log‑loss closer to the target value.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
from pathlib import Path

BASE_INPUT = Path("../input/rsna-2022-cervical-spine-fracture-detection")
TRAIN_CSV = BASE_INPUT / "train.csv"
TEST_CSV = BASE_INPUT / "test.csv"
BBOX_CSV = BASE_INPUT / "train_bounding_boxes.csv"
SUBMISSION_PATH = "submission.csv"

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)  # columns: StudyInstanceUID, prediction_type, row_id
bbox_df = pd.read_csv(
    BBOX_CSV
)  # columns: StudyInstanceUID, x, y, width, height, slice_number

label_cols = ["patient_overall", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]
epsilon = 1e-5  # avoid exact 0/1 predictions


def laplace_prob(series, mask):
    """Return Laplace‑smoothed probability of a binary series under the given mask."""
    n = mask.sum()
    if n == 0:
        return (series.sum() + 1) / (len(series) + 2)
    return (series[mask].sum() + 1) / (n + 2)


overall_patient_prob = laplace_prob(
    train_df["patient_overall"], pd.Series([True] * len(train_df))
)
overall_patient_prob = max(epsilon, min(1 - epsilon, overall_patient_prob))

train_counts = (
    bbox_df.groupby("StudyInstanceUID").size().rename("box_count").reset_index()
)
train_df = train_df.merge(train_counts, on="StudyInstanceUID", how="left")
train_df["box_count"] = train_df["box_count"].fillna(0)

test_counts = (
    bbox_df.groupby("StudyInstanceUID").size().rename("box_count").reset_index()
)
test_df = test_df.merge(test_counts, on="StudyInstanceUID", how="left")
test_df["box_count"] = test_df["box_count"].fillna(0)

median_count = train_df["box_count"].median()

cond_prob = {col: {0: {}, 1: {}} for col in label_cols if col != "patient_overall"}

for col in cond_prob.keys():
    for grp in (0, 1):
        grp_mask = (
            (train_df["box_count"] > median_count)
            if grp == 1
            else (train_df["box_count"] <= median_count)
        )

        mask_pos = grp_mask & (train_df["patient_overall"] == 1)
        prob_pos = laplace_prob(train_df[col], mask_pos)
        prob_pos = max(epsilon, min(1 - epsilon, prob_pos))

        mask_neg = grp_mask & (train_df["patient_overall"] == 0)
        prob_neg = laplace_prob(train_df[col], mask_neg)
        prob_neg = max(epsilon, min(1 - epsilon, prob_neg))

        cond_prob[col][grp][1] = prob_pos
        cond_prob[col][grp][0] = prob_neg


def get_probability(row):
    pred_type = row["prediction_type"]
    if pred_type == "patient_overall":
        return overall_patient_prob

    grp = 1 if row["box_count"] > median_count else 0

    p_given_pos = cond_prob[pred_type][grp][1]  # P(label|patient=1, group)
    p_given_neg = cond_prob[pred_type][grp][0]  # P(label|patient=0, group)

    prob = overall_patient_prob * p_given_pos + (1 - overall_patient_prob) * p_given_neg
    return max(epsilon, min(1 - epsilon, prob))


test_df["fractured"] = test_df.apply(get_probability, axis=1)




## === cell 1
submission = test_df[["row_id", "fractured"]].copy()
submission.to_csv(SUBMISSION_PATH, index=False)

print(f"Submission written to {SUBMISSION_PATH}")
print("First few rows of the submission:")
print(submission.head())
