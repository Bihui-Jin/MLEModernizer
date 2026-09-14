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

0.3693149073626787

# 6. Current score

0.5639

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5639) has done: 'I replace the failing heavy‑model code with a lightweight baseline that reads the training labels, computes the overall positive rate for each fracture type, and uses those rates as prediction probabilities for every test row. This eliminates the missing‑module errors, ensures a valid `submission.csv` with the required columns, and provides reasonable scores without altering the competition’s evaluation semantics.'
- What this solution (achieved 0.69067) has done: 'I keep the overall structure but replace the constant‑mean predictions with a tiny study‑level model: each StudyInstanceUID is hashed to a numeric feature and a separate logistic‑regression is fitted for every target column. This adds a modest amount of information from the study IDs, which often correlate with fracture prevalence, and should lower the weighted log‑loss toward the target without altering the core pipeline. The code also clips probabilities to avoid extreme values.'
- What this solution (achieved 0.5639) has done: 'I replace the study‑ID logistic‑regression predictions with the overall label means, which were shown to give a much lower log‑loss (≈0.56) than the current model‑based approach (≈0.69). This keeps the existing data loading and model‑training code untouched but discards the over‑fitted probabilities, reverting to the stable global‑mean baseline that moves the score toward the target.'
- What this solution (achieved 0.5639) has done: 'I replace the simple global‑mean predictions with a modest hierarchical adjustment: for each vertebra I compute its fracture probability conditioned on the patient‑overall fracture status in the training data, then combine these with the overall patient‑overall probability for a more informed estimate. This keeps the original lightweight design, adds only a few pre‑computed statistics, and is expected to lower the weighted log‑loss toward the target without changing the overall pipeline.'
- What this solution (achieved 0.5639) has done: 'I blend the conditional probability (based on patient_overall) with each label’s overall mean, using a simple weighting (e.g., 0.8 × conditional + 0.2 × global). This keeps the original logic, adds only a minor calibration step, and is expected to reduce the weighted log‑loss, moving the score closer to the target.'
- What this solution (achieved 0.56345) has done: 'I blend the study‑level logistic‑regression predictions with the existing global/conditional probabilities, giving the model a modest amount of extra information from the hashed StudyInstanceUID while keeping the original simple statistics. This small calibration is expected to reduce the weighted log‑loss toward the target without altering the overall pipeline.'
- What this solution (achieved 0.68741) has done: 'I replace the simple logistic‑regression per‑label models with a more flexible GradientBoostingClassifier, which can capture non‑linear patterns in the hashed StudyInstanceUID feature. I also increase the blend weight toward the model predictions (lr_blend_weight = 0.7) so the improved estimator contributes more to the final probability. The rest of the pipeline and submission format stay unchanged.'
- What this solution (achieved 0.67871) has done: 'Implemented a lightweight calibration that leverages conditional fracture probabilities based on the overall patient fracture likelihood.  
- Computes per‑vertebra conditional differences (fracture probability when `patient_overall` = 1 vs 0).  
- Uses the global mean of `patient_overall` as a proxy for the overall fracture chance and adjusts each vertebra’s base probability accordingly.  
- Retains the original UID‑hash feature and GradientBoosting models, but defaults to the calibrated baseline when model predictions are unavailable, ensuring a valid submission CSV is always written.'
- What this solution (achieved 0.64653) has done: 'I replace the conditional‑diff adjustment with a full Bayesian‐style calibration that uses the predicted patient_overall probability to weigh the vertebra‑specific fracture rates observed when the patient is fractured vs. not fractured. This uses the same UID‑hash feature and GradientBoosting models (kept unchanged) but computes per‑label conditional probabilities and combines them with the model‑based patient‑overall estimate, which should lower the log‑loss toward the target. The submission CSV writing logic remains the same.'
- What this solution (achieved 0.60466) has done: 'I add a direct‑lookup for any study‑ID that appears in the training set so that we can return the exact known label for that row (clipped to a safe probability). This deterministic fallback eliminates the model‑related error on those rows and moves the log‑loss closer to the target. I also reduce the blending weight to 0.5 to keep the model influence modest while preserving the existing pipeline logic.'
- What this solution (achieved 0.5639) has done: 'I added the missing imports and helper functions, created the required statistics (global means, exact‑label lookup, conditional probabilities), trained tiny LogisticRegression models on a numeric hash of each StudyInstanceUID, and then used these models blended with the global/conditional baselines to generate predictions for every test row. Finally the script writes a proper `submission.csv` with the required columns, ensuring the file is created without errors. This fixes the NameError bugs and adds modest model‑based calibration to move the log‑loss toward the target score.'
- What this solution (achieved 0.56391) has done: 'I tighten the prediction logic for the vertebra‑specific rows: instead of an extra blend between the baseline‑plus‑model probability and the conditional adjustment, I use the conditional adjustment directly (which already incorporates the model‑based patient‑overall estimate). This gives the conditional calibration more influence, which should lower the weighted log‑loss toward the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.5639) has done: 'I keep the overall pipeline unchanged but make the prediction for the “patient_overall” label rely only on the global mean (removing the noisy UID‑based logistic regression for this label) and keep a modest blend (weight 0.2) for the vertebra‑specific predictions. This stabilises the conditional adjustment that uses the patient‑overall probability and should lower the log‑loss toward the target value.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import hashlib
from sklearn.linear_model import LogisticRegression


def uid_to_float(uid: str) -> float:
    h = int(hashlib.sha256(uid.encode()).hexdigest(), 16)
    return (h % (10**9)) / 1e9


lr_blend_weight = 0.2




## === cell 1
train_path = "../input/rsna-2022-cervical-spine-fracture-detection/train.csv"
df_train = pd.read_csv(train_path)

label_cols = ["patient_overall", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]
global_means = {col: df_train[col].mean() for col in label_cols}

exact_label_lookup = {}
for _, row in df_train.iterrows():
    uid = row["StudyInstanceUID"]
    for col in label_cols:
        exact_label_lookup[(uid, col)] = row[col]

cond_probs = {}
overall_mean = global_means["patient_overall"]
if overall_mean > 0:
    for col in label_cols:
        if col == "patient_overall":
            continue
        prob1 = df_train[df_train["patient_overall"] == 1][col].mean()
        prob0 = df_train[df_train["patient_overall"] == 0][col].mean()
        prob1 = prob1 if not np.isnan(prob1) else global_means[col]
        prob0 = prob0 if not np.isnan(prob0) else global_means[col]
        cond_probs[col] = (prob1, prob0)




## === cell 2
models = {}
X_uid = df_train["StudyInstanceUID"].apply(uid_to_float).values.reshape(-1, 1)

for col in label_cols:
    y = df_train[col].values
    lr = LogisticRegression(solver="lbfgs", max_iter=1000)
    lr.fit(X_uid, y)
    models[col] = lr




## === cell 3
test_path = "../input/rsna-2022-cervical-spine-fracture-detection/test.csv"
df_test = pd.read_csv(test_path)


def map_probability(row):
    pred_type = row["prediction_type"]
    uid = row["StudyInstanceUID"]

    if (uid, pred_type) in exact_label_lookup:
        label = exact_label_lookup[(uid, pred_type)]
        return np.clip(label, 1e-5, 1 - 1e-5)

    base_prob = global_means.get(pred_type, 0.0)

    try:
        lr_prob = models[pred_type].predict_proba([[uid_to_float(uid)]])[0, 1]
    except Exception:
        lr_prob = None

    if lr_prob is not None:
        blended = (1 - lr_blend_weight) * base_prob + lr_blend_weight * lr_prob
    else:
        blended = base_prob

    if pred_type != "patient_overall":
        overall_blended = global_means.get("patient_overall", 0.0)

        prob1, prob0 = cond_probs.get(pred_type, (base_prob, base_prob))
        adjusted = overall_blended * prob1 + (1 - overall_blended) * prob0

        final_prob = adjusted
    else:
        final_prob = blended

    return np.clip(final_prob, 1e-5, 1 - 1e-5)


df_test["fractured"] = df_test.apply(map_probability, axis=1)




## === cell 4
submission_path = "submission.csv"
df_test[["row_id", "fractured"]].to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path} with {df_test.shape[0]} rows.")
