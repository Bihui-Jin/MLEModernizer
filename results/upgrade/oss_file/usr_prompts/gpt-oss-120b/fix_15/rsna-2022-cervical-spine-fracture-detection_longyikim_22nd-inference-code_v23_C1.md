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

0.4352621600220207

# 6. Current score

0.56959

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5639) has done: 'I replace the failing image‑model pipeline with a lightweight baseline that computes the average fracture rate for each label from the training set and uses these averages as the predicted probabilities for every test row. This removes the unavailable external modules, ensures the script runs end‑to‑end, and creates a correctly formatted `submission.csv`. The core logic of training data handling is retained, while the heavy model code is omitted.'
- What this solution (achieved 0.56385) has done: 'I replace the simple global‑mean predictions with conditional probabilities that factor in the patient‑overall label. By using P(label | patient_overall) and the overall patient‑overall prevalence, the estimates become better calibrated to the true distribution, which should lower the weighted log‑loss and move the score closer to the target.'
- What this solution (achieved 0.56041) has done: 'The update keeps the same simple probability‑based approach but improves the patient‑overall prediction by estimating the chance that **any** cervical fracture is present ( 1 – ∏(1 – p_i) ), which better matches the weighted log‑loss focus on this label. This small calibration change moves the log‑loss closer to the target without altering the core logic or adding heavyweight modeling.'
- What this solution (achieved 0.56385) has done: 'I replace the product‑based estimate of the `patient_overall` probability with the plain prevalence of that label from the training set (already computed as `overall_pat_overall`). This aligns the heavily weighted “any fracture” prediction with its true frequency, which should lower the weighted log‑loss and move the score closer to the target while keeping the overall simple conditional‑mean approach unchanged.'
- What this solution (achieved 0.5639) has done: 'I replace the conditional‑mean estimates with simple marginal means for each fracture label, which removes noisy conditioning on the small `patient_overall` groups and should lower the weighted log‑loss toward the target. The rest of the pipeline, file handling and submission formatting stay unchanged, and the cell indices are renumbered to start at 1 as required.'
- What this solution (achieved 0.56041) has done: 'I replace the simple global‑mean predictions with conditional probabilities that combine the prevalence of each vertebra given the patient‑overall label, and I compute the patient‑overall probability from the resulting vertebra probabilities (using 1 – ∏(1 – p)). This keeps the pipeline lightweight while better matching the weighted log‑loss, so the score should move closer to the target.'
- What this solution (achieved 0.5639) has done: 'Implemented a lightweight calibration step that switches from the conditional‑mixing scheme to pure marginal probabilities for each vertebra and selects the best‑performing estimate for the heavily‑weighted `patient_overall` label by evaluating both the marginal and product‑based versions on the training set. This keeps the core logic intact, adds only minimal computation, and chooses the prediction that yields the lower log‑loss, moving the score nearer to the target.'
- What this solution (achieved 0.56385) has done: 'I add a light calibration step that computes conditional fracture probabilities P(Ci | patient_overall) from the training data and combines them with the overall patient‑overall prevalence. This keeps the original simple‑mean approach while providing a more informed estimate for each vertebra, which should lower the weighted log‑loss and move the score nearer to the target.'
- What this solution (achieved 0.5639) has done: 'I added a small model‑selection step that evaluates the three probability schemes (marginal, product‑based patient overall, and conditional‑calibrated) plus a calibrated + product combination on the training data and chooses the one with the lowest log‑loss. The chosen scheme’s vertebra probabilities and patient‑overall probability are then used for the test predictions, keeping the core logic unchanged while aiming to reduce the loss toward the target.'
- What this solution (achieved 0.5639) has done: 'I add a lightweight scaling calibration that adjusts the vertebra probabilities so that their combined product‑based overall probability matches the observed overall prevalence. This new “scaled” scheme is evaluated alongside the existing ones, and the best‑performing scheme (now possibly the scaled one) is selected for the final submission. The change keeps the overall structure unchanged while aiming to reduce the log‑loss toward the target.'
- What this solution (achieved 0.56369) has done: 'I added a small Laplace‑smoothing step when computing all label frequencies (marginal and conditional means) to avoid extreme 0/1 probabilities that inflate log‑loss. The smoothed frequencies replace the original raw means and are used throughout the same scheme‑selection logic, keeping the overall pipeline unchanged while giving a modest improvement toward the target score.'
- What this solution (achieved 0.56386) has done: 'We lower the Laplace smoothing factor to 0.1 for less bias toward 0.5 and evaluate schemes using a weighted log‑loss that gives the heavily‑weighted patient_overall label a larger weight (5×). The scheme with the lowest weighted loss is then chosen for the final predictions, keeping the overall pipeline unchanged while nudging the score toward the target.'
- What this solution (achieved 0.56959) has done: 'The update keeps the lightweight probability‑based pipeline but adds a final scaling step: after the best scheme (marginal, product, calibrated, …) is chosen, the vertebra probabilities are uniformly scaled so that the derived “any fracture” probability matches the observed patient‑overall prevalence. This better aligns the heavily weighted patient_overall label with the training distribution, which is expected to lower the weighted log‑loss and move the score nearer to the target while preserving the core logic.'
- What this solution (achieved 0.56959) has done: 'I add a lightweight “blended” probability scheme that averages the marginal and calibrated vertebra probabilities, evaluate its weighted log‑loss alongside the existing schemes, and let the selection logic choose the best one. This small calibration tweak keeps the core pipeline unchanged while giving a chance to lower the loss toward the target.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import os

DATA_ROOT = "../input/rsna-2022-cervical-spine-fracture-detection"

train_path = os.path.join(DATA_ROOT, "train.csv")
df_train = pd.read_csv(train_path)

target_cols = ["patient_overall"] + [f"C{i}" for i in range(1, 8)]
epsilon = 1e-3
alpha = 0.1  # reduced Laplace smoothing for less bias


def smoothed_mean(series, a=alpha):
    return (series.sum() + a) / (len(series) + 2 * a)


marginal_means = {
    col: np.clip(smoothed_mean(df_train[col]), epsilon, 1 - epsilon)
    for col in target_cols
}
col_probs = {f"C{i}": marginal_means[f"C{i}"] for i in range(1, 8)}
vertebra_probs = np.array([col_probs[f"C{i}"] for i in range(1, 8)])

patient_overall_marginal = marginal_means["patient_overall"]
patient_overall_product = np.clip(1 - np.prod(1 - vertebra_probs), epsilon, 1 - epsilon)

cond_means = {}
for i in range(1, 8):
    col = f"C{i}"
    cond_means[col] = {
        1: np.clip(
            smoothed_mean(df_train.loc[df_train["patient_overall"] == 1, col]),
            epsilon,
            1 - epsilon,
        ),
        0: np.clip(
            smoothed_mean(df_train.loc[df_train["patient_overall"] == 0, col]),
            epsilon,
            1 - epsilon,
        ),
    }

calibrated_probs = {}
for i in range(1, 8):
    col = f"C{i}"
    calibrated_probs[col] = (
        patient_overall_marginal * cond_means[col][1]
        + (1 - patient_overall_marginal) * cond_means[col][0]
    )

blended_probs = {
    f"C{i}": np.clip(
        (col_probs[f"C{i}"] + calibrated_probs[f"C{i}"]) / 2.0, epsilon, 1 - epsilon
    )
    for i in range(1, 8)
}


def compute_scaled_probs(base_probs, target_overall, eps=epsilon):
    """Scale all vertebra probabilities by a common factor so that the
    derived product‑based overall probability equals `target_overall`."""
    upper = min((1 - eps) / v if v > 0 else 1e6 for v in base_probs.values())
    lower = 0.0
    for _ in range(30):  # binary search for scaling factor
        mid = (lower + upper) / 2.0
        scaled = {k: np.clip(mid * v, eps, 1 - eps) for k, v in base_probs.items()}
        overall = 1 - np.prod([1 - scaled[k] for k in scaled])
        if overall > target_overall:
            upper = mid
        else:
            lower = mid
    final_scaled = {k: np.clip(mid * v, eps, 1 - eps) for k, v in base_probs.items()}
    return final_scaled, overall


scaled_probs, _ = compute_scaled_probs(col_probs, patient_overall_marginal)
scaled_calibrated, _ = compute_scaled_probs(calibrated_probs, patient_overall_marginal)
scaled_blended, _ = compute_scaled_probs(blended_probs, patient_overall_marginal)

rows = []
for _, row in df_train.iterrows():
    uid = row["StudyInstanceUID"]
    for col in target_cols:
        rows.append({"StudyInstanceUID": uid, "prediction_type": col, "true": row[col]})
df_long = pd.DataFrame(rows)


def log_loss(y_true, y_pred):
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


df_long["pred_marginal"] = df_long.apply(
    lambda r: (
        patient_overall_marginal
        if r["prediction_type"] == "patient_overall"
        else col_probs[r["prediction_type"]]
    ),
    axis=1,
)
df_long["pred_product"] = df_long.apply(
    lambda r: (
        patient_overall_product
        if r["prediction_type"] == "patient_overall"
        else col_probs[r["prediction_type"]]
    ),
    axis=1,
)
df_long["pred_calibrated"] = df_long.apply(
    lambda r: (
        patient_overall_marginal
        if r["prediction_type"] == "patient_overall"
        else calibrated_probs[r["prediction_type"]]
    ),
    axis=1,
)
df_long["pred_calibrated_product"] = df_long.apply(
    lambda r: (
        patient_overall_product
        if r["prediction_type"] == "patient_overall"
        else calibrated_probs[r["prediction_type"]]
    ),
    axis=1,
)
df_long["pred_scaled"] = df_long.apply(
    lambda r: (
        patient_overall_marginal
        if r["prediction_type"] == "patient_overall"
        else scaled_probs[r["prediction_type"]]
    ),
    axis=1,
)
df_long["pred_calibrated_scaled"] = df_long.apply(
    lambda r: (
        patient_overall_marginal
        if r["prediction_type"] == "patient_overall"
        else scaled_calibrated[r["prediction_type"]]
    ),
    axis=1,
)
df_long["pred_blended"] = df_long.apply(
    lambda r: (
        patient_overall_marginal
        if r["prediction_type"] == "patient_overall"
        else blended_probs[r["prediction_type"]]
    ),
    axis=1,
)
df_long["pred_blended_scaled"] = df_long.apply(
    lambda r: (
        patient_overall_marginal
        if r["prediction_type"] == "patient_overall"
        else scaled_blended[r["prediction_type"]]
    ),
    axis=1,
)

weights = (
    df_long["prediction_type"]
    .apply(lambda x: 5 if x == "patient_overall" else 1)
    .values
)


def weighted_mean(loss_vals):
    return np.sum(loss_vals * weights) / np.sum(weights)


loss_marginal = weighted_mean(log_loss(df_long["true"], df_long["pred_marginal"]))
loss_product = weighted_mean(log_loss(df_long["true"], df_long["pred_product"]))
loss_calibrated = weighted_mean(log_loss(df_long["true"], df_long["pred_calibrated"]))
loss_calibrated_product = weighted_mean(
    log_loss(df_long["true"], df_long["pred_calibrated_product"])
)
loss_scaled = weighted_mean(log_loss(df_long["true"], df_long["pred_scaled"]))
loss_calibrated_scaled = weighted_mean(
    log_loss(df_long["true"], df_long["pred_calibrated_scaled"])
)
loss_blended = weighted_mean(log_loss(df_long["true"], df_long["pred_blended"]))
loss_blended_scaled = weighted_mean(
    log_loss(df_long["true"], df_long["pred_blended_scaled"])
)

losses = {
    "marginal": loss_marginal,
    "product": loss_product,
    "calibrated": loss_calibrated,
    "calibrated_product": loss_calibrated_product,
    "scaled": loss_scaled,
    "calibrated_scaled": loss_calibrated_scaled,
    "blended": loss_blended,
    "blended_scaled": loss_blended_scaled,
}
best_scheme = min(losses, key=losses.get)

if best_scheme == "marginal":
    patient_overall_prob = patient_overall_marginal
    vertebra_prob_dict = col_probs
elif best_scheme == "product":
    patient_overall_prob = patient_overall_product
    vertebra_prob_dict = col_probs
elif best_scheme == "calibrated":
    patient_overall_prob = patient_overall_marginal
    vertebra_prob_dict = calibrated_probs
elif best_scheme == "calibrated_product":
    patient_overall_prob = patient_overall_product
    vertebra_prob_dict = calibrated_probs
elif best_scheme == "scaled":
    patient_overall_prob = patient_overall_marginal
    vertebra_prob_dict = scaled_probs
elif best_scheme == "calibrated_scaled":
    patient_overall_prob = patient_overall_marginal
    vertebra_prob_dict = scaled_calibrated
elif best_scheme == "blended":
    patient_overall_prob = patient_overall_marginal
    vertebra_prob_dict = blended_probs
else:  # blended_scaled
    patient_overall_prob = patient_overall_marginal
    vertebra_prob_dict = scaled_blended

print(
    f"Weighted log‑loss – marginal: {loss_marginal:.5f}, product: {loss_product:.5f}, "
    f"calibrated: {loss_calibrated:.5f}, calibrated+product: {loss_calibrated_product:.5f}, "
    f"scaled: {loss_scaled:.5f}, calibrated_scaled: {loss_calibrated_scaled:.5f}, "
    f"blended: {loss_blended:.5f}, blended_scaled: {loss_blended_scaled:.5f}"
)
print(
    f"Chosen scheme: {best_scheme} → patient_overall_prob = {patient_overall_prob:.5f}"
)

vertebra_prob_dict, _ = compute_scaled_probs(
    vertebra_prob_dict, patient_overall_marginal
)




## === cell 1
test_path = os.path.join(DATA_ROOT, "test.csv")
df_test = pd.read_csv(test_path)

assert {"row_id", "StudyInstanceUID", "prediction_type"}.issubset(df_test.columns)




## === cell 2
def get_prob(row):
    pred_type = row["prediction_type"]
    if pred_type == "patient_overall":
        return patient_overall_prob
    else:
        return vertebra_prob_dict[pred_type]


df_test["fractured"] = df_test.apply(get_prob, axis=1)




## === cell 3
submission_path = "submission.csv"
df_test[["row_id", "fractured"]].to_csv(submission_path, index=False)

print(f"Submission file written to {submission_path}")
