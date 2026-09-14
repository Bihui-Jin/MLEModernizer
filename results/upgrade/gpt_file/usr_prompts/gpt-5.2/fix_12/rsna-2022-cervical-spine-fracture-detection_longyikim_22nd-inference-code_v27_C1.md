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

0.60686

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.7459) has done: 'I remove the notebook-style `pip install`/custom model dependencies (effdet/efficientunet/pylibjpeg) that are not available in your current environment and are causing the import crashes. To keep the pipeline end-to-end and submission-valid, I replace the broken model inference with a lightweight baseline that uses only `train.csv` label priors to generate probabilities, while preserving the same submission structure (joining on `StudyInstanceUID` and selecting by `prediction_type`). I also fix the undefined-name issues by consolidating imports and ensuring `df_patient_pred` is always created before submission assembly. Finally, I guarantee `submission.csv` is written with the exact required columns (`row_id, fractured`) and correct row order/length matching `test.csv`.'
- What this solution (achieved 0.75581) has done: 'Your current solution predicts the same constant priors for every study, which is a weak baseline and explains the high (worse) log loss. To move the score down toward the target with minimal logic changes, I keep the “label-prior baseline” approach but make it slightly study-specific by using the provided `train_bounding_boxes.csv` as a lightweight proxy feature: scans with more vertebrae boxes and larger total box area get a modestly higher predicted fracture probability. I calibrate this effect conservatively (small shift in logit space) and keep `patient_overall` as the max over C1–C7 to match the evaluation semantics. The output format, join logic, clipping, and `submission.csv` writing remain unchanged.'
- What this solution (achieved 0.74148) has done: 'Your current baseline already runs end-to-end, but it’s likely not improving because it (a) uses `train_bounding_boxes.csv` features that don’t exist for the hidden test set (so everything becomes “missing -> 0” and collapses back to nearly-constant priors), and (b) fits a linear model in probability space, which can generate poorly-calibrated adjustments for log-loss. I keep the same core idea (“priors + small study-specific adjustment”) but switch the study-specific signal to something available at inference time: the number of DICOM slices per study (folder file count) from `test_images/`, and I fit/apply the adjustment in logit space via a tiny ridge-regularized logistic regression using only NumPy. This should legitimately reduce log-loss (move the score down toward your target) while preserving the overall structure and submission semantics (same columns, same join logic, same clipping, `patient_overall = max(C1..C7)`). The changes are minimal and designed to finish within the time limit by caching slice counts and only scanning directories once.'
- What this solution (achieved 0.73818) has done: 'Your current slice-count logistic adjustment is reasonable, but it’s likely underpowered because (1) you’re fitting only on `patient_overall` while the metric heavily weights `patient_overall` yet still includes C1–C7, and (2) you’re applying a small, uniform delta transfer to all C-levels that may be miscalibrated. I keep the same core logic (“priors + small study-specific adjustment from DICOM slice count in logit space”) but fit the same 1D ridge logistic model separately for each of the 8 labels, then (as before) set `patient_overall = max(C1..C7)` to preserve evaluation semantics. I also make the shrinkage and ridge strength slightly more conservative per-label (shared constants) and keep clipping identical to avoid log-loss blowups. This is a minimal change that should legitimately reduce the log loss from 0.74148 toward your target band.'
- What this solution (achieved 0.73818) has done: 'Your current pipeline is already valid and fairly stable, but it’s still far above the target (0.738 vs 0.435, lower is better), so we need a modest real improvement without changing the overall “priors + 1D slice-count ridge-logistic + shrink + patient_overall=max(C1..C7)” core logic. The main low-risk gain is to fit the 1D logistic adjustment using a patient-level sample weight consistent with the competition metric (patient_overall is weighted higher), which should reduce loss primarily where it matters most. I also make the `patient_overall` fitting target consistent with the enforced post-processing (`max(C1..C7)`) by training the patient_overall model on that derived label rather than the raw column, while keeping the submission semantics unchanged. Finally, I keep clipping/shrinkage and directory scanning behavior the same to preserve stability and runtime.'
- What this solution (achieved 0.73818) has done: 'Your current approach is stable and already aligned with the metric (probabilities + clipping), but it’s likely leaving score on the table because the model is trained unweighted for C1–C7 even though the competition uses per-label weights (and patient_overall is only indirectly enforced via post-processing). I keep the exact same “priors + 1D z-scored slice-count ridge-logistic via IRLS + shrinkage + patient_overall = max(C1..C7)” core logic, but I incorporate competition-style label weights directly into the IRLS fit for all 8 labels (not just patient_overall) to better match the evaluation objective. I also make the training target for the patient_overall model consistent with your enforced semantics by using the derived label (max of C1–C7) as you already do, and I keep clipping/paths/output unchanged to ensure a valid submission. These are minimal changes intended to legitimately reduce weighted log loss (move your score down toward the 0.435 target) without altering the modeling approach.'
- What this solution (achieved 0.80595) has done: 'I keep your existing “priors + 1D z-scored slice-count ridge-logistic via IRLS + shrinkage + patient_overall=max(C1..C7)” pipeline intact, but make two minimal score-relevant fixes to better match the competition metric. First, I replace the current constant per-label sample weights (which don’t actually change the fit) with a proper weighted log-loss IRLS fit (using label weights and per-class balancing) so the learned slice-count adjustment is calibrated toward the weighted metric. Second, I ensure patient_overall training target and inference remain consistent with your enforced semantics by fitting patient_overall directly on the derived max(C1..C7) label and by using the same clipping after the final patient_overall recomputation. These changes are small, keep runtime within limits, and should legitimately lower log loss from 0.738 toward your target 0.435 without changing the core approach.'
- What this solution (achieved 0.80594) has done: 'I fix the runtime error caused by `DataFrame.lookup`, which was removed from modern pandas, by replacing it with a vectorized NumPy-based selection that preserves the exact mapping semantics (`fractured = column named by prediction_type` for each row). This unblocks the pipeline so `sub` is created and cell 6 can write `submission.csv` successfully. I also add a safety clip after filling NaNs to ensure probabilities stay in a valid log-loss range, without changing the core modeling logic. No model/training logic is changed; only the submission assembly bug and robustness are addressed.'
- What this solution (achieved 0.60596) has done: 'Your current pipeline is producing a valid submission but is still far above the target (0.80594 vs 0.43539, lower is better), so we need a small, legitimate improvement without changing the core “priors + 1D slice-count ridge-logistic + shrink + patient_overall=max(C1..C7)” approach. The biggest score-relevant bug is that you z-score using **test** slice-count statistics, which creates a train/test feature scale mismatch and weakens the learned relationship; switching to train-derived normalization is a minimal, semantics-preserving fix that should lower log loss. I also make the per-label class-balance weights numerically safer on this tiny train set (clip extreme weights) to avoid unstable over/under-confidence that hurts log loss. All paths, model form, fitting loop, post-processing, clipping, and submission assembly remain the same, and it still writes `submission.csv`.'
- What this solution (achieved 0.60686) has done: 'Your current score (0.60596, lower-is-better) is still far above the target (0.43539), so we need a small but legitimate improvement without changing the core “priors + 1D slice-count ridge-logistic + shrink + patient_overall=max(C1..C7)” approach. The biggest low-risk gain is to fit the 1D logistic adjustment on a feature that better captures scan extent: use both `n_slices` and an inexpensive proxy for slice spacing (the max DICOM instance number / `n_slices`) computed from filenames only (no DICOM decoding), then compress back to a single 1D feature via a fixed linear combination to preserve the 1D model. This keeps the same training loop/IRLS and semantics, but provides a slightly more informative per-study signal than slice count alone. I also keep train-derived normalization, clipping, and submission assembly unchanged to preserve stability and ensure a valid `submission.csv`.'

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
def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -50, 50)
    return 1 / (1 + np.exp(-z))


def _zscore_with_stats(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    if std < 1e-12:
        return x * 0.0
    return (x - mean) / std


def _count_dicoms_and_max_instance_in_study_folder(study_dir: str) -> tuple[int, int]:
    """
    Change (score-relevant, minimal): still only uses filesystem metadata (filenames),
    but captures both how many slices exist and how far the instance numbering spans.
    This can better reflect scan extent/coverage than n_slices alone, improving the
    study-specific adjustment without changing the 1D ridge-logistic core logic.
    """
    try:
        cnt = 0
        max_inst = 0
        with os.scandir(study_dir) as it:
            for e in it:
                if e.is_file() and e.name.lower().endswith(".dcm"):
                    cnt += 1
                    stem = e.name[:-4]
                    if stem.isdigit():
                        v = int(stem)
                        if v > max_inst:
                            max_inst = v
        return cnt, max_inst
    except FileNotFoundError:
        return 0, 0


def _build_slice_feat_df(study_uids: np.ndarray, images_root: str) -> pd.DataFrame:
    n_slices = []
    max_inst = []
    for uid in study_uids:
        c, m = _count_dicoms_and_max_instance_in_study_folder(
            os.path.join(images_root, uid)
        )
        n_slices.append(c)
        max_inst.append(m)
    df = pd.DataFrame(
        {"StudyInstanceUID": study_uids, "n_slices": n_slices, "max_instance": max_inst}
    )
    df["inst_ratio"] = df["max_instance"] / np.maximum(df["n_slices"], 1)
    return df


def _fit_ridge_logistic_1d_weighted(
    x: np.ndarray,
    y: np.ndarray,
    prior_p: float,
    lam: float,
    sample_weight: np.ndarray | None = None,
    iters: int = 25,
) -> np.ndarray:
    """
    Core logic preserved: 1D ridge logistic regression via IRLS.
    """
    X = np.stack([np.ones_like(x), x], axis=1)  # (N, 2)
    beta = np.array([_logit(np.array([prior_p], dtype=float))[0], 0.0], dtype=float)

    if sample_weight is None:
        sw = np.ones_like(y, dtype=float)
    else:
        sw = np.asarray(sample_weight, dtype=float)
        sw = np.clip(sw, 0.0, np.inf)
        if sw.shape != y.shape:
            raise ValueError("sample_weight must have same shape as y")

    for _ in range(iters):
        z = X @ beta
        p = _sigmoid(z)

        w = (p * (1 - p) + 1e-9) * sw  # (N,)
        H = X.T @ (X * w[:, None])  # (2,2)

        g = X.T @ ((p - y) * sw)  # (2,)

        H[1, 1] += lam
        g[1] += lam * beta[1]

        step = np.linalg.solve(H, g)
        beta_new = beta - step
        if float(np.max(np.abs(beta_new - beta))) < 1e-8:
            beta = beta_new
            break
        beta = beta_new
    return beta


train_uids = df_train["StudyInstanceUID"].values
df_train_feat = _build_slice_feat_df(train_uids, TRAIN_IMAGES_PATH)

df_fit = df_train[["StudyInstanceUID"] + target_cols].merge(
    df_train_feat, on="StudyInstanceUID", how="left"
)
df_fit["n_slices"] = df_fit["n_slices"].fillna(0.0).astype(float)
df_fit["inst_ratio"] = df_fit["inst_ratio"].fillna(1.0).astype(float)

df_fit["patient_overall_derived"] = df_fit[[f"C{i}" for i in range(1, 8)]].max(axis=1)

test_uids_unique = df_test["StudyInstanceUID"].unique()
df_test_feat = _build_slice_feat_df(test_uids_unique, TEST_IMAGES_PATH)
df_test_feat["n_slices"] = df_test_feat["n_slices"].fillna(0.0).astype(float)
df_test_feat["inst_ratio"] = df_test_feat["inst_ratio"].fillna(1.0).astype(float)

x_raw_train = np.log1p(df_fit["n_slices"].values.astype(float)) + 0.25 * np.log1p(
    df_fit["inst_ratio"].values.astype(float)
)
x_raw_test = np.log1p(df_test_feat["n_slices"].values.astype(float)) + 0.25 * np.log1p(
    df_test_feat["inst_ratio"].values.astype(float)
)

x_mean = float(x_raw_train.mean())
x_std = float(x_raw_train.std(ddof=0))

x_train = _zscore_with_stats(x_raw_train, x_mean, x_std).astype(float)
x_test = _zscore_with_stats(x_raw_test, x_mean, x_std).astype(float)

lam = 2.0  # unchanged for stability
shrink = 0.70  # unchanged to avoid overfitting on tiny train set

LABEL_WEIGHT = {"patient_overall": 7.0, **{f"C{i}": 1.0 for i in range(1, 8)}}

df_patient_pred = pd.DataFrame({"StudyInstanceUID": test_uids_unique})
X_test = np.stack([np.ones_like(x_test), x_test], axis=1)

for lab in target_cols:
    if lab == "patient_overall":
        y_lab = df_fit["patient_overall_derived"].astype(float).values
        prior_p = float(priors["patient_overall"])
    else:
        y_lab = df_fit[lab].astype(float).values
        prior_p = float(priors[lab])

    pos = float(np.sum(y_lab > 0.5))
    neg = float(len(y_lab) - pos)
    if pos < 1.0 or neg < 1.0:
        class_w = np.ones_like(y_lab, dtype=float)
    else:
        w_pos = 0.5 / (pos / len(y_lab))
        w_neg = 0.5 / (neg / len(y_lab))
        class_w = np.where(y_lab > 0.5, w_pos, w_neg).astype(float)

        class_w = np.clip(class_w, 0.25, 4.0)

    sw = class_w * float(LABEL_WEIGHT[lab])

    beta_lab = _fit_ridge_logistic_1d_weighted(
        x_train, y_lab, prior_p=prior_p, lam=lam, sample_weight=sw, iters=25
    )
    p_raw = _sigmoid(X_test @ beta_lab)
    p_pred = shrink * p_raw + (1 - shrink) * prior_p
    df_patient_pred[lab] = p_pred

df_patient_pred = df_patient_pred.set_index("StudyInstanceUID")

min_clip_value = 0.005
max_clip_value = 0.005

df_patient_pred[target_cols] = df_patient_pred[target_cols].clip(
    lower=min_clip_value, upper=1 - max_clip_value
)

df_patient_pred["patient_overall"] = df_patient_pred[
    [f"C{i}" for i in range(1, 8)]
].max(axis=1)

df_patient_pred[target_cols] = df_patient_pred[target_cols].clip(
    lower=min_clip_value, upper=1 - max_clip_value
)

df_patient_pred = df_patient_pred[["patient_overall"] + [f"C{i}" for i in range(1, 8)]]
df_patient_pred.head()



## === cell 5
df_sub = df_test.copy()
df_sub = df_sub.set_index("StudyInstanceUID").join(df_patient_pred, how="left")

for c in target_cols:
    df_sub[c] = df_sub[c].fillna(priors[c])

df_sub = df_sub.reset_index(drop=False)

col_to_idx = {c: i for i, c in enumerate(target_cols)}
pred_type = df_sub["prediction_type"].values
if not np.isin(pred_type, np.array(target_cols, dtype=object)).all():
    bad = pd.Series(
        pred_type[~np.isin(pred_type, np.array(target_cols, dtype=object))]
    ).unique()[:10]
    raise ValueError(f"Unexpected prediction_type values (showing up to 10): {bad}")

col_idx = np.array([col_to_idx[t] for t in pred_type], dtype=int)
pred_matrix = df_sub[target_cols].to_numpy(dtype=float, copy=False)
row_idx = np.arange(len(df_sub), dtype=int)
df_sub["fractured"] = pred_matrix[row_idx, col_idx].astype(float)

df_sub["fractured"] = df_sub["fractured"].clip(
    lower=min_clip_value, upper=1 - max_clip_value
)

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
