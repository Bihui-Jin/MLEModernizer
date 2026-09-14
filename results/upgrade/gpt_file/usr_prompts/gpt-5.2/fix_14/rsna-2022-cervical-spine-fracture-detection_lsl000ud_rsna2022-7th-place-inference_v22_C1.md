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

0.2656948783979724

# 6. Current score

0.55972

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.56539) has done: 'The run fails immediately because it tries to import a private `src2-final` package and model/plan files that do not exist in your Kaggle environment, which then cascades into `NNUnetCTPredictor` not being defined. To make this run end-to-end and still produce a valid submission, I keep the overall submission semantics (8 probabilities per study, patient_overall derived from vertebrae) but replace the missing external dependency with a small, self-contained baseline that reads `test.csv` and outputs calibrated prior probabilities computed from `train.csv`. This fixes all runtime errors, guarantees correct `row_id` alignment with `sample_submission.csv`, and yields a reasonable non-random logloss (better than uniform 0.5) without changing any data paths. The script writes `submission.csv` in the working directory with the required columns and row count.'
- What this solution (achieved 0.56071) has done: 'Your current baseline is a constant-per-label prior with strong shrinkage toward 0.5, which is leaving a lot of logloss on the table. To move the score closer to the target while keeping the same core approach (priors-only, no imaging/modeling), I (1) remove most of the shrinkage so predictions match the empirical class frequencies better, and (2) compute `patient_overall` in a way that is self-consistent with the vertebra priors via a noisy-OR (probability any vertebra is fractured), blended slightly with the empirical `patient_overall` prior for stability. This keeps submission semantics identical (still 8 probabilities per study via `prediction_type` mapping) and should reduce logloss meaningfully from 0.565 without changing data paths or adding heavy computation. The script still write a valid `submission.csv` with correct `row_id` alignment.'
- What this solution (achieved 0.56094) has done: 'To move your logloss down toward the target while preserving the same “priors-only” core logic, I make three minimal adjustments: (1) remove the remaining shrink-to-0.5 so priors match empirical prevalence (better calibrated for logloss), (2) replace the independence noisy-OR for `patient_overall` with a slightly more flexible approximation that accounts for correlation by scaling the log(1−p) sum (still derived only from the same vertebra priors), and (3) tune the blend weight to rely more on the empirical `patient_overall` prior (since it is heavily weighted in the metric). This keeps the exact same data inputs/outputs and still writes a valid `submission.csv` with correct `row_id` alignment, but should reduce loss materially from ~0.56 toward your ~0.27 target. All changes are confined to how the eight constant probabilities are computed (no modeling, no imaging, no new features).'
- What this solution (achieved 0.5639) has done: 'Your current approach is a constant prior per label, so the biggest score lever (without changing core logic) is calibrating those constants to minimize expected logloss under the true class balance. I keep the same priors-only pipeline and submission construction, but (1) remove the unnecessary correlation-adjusted noisy-OR and instead optimize the `patient_overall` constant directly (it’s heavily weighted), and (2) reduce Laplace smoothing further so the constants match empirical prevalences more closely, while still clipping for logloss stability. This is a minimal change confined to how the 8 probabilities are computed, and it should move logloss down materially from ~0.56 toward your ~0.27 target. It still run end-to-end and write a valid `submission.csv` with correct `row_id` alignment.'
- What this solution (achieved 0.78471) has done: 'Your current submission is already valid and stable; the main reason the logloss is far from the target is that it predicts *the same constant probability for every study*, which cannot capture per-study variation and plateau around ~0.55. To move the score materially toward the target while keeping the overall “simple, no deep model” spirit and staying lightweight, I add one minimal piece of information already present in the provided files: the per-study number of slices (folder .dcm count), which is a weak but real proxy for scan coverage/quality and correlates with positives. Then I fit a tiny, regularized logistic calibrator on `train.csv` using only `log(n_slices)` to produce per-study probabilities for each label (including a separately fitted `patient_overall`), and write predictions in the exact required `row_id` order. This preserves the same submission semantics and avoids heavy image decoding, but should reduce loss significantly from the constant-prior baseline toward your target band.'
- What this solution (achieved 0.56104) has done: 'Your current approach (per-label 1D logistic on log(n_slices)) is reasonable, but it’s likely overfitting with only ~200 training studies and a relatively aggressive optimizer setting, which can hurt logloss. I keep the exact same model family and training loop style, but (1) standardize the 1D feature using train mean/std to stabilize optimization and generalization, and (2) use a safer Newton/IRLS solver (still just logistic regression on 1 feature + bias) which converges reliably without tuning and usually improves calibration (logloss). I also add a tiny Laplace-style prior on the intercept initialization from the empirical prevalence to reduce bad local behavior on rare labels, while preserving the same semantics and output format. The result still runs fast, reads the same files, and writes a valid `submission.csv`.'
- What this solution (achieved 0.55994) has done: 'Your current 1D IRLS logistic setup is fine; the main issue is that this feature (`log1p(n_slices)`) is weak, and the model can overreact for rare labels, hurting weighted logloss—especially for the heavily weighted `patient_overall`. I keep the exact same core pipeline (same feature, same per-label 1D logistic via IRLS, same submission construction), but make two minimal, score-relevant changes: (1) tune regularization strength per label (stronger for rare C-levels, lighter for `patient_overall`) to improve calibration/generalization, and (2) enforce `patient_overall >= max(C1..C7)` at the per-study prediction stage to make outputs logically consistent and reduce high-penalty contradictions. These changes are lightweight, deterministic, and only adjust probability computation/post-processing (no new data sources, no architecture/training-loop changes), and should move logloss down from 0.56104 toward your 0.2657 target.'
- What this solution (achieved 0.55972) has done: 'Your score is far worse than the target (lower is better), and the main lever without changing the core approach is to improve the per-label logistic calibration so it generalizes better from only ~200 studies. I keep the same single feature (`log1p(n_slices)`), the same IRLS/Newton logistic fitting, and the same submission construction, but I (1) fit on the *training fold predictions* via a deterministic leave-one-out (LOO) procedure for each label to reduce optimistic bias, then refit on full data for test-time, and (2) learn a tiny per-label probability blend `p_final = alpha*p_model + (1-alpha)*p_prior` (alpha chosen by minimizing in-sample logloss on LOO preds) to stabilize rare-label behavior and especially `patient_overall` (heavily weighted). I keep your logical constraint `patient_overall >= max(C1..C7)` and keep clipping for logloss safety. These changes are minimal, deterministic, fast (202 studies), and targeted at lowering logloss toward your target.'
- What this solution (achieved 0.55972) has done: 'Your current score (0.55972, lower-is-better) is far from the target (0.26569), and most of that gap comes from using a single weak per-study feature (`log1p(n_slices)`) with only ~200 training studies. Keeping the same core logic (count slices → 1D logistic via IRLS per label → blend with a prior → enforce `patient_overall >= max(C1..C7)`), the smallest meaningful improvement is to (1) optimize each label’s blend weight `alpha` on the *true weighted objective* used by Kaggle (patient_overall weighted higher), and (2) use a more robust LOO procedure that reuses cached sufficient statistics so it’s deterministic and less noisy while staying within time. These changes only affect calibration/post-processing and are aimed at reducing weighted logloss toward your target without changing the fundamental modeling approach or data sources.'
- What this solution (achieved 0.55972) has done: 'Your current score (0.55972, lower-is-better) is far from the target (0.26569), so we need a meaningful improvement while still keeping your core pipeline intact (same single feature `log1p(n_slices)`, same per-label 1D IRLS logistic, same blending-with-prior idea, same submission construction). The biggest likely issue is that your LOO-based alpha tuning is optimizing a surrogate objective that doesn’t match the Kaggle weighting/aggregation (your `total_weighted_multilabel_logloss` averages per-label losses then divides by sum of label weights, which is not the same as Kaggle’s row-wise weighted average across all label-rows). I change only the alpha-selection objective to match the competition’s weighted row-wise logloss (patient_overall weight 2, others 1), keeping everything else the same. Additionally, I keep your logical constraint `patient_overall >= max(C1..C7)` but apply it during LOO evaluation too (only inside the alpha-tuning objective) so the tuned alphas reflect the same post-processing used at inference.'
- What this solution (achieved 0.55972) has done: 'Your current pipeline is dominated by a very weak feature (`log1p(n_slices)`) and small-data instability, so the biggest safe gain (without changing the model family) is to compute `patient_overall` in a way that better matches the label definition and metric weighting. I keep your per-label 1D IRLS logistic + prior blending intact for C1–C7, but derive `patient_overall` from the predicted C-level probabilities using a correlation-adjusted noisy-OR, then blend that with the direct `patient_overall` logistic prediction (still the same 1D logistic core). I tune only the `patient_overall` blend weight on the correct Kaggle row-wise weighted logloss using your existing LOO predictions, and I apply the same patient_overall post-processing consistently during tuning and inference. This is a minimal, metric-aligned change targeted specifically at lowering the heavily-weighted `patient_overall` loss, which should move your score down toward the target.'
- What this solution (achieved 0.55974) has done: 'Your current score (0.55972, lower-is-better) is far from the target, and the biggest issue is that `n_slices` is being computed with a very slow Python loop over `os.listdir`, which both risks timeouts and gives a noisy/unstable feature (especially when non-`.dcm` files appear). I keep the exact same model family (1D IRLS logistic per label, LOO blending, same patient_overall construction), but make two minimal, score-relevant fixes: (1) compute slice counts robustly and deterministically from numeric DICOM filenames (so the feature is cleaner), and (2) cache LOO fits per label by precomputing leave-one-out sufficient statistics for 1D IRLS (same Newton updates, same objective) to reduce numerical noise and make alpha tuning more reliable. These changes preserve the core approach and semantics, but should reduce logloss by stabilizing the only input feature and making calibration closer to what the tuning expects. The script still runs end-to-end and writes a valid `submission.csv` with correct `row_id` alignment.'
- What this solution (achieved 0.55972) has done: 'I keep your exact pipeline (1D `log1p(n_slices)` feature + per-label IRLS logistic + LOO alpha blending + patient_overall built from any(C) and direct model) but fix two score-relevant issues that are currently limiting performance. First, your slice counting sometimes returns the *maximum DICOM index* rather than the *number of slices*, which injects label noise into the only feature; I make it consistently count `.dcm` files (fast, deterministic) while still being robust to naming. Second, your LOO tuning for C1–C7 currently evaluates logloss using the *true* patient_overall labels but leaves the *patient_overall prediction* fixed; I instead tune C-level alphas on a C-only objective (since you later tune patient_overall separately), so the optimization signal matches what’s being tuned and reduces unintended interactions. These are minimal changes confined to feature computation and the alpha-tuning objective, and the script still run end-to-end and write `submission.csv` with the required schema.'

# 9. Code solution

## === cell 0
import os
import math
import numpy as np
import pandas as pd

DATA_ROOT = "/kaggle/data"
COMP_DIR = os.path.join(DATA_ROOT, "rsna-2022-cervical-spine-fracture-detection")

TRAIN_CSV = os.path.join(COMP_DIR, "train.csv")
TEST_CSV = os.path.join(COMP_DIR, "test.csv")
SAMPLE_SUB = os.path.join(COMP_DIR, "sample_submission.csv")

TRAIN_IMG_DIR = os.path.join(COMP_DIR, "train_images")
TEST_IMG_DIR = os.path.join(COMP_DIR, "test_images")

SAVE_CSV = "submission.csv"

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(TEST_CSV), f"Missing {TEST_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"

print("Using data from:", COMP_DIR)
print("Will write:", os.path.abspath(SAVE_CSV))



## === cell 1
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)[["row_id"]]

label_cols = ["patient_overall", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]
c_cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]

print(
    "Train shape:",
    train_df.shape,
    "Test rows:",
    test_df.shape,
    "Sample rows:",
    sample_df.shape,
)




## === cell 2
def count_slices_for_uids(img_root: str, uids):
    out = {}
    for uid in uids:
        d = os.path.join(img_root, uid)
        try:
            files = os.listdir(d)
        except FileNotFoundError:
            out[uid] = 0
            continue

        n = 0
        for f in files:
            if f.lower().endswith(".dcm"):
                n += 1
        out[uid] = int(n)
    return out


train_uids = train_df["StudyInstanceUID"].tolist()
test_uids = test_df["StudyInstanceUID"].unique().tolist()

train_slice_count = count_slices_for_uids(TRAIN_IMG_DIR, train_uids)
test_slice_count = count_slices_for_uids(TEST_IMG_DIR, test_uids)

train_df["n_slices"] = (
    train_df["StudyInstanceUID"].map(train_slice_count).fillna(0).astype(np.int32)
)
test_uid_to_slices = pd.Series(test_slice_count, name="n_slices")

print(
    "n_slices train min/med/max:",
    int(train_df["n_slices"].min()),
    float(train_df["n_slices"].median()),
    int(train_df["n_slices"].max()),
)
print(
    "n_slices test min/med/max:",
    int(test_uid_to_slices.min()),
    float(test_uid_to_slices.median()),
    int(test_uid_to_slices.max()),
)




## === cell 3
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def fit_logistic_1d_irls(x, y, l2=1.0, iters=50, laplace=1.0):
    """
    Logistic regression on 1D feature x with bias b using IRLS/Newton steps.
    Minimize mean NLL + 0.5*l2*w^2 (do not regularize bias).
    Returns (b, w).
    """
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    p0 = (y.sum() + laplace) / (len(y) + 2.0 * laplace)
    p0 = float(np.clip(p0, 1e-6, 1 - 1e-6))
    b = math.log(p0 / (1.0 - p0))
    w = 0.0

    for _ in range(iters):
        z = b + w * x
        p = sigmoid(z)

        r = np.clip(p * (1.0 - p), 1e-6, None)
        gb = (p - y).mean()
        gw = ((p - y) * x).mean() + l2 * w

        hbb = r.mean()
        hbw = (r * x).mean()
        hww = (r * x * x).mean() + l2

        det = hbb * hww - hbw * hbw
        if not np.isfinite(det) or det <= 1e-12:
            break

        db = (hww * gb - hbw * gw) / det
        dw = (-hbw * gb + hbb * gw) / det

        b -= db
        w -= dw

        if max(abs(db), abs(dw)) < 1e-8:
            break

    return float(b), float(w)


def predict_logistic_1d(x, b, w):
    x = x.astype(np.float64)
    return sigmoid(b + w * x)


def weighted_logloss_binary(y_true, p_pred, w=1.0, eps=1e-6):
    y_true = y_true.astype(np.float64)
    p_pred = np.clip(p_pred.astype(np.float64), eps, 1.0 - eps)
    return float(
        w * (-(y_true * np.log(p_pred) + (1.0 - y_true) * np.log(1.0 - p_pred))).mean()
    )


def kaggle_rowwise_weighted_logloss(
    y_true_by_label, p_pred_by_label, weight_by_label, eps=1e-6
):
    tot = 0.0
    wsum = 0.0
    for c, y in y_true_by_label.items():
        w = float(weight_by_label.get(c, 1.0))
        p = np.clip(p_pred_by_label[c].astype(np.float64), eps, 1.0 - eps)
        y = y.astype(np.float64)
        tot += w * (-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))).mean()
        wsum += w
    return float(tot / max(wsum, 1e-12))


def noisy_or_any(p_mat, gamma=1.0, eps=1e-6):
    """
    p_mat: (n, 7) probabilities for C1..C7
    gamma>1 reduces overconfidence when C-levels are correlated (common in this dataset).
    """
    p = np.clip(p_mat.astype(np.float64), eps, 1.0 - eps)
    log_q = np.log1p(-p)  # log(1-p)
    any_p = 1.0 - np.exp(np.clip(gamma * log_q.sum(axis=1), -50, 0))
    return np.clip(any_p, eps, 1.0 - eps)


x_train_raw = np.log1p(train_df["n_slices"].to_numpy(dtype=np.float64))
x_test_raw = np.log1p(test_uid_to_slices.to_numpy(dtype=np.float64))

x_mean = float(x_train_raw.mean())
x_std = float(x_train_raw.std())
if not np.isfinite(x_std) or x_std <= 1e-12:
    x_std = 1.0

x_train = (x_train_raw - x_mean) / x_std
x_test_uid = (x_test_raw - x_mean) / x_std

prevalence = {c: float(train_df[c].mean()) for c in label_cols}

l2_by_label = {"patient_overall": 0.5}
for c in c_cols:
    p = prevalence[c]
    if p < 0.03:
        l2_by_label[c] = 10.0
    elif p < 0.06:
        l2_by_label[c] = 5.0
    else:
        l2_by_label[c] = 2.0


def loo_preds_for_label(x, y, l2, iters=50, laplace=1.0):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    out = np.zeros(n, dtype=np.float64)
    idx = np.arange(n)

    for i in range(n):
        m = idx != i
        b, w = fit_logistic_1d_irls(x[m], y[m], l2=l2, iters=iters, laplace=laplace)
        out[i] = predict_logistic_1d(x[i : i + 1], b, w)[0]
    return out


weight_by_label = {c: 1.0 for c in label_cols}
weight_by_label["patient_overall"] = 2.0

alphas = {}
params = {}
p0_by_label = {}
p_loo_by_label = {}
y_by_label = {}

for c in label_cols:
    y = train_df[c].to_numpy(dtype=np.float64)
    y_by_label[c] = y
    l2 = float(l2_by_label[c])

    p_loo = loo_preds_for_label(x_train, y, l2=l2, iters=50, laplace=1.0)
    p_loo_by_label[c] = p_loo

    p0 = float((y.sum() + 1.0) / (len(y) + 2.0))
    p0_by_label[c] = p0

    b, w = fit_logistic_1d_irls(x_train, y, l2=l2, iters=50, laplace=1.0)
    params[c] = (b, w)

alpha_grid = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 1.0)

for c in label_cols:
    alphas[c] = 1.0


def blended_preds_for_label(c, alpha):
    return alpha * p_loo_by_label[c] + (1.0 - alpha) * p0_by_label[c]


def apply_patient_overall_constraint(p_pred_by_label, eps=1e-6):
    out = {k: v.copy() for k, v in p_pred_by_label.items()}
    mx = np.maximum.reduce([out[c] for c in c_cols])
    out["patient_overall"] = np.maximum(out["patient_overall"], mx)
    out["patient_overall"] = np.clip(out["patient_overall"], eps, 1.0 - eps)
    return out


def c_only_weighted_logloss(y_by_label, p_pred_by_label, eps=1e-6):
    tot = 0.0
    for c in c_cols:
        y = y_by_label[c]
        p = np.clip(p_pred_by_label[c].astype(np.float64), eps, 1.0 - eps)
        y = y.astype(np.float64)
        tot += (-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))).mean()
    return float(tot / 7.0)


for _pass in range(2):
    for c in c_cols:
        best_a = None
        best_loss = None
        for a in alpha_grid:
            p_pred_by_label = {}
            for cc in c_cols:
                aa = a if (cc == c) else alphas[cc]
                p_pred_by_label[cc] = blended_preds_for_label(cc, aa)

            loss = c_only_weighted_logloss(y_by_label, p_pred_by_label, eps=1e-6)
            if best_loss is None or loss < best_loss:
                best_loss = loss
                best_a = a
        alphas[c] = float(best_a)

gamma_grid = (1.0, 1.2, 1.5, 1.8, 2.2)  # small grid, deterministic and fast
best_po = {"loss": None, "alpha_po": None, "gamma": None}

p_c_loo_tuned = np.stack(
    [blended_preds_for_label(c, alphas[c]) for c in c_cols], axis=1
)

p_po_direct = blended_preds_for_label(
    "patient_overall", 1.0
)  # pure LOO model for patient_overall
p_po_prior = np.full_like(p_po_direct, p0_by_label["patient_overall"], dtype=np.float64)

alpha_po_grid = (0.0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0)
beta_grid = (
    0.0,
    0.25,
    0.5,
    0.75,
    1.0,
)  # weight on any(C); 1.0=all any(C), 0=all direct
best_beta = None

for gamma in gamma_grid:
    p_any = noisy_or_any(p_c_loo_tuned, gamma=gamma, eps=1e-6)
    for alpha_po in alpha_po_grid:
        p_dir = alpha_po * p_po_direct + (1.0 - alpha_po) * p_po_prior
        for beta in beta_grid:
            p_pred_by_label = {c: p_c_loo_tuned[:, i] for i, c in enumerate(c_cols)}
            p_pred_by_label["patient_overall"] = beta * p_any + (1.0 - beta) * p_dir
            p_pred_by_label = apply_patient_overall_constraint(
                p_pred_by_label, eps=1e-6
            )
            loss = kaggle_rowwise_weighted_logloss(
                y_by_label, p_pred_by_label, weight_by_label, eps=1e-6
            )
            if best_po["loss"] is None or loss < best_po["loss"]:
                best_po = {
                    "loss": float(loss),
                    "alpha_po": float(alpha_po),
                    "gamma": float(gamma),
                }
                best_beta = float(beta)

alphas["patient_overall"] = best_po["alpha_po"]
patient_overall_gamma = best_po["gamma"]
patient_overall_beta = best_beta

print(
    "Tuned patient_overall: alpha_po(prior-vs-direct)=",
    alphas["patient_overall"],
    "gamma(anyC)=",
    patient_overall_gamma,
    "beta(anyC-vs-direct)=",
    patient_overall_beta,
    "LOO objective=",
    best_po["loss"],
)

for c in label_cols:
    b, w = params[c]
    p_med = float(predict_logistic_1d(np.array([np.median(x_train)]), b, w)[0])
    y = y_by_label[c]
    if c in c_cols:
        p_blend = blended_preds_for_label(c, alphas[c])
        loss_c = weighted_logloss_binary(y, p_blend, w=weight_by_label[c], eps=1e-6)
        print(
            f"{c:15s} prev={prevalence[c]:.4f} l2={l2_by_label[c]:.2f} "
            f"alpha={alphas[c]:.2f} b={b:+.4f} w={w:+.4f} p(median)={p_med:.4f} "
            f"LOO_loss(label)~{loss_c:.5f}"
        )
    else:
        loss_c = weighted_logloss_binary(
            y, blended_preds_for_label(c, 1.0), w=weight_by_label[c], eps=1e-6
        )
        print(
            f"{c:15s} prev={prevalence[c]:.4f} l2={l2_by_label[c]:.2f} "
            f"alpha_po={alphas[c]:.2f} b={b:+.4f} w={w:+.4f} p(median)={p_med:.4f} "
            f"LOO_loss(direct_only)~{loss_c:.5f}"
        )



## === cell 4
eps = 1e-4  # logloss safety clip

test_uid_arr = test_uid_to_slices.index.to_numpy()
x_test_arr = x_test_uid

uid_pred = {uid: {} for uid in test_uid_arr}

p_c_test = []
for c in c_cols:
    b, w = params[c]
    p_model = predict_logistic_1d(x_test_arr, b, w)

    y_train = train_df[c].to_numpy(dtype=np.float64)
    p0 = (y_train.sum() + 1.0) / (len(y_train) + 2.0)
    p = alphas[c] * p_model + (1.0 - alphas[c]) * p0

    p = np.clip(p, eps, 1.0 - eps).astype(np.float64)
    p_c_test.append(p)
    for uid, pv in zip(test_uid_arr, p):
        uid_pred[uid][c] = float(pv)

p_c_test = np.stack(p_c_test, axis=1)  # (n_test, 7)

b_po, w_po = params["patient_overall"]
p_po_model = predict_logistic_1d(x_test_arr, b_po, w_po)

y0 = train_df["patient_overall"].to_numpy(dtype=np.float64)
p0_po = float((y0.sum() + 1.0) / (len(y0) + 2.0))
p_po_direct = (
    alphas["patient_overall"] * p_po_model + (1.0 - alphas["patient_overall"]) * p0_po
)

p_po_any = noisy_or_any(p_c_test, gamma=patient_overall_gamma, eps=eps)
p_po = patient_overall_beta * p_po_any + (1.0 - patient_overall_beta) * p_po_direct
p_po = np.clip(p_po, eps, 1.0 - eps).astype(np.float64)

for uid, pv in zip(test_uid_arr, p_po):
    uid_pred[uid]["patient_overall"] = float(pv)

for uid in test_uid_arr:
    mx = 0.0
    for c in c_cols:
        v = uid_pred[uid][c]
        if v > mx:
            mx = v
    po = uid_pred[uid]["patient_overall"]
    if po < mx:
        uid_pred[uid]["patient_overall"] = float(np.clip(mx, eps, 1.0 - eps))

sub_work = test_df[["row_id", "StudyInstanceUID", "prediction_type"]].copy()


def row_pred(row):
    uid = row["StudyInstanceUID"]
    t = row["prediction_type"]
    if uid in uid_pred and t in uid_pred[uid]:
        return uid_pred[uid][t]
    if uid in uid_pred:
        return uid_pred[uid]["patient_overall"]
    return 0.5


sub_work["fractured"] = sub_work.apply(row_pred, axis=1).astype(np.float32)

sub = sample_df.merge(sub_work[["row_id", "fractured"]], on="row_id", how="left")

p_med_model = float(predict_logistic_1d(np.array([np.median(x_train)]), b_po, w_po)[0])
fill_val = float(
    np.clip(
        patient_overall_beta * float(np.clip(p_med_model, eps, 1.0 - eps))
        + (1.0 - patient_overall_beta) * float(np.clip(p0_po, eps, 1.0 - eps)),
        eps,
        1.0 - eps,
    )
)

sub["fractured"] = sub["fractured"].fillna(fill_val).astype(np.float32)

assert sub.shape[0] == sample_df.shape[0], "Row count mismatch vs sample_submission"
assert list(sub.columns) == ["row_id", "fractured"], "Submission columns mismatch"
assert sub["fractured"].between(0.0, 1.0).all(), "Probabilities out of range"

sub.to_csv(SAVE_CSV, index=False)
print("Saved submission:", SAVE_CSV)
print(sub.head())
print("Rows:", len(sub))
print(
    "fractured min/mean/max:",
    float(sub["fractured"].min()),
    float(sub["fractured"].mean()),
    float(sub["fractured"].max()),
)
