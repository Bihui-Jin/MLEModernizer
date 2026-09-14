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

0.2634027999760441

# 6. Current score

0.56549

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.75092) has done: 'I remove the hard dependency on an external `/kaggle/input/.../src` folder (which isn’t present in your environment) by adding lightweight local fallback implementations for the missing `Utils.*` and `Training.*` imports so the notebook runs end-to-end. To keep core semantics intact while making it runnable within the 600s constraint, the fallback path generate stable, reasonable probabilities from `train.csv` priors (and always construct `patient_overall` from the max of C1–C7), producing a valid `submission.csv` with the exact required columns/row order from `test.csv`. I also fix the cell numbering to start at 1 and ensure `NNUnetCTPredictor` is always defined so later classes don’t crash. These changes are primarily bug-fixes to produce a valid submission; without the original model weights/src code present, meaningful model inference isn’t possible, so the fallback focuses on correctness and metric-safe calibration.'
- What this solution (achieved 0.65148) has done: 'Your current submission is a simple prior baseline, and the score gap to the target is large (0.75092 vs 0.2634; lower is better), so we need a small but meaningful boost without changing the overall approach. I keep the “train.csv priors only” core logic, but make it metric-aware by (1) computing per-label priors with Laplace smoothing to reduce overconfident probabilities on rare labels, and (2) calibrating all priors toward 0.5 using a single shrinkage factor chosen via a quick cross-validated weighted logloss on the training set. This stays within your existing semantics (constant probabilities per label, patient_overall derived from C1–C7), but should noticeably reduce logloss. The script still run fast and write a valid `submission.csv` with the exact `row_id,fractured` format and correct row order from `test.csv`.'
- What this solution (achieved 0.65215) has done: 'Your current approach is a constant-per-label prior baseline, so the safest way to improve logloss (lower is better) without changing core semantics is to make the priors more metric-aware and better calibrated. I (1) choose the Laplace smoothing strength per label via quick CV (instead of fixed alpha=beta=1), because rare-label priors are most sensitive in logloss, and (2) replace the coarse shrinkage grid with a finer, still-fast grid to better match the optimal calibration. I also compute `patient_overall` in the same spirit but keep your invariant that it is at least the max of C1–C7 predictions per study and never violates [eps, 1-eps]. The rest of the pipeline (no image inference; constant predictions per label; submission alignment to `test.csv`) stays unchanged.'
- What this solution (achieved 0.65277) has done: 'Your current baseline is still a constant-per-label prior, so the biggest safe gain (lower logloss) comes from (1) making the cross-validation selection actually “fold-aware” (compute priors on train-fold only, not on the full dataset each time), and (2) selecting the shrinkage factor using the same fold-aware priors so it better calibrates probabilities under the weighted metric. This keeps the exact same prediction semantics (one constant probability per label; patient_overall is at least the max of C1–C7 per study) but removes a subtle CV bug that was preventing good hyperparameter selection. I also keep strict clipping and the same submission construction to ensure a valid `submission.csv` in the correct order. These are minimal changes aimed at moving score down from ~0.65 toward your ~0.26 target without changing model architecture/training loops (there are none used for inference here).'
- What this solution (achieved 0.56319) has done: 'Your current solution is a constant-per-label prior baseline; the biggest safe way to lower weighted logloss without changing the overall approach is to make the priors respect the true dependency between labels. I keep your Laplace-smoothed priors and fold-aware CV selection, but compute `patient_overall` on each fold as the Laplace-smoothed prior of the *true* `patient_overall` column (not as `max(C1..C7)`), then still enforce the submission-time constraint that `patient_overall >= max(C1..C7)` to stay logically consistent. I also include `patient_overall` in the Laplace alpha selection (same procedure as C1–C7) and use this fold-aware `patient_overall` during shrinkage selection, which should reduce the metric substantially while preserving your core “constant prediction per label” semantics. All I/O paths and the submission schema/order remain unchanged, and the script still finishes quickly and writes `submission.csv`.'
- What this solution (achieved 0.57145) has done: 'Your current method is a constant-per-label prior baseline, so the safest way to reduce weighted logloss (lower is better) without changing the overall approach is to better match the metric by calibrating *away from extremes* and by using a patient_overall rule that is consistent but not overly aggressive. I keep your fold-aware Laplace smoothing selection, but (1) switch the CV folding to be stratified on `patient_overall` to reduce variance on this heavily-weighted label, and (2) replace the simple “shrink-to-0.5” with a minimal, still-constant per-label logit scaling chosen by CV, which typically improves logloss for imbalanced labels. Finally, I keep the required constraint `patient_overall >= max(C1..C7)` but make it a *soft* aggregation via `1 - Π(1-p_level)` (then max with the base patient prior), which is still logically consistent and often yields better calibrated overall probabilities under logloss.'
- What this solution (achieved 0.57145) has done: 'I keep your constant-per-label prior baseline intact, but fix a key metric mismatch: your CV uses a “mean of per-label logloss” normalization that does not match the competition’s “average over all rows with weights applied” definition. I compute CV loss exactly the Kaggle way (sum weighted loss over all label-rows divided by total number of label-rows), then re-select the global logit scale using that corrected objective (and keep your per-label Laplace alpha selection as-is to minimize changes). This should lower your public score (lower is better) because calibration is being optimized against the right objective, without changing the prediction form (still constant per label with the same `patient_overall >= max(C1..C7)` soft-OR rule). The script still run fast and write a valid `submission.csv` with correct `row_id,fractured` and row order from `test.csv`.'
- What this solution (achieved 0.56903) has done: 'Your current baseline is still a constant-per-label prior, and the gap to the target is large, so the most “minimal but meaningful” score improvement is to make the CV objective match the competition metric exactly and then re-tune calibration against that exact objective. I (1) replace the current per-label averaged loss with a single Kaggle-style weighted-logloss computed over all 8×N rows, (2) tune a per-label logit scale (8 parameters, small grid) instead of one global scale so the heavily-weighted `patient_overall` can calibrate differently, and (3) keep your exact semantics (Laplace-smoothed priors; constant predictions per label; submission-time `patient_overall >= max(C1..C7)` via soft-OR + max). These changes are directly metric-aligned and should reduce the public logloss from ~0.57 toward your 0.263 target without changing the fundamental approach or adding slow image inference.'
- What this solution (achieved 0.56903) has done: 'I keep your “constant per-label priors + CV calibration + patient_overall constraint” core logic, but fix the weighted-logloss normalization to match the competition definition (your current `_kaggle_weighted_logloss_from_constants` divides by an extra `len(TARGETS)`, which misguides CV selection). Then I re-tune the per-label logit scales using that corrected objective so calibration is optimized against the true metric, without changing the prediction form. Finally, I clip probabilities with a slightly safer epsilon (still negligible) to avoid extreme logloss penalties and keep submission generation unchanged and valid.'
- What this solution (achieved 0.56793) has done: 'Your current solution is a constant-per-label prior baseline, so the biggest safe improvement toward the much lower target is to make the calibration step closer to the true optimum without changing the prediction form. I keep the exact same semantics (Laplace-smoothed per-label constants; per-label logit scaling; `patient_overall` enforced via soft-OR/max) but improve the hyperparameter search resolution and stability by (1) switching from a coarse fixed scale grid to a two-stage coarse-to-fine grid per label, and (2) using more stratified folds (still fast given only 202 rows) to reduce variance on the heavily-weighted `patient_overall`. This is a minimal change that directly targets the weighted logloss objective you already implemented and should move the public score down from ~0.569 toward the target band. Submission writing/format and all paths remain unchanged.'
- What this solution (achieved 0.56549) has done: 'Your current solution is already running end-to-end and producing a valid submission, but the score (0.56793, lower is better) is still far from the target (0.2634), so we need a small, metric-aligned improvement without changing the core “constant per-label priors + CV calibration + patient_overall constraint” logic. The most leverage with minimal semantic change is to make the calibrated constants better reflect the competition’s heavy weighting of `patient_overall` by (1) selecting Laplace smoothing (`alpha`) using the exact Kaggle-style *global* weighted loss (not per-label mean), and (2) jointly re-tuning only two logit-scale groups (one for `patient_overall`, one shared for C1–C7) against the exact Kaggle-style loss to reduce overfitting/noise from 8 separate scales on just 202 studies. This keeps the same prediction form (still constants by label, still patient_overall >= soft-OR/max), but should move logloss down more reliably. All paths and submission format stay unchanged, and runtime remains well under the 600s limit.'

# 9. Code solution

## === cell 0
import os
import time
import numpy as np
import pandas as pd
import torch

try:
    import SimpleITK as sitk  # noqa: F401

    _HAS_SITK = True
except Exception:
    _HAS_SITK = False


def _safe_read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


DATA_ROOT = "/kaggle/input/rsna-2022-cervical-spine-fracture-detection"
ALT_DATA_ROOT = "/kaggle/data/rsna-2022-cervical-spine-fracture-detection"
if not os.path.isdir(DATA_ROOT) and os.path.isdir(ALT_DATA_ROOT):
    DATA_ROOT = ALT_DATA_ROOT

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")

train_df = _safe_read_csv(TRAIN_CSV)
test_df = _safe_read_csv(TEST_CSV)

TARGETS = ["patient_overall", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]

eps = 1e-5


def _laplace_prior(y, alpha=1.0, beta=1.0):
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    s = float(y.sum())
    p = (s + alpha) / (n + alpha + beta)
    return float(np.clip(p, eps, 1.0 - eps))


print("Using DATA_ROOT:", DATA_ROOT)


class NNUnetCTPredictor:
    def __init__(self, *args, **kwargs):
        self.plan = {
            "plans_per_stage": {
                kwargs.get("plan_stage", -1): {"current_spacing": [1.0, 1.0, 1.0]}
            }
        }
        self.plan_stage = kwargs.get("plan_stage", -1)
        self.resampling_tolerance = kwargs.get("resampling_tolerance", 0.01)
        self.resampling_mode = kwargs.get("resampling_mode", None)
        self.resampling_dtype = kwargs.get("resampling_dtype", None)
        self.resampling_constance_value = kwargs.get(
            "resampling_constance_value", -1024
        )
        self.device = kwargs.get("device", torch.device("cpu"))

    def resampling(self, ct_nii):
        return ct_nii

    def pre_processing(self, image):
        return image

    def sliding_window_inference(self, image):
        return np.zeros((2,) + tuple(image.shape[1:]), dtype=np.float32)


class ModelStage3(torch.nn.Module):
    def __init__(self, in_ch=2, out_ch=1, list_ch=None, random_init=False):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv3d(in_ch, 8, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool3d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(8, out_ch),
        )

    def forward(self, x):
        out = self.net(x)
        out = out.view(-1, 1, 1, 1, 1)
        return (out,)




## === cell 1
import threading


class PredictorStage2(NNUnetCTPredictor):
    def __init__(self, *args, **kwargs):
        super(PredictorStage2, self).__init__(*args, **kwargs)

    def resampling(self, ct_nii):
        return ct_nii


class PredictorStage3:
    def __init__(self, list_model_pth, device, tta=False, tta_flip_axis=(4,)):
        self.list_model_pth = list_model_pth
        self.device = device
        self.tta = tta
        self.tta_flip_axis = tta_flip_axis

        self.list_model = None
        self.in_ch = 2
        self.out_ch = 1
        self.list_ch = [-1, 16, 32, 64, 128]
        self.init_model()

    def init_model(self):
        self.list_model = []
        for pth in self.list_model_pth:
            model = ModelStage3(
                in_ch=self.in_ch,
                out_ch=self.out_ch,
                list_ch=self.list_ch,
                random_init=False,
            )
            if os.path.exists(pth):
                ckpt = torch.load(pth, map_location="cpu")
                try:
                    model.load_state_dict(ckpt)
                except Exception:
                    pass
            model.eval()
            model = model.to(self.device)
            self.list_model.append(model)

    def predict(self, image):
        with torch.no_grad():
            patch_input = torch.from_numpy(image).to(self.device).unsqueeze(0)
            preds = []
            for model in self.list_model:
                pred = model(patch_input)[0]
                pred = torch.sigmoid(pred[0, 0])
                preds.append(pred.item())
            return float(np.mean(preds)) if preds else 0.5


class DICOMReader(threading.Thread):
    def __init__(self, func=None, args=()):
        super(DICOMReader, self).__init__()
        self.func = func
        self.args = args
        self.result = None

    def run(self):
        if self.func is None:
            self.result = None
        else:
            self.result = self.func(*self.args)

    def get_result(self):
        threading.Thread.join(self)
        return self.result


class FractureDetector:
    def __init__(
        self,
        predictor_stage1,
        predictor_stage2,
        predictor_stage3,
        extend_roi=(5.0, 5.0, 5.0),
    ):
        self.predictor_stage1 = predictor_stage1
        self.predictor_stage2 = predictor_stage2
        self.predictor_stage3 = predictor_stage3
        self.extend_roi = extend_roi
        self.results = {}

    def predict(self, list_test_files, num_thread=4, output_dir=None):
        return




## === cell 2
time_start = time.time()

SAVE_CSV = "submission.csv"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

pred_map = {
    "patient_overall": "patient_overall",
    "C1": "C1",
    "C2": "C2",
    "C3": "C3",
    "C4": "C4",
    "C5": "C5",
    "C6": "C6",
    "C7": "C7",
}

WEIGHTS = {
    "patient_overall": 7.0,
    "C1": 1.0,
    "C2": 1.0,
    "C3": 1.0,
    "C4": 1.0,
    "C5": 1.0,
    "C6": 1.0,
    "C7": 1.0,
}


def _weighted_logloss_vec(y_true, y_pred, weight=1.0, eps=1e-6):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    loss = -(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
    return weight * loss


def _make_stratified_folds(y_strat, n_folds=8, seed=42):
    y_strat = np.asarray(y_strat).astype(int)
    rng = np.random.default_rng(seed)
    idx0 = np.where(y_strat == 0)[0]
    idx1 = np.where(y_strat == 1)[0]
    rng.shuffle(idx0)
    rng.shuffle(idx1)
    folds0 = np.array_split(idx0, n_folds)
    folds1 = np.array_split(idx1, n_folds)
    folds = []
    for k in range(n_folds):
        f = np.concatenate([folds0[k], folds1[k]])
        rng.shuffle(f)
        folds.append(f)
    return folds


def _logit(p):
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p) - np.log1p(-p)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _scale_logit(p, s):
    return float(np.clip(_sigmoid(_logit(p) * float(s)), eps, 1.0 - eps))


def _kaggle_weighted_logloss_from_constants(y_by_label, p_by_label, weights, idx):
    total = 0.0
    n_rows = 0
    for t in TARGETS:
        w = float(weights.get(t, 1.0))
        y = y_by_label[t][idx]
        p = np.full(y.shape[0], float(p_by_label[t]), dtype=np.float64)
        total += float(_weighted_logloss_vec(y, p, weight=w, eps=eps).sum())
        n_rows += int(y.shape[0])
    return total / float(n_rows)


def _apply_patient_overall_constraint(p_label):
    level_ps = np.array([p_label[c] for c in TARGETS[1:]], dtype=np.float64)
    level_ps = np.clip(level_ps, eps, 1.0 - eps)
    soft_or = float(1.0 - np.prod(1.0 - level_ps))
    max_level = float(np.max(level_ps))
    p_label = dict(p_label)
    p_label["patient_overall"] = float(
        np.clip(max(p_label["patient_overall"], soft_or, max_level), eps, 1.0 - eps)
    )
    return p_label


def _cv_select_laplace_alpha_global(train_df, label, folds, alpha_grid=None):
    if alpha_grid is None:
        alpha_grid = np.array(
            [0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
            dtype=np.float64,
        )

    y = {t: train_df[t].values.astype(np.float64) for t in TARGETS}
    n_folds = len(folds)

    best_a = 1.0
    best_loss = float("inf")

    for a_try in alpha_grid:
        fold_losses = []
        for k in range(n_folds):
            val_idx = folds[k]
            tr_idx = np.concatenate([folds[j] for j in range(n_folds) if j != k])

            p_label = {}
            for col in TARGETS:
                if col == label:
                    p_label[col] = _laplace_prior(
                        y[col][tr_idx], alpha=float(a_try), beta=float(a_try)
                    )
                else:
                    p_label[col] = float(np.clip(y[col][tr_idx].mean(), eps, 1.0 - eps))

            p_label = _apply_patient_overall_constraint(p_label)
            fold_losses.append(
                _kaggle_weighted_logloss_from_constants(y, p_label, WEIGHTS, val_idx)
            )

        m = float(np.mean(fold_losses))
        if m < best_loss:
            best_loss = m
            best_a = float(a_try)

    return best_a, best_loss


def _cv_select_two_group_logit_scales(train_df, alpha_by_label, folds):
    y = {t: train_df[t].values.astype(np.float64) for t in TARGETS}
    n_folds = len(folds)

    coarse = np.array(
        [0.55, 0.70, 0.85, 1.00, 1.15, 1.35, 1.60, 1.90, 2.25], dtype=np.float64
    )

    def eval_loss(s_overall, s_level):
        fold_losses = []
        for k in range(n_folds):
            val_idx = folds[k]
            tr_idx = np.concatenate([folds[j] for j in range(n_folds) if j != k])

            raw = {}
            for col in TARGETS:
                a = float(alpha_by_label[col])
                raw[col] = _laplace_prior(y[col][tr_idx], alpha=a, beta=a)

            p_label = {}
            for col in TARGETS:
                s = float(s_overall) if col == "patient_overall" else float(s_level)
                p_label[col] = _scale_logit(raw[col], s)

            p_label = _apply_patient_overall_constraint(p_label)
            fold_losses.append(
                _kaggle_weighted_logloss_from_constants(y, p_label, WEIGHTS, val_idx)
            )
        return float(np.mean(fold_losses))

    best_s_overall, best_s_level = 1.0, 1.0
    best_loss = float("inf")

    for s_o in coarse:
        for s_l in coarse:
            m = eval_loss(s_o, s_l)
            if m < best_loss:
                best_loss = m
                best_s_overall, best_s_level = float(s_o), float(s_l)

    fine_overall = np.clip(
        np.linspace(best_s_overall * 0.85, best_s_overall * 1.15, 17), 0.25, 3.0
    )
    fine_level = np.clip(
        np.linspace(best_s_level * 0.85, best_s_level * 1.15, 17), 0.25, 3.0
    )

    for s_o in fine_overall:
        for s_l in fine_level:
            m = eval_loss(s_o, s_l)
            if m < best_loss:
                best_loss = m
                best_s_overall, best_s_level = float(s_o), float(s_l)

    scales = {t: best_s_level for t in TARGETS}
    scales["patient_overall"] = best_s_overall
    return scales, best_loss


folds = _make_stratified_folds(train_df["patient_overall"].values, n_folds=8, seed=42)

alpha_by_label = {}
raw_priors = {}
for col in TARGETS:
    a, a_loss = _cv_select_laplace_alpha_global(train_df, col, folds)
    alpha_by_label[col] = float(a)
    raw_priors[col] = _laplace_prior(train_df[col].values, alpha=a, beta=a)

print("Selected per-label Laplace alpha (global-metric tuned):", alpha_by_label)
print("Raw (smoothed) priors:", raw_priors)

scales_by_label, cv_loss = _cv_select_two_group_logit_scales(
    train_df, alpha_by_label, folds
)
print(
    f"Selected 2-group logit scales: overall={scales_by_label['patient_overall']:.6f}, level={scales_by_label['C1']:.6f}"
)
print(
    f"{len(folds)}-fold approx Kaggle-style weighted logloss (after 2-group scaling)={cv_loss:.6f}"
)

priors = {t: _scale_logit(raw_priors[t], float(scales_by_label[t])) for t in TARGETS}
priors = _apply_patient_overall_constraint(priors)
print("Calibrated priors:", priors)


def _get_pred_from_priors(row):
    ptype = row["prediction_type"]
    key = pred_map.get(ptype, None)
    if key is None:
        val = 0.5
    else:
        val = priors[key]
    return float(np.clip(val, eps, 1.0 - eps))


sub = pd.DataFrame(
    {
        "row_id": test_df["row_id"].values,
        "fractured": test_df.apply(_get_pred_from_priors, axis=1).astype(np.float32),
    }
)

tmp = test_df.copy()
tmp["fractured"] = sub["fractured"].values

is_level = tmp["prediction_type"].isin(["C1", "C2", "C3", "C4", "C5", "C6", "C7"])
level_df = tmp.loc[
    is_level, ["StudyInstanceUID", "fractured", "prediction_type"]
].copy()

prod_by_uid = (
    level_df.assign(one_minus=lambda d: 1.0 - d["fractured"])
    .groupby("StudyInstanceUID")["one_minus"]
    .prod()
)
soft_or_by_uid = 1.0 - prod_by_uid
max_by_uid = level_df.groupby("StudyInstanceUID")["fractured"].max()

is_overall = tmp["prediction_type"].eq("patient_overall")
overall_idx = tmp.index[is_overall]
overall_uids = tmp.loc[overall_idx, "StudyInstanceUID"].values
overall_vals = np.array(
    [
        max(
            priors["patient_overall"],
            float(soft_or_by_uid.get(uid, priors["patient_overall"])),
            float(max_by_uid.get(uid, priors["patient_overall"])),
        )
        for uid in overall_uids
    ],
    dtype=np.float32,
)

sub.loc[overall_idx, "fractured"] = np.clip(overall_vals, eps, 1.0 - eps).astype(
    np.float32
)

sub.to_csv(SAVE_CSV, index=False)
print(f"==> Wrote {SAVE_CSV} with shape {sub.shape} and columns {list(sub.columns)}")
print(sub.head(10))
print(f"==> Finish using time: {time.time() - time_start:.2f}s")
