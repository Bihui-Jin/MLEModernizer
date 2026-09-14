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

0.3313313428692946

# 6. Current score

0.58106

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.00166) has done: 'The timeout is almost certainly dominated by DICOM decoding and per-case SimpleITK conversions in the fallback path (and in the private nnUNet path, if enabled). I keep the exact prediction logic intact but remove repeated heavy work: (1) avoid reading pixel data at all (metadata-only DICOM read) and compute the exact same fallback probabilities (the priors already used on exceptions), (2) replace the per-UID boolean masking loop when building the submission with a vectorized join, and (3) reduce Python overhead in directory listing and mapping. These changes are provably equivalent to what the current code already does in many cases (exception path), preserve evaluation semantics, and should bring runtime comfortably under 600s.'
- What this solution (achieved 0.56293) has done: 'Your current score (1.00166, lower-is-better) is far worse than the target (0.3313), so we need a real (but still minimal) improvement rather than runtime-only tweaks. Keeping the same fallback “no model” core logic, I replace the fixed priors with out-of-fold (OOF) priors learned from `train.csv` per label (patient_overall, C1–C7), then apply a tiny amount of probability shrinkage toward 0.5 to reduce logloss blow-ups; this preserves the “constant-per-label” prediction style while moving probabilities closer to the empirical label rates. I also compute `patient_overall` as `max(patient_overall_prior, max(C1..C7 priors))` to respect label hierarchy, which typically helps logloss on the heavier-weighted overall label. Submission writing stays the same schema and paths and still run fast and end-to-end.'
- What this solution (achieved 0.56004) has done: 'We need to lower logloss from 0.56293 toward 0.33133; since the fallback predicts constant per-label priors, the safest minimal improvement is to use better-estimated priors and slightly better calibration without changing the core “constant priors” logic. I replace the current pseudo-OOF (which doesn’t actually randomize and can be noisy with only 202 rows) with a simple Bayesian-smoothed prior per label (Beta prior / Laplace smoothing), which is equivalent in spirit but yields more stable probabilities for small data and typically reduces logloss. I also add a tiny hierarchy-consistent adjustment so `patient_overall` is computed from the vertebra priors via `1 - Π(1 - Ck)` and then max’d with the smoothed `patient_overall` rate; this preserves semantics (still constant per label) but better matches the label definition and usually improves the heavily-weighted overall term. Finally, I keep clipping/shrinkage minimal and deterministic, and keep submission schema/paths unchanged.'
- What this solution (achieved 0.56251) has done: 'Your current score (0.56004, lower-is-better) is still far above the target (0.33133), so we should safely improve calibration while keeping the same “constant-per-label priors” fallback core logic. I (1) compute label priors with a slightly stronger Bayesian smoothing to reduce small-sample noise, and (2) tune the tiny shrinkage-to-0.5 strength downward (less shrink) because excessive shrink tends to hurt logloss when the true class imbalance is strong, and (3) keep the hierarchy-consistent `patient_overall` aggregation but compute it after shrink to avoid inflating it too much. These are minimal changes that only affect the constant probabilities used for submission, keeping the overall approach identical and deterministic. The script still run end-to-end and write `submission.csv` with the required columns.'
- What this solution (achieved 0.57003) has done: 'To move logloss down toward the 0.3313 target (lower is better) while keeping your “constant-per-label priors” core logic intact, I only adjust how those priors are estimated and calibrated. Specifically, I compute vertebra priors and `patient_overall` from a small K-fold out-of-fold (OOF) scheme (still constant per label at inference) to reduce optimism/overfitting from using full-train rates on just 202 studies. I also replace the fixed shrink-to-0.5 with a tiny, label-aware logit shrink (stronger for rarer labels) and keep the hierarchy-consistent `patient_overall = max(empirical_overall, 1-∏(1-Ck))`. All I/O paths and submission schema remain identical, and runtime stays fast because we still do no DICOM decoding in fallback mode.'
- What this solution (achieved 0.58452) has done: 'We keep your core “constant-per-label prior” fallback logic intact, but fix the main reason your logloss is stuck around ~0.57: the submission probabilities are not tuned to the competition’s *weighted* logloss where `patient_overall` matters more. The smallest legitimate improvement is to fit a single scalar logit temperature (global calibration) and a single scalar multiplier for the `patient_overall` logit using out-of-fold predictions on `train.csv`, then apply those scalars to the constant priors at inference. This does not change the modeling approach (still constant priors), but better matches the evaluation weighting and typically reduces loss meaningfully. All paths/I/O stay the same and it still writes `submission.csv`.'
- What this solution (achieved 0.57003) has done: 'Your current score (0.58452, lower-is-better) is still far above the target (0.33133), so we should improve calibration while keeping the exact same “constant-per-label priors” core logic. The biggest safe win here is to fit the two existing calibration scalars (global temperature `t` and overall multiplier `g`) in a way that better matches the *weighted* logloss by learning the effective weight for `patient_overall` from the competition’s `train.csv` class ratios, instead of hardcoding 7.0. I also keep the hierarchy-consistent `patient_overall` adjustment, but make it consistent with the fitted weighting by applying it after calibration as you already do. These are minimal changes (no new model, no image decoding, same submission schema) and should move logloss down toward the target.'
- What this solution (achieved 0.58106) has done: 'We keep your core “constant-per-label priors” fallback approach intact, but adjust the calibration to better match the competition’s weighted logloss and the label hierarchy. Specifically, we (1) fit the two scalar calibration parameters (global temperature `t` and overall-logit multiplier `g`) using a stratified K-fold out-of-fold objective to reduce overfitting from calibrating on the same data, and (2) explicitly weight `patient_overall` higher during that calibration using the standard RSNA weight (7.0), which is the safest assumption for this competition and should move logloss down from ~0.57 toward the 0.33 target. No image decoding is added, no model/architecture/training loop changes occur, and the script still writes a valid `submission.csv` with the correct schema.'
- What this solution (achieved 0.56906) has done: 'To move your logloss down toward the target while preserving the exact same “constant-per-label priors + 2-scalar calibration” core logic, I only adjust the calibration objective to better match the competition’s *true row weighting* (patient_overall weighted higher) instead of using a hardcoded 7.0. Specifically, I infer an effective patient_overall weight from `train.csv` by matching the public RSNA weighting rule (weight inversely proportional to class prevalence per label, then normalized so non-overall labels average to 1), and use that weight in the existing OOF grid-search for (t, g). This is a minimal, score-relevant change: it keeps inference identical in form (still one constant probability per label), but should reduce the dominant weighted loss term for `patient_overall`. Submission writing, paths, and runtime behavior remain unchanged.'
- What this solution (achieved 0.58106) has done: 'We keep your “constant per-label priors + two-scalar (t,g) calibration” fallback logic intact, but fix the main reason it’s plateauing around ~0.57: the calibration is being fit against an inferred weighting that likely mismatches the competition’s actual weighting scheme, so it over-optimizes the wrong objective. I switch the calibration objective to the competition’s known effective weighting (patient_overall weighted 7x vs each C1–C7 weighted 1x), and keep everything else (OOF priors, logit shrink, hierarchy constraint, submission join/alignment) the same. This is a minimal change localized to `_infer_competition_like_row_weights_from_train` usage and should move logloss downward toward your 0.331 target without altering the pipeline or I/O. The script still run end-to-end quickly and write `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os
import time
import math
import gc
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import torch
import SimpleITK as sitk

SRC_PATH = "../input/srccode/src"
HAVE_PRIVATE_SRC = os.path.isdir(SRC_PATH)

if HAVE_PRIVATE_SRC:
    import sys

    if SRC_PATH not in sys.path:
        sys.path.insert(0, SRC_PATH)
    print("==> Found private src, attempting imports:", SRC_PATH)
    try:
        from Utils.CommonTools.NiiIO import read_from_DICOM_dir
        from Utils.PreProcessing.resampling import sitk_dummy_3D_resample
        from Utils.CommonTools.bbox import get_bbox, extend_bbox
        from Utils.post_processing import keep_largest_cervical_cc
        from Utils.Inference.nnunet_inference import NNUnetCTPredictor
        from Utils.CommonTools.sitk_base import resample, copy_nii_info, get_nii_info

        HAVE_PRIVATE_IMPORTS = True
        print("==> Private imports success")
    except Exception as e:
        HAVE_PRIVATE_IMPORTS = False
        print(
            "==> Private src found but imports failed; will fallback. Error:", repr(e)
        )
else:
    HAVE_PRIVATE_IMPORTS = False
    print("==> Private src not found; will fallback:", SRC_PATH)


def _safe_read_dicom_volume(dicom_dir: str) -> sitk.Image:
    """
    Robust DICOM series reader via SimpleITK.
    """
    reader = sitk.ImageSeriesReader()
    series_IDs = reader.GetGDCMSeriesIDs(dicom_dir)
    if not series_IDs:
        raise FileNotFoundError(f"No DICOM series found in {dicom_dir}")
    series_file_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_IDs[0])
    reader.SetFileNames(series_file_names)
    img = reader.Execute()
    return img


def _extract_features_from_volume(ct_nii: sitk.Image, max_slices: int = 64) -> dict:
    """
    Very light feature extraction to keep runtime within limits.
    Uses a subset of axial slices and simple intensity statistics.
    """
    arr = sitk.GetArrayFromImage(ct_nii)  # z,y,x
    z = arr.shape[0]
    if z <= 0:
        return {"mean": -1024.0, "std": 0.0, "p99": -1024.0, "bone_frac": 0.0}

    if z > max_slices:
        idx = np.linspace(0, z - 1, max_slices).astype(np.int32)
        sub = arr[idx]
    else:
        sub = arr

    sub = sub.astype(np.float32)

    mean = float(np.mean(sub))
    std = float(np.std(sub))
    p99 = float(np.percentile(sub, 99.0))

    bone_frac = float(np.mean(sub > 300.0))
    return {"mean": mean, "std": std, "p99": p99, "bone_frac": bone_frac}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _calibrated_probs_from_features(feat: dict) -> np.ndarray:
    """
    Produce 8 probabilities: [patient_overall, C1..C7]
    Conservative calibration to avoid extreme logloss.
    """
    base_overall = 0.08
    base_level = np.array([0.03] * 7, dtype=np.float32)

    adj = 0.0
    adj += 1.5 * (feat["bone_frac"] - 0.08)  # centered
    adj += 0.002 * (feat["p99"] - 800.0)  # centered

    adj = float(np.clip(adj, -0.75, 0.75))

    p_overall = _sigmoid(math.log(base_overall / (1 - base_overall)) + adj)
    p_levels = []
    for i in range(7):
        pi = _sigmoid(
            math.log(float(base_level[i]) / (1 - float(base_level[i]))) + 0.6 * adj
        )
        p_levels.append(pi)

    max_level = max(p_levels) if p_levels else 0.0
    p_overall = float(np.clip(max(p_overall, max_level), 1e-4, 1 - 1e-4))

    out = np.zeros(8, dtype=np.float32)
    out[0] = p_overall
    out[1:] = np.clip(np.array(p_levels, dtype=np.float32), 1e-4, 1 - 1e-4)
    return out


if HAVE_PRIVATE_IMPORTS:

    class PredictorStage2(NNUnetCTPredictor):
        def __init__(self, *args, **kwargs):
            super(PredictorStage2, self).__init__(*args, **kwargs)

        def resampling(self, ct_nii):
            ori_spacing = ct_nii.GetSpacing()[::-1]  # to z,y,x
            ori_size = ct_nii.GetSize()[::-1]
            new_spacing = self.plan["plans_per_stage"][self.plan_stage][
                "current_spacing"
            ]

            new_size = [int(math.ceil(ori_size[0] * ori_spacing[0] / 0.8)), 224, 224]

            new_spacing[0] = 0.8
            new_spacing[1] = ori_size[1] * ori_spacing[1] / 224.0
            new_spacing[2] = ori_size[2] * ori_spacing[2] / 224.0

            do_resampling = np.any(
                np.abs(np.array(ori_spacing) - np.array(new_spacing))
                > self.resampling_tolerance
            )
            if do_resampling:
                ct_nii = sitk_dummy_3D_resample(
                    ct_nii,
                    new_spacing=new_spacing[::-1],
                    new_size=new_size[::-1],
                    interp_xy=self.resampling_mode,
                    interp_z=sitk.sitkNearestNeighbor,
                    out_dtype=self.resampling_dtype,
                    constant_value=self.resampling_constance_value,
                )
            else:
                print(
                    f"==> No necessary to do resampling ori {ori_spacing}, new: {new_spacing}"
                )
            return ct_nii

    class DICOMReader(threading.Thread):
        def __init__(self, func=read_from_DICOM_dir, args=()):
            super(DICOMReader, self).__init__()
            self.func = func
            self.args = args
            self.result = None

        def run(self):
            self.result = self.func(*self.args)

        def get_result(self):
            threading.Thread.join(self)
            return self.result

    class FractureDetector:
        def __init__(
            self, predictor_stage1, predictor_stage2, extend_roi=(5.0, 5.0, 5.0)
        ):
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

        def get_c1_c7_bbox(self, pred, image_spacing):
            c1_c7_bbox = get_bbox(pred > 0)
            if c1_c7_bbox is None:
                return None
            c1_c7_bbox = extend_bbox(
                c1_c7_bbox,
                max_shape=pred.shape,
                list_extend_length=self.extend_roi,
                spacing=image_spacing,
                approximate_method=np.ceil,
            )
            return c1_c7_bbox

        def predict_stage1(self, ct_nii):
            time_start = time.time()
            ori_nii_info = get_nii_info(ct_nii)
            ct_nii = self.predictor_stage1.resampling(ct_nii)
            print(f"                  Resampling use: {time.time() - time_start}")

            time_start = time.time()
            image = sitk.GetArrayFromImage(ct_nii)[np.newaxis]
            image = self.predictor_stage1.pre_processing(image)
            print(f"                  Pre_processing use: {time.time() - time_start}")

            time_start = time.time()
            pred = self.predictor_stage1.sliding_window_inference(image)
            print(f"                  Model forward use: {time.time() - time_start}")

            time_start = time.time()
            pred = np.argmax(pred, axis=0)
            pred = keep_largest_cervical_cc(pred, ct_nii.GetSpacing()[::-1])
            pred[pred > 7] = 0
            print(f"                  Post processing use: {time.time() - time_start}")

            time_start = time.time()
            pred_nii = sitk.GetImageFromArray(np.uint8(pred))
            pred_nii = copy_nii_info(ct_nii, pred_nii)
            pred_nii = resample(
                pred_nii,
                new_spacing=ori_nii_info["spacing"],
                new_origin=ori_nii_info["origin"],
                new_size=ori_nii_info["size"],
                new_direction=ori_nii_info["direction"],
                center_origin=None,
                interp=sitk.sitkNearestNeighbor,
                dtype=sitk.sitkUInt8,
                constant_value=0,
            )
            pred = sitk.GetArrayFromImage(pred_nii)
            print(f"                  Resampling back use: {time.time() - time_start}")
            return pred

        def predict_stage2(self, ct_nii):
            time_start = time.time()
            ori_nii_info = get_nii_info(ct_nii)
            ct_nii = self.predictor_stage2.resampling(ct_nii)
            print(f"                  Resampling use: {time.time() - time_start}")

            time_start = time.time()
            image = sitk.GetArrayFromImage(ct_nii)[np.newaxis]
            image = self.predictor_stage2.pre_processing(image)
            print(f"                  Pre_processing use: {time.time() - time_start}")

            time_start = time.time()
            pred = self.predictor_stage2.sliding_window_inference(image)
            print(f"                  Model forward use: {time.time() - time_start}")

            time_start = time.time()
            pred = pred[1]  # 0 for background, 1 for foreground
            print(f"                  Post processing use: {time.time() - time_start}")

            time_start = time.time()
            pred_nii = sitk.GetImageFromArray(pred)
            pred_nii = copy_nii_info(ct_nii, pred_nii)
            pred_nii = resample(
                pred_nii,
                new_spacing=ori_nii_info["spacing"],
                new_origin=ori_nii_info["origin"],
                new_size=ori_nii_info["size"],
                new_direction=ori_nii_info["direction"],
                center_origin=None,
                interp=sitk.sitkLinear,
                dtype=sitk.sitkFloat32,
                constant_value=0.0,
            )
            pred = sitk.GetArrayFromImage(pred_nii)
            print(f"                  Resampling back use: {time.time() - time_start}")
            return pred

        def get_score(self, pred_c1_c7, pred_fracture):
            output = np.zeros(8, np.float32)  # Overall, C1-C7

            if (pred_c1_c7 is not None) and (pred_fracture is not None):
                for C_i in range(8):
                    if C_i == 0:
                        roi_fracture = pred_fracture[
                            np.logical_and(
                                pred_fracture >= self.params["alpha"][C_i],
                                pred_c1_c7 > 0,
                            )
                        ]
                    else:
                        roi_fracture = pred_fracture[
                            np.logical_and(
                                pred_fracture >= self.params["alpha"][C_i],
                                pred_c1_c7 == C_i,
                            )
                        ]

                    if len(roi_fracture) == 0:
                        output[C_i] = self.params["min_score"][C_i]
                    else:
                        output[C_i] = max(
                            self.params["min_score"][C_i],
                            min(
                                self.params["max_score"][C_i],
                                np.percentile(
                                    roi_fracture, 100 * self.params["beta"][C_i]
                                ),
                            ),
                        )
            else:
                for C_i in range(8):
                    output[C_i] = self.params["min_score"][C_i]
            return output

        @staticmethod
        def read_DICOM_multi_thread(list_DICOM_dirs):
            list_thread = []
            list_outputs = []

            for DICOM_dir in list_DICOM_dirs:
                cur_thread = DICOMReader(func=read_from_DICOM_dir, args=(DICOM_dir,))
                cur_thread.start()
                list_thread.append(cur_thread)

            for cur_thread in list_thread:
                list_outputs.append(cur_thread.get_result())
            list_thread.clear()

            return list_outputs

        def predict(self, list_test_files, num_thread=4, output_dir=None):
            with torch.no_grad():
                overall_time_start = time.time()
                num_split = math.ceil(len(list_test_files) / num_thread)
                for split_i in range(num_split):
                    cur_test_files = list_test_files[
                        num_thread * split_i : num_thread * (split_i + 1)
                    ]
                    cur_case_ids = [
                        test_file.split("/")[-1] for test_file in cur_test_files
                    ]
                    print(f"==> Predicting {split_i}: {cur_case_ids}")

                    time_start = time.time()
                    cur_ct_niis = self.read_DICOM_multi_thread(cur_test_files)
                    print(
                        f"    Finish Reading use : {time.time() - time_start} seconds"
                    )

                    time_start = time.time()
                    for case_i in range(len(cur_ct_niis)):
                        case_id = cur_case_ids[case_i]
                        ct_nii = cur_ct_niis[case_i]

                        ori_nii_info = get_nii_info(ct_nii)
                        pred_1 = self.predict_stage1(ct_nii)
                        c1_c7_bbox = self.get_c1_c7_bbox(
                            pred_1, ori_nii_info["spacing"][::-1]
                        )

                        if c1_c7_bbox is not None:
                            bz, ez, by, ey, bx, ex = c1_c7_bbox
                            roi_ct_nii = ct_nii[bx : ex + 1, by : ey + 1, bz : ez + 1]
                            roi_pred_1 = pred_1[bz : ez + 1, by : ey + 1, bx : ex + 1]
                            roi_pred_2 = self.predict_stage2(roi_ct_nii)
                        else:
                            roi_pred_1 = None
                            roi_pred_2 = None

                        if roi_pred_1 is None:
                            roi_pred_1 = np.zeros((2, 2, 2), np.uint8)
                            roi_pred_2 = np.zeros((2, 2, 2), np.float32)

                        score = self.get_score(roi_pred_1, roi_pred_2)
                        self.results[case_id] = score
                        print(f"    Score: {score}")

                    print(
                        f"    Finish this split use : {time.time() - time_start} seconds"
                    )
                    print(
                        f"    Overall use : {time.time() - overall_time_start} seconds"
                    )
                    gc.collect()

else:

    class NNUnetCTPredictor:
        def __init__(self, *args, **kwargs):
            pass

    class PredictorStage2(NNUnetCTPredictor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        p = np.clip(p, eps, 1.0 - eps)
        return np.log(p / (1.0 - p))

    def _expit(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def _oof_label_priors(
        train_csv_path: str, n_splits: int = 8, seed: int = 1337, eps: float = 1e-4
    ) -> dict:
        df = pd.read_csv(train_csv_path)
        cols = ["patient_overall", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]
        y = df[cols].astype(np.float32).to_numpy()
        n = y.shape[0]

        rng = np.random.RandomState(seed)
        idx = np.arange(n)
        rng.shuffle(idx)

        splits = np.array_split(idx, n_splits)
        oof_p = np.zeros((n, len(cols)), dtype=np.float32)

        alpha = 1.0
        for fold_idx in range(n_splits):
            val_idx = splits[fold_idx]
            train_idx = np.setdiff1d(idx, val_idx, assume_unique=False)

            y_tr = y[train_idx]
            pos = y_tr.sum(axis=0)
            n_tr = y_tr.shape[0]
            p = (pos + alpha) / (n_tr + 2.0 * alpha)
            p = np.clip(p, eps, 1.0 - eps)
            oof_p[val_idx] = p

        pri = oof_p.mean(axis=0)
        pri = np.clip(pri, eps, 1.0 - eps)
        return {cols[i]: float(pri[i]) for i in range(len(cols))}

    def _label_aware_logit_shrink(p: np.ndarray) -> np.ndarray:
        p = p.astype(np.float32, copy=False)
        base_logit = _logit(p, eps=1e-6)
        mag = np.abs(base_logit)
        strength = 0.06 + 0.04 * (mag / (mag + 2.0))  # in [0.06, 0.10)
        new_logit = (1.0 - strength) * base_logit
        out = _expit(new_logit).astype(np.float32)
        return out

    def _weighted_logloss(
        y: np.ndarray, p: np.ndarray, w: np.ndarray, eps: float = 1e-6
    ) -> float:
        p = np.clip(p, eps, 1.0 - eps)
        return float(np.mean(-w * (y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))

    def _competition_row_weights_rsna() -> np.ndarray:
        """
        Score-relevant fix (minimal): use the RSNA competition weighting directly for calibration.
        This matches the stated metric behavior (patient_overall weighted higher).
        Keeping calibration aligned to the true weighting should reduce logloss vs inferring weights.
        """
        w = np.ones(8, dtype=np.float32)
        w[0] = 7.0  # patient_overall weight (RSNA 2022 cervical spine fracture)
        return w

    def _fit_two_scalar_calibration_oof(
        y: np.ndarray,
        p_row: np.ndarray,
        w_row: np.ndarray,
        n_splits: int = 6,
        seed: int = 1337,
        idx_overall: int = 0,
    ) -> tuple[float, float]:
        rng = np.random.RandomState(seed)
        n = y.shape[0]
        idx = np.arange(n)

        y_overall = y[:, idx_overall].astype(np.int32)
        idx0 = idx[y_overall == 0]
        idx1 = idx[y_overall == 1]
        rng.shuffle(idx0)
        rng.shuffle(idx1)
        folds0 = np.array_split(idx0, n_splits)
        folds1 = np.array_split(idx1, n_splits)
        folds = [np.concatenate([folds0[i], folds1[i]]) for i in range(n_splits)]

        t_grid = np.linspace(0.80, 1.30, 26, dtype=np.float32)
        g_grid = np.linspace(0.85, 1.60, 31, dtype=np.float32)

        base_logit = _logit(p_row, eps=1e-6).astype(np.float32)

        best = (1e18, 1.0, 1.0)
        for t in t_grid:
            z = base_logit * float(t)
            for g in g_grid:
                z2 = z.copy()
                z2[idx_overall] *= float(g)
                p_cal = _expit(z2).astype(np.float32)

                loss_sum = 0.0
                count = 0
                for f in folds:
                    y_f = y[f]
                    w_f = w_row[None, :].repeat(len(f), axis=0)
                    p_f = p_cal[None, :].repeat(len(f), axis=0)
                    loss_sum += _weighted_logloss(y_f, p_f, w_f, eps=1e-6) * len(f)
                    count += len(f)
                loss = loss_sum / max(1, count)

                if loss < best[0]:
                    best = (loss, float(t), float(g))
        return best[1], best[2]

    class FractureDetector:
        """
        Fallback detector that produces calibrated constant priors.
        results[StudyInstanceUID] -> np.ndarray(8): [patient_overall, C1..C7]
        """

        def __init__(
            self,
            predictor_stage1=None,
            predictor_stage2=None,
            extend_roi=(5.0, 5.0, 5.0),
            train_csv_path: str = "../input/rsna-2022-cervical-spine-fracture-detection/train.csv",
        ):
            self.predictor_stage1 = predictor_stage1
            self.predictor_stage2 = predictor_stage2
            self.extend_roi = extend_roi
            self.results = {}

            cols = ["patient_overall", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]

            pri = _oof_label_priors(
                train_csv_path=train_csv_path, n_splits=8, seed=1337, eps=1e-4
            )
            base = np.array([pri[c] for c in cols], dtype=np.float32)

            base = _label_aware_logit_shrink(base)

            p_any_from_levels = 1.0 - float(
                np.prod(1.0 - np.clip(base[1:], 1e-6, 1.0 - 1e-6))
            )
            base[0] = max(float(base[0]), p_any_from_levels, float(base[1:].max()))
            base = np.clip(base, 1e-4, 1.0 - 1e-4).astype(np.float32)

            df_tr = pd.read_csv(train_csv_path)
            y = df_tr[cols].astype(np.float32).to_numpy()

            w_row = _competition_row_weights_rsna()

            t, g = _fit_two_scalar_calibration_oof(
                y=y, p_row=base, w_row=w_row, n_splits=6, seed=1337, idx_overall=0
            )

            logit_base = _logit(base, eps=1e-6) * t
            logit_base[0] *= g
            base_cal = _expit(logit_base).astype(np.float32)

            p_any_from_levels2 = 1.0 - float(
                np.prod(1.0 - np.clip(base_cal[1:], 1e-6, 1.0 - 1e-6))
            )
            base_cal[0] = max(
                float(base_cal[0]), p_any_from_levels2, float(base_cal[1:].max())
            )
            self._FALLBACK_PRIORS = np.clip(base_cal, 1e-4, 1.0 - 1e-4).astype(
                np.float32
            )

            print(
                "==> [Fallback] calibration weights:",
                {cols[i]: float(w_row[i]) for i in range(8)},
            )
            print("==> [Fallback] fitted (t,g):", float(t), float(g))
            print(
                "==> [Fallback] priors (post-cal):",
                {cols[i]: float(self._FALLBACK_PRIORS[i]) for i in range(8)},
            )

        def predict(self, list_test_files, num_thread=32, output_dir=None):
            t0 = time.time()
            for i, dicom_dir in enumerate(list_test_files, 1):
                case_id = os.path.basename(dicom_dir)
                self.results[case_id] = self._FALLBACK_PRIORS
                if i % 500 == 0 or i == len(list_test_files):
                    print(
                        f"==> [Fallback] Done {i}/{len(list_test_files)}. Elapsed: {time.time() - t0:.1f}s"
                    )
            gc.collect()




## === cell 1
time_start = time.time()

BASE_DIR = "../input/rsna-2022-cervical-spine-fracture-detection"
DATA_DIR = f"{BASE_DIR}/test_images"
TEST_CSV = f"{BASE_DIR}/test.csv"
SAVE_CSV = "submission.csv"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("==> device:", device)

with torch.no_grad():
    if HAVE_PRIVATE_IMPORTS:
        list_model_C1_C7_segmentation = [
            "../input/models/models/stage1_0.model",
            "../input/models/models/stage1_1.model",
            "../input/models/models/stage1_2.model",
        ]
        plan_C1_C7_segmentation = "../input/plans-nnunet/stage1.pkl"

        list_model_fracture_detection = [
            "../input/models/models/stage2_0.model",
            "../input/models/models/stage2_1.model",
            "../input/models/models/stage2_2.model",
            "../input/models/models/stage2_3.model",
            "../input/models/models/stage2_4.model",
        ]
        plan_fracture_detection = "../input/plans-nnunet/stage2.pkl"

        predictor_1 = NNUnetCTPredictor(
            list_model_pth=list_model_C1_C7_segmentation,
            plan_file=plan_C1_C7_segmentation,
            plan_stage=-1,
            device=device,
            use_gaussian_for_sliding_window=True,
            patch_size=None,
            stride=None,
            tta=False,
            tta_flip_axis=(4,),
            resampling_tolerance=0.01,
            resampling_mode=sitk.sitkNearestNeighbor,
            resampling_dtype=sitk.sitkInt16,
            resampling_constance_value=-1024,
            remove_air_CT=True,
        )
        predictor_2 = PredictorStage2(
            list_model_pth=list_model_fracture_detection,
            plan_file=plan_fracture_detection,
            plan_stage=-1,
            device=device,
            use_gaussian_for_sliding_window=True,
            patch_size=(96, 224, 224),
            stride=(96, 224, 224),
            tta=True,
            tta_flip_axis=(4,),
            resampling_tolerance=0.01,
            resampling_mode=sitk.sitkNearestNeighbor,
            resampling_dtype=sitk.sitkInt16,
            resampling_constance_value=-1024,
            remove_air_CT=False,
            save_dtype=np.float32,
        )
        c2f_predictor = FractureDetector(
            predictor_stage1=predictor_1, predictor_stage2=predictor_2
        )
    else:
        c2f_predictor = FractureDetector()

    list_DICOM_dirs = []
    with os.scandir(DATA_DIR) as it:
        for e in it:
            if e.is_dir():
                list_DICOM_dirs.append(e.path)
    list_DICOM_dirs.sort()
    print(f"==> Total {len(list_DICOM_dirs)} cases in {DATA_DIR}")

    c2f_predictor.predict(list_test_files=list_DICOM_dirs)

    test_df = pd.read_csv(TEST_CSV)
    results = c2f_predictor.results

    idx_map = {
        "patient_overall": 0,
        "C1": 1,
        "C2": 2,
        "C3": 3,
        "C4": 4,
        "C5": 5,
        "C6": 6,
        "C7": 7,
    }

    uid_arr = test_df["StudyInstanceUID"].to_numpy()
    ptype_arr = test_df["prediction_type"].to_numpy()

    uniq_uids, inv = np.unique(uid_arr, return_inverse=True)

    mat = np.empty((len(uniq_uids), 8), dtype=np.float32)
    present = np.zeros(len(uniq_uids), dtype=bool)
    for i, uid in enumerate(uniq_uids):
        v = results.get(uid, None)
        if v is None:
            continue
        mat[i] = v
        present[i] = True

    if not present.all():
        if HAVE_PRIVATE_IMPORTS:
            mat[~present, 0] = 0.08
            mat[~present, 1:] = 0.03
        else:
            mat[~present] = getattr(
                c2f_predictor,
                "_FALLBACK_PRIORS",
                np.array([0.08] + [0.03] * 7, dtype=np.float32),
            )

    idxs = np.fromiter(
        (idx_map[x] for x in ptype_arr), dtype=np.int64, count=len(ptype_arr)
    )
    fractured = mat[inv, idxs]

    sub = pd.DataFrame({"row_id": test_df["row_id"].values, "fractured": fractured})
    sub["fractured"] = sub["fractured"].clip(1e-4, 1 - 1e-4)

    sub.to_csv(SAVE_CSV, index=False)
    print(
        f"==> Wrote {SAVE_CSV} with shape {sub.shape}. Missing uids: {int((~present).sum())}"
    )
    print(f"==> Finish using time: {time.time() - time_start:.1f} seconds")
