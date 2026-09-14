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

0.3304388721967922

# 6. Current score

0.57504

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5639) has done: 'I remove the hard dependency on the missing external `srccode/src` and missing nnU-Net model/plan files, which currently prevents any submission from being generated. To keep the pipeline valid end-to-end in this Kaggle environment, I add a safe fallback that produces a correctly formatted `submission.csv` by using the `sample_submission.csv` structure and a constant probability prior derived from `train.csv` (score-improving vs. all-zeros/min-score, while still minimal and legitimate). The original stage1/stage2 predictor logic is preserved and automatically be used if the required `srccode` and model/plan artifacts are present; otherwise it transparently fall back without crashing. This guarantees a valid `.csv` submission file is always written.'
- What this solution (achieved 0.55917) has done: 'Your current score (0.5639, lower-is-better) is worse than the target (0.3304), so we need a small legitimate improvement without changing the core modeling path; since the nnU-Net artifacts are absent, the only active path is the fallback prior submission. I improve that fallback by (1) applying a calibrated mapping from per-vertebra priors to `patient_overall` using the logical “any” relationship (`1 - ∏(1-pk)`) and (2) adding mild, safe probability clipping to avoid log-loss blowups, while keeping the same CSV format and I/O paths. This keeps your overall pipeline identical (still uses full model if present; otherwise fallback), but makes the fallback probabilities better aligned with the competition’s label structure and metric weighting. The changes are minimal and should move the score downward toward the target without introducing new dependencies or heavier compute.'
- What this solution (achieved 0.559) has done: 'Your current score (0.55917, lower-is-better) is still far from the target (0.33044), and since the nnU-Net path is unavailable the only way to legitimately improve is to make the fallback predictions closer to the label distribution/structure. I keep the same fallback “use train priors” core logic, but improve it minimally by (1) using per-`StudyInstanceUID` risk scaling derived from the provided `train_bounding_boxes.csv` (a weak but legitimate signal correlated with fracture prevalence), and (2) recomputing `patient_overall` from the scaled vertebra probabilities via the logical “any” formula, which better matches the label semantics and metric weighting. I also keep conservative probability clipping to avoid log-loss blowups. This stays fast (no DICOM reads), preserves the full-model branch unchanged, and only changes the fallback submission values to reduce loss.'
- What this solution (achieved 0.559) has done: 'Your current score (0.559, lower-is-better) is still far from the target (0.3304), and since the full nnU-Net path is unavailable the only active path is the fast “prior-based” fallback; we make that fallback slightly more informative without changing the overall pipeline. Specifically, we (1) actually use `train_bounding_boxes.csv` to map each *test* StudyInstanceUID to a calibrated risk multiplier (your current code computes bins but never applies them to test), and (2) smooth the multiplier toward 1.0 to avoid overconfident shifts that can hurt weighted log loss. We keep the same prior-based core logic (train priors → per-vertebra probs → patient_overall via “any”), the same I/O paths, and still default to the original full-model branch if artifacts exist. These minimal changes should move the log loss downward versus the current fallback while staying well within runtime limits.'
- What this solution (achieved 0.559) has done: 'Your current score (0.559, lower-is-better) is still far from the target (0.3304) and the full nnU-Net branch is not active, so the only legitimate way to move toward the target is to make the fallback probabilities better match the metric/label structure. I keep your existing fallback core (train priors → per-vertebra probs → patient_overall via “any”), but add a tiny “shrinkage toward the global mean” per StudyInstanceUID using information already present in `test.csv` (the empirical frequency of each `prediction_type`), then re-compute `patient_overall` from the adjusted vertebra probabilities. This remains extremely fast (no DICOM reads), doesn’t change the full-model path at all, and is a minimal, stable calibration step that should reduce weighted log loss versus the current constant-per-UID outputs. I also keep your conservative clipping to avoid log-loss explosions.'
- What this solution (achieved 3.18236) has done: 'Your current score (0.559, lower-is-better) is far above the target (0.3304), and since the nnU-Net artifacts are absent the only active path is the fast fallback; so we improve only that fallback while keeping the full-model branch unchanged. The minimal legitimate gain available without reading DICOMs is to fit a tiny calibration model on `train.csv` that maps the 7 vertebra priors to `patient_overall` (matching the “any” semantics and the heavier weight on `patient_overall`) and then use it to produce better-calibrated `patient_overall` probabilities in the submission. We keep your existing per-vertebra priors, bbox-based risk multiplier, and clipping, but replace the fixed `0.25/0.75` blend with a trained logistic mapping learned from train labels (very fast, no new dependencies). This should reduce weighted log loss mainly through better `patient_overall` calibration, moving the score downward toward the target.'
- What this solution (achieved 0.56032) has done: 'Your score is much worse than the target (lower-is-better), so we should only adjust the fallback branch (since nnU-Net artifacts aren’t present) in a way that legitimately reduces weighted log loss, especially on the heavily-weighted `patient_overall` rows. The current fallback likely got much worse because the learned `patient_overall` calibrator is numerically unstable/overfitting on only 202 studies and can push probabilities toward extremes, which is heavily penalized by log loss when wrong. I keep your same priors + bbox-based odds multiplier + “any” computation, but replace the fragile Newton-fit with a stable, bounded 1-parameter calibration for `patient_overall` (power on the “any” probability) chosen by minimizing *weighted* log loss on train. I also keep conservative clipping and ensure the submission is aligned to `test.csv` rows exactly as required.'
- What this solution (achieved 0.56289) has done: 'Your current score (0.56032, lower-is-better) is still much worse than the target (0.33044), and in this environment the nnU-Net branch is typically inactive, so the only way to legitimately move toward the target is to improve the fallback probabilities without changing the overall pipeline structure. I keep the same fallback core (train priors → optional bbox risk multiplier → patient_overall derived from “any”) but replace the fragile, train-only calibration tricks with a stable isotonic calibration for the *risk multiplier* (learned from train bbox counts → patient_overall), then apply that calibrated multiplier to vertebra odds and recompute patient_overall via the same power-on-any mapping you already use. This uses only provided CSV metadata (no DICOM reads), stays fast, and targets the most heavily weighted label via better-calibrated study-level risk. I also keep conservative clipping and preserve the full-model branch unchanged.'
- What this solution (achieved 0.56114) has done: 'We keep your full nnU-Net branch untouched and only make a small, stable improvement to the fallback (since that’s what’s scoring ~0.56). The biggest lever for weighted log loss here is better calibration of `patient_overall`, so we fit a *regularized* (L2) logistic calibration on train that maps the computed “any” probability (from the 7 vertebra probs) to `patient_overall`, avoiding extreme probabilities that hurt log loss. We also slightly tune the bbox risk multiplier clip range using a tiny grid-search on train (very fast) to reduce miscalibration while keeping the same multiplier mechanism. All changes are fast (no DICOM reads), preserve the same submission schema/paths, and should move the score downward toward the target.'
- What this solution (achieved 0.56018) has done: 'Your current score (0.56114, lower-is-better) is still far above the target (0.33044), and in this environment the nnU-Net artifacts are typically missing so only the fallback branch affects scoring. I keep the fallback’s overall structure (train priors → bbox-based odds multiplier → recompute “any” → calibrate patient_overall), but make two minimal, stability-focused changes that should reduce weighted log loss: (1) replace the train-only logistic patient_overall calibrator with a 1-parameter monotonic “power” calibration on p_any (much harder to overfit on 202 rows), and (2) tune the bbox multiplier clip range using the true *competition-weighted* objective over all 8 labels (not patient_overall-only), so the multiplier doesn’t inadvertently hurt the 7 vertebra losses. The full-model path remains untouched and still be used automatically if the required artifacts exist, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.56032) has done: 'Your current score (0.56018, lower-is-better) is still far from the target (0.33044), and since the nnU-Net artifacts are absent the fallback branch is what determines performance. With minimal changes, we can make the fallback less “flat” by learning a tiny, regularized mapping from bbox-count to a risk multiplier (instead of isotonic-on-logit which can be noisy on 202 rows) and selecting its strength by directly minimizing the *competition-weighted* logloss on train. We keep your existing core fallback structure (train priors → odds scaling per study → recompute p_any → power-calibrate patient_overall) and only swap the bbox-to-mult calibrator and tune it safely. This stays fast (CSV-only), preserves the full-model branch untouched, and should move the score downward toward the target.'
- What this solution (achieved 0.57504) has done: 'Your current score (0.56032, lower-is-better) is far above the target (0.33044), and since the nnU-Net artifacts are typically missing, the fallback branch is what determines performance. The smallest, most reliable improvement without changing your overall approach is to recalibrate the fallback probabilities using only `train.csv`: we keep your priors + bbox-based odds scaling + “any” structure, but add a tiny, strongly-regularized ridge logistic calibration for *all 8 labels* jointly (one-vs-rest), which improves log-loss by correcting miscalibration while staying stable on 202 studies. We also tune just one scalar “strength” of that calibrator (blend with baseline) by minimizing the competition-weighted log loss on train, so we move downward toward the target without overfitting or altering the full-model path. The full nnU-Net branch remains untouched and is still used automatically if artifacts exist, and the script still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import sys
import time
import math
import gc
import threading
import numpy as np
import pandas as pd
import torch

import SimpleITK as sitk

DATASET_ROOT_CANDIDATES = [
    "/kaggle/input/rsna-2022-cervical-spine-fracture-detection",
    "/kaggle/data/rsna-2022-cervical-spine-fracture-detection",
    "../input/rsna-2022-cervical-spine-fracture-detection",
]
DATASET_ROOT = next((p for p in DATASET_ROOT_CANDIDATES if os.path.exists(p)), None)
if DATASET_ROOT is None:
    raise FileNotFoundError(
        f"Could not find dataset root. Tried: {DATASET_ROOT_CANDIDATES}"
    )

print("Using DATASET_ROOT:", DATASET_ROOT)

SRC_CANDIDATES = [
    os.path.join(DATASET_ROOT, "srccode", "src"),
    "/kaggle/input/srccode/src",
    "../input/srccode/src",
]
src_path = next((p for p in SRC_CANDIDATES if os.path.isdir(p)), None)
HAS_SRCCODE = src_path is not None
if HAS_SRCCODE and src_path not in sys.path:
    sys.path.insert(0, src_path)
print("Using src_path:", src_path if HAS_SRCCODE else "(not found; using fallback)")

HAS_NNUNET_DEPS = False
if HAS_SRCCODE:
    try:
        from Utils.CommonTools.NiiIO import read_from_DICOM_dir
        from Utils.PreProcessing.resampling import sitk_dummy_3D_resample
        from Utils.CommonTools.bbox import get_bbox, extend_bbox
        from Utils.post_processing import keep_largest_cervical_cc
        from Utils.Inference.nnunet_inference import NNUnetCTPredictor
        from Utils.CommonTools.sitk_base import resample, copy_nii_info, get_nii_info

        HAS_NNUNET_DEPS = True
        print("==> Imported srccode Utils successfully")
    except Exception as e:
        HAS_NNUNET_DEPS = False
        print(
            "==> srccode found but imports failed; using fallback. Import error:",
            repr(e),
        )
else:
    print("==> srccode not found; using fallback (prior-based submission)")




## === cell 1
if HAS_NNUNET_DEPS:

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
                "min_score": [0.116, 0.01, 0.015, 0.015, 0.01, 0.02, 0.032, 0.048],
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

        def predict(self, list_test_files, num_thread=4):
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
                        print(f"    Case {case_id} score: {score}")

                    print(
                        f"    Finish this split use : {time.time() - time_start} seconds"
                    )
                    print(
                        f"    Overall use : {time.time() - overall_time_start} seconds"
                    )
                    gc.collect()




## === cell 2
time_start = time.time()

SAVE_CSV = "submission.csv"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

test_csv_path = os.path.join(DATASET_ROOT, "test.csv")
sample_sub_path = os.path.join(DATASET_ROOT, "sample_submission.csv")
train_csv_path = os.path.join(DATASET_ROOT, "train.csv")
bbox_csv_path = os.path.join(DATASET_ROOT, "train_bounding_boxes.csv")

test_df = pd.read_csv(test_csv_path)


def build_prior_submission():
    train_df = pd.read_csv(train_csv_path)

    vertebra_cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    cols = ["patient_overall"] + vertebra_cols

    priors_raw = train_df[cols].mean().to_dict()

    eps = 1e-5
    priors_raw = {k: float(np.clip(v, eps, 1.0 - eps)) for k, v in priors_raw.items()}

    def odds(p):
        p = float(np.clip(p, eps, 1.0 - eps))
        return p / (1.0 - p)

    def inv_odds(o):
        o = float(max(o, eps))
        return o / (1.0 + o)

    def comp_weighted_logloss_df(df_true, df_pred):
        weights = {
            "patient_overall": 7.0,
            "C1": 1.0,
            "C2": 1.0,
            "C3": 1.0,
            "C4": 1.0,
            "C5": 1.0,
            "C6": 1.0,
            "C7": 1.0,
        }
        loss_sum = 0.0
        w_sum = 0.0
        for c in cols:
            y = df_true[c].astype(np.float64).values
            p = np.clip(df_pred[c].astype(np.float64).values, eps, 1.0 - eps)
            w = weights[c]
            loss_sum += float(
                np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))) * w
            )
            w_sum += w
        return float(loss_sum / w_sum)

    def apply_mult_to_vertebra_probs(df_probs, mults):
        Xv = df_probs[vertebra_cols].astype(np.float64).values
        Xv = np.clip(Xv, eps, 1.0 - eps)
        mults = np.asarray(mults, dtype=np.float64).reshape(-1)
        Pv = np.empty_like(Xv, dtype=np.float64)
        for j in range(7):
            p = Xv[:, j]
            o = p / (1.0 - p)
            o2 = o * mults
            Pv[:, j] = o2 / (1.0 + o2)
        Pv = np.clip(Pv, eps, 1.0 - eps)
        out = pd.DataFrame(Pv, columns=vertebra_cols, index=df_probs.index)
        return out

    def fit_gamma_power_calibrator(train_df_):
        Xv = np.clip(train_df_[vertebra_cols].astype(np.float64).values, eps, 1.0 - eps)
        y_po = train_df_["patient_overall"].astype(np.float64).values
        p_any = 1.0 - np.prod(1.0 - Xv, axis=1)
        p_any = np.clip(p_any, eps, 1.0 - eps)

        gammas = np.concatenate(
            [
                np.linspace(0.60, 1.40, 81, dtype=np.float64),
                np.array([0.50, 0.55, 0.45, 1.50], dtype=np.float64),
            ]
        )
        gammas = np.unique(np.clip(gammas, 0.35, 2.0))

        best_gamma = 1.0
        best_loss = 1e18
        w = 7.0

        for g in gammas:
            p_po = 1.0 - np.power(1.0 - p_any, g)
            p_po = np.clip(p_po, eps, 1.0 - eps)
            loss = float(
                np.mean(-(y_po * np.log(p_po) + (1.0 - y_po) * np.log(1.0 - p_po))) * w
            )
            if loss < best_loss:
                best_loss = loss
                best_gamma = float(g)

        return {
            "gamma": float(best_gamma),
            "train_wloss_patient_overall": float(best_loss),
        }

    po_cal = fit_gamma_power_calibrator(train_df)
    print("==> Fallback patient_overall calibrator: power on p_any", po_cal)

    use_scaling = False
    uid_to_mult = {}

    def evaluate_clip_range_on_train(mult_from_cnt_func, clip_lo, clip_hi, gamma):
        train_uids = train_df["StudyInstanceUID"].values
        base_probs = train_df[vertebra_cols].copy()

        mults = np.array(
            [
                float(np.clip(mult_from_cnt_func(uid), clip_lo, clip_hi))
                for uid in train_uids
            ],
            dtype=np.float64,
        )

        Pv = apply_mult_to_vertebra_probs(base_probs, mults)

        p_any = 1.0 - np.prod(1.0 - np.clip(Pv.values, eps, 1.0 - eps), axis=1)
        p_any = np.clip(p_any, eps, 1.0 - eps)
        p_po = 1.0 - np.power(1.0 - p_any, float(gamma))
        p_po = np.clip(p_po, eps, 1.0 - eps)

        pred_df = pd.DataFrame(index=train_df.index)
        pred_df[vertebra_cols] = Pv.values
        pred_df["patient_overall"] = p_po

        return comp_weighted_logloss_df(train_df[cols], pred_df[cols])

    if os.path.exists(bbox_csv_path):
        bbox_df = pd.read_csv(bbox_csv_path)
        bbox_cnt_train = bbox_df.groupby("StudyInstanceUID").size().astype(np.float64)

        train_risk = (
            train_df.set_index("StudyInstanceUID")
            .join(bbox_cnt_train.rename("bbox_cnt"), how="left")
            .fillna({"bbox_cnt": 0.0})
        )

        x_cnt = train_risk["bbox_cnt"].astype(np.float64).values

        def mult_from_cnt_params(cnt, a, b):
            xf = math.log1p(float(cnt))
            m = math.exp(float(a) * xf + float(b))
            return float(m)

        a_grid = np.linspace(-0.25, 0.60, 18, dtype=np.float64)
        b_grid = np.linspace(-0.20, 0.20, 9, dtype=np.float64)

        clip_grid = [
            (0.90, 1.20),
            (0.85, 1.25),
            (0.80, 1.30),
            (0.75, 1.35),
        ]

        best = {"wl": 1e18, "a": 0.0, "b": 0.0, "clip": (0.80, 1.30)}

        train_uid_to_cnt = train_risk["bbox_cnt"].to_dict()

        for a in a_grid:
            for b in b_grid:

                def mult_from_uid(uid, aa=float(a), bb=float(b)):
                    return mult_from_cnt_params(train_uid_to_cnt.get(uid, 0.0), aa, bb)

                for lo, hi in clip_grid:
                    wl = evaluate_clip_range_on_train(
                        mult_from_uid, float(lo), float(hi), po_cal["gamma"]
                    )
                    if wl < best["wl"]:
                        best = {
                            "wl": wl,
                            "a": float(a),
                            "b": float(b),
                            "clip": (float(lo), float(hi)),
                        }

        a_best, b_best = best["a"], best["b"]
        clip_lo, clip_hi = best["clip"]
        print(
            "==> Tuned bbox mult params on train (competition-weighted):",
            {
                "a": a_best,
                "b": b_best,
                "clip": (clip_lo, clip_hi),
                "train_wloss": float(best["wl"]),
            },
        )

        def bbox_cnt_to_mult(cnt):
            m = mult_from_cnt_params(cnt, a_best, b_best)
            return float(np.clip(m, clip_lo, clip_hi))

        bbox_cnt_test = (
            test_df[["StudyInstanceUID"]]
            .drop_duplicates()
            .set_index("StudyInstanceUID")
            .join(bbox_cnt_train.rename("bbox_cnt"), how="left")
            .fillna({"bbox_cnt": 0.0})
        )
        uid_to_mult = {
            uid: bbox_cnt_to_mult(cnt)
            for uid, cnt in bbox_cnt_test["bbox_cnt"].to_dict().items()
        }
        use_scaling = True
        print("==> Using tuned monotonic bbox_cnt->mult map via exp(a*log1p(cnt)+b).")

    if not use_scaling:
        study_uids = test_df["StudyInstanceUID"].unique()
        uid_to_mult = {uid: 1.0 for uid in study_uids}

    base_v = {k: priors_raw[k] for k in vertebra_cols}

    uid_type_counts = (
        test_df.groupby(["StudyInstanceUID", "prediction_type"])
        .size()
        .unstack(fill_value=0)
    )
    uid_total = uid_type_counts.sum(axis=1).replace(0, 1).astype(np.float32)
    uid_type_freq = (uid_type_counts.T / uid_total.values).T
    global_type_freq = (
        test_df["prediction_type"].value_counts(normalize=True)
    ).to_dict()

    def sigmoid_np(x):
        x = np.clip(x, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-x))

    def ridge_logistic_fit_predict(X_train, y_train, X_pred, lam=20.0, iters=50):
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
        X_pred = np.asarray(X_pred, dtype=np.float64)

        n, d = X_train.shape
        X1 = np.concatenate([np.ones((n, 1), dtype=np.float64), X_train], axis=1)
        Xp1 = np.concatenate(
            [np.ones((X_pred.shape[0], 1), dtype=np.float64), X_pred], axis=1
        )
        w = np.zeros(d + 1, dtype=np.float64)

        lam_vec = np.full(d + 1, float(lam), dtype=np.float64)
        lam_vec[0] = 0.0  # do not penalize intercept

        for _ in range(int(iters)):
            z = X1 @ w
            z = np.clip(z, -30.0, 30.0)
            p = 1.0 / (1.0 + np.exp(-z))
            r = p * (1.0 - p)
            Xw = X1 * r[:, None]
            H = (X1.T @ Xw) + np.diag(lam_vec)
            g = X1.T @ (p - y_train) + lam_vec * w
            try:
                step = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(H, g, rcond=None)[0]
            w -= step
            if float(np.max(np.abs(step))) < 1e-8:
                break

        z_pred = Xp1 @ w
        z_pred = np.clip(z_pred, -30.0, 30.0)
        p_pred = 1.0 / (1.0 + np.exp(-z_pred))
        return np.clip(p_pred, eps, 1.0 - eps)

    def logit(p):
        p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
        return np.log(p / (1.0 - p))

    uid_to_probs = {}
    study_uids = test_df["StudyInstanceUID"].unique()
    for uid in study_uids:
        mult = float(uid_to_mult.get(uid, 1.0))

        probs = {}
        for c in vertebra_cols:
            p = float(base_v[c])
            o = odds(p)
            o2 = o * mult
            p2 = inv_odds(o2)
            probs[c] = float(np.clip(p2, eps, 1.0 - eps))

        freqs = uid_type_freq.loc[uid] if uid in uid_type_freq.index else pd.Series({})
        for c in vertebra_cols:
            f_uid = float(freqs.get(c, global_type_freq.get(c, 0.0)))
            f_glb = float(global_type_freq.get(c, f_uid))
            log_ratio = math.log((f_uid + 1e-6) / (f_glb + 1e-6))
            strength = float(
                np.clip(0.15 * float(sigmoid_np(log_ratio)) + 0.05, 0.05, 0.20)
            )
            probs[c] = float((1.0 - strength) * probs[c] + strength * base_v[c])
            probs[c] = float(np.clip(probs[c], eps, 1.0 - eps))

        p_any_not = 1.0
        for c in vertebra_cols:
            p_any_not *= 1.0 - probs[c]
        p_any = float(np.clip(1.0 - p_any_not, eps, 1.0 - eps))

        gamma = float(po_cal["gamma"])
        p_po = 1.0 - math.pow(1.0 - p_any, gamma)
        probs["patient_overall"] = float(np.clip(p_po, eps, 1.0 - eps))

        uid_to_probs[uid] = probs

    train_uids = train_df["StudyInstanceUID"].values
    if os.path.exists(bbox_csv_path):
        bbox_df = pd.read_csv(bbox_csv_path)
        bbox_cnt_train = bbox_df.groupby("StudyInstanceUID").size().astype(np.float64)
        train_cnt = (
            train_df[["StudyInstanceUID"]]
            .set_index("StudyInstanceUID")
            .join(bbox_cnt_train.rename("bbox_cnt"), how="left")
            .fillna({"bbox_cnt": 0.0})["bbox_cnt"]
            .to_dict()
        )
    else:
        train_cnt = {uid: 0.0 for uid in train_uids}

    if use_scaling and os.path.exists(bbox_csv_path):
        def get_train_mult(uid):
            return float(uid_to_mult.get(uid, 1.0))

    else:

        def get_train_mult(uid):
            return 1.0

    base_train_probs = pd.DataFrame(
        index=np.arange(len(train_df)), columns=cols, dtype=np.float64
    )
    for i, uid in enumerate(train_uids):
        mult = float(get_train_mult(uid))
        for c in vertebra_cols:
            p = float(base_v[c])
            o2 = odds(p) * mult
            base_train_probs.loc[i, c] = float(np.clip(inv_odds(o2), eps, 1.0 - eps))
        p_any = 1.0 - float(
            np.prod(
                1.0 - base_train_probs.loc[i, vertebra_cols].values.astype(np.float64)
            )
        )
        p_any = float(np.clip(p_any, eps, 1.0 - eps))
        base_train_probs.loc[i, "patient_overall"] = float(
            np.clip(1.0 - (1.0 - p_any) ** float(po_cal["gamma"]), eps, 1.0 - eps)
        )

    X_train = np.concatenate(
        [
            logit(base_train_probs[cols].values),
            np.log1p(
                np.array(
                    [train_cnt.get(uid, 0.0) for uid in train_uids], dtype=np.float64
                )
            ).reshape(-1, 1),
        ],
        axis=1,
    )

    test_uids_unique = study_uids
    test_cnt_feat = np.zeros(len(test_uids_unique), dtype=np.float64)
    if os.path.exists(bbox_csv_path):
        test_cnt_feat[:] = 0.0
    X_test = np.zeros((len(test_uids_unique), len(cols) + 1), dtype=np.float64)
    for i, uid in enumerate(test_uids_unique):
        pvec = np.array([uid_to_probs[uid][c] for c in cols], dtype=np.float64)
        X_test[i, : len(cols)] = logit(pvec)
        X_test[i, len(cols)] = math.log1p(float(0.0))

    lam = 25.0
    iters = 60

    calib_train = pd.DataFrame(
        index=np.arange(len(train_df)), columns=cols, dtype=np.float64
    )
    calib_test_uid = pd.DataFrame(
        index=test_uids_unique, columns=cols, dtype=np.float64
    )

    for c in cols:
        y_c = train_df[c].astype(np.float64).values
        p_train_c = ridge_logistic_fit_predict(
            X_train, y_c, X_train, lam=lam, iters=iters
        )
        p_test_c = ridge_logistic_fit_predict(
            X_train, y_c, X_test, lam=lam, iters=iters
        )
        calib_train[c] = p_train_c
        calib_test_uid[c] = p_test_c

    strengths = np.linspace(0.0, 1.0, 21, dtype=np.float64)
    best_s = 0.0
    best_wl = 1e18
    for s in strengths:
        blend = (1.0 - s) * base_train_probs[cols].values + s * calib_train[cols].values
        blend = np.clip(blend, eps, 1.0 - eps)
        pred_df = pd.DataFrame(blend, columns=cols)
        wl = comp_weighted_logloss_df(train_df[cols], pred_df[cols])
        if wl < best_wl:
            best_wl = float(wl)
            best_s = float(s)

    print(
        "==> Ridge calibrator tuned blend strength on train (competition-weighted):",
        {"strength": best_s, "train_wloss": best_wl},
    )

    for uid in test_uids_unique:
        for c in cols:
            p0 = float(uid_to_probs[uid][c])
            p1 = float(calib_test_uid.loc[uid, c])
            uid_to_probs[uid][c] = float(
                np.clip((1.0 - best_s) * p0 + best_s * p1, eps, 1.0 - eps)
            )

    sub_pred = np.zeros(len(test_df), dtype=np.float32)
    for i, (uid, ptype) in enumerate(
        zip(test_df["StudyInstanceUID"].values, test_df["prediction_type"].values)
    ):
        sub_pred[i] = uid_to_probs[uid][ptype]

    sub = pd.DataFrame({"row_id": test_df["row_id"].values, "fractured": sub_pred})
    sub["fractured"] = sub["fractured"].astype(np.float32).clip(eps, 1.0 - eps)
    sub.to_csv(SAVE_CSV, index=False)

    print(
        "==> Fallback prior submission created (priors + tuned bbox risk scaling + power patient_overall + ridge calibration blend)."
    )
    print(
        "    Base priors:",
        {k: priors_raw[k] for k in (["patient_overall"] + vertebra_cols)},
    )
    if use_scaling:
        mult_vals = np.array(list(uid_to_mult.values()), dtype=np.float32)
        print(
            "    use_scaling_from_bboxes: True; mult stats:",
            {
                "min": float(mult_vals.min()),
                "p10": float(np.quantile(mult_vals, 0.10)),
                "median": float(np.quantile(mult_vals, 0.50)),
                "p90": float(np.quantile(mult_vals, 0.90)),
                "max": float(mult_vals.max()),
            },
        )
    else:
        print("    use_scaling_from_bboxes: False")
    return sub


MODEL_ROOT_CANDIDATES = [
    os.path.join(DATASET_ROOT, "models", "models"),
    "/kaggle/input/models/models",
    "../input/models/models",
]
PLAN_ROOT_CANDIDATES = [
    os.path.join(DATASET_ROOT, "plans-nnunet"),
    "/kaggle/input/plans-nnunet",
    "../input/plans-nnunet",
]
MODEL_ROOT = next((p for p in MODEL_ROOT_CANDIDATES if os.path.isdir(p)), None)
PLAN_ROOT = next((p for p in PLAN_ROOT_CANDIDATES if os.path.isdir(p)), None)

USE_FULL_MODEL = bool(
    HAS_NNUNET_DEPS and MODEL_ROOT is not None and PLAN_ROOT is not None
)
if not USE_FULL_MODEL:
    sub = build_prior_submission()
else:
    DATA_DIR = os.path.join(DATASET_ROOT, "test_images")

    list_model_C1_C7_segmentation = [
        os.path.join(MODEL_ROOT, "stage1_0.model"),
        os.path.join(MODEL_ROOT, "stage1_1.model"),
        os.path.join(MODEL_ROOT, "stage1_2.model"),
    ]
    plan_C1_C7_segmentation = os.path.join(PLAN_ROOT, "stage1.pkl")

    list_model_fracture_detection = [
        os.path.join(MODEL_ROOT, "stage2_0.model"),
        os.path.join(MODEL_ROOT, "stage2_1.model"),
        os.path.join(MODEL_ROOT, "stage2_2.model"),
        os.path.join(MODEL_ROOT, "stage2_3.model"),
        os.path.join(MODEL_ROOT, "stage2_4.model"),
    ]
    plan_fracture_detection = os.path.join(PLAN_ROOT, "stage2.pkl")

    required_files = (
        list_model_C1_C7_segmentation
        + list_model_fracture_detection
        + [
            plan_C1_C7_segmentation,
            plan_fracture_detection,
        ]
    )
    missing = [p for p in required_files if not os.path.exists(p)]
    if missing:
        print(
            "==> Model/plan artifacts missing; using fallback prior submission. Missing examples:",
            missing[:3],
        )
        sub = build_prior_submission()
    else:
        with torch.no_grad():
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

            list_DICOM_dirs = sorted(os.listdir(DATA_DIR))
            list_DICOM_dirs = [
                os.path.join(DATA_DIR, sub_dir) for sub_dir in list_DICOM_dirs
            ]
            print(f"==> Total {len(list_DICOM_dirs)} cases")

            list_size_DICOM_dirs = []
            for case_dir in list_DICOM_dirs:
                size_ = 0
                for file in os.listdir(case_dir):
                    size_ += os.path.getsize(os.path.join(case_dir, file))
                list_size_DICOM_dirs.append(size_)

            list_DICOM_dirs = list(
                np.array(list_DICOM_dirs)[np.argsort(list_size_DICOM_dirs)[::-1]]
            )
            print(f"==> Sort DICOM dirs by size; first: {list_DICOM_dirs[0]}")

            c2f_predictor.predict(list_test_files=list_DICOM_dirs)
            results = c2f_predictor.results

            label_map = {
                "patient_overall": 0,
                "C1": 1,
                "C2": 2,
                "C3": 3,
                "C4": 4,
                "C5": 5,
                "C6": 6,
                "C7": 7,
            }

            def get_pred(uid, pred_type):
                s = results.get(uid, None)
                if s is None:
                    idx = label_map[pred_type]
                    return float(c2f_predictor.params["min_score"][idx])
                return float(s[label_map[pred_type]])

            sub = pd.DataFrame(
                {
                    "row_id": test_df["row_id"].values,
                    "fractured": [
                        get_pred(uid, ptype)
                        for uid, ptype in zip(
                            test_df["StudyInstanceUID"].values,
                            test_df["prediction_type"].values,
                        )
                    ],
                }
            )
            sub["fractured"] = (
                sub["fractured"].astype(np.float32).clip(1e-6, 1.0 - 1e-6)
            )
            sub.to_csv(SAVE_CSV, index=False)

print(f"==> Wrote {SAVE_CSV} with shape {sub.shape}")
print(f"==> Finish using time: {time.time() - time_start}")
