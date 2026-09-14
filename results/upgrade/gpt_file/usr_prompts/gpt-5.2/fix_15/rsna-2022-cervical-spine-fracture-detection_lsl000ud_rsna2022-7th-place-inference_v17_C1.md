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

0.55643

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.69315) has done: 'I fix the immediate import/path failure by removing the hard dependency on a missing `../input/srccode2/src` folder and instead dynamically locating the `Utils/Training` modules in the competition dataset tree (falling back safely if not found). Then I add a “safe mode” fallback that still produces a valid `submission.csv` even if the private model/plan files aren’t present in this Kaggle environment (so you always get a valid .csv submission). Finally, I fix a few robustness issues that can crash long runs (CUDA availability, missing data directories, and ensuring predictions are numeric floats clipped to (0,1) for log-loss stability) without changing the core inference logic when the real models are available. This should make the notebook run end-to-end and, when model assets exist, yield the intended model-based predictions (and thus move score toward the target rather than failing).'
- What this solution (achieved 0.5639) has done: 'Your 0.69315 score indicates the fallback path is being used (constant 0.5 predictions), because the required model/plan assets are not present at `../input/models2/models` and `../input/plans-nnunet`. To move the score toward the 0.2789 target without changing the model/inference core logic, I replace the constant fallback with a simple, label-distribution prior learned from `train.csv` (per-label prevalence) and use it to generate probabilities for each `prediction_type`. This is a minimal, metric-aligned improvement that usually beats 0.69315 on weighted log loss, while keeping the intended custom-model path unchanged when assets exist. I also ensure strict alignment to `test.csv` row order and keep probabilities clipped for log-loss stability.'
- What this solution (achieved 0.56136) has done: 'Your current score (0.5639, lower-is-better) is still far from the target (0.2789), and the run is almost certainly using the fallback (no model assets). To move toward the target without changing your core model/inference logic, I only improve the fallback by (1) applying a simple Laplace/Beta smoothing to the per-label train prevalences (reduces overconfident priors on a tiny 202-scan train set), and (2) ensuring `patient_overall` is computed as a consistent “soft-OR” of C1–C7 (metric-aligned and typically better than max/mean for log loss). I keep all custom-code/model paths untouched; these changes only affect the fallback submission generation. The submission still be written as `submission.csv` with the exact `row_id,fractured` schema and clipped probabilities for log-loss stability.'
- What this solution (achieved 0.56037) has done: 'Your current score (0.56136, lower-is-better) is still far from the target (0.27894), and since the model assets are missing in this environment the fallback path is what determines the score. To improve the fallback while keeping the core “use real models if present” logic unchanged, I (1) compute per-label probabilities from `train.csv` using a Beta prior whose strength adapts to the tiny 202-study size (less over/under-confidence), and (2) calibrate those probabilities via a very small grid-search temperature scaling on the training set using out-of-fold predictions (no architecture/training changes; just better-calibrated constants for log loss). I also keep `patient_overall` consistent by recomputing it as a soft-OR of the calibrated C1–C7 probabilities, which is metric-aligned for weighted log loss. All changes only affect the fallback submission generation; the custom inference path remains identical.'
- What this solution (achieved 0.56001) has done: 'Your current score (0.56037, lower-is-better) is still far from the target (0.27894), and it strongly suggests you’re still on the fallback path (no external model assets). To move the score toward the target without changing any of your intended model/inference pipeline, I only improve the fallback by replacing the single global constant-per-label prior with a tiny logistic-regression baseline trained on `train.csv` using just the available 8 label columns as features (plus cross-validated out-of-fold prediction to reduce overconfidence). This keeps the “no-image” fallback semantics but makes probabilities better aligned to the competition’s weighted log-loss, especially for the heavier `patient_overall` label. The custom-code + model-assets path remains untouched, and the script still writes a valid `submission.csv` in the exact `row_id,fractured` format.'
- What this solution (achieved 0.55991) has done: 'Your current score (0.56001, lower-is-better) is still far from the target (0.27894), and since the external model assets are missing, the fallback path determines the score. To move toward the target with minimal changes and without altering the intended image-model pipeline, I only strengthen the fallback by learning per-label probabilities from `train.csv` using a tiny OOF logistic regression that uses simple non-image metadata features derived from the public `train_bounding_boxes.csv` (bbox counts/areas and slice coverage), which is legitimate and often informative. I keep the same submission alignment to `test.csv` row order and keep probabilities clipped for log-loss stability. If bounding boxes are unavailable for some studies, the code safely falls back to the previous label-only baseline behavior.'
- What this solution (achieved 0.56034) has done: 'Your score is still far above the target (0.5599 vs 0.2789, lower-is-better), and given the missing external model assets the fallback is what determines performance. I keep your current custom-model path unchanged, but improve the fallback in a minimal, metric-aligned way by adding a tiny per-label calibration step: for each label, choose a single constant probability that minimizes (unweighted) log loss on out-of-fold predictions, which tends to be closer to the true optimum constant under log loss than using the raw OOF mean. I also add a small second calibration only for `patient_overall` to ensure it stays consistent and well-calibrated (since it is heavily weighted), while still using the same learned vertebra probabilities. These changes keep the same “no-image” fallback semantics and should move the score downward toward the target without altering your intended core inference pipeline.'
- What this solution (achieved 0.56357) has done: 'Your current score (0.56034, lower-is-better) is far above the target (0.27894), so we should improve performance while keeping your intended custom-model path untouched. Since the external model assets are missing, your score is determined by the fallback; the smallest meaningful improvement is to make the fallback constants metric-aligned by explicitly minimizing *weighted* log loss (patient_overall is weighted higher) rather than unweighted log loss. Concretely, we keep your OOF logistic-regression feature generation exactly as-is, but replace the “best constant” selection with a per-label weighted constant optimizer, and we also select `patient_overall` as the constant that minimizes its own weighted log loss (instead of forcing it to soft-OR of vertebrae), which should reduce the competition’s dominant error term. Submission format, ordering, clipping, and file path remain unchanged and it still writes `submission.csv`.'
- What this solution (achieved 0.5639) has done: 'Your current score (0.56357, lower-is-better) is still far from the target (0.27894), and the environment is almost certainly still using the fallback (no external model assets). To move the score down with minimal changes and without touching the intended 3-stage image inference path, I improve only the fallback constant selection to match the competition’s *weighted* log-loss more faithfully: (1) use the correct per-label weights (patient_overall weight 7, vertebrae weights 1/2/2/2/2/2/2 as per RSNA 2022), and (2) actually optimize the constant using the model’s OOF probabilities (not just y-mean), via a simple 1D weighted log-loss minimization over the convex objective. This keeps your existing OOF logistic-regression feature generation intact and only changes the final calibration step that determines the submission when assets are missing.'
- What this solution (achieved 0.562) has done: 'Your score is far worse than the target (0.5639 vs 0.2789, lower-is-better) and the run is almost certainly using the fallback path (no external model assets), so we should improve only the fallback predictions while leaving the full 3-stage image inference path untouched. The smallest high-impact fix is that the current fallback learns a logistic-regression model but then ignores it and outputs the raw label prevalence (y.mean) as a constant; we instead (1) use the OOF logistic-regression probabilities and apply a single per-label temperature scaling (1D, convex) to reduce log-loss, and (2) keep `patient_overall` consistent by computing it as a calibrated soft-OR of the calibrated C1–C7 probabilities (this aligns with the definition of “any fracture”). These changes do not alter any model architecture/training loops in your intended pipeline; they only improve the “no-assets” fallback that determines your current score. Submission writing, ordering, and probability clipping remain the same to ensure a valid `submission.csv`.'
- What this solution (achieved 0.81226) has done: 'Your current score (0.562, lower-is-better) is far above the target (0.2789), and given the missing external model assets this run is determined entirely by the fallback path. To move the score down without touching your intended 3-stage image inference, I only change the fallback calibration so it no longer collapses the learned OOF logistic-regression outputs into a single mean probability per label. Instead, I (1) fit one final logistic-regression model per label on all training data, (2) temperature-calibrate each label using weighted log loss on OOF predictions, and (3) use those calibrated models to produce per-test-study probabilities (using available bbox-derived features; safe default zeros otherwise). This keeps the same fallback “no-images” approach and data sources, but makes predictions vary by study (critical for log loss), and it still writes a valid `submission.csv` aligned exactly to `test.csv`.'
- What this solution (achieved 0.55601) has done: 'Your current score (0.81226, lower-is-better) indicates the fallback path is producing very poor (near-constant/uncorrelated) probabilities for the hidden test set; the biggest issue is that the fallback logistic models are trained on features that are essentially constants on test (priors + zero bbox features), so they can’t generalize and can output badly calibrated extremes. I keep the fallback “no-image” approach, but make it stable and more metric-aligned by (1) actually building *test* features from `test_images` DICOM directories (slice count and total bytes) which are available at inference time and correlate with scan properties, (2) using those features in the same IRLS logistic regression + temperature calibration you already have, and (3) ensuring `patient_overall` is finally made consistent as a soft-OR of C1–C7 (this typically reduces the heavily-weighted patient_overall logloss). The custom model path (when external assets exist) is left unchanged. The script still runs end-to-end and writes `submission.csv` with the exact required schema and order.'
- What this solution (achieved 0.55615) has done: 'Your current score (0.55601, lower-is-better) is still far above the target (0.27894), and since the external model assets are missing, only the fallback path affects the score. The smallest reliable improvement that keeps the fallback’s core “no-image ML on cheap features” logic is to add one more inference-time feature that varies across hidden test: the slice index range (min/max) parsed from DICOM filenames, which is fast (no DICOM reading) and tends to correlate with scan length/protocol. I keep your IRLS logistic regression + per-label temperature scaling unchanged, just extend the design matrix with this extra directory-derived feature for both train and test so the model doesn’t see a distribution shift. Submission writing/order/schema and probability clipping remain identical, and it still writes `submission.csv`.'
- What this solution (achieved 0.55643) has done: 'Your current score (0.55615, lower-is-better) is still far from the target (0.27894), so we should improve the fallback (since the external model assets are missing and the custom path likely isn’t running). The smallest high-impact change without altering your fallback’s core “cheap logistic regression + temperature scaling” approach is to add a few more *test-available, no-DICOM-read* directory features that vary across hidden test: mean/median/max DICOM file size and a simple filename “density” (count/span). These features are also computed for train to avoid train/test feature mismatch, and they often help the fallback model correlate with protocol/scan length artifacts in this dataset, reducing log loss. Everything else (IRLS training, per-label temperature scaling, soft-OR for patient_overall, submission format/order, clipping) is kept identical.'

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


def _find_src_root_with_utils(start_roots):
    for root in start_roots:
        if not root or not os.path.exists(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            if "Utils" in dirnames and "Training" in dirnames:
                return dirpath
    return None


candidate_roots = [
    "../input",
    "/kaggle/input",
    "../input/rsna-2022-cervical-spine-fracture-detection",
    "/kaggle/input/rsna-2022-cervical-spine-fracture-detection",
]
src_root = _find_src_root_with_utils(candidate_roots)

if src_root and src_root not in sys.path:
    sys.path.insert(0, src_root)

print("==> src_root:", src_root)
print("==> sys.path[0:5]:", sys.path[:5])

HAVE_CUSTOM_CODE = True
try:
    from Utils.CommonTools import sitk_base
    from Utils.CommonTools.NiiIO import read_from_DICOM_dir
    from Utils.PreProcessing.resampling import sitk_dummy_3D_resample
    from Utils.CommonTools.bbox import get_bbox, extend_bbox
    from Utils.post_processing import keep_largest_cervical_cc
    from Utils.Inference.nnunet_inference import NNUnetCTPredictor
    from Utils.CommonTools.sitk_base import resample, copy_nii_info, get_nii_info
    from Training.Task_301_PostProcessing_Overall.model import Model as ModelStage3

    print("==> Import success (custom Utils/Training found)")
except Exception as e:
    HAVE_CUSTOM_CODE = False
    print(
        "==> WARNING: Could not import custom code. Will run fallback submission path."
    )
    print("    Import error:", repr(e))



## === cell 1
if HAVE_CUSTOM_CODE:

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
            with torch.no_grad():
                self.list_model = []
                for i in range(len(self.list_model_pth)):
                    model = ModelStage3(
                        in_ch=self.in_ch,
                        out_ch=self.out_ch,
                        list_ch=self.list_ch,
                        random_init=False,
                    )

                    ckpt = torch.load(self.list_model_pth[i], map_location="cpu")

                    model.load_state_dict(ckpt)
                    model.eval()
                    model = model.to(self.device)
                    self.list_model.append(model)

                    print(
                        f"==> Init model from {self.list_model_pth[i]} to device {self.device}"
                    )

        def predict(self, image):
            with torch.no_grad():
                list_pred = []
                input_ori = image.copy()

                if self.tta:
                    p_flip_z = (0, 1) if 2 in self.tta_flip_axis else (0,)
                    p_flip_y = (0, 1) if 3 in self.tta_flip_axis else (0,)
                    p_flip_x = (0, 1) if 4 in self.tta_flip_axis else (0,)
                else:
                    p_flip_z = (0,)
                    p_flip_y = (0,)
                    p_flip_x = (0,)

                for flip_z in p_flip_z:
                    for flip_y in p_flip_y:
                        for flip_x in p_flip_x:
                            patch_input = (
                                torch.from_numpy(input_ori).to(self.device).unsqueeze(0)
                            )

                            flip_axis = []
                            if flip_z == 1:
                                flip_axis.append(2)
                            if flip_y == 1:
                                flip_axis.append(3)
                            if flip_x == 1:
                                flip_axis.append(4)

                            do_flip = (flip_z == 1) or (flip_y == 1) or (flip_x == 1)
                            if do_flip:
                                patch_input = torch.flip(patch_input, dims=flip_axis)

                            for model in self.list_model:
                                pred = model(patch_input)[0]
                                pred = torch.sigmoid(pred[0, 0])
                                list_pred.append(pred.cpu().numpy())

                return float(np.mean(list_pred))

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

        def predict_stage3(self, pred_c1_c7, pred_fracture):
            pred_c1_c7_nii = sitk.GetImageFromArray(pred_c1_c7)
            pred_fracture_nii = sitk.GetImageFromArray(pred_fracture)

            new_size = (96, 96, 96)
            new_spacing = list(
                np.array(pred_fracture_nii.GetSize()[::-1]) / np.array(new_size)
            )

            pred_fracture_nii = sitk_base.resample(
                pred_fracture_nii,
                new_spacing[::-1],
                new_origin=None,
                new_size=new_size[::-1],
                new_direction=None,
                center_origin=None,
                interp=sitk.sitkLinear,
                dtype=sitk.sitkFloat32,
                constant_value=0,
            )
            pred_c1_c7_nii = sitk_base.resample(
                pred_c1_c7_nii,
                new_spacing[::-1],
                new_origin=None,
                new_size=new_size[::-1],
                new_direction=None,
                center_origin=None,
                interp=sitk.sitkNearestNeighbor,
                dtype=sitk.sitkUInt8,
                constant_value=0,
            )

            pred_c1_c7 = sitk.GetArrayFromImage(pred_c1_c7_nii)
            pred_fracture = sitk.GetArrayFromImage(pred_fracture_nii)

            input_ = np.concatenate(
                (pred_fracture[np.newaxis], pred_c1_c7[np.newaxis]), axis=0
            )
            score = self.predictor_stage3.predict(input_)
            return score

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

            output[0] = max(self.params["min_score"][0], np.max(output[1:]))
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
                        print(f"    score: {score}")

                    print(
                        f"    Finish this split use : {time.time() - time_start} seconds"
                    )
                    print(
                        f"    Overall use : {time.time() - overall_time_start} seconds"
                    )

                    gc.collect()




## === cell 2
time_start = time.time()

DATA_DIR = "../input/rsna-2022-cervical-spine-fracture-detection/test_images"
SAVE_CSV = "submission.csv"

test_csv_path = "../input/rsna-2022-cervical-spine-fracture-detection/test.csv"
sample_sub_path = (
    "../input/rsna-2022-cervical-spine-fracture-detection/sample_submission.csv"
)
train_csv_path = "../input/rsna-2022-cervical-spine-fracture-detection/train.csv"
train_bbox_csv_path = (
    "../input/rsna-2022-cervical-spine-fracture-detection/train_bounding_boxes.csv"
)
train_img_dir = "../input/rsna-2022-cervical-spine-fracture-detection/train_images"

test_df = pd.read_csv(test_csv_path)
sample_sub = pd.read_csv(sample_sub_path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("==> device:", device)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1.0 - p))


def _fit_logreg_irls(X, y, l2=1.0, max_iter=50):
    """
    Minimal logistic regression (IRLS/Newton) with L2 on weights (excluding intercept).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = X.shape
    w = np.zeros(d, dtype=np.float64)

    for _ in range(max_iter):
        z = X @ w
        p = _sigmoid(z)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        r = p * (1.0 - p)

        XR = X * r[:, None]
        H = XR.T @ X
        reg = np.eye(d, dtype=np.float64) * l2
        reg[0, 0] = 0.0
        H = H + reg

        g = X.T @ (y - p)
        g[1:] -= l2 * w[1:]

        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break

        w_new = w + step
        if np.max(np.abs(w_new - w)) < 1e-6:
            w = w_new
            break
        w = w_new

    return w


def _predict_logreg(X, w):
    X = np.asarray(X, dtype=np.float64)
    return _sigmoid(X @ w)


def _weighted_logloss_binary(y, p, sample_weight=None):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    if sample_weight is None:
        return -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    sw = np.asarray(sample_weight, dtype=np.float64)
    sw = sw / (sw.mean() + 1e-12)
    return -np.mean(sw * (y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _apply_temperature(p, T):
    p = np.asarray(p, dtype=np.float64)
    return _sigmoid(_logit(p) / float(T))


def _fit_temperature_1d(p, y, w=None, T_min=0.5, T_max=3.0, n_grid=61):
    Ts = np.linspace(T_min, T_max, n_grid, dtype=np.float64)
    best_T = 1.0
    best_loss = float("inf")
    for T in Ts:
        pp = _apply_temperature(p, T)
        loss = _weighted_logloss_binary(y, pp, sample_weight=w)
        if loss < best_loss:
            best_loss = loss
            best_T = float(T)
    return best_T


def _build_design_matrix(df, feat_cols):
    X_base = df[feat_cols].values.astype(np.float64)
    n = len(df)
    X = np.concatenate([np.ones((n, 1), dtype=np.float64), X_base], axis=1)
    return X


def _bbox_features_for_uids(bbox_csv_path: str):
    """
    Existing train-only bbox features (kept as-is).
    """
    if not bbox_csv_path or (not os.path.exists(bbox_csv_path)):
        return None

    bb = pd.read_csv(bbox_csv_path)
    if "StudyInstanceUID" not in bb.columns:
        return None

    for c in ["x", "y", "width", "height", "slice_number"]:
        if c in bb.columns:
            bb[c] = pd.to_numeric(bb[c], errors="coerce")

    bb["area"] = (
        bb["width"].fillna(0.0).clip(lower=0.0)
        * bb["height"].fillna(0.0).clip(lower=0.0)
    ).astype(np.float64)

    g = bb.groupby("StudyInstanceUID", sort=False)

    feat = pd.DataFrame(
        {
            "StudyInstanceUID": g.size().index.astype(str),
            "bb_count": g.size().values.astype(np.float64),
            "bb_area_mean": g["area"].mean().fillna(0.0).values.astype(np.float64),
            "bb_area_max": g["area"].max().fillna(0.0).values.astype(np.float64),
            "bb_slice_min": g["slice_number"]
            .min()
            .fillna(0.0)
            .values.astype(np.float64),
            "bb_slice_max": g["slice_number"]
            .max()
            .fillna(0.0)
            .values.astype(np.float64),
        }
    )
    feat["bb_slice_span"] = (feat["bb_slice_max"] - feat["bb_slice_min"]).clip(
        lower=0.0
    )

    for c in ["bb_count", "bb_area_mean", "bb_area_max", "bb_slice_span"]:
        feat[c] = np.log1p(feat[c].values.astype(np.float64))

    return feat[
        ["StudyInstanceUID", "bb_count", "bb_area_mean", "bb_area_max", "bb_slice_span"]
    ]


def _dir_features_for_uids(img_root: str, uids):
    """
    Change (score-relevant, minimal): add a few extra directory-level features that vary on hidden test
    but are still cheap and do NOT require reading DICOM pixels:
      - mean/median/max DICOM file size
      - filename density: count / (span+1)
    These are computed for both train and test to avoid distribution shift.
    """
    if not img_root or (not os.path.exists(img_root)):
        return None

    uids = pd.Series(uids, dtype=str).unique().tolist()
    rows = []
    for uid in uids:
        d = os.path.join(img_root, uid)
        n_files = 0
        total_bytes = 0
        sizes = []
        smin = None
        smax = None
        if os.path.isdir(d):
            try:
                for fn in os.listdir(d):
                    if not fn.lower().endswith(".dcm"):
                        continue
                    n_files += 1
                    base = fn.rsplit(".", 1)[0]
                    try:
                        si = int(base)
                        if smin is None or si < smin:
                            smin = si
                        if smax is None or si > smax:
                            smax = si
                    except Exception:
                        pass
                    try:
                        sz = os.path.getsize(os.path.join(d, fn))
                        total_bytes += sz
                        sizes.append(float(sz))
                    except OSError:
                        pass
            except OSError:
                pass

        if smin is None or smax is None:
            slice_span = 0.0
        else:
            slice_span = float(max(0, smax - smin))

        if len(sizes) == 0:
            fsz_mean = 0.0
            fsz_med = 0.0
            fsz_max = 0.0
        else:
            arr = np.asarray(sizes, dtype=np.float64)
            fsz_mean = float(arr.mean())
            fsz_med = float(np.median(arr))
            fsz_max = float(arr.max())

        density = float(n_files) / float(slice_span + 1.0)

        rows.append(
            (
                uid,
                float(n_files),
                float(total_bytes),
                slice_span,
                fsz_mean,
                fsz_med,
                fsz_max,
                density,
            )
        )

    feat = pd.DataFrame(
        rows,
        columns=[
            "StudyInstanceUID",
            "dcm_count",
            "dcm_bytes",
            "dcm_slice_span",
            "dcm_fsize_mean",
            "dcm_fsize_median",
            "dcm_fsize_max",
            "dcm_density",
        ],
    )

    for c in [
        "dcm_count",
        "dcm_bytes",
        "dcm_slice_span",
        "dcm_fsize_mean",
        "dcm_fsize_median",
        "dcm_fsize_max",
        "dcm_density",
    ]:
        feat[c] = np.log1p(feat[c].values.astype(np.float64))
    return feat


def _soft_or(ps):
    ps = np.asarray(ps, dtype=np.float64)
    ps = np.clip(ps, 1e-6, 1 - 1e-6)
    return 1.0 - np.prod(1.0 - ps, axis=0)


def _make_fallback_models_and_predict(
    train_csv: str, bbox_csv: str, test_df_in: pd.DataFrame
):
    labels = ["patient_overall", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]

    feat_cols = [
        "dcm_count",
        "dcm_bytes",
        "dcm_slice_span",
        "dcm_fsize_mean",
        "dcm_fsize_median",
        "dcm_fsize_max",
        "dcm_density",
        "bb_count",
        "bb_area_mean",
        "bb_area_max",
        "bb_slice_span",
    ]

    label_weights = {
        "patient_overall": 7.0,
        "C1": 1.0,
        "C2": 2.0,
        "C3": 2.0,
        "C4": 2.0,
        "C5": 2.0,
        "C6": 2.0,
        "C7": 2.0,
    }

    if not os.path.exists(train_csv):
        uid_probs = {}
        for uid in test_df_in["StudyInstanceUID"].astype(str).unique():
            uid_probs[uid] = {k: 0.5 for k in labels}
        return uid_probs

    tr = pd.read_csv(train_csv).copy()
    tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)

    for k in labels:
        if k not in tr.columns:
            tr[k] = 0.0
        tr[k] = pd.to_numeric(tr[k], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    dir_tr = _dir_features_for_uids(train_img_dir, tr["StudyInstanceUID"].values)
    if dir_tr is not None:
        tr = tr.merge(dir_tr, on="StudyInstanceUID", how="left")

    feat_bb = _bbox_features_for_uids(bbox_csv)
    if feat_bb is not None:
        tr = tr.merge(feat_bb, on="StudyInstanceUID", how="left")

    for c in feat_cols:
        if c not in tr.columns:
            tr[c] = 0.0
        tr[c] = pd.to_numeric(tr[c], errors="coerce").fillna(0.0).astype(np.float64)

    X_tr = _build_design_matrix(tr, feat_cols=feat_cols)

    uids = tr["StudyInstanceUID"].astype(str).values
    fold = (pd.util.hash_pandas_object(pd.Series(uids), index=False).values % 5).astype(
        int
    )

    per_label = {}
    for target in labels:
        y = tr[target].values.astype(np.float64)
        preds_oof = np.zeros(len(tr), dtype=np.float64)

        l2 = 2.0 if target == "patient_overall" else 1.0
        for f in range(5):
            tr_idx = fold != f
            va_idx = fold == f
            if va_idx.sum() == 0 or tr_idx.sum() == 0:
                continue
            w_hat = _fit_logreg_irls(X_tr[tr_idx], y[tr_idx], l2=l2, max_iter=50)
            preds_oof[va_idx] = _predict_logreg(X_tr[va_idx], w_hat)

        preds_oof = np.clip(preds_oof, 1e-6, 1 - 1e-6)

        sw = np.full(len(tr), float(label_weights[target]), dtype=np.float64)
        T = _fit_temperature_1d(preds_oof, y, w=sw, T_min=0.5, T_max=3.0, n_grid=61)

        w_full = _fit_logreg_irls(X_tr, y, l2=l2, max_iter=50)
        per_label[target] = {"w": w_full, "T": float(T)}

    u_test = test_df_in[["StudyInstanceUID"]].drop_duplicates().copy()
    u_test["StudyInstanceUID"] = u_test["StudyInstanceUID"].astype(str)

    dir_te = _dir_features_for_uids(DATA_DIR, u_test["StudyInstanceUID"].values)
    if dir_te is not None:
        u_test = u_test.merge(dir_te, on="StudyInstanceUID", how="left")

    for c in ["bb_count", "bb_area_mean", "bb_area_max", "bb_slice_span"]:
        u_test[c] = 0.0

    for c in [
        "dcm_count",
        "dcm_bytes",
        "dcm_slice_span",
        "dcm_fsize_mean",
        "dcm_fsize_median",
        "dcm_fsize_max",
        "dcm_density",
    ]:
        if c not in u_test.columns:
            u_test[c] = 0.0

    for c in feat_cols:
        u_test[c] = (
            pd.to_numeric(u_test[c], errors="coerce").fillna(0.0).astype(np.float64)
        )

    X_te = _build_design_matrix(u_test, feat_cols=feat_cols)

    uid_probs = {uid: {} for uid in u_test["StudyInstanceUID"].values}

    for target in ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]:
        p = _predict_logreg(X_te, per_label[target]["w"])
        p = _apply_temperature(p, per_label[target]["T"])
        p = np.clip(p, 1e-6, 1 - 1e-6)
        for i, uid in enumerate(u_test["StudyInstanceUID"].values):
            uid_probs[uid][target] = float(p[i])

    c_stack = np.vstack(
        [
            np.array([uid_probs[uid][c] for uid in u_test["StudyInstanceUID"].values])
            for c in ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
        ]
    )
    p_overall = _soft_or(c_stack)
    p_overall = np.clip(p_overall, 1e-6, 1 - 1e-6)
    for i, uid in enumerate(u_test["StudyInstanceUID"].values):
        uid_probs[uid]["patient_overall"] = float(p_overall[i])

    return uid_probs


def _write_fallback_submission():
    uid_probs = _make_fallback_models_and_predict(
        train_csv=train_csv_path,
        bbox_csv=train_bbox_csv_path,
        test_df_in=test_df,
    )

    sub = test_df[["row_id", "StudyInstanceUID", "prediction_type"]].copy()
    sub["StudyInstanceUID"] = sub["StudyInstanceUID"].astype(str)

    def _lookup(row):
        uid = row["StudyInstanceUID"]
        ptype = row["prediction_type"]
        if uid in uid_probs and ptype in uid_probs[uid]:
            return uid_probs[uid][ptype]
        return 0.5

    sub["fractured"] = sub.apply(_lookup, axis=1).astype(np.float32)
    sub["fractured"] = np.clip(sub["fractured"].values, 1e-6, 1 - 1e-6).astype(
        np.float32
    )
    sub = sub[["row_id", "fractured"]]
    sub.to_csv(SAVE_CSV, index=False)

    vals = sub["fractured"].values.astype(np.float64)
    print(f"==> Wrote fallback {SAVE_CSV} with {len(sub)} rows.")
    print(
        f"    fallback pred summary: mean={vals.mean():.6f} std={vals.std():.6f} min={vals.min():.6f} max={vals.max():.6f}"
    )


if not HAVE_CUSTOM_CODE:
    _write_fallback_submission()
else:
    required_paths = [
        DATA_DIR,
        "../input/models2/models",
        "../input/plans-nnunet",
    ]
    missing = [p for p in required_paths if not os.path.exists(p)]
    if missing:
        print("==> WARNING: Missing required asset paths:", missing)
        _write_fallback_submission()
    else:
        with torch.no_grad():
            list_model_C1_C7_segmentation = [
                "../input/models2/models/stage1_0.model",
                "../input/models2/models/stage1_1.model",
                "../input/models2/models/stage1_2.model",
            ]
            plan_C1_C7_segmentation = "../input/plans-nnunet/stage1.pkl"

            list_model_fracture_detection = [
                "../input/models2/models/stage2_5.model",
            ]
            list_model_post_processing = [
                "../input/models2/models/stage3_0.pkl",
            ]
            plan_fracture_detection = "../input/plans-nnunet/stage2.pkl"

            required_files = (
                list_model_C1_C7_segmentation
                + list_model_fracture_detection
                + list_model_post_processing
                + [
                    plan_C1_C7_segmentation,
                    plan_fracture_detection,
                ]
            )
            missing_files = [p for p in required_files if not os.path.exists(p)]
            if missing_files:
                print(
                    "==> WARNING: Missing required model/plan files (showing up to 10):",
                    missing_files[:10],
                )
                _write_fallback_submission()
            else:
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

                predictor_3 = PredictorStage3(
                    list_model_pth=list_model_post_processing,
                    device=device,
                    tta=True,
                    tta_flip_axis=(4,),
                )

                c2f_predictor = FractureDetector(
                    predictor_stage1=predictor_1,
                    predictor_stage2=predictor_2,
                    predictor_stage3=predictor_3,
                )

                list_DICOM_dirs = os.listdir(DATA_DIR)
                list_DICOM_dirs = [
                    f"{DATA_DIR}/{sub_dir}" for sub_dir in list_DICOM_dirs
                ]
                print(f"==> Total {len(list_DICOM_dirs)} cases")

                list_size_DICOM_dirs = []
                for case_dir in list_DICOM_dirs:
                    size_ = 0
                    for file in os.listdir(case_dir):
                        size_ += os.path.getsize(f"{case_dir}/{file}")
                    list_size_DICOM_dirs.append(size_)

                list_DICOM_dirs = list(
                    np.array(list_DICOM_dirs)[np.argsort(list_size_DICOM_dirs)[::-1]]
                )
                print(f"==> Sort DICOM dirs by size")
                print(list_DICOM_dirs[0])

                c2f_predictor.predict(list_test_files=list_DICOM_dirs)
                results = c2f_predictor.results

                pred_by_uid = {}
                for uid, score in results.items():
                    pred_by_uid[uid] = {
                        "patient_overall": float(score[0]),
                        "C1": float(score[1]),
                        "C2": float(score[2]),
                        "C3": float(score[3]),
                        "C4": float(score[4]),
                        "C5": float(score[5]),
                        "C6": float(score[6]),
                        "C7": float(score[7]),
                    }

                sub = test_df[["row_id", "StudyInstanceUID", "prediction_type"]].copy()

                def _lookup(row):
                    uid = row["StudyInstanceUID"]
                    ptype = row["prediction_type"]
                    if uid in pred_by_uid:
                        return pred_by_uid[uid].get(ptype, 0.5)
                    return 0.5

                sub["fractured"] = sub.apply(_lookup, axis=1).astype(np.float32)
                sub["fractured"] = np.clip(sub["fractured"].values, 1e-6, 1 - 1e-6)
                sub = sub[["row_id", "fractured"]]
                sub.to_csv(SAVE_CSV, index=False)

                print(f"==> Wrote {SAVE_CSV} with {len(sub)} rows")
                print(f"==> Finish using time: {time.time() - time_start}")
