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

0.2887448668478558

# 6. Current score

0.56494

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5639) has done: 'I fix the immediate import/path failure by making the code robust to the missing `../input/srccode/src` dependency and instead run a safe fallback pipeline that still produces a valid `submission.csv`. Specifically, if the custom nnUNet utilities/models are unavailable, the script generate calibrated constant probabilities derived from the training label prevalences (a score-improving baseline vs. arbitrary zeros) and map them correctly to `test.csv`’s `row_id` structure. I also fix submission formatting issues (ensure numeric `fractured`, correct ordering/coverage of all `row_id`s) so Kaggle accepts the file. The original core inference logic is preserved and be used automatically if the external src/models exist; otherwise the fallback runs end-to-end.'
- What this solution (achieved 0.56357) has done: 'Your current score (0.5639, lower-is-better) is far from the target (0.2887), so we should improve legitimately with minimal disruption. Since the external nnUNet pipeline isn’t available, the only place to improve is the fallback: instead of raw label prevalences, we compute a leave-one-out (LOO) smoothed prevalence per label (reduces overfitting on tiny train=202) and then apply a single global temperature scaling (logit shrink) to make probabilities less extreme, which usually improves weighted log loss. Core logic is preserved (still a constant-per-label baseline), submission schema stays identical, and we still write `submission.csv` end-to-end. These changes are small, deterministic, and directly targeted at lowering log loss without changing the overall approach.'
- What this solution (achieved 0.58594) has done: 'Your current score (0.56357, lower-is-better) is far from the target (0.28874), so we should improve the fallback baseline while keeping the core approach (constant per-label probabilities) unchanged. The biggest score leak in the current fallback is that `patient_overall` is predicted independently from C1–C7, even though by definition it should be the probability of “any fracture,” which is better modeled as `1 - Π(1 - p_Ci)` under an independence approximation. We keep the same LOO-smoothed prevalence + temperature scaling for C1–C7, but compute `patient_overall` from those calibrated vertebra probabilities, then apply the same temperature scaling to the derived overall probability for stability. This is a minimal, metric-aligned post-processing change that typically reduces weighted log loss without changing the training/inference paradigm or requiring image inference.'
- What this solution (achieved 0.56107) has done: 'We keep your fallback approach (constant per-label probabilities) but make two minimal metric-aligned improvements to reduce log loss: (1) fit the single temperature `T` by cross-validation on `train.csv` (instead of a fixed 1.25) to better calibrate probabilities, and (2) compute `patient_overall` consistently from the calibrated C1–C7 probabilities using the any-fracture aggregation you already introduced. This preserves the same core logic and semantics (no new model, no image inference), but should move the score down toward the 0.2887 target. The external pipeline branch is left intact; only the fallback calibration is adjusted. The script still writes a valid `submission.csv` with correct `row_id` alignment.'
- What this solution (achieved 0.56107) has done: 'Your current score (0.56107, lower-is-better) is still far above the target (0.28874), so we should improve the fallback calibration while keeping the same core logic (constant per-label probabilities with temperature scaling + patient_overall computed as any-fracture aggregation). The main issue in the current code is that the “CV” temperature selection isn’t actually doing cross-validation (it evaluates the same constant predictions against the full training labels), which can pick a suboptimal T. I replace that with a true out-of-fold procedure: for each fold, compute LOO-smoothed prevalences on the training folds only, apply a candidate T, derive patient_overall from the calibrated C1–C7, and score on the held-out fold; then pick T with the best mean weighted log loss. This is a minimal change, preserves the fallback paradigm, and should legitimately reduce log loss toward your target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.56156) has done: 'We keep your exact fallback paradigm (constant per-label probabilities + temperature scaling + `patient_overall` derived as “any fracture”) but fix two calibration weaknesses that limit log-loss: (1) use stratified folds on `patient_overall` so the temperature CV is less noisy on only 202 studies, and (2) after selecting the global temperature `T`, fit one additional scalar `Toverall` specifically for `patient_overall` (because it has different weighting/behavior) while still computing it from C1–C7 via the same any-aggregation. This does not change the core model/logic (still constant probabilities) and is a minimal, metric-aligned calibration step that should reduce loss toward your target. Submission formatting and alignment remain identical, and it still writes a valid `submission.csv` end-to-end within the time limit.'
- What this solution (achieved 0.56242) has done: 'Your current fallback is still a constant-per-label baseline, so the only safe way to reduce log loss (lower-is-better) without changing core logic is to calibrate those constants better. I keep the same “LOO-smoothed prevalence + temperature scaling + patient_overall as any-fracture aggregation” approach, but (1) replace the ad-hoc LOO formula with a standard Beta prior smoothing on prevalence (less biased on n=202) and (2) do a true out-of-fold search for both temperatures using those same smoothed prevalences (so the CV objective matches what you deploy). These are minimal, deterministic calibration fixes that should legitimately move the score downward toward your target while still producing a valid `submission.csv` end-to-end within the time limit.'
- What this solution (achieved 0.55995) has done: 'Your current score (0.56242, lower-is-better) is still far from the target (0.28874), so we should improve the fallback calibration while keeping the same constant-per-label paradigm. The main minimal win left is to make the constant predictions better match the metric by optimizing *two* temperatures: one for C1–C7 and one for patient_overall, but doing it jointly (because patient_overall is derived from C1–C7 via the any-fracture aggregation). I also update the CV scoring to use the competition-style normalization (sum of weights divided by sum of weights) rather than a plain mean of weighted losses, so the temperature search aligns more closely with Kaggle’s weighted log loss. Everything else (no image inference, same mapping to `test.csv` row structure, same CSV output) remains unchanged.'
- What this solution (achieved 0.56005) has done: 'We keep your exact fallback paradigm (constant per-label probabilities with Beta-smoothing + temperature scaling, and `patient_overall` derived via the any-fracture aggregation), but fix the CV objective to match Kaggle’s **true weighted mean** (weighted sum divided by sum of weights) instead of “weights normalized by mean.” Then we make the temperature/Beta grid slightly finer in a tightly bounded range (still a deterministic grid search) so calibration can move meaningfully toward your target loss without changing modeling logic. Finally, we keep the external nnUNet branch untouched and ensure the submission mapping/format remains identical.'
- What this solution (achieved 0.55936) has done: 'We keep your fallback paradigm (constant-per-label probabilities + Beta smoothing + temperature scaling + `patient_overall` derived as “any fracture”) but make two minimal calibration fixes to legitimately lower weighted log loss. First, instead of a coarse 3D grid over `(a=b, T, Toverall)`, we do a fast coordinate-descent style search: pick `a=b`, then optimize `T`, then `Toverall`, and repeat once—this tends to find a better calibration without changing modeling semantics and stays well under the time limit. Second, we compute the CV objective exactly as Kaggle’s weighted mean over the 8 labels per study (same as you intended), but avoid repeated recomputation overhead to allow a slightly finer temperature grid for better matching.'
- What this solution (achieved 0.55936) has done: 'Your current solution is still a constant-per-label fallback, so the only legitimate way to move the (lower-is-better) score toward the target is to better align the constant probabilities with the competition’s weighted log loss. I keep the same Beta-smoothed prevalence + temperature scaling + “patient_overall as any-fracture aggregation” logic, but change the calibration search to optimize **directly on the true weighted loss with patient_overall weight 2** without recomputing folds repeatedly. Concretely, I precompute fold-wise label sums and then run a small deterministic coordinate-descent search over `(a=b, T, Toverall)` using those cached stats, which improves calibration while preserving the exact modeling semantics. The external nnUNet branch remains untouched, and the script still writes a valid `submission.csv` with correct `row_id` alignment.'
- What this solution (achieved 0.55936) has done: 'We keep your fallback paradigm (constant-per-label probabilities derived from train prevalence, Beta smoothing, temperature scaling, and `patient_overall` computed as “any fracture”) but make the calibration closer to the competition metric with very small, targeted changes. Specifically: (1) evaluate CV on the true per-row structure (8 rows per study with weights) by computing the weighted loss per fold without tiling large arrays, (2) expand the smoothing/temperature search slightly in a bounded way to find a better-calibrated constant baseline, and (3) add a tiny final clipping inside the loss to prevent extreme log loss penalties during CV. This should legitimately reduce your current loss (0.55936, lower-is-better) toward the target (0.2887) without changing the overall approach or I/O, and still writes a valid `submission.csv`.'
- What this solution (achieved 0.56262) has done: 'Your score is still far from the target (lower-is-better), so the only safe way to move toward it without changing the core “constant-per-label fallback” logic is to make the calibration objective match the competition metric more exactly. I update the CV scorer to compute the weighted log loss on the exact per-row structure (8 rows per study) and weight-average across all rows (not fold-means), and I keep the same Beta-smoothing + temperature scaling + patient_overall any-aggregation. Then I replace the coarse coordinate-descent search with a small deterministic grid search over the same parameters but evaluated with the corrected global weighted objective (still fast at n=202). This should legitimately reduce loss versus the current misaligned CV selection while preserving your architecture/inference branches and still writing a valid `submission.csv`.'
- What this solution (achieved 0.56494) has done: 'Your current score (0.56262, lower-is-better) is still far from the target (0.28874), so we should improve the fallback calibration while keeping the same constant-per-label paradigm intact. The main minimal win is to stop forcing `patient_overall` to be derived purely from the C1–C7 “independence” aggregation (which can be miscalibrated), and instead blend it with its own Beta-smoothed prevalence, with the blend weight chosen by the same CV weighted log loss you already use. This preserves the core logic (still constant probabilities + Beta smoothing + temperature scaling; no new model, no image inference) while directly optimizing the metric for the most heavily-weighted label. I also keep the existing external nnUNet branch untouched and ensure we still write a valid `submission.csv` aligned exactly to `test.csv` row_ids.'

# 9. Code solution

## === cell 0
import os
import sys
import time
import numpy as np
import pandas as pd
import torch

try:
    import SimpleITK as sitk  # noqa: F401
except Exception:
    sitk = None

SRC_PATH = "../input/srccode/src"
USE_EXTERNAL_PIPELINE = os.path.isdir(SRC_PATH)

if USE_EXTERNAL_PIPELINE:
    if SRC_PATH not in sys.path:
        sys.path.insert(0, SRC_PATH)
    print("Found external src at:", SRC_PATH)
    print("SRC contents:", os.listdir(SRC_PATH))

    from Utils.CommonTools.bbox import get_bbox, extend_bbox  # type: ignore
    from Utils.post_processing import keep_largest_cervical_cc  # type: ignore
    from Utils.Inference.nnunet_inference import NNUnetCTPredictor  # type: ignore
    from Utils.CommonTools.NiiIO import read_from_DICOM_dir  # type: ignore
    from Utils.CommonTools.sitk_base import resample, copy_nii_info, get_nii_info  # type: ignore

    print("==> External imports success")
else:
    print("WARNING: External src not found at ../input/srccode/src")
    print(
        "         Falling back to prevalence-based baseline submission (no image inference)."
    )



## === cell 1
if USE_EXTERNAL_PIPELINE:

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
            c1_c7_bbox = get_bbox(np.logical_and(pred >= 1, pred <= 7))
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
            print(f"        ----> Resampling ...")
            ori_nii_info = get_nii_info(ct_nii)
            ct_nii = self.predictor_stage1.resampling(ct_nii)

            image = sitk.GetArrayFromImage(ct_nii)[np.newaxis]
            image = self.predictor_stage1.pre_processing(image)

            pred = self.predictor_stage1.sliding_window_inference(image)

            pred = np.argmax(pred, axis=0)
            pred = keep_largest_cervical_cc(pred, ct_nii.GetSpacing()[::-1])

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
            return pred

        def predict_stage2(self, ct_nii):
            print(f"        ----> Resampling ...")
            ori_nii_info = get_nii_info(ct_nii)
            ct_nii = self.predictor_stage2.resampling(ct_nii)

            image = sitk.GetArrayFromImage(ct_nii)[np.newaxis]
            image = self.predictor_stage2.pre_processing(image)

            pred = self.predictor_stage2.sliding_window_inference(image)

            pred = pred[1]  # 0 for background, 1 for foreground

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
            return pred

        def get_score(self, pred_c1_c7, pred_fracture):
            output = np.zeros(8, np.float32)  # Overall, C1-C7

            if (pred_c1_c7 is not None) and (pred_fracture is not None):
                pred_c1_c7[pred_c1_c7 > 7] = 0

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
            return output

        def predict(self, list_test_files):
            count = 0
            overall_time_start = time.time()
            for file in list_test_files:
                case_id = file.split("/")[-1].split(".nii")[0]

                count += 1
                print(f"==> Predicting {count}: {case_id}")

                time_start = time.time()

                ct_nii = read_from_DICOM_dir(file)
                ori_nii_info = get_nii_info(ct_nii)
                print(
                    f"        ----> Finish Reading use : {time.time() - time_start} seconds"
                )

                pred_1 = self.predict_stage1(ct_nii)
                print(
                    f"        ----> Finish stage1 use : {time.time() - time_start} seconds"
                )

                c1_c7_bbox = self.get_c1_c7_bbox(pred_1, ori_nii_info["spacing"][::-1])
                print(
                    f"        ----> Finish cropping c1_c7 bbox use : {time.time() - time_start} seconds"
                )

                if c1_c7_bbox is not None:
                    bz, ez, by, ey, bx, ex = c1_c7_bbox
                    roi_ct_nii = ct_nii[bx : ex + 1, by : ey + 1, bz : ez + 1]
                    roi_pred_1 = pred_1[bz : ez + 1, by : ey + 1, bx : ex + 1]
                    roi_pred_2 = self.predict_stage2(roi_ct_nii)
                else:
                    roi_pred_1 = None
                    roi_pred_2 = None

                print(
                    f"        ----> Finish stage2 use : {time.time() - time_start} seconds"
                )

                score = self.get_score(roi_pred_1, roi_pred_2)
                self.results[case_id] = score
                print(
                    f"        ----> Overall use : {time.time() - overall_time_start} seconds"
                )




## === cell 2
time_start = time.time()

BASE_DIR = "../input/rsna-2022-cervical-spine-fracture-detection"
TEST_CSV_PATH = f"{BASE_DIR}/test.csv"
TRAIN_CSV_PATH = f"{BASE_DIR}/train.csv"
DATA_DIR = f"{BASE_DIR}/test_images"
SAVE_CSV = "submission.csv"

test_df = pd.read_csv(TEST_CSV_PATH)
assert {"StudyInstanceUID", "prediction_type", "row_id"}.issubset(test_df.columns)

if USE_EXTERNAL_PIPELINE:
    total_scans = len(os.listdir(DATA_DIR))
    print("==> Total " + str(total_scans))

    with torch.no_grad():
        list_model_C1_C7_segmentation = [
            "../input/models/models/stage1_0.model",
            "../input/models/models/stage1_1.model",
        ]
        plan_C1_C7_segmentation = "../input/plans-nnunet/stage1.pkl"

        list_model_fracture_detection = [
            "../input/models/models/stage2_0.model",
        ]
        plan_fracture_detection = "../input/plans-nnunet/stage2.pkl"

        predictor_1 = NNUnetCTPredictor(
            list_model_pth=list_model_C1_C7_segmentation,
            plan_file=plan_C1_C7_segmentation,
            plan_stage=-1,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
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
        predictor_2 = NNUnetCTPredictor(
            list_model_pth=list_model_fracture_detection,
            plan_file=plan_fracture_detection,
            plan_stage=-1,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            use_gaussian_for_sliding_window=True,
            patch_size=None,
            stride=None,
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

        list_DICOM_dirs = os.listdir(DATA_DIR)
        list_DICOM_dirs = [f"{DATA_DIR}/{sub_dir}" for sub_dir in list_DICOM_dirs]
        print(f"==> Total {len(list_DICOM_dirs)} cases")
        c2f_predictor.predict(list_test_files=list_DICOM_dirs)

        results = c2f_predictor.results

        results_csv = {"row_id": [], "fractured": []}
        for case_id in results.keys():
            for C_i in range(1, 8):
                results_csv["row_id"].append(f"{case_id}_C{C_i}")
                results_csv["fractured"].append(float(results[case_id][C_i]))
            results_csv["row_id"].append(f"{case_id}_patient_overall")
            results_csv["fractured"].append(float(results[case_id][0]))

        sub_df = pd.DataFrame(results_csv)

        sub_df = test_df[["row_id"]].merge(sub_df, on="row_id", how="left")
        sub_df["fractured"] = (
            sub_df["fractured"].astype("float32").fillna(0.5).clip(1e-6, 1 - 1e-6)
        )

        sub_df.to_csv(SAVE_CSV, index=False)
        print(f"==> Wrote {SAVE_CSV} with shape {sub_df.shape}")
        print(f"==> Finish using time: {time.time() - time_start:.2f}s")

else:
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    target_cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]
    assert set(target_cols).issubset(train_df.columns)

    n = int(train_df.shape[0])
    eps = 1e-6  # numerical clipping right before submission
    base_clip = 5e-5

    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    def logit(p: float) -> float:
        p = float(np.clip(p, 1e-12, 1 - 1e-12))
        return float(np.log(p / (1 - p)))

    def clip01(p: float, lo: float = base_clip, hi: float = 1 - base_clip) -> float:
        return float(np.clip(p, lo, hi))

    def apply_temperature(p: float, T: float) -> float:
        return clip01(sigmoid(logit(clip01(p)) / float(T)))

    def weighted_logloss_sum(
        y_flat: np.ndarray, p_flat: np.ndarray, w_flat: np.ndarray
    ) -> float:
        p_flat = np.clip(p_flat, 1e-15, 1 - 1e-15)
        loss = -(y_flat * np.log(p_flat) + (1.0 - y_flat) * np.log(1.0 - p_flat))
        return float(np.sum(w_flat * loss))

    weights = {c: 1.0 for c in ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]}
    weights["patient_overall"] = 2.0
    w_row = np.array([weights[c] for c in target_cols], dtype=np.float64)

    rng = np.random.RandomState(42)
    idx = np.arange(n)
    y_patient = train_df["patient_overall"].to_numpy(dtype=np.int64)

    pos_idx = idx[y_patient == 1]
    neg_idx = idx[y_patient == 0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    pos_folds = np.array_split(pos_idx, 5)
    neg_folds = np.array_split(neg_idx, 5)
    folds = [np.sort(np.concatenate([pos_folds[i], neg_folds[i]])) for i in range(5)]

    y_full = train_df[target_cols].to_numpy(dtype=np.float64)
    full_sums = y_full.sum(axis=0)

    fold_val_sums = []
    fold_train_ns = []
    fold_val_y = []
    for val_idx in folds:
        val_s = y_full[val_idx].sum(axis=0)
        fold_val_sums.append(val_s)
        fold_train_ns.append(n - len(val_idx))
        fold_val_y.append(y_full[val_idx])

    def beta_smoothed_prev_from_sum(
        sum_pos: float, n_rows: int, a: float, b: float
    ) -> float:
        denom = float(n_rows + a + b)
        return clip01((float(sum_pos) + a) / denom)

    def cv_score_for_params_cached(
        a: float, T: float, Toverall: float, lam: float
    ) -> float:
        lam = float(np.clip(lam, 0.0, 1.0))
        total_loss = 0.0
        total_w = 0.0
        for k in range(len(folds)):
            n_tr = int(fold_train_ns[k])
            tr_sums = full_sums - fold_val_sums[k]

            c_ps = []
            for j in range(7):
                prev_c = beta_smoothed_prev_from_sum(tr_sums[j], n_tr, a=a, b=a)
                c_ps.append(apply_temperature(prev_c, T))

            p_any_from_c = 1.0 - float(np.prod([1.0 - p for p in c_ps]))
            prev_overall = beta_smoothed_prev_from_sum(tr_sums[7], n_tr, a=a, b=a)

            p_any_blend = lam * p_any_from_c + (1.0 - lam) * prev_overall
            p_any = apply_temperature(p_any_blend, Toverall)

            p_row = np.array(c_ps + [p_any], dtype=np.float64)

            y_val = fold_val_y[k]  # (n_val, 8)
            y_flat = y_val.reshape(-1)
            p_flat = np.repeat(p_row[np.newaxis, :], y_val.shape[0], axis=0).reshape(-1)
            w_flat = np.repeat(w_row[np.newaxis, :], y_val.shape[0], axis=0).reshape(-1)

            total_loss += weighted_logloss_sum(y_flat, p_flat, w_flat)
            total_w += float(np.sum(w_flat))
        return float(total_loss / total_w)

    smooth_grid = [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

    T_grid = [
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
        1.0,
        1.05,
        1.1,
        1.15,
        1.2,
        1.25,
        1.3,
        1.35,
        1.45,
        1.55,
        1.7,
        1.9,
        2.2,
        2.6,
    ]
    Toverall_grid = T_grid[:]

    lam_grid = [0.0, 0.25, 0.5, 0.75, 1.0]

    best = None  # (loss, a, T, Toverall, lam)

    for a_try in smooth_grid:
        for T_try in T_grid:
            for To_try in Toverall_grid:
                for lam_try in lam_grid:
                    loss = cv_score_for_params_cached(
                        float(a_try), float(T_try), float(To_try), float(lam_try)
                    )
                    cand = (
                        loss,
                        float(a_try),
                        float(T_try),
                        float(To_try),
                        float(lam_try),
                    )
                    if best is None or cand[0] < best[0]:
                        best = cand

    best_loss, best_ab, T, Toverall, lam = best

    prev = {}
    for j, c in enumerate(target_cols):
        prev[c] = beta_smoothed_prev_from_sum(full_sums[j], n, a=best_ab, b=best_ab)

    pred_map = {}
    c_probs = []
    for c in ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]:
        p_cal = apply_temperature(prev[c], T)
        pred_map[c] = float(p_cal)
        c_probs.append(float(p_cal))

    p_any_from_c = 1.0 - float(np.prod([1.0 - p for p in c_probs]))
    p_overall_prev = float(prev["patient_overall"])

    p_any_blend = (
        float(np.clip(lam, 0.0, 1.0)) * p_any_from_c
        + (1.0 - float(np.clip(lam, 0.0, 1.0))) * p_overall_prev
    )
    pred_map["patient_overall"] = float(apply_temperature(p_any_blend, Toverall))

    sub_df = test_df[["row_id", "prediction_type"]].copy()
    sub_df["fractured"] = (
        sub_df["prediction_type"].map(pred_map).astype("float32").clip(eps, 1 - eps)
    )
    sub_df = sub_df[["row_id", "fractured"]]

    assert sub_df.shape[0] == test_df.shape[0]
    assert sub_df["row_id"].is_unique

    sub_df.to_csv(SAVE_CSV, index=False)
    print(f"==> Fallback wrote {SAVE_CSV} with shape {sub_df.shape}")
    print("==> Selected Beta smoothing a=b via stratified 5-fold CV:", best_ab)
    print(
        "==> Selected (T, Toverall, lambda_blend) via grid-search CV (global weighted objective):",
        (T, Toverall, lam),
    )
    print(
        "==> Best (a=b, T, Toverall, lam, weighted loss):",
        (best_ab, T, Toverall, lam, best_loss),
    )
    print(
        "==> Using Beta-smoothed+temperature calibrated prevalences (with blended any-aggregation):",
        pred_map,
    )
    print(f"==> Finish using time: {time.time() - time_start:.2f}s")
