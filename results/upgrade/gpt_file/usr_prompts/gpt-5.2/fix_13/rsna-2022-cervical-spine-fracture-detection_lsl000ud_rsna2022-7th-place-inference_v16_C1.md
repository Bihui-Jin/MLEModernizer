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

0.2795580085814313

# 6. Current score

0.5639

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.10939) has done: 'I first fix the missing `srccode2` dependency by making the script robust to the Kaggle filesystem and providing a safe fallback that still produces a valid submission CSV even when external custom modules/models aren’t present. Then I correct the cell numbering (your notebook starts at cell 0, but the required format starts at cell 1) and ensure `submission.csv` is always written with the exact required columns (`row_id,fractured`) and correct row coverage matching `test.csv`. Finally, if the external code/models do exist in another expected location, the script automatically use them; otherwise it run a simple, deterministic baseline (constant probabilities) to avoid runtime failure and yield a valid file.'
- What this solution (achieved 0.56119) has done: 'Your current score (1.10939, lower-is-better) is far worse than the target (0.27956), and the main reason is that your script is almost certainly using the fallback constant predictions because the external model files aren’t available/resolved; that guarantees a poor weighted log loss. I keep your core external pipeline logic intact, but add a minimal, deterministic “train.csv prior” fallback that estimates per-label probabilities from the provided training labels (with light Laplace smoothing) and correctly ties `patient_overall` to the per-vertebra probabilities; this should move the score substantially toward the target without changing the evaluation semantics. I also make the external pipeline selection stricter (require the model files) so we don’t mistakenly attempt it and then silently fall back mid-way. The script still always write a valid `submission.csv` with the right columns/row alignment.'
- What this solution (achieved 0.569) has done: 'I keep your external pipeline untouched and instead make a minimal improvement to the fallback, because your current score (0.56119, lower-is-better) is still far from the target (0.27956) and is dominated by fallback behavior. The smallest legitimate gain is to compute per-label priors with lighter smoothing and to calibrate them slightly toward 0/1 (log-loss-optimal under label imbalance is sensitive to calibration), while still being a pure train-label-statistics baseline. I also set `patient_overall` prior from the actual `train.csv` column (rather than independence-from-C1..C7), which is more consistent with the label definition and typically reduces weighted log loss. Submission format, paths, and the rest of your logic stay identical.'
- What this solution (achieved 0.56001) has done: 'Your score is still dominated by the fallback path, so the most direct way to move toward the lower target loss is to make the fallback probabilities better aligned with the weighted log-loss optimum. I keep the exact same fallback concept (train-label priors) but (1) compute a per-study `patient_overall` from the per-level priors via `1-Π(1-p)` (better matches the label semantics) while also blending a little with the empirical `patient_overall` prior for stability, and (2) replace the current “gamma logit scaling” with a tiny grid-search on the training set (using the actual competition weighting) to pick a single global shrinkage temperature and clipping that reduces weighted log loss without changing any model/pipeline logic. External pipeline behavior, file paths, and submission schema stay unchanged; this only improves the fallback predictions used in your current run.'
- What this solution (achieved 0.56506) has done: 'We’re far above the target loss (0.56001 vs 0.27956, lower-is-better), and your runs are clearly dominated by the fallback (since external model files aren’t present), so the best minimal lever is improving the fallback calibration without changing the modeling approach. I keep the same “train-label prior” fallback, but make it (1) optimize the *actual weighted log-loss* more directly by searching a slightly richer but still tiny calibration space: separate temperatures for vertebra-level vs patient_overall, and (2) set patient_overall prediction as a calibrated blend of the empirical patient_overall prior and the union-from-levels prior, with the blend weight also chosen by the same weighted-logloss grid search. I also ensure row_id mapping is always correct by building predictions keyed by `StudyInstanceUID` + `prediction_type` directly (removes any risk of mismatch if formatting ever differs), while still producing the exact required `submission.csv`.'
- What this solution (achieved 0.56375) has done: 'Your current score (0.56506, lower-is-better) is still far above the target (0.27956), and since the external pipeline/models aren’t available, the only lever is making the fallback probabilities better match the competition’s weighted log-loss. I keep the same “train-prior fallback” core logic, but (1) compute patient_overall from the per-level probabilities after calibration (union), which better matches the label definition than calibrating it independently, and (2) choose separate label-specific temperatures for C1–C7 (not a new model—just post-calibration) via a tiny grid-search on the weighted log-loss. Finally, I make the fallback optimize on a per-label basis while keeping clipping and submission mapping unchanged, which should reduce loss noticeably without changing the overall approach.'
- What this solution (achieved 0.60059) has done: 'Your current loss (0.56375, lower-is-better) is far above the target (0.27956), and since the external pipeline/models aren’t available the only lever is improving the fallback probabilities while keeping the same “train-prior fallback” core approach. I make a minimal but meaningful change: instead of predicting constant per-label priors for every study, I still use only `train.csv` but fit a tiny per-study logistic regression on the 8 train-label columns (C1–C7 + patient_overall) using just one safe, DICOM-free feature: number of slices in the study folder (fast to compute and available for train/test). This keeps the architecture/training loops untouched (we’re only changing the fallback), but it typically improves log loss because slice count correlates with scan coverage/protocol and can indirectly correlate with fracture prevalence, especially for patient_overall which is heavily weighted. I also keep your existing calibrated-prior baseline as a backstop and blend it with the logistic model using a single blend weight chosen by the same weighted-logloss grid search on train, so we move loss downward without risking a collapse.'
- What this solution (achieved 0.55333) has done: 'You’re still far above the target loss (0.60059 vs 0.27956, lower-is-better), and since the external pipeline is not being used, the only realistic lever is improving the fallback without changing the overall approach. The biggest issue in the current fallback is that it builds the slice-count feature using `train_images_root`, which can silently point to `test_images` and yields all-zero slice counts for train UIDs—making the logistic component ineffective and potentially harmful. I minimally fix this by computing slice-count features from the correct roots for train and test separately, and I also change the final mapping so each `StudyInstanceUID` gets one set of 8 predictions (instead of accidentally aligning one per row), which is the intended evaluation semantics for this competition. Everything else (priors, calibration grid search, 1D logistic fitting, blend weight search, submission format/path) remains the same.'
- What this solution (achieved 0.55592) has done: 'Your current loss (0.55333, lower-is-better) is still far above the target (0.27956), and since the external pipeline/models are not available, we should only improve the fallback while keeping the same overall approach (train-label priors + slice-count 1D logistic + blending). The biggest low-risk gain is to (1) correctly optimize the blend weight under the competition’s *weighted* log loss by using the label weights during fitting (your current 1D logistic fit is unweighted and can mis-calibrate the heavily weighted `patient_overall`), and (2) add a tiny, explicit calibration step on the final mixed predictions (separate temperature for C1–C7 vs patient_overall, chosen by a small grid-search on train) without changing features or model structure. These are minimal post-processing changes that directly target the metric and should reduce loss toward the target band while preserving evaluation semantics. The script still runs end-to-end and always writes a valid `submission.csv` with the correct row alignment.'
- What this solution (achieved 0.55649) has done: 'We keep your existing fallback approach (train-label priors + 1D slice-count logistic + blending + temperature calibration), but make two minimal metric-aligned fixes to move log loss down toward the target. First, we fit the slice-count logistic models with *properly normalized sample weights* (so Newton steps don’t get distorted by the heavy `patient_overall` weight), and we search `l2` over a tiny grid to avoid over/under-regularization from the current fixed value. Second, we compute `patient_overall` consistently as a calibrated union-from-levels after blending (instead of mixing a “direct any” model with union), because the label definition is “any fracture” and this generally reduces weighted log loss for the heavily weighted row. These are small, local changes that preserve your core logic and still produce the same `submission.csv` schema and row alignment.'
- What this solution (achieved 0.55824) has done: 'Your current loss (0.55649, lower-is-better) is still far above the target (0.27956), and since the external pipeline almost certainly isn’t running, the only practical lever is improving the fallback while keeping its core structure (train-priors + 1D slice-count logistic + blending + calibration). I make two minimal, metric-aligned changes: (1) optimize the logistic-vs-prior blend weight per-label (instead of a single global weight), because `patient_overall` is weighted much more heavily and benefits from different mixing than C1–C7; and (2) compute the final `patient_overall` prediction as a blend between the union-from-levels and the direct patient_overall channel (with a tiny grid search), which reduces log-loss when independence is imperfect while still respecting the “any fracture” semantics. Everything else (features, fitting method, temperature scaling, clipping, output format/paths) stays the same, and the script still always writes a valid `submission.csv`.'
- What this solution (achieved 0.5639) has done: 'I keep your existing external-pipeline logic untouched and focus only on the fallback path (which your scores indicate is what’s being used). The current fallback’s slice-count logistic can’t generalize well and may be adding noise; instead, I make the fallback strictly metric-aligned by fitting per-label intercept-only probabilities that directly minimize the competition’s weighted log loss (a small, legitimate calibration improvement without changing evaluation semantics). I also compute `patient_overall` as the union from the calibrated C1–C7 probabilities (matching the label definition) and clip safely to avoid extreme log-loss penalties. This is a minimal change that should reduce loss (move toward 0.2796) while keeping runtime well under 600s and always producing a valid `submission.csv`.'

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

try:
    import torch
except Exception:
    torch = None

try:
    import SimpleITK as sitk
except Exception:
    sitk = None

BASE_INPUT = "/kaggle/data"
COMP_DIR = os.path.join(BASE_INPUT, "rsna-2022-cervical-spine-fracture-detection")
ALT_COMP_DIR = os.path.join(
    BASE_INPUT, "input", "rsna-2022-cervical-spine-fracture-detection"
)
if os.path.isdir(ALT_COMP_DIR) and not os.path.isdir(COMP_DIR):
    COMP_DIR = ALT_COMP_DIR

TEST_CSV = os.path.join(COMP_DIR, "test.csv")
TRAIN_CSV = os.path.join(COMP_DIR, "train.csv")
DATA_DIR = os.path.join(COMP_DIR, "test_images")
SAVE_CSV = "submission.csv"

assert os.path.isfile(TEST_CSV), f"test.csv not found at {TEST_CSV}"
assert os.path.isdir(DATA_DIR), f"test_images dir not found at {DATA_DIR}"
assert os.path.isfile(TRAIN_CSV), f"train.csv not found at {TRAIN_CSV}"


def _find_existing_src_path():
    candidates = [
        "../input/srccode2/src",
        "/kaggle/input/srccode2/src",
        "/kaggle/data/input/srccode2/src",
        os.path.join(BASE_INPUT, "srccode2", "src"),
        os.path.join(BASE_INPUT, "input", "srccode2", "src"),
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    return None


src_path = _find_existing_src_path()
HAS_EXTERNAL_PIPELINE = False

if src_path is not None:
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    try:
        from Utils.CommonTools import sitk_base
        from Utils.CommonTools.NiiIO import read_from_DICOM_dir
        from Utils.PreProcessing.resampling import sitk_dummy_3D_resample
        from Utils.CommonTools.bbox import get_bbox, extend_bbox
        from Utils.post_processing import keep_largest_cervical_cc
        from Utils.Inference.nnunet_inference import NNUnetCTPredictor
        from Utils.CommonTools.sitk_base import resample, copy_nii_info, get_nii_info
        from Training.Task_301_PostProcessing_Overall.model import Model as ModelStage3

        HAS_EXTERNAL_PIPELINE = True
        print(f"==> External pipeline import success from {src_path}")
    except Exception as e:
        HAS_EXTERNAL_PIPELINE = False
        print(
            f"==> External pipeline found at {src_path} but import failed; using fallback. Import error: {repr(e)}"
        )
else:
    print("==> External pipeline path not found; using fallback.")

np.random.seed(0)



## === cell 1
if HAS_EXTERNAL_PIPELINE:

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
                return np.mean(list_pred)

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
                        score[0] = self.predict_stage3(roi_pred_1, roi_pred_2)
                        self.results[case_id] = score

                        print(f"    {case_id}: {score}")

                    print(
                        f"    Finish this split use : {time.time() - time_start} seconds"
                    )
                    print(
                        f"    Overall use : {time.time() - overall_time_start} seconds"
                    )
                    gc.collect()




## === cell 2
time_start = time.time()

test_df = pd.read_csv(TEST_CSV)


def _weighted_logloss(
    y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray
) -> float:
    eps = 1e-7
    p = np.clip(y_pred, eps, 1.0 - eps)
    loss = -(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)) * weights[None, :]
    return float(loss.mean())


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    return np.log(p / (1.0 - p))


def _apply_temp(p: np.ndarray, t: float) -> np.ndarray:
    return _sigmoid(_logit(p) / float(t))


def _train_prior_fallback_submission(
    test_df: pd.DataFrame,
    train_csv_path: str,
) -> pd.DataFrame:
    train_df = pd.read_csv(train_csv_path)

    labels_lvl = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    all_labels = labels_lvl + ["patient_overall"]

    weights = np.array([1, 1, 1, 1, 1, 1, 1, 7], dtype=np.float64)

    y_true = train_df[all_labels].astype(np.float64).values  # (n, 8)

    p_raw = np.array([train_df[c].mean() for c in all_labels], dtype=np.float64)

    clip_grid = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    temp_lvl_grid = [0.70, 0.85, 1.00, 1.15, 1.30]
    temp_any_grid = [0.70, 0.85, 1.00, 1.15, 1.30]
    union_blend_grid = [0.0, 0.25, 0.50, 0.75, 1.0]  # 1.0 => pure union-from-levels

    best = None
    for clip in clip_grid:
        for t_lvl in temp_lvl_grid:
            p_lvl = _apply_temp(p_raw[:7], t_lvl)
            p_lvl = np.clip(p_lvl, clip, 1.0 - clip)

            p_any_union = 1.0 - float(np.prod(1.0 - p_lvl))

            for t_any in temp_any_grid:
                p_any_direct = float(_apply_temp(np.array([p_raw[7]]), t_any)[0])
                p_any_direct = float(np.clip(p_any_direct, clip, 1.0 - clip))

                for w_union in union_blend_grid:
                    p_any = w_union * p_any_union + (1.0 - w_union) * p_any_direct
                    p_any = float(np.clip(p_any, clip, 1.0 - clip))

                    p_row = np.concatenate([p_lvl, np.array([p_any], dtype=np.float64)])
                    y_pred = np.tile(p_row[None, :], (len(train_df), 1))
                    loss = _weighted_logloss(
                        y_true=y_true, y_pred=y_pred, weights=weights
                    )

                    cand = (loss, clip, t_lvl, t_any, w_union, p_row)
                    if (best is None) or (cand[0] < best[0]):
                        best = cand

    best_loss, best_clip, best_t_lvl, best_t_any, best_w_union, best_p_row = best

    lab_to_col = {f"C{i}": i - 1 for i in range(1, 8)}
    lab_to_col["patient_overall"] = 7

    fractured = np.zeros(len(test_df), dtype=np.float32)
    for i, ptype in enumerate(test_df["prediction_type"].values):
        fractured[i] = float(best_p_row[lab_to_col.get(ptype, 7)])

    sub = test_df[["row_id"]].copy()
    sub["fractured"] = fractured
    sub["fractured"] = sub["fractured"].clip(best_clip, 1.0 - best_clip)

    print(
        "==> Fallback (intercept-only, metric-aligned): "
        f"best_clip={best_clip}, t_lvl={best_t_lvl}, t_any={best_t_any}, "
        f"w_union={best_w_union}, train_weighted_logloss={best_loss:.6f}"
    )
    return sub


if HAS_EXTERNAL_PIPELINE and torch is not None and sitk is not None:

    def _resolve(p):
        if os.path.isfile(p):
            return p
        alt = p.replace("../input/", "/kaggle/input/")
        if os.path.isfile(alt):
            return alt
        alt2 = p.replace("../input/", "/kaggle/data/input/")
        if os.path.isfile(alt2):
            return alt2
        alt3 = p.replace("../input/", "/kaggle/data/")
        if os.path.isfile(alt3):
            return alt3
        return p

    list_model_C1_C7_segmentation = list(
        map(
            _resolve,
            [
                "../input/models2/models/stage1_0.model",
                "../input/models2/models/stage1_1.model",
                "../input/models2/models/stage1_2.model",
            ],
        )
    )
    plan_C1_C7_segmentation = _resolve("../input/plans-nnunet/stage1.pkl")

    list_model_fracture_detection = list(
        map(
            _resolve,
            [
                "../input/models2/models/stage2_0.model",
            ],
        )
    )
    list_model_post_processing = list(
        map(_resolve, ["../input/models2/models/stage3_0.pkl"])
    )
    plan_fracture_detection = _resolve("../input/plans-nnunet/stage2.pkl")

    have_all_files = (
        all(os.path.isfile(p) for p in list_model_C1_C7_segmentation)
        and os.path.isfile(plan_C1_C7_segmentation)
        and all(os.path.isfile(p) for p in list_model_fracture_detection)
        and os.path.isfile(plan_fracture_detection)
        and all(os.path.isfile(p) for p in list_model_post_processing)
    )

    if have_all_files and torch.cuda.is_available():
        device = torch.device("cuda:0")
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

            list_DICOM_dirs = [
                os.path.join(DATA_DIR, sub_dir) for sub_dir in os.listdir(DATA_DIR)
            ]
            print(f"==> Total {len(list_DICOM_dirs)} cases")

            list_size_DICOM_dirs = []
            for d in list_DICOM_dirs:
                size_ = 0
                for fn in os.listdir(d):
                    fp = os.path.join(d, fn)
                    try:
                        size_ += os.path.getsize(fp)
                    except OSError:
                        pass
                list_size_DICOM_dirs.append(size_)
            list_DICOM_dirs = list(
                np.array(list_DICOM_dirs)[np.argsort(list_size_DICOM_dirs)[::-1]]
            )
            print("==> Sort DICOM dirs by size")

            c2f_predictor.predict(list_test_files=list_DICOM_dirs)

            results = c2f_predictor.results

        pred_by_uid_type = {}
        for case_id, score in results.items():
            for C_i in range(1, 8):
                pred_by_uid_type[(case_id, f"C{C_i}")] = float(score[C_i])
            pred_by_uid_type[(case_id, "patient_overall")] = float(score[0])

        fractured = []
        for sid, ptype in zip(
            test_df["StudyInstanceUID"].values, test_df["prediction_type"].values
        ):
            fractured.append(pred_by_uid_type.get((sid, ptype), 0.05))

        sub = test_df[["row_id"]].copy()
        sub["fractured"] = np.asarray(fractured, dtype=np.float32)
        sub["fractured"] = sub["fractured"].clip(1e-6, 1.0 - 1e-6)
        sub.to_csv(SAVE_CSV, index=False)
        print(
            f"==> Wrote {SAVE_CSV} using external pipeline. Time: {time.time() - time_start:.1f}s"
        )
    else:
        HAS_EXTERNAL_PIPELINE = False

if not HAS_EXTERNAL_PIPELINE:
    sub = _train_prior_fallback_submission(
        test_df=test_df,
        train_csv_path=TRAIN_CSV,
    )
    sub.to_csv(SAVE_CSV, index=False)
    print(
        f"==> Wrote {SAVE_CSV} using intercept-only fallback predictions. Time: {time.time() - time_start:.1f}s"
    )

sub_check = pd.read_csv(SAVE_CSV)
assert list(sub_check.columns) == ["row_id", "fractured"]
assert len(sub_check) == len(test_df)
assert sub_check["fractured"].notna().all()
print("==> Submission sanity checks passed.")
