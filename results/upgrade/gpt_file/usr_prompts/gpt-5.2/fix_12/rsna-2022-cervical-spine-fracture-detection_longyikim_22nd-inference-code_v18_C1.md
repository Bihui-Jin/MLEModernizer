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

0.3601462000857207

# 6. Current score

4.42545

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.75092) has done: 'I remove the notebook-only `!pip/!cp` cell and instead make imports robust to the Kaggle environment by relying on the already-present `pydicom` and gracefully handling missing `pylibjpeg` (it’s optional for many DICOMs). I also fix missing/late imports and cell-order issues that caused `glob`, `DataLoader`, `ds`, and downstream variables to be undefined after the first exception. To ensure the pipeline completes within the time limit and always writes a valid `submission.csv`, I add a safe fallback path: if model code/weights aren’t available, produce calibrated constant probabilities derived from `train.csv` prevalence (score-improving vs all-zeros, yet still minimal/metric-aligned). Finally, I harden submission creation so `row_id` alignment is correct and the `fractured` column always exists before writing the CSV.'
- What this solution (achieved 4.42545) has done: 'Your current pipeline aggregates per-slice predictions to the patient level using `max`, which tends to over-predict positives and is usually suboptimal for weighted log loss; the smallest metric-aligned improvement is to switch that aggregation to a probability “OR” (noisy-OR) across slices, which better matches “any fracture somewhere in the scan”. I keep the model/inference code intact and only change the patient-level aggregation in a numerically stable way, then derive `patient_overall` as an OR across C1–C7 to stay consistent with the competition’s semantics. This should improve log loss (reduce it) toward the target without changing architecture, training, or feature extraction. The submission writing and row alignment remain unchanged and still produce a valid `submission.csv`.'
- What this solution (achieved 4.42579) has done: 'Your current loss (4.42545, lower-is-better) is far worse than the target (0.3601), so we should make the smallest high-impact fixes that are most likely to legitimately reduce weighted log loss without changing the model itself. The biggest issue is that the per-slice model outputs you aggregate are *detection scores* (not calibrated probabilities) and can exceed 1 after ensembling, which breaks the probabilistic assumptions behind log loss and makes `noisy-or` aggregation unstable/overconfident. I keep your architecture/inference unchanged, but (1) convert slice scores to bounded probabilities via a light calibration (sigmoid + clip), (2) change aggregation to a numerically-stable log-space noisy-or to avoid underflow/overflow, and (3) keep `patient_overall` consistent as an OR over C1–C7. These changes are minimal, metric-aligned, and should move the score substantially toward the target while still producing the same `submission.csv` schema.'
- What this solution (achieved 4.42545) has done: 'Your current log loss (4.42579) is far worse than the target (0.3601), so we should make the smallest metric-aligned fix that can drastically reduce loss without changing the model/feature pipeline. The biggest issue is that you are treating detector “scores” as logits and passing them through a sigmoid; EfficientDet scores are already probabilities in \[0,1\], so the extra sigmoid pushes them toward ~0.5 and makes the noisy-OR aggregation badly miscalibrated. I keep your inference and aggregation structure, but change `_score_to_prob` to an identity+clip (and remove temperature usage) so the slice probabilities remain meaningful, then keep the same numerically-stable log-space noisy-OR and patient_overall OR-over-C1..C7. This is a minimal change that should move the weighted log loss sharply down toward the target band while still writing a valid `submission.csv`.'
- What this solution (achieved 4.42545) has done: 'Your current loss (4.42545, lower-is-better) is far worse than the target (0.3601), so we should make a minimal, metric-aligned calibration fix rather than changing any model code. The biggest likely cause is that your per-slice “scores” are being averaged across detectors and then combined across slices, but they aren’t guaranteed to be valid probabilities and can exceed 1.0, which makes noisy-OR and log loss behave very badly. I keep your exact inference pipeline and aggregation structure, but (1) bound detector-derived slice outputs into \[0,1\] before any aggregation via a simple clip, and (2) make the noisy-OR computation explicitly robust to values near 0/1 using consistent clipping. This should substantially reduce catastrophic log-loss penalties and move the score toward the target while still producing the same valid `submission.csv`.'
- What this solution (achieved 4.42545) has done: 'Your current loss (4.42545, lower-is-better) is far worse than the target (0.3601), so we should focus on the smallest high-impact calibration bug that can legitimately collapse log-loss penalties without changing your model or data flow. Right now you *add* per-model probabilities into `batch_probs` and only divide by `len(det_models)` at the end; because `batch_probs` starts at `1e-2`, this makes the baseline prior effectively multiplied by the number of models, pushing many outputs to 1.0 after clipping and causing huge log-loss. I keep the exact same inference/aggregation core logic, but change the ensembling to be a true mean: accumulate into a separate tensor initialized at 0, average across models, then add the small epsilon floor once (not per-model). I also ensure the baseline eps is not double-counted for non-active slices, keeping submission formatting and row alignment unchanged.'
- What this solution (achieved 4.42545) has done: 'Your current loss is far above target (lower-is-better), so the smallest high-impact change is to fix a misalignment bug: your per-slice predictions are only written for every other slice (`active_indices * 2`), leaving many slices at the epsilon floor and causing the patient-level noisy-OR to be severely miscalibrated. I keep your exact segmentation/detection pipeline, but map predictions back to the correct slice indices using `idx_rgb = np.arange(0, g.shape[0], 2)` and assign `batch_probs[idx_rgb[active_indices]]` instead. I also make `pixel_spacings2` align with the same `idx_rgb` (not `pixel_spacings[::2]`, which currently includes the 2 “prev” slices and can be off by 1-2), which affects boundary sizing and thus downstream scores. These minimal indexing fixes should legitimately reduce weighted log loss substantially toward the target while preserving architecture, training approach, and submission format.'
- What this solution (achieved 4.42545) has done: 'Your current loss (4.42545) is far worse than the target (0.3601, lower-is-better), so the most likely “minimal but high-impact” fix is to correct a column-order mismatch that silently assigns the wrong probabilities to the wrong labels. Right now `predictions` is created as `[patient_overall, C1..C7]` but later you name the DataFrame columns as `["patient_overall","C1"..]` while the detection model outputs are in class indices `0..7` where `0` is background/none and `1..7` correspond to `C1..C7`; treating index 0 as `patient_overall` badly miscalibrate and blow up weighted log loss. I keep your entire model/inference/aggregation logic intact, but remap the 8-class slice outputs into the 8 competition labels by dropping background, computing `patient_overall` as an OR over C1–C7 at the slice level, and then applying the same patient-level noisy-OR aggregation as you already do. This preserves evaluation semantics and should materially reduce catastrophic log-loss penalties without changing architecture or training.'
- What this solution (achieved 4.42545) has done: 'Your current loss is far above the target (lower-is-better), so we should apply the smallest high-impact metric-aligned calibration fixes without changing the model architecture or inference flow. The main issue is that you add `eps_floor` twice (once when initializing `batch_probs`, and again when assigning `det_comp + eps_floor`), and you also allow detector-derived per-class “probabilities” to exceed 1.0 after ensembling—both can push many outputs to 1 and cause catastrophic weighted log-loss. I (1) remove the double-counting of `eps_floor`, and (2) clip the detector scores into a valid probability range *before* building `patient_overall` and before inserting into `batch_probs`. These changes preserve your core logic (same models, same aggregation approach, same submission schema) but should significantly reduce extreme overconfidence and move the score toward the target.'
- What this solution (achieved 4.42545) has done: 'Your score is far worse than the target (lower-is-better), so the smallest likely high-impact fix is to correct a semantic mismatch: your detection model is created with `num_classes=1`, yet downstream code assumes 8 classes (background + C1–C7) and uses segmentation-derived class indices to place scores into 8 bins. That mismatch makes almost all vertebra outputs effectively the epsilon floor, leading to very poor weighted log loss. I keep your exact pipeline (same models, same inference loop, same aggregation), but (1) set `num_classes=8` to match the intended class mapping, and (2) ensure the score inserted into the selected class is properly bounded as a probability before aggregation (clip only, no extra sigmoid). This should materially reduce log-loss toward the target while preserving core logic and still writing a valid `submission.csv`.'
- What this solution (achieved 4.42545) has done: 'Your current loss is far above the target (lower-is-better), so the smallest high-impact change is to fix a likely label/column semantic mismatch: the per-slice model output already contains a `patient_overall` value in column 0 that is not meaningful (it’s derived from a “background/none” bucket), and then you noisy-OR aggregate it across slices, which can badly miscalibrate the heavily-weighted `patient_overall` row. I keep your model inference and aggregation exactly as-is for C1–C7, but change the patient-level aggregation to compute `patient_overall` *only* from the aggregated C1–C7 via OR (and ignore the per-slice `patient_overall` column entirely). This preserves the competition semantics (“any fracture”) and should substantially reduce weighted log loss without changing architecture, training, or feature extraction. The submission writing stays identical and still produce a valid `submission.csv`.'

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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import torchvision.transforms.functional as TF

warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

DATA_ROOT = "../input/rsna-2022-cervical-spine-fracture-detection"
IMAGES_DIR = f"{DATA_ROOT}/test_images"
TRAIN_IMAGES_PATH = f"{DATA_ROOT}/train_images"
TEST_IMAGES_PATH = f"{DATA_ROOT}/test_images"

try:
    import pydicom as dicom
except Exception as e:
    raise RuntimeError("pydicom is required but could not be imported") from e

try:
    import pylibjpeg  # noqa: F401

    _HAS_PYLIBJPEG = True
except Exception:
    _HAS_PYLIBJPEG = False

print("pylibjpeg available:", _HAS_PYLIBJPEG)



## === cell 1
segmentation_checkpoint = (
    "../input/effdet-models/axial_segmentation_effseg_132508-epoch-100.pth"
)
axial_det_checkpoint1 = (
    "../input/effdet-models/axial_detection_effdet_134352-epoch-52.pth"
)
axial_det_checkpoint2 = (
    "../input/effdet-models/axial_detection_effdet_001015-epoch-150.pth"
)

EFFDET_ROOT = "../input/effdet-models"
effdet_path = os.path.join(EFFDET_ROOT, "effdet")
timm_path = os.path.join(EFFDET_ROOT, "timm-pytorch-image-models")
omega_path = os.path.join(EFFDET_ROOT, "omegaconf")
effunet_path = os.path.join(EFFDET_ROOT, "efficientunet-pytorch-0.0.6")

for p in [effdet_path, timm_path, omega_path, effunet_path]:
    if os.path.isdir(p):
        sys.path.append(p)

print("effdet_path exists:", os.path.isdir(effdet_path))
print("timm_path exists:", os.path.isdir(timm_path))
print("effunet_path exists:", os.path.isdir(effunet_path))




## === cell 2
def rescale_img_to_hu(dcm_ds):
    """Rescales the image to Hounsfield unit."""
    slope = float(getattr(dcm_ds, "RescaleSlope", 1.0))
    intercept = float(getattr(dcm_ds, "RescaleIntercept", 0.0))
    return dcm_ds.pixel_array.astype(np.float32) * slope + intercept


def normalize_hu_t(data):
    return np.clip(data, a_min=-2242.0, a_max=2242.0) / 2242.0


def load_dicom(path):
    """
    Attempts to read a DICOM file. If JPEG-compressed and pylibjpeg/gdcm are missing,
    this may fail; caller should handle exceptions.
    """
    ds = dicom.dcmread(path)
    img = rescale_img_to_hu(ds)
    pixel_spacing = float(ds.PixelSpacing[0]) if hasattr(ds, "PixelSpacing") else 1.0
    return img, pixel_spacing




## === cell 3
test_slices = glob.glob(f"{TEST_IMAGES_PATH}/*/*.dcm")
parsed = []
pat = re.compile(re.escape(TEST_IMAGES_PATH) + r"/([^/]+)/(\d+)\.dcm$")
for s in test_slices:
    m = pat.search(s)
    if m:
        parsed.append((m.group(1), int(m.group(2))))
df_test_slices = pd.DataFrame(parsed, columns=["StudyInstanceUID", "Slice"])
if df_test_slices.empty:
    raise RuntimeError(
        f"No DICOMs found under {TEST_IMAGES_PATH}. Check dataset mount/path."
    )

df_test_slices["Start"] = df_test_slices.groupby("StudyInstanceUID")["Slice"].transform(
    "min"
)
df_test_slices = df_test_slices.sort_values(["StudyInstanceUID", "Slice"]).reset_index(
    drop=True
)
print(df_test_slices.shape)
df_test_slices.head()




## === cell 4
class DcmDataSet(torch.utils.data.Dataset):
    def __init__(self, df, path, transforms=None, image_size=512):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.path = path
        self.transforms = transforms
        self.len = len(self.df)
        self.image_size = image_size
        self.transform = T.Resize((image_size, image_size))

    def __getitem__(self, i):
        s = self.df.iloc[i]
        gpath = os.path.join(self.path, s.StudyInstanceUID, f"{int(s.Slice)}.dcm")
        try:
            g, pixel_spacing = load_dicom(gpath)
            g = normalize_hu_t(g)
            x = torch.as_tensor(g, dtype=torch.float32).unsqueeze(0)  # 1 x H x W
            x = self.transform(x)
            is_start = bool(s.Slice == s.Start)
            return x, float(pixel_spacing), is_start
        except Exception:
            x = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)
            return x, 1.0, bool(s.Slice == s.Start)

    def __len__(self):
        return self.len


ds = DcmDataSet(df_test_slices, IMAGES_DIR)
batch_size = 16
dl = DataLoader(
    ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=min(os.cpu_count() or 2, batch_size),
    pin_memory=torch.cuda.is_available(),
)
x, pixel_spacings, is_start = next(iter(dl))
print(
    "batch:", x.shape, "pixel_spacing:", pixel_spacings[:3], "is_start:", is_start[:10]
)



## === cell 5
_HAS_MODELS = True
try:
    from efficientunet import get_efficientunet_b5  # type: ignore
except Exception:
    _HAS_MODELS = False

try:
    from effdet import create_model  # type: ignore
except Exception:
    _HAS_MODELS = False

print("Model code available:", _HAS_MODELS)


def get_axial_segmentation_model(checkpoint):
    model = get_efficientunet_b5(out_channels=2, concat_input=True, pretrained=False)
    state = torch.load(checkpoint, map_location=torch.device(device))
    model.load_state_dict(state["model"])
    model.eval()
    return model.to(device)


def get_axial_detection_model(checkpoint, image_size=512):
    model = create_model(
        "efficientdetv2_ds",
        bench_task="predict",
        num_classes=8,
        image_size=(image_size, image_size),
        pretrained=False,
        max_det_per_image=1,
    )
    state = torch.load(checkpoint, map_location=torch.device(device))
    model.load_state_dict(state["model"])
    model.eval()
    return model.to(device)


def get_axial_detection_models(checkpoints):
    models_list = []
    for checkpoint in checkpoints:
        models_list.append(get_axial_detection_model(checkpoint))
    return models_list


seg_model = None
det_models = None
if (
    _HAS_MODELS
    and os.path.exists(segmentation_checkpoint)
    and os.path.exists(axial_det_checkpoint1)
    and os.path.exists(axial_det_checkpoint2)
):
    seg_model = get_axial_segmentation_model(segmentation_checkpoint)
    det_models = get_axial_detection_models(
        [axial_det_checkpoint1, axial_det_checkpoint2]
    )
else:
    _HAS_MODELS = False
print("Models ready:", _HAS_MODELS)




## === cell 6
def get_axial_boundary_from_segmentation(
    seg, pixel_spacing, throw=100, tol=0.2, max_mm=100
):
    image_size = seg.shape[0]
    min_size = min(image_size, max_mm / float(pixel_spacing))

    rows, columns = seg.nonzero(as_tuple=True)
    if len(rows) == 0:
        return torch.tensor(
            [0, 0, image_size, image_size], device=seg.device, dtype=torch.float32
        )

    rows, _ = torch.sort(rows)
    columns, _ = torch.sort(columns)

    throw = min(len(rows) // 2, int(throw))
    xmin, xmax = columns[throw], columns[-throw]
    ymin, ymax = rows[throw], rows[-throw]

    w = (xmax - xmin) * (1 + tol)
    h = (ymax - ymin) * (1 + tol)
    new_size = max(float(w), float(h), float(min_size))
    new_size = min(float(image_size), float(new_size))

    xcenter, ycenter = (xmax + xmin) / 2, (ymax + ymin) / 2

    xmin2 = torch.minimum(
        torch.tensor(image_size - new_size, device=seg.device), xcenter - new_size / 2
    ).clamp(min=0)
    ymin2 = torch.minimum(
        torch.tensor(image_size - new_size, device=seg.device), ycenter - new_size / 2
    ).clamp(min=0)

    return torch.stack([xmin2, ymin2, xmin2 + new_size, ymin2 + new_size]).to(
        torch.float32
    )


def predict_seg(x, model, img_size=256):
    x = TF.resize(x, (img_size, img_size))
    logits = model(x)
    classification_score, mse_score = logits.sigmoid().chunk(2, dim=1)
    classification_pred = classification_score.gt(0.5).float()
    pred = classification_pred * mse_score
    return pred


def get_axial_boundary(segs, pixel_spacings, seg_img_size=256):
    boundary_list = []
    for i in range(segs.shape[0]):
        seg = segs[i, 0, :, :]
        boundary = get_axial_boundary_from_segmentation(
            seg,
            pixel_spacings[i],
            throw=int(100 / 512 * seg_img_size),
            tol=0.2,
            max_mm=100 / 512 * seg_img_size,
        )
        boundary_list.append(boundary)
    boundary_list = torch.stack(boundary_list, axis=0) * (512.0 / seg_img_size)
    return boundary_list


def predict_det(x, model):
    bboxes = model(x)  # N x max_det x 6 (x1,y1,x2,y2,score,class)
    return bboxes[:, 0, :]


def crop_resize_images(imgs_tensor, boundary_list, img_size=512):
    cropped_list = []
    for i in range(imgs_tensor.shape[0]):
        xmin, ymin, xmax, ymax = boundary_list[i, :]
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        if xmax <= xmin or ymax <= ymin:
            croped = imgs_tensor[i, :, :, :]
        else:
            croped = TF.crop(
                imgs_tensor[i, :, :, :],
                top=ymin,
                left=xmin,
                height=ymax - ymin,
                width=xmax - xmin,
            )
        croped = TF.resize(croped, (img_size, img_size))
        cropped_list.append(croped)
    return torch.stack(cropped_list, 0)


def get_original_bbox(bbox, boundary):
    scale = 512.0 / (boundary[:, [2]] - boundary[:, [0]]).clamp(min=1.0)
    org_bbox = bbox / scale
    org_bbox[:, 0] += boundary[:, 0]
    org_bbox[:, 1] += boundary[:, 1]
    org_bbox[:, 2] += boundary[:, 0]
    org_bbox[:, 3] += boundary[:, 1]
    return org_bbox


def get_bbox_class(seg, bbox):
    xmin, ymin, xmax, ymax = bbox.int()
    xmin = int(torch.clamp(xmin, 0, seg.shape[1] - 1))
    xmax = int(torch.clamp(xmax, 0, seg.shape[1]))
    ymin = int(torch.clamp(ymin, 0, seg.shape[0] - 1))
    ymax = int(torch.clamp(ymax, 0, seg.shape[0]))
    if xmax <= xmin or ymax <= ymin:
        return torch.tensor(0.0, device=seg.device)

    area = seg[ymin:ymax, xmin:xmax]
    mask = area > 0
    if mask.sum() == 0:
        return torch.tensor(0.0, device=seg.device)
    result = torch.mean(area[mask])
    result = torch.round(result / 0.125)
    return result


def get_bbox_class_list(seg_list, seg_bboxes):
    class_list = []
    for i in range(seg_list.shape[0]):
        class_index = get_bbox_class(seg_list[i, :, :], seg_bboxes[i, :])
        class_list.append(class_index)
    return torch.stack(class_list)


def get_class_score(scores, class_list, eps=1e-2):
    scores = torch.nan_to_num(scores, nan=eps, posinf=1.0, neginf=eps).clamp(
        eps, 1.0 - eps
    )
    result = scores.new_zeros((scores.shape[0], 8)) + eps
    class_list = torch.nan_to_num(class_list).long().clamp(min=0, max=7)
    result[torch.arange(scores.shape[0]), class_list] = scores
    return result




## === cell 7
def predict_with_models():
    with torch.no_grad():
        predictions = []

        x0, _, _ = ds[0]
        x1, _, _ = ds[1]
        x0, x1 = x0.to(device), x1.to(device)
        prev2 = torch.stack((x0, x1))

        for x, pixel_spacings, is_starts in dl:
            x = x.to(device)
            pixel_spacings = (
                pixel_spacings.to(device)
                if torch.is_tensor(pixel_spacings)
                else torch.tensor(pixel_spacings, device=device)
            )

            x = torch.cat((prev2, x), dim=0)

            r = x[:-2, :, :, :]
            g = x[1:-1, :, :, :]
            b = x[2:, :, :, :]

            start_indices = torch.argwhere(is_starts).reshape(-1)
            if start_indices.numel() > 0:
                r[start_indices, :, :, :] = b[start_indices, :, :, :]
                g[start_indices, :, :, :] = b[start_indices, :, :, :]

            prev2 = b[-2:, :, :, :]

            eps_floor = 1e-2
            batch_probs = x.new_zeros((g.shape[0], 8)) + eps_floor

            idx_rgb = torch.arange(0, g.shape[0], 2, device=device)

            xrgb = torch.cat(
                (r[idx_rgb, :, :, :], g[idx_rgb, :, :, :], b[idx_rgb, :, :, :]), dim=1
            )

            pixel_spacings2 = pixel_spacings[: g.shape[0]][idx_rgb]

            seg_result = predict_seg(xrgb, seg_model)  # N_rgb x 1 x 256 x 256

            active_indices = seg_result.sum(dim=[1, 2, 3]).nonzero().reshape(-1)
            if active_indices.numel() == 0:
                predictions.append(batch_probs.clamp(1e-6, 1.0 - 1e-6))
                continue

            x_act = xrgb[active_indices, :, :, :]
            seg_act = seg_result[active_indices, :, :, :]
            ps_act = pixel_spacings2[active_indices]

            axial_boundary = get_axial_boundary(seg_act, ps_act, seg_img_size=256)
            x_crop = crop_resize_images(x_act, axial_boundary)

            det_sum = x.new_zeros((active_indices.shape[0], 8))
            for det_model in det_models:
                det_result = predict_det(x_crop, det_model)
                bboxes, scores = (
                    get_original_bbox(det_result[:, :4], axial_boundary),
                    det_result[:, 4],
                )
                class_list = get_bbox_class_list(seg_act[:, 0, :, :], bboxes / 2)
                probs = get_class_score(scores, class_list)  # includes eps inside
                det_sum += probs

            det_mean = det_sum / float(len(det_models))
            det_mean = det_mean.clamp(1e-6, 1.0 - 1e-6)

            c = det_mean[:, 1:8].clamp(1e-6, 1.0 - 1e-6)
            po = 1.0 - torch.prod(
                (1.0 - c).clamp(1e-6, 1.0 - 1e-6), dim=1, keepdim=True
            )
            det_comp = torch.cat([po, c], dim=1).clamp(1e-6, 1.0 - 1e-6)

            batch_probs[idx_rgb[active_indices], :] = det_comp

            out = batch_probs.clamp(1e-6, 1.0 - 1e-6)
            predictions.append(out)

        return torch.cat(predictions).cpu().numpy()


def predict_baseline_calibrated():
    """
    Score-oriented minimal baseline: use train prevalence per label, clipped.
    This is legitimate (no leakage) and typically beats all-zeros on logloss.
    """
    df_train = pd.read_csv(f"{DATA_ROOT}/train.csv")
    cols = ["patient_overall"] + [f"C{i}" for i in range(1, 8)]
    priors = df_train[cols].mean().values.astype(np.float32)
    priors = np.clip(priors, 1e-3, 1 - 1e-3)

    n = len(df_test_slices)
    preds = np.tile(priors.reshape(1, -1), (n, 1))
    return preds


if _HAS_MODELS:
    predictions = predict_with_models()
else:
    predictions = predict_baseline_calibrated()

print("predictions shape:", predictions.shape)



## === cell 8
df_effnet_pred = pd.DataFrame(
    predictions, columns=["patient_overall"] + [f"C{i}" for i in range(1, 8)]
)

df_test_pred = pd.concat(
    [df_test_slices[["StudyInstanceUID", "Slice"]], df_effnet_pred], axis=1
).sort_values(["StudyInstanceUID", "Slice"])


def _score_to_prob(p, clip=1e-6):
    p = np.asarray(p, dtype=np.float64)
    p = np.nan_to_num(p, nan=clip, posinf=1.0 - clip, neginf=clip)
    return np.clip(p, clip, 1.0 - clip)


def noisy_or_aggregate(group_df, cols, clip=1e-3):
    raw = group_df[cols].to_numpy(dtype=np.float64)
    prob = _score_to_prob(raw, clip=clip)

    log_prod_not = np.sum(np.log1p(-prob), axis=0)
    agg = 1.0 - np.exp(log_prod_not)
    agg = np.clip(agg, clip, 1.0 - clip)
    return pd.Series(agg, index=cols)


clip_value = 1e-3

c_cols = [f"C{i}" for i in range(1, 8)]

df_c_agg = (
    df_test_pred.groupby("StudyInstanceUID", sort=False)
    .apply(lambda g: noisy_or_aggregate(g, c_cols, clip=clip_value))
    .astype(np.float32)
)

for c in c_cols:
    df_c_agg[c] = df_c_agg[c].clip(lower=clip_value, upper=1 - clip_value)

df_patient_pred = df_c_agg.copy()
df_patient_pred["patient_overall"] = (
    1.0 - (1.0 - df_patient_pred[c_cols].clip(clip_value, 1 - clip_value)).prod(axis=1)
).clip(lower=clip_value, upper=1 - clip_value)

df_patient_pred = df_patient_pred[["patient_overall"] + c_cols]
df_patient_pred.head()



## === cell 9
df_test = pd.read_csv(f"{DATA_ROOT}/test.csv")

if len(df_test) > 0 and df_test.iloc[0].row_id == "1.2.826.0.1.3680043.10197_C1":
    df_test = pd.DataFrame(
        {
            "row_id": [
                "1.2.826.0.1.3680043.22327_C1",
                "1.2.826.0.1.3680043.25399_C1",
                "1.2.826.0.1.3680043.5876_patient_overall",
            ],
            "StudyInstanceUID": [
                "1.2.826.0.1.3680043.22327",
                "1.2.826.0.1.3680043.25399",
                "1.2.826.0.1.3680043.5876",
            ],
            "prediction_type": ["C1", "C1", "patient_overall"],
        }
    )

df_sub = df_test.set_index("StudyInstanceUID").join(df_patient_pred, how="left")

if df_sub[[c for c in df_patient_pred.columns]].isna().any().any():
    df_train = pd.read_csv(f"{DATA_ROOT}/train.csv")
    cols = ["patient_overall"] + [f"C{i}" for i in range(1, 8)]
    priors = np.clip(df_train[cols].mean(), 1e-3, 1 - 1e-3).to_dict()
    for c in cols:
        df_sub[c] = df_sub[c].fillna(priors[c])

df_sub["fractured"] = df_sub.apply(lambda r: float(r[r.prediction_type]), axis=1)
df_out = df_sub.reset_index(drop=False)[["row_id", "fractured"]]

df_out["fractured"] = df_out["fractured"].clip(1e-3, 1 - 1e-3)

df_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_out.shape)
print(df_out.head())
