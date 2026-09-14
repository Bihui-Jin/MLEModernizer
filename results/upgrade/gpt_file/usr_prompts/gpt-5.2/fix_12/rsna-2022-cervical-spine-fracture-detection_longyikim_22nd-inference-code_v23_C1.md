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

0.56278

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5639) has done: 'I remove the failing dependency bootstrap cell and make the script run with only packages available in the Kaggle image by using safe optional imports and fallbacks. Since the provided external model code/checkpoints (effdet/yolov7/efficientunet/pylibjpeg) are not available at the given paths, I preserve the end-to-end prediction→aggregation→submission semantics by producing calibrated constant probabilities derived from `train.csv` label prevalences (a legitimate baseline that avoids runtime crashes). I also fix pathing to the provided dataset location, ensure all missing imports/variables are defined in-order, and guarantee that `submission.csv` is written with the exact required columns (`row_id, fractured`) and correct row alignment to `test.csv`. This should yield a valid submission (current score was “Not yielded”) and move the score toward the target by using prevalence-based probabilities rather than arbitrary constants.'
- What this solution (achieved 0.56032) has done: 'Your current submission is a pure prevalence baseline; to move the logloss down toward the target with minimal risk, we keep the exact same “constant-per-study” logic but improve calibration in two small, metric-aligned ways. First, we compute out-of-fold (leave-one-out) prevalences per label so each study’s own label doesn’t slightly bias the prior (this usually reduces logloss vs using the full-data mean). Second, we set `patient_overall` deterministically from the vertebra probabilities using `p_any = 1 - Π(1 - p_Ck)`, which better matches the semantics of “any fracture” without changing the overall approach. We keep clipping (required for stable logloss) and preserve the submission alignment checks and output format.'
- What this solution (achieved 0.56032) has done: 'To move your (lower-is-better) logloss down toward the target with minimal change, I keep the same “constant-per-study” prevalence baseline but fix two calibration issues that typically help weighted logloss. First, I use your already-computed leave-one-out (LOO) prevalences for the 202 train studies as the per-study prior when that StudyInstanceUID appears in test (public test includes those rows), while keeping the global prevalence for unseen studies; this stays within the same logic but is a strictly better prior than the global mean for known IDs. Second, I compute `patient_overall` from the C1–C7 probabilities using `1 - Π(1-p)` after joining/filling, so it’s consistent for both LOO and global cases. I also keep probability clipping to avoid infinities in logloss and keep the submission alignment checks unchanged.'
- What this solution (achieved 0.56119) has done: 'Your current solution is a prevalence-only baseline; to move the (lower-is-better) logloss down toward the target with minimal changes, we keep the same constant-per-study prediction logic but improve calibration in a metric-aligned way. Specifically, we apply Laplace/Beta smoothing to the per-label prevalences (and the LOO prevalences) to reduce overconfident extremes on rare labels, and we use a slightly tighter probability clip to further protect logloss. We keep your “patient_overall from any(C1–C7)” construction intact, but compute it after smoothing so it remains consistent. The script still run end-to-end and write a valid `submission.csv` with the same schema and row alignment checks.'
- What this solution (achieved 0.56262) has done: 'We keep your prevalence-based baseline intact (same per-study constant probabilities + LOO override + `patient_overall = 1-Π(1-p_Ck)`) and only adjust calibration to move logloss downward toward the 0.435 target. The minimal, metric-aligned change is to tune the Beta/Laplace smoothing strength and clipping, because weighted logloss heavily penalizes overconfident wrong probabilities—especially for the high-weight `patient_overall`. Concretely, we slightly increase `(alpha, beta)` to pull rare-label probabilities closer to the global mean and use a slightly tighter clip to reduce catastrophic logloss, while preserving the exact submission construction and row alignment. This should improve score modestly without changing the modeling approach or any training/feature logic.'
- What this solution (achieved 0.56321) has done: 'We keep your exact prevalence/LOO baseline and submission construction, and only make a minimal, metric-aligned calibration adjustment to move the (lower-is-better) weighted logloss down toward the target. Specifically, we increase the Beta smoothing slightly to reduce overconfident probabilities on rare labels (which weighted logloss punishes heavily), and we tighten clipping a bit to prevent extreme probabilities from creating large penalties. We also ensure `patient_overall` is computed from the (smoothed, clipped) C1–C7 probabilities everywhere, preserving your current semantics. No model/feature/training logic is added; the script still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.56444) has done: 'To move your (lower-is-better) weighted logloss closer to the 0.435 target from 0.56321 without changing the core “prevalence/LOO baseline” logic, I keep the same per-study constant probabilities but tune calibration in the safest way: adjust the Beta/Laplace smoothing strength and the probability clipping. Your current settings (ALPHA=4, BETA=14, CLIP=2e-4) are still slightly overconfident for rare labels; increasing smoothing and slightly tightening clipping typically reduces catastrophic penalties in logloss, especially on the high-weight `patient_overall`. I not add any model, features, or training—only these minimal metric-aligned calibration constants—while preserving the exact submission construction and alignment checks.'
- What this solution (achieved 0.76224) has done: 'We keep your exact prevalence/LOO baseline and submission construction, and only apply a minimal, metric-aligned calibration change to reduce weighted logloss (lower-is-better) toward the 0.435 target. The safest lever here is shrinking probabilities toward 0.5 (temperature/logit scaling), which reduces overconfident penalties without changing any modeling/training logic. We apply this shrinkage only to the C1–C7 probabilities, then recompute `patient_overall` via `1 - Π(1-p)` as you already do, and finally clip to keep logloss finite. This stays within the same semantics (constant per-study priors) but should modestly improve calibration and score.'
- What this solution (achieved 0.63645) has done: 'I fix the runtime error caused by the removed `DataFrame.lookup` API by replacing it with a stable vectorized row-wise selection using NumPy indexing, which preserves the exact prediction semantics. Because cell 25 crashed before creating the `fractured` column, cell 26 also failed; fixing the selection step unblock submission writing. I also add a small safety clip after selection to guarantee probabilities are within `(CLIP_VALUE, 1-CLIP_VALUE)` for logloss stability, without changing the core prevalence/LOO + shrinkage logic. The script then run end-to-end and write a valid `submission.csv` with the required columns and correct row alignment.'
- What this solution (achieved 0.56278) has done: 'Your current score (0.63645, lower-is-better) is worse than the target (0.43526), so we need a cautious improvement without changing the core “prevalence/LOO constant per-study” logic. The biggest likely issue is that you shrink C1–C7 toward 0.5 and then recompute `patient_overall` from them, which can badly miscalibrate the high-weight patient label versus the true patient prevalence prior you already computed. I keep your exact prevalence + LOO construction, but apply the shrinkage only to the vertebrae labels while leaving the patient prior as-is (except for a very light blend with the any-from-vertebrae probability), which typically reduces weighted logloss. Finally, I keep the same clipping and submission alignment to ensure a valid `submission.csv`.'

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

random.seed(0)
np.random.seed(0)

try:
    import torch  # noqa: F401

    device = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    device = "cpu"

device



## === cell 1
BASE_DIR = "/kaggle/data/rsna-2022-cervical-spine-fracture-detection"

TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TEST_CSV = os.path.join(BASE_DIR, "test.csv")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")

TRAIN_IMAGES_PATH = os.path.join(BASE_DIR, "train_images")
TEST_IMAGES_PATH = os.path.join(BASE_DIR, "test_images")
IMAGES_DIR = TEST_IMAGES_PATH

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(TEST_CSV), f"Missing {TEST_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"



## === cell 2
segmentation_checkpoint = (
    "../input/effdet-models/axial_segmentation_effseg_132508-epoch-100.pth"
)
axial_det_checkpoint1 = (
    "../input/effdet-models/axial_detection_effdet_134352-epoch-52.pth"
)
axial_det_checkpoint2 = (
    "../input/effdet-models/axial_detection_effdet_001015-epoch-150.pth"
)
axial_yolo_checkpoint1 = "../input/effdet-models/yolo_custom4_epoch_099.pt"



## === cell 3
yolo_model = None
try:
    yolo_path = "../input/effdet-models/yolov7"
    if os.path.isdir(yolo_path):
        sys.path.append(yolo_path)
        from models.experimental import attempt_load  # type: ignore

        import torch

        yolo_model = attempt_load(axial_yolo_checkpoint1, map_location=device).eval()
except Exception as e:
    yolo_model = None
    yolo_import_error = str(e)

yolo_model is not None



## === cell 4
df_test = pd.read_csv(TEST_CSV)

df_test_slices = (
    df_test[["StudyInstanceUID"]]
    .drop_duplicates()
    .assign(Slice=0, Start=0)
    .reset_index(drop=True)
)

df_test_slices.head()



## === cell 5
df_test_slices = df_test_slices.copy()
df_test_slices.head()




## === cell 6
def rescale_img_to_hu(dcm_ds):
    return dcm_ds.pixel_array * dcm_ds.RescaleSlope + dcm_ds.RescaleIntercept


def normalize_hu(data):
    data = np.clip(data, a_min=-2242, a_max=2242) / 4484 + 0.5
    return data


def normalize_hu_t(data):
    return np.clip(data, a_min=-2242.0, a_max=2242.0) / 2242.0


def normalize_hu_tensor(data):
    import torch

    return torch.clip(data, min=-2242.0, max=2242.0) / 2242.0


def load_dicom(path):
    import pydicom as dicom

    ds = dicom.dcmread(path)
    img = rescale_img_to_hu(ds)
    return img, float(ds.PixelSpacing[0])




## === cell 7
try:
    import torch
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF
    from torch.utils.data import DataLoader, Dataset
except Exception:
    torch = None
    T = None
    TF = None
    DataLoader = None
    Dataset = object


class DcmDataSet(Dataset):
    def __init__(self, df, path, image_size=512):
        self.df = df
        self.path = path
        self.len = len(self.df)
        self.image_size = image_size
        self.transform = T.Resize((image_size, image_size)) if T is not None else None

    def __getitem__(self, i):
        if torch is None:
            raise RuntimeError(
                "Torch/torchvision not available for DICOM dataset reading."
            )
        s = self.df.iloc[i]
        gpath = os.path.join(self.path, s.StudyInstanceUID, f"{int(s.Slice)}.dcm")
        g, pixel_spacing = load_dicom(gpath)
        g = normalize_hu_t(g)
        x = torch.as_tensor(g, dtype=torch.float32).unsqueeze(0)
        if self.transform is not None:
            x = self.transform(x)
        return x, pixel_spacing, bool(s.Slice == s.Start)

    def __len__(self):
        return self.len


ds = None  # not used in fallback



## === cell 8
batch_size = 16
dl = None



## === cell 9
seg_model = None
try:
    effunet_path = "../input/effdet-models/efficientunet-pytorch-0.0.6"
    if os.path.isdir(effunet_path):
        sys.path.append(effunet_path)
        from efficientunet import get_efficientunet_b5  # type: ignore

        def get_axial_segmentation_model(checkpoint):
            import torch

            model = get_efficientunet_b5(
                out_channels=2, concat_input=True, pretrained=False
            )
            state = torch.load(checkpoint, map_location=torch.device(device))
            model.load_state_dict(state["model"])
            model.eval()
            return model.to(device)

        if os.path.exists(segmentation_checkpoint):
            seg_model = get_axial_segmentation_model(segmentation_checkpoint)
except Exception as e:
    seg_model = None
    seg_import_error = str(e)

seg_model is not None



## === cell 10
effdet_models = []
try:
    effdet_path = "../input/effdet-models/effdet"
    if os.path.isdir(effdet_path):
        sys.path.append(effdet_path)
        from effdet import create_model  # type: ignore

        import torch

        def get_axial_detection_model(checkpoint, image_size=512):
            model = create_model(
                "efficientdetv2_ds",
                bench_task="predict",
                num_classes=1,
                image_size=(image_size, image_size),
                pretrained=False,
                max_det_per_image=1,
            )
            state = torch.load(checkpoint, map_location=torch.device(device))
            model.load_state_dict(state["model"])
            return model.eval().to(device)

        if os.path.exists(axial_det_checkpoint1):
            effdet_models = [get_axial_detection_model(axial_det_checkpoint1)]
except Exception as e:
    effdet_models = []
    effdet_import_error = str(e)

effdet_models



## === cell 11
IMAGE_SIZES = [640, 512]
det_model_names = ["yolo", "effdet"]
det_models = ([yolo_model] if yolo_model is not None else []) + effdet_models




## === cell 12
def get_axial_boundary_from_segmentation(
    seg, pixel_spacing, throw=100, tol=0.2, max_mm=100
):
    import torch

    image_size = seg.shape[0]
    min_size = min(image_size, max_mm / pixel_spacing)

    rows, columns = seg.nonzero(as_tuple=True)
    rows.sort()
    columns.sort()

    throw = min(len(rows) // 2, throw)

    if (len(rows)) == 0:
        return torch.tensor([0, 0, image_size, image_size]).to(seg.device)

    xmin, xmax = columns[throw], columns[-throw]
    ymin, ymax = rows[throw], rows[-throw]

    w = (xmax - xmin) * (1 + tol)
    h = (ymax - ymax) * (1 + tol)
    new_size = max(w, h, min_size)
    new_size = min(image_size, new_size)

    xcenter, ycenter = (xmax + xmin) / 2, (ymax + ymin) / 2

    xmin = torch.min(
        torch.tensor(image_size - new_size, device=seg.device), xcenter - new_size / 2
    )
    xmin = xmin.clip(min=0)

    ymin = torch.min(
        torch.tensor(image_size - new_size, device=seg.device), ycenter - new_size / 2
    )
    ymin = ymin.clip(min=0)

    return torch.stack([xmin, ymin, xmin + new_size, ymin + new_size])




## === cell 13
def predict_seg(x, model, seg_img_size=256):
    import torchvision.transforms.functional as TF

    x = TF.resize(x, (seg_img_size, seg_img_size))
    logits = model(x)
    classification_score, mse_score = logits.sigmoid().chunk(2, dim=1)
    classification_pred = classification_score.gt(0.5).float()
    pred = classification_pred * mse_score
    return pred




## === cell 14
def get_axial_boundary_from_seg(segs, pixel_spacings, seg_img_size=256):
    import torch

    boundary_list = []
    for i in range(segs.shape[0]):
        seg = segs[i, 0, :, :]
        boundary = get_axial_boundary_from_segmentation(
            seg,
            float(pixel_spacings[i]),
            throw=int(100.0 / 512.0 * seg_img_size),
            tol=0.2,
            max_mm=100.0 / 512.0 * seg_img_size,
        )
        boundary_list.append(boundary)
    boundary_list = torch.stack(boundary_list, axis=0) * (512.0 / seg_img_size)
    return boundary_list




## === cell 15
def convert_yolo_result(pred):
    import torch

    max_indices = torch.argmax(pred[:, :, 4], dim=1)
    max_values = pred[torch.arange(pred.shape[0]), max_indices, :]
    bboxes, scores = max_values[:, :4], max_values[:, 4]
    bboxes[:, 2] += bboxes[:, 0]
    bboxes[:, 3] += bboxes[:, 1]
    return bboxes, scores


def predict_det(x, model):
    pred_result = model(x)
    if isinstance(pred_result, tuple) is True:
        return convert_yolo_result(pred_result[0])
    else:
        return pred_result[:, 0, :4], pred_result[:, 0, 4]


def crop_resize_images(imgs_tensor, boundary_list, img_size=512):
    import torchvision.transforms.functional as TF
    import torch

    croped_list = []
    for i in range(imgs_tensor.shape[0]):
        xmin, ymin, xmax, ymax = boundary_list[i, :]
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        croped = TF.crop(
            imgs_tensor[i, :, :, :],
            top=ymin,
            left=xmin,
            height=ymax - ymin,
            width=xmax - xmin,
        )
        croped = TF.resize(croped, (img_size, img_size))
        croped_list.append(croped)
    return torch.stack(croped_list, 0)


def get_original_bbox(bbox, boundary, image_size=512.0):
    scale = image_size / (boundary[:, [2]] - boundary[:, [0]])
    org_bbox = bbox / scale
    org_bbox[:, 0] += boundary[:, 0]
    org_bbox[:, 1] += boundary[:, 1]
    org_bbox[:, 2] += boundary[:, 0]
    org_bbox[:, 3] += boundary[:, 1]
    return org_bbox




## === cell 16
def get_bbox_class(seg, bbox):
    import torch

    xmin, ymin, xmax, ymax = bbox.int()
    area = seg[ymin:ymax, xmin:xmax]
    result = torch.mean(area[area > 0])
    result = torch.round(result / 0.125)
    return result




## === cell 17
def get_bbox_class_list(seg_list, seg_bboxes):
    import torch

    class_list = []
    for i in range(seg_list.shape[0]):
        class_index = get_bbox_class(seg_list[i, :, :], seg_bboxes[i, :])
        class_list.append(class_index)
    return torch.stack(class_list)




## === cell 18
def get_class_score(scores, class_list, eps=1e-2):
    import torch

    result = scores.new_zeros((scores.shape[0], 8)) + eps
    class_list = torch.nan_to_num(class_list).long()
    result[torch.arange(scores.shape[0]), class_list] = scores
    return result


def check_detection_result(det_result, img_size=512.0, threshold=0.2):
    import torch

    areas = (
        (det_result[:, 2] - det_result[:, 0])
        * (det_result[:, 3] - det_result[:, 1])
        / (img_size * img_size)
    )
    big_indices = torch.argwhere(areas > threshold)
    det_result[big_indices, 4] = 0.0
    return det_result




## === cell 19
def cal_loss(prob, label):
    pos_weight = np.array([14, 2, 2, 2, 2, 2, 2, 2])
    neg_weight = np.array([7, 1, 1, 1, 1, 1, 1, 1])
    score = pos_weight * label * np.log(prob) + neg_weight * (1 - label) * np.log(
        1 - prob
    )
    weight_total = pos_weight * label + neg_weight * (1 - label)
    return -score.sum(axis=1) / weight_total.sum(axis=1)




## === cell 20
def compute_leave_one_out_prevalences(
    train_csv: str, clip_value: float = 5e-4, alpha: float = 6.0, beta: float = 24.0
) -> pd.DataFrame:
    df_train = pd.read_csv(train_csv)
    target_cols = ["patient_overall"] + [f"C{i}" for i in range(1, 8)]

    n = len(df_train)
    sums = df_train[target_cols].sum(axis=0).astype(float)

    loo = pd.DataFrame(
        index=df_train["StudyInstanceUID"], columns=target_cols, dtype=float
    )
    denom = (n - 1) + alpha + beta
    denom = max(float(denom), 1.0)

    for c in target_cols:
        loo[c] = (sums[c] - df_train[c].astype(float).values + alpha) / denom

    loo = loo.clip(clip_value, 1 - clip_value)
    loo.index.name = "StudyInstanceUID"
    return loo


def predict_patient_level_from_prevalence(
    train_csv: str, clip_value: float = 5e-4, alpha: float = 6.0, beta: float = 24.0
) -> pd.DataFrame:
    df_train = pd.read_csv(train_csv)
    target_cols = ["patient_overall"] + [f"C{i}" for i in range(1, 8)]

    n = len(df_train)
    sums = df_train[target_cols].sum(axis=0).astype(float)

    prevalences = (sums + alpha) / (float(n) + alpha + beta)
    prevalences = prevalences.clip(clip_value, 1 - clip_value)
    return pd.DataFrame([prevalences.values], columns=target_cols)


CLIP_VALUE = 5e-4
ALPHA = 6.0
BETA = 24.0

df_prev = predict_patient_level_from_prevalence(
    TRAIN_CSV, clip_value=CLIP_VALUE, alpha=ALPHA, beta=BETA
)
df_loo = compute_leave_one_out_prevalences(
    TRAIN_CSV, clip_value=CLIP_VALUE, alpha=ALPHA, beta=BETA
)

df_prev, df_loo.head()




## === cell 21
def set_patient_overall_from_any(
    df_probs: pd.DataFrame, clip_value: float = 5e-4
) -> pd.DataFrame:
    df_probs = df_probs.copy()
    c_cols = [f"C{i}" for i in range(1, 8)]
    p_any = 1.0 - np.prod(1.0 - df_probs[c_cols].values, axis=1)
    p_any = np.clip(p_any, clip_value, 1 - clip_value)
    df_probs["patient_overall"] = p_any.astype(float)
    return df_probs


def shrink_probs_toward_half(p: np.ndarray, temperature: float = 1.35) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, CLIP_VALUE, 1.0 - CLIP_VALUE)
    logit = np.log(p / (1.0 - p))
    logit = logit / float(temperature)
    out = 1.0 / (1.0 + np.exp(-logit))
    return np.clip(out, CLIP_VALUE, 1.0 - CLIP_VALUE)


TEMP = 1.35  # keep your current calibration setting

PATIENT_BLEND_LAMBDA = 0.15  # small blend to avoid drastic behavior change



## === cell 22
df_test_studies = df_test[["StudyInstanceUID"]].drop_duplicates().reset_index(drop=True)

target_cols = ["patient_overall"] + [f"C{i}" for i in range(1, 8)]
c_cols = [f"C{i}" for i in range(1, 8)]

df_patient_pred = df_test_studies.copy().set_index("StudyInstanceUID")
for c in target_cols:
    df_patient_pred[c] = float(df_prev.iloc[0][c])

overlap_uids = df_patient_pred.index.intersection(df_loo.index)
if len(overlap_uids) > 0:
    df_patient_pred.loc[overlap_uids, target_cols] = df_loo.loc[
        overlap_uids, target_cols
    ]

df_patient_pred[c_cols] = shrink_probs_toward_half(
    df_patient_pred[c_cols].values, temperature=TEMP
)

p_any = 1.0 - np.prod(1.0 - df_patient_pred[c_cols].values, axis=1)
p_any = np.clip(p_any, CLIP_VALUE, 1.0 - CLIP_VALUE)
p_prior = np.clip(
    df_patient_pred["patient_overall"].values.astype(np.float64),
    CLIP_VALUE,
    1.0 - CLIP_VALUE,
)
df_patient_pred["patient_overall"] = np.clip(
    (1.0 - PATIENT_BLEND_LAMBDA) * p_prior + PATIENT_BLEND_LAMBDA * p_any,
    CLIP_VALUE,
    1.0 - CLIP_VALUE,
).astype(float)

df_patient_pred.head()



## === cell 23
pass



## === cell 24
df_test = pd.read_csv(TEST_CSV)
df_test.head()



## === cell 25
df_sub = df_test.set_index("StudyInstanceUID").join(df_patient_pred, how="left")

fill_map = df_prev.iloc[0].to_dict()
for c in target_cols:
    df_sub[c] = df_sub[c].fillna(fill_map[c]).astype(float)

p_any_row = 1.0 - np.prod(1.0 - df_sub[c_cols].to_numpy(dtype=np.float64), axis=1)
p_any_row = np.clip(p_any_row, CLIP_VALUE, 1.0 - CLIP_VALUE)
p_prior_row = np.clip(
    df_sub["patient_overall"].to_numpy(dtype=np.float64), CLIP_VALUE, 1.0 - CLIP_VALUE
)
df_sub["patient_overall"] = np.clip(
    (1.0 - PATIENT_BLEND_LAMBDA) * p_prior_row + PATIENT_BLEND_LAMBDA * p_any_row,
    CLIP_VALUE,
    1.0 - CLIP_VALUE,
)

pred_types = df_sub["prediction_type"].astype(str).to_numpy()
col_idx = pd.Index(target_cols).get_indexer(pred_types)
if (col_idx < 0).any():
    bad = np.unique(pred_types[col_idx < 0]).tolist()
    raise ValueError(f"Unexpected prediction_type values not in target_cols: {bad}")

prob_matrix = df_sub[target_cols].to_numpy(dtype=np.float64)
df_sub["fractured"] = prob_matrix[np.arange(len(df_sub)), col_idx]
df_sub["fractured"] = np.clip(
    df_sub["fractured"].to_numpy(dtype=np.float64), CLIP_VALUE, 1.0 - CLIP_VALUE
)

df_sub = df_sub.reset_index(drop=False)
df_sub[["row_id", "fractured"]].head()



## === cell 26
sub_path = "submission.csv"
df_sub[["row_id", "fractured"]].to_csv(sub_path, index=False)

df_sample = pd.read_csv(SAMPLE_SUB)
assert list(df_sample.columns) == ["row_id", "fractured"]
assert df_sub.shape[0] == df_sample.shape[0], "Row count mismatch vs sample submission"
assert set(df_sub["row_id"]) == set(
    df_sample["row_id"]
), "row_id set mismatch vs sample submission"
print(f"Wrote {sub_path} with shape {df_sub[['row_id','fractured']].shape}")
