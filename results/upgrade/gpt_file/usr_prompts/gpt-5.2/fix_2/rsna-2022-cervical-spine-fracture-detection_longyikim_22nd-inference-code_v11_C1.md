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

0.3711719783783422

# 6. Current score

0.5639

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 0.5639) has done: 'I fix the environment/import issues by removing the failing pip-install cell and instead using only packages available in the Kaggle image (with safe fallbacks when optional DICOM JPEG support packages are missing). I also fix the execution-order errors by ensuring all needed imports (glob/tqdm/nn/DataLoader, etc.) exist before use, and by making the dataset/dataloader robust (drop `None` samples via a custom `collate_fn`, use single-worker to avoid DICOM decoder issues). Because the original external model packages/checkpoints aren’t available in your provided paths, I keep the overall pipeline structure but replace the unavailable segmentation/detection inference with a stable, score-reasonable baseline: per-study probabilities derived from training-set label priors (and consistent patient_overall definition), which yields a valid submission CSV end-to-end. This is score-oriented versus a constant-0.01 guess while remaining minimal and stable under the constraints of the available environment.'

# 9. Code solution

## === cell 0
import os
import re
import sys
import math
import glob
import random
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

try:
    import pydicom as dicom
except Exception as e:
    dicom = None
    print("WARNING: pydicom not available; image loading will be disabled.", repr(e))

try:
    import pylibjpeg  # optional
except Exception:
    pylibjpeg = None



## === cell 1
IMAGES_DIR = "../input/rsna-2022-cervical-spine-fracture-detection/test_images"
TRAIN_IMAGES_PATH = "../input/rsna-2022-cervical-spine-fracture-detection/train_images"
TEST_IMAGES_PATH = "../input/rsna-2022-cervical-spine-fracture-detection/test_images"

TRAIN_CSV_PATH = "../input/rsna-2022-cervical-spine-fracture-detection/train.csv"
TEST_CSV_PATH = "../input/rsna-2022-cervical-spine-fracture-detection/test.csv"
SAMPLE_SUB_PATH = (
    "../input/rsna-2022-cervical-spine-fracture-detection/sample_submission.csv"
)

assert os.path.exists(TRAIN_CSV_PATH), f"Missing {TRAIN_CSV_PATH}"
assert os.path.exists(TEST_CSV_PATH), f"Missing {TEST_CSV_PATH}"
assert os.path.exists(SAMPLE_SUB_PATH), f"Missing {SAMPLE_SUB_PATH}"



## === cell 2
segmentation_checkpoint = (
    "../input/effdet-models/axial_segmentation_effseg_132508-epoch-100.pth"
)
axial_det_checkpoint = (
    "../input/effdet-models/axial_detection_effdet_134352-epoch-52.pth"
)



## === cell 3
df_test = pd.read_csv(TEST_CSV_PATH)
df_test.head()




## === cell 4
def rescale_img_to_hu(dcm_ds):
    return dcm_ds.pixel_array * float(getattr(dcm_ds, "RescaleSlope", 1.0)) + float(
        getattr(dcm_ds, "RescaleIntercept", 0.0)
    )


def normalize_hu(data):
    data = np.clip(data, a_min=-2242, a_max=2242) / 4484 + 0.5
    return data


def load_dicom(path):
    if dicom is None:
        raise RuntimeError("pydicom is not available in this environment.")
    ds = dicom.dcmread(path)
    img = rescale_img_to_hu(ds)
    pixel_spacing = float(ds.PixelSpacing[0]) if hasattr(ds, "PixelSpacing") else 1.0
    return img, pixel_spacing




## === cell 5
class DcmDataSet(Dataset):
    def __init__(self, df, path, transforms=None):
        super().__init__()
        self.df = df
        self.path = path
        self.transforms = transforms
        self.len = len(self.df)

    def __getitem__(self, i):
        try:
            s = self.df.iloc[i]
            prev_s = self.df.iloc[i - 1] if i > 0 else s
            prev_s = s if prev_s.StudyInstanceUID != s.StudyInstanceUID else prev_s

            next_s = self.df.iloc[i + 1] if i < (self.len - 1) else s
            next_s = s if next_s.StudyInstanceUID != s.StudyInstanceUID else next_s

            rpath = os.path.join(
                self.path, s.StudyInstanceUID, f"{int(prev_s.Slice)}.dcm"
            )
            gpath = os.path.join(self.path, s.StudyInstanceUID, f"{int(s.Slice)}.dcm")
            bpath = os.path.join(
                self.path, s.StudyInstanceUID, f"{int(next_s.Slice)}.dcm"
            )

            g, pixel_spacing = load_dicom(gpath)
            r, _ = load_dicom(rpath)
            b, _ = load_dicom(bpath)

            img = np.stack((r, g, b))
            img = normalize_hu(img)
            if self.transforms is not None:
                img = self.transforms(img)
        except Exception:
            return None, None

        return img, pixel_spacing

    def __len__(self):
        return self.len


class DataTransform(nn.Module):
    def __init__(self, image_size=512):
        super().__init__()
        self.image_size = image_size

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        return x




## === cell 6
def safe_collate(batch):
    batch = [b for b in batch if b[0] is not None]
    if len(batch) == 0:
        return torch.empty((0, 3, 512, 512)), torch.empty((0,))
    xs, ps = zip(*batch)
    return torch.stack(xs, 0), torch.as_tensor(ps, dtype=torch.float32)


dl = None  # not used in baseline




## === cell 7
def get_axial_segmentation_model(checkpoint):
    raise ModuleNotFoundError(
        "efficientunet/weights not available in this environment; baseline uses priors only."
    )


seg_model = None




## === cell 8
def get_axial_detection_model(checkpoint, image_size=512):
    raise ModuleNotFoundError(
        "effdet/weights not available in this environment; baseline uses priors only."
    )


det_model = None




## === cell 9
def get_axial_boundary_from_segmentation(
    seg, pixel_spacing, throw=100, tol=0.2, max_mm=100
):
    image_size = seg.shape[0]
    min_size = min(image_size, max_mm / pixel_spacing)

    rows, columns = seg.nonzero(as_tuple=True)
    rows, _ = torch.sort(rows)
    columns, _ = torch.sort(columns)

    throw = min(len(rows) // 2, throw)

    if len(rows) == 0:
        return torch.tensor([0, 0, image_size, image_size], device=seg.device)

    xmin, xmax = columns[throw], columns[-throw]
    ymin, ymax = rows[throw], rows[-throw]

    w = (xmax - xmin) * (1 + tol)
    h = (ymax - ymin) * (1 + tol)
    new_size = max(w, h, torch.as_tensor(min_size, device=seg.device))
    new_size = min(image_size, float(new_size))

    xcenter, ycenter = (xmax + xmin) / 2, (ymax + ymin) / 2

    xmin = torch.minimum(
        torch.tensor(image_size - new_size, device=seg.device), xcenter - new_size / 2
    ).clamp(min=0)
    ymin = torch.minimum(
        torch.tensor(image_size - new_size, device=seg.device), ycenter - new_size / 2
    ).clamp(min=0)

    return torch.stack([xmin, ymin, xmin + new_size, ymin + new_size])




## === cell 10
def predict_seg(x, model, img_size=256):
    raise RuntimeError("Segmentation model not available in this baseline.")




## === cell 11
def get_axial_boundary(segs, pixel_spacings, seg_img_size=256):
    raise RuntimeError("Segmentation model not available in this baseline.")




## === cell 12
def predict_det(x, model):
    raise RuntimeError("Detection model not available in this baseline.")


def crop_resize_images(imgs_tensor, boundary_list, img_size=512):
    raise RuntimeError("Detection model not available in this baseline.")


def get_original_bbox(bbox, boundary):
    raise RuntimeError("Detection model not available in this baseline.")




## === cell 13
def get_bbox_class(seg, bbox):
    raise RuntimeError("Detection model not available in this baseline.")




## === cell 14
def get_bbox_class_list(seg_list, seg_bboxes):
    raise RuntimeError("Detection model not available in this baseline.")




## === cell 15
def get_class_score(scores, class_list, eps=1e-2):
    raise RuntimeError("Detection model not available in this baseline.")


def check_detection_result(det_result, img_size=512.0, threshold=0.2):
    raise RuntimeError("Detection model not available in this baseline.")




## === cell 16
def cal_loss(prob, label):
    pos_weight = np.array([14, 2, 2, 2, 2, 2, 2, 2])
    neg_weight = np.array([7, 1, 1, 1, 1, 1, 1, 1])
    prob = np.clip(prob, 1e-6, 1 - 1e-6)
    score = pos_weight * label * np.log(prob) + neg_weight * (1 - label) * np.log(
        1 - prob
    )
    weight_total = pos_weight * label + neg_weight * (1 - label)
    return -score.sum(axis=1) / weight_total.sum(axis=1)




## === cell 17
def predict_priors(train_csv_path: str):
    df_train = pd.read_csv(train_csv_path)
    target_cols = ["patient_overall"] + [f"C{i}" for i in range(1, 8)]
    priors = df_train[target_cols].mean().astype(float)

    priors["patient_overall"] = max(
        priors["patient_overall"], float(priors[[f"C{i}" for i in range(1, 8)]].max())
    )

    priors = priors.clip(1e-3, 1 - 1e-3)
    return priors.to_dict()


priors = predict_priors(TRAIN_CSV_PATH)
priors



## === cell 18
study_ids = df_test["StudyInstanceUID"].unique()
df_patient_pred = pd.DataFrame({"StudyInstanceUID": study_ids})
for k in ["patient_overall"] + [f"C{i}" for i in range(1, 8)]:
    df_patient_pred[k] = float(priors[k])

df_patient_pred = df_patient_pred.set_index("StudyInstanceUID")
df_patient_pred.head()



## === cell 19
pass



## === cell 20
if df_test.iloc[0].row_id == "1.2.826.0.1.3680043.10197_C1":
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

df_test.head()



## === cell 21
df_sub = df_test.copy()
df_sub = (
    df_sub.set_index("StudyInstanceUID").join(df_patient_pred, how="left").reset_index()
)

for k in ["patient_overall"] + [f"C{i}" for i in range(1, 8)]:
    df_sub[k] = df_sub[k].fillna(float(priors[k]))

df_sub["fractured"] = df_sub.apply(lambda r: float(r[r["prediction_type"]]), axis=1)
df_sub[["row_id", "fractured"]].head()



## === cell 22
sub_path = "submission.csv"
df_sub[["row_id", "fractured"]].to_csv(sub_path, index=False)

df_sample = pd.read_csv(SAMPLE_SUB_PATH)
assert list(df_sample.columns) == ["row_id", "fractured"]
assert df_sub.shape[0] == df_sample.shape[0], (df_sub.shape, df_sample.shape)
print("Wrote:", sub_path, "rows:", df_sub.shape[0])
print(df_sub.head())
