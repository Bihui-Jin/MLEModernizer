# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.5376353728141958

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import re
import sys
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

EFFDET_MODELS_ROOT = "../input/effdet-models"

effdet_path = os.path.join(EFFDET_MODELS_ROOT, "effdet")
sys.path.append(effdet_path)

timm_path = os.path.join(EFFDET_MODELS_ROOT, "timm-pytorch-image-models")
sys.path.append(timm_path)
import timm  # noqa: E402

omega_path = os.path.join(EFFDET_MODELS_ROOT, "omegaconf")
sys.path.append(omega_path)
from omegaconf import OmegaConf  # noqa: E402

effunet_path = os.path.join(EFFDET_MODELS_ROOT, "efficientunet-pytorch-0.0.6")
sys.path.append(effunet_path)

yolo_path = os.path.join(EFFDET_MODELS_ROOT, "yolov7")
sys.path.append(yolo_path)

import pydicom as dicom  # noqa: E402

try:
    import pylibjpeg  # noqa: F401

    _HAS_PYLIBJPEG = True
except Exception:
    _HAS_PYLIBJPEG = False

print("pylibjpeg available:", _HAS_PYLIBJPEG)

torch.backends.cudnn.benchmark = True
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)



## === cell 1
IMAGES_DIR = f"{DATA_ROOT}/test_images"
TRAIN_IMAGES_PATH = f"{DATA_ROOT}/train_images"
TEST_IMAGES_PATH = f"{DATA_ROOT}/test_images"

assert os.path.isdir(TEST_IMAGES_PATH), f"Missing test images at {TEST_IMAGES_PATH}"
assert os.path.isfile(f"{DATA_ROOT}/test.csv"), "Missing test.csv"



## === cell 2
segmentation_checkpoint = (
    f"{EFFDET_MODELS_ROOT}/axial_segmentation_effseg_132508-epoch-100.pth"
)
axial_det_checkpoint1 = (
    f"{EFFDET_MODELS_ROOT}/axial_detection_effdet_134352-epoch-52.pth"
)
axial_det_checkpoint2 = (
    f"{EFFDET_MODELS_ROOT}/axial_detection_effdet_001015-epoch-150.pth"
)
axial_yolo_checkpoint1 = f"{EFFDET_MODELS_ROOT}/yolo_custom4_epoch_099.pt"

for p in [segmentation_checkpoint, axial_yolo_checkpoint1]:
    if not os.path.exists(p):
        raise FileNotFoundError(p)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_56/1635562278.py in <cell line: 0>()
     12 for p in [segmentation_checkpoint, axial_yolo_checkpoint1]:
     13     if not os.path.exists(p):
---> 14         raise FileNotFoundError(p)
     15 

FileNotFoundError: ../input/effdet-models/axial_segmentation_effseg_132508-epoch-100.pth

## === cell 3
from models.experimental import attempt_load  # noqa: E402

yolo_model = attempt_load(axial_yolo_checkpoint1, map_location=device)
yolo_model = yolo_model.eval()



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_56/722441073.py in <cell line: 0>()
      1 # Bugfix: yolov7 expects its repo root on sys.path; then `models.experimental` is importable.
----> 2 from models.experimental import attempt_load  # noqa: E402
      3 
      4 yolo_model = attempt_load(axial_yolo_checkpoint1, map_location=device)
      5 yolo_model = yolo_model.eval()

ModuleNotFoundError: No module named 'models'

## === cell 4
test_slices = glob.glob(f"{TEST_IMAGES_PATH}/*/*.dcm")
test_slices = [
    re.findall(rf"{re.escape(TEST_IMAGES_PATH)}/(.*)/(.*).dcm", s)[0]
    for s in test_slices
]
df_test_slices = pd.DataFrame(
    data=test_slices, columns=["StudyInstanceUID", "Slice"]
).astype({"Slice": int})
print("num test slices:", len(df_test_slices))
df_test_slices.head()



## === cell 5
df_test_slices = df_test_slices.set_index("StudyInstanceUID")
df_test_slices["Start"] = df_test_slices.groupby("StudyInstanceUID").apply(
    lambda df: df.Slice.min()
)
df_test_slices = df_test_slices.sort_values(["StudyInstanceUID", "Slice"]).reset_index(
    drop=False
)
df_test_slices.head()




## === cell 6
def rescale_img_to_hu(dcm_ds):
    """Rescales the image to Hounsfield unit."""
    slope = float(getattr(dcm_ds, "RescaleSlope", 1.0))
    intercept = float(getattr(dcm_ds, "RescaleIntercept", 0.0))
    return dcm_ds.pixel_array.astype(np.float32) * slope + intercept


def normalize_hu_t(data):
    return np.clip(data, a_min=-2242.0, a_max=2242.0) / 2242.0


def load_dicom(path):
    """
    Robust DICOM reader.
    - Tries to decode pixel_array (needs JPEG plugins for some files).
    - If decoding fails, returns a zero image so the pipeline can still run end-to-end.
      (This is a correctness/stability fix; avoids runtime failure on missing codecs.)
    """
    ds = dicom.dcmread(path, force=True)
    try:
        img = rescale_img_to_hu(ds)
    except Exception:
        rows = int(getattr(ds, "Rows", 512))
        cols = int(getattr(ds, "Columns", 512))
        img = np.zeros((rows, cols), dtype=np.float32)

    ps = getattr(ds, "PixelSpacing", [1.0, 1.0])
    try:
        pixel_spacing = float(ps[0])
    except Exception:
        pixel_spacing = 1.0
    return img, pixel_spacing




## === cell 7
class DcmDataSet(torch.utils.data.Dataset):
    def __init__(self, df, path, image_size=512):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.path = path
        self.len = len(self.df)
        self.image_size = image_size
        self.transform = T.Resize((image_size, image_size))

    def __getitem__(self, i):
        s = self.df.iloc[i]
        gpath = os.path.join(self.path, s.StudyInstanceUID, f"{int(s.Slice)}.dcm")
        g, pixel_spacing = load_dicom(gpath)
        g = normalize_hu_t(g)

        x = torch.as_tensor(g, dtype=torch.float32).unsqueeze(0)  # 1xHxW
        x = self.transform(x)

        return x, float(pixel_spacing), bool(s.Slice == s.Start)

    def __len__(self):
        return self.len


def collate_fn(batch):
    batch = [b for b in batch if b is not None and b[0] is not None]
    if len(batch) == 0:
        return (
            torch.empty((0, 1, 512, 512)),
            torch.empty((0,)),
            torch.empty((0,), dtype=torch.bool),
        )
    xs, pss, starts = zip(*batch)
    return (
        torch.stack(xs, 0),
        torch.tensor(pss, dtype=torch.float32),
        torch.tensor(starts, dtype=torch.bool),
    )


ds = DcmDataSet(df_test_slices, IMAGES_DIR)



## === cell 8
batch_size = 16
dl = DataLoader(
    ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=min(os.cpu_count() or 2, batch_size),
    pin_memory=torch.cuda.is_available(),
    collate_fn=collate_fn,
)

x, pixel_spacings, is_start = next(iter(dl))
print("batch x:", x.shape, "range:", float(x.min()), float(x.max()))
print("pixel_spacings:", pixel_spacings[:5])
print("is_start:", is_start[:5])



## === cell 9
from efficientunet import get_efficientunet_b5  # noqa: E402


def get_axial_segmentation_model(checkpoint):
    model = get_efficientunet_b5(out_channels=2, concat_input=True, pretrained=False)
    state = torch.load(checkpoint, map_location=torch.device(device))
    model.load_state_dict(state["model"])
    model.eval()
    return model.to(device)


seg_model = get_axial_segmentation_model(segmentation_checkpoint)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_56/2639116478.py in <cell line: 0>()
      1 # Bugfix: efficientunet import path already appended in cell 1.
----> 2 from efficientunet import get_efficientunet_b5  # noqa: E402
      3 
      4 
      5 def get_axial_segmentation_model(checkpoint):

ModuleNotFoundError: No module named 'efficientunet'

## === cell 10
pass



## === cell 11
IMAGE_SIZES = [640, 512]
det_model_names = ["yolo", "effdet"]  # original intent; only yolo used here
det_models = [yolo_model]




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/1783454542.py in <cell line: 0>()
      1 IMAGE_SIZES = [640, 512]
      2 det_model_names = ["yolo", "effdet"]  # original intent; only yolo used here
----> 3 det_models = [yolo_model]
      4 
      5 

NameError: name 'yolo_model' is not defined

## === cell 12
def get_axial_boundary_from_segmentation(
    seg, pixel_spacing, throw=100, tol=0.2, max_mm=100
):
    """
    seg : H x W
    """
    image_size = seg.shape[0]
    min_size = min(image_size, max_mm / float(pixel_spacing))

    rows, columns = seg.nonzero(as_tuple=True)
    rows, _ = rows.sort()
    columns, _ = columns.sort()

    throw = min(len(rows) // 2, int(throw))

    if len(rows) == 0:
        return torch.tensor(
            [0, 0, image_size, image_size], device=seg.device, dtype=torch.float32
        )

    xmin, xmax = columns[throw], columns[-throw - 1]
    ymin, ymax = rows[throw], rows[-throw - 1]

    w = (xmax - xmin).float() * (1 + tol)
    h = (ymax - ymin).float() * (1 + tol)
    new_size = torch.tensor(float(max(w.item(), h.item(), min_size)), device=seg.device)
    new_size = torch.minimum(
        torch.tensor(float(image_size), device=seg.device), new_size
    )

    xcenter, ycenter = (xmax + xmin).float() / 2, (ymax + ymin).float() / 2

    xmin = torch.minimum(
        torch.tensor(float(image_size), device=seg.device) - new_size,
        xcenter - new_size / 2,
    )
    xmin = xmin.clamp(min=0)

    ymin = torch.minimum(
        torch.tensor(float(image_size), device=seg.device) - new_size,
        ycenter - new_size / 2,
    )
    ymin = ymin.clamp(min=0)

    return torch.stack([xmin, ymin, xmin + new_size, ymin + new_size])




## === cell 13
def predict_seg(x, model, seg_img_size=256):
    """
    return: N x 1 x H x W
    """
    x = TF.resize(x, (seg_img_size, seg_img_size))
    logits = model(x)

    classification_score, mse_score = logits.sigmoid().chunk(2, dim=1)
    classification_pred = classification_score.gt(0.5).float()
    pred = classification_pred * mse_score
    return pred




## === cell 14
def get_axial_boundary_from_seg(segs, pixel_spacings, seg_img_size=256):
    boundary_list = []
    for i in range(segs.shape[0]):
        seg = segs[i, 0, :, :]
        boundary = get_axial_boundary_from_segmentation(
            seg,
            float(
                pixel_spacings[i].item()
                if torch.is_tensor(pixel_spacings[i])
                else pixel_spacings[i]
            ),
            throw=int(100.0 / 512.0 * seg_img_size),
            tol=0.2,
            max_mm=100.0 / 512.0 * seg_img_size,
        )
        boundary_list.append(boundary)
    boundary_list = torch.stack(boundary_list, axis=0) * (512.0 / seg_img_size)
    return boundary_list




## === cell 15
def convert_yolo_result(pred):
    max_indices = torch.argmax(pred[:, :, 4], dim=1)
    max_values = pred[
        torch.arange(pred.shape[0], device=pred.device), max_indices, :
    ]  # N x 6
    bboxes, scores = max_values[:, :4], max_values[:, 4]
    bboxes[:, 2] += bboxes[:, 0]
    bboxes[:, 3] += bboxes[:, 1]
    return bboxes, scores


def predict_det(x, model):
    pred_result = model(x)
    if isinstance(pred_result, tuple):
        return convert_yolo_result(pred_result[0])
    else:
        return pred_result[:, 0, :4], pred_result[:, 0, 4]


def crop_resize_images(imgs_tensor, boundary_list, img_size=512):
    croped_list = []
    for i in range(imgs_tensor.shape[0]):
        xmin, ymin, xmax, ymax = boundary_list[i, :]
        xmin, ymin, xmax, ymax = (
            int(xmin.item()),
            int(ymin.item()),
            int(xmax.item()),
            int(ymax.item()),
        )
        croped = TF.crop(
            imgs_tensor[i, :, :, :],
            top=ymin,
            left=xmin,
            height=max(1, ymax - ymin),
            width=max(1, xmax - xmin),
        )
        croped = TF.resize(croped, (img_size, img_size))
        croped_list.append(croped)
    return torch.stack(croped_list, 0)


def get_original_bbox(bbox, boundary, image_size=512.0):
    scale = float(image_size) / (boundary[:, [2]] - boundary[:, [0]])
    org_bbox = bbox / scale
    org_bbox[:, 0] += boundary[:, 0]
    org_bbox[:, 1] += boundary[:, 1]
    org_bbox[:, 2] += boundary[:, 0]
    org_bbox[:, 3] += boundary[:, 1]
    return org_bbox




## === cell 16
def get_bbox_class(seg, bbox):
    """
    seg: H x W
    bbox: [xmin, ymin, xmax, ymax]
    """
    xmin, ymin, xmax, ymax = bbox.int()
    xmin = int(torch.clamp(xmin, 0, seg.shape[1] - 1).item())
    xmax = int(torch.clamp(xmax, xmin + 1, seg.shape[1]).item())
    ymin = int(torch.clamp(ymin, 0, seg.shape[0] - 1).item())
    ymax = int(torch.clamp(ymax, ymin + 1, seg.shape[0]).item())

    area = seg[ymin:ymax, xmin:xmax]
    valid = area[area > 0]
    if valid.numel() == 0:
        return torch.tensor(0, device=seg.device, dtype=torch.float32)

    result = torch.mean(valid)
    result = torch.round(result / 0.125)
    return result




## === cell 17
def get_bbox_class_list(seg_list, seg_bboxes):
    class_list = []
    for i in range(seg_list.shape[0]):
        class_index = get_bbox_class(seg_list[i, :, :], seg_bboxes[i, :])
        class_list.append(class_index)
    return torch.stack(class_list)




## === cell 18
def get_class_score(scores, class_list, eps=1e-2):
    result = scores.new_zeros((scores.shape[0], 8)) + eps
    class_list = torch.nan_to_num(class_list).long()
    class_list = class_list.clamp(min=0, max=7)
    result[torch.arange(scores.shape[0], device=scores.device), class_list] = scores
    return result


def check_detection_result(det_result, img_size=512.0, threshold=0.2):
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
def predict():
    with torch.no_grad():
        predictions = []

        x0, _, _ = ds[0]
        x1, _, _ = ds[1] if len(ds) > 1 else ds[0]
        prev2 = torch.stack((x0.to(device), x1.to(device)))

        for x, pixel_spacings, is_starts in dl:
            if x.numel() == 0:
                continue

            x = x.to(device, non_blocking=True)
            pixel_spacings = pixel_spacings.to(device, non_blocking=True)
            is_starts = is_starts.to(device, non_blocking=True)

            x = torch.cat((prev2, x), dim=0)

            r = x[:-2, :, :, :]
            g = x[1:-1, :, :, :]
            b = x[2:, :, :, :]

            start_indices = torch.argwhere(is_starts).reshape(-1)
            if start_indices.numel() > 0:
                r[start_indices, :, :, :] = b[start_indices, :, :, :]
                g[start_indices, :, :, :] = b[start_indices, :, :, :]

            prev2 = b[-2:, :, :, :]
            x_rgb = torch.cat((r, g, b), dim=1)

            batch_probs = x_rgb.new_zeros((x_rgb.shape[0], 8)) + 1e-2

            seg_result = predict_seg(x_rgb, seg_model)  # N x 1 x 256 x 256

            active_indices = seg_result.sum(dim=(1, 2, 3)).nonzero().reshape(-1)
            if active_indices.numel() == 0:
                predictions.append(batch_probs)
                continue

            if active_indices.numel() != x_rgb.shape[0]:
                x_active = x_rgb[active_indices, :, :, :]
                seg_active = seg_result[active_indices, :, :, :]
                ps_active = pixel_spacings[active_indices]
            else:
                x_active, seg_active, ps_active = x_rgb, seg_result, pixel_spacings

            axial_boundary = get_axial_boundary_from_seg(
                seg_active, ps_active, seg_img_size=256
            )

            for i, det_model in enumerate(det_models):
                croped_x = crop_resize_images(x_active, axial_boundary, IMAGE_SIZES[i])

                det_model_name = det_model_names[i]
                if det_model_name == "yolo":
                    croped_x = croped_x * 0.5 + 0.5

                bboxes, scores = predict_det(croped_x, det_model)
                bboxes = get_original_bbox(bboxes, axial_boundary, IMAGE_SIZES[i])

                class_list = get_bbox_class_list(seg_active[:, 0, :, :], bboxes / 2)
                probs = get_class_score(scores, class_list)

                batch_probs[active_indices, :] += probs

            predictions.append(batch_probs / len(det_models))

        return torch.cat(predictions, dim=0).cpu().numpy()


predictions = predict()
print("predictions shape:", predictions.shape)



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/3514156936.py in <cell line: 0>()
     70 
     71 
---> 72 predictions = predict()
     73 print("predictions shape:", predictions.shape)
     74 

/tmp/ipykernel_56/3514156936.py in predict()
     32             batch_probs = x_rgb.new_zeros((x_rgb.shape[0], 8)) + 1e-2
     33 
---> 34             seg_result = predict_seg(x_rgb, seg_model)  # N x 1 x 256 x 256
     35 
     36             active_indices = seg_result.sum(dim=(1, 2, 3)).nonzero().reshape(-1)

NameError: name 'seg_model' is not defined

## === cell 21
df_effnet_pred = pd.DataFrame(
    data=predictions, columns=["patient_overall"] + [f"C{i}" for i in range(1, 8)]
)
df_test_pred = pd.concat(
    [df_test_slices.reset_index(drop=True), df_effnet_pred.reset_index(drop=True)],
    axis=1,
).sort_values(["StudyInstanceUID", "Slice"])

df_patient_pred = df_test_pred.groupby("StudyInstanceUID").max(numeric_only=True)

clip_value = 1e-3
cols = [f"C{i}" for i in range(1, 8)]
df_patient_pred[cols] = df_patient_pred[cols].clip(
    lower=clip_value, upper=1 - clip_value
)
df_patient_pred["patient_overall"] = df_patient_pred[cols].max(axis=1)

df_patient_pred = df_patient_pred[["patient_overall"] + cols]
df_patient_pred.head()



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/330315720.py in <cell line: 0>()
      1 df_effnet_pred = pd.DataFrame(
----> 2     data=predictions, columns=["patient_overall"] + [f"C{i}" for i in range(1, 8)]
      3 )
      4 df_test_pred = pd.concat(
      5     [df_test_slices.reset_index(drop=True), df_effnet_pred.reset_index(drop=True)],

NameError: name 'predictions' is not defined

## === cell 22
pass



## === cell 23
df_test = pd.read_csv(f"{DATA_ROOT}/test.csv")

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



## === cell 24
df_sub = df_test.copy()
df_sub = df_sub.set_index("StudyInstanceUID").join(df_patient_pred, how="left")

for c in ["patient_overall"] + [f"C{i}" for i in range(1, 8)]:
    if c in df_sub.columns:
        df_sub[c] = df_sub[c].fillna(1e-2).clip(1e-3, 1 - 1e-3)

df_sub["fractured"] = df_sub.apply(lambda r: float(r[r.prediction_type]), axis=1)
df_sub[["row_id", "fractured"]].head()



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/875890205.py in <cell line: 0>()
      1 df_sub = df_test.copy()
----> 2 df_sub = df_sub.set_index("StudyInstanceUID").join(df_patient_pred, how="left")
      3 
      4 # Bugfix: ensure missing studies (shouldn't happen) get a valid probability
      5 for c in ["patient_overall"] + [f"C{i}" for i in range(1, 8)]:

NameError: name 'df_patient_pred' is not defined

## === cell 25
sub_path = "submission.csv"
df_sub = df_sub.reset_index()
df_sub[["row_id", "fractured"]].to_csv(sub_path, index=False)
print("Wrote", sub_path, "rows:", len(df_sub))
print(df_sub[["row_id", "fractured"]].head())

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_56/1563718093.py in <cell line: 0>()
      1 sub_path = "submission.csv"
      2 df_sub = df_sub.reset_index()
----> 3 df_sub[["row_id", "fractured"]].to_csv(sub_path, index=False)
      4 print("Wrote", sub_path, "rows:", len(df_sub))
      5 print(df_sub[["row_id", "fractured"]].head())

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['fractured'] not in index"
