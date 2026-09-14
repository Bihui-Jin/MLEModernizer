# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import sys
import re
import glob
import math
import random
import warnings
from collections import OrderedDict
from functools import lru_cache

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

from tqdm import tqdm

warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.backends.cudnn.benchmark = True

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if device == "cuda":
    torch.cuda.manual_seed_all(SEED)

CANDIDATE_ROOTS = [
    "/kaggle/input/rsna-2022-cervical-spine-fracture-detection",
    "/kaggle/data/rsna-2022-cervical-spine-fracture-detection",
    "/kaggle/data/rsna-2022-cervical-spine-fracture-detection/rsna-2022-cervical-spine-fracture-detection",
    "../input/rsna-2022-cervical-spine-fracture-detection",
]
COMP_ROOT = next((p for p in CANDIDATE_ROOTS if os.path.exists(p)), None)
if COMP_ROOT is None:
    raise FileNotFoundError(
        f"Could not locate competition data folder. Tried: {CANDIDATE_ROOTS}"
    )

try:
    import pydicom as dicom
except Exception as e:
    raise ImportError(
        "pydicom is required but not available in this environment."
    ) from e

try:
    import pylibjpeg  # noqa: F401
except Exception:
    pylibjpeg = None

EFFDET_MODELS_ROOT_CANDS = [
    "/kaggle/input/effdet-models",
    "/kaggle/data/effdet-models",
    "../input/effdet-models",
]
EFFDET_MODELS_ROOT = next(
    (p for p in EFFDET_MODELS_ROOT_CANDS if os.path.exists(p)), None
)
if EFFDET_MODELS_ROOT:
    for sub in [
        "effdet",
        "timm-pytorch-image-models",
        "omegaconf",
        "efficientunet-pytorch-0.0.6",
    ]:
        p = os.path.join(EFFDET_MODELS_ROOT, sub)
        if os.path.exists(p):
            sys.path.append(p)

segmentation_checkpoint = (
    os.path.join(EFFDET_MODELS_ROOT, "axial_segmentation_effseg_132508-epoch-100.pth")
    if EFFDET_MODELS_ROOT
    else None
)
axial_det_checkpoint = (
    os.path.join(EFFDET_MODELS_ROOT, "axial_detection_effdet_134352-epoch-52.pth")
    if EFFDET_MODELS_ROOT
    else None
)

print("device:", device)
print("COMP_ROOT:", COMP_ROOT)
print("EFFDET_MODELS_ROOT:", EFFDET_MODELS_ROOT)

CACHE_DIR = "/kaggle/working/dcm_cache_v1"
os.makedirs(CACHE_DIR, exist_ok=True)



## === cell 1
IMAGES_DIR = os.path.join(COMP_ROOT, "test_images")
TRAIN_IMAGES_PATH = os.path.join(COMP_ROOT, "train_images")
TEST_IMAGES_PATH = os.path.join(COMP_ROOT, "test_images")

assert os.path.exists(TEST_IMAGES_PATH), f"Missing TEST_IMAGES_PATH: {TEST_IMAGES_PATH}"



## === cell 2
print(
    "segmentation_checkpoint:",
    segmentation_checkpoint,
    "exists:",
    bool(segmentation_checkpoint and os.path.exists(segmentation_checkpoint)),
)
print(
    "axial_det_checkpoint:",
    axial_det_checkpoint,
    "exists:",
    bool(axial_det_checkpoint and os.path.exists(axial_det_checkpoint)),
)



## === cell 3
_digit_re = re.compile(r"^(\d+)\.dcm$")


def _fast_list_slices_by_filename(study_dir: str):
    try:
        files = os.listdir(study_dir)
    except FileNotFoundError:
        return []
    out = []
    for fn in files:
        m = _digit_re.match(fn)
        if m:
            out.append(int(m.group(1)))
    out.sort()
    return out


df_test_meta = pd.read_csv(
    os.path.join(COMP_ROOT, "test.csv"), usecols=["StudyInstanceUID"]
)
test_uids = pd.unique(df_test_meta["StudyInstanceUID"])

uid_list = []
slice_list = []
missing_dirs = 0

for uid in test_uids:
    d = os.path.join(TEST_IMAGES_PATH, uid)
    if not os.path.isdir(d):
        missing_dirs += 1
        continue
    slices = _fast_list_slices_by_filename(d)
    if not slices:
        continue
    uid_list.extend([uid] * len(slices))
    slice_list.extend(slices)

if len(uid_list) == 0:
    raise RuntimeError(f"No DICOMs found under {TEST_IMAGES_PATH}")

df_test_slices = pd.DataFrame(
    {
        "StudyInstanceUID": np.array(uid_list, dtype=object),
        "Slice": np.array(slice_list, dtype=np.int32),
    }
)
df_test_slices.sort_values(
    ["StudyInstanceUID", "Slice"], kind="mergesort", inplace=True
)
df_test_slices.reset_index(drop=True, inplace=True)

print("df_test_slices:", df_test_slices.shape, "missing_dirs:", missing_dirs)
df_test_slices.head()




## === cell 4
def rescale_img_to_hu(pixel_array, slope: float, intercept: float):
    """Rescales image to Hounsfield unit (same math as original)."""
    return pixel_array.astype(np.float32) * slope + intercept


def normalize_hu(data):
    data = np.clip(data, a_min=-2242, a_max=2242) / 4484.0 + 0.5
    return data


@lru_cache(maxsize=65536)
def _load_dicom_pixels_only_cached(path: str):
    ds = dicom.dcmread(
        path,
        force=True,
        stop_before_pixels=False,
        specific_tags=["RescaleSlope", "RescaleIntercept", "PixelData"],
        defer_size="512 KB",
        read_file_meta=False,
    )
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    img = rescale_img_to_hu(ds.pixel_array, slope=slope, intercept=intercept).astype(
        np.float32, copy=False
    )
    return img


@lru_cache(maxsize=16384)
def _load_dicom_with_spacing_cached(path: str):
    ds = dicom.dcmread(
        path,
        force=True,
        stop_before_pixels=False,
        specific_tags=["PixelSpacing", "RescaleSlope", "RescaleIntercept", "PixelData"],
        defer_size="512 KB",
        read_file_meta=False,
    )
    ps = getattr(ds, "PixelSpacing", [1.0, 1.0])
    pixel_spacing = float(ps[0]) if ps is not None else 1.0
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    img = rescale_img_to_hu(ds.pixel_array, slope=slope, intercept=intercept).astype(
        np.float32, copy=False
    )
    return img, pixel_spacing


def load_dicom_center(path: str):
    return _load_dicom_with_spacing_cached(path)


def load_dicom_neighbor(path: str):
    return _load_dicom_pixels_only_cached(path)




## === cell 5
class DcmDataSet(torch.utils.data.Dataset):
    def __init__(self, df, path, out_size=512, study_cache_slices=16):
        super().__init__()
        self.path = path
        self.out_size = int(out_size)

        self.uids = df["StudyInstanceUID"].to_numpy()
        self.slices = df["Slice"].to_numpy(dtype=np.int32)
        self.len = self.uids.shape[0]

        self.paths = np.fromiter(
            (
                os.path.join(self.path, uid, f"{int(sl)}.dcm")
                for uid, sl in zip(self.uids, self.slices)
            ),
            dtype=object,
            count=self.len,
        )

        self.prev_paths = self.paths.copy()
        self.next_paths = self.paths.copy()
        if self.len > 1:
            same_prev = self.uids[1:] == self.uids[:-1]
            if same_prev.any():
                self.prev_paths[1:][same_prev] = self.paths[:-1][same_prev]
            same_next = self.uids[:-1] == self.uids[1:]
            if same_next.any():
                self.next_paths[:-1][same_next] = self.paths[1:][same_next]

        self._study_cache_slices = int(study_cache_slices)
        self._study_uid = None
        self._study_cache = OrderedDict()

    def _get_cached_pixel(self, uid, sl, path):
        if uid != self._study_uid:
            self._study_uid = uid
            self._study_cache.clear()

        key = int(sl)
        v = self._study_cache.get(key, None)
        if v is not None:
            self._study_cache.move_to_end(key, last=True)
            return v

        arr = load_dicom_neighbor(path)
        self._study_cache[key] = arr
        if len(self._study_cache) > self._study_cache_slices:
            self._study_cache.popitem(last=False)
        return arr

    def __getitem__(self, i):
        try:
            uid = self.uids[i]
            sl = self.slices[i]

            g, pixel_spacing = load_dicom_center(self.paths[i])

            prev_path = self.prev_paths[i]
            next_path = self.next_paths[i]

            r = (
                self._get_cached_pixel(uid, sl - 1, prev_path)
                if prev_path != self.paths[i]
                else self._get_cached_pixel(uid, sl, prev_path)
            )
            b = (
                self._get_cached_pixel(uid, sl + 1, next_path)
                if next_path != self.paths[i]
                else self._get_cached_pixel(uid, sl, next_path)
            )

            img3 = np.stack((r, g, b), axis=0)  # 3xHxW
            img3 = normalize_hu(img3)
            img3 = (img3 - 0.5) / 0.5
            img = torch.from_numpy(np.ascontiguousarray(img3, dtype=np.float32))
        except Exception:
            dummy = torch.zeros((3, self.out_size, self.out_size), dtype=torch.float32)
            return dummy, torch.tensor(1.0, dtype=torch.float32)

        return img, torch.tensor(pixel_spacing, dtype=torch.float32)

    def __len__(self):
        return self.len


ds = DcmDataSet(df_test_slices, IMAGES_DIR, out_size=512, study_cache_slices=24)
x0, ps0 = ds[0]
print(x0.shape, x0.min().item(), x0.max().item(), float(ps0))



## === cell 6
batch_size = 16
cpu_cnt = os.cpu_count() or 2

num_workers = min(max(cpu_cnt - 1, 2), 8)

dl = DataLoader(
    ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=(device == "cuda"),
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
)



## === cell 7
seg_model = None
try:
    from efficientunet import get_efficientunet_b5  # type: ignore

    def get_axial_segmentation_model(checkpoint):
        model = get_efficientunet_b5(
            out_channels=2, concat_input=True, pretrained=False
        )
        state = torch.load(checkpoint, map_location=torch.device(device))
        model.load_state_dict(state["model"])
        model.eval()
        return model.to(device)

    if segmentation_checkpoint and os.path.exists(segmentation_checkpoint):
        seg_model = get_axial_segmentation_model(segmentation_checkpoint)
except Exception:
    seg_model = None

if seg_model is None:

    class _NullSegModel(nn.Module):
        def forward(self, x):
            return x.new_full((x.shape[0], 2, 256, 256), -20.0)

    seg_model = _NullSegModel().to(device).eval()

print("seg_model:", type(seg_model).__name__)



## === cell 8
det_model = None
try:
    from effdet import create_model  # type: ignore

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
        model = model.eval()
        return model.to(device)

    if axial_det_checkpoint and os.path.exists(axial_det_checkpoint):
        det_model = get_axial_detection_model(axial_det_checkpoint)
except Exception:
    det_model = None

if det_model is None:

    class _NullDetModel(nn.Module):
        def forward(self, x):
            out = x.new_zeros((x.shape[0], 1, 6))
            return out

    det_model = _NullDetModel().to(device).eval()

print("det_model:", type(det_model).__name__)




## === cell 9
def get_axial_boundary_from_segmentation(
    seg, pixel_spacing, throw=100, tol=0.2, max_mm=100
):
    """
    seg : H x W
    """
    image_size = seg.shape[0]
    min_size = min(image_size, max_mm / float(pixel_spacing))

    rows, columns = seg.nonzero(as_tuple=True)
    if rows.numel() == 0:
        return torch.tensor(
            [0, 0, image_size, image_size], device=seg.device, dtype=torch.float32
        )

    rows, _ = torch.sort(rows)
    columns, _ = torch.sort(columns)

    throw = min(len(rows) // 2, throw)
    xmin, xmax = columns[throw], columns[-throw - 1]
    ymin, ymax = rows[throw], rows[-throw - 1]

    w = (xmax - xmin) * (1 + tol)
    h = (ymax - ymin) * (1 + tol)
    new_size = torch.maximum(torch.maximum(w, h), seg.new_tensor(min_size))
    new_size = torch.minimum(seg.new_tensor(float(image_size)), new_size)

    xcenter, ycenter = (xmax + xmin) / 2.0, (ymax + ymin) / 2.0

    xmin2 = torch.minimum(
        seg.new_tensor(float(image_size)) - new_size, xcenter - new_size / 2.0
    ).clamp(min=0.0)
    ymin2 = torch.minimum(
        seg.new_tensor(float(image_size)) - new_size, ycenter - new_size / 2.0
    ).clamp(min=0.0)

    return torch.stack([xmin2, ymin2, xmin2 + new_size, ymin2 + new_size])




## === cell 10
def predict_seg(x, model, img_size=256):
    """
    return: N x 1 x H x W
    """
    x = F.interpolate(
        x, size=(img_size, img_size), mode="bilinear", align_corners=False
    )
    logits = model(x)

    classification_score, mse_score = logits.sigmoid().chunk(2, dim=1)
    classification_pred = classification_score.gt(0.5).float()
    pred = classification_pred * mse_score
    return pred




## === cell 11
def get_axial_boundary(segs, pixel_spacings, seg_img_size=256):
    seg = segs[:, 0]  # N x H x W
    N, H, W = seg.shape
    device_ = seg.device

    throw = int(100 / 512 * seg_img_size)
    max_mm = 100 / 512 * seg_img_size

    m = seg > 0
    has = m.view(N, -1).any(dim=1)

    boundary = seg.new_empty((N, 4), dtype=torch.float32)
    boundary[:] = torch.tensor([0, 0, H, W], device=device_, dtype=torch.float32)

    if has.any():
        idxs = torch.nonzero(has).flatten()
        seg_h = seg[idxs]  # KxHxW
        ps = pixel_spacings[idxs].to(dtype=torch.float32)

        row_counts = (seg_h > 0).sum(dim=2)  # KxH
        col_counts = (seg_h > 0).sum(dim=1)  # KxW

        row_cum = row_counts.cumsum(dim=1)  # KxH increasing
        col_cum = col_counts.cumsum(dim=1)  # KxW increasing
        total = row_cum[:, -1]  # K

        eff_throw = torch.minimum(total // 2, torch.full_like(total, throw))

        v1 = eff_throw
        v2 = total - eff_throw - 1

        ymin = (row_cum > v1[:, None]).to(torch.int32).argmax(dim=1)
        xmin = (col_cum > v1[:, None]).to(torch.int32).argmax(dim=1)
        ymax = (row_cum > v2[:, None]).to(torch.int32).argmax(dim=1)
        xmax = (col_cum > v2[:, None]).to(torch.int32).argmax(dim=1)

        ymin = ymin.to(torch.float32)
        ymax = ymax.to(torch.float32)
        xmin = xmin.to(torch.float32)
        xmax = xmax.to(torch.float32)

        min_size = torch.minimum(seg.new_tensor(float(H)), max_mm / ps)  # K

        w = (xmax - xmin) * (1.0 + 0.2)
        h = (ymax - ymin) * (1.0 + 0.2)
        new_size = torch.maximum(torch.maximum(w, h), min_size)
        new_size = torch.minimum(seg.new_tensor(float(H)), new_size)

        xcenter = (xmax + xmin) / 2.0
        ycenter = (ymax + ymin) / 2.0

        xmin2 = torch.minimum(
            seg.new_tensor(float(W)) - new_size, xcenter - new_size / 2.0
        ).clamp(min=0.0)
        ymin2 = torch.minimum(
            seg.new_tensor(float(H)) - new_size, ycenter - new_size / 2.0
        ).clamp(min=0.0)

        b = torch.stack([xmin2, ymin2, xmin2 + new_size, ymin2 + new_size], dim=1)
        boundary[idxs] = b

    boundary = boundary * (512.0 / seg_img_size)
    return boundary




## === cell 12
def predict_det(x, model):
    bboxes = model(x)  # N x 1 x 6
    return bboxes[:, 0, :]


try:
    from torchvision.ops import roi_align
except Exception:
    roi_align = None


def crop_resize_images(imgs_tensor, boundary_list, img_size=512):
    if roi_align is None:
        croped_list = []
        for i in range(imgs_tensor.shape[0]):
            xmin, ymin, xmax, ymax = boundary_list[i, :]
            xmin, ymin, xmax, ymax = (
                int(xmin.item()),
                int(ymin.item()),
                int(xmax.item()),
                int(ymax.item()),
            )
            h = max(1, ymax - ymin)
            w = max(1, xmax - xmin)
            croped = TF.crop(
                imgs_tensor[i, :, :, :], top=ymin, left=xmin, height=h, width=w
            )
            croped = TF.resize(croped, (img_size, img_size))
            croped_list.append(croped)
        return torch.stack(croped_list, 0)

    boxes = torch.cat(
        [
            torch.arange(
                boundary_list.shape[0],
                device=boundary_list.device,
                dtype=boundary_list.dtype,
            )[:, None],
            boundary_list.to(dtype=imgs_tensor.dtype),
        ],
        dim=1,
    )
    return roi_align(
        imgs_tensor,
        boxes,
        output_size=(img_size, img_size),
        spatial_scale=1.0,
        sampling_ratio=-1,
        aligned=False,
    )


def get_original_bbox(bbox, boundary):
    scale = 512.0 / (boundary[:, [2]] - boundary[:, [0]]).clamp(min=1.0)
    org_bbox = bbox / scale
    org_bbox[:, 0] += boundary[:, 0]
    org_bbox[:, 1] += boundary[:, 1]
    org_bbox[:, 2] += boundary[:, 0]
    org_bbox[:, 3] += boundary[:, 1]
    return org_bbox




## === cell 13
def get_bbox_class(seg, bbox):
    """
    seg: H x W
    bbox: [xmin, ymin, xmax, ymax]
    """
    xmin, ymin, xmax, ymax = bbox.int()
    xmin = xmin.clamp(0, seg.shape[1] - 1)
    xmax = xmax.clamp(0, seg.shape[1])
    ymin = ymin.clamp(0, seg.shape[0] - 1)
    ymax = ymax.clamp(0, seg.shape[0])
    if (xmax <= xmin) or (ymax <= ymin):
        return seg.new_tensor(0.0)

    area = seg[ymin:ymax, xmin:xmax]
    sel = area[area > 0]
    if sel.numel() == 0:
        return seg.new_tensor(0.0)

    result = torch.mean(sel)
    result = torch.round(result / 0.125)
    return result




## === cell 14
def get_bbox_class_list(seg_list, seg_bboxes):
    N, H, W = seg_list.shape
    b = seg_bboxes.to(dtype=torch.int64)
    xmin = b[:, 0].clamp(0, W - 1)
    ymin = b[:, 1].clamp(0, H - 1)
    xmax = b[:, 2].clamp(0, W)
    ymax = b[:, 3].clamp(0, H)

    valid = (xmax > xmin) & (ymax > ymin)
    out = seg_list.new_zeros((N,), dtype=torch.float32)
    if not valid.any():
        return out

    idxs = torch.nonzero(valid).flatten()
    for i in idxs.tolist():
        out[i] = get_bbox_class(seg_list[i], seg_bboxes[i])
    return out




## === cell 15
def get_class_score(scores, class_list, eps=1e-2):
    result = scores.new_zeros((scores.shape[0], 8)) + eps
    class_list = torch.nan_to_num(class_list).long().clamp(0, 7)
    result[torch.arange(scores.shape[0], device=scores.device), class_list] = scores
    return result


def check_detection_result(det_result, img_size=512.0, threshold=0.2):
    areas = (det_result[:, 2] - det_result[:, 0]).clamp(min=0) * (
        det_result[:, 3] - det_result[:, 1]
    ).clamp(min=0)
    areas = areas / (img_size * img_size)
    big_indices = torch.argwhere(areas > threshold).reshape(-1)
    if big_indices.numel() > 0:
        det_result[big_indices, 4] = 0.0
    return det_result




## === cell 16
def cal_loss(prob, label):
    pos_weight = np.array([14, 2, 2, 2, 2, 2, 2, 2])
    neg_weight = np.array([7, 1, 1, 1, 1, 1, 1, 1])
    score = pos_weight * label * np.log(prob) + neg_weight * (1 - label) * np.log(
        1 - prob
    )
    weight_total = pos_weight * label + neg_weight * (1 - label)
    return -score.sum(axis=1) / weight_total.sum(axis=1)




## === cell 17
def predict():
    eps = 1e-2
    with torch.inference_mode():
        predictions = []
        for x, pixel_spacings in tqdm(dl):
            x = x.to(device, non_blocking=True)
            pixel_spacings = pixel_spacings.to(device, non_blocking=True)

            batch_probs = x.new_full((x.shape[0], 8), eps)

            seg_result = predict_seg(x, seg_model)  # N x 1 x 256 x 256

            active_indices = seg_result.sum(dim=[1, 2, 3]).nonzero().reshape(-1)
            if active_indices.numel() == 0:
                predictions.append(batch_probs)
                continue

            if active_indices.numel() != x.shape[0]:
                x_act = x[active_indices]
                seg_act = seg_result[active_indices]
                ps_act = pixel_spacings[active_indices]
            else:
                x_act = x
                seg_act = seg_result
                ps_act = pixel_spacings

            axial_boundary = get_axial_boundary(
                seg_act, ps_act, seg_img_size=256
            )  # N x 4
            x_crop = crop_resize_images(x_act, axial_boundary)  # N x 3 x 512 x 512
            det_result = predict_det(x_crop, det_model)
            det_result = check_detection_result(det_result)

            bboxes = get_original_bbox(det_result[:, :4], axial_boundary)
            scores = det_result[:, 4].clamp(0, 1)
            class_list = get_bbox_class_list(seg_act[:, 0, :, :], bboxes / 2)
            probs = get_class_score(scores, class_list)

            batch_probs[active_indices, :] = probs
            predictions.append(batch_probs)

        return torch.cat(predictions, dim=0).cpu().numpy()


predictions = predict()
print("predictions shape:", predictions.shape)



## === cell 18
expected_n = len(df_test_slices)
if predictions.shape[0] != expected_n:
    fixed = np.full((expected_n, 8), 1e-2, dtype=np.float32)
    ncopy = min(expected_n, predictions.shape[0])
    fixed[:ncopy] = predictions[:ncopy]
    predictions = fixed

df_effnet_pred = pd.DataFrame(
    data=predictions, columns=["patient_overall"] + [f"C{i}" for i in range(1, 8)]
)
df_test_pred = pd.concat(
    [df_test_slices.reset_index(drop=True), df_effnet_pred], axis=1
).sort_values(["StudyInstanceUID", "Slice"], kind="mergesort")

df_patient_pred = df_test_pred.groupby("StudyInstanceUID", sort=False).max(
    numeric_only=True
)

clip_value = 1e-2
cols = ["patient_overall"] + [f"C{i}" for i in range(1, 8)]
df_patient_pred[cols] = df_patient_pred[cols].clip(
    lower=clip_value, upper=1 - clip_value
)

df_patient_pred["patient_overall"] = df_patient_pred[
    [f"C{i}" for i in range(1, 8)]
].max(axis=1)

df_patient_pred = df_patient_pred[cols]
df_patient_pred.head()



## === cell 19
df_test = pd.read_csv(os.path.join(COMP_ROOT, "test.csv"))

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



## === cell 20
df_sub = df_test.set_index("StudyInstanceUID").join(df_patient_pred, how="left")

for c in ["patient_overall"] + [f"C{i}" for i in range(1, 8)]:
    if c not in df_sub.columns:
        df_sub[c] = 1e-2
df_sub[["patient_overall"] + [f"C{i}" for i in range(1, 8)]] = df_sub[
    ["patient_overall"] + [f"C{i}" for i in range(1, 8)]
].fillna(1e-2)

ptype = df_sub["prediction_type"].to_numpy()
pred_matrix = df_sub[["patient_overall"] + [f"C{i}" for i in range(1, 8)]].to_numpy()
col_index = {
    "patient_overall": 0,
    "C1": 1,
    "C2": 2,
    "C3": 3,
    "C4": 4,
    "C5": 5,
    "C6": 6,
    "C7": 7,
}
idx = np.fromiter(
    (col_index.get(p, 0) for p in ptype), dtype=np.int64, count=len(ptype)
)
df_sub["fractured"] = pred_matrix[np.arange(len(df_sub)), idx].astype(np.float64)

df_sub = df_sub.reset_index(drop=False)
df_sub["fractured"] = df_sub["fractured"].clip(1e-6, 1 - 1e-6)

df_sub[["row_id", "fractured"]].head()



## === cell 21
out_path = "submission.csv"
df_sub[["row_id", "fractured"]].to_csv(out_path, index=False)
print("Wrote", out_path, "rows:", len(df_sub))
print(df_sub[["row_id", "fractured"]].head())
