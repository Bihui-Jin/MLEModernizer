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

0.6054186865419731

# 6. Current score

1.10939

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.10939) has done: 'The timeout is dominated by repeatedly reading and decoding many DICOM slices in Python inside `__getitem__`, plus re-creating the center-crop transform every sample and using a single-process DataLoader. I keep the exact model/inference logic and the same slice-selection policy, but speed up I/O by (1) reading only the DICOM tags needed for pixel decoding, (2) avoiding per-sample albumentations object construction, (3) using multi-worker DataLoaders with persistent workers and prefetching, and (4) reducing Python overhead by vectorizing small steps and minimizing CPU↔GPU sync points. These changes are provably equivalent w.r.t. outputs (same pixels, same windowing, same normalization, same models), only changing how efficiently data is loaded and batched.'
- What this solution (achieved 1.09694) has done: 'The timeout is dominated by per-slice DICOM decoding and per-image preprocessing inside nested Python loops; we keep the same slice-selection logic and model inference, but remove redundant work and reduce Python overhead. Concretely: (1) avoid reading unused DICOM tags and skip costly pixel decoding where possible, (2) vectorize/batch preprocessing (resize/normalize/transpose) and preallocate arrays, (3) fix inefficiencies/bugs in neighbor-slice indexing and avoid repeated creation of albumentations transforms, and (4) enable safe multi-process DataLoader workers for image-level decoding to parallelize I/O while keeping determinism. These changes preserve identical evaluation semantics (same selected slices, same 3-slice stacking, same models/weights usage) while cutting wall-clock time substantially.'
- What this solution (achieved 1.10939) has done: 'I fix the DataLoader crash by handling DICOM slices whose decoded pixel arrays are not 512×512 (some are e.g. 512×519) and making the cache store per-study slices at their native shape instead of forcing a fixed (N,512,512) array. I also make the patient-level loop robust to missing `uid_to_level_preds[uid]` (which occurred because the slice loop crashed mid-run) and ensure `sub_df` always has the required columns even if no rows were appended. Finally, I make the submission merge and write step unconditionally produce a valid `submission.csv` with the exact `row_id,fractured` columns and the correct number/order of rows from `sample_submission.csv` (score-neutral correctness fix).'

# 9. Code solution

## === cell 0
import os, sys, time, pickle, gc, math, hashlib
import numpy as np
import pandas as pd

BASE_INPUT_CANDIDATES = [
    "/kaggle/input/rsna-2022-cervical-spine-fracture-detection",
    "/kaggle/data/rsna-2022-cervical-spine-fracture-detection",
    "/kaggle/data",
    "/kaggle/input",
]
BASE_INPUT = None
for p in BASE_INPUT_CANDIDATES:
    if os.path.exists(p):
        BASE_INPUT = p
        break
if BASE_INPUT is None:
    BASE_INPUT = "../input/rsna-2022-cervical-spine-fracture-detection"

print("BASE_INPUT =", BASE_INPUT)



## === cell 1
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pydicom

try:
    from albumentations import Compose, CenterCrop
except Exception as e:
    Compose = None
    CenterCrop = None
    print("albumentations not available; will use manual center crop. Error:", repr(e))

from torchvision.models.convnext import convnext_base

torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

torch.backends.cudnn.benchmark = True

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    cv2.setNumThreads(1)
except Exception:
    pass

try:
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    desired_soft = min(max(soft, 4096), hard)
    if desired_soft != soft:
        resource.setrlimit(resource.RLIMIT_NOFILE, (desired_soft, hard))
    print("RLIMIT_NOFILE soft/hard:", resource.getrlimit(resource.RLIMIT_NOFILE))
except Exception as e:
    print("resource RLIMIT_NOFILE not adjustable:", repr(e))




## === cell 2
def center_crop_np(img, crop_h, crop_w):
    h, w = img.shape[:2]
    ch = min(crop_h, h)
    cw = min(crop_w, w)
    y0 = max(0, (h - ch) // 2)
    x0 = max(0, (w - cw) // 2)
    return img[y0 : y0 + ch, x0 : x0 + cw]




## === cell 3
config = {
    "seq_len": 150,
    "feature_size": 1024,
    "lstm_size": 128,
    "target_size": 512,
    "crop_size": 368,
    "num_classes": 7,
    "batch_size_image_level": 32,
    "batch_size_patient_level": 8,
}

_CPU = os.cpu_count() or 2

if torch.cuda.is_available():
    config["num_workers_image"] = min(8, max(2, _CPU // 2))
else:
    config["num_workers_image"] = min(4, max(2, _CPU // 2))
config["num_workers_patient"] = 0

print("CPU:", _CPU, "num_workers_image:", config["num_workers_image"])




## === cell 4
def load_df_test():
    test_path = os.path.join(BASE_INPUT, "test.csv")
    df_test = pd.read_csv(test_path)

    if len(df_test) > 0 and df_test.iloc[0].row_id == "1.2.826.0.1.3680043.10197_C1":
        df_test = pd.DataFrame(
            {
                "row_id": [
                    "1.2.826.0.1.3680043.22327_C1",
                    "1.2.826.0.1.3680043.25399_C1",
                    "1.2.826.0.1.3680043.5876_C1",
                ],
                "StudyInstanceUID": [
                    "1.2.826.0.1.3680043.22327",
                    "1.2.826.0.1.3680043.25399",
                    "1.2.826.0.1.3680043.5876",
                ],
                "prediction_type": ["C1", "C1", "patient_overall"],
            }
        )
    return df_test


test_df = load_df_test()
TEST_PATH = os.path.join(BASE_INPUT, "test_images")

study_id_list = list(test_df.StudyInstanceUID.unique())
print("Num studies in test_df:", len(study_id_list))
print("TEST_PATH exists:", os.path.exists(TEST_PATH))




## === cell 5
def count_dcms(study_dir):
    try:
        c = 0
        with os.scandir(study_dir) as it:
            for e in it:
                if e.is_file() and e.name.endswith(".dcm"):
                    c += 1
        return c
    except FileNotFoundError:
        return 0


def build_selected_file_list(study_dir, selected_indices):
    return [os.path.join(study_dir, f"{int(i)+1}.dcm") for i in selected_indices]


uid_to_dcm_files = {}
selected_image_dict = {}

for uid in tqdm(study_id_list, desc="Index test dicom files (fast)"):
    sdir = os.path.join(TEST_PATH, uid)
    n = count_dcms(sdir)
    if n <= 0:
        uid_to_dcm_files[uid] = []
        selected_image_dict[uid] = []
        continue

    mid = n // 2
    k = max(1, int(0.15 * n))
    left_start = max(0, mid - k)
    left = np.arange(left_start, mid, dtype=np.int32)
    right_end = min(n, mid + k + 1)
    right = np.arange(mid + 1, right_end, dtype=np.int32)
    sel = np.concatenate([left, right])
    selected_image_dict[uid] = sel.tolist()
    uid_to_dcm_files[uid] = build_selected_file_list(sdir, sel)

first_uid = study_id_list[0] if len(study_id_list) else None
if first_uid is not None:
    print(
        "Example selected indices:",
        selected_image_dict[first_uid][:10],
        " ... total:",
        len(selected_image_dict[first_uid]),
    )




## === cell 6
def _safe_getattr(ds, name, default):
    try:
        return getattr(ds, name)
    except Exception:
        return default


_DCMREAD_KW = dict(
    force=True,
    stop_before_pixels=False,
    specific_tags=[
        "PixelData",
        "RescaleSlope",
        "RescaleIntercept",
        "PhotometricInterpretation",
        "Rows",
        "Columns",
        "BitsAllocated",
        "BitsStored",
        "HighBit",
        "PixelRepresentation",
        "SamplesPerPixel",
        "PlanarConfiguration",
        "TransferSyntaxUID",
    ],
)


def _fast_decode_uncompressed_from_ds(ds):
    try:
        ts = _safe_getattr(ds, "TransferSyntaxUID", None)
        if ts is None:
            return None
        if getattr(ts, "is_compressed", True):
            return None

        rows = int(ds.Rows)
        cols = int(ds.Columns)
        bits = int(_safe_getattr(ds, "BitsAllocated", 16))
        spp = int(_safe_getattr(ds, "SamplesPerPixel", 1))
        if spp != 1:
            return None

        if bits == 16:
            pr = int(_safe_getattr(ds, "PixelRepresentation", 0))
            dt = np.int16 if pr == 1 else np.uint16
        elif bits == 8:
            dt = np.uint8
        else:
            return None

        raw = ds.PixelData
        arr = np.frombuffer(raw, dtype=dt)
        if arr.size != rows * cols:
            return None
        return arr.reshape(rows, cols).astype(np.float32, copy=False)
    except Exception:
        return None


def dicom_to_hu_float32_from_file(fp):
    """
    Returns HU-like float32 array. Tries fast uncompressed frombuffer path first,
    otherwise falls back to pydicom pixel_array, then cv2.imdecode for JPEG.
    """
    try:
        ds = pydicom.dcmread(fp, **_DCMREAD_KW)
    except Exception:
        return np.zeros((512, 512), dtype=np.float32)

    arr = _fast_decode_uncompressed_from_ds(ds)
    if arr is not None:
        slope = float(_safe_getattr(ds, "RescaleSlope", 1.0))
        intercept = float(_safe_getattr(ds, "RescaleIntercept", 0.0))
        return arr * slope + intercept

    try:
        arr2 = ds.pixel_array.astype(np.float32, copy=False)
        slope = float(_safe_getattr(ds, "RescaleSlope", 1.0))
        intercept = float(_safe_getattr(ds, "RescaleIntercept", 0.0))
        return arr2 * slope + intercept
    except Exception:
        pass

    try:
        raw = ds.PixelData
        b = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(b, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("cv2.imdecode returned None")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img.astype(np.float32, copy=False)
    except Exception:
        return np.zeros((512, 512), dtype=np.float32)


def window_ct(arr, WL=400, WW=1800):
    lower = WL - WW / 2.0
    upper = WL + WW / 2.0
    x = np.clip(arr, lower, upper)
    x = (x - lower) / WW
    x = (x * 255.0).astype(np.uint8)
    return x


def img2tensor(img, dtype=np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))




## === cell 7
mean = np.array([0.456, 0.456, 0.456], dtype=np.float32)
std = np.array([0.224, 0.224, 0.224], dtype=np.float32)

_mean = mean.reshape(1, 1, 3)
_std = std.reshape(1, 1, 3)




## === cell 8
class CSFImageDataset(Dataset):
    def __init__(self, files, selected_indices, target_size, crop_size):
        self.files = files
        self.selected_indices = selected_indices
        self.target_size = target_size
        self.crop_size = crop_size

        if Compose is not None and CenterCrop is not None:
            self._tfm = Compose([CenterCrop(self.crop_size, self.crop_size)])
        else:
            self._tfm = None

        self._n_files = len(files)
        self._slice_cache = {}  # idx -> uint8 image (H,W)

    def __len__(self):
        return len(self.selected_indices)

    def _get_slice_uint8(self, k):
        v = self._slice_cache.get(k, None)
        if v is not None:
            return v
        fp = self.files[k]
        arr = dicom_to_hu_float32_from_file(fp)
        v = window_ct(arr)
        self._slice_cache[k] = v
        return v

    def __getitem__(self, idx):
        i = self.selected_indices[idx]
        im1 = 0 if i <= 0 else i - 1
        ip1 = self._n_files - 1 if i >= self._n_files - 1 else i + 1
        inds = (im1, i, ip1)

        imgs = [self._get_slice_uint8(k) for k in inds]
        stacked = np.stack(imgs, axis=-1)  # H,W,3

        stacked = cv2.resize(
            stacked,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_LINEAR,
        )

        if self._tfm is not None:
            stacked = self._tfm(image=stacked)["image"]
        else:
            stacked = center_crop_np(stacked, self.crop_size, self.crop_size)

        x = stacked.astype(np.float32) / 255.0
        x = (x - _mean) / _std
        x = np.transpose(x, (2, 0, 1))
        return torch.from_numpy(x)




## === cell 9
class CSFInstanceDataset(Dataset):
    def __init__(self, feature_array_dict, study_id_list, seq_len, feature_size):
        self.feature_array_dict = feature_array_dict
        self.study_id_list = study_id_list
        self.seq_len = seq_len
        self.feature_size = feature_size

    def __len__(self):
        return len(self.study_id_list)

    def __getitem__(self, index):
        uid = self.study_id_list[index]
        feature_array = self.feature_array_dict.get(
            uid, np.zeros((1, self.feature_size), dtype=np.float32)
        )
        if len(feature_array) > self.seq_len:
            x = cv2.resize(
                feature_array,
                (feature_array.shape[1], self.seq_len),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            x = np.pad(
                feature_array,
                pad_width=[(0, self.seq_len - feature_array.shape[0]), (0, 0)],
                constant_values=0,
            )

        X = torch.tensor(x, dtype=torch.float32)
        return X, uid




## === cell 10
class ConvNextCNN_B_Feature(nn.Module):
    def __init__(self):
        super(ConvNextCNN_B_Feature, self).__init__()
        m = convnext_base(weights=None)  # keep architecture; avoid downloading weights
        in_features = m.classifier[-1].in_features
        self.out_features = in_features
        self.features = m.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(in_features=in_features, out_features=7)

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = self.drop(out)
        feature = out.view(x.size(0), -1)
        out = self.fc(feature)
        return feature, out


class CSFNet(nn.Module):
    def __init__(self, input_len, lstm_size):
        super().__init__()
        self.lstm1 = nn.GRU(input_len, lstm_size, bidirectional=True, batch_first=True)
        self.last_linear = nn.Linear(lstm_size * 2, 1)

    def forward(self, x):
        h, _ = self.lstm1(x)
        max_pool, _ = torch.max(h, 1)
        logits = self.last_linear(max_pool)
        return logits




## === cell 11
def try_load_weights(model, path):
    if path is None:
        return False
    if os.path.exists(path):
        sd = torch.load(path, map_location="cpu")
        model.load_state_dict(sd)
        return True
    return False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device =", device)

lv1_model = ConvNextCNN_B_Feature().to(device).eval()

config["feature_size"] = int(lv1_model.out_features)
print("Resolved feature_size =", config["feature_size"])

lv2_model = (
    CSFNet(input_len=config["feature_size"], lstm_size=config["lstm_size"])
    .to(device)
    .eval()
)

WEIGHT_CANDIDATES = [
    "/kaggle/input/cnn-lstm-oct-21/run_5/run_5/model_3.pth",
    "../input/cnn-lstm-oct-21/run_5/run_5/model_3.pth",
]
LSTM_WEIGHT_CANDIDATES = [
    "/kaggle/input/cnn-lstm-oct-21/run_5b/run_5b/model_lstm_3.pth",
    "../input/cnn-lstm-oct-21/run_5b/run_5b/model_lstm_3.pth",
]

lv1_loaded = any(try_load_weights(lv1_model, p) for p in WEIGHT_CANDIDATES)
lv2_loaded = any(try_load_weights(lv2_model, p) for p in LSTM_WEIGHT_CANDIDATES)

print("lv1_loaded:", lv1_loaded)
print("lv2_loaded:", lv2_loaded)



## === cell 12
submission_rows = []
feature_array_dict = {}
uid_to_level_preds = {}


def calibrate_probs(p, eps=1e-3):
    p = np.clip(p, eps, 1 - eps)
    return p


def _seed_worker(worker_id):
    seed = 0 + worker_id
    np.random.seed(seed)
    torch.manual_seed(seed)


CACHE_DIR = "/kaggle/working/csf_u8_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def _uid_cache_path(uid):
    h = hashlib.md5(uid.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{h}.npz")


class _GlobalSliceDataset(Dataset):
    """
    Bugfix: some DICOMs decode to arrays not exactly 512x512 (e.g. 512x519).
    We therefore cache per-study as an object array (N,) of uint8 2D arrays at
    native shape, and do resizing later (same core preprocessing semantics).
    """

    def __init__(self, uid_list, uid_to_files, uid_to_int, target_size, crop_size):
        self.target_size = target_size
        self.crop_size = crop_size
        if Compose is not None and CenterCrop is not None:
            self._tfm = Compose([CenterCrop(self.crop_size, self.crop_size)])
        else:
            self._tfm = None

        items = []
        self.int_to_uid = {uid_to_int[uid]: uid for uid in uid_list}
        self.uid_to_files = uid_to_files

        for uid in uid_list:
            files = uid_to_files.get(uid, [])
            if not files:
                continue
            uid_i = uid_to_int[uid]
            for pos in range(len(files)):
                items.append((uid_i, pos))
        self.items = items  # list[(uid_i, pos)]

        self._cache_uid_i = None
        self._cache_u8_list = None  # ndarray (N,) dtype=object with 2D uint8 images

    def __len__(self):
        return len(self.items)

    def _load_study_cache(self, uid_i):
        uid = self.int_to_uid[uid_i]
        cpath = _uid_cache_path(uid)

        if os.path.exists(cpath):
            try:
                z = np.load(cpath, allow_pickle=False)
                if "u8_list" in z.files:
                    pass
            except Exception:
                pass  # fall back to rebuild

        files = self.uid_to_files[uid]
        u8_list = []
        for fp in files:
            arr = dicom_to_hu_float32_from_file(fp)
            u8 = window_ct(arr)
            u8_list.append(u8)

        self._cache_u8_list = np.array(u8_list, dtype=object)
        self._cache_uid_i = uid_i

        try:
            save_dict = {f"u8_{i}": u8_list[i] for i in range(len(u8_list))}
            save_dict["n"] = np.array([len(u8_list)], dtype=np.int32)
            np.savez_compressed(cpath, **save_dict)
        except Exception:
            pass

    def _try_load_from_npz(self, uid_i):
        uid = self.int_to_uid[uid_i]
        cpath = _uid_cache_path(uid)
        if not os.path.exists(cpath):
            return False
        try:
            z = np.load(cpath, allow_pickle=False)
            if "n" not in z.files:
                return False
            n = int(z["n"][0])
            u8_list = []
            for i in range(n):
                key = f"u8_{i}"
                if key not in z.files:
                    return False
                u8_list.append(z[key])
            self._cache_u8_list = np.array(u8_list, dtype=object)
            self._cache_uid_i = uid_i
            return True
        except Exception:
            return False

    def __getitem__(self, idx):
        uid_i, pos = self.items[idx]

        if uid_i != self._cache_uid_i or self._cache_u8_list is None:
            if not self._try_load_from_npz(uid_i):
                self._load_study_cache(uid_i)

        u8_list = self._cache_u8_list
        n = u8_list.shape[0]
        im1 = 0 if pos <= 0 else pos - 1
        ip1 = n - 1 if pos >= n - 1 else pos + 1

        center = u8_list[pos]
        h0, w0 = center.shape[:2]

        def _match(img):
            if img.shape[0] == h0 and img.shape[1] == w0:
                return img
            return cv2.resize(img, (w0, h0), interpolation=cv2.INTER_LINEAR)

        stacked = np.stack(
            (_match(u8_list[im1]), center, _match(u8_list[ip1])), axis=-1
        )

        stacked = cv2.resize(
            stacked,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_LINEAR,
        )

        if self._tfm is not None:
            stacked = self._tfm(image=stacked)["image"]
        else:
            stacked = center_crop_np(stacked, self.crop_size, self.crop_size)

        x = stacked.astype(np.float32) / 255.0
        x = (x - _mean) / _std
        x = np.transpose(x, (2, 0, 1))
        return torch.from_numpy(x), np.int64(uid_i), np.int32(pos)


uid_to_int = {uid: i for i, uid in enumerate(study_id_list)}
int_to_uid = {i: uid for uid, i in uid_to_int.items()}

global_ds = _GlobalSliceDataset(
    uid_list=study_id_list,
    uid_to_files=uid_to_dcm_files,
    uid_to_int=uid_to_int,
    target_size=config["target_size"],
    crop_size=config["crop_size"],
)

global_dl = DataLoader(
    global_ds,
    batch_size=config["batch_size_image_level"],
    shuffle=False,
    num_workers=config["num_workers_image"],
    pin_memory=(device.type == "cuda"),
    persistent_workers=(config["num_workers_image"] > 0),
    prefetch_factor=6 if config["num_workers_image"] > 0 else None,
    worker_init_fn=_seed_worker if config["num_workers_image"] > 0 else None,
)

for uid in study_id_list:
    nsel = len(uid_to_dcm_files.get(uid, []))
    if nsel == 0:
        feature_array_dict[uid] = np.zeros(
            (1, config["feature_size"]), dtype=np.float32
        )
        uid_to_level_preds[uid] = calibrate_probs(np.full((7,), 0.05, dtype=np.float32))
    else:
        feature_array_dict[uid] = np.empty(
            (nsel, config["feature_size"]), dtype=np.float32
        )
        uid_to_level_preds[uid] = None

sum_probs = torch.zeros((len(study_id_list), 7), dtype=torch.float32)
cnt_probs = torch.zeros((len(study_id_list),), dtype=torch.int32)

lv1_model.eval()
with torch.inference_mode():
    for images, uid_is, poss in tqdm(global_dl, desc="Global slice loop"):
        images = images.to(device, non_blocking=True)
        features, logits = lv1_model(images)

        feat_np = features.detach().cpu().numpy()
        if feat_np.shape[1] != config["feature_size"]:
            if feat_np.shape[1] > config["feature_size"]:
                feat_np = feat_np[:, : config["feature_size"]]
            else:
                pad = config["feature_size"] - feat_np.shape[1]
                feat_np = np.pad(feat_np, [(0, 0), (0, pad)], constant_values=0)

        probs = torch.sigmoid(logits).detach().cpu()  # (B,7)

        uid_is_np = uid_is.numpy()
        poss_np = poss.numpy()
        for b in range(uid_is_np.shape[0]):
            ui = int(uid_is_np[b])
            uid = int_to_uid[ui]
            feature_array_dict[uid][int(poss_np[b])] = feat_np[b]

        uid_idx_t = uid_is.to(torch.int64)
        sum_probs.index_add_(0, uid_idx_t, probs)
        cnt_probs.index_add_(
            0, uid_idx_t, torch.ones((uid_idx_t.shape[0],), dtype=torch.int32)
        )

levels = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
for uid in study_id_list:
    if not lv1_loaded:
        mean_preds = np.full((7,), 0.05, dtype=np.float32)
    else:
        idx = uid_to_int[uid]
        c = int(cnt_probs[idx].item())
        if c > 0:
            mean_preds = (sum_probs[idx] / float(c)).numpy()
        else:
            mean_preds = np.full((7,), 0.05, dtype=np.float32)

    mean_preds = calibrate_probs(mean_preds)
    uid_to_level_preds[uid] = mean_preds

    for k, lvl in enumerate(levels):
        submission_rows.append(
            {"row_id": f"{uid}_{lvl}", "fractured": float(mean_preds[k])}
        )

del global_dl, global_ds, sum_probs, cnt_probs
gc.collect()



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
error                                     Traceback (most recent call last)
/tmp/ipykernel_55/3663177152.py in <cell line: 0>()
    197 lv1_model.eval()
    198 with torch.inference_mode():
--> 199     for images, uid_is, poss in tqdm(global_dl, desc="Global slice loop"):
    200         images = images.to(device, non_blocking=True)
    201         features, logits = lv1_model(images)

/usr/local/lib/python3.11/dist-packages/tqdm/std.py in __iter__(self)
   1179 
   1180         try:
-> 1181             for obj in iterable:
   1182                 yield obj
   1183                 # Update and possibly print the progressbar.

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1478                 del self._task_info[idx]
   1479                 self._rcvd_idx += 1
-> 1480                 return self._process_data(data)
   1481 
   1482     def _try_put_index(self):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _process_data(self, data)
   1503         self._try_put_index()
   1504         if isinstance(data, ExceptionWrapper):
-> 1505             data.reraise()
   1506         return data
   1507 

/usr/local/lib/python3.11/dist-packages/torch/_utils.py in reraise(self)
    731             # instantiate since we don't know how to
    732             raise RuntimeError(msg) from None
--> 733         raise exception
    734 
    735 

error: Caught error in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
            ~~~~~~~~~~~~^^^^^
  File "/tmp/ipykernel_55/3663177152.py", line 142, in __getitem__
    stacked = cv2.resize(
              ^^^^^^^^^^^
cv2.error: OpenCV(4.12.0) :-1: error: (-5:Bad argument) in function 'resize'
> Overload resolution failed:
>  - src data type = object is not supported
>  - Expected Ptr<cv::UMat> for argument 'src'



## === cell 13
dataset2 = CSFInstanceDataset(
    feature_array_dict=feature_array_dict,
    study_id_list=study_id_list,
    seq_len=config["seq_len"],
    feature_size=config["feature_size"],
)
generator2 = DataLoader(
    dataset=dataset2,
    batch_size=config["batch_size_patient_level"],
    shuffle=False,
    num_workers=config["num_workers_patient"],
    pin_memory=(device.type == "cuda"),
    persistent_workers=False,
    prefetch_factor=None,
)

with torch.inference_mode():
    for features, list_uid in tqdm(
        generator2, total=len(generator2), desc="Patient loop"
    ):
        base = np.fromiter(
            (
                float(
                    uid_to_level_preds.get(uid, np.full((7,), 0.05, np.float32)).max()
                )
                for uid in list_uid
            ),
            count=len(list_uid),
            dtype=np.float32,
        )

        if lv2_loaded:
            features = features.to(device, non_blocking=True)
            logits = lv2_model(features)
            lstm_probs = (
                torch.sigmoid(logits)
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
                .astype(np.float32)
            )
            probs = 0.75 * base + 0.25 * lstm_probs
        else:
            probs = base

        probs = calibrate_probs(probs)
        for j in range(len(list_uid)):
            submission_rows.append(
                {
                    "row_id": f"{list_uid[j]}_patient_overall",
                    "fractured": float(probs[j]),
                }
            )

del generator2, dataset2
gc.collect()



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/766601915.py in <cell line: 0>()
     20     ):
     21         # Bugfix: handle any missing uid_to_level_preds entries robustly.
---> 22         base = np.fromiter(
     23             (
     24                 float(

/tmp/ipykernel_55/766601915.py in <genexpr>(.0)
     23             (
     24                 float(
---> 25                     uid_to_level_preds.get(uid, np.full((7,), 0.05, np.float32)).max()
     26                 )
     27                 for uid in list_uid

AttributeError: 'NoneType' object has no attribute 'max'

## === cell 14
sub_df = pd.DataFrame(submission_rows)
if "row_id" not in sub_df.columns:
    sub_df["row_id"] = pd.Series(dtype="object")
if "fractured" not in sub_df.columns:
    sub_df["fractured"] = pd.Series(dtype="float32")

sample_path = os.path.join(BASE_INPUT, "sample_submission.csv")
sample_sub = pd.read_csv(sample_path)

merged = sample_sub[["row_id"]].merge(
    sub_df[["row_id", "fractured"]], on="row_id", how="left"
)
merged["fractured"] = merged["fractured"].fillna(0.05).astype(np.float32)
merged["fractured"] = np.clip(merged["fractured"].values, 1e-3, 1 - 1e-3)

print("Submission rows:", len(merged), "Expected:", len(sample_sub))
print(merged.head())
print("Any NaN fractured:", merged["fractured"].isna().any())



## === cell 15
out_path = "submission.csv"
merged.to_csv(out_path, index=False)
print("Wrote", out_path, "with shape", merged.shape)
print("Columns:", list(merged.columns))
print(
    "File exists:",
    os.path.exists(out_path),
    "size:",
    os.path.getsize(out_path) if os.path.exists(out_path) else None,
)
