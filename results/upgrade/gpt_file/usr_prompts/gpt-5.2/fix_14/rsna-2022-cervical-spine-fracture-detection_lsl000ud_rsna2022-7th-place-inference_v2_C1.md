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

# 5. Code solution

## === cell 0
import os
import time
import sys
import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import SimpleITK as sitk

DATA_ROOT = "../input/rsna-2022-cervical-spine-fracture-detection"
DATA_DIR_TEST = f"{DATA_ROOT}/test_images"
TRAIN_CSV = f"{DATA_ROOT}/train.csv"
TEST_CSV = f"{DATA_ROOT}/test.csv"
SAMPLE_SUB = f"{DATA_ROOT}/sample_submission.csv"

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 1))))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(max(1, (os.cpu_count() or 1))))
os.environ.setdefault("MKL_NUM_THREADS", str(max(1, (os.cpu_count() or 1))))
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(max(1, (os.cpu_count() or 1))))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(max(1, (os.cpu_count() or 1))))

np.random.seed(0)
torch.manual_seed(0)
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

try:
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(max(1, os.cpu_count() or 1))
except Exception:
    pass


def _safe_listdir(path_):
    try:
        return os.listdir(path_)
    except FileNotFoundError:
        return []


_DICOM_SERIES_CACHE = {}  # dicom_dir -> list[str] or None
_DICOM_IMAGE_CACHE = {}  # dicom_dir -> sitk.Image (full series)
_DICOM_ARRAY_CACHE = {}  # dicom_dir -> np.ndarray (z,y,x)
_DICOM_MIDSLICE_CACHE = {}  # dicom_dir -> np.ndarray (y,x)
_DICOM_MIDSLICE_INDEX_CACHE = {}  # dicom_dir -> int z_mid
_DICOM_SIZE_CACHE = {}  # dicom_dir -> (x,y,z)

_DICOM_SLICE2D_CACHE = {}  # (dicom_dir, z) -> np.ndarray (y,x) int16


def _get_dicom_series_file_names(dicom_dir: str):
    cached = _DICOM_SERIES_CACHE.get(dicom_dir, "__MISSING__")
    if cached != "__MISSING__":
        if cached is None:
            raise FileNotFoundError(f"No DICOM series found in dir: {dicom_dir}")
        return cached

    reader = sitk.ImageSeriesReader()
    series_file_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    if not series_file_names:
        _DICOM_SERIES_CACHE[dicom_dir] = None
        raise FileNotFoundError(f"No DICOM series found in dir: {dicom_dir}")
    _DICOM_SERIES_CACHE[dicom_dir] = series_file_names
    return series_file_names


def _read_dicom_slice2d_yx(dicom_dir: str, zi: int) -> np.ndarray:
    key = (dicom_dir, int(zi))
    cached = _DICOM_SLICE2D_CACHE.get(key)
    if cached is not None:
        return cached
    series_file_names = _get_dicom_series_file_names(dicom_dir)
    if not series_file_names:
        raise FileNotFoundError(f"No DICOM series found in dir: {dicom_dir}")
    zi = (
        0
        if zi < 0
        else (len(series_file_names) - 1 if zi >= len(series_file_names) else zi)
    )
    img2d = sitk.ReadImage(series_file_names[int(zi)])
    arr = sitk.GetArrayFromImage(img2d)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.dtype != np.int16:
        arr = arr.astype(np.int16, copy=False)
    _DICOM_SLICE2D_CACHE[key] = arr
    return arr


def get_dicom_size_xyz(dicom_dir: str) -> tuple[int, int, int]:
    cached = _DICOM_SIZE_CACHE.get(dicom_dir)
    if cached is not None:
        return cached
    series_file_names = _get_dicom_series_file_names(dicom_dir)
    n = len(series_file_names)
    z_mid = n // 2
    arr = _read_dicom_slice2d_yx(dicom_dir, z_mid)
    y, x = int(arr.shape[0]), int(arr.shape[1])
    out = (x, y, n)
    _DICOM_SIZE_CACHE[dicom_dir] = out
    return out


def read_dicom_midslice_yx(dicom_dir: str) -> tuple[np.ndarray, int]:
    cached = _DICOM_MIDSLICE_CACHE.get(dicom_dir)
    cached_idx = _DICOM_MIDSLICE_INDEX_CACHE.get(dicom_dir)
    if cached is not None and cached_idx is not None:
        return cached, cached_idx

    series_file_names = _get_dicom_series_file_names(dicom_dir)
    n = len(series_file_names)
    z_mid = n // 2
    arr = _read_dicom_slice2d_yx(dicom_dir, z_mid)

    _DICOM_MIDSLICE_CACHE[dicom_dir] = arr
    _DICOM_MIDSLICE_INDEX_CACHE[dicom_dir] = z_mid
    _DICOM_SIZE_CACHE.setdefault(dicom_dir, (int(arr.shape[1]), int(arr.shape[0]), n))
    return arr, z_mid


def read_from_DICOM_dir(dicom_dir: str) -> sitk.Image:
    """
    Robust DICOM series reader using SimpleITK (works with GDCM in Kaggle RSNA dataset).
    """
    cached_img = _DICOM_IMAGE_CACHE.get(dicom_dir)
    if cached_img is not None:
        return cached_img

    series_file_names = _get_dicom_series_file_names(dicom_dir)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(series_file_names)
    reader.MetaDataDictionaryArrayUpdateOff()
    reader.LoadPrivateTagsOff()
    img = reader.Execute()
    _DICOM_IMAGE_CACHE[dicom_dir] = img
    return img


def get_ct_midslice_yx(ct_nii: sitk.Image, dicom_dir: str | None = None) -> np.ndarray:
    if dicom_dir is not None:
        cached = _DICOM_MIDSLICE_CACHE.get(dicom_dir)
        if cached is not None:
            return cached
    z = int(ct_nii.GetSize()[2])
    z_mid = z // 2
    sl = ct_nii[:, :, z_mid]
    arr_yx = sitk.GetArrayFromImage(sl)
    if arr_yx.ndim == 3:
        arr_yx = arr_yx[0]
    if arr_yx.dtype != np.int16:
        arr_yx = arr_yx.astype(np.int16, copy=False)
    if dicom_dir is not None:
        _DICOM_MIDSLICE_CACHE[dicom_dir] = arr_yx
        _DICOM_MIDSLICE_INDEX_CACHE[dicom_dir] = z_mid
        _DICOM_SIZE_CACHE.setdefault(
            dicom_dir, (int(arr_yx.shape[1]), int(arr_yx.shape[0]), z)
        )
    return arr_yx


def get_ct_array_zyx(ct_nii: sitk.Image, dicom_dir: str | None = None) -> np.ndarray:
    if dicom_dir is not None:
        arr = _DICOM_ARRAY_CACHE.get(dicom_dir)
        if arr is not None:
            return arr
    arr = sitk.GetArrayFromImage(ct_nii)  # (z,y,x)
    if arr.dtype != np.int16:
        arr = arr.astype(np.int16, copy=False)
    if dicom_dir is not None:
        _DICOM_ARRAY_CACHE[dicom_dir] = arr
    return arr


def read_dicom_slices_zyx(dicom_dir: str, z_indices: np.ndarray) -> np.ndarray:
    z_idx = np.asarray(z_indices, dtype=np.int32)
    if z_idx.size == 0:
        return np.empty((0, 0, 0), dtype=np.int16)
    series_file_names = _get_dicom_series_file_names(dicom_dir)
    n = len(series_file_names)
    z_idx = np.clip(z_idx, 0, max(0, n - 1))
    uniq = np.unique(z_idx)
    stack = [_read_dicom_slice2d_yx(dicom_dir, int(zi)) for zi in uniq.tolist()]
    stack = np.stack(stack, axis=0)  # (nuniq,y,x)
    pos = {int(v): i for i, v in enumerate(uniq.tolist())}
    order = np.fromiter(
        (pos[int(v)] for v in z_idx.tolist()), dtype=np.int32, count=z_idx.size
    )
    return stack[order]


def read_dicom_roi_arr_zyx(
    dicom_dir: str, bz: int, ez: int
) -> tuple[np.ndarray, tuple[float, float, float]]:
    series_file_names = _get_dicom_series_file_names(dicom_dir)
    n = len(series_file_names)
    bz = max(0, int(bz))
    ez = min(n - 1, int(ez))
    if ez < bz:
        ez = bz
    img0 = sitk.ReadImage(series_file_names[bz])
    try:
        sp_x, sp_y = img0.GetSpacing()[:2]
    except Exception:
        sp_x, sp_y = 1.0, 1.0
    sp = (float(sp_x), float(sp_y), 1.0)
    arrs = [_read_dicom_slice2d_yx(dicom_dir, zi) for zi in range(bz, ez + 1)]
    return np.stack(arrs, axis=0), sp


def get_nii_info(img: sitk.Image):
    return {
        "spacing": img.GetSpacing(),
        "origin": img.GetOrigin(),
        "direction": img.GetDirection(),
        "size": img.GetSize(),
    }


def copy_nii_info(src: sitk.Image, dst: sitk.Image) -> sitk.Image:
    dst.SetSpacing(src.GetSpacing())
    dst.SetOrigin(src.GetOrigin())
    dst.SetDirection(src.GetDirection())
    return dst


def resample(
    img: sitk.Image,
    new_spacing,
    new_origin,
    new_size,
    new_direction,
    interp,
    dtype,
    constant_value=0,
):
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(tuple(new_spacing))
    resampler.SetOutputOrigin(tuple(new_origin))
    resampler.SetSize([int(x) for x in new_size])
    resampler.SetOutputDirection(tuple(new_direction))
    resampler.SetInterpolator(interp)
    resampler.SetDefaultPixelValue(constant_value)
    out = resampler.Execute(img)
    return sitk.Cast(out, dtype)


def get_bbox(mask: np.ndarray):
    if mask is None or mask.size == 0:
        return None
    coords = np.where(mask)
    if len(coords[0]) == 0:
        return None
    bz, ez = int(coords[0].min()), int(coords[0].max())
    by, ey = int(coords[1].min()), int(coords[1].max())
    bx, ex = int(coords[2].min()), int(coords[2].max())
    return (bz, ez, by, ey, bx, ex)


def extend_bbox(
    bbox,
    max_shape,
    list_extend_length=(5.0, 5.0, 5.0),
    spacing=(1.0, 1.0, 1.0),
    approximate_method=np.ceil,
):
    bz, ez, by, ey, bx, ex = bbox
    ext_mm_z, ext_mm_y, ext_mm_x = list_extend_length
    sp_z, sp_y, sp_x = spacing

    ext_z = int(approximate_method(ext_mm_z / max(sp_z, 1e-6)))
    ext_y = int(approximate_method(ext_mm_y / max(sp_y, 1e-6)))
    ext_x = int(approximate_method(ext_mm_x / max(sp_x, 1e-6)))

    bz2 = max(0, bz - ext_z)
    ez2 = min(max_shape[0] - 1, ez + ext_z)
    by2 = max(0, by - ext_y)
    ey2 = min(max_shape[1] - 1, ey + ext_y)
    bx2 = max(0, bx - ext_x)
    ex2 = min(max_shape[2] - 1, ex + ext_x)

    return (bz2, ez2, by2, ey2, bx2, ex2)


def keep_largest_cervical_cc(pred: np.ndarray, spacing_zyx=(1.0, 1.0, 1.0)):
    return pred


print("==> Local imports/utilities ready")
print("==> Test images folders:", len(_safe_listdir(DATA_DIR_TEST)))




## === cell 1
class FractureDetector:
    def __init__(self, predictor_stage1, predictor_stage2, extend_roi=(5.0, 5.0, 5.0)):
        self.predictor_stage1 = predictor_stage1
        self.predictor_stage2 = predictor_stage2
        self.extend_roi = extend_roi

        self.params = {
            "alpha": [0.055, 0.044, 0.052, 0.05, 0.07, 0.077, 0.09, 0.024],
            "beta": [0.475, 0.34, 0.37, 0.38, 0.31, 0.35, 0.38, 0.36],
            "min_score": [0.116, 0.075, 0.015, 0.015, 0.01, 0.02, 0.032, 0.048],
            "max_score": [0.99, 0.999, 0.993, 0.99, 1.0, 0.943, 0.997, 0.999],
        }

        self._alpha = np.asarray(self.params["alpha"], dtype=np.float32)
        self._beta = np.asarray(self.params["beta"], dtype=np.float32)
        self._min_score = np.asarray(self.params["min_score"], dtype=np.float32)
        self._max_score = np.asarray(self.params["max_score"], dtype=np.float32)

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

    def predict_stage1(self, ct_nii, ct_arr_zyx=None, dicom_dir: str | None = None):
        pred = self.predictor_stage1.predict_mask_labels_zyx(
            ct_nii, ct_arr_zyx=ct_arr_zyx, dicom_dir=dicom_dir
        )
        pred = keep_largest_cervical_cc(pred, ct_nii.GetSpacing()[::-1])
        return pred

    def predict_stage2(self, ct_nii, ct_arr_zyx=None):
        pred = self.predictor_stage2.predict_prob_map_zyx(ct_nii, ct_arr_zyx=ct_arr_zyx)
        return pred.astype(np.float32, copy=False)

    def get_score(self, pred_c1_c7, pred_fracture):
        output = np.zeros(8, np.float32)  # Overall, C1-C7

        if (pred_c1_c7 is not None) and (pred_fracture is not None):
            lbl = pred_c1_c7.astype(np.uint8, copy=False)
            pr = pred_fracture.astype(np.float32, copy=False)

            mx = int(lbl.max(initial=0))
            if mx > 7:
                lbl = lbl.copy()
                lbl[lbl > 7] = 0

            mask0 = (lbl > 0) & (pr >= self._alpha[0])
            roi0 = pr[mask0]
            if roi0.size == 0:
                output[0] = self._min_score[0]
            else:
                output[0] = max(
                    float(self._min_score[0]),
                    min(
                        float(self._max_score[0]),
                        float(np.percentile(roi0, 100.0 * float(self._beta[0]))),
                    ),
                )

            for ci in range(1, 8):
                m = (lbl == ci) & (pr >= self._alpha[ci])
                roi = pr[m]
                if roi.size == 0:
                    output[ci] = self._min_score[ci]
                else:
                    output[ci] = max(
                        float(self._min_score[ci]),
                        min(
                            float(self._max_score[ci]),
                            float(np.percentile(roi, 100.0 * float(self._beta[ci]))),
                        ),
                    )
        return output

    def predict(self, list_test_files, log_every=50):
        count = 0
        overall_time_start = time.time()
        n_total = len(list_test_files)

        for file in list_test_files:
            case_id = os.path.basename(file.rstrip("/"))
            count += 1
            if (count % log_every == 1) or (count == 1) or (count == n_total):
                print(f"==> Predicting {count}/{n_total}: {case_id}")

            time_start = time.time()

            mid_yx, _ = read_dicom_midslice_yx(file)
            if int((mid_yx > -500).sum()) < 1000:
                self.results[case_id] = self.get_score(None, None)
                if (count % log_every == 1) or (count == n_total):
                    print(
                        f"        ----> Last case time: {time.time() - time_start:.3f}s | "
                        f"Total: {time.time() - overall_time_start:.1f}s"
                    )
                continue

            x, y, z = get_dicom_size_xyz(file)

            y0, y1 = int(0.25 * y), int(0.85 * y)
            x0, x1 = int(0.25 * x), int(0.75 * x)
            denom = max(1, (y1 - y0) * (x1 - x0))

            n_samp = min(96, z)
            if n_samp <= 1:
                z_indices = np.array([0], dtype=np.int32)
            else:
                z_indices = np.linspace(0, z - 1, n_samp).astype(np.int32)

            samp_arr_zyx = read_dicom_slices_zyx(file, z_indices)
            region = samp_arr_zyx[:, y0:y1, x0:x1] > -500
            z_counts = (
                region.reshape(region.shape[0], -1)
                .sum(axis=1)
                .astype(np.int64, copy=False)
            )
            valid = z_counts > (0.02 * denom)

            if int(valid.sum()) < max(10, z // 10):
                z_min, z_max = 0, z - 1
            else:
                idx = np.where(valid)[0]
                z_min, z_max = int(z_indices[idx.min()]), int(z_indices[idx.max()])

            if z_max <= z_min:
                self.results[case_id] = self.get_score(None, None)
                if (count % log_every == 1) or (count == n_total):
                    print(
                        f"        ----> Last case time: {time.time() - time_start:.3f}s | "
                        f"Total: {time.time() - overall_time_start:.1f}s"
                    )
                continue

            pred1_bbox = (z_min, z_max, y0, y1 - 1, x0, x1 - 1)

            spacing_zyx = (1.0, 1.0, 1.0)
            ct_full = _DICOM_IMAGE_CACHE.get(file)
            if ct_full is not None:
                spacing_xyz = ct_full.GetSpacing()
                spacing_zyx = spacing_xyz[::-1]
            c1_c7_bbox = extend_bbox(
                pred1_bbox,
                max_shape=(z, y, x),
                list_extend_length=self.extend_roi,
                spacing=spacing_zyx,
                approximate_method=np.ceil,
            )

            bz, ez, by, ey, bx, ex = c1_c7_bbox

            roi_full_arr_zyx, _ = read_dicom_roi_arr_zyx(file, bz=bz, ez=ez)
            roi_arr_zyx = roi_full_arr_zyx[:, by : ey + 1, bx : ex + 1].astype(
                np.int16, copy=False
            )

            roi_z = ez - bz + 1
            roi_y = ey - by + 1
            roi_x = ex - bx + 1
            roi_pred_1 = np.zeros((roi_z, roi_y, roi_x), dtype=np.uint8)

            bin_edges = np.linspace(z_min, z_max + 1, 8).astype(int)
            for i in range(7):
                a, b = int(bin_edges[i]), int(bin_edges[i + 1])
                if b <= a:
                    continue
                aa = max(a, bz)
                bb = min(b, ez + 1)
                if bb <= aa:
                    continue

                yy0 = max(y0, by)
                yy1 = min(y1, ey + 1)
                xx0 = max(x0, bx)
                xx1 = min(x1, ex + 1)
                if (yy1 <= yy0) or (xx1 <= xx0):
                    continue

                roi_pred_1[
                    aa - bz : bb - bz, yy0 - by : yy1 - by, xx0 - bx : xx1 - bx
                ] = (i + 1)

            roi_pred_2 = self.predictor_stage2.predict_prob_map_zyx(
                ct_nii=None, ct_arr_zyx=roi_arr_zyx
            )
            roi_pred_2 = roi_pred_2.astype(np.float32, copy=False)
            self.results[case_id] = self.get_score(roi_pred_1, roi_pred_2)

            if (count % log_every == 1) or (count == n_total):
                print(
                    f"        ----> Last case time: {time.time() - time_start:.3f}s | "
                    f"Total: {time.time() - overall_time_start:.1f}s"
                )




## === cell 2
class DummyStage1Predictor:
    def __init__(self):
        pass

    def predict_mask_labels_zyx(
        self,
        ct_nii: sitk.Image,
        ct_arr_zyx: np.ndarray | None = None,
        dicom_dir: str | None = None,
    ) -> np.ndarray:
        size_xyz = ct_nii.GetSize()  # (x,y,z)
        x, y, z = int(size_xyz[0]), int(size_xyz[1]), int(size_xyz[2])

        if ct_arr_zyx is None:
            mid_arr_yx = get_ct_midslice_yx(ct_nii, dicom_dir=dicom_dir)
        else:
            z_mid = z // 2
            mid_arr_yx = ct_arr_zyx[z_mid]  # (y,x)

        if int((mid_arr_yx > -500).sum()) < 1000:
            return np.zeros((z, y, x), dtype=np.uint8)

        y0, y1 = int(0.25 * y), int(0.85 * y)
        x0, x1 = int(0.25 * x), int(0.75 * x)
        denom = max(1, (y1 - y0) * (x1 - x0))

        n_samp = min(96, z)
        if n_samp <= 1:
            z_indices = np.array([0], dtype=np.int32)
        else:
            z_indices = np.linspace(0, z - 1, n_samp).astype(np.int32)

        if ct_arr_zyx is None:
            ct_arr_zyx = get_ct_array_zyx(ct_nii, dicom_dir=dicom_dir)

        region = ct_arr_zyx[z_indices, y0:y1, x0:x1] > -500
        z_counts = (
            region.reshape(region.shape[0], -1).sum(axis=1).astype(np.int64, copy=False)
        )

        valid = z_counts > (0.02 * denom)

        if int(valid.sum()) < max(10, z // 10):
            z_min, z_max = 0, z - 1
        else:
            idx = np.where(valid)[0]
            z_min, z_max = int(z_indices[idx.min()]), int(z_indices[idx.max()])

        labels = np.zeros((z, y, x), dtype=np.uint8)
        if z_max <= z_min:
            return labels

        bin_edges = np.linspace(z_min, z_max + 1, 8).astype(int)
        for i in range(7):
            a, b = int(bin_edges[i]), int(bin_edges[i + 1])
            if b <= a:
                continue
            labels[a:b, y0:y1, x0:x1] = i + 1

        return labels


_GAUSS_KERNEL_CACHE = {}
_GAUSS_KERNEL_3D_CACHE = {}


def _gaussian_kernel1d(sigma: float, device: torch.device, dtype: torch.dtype):
    key = (float(sigma), str(device), str(dtype))
    k = _GAUSS_KERNEL_CACHE.get(key)
    if k is not None:
        return k
    radius = max(1, int(math.ceil(3.0 * float(sigma))))
    xs = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(xs * xs) / (2.0 * float(sigma) * float(sigma)))
    kernel = kernel / kernel.sum()
    _GAUSS_KERNEL_CACHE[key] = kernel
    return kernel


def _gaussian_kernels3d(sigma: float, device: torch.device, dtype: torch.dtype):
    key = (float(sigma), str(device), str(dtype))
    out = _GAUSS_KERNEL_3D_CACHE.get(key)
    if out is not None:
        return out
    k1 = _gaussian_kernel1d(sigma, device=device, dtype=dtype)
    rz = (k1.numel() - 1) // 2
    kz = k1.view(1, 1, -1, 1, 1)
    ky = k1.view(1, 1, 1, -1, 1)
    kx = k1.view(1, 1, 1, 1, -1)
    _GAUSS_KERNEL_3D_CACHE[key] = (kz, ky, kx, rz)
    return kz, ky, kx, rz


_TORCH_SMOOTH_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _smooth_gaussian_zyx_torch(prob_zyx: np.ndarray, sigma: float = 0.7) -> np.ndarray:
    if sigma <= 0:
        return prob_zyx.astype(np.float32, copy=False)

    device = _TORCH_SMOOTH_DEVICE
    prob_f32 = (
        prob_zyx
        if prob_zyx.dtype == np.float32
        else prob_zyx.astype(np.float32, copy=False)
    )
    t = torch.from_numpy(prob_f32).to(device=device, non_blocking=False)
    t = t.unsqueeze(0).unsqueeze(0)  # (1,1,z,y,x)

    kz, ky, kx, rz = _gaussian_kernels3d(float(sigma), device=device, dtype=t.dtype)

    t = F.conv3d(t, kz, padding=(rz, 0, 0))
    t = F.conv3d(t, ky, padding=(0, rz, 0))
    t = F.conv3d(t, kx, padding=(0, 0, rz))

    out = t.squeeze(0).squeeze(0).detach().to(device="cpu").numpy()
    return out.astype(np.float32, copy=False)


class DummyStage2Predictor:
    def __init__(self):
        self._sigma = 0.7

    def predict_prob_map_zyx(
        self, ct_nii: sitk.Image | None, ct_arr_zyx: np.ndarray | None = None
    ) -> np.ndarray:
        if ct_arr_zyx is None:
            arr = sitk.GetArrayFromImage(ct_nii)  # (z,y,x)
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32, copy=False)
        else:
            arr = ct_arr_zyx.astype(np.float32, copy=False)

        bone = (arr - 150.0) / 850.0
        np.clip(bone, 0.0, 1.0, out=bone)

        dz = np.abs(np.diff(bone, axis=0, prepend=bone[:1]))
        dy = np.abs(np.diff(bone, axis=1, prepend=bone[:, :1]))
        dx = np.abs(np.diff(bone, axis=2, prepend=bone[:, :, :1]))
        edge = dz
        edge += dy
        edge += dx
        edge /= 3.0
        np.clip(edge, 0.0, 1.0, out=edge)

        prob = bone
        prob = 0.65 * prob + 0.75 * edge
        np.clip(prob, 0.0, 1.0, out=prob)

        prob = _smooth_gaussian_zyx_torch(prob, sigma=self._sigma)
        np.clip(prob, 0.0, 1.0, out=prob)
        return prob.astype(np.float32, copy=False)


def estimate_label_priors(train_csv_path: str):
    df = pd.read_csv(train_csv_path)
    cols = ["patient_overall", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    priors = df[cols].mean().values.astype(np.float32)
    priors = np.clip(priors, 1e-3, 1 - 1e-3)
    return priors


PRIORS = estimate_label_priors(TRAIN_CSV)
print("==> Estimated priors (overall, C1..C7):", PRIORS)


def calibrate_scores(raw_scores_8, priors_8, blend=0.65):
    raw = np.asarray(raw_scores_8, dtype=np.float32)
    pri = np.asarray(priors_8, dtype=np.float32)
    out = blend * raw + (1.0 - blend) * pri
    return np.clip(out, 1e-4, 1 - 1e-4)




## === cell 3
time_start = time.time()

SAVE_CSV = "submission.csv"

test_df = pd.read_csv(
    TEST_CSV, usecols=["StudyInstanceUID", "prediction_type", "row_id"]
)
test_df = test_df.sort_values("row_id", kind="mergesort").reset_index(drop=True)

try:
    with os.scandir(DATA_DIR_TEST) as it:
        list_DICOM_dirs = [e.path for e in it if e.is_dir()]
    list_DICOM_dirs.sort()
except FileNotFoundError:
    list_DICOM_dirs = []

print(f"==> Total test cases (folders): {len(list_DICOM_dirs)}")
print(f"==> Total submission rows expected: {len(test_df)}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("==> Using device:", device)

with torch.no_grad():
    predictor_1 = DummyStage1Predictor()
    predictor_2 = DummyStage2Predictor()

    c2f_predictor = FractureDetector(
        predictor_stage1=predictor_1, predictor_stage2=predictor_2
    )
    c2f_predictor.predict(list_test_files=list_DICOM_dirs, log_every=75)
    results = c2f_predictor.results

priors_8 = PRIORS.astype(np.float32, copy=False)

case_ids = np.fromiter(
    (os.path.basename(p.rstrip("/")) for p in list_DICOM_dirs),
    dtype=object,
    count=len(list_DICOM_dirs),
)

raw_scores_mat = np.vstack([results[c] for c in case_ids]).astype(
    np.float32, copy=False
)
cal_scores_mat = calibrate_scores(raw_scores_mat, priors_8[None, :], blend=0.65)

uid_to_idx = {uid: i for i, uid in enumerate(case_ids.tolist())}
ptype_to_col = {
    "patient_overall": 0,
    "C1": 1,
    "C2": 2,
    "C3": 3,
    "C4": 4,
    "C5": 5,
    "C6": 6,
    "C7": 7,
}

uids = test_df["StudyInstanceUID"].to_numpy(dtype=object, copy=False)
ptypes = test_df["prediction_type"].to_numpy(dtype=object, copy=False)

out_pred = np.empty(len(test_df), dtype=np.float32)
missing_mask = np.zeros(len(test_df), dtype=bool)

for i in range(len(test_df)):
    uid = uids[i]
    idx = uid_to_idx.get(uid, None)
    if idx is None:
        missing_mask[i] = True
        continue
    col = ptype_to_col.get(ptypes[i], None)
    if col is None:
        missing_mask[i] = True
        continue
    out_pred[i] = cal_scores_mat[idx, col]

default_scores = calibrate_scores(
    raw_scores_8=np.array(
        [0.116, 0.075, 0.015, 0.015, 0.01, 0.02, 0.032, 0.048], dtype=np.float32
    ),
    priors_8=priors_8,
    blend=0.0,
).astype(np.float32, copy=False)

if missing_mask.any():
    miss_types = ptypes[missing_mask].astype(str)
    fill = np.empty(miss_types.shape[0], dtype=np.float32)
    is_overall = miss_types == "patient_overall"
    fill[is_overall] = float(default_scores[0])
    if (~is_overall).any():
        c_types = miss_types[~is_overall]
        ci = np.fromiter(
            (int(s[1:]) for s in c_types.tolist()), dtype=np.int64, count=len(c_types)
        )
        fill[~is_overall] = default_scores[ci]
    out_pred[missing_mask] = fill

sub = pd.DataFrame(
    {
        "row_id": test_df["row_id"].values,
        "fractured": np.clip(out_pred, 1e-4, 1 - 1e-4).astype(np.float32, copy=False),
    }
)
sub.to_csv(SAVE_CSV, index=False)

print("==> Wrote", SAVE_CSV, "rows:", len(sub), "cols:", list(sub.columns))
print(sub.head(10))
print(f"==> Finish using time: {time.time() - time_start:.2f} seconds")
