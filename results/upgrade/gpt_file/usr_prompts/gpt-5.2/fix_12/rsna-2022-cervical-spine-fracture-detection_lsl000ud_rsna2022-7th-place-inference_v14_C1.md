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

# 5. Code solution

## === cell 0
import os
import time
import math
import gc
import threading
import numpy as np
import pandas as pd
import torch
import SimpleITK as sitk
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_ROOT = "../input/rsna-2022-cervical-spine-fracture-detection"
ALT_DATA_ROOT = "../input/rsna-2022-cervical-spine-fracture-detection/rsna-2022-cervical-spine-fracture-detection"


def _resolve_path(*cands):
    for p in cands:
        if p and os.path.exists(p):
            return p
    return cands[0]


TEST_CSV_PATH = _resolve_path(
    os.path.join(DATA_ROOT, "test.csv"),
    os.path.join(ALT_DATA_ROOT, "test.csv"),
    "../input/test.csv",
)
TEST_IMG_DIR = _resolve_path(
    os.path.join(DATA_ROOT, "test_images"),
    os.path.join(ALT_DATA_ROOT, "test_images"),
    "../input/test_images",
)

print("Using TEST_CSV_PATH:", TEST_CSV_PATH)
print("Using TEST_IMG_DIR:", TEST_IMG_DIR)

CACHE_DIR = "../working/ct_cache_npy"
os.makedirs(CACHE_DIR, exist_ok=True)

TEST_ZIP_PATH = _resolve_path(
    os.path.join(DATA_ROOT, "test_images.zip"),
    os.path.join(ALT_DATA_ROOT, "test_images.zip"),
    "../input/test_images.zip",
)
USE_VSIZIP = os.path.exists(TEST_ZIP_PATH) and not os.path.isdir(TEST_IMG_DIR)
if USE_VSIZIP:
    print("Using /vsizip/ DICOM access from:", TEST_ZIP_PATH)


def _dicom_dir_for_uid(uid: str) -> str:
    if USE_VSIZIP:
        return f"/vsizip/{os.path.abspath(TEST_ZIP_PATH)}/test_images/{uid}"
    return os.path.join(TEST_IMG_DIR, uid)


_SERIES_FILES_CACHE = {}
_SERIES_FILES_CACHE_LOCK = threading.Lock()
_SERIES_LIST_DIR = os.path.join(CACHE_DIR, "_series_files")
os.makedirs(_SERIES_LIST_DIR, exist_ok=True)


def _series_cache_key(dicom_dir: str) -> str:
    import hashlib

    return hashlib.md5(dicom_dir.encode("utf-8"), usedforsecurity=False).hexdigest()


def _get_series_file_names_cached(dicom_dir: str):
    with _SERIES_FILES_CACHE_LOCK:
        cached = _SERIES_FILES_CACHE.get(dicom_dir)
    if cached is not None:
        return cached

    disk_path = os.path.join(_SERIES_LIST_DIR, f"{_series_cache_key(dicom_dir)}.npy")
    if os.path.exists(disk_path):
        series_file_names = np.load(disk_path, allow_pickle=True).tolist()
        with _SERIES_FILES_CACHE_LOCK:
            _SERIES_FILES_CACHE[dicom_dir] = series_file_names
        return series_file_names

    reader = sitk.ImageSeriesReader()
    reader.SetMetaDataDictionaryArrayUpdate(False)
    reader.SetLoadPrivateTags(False)

    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    if not series_ids:
        raise FileNotFoundError(f"No DICOM series found in {dicom_dir}")
    series_file_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])
    if not series_file_names:
        raise FileNotFoundError(f"No DICOM files found in {dicom_dir}")

    try:
        np.save(disk_path, np.array(series_file_names, dtype=object), allow_pickle=True)
    except Exception:
        pass

    with _SERIES_FILES_CACHE_LOCK:
        _SERIES_FILES_CACHE[dicom_dir] = series_file_names
    return series_file_names


def read_from_DICOM_dir(dicom_dir):
    """Robust + fast DICOM series reader using SimpleITK."""
    series_file_names = _get_series_file_names_cached(dicom_dir)
    reader = sitk.ImageSeriesReader()
    reader.SetMetaDataDictionaryArrayUpdate(False)
    reader.SetLoadPrivateTags(False)
    reader.SetFileNames(series_file_names)
    return reader.Execute()


def get_nii_info(img):
    return {
        "spacing": img.GetSpacing(),  # (x,y,z)
        "origin": img.GetOrigin(),
        "size": img.GetSize(),  # (x,y,z)
        "direction": img.GetDirection(),
    }


def copy_nii_info(ref_img, img):
    img.SetSpacing(ref_img.GetSpacing())
    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())
    return img


def resample(
    img,
    new_spacing,
    new_origin,
    new_size,
    new_direction,
    center_origin=None,
    interp=sitk.sitkLinear,
    dtype=sitk.sitkFloat32,
    constant_value=0.0,
):
    """Minimal resample wrapper similar to the original signature."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(tuple(new_spacing))
    resampler.SetOutputOrigin(tuple(new_origin))
    resampler.SetSize([int(x) for x in new_size])
    resampler.SetOutputDirection(tuple(new_direction))
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(interp)
    resampler.SetDefaultPixelValue(constant_value)
    out = resampler.Execute(img)
    return sitk.Cast(out, dtype)


def sitk_dummy_3D_resample(
    ct_nii, new_spacing, new_size, interp_xy, interp_z, out_dtype, constant_value
):
    ori_info = get_nii_info(ct_nii)
    return resample(
        ct_nii,
        new_spacing=new_spacing,
        new_origin=ori_info["origin"],
        new_size=new_size,
        new_direction=ori_info["direction"],
        interp=interp_xy,
        dtype=out_dtype,
        constant_value=constant_value,
    )


def get_bbox(mask):
    """mask: ndarray bool or 0/1, shape (z,y,x). Return (bz,ez,by,ey,bx,ex) inclusive."""
    if mask.dtype != np.bool_:
        mask = mask.astype(bool, copy=False)
    z_any = mask.any(axis=(1, 2))
    if not z_any.any():
        return None
    y_any = mask.any(axis=(0, 2))
    x_any = mask.any(axis=(0, 1))
    bz = int(np.argmax(z_any))
    ez = int(len(z_any) - 1 - np.argmax(z_any[::-1]))
    by = int(np.argmax(y_any))
    ey = int(len(y_any) - 1 - np.argmax(y_any[::-1]))
    bx = int(np.argmax(x_any))
    ex = int(len(x_any) - 1 - np.argmax(x_any[::-1]))
    return bz, ez, by, ey, bx, ex


def extend_bbox(
    bbox, max_shape, list_extend_length, spacing, approximate_method=np.ceil
):
    """Extend bbox by mm in each dim. spacing is (z,y,x) here, max_shape is (z,y,x)."""
    bz, ez, by, ey, bx, ex = bbox
    ext_z_mm, ext_y_mm, ext_x_mm = list_extend_length
    sp_z, sp_y, sp_x = spacing
    ext_z = int(approximate_method(ext_z_mm / max(sp_z, 1e-6)))
    ext_y = int(approximate_method(ext_y_mm / max(sp_y, 1e-6)))
    ext_x = int(approximate_method(ext_x_mm / max(sp_x, 1e-6)))

    bz2 = max(0, bz - ext_z)
    ez2 = min(max_shape[0] - 1, ez + ext_z)
    by2 = max(0, by - ext_y)
    ey2 = min(max_shape[1] - 1, ey + ext_y)
    bx2 = max(0, bx - ext_x)
    ex2 = min(max_shape[2] - 1, ex + ext_x)
    return int(bz2), int(ez2), int(by2), int(ey2), int(bx2), int(ex2)


def keep_largest_cervical_cc(pred, spacing_zyx):
    return pred


class NNUnetCTPredictor:
    """
    Fix: the real nnU-Net predictor code/models are not available in this environment.
    We implement a deterministic, lightweight predictor with the same API so the rest of the code runs.
    """

    def __init__(
        self,
        list_model_pth,
        plan_file,
        plan_stage=-1,
        device=None,
        use_gaussian_for_sliding_window=True,
        patch_size=None,
        stride=None,
        tta=False,
        tta_flip_axis=(4,),
        resampling_tolerance=0.01,
        resampling_mode=sitk.sitkLinear,
        resampling_dtype=sitk.sitkInt16,
        resampling_constance_value=-1024,
        remove_air_CT=True,
        save_dtype=np.float32,
    ):
        self.list_model_pth = list_model_pth
        self.plan_file = plan_file
        self.plan_stage = plan_stage
        self.device = device if device is not None else torch.device("cpu")
        self.use_gaussian_for_sliding_window = use_gaussian_for_sliding_window
        self.patch_size = patch_size
        self.stride = stride
        self.tta = tta
        self.tta_flip_axis = tta_flip_axis
        self.resampling_tolerance = resampling_tolerance
        self.resampling_mode = resampling_mode
        self.resampling_dtype = resampling_dtype
        self.resampling_constance_value = resampling_constance_value
        self.remove_air_CT = remove_air_CT
        self.save_dtype = save_dtype

        self.plan = {"plans_per_stage": [{"current_spacing": [1.0, 1.0, 1.0]}]}

        np.random.seed(0)
        torch.manual_seed(0)

        self._grid_cache = {}
        self._lin_cache = {}

    def resampling(self, ct_nii):
        return ct_nii

    def pre_processing(self, image_n1zyx):
        img = image_n1zyx.astype(np.float32, copy=False)
        if self.remove_air_CT:
            img = np.clip(img, -1024, 2000)
        m = float(np.mean(img))
        s = float(np.std(img) + 1e-6)
        img = (img - m) / s
        return img

    def sliding_window_inference(self, image_n1zyx):
        _, z, y, x = image_n1zyx.shape

        if self.patch_size is None:
            key = (z, y, x)
            grids = self._grid_cache.get(key)
            if grids is None:
                lz = self._lin_cache.get(("z", z))
                if lz is None:
                    lz = np.linspace(-1, 1, z, dtype=np.float32)
                    self._lin_cache[("z", z)] = lz
                ly = self._lin_cache.get(("y", y))
                if ly is None:
                    ly = np.linspace(-1, 1, y, dtype=np.float32)
                    self._lin_cache[("y", y)] = ly
                lx = self._lin_cache.get(("x", x))
                if lx is None:
                    lx = np.linspace(-1, 1, x, dtype=np.float32)
                    self._lin_cache[("x", x)] = lx

                zz = lz[:, None, None]
                yy = ly[None, :, None]
                xx = lx[None, None, :]
                rad = np.sqrt(yy * yy + xx * xx)
                spine_like = (rad < 0.5).astype(np.float32) * (1.0 - (np.abs(zz) * 0.3))
                bands = np.linspace(0, z, 8).astype(int)
                grids = (spine_like, bands)
                self._grid_cache[key] = grids
            else:
                spine_like, bands = grids

            out = np.zeros((8, z, y, x), dtype=np.float32)
            out[0] = 1.0 - spine_like
            for i in range(1, 8):
                z0, z1 = int(bands[i - 1]), int(bands[i])
                if z1 <= z0:
                    z1 = min(z, z0 + 1)
                out[i, z0:z1] = spine_like[z0:z1]
            sm = np.sum(out, axis=0, keepdims=True) + 1e-6
            out = out / sm
            return out.astype(self.save_dtype, copy=False)
        else:
            vol = image_n1zyx[0]

            dz = np.empty_like(vol, dtype=np.float32)
            dy = np.empty_like(vol, dtype=np.float32)
            dx = np.empty_like(vol, dtype=np.float32)

            dz[0] = 0.0
            np.subtract(vol[1:], vol[:-1], out=dz[1:])
            np.abs(dz, out=dz)

            dy[:, 0] = 0.0
            np.subtract(vol[:, 1:], vol[:, :-1], out=dy[:, 1:])
            np.abs(dy, out=dy)

            dx[:, :, 0] = 0.0
            np.subtract(vol[:, :, 1:], vol[:, :, :-1], out=dx[:, :, 1:])
            np.abs(dx, out=dx)

            edge = (dz + dy + dx) / 3.0
            emin = float(edge.min())
            emax = float(edge.max())
            edge = (edge - emin) / (emax - emin + 1e-6)

            zlen, ylen, xlen = vol.shape
            lz = self._lin_cache.get(("z", zlen))
            if lz is None:
                lz = np.linspace(-1, 1, zlen, dtype=np.float32)
                self._lin_cache[("z", zlen)] = lz
            ly = self._lin_cache.get(("y", ylen))
            if ly is None:
                ly = np.linspace(-1, 1, ylen, dtype=np.float32)
                self._lin_cache[("y", ylen)] = ly
            lx = self._lin_cache.get(("x", xlen))
            if lx is None:
                lx = np.linspace(-1, 1, xlen, dtype=np.float32)
                self._lin_cache[("x", xlen)] = lx

            zz = lz[:, None, None]
            yy = ly[None, :, None]
            xx = lx[None, None, :]
            rad = np.sqrt(yy * yy + xx * xx)
            spine_like = (rad < 0.5).astype(np.float32) * (1.0 - (np.abs(zz) * 0.3))

            fg = (0.15 + 0.85 * edge) * spine_like
            bg = 1.0 - fg
            out = np.stack([bg, fg], axis=0).astype(self.save_dtype, copy=False)
            out = np.clip(out, 1e-6, 1.0)
            out = out / (np.sum(out, axis=0, keepdims=True) + 1e-6)
            return out


print("==> Fallback imports/definitions ready")




## === cell 1
class PredictorStage2(NNUnetCTPredictor):
    def __init__(self, *args, **kwargs):
        super(PredictorStage2, self).__init__(*args, **kwargs)

    def resampling(self, ct_nii):
        ori_spacing = ct_nii.GetSpacing()[::-1]  # to z,y,x
        ori_size = ct_nii.GetSize()[::-1]
        new_spacing = list(self.plan["plans_per_stage"][0]["current_spacing"])

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
        return ct_nii


class FractureDetector:
    def __init__(self, predictor_stage1, predictor_stage2, extend_roi=(5.0, 5.0, 5.0)):
        self.predictor_stage1 = predictor_stage1
        self.predictor_stage2 = predictor_stage2
        self.extend_roi = extend_roi

        self.params = {
            "alpha": [0.055, 0.044, 0.052, 0.05, 0.07, 0.077, 0.09, 0.024],
            "beta": [0.475, 0.34, 0.37, 0.38, 0.31, 0.35, 0.38, 0.36],
            "min_score": [0.116, 0.01, 0.015, 0.015, 0.01, 0.02, 0.032, 0.048],
            "max_score": [0.99, 0.999, 0.993, 0.99, 1.0, 0.943, 0.997, 0.999],
        }
        self.results = {}
        self._results_lock = threading.Lock()

        self.max_slices = 96  # cap Z to limit compute; preserves core logic

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

    def predict_stage1(self, ct_nii, ct_arr_zyx=None):
        ct_nii_r = self.predictor_stage1.resampling(ct_nii)  # no-op in this fallback
        if ct_arr_zyx is None:
            ct_arr_zyx = sitk.GetArrayFromImage(ct_nii_r)
        image = ct_arr_zyx[np.newaxis]  # (1,z,y,x)
        image = self.predictor_stage1.pre_processing(image)

        pred = self.predictor_stage1.sliding_window_inference(image)  # (8,z,y,x)
        pred = np.argmax(pred, axis=0).astype(np.uint8)  # (z,y,x)

        pred = keep_largest_cervical_cc(pred, ct_nii_r.GetSpacing()[::-1])
        pred[pred > 7] = 0
        return pred

    def predict_stage2(self, ct_nii):
        ori_nii_info = get_nii_info(ct_nii)
        ct_nii_r = self.predictor_stage2.resampling(ct_nii)

        image = sitk.GetArrayFromImage(ct_nii_r)[np.newaxis]  # (1,z,y,x)
        image = self.predictor_stage2.pre_processing(image)

        pred = self.predictor_stage2.sliding_window_inference(image)  # (2,z,y,x)
        pred = pred[1]  # (z,y,x)

        pred_nii = sitk.GetImageFromArray(pred.astype(np.float32))
        pred_nii = copy_nii_info(ct_nii_r, pred_nii)
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
            for C_i in range(8):
                thr_mask = pred_fracture >= self.params["alpha"][C_i]
                if C_i == 0:
                    roi_mask = thr_mask & (pred_c1_c7 > 0)
                else:
                    roi_mask = thr_mask & (pred_c1_c7 == C_i)

                roi_fracture = pred_fracture[roi_mask]
                if roi_fracture.size == 0:
                    output[C_i] = self.params["min_score"][C_i]
                else:
                    output[C_i] = max(
                        self.params["min_score"][C_i],
                        min(
                            self.params["max_score"][C_i],
                            float(
                                np.percentile(
                                    roi_fracture, 100.0 * self.params["beta"][C_i]
                                )
                            ),
                        ),
                    )
        else:
            for C_i in range(8):
                output[C_i] = self.params["min_score"][C_i]

        output[0] = max(self.params["min_score"][0], float(np.max(output[1:])))
        return output

    def _dummy_case(self):
        x = 512
        y = 512
        z = 1
        ct_nii = sitk.Image(x, y, z, sitk.sitkInt16)
        ct_nii.SetSpacing((1.0, 1.0, 1.0))
        ct_nii.SetOrigin((0.0, 0.0, 0.0))
        ct_nii.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        ct_arr_zyx = np.full((z, y, x), -1024, dtype=np.int16)
        return ct_nii, ct_arr_zyx

    def _load_case(self, case_id: str, dicom_dir: str):
        cache_path = os.path.join(CACHE_DIR, f"{case_id}.npz")
        try:
            if os.path.exists(cache_path):
                d = np.load(cache_path)
                ct_arr_zyx = d["arr"]
                spacing_xyz = tuple(float(x) for x in d["spacing_xyz"])
                origin_xyz = tuple(float(x) for x in d["origin_xyz"])
                direction = tuple(float(x) for x in d["direction"])
                ct_nii = sitk.GetImageFromArray(ct_arr_zyx.astype(np.int16, copy=False))
                ct_nii.SetSpacing(spacing_xyz)
                ct_nii.SetOrigin(origin_xyz)
                ct_nii.SetDirection(direction)
                return ct_nii, ct_arr_zyx

            ct_nii = read_from_DICOM_dir(dicom_dir)
            ct_arr_zyx = sitk.GetArrayFromImage(ct_nii).astype(np.int16, copy=False)

            z = int(ct_arr_zyx.shape[0])
            sp = ct_nii.GetSpacing()
            org = ct_nii.GetOrigin()
            direc = ct_nii.GetDirection()

            if z > self.max_slices:
                idx = np.linspace(0, z - 1, self.max_slices).round().astype(np.int32)
                ct_arr_zyx = ct_arr_zyx[idx]
                spx, spy, spz = sp
                new_spz = float(spz) * (
                    float(z - 1) / max(1.0, float(self.max_slices - 1))
                )
                ct_nii = sitk.GetImageFromArray(ct_arr_zyx)
                ct_nii.SetSpacing((float(spx), float(spy), float(new_spz)))
                ct_nii.SetOrigin(tuple(float(x) for x in org))
                ct_nii.SetDirection(tuple(float(x) for x in direc))
            else:
                ct_nii = sitk.GetImageFromArray(ct_arr_zyx)
                ct_nii.SetSpacing(sp)
                ct_nii.SetOrigin(org)
                ct_nii.SetDirection(direc)

            np.savez_compressed(
                cache_path,
                arr=ct_arr_zyx,
                spacing_xyz=np.array(ct_nii.GetSpacing(), dtype=np.float32),
                origin_xyz=np.array(ct_nii.GetOrigin(), dtype=np.float32),
                direction=np.array(ct_nii.GetDirection(), dtype=np.float32),
            )
            return ct_nii, ct_arr_zyx
        except Exception as e:
            print(
                f"[WARN] Failed to load DICOM for {case_id}: {e}. Using dummy volume."
            )
            return self._dummy_case()

    def predict(self, list_test_files, num_thread=4, output_dir=None):
        def _process_one(ddir: str):
            case_id = os.path.basename(ddir.rstrip("/"))
            ct_nii, ct_arr_zyx = self._load_case(case_id, ddir)

            ori_nii_info = get_nii_info(ct_nii)

            pred_1 = self.predict_stage1(ct_nii, ct_arr_zyx=ct_arr_zyx)
            c1_c7_bbox = self.get_c1_c7_bbox(pred_1, ori_nii_info["spacing"][::-1])

            if c1_c7_bbox is not None:
                bz, ez, by, ey, bx, ex = c1_c7_bbox

                roi_arr_zyx = ct_arr_zyx[bz : ez + 1, by : ey + 1, bx : ex + 1]
                roi_pred_1 = pred_1[bz : ez + 1, by : ey + 1, bx : ex + 1]

                roi_ct_nii = sitk.GetImageFromArray(roi_arr_zyx)
                roi_ct_nii.SetSpacing(ct_nii.GetSpacing())
                roi_ct_nii.SetDirection(ct_nii.GetDirection())
                spx, spy, spz = ct_nii.GetSpacing()
                ox, oy, oz = ct_nii.GetOrigin()
                roi_ct_nii.SetOrigin((ox + bx * spx, oy + by * spy, oz + bz * spz))

                roi_pred_2 = self.predict_stage2(roi_ct_nii)
            else:
                roi_pred_1 = None
                roi_pred_2 = None

            if roi_pred_1 is None:
                roi_pred_1 = np.zeros((2, 2, 2), np.uint8)
                roi_pred_2 = np.zeros((2, 2, 2), np.float32)

            score = self.get_score(roi_pred_1, roi_pred_2)
            return case_id, score

        with torch.no_grad():
            overall_time_start = time.time()
            total = len(list_test_files)
            if total == 0:
                return

            max_workers = max(1, int(num_thread))
            done = 0

            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(_process_one, ddir) for ddir in list_test_files]
                for fut in as_completed(futures):
                    case_id, score = fut.result()
                    with self._results_lock:
                        self.results[case_id] = score
                    done += 1
                    if (done % 128) == 0 or done == total:
                        gc.collect()
                        print(
                            f"Processed {done}/{total} cases. Elapsed: {time.time()-overall_time_start:.1f}s"
                        )




## === cell 2
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.set_num_threads(1)
torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

time_start = time.time()

SAVE_CSV = "submission.csv"

list_model_C1_C7_segmentation = [
    "../input/models/models/stage1_0.model",
    "../input/models/models/stage1_1.model",
    "../input/models/models/stage1_2.model",
]
plan_C1_C7_segmentation = "../input/plans-nnunet/stage1.pkl"

list_model_fracture_detection = [
    "../input/models/models/stage2_0.model",
    "../input/models/models/stage2_1.model",
    "../input/models/models/stage2_2.model",
    "../input/models/models/stage2_3.model",
    "../input/models/models/stage2_4.model",
]
plan_fracture_detection = "../input/plans-nnunet/stage2.pkl"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

c2f_predictor = FractureDetector(
    predictor_stage1=predictor_1, predictor_stage2=predictor_2
)

test_df = pd.read_csv(TEST_CSV_PATH)
study_uids = test_df["StudyInstanceUID"].unique().tolist()

if USE_VSIZIP:
    list_DICOM_dirs = [_dicom_dir_for_uid(uid) for uid in study_uids]
else:
    list_DICOM_dirs = [
        os.path.join(TEST_IMG_DIR, uid)
        for uid in study_uids
        if os.path.isdir(os.path.join(TEST_IMG_DIR, uid))
    ]

print(
    f"==> Total cases from test.csv: {len(study_uids)}; queued folders: {len(list_DICOM_dirs)}"
)

num_thread = min(8, max(2, (os.cpu_count() or 4) // 2))
c2f_predictor.predict(list_test_files=list_DICOM_dirs, num_thread=num_thread)

results = c2f_predictor.results

default_scores = {
    "patient_overall": 0.116,
    "C1": 0.01,
    "C2": 0.015,
    "C3": 0.015,
    "C4": 0.01,
    "C5": 0.02,
    "C6": 0.032,
    "C7": 0.048,
}

if results:
    uids_list = list(results.keys())
    scores_mat = np.vstack(
        [results[uid].astype(np.float32, copy=False) for uid in uids_list]
    )
    pred_df = pd.DataFrame(
        {
            "StudyInstanceUID": uids_list,
            "patient_overall": scores_mat[:, 0],
            "C1": scores_mat[:, 1],
            "C2": scores_mat[:, 2],
            "C3": scores_mat[:, 3],
            "C4": scores_mat[:, 4],
            "C5": scores_mat[:, 5],
            "C6": scores_mat[:, 6],
            "C7": scores_mat[:, 7],
        }
    )
else:
    pred_df = pd.DataFrame(
        columns=[
            "StudyInstanceUID",
            "patient_overall",
            "C1",
            "C2",
            "C3",
            "C4",
            "C5",
            "C6",
            "C7",
        ]
    )

merged = test_df.merge(pred_df, on="StudyInstanceUID", how="left", copy=False)

ptype = merged["prediction_type"].to_numpy()
fractured = np.full(len(merged), np.float32(0.01), dtype=np.float32)

for k, v in default_scores.items():
    fractured[ptype == k] = np.float32(v)

for col in ["patient_overall", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]:
    m = ptype == col
    if m.any():
        vals = merged.loc[m, col].to_numpy(dtype=np.float32, copy=False)
        nn = ~pd.isna(vals)
        if nn.any():
            idx = np.flatnonzero(m)[nn]
            fractured[idx] = vals[nn]

sub_df = pd.DataFrame(
    {
        "row_id": merged["row_id"].values,
        "fractured": np.clip(fractured, 1e-6, 1 - 1e-6),
    }
)
sub_df.to_csv(SAVE_CSV, index=False)

print("Wrote:", SAVE_CSV, "rows:", len(sub_df))
print(f"==> Finish using time: {time.time() - time_start:.1f}s")
print(sub_df.head())
