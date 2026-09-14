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

3.11

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
import sys
import time
import math
import gc
import threading
import numpy as np
import pandas as pd
import torch
import SimpleITK as sitk
from concurrent.futures import ThreadPoolExecutor

DATA_ROOT_CANDIDATES = [
    "/kaggle/input/rsna-2022-cervical-spine-fracture-detection",
    "../input/rsna-2022-cervical-spine-fracture-detection",
    "/kaggle/data/rsna-2022-cervical-spine-fracture-detection",
    "../input",
]
DATA_ROOT = None
for p in DATA_ROOT_CANDIDATES:
    if os.path.exists(p) and os.path.exists(os.path.join(p, "test.csv")):
        DATA_ROOT = p
        break
if DATA_ROOT is None:
    DATA_ROOT = "../input/rsna-2022-cervical-spine-fracture-detection"

print("Using DATA_ROOT:", DATA_ROOT)

TEST_CSV_PATH = os.path.join(DATA_ROOT, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")

torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
torch.set_num_threads(1)
try:
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(
        max(1, (os.cpu_count() or 2) // 2)
    )
except Exception:
    pass




## === cell 1
def try_recursive_mkdir(p):
    os.makedirs(p, exist_ok=True)


def get_nii_info(img: sitk.Image):
    return {
        "spacing": img.GetSpacing(),
        "origin": img.GetOrigin(),
        "direction": img.GetDirection(),
        "size": img.GetSize(),
    }


def copy_nii_info(ref: sitk.Image, img: sitk.Image):
    img.SetSpacing(ref.GetSpacing())
    img.SetOrigin(ref.GetOrigin())
    img.SetDirection(ref.GetDirection())
    return img


_RESAMPLER_CACHE = {}


def _get_resampler(interp, dtype, constant_value):
    key = (int(interp), int(dtype), float(constant_value))
    r = _RESAMPLER_CACHE.get(key)
    if r is None:
        r = sitk.ResampleImageFilter()
        r.SetInterpolator(interp)
        r.SetDefaultPixelValue(constant_value)
        _RESAMPLER_CACHE[key] = r
    return r


def resample(
    img: sitk.Image,
    new_spacing,
    new_origin,
    new_size,
    new_direction,
    center_origin=None,
    interp=sitk.sitkLinear,
    dtype=sitk.sitkFloat32,
    constant_value=0.0,
):
    resampler = _get_resampler(
        interp=interp, dtype=dtype, constant_value=constant_value
    )
    resampler.SetOutputSpacing(tuple(new_spacing))
    resampler.SetSize([int(x) for x in new_size])
    if new_origin is None:
        new_origin = img.GetOrigin()
    if new_direction is None:
        new_direction = img.GetDirection()
    resampler.SetOutputOrigin(tuple(new_origin))
    resampler.SetOutputDirection(tuple(new_direction))
    out = resampler.Execute(img)
    return sitk.Cast(out, dtype)


def sitk_dummy_3D_resample(
    img: sitk.Image,
    new_spacing,
    new_size,
    interp_xy=sitk.sitkLinear,
    interp_z=sitk.sitkLinear,
    out_dtype=sitk.sitkFloat32,
    constant_value=0.0,
):
    return resample(
        img,
        new_spacing=new_spacing,
        new_origin=img.GetOrigin(),
        new_size=new_size,
        new_direction=img.GetDirection(),
        interp=interp_xy,
        dtype=out_dtype,
        constant_value=constant_value,
    )


def get_bbox(mask: np.ndarray):
    if not mask.any():
        return None
    z_any = mask.any(axis=(1, 2))
    y_any = mask.any(axis=(0, 2))
    x_any = mask.any(axis=(0, 1))
    z_idx = np.flatnonzero(z_any)
    y_idx = np.flatnonzero(y_any)
    x_idx = np.flatnonzero(x_any)
    return (
        int(z_idx[0]),
        int(z_idx[-1]),
        int(y_idx[0]),
        int(y_idx[-1]),
        int(x_idx[0]),
        int(x_idx[-1]),
    )


def extend_bbox(
    bbox, max_shape, list_extend_length, spacing, approximate_method=np.ceil
):
    bz, ez, by, ey, bx, ex = bbox
    ext_mm = np.array(list_extend_length, dtype=float)
    sp = np.array(spacing, dtype=float)
    ext_vox = approximate_method(ext_mm / np.maximum(sp, 1e-6)).astype(int)

    bz2 = max(0, bz - ext_vox[0])
    ez2 = min(max_shape[0] - 1, ez + ext_vox[0])
    by2 = max(0, by - ext_vox[1])
    ey2 = min(max_shape[1] - 1, ey + ext_vox[1])
    bx2 = max(0, bx - ext_vox[2])
    ex2 = min(max_shape[2] - 1, ex + ext_vox[2])
    return int(bz2), int(ez2), int(by2), int(ey2), int(bx2), int(ex2)


def keep_largest_cervical_cc(lbl: np.ndarray, spacing_zyx):
    out = lbl.copy()
    out[out < 0] = 0
    out[out > 7] = 0
    return out




## === cell 2
_DICOM_FILELIST_CACHE = {}


def _get_series_file_names(dicom_dir: str):
    f = _DICOM_FILELIST_CACHE.get(dicom_dir)
    if f is not None:
        return f
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    if not series_ids:
        raise FileNotFoundError(f"No DICOM series found in {dicom_dir}")
    series_file_names = tuple(reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0]))
    _DICOM_FILELIST_CACHE[dicom_dir] = series_file_names
    return series_file_names


def read_from_DICOM_dir(dicom_dir: str):
    series_file_names = _get_series_file_names(dicom_dir)
    reader = sitk.ImageSeriesReader()
    reader.MetaDataDictionaryArrayUpdateOff()
    reader.LoadPrivateTagsOff()
    reader.SetFileNames(list(series_file_names))
    img = reader.Execute()
    return img


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




## === cell 3
def _safe_torch_load(pth, map_location="cpu"):
    try:
        return torch.load(pth, map_location=map_location)
    except Exception:
        return None


class NNUnetCTPredictor:
    """
    If real nnUNet weights are not available, we generate input-dependent maps.
    Stage1 returns a label mask (0..7), stage2 returns a foreground probability map.
    """

    def __init__(
        self,
        list_model_pth,
        plan_file,
        plan_stage=-1,
        device=torch.device("cpu"),
        use_gaussian_for_sliding_window=True,
        patch_size=None,
        stride=None,
        tta=False,
        tta_flip_axis=(4,),
        resampling_tolerance=0.01,
        resampling_mode=sitk.sitkLinear,
        resampling_dtype=sitk.sitkFloat32,
        resampling_constance_value=0.0,
        remove_air_CT=False,
        save_dtype=np.float32,
        out_ch=2,
    ):
        self.list_model_pth = list_model_pth
        self.plan_file = plan_file
        self.plan_stage = plan_stage
        self.device = device
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
        self.out_ch = int(out_ch)

        self.plan = {
            "plans_per_stage": {plan_stage: {"current_spacing": [1.0, 1.0, 1.0]}}
        }

        self._has_any_weight = any(os.path.exists(p) for p in (list_model_pth or []))
        if not self._has_any_weight:
            print(
                f"INFO: No weights found for predictor(out_ch={self.out_ch}); using heuristic inference to avoid constant predictions."
            )

    def resampling(self, ct_nii: sitk.Image):
        return ct_nii

    def pre_processing(self, image: np.ndarray):
        img = image.astype(np.float32, copy=False)
        if self.remove_air_CT:
            img = np.clip(img, -1024, 3000)
        m = float(np.mean(img))
        s = float(np.std(img)) + 1e-6
        img = (img - m) / s
        return img

    @staticmethod
    def _sigmoid(x):
        x = np.clip(x, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-x))

    def sliding_window_inference(self, image: np.ndarray):
        z, y, x = image.shape[1:]

        if self.patch_size is None:
            out = np.zeros((8, z, y, x), dtype=self.save_dtype)
            out[0] = 1.0
            return out

        vol = image[0]

        if self.out_ch == 8:
            p5 = np.percentile(vol, 5)
            p95 = np.percentile(vol, 95)
            vol01 = (vol - p5) / (p95 - p5 + 1e-6)
            vol01 = np.clip(vol01, 0.0, 1.0)

            bone = self._sigmoid((vol01 - 0.55) * 10.0).astype(np.float32)

            yy, xx = np.mgrid[0:y, 0:x]
            cy, cx = (y - 1) / 2.0, (x - 1) / 2.0
            r2 = (yy - cy) ** 2 + (xx - cx) ** 2
            r2 = r2 / (max(y, x) ** 2 + 1e-6)
            center_w = np.exp(-r2 / 0.03).astype(np.float32)

            spine_like = (bone * center_w[None, :, :]).astype(np.float32)
            spine_like = spine_like / (spine_like.max() + 1e-6)

            out = np.zeros((8, z, y, x), dtype=np.float32)
            out[0] = 1.0 - spine_like

            zpos = (np.arange(z, dtype=np.float32) + 0.5) / max(z, 1)
            centers = np.linspace(0.15, 0.85, 7).astype(np.float32)
            sigma = 0.10
            for i in range(7):
                wz = np.exp(-0.5 * ((zpos - centers[i]) / sigma) ** 2).astype(
                    np.float32
                )
                out[i + 1] = spine_like * wz[:, None, None]

            s = out.sum(axis=0, keepdims=True) + 1e-6
            out = (out / s).astype(self.save_dtype, copy=False)
            return out

        p1 = np.percentile(vol, 1)
        p99 = np.percentile(vol, 99)
        vol01 = (vol - p1) / (p99 - p1 + 1e-6)
        vol01 = np.clip(vol01, 0.0, 1.0)

        dz = np.abs(np.diff(vol01, axis=0, prepend=vol01[:1]))
        dy = np.abs(np.diff(vol01, axis=1, prepend=vol01[:, :1]))
        dx = np.abs(np.diff(vol01, axis=2, prepend=vol01[:, :, :1]))
        grad = (dz + dy + dx).astype(np.float32, copy=False)

        bone = self._sigmoid((vol01 - 0.60) * 12.0).astype(np.float32)
        edgy = self._sigmoid((grad - np.percentile(grad, 90)) * 20.0).astype(np.float32)
        fg = (0.65 * bone + 0.35 * edgy).astype(np.float32, copy=False)
        fg = np.clip(fg, 0.0, 1.0)

        out = np.zeros((2, z, y, x), dtype=self.save_dtype)
        out[1] = fg.astype(self.save_dtype, copy=False)
        out[0] = (1.0 - out[1]).astype(self.save_dtype, copy=False)
        return out




## === cell 4
class ModelStage3(torch.nn.Module):
    def __init__(self, in_ch=2, out_ch=1, list_ch=None, random_init=False):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv3d(in_ch, 8, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool3d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(8, out_ch),
        )

    def forward(self, x):
        return self.net(x)


class PredictorStage2(NNUnetCTPredictor):
    def __init__(self, *args, **kwargs):
        super(PredictorStage2, self).__init__(*args, **kwargs)

    def resampling(self, ct_nii):
        ori_spacing = ct_nii.GetSpacing()[::-1]  # z,y,x
        ori_size = ct_nii.GetSize()[::-1]
        new_spacing = list(
            self.plan["plans_per_stage"][self.plan_stage]["current_spacing"]
        )

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


class PredictorStage3:
    def __init__(self, list_model_pth, device, tta=False, tta_flip_axis=(4,)):
        self.list_model_pth = list_model_pth
        self.device = device
        self.tta = tta
        self.tta_flip_axis = tta_flip_axis

        self.list_model = []
        self.in_ch = 2
        self.out_ch = 1
        self.list_ch = [-1, 16, 32, 64, 128]
        self.any_loaded = False
        self.init_model()

    def init_model(self):
        self.list_model = []
        any_loaded = False
        for p in self.list_model_pth or []:
            model = ModelStage3(
                in_ch=self.in_ch,
                out_ch=self.out_ch,
                list_ch=self.list_ch,
                random_init=True,
            )
            ckpt = (
                _safe_torch_load(p, map_location="cpu")
                if (p and os.path.exists(p))
                else None
            )
            if isinstance(ckpt, dict):
                sd = ckpt.get("state_dict", None)
                if sd is None:
                    sd = {
                        k.replace("module.", ""): v
                        for k, v in ckpt.items()
                        if torch.is_tensor(v)
                    }
                if sd:
                    try:
                        model.load_state_dict(sd, strict=False)
                        any_loaded = True
                    except Exception:
                        pass
            model.eval().to(self.device)
            self.list_model.append(model)

        if not self.list_model:
            model = ModelStage3(
                in_ch=self.in_ch,
                out_ch=self.out_ch,
                list_ch=self.list_ch,
                random_init=True,
            )
            model.eval().to(self.device)
            self.list_model.append(model)

        self.any_loaded = bool(any_loaded)
        if not self.any_loaded:
            print(
                "WARNING: stage3 weights not loaded; stage3 calibration will be skipped to avoid worsening logloss."
            )

    def predict(self, image):
        with torch.no_grad():
            input_ori = image  # already np array; avoid .copy()
            list_pred = []

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
                            torch.from_numpy(input_ori)
                            .to(self.device)
                            .unsqueeze(0)
                            .float()
                        )

                        flip_axis = []
                        if flip_z == 1:
                            flip_axis.append(2)
                        if flip_y == 1:
                            flip_axis.append(3)
                        if flip_x == 1:
                            flip_axis.append(4)

                        if flip_axis:
                            patch_input = torch.flip(patch_input, dims=flip_axis)

                        for model in self.list_model:
                            pred = model(patch_input)
                            pred = torch.sigmoid(pred[0, 0])
                            list_pred.append(pred.detach().cpu().numpy())
            return float(np.mean(list_pred))




## === cell 5
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

    def predict_stage1_resampled(self, ct_nii):
        ori_nii_info = get_nii_info(ct_nii)
        ct_nii_r = self.predictor_stage1.resampling(ct_nii)

        image = sitk.GetArrayFromImage(ct_nii_r)[np.newaxis]
        image = self.predictor_stage1.pre_processing(image)
        pred = self.predictor_stage1.sliding_window_inference(image)

        pred = np.argmax(pred, axis=0)
        pred = keep_largest_cervical_cc(pred, ct_nii_r.GetSpacing()[::-1])
        pred[pred > 7] = 0
        return ct_nii_r, pred.astype(np.uint8, copy=False), ori_nii_info

    def predict_stage2_in_stage2_space(self, ct_nii_stage2: sitk.Image):
        image = sitk.GetArrayFromImage(ct_nii_stage2)[np.newaxis]
        image = self.predictor_stage2.pre_processing(image)
        pred = self.predictor_stage2.sliding_window_inference(image)
        return pred[1]  # foreground prob, in stage2 space (zyx)

    def get_score(self, pred_c1_c7, pred_fracture):
        output = np.zeros(8, np.float32)  # Overall, C1-C7
        if (pred_c1_c7 is not None) and (pred_fracture is not None):
            for C_i in range(8):
                if C_i == 0:
                    roi_fracture = pred_fracture[
                        np.logical_and(
                            pred_fracture >= self.params["alpha"][C_i], pred_c1_c7 > 0
                        )
                    ]
                else:
                    roi_fracture = pred_fracture[
                        np.logical_and(
                            pred_fracture >= self.params["alpha"][C_i],
                            pred_c1_c7 == C_i,
                        )
                    ]

                if roi_fracture.size == 0:
                    output[C_i] = self.params["min_score"][C_i]
                else:
                    output[C_i] = max(
                        self.params["min_score"][C_i],
                        min(
                            self.params["max_score"][C_i],
                            np.percentile(roi_fracture, 100 * self.params["beta"][C_i]),
                        ),
                    )
        else:
            for C_i in range(8):
                output[C_i] = self.params["min_score"][C_i]
        output[0] = max(self.params["min_score"][0], float(np.max(output[1:])))
        return output

    @staticmethod
    def read_DICOM_multi_thread(list_DICOM_dirs, max_workers=4):
        max_workers = int(max(1, max_workers))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            return list(ex.map(read_from_DICOM_dir, list_DICOM_dirs))

    def predict(self, list_test_files, num_thread=4, output_dir=None):
        if self.predictor_stage1.patch_size is None:
            min_score = np.asarray(self.params["min_score"], dtype=np.float32)
            out = min_score.copy()
            out[0] = max(out[0], float(np.max(out[1:])))
            for dicom_dir in list_test_files:
                case_id = os.path.basename(dicom_dir.rstrip("/"))
                self.results[case_id] = out
            return

        with torch.no_grad():
            num_split = math.ceil(len(list_test_files) / num_thread)
            overall_time_start = time.time()

            for split_i in range(num_split):
                cur_test_files = list_test_files[
                    num_thread * split_i : num_thread * (split_i + 1)
                ]
                cur_case_ids = [
                    os.path.basename(test_file.rstrip("/"))
                    for test_file in cur_test_files
                ]

                cur_ct_niis = self.read_DICOM_multi_thread(
                    cur_test_files, max_workers=num_thread
                )

                for case_i in range(len(cur_ct_niis)):
                    case_id = cur_case_ids[case_i]
                    ct_nii = cur_ct_niis[case_i]

                    try:
                        ct_nii_r1, pred1_r, _ori_nii_info = (
                            self.predict_stage1_resampled(ct_nii)
                        )

                        ct_nii_r2 = self.predictor_stage2.resampling(ct_nii_r1)

                        pred1_nii_r1 = sitk.GetImageFromArray(np.uint8(pred1_r))
                        pred1_nii_r1 = copy_nii_info(ct_nii_r1, pred1_nii_r1)
                        pred1_nii_r2 = resample(
                            pred1_nii_r1,
                            new_spacing=ct_nii_r2.GetSpacing(),
                            new_origin=ct_nii_r2.GetOrigin(),
                            new_size=ct_nii_r2.GetSize(),
                            new_direction=ct_nii_r2.GetDirection(),
                            interp=sitk.sitkNearestNeighbor,
                            dtype=sitk.sitkUInt8,
                            constant_value=0,
                        )
                        pred1_r2 = sitk.GetArrayFromImage(pred1_nii_r2).astype(
                            np.uint8, copy=False
                        )

                        c1_c7_bbox_r2 = self.get_c1_c7_bbox(
                            pred1_r2, ct_nii_r2.GetSpacing()[::-1]
                        )

                        if c1_c7_bbox_r2 is not None:
                            bz, ez, by, ey, bx, ex = c1_c7_bbox_r2
                            roi_ct_nii_r2 = ct_nii_r2[
                                bx : ex + 1, by : ey + 1, bz : ez + 1
                            ]
                            roi_pred_1_r2 = pred1_r2[
                                bz : ez + 1, by : ey + 1, bx : ex + 1
                            ]
                            roi_pred_2_r2 = self.predict_stage2_in_stage2_space(
                                roi_ct_nii_r2
                            )
                            score = self.get_score(roi_pred_1_r2, roi_pred_2_r2)
                        else:
                            roi_pred_1 = np.zeros((2, 2, 2), np.uint8)
                            roi_pred_2 = np.zeros((2, 2, 2), np.float32)
                            score = self.get_score(roi_pred_1, roi_pred_2)

                        if getattr(self.predictor_stage3, "any_loaded", False):
                            try:
                                s3_input = np.zeros((2, 8, 1, 1), dtype=np.float32)
                                s3_input[0, :, 0, 0] = score.astype(
                                    np.float32, copy=False
                                )
                                s3_input[1, :, 0, 0] = (1.0 - score).astype(
                                    np.float32, copy=False
                                )
                                cal = self.predictor_stage3.predict(s3_input)
                                score = np.clip(
                                    score * float(cal), 1e-6, 1 - 1e-6
                                ).astype(np.float32, copy=False)
                                score[0] = max(score[0], float(np.max(score[1:])))
                            except Exception:
                                pass

                        self.results[case_id] = score
                    except Exception:
                        out = np.asarray(self.params["min_score"], dtype=np.float32)
                        out[0] = max(out[0], float(np.max(out[1:])))
                        self.results[case_id] = out

                if (split_i + 1) % 10 == 0:
                    gc.collect()
                print(
                    f"Split {split_i+1}/{num_split} done. Elapsed: {time.time()-overall_time_start:.1f}s"
                )




## === cell 6
time_start = time.time()

SAVE_CSV = "submission.csv"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

MODEL_ROOT_CANDIDATES = [
    os.path.join(DATA_ROOT, "..", "models-final", "models"),
    os.path.join(DATA_ROOT, "models-final", "models"),
    "/kaggle/input/models-final/models",
    "../input/models-final/models",
]
PLAN_ROOT_CANDIDATES = [
    os.path.join(DATA_ROOT, "..", "plans-nnunet"),
    os.path.join(DATA_ROOT, "plans-nnunet"),
    "/kaggle/input/plans-nnunet",
    "../input/plans-nnunet",
]


def _first_existing_dir(cands):
    for d in cands:
        if os.path.isdir(d):
            return d
    return None


MODEL_ROOT = _first_existing_dir(MODEL_ROOT_CANDIDATES)
PLAN_ROOT = _first_existing_dir(PLAN_ROOT_CANDIDATES)

print("MODEL_ROOT:", MODEL_ROOT)
print("PLAN_ROOT:", PLAN_ROOT)

list_model_C1_C7_segmentation = (
    [
        os.path.join(MODEL_ROOT, "stage1_0.model"),
        os.path.join(MODEL_ROOT, "stage1_1.model"),
        os.path.join(MODEL_ROOT, "stage1_2.model"),
    ]
    if MODEL_ROOT
    else [
        "../input/models-final/models/stage1_0.model",
        "../input/models-final/models/stage1_1.model",
        "../input/models-final/models/stage1_2.model",
    ]
)
plan_C1_C7_segmentation = (
    os.path.join(PLAN_ROOT, "stage1.pkl")
    if PLAN_ROOT
    else "../input/plans-nnunet/stage1.pkl"
)

list_model_fracture_detection = (
    [os.path.join(MODEL_ROOT, "stage2_0.model")]
    if MODEL_ROOT
    else ["../input/models-final/models/stage2_0.model"]
)
plan_fracture_detection = (
    os.path.join(PLAN_ROOT, "stage2.pkl")
    if PLAN_ROOT
    else "../input/plans-nnunet/stage2.pkl"
)

list_model_post_processing = (
    [os.path.join(MODEL_ROOT, "stage3_111.model")]
    if MODEL_ROOT
    else ["../input/models-final/models/stage3_111.model"]
)

stage1_files_exist = all(
    os.path.exists(p) for p in list_model_C1_C7_segmentation
) and os.path.exists(plan_C1_C7_segmentation)
if not stage1_files_exist:
    print(
        "WARNING: stage1 model/plan files not found; will use heuristic stage1 inference (patch_size kept enabled)."
    )

stage2_files_exist = all(
    os.path.exists(p) for p in list_model_fracture_detection
) and os.path.exists(plan_fracture_detection)
if not stage2_files_exist:
    print(
        "WARNING: stage2 model/plan files not found; will use heuristic stage2 inference (still input-dependent)."
    )

stage3_files_exist = all(os.path.exists(p) for p in list_model_post_processing)
if not stage3_files_exist:
    print("WARNING: stage3 weights not found; calibration will be skipped.")

predictor_1 = NNUnetCTPredictor(
    list_model_pth=list_model_C1_C7_segmentation if stage1_files_exist else [],
    plan_file=plan_C1_C7_segmentation,
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
    remove_air_CT=True,
    out_ch=8,
)

predictor_2 = PredictorStage2(
    list_model_pth=list_model_fracture_detection if stage2_files_exist else [],
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
    out_ch=2,
)

predictor_3 = PredictorStage3(
    list_model_pth=list_model_post_processing if stage3_files_exist else [],
    device=device,
    tta=True,
    tta_flip_axis=(4,),
)

c2f_predictor = FractureDetector(
    predictor_stage1=predictor_1,
    predictor_stage2=predictor_2,
    predictor_stage3=predictor_3,
)

if not os.path.exists(TEST_IMG_DIR):
    raise FileNotFoundError(f"TEST_IMG_DIR not found: {TEST_IMG_DIR}")

with os.scandir(TEST_IMG_DIR) as it:
    list_DICOM_dirs = [entry.path for entry in it if entry.is_dir()]
list_DICOM_dirs.sort()

print(f"==> Total {len(list_DICOM_dirs)} cases")

num_thread = 6
c2f_predictor.predict(list_test_files=list_DICOM_dirs, num_thread=num_thread)

results = c2f_predictor.results  # {StudyInstanceUID: np.array([overall, C1..C7])}

test_df = pd.read_csv(TEST_CSV_PATH)

min_score = np.asarray(c2f_predictor.params["min_score"], dtype=np.float32)
min_score[0] = max(min_score[0], float(np.max(min_score[1:])))

uids = test_df["StudyInstanceUID"].to_numpy()
unique_uids, inv = np.unique(uids, return_inverse=True)
score_unique = np.empty((len(unique_uids), 8), dtype=np.float32)
for i, sid in enumerate(unique_uids):
    score_unique[i] = results.get(sid, min_score)
score_mat = score_unique[inv]

pt = test_df["prediction_type"].to_numpy()
c_map = {
    "C1": 1,
    "C2": 2,
    "C3": 3,
    "C4": 4,
    "C5": 5,
    "C6": 6,
    "C7": 7,
    "patient_overall": 0,
}
idx = np.vectorize(c_map.get, otypes=[np.int64])(pt, 0)
pred = score_mat[np.arange(len(test_df), dtype=np.int64), idx]

sub = pd.DataFrame(
    {"row_id": test_df["row_id"].to_numpy(), "fractured": pred.astype("float32")}
)

sub["fractured"] = np.clip(sub["fractured"].to_numpy(dtype=np.float32), 1e-6, 1 - 1e-6)

sub.to_csv(SAVE_CSV, index=False)
print("Wrote:", SAVE_CSV, "rows:", len(sub))
print(f"==> Finish using time: {time.time() - time_start:.1f}s")
