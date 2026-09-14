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

# 5. Target score

0.285187744253774

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import time
import sys
import numpy as np
import pandas as pd
import torch
import SimpleITK as sitk

_DEFAULT_CPU_THREADS = int(os.environ.get("KAGGLE_CPU_THREADS", "4"))
_DEFAULT_CPU_THREADS = max(1, min(_DEFAULT_CPU_THREADS, (os.cpu_count() or 4)))

os.environ.setdefault("OMP_NUM_THREADS", str(_DEFAULT_CPU_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(_DEFAULT_CPU_THREADS))
os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", str(_DEFAULT_CPU_THREADS))

try:
    torch.set_num_threads(_DEFAULT_CPU_THREADS)
except Exception:
    pass

try:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
except Exception:
    pass

try:
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(_DEFAULT_CPU_THREADS)
except Exception:
    pass


def try_recursive_mkdir(dir_path: str):
    os.makedirs(dir_path, exist_ok=True)


def get_bbox(mask: np.ndarray):
    """Return bbox as (bz, ez, by, ey, bx, ex) for a 3D mask (z,y,x)."""
    if mask is None or mask.size == 0:
        return None
    if not np.any(mask):
        return None

    z_any = np.any(mask, axis=(1, 2))
    y_any = np.any(mask, axis=(0, 2))
    x_any = np.any(mask, axis=(0, 1))

    bz = int(np.argmax(z_any))
    ez = int(len(z_any) - 1 - np.argmax(z_any[::-1]))
    by = int(np.argmax(y_any))
    ey = int(len(y_any) - 1 - np.argmax(y_any[::-1]))
    bx = int(np.argmax(x_any))
    ex = int(len(x_any) - 1 - np.argmax(x_any[::-1]))
    return bz, ez, by, ey, bx, ex


def extend_bbox(
    bbox,
    max_shape,
    list_extend_length=(5.0, 5.0, 5.0),
    spacing=(1.0, 1.0, 1.0),
    approximate_method=np.ceil,
):
    """Extend bbox in mm given spacing (z,y,x) and clamp to max_shape (z,y,x)."""
    bz, ez, by, ey, bx, ex = bbox
    ez_len_mm, ey_len_mm, ex_len_mm = list_extend_length  # assumes (z,y,x)
    sz, sy, sx = spacing
    dz = int(approximate_method(ez_len_mm / max(sz, 1e-6)))
    dy = int(approximate_method(ey_len_mm / max(sy, 1e-6)))
    dx = int(approximate_method(ex_len_mm / max(sx, 1e-6)))

    bz2 = max(0, bz - dz)
    ez2 = min(max_shape[0] - 1, ez + dz)
    by2 = max(0, by - dy)
    ey2 = min(max_shape[1] - 1, ey + dy)
    bx2 = max(0, bx - dx)
    ex2 = min(max_shape[2] - 1, ex + dx)
    return int(bz2), int(ez2), int(by2), int(ey2), int(bx2), int(ex2)


def _largest_cc_mask_3d_6(mask: np.ndarray) -> np.ndarray:
    """Return boolean mask of the largest 6-connected component in a 3D boolean array (z,y,x)."""
    if mask is None or mask.size == 0:
        return np.zeros_like(mask, dtype=bool)
    if mask.dtype != np.bool_:
        mask = mask.astype(bool, copy=False)
    if not mask.any():
        return mask

    img = sitk.GetImageFromArray(mask.astype(np.uint8, copy=False))
    cc = sitk.ConnectedComponent(img, fullyConnected=False)  # 6-connected in 3D
    relabeled = sitk.RelabelComponent(cc, sortByObjectSize=True)
    largest = sitk.Equal(relabeled, 1)
    return sitk.GetArrayFromImage(largest).astype(bool, copy=False)


def keep_largest_cervical_cc(pred: np.ndarray, spacing_zyx=None):
    """
    Keep the largest connected component among labels 1..7, output unchanged labels.
    """
    mask = (pred >= 1) & (pred <= 7)
    if not np.any(mask):
        return pred
    keep = _largest_cc_mask_3d_6(mask)
    out = pred.copy()
    out[mask & (~keep)] = 0
    return out


def get_nii_info(img: sitk.Image):
    return {
        "spacing": tuple(img.GetSpacing()),
        "origin": tuple(img.GetOrigin()),
        "size": tuple(img.GetSize()),
        "direction": tuple(img.GetDirection()),
    }


def copy_nii_info(src: sitk.Image, dst: sitk.Image):
    dst.SetSpacing(src.GetSpacing())
    dst.SetOrigin(src.GetOrigin())
    dst.SetDirection(src.GetDirection())
    return dst


_RESAMPLER_CACHE = {}


def resample(
    img: sitk.Image,
    new_spacing,
    new_origin,
    new_size,
    new_direction,
    center_origin=None,
    interp=sitk.sitkLinear,
    dtype=sitk.sitkFloat32,
    constant_value=0,
):
    """SimpleITK resample to explicit geometry."""
    key = (
        tuple(new_spacing),
        tuple(new_origin),
        tuple(int(x) for x in new_size),
        tuple(new_direction),
        int(interp),
        int(dtype),
        float(constant_value),
    )
    resampler = _RESAMPLER_CACHE.get(key)
    if resampler is None:
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(tuple(new_spacing))
        resampler.SetOutputOrigin(tuple(new_origin))
        resampler.SetSize([int(x) for x in new_size])
        resampler.SetOutputDirection(tuple(new_direction))
        resampler.SetInterpolator(interp)
        resampler.SetDefaultPixelValue(constant_value)
        _RESAMPLER_CACHE[key] = resampler
    out = resampler.Execute(sitk.Cast(img, dtype))
    return out


_DICOM_SERIES_FILES_CACHE = {}
_DICOM_SERIES_READER_CACHE = {}


def read_from_DICOM_dir(dicom_dir: str):
    """
    Read a DICOM series from a directory into a 3D SimpleITK image.
    """
    series_file_names = _DICOM_SERIES_FILES_CACHE.get(dicom_dir)
    if series_file_names is None:
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dicom_dir)
        if not series_ids:
            raise FileNotFoundError(f"No DICOM series found in: {dicom_dir}")
        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
            dicom_dir, series_ids[0]
        )
        _DICOM_SERIES_FILES_CACHE[dicom_dir] = series_file_names

    reader = _DICOM_SERIES_READER_CACHE.get(dicom_dir)
    if reader is None:
        reader = sitk.ImageSeriesReader()
        reader.MetaDataDictionaryArrayUpdateOff()
        reader.LoadPrivateTagsOff()
        reader.SetFileNames(series_file_names)
        _DICOM_SERIES_READER_CACHE[dicom_dir] = reader
    else:
        reader.SetFileNames(series_file_names)

    return reader.Execute()


class NNUnetCTPredictor:
    """
    Fix: external NNUnetCTPredictor code/models are missing.
    Provide a minimal stub with identical method names so the pipeline runs end-to-end.
    This stub returns deterministic, conservative predictions (not optimized for score).
    """

    is_fallback_stub = True

    def __init__(
        self,
        list_model_pth,
        plan_file,
        plan_stage,
        device,
        use_gaussian_for_sliding_window,
        patch_size,
        stride,
        tta,
        tta_flip_axis,
        resampling_tolerance,
        resampling_mode,
        resampling_dtype,
        resampling_constance_value,
        remove_air_CT,
        save_dtype=np.float32,
    ):
        self.device = device
        self.resampling_mode = resampling_mode
        self.resampling_dtype = resampling_dtype
        self.resampling_constance_value = resampling_constance_value
        self.remove_air_CT = remove_air_CT
        self.save_dtype = save_dtype

    def resampling(self, ct_nii: sitk.Image):
        return ct_nii

    def pre_processing(self, image: np.ndarray):
        x = image.astype(np.float32)
        if self.remove_air_CT:
            x = np.clip(x, -1024, 2000)
        m = np.mean(x)
        s = np.std(x) + 1e-6
        return (x - m) / s

    def sliding_window_inference(self, image: np.ndarray):
        z, y, x = image.shape[1:]
        if self.remove_air_CT:
            out = np.zeros((8, z, y, x), dtype=np.float32)
            out[0, ...] = 1.0
            return out
        else:
            out = np.zeros((2, z, y, x), dtype=np.float32)
            out[0, ...] = 1.0
            out[1, ...] = 0.0
            return out


print("==> Local fallback imports ready")




## === cell 1
class FractureDetector:
    def __init__(self, predictor_stage1, predictor_stage2, extend_roi=(5.0, 5.0, 5.0)):
        self.predictor_stage1 = predictor_stage1
        self.predictor_stage2 = predictor_stage2
        self.extend_roi = extend_roi

        self.params = {
            "alpha": np.array(
                [0.055, 0.044, 0.052, 0.05, 0.07, 0.077, 0.09, 0.024], dtype=np.float32
            ),
            "beta": np.array(
                [0.475, 0.34, 0.37, 0.38, 0.31, 0.35, 0.38, 0.36], dtype=np.float32
            ),
            "min_score": np.array(
                [0.116, 0.075, 0.015, 0.015, 0.01, 0.02, 0.032, 0.048], dtype=np.float32
            ),
            "max_score": np.array(
                [0.99, 0.999, 0.993, 0.99, 1.0, 0.943, 0.997, 0.999], dtype=np.float32
            ),
        }

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

    def predict_stage1(self, ct_nii, ct_array_zyx=None):
        ct_nii_rs = self.predictor_stage1.resampling(ct_nii)
        same_geom = ct_nii_rs is ct_nii or (
            ct_nii_rs.GetSpacing() == ct_nii.GetSpacing()
            and ct_nii_rs.GetOrigin() == ct_nii.GetOrigin()
            and ct_nii_rs.GetDirection() == ct_nii.GetDirection()
            and ct_nii_rs.GetSize() == ct_nii.GetSize()
        )

        if same_geom:
            if ct_array_zyx is None:
                ct_array_zyx = sitk.GetArrayFromImage(ct_nii)
            image = ct_array_zyx[np.newaxis]
            image = self.predictor_stage1.pre_processing(image)
            pred = self.predictor_stage1.sliding_window_inference(image)
            pred = np.argmax(pred, axis=0)
            pred = keep_largest_cervical_cc(pred, ct_nii.GetSpacing()[::-1])
            return pred.astype(np.uint8, copy=False)

        ori_nii_info = get_nii_info(ct_nii)
        ct_nii = ct_nii_rs
        image = sitk.GetArrayFromImage(ct_nii)[np.newaxis]
        image = self.predictor_stage1.pre_processing(image)
        pred = self.predictor_stage1.sliding_window_inference(image)
        pred = np.argmax(pred, axis=0)
        pred = keep_largest_cervical_cc(pred, ct_nii.GetSpacing()[::-1])
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
        return pred

    def predict_stage2(self, ct_nii, ct_array_zyx=None):
        if ct_array_zyx is not None:
            image = ct_array_zyx[np.newaxis]
            image = self.predictor_stage2.pre_processing(image)
            pred = self.predictor_stage2.sliding_window_inference(image)
            return pred[1].astype(np.float32, copy=False)

        ct_nii_rs = self.predictor_stage2.resampling(ct_nii)
        same_geom = ct_nii_rs is ct_nii or (
            ct_nii_rs.GetSpacing() == ct_nii.GetSpacing()
            and ct_nii_rs.GetOrigin() == ct_nii.GetOrigin()
            and ct_nii_rs.GetDirection() == ct_nii.GetDirection()
            and ct_nii_rs.GetSize() == ct_nii.GetSize()
        )

        if same_geom:
            ct_array_zyx = sitk.GetArrayFromImage(ct_nii)
            image = ct_array_zyx[np.newaxis]
            image = self.predictor_stage2.pre_processing(image)
            pred = self.predictor_stage2.sliding_window_inference(image)
            return pred[1].astype(np.float32, copy=False)

        ori_nii_info = get_nii_info(ct_nii)
        ct_nii = ct_nii_rs
        image = sitk.GetArrayFromImage(ct_nii)[np.newaxis]
        image = self.predictor_stage2.pre_processing(image)
        pred = self.predictor_stage2.sliding_window_inference(image)
        pred = pred[1]  # 0 for background, 1 for foreground
        pred_nii = sitk.GetImageFromArray(pred.astype(np.float32))
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
        return pred

    def get_score(self, pred_c1_c7, pred_fracture):
        output = self.params["min_score"].copy()  # Overall, C1-C7 default mins

        if (pred_c1_c7 is None) or (pred_fracture is None):
            return output

        pred_c1_c7 = pred_c1_c7.copy()
        pred_c1_c7[pred_c1_c7 > 7] = 0

        cerv_mask = pred_c1_c7 > 0

        alpha0 = float(self.params["alpha"][0])
        roi0 = pred_fracture[(pred_fracture >= alpha0) & cerv_mask]
        if roi0.size != 0:
            v0 = np.percentile(roi0, 100.0 * float(self.params["beta"][0]))
            output[0] = max(
                float(self.params["min_score"][0]),
                min(float(self.params["max_score"][0]), float(v0)),
            )

        for C_i in range(1, 8):
            roi = pred_fracture[
                (pred_fracture >= float(self.params["alpha"][C_i]))
                & (pred_c1_c7 == C_i)
            ]
            if roi.size != 0:
                v = np.percentile(roi, 100.0 * float(self.params["beta"][C_i]))
                output[C_i] = max(
                    float(self.params["min_score"][C_i]),
                    min(float(self.params["max_score"][C_i]), float(v)),
                )

        return output.astype(np.float32, copy=False)

    def predict(self, list_test_files):
        count = 0
        overall_time_start = time.time()
        n_cases = len(list_test_files)

        if getattr(self.predictor_stage1, "is_fallback_stub", False):
            default_score = self.params["min_score"].astype(np.float32, copy=True)
            for dicom_dir in list_test_files:
                case_id = os.path.basename(dicom_dir).split(".nii")[0]
                self.results[case_id] = default_score
                count += 1
                if count % 250 == 0 or count == 1:
                    print(f"==> Predicting {count}/{n_cases}: {case_id} (fast-path)")
            print(
                f"==> Finished inference on {n_cases} cases in {time.time() - overall_time_start:.1f}s (fast-path)"
            )
            return

        for dicom_dir in list_test_files:
            case_id = os.path.basename(dicom_dir).split(".nii")[0]

            count += 1
            if count % 25 == 0 or count == 1:
                print(f"==> Predicting {count}/{n_cases}: {case_id}")

            ct_nii = read_from_DICOM_dir(dicom_dir)
            spacing_zyx = ct_nii.GetSpacing()[::-1]

            ct_array_zyx = sitk.GetArrayFromImage(ct_nii)

            pred_1 = self.predict_stage1(ct_nii, ct_array_zyx=ct_array_zyx)
            c1_c7_bbox = self.get_c1_c7_bbox(pred_1, spacing_zyx)

            if c1_c7_bbox is not None:
                bz, ez, by, ey, bx, ex = c1_c7_bbox
                roi_ct_array = ct_array_zyx[bz : ez + 1, by : ey + 1, bx : ex + 1]
                roi_pred_1 = pred_1[bz : ez + 1, by : ey + 1, bx : ex + 1]
                roi_pred_2 = self.predict_stage2(ct_nii=None, ct_array_zyx=roi_ct_array)
            else:
                roi_pred_1 = None
                roi_pred_2 = None

            score = self.get_score(roi_pred_1, roi_pred_2)
            self.results[case_id] = score

        print(
            f"==> Finished inference on {n_cases} cases in {time.time() - overall_time_start:.1f}s"
        )




## === cell 2
time_start = time.time()

DATA_DIR = "../input/rsna-2022-cervical-spine-fracture-detection/test_images"
TEST_CSV = "../input/rsna-2022-cervical-spine-fracture-detection/test.csv"
SAVE_CSV = "submission.csv"

test_df = pd.read_csv(TEST_CSV)
study_ids = test_df["StudyInstanceUID"].unique()

_data_dir = DATA_DIR
_join = os.path.join
_isdir = os.path.isdir
list_DICOM_dirs = []
filtered_study_ids = []
for sid in study_ids:
    d = _join(_data_dir, sid)
    if _isdir(d):
        filtered_study_ids.append(sid)
        list_DICOM_dirs.append(d)

print(
    f"==> Will run inference on {len(list_DICOM_dirs)} studies (from test.csv: {test_df['StudyInstanceUID'].nunique()})"
)

with torch.no_grad():
    list_model_C1_C7_segmentation = ["../input/models/models/stage1_0.model"]
    plan_C1_C7_segmentation = "../input/plans-nnunet/stage1.pkl"

    list_model_fracture_detection = ["../input/models/models/stage2_0.model"]
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
        tta=True,
        tta_flip_axis=(4,),
        resampling_tolerance=0.01,
        resampling_mode=sitk.sitkNearestNeighbor,
        resampling_dtype=sitk.sitkInt16,
        resampling_constance_value=-1024,
        remove_air_CT=True,
    )
    predictor_2 = NNUnetCTPredictor(
        list_model_pth=list_model_fracture_detection,
        plan_file=plan_fracture_detection,
        plan_stage=-1,
        device=device,
        use_gaussian_for_sliding_window=True,
        patch_size=None,
        stride=None,
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

    c2f_predictor.predict(list_test_files=list_DICOM_dirs)
    results = c2f_predictor.results

default_scores = {
    "patient_overall": 0.116,  # matches min_score[0]
    "C1": 0.075,
    "C2": 0.015,
    "C3": 0.015,
    "C4": 0.01,
    "C5": 0.02,
    "C6": 0.032,
    "C7": 0.048,
}

pred_type_arr = test_df["prediction_type"].to_numpy()
sid_arr = test_df["StudyInstanceUID"].to_numpy()
row_id_arr = test_df["row_id"].to_numpy()
n = len(test_df)

col = np.empty(n, dtype=np.int32)
is_po = pred_type_arr == "patient_overall"
col[is_po] = 0
if (~is_po).any():
    col[~is_po] = np.char.str_len(pred_type_arr[~is_po])  # dummy to allocate via ufunc
    col[~is_po] = pred_type_arr[~is_po].view("U16").astype("U16")  # no-op, stable dtype
    col[~is_po] = (np.char.array(pred_type_arr[~is_po]).slice(1, None)).astype(np.int32)

default_by_col = np.array(
    [0.116, 0.075, 0.015, 0.015, 0.01, 0.02, 0.032, 0.048], dtype=np.float32
)
vals = default_by_col[col].copy()

if results:
    sids_unique = np.array(list(results.keys()), dtype=object)
    scores_mat = np.stack([results[sid] for sid in sids_unique], axis=0).astype(
        np.float32, copy=False
    )  # (n_studies, 8)
    sid_to_idx = {sid: i for i, sid in enumerate(sids_unique)}
    idx = np.fromiter((sid_to_idx.get(s, -1) for s in sid_arr), count=n, dtype=np.int32)
    present = idx >= 0
    if present.any():
        vals[present] = scores_mat[idx[present], col[present]]

eps = 1e-6
vals = np.clip(vals, eps, 1.0 - eps).astype(np.float32, copy=False)

sub = pd.DataFrame({"row_id": row_id_arr, "fractured": vals})
sub.to_csv(SAVE_CSV, index=False)

print(sub.head(10))
print(f"==> Wrote {SAVE_CSV} with shape {sub.shape} in {time.time() - time_start:.1f}s")
print(f"==> File exists: {os.path.exists(SAVE_CSV)}")

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_56/1221355716.py in <cell line: 0>()
     97 if (~is_po).any():
     98     # pred_type_arr values are exactly "C1".."C7" for the rest
---> 99     col[~is_po] = np.char.str_len(pred_type_arr[~is_po])  # dummy to allocate via ufunc
    100     col[~is_po] = pred_type_arr[~is_po].view("U16").astype("U16")  # no-op, stable dtype
    101     col[~is_po] = (np.char.array(pred_type_arr[~is_po]).slice(1, None)).astype(np.int32)

/usr/local/lib/python3.11/dist-packages/numpy/core/defchararray.py in str_len(a)
    296     # Generally intp would be expected for lengths, although int is sufficient
    297     # due to the dtype itemsize limitation.
--> 298     return _vec_string(a, int_, '__len__')
    299 
    300 

TypeError: string operation on non-string array
