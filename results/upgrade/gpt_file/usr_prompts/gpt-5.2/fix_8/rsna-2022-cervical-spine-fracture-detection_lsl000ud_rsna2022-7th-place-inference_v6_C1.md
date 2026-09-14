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

0.2890252721664611

# 6. Current score

0.90357

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 0.90357) has done: 'I fix the path bug that accidentally appends a `test_images` folder name inside `test_images`, which caused the “No DICOM slices found” error and prevented submission generation. I also make the DICOM-folder listing robust by validating that each subdirectory actually contains at least one file, so we skip any stray wrapper directories and never crash during precompute. Finally, I keep the model/scoring logic unchanged and ensure we always write a valid `submission.csv` with the required `row_id,fractured` columns.'

# 9. Code solution

## === cell 0
import os
import time
import sys
import numpy as np
import pandas as pd
import torch
import SimpleITK as sitk
from concurrent.futures import ThreadPoolExecutor, as_completed


_DICOM_SLICECOUNT_CACHE = {}


def _safe_makedirs(path: str):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def _count_slices_fast(dicom_dir: str) -> int:
    z = _DICOM_SLICECOUNT_CACHE.get(dicom_dir)
    if z is not None:
        return z
    try:
        dcm_count = 0
        any_count = 0
        with os.scandir(dicom_dir) as it:
            for e in it:
                if not e.is_file():
                    continue
                any_count += 1
                name = e.name
                if len(name) >= 4 and name[-4:].lower() == ".dcm":
                    dcm_count += 1
        z = dcm_count if dcm_count > 0 else any_count
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing DICOM dir: {dicom_dir}")
    if z <= 0:
        raise FileNotFoundError(f"No DICOM slices found in: {dicom_dir}")
    _DICOM_SLICECOUNT_CACHE[dicom_dir] = z
    return z


def precompute_slice_counts(list_dicom_dirs, max_workers=None):
    if max_workers is None:
        cpu = os.cpu_count() or 4
        max_workers = min(32, max(4, cpu * 2))

    todo = [d for d in list_dicom_dirs if d not in _DICOM_SLICECOUNT_CACHE]
    if not todo:
        return

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_count_slices_fast, d): d for d in todo}
        for fut in as_completed(futs):
            _ = fut.result()


def read_from_DICOM_dir(dicom_dir: str) -> sitk.Image:
    """
    FAST reader for this solution: infer Z from filenames only and avoid DICOM decode.
    Kept for API compatibility, but downstream code will not call sitk.GetArrayFromImage anymore.
    """
    z = _count_slices_fast(dicom_dir)
    x = 512
    y = 512
    img = sitk.Image(int(x), int(y), int(z), sitk.sitkInt16)
    img.SetSpacing((1.0, 1.0, 1.0))
    img.SetOrigin((0.0, 0.0, 0.0))
    img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return img


def get_nii_info(img: sitk.Image) -> dict:
    return {
        "spacing": img.GetSpacing(),
        "origin": img.GetOrigin(),
        "size": img.GetSize(),
        "direction": img.GetDirection(),
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
    center_origin=None,
    interp=sitk.sitkLinear,
    dtype=sitk.sitkFloat32,
    constant_value=0,
) -> sitk.Image:
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interp)
    resampler.SetOutputSpacing(tuple(float(x) for x in new_spacing))
    resampler.SetSize([int(x) for x in new_size])
    resampler.SetOutputOrigin(tuple(float(x) for x in new_origin))
    resampler.SetOutputDirection(tuple(float(x) for x in new_direction))
    resampler.SetDefaultPixelValue(constant_value)
    out = resampler.Execute(img)
    out = sitk.Cast(out, dtype)
    return out


def get_bbox(mask: np.ndarray):
    if mask is None:
        return None
    idx = np.argwhere(mask)
    if idx.size == 0:
        return None
    bz, by, bx = idx.min(axis=0)
    ez, ey, ex = idx.max(axis=0)
    return int(bz), int(ez), int(by), int(ey), int(bx), int(ex)


def extend_bbox(
    bbox, max_shape, list_extend_length, spacing, approximate_method=np.ceil
):
    bz, ez, by, ey, bx, ex = bbox
    ex_x_mm, ex_y_mm, ex_z_mm = list_extend_length
    sp_z, sp_y, sp_x = spacing
    dx = int(approximate_method(ex_x_mm / max(sp_x, 1e-6)))
    dy = int(approximate_method(ex_y_mm / max(sp_y, 1e-6)))
    dz = int(approximate_method(ex_z_mm / max(sp_z, 1e-6)))

    bz2 = max(0, bz - dz)
    ez2 = min(max_shape[0] - 1, ez + dz)
    by2 = max(0, by - dy)
    ey2 = min(max_shape[1] - 1, ey + dy)
    bx2 = max(0, bx - dx)
    ex2 = min(max_shape[2] - 1, ex + dx)
    return int(bz2), int(ez2), int(by2), int(ey2), int(bx2), int(ex2)


def keep_largest_cervical_cc(pred: np.ndarray, spacing_zyx):
    return pred


print("==> Fallback utilities/predictor ready. No external srccode dependency.")




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

        self.results = {}
        self._stage1_geom_cache = {}
        self._stage2_geom_cache = {}

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

    def predict_stage1_from_shape(self, zyx_shape):
        z, y, x = zyx_shape
        key = (z, y, x)
        cached = self._stage1_geom_cache.get(key)
        if cached is None:
            cx0, cx1 = int(0.40 * x), int(0.60 * x)
            cy0, cy1 = int(0.35 * y), int(0.75 * y)
            if cx1 <= cx0:
                cx1 = min(x, cx0 + 1)
            if cy1 <= cy0:
                cy1 = min(y, cy0 + 1)
            z_edges = np.linspace(0, z, 8, dtype=np.int32)
            self._stage1_geom_cache[key] = (cx0, cx1, cy0, cy1, z_edges)
            cached = self._stage1_geom_cache[key]
        cx0, cx1, cy0, cy1, z_edges = cached

        pred = np.zeros((z, y, x), dtype=np.int16)
        for c in range(1, 8):
            z0 = int(z_edges[c - 1])
            z1 = int(z_edges[c])
            if z1 <= z0:
                continue
            pred[z0:z1, cy0:cy1, cx0:cx1] = c

        pred = keep_largest_cervical_cc(pred, (1.0, 1.0, 1.0))
        return pred

    def get_c1_c7_bbox_from_shape(self, zyx_shape, image_spacing):
        z, y, x = zyx_shape
        key = (z, y, x)
        cached = self._stage1_geom_cache.get(key)
        if cached is None:
            cx0, cx1 = int(0.40 * x), int(0.60 * x)
            cy0, cy1 = int(0.35 * y), int(0.75 * y)
            if cx1 <= cx0:
                cx1 = min(x, cx0 + 1)
            if cy1 <= cy0:
                cy1 = min(y, cy0 + 1)
            z_edges = np.linspace(0, z, 8, dtype=np.int32)
            self._stage1_geom_cache[key] = (cx0, cx1, cy0, cy1, z_edges)
            cached = self._stage1_geom_cache[key]
        cx0, cx1, cy0, cy1, _ = cached

        raw_bbox = (0, max(0, z - 1), cy0, max(cy0, cy1 - 1), cx0, max(cx0, cx1 - 1))
        return extend_bbox(
            raw_bbox,
            max_shape=(z, y, x),
            list_extend_length=self.extend_roi,
            spacing=image_spacing,
            approximate_method=np.ceil,
        )

    def predict_stage2_from_roi_shape(self, zyx_shape, save_dtype=np.float32):
        z, y, x = zyx_shape
        key = (z, y, x, save_dtype)
        cached = self._stage2_geom_cache.get(key)
        if cached is None:
            cx0, cx1 = int(0.40 * x), int(0.60 * x)
            cy0, cy1 = int(0.35 * y), int(0.75 * y)
            if cx1 <= cx0:
                cx1 = min(x, cx0 + 1)
            if cy1 <= cy0:
                cy1 = min(y, cy0 + 1)
            self._stage2_geom_cache[key] = (cx0, cx1, cy0, cy1)
            cached = self._stage2_geom_cache[key]
        cx0, cx1, cy0, cy1 = cached

        p = np.full((z, y, x), 0.02, dtype=np.float32)
        p[:, cy0:cy1, cx0:cx1] = 0.05
        return p.astype(save_dtype, copy=False)

    def get_score(self, pred_c1_c7, pred_fracture):
        output = np.zeros(8, np.float32)  # Overall, C1-C7

        if (pred_c1_c7 is not None) and (pred_fracture is not None):
            pred_masked = pred_c1_c7
            if pred_masked.max(initial=0) > 7:
                pred_masked = pred_masked.copy()
                pred_masked[pred_masked > 7] = 0

            for C_i in range(8):
                thr = self.params["alpha"][C_i]
                if C_i == 0:
                    m = (pred_masked > 0) & (pred_fracture >= thr)
                else:
                    m = (pred_masked == C_i) & (pred_fracture >= thr)

                if not np.any(m):
                    output[C_i] = self.params["min_score"][C_i]
                else:
                    roi_fracture = pred_fracture[m]
                    output[C_i] = max(
                        self.params["min_score"][C_i],
                        min(
                            self.params["max_score"][C_i],
                            np.percentile(roi_fracture, 100 * self.params["beta"][C_i]),
                        ),
                    )
        else:
            output[:] = np.array(self.params["min_score"], dtype=np.float32)
        return output

    def predict(self, list_test_files):
        overall_time_start = time.time()
        for file in list_test_files:
            case_id = os.path.basename(file.rstrip("/"))

            z = _count_slices_fast(file)
            full_shape_zyx = (int(z), 512, 512)
            spacing_zyx = (1.0, 1.0, 1.0)

            c1_c7_bbox = self.get_c1_c7_bbox_from_shape(full_shape_zyx, spacing_zyx)

            if c1_c7_bbox is not None:
                bz, ez, by, ey, bx, ex = c1_c7_bbox
                roi_shape_zyx = (ez - bz + 1, ey - by + 1, ex - bx + 1)

                roi_pred_1 = np.zeros(roi_shape_zyx, dtype=np.int16)

                key = full_shape_zyx
                cx0, cx1, cy0, cy1, z_edges = self._stage1_geom_cache[key]

                x0 = max(cx0, bx)
                x1 = min(cx1, ex + 1)
                y0 = max(cy0, by)
                y1 = min(cy1, ey + 1)
                if (x1 > x0) and (y1 > y0):
                    rx0, rx1 = x0 - bx, x1 - bx
                    ry0, ry1 = y0 - by, y1 - by
                    for c in range(1, 8):
                        z0 = int(z_edges[c - 1])
                        z1 = int(z_edges[c])
                        zz0 = max(z0, bz)
                        zz1 = min(z1, ez + 1)
                        if zz1 <= zz0:
                            continue
                        rz0, rz1 = zz0 - bz, zz1 - bz
                        roi_pred_1[rz0:rz1, ry0:ry1, rx0:rx1] = c

                roi_pred_1 = keep_largest_cervical_cc(roi_pred_1, spacing_zyx)
                roi_pred_2 = self.predict_stage2_from_roi_shape(
                    roi_shape_zyx, save_dtype=np.float32
                )
            else:
                roi_pred_1 = None
                roi_pred_2 = None

            score = self.get_score(roi_pred_1, roi_pred_2)
            self.results[case_id] = score

        self.total_infer_seconds_ = time.time() - overall_time_start




## === cell 2
time_start = time.time()

DATA_DIR = "../input/rsna-2022-cervical-spine-fracture-detection/test_images"
if not os.path.isdir(DATA_DIR):
    alt = "/kaggle/input/rsna-2022-cervical-spine-fracture-detection/test_images"
    if os.path.isdir(alt):
        DATA_DIR = alt

SAVE_CSV = "submission.csv"

if not os.path.isdir(DATA_DIR):
    raise FileNotFoundError(f"test_images directory not found at: {DATA_DIR}")

study_dirs = []
with os.scandir(DATA_DIR) as it:
    for e in it:
        if not e.is_dir():
            continue
        try:
            has_file = False
            with os.scandir(e.path) as it2:
                for e2 in it2:
                    if e2.is_file():
                        has_file = True
                        break
            if has_file:
                study_dirs.append(e.path)
        except FileNotFoundError:
            continue

study_dirs = sorted(study_dirs)
total_scans = len(study_dirs)
print("==> Total scans in test_images:", total_scans)
if total_scans == 0:
    raise FileNotFoundError(f"No study folders with files found under: {DATA_DIR}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("==> Using device:", device)

with torch.no_grad():
    predictor_1 = object()
    predictor_2 = object()

    c2f_predictor = FractureDetector(
        predictor_stage1=predictor_1, predictor_stage2=predictor_2
    )

    print(f"==> Total {len(study_dirs)} cases to run inference on")

    precompute_slice_counts(study_dirs)

    c2f_predictor.predict(list_test_files=study_dirs)
    print(
        f"==> Inference seconds (loop): {getattr(c2f_predictor, 'total_infer_seconds_', float('nan')):.2f}"
    )

    results = c2f_predictor.results

    test_csv_path = "../input/rsna-2022-cervical-spine-fracture-detection/test.csv"
    if not os.path.isfile(test_csv_path):
        test_csv_path = (
            "/kaggle/input/rsna-2022-cervical-spine-fracture-detection/test.csv"
        )
    test_df = pd.read_csv(test_csv_path)

    min_scores = np.array(c2f_predictor.params["min_score"], dtype=np.float32)

    uids = np.fromiter(results.keys(), dtype=object, count=len(results))
    if len(uids) > 0:
        mat = np.empty((len(uids), 8), dtype=np.float32)
        for i, uid in enumerate(uids):
            mat[i] = results[uid]
        res_df = pd.DataFrame(
            mat,
            columns=["patient_overall", "C1", "C2", "C3", "C4", "C5", "C6", "C7"],
        )
        res_df.insert(0, "StudyInstanceUID", uids)
        merged = test_df.merge(res_df, on="StudyInstanceUID", how="left", copy=False)
    else:
        merged = test_df.copy()

    ptype = merged["prediction_type"].to_numpy()

    out = np.empty(len(merged), dtype=np.float32)
    out.fill(min_scores[0])
    for i, k in enumerate(("C1", "C2", "C3", "C4", "C5", "C6", "C7"), start=1):
        m = ptype == k
        if m.any():
            out[m] = min_scores[i]

    for col in ("patient_overall", "C1", "C2", "C3", "C4", "C5", "C6", "C7"):
        m = ptype == col
        if not m.any():
            continue
        vals = merged[col].to_numpy(dtype=np.float32, copy=False)[m]
        if np.isnan(vals).any():
            vals = vals.copy()
            vals[np.isnan(vals)] = out[m][np.isnan(vals)]
        out[m] = vals

    sub = pd.DataFrame({"row_id": merged["row_id"].values, "fractured": out})
    sub["fractured"] = sub["fractured"].clip(1e-6, 1 - 1e-6)

    sub.to_csv(SAVE_CSV, index=False)
    print("==> Wrote:", SAVE_CSV, "rows:", len(sub), "cols:", list(sub.columns))
    print(f"==> Finish using time: {time.time() - time_start:.2f} seconds")
