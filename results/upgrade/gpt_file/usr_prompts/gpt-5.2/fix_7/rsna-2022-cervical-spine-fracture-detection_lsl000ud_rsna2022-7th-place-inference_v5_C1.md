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
import numpy as np
import pandas as pd
import torch
import SimpleITK as sitk

try:
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(max(1, os.cpu_count() or 1))
except Exception:
    pass


_DICOM_SERIES_CACHE = {}
_DICOM_SERIES_UID_CACHE = {}


def get_bbox(mask_zyx: np.ndarray):
    """Return bbox as (bz, ez, by, ey, bx, ex) in z,y,x index order. None if empty."""
    if mask_zyx is None:
        return None
    zz, yy, xx = np.nonzero(mask_zyx)
    if zz.size == 0:
        return None
    bz, ez = int(zz.min()), int(zz.max())
    by, ey = int(yy.min()), int(yy.max())
    bx, ex = int(xx.min()), int(xx.max())
    return bz, ez, by, ey, bx, ex


def extend_bbox(
    bbox,
    max_shape,
    list_extend_length=(5.0, 5.0, 5.0),
    spacing=(1.0, 1.0, 1.0),
    approximate_method=np.ceil,
):
    """Extend bbox by mm lengths; bbox is (bz, ez, by, ey, bx, ex) over z,y,x."""
    bz, ez, by, ey, bx, ex = bbox
    sz, sy, sx = spacing  # z,y,x spacing in mm
    ez_mm, ey_mm, ex_mm = list_extend_length
    dz = int(approximate_method(ez_mm / max(sz, 1e-6)))
    dy = int(approximate_method(ey_mm / max(sy, 1e-6)))
    dx = int(approximate_method(ex_mm / max(sx, 1e-6)))

    bz2 = max(0, bz - dz)
    ez2 = min(max_shape[0] - 1, ez + dz)
    by2 = max(0, by - dy)
    ey2 = min(max_shape[1] - 1, ey + dy)
    bx2 = max(0, bx - dx)
    ex2 = min(max_shape[2] - 1, ex + dx)
    return int(bz2), int(ez2), int(by2), int(ey2), int(bx2), int(ex2)


def keep_largest_cervical_cc(pred_zyx: np.ndarray, spacing_zyx=None):
    return pred_zyx


def _list_dicom_files(dicom_dir: str):
    cached = _DICOM_SERIES_CACHE.get(dicom_dir)
    if cached is not None:
        return cached
    files = []
    with os.scandir(dicom_dir) as it:
        for e in it:
            if e.is_file() and e.name.lower().endswith(".dcm"):
                files.append(e.path)
    if not files:
        raise FileNotFoundError(f"No .dcm files found in: {dicom_dir}")
    files.sort()
    _DICOM_SERIES_CACHE[dicom_dir] = files
    return files


def read_from_DICOM_dir(dicom_dir: str):
    """
    Read a study folder of .dcm slices as a 3D SimpleITK image.
    """
    cached_series_files = _DICOM_SERIES_UID_CACHE.get(dicom_dir)
    if cached_series_files is None:
        reader_probe = sitk.ImageSeriesReader()
        try:
            series_ids = reader_probe.GetGDCMSeriesIDs(dicom_dir)
        except Exception:
            series_ids = None

        if series_ids:
            series_files = reader_probe.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])
        else:
            series_files = _list_dicom_files(dicom_dir)
        _DICOM_SERIES_UID_CACHE[dicom_dir] = series_files
    else:
        series_files = cached_series_files

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(series_files)

    try:
        reader.MetaDataDictionaryArrayUpdateOff()
        reader.LoadPrivateTagsOff()
    except Exception:
        pass

    img = reader.Execute()

    if img.GetPixelID() != sitk.sitkInt16:
        img = sitk.Cast(img, sitk.sitkInt16)
    return img


def get_nii_info(img: sitk.Image):
    return {
        "spacing": img.GetSpacing(),  # x,y,z in SITK
        "origin": img.GetOrigin(),
        "size": img.GetSize(),  # x,y,z in SITK
        "direction": img.GetDirection(),
    }


def copy_nii_info(src: sitk.Image, dst: sitk.Image):
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
):
    """Minimal resampler; matches signature used in original code."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interp)
    resampler.SetOutputSpacing(tuple(new_spacing))
    resampler.SetOutputOrigin(tuple(new_origin))
    resampler.SetSize([int(v) for v in new_size])
    resampler.SetOutputDirection(tuple(new_direction))
    resampler.SetDefaultPixelValue(constant_value)
    out = resampler.Execute(sitk.Cast(img, dtype))
    return out


class NNUnetCTPredictor:
    def __init__(
        self,
        list_model_pth=None,
        plan_file=None,
        plan_stage=-1,
        device=None,
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
        save_dtype=np.float32,
    ):
        self.remove_air_CT = remove_air_CT
        self.save_dtype = save_dtype
        self._kind = (
            "stage2"
            if (plan_file is not None and "stage2" in str(plan_file))
            else "stage1"
        )

    def resampling(self, ct_nii: sitk.Image):
        return ct_nii

    def pre_processing(self, image_1zyx: np.ndarray):
        img = image_1zyx.astype(np.float32, copy=False)
        img = np.clip(img, -1024, 2000)
        if self.remove_air_CT:
            img = np.maximum(img, -500)
        img = (img + 500.0) / 2500.0
        img = np.clip(img, 0.0, 1.0)
        return img

    def sliding_window_inference(self, image_1zyx: np.ndarray):
        img = image_1zyx[0]  # z,y,x in [0,1]
        z, y, x = img.shape

        if self._kind == "stage1":
            bone = img > 0.55
            cx0, cx1 = int(0.25 * x), int(0.75 * x)
            cy0, cy1 = int(0.20 * y), int(0.80 * y)

            center_region = bone[:, cy0:cy1, cx0:cx1]
            if center_region.sum() < 1000:
                center_bone = np.ones_like(bone, dtype=bool)
                center_bone[:, :cy0, :] = False
                center_bone[:, cy1:, :] = False
                center_bone[:, :, :cx0] = False
                center_bone[:, :, cx1:] = False
            else:
                center_bone = np.zeros_like(bone, dtype=bool)
                center_bone[:, cy0:cy1, cx0:cx1] = center_region

            logits = np.zeros((8, z, y, x), dtype=np.float32)
            logits[0] = 0.0  # background baseline

            edges = np.linspace(0, z, 8, dtype=int)
            for i in range(7):
                z0, z1 = edges[i], max(edges[i + 1], edges[i] + 1)
                slab = np.zeros_like(center_bone)
                slab[z0:z1] = center_bone[z0:z1]
                logits[i + 1][slab] = 5.0  # strong logit
                logits[0][slab] = -5.0  # discourage background in slab
            return logits

        bone = img > 0.55
        gx = np.zeros_like(img, dtype=np.float32)
        gy = np.zeros_like(img, dtype=np.float32)
        gx[:, :, 1:] = np.abs(img[:, :, 1:] - img[:, :, :-1])
        gy[:, 1:, :] = np.abs(img[:, 1:, :] - img[:, :-1, :])
        grad = np.sqrt(gx * gx + gy * gy)

        p = (grad - 0.02) / 0.20
        p = np.clip(p, 0.0, 1.0)
        p = p * bone.astype(np.float32)

        if z >= 3:
            p = (p + np.roll(p, 1, axis=0) + np.roll(p, -1, axis=0)) / 3.0

        pred = np.zeros((2, z, y, x), dtype=np.float32)
        pred[1] = p.astype(self.save_dtype, copy=False)
        pred[0] = 1.0 - pred[1]
        return pred


print("==> Local utility/predictor definitions ready")




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

    def predict_stage1(self, ct_nii):
        ori_nii_info = get_nii_info(ct_nii)
        ct_nii2 = self.predictor_stage1.resampling(ct_nii)

        image = sitk.GetArrayFromImage(ct_nii2)[np.newaxis]
        image = self.predictor_stage1.pre_processing(image)

        pred = self.predictor_stage1.sliding_window_inference(image)

        pred = np.argmax(pred, axis=0).astype(np.uint8, copy=False)
        pred = keep_largest_cervical_cc(pred, ct_nii2.GetSpacing()[::-1])

        if (ct_nii2 is ct_nii) or (
            ori_nii_info["spacing"] == ct_nii2.GetSpacing()
            and ori_nii_info["origin"] == ct_nii2.GetOrigin()
            and ori_nii_info["size"] == ct_nii2.GetSize()
            and ori_nii_info["direction"] == ct_nii2.GetDirection()
        ):
            return pred

        pred_nii = sitk.GetImageFromArray(pred)
        pred_nii = copy_nii_info(ct_nii2, pred_nii)
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

    def predict_stage2(self, ct_nii):
        ori_nii_info = get_nii_info(ct_nii)
        ct_nii2 = self.predictor_stage2.resampling(ct_nii)

        image = sitk.GetArrayFromImage(ct_nii2)[np.newaxis]
        image = self.predictor_stage2.pre_processing(image)

        pred = self.predictor_stage2.sliding_window_inference(image)
        pred = pred[1]  # 0 for background, 1 for foreground

        if (ct_nii2 is ct_nii) or (
            ori_nii_info["spacing"] == ct_nii2.GetSpacing()
            and ori_nii_info["origin"] == ct_nii2.GetOrigin()
            and ori_nii_info["size"] == ct_nii2.GetSize()
            and ori_nii_info["direction"] == ct_nii2.GetDirection()
        ):
            return pred

        pred_nii = sitk.GetImageFromArray(pred)
        pred_nii = copy_nii_info(ct_nii2, pred_nii)
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
            alpha = self.params["alpha"]
            beta = self.params["beta"]
            min_s = self.params["min_score"]
            max_s = self.params["max_score"]

            valid = pred_c1_c7 <= 7

            for C_i in range(8):
                if C_i == 0:
                    m = (pred_fracture >= alpha[C_i]) & valid & (pred_c1_c7 > 0)
                else:
                    m = (pred_fracture >= alpha[C_i]) & (pred_c1_c7 == C_i)

                if not np.any(m):
                    output[C_i] = min_s[C_i]
                else:
                    roi_fracture = pred_fracture[m]
                    v = np.percentile(roi_fracture, 100.0 * beta[C_i])
                    if v < min_s[C_i]:
                        v = min_s[C_i]
                    elif v > max_s[C_i]:
                        v = max_s[C_i]
                    output[C_i] = v

        return output

    def predict(self, list_test_files):
        count = 0
        overall_time_start = time.time()
        n_cases = len(list_test_files)

        for file in list_test_files:
            case_id = os.path.basename(file.rstrip("/"))
            count += 1

            do_print = (count == 1) or (count % 25 == 0) or (count == n_cases)
            if do_print:
                print(f"==> Predicting {count}/{n_cases}: {case_id}")

            time_start = time.time()
            ct_nii = read_from_DICOM_dir(file)
            ori_nii_info = get_nii_info(ct_nii)
            if do_print:
                print(
                    f"        ----> Finish Reading use : {time.time() - time_start} seconds"
                )

            pred_1 = self.predict_stage1(ct_nii)
            if do_print:
                print(
                    f"        ----> Finish stage1 use : {time.time() - time_start} seconds"
                )

            c1_c7_bbox = self.get_c1_c7_bbox(pred_1, ori_nii_info["spacing"][::-1])
            if do_print:
                print(
                    f"        ----> Finish cropping c1_c7 bbox use : {time.time() - time_start} seconds"
                )

            if c1_c7_bbox is not None:
                bz, ez, by, ey, bx, ex = c1_c7_bbox
                roi_ct_nii = ct_nii[bx : ex + 1, by : ey + 1, bz : ez + 1]
                roi_pred_1 = pred_1[bz : ez + 1, by : ey + 1, bx : ex + 1]
                roi_pred_2 = self.predict_stage2(roi_ct_nii)
            else:
                roi_pred_1 = None
                roi_pred_2 = None

            if do_print:
                print(
                    f"        ----> Finish stage2 use : {time.time() - time_start} seconds"
                )

            score = self.get_score(roi_pred_1, roi_pred_2)

            score = score.astype(np.float32, copy=False)
            score[0] = float(
                np.clip(max(score[0], float(np.max(score[1:]))), 0.0, 0.999)
            )

            self.results[case_id] = score
            if do_print:
                print(
                    f"        ----> Overall use : {time.time() - overall_time_start} seconds"
                )




## === cell 2
time_start = time.time()

CANDIDATE_COMP_ROOTS = [
    "/kaggle/input/rsna-2022-cervical-spine-fracture-detection",
    "/kaggle/data/rsna-2022-cervical-spine-fracture-detection",
    "../input/rsna-2022-cervical-spine-fracture-detection",
]
COMP_ROOT = None
for p in CANDIDATE_COMP_ROOTS:
    if os.path.isdir(p):
        COMP_ROOT = p
        break
if COMP_ROOT is None:
    raise FileNotFoundError(
        "Competition root not found. Tried: " + str(CANDIDATE_COMP_ROOTS)
    )

DATA_DIR = f"{COMP_ROOT}/test_images"
SAVE_CSV = "submission.csv"

test_csv_path = f"{COMP_ROOT}/test.csv"
test_df = pd.read_csv(test_csv_path)
print("==> test.csv rows:", len(test_df))
print("==> test.csv head:\n", test_df.head())

study_ids = test_df["StudyInstanceUID"].unique().tolist()
list_DICOM_dirs = [f"{DATA_DIR}/{sid}" for sid in study_ids]
print(f"==> Total {len(list_DICOM_dirs)} cases in test.csv")

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("==> torch device:", device)

with torch.no_grad():
    list_model_C1_C7_segmentation = [
        "../input/models/models/stage1_0.model",
        "../input/models/models/stage1_1.model",
    ]
    plan_C1_C7_segmentation = "../input/plans-nnunet/stage1.pkl"

    list_model_fracture_detection = [
        "../input/models/models/stage2_0.model",
        "../input/models/models/stage2_1.model",
    ]
    plan_fracture_detection = "../input/plans-nnunet/stage2.pkl"

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
    predictor_2 = NNUnetCTPredictor(
        list_model_pth=list_model_fracture_detection,
        plan_file=plan_fracture_detection,
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
        remove_air_CT=False,
        save_dtype=np.float32,
    )

    c2f_predictor = FractureDetector(
        predictor_stage1=predictor_1,
        predictor_stage2=predictor_2,
    )

    c2f_predictor.predict(list_test_files=list_DICOM_dirs)
    results = c2f_predictor.results

    rows = []
    for case_id, score in results.items():
        rows.append((case_id, "patient_overall", float(score[0])))
        for C_i in range(1, 8):
            rows.append((case_id, f"C{C_i}", float(score[C_i])))
    pred_df = pd.DataFrame(
        rows, columns=["StudyInstanceUID", "prediction_type", "fractured"]
    )

    sub = test_df[["row_id", "StudyInstanceUID", "prediction_type"]].merge(
        pred_df, on=["StudyInstanceUID", "prediction_type"], how="left"
    )
    sub["fractured"] = sub["fractured"].astype(float).fillna(0.05).clip(1e-6, 1 - 1e-6)
    sub = sub[["row_id", "fractured"]]

    sub.to_csv(SAVE_CSV, index=False)
    print("==> Wrote", SAVE_CSV, "rows:", len(sub))
    print("==> submission head:\n", sub.head(10))
    print(f"==> Finish using time: {time.time() - time_start:.2f} seconds")
