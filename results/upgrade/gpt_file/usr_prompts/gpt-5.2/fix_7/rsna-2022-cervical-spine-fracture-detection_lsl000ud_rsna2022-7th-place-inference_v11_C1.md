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

0.2763309904508705

# 6. Current score

0.90573

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.90573) has done: 'I remove the hard dependency on the missing custom `src/Utils` package by implementing small, local replacements for the needed functions (DICOM read, bbox, simple post-processing) so the notebook runs in this Kaggle environment. I also make the model-loading robust: if the nnUNet predictor code/models aren’t available, the script safely fall back to constant baseline probabilities (using the provided `min_score` priors) and still write a valid `submission.csv`. Finally, I fix the `NameError` by defining `NNUnetCTPredictor` as a lightweight wrapper that uses the external implementation when present, otherwise a no-op fallback, keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.90573) has done: 'Your current score (0.90573, lower-is-better) is far worse than the target (0.27633), and the biggest issue is that the code is silently falling back to fixed priors because the nnUNet code/models aren’t found—so it’s not doing real inference at all. The minimal change to move the score toward the target is to actually locate and use the pretrained nnUNet weights/plans if they exist anywhere under the provided input roots, without changing the detector logic. I add a robust recursive search for `stage1_*.model`, `stage2_*.model`, and `stage1.pkl`/`stage2.pkl` under `/kaggle/input` and `/kaggle/data`, then keep everything else identical. If the assets truly don’t exist, it still produce the same valid baseline submission as before.'
- What this solution (achieved 0.90573) has done: 'Your score is far worse than the target (lower is better), and the dominant reason is that you’re still effectively producing a “min_score prior” submission because nnUNet code/models are not actually being found/used at runtime. I keep your detector logic unchanged and make the smallest practical change that increases the chance real inference runs: (1) broaden and harden the search for nnUNet assets (models + plans + the `Utils/Inference/nnunet_inference.py` code) specifically within common Kaggle dataset structures, and (2) if the external predictor is available, pass it the *found* assets with a strict sanity check so we don’t silently fall back. This should move the score substantially toward the target if the assets exist in the environment; if they truly don’t, it still produce the same valid submission.csv as before.'
- What this solution (achieved 0.90573) has done: 'Your current score is far worse than the target (lower is better), and the biggest lever without changing core logic is to ensure you actually run nnUNet inference instead of silently falling back to constant priors. I make the model/plan discovery stricter and more targeted by (1) prioritizing searches under likely “models/plans/nnunet” folders and (2) validating that we found a consistent set (>=1 stage1 model, >=1 stage2 model, and both plans) before constructing predictors. I also add a clear, single diagnostic print of exactly why inference is disabled (missing code vs missing assets vs init failure), so you don’t unknowingly submit the baseline again. No changes are made to FractureDetector logic, scoring, or post-processing—only asset discovery/selection and the gating that decides whether to run inference.'
- What this solution (achieved 0.90573) has done: 'Your score is far worse than the target (lower is better), and the most likely reason is that you are still running the “min_score prior” fallback because the external nnUNet inference code/models/plans are not being discovered/loaded correctly. I keep your FractureDetector logic unchanged and make the smallest changes that increase the chance real inference runs: (1) broaden model/plan discovery to also match common nnUNet filenames like `fold_*.pth`, `checkpoint_final.pth`, and `plans.pkl` (not only `stage1.pkl/.model`), and (2) only construct predictors when we have a consistent asset set, otherwise keep the existing safe fallback. I also ensure we map found assets to the correct stage by folder/name heuristics (stage1 vs stage2) without changing any inference/post-processing semantics. If assets still don’t exist in this environment, the output remains a valid submission identical in behavior to your current baseline.'

# 9. Code solution

## === cell 0
import os
import time
import sys
import numpy as np
import pandas as pd
import torch
import SimpleITK as sitk

INPUT_ROOTS = [
    "/kaggle/input/rsna-2022-cervical-spine-fracture-detection",
    "/kaggle/data/rsna-2022-cervical-spine-fracture-detection",
    "../input/rsna-2022-cervical-spine-fracture-detection",
    "/kaggle/input",
    "/kaggle/data",
    "../input",
]


def _first_existing_path(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


COMP_ROOT = _first_existing_path(INPUT_ROOTS)
if COMP_ROOT is None:
    raise FileNotFoundError(
        "Could not locate competition data under expected Kaggle input paths."
    )

TEST_CSV_PATH = _first_existing_path(
    [
        os.path.join(COMP_ROOT, "test.csv"),
        "/kaggle/input/rsna-2022-cervical-spine-fracture-detection/test.csv",
        "/kaggle/data/rsna-2022-cervical-spine-fracture-detection/test.csv",
        "../input/rsna-2022-cervical-spine-fracture-detection/test.csv",
    ]
)
SAMPLE_SUB_PATH = _first_existing_path(
    [
        os.path.join(COMP_ROOT, "sample_submission.csv"),
        "/kaggle/input/rsna-2022-cervical-spine-fracture-detection/sample_submission.csv",
        "/kaggle/data/rsna-2022-cervical-spine-fracture-detection/sample_submission.csv",
        "../input/rsna-2022-cervical-spine-fracture-detection/sample_submission.csv",
    ]
)

TEST_IMG_DIR = _first_existing_path(
    [
        os.path.join(COMP_ROOT, "test_images"),
        "/kaggle/input/rsna-2022-cervical-spine-fracture-detection/test_images",
        "/kaggle/data/rsna-2022-cervical-spine-fracture-detection/test_images",
        "../input/rsna-2022-cervical-spine-fracture-detection/test_images",
    ]
)

if TEST_CSV_PATH is None or SAMPLE_SUB_PATH is None or TEST_IMG_DIR is None:
    raise FileNotFoundError(
        "Missing one of required paths: test.csv, sample_submission.csv, test_images."
    )


def get_nii_info(img: sitk.Image):
    return {
        "spacing": img.GetSpacing(),
        "origin": img.GetOrigin(),
        "size": img.GetSize(),
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
    constant_value=0.0,
):
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(tuple(new_spacing))
    resampler.SetOutputOrigin(tuple(new_origin))
    resampler.SetSize([int(x) for x in new_size])
    resampler.SetOutputDirection(tuple(new_direction))
    resampler.SetInterpolator(interp)
    resampler.SetDefaultPixelValue(constant_value)
    out = resampler.Execute(sitk.Cast(img, dtype))
    return out


def read_from_DICOM_dir(dicom_dir: str):
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    if not series_ids:
        raise FileNotFoundError(f"No DICOM series found in {dicom_dir}")
    series_file_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])
    reader.SetFileNames(series_file_names)
    return reader.Execute()


def get_bbox(mask3d: np.ndarray):
    if mask3d is None or mask3d.size == 0:
        return None
    coords = np.argwhere(mask3d)
    if coords.size == 0:
        return None
    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0)
    return int(z0), int(z1), int(y0), int(y1), int(x0), int(x1)


def extend_bbox(
    bbox,
    max_shape,
    list_extend_length=(0.0, 0.0, 0.0),
    spacing=(1.0, 1.0, 1.0),
    approximate_method=np.ceil,
):
    bz, ez, by, ey, bx, ex = bbox
    ez = int(ez)
    ey = int(ey)
    ex = int(ex)
    bz = int(bz)
    by = int(by)
    bx = int(bx)
    spz, spy, spx = spacing
    extz = int(approximate_method(list_extend_length[0] / max(spz, 1e-6)))
    exty = int(approximate_method(list_extend_length[1] / max(spy, 1e-6)))
    extx = int(approximate_method(list_extend_length[2] / max(spx, 1e-6)))
    bz2 = max(0, bz - extz)
    ez2 = min(max_shape[0] - 1, ez + extz)
    by2 = max(0, by - exty)
    ey2 = min(max_shape[1] - 1, ey + exty)
    bx2 = max(0, bx - extx)
    ex2 = min(max_shape[2] - 1, ex + extx)
    return int(bz2), int(ez2), int(by2), int(ey2), int(bx2), int(ex2)


def keep_largest_cervical_cc(seg: np.ndarray, spacing_zyx=None):
    return seg


_EXTERNAL_PREDICTOR = None
try:
    candidate_src_roots = []
    for root in ["/kaggle/input", "/kaggle/data", "../input"]:
        if os.path.exists(root):
            candidate_src_roots.append(root)

    for root in candidate_src_roots:
        for dirpath, dirnames, filenames in os.walk(root):
            if (
                "nnunet_inference.py" in filenames
                and os.path.basename(dirpath) == "Inference"
            ):
                utils_dir = os.path.dirname(dirpath)  # .../Utils
                src_dir = os.path.dirname(utils_dir)  # .../src
                if (
                    os.path.basename(utils_dir) == "Utils"
                    and os.path.basename(src_dir) == "src"
                ):
                    if src_dir not in sys.path:
                        sys.path.insert(0, src_dir)

    from Utils.Inference.nnunet_inference import NNUnetCTPredictor as _NNUnetCTPredictor  # type: ignore

    _EXTERNAL_PREDICTOR = _NNUnetCTPredictor
except Exception:
    _EXTERNAL_PREDICTOR = None


class NNUnetCTPredictor:
    """
    Wrapper that uses the original NNUnetCTPredictor when available.
    Otherwise falls back to a no-op predictor returning empty/constant predictions
    so a valid submission is still produced.
    """

    def __init__(self, *args, **kwargs):
        self._ok = False
        self._inner = None
        if _EXTERNAL_PREDICTOR is not None:
            try:
                self._inner = _EXTERNAL_PREDICTOR(*args, **kwargs)
                self._ok = True
            except Exception as e:
                print(
                    "Warning: Failed to initialize external NNUnetCTPredictor; falling back. Error:",
                    repr(e),
                )
                self._inner = None
                self._ok = False

        self._resampling_mode = kwargs.get("resampling_mode", sitk.sitkNearestNeighbor)
        self._resampling_dtype = kwargs.get("resampling_dtype", sitk.sitkInt16)
        self._resampling_constance_value = kwargs.get(
            "resampling_constance_value", -1024
        )

    def resampling(self, ct_nii: sitk.Image):
        if self._ok:
            return self._inner.resampling(ct_nii)
        return sitk.Cast(ct_nii, self._resampling_dtype)

    def pre_processing(self, image_np: np.ndarray):
        if self._ok:
            return self._inner.pre_processing(image_np)
        return image_np.astype(np.float32, copy=False)

    def sliding_window_inference(self, image_np: np.ndarray):
        if self._ok:
            return self._inner.sliding_window_inference(image_np)
        if image_np.ndim != 4:
            raise ValueError(
                f"Expected image_np with shape (1,z,y,x), got {image_np.shape}"
            )
        _, z, y, x = image_np.shape
        return np.zeros((8, z, y, x), dtype=np.float32)


print(
    "Environment ready. External NNUnetCTPredictor available:",
    _EXTERNAL_PREDICTOR is not None,
)
print("COMP_ROOT:", COMP_ROOT)
print("TEST_IMG_DIR:", TEST_IMG_DIR)




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
        print(f"        ----> Resampling ...")
        ori_nii_info = get_nii_info(ct_nii)
        ct_nii = self.predictor_stage1.resampling(ct_nii)

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

    def predict_stage2(self, ct_nii):
        print(f"        ----> Resampling ...")
        ori_nii_info = get_nii_info(ct_nii)
        ct_nii = self.predictor_stage2.resampling(ct_nii)

        image = sitk.GetArrayFromImage(ct_nii)[np.newaxis]
        image = self.predictor_stage2.pre_processing(image)

        pred = self.predictor_stage2.sliding_window_inference(image)

        pred = pred[1]  # 0 for background, 1 for foreground

        pred_nii = sitk.GetImageFromArray(pred)
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
        output = np.zeros(8, np.float32)  # Overall, C1-C7

        if (pred_c1_c7 is not None) and (pred_fracture is not None):
            pred_c1_c7[pred_c1_c7 > 7] = 0

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

                if len(roi_fracture) == 0:
                    output[C_i] = self.params["min_score"][C_i]
                else:
                    output[C_i] = max(
                        self.params["min_score"][C_i],
                        min(
                            self.params["max_score"][C_i],
                            np.percentile(roi_fracture, 100 * self.params["beta"][C_i]),
                        ),
                    )
            output[0] = np.max(output[1:])

        return output

    def predict(self, list_test_files):
        count = 0
        overall_time_start = time.time()
        for file in list_test_files:
            case_id = os.path.basename(file.rstrip("/"))

            count += 1
            print(f"==> Predicting {count}: {case_id}")

            time_start = time.time()

            ct_nii = read_from_DICOM_dir(file)
            ori_nii_info = get_nii_info(ct_nii)
            print(
                f"        ----> Finish Reading use : {time.time() - time_start} seconds"
            )

            pred_1 = self.predict_stage1(ct_nii)
            print(
                f"        ----> Finish stage1 use : {time.time() - time_start} seconds"
            )

            c1_c7_bbox = self.get_c1_c7_bbox(pred_1, ori_nii_info["spacing"][::-1])
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

            print(
                f"        ----> Finish stage2 use : {time.time() - time_start} seconds"
            )

            score = self.get_score(roi_pred_1, roi_pred_2)
            self.results[case_id] = score
            print(
                f"        ----> Overall use : {time.time() - overall_time_start} seconds"
            )




## === cell 2
time_start = time.time()

DATA_DIR = TEST_IMG_DIR
SAVE_CSV = "submission.csv"

list_DICOM_dirs = sorted(
    [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
)
list_DICOM_dirs = [os.path.join(DATA_DIR, sub_dir) for sub_dir in list_DICOM_dirs]
print(f"==> Total {len(list_DICOM_dirs)} cases in {DATA_DIR}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def _find_files_by_basename(search_roots, target_basenames, max_hits_per_name=200):
    hits = {bn: [] for bn in target_basenames}
    for root in search_roots:
        if not root or not os.path.exists(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            for fn in filenames:
                if fn in hits:
                    lst = hits[fn]
                    if len(lst) < max_hits_per_name:
                        lst.append(os.path.join(dirpath, fn))
    return hits


def _iter_candidate_dirs(
    search_roots,
    must_contain_any=("model", "models", "nnunet", "plans", "checkpoint", "fold"),
):
    seen = set()
    for root in search_roots:
        if not root or not os.path.exists(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            low = dirpath.lower()
            if any(tok in low for tok in must_contain_any):
                if dirpath not in seen:
                    seen.add(dirpath)
                    yield dirpath
    for root in search_roots:
        if root and os.path.exists(root) and root not in seen:
            yield root


def _find_weight_files_targeted(
    search_roots, allow_ext=(".model", ".pth"), max_hits=400
):
    out = []
    for base in _iter_candidate_dirs(search_roots):
        for dirpath, dirnames, filenames in os.walk(base):
            for fn in filenames:
                if fn.endswith(allow_ext):
                    out.append(os.path.join(dirpath, fn))
                    if len(out) >= max_hits:
                        return sorted(out)
    return sorted(out)


def _filter_existing(files):
    return [f for f in files if os.path.isfile(f)]


def _is_stage1(path: str):
    low = path.lower()
    return (
        ("stage1" in low)
        or ("c1" in low and "c7" in low)
        or ("seg" in low and "fract" not in low)
    )


def _is_stage2(path: str):
    low = path.lower()
    return ("stage2" in low) or ("fract" in low) or ("det" in low)


def _choose_plan(plans: list[str], prefer_stage: int):
    if not plans:
        return None
    scored = []
    for p in plans:
        low = p.lower()
        s = 0
        if prefer_stage == 1 and ("stage1" in low or "c1" in low or "seg" in low):
            s += 5
        if prefer_stage == 2 and ("stage2" in low or "fract" in low or "det" in low):
            s += 5
        if os.path.basename(low) in ("plans.pkl", "stage1.pkl", "stage2.pkl"):
            s += 1
        scored.append((s, p))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[0][1]


SEARCH_ROOTS = ["/kaggle/input", "/kaggle/data", "../input"]

explicit_stage1 = [
    "../input/models/models/stage1_0.model",
    "../input/models/models/stage1_1.model",
    "/kaggle/input/models/models/stage1_0.model",
    "/kaggle/input/models/models/stage1_1.model",
]
explicit_stage2 = [
    "../input/models/models/stage2_0.model",
    "/kaggle/input/models/models/stage2_0.model",
]
explicit_plan_stage1 = [
    "../input/plans-nnunet/stage1.pkl",
    "/kaggle/input/plans-nnunet/stage1.pkl",
]
explicit_plan_stage2 = [
    "../input/plans-nnunet/stage2.pkl",
    "/kaggle/input/plans-nnunet/stage2.pkl",
]

list_model_C1_C7_segmentation = _filter_existing(explicit_stage1)
list_model_fracture_detection = _filter_existing(explicit_stage2)
plan_C1_C7_segmentation = _first_existing_path(explicit_plan_stage1)
plan_fracture_detection = _first_existing_path(explicit_plan_stage2)

if len(list_model_C1_C7_segmentation) == 0 or len(list_model_fracture_detection) == 0:
    all_weights = _find_weight_files_targeted(
        SEARCH_ROOTS, allow_ext=(".model", ".pth")
    )
    stage1_w = [p for p in all_weights if _is_stage1(p)]
    stage2_w = [p for p in all_weights if _is_stage2(p)]

    if len(stage1_w) == 0:
        stage1_w = [
            p for p in all_weights if os.path.basename(p).lower().startswith("stage1_")
        ]
    if len(stage2_w) == 0:
        stage2_w = [
            p for p in all_weights if os.path.basename(p).lower().startswith("stage2_")
        ]

    if len(list_model_C1_C7_segmentation) == 0:
        list_model_C1_C7_segmentation = sorted(stage1_w)[:50]
    if len(list_model_fracture_detection) == 0:
        list_model_fracture_detection = sorted(stage2_w)[:50]

if plan_C1_C7_segmentation is None or plan_fracture_detection is None:
    plan_hits = _find_files_by_basename(
        SEARCH_ROOTS,
        target_basenames=["stage1.pkl", "stage2.pkl", "plans.pkl", "plans.json"],
        max_hits_per_name=400,
    )
    stage1_plans = sorted(plan_hits.get("stage1.pkl", []))
    stage2_plans = sorted(plan_hits.get("stage2.pkl", []))
    generic_plans = sorted(plan_hits.get("plans.pkl", [])) + sorted(
        plan_hits.get("plans.json", [])
    )

    if plan_C1_C7_segmentation is None:
        plan_C1_C7_segmentation = _choose_plan(
            stage1_plans + generic_plans, prefer_stage=1
        )
    if plan_fracture_detection is None:
        plan_fracture_detection = _choose_plan(
            stage2_plans + generic_plans, prefer_stage=2
        )

print(
    "Stage1 models found:",
    len(list_model_C1_C7_segmentation),
    "plan:",
    plan_C1_C7_segmentation,
)
print(
    "Stage2 models found:",
    len(list_model_fracture_detection),
    "plan:",
    plan_fracture_detection,
)

reasons = []
if _EXTERNAL_PREDICTOR is None:
    reasons.append("external_nnunet_code_missing")
if plan_C1_C7_segmentation is None:
    reasons.append("stage1_plan_missing")
if plan_fracture_detection is None:
    reasons.append("stage2_plan_missing")
if len(list_model_C1_C7_segmentation) == 0:
    reasons.append("stage1_models_missing")
if len(list_model_fracture_detection) == 0:
    reasons.append("stage2_models_missing")

predictor_1 = None
predictor_2 = None
c2f_predictor = None
results = {}

with torch.no_grad():
    if len(reasons) == 0:
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
            tta=True,
            tta_flip_axis=(4,),
            resampling_tolerance=0.01,
            resampling_mode=sitk.sitkNearestNeighbor,
            resampling_dtype=sitk.sitkInt16,
            resampling_constance_value=-1024,
            remove_air_CT=False,
            save_dtype=np.float32,
        )
    else:
        predictor_1 = NNUnetCTPredictor(
            list_model_pth=[],
            plan_file=None,
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
            list_model_pth=[],
            plan_file=None,
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

    if len(reasons) == 0 and not getattr(predictor_1, "_ok", False):
        reasons.append("stage1_predictor_init_failed")
    if len(reasons) == 0 and not getattr(predictor_2, "_ok", False):
        reasons.append("stage2_predictor_init_failed")

    can_run_inference = (
        len(reasons) == 0
        and getattr(predictor_1, "_ok", False)
        and getattr(predictor_2, "_ok", False)
    )

    if can_run_inference:
        c2f_predictor.predict(list_test_files=list_DICOM_dirs)
        results = c2f_predictor.results
    else:
        print(
            "Warning: Inference disabled; writing prior-based baseline submission. Reasons:",
            sorted(set(reasons)),
        )
        results = {}  # will trigger min_score fallback per row

    test_df = pd.read_csv(TEST_CSV_PATH)

    pred_type_to_idx = {f"C{i}": i for i in range(1, 8)}
    pred_type_to_idx["patient_overall"] = 0

    fractured = np.empty(len(test_df), dtype=np.float32)
    for i, (uid, ptype) in enumerate(
        zip(test_df["StudyInstanceUID"].values, test_df["prediction_type"].values)
    ):
        idx = pred_type_to_idx.get(ptype, 0)
        if uid in results and ptype in pred_type_to_idx:
            val = float(results[uid][idx])
        else:
            val = float(c2f_predictor.params["min_score"][idx])
        fractured[i] = np.float32(min(max(val, 1e-6), 1 - 1e-6))

    sub_df = pd.DataFrame({"row_id": test_df["row_id"].values, "fractured": fractured})
    sub_df.to_csv(SAVE_CSV, index=False)

    print(f"==> Wrote {SAVE_CSV} with shape {sub_df.shape}")
    print(sub_df.head())
    print(f"==> Finish using time: {time.time() - time_start:.2f} seconds")
