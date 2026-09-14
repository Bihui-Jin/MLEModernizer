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

# 5. Target score

0.3113245697828474

# 6. Current score

0.56315

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.62268) has done: 'The timeout is dominated by per-slice DICOM decoding and Python-level loops inside feature extraction (especially `np.unique` per slice and recomputing percentiles via a Python dict). I keep the exact same features and prediction mapping, but make feature extraction asymptotically cheaper by replacing the per-slice `unique->dict` accumulation with a fixed-size histogram over the int16 HU range (provably equivalent for computing the empirical 95th percentile). I also avoid extra allocations by using `np.count_nonzero` and `np.bincount` directly on flattened arrays, and I reduce overhead in directory listing and merge by building predictions directly aligned to `test.csv`’s study order (while preserving identical semantics). Multiprocessing remains, but with tuned chunksize and a lighter worker function to reduce IPC overhead.'
- What this solution (achieved 0.59059) has done: 'Your current score (0.62268, lower-is-better) is far worse than the target (0.3113), so we should improve—but with minimal changes and without altering the model/training (none exists here). The biggest “free” gain is to make the `patient_overall` prediction logically consistent with the per-level predictions (since the metric heavily weights `patient_overall`): compute it as the union probability `1 - ∏(1 - p(Ci))` from the already-computed C1–C7 outputs, then apply a tiny blend with the original overall to avoid drastic shifts. Additionally, apply a very light temperature sharpening to probabilities (logit scaling) to reduce logloss when the baseline is under-confident, keeping changes small and fully deterministic. Everything else (feature extraction, multiprocessing, DICOM reading, submission formatting/paths) stays the same.'
- What this solution (achieved 0.57864) has done: 'We’re currently far worse than the target (0.59059 vs 0.3113, lower-is-better), so we should improve with the smallest changes that affect the most heavily-weighted part of the metric: `patient_overall`. I keep your exact feature extraction and per-level mapping, but make `patient_overall` strictly coherent with the C1–C7 predictions by applying the union-probability after temperature scaling (so it matches the final calibrated per-level outputs). I also slightly reduce the residual mixing with the old overall (more reliance on union) to better align with the metric, while keeping everything deterministic and preserving the same submission schema and paths.'
- What this solution (achieved 0.74079) has done: 'Your current score (0.57864, lower-is-better) is still far from the target (0.3113), so we should improve with minimal semantic changes that mostly affect the heavily-weighted `patient_overall` rows. I keep the same features, multiprocessing, and per-level mapping, but (1) compute a study-level prior from `train.csv` (instead of fixed constants) and use it in the same blending spots, and (2) lightly tune the two existing calibration knobs (`_temp`, `_overall_union_mix`) to rely a bit more on the union-of-levels consistency after scaling. These are small, deterministic changes that usually reduce weighted logloss without changing the core pipeline or adding any new modeling/training. Submission formatting, paths, and row alignment remain identical.'
- What this solution (achieved 0.56816) has done: 'Your score (0.74079, lower-is-better) is still far from the target (0.3113), so we should improve with the smallest changes that most affect weighted logloss: probability calibration and especially `patient_overall`. I keep the exact same feature extraction and the same mapping from features→logit→sigmoid, but (1) calibrate probabilities with a prior-preserving logit adjustment (adds a constant in logit space so the mean matches the empirical prior) and (2) make `patient_overall` strictly coherent with the per-level predictions by using the union probability without extra mixing. These are minimal, deterministic post-processing changes that typically reduce weighted logloss without changing the pipeline structure or adding any model/training. Submission schema, paths, multiprocessing, and row alignment remain identical.'
- What this solution (achieved 0.56315) has done: 'We’re still far above the target loss (0.56816 vs 0.3113, lower-is-better), so we should improve mainly where the metric is most sensitive: `patient_overall`. I keep your exact feature extraction and feature→logit→sigmoid mapping, but make the prior-matching calibration stable by computing the logit shift from the study-level mean (not the row-level mean) so the heavily-weighted overall prior is matched to the right quantity. Then I recompute `patient_overall` *after* calibration as the union of the calibrated C1–C7 probabilities (coherent and usually lowers weighted logloss), leaving C1–C7 untouched. Finally, I remove the redundant pre-calibration fill loop so we don’t accidentally mix pre/post calibrated values, without changing I/O paths or submission schema.'

# 9. Code solution

## === cell 0
import os
import time
import math
import gc
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch
import SimpleITK as sitk


DATA_ROOT = "/kaggle/input/rsna-2022-cervical-spine-fracture-detection"
TEST_CSV_PATH = os.path.join(DATA_ROOT, "test.csv")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
TRAIN_CSV_PATH = os.path.join(DATA_ROOT, "train.csv")
SAVE_CSV = "submission.csv"

LABELS = ["patient_overall"] + [f"C{i}" for i in range(1, 8)]


_SITK_READER = None


def _init_worker_sitk_reader():
    global _SITK_READER
    r = sitk.ImageFileReader()
    r.LoadPrivateTagsOff()
    r.SetOutputPixelType(sitk.sitkInt16)
    _SITK_READER = r


def _read_slice_array(dcm_path: str) -> np.ndarray:
    global _SITK_READER
    if _SITK_READER is None:
        _init_worker_sitk_reader()
    _SITK_READER.SetFileName(dcm_path)
    img = _SITK_READER.Execute()
    arr = sitk.GetArrayFromImage(img)  # (1, H, W) or (H, W)
    if arr.ndim == 3:
        arr = arr[0]
    return arr


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 1e-6, 1 - 1e-6)


def _logit(p: np.ndarray) -> np.ndarray:
    p = _clip01(p)
    return np.log(p) - np.log1p(-p)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _cpu_worker_count(max_workers: int) -> int:
    try:
        n = os.cpu_count() or 4
    except Exception:
        n = 4
    return max(1, min(int(max_workers), int(n)))


os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(0)
torch.manual_seed(0)


def _list_dcm_files_fast(dicom_dir: str) -> list[str]:
    files = []
    with os.scandir(dicom_dir) as it:
        for e in it:
            if e.is_file() and e.name.endswith(".dcm"):
                files.append((e.path, e.name))

    def _key(t):
        name = t[1]
        stem = name[:-4]
        if stem.isdigit():
            return (0, int(stem))
        return (1, stem)

    files.sort(key=_key)
    return [p for p, _ in files]


def _features_from_dicom_dir_fast(
    dicom_dir: str, z_stride: int = 4, xy_stride: int = 4
):
    try:
        files = _list_dcm_files_fast(dicom_dir)
        if not files:
            raise FileNotFoundError(f"No DICOM files found in: {dicom_dir}")
        files = files[::z_stride]

        total_px = 0
        air_px = 0
        bone_px = 0

        hist = np.zeros(65536, dtype=np.int64)
        bone_sum = 0.0
        bone_sumsq = 0.0

        for fp in files:
            a2 = _read_slice_array(fp)  # int16
            if xy_stride != 1:
                a2 = a2[::xy_stride, ::xy_stride]

            total_px += a2.size
            air_px += int(np.count_nonzero(a2 < -800))

            bone_mask = a2 > 150
            if not bone_mask.any():
                continue

            b = a2[bone_mask]  # int16
            bone_px += b.size

            idx = (b.astype(np.int32, copy=False) + 32768).ravel()
            hist += np.bincount(idx, minlength=65536)

            bf = b.astype(np.float64, copy=False)
            bone_sum += float(bf.sum())
            bone_sumsq += float((bf * bf).sum())

        if total_px == 0:
            return {"bone_frac": 0.0, "bone_p95": 0.0, "bone_std": 0.0, "air_frac": 0.0}

        air_frac = float(air_px / total_px)

        if bone_px < 128:
            return {
                "bone_frac": 0.0,
                "bone_p95": 0.0,
                "bone_std": 0.0,
                "air_frac": air_frac,
            }

        target = 0.95 * (bone_px - 1)
        lo = int(math.floor(target))
        hi = int(math.ceil(target))

        cdf = np.cumsum(hist)
        v_lo = int(np.searchsorted(cdf, lo + 1, side="left") - 32768)
        if lo == hi:
            p95 = float(v_lo)
        else:
            v_hi = int(np.searchsorted(cdf, hi + 1, side="left") - 32768)
            w = target - lo
            p95 = float(v_lo * (1.0 - w) + v_hi * w)

        mean = bone_sum / bone_px
        var = bone_sumsq / bone_px - mean * mean
        if var < 0.0:
            var = 0.0
        std = float(math.sqrt(var))

        return {
            "bone_frac": float(bone_px / total_px),
            "bone_p95": p95,
            "bone_std": std,
            "air_frac": air_frac,
        }
    except Exception as e:
        return ("__EXC__", dicom_dir, repr(e))


def _estimate_label_priors_from_train(train_csv_path: str) -> np.ndarray:
    """
    Use empirical priors from train.csv (same label set) instead of fixed constants.
    Improves logloss calibration, especially for heavily-weighted patient_overall.
    """
    try:
        tr = pd.read_csv(train_csv_path)
        cols = ["patient_overall"] + [f"C{i}" for i in range(1, 8)]
        for c in cols:
            if c not in tr.columns:
                raise KeyError(c)
        pri = tr[cols].mean(axis=0).to_numpy(dtype=np.float32)
        pri = _clip01(pri.astype(np.float64)).astype(np.float32)
        return pri
    except Exception:
        return np.array(
            [0.08, 0.03, 0.025, 0.02, 0.02, 0.025, 0.03, 0.035], dtype=np.float32
        )


def _logit_shift_to_match_prior(p: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """
    Change (score): stable prior-matching calibration.
    Compute the shift using the *study-level mean per label* (not the row-level mean across all 8 labels),
    so each label's mean aligns to its empirical prior; this especially stabilizes patient_overall.
    """
    p64 = _clip01(p.astype(np.float64, copy=False))
    prior64 = _clip01(prior.astype(np.float64, copy=False))

    mean_per_label = _clip01(p64.mean(axis=0))
    b = _logit(prior64) - _logit(mean_per_label)
    p_cal = _sigmoid(_logit(p64) + b)
    return _clip01(p_cal).astype(np.float32)


def _patient_union_from_levels(p_levels: np.ndarray) -> np.ndarray:
    """
    Change (score): enforce patient_overall coherence with per-level probabilities
    using union probability: 1 - Π(1 - p(Ci)).
    """
    p = _clip01(p_levels.astype(np.float64, copy=False))
    return _clip01(1.0 - np.prod(1.0 - p, axis=1)).astype(np.float32)




## === cell 1
class FractureDetector:
    """
    Fallback detector (no external model files).
    Produces 8 probabilities per study (overall + C1..C7).
    Core IO semantics preserved: per StudyInstanceUID produce 8 outputs.
    """

    def __init__(self, prior: np.ndarray | None = None):
        if prior is None:
            prior = np.array(
                [0.08, 0.03, 0.025, 0.02, 0.02, 0.025, 0.03, 0.035], dtype=np.float32
            )
        self.prior = prior.astype(np.float32, copy=True)
        self.results = {}

        self._temp = 0.85
        self._overall_union_mix = 1.0

    @staticmethod
    def _dir_to_features(dicom_dir: str):
        out = _features_from_dicom_dir_fast(dicom_dir)
        return (dicom_dir, out)

    @staticmethod
    def read_DICOM_multi_thread(list_DICOM_dirs, num_workers=4):
        if len(list_DICOM_dirs) == 0:
            return []

        if num_workers is None or num_workers <= 0:
            num_workers = os.cpu_count() or 4
        num_workers = _cpu_worker_count(num_workers)

        try:
            ctx = mp.get_context("fork")
        except ValueError:
            ctx = mp.get_context("spawn")

        chunksize = max(
            64, len(list_DICOM_dirs) // (num_workers * 3) if num_workers else 64
        )

        with ctx.Pool(
            processes=num_workers,
            initializer=_init_worker_sitk_reader,
            maxtasksperchild=512,
        ) as pool:
            outs = list(
                pool.imap_unordered(
                    FractureDetector._dir_to_features,
                    list_DICOM_dirs,
                    chunksize=chunksize,
                )
            )
        return outs

    def _predict_from_features(self, feats: dict) -> np.ndarray:
        """
        Map features to 8 probabilities. Kept simple and stable.
        """
        bone_frac = feats["bone_frac"]
        bone_std = feats["bone_std"]
        bone_p95 = feats["bone_p95"]
        air_frac = feats["air_frac"]

        s = (
            0.6 * (bone_frac - 0.08)
            + 0.002 * (bone_std - 600.0)
            + 0.0002 * (bone_p95 - 1200.0)
            - 0.3 * (air_frac - 0.35)
        )
        overall = float(1.0 / (1.0 + math.exp(-s)))
        overall = 0.65 * overall + 0.35 * float(self.prior[0])

        per_level = np.array([0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0], dtype=np.float32)
        c_probs = (overall * 0.55) * per_level + self.prior[1:] * 0.6
        out = np.concatenate([[overall], c_probs.astype(np.float32)], axis=0)
        out = _clip01(out).astype(np.float32)

        lt = _logit(out.astype(np.float64))
        out = _sigmoid(lt / float(self._temp)).astype(np.float32)
        out = _clip01(out).astype(np.float32)

        c = out[1:].astype(np.float64, copy=False)
        union = float(1.0 - np.prod(1.0 - _clip01(c)))
        out0 = float(out[0])
        out[0] = float(
            self._overall_union_mix * union + (1.0 - self._overall_union_mix) * out0
        )
        return _clip01(out).astype(np.float32)

    def predict(self, list_test_files, num_thread=4):
        overall_time_start = time.time()

        time_start = time.time()
        dir_and_feats = self.read_DICOM_multi_thread(
            list_test_files, num_workers=num_thread
        )

        feats_by_uid = {}
        exc_count = 0
        for dicom_dir, out in dir_and_feats:
            if isinstance(out, tuple) and len(out) >= 3 and out[0] == "__EXC__":
                exc_count += 1
                continue
            case_id = os.path.basename(dicom_dir.rstrip("/"))
            feats_by_uid[case_id] = out

        print(
            f"==> Finish Reading+Feature Extraction use : {time.time() - time_start:.2f} seconds (exc: {exc_count})"
        )

        time_start = time.time()
        for case_id, feats in feats_by_uid.items():
            self.results[case_id] = self._predict_from_features(feats)

        print(
            f"==> Finish Predict+Assemble use : {time.time() - time_start:.2f} seconds"
        )
        print(f"==> Overall use : {time.time() - overall_time_start:.2f} seconds")

        gc.collect()




## === cell 2
time_start = time.time()

test_df = pd.read_csv(TEST_CSV_PATH)
assert {"StudyInstanceUID", "prediction_type", "row_id"}.issubset(test_df.columns)

test_uids = test_df["StudyInstanceUID"].unique()
uid_set = set(test_uids.tolist())

list_DICOM_dirs = []
with os.scandir(TEST_IMG_DIR) as it:
    for e in it:
        if e.is_dir() and e.name in uid_set:
            list_DICOM_dirs.append(e.path)
list_DICOM_dirs.sort()

print(
    f"==> Total {len(list_DICOM_dirs)} cases found under {TEST_IMG_DIR} (from test.csv: {len(test_uids)})"
)

priors = _estimate_label_priors_from_train(TRAIN_CSV_PATH)
detector = FractureDetector(prior=priors)

detector.predict(
    list_test_files=list_DICOM_dirs, num_thread=min(12, os.cpu_count() or 4)
)
results = detector.results
fallback = detector.prior

uids_all = test_df["StudyInstanceUID"].unique()
preds_by_uid = {}
for uid in uids_all:
    r = results.get(uid)
    if r is None:
        preds_by_uid[uid] = fallback.astype(np.float32, copy=True)
    else:
        preds_by_uid[uid] = r.astype(np.float32, copy=True)

P = np.stack([preds_by_uid[uid] for uid in uids_all], axis=0)  # (n_study, 8)
P_cal = _logit_shift_to_match_prior(P, priors)

P_cal[:, 0] = _patient_union_from_levels(P_cal[:, 1:])

for k, uid in enumerate(uids_all):
    preds_by_uid[uid] = P_cal[k]

ptypes = test_df["prediction_type"].to_numpy(dtype=object)
uids = test_df["StudyInstanceUID"].to_numpy(dtype=object)
out = np.empty(test_df.shape[0], dtype=np.float32)

for i in range(test_df.shape[0]):
    r = preds_by_uid.get(uids[i])
    if r is None:
        r = fallback
    pt = ptypes[i]
    if pt == "patient_overall":
        out[i] = r[0]
    else:
        out[i] = r[int(pt[1:])]  # C1..C7 => 1..7

sub_df = pd.DataFrame({"row_id": test_df["row_id"].values, "fractured": _clip01(out)})
assert sub_df.shape[0] == test_df.shape[0]
assert sub_df["row_id"].is_unique
assert np.all((sub_df["fractured"].values >= 0.0) & (sub_df["fractured"].values <= 1.0))

sub_df.to_csv(SAVE_CSV, index=False)
print(f"==> Wrote {SAVE_CSV} with shape {sub_df.shape}")
print(f"==> Finish using time: {time.time() - time_start:.2f} seconds")
print(sub_df.head())
