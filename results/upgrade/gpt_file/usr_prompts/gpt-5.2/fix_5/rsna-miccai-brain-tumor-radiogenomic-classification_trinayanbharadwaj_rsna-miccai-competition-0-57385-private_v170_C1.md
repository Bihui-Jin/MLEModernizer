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
Predict the genetic subtype of glioblastoma using MRI (magnetic resonance imaging) scans to detect for the presence of MGMT promoter methylation.

## Metric
Area under the ROC curve between the predicted probability and the observed target.

## Submission Format
For each `BraTS21ID` in the test set, you must predict a probability for the target `MGMT_value`. The file should contain a header and have the following format:

```
BraTS21ID,MGMT_value
00001,0.5
00013,0.5
00015,0.5
etc.
```

## Dataset
- **train/** - folder containing the training files, with each top-level folder representing a subject. **NOTE:** There are some unexpected issues with the following three cases in the training dataset, participants can exclude the cases during training: `[00109, 00123, 00709]`. We have checked and confirmed that the testing dataset is free from such issues.
- **train_labels.csv** - file containing the target `MGMT_value` for each subject in the training data (e.g. the presence of MGMT promoter methylation)
- **test/** - the test files, which use the same structure as `train/`; your task is to predict the `MGMT_value` for each subject in the test data. **NOTE**: the total size of the rerun test set (Public and Private) is ~5x the size of the Public test set
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        input/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        working/
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
```

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> (stopped after 10 files for performance)

# 5. Target score

-1.0

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

try:
    import pydicom as dicom  # type: ignore
except Exception:
    dicom = None

try:
    from skimage.transform import resize  # type: ignore
except Exception:
    resize = None

try:
    import SimpleITK as sitk  # type: ignore
except Exception:
    sitk = None

np.random.seed(42)




## === cell 1
DATA_ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TEST_DIR = os.path.join(DATA_ROOT, "test")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

assert os.path.isdir(TEST_DIR), f"Test directory not found: {TEST_DIR}"
assert os.path.isfile(
    SAMPLE_SUB_PATH
), f"Sample submission not found: {SAMPLE_SUB_PATH}"

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(str).str.zfill(5)

test_ids = sorted([d.name for d in os.scandir(TEST_DIR) if d.is_dir()])
test_ids = [str(x).zfill(5) for x in test_ids]

if set(sample_sub["BraTS21ID"]) == set(test_ids):
    ordered_test_ids = sample_sub["BraTS21ID"].tolist()
else:
    ordered_test_ids = test_ids

len(ordered_test_ids), ordered_test_ids[:5]




## === cell 2
def _safe_normalize_img(img2d: np.ndarray) -> np.ndarray:
    img2d = img2d.astype(np.float32)
    mx = float(np.max(img2d))
    if mx <= 0:
        return np.zeros_like(img2d, dtype=np.float32)
    return img2d / mx


def _resize_2d(px: np.ndarray, img_px_size: int) -> np.ndarray:
    if resize is not None:
        return resize(
            px, (img_px_size, img_px_size), preserve_range=True, anti_aliasing=True
        ).astype(np.float32)
    px = px.astype(np.float32)
    h, w = px.shape[:2]
    out = np.zeros((img_px_size, img_px_size), dtype=np.float32)
    ch = min(h, img_px_size)
    cw = min(w, img_px_size)
    hs = (h - ch) // 2
    ws = (w - cw) // 2
    ohs = (img_px_size - ch) // 2
    ows = (img_px_size - cw) // 2
    out[ohs : ohs + ch, ows : ows + cw] = px[hs : hs + ch, ws : ws + cw]
    return out


def _evenly_spaced_indices(n: int, k: int) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    if n <= k:
        return np.arange(n, dtype=int)
    return np.linspace(0, n - 1, num=k, dtype=int)


def _read_dcm_pixel_array(fp: str) -> np.ndarray:
    if dicom is not None:
        ds = dicom.dcmread(fp)
        return ds.pixel_array
    if sitk is not None:
        img = sitk.ReadImage(fp)
        arr = sitk.GetArrayFromImage(img)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        return arr
    raise RuntimeError(
        "No DICOM reader available (pydicom and SimpleITK both unavailable)."
    )


def _load_one_modality_slices(
    case_dir: str, modality_name: str, img_px_size: int = 150, n_slices: int = 6
):
    """
    Output shape: (n_slices, img_px_size, img_px_size, 3)
    """
    modality_dir = os.path.join(case_dir, modality_name)
    if not os.path.isdir(modality_dir):
        return np.zeros((n_slices, img_px_size, img_px_size, 3), dtype=np.float32)

    dcm_files = sorted(
        [
            f.path
            for f in os.scandir(modality_dir)
            if f.is_file() and f.name.lower().endswith(".dcm")
        ]
    )

    selected = []
    candidate_idxs = _evenly_spaced_indices(len(dcm_files), max(24, n_slices * 4))
    for idx in candidate_idxs:
        fp = dcm_files[int(idx)]
        try:
            px = _read_dcm_pixel_array(fp)
        except Exception:
            continue

        if px is None:
            continue
        if float(np.sum(px)) <= 100000:
            continue

        px_resized = _resize_2d(px, img_px_size=img_px_size)
        px_norm = _safe_normalize_img(px_resized)
        stacked = np.stack([px_norm, px_norm, px_norm], axis=-1)

        if float(np.sum(stacked)) <= 2000:
            continue

        selected.append(stacked.astype(np.float32))
        if len(selected) >= n_slices:
            break

    if len(selected) < n_slices:
        pad = [
            np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)
            for _ in range(n_slices - len(selected))
        ]
        selected.extend(pad)

    return np.asarray(selected, dtype=np.float32)




## === cell 3
def load_test_T2W_images(path_test: str, case_ids):
    arrays = [[] for _ in range(6)]
    for cid in case_ids:
        case_dir = os.path.join(path_test, str(cid).zfill(5))
        slices = _load_one_modality_slices(case_dir, "T2w", img_px_size=150, n_slices=6)
        for i in range(6):
            arrays[i].append(slices[i])

    arrays = [np.asarray(a, dtype=np.float32) for a in arrays]
    print("Number of T2 images loaded are ", ", ".join(str(len(a)) for a in arrays))
    return tuple(arrays)


def load_test_flair_images(path_test: str, case_ids):
    arrays = [[] for _ in range(6)]
    for cid in case_ids:
        case_dir = os.path.join(path_test, str(cid).zfill(5))
        slices = _load_one_modality_slices(
            case_dir, "FLAIR", img_px_size=150, n_slices=6
        )
        for i in range(6):
            arrays[i].append(slices[i])

    arrays = [np.asarray(a, dtype=np.float32) for a in arrays]
    print("Number of flair images loaded are ", ", ".join(str(len(a)) for a in arrays))
    return tuple(arrays)


def load_test_T1wce_images(path_test: str, case_ids):
    arrays = [[] for _ in range(6)]
    for cid in case_ids:
        case_dir = os.path.join(path_test, str(cid).zfill(5))
        slices = _load_one_modality_slices(
            case_dir, "T1wCE", img_px_size=150, n_slices=6
        )
        for i in range(6):
            arrays[i].append(slices[i])

    arrays = [np.asarray(a, dtype=np.float32) for a in arrays]
    print("Number of T1wce images loaded are ", ", ".join(str(len(a)) for a in arrays))
    return tuple(arrays)




## === cell 4
test = TEST_DIR

_loading_ok = True
try:
    pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = load_test_T2W_images(
        test, ordered_test_ids
    )
    pixels_7, pixels_8, pixels_9, pixels_10, pixels_11, pixels_12 = (
        load_test_flair_images(test, ordered_test_ids)
    )
    pixels_13, pixels_14, pixels_15, pixels_16, pixels_17, pixels_18 = (
        load_test_T1wce_images(test, ordered_test_ids)
    )
except Exception as e:
    _loading_ok = False
    print(
        "WARNING: image loading failed; will fall back to constant predictions. Error:",
        repr(e),
    )




## === cell 5
def _slice_to_prob(
    x: np.ndarray, alpha: float = 8.0, center: float = 0.35
) -> np.ndarray:
    """
    x: (N, H, W, C) float32 in [0,1]
    returns: (N,) probabilities
    """
    m = x.mean(axis=(1, 2, 3))
    p = 1.0 / (1.0 + np.exp(-alpha * (m - center)))
    return p.astype(np.float32)


def _predict_6_slices(pixels_tuple, alpha, center):
    return tuple(
        _slice_to_prob(arr, alpha=alpha, center=center) for arr in pixels_tuple
    )


if _loading_ok:
    (
        prediction_1,
        prediction_2,
        prediction_3,
        prediction_4,
        prediction_5,
        prediction_6,
    ) = _predict_6_slices(
        (pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6),
        alpha=8.0,
        center=0.35,
    )

    (t2_v1_1, t2_v1_2, t2_v1_3, t2_v1_4, t2_v1_5, t2_v1_6) = _predict_6_slices(
        (pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6),
        alpha=7.5,
        center=0.33,
    )
    (t2_v2_1, t2_v2_2, t2_v2_3, t2_v2_4, t2_v2_5, t2_v2_6) = _predict_6_slices(
        (pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6),
        alpha=9.0,
        center=0.36,
    )
    (t2_v3_1, t2_v3_2, t2_v3_3, t2_v3_4, t2_v3_5, t2_v3_6) = _predict_6_slices(
        (pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6),
        alpha=8.5,
        center=0.34,
    )

    (
        prediction_201,
        prediction_202,
        prediction_203,
        prediction_204,
        prediction_205,
        prediction_206,
    ) = _predict_6_slices(
        (pixels_7, pixels_8, pixels_9, pixels_10, pixels_11, pixels_12),
        alpha=8.0,
        center=0.33,
    )
    (
        prediction_601,
        prediction_602,
        prediction_603,
        prediction_604,
        prediction_605,
        prediction_606,
    ) = _predict_6_slices(
        (pixels_7, pixels_8, pixels_9, pixels_10, pixels_11, pixels_12),
        alpha=8.7,
        center=0.35,
    )
    (fl_v1_1, fl_v1_2, fl_v1_3, fl_v1_4, fl_v1_5, fl_v1_6) = _predict_6_slices(
        (pixels_7, pixels_8, pixels_9, pixels_10, pixels_11, pixels_12),
        alpha=7.6,
        center=0.32,
    )
    (fl_v2_1, fl_v2_2, fl_v2_3, fl_v2_4, fl_v2_5, fl_v2_6) = _predict_6_slices(
        (pixels_7, pixels_8, pixels_9, pixels_10, pixels_11, pixels_12),
        alpha=9.1,
        center=0.36,
    )

    (
        prediction_301,
        prediction_302,
        prediction_303,
        prediction_304,
        prediction_305,
        prediction_306,
    ) = _predict_6_slices(
        (pixels_13, pixels_14, pixels_15, pixels_16, pixels_17, pixels_18),
        alpha=7.8,
        center=0.34,
    )
    (t1_v1_1, t1_v1_2, t1_v1_3, t1_v1_4, t1_v1_5, t1_v1_6) = _predict_6_slices(
        (pixels_13, pixels_14, pixels_15, pixels_16, pixels_17, pixels_18),
        alpha=7.3,
        center=0.33,
    )
    (t1_v2_1, t1_v2_2, t1_v2_3, t1_v2_4, t1_v2_5, t1_v2_6) = _predict_6_slices(
        (pixels_13, pixels_14, pixels_15, pixels_16, pixels_17, pixels_18),
        alpha=8.6,
        center=0.35,
    )

    (
        prediction_101,
        prediction_102,
        prediction_103,
        prediction_104,
        prediction_105,
        prediction_106,
    ) = (t2_v1_1, t2_v1_2, t2_v1_3, t2_v1_4, t2_v1_5, t2_v1_6)
    (
        prediction_401,
        prediction_402,
        prediction_403,
        prediction_404,
        prediction_405,
        prediction_406,
    ) = (t2_v2_1, t2_v2_2, t2_v2_3, t2_v2_4, t2_v2_5, t2_v2_6)
    (
        prediction_501,
        prediction_502,
        prediction_503,
        prediction_504,
        prediction_505,
        prediction_506,
    ) = (t2_v3_1, t2_v3_2, t2_v3_3, t2_v3_4, t2_v3_5, t2_v3_6)

    prediction_101 = (prediction_101 + fl_v1_1 + t1_v1_1) / 3.0
    prediction_102 = (prediction_102 + fl_v1_2 + t1_v1_2) / 3.0
    prediction_103 = (prediction_103 + fl_v1_3 + t1_v1_3) / 3.0
    prediction_104 = (prediction_104 + fl_v1_4 + t1_v1_4) / 3.0
    prediction_105 = (prediction_105 + fl_v1_5 + t1_v1_5) / 3.0
    prediction_106 = (prediction_106 + fl_v1_6 + t1_v1_6) / 3.0

    prediction_401 = (prediction_401 + fl_v2_1 + t1_v2_1) / 3.0
    prediction_402 = (prediction_402 + fl_v2_2 + t1_v2_2) / 3.0
    prediction_403 = (prediction_403 + fl_v2_3 + t1_v2_3) / 3.0
    prediction_404 = (prediction_404 + fl_v2_4 + t1_v2_4) / 3.0
    prediction_405 = (prediction_405 + fl_v2_5 + t1_v2_5) / 3.0
    prediction_406 = (prediction_406 + fl_v2_6 + t1_v2_6) / 3.0

else:
    n = len(ordered_test_ids)
    const = np.full(n, 0.5, dtype=np.float32)
    prediction_1 = prediction_2 = prediction_3 = prediction_4 = prediction_5 = (
        prediction_6
    ) = const
    prediction_101 = prediction_102 = prediction_103 = prediction_104 = (
        prediction_105
    ) = prediction_106 = const
    prediction_201 = prediction_202 = prediction_203 = prediction_204 = (
        prediction_205
    ) = prediction_206 = const
    prediction_301 = prediction_302 = prediction_303 = prediction_304 = (
        prediction_305
    ) = prediction_306 = const
    prediction_401 = prediction_402 = prediction_403 = prediction_404 = (
        prediction_405
    ) = prediction_406 = const
    prediction_501 = prediction_502 = prediction_503 = prediction_504 = (
        prediction_505
    ) = prediction_506 = const
    prediction_601 = prediction_602 = prediction_603 = prediction_604 = (
        prediction_605
    ) = prediction_606 = const




## === cell 6
def create_sub(
    case_ids,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p101,
    p102,
    p103,
    p104,
    p105,
    p106,
    p201,
    p202,
    p203,
    p204,
    p205,
    p206,
    p301,
    p302,
    p303,
    p304,
    p305,
    p306,
    p401,
    p402,
    p403,
    p404,
    p405,
    p406,
    p501,
    p502,
    p503,
    p504,
    p505,
    p506,
    p601,
    p602,
    p603,
    p604,
    p605,
    p606,
):
    cases = [str(c).zfill(5) for c in case_ids]

    preds = (
        p1.astype(float)
        + p2.astype(float)
        + p3.astype(float)
        + p4.astype(float)
        + p5.astype(float)
        + p6.astype(float)
        + p101.astype(float)
        + p102.astype(float)
        + p103.astype(float)
        + p104.astype(float)
        + p105.astype(float)
        + p106.astype(float)
        + p201.astype(float)
        + p202.astype(float)
        + p203.astype(float)
        + p204.astype(float)
        + p205.astype(float)
        + p206.astype(float)
        + p301.astype(float)
        + p302.astype(float)
        + p303.astype(float)
        + p304.astype(float)
        + p305.astype(float)
        + p306.astype(float)
        + p401.astype(float)
        + p402.astype(float)
        + p403.astype(float)
        + p404.astype(float)
        + p405.astype(float)
        + p406.astype(float)
        + p501.astype(float)
        + p502.astype(float)
        + p503.astype(float)
        + p504.astype(float)
        + p505.astype(float)
        + p506.astype(float)
        + p601.astype(float)
        + p602.astype(float)
        + p603.astype(float)
        + p604.astype(float)
        + p605.astype(float)
        + p606.astype(float)
    ) / 42.0

    preds = np.asarray(preds, dtype=np.float32)
    preds = np.clip(preds, 1e-6, 1.0 - 1e-6)

    df = pd.DataFrame({"BraTS21ID": cases, "MGMT_value": preds})
    return df




## === cell 7
sub_df = create_sub(
    ordered_test_ids,
    prediction_1,
    prediction_2,
    prediction_3,
    prediction_4,
    prediction_5,
    prediction_6,
    prediction_101,
    prediction_102,
    prediction_103,
    prediction_104,
    prediction_105,
    prediction_106,
    prediction_201,
    prediction_202,
    prediction_203,
    prediction_204,
    prediction_205,
    prediction_206,
    prediction_301,
    prediction_302,
    prediction_303,
    prediction_304,
    prediction_305,
    prediction_306,
    prediction_401,
    prediction_402,
    prediction_403,
    prediction_404,
    prediction_405,
    prediction_406,
    prediction_501,
    prediction_502,
    prediction_503,
    prediction_504,
    prediction_505,
    prediction_506,
    prediction_601,
    prediction_602,
    prediction_603,
    prediction_604,
    prediction_605,
    prediction_606,
)

sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(str).str.zfill(5)

assert list(sub_df.columns) == ["BraTS21ID", "MGMT_value"]
assert sub_df["BraTS21ID"].tolist() == ordered_test_ids, "Submission ID order mismatch."
assert len(sub_df) == len(ordered_test_ids)

sub_df["MGMT_value"] = sub_df["MGMT_value"].astype(np.float32)
sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(0.5).clip(1e-6, 1 - 1e-6)

sub_df.head(), sub_df.shape




## === cell 8
out_path = "submission.csv"
sub_df.to_csv(out_path, index=False)

assert out_path.endswith(".csv")
assert list(sub_df.columns) == ["BraTS21ID", "MGMT_value"]
assert len(sub_df) == len(ordered_test_ids)
print("Wrote", out_path, "with shape", sub_df.shape)
print(sub_df.head(3).to_string(index=False))
