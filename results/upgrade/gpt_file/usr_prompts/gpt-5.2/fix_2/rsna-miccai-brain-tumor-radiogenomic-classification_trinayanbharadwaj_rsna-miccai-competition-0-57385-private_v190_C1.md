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

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import pydicom as dicom
from skimage.transform import resize

try:
    import seaborn as sns  # noqa: F401
except Exception:
    sns = None

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
assert list(sample_sub.columns) == ["BraTS21ID", "MGMT_value"]




## === cell 2
def _safe_dcm_pixel_array(dcm_path: str):
    """Read DICOM robustly; return None if it cannot be read."""
    try:
        d = dicom.dcmread(dcm_path)
        arr = d.pixel_array.astype(np.float32)
        return arr
    except Exception:
        return None


def _normalize_to_01(x: np.ndarray):
    x = x.astype(np.float32)
    mn = np.nanmin(x)
    mx = np.nanmax(x)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return None
    x = (x - mn) / (mx - mn)
    return x


def _stack3(x2d: np.ndarray):
    return np.stack((x2d,) * 3, axis=-1).astype(np.float32)


def _case_ids_from_dir(path_test: str):
    case_paths = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    case_ids = [os.path.basename(p) for p in case_paths]
    return case_paths, case_ids




## === cell 3
def load_test_T2W_images(path_test):
    array_1, array_2, array_3, array_4, array_5, array_6, array_7 = (
        [] for _ in range(7)
    )
    IMG_PX_SIZE = 150

    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    for case_path in path_cases:
        count = 0
        mri_type = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])
        if len(mri_type) < 4:
            continue
        img_dir = mri_type[3]
        img_path = sorted([f.path for f in os.scandir(img_dir) if f.is_file()])

        for p in img_path:
            arr = _safe_dcm_pixel_array(p)
            if arr is None:
                continue
            if arr.sum() <= 100000:
                continue

            resized_img = resize(
                arr, (IMG_PX_SIZE, IMG_PX_SIZE), preserve_range=True, anti_aliasing=True
            ).astype(np.float32)
            stacked_img = _stack3(resized_img)
            mx = np.max(stacked_img)
            if mx <= 0:
                continue
            stacked_img_normalize = stacked_img / mx

            if stacked_img_normalize.sum() <= 2000:
                continue

            if count == 0:
                array_1.append(stacked_img_normalize)
            elif count == 1:
                array_2.append(stacked_img_normalize)
            elif count == 2:
                array_3.append(stacked_img_normalize)
            elif count == 3:
                array_4.append(stacked_img_normalize)
            elif count == 4:
                array_5.append(stacked_img_normalize)
            elif count == 5:
                array_6.append(stacked_img_normalize)
            elif count == 6:
                array_7.append(stacked_img_normalize)
            count += 1

            if count == 7:
                break

    arrays = []
    for a in (array_1, array_2, array_3, array_4, array_5, array_6, array_7):
        a = np.asarray(a, dtype=np.float32)
        if a.size > 0:
            a = a / max(np.max(a), 1e-6)
        arrays.append(a)

    print("Number of T2 images loaded are ", ", ".join(str(len(a)) for a in arrays))
    return tuple(arrays)




## === cell 4
def load_test_flair_images(path_test):
    array_1, array_2, array_3, array_4, array_5, array_6, array_7 = (
        [] for _ in range(7)
    )
    IMG_PX_SIZE = 150

    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    for case_path in path_cases:
        count = 0
        mri_type = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])
        if len(mri_type) < 1:
            continue
        img_dir = mri_type[0]
        img_path = sorted([f.path for f in os.scandir(img_dir) if f.is_file()])

        for p in img_path:
            arr = _safe_dcm_pixel_array(p)
            if arr is None:
                continue
            if arr.sum() <= 100000:
                continue

            resized_img = resize(
                arr, (IMG_PX_SIZE, IMG_PX_SIZE), preserve_range=True, anti_aliasing=True
            ).astype(np.float32)
            stacked_img = _stack3(resized_img)
            mx = np.max(stacked_img)
            if mx <= 0:
                continue
            stacked_img_normalize = stacked_img / mx

            if stacked_img_normalize.sum() <= 2000:
                continue

            if count == 0:
                array_1.append(stacked_img_normalize)
            elif count == 1:
                array_2.append(stacked_img_normalize)
            elif count == 2:
                array_3.append(stacked_img_normalize)
            elif count == 3:
                array_4.append(stacked_img_normalize)
            elif count == 4:
                array_5.append(stacked_img_normalize)
            elif count == 5:
                array_6.append(stacked_img_normalize)
            elif count == 6:
                array_7.append(stacked_img_normalize)
            count += 1

            if count == 7:
                break

    arrays = []
    for a in (array_1, array_2, array_3, array_4, array_5, array_6, array_7):
        a = np.asarray(a, dtype=np.float32)
        if a.size > 0:
            a = a / max(np.max(a), 1e-6)
        arrays.append(a)

    print("Number of flair images loaded are ", ", ".join(str(len(a)) for a in arrays))
    return tuple(arrays)




## === cell 5
def load_test_T1wce_images(path_test):
    array_1, array_2, array_3, array_4, array_5, array_6, array_7 = (
        [] for _ in range(7)
    )
    IMG_PX_SIZE = 150

    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    for case_path in path_cases:
        count = 0
        mri_type = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])
        if len(mri_type) < 3:
            continue
        img_dir = mri_type[2]
        img_path = sorted([f.path for f in os.scandir(img_dir) if f.is_file()])

        for p in img_path:
            arr = _safe_dcm_pixel_array(p)
            if arr is None:
                continue
            if arr.sum() <= 100000:
                continue

            resized_img = resize(
                arr, (IMG_PX_SIZE, IMG_PX_SIZE), preserve_range=True, anti_aliasing=True
            ).astype(np.float32)
            stacked_img = _stack3(resized_img)
            mx = np.max(stacked_img)
            if mx <= 0:
                continue
            stacked_img_normalize = stacked_img / mx

            if stacked_img_normalize.sum() <= 2000:
                continue

            if count == 0:
                array_1.append(stacked_img_normalize)
            elif count == 1:
                array_2.append(stacked_img_normalize)
            elif count == 2:
                array_3.append(stacked_img_normalize)
            elif count == 3:
                array_4.append(stacked_img_normalize)
            elif count == 4:
                array_5.append(stacked_img_normalize)
            elif count == 5:
                array_6.append(stacked_img_normalize)
            elif count == 6:
                array_7.append(stacked_img_normalize)
            count += 1

            if count == 7:
                break

    arrays = []
    for a in (array_1, array_2, array_3, array_4, array_5, array_6, array_7):
        a = np.asarray(a, dtype=np.float32)
        if a.size > 0:
            a = a / max(np.max(a), 1e-6)
        arrays.append(a)

    print("Number of T1wce images loaded are ", ", ".join(str(len(a)) for a in arrays))
    return tuple(arrays)




## === cell 6
pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6, pixels_7 = (
    load_test_T2W_images(TEST_DIR)
)
pixels_7_f, pixels_8, pixels_9, pixels_10, pixels_11, pixels_12, pixels_13_f = (
    load_test_flair_images(TEST_DIR)
)
pixels_13_t, pixels_14, pixels_15, pixels_16, pixels_17, pixels_18, pixels_19 = (
    load_test_T1wce_images(TEST_DIR)
)

n_cases = len(sample_sub)
for idx, arr in enumerate(
    [
        pixels_1,
        pixels_2,
        pixels_3,
        pixels_4,
        pixels_5,
        pixels_6,
        pixels_7,
        pixels_7_f,
        pixels_8,
        pixels_9,
        pixels_10,
        pixels_11,
        pixels_12,
        pixels_13_f,
        pixels_13_t,
        pixels_14,
        pixels_15,
        pixels_16,
        pixels_17,
        pixels_18,
        pixels_19,
    ],
    start=1,
):
    if arr.shape[0] != n_cases:
        print(
            f"Warning: slice array {idx} has {arr.shape[0]} cases, expected {n_cases}. Will align by truncation."
        )




## === cell 7
def heuristic_predict_proba(x: np.ndarray) -> np.ndarray:
    """
    x: (N, H, W, 3) normalized to ~[0,1]
    returns: (N,) probabilities in [0,1]
    """
    if x.size == 0:
        return np.array([], dtype=np.float32)
    m = x.mean(axis=(1, 2, 3))
    s = x.std(axis=(1, 2, 3))
    z = 2.0 * (m - 0.5) + 1.0 * (s - 0.25)
    p = 1.0 / (1.0 + np.exp(-z))
    return p.astype(np.float32)


def _align_len(arr: np.ndarray, n: int) -> np.ndarray:
    if arr.shape[0] == n:
        return arr
    if arr.shape[0] > n:
        return arr[:n]
    pad_n = n - arr.shape[0]
    pad = np.full((pad_n,) + arr.shape[1:], 0.5, dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=0)


all_pixels = [
    pixels_1,
    pixels_2,
    pixels_3,
    pixels_4,
    pixels_5,
    pixels_6,
    pixels_7,
    pixels_7_f,
    pixels_8,
    pixels_9,
    pixels_10,
    pixels_11,
    pixels_12,
    pixels_13_f,
    pixels_13_t,
    pixels_14,
    pixels_15,
    pixels_16,
    pixels_17,
    pixels_18,
    pixels_19,
]
all_pixels = [_align_len(a, n_cases) for a in all_pixels]



## === cell 8

t2_slices = all_pixels[0:7]
flair_slices = all_pixels[7:14]
t1wce_slices = all_pixels[14:21]

t2_p = [heuristic_predict_proba(x) for x in t2_slices]
flair_p = [heuristic_predict_proba(x) for x in flair_slices]
t1wce_p = [heuristic_predict_proba(x) for x in t1wce_slices]

(
    prediction_1,
    prediction_2,
    prediction_3,
    prediction_4,
    prediction_5,
    prediction_6,
    prediction_7,
) = t2_p
(
    prediction_101,
    prediction_102,
    prediction_103,
    prediction_104,
    prediction_105,
    prediction_106,
    prediction_107,
) = t2_p

(
    prediction_201,
    prediction_202,
    prediction_203,
    prediction_204,
    prediction_205,
    prediction_206,
    prediction_207,
) = flair_p

(
    prediction_301,
    prediction_302,
    prediction_303,
    prediction_304,
    prediction_305,
    prediction_306,
    prediction_307,
) = t1wce_p

(
    prediction_401,
    prediction_402,
    prediction_403,
    prediction_404,
    prediction_405,
    prediction_406,
    prediction_407,
) = t2_p
(
    prediction_501,
    prediction_502,
    prediction_503,
    prediction_504,
    prediction_505,
    prediction_506,
    prediction_507,
) = t2_p




## === cell 9
def create_sub(
    path_test,
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
    p207,
    p301,
    p302,
    p303,
    p304,
    p305,  # original intentionally omitted p306,p307
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
):
    _, case_ids = _case_ids_from_dir(path_test)
    cases_int = [int(c) for c in case_ids]

    n = len(cases_int)
    preds = [
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
        p207,
        p301,
        p302,
        p303,
        p304,
        p305,
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
    ]
    preds = [np.asarray(p, dtype=np.float32).reshape(-1) for p in preds]
    preds = [
        p[:n] if len(p) >= n else np.pad(p, (0, n - len(p)), constant_values=0.5)
        for p in preds
    ]

    prediction = np.mean(np.stack(preds, axis=1), axis=1)
    prediction = np.clip(prediction, 0.0, 1.0)

    df = pd.DataFrame(
        {"BraTS21ID": cases_int, "MGMT_value": prediction.astype(np.float32)}
    )
    return df




## === cell 10
sub_df = create_sub(
    TEST_DIR,
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
    prediction_207,
    prediction_301,
    prediction_302,
    prediction_303,
    prediction_304,
    prediction_305,
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
)

sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(int)
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(int)

sub_df = sample_sub[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(0.5).clip(0.0, 1.0).astype(float)

assert sub_df.shape[0] == sample_sub.shape[0]
assert list(sub_df.columns) == ["BraTS21ID", "MGMT_value"]

sub_df.head()



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2507358918.py in <cell line: 0>()
----> 1 sub_df = create_sub(
      2     TEST_DIR,
      3     prediction_1,
      4     prediction_2,
      5     prediction_3,

/tmp/ipykernel_11/707734733.py in create_sub(path_test, p1, p2, p3, p4, p5, p6, p101, p102, p103, p104, p105, p106, p201, p202, p203, p204, p205, p206, p207, p301, p302, p303, p304, p305, p401, p402, p403, p404, p405, p406, p501, p502, p503, p504, p505, p506)
     40     # Fix: compute one prediction per case; do not overwrite `prediction` in a loop.
     41     _, case_ids = _case_ids_from_dir(path_test)
---> 42     cases_int = [int(c) for c in case_ids]
     43 
     44     # Ensure all prediction vectors align to number of cases

/tmp/ipykernel_11/707734733.py in <listcomp>(.0)
     40     # Fix: compute one prediction per case; do not overwrite `prediction` in a loop.
     41     _, case_ids = _case_ids_from_dir(path_test)
---> 42     cases_int = [int(c) for c in case_ids]
     43 
     44     # Ensure all prediction vectors align to number of cases

ValueError: invalid literal for int() with base 10: 'test'

## === cell 11
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.describe(include="all"))

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4148029191.py in <cell line: 0>()
----> 1 sub_df.to_csv("submission.csv", index=False)
      2 print("Wrote submission.csv with shape:", sub_df.shape)
      3 print(sub_df.describe(include="all"))

NameError: name 'sub_df' is not defined
