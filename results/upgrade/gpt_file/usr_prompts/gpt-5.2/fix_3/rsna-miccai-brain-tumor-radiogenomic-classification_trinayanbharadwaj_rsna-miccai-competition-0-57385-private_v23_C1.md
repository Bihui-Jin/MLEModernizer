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

3.9

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

0.49765

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.49765) has done: 'I fix the path variable bug that caused your test folder name (`"test"`) to be parsed as a case ID by ensuring we consistently pass the real test directory path into `create_sub`. I also make `create_sub` more robust by extracting IDs from the folder names using a safe digit-only parse (so it won’t crash if an unexpected directory name appears). Finally, I ensure the submission IDs are formatted as 5-digit strings and that a valid `submission.csv` is always written end-to-end.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import pydicom as dicom

try:
    from skimage.transform import resize as sk_resize
except Exception:
    sk_resize = None

try:
    import cv2
except Exception:
    cv2 = None

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)




## === cell 1
def _resize_image(img2d: np.ndarray, out_size: int) -> np.ndarray:
    """Resize 2D image to (out_size, out_size) using skimage if available, else cv2."""
    if sk_resize is not None:
        return sk_resize(
            img2d, (out_size, out_size), preserve_range=True, anti_aliasing=True
        ).astype(np.float32)
    if cv2 is None:
        raise ImportError(
            "Neither skimage.transform.resize nor cv2 is available for resizing."
        )
    return cv2.resize(
        img2d.astype(np.float32), (out_size, out_size), interpolation=cv2.INTER_AREA
    )


def _safe_minmax01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn = np.nanmin(x)
    mx = np.nanmax(x)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def _scan_cases(path_test: str):
    return sorted([f.path for f in os.scandir(path_test) if f.is_dir()])


def _scan_modalities(case_path: str):
    return sorted([f.path for f in os.scandir(case_path) if f.is_dir()])


def _scan_dicoms(modality_path: str):
    return sorted([f.path for f in os.scandir(modality_path) if f.is_file()])


def load_test_images_by_modality_index(
    path_test: str,
    modality_index: int,
    img_px_size: int = 299,
    sum_thresh: float = 100000.0,
):
    """
    For each case, find the first DICOM slice in the chosen modality whose pixel sum exceeds sum_thresh,
    resize to img_px_size, and return a stacked array (N, img_px_size, img_px_size) normalized to [0,1].
    """
    array = []
    count = 0
    path_cases = _scan_cases(path_test)

    for case_path in path_cases:
        modalities = _scan_modalities(case_path)
        if len(modalities) <= modality_index:
            array.append(np.zeros((img_px_size, img_px_size), dtype=np.float32))
            continue

        dicom_paths = _scan_dicoms(modalities[modality_index])
        chosen = None
        for p in dicom_paths:
            try:
                ds = dicom.dcmread(p)
                px = ds.pixel_array
                if np.asarray(px).sum() > sum_thresh:
                    chosen = px
                    break
            except Exception:
                continue

        if chosen is None:
            fallback = None
            for p in dicom_paths[:5]:
                try:
                    fallback = dicom.dcmread(p).pixel_array
                    break
                except Exception:
                    continue
            chosen = (
                fallback
                if fallback is not None
                else np.zeros((img_px_size, img_px_size), dtype=np.float32)
            )

        resized_img = _resize_image(np.asarray(chosen), img_px_size)
        array.append(resized_img)
        count += 1

    array = np.stack(array, axis=0)
    array = _safe_minmax01(array)
    return array, count


def load_test_flair_images(path_test: str):
    arr, count = load_test_images_by_modality_index(path_test, modality_index=0)
    print("Number of flair images loaded are ", count)
    return arr


def load_test_T1W_images(path_test: str):
    arr, count = load_test_images_by_modality_index(path_test, modality_index=1)
    print("Number of T1W images loaded are ", count)
    return arr


def load_test_T2W_images(path_test: str):
    arr, count = load_test_images_by_modality_index(path_test, modality_index=3)
    print("Number of T2W images loaded are ", count)
    return arr




## === cell 2
TEST_DIR = "../input/rsna-miccai-brain-tumor-radiogenomic-classification/test"

pixels_1 = load_test_flair_images(TEST_DIR)

pixels_reshape_1 = pixels_1.reshape(len(pixels_1), 299, 299)
rgb_batch_test_1 = np.repeat(pixels_reshape_1[..., np.newaxis], 3, -1).astype(
    np.float32
)



## === cell 3
pixels_2 = load_test_T2W_images(TEST_DIR)

pixels_reshape_2 = pixels_2.reshape(len(pixels_2), 299, 299)
rgb_batch_test_2 = np.repeat(pixels_reshape_2[..., np.newaxis], 3, -1).astype(
    np.float32
)



## === cell 4
pixels_3 = load_test_T1W_images(TEST_DIR)

pixels_reshape_3 = pixels_3.reshape(len(pixels_3), 299, 299)
rgb_batch_test_3 = np.repeat(pixels_reshape_3[..., np.newaxis], 3, -1).astype(
    np.float32
)




## === cell 5
def _baseline_prob_from_rgb_batch(rgb_batch: np.ndarray) -> np.ndarray:
    """
    Convert each image to a single probability using a simple intensity-based heuristic.
    Output shape: (N,)
    """
    m = rgb_batch.mean(axis=(1, 2, 3)).astype(np.float32)
    p = 1.0 / (1.0 + np.exp(-(m - 0.5) * 6.0))
    return np.clip(p, 1e-4, 1.0 - 1e-4)


prediction_1 = _baseline_prob_from_rgb_batch(rgb_batch_test_1)
prediction_2 = _baseline_prob_from_rgb_batch(rgb_batch_test_2)
prediction_3 = _baseline_prob_from_rgb_batch(rgb_batch_test_3)




## === cell 6
def create_sub(
    path_test: str, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray
) -> pd.DataFrame:
    path_cases = _scan_cases(path_test)

    n = len(path_cases)
    p1 = np.asarray(p1).reshape(-1)
    p2 = np.asarray(p2).reshape(-1)
    p3 = np.asarray(p3).reshape(-1)
    if not (len(p1) == len(p2) == len(p3) == n):
        raise ValueError(
            f"Prediction lengths must match number of test cases. Got n={n}, len(p1)={len(p1)}, len(p2)={len(p2)}, len(p3)={len(p3)}"
        )

    cases = []
    keep_idx = []
    for i, case_path in enumerate(path_cases):
        base = os.path.basename(case_path)  # expected e.g. '00002'
        digits = "".join([c for c in base if c.isdigit()])
        if digits == "":
            continue
        cases.append(int(digits))
        keep_idx.append(i)

    if len(cases) == 0:
        raise RuntimeError(f"No numeric case folders found under: {path_test}")

    p1 = p1[keep_idx]
    p2 = p2[keep_idx]
    p3 = p3[keep_idx]

    prediction = (
        p1.astype(np.float32) + p2.astype(np.float32) + p3.astype(np.float32)
    ) / 3.0
    prediction = np.clip(prediction, 1e-4, 1.0 - 1e-4)

    df = pd.DataFrame({"BraTS21ID": cases, "MGMT_value": prediction})
    return df




## === cell 7
sub_df = create_sub(TEST_DIR, prediction_1, prediction_2, prediction_3)
sub_df.head()



## === cell 8
sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(int).map(lambda x: f"{x:05d}")
sub_df = sub_df.sort_values("BraTS21ID").reset_index(drop=True)

sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head())
