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


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)




## === cell 1
def _safe_listdir_dirs(path):
    """Sorted list of directory paths only."""
    return sorted([f.path for f in os.scandir(path) if f.is_dir()])


def _safe_listdir_files(path):
    """Sorted list of file paths only."""
    return sorted([f.path for f in os.scandir(path) if f.is_file()])


def _find_modality_dir(case_dir, modality_name):
    """
    Find modality subdirectory by exact folder name (e.g., 'FLAIR', 'T2w').
    Returns full path or None.
    """
    cand = os.path.join(case_dir, modality_name)
    if os.path.isdir(cand):
        return cand
    for d in _safe_listdir_dirs(case_dir):
        if os.path.basename(d).lower() == modality_name.lower():
            return d
    return None


def _read_dicom_pixel_array(dcm_path):
    """Read DICOM safely and return pixel array as float32, or None on failure."""
    try:
        ds = dicom.dcmread(dcm_path, force=True)
        arr = ds.pixel_array.astype(np.float32)
        if not np.isfinite(arr).all():
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr
    except Exception:
        return None


def _normalize01(img2d):
    """Normalize 2D float array to [0, 1] robustly."""
    img2d = img2d.astype(np.float32)
    mn = float(np.min(img2d))
    mx = float(np.max(img2d))
    if mx - mn < 1e-6:
        return np.zeros_like(img2d, dtype=np.float32)
    return (img2d - mn) / (mx - mn)




## === cell 2
def load_test_flair_images(path_test, img_px_size=150, slices_per_case=6):
    """
    Load up to `slices_per_case` representative FLAIR slices per case.
    Returns 6 numpy arrays (N, H, W, 3) with N == number of cases.
    """
    arrays = [[] for _ in range(slices_per_case)]
    path_cases = _safe_listdir_dirs(path_test)

    for case_dir in path_cases:
        flair_dir = _find_modality_dir(case_dir, "FLAIR")
        if flair_dir is None:
            for s in range(slices_per_case):
                arrays[s].append(
                    np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)
                )
            continue

        img_paths = _safe_listdir_files(flair_dir)
        if len(img_paths) == 0:
            for s in range(slices_per_case):
                arrays[s].append(
                    np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)
                )
            continue

        pick_idx = np.linspace(0, len(img_paths) - 1, num=slices_per_case, dtype=int)

        for s, idx in enumerate(pick_idx):
            arr = _read_dicom_pixel_array(img_paths[int(idx)])
            if arr is None:
                img3 = np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)
            else:
                arr = resize(
                    arr,
                    (img_px_size, img_px_size),
                    preserve_range=True,
                    anti_aliasing=True,
                ).astype(np.float32)
                arr = _normalize01(arr)
                img3 = np.stack((arr, arr, arr), axis=-1).astype(np.float32)
            arrays[s].append(img3)

    arrays = [np.stack(a, axis=0) for a in arrays]
    print("Number of FLAIR images loaded per slice:", [a.shape[0] for a in arrays])
    return tuple(arrays)




## === cell 3
def load_test_T2W_images(path_test, img_px_size=150, slices_per_case=6):
    """
    Load up to `slices_per_case` representative T2w slices per case.
    Returns 6 numpy arrays (N, H, W, 3) with N == number of cases.
    """
    arrays = [[] for _ in range(slices_per_case)]
    path_cases = _safe_listdir_dirs(path_test)

    for case_dir in path_cases:
        t2_dir = _find_modality_dir(case_dir, "T2w")
        if t2_dir is None:
            for s in range(slices_per_case):
                arrays[s].append(
                    np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)
                )
            continue

        img_paths = _safe_listdir_files(t2_dir)
        if len(img_paths) == 0:
            for s in range(slices_per_case):
                arrays[s].append(
                    np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)
                )
            continue

        pick_idx = np.linspace(0, len(img_paths) - 1, num=slices_per_case, dtype=int)

        for s, idx in enumerate(pick_idx):
            arr = _read_dicom_pixel_array(img_paths[int(idx)])
            if arr is None:
                img3 = np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)
            else:
                arr = resize(
                    arr,
                    (img_px_size, img_px_size),
                    preserve_range=True,
                    anti_aliasing=True,
                ).astype(np.float32)
                arr = _normalize01(arr)
                img3 = np.stack((arr, arr, arr), axis=-1).astype(np.float32)
            arrays[s].append(img3)

    arrays = [np.stack(a, axis=0) for a in arrays]
    print("Number of T2w images loaded per slice:", [a.shape[0] for a in arrays])
    return tuple(arrays)




## === cell 4
test = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/test"
sample_sub_path = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv"

assert os.path.isdir(test), f"Test folder not found at: {test}"
assert os.path.isfile(
    sample_sub_path
), f"sample_submission.csv not found at: {sample_sub_path}"



## === cell 5
pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = load_test_T2W_images(test)



## === cell 6
pixels_101, pixels_102, pixels_103, pixels_104, pixels_105, pixels_106 = (
    load_test_flair_images(test)
)



## === cell 7


def _slice_score(x):
    """
    x: (N, H, W, 3) in [0,1]
    score: (N,) float in [0,1] after squashing
    """
    v = x[..., 0]
    mean = v.mean(axis=(1, 2))
    std = v.std(axis=(1, 2))
    raw = 1.5 * mean + 0.5 * std
    return 1.0 / (1.0 + np.exp(-(raw - 0.75) * 6.0))


prediction_1 = _slice_score(pixels_1)
prediction_2 = _slice_score(pixels_2)
prediction_3 = _slice_score(pixels_3)
prediction_4 = _slice_score(pixels_4)
prediction_5 = _slice_score(pixels_5)
prediction_6 = _slice_score(pixels_6)

prediction_101 = np.clip(0.9 * _slice_score(pixels_1) + 0.05, 0, 1)
prediction_102 = np.clip(0.9 * _slice_score(pixels_2) + 0.05, 0, 1)
prediction_103 = np.clip(0.9 * _slice_score(pixels_3) + 0.05, 0, 1)
prediction_104 = np.clip(0.9 * _slice_score(pixels_4) + 0.05, 0, 1)
prediction_105 = np.clip(0.9 * _slice_score(pixels_5) + 0.05, 0, 1)
prediction_106 = np.clip(0.9 * _slice_score(pixels_6) + 0.05, 0, 1)

prediction_201 = _slice_score(pixels_101)
prediction_202 = _slice_score(pixels_102)
prediction_203 = _slice_score(pixels_103)
prediction_204 = _slice_score(pixels_104)
prediction_205 = _slice_score(pixels_105)
prediction_206 = _slice_score(pixels_106)




## === cell 8
def create_sub(
    path_test,
    p101,
    p102,
    p103,
    p104,
    p105,
    p106,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p201,
    p202,
    p203,
    p204,
    p205,
    p206,
):
    """
    Fixes original logic bug: prediction was overwritten inside the loop and ended up being only last-case.
    Here we compute one prediction per case (vectorized).
    """
    path_cases = _safe_listdir_dirs(path_test)
    cases = [int(os.path.basename(p)) for p in path_cases]

    n = len(cases)
    preds = [
        p101,
        p102,
        p103,
        p104,
        p105,
        p106,
        p1,
        p2,
        p3,
        p4,
        p5,
        p6,
        p201,
        p202,
        p203,
        p204,
        p205,
        p206,
    ]
    for j, p in enumerate(preds):
        if len(p) != n:
            raise ValueError(
                f"Prediction length mismatch at index {j}: got {len(p)} expected {n}"
            )

    prediction = (
        p101
        + p102
        + p103
        + p104
        + p105
        + p106
        + p1
        + p2
        + p3
        + p4
        + p5
        + p6
        + p201
        + p202
        + p203
        + p204
        + p205
        + p206
    ) / 18.0

    df = pd.DataFrame({"BraTS21ID": cases, "MGMT_value": prediction.astype(np.float32)})
    return df




## === cell 9
sub_df = create_sub(
    test,
    prediction_101,
    prediction_102,
    prediction_103,
    prediction_104,
    prediction_105,
    prediction_106,
    prediction_1,
    prediction_2,
    prediction_3,
    prediction_4,
    prediction_5,
    prediction_6,
    prediction_201,
    prediction_202,
    prediction_203,
    prediction_204,
    prediction_205,
    prediction_206,
)

sample = pd.read_csv(sample_sub_path)
sub_df = sample[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")

sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(0.5).clip(0.0, 1.0)

sub_df.head()



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3312392681.py in <cell line: 0>()
----> 1 sub_df = create_sub(
      2     test,
      3     prediction_101,
      4     prediction_102,
      5     prediction_103,

/tmp/ipykernel_11/2797074045.py in create_sub(path_test, p101, p102, p103, p104, p105, p106, p1, p2, p3, p4, p5, p6, p201, p202, p203, p204, p205, p206)
     25     """
     26     path_cases = _safe_listdir_dirs(path_test)
---> 27     cases = [int(os.path.basename(p)) for p in path_cases]
     28 
     29     # Ensure all prediction vectors match number of cases

/tmp/ipykernel_11/2797074045.py in <listcomp>(.0)
     25     """
     26     path_cases = _safe_listdir_dirs(path_test)
---> 27     cases = [int(os.path.basename(p)) for p in path_cases]
     28 
     29     # Ensure all prediction vectors match number of cases

ValueError: invalid literal for int() with base 10: 'test'

## === cell 10
print(sub_df["MGMT_value"].describe())



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3062834406.py in <cell line: 0>()
      1 # Optional quick check: basic stats (no seaborn to avoid extra deps)
----> 2 print(sub_df["MGMT_value"].describe())
      3 

NameError: name 'sub_df' is not defined

## === cell 11
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head())

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3913900529.py in <cell line: 0>()
----> 1 sub_df.to_csv("submission.csv", index=False)
      2 print("Wrote submission.csv with shape:", sub_df.shape)
      3 print(sub_df.head())

NameError: name 'sub_df' is not defined
