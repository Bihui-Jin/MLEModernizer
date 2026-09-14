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


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)



## === cell 1
DATA_ROOT_CANDIDATES = [
    "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification",
    "/kaggle/data/rsna-miccai-brain-tumor-radiogenomic-classification",
    "/kaggle/input",
    "/kaggle/data",
]


def find_existing_path(*parts):
    for root in DATA_ROOT_CANDIDATES:
        p = os.path.join(root, *parts)
        if os.path.exists(p):
            return p
    return None


test = find_existing_path("test")
sample_sub_path = find_existing_path("sample_submission.csv")
train_labels_path = find_existing_path("train_labels.csv")

if test is None:
    raise FileNotFoundError("Could not find test folder under expected Kaggle paths.")
if sample_sub_path is None:
    raise FileNotFoundError(
        "Could not find sample_submission.csv under expected Kaggle paths."
    )

sample_sub = pd.read_csv(sample_sub_path)
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(str).str.zfill(5)
sample_sub.head()




## === cell 2
def _safe_stack_and_norm(img2d: np.ndarray, img_px_size: int = 150) -> np.ndarray:
    resized = resize(
        img2d, (img_px_size, img_px_size), preserve_range=True, anti_aliasing=True
    ).astype(np.float32)
    stacked = np.stack([resized, resized, resized], axis=-1)
    mx = float(np.max(stacked))
    if mx <= 0:
        return stacked  # all zeros
    return stacked / mx


def _load_modality_images(
    path_test: str, modality_index: int, img_px_size: int = 150, max_slices: int = 6
):
    """
    Returns a list of length max_slices, each element is an (N, H, W, 3) float array.
    Keeps the original core logic idea: for each case, pick up to 6 'good' slices.
    """
    arrays = [[] for _ in range(max_slices)]
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])

    for case_path in path_cases:
        count = 0
        mri_type = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])
        if len(mri_type) <= modality_index:
            continue

        img_paths = sorted(
            [f.path for f in os.scandir(mri_type[modality_index]) if f.is_file()]
        )

        for p in img_paths:
            try:
                ds = dicom.dcmread(p)
                px = ds.pixel_array
            except Exception:
                continue

            if px.sum() > 100000:
                stacked_norm = _safe_stack_and_norm(px, img_px_size=img_px_size)
                if stacked_norm.sum() > 2000:
                    arrays[count].append(stacked_norm)
                    count += 1
                    if count >= max_slices:
                        break

    out = []
    for a in arrays:
        a = np.asarray(a, dtype=np.float32)
        if a.size == 0:
            a = np.zeros((0, img_px_size, img_px_size, 3), dtype=np.float32)
        else:
            mx = float(np.max(a))
            if mx > 0:
                a = a / mx
        out.append(a)
    return out


def load_test_flair_images(path_test):
    return _load_modality_images(path_test, modality_index=0)


def load_test_T1W_images(path_test):
    return _load_modality_images(path_test, modality_index=1)


def load_test_T1wce_images(path_test):
    return _load_modality_images(path_test, modality_index=2)


def load_test_T2W_images(path_test):
    return _load_modality_images(path_test, modality_index=3)




## === cell 3
pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = load_test_T2W_images(test)
pixels_7, pixels_8, pixels_9, pixels_10, pixels_11, pixels_12 = load_test_flair_images(
    test
)
pixels_13, pixels_14, pixels_15, pixels_16, pixels_17, pixels_18 = (
    load_test_T1wce_images(test)
)
pixels_19, pixels_20, pixels_21, pixels_22, pixels_23, pixels_24 = load_test_T1W_images(
    test
)

print(
    "Loaded T2 slice counts:",
    [len(x) for x in [pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6]],
)
print(
    "Loaded FLAIR slice counts:",
    [len(x) for x in [pixels_7, pixels_8, pixels_9, pixels_10, pixels_11, pixels_12]],
)
print(
    "Loaded T1wCE slice counts:",
    [
        len(x)
        for x in [pixels_13, pixels_14, pixels_15, pixels_16, pixels_17, pixels_18]
    ],
)
print(
    "Loaded T1w slice counts:",
    [
        len(x)
        for x in [pixels_19, pixels_20, pixels_21, pixels_22, pixels_23, pixels_24]
    ],
)



## === cell 4


def _case_order(path_test: str):
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    ids = [os.path.basename(p) for p in path_cases]
    ids = [str(x).zfill(5) for x in ids]
    return ids


case_ids = _case_order(test)


def _per_case_feature_from_slices(slice_arrays_list):
    """
    slice_arrays_list: list of 6 arrays for a modality, each of shape (N, H, W, 3)
    Returns per-case feature vector length N_cases with robust handling.
    """
    n_cases = max([a.shape[0] for a in slice_arrays_list])
    feat = np.zeros(n_cases, dtype=np.float32)

    for i in range(n_cases):
        vals = []
        for a in slice_arrays_list:
            if i < a.shape[0]:
                vals.append(float(a[i].mean()))
        feat[i] = np.mean(vals) if len(vals) else 0.0
    return feat


feat_t2 = _per_case_feature_from_slices(
    [pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6]
)
feat_flair = _per_case_feature_from_slices(
    [pixels_7, pixels_8, pixels_9, pixels_10, pixels_11, pixels_12]
)
feat_t1ce = _per_case_feature_from_slices(
    [pixels_13, pixels_14, pixels_15, pixels_16, pixels_17, pixels_18]
)
feat_t1 = _per_case_feature_from_slices(
    [pixels_19, pixels_20, pixels_21, pixels_22, pixels_23, pixels_24]
)

raw_score = 0.30 * feat_flair + 0.30 * feat_t2 + 0.25 * feat_t1ce + 0.15 * feat_t1

mu = float(np.mean(raw_score))
sd = float(np.std(raw_score)) if float(np.std(raw_score)) > 1e-6 else 1.0
z = (raw_score - mu) / sd
pred_prob = 1.0 / (1.0 + np.exp(-z))
pred_prob = pred_prob.astype(np.float32)

print(
    "Pred prob summary:",
    float(pred_prob.min()),
    float(pred_prob.mean()),
    float(pred_prob.max()),
)



## === cell 5
pred_df = pd.DataFrame({"BraTS21ID": case_ids, "MGMT_value": pred_prob})
pred_df["BraTS21ID"] = pred_df["BraTS21ID"].astype(str).str.zfill(5)

sub_df = sample_sub[["BraTS21ID"]].merge(pred_df, on="BraTS21ID", how="left")

sub_df["MGMT_value"] = sub_df["MGMT_value"].astype(float).fillna(0.5).clip(0.0, 1.0)

sub_df = sub_df[["BraTS21ID", "MGMT_value"]]
sub_df.head()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2602263537.py in <cell line: 0>()
      1 # Create submission aligned to sample_submission order and required ID format
----> 2 pred_df = pd.DataFrame({"BraTS21ID": case_ids, "MGMT_value": pred_prob})
      3 pred_df["BraTS21ID"] = pred_df["BraTS21ID"].astype(str).str.zfill(5)
      4 
      5 sub_df = sample_sub[["BraTS21ID"]].merge(pred_df, on="BraTS21ID", how="left")

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __init__(self, data, index, columns, dtype, copy)
    776         elif isinstance(data, dict):
    777             # GH#38939 de facto copy defaults to False only in non-dict cases
--> 778             mgr = dict_to_mgr(data, index, columns, dtype=dtype, copy=copy, typ=manager)
    779         elif isinstance(data, ma.MaskedArray):
    780             from numpy.ma import mrecords

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in dict_to_mgr(data, index, columns, dtype, typ, copy)
    501             arrays = [x.copy() if hasattr(x, "dtype") else x for x in arrays]
    502 
--> 503     return arrays_to_mgr(arrays, columns, index, dtype=dtype, typ=typ, consolidate=copy)
    504 
    505 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in arrays_to_mgr(arrays, columns, index, dtype, verify_integrity, typ, consolidate)
    112         # figure out the index, if necessary
    113         if index is None:
--> 114             index = _extract_index(arrays)
    115         else:
    116             index = ensure_index(index)

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in _extract_index(data)
    675         lengths = list(set(raw_lengths))
    676         if len(lengths) > 1:
--> 677             raise ValueError("All arrays must be of the same length")
    678 
    679         if have_dicts:

ValueError: All arrays must be of the same length

## === cell 6
sub_df.to_csv("submission.csv", index=False)

assert os.path.exists("submission.csv")
assert list(sub_df.columns) == ["BraTS21ID", "MGMT_value"]
assert sub_df["BraTS21ID"].dtype == object
assert sub_df["MGMT_value"].between(0, 1).all()

print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head(3).to_string(index=False))

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/838614256.py in <cell line: 0>()
      1 # Write submission
----> 2 sub_df.to_csv("submission.csv", index=False)
      3 
      4 # Basic sanity checks
      5 assert os.path.exists("submission.csv")

NameError: name 'sub_df' is not defined
