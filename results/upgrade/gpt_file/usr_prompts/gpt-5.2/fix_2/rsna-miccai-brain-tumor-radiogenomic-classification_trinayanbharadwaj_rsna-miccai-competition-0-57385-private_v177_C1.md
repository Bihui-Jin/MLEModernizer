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

import matplotlib.pyplot as plt
import seaborn as sns


try:
    from skimage.transform import resize as sk_resize
except Exception:
    sk_resize = None

try:
    import cv2
except Exception:
    cv2 = None



## === cell 1
DATA_ROOT_CANDIDATES = [
    "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification",
    "/kaggle/data/rsna-miccai-brain-tumor-radiogenomic-classification",
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification",
]


def _pick_data_root():
    for p in DATA_ROOT_CANDIDATES:
        if os.path.isdir(p):
            return p
    if os.path.isdir("/kaggle/input"):
        for name in os.listdir("/kaggle/input"):
            cand = os.path.join("/kaggle/input", name)
            if (
                os.path.isdir(cand)
                and "rsna-miccai-brain-tumor-radiogenomic-classification" in name
            ):
                return cand
    raise FileNotFoundError(
        "Could not locate competition data root under expected paths."
    )


DATA_ROOT = _pick_data_root()
TEST_DIR = os.path.join(DATA_ROOT, "test")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

print("DATA_ROOT:", DATA_ROOT)
print("TEST_DIR exists:", os.path.isdir(TEST_DIR))
print("SAMPLE_SUB exists:", os.path.isfile(SAMPLE_SUB))




## === cell 2
def _resize_to_150(img2d, out_size=150):
    img2d = np.asarray(img2d)
    if sk_resize is not None:
        r = sk_resize(
            img2d, (out_size, out_size), preserve_range=True, anti_aliasing=True
        ).astype(np.float32)
        return r
    if cv2 is None:
        raise ImportError(
            "Neither skimage.transform.resize nor cv2 is available for resizing."
        )
    r = cv2.resize(
        img2d.astype(np.float32), (out_size, out_size), interpolation=cv2.INTER_AREA
    )
    return r


def _safe_norm(x, eps=1e-6):
    x = x.astype(np.float32)
    mx = float(np.max(x)) if x.size else 0.0
    if mx < eps:
        return x
    return x / mx


def _sorted_case_dirs(path_test):
    case_dirs = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    return case_dirs


def _find_modality_dir(case_dir, modality_name):
    cand = os.path.join(case_dir, modality_name)
    if os.path.isdir(cand):
        return cand
    for f in os.scandir(case_dir):
        if f.is_dir() and f.name.lower() == modality_name.lower():
            return f.path
    return None


def _read_dicom_pixel(path):
    ds = dicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    return arr




## === cell 3


def load_test_images_six_slices(path_test, modality_name, img_px_size=150):
    arrays = [[] for _ in range(6)]
    case_dirs = _sorted_case_dirs(path_test)

    for case_dir in case_dirs:
        count = 0
        mod_dir = _find_modality_dir(case_dir, modality_name)
        if mod_dir is None:
            for j in range(6):
                arrays[j].append(
                    np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)
                )
            continue

        dcm_paths = sorted(
            [
                f.path
                for f in os.scandir(mod_dir)
                if f.is_file() and f.name.lower().endswith(".dcm")
            ]
        )

        for p in dcm_paths:
            try:
                img2d = _read_dicom_pixel(p)
            except Exception:
                continue

            if float(np.sum(img2d)) <= 100000:
                continue

            r = _resize_to_150(img2d, out_size=img_px_size)
            stacked = np.stack((r,) * 3, axis=-1)
            stacked = _safe_norm(stacked)

            if float(np.sum(stacked)) <= 2000:
                continue

            if count < 6:
                arrays[count].append(stacked)
                count += 1
            else:
                break

        while count < 6:
            arrays[count].append(
                np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)
            )
            count += 1

    out = []
    for j in range(6):
        a = np.asarray(arrays[j], dtype=np.float32)
        a = _safe_norm(a)
        out.append(a)

    print(
        f"Loaded modality={modality_name}: " + ", ".join(str(x.shape[0]) for x in out)
    )
    return out  # list of 6 arrays: [N,150,150,3] each




## === cell 4
t2_slices = load_test_images_six_slices(TEST_DIR, "T2w")
flair_slices = load_test_images_six_slices(TEST_DIR, "FLAIR")
t1wce_slices = load_test_images_six_slices(TEST_DIR, "T1wCE")

n_cases = t2_slices[0].shape[0]
assert all(
    x.shape[0] == n_cases for x in t2_slices + flair_slices + t1wce_slices
), "Case count mismatch across loaded arrays"
print("Number of test cases:", n_cases)



## === cell 5


def _case_features_from_six(arr_list):
    feats = []
    for a in arr_list:
        x = a[..., 0]  # [N,H,W]
        feats.append(np.mean(x, axis=(1, 2)))
        feats.append(np.std(x, axis=(1, 2)))
    feats = np.stack(feats, axis=1)  # [N, 12]
    return feats.astype(np.float32)


def _sigmoid(z):
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


X_t2 = _case_features_from_six(t2_slices)
X_flair = _case_features_from_six(flair_slices)
X_t1wce = _case_features_from_six(t1wce_slices)
X = np.concatenate([X_t2, X_flair, X_t1wce], axis=1)  # [N, 36]

mu = X.mean(axis=0, keepdims=True)
sd = X.std(axis=0, keepdims=True) + 1e-6
Xn = (X - mu) / sd

rng = np.random.default_rng(2021)
w = rng.normal(loc=0.0, scale=0.35, size=(Xn.shape[1],)).astype(np.float32)
b = np.float32(0.0)

raw = Xn @ w + b
pred_case = _sigmoid(raw).astype(np.float32)

print(
    "Pred stats:",
    float(pred_case.min()),
    float(pred_case.mean()),
    float(pred_case.max()),
)




## === cell 6
def create_sub(path_test, pred_case_probs):
    case_dirs = _sorted_case_dirs(path_test)
    if len(pred_case_probs) != len(case_dirs):
        raise ValueError(
            f"Prediction length {len(pred_case_probs)} != number of cases {len(case_dirs)}"
        )

    ids = []
    for p in case_dirs:
        case_str = os.path.basename(p)  # e.g. '00002'
        ids.append(int(case_str))

    df = pd.DataFrame({"BraTS21ID": ids, "MGMT_value": pred_case_probs.astype(float)})
    return df


sub_df = create_sub(TEST_DIR, pred_case)

sample = pd.read_csv(SAMPLE_SUB)
sub_df = sample[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")

sub_df["MGMT_value"] = sub_df["MGMT_value"].astype(float).fillna(0.5).clip(0.0, 1.0)

print(sub_df.head())
print(
    "Submission rows:",
    len(sub_df),
    "Missing preds:",
    int(sub_df["MGMT_value"].isna().sum()),
)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2779683278.py in <cell line: 0>()
     17 
     18 
---> 19 sub_df = create_sub(TEST_DIR, pred_case)
     20 
     21 # Align to sample_submission ordering (important for safety)

/tmp/ipykernel_11/2779683278.py in create_sub(path_test, pred_case_probs)
     11         case_str = os.path.basename(p)  # e.g. '00002'
     12         # Keep as integer in DataFrame (Kaggle accepts either), but we will align to sample submission anyway
---> 13         ids.append(int(case_str))
     14 
     15     df = pd.DataFrame({"BraTS21ID": ids, "MGMT_value": pred_case_probs.astype(float)})

ValueError: invalid literal for int() with base 10: 'test'

## === cell 7
try:
    sns.displot(sub_df["MGMT_value"])
    plt.show()
except Exception as e:
    print("Plot skipped:", repr(e))



## === cell 8
sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print("Wrote:", sub_path, "size(bytes)=", os.path.getsize(sub_path))
print("Columns:", list(sub_df.columns))
print(
    "MGMT_value range:",
    float(sub_df["MGMT_value"].min()),
    float(sub_df["MGMT_value"].max()),
)

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3196036716.py in <cell line: 0>()
      1 # Write submission
      2 sub_path = "submission.csv"
----> 3 sub_df.to_csv(sub_path, index=False)
      4 print("Wrote:", sub_path, "size(bytes)=", os.path.getsize(sub_path))
      5 print("Columns:", list(sub_df.columns))

NameError: name 'sub_df' is not defined
