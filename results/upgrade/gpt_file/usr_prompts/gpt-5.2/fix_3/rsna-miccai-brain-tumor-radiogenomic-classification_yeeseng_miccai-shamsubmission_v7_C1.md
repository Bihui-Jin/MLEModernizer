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

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

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

0.55647

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I remove the dependency on the missing `../input/miccai-testsubmissions/submission02.csv` file (the root cause of the crash) and instead build predictions directly for the provided test set. To keep the core approach minimal and stable, the script generate the required submission by filling every test `BraTS21ID` with a constant probability (0.5), which yields a valid baseline and ensures the pipeline runs end-to-end. I also fix path robustness by auto-detecting whether the dataset is under `/kaggle/input/...` or the relative `../input/...` location, and ensure IDs are properly formatted as 5-digit strings. Finally, it always write a `submission.csv` with the exact required columns.'
- What this solution (achieved 0.55647) has done: 'Your current 0.5 score comes from constant predictions, so to move the AUC upward (toward a meaningful target) with minimal logic change, we replace the constant with a simple, legitimate heuristic derived from the test images themselves. We keep the “no ML training” core approach, but compute per-subject intensity statistics from a small, fixed set of DICOM slices (middle slices) across the four MRI sequences and map that to a probability via a sigmoid. This preserves evaluation semantics (probabilities for each BraTS21ID) while adding real signal, and it still runs within the time limit by limiting slices read per series and avoiding heavy dependencies. We also keep the sample_submission alignment to guarantee correct row order and formatting.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd

try:
    import pydicom
except Exception as e:
    raise ImportError(
        "This script requires 'pydicom' to read DICOM files. "
        "It is typically available on Kaggle for this competition environment."
    ) from e

CANDIDATE_ROOTS = [
    "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification",
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification",
    "/kaggle/data/rsna-miccai-brain-tumor-radiogenomic-classification",
    "../data/rsna-miccai-brain-tumor-radiogenomic-classification",
]
DATA_ROOT = next((p for p in CANDIDATE_ROOTS if os.path.exists(p)), None)
if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not find rsna-miccai-brain-tumor-radiogenomic-classification dataset folder in expected locations."
    )

TEST_DIR = os.path.join(DATA_ROOT, "test")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

print("Using DATA_ROOT:", DATA_ROOT)
print("Test dir exists:", os.path.exists(TEST_DIR))
print("Sample submission exists:", os.path.exists(SAMPLE_SUB_PATH))




## === cell 1

SEQUENCES = ["FLAIR", "T1w", "T1wCE", "T2w"]


def _safe_dcm_mean_std(dcm_path: str):
    """Read DICOM safely and return (mean, std) in float; return (nan, nan) if unreadable."""
    try:
        ds = pydicom.dcmread(dcm_path, stop_before_pixels=False, force=True)
        arr = ds.pixel_array.astype(np.float32)
        if arr.size == 0:
            return np.nan, np.nan
        return float(np.mean(arr)), float(np.std(arr))
    except Exception:
        return np.nan, np.nan


def subject_feature_from_dir(subject_dir: str, slices_per_seq: int = 5):
    """
    Compute simple per-subject features from test folder:
    for each sequence, take a few evenly spaced slices around the middle and aggregate mean/std.
    Returns a single scalar score (float).
    """
    feats = []
    for seq in SEQUENCES:
        seq_dir = os.path.join(subject_dir, seq)
        if not os.path.isdir(seq_dir):
            feats.extend([np.nan, np.nan])
            continue

        files = sorted(glob.glob(os.path.join(seq_dir, "*.dcm")))
        if len(files) == 0:
            feats.extend([np.nan, np.nan])
            continue

        n = len(files)
        if slices_per_seq <= 1:
            idxs = [n // 2]
        else:
            lo = max(0, int(0.25 * (n - 1)))
            hi = max(lo, int(0.75 * (n - 1)))
            if hi == lo:
                idxs = [n // 2]
            else:
                idxs = (
                    np.linspace(lo, hi, num=slices_per_seq).round().astype(int).tolist()
                )

        means, stds = [], []
        for i in idxs:
            m, s = _safe_dcm_mean_std(files[i])
            means.append(m)
            stds.append(s)

        feats.append(
            float(np.nanmean(means)) if np.isfinite(np.nanmean(means)) else np.nan
        )
        feats.append(
            float(np.nanmean(stds)) if np.isfinite(np.nanmean(stds)) else np.nan
        )

    feats = np.array(feats, dtype=np.float32)

    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    means = feats[0::2]
    stds = feats[1::2]
    score = float(0.8 * np.sum(stds) + 0.2 * np.sum(means))

    return score


def sigmoid(x: float):
    x = float(np.clip(x, -50.0, 50.0))
    return 1.0 / (1.0 + np.exp(-x))




## === cell 2
listOfStudyPaths = sorted(glob.glob(os.path.join(TEST_DIR, "*")))
listOfStudies = [os.path.basename(p) for p in listOfStudyPaths if os.path.isdir(p)]

if len(listOfStudies) == 0:
    raise RuntimeError(f"No test study folders found under: {TEST_DIR}")

listOfStudies = [str(s).zfill(5) for s in listOfStudies]

raw_scores = []
for sid in listOfStudies:
    subj_dir = os.path.join(TEST_DIR, sid)
    raw_scores.append(subject_feature_from_dir(subj_dir, slices_per_seq=5))

raw_scores = np.array(raw_scores, dtype=np.float32)

med = float(np.median(raw_scores))
mad = float(np.median(np.abs(raw_scores - med))) + 1e-6  # avoid divide-by-zero
z = (raw_scores - med) / (1.4826 * mad)

probs = np.array([sigmoid(0.7 * float(v)) for v in z], dtype=np.float32)

submissionDF = pd.DataFrame(
    {"BraTS21ID": listOfStudies, "MGMT_value": probs.astype(np.float32)}
)

if os.path.exists(SAMPLE_SUB_PATH):
    sampleDF = pd.read_csv(SAMPLE_SUB_PATH, dtype={"BraTS21ID": str})
    sampleDF["BraTS21ID"] = sampleDF["BraTS21ID"].astype(str).str.zfill(5)
    submissionDF["BraTS21ID"] = submissionDF["BraTS21ID"].astype(str).str.zfill(5)
    submissionDF = sampleDF[["BraTS21ID"]].merge(
        submissionDF, on="BraTS21ID", how="left"
    )
    submissionDF["MGMT_value"] = (
        submissionDF["MGMT_value"].astype(np.float32).fillna(0.5)
    )

submission_path = "submission.csv"
submissionDF.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(submissionDF.head())
print("Rows:", len(submissionDF), "Cols:", submissionDF.shape[1])
print("Columns:", submissionDF.columns.tolist())
print("MGMT_value summary:", submissionDF["MGMT_value"].describe())
