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

0.47294

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I tighten the case‑folder discovery to only keep numeric directories, preventing the “test” string from being parsed as an ID. This resolves the ValueError and cascade errors, allowing the script to compute dummy predictions, build a proper DataFrame, and write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.47294) has done: 'I replace the constant dummy prediction with a simple decreasing sequence (high‑to‑low values). This keeps the overall pipeline unchanged while likely reducing the AUC a bit, moving the score from 0.5 toward the low target of ‑1.0 (higher‑is‑better, so we want a lower value). No other logic is altered.'
- What this solution (achieved 0.44059) has done: 'I adjust the dummy prediction generator to output a sharper high‑to‑low step (first half = 1, second half = 0) instead of a smooth linear decline. This stronger anti‑ordering can push the AUC further down, moving the score closer to the low target while keeping the rest of the workflow unchanged.'
- What this solution (achieved 0.55941) has done: 'I invert the dummy prediction pattern so the first half of cases receive 0 and the second half receive 1. This reverses the ordering used previously, which should further anti‑correlate the predictions with any monotonic label trend and therefore push the AUC lower, moving the score closer to the low target ‑1.0 while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.44059) has done: 'I invert the dummy prediction pattern so the first half of cases receive 1 and the second half receive 0, creating a high‑to‑low ordering that is more anti‑correlated with any monotonic label trend and therefore pushes the AUC lower toward the target –1.0. The rest of the pipeline stays unchanged, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.44059) has done: 'I keep the overall pipeline unchanged but decide the dummy prediction direction based on the observed correlation between subject IDs and the training labels. If higher IDs tend to have a positive label, I output a decreasing (high‑to‑low) sequence; if the correlation is negative I output an increasing (low‑to‑high) sequence. This choice should produce predictions that are more anti‑correlated with any monotonic trend in the data, thereby lowering the AUC and moving the score closer to the low target ‑1.0 while still writing a valid `submission.csv`.'
- What this solution (achieved 0.44059) has done: 'I keep the overall pipeline unchanged but add a small evaluation on the training data to choose the dummy‑prediction ordering that gives the lower AUC (i.e., the more anti‑correlated pattern). This deterministic choice should push the test‑set AUC farther down, moving the score closer to the target ‑1 while still producing a valid `submission.csv`. The core logic, data handling and file paths remain the same.'
- What this solution (achieved 0.47294) has done: 'I replace the simple step‑wise dummy generator with a monotonic prediction that follows the ordering of the numeric case IDs. By assigning a smoothly decreasing (or increasing) probability according to the ID rank, we create a stronger anti‑correlation with any monotonic trend in the data, which should reduce the AUC further and move the score closer to the target –1.0. The rest of the pipeline (file handling, submission creation) stays unchanged.'
- What this solution (achieved 0.44059) has done: 'I keep the overall workflow unchanged but replace the smooth monotonic prediction with a sharper step‑wise version (first half = 1, second half = 0) while still selecting the orientation that gives the lower AUC on the training set. This stronger anti‑correlated pattern should push the validation AUC farther below the current 0.473 and move the score closer to the low target ‑1.0.'
- What this solution (achieved 0.47294) has done: 'I replace the use of the binary step‑wise dummy predictions with the smoother monotonic predictions that were already evaluated on the training set. Since the orientation (high_first vs low_first) is chosen to give the lower AUC on the training data, using the same monotonic function for the test set should produce a more anti‑correlated ranking and therefore a lower AUC, moving the score closer to the low target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from skimage.transform import resize  # kept for potential future use
from sklearn.metrics import (
    roc_auc_score,
)  # for selecting the more anti‑correlated pattern




## === cell 1
TEST_PATH = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/test"


def monotonic_predictions(ids, high_first=True):
    """
    Generate smooth monotonic predictions based on the rank of numeric IDs.
    If high_first=True → larger IDs receive lower probabilities (decreasing).
    If high_first=False → larger IDs receive higher probabilities (increasing).
    """
    ids = np.asarray(ids, dtype=float)
    n = len(ids)
    if n == 0:
        return np.array([], dtype=float)
    ranks = np.argsort(np.argsort(ids))
    if high_first:
        preds = 1.0 - ranks / (n - 1) if n > 1 else np.ones(1)
    else:
        preds = ranks / (n - 1) if n > 1 else np.zeros(1)
    return preds.astype(float)


def step_predictions(ids, high_first=True):
    """
    Generate a sharp step‑wise prediction:
    first half of sorted IDs get 1, second half get 0 when high_first=True
    (i.e., larger IDs receive lower probabilities). The opposite ordering
    is used when high_first=False.
    """
    ids = np.asarray(ids, dtype=float)
    n = len(ids)
    if n == 0:
        return np.array([], dtype=float)
    ranks = np.argsort(np.argsort(ids))
    half = n // 2
    if high_first:
        preds = (ranks < half).astype(float)  # low IDs → 1, high IDs → 0
    else:
        preds = (ranks >= half).astype(float)  # low IDs → 0, high IDs → 1
    return preds.astype(float)


def dummy_predictions(num, high_first=True):
    """
    Legacy step‑wise generator kept for compatibility (not used in the new flow).
    """
    if num <= 0:
        return np.array([], dtype=float)
    half = num // 2
    if high_first:
        preds = np.concatenate([np.ones(half), np.zeros(num - half)])
    else:
        preds = np.concatenate([np.zeros(half), np.ones(num - half)])
    return preds




## === cell 2
def get_case_ids(path):
    """
    Retrieve sorted case folder names and numeric IDs, ignoring any
    non‑numeric entries that may appear (e.g., stray files or folders).
    """
    case_dirs = sorted(
        [
            d
            for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d)) and d.isdigit()
        ]
    )
    ids = [int(d.lstrip("0") or "0") for d in case_dirs]
    return case_dirs, ids


case_dirs, case_ids = get_case_ids(TEST_PATH)




## === cell 3
num_cases = len(case_ids)

TRAIN_LABELS_PATH = (
    "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv"
)
train_df = pd.read_csv(TRAIN_LABELS_PATH)

train_df["numeric_id"] = (
    train_df["BraTS21ID"].astype(str).apply(lambda x: int(x.lstrip("0") or "0"))
)

auc_high_first = roc_auc_score(
    train_df["MGMT_value"],
    monotonic_predictions(train_df["numeric_id"].values, high_first=True),
)
auc_low_first = roc_auc_score(
    train_df["MGMT_value"],
    monotonic_predictions(train_df["numeric_id"].values, high_first=False),
)

high_first = auc_high_first < auc_low_first  # pick the more anti‑correlated orientation

predictions = monotonic_predictions(case_ids, high_first=high_first)




## === cell 4
SAMPLE_SUB_PATH = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv"
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

if len(sample_sub) == num_cases:
    submission = sample_sub.copy()
    submission["MGMT_value"] = predictions
else:
    submission = pd.DataFrame({"BraTS21ID": case_dirs, "MGMT_value": predictions})




## === cell 5
OUTPUT_PATH = "submission.csv"
submission.to_csv(OUTPUT_PATH, index=False)
print(f"Submission written to {OUTPUT_PATH}")
