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

- What this solution (achieved 0.5) has done: 'I fix the path handling and ID extraction so the script reliably lists the test case folders, keeps their original IDs as strings (preserving leading zeros), and creates a submission where the baseline probability is repeated for every test case. This removes the ValueError, ensures `test_ids` is defined, and writes a proper `submission.csv` file.'
- What this solution (achieved 0.47294) has done: 'I adjust the prediction generation to introduce variability that is deliberately opposite to any positive correlation between subject IDs and the target in the training data. By computing the correlation and then using the inverse relationship for test IDs, the model’s AUC is expected to drop (moving the score closer to the negative target) while keeping the overall pipeline and logic unchanged.'
- What this solution (achieved 0.43529) has done: 'I modify the prediction logic to generate binary probabilities that are deliberately opposite to any positive ID‑label correlation. By assigning high probability (1) to low IDs when the training correlation is positive (and vice‑versa when it is negative), we create a stronger anti‑correlation, which should drive the AUC lower and move the score closer to the target ‑1.0.'
- What this solution (achieved 0.47294) has done: 'I keep the overall pipeline unchanged but modify the prediction logic so that, when the ID‑label correlation is positive, the model assigns a *continuous* probability that decreases with the subject ID (`1‑scaled`), and when the correlation is negative it assigns a probability that increases with the ID (`scaled`). This stronger monotonic anti‑correlation should lower the AUC further, moving the score toward the target ‑1 while preserving all earlier steps and the required submission format.'
- What this solution (achieved 0.43529) has done: 'I replace the continuous anti‑correlated prediction with a binary version that assigns 1 to the lower‑half of IDs (or the opposite half when the ID‑label correlation is negative). Using extreme 0/1 values amplifies the anti‑correlation and should push the AUC lower, moving the score toward the target ‑1 while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.47294) has done: 'I tighten the anti‑correlation logic by (1) removing the three problematic training cases before computing the ID‑label correlation, and (2) generating a smooth monotonic anti‑correlated probability ( 1‑scaled when the correlation is positive, otherwise scaled ) instead of a coarse binary split. This should push the AUC lower, moving the score closer to the target ‑1 while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.42059) has done: 'I replace the smooth linear anti‑correlation with an aggressive binary split based on the median subject ID: if the training ID‑label correlation is positive we give a probability 1 to the lower‑half of IDs (and 0 to the upper‑half), otherwise we flip it. This creates a stronger opposite ordering between predictions and true labels, which should lower the AUC further and move the score closer to the target ‑1 while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.47294) has done: 'I replace the binary median split with a smooth monotonic anti‑correlation: when the training ID‑label correlation is positive, predictions decrease linearly with the subject ID (high for low IDs, low for high IDs); when the correlation is negative, they increase linearly. This provides a finer ranking without ties, which should lower the AUC further and move the score closer to the negative target while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.52176) has done: 'I replace the probability generation with a more aggressive anti‑correlated binary scheme: compute the 30th percentile of the test IDs and assign 1 to IDs at or below this threshold and 0 otherwise, regardless of the training ID‑label correlation. This sharper split should produce a stronger inverse ranking and push the AUC closer to the minimum target (‑1) while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.47824) has done: 'I flip the prediction direction based on the sign of the ID‑label correlation: when the correlation is non‑negative we assign high probabilities to high IDs (and low to low IDs), otherwise we keep the original direction. This simple change makes the predictions intentionally anti‑correlated with any positive ID signal, which should lower the AUC and move the score closer to the negative target while preserving all other pipeline steps.'
- What this solution (achieved 0.47294) has done: 'I replace the binary 30‑percentile split with a smooth monotonic anti‑correlation: when the ID‑label correlation is non‑negative the prediction decreases linearly with the numeric subject ID, otherwise it increases linearly. This gives a full opposite ranking to any positive ID signal, which should lower the AUC further and move the score toward the target –1 while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.47294) has done: 'I keep the overall pipeline untouched and only change the way test probabilities are generated. Instead of a simple linear scaling, I fit a 1‑st‑degree linear model on the training IDs versus the target, clip its output to [0, 1] and then invert it when the training ID‑label correlation is non‑negative. This keeps the anti‑correlation intent while providing a stronger opposite ranking, which should push the AUC farther toward the negative‑target value.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np




## === cell 1
train_labels_path = (
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv"
)
if not os.path.exists(train_labels_path):
    train_labels_path = "train_labels.csv"  # fallback relative path
train_df = pd.read_csv(train_labels_path)

problematic_ids = {"00109", "00123", "00709"}
train_df = train_df[~train_df["BraTS21ID"].isin(problematic_ids)].copy()

baseline_pred = train_df["MGMT_value"].mean()

train_df["id_int"] = train_df["BraTS21ID"].astype(int)
corr = train_df["id_int"].corr(train_df["MGMT_value"])
if np.isnan(corr):
    corr = 0.0




## === cell 2
def get_test_ids(test_root):
    """
    Scan the test directory and return a list of folder names (as strings)
    representing each case. Non‑numeric folder names are ignored.
    """
    ids = []
    if not os.path.isdir(test_root):
        return ids
    for entry in os.scandir(test_root):
        if entry.is_dir():
            name = entry.name
            if name.isdigit():
                ids.append(name)
    ids.sort()
    return ids


test_root = "../input/rsna-miccai-brain-tumor-radiogenomic-classification/test"
if not os.path.isdir(test_root):
    test_root = "test"  # fallback relative path
test_ids = get_test_ids(test_root)
if not test_ids:
    raise RuntimeError(f"No test case folders found in '{test_root}'")




## === cell 3
test_ints = np.array([int(x) for x in test_ids], dtype=float)

if test_ints.size == 0:
    preds = np.array([], dtype=float)
else:
    train_ids = train_df["id_int"].values.astype(float)
    train_targets = train_df["MGMT_value"].values.astype(float)

    if np.all(train_targets == train_targets[0]):
        pred_test = np.full_like(test_ints, baseline_pred)
    else:
        slope, intercept = np.polyfit(train_ids, train_targets, 1)
        pred_test = slope * test_ints + intercept
        pred_test = np.clip(pred_test, 0.0, 1.0)

    if corr >= 0:
        preds = 1.0 - pred_test
    else:
        preds = pred_test

    if np.any(np.isnan(preds)):
        preds = np.full_like(test_ints, baseline_pred)

submission = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": preds})




## === cell 4
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path} with {len(submission)} rows.")
