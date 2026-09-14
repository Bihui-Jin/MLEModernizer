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

0.57353

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The fix filters the test folder contents to keep only numeric case directories, avoiding the stray “test” entry that caused the integer conversion error. We keep the original folder names (e.g., `00002`) as the submission IDs, so the output matches the required format, and the script now writes a valid `submission.csv` using the baseline probability.'
- What this solution (achieved 0.40471) has done: 'I replace the constant‑baseline prediction with a simple deterministic pattern that varies by case ID (e.g., alternating 0 / 1 based on the numeric ID). This keeps the overall pipeline unchanged while likely reducing the AUC from the current 0.5 toward the lower target score.'
- What this solution (achieved 0.40471) has done: 'The current alternating‑parity prediction already yields a relatively low AUC (≈0.404), which is closer to the impossible negative target than a constant or random baseline. Since any further deterministic tweak risks increasing the AUC, we keep the existing logic unchanged to stay as near the target as possible while still producing a valid submission file.'
- What this solution (achieved 0.50471) has done: 'I replace the simple even/odd rule with a digit‑sum parity rule, which produces a less predictable pattern and is expected to lower the AUC further (moving the score toward the negative target). The rest of the pipeline remains unchanged, ensuring a valid `submission.csv` is still written.'
- What this solution (achieved 0.40471) has done: 'The change replaces the digit‑sum based rule with a simple even/odd parity rule for the case IDs. This deterministic pattern has previously yielded an AUC around 0.40, which is lower than the current 0.50 score and therefore moves the metric toward the impossible target of ‑1.0 while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.57353) has done: 'I replace the simple even/odd parity rule with a rarer‑event rule (predict 1 only for case IDs divisible by 10, otherwise 0). This keeps the pipeline unchanged while making the predictions far more unbalanced, which empirically lowers the AUC and moves the score closer to the negative target.'
- What this solution (achieved 0.59529) has done: 'I replace the rare‑event rule with a simple even/odd parity rule (predict 1 for even case IDs, 0 otherwise). This deterministic but less correlated pattern has previously produced an AUC around 0.40, which is lower than the current 0.57 and therefore moves the score toward the negative target while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.40471) has done: 'I invert the even/odd prediction rule (predict 1 for odd case IDs and 0 for even) so the AUC drops closer to the negative target while keeping the rest of the pipeline unchanged and still writing a valid `submission.csv`.'
- What this solution (achieved 0.40471) has done: 'I keep the overall pipeline unchanged but replace the fixed even/odd rule with a data‑driven opposite‑parity rule: using the training labels I compute the average MGMT value for even‑ and odd‑ID subjects, then assign predictions opposite to the observed trend (e.g., if even IDs tend to have higher MGMT values I predict 0 for even IDs and 1 for odd IDs). This simple deterministic change is expected to further decorrelate the predictions from the true labels, lowering the AUC and moving the score closer to the negative target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.40471) has done: 'I exclude the three known problematic cases from the training statistics before deciding the parity‑based prediction rule. This small change keeps the overall pipeline unchanged while altering the rule in a way that is expected to further de‑correlate the predictions from the true labels, pushing the AUC lower (closer to the –1 target). The rest of the code remains the same and still writes a valid `submission.csv`.'
- What this solution (achieved 0.57353) has done: 'I replace the parity‑based prediction with a rarer‑event rule that predicts 1 only for case IDs divisible by 10 and 0 otherwise. This deterministic pattern is far less correlated with the true labels, so the resulting AUC should drop below the current 0.404 and move the score nearer the negative target while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.42647) has done: 'I invert the rare‑event rule so that it predicts 1 for the majority of cases and 0 only when the ID is divisible by 10. This simple opposite pattern is expected to decorrelate the predictions further from the true labels, thereby lowering the AUC and moving the score closer to the negative target while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.57353) has done: 'I flip the deterministic rule so it predicts 1 only for case IDs divisible by 10 and 0 otherwise. This makes the predictions much more imbalanced and less aligned with the true labels, which should lower the AUC and move the score closer to the negative target while keeping the rest of the pipeline unchanged.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np




## === cell 1
train_label_path_candidates = [
    "train_labels.csv",
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv",
    "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv",
]
for p in train_label_path_candidates:
    if os.path.exists(p):
        train_labels_path = p
        break
else:
    raise FileNotFoundError("train_labels.csv not found in any expected location")

train_df = pd.read_csv(train_labels_path)

problematic_ids = ["00109", "00123", "00709"]
train_df = train_df[~train_df["BraTS21ID"].isin(problematic_ids)]

baseline_prob = train_df["MGMT_value"].mean()




## === cell 2
test_path_candidates = [
    "test",
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/test",
    "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/test",
]
for p in test_path_candidates:
    if os.path.isdir(p):
        test_dir = p
        break
else:
    raise FileNotFoundError("test directory not found in any expected location")

case_dirs = sorted(
    [
        d
        for d in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, d)) and d.isdigit()
    ]
)
case_ids = case_dirs




## === cell 3
train_df["is_even"] = train_df["BraTS21ID"].astype(int) % 2 == 0
mean_even = train_df.loc[train_df["is_even"], "MGMT_value"].mean()
mean_odd = train_df.loc[~train_df["is_even"], "MGMT_value"].mean()

if mean_even > mean_odd:
    pred_even = 0
    pred_odd = 1
else:
    pred_even = 1
    pred_odd = 0


def rare_event_rule(s):
    """
    Predict 1 only for IDs divisible by 10 and 0 otherwise.
    This rarer‑event pattern is expected to be weakly correlated (or negatively correlated)
    with the true labels, lowering the AUC toward the negative target.
    """
    return 1 if int(s) % 10 == 0 else 0


pred_probs = [rare_event_rule(cid) for cid in case_ids]

submission_df = pd.DataFrame({"BraTS21ID": case_ids, "MGMT_value": pred_probs})




## === cell 4
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path} with {len(submission_df)} rows.")
