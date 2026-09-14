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

0.46588

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The fix replaces the fragile directory‑scanning logic with a reliable read of the provided sample_submission.csv to obtain the exact test case IDs, falling back to a safe numeric‑folder scan if the file is missing. This prevents the ValueError when non‑numeric folder names appear, ensures the predictions array matches the IDs length, and guarantees a correctly‑formatted submission.csv is written.'
- What this solution (achieved 0.40471) has done: 'We replace the constant‑value baseline with a simple alternating pattern based on the case ID (even → 0, odd → 1). This keeps the overall pipeline unchanged but intentionally decorrelates the predictions from the true labels, aiming to lower the AUC score toward the (unrealistically low) target. The rest of the code remains the same, ensuring a valid CSV is still written.'
- What this solution (achieved 0.40471) has done: 'We compute the average MGMT label for even and odd BraTS21IDs from the training set and set the prediction pattern to be opposite of the parity that shows the higher positive rate. This keeps the overall pipeline unchanged while intentionally worsening the ranking, moving the AUC closer to the low target score.'
- What this solution (achieved 0.47294) has done: 'We replace the simple parity‑based prediction with a linear‐trend model fitted on the training IDs and then invert its output for the test cases. This keeps the overall pipeline unchanged while deliberately creating a prediction that is negatively correlated with the true label (if the original trend is positive), moving the AUC closer to the low target score. Minor clipping ensures valid probability values and the rest of the code (submission creation) stays the same.'
- What this solution (achieved 0.48471) has done: 'We make the model fit a higher‑degree polynomial (degree 5) to the training IDs and labels, then keep the inversion step. A more flexible fit can over‑fit the training data, giving a strong positive correlation that, after inversion, flips the ranking and pushes the AUC notably lower toward the negative target. The rest of the pipeline and submission format stay unchanged.'
- What this solution (achieved 0.49765) has done: 'I increase the polynomial degree used to fit the training IDs to the MGMT labels from 5 to 10. A higher‑degree fit can capture more of the true ID‑label relationship, giving a stronger positive correlation. Since the pipeline inverts the fitted values (`1.0 - raw_preds`), a better‑correlated model produce predictions that are more oppositely ranked to the true labels, thereby lowering the AUC and moving the score closer to the very low target. The rest of the code and submission format remain unchanged.'
- What this solution (achieved 0.4) has done: 'I increase the polynomial degree used to fit the training IDs (from 10 to 20). A higher‑degree fit can capture more of the true ID‑label relationship, yielding a stronger positive correlation; after the existing inversion step this produces predictions that are more oppositely ranked to the true labels, thereby lowering the AUC and moving the score closer to the very low target. The rest of the pipeline and submission format remain unchanged.'
- What this solution (achieved 0.47294) has done: 'I replace the high‑degree polynomial fitting with a simple rank‑based prediction that flips the ordering when the training IDs are positively correlated with the label. This yields predictions that are deliberately poorly ranked (lower AUC), moving the score down toward the very low target while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.46588) has done: 'The patch strengthens the anti‑correlation of the predictions by forcing a steep decreasing relationship with the case IDs: we map the normalized ID rank to `1 – 2*rank` and clip to [0, 1]. This creates many high‑probability predictions for low IDs and zeros for higher IDs, which pushes the AUC lower and moves the score toward the very low target while keeping the rest of the pipeline unchanged.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np


def get_case_ids(test_dir: str, sample_submission_path: str):
    """
    Return ordered list of test case IDs.
    Primary source: the sample_submission.csv (ensures correct order and formatting).
    Fallback: scan sub‑directories of test_dir that have numeric names.
    """
    if os.path.exists(sample_submission_path):
        df = pd.read_csv(sample_submission_path, dtype=str)
        ids = df["BraTS21ID"].apply(lambda x: int(x.lstrip("0") or "0")).tolist()
        return ids

    ids = []
    for entry in os.scandir(test_dir):
        if entry.is_dir():
            name = entry.name
            if name.isdigit():
                ids.append(int(name.lstrip("0") or "0"))
    ids.sort()
    return ids


TEST_PATH = "../input/rsna-miccai-brain-tumor-radiogenomic-classification/test"
SAMPLE_SUB_PATH = (
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv"
)

case_ids = get_case_ids(TEST_PATH, SAMPLE_SUB_PATH)

TRAIN_LABELS_PATH = (
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv"
)
train_df = pd.read_csv(TRAIN_LABELS_PATH, dtype=str)
train_df["int_id"] = train_df["BraTS21ID"].apply(lambda x: int(x.lstrip("0") or "0"))
train_df["MGMT_value"] = train_df["MGMT_value"].astype(float)

min_id = train_df["int_id"].min()
max_id = train_df["int_id"].max()
if max_id == min_id:
    test_rank = np.full(len(case_ids), 0.5, dtype=np.float32)
else:
    test_ids_arr = np.array(case_ids, dtype=float)
    test_rank = (test_ids_arr - min_id) / (max_id - min_id)

factor = 2.0  # makes most high IDs map to 0 after clipping
raw_preds = 1.0 - factor * test_rank
predictions = np.clip(raw_preds, 0.0, 1.0).astype(np.float32)




## === cell 1
def create_submission_dataframe(ids, preds):
    """
    Build the DataFrame in the exact format required by Kaggle:
    columns -> BraTS21ID, MGMT_value
    """
    df = pd.DataFrame({"BraTS21ID": ids, "MGMT_value": preds})
    return df


submission_df = create_submission_dataframe(case_ids, predictions)




## === cell 2
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path} with {len(submission_df)} rows.")
