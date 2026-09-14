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

- What this solution (achieved 0.5) has done: 'I make the submission generation robust by loading the official sample_submission file to obtain the exact order of test IDs (ensuring column names and row count match Kaggle’s expectations) and write the file to the standard `/kaggle/working/` directory. The prediction itself remains the global mean probability, preserving the original model logic while guaranteeing a valid CSV output.'
- What this solution (achieved 0.5) has done: 'I replace the global‑mean baseline with a constant probability of 0.0, which is expected to give a poorer AUC (closer to the target –1.0) while keeping the overall pipeline unchanged and still producing a valid CSV submission.'
- What this solution (achieved 0.47294) has done: 'I introduce a tiny variation in the predicted probabilities by mapping each BraTS21ID to a numeric value and scaling it to the [0, 1] range (then inverting). This keeps the overall pipeline unchanged while turning the constant prediction into a simple monotonic signal, which is expected to lower the AUC from the current 0.5 toward the target ‑1.0 without altering file paths or submission format.'
- What this solution (achieved 0.47294) has done: 'The update adds a lightweight analysis of the training IDs versus their true MGMT values to decide whether the current ID‑based probability mapping should be inverted or not. By aligning the prediction direction with the observed correlation in the training set, the submission becomes more likely to be anti‑correlated with the true labels, which pushes the AUC lower toward the target –1.0 while keeping the original pipeline and file handling unchanged.'
- What this solution (achieved 0.47294) has done: 'I replace the simple ID‑based scaling with a tiny linear model fitted on the training numeric IDs versus their MGMT values and then invert the predictions when the fitted slope is positive. This keeps the overall pipeline unchanged while creating a stronger anti‑correlation between the predicted probabilities and the true labels, which should lower the AUC and move the score closer to the target ‑1.0.'
- What this solution (achieved 0.47294) has done: 'I force the predictions to be the opposite of the linear fit for every test case, removing the conditional that sometimes leaves the fit unchanged. By always using `1 - pred_test` we create a stronger anti‑correlation with the true labels, which should lower the AUC and move the score closer to the target ‑1.0 while keeping all other logic and file handling intact.'
- What this solution (achieved 0.47294) has done: 'I replace the linear‑fit based prediction with a full‑range rank‑based mapping of the test IDs, then invert it to create a stronger anti‑correlation with any positive ID‑label relationship. This uses the same ID extraction logic but generates probabilities spanning 0‑1, which should push the AUC lower (toward the –1.0 target) while preserving all file handling and output format.'
- What this solution (achieved 0.47294) has done: 'I add a lightweight analysis of the correlation between the numeric subject IDs and the training MGMT labels. If the IDs are positively correlated with the target, I keep the inverted rank‑based probabilities (as before); if the correlation is negative (or undefined) I use the direct rank‑based probabilities. This simple sign‑check gives a stronger anti‑correlation with the true labels and should push the AUC lower, moving the score closer to the –1.0 target while preserving the overall pipeline and file handling.'
- What this solution (achieved 0.47294) has done: 'I simplify the probability logic by always using the inverted rank‑based mapping ( `1‑prob` ). This removes the conditional on the ID‑label correlation, which should make the predictions consistently anti‑correlated with any positive relationship and push the AUC lower, moving the score toward the target -1.0 while preserving all other pipeline steps and output handling.'
- What this solution (achieved 0.47294) has done: 'I add a simple sign‑check on the correlation between numeric subject IDs and the training MGMT labels. If the correlation is negative, I keep the rank‑based probabilities as‑is; otherwise I keep the inverted mapping. This keeps the pipeline unchanged while steering predictions toward stronger anti‑correlation, which should lower the AUC and move the score nearer to the –1.0 target.'
- What this solution (achieved 0.47294) has done: 'I force the prediction to always use the inverted rank‑based probabilities (`1‑prob`) instead of conditionally inverting based on the ID‑label correlation. This stronger anti‑correlation should lower the AUC further, moving the score closer to the negative target while keeping the overall pipeline and file handling unchanged.'
- What this solution (achieved 0.47294) has done: 'I replace the simple rank‑based probability with a linear‑fit based prediction that is scaled to [0, 1] and then inverted only when the numeric ID is positively correlated with the training labels. This stronger (and sign‑aware) anti‑correlation should push the AUC lower, moving the score toward the target ‑1.0 while keeping all file handling unchanged.'
- What this solution (achieved 0.47294) has done: 'I force the probability predictions to always be the inverted, scaled linear fit (`1 - pred_test`). This removes the conditional that sometimes leaves the predictions unchanged and creates a stronger anti‑correlation with any positive relationship between IDs and labels, which should lower the AUC and move the score closer to the negative target while keeping the rest of the pipeline intact.'
- What this solution (achieved 0.47294) has done: 'I adjust the prediction logic to use the observed correlation between numeric IDs and the training MGMT values: if the IDs are positively correlated with the target we keep the inverted rank‑based probabilities (producing anti‑correlation), otherwise we use the direct rank‑based probabilities. This conditional inversion should move the AUC lower (toward the –1 target) while preserving the existing pipeline and output format.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

BASE_INPUT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_LABELS_PATH = os.path.join(BASE_INPUT, "train_labels.csv")
TEST_DIR = os.path.join(BASE_INPUT, "test")
SAMPLE_SUBMISSION_PATH = os.path.join(BASE_INPUT, "sample_submission.csv")


def load_train_mean(label_path: str) -> float:
    """Load training labels and return the mean MGMT_value."""
    df = pd.read_csv(label_path)
    return df["MGMT_value"].mean()


def get_ordered_test_ids(sample_submission_path: str) -> list:
    """
    Load the official sample_submission file to obtain the exact ordering
    of BraTS21ID values expected in the final submission.
    """
    df = pd.read_csv(sample_submission_path, dtype=str)
    return df["BraTS21ID"].tolist()




## === cell 1
def id_to_num(id_str):
    """Convert a BraTS21ID like '00012' to an integer 12."""
    try:
        return int(id_str)
    except ValueError:
        digits = "".join(filter(str.isdigit, id_str))
        return int(digits) if digits else 0


test_ids = get_ordered_test_ids(SAMPLE_SUBMISSION_PATH)

numeric_test = np.array([id_to_num(i) for i in test_ids])

train_df = pd.read_csv(TRAIN_LABELS_PATH, dtype=str)
train_numeric = np.array([id_to_num(i) for i in train_df["BraTS21ID"]])
train_labels = train_df["MGMT_value"].astype(float).values

if len(train_numeric) > 1:
    corr = np.corrcoef(train_numeric, train_labels)[0, 1]
else:
    corr = np.nan

if len(train_numeric) > 1:
    slope, intercept = np.polyfit(train_numeric, train_labels, 1)
    pred_test_raw = slope * numeric_test + intercept
    min_pred, max_pred = pred_test_raw.min(), pred_test_raw.max()
    if max_pred > min_pred:
        pred_test = (pred_test_raw - min_pred) / (max_pred - min_pred)
    else:
        pred_test = np.zeros_like(pred_test_raw)
else:
    pred_test = np.zeros_like(numeric_test, dtype=float)

if not np.isnan(corr) and corr > 0:
    probabilities = 1.0 - pred_test
else:
    probabilities = pred_test

probabilities = np.clip(probabilities, 0.0, 1.0)

submission_df = pd.DataFrame(
    {
        "BraTS21ID": test_ids,
        "MGMT_value": probabilities,
    }
)




## === cell 2
submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path} with {len(submission_df)} rows.")
