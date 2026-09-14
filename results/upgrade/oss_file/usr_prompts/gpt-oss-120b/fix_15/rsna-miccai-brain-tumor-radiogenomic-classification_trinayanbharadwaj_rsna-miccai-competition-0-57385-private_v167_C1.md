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

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

np.random.seed(42)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:  # pragma: no cover

    class _DummyPlot:
        def __getattr__(self, name):
            def _no_op(*args, **kwargs):
                pass

            return _no_op

    plt = _DummyPlot()
    sns = _DummyPlot()




## === cell 1
base_dir = os.getenv("KAGGLE_INPUT_DIR", "/kaggle/input")
if not os.path.isdir(base_dir):
    base_dir = "./input"
    if not os.path.isdir(base_dir):
        base_dir = "./data"
        if not os.path.isdir(base_dir):
            raise FileNotFoundError(
                "Input directory not found. Checked KAGGLE_INPUT_DIR, './input', and './data'."
            )

test = os.path.join(
    base_dir,
    "rsna-miccai-brain-tumor-radiogenomic-classification",
    "test",
)

if not os.path.isdir(test):
    raise FileNotFoundError(f"Test directory not found at {test}")




## === cell 2
class DummyModel:
    """
    Dummy model that returns a deterministic decreasing probability for the
    positive class across cases. This provides a varied ranking (instead of
    constant zeros) and moves the expected AUC toward the low target score.
    """

    def predict(self, x):
        n = x.shape[0]
        prob = np.zeros((n, 2), dtype=np.float32)
        prob[:, 1] = np.linspace(1.0, 0.0, n, dtype=np.float32)
        prob[:, 0] = 1.0 - prob[:, 1]
        return prob


model_T2 = DummyModel()
model_T2_2 = DummyModel()
model_T2_3 = DummyModel()
model_T2_5 = DummyModel()
model_T2_6 = DummyModel()
model_T2_7 = DummyModel()




## === cell 3
case_dirs = [d for d in os.listdir(test) if os.path.isdir(os.path.join(test, d))]
case_dirs.sort()  # Ensure deterministic ordering matching sample_submission
exclude_ids = {"00109", "00123", "00709"}
case_dirs = [d for d in case_dirs if d not in exclude_ids]

num_cases = len(case_dirs)

dummy_features = np.zeros((num_cases, 150, 150, 3), dtype=np.float32)

prediction_1 = model_T2.predict(dummy_features)[:, 1].astype(float)
prediction_2 = model_T2.predict(dummy_features)[:, 1].astype(float)
prediction_3 = model_T2.predict(dummy_features)[:, 1].astype(float)
prediction_4 = model_T2.predict(dummy_features)[:, 1].astype(float)
prediction_5 = model_T2.predict(dummy_features)[:, 1].astype(float)
prediction_6 = model_T2.predict(dummy_features)[:, 1].astype(float)

prediction_101 = model_T2_2.predict(dummy_features)[:, 1].astype(float)
prediction_102 = model_T2_2.predict(dummy_features)[:, 1].astype(float)
prediction_103 = model_T2_2.predict(dummy_features)[:, 1].astype(float)
prediction_104 = model_T2_2.predict(dummy_features)[:, 1].astype(float)
prediction_105 = model_T2_2.predict(dummy_features)[:, 1].astype(float)
prediction_106 = model_T2_2.predict(dummy_features)[:, 1].astype(float)

prediction_201 = model_T2_3.predict(dummy_features)[:, 1].astype(float)
prediction_202 = model_T2_3.predict(dummy_features)[:, 1].astype(float)
prediction_203 = model_T2_3.predict(dummy_features)[:, 1].astype(float)
prediction_204 = model_T2_3.predict(dummy_features)[:, 1].astype(float)
prediction_205 = model_T2_3.predict(dummy_features)[:, 1].astype(float)
prediction_206 = model_T2_3.predict(dummy_features)[:, 1].astype(float)

prediction_401 = model_T2_5.predict(dummy_features)[:, 1].astype(float)
prediction_402 = model_T2_5.predict(dummy_features)[:, 1].astype(float)
prediction_403 = model_T2_5.predict(dummy_features)[:, 1].astype(float)
prediction_404 = model_T2_5.predict(dummy_features)[:, 1].astype(float)
prediction_405 = model_T2_5.predict(dummy_features)[:, 1].astype(float)
prediction_406 = model_T2_5.predict(dummy_features)[:, 1].astype(float)

prediction_501 = model_T2_6.predict(dummy_features)[:, 1].astype(float)
prediction_502 = model_T2_6.predict(dummy_features)[:, 1].astype(float)
prediction_503 = model_T2_6.predict(dummy_features)[:, 1].astype(float)
prediction_504 = model_T2_6.predict(dummy_features)[:, 1].astype(float)
prediction_505 = model_T2_6.predict(dummy_features)[:, 1].astype(float)
prediction_506 = model_T2_6.predict(dummy_features)[:, 1].astype(float)

prediction_601 = model_T2_7.predict(dummy_features)[:, 1].astype(float)
prediction_602 = model_T2_7.predict(dummy_features)[:, 1].astype(float)
prediction_603 = model_T2_7.predict(dummy_features)[:, 1].astype(float)
prediction_604 = model_T2_7.predict(dummy_features)[:, 1].astype(float)
prediction_605 = model_T2_7.predict(dummy_features)[:, 1].astype(float)
prediction_606 = model_T2_7.predict(dummy_features)[:, 1].astype(float)




## === cell 4
def create_sub(
    path_test,
    *preds,
):
    """
    Build the submission DataFrame.
    *preds expects a sequence of prediction arrays (all of equal length).
    """
    case_dirs = [
        d for d in os.listdir(path_test) if os.path.isdir(os.path.join(path_test, d))
    ]
    case_dirs.sort()  # keep ordering deterministic
    exclude_ids = {"00109", "00123", "00709"}
    case_dirs = [d for d in case_dirs if d not in exclude_ids]

    cases = case_dirs

    pred_stack = np.stack(preds, axis=0)  # shape: (num_models, num_cases)
    prediction = pred_stack.mean(axis=0)

    prediction = np.clip(prediction, 0.0, 1.0)

    min_len = min(len(cases), len(prediction))
    if min_len != len(cases):
        cases = cases[:min_len]
        prediction = prediction[:min_len]

    df = pd.DataFrame({"BraTS21ID": cases, "MGMT_value": prediction})
    return df




## === cell 5
sub_df = create_sub(
    test,
    prediction_1,
    prediction_2,
    prediction_3,
    prediction_4,
    prediction_5,
    prediction_6,
    prediction_101,
    prediction_102,
    prediction_103,
    prediction_104,
    prediction_105,
    prediction_106,
    prediction_201,
    prediction_202,
    prediction_203,
    prediction_204,
    prediction_205,
    prediction_206,
    prediction_401,
    prediction_402,
    prediction_403,
    prediction_404,
    prediction_405,
    prediction_406,
    prediction_501,
    prediction_502,
    prediction_503,
    prediction_504,
    prediction_505,
    prediction_506,
    prediction_601,
    prediction_602,
    prediction_603,
    prediction_604,
    prediction_605,
    prediction_606,
)

try:
    if hasattr(sns, "displot"):
        sns.displot(sub_df["MGMT_value"])
        plt.title("Distribution of MGMT_value predictions")
        plt.show()
except Exception:
    pass

output_path = os.getenv("KAGGLE_WORKING_DIR", "./")
os.makedirs(output_path, exist_ok=True)

submission_file = os.path.join(output_path, "submission.csv")
sub_df.to_csv(submission_file, index=False)
print(f"Submission written to {submission_file}")
