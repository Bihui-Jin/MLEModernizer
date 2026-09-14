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

0.50235

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fix the ID‑extraction function so it only keeps directories whose names are numeric, which prevents the “test” folder (or any non‑numeric entry) from causing a conversion error. The rest of the pipeline remains unchanged, allowing a constant‑probability baseline to run and write a proper `submission.csv` file.'
- What this solution (achieved 0.47294) has done: 'I modify the prediction generation to produce a monotonic decreasing probability vector instead of a constant 0.5 baseline. By varying the scores across cases we break the trivial random‑guess behaviour, which tends to lower the AUC (moving the score from 0.5 toward the target ‑1.0). The rest of the pipeline and submission format remain unchanged.'
- What this solution (achieved 0.48059) has done: 'I introduce a small random perturbation to the monotonically decreasing baseline predictions (while keeping them within [0, 1]) so the ranking becomes less perfectly anti‑correlated with the case IDs. This slight noise tends to push the AUC a bit lower than the previous 0.47294, moving the score toward the target ‑1.0 without changing any core logic or the submission format.'
- What this solution (achieved 0.53647) has done: 'I invert the baseline ordering so predictions rise from 0 → 1 (instead of falling) and keep the same small random noise. This simple reversal can turn a modestly positive AUC into a lower one, moving the score toward the target ‑1.0 while preserving the overall pipeline and its core logic.'
- What this solution (achieved 0.49118) has done: 'I keep the overall pipeline unchanged but flip the baseline ordering so predictions decrease from 1 to 0, and increase the random noise range slightly. This creates a stronger anti‑correlation with the case IDs, which empirically lowers the AUC and moves the score closer to the target ‑1.0 while still producing a valid submission CSV.'
- What this solution (achieved 0.49059) has done: 'We increase the random perturbation applied to the decreasing baseline predictions (range −0.4 to 0.4) so the ranking becomes less correlated with the case IDs, which is expected to lower the AUC and move the score closer to the target ‑1.0 while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.57118) has done: 'We replace the simple linear‐noise baseline with a very cheap image‑based feature: the mean intensity of the first DICOM slice in each test case. By normalising this mean and then inverting it we obtain predictions that are likely anti‑correlated with the true label, moving the AUC lower (closer to the ‑1 target) while keeping the overall pipeline unchanged. The rest of the code (submission creation and saving) remains the same.'
- What this solution (achieved 0.56529) has done: 'I increase the random‑noise amplitude added to the inverted, intensity‑based predictions (from ±0.05 to ±0.30).  Larger noise disrupts any accidental positive correlation with the true labels, pushing the AUC lower and thus moving the score closer to the target ‑1.0 while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.49059) has done: 'I replace the intensity‑based prediction with a simple decreasing linear baseline (1 → 0) plus a broader random perturbation (±0.4). This keeps the overall pipeline (directory scanning, DICOM handling) unchanged while producing predictions that are less correlated with any true signal, thereby lowering the AUC toward the target −1.0. The random seed is fixed for reproducibility, and the final submission format remains the same.'
- What this solution (achieved 0.47294) has done: 'We eliminate the added random noise and use the pure decreasing linear baseline (1 → 0) as the prediction vector. This creates a strong anti‑correlation with any monotonic label ordering, which should lower the AUC and move the score closer to the target ‑1 while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.54824) has done: 'I added the missing standard‑library and third‑party imports (os, numpy, pandas, matplotlib, seaborn, pydicom) at the top of the notebook so that all subsequent cells can run. The rest of the logic—including the intensity‑based baseline, noise addition, DataFrame construction, and CSV writing—remains unchanged, ensuring a valid `submission.csv` is produced.'
- What this solution (achieved 0.50235) has done: 'Implemented a simpler anti‑correlated baseline: predictions now follow a decreasing linear trend across the sorted test IDs (1 → 0) with modest random noise. This removes the intensity‑based signal that was inadvertently boosting the AUC, moving the score closer to the target ‑1.0 while preserving all I/O and submission logic.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import pydicom as dicom
except ImportError:
    raise ImportError("pydicom is required for DICOM handling but is not installed.")


def get_test_case_ids(test_dir: str) -> list[int]:
    """
    Scan the test directory and return a sorted list of integer case IDs.
    Only folders whose names consist solely of digits are considered.
    """
    case_ids = []
    for entry in os.scandir(test_dir):
        if entry.is_dir():
            name = entry.name
            if name.isdigit():  # keep only numeric folders
                stripped = name.lstrip("0")
                case_ids.append(int(stripped) if stripped else 0)
    case_ids.sort()
    return case_ids




## === cell 1
test_dir = "../input/rsna-miccai-brain-tumor-radiogenomic-classification/test"




## === cell 2
test_ids = get_test_case_ids(test_dir)
print(f"Found {len(test_ids)} test cases.")




## === cell 3
np.random.seed(42)  # reproducibility

base_pred = np.linspace(1.0, 0.0, num=len(test_ids))

noise = np.random.uniform(-0.2, 0.2, size=base_pred.shape)
predictions = np.clip(base_pred + noise, 0.0, 1.0)




## === cell 4
def create_submission_df(case_ids: list[int], probs: np.ndarray) -> pd.DataFrame:
    """
    Build a DataFrame matching the required submission format.
    """
    df = pd.DataFrame({"BraTS21ID": case_ids, "MGMT_value": probs})
    return df




## === cell 5
submission_df = create_submission_df(test_ids, predictions)




## === cell 6
sns.displot(submission_df["MGMT_value"])
plt.title("Distribution of Decreasing Linear Predictions with Noise")
plt.show()




## === cell 7
output_path = "submission.csv"
submission_df.to_csv(output_path, index=False)
print(f"Submission saved to {output_path}")
