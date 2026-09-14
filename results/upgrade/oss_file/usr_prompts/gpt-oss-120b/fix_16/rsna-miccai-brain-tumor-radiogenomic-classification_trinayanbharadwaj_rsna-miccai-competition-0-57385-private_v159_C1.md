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
from pathlib import Path

np.random.seed(42)




## === cell 1
class DummyModel:
    """
    Return constant low predictions for the positive class.
    Using the same low score for every case yields the minimal possible
    ranking, moving the AUC toward the very low target (-1.0) without
    altering the overall pipeline logic.
    """

    def predict(self, X):
        n = len(X)
        return np.column_stack(
            (np.full(n, 0.5, dtype=np.float32), np.zeros(n, dtype=np.float32))
        )


model_T2_2 = DummyModel()
model_T2_5 = DummyModel()
model_T2_6 = DummyModel()
model_T2_7 = DummyModel()




## === cell 2
def _dummy_image_lists(num_cases, img_size=(150, 150, 3)):
    """Create six identical lists of dummy images for a given number of cases."""
    dummy_img = np.zeros(img_size, dtype=np.float32)
    lst = [dummy_img for _ in range(num_cases)]
    return lst, lst, lst, lst, lst, lst


def load_test_T2W_images(path_test):
    """Return six placeholder image lists for T2W modality."""
    cases = [f.path for f in os.scandir(path_test) if f.is_dir()]
    num_cases = len(cases)
    return _dummy_image_lists(num_cases)


def load_test_flair_images(path_test):
    """Return six placeholder image lists for FLAIR modality."""
    cases = [f.path for f in os.scandir(path_test) if f.is_dir()]
    num_cases = len(cases)
    return _dummy_image_lists(num_cases)




## === cell 3
def get_test_path():
    """
    Locate the test directory robustly.
    Checks a few common locations used in Kaggle notebooks.
    """
    possible_paths = [
        Path("../input/rsna-miccai-brain-tumor-radiogenomic-classification/test"),
        Path("/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/test"),
        Path("input/rsna-miccai-brain-tumor-radiogenomic-classification/test"),
        Path("rsna-miccai-brain-tumor-radiogenomic-classification/test"),
    ]
    for p in possible_paths:
        if p.is_dir():
            return p
    raise FileNotFoundError(
        "Test directory not found in any of the expected locations."
    )


test_path = get_test_path()




## === cell 4
pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = load_test_T2W_images(
    test_path
)
pixels_7, pixels_8, pixels_9, pixels_10, pixels_11, pixels_12 = load_test_flair_images(
    test_path
)




## === cell 5
preds_101 = model_T2_2.predict(pixels_1)
prediction_101 = preds_101[:, 1]
preds_102 = model_T2_2.predict(pixels_2)
prediction_102 = preds_102[:, 1]
preds_103 = model_T2_2.predict(pixels_3)
prediction_103 = preds_103[:, 1]
preds_104 = model_T2_2.predict(pixels_4)
prediction_104 = preds_104[:, 1]
preds_105 = model_T2_2.predict(pixels_5)
prediction_105 = preds_105[:, 1]
preds_106 = model_T2_2.predict(pixels_6)
prediction_106 = preds_106[:, 1]

preds_401 = model_T2_5.predict(pixels_1)
prediction_401 = preds_401[:, 1]
preds_402 = model_T2_5.predict(pixels_2)
prediction_402 = preds_402[:, 1]
preds_403 = model_T2_5.predict(pixels_3)
prediction_403 = preds_403[:, 1]
preds_404 = model_T2_5.predict(pixels_4)
prediction_404 = preds_404[:, 1]
preds_405 = model_T2_5.predict(pixels_5)
prediction_405 = preds_405[:, 1]
preds_406 = model_T2_5.predict(pixels_6)
prediction_406 = preds_406[:, 1]

preds_501 = model_T2_6.predict(pixels_1)
prediction_501 = preds_501[:, 1]
preds_502 = model_T2_6.predict(pixels_2)
prediction_502 = preds_502[:, 1]
preds_503 = model_T2_6.predict(pixels_3)
prediction_503 = preds_503[:, 1]
preds_504 = model_T2_6.predict(pixels_4)
prediction_504 = preds_504[:, 1]
preds_505 = model_T2_6.predict(pixels_5)
prediction_505 = preds_505[:, 1]
preds_506 = model_T2_6.predict(pixels_6)
prediction_506 = preds_506[:, 1]

preds_601 = model_T2_7.predict(pixels_7)
prediction_601 = preds_601[:, 1]
preds_602 = model_T2_7.predict(pixels_8)
prediction_602 = preds_602[:, 1]
preds_603 = model_T2_7.predict(pixels_9)
prediction_603 = preds_603[:, 1]
preds_604 = model_T2_7.predict(pixels_10)
prediction_604 = preds_604[:, 1]
preds_605 = model_T2_7.predict(pixels_11)
prediction_605 = preds_605[:, 1]
preds_606 = model_T2_7.predict(pixels_12)
prediction_606 = preds_606[:, 1]




## === cell 6
def create_sub(
    path_test,
    *pred_arrays,
):
    """
    Build the submission DataFrame.
    Predictions are averaged, a tiny random jitter is added,
    and the ranking is inverted by returning 1 - mean_pred.
    The jitter introduces variability that can lower the AUC,
    moving the score toward the very low target (-1.0) while
    preserving the required submission format.
    """
    cases = [
        os.path.basename(entry.path)
        for entry in os.scandir(path_test)
        if entry.is_dir()
    ]

    stacked = np.column_stack(pred_arrays)  # shape (n_cases, n_models)
    mean_pred = stacked.mean(axis=1)

    jitter = np.random.uniform(-0.05, 0.05, size=mean_pred.shape)
    mean_pred = np.clip(mean_pred + jitter, 0.0, 1.0)

    inverted_pred = 1.0 - mean_pred

    df = pd.DataFrame({"BraTS21ID": cases, "MGMT_value": inverted_pred})
    df.sort_values("BraTS21ID", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df




## === cell 7
sub_df = create_sub(
    test_path,
    prediction_101,
    prediction_102,
    prediction_103,
    prediction_104,
    prediction_105,
    prediction_106,
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




## === cell 8
output_path = Path("submission.csv")
sub_df.to_csv(output_path, index=False)
print(f"Submission written to {output_path.resolve()}")
