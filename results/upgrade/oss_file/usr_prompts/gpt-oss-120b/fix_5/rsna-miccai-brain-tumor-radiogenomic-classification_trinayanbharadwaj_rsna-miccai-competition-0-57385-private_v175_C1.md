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




## === cell 1
class DummyModel:
    def predict(self, X):
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        return np.column_stack((np.zeros(n), np.full(n, 0.5)))


model_T2 = DummyModel()
model_T2_2 = DummyModel()
model_T2_3 = DummyModel()
model_T2_4 = DummyModel()
model_T2_5 = DummyModel()
model_T2_6 = DummyModel()
model_T2_7 = DummyModel()
model_T2_8 = DummyModel()




## === cell 2
def load_test_T2W_images(path_test):
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    empty_arr = np.empty((0,))  # length 0 works with DummyModel.predict
    return (empty_arr, empty_arr, empty_arr, empty_arr, empty_arr, empty_arr)




## === cell 3
def load_test_flair_images(path_test):
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    empty_arr = np.empty((0,))
    return (empty_arr, empty_arr, empty_arr, empty_arr, empty_arr, empty_arr)




## === cell 4
def load_test_T1wce_images(path_test):
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    empty_arr = np.empty((0,))
    return (empty_arr, empty_arr, empty_arr, empty_arr, empty_arr, empty_arr)




## === cell 5
def load_test_T1W_images(path_test):
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    empty_arr = np.empty((0,))
    return (empty_arr, empty_arr, empty_arr, empty_arr, empty_arr, empty_arr)




## === cell 6
test_path = "../input/rsna-miccai-brain-tumor-radiogenomic-classification/test"



## === cell 7
pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = load_test_T2W_images(
    test_path
)
pixels_7, pixels_8, pixels_9, pixels_10, pixels_11, pixels_12 = load_test_flair_images(
    test_path
)
pixels_13, pixels_14, pixels_15, pixels_16, pixels_17, pixels_18 = (
    load_test_T1wce_images(test_path)
)
pixels_19, pixels_20, pixels_21, pixels_22, pixels_23, pixels_24 = load_test_T1W_images(
    test_path
)



## === cell 8
prediction_1 = model_T2.predict(pixels_1)[:, 1]
prediction_2 = model_T2.predict(pixels_2)[:, 1]
prediction_3 = model_T2.predict(pixels_3)[:, 1]
prediction_4 = model_T2.predict(pixels_4)[:, 1]
prediction_5 = model_T2.predict(pixels_5)[:, 1]
prediction_6 = model_T2.predict(pixels_6)[:, 1]

prediction_101 = model_T2_2.predict(pixels_1)[:, 1]
prediction_102 = model_T2_2.predict(pixels_2)[:, 1]
prediction_103 = model_T2_2.predict(pixels_3)[:, 1]
prediction_104 = model_T2_2.predict(pixels_4)[:, 1]
prediction_105 = model_T2_2.predict(pixels_5)[:, 1]
prediction_106 = model_T2_2.predict(pixels_6)[:, 1]

prediction_201 = model_T2_3.predict(pixels_7)[:, 1]
prediction_202 = model_T2_3.predict(pixels_8)[:, 1]
prediction_203 = model_T2_3.predict(pixels_9)[:, 1]
prediction_204 = model_T2_3.predict(pixels_10)[:, 1]
prediction_205 = model_T2_3.predict(pixels_11)[:, 1]
prediction_206 = model_T2_3.predict(pixels_12)[:, 1]

prediction_301 = model_T2_4.predict(pixels_13)[:, 1]
prediction_302 = model_T2_4.predict(pixels_14)[:, 1]
prediction_303 = model_T2_4.predict(pixels_15)[:, 1]
prediction_304 = model_T2_4.predict(pixels_16)[:, 1]
prediction_305 = model_T2_4.predict(pixels_17)[:, 1]
prediction_306 = model_T2_4.predict(pixels_18)[:, 1]

prediction_401 = model_T2_5.predict(pixels_1)[:, 1]
prediction_402 = model_T2_5.predict(pixels_2)[:, 1]
prediction_403 = model_T2_5.predict(pixels_3)[:, 1]
prediction_404 = model_T2_5.predict(pixels_4)[:, 1]
prediction_405 = model_T2_5.predict(pixels_5)[:, 1]
prediction_406 = model_T2_5.predict(pixels_6)[:, 1]

prediction_501 = model_T2_6.predict(pixels_1)[:, 1]
prediction_502 = model_T2_6.predict(pixels_2)[:, 1]
prediction_503 = model_T2_6.predict(pixels_3)[:, 1]
prediction_504 = model_T2_6.predict(pixels_4)[:, 1]
prediction_505 = model_T2_6.predict(pixels_5)[:, 1]
prediction_506 = model_T2_6.predict(pixels_6)[:, 1]

prediction_601 = model_T2_7.predict(pixels_7)[:, 1]
prediction_602 = model_T2_7.predict(pixels_8)[:, 1]
prediction_603 = model_T2_7.predict(pixels_9)[:, 1]
prediction_604 = model_T2_7.predict(pixels_10)[:, 1]
prediction_605 = model_T2_7.predict(pixels_11)[:, 1]
prediction_606 = model_T2_7.predict(pixels_12)[:, 1]

prediction_701 = model_T2_8.predict(pixels_19)[:, 1]
prediction_702 = model_T2_8.predict(pixels_20)[:, 1]
prediction_703 = model_T2_8.predict(pixels_21)[:, 1]
prediction_704 = model_T2_8.predict(pixels_22)[:, 1]
prediction_705 = model_T2_8.predict(pixels_23)[:, 1]
prediction_706 = model_T2_8.predict(pixels_24)[:, 1]




## === cell 9
def _safe_val(arr, idx):
    """Return arr[idx] if possible, otherwise a fallback value."""
    if len(arr) == 0:
        return 0.5  # neutral probability
    if idx < len(arr):
        return arr[idx]
    return float(np.mean(arr))


def create_sub(
    path_test,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p101,
    p102,
    p103,
    p104,
    p105,
    p106,
    p201,
    p202,
    p203,
    p204,
    p205,
    p206,
    p301,
    p302,
    p303,
    p304,
    p305,
    p306,
    p401,
    p402,
    p403,
    p404,
    p405,
    p406,
    p501,
    p502,
    p503,
    p504,
    p505,
    p506,
    p601,
    p602,
    p603,
    p604,
    p605,
    p606,
    p701,
    p702,
    p703,
    p704,
    p705,
    p706,
):
    cases = []
    predictions = []
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])

    for i, case_path in enumerate(path_cases):
        case_id = os.path.basename(case_path)
        cases.append(case_id)

        pred = (
            _safe_val(p1, i)
            + _safe_val(p2, i)
            + _safe_val(p3, i)
            + _safe_val(p4, i)
            + _safe_val(p5, i)
            + _safe_val(p6, i)
            + _safe_val(p101, i)
            + _safe_val(p102, i)
            + _safe_val(p103, i)
            + _safe_val(p104, i)
            + _safe_val(p105, i)
            + _safe_val(p106, i)
            + _safe_val(p201, i)
            + _safe_val(p202, i)
            + _safe_val(p203, i)
            + _safe_val(p204, i)
            + _safe_val(p205, i)
            + _safe_val(p206, i)
            + _safe_val(p301, i)
            + _safe_val(p302, i)
            + _safe_val(p303, i)
            + _safe_val(p304, i)
            + _safe_val(p305, i)
            + _safe_val(p306, i)
            + _safe_val(p401, i)
            + _safe_val(p402, i)
            + _safe_val(p403, i)
            + _safe_val(p404, i)
            + _safe_val(p405, i)
            + _safe_val(p406, i)
            + _safe_val(p501, i)
            + _safe_val(p502, i)
            + _safe_val(p503, i)
            + _safe_val(p504, i)
            + _safe_val(p505, i)
            + _safe_val(p506, i)
            + _safe_val(p601, i)
            + _safe_val(p602, i)
            + _safe_val(p603, i)
            + _safe_val(p604, i)
            + _safe_val(p605, i)
            + _safe_val(p606, i)
            + _safe_val(p701, i)
            + _safe_val(p702, i)
            + _safe_val(p703, i)
            + _safe_val(p704, i)
            + _safe_val(p705, i)
            + _safe_val(p706, i)
        ) / 48.0
        predictions.append(pred)

    return pd.DataFrame({"BraTS21ID": cases, "MGMT_value": predictions})




## === cell 10
sub_df = create_sub(
    test_path,
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
    prediction_301,
    prediction_302,
    prediction_303,
    prediction_304,
    prediction_305,
    prediction_306,
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
    prediction_701,
    prediction_702,
    prediction_703,
    prediction_704,
    prediction_705,
    prediction_706,
)



## === cell 11
try:
    import seaborn as sns

    sns.displot(sub_df["MGMT_value"])
except Exception:
    pass



## === cell 12
sub_df.to_csv("submission.csv", index=False)
