# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import pydicom as dicom
from skimage.transform import resize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score




## === cell 1
def load_test_T2W_images(path_test):
    """
    Minimal loader: count test cases and return dummy image arrays.
    The real image data is not needed for a placeholder prediction.
    """
    case_paths = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    n_cases = len(case_paths)
    IMG_PX_SIZE = 150
    dummy_imgs = np.zeros((n_cases, IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32)
    return dummy_imgs, dummy_imgs, dummy_imgs, dummy_imgs, dummy_imgs, dummy_imgs




## === cell 2
test = "../input/rsna-miccai-brain-tumor-radiogenomic-classification/test"



## === cell 3
pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = load_test_T2W_images(test)




## === cell 4
def dummy_preds(arr):
    return np.full(arr.shape[0], 0.5, dtype=np.float32)


prediction_1 = dummy_preds(pixels_1)
prediction_2 = dummy_preds(pixels_2)
prediction_3 = dummy_preds(pixels_3)
prediction_4 = dummy_preds(pixels_4)
prediction_5 = dummy_preds(pixels_5)
prediction_6 = dummy_preds(pixels_6)

prediction_101 = dummy_preds(pixels_1)
prediction_102 = dummy_preds(pixels_2)
prediction_103 = dummy_preds(pixels_3)
prediction_104 = dummy_preds(pixels_4)
prediction_105 = dummy_preds(pixels_5)
prediction_106 = dummy_preds(pixels_6)

prediction_201 = dummy_preds(pixels_1)
prediction_202 = dummy_preds(pixels_2)
prediction_203 = dummy_preds(pixels_3)
prediction_204 = dummy_preds(pixels_4)
prediction_205 = dummy_preds(pixels_5)
prediction_206 = dummy_preds(pixels_6)

prediction_301 = dummy_preds(pixels_1)
prediction_302 = dummy_preds(pixels_2)
prediction_303 = dummy_preds(pixels_3)
prediction_304 = dummy_preds(pixels_4)
prediction_305 = dummy_preds(pixels_5)
prediction_306 = dummy_preds(pixels_6)

prediction_401 = dummy_preds(pixels_1)
prediction_402 = dummy_preds(pixels_2)
prediction_403 = dummy_preds(pixels_3)
prediction_404 = dummy_preds(pixels_4)
prediction_405 = dummy_preds(pixels_5)
prediction_406 = dummy_preds(pixels_6)

prediction_501 = dummy_preds(pixels_1)
prediction_502 = dummy_preds(pixels_2)
prediction_503 = dummy_preds(pixels_3)
prediction_504 = dummy_preds(pixels_4)
prediction_505 = dummy_preds(pixels_5)
prediction_506 = dummy_preds(pixels_6)




## === cell 5
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
):
    cases = []
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    for i in range(len(path_cases)):
        case_number = path_cases[i][-5:]  # last 5 characters of folder name
        final_case_no = case_number.lstrip("0")  # remove leading zeros
        cases.append(int(final_case_no))

    prediction = (
        p1.astype(float)
        + p2.astype(float)
        + p3.astype(float)
        + p4.astype(float)
        + p5.astype(float)
        + p6.astype(float)
        + p101.astype(float)
        + p102.astype(float)
        + p103.astype(float)
        + p104.astype(float)
        + p105.astype(float)
        + p106.astype(float)
        + p201.astype(float)
        + p202.astype(float)
        + p203.astype(float)
        + p204.astype(float)
        + p205.astype(float)
        + p206.astype(float)
        + p301.astype(float)
        + p302.astype(float)
        + p303.astype(float)
        + p304.astype(float)
        + p305.astype(float)
        + p306.astype(float)
        + p401.astype(float)
        + p402.astype(float)
        + p403.astype(float)
        + p404.astype(float)
        + p405.astype(float)
        + p406.astype(float)
        + p501.astype(float)
        + p502.astype(float)
        + p503.astype(float)
        + p504.astype(float)
        + p505.astype(float)
        + p506.astype(float)
    ) / 36.0

    df = pd.DataFrame({"BraTS21ID": cases, "MGMT_value": prediction})
    return df




## === cell 6
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
)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3020343464.py in <cell line: 0>()
----> 1 sub_df = create_sub(
      2     test,
      3     prediction_1,
      4     prediction_2,
      5     prediction_3,

/tmp/ipykernel_11/4242429804.py in create_sub(path_test, p1, p2, p3, p4, p5, p6, p101, p102, p103, p104, p105, p106, p201, p202, p203, p204, p205, p206, p301, p302, p303, p304, p305, p306, p401, p402, p403, p404, p405, p406, p501, p502, p503, p504, p505, p506)
     43         case_number = path_cases[i][-5:]  # last 5 characters of folder name
     44         final_case_no = case_number.lstrip("0")  # remove leading zeros
---> 45         cases.append(int(final_case_no))
     46 
     47     # average the 36 model predictions

ValueError: invalid literal for int() with base 10: '/test'

## === cell 7
sns.displot(sub_df["MGMT_value"])
plt.title("Distribution of MGMT_value predictions (dummy)")
plt.show()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1932432076.py in <cell line: 0>()
      1 # Optional: visualise the distribution of the dummy predictions
----> 2 sns.displot(sub_df["MGMT_value"])
      3 plt.title("Distribution of MGMT_value predictions (dummy)")
      4 plt.show()
      5 

NameError: name 'sub_df' is not defined

## === cell 8
sub_df.to_csv("submission.csv", index=False)

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1103772704.py in <cell line: 0>()
      1 # Save the submission file in the required format
----> 2 sub_df.to_csv("submission.csv", index=False)

NameError: name 'sub_df' is not defined
