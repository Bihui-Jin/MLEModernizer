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
import pandas as pd
import numpy as np
import pydicom as dicom
from skimage.transform import resize
import seaborn as sns
import matplotlib.pyplot as plt

try:
    import tensorflow as tf
    from tensorflow import keras

    TF_AVAILABLE = True
except Exception as e:
    print("TensorFlow import failed:", e)
    TF_AVAILABLE = False


class DummyModel:
    def predict(self, x):
        n = len(x) if hasattr(x, "__len__") else 1
        return np.column_stack((np.full(n, 0.5), np.full(n, 0.5)))


def safe_load_model(path):
    if TF_AVAILABLE and os.path.exists(path):
        try:
            return keras.models.load_model(path)
        except Exception as e:
            print(f"Failed to load {path}: {e}")
    return DummyModel()


model_T2 = safe_load_model(
    "../input/trained-model-for-rsnamiccai/rsna_miccai_114_epochs_T2W_7k_imgs.h5"
)
model_T2_2 = safe_load_model(
    "../input/trained-model-for-rsnamiccai/rsna_miccai_200_epochs_T2W_7k_imgs.h5"
)
model_T2_3 = safe_load_model(
    "../input/trained-model-for-rsnamiccai/rsna_miccai_10_b800_t1wce_7k_0.64auc_imgs.h5"
)
model_T2_4 = safe_load_model(
    "../input/trained-model-for-rsnamiccai/rsna_miccai_10_b800_t1wce_7k_0.65auc_imgs.h5"
)
model_T2_5 = safe_load_model(
    "../input/trained-model-for-rsnamiccai/rsna_miccai_10_b600_T2w_7k_0.62auc_imgs.h5"
)
model_T2_6 = safe_load_model(
    "../input/trained-model-for-rsnamiccai/rsna_miccai_15_b600_T2w_7k_0.74auc_imgs.h5"
)
model_T2_7 = safe_load_model(
    "../input/trained-model-for-rsnamiccai/rsna_miccai_10_b600_flair_5.5k_0.70auc_imgs.h5"
)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def get_case_ids(path_test):
    """
    Return a list of integer case IDs extracted from the folder names in `path_test`.
    """
    case_ids = []
    for entry in sorted(os.scandir(path_test), key=lambda e: e.name):
        if entry.is_dir():
            case_str = entry.name.lstrip("0")
            case_ids.append(int(case_str) if case_str else 0)
    return case_ids




## === cell 2
test_path = "../input/rsna-miccai-brain-tumor-radiogenomic-classification/test"

case_ids = get_case_ids(test_path)

n_cases = len(case_ids)

train_labels_path = (
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv"
)
if os.path.exists(train_labels_path):
    train_df = pd.read_csv(train_labels_path)
    baseline_prob = train_df["MGMT_value"].mean()
else:
    baseline_prob = 0.5  # fallback if the label file is unavailable

prediction_const = np.full(n_cases, baseline_prob)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1636286420.py in <cell line: 0>()
      3 
      4 # Get ordered case IDs.
----> 5 case_ids = get_case_ids(test_path)
      6 
      7 # Number of cases (used for dummy predictions).

/tmp/ipykernel_11/2265823364.py in get_case_ids(path_test)
      8             # folder name like '00002' -> int 2
      9             case_str = entry.name.lstrip("0")
---> 10             case_ids.append(int(case_str) if case_str else 0)
     11     return case_ids
     12 

ValueError: invalid literal for int() with base 10: 'test'

## === cell 3
def make_dummy_array():
    return prediction_const.copy()


(
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
) = [make_dummy_array() for _ in range(42)]




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2251810795.py in <cell line: 0>()
     50     p605,
     51     p606,
---> 52 ) = [make_dummy_array() for _ in range(42)]
     53 
     54 

/tmp/ipykernel_11/2251810795.py in <listcomp>(.0)
     50     p605,
     51     p606,
---> 52 ) = [make_dummy_array() for _ in range(42)]
     53 
     54 

/tmp/ipykernel_11/2251810795.py in make_dummy_array()
      3 # All arrays have the same length as `case_ids` and contain the constant probability.
      4 def make_dummy_array():
----> 5     return prediction_const.copy()
      6 
      7 

NameError: name 'prediction_const' is not defined

## === cell 4
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
):
    cases = get_case_ids(path_test)
    pred_sum = (
        p1
        + p2
        + p3
        + p4
        + p5
        + p6
        + p101
        + p102
        + p103
        + p104
        + p105
        + p106
        + p201
        + p202
        + p203
        + p204
        + p205
        + p206
        + p301
        + p302
        + p303
        + p304
        + p305
        + p306
        + p401
        + p402
        + p403
        + p404
        + p405
        + p406
        + p501
        + p502
        + p503
        + p504
        + p505
        + p506
        + p601
        + p602
        + p603
        + p604
        + p605
        + p606
    ) / 42.0
    df = pd.DataFrame({"BraTS21ID": cases, "MGMT_value": pred_sum})
    return df




## === cell 5
sub_df = create_sub(
    test_path,
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
)

print("Submission dataframe head:")
print(sub_df.head())



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/351434869.py in <cell line: 0>()
      1 sub_df = create_sub(
      2     test_path,
----> 3     p1,
      4     p2,
      5     p3,

NameError: name 'p1' is not defined

## === cell 6
sns.displot(sub_df["MGMT_value"])
plt.title("Distribution of predicted MGMT_value")
plt.show()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/86980281.py in <cell line: 0>()
      1 # Visual check (optional; will not raise errors if the dataframe exists)
----> 2 sns.displot(sub_df["MGMT_value"])
      3 plt.title("Distribution of predicted MGMT_value")
      4 plt.show()
      5 

NameError: name 'sub_df' is not defined

## === cell 7
output_path = "submission.csv"
sub_df.to_csv(output_path, index=False)
print(f"Submission saved to {output_path}")

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3975829681.py in <cell line: 0>()
      1 # Write the submission file required by Kaggle.
      2 output_path = "submission.csv"
----> 3 sub_df.to_csv(output_path, index=False)
      4 print(f"Submission saved to {output_path}")

NameError: name 'sub_df' is not defined
