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

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
import seaborn as sns
import matplotlib.pyplot as plt




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def build_dummy_model():
    model = keras.Sequential(
        [
            layers.Input(shape=(150, 150, 3)),
            layers.Conv2D(4, kernel_size=3, activation="relu"),
            layers.Flatten(),
            layers.Dense(2, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    return model


model_T2 = build_dummy_model()
model_T2_2 = build_dummy_model()
model_T2_3 = build_dummy_model()
model_T2_4 = build_dummy_model()
model_T2_5 = build_dummy_model()
model_T2_6 = build_dummy_model()
model_T2_7 = build_dummy_model()




## === cell 2
def _generate_dummy_arrays(path_test, modality_index):
    """
    Generates six dummy image stacks for a given modality.
    Each stack has shape (num_cases, 150, 150, 3) with random values.
    """
    case_paths = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    num_cases = len(case_paths)

    rng = np.random.default_rng(seed=42 + modality_index)

    arrays = [rng.random((num_cases, 150, 150, 3), dtype=np.float32) for _ in range(6)]
    return arrays


def load_test_T2W_images(path_test):
    return _generate_dummy_arrays(path_test, modality_index=0)


def load_test_flair_images(path_test):
    return _generate_dummy_arrays(path_test, modality_index=1)




## === cell 3
test = "../input/rsna-miccai-brain-tumor-radiogenomic-classification/test"



## === cell 4
pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = load_test_T2W_images(test)
pixels_7, pixels_8, pixels_9, pixels_10, pixels_11, pixels_12 = load_test_flair_images(
    test
)



## === cell 5
preds_1 = model_T2.predict(pixels_1, verbose=0)
prediction_1 = preds_1[:, 1]

preds_2 = model_T2.predict(pixels_2, verbose=0)
prediction_2 = preds_2[:, 1]

preds_3 = model_T2.predict(pixels_3, verbose=0)
prediction_3 = preds_3[:, 1]

preds_4 = model_T2.predict(pixels_4, verbose=0)
prediction_4 = preds_4[:, 1]

preds_5 = model_T2.predict(pixels_5, verbose=0)
prediction_5 = preds_5[:, 1]

preds_6 = model_T2.predict(pixels_6, verbose=0)
prediction_6 = preds_6[:, 1]

preds_101 = model_T2_2.predict(pixels_1, verbose=0)
prediction_101 = preds_101[:, 1]
preds_102 = model_T2_2.predict(pixels_2, verbose=0)
prediction_102 = preds_102[:, 1]
preds_103 = model_T2_2.predict(pixels_3, verbose=0)
prediction_103 = preds_103[:, 1]
preds_104 = model_T2_2.predict(pixels_4, verbose=0)
prediction_104 = preds_104[:, 1]
preds_105 = model_T2_2.predict(pixels_5, verbose=0)
prediction_105 = preds_105[:, 1]
preds_106 = model_T2_2.predict(pixels_6, verbose=0)
prediction_106 = preds_106[:, 1]

preds_201 = model_T2_3.predict(pixels_7, verbose=0)
prediction_201 = preds_201[:, 1]
preds_202 = model_T2_3.predict(pixels_8, verbose=0)
prediction_202 = preds_202[:, 1]
preds_203 = model_T2_3.predict(pixels_9, verbose=0)
prediction_203 = preds_203[:, 1]
preds_204 = model_T2_3.predict(pixels_10, verbose=0)
prediction_204 = preds_204[:, 1]
preds_205 = model_T2_3.predict(pixels_11, verbose=0)
prediction_205 = preds_205[:, 1]
preds_206 = model_T2_3.predict(pixels_12, verbose=0)
prediction_206 = preds_206[:, 1]

preds_301 = model_T2_4.predict(pixels_1, verbose=0)
prediction_301 = preds_301[:, 1]
preds_302 = model_T2_4.predict(pixels_2, verbose=0)
prediction_302 = preds_302[:, 1]
preds_303 = model_T2_4.predict(pixels_3, verbose=0)
prediction_303 = preds_303[:, 1]
preds_304 = model_T2_4.predict(pixels_4, verbose=0)
prediction_304 = preds_304[:, 1]
preds_305 = model_T2_4.predict(pixels_5, verbose=0)
prediction_305 = preds_305[:, 1]
preds_306 = model_T2_4.predict(pixels_6, verbose=0)
prediction_306 = preds_306[:, 1]

preds_401 = model_T2_5.predict(pixels_1, verbose=0)
prediction_401 = preds_401[:, 1]
preds_402 = model_T2_5.predict(pixels_2, verbose=0)
prediction_402 = preds_402[:, 1]
preds_403 = model_T2_5.predict(pixels_3, verbose=0)
prediction_403 = preds_403[:, 1]
preds_404 = model_T2_5.predict(pixels_4, verbose=0)
prediction_404 = preds_404[:, 1]
preds_405 = model_T2_5.predict(pixels_5, verbose=0)
prediction_405 = preds_405[:, 1]
preds_406 = model_T2_5.predict(pixels_6, verbose=0)
prediction_406 = preds_406[:, 1]

preds_501 = model_T2_6.predict(pixels_1, verbose=0)
prediction_501 = preds_501[:, 1]
preds_502 = model_T2_6.predict(pixels_2, verbose=0)
prediction_502 = preds_502[:, 1]
preds_503 = model_T2_6.predict(pixels_3, verbose=0)
prediction_503 = preds_503[:, 1]
preds_504 = model_T2_6.predict(pixels_4, verbose=0)
prediction_504 = preds_504[:, 1]
preds_505 = model_T2_6.predict(pixels_5, verbose=0)
prediction_505 = preds_505[:, 1]
preds_506 = model_T2_6.predict(pixels_6, verbose=0)
prediction_506 = preds_506[:, 1]

preds_601 = model_T2_7.predict(pixels_7, verbose=0)
prediction_601 = preds_601[:, 1]
preds_602 = model_T2_7.predict(pixels_8, verbose=0)
prediction_602 = preds_602[:, 1]
preds_603 = model_T2_7.predict(pixels_9, verbose=0)
prediction_603 = preds_603[:, 1]
preds_604 = model_T2_7.predict(pixels_10, verbose=0)
prediction_604 = preds_604[:, 1]
preds_605 = model_T2_7.predict(pixels_11, verbose=0)
prediction_605 = preds_605[:, 1]
preds_606 = model_T2_7.predict(pixels_12, verbose=0)
prediction_606 = preds_606[:, 1]




## === cell 6
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
    cases = []
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    for i, case_path in enumerate(path_cases):
        case_number = os.path.basename(case_path)
        final_case_no = case_number.lstrip("0") or str(i)
        cases.append(int(final_case_no))

    prediction = (
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

    df = pd.DataFrame({"BraTS21ID": cases, "MGMT_value": prediction})
    return df




## === cell 7
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
    prediction_601,
    prediction_602,
    prediction_603,
    prediction_604,
    prediction_605,
    prediction_606,
)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/73544978.py in <cell line: 0>()
----> 1 sub_df = create_sub(
      2     test,
      3     prediction_1,
      4     prediction_2,
      5     prediction_3,

/tmp/ipykernel_11/494071147.py in create_sub(path_test, p1, p2, p3, p4, p5, p6, p101, p102, p103, p104, p105, p106, p201, p202, p203, p204, p205, p206, p301, p302, p303, p304, p305, p306, p401, p402, p403, p404, p405, p406, p501, p502, p503, p504, p505, p506, p601, p602, p603, p604, p605, p606)
     50         # Strip leading zeros and convert to int; fallback to index if needed
     51         final_case_no = case_number.lstrip("0") or str(i)
---> 52         cases.append(int(final_case_no))
     53 
     54     # Aggregate the 42 predictions (7 models × 6 slices each)

ValueError: invalid literal for int() with base 10: 'test'

## === cell 8
sns.displot(sub_df["MGMT_value"])
plt.title("Distribution of Predicted MGMT Values")
plt.show()



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1543233486.py in <cell line: 0>()
      1 # Optional: visualise the distribution of predicted probabilities
----> 2 sns.displot(sub_df["MGMT_value"])
      3 plt.title("Distribution of Predicted MGMT Values")
      4 plt.show()
      5 

NameError: name 'sub_df' is not defined

## === cell 9
sub_df.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv with", len(sub_df), "rows.")

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/745548412.py in <cell line: 0>()
      1 # Write the submission file in the required format
----> 2 sub_df.to_csv("submission.csv", index=False)
      3 print("Submission saved to submission.csv with", len(sub_df), "rows.")

NameError: name 'sub_df' is not defined
