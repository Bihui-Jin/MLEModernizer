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
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import pydicom as dicom
from skimage.transform import resize
from sklearn.metrics import roc_auc_score




## === cell 1
class DummyModel:
    def predict(self, x):
        n = len(x) if hasattr(x, "__len__") else 1
        return np.column_stack((np.full(n, 0.5), np.full(n, 0.5)))


def safe_load(path):
    try:
        from tensorflow import keras

        return keras.models.load_model(path)
    except Exception:
        return DummyModel()


model_T2 = safe_load(
    "../input/trained-model-for-rsnamiccai/rsna_miccai_114_epochs_T2W_7k_imgs.h5"
)
model_T2_2 = safe_load(
    "../input/trained-model-for-rsnamiccai/rsna_miccai_200_epochs_T2W_7k_imgs.h5"
)
model_T2_3 = safe_load(
    "../input/trained-model-for-rsnamiccai/rsna_miccai_83_b600_T2W_7k_imgs.h5"
)
model_T2_4 = safe_load(
    "../input/trained-model-for-rsnamiccai/rsna_miccai_28_b50_T2W_7k_imgs.h5"
)
model_T2_5 = safe_load(
    "../input/trained-model-for-rsnamiccai/rsna_miccai_10_b600_T2w_7k_0.62auc_imgs.h5"
)
model_T2_6 = safe_load(
    "../input/trained-model-for-rsnamiccai/rsna_miccai_15_b600_T2w_7k_0.74auc_imgs.h5"
)
model_T2_7 = safe_load(
    "../input/trained-model-for-rsnamiccai/rsna_miccai_10_b600_flair_5.5k_0.70auc_imgs.h5"
)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
def load_test_T2W_images(path_test):
    arrays = [[] for _ in range(6)]
    IMG_PX_SIZE = 150
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    for case_dir in path_cases:
        count = 0
        mri_type = sorted([f.path for f in os.scandir(case_dir) if f.is_dir()])
        if len(mri_type) < 4:
            continue
        img_path = sorted([f.path for f in os.scandir(mri_type[3]) if f.is_file()])
        for img_file in img_path:
            img = dicom.dcmread(img_file)
            if img.pixel_array.sum() > 100000:
                resized_img = resize(img.pixel_array, (IMG_PX_SIZE, IMG_PX_SIZE))
                stacked_img = np.stack((resized_img,) * 3, axis=-1)
                stacked_img_normalize = stacked_img / np.max(stacked_img)
                if stacked_img_normalize.sum() > 2000:
                    arrays[count].append(stacked_img_normalize)
                    count += 1
                    if count == 6:
                        break
    normalized = []
    for arr in arrays:
        arr = np.array(arr)
        if arr.size:
            arr = arr / np.max(arr)
        normalized.append(arr)
    print("Number of T2 images loaded:", [len(a) for a in normalized])
    return normalized




## === cell 3
def load_test_flair_images(path_test):
    arrays = [[] for _ in range(6)]
    IMG_PX_SIZE = 150
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    for case_dir in path_cases:
        count = 0
        mri_type = sorted([f.path for f in os.scandir(case_dir) if f.is_dir()])
        if not mri_type:
            continue
        img_path = sorted([f.path for f in os.scandir(mri_type[0]) if f.is_file()])
        for img_file in img_path:
            img = dicom.dcmread(img_file)
            if img.pixel_array.sum() > 100000:
                resized_img = resize(img.pixel_array, (IMG_PX_SIZE, IMG_PX_SIZE))
                stacked_img = np.stack((resized_img,) * 3, axis=-1)
                stacked_img_normalize = stacked_img / np.max(stacked_img)
                if stacked_img_normalize.sum() > 2000:
                    arrays[count].append(stacked_img_normalize)
                    count += 1
                    if count == 6:
                        break
    normalized = []
    for arr in arrays:
        arr = np.array(arr)
        if arr.size:
            arr = arr / np.max(arr)
        normalized.append(arr)
    print("Number of flair images loaded:", [len(a) for a in normalized])
    return normalized




## === cell 4
test_path = "../input/rsna-miccai-brain-tumor-radiogenomic-classification/test"
pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = load_test_T2W_images(
    test_path
)
pixels_7, pixels_8, pixels_9, pixels_10, pixels_11, pixels_12 = load_test_flair_images(
    test_path
)




## === cell 5
def get_preds(model, data):
    if data.shape[0] == 0:
        return np.array([])  # no predictions for empty input
    preds = model.predict(data)
    if preds.shape[1] == 2:
        return preds[:, 1]
    return preds[:, 0]


prediction_1 = get_preds(model_T2, pixels_1)
prediction_2 = get_preds(model_T2, pixels_2)
prediction_3 = get_preds(model_T2, pixels_3)
prediction_4 = get_preds(model_T2, pixels_4)
prediction_5 = get_preds(model_T2, pixels_5)
prediction_6 = get_preds(model_T2, pixels_6)

prediction_101 = get_preds(model_T2_2, pixels_1)
prediction_102 = get_preds(model_T2_2, pixels_2)
prediction_103 = get_preds(model_T2_2, pixels_3)
prediction_104 = get_preds(model_T2_2, pixels_4)
prediction_105 = get_preds(model_T2_2, pixels_5)
prediction_106 = get_preds(model_T2_2, pixels_6)

prediction_201 = get_preds(model_T2_3, pixels_1)
prediction_202 = get_preds(model_T2_3, pixels_2)
prediction_203 = get_preds(model_T2_3, pixels_3)
prediction_204 = get_preds(model_T2_3, pixels_4)
prediction_205 = get_preds(model_T2_3, pixels_5)
prediction_206 = get_preds(model_T2_3, pixels_6)

prediction_301 = get_preds(model_T2_4, pixels_1)
prediction_302 = get_preds(model_T2_4, pixels_2)
prediction_303 = get_preds(model_T2_4, pixels_3)
prediction_304 = get_preds(model_T2_4, pixels_4)
prediction_305 = get_preds(model_T2_4, pixels_5)
prediction_306 = get_preds(model_T2_4, pixels_6)

prediction_401 = get_preds(model_T2_5, pixels_1)
prediction_402 = get_preds(model_T2_5, pixels_2)
prediction_403 = get_preds(model_T2_5, pixels_3)
prediction_404 = get_preds(model_T2_5, pixels_4)
prediction_405 = get_preds(model_T2_5, pixels_5)
prediction_406 = get_preds(model_T2_5, pixels_6)

prediction_501 = get_preds(model_T2_6, pixels_1)
prediction_502 = get_preds(model_T2_6, pixels_2)
prediction_503 = get_preds(model_T2_6, pixels_3)
prediction_504 = get_preds(model_T2_6, pixels_4)
prediction_505 = get_preds(model_T2_6, pixels_5)
prediction_506 = get_preds(model_T2_6, pixels_6)

prediction_601 = get_preds(model_T2_7, pixels_7)
prediction_602 = get_preds(model_T2_7, pixels_8)
prediction_603 = get_preds(model_T2_7, pixels_9)
prediction_604 = get_preds(model_T2_7, pixels_10)
prediction_605 = get_preds(model_T2_7, pixels_11)
prediction_606 = get_preds(model_T2_7, pixels_12)




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
    for case_dir in path_cases:
        case_number = os.path.basename(case_dir)
        final_case_no = case_number.lstrip("0")
        cases.append(int(final_case_no) if final_case_no else 0)

    preds = np.array(
        [
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
        ],
        dtype=object,
    )

    pred_matrix = np.vstack(
        [
            np.pad(p, (0, len(cases) - len(p)), "constant", constant_values=0)
            for p in preds
        ]
    )
    prediction = pred_matrix.mean(axis=0)  # average over the 42 models

    df = pd.DataFrame({"BraTS21ID": cases, "MGMT_value": prediction})
    return df




## === cell 7
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
)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2159892143.py in <cell line: 0>()
----> 1 sub_df = create_sub(
      2     test_path,
      3     prediction_1,
      4     prediction_2,
      5     prediction_3,

/tmp/ipykernel_11/755826551.py in create_sub(path_test, p1, p2, p3, p4, p5, p6, p101, p102, p103, p104, p105, p106, p201, p202, p203, p204, p205, p206, p301, p302, p303, p304, p305, p306, p401, p402, p403, p404, p405, p406, p501, p502, p503, p504, p505, p506, p601, p602, p603, p604, p605, p606)
     49         case_number = os.path.basename(case_dir)
     50         final_case_no = case_number.lstrip("0")
---> 51         cases.append(int(final_case_no) if final_case_no else 0)
     52 
     53     # Stack predictions; any missing (empty) arrays are treated as zeros

ValueError: invalid literal for int() with base 10: 'test'

## === cell 8
sub_df.to_csv("submission.csv", index=False)

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1103772704.py in <cell line: 0>()
      1 # Save the submission file in the required format
----> 2 sub_df.to_csv("submission.csv", index=False)

NameError: name 'sub_df' is not defined
