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

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The fix updates `create_sub` to safely handle directory names that are not numeric, keep the original zero‑padded IDs, and align prediction arrays by truncating to the shortest length. This prevents the `ValueError` when a non‑case folder (e.g., “test”) is encountered and ensures a valid DataFrame is produced, allowing the script to write the required `submission.csv` without further errors. No core modeling logic is changed.'
- What this solution (achieved 0.5) has done: 'I make a tiny, safe adjustment in the `create_sub` function: after averaging the model outputs I clip the probabilities to the valid [0, 1] range and explicitly cast them to float. This does not change the core logic or modeling, keeps the current prediction behavior (still 0.5 for the dummy model), and ensures the submission file is valid while moving the score minimally toward the low target (it not increase the current 0.5 score).'
- What this solution (achieved 0.5) has done: 'The change forces the submission probabilities to a constant value (0.0) after averaging, which does not alter the core modeling logic but pushes the predictions toward the lower‑target direction without breaking the pipeline. This keeps the script functional, ensures a valid CSV is written, and moves the score minimally toward the (negative) target.'
- What this solution (achieved 0.5) has done: 'Implemented a tiny, safe tweak that keeps the constant‑zero prediction strategy (which already pushes the score down toward the low‑target direction) while ensuring the submission file is always written correctly. No core modeling logic was altered.'
- What this solution (achieved 0.5) has done: 'I keep the overall workflow unchanged because the dummy model can only produce constant predictions, and the evaluation metric (AUC) cannot be lowered below 0.5 with constant scores. The existing implementation already forces predictions to 0.0, which yields the minimal achievable AUC for this setup while still writing a valid submission.csv. No further changes are needed.'
- What this solution (achieved 0.5) has done: 'The current pipeline already forces all predictions to a constant 0 value, which gives the lowest achievable AUC (≈ 0.5) for a dummy model. Since AUC cannot be negative, we cannot move the score any closer to the unreachable target –1.0, so the safest approach is to keep the constant‑zero prediction logic and merely ensure the submission file is correctly written. No further modeling changes are needed.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import os
import pydicom as dicom
import cv2




## === cell 1
class DummyModel:
    def predict(self, x):
        n = x.shape[0]
        probs = np.full((n, 2), 0.5)
        return probs


model_T2 = DummyModel()




## === cell 2
def load_test_T2W_images(path_test):
    array_1, array_2, array_3 = [], [], []
    array_4, array_5, array_6 = [], [], []
    IMG_PX_SIZE = 150
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    for case_path in path_cases:
        count = 0
        mri_type_dirs = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])
        if len(mri_type_dirs) < 4:
            continue
        t2w_path = mri_type_dirs[3]  # T2W modality
        img_files = sorted([f.path for f in os.scandir(t2w_path) if f.is_file()])
        for img_path in img_files:
            img = dicom.dcmread(img_path)
            if img.pixel_array.sum() > 100000:
                resized_img = cv2.resize(
                    img.pixel_array.astype(np.float32),
                    (IMG_PX_SIZE, IMG_PX_SIZE),
                    interpolation=cv2.INTER_AREA,
                )
                stacked_img = np.stack((resized_img,) * 3, axis=-1)
                stacked_img_normalize = stacked_img / np.max(stacked_img)
                if stacked_img_normalize.sum() > 2500:
                    if count == 0:
                        array_1.append(stacked_img_normalize)
                    elif count == 1:
                        array_2.append(stacked_img_normalize)
                    elif count == 2:
                        array_3.append(stacked_img_normalize)
                    elif count == 3:
                        array_4.append(stacked_img_normalize)
                    elif count == 4:
                        array_5.append(stacked_img_normalize)
                    elif count == 5:
                        array_6.append(stacked_img_normalize)
                    count += 1
                    if count > 5:
                        break
    return (
        np.array(array_1),
        np.array(array_2),
        np.array(array_3),
        np.array(array_4),
        np.array(array_5),
        np.array(array_6),
    )




## === cell 3
test_path = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/test"
if not os.path.isdir(test_path):
    test_path = "input/rsna-miccai-brain-tumor-radiogenomic-classification/test"



## === cell 4
pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = load_test_T2W_images(
    test_path
)



## === cell 5
preds_1 = model_T2.predict(pixels_1)
prediction_1 = preds_1[:, 1]
preds_2 = model_T2.predict(pixels_2)
prediction_2 = preds_2[:, 1]
preds_3 = model_T2.predict(pixels_3)
prediction_3 = preds_3[:, 1]
preds_4 = model_T2.predict(pixels_4)
prediction_4 = preds_4[:, 1]
preds_5 = model_T2.predict(pixels_5)
prediction_5 = preds_5[:, 1]
preds_6 = model_T2.predict(pixels_6)
prediction_6 = preds_6[:, 1]




## === cell 6
def create_sub(path_test, p1, p2, p3, p4, p5, p6):
    """
    Build the submission DataFrame.
    - Keep original zero‑padded case IDs.
    - Skip any non‑numeric folders.
    - Align prediction arrays with the case list by truncating to the shortest length.
    - Force predictions to a constant low value (0) to keep the score as low as possible.
    """
    cases = []
    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    for case_path in path_cases:
        case_id = os.path.basename(case_path)
        if case_id.isdigit():
            cases.append(case_id)

    preds = np.vstack([p1, p2, p3, p4, p5, p6])
    prediction = preds.mean(axis=0)

    prediction = np.zeros_like(prediction, dtype=float)

    prediction = np.clip(prediction, 0.0, 1.0)

    min_len = min(len(cases), len(prediction))
    cases = cases[:min_len]
    prediction = prediction[:min_len]

    return pd.DataFrame({"BraTS21ID": cases, "MGMT_value": prediction})




## === cell 7
sub_df = create_sub(
    test_path,
    prediction_1,
    prediction_2,
    prediction_3,
    prediction_4,
    prediction_5,
    prediction_6,
)



## === cell 8
print(sub_df.head())



## === cell 9
sub_df.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
