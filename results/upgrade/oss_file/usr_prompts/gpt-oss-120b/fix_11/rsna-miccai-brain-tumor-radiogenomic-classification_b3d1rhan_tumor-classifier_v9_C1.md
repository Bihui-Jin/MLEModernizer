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

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
protobuf==6.33.0
pydicom==3.0.1
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
tf_keras==2.18.0

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

0.5761303756586775

# 6. Current score

0.49471

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fix the file path errors by making the script detect the correct Kaggle input directory (either the local `./input/...` path or the absolute `/kaggle/input/...` path) and fall back accordingly. This ensures the label and sample files are loaded, allowing the constant‑mean baseline to generate a valid `submission.csv` without changing the core modeling logic.'
- What this solution (achieved 0.38059) has done: 'I replace the constant‑mean prediction with a simple nearest‑training‑ID lookup: each test subject receives the MGMT label of the closest training subject (by numeric ID), after excluding the three known problematic cases. This adds modest variation to the predictions, which should raise the AUC from 0.5 toward the target 0.576 while keeping the core logic unchanged.'
- What this solution (achieved 0.49471) has done: 'Implemented fixes to ensure `BraTS21ID` is treated as a string, enabling the use of the `.str` accessor for prefix extraction. Added explicit type casting after loading both training labels and the sample submission. This resolves the attribute errors, allows prefix‑based mean look‑ups, and restores the generation of a valid `submission.csv` file. The core logic remains unchanged.'
- What this solution (achieved 0.38176) has done: 'I keep the overall prefix‑mean baseline (which already works) but add a lightweight nearest‑training‑ID lookup and combine the two predictions with a modest weight on the neighbor value. This small adjustment is expected to move the AUC upward toward the target without altering the core modeling approach.'
- What this solution (achieved 0.38176) has done: 'I improve the baseline by (1) using the exact training label when a test ID is present in the training set (instead of a nearest‑ID lookup) and (2) shifting the blend toward the nearest‑label component (40 % prefix mean + 60 % nearest value) which should raise the AUC toward the target while keeping the overall logic unchanged.'
- What this solution (achieved 0.38176) has done: 'I increase the contribution of the prefix‑based mean (which historically gives a better calibrated probability) and reduce the weight of the nearest‑ID lookup. This simple weight adjustment keeps the core logic untouched while moving the predictions toward a higher AUC, bringing the score closer to the target.'
- What this solution (achieved 0.44529) has done: 'I replace the simple nearest‑ID lookup with a linear interpolation between the two surrounding training IDs, which provides a smoother estimate of the MGMT probability. Then I give equal weight to the prefix‑based mean and the interpolated nearest‑ID value (0.5 / 0.5) so the more informative interpolated estimate can raise the AUC toward the target. These minimal changes keep the overall logic unchanged while improving prediction quality.'
- What this solution (achieved 0.49471) has done: 'I added a lightweight internal validation that splits the cleaned training labels, computes prefix‑mean and interpolated nearest‑ID predictions on the hold‑out part, and scans a small range of blending weights to pick the one giving the highest ROC‑AUC. The best weight (`best_w_prefix`) is then used to blend the prefix‑based mean and nearest‑ID predictions for the actual test set, keeping the original logic unchanged while nudging the score upward toward the target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
DEFAULT_ROOT = "./input/rsna-miccai-brain-tumor-radiogenomic-classification"
ALT_ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
DATA_ROOT = DEFAULT_ROOT if os.path.isdir(DEFAULT_ROOT) else ALT_ROOT

TRAIN_LABELS_PATH = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
SUBMISSION_PATH = "submission.csv"



## === cell 2
train_labels = pd.read_csv(TRAIN_LABELS_PATH, dtype={"BraTS21ID": str})

problematic_ids = {"00109", "00123", "00709"}
train_labels = train_labels[
    ~train_labels["BraTS21ID"].isin(problematic_ids)
].reset_index(drop=True)

train_labels["BraTS21ID_int"] = train_labels["BraTS21ID"].astype(int)
global_mean = train_labels["MGMT_value"].mean()
print(f"Global mean MGMT probability (cleaned): {global_mean:.5f}")

train_labels["prefix"] = train_labels["BraTS21ID"].str[:2]
prefix_means = train_labels.groupby("prefix")["MGMT_value"].mean()

train_ids_array = train_labels["BraTS21ID_int"].values
train_vals_array = train_labels["MGMT_value"].values
sorted_idx = np.argsort(train_ids_array)
train_ids_sorted = train_ids_array[sorted_idx]
train_vals_sorted = train_vals_array[sorted_idx]

train_id_to_val = dict(zip(train_ids_array, train_vals_array))

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

dev_train, dev_val = train_test_split(
    train_labels, test_size=0.2, random_state=42, stratify=train_labels["MGMT_value"]
)

global_mean_dev = dev_train["MGMT_value"].mean()
prefix_means_dev = dev_train.groupby("prefix")["MGMT_value"].mean()

dev_ids = dev_train["BraTS21ID_int"].values
dev_vals = dev_train["MGMT_value"].values
dev_sorted_idx = np.argsort(dev_ids)
dev_ids_sorted = dev_ids[dev_sorted_idx]
dev_vals_sorted = dev_vals[dev_sorted_idx]
dev_id_to_val = dict(zip(dev_ids, dev_vals))


def _interp_val(test_id):
    if test_id in dev_id_to_val:
        return dev_id_to_val[test_id]
    idx = np.searchsorted(dev_ids_sorted, test_id)
    if idx == 0:
        return dev_vals_sorted[0]
    if idx == len(dev_ids_sorted):
        return dev_vals_sorted[-1]
    left_id, right_id = dev_ids_sorted[idx - 1], dev_ids_sorted[idx]
    left_val, right_val = dev_vals_sorted[idx - 1], dev_vals_sorted[idx]
    weight = (test_id - left_id) / (right_id - left_id)
    return left_val + weight * (right_val - left_val)


val_prefix_pred = (
    dev_val["prefix"]
    .map(prefix_means_dev)
    .fillna(global_mean_dev)
    .astype(np.float32)
    .values
)
val_nearest = np.array(
    [_interp_val(tid) for tid in dev_val["BraTS21ID_int"]], dtype=np.float32
)

best_auc = -1.0
best_w = 0.5
for w in np.arange(0.0, 1.01, 0.05):
    blended = w * val_prefix_pred + (1 - w) * val_nearest
    auc = roc_auc_score(dev_val["MGMT_value"], blended)
    if auc > best_auc:
        best_auc = auc
        best_w = w
best_w_prefix = best_w
print(
    f"Selected blending weight (prefix): {best_w_prefix:.2f} with internal AUC {best_auc:.4f}"
)



## === cell 3
sample_sub = pd.read_csv(SAMPLE_SUB_PATH, dtype={"BraTS21ID": str})
sample_sub["BraTS21ID_int"] = sample_sub["BraTS21ID"].astype(int)

sample_sub["prefix"] = sample_sub["BraTS21ID"].str[:2]



## === cell 4
prefix_pred = (
    sample_sub["prefix"].map(prefix_means).fillna(global_mean).astype(np.float32).values
)

nearest_vals = []
for test_id in sample_sub["BraTS21ID_int"]:
    if test_id in train_id_to_val:
        interpolated = train_id_to_val[test_id]
    else:
        idx = np.searchsorted(train_ids_sorted, test_id)
        if idx == 0:
            interpolated = train_vals_sorted[0]
        elif idx == len(train_ids_sorted):
            interpolated = train_vals_sorted[-1]
        else:
            left_id, right_id = train_ids_sorted[idx - 1], train_ids_sorted[idx]
            left_val, right_val = train_vals_sorted[idx - 1], train_vals_sorted[idx]
            weight = (test_id - left_id) / (right_id - left_id)
            interpolated = left_val + weight * (right_val - left_val)
    nearest_vals.append(interpolated)
nearest_vals = np.array(nearest_vals, dtype=np.float32)

w_prefix = best_w_prefix
w_nearest = 1.0 - best_w_prefix
predictions = (w_prefix * prefix_pred + w_nearest * nearest_vals).astype(np.float32)



## === cell 5
final_submission = pd.DataFrame(
    {"BraTS21ID": sample_sub["BraTS21ID"], "MGMT_value": predictions}
)



## === cell 6
final_submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission written to {SUBMISSION_PATH} with shape {final_submission.shape}")
