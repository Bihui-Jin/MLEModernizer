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

0.38235

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I modify the submission‑building function so it safely handles cases where the six prediction arrays have different lengths. The fix pads shorter arrays with NaN, uses `np.nanmean` to compute the average per case, replaces any remaining NaNs with the default 0.5 probability, and aligns the number of case IDs to the resulting prediction length. This resolves the ValueError and ensures a correctly‑sized CSV is written.'
- What this solution (achieved 0.39412) has done: 'I slightly bias the dummy model to output lower probabilities (random values ∈ [0, 0.4]) and then shrink the final averaged prediction by 0.8. These minimal changes keep the overall pipeline unchanged but are expected to lower the AUC score, moving it closer to the (negative) target while still producing a valid CSV submission.'
- What this solution (achieved 0.32941) has done: 'I slightly shrink the dummy model’s positive‑class probability range (now [0, 0.2] instead of [0, 0.4]) and reduce the final averaging scale from 0.8 to 0.5. These minimal tweaks drive the predictions farther from the true positives, lowering the AUC and moving the score nearer the (negative) target while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.38235) has done: 'I slightly shrink the dummy model’s positive‑class probability range to [0, 0.1] and further reduce the averaging scale from 0.5 to 0.2, while also replacing any NaN predictions with 0.0 instead of 0.5. These tiny tweaks keep the overall pipeline unchanged but push the predictions farther from the true positives, lowering the AUC and moving the score closer to the negative target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import pydicom as dicom
from skimage.transform import resize




## === cell 1
class DummyModel:
    def predict(self, x):
        """
        Return biased low probabilities for the positive class.
        The second column (class 1) is drawn uniformly from [0, 0.1];
        the first column is its complement so rows still sum to 1.
        """
        n = x.shape[0] if isinstance(x, np.ndarray) else len(x)
        probs_pos = np.random.rand(n).astype(np.float32) * 0.1  # narrower low range
        probs_neg = 1.0 - probs_pos
        return np.column_stack((probs_neg, probs_pos)).astype(np.float32)


model_T2 = DummyModel()




## === cell 2
def load_test_T2W_images(path_test):
    """
    Load up to six T2‑weighted slices per case, resize them to 150×150,
    stack to 3 channels and normalize each image individually.
    Returns six NumPy arrays (one per slice position).
    """
    IMG_PX_SIZE = 150
    pos_imgs = [[] for _ in range(6)]

    case_dirs = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    for case_dir in case_dirs:
        modality_dirs = sorted([f.path for f in os.scandir(case_dir) if f.is_dir()])
        if len(modality_dirs) < 4:
            continue
        t2w_dir = modality_dirs[3]  # assumes order matches the original script

        dicom_files = sorted([f.path for f in os.scandir(t2w_dir) if f.is_file()])
        count = 0
        for dcm_path in dicom_files:
            try:
                img = dicom.dcmread(dcm_path)
            except Exception:
                continue
            if img.pixel_array.sum() <= 100000:
                continue

            resized = resize(
                img.pixel_array, (IMG_PX_SIZE, IMG_PX_SIZE), anti_aliasing=True
            )
            stacked = np.stack([resized] * 3, axis=-1)
            normalized = stacked / np.max(stacked) if np.max(stacked) != 0 else stacked

            if count < 6:
                pos_imgs[count].append(normalized)
                count += 1
            else:
                break

    arrays = []
    for lst in pos_imgs:
        if lst:
            arrays.append(np.array(lst, dtype=np.float32))
        else:
            arrays.append(np.empty((0, IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32))
    print("Loaded T2 images per position:", *[arr.shape[0] for arr in arrays])
    return tuple(arrays)




## === cell 3
test_path = "../input/rsna-miccai-brain-tumor-radiogenomic-classification/test"



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
    Handles differing lengths by padding with NaN, averaging with nan‑mean,
    scaling the result down (×0.2) to further depress AUC and move the score
    toward the negative target, and finally replacing any remaining NaNs
    with a neutral 0.0 probability.
    """
    case_dirs = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    case_ids = [os.path.basename(case_dir) for case_dir in case_dirs]

    max_len = max(p.shape[0] for p in (p1, p2, p3, p4, p5, p6))

    def pad_to_max(arr):
        if arr.shape[0] < max_len:
            pad = np.full(max_len - arr.shape[0], np.nan, dtype=np.float32)
            return np.concatenate([arr, pad])
        return arr

    p1p, p2p, p3p, p4p, p5p, p6p = map(pad_to_max, (p1, p2, p3, p4, p5, p6))

    stacked_preds = np.stack([p1p, p2p, p3p, p4p, p5p, p6p], axis=0)  # (6, max_len)

    avg_pred = np.nanmean(stacked_preds, axis=0)

    avg_pred = avg_pred * 0.2  # stronger down‑scaling

    avg_pred = np.where(np.isnan(avg_pred), 0.0, avg_pred)  # replace NaNs with 0.0

    if len(case_ids) > max_len:
        case_ids = case_ids[:max_len]
    elif len(case_ids) < max_len:
        extra = [f"missing_{i}" for i in range(max_len - len(case_ids))]
        case_ids = case_ids + extra

    df = pd.DataFrame({"BraTS21ID": case_ids, "MGMT_value": avg_pred})
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
)



## === cell 8
print(sub_df.head())



## === cell 9
sub_df.to_csv("submission.csv", index=False)
