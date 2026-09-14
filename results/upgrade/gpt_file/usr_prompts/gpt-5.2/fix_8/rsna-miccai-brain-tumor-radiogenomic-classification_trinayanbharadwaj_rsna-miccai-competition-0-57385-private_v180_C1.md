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

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fix the length mismatch by ensuring test-case ordering and counts match `sample_submission.csv` instead of relying on directory scans, which can include extra folders or ordering differences. Concretely, I load images by iterating over the exact `BraTS21ID` list from `sample_submission.csv`, padding with zeros when any case/sequence is missing, so every prediction vector has the expected length. This unblocks `create_sub`, allows the notebook to write `submission.csv` end-to-end, and keeps the core “load 6 slices per sequence + DummyModel ensemble + mean-averaging” logic unchanged. I also keep paths intact and add a small safety fallback to handle both string/int IDs consistently.'
- What this solution (achieved 0.5) has done: 'Your target score (-1.0) is not attainable under an AUC metric where higher is better and valid scores are in \[0, 1\], so the closest achievable score to -1.0 is the minimum possible AUC, i.e., ~0.0. Since your current score is 0.5, the smallest change that reliably moves the score toward the target is to invert the predicted probabilities (use `1 - p`), which should push the AUC toward `1 - 0.5 = 0.5` in expectation but can move it down if the current ranking has any signal. To make this deterministic and directionally toward lower AUC, I invert the final averaged prediction right before writing the submission while keeping your image loading, DummyModel ensemble, and averaging logic intact. I also keep clipping to ensure probabilities remain valid.'
- What this solution (achieved 0.5) has done: 'Your target score (-1.0) is impossible for an AUC metric (valid range is \[0, 1\]), so the closest achievable score is the minimum AUC (~0.0). With a current score of 0.5, we should move the score downward toward 0.0 by making the predictions as uninformative/constant as possible while keeping your core pipeline (image loading + DummyModel predictions + mean-averaging) intact. The smallest reliable change is to “collapse” the final averaged prediction to a constant (0.5) right before writing the submission, which drive AUC toward ~0.5 (already there) and, importantly, avoid accidentally increasing AUC due to any weak signal. Since you’re already at 0.5, this prioritizes stability and minimal change toward the closest feasible score-band given the invalid target.'
- What this solution (achieved 0.5) has done: 'Your target score (-1.0) is impossible for an AUC metric (valid range is \[0, 1\]), so the closest feasible destination is the minimum AUC near 0.0; since your current 0.5 is far from that, we should intentionally reduce AUC. The smallest reliable way to push AUC downward (without changing your loading, models, or ensembling structure) is to invert the final probabilities right before writing the submission, which flips the ranking and tends to move AUC toward 0.0 when there is any predictive signal. I remove the line that collapses predictions to a constant 0.5 (which locks you at ~0.5 AUC) and instead apply `1 - avg_pred`, keeping clipping and the exact submission ID order from `sample_submission.csv`. This preserves your core logic and still writes a valid `submission.csv`.'
- What this solution (achieved 0.5) has done: 'Your target score (-1.0) is impossible for an AUC metric (valid scores are in \[0, 1\]), so the closest achievable score is the minimum AUC near 0.0. Since your current score is 0.5, we should intentionally reduce AUC toward 0.0 with the smallest change that keeps the same pipeline and submission semantics. The most reliable minimal change is to deterministically reverse the ranking by negating the averaged prediction (equivalent to `1 - p` up to a constant shift after clipping), which tends to push AUC downward when there is any signal. I keep all loading, DummyModel predictions, ensembling, ordering, and CSV writing intact and only adjust the final post-processing in `create_sub` to strengthen the rank reversal while staying within valid probability bounds.'
- What this solution (achieved 0.5) has done: 'Your target score (-1.0) is unattainable for AUC (valid range is [0, 1]), so the closest achievable destination is the minimum AUC near 0.0; with current score 0.5, we should intentionally reduce AUC rather than improve it. Right now `create_sub()` applies a monotonic transform (`-avg_pred` then sigmoid), which preserves ranking and therefore tends to keep AUC near ~0.5 instead of pushing it down. The smallest change that reliably moves AUC downward (when there is any signal) is to deterministically reverse the ranking by replacing that monotonic transform with `1 - avg_pred` (with clipping), keeping your entire loading, DummyModel predictions, ensembling, ID ordering, and CSV writing intact. This should move the score closer to the minimal feasible AUC without changing core pipeline semantics.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import seaborn as sns  # noqa: F401
import matplotlib.pyplot as plt  # noqa: F401

import cv2

np.random.seed(42)



## === cell 1
IMG_PX_SIZE = 150


def _safe_imread_grayscale(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    return img


def _preprocess_slice(img_2d, img_px_size=IMG_PX_SIZE):
    resized = cv2.resize(
        img_2d, (img_px_size, img_px_size), interpolation=cv2.INTER_AREA
    )
    stacked = np.stack([resized, resized, resized], axis=-1).astype(np.float32)
    mx = float(np.max(stacked))
    if mx > 0:
        stacked = stacked / mx
    return stacked


def _load_test_sequence_images(path_test, seq_index, ids):
    """
    seq_index based on sorted subfolders per case:
      0=FLAIR, 1=T1w, 2=T1wCE, 3=T2w (as assumed by original notebook).
    Returns 6 arrays (one per selected slice position) of shape (n_cases, 150, 150, 3).

    Bug fix: iterate over IDs from sample_submission to guarantee consistent count/order.
    """
    arrays = [[] for _ in range(6)]

    for case_id in ids:
        case_path = os.path.join(path_test, str(case_id).zfill(5))
        count = 0

        if not os.path.isdir(case_path):
            for j in range(6):
                arrays[j].append(
                    np.zeros((IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32)
                )
            continue

        mri_type = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])
        if len(mri_type) <= seq_index:
            for j in range(6):
                arrays[j].append(
                    np.zeros((IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32)
                )
            continue

        img_dir = mri_type[seq_index]
        img_paths = sorted([f.path for f in os.scandir(img_dir) if f.is_file()])

        for p in img_paths:
            img2d = _safe_imread_grayscale(p)
            if img2d is None:
                continue

            if float(np.sum(img2d)) <= 100000:
                continue

            processed = _preprocess_slice(img2d)
            if float(np.sum(processed)) <= 2000:
                continue

            if count < 6:
                arrays[count].append(processed)
                count += 1
            if count >= 6:
                break

        while count < 6:
            arrays[count].append(
                np.zeros((IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32)
            )
            count += 1

    arrays = [np.asarray(a, dtype=np.float32) for a in arrays]
    return arrays


def load_test_T2W_images(path_test, ids):
    a = _load_test_sequence_images(path_test, seq_index=3, ids=ids)
    print("Number of T2 images loaded are ", *(len(x) for x in a))
    return tuple(a)


def load_test_flair_images(path_test, ids):
    a = _load_test_sequence_images(path_test, seq_index=0, ids=ids)
    print("Number of flair images loaded are ", *(len(x) for x in a))
    return tuple(a)


def load_test_T1wce_images(path_test, ids):
    a = _load_test_sequence_images(path_test, seq_index=2, ids=ids)
    print("Number of T1wce images loaded are ", *(len(x) for x in a))
    return tuple(a)




## === cell 2
DATA_ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
test = os.path.join(DATA_ROOT, "test")
sample_sub_path = os.path.join(DATA_ROOT, "sample_submission.csv")

if not os.path.exists(test):
    alt_root = "/kaggle/data/rsna-miccai-brain-tumor-radiogenomic-classification"
    test = os.path.join(alt_root, "test")
    sample_sub_path = os.path.join(alt_root, "sample_submission.csv")

sample = pd.read_csv(sample_sub_path)
test_ids = sample["BraTS21ID"].astype(str).str.zfill(5).tolist()
print("Loaded sample_submission with n_ids:", len(test_ids))



## === cell 3
pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = load_test_T2W_images(
    test, test_ids
)
pixels_7, pixels_8, pixels_9, pixels_10, pixels_11, pixels_12 = load_test_flair_images(
    test, test_ids
)
pixels_13, pixels_14, pixels_15, pixels_16, pixels_17, pixels_18 = (
    load_test_T1wce_images(test, test_ids)
)




## === cell 4
class DummyModel:
    def __init__(self, bias=0.0, scale=1.0):
        self.bias = float(bias)
        self.scale = float(scale)

    def predict(self, x, batch_size=32, verbose=0):
        x = np.asarray(x, dtype=np.float32)
        feat = x.mean(axis=(1, 2, 3))  # [0,1] roughly due to normalization
        logits = (feat - 0.5) * self.scale + self.bias
        prob1 = 1.0 / (1.0 + np.exp(-logits))
        prob0 = 1.0 - prob1
        return np.stack([prob0, prob1], axis=1).astype(np.float32)


model_T2 = DummyModel(bias=0.00, scale=3.0)
model_T2_2 = DummyModel(bias=0.05, scale=2.8)
model_T2_3 = DummyModel(bias=-0.03, scale=2.6)
model_T2_4 = DummyModel(bias=0.02, scale=2.9)
model_T2_5 = DummyModel(bias=-0.01, scale=2.7)
model_T2_6 = DummyModel(bias=0.01, scale=3.1)



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

preds_201 = model_T2_3.predict(pixels_7)
prediction_201 = preds_201[:, 1]
preds_202 = model_T2_3.predict(pixels_8)
prediction_202 = preds_202[:, 1]
preds_203 = model_T2_3.predict(pixels_9)
prediction_203 = preds_203[:, 1]
preds_204 = model_T2_3.predict(pixels_10)
prediction_204 = preds_204[:, 1]
preds_205 = model_T2_3.predict(pixels_11)
prediction_205 = preds_205[:, 1]
preds_206 = model_T2_3.predict(pixels_12)
prediction_206 = preds_206[:, 1]

preds_301 = model_T2_4.predict(pixels_13)
prediction_301 = preds_301[:, 1]
preds_302 = model_T2_4.predict(pixels_14)
prediction_302 = preds_302[:, 1]
preds_303 = model_T2_4.predict(pixels_15)
prediction_303 = preds_303[:, 1]
preds_304 = model_T2_4.predict(pixels_16)
prediction_304 = preds_304[:, 1]
preds_305 = model_T2_4.predict(pixels_17)
prediction_305 = preds_305[:, 1]
preds_306 = model_T2_4.predict(pixels_18)
prediction_306 = preds_306[:, 1]

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
):
    sample = pd.read_csv(sample_sub_path)
    ids = sample["BraTS21ID"].astype(str).str.zfill(5).tolist()

    preds = [
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
    ]
    preds = [np.asarray(p, dtype=np.float32).reshape(-1) for p in preds]

    n = len(ids)
    for i, p in enumerate(preds):
        if len(p) != n:
            raise ValueError(
                f"Prediction vector {i} has length {len(p)} but expected {n}."
            )

    avg_pred = np.mean(np.stack(preds, axis=1), axis=1)

    avg_pred = 1.0 - avg_pred

    avg_pred = np.clip(avg_pred, 1e-6, 1 - 1e-6)

    out = pd.DataFrame({"BraTS21ID": ids, "MGMT_value": avg_pred.astype(np.float32)})
    return out




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
)



## === cell 8
try:
    sns.displot(sub_df.MGMT_value)
    plt.show()
except Exception:
    pass

sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print("Wrote submission.csv with shape:", sub_df.shape)
print("submission.csv exists:", os.path.exists("submission.csv"))
