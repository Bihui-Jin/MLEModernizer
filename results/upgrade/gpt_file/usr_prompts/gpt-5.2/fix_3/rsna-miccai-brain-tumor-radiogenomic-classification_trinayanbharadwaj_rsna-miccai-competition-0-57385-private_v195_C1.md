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

0.48118

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I remove/guard the imports that trigger the protobuf `MessageFactory.GetPrototype` crash, since they aren’t needed for generating a submission here. Because the referenced pretrained `.h5` model file is not present in your environment, I replace the missing-model inference step with a deterministic baseline that still produces valid probabilities and a correctly formatted `submission.csv`. I also fix runtime issues in the image-loading function (missing `resize`, invalid list/ndarray math) and fix the submission creation logic so predictions align 1-to-1 with `BraTS21ID` rows. These changes are the smallest set that makes the notebook run end-to-end and write a valid CSV.'
- What this solution (achieved 0.48118) has done: 'Your current 0.5 score comes from predicting a constant probability for every test case; to move the AUC upward toward a more competitive level with minimal logic change, we keep the same “simple baseline” approach but replace the constant with a lightweight, legitimate imaging-derived score per subject. Specifically, we compute a per-case intensity statistic from a few central DICOM slices (using only `pydicom`/`numpy`) and map it through a logistic transform to produce calibrated probabilities, then align predictions strictly to `sample_submission.csv` order. This preserves the overall workflow (read labels → make predictions → write submission) while adding just enough signal to improve ranking-based AUC. We also keep robust fallbacks so the code always produces a valid `submission.csv` within the time limit.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
DATA_ROOT = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_LABELS_PATH = os.path.join(DATA_ROOT, "train_labels.csv")
TEST_PATH = os.path.join(DATA_ROOT, "test")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

train_labels = pd.read_csv(TRAIN_LABELS_PATH)
pos_rate = float(train_labels["MGMT_value"].mean())
pos_rate




## === cell 2
def load_test_flair_images(path_test):
    """
    Kept for compatibility with the original workflow, but not required for the baseline submission.
    Fixed issues:
      - 'resize' was not defined (skimage not guaranteed / removed heavy deps)
      - list / scalar normalization was invalid
      - safer handling when no valid slices found
    """
    try:
        import pydicom as dicom
    except Exception:
        dicom = None

    try:
        import cv2
    except Exception:
        cv2 = None

    array_1, array_2, array_3, array_4, array_5, array_6 = [], [], [], [], [], []
    IMG_PX_SIZE = 150

    path_cases = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    for case_path in path_cases:
        count = 0
        mri_types = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])
        if not mri_types:
            continue

        flair_path = mri_types[0]
        img_paths = sorted([f.path for f in os.scandir(flair_path) if f.is_file()])

        for img_p in img_paths:
            if dicom is None:
                break
            try:
                ds = dicom.dcmread(img_p)
                px = ds.pixel_array.astype(np.float32)
            except Exception:
                continue

            if px.sum() <= 100000:
                continue

            if cv2 is not None:
                resized = cv2.resize(
                    px, (IMG_PX_SIZE, IMG_PX_SIZE), interpolation=cv2.INTER_AREA
                )
            else:
                h, w = px.shape
                out = np.zeros((IMG_PX_SIZE, IMG_PX_SIZE), dtype=np.float32)
                hh = min(h, IMG_PX_SIZE)
                ww = min(w, IMG_PX_SIZE)
                out[:hh, :ww] = px[:hh, :ww]
                resized = out

            stacked = np.stack([resized, resized, resized], axis=-1)
            mx = float(np.max(stacked))
            if mx > 0:
                stacked = stacked / mx

            if stacked.sum() <= 2000:
                continue

            if count == 0:
                array_1.append(stacked)
            elif count == 1:
                array_2.append(stacked)
            elif count == 2:
                array_3.append(stacked)
            elif count == 3:
                array_4.append(stacked)
            elif count == 4:
                array_5.append(stacked)
            elif count == 5:
                array_6.append(stacked)
            count += 1
            if count >= 6:
                break

    def _to_array(lst):
        if len(lst) == 0:
            return np.zeros((0, IMG_PX_SIZE, IMG_PX_SIZE, 3), dtype=np.float32)
        return np.asarray(lst, dtype=np.float32)

    array_1 = _to_array(array_1)
    array_2 = _to_array(array_2)
    array_3 = _to_array(array_3)
    array_4 = _to_array(array_4)
    array_5 = _to_array(array_5)
    array_6 = _to_array(array_6)

    print(
        "Number of flair images loaded are ",
        len(array_1),
        ",",
        len(array_2),
        ",",
        len(array_3),
        ",",
        len(array_4),
        ",",
        len(array_5),
        ",",
        len(array_6),
    )
    return array_1, array_2, array_3, array_4, array_5, array_6




## === cell 3
test = TEST_PATH
os.path.isdir(test), test



## === cell 4
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
sample_sub["MGMT_value"] = pos_rate
sample_sub.head()




## === cell 5
def _safe_sigmoid(x):
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def _read_dicom_pixel_array(dicom_path):
    try:
        import pydicom
    except Exception:
        return None
    try:
        ds = pydicom.dcmread(dicom_path)
        arr = ds.pixel_array.astype(np.float32)
        return arr
    except Exception:
        return None


def _subject_score_from_t2w(subject_dir, n_slices=5):
    t2w_dir = os.path.join(subject_dir, "T2w")
    if not os.path.isdir(t2w_dir):
        return np.nan

    files = sorted(
        [
            os.path.join(t2w_dir, f)
            for f in os.listdir(t2w_dir)
            if f.lower().endswith(".dcm")
        ]
    )
    if len(files) == 0:
        return np.nan

    mid = len(files) // 2
    half = n_slices // 2
    pick = files[max(0, mid - half) : min(len(files), mid + half + 1)]
    if len(pick) == 0:
        return np.nan

    vals = []
    for fp in pick:
        arr = _read_dicom_pixel_array(fp)
        if arr is None:
            continue
        a = arr.ravel()
        if a.size == 0:
            continue
        p95 = float(np.percentile(a, 95))
        med = float(np.median(a))
        vals.append(p95 - med)

    if len(vals) == 0:
        return np.nan
    return float(np.mean(vals))


def predict_from_images(sample_df, test_root, fallback_constant):
    ids = sample_df["BraTS21ID"].astype(str).str.zfill(5).tolist()

    raw_scores = np.empty(len(ids), dtype=np.float32)
    for i, sid in enumerate(ids):
        subject_dir = os.path.join(test_root, sid)
        raw_scores[i] = _subject_score_from_t2w(subject_dir, n_slices=5)

    if np.all(np.isnan(raw_scores)):
        return np.full(len(ids), float(fallback_constant), dtype=np.float32)

    med = float(np.nanmedian(raw_scores))
    raw_scores = np.where(np.isnan(raw_scores), med, raw_scores).astype(np.float32)

    mu = float(raw_scores.mean())
    sigma = float(raw_scores.std() + 1e-6)
    z = (raw_scores - mu) / sigma

    p0 = _safe_sigmoid(0.7 * z)

    eps = 1e-6
    logit = np.log(np.clip(p0, eps, 1 - eps) / np.clip(1 - p0, eps, 1 - eps))
    target_logit_mean = np.log(
        np.clip(pos_rate, eps, 1 - eps) / np.clip(1 - pos_rate, eps, 1 - eps)
    )
    shift = target_logit_mean - float(np.mean(logit))
    p = _safe_sigmoid(logit + shift).astype(np.float32)

    return np.clip(p, 0.0, 1.0)




## === cell 6
def create_sub_from_sample(sample_df, preds):
    """
    Create submission aligned to sample_submission order.
    Updated to accept vector predictions (still aligned 1-to-1 with BraTS21ID).
    """
    sub = sample_df.copy()
    sub["BraTS21ID"] = sub["BraTS21ID"].astype(str).str.zfill(5)
    sub["MGMT_value"] = np.asarray(preds, dtype=np.float32)
    sub["MGMT_value"] = sub["MGMT_value"].astype(float).clip(0.0, 1.0)
    return sub




## === cell 7
preds = predict_from_images(sample_sub, TEST_PATH, fallback_constant=pos_rate)
sub_df = create_sub_from_sample(sample_sub, preds)
sub_df.shape, sub_df.head()



## === cell 8
assert list(sub_df.columns) == ["BraTS21ID", "MGMT_value"]
assert sub_df["MGMT_value"].between(0, 1).all()
assert sub_df["BraTS21ID"].isna().sum() == 0
assert sub_df["MGMT_value"].isna().sum() == 0
assert len(sub_df) == len(sample_sub)



## === cell 9
sub_df.to_csv("submission.csv", index=False)
print(
    "Wrote submission.csv with",
    len(sub_df),
    "rows; mean pred =",
    float(sub_df["MGMT_value"].mean()),
)
