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

0.5412629610742818

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

try:
    import pydicom
except Exception:
    pydicom = None  # fallback will use file‑count based features

try:
    import cv2  # noqa: F401
except Exception:
    cv2 = None
import concurrent.futures  # added for parallel execution

BASE_INPUT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
train_dir = os.path.join(BASE_INPUT, "train")
test_dir = os.path.join(BASE_INPUT, "test")
labels_path = os.path.join(BASE_INPUT, "train_labels.csv")




## === cell 1
def average(l):
    return sum(l) / len(l)


def layers(l, n):
    for i in range(0, len(l), n):
        yield l[i : i + n]


def adjuster(file):
    m5 = [17]
    m4 = [18, 27, 28]
    m3 = [19, 20, 21, 25, 26, 33, 34, 35, 41, 42, 49]
    m2 = [
        22,
        23,
        24,
        36,
        37,
        38,
        39,
        40,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        65,
        66,
        67,
        68,
        69,
        70,
        81,
        82,
        83,
        84,
        97,
        98,
    ]
    if len(file) in m5:
        n = 5
    elif len(file) in m4:
        n = 4
    elif len(file) in m3:
        n = 3
    elif len(file) in m2:
        n = 2
    else:
        n = 1
    new_file = []
    for i in range(len(file)):
        for _ in range(n):
            new_file.append(file[i])
    return new_file


def decider(length, layer_number):
    return int(np.ceil(length / layer_number))


def adjuster2(layers, layer_number):
    if len(layers) == layer_number - 1:
        layers.append(layers[-1])




## === cell 2
df = pd.read_csv(labels_path)
bad_ids = {"00109", "00123", "00709"}
df = df[~df["BraTS21ID"].astype(str).isin(bad_ids)]

df["MGMT_value"] = pd.to_numeric(df["MGMT_value"], errors="coerce")
baseline_prob = df["MGMT_value"].mean()
print(f"Baseline probability (mean MGMT value): {baseline_prob:.5f}")




## === cell 3
def compute_modality_mean(subject_path, modality):
    """
    Compute a numeric proxy for a modality.
    If pydicom is available, use the mean pixel intensity.
    Otherwise, fall back to the number of DICOM files (slice count).
    Returns np.nan if the modality folder is missing or contains no readable data.
    """
    mod_dir = os.path.join(subject_path, modality)
    if not os.path.isdir(mod_dir):
        return np.nan
    files = sorted([f for f in os.listdir(mod_dir) if f.lower().endswith(".dcm")])
    if not files:
        return np.nan
    if pydicom is not None:
        pixel_sums = 0.0
        pixel_counts = 0
        for fname in files:
            try:
                ds = pydicom.dcmread(
                    os.path.join(mod_dir, fname), stop_before_pixels=False
                )
                arr = ds.pixel_array.astype(np.float32)
                pixel_sums += arr.sum()
                pixel_counts += arr.size
            except Exception:
                continue
        if pixel_counts == 0:
            return np.nan
        return pixel_sums / pixel_counts
    else:
        return float(len(files))


def compute_features(subject_path):
    flair_mean = compute_modality_mean(subject_path, "FLAIR")
    t1ce_mean = compute_modality_mean(subject_path, "T1wCE")
    return (flair_mean, t1ce_mean)


subject_ids = [str(row["BraTS21ID"]).zfill(5) for _, row in df.iterrows()]
subject_paths = [os.path.join(train_dir, sid) for sid in subject_ids]

with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    train_feat_tuples = list(executor.map(compute_features, subject_paths))

train_flair_feats = [t[0] for t in train_feat_tuples]
train_t1ce_feats = [t[1] for t in train_feat_tuples]

flair_series = pd.Series(train_flair_feats)
t1ce_series = pd.Series(train_t1ce_feats)

flair_mean = flair_series.mean()
flair_std = flair_series.std(ddof=0) if flair_series.std(ddof=0) != 0 else 1.0
t1ce_mean = t1ce_series.mean()
t1ce_std = t1ce_series.std(ddof=0) if t1ce_series.std(ddof=0) != 0 else 1.0

print(f"Training FLAIR mean={flair_mean:.2f}, std={flair_std:.2f}")
print(f"Training T1wCE mean={t1ce_mean:.2f}, std={t1ce_std:.2f}")




## === cell 4
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


z_flair_train = np.where(
    np.isnan(train_flair_feats),
    0.0,
    (np.array(train_flair_feats) - flair_mean) / flair_std,
)
z_t1ce_train = np.where(
    np.isnan(train_t1ce_feats),
    0.0,
    (np.array(train_t1ce_feats) - t1ce_mean) / t1ce_std,
)

X_train = np.column_stack([z_flair_train, z_t1ce_train, np.ones_like(z_flair_train)])
y_train = df["MGMT_value"].values.astype(np.float32)

lam = 1e-5
A = X_train.T @ X_train + lam * np.eye(3)
b = X_train.T @ y_train
coef = np.linalg.solve(A, b)  # coef = [w_flair, w_t1ce, bias]

print(
    f"Fitted coefficients: w_flair={coef[0]:.4f}, w_t1ce={coef[1]:.4f}, bias={coef[2]:.4f}"
)


def corr_with_scale(scale):
    logits = X_train @ coef
    probs = sigmoid(logits * scale)
    if np.std(probs) == 0:
        return -1.0
    return np.corrcoef(probs, y_train)[0, 1]


candidate_scales = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
best_scale = 1.0
best_corr = -np.inf
for s in candidate_scales:
    c = corr_with_scale(s)
    if c > best_corr:
        best_corr = c
        best_scale = s
print(f"Selected logit scaling factor: {best_scale:.2f} (correlation={best_corr:.4f})")



## === cell 5
test_ids = sorted(
    [
        name
        for name in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, name))
    ]
)
print(f"Number of test subjects: {len(test_ids)}")



## === cell 6
predictions = np.empty(len(test_ids), dtype=np.float32)

test_paths = [os.path.join(test_dir, sid) for sid in test_ids]

with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    test_feat_tuples = list(executor.map(compute_features, test_paths))

for i, (flair_feat, t1ce_feat) in enumerate(test_feat_tuples):
    if np.isnan(flair_feat) and np.isnan(t1ce_feat):
        prob = baseline_prob
    else:
        if np.isnan(flair_feat):
            flair_feat = flair_mean
        if np.isnan(t1ce_feat):
            t1ce_feat = t1ce_mean
        z_flair = (flair_feat - flair_mean) / flair_std
        z_t1ce = (t1ce_feat - t1ce_mean) / t1ce_std
        logit = coef[0] * z_flair + coef[1] * z_t1ce + coef[2]
        prob = sigmoid(logit * best_scale)  # apply calibrated scaling
        prob = np.clip(prob, 0.0, 1.0)
    predictions[i] = prob



## === cell 7
submission = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": predictions})
submission_path = "/kaggle/working/submission.csv"
os.makedirs(os.path.dirname(submission_path), exist_ok=True)
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")
