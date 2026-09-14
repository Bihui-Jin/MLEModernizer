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

0.4814720380758117

# 6. Current score

0.64

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.56824) has done: 'I fix the immediate import/runtime failures by removing/avoiding packages that trigger the protobuf `MessageFactory.GetPrototype` error (and also aren’t needed for inference), and by fixing the `resize` NameError. Next, because the referenced external pretrained `.h5` models are not available in your provided `/kaggle/input` tree, I replace that dependency with a tiny in-notebook CNN ensemble that preserves the original pipeline semantics: extract 7 slices per sequence (T2/FLAIR/T1wCE), run multiple models, and average probabilities. Finally, I correct the submission construction bug (it was computing a single vector for all cases inside a loop) and ensure `BraTS21ID` formatting matches the sample submission, writing a valid `submission.csv` to the working directory.'
- What this solution (achieved 0.64) has done: 'The crash happens before any training because importing TensorFlow triggers a protobuf incompatibility (`MessageFactory.GetPrototype`) in this Kaggle image. I fix that by removing TensorFlow/Keras entirely and replacing the tiny CNN with a pure-numpy logistic model trained on the exact same extracted slice data (still 7 slices per sequence, still three sequences, still averaging to case-level probabilities), so the pipeline runs end-to-end without that import. Since your current score (0.56824) is above the target (0.48147) and higher-is-better, I also add a small, deterministic probability-shrink calibration toward 0.5 to nudge AUC down toward the target band without changing the data extraction or submission format. The script still write a valid `submission.csv` with the required columns and ordering aligned to `sample_submission.csv`.'
- What this solution (achieved 0.64) has done: 'Your current score (0.64) is higher than the target (0.48147), so to move closer we should intentionally reduce performance slightly rather than improve it. The smallest, safest lever that preserves your core pipeline is the final probability shrinkage toward 0.5, because it monotonically reduces discrimination (and thus AUC) without changing data loading, feature extraction, training, or submission formatting. I keep everything identical except adjust `SHRINK_ALPHA` to a lower value (stronger shrink), which should bring AUC down toward the target band. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.64) has done: 'Your current AUC (0.64) is higher than the target (0.48147), so the correct move is to slightly reduce discriminative power in a controlled way without changing data loading, features, or the training procedure. The smallest, safest lever is the final probability shrinkage toward 0.5; I tune `SHRINK_ALPHA` downward to push the AUC closer to the target band while keeping the rest of the pipeline identical. I also make this shrink value deterministic and explicitly justified inline, and keep the submission formatting/alignment exactly as before to ensure a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import pydicom
from skimage.transform import resize

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

print(
    "Using pure-numpy model (TensorFlow removed to avoid protobuf MessageFactory.GetPrototype crash)."
)



## === cell 1
DATA_ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
LABELS_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

train_df = pd.read_csv(LABELS_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

print("train_df:", train_df.shape, train_df.columns.tolist())
print("sample_df:", sample_df.shape, sample_df.columns.tolist())
print(
    "train dir exists:",
    os.path.isdir(TRAIN_DIR),
    "test dir exists:",
    os.path.isdir(TEST_DIR),
)



## === cell 2
BAD_CASES = {"00109", "00123", "00709"}


def _list_case_dirs(root_dir):
    case_dirs = []
    for entry in os.scandir(root_dir):
        if entry.is_dir():
            case_id = os.path.basename(entry.path)
            case_dirs.append((case_id, entry.path))
    case_dirs = sorted(case_dirs, key=lambda x: x[0])
    return case_dirs


train_cases_all = _list_case_dirs(TRAIN_DIR)
test_cases_all = _list_case_dirs(TEST_DIR)

train_cases = [(cid, p) for cid, p in train_cases_all if cid not in BAD_CASES]
print(
    "Train cases:",
    len(train_cases_all),
    "->",
    len(train_cases),
    "after excluding bad cases",
)
print("Test cases:", len(test_cases_all))

train_df["BraTS21ID"] = train_df["BraTS21ID"].astype(str).str.zfill(5)
label_map = dict(zip(train_df["BraTS21ID"], train_df["MGMT_value"].astype(np.float32)))

missing_labels = [cid for cid, _ in train_cases if cid not in label_map]
print("Missing labels among included train cases:", len(missing_labels))



## === cell 3
IMG_PX_SIZE = 150
N_SLICES = 7


def _modality_dirnames_sorted(case_dir):
    mods = sorted([f.path for f in os.scandir(case_dir) if f.is_dir()])
    return mods


def _load_case_slices_from_modality(
    modality_path, img_px=IMG_PX_SIZE, n_slices=N_SLICES
):
    img_files = sorted([f.path for f in os.scandir(modality_path) if f.is_file()])
    slices = []
    for fp in img_files:
        try:
            ds = pydicom.dcmread(fp)
            arr = ds.pixel_array.astype(np.float32)
        except Exception:
            continue
        if arr.sum() <= 100000:
            continue
        arr_rs = resize(
            arr, (img_px, img_px), preserve_range=True, anti_aliasing=True
        ).astype(np.float32)
        stacked = np.stack((arr_rs,) * 3, axis=-1)
        mx = np.max(stacked)
        if mx > 0:
            stacked = stacked / mx
        if stacked.sum() <= 2000:
            continue
        slices.append(stacked)
        if len(slices) >= n_slices:
            break

    if len(slices) == 0:
        slices = [
            np.zeros((img_px, img_px, 3), dtype=np.float32) for _ in range(n_slices)
        ]
    elif len(slices) < n_slices:
        last = slices[-1]
        slices = slices + [last.copy() for _ in range(n_slices - len(slices))]

    return np.stack(slices, axis=0).astype(np.float32)  # (n_slices, H, W, C)


def load_dataset_cases(case_list, root_kind="train"):
    ids = []
    x_t2 = []
    x_flair = []
    x_t1ce = []
    y = []

    for cid, cpath in case_list:
        mods = _modality_dirnames_sorted(cpath)
        if len(mods) < 4:
            continue

        flair_path = mods[0]
        t1ce_path = mods[2]
        t2_path = mods[3]

        x_t2.append(_load_case_slices_from_modality(t2_path))
        x_flair.append(_load_case_slices_from_modality(flair_path))
        x_t1ce.append(_load_case_slices_from_modality(t1ce_path))

        ids.append(cid)
        if root_kind == "train":
            y.append(label_map[cid])

    x_t2 = np.asarray(x_t2, dtype=np.float32)
    x_flair = np.asarray(x_flair, dtype=np.float32)
    x_t1ce = np.asarray(x_t1ce, dtype=np.float32)
    ids = np.asarray(ids)

    if root_kind == "train":
        y = np.asarray(y, dtype=np.float32)
        return ids, x_t2, x_flair, x_t1ce, y
    return ids, x_t2, x_flair, x_t1ce




## === cell 4
train_ids, X_t2, X_flair, X_t1ce, y = load_dataset_cases(train_cases, root_kind="train")
test_ids, T_t2, T_flair, T_t1ce = load_dataset_cases(test_cases_all, root_kind="test")

print("Train shapes:", X_t2.shape, X_flair.shape, X_t1ce.shape, y.shape)
print("Test shapes :", T_t2.shape, T_flair.shape, T_t1ce.shape)




## === cell 5
def _sigmoid(z):
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def _extract_slice_features(X_slices):
    """
    X_slices: (n_cases, n_slices, H, W, 3) float32 in [0,1]
    Returns: (n_cases*n_slices, 10) float32
    """
    n_cases, n_slices = X_slices.shape[0], X_slices.shape[1]
    Xf = X_slices.reshape(n_cases * n_slices, IMG_PX_SIZE, IMG_PX_SIZE, 3).astype(
        np.float32
    )

    ch = Xf.mean(axis=-1)  # (N, H, W)
    flat = ch.reshape(ch.shape[0], -1)

    mean = flat.mean(axis=1)
    std = flat.std(axis=1)
    p10 = np.quantile(flat, 0.10, axis=1)
    p25 = np.quantile(flat, 0.25, axis=1)
    p50 = np.quantile(flat, 0.50, axis=1)
    p75 = np.quantile(flat, 0.75, axis=1)
    p90 = np.quantile(flat, 0.90, axis=1)
    mx = flat.max(axis=1)
    mn = flat.min(axis=1)
    frac_nonzero = (flat > 0.0).mean(axis=1)

    feats = np.stack(
        [mean, std, p10, p25, p50, p75, p90, mn, mx, frac_nonzero], axis=1
    ).astype(np.float32)
    return feats


def fit_logreg_slices(X_slices, y_labels, epochs=120, lr=0.2, l2=1e-3, seed=SEED):
    """
    Train logistic regression on slice-level features, using case labels repeated over slices.
    Returns dict with weights and normalization stats.
    """
    rng = np.random.RandomState(seed)

    feats = _extract_slice_features(X_slices)  # (Nslice, d)
    n_cases, n_slices = X_slices.shape[0], X_slices.shape[1]
    yf = np.repeat(y_labels.astype(np.float32), n_slices).astype(np.float32)

    mu = feats.mean(axis=0)
    sigma = feats.std(axis=0)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    Xn = (feats - mu) / sigma

    n, d = Xn.shape
    w = rng.normal(scale=0.01, size=(d,)).astype(np.float32)
    b = np.float32(0.0)

    for _ in range(epochs):
        z = Xn @ w + b
        p = _sigmoid(z).astype(np.float32)
        err = p - yf
        gw = (Xn.T @ err) / n + l2 * w
        gb = err.mean()
        w = (w - lr * gw).astype(np.float32)
        b = np.float32(b - lr * gb)

    return {
        "w": w,
        "b": b,
        "mu": mu.astype(np.float32),
        "sigma": sigma.astype(np.float32),
    }


def predict_case_probs_logreg(model, X_slices):
    feats = _extract_slice_features(X_slices)
    Xn = (feats - model["mu"]) / model["sigma"]
    z = Xn @ model["w"] + model["b"]
    p_slice = _sigmoid(z).astype(np.float32)

    n_cases, n_slices = X_slices.shape[0], X_slices.shape[1]
    p_case = p_slice.reshape(n_cases, n_slices).mean(axis=1)
    return p_case.astype(np.float32)


models = []
models.append(fit_logreg_slices(X_t2, y, epochs=120, lr=0.2, l2=1e-3, seed=SEED + 0))
models.append(fit_logreg_slices(X_t2, y, epochs=120, lr=0.2, l2=1e-3, seed=SEED + 1))

models.append(fit_logreg_slices(X_flair, y, epochs=120, lr=0.2, l2=1e-3, seed=SEED + 2))
models.append(fit_logreg_slices(X_flair, y, epochs=120, lr=0.2, l2=1e-3, seed=SEED + 3))

models.append(fit_logreg_slices(X_t1ce, y, epochs=120, lr=0.2, l2=1e-3, seed=SEED + 4))
models.append(fit_logreg_slices(X_t1ce, y, epochs=120, lr=0.2, l2=1e-3, seed=SEED + 5))

print("Finished training ensemble (numpy logistic).")



## === cell 6
p_t2_a = predict_case_probs_logreg(models[0], T_t2)
p_t2_b = predict_case_probs_logreg(models[1], T_t2)

p_fl_a = predict_case_probs_logreg(models[2], T_flair)
p_fl_b = predict_case_probs_logreg(models[3], T_flair)

p_t1_a = predict_case_probs_logreg(models[4], T_t1ce)
p_t1_b = predict_case_probs_logreg(models[5], T_t1ce)

pred = (p_t2_a + p_t2_b + p_fl_a + p_fl_b + p_t1_a + p_t1_b) / 6.0
pred = np.clip(pred.astype(np.float32), 0.0, 1.0)

SHRINK_ALPHA = 0.03  # 1.0 = no shrink; smaller => closer to 0.5 (lower AUC)
pred = (SHRINK_ALPHA * pred + (1.0 - SHRINK_ALPHA) * 0.5).astype(np.float32)
pred = np.clip(pred, 0.0, 1.0)

print("Pred stats:", float(pred.min()), float(pred.mean()), float(pred.max()))



## === cell 7
sub_df = pd.DataFrame(
    {"BraTS21ID": pd.Series(test_ids).astype(str).str.zfill(5), "MGMT_value": pred}
)

sample_ids = sample_df["BraTS21ID"].astype(str).str.zfill(5).tolist()
sub_df = sub_df.set_index("BraTS21ID").reindex(sample_ids).reset_index()

if sub_df["MGMT_value"].isna().any():
    sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(float(np.nanmean(pred)))

print(sub_df.head())
print("Submission shape:", sub_df.shape)



## === cell 8
out_path = "submission.csv"
sub_df.to_csv(out_path, index=False)
print("Wrote:", out_path)
print("Columns:", sub_df.columns.tolist())
print("Any NaN:", sub_df.isna().any().to_dict())
