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

0.61294

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.44824) has done: 'I remove/avoid imports that trigger the protobuf `MessageFactory.GetPrototype` crash (not needed for this pipeline) and make the DICOM loading resilient to missing dependencies by implementing a small local `resize()` fallback so the image loader can run. Since the referenced pretrained `.h5` models are not present in your environment, I keep the same “3-model ensemble over 6 slices” core logic but instantiate three small Keras CNNs with identical I/O semantics and train them briefly on the available training set to produce real probabilities. I also fix the submission-building logic (it currently overwrites `prediction` inside the loop and mismatches per-case vs per-slice arrays), ensuring exactly one probability per test `BraTS21ID` and writing a valid `submission.csv`. Finally, I exclude the known-bad training cases `[00109, 00123, 00709]` as recommended to improve stability and score.'
- What this solution (achieved 0.42471) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by avoiding the problematic `pydicom` import path and reading DICOMs through `tensorflow.io.gfile` + `SimpleITK` (available in Kaggle for this competition) with a small fallback to `pydicom` only if needed. I also fix the training crash caused by `keras.metrics.AUC` expecting probability inputs while the model outputs a 2-class softmax with sparse labels; the minimal score-neutral fix is to remove the AUC metric during `compile()` (loss/training logic stays the same). Finally, I keep your 3-model / 6-slice ensemble logic unchanged, ensure shapes align for training/prediction, and write a valid `submission.csv` with the exact required columns and ordering from `sample_submission.csv`.'
- What this solution (achieved 0.42471) has done: 'The crash happens before any training because importing `SimpleITK` triggers a protobuf `MessageFactory.GetPrototype` incompatibility in this Kaggle image. To keep your core pipeline unchanged (6 slices per case, 3-model ensemble, same CNN architecture/training loop, same averaging for submission), I remove the `SimpleITK` dependency entirely and switch DICOM reading to a safe local `pydicom` import with a robust fallback that skips unreadable slices. This unblocks end-to-end execution and still produces `submission.csv` in the required format/order. I also add a tiny guard so if `pydicom` is unavailable for any reason, the loader fail loudly with a clear message rather than crashing later with obscure errors.'
- What this solution (achieved 0.42471) has done: 'The crash happens at import time in cell 1 because something in the TensorFlow/Keras stack triggers a protobuf incompatibility (`MessageFactory.GetPrototype`). To unblock execution end-to-end without changing your modeling/training/inference logic, I switch to the `tf.keras` namespace consistently (and avoid importing standalone `keras`), which prevents the problematic import path in this Kaggle image. I also keep your DICOM loading via local `pydicom` import, preserve the 3-model/6-slice ensemble and averaging exactly, and add a small safety guard to ensure IDs stay aligned and submission rows match `sample_submission.csv`. This should run cleanly and produce a valid `submission.csv` while keeping score behavior essentially the same (only import-path/stability changes).'
- What this solution (achieved 0.61294) has done: 'The import-time crash (`MessageFactory.GetPrototype`) is occurring in cell 0 when importing TensorFlow/Keras due to a protobuf incompatibility in this Kaggle image, so I switch the pipeline to a lightweight pure-numpy baseline that avoids TensorFlow entirely while keeping the core semantics (load 6 T2w slices per case, aggregate into one probability per case, and write `submission.csv`). I preserve your existing DICOM loading + slice selection logic, but make it more deterministic and slightly more predictive by learning a simple calibrated linear model (logistic regression implemented in numpy) from the same extracted slice tensors, using only training data (no leakage). This should run end-to-end within time, produce a valid submission, and typically improves AUC over an untrained/constant baseline while staying minimal and robust. Submission formatting/ordering still follow `sample_submission.csv` exactly.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE_PATH = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_PATH = os.path.join(BASE_PATH, "train")
TEST_PATH = os.path.join(BASE_PATH, "test")
LABELS_CSV = os.path.join(BASE_PATH, "train_labels.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")

print("TRAIN_PATH exists:", os.path.exists(TRAIN_PATH))
print("TEST_PATH exists:", os.path.exists(TEST_PATH))
print("LABELS_CSV exists:", os.path.exists(LABELS_CSV))
print("SAMPLE_SUB exists:", os.path.exists(SAMPLE_SUB))




## === cell 1
def resize(image2d: np.ndarray, out_hw):
    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    img = image2d.astype(np.float32)

    in_h, in_w = img.shape[:2]
    if in_h == out_h and in_w == out_w:
        return img

    y_idx = (np.linspace(0, in_h - 1, out_h)).astype(np.int32)
    x_idx = (np.linspace(0, in_w - 1, out_w)).astype(np.int32)
    return img[np.ix_(y_idx, x_idx)]


def _safe_normalize(img3: np.ndarray):
    m = float(np.max(img3))
    if not np.isfinite(m) or m <= 0:
        return None
    out = img3 / m
    if not np.isfinite(out).all():
        return None
    return out


def _get_case_id_from_path(case_path: str) -> str:
    return os.path.basename(case_path)


def _read_dicom_pixel_array(dcm_path: str):
    """
    Safe local pydicom import (avoids SimpleITK/protobuf issues).
    Returns 2D numpy array or raises.
    """
    try:
        import pydicom
    except Exception as e:
        raise RuntimeError("pydicom import failed; cannot read DICOMs.") from e

    ds = pydicom.dcmread(dcm_path, force=True)
    arr = ds.pixel_array
    if arr.ndim != 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(
            f"Unexpected DICOM pixel_array shape {arr.shape} for {dcm_path}"
        )
    return arr.astype(np.float32)




## === cell 2
def load_T2W_images_for_cases(
    path_root, case_ids=None, img_px_size=150, slices_per_case=6, mri_type_index=3
):
    """
    Core logic preserved: choose MRI type at index 3 (T2w in sorted order), scan slices,
    keep first N slices passing intensity filters, resize to IMG_PX_SIZE, stack to 3 channels,
    normalize, return 6 arrays (one per slice position) plus the ordered case ids used.
    """
    if case_ids is None:
        path_cases = sorted([f.path for f in os.scandir(path_root) if f.is_dir()])
        case_ids = [_get_case_id_from_path(p) for p in path_cases]
    else:
        case_ids = [str(c).zfill(5) for c in case_ids]
        path_cases = [os.path.join(path_root, cid) for cid in case_ids]

    arrays = [[] for _ in range(slices_per_case)]
    kept_case_ids = []

    for case_path, case_id in zip(path_cases, case_ids):
        if not os.path.isdir(case_path):
            continue

        mri_types = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])
        if len(mri_types) <= mri_type_index:
            continue

        series_path = mri_types[mri_type_index]
        img_paths = sorted(
            [
                f.path
                for f in os.scandir(series_path)
                if f.is_file() and f.name.lower().endswith(".dcm")
            ]
        )

        count = 0
        for p in img_paths:
            try:
                px = _read_dicom_pixel_array(p)
            except Exception:
                continue

            if float(np.sum(px)) <= 100000:
                continue

            rimg = resize(px, (img_px_size, img_px_size))
            img = np.array(rimg, dtype=np.float32)

            stacked = np.stack((img,) * 3, axis=-1)
            stacked_norm = _safe_normalize(stacked)
            if stacked_norm is None:
                continue

            if float(np.sum(stacked_norm)) <= 2000:
                continue

            if count < slices_per_case:
                arrays[count].append(stacked_norm)
                count += 1
            if count >= slices_per_case:
                break

        if count == 0:
            pad = np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)
            for j in range(slices_per_case):
                arrays[j].append(pad)
        elif count < slices_per_case:
            last = arrays[count - 1][-1]
            for j in range(count, slices_per_case):
                arrays[j].append(last)

        kept_case_ids.append(case_id)

    arrays = [np.asarray(a, dtype=np.float32) for a in arrays]
    normed = []
    for a in arrays:
        mx = float(np.max(a)) if a.size else 0.0
        if mx > 0:
            normed.append(a / mx)
        else:
            normed.append(a)

    print("Loaded cases:", len(kept_case_ids))
    print("Per-slice batch sizes:", [len(a) for a in normed])
    return kept_case_ids, normed




## === cell 3
labels_df = pd.read_csv(LABELS_CSV)
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(str).str.zfill(5)

bad_cases = {"00109", "00123", "00709"}
labels_df = labels_df[~labels_df["BraTS21ID"].isin(bad_cases)].reset_index(drop=True)

print(labels_df.head())
print("Train labels rows:", len(labels_df))



## === cell 4
train_case_ids = labels_df["BraTS21ID"].tolist()
y = labels_df["MGMT_value"].astype(np.float32).values

train_case_ids_loaded, train_slices = load_T2W_images_for_cases(
    TRAIN_PATH,
    case_ids=train_case_ids,
    img_px_size=150,
    slices_per_case=6,
    mri_type_index=3,
)

labels_map = dict(zip(labels_df["BraTS21ID"].tolist(), y))
y_loaded = np.array([labels_map[c] for c in train_case_ids_loaded], dtype=np.float32)

test_case_ids_loaded, test_slices = load_T2W_images_for_cases(
    TEST_PATH, case_ids=None, img_px_size=150, slices_per_case=6, mri_type_index=3
)

print(
    "Train loaded:",
    len(train_case_ids_loaded),
    " Test loaded:",
    len(test_case_ids_loaded),
)
print("y_loaded shape:", y_loaded.shape)
print(
    "Train slice[0] shape:",
    train_slices[0].shape,
    "Test slice[0] shape:",
    test_slices[0].shape,
)

for i, arr in enumerate(train_slices):
    if arr.shape[0] != len(train_case_ids_loaded):
        raise RuntimeError(
            f"Train slice {i} has {arr.shape[0]} rows but {len(train_case_ids_loaded)} case_ids."
        )
for i, arr in enumerate(test_slices):
    if arr.shape[0] != len(test_case_ids_loaded):
        raise RuntimeError(
            f"Test slice {i} has {arr.shape[0]} rows but {len(test_case_ids_loaded)} case_ids."
        )




## === cell 5
def _sigmoid(z):
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def _extract_features_from_slices(slices_list):
    """
    Score-improving but minimal: derive a small numeric feature vector per case from the same
    6 slice tensors (no architecture/training loop changes because we removed TF entirely).
    """
    feats = []
    for s in slices_list:
        x = s[..., 0].astype(np.float32)
        mu = x.mean(axis=(1, 2))
        sd = x.std(axis=(1, 2))
        q10 = np.quantile(x, 0.10, axis=(1, 2))
        q50 = np.quantile(x, 0.50, axis=(1, 2))
        q90 = np.quantile(x, 0.90, axis=(1, 2))
        bright = (x > 0.8).mean(axis=(1, 2)).astype(np.float32)
        feats.append(np.stack([mu, sd, q10, q50, q90, bright], axis=1))
    X = np.concatenate(feats, axis=1).astype(np.float32)
    return X


def _standardize_fit(X):
    mean = X.mean(axis=0, dtype=np.float64)
    std = X.std(axis=0, dtype=np.float64)
    std = np.where(std > 1e-6, std, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)


def _standardize_apply(X, mean, std):
    return ((X - mean) / std).astype(np.float32)


def train_logreg_numpy(X, y, lr=0.05, epochs=600, l2=1e-3, seed=42):
    """
    Minimal stable classifier: binary logistic regression trained with full-batch GD.
    Deterministic and fast; outputs calibrated probabilities.
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    w = rng.normal(0, 0.01, size=(d,)).astype(np.float32)
    b = np.float32(0.0)
    y = y.astype(np.float32)

    for _ in range(int(epochs)):
        z = X @ w + b
        p = _sigmoid(z).astype(np.float32)
        err = (p - y).astype(np.float32)
        grad_w = (X.T @ err) / n + (l2 * w)
        grad_b = err.mean(dtype=np.float32)
        w -= lr * grad_w
        b -= lr * grad_b

    return w, b


X_train = _extract_features_from_slices(train_slices)
X_test = _extract_features_from_slices(test_slices)

mean, std = _standardize_fit(X_train)
X_train_s = _standardize_apply(X_train, mean, std)
X_test_s = _standardize_apply(X_test, mean, std)

w, b = train_logreg_numpy(X_train_s, y_loaded, lr=0.05, epochs=600, l2=1e-3, seed=SEED)
test_pred = _sigmoid(X_test_s @ w + b).astype(np.float32)

print("Feature shapes:", X_train_s.shape, X_test_s.shape)
print(
    "Pred stats:",
    float(test_pred.min()),
    float(test_pred.max()),
    float(test_pred.mean()),
)




## === cell 6
def create_sub(case_ids, pred):
    pred = np.asarray(pred, dtype=np.float32).reshape(-1)
    pred = np.clip(pred, 0.0, 1.0)
    df = pd.DataFrame(
        {"BraTS21ID": [str(c).zfill(5) for c in case_ids], "MGMT_value": pred}
    )
    return df


sub_df = create_sub(test_case_ids_loaded, test_pred)
print(sub_df.head())
print("Raw sub shape:", sub_df.shape)



## === cell 7
sample = pd.read_csv(SAMPLE_SUB)
sample["BraTS21ID"] = sample["BraTS21ID"].astype(str).str.zfill(5)

sub_df = sample[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")
sub_df["MGMT_value"] = sub_df["MGMT_value"].astype(np.float32).fillna(0.5)

print(sub_df.head())
print(
    "Submission rows:",
    len(sub_df),
    "Missing preds after fill:",
    int(sub_df["MGMT_value"].isna().sum()),
)



## === cell 8
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with columns:", sub_df.columns.tolist())
print(sub_df.describe(include="all"))
