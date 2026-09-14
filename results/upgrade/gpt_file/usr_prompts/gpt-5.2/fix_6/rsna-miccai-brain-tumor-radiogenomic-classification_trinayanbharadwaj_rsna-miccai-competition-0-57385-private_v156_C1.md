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

0.54235

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.45765) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation before importing TensorFlow (and keeping the TF import optional so the fallback predictor still runs). Then I fix the test-case ID extraction to exactly match the loader’s directory order (including non-numeric folders) and to align predictions to the provided `sample_submission.csv` order, eliminating the 60-vs-59 mismatch. Finally, I ensure `sub_df` is always created and that `submission.csv` is written end-to-end even when pretrained models are unavailable. These changes are score-neutral aside from preventing ordering/ID mismatches that would otherwise destroy AUC.'
- What this solution (achieved 0.54235) has done: 'I fix the TensorFlow/protobuf crash by avoiding TensorFlow entirely (since the pretrained models are not available at the referenced path anyway), ensuring the notebook runs end-to-end reliably. Then I improve the fallback predictor slightly but still within the same “deterministic intensity-based heuristic” core approach by calibrating it on the training labels (no leakage) using a simple 1D logistic regression on the same extracted intensity features, which should move AUC up toward the target band. Finally, I keep the submission creation aligned to `sample_submission.csv` order and always write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    plt = None
    sns = None

import pydicom as dicom
from skimage.transform import resize

np.random.seed(42)

tf = None
keras = None
_TF_AVAILABLE = False
print(
    "TensorFlow disabled to avoid protobuf crash; using calibrated fallback predictor."
)



## === cell 1
MODEL_DIR = "../input/trained-model-for-rsnamiccai"

MODEL_PATHS = [
    "rsna_miccai_114_epochs_T2W_7k_imgs.h5",
    "rsna_miccai_200_epochs_T2W_7k_imgs.h5",
    "rsna_miccai_10_b800_t1wce_7k_0.64auc_imgs.h5",
    "rsna_miccai_10_b800_t1wce_7k_0.65auc_imgs.h5",
    "rsna_miccai_10_b600_T2w_7k_0.62auc_imgs.h5",
    "rsna_miccai_15_b600_T2w_7k_0.74auc_imgs.h5",
    "rsna_miccai_10_b600_flair_5.5k_0.70auc_imgs.h5",
]


def _safe_load_model(path):
    return None


models = [None for _ in MODEL_PATHS]
use_pretrained = False
print(
    "Pretrained model loading skipped (TF disabled / model dir likely unavailable). Using fallback predictor."
)




## === cell 2
def _to_float32_image(img2d, img_px_size=150):
    """Resize 2D to (img_px_size,img_px_size), stack to 3ch, normalize to [0,1]."""
    resized_img = resize(
        img2d, (img_px_size, img_px_size), preserve_range=True, anti_aliasing=True
    ).astype(np.float32)
    stacked = np.stack([resized_img, resized_img, resized_img], axis=-1)
    mx = float(np.max(stacked))
    if mx > 0:
        stacked = stacked / mx
    return stacked


def _get_series_dir(subject_dir, series_name):
    p = os.path.join(subject_dir, series_name)
    return p if os.path.isdir(p) else None


def _read_best_slices(
    series_dir, n_slices=6, img_px_size=150, sum_thr=100000, normsum_thr=2000
):
    """
    Reads DICOMs in series_dir, returns up to n_slices normalized RGB images (H,W,3).
    Selection logic preserved: pixel sum threshold + normalized sum threshold.
    If fewer found, pads by repeating the last found slice or zeros.
    """
    if series_dir is None:
        imgs = []
    else:
        files = sorted([f.path for f in os.scandir(series_dir) if f.is_file()])
        imgs = []
        for fp in files:
            try:
                d = dicom.dcmread(fp, force=True)
                arr = d.pixel_array
            except Exception:
                continue
            if arr is None:
                continue
            if float(np.sum(arr)) > sum_thr:
                im = _to_float32_image(arr, img_px_size=img_px_size)
                if float(np.sum(im)) > normsum_thr:
                    imgs.append(im)
                    if len(imgs) >= n_slices:
                        break

    if len(imgs) == 0:
        imgs = [np.zeros((img_px_size, img_px_size, 3), dtype=np.float32)]
    while len(imgs) < n_slices:
        imgs.append(imgs[-1].copy())
    return np.stack(imgs[:n_slices], axis=0)


def load_test_images_by_modality(path_test, modality, n_slices=6, img_px_size=150):
    """
    Returns list of per-slice batches: [slice0_batch, slice1_batch, ... slice5_batch]
    where each slice batch is (N, H, W, 3).
    """
    subject_dirs = sorted([f.path for f in os.scandir(path_test) if f.is_dir()])
    per_slice = [[] for _ in range(n_slices)]

    for subj in subject_dirs:
        series_dir = _get_series_dir(subj, modality)
        stack = _read_best_slices(
            series_dir, n_slices=n_slices, img_px_size=img_px_size
        )
        for s in range(n_slices):
            per_slice[s].append(stack[s])

    per_slice = [np.stack(x, axis=0).astype(np.float32) for x in per_slice]
    out = []
    for arr in per_slice:
        mx = float(np.max(arr))
        out.append(arr / mx if mx > 0 else arr)
    return out




## === cell 3
BASE_PATH = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"
TEST_PATH = os.path.join(BASE_PATH, "test")

if not os.path.isdir(TEST_PATH):
    raise FileNotFoundError(f"Test folder not found: {TEST_PATH}")

outer_dirs = [d.name for d in os.scandir(TEST_PATH) if d.is_dir()]
has_numeric_outer = any(name.isdigit() for name in outer_dirs)
nested_test = os.path.join(TEST_PATH, "test")
if (not has_numeric_outer) and os.path.isdir(nested_test):
    TEST_PATH = nested_test

print("Using TEST_PATH =", TEST_PATH)

pixels_T2 = load_test_images_by_modality(TEST_PATH, "T2w", n_slices=6, img_px_size=150)
pixels_flair = load_test_images_by_modality(
    TEST_PATH, "FLAIR", n_slices=6, img_px_size=150
)
pixels_t1wce = load_test_images_by_modality(
    TEST_PATH, "T1wCE", n_slices=6, img_px_size=150
)

print(
    "Loaded test batches per slice:",
    "T2:",
    [p.shape for p in pixels_T2],
    "FLAIR:",
    [p.shape for p in pixels_flair],
    "T1wCE:",
    [p.shape for p in pixels_t1wce],
)




## === cell 4
def _extract_subject_features(pixels_T2, pixels_flair, pixels_t1wce):
    """
    Feature extraction consistent with existing fallback core logic:
    per modality -> mean of per-slice mean intensity, giving 3 features per subject.
    """
    feats = []
    for modality in (pixels_T2, pixels_flair, pixels_t1wce):
        slice_means = [arr.mean(axis=(1, 2, 3)) for arr in modality]  # list of (N,)
        feats.append(np.mean(np.stack(slice_means, axis=1), axis=1))  # (N,)
    X = np.stack(feats, axis=1).astype(np.float32)  # (N,3)
    return X


def _sigmoid(z):
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def _fit_logreg_1d(X, y, lr=0.1, n_iter=4000, l2=1e-3):
    """
    Minimal calibration model: logistic regression on 1D score s = X @ w0.
    Fits (a,b) on s via gradient descent (deterministic).
    """
    y = y.astype(np.float32).reshape(-1)
    s = X.astype(np.float32).reshape(-1)

    a = np.float32(0.0)
    b = np.float32(0.0)

    n = np.float32(len(y))
    for _ in range(int(n_iter)):
        p = _sigmoid(a * s + b).astype(np.float32)
        da = np.sum((p - y) * s) / n + l2 * a
        db = np.sum((p - y)) / n + l2 * b
        a -= np.float32(lr) * np.float32(da)
        b -= np.float32(lr) * np.float32(db)
    return float(a), float(b)


def _calibrated_fallback_prob(path_train, pixels_T2, pixels_flair, pixels_t1wce):
    """
    Score-improving but still minimal: use the same 3 intensity features, then
    calibrate a logistic mapping using train labels (no test leakage).
    If training feature extraction fails, revert to the old fixed-weight fallback.
    """
    X_test = _extract_subject_features(pixels_T2, pixels_flair, pixels_t1wce)

    def old_fixed_prob(X):
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True)
        sd = np.where(sd > 1e-6, sd, 1.0)
        Z = (X - mu) / sd
        w = np.array([0.6, 0.3, 0.4], dtype=np.float32)
        logits = (Z * w.reshape(1, -1)).sum(axis=1)
        return _sigmoid(logits).astype(np.float32)

    labels_path = os.path.join(BASE_PATH, "train_labels.csv")
    if not os.path.exists(labels_path):
        labels_path = "../input/train_labels.csv"
    if not os.path.exists(labels_path):
        print("Warning: train_labels.csv not found; using fixed-weight fallback.")
        return old_fixed_prob(X_test)

    train_labels = pd.read_csv(labels_path)
    bad_ids = {109, 123, 709}
    train_labels = train_labels[
        ~train_labels["BraTS21ID"].astype(int).isin(bad_ids)
    ].copy()

    train_path = os.path.join(BASE_PATH, "train")
    if not os.path.isdir(train_path):
        train_path = (
            "../input/rsna-miccai-brain-tumor-radiogenomic-classification/train"
        )
    if not os.path.isdir(train_path):
        print("Warning: train folder not found; using fixed-weight fallback.")
        return old_fixed_prob(X_test)

    try:
        pixels_T2_tr = load_test_images_by_modality(
            train_path, "T2w", n_slices=6, img_px_size=150
        )
        pixels_flair_tr = load_test_images_by_modality(
            train_path, "FLAIR", n_slices=6, img_px_size=150
        )
        pixels_t1wce_tr = load_test_images_by_modality(
            train_path, "T1wCE", n_slices=6, img_px_size=150
        )
        X_train = _extract_subject_features(
            pixels_T2_tr, pixels_flair_tr, pixels_t1wce_tr
        )

        train_subject_dirs = sorted(
            [f.path for f in os.scandir(train_path) if f.is_dir()]
        )
        train_ids = []
        for p in train_subject_dirs:
            name = os.path.basename(p)
            if name.isdigit():
                train_ids.append(int(name))
            else:
                train_ids.append(None)

        y_map = dict(
            zip(
                train_labels["BraTS21ID"].astype(int).values,
                train_labels["MGMT_value"].astype(int).values,
            )
        )
        keep_idx = [
            i for i, _id in enumerate(train_ids) if _id is not None and _id in y_map
        ]
        if len(keep_idx) < 50:
            print(
                "Warning: too few labeled train cases aligned; using fixed-weight fallback."
            )
            return old_fixed_prob(X_test)

        X_train_k = X_train[keep_idx]
        y_train_k = np.array([y_map[train_ids[i]] for i in keep_idx], dtype=np.float32)

        mu = X_train_k.mean(axis=0, keepdims=True)
        sd = X_train_k.std(axis=0, keepdims=True)
        sd = np.where(sd > 1e-6, sd, 1.0)
        Z_train = (X_train_k - mu) / sd
        Z_test = (X_test - mu) / sd

        w0 = np.array([0.6, 0.3, 0.4], dtype=np.float32)
        s_train = (Z_train * w0.reshape(1, -1)).sum(axis=1)
        s_test = (Z_test * w0.reshape(1, -1)).sum(axis=1)

        a, b = _fit_logreg_1d(s_train, y_train_k, lr=0.1, n_iter=4000, l2=1e-3)
        prob = _sigmoid(a * s_test + b).astype(np.float32)
        return prob
    except Exception as e:
        print(
            "Warning: train-based calibration failed; using fixed-weight fallback. Error:",
            repr(e),
        )
        return old_fixed_prob(X_test)


prediction = _calibrated_fallback_prob(
    os.path.join(BASE_PATH, "train"), pixels_T2, pixels_flair, pixels_t1wce
)

print(
    "Final prediction stats:",
    "N=",
    int(prediction.shape[0]),
    "min=",
    float(np.min(prediction)),
    "max=",
    float(np.max(prediction)),
    "mean=",
    float(np.mean(prediction)),
)




## === cell 5
def _list_subject_dirnames_in_loader_order(path_test):
    """Must exactly match load_test_images_by_modality() subject ordering."""
    return [
        os.path.basename(f.path)
        for f in sorted(os.scandir(path_test), key=lambda x: x.path)
        if f.is_dir()
    ]


def _is_valid_brats_id(name: str) -> bool:
    return name.isdigit()


def create_sub(path_test, prediction, base_path_for_sample):
    sample_path = os.path.join(base_path_for_sample, "sample_submission.csv")
    if not os.path.exists(sample_path):
        sample_path = "../input/sample_submission.csv"

    sample = pd.read_csv(sample_path)
    if "BraTS21ID" not in sample.columns:
        raise ValueError(
            f"sample_submission.csv missing BraTS21ID column: {sample.columns.tolist()}"
        )

    subject_names = _list_subject_dirnames_in_loader_order(path_test)

    id_in_dir = []
    pred_in_dir = []
    for name, pred in zip(subject_names, prediction):
        if _is_valid_brats_id(name):
            id_in_dir.append(int(name))
            pred_in_dir.append(float(pred))

    df = pd.DataFrame({"BraTS21ID": id_in_dir, "MGMT_value": pred_in_dir})

    df = df.set_index("BraTS21ID")
    sample_ids = sample["BraTS21ID"].astype(int).values
    missing = [i for i in sample_ids if i not in df.index]
    if missing:
        raise ValueError(
            f"Missing {len(missing)} IDs present in sample_submission but not predicted from folders. "
            f"First few missing: {missing[:10]}"
        )

    df = df.reindex(sample_ids).reset_index()
    df.columns = ["BraTS21ID", "MGMT_value"]
    df["MGMT_value"] = df["MGMT_value"].clip(0.0, 1.0)
    return df


sub_df = create_sub(TEST_PATH, prediction, BASE_PATH)

print("Submission preview:")
print(sub_df.head())
print("Submission shape:", sub_df.shape)



## === cell 6
if sns is not None:
    try:
        sns.displot(sub_df["MGMT_value"])
    except Exception:
        pass



## === cell 7
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head())
