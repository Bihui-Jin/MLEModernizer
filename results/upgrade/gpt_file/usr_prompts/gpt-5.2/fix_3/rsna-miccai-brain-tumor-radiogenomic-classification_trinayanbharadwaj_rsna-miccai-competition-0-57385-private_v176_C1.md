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

0.61882

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fix the immediate runtime issues by removing/guarding imports that trigger the protobuf `MessageFactory.GetPrototype` crash and by ensuring `resize` is always available. Because the referenced pre-trained `.h5` models are not present in your environment, I add a minimal fallback that keeps the same “ensemble over a few slices per modality” logic but uses a lightweight in-notebook Keras CNN trained on the provided training set. I also fix the submission-building logic so predictions align per-case (the original code incorrectly averaged arrays inside a loop), handle missing/short slice lists robustly, and ensure `BraTS21ID` formatting matches the sample submission. Finally, the script always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.61882) has done: 'I fix the TensorFlow/protobuf crash by avoiding TensorFlow/Keras entirely and replacing the tiny CNN fallback with a lightweight, deterministic numpy-based slice-level logistic regression (same “take a few slices per modality and average per case” core semantics). I also fix the directory listing bug that caused `test`/`train` to be treated as a “case id” by filtering case folders to only numeric names. Finally, I ensure predictions align with `sample_submission.csv` IDs (proper 5-digit formatting) and always write a valid `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pydicom as dicom

try:
    from skimage.transform import resize
except Exception as e:
    raise ImportError(
        "scikit-image is required for resize(). Ensure scikit-image is available."
    ) from e

SEED = 42
np.random.seed(SEED)




## === cell 1
DATA_ROOT_CANDIDATES = [
    "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification",
    "/kaggle/data/rsna-miccai-brain-tumor-radiogenomic-classification",
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification",
]

DATA_ROOT = None
for p in DATA_ROOT_CANDIDATES:
    if os.path.exists(p):
        DATA_ROOT = p
        break

if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate competition dataset folder in expected paths."
    )

TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
TRAIN_LABELS_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_ROOT, "sample_submission.csv")

assert os.path.isdir(TRAIN_DIR), f"Missing train dir: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing test dir: {TEST_DIR}"
assert os.path.isfile(TRAIN_LABELS_CSV), f"Missing train_labels.csv: {TRAIN_LABELS_CSV}"
assert os.path.isfile(
    SAMPLE_SUB_CSV
), f"Missing sample_submission.csv: {SAMPLE_SUB_CSV}"

train_labels = pd.read_csv(TRAIN_LABELS_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)

BAD_CASES = {109, 123, 709}
train_labels = train_labels[~train_labels["BraTS21ID"].isin(BAD_CASES)].reset_index(
    drop=True
)

train_labels.head(), sample_sub.head()




## === cell 2
MODALITY_TO_INDEX = {
    "FLAIR": 0,
    "T1w": 1,
    "T1wCE": 2,
    "T2w": 3,
}


def _safe_norm01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    mx = float(np.max(x)) if x.size else 0.0
    if mx <= 0:
        return x
    return x / mx


def _is_case_dir_entry(entry: os.DirEntry) -> bool:
    if not entry.is_dir():
        return False
    name = entry.name
    return name.isdigit()


def _list_case_dirs(path_root: str):
    return sorted([e.path for e in os.scandir(path_root) if _is_case_dir_entry(e)])


def _case_id_from_path(case_path: str) -> int:
    return int(os.path.basename(case_path))


def load_cases_slices(
    path_root: str,
    modality: str,
    img_px_size: int = 150,
    n_slices: int = 6,
    pixel_sum_threshold: float = 100000.0,
    norm_sum_threshold: float = 2000.0,
):
    """
    Returns:
      case_ids: list[int] length N_cases
      slices: list[np.ndarray] length n_slices, each of shape (N_cases, H, W, 3)
              (filled with zeros if not enough qualifying slices)
      valid_mask: np.ndarray shape (N_cases, n_slices) boolean indicating real vs padded slices
    """
    assert modality in MODALITY_TO_INDEX, f"Unknown modality: {modality}"
    m_idx = MODALITY_TO_INDEX[modality]

    case_paths = _list_case_dirs(path_root)
    case_ids = [_case_id_from_path(p) for p in case_paths]
    n_cases = len(case_paths)

    slices = [
        np.zeros((n_cases, img_px_size, img_px_size, 3), dtype=np.float32)
        for _ in range(n_slices)
    ]
    valid_mask = np.zeros((n_cases, n_slices), dtype=bool)

    for i, case_path in enumerate(case_paths):
        count = 0
        mri_type_dirs = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])
        if len(mri_type_dirs) <= m_idx:
            continue

        img_dir = mri_type_dirs[m_idx]
        dcm_files = sorted([f.path for f in os.scandir(img_dir) if f.is_file()])

        for fp in dcm_files:
            if count >= n_slices:
                break
            try:
                ds = dicom.dcmread(fp)
                arr = ds.pixel_array
            except Exception:
                continue

            if float(np.sum(arr)) <= pixel_sum_threshold:
                continue

            try:
                resized = resize(
                    arr,
                    (img_px_size, img_px_size),
                    preserve_range=True,
                    anti_aliasing=True,
                ).astype(np.float32)
            except Exception:
                continue

            stacked = np.stack((resized,) * 3, axis=-1)  # H,W,3
            stacked = _safe_norm01(stacked)

            if float(np.sum(stacked)) <= norm_sum_threshold:
                continue

            slices[count][i] = stacked
            valid_mask[i, count] = True
            count += 1

    return case_ids, slices, valid_mask




## === cell 3
test_ids_t2, test_t2_slices, test_t2_mask = load_cases_slices(TEST_DIR, "T2w")
test_ids_flair, test_flair_slices, test_flair_mask = load_cases_slices(
    TEST_DIR, "FLAIR"
)
test_ids_t1ce, test_t1ce_slices, test_t1ce_mask = load_cases_slices(TEST_DIR, "T1wCE")
test_ids_t1, test_t1_slices, test_t1_mask = load_cases_slices(TEST_DIR, "T1w")

assert (
    test_ids_t2 == test_ids_flair == test_ids_t1ce == test_ids_t1
), "Case ordering mismatch across modalities."
test_case_ids = test_ids_t2
len(test_case_ids), test_case_ids[:5]




## === cell 4


def _sigmoid(z):
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def extract_features_from_slice_batch(X: np.ndarray) -> np.ndarray:
    """
    X: (N,H,W,3) float32 in [0,1]
    Returns: (N,F) float32
    Lightweight summary stats (keeps runtime small).
    """
    x = X[..., 0]  # all channels identical by construction
    n = x.shape[0]
    flat = x.reshape(n, -1)
    mean = flat.mean(axis=1)
    std = flat.std(axis=1)
    q10 = np.quantile(flat, 0.10, axis=1)
    q50 = np.quantile(flat, 0.50, axis=1)
    q90 = np.quantile(flat, 0.90, axis=1)

    gx = np.abs(x[:, :, 1:] - x[:, :, :-1]).mean(axis=(1, 2))
    gy = np.abs(x[:, 1:, :] - x[:, :-1, :]).mean(axis=(1, 2))

    feats = np.stack([mean, std, q10, q50, q90, gx, gy], axis=1).astype(np.float32)
    return feats


def fit_logreg_l2(X: np.ndarray, y: np.ndarray, lr=0.2, steps=500, l2=1e-2) -> dict:
    """
    Simple batch GD logistic regression with L2 on weights.
    X: (N,F), y: (N,)
    """
    X = X.astype(np.float32, copy=False)
    y = y.astype(np.float32, copy=False).reshape(-1)
    n, f = X.shape

    mu = X.mean(axis=0)
    sig = X.std(axis=0) + 1e-6
    Xs = (X - mu) / sig

    w = np.zeros((f,), dtype=np.float32)
    b = np.float32(0.0)

    for _ in range(steps):
        p = _sigmoid(Xs @ w + b).astype(np.float32)
        err = (p - y).astype(np.float32)
        gw = (Xs.T @ err) / n + l2 * w
        gb = err.mean(dtype=np.float32)
        w -= lr * gw.astype(np.float32)
        b -= lr * gb.astype(np.float32)

    return {"w": w, "b": float(b), "mu": mu, "sig": sig}


def predict_logreg(model: dict, X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32, copy=False)
    Xs = (X - model["mu"]) / model["sig"]
    return _sigmoid(Xs @ model["w"] + model["b"]).astype(np.float32)


def load_train_slices_and_labels(train_dir, labels_df, modality):
    case_id_to_label = dict(
        zip(
            labels_df["BraTS21ID"].values.tolist(),
            labels_df["MGMT_value"].values.tolist(),
        )
    )

    all_ids, slices, mask = load_cases_slices(train_dir, modality)
    keep = [i for i, cid in enumerate(all_ids) if cid in case_id_to_label]

    slices_f = [s[keep] for s in slices]
    mask_f = mask[keep]
    y = np.array([case_id_to_label[all_ids[i]] for i in keep], dtype=np.float32)
    kept_ids = [all_ids[i] for i in keep]
    return kept_ids, slices_f, mask_f, y


def train_modality_model(modality: str) -> dict:
    tr_ids, tr_slices, tr_mask, y_case = load_train_slices_and_labels(
        TRAIN_DIR, train_labels, modality
    )

    X_list, y_list = [], []
    for s_idx in range(6):
        valid = tr_mask[:, s_idx]
        if np.any(valid):
            feats = extract_features_from_slice_batch(tr_slices[s_idx][valid])
            X_list.append(feats)
            y_list.append(y_case[valid])

    if len(X_list) == 0:
        return {"dummy": True}

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    rng = np.random.default_rng(SEED)
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)
    X = X[idx]
    y = y[idx]

    model = fit_logreg_l2(X, y, lr=0.2, steps=600, l2=1e-2)
    model["dummy"] = False
    return model


fallback_models = {m: train_modality_model(m) for m in ["T2w", "FLAIR", "T1wCE", "T1w"]}
list(fallback_models.keys())




## === cell 5
def predict_casewise_logreg(model: dict, slices, mask) -> np.ndarray:
    n_cases = slices[0].shape[0]
    if model.get("dummy", False):
        return np.full((n_cases,), 0.5, dtype=np.float32)

    preds = np.zeros((n_cases,), dtype=np.float32)
    counts = np.zeros((n_cases,), dtype=np.float32)

    for s_idx in range(len(slices)):
        valid = mask[:, s_idx]
        if not np.any(valid):
            continue
        feats = extract_features_from_slice_batch(slices[s_idx][valid])
        p = predict_logreg(model, feats)
        preds[valid] += p
        counts[valid] += 1.0

    out = np.where(counts > 0, preds / counts, 0.5).astype(np.float32)
    return out


def predict_ensemble(test_case_ids):
    preds_t2 = predict_casewise_logreg(
        fallback_models["T2w"], test_t2_slices, test_t2_mask
    )
    preds_flair = predict_casewise_logreg(
        fallback_models["FLAIR"], test_flair_slices, test_flair_mask
    )
    preds_t1ce = predict_casewise_logreg(
        fallback_models["T1wCE"], test_t1ce_slices, test_t1ce_mask
    )
    preds_t1 = predict_casewise_logreg(
        fallback_models["T1w"], test_t1_slices, test_t1_mask
    )

    ens = (preds_t2 + preds_flair + preds_t1ce + preds_t1) / 4.0
    return ens.astype(np.float32)


test_pred = predict_ensemble(test_case_ids)
test_pred[:10], float(test_pred.min()), float(test_pred.max())




## === cell 6
sub = sample_sub.copy()
sub["BraTS21ID"] = sub["BraTS21ID"].astype(int)

pred_map = {int(cid): float(p) for cid, p in zip(test_case_ids, test_pred)}
sub["MGMT_value"] = sub["BraTS21ID"].map(pred_map).astype(float)

sub["MGMT_value"] = sub["MGMT_value"].fillna(0.5).clip(0.0, 1.0)
sub["BraTS21ID"] = sub["BraTS21ID"].apply(lambda x: f"{int(x):05d}")

sub.head(), sub.shape




## === cell 7
sub.to_csv("submission.csv", index=False)

assert os.path.isfile("submission.csv")
check = pd.read_csv("submission.csv")
assert list(check.columns) == ["BraTS21ID", "MGMT_value"]
check.head()
