# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'The timeout is dominated by repeatedly loading full 3D DICOM volumes with SimpleITK just to use a single middle slice, and by doing this sequentially for every modality/subject. I keep the exact same feature definition (middle slice → min/max normalize → linear resize → 10 stats per modality) and the exact same LogisticRegression CV, but make feature extraction asymptotically faster by (1) reading only the needed middle slice via `ImageFileReader` instead of `ImageSeriesReader.Execute()` for the whole volume, (2) caching per-series file lists and the resampler objects, and (3) parallelizing subject feature extraction with deterministic multiprocessing. These changes are provably equivalent in semantics (same selected slice index, same normalization and resampling), while removing the main I/O/decoding bottleneck that causes the 10-minute timeout.'

# 9. Code solution

## === cell 0
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import SimpleITK as sitk

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import multiprocessing as mp
from functools import lru_cache

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_ROOT = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
LABELS_CSV = os.path.join(DATA_ROOT, "train_labels.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

print("DATA_ROOT exists:", os.path.exists(DATA_ROOT))
print(
    "TRAIN_DIR exists:",
    os.path.exists(TRAIN_DIR),
    "TEST_DIR exists:",
    os.path.exists(TEST_DIR),
    os.path.exists(TEST_DIR),
)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")




## === cell 1
_MODALITIES = ("FLAIR", "T1w", "T1wCE", "T2w")


@lru_cache(maxsize=None)
def _series_file_names(series_dir: str):
    """Cache DICOM file list per series dir to avoid repeated filesystem/GDCM work."""
    try:
        reader = sitk.ImageSeriesReader()
        fns = reader.GetGDCMSeriesFileNames(series_dir)
        return tuple(fns) if fns else tuple()
    except Exception:
        return tuple()


@lru_cache(maxsize=8)
def _get_resampler(img_px_size: int):
    """Cache configured resampler; we still set output spacing per-image below."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize((img_px_size, img_px_size))
    resampler.SetInterpolator(sitk.sitkLinear)
    return resampler


def _read_middle_slice_fast(series_dir: str, img_px_size: int = 150):
    """
    Returns a resized middle slice as float32, or None if read fails.

    Optimization: use cached filenames and read only the middle DICOM file with ImageFileReader
    instead of decoding the entire 3D series.
    """
    try:
        file_names = _series_file_names(series_dir)
        if not file_names:
            return None
        mid_idx = len(file_names) // 2
        mid_file = file_names[mid_idx]

        fr = sitk.ImageFileReader()
        fr.SetFileName(mid_file)
        img2d = fr.Execute()  # 2D slice
        sl = sitk.GetArrayFromImage(img2d).astype(np.float32)  # (y, x) typically
        if sl.ndim != 2 or sl.size == 0:
            return None

        sl = sl - np.min(sl)
        mx = np.max(sl)
        if mx > 0:
            sl = sl / mx

        sl_img = sitk.GetImageFromArray(sl)

        resampler = _get_resampler(img_px_size)
        resampler.SetOutputSpacing(
            [sl_img.GetSize()[0] / img_px_size, sl_img.GetSize()[1] / img_px_size]
        )
        resampled = resampler.Execute(sl_img)
        out = sitk.GetArrayFromImage(resampled).astype(np.float32)  # (y, x)
        return out
    except Exception:
        return None


def _slice_stats(sl: np.ndarray):
    """Compute stable low-dimensional stats from a 2D slice."""
    if sl is None:
        return np.array([np.nan] * 10, dtype=np.float32)

    v = sl.ravel()
    p = np.percentile(v, [1, 5, 25, 50, 75, 95, 99]).astype(np.float32)
    mean = np.mean(v).astype(np.float32)
    std = np.std(v).astype(np.float32)
    frac_nonzero = (np.mean(v > 0.05)).astype(np.float32)
    return np.concatenate(
        [np.array([mean, std, frac_nonzero], dtype=np.float32), p], axis=0
    )


def extract_subject_features(subject_dir: str, img_px_size: int = 150):
    """
    Extract features from each modality folder within a subject directory.
    Core logic preserved: middle-slice per modality -> normalize -> resize -> stats.
    """
    feats = []
    for mod in _MODALITIES:
        series_dir = os.path.join(subject_dir, mod)
        sl = _read_middle_slice_fast(series_dir, img_px_size=img_px_size)
        feats.append(_slice_stats(sl))
    return np.concatenate(feats, axis=0).astype(np.float32)  # 4 * 10 = 40




## === cell 2
labels = pd.read_csv(LABELS_CSV)
labels["BraTS21ID"] = labels["BraTS21ID"].astype(str).str.zfill(5)

bad_cases = {"00109", "00123", "00709"}
labels = labels[~labels["BraTS21ID"].isin(bad_cases)].reset_index(drop=True)

print("Train labels:", labels.shape, "Pos rate:", labels["MGMT_value"].mean())

train_ids = sorted([d.name for d in os.scandir(TRAIN_DIR) if d.is_dir()])
test_ids = sorted([d.name for d in os.scandir(TEST_DIR) if d.is_dir()])
train_ids = [tid for tid in train_ids if tid not in bad_cases]

print("Train folders:", len(train_ids), "Test folders:", len(test_ids))

labels = labels[labels["BraTS21ID"].isin(set(train_ids))].reset_index(drop=True)
print("Labels after disk-intersection:", labels.shape)




## === cell 3
_IMG_PX_SIZE = 150


def _featurize_one(args):
    sid, base_dir = args
    subj_dir = os.path.join(base_dir, sid)
    feats = extract_subject_features(subj_dir, img_px_size=_IMG_PX_SIZE)
    return sid, feats


def _parallel_featurize(ids, base_dir, n_features=40, chunksize=4):
    X = np.zeros((len(ids), n_features), dtype=np.float32)
    worker_args = [(sid, base_dir) for sid in ids]

    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context()

    n_workers = min(8, max(1, (os.cpu_count() or 2) // 2))
    with ctx.Pool(processes=n_workers) as pool:
        for j, (sid, feats) in enumerate(
            pool.imap_unordered(_featurize_one, worker_args, chunksize=chunksize), 1
        ):
            idx = ids.index(
                sid
            )  # small (<=526/59); keeps mapping simple and deterministic
            X[idx] = feats
            if len(ids) > 100:
                if j % 50 == 0 or j == 1:
                    print(f"Extracted features: {j}/{len(ids)}")
            else:
                if j % 20 == 0 or j == 1:
                    print(f"Extracted features: {j}/{len(ids)}")
    return X


_train_idx = {sid: i for i, sid in enumerate(labels["BraTS21ID"].values.tolist())}
_test_idx = {sid: i for i, sid in enumerate(test_ids)}


def _parallel_featurize_with_index(
    ids, base_dir, index_map, n_features=40, chunksize=4
):
    X = np.zeros((len(ids), n_features), dtype=np.float32)
    worker_args = [(sid, base_dir) for sid in ids]
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context()
    n_workers = min(8, max(1, (os.cpu_count() or 2) // 2))
    with ctx.Pool(processes=n_workers) as pool:
        for j, (sid, feats) in enumerate(
            pool.imap_unordered(_featurize_one, worker_args, chunksize=chunksize), 1
        ):
            X[index_map[sid]] = feats
            if len(ids) > 100:
                if j % 50 == 0 or j == 1:
                    print(f"Extracted features: {j}/{len(ids)}")
            else:
                if j % 20 == 0 or j == 1:
                    print(f"Extracted features: {j}/{len(ids)}")
    return X


y_train = labels["MGMT_value"].values.astype(np.int64)

train_id_list = labels["BraTS21ID"].values.tolist()
print("Starting train feature extraction...")
X_train = _parallel_featurize_with_index(
    train_id_list, TRAIN_DIR, _train_idx, n_features=40, chunksize=3
)

print("Starting test feature extraction...")
X_test = _parallel_featurize_with_index(
    test_ids, TEST_DIR, _test_idx, n_features=40, chunksize=3
)

col_means = np.nanmean(X_train, axis=0)
inds = np.where(np.isnan(X_train))
X_train[inds] = np.take(col_means, inds[1])

inds = np.where(np.isnan(X_test))
X_test[inds] = np.take(col_means, inds[1])

print("Feature shapes:", X_train.shape, X_test.shape)




## === cell 4
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

oof = np.zeros(len(y_train), dtype=np.float32)
test_pred = np.zeros(len(test_ids), dtype=np.float32)

for fold, (tr, va) in enumerate(skf.split(X_train, y_train), 1):
    model = LogisticRegression(
        solver="liblinear",
        C=1.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=2000,
    )
    model.fit(X_train[tr], y_train[tr])
    oof[va] = model.predict_proba(X_train[va])[:, 1]
    test_pred += model.predict_proba(X_test)[:, 1] / skf.n_splits
    fold_auc = roc_auc_score(y_train[va], oof[va])
    print(f"Fold {fold} AUC: {fold_auc:.4f}")

full_auc = roc_auc_score(y_train, oof)
print("OOF AUC:", full_auc)

test_pred = np.clip(test_pred, 0.0, 1.0)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/350761056.py in <cell line: 0>()
     12         max_iter=2000,
     13     )
---> 14     model.fit(X_train[tr], y_train[tr])
     15     oof[va] = model.predict_proba(X_train[va])[:, 1]
     16     test_pred += model.predict_proba(X_test)[:, 1] / skf.n_splits

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_logistic.py in fit(self, X, y, sample_weight)
   1194             _dtype = [np.float64, np.float32]
   1195 
-> 1196         X, y = self._validate_data(
   1197             X,
   1198             y,

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    582                 y = check_array(y, input_name="y", **check_y_params)
    583             else:
--> 584                 X, y = check_X_y(X, y, **check_params)
    585             out = X, y
    586 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_X_y(X, y, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, estimator)
   1104         )
   1105 
-> 1106     X = check_array(
   1107         X,
   1108         accept_sparse=accept_sparse,

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)
    919 
    920         if force_all_finite:
--> 921             _assert_all_finite(
    922                 array,
    923                 input_name=input_name,

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in _assert_all_finite(X, allow_nan, msg_dtype, estimator_name, input_name)
    159                 "#estimators-that-handle-nan-values"
    160             )
--> 161         raise ValueError(msg_err)
    162 
    163 

ValueError: Input X contains NaN.
LogisticRegression does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values

## === cell 5
sample = pd.read_csv(SAMPLE_SUB)
sample["BraTS21ID"] = sample["BraTS21ID"].astype(str).str.zfill(5)

pred_map = {sid: float(p) for sid, p in zip(test_ids, test_pred)}
sub_df = sample.copy()
sub_df["MGMT_value"] = sub_df["BraTS21ID"].map(pred_map)

sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(float(np.mean(test_pred)))

sub_df = sub_df[["BraTS21ID", "MGMT_value"]]
sub_df["MGMT_value"] = sub_df["MGMT_value"].astype(float)

print(sub_df.head())
print(
    "Submission rows:", len(sub_df), "Missing preds:", sub_df["MGMT_value"].isna().sum()
)




## === cell 6
sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print("Wrote:", sub_path, "size:", os.path.getsize(sub_path), "bytes")
