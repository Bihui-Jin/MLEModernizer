# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import pydicom as dicom
from pydicom.errors import InvalidDicomError

from skimage.transform import resize

try:
    import seaborn as sns
except Exception:
    sns = None

dicom.config.enforce_valid_values = False




## === cell 1
def load_test_T2W_images(path_test):
    print("Skipping T2 image loading (unused by downstream prediction).")
    empty = np.empty((0,), dtype=np.float32)
    return empty, empty, empty, empty, empty, empty


def load_test_flair_images(path_test):
    print("Skipping FLAIR image loading (unused by downstream prediction).")
    empty = np.empty((0,), dtype=np.float32)
    return empty, empty, empty, empty, empty, empty




## === cell 2
test = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/test"
sample_sub_path = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv"


def _resolve_test_dir(p):
    if os.path.isdir(p) and any(e.is_dir() for e in os.scandir(p)):
        names = [e.name for e in os.scandir(p) if e.is_dir()]
        if any(n.isdigit() for n in names):
            return p
        nested = os.path.join(
            p, "rsna-miccai-brain-tumor-radiogenomic-classification", "test"
        )
        if os.path.isdir(nested):
            return nested
    return p


test = _resolve_test_dir(test)
print("Using test directory:", test)




## === cell 3
pixels_1, pixels_2, pixels_3, pixels_4, pixels_5, pixels_6 = load_test_T2W_images(test)
pixels_7, pixels_8, pixels_9, pixels_10, pixels_11, pixels_12 = load_test_flair_images(
    test
)




## === cell 4
train_labels_path = (
    "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv"
)
labels_df = pd.read_csv(train_labels_path)
base_rate = float(labels_df["MGMT_value"].mean())
base_rate = min(max(base_rate, 1e-6), 1 - 1e-6)

_CASEDIR_CACHE = {}


def _list_numeric_case_dirs(path_root):
    cached = _CASEDIR_CACHE.get(path_root)
    if cached is not None:
        return cached
    case_dirs = []
    try:
        for e in os.scandir(path_root):
            if e.is_dir() and e.name.isdigit():
                case_dirs.append(e.path)
    except FileNotFoundError:
        case_dirs = []
    case_dirs = sorted(case_dirs)
    _CASEDIR_CACHE[path_root] = case_dirs
    return case_dirs


n_test = len(_list_numeric_case_dirs(test))


def _make_pred_array(n):
    return np.full((n,), base_rate, dtype=np.float32)


prediction_1 = _make_pred_array(n_test)
prediction_2 = _make_pred_array(n_test)
prediction_3 = _make_pred_array(n_test)
prediction_4 = _make_pred_array(n_test)
prediction_5 = _make_pred_array(n_test)
prediction_6 = _make_pred_array(n_test)

prediction_101 = _make_pred_array(n_test)
prediction_102 = _make_pred_array(n_test)
prediction_103 = _make_pred_array(n_test)
prediction_104 = _make_pred_array(n_test)
prediction_105 = _make_pred_array(n_test)
prediction_106 = _make_pred_array(n_test)

prediction_401 = _make_pred_array(n_test)
prediction_402 = _make_pred_array(n_test)
prediction_403 = _make_pred_array(n_test)
prediction_404 = _make_pred_array(n_test)
prediction_405 = _make_pred_array(n_test)
prediction_406 = _make_pred_array(n_test)

prediction_501 = _make_pred_array(n_test)
prediction_502 = _make_pred_array(n_test)
prediction_503 = _make_pred_array(n_test)
prediction_504 = _make_pred_array(n_test)
prediction_505 = _make_pred_array(n_test)
prediction_506 = _make_pred_array(n_test)

prediction_601 = _make_pred_array(n_test)
prediction_602 = _make_pred_array(n_test)
prediction_603 = _make_pred_array(n_test)
prediction_604 = _make_pred_array(n_test)
prediction_605 = _make_pred_array(n_test)
prediction_606 = _make_pred_array(n_test)

print(
    f"Using baseline probability (train positive rate): {base_rate:.6f} for n_test={n_test}"
)




## === cell 5
def _safe_sigmoid(x):
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def _iter_sorted_files(series_dir):
    try:
        files = [e.name for e in os.scandir(series_dir) if e.is_file()]
    except Exception:
        return []
    if not files:
        return []
    files.sort()
    return [os.path.join(series_dir, fn) for fn in files]


def _case_feature_from_series(series_dir, img_px_size=96, max_slices=24):
    files = _iter_sorted_files(series_dir)
    if not files:
        return np.nan

    vals = []
    for p in files:
        try:
            ds = dicom.dcmread(p, stop_before_pixels=True, force=True)
            rows = int(getattr(ds, "Rows", 0) or 0)
            cols = int(getattr(ds, "Columns", 0) or 0)
            if rows <= 0 or cols <= 0:
                continue

            max_tag = None
            if "LargestImagePixelValue" in ds:
                try:
                    max_tag = float(ds.LargestImagePixelValue)
                except Exception:
                    max_tag = None

            if max_tag is not None:
                if max_tag * (rows * cols) <= 100000.0:
                    continue
        except (InvalidDicomError, Exception):
            continue

        try:
            img = dicom.dcmread(p, force=True)
            px = img.pixel_array
        except Exception:
            continue

        if px.sum() <= 100000:
            continue

        resized_img = resize(
            px,
            (img_px_size, img_px_size),
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.float32)

        mx = float(np.max(resized_img))
        if mx <= 0:
            continue
        arr = resized_img / mx
        if float(arr.sum()) <= 2000:
            continue

        vals.append(float(arr.mean() + 0.5 * arr.std()))
        if len(vals) >= max_slices:
            break

    if not vals:
        return np.nan
    return float(np.median(vals))


def _compute_features_for_path(root_dir):
    case_dirs = _list_numeric_case_dirs(root_dir)
    case_ids = np.fromiter(
        (int(os.path.basename(p)) for p in case_dirs), dtype=int, count=len(case_dirs)
    )

    feats = np.empty((len(case_dirs),), dtype=np.float32)
    feats.fill(np.nan)

    for i, case_path in enumerate(case_dirs):
        try:
            mri_type = sorted([f.path for f in os.scandir(case_path) if f.is_dir()])
        except Exception:
            continue

        f_flair = np.nan
        f_t2 = np.nan
        if len(mri_type) >= 1:
            f_flair = _case_feature_from_series(mri_type[0])
        if len(mri_type) >= 4:
            f_t2 = _case_feature_from_series(mri_type[3])

        if np.isnan(f_flair) and np.isnan(f_t2):
            feats[i] = np.nan
        elif np.isnan(f_flair):
            feats[i] = f_t2
        elif np.isnan(f_t2):
            feats[i] = f_flair
        else:
            feats[i] = np.float32(0.5 * (f_flair + f_t2))

    return case_ids, feats


train_root = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train"
train_root = train_root  # keep path unchanged/explicit

bad_ids = set([109, 123, 709])

train_case_ids, train_feats = _compute_features_for_path(train_root)
train_df = pd.DataFrame({"BraTS21ID": train_case_ids, "feat": train_feats})
train_df = train_df[~train_df["BraTS21ID"].isin(bad_ids)].copy()

train_lab = labels_df.copy()
train_lab["BraTS21ID"] = train_lab["BraTS21ID"].astype(int)
train_df = train_df.merge(train_lab, on="BraTS21ID", how="inner")
train_df = (
    train_df.replace([np.inf, -np.inf], np.nan)
    .dropna(subset=["feat"])
    .reset_index(drop=True)
)

if len(train_df) < 20:
    print(
        "Warning: too few valid train features; falling back to constant predictions."
    )
    calib_a, calib_b = 0.0, float(np.log(base_rate / (1.0 - base_rate)))
    feat_mean, feat_std = 0.0, 1.0
else:
    feat_mean = float(train_df["feat"].mean())
    feat_std = float(train_df["feat"].std())
    if not np.isfinite(feat_std) or feat_std <= 1e-6:
        feat_std = 1.0

    z = ((train_df["feat"].values.astype(np.float32) - feat_mean) / feat_std).astype(
        np.float32
    )
    y = train_df["MGMT_value"].values.astype(np.float32)

    a = 0.0
    b = float(np.log(base_rate / (1.0 - base_rate)))
    for _ in range(25):
        p = _safe_sigmoid(a * z + b).astype(np.float32)
        ga = float(np.sum((p - y) * z))
        gb = float(np.sum(p - y))
        w = (p * (1.0 - p)).astype(np.float32)
        haa = float(np.sum(w * z * z)) + 1e-6
        hbb = float(np.sum(w)) + 1e-6
        hab = float(np.sum(w * z))
        det = haa * hbb - hab * hab
        if det <= 1e-12:
            break
        da = (hbb * ga - hab * gb) / det
        db = (-hab * ga + haa * gb) / det
        a -= da
        b -= db
        if abs(da) + abs(db) < 1e-6:
            break

    calib_a, calib_b = float(a), float(b)

print(
    "Calibration params:",
    {
        "a": calib_a,
        "b": calib_b,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "n_train_used": len(train_df),
    },
)

test_case_ids, test_feats = _compute_features_for_path(test)
test_z = ((test_feats - feat_mean) / feat_std).astype(np.float32)
test_p = _safe_sigmoid(calib_a * test_z + calib_b).astype(np.float32)

test_p = np.where(np.isfinite(test_p), test_p, base_rate).astype(np.float32)
test_p = np.clip(test_p, 1e-6, 1 - 1e-6)

n_test = len(_list_numeric_case_dirs(test))
if len(test_p) != n_test:
    tmp = np.full((n_test,), base_rate, dtype=np.float32)
    tmp[: min(n_test, len(test_p))] = test_p[: min(n_test, len(test_p))]
    test_p = tmp

prediction_1 = test_p.copy()
prediction_2 = test_p.copy()
prediction_3 = test_p.copy()
prediction_4 = test_p.copy()
prediction_5 = test_p.copy()
prediction_6 = test_p.copy()

prediction_101 = test_p.copy()
prediction_102 = test_p.copy()
prediction_103 = test_p.copy()
prediction_104 = test_p.copy()
prediction_105 = test_p.copy()
prediction_106 = test_p.copy()

prediction_401 = test_p.copy()
prediction_402 = test_p.copy()
prediction_403 = test_p.copy()
prediction_404 = test_p.copy()
prediction_405 = test_p.copy()
prediction_406 = test_p.copy()

prediction_501 = test_p.copy()
prediction_502 = test_p.copy()
prediction_503 = test_p.copy()
prediction_504 = test_p.copy()
prediction_505 = test_p.copy()
prediction_506 = test_p.copy()

prediction_601 = test_p.copy()
prediction_602 = test_p.copy()
prediction_603 = test_p.copy()
prediction_604 = test_p.copy()
prediction_605 = test_p.copy()
prediction_606 = test_p.copy()

print("Computed non-constant test predictions. Mean pred:", float(np.mean(test_p)))




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
    p601,
    p602,
    p603,
    p604,
    p605,
    p606,
):
    path_cases = _list_numeric_case_dirs(path_test)
    cases = [int(os.path.basename(p)) for p in path_cases]
    n = len(cases)

    preds = (
        np.asarray(p1, dtype=np.float32)[:n]
        + np.asarray(p2, dtype=np.float32)[:n]
        + np.asarray(p3, dtype=np.float32)[:n]
        + np.asarray(p4, dtype=np.float32)[:n]
        + np.asarray(p5, dtype=np.float32)[:n]
        + np.asarray(p6, dtype=np.float32)[:n]
        + np.asarray(p101, dtype=np.float32)[:n]
        + np.asarray(p102, dtype=np.float32)[:n]
        + np.asarray(p103, dtype=np.float32)[:n]
        + np.asarray(p104, dtype=np.float32)[:n]
        + np.asarray(p105, dtype=np.float32)[:n]
        + np.asarray(p106, dtype=np.float32)[:n]
        + np.asarray(p401, dtype=np.float32)[:n]
        + np.asarray(p402, dtype=np.float32)[:n]
        + np.asarray(p403, dtype=np.float32)[:n]
        + np.asarray(p404, dtype=np.float32)[:n]
        + np.asarray(p405, dtype=np.float32)[:n]
        + np.asarray(p406, dtype=np.float32)[:n]
        + np.asarray(p501, dtype=np.float32)[:n]
        + np.asarray(p502, dtype=np.float32)[:n]
        + np.asarray(p503, dtype=np.float32)[:n]
        + np.asarray(p504, dtype=np.float32)[:n]
        + np.asarray(p505, dtype=np.float32)[:n]
        + np.asarray(p506, dtype=np.float32)[:n]
        + np.asarray(p601, dtype=np.float32)[:n]
        + np.asarray(p602, dtype=np.float32)[:n]
        + np.asarray(p603, dtype=np.float32)[:n]
        + np.asarray(p604, dtype=np.float32)[:n]
        + np.asarray(p605, dtype=np.float32)[:n]
        + np.asarray(p606, dtype=np.float32)[:n]
    ) / 30.0

    preds = np.clip(preds, 1e-6, 1 - 1e-6)

    df = pd.DataFrame({"BraTS21ID": cases, "MGMT_value": preds})
    df = df.sort_values("BraTS21ID").reset_index(drop=True)
    return df




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
    prediction_601,
    prediction_602,
    prediction_603,
    prediction_604,
    prediction_605,
    prediction_606,
)

sample_sub = pd.read_csv(sample_sub_path)
sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(int)

sub_df["BraTS21ID"] = sub_df["BraTS21ID"].astype(int)
sub_df = sample_sub[["BraTS21ID"]].merge(sub_df, on="BraTS21ID", how="left")

sub_df["MGMT_value"] = sub_df["MGMT_value"].fillna(base_rate).astype(float)
sub_df["MGMT_value"] = sub_df["MGMT_value"].clip(1e-6, 1 - 1e-6)

sub_df["BraTS21ID"] = sub_df["BraTS21ID"].map(lambda x: f"{int(x):05d}")

print(sub_df.head())
print("Submission rows:", len(sub_df))
print("Submission columns:", list(sub_df.columns))




## === cell 8
if sns is not None:
    try:
        sns.displot(sub_df.MGMT_value)
    except Exception as e:
        print("Plotting skipped:", repr(e))




## === cell 9
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with columns:", list(sub_df.columns))
print("Saved to:", os.path.abspath("submission.csv"))
