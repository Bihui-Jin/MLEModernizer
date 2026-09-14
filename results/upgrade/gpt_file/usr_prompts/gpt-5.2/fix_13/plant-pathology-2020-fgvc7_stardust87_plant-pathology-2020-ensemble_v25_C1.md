# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Detect apple diseases from images.

## Metric
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

from PIL import Image

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.calibration import _sigmoid_calibration

from joblib import Parallel, delayed

os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(0)



## === cell 1
CANDIDATE_DATA_DIRS = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/input",
    "/kaggle/data",
]

DATA_DIR = None
for d in CANDIDATE_DATA_DIRS:
    if os.path.isfile(os.path.join(d, "train.csv")) and os.path.isdir(
        os.path.join(d, "images")
    ):
        DATA_DIR = d
        break

if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not find dataset directory containing train.csv and images/. "
        f"Tried: {CANDIDATE_DATA_DIRS}"
    )

SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_CSV_PATH = os.path.join(DATA_DIR, "test.csv")
IMAGES_DIR = os.path.join(DATA_DIR, "images")

SUBMISSIONS_PATH = "/kaggle/input/submissions/"

print("Using DATA_DIR:", DATA_DIR)
print("Sample submission exists:", os.path.isfile(SAMPLE_SUB_PATH))
print("Images dir exists:", os.path.isdir(IMAGES_DIR))



## === cell 2
submissions_all = []
if os.path.isdir(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                submissions_all.append(os.path.join(dirname, filename))
submissions_all.sort()
print("Found external submissions:", submissions_all)




## === cell 3
def ensemble(submissions_all, sub_idx, weights=[]):
    if len(sub_idx) == 0:
        raise ValueError("sub_idx is empty; nothing to ensemble.")
    if len(weights) != len(sub_idx):
        raise ValueError(
            f"weights length ({len(weights)}) must match sub_idx length ({len(sub_idx)})."
        )
    if len(submissions_all) == 0:
        raise ValueError(
            "submissions_all is empty; no submission files found to ensemble."
        )
    for j in sub_idx:
        if j < 0 or j >= len(submissions_all):
            raise IndexError(
                f"Requested submission index {j} out of range for {len(submissions_all)} files."
            )

    submission_sum = None
    for i in range(len(sub_idx)):
        path = submissions_all[sub_idx[i]]
        w = weights[i]
        print(f"I'm taking submission {path} with weight {w}")
        arr = pd.read_csv(
            path, usecols=["healthy", "multiple_diseases", "rust", "scab"]
        ).to_numpy(dtype=np.float64)
        if submission_sum is None:
            submission_sum = w * arr
        else:
            submission_sum += w * arr
    return submission_sum.astype(np.float32)




## === cell 4
def make_submission_file(submission_avg, image_ids=None, submissions_all=None):
    submission_df = pd.read_csv(SAMPLE_SUB_PATH)
    if image_ids is None:
        submission_df[["healthy", "multiple_diseases", "rust", "scab"]] = submission_avg
    else:
        pred_df = pd.DataFrame(
            submission_avg,
            columns=["healthy", "multiple_diseases", "rust", "scab"],
        )
        pred_df["image_id"] = image_ids

        submission_df = submission_df[["image_id"]].merge(
            pred_df, on="image_id", how="left"
        )

        if (
            submission_df[["healthy", "multiple_diseases", "rust", "scab"]]
            .isna()
            .any()
            .any()
        ):
            missing = submission_df.loc[
                submission_df[["healthy", "multiple_diseases", "rust", "scab"]]
                .isna()
                .any(axis=1),
                "image_id",
            ].tolist()
            raise ValueError(
                f"Missing predictions for {len(missing)} image_id(s), e.g. {missing[:5]}"
            )
    submission_df.to_csv("submission.csv", index=False)




## === cell 5
TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]

_RGB2GRAY = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)

_FEATURE_CACHE = {}


def _channel_hist(arr, bins=16):
    out = np.empty((3 * bins,), dtype=np.float32)
    for c in range(3):
        hist, _ = np.histogram(arr[..., c], bins=bins, range=(0.0, 1.0), density=False)
        hist = hist.astype(np.float32, copy=False)
        hist = hist / (hist.sum() + 1e-8)
        out[c * bins : (c + 1) * bins] = hist
    return out


def _edge_features(arr, edge_bins=16):
    gray = (arr @ _RGB2GRAY).astype(np.float32, copy=False)
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)

    mag = np.sqrt(gx[:-1, :] * gx[:-1, :] + gy[:, :-1] * gy[:, :-1]).astype(np.float32)

    mag_mean = float(mag.mean())
    mag_std = float(mag.std())
    mag_p90 = float(np.quantile(mag, 0.90))

    mag_cap = np.clip(mag, 0.0, 0.5)
    hist, _ = np.histogram(mag_cap, bins=edge_bins, range=(0.0, 0.5), density=False)
    hist = hist.astype(np.float32, copy=False)
    hist = hist / (hist.sum() + 1e-8)

    return np.concatenate(
        [np.array([mag_mean, mag_std, mag_p90], dtype=np.float32), hist], axis=0
    )


def _hog_like(gray, cell_size=16, n_bins=9):
    gx = np.diff(gray, axis=1).astype(np.float32)  # (H, W-1)
    gy = np.diff(gray, axis=0).astype(np.float32)  # (H-1, W)

    gx = gx[:-1, :]
    gy = gy[:, :-1]

    mag = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    ang = (np.degrees(np.arctan2(gy, gx)).astype(np.float32) + 180.0) % 180.0  # [0,180)

    H, W = mag.shape
    n_cells_y = max(1, H // cell_size)
    n_cells_x = max(1, W // cell_size)

    Hc = n_cells_y * cell_size
    Wc = n_cells_x * cell_size
    mag = mag[:Hc, :Wc]
    ang = ang[:Hc, :Wc]

    mag_c = mag.reshape(n_cells_y, cell_size, n_cells_x, cell_size).transpose(
        0, 2, 1, 3
    )
    ang_c = ang.reshape(n_cells_y, cell_size, n_cells_x, cell_size).transpose(
        0, 2, 1, 3
    )

    bin_idx = np.floor((ang_c * n_bins) / 180.0).astype(np.int32)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    n_cells = n_cells_y * n_cells_x
    pix = cell_size * cell_size
    b = bin_idx.reshape(n_cells, pix)
    w = mag_c.reshape(n_cells, pix)

    cell_ids = np.repeat(np.arange(n_cells, dtype=np.int32), pix)
    combined = cell_ids * n_bins + b.ravel()
    hist_flat = np.bincount(
        combined, weights=w.ravel(), minlength=n_cells * n_bins
    ).astype(np.float32, copy=False)
    hists = hist_flat.reshape(n_cells, n_bins)

    denom = hists.sum(axis=1, keepdims=True) + 1e-8
    hists = hists / denom

    return hists.ravel().astype(np.float32, copy=False)


def _lbp_hist(gray):
    g = gray.astype(np.float32, copy=False)
    center = g[1:-1, 1:-1]
    codes = np.zeros_like(center, dtype=np.uint8)

    neighbors = [
        g[:-2, :-2],
        g[:-2, 1:-1],
        g[:-2, 2:],
        g[1:-1, 2:],
        g[2:, 2:],
        g[2:, 1:-1],
        g[2:, :-2],
        g[1:-1, :-2],
    ]
    for i, n in enumerate(neighbors):
        codes |= (n >= center).astype(np.uint8) << i

    hist = np.bincount(codes.ravel(), minlength=256).astype(np.float32, copy=False)
    hist = hist / (hist.sum() + 1e-8)
    return hist


_HOG_DIMS_CACHE = {}


def _feature_dim_for_scale(img_size, hist_bins, edge_bins, hog_cell_size, hog_bins):
    key = (img_size, hist_bins, edge_bins, hog_cell_size, hog_bins)
    if key in _HOG_DIMS_CACHE:
        return _HOG_DIMS_CACHE[key]
    dummy = np.zeros((img_size[0], img_size[1]), dtype=np.float32)
    hog_dim = _hog_like(dummy, cell_size=hog_cell_size, n_bins=hog_bins).shape[0]
    n_feat = 14 + 3 * hist_bins + (3 + edge_bins) + hog_dim + 256
    _HOG_DIMS_CACHE[key] = (hog_dim, n_feat)
    return hog_dim, n_feat


def _extract_features_for_image_id(
    image_id,
    img_size1=(128, 128),
    img_size2=(96, 96),
    hist_bins=16,
    edge_bins=16,
    hog_cell_size=16,
    hog_bins=9,
):
    cache_key = (
        image_id,
        img_size1,
        img_size2,
        hist_bins,
        edge_bins,
        hog_cell_size,
        hog_bins,
    )
    if cache_key in _FEATURE_CACHE:
        return _FEATURE_CACHE[cache_key]

    img_path = os.path.join(IMAGES_DIR, f"{image_id}.jpg")
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        im1 = im.resize(img_size1, resample=Image.BILINEAR)
        im2 = im.resize(img_size2, resample=Image.BILINEAR)

    arr1 = np.asarray(im1, dtype=np.float32) / 255.0
    arr2 = np.asarray(im2, dtype=np.float32) / 255.0

    def _single(arr):
        flat = arr.reshape(-1, 3)
        ch_means = flat.mean(axis=0)
        ch_stds = flat.std(axis=0)
        ch_mins = flat.min(axis=0)
        ch_maxs = flat.max(axis=0)
        overall_mean = arr.mean()
        overall_std = arr.std()
        base = np.concatenate(
            [ch_means, ch_stds, ch_mins, ch_maxs, [overall_mean, overall_std]]
        ).astype(np.float32, copy=False)

        hist = _channel_hist(arr, bins=hist_bins)
        edge = _edge_features(arr, edge_bins=edge_bins)

        gray = (arr @ _RGB2GRAY).astype(np.float32, copy=False)
        hog = _hog_like(gray, cell_size=hog_cell_size, n_bins=hog_bins)
        lbp = _lbp_hist(gray)

        return np.concatenate([base, hist, edge, hog, lbp]).astype(
            np.float32, copy=False
        )

    f1 = _single(arr1)
    f2 = _single(arr2)
    out = np.concatenate([f1, f2], axis=0).astype(np.float32, copy=False)
    _FEATURE_CACHE[cache_key] = out
    return out


def extract_features(
    image_ids,
    img_size=(128, 128),
    hist_bins=16,
    edge_bins=16,
    hog_cell_size=16,
    hog_bins=9,
):
    img_size1 = img_size
    img_size2 = (96, 96)

    _, n_feat1 = _feature_dim_for_scale(
        img_size1, hist_bins, edge_bins, hog_cell_size, hog_bins
    )
    _, n_feat2 = _feature_dim_for_scale(
        img_size2, hist_bins, edge_bins, hog_cell_size, hog_bins
    )
    n_feat = n_feat1 + n_feat2

    image_ids = list(image_ids)
    n = len(image_ids)

    n_jobs = min(8, (os.cpu_count() or 2))
    feats_list = Parallel(n_jobs=n_jobs, prefer="threads", batch_size=32)(
        delayed(_extract_features_for_image_id)(
            image_id,
            img_size1=img_size1,
            img_size2=img_size2,
            hist_bins=hist_bins,
            edge_bins=edge_bins,
            hog_cell_size=hog_cell_size,
            hog_bins=hog_bins,
        )
        for image_id in image_ids
    )

    feats = np.vstack(feats_list).astype(np.float32, copy=False)
    if feats.shape != (n, n_feat):
        raise ValueError(
            f"Feature shape mismatch: got {feats.shape}, expected {(n, n_feat)}"
        )
    return feats


def _choose_cv_for_binary_label(y_bin, n_splits=5, random_state=0):
    y_bin = np.asarray(y_bin).astype(int)
    pos = int(y_bin.sum())
    neg = int((1 - y_bin).sum())
    if pos == 0 or neg == 0:
        return KFold(n_splits=2, shuffle=True, random_state=random_state)
    max_splits = min(n_splits, pos, neg)
    if max_splits < 2:
        return KFold(n_splits=2, shuffle=True, random_state=random_state)
    return StratifiedKFold(n_splits=max_splits, shuffle=True, random_state=random_state)


def _oof_sigmoid_calibrated_predict_proba(
    X_train_s, y_bin, X_test_s, base_lr, n_splits=5, random_state=0
):
    y_bin = np.asarray(y_bin).astype(int)
    n_train = X_train_s.shape[0]
    oof_scores = np.zeros(n_train, dtype=np.float64)

    cv = _choose_cv_for_binary_label(
        y_bin, n_splits=n_splits, random_state=random_state
    )

    for tr_idx, va_idx in cv.split(
        np.zeros(n_train), y_bin if isinstance(cv, StratifiedKFold) else None
    ):
        lr_fold = LogisticRegression(
            max_iter=base_lr.max_iter,
            solver=base_lr.solver,
            C=base_lr.C,
            random_state=base_lr.random_state,
            n_jobs=base_lr.n_jobs,
            class_weight=base_lr.class_weight,
        )
        lr_fold.fit(X_train_s[tr_idx], y_bin[tr_idx])
        oof_scores[va_idx] = lr_fold.predict_proba(X_train_s[va_idx])[:, 1]

    a, b = _sigmoid_calibration(oof_scores, y_bin)

    def _sigmoid(x):
        x = np.asarray(x, dtype=np.float64)
        return 1.0 / (1.0 + np.exp(a * x + b))

    lr_full = LogisticRegression(
        max_iter=base_lr.max_iter,
        solver=base_lr.solver,
        C=base_lr.C,
        random_state=base_lr.random_state,
        n_jobs=base_lr.n_jobs,
        class_weight=base_lr.class_weight,
    )
    lr_full.fit(X_train_s, y_bin)
    test_scores = lr_full.predict_proba(X_test_s)[:, 1].astype(np.float64)
    test_proba = _sigmoid(test_scores)
    return test_proba.astype(np.float32)


def train_and_predict_baseline():
    train_df = pd.read_csv(TRAIN_CSV_PATH, usecols=["image_id"] + TARGET_COLS)
    test_df = pd.read_csv(TEST_CSV_PATH, usecols=["image_id"])

    all_image_ids = np.concatenate(
        [train_df["image_id"].values, test_df["image_id"].values], axis=0
    )
    X_all = extract_features(
        all_image_ids,
        img_size=(128, 128),
        hist_bins=16,
        edge_bins=16,
        hog_cell_size=16,
        hog_bins=9,
    )
    n_train = train_df.shape[0]
    X_train = X_all[:n_train]
    X_test = X_all[n_train:]

    y_train = train_df[TARGET_COLS].values.astype(int)

    base_lr = LogisticRegression(
        max_iter=4000,
        solver="saga",
        C=3.0,
        random_state=0,
        n_jobs=-1,
        class_weight="balanced",
    )

    scaler = StandardScaler(with_mean=False)
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    proba = np.zeros((X_test_s.shape[0], len(TARGET_COLS)), dtype=np.float32)
    for k, col in enumerate(TARGET_COLS):
        yk = y_train[:, k].astype(int)
        proba[:, k] = _oof_sigmoid_calibrated_predict_proba(
            X_train_s,
            yk,
            X_test_s,
            base_lr=base_lr,
            n_splits=5,
            random_state=0,
        )

    proba = np.clip(proba, 0.0, 1.0)

    if proba.shape[1] != 4:
        raise ValueError(f"Expected 4 target columns, got proba shape {proba.shape}")

    return test_df["image_id"].values, proba




## === cell 6
if len(submissions_all) >= 3:
    submission_avg = ensemble(submissions_all, [1, 2], [0.5, 0.5])
    make_submission_file(
        submission_avg, image_ids=None, submissions_all=submissions_all
    )
else:
    test_image_ids, submission_avg = train_and_predict_baseline()
    make_submission_file(
        submission_avg, image_ids=test_image_ids, submissions_all=submissions_all
    )

out = pd.read_csv("submission.csv")
print("Wrote submission.csv with shape:", out.shape)
print(out.head())
print("Submission columns:", list(out.columns))
print("Min/max per target:\n", out[TARGET_COLS].min(), "\n", out[TARGET_COLS].max())
