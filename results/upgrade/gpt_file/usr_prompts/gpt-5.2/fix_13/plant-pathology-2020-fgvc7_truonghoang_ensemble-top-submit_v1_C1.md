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
numpy==1.26.4
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

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

CANDIDATE_ROOTS = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "../input/plant-pathology-2020-fgvc7",
    "../data/plant-pathology-2020-fgvc7",
    "/kaggle/input",
    "/kaggle/data",
]


def find_competition_root():
    for r in CANDIDATE_ROOTS:
        if os.path.isdir(r):
            if os.path.isfile(os.path.join(r, "train.csv")) and os.path.isdir(
                os.path.join(r, "images")
            ):
                return r
            pp = os.path.join(r, "plant-pathology-2020-fgvc7")
            if os.path.isfile(os.path.join(pp, "train.csv")) and os.path.isdir(
                os.path.join(pp, "images")
            ):
                return pp
    raise FileNotFoundError(
        "Could not locate plant-pathology-2020-fgvc7 dataset folder in known locations."
    )


DATA_ROOT = find_competition_root()
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_ROOT, "sample_submission.csv")
IMG_DIR = os.path.join(DATA_ROOT, "images")

print("DATA_ROOT:", DATA_ROOT)
print(
    "Files:",
    [
        os.path.basename(TRAIN_CSV),
        os.path.basename(TEST_CSV),
        os.path.basename(SAMPLE_SUB_CSV),
    ],
)
print(
    "Images dir exists:",
    os.path.isdir(IMG_DIR),
    "n_files:",
    len(os.listdir(IMG_DIR)) if os.path.isdir(IMG_DIR) else 0,
)

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)

TARGETS = [c for c in sample_sub.columns if c != "image_id"]
assert TARGETS == [
    "healthy",
    "multiple_diseases",
    "rust",
    "scab",
], f"Unexpected target columns: {TARGETS}"

print("Train shape:", train_df.shape, "Test shape:", test_df.shape)
print("Targets:", TARGETS)



## === cell 1
from PIL import Image, ImageOps
from concurrent.futures import ThreadPoolExecutor


def _safe_open_image(path: str):
    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        try:
            img.draft("RGB", (256, 256))
        except Exception:
            pass
        img.load()
        return img
    except Exception:
        return None


FEAT_THUMB = 32  # keep feature core as-is
_STAT_SIZE = 256


def _radial_fft_features_from_gray(
    gray_arr_01: np.ndarray, n_bins: int = 16
) -> np.ndarray:
    g = gray_arr_01.astype(np.float32, copy=False)
    g = g - float(g.mean())

    F = np.fft.fft2(g)
    P = (np.abs(F) ** 2).astype(np.float32)
    P = np.fft.fftshift(P)

    h, w = P.shape
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    yy, xx = np.indices((h, w))
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.float32)
    rr = rr / (rr.max() + 1e-6)

    bin_idx = np.minimum((rr * n_bins).astype(np.int32), n_bins - 1).ravel()
    P_flat = P.ravel()

    sums = np.bincount(bin_idx, weights=P_flat, minlength=n_bins).astype(np.float32)
    cnts = np.bincount(bin_idx, minlength=n_bins).astype(np.float32)
    feats = sums / (cnts + 1e-12)

    feats = np.log1p(feats)
    s = float(feats.sum()) + 1e-6
    feats = feats / s
    return feats.astype(np.float32, copy=False)


def image_pixel_features(image_path: str) -> np.ndarray:
    FFT_BINS = 16
    total_dim = (
        3
        + 6
        + 6
        + 2
        + 2
        + (FEAT_THUMB * FEAT_THUMB)
        + (FEAT_THUMB * FEAT_THUMB)
        + 6
        + 6
        + FFT_BINS
    )

    img = _safe_open_image(image_path)
    if img is None:
        return np.zeros(total_dim, dtype=np.float32)

    w, h = img.size
    ar = float(w) / float(h + 1e-6)

    rgb = img.convert("RGB")

    if max(rgb.size) > _STAT_SIZE:
        rgb_stat = rgb.resize((_STAT_SIZE, _STAT_SIZE), resample=Image.BILINEAR)
    else:
        rgb_stat = rgb

    arr = np.asarray(rgb_stat, dtype=np.float32) / 255.0  # HxWx3
    flat = arr.reshape(-1, 3)
    ch_mean = flat.mean(axis=0)
    ch_std = flat.std(axis=0)

    denom = flat.sum(axis=1, keepdims=True) + 1e-6
    rgb_ratio = flat / denom
    rgb_ratio_mean = rgb_ratio.mean(axis=0)
    rgb_ratio_std = rgb_ratio.std(axis=0)

    hsv = rgb_stat.convert("HSV")
    harr = np.asarray(hsv, dtype=np.float32) / 255.0
    hflat = harr.reshape(-1, 3)
    hsv_mean = hflat.mean(axis=0)
    hsv_std = hflat.std(axis=0)

    hden = hflat.sum(axis=1, keepdims=True) + 1e-6
    hsv_ratio = hflat / hden
    hsv_ratio_mean = hsv_ratio.mean(axis=0)
    hsv_ratio_std = hsv_ratio.std(axis=0)

    gray_stat = rgb_stat.convert("L")
    garr = np.asarray(gray_stat, dtype=np.float32) / 255.0
    g_mean = float(garr.mean())
    g_std = float(garr.std())

    dx = np.diff(garr, axis=1)
    dy = np.diff(garr, axis=0)
    dx2 = dx[:-1, :]
    dy2 = dy[:, :-1]
    mag = np.sqrt(dx2 * dx2 + dy2 * dy2)
    e_mean = float(mag.mean())
    e_std = float(mag.std())

    gray = rgb.convert("L")
    thumb = gray.resize((FEAT_THUMB, FEAT_THUMB), resample=Image.BILINEAR)
    t = (np.asarray(thumb, dtype=np.float32) / 255.0).reshape(-1).astype(np.float32)

    t_mean = float(t.mean())
    t_std = float(t.std())
    t_norm = (t - t_mean) / (t_std + 1e-6)

    fft_feats = _radial_fft_features_from_gray(garr, n_bins=FFT_BINS)

    meta = np.array([float(w), float(h), ar], dtype=np.float32)
    extra = np.array([g_mean, g_std, e_mean, e_std], dtype=np.float32)

    feats = np.concatenate(
        [
            meta,
            ch_mean.astype(np.float32),
            ch_std.astype(np.float32),
            hsv_mean.astype(np.float32),
            hsv_std.astype(np.float32),
            extra,
            t,
            t_norm.astype(np.float32),
            rgb_ratio_mean.astype(np.float32),
            rgb_ratio_std.astype(np.float32),
            hsv_ratio_mean.astype(np.float32),
            hsv_ratio_std.astype(np.float32),
            fft_feats.astype(np.float32),
        ],
        axis=0,
    )

    if feats.shape[0] != total_dim:
        out = np.zeros(total_dim, dtype=np.float32)
        n = min(total_dim, feats.shape[0])
        out[:n] = feats[:n]
        return out
    return feats.astype(np.float32)


def _resolve_image_path(iid: str) -> str:
    p = os.path.join(IMG_DIR, f"{iid}.jpg")
    if os.path.isfile(p):
        return p
    p2 = os.path.join(IMG_DIR, f"{iid}.JPG")
    if os.path.isfile(p2):
        return p2
    return ""


def build_feature_matrix(image_ids: pd.Series) -> np.ndarray:
    FFT_BINS = 16
    total_dim = (
        3
        + 6
        + 6
        + 2
        + 2
        + (FEAT_THUMB * FEAT_THUMB)
        + (FEAT_THUMB * FEAT_THUMB)
        + 6
        + 6
        + FFT_BINS
    )

    ids = image_ids.tolist()
    paths = [_resolve_image_path(iid) for iid in ids]

    missing = sum(1 for p in paths if p == "")
    if missing:
        print(f"Warning: missing {missing} images; filled features with zeros.")

    X = np.zeros((len(paths), total_dim), dtype=np.float32)

    def _feat_from_path(p: str) -> np.ndarray:
        if not p:
            return None
        return image_pixel_features(p)

    max_workers = min(32, (os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, feat in enumerate(ex.map(_feat_from_path, paths, chunksize=64)):
            if feat is not None:
                X[i, :] = feat

    return X


X_train = build_feature_matrix(train_df["image_id"])
X_test = build_feature_matrix(test_df["image_id"])

print("X_train:", X_train.shape, "X_test:", X_test.shape)



## === cell 2
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

Y_train = train_df[TARGETS].values.astype(np.int32)

proba_test = np.zeros((len(test_df), len(TARGETS)), dtype=np.float32)

C_GRID = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]
N_SPLITS = 5

_cpu = os.cpu_count() or 4

global_scaler = StandardScaler(with_mean=True, with_std=True)
X_train_s = np.ascontiguousarray(
    global_scaler.fit_transform(X_train).astype(np.float32, copy=False)
)
X_test_s = np.ascontiguousarray(
    global_scaler.transform(X_test).astype(np.float32, copy=False)
)

SOLVER_GRID = ["lbfgs", "liblinear"]

for j, tgt in enumerate(TARGETS):
    y = Y_train[:, j]
    if y.min() == y.max():
        const = float(y.mean())
        proba_test[:, j] = const
        print(f"{tgt}: degenerate labels; using constant {const:.6f}")
        continue

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    best_params = None
    best_auc = -1.0

    fold_indices = list(skf.split(X_train_s, y))

    for solver in SOLVER_GRID:
        n_jobs = _cpu if solver != "liblinear" else None
        for C in C_GRID:
            fold_aucs = []

            for tr_idx, va_idx in fold_indices:
                lr = LogisticRegression(
                    solver=solver,
                    penalty="l2",
                    max_iter=3000,
                    C=C,
                    random_state=RANDOM_STATE,
                    n_jobs=n_jobs,
                    fit_intercept=True,
                    tol=1e-5,
                )
                lr.fit(X_train_s[tr_idx], y[tr_idx])
                va_proba = lr.predict_proba(X_train_s[va_idx])[:, 1].astype(
                    np.float32, copy=False
                )
                fold_aucs.append(roc_auc_score(y[va_idx], va_proba))

            auc = float(np.mean(fold_aucs))
            if auc > best_auc:
                best_auc = auc
                best_params = (solver, C)

    best_solver, best_C = best_params

    final_lr = LogisticRegression(
        solver=best_solver,
        penalty="l2",
        max_iter=3000,
        C=best_C,
        random_state=RANDOM_STATE,
        n_jobs=_cpu if best_solver != "liblinear" else None,
        fit_intercept=True,
        tol=1e-5,
    )
    final_lr.fit(X_train_s, y)

    proba_test[:, j] = final_lr.predict_proba(X_test_s)[:, 1].astype(
        np.float32, copy=False
    )

    print(
        f"{tgt}: best_solver={best_solver} best_C={best_C} cv_auc={best_auc:.5f}, "
        f"test proba range [{proba_test[:, j].min():.4f}, {proba_test[:, j].max():.4f}]"
    )



## === cell 3
sub = pd.DataFrame({"image_id": test_df["image_id"].values})
sub[TARGETS] = np.clip(proba_test, 0.0, 1.0)

assert (
    sub.shape[0] == test_df.shape[0]
), "Submission row count must match test row count."
assert (
    list(sub.columns) == ["image_id"] + TARGETS
), f"Submission columns mismatch: {sub.columns.tolist()}"

out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path, "shape:", sub.shape)
print(sub.head())
