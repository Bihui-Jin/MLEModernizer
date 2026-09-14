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

os.environ.setdefault("PYTHONHASHSEED", "0")



## === cell 1
DATA_ROOT_CANDIDATES = [
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data",
    "/kaggle/input",
]
DATA_ROOT = None
for p in DATA_ROOT_CANDIDATES:
    if os.path.exists(p):
        if os.path.exists(os.path.join(p, "sample_submission.csv")):
            DATA_ROOT = p
            break
        if os.path.exists(
            os.path.join(p, "plant-pathology-2020-fgvc7", "sample_submission.csv")
        ):
            DATA_ROOT = os.path.join(p, "plant-pathology-2020-fgvc7")
            break
if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate competition data folder in expected Kaggle paths."
    )

print("Using DATA_ROOT:", DATA_ROOT)
print("Files in DATA_ROOT (first 20):", sorted(os.listdir(DATA_ROOT))[:20])



## === cell 2
sub = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))
test = pd.read_csv(os.path.join(DATA_ROOT, "test.csv"))
train = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"))

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]

assert all(c in sub.columns for c in ["image_id"] + TARGET_COLS)
assert (
    list(sub.columns) == ["image_id"] + TARGET_COLS
), "Submission column order mismatch vs sample_submission.csv"

print(
    "train shape:",
    train.shape,
    "test shape:",
    test.shape,
    "sample_submission shape:",
    sub.shape,
)
train.head()



## === cell 3
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

IMG_DIR_CANDIDATES = [
    os.path.join(DATA_ROOT, "images"),
    os.path.join(os.path.dirname(DATA_ROOT), "images"),
    "/kaggle/data/images",
    "/kaggle/input/images",
]
IMG_DIR = None
for d in IMG_DIR_CANDIDATES:
    if os.path.exists(d):
        IMG_DIR = d
        break
if IMG_DIR is None:
    raise FileNotFoundError("Could not locate images/ folder in expected paths.")
print("Using IMG_DIR:", IMG_DIR)

try:
    _RESAMPLE = Image.Resampling.BICUBIC
except Exception:
    _RESAMPLE = Image.BICUBIC


def _safe_open_rgb(path):
    with Image.open(path) as im:
        return im.convert("RGB")


def _rgb_to_lab_np(rgb01):
    """
    rgb01: float32 array in [0,1], shape (H,W,3), sRGB.
    returns Lab scaled roughly to [0,1] for each channel for stable stats.
    """
    rgb = np.clip(rgb01, 0.0, 1.0).astype(np.float32)

    a = 0.055
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + a) / (1.0 + a)) ** 2.4)

    M = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float32,
    )
    xyz = lin @ M  # (H,W,3)

    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    x = xyz[..., 0] / Xn
    y = xyz[..., 1] / Yn
    z = xyz[..., 2] / Zn

    eps = (6.0 / 29.0) ** 3
    kappa = (29.0 / 3.0) ** 3

    def f(t):
        return np.where(t > eps, np.cbrt(t), (kappa * t + 16.0) / 116.0)

    fx, fy, fz = f(x), f(y), f(z)

    L = 116.0 * fy - 16.0
    a_ = 500.0 * (fx - fy)
    b_ = 200.0 * (fy - fz)

    Ls = np.clip(L / 100.0, 0.0, 1.0)
    as_ = np.clip((a_ + 128.0) / 255.0, 0.0, 1.0)
    bs_ = np.clip((b_ + 128.0) / 255.0, 0.0, 1.0)

    return np.stack([Ls, as_, bs_], axis=-1).astype(np.float32)


def extract_features(
    image_id,
    size=96,
    gray_ds=24,
    rgb_ds=16,
    rgb_hist_bins=12,
    hsv_hist_bins=10,
):
    """
    Keep the same handcrafted-feature family (global stats + gradients + downsampled pixels + histograms).
    """
    path = os.path.join(IMG_DIR, f"{image_id}.jpg")
    im = _safe_open_rgb(path).resize((size, size), resample=_RESAMPLE)

    arr = np.asarray(im, dtype=np.float32) / 255.0  # (H,W,3) in [0,1]
    flat = arr.reshape(-1, 3)
    rgb_mean = flat.mean(axis=0)
    rgb_std = flat.std(axis=0)

    denom = (flat.sum(axis=1) + 1e-6).astype(np.float32)
    g_ratio = (flat[:, 1] / denom).astype(np.float32)
    g_ratio_mean = np.array([g_ratio.mean()], dtype=np.float32)
    g_ratio_std = np.array([g_ratio.std()], dtype=np.float32)

    exg = (2.0 * arr[:, :, 1] - arr[:, :, 0] - arr[:, :, 2]).astype(np.float32)
    exg_mean = np.array([exg.mean()], dtype=np.float32)
    exg_std = np.array([exg.std()], dtype=np.float32)

    hsv_im = im.convert("HSV")
    hsv = np.asarray(hsv_im, dtype=np.float32) / 255.0
    hsv_flat = hsv.reshape(-1, 3)
    hsv_mean = hsv_flat.mean(axis=0)
    hsv_std = hsv_flat.std(axis=0)

    lab = _rgb_to_lab_np(arr)
    lab_flat = lab.reshape(-1, 3)
    lab_mean = lab_flat.mean(axis=0)
    lab_std = lab_flat.std(axis=0)

    gray = (
        0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]
    ).astype(np.float32)
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    gx = np.pad(gx, ((0, 0), (0, 1)), mode="edge")
    gy = np.pad(gy, ((0, 1), (0, 0)), mode="edge")
    gmag = np.sqrt(gx * gx + gy * gy)
    g_mean = np.array([gmag.mean()], dtype=np.float32)
    g_std = np.array([gmag.std()], dtype=np.float32)

    im_g = im.convert("L").resize((gray_ds, gray_ds), resample=_RESAMPLE)
    gds = (np.asarray(im_g, dtype=np.float32) / 255.0).reshape(-1)

    im_rgb_ds = im.resize((rgb_ds, rgb_ds), resample=_RESAMPLE)
    rgbds = (np.asarray(im_rgb_ds, dtype=np.float32) / 255.0).reshape(-1)

    rgb_edges = np.linspace(0.0, 1.0, rgb_hist_bins + 1, dtype=np.float32)
    rgb_hists = []
    for ch in range(3):
        hist, _ = np.histogram(arr[:, :, ch], bins=rgb_edges)
        hist = hist.astype(np.float32)
        hist = hist / (hist.sum() + 1e-6)
        rgb_hists.append(hist)
    rgb_hist = np.concatenate(rgb_hists, axis=0)

    hsv_edges = np.linspace(0.0, 1.0, hsv_hist_bins + 1, dtype=np.float32)
    hsv_hists = []
    for ch in range(3):
        hist, _ = np.histogram(hsv[:, :, ch], bins=hsv_edges)
        hist = hist.astype(np.float32)
        hist = hist / (hist.sum() + 1e-6)
        hsv_hists.append(hist)
    hsv_hist = np.concatenate(hsv_hists, axis=0)

    return np.concatenate(
        [
            rgb_mean,
            rgb_std,
            g_ratio_mean,
            g_ratio_std,
            exg_mean,
            exg_std,
            hsv_mean,
            hsv_std,
            lab_mean,
            lab_std,
            g_mean,
            g_std,
            gds,
            rgbds,
            rgb_hist,
            hsv_hist,
        ],
        axis=0,
    )


train_ids = train["image_id"].tolist()
test_ids = test["image_id"].tolist()

from concurrent.futures import ThreadPoolExecutor

FEAT_PARAMS = dict(size=96, gray_ds=24, rgb_ds=16, rgb_hist_bins=12, hsv_hist_bins=10)
_feat_tag = "s{size}_g{gray_ds}_r{rgb_ds}_rh{rgb_hist_bins}_hh{hsv_hist_bins}".format(
    **FEAT_PARAMS
)
CACHE_DIR = "/kaggle/working"
TRAIN_CACHE = os.path.join(CACHE_DIR, f"X_train_{_feat_tag}.npy")
TEST_CACHE = os.path.join(CACHE_DIR, f"X_test_{_feat_tag}.npy")


def _extract_one(i):
    return extract_features(i, **FEAT_PARAMS)


def _build_matrix(ids, cache_path):
    if os.path.exists(cache_path):
        X = np.load(cache_path, mmap_mode=None)
        if X.shape[0] == len(ids):
            return X
    max_workers = min(32, (os.cpu_count() or 4))
    feats = [None] * len(ids)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for idx, f in enumerate(ex.map(_extract_one, ids, chunksize=32)):
            feats[idx] = f
    X = np.vstack(feats)
    np.save(cache_path, X)
    return X


X_train = _build_matrix(train_ids, TRAIN_CACHE)
X_test = _build_matrix(test_ids, TEST_CACHE)

print("X_train:", X_train.shape, "X_test:", X_test.shape)



## === cell 4
y_mat = train[TARGET_COLS].astype(int).values

y_strat = y_mat[:, 0] * 1 + y_mat[:, 1] * 2 + y_mat[:, 2] * 4 + y_mat[:, 3] * 8

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

test_pred = np.zeros((len(test_ids), len(TARGET_COLS)), dtype=np.float64)

fold_data = []
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_strat), 1):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train[tr_idx])
    X_va_s = scaler.transform(X_train[va_idx])
    X_te_s = scaler.transform(X_test)
    fold_data.append((tr_idx, va_idx, X_tr_s, X_va_s, X_te_s))

for t, col in enumerate(TARGET_COLS):
    y = y_mat[:, t]
    col_test_pred = np.zeros(len(test_ids), dtype=np.float64)
    oof = np.zeros(len(train_ids), dtype=np.float64)

    for fold, (tr_idx, va_idx, X_tr_s, X_va_s, X_te_s) in enumerate(fold_data, 1):
        y_tr = y[tr_idx]

        clf_kwargs = dict(
            C=3.5,
            penalty="l2",
            solver="saga",
            max_iter=4000,
            random_state=42,
            tol=1e-5,
        )
        try:
            clf = LogisticRegression(**clf_kwargs, n_jobs=-1)
        except TypeError:
            clf = LogisticRegression(**clf_kwargs)

        clf.fit(X_tr_s, y_tr)

        oof[va_idx] = clf.predict_proba(X_va_s)[:, 1]
        col_test_pred += clf.predict_proba(X_te_s)[:, 1] / skf.n_splits

    test_pred[:, t] = col_test_pred

test_pred = np.clip(test_pred, 0.0, 1.0)



## === cell 5
out = sub[["image_id"]].merge(test[["image_id"]], on="image_id", how="right")

assert (
    out["image_id"].tolist() == test["image_id"].tolist()
), "Output row order mismatch vs test.csv"

for j, col in enumerate(TARGET_COLS):
    out[col] = test_pred[:, j].astype(float)

out = out[["image_id"] + TARGET_COLS]
out.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", out.shape)
print(out.head())
print("Per-column mean preds:", {c: float(out[c].mean()) for c in TARGET_COLS})
