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
import numpy as np
import pandas as pd
import os

os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(0)



## === cell 1
DATA_DIR = "/kaggle/input/plant-pathology-2020-fgvc7"
if not os.path.exists(DATA_DIR):
    DATA_DIR = "/kaggle/data/plant-pathology-2020-fgvc7"

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)

target_cols = [c for c in sample_sub.columns if c != "image_id"]
assert "image_id" in sample_sub.columns, "sample_submission must contain image_id"
assert set(target_cols).issubset(
    set(train_df.columns)
), "Train is missing one or more target columns"

train_df.head(), test_df.head(), sample_sub.head()



## === cell 2
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from PIL import Image

images_dir = os.path.join(DATA_DIR, "images")
if not os.path.isdir(images_dir):
    images_dir = os.path.join(DATA_DIR, "plant-pathology-2020-fgvc7", "images")


def _image_path(image_id: str) -> str:
    return os.path.join(images_dir, f"{image_id}.jpg")


_RESIZE_WH = (224, 224)
_NBINS = 8
_THUMB_WH = (32, 32)  # 32*32*3 = 3072 dims

try:
    _BILINEAR = Image.Resampling.BILINEAR  # Pillow>=9
except Exception:
    _BILINEAR = Image.BILINEAR


def _lbp_hist(gray01: np.ndarray) -> np.ndarray:
    """8-neighbor LBP on a small downsampled grayscale image, returned as a normalized 256-bin histogram."""
    g = gray01[::4, ::4]  # ~56x56 if input is 224x224
    g = g.astype(np.float32, copy=False)

    c = g[1:-1, 1:-1]
    code = np.zeros_like(c, dtype=np.uint8)

    code |= ((g[:-2, :-2] >= c) << 7).astype(np.uint8)
    code |= ((g[:-2, 1:-1] >= c) << 6).astype(np.uint8)
    code |= ((g[:-2, 2:] >= c) << 5).astype(np.uint8)
    code |= ((g[1:-1, 2:] >= c) << 4).astype(np.uint8)
    code |= ((g[2:, 2:] >= c) << 3).astype(np.uint8)
    code |= ((g[2:, 1:-1] >= c) << 2).astype(np.uint8)
    code |= ((g[2:, :-2] >= c) << 1).astype(np.uint8)
    code |= ((g[1:-1, :-2] >= c) << 0).astype(np.uint8)

    h = np.bincount(code.ravel(), minlength=256).astype(np.float32)
    h /= h.sum() + 1e-6
    return h


def _grad_stats(gray01: np.ndarray) -> np.ndarray:
    """Finite-difference gradient magnitude stats on downsampled grayscale."""
    g = gray01[::2, ::2].astype(np.float32, copy=False)  # ~112x112
    dx = g[:, 1:] - g[:, :-1]
    dy = g[1:, :] - g[:-1, :]
    dx2 = dx[:-1, :]
    dy2 = dy[:, :-1]
    mag = np.sqrt(dx2 * dx2 + dy2 * dy2).ravel()

    if mag.size == 0:
        return np.zeros(4, dtype=np.float32)

    mean = float(mag.mean())
    std = float(mag.std())
    p90 = float(np.quantile(mag, 0.90))
    p99 = float(np.quantile(mag, 0.99))
    return np.array([mean, std, p90, p99], dtype=np.float32)


def _hog_u8(gray01: np.ndarray, cell_size: int = 8, nbins: int = 9) -> np.ndarray:
    g = (gray01 * 255.0).clip(0.0, 255.0).astype(np.float32, copy=False)

    gx = g[1:-1, 2:] - g[1:-1, :-2]
    gy = g[2:, 1:-1] - g[:-2, 1:-1]

    mag = np.sqrt(gx * gx + gy * gy)
    ang = np.arctan2(gy, gx)
    ang = np.mod(ang, np.pi)  # [0, pi)

    H, W = mag.shape
    Hc = (H // cell_size) * cell_size
    Wc = (W // cell_size) * cell_size
    if Hc <= 0 or Wc <= 0:
        return np.zeros((nbins,), dtype=np.float32)

    mag = mag[:Hc, :Wc]
    ang = ang[:Hc, :Wc]

    bin_f = (ang * (nbins / np.pi)).astype(np.int32, copy=False)
    np.minimum(bin_f, nbins - 1, out=bin_f)

    ncy = Hc // cell_size
    ncx = Wc // cell_size

    mag_r = mag.reshape(-1).astype(np.float32, copy=False)
    bin_r = bin_f.reshape(-1)

    y = np.repeat(np.arange(Hc, dtype=np.int32), Wc)
    x = np.tile(np.arange(Wc, dtype=np.int32), Hc)
    cy = y // cell_size
    cx = x // cell_size

    lin = (cy * ncx + cx) * nbins + bin_r
    hog_flat = np.bincount(lin, weights=mag_r, minlength=ncy * ncx * nbins).astype(
        np.float32, copy=False
    )
    hog_cell = hog_flat.reshape(ncy, ncx, nbins)

    if ncy >= 2 and ncx >= 2:
        blocks = np.stack(
            [
                hog_cell[:-1, :-1, :],
                hog_cell[:-1, 1:, :],
                hog_cell[1:, :-1, :],
                hog_cell[1:, 1:, :],
            ],
            axis=2,
        ).reshape((ncy - 1) * (ncx - 1), -1)
        norms = np.linalg.norm(blocks, axis=1, keepdims=True) + 1e-6
        blocks = (blocks / norms).astype(np.float32, copy=False)
        return blocks.reshape(-1).astype(np.float32, copy=False)
    else:
        v = hog_cell.reshape(-1).astype(np.float32, copy=False)
        v = v / (np.linalg.norm(v) + 1e-6)
        return v.astype(np.float32, copy=False)


def extract_features(image_id: str) -> np.ndarray:
    p = _image_path(image_id)
    with Image.open(p) as img:
        img = img.convert("RGB")
        img = img.resize(_RESIZE_WH, resample=_BILINEAR)
        thumb_img = img.resize(_THUMB_WH, resample=_BILINEAR)

        arr = np.asarray(img, dtype=np.float32) * (1.0 / 255.0)  # [H,W,3] in [0,1]
        thumb01 = (np.asarray(thumb_img, dtype=np.float32) * (1.0 / 255.0)).reshape(-1)

    arr = np.ascontiguousarray(arr)
    pix = arr.reshape(-1, 3)

    mean_rgb = pix.mean(axis=0)
    std_rgb = pix.std(axis=0)

    brightness = float(mean_rgb.mean())
    gr_ratio = float((mean_rgb[1] - mean_rgb[0]) / (mean_rgb[1] + mean_rgb[0] + 1e-6))

    q = (pix * _NBINS).astype(np.int32)
    np.minimum(q, _NBINS - 1, out=q)

    h0 = np.bincount(q[:, 0], minlength=_NBINS).astype(np.float32)
    h1 = np.bincount(q[:, 1], minlength=_NBINS).astype(np.float32)
    h2 = np.bincount(q[:, 2], minlength=_NBINS).astype(np.float32)

    h0 /= h0.sum() + 1e-6
    h1 /= h1.sum() + 1e-6
    h2 /= h2.sum() + 1e-6

    hist_feat = np.concatenate([h0, h1, h2], axis=0).astype(np.float32)  # 24 dims

    gray = (
        0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]
    ).astype(np.float32, copy=False)
    grad_feat = _grad_stats(gray)  # 4 dims
    lbp_feat = _lbp_hist(gray)  # 256 dims
    hog_feat = _hog_u8(gray, cell_size=8, nbins=9)

    thumb_norm = float(np.linalg.norm(thumb01))
    if thumb_norm > 0:
        thumb01 = thumb01 / thumb_norm

    feats = np.concatenate(
        [
            mean_rgb.astype(np.float32),
            std_rgb.astype(np.float32),
            np.array([brightness, gr_ratio], dtype=np.float32),
            hist_feat,
            grad_feat,
            lbp_feat,
            hog_feat,
            thumb01.astype(np.float32),
        ],
        axis=0,
    ).astype(np.float32, copy=False)

    return feats


train_ids = train_df["image_id"].to_numpy()
test_ids = test_df["image_id"].to_numpy()
all_ids = np.concatenate([train_ids, test_ids])

_first = extract_features(all_ids[0])
feat_dim = int(_first.shape[0])


def _extract_pair(idx_and_id):
    i, image_id = idx_and_id
    return i, extract_features(image_id)


X_all = np.empty((len(all_ids), feat_dim), dtype=np.float32)
X_all[0] = _first

try:
    import multiprocessing as mp

    n = len(all_ids)
    if n > 1:
        start = 1
        items = [(i, all_ids[i]) for i in range(start, n)]

        try:
            ctx = mp.get_context("fork")
        except ValueError:
            ctx = mp.get_context("spawn")

        workers = min(8, (os.cpu_count() or 2))
        chunksize = max(64, (n - start) // (workers * 2) + 1)

        with ctx.Pool(processes=workers) as pool:
            for i, feat in pool.imap_unordered(
                _extract_pair, items, chunksize=chunksize
            ):
                X_all[i] = feat
except Exception:
    for k in range(1, len(all_ids)):
        X_all[k] = extract_features(all_ids[k])

X_train = X_all[: len(train_df)]
X_test = X_all[len(train_df) :]

preds = np.zeros((len(test_df), len(target_cols)), dtype=np.float64)

for j, c in enumerate(target_cols):
    y = train_df[c].values.astype(int)

    if y.min() == y.max():
        preds[:, j] = float(y.mean())
        continue

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    solver="saga",
                    penalty="elasticnet",
                    l1_ratio=0.05,
                    max_iter=1200,
                    C=1.5,
                    class_weight="balanced",
                    random_state=0,
                    n_jobs=1,
                ),
            ),
        ]
    )
    clf.fit(X_train, y)
    preds[:, j] = clf.predict_proba(X_test)[:, 1]

sub = sample_sub.copy()
sub = sub.merge(
    test_df[["image_id"]], on="image_id", how="right", validate="one_to_one"
)
for j, c in enumerate(target_cols):
    sub[c] = preds[:, j].astype(float)

sub = sub[["image_id"] + target_cols]
for c in target_cols:
    sub[c] = sub[c].clip(0.0, 1.0)

sub.to_csv("submission.csv", index=False)
sub.head()



## === cell 3
print("Wrote submission.csv")
print("Shape:", sub.shape)
print("Columns:", list(sub.columns))
print(sub.describe(include="all"))
