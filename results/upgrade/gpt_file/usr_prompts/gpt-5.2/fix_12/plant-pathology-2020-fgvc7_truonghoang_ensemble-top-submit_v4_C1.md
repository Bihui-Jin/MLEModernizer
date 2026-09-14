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



## === cell 1
BASE_INPUT = "/kaggle/input"

CANDIDATES = [
    os.path.join("/kaggle", "data", "plant-pathology-2020-fgvc7"),
    os.path.join("/kaggle", "data"),
    os.path.join(BASE_INPUT, "plant-pathology-2020-fgvc7"),
    BASE_INPUT,
]

COMP_DIR = None
for c in CANDIDATES:
    if os.path.exists(os.path.join(c, "train.csv")) and os.path.exists(
        os.path.join(c, "test.csv")
    ):
        COMP_DIR = c
        break

if COMP_DIR is None:
    raise FileNotFoundError(
        "Could not find train.csv/test.csv under expected Kaggle input/data directories."
    )

train_path = os.path.join(COMP_DIR, "train.csv")
test_path = os.path.join(COMP_DIR, "test.csv")
sample_path = os.path.join(COMP_DIR, "sample_submission.csv")

images_dir = os.path.join(COMP_DIR, "images")
if not os.path.isdir(images_dir):
    alt_images_dir = os.path.join("/kaggle", "data", "images")
    if os.path.isdir(alt_images_dir):
        images_dir = alt_images_dir

for p in [train_path, test_path, sample_path]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Required file not found: {p}")

if not os.path.isdir(images_dir):
    raise FileNotFoundError(f"Images directory not found: {images_dir}")

print("Using COMP_DIR =", COMP_DIR)
print("Using images_dir =", images_dir)
print(
    "Files:",
    [
        os.path.basename(train_path),
        os.path.basename(test_path),
        os.path.basename(sample_path),
    ],
)



## === cell 2
from PIL import Image
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)

target_cols = ["healthy", "multiple_diseases", "rust", "scab"]

missing_train = [c for c in (["image_id"] + target_cols) if c not in train.columns]
missing_test = [c for c in (["image_id"]) if c not in test.columns]
missing_sub = [c for c in (["image_id"] + target_cols) if c not in sample_sub.columns]
if missing_train:
    raise ValueError(f"train.csv missing columns: {missing_train}")
if missing_test:
    raise ValueError(f"test.csv missing columns: {missing_test}")
if missing_sub:
    raise ValueError(f"sample_submission.csv missing columns: {missing_sub}")

np.random.seed(42)

_IMAGE_ID_TO_PATH = None


def _build_image_index():
    idx = {}
    for fn in os.listdir(images_dir):
        if not fn.lower().endswith(".jpg"):
            continue
        idx[fn] = os.path.join(images_dir, fn)
        idx[os.path.splitext(fn)[0]] = os.path.join(images_dir, fn)
    return idx


def _resolve_image_path(image_id: str) -> str:
    global _IMAGE_ID_TO_PATH
    if _IMAGE_ID_TO_PATH is None:
        _IMAGE_ID_TO_PATH = _build_image_index()

    key = str(image_id)
    if key in _IMAGE_ID_TO_PATH:
        return _IMAGE_ID_TO_PATH[key]
    if f"{key}.jpg" in _IMAGE_ID_TO_PATH:
        return _IMAGE_ID_TO_PATH[f"{key}.jpg"]

    for pref in ("Train_", "Test_", "train_", "test_"):
        k2 = f"{pref}{key}"
        if k2 in _IMAGE_ID_TO_PATH:
            return _IMAGE_ID_TO_PATH[k2]
        if f"{k2}.jpg" in _IMAGE_ID_TO_PATH:
            return _IMAGE_ID_TO_PATH[f"{k2}.jpg"]

    raise FileNotFoundError(f"Image not found for image_id={image_id} in {images_dir}")


_IMG_CACHE = {}


def _safe_open_image(path):
    if path in _IMG_CACHE:
        return _IMG_CACHE[path].copy()
    with Image.open(path) as im:
        rgb = im.convert("RGB")
    _IMG_CACHE[path] = rgb
    return rgb.copy()


def _hog_like(gray, cell_size=12, num_bins=9):
    """
    Minimal, deterministic HOG-like descriptor:
    - central-diff gradients
    - unsigned orientation hist per cell
    """
    h, w = gray.shape
    if h < 3 or w < 3:
        return np.zeros((0,), dtype=np.float32)

    gx = gray[:, 2:] - gray[:, :-2]
    gy = gray[2:, :] - gray[:-2, :]
    gx = gx[1:-1, :]
    gy = gy[:, 1:-1]

    mag = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    ang = (np.arctan2(gy, gx) % np.pi).astype(np.float32)

    hh, ww = mag.shape
    ncy = hh // cell_size
    ncx = ww // cell_size
    if ncy == 0 or ncx == 0:
        return np.zeros((0,), dtype=np.float32)

    mag = mag[: ncy * cell_size, : ncx * cell_size]
    ang = ang[: ncy * cell_size, : ncx * cell_size]

    bin_edges = np.linspace(0.0, np.pi, num_bins + 1, dtype=np.float32)
    feat = np.zeros((ncy, ncx, num_bins), dtype=np.float32)

    for cy in range(ncy):
        y0 = cy * cell_size
        y1 = y0 + cell_size
        for cx in range(ncx):
            x0 = cx * cell_size
            x1 = x0 + cell_size
            a = ang[y0:y1, x0:x1].reshape(-1)
            m = mag[y0:y1, x0:x1].reshape(-1)
            b = np.searchsorted(bin_edges, a, side="right") - 1
            b = np.clip(b, 0, num_bins - 1)
            hist = np.bincount(b, weights=m, minlength=num_bins).astype(np.float32)
            hist /= np.linalg.norm(hist) + 1e-6
            feat[cy, cx, :] = hist

    return feat.reshape(-1).astype(np.float32)


def _spatial_pool_stats(arr01, grid=(2, 2)):
    """
    Spatial pooling of simple stats to capture localized changes.
    """
    h, w, _ = arr01.shape
    gy, gx = grid
    ys = np.linspace(0, h, gy + 1).astype(int)
    xs = np.linspace(0, w, gx + 1).astype(int)

    feats = []
    for i in range(gy):
        y0, y1 = ys[i], ys[i + 1]
        for j in range(gx):
            x0, x1 = xs[j], xs[j + 1]
            patch = arr01[y0:y1, x0:x1, :]
            if patch.size == 0:
                feats.append(np.zeros((6,), dtype=np.float32))
                continue
            flat = patch.reshape(-1, 3)
            m = flat.mean(axis=0).astype(np.float32)
            s = flat.std(axis=0).astype(np.float32)
            feats.append(np.concatenate([m, s], axis=0))
    return np.concatenate(feats, axis=0).astype(np.float32)


def _tiny_gray_patch(gray, out_hw=(24, 24)):
    """
    Add a very small downsampled grayscale "patch" as features.
    """
    im = Image.fromarray(np.clip(gray * 255.0, 0, 255).astype(np.uint8), mode="L")
    im = im.resize((out_hw[1], out_hw[0]), resample=Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32) / 255.0
    a = a.reshape(-1)
    a = (a - a.mean()) / (a.std() + 1e-6)
    return a.astype(np.float32)


def _gray_world_normalize(arr01, eps=1e-6):
    """
    Apply simple gray-world color constancy to reduce illumination differences.
    """
    ch_mean = arr01.reshape(-1, 3).mean(axis=0).astype(np.float32)
    scale = (ch_mean.mean() / (ch_mean + eps)).astype(np.float32)  # per-channel
    out = arr01 * scale[None, None, :]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def extract_features(image_id, resize=(160, 160), hist_bins=16):
    img_path = _resolve_image_path(image_id)
    im = _safe_open_image(img_path)
    if resize is not None:
        im = im.resize(resize)
    arr = np.asarray(im, dtype=np.float32) / 255.0  # H,W,3

    arr = _gray_world_normalize(arr)

    flat = arr.reshape(-1, 3)
    ch_mean = flat.mean(axis=0).astype(np.float32)
    ch_std = flat.std(axis=0).astype(np.float32)

    gray = (0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]).astype(
        np.float32
    )
    g_mean = np.float32(gray.mean())
    g_std = np.float32(gray.std())

    dh = (
        np.float32(np.abs(gray[:, 1:] - gray[:, :-1]).mean())
        if gray.shape[1] > 1
        else np.float32(0.0)
    )
    dv = (
        np.float32(np.abs(gray[1:, :] - gray[:-1, :]).mean())
        if gray.shape[0] > 1
        else np.float32(0.0)
    )

    hist, _ = np.histogram(gray, bins=hist_bins, range=(0.0, 1.0))
    hist = hist.astype(np.float32)
    hist = hist / (hist.sum() + 1e-6)

    gx = gray[:, 2:] - gray[:, :-2] if gray.shape[1] > 2 else np.zeros_like(gray)
    gy = gray[2:, :] - gray[:-2, :] if gray.shape[0] > 2 else np.zeros_like(gray)

    if gx.ndim == 2 and gy.ndim == 2 and gx.size > 0 and gy.size > 0:
        gx2 = gx[1:-1, :] if gx.shape[0] > 2 else gx
        gy2 = gy[:, 1:-1] if gy.shape[1] > 2 else gy
        h0 = min(gx2.shape[0], gy2.shape[0])
        w0 = min(gx2.shape[1], gy2.shape[1])
        gx2 = gx2[:h0, :w0]
        gy2 = gy2[:h0, :w0]
        mag = np.sqrt(gx2 * gx2 + gy2 * gy2).astype(np.float32)
        edge_mean = np.float32(mag.mean())
        edge_std = np.float32(mag.std())
        edge_p90 = np.float32(np.quantile(mag.reshape(-1), 0.90))
    else:
        edge_mean, edge_std, edge_p90 = (
            np.float32(0.0),
            np.float32(0.0),
            np.float32(0.0),
        )

    hog_s1 = _hog_like(gray, cell_size=12, num_bins=9)
    hog_s2 = _hog_like(gray, cell_size=20, num_bins=9)

    pool = _spatial_pool_stats(arr, grid=(2, 2))
    tiny = _tiny_gray_patch(gray, out_hw=(24, 24))

    feat = np.concatenate(
        [
            ch_mean,
            ch_std,
            np.array(
                [g_mean, g_std, dh, dv, edge_mean, edge_std, edge_p90], dtype=np.float32
            ),
            hist,
            pool,
            hog_s1,
            hog_s2,
            tiny,
        ],
        axis=0,
    ).astype(np.float32)
    return feat


X_train = np.vstack([extract_features(iid) for iid in train["image_id"].values])
X_test = np.vstack([extract_features(iid) for iid in test["image_id"].values])

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_sp = poly.fit_transform(X_train_s).astype(np.float32, copy=False)
X_test_sp = poly.transform(X_test_s).astype(np.float32, copy=False)

models = {}
test_pred = np.zeros((len(test), len(target_cols)), dtype=np.float32)

for j, c in enumerate(target_cols):
    y = train[c].astype(int).values

    clf = LogisticRegression(
        solver="lbfgs",
        penalty="l2",
        max_iter=3000,
        class_weight=None,
        random_state=42,
        C=6.0,
        n_jobs=1,  # critical to prevent joblib pickling/memmapping "No space left on device"
    )
    clf.fit(X_train_sp, y)
    test_pred[:, j] = clf.predict_proba(X_test_sp)[:, 1].astype(np.float32)
    models[c] = clf

sub = pd.DataFrame({"image_id": test["image_id"].values})
for j, c in enumerate(target_cols):
    sub[c] = np.clip(test_pred[:, j], 0.0, 1.0)

sub = sub[sample_sub.columns.tolist()]
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
