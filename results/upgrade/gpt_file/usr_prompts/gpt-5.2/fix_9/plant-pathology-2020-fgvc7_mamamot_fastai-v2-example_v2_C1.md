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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy import sparse

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

np.random.seed(0)

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

for _k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_k, str(os.cpu_count() or 2))




## === cell 1
data_path = Path("/kaggle/input/plant-pathology-2020-fgvc7")

if not data_path.exists():
    data_path = Path("/kaggle/data/plant-pathology-2020-fgvc7")

data_path




## === cell 2
train_path = data_path / "train.csv"
test_path = data_path / "test.csv"
sample_path = data_path / "sample_submission.csv"
images_dir = data_path / "images"

df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)

df.shape, test_df.shape, sample_sub.shape, images_dir.exists()




## === cell 3
df.head()




## === cell 4
target_cols = [c for c in sample_sub.columns if c != "image_id"]
target_cols




## === cell 5
imglabels = target_cols  # expected order: healthy, multiple_diseases, rust, scab

y_mat = df[imglabels].to_numpy(dtype=float)
df["labels"] = np.asarray(imglabels, dtype=object)[y_mat.argmax(axis=1)]
df["labels"].value_counts()




## === cell 6
priors = df[imglabels].mean(axis=0).astype(float)
priors = priors.clip(0.0, 1.0)
priors




## === cell 7
def _image_path(image_id: str) -> Path:
    iid = str(image_id)
    if not iid.lower().endswith(".jpg"):
        iid = iid + ".jpg"
    return images_dir / iid


_IMAGE_CACHE = {}


def _load_resized_rgb01(image_id: str, size_wh):
    key = (str(image_id), size_wh)
    arr = _IMAGE_CACHE.get(key)
    if arr is not None:
        return arr
    p = _image_path(image_id)
    try:
        with Image.open(p) as im:
            im = im.convert("RGB")
            im = im.resize(size_wh, resample=Image.BILINEAR)
            im.load()
            arr = np.asarray(im, dtype=np.float32) / 255.0
    except Exception:
        arr = None
    _IMAGE_CACHE[key] = arr
    return arr


def _features_for_one(iid: str, size_wh, rgb_dim: int):
    arr = _load_resized_rgb01(iid, size_wh)
    if arr is None:
        return None

    feat = np.empty((rgb_dim + 14,), dtype=np.float32)
    feat[:rgb_dim] = arr.reshape(-1)

    ch_mean = arr.mean(axis=(0, 1))
    ch_std = arr.std(axis=(0, 1))

    gray = (0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]).astype(
        np.float32
    )
    gx = np.diff(gray, axis=1, prepend=gray[:, :1])
    gy = np.diff(gray, axis=0, prepend=gray[:1, :])
    mag = np.sqrt(gx * gx + gy * gy, dtype=np.float32)
    edge_mean = float(mag.mean())
    edge_std = float(mag.std())

    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin + 1e-8

    hue = np.zeros_like(cmax, dtype=np.float32)
    mask = cmax == r
    hue[mask] = ((g[mask] - b[mask]) / delta[mask]) % 6.0
    mask = cmax == g
    hue[mask] = ((b[mask] - r[mask]) / delta[mask]) + 2.0
    mask = cmax == b
    hue[mask] = ((r[mask] - g[mask]) / delta[mask]) + 4.0
    hue = (hue / 6.0).astype(np.float32)

    sat = (delta / (cmax + 1e-8)).astype(np.float32)
    val = cmax.astype(np.float32)

    hsv_mean = np.array([hue.mean(), sat.mean(), val.mean()], dtype=np.float32)
    hsv_std = np.array([hue.std(), sat.std(), val.std()], dtype=np.float32)

    base = rgb_dim
    feat[base : base + 3] = ch_mean
    feat[base + 3 : base + 6] = ch_std
    feat[base + 6] = edge_mean
    feat[base + 7] = edge_std
    feat[base + 8 : base + 11] = hsv_mean
    feat[base + 11 : base + 14] = hsv_std
    return feat


def _mp_worker_feature(args):
    i, iid, size_wh, rgb_dim = args
    feat = _features_for_one(iid, size_wh, rgb_dim)
    return i, feat


_MP_GLOBALS = {}


def _mp_init(images_dir_, size_wh_, rgb_dim_):
    global images_dir
    images_dir = images_dir_
    _MP_GLOBALS["size_wh"] = size_wh_
    _MP_GLOBALS["rgb_dim"] = rgb_dim_


def _mp_worker_feature_fast(tup):
    i, iid = tup
    feat = _features_for_one(iid, _MP_GLOBALS["size_wh"], _MP_GLOBALS["rgb_dim"])
    return i, feat


def extract_image_features(image_ids, size=(64, 64)):
    """
    Returns a dense float32 matrix of shape (n, d).
    Features: resized RGB (flattened), per-channel mean/std, edge magnitude mean/std,
    and simple HSV-like stats computed from RGB.
    """
    image_ids = np.asarray(image_ids, dtype=object)
    n = len(image_ids)
    w, h = int(size[0]), int(size[1])
    size_wh = (w, h)

    rgb_dim = h * w * 3
    d = rgb_dim + 14
    X = np.zeros((n, d), dtype=np.float32)

    import multiprocessing as mp

    n_jobs = min(8, os.cpu_count() or 2)

    if n_jobs <= 1 or n < 32:
        for i, iid in enumerate(image_ids):
            feat = _features_for_one(str(iid), size_wh, rgb_dim)
            if feat is not None:
                X[i] = feat
        return X

    try:
        ctx = mp.get_context("fork")
    except Exception:
        ctx = mp.get_context("spawn")

    chunksize = max(32, n // (n_jobs * 2) if n_jobs > 0 else 32)

    tasks = ((i, str(iid)) for i, iid in enumerate(image_ids))

    try:
        with ctx.Pool(
            processes=n_jobs,
            initializer=_mp_init,
            initargs=(images_dir, size_wh, rgb_dim),
            maxtasksperchild=0,
        ) as pool:
            for i, feat in pool.imap(
                _mp_worker_feature_fast, tasks, chunksize=chunksize
            ):
                if feat is not None:
                    X[i] = feat
    except Exception:
        for i, iid in enumerate(image_ids):
            feat = _features_for_one(str(iid), size_wh, rgb_dim)
            if feat is not None:
                X[i] = feat

    return X




## === cell 8
train_ids = df["image_id"].astype(str).values
test_ids = test_df["image_id"].astype(str).values

X_train_text = train_ids
X_test_text = test_ids

vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1)
Xtr_txt = vectorizer.fit_transform(X_train_text)
Xte_txt = vectorizer.transform(X_test_text)

Xtr_img = extract_image_features(train_ids, size=(64, 64))
Xte_img = extract_image_features(test_ids, size=(64, 64))

scaler = StandardScaler(with_mean=False, with_std=True)
Xtr_img_s = scaler.fit_transform(Xtr_img).astype(np.float32, copy=False)
Xte_img_s = scaler.transform(Xte_img).astype(np.float32, copy=False)

Xtr = sparse.hstack([Xtr_txt, sparse.csr_matrix(Xtr_img_s)], format="csr")
Xte = sparse.hstack([Xte_txt, sparse.csr_matrix(Xte_img_s)], format="csr")

preds = np.zeros((len(test_df), len(imglabels)), dtype=float)

for j, c in enumerate(imglabels):
    y = df[c].astype(int).values

    if y.min() == y.max():
        preds[:, j] = float(priors[c])
        continue

    clf = LogisticRegression(
        solver="saga",
        C=2.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=0,
        n_jobs=-1,
    )
    clf.fit(Xtr, y)
    preds[:, j] = clf.predict_proba(Xte)[:, 1]

alpha = 0.95  # keep same blend: mostly model signal, small prior smoothing
prior_vec = priors[imglabels].values.reshape(1, -1)
preds = alpha * preds + (1.0 - alpha) * prior_vec
preds = np.clip(preds, 1e-6, 1 - 1e-6)

submission = pd.DataFrame({"image_id": test_ids})
for j, c in enumerate(imglabels):
    submission[c] = preds[:, j].astype(float)

submission = submission[sample_sub.columns]
submission.head(10)




## === cell 9
out_path = Path("submission.csv")
submission.to_csv(out_path, index=False)

out_path, out_path.exists(), submission.shape




## === cell 10
assert out_path.suffix == ".csv"
assert list(submission.columns) == list(sample_sub.columns)
assert submission.isna().sum().sum() == 0
assert len(submission) == len(test_df)
submission.describe(include="all")
