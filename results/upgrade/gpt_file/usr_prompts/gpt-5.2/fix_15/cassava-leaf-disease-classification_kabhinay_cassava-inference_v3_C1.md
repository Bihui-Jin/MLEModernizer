# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
np.random.seed(42)

print("Starting (TensorFlow-free fallback)")



## === cell 1
from PIL import Image, ImageFile
import multiprocessing as mp
from multiprocessing import shared_memory

ImageFile.LOAD_TRUNCATED_IMAGES = True

_U8_TO_BIN16 = (np.arange(256, dtype=np.uint8) >> 4).astype(np.uint8)

_G_BASE_DIR = None
_G_FEAT_DIM = None

_G_SHM_NAME = None
_G_SHM_SHAPE = None
_G_SHM_DTYPE = None
_G_SHM = None
_G_X = None
_G_Y = None
_G_IDS = None


def _pool_init_shared(base_dir, feat_dim, shm_name, shm_shape, shm_dtype, mode):
    """
    Speed: Use shared memory so workers write features directly into the preallocated array.
    Correctness: Features and downstream logic unchanged; only transport mechanism changes.
    """
    global _G_BASE_DIR, _G_FEAT_DIM, _G_SHM_NAME, _G_SHM_SHAPE, _G_SHM_DTYPE, _G_SHM, _G_X, _G_Y, _G_IDS
    _G_BASE_DIR = base_dir
    _G_FEAT_DIM = feat_dim

    _G_SHM_NAME = shm_name
    _G_SHM_SHAPE = tuple(shm_shape)
    _G_SHM_DTYPE = np.dtype(shm_dtype)

    _G_SHM = shared_memory.SharedMemory(name=_G_SHM_NAME)
    _G_X = np.ndarray(_G_SHM_SHAPE, dtype=_G_SHM_DTYPE, buffer=_G_SHM.buf)

    _G_Y = None
    _G_IDS = None


def _hist_channel_uint8_bincount(x_uint8, bins=16):
    idx = _U8_TO_BIN16[x_uint8].ravel()
    h = np.bincount(idx, minlength=bins).astype(np.float32, copy=False)
    s = h.sum()
    if s > 0:
        h /= s
    return h


def extract_features_uint8(rgb):
    """
    Same features as before (core logic unchanged).
    Speed: avoid allocating full float32 image; compute mean/std/brightness/saturation from uint8 with float64 accumulators.
    Correctness: results are numerically equivalent up to negligible FP differences.
    """
    rgb_f = rgb.astype(np.float32, copy=False)

    mean = (rgb_f.mean(axis=(0, 1)) * (1.0 / 255.0)).astype(np.float32, copy=False)

    std = (rgb_f.std(axis=(0, 1)) * (1.0 / 255.0)).astype(np.float32, copy=False)

    bright = np.float32(rgb_f.mean() * (1.0 / 255.0))

    mx = rgb_f.max(axis=2)
    mn = rgb_f.min(axis=2)
    sat = np.float32(((mx - mn).mean()) * (1.0 / 255.0))

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    hr = _hist_channel_uint8_bincount(r, bins=16)
    hg = _hist_channel_uint8_bincount(g, bins=16)
    hb = _hist_channel_uint8_bincount(b, bins=16)

    gray = (
        0.2989 * (rgb_f[:, :, 0] * (1.0 / 255.0))
        + 0.5870 * (rgb_f[:, :, 1] * (1.0 / 255.0))
        + 0.1140 * (rgb_f[:, :, 2] * (1.0 / 255.0))
    ).astype(np.float32, copy=False)

    gx = np.abs(gray[:, 1:] - gray[:, :-1]).mean()
    gy = np.abs(gray[1:, :] - gray[:-1, :]).mean()
    edge = np.float32(0.5 * (gx + gy))

    feats = np.concatenate(
        [mean, std, np.array([bright, sat, edge], dtype=np.float32), hr, hg, hb],
        axis=0,
    ).astype(np.float32, copy=False)
    return feats


def _open_rgb_drafted(path, target_size):
    im = Image.open(path)
    try:
        im.draft("RGB", target_size)
    except Exception:
        pass
    return im.convert("RGB")


def _resize_fast_bilinear(im, target_size):
    tw, th = target_size
    w, h = im.size
    rw = max(1, w // tw)
    rh = max(1, h // th)
    r = min(rw, rh)
    if r > 1:
        im = im.reduce(r)
    return im.resize((tw, th), resample=Image.BILINEAR)


def extract_features_multiscale(path, size_a=(224, 224), size_b=(128, 128)):
    """
    Same two-scale features as before (core logic unchanged).
    Speed: decode image once, resize to size_a, then derive size_b by resizing from size_a image (avoids second decode).
    Correctness: size_a is identical; size_b differs only by negligible FP/pixel resampling path and preserves semantics.
    """
    im = None
    im_a = None
    im_b = None
    try:
        im = _open_rgb_drafted(path, size_a)

        im_a = _resize_fast_bilinear(im, size_a)
        rgb_a = np.asarray(im_a, dtype=np.uint8)
        fa = extract_features_uint8(rgb_a)

        im_b = im_a.resize(size_b, resample=Image.BILINEAR)
        rgb_b = np.asarray(im_b, dtype=np.uint8)
        fb = extract_features_uint8(rgb_b)

        return np.concatenate([fa, fb], axis=0).astype(np.float32, copy=False)
    finally:
        for _im in (im_b, im_a, im):
            if _im is not None:
                try:
                    _im.close()
                except Exception:
                    pass


def standardize_fit(X):
    mu = X.mean(axis=0, keepdims=True).astype(np.float32, copy=False)
    sigma = X.std(axis=0, keepdims=True).astype(np.float32, copy=False)
    sigma = np.where(sigma < 1e-6, 1.0, sigma).astype(np.float32, copy=False)
    return mu, sigma


def standardize_apply(X, mu, sigma):
    return ((X - mu) / sigma).astype(np.float32, copy=False)


def fit_nearest_centroids(X, y, num_classes=5):
    """
    Same prototype-based classifier as before (core logic unchanged).
    """
    centroids = np.zeros((num_classes, X.shape[1]), dtype=np.float32)
    counts = np.zeros((num_classes,), dtype=np.int64)
    for c in range(num_classes):
        mask = y == c
        counts[c] = int(mask.sum())
        if mask.any():
            centroids[c] = X[mask].mean(axis=0)
        else:
            centroids[c] = X.mean(axis=0)

    resid = X - centroids[y]
    var = (resid * resid).mean(axis=0).astype(np.float32, copy=False)
    var = np.maximum(var, 1e-6).astype(np.float32, copy=False)

    shrink = np.float32(0.15)
    var = (1.0 - shrink) * var + shrink * np.float32(1.0)

    return centroids, var


def predict_nearest_centroids(X, centroids, var_diag):
    inv_var = (1.0 / var_diag).astype(np.float32, copy=False)  # (D,)
    Xw = X * inv_var[None, :]  # (N,D)
    term_x = (X * Xw).sum(axis=1, keepdims=True)  # (N,1)
    cent_w = centroids * inv_var[None, :]  # (C,D)
    term_c = (centroids * cent_w).sum(axis=1)[None, :]  # (1,C)
    cross = X @ cent_w.T  # (N,C)
    dists = term_x + term_c - 2.0 * cross
    return np.argmin(dists, axis=1).astype(int)


def _featurize_one_train_shared(args):
    """
    Speed: worker writes directly into shared X_train/Y_train at its assigned index.
    Correctness: extracted features identical; ordering preserved by using deterministic index mapping.
    """
    idx, image_id, label, train_img_dir = args
    path = os.path.join(train_img_dir, image_id)
    if not os.path.isfile(path):
        return 0
    try:
        feats = extract_features_multiscale(path, size_a=(224, 224), size_b=(128, 128))
        if feats.shape[0] != _G_FEAT_DIM:
            return 0
        _G_X[idx, :] = feats
        return 1
    except Exception:
        return 0


def _featurize_one_test_shared(args):
    """
    Speed: worker writes directly into shared X_test at its assigned index.
    Correctness: extracted features identical; ordering preserved by index mapping.
    """
    idx, image_id, test_dir = args
    path = os.path.join(test_dir, image_id)
    if not os.path.isfile(path):
        return 0
    try:
        feats = extract_features_multiscale(path, size_a=(224, 224), size_b=(128, 128))
        if feats.shape[0] != _G_FEAT_DIM:
            return 0
        _G_X[idx, :] = feats
        return 1
    except Exception:
        return 0




## === cell 2
BASE_PATH = "/kaggle/input/cassava-leaf-disease-classification"

TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TEST_DIR = os.path.join(BASE_PATH, "test_images")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")

assert os.path.isfile(TRAIN_CSV), f"Missing train.csv at: {TRAIN_CSV}"
assert os.path.isdir(TEST_DIR), f"Missing test_images directory at: {TEST_DIR}"
assert os.path.isfile(SAMPLE_SUB), f"Missing sample_submission.csv at: {SAMPLE_SUB}"

train_df = pd.read_csv(TRAIN_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

test_df = sample_df[["image_id"]].copy()

print("train rows:", len(train_df), "test rows:", len(test_df))



## === cell 3
MAX_TRAIN_USED = None  # use full train set

train_df = train_df.sort_values("image_id").reset_index(drop=True)
if (MAX_TRAIN_USED is not None) and (len(train_df) > MAX_TRAIN_USED):
    parts = []
    per_class = MAX_TRAIN_USED // train_df["label"].nunique()
    for c in sorted(train_df["label"].unique()):
        parts.append(train_df[train_df["label"] == c].head(per_class))
    train_small = pd.concat(parts, axis=0).reset_index(drop=True)
    if len(train_small) < MAX_TRAIN_USED:
        remaining = train_df[~train_df.index.isin(train_small.index)]
        need = MAX_TRAIN_USED - len(train_small)
        train_small = pd.concat(
            [train_small, remaining.head(need)], axis=0
        ).reset_index(drop=True)
else:
    train_small = train_df

print("Using train images for fitting:", len(train_small))

train_img_dir = os.path.join(BASE_PATH, "train_images")

num_train = len(train_small)
feat_dim = 2 * (3 + 3 + 3 + 16 * 3)

X_train_local = np.empty((num_train, feat_dim), dtype=np.float32)
shm_train = shared_memory.SharedMemory(create=True, size=X_train_local.nbytes)
X_train = np.ndarray(
    X_train_local.shape, dtype=X_train_local.dtype, buffer=shm_train.buf
)
del X_train_local  # free local buffer
y_train = train_small["label"].values.astype(np.int64, copy=True)

train_ids = train_small["image_id"].values.tolist()

cpu = os.cpu_count() or 2
workers = min(12, max(1, cpu - 1))

mp_chunksize = 256

try:
    mp.set_start_method("forkserver", force=True)
except Exception:
    pass

train_tasks = [
    (i, train_ids[i], int(y_train[i]), train_img_dir) for i in range(num_train)
]

with mp.Pool(
    processes=workers,
    initializer=_pool_init_shared,
    initargs=(
        train_img_dir,
        feat_dim,
        shm_train.name,
        X_train.shape,
        X_train.dtype.str,
        "train",
    ),
    maxtasksperchild=2000,
) as pool:
    ok_flags = list(
        pool.imap_unordered(
            _featurize_one_train_shared, train_tasks, chunksize=mp_chunksize
        )
    )

ok_mask = np.zeros((num_train,), dtype=bool)
with mp.Pool(
    processes=workers,
    initializer=_pool_init_shared,
    initargs=(
        train_img_dir,
        feat_dim,
        shm_train.name,
        X_train.shape,
        X_train.dtype.str,
        "train",
    ),
    maxtasksperchild=2000,
) as pool:
    for idx, ok in pool.imap_unordered(
        lambda t: (t[0], _featurize_one_train_shared(t)),
        train_tasks,
        chunksize=mp_chunksize,
    ):
        ok_mask[idx] = bool(ok)

X_train_f = X_train[ok_mask]
y_train_f = y_train[ok_mask]

shm_train.close()
shm_train.unlink()

assert len(X_train_f) > 0, "No training features extracted; cannot proceed."
print("Extracted train feature matrix:", X_train_f.shape)

mu, sigma = standardize_fit(X_train_f)
X_train_s = standardize_apply(X_train_f, mu, sigma)

centroids, var_diag = fit_nearest_centroids(X_train_s, y_train_f, num_classes=5)
print("Fitted centroids shape:", centroids.shape, "var_diag shape:", var_diag.shape)

num_test = len(test_df)

X_test_local = np.empty((num_test, feat_dim), dtype=np.float32)
shm_test = shared_memory.SharedMemory(create=True, size=X_test_local.nbytes)
X_test = np.ndarray(X_test_local.shape, dtype=X_test_local.dtype, buffer=shm_test.buf)
del X_test_local

test_ids = test_df["image_id"].values.tolist()
test_tasks = [(i, test_ids[i], TEST_DIR) for i in range(num_test)]

with mp.Pool(
    processes=workers,
    initializer=_pool_init_shared,
    initargs=(
        TEST_DIR,
        feat_dim,
        shm_test.name,
        X_test.shape,
        X_test.dtype.str,
        "test",
    ),
    maxtasksperchild=2000,
) as pool:
    ok_mask_test = np.zeros((num_test,), dtype=bool)
    for idx, ok in pool.imap_unordered(
        lambda t: (t[0], _featurize_one_test_shared(t)),
        test_tasks,
        chunksize=mp_chunksize,
    ):
        ok_mask_test[idx] = bool(ok)

X_test_valid = X_test[ok_mask_test]
valid_ids_list = [test_ids[i] for i in range(num_test) if ok_mask_test[i]]
missing = int((~ok_mask_test).sum())

shm_test.close()
shm_test.unlink()

print(
    "Extracted test feature matrix:", X_test_valid.shape, "missing/unreadable:", missing
)

X_test_s = standardize_apply(X_test_valid, mu, sigma)
pred_valid = predict_nearest_centroids(X_test_s, centroids, var_diag)

pred_map = dict(zip(valid_ids_list, pred_valid))
fallback_label = int(pd.Series(y_train_f).value_counts().index[0])

mapped = test_df["image_id"].map(pred_map)
pred_all = mapped.fillna(fallback_label).astype(np.int64).values

submission = pd.DataFrame({"image_id": test_df["image_id"].values, "label": pred_all})

assert submission.shape[0] == sample_df.shape[0], "Submission row count mismatch."
assert list(submission.columns) == ["image_id", "label"], "Submission columns mismatch."
assert submission["label"].between(0, 4).all(), "Labels must be integers in [0,4]."

out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission.head())
