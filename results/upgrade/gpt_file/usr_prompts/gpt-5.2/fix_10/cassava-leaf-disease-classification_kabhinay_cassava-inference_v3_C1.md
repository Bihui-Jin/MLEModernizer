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

ImageFile.LOAD_TRUNCATED_IMAGES = True

_U8_TO_BIN16 = (np.arange(256, dtype=np.uint8) >> 4).astype(np.uint8)


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
    """
    x = rgb.astype(np.float32) / 255.0
    mean = x.mean(axis=(0, 1))  # (3,)
    std = x.std(axis=(0, 1))  # (3,)

    bright = x.mean()  # scalar
    sat = (x.max(axis=2) - x.min(axis=2)).mean()  # scalar saturation proxy

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    hr = _hist_channel_uint8_bincount(r, bins=16)
    hg = _hist_channel_uint8_bincount(g, bins=16)
    hb = _hist_channel_uint8_bincount(b, bins=16)

    gray = (0.2989 * x[:, :, 0] + 0.5870 * x[:, :, 1] + 0.1140 * x[:, :, 2]).astype(
        np.float32, copy=False
    )
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
    im = None
    try:
        im = _open_rgb_drafted(path, size_a)
        im_a = _resize_fast_bilinear(im, size_a)
        rgb_a = np.asarray(im_a, dtype=np.uint8)
        fa = extract_features_uint8(rgb_a)

        w, h = im_a.size
        tw, th = size_b
        rw = max(1, w // tw)
        rh = max(1, h // th)
        r = min(rw, rh)
        im_b = im_a.reduce(r) if r > 1 else im_a
        im_b = im_b.resize((tw, th), resample=Image.BILINEAR)
        rgb_b = np.asarray(im_b, dtype=np.uint8)
        fb = extract_features_uint8(rgb_b)

        return np.concatenate([fa, fb], axis=0).astype(np.float32, copy=False)
    finally:
        if im is not None:
            try:
                im.close()
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


def _featurize_one(args):
    image_id, label_or_none, base_dir, is_train, feat_dim = args
    path = os.path.join(base_dir, image_id)
    if not os.path.isfile(path):
        return None, image_id, (int(label_or_none) if is_train else None)
    try:
        feats = extract_features_multiscale(path, size_a=(224, 224), size_b=(128, 128))
        if feats.shape[0] != feat_dim:
            return None, image_id, (int(label_or_none) if is_train else None)
        return feats, image_id, (int(label_or_none) if is_train else None)
    except Exception:
        return None, image_id, (int(label_or_none) if is_train else None)


def _featurize_batch(batch_args):
    out = []
    for args in batch_args:
        out.append(_featurize_one(args))
    return out


def _iter_batches(seq, batch_size):
    for i in range(0, len(seq), batch_size):
        yield seq[i : i + batch_size]




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

X_train = np.empty((num_train, feat_dim), dtype=np.float32)
y_train = np.empty((num_train,), dtype=np.int64)

train_args = [
    (img_id, lbl, train_img_dir, True, feat_dim)
    for img_id, lbl in zip(train_small["image_id"].values, train_small["label"].values)
]

workers = min(8, max(1, (os.cpu_count() or 2) - 1))

mp_chunksize = 8
batch_size = 128

write = 0
with mp.Pool(processes=workers, maxtasksperchild=500) as pool:
    for batch_out in pool.imap_unordered(
        _featurize_batch, _iter_batches(train_args, batch_size), chunksize=mp_chunksize
    ):
        for feats, _img_id, lbl in batch_out:
            if feats is None:
                continue
            X_train[write] = feats
            y_train[write] = lbl
            write += 1

X_train = X_train[:write]
y_train = y_train[:write]

assert len(X_train) > 0, "No training features extracted; cannot proceed."
print("Extracted train feature matrix:", X_train.shape)

mu, sigma = standardize_fit(X_train)
X_train_s = standardize_apply(X_train, mu, sigma)

centroids, var_diag = fit_nearest_centroids(X_train_s, y_train, num_classes=5)
print("Fitted centroids shape:", centroids.shape, "var_diag shape:", var_diag.shape)



## === cell 4
num_test = len(test_df)
X_test = np.empty((num_test, feat_dim), dtype=np.float32)
valid_ids = np.empty((num_test,), dtype=object)

test_args = [
    (img_id, None, TEST_DIR, False, feat_dim) for img_id in test_df["image_id"].values
]

write = 0
missing = 0
with mp.Pool(processes=workers, maxtasksperchild=500) as pool:
    for batch_out in pool.imap_unordered(
        _featurize_batch, _iter_batches(test_args, batch_size), chunksize=mp_chunksize
    ):
        for feats, img_id, _ in batch_out:
            if feats is None:
                missing += 1
                continue
            X_test[write] = feats
            valid_ids[write] = img_id
            write += 1

X_test_valid = X_test[:write]
valid_ids_list = valid_ids[:write].tolist()

print(
    "Extracted test feature matrix:", X_test_valid.shape, "missing/unreadable:", missing
)

X_test_s = standardize_apply(X_test_valid, mu, sigma)
pred_valid = predict_nearest_centroids(X_test_s, centroids, var_diag)

pred_map = dict(zip(valid_ids_list, pred_valid))
fallback_label = int(pd.Series(y_train).value_counts().index[0])

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
