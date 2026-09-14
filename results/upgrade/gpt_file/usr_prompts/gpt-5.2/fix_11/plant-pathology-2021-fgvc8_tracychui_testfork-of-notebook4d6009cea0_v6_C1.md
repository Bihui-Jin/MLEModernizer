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
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import glob
import gc
import random
import numpy as np
import pandas as pd

from PIL import Image as PILImage

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

import tensorflow.keras.applications.resnet50 as resnet
from keras.preprocessing import image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

K.set_image_data_format("channels_last")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass
try:
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    tf.config.optimizer.set_jit(False)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

print("keras:", keras.__version__, "tf:", tf.__version__)




## === cell 1
IMG_WIDTH = 300
IMG_HEIGHT = 300
NR_CHANNELS = 3




## === cell 2
TEST_DIR = "../input/plant-pathology-2021-fgvc8/test_images"
TRAIN_DIR = "../input/plant-pathology-2021-fgvc8/train_images"
TRAIN_CSV_PATH = "../input/plant-pathology-2021-fgvc8/train.csv"
SAMPLE_SUB_PATH = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"

sub0 = pd.read_csv(SAMPLE_SUB_PATH, usecols=["image"])
imglist_test = [os.path.join(TEST_DIR, fn) for fn in sub0["image"].astype(str).tolist()]
len(imglist_test)




## === cell 3
base = resnet.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_HEIGHT, IMG_WIDTH, NR_CHANNELS),
    pooling="avg",
)
base.trainable = False

AUTOTUNE = tf.data.AUTOTUNE


def _decode_resize_preprocess(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)  # RGB
    img = tf.image.resize(
        img,
        [IMG_HEIGHT, IMG_WIDTH],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    img = tf.cast(img, tf.float32)
    img = img[..., ::-1]  # RGB -> BGR
    mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32)  # BGR means
    img = img - mean
    img.set_shape([IMG_HEIGHT, IMG_WIDTH, 3])
    return img


@tf.function(jit_compile=True)
def _base_forward(batch):
    return base(batch, training=False)


def extract_features(paths, batch_size=128, verbose=1, cache_path=None):
    """
    Runtime fixes (no core-logic change):
    - Keep exact preprocessing, model, and output features.
    - Make tf.data pipeline more efficient (parallel map, prefetch, deterministic options).
    - Cache decoded+resized tensors in-memory during a single run via ds.cache().
    - Persist features to disk keyed by the exact ordered path list to skip recomputation.
    - Avoid Keras predict() overhead by iterating batches and calling compiled forward.
    """
    paths = list(paths)
    n = len(paths)
    if n == 0:
        feats = np.zeros((0, 2048), dtype=np.float32)
        if verbose:
            print("Extracted features:", feats.shape)
        return feats

    cache_feats = None
    cache_manifest = None
    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        cache_feats = cache_path + "_feats.npy"
        cache_manifest = cache_path + "_paths.npy"
        if os.path.exists(cache_feats) and os.path.exists(cache_manifest):
            try:
                cached_paths = np.load(cache_manifest, allow_pickle=True)
                if cached_paths.shape[0] == n and np.array_equal(
                    cached_paths.astype(str), np.asarray(paths, dtype=str)
                ):
                    feats = np.load(cache_feats, mmap_mode=None).astype(
                        np.float32, copy=False
                    )
                    if verbose:
                        print(
                            "Loaded cached features:", feats.shape, "from", cache_feats
                        )
                    return feats
            except Exception:
                pass

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    try:
        opts.experimental_optimization.apply_default_optimizations = True
        opts.experimental_optimization.map_parallelization = True
        opts.experimental_optimization.map_and_batch_fusion = True
        opts.experimental_optimization.parallel_batch = True
    except Exception:
        pass

    ds = tf.data.Dataset.from_tensor_slices(tf.constant(paths)).with_options(opts)
    ds = ds.map(
        _decode_resize_preprocess, num_parallel_calls=AUTOTUNE, deterministic=True
    )

    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)

    out = np.empty((n, 2048), dtype=np.float32)
    offset = 0

    _ = _base_forward(tf.zeros((1, IMG_HEIGHT, IMG_WIDTH, 3), dtype=tf.float32))

    for batch in ds:
        bsz = int(batch.shape[0])
        pred = _base_forward(batch)
        pred_np = pred.numpy().astype(np.float32, copy=False)
        out[offset : offset + bsz] = pred_np
        offset += bsz

    feats = out

    if cache_path is not None:
        try:
            np.save(cache_feats, feats)
            np.save(cache_manifest, np.asarray(paths, dtype=object))
            if verbose:
                print("Saved cached features:", feats.shape, "to", cache_feats)
        except Exception:
            pass

    if verbose:
        print("Extracted features:", feats.shape)
    return feats


try:
    base.compile(run_eagerly=False)
except Exception:
    pass




## === cell 4
training_csv = pd.read_csv(TRAIN_CSV_PATH)

labels_s = training_csv["labels"].astype(str)
tagnames = np.array(sorted(pd.unique(labels_s.str.split().explode().dropna())))
print("Num classes:", len(tagnames))
print("Classes:", tagnames)

tag2idx = {t: i for i, t in enumerate(tagnames)}

Y = np.zeros((len(training_csv), len(tagnames)), dtype=np.int8)
labels_split = labels_s.str.split().tolist()
for r, parts in enumerate(labels_split):
    if parts:
        cols = [tag2idx[t] for t in parts]
        Y[r, cols] = 1

train_paths = [os.path.join(TRAIN_DIR, fn) for fn in training_csv["image"].values]

print("Missing train images: (skipped for speed)")




## === cell 5
idx = np.arange(len(train_paths))
tr_idx, va_idx = train_test_split(idx, test_size=0.2, random_state=SEED, shuffle=True)

train_paths_tr = [train_paths[i] for i in tr_idx]
train_paths_va = [train_paths[i] for i in va_idx]
Y_tr = Y[tr_idx]
Y_va = Y[va_idx]

print("Train:", len(train_paths_tr), "Val:", len(train_paths_va))




## === cell 6
cache_dir = "./tfdata_cache"
Xf_tr = extract_features(
    train_paths_tr,
    batch_size=128,
    verbose=1,
    cache_path=os.path.join(cache_dir, "train"),
)
Xf_va = extract_features(
    train_paths_va, batch_size=128, verbose=1, cache_path=os.path.join(cache_dir, "val")
)

scaler = MinMaxScaler(feature_range=(0, 1))
Xn_tr = scaler.fit_transform(Xf_tr)
Xn_va = scaler.transform(Xf_va)

del Xf_tr, Xf_va
gc.collect()




## === cell 7
tagmodels1 = {}
val_pred_proba = np.zeros((Xn_va.shape[0], len(tagnames)), dtype=np.float32)

col_has_both = Y_tr.min(axis=0) != Y_tr.max(axis=0)
idx_fit = np.flatnonzero(col_has_both)
idx_const = np.flatnonzero(~col_has_both)

for i in idx_const:
    t = tagnames[i]
    tagmodels1[t] = None
    prior = float(Y_tr[:, i].mean())
    val_pred_proba[:, i] = prior

if idx_fit.size:
    clf_multi = LogisticRegression(
        solver="liblinear",
        max_iter=1000,
        random_state=SEED,
    )
    clf_multi.fit(Xn_tr, Y_tr[:, idx_fit])

    for k, i in enumerate(idx_fit):
        tagmodels1[tagnames[i]] = clf_multi  # shared estimator

    proba_list = clf_multi.predict_proba(Xn_va)
    for k, i in enumerate(idx_fit):
        val_pred_proba[:, i] = proba_list[k][:, 1].astype(np.float32, copy=False)

print("Trained models:", int(idx_fit.size), "/", len(tagnames))




## === cell 8
thr_grid = np.linspace(0.2, 0.9, 15).astype(np.float32)
best_thr = np.full(len(tagnames), 0.5, dtype=np.float32)

P = val_pred_proba.astype(np.float32, copy=False)  # (n, c)
Yv = Y_va.astype(np.int8, copy=False)
Yb = Yv > 0

n, c = P.shape
T = thr_grid.size

pos_cnt = Yb.sum(axis=0).astype(np.float32)  # (c,)
neg_cnt = (n - pos_cnt).astype(np.float32)  # (c,)

tp = np.zeros((c, T), dtype=np.float32)
pp = np.zeros((c, T), dtype=np.float32)  # predicted positives per class+thr

for ti, thr in enumerate(thr_grid):
    pred = P >= thr
    pp[:, ti] = pred.sum(axis=0).astype(np.float32)
    tp[:, ti] = (pred & Yb).sum(axis=0).astype(np.float32)

fp = pp - tp
fn = pos_cnt[:, None] - tp

den = 2.0 * tp + fp + fn
f1 = np.where(den > 0, (2.0 * tp) / den, 0.0).astype(np.float32)

has_pos = pos_cnt > 0
best_idx = np.argmax(f1, axis=1)
best_thr[has_pos] = thr_grid[best_idx[has_pos]]

val_pred_bin = (P >= best_thr.reshape(1, -1)).astype(np.int8)

y_true = Yv.astype(np.int8, copy=False)
y_pred = val_pred_bin.astype(np.int8, copy=False)
tp2 = (y_true & y_pred).sum(axis=0).astype(np.float32)
fp2 = ((1 - y_true) & y_pred).sum(axis=0).astype(np.float32)
fn2 = (y_true & (1 - y_pred)).sum(axis=0).astype(np.float32)
den2 = 2.0 * tp2 + fp2 + fn2
f1_per_class = np.where(den2 > 0, (2.0 * tp2) / den2, 0.0)
mean_f1 = float(f1_per_class.mean())
print("Validation mean per-class F1 (approx proxy):", mean_f1)

del Xn_tr, Xn_va, val_pred_proba, val_pred_bin, tp, pp, fp, fn, den, f1
gc.collect()




## === cell 9
Xf_test = extract_features(
    imglist_test, batch_size=128, verbose=1, cache_path=os.path.join(cache_dir, "test")
)
Xn_test = scaler.transform(Xf_test)

testKaggle_ppredscore1 = np.zeros((Xn_test.shape[0], len(tagnames)), dtype=np.float32)

if idx_const.size:
    priors = Y_tr[:, idx_const].mean(axis=0).astype(np.float32)
    testKaggle_ppredscore1[:, idx_const] = priors[None, :]

if idx_fit.size:
    proba_list_test = clf_multi.predict_proba(Xn_test)
    for k, i in enumerate(idx_fit):
        testKaggle_ppredscore1[:, i] = proba_list_test[k][:, 1].astype(
            np.float32, copy=False
        )

print("Test proba matrix:", testKaggle_ppredscore1.shape)

del Xf_test, Xn_test
gc.collect()




## === cell 10
def class2tags(classes, tagnames):
    classes = np.asarray(classes, dtype=bool)
    ii, jj = np.nonzero(classes)
    out = [""] * classes.shape[0]
    if ii.size:
        starts = np.r_[0, np.flatnonzero(np.diff(ii)) + 1]
        ends = np.r_[starts[1:], ii.size]
        for s, e in zip(starts, ends):
            r = ii[s]
            out[r] = " ".join(tagnames[jj[s:e]])
    return out




## === cell 11
test_predclass = testKaggle_ppredscore1 >= best_thr.reshape(1, -1)
test_predtags = class2tags(test_predclass, tagnames)

test_predtags = [t if len(t) else "healthy" for t in test_predtags]

del test_predclass, testKaggle_ppredscore1
gc.collect()




## === cell 12
sub = pd.read_csv(SAMPLE_SUB_PATH)

test_images = [os.path.basename(p) for p in imglist_test]
pred_map = dict(zip(test_images, test_predtags))

sub["labels"] = sub["image"].map(pred_map).fillna("healthy")

sub["image"] = sub["image"].astype(str)
sub["labels"] = sub["labels"].astype(str)

sub.head()




## === cell 13
out_path = "./submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote", out_path, "with shape", sub.shape)
print(sub.head())
