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
import pickle
import random
import hashlib
import time

import numpy as np
import pandas as pd

from PIL import Image as PILImage

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

K.set_image_data_format("channels_last")
print("keras:", keras.__version__, "tf:", tf.__version__)



## === cell 1
IMG_WIDTH = 300
IMG_HEIGHT = 300
NR_CHANNELS = 3

DATA_ROOT = "../input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing dir: {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing dir: {TEST_IMG_DIR}"



## === cell 2
sub_df = pd.read_csv(SAMPLE_SUB)
test_images = sub_df["image"].astype(str).tolist()
print("sample_submission rows:", len(test_images), "example:", test_images[:3])

test_paths = [os.path.join(TEST_IMG_DIR, img) for img in test_images]
print("Test image dir exists:", os.path.isdir(TEST_IMG_DIR))
print("First path:", test_paths[0])



## === cell 3
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

base = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_HEIGHT, IMG_WIDTH, NR_CHANNELS),
    pooling="avg",
)
model_f = Model(inputs=base.input, outputs=base.output)
model_f.trainable = False

print("Feature dim:", model_f.output_shape)




## === cell 4
def _build_image_dataset(image_paths, batch_size, cache=False):
    paths = tf.constant(image_paths)

    def _load_one(path):
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(
            img, [IMG_HEIGHT, IMG_WIDTH], method=tf.image.ResizeMethod.BILINEAR
        )
        img = tf.cast(img, tf.float32)
        return img

    opts = tf.data.Options()
    opts.experimental_deterministic = True

    ds = tf.data.Dataset.from_tensor_slices(paths).with_options(opts)
    ds = ds.map(_load_one, num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


@tf.function(reduce_retracing=True)
def _feats_step(batch):
    batch = preprocess_input(batch)
    return model_f(batch, training=False)


def _paths_fingerprint(paths):
    h = hashlib.sha1()
    for p in paths:
        h.update(p.encode("utf-8", "ignore"))
        h.update(b"\n")
    h.update(f"{IMG_HEIGHT}x{IMG_WIDTH}x{NR_CHANNELS}".encode())
    h.update(model_f.name.encode())
    h.update(str(model_f.output_shape[-1]).encode())
    return h.hexdigest()


def extract_features(image_paths, batch_size=32, cache_ds=False, cache_dir="./_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    n = len(image_paths)
    fp = _paths_fingerprint(image_paths)
    cache_path = os.path.join(cache_dir, f"resnet50_avg_{fp}.npy")

    if os.path.exists(cache_path):
        feats = np.load(cache_path, mmap_mode="r")
        if feats.shape[0] == n:
            return np.asarray(feats, dtype=np.float32)

    ds = _build_image_dataset(image_paths, batch_size=batch_size, cache=cache_ds)

    feat_dim = int(model_f.output_shape[-1])
    feats = np.empty((n, feat_dim), dtype=np.float32)

    offset = 0
    for X in ds:
        out = _feats_step(X).numpy().astype(np.float32, copy=False)
        bs = out.shape[0]
        feats[offset : offset + bs] = out
        offset += bs

    feats = feats[:n]  # safety; preserves ordering/shape
    np.save(cache_path, feats)
    return feats




## === cell 5
train_df = pd.read_csv(TRAIN_CSV)
train_df["image"] = train_df["image"].astype(str)
train_df["labels"] = train_df["labels"].astype(str)

all_labels = []
for labels in pd.unique(train_df["labels"]):
    all_labels.extend(labels.split())
tagnames = np.unique(np.array(all_labels, dtype=object))

print("Num train:", len(train_df), "Num tags:", len(tagnames))
print("Tags:", tagnames)



## === cell 6
train_paths = [os.path.join(TRAIN_IMG_DIR, img) for img in train_df["image"].tolist()]
for p in train_paths[:5]:
    assert os.path.exists(p), f"Missing train image: {p}"

avail_mask = np.fromiter(
    (os.path.exists(p) for p in test_paths), count=len(test_paths), dtype=bool
)
avail_idx = np.where(avail_mask)[0]
print("Available test images in this runtime:", len(avail_idx), "of", len(test_paths))

all_paths = train_paths + [test_paths[i] for i in avail_idx]

all_feats = extract_features(all_paths, batch_size=64, cache_ds=False)

Xf_train = all_feats[: len(train_paths)]
Xf_test = all_feats[len(train_paths) :]

print("Xf_train:", Xf_train.shape)
print("Xf_test:", Xf_test.shape)



## === cell 7
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
trainXn = scaler.fit_transform(Xf_train)
testXn_avail = scaler.transform(Xf_test)

print("Scaled train/test:", trainXn.shape, testXn_avail.shape)



## === cell 8
tag_to_i = {t: i for i, t in enumerate(tagnames)}
Y = np.zeros((len(train_df), len(tagnames)), dtype=np.int32)

rows = []
cols = []
for r, lbl in enumerate(train_df["labels"].tolist()):
    for t in lbl.split():
        i = tag_to_i.get(t)
        if i is not None:
            rows.append(r)
            cols.append(i)
Y[np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32)] = 1

from joblib import Parallel, delayed


def _fit_one_tag(i, t):
    y = Y[:, i]
    if y.min() == y.max():
        return t, None
    lr = LogisticRegression(
        solver="liblinear",
        max_iter=200,
        random_state=SEED,
    )
    lr.fit(trainXn, y)
    return t, lr


n_jobs = max(1, min(os.cpu_count() or 1, 8))
fit_results = Parallel(n_jobs=n_jobs, backend="threading")(
    delayed(_fit_one_tag)(i, t) for i, t in enumerate(tagnames)
)
tagmodels1 = dict(fit_results)

skipped = [t for t, m in tagmodels1.items() if m is None]
for t in skipped:
    print(f"Tag {t}: skipped (only one class present in y)")

print(
    "Trained models:",
    sum(m is not None for m in tagmodels1.values()),
    "of",
    len(tagnames),
)




## === cell 9
def _pred_one_tag(i, t):
    mdl = tagmodels1[t]
    if mdl is None:
        return i, None
    return i, mdl.predict_proba(testXn_avail)[:, 1].astype(np.float32, copy=False)


testKaggle_ppredscore1_avail = np.zeros(
    (testXn_avail.shape[0], len(tagnames)), dtype=np.float32
)

pred_results = Parallel(n_jobs=n_jobs, backend="threading")(
    delayed(_pred_one_tag)(i, t) for i, t in enumerate(tagnames)
)
for i, col in pred_results:
    if col is not None:
        testKaggle_ppredscore1_avail[:, i] = col

print("Pred score shape:", testKaggle_ppredscore1_avail.shape)




## === cell 10
def class2tags(classes, tagnames):
    tags = []
    for row in classes:
        idx = np.flatnonzero(row)
        if idx.size:
            tags.append(" ".join(tagnames[idx]))
        else:
            tags.append("")
    return tags




## === cell 11
TH = 0.90
test_predclass_avail = testKaggle_ppredscore1_avail > TH
test_predtags_avail = class2tags(test_predclass_avail, tagnames)

test_predtags_avail = [
    s if len(s.strip()) > 0 else "healthy" for s in test_predtags_avail
]

print("Example preds:", test_predtags_avail[:10])



## === cell 12
full_predtags = ["healthy"] * len(test_images)
for out_pos, idx in enumerate(avail_idx):
    full_predtags[idx] = test_predtags_avail[out_pos]

submission = pd.DataFrame({"image": test_images, "labels": full_predtags})
print(submission.head())
print("Submission shape:", submission.shape)



## === cell 13
out_path = "./submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path, "size:", os.path.getsize(out_path), "bytes")
