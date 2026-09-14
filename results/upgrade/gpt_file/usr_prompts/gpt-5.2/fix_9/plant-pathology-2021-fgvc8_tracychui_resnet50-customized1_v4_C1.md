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

K.set_image_data_format("channels_last")

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

try:
    tf.config.threading.set_intra_op_parallelism_threads(max(1, os.cpu_count() // 2))
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

print("keras:", keras.__version__, "tf:", tf.__version__)




## === cell 1
IMG_WIDTH = 300
IMG_HEIGHT = 300
NR_CHANNELS = 3
TOTAL_INPUTS = NR_CHANNELS * IMG_HEIGHT * IMG_WIDTH




## === cell 2
DATA_DIR = "../input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing dir: {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing dir: {TEST_IMG_DIR}"

sample_df = pd.read_csv(SAMPLE_SUB)
imglist_test = [os.path.join(TEST_IMG_DIR, fn) for fn in sample_df["image"].tolist()]
print("num test images:", len(imglist_test))
print("first test path:", imglist_test[0])




## === cell 3
import tensorflow.keras.applications.resnet50 as resnet
from tensorflow.keras.preprocessing import image

AUTOTUNE = tf.data.AUTOTUNE

BATCH_SIZE = 32

CACHE_DIR = "./tf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


@tf.function(reduce_retracing=True)
def _read_decode_x(path):
    bytes_ = tf.io.read_file(path)
    img = tf.io.decode_jpeg(bytes_, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(
        img, [IMG_HEIGHT, IMG_WIDTH], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32)
    img = resnet.preprocess_input(img)
    img = tf.ensure_shape(img, (IMG_HEIGHT, IMG_WIDTH, NR_CHANNELS))
    return img


test_paths = tf.constant(imglist_test)
ds_test = tf.data.Dataset.from_tensor_slices(test_paths)

_test_opts = tf.data.Options()
_test_opts.experimental_deterministic = True
_test_opts.experimental_optimization.apply_default_optimizations = True
_test_opts.experimental_optimization.map_parallelization = True
_test_opts.experimental_optimization.map_and_batch_fusion = True
_test_opts.experimental_optimization.parallel_batch = True
ds_test = ds_test.with_options(_test_opts)

test_cache_path = os.path.join(
    CACHE_DIR, f"test_{IMG_HEIGHT}x{IMG_WIDTH}_bs{BATCH_SIZE}.cache"
)
ds_test = (
    ds_test.map(_read_decode_x, num_parallel_calls=AUTOTUNE, deterministic=True)
    .apply(tf.data.experimental.ignore_errors())
    .cache(test_cache_path)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

Xim_test_sample = next(iter(ds_test.take(1)))
print("Xim_test_sample:", tuple(Xim_test_sample.shape), Xim_test_sample.dtype)




## === cell 4
minv = tf.reduce_min(Xim_test_sample[0])
maxv = tf.reduce_max(Xim_test_sample[0])
meanv = tf.reduce_mean(Xim_test_sample[0])
print(
    "Sample pixel stats:",
    float(minv.numpy()),
    float(maxv.numpy()),
    float(meanv.numpy()),
)




## === cell 5
train_df = pd.read_csv(TRAIN_CSV)

all_tags = " ".join(train_df["labels"].astype(str).tolist()).split()
tagnames = np.unique(np.array(all_tags, dtype=object))
num_classes = len(tagnames)
print("num_classes:", num_classes)
print("classes:", tagnames)

tag2idx = {t: i for i, t in enumerate(tagnames)}


def encode_labels_matrix(
    labels_series: pd.Series, tag2idx: dict, num_classes: int
) -> np.ndarray:
    rows = []
    cols = []
    for i, s in enumerate(labels_series.astype(str).tolist()):
        for t in s.split():
            j = tag2idx.get(t)
            if j is not None:
                rows.append(i)
                cols.append(j)
    y = np.zeros((len(labels_series), num_classes), dtype=np.float32)
    if rows:
        y[np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32)] = 1.0
    return y


train_img_paths = [os.path.join(TRAIN_IMG_DIR, fn) for fn in train_df["image"].tolist()]
Y = encode_labels_matrix(train_df["labels"], tag2idx, num_classes)

missing = [p for p in train_img_paths if not os.path.exists(p)]
assert len(missing) == 0, f"Missing {len(missing)} training images, e.g. {missing[:3]}"

idx = np.arange(len(train_df))
rng = np.random.default_rng(SEED)
rng.shuffle(idx)
val_frac = 0.1
val_size = int(len(idx) * val_frac)
val_idx = idx[:val_size]
trn_idx = idx[val_size:]

print("train size:", len(trn_idx), "val size:", len(val_idx))


@tf.function(reduce_retracing=True)
def _read_decode(path, y):
    bytes_ = tf.io.read_file(path)
    img = tf.io.decode_jpeg(bytes_, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(
        img, [IMG_HEIGHT, IMG_WIDTH], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32)
    img = resnet.preprocess_input(img)
    img = tf.ensure_shape(img, (IMG_HEIGHT, IMG_WIDTH, NR_CHANNELS))
    y = tf.ensure_shape(y, (num_classes,))
    return img, y


train_img_paths_arr = np.asarray(train_img_paths, dtype=object)
trn_paths = tf.constant(train_img_paths_arr[trn_idx])
trn_y = tf.constant(Y[trn_idx])
val_paths = tf.constant(train_img_paths_arr[val_idx])
val_y = tf.constant(Y[val_idx])

_trn_opts = tf.data.Options()
_trn_opts.experimental_deterministic = True
_trn_opts.experimental_optimization.apply_default_optimizations = True
_trn_opts.experimental_optimization.map_parallelization = True
_trn_opts.experimental_optimization.map_and_batch_fusion = True
_trn_opts.experimental_optimization.parallel_batch = True

_val_opts = tf.data.Options()
_val_opts.experimental_deterministic = True
_val_opts.experimental_optimization.apply_default_optimizations = True
_val_opts.experimental_optimization.map_parallelization = True
_val_opts.experimental_optimization.map_and_batch_fusion = True
_val_opts.experimental_optimization.parallel_batch = True

shuffle_buf = int(min(8192, max(2048, len(trn_idx))))

train_n = len(trn_idx)
val_n = len(val_idx)
steps_per_epoch = int(np.ceil(train_n / BATCH_SIZE))
val_steps = int(np.ceil(val_n / BATCH_SIZE))

ds_trn = tf.data.Dataset.from_tensor_slices((trn_paths, trn_y)).with_options(_trn_opts)
ds_trn = ds_trn.shuffle(shuffle_buf, seed=SEED, reshuffle_each_iteration=True)
ds_trn = ds_trn.repeat()

train_cache_path = os.path.join(
    CACHE_DIR, f"train_{IMG_HEIGHT}x{IMG_WIDTH}_bs{BATCH_SIZE}.cache"
)
ds_trn = (
    ds_trn.map(_read_decode, num_parallel_calls=AUTOTUNE, deterministic=True)
    .apply(tf.data.experimental.ignore_errors())
    .cache(train_cache_path)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(AUTOTUNE)
)

ds_val = tf.data.Dataset.from_tensor_slices((val_paths, val_y)).with_options(_val_opts)
ds_val = (
    ds_val.map(_read_decode, num_parallel_calls=AUTOTUNE, deterministic=True)
    .cache()
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

base = keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_HEIGHT, IMG_WIDTH, NR_CHANNELS),
    pooling="avg",
)
base.trainable = False  # unchanged

inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, NR_CHANNELS))
x = base(inputs, training=False)
outputs = keras.layers.Dense(num_classes, activation="sigmoid")(x)
model_f = keras.Model(inputs, outputs)

model_f.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

EPOCHS = 2
history = model_f.fit(
    ds_trn,
    validation_data=ds_val,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    verbose=2,
)

test_steps = int(np.ceil(len(imglist_test) / BATCH_SIZE))
X_test = model_f.predict(ds_test, steps=test_steps, verbose=1)
print("X_test:", X_test.shape, X_test.dtype)




## === cell 6
print("X_test sample:", X_test[0])




## === cell 7
def class2tags(classes, tagnames):
    tagnames_arr = np.asarray(tagnames, dtype=object)
    out = []
    for row in classes:
        idxs = np.flatnonzero(row)
        if idxs.size:
            out.append(" ".join(tagnames_arr[idxs]))
        else:
            out.append("")
    return out




## === cell 8
test_predclass = X_test > 0.28
test_predtags = class2tags(test_predclass, tagnames)

print("Example predicted tags:", test_predtags[0])




## === cell 9
del test_predclass
gc.collect()




## === cell 10
df1 = sample_df[["image"]].copy()
print(df1.head())




## === cell 11
df2 = pd.DataFrame(test_predtags, columns=["labels"])
print(df2.head())




## === cell 12
sub_df = pd.concat([df1.reset_index(drop=True), df2.reset_index(drop=True)], axis=1)

assert list(sub_df.columns) == ["image", "labels"]
assert len(sub_df) == len(sample_df)

sub_df["labels"] = sub_df["labels"].replace("", "healthy")

print(sub_df.head())

out_path = "./submission.csv"
sub_df.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(sub_df))
print("Submission columns:", sub_df.columns.tolist())
