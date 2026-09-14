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
import random
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras as keras

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

print("TF version:", tf.__version__)

tf.config.optimizer.set_jit(False)  # keep deterministic/portable behavior



## === cell 1
DATA_DIR = "../input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

train = pd.read_csv(TRAIN_CSV)
submissions = pd.read_csv(SAMPLE_SUB_CSV)

print(train.shape, submissions.shape)
train.head()



## === cell 2
from sklearn.preprocessing import MultiLabelBinarizer

label_split = train.labels.apply(lambda x: x.split())
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(label_split)
class_names = list(mlb.classes_)

labels_df = pd.DataFrame(y, columns=class_names)
print("Classes:", class_names)
labels_df.head()



## === cell 3
h_target = 256
w_target = 256
batch_size = 32

from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(
    train, test_size=0.15, random_state=SEED, shuffle=True
)

print("train_df:", train_df.shape, "val_df:", val_df.shape)



## === cell 4
AUTOTUNE = tf.data.AUTOTUNE

train_df = train_df.copy()
val_df = val_df.copy()

train_y = mlb.transform(train_df.labels.apply(lambda x: x.split())).astype(np.float32)
val_y = mlb.transform(val_df.labels.apply(lambda x: x.split())).astype(np.float32)

train_paths = (TRAIN_IMG_DIR + "/" + train_df["image"].values).astype(str)
val_paths = (TRAIN_IMG_DIR + "/" + val_df["image"].values).astype(str)
test_paths = (TEST_IMG_DIR + "/" + submissions["image"].values).astype(str)

_INV_255 = tf.constant(1.0 / 255.0, dtype=tf.float32)
_PI_OVER_180 = tf.constant(np.pi / 180.0, dtype=tf.float32)


@tf.function(reduce_retracing=True)
def _decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(
        img, [h_target, w_target], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) * _INV_255
    return img


_BASE_SEED = tf.constant([SEED, 0], dtype=tf.int32)


@tf.function(reduce_retracing=True)
def _augment(img, idx):
    idx = tf.cast(idx, tf.int32)
    s0 = _BASE_SEED + tf.stack([0, idx])
    s1 = _BASE_SEED + tf.stack([1, idx])
    s2 = _BASE_SEED + tf.stack([2, idx])
    s3 = _BASE_SEED + tf.stack([3, idx])
    s4 = _BASE_SEED + tf.stack([4, idx])

    img = tf.image.stateless_random_flip_left_right(img, seed=s0)

    max_dx = tf.cast(tf.round(0.05 * tf.cast(w_target, tf.float32)), tf.int32)
    max_dy = tf.cast(tf.round(0.05 * tf.cast(h_target, tf.float32)), tf.int32)
    dx = tf.random.stateless_uniform(
        [], seed=s1, minval=-max_dx, maxval=max_dx + 1, dtype=tf.int32
    )
    dy = tf.random.stateless_uniform(
        [], seed=s2, minval=-max_dy, maxval=max_dy + 1, dtype=tf.int32
    )
    img = tf.roll(img, shift=[dy, dx], axis=[0, 1])

    scale = tf.random.stateless_uniform(
        [], seed=s3, minval=0.9, maxval=1.1, dtype=tf.float32
    )
    new_h = tf.cast(tf.round(scale * tf.cast(h_target, tf.float32)), tf.int32)
    new_w = tf.cast(tf.round(scale * tf.cast(w_target, tf.float32)), tf.int32)
    img2 = tf.image.resize(img, [new_h, new_w], method=tf.image.ResizeMethod.BILINEAR)
    img2 = tf.image.resize_with_crop_or_pad(img2, h_target, w_target)

    angle = (
        tf.random.stateless_uniform(
            [], seed=s4, minval=-15.0, maxval=15.0, dtype=tf.float32
        )
        * _PI_OVER_180
    )
    cos_a = tf.cos(angle)
    sin_a = tf.sin(angle)
    cx = (tf.cast(w_target, tf.float32) - 1.0) / 2.0
    cy = (tf.cast(h_target, tf.float32) - 1.0) / 2.0

    a0 = cos_a
    a1 = -sin_a
    a2 = cx - cos_a * cx + sin_a * cy
    b0 = sin_a
    b1 = cos_a
    b2 = cy - sin_a * cx - cos_a * cy
    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])[tf.newaxis, :]
    img3 = tf.raw_ops.ImageProjectiveTransformV3(
        images=img2[tf.newaxis, ...],
        transforms=transform,
        output_shape=tf.constant([h_target, w_target], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )[0]
    return img3


options = tf.data.Options()
options.experimental_deterministic = True
options.threading.private_threadpool_size = max(8, (os.cpu_count() or 8))
options.threading.max_intra_op_parallelism = (
    1  # helps reduce contention in input pipeline
)

shuffle_buf = min(len(train_paths), 4096)

CACHE_DIR = os.path.join(".", "tf_cache_pp2021")
os.makedirs(CACHE_DIR, exist_ok=True)
train_cache_file = os.path.join(CACHE_DIR, "train_decode.cache")
val_cache_file = os.path.join(CACHE_DIR, "val_decode.cache")
test_cache_file = os.path.join(CACHE_DIR, "test_decode.cache")

train_base = tf.data.Dataset.from_tensor_slices((train_paths, train_y)).with_options(
    options
)
train_base = train_base.shuffle(
    buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True
)

train_decoded = train_base.map(
    lambda p, y_: (_decode_resize(p), y_),
    num_parallel_calls=AUTOTUNE,
    deterministic=True,
).cache(train_cache_file)

train_ds = train_decoded.enumerate(start=0).map(
    lambda idx, img_y: (_augment(img_y[0], idx), img_y[1]),
    num_parallel_calls=AUTOTUNE,
    deterministic=True,
)
train_ds = train_ds.apply(tf.data.experimental.ignore_errors())
train_ds = train_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_y)).with_options(options)
val_ds = val_ds.map(
    lambda p, y_: (_decode_resize(p), y_),
    num_parallel_calls=AUTOTUNE,
    deterministic=True,
).cache(val_cache_file)
val_ds = val_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices(test_paths).with_options(options)
test_ds = test_ds.map(
    _decode_resize, num_parallel_calls=AUTOTUNE, deterministic=True
).cache(test_cache_file)
test_ds = test_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)


def _warm_cache(ds, max_batches=None):
    it = iter(ds)
    n = 0
    try:
        while True:
            _ = next(it)
            n += 1
            if max_batches is not None and n >= max_batches:
                break
    except StopIteration:
        pass


_warm_cache(train_decoded.batch(batch_size), max_batches=None)
_warm_cache(val_ds, max_batches=None)
_warm_cache(test_ds, max_batches=None)



## === cell 5
inputs = keras.Input(shape=(h_target, w_target, 3))
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.3)(x)
outputs = keras.layers.Dense(len(class_names), activation="sigmoid")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    steps_per_execution=32,
)

model.summary()



## === cell 6
EPOCHS = 5

history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)



## === cell 7
preds = model.predict(test_ds, verbose=1)
print("preds shape:", preds.shape)



## === cell 8
thresh = {
    "complex": 0.25,
    "frog_eye_leaf_spot": 0.25,
    "healthy": 0.25,
    "powdery_mildew": 0.25,
    "rust": 0.25,
    "scab": 0.25,
}

thresh_arr = np.array([thresh[c] for c in class_names], dtype=np.float32)



## === cell 9
preds_np = np.asarray(preds)
mask = preds_np >= thresh_arr[None, :]

argmax_idx = preds_np.argmax(axis=1)
healthy_idx = class_names.index("healthy") if "healthy" in class_names else None

chosen_lists = [np.flatnonzero(row) for row in mask]

pred_labels = []
if healthy_idx is None:
    for i, chosen_idx in enumerate(chosen_lists):
        if chosen_idx.size == 0:
            chosen_idx = np.array([argmax_idx[i]], dtype=np.int64)
        pred_labels.append(" ".join([class_names[j] for j in chosen_idx]))
else:
    for i, chosen_idx in enumerate(chosen_lists):
        if chosen_idx.size == 0:
            chosen_idx = np.array([argmax_idx[i]], dtype=np.int64)
        if argmax_idx[i] == healthy_idx:
            pred_labels.append("healthy")
        else:
            pred_labels.append(" ".join([class_names[j] for j in chosen_idx]))

submissions = submissions.copy()
submissions["labels"] = pred_labels

submissions.head()



## === cell 10
out_path = "submission.csv"
submissions[["image", "labels"]].to_csv(out_path, index=False)
print("Wrote:", out_path, "shape:", submissions.shape)



## === cell 11
assert os.path.exists("submission.csv")
chk = pd.read_csv("submission.csv")
assert list(chk.columns) == ["image", "labels"]
assert len(chk) == len(submissions)
print(chk.head())
