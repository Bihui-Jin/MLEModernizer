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

from sklearn.preprocessing import MultiLabelBinarizer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

tf.config.optimizer.set_jit(True)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TF:", tf.__version__)




## === cell 1
TRAIN_CSV = "../input/plant-pathology-2021-fgvc8/train.csv"
SAMPLE_SUB = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"
TRAIN_DIR = "../input/plant-pathology-2021-fgvc8/train_images"
TEST_DIR = "../input/plant-pathology-2021-fgvc8/test_images"

train = pd.read_csv(TRAIN_CSV)
submissions = pd.read_csv(SAMPLE_SUB)

print(train.shape, submissions.shape)
train.head()




## === cell 2
label_split = train.labels.apply(lambda x: x.split())
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(label_split)
labels = pd.DataFrame(y, columns=mlb.classes_)

print("Classes:", list(mlb.classes_))
labels.head()




## === cell 3
h_target = 256
w_target = 256

batch_size = 64

train_df = train.copy()
for c in mlb.classes_:
    train_df[c] = labels[c].values

idx = np.arange(len(train_df))
np.random.shuffle(idx)
val_size = int(0.1 * len(train_df))
val_idx = idx[:val_size]
trn_idx = idx[val_size:]

trn_df = train_df.iloc[trn_idx].reset_index(drop=True)
val_df = train_df.iloc[val_idx].reset_index(drop=True)

print("Train/Val:", trn_df.shape, val_df.shape)




## === cell 4
AUTOTUNE = tf.data.AUTOTUNE

_PI_OVER_180 = tf.constant(np.pi / 180.0, dtype=tf.float32)
_SEED_I32 = tf.constant(SEED, dtype=tf.int32)
_C01 = tf.constant([0, 1], tf.int32)
_C02 = tf.constant([0, 2], tf.int32)
_C03 = tf.constant([0, 3], tf.int32)
_C04 = tf.constant([0, 4], tf.int32)
_C05 = tf.constant([0, 5], tf.int32)
_TRAIN_DIR_T = tf.constant(TRAIN_DIR, dtype=tf.string)
_TEST_DIR_T = tf.constant(TEST_DIR, dtype=tf.string)


@tf.function
def _decode_resize_only_from_bytes(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(
        img,
        [h_target, w_target],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


@tf.function
def _augment(img, seed2):
    h = tf.cast(h_target, tf.float32)
    w = tf.cast(w_target, tf.float32)

    base_seed = tf.stack([_SEED_I32, tf.cast(seed2, tf.int32)])

    angle = (
        tf.random.stateless_uniform([], base_seed + _C01, -15.0, 15.0) * _PI_OVER_180
    )
    tx = tf.random.stateless_uniform([], base_seed + _C02, -0.05, 0.05) * w
    ty = tf.random.stateless_uniform([], base_seed + _C03, -0.05, 0.05) * h
    zoom = tf.random.stateless_uniform([], base_seed + _C04, 0.9, 1.1)

    cos_a = tf.math.cos(angle) / zoom
    sin_a = tf.math.sin(angle) / zoom

    cx = (w - 1.0) / 2.0
    cy = (h - 1.0) / 2.0

    a0 = cos_a
    a1 = -sin_a
    b0 = sin_a
    b1 = cos_a

    a2 = cx - a0 * cx - a1 * cy - tx
    b2 = cy - b0 * cx - b1 * cy - ty

    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])[None, :]

    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[None, ...],
        transforms=transform,
        output_shape=[h_target, w_target],
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )[0]

    rnd = tf.random.stateless_uniform([], base_seed + _C05, 0.0, 1.0)
    img = tf.cond(rnd < 0.5, lambda: tf.image.flip_left_right(img), lambda: img)
    return img


@tf.function
def _read_bytes(path):
    return tf.io.read_file(path)


@tf.function
def _train_map_from_bytes(img_bytes, y, seed2):
    img = _decode_resize_only_from_bytes(img_bytes)
    img = _augment(img, tf.cast(seed2, tf.int32))
    return img, y


@tf.function
def _val_map_from_bytes(img_bytes, y):
    img = _decode_resize_only_from_bytes(img_bytes)
    return img, y


@tf.function
def _test_map_from_bytes(img_bytes):
    return _decode_resize_only_from_bytes(img_bytes)


_DS_OPTIONS = tf.data.Options()
_DS_OPTIONS.experimental_deterministic = False
_DS_OPTIONS.experimental_optimization.apply_default_optimizations = True
try:
    _DS_OPTIONS.threading.private_threadpool_size = 0
    _DS_OPTIONS.threading.max_intra_op_parallelism = 0
except Exception:
    pass


def make_train_ds(df, shuffle=True):
    imgs = tf.constant(df["image"].values.astype(str), dtype=tf.string)
    paths = tf.strings.join([_TRAIN_DIR_T, "/", imgs])
    y_arr = df[list(mlb.classes_)].values.astype(np.float32)
    seed2 = tf.strings.to_hash_bucket_fast(paths, 2**31 - 1)

    ds = tf.data.Dataset.from_tensor_slices((paths, y_arr, seed2))
    ds = ds.with_options(_DS_OPTIONS)

    if shuffle:
        ds = ds.shuffle(
            buffer_size=min(len(df), 4096), seed=SEED, reshuffle_each_iteration=True
        )

    ds = ds.map(
        lambda p, y, s: (_read_bytes(p), y, s),
        num_parallel_calls=AUTOTUNE,
        deterministic=False,
    )
    ds = ds.map(_train_map_from_bytes, num_parallel_calls=AUTOTUNE, deterministic=False)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_val_ds(df):
    imgs = tf.constant(df["image"].values.astype(str), dtype=tf.string)
    paths = tf.strings.join([_TRAIN_DIR_T, "/", imgs])
    y_arr = df[list(mlb.classes_)].values.astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, y_arr))
    ds = ds.with_options(_DS_OPTIONS)

    ds = ds.map(
        lambda p, y: (_read_bytes(p), y),
        num_parallel_calls=AUTOTUNE,
        deterministic=False,
    )
    ds = ds.map(_val_map_from_bytes, num_parallel_calls=AUTOTUNE, deterministic=False)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_test_ds(df):
    imgs = tf.constant(df["image"].values.astype(str), dtype=tf.string)
    paths = tf.strings.join([_TEST_DIR_T, "/", imgs])

    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.with_options(_DS_OPTIONS)

    ds = ds.map(_read_bytes, num_parallel_calls=AUTOTUNE, deterministic=False)
    ds = ds.map(_test_map_from_bytes, num_parallel_calls=AUTOTUNE, deterministic=False)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_train_ds(trn_df, shuffle=True)
val_ds = make_val_ds(val_df)
test_ds = make_test_ds(submissions)

train_steps = int(np.ceil(len(trn_df) / batch_size))
val_steps = int(np.ceil(len(val_df) / batch_size))
test_steps = int(np.ceil(len(submissions) / batch_size))

print("Steps train/val/test:", train_steps, val_steps, test_steps)




## === cell 5
inputs = keras.Input(shape=(h_target, w_target, 3))
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.3)(x)
outputs = keras.layers.Dense(len(mlb.classes_), activation="sigmoid")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    steps_per_execution=20,
)

model.summary()




## === cell 6
EPOCHS = 3

train_ds_rep = train_ds.repeat()
val_ds_rep = val_ds.repeat()

history = model.fit(
    train_ds_rep,
    validation_data=val_ds_rep,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    epochs=EPOCHS,
    verbose=1,
)




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
class_list = list(thresh.keys())
assert set(class_list) == set(
    mlb.classes_
), "Threshold keys must match class order/classes."

thr_vec = np.array([thresh[c] for c in class_list], dtype=np.float32)
above = preds > thr_vec[None, :]
argm = preds.argmax(axis=1)

rows_no = ~above.any(axis=1)
above[rows_no, :] = False
above[rows_no, argm[rows_no]] = True

healthy_idx = class_list.index("healthy")
multi = above.sum(axis=1) > 1
above[multi, healthy_idx] = False

pred_labels = [
    " ".join(np.array(class_list, dtype=object)[row].tolist()) for row in above
]

submission = submissions.copy()
submission["labels"] = pred_labels

submission = submission[["image", "labels"]]
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with", len(submission), "rows")

assert os.path.exists("submission.csv")
chk = pd.read_csv("submission.csv")
assert list(chk.columns) == ["image", "labels"]
assert len(chk) == len(submissions)
chk.tail()
