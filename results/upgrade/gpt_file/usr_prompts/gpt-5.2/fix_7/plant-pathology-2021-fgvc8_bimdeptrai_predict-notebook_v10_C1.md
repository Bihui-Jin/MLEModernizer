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

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TF:", tf.__version__)
print("Eager:", tf.executing_eagerly())




## === cell 1
BASE_PATH = "../input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")
TRAIN_DIR = os.path.join(BASE_PATH, "train_images")
TEST_DIR = os.path.join(BASE_PATH, "test_images")

train = pd.read_csv(TRAIN_CSV)
submissions = pd.read_csv(SAMPLE_SUB)

print(train.shape, submissions.shape)
train.head()




## === cell 2
h_target = 256
w_target = 256
batch_size = 32

label_split = train.labels.apply(lambda x: x.split())
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(label_split)
classes = list(mlb.classes_)

print("Num classes:", len(classes))
print("Classes:", classes)

for j, c in enumerate(classes):
    train[c] = y[:, j].astype(np.float32)

train.head()




## === cell 3
idx = np.arange(len(train))
rng = np.random.default_rng(SEED)
rng.shuffle(idx)
val_size = int(0.15 * len(train))
val_idx = idx[:val_size]
trn_idx = idx[val_size:]

train_df = train.iloc[trn_idx].reset_index(drop=True)
val_df = train.iloc[val_idx].reset_index(drop=True)

print("Train/Val:", train_df.shape, val_df.shape)




## === cell 4
AUTOTUNE = tf.data.AUTOTUNE

x_col = "image"
y_cols = classes

SEED_T = tf.constant(SEED, tf.int32)
C1 = tf.constant([1, 0], tf.int32)
C2 = tf.constant([2, 0], tf.int32)
C3 = tf.constant([3, 0], tf.int32)
C4 = tf.constant([4, 0], tf.int32)
C5 = tf.constant([5, 0], tf.int32)
PI_OVER_180 = tf.constant(np.pi / 180.0, tf.float32)

MAX_DX = tf.constant(int(round(0.05 * w_target)), tf.int32)
MAX_DY = tf.constant(int(round(0.05 * h_target)), tf.int32)


def _read_decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(
        img, [h_target, w_target], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) / 255.0  # rescale=1/255
    return img


def _nearest_translate(img, dx, dy):
    dx = tf.cast(dx, tf.int32)
    dy = tf.cast(dy, tf.int32)
    return tf.roll(img, shift=[dy, dx], axis=[0, 1])


def _nearest_zoom(img, zoom):
    zoom = tf.cast(zoom, tf.float32)
    new_h = tf.cast(tf.round(tf.cast(h_target, tf.float32) * zoom), tf.int32)
    new_w = tf.cast(tf.round(tf.cast(w_target, tf.float32) * zoom), tf.int32)
    z = tf.image.resize(img, [new_h, new_w], method=tf.image.ResizeMethod.BILINEAR)
    z = tf.image.resize_with_crop_or_pad(z, h_target, w_target)
    return z


def _rotate_nearest(img, radians):
    try:
        return tf.image.rotate(
            img, radians, interpolation="BILINEAR", fill_mode="NEAREST"
        )
    except Exception:
        return img


def _augment(img, seed_pair):
    seed_pair = tf.cast(seed_pair, tf.int32)

    flip_r = tf.random.stateless_uniform([], seed=seed_pair + C1)
    img = tf.cond(flip_r < 0.5, lambda: tf.image.flip_left_right(img), lambda: img)

    ang = (
        tf.random.stateless_uniform([], seed=seed_pair + C2, minval=-15.0, maxval=15.0)
        * PI_OVER_180
    )
    img = _rotate_nearest(img, ang)

    dx = tf.random.stateless_uniform(
        [],
        seed=seed_pair + C3,
        minval=-MAX_DX,
        maxval=MAX_DX + 1,
        dtype=tf.int32,
    )
    dy = tf.random.stateless_uniform(
        [],
        seed=seed_pair + C4,
        minval=-MAX_DY,
        maxval=MAX_DY + 1,
        dtype=tf.int32,
    )
    img = _nearest_translate(img, dx, dy)

    zoom = tf.random.stateless_uniform([], seed=seed_pair + C5, minval=0.9, maxval=1.1)
    img = _nearest_zoom(img, zoom)

    return img


def make_train_ds(df, training, snapshot_name=None):
    img_names = df[x_col].astype("string").to_numpy(dtype=str)
    paths = np.asarray([os.path.join(TRAIN_DIR, n) for n in img_names], dtype=str)
    labels = df[y_cols].to_numpy(dtype=np.float32, copy=False)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    ds = ds.with_options(options)

    if training:
        buf = min(len(df), 4096)
        ds = ds.shuffle(buffer_size=buf, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(
        lambda p, y: (_read_decode_resize(p), y),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    if snapshot_name is not None:
        snap_dir = os.path.join("/kaggle/working", snapshot_name)
        ds = ds.apply(tf.data.experimental.snapshot(snap_dir))

    ds = ds.cache()

    if training:
        ds = ds.enumerate()

        def _aug_map(i, xy):
            img, y = xy
            img = _augment(img, tf.stack([SEED_T, tf.cast(i, tf.int32)]))
            return img, y

        ds = ds.map(_aug_map, num_parallel_calls=AUTOTUNE, deterministic=True)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_test_ds(df, snapshot_name=None):
    img_names = df["image"].astype("string").to_numpy(dtype=str)
    paths = np.asarray([os.path.join(TEST_DIR, n) for n in img_names], dtype=str)

    ds = tf.data.Dataset.from_tensor_slices(paths)

    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    ds = ds.with_options(options)

    ds = ds.map(_read_decode_resize, num_parallel_calls=AUTOTUNE, deterministic=True)
    if snapshot_name is not None:
        snap_dir = os.path.join("/kaggle/working", snapshot_name)
        ds = ds.apply(tf.data.experimental.snapshot(snap_dir))
    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_train_ds(train_df, training=True, snapshot_name="snap_train_decode_256")
val_ds = make_train_ds(val_df, training=False, snapshot_name="snap_val_decode_256")
test_ds = make_test_ds(submissions, snapshot_name="snap_test_decode_256")

train_gen = train_ds
val_gen = val_ds
test_gen = test_ds




## === cell 5
inputs = keras.Input(shape=(h_target, w_target, 3))
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.3)(x)
outputs = keras.layers.Dense(len(classes), activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

model.summary()




## === cell 6
epochs = 3
history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, verbose=1)




## === cell 7
preds = model.predict(test_gen, verbose=1)
print("Preds shape:", preds.shape)




## === cell 8
thresh = 0.5  # keep identical thresholding logic

preds_np = np.asarray(preds)
chosen_mask = preds_np >= thresh

empty = ~chosen_mask.any(axis=1)
if empty.any():
    argm = preds_np[empty].argmax(axis=1)
    chosen_mask[empty, :] = False
    chosen_mask[empty, argm] = True

if "healthy" in classes:
    healthy_idx = classes.index("healthy")
    has_healthy = chosen_mask[:, healthy_idx]
    more_than_one = chosen_mask.sum(axis=1) > 1
    drop = has_healthy & more_than_one
    chosen_mask[drop, healthy_idx] = False

classes_arr = np.array(classes, dtype=object)
pred_labels = [" ".join(classes_arr[row_mask]) for row_mask in chosen_mask]

submissions["labels"] = pred_labels
submissions.head()




## === cell 9
submissions.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submissions.shape)
print(submissions.head(10))




## === cell 10
submissions
