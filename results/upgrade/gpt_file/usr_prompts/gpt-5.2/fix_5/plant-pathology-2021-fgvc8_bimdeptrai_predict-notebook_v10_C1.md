# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

# 5. Target score

0.1578947368421052

# 6. Current score

0.24507

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.24507) has done: 'The timeout is dominated by slow Python-side image loading/augmentation in `ImageDataGenerator.flow_from_dataframe` plus extra overhead per step (validation each epoch and non-maximal prefetch). To keep identical model/epochs/loss and preserve semantics, I switch the input pipeline to `tf.data` with the same augmentations implemented using TensorFlow ops, enabling parallel decode, caching/prefetch, and larger I/O concurrency while keeping the same train/val split and labels. I also ensure deterministic behavior via stateless RNG keyed by (seed, index) so results remain stable across runs. Finally, prediction and label post-processing be vectorized to remove Python loops without changing thresholding logic.'
- What this solution (achieved 0.24507) has done: 'The timeout is dominated by the input pipeline (JPEG decode/resize + heavy per-image stateless augmentations) and by retracing/py overhead inside `tf.data` maps. I keep the exact same model, loss, epochs, and augmentation semantics, but make the pipeline faster by (1) prebuilding file paths without pandas `apply`, (2) caching the *decoded+resized* images (safe and exactly equivalent) and then applying augmentations on cached tensors, and (3) using a single parallel map with an explicit `num_parallel_calls` and `deterministic=True` to reduce overhead while preserving determinism. I also enable `tf.data` default optimizations explicitly and keep deterministic behavior/seed usage unchanged.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

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




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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

    flip_r = tf.random.stateless_uniform(
        [], seed=seed_pair + tf.constant([1, 0], tf.int32)
    )
    img = tf.cond(flip_r < 0.5, lambda: tf.image.flip_left_right(img), lambda: img)

    ang = tf.random.stateless_uniform(
        [], seed=seed_pair + tf.constant([2, 0], tf.int32), minval=-15.0, maxval=15.0
    ) * (np.pi / 180.0)
    img = _rotate_nearest(img, ang)

    max_dx = int(round(0.05 * w_target))
    max_dy = int(round(0.05 * h_target))
    dx = tf.random.stateless_uniform(
        [],
        seed=seed_pair + tf.constant([3, 0], tf.int32),
        minval=-max_dx,
        maxval=max_dx + 1,
        dtype=tf.int32,
    )
    dy = tf.random.stateless_uniform(
        [],
        seed=seed_pair + tf.constant([4, 0], tf.int32),
        minval=-max_dy,
        maxval=max_dy + 1,
        dtype=tf.int32,
    )
    img = _nearest_translate(img, dx, dy)

    zoom = tf.random.stateless_uniform(
        [], seed=seed_pair + tf.constant([5, 0], tf.int32), minval=0.9, maxval=1.1
    )
    img = _nearest_zoom(img, zoom)

    return img


def make_train_ds(df, training):
    img_names = df[x_col].astype(str).to_numpy()
    paths = np.char.add(TRAIN_DIR + os.sep, img_names)
    labels = df[y_cols].to_numpy(dtype=np.float32, copy=False)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    ds = ds.with_options(options)

    if training:
        ds = ds.shuffle(buffer_size=len(df), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(
        lambda p, y: (_read_decode_resize(p), y),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.cache()

    if training:
        ds = ds.enumerate()

        def _aug_map(i, xy):
            img, y = xy
            img = _augment(
                img, tf.stack([tf.constant(SEED, tf.int32), tf.cast(i, tf.int32)])
            )
            return img, y

        ds = ds.map(_aug_map, num_parallel_calls=AUTOTUNE, deterministic=True)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_test_ds(df):
    img_names = df["image"].astype(str).to_numpy()
    paths = np.char.add(TEST_DIR + os.sep, img_names)

    ds = tf.data.Dataset.from_tensor_slices(paths)

    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    ds = ds.with_options(options)

    ds = ds.map(_read_decode_resize, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_train_ds(train_df, training=True)
val_ds = make_train_ds(val_df, training=False)
test_ds = make_test_ds(submissions)

train_gen = train_ds
val_gen = val_ds
test_gen = test_ds




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1619349872.py in <cell line: 0>()
    143 
    144 
--> 145 train_ds = make_train_ds(train_df, training=True)
    146 val_ds = make_train_ds(val_df, training=False)
    147 test_ds = make_test_ds(submissions)

/tmp/ipykernel_11/1619349872.py in make_train_ds(df, training)
     84     # Build paths without per-row python lambda
     85     img_names = df[x_col].astype(str).to_numpy()
---> 86     paths = np.char.add(TRAIN_DIR + os.sep, img_names)
     87     labels = df[y_cols].to_numpy(dtype=np.float32, copy=False)
     88 

/usr/local/lib/python3.11/dist-packages/numpy/core/defchararray.py in add(x1, x2)
    330         # object dtype itemsize as num chars (worked on short strings).
    331         # bytes + void worked but promoting void->bytes is dubious also.
--> 332         raise TypeError(
    333             "np.char.add() requires both arrays of the same dtype kind, but "
    334             f"got dtypes: '{arr1.dtype}' and '{arr2.dtype}' (the few cases "

TypeError: np.char.add() requires both arrays of the same dtype kind, but got dtypes: '<U49' and 'object' (the few cases where this used to work often lead to incorrect results).

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




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3649304392.py in <cell line: 0>()
      1 epochs = 3
----> 2 history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, verbose=1)
      3 
      4 

NameError: name 'train_gen' is not defined

## === cell 7
preds = model.predict(test_gen, verbose=1)
print("Preds shape:", preds.shape)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3730728747.py in <cell line: 0>()
----> 1 preds = model.predict(test_gen, verbose=1)
      2 print("Preds shape:", preds.shape)
      3 
      4 

NameError: name 'test_gen' is not defined

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




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2189088049.py in <cell line: 0>()
      1 thresh = 0.5  # keep identical thresholding logic
      2 
----> 3 preds_np = np.asarray(preds)
      4 chosen_mask = preds_np >= thresh
      5 

NameError: name 'preds' is not defined

## === cell 9
submissions.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submissions.shape)
print(submissions.head(10))




## === cell 10
submissions
