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

from sklearn.preprocessing import MultiLabelBinarizer

keras = tf.keras

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

tf.config.optimizer.set_jit(True)
tf.config.experimental.enable_op_determinism()

print("TF version:", tf.__version__)
print("Keras version:", tf.keras.__version__)



## === cell 1
DATA_DIR = "../input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

train = pd.read_csv(TRAIN_CSV)
submissions = pd.read_csv(SAMPLE_SUB)

print(train.shape, submissions.shape)
train.head()



## === cell 2
label_split = train["labels"].str.split()

mlb = MultiLabelBinarizer()
y = pd.DataFrame(mlb.fit_transform(label_split), columns=mlb.classes_)

class_names = list(mlb.classes_)
num_classes = len(class_names)

print("Classes:", class_names)
y.head()



## === cell 3
train_df = train.copy()
train_df[class_names] = y.values
train_df.head()



## === cell 4
h_target = 256
w_target = 256
batch_size = 32
val_split = 0.1

AUTOTUNE = tf.data.AUTOTUNE

_HWT = tf.constant([h_target, w_target], dtype=tf.int32)
_PI_OVER_180 = tf.constant(np.pi / 180.0, dtype=tf.float32)
_SEED_TF = tf.constant(SEED, dtype=tf.int32)
_SEED_MIX = tf.constant(1000003, dtype=tf.int32)
_W_F = tf.cast(w_target, tf.float32)
_H_F = tf.cast(h_target, tf.float32)
_CX = (tf.cast(w_target, tf.float32) - 1.0) / 2.0
_CY = (tf.cast(h_target, tf.float32) - 1.0) / 2.0

CACHE_DIR = os.path.join("/kaggle/working", "tfdata_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _build_train_val_split(df, seed=SEED, val_split=0.1):
    n = len(df)
    n_val = int(np.floor(n * val_split))
    val_idx = np.arange(n_val)
    train_idx = np.arange(n_val, n)
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(
        drop=True
    )


train_df_train, train_df_val = _build_train_val_split(
    train_df, seed=SEED, val_split=val_split
)


@tf.function
def _decode_resize_rescale(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img,
        _HWT,
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


@tf.function
def _augment(img, seed):
    seed = tf.cast(seed, tf.int32)
    seed1 = tf.stack([seed, 1])
    seed2 = tf.stack([seed, 2])
    seed3 = tf.stack([seed, 3])
    seed4 = tf.stack([seed, 4])
    seed5 = tf.stack([seed, 5])

    do_flip = tf.random.stateless_uniform([], seed1) < 0.5
    img = tf.cond(do_flip, lambda: tf.image.flip_left_right(img), lambda: img)

    dx = tf.random.stateless_uniform([], seed2, minval=-0.05, maxval=0.05) * _W_F
    dy = tf.random.stateless_uniform([], seed3, minval=-0.05, maxval=0.05) * _H_F
    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[None, ...],
        transforms=tf.stack([1.0, 0.0, -dx, 0.0, 1.0, -dy, 0.0, 0.0])[None, :],
        output_shape=_HWT,
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )[0]

    z = tf.random.stateless_uniform([], seed4, minval=0.9, maxval=1.1)
    a0 = z
    a1 = 0.0
    a2 = _CX - z * _CX
    b0 = 0.0
    b1 = z
    b2 = _CY - z * _CY
    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[None, ...],
        transforms=tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])[None, :],
        output_shape=_HWT,
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )[0]

    angle = (
        tf.random.stateless_uniform([], seed5, minval=-15.0, maxval=15.0) * _PI_OVER_180
    )
    cos_a = tf.math.cos(angle)
    sin_a = tf.math.sin(angle)
    a0 = cos_a
    a1 = -sin_a
    a2 = _CX - cos_a * _CX + sin_a * _CY
    b0 = sin_a
    b1 = cos_a
    b2 = _CY - sin_a * _CX - cos_a * _CY
    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[None, ...],
        transforms=tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])[None, :],
        output_shape=_HWT,
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )[0]

    return img


@tf.function
def _map_train_from_decoded(img, y_, idx):
    s = _SEED_TF * _SEED_MIX + tf.cast(idx, tf.int32)
    img = _augment(img, s)
    return img, y_


@tf.function
def _map_eval_from_decoded(img, y_):
    return img, y_


def _make_ds(df, img_dir, training, shuffle, batch_size, cache_path):
    paths = (img_dir + "/" + df["image"].values).astype(str)
    labels = df[class_names].values.astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    ds = ds.with_options(options)

    ds = ds.map(
        lambda path, y_: (_decode_resize_rescale(path), y_),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    ).cache(cache_path)

    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=SEED, reshuffle_each_iteration=True)

    if training:
        ds = ds.enumerate()
        ds = ds.map(
            lambda idx, inp: _map_train_from_decoded(inp[0], inp[1], idx),
            num_parallel_calls=AUTOTUNE,
            deterministic=True,
        )
    else:
        ds = ds.map(
            lambda img, y_: _map_eval_from_decoded(img, y_),
            num_parallel_calls=AUTOTUNE,
            deterministic=True,
        )

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = _make_ds(
    train_df_train,
    TRAIN_IMG_DIR,
    training=True,
    shuffle=True,
    batch_size=batch_size,
    cache_path=os.path.join(CACHE_DIR, "train_decode.cache"),
)
valid_ds = _make_ds(
    train_df_val,
    TRAIN_IMG_DIR,
    training=False,
    shuffle=False,
    batch_size=batch_size,
    cache_path=os.path.join(CACHE_DIR, "val_decode.cache"),
)

test_paths = (TEST_IMG_DIR + "/" + submissions["image"].values).astype(str)
test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
options = tf.data.Options()
options.experimental_deterministic = True
options.experimental_optimization.apply_default_optimizations = True
options.experimental_optimization.map_parallelization = True
options.experimental_optimization.parallel_batch = True
test_ds = test_ds.with_options(options)

test_ds = (
    test_ds.map(_decode_resize_rescale, num_parallel_calls=AUTOTUNE, deterministic=True)
    .cache(os.path.join(CACHE_DIR, "test_decode.cache"))
    .batch(batch_size, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

train_steps = int(np.ceil(len(train_df_train) / batch_size))
valid_steps = int(np.ceil(len(train_df_val) / batch_size))
test_steps = int(np.ceil(len(submissions) / batch_size))

print("Train/Val/Test sizes:", len(train_df_train), len(train_df_val), len(submissions))
print("Train/Val/Test steps:", train_steps, valid_steps, test_steps)



## === cell 5
inputs = keras.Input(shape=(h_target, w_target, 3))
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.3)(x)
outputs = keras.layers.Dense(num_classes, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

model.summary()



## === cell 6
epochs = 3

history = model.fit(
    train_ds,
    steps_per_epoch=train_steps,
    validation_data=valid_ds,
    validation_steps=valid_steps,
    epochs=epochs,
    verbose=1,
)



## === cell 7
preds = model.predict(
    test_ds,
    steps=test_steps,
    verbose=1,
)
print("Pred shape:", preds.shape)



## === cell 8
default_thr = 0.25
thresh = {c: default_thr for c in class_names}

for k in ["complex", "frog_eye_leaf_spot", "healthy", "powdery_mildew", "rust", "scab"]:
    if k in thresh:
        thresh[k] = 0.25



## === cell 9
submissions = submissions.copy()
thr_vec = np.array([thresh[c] for c in class_names], dtype=np.float32)

p = preds.astype(np.float32, copy=False)
argmax_idx = np.argmax(p, axis=1)

class_arr_obj = np.asarray(class_names, dtype=object)

healthy_idx = class_names.index("healthy") if "healthy" in class_names else None
is_healthy_argmax = (
    (argmax_idx == healthy_idx)
    if healthy_idx is not None
    else np.zeros(len(p), dtype=bool)
)

above = p > thr_vec[None, :]

label_lists = [" ".join(class_arr_obj[m]).strip() for m in above]

labels = np.asarray(label_lists, dtype=object)
has_healthy = (
    above[:, healthy_idx] if healthy_idx is not None else np.zeros(len(p), dtype=bool)
)

needs_fallback = (labels == "") | has_healthy
labels[needs_fallback] = class_arr_obj[argmax_idx[needs_fallback]]
labels[is_healthy_argmax] = "healthy"

submissions["labels"] = labels
submissions.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submissions.shape)
submissions.head()



## === cell 10
assert os.path.exists("submission.csv"), "submission.csv was not created"
sub_check = pd.read_csv("submission.csv")
assert list(sub_check.columns) == [
    "image",
    "labels",
], f"Bad columns: {sub_check.columns.tolist()}"
assert len(sub_check) == len(
    pd.read_csv(SAMPLE_SUB)
), "Row count mismatch vs sample_submission"
print(sub_check.head())
