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
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

tf.config.experimental.enable_op_determinism()

print("TF:", tf.__version__)



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
h_target = 256
w_target = 256
batch_size = 32

label_split = train.labels.apply(lambda x: x.split())
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(label_split)
classes = list(mlb.classes_)
num_classes = len(classes)

print("num_classes:", num_classes)
print("classes:", classes)



## === cell 3
idx = np.arange(len(train))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)

val_frac = 0.1
n_val = int(len(train) * val_frac)
val_idx = idx[:n_val]
trn_idx = idx[n_val:]

train_df = train.iloc[trn_idx].reset_index(drop=True)
val_df = train.iloc[val_idx].reset_index(drop=True)
Y_train = Y[trn_idx].astype(np.float32)
Y_val = Y[val_idx].astype(np.float32)

print("train/val:", train_df.shape, val_df.shape)



## === cell 4
AUTOTUNE = tf.data.AUTOTUNE


def _decode_and_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [h_target, w_target], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape((h_target, w_target, 3))
    return img


def _augment(img, seed):
    seed = tf.convert_to_tensor(seed, dtype=tf.int32)
    img = tf.image.stateless_random_flip_left_right(img, seed=seed)

    seed_r = seed + tf.constant([1, 0], tf.int32)
    angle = tf.random.stateless_uniform([], seed_r, minval=-15.0, maxval=15.0) * (
        np.pi / 180.0
    )

    seed_s = seed + tf.constant([2, 0], tf.int32)
    tx = tf.random.stateless_uniform([], seed_s, minval=-0.05, maxval=0.05) * tf.cast(
        w_target, tf.float32
    )
    ty = tf.random.stateless_uniform(
        [], seed_s + tf.constant([0, 1], tf.int32), minval=-0.05, maxval=0.05
    ) * tf.cast(h_target, tf.float32)

    seed_z = seed + tf.constant([3, 0], tf.int32)
    z = tf.random.stateless_uniform([], seed_z, minval=0.90, maxval=1.10)

    cx = (tf.cast(w_target, tf.float32) - 1.0) / 2.0
    cy = (tf.cast(h_target, tf.float32) - 1.0) / 2.0
    c = tf.math.cos(angle)
    s = tf.math.sin(angle)

    def apply_orig(x, y):
        xr = c * x - s * y + (cx - c * cx + s * cy)
        yr = s * x + c * y + (cy - s * cx - c * cy)
        xt = xr + tx
        yt = yr + ty
        xz = z * xt + (cx - z * cx)
        yz = z * yt + (cy - z * cy)
        return xz, yz

    x0, y0 = apply_orig(0.0, 0.0)
    x1, y1 = apply_orig(1.0, 0.0)
    x2, y2 = apply_orig(0.0, 1.0)

    a0 = x1 - x0
    a1 = x2 - x0
    a2 = x0
    b0 = y1 - y0
    b1 = y2 - y0
    b2 = y0

    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])

    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[None, ...],
        transforms=transform[None, ...],
        output_shape=tf.constant([h_target, w_target], tf.int32),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )[0]
    img.set_shape((h_target, w_target, 3))
    return img


def _with_options(ds):
    opts = tf.data.Options()
    opts.experimental_deterministic = True
    opts.autotune.enabled = True
    return ds.with_options(opts)


def make_train_ds(df, y_arr):
    paths_np = np.array(
        [os.path.join(TRAIN_IMG_DIR, f) for f in df["image"].values], dtype=object
    )
    paths = tf.constant(paths_np)
    labels = tf.constant(y_arr, dtype=tf.float32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    shuffle_buf = int(min(len(df), 4096))
    ds = ds.shuffle(buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True)

    def _decode_map(path, y):
        img = _decode_and_resize(path)
        return path, img, y

    ds = ds.map(_decode_map, num_parallel_calls=AUTOTUNE, deterministic=True)

    def _aug_map(path, img, y):
        h = tf.strings.to_hash_bucket_fast(path, 2**31 - 1)
        seed = tf.stack([tf.cast(h, tf.int32), tf.constant(SEED, tf.int32)])
        img = _augment(img, seed)
        return img, y

    ds = ds.map(_aug_map, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    ds = _with_options(ds)
    return ds


def make_eval_ds(df, img_dir, y_arr=None):
    paths_np = np.array(
        [os.path.join(img_dir, f) for f in df["image"].values], dtype=object
    )
    paths = tf.constant(paths_np)

    if y_arr is None:
        ds = tf.data.Dataset.from_tensor_slices(paths)

        def _map_fn(path):
            img = _decode_and_resize(path)
            return img

        ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)
        ds = ds.cache()
    else:
        labels = tf.constant(y_arr, dtype=tf.float32)
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))

        def _map_fn(path, y):
            img = _decode_and_resize(path)
            return img, y

        ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)
        ds = ds.cache()

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    ds = _with_options(ds)
    return ds


train_ds = make_train_ds(train_df, Y_train)
val_ds = make_eval_ds(val_df, TRAIN_IMG_DIR, Y_val)
test_ds = make_eval_ds(submissions, TEST_IMG_DIR, y_arr=None)



## === cell 5
inputs = keras.Input(shape=(h_target, w_target, 3))
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)
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
    validation_data=val_ds,
    epochs=epochs,
    verbose=1,
)



## === cell 7
preds = model.predict(test_ds, verbose=1)
print("preds shape:", preds.shape)



## === cell 8
thresh = 0.2

above = preds >= thresh
any_above = above.any(axis=1)
argmax_idx = np.argmax(preds, axis=1)

rows = np.arange(preds.shape[0])
above_fixed = above.copy()
above_fixed[~any_above, :] = False
above_fixed[~any_above, argmax_idx[~any_above]] = True

healthy_idx = classes.index("healthy") if "healthy" in classes else None
if healthy_idx is not None:
    counts = above_fixed.sum(axis=1)
    drop_healthy = above_fixed[:, healthy_idx] & (counts > 1)
    above_fixed[drop_healthy, healthy_idx] = False

class_arr = np.array(classes, dtype=object)
pred_labels = [" ".join(class_arr[row_mask]) for row_mask in above_fixed]

submissions_out = submissions.copy()
submissions_out["labels"] = pred_labels
submissions_out.to_csv("submission.csv", index=False)

print(submissions_out.head())
print("Wrote submission.csv with shape:", submissions_out.shape)



## === cell 9
assert os.path.exists("submission.csv")
check = pd.read_csv("submission.csv")
assert list(check.columns) == ["image", "labels"]
assert len(check) == len(submissions_out)
check.head()
