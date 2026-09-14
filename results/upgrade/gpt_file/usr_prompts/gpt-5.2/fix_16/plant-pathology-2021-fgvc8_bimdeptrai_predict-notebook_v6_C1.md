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

TFDATA_OPTIONS = tf.data.Options()
TFDATA_OPTIONS.experimental_optimization.apply_default_optimizations = True
TFDATA_OPTIONS.experimental_optimization.map_parallelization = True
TFDATA_OPTIONS.experimental_optimization.parallel_batch = True
TFDATA_OPTIONS.experimental_slack = True
TFDATA_OPTIONS.deterministic = True

tf.config.run_functions_eagerly(False)

print("TF version:", tf.__version__)



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
h_target = 512
w_target = 512



## === cell 3
label_split = train.labels.apply(lambda x: x.split())
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(label_split)
class_names = list(mlb.classes_)

num_classes = y.shape[1]
print("Num classes:", num_classes)
print("Classes:", class_names)



## === cell 4
from sklearn.model_selection import train_test_split

train_df = train.copy()
train_df["labels_list"] = label_split

train_idx, val_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.15, random_state=SEED, shuffle=True
)

df_train = train_df.iloc[train_idx].reset_index(drop=True)
df_val = train_df.iloc[val_idx].reset_index(drop=True)

y_train = y[train_idx]
y_val = y[val_idx]

print(df_train.shape, df_val.shape, y_train.shape, y_val.shape)



## === cell 5
BATCH_SIZE = 16

train_steps = int(np.ceil(len(df_train) / BATCH_SIZE))
val_steps = int(np.ceil(len(df_val) / BATCH_SIZE))

AUTOTUNE = tf.data.AUTOTUNE

H_T = tf.constant(h_target, tf.int32)
W_T = tf.constant(w_target, tf.int32)
SEED_T = tf.constant(SEED, tf.int64)
PI_T = tf.constant(np.pi, tf.float32)


def _stable_path_hash_u31(paths_np: np.ndarray) -> np.ndarray:
    out = np.empty((len(paths_np),), dtype=np.int64)
    fnv_offset = np.uint64(14695981039346656037)
    fnv_prime = np.uint64(1099511628211)
    mask31 = np.uint64((1 << 31) - 1)
    for i, s in enumerate(paths_np.tolist()):
        h = fnv_offset
        b = s.encode("utf-8")
        for c in b:
            h ^= np.uint64(c)
            h *= fnv_prime
        out[i] = np.int64(h & mask31)
    return out


def _fold_in(seed_vec, salt):
    seed_vec = tf.cast(seed_vec, tf.int64)
    salt = tf.cast(salt, tf.int64)
    return tf.stack([seed_vec[0] + salt * 1000003, seed_vec[1] + salt * 9176])


def _decode_and_resize(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img = tf.image.resize(
        img,
        [H_T, W_T],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    img.set_shape([h_target, w_target, 3])
    return img


def _augment_then_resize(img, seed):
    seed = tf.cast(seed, tf.int64)

    s1 = _fold_in(seed, 1)
    img = tf.image.stateless_random_flip_left_right(img, seed=tf.cast(s1, tf.int32))

    max_dx = tf.cast(tf.round(0.05 * tf.cast(W_T, tf.float32)), tf.int32)
    max_dy = tf.cast(tf.round(0.05 * tf.cast(H_T, tf.float32)), tf.int32)
    pad_x = max_dx
    pad_y = max_dy

    img_pad = tf.pad(img, [[pad_y, pad_y], [pad_x, pad_x], [0, 0]], mode="REFLECT")

    s2 = _fold_in(seed, 2)
    img = tf.image.stateless_random_crop(
        img_pad, size=[H_T, W_T, 3], seed=tf.cast(s2, tf.int32)
    )

    s3 = _fold_in(seed, 3)
    z = tf.random.stateless_uniform(
        (), seed=tf.cast(s3, tf.int32), minval=0.9, maxval=1.1, dtype=tf.float32
    )
    new_h = tf.cast(tf.round(tf.cast(H_T, tf.float32) / z), tf.int32)
    new_w = tf.cast(tf.round(tf.cast(W_T, tf.float32) / z), tf.int32)
    new_h = tf.clip_by_value(new_h, 1, H_T * 2)
    new_w = tf.clip_by_value(new_w, 1, W_T * 2)

    def _zoom_in():
        s = _fold_in(seed, 4)
        cropped = tf.image.stateless_random_crop(
            img, [new_h, new_w, 3], seed=tf.cast(s, tf.int32)
        )
        return tf.image.resize(
            cropped,
            [H_T, W_T],
            method=tf.image.ResizeMethod.BILINEAR,
            antialias=False,
        )

    def _zoom_out():
        pad_h = tf.maximum(0, new_h - H_T)
        pad_w = tf.maximum(0, new_w - W_T)
        p_top = pad_h // 2
        p_bottom = pad_h - p_top
        p_left = pad_w // 2
        p_right = pad_w - p_left
        padded = tf.pad(
            img, [[p_top, p_bottom], [p_left, p_right], [0, 0]], mode="REFLECT"
        )
        padded = tf.image.resize(
            padded,
            [new_h, new_w],
            method=tf.image.ResizeMethod.BILINEAR,
            antialias=False,
        )
        s = _fold_in(seed, 5)
        cropped = tf.image.stateless_random_crop(
            padded, [H_T, W_T, 3], seed=tf.cast(s, tf.int32)
        )
        return cropped

    img = tf.cond(z >= 1.0, _zoom_in, _zoom_out)

    angle = tf.constant(15.0, dtype=tf.float32) * PI_T / tf.constant(180.0, tf.float32)
    s4 = _fold_in(seed, 6)
    theta = tf.random.stateless_uniform(
        (), seed=tf.cast(s4, tf.int32), minval=-angle, maxval=angle, dtype=tf.float32
    )

    cos_t = tf.cos(theta)
    sin_t = tf.sin(theta)
    cx = (tf.cast(W_T, tf.float32) - 1.0) / 2.0
    cy = (tf.cast(H_T, tf.float32) - 1.0) / 2.0

    a0 = cos_t
    a1 = -sin_t
    a2 = cx - cos_t * cx + sin_t * cy
    b0 = sin_t
    b1 = cos_t
    b2 = cy - sin_t * cx - cos_t * cy

    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])[None, :]

    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[None, ...],
        transforms=transform,
        output_shape=tf.stack([H_T, W_T]),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )[0]
    img = tf.clip_by_value(img, 0.0, 1.0)
    img.set_shape([h_target, w_target, 3])
    return img


@tf.function(reduce_retracing=True)
def _train_map(path, path_hash, label):
    img = _decode_and_resize(path)
    seed = tf.stack([SEED_T, tf.cast(path_hash, tf.int64)])
    img = _augment_then_resize(img, seed)
    return img, tf.cast(label, tf.float32)


@tf.function(reduce_retracing=True)
def _val_map(path, label):
    img = _decode_and_resize(path)
    return img, tf.cast(label, tf.float32)


@tf.function(reduce_retracing=True)
def _test_map(path):
    return _decode_and_resize(path)


def _make_train_ds(df, y_arr):
    filenames_np = df["image"].to_numpy(dtype=str)
    full_paths_np = np.char.add(TRAIN_IMG_DIR + os.sep, filenames_np).astype(str)

    path_hash_np = _stable_path_hash_u31(full_paths_np)

    paths = tf.convert_to_tensor(full_paths_np, dtype=tf.string)
    path_hash = tf.convert_to_tensor(path_hash_np, dtype=tf.int64)
    labels = tf.convert_to_tensor(y_arr, dtype=tf.int64)

    ds = tf.data.Dataset.from_tensor_slices((paths, path_hash, labels)).with_options(
        TFDATA_OPTIONS
    )

    shuffle_buf = int(min(len(df), 2048))
    ds = ds.shuffle(buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(_train_map, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def _make_val_ds(df, y_arr):
    filenames_np = df["image"].to_numpy(dtype=str)
    full_paths_np = np.char.add(TRAIN_IMG_DIR + os.sep, filenames_np).astype(str)

    paths = tf.convert_to_tensor(full_paths_np, dtype=tf.string)
    labels = tf.convert_to_tensor(y_arr, dtype=tf.int64)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels)).with_options(
        TFDATA_OPTIONS
    )
    ds = ds.map(_val_map, num_parallel_calls=AUTOTUNE).cache()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def _make_test_ds(df):
    filenames_np = df["image"].to_numpy(dtype=str)
    full_paths_np = np.char.add(TEST_IMG_DIR + os.sep, filenames_np).astype(str)

    paths = tf.convert_to_tensor(full_paths_np, dtype=tf.string)
    ds = tf.data.Dataset.from_tensor_slices(paths).with_options(TFDATA_OPTIONS)
    ds = ds.map(_test_map, num_parallel_calls=AUTOTUNE).cache()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = _make_train_ds(df_train, y_train)
val_ds = _make_val_ds(df_val, y_val)
test_ds = _make_test_ds(submissions)

print("Datasets ready:", train_ds, val_ds, test_ds)



## === cell 6
base = tf.keras.applications.MobileNetV2(
    input_shape=(h_target, w_target, 3), include_top=False, weights="imagenet"
)
base.trainable = False

inputs = keras.Input(shape=(h_target, w_target, 3))
x = inputs
x = x * 255.0
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)

outputs = keras.layers.Dense(num_classes, activation="sigmoid", dtype="float32")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

model.summary()



## === cell 7
EPOCHS = 2

history = model.fit(
    train_ds,
    validation_data=val_ds,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    epochs=EPOCHS,
    verbose=1,
)



## === cell 8
preds = model.predict(test_ds, verbose=1)
print(preds.shape)



## === cell 9
thresh = 0.5

class_names_arr = np.array(class_names)

mask = preds >= thresh
any_pos = mask.any(axis=1)
argmax_idx = preds.argmax(axis=1)

pred_labels = []
for i in range(preds.shape[0]):
    if any_pos[i]:
        chosen = class_names_arr[mask[i]]
        pred_labels.append(" ".join(chosen.tolist()))
    else:
        pred_labels.append(class_names_arr[argmax_idx[i]])

submissions["labels"] = pred_labels
submissions = submissions[["image", "labels"]]
submissions.to_csv("submission.csv", index=False)

print(submissions.head())
print("Wrote submission.csv with", len(submissions), "rows")
print("Unique label strings (sample):", submissions["labels"].value_counts().head(10))
