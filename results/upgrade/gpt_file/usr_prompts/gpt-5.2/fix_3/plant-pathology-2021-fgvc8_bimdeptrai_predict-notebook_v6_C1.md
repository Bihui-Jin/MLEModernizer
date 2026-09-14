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
os.environ["TF_DETERMINISTIC_OPS"] = "1"
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass




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

BATCH_SIZE = 8

train_steps = int(np.ceil(len(df_train) / BATCH_SIZE))
val_steps = int(np.ceil(len(df_val) / BATCH_SIZE))

AUTOTUNE = tf.data.AUTOTUNE


def _read_jpeg(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    return img


def _resize(img):
    return tf.image.resize(
        img,
        [h_target, w_target],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )


def _augment(img, seed):
    seed = tf.cast(seed, tf.int32)

    s1 = tf.random.experimental.stateless_split(seed, 1)[0]
    img = tf.image.stateless_random_flip_left_right(img, seed=s1)

    max_dx = tf.cast(tf.round(0.05 * tf.cast(w_target, tf.float32)), tf.int32)
    max_dy = tf.cast(tf.round(0.05 * tf.cast(h_target, tf.float32)), tf.int32)
    pad_x = max_dx
    pad_y = max_dy
    img_pad = tf.pad(img, [[pad_y, pad_y], [pad_x, pad_x], [0, 0]], mode="REFLECT")
    s2 = tf.random.experimental.stateless_split(seed, 2)[1]
    img = tf.image.stateless_random_crop(img_pad, size=[h_target, w_target, 3], seed=s2)

    s3 = tf.random.experimental.stateless_split(seed, 3)[2]
    z = tf.random.stateless_uniform(
        (), seed=s3, minval=0.9, maxval=1.1, dtype=tf.float32
    )
    new_h = tf.cast(tf.round(tf.cast(h_target, tf.float32) / z), tf.int32)
    new_w = tf.cast(tf.round(tf.cast(w_target, tf.float32) / z), tf.int32)
    new_h = tf.clip_by_value(new_h, 1, h_target * 2)
    new_w = tf.clip_by_value(new_w, 1, w_target * 2)

    def _zoom_in():
        s = tf.random.experimental.stateless_split(seed, 4)[3]
        cropped = tf.image.stateless_random_crop(img, [new_h, new_w, 3], seed=s)
        return tf.image.resize(
            cropped,
            [h_target, w_target],
            method=tf.image.ResizeMethod.BILINEAR,
            antialias=False,
        )

    def _zoom_out():
        pad_h = tf.maximum(0, new_h - h_target)
        pad_w = tf.maximum(0, new_w - w_target)
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
        s = tf.random.experimental.stateless_split(seed, 5)[4]
        cropped = tf.image.stateless_random_crop(
            padded, [h_target, w_target, 3], seed=s
        )
        return cropped

    img = tf.cond(z >= 1.0, _zoom_in, _zoom_out)

    angle = 15.0 * np.pi / 180.0
    s4 = tf.random.experimental.stateless_split(seed, 6)[5]
    theta = tf.random.stateless_uniform(
        (), seed=s4, minval=-angle, maxval=angle, dtype=tf.float32
    )

    cos_t = tf.cos(theta)
    sin_t = tf.sin(theta)
    cx = (w_target - 1) / 2.0
    cy = (h_target - 1) / 2.0

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
        output_shape=tf.constant([h_target, w_target], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )[0]
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img


def _make_train_ds(df, y_arr):
    paths = (
        tf.constant(TRAIN_IMG_DIR) + tf.constant("/") + tf.constant(df["image"].values)
    )
    labels = tf.constant(y_arr, dtype=tf.int64)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    ds = ds.shuffle(buffer_size=len(df), seed=SEED, reshuffle_each_iteration=True)

    idx_ds = tf.data.Dataset.range(len(df), output_type=tf.int64)
    ds = tf.data.Dataset.zip((ds, idx_ds))

    def _map(item, idx):
        path, label = item
        img = _read_jpeg(path)
        img = _resize(img)
        seed = tf.stack([tf.cast(SEED, tf.int64), idx])
        img = _augment(img, seed)
        return img, label

    ds = ds.map(_map, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def _make_val_ds(df, y_arr):
    paths = (
        tf.constant(TRAIN_IMG_DIR) + tf.constant("/") + tf.constant(df["image"].values)
    )
    labels = tf.constant(y_arr, dtype=tf.int64)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _map(path, label):
        img = _read_jpeg(path)
        img = _resize(img)
        return img, label

    ds = ds.map(_map, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def _make_test_ds(df):
    paths = (
        tf.constant(TEST_IMG_DIR) + tf.constant("/") + tf.constant(df["image"].values)
    )
    ds = tf.data.Dataset.from_tensor_slices(paths)

    def _map(path):
        img = _read_jpeg(path)
        img = _resize(img)
        return img

    ds = ds.map(_map, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


train_ds = _make_train_ds(df_train, y_train)
val_ds = _make_val_ds(df_val, y_val)
test_ds = _make_test_ds(submissions)




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
outputs = keras.layers.Dense(num_classes, activation="sigmoid")(x)
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

pred_labels = []
class_names_arr = np.array(class_names)
for i in range(preds.shape[0]):
    mask = preds[i] >= thresh
    chosen = class_names_arr[mask]
    if chosen.size == 0:
        chosen = np.array([class_names[int(np.argmax(preds[i]))]])
    pred_labels.append(" ".join(chosen.tolist()))

submissions["labels"] = pred_labels

submissions = submissions[["image", "labels"]]
submissions.to_csv("submission.csv", index=False)

print(submissions.head())
print("Wrote submission.csv with", len(submissions), "rows")
