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
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF decide
    tf.config.threading.set_inter_op_parallelism_threads(0)  # let TF decide
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE




## === cell 1
DATA_DIR = "../input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

train = pd.read_csv(TRAIN_CSV)
submissions = pd.read_csv(SAMPLE_SUB)

train.head(), submissions.head(), train.shape, submissions.shape




## === cell 2
h_target = 384
w_target = 384
batch_size = 32

label_split = train["labels"].str.split()
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(label_split)
class_names = list(mlb.classes_)
n_classes = len(class_names)

n_classes, class_names[:10]




## === cell 3
from sklearn.model_selection import train_test_split

train_df, valid_df, y_train, y_valid = train_test_split(
    train[["image"]],
    Y.astype(np.float32),
    test_size=0.15,
    random_state=SEED,
    shuffle=True,
)

TRAIN_IMG_DIR_T = tf.constant(TRAIN_IMG_DIR, dtype=tf.string)
TEST_IMG_DIR_T = tf.constant(TEST_IMG_DIR, dtype=tf.string)
SLASH_T = tf.constant("/", dtype=tf.string)
SEED_T = tf.constant(SEED, tf.int32)
H_T = tf.constant(h_target, tf.int32)
W_T = tf.constant(w_target, tf.int32)


def _read_decode_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img,
        [H_T, W_T],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


try:
    import tensorflow_addons as tfa

    _HAS_TFA = True
except Exception:
    _HAS_TFA = False


def _augment_stateless(img, seed2):
    img = tf.image.stateless_random_flip_left_right(img, seed2)

    angle = tf.random.stateless_uniform(
        [],
        seed2 + tf.constant([1, 0], tf.int32),
        minval=-15.0,
        maxval=15.0,
        dtype=tf.float32,
    ) * (np.pi / 180.0)
    img = tfa.image.rotate(
        img, angles=angle, interpolation="BILINEAR", fill_mode="nearest"
    )

    dx = tf.random.stateless_uniform(
        [],
        seed2 + tf.constant([2, 0], tf.int32),
        minval=-0.05,
        maxval=0.05,
        dtype=tf.float32,
    ) * tf.cast(W_T, tf.float32)
    dy = tf.random.stateless_uniform(
        [],
        seed2 + tf.constant([3, 0], tf.int32),
        minval=-0.05,
        maxval=0.05,
        dtype=tf.float32,
    ) * tf.cast(H_T, tf.float32)
    img = tfa.image.translate(
        img, translations=[dx, dy], interpolation="BILINEAR", fill_mode="nearest"
    )

    z = tf.random.stateless_uniform(
        [],
        seed2 + tf.constant([4, 0], tf.int32),
        minval=0.9,
        maxval=1.1,
        dtype=tf.float32,
    )
    if_zoom_in = tf.greater(z, 1.0)

    def _zoom_in():
        crop_h = tf.cast(tf.cast(H_T, tf.float32) / z, tf.int32)
        crop_w = tf.cast(tf.cast(W_T, tf.float32) / z, tf.int32)
        cropped = tf.image.stateless_random_crop(
            img, size=[crop_h, crop_w, 3], seed=seed2 + tf.constant([5, 0], tf.int32)
        )
        return tf.image.resize(
            cropped,
            [H_T, W_T],
            method=tf.image.ResizeMethod.BILINEAR,
            antialias=False,
        )

    def _zoom_out():
        pad_h = tf.cast(tf.cast(H_T, tf.float32) * (1.0 / z), tf.int32)
        pad_w = tf.cast(tf.cast(W_T, tf.float32) * (1.0 / z), tf.int32)
        resized = tf.image.resize(
            img, [pad_h, pad_w], method=tf.image.ResizeMethod.BILINEAR, antialias=False
        )
        dh = H_T - pad_h
        dw = W_T - pad_w
        top = dh // 2
        bottom = dh - top
        left = dw // 2
        right = dw - left
        padded = tf.pad(
            resized, [[top, bottom], [left, right], [0, 0]], mode="SYMMETRIC"
        )
        return padded

    img = tf.cond(if_zoom_in, _zoom_in, _zoom_out)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img


def make_dataset(images, labels=None, training=False, batch_size=32):
    base_dir = TRAIN_IMG_DIR_T if labels is not None else TEST_IMG_DIR_T

    image_names = tf.convert_to_tensor(images.values.astype("U"), dtype=tf.string)
    paths = tf.strings.join([base_dir, SLASH_T, image_names])

    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(paths)
    else:
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(
            buffer_size=len(images), seed=SEED, reshuffle_each_iteration=True
        )

    options = tf.data.Options()
    options.deterministic = True
    ds = ds.with_options(options)

    if labels is None:

        def _map_no_label(path):
            img = _read_decode_resize(path)
            return img

        ds = ds.map(_map_no_label, num_parallel_calls=AUTOTUNE, deterministic=True)
    else:
        if training and _HAS_TFA:

            def _map_train(path, y):
                img = _read_decode_resize(path)
                ph = tf.strings.to_hash_bucket_fast(path, 2**31 - 1)
                seed2 = tf.stack([tf.cast(ph, tf.int32), SEED_T])
                img = _augment_stateless(img, seed2)
                return img, y

            ds = ds.map(_map_train, num_parallel_calls=AUTOTUNE, deterministic=True)
        elif training and (not _HAS_TFA):

            def _map_train_simple(path, y):
                img = _read_decode_resize(path)
                ph = tf.strings.to_hash_bucket_fast(path, 2**31 - 1)
                seed2 = tf.stack([tf.cast(ph, tf.int32), SEED_T])
                img = tf.image.stateless_random_flip_left_right(img, seed2)
                return img, y

            ds = ds.map(
                _map_train_simple, num_parallel_calls=AUTOTUNE, deterministic=True
            )
        else:

            def _map_valid(path, y):
                img = _read_decode_resize(path)
                return img, y

            ds = ds.map(_map_valid, num_parallel_calls=AUTOTUNE, deterministic=True)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_dataset(
    train_df["image"], y_train, training=True, batch_size=batch_size
)
valid_ds = make_dataset(
    valid_df["image"], y_valid, training=False, batch_size=batch_size
)
test_ds = make_dataset(
    submissions["image"], labels=None, training=False, batch_size=batch_size
)

steps_per_epoch = int(np.ceil(len(train_df) / batch_size))
validation_steps = int(np.ceil(len(valid_df) / batch_size))




## === cell 4
inputs = keras.Input(shape=(h_target, w_target, 3))
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.3)(x)
outputs = keras.layers.Dense(n_classes, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

model.summary()




## === cell 5
EPOCHS = 3

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    verbose=1,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
)




## === cell 6
preds = model.predict(test_ds, verbose=1)
preds.shape, preds[:2]




## === cell 7
thresh = 0.4

above = preds >= thresh
argmax_idx = preds.argmax(axis=1)

pred_labels = []
for i in range(preds.shape[0]):
    chosen_idx = np.flatnonzero(above[i])
    if chosen_idx.size == 0:
        chosen_idx = np.array([argmax_idx[i]])
    chosen = [class_names[j] for j in chosen_idx.tolist()]

    if "healthy" in chosen and len(chosen) > 1:
        chosen = [class_names[int(argmax_idx[i])]]

    pred_labels.append(" ".join(chosen))

submissions = submissions.copy()
submissions["labels"] = pred_labels

submissions = submissions[["image", "labels"]]
submissions["image"] = submissions["image"].astype(str)
submissions["labels"] = submissions["labels"].astype(str)

submissions.head()




## === cell 8
out_path = "submission.csv"
submissions.to_csv(out_path, index=False)
print(f"Wrote {out_path} with shape {submissions.shape}")
print(submissions.iloc[:5])
