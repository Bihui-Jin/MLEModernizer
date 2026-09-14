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

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

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
SEED_T = tf.constant(SEED, tf.int32)
H_T = tf.constant(h_target, tf.int32)
W_T = tf.constant(w_target, tf.int32)


@tf.function
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
    img.set_shape([h_target, w_target, 3])
    return img


_HAS_TFA = False


def _build_paths_np(image_series, base_dir):
    arr = image_series.to_numpy(dtype=str, copy=False)
    return np.char.add(base_dir + "/", arr).astype("U")


@tf.function
def _map_test(path):
    return _read_decode_resize(path)


@tf.function
def _map_valid(path, y):
    img = _read_decode_resize(path)
    return img, y


@tf.function
def _map_train_simple(path, y):
    img = _read_decode_resize(path)
    ph = tf.strings.to_hash_bucket_fast(path, 2**31 - 1)
    seed2 = tf.stack([tf.cast(ph, tf.int32), SEED_T])
    img = tf.image.stateless_random_flip_left_right(img, seed2)
    img.set_shape([h_target, w_target, 3])
    return img, y


def make_dataset(images, labels=None, training=False, batch_size=32):
    if labels is None:
        paths_np = _build_paths_np(images, TEST_IMG_DIR)
        paths = tf.convert_to_tensor(paths_np, dtype=tf.string)
        ds = tf.data.Dataset.from_tensor_slices(paths)
    else:
        paths_np = _build_paths_np(images, TRAIN_IMG_DIR)
        paths = tf.convert_to_tensor(paths_np, dtype=tf.string)
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    options = tf.data.Options()
    options.deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    try:
        options.experimental_optimization.autotune_buffers = True
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.parallel_batch = True
        options.experimental_optimization.map_vectorization.enabled = True
    except Exception:
        pass
    ds = ds.with_options(options)

    if training:
        buffer_size = int(min(len(images), 4096))
        ds = ds.shuffle(
            buffer_size=buffer_size, seed=SEED, reshuffle_each_iteration=True
        )

    if labels is None:
        ds = ds.map(_map_test, num_parallel_calls=AUTOTUNE, deterministic=True)
    else:
        if training:
            ds = ds.map(
                _map_train_simple, num_parallel_calls=AUTOTUNE, deterministic=True
            )
        else:
            ds = ds.map(_map_valid, num_parallel_calls=AUTOTUNE, deterministic=True)

    ds = ds.apply(tf.data.experimental.ignore_errors())
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

healthy_idx = None
for i, c in enumerate(class_names):
    if c == "healthy":
        healthy_idx = i
        break

idxs = [np.flatnonzero(row).tolist() for row in above]
pred_labels = []
for i, cols in enumerate(idxs):
    if cols:
        if healthy_idx is not None and (healthy_idx in cols) and (len(cols) > 1):
            cols = [int(argmax_idx[i])]
        chosen = [class_names[c] for c in cols]
    else:
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
