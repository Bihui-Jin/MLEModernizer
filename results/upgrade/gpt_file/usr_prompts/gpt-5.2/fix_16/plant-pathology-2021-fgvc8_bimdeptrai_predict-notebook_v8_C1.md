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
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    cpu = max(1, os.cpu_count() or 1)
    tf.config.threading.set_intra_op_parallelism_threads(cpu)
    tf.config.threading.set_inter_op_parallelism_threads(min(8, cpu))
except Exception:
    pass

print("TensorFlow:", tf.__version__)
print("CPUs:", os.cpu_count())



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

label_split = train.labels.apply(lambda x: x.split())
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(label_split)
class_names = list(mlb.classes_)

print("Num classes:", len(class_names))
print("Classes:", class_names)

for j, c in enumerate(class_names):
    train[c] = Y[:, j].astype(np.float32)



## === cell 3
BATCH_SIZE = 16
TEST_BATCH_SIZE = 32

AUTOTUNE = tf.data.AUTOTUNE

n = len(train)
val_size = int(np.ceil(0.1 * n))
train_size = n - val_size

train_df = train.iloc[:train_size].reset_index(drop=True)
valid_df = train.iloc[train_size:].reset_index(drop=True)

train_paths = (TRAIN_IMG_DIR + "/" + train_df["image"].astype(str)).values
valid_paths = (TRAIN_IMG_DIR + "/" + valid_df["image"].astype(str)).values
test_paths = (TEST_IMG_DIR + "/" + submissions["image"].astype(str)).values

y_train = train_df[class_names].to_numpy(dtype=np.float32, copy=False)
y_valid = valid_df[class_names].to_numpy(dtype=np.float32, copy=False)

H_T = tf.constant(h_target, dtype=tf.int32)
W_T = tf.constant(w_target, dtype=tf.int32)
INV_255 = tf.constant(1.0 / 255.0, dtype=tf.float32)


@tf.function
def _decode_resize_rescale(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize_with_pad(
        img, H_T, W_T, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32) * INV_255
    return img


@tf.function
def _decode_with_label(path, y):
    return _decode_resize_rescale(path), y


def _make_ds(
    paths,
    labels=None,
    batch_size=32,
    training=False,
    repeat=False,
    cache=False,
    cache_path=None,
    drop_remainder=False,
):
    options = tf.data.Options()
    options.deterministic = True
    try:
        options.experimental_optimization.apply_default_optimizations = True
        options.experimental_optimization.autotune_buffers = True
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.parallel_batch = True
        options.experimental_slack = True
        options.threading.private_threadpool_size = min(
            16, max(1, (os.cpu_count() or 4))
        )
    except Exception:
        pass

    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(paths)
    else:
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    ds = ds.with_options(options)

    if training:
        buf = min(len(paths), 1024)
        ds = ds.shuffle(buffer_size=buf, seed=SEED, reshuffle_each_iteration=True)

    if labels is None:
        ds = ds.map(
            _decode_resize_rescale, num_parallel_calls=AUTOTUNE, deterministic=True
        )
    else:
        ds = ds.map(_decode_with_label, num_parallel_calls=AUTOTUNE, deterministic=True)

    ds = ds.apply(tf.data.experimental.ignore_errors())

    if cache:
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            ds = ds.cache(cache_path)
        else:
            ds = ds.cache()

    ds = ds.batch(batch_size, drop_remainder=drop_remainder)

    if repeat:
        ds = ds.repeat()

    ds = ds.prefetch(AUTOTUNE)
    return ds


CACHE_DIR = "./tfdata_cache"
train_cache_path = os.path.join(CACHE_DIR, "train_cache.tf-data")
valid_cache_path = os.path.join(CACHE_DIR, "valid_cache.tf-data")

train_ds = _make_ds(
    train_paths,
    y_train,
    batch_size=BATCH_SIZE,
    training=True,
    repeat=True,
    cache=True,
    cache_path=train_cache_path,
    drop_remainder=True,
)

valid_ds = _make_ds(
    valid_paths,
    y_valid,
    batch_size=BATCH_SIZE,
    training=False,
    repeat=False,
    cache=True,
    cache_path=valid_cache_path,
    drop_remainder=False,
)

test_ds = _make_ds(
    test_paths,
    labels=None,
    batch_size=TEST_BATCH_SIZE,
    training=False,
    repeat=False,
    cache=False,
    cache_path=None,
    drop_remainder=False,
)

steps_per_epoch = int(np.ceil(train_size / BATCH_SIZE))
validation_steps = int(np.ceil(val_size / BATCH_SIZE))
test_steps = int(np.ceil(len(submissions) / TEST_BATCH_SIZE))

print(
    "steps_per_epoch:",
    steps_per_epoch,
    "validation_steps:",
    validation_steps,
    "test_steps:",
    test_steps,
)



## === cell 4
base = tf.keras.applications.MobileNetV2(
    input_shape=(h_target, w_target, 3), include_top=False, weights="imagenet"
)
base.trainable = False

inputs = keras.Input(shape=(h_target, w_target, 3))
x = base(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(len(class_names), activation="sigmoid")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    jit_compile=True,
)

model.summary()



## === cell 5
EPOCHS = 2

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)



## === cell 6
preds = model.predict(
    test_ds,
    steps=test_steps,
    verbose=1,
)
print("preds shape:", preds.shape)

preds = preds[: len(submissions)]
print("preds shape (trimmed):", preds.shape)



## === cell 7
thresh = 0.2

mask = preds >= thresh  # (N, C) boolean
any_pos = mask.any(axis=1)
argmax_idx = preds.argmax(axis=1)

class_names_arr = np.asarray(class_names, dtype=object)
pred_labels = np.empty(preds.shape[0], dtype=object)

pos_rows = np.flatnonzero(any_pos)
if pos_rows.size:
    pos_mask = mask[pos_rows]
    packed = np.packbits(pos_mask, axis=1)
    cache = {}
    for i, key_bytes in enumerate(map(bytes, packed)):
        s = cache.get(key_bytes)
        if s is None:
            s = " ".join(class_names_arr[pos_mask[i]])
            cache[key_bytes] = s
        pred_labels[pos_rows[i]] = s

neg_rows = np.flatnonzero(~any_pos)
if neg_rows.size:
    pred_labels[neg_rows] = class_names_arr[argmax_idx[neg_rows]]

submissions["labels"] = pred_labels.tolist()
submissions = submissions[["image", "labels"]]
submissions.to_csv("submission.csv", index=False)

print(submissions.head())
print("Wrote submission.csv with", len(submissions), "rows")



## === cell 8
submissions
