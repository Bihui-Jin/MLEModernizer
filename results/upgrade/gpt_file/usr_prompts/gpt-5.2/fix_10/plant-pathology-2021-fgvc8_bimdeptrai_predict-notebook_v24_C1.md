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
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

from sklearn.preprocessing import MultiLabelBinarizer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

print("TF:", tf.__version__)



## === cell 1
train = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
train.head()



## === cell 2
submissions = pd.read_csv("../input/plant-pathology-2021-fgvc8/sample_submission.csv")
submissions.head()



## === cell 3
h_target = 256
w_target = 256
batch_size = 32



## === cell 4
label_split = train.labels.apply(lambda x: x.split())
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(label_split)
classes = list(mlb.classes_)
num_classes = len(classes)

print("Num classes:", num_classes)
print("Classes:", classes)

train_df = train.copy()
for idx, c in enumerate(classes):
    train_df[c] = y[:, idx].astype(np.float32)

train_df.head()



## === cell 5
AUTOTUNE = tf.data.AUTOTUNE

train_dir = "../input/plant-pathology-2021-fgvc8/train_images"
test_dir = (
    "../input/plant-pathology-2021-fgvc8/test_images"
    if os.path.exists("../input/plant-pathology-2021-fgvc8/test_images")
    else "../input/plant-pathology-2021-fgvcvc8/test_images"
)

val_split = 0.10

idx = np.arange(len(train_df), dtype=np.int32)
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
val_size = int(np.floor(len(idx) * val_split))
train_idx, val_idx = idx[:-val_size], idx[-val_size:]

train_part = train_df.iloc[train_idx].reset_index(drop=True)
valid_part = train_df.iloc[val_idx].reset_index(drop=True)

train_paths = (train_dir + "/" + train_part["image"].astype(str)).to_numpy()
valid_paths = (train_dir + "/" + valid_part["image"].astype(str)).to_numpy()
test_paths = (test_dir + "/" + submissions["image"].astype(str)).to_numpy()

train_labels = train_part[classes].values.astype(np.float32, copy=False)
valid_labels = valid_part[classes].values.astype(np.float32, copy=False)


def _base_options():
    opts = tf.data.Options()
    opts.experimental_deterministic = True
    try:
        opts.experimental_optimization.apply_default_optimizations = True
        opts.experimental_optimization.autotune_buffers = True
        opts.experimental_optimization.map_and_batch_fusion = True
        opts.experimental_optimization.map_parallelization = True
        opts.experimental_optimization.parallel_batch = True
    except Exception:
        pass
    return opts


def _cache_path(name: str) -> str:
    return os.path.join(
        "/kaggle/working", f"tfdata_cache_{name}_{h_target}x{w_target}_bs{batch_size}"
    )


@tf.function(reduce_retracing=True)
def _read_bytes(path):
    return tf.io.read_file(path)


@tf.function(reduce_retracing=True)
def _decode_resize_rescale_from_bytes(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(
        img, [h_target, w_target], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


augmenter = keras.Sequential(
    [
        keras.layers.RandomFlip("horizontal", seed=SEED),
        keras.layers.RandomRotation(
            factor=10.0 / 180.0, fill_mode="reflect", seed=SEED
        ),
        keras.layers.RandomTranslation(
            height_factor=0.05, width_factor=0.05, fill_mode="reflect", seed=SEED
        ),
        keras.layers.RandomZoom(
            height_factor=(-0.1, 0.1),
            width_factor=(-0.1, 0.1),
            fill_mode="reflect",
            seed=SEED,
        ),
    ],
    name="augmenter",
)


@tf.function(reduce_retracing=True)
def _augment_batch(imgs, labels):
    imgs = augmenter(imgs, training=True)
    return imgs, labels


@tf.function(reduce_retracing=True)
def _decode_xy(img_bytes, y):
    return _decode_resize_rescale_from_bytes(img_bytes), y


def make_train_ds(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels)).with_options(
        _base_options()
    )
    ds = ds.map(lambda p, y: (_read_bytes(p), y), num_parallel_calls=AUTOTUNE)
    ds = ds.cache(_cache_path("train_bytes"))
    ds = ds.shuffle(
        buffer_size=min(len(paths), 4096), seed=SEED, reshuffle_each_iteration=True
    )
    ds = ds.repeat()
    ds = ds.map(_decode_xy, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.map(_augment_batch, num_parallel_calls=AUTOTUNE)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_valid_ds(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels)).with_options(
        _base_options()
    )
    ds = ds.map(lambda p, y: (_read_bytes(p), y), num_parallel_calls=AUTOTUNE)
    ds = ds.cache(_cache_path("valid_bytes"))
    ds = ds.map(_decode_xy, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


@tf.function(reduce_retracing=True)
def _decode_x(img_bytes):
    return _decode_resize_rescale_from_bytes(img_bytes)


def make_test_ds(paths):
    ds = tf.data.Dataset.from_tensor_slices(paths).with_options(_base_options())
    ds = ds.map(_read_bytes, num_parallel_calls=AUTOTUNE)
    ds = ds.map(_decode_x, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_train_ds(train_paths, train_labels)
valid_ds = make_valid_ds(valid_paths, valid_labels)
test_ds = make_test_ds(test_paths)

steps_per_epoch = int(len(train_paths) // batch_size)
validation_steps = int(np.ceil(len(valid_paths) / batch_size))

print(
    "Train batches:",
    steps_per_epoch,
    "Valid batches:",
    validation_steps,
    "Test batches:",
    int(np.ceil(len(test_paths) / batch_size)),
)



## === cell 6
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



## === cell 7
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=3,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)



## === cell 8
preds = model.predict(test_ds, verbose=1)
print("preds shape:", preds.shape)



## === cell 9
thresh = 0.25

mask = preds >= thresh
has_any = mask.any(axis=1)
argmax_idx = preds.argmax(axis=1)

class_arr = np.array(classes, dtype=object)
healthy_idx = (
    int(np.where(class_arr == "healthy")[0][0]) if "healthy" in classes else -1
)

pred_labels = []
for i in range(preds.shape[0]):
    if has_any[i]:
        chosen_idx = np.flatnonzero(mask[i])
    else:
        chosen_idx = np.array([int(argmax_idx[i])], dtype=np.int64)

    if healthy_idx != -1 and healthy_idx in chosen_idx:
        pred_labels.append("healthy")
    else:
        pred_labels.append(" ".join(class_arr[chosen_idx].tolist()))

submissions_out = submissions.copy()
submissions_out["labels"] = pred_labels
submissions_out = submissions_out[["image", "labels"]]
submissions_out.to_csv("submission.csv", index=False)
print(submissions_out.head())
print("Wrote submission.csv with", len(submissions_out), "rows")



## === cell 10
submissions_out
