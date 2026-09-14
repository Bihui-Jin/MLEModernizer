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

geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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
import sys
import random
import numpy as np
import pandas as pd

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TFDF_NO_TF_IMPORT", "1")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "0")

import tensorflow as tf

print("TF version:", tf.__version__)
print("Eager:", tf.executing_eagerly())

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("Determinism not enabled:", repr(e))

try:
    for _gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(_gpu, True)
except Exception as e:
    print("GPU memory growth not set:", repr(e))

try:
    tf.config.optimizer.set_jit(True)
except Exception as e:
    print("XLA JIT not set:", repr(e))

output_dir = "./"

CANDIDATE_TEST_DIRS = [
    "/kaggle/input/plant-pathology-2021-fgvc8/test_images/",
    "/kaggle/data/plant-pathology-2021-fgvc8/test_images/",
    "/kaggle/input/test_images/",
    "/kaggle/data/test_images/",
    "../input/plant-pathology-2021-fgvc8/test_images/",
    "../input/test_images/",
]
test_dir = next((d for d in CANDIDATE_TEST_DIRS if os.path.isdir(d)), None)
if test_dir is None:
    raise FileNotFoundError(
        f"Could not find test_images dir. Tried: {CANDIDATE_TEST_DIRS}"
    )

image_dims = (300, 300, 3)

CANDIDATE_TRAIN_CSVS = [
    "/kaggle/input/plant-pathology-2021-fgvc8/train.csv",
    "/kaggle/data/plant-pathology-2021-fgvc8/train.csv",
    "/kaggle/input/train.csv",
    "/kaggle/data/train.csv",
    "../input/plant-pathology-2021-fgvc8/train.csv",
    "../input/train.csv",
]
train_csv_path = next((p for p in CANDIDATE_TRAIN_CSVS if os.path.isfile(p)), None)
if train_csv_path is None:
    raise FileNotFoundError(f"Could not find train.csv. Tried: {CANDIDATE_TRAIN_CSVS}")

CANDIDATE_TRAIN_DIRS = [
    "/kaggle/input/plant-pathology-2021-fgvc8/train_images/",
    "/kaggle/data/plant-pathology-2021-fgvc8/train_images/",
    "/kaggle/input/train_images/",
    "/kaggle/data/train_images/",
    "../input/plant-pathology-2021-fgvc8/train_images/",
    "../input/train_images/",
]
train_dir = next((d for d in CANDIDATE_TRAIN_DIRS if os.path.isdir(d)), None)
if train_dir is None:
    raise FileNotFoundError(
        f"Could not find train_images dir. Tried: {CANDIDATE_TRAIN_DIRS}"
    )

data_set = pd.read_csv(train_csv_path)

one_hot = data_set["labels"].str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()

print("Using train.csv:", train_csv_path)
print("Using train_dir:", train_dir)
print("Using test_dir:", test_dir)
print("Num classes:", len(dataset_labels))
print("Train rows:", len(data_set))
print(
    "Num test images (dir listing):",
    len([e for e in os.scandir(test_dir) if e.is_file()]),
)



## === cell 1
from tensorflow import keras
from tensorflow.keras import layers as keras_layers

y = one_hot.reindex(columns=dataset_labels, fill_value=0).astype(np.float32)
data_set = data_set.copy()
data_set["filepath"] = (train_dir.rstrip("/") + "/") + data_set["image"].astype(str)

idx = np.arange(len(data_set))
rng = np.random.default_rng(SEED)
rng.shuffle(idx)
val_frac = 0.1
val_size = int(len(idx) * val_frac)
val_idx = idx[:val_size]
train_idx = idx[val_size:]

train_df = data_set.iloc[train_idx].reset_index(drop=True)
val_df = data_set.iloc[val_idx].reset_index(drop=True)
y_train = y.iloc[train_idx].reset_index(drop=True).values
y_val = y.iloc[val_idx].reset_index(drop=True).values

print("Train/Val sizes:", len(train_df), len(val_df))

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 16


@tf.function
def _load_and_preprocess(path, label):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1] float32
    img = tf.image.resize(img, [image_dims[0], image_dims[1]])
    return img, label


def make_ds(df, labels, training, cache_to_disk=True, cache_tag="train"):
    paths = df["filepath"].to_numpy(dtype=str)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(
            buffer_size=min(len(df), 2048), seed=SEED, reshuffle_each_iteration=True
        )

    cache_path = None
    if cache_to_disk:
        cache_path = os.path.join(output_dir, f"tf_cache_{cache_tag}")

    ds = ds.map(_load_and_preprocess, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    if cache_path is not None:
        ds = ds.cache(cache_path)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_ds(
    train_df, y_train, training=True, cache_to_disk=True, cache_tag="train"
)
val_ds = make_ds(val_df, y_val, training=False, cache_to_disk=True, cache_tag="val")

num_classes = len(dataset_labels)
inputs = keras.Input(shape=image_dims, name="image")
x = keras_layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = keras_layers.MaxPooling2D()(x)
x = keras_layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = keras_layers.MaxPooling2D()(x)
x = keras_layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = keras_layers.GlobalAveragePooling2D()(x)
x = keras_layers.Dropout(0.2)(x)
outputs = keras_layers.Dense(num_classes, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    steps_per_execution=32,
)

EPOCHS = 3
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)



## === cell 2
val_preds = model.predict(val_ds, verbose=0)


def macro_f1_from_probs(y_true, y_prob, thr: float) -> float:
    y_pred = (y_prob > thr).astype(np.int32)
    y_true = (y_true > 0.5).astype(np.int32)

    tp = (y_true & y_pred).sum(axis=0).astype(np.float64)
    fp = ((1 - y_true) & y_pred).sum(axis=0).astype(np.float64)
    fn = (y_true & (1 - y_pred)).sum(axis=0).astype(np.float64)

    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, (2 * tp) / denom, 0.0)
    return float(np.mean(f1))


grid_coarse = np.round(np.linspace(0.30, 0.90, 13), 2)
scores_coarse = [
    (t, macro_f1_from_probs(y_val, val_preds, float(t))) for t in grid_coarse
]
best_t0, best_f10 = max(scores_coarse, key=lambda x: x[1])

t_low = max(0.05, best_t0 - 0.10)
t_high = min(0.95, best_t0 + 0.10)
grid_fine = np.round(np.linspace(t_low, t_high, 21), 3)
scores_fine = [(t, macro_f1_from_probs(y_val, val_preds, float(t))) for t in grid_fine]
best_t, best_f1 = max(scores_fine, key=lambda x: x[1])

print("Coarse threshold grid scores:", scores_coarse)
print("Fine threshold grid range:", (t_low, t_high))
print("Selected threshold:", float(best_t), "Val macro-F1:", float(best_f1))

threshold = float(best_t)



## === cell 3
images_path_list = sorted([e.name for e in os.scandir(test_dir) if e.is_file()])
test_paths = np.array(
    [os.path.join(test_dir, fname) for fname in images_path_list], dtype=object
)


@tf.function
def load_test_image(full_path):
    img_bytes = tf.io.read_file(full_path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [image_dims[0], image_dims[1]])
    return img


TEST_BATCH_SIZE = 64  # batching only; does not change predictions
test_ds = tf.data.Dataset.from_tensor_slices(test_paths.astype(str))
test_ds = test_ds.map(load_test_image, num_parallel_calls=AUTOTUNE, deterministic=True)
test_ds = test_ds.apply(tf.data.experimental.ignore_errors())
test_ds = test_ds.batch(TEST_BATCH_SIZE, drop_remainder=False)
test_ds = test_ds.prefetch(AUTOTUNE)

preds = model.predict(test_ds, verbose=0)
pred_mask = preds > threshold

labels_arr = np.asarray(dataset_labels, dtype=object)

values = []
for fname, mask in zip(images_path_list, pred_mask):
    idxs = np.flatnonzero(mask)
    classes_img = "healthy" if idxs.size == 0 else " ".join(labels_arr[idxs].tolist())
    values.append((fname, classes_img))

csv_pd = pd.DataFrame(values, columns=["image", "labels"])
out_path = os.path.join(output_dir, "submission.csv")
csv_pd.to_csv(out_path, index=False)

print("Wrote submission.csv with shape:", csv_pd.shape, "to:", out_path)
print(csv_pd.head())
