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
from pathlib import Path

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import numpy as np
import pandas as pd
import tensorflow as tf

output_dir = "./"
train_csv_path = "../input/plant-pathology-2021-fgvc8/train.csv"
train_dir = "../input/plant-pathology-2021-fgvc8/train_images/"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
sample_sub_path = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"

image_dims = (300, 300, 3)

SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

tf.config.experimental.enable_op_determinism()

data_set = pd.read_csv(train_csv_path)
df_labels = data_set["labels"].fillna("")
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()
num_classes = len(dataset_labels)

data_set = data_set.copy()
data_set["image_path"] = data_set["image"].map(lambda x: os.path.join(train_dir, x))

paths_arr = data_set["image_path"].to_numpy()

exists_mask = np.fromiter(
    (os.path.isfile(p) for p in paths_arr), count=paths_arr.size, dtype=bool
)

if not bool(exists_mask.all()):
    data_set = data_set.loc[exists_mask].reset_index(drop=True)
    one_hot = data_set["labels"].fillna("").str.get_dummies(sep=" ")
    dataset_labels = one_hot.columns.to_list()
    num_classes = len(dataset_labels)

one_hot = one_hot.reindex(columns=dataset_labels, fill_value=0).astype("float32")

print("Train rows:", len(data_set))
print("Num classes:", num_classes)
print("Classes:", dataset_labels)



## === cell 1
AUTOTUNE = tf.data.AUTOTUNE


@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.string)])
def decode_and_resize(img_bytes):
    image = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)  # [0,1]
    image = tf.image.resize(image, [image_dims[0], image_dims[1]])
    return image


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
    ]
)
def load_image_and_label(path, label_vec):
    img_bytes = tf.io.read_file(path)
    image = decode_and_resize(img_bytes)
    return image, label_vec


idx = np.arange(len(data_set))
rng = np.random.default_rng(SEED)
rng.shuffle(idx)
val_frac = 0.1
val_size = int(len(idx) * val_frac)
val_idx = idx[:val_size]
train_idx = idx[val_size:]

train_paths = data_set.loc[train_idx, "image_path"].values
val_paths = data_set.loc[val_idx, "image_path"].values
train_y = one_hot.iloc[train_idx].values.astype("float32")
val_y = one_hot.iloc[val_idx].values.astype("float32")

BATCH_SIZE = 32

options = tf.data.Options()
options.deterministic = True

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_y)).with_options(
    options
)
train_ds = train_ds.shuffle(
    min(len(train_paths), 4096), seed=SEED, reshuffle_each_iteration=True
)

cache_dir = os.path.join(output_dir, "tfdata_cache")
os.makedirs(cache_dir, exist_ok=True)
train_cache_path = os.path.join(cache_dir, "train_decode_resize.cache")
val_cache_path = os.path.join(cache_dir, "val_decode_resize.cache")

train_ds = train_ds.map(load_image_and_label, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.cache(train_cache_path)

train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True)
train_ds = train_ds.prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_y)).with_options(options)
val_ds = val_ds.map(load_image_and_label, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.cache(val_cache_path)
val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=False)
val_ds = val_ds.prefetch(AUTOTUNE)

print("Train batches:", tf.data.experimental.cardinality(train_ds).numpy())
print("Val batches:", tf.data.experimental.cardinality(val_ds).numpy())



## === cell 2
inputs = tf.keras.Input(shape=image_dims, name="image")
x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="sigmoid", name="probs")(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
)

EPOCHS = 5
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2)



## === cell 3
sample_sub = pd.read_csv(sample_sub_path)
test_images = sample_sub["image"].tolist()


@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.string)])
def load_image_only(path):
    img_bytes = tf.io.read_file(path)
    image = decode_and_resize(img_bytes)
    return image


test_paths = [os.path.join(test_dir, fn) for fn in test_images]
test_ds = tf.data.Dataset.from_tensor_slices(test_paths)

test_options = tf.data.Options()
test_options.deterministic = True

test_ds = test_ds.with_options(test_options).map(
    load_image_only, num_parallel_calls=AUTOTUNE
)

test_cache_path = os.path.join(cache_dir, "test_decode_resize.cache")
test_ds = test_ds.cache(test_cache_path).batch(BATCH_SIZE).prefetch(AUTOTUNE)

pred = model.predict(test_ds, verbose=0).astype(np.float32)

threshold = 0.5
pred_bin = pred > threshold
label_arr = np.asarray(dataset_labels, dtype=object)

row_idx, col_idx = np.nonzero(pred_bin)
pred_labels = np.full(pred_bin.shape[0], "healthy", dtype=object)
if row_idx.size:
    order = np.argsort(row_idx, kind="mergesort")  # stable for determinism
    row_idx = row_idx[order]
    col_idx = col_idx[order]
    splits = np.flatnonzero(np.r_[True, row_idx[1:] != row_idx[:-1], True])
    for s, e in zip(splits[:-1], splits[1:]):
        r = row_idx[s]
        pred_labels[r] = " ".join(label_arr[col_idx[s:e]].tolist())

sub = pd.DataFrame({"image": test_images, "labels": pred_labels})

sub = sub.merge(sample_sub[["image"]], on="image", how="right")
sub["labels"] = sub["labels"].fillna("healthy")

out_path = os.path.join(output_dir, "submission.csv")
sub.to_csv(out_path, index=False)

print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
print("Saved to:", out_path)
