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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

print("TF version:", tf.__version__)
print("Eager:", tf.executing_eagerly())

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)



## === cell 1
output_dir = "./"

CANDIDATE_TEST_DIRS = [
    "../input/plant-pathology-2021-fgvc8/test_images/",
    "/kaggle/input/plant-pathology-2021-fgvc8/test_images/",
    "../input/test_images/",
    "/kaggle/input/test_images/",
    "/kaggle/data/plant-pathology-2021-fgvc8/test_images/",
    "/kaggle/data/test_images/",
]
test_dir = None
for d in CANDIDATE_TEST_DIRS:
    if os.path.isdir(d):
        test_dir = d
        break
if test_dir is None:
    raise FileNotFoundError(
        f"Could not find test_images dir. Tried: {CANDIDATE_TEST_DIRS}"
    )

image_dims = (300, 300, 3)

CANDIDATE_TRAIN_CSVS = [
    "../input/plant-pathology-2021-fgvc8/train.csv",
    "/kaggle/input/plant-pathology-2021-fgvc8/train.csv",
    "../input/train.csv",
    "/kaggle/input/train.csv",
    "/kaggle/data/plant-pathology-2021-fgvc8/train.csv",
    "/kaggle/data/train.csv",
]
train_csv_path = None
for p in CANDIDATE_TRAIN_CSVS:
    if os.path.isfile(p):
        train_csv_path = p
        break
if train_csv_path is None:
    raise FileNotFoundError(f"Could not find train.csv. Tried: {CANDIDATE_TRAIN_CSVS}")

CANDIDATE_TRAIN_DIRS = [
    "../input/plant-pathology-2021-fgvc8/train_images/",
    "/kaggle/input/plant-pathology-2021-fgvc8/train_images/",
    "../input/train_images/",
    "/kaggle/input/train_images/",
    "/kaggle/data/plant-pathology-2021-fgvc8/train_images/",
    "/kaggle/data/train_images/",
]
train_dir = None
for d in CANDIDATE_TRAIN_DIRS:
    if os.path.isdir(d):
        train_dir = d
        break
if train_dir is None:
    raise FileNotFoundError(
        f"Could not find train_images dir. Tried: {CANDIDATE_TRAIN_DIRS}"
    )

data_set = pd.read_csv(train_csv_path)
df_labels = data_set["labels"]
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()

print("Using train.csv:", train_csv_path)
print("Using train_dir:", train_dir)
print("Using test_dir:", test_dir)
print("Num classes:", len(dataset_labels))
print("Train rows:", len(data_set))
print("Num test images (dir listing):", len(os.listdir(test_dir)))



## === cell 2

from tensorflow import keras
from tensorflow.keras import layers as keras_layers

y = (
    data_set["labels"]
    .str.get_dummies(sep=" ")
    .reindex(columns=dataset_labels, fill_value=0)
    .astype(np.float32)
)
data_set = data_set.copy()
data_set["filepath"] = data_set["image"].apply(lambda x: os.path.join(train_dir, x))

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


def _load_and_preprocess(path, label=None):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_image(
        img_bytes, channels=3, dtype=tf.float32, expand_animations=False
    )
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [image_dims[0], image_dims[1]])
    if label is None:
        return img
    return img, label


def make_ds(df, labels, training):
    paths = df["filepath"].values
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(
            buffer_size=min(len(df), 2048), seed=SEED, reshuffle_each_iteration=True
        )
    ds = ds.map(
        lambda p, lab: _load_and_preprocess(p, lab), num_parallel_calls=AUTOTUNE
    )
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


train_ds = make_ds(train_df, y_train, training=True)
val_ds = make_ds(val_df, y_val, training=False)

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
)

EPOCHS = 3
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)



## === cell 3
images_path_list = sorted(list(os.listdir(test_dir)))


def load_test_image(full_path):
    img_bytes = tf.io.read_file(full_path)
    img = tf.io.decode_image(
        img_bytes, channels=3, dtype=tf.float32, expand_animations=False
    )
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [image_dims[0], image_dims[1]])
    return img


threshold = 0.7  # preserve original logic

values = []
for fname in images_path_list:
    full_path = os.path.join(test_dir, fname)
    img = load_test_image(full_path)
    pred = model(tf.expand_dims(img, axis=0), training=False)
    pred = tf.convert_to_tensor(pred)[0].numpy().tolist()

    index_values = [i for i, v in enumerate(pred) if v > threshold]

    if len(index_values) == 0:
        classes_img = "healthy"
    else:
        classes_img = " ".join([dataset_labels[i] for i in index_values])

    values.append([fname, classes_img])

csv_pd = pd.DataFrame(values, columns=["image", "labels"])
out_path = os.path.join(output_dir, "submission.csv")
csv_pd.to_csv(out_path, index=False)

print("Wrote submission.csv with shape:", csv_pd.shape, "to:", out_path)
print(csv_pd.head())
