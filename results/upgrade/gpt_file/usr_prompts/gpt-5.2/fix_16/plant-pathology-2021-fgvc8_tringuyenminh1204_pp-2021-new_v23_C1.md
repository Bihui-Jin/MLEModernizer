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
import math
import re
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K

print("tf:", tf.__version__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
except Exception:
    pass



## === cell 1
path = "../input/plant-pathology-2021-fgvc8/"
train = pd.read_csv(path + "train.csv")
sub = pd.read_csv(path + "sample_submission.csv")

train.head(), sub.head()



## === cell 2
AUTO = tf.data.experimental.AUTOTUNE



## === cell 3
data_opts = tf.data.Options()
data_opts.deterministic = True
try:
    data_opts.threading.private_threadpool_size = 0
except Exception:
    pass
try:
    data_opts.threading.max_intra_op_parallelism = 0
except Exception:
    pass
try:
    data_opts.experimental_optimization.apply_default_optimizations = True
except Exception:
    pass
try:
    data_opts.experimental_optimization.map_parallelization = True
except Exception:
    pass
try:
    data_opts.experimental_optimization.parallel_batch = True
except Exception:
    pass
try:
    data_opts.experimental_optimization.optimize_parallelization = True
except Exception:
    pass
try:
    data_opts.experimental_slack = True
except Exception:
    pass



## === cell 4
import pathlib



## === cell 5
train_image_dir = "../input/plant-pathology-2021-fgvc8/train_images/"
test_image_dir = "../input/plant-pathology-2021-fgvc8/test_images/"

train_paths = [os.path.join(train_image_dir, fn) for fn in train["image"].tolist()]
test_paths = [os.path.join(test_image_dir, fn) for fn in sub["image"].tolist()]

print(len(train_paths), len(test_paths))



## === cell 6
CLASSES = ["scab", "frog_eye_leaf_spot", "complex", "rust", "powdery_mildew", "healthy"]
class_to_idx = {c: i for i, c in enumerate(CLASSES)}
idx_to_class = {i: c for i, c in enumerate(CLASSES)}
CLASSES




## === cell 7
def labels_to_multihot(label_str: str, classes=CLASSES):
    parts = str(label_str).split()
    y = np.zeros(len(classes), dtype=np.float32)
    for p in parts:
        if p in class_to_idx:
            y[class_to_idx[p]] = 1.0
    return y


Y = np.stack([labels_to_multihot(s) for s in train["labels"].values], axis=0)
new_train = pd.concat([train[["image"]], pd.DataFrame(Y, columns=CLASSES)], axis=1)

new_train.head()



## === cell 8
new_train



## === cell 9
IMAGE_SIZE = (512, 512)


@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.string)])
def read_image_bytes(filename):
    return tf.io.read_file(filename)


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(len(CLASSES),), dtype=tf.float32),
    ]
)
def decode_image_with_label_from_bytes(bits, label):
    image = tf.image.decode_jpeg(bits, channels=3, dct_method="INTEGER_FAST")
    image = tf.image.convert_image_dtype(image, tf.float32)  # exact /255.0
    image = tf.image.resize(
        image, IMAGE_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    return image, label


@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.string)])
def decode_image_no_label_from_bytes(bits):
    image = tf.image.decode_jpeg(bits, channels=3, dct_method="INTEGER_FAST")
    image = tf.image.convert_image_dtype(image, tf.float32)  # exact /255.0
    image = tf.image.resize(
        image, IMAGE_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    return image


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(len(CLASSES),), dtype=tf.float32),
    ]
)
def decode_image_with_label(filename, label):
    bits = tf.io.read_file(filename)
    return decode_image_with_label_from_bytes(bits, label)


@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.string)])
def decode_image_no_label(filename):
    bits = tf.io.read_file(filename)
    return decode_image_no_label_from_bytes(bits)




## === cell 10
test_paths[:5]



## === cell 11
BATCH_SIZE = 64



## === cell 12
test_dataset = (
    tf.data.Dataset.from_tensor_slices(tf.constant(test_paths, dtype=tf.string))
    .with_options(data_opts)
    .map(read_image_bytes, num_parallel_calls=AUTO, deterministic=True)
    .map(decode_image_no_label_from_bytes, num_parallel_calls=AUTO, deterministic=True)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)



## === cell 13
import tensorflow as tf
from tensorflow import keras



## === cell 14
train["filepath"] = (train_image_dir + train["image"].astype(str)).values

from sklearn.model_selection import train_test_split

train_df, val_df, y_train, y_val = train_test_split(
    train[["image", "filepath"]], Y, test_size=0.1, random_state=SEED, shuffle=True
)

print(train_df.shape, val_df.shape, y_train.shape, y_val.shape)




## === cell 15
def make_dataset(df, y=None, training=False, cache_in_memory=False):
    filepaths = tf.constant(df["filepath"].astype(str).values, dtype=tf.string)

    if y is None:
        ds = tf.data.Dataset.from_tensor_slices(filepaths).with_options(data_opts)
        ds = ds.map(read_image_bytes, num_parallel_calls=AUTO, deterministic=True)
        if cache_in_memory:
            ds = ds.cache()
        ds = ds.map(
            decode_image_no_label_from_bytes,
            num_parallel_calls=AUTO,
            deterministic=True,
        )
    else:
        y = np.asarray(y, dtype=np.float32, order="C")
        ds = tf.data.Dataset.from_tensor_slices((filepaths, y)).with_options(data_opts)
        ds = ds.map(
            lambda fp, lab: (read_image_bytes(fp), lab),
            num_parallel_calls=AUTO,
            deterministic=True,
        )
        if cache_in_memory:
            ds = ds.cache()
        ds = ds.map(
            decode_image_with_label_from_bytes,
            num_parallel_calls=AUTO,
            deterministic=True,
        )

    if training:
        ds = ds.shuffle(4096, seed=SEED, reshuffle_each_iteration=True)
        ds = ds.batch(BATCH_SIZE, drop_remainder=True)
    else:
        ds = ds.batch(BATCH_SIZE, drop_remainder=False)

    ds = ds.prefetch(AUTO)
    return ds


train_ds = make_dataset(train_df, y_train, training=True, cache_in_memory=True)
val_ds = make_dataset(val_df, y_val, training=False, cache_in_memory=True)



## === cell 16
base = keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    pooling="avg",
)
base.trainable = False

inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
x = keras.applications.resnet.preprocess_input(inputs * 255.0)
x = base(x, training=False)
outputs = keras.layers.Dense(len(CLASSES), activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    jit_compile=False,
)

model.summary()



## === cell 17
EPOCHS = 2
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)



## === cell 18
probs = model.predict(test_dataset, verbose=1)
probs.shape



## === cell 19
temp_probs = probs



## === cell 20
temp_probs[:2]



## === cell 21
name = {
    0: "scab",
    1: "frog_eye_leaf_spot",
    2: "complex",
    3: "rust",
    4: "powdery_mildew",
    5: "healthy",
}

threshold = {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1}

thr = np.array([threshold[i] for i in range(5)], dtype=np.float32)
p5 = temp_probs[:, :5].astype(np.float32, copy=False)

mask = p5 > thr[None, :]
count = mask.sum(axis=1).astype(np.int32)
argm = p5.argmax(axis=1).astype(np.int32)

pred_string = []
pred_string_append = pred_string.append  # minor speedup, identical result

for j in range(p5.shape[0]):
    if count[j] >= 3:
        key = int(argm[j])
        if name[key] != "complex":
            s = name[key] + " " + "complex"
        else:
            s = "complex"
    else:
        parts = []
        rowmask = mask[j]
        if rowmask[0]:
            parts.append(name[0])
        if rowmask[1]:
            parts.append(name[1])
        if rowmask[2]:
            parts.append(name[2])
        if rowmask[3]:
            parts.append(name[3])
        if rowmask[4]:
            parts.append(name[4])
        s = " ".join(parts)
        if s == "":
            s = name[5]
    pred_string_append(s.strip())



## === cell 22
pred_df = pd.DataFrame(
    {"image": [os.path.basename(p) for p in test_paths], "labels": pred_string}
)

test = sub.copy()
test = test.merge(pred_df, on="image", how="left", suffixes=("", "_pred"))
test["labels"] = test["labels_pred"].fillna("healthy")
test = test[["image", "labels"]]

test.to_csv("submission.csv", index=False)
test.head()



## === cell 23
print("Wrote submission.csv with shape:", test.shape)
print(test["labels"].value_counts().head(10))
