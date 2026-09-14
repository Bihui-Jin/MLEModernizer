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
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
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
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import gc
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.xception import preprocess_input

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

INPUT_DIR = "/kaggle/input/plant-pathology-2020-fgvc7"
IMG_DIR = os.path.join(INPUT_DIR, "images")

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 16
IMG_W, IMG_H = (410, 273)  # keep identical resize target (width=410, height=273)



## === cell 1
gc.collect()

train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))
sample_sub = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))

train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

target_cols = ["healthy", "multiple_diseases", "rust", "scab"]
assert all(c in train_df.columns for c in ["image_id"] + target_cols)
assert list(sample_sub.columns) == ["image_id"] + target_cols

train_x_images = train_df["image_id"].values
train_y_multi = train_df[target_cols].values.astype(np.int32)
training_y = np.argmax(train_y_multi, axis=1).astype(np.int32)



## === cell 2
gc.collect()

train_ids_all = train_x_images  # shuffled already
train_labels_all = training_y

train_ids = train_ids_all[0:1120]
train_y = train_labels_all[0:1120]

val_ids = train_ids_all[1120:1821]
val_y = train_labels_all[1120:1821]

y_binary_train = to_categorical(train_y, num_classes=4).astype(np.float32)
y_binary_val = to_categorical(val_y, num_classes=4).astype(np.float32)


def _build_paths(ids_np):
    ids_np = ids_np.astype(str)
    return np.char.add(np.char.add(IMG_DIR + os.sep, ids_np), ".jpg")


train_paths = _build_paths(train_ids)
val_paths = _build_paths(val_ids)


@tf.function(reduce_retracing=True)
def _decode_resize_preprocess(path, y):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [IMG_H, IMG_W], method=tf.image.ResizeMethod.AREA, antialias=True
    )
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)  # identical preprocessing
    img.set_shape([IMG_H, IMG_W, 3])
    return img, y




## === cell 3
gc.collect()

augmenter = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(factor=1.0, fill_mode="nearest", seed=42),
        tf.keras.layers.RandomTranslation(
            height_factor=0.2, width_factor=0.2, fill_mode="nearest", seed=42
        ),
        tf.keras.layers.RandomZoom(
            height_factor=(-0.2, 0.2),
            width_factor=(-0.2, 0.2),
            fill_mode="nearest",
            seed=42,
        ),
        tf.keras.layers.RandomFlip(mode="horizontal", seed=42),
    ],
    name="augmenter",
)


@tf.function(reduce_retracing=True)
def _augment_only(img, y):
    img = augmenter(img, training=True)
    img.set_shape([IMG_H, IMG_W, 3])
    return img, y


train_opts = tf.data.Options()
train_opts.deterministic = True

val_opts = tf.data.Options()
val_opts.deterministic = True

train_decoded_cached = (
    tf.data.Dataset.from_tensor_slices((train_paths, y_binary_train))
    .with_options(train_opts)
    .shuffle(buffer_size=len(train_paths), seed=42, reshuffle_each_iteration=True)
    .map(_decode_resize_preprocess, num_parallel_calls=AUTOTUNE)
    .cache()
)

train_ds = (
    train_decoded_cached.map(_augment_only, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((val_paths, y_binary_val))
    .with_options(val_opts)
    .map(_decode_resize_preprocess, num_parallel_calls=AUTOTUNE)
    .cache()
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)



## === cell 4
gc.collect()

model = tf.keras.Sequential(
    [
        tf.keras.applications.Xception(
            weights="imagenet", include_top=False, input_shape=(IMG_H, IMG_W, 3)
        ),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(4, activation=tf.nn.softmax),
    ]
)

model.layers[0].trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adamax(),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    jit_compile=True,
    steps_per_execution=64,
)

annealer = ReduceLROnPlateau(
    monitor="val_accuracy", factor=0.5, patience=5, verbose=1, min_lr=1e-5
)

checkpoint = ModelCheckpoint(
    "model.h5", verbose=1, save_best_only=True, monitor="val_accuracy", mode="max"
)


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if logs.get("val_accuracy", 0) > 0.99:
            print("\nReached 99% validation accuracy so cancelling training!")
            self.model.stop_training = True




## === cell 5
gc.collect()

MAX_EPOCHS_CAP = 40

steps_per_epoch = int(np.ceil(len(train_paths) / BATCH_SIZE))
validation_steps = int(np.ceil(len(val_paths) / BATCH_SIZE))

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=MAX_EPOCHS_CAP,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[annealer, checkpoint, myCallback()],
    verbose=1,
)



## === cell 6
print("Skipping plots for performance.")



## === cell 7
print("Skipping loss plot for performance.")



## === cell 8
gc.collect()

test_ids = test_df["image_id"].values
test_paths = _build_paths(test_ids)


@tf.function(reduce_retracing=True)
def _decode_resize_preprocess_x(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [IMG_H, IMG_W], method=tf.image.ResizeMethod.AREA, antialias=True
    )
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    img.set_shape([IMG_H, IMG_W, 3])
    return img


test_opts = tf.data.Options()
test_opts.deterministic = True

test_ds = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .with_options(test_opts)
    .map(_decode_resize_preprocess_x, num_parallel_calls=AUTOTUNE)
    .cache()
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)



## === cell 9
gc.collect()

results = model.predict(test_ds, verbose=1)



## === cell 10
df = pd.DataFrame(results, columns=target_cols)
df.insert(0, "image_id", test_ids)
df = df[["image_id"] + target_cols]



## === cell 11
out_path = "submission.csv"
df.to_csv(out_path, index=False)
print(f"Wrote submission to: {out_path} with shape {df.shape}")



## === cell 12
df.head(10)



## === cell 13
assert (
    df.shape[0] == sample_sub.shape[0]
), f"Row mismatch: {df.shape[0]} vs {sample_sub.shape[0]}"
assert list(df.columns) == list(
    sample_sub.columns
), f"Column mismatch: {df.columns} vs {sample_sub.columns}"
df.describe(include="all")
