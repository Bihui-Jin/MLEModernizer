# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.13

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import re
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import tensorflow as tf

SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

_CPU_COUNT = os.cpu_count() or 4
FIT_WORKERS = min(8, _CPU_COUNT)  # compatibility; not used with Keras 3 trainer
PRED_WORKERS = min(8, _CPU_COUNT)  # compatibility; not used with Keras 3 trainer




## === cell 1
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

label_to_disease = pd.read_json(
    "/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json",
    typ="series",
)
train_csv = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/train.csv")

train_csv["disease"] = train_csv["label"].map(label_to_disease)
train_csv["path"] = (
    "/kaggle/input/cassava-leaf-disease-classification/train_images/"
    + train_csv["image_id"]
)

train_csv["label_int"] = train_csv["label"].astype(int)
train_csv["label_str"] = train_csv["label_int"].astype(str)

train, valid = train_test_split(
    train_csv, test_size=0.2, stratify=train_csv["label_str"], random_state=SEED
)

IMG_SIZE = (224, 224)
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_VALID = 32
NUM_CLASSES = 5

class_names = [str(i) for i in range(NUM_CLASSES)]
table_init = tf.lookup.KeyValueTensorInitializer(
    keys=tf.constant(class_names),
    values=tf.constant(list(range(NUM_CLASSES)), dtype=tf.int32),
)
label_lookup = tf.lookup.StaticHashTable(table_init, default_value=-1)


@tf.function
def _decode_and_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(
        img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    return tf.cast(img, tf.float32)


_rand_rot = tf.keras.layers.RandomRotation(
    factor=45.0 / 360.0, fill_mode="nearest", seed=SEED
)
_rand_trans = tf.keras.layers.RandomTranslation(
    height_factor=0.2, width_factor=0.2, fill_mode="nearest", seed=SEED
)
_rand_zoom = tf.keras.layers.RandomZoom(
    height_factor=(-0.2, 0.2),
    width_factor=(-0.2, 0.2),
    fill_mode="nearest",
    seed=SEED,
)


@tf.function
def _augment(img, seed2):
    seeds = tf.random.experimental.stateless_split(seed2, 3)
    seed_rot = seeds[0]
    seed_flip = seeds[1]
    seed_aff = seeds[2]

    angle = tf.random.stateless_uniform(
        [], seed_rot, minval=-0.78539816339, maxval=0.78539816339
    )
    try:
        img = tf.image.rotate(img, angle, fill_mode="nearest")
    except Exception:
        img = _rand_rot(tf.expand_dims(img, 0), training=True)[0]

    x = tf.expand_dims(img, 0)
    x = _rand_trans(x, training=True)
    x = _rand_zoom(x, training=True)
    img = x[0]

    img = tf.image.stateless_random_flip_left_right(img, seed_flip)
    img = tf.image.stateless_random_flip_up_down(img, seed_aff)
    return img


@tf.function
def _to_onehot(label_str):
    idx = label_lookup.lookup(label_str)
    return tf.one_hot(idx, depth=NUM_CLASSES, dtype=tf.float32)


import zlib


def _stable_int32_seed_from_paths(paths_np, seed=SEED):
    out = np.empty((len(paths_np), 2), dtype=np.int32)
    for i, p in enumerate(paths_np):
        out[i, 0] = np.int32(zlib.adler32(str(p).encode("utf-8")) & 0x7FFFFFFF)
    out[:, 1] = np.int32(seed)
    return out


_DS_OPTS = tf.data.Options()
_DS_OPTS.experimental_deterministic = True


def make_train_ds(df):
    paths = df["path"].to_numpy()
    labels = df["label_str"].to_numpy()
    seeds_np = _stable_int32_seed_from_paths(paths, seed=SEED)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels, seeds_np))
    ds = ds.with_options(_DS_OPTS)

    shuffle_buf = min(len(df), 4096)
    ds = ds.shuffle(shuffle_buf, seed=SEED, reshuffle_each_iteration=True)

    def _map_all(path, label_str, seed2):
        img = _decode_and_resize(path)
        img = _augment(img, seed2)
        img = preprocess_input(img)
        y = _to_onehot(label_str)
        return img, y

    ds = ds.map(_map_all, num_parallel_calls=tf.data.AUTOTUNE)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels, seeds_np)).with_options(
        _DS_OPTS
    )
    ds = ds.shuffle(shuffle_buf, seed=SEED, reshuffle_each_iteration=True)

    def _decode_only(path, label_str, seed2):
        img = _decode_and_resize(path)
        return img, label_str, seed2

    ds = ds.map(_decode_only, num_parallel_calls=tf.data.AUTOTUNE)

    cache_path = "/kaggle/working/train_decode_cache.tfdata"
    ds = ds.cache(cache_path)

    def _final_map(img, label_str, seed2):
        img = _augment(img, seed2)
        img = preprocess_input(img)
        y = _to_onehot(label_str)
        return img, y

    ds = ds.map(_final_map, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.batch(BATCH_SIZE_TRAIN, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def make_valid_ds(df):
    paths = df["path"].to_numpy()
    labels = df["label_str"].to_numpy()
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.with_options(_DS_OPTS)

    def _map_fn(path, label_str):
        img = _decode_and_resize(path)
        img = preprocess_input(img)
        y = _to_onehot(label_str)
        return img, y

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)

    cache_path = "/kaggle/working/valid_cache.tfdata"
    ds = ds.cache(cache_path)

    ds = ds.batch(BATCH_SIZE_VALID, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = make_train_ds(train)
valid_ds = make_valid_ds(valid)




## === cell 2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback


class EarlyStoppingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if self.model.stop_training:
            print(f"Early stopping triggered at epoch {epoch + 1}.")


early_stopping = EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

learning_rate_reduction = ReduceLROnPlateau(
    monitor="val_loss", patience=2, factor=0.5, min_lr=1e-6, verbose=1
)




## === cell 3
num_classes = NUM_CLASSES  # should be 5

base_model = EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3)
)
base_model.trainable = False  # keep it light/fast; avoids long runtimes

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
outputs = Dense(num_classes, activation="softmax")(x)

trained_model = Model(inputs=base_model.input, outputs=outputs)
trained_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

EPOCHS = 3

history = trained_model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping, learning_rate_reduction],
    verbose=1,
)

cropnet_model = trained_model
old_densenet_model = trained_model
efficientnet_model = trained_model




## === cell 4
import numpy as np
import pandas as pd

sample_sub = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
image_names = sample_sub["image_id"].tolist()

image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
img_size = (224, 224)

test_df = pd.DataFrame(
    {"image_id": image_names, "path": [os.path.join(image_dir, n) for n in image_names]}
)


def make_test_ds(df):
    paths = df["path"].to_numpy()
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.with_options(_DS_OPTS)

    def _map_fn(path):
        img = _decode_and_resize(path)
        img = preprocess_input(img)
        return img

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(64, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


test_ds = make_test_ds(test_df)

pred_proba = trained_model.predict(
    test_ds,
    verbose=1,
)

predictions = pred_proba.argmax(axis=1).astype(int).tolist()

submission_df = pd.DataFrame({"image_id": image_names, "label": predictions})

submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission file created: {submission_path}")
print(submission_df.head())
