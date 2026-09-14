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

3.9

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
pillow==11.3.0
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import layers, models

tf.keras.utils.set_random_seed(42)
np.random.seed(42)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE



## === cell 1
base_dir = "../input/cassava-leaf-disease-classification"
os.listdir(base_dir)



## === cell 2
train_labels = pd.read_csv(os.path.join(base_dir, "train.csv"))
train_labels.head()



## === cell 3
BATCH_SIZE = 20
EPOCHS = 10
TARGET_SIZE = 224

STEPS_PER_EPOCH = int((len(train_labels) * 0.8) // BATCH_SIZE)
VALIDATION_STEPS = int((len(train_labels) * 0.2) // BATCH_SIZE)
STEPS_PER_EPOCH, VALIDATION_STEPS



## === cell 4
train_labels = train_labels.copy()
train_labels["label"] = train_labels["label"].astype(np.int32)

idx = np.arange(len(train_labels))
rng = np.random.RandomState(42)
rng.shuffle(idx)

n_train = int(np.floor(len(idx) * 0.8))
train_idx = idx[:n_train]
val_idx = idx[n_train:]

train_df = train_labels.iloc[train_idx].reset_index(drop=True)
val_df = train_labels.iloc[val_idx].reset_index(drop=True)

train_img_dir = os.path.join(base_dir, "train_images")
train_paths = (train_img_dir + "/" + train_df["image_id"].values).astype(str)
train_y = train_df["label"].values.astype(np.int32)

val_paths = (train_img_dir + "/" + val_df["image_id"].values).astype(str)
val_y = val_df["label"].values.astype(np.int32)

augment = tf.keras.Sequential(
    [
        layers.RandomFlip(mode="horizontal_and_vertical", seed=42),
        layers.RandomRotation(factor=40.0 / 360.0, fill_mode="nearest", seed=42),
        layers.RandomZoom(
            height_factor=(-0.2, 0.2),
            width_factor=(-0.2, 0.2),
            fill_mode="nearest",
            seed=42,
        ),
        layers.RandomTranslation(
            height_factor=0.2, width_factor=0.2, fill_mode="nearest", seed=42
        ),
        layers.RandomShear(x_factor=0.2, y_factor=0.2, fill_mode="nearest", seed=42),
    ],
    name="augment",
)

TFREC_TRAIN_DIR = os.path.join(base_dir, "train_tfrecords")
TFREC_TEST_DIR = os.path.join(base_dir, "test_tfrecords")

train_tfrec_files = sorted(tf.io.gfile.glob(os.path.join(TFREC_TRAIN_DIR, "*.tfrec")))
test_tfrec_files = sorted(tf.io.gfile.glob(os.path.join(TFREC_TEST_DIR, "*.tfrec")))

opts = tf.data.Options()
opts.experimental_deterministic = True
opts.experimental_optimization.map_parallelization = True
opts.experimental_optimization.parallel_batch = True
opts.experimental_optimization.apply_default_optimizations = True

CACHE_TRAIN_PATH = None
CACHE_VAL_PATH = None

_TRAIN_FEATURE_DESC = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}
_TEST_FEATURE_DESC = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


@tf.function
def _decode_resize_from_bytes(image_bytes):
    img = tf.image.decode_jpeg(image_bytes, channels=3)
    img = tf.image.resize(
        img, [TARGET_SIZE, TARGET_SIZE], method="bilinear", antialias=True
    )
    img = tf.cast(img, tf.float32)
    return img


@tf.function
def _parse_train_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TRAIN_FEATURE_DESC)
    img = _decode_resize_from_bytes(ex["image"])
    label = tf.cast(ex["target"], tf.int32)
    return img, label


@tf.function
def _parse_test_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TEST_FEATURE_DESC)
    img = _decode_resize_from_bytes(ex["image"])
    return img, ex["image_name"]


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(TARGET_SIZE, TARGET_SIZE, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    ]
)
def _augment_only(img, label):
    img = augment(img, training=True)
    return img, label


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    ]
)
def _decode_resize_from_path(path, label):
    img = tf.io.read_file(path)
    img = _decode_resize_from_bytes(img)
    return img, tf.cast(label, tf.int32)


SHUFFLE_BUFFER = min(len(train_paths), 4096)


def _maybe_cache(ds, cache_path):
    if cache_path is None:
        return ds.cache()
    return ds.cache(cache_path)


def _build_tfrecord_train_val_datasets(train_tfrec_files):
    files = sorted(train_tfrec_files)
    if not files:
        return None, None

    shard_size = 1338
    n_total = len(train_labels)
    n_train_ex = int(np.floor(n_total * 0.8))
    full_shards = n_train_ex // shard_size
    rem = n_train_ex % shard_size

    train_full_files = files[:full_shards]
    next_file = files[full_shards : full_shards + 1]
    val_files = files[full_shards + 1 :] if rem > 0 else files[full_shards:]

    ds_full = tf.data.TFRecordDataset(
        train_full_files, num_parallel_reads=AUTOTUNE
    ).with_options(opts)

    if rem > 0 and len(next_file) == 1:
        ds_next = tf.data.TFRecordDataset(next_file, num_parallel_reads=1).with_options(
            opts
        )
        ds_train = ds_full.concatenate(ds_next.take(rem))
        ds_val_head = ds_next.skip(rem)
        if val_files:
            ds_val = ds_val_head.concatenate(
                tf.data.TFRecordDataset(
                    val_files, num_parallel_reads=AUTOTUNE
                ).with_options(opts)
            )
        else:
            ds_val = ds_val_head
    else:
        ds_train = ds_full
        ds_val = (
            tf.data.TFRecordDataset(
                val_files, num_parallel_reads=AUTOTUNE
            ).with_options(opts)
            if val_files
            else tf.data.Dataset.from_tensor_slices([])
        )

    ds_train = ds_train.map(
        _parse_train_example, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    ds_val = ds_val.map(
        _parse_train_example, num_parallel_calls=AUTOTUNE, deterministic=True
    )

    return ds_train, ds_val


if len(train_tfrec_files) > 0:
    train_ds_parsed, val_ds_parsed = _build_tfrecord_train_val_datasets(
        train_tfrec_files
    )

    train_ds = (
        _maybe_cache(train_ds_parsed, CACHE_TRAIN_PATH)
        .shuffle(buffer_size=SHUFFLE_BUFFER, seed=42, reshuffle_each_iteration=True)
        .repeat()
        .map(_augment_only, num_parallel_calls=AUTOTUNE, deterministic=True)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(AUTOTUNE)
    )

    val_ds = (
        _maybe_cache(val_ds_parsed, CACHE_VAL_PATH)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(AUTOTUNE)
    )
else:
    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_y)).with_options(
        opts
    )
    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_y)).with_options(opts)

    val_ds = (
        val_ds.map(
            _decode_resize_from_path, num_parallel_calls=AUTOTUNE, deterministic=True
        )
        .cache()  # in-memory cache
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(AUTOTUNE)
    )

    train_ds = (
        train_ds.shuffle(
            buffer_size=SHUFFLE_BUFFER, seed=42, reshuffle_each_iteration=True
        )
        .map(_decode_resize_from_path, num_parallel_calls=AUTOTUNE, deterministic=True)
        .cache()  # in-memory cache
        .repeat()
        .map(_augment_only, num_parallel_calls=AUTOTUNE, deterministic=True)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(AUTOTUNE)
    )



## === cell 5
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications.efficientnet import preprocess_input

eff_base = EfficientNetB7(
    include_top=False,
    weights="imagenet",
    input_shape=(TARGET_SIZE, TARGET_SIZE, 3),
)

eff_base.trainable = False



## === cell 6
model = models.Sequential(
    [
        layers.Input(shape=(TARGET_SIZE, TARGET_SIZE, 3)),
        layers.Lambda(preprocess_input, name="effnet_preprocess"),
        eff_base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(5, activation="softmax", name="Output"),
    ]
)



## === cell 7
model.compile(
    optimizer="Adam",
    loss="sparse_categorical_crossentropy",
    metrics=["acc"],
    steps_per_execution=32,
)



## === cell 8
model_save = ModelCheckpoint(
    "./EffNetB7_best_weights.weights.h5",
    save_best_only=True,
    save_weights_only=True,
    monitor="val_loss",
    mode="min",
    verbose=1,
)

early_stop = EarlyStopping(
    monitor="val_loss",
    min_delta=0.001,
    patience=5,
    mode="min",
    verbose=1,
    restore_best_weights=True,
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=2,
    min_delta=0.001,
    mode="min",
    verbose=1,
)



## === cell 9
history = model.fit(
    train_ds,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    validation_data=val_ds,
    validation_steps=VALIDATION_STEPS,
    callbacks=[model_save, early_stop, reduce_lr],
)



## === cell 10
sub = pd.read_csv(os.path.join(base_dir, "sample_submission.csv"))
sub.head()



## === cell 11
test_dir = os.path.join(base_dir, "test_images")
test_paths = (test_dir + "/" + sub["image_id"].values).astype(str)


@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.string)])
def _load_test_image_from_path(path):
    img = tf.io.read_file(path)
    img = _decode_resize_from_bytes(img)
    return img


if len(test_tfrec_files) > 0:
    test_ds = tf.data.TFRecordDataset(
        test_tfrec_files, num_parallel_reads=AUTOTUNE
    ).with_options(opts)
    test_ds = test_ds.map(
        _parse_test_example, num_parallel_calls=AUTOTUNE, deterministic=True
    )

    names = np.fromiter(
        (n.decode("utf-8") for n in test_ds.map(lambda x, n: n).as_numpy_iterator()),
        dtype=object,
        count=len(sub),
    )

    test_img_ds = (
        test_ds.map(lambda x, n: x, num_parallel_calls=AUTOTUNE, deterministic=True)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTOTUNE)
    )

    probs = model.predict(test_img_ds, verbose=1)
    preds = np.argmax(probs, axis=1).astype(int)

    pred_df = pd.DataFrame({"image_id": names, "label": preds})
    sub = sub.drop(columns=["label"]).merge(pred_df, on="image_id", how="left")
else:
    test_ds = tf.data.Dataset.from_tensor_slices(test_paths).with_options(opts)
    test_ds = (
        test_ds.map(
            _load_test_image_from_path, num_parallel_calls=AUTOTUNE, deterministic=True
        )
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTOTUNE)
    )

    probs = model.predict(test_ds, verbose=1)
    preds = np.argmax(probs, axis=1).astype(int)
    sub["label"] = preds

sub = sub[["image_id", "label"]]
sub.head()



## === cell 12
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.dtypes)
print(sub.head())
