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
TRAIN_IMG_LOC = "../input/cassava-leaf-disease-classification/train_images"
TEST_IMG_LOC = "../input/cassava-leaf-disease-classification/test_images"
TRAIN_CSV = "../input/cassava-leaf-disease-classification/train.csv"
SAMPLE_CSV = "../input/cassava-leaf-disease-classification/sample_submission.csv"

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(False)
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

print("TensorFlow:", tf.__version__)
print("ALL Modules are successfully loaded")




## === cell 1
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

IMG_SIZE = (300, 300)  # keep same as original inference smart_resize target
BATCH_SIZE = 16
NUM_CLASSES = 5
EPOCHS = 5  # keep as-is to preserve training semantics

STEPS_PER_EXECUTION = 32


def build_model():
    base = EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    return model, base


model, base_model = build_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
    steps_per_execution=STEPS_PER_EXECUTION,  # runtime-only optimization
)
print("Model build/compile complete")




## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
train_df["filepath"] = TRAIN_IMG_LOC + "/" + train_df["image_id"].astype(str)

from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
train_idx, val_idx = next(sss.split(train_df["filepath"], train_df["label"]))
tr_df = train_df.iloc[train_idx].reset_index(drop=True)
va_df = train_df.iloc[val_idx].reset_index(drop=True)

print("Train size:", len(tr_df), "Val size:", len(va_df))
print("Label distribution (train):")
print(tr_df["label"].value_counts().sort_index())

aug = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal", seed=SEED),
        layers.RandomRotation(0.05, seed=SEED),
        layers.RandomZoom(0.1, seed=SEED),
    ],
    name="aug",
)

ds_options = tf.data.Options()
ds_options.experimental_deterministic = True
try:
    ds_options.experimental_optimization.apply_default_optimizations = True
    ds_options.experimental_optimization.autotune_buffers = True
    ds_options.experimental_optimization.map_fusion = True
    ds_options.experimental_optimization.parallel_batch = True
except Exception:
    pass

TRAIN_TFREC_GLOB = (
    "../input/cassava-leaf-disease-classification/train_tfrecords/*.tfrec"
)
TEST_TFREC_GLOB = "../input/cassava-leaf-disease-classification/test_tfrecords/*.tfrec"
train_tfrec_files = sorted(tf.io.gfile.glob(TRAIN_TFREC_GLOB))
test_tfrec_files = sorted(tf.io.gfile.glob(TEST_TFREC_GLOB))

_TFREC_FEATURES_TRAIN = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}
_TFREC_FEATURES_TEST = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


@tf.function
def _decode_resize_from_bytes(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, IMG_SIZE, method="bilinear")
    img = tf.cast(img, tf.float32)
    img = tf.ensure_shape(img, (IMG_SIZE[0], IMG_SIZE[1], 3))
    return img


@tf.function
def _parse_tfrecord_train(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TFREC_FEATURES_TRAIN)
    img = _decode_resize_from_bytes(ex["image"])
    label = tf.cast(ex["target"], tf.int32)
    image_id = ex["image_name"]
    return img, label, image_id


@tf.function
def _parse_tfrecord_test(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TFREC_FEATURES_TEST)
    img = _decode_resize_from_bytes(ex["image"])
    image_id = ex["image_name"]
    return img, image_id


@tf.function
def _apply_aug(img, label, image_id):
    img = aug(img, training=True)
    return img, label, image_id


@tf.function
def _drop_id(img, label, image_id):
    return img, label


def _shard_index_from_image_id(series_image_id):
    s = series_image_id.astype(str).str.replace(".jpg", "", regex=False)
    return (s.astype(np.int64) // 100_000_000).astype(np.int32)


val_shards = set(_shard_index_from_image_id(va_df["image_id"]).tolist())
train_shards = set(_shard_index_from_image_id(tr_df["image_id"]).tolist())


def _select_tfrec_files_by_shard(all_files, shard_set):
    out = []
    for f in all_files:
        base = os.path.basename(f)
        shard = None
        if base.startswith("ld_train"):
            try:
                shard = int(base.split("ld_train", 1)[1].split("-", 1)[0])
            except Exception:
                shard = None
        if shard is not None and shard in shard_set:
            out.append(f)
    return out


train_tfrec_files_sel = _select_tfrec_files_by_shard(
    train_tfrec_files, train_shards | val_shards
)
val_tfrec_files_sel = _select_tfrec_files_by_shard(train_tfrec_files, val_shards)

val_ids = va_df["image_id"].astype(str).tolist()
train_ids = tr_df["image_id"].astype(str).tolist()

val_ids_tf = tf.constant(val_ids, dtype=tf.string)
train_ids_tf = tf.constant(train_ids, dtype=tf.string)

val_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=val_ids_tf,
        values=tf.ones([len(val_ids)], dtype=tf.int32),
        key_dtype=tf.string,
        value_dtype=tf.int32,
    ),
    default_value=tf.constant(0, dtype=tf.int32),
)

train_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=train_ids_tf,
        values=tf.ones([len(train_ids)], dtype=tf.int32),
        key_dtype=tf.string,
        value_dtype=tf.int32,
    ),
    default_value=tf.constant(0, dtype=tf.int32),
)


@tf.function
def _is_in_val(img, label, image_id):
    return tf.not_equal(val_table.lookup(image_id), 0)


@tf.function
def _is_in_train(img, label, image_id):
    return tf.not_equal(train_table.lookup(image_id), 0)


def make_split_ds_from_tfrecords(train_files_for_read, val_files_for_read):
    train_base = tf.data.TFRecordDataset(
        train_files_for_read, num_parallel_reads=tf.data.AUTOTUNE
    ).with_options(ds_options)
    train_base = train_base.map(
        _parse_tfrecord_train, num_parallel_calls=tf.data.AUTOTUNE
    )
    train_base = train_base.apply(tf.data.experimental.ignore_errors())
    train_stream = train_base.filter(
        lambda img, label, image_id: _is_in_train(img, label, image_id)
    )

    val_base = tf.data.TFRecordDataset(
        val_files_for_read, num_parallel_reads=tf.data.AUTOTUNE
    ).with_options(ds_options)
    val_base = val_base.map(_parse_tfrecord_train, num_parallel_calls=tf.data.AUTOTUNE)
    val_base = val_base.apply(tf.data.experimental.ignore_errors())
    val_stream = val_base.filter(
        lambda img, label, image_id: _is_in_val(img, label, image_id)
    )

    train_stream = train_stream.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    train_stream = train_stream.map(_apply_aug, num_parallel_calls=tf.data.AUTOTUNE)
    train_stream = train_stream.map(_drop_id, num_parallel_calls=tf.data.AUTOTUNE)
    train_stream = train_stream.batch(BATCH_SIZE, drop_remainder=True).prefetch(
        tf.data.AUTOTUNE
    )

    val_stream = val_stream.map(_drop_id, num_parallel_calls=tf.data.AUTOTUNE)
    val_stream = val_stream.cache()
    val_stream = val_stream.batch(BATCH_SIZE, drop_remainder=False).prefetch(
        tf.data.AUTOTUNE
    )

    return train_stream, val_stream


train_ds, val_ds = make_split_ds_from_tfrecords(
    train_tfrec_files_sel, val_tfrec_files_sel
)
print("Built train_ds and val_ds")
print(
    f"Selected TFREC files -> train_read: {len(train_tfrec_files_sel)} / {len(train_tfrec_files)}, "
    f"val_read: {len(val_tfrec_files_sel)} / {len(train_tfrec_files)}"
)




## === cell 3
base_model.trainable = False
history1 = model.fit(
    train_ds, validation_data=val_ds, epochs=max(1, EPOCHS // 2), verbose=2
)

base_model.trainable = True
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
    steps_per_execution=STEPS_PER_EXECUTION,  # runtime-only optimization
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    initial_epoch=max(1, EPOCHS // 2),
    verbose=2,
)

print("Training complete")




## === cell 4
ss = pd.read_csv(SAMPLE_CSV)


def make_test_ds_with_ids(tfrec_files):
    ds = tf.data.TFRecordDataset(
        tfrec_files, num_parallel_reads=tf.data.AUTOTUNE
    ).with_options(ds_options)
    ds = ds.map(_parse_tfrecord_test, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


test_ds_full = make_test_ds_with_ids(test_tfrec_files)

all_ids = []
all_preds = []

for img_batch, ids_batch in test_ds_full:
    probs_batch = model(img_batch, training=False).numpy()
    all_preds.append(np.argmax(probs_batch, axis=1).astype(np.int64))
    all_ids.append(ids_batch.numpy())

preds = np.concatenate(all_preds, axis=0).astype(int)
test_image_ids = np.concatenate(all_ids, axis=0).astype("U")

pred_df = pd.DataFrame({"image_id": test_image_ids.astype(object), "label": preds})

my_submission = ss[["image_id"]].merge(pred_df, on="image_id", how="left")
my_submission["label"] = my_submission["label"].fillna(0).astype(int)

my_submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", my_submission.shape)
print(my_submission.head())
