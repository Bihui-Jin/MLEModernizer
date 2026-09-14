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
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import glob
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import EfficientNetB3

SEED = 42
DEBUG = False

tf.random.set_seed(SEED)
np.random.seed(SEED)

DATA_DIR = "../input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
TRAIN_TFREC_DIR = os.path.join(DATA_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(DATA_DIR, "test_tfrecords")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing: {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing: {TEST_IMG_DIR}"
assert os.path.isdir(TRAIN_TFREC_DIR), f"Missing: {TRAIN_TFREC_DIR}"
assert os.path.isdir(TEST_TFREC_DIR), f"Missing: {TEST_TFREC_DIR}"

AUTOTUNE = tf.data.AUTOTUNE

print("TensorFlow:", tf.__version__)
print("DATA_DIR:", DATA_DIR)




## === cell 1
IMG_SIZE = (300, 300)
BATCH_SIZE = 32
EPOCHS = 4  # unchanged
NUM_CLASSES = 5
val_frac = 0.1

df_train = pd.read_csv(TRAIN_CSV)
df_train["image_id"] = df_train["image_id"].astype(str)
df_train = df_train.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
val_size = int(len(df_train) * val_frac)
df_val = df_train.iloc[:val_size].copy()
df_trn = df_train.iloc[val_size:].copy()

data_opts = tf.data.Options()
data_opts.experimental_deterministic = True
data_opts.experimental_optimization.apply_default_optimizations = True

data_opts.experimental_optimization.map_and_batch_fusion = True
data_opts.experimental_optimization.parallel_batch = True


def _decode_and_resize_from_path(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, IMG_SIZE, method="bilinear", antialias=False)
    img = tf.cast(img, tf.float32) / 255.0
    return img


def _decode_and_resize(image_bytes):
    img = tf.io.decode_jpeg(image_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, IMG_SIZE, method="bilinear", antialias=False)
    img = tf.cast(img, tf.float32) / 255.0
    return img


@tf.function
def _augment_image(img):
    img = tf.image.random_flip_left_right(img, seed=SEED)

    dx = tf.random.uniform([], -0.05, 0.05, seed=SEED)
    dy = tf.random.uniform([], -0.05, 0.05, seed=SEED + 1)
    tx = tf.cast(tf.round(dx * tf.cast(IMG_SIZE[1], tf.float32)), tf.int32)
    ty = tf.cast(tf.round(dy * tf.cast(IMG_SIZE[0], tf.float32)), tf.int32)
    img = tf.roll(img, shift=[ty, tx], axis=[0, 1])

    z = tf.random.uniform([], 0.9, 1.1, seed=SEED + 2)
    new_h = tf.cast(tf.round(z * tf.cast(IMG_SIZE[0], tf.float32)), tf.int32)
    new_w = tf.cast(tf.round(z * tf.cast(IMG_SIZE[1], tf.float32)), tf.int32)
    img_zoom = tf.image.resize(img, [new_h, new_w], method="bilinear", antialias=False)
    img = tf.image.resize_with_crop_or_pad(img_zoom, IMG_SIZE[0], IMG_SIZE[1])
    return img


_random_rotation = tf.keras.layers.RandomRotation(
    factor=15 / 360.0, fill_mode="nearest", seed=SEED
)


@tf.function
def _augment_with_rotation(img):
    img = _augment_image(img)
    img = _random_rotation(img, training=True)
    return img


train_tfrecs = sorted(tf.io.gfile.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec")))
use_train_tfrecords = len(train_tfrecs) > 0

if use_train_tfrecords:
    trn_ids_tensor = tf.constant(df_trn["image_id"].values, dtype=tf.string)
    val_ids_tensor = tf.constant(df_val["image_id"].values, dtype=tf.string)

    trn_ids_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=trn_ids_tensor,
            values=tf.ones([tf.shape(trn_ids_tensor)[0]], dtype=tf.int32),
        ),
        default_value=0,
    )
    val_ids_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=val_ids_tensor,
            values=tf.ones([tf.shape(val_ids_tensor)[0]], dtype=tf.int32),
        ),
        default_value=0,
    )

    _feats_train_min = {
        "image": tf.io.FixedLenFeature([], tf.string, default_value=b""),
        "image_name": tf.io.FixedLenFeature([], tf.string, default_value=b""),
        "image_id": tf.io.FixedLenFeature([], tf.string, default_value=b""),
        "label": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
        "target": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
    }

    def _parse_train_min(example_proto):
        ex = tf.io.parse_single_example(example_proto, _feats_train_min)
        name = tf.where(
            tf.strings.length(ex["image_name"]) > 0, ex["image_name"], ex["image_id"]
        )
        lbl = tf.where(ex["label"] >= 0, ex["label"], ex["target"])
        lbl = tf.cast(lbl, tf.int32)
        return ex["image"], lbl, name

    def _is_in_trn(image_bytes, lbl, name):
        return tf.equal(trn_ids_table.lookup(name), 1)

    def _is_in_val(image_bytes, lbl, name):
        return tf.equal(val_ids_table.lookup(name), 1)

    def _decode_pair(image_bytes, lbl):
        img = _decode_and_resize(image_bytes)
        return img, lbl

    raw_ds = (
        tf.data.TFRecordDataset(train_tfrecs, num_parallel_reads=AUTOTUNE)
        .with_options(data_opts)
        .cache()
    )

    parsed_ds = raw_ds.map(_parse_train_min, num_parallel_calls=AUTOTUNE)

    ds_trn = (
        parsed_ds.shard(num_shards=2, index=0)
        .filter(_is_in_trn)
        .map(lambda b, y, n: (b, y), num_parallel_calls=AUTOTUNE)
    )
    ds_val = (
        parsed_ds.shard(num_shards=2, index=1)
        .filter(_is_in_val)
        .map(lambda b, y, n: (b, y), num_parallel_calls=AUTOTUNE)
    )

    ds_trn = ds_trn.shuffle(8192, seed=SEED, reshuffle_each_iteration=True)
    ds_trn = ds_trn.map(_decode_pair, num_parallel_calls=AUTOTUNE)
    ds_trn = ds_trn.map(
        lambda img, lbl: (_augment_with_rotation(img), lbl), num_parallel_calls=AUTOTUNE
    )
    ds_trn = ds_trn.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

    ds_val = ds_val.map(_decode_pair, num_parallel_calls=AUTOTUNE)
    ds_val = ds_val.cache()
    ds_val = ds_val.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

else:

    def _make_train_ds(df, training: bool):
        paths = np.array(
            [os.path.join(TRAIN_IMG_DIR, fn) for fn in df["image_id"].values],
            dtype=np.str_,
        )
        labels = df["label"].values.astype(np.int32)

        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.with_options(data_opts)

        if training:
            ds = ds.shuffle(8192, seed=SEED, reshuffle_each_iteration=True)

        def _load(path, lbl):
            img = _decode_and_resize_from_path(path)
            if training:
                img = _augment_with_rotation(img)
            return img, lbl

        ds = ds.map(_load, num_parallel_calls=AUTOTUNE)

        if not training:
            ds = ds.cache()

        ds = ds.batch(BATCH_SIZE, drop_remainder=False)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    ds_trn = _make_train_ds(df_trn, training=True)
    ds_val = _make_train_ds(df_val, training=False)

inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base = EfficientNetB3(include_top=False, weights="imagenet", input_tensor=inputs)
x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)
my_model = Model(inputs=inputs, outputs=outputs)

base.trainable = False
my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

my_model.fit(
    ds_trn,
    validation_data=ds_val,
    epochs=EPOCHS,
    verbose=1,
)




## === cell 2
sample = pd.read_csv(SAMPLE_SUB)
sample["image_id"] = sample["image_id"].astype(str)
sample_image_ids = sample["image_id"].tolist()

test_tfrecs = sorted(tf.io.gfile.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec")))

_feets_test = {
    "image": tf.io.FixedLenFeature([], tf.string, default_value=b""),
    "image_name": tf.io.FixedLenFeature([], tf.string, default_value=b""),
    "image_id": tf.io.FixedLenFeature([], tf.string, default_value=b""),
}


def _parse_test_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _feets_test)
    img = _decode_and_resize(ex["image"])
    name = tf.where(
        tf.strings.length(ex["image_name"]) > 0, ex["image_name"], ex["image_id"]
    )
    return img, name


use_test_tfrecords = len(test_tfrecs) > 0

if use_test_tfrecords:
    test_ds = tf.data.TFRecordDataset(
        test_tfrecs, num_parallel_reads=AUTOTUNE
    ).with_options(data_opts)
    test_ds = test_ds.map(_parse_test_example, num_parallel_calls=AUTOTUNE)

    test_ds_batched = test_ds.batch(128).prefetch(AUTOTUNE)

    name_batches = []
    for _, batch_names in test_ds_batched:
        name_batches.append(batch_names.numpy())
    pred_names = np.concatenate(name_batches, axis=0)
    pred_names = np.array([x.decode("utf-8") for x in pred_names], dtype=object)

    pred_probs = my_model.predict(
        test_ds_batched.map(lambda x, n: x, num_parallel_calls=AUTOTUNE), verbose=0
    )
    pred_labels = np.argmax(pred_probs, axis=-1).astype(int)

    pred_df = pd.DataFrame({"image_id": pred_names, "label": pred_labels})

    final_csv = sample[["image_id"]].merge(pred_df, on="image_id", how="left")
    if final_csv["label"].isna().any():
        fill_label = int(df_train["label"].mode().iloc[0])
        final_csv["label"] = final_csv["label"].fillna(fill_label).astype(int)
    else:
        final_csv["label"] = final_csv["label"].astype(int)
else:
    test_paths = [os.path.join(TEST_IMG_DIR, x) for x in sample_image_ids]
    ds_test = tf.data.Dataset.from_tensor_slices(test_paths)
    ds_test = ds_test.with_options(data_opts)

    def _load_test(path):
        img = _decode_and_resize_from_path(path)
        return img

    ds_test = ds_test.map(_load_test, num_parallel_calls=AUTOTUNE)
    ds_test = ds_test.batch(128).prefetch(AUTOTUNE)

    pred_test = my_model.predict(ds_test, verbose=1)
    pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

    final_csv = pd.DataFrame({"image_id": sample_image_ids, "label": pred_test_labels})

final_csv.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", final_csv.shape)
print(final_csv.head())




## === cell 3
final_csv.head()
