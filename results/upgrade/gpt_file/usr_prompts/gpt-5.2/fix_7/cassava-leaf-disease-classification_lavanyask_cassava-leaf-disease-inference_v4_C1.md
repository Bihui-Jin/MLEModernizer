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
import numpy as np
import pandas as pd

ROOT_DIR = "../input/cassava-leaf-disease-classification/"
TRAIN_CSV = os.path.join(ROOT_DIR, "train.csv")
TRAIN_DIR = os.path.join(ROOT_DIR, "train_images/")
TEST_DIR = os.path.join(ROOT_DIR, "test_images/")
SAMPLE_SUB = os.path.join(ROOT_DIR, "sample_submission.csv")

TRAIN_TFREC_DIR = os.path.join(ROOT_DIR, "train_tfrecords/")
TEST_TFREC_DIR = os.path.join(ROOT_DIR, "test_tfrecords/")

print("ROOT_DIR exists:", os.path.exists(ROOT_DIR))
print("TRAIN_CSV exists:", os.path.exists(TRAIN_CSV))
print("TRAIN_DIR exists:", os.path.exists(TRAIN_DIR))
print("TEST_DIR exists:", os.path.exists(TEST_DIR))
print("SAMPLE_SUB exists:", os.path.exists(SAMPLE_SUB))
print("TRAIN_TFREC_DIR exists:", os.path.exists(TRAIN_TFREC_DIR))
print("TEST_TFREC_DIR exists:", os.path.exists(TEST_TFREC_DIR))
print("ROOT_DIR list sample:", os.listdir(ROOT_DIR)[:10])



## === cell 1
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

print("TensorFlow:", tf.__version__)



## === cell 2
IMG_SIZE = 300
size = (IMG_SIZE, IMG_SIZE)

BATCH_SIZE = 32
EPOCHS = 3
NUM_CLASSES = 5



## === cell 3
train_df = pd.read_csv(TRAIN_CSV)
print(train_df.head())
print(train_df["label"].value_counts().sort_index())

from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(
    train_df,
    test_size=0.1,
    random_state=SEED,
    stratify=train_df["label"],
)

print("Train size:", len(train_df), "Val size:", len(val_df))



## === cell 4
AUTO = tf.data.AUTOTUNE


@tf.function
def _resize_normalize(img):
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


def _fast_deterministic_options():
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    return options


_TRAIN_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}
_TEST_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
}


def _parse_train_example_with_id(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TRAIN_FEATURES)
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = _resize_normalize(img)
    y = tf.cast(ex["target"], tf.int32)
    image_id = ex["image_name"]
    return image_id, img, y


def _drop_id(image_id, img, y):
    return img, y


def _parse_test_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TEST_FEATURES)
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = _resize_normalize(img)
    return img


def _list_tfrecords(tfrecord_dir, prefix):
    files = tf.io.gfile.glob(os.path.join(tfrecord_dir, f"{prefix}*.tfrec"))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(
            f"No TFRecord files found in {tfrecord_dir} with prefix {prefix}"
        )
    return files


def make_tfrecord_train_val_datasets(
    train_df, val_df, batch_size=BATCH_SIZE, shuffle=True
):
    train_ids = tf.constant(train_df["image_id"].values.astype("U"), dtype=tf.string)
    val_ids = tf.constant(val_df["image_id"].values.astype("U"), dtype=tf.string)

    train_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=train_ids,
            values=tf.ones([tf.shape(train_ids)[0]], dtype=tf.int32),
        ),
        default_value=0,
    )
    val_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=val_ids,
            values=tf.ones([tf.shape(val_ids)[0]], dtype=tf.int32),
        ),
        default_value=0,
    )

    def _is_in_train(image_id, img, y):
        return tf.equal(train_table.lookup(image_id), 1)

    def _is_in_val(image_id, img, y):
        return tf.equal(val_table.lookup(image_id), 1)

    files = _list_tfrecords(TRAIN_TFREC_DIR, "ld_train")

    ds_all = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO).with_options(
        _fast_deterministic_options()
    )
    ds_all = ds_all.map(
        _parse_train_example_with_id, num_parallel_calls=AUTO
    ).ignore_errors()

    ds_train = ds_all.filter(_is_in_train).map(_drop_id, num_parallel_calls=AUTO)
    if shuffle:
        ds_train = ds_train.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    ds_train = ds_train.batch(batch_size, drop_remainder=False).prefetch(AUTO)

    ds_val = ds_all.filter(_is_in_val).map(_drop_id, num_parallel_calls=AUTO)
    ds_val = ds_val.batch(batch_size, drop_remainder=False).prefetch(AUTO)

    return ds_train, ds_val


@tf.function
def _decode_resize_normalize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)  # RGB
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


def make_file_dataset(
    df, images_dir, shuffle=False, batch_size=BATCH_SIZE, cache_path=None
):
    full_paths = (images_dir + df["image_id"].values).astype("U")
    filenames = tf.constant(full_paths, dtype=tf.string)
    labels = tf.constant(df["label"].values.astype(np.int32), dtype=tf.int32)

    ds = tf.data.Dataset.from_tensor_slices((filenames, labels))
    ds = ds.with_options(_fast_deterministic_options())

    if shuffle:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)

    def _map_fn(p, y):
        return _decode_resize_normalize(p), y

    ds = ds.map(_map_fn, num_parallel_calls=AUTO).ignore_errors()

    if cache_path is not None:
        ds = ds.cache(cache_path)

    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTO)
    return ds


train_ds, val_ds = make_tfrecord_train_val_datasets(
    train_df, val_df, batch_size=BATCH_SIZE, shuffle=True
)



## === cell 5
inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()



## === cell 6
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=2,
)



## === cell 7
sample_sub = pd.read_csv(SAMPLE_SUB)
test_images = sample_sub["image_id"].tolist()


def make_test_tfrecord_dataset(batch_size=BATCH_SIZE):
    test_files = _list_tfrecords(TEST_TFREC_DIR, "ld_test")
    ds = tf.data.TFRecordDataset(test_files, num_parallel_reads=AUTO).with_options(
        _fast_deterministic_options()
    )
    ds = ds.map(_parse_test_example, num_parallel_calls=AUTO).ignore_errors()
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTO)
    return ds


test_ds = make_test_tfrecord_dataset(BATCH_SIZE)

probs = model.predict(test_ds, verbose=1)
preds = probs.argmax(axis=1).astype(int)

print("Num test images (from sample):", len(test_images), "Num preds:", len(preds))



## === cell 8
if len(preds) != len(test_images):
    raise RuntimeError(
        f"Prediction count {len(preds)} != sample_submission rows {len(test_images)}"
    )

sub = pd.DataFrame({"image_id": test_images, "label": preds})
print(sub.head())
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print("File exists:", os.path.exists("submission.csv"))
