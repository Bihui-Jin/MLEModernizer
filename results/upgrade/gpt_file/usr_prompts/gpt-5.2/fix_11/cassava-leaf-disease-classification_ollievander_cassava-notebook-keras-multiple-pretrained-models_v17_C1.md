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

INPUT_DIR = "../input/cassava-leaf-disease-classification/"
OUTPUT_DIR = "./"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(INPUT_DIR, "train_images")
TEST_PATH = os.path.join(INPUT_DIR, "test_images")




## === cell 1
import json
import math
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

A = None

from sklearn.model_selection import train_test_split

sns = None
plt = None

SEED = 100
tf.random.set_seed(SEED)
np.random.seed(SEED)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF decide
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(
        False
    )  # keep stable semantics; JIT may change numerics slightly
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE




## === cell 2
train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
train.head()




## === cell 3
with open(os.path.join(INPUT_DIR, "label_num_to_disease_map.json")) as f:
    classes = json.load(f)

train["class"] = train["label"].apply(lambda x: classes[str(x)])
train[["image_id", "label", "class"]].head()




## === cell 4
if sns is not None and plt is not None:
    plt.figure(figsize=(15, 7))
    _ = sns.countplot(x=train["class"], order=train["class"].value_counts().index)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.show()




## === cell 5
train = train.copy()
train = train.astype({"image_id": "str", "label": "str"})

train_df, val_df = train_test_split(
    train, test_size=0.05, random_state=SEED, stratify=train["label"].values
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

train_df.head(), val_df.head()




## === cell 6
batch_size = 4
IMG_SIZE = (512, 512)


def _transform_np_uint8(image_uint8: np.ndarray) -> np.ndarray:
    return image_uint8


class_names = sorted(train_df["label"].unique().tolist())
class_indices = {name: i for i, name in enumerate(class_names)}
inv_class_indices = {v: k for k, v in class_indices.items()}
NUM_CLASSES = len(class_names)


def _build_paths_and_labels(df: pd.DataFrame, image_dir: str):
    paths = (image_dir.rstrip("/") + "/" + df["image_id"].astype(str)).values
    labels = df["label"].map(class_indices).astype(np.int32).values
    return paths, labels


train_paths, train_labels = _build_paths_and_labels(train_df, TRAIN_PATH)
val_paths, val_labels = _build_paths_and_labels(val_df, TRAIN_PATH)

TRAIN_TFREC_DIR = os.path.join(INPUT_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(INPUT_DIR, "test_tfrecords")

train_tfrecs = sorted(tf.io.gfile.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec")))
test_tfrecs = sorted(tf.io.gfile.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec")))

_TFREC_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


def _decode_resize_and_rescale_from_bytes(img_bytes: tf.Tensor) -> tf.Tensor:
    img = tf.image.decode_jpeg(img_bytes, channels=3)  # uint8
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


def _decode_resize_uint8_from_bytes(img_bytes: tf.Tensor) -> tf.Tensor:
    img = tf.image.decode_jpeg(img_bytes, channels=3)  # uint8
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.uint8)
    return img


def _apply_albu_and_rescale(img_uint8: tf.Tensor) -> tf.Tensor:
    out = tf.numpy_function(_transform_np_uint8, [img_uint8], Tout=tf.uint8)
    out.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
    out = tf.cast(out, tf.float32) * (1.0 / 255.0)
    return out


def _one_hot(label_idx: tf.Tensor) -> tf.Tensor:
    return tf.one_hot(label_idx, depth=NUM_CLASSES, dtype=tf.float32)


def _decode_resize_and_rescale(path: tf.Tensor) -> tf.Tensor:
    img = tf.io.read_file(path)
    return _decode_resize_and_rescale_from_bytes(img)


def _decode_resize_uint8(path: tf.Tensor) -> tf.Tensor:
    img = tf.io.read_file(path)
    return _decode_resize_uint8_from_bytes(img)


def _train_map(path: tf.Tensor, label_idx: tf.Tensor):
    img = _decode_resize_and_rescale(path)
    y = _one_hot(label_idx)
    return img, y


def _val_map(path: tf.Tensor, label_idx: tf.Tensor):
    img = _decode_resize_and_rescale(path)
    y = _one_hot(label_idx)
    return img, y


_label_keys = tf.constant(list(class_indices.keys()), dtype=tf.string)
_label_vals = tf.constant(list(class_indices.values()), dtype=tf.int32)
_label_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(_label_keys, _label_vals),
    default_value=tf.constant(-1, tf.int32),
)


def _parse_train_tfrecord(example_proto: tf.Tensor):
    ex = tf.io.parse_single_example(example_proto, _TFREC_FEATURES)
    img_bytes = ex["image"]
    label_str = tf.strings.as_string(ex["target"])
    label_idx = _label_table.lookup(label_str)
    return img_bytes, label_idx


def _parse_test_tfrecord(example_proto: tf.Tensor):
    ex = tf.io.parse_single_example(example_proto, _TFREC_FEATURES)
    return ex["image"], ex["image_name"]


def _train_map_from_bytes(img_bytes: tf.Tensor, label_idx: tf.Tensor):
    img = _decode_resize_and_rescale_from_bytes(img_bytes)
    y = _one_hot(label_idx)
    return img, y


def _val_map_from_bytes(img_bytes: tf.Tensor, label_idx: tf.Tensor):
    img = _decode_resize_and_rescale_from_bytes(img_bytes)
    y = _one_hot(label_idx)
    return img, y


_ds_opts = tf.data.Options()
_ds_opts.deterministic = False


def _split_tfrecs_by_ratio(files, val_ratio=0.05):
    files = list(files)
    if not files:
        return [], []
    rng = np.random.RandomState(SEED)
    idx = np.arange(len(files))
    rng.shuffle(idx)
    files = [files[i] for i in idx.tolist()]
    n_val = max(1, int(round(len(files) * val_ratio)))
    val_files = files[:n_val]
    train_files = files[n_val:]
    if not train_files:  # extremely small file count safety
        train_files, val_files = files[:-1], files[-1:]
    return train_files, val_files


if train_tfrecs:
    tfrec_train_files, tfrec_val_files = _split_tfrecs_by_ratio(
        train_tfrecs, val_ratio=0.05
    )

    raw_train = tf.data.TFRecordDataset(
        tfrec_train_files, num_parallel_reads=AUTOTUNE
    ).with_options(_ds_opts)
    raw_val = tf.data.TFRecordDataset(
        tfrec_val_files, num_parallel_reads=AUTOTUNE
    ).with_options(_ds_opts)

    raw_train = raw_train.map(_parse_train_tfrecord, num_parallel_calls=AUTOTUNE)
    raw_val = raw_val.map(_parse_train_tfrecord, num_parallel_calls=AUTOTUNE)

    shuffle_buf = min(4096, len(train_paths))
    train_ds = raw_train.shuffle(
        buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True
    )
    train_ds = train_ds.repeat()
    train_ds = train_ds.map(_train_map_from_bytes, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.batch(batch_size, drop_remainder=False)
    train_ds = train_ds.prefetch(AUTOTUNE).with_options(_ds_opts)

    val_ds = raw_val.map(_val_map_from_bytes, num_parallel_calls=AUTOTUNE).cache()
    val_ds = val_ds.repeat()
    val_ds = val_ds.batch(batch_size, drop_remainder=False)
    val_ds = val_ds.prefetch(AUTOTUNE).with_options(_ds_opts)
else:
    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    shuffle_buf = min(4096, len(train_paths))
    train_ds = train_ds.shuffle(
        buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True
    )
    train_ds = train_ds.repeat()
    train_ds = train_ds.map(_train_map, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.batch(batch_size, drop_remainder=False)
    train_ds = train_ds.prefetch(AUTOTUNE).with_options(_ds_opts)

    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    val_ds = val_ds.map(_val_map, num_parallel_calls=AUTOTUNE).cache()
    val_ds = val_ds.repeat()
    val_ds = val_ds.batch(batch_size, drop_remainder=False)
    val_ds = val_ds.prefetch(AUTOTUNE).with_options(_ds_opts)

steps_per_epoch = int(math.ceil(len(train_df) / batch_size))
validation_steps = int(math.ceil(len(val_df) / batch_size))

NUM_CLASSES, list(class_indices.items())[:5]




## === cell 7
inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = layers.Conv2D(16, 3, activation="relu", padding="same")(inputs)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D()(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()




## === cell 8
EPOCHS = 2  # kept minimal as in original script to ensure runtime

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)




## === cell 9
sample_sub = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))
test_image_ids = sample_sub["image_id"].astype(str).tolist()

if test_tfrecs:
    test_raw = tf.data.TFRecordDataset(
        test_tfrecs, num_parallel_reads=AUTOTUNE
    ).with_options(_ds_opts)
    test_raw = test_raw.map(_parse_test_tfrecord, num_parallel_calls=AUTOTUNE)

    def _test_map(img_bytes, img_name):
        img = _decode_resize_and_rescale_from_bytes(img_bytes)
        return img, img_name

    test_ds_named = test_raw.map(_test_map, num_parallel_calls=AUTOTUNE).cache()
    test_ds_named = (
        test_ds_named.batch(batch_size, drop_remainder=False)
        .prefetch(AUTOTUNE)
        .with_options(_ds_opts)
    )

    pred_proba = model.predict(
        test_ds_named.map(lambda x, n: x, num_parallel_calls=AUTOTUNE), verbose=1
    )
    pred_idx = pred_proba.argmax(axis=1).astype(int)

    name_ds = test_ds_named.map(lambda x, n: n, num_parallel_calls=AUTOTUNE)
    name_batches = [nb.numpy() for nb in name_ds]
    pred_names = [n.decode("utf-8") for batch in name_batches for n in batch.tolist()]

    name_to_predidx = dict(zip(pred_names, pred_idx.tolist()))
    pred_idx_ordered = np.array([name_to_predidx[i] for i in test_image_ids], dtype=int)
else:
    test_paths = (TEST_PATH.rstrip("/") + "/" + pd.Series(test_image_ids)).values
    test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
    test_ds = test_ds.map(_decode_resize_and_rescale, num_parallel_calls=AUTOTUNE)

    test_ds = test_ds.cache()
    test_ds = (
        test_ds.batch(batch_size, drop_remainder=False)
        .prefetch(AUTOTUNE)
        .with_options(_ds_opts)
    )

    pred_proba = model.predict(test_ds, verbose=1)
    pred_idx_ordered = pred_proba.argmax(axis=1).astype(int)

if inv_class_indices:
    inv_arr = np.empty((len(inv_class_indices),), dtype=int)
    for k, v in inv_class_indices.items():
        inv_arr[int(k)] = int(v)
    pred_labels = inv_arr[pred_idx_ordered]
else:
    pred_labels = pred_idx_ordered

pred_labels[:10], (len(test_image_ids), NUM_CLASSES)




## === cell 10
submission = pd.DataFrame({"image_id": test_image_ids, "label": pred_labels})
sub_path = os.path.join(OUTPUT_DIR, "submission.csv")
submission.to_csv(sub_path, index=False)

submission.head(), sub_path
