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
import json
import warnings

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")

print("Input root exists:", os.path.exists("/kaggle/input"))



## === cell 1
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Conv2D,
    MaxPooling2D,
    GlobalAveragePooling2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

print("TF version:", tf.__version__)

seed = 42
tf.keras.backend.clear_session()
tf.random.set_seed(seed)
np.random.seed(seed)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass



## === cell 2
general_path = "../input/cassava-leaf-disease-classification/"
if not os.path.exists(general_path):
    general_path = "/kaggle/input/cassava-leaf-disease-classification/"
print("general_path:", general_path)
print("Exists:", os.path.exists(general_path))
print("Listing top-level:", os.listdir(general_path)[:10])



## === cell 3
with open(os.path.join(general_path, "label_num_to_disease_map.json")) as file:
    map_classes = json.loads(file.read())
    map_classes = {int(k): v for k, v in map_classes.items()}

print("Class map:", map_classes)



## === cell 4
train_csv_path = os.path.join(general_path, "train.csv")
df_train = pd.read_csv(train_csv_path)
df_train["class_name"] = df_train["label"].map(map_classes)
print(df_train.head())
print("Train rows:", len(df_train))



## === cell 5
img_width, img_height = 260, 260
batch_size = 64
num_classes = 5

train = df_train.copy()
train["label"] = train["label"].astype("int32")

train_images_dir = os.path.join(general_path, "train_images")
test_images_dir = os.path.join(general_path, "test_images")




## === cell 6
def _make_train_val_splits(df, validation_split=0.2, seed=42):
    df_shuf = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df_shuf)
    n_train = int(np.floor(n * (1.0 - validation_split)))
    df_train_split = df_shuf.iloc[:n_train].reset_index(drop=True)
    df_val_split = df_shuf.iloc[n_train:].reset_index(drop=True)
    return df_train_split, df_val_split


df_tr, df_val = _make_train_val_splits(
    train[["image_id", "label"]], validation_split=0.2, seed=seed
)
print("Train split:", len(df_tr), "Val split:", len(df_val))

tr_paths = (train_images_dir + os.sep + df_tr["image_id"].values).astype(str)
tr_labels = df_tr["label"].values.astype(np.int32)

val_paths = (train_images_dir + os.sep + df_val["image_id"].values).astype(str)
val_labels = df_val["label"].values.astype(np.int32)

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(mode="horizontal_and_vertical", seed=seed),
        tf.keras.layers.RandomZoom(height_factor=0.2, width_factor=0.2, seed=seed),
        tf.keras.layers.RandomRotation(factor=0.05, seed=seed),
    ],
    name="data_augmentation",
)


@tf.function
def _decode_resize_normalize(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img.set_shape([None, None, 3])
    img = tf.image.resize(
        img, [img_width, img_height], method=tf.image.ResizeMethod.BILINEAR
    )
    img.set_shape([img_width, img_height, 3])
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


@tf.function
def _path_to_img(path):
    img_bytes = tf.io.read_file(path)
    return _decode_resize_normalize(img_bytes)


@tf.function
def _batch_one_hot(x, y):
    return x, tf.one_hot(tf.cast(y, tf.int32), depth=num_classes)


@tf.function
def _augment_batch(x, y):
    x = data_augmentation(x, training=True)
    return x, y


@tf.function
def _path_to_x(path):
    return _path_to_img(path)


def _list_tfrec_files(subdir, prefix):
    tfrec_dir = os.path.join(general_path, subdir)
    if not os.path.exists(tfrec_dir):
        return []
    files = [
        os.path.join(tfrec_dir, f)
        for f in os.listdir(tfrec_dir)
        if f.startswith(prefix) and f.endswith(".tfrec")
    ]
    return sorted(files)


TRAIN_TFRECS = _list_tfrec_files("train_tfrecords", "ld_train")
TEST_TFRECS = _list_tfrec_files("test_tfrecords", "ld_test")
USE_TFRECORDS = (len(TRAIN_TFRECS) > 0) and (len(TEST_TFRECS) > 0)
print(
    "TFRecords available:",
    USE_TFRECORDS,
    "train:",
    len(TRAIN_TFRECS),
    "test:",
    len(TEST_TFRECS),
)


def _make_dataset(paths, labels=None, training=False, cache_path=None):
    options = tf.data.Options()
    options.deterministic = True
    options.experimental_optimization.apply_default_optimizations = True

    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(paths).with_options(options)
        ds = ds.map(_path_to_x, num_parallel_calls=AUTOTUNE, deterministic=True)
        if cache_path is not None:
            ds = ds.cache(cache_path)
        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    ds = tf.data.Dataset.from_tensor_slices((paths, labels)).with_options(options)
    if training:
        ds = ds.shuffle(
            buffer_size=min(len(paths), 8192), seed=seed, reshuffle_each_iteration=True
        )
    ds = ds.map(
        lambda p, y: (_path_to_img(p), y),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    if cache_path is not None:
        ds = ds.cache(cache_path)
    ds = ds.batch(batch_size, drop_remainder=training)
    ds = ds.map(_batch_one_hot, num_parallel_calls=AUTOTUNE, deterministic=True)
    if training:
        ds = ds.map(_augment_batch, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.prefetch(AUTOTUNE)
    return ds


_FEATURE_SPEC = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


@tf.function
def _parse_train_tfrec(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURE_SPEC)
    img = _decode_resize_normalize(ex["image"])
    y = tf.cast(ex["target"], tf.int32)
    return img, y


@tf.function
def _parse_test_tfrec(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURE_SPEC)
    img = _decode_resize_normalize(ex["image"])
    name = ex["image_name"]
    return img, name


def _make_tfrec_trainval_datasets(
    train_files, val_files, cache_train_path=None, cache_val_path=None
):
    options = tf.data.Options()
    options.deterministic = True
    options.experimental_optimization.apply_default_optimizations = True

    dstrain = tf.data.TFRecordDataset(
        train_files, num_parallel_reads=AUTOTUNE
    ).with_options(options)
    dstrain = dstrain.shuffle(8192, seed=seed, reshuffle_each_iteration=True)
    dstrain = dstrain.map(
        _parse_train_tfrec, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    if cache_train_path is not None:
        dstrain = dstrain.cache(cache_train_path)
    dstrain = dstrain.batch(batch_size, drop_remainder=True)
    dstrain = dstrain.map(
        _batch_one_hot, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    dstrain = dstrain.map(
        _augment_batch, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    dstrain = dstrain.prefetch(AUTOTUNE)

    dsval = tf.data.TFRecordDataset(
        val_files, num_parallel_reads=AUTOTUNE
    ).with_options(options)
    dsval = dsval.map(
        _parse_train_tfrec, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    if cache_val_path is not None:
        dsval = dsval.cache(cache_val_path)
    dsval = dsval.batch(batch_size, drop_remainder=False)
    dsval = dsval.map(_batch_one_hot, num_parallel_calls=AUTOTUNE, deterministic=True)
    dsval = dsval.prefetch(AUTOTUNE)

    return dstrain, dsval


def _make_tfrec_test_dataset(test_files, cache_test_path=None):
    options = tf.data.Options()
    options.deterministic = True
    options.experimental_optimization.apply_default_optimizations = True

    ds = tf.data.TFRecordDataset(test_files, num_parallel_reads=AUTOTUNE).with_options(
        options
    )
    ds = ds.map(_parse_test_tfrec, num_parallel_calls=AUTOTUNE, deterministic=True)
    if cache_test_path is not None:
        ds = ds.cache(cache_test_path)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_cache = None
val_cache = None

if USE_TFRECORDS:
    nfiles = len(TRAIN_TFRECS)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(nfiles)
    n_train_files = max(1, int(np.floor(nfiles * 0.8)))
    tr_idx = perm[:n_train_files]
    va_idx = perm[n_train_files:]
    if len(va_idx) == 0:
        va_idx = perm[-1:]
    train_tfrec_files = [TRAIN_TFRECS[i] for i in tr_idx]
    val_tfrec_files = [TRAIN_TFRECS[i] for i in va_idx]

    train_ds, valid_ds = _make_tfrec_trainval_datasets(
        train_tfrec_files,
        val_tfrec_files,
        cache_train_path=train_cache,
        cache_val_path=val_cache,
    )
else:
    train_ds = _make_dataset(tr_paths, tr_labels, training=True, cache_path=train_cache)
    valid_ds = _make_dataset(
        val_paths, val_labels, training=False, cache_path=val_cache
    )

class_indices = {str(i): i for i in range(num_classes)}
print("Class indices:", class_indices)

steps_per_epoch = int(np.ceil(len(tr_paths) / batch_size))
validation_steps = int(np.ceil(len(val_paths) / batch_size))
print("steps_per_epoch:", steps_per_epoch, "validation_steps:", validation_steps)



## === cell 7
for xb, yb in train_ds.take(1):
    print("Batch X:", xb.shape, "Batch y:", yb.shape)



## === cell 8
tf.keras.backend.clear_session()
tf.random.set_seed(seed)
np.random.seed(seed)

model = Sequential(
    [
        tf.keras.layers.Input(shape=(img_width, img_height, 3)),
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ]
)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    jit_compile=True,
)
model.summary()



## === cell 9
epochs = 5

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)



## === cell 10
sample_path = os.path.join(general_path, "sample_submission.csv")
ss = pd.read_csv(sample_path)

assert os.path.exists(test_images_dir), f"Missing test_images dir: {test_images_dir}"

test_cache = None

if USE_TFRECORDS:
    test_ds = _make_tfrec_test_dataset(TEST_TFRECS, cache_test_path=test_cache)

    img_ds = test_ds.map(
        lambda x, n: x, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    pred_proba = model.predict(img_ds, verbose=1)

    names = []
    for _, nb in test_ds:
        names.extend(nb.numpy().tolist())
    names = [n.decode("utf-8") for n in names]

    pred_by_name = dict(zip(names, np.argmax(pred_proba, axis=1).astype(int)))
    preds = ss["image_id"].map(pred_by_name).values.astype(int)
else:
    test_paths = (test_images_dir + os.sep + ss["image_id"].values).astype(str)
    test_ds = _make_dataset(
        test_paths,
        labels=None,
        training=False,
        cache_path=test_cache,
    )
    pred_proba = model.predict(test_ds, verbose=1)
    preds = np.argmax(pred_proba, axis=1).astype(int)

my_submission = pd.DataFrame({"image_id": ss["image_id"].values, "label": preds})
my_submission.to_csv("submission.csv", index=False)
print(my_submission.head())
print("Wrote submission.csv with rows:", len(my_submission))
