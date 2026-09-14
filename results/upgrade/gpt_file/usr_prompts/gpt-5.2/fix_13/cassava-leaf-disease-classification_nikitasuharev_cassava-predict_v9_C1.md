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

import gc, random, glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

print("Tensorflow version " + tf.__version__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF choose
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass




## === cell 1
DATA_DIR = "../input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
TRAIN_TFREC_DIR = os.path.join(DATA_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(DATA_DIR, "test_tfrecords")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"
assert os.path.exists(TRAIN_IMG_DIR), f"Missing: {TRAIN_IMG_DIR}"
assert os.path.exists(TEST_IMG_DIR), f"Missing: {TEST_IMG_DIR}"

train_df = pd.read_csv(TRAIN_CSV)
submission = pd.read_csv(SAMPLE_SUB)

num_classes = train_df["label"].nunique()
print("Train rows:", len(train_df), " Num classes:", num_classes)
print("Test rows:", len(submission))

TRAIN_TFRECS = sorted(glob.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec")))
TEST_TFRECS = sorted(glob.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec")))
print("Found train TFRecords:", len(TRAIN_TFRECS), " test TFRecords:", len(TEST_TFRECS))




## === cell 2
IMG_SIZE = 512  # keep consistent with the original code intent (512x512)
BATCH_SIZE = 16  # keep moderate to fit typical Kaggle GPU/CPU memory
AUTOTUNE = tf.data.AUTOTUNE


@tf.function(reduce_retracing=True)
def _decode_jpeg_bytes(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    return img


@tf.function(reduce_retracing=True)
def decode_and_resize(path):
    img_bytes = tf.io.read_file(path)
    return _decode_jpeg_bytes(img_bytes)


def make_train_val_split(df, val_frac=0.1):
    labels = df["label"].values
    idx = np.arange(len(df))
    rng = np.random.RandomState(SEED)

    train_idx = []
    val_idx = []
    for c in np.unique(labels):
        c_idx = idx[labels == c]
        rng.shuffle(c_idx)
        n_val = max(1, int(len(c_idx) * val_frac))
        val_idx.extend(c_idx[:n_val].tolist())
        train_idx.extend(c_idx[n_val:].tolist())

    train_df2 = (
        df.iloc[train_idx].sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    )
    val_df2 = (
        df.iloc[val_idx].sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    )
    return train_df2, val_df2


tr_df, va_df = make_train_val_split(train_df, val_frac=0.1)
print("Train split:", len(tr_df), "Val split:", len(va_df))

tr_paths = (TRAIN_IMG_DIR + "/" + tr_df["image_id"].astype(str)).values
va_paths = (TRAIN_IMG_DIR + "/" + va_df["image_id"].astype(str)).values

data_options = tf.data.Options()
data_options.experimental_deterministic = (
    True  # preserve determinism/semantics for training/val
)

try:
    data_options.experimental_optimization.apply_default_optimizations = True
except Exception:
    pass
try:
    data_options.experimental_optimization.autotune_buffers = True
except Exception:
    pass
try:
    data_options.experimental_optimization.parallel_batch = True
except Exception:
    pass
try:
    data_options.experimental_optimization.map_and_batch_fusion = True
except Exception:
    pass
try:
    data_options.experimental_optimization.map_parallelization = True
except Exception:
    pass

_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


@tf.function(reduce_retracing=True)
def _parse_tfrec(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURES)
    img = _decode_jpeg_bytes(ex["image"])
    name = ex["image_name"]
    y = tf.cast(ex["target"], tf.int32)
    return img, y, name


@tf.function(reduce_retracing=True)
def _effnet_preprocess(x01):
    return keras.applications.efficientnet.preprocess_input(x01 * 255.0)


def build_dataset(paths_np, labels_np, training=True):
    paths = tf.convert_to_tensor(paths_np, dtype=tf.string)
    labels = tf.convert_to_tensor(labels_np, dtype=tf.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.with_options(data_options)

    @tf.function(reduce_retracing=True)
    def _map_fn(path, y):
        x = decode_and_resize(path)
        if training:
            x = tf.image.random_flip_left_right(x, seed=SEED)
            x = tf.image.random_brightness(x, max_delta=0.08, seed=SEED)
            x = tf.image.random_contrast(x, lower=0.9, upper=1.1, seed=SEED)
        x = _effnet_preprocess(x)
        return x, y

    if training:
        ds = ds.shuffle(4096, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def build_dataset_from_tfrecords(tfrec_files, training=True):
    ds = tf.data.TFRecordDataset(
        tfrec_files,
        num_parallel_reads=AUTOTUNE,
        compression_type=None,
    )
    ds = ds.with_options(data_options)
    ds = ds.map(_parse_tfrec, num_parallel_calls=AUTOTUNE)

    @tf.function(reduce_retracing=True)
    def _augment_drop_name(img, y, name):
        if training:
            img = tf.image.random_flip_left_right(img, seed=SEED)
            img = tf.image.random_brightness(img, max_delta=0.08, seed=SEED)
            img = tf.image.random_contrast(img, lower=0.9, upper=1.1, seed=SEED)
        img = _effnet_preprocess(img)
        return img, y

    if training:
        ds = ds.shuffle(4096, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(_augment_drop_name, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def _shard_tfrec_files(tfrec_files, val_frac=0.1):
    tfrec_files = list(tfrec_files)
    if not tfrec_files:
        return [], []
    rng = np.random.RandomState(SEED)
    order = np.arange(len(tfrec_files))
    rng.shuffle(order)
    tfrec_files = [tfrec_files[i] for i in order]
    n_val = max(1, int(round(len(tfrec_files) * val_frac)))
    val_files = tfrec_files[:n_val]
    train_files = tfrec_files[n_val:]
    if not train_files:  # safety: ensure at least 1 train file
        train_files, val_files = val_files, train_files
    return train_files, val_files


if len(TRAIN_TFRECS) > 0:
    tr_files, va_files = _shard_tfrec_files(TRAIN_TFRECS, val_frac=0.1)
    print(
        "Using TFRecord file split - train files:",
        len(tr_files),
        "val files:",
        len(va_files),
    )
    train_ds = build_dataset_from_tfrecords(tr_files, training=True)
    val_ds = build_dataset_from_tfrecords(va_files, training=False)

    PER_TFREC = 1338
    n_train_for_steps = len(tr_files) * PER_TFREC
    n_val_for_steps = len(va_files) * PER_TFREC
else:
    train_ds = build_dataset(tr_paths, tr_df["label"].values, training=True)
    val_ds = build_dataset(va_paths, va_df["label"].values, training=False)
    n_train_for_steps = len(tr_df)
    n_val_for_steps = len(va_df)


def _steps(n):
    return int((n + BATCH_SIZE - 1) // BATCH_SIZE)


steps_per_epoch = _steps(n_train_for_steps)
validation_steps = _steps(n_val_for_steps)
print("steps_per_epoch:", steps_per_epoch, "validation_steps:", validation_steps)




## === cell 3
base = keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling="avg",
)
base.trainable = False  # stable + fast; avoids long fine-tuning

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = inputs
x = base(x, training=False)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=16,
)

model.summary()




## === cell 4
EPOCHS = 4
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)

gc.collect()




## === cell 5
if len(TEST_TFRECS) > 0:
    _TEST_FEATURES = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
    }

    @tf.function(reduce_retracing=True)
    def _parse_test_tfrec(example_proto):
        ex = tf.io.parse_single_example(example_proto, _TEST_FEATURES)
        img = _decode_jpeg_bytes(ex["image"])
        img = _effnet_preprocess(img)
        name = ex["image_name"]
        return name, img

    test_options = tf.data.Options()
    test_options.experimental_deterministic = False
    try:
        test_options.experimental_optimization.apply_default_optimizations = True
    except Exception:
        pass

    test_ds = tf.data.TFRecordDataset(TEST_TFRECS, num_parallel_reads=AUTOTUNE)
    test_ds = test_ds.with_options(test_options)
    test_ds = test_ds.map(_parse_test_tfrec, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

    names_ds = test_ds.map(lambda n, x: n, num_parallel_calls=AUTOTUNE)
    imgs_ds = test_ds.map(lambda n, x: x, num_parallel_calls=AUTOTUNE)

    all_names = np.concatenate(list(names_ds.as_numpy_iterator())).astype("U")
    probs = model.predict(imgs_ds, verbose=1)
    all_preds = np.argmax(probs, axis=1).astype(int)

    assert len(all_names) == len(all_preds)
    name_to_pred = dict(zip(all_names.tolist(), all_preds.tolist()))
    submission["label"] = submission["image_id"].map(name_to_pred).astype(int).values
else:
    test_paths = (TEST_IMG_DIR + "/" + submission["image_id"].astype(str)).values
    test_ds = tf.data.Dataset.from_tensor_slices(
        tf.convert_to_tensor(test_paths, dtype=tf.string)
    )
    test_ds = test_ds.with_options(data_options)

    @tf.function(reduce_retracing=True)
    def _load_test(path):
        x = decode_and_resize(path)
        x = _effnet_preprocess(x)
        return x

    test_ds = test_ds.map(_load_test, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    probs = model.predict(test_ds, verbose=1)
    preds = np.argmax(probs, axis=1).astype(int)
    assert len(preds) == len(
        submission
    ), "Prediction length mismatch with submission rows."
    submission["label"] = preds

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
