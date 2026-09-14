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

import gc, random, math, glob
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

print("TensorFlow version:", tf.__version__)

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
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
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
except Exception:
    pass




## === cell 1
class FixedDropout(tf.keras.layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        symbolic_shape = K.shape(inputs)
        noise_shape = [
            symbolic_shape[axis] if shape is None else shape
            for axis, shape in enumerate(self.noise_shape)
        ]
        return tuple(noise_shape)




## === cell 2
BASE1 = "/kaggle/input/cassava-leaf-disease-classification"
BASE2 = "../input/cassava-leaf-disease-classification"

BASE = BASE1 if os.path.exists(BASE1) else BASE2
TRAIN_CSV = os.path.join(BASE, "train.csv")
SAMPLE_SUB = os.path.join(BASE, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(BASE, "train_images")
TEST_IMG_DIR = os.path.join(BASE, "test_images")
TRAIN_TFREC_DIR = os.path.join(BASE, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(BASE, "test_tfrecords")

print("Using BASE:", BASE)
print("TRAIN_CSV exists:", os.path.exists(TRAIN_CSV))
print("SAMPLE_SUB exists:", os.path.exists(SAMPLE_SUB))
print("TRAIN_IMG_DIR exists:", os.path.exists(TRAIN_IMG_DIR))
print("TEST_IMG_DIR exists:", os.path.exists(TEST_IMG_DIR))
print("TRAIN_TFREC_DIR exists:", os.path.exists(TRAIN_TFREC_DIR))
print("TEST_TFREC_DIR exists:", os.path.exists(TEST_TFREC_DIR))




## === cell 3
PRETRAIN_CANDIDATES = [
    "/kaggle/input/train-model-cassava",
    "../input/train-model-cassava",
    "/kaggle/data/input/train-model-cassava",
    "/kaggle/data/train-model-cassava",
    "/kaggle/input",  # sometimes datasets are nested one level deeper
    "../input",
]


def find_file_in_candidates(filename, candidates):
    for base in candidates:
        p = os.path.join(base, filename)
        if os.path.exists(p):
            return p
        nested = glob.glob(os.path.join(base, "*", filename))
        if nested:
            return nested[0]
    return None


dense_path = find_file_in_candidates("densenet201.h5", PRETRAIN_CANDIDATES)
inception_path = find_file_in_candidates("inceptionv3.h5", PRETRAIN_CANDIDATES)
eff_path = find_file_in_candidates("efficient_netb3.h5", PRETRAIN_CANDIDATES)


def try_load_model(path, custom_objects=None, compile_flag=True):
    if path is not None and os.path.exists(path):
        print("Loading model:", path)
        return tf.keras.models.load_model(
            path, custom_objects=custom_objects, compile=compile_flag
        )
    print("Model not found (will fallback):", path)
    return None


dense201 = try_load_model(dense_path, compile_flag=True)
inception = try_load_model(inception_path, compile_flag=True)
efficient_net = try_load_model(
    eff_path, custom_objects={"FixedDropout": FixedDropout}, compile_flag=False
)




## === cell 4
submission = pd.read_csv(SAMPLE_SUB)
test_image_ids = np.sort(submission.image_id.values.astype(str))
print("Test rows:", len(submission))




## === cell 5
train_df = pd.read_csv(TRAIN_CSV)
num_classes = train_df["label"].nunique()
assert num_classes == 5, f"Expected 5 classes, got {num_classes}"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 3  # keep identical to original script

AUTOTUNE = tf.data.AUTOTUNE

data_opts = tf.data.Options()
data_opts.experimental_deterministic = (
    True  # stable iteration order; negligible performance impact vs correctness
)

try:
    data_opts.experimental_optimization.apply_default_optimizations = True
    data_opts.experimental_optimization.autotune_buffers = True
    data_opts.experimental_optimization.autotune_cpu_budget = True
    data_opts.experimental_slack = True
except Exception:
    pass

USE_TFRECORDS = os.path.isdir(TRAIN_TFREC_DIR) and os.path.isdir(TEST_TFREC_DIR)


@tf.function
def _decode_jpeg_resize_norm(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape([IMG_SIZE, IMG_SIZE, 3])
    return img


@tf.function
def decode_and_resize(path, label=None):
    img = tf.io.read_file(path)
    img = _decode_jpeg_resize_norm(img)
    if label is None:
        return img
    return img, tf.cast(label, tf.int32)


_FEATURE_DESC_TRAIN = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}
_FEATURE_DESC_TEST = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


@tf.function
def parse_train_tfrecord(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURE_DESC_TRAIN)
    img = _decode_jpeg_resize_norm(ex["image"])
    lbl = tf.cast(ex["target"], tf.int32)
    return img, lbl


@tf.function
def parse_test_tfrecord(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURE_DESC_TEST)
    img = _decode_jpeg_resize_norm(ex["image"])
    return img


idx = np.arange(len(train_df))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
split = int(0.9 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]

train_image_ids = train_df["image_id"].astype(str).values

if not USE_TFRECORDS:
    tr_paths_np = np.char.add(
        np.char.add(TRAIN_IMG_DIR + os.sep, train_image_ids[tr_idx].astype(str)), ""
    )
    va_paths_np = np.char.add(
        np.char.add(TRAIN_IMG_DIR + os.sep, train_image_ids[va_idx].astype(str)), ""
    )
    test_paths_np = np.char.add(
        np.char.add(TEST_IMG_DIR + os.sep, test_image_ids.astype(str)), ""
    )

    tr_paths = tf.constant(tr_paths_np)
    tr_labels = tf.constant(train_df.iloc[tr_idx]["label"].values, dtype=tf.int32)
    va_paths = tf.constant(va_paths_np)
    va_labels = tf.constant(train_df.iloc[va_idx]["label"].values, dtype=tf.int32)
    test_paths = tf.constant(test_paths_np)

    ds_train = (
        tf.data.Dataset.from_tensor_slices((tr_paths, tr_labels))
        .shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
        .map(decode_and_resize, num_parallel_calls=AUTOTUNE, deterministic=True)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTOTUNE)
    ).with_options(data_opts)

    ds_val = (
        tf.data.Dataset.from_tensor_slices((va_paths, va_labels))
        .map(decode_and_resize, num_parallel_calls=AUTOTUNE, deterministic=True)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTOTUNE)
    ).with_options(data_opts)

    ds_test = (
        tf.data.Dataset.from_tensor_slices(test_paths)
        .map(decode_and_resize, num_parallel_calls=AUTOTUNE, deterministic=True)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTOTUNE)
    ).with_options(data_opts)
else:
    train_files = sorted(tf.io.gfile.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec")))
    test_files = sorted(tf.io.gfile.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec")))
    print(
        "Using TFRecords. Train shards:",
        len(train_files),
        "Test shards:",
        len(test_files),
    )

    n_train_shards = max(1, int(0.9 * len(train_files)))
    tr_files = train_files[:n_train_shards]
    va_files = (
        train_files[n_train_shards:]
        if n_train_shards < len(train_files)
        else train_files[-1:]
    )

    def _make_ds_from_files(files, training):
        ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTOTUNE)
        if training:
            ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(
            parse_train_tfrecord, num_parallel_calls=AUTOTUNE, deterministic=True
        )
        ds = ds.batch(BATCH_SIZE, drop_remainder=False)
        ds = ds.prefetch(AUTOTUNE)
        return ds.with_options(data_opts)

    ds_train = _make_ds_from_files(tr_files, training=True)
    ds_val = _make_ds_from_files(va_files, training=False)

    ds_test = (
        tf.data.TFRecordDataset(test_files, num_parallel_reads=AUTOTUNE)
        .map(parse_test_tfrecord, num_parallel_calls=AUTOTUNE, deterministic=True)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTOTUNE)
    ).with_options(data_opts)

TEST_STEPS = int(math.ceil(len(submission) / BATCH_SIZE))


def build_small_cnn(seed_offset=0):
    tf.keras.utils.set_random_seed(SEED + seed_offset)
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


fallback_models = []
if dense201 is None:
    dense201 = build_small_cnn(seed_offset=1)
    fallback_models.append(("dense201", dense201))
if inception is None:
    inception = build_small_cnn(seed_offset=2)
    fallback_models.append(("inception", inception))
if efficient_net is None:
    efficient_net = build_small_cnn(seed_offset=3)
    fallback_models.append(("efficient_net", efficient_net))

for name, m in fallback_models:
    print(f"Training fallback model: {name}")
    m.fit(ds_train, validation_data=ds_val, epochs=EPOCHS, verbose=2)

gc.collect()




## === cell 6
@tf.function
def _batch_vote(m1, m2, m3, batch):
    out1 = m1(batch, training=False)
    out2 = m2(batch, training=False)
    out3 = m3(batch, training=False)
    v1 = tf.argmax(out1, axis=-1, output_type=tf.int32)
    v2 = tf.argmax(out2, axis=-1, output_type=tf.int32)
    v3 = tf.argmax(out3, axis=-1, output_type=tf.int32)
    voted = tf.where(
        tf.equal(v1, v2),
        v1,
        tf.where(tf.equal(v2, v3), v2, tf.where(tf.equal(v1, v3), v3, v1)),
    )
    return voted  # int32


def predict_voted_single_pass(m1, m2, m3, ds, total_n):
    parts = []
    seen = 0
    for batch in ds:
        voted = _batch_vote(m1, m2, m3, batch)
        parts.append(voted)
        seen += int(voted.shape[0])
        if seen >= total_n:
            break
    result = tf.concat(parts, axis=0)[:total_n]
    return result.numpy().astype(np.int64, copy=False)


n = len(submission)
result = predict_voted_single_pass(dense201, inception, efficient_net, ds_test, n)

submission["label"] = result
submission = submission[["image_id", "label"]]
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
