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
TRAIN_IMG_LOC = "/kaggle/input/cassava-leaf-disease-classification/train_images"
TRAIN_CSV = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
SAMPLE_CSV = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
TEST_IMG_LOC = "/kaggle/input/cassava-leaf-disease-classification/test_images"

MODELS_WEIGHTS = "/kaggle/input/cassavaeffentb7models/content/Models"

import os
import random
import numpy as np
import pandas as pd

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

IMG_SIZE = 224
BATCH_SIZE = 64
NUM_CLASSES = 5

print("TRAIN_IMG_LOC exists:", os.path.exists(TRAIN_IMG_LOC))
print("TEST_IMG_LOC exists:", os.path.exists(TEST_IMG_LOC))
print("MODELS_WEIGHTS exists:", os.path.exists(MODELS_WEIGHTS))




## === cell 1
model = None

candidate_paths = [
    os.path.join(MODELS_WEIGHTS, "effnetB7_model_sparse_44acc.h5"),
    os.path.join(MODELS_WEIGHTS, "effnetB7_model_sparse_44acc.keras"),
    os.path.join(
        MODELS_WEIGHTS, "effnetB7_model_sparse_44acc"
    ),  # may be SavedModel dir
    os.path.join(MODELS_WEIGHTS, "model.h5"),
    os.path.join(MODELS_WEIGHTS, "model.keras"),
    os.path.join(MODELS_WEIGHTS, "model"),  # may be SavedModel dir
]
found_path = None

for p in candidate_paths:
    if os.path.exists(p):
        if os.path.isfile(p) or tf.io.gfile.isdir(p):
            found_path = p
            break

if found_path is None and os.path.exists(MODELS_WEIGHTS):
    try:
        for name in os.listdir(MODELS_WEIGHTS):
            p = os.path.join(MODELS_WEIGHTS, name)
            if os.path.isfile(p) and (p.endswith(".h5") or p.endswith(".keras")):
                found_path = p
                break
            if tf.io.gfile.isdir(p):
                if tf.io.gfile.exists(os.path.join(p, "saved_model.pb")):
                    found_path = p
                    break
    except Exception:
        found_path = None

if found_path is not None:
    model = keras.models.load_model(found_path, compile=False)
    print("Loaded pretrained model (compile=False):", found_path)
    _ = model(tf.zeros([1, IMG_SIZE, IMG_SIZE, 3], dtype=tf.float32), training=False)
else:
    print(
        f"Pretrained model not found under: {MODELS_WEIGHTS}\n"
        "Falling back to training a small EfficientNet model from scratch/imagenet init."
    )

    train_df = pd.read_csv(TRAIN_CSV)
    train_df["filepath"] = TRAIN_IMG_LOC + "/" + train_df["image_id"].astype(str)
    train_paths = train_df["filepath"].to_numpy(dtype=str)
    train_labels = train_df["label"].to_numpy(dtype=np.int32)

    n = len(train_df)
    idx = np.arange(n)
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)
    split = int(0.9 * n)
    tr_idx, va_idx = idx[:split], idx[split:]

    tr_paths, tr_labels = train_paths[tr_idx], train_labels[tr_idx]
    va_paths, va_labels = train_paths[va_idx], train_labels[va_idx]

    @tf.function
    def _load_train(path, label):
        img = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img, channels=3, dct_method="INTEGER_FAST")
        img = tf.image.resize(
            img,
            [IMG_SIZE, IMG_SIZE],
            method=tf.image.ResizeMethod.AREA,
            antialias=False,
        )
        img = tf.cast(img, tf.float32)
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        img = tf.ensure_shape(img, [IMG_SIZE, IMG_SIZE, 3])
        label = tf.cast(label, tf.int32)
        return img, label

    options = tf.data.Options()
    try:
        options.deterministic = True
    except Exception:
        options.experimental_deterministic = True
    try:
        options.experimental_optimization.apply_default_optimizations = True
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.map_and_batch_fusion = True
        options.experimental_optimization.parallel_batch = True
    except Exception:
        pass

    train_ds = tf.data.Dataset.from_tensor_slices((tr_paths, tr_labels)).with_options(
        options
    )
    train_ds = (
        train_ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
        .map(_load_train, num_parallel_calls=AUTOTUNE)
        .apply(tf.data.experimental.ignore_errors())
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTOTUNE)
    )

    val_ds = tf.data.Dataset.from_tensor_slices((va_paths, va_labels)).with_options(
        options
    )
    val_ds = (
        val_ds.map(_load_train, num_parallel_calls=AUTOTUNE)
        .apply(tf.data.experimental.ignore_errors())
        .batch(BATCH_SIZE, drop_remainder=False)
        .cache()  # exact: caches decoded tensors; no randomness in val pipeline
        .prefetch(AUTOTUNE)
    )

    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling="avg",
    )
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    base.trainable = False
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )

    history = model.fit(train_ds, validation_data=val_ds, epochs=2, verbose=1)

    base.trainable = True
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    history = model.fit(train_ds, validation_data=val_ds, epochs=1, verbose=1)

assert model is not None, "Model was not created/loaded."




## === cell 2
sub = pd.read_csv(SAMPLE_CSV)
assert list(sub.columns) == ["image_id", "label"], "Unexpected submission columns"

sub["filepath"] = TEST_IMG_LOC + "/" + sub["image_id"].astype(str)

test_paths = sub["filepath"].to_numpy(dtype=str)
test_ds = tf.data.Dataset.from_tensor_slices(test_paths)


@tf.function
def _load_test(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.AREA, antialias=False
    )
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    img = tf.ensure_shape(img, [IMG_SIZE, IMG_SIZE, 3])
    return img


options = tf.data.Options()
try:
    options.deterministic = True
except Exception:
    options.experimental_deterministic = True
try:
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.map_and_batch_fusion = True
    options.experimental_optimization.parallel_batch = True
except Exception:
    pass
test_ds = test_ds.with_options(options)

test_ds = (
    test_ds.map(_load_test, num_parallel_calls=AUTOTUNE)
    .apply(tf.data.experimental.ignore_errors())
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)




## === cell 3
n_test = len(sub)

probs = model.predict(test_ds, verbose=1)
pred_labels = np.argmax(probs, axis=1).astype(int)

pred_labels = pred_labels[:n_test]

sub_out = sub[["image_id"]].copy()
sub_out["label"] = pred_labels

assert len(sub_out) == len(sub), "Row count mismatch in submission"
assert sub_out["label"].between(0, 4).all(), "Predicted labels out of range 0-4"

sub_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_out.shape)
print(sub_out.head(10))




## === cell 4
assert os.path.exists("submission.csv"), "submission.csv was not created"
_check = pd.read_csv("submission.csv")
assert list(_check.columns) == ["image_id", "label"]
assert len(_check) == len(sub)
print("submission.csv looks valid.")
