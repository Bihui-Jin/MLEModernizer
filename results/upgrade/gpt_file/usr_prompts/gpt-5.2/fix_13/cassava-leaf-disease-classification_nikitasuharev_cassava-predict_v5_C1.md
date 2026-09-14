# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

# 5. Target score

0.8608340888485947

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, math, random, gc
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print("Tensorflow version " + tf.__version__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

tf.config.run_functions_eagerly(False)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_DIR = "../input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

train_df = pd.read_csv(TRAIN_CSV)
submission = pd.read_csv(SAMPLE_SUB)

num_classes = train_df["label"].nunique()
assert num_classes == 5, f"Expected 5 classes, got {num_classes}"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 3  # keep identical core training budget

train_df["filepath"] = TRAIN_IMG_DIR + "/" + train_df["image_id"].astype(str)
submission["filepath"] = TEST_IMG_DIR + "/" + submission["image_id"].astype(str)

idx = np.arange(len(train_df))
np.random.shuffle(idx)
split = int(0.9 * len(idx))
train_idx, val_idx = idx[:split], idx[split:]

train_split = train_df.iloc[train_idx].reset_index(drop=True)
val_split = train_df.iloc[val_idx].reset_index(drop=True)

print("Train/Val sizes:", len(train_split), len(val_split))

steps_per_epoch = int(math.ceil(len(train_split) / BATCH_SIZE))
val_steps = int(math.ceil(len(val_split) / BATCH_SIZE))
test_steps = int(math.ceil(len(submission) / BATCH_SIZE))



## === cell 2
AUTOTUNE = tf.data.AUTOTUNE


@tf.function
def _decode_resize_to_float01(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")  # uint8
    img = tf.image.resize_with_pad(
        img, IMG_SIZE, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.image.convert_image_dtype(img, tf.float32)
    img.set_shape((IMG_SIZE, IMG_SIZE, 3))
    return img


def _with_options(ds, deterministic=True):
    opts = tf.data.Options()
    opts.experimental_deterministic = deterministic
    opts.experimental_optimization.apply_default_optimizations = True
    opts.experimental_optimization.map_and_batch_fusion = True
    opts.experimental_optimization.parallel_batch = True
    opts.experimental_optimization.autotune_buffers = True
    try:
        opts.threading.private_threadpool_size = min(16, max(4, (os.cpu_count() or 8)))
        opts.threading.max_intra_op_parallelism = 1
    except Exception:
        pass
    return ds.with_options(opts)


def _finalize_pipeline(ds, deterministic=True):
    return _with_options(ds, deterministic=deterministic).prefetch(AUTOTUNE)


def make_train_ds(df, training=True, cache_decoded_to_disk=True):
    paths = df["filepath"].values
    labels = df["label"].values.astype(np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(
            buffer_size=min(len(df), 8192), seed=SEED, reshuffle_each_iteration=True
        )

    ds = ds.map(
        lambda p, y: (_decode_resize_to_float01(p), y),
        num_parallel_calls=AUTOTUNE,
        deterministic=not training,  # allow faster scheduling for training only
    )

    if training:
        if cache_decoded_to_disk:
            cache_path = os.path.join(
                tf.io.get_temp_dir(), "cassava_train_decoded_cache"
            )
            ds = ds.cache(cache_path)
        else:
            ds = ds.cache()
        ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE, drop_remainder=training)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = _finalize_pipeline(ds, deterministic=not training)
    return ds


def make_val_ds(df):
    paths = df["filepath"].values
    labels = df["label"].values.astype(np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    ds = ds.map(
        lambda p, y: (_decode_resize_to_float01(p), y),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )

    ds = ds.cache()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = _finalize_pipeline(ds, deterministic=True)
    return ds


def make_test_ds(df):
    paths = df["filepath"].values
    ds = tf.data.Dataset.from_tensor_slices(paths)

    ds = ds.map(
        lambda p: _decode_resize_to_float01(p),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )

    ds = ds.cache()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = _finalize_pipeline(ds, deterministic=True)
    return ds


train_ds = make_train_ds(train_split, training=True, cache_decoded_to_disk=True)
val_ds = make_val_ds(val_split)
test_ds = make_test_ds(submission)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/228707398.py in <cell line: 0>()
    110 
    111 
--> 112 train_ds = make_train_ds(train_split, training=True, cache_decoded_to_disk=True)
    113 val_ds = make_val_ds(val_split)
    114 test_ds = make_test_ds(submission)

/tmp/ipykernel_11/228707398.py in make_train_ds(df, training, cache_decoded_to_disk)
     59         if cache_decoded_to_disk:
     60             cache_path = os.path.join(
---> 61                 tf.io.get_temp_dir(), "cassava_train_decoded_cache"
     62             )
     63             ds = ds.cache(cache_path)

AttributeError: module 'tensorflow._api.v2.io' has no attribute 'get_temp_dir'

## === cell 3
inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = layers.RandomFlip(mode="horizontal_and_vertical", seed=SEED)(inputs)

x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    run_eagerly=False,
)

model.summary()



## === cell 4
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    verbose=2,
)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2702287040.py in <cell line: 0>()
      1 history = model.fit(
----> 2     train_ds,
      3     validation_data=val_ds,
      4     epochs=EPOCHS,
      5     steps_per_epoch=steps_per_epoch,

NameError: name 'train_ds' is not defined

## === cell 5
probs = model.predict(test_ds, steps=test_steps, verbose=1)
preds = np.argmax(probs, axis=-1).astype(int)

assert len(preds) == len(
    submission
), f"Pred length {len(preds)} != submission length {len(submission)}"

submission_out = submission[["image_id"]].copy()
submission_out["label"] = preds
submission_out.to_csv("submission.csv", index=False)

print(submission_out.head())
print("Wrote submission.csv with", len(submission_out), "rows")



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2772184211.py in <cell line: 0>()
----> 1 probs = model.predict(test_ds, steps=test_steps, verbose=1)
      2 preds = np.argmax(probs, axis=-1).astype(int)
      3 
      4 assert len(preds) == len(
      5     submission

NameError: name 'test_ds' is not defined

## === cell 6
preds

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/222146027.py in <cell line: 0>()
----> 1 preds

NameError: name 'preds' is not defined
