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

TRAIN_IMG_LOC = "../input/cassava-leaf-disease-classification/train_images"
TEST_IMG_LOC = "../input/cassava-leaf-disease-classification/test_images"
TRAIN_CSV = "../input/cassava-leaf-disease-classification/train.csv"
SAMPLE_CSV = "../input/cassava-leaf-disease-classification/sample_submission.csv"

import re
import random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint

print("TensorFlow:", tf.__version__)
print("ALL required modules loaded successfully")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

NUM_CLASSES = 5
IMG_SIZE = 300
BATCH_SIZE = 32

AUTOTUNE = tf.data.AUTOTUNE

TRAIN_TFREC_GLOB = (
    "../input/cassava-leaf-disease-classification/train_tfrecords/*.tfrec"
)
TEST_TFREC_GLOB = "../input/cassava-leaf-disease-classification/test_tfrecords/*.tfrec"

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass




## === cell 1
train_df = pd.read_csv(TRAIN_CSV)
train_df["filepath"] = TRAIN_IMG_LOC + "/" + train_df["image_id"].astype(str)

train_df = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

val_frac = 0.1
val_size = int(len(train_df) * val_frac)
val_df = train_df.iloc[:val_size].copy()
trn_df = train_df.iloc[val_size:].copy()

print("Train size:", len(trn_df), "Val size:", len(val_df))
print(trn_df.head())




## === cell 2
from tensorflow.keras.applications.efficientnet import preprocess_input

_FEATURE_DESC = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


def decode_and_preprocess(
    img_bytes_or_path, label=None, training=False, from_bytes=False
):
    if from_bytes:
        img_bytes = img_bytes_or_path
    else:
        img_bytes = tf.io.read_file(img_bytes_or_path)

    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear")
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)  # EfficientNet expects this preprocessing

    if training:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.1)

    if label is None:
        return img
    label = tf.cast(label, tf.int32)
    return img, label


def _split_tfrecord_files_by_shard_index(tfrecord_files, val_frac=0.1):
    """
    Stable, non-empty shard-level split based on shard index in filenames.
    """
    tfrecord_files = sorted(list(tfrecord_files))
    if not tfrecord_files:
        raise ValueError("Empty TFRecord file list provided.")

    shard_re = re.compile(r"ld_train(\d+)-\d+\.tfrec$")
    pairs = []
    for fn in tfrecord_files:
        base = os.path.basename(fn)
        m = shard_re.search(base)
        if m is None:
            idx = len(pairs)
        else:
            idx = int(m.group(1))
        pairs.append((idx, fn))

    pairs.sort(key=lambda x: x[0])
    ordered_files = [fn for _, fn in pairs]

    n = len(ordered_files)
    n_val = max(1, int(round(n * val_frac)))
    val_files = ordered_files[:n_val]
    train_files = ordered_files[n_val:]

    if not train_files:
        train_files, val_files = ordered_files[:-1], ordered_files[-1:]

    return train_files, val_files


def make_ds_from_tfrecords_files(tfrecord_files, training, cache=False):
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    try:
        options.experimental_slack = True
    except Exception:
        pass

    tfrecord_files = list(tfrecord_files)
    if not tfrecord_files:
        raise ValueError("Empty TFRecord file list provided.")

    files_ds = tf.data.Dataset.from_tensor_slices(tf.constant(tfrecord_files))

    if training:
        files_ds = files_ds.shuffle(
            buffer_size=len(tfrecord_files),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    ds = files_ds.interleave(
        lambda fn: tf.data.TFRecordDataset(fn, num_parallel_reads=AUTOTUNE),
        cycle_length=min(8, len(tfrecord_files)),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    ).with_options(options)

    def _parse_decode(example_proto):
        ex = tf.io.parse_single_example(example_proto, _FEATURE_DESC)
        img, y = decode_and_preprocess(
            ex["image"],
            tf.cast(ex["target"], tf.int32),
            training=training,
            from_bytes=True,
        )
        return img, y

    ds = ds.map(_parse_decode, num_parallel_calls=AUTOTUNE, deterministic=True)

    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)

    if cache:
        ds = ds.cache()

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_tfrecord_files = tf.io.gfile.glob(TRAIN_TFREC_GLOB)
if not train_tfrecord_files:
    raise FileNotFoundError(f"No TFRecords found at glob: {TRAIN_TFREC_GLOB}")

train_files, val_files = _split_tfrecord_files_by_shard_index(
    train_tfrecord_files, val_frac=0.1
)
print(f"TFRecord shards -> train: {len(train_files)}, val: {len(val_files)}")

train_ds = make_ds_from_tfrecords_files(train_files, training=True, cache=False)
val_ds = make_ds_from_tfrecords_files(val_files, training=False, cache=True)

print(
    "Datasets built (no TFRecord record-count pass; Keras will use dataset cardinality/exhaustion)."
)




## === cell 3
base = tf.keras.applications.EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base.trainable = False  # quick and stable; avoids long training

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()




## === cell 4
ckpt_path = "best_model.keras"
callbacks = [
    ModelCheckpoint(
        ckpt_path,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )
]

EPOCHS = 3
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=2,
)

model = tf.keras.models.load_model(ckpt_path)
print("Model training complete and best checkpoint loaded.")




## === cell 5
ss = pd.read_csv(SAMPLE_CSV)

test_tfrecord_files = tf.io.gfile.glob(TEST_TFREC_GLOB)
if not test_tfrecord_files:
    raise FileNotFoundError(f"No TFRecords found at glob: {TEST_TFREC_GLOB}")

options = tf.data.Options()
options.experimental_deterministic = True
options.experimental_optimization.apply_default_optimizations = True
try:
    options.experimental_slack = True
except Exception:
    pass

test_tfrecord_files = sorted(test_tfrecord_files)
files_ds = tf.data.Dataset.from_tensor_slices(test_tfrecord_files)
test_raw = files_ds.interleave(
    lambda fn: tf.data.TFRecordDataset(fn, num_parallel_reads=AUTOTUNE),
    cycle_length=min(8, len(test_tfrecord_files)),
    num_parallel_calls=AUTOTUNE,
    deterministic=True,
).with_options(options)


def _decode_test_with_name(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURE_DESC)
    img = decode_and_preprocess(
        ex["image"], label=None, training=False, from_bytes=True
    )
    return img, ex["image_name"]


test_ds = test_raw.map(
    _decode_test_with_name, num_parallel_calls=AUTOTUNE, deterministic=True
)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

pred_probs = model.predict(
    test_ds.map(lambda x, n: x, num_parallel_calls=AUTOTUNE), verbose=0
)
preds = np.argmax(pred_probs, axis=1).astype(int)

names_list = []
for _, nb in test_ds:
    names_list.append(nb.numpy())
all_names = np.concatenate(names_list, axis=0).astype("U")

pred_map = dict(zip(all_names, preds))
ordered_preds = ss["image_id"].map(pred_map).values

if np.any(pd.isna(ordered_preds)):
    mode_label = int(pd.Series(preds).mode().iloc[0])
    ordered_preds = pd.Series(ordered_preds).fillna(mode_label).astype(int).values
else:
    ordered_preds = ordered_preds.astype(int)

submission = pd.DataFrame({"image_id": ss["image_id"].values, "label": ordered_preds})
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
