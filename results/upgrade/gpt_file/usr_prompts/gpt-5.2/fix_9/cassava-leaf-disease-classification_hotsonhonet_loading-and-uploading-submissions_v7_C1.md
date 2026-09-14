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

0.8245693563009973

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.18348) has done: 'The main bottleneck is that you build `val_ds` by re-reading and preprocessing the *entire* training TFRecord set (and even caching it), rather than only the validation fraction; that roughly doubles input pipeline work and can blow memory/time. The fix is to construct train/val datasets from the TFRecords using a deterministic per-example split (hash of `image_name`) so each epoch reads each example only once, while preserving the same model/training loop and semantics. I also remove the unused CSV-based dataset code and switch test inference to a single `model.predict` call (same computation, less Python overhead). Finally, I add `ignore_order` and tune `interleave`/`prefetch` options in a deterministic-safe way to reduce input overhead without changing results.'
- What this solution (achieved 0.18348) has done: 'Main bottlenecks are (1) training/validation splitting done via hash-bucket filtering inside the TFRecord pipeline (extra parse/decode work, plus filter-induced pipeline inefficiency), and (2) an unnecessary Keras wrapper model for prediction that forces string tensors through Keras. I keep the exact same split rule and preprocessing/augmentation/model/training loop, but move the split to the file-list level so each dataset reads only its own TFRecords (no per-example filtering). I also simplify test prediction to use `model.predict()` directly while separately collecting names in the same order from the dataset, preserving identical evaluation semantics and outputs.'

# 9. Code solution

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




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
    FIX: Previous hash-based shard split could produce an empty val_files list.
    This keeps the same core TFRecord training approach, but ensures a stable,
    non-empty shard-level split based on the shard index in filenames.
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


_record_count_cache = {}


def _count_tfrecord_records(fn: str) -> int:
    if fn in _record_count_cache:
        return _record_count_cache[fn]
    c = 0
    for _ in tf.data.TFRecordDataset(fn):
        c += 1
    _record_count_cache[fn] = c
    return c


def _count_files_records(files):
    return int(sum(_count_tfrecord_records(f) for f in files))


def make_ds_from_tfrecords_files(tfrecord_files, training, cache=False):
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.autotune = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True

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

    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
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

train_count = _count_files_records(train_files)
val_count = _count_files_records(val_files)
steps_per_epoch = (train_count + BATCH_SIZE - 1) // BATCH_SIZE
validation_steps = (val_count + BATCH_SIZE - 1) // BATCH_SIZE
print("Train records:", train_count, "Val records:", val_count)
print("steps_per_epoch:", steps_per_epoch, "validation_steps:", validation_steps)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/978219493.py in <cell line: 0>()
    145 print(f"TFRecord shards -> train: {len(train_files)}, val: {len(val_files)}")
    146 
--> 147 train_ds = make_ds_from_tfrecords_files(train_files, training=True, cache=False)
    148 val_ds = make_ds_from_tfrecords_files(val_files, training=False, cache=True)
    149 

/tmp/ipykernel_11/978219493.py in make_ds_from_tfrecords_files(tfrecord_files, training, cache)
     90     options = tf.data.Options()
     91     options.experimental_deterministic = True
---> 92     options.experimental_optimization.autotune = True
     93     options.experimental_optimization.map_parallelization = True
     94     options.experimental_optimization.parallel_batch = True

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/options.py in __setattr__(self, name, value)
     59       object.__setattr__(self, name, value)
     60     else:
---> 61       raise AttributeError("Cannot set the property {} on {}.".format(
     62           name,
     63           type(self).__name__))

AttributeError: Cannot set the property autotune on OptimizationOptions.

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
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
)

model = tf.keras.models.load_model(ckpt_path)
print("Model training complete and best checkpoint loaded.")




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/581271209.py in <cell line: 0>()
     13 EPOCHS = 3
     14 history = model.fit(
---> 15     train_ds,
     16     validation_data=val_ds,
     17     epochs=EPOCHS,

NameError: name 'train_ds' is not defined

## === cell 5
ss = pd.read_csv(SAMPLE_CSV)

test_tfrecord_files = tf.io.gfile.glob(TEST_TFREC_GLOB)
if not test_tfrecord_files:
    raise FileNotFoundError(f"No TFRecords found at glob: {TEST_TFREC_GLOB}")

options = tf.data.Options()
options.experimental_deterministic = True
options.experimental_optimization.autotune = True
options.experimental_optimization.map_parallelization = True
options.experimental_optimization.parallel_batch = True

files_ds = tf.data.Dataset.from_tensor_slices(sorted(test_tfrecord_files))
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

all_names = []
all_preds = []
for batch_imgs, batch_names in test_ds:
    batch_probs = model(batch_imgs, training=False).numpy()
    all_preds.append(np.argmax(batch_probs, axis=1).astype(np.int64))
    all_names.append(batch_names.numpy())

names = np.concatenate(all_names, axis=0).astype("U")
preds = np.concatenate(all_preds, axis=0).astype(int)

pred_map = dict(zip(names, preds))
ordered_preds = ss["image_id"].map(pred_map).values.astype(int)

submission = pd.DataFrame({"image_id": ss["image_id"].values, "label": ordered_preds})
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2918461862.py in <cell line: 0>()
      8 options = tf.data.Options()
      9 options.experimental_deterministic = True
---> 10 options.experimental_optimization.autotune = True
     11 options.experimental_optimization.map_parallelization = True
     12 options.experimental_optimization.parallel_batch = True

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/options.py in __setattr__(self, name, value)
     59       object.__setattr__(self, name, value)
     60     else:
---> 61       raise AttributeError("Cannot set the property {} on {}.".format(
     62           name,
     63           type(self).__name__))

AttributeError: Cannot set the property autotune on OptimizationOptions.
