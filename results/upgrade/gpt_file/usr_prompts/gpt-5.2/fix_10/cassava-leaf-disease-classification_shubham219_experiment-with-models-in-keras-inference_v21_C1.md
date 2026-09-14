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

0.6569960713206406

# 6. Current score

0.12855

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11024) has done: 'Main bottlenecks are (1) PIL-based JPEG decoding/augmentation inside `ImageDataGenerator.flow_from_dataframe` (slow, single-process) and (2) inefficient input pipelining that can’t overlap CPU preprocessing with GPU/accelerator compute. I keep the exact same model, loss, optimizer, epochs, and augmentation semantics, but switch the data pipeline to `tf.data` using the provided TFRecords (same images/labels) with parallel decode, vectorized augmentations equivalent to your generator settings, caching, and prefetch. I also avoid expensive `glob`/string-splitting for test IDs by reading `sample_submission.csv` order directly and using TFRecords for test input, preserving submission semantics. These changes are performance-only: they remove Python/PIL overhead and enable parallel, pipelined input without altering the training loop logic or model.'
- What this solution (achieved 0.10426) has done: 'The timeout is dominated by slow JPEG file I/O + Python path handling in the training pipeline, plus extra work in test inference (building two separate dataset traversals and probing TFRecords with a dataset iteration). I keep the same model, epochs, augmentations, and loss, but switch the training input from `train_images/` JPEGs to `train_tfrecords/` (same content, much faster sequential reads) while preserving deterministic behavior and identical resize/normalize/augment steps. For test, I avoid the expensive “read names in a second pass” by predicting in one pass and collecting names in the same loop, and I remove the TFRecord “can_parse” probe in favor of a cheap existence check since the competition TFRecords are well-formed. These changes reduce overhead without changing the learning/inference semantics.'
- What this solution (achieved 0.12855) has done: 'The timeout is dominated by two avoidable overheads: forcing the pure-Python protobuf implementation (much slower TFRecord parsing) and repeatedly filtering the entire TFRecord dataset twice (for train/val) which causes two full passes and heavy tf.data filter costs. I switch protobuf back to TensorFlow’s default fast backend (keeping deterministic ops), and I build train/val TFRecord datasets by selecting the needed shards directly (so no expensive per-example filtering), while keeping the same split semantics (based on `train.csv`). I also add `.cache()` after TFRecord parsing (before shuffle/augment) so decoding/resize is done once per epoch rather than re-decoding each epoch, preserving exact augment logic and training loop. Prediction is also streamlined to use `model.predict` to avoid Python loops and repeated `.numpy()` transfers, with the same outputs.'

# 9. Code solution

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



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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


def _decode_and_resize_from_path(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
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


def _decode_and_resize(image_bytes):
    img = tf.io.decode_jpeg(image_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, IMG_SIZE, method="bilinear", antialias=False)
    img = tf.cast(img, tf.float32) / 255.0
    return img


def _parse_train_example(example_proto):
    feats = {
        "image": tf.io.FixedLenFeature([], tf.string, default_value=b""),
        "image_name": tf.io.FixedLenFeature([], tf.string, default_value=b""),
        "image_id": tf.io.FixedLenFeature([], tf.string, default_value=b""),
        "label": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
        "target": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
    }
    ex = tf.io.parse_single_example(example_proto, feats)

    img = _decode_and_resize(ex["image"])

    lbl = tf.where(ex["label"] >= 0, ex["label"], ex["target"])
    lbl = tf.cast(lbl, tf.int32)

    name = tf.where(
        tf.strings.length(ex["image_name"]) > 0, ex["image_name"], ex["image_id"]
    )
    return img, lbl, name


if use_train_tfrecords:
    def _shard_index_from_image_id_series(image_id_series, shard_size=1338):
        ids = (
            image_id_series.str.replace(".jpg", "", regex=False).astype(np.int64).values
        )
        return (ids // shard_size).astype(np.int32)

    trn_shards = np.unique(_shard_index_from_image_id_series(df_trn["image_id"]))
    val_shards = np.unique(_shard_index_from_image_id_series(df_val["image_id"]))

    def _tfrecs_for_shards(shards):
        return [
            os.path.join(TRAIN_TFREC_DIR, f"ld_train{int(s):02d}-1338.tfrec")
            for s in shards
        ]

    trn_tfrecs = _tfrecs_for_shards(trn_shards)
    val_tfrecs = _tfrecs_for_shards(val_shards)

    ds_trn = tf.data.TFRecordDataset(
        trn_tfrecs, num_parallel_reads=AUTOTUNE
    ).with_options(data_opts)
    ds_val = tf.data.TFRecordDataset(
        val_tfrecs, num_parallel_reads=AUTOTUNE
    ).with_options(data_opts)

    ds_trn = ds_trn.map(_parse_train_example, num_parallel_calls=AUTOTUNE).cache()
    ds_val = ds_val.map(_parse_train_example, num_parallel_calls=AUTOTUNE).cache()

    trn_ids = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(df_trn["image_id"].values, dtype=tf.string),
            values=tf.ones([len(df_trn)], dtype=tf.int32),
        ),
        default_value=0,
    )
    val_ids = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(df_val["image_id"].values, dtype=tf.string),
            values=tf.ones([len(df_val)], dtype=tf.int32),
        ),
        default_value=0,
    )

    def _is_trn(img, lbl, name):
        return tf.equal(trn_ids.lookup(name), 1)

    def _is_val(img, lbl, name):
        return tf.equal(val_ids.lookup(name), 1)

    ds_trn = ds_trn.filter(_is_trn).map(
        lambda img, lbl, name: (img, lbl), num_parallel_calls=AUTOTUNE
    )
    ds_val = ds_val.filter(_is_val).map(
        lambda img, lbl, name: (img, lbl), num_parallel_calls=AUTOTUNE
    )

    ds_trn = ds_trn.shuffle(8192, seed=SEED, reshuffle_each_iteration=True)
    ds_trn = ds_trn.map(
        lambda img, lbl: (_augment_with_rotation(img), lbl), num_parallel_calls=AUTOTUNE
    )

    ds_trn = ds_trn.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
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



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NotFoundError                             Traceback (most recent call last)
/tmp/ipykernel_11/1423361636.py in <cell line: 0>()
    211 )
    212 
--> 213 my_model.fit(
    214     ds_trn,
    215     validation_data=ds_val,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
     57       e.message += " name: " + name
     58     raise core._status_to_exception(e) from None
---> 59   except TypeError as e:
     60     keras_symbolic_tensors = [x for x in inputs if _is_keras_symbolic_tensor(x)]
     61     if keras_symbolic_tensors:

NotFoundError: Graph execution error:

Detected at node IteratorGetNext defined at (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main

  File "<frozen runpy>", line 88, in _run_code

  File "/usr/local/lib/python3.11/dist-packages/colab_kernel_launcher.py", line 37, in <module>

  File "/usr/local/lib/python3.11/dist-packages/traitlets/config/application.py", line 992, in launch_instance

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelapp.py", line 712, in start

  File "/usr/local/lib/python3.11/dist-packages/tornado/platform/asyncio.py", line 211, in start

  File "/usr/lib/python3.11/asyncio/base_events.py", line 608, in run_forever

  File "/usr/lib/python3.11/asyncio/base_events.py", line 1936, in _run_once

  File "/usr/lib/python3.11/asyncio/events.py", line 84, in _run

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 510, in dispatch_queue

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 499, in process_one

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 406, in dispatch_shell

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 730, in execute_request

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/ipkernel.py", line 383, in do_execute

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/zmqshell.py", line 528, in run_cell

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 2975, in run_cell

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3030, in _run_cell

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/async_helpers.py", line 78, in _pseudo_sync_runner

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3257, in run_cell_async

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3473, in run_ast_nodes

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3553, in run_code

  File "/tmp/ipykernel_11/1423361636.py", line 213, in <cell line: 0>

  File "/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py", line 117, in error_handler

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 371, in fit

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 219, in function

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 132, in multi_step_on_iterator

../input/cassava-leaf-disease-classification/train_tfrecords/ld_train163-1338.tfrec; No such file or directory
	 [[{{node IteratorGetNext}}]] [Op:__inference_multi_step_on_iterator_26957]

## === cell 2
sample = pd.read_csv(SAMPLE_SUB)
sample["image_id"] = sample["image_id"].astype(str)
sample_image_ids = sample["image_id"].tolist()

test_tfrecs = sorted(tf.io.gfile.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec")))


def _parse_test_example(example_proto):
    feats = {
        "image": tf.io.FixedLenFeature([], tf.string, default_value=b""),
        "image_name": tf.io.FixedLenFeature([], tf.string, default_value=b""),
        "image_id": tf.io.FixedLenFeature([], tf.string, default_value=b""),
    }
    ex = tf.io.parse_single_example(example_proto, feats)
    img = _decode_and_resize(ex["image"])
    name = tf.where(
        tf.strings.length(ex["image_name"]) > 0, ex["image_name"], ex["image_id"]
    )
    return img, name


use_test_tfrecords = len(test_tfrecs) > 0

if use_test_tfrecords:
    test_ds = tf.data.TFRecordDataset(test_tfrecs, num_parallel_reads=AUTOTUNE)
    test_ds = test_ds.with_options(data_opts)
    test_ds = test_ds.map(_parse_test_example, num_parallel_calls=AUTOTUNE)

    test_ds_cached = test_ds.cache()
    test_imgs = (
        test_ds_cached.map(lambda img, name: img, num_parallel_calls=AUTOTUNE)
        .batch(128)
        .prefetch(AUTOTUNE)
    )
    test_names = np.concatenate(
        [n.numpy() for n in test_ds_cached.map(lambda img, name: name).batch(1024)],
        axis=0,
    ).astype("U")

    pred_test = my_model.predict(test_imgs, verbose=0)
    pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

    pred_df = pd.DataFrame({"image_id": test_names, "label": pred_test_labels})

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
