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

Not yielded

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
data_opts.experimental_optimization.apply_default_optimizations = True
data_opts.experimental_optimization.autotune_buffers = True
data_opts.experimental_optimization.autotune_cpu_budget = 0
data_opts.experimental_optimization.autotune_ram_budget = 0


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


if use_train_tfrecords:
    trn_ids_tensor = tf.constant(df_trn["image_id"].values, dtype=tf.string)
    val_ids_tensor = tf.constant(df_val["image_id"].values, dtype=tf.string)

    trn_ids_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=trn_ids_tensor,
            values=tf.ones([tf.shape(trn_ids_tensor)[0]], dtype=tf.int32),
        ),
        default_value=0,
    )
    val_ids_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=val_ids_tensor,
            values=tf.ones([tf.shape(val_ids_tensor)[0]], dtype=tf.int32),
        ),
        default_value=0,
    )

    _feats_train_min = {
        "image": tf.io.FixedLenFeature([], tf.string, default_value=b""),
        "image_name": tf.io.FixedLenFeature([], tf.string, default_value=b""),
        "image_id": tf.io.FixedLenFeature([], tf.string, default_value=b""),
        "label": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
        "target": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
    }

    def _parse_train_min(example_proto):
        ex = tf.io.parse_single_example(example_proto, _feats_train_min)
        name = tf.where(
            tf.strings.length(ex["image_name"]) > 0, ex["image_name"], ex["image_id"]
        )
        lbl = tf.where(ex["label"] >= 0, ex["label"], ex["target"])
        lbl = tf.cast(lbl, tf.int32)
        return ex["image"], lbl, name

    def _is_in_trn(image_bytes, lbl, name):
        return tf.equal(trn_ids_table.lookup(name), 1)

    def _is_in_val(image_bytes, lbl, name):
        return tf.equal(val_ids_table.lookup(name), 1)

    ds_all = tf.data.TFRecordDataset(
        train_tfrecs, num_parallel_reads=AUTOTUNE
    ).with_options(data_opts)
    ds_all = ds_all.map(_parse_train_min, num_parallel_calls=AUTOTUNE)

    ds_trn = ds_all.filter(_is_in_trn).map(
        lambda b, y, n: (b, y), num_parallel_calls=AUTOTUNE
    )
    ds_val = ds_all.filter(_is_in_val).map(
        lambda b, y, n: (b, y), num_parallel_calls=AUTOTUNE
    )

    def _decode_pair(image_bytes, lbl):
        img = _decode_and_resize(image_bytes)
        return img, lbl

    ds_trn = ds_trn.shuffle(8192, seed=SEED, reshuffle_each_iteration=True)
    ds_trn = ds_trn.map(_decode_pair, num_parallel_calls=AUTOTUNE)
    ds_trn = ds_trn.map(
        lambda img, lbl: (_augment_with_rotation(img), lbl), num_parallel_calls=AUTOTUNE
    )
    ds_trn = ds_trn.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

    ds_val = ds_val.map(_decode_pair, num_parallel_calls=AUTOTUNE)
    ds_val = ds_val.cache()
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
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/4152121766.py in <cell line: 0>()
     17 data_opts.experimental_deterministic = True
     18 data_opts.experimental_optimization.apply_default_optimizations = True
---> 19 data_opts.experimental_optimization.autotune_buffers = True
     20 data_opts.experimental_optimization.autotune_cpu_budget = 0
     21 data_opts.experimental_optimization.autotune_ram_budget = 0

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/options.py in __setattr__(self, name, value)
     59       object.__setattr__(self, name, value)
     60     else:
---> 61       raise AttributeError("Cannot set the property {} on {}.".format(
     62           name,
     63           type(self).__name__))

AttributeError: Cannot set the property autotune_buffers on OptimizationOptions.

## === cell 2
sample = pd.read_csv(SAMPLE_SUB)
sample["image_id"] = sample["image_id"].astype(str)
sample_image_ids = sample["image_id"].tolist()

test_tfrecs = sorted(tf.io.gfile.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec")))

_feets_test = {
    "image": tf.io.FixedLenFeature([], tf.string, default_value=b""),
    "image_name": tf.io.FixedLenFeature([], tf.string, default_value=b""),
    "image_id": tf.io.FixedLenFeature([], tf.string, default_value=b""),
}


def _parse_test_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _feets_test)
    img = _decode_and_resize(ex["image"])
    name = tf.where(
        tf.strings.length(ex["image_name"]) > 0, ex["image_name"], ex["image_id"]
    )
    return img, name


use_test_tfrecords = len(test_tfrecs) > 0

if use_test_tfrecords:
    test_ds = tf.data.TFRecordDataset(
        test_tfrecs, num_parallel_reads=AUTOTUNE
    ).with_options(data_opts)
    test_ds = test_ds.map(_parse_test_example, num_parallel_calls=AUTOTUNE)

    test_ds_batched = test_ds.batch(128).prefetch(AUTOTUNE)

    pred_names = []
    pred_labels = []
    for batch_imgs, batch_names in test_ds_batched:
        batch_pred = my_model(batch_imgs, training=False)
        batch_pred = tf.argmax(batch_pred, axis=-1, output_type=tf.int32)
        pred_labels.append(batch_pred.numpy())
        pred_names.append(batch_names.numpy())

    pred_labels = np.concatenate(pred_labels, axis=0).astype(int)
    pred_names = np.concatenate(pred_names, axis=0)
    pred_names = np.array([x.decode("utf-8") for x in pred_names], dtype=object)

    pred_df = pd.DataFrame({"image_id": pred_names, "label": pred_labels})

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



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1053646064.py in <cell line: 0>()
     29         test_tfrecs, num_parallel_reads=AUTOTUNE
     30     ).with_options(data_opts)
---> 31     test_ds = test_ds.map(_parse_test_example, num_parallel_calls=AUTOTUNE)
     32 
     33     test_ds_batched = test_ds.batch(128).prefetch(AUTOTUNE)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in map(self, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)
   2339     from tensorflow.python.data.ops import map_op
   2340 
-> 2341     return map_op._map_v2(
   2342         self,
   2343         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py in _map_v2(input_dataset, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)
     55           num_parallel_calls,
     56       )
---> 57     return _ParallelMapDataset(
     58         input_dataset,
     59         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py in __init__(self, input_dataset, map_func, num_parallel_calls, deterministic, use_inter_op_parallelism, preserve_cardinality, use_legacy_function, use_unbounded_threadpool, name)
    200     self._input_dataset = input_dataset
    201     self._use_inter_op_parallelism = use_inter_op_parallelism
--> 202     self._map_func = structured_function.StructuredFunctionWrapper(
    203         map_func,
    204         self._transformation_name(),

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in __init__(self, func, transformation_name, dataset, input_classes, input_shapes, input_types, input_structure, add_to_graph, use_legacy_function, defun_kwargs)
    263         fn_factory = trace_tf_function(defun_kwargs)
    264 
--> 265     self._function = fn_factory()
    266     # There is no graph to add in eager mode.
    267     add_to_graph &= not context.executing_eagerly()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in get_concrete_function(self, *args, **kwargs)
   1249   def get_concrete_function(self, *args, **kwargs):
   1250     # Implements PolymorphicFunction.get_concrete_function.
-> 1251     concrete = self._get_concrete_function_garbage_collected(*args, **kwargs)
   1252     concrete._garbage_collector.release()  # pylint: disable=protected-access
   1253     return concrete

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in _get_concrete_function_garbage_collected(self, *args, **kwargs)
   1219       if self._variable_creation_config is None:
   1220         initializers = []
-> 1221         self._initialize(args, kwargs, add_initializers_to=initializers)
   1222         self._initialize_uninitialized_variables(initializers)
   1223 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in _initialize(self, args, kwds, add_initializers_to)
    694     )
    695     # Force the definition of the function for these arguments
--> 696     self._concrete_variable_creation_fn = tracing_compilation.trace_function(
    697         args, kwds, self._variable_creation_config
    698     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in trace_function(args, kwargs, tracing_options)
    176       kwargs = {}
    177 
--> 178     concrete_function = _maybe_define_function(
    179         args, kwargs, tracing_options
    180     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in _maybe_define_function(args, kwargs, tracing_options)
    281         else:
    282           target_func_type = lookup_func_type
--> 283         concrete_function = _create_concrete_function(
    284             target_func_type, lookup_func_context, func_graph, tracing_options
    285         )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in _create_concrete_function(function_type, type_context, func_graph, tracing_options)
    308       attributes_lib.DISABLE_ACD, False
    309   )
--> 310   traced_func_graph = func_graph_module.func_graph_from_py_func(
    311       tracing_options.name,
    312       tracing_options.python_function,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/func_graph.py in func_graph_from_py_func(name, python_func, args, kwargs, signature, func_graph, add_control_dependencies, arg_names, op_return_value, collections, capture_by_value, create_placeholders)
   1057 
   1058     _, original_func = tf_decorator.unwrap(python_func)
-> 1059     func_outputs = python_func(*func_args, **func_kwargs)
   1060 
   1061     # invariant: `func_outputs` contains only Tensors, CompositeTensors,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in wrapped_fn(*args, **kwds)
    597         # the function a weak reference to itself to avoid a reference cycle.
    598         with OptionalXlaContext(compile_with_xla):
--> 599           out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    600         return out
    601 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in wrapped_fn(*args)
    229       # Note: wrapper_helper will apply autograph based on context.
    230       def wrapped_fn(*args):  # pylint: disable=missing-docstring
--> 231         ret = wrapper_helper(*args)
    232         ret = structure.to_tensor_list(self._output_structure, ret)
    233         return [ops.convert_to_tensor(t) for t in ret]

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in wrapper_helper(*args)
    159       if not _should_unpack(nested_args):
    160         nested_args = (nested_args,)
--> 161       ret = autograph.tf_convert(self._func, ag_ctx)(*nested_args)
    162       ret = variable_utils.convert_variables_to_tensors(ret)
    163       if _should_pack(ret):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in wrapper(*args, **kwargs)
    691       except Exception as e:  # pylint:disable=broad-except
    692         if hasattr(e, 'ag_error_metadata'):
--> 693           raise e.ag_error_metadata.to_exception(e)
    694         else:
    695           raise

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in wrapper(*args, **kwargs)
    688       try:
    689         with conversion_ctx:
--> 690           return converted_call(f, args, kwargs, options=options)
    691       except Exception as e:  # pylint:disable=broad-except
    692         if hasattr(e, 'ag_error_metadata'):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    437     try:
    438       if kwargs is not None:
--> 439         result = converted_f(*effective_args, **kwargs)
    440       else:
    441         result = converted_f(*effective_args)

/tmp/__autograph_generated_filexquehwqq.py in tf___parse_test_example(example_proto)
      9                 retval_ = ag__.UndefinedReturnValue()
     10                 ex = ag__.converted_call(ag__.ld(tf).io.parse_single_example, (ag__.ld(example_proto), ag__.ld(_feets_test)), None, fscope)
---> 11                 img = ag__.converted_call(ag__.ld(_decode_and_resize), (ag__.ld(ex)['image'],), None, fscope)
     12                 name = ag__.converted_call(ag__.ld(tf).where, (ag__.converted_call(ag__.ld(tf).strings.length, (ag__.ld(ex)['image_name'],), None, fscope) > 0, ag__.ld(ex)['image_name'], ag__.ld(ex)['image_id']), None, fscope)
     13                 try:

NameError: in user code:

    File "/tmp/ipykernel_11/1053646064.py", line 16, in _parse_test_example  *
        img = _decode_and_resize(ex["image"])

    NameError: name '_decode_and_resize' is not defined


## === cell 3
final_csv.head()

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1842027079.py in <cell line: 0>()
----> 1 final_csv.head()

NameError: name 'final_csv' is not defined
