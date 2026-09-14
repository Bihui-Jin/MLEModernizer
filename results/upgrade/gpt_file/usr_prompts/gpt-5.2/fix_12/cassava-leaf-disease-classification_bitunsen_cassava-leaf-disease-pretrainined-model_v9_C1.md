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

0.885312783318223

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import json
import random
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")




## === cell 1
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model

print("TF version:", tf.__version__)

try:
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(
        max(1, (os.cpu_count() or 2) // 2)
    )
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images/"

TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUB_CSV = os.path.join(BASE_DIR, "sample_submission.csv")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB_CSV), f"Missing {SAMPLE_SUB_CSV}"
assert os.path.isdir(TRAIN_DIR), f"Missing {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing {TEST_DIR}"




## === cell 3
with open(os.path.join(BASE_DIR, "label_num_to_disease_map.json")) as file:
    map_classes = json.loads(file.read())

print(json.dumps(map_classes, indent=2))

label_list = sorted([int(k) for k in map_classes.keys()])
NUM_CLASSES = len(label_list)
print("Labels:", label_list, "NUM_CLASSES:", NUM_CLASSES)




## === cell 4
train_df_tmp = pd.read_csv(TRAIN_CSV, usecols=["image_id", "label"])
sample_sub_tmp = pd.read_csv(SAMPLE_SUB_CSV, usecols=["image_id", "label"])
print(f"Number of train images (from CSV): {len(train_df_tmp)}")
print(f"Number of test images (from sample_submission): {len(sample_sub_tmp)}")




## === cell 5
IMG_HEIGHT = 500
IMG_WIDTH = 500
batch_size = 16

PRE_TRAINED_MODEL = "../input/xceptionv8/Cassava_Best_Xception_Model_V08.hdf5"
PRETRAINED_EXISTS = os.path.exists(PRE_TRAINED_MODEL)
print("Pretrained path exists:", PRETRAINED_EXISTS, PRE_TRAINED_MODEL)




## === cell 6
AUTOTUNE = tf.data.AUTOTUNE

TRAIN_TFREC_DIR = os.path.join(BASE_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(BASE_DIR, "test_tfrecords")


def _list_tfrec_files(tfrec_dir, prefix):
    if not tf.io.gfile.exists(tfrec_dir):
        return []
    files = tf.io.gfile.glob(os.path.join(tfrec_dir, f"{prefix}*.tfrec"))
    return sorted(files)


TEST_TFRECS = _list_tfrec_files(TEST_TFREC_DIR, "ld_test")
TRAIN_TFRECS = _list_tfrec_files(TRAIN_TFREC_DIR, "ld_train")


def _build_path(data_type, image_id):
    base = TEST_DIR if data_type == "TEST_DATA" else TRAIN_DIR
    return tf.strings.join([base, image_id], separator=os.sep)


def _center_crop_or_pad(img, target_h, target_w):
    return tf.image.resize_with_crop_or_pad(img, target_h, target_w)


def _decode_center_crop_normalize_from_path(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)  # uint8
    img = _center_crop_or_pad(img, IMG_HEIGHT, IMG_WIDTH)
    img = tf.cast(img, tf.float32) / 255.0
    return img


def _augment_train_tf(img, seed2):
    do_aug = tf.random.stateless_uniform([], seed=seed2, minval=0.0, maxval=1.0) > 0.5

    def _apply():
        s = tf.cast(seed2, tf.int32)
        img2 = img

        img2 = tf.image.stateless_random_flip_left_right(
            img2, seed=s + tf.constant([1, 0], tf.int32)
        )
        img2 = tf.image.stateless_random_flip_up_down(
            img2, seed=s + tf.constant([2, 0], tf.int32)
        )

        img2 = tf.image.stateless_random_brightness(
            img2, max_delta=0.2, seed=s + tf.constant([3, 0], tf.int32)
        )
        img2 = tf.image.stateless_random_contrast(
            img2, lower=0.8, upper=1.2, seed=s + tf.constant([4, 0], tf.int32)
        )
        img2 = tf.clip_by_value(img2, 0.0, 1.0)

        k = tf.random.stateless_uniform(
            [],
            seed=s + tf.constant([5, 0], tf.int32),
            minval=0,
            maxval=4,
            dtype=tf.int32,
        )
        img2 = tf.image.rot90(img2, k=k)

        img2 = _center_crop_or_pad(img2, IMG_HEIGHT, IMG_WIDTH)

        return img2

    return tf.cond(do_aug, _apply, lambda: img)


_DATASET_OPTIONS = tf.data.Options()
_DATASET_OPTIONS.experimental_deterministic = True
try:
    _DATASET_OPTIONS.experimental_optimization.apply_default_optimizations = True
except Exception:
    pass
try:
    _DATASET_OPTIONS.threading.private_threadpool_size = max(4, (os.cpu_count() or 8))
except Exception:
    pass
try:
    _DATASET_OPTIONS.experimental_optimization.map_parallelization = True
    _DATASET_OPTIONS.experimental_optimization.parallel_batch = True
    _DATASET_OPTIONS.experimental_optimization.autotune_buffers = True
    _DATASET_OPTIONS.experimental_optimization.autotune = True
    _DATASET_OPTIONS.experimental_optimization.map_and_batch_fusion = True
    _DATASET_OPTIONS.experimental_optimization.filter_fusion = True
    _DATASET_OPTIONS.experimental_optimization.map_fusion = True
    _DATASET_OPTIONS.experimental_optimization.noop_elimination = True
    _DATASET_OPTIONS.experimental_optimization.shuffle_and_repeat_fusion = True
except Exception:
    pass


def make_train_dataset(image_ids, labels, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((image_ids, labels))
    ds = ds.shuffle(
        buffer_size=len(image_ids), seed=SEED, reshuffle_each_iteration=True
    )

    def _map_fn(idx, x):
        image_id, y = x
        path = _build_path("TRAIN_DATA", image_id)
        img = _decode_center_crop_normalize_from_path(path)
        seed = tf.stack([tf.cast(SEED, tf.int32), tf.cast(idx, tf.int32)])
        img = _augment_train_tf(img, seed)
        return img, tf.cast(y, tf.int64)

    ds = ds.enumerate()
    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    ds = ds.with_options(_DATASET_OPTIONS)
    return ds


def make_eval_dataset(data_type, image_ids, labels, batch_size, cache=False):
    cache = False

    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(image_ids)

        def _map_fn(image_id):
            path = _build_path(data_type, image_id)
            img = _decode_center_crop_normalize_from_path(path)
            return img

        ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)
        if cache:
            ds = ds.cache()
        ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
        ds = ds.with_options(_DATASET_OPTIONS)
        return ds

    ds = tf.data.Dataset.from_tensor_slices((image_ids, labels))

    def _map_fn(image_id, y):
        path = _build_path(data_type, image_id)
        img = _decode_center_crop_normalize_from_path(path)
        return img, tf.cast(y, tf.int64)

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)
    if cache:
        ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    ds = ds.with_options(_DATASET_OPTIONS)
    return ds


_FEATURE_DESC_TEST = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}

_FEATURE_DESC_TRAIN = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


def make_test_dataset_from_tfrecords(tfrec_files, batch_size, cache=False):
    cache = False

    ds = tf.data.TFRecordDataset(tfrec_files, num_parallel_reads=AUTOTUNE)

    def _parse(ex):
        ex = tf.io.parse_single_example(ex, _FEATURE_DESC_TEST)
        img = tf.image.decode_jpeg(ex["image"], channels=3)
        img = _center_crop_or_pad(img, IMG_HEIGHT, IMG_WIDTH)
        img = tf.cast(img, tf.float32) / 255.0
        return img

    ds = ds.map(_parse, num_parallel_calls=AUTOTUNE, deterministic=True)
    if cache:
        ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    ds = ds.with_options(_DATASET_OPTIONS)
    return ds


def make_eval_dataset_from_tfrecords_indexed(
    tfrec_files, batch_size, valid_image_ids, cache=False
):
    name_to_pos = {}
    pos = 0
    for raw in tf.data.TFRecordDataset(tfrec_files, num_parallel_reads=1):
        ex = tf.train.Example.FromString(raw.numpy())
        nm = ex.features.feature["image_name"].bytes_list.value[0].decode("utf-8")
        name_to_pos[nm] = pos
        pos += 1

    positions = [name_to_pos[i] for i in valid_image_ids if i in name_to_pos]
    positions = np.array(positions, dtype=np.int64)

    def _parse(ex):
        ex = tf.io.parse_single_example(ex, _FEATURE_DESC_TRAIN)
        img = tf.image.decode_jpeg(ex["image"], channels=3)
        img = _center_crop_or_pad(img, IMG_HEIGHT, IMG_WIDTH)
        img = tf.cast(img, tf.float32) / 255.0
        y = tf.cast(ex["target"], tf.int64)
        return img, y

    pos_set = tf.constant(positions, dtype=tf.int64)
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            pos_set, tf.ones_like(pos_set, dtype=tf.int64)
        ),
        default_value=0,
    )

    ds = tf.data.TFRecordDataset(tfrec_files, num_parallel_reads=AUTOTUNE)
    ds = ds.map(_parse, num_parallel_calls=AUTOTUNE, deterministic=True).enumerate()

    def _keep(i, xy):
        return table.lookup(i) > 0

    ds = ds.filter(_keep).map(
        lambda i, xy: xy, num_parallel_calls=AUTOTUNE, deterministic=True
    )

    if cache:
        ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    ds = ds.with_options(_DATASET_OPTIONS)
    return ds




## === cell 7
train_df = train_df_tmp
sample_sub = sample_sub_tmp

print("train_df:", train_df.shape, "sample_sub:", sample_sub.shape)
assert set(train_df.columns) == {"image_id", "label"}
assert set(sample_sub.columns) == {"image_id", "label"}

from sklearn.model_selection import train_test_split

train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=SEED,
    shuffle=True,
    stratify=train_df["label"].values,
)

df_train = train_df.iloc[train_idx].reset_index(drop=True)
df_val = train_df.iloc[val_idx].reset_index(drop=True)

print("df_train:", df_train.shape, "df_val:", df_val.shape)

if len(TRAIN_TFRECS) > 0:
    print(
        f"Using train TFRecords ({len(TRAIN_TFRECS)} files) for faster training input."
    )
    train_ds_raw = tf.data.TFRecordDataset(TRAIN_TFRECS, num_parallel_reads=AUTOTUNE)

    def _parse_train(ex):
        ex = tf.io.parse_single_example(ex, _FEATURE_DESC_TRAIN)
        img = tf.image.decode_jpeg(ex["image"], channels=3)
        img = _center_crop_or_pad(img, IMG_HEIGHT, IMG_WIDTH)
        img = tf.cast(img, tf.float32) / 255.0
        y = tf.cast(ex["target"], tf.int64)
        return img, y

    train_gen = (
        train_ds_raw.map(_parse_train, num_parallel_calls=AUTOTUNE, deterministic=True)
        .shuffle(buffer_size=8192, seed=SEED, reshuffle_each_iteration=True)
        .enumerate()
        .map(
            lambda idx, xy: (
                _augment_train_tf(
                    xy[0], tf.stack([tf.cast(SEED, tf.int32), tf.cast(idx, tf.int32)])
                ),
                xy[1],
            ),
            num_parallel_calls=AUTOTUNE,
            deterministic=True,
        )
        .batch(batch_size, drop_remainder=False)
        .prefetch(AUTOTUNE)
        .with_options(_DATASET_OPTIONS)
    )
else:
    train_gen = make_train_dataset(
        df_train["image_id"].values.astype(str),
        df_train["label"].values.astype(np.int64),
        batch_size=batch_size,
    )

val_gen = make_eval_dataset(
    "VALIDATE_DATA",
    df_val["image_id"].values.astype(str),
    df_val["label"].values.astype(np.int64),
    batch_size=batch_size,
    cache=True,  # ignored internally to prevent huge cache spill; semantics unchanged.
)




## === cell 8
def build_xception_model(img_h, img_w, num_classes):
    inp = layers.Input(shape=(img_h, img_w, 3))
    base = tf.keras.applications.Xception(
        include_top=False, weights="imagenet", input_tensor=inp, pooling="avg"
    )
    x = base.output
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=inp, outputs=out)
    return model


if PRETRAINED_EXISTS:
    model = tf.keras.models.load_model(PRE_TRAINED_MODEL, compile=False)
    print("Loaded pretrained model from:", PRE_TRAINED_MODEL)
else:
    model = build_xception_model(IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES)
    print("Built new Xception-based model (imagenet weights).")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=32,
)

model.summary()




## === cell 9
if not PRETRAINED_EXISTS:
    EPOCHS = 2
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        verbose=1,
    )




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3747223592.py in <cell line: 0>()
      1 if not PRETRAINED_EXISTS:
      2     EPOCHS = 2
----> 3     history = model.fit(
      4         train_gen,
      5         validation_data=val_gen,

/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
     68             # To get the full stack trace, call:
     69             # `tf.debugging.disable_traceback_filtering()`
---> 70             raise e.with_traceback(filtered_tb) from None
     71         finally:
     72             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tf_keras/src/engine/data_adapter.py in _validate_data_handler(self)
   1505             and self._inferred_steps is None
   1506         ):
-> 1507             raise ValueError(
   1508                 "Could not infer the size of the data. With "
   1509                 "`steps_per_execution > 1`, you must specify the number of "

ValueError: Could not infer the size of the data. With `steps_per_execution > 1`, you must specify the number of steps to run.

## === cell 10
test_df = pd.DataFrame({"image_id": sample_sub["image_id"].values})

if len(TEST_TFRECS) > 0:
    print(f"Using test TFRecords ({len(TEST_TFRECS)} files) for faster input.")
    test_gen = make_test_dataset_from_tfrecords(
        TEST_TFRECS,
        batch_size=batch_size,
        cache=True,  # ignored internally; semantics unchanged.
    )
else:
    print("Test TFRecords not found; using test_images directory.")
    test_gen = make_eval_dataset(
        "TEST_DATA",
        test_df["image_id"].values.astype(str),
        labels=None,
        batch_size=batch_size,
        cache=True,  # ignored internally; semantics unchanged.
    )




## === cell 11
try:
    model.compile(
        optimizer=model.optimizer,
        loss=model.loss,
        metrics=model.metrics,
        steps_per_execution=32,
    )
except Exception:
    pass

pred_probs = model.predict(
    test_gen,
    verbose=1,
)
pred_labels = np.argmax(pred_probs, axis=1).astype(int)

print("Pred probs shape:", pred_probs.shape, "Pred labels shape:", pred_labels.shape)
assert len(pred_labels) == len(test_df)




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1175616914.py in <cell line: 0>()
     11     pass
     12 
---> 13 pred_probs = model.predict(
     14     test_gen,
     15     verbose=1,

/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
     68             # To get the full stack trace, call:
     69             # `tf.debugging.disable_traceback_filtering()`
---> 70             raise e.with_traceback(filtered_tb) from None
     71         finally:
     72             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tf_keras/src/engine/data_adapter.py in _validate_data_handler(self)
   1505             and self._inferred_steps is None
   1506         ):
-> 1507             raise ValueError(
   1508                 "Could not infer the size of the data. With "
   1509                 "`steps_per_execution > 1`, you must specify the number of "

ValueError: Could not infer the size of the data. With `steps_per_execution > 1`, you must specify the number of steps to run.

## === cell 12
submission = sample_sub.copy()
submission["label"] = pred_labels
submission["label"] = submission["label"].astype(int)

out_path = "submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission.head(3))
print("Submission shape:", submission.shape)
assert out_path.endswith(".csv") and os.path.exists(out_path)
assert list(submission.columns) == ["image_id", "label"]
assert submission.shape[0] == sample_sub.shape[0]

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2464716658.py in <cell line: 0>()
      1 submission = sample_sub.copy()
----> 2 submission["label"] = pred_labels
      3 submission["label"] = submission["label"].astype(int)
      4 
      5 out_path = "submission.csv"

NameError: name 'pred_labels' is not defined
