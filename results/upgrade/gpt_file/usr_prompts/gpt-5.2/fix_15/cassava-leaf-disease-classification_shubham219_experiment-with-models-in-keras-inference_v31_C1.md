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
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import glob
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.model_selection import train_test_split

SEED = 42
DEBUG = False

tf.random.set_seed(SEED)
np.random.seed(SEED)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

print("TF:", tf.__version__)




## === cell 1
def first_existing_path(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


DATA_DIR = first_existing_path(
    [
        "/kaggle/input/cassava-leaf-disease-classification",
        "../input/cassava-leaf-disease-classification",
        "/kaggle/data/cassava-leaf-disease-classification",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    ]
)
if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not find cassava-leaf-disease-classification dataset directory in expected locations."
    )

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
TRAIN_TFREC_DIR = os.path.join(DATA_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(DATA_DIR, "test_tfrecords")

weight_path = (
    first_existing_path(
        [
            "../input/model-v04/clf_new_21 (4).h5",
            "/kaggle/input/model-v04/clf_new_21 (4).h5",
            "/kaggle/input/model-v04/clf_new_21%20(4).h5",
        ]
    )
    or "../input/model-v04/clf_new_21 (4).h5"
)

IMG_SIZE = (300, 300)
NUM_CLASSES = 5

print("DATA_DIR:", DATA_DIR)
print("TRAIN_CSV exists:", os.path.exists(TRAIN_CSV))
print("SAMPLE_SUB exists:", os.path.exists(SAMPLE_SUB))
print("TRAIN_IMG_DIR exists:", os.path.exists(TRAIN_IMG_DIR))
print("TEST_IMG_DIR exists:", os.path.exists(TEST_IMG_DIR))
print("TRAIN_TFREC_DIR exists:", os.path.exists(TRAIN_TFREC_DIR))
print("TEST_TFREC_DIR exists:", os.path.exists(TEST_TFREC_DIR))
print("weight_path:", weight_path)
print("weight_path exists:", os.path.exists(weight_path))




## === cell 2
AUTOTUNE = tf.data.AUTOTUNE


def _ds_options(deterministic=True):
    opts = tf.data.Options()
    opts.experimental_deterministic = bool(deterministic)
    try:
        opts.experimental_optimization.apply_default_optimizations = True
        opts.experimental_optimization.map_parallelization = True
        opts.experimental_optimization.parallel_batch = True
        opts.experimental_optimization.autotune_buffers = True
    except Exception:
        pass
    return opts


@tf.function(reduce_retracing=True)
def _decode_resize_preprocess(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(
        img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)  # keep same preprocessing semantics
    return img


@tf.function(reduce_retracing=True)
def _load_preprocess_with_label(path, label):
    img_bytes = tf.io.read_file(path)
    img = _decode_resize_preprocess(img_bytes)
    return img, tf.cast(label, tf.int32)


@tf.function(reduce_retracing=True)
def _load_preprocess(path):
    img_bytes = tf.io.read_file(path)
    img = _decode_resize_preprocess(img_bytes)
    return img


_FEATURE_DESC_TRAIN = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}
_FEATURE_DESC_TEST = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


@tf.function(reduce_retracing=True)
def _parse_train_tfrecord(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURE_DESC_TRAIN)
    img = _decode_resize_preprocess(ex["image"])
    label = tf.cast(ex["target"], tf.int32)
    return img, label


@tf.function(reduce_retracing=True)
def _parse_test_tfrecord(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURE_DESC_TEST)
    img = _decode_resize_preprocess(ex["image"])
    image_name = ex["image_name"]
    return img, image_name


def make_train_val_ds(df_train, df_val, batch_size=32):
    train_paths = tf.constant(df_train["path"].values)
    train_labels = tf.constant(df_train["label"].values)
    val_paths = tf.constant(df_val["path"].values)
    val_labels = tf.constant(df_val["label"].values)

    opts = _ds_options(deterministic=True)

    ds_train = tf.data.Dataset.from_tensor_slices(
        (train_paths, train_labels)
    ).with_options(opts)
    ds_train = ds_train.shuffle(
        buffer_size=min(len(df_train), 4096), seed=SEED, reshuffle_each_iteration=True
    )
    ds_train = ds_train.map(_load_preprocess_with_label, num_parallel_calls=AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.batch(batch_size, drop_remainder=False)
    ds_train = ds_train.prefetch(AUTOTUNE)
    ds_train = ds_train.apply(tf.data.experimental.ignore_errors())

    ds_val = tf.data.Dataset.from_tensor_slices((val_paths, val_labels)).with_options(
        opts
    )
    ds_val = ds_val.map(_load_preprocess_with_label, num_parallel_calls=AUTOTUNE)
    ds_val = ds_val.cache()
    ds_val = ds_val.batch(batch_size, drop_remainder=False)
    ds_val = ds_val.prefetch(AUTOTUNE)
    ds_val = ds_val.apply(tf.data.experimental.ignore_errors())

    return ds_train, ds_val


def make_test_ds(paths, batch_size=128):
    paths = tf.constant(paths)
    ds = tf.data.Dataset.from_tensor_slices(paths).with_options(
        _ds_options(deterministic=True)
    )
    ds = ds.map(_load_preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    return ds


def make_train_val_ds_from_tfrecords(tfrecord_files, df_train, df_val, batch_size=32):
    opts = _ds_options(deterministic=True)

    train_keys = tf.constant(df_train["image_id"].values.astype("S"))
    val_keys = tf.constant(df_val["image_id"].values.astype("S"))
    train_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            train_keys, tf.ones_like(train_keys, dtype=tf.int32)
        ),
        default_value=0,
    )
    val_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            val_keys, tf.ones_like(val_keys, dtype=tf.int32)
        ),
        default_value=0,
    )

    files_ds = tf.data.Dataset.from_tensor_slices(
        tf.constant(tfrecord_files)
    ).with_options(opts)
    files_ds = files_ds.interleave(
        lambda f: tf.data.TFRecordDataset(f, num_parallel_reads=AUTOTUNE),
        cycle_length=min(len(tfrecord_files), 16),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )

    parsed = files_ds.map(_parse_train_tfrecord, num_parallel_calls=AUTOTUNE)

    raise RuntimeError(
        "Train TFRecords in this dataset do not include image_name; cannot split by df without changing semantics."
    )


def make_test_ds_from_tfrecords(tfrecord_files, batch_size=256):
    opts = _ds_options(deterministic=False)
    files_ds = tf.data.Dataset.from_tensor_slices(
        tf.constant(tfrecord_files)
    ).with_options(opts)
    ds = files_ds.interleave(
        lambda f: tf.data.TFRecordDataset(f, num_parallel_reads=AUTOTUNE),
        cycle_length=min(len(tfrecord_files), 32),
        num_parallel_calls=AUTOTUNE,
        deterministic=False,
    )
    ds = ds.map(_parse_test_tfrecord, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    return ds




## === cell 3
my_model = None

if os.path.exists(weight_path):
    my_model = load_model(weight_path, compile=False)
    print("Loaded pretrained model from:", weight_path)
else:
    print(
        "Pretrained weights not found; training a fallback EfficientNetB3 model to produce a valid submission."
    )

    df = pd.read_csv(TRAIN_CSV)

    df["path"] = (TRAIN_IMG_DIR.rstrip("/") + "/" + df["image_id"].values).astype(str)

    df_train, df_val = train_test_split(
        df, test_size=0.15, random_state=SEED, stratify=df["label"]
    )

    BATCH_SIZE_TRAIN = 32

    ds_train, ds_val = make_train_val_ds(df_train, df_val, batch_size=BATCH_SIZE_TRAIN)

    inp = keras.Input(shape=(*IMG_SIZE, 3))
    base = EfficientNetB3(include_top=False, weights="imagenet", input_tensor=inp)
    x = keras.layers.GlobalAveragePooling2D()(base.output)
    x = keras.layers.Dropout(0.2)(x)
    out = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    my_model = keras.Model(inputs=inp, outputs=out)

    base.trainable = False
    my_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    my_model.fit(ds_train, validation_data=ds_val, epochs=1, verbose=2)

    base.trainable = True
    my_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    my_model.fit(ds_train, validation_data=ds_val, epochs=1, verbose=2)

print("Model ready:", isinstance(my_model, keras.Model))




## === cell 4
sample = pd.read_csv(SAMPLE_SUB)
test_image_ids = sample["image_id"].values

BATCH_SIZE_TEST = 256

test_tfrecord_files = (
    sorted(glob.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec")))
    if os.path.exists(TEST_TFREC_DIR)
    else []
)

if len(test_tfrecord_files) > 0:
    test_ds = make_test_ds_from_tfrecords(
        test_tfrecord_files, batch_size=BATCH_SIZE_TEST
    )

    test_ds_imgs = test_ds.map(lambda img, name: img, num_parallel_calls=AUTOTUNE)
    pred_probs = my_model.predict(test_ds_imgs, verbose=0)

    names = np.concatenate([bn.numpy() for _, bn in test_ds], axis=0).astype("U")

    name_to_idx = {n: i for i, n in enumerate(names)}
    order = np.fromiter(
        (name_to_idx[n] for n in test_image_ids),
        dtype=np.int64,
        count=len(test_image_ids),
    )
    pred_test = pred_probs[order]
else:
    test_paths = (
        (TEST_IMG_DIR.rstrip("/") + "/" + test_image_ids.astype(str))
        .astype(str)
        .tolist()
    )
    test_ds = make_test_ds(test_paths, batch_size=BATCH_SIZE_TEST)
    pred_test = my_model.predict(test_ds, verbose=0)

pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

final_csv = sample.copy()
final_csv["label"] = pred_test_labels

final_csv.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", final_csv.shape)
print(final_csv.head())




## === cell 5
final_csv.head()
