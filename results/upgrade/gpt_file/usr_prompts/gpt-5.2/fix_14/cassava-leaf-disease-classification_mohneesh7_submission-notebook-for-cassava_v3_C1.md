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

0.6143850105772136

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

print("TF version:", tf.__version__)

BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_images")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test_images")
TRAIN_TFREC_DIR = os.path.join(BASE_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(BASE_DIR, "test_tfrecords")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing: {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing: {TEST_IMG_DIR}"
assert os.path.isdir(TRAIN_TFREC_DIR), f"Missing: {TRAIN_TFREC_DIR}"
assert os.path.isdir(TEST_TFREC_DIR), f"Missing: {TEST_TFREC_DIR}"




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
df = pd.read_csv(TRAIN_CSV)
df["label"] = df["label"].astype(str)  # categorical mode expects string labels

df["filepath"] = (
    TRAIN_IMG_DIR + os.sep + df["image_id"].astype(str).to_numpy()
).astype(str)

sample_paths = df["filepath"].head(10).to_list()
missing = sum(not os.path.exists(p) for p in sample_paths)
print("Train rows:", len(df), "| Missing (first 10 checked):", missing)
df.head()




## === cell 2
def stratified_split_dataframe(dataframe, label_col, test_size=0.15, seed=42):
    rng = np.random.RandomState(seed)
    labels = dataframe[label_col].to_numpy()
    all_idx = np.arange(len(dataframe))

    train_idx_list = []
    valid_idx_list = []

    uniq, inv = np.unique(labels, return_inverse=True)
    for k in range(len(uniq)):
        idx = all_idx[inv == k]
        rng.shuffle(idx)
        n_valid = int(np.round(len(idx) * test_size))
        n_valid = min(max(n_valid, 1), len(idx) - 1) if len(idx) > 1 else 0
        valid_idx_list.append(idx[:n_valid])
        train_idx_list.append(idx[n_valid:])

    train_idx = (
        np.concatenate(train_idx_list) if train_idx_list else np.array([], dtype=int)
    )
    valid_idx = (
        np.concatenate(valid_idx_list) if valid_idx_list else np.array([], dtype=int)
    )

    rng2 = np.random.RandomState(seed)
    rng2.shuffle(train_idx)
    rng2.shuffle(valid_idx)

    train_df = dataframe.iloc[train_idx].reset_index(drop=True)
    valid_df = dataframe.iloc[valid_idx].reset_index(drop=True)
    return train_df, valid_df


train_df, valid_df = stratified_split_dataframe(df, "label", test_size=0.15, seed=SEED)
print("Train:", len(train_df), "Valid:", len(valid_df))




## === cell 3
IMG_SIZE = 380
BATCH_SIZE = 16  # keep moderate to fit typical Kaggle GPU memory
NUM_CLASSES = 5
EPOCHS = 5  # unchanged to preserve training approach/semantics

AUTO = tf.data.AUTOTUNE

train_tfrecs = tf.io.gfile.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec"))
test_tfrecs = tf.io.gfile.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec"))
assert len(train_tfrecs) > 0, "No train TFRecords found."
assert len(test_tfrecs) > 0, "No test TFRecords found."
train_tfrecs = sorted(train_tfrecs)
test_tfrecs = sorted(test_tfrecs)

FEATURES_TRAIN = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}
FEATURES_TEST = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


def _decode_resize_from_bytes(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear")
    return img


def _augment_stateless(img, seed2):
    img = tf.image.stateless_random_flip_left_right(img, seed=seed2)

    k = tf.random.stateless_uniform(
        [],
        seed=seed2 ^ tf.constant([17, 23], tf.int32),
        minval=0,
        maxval=4,
        dtype=tf.int32,
    )
    img = tf.image.rot90(img, k)

    scale = tf.random.stateless_uniform(
        [], seed=seed2 ^ tf.constant([29, 31], tf.int32), minval=0.9, maxval=1.1
    )
    new_size = tf.cast(tf.round(scale * IMG_SIZE), tf.int32)
    img2 = tf.image.resize(img, [new_size, new_size], method="bilinear")

    max_shift = tf.cast(tf.round(0.05 * IMG_SIZE), tf.int32)
    dx = tf.random.stateless_uniform(
        [],
        seed=seed2 ^ tf.constant([37, 41], tf.int32),
        minval=-max_shift,
        maxval=max_shift + 1,
        dtype=tf.int32,
    )
    dy = tf.random.stateless_uniform(
        [],
        seed=seed2 ^ tf.constant([43, 47], tf.int32),
        minval=-max_shift,
        maxval=max_shift + 1,
        dtype=tf.int32,
    )

    img2 = tf.image.resize_with_crop_or_pad(
        img2, IMG_SIZE + 2 * max_shift, IMG_SIZE + 2 * max_shift
    )
    start_x = max_shift + dx
    start_y = max_shift + dy
    img2 = tf.image.crop_to_bounding_box(img2, start_y, start_x, IMG_SIZE, IMG_SIZE)
    return img2


def _preprocess(img):
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)  # identical Keras ResNet50 preprocessing
    return img


options = tf.data.Options()
options.experimental_deterministic = True


def _tfrecord_has_feature(tfrec_paths, feature_name: str) -> bool:
    raw = next(iter(tf.data.TFRecordDataset(tfrec_paths).take(1))).numpy()
    ex = tf.train.Example.FromString(raw)
    return feature_name in ex.features.feature


has_name = _tfrecord_has_feature(train_tfrecs, "image_name")

raw_train = tf.data.TFRecordDataset(train_tfrecs, num_parallel_reads=AUTO).with_options(
    options
)


def _attach_name_train(example_proto):
    ex = tf.io.parse_single_example(
        example_proto,
        {
            "image": tf.io.FixedLenFeature([], tf.string),
            "target": tf.io.FixedLenFeature([], tf.int64),
            "image_name": tf.io.FixedLenFeature([], tf.string, default_value=b""),
        },
    )
    img = _decode_resize_from_bytes(ex["image"])
    label_int = tf.cast(ex["target"], tf.int32)
    name = ex["image_name"]
    return img, label_int, name


def _parse_train_no_name(example_proto):
    ex = tf.io.parse_single_example(example_proto, FEATURES_TRAIN)
    img = _decode_resize_from_bytes(ex["image"])
    return img, tf.cast(ex["target"], tf.int32)


def _to_onehot(label_int):
    return tf.one_hot(label_int, NUM_CLASSES)


if not has_name:
    def _decode_resize_from_path(path):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear")
        return img

    label_lookup = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant([str(i) for i in range(NUM_CLASSES)], dtype=tf.string),
            values=tf.constant(list(range(NUM_CLASSES)), dtype=tf.int32),
        ),
        default_value=tf.constant(-1, dtype=tf.int32),
    )

    def _parse_train_from_df(filepath, label_str):
        img = _decode_resize_from_path(filepath)
        h = tf.strings.to_hash_bucket_fast(filepath, 2**31 - 1)
        seed2 = tf.stack([tf.constant(SEED, tf.int32), tf.cast(h, tf.int32)])
        img = _augment_stateless(img, seed2)
        img = _preprocess(img)
        label_int = label_lookup.lookup(label_str)
        label_oh = tf.one_hot(label_int, NUM_CLASSES)
        return img, label_oh

    def _parse_valid_from_df(filepath, label_str):
        img = _decode_resize_from_path(filepath)
        img = _preprocess(img)
        label_int = label_lookup.lookup(label_str)
        label_oh = tf.one_hot(label_int, NUM_CLASSES)
        return img, label_oh

    train_ds = (
        tf.data.Dataset.from_tensor_slices(
            (
                train_df["filepath"].values.astype(str),
                train_df["label"].values.astype(str),
            )
        )
        .with_options(options)
        .shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
        .map(_parse_train_from_df, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTO)
    )

    valid_ds = (
        tf.data.Dataset.from_tensor_slices(
            (
                valid_df["filepath"].values.astype(str),
                valid_df["label"].values.astype(str),
            )
        )
        .with_options(options)
        .map(_parse_valid_from_df, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE, drop_remainder=False)
        .cache()
        .prefetch(AUTO)
    )
else:
    base_ds = raw_train.map(_attach_name_train, num_parallel_calls=AUTO)

    id_to_index = {
        img_id: i for i, img_id in enumerate(df["image_id"].astype(str).values)
    }
    train_idx = np.fromiter(
        (id_to_index[i] for i in train_df["image_id"].astype(str).values),
        dtype=np.int64,
    )
    valid_idx = np.fromiter(
        (id_to_index[i] for i in valid_df["image_id"].astype(str).values),
        dtype=np.int64,
    )

    train_idx_tf = tf.constant(np.sort(train_idx), dtype=tf.int64)
    valid_idx_tf = tf.constant(np.sort(valid_idx), dtype=tf.int64)

    train_idx_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=train_idx_tf,
            values=tf.ones_like(train_idx_tf, dtype=tf.int8),
        ),
        default_value=tf.constant(0, tf.int8),
    )
    valid_idx_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=valid_idx_tf,
            values=tf.ones_like(valid_idx_tf, dtype=tf.int8),
        ),
        default_value=tf.constant(0, tf.int8),
    )

    def _format_for_train(img, label_int, name):
        h = tf.strings.to_hash_bucket_fast(name, 2**31 - 1)
        seed2 = tf.stack([tf.constant(SEED, tf.int32), tf.cast(h, tf.int32)])
        img = _augment_stateless(img, seed2)
        img = _preprocess(img)
        return img, _to_onehot(label_int)

    def _format_for_valid(img, label_int, name):
        img = _preprocess(img)
        return img, _to_onehot(label_int)

    def _keep_train(i, x):
        return train_idx_table.lookup(i) > 0

    def _keep_valid(i, x):
        return valid_idx_table.lookup(i) > 0

    def _drop_i(i, x):
        return x

    enum_ds = (
        base_ds.enumerate()
    )  # yields (index_in_tfrecord_stream, (img, label, name))

    train_ds = (
        enum_ds.filter(_keep_train)
        .map(_drop_i, num_parallel_calls=AUTO)
        .shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
        .map(_format_for_train, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTO)
    )

    valid_ds = (
        enum_ds.filter(_keep_valid)
        .map(_drop_i, num_parallel_calls=AUTO)
        .map(_format_for_valid, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE, drop_remainder=False)
        .cache()
        .prefetch(AUTO)
    )

train_steps = int(np.ceil(len(train_df) / BATCH_SIZE))
valid_steps = int(np.ceil(len(valid_df) / BATCH_SIZE))

print(
    "Steps - train:",
    train_steps,
    "valid:",
    valid_steps,
    "| TFRecord image_name present:",
    has_name,
)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NotFoundError                             Traceback (most recent call last)
/tmp/ipykernel_11/3107656600.py in <cell line: 0>()
    207     valid_idx_tf = tf.constant(np.sort(valid_idx), dtype=tf.int64)
    208 
--> 209     train_idx_table = tf.lookup.StaticHashTable(
    210         tf.lookup.KeyValueTensorInitializer(
    211             keys=train_idx_tf,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/trackable/resource.py in __call__(cls, *args, **kwargs)
    101       previous_getter = _make_getter(getter, previous_getter)
    102 
--> 103     return previous_getter(*args, **kwargs)
    104 
    105 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/trackable/resource.py in <lambda>(*a, **kw)
     96       return obj
     97 
---> 98     previous_getter = lambda *a, **kw: default_resource_creator(None, *a, **kw)
     99     resource_creator_stack = ops.get_default_graph()._resource_creator_stack
    100     for getter in resource_creator_stack[cls._resource_type()]:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/trackable/resource.py in default_resource_creator(next_creator, *a, **kw)
     93       assert next_creator is None
     94       obj = cls.__new__(cls, *a, **kw)
---> 95       obj.__init__(*a, **kw)
     96       return obj
     97 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/lookup_ops.py in __init__(self, initializer, default_value, name, experimental_is_anonymous)
    351     self._name = name or "hash_table"
    352     self._table_name = None
--> 353     super(StaticHashTable, self).__init__(default_value, initializer)
    354     self._value_shape = self._default_value.get_shape()
    355 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/lookup_ops.py in __init__(self, default_value, initializer)
    199       self._initializer = self._track_trackable(initializer, "_initializer")
    200     with ops.init_scope():
--> 201       self._resource_handle = self._create_resource()
    202     if (not context.executing_eagerly() and
    203         ops.get_default_graph()._get_control_flow_context() is not None):  # pylint: disable=protected-access

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/lookup_ops.py in _create_resource(self)
    361           name=self._name)
    362     else:
--> 363       table_ref = gen_lookup_ops.hash_table_v2(
    364           shared_name=self._shared_name,
    365           key_dtype=self._initializer.key_dtype,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_lookup_ops.py in hash_table_v2(key_dtype, value_dtype, container, shared_name, use_node_name_sharing, name)
    483       return _result
    484     except _core._NotOkStatusException as e:
--> 485       _ops.raise_from_not_ok_status(e, name)
    486     except _core._FallbackException:
    487       pass

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

NotFoundError: Could not find device for node: {{node HashTableV2}} = HashTableV2[container="", key_dtype=DT_INT64, shared_name="69", use_node_name_sharing=false, value_dtype=DT_INT8]
All kernels registered for op HashTableV2:
  device='CPU'; key_dtype in [DT_STRING]; value_dtype in [DT_STRING]
  device='CPU'; key_dtype in [DT_STRING]; value_dtype in [DT_INT64]
  device='CPU'; key_dtype in [DT_STRING]; value_dtype in [DT_INT32]
  device='CPU'; key_dtype in [DT_STRING]; value_dtype in [DT_FLOAT]
  device='CPU'; key_dtype in [DT_STRING]; value_dtype in [DT_DOUBLE]
  device='CPU'; key_dtype in [DT_STRING]; value_dtype in [DT_BOOL]
  device='CPU'; key_dtype in [DT_INT64]; value_dtype in [DT_STRING]
  device='CPU'; key_dtype in [DT_INT64]; value_dtype in [DT_INT64]
  device='CPU'; key_dtype in [DT_INT64]; value_dtype in [DT_INT32]
  device='CPU'; key_dtype in [DT_INT64]; value_dtype in [DT_FLOAT]
  device='CPU'; key_dtype in [DT_INT64]; value_dtype in [DT_DOUBLE]
  device='CPU'; key_dtype in [DT_INT32]; value_dtype in [DT_STRING]
  device='CPU'; key_dtype in [DT_INT32]; value_dtype in [DT_INT32]
  device='CPU'; key_dtype in [DT_INT32]; value_dtype in [DT_FLOAT]
  device='CPU'; key_dtype in [DT_INT32]; value_dtype in [DT_DOUBLE]
 [Op:HashTableV2] name: hash_table

## === cell 4
base = ResNet50(
    include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base.trainable = False  # unchanged transfer-learning approach

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=16,
)

model.summary()




## === cell 5
ckpt_path = "/kaggle/working/best_model.keras"
callbacks = [
    ModelCheckpoint(
        ckpt_path, monitor="val_accuracy", save_best_only=True, mode="max", verbose=1
    ),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, verbose=1),
]

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
    steps_per_epoch=train_steps,
    validation_steps=valid_steps,
)

best_model = (
    tf.keras.models.load_model(ckpt_path) if os.path.exists(ckpt_path) else model
)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/308832395.py in <cell line: 0>()
      8 
      9 history = model.fit(
---> 10     train_ds,
     11     validation_data=valid_ds,
     12     epochs=EPOCHS,

NameError: name 'train_ds' is not defined

## === cell 6
sub = pd.read_csv(SAMPLE_SUB)
sub["filepath"] = (
    TEST_IMG_DIR + os.sep + sub["image_id"].astype(str).to_numpy()
).astype(str)

sample_test_paths = sub["filepath"].head(10).to_list()
missing_test = sum(not os.path.exists(p) for p in sample_test_paths)
print("Test rows:", len(sub), "| Missing (first 10 checked):", missing_test)

raw_test = tf.data.TFRecordDataset(test_tfrecs, num_parallel_reads=AUTO).with_options(
    options
)
raw_test = raw_test.map(_parse_test_example, num_parallel_calls=AUTO)


def _prep_test(img, name):
    img = _preprocess(img)
    return img, name


test_ds_named = (
    raw_test.map(_prep_test, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE, drop_remainder=False)
    .cache()
    .prefetch(AUTO)
)

test_steps = int(np.ceil(len(sub) / BATCH_SIZE))

probs_and_names = best_model.predict(test_ds_named, verbose=1, steps=test_steps)
name_ds = (
    raw_test.map(lambda img, name: name, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)
names = np.concatenate([b.numpy() for b in name_ds], axis=0).astype("S")

probs = probs_and_names
preds = probs.argmax(axis=1).astype(int)

pred_df = pd.DataFrame(
    {"image_id": [n.decode("utf-8") for n in names], "label": preds.astype(int)}
)

submission = sub[["image_id"]].merge(pred_df, on="image_id", how="left")

if submission["label"].isna().any():

    def _decode_resize_from_path(path):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear")
        return img

    def _parse_test_from_path(filepath):
        img = _decode_resize_from_path(filepath)
        img = _preprocess(img)
        return img

    test_ds = (
        tf.data.Dataset.from_tensor_slices(sub["filepath"].values.astype(str))
        .with_options(options)
        .map(_parse_test_from_path, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE, drop_remainder=False)
        .cache()
        .prefetch(AUTO)
    )
    probs2 = best_model.predict(test_ds, verbose=1, steps=test_steps)
    submission["label"] = probs2.argmax(axis=1).astype(int)
else:
    submission["label"] = submission["label"].astype(int)

submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote:", os.path.abspath("submission.csv"), "| rows:", len(submission))




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2785880594.py in <cell line: 0>()
     11     options
     12 )
---> 13 raw_test = raw_test.map(_parse_test_example, num_parallel_calls=AUTO)
     14 
     15 

NameError: name '_parse_test_example' is not defined

## === cell 7
assert os.path.exists("submission.csv"), "submission.csv was not created."
chk = pd.read_csv("submission.csv")
assert list(chk.columns) == [
    "image_id",
    "label",
], f"Wrong columns: {chk.columns.tolist()}"
assert len(chk) == len(
    pd.read_csv(SAMPLE_SUB)
), "Row count mismatch vs sample_submission."
chk.head()

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_11/1925187473.py in <cell line: 0>()
----> 1 assert os.path.exists("submission.csv"), "submission.csv was not created."
      2 chk = pd.read_csv("submission.csv")
      3 assert list(chk.columns) == [
      4     "image_id",
      5     "label",

AssertionError: submission.csv was not created.
