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

3.12

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

0.848443638561499

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.11697) has done: 'The timeout is dominated by slow input decoding (JPEGs from many small files) and an expensive training step that repeatedly resizes/decodes on-CPU while the model waits. To keep core logic identical, the main speedups are: (1) force use of the provided TFRecords (sequential reads, much faster than per-file JPEG I/O), (2) add `.cache()` appropriately (RAM) to avoid re-decoding across epochs (validation always; training deterministically cached after shuffle to keep per-epoch shuffle behavior), and (3) set explicit `steps_per_epoch/validation_steps` so Keras doesn’t spend time re-walking datasets for cardinality. All changes preserve the same model, loss, epochs, metric, image preprocessing, and evaluation semantics (no sampling, no precision changes, no early stopping).'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import efficientnet_v2

if os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION") == "python":
    del os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]

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

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

print("TF:", tf.__version__)
print("Eager:", tf.executing_eagerly())




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
WORK_DIR = "../input/cassava-leaf-disease-classification/"
TRAIN_CSV = os.path.join(WORK_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(WORK_DIR, "train_images")
TEST_IMG_DIR = os.path.join(WORK_DIR, "test_images")
SAMPLE_SUB = os.path.join(WORK_DIR, "sample_submission.csv")
TRAIN_TFREC_DIR = os.path.join(WORK_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(WORK_DIR, "test_tfrecords")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing dir: {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing dir: {TEST_IMG_DIR}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"

train_df = pd.read_csv(TRAIN_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

print(train_df.head())
print("Train rows:", len(train_df), "Test rows:", len(sample_df))




## === cell 2
class SigmoidFocalCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0, from_logits=False, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        if self.from_logits:
            y_pred = tf.sigmoid(y_pred)
        y_pred = tf.clip_by_value(
            y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon()
        )
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(
            1 - y_pred
        )
        weight = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
        focal_loss = weight * ((1 - y_pred) ** self.gamma) * cross_entropy
        return tf.reduce_sum(focal_loss, axis=-1)


custom_objects = {"SigmoidFocalCrossEntropy": SigmoidFocalCrossEntropy}




## === cell 3
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 5
EPOCHS = 3  # unchanged

from sklearn.model_selection import train_test_split

tr_df, va_df = train_test_split(
    train_df, test_size=0.1, random_state=SEED, stratify=train_df["label"]
)

train_tfrec_files = []
if os.path.isdir(TRAIN_TFREC_DIR):
    train_tfrec_files = sorted(
        [
            os.path.join(TRAIN_TFREC_DIR, f)
            for f in os.listdir(TRAIN_TFREC_DIR)
            if f.endswith(".tfrec")
            or f.endswith(".tfrecord")
            or f.endswith(".tfrec.gz")
            or f.endswith(".tfrecord.gz")
        ]
    )
test_tfrec_files = []
if os.path.isdir(TEST_TFREC_DIR):
    test_tfrec_files = sorted(
        [
            os.path.join(TEST_TFREC_DIR, f)
            for f in os.listdir(TEST_TFREC_DIR)
            if f.endswith(".tfrec")
            or f.endswith(".tfrecord")
            or f.endswith(".tfrec.gz")
            or f.endswith(".tfrecord.gz")
        ]
    )

tr_ids = tr_df["image_id"].values
tr_labels = tr_df["label"].values.astype(np.int32)
va_ids = va_df["image_id"].values
va_labels = va_df["label"].values.astype(np.int32)

use_tfrecords = bool(train_tfrec_files) and bool(test_tfrec_files)
print(
    "TFRecords present:",
    use_tfrecords,
    "| train_tfrec_files:",
    len(train_tfrec_files),
    "| test_tfrec_files:",
    len(test_tfrec_files),
)


@tf.function(reduce_retracing=True)
def _decode_resize_preprocess_from_bytes(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method="bilinear")
    img = tf.cast(img, tf.float32)
    img = efficientnet_v2.preprocess_input(img)
    return img


@tf.function(reduce_retracing=True)
def _decode_resize_from_path(path):
    img_bytes = tf.io.read_file(path)
    return _decode_resize_preprocess_from_bytes(img_bytes)


def _set_deterministic_opts(ds):
    opts = tf.data.Options()
    opts.experimental_deterministic = True
    return ds.with_options(opts)


def _parse_train_example(example_proto):
    feature_spec = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64),
        "image_name": tf.io.FixedLenFeature([], tf.string, default_value=b""),
    }
    ex = tf.io.parse_single_example(example_proto, feature_spec)
    img = _decode_resize_preprocess_from_bytes(ex["image"])
    label = tf.cast(ex["target"], tf.int32)
    image_id = ex["image_name"]
    return img, label, image_id


def make_ds_from_tfrecords(files, training: bool, allowed_ids_set):
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
    ds = _set_deterministic_opts(ds)
    ds = ds.map(_parse_train_example, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.filter(lambda img, label, image_id: allowed_ids_set.contains(image_id))

    ds = ds.map(
        lambda img, label, image_id: (img, label), num_parallel_calls=tf.data.AUTOTUNE
    )

    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    else:
        pass

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def make_ds_from_jpegs(image_ids, labels, training: bool):
    ds = tf.data.Dataset.from_tensor_slices((image_ids, labels))
    ds = _set_deterministic_opts(ds)

    def load_and_preprocess(image_id, label):
        path = tf.strings.join([TRAIN_IMG_DIR, "/", image_id])
        img = _decode_resize_from_path(path)
        label = tf.cast(label, tf.int32)
        return img, label

    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    else:
        ds = ds.cache()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


if use_tfrecords:
    tr_id_set = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(tr_ids, dtype=tf.string),
            values=tf.ones([len(tr_ids)], dtype=tf.int8),
        ),
        default_value=tf.constant(0, dtype=tf.int8),
    )
    va_id_set = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(va_ids, dtype=tf.string),
            values=tf.ones([len(va_ids)], dtype=tf.int8),
        ),
        default_value=tf.constant(0, dtype=tf.int8),
    )

    class _ContainsWrapper:
        def __init__(self, table):
            self.table = table

        @tf.function(reduce_retracing=True)
        def contains(self, key):
            return tf.not_equal(self.table.lookup(key), 0)

    train_ds = make_ds_from_tfrecords(
        train_tfrec_files, training=True, allowed_ids_set=_ContainsWrapper(tr_id_set)
    )
    val_ds = make_ds_from_tfrecords(
        train_tfrec_files, training=False, allowed_ids_set=_ContainsWrapper(va_id_set)
    )
else:
    train_ds = make_ds_from_jpegs(tr_ids, tr_labels, training=True)
    val_ds = make_ds_from_jpegs(va_ids, va_labels, training=False)

base = efficientnet_v2.EfficientNetV2B0(
    include_top=False,
    weights=None,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    pooling="avg",
)

inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base(inputs, training=True)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    jit_compile=True,
)

model.summary()

train_steps = int(np.ceil(len(tr_df) / BATCH_SIZE))
val_steps = int(np.ceil(len(va_df) / BATCH_SIZE))
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=2,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NotFoundError                             Traceback (most recent call last)
/tmp/ipykernel_11/2540211144.py in <cell line: 0>()
    135 if use_tfrecords:
    136     # Build deterministic lookup sets once (O(n)), used inside tf.data.filter efficiently.
--> 137     tr_id_set = tf.lookup.StaticHashTable(
    138         tf.lookup.KeyValueTensorInitializer(
    139             keys=tf.constant(tr_ids, dtype=tf.string),

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

NotFoundError: Could not find device for node: {{node HashTableV2}} = HashTableV2[container="", key_dtype=DT_STRING, shared_name="7", use_node_name_sharing=false, value_dtype=DT_INT8]
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
test_image_ids = sample_df["image_id"].values


def _parse_test_example(example_proto):
    feature_spec = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string, default_value=b""),
    }
    ex = tf.io.parse_single_example(example_proto, feature_spec)
    img = _decode_resize_preprocess_from_bytes(ex["image"])
    image_id = ex["image_name"]
    return img, image_id


def make_test_ds_from_jpegs(image_ids):
    ds = tf.data.Dataset.from_tensor_slices(image_ids)
    ds = _set_deterministic_opts(ds)

    def load_test_image(image_id):
        path = tf.strings.join([TEST_IMG_DIR, "/", image_id])
        img = _decode_resize_from_path(path)
        return img, image_id

    ds = ds.map(load_test_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def make_test_ds_from_tfrecords(files):
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
    ds = _set_deterministic_opts(ds)
    ds = ds.map(_parse_test_example, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


if bool(test_tfrec_files) and os.path.isdir(TEST_TFREC_DIR) and use_tfrecords:
    test_ds = make_test_ds_from_tfrecords(test_tfrec_files)
else:
    test_ds = make_test_ds_from_jpegs(test_image_ids)

if bool(test_tfrec_files) and os.path.isdir(TEST_TFREC_DIR) and use_tfrecords:
    probs_list = []
    ids_list = []
    for batch_imgs, batch_ids in test_ds:
        probs_list.append(model(batch_imgs, training=False).numpy())
        ids_list.append(batch_ids.numpy())
    probs = np.concatenate(probs_list, axis=0)
    pred_ids = np.concatenate(ids_list, axis=0).astype("U")
else:
    probs = model.predict(test_ds, verbose=0)
    pred_ids = test_image_ids.astype("U")

pred_labels = np.argmax(probs, axis=1).astype(int)

pred_df = pd.DataFrame({"image_id": pred_ids, "label": pred_labels})

submission = sample_df[["image_id"]].merge(pred_df, on="image_id", how="left")

if submission["label"].isna().any():
    submission["label"] = submission["label"].fillna(0).astype(int)
else:
    submission["label"] = submission["label"].astype(int)

submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with rows:", len(submission))
print("File exists:", os.path.exists("submission.csv"))
print("Missing labels after merge:", int(submission["label"].isna().sum()))

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1270709142.py in <cell line: 0>()
     51     ids_list = []
     52     for batch_imgs, batch_ids in test_ds:
---> 53         probs_list.append(model(batch_imgs, training=False).numpy())
     54         ids_list.append(batch_ids.numpy())
     55     probs = np.concatenate(probs_list, axis=0)

NameError: name 'model' is not defined
