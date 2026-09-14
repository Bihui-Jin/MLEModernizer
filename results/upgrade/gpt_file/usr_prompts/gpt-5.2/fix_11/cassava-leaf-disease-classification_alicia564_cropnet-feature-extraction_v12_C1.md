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

3.13

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

0.8881837413115745

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.19955) has done: 'I remove `tensorflow_hub` usage (it’s triggering the protobuf `MessageFactory.GetPrototype` crash in this environment) and instead build the same kind of image classifier directly with `tf.keras` EfficientNetB0, which you already import. I also fix the broken SavedModel/TFSMLayer path logic by deleting that dependency and ensuring `model` is always defined via the trained Keras model. To keep the core approach intact (transfer learning on 224×224 images with augmentation and categorical loss), I train using your existing `ImageDataGenerator` generators and then generate predictions aligned to `sample_submission.csv` order. Finally, I guarantee a valid `/kaggle/working/submission.csv` with `image_id,label` columns is written.'
- What this solution (achieved 0.23057) has done: 'The timeout is dominated by expensive JPEG decode/resize every epoch plus heavy on-the-fly augmentation, and by an inefficient caching setup that forces large disk cache writes and still re-runs the augmentation pipeline each epoch. I keep the same model, losses, epochs, and augmentation, but make the input pipeline provably equivalent while reducing redundant work: (1) switch to TFRecords (already provided) to avoid filesystem JPEG overhead, (2) cache only the *decoded+resized* tensors in-memory (not to disk) to eliminate slow cache file I/O, and (3) add pipeline knobs that improve throughput without changing semantics (parallel reads, non-blocking prefetch, and deterministic options retained). Prediction also read TFRecord test shards instead of individual JPEG files, preserving the same preprocessing and output format.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import backend as K

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("enable_op_determinism() not available; continuing. Reason:", repr(e))

tf.config.run_functions_eagerly(False)

print("TensorFlow:", tf.__version__)
print("Eager:", tf.executing_eagerly())




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
from sklearn.model_selection import train_test_split

from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = f"{DATA_DIR}/train.csv"
SAMPLE_SUB = f"{DATA_DIR}/sample_submission.csv"
TRAIN_IMG_DIR = f"{DATA_DIR}/train_images"
TEST_IMG_DIR = f"{DATA_DIR}/test_images"
MAP_JSON = f"{DATA_DIR}/label_num_to_disease_map.json"
TRAIN_TFREC_DIR = f"{DATA_DIR}/train_tfrecords"
TEST_TFREC_DIR = f"{DATA_DIR}/test_tfrecords"

label_to_disease = pd.read_json(MAP_JSON, typ="series")
train_csv = pd.read_csv(TRAIN_CSV)

train_csv["path"] = TRAIN_IMG_DIR + "/" + train_csv["image_id"]
train_csv["label"] = train_csv["label"].astype(int)

train_df, valid_df = train_test_split(
    train_csv, test_size=0.2, stratify=train_csv["label"], random_state=SEED
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = train_csv["label"].nunique()

print("NUM_CLASSES:", NUM_CLASSES)




## === cell 2
base_model = EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)
base_model.trainable = False  # feature extraction stage

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=16,
)

model.summary()




## === cell 3
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback
from tensorflow.keras import layers as L


class EpochEndPrinter(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        msg = {
            k: float(v)
            for k, v in logs.items()
            if isinstance(v, (int, float, np.floating))
        }
        print(f"Epoch {epoch+1} end logs:", msg)


learning_rate_reduction = ReduceLROnPlateau(
    monitor="val_loss", patience=2, factor=0.5, min_lr=1e-6, verbose=1
)

augmenter = tf.keras.Sequential(
    [
        L.RandomRotation(factor=0.125, fill_mode="nearest", seed=SEED),  # ~45 degrees
        L.RandomTranslation(0.2, 0.2, fill_mode="nearest", seed=SEED),
        L.RandomZoom(0.2, 0.2, fill_mode="nearest", seed=SEED),
        L.RandomFlip("horizontal_and_vertical", seed=SEED),
    ],
    name="aug",
)

AUTOTUNE = tf.data.AUTOTUNE
SHUFFLE_BUFFER = 4096

TRAIN_TFRECS = sorted(
    [
        os.path.join(TRAIN_TFREC_DIR, f)
        for f in tf.io.gfile.listdir(TRAIN_TFREC_DIR)
        if f.endswith(".tfrec") or f.endswith(".tfrecord")
    ]
)
TEST_TFRECS = sorted(
    [
        os.path.join(TEST_TFREC_DIR, f)
        for f in tf.io.gfile.listdir(TEST_TFREC_DIR)
        if f.endswith(".tfrec") or f.endswith(".tfrecord")
    ]
)

print("Train TFRecords:", len(TRAIN_TFRECS), "Test TFRecords:", len(TEST_TFRECS))

train_ids = train_df["image_id"].astype(str).tolist()
valid_ids = valid_df["image_id"].astype(str).tolist()

_train_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(train_ids, dtype=tf.string),
        values=tf.ones([len(train_ids)], dtype=tf.int8),
    ),
    default_value=0,
)
_valid_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(valid_ids, dtype=tf.string),
        values=tf.ones([len(valid_ids)], dtype=tf.int8),
    ),
    default_value=0,
)


def _with_ds_options(ds):
    opts = tf.data.Options()
    opts.deterministic = True
    opts.experimental_optimization.apply_default_optimizations = True
    opts.experimental_optimization.map_parallelization = True
    return ds.with_options(opts)


def _decode_resize_from_bytes(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img


_TFREC_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


def _parse_tfrecord(example):
    ex = tf.io.parse_single_example(example, _TFREC_FEATURES)
    img = _decode_resize_from_bytes(ex["image"])
    label = tf.cast(ex["target"], tf.int32)
    image_id = ex["image_name"]
    return img, label, image_id


def _augment_map(img, label):
    img = augmenter(img, training=True)
    label = tf.cast(label, tf.int32)
    return img, label


def _valid_map(img, label):
    label = tf.cast(label, tf.int32)
    return img, label


def _in_train_split(image_id):
    return tf.equal(_train_table.lookup(image_id), 1)


def _in_valid_split(image_id):
    return tf.equal(_valid_table.lookup(image_id), 1)


def make_train_ds_from_tfrecords():
    files = tf.data.Dataset.from_tensor_slices(TRAIN_TFRECS)
    files = files.shuffle(len(TRAIN_TFRECS), seed=SEED, reshuffle_each_iteration=True)
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTOTUNE)
    ds = _with_ds_options(ds)

    ds = ds.map(_parse_tfrecord, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.filter(lambda img, label, image_id: _in_train_split(image_id))
    ds = ds.map(
        lambda img, label, image_id: (img, label),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )

    ds = ds.shuffle(SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.cache()
    ds = ds.map(_augment_map, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_valid_ds_from_tfrecords():
    files = tf.data.Dataset.from_tensor_slices(TRAIN_TFRECS)
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTOTUNE)
    ds = _with_ds_options(ds)

    ds = ds.map(_parse_tfrecord, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.filter(lambda img, label, image_id: _in_valid_split(image_id))
    ds = ds.map(
        lambda img, label, image_id: (img, label),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )

    ds = ds.cache()
    ds = ds.map(_valid_map, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_train_ds_from_tfrecords()
valid_ds = make_valid_ds_from_tfrecords()




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NotFoundError                             Traceback (most recent call last)
/tmp/ipykernel_11/3273661011.py in <cell line: 0>()
     53 valid_ids = valid_df["image_id"].astype(str).tolist()
     54 
---> 55 _train_table = tf.lookup.StaticHashTable(
     56     tf.lookup.KeyValueTensorInitializer(
     57         keys=tf.constant(train_ids, dtype=tf.string),

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

NotFoundError: Could not find device for node: {{node HashTableV2}} = HashTableV2[container="", key_dtype=DT_STRING, shared_name="2877", use_node_name_sharing=false, value_dtype=DT_INT8]
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
EPOCHS = 10  # keep as originally set; no early stopping now

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    callbacks=[learning_rate_reduction, EpochEndPrinter()],
    verbose=1,
)

val_metrics = model.evaluate(valid_ds, verbose=0)
print({"val_loss": float(val_metrics[0]), "val_accuracy": float(val_metrics[1])})




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1154139239.py in <cell line: 0>()
      2 
      3 history = model.fit(
----> 4     train_ds,
      5     validation_data=valid_ds,
      6     epochs=EPOCHS,

NameError: name 'train_ds' is not defined

## === cell 5
FINE_TUNE = True
if FINE_TUNE:
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        steps_per_execution=16,
    )

    history_ft = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=3,
        callbacks=[learning_rate_reduction, EpochEndPrinter()],
        verbose=1,
    )

    val_metrics = model.evaluate(valid_ds, verbose=0)
    print(
        {
            "val_loss_after_ft": float(val_metrics[0]),
            "val_accuracy_after_ft": float(val_metrics[1]),
        }
    )




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2689025232.py in <cell line: 0>()
     13 
     14     history_ft = model.fit(
---> 15         train_ds,
     16         validation_data=valid_ds,
     17         epochs=3,

NameError: name 'train_ds' is not defined

## === cell 6
sample_sub = pd.read_csv(SAMPLE_SUB)
sample_sub["path"] = TEST_IMG_DIR + "/" + sample_sub["image_id"]


def _parse_test_tfrecord(example):
    ex = tf.io.parse_single_example(
        example,
        {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_name": tf.io.FixedLenFeature([], tf.string),
        },
    )
    img = _decode_resize_from_bytes(ex["image"])
    image_id = ex["image_name"]
    return img, image_id


test_files = tf.data.Dataset.from_tensor_slices(TEST_TFRECS)
test_ds_kv = tf.data.TFRecordDataset(test_files, num_parallel_reads=AUTOTUNE)
test_ds_kv = _with_ds_options(test_ds_kv)
test_ds_kv = test_ds_kv.map(
    _parse_test_tfrecord, num_parallel_calls=AUTOTUNE, deterministic=True
)
test_ds_kv = test_ds_kv.cache()
test_ds_kv = test_ds_kv.batch(BATCH_SIZE, drop_remainder=False)
test_ds_kv = test_ds_kv.prefetch(AUTOTUNE)

test_ds = test_ds_kv.map(
    lambda img, image_id: img, num_parallel_calls=AUTOTUNE, deterministic=True
)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2942326373.py in <cell line: 0>()
     18 test_files = tf.data.Dataset.from_tensor_slices(TEST_TFRECS)
     19 test_ds_kv = tf.data.TFRecordDataset(test_files, num_parallel_reads=AUTOTUNE)
---> 20 test_ds_kv = _with_ds_options(test_ds_kv)
     21 test_ds_kv = test_ds_kv.map(
     22     _parse_test_tfrecord, num_parallel_calls=AUTOTUNE, deterministic=True

NameError: name '_with_ds_options' is not defined

## === cell 7
assert "model" in globals() and model is not None

probs = model.predict(test_ds, verbose=1)
preds = np.argmax(probs, axis=1).astype(int)

image_ids = []
for _, batch_ids in test_ds_kv:
    image_ids.extend([x.decode("utf-8") for x in batch_ids.numpy().tolist()])

submission_df = pd.DataFrame({"image_id": image_ids, "label": preds})

submission_df = (
    submission_df.set_index("image_id").loc[sample_sub["image_id"].values].reset_index()
)

out_path = "/kaggle/working/submission.csv"
submission_df.to_csv(out_path, index=False)

print("Submission file created:", out_path)
print(submission_df.head())
print("Rows:", len(submission_df), "Cols:", submission_df.columns.tolist())
assert out_path.endswith(".csv") and len(submission_df) == len(sample_sub)
assert submission_df.columns.tolist() == ["image_id", "label"]

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2527173515.py in <cell line: 0>()
      1 assert "model" in globals() and model is not None
      2 
----> 3 probs = model.predict(test_ds, verbose=1)
      4 preds = np.argmax(probs, axis=1).astype(int)
      5 

NameError: name 'test_ds' is not defined
