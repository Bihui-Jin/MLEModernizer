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

0.8759443940767604

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

keras = tf.keras

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    if hasattr(tf.config.experimental, "enable_op_determinism"):
        tf.config.experimental.enable_op_determinism()
except Exception:
    pass

tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)

print("TensorFlow:", tf.__version__)
print("tf.keras:", tf.keras.__name__)

BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification"
train_dir = os.path.join(BASE_DIR, "train_images")
test_dir = os.path.join(BASE_DIR, "test_images")
train_tfrecord_dir = os.path.join(BASE_DIR, "train_tfrecords")
test_tfrecord_dir = os.path.join(BASE_DIR, "test_tfrecords")

train_csv_path = os.path.join(BASE_DIR, "train.csv")
sample_sub_path = os.path.join(BASE_DIR, "sample_submission.csv")

assert os.path.exists(train_csv_path), f"Missing: {train_csv_path}"
assert os.path.exists(sample_sub_path), f"Missing: {sample_sub_path}"
assert os.path.isdir(train_dir), f"Missing dir: {train_dir}"
assert os.path.isdir(test_dir), f"Missing dir: {test_dir}"
assert os.path.isdir(train_tfrecord_dir), f"Missing dir: {train_tfrecord_dir}"
assert os.path.isdir(test_tfrecord_dir), f"Missing dir: {test_tfrecord_dir}"

train = pd.read_csv(train_csv_path)
sample_sub = pd.read_csv(sample_sub_path)

print(train.head())
print(sample_sub.head())
print("Train size:", len(train), " Test size:", len(sample_sub))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
NUM_CLASSES = 5
IMG_SIZE = (512, 512)
BATCH_SIZE = 16  # keep conservative for memory
AUTOTUNE = tf.data.AUTOTUNE

from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")
try:
    tf.config.optimizer.set_jit(True)  # XLA (same math, faster graph execution)
except Exception:
    pass


def build_model(backbone_name: str, input_shape=(512, 512, 3), num_classes=5):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    if backbone_name == "inceptionresnetv2":
        backbone = keras.applications.InceptionResNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            pooling="avg",
        )
    elif backbone_name == "efficientnetv2b0":
        backbone = keras.applications.EfficientNetV2B0(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            pooling="avg",
        )
    else:
        raise ValueError("Unknown backbone")

    backbone.trainable = True  # preserve original intent (fine-tuning)
    x = backbone(x, training=True)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    model = keras.Model(inputs, outputs)
    return model


model1 = build_model(
    "inceptionresnetv2", input_shape=IMG_SIZE + (3,), num_classes=NUM_CLASSES
)
model2 = build_model(
    "efficientnetv2b0", input_shape=IMG_SIZE + (3,), num_classes=NUM_CLASSES
)

opt1 = keras.optimizers.Adam(learning_rate=1e-4)
opt2 = keras.optimizers.Adam(learning_rate=1e-4)

model1.compile(
    optimizer=opt1,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=32,
)
model2.compile(
    optimizer=opt2,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=32,
)

print("Model1 params:", model1.count_params())
print("Model2 params:", model2.count_params())




## === cell 2
def _make_tfdata_options(deterministic=True):
    options = tf.data.Options()
    options.experimental_deterministic = deterministic
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.OFF
    )
    return options


TFDATA_OPTS_DETERMINISTIC = _make_tfdata_options(deterministic=True)
TFDATA_OPTS_NOND = _make_tfdata_options(deterministic=False)

TRAIN_TFRECS = sorted(
    [
        os.path.join(train_tfrecord_dir, f)
        for f in os.listdir(train_tfrecord_dir)
        if f.endswith(".tfrec")
    ]
)
TEST_TFRECS = sorted(
    [
        os.path.join(test_tfrecord_dir, f)
        for f in os.listdir(test_tfrecord_dir)
        if f.endswith(".tfrec")
    ]
)
assert len(TRAIN_TFRECS) > 0, "No train tfrecords found"
assert len(TEST_TFRECS) > 0, "No test tfrecords found"

FEATURES_TRAIN = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}
FEATURES_TEST = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


@tf.function
def _decode_and_resize(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, IMG_SIZE, method="bilinear", antialias=True)
    img = tf.ensure_shape(img, [IMG_SIZE[0], IMG_SIZE[1], 3])
    img = tf.cast(img, tf.float32) / 255.0
    return img


@tf.function
def _augment_train(x, y):
    x = tf.image.random_flip_left_right(x, seed=SEED)
    return x, y


@tf.function
def _parse_train_example(serialized):
    ex = tf.io.parse_single_example(serialized, FEATURES_TRAIN)
    x = _decode_and_resize(ex["image"])
    y = tf.cast(ex["target"], tf.int32)
    return x, y


@tf.function
def _parse_test_example(serialized):
    ex = tf.io.parse_single_example(serialized, FEATURES_TEST)
    x = _decode_and_resize(ex["image"])
    return x


def make_train_ds_from_tfrecords(tfrecs, training=True):
    ds = tf.data.TFRecordDataset(
        tfrecs, num_parallel_reads=AUTOTUNE, compression_type="GZIP"
    )
    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(_parse_train_example, num_parallel_calls=AUTOTUNE, deterministic=True)
    if training:
        ds = ds.map(_augment_train, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.with_options(TFDATA_OPTS_DETERMINISTIC)
    ds = ds.batch(BATCH_SIZE, drop_remainder=training).prefetch(AUTOTUNE)
    return ds


def make_test_ds_from_tfrecords(tfrecs, batch_size):
    ds = tf.data.TFRecordDataset(
        tfrecs, num_parallel_reads=AUTOTUNE, compression_type="GZIP"
    )
    ds = ds.map(_parse_test_example, num_parallel_calls=AUTOTUNE, deterministic=False)
    ds = ds.with_options(TFDATA_OPTS_NOND)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


num_train_files = len(TRAIN_TFRECS)
val_frac = 0.1
val_files = max(1, int(round(num_train_files * val_frac)))
train_files = num_train_files - val_files
assert train_files > 0 and val_files > 0

TRN_TFRECS = TRAIN_TFRECS[val_files:]
VAL_TFRECS = TRAIN_TFRECS[:val_files]

train_ds = make_train_ds_from_tfrecords(TRN_TFRECS, training=True)
val_ds = make_train_ds_from_tfrecords(VAL_TFRECS, training=False)

print("Train tfrecs:", len(TRN_TFRECS), " Val tfrecs:", len(VAL_TFRECS))
print("Train batches:", tf.data.experimental.cardinality(train_ds).numpy())
print("Val batches:", tf.data.experimental.cardinality(val_ds).numpy())



## === cell 3
EPOCHS = 2  # preserve original training plan

history1 = model1.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)
history2 = model2.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
DataLossError                             Traceback (most recent call last)
/tmp/ipykernel_11/3507870888.py in <cell line: 0>()
      1 EPOCHS = 2  # preserve original training plan
      2 
----> 3 history1 = model1.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)
      4 history2 = model2.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)
      5 

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

DataLossError: Graph execution error:

Detected at node IteratorGetNextAsOptional defined at (most recent call last):
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

  File "/tmp/ipykernel_11/3507870888.py", line 3, in <cell line: 0>

  File "/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py", line 117, in error_handler

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 371, in fit

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 219, in function

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 199, in multi_step_on_iterator

inflate() failed with error -3: incorrect header check
	 [[{{node IteratorGetNextAsOptional}}]] [Op:__inference_multi_step_on_iterator_144324]

## === cell 4
val_batch = next(iter(val_ds.unbatch().batch(50)))
x50, y50 = val_batch
probs1_s = model1.predict(x50, verbose=0)
probs2_s = model2.predict(x50, verbose=0)
probs_s = 0.5 * probs1_s + 0.5 * probs2_s
preds = np.argmax(probs_s, axis=1).astype(int).tolist()

y_true = y50.numpy().astype(np.int32)
acc = (np.array(preds) == y_true).mean()
print("Sample accuracy (50 imgs):", acc)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
DataLossError                             Traceback (most recent call last)
/tmp/ipykernel_11/214793870.py in <cell line: 0>()
----> 1 val_batch = next(iter(val_ds.unbatch().batch(50)))
      2 x50, y50 = val_batch
      3 probs1_s = model1.predict(x50, verbose=0)
      4 probs2_s = model2.predict(x50, verbose=0)
      5 probs_s = 0.5 * probs1_s + 0.5 * probs2_s

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in __next__(self)
    824   def __next__(self):
    825     try:
--> 826       return self._next_internal()
    827     except errors.OutOfRangeError:
    828       raise StopIteration

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in _next_internal(self)
    774     # to communicate that there is no more data to iterate over.
    775     with context.execution_mode(context.SYNC):
--> 776       ret = gen_dataset_ops.iterator_get_next(
    777           self._iterator_resource,
    778           output_types=self._flat_output_types,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_dataset_ops.py in iterator_get_next(iterator, output_types, output_shapes, name)
   3084       return _result
   3085     except _core._NotOkStatusException as e:
-> 3086       _ops.raise_from_not_ok_status(e, name)
   3087     except _core._FallbackException:
   3088       pass

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

DataLossError: {{function_node __wrapped__IteratorGetNext_output_types_2_device_/job:localhost/replica:0/task:0/device:CPU:0}} inflate() failed with error -3: incorrect header check [Op:IteratorGetNext] name: 

## === cell 5
sample_test = pd.DataFrame({"Prediction": preds, "Actual": y_true})
print(sample_test.head(30))



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3081212156.py in <cell line: 0>()
----> 1 sample_test = pd.DataFrame({"Prediction": preds, "Actual": y_true})
      2 print(sample_test.head(30))
      3 

NameError: name 'preds' is not defined

## === cell 6
PRED_BATCH_SIZE = 32

test_ds_pred = make_test_ds_from_tfrecords(TEST_TFRECS, batch_size=PRED_BATCH_SIZE)
print("Test batches:", tf.data.experimental.cardinality(test_ds_pred).numpy())


@tf.function
def _ensemble_predict_step(x):
    p1 = model1(x, training=False)
    p2 = model2(x, training=False)
    return 0.5 * p1 + 0.5 * p2


num_test = int(sample_sub.shape[0])
predictions = np.empty((num_test,), dtype=np.int64)

idx = 0
for xb in test_ds_pred:
    pb = _ensemble_predict_step(xb)
    pred_b = tf.argmax(pb, axis=1, output_type=tf.int64).numpy()
    bs = pred_b.shape[0]
    predictions[idx : idx + bs] = pred_b
    idx += bs

assert idx == num_test, (idx, num_test)

print("Predictions shape:", predictions.shape, "Unique labels:", np.unique(predictions))

submission = pd.DataFrame(
    {"image_id": sample_sub["image_id"].values, "label": predictions.astype(int)}
)
assert submission.shape[0] == sample_sub.shape[0]
assert list(submission.columns) == ["image_id", "label"]

print(submission.head())
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with rows:", len(submission))

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
DataLossError                             Traceback (most recent call last)
/tmp/ipykernel_11/733170345.py in <cell line: 0>()
     20 
     21 idx = 0
---> 22 for xb in test_ds_pred:
     23     pb = _ensemble_predict_step(xb)
     24     pred_b = tf.argmax(pb, axis=1, output_type=tf.int64).numpy()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in __next__(self)
    824   def __next__(self):
    825     try:
--> 826       return self._next_internal()
    827     except errors.OutOfRangeError:
    828       raise StopIteration

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in _next_internal(self)
    774     # to communicate that there is no more data to iterate over.
    775     with context.execution_mode(context.SYNC):
--> 776       ret = gen_dataset_ops.iterator_get_next(
    777           self._iterator_resource,
    778           output_types=self._flat_output_types,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_dataset_ops.py in iterator_get_next(iterator, output_types, output_shapes, name)
   3084       return _result
   3085     except _core._NotOkStatusException as e:
-> 3086       _ops.raise_from_not_ok_status(e, name)
   3087     except _core._FallbackException:
   3088       pass

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

DataLossError: {{function_node __wrapped__IteratorGetNext_output_types_1_device_/job:localhost/replica:0/task:0/device:CPU:0}} inflate() failed with error -3: incorrect header check [Op:IteratorGetNext] name:
