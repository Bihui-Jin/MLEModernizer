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

0.8785131459655485

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

import random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

TRAIN_IMG_LOC = "../input/cassava-leaf-disease-classification/train_images"
TEST_IMG_LOC = "../input/cassava-leaf-disease-classification/test_images"
TRAIN_CSV = "../input/cassava-leaf-disease-classification/train.csv"
SAMPLE_CSV = "../input/cassava-leaf-disease-classification/sample_submission.csv"
MODELS_WEIGHTS = "../input/cassavaeffentb7models/content/Models"

TRAIN_TFREC_GLOB = (
    "../input/cassava-leaf-disease-classification/train_tfrecords/*.tfrec"
)
TEST_TFREC_GLOB = "../input/cassava-leaf-disease-classification/test_tfrecords/*.tfrec"

print("TensorFlow:", tf.__version__)
print("ALL Modules are successfully loaded")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_df = pd.read_csv(TRAIN_CSV)
ss = pd.read_csv(SAMPLE_CSV)

assert {"image_id", "label"}.issubset(train_df.columns)
assert {"image_id", "label"}.issubset(ss.columns)

train_df["filepath"] = (TRAIN_IMG_LOC.rstrip("/") + "/" + train_df["image_id"]).astype(
    str
)

missing = [p for p in train_df["filepath"].head(10).tolist() if not os.path.exists(p)]
if missing:
    raise FileNotFoundError(f"Some train image files not found (sample): {missing[:3]}")

NUM_CLASSES = int(train_df["label"].nunique())
print("Train rows:", len(train_df), " Num classes:", NUM_CLASSES)
print("Test rows:", len(ss))



## === cell 2
IMG_SIZE = (300, 300)
BATCH_SIZE = 32
VAL_SPLIT = 0.1
AUTOTUNE = tf.data.AUTOTUNE
SHUFFLE_BUFFER = 8192

DATA_OPTIONS = tf.data.Options()
DATA_OPTIONS.experimental_deterministic = True
DATA_OPTIONS.autotune.enabled = True


def _decode_jpeg_bytes(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    return img


def decode_image(path):
    img = tf.io.read_file(path)
    return _decode_jpeg_bytes(img)


_TFREC_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


def _parse_tfrecord(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TFREC_FEATURES)
    img = _decode_jpeg_bytes(ex["image"])
    label = tf.cast(ex["label"], tf.int32)
    image_name = ex["image_name"]
    return img, label, image_name


def _list_tfrecord_files(glob_pattern):
    files = tf.io.gfile.glob(glob_pattern)
    if not files:
        raise FileNotFoundError(f"No TFRecord files found for pattern: {glob_pattern}")
    files = sorted(files)  # deterministic shard order
    return files


def make_ds_from_paths(paths, labels=None, training=False):
    paths = tf.convert_to_tensor(paths, dtype=tf.string)
    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(paths)
        ds = ds.map(decode_image, num_parallel_calls=AUTOTUNE)
    else:
        labels = tf.convert_to_tensor(labels, dtype=tf.int32)
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(lambda p, y: (decode_image(p), y), num_parallel_calls=AUTOTUNE)

    ds = ds.with_options(DATA_OPTIONS)

    if not training:
        ds = ds.cache()

    if training:
        ds = ds.shuffle(
            buffer_size=min(int(paths.shape[0]), SHUFFLE_BUFFER),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


def make_train_val_from_tfrecords(val_image_id_set):
    files = _list_tfrecord_files(TRAIN_TFREC_GLOB)

    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTOTUNE)
    ds = ds.with_options(DATA_OPTIONS)
    ds = ds.map(_parse_tfrecord, num_parallel_calls=AUTOTUNE)

    val_ids = tf.constant(sorted(list(val_image_id_set)), dtype=tf.string)

    def _is_in_val(_img, _label, image_name):
        eq = tf.equal(image_name, val_ids)
        return tf.reduce_any(eq)

    val_ds = ds.filter(_is_in_val).map(
        lambda img, label, _name: (img, label), num_parallel_calls=AUTOTUNE
    )
    train_ds = ds.filter(
        lambda img, label, name: tf.logical_not(_is_in_val(img, label, name))
    ).map(lambda img, label, _name: (img, label), num_parallel_calls=AUTOTUNE)

    val_ds = val_ds.cache()

    train_ds = train_ds.shuffle(
        buffer_size=SHUFFLE_BUFFER,
        seed=SEED,
        reshuffle_each_iteration=True,
    )

    train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    return train_ds, val_ds


idx = np.arange(len(train_df))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)

val_size = int(round(len(idx) * VAL_SPLIT))
val_idx = idx[:val_size]
trn_idx = idx[val_size:]

val_image_ids = set(train_df.loc[val_idx, "image_id"].astype(str).tolist())

train_ds, val_ds = make_train_val_from_tfrecords(val_image_ids)

print(
    "Built datasets (TFRecords).",
    "train batches:",
    tf.data.experimental.cardinality(train_ds).numpy(),
    "val batches:",
    tf.data.experimental.cardinality(val_ds).numpy(),
)



## === cell 3
base = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=IMG_SIZE + (3,),
)
base.trainable = False  # initial training head only (fast, stable)

inputs = keras.Input(shape=IMG_SIZE + (3,), name="image")
x = tf.keras.applications.efficientnet.preprocess_input(inputs)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

model.summary()



## === cell 4
EPOCHS_HEAD = 2
history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
    verbose=1,
)

base.trainable = True
fine_tune_at = max(0, len(base.layers) - 20)
for layer in base.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

EPOCHS_FT = 1
history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD + EPOCHS_FT,
    initial_epoch=history1.epoch[-1] + 1 if history1.epoch else 0,
    verbose=1,
)

print("Training complete.")




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/532016283.py in <cell line: 0>()
      1 EPOCHS_HEAD = 2
----> 2 history1 = model.fit(
      3     train_ds,
      4     validation_data=val_ds,
      5     epochs=EPOCHS_HEAD,

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

InvalidArgumentError: Graph execution error:

Detected at node ParseSingleExample/ParseExample/ParseExampleV2 defined at (most recent call last):
<stack traces unavailable>
Error in user-defined function passed to ParallelMapDatasetV2:3 transformation with iterator: Iterator::Root::Prefetch::BatchV2::Shuffle::ParallelMapV2::Filter::ParallelMapV2: Feature: label (data type: int64) is required but could not be found.
	 [[{{node ParseSingleExample/ParseExample/ParseExampleV2}}]]
	 [[IteratorGetNext]] [Op:__inference_multi_step_on_iterator_16652]

## === cell 5
def make_test_ds_from_tfrecords():
    files = _list_tfrecord_files(TEST_TFREC_GLOB)
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTOTUNE)
    ds = ds.with_options(DATA_OPTIONS)

    try:
        ds = ds.map(_parse_tfrecord, num_parallel_calls=AUTOTUNE)
        ds = ds.map(
            lambda img, _label, image_name: (img, image_name),
            num_parallel_calls=AUTOTUNE,
        )
    except Exception:
        raise

    ds = ds.cache()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


test_paths = (TEST_IMG_LOC.rstrip("/") + "/" + ss["image_id"]).astype(str).tolist()
missing_test = [p for p in test_paths[:20] if not os.path.exists(p)]
if missing_test:
    raise FileNotFoundError(
        f"Some test image files not found (sample): {missing_test[:3]}"
    )

use_tfrecord_test = True
try:
    test_ds_named = make_test_ds_from_tfrecords()
except Exception:
    use_tfrecord_test = False
    test_ds = make_ds_from_paths(np.array(test_paths), labels=None, training=False)

if use_tfrecord_test:
    names = []
    preds_list = []

    for batch_imgs, batch_names in test_ds_named:
        batch_probs = model(batch_imgs, training=False).numpy()
        batch_pred = np.argmax(batch_probs, axis=1).astype(int)
        names.extend([n.decode("utf-8") for n in batch_names.numpy().tolist()])
        preds_list.extend(batch_pred.tolist())

    pred_map = dict(zip(names, preds_list))
    preds = np.array(
        [pred_map[i] for i in ss["image_id"].astype(str).tolist()], dtype=int
    )
else:
    probs = model.predict(test_ds, verbose=1)
    preds = np.argmax(probs, axis=1).astype(int)

assert len(preds) == len(ss), (len(preds), len(ss))

my_submission = pd.DataFrame({"image_id": ss["image_id"].values, "label": preds})
my_submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", my_submission.shape)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/3129264975.py in <cell line: 0>()
     43     preds_list = []
     44 
---> 45     for batch_imgs, batch_names in test_ds_named:
     46         batch_probs = model(batch_imgs, training=False).numpy()
     47         batch_pred = np.argmax(batch_probs, axis=1).astype(int)

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

InvalidArgumentError: {{function_node __wrapped__IteratorGetNext_output_types_2_device_/job:localhost/replica:0/task:0/device:CPU:0}} Error in user-defined function passed to ParallelMapDatasetV2:24 transformation with iterator: Iterator::Root::Prefetch::BatchV2::MemoryCacheImpl::ParallelMapV2::ParallelMapV2: Feature: label (data type: int64) is required but could not be found.
	 [[{{node ParseSingleExample/ParseExample/ParseExampleV2}}]] [Op:IteratorGetNext] name: 

## === cell 6
my_submission.head()

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2534285131.py in <cell line: 0>()
----> 1 my_submission.head()

NameError: name 'my_submission' is not defined
