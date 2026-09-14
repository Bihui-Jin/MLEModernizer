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

0.6660622544575401

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
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")

TRAIN_TFREC_DIR = os.path.join(DATA_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(DATA_DIR, "test_tfrecords")

assert os.path.exists(TRAIN_CSV), f"Missing train.csv at {TRAIN_CSV}"
assert os.path.exists(TRAIN_IMG_DIR), f"Missing train_images at {TRAIN_IMG_DIR}"
assert os.path.exists(TEST_IMG_DIR), f"Missing test_images at {TEST_IMG_DIR}"
assert os.path.exists(SAMPLE_SUB), f"Missing sample_submission.csv at {SAMPLE_SUB}"
assert os.path.exists(TRAIN_TFREC_DIR), f"Missing train_tfrecords at {TRAIN_TFREC_DIR}"
assert os.path.exists(TEST_TFREC_DIR), f"Missing test_tfrecords at {TEST_TFREC_DIR}"

print("TF version:", tf.__version__)
print("Data dir:", DATA_DIR)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
IMG_SIZE = (300, 300)  # kept consistent with original target_size
BATCH_SIZE = 32
EPOCHS = 3  # keep identical fixed training budget

train_df = pd.read_csv(TRAIN_CSV)
train_df["path"] = TRAIN_IMG_DIR + "/" + train_df["image_id"].astype(str)
train_df["label"] = train_df["label"].astype(np.int32)

idx = np.arange(len(train_df))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
val_size = int(0.1 * len(idx))
val_idx = idx[:val_size]
trn_idx = idx[val_size:]

df_train = train_df.iloc[trn_idx].reset_index(drop=True)
df_val = train_df.iloc[val_idx].reset_index(drop=True)

AUTO = tf.data.AUTOTUNE
IMG_H, IMG_W = IMG_SIZE


@tf.function
def _decode_resize_rescale(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, [IMG_H, IMG_W], method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0
    return img


def _tfrecord_feature_description(is_train=True):
    if is_train:
        return {
            "image": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64),
        }
    else:
        return {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_id": tf.io.FixedLenFeature([], tf.string),
        }


@tf.function
def _decode_resize_rescale_from_bytes(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, [IMG_H, IMG_W], method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0
    return img


@tf.function
def _parse_train_example(example_proto):
    ex = tf.io.parse_single_example(
        example_proto, _tfrecord_feature_description(is_train=True)
    )
    img = _decode_resize_rescale_from_bytes(ex["image"])
    label = tf.cast(ex["label"], tf.int32)
    return img, label


@tf.function
def _parse_test_example(example_proto):
    ex = tf.io.parse_single_example(
        example_proto, _tfrecord_feature_description(is_train=False)
    )
    img = _decode_resize_rescale_from_bytes(ex["image"])
    image_id = ex["image_id"]
    return img, image_id


def _stateless_seed_from_index(i):
    i = tf.cast(i, tf.int32)
    return tf.stack([tf.constant(SEED, tf.int32), i], axis=0)


@tf.function
def _random_affine_equivalent(img, seed2):
    img = tf.image.stateless_random_flip_left_right(img, seed=seed2)

    max_dx = tf.cast(tf.round(0.05 * tf.cast(IMG_W, tf.float32)), tf.int32)
    max_dy = tf.cast(tf.round(0.05 * tf.cast(IMG_H, tf.float32)), tf.int32)
    dx = tf.random.stateless_uniform(
        [],
        seed=seed2 + tf.constant([1, 0], tf.int32),
        minval=-max_dx,
        maxval=max_dx + 1,
        dtype=tf.int32,
    )
    dy = tf.random.stateless_uniform(
        [],
        seed=seed2 + tf.constant([2, 0], tf.int32),
        minval=-max_dy,
        maxval=max_dy + 1,
        dtype=tf.int32,
    )
    img = tf.roll(img, shift=[dy, dx], axis=[0, 1])

    angle = tf.random.stateless_uniform(
        [],
        seed=seed2 + tf.constant([3, 0], tf.int32),
        minval=-15.0 * np.pi / 180.0,
        maxval=15.0 * np.pi / 180.0,
        dtype=tf.float32,
    )
    cos_a = tf.math.cos(angle)
    sin_a = tf.math.sin(angle)

    scale = tf.random.stateless_uniform(
        [],
        seed=seed2 + tf.constant([4, 0], tf.int32),
        minval=0.9,
        maxval=1.1,
        dtype=tf.float32,
    )

    cx = (tf.cast(IMG_W, tf.float32) - 1.0) / 2.0
    cy = (tf.cast(IMG_H, tf.float32) - 1.0) / 2.0

    a0 = scale * cos_a
    a1 = -scale * sin_a
    b0 = scale * sin_a
    b1 = scale * cos_a

    a2 = cx - a0 * cx - a1 * cy
    b2 = cy - b0 * cx - b1 * cy
    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0], axis=0)

    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(img, axis=0),
        transforms=tf.expand_dims(transform, axis=0),
        output_shape=tf.constant([IMG_H, IMG_W], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )
    img = tf.squeeze(img, axis=0)
    return img


@tf.function
def _train_map_enum(i, data):
    img, label = data
    seed2 = _stateless_seed_from_index(i)
    img = _random_affine_equivalent(img, seed2)
    return img, label


@tf.function
def _val_map(img, label):
    return img, label


_TFDATA_OPTIONS = tf.data.Options()
try:
    _TFDATA_OPTIONS.experimental_optimization.apply_default_optimizations = True
except Exception:
    pass
try:
    _TFDATA_OPTIONS.experimental_deterministic = True
except Exception:
    pass


def _list_train_tfrecords():
    files = sorted(glob.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec")))
    if not files:
        raise FileNotFoundError(f"No tfrec files found in {TRAIN_TFREC_DIR}")
    return files


def _list_test_tfrecords():
    files = sorted(glob.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec")))
    if not files:
        raise FileNotFoundError(f"No tfrec files found in {TEST_TFREC_DIR}")
    return files


def make_train_ds_from_tfrecord(batch_size):
    files = _list_train_tfrecords()
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO)
    ds = ds.with_options(_TFDATA_OPTIONS)

    ds = ds.map(_parse_train_example, num_parallel_calls=AUTO, deterministic=True)

    shuffle_buf = min(len(df_train), 4096)
    ds = ds.shuffle(buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.enumerate()
    ds = ds.map(_train_map_enum, num_parallel_calls=AUTO, deterministic=True)

    ds = ds.repeat()
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(AUTO)
    return ds


def make_val_ds_from_tfrecord(batch_size):
    files = _list_train_tfrecords()
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO)
    ds = ds.with_options(_TFDATA_OPTIONS)

    ds = ds.map(_parse_train_example, num_parallel_calls=AUTO, deterministic=True)

    val_mask = np.zeros(len(train_df), dtype=np.bool_)
    val_mask[val_idx] = True
    val_mask_tf = tf.constant(val_mask)

    ds = ds.enumerate()

    def _is_val(i, data):
        return tf.gather(val_mask_tf, tf.cast(i, tf.int32))

    def _drop_index(i, data):
        img, label = data
        return img, label

    ds = ds.filter(_is_val).map(
        _drop_index, num_parallel_calls=AUTO, deterministic=True
    )

    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTO)
    return ds


train_ds = make_train_ds_from_tfrecord(BATCH_SIZE)
val_ds = make_val_ds_from_tfrecord(BATCH_SIZE)

inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base = EfficientNetB3(include_top=False, weights="imagenet", input_tensor=inputs)
x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.2, seed=SEED)(x)
outputs = Dense(5, activation="softmax")(x)
my_model = Model(inputs=inputs, outputs=outputs)

base.trainable = False

my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=32,
)

steps_per_epoch = max(1, len(df_train) // BATCH_SIZE)
val_steps = max(1, len(df_val) // BATCH_SIZE)

my_model.fit(
    train_ds,
    epochs=1,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=val_steps,
    verbose=1,
)

base.trainable = True
my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=32,
)

my_model.fit(
    train_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=val_steps,
    verbose=1,
)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/4091995309.py in <cell line: 0>()
    257 val_steps = max(1, len(df_val) // BATCH_SIZE)
    258 
--> 259 my_model.fit(
    260     train_ds,
    261     epochs=1,

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
Error in user-defined function passed to ParallelMapDatasetV2:3 transformation with iterator: Iterator::Root::Prefetch::BatchV2::ForeverRepeat[0]::ParallelMapV2::Zip[1]::Shuffle::ParallelMapV2: Feature: label (data type: int64) is required but could not be found.
	 [[{{node ParseSingleExample/ParseExample/ParseExampleV2}}]]
	 [[IteratorGetNextAsOptional]] [Op:__inference_multi_step_on_iterator_27527]

## === cell 2
sample = pd.read_csv(SAMPLE_SUB, usecols=["image_id"])
test_image_ids = sample["image_id"].to_numpy(dtype=object)


def make_test_ds_from_tfrecord(batch_size=128):
    files = _list_test_tfrecords()
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO)
    ds = ds.with_options(_TFDATA_OPTIONS)
    ds = ds.map(_parse_test_example, num_parallel_calls=AUTO, deterministic=True)
    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTO)
    return ds


test_ds = make_test_ds_from_tfrecord(batch_size=128)

test_ids_stream = []
for _, ids in test_ds:
    test_ids_stream.append(ids.numpy())
test_ids_stream = np.concatenate(test_ids_stream).astype("S")  # bytes

pred_test = my_model.predict(
    test_ds.map(lambda x, y: x, num_parallel_calls=AUTO), verbose=1
)
pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

id_to_label = {k.decode("utf-8"): v for k, v in zip(test_ids_stream, pred_test_labels)}
pred_ordered = np.fromiter(
    (id_to_label[i] for i in test_image_ids), dtype=np.int64, count=len(test_image_ids)
).astype(int)

final_csv = pd.DataFrame({"image_id": test_image_ids, "label": pred_ordered})
final_csv.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", final_csv.shape)
print(final_csv.head())




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/3658925039.py in <cell line: 0>()
     21 # Collect image_ids in the same order as the TFRecord stream (deterministic due to sorted shards + determinism)
     22 test_ids_stream = []
---> 23 for _, ids in test_ds:
     24     test_ids_stream.append(ids.numpy())
     25 test_ids_stream = np.concatenate(test_ids_stream).astype("S")  # bytes

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

InvalidArgumentError: {{function_node __wrapped__IteratorGetNext_output_types_2_device_/job:localhost/replica:0/task:0/device:CPU:0}} Error in user-defined function passed to ParallelMapDatasetV2:34 transformation with iterator: Iterator::Root::Prefetch::BatchV2::MemoryCacheImpl::ParallelMapV2: Feature: image_id (data type: string) is required but could not be found.
	 [[{{function_node __inference__parse_test_example_27561}}{{node ParseSingleExample/ParseExample/ParseExampleV2}}]] [Op:IteratorGetNext] name: 

## === cell 3
final_csv.head()

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1842027079.py in <cell line: 0>()
----> 1 final_csv.head()

NameError: name 'final_csv' is not defined
