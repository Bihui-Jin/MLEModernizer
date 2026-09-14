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

0.7334542157751587

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.12182) has done: 'I fix the import/runtime crash by removing the incompatible standalone `keras` import and consistently using `tf.keras`, which avoids the protobuf `MessageFactory.GetPrototype` error and restores `ImageDataGenerator`. Then I fix the missing external model file issue by replacing the unavailable `../input/resnet-model/model.json/.h5` load with a small, standard `tf.keras.applications.ResNet50` inference model (no training), so the notebook runs end-to-end. Finally, I ensure the submission is aligned exactly to `sample_submission.csv` ordering and writes a valid `submission.csv` with the required `image_id,label` columns.'
- What this solution (achieved 0.13042) has done: 'I fix the TensorFlow/protobuf crash (`MessageFactory` missing `GetPrototype`) by removing the unused `model_from_json` import (it triggers the incompatible protobuf path in this environment) and keeping everything on `tf.keras`. Then I fix a logic issue that makes the score extremely low: the current model uses an ImageNet head with 5 random-output classes, so predictions are essentially meaningless; instead, I keep the same ResNet50 backbone but train only the final Dense(5) head on `train.csv` images (backbone frozen) and then predict test labels. Finally, I ensure the submission is aligned exactly to `sample_submission.csv` order and always writes a valid `submission.csv` with `image_id,label`.'

# 9. Code solution

## === cell 0
import os, glob, json

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["KERAS_BACKEND"] = "tensorflow"

os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

INPUT_DIR = "../input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
SAMPLE_SUB_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")

IMG_SIZE = 224
SIZE = (IMG_SIZE, IMG_SIZE)
BATCH_SIZE = 32
NUM_CLASSES = 5

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

print("TensorFlow:", tf.__version__)
print("Train CSV exists:", os.path.exists(TRAIN_CSV))
print("Sample submission exists:", os.path.exists(SAMPLE_SUB_CSV))
print(
    "Train images dir exists:", os.path.isdir(os.path.join(INPUT_DIR, "train_images"))
)
print("Test images dir exists:", os.path.isdir(os.path.join(INPUT_DIR, "test_images")))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
df_train = pd.read_csv(TRAIN_CSV)
df_train["label"] = df_train["label"].astype(int)

label_int = df_train["label"].to_numpy()

rng = np.random.RandomState(SEED)
idx = np.arange(len(df_train))
rng.shuffle(idx)

val_size = int(round(0.1 * len(idx)))
val_idx = idx[:val_size]
train_idx = idx[val_size:]

train_image_ids = df_train["image_id"].iloc[train_idx].to_numpy()
train_labels = label_int[train_idx]
val_image_ids = df_train["image_id"].iloc[val_idx].to_numpy()
val_labels = label_int[val_idx]

print("Split sizes:", len(train_image_ids), len(val_image_ids))

TRAIN_TFREC_GLOB = os.path.join(INPUT_DIR, "train_tfrecords", "*.tfrec")
TEST_TFREC_GLOB = os.path.join(INPUT_DIR, "test_tfrecords", "*.tfrec")
train_tfrecord_files = sorted(glob.glob(TRAIN_TFREC_GLOB))
test_tfrecord_files = sorted(glob.glob(TEST_TFREC_GLOB))

print("Num train tfrecords:", len(train_tfrecord_files))
print("Num test tfrecords:", len(test_tfrecord_files))
assert len(train_tfrecord_files) > 0, "No train TFRecords found."
assert len(test_tfrecord_files) > 0, "No test TFRecords found."

AUTOTUNE = tf.data.AUTOTUNE

opts = tf.data.Options()
opts.experimental_deterministic = True
try:
    opts.threading.private_threadpool_size = 0
    opts.threading.max_intra_op_parallelism = 0
except Exception:
    pass

train_id_to_label = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(train_image_ids, dtype=tf.string),
        values=tf.constant(train_labels, dtype=tf.int64),
    ),
    default_value=tf.constant(-1, dtype=tf.int64),
)
val_id_to_label = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(val_image_ids, dtype=tf.string),
        values=tf.constant(val_labels, dtype=tf.int64),
    ),
    default_value=tf.constant(-1, dtype=tf.int64),
)

feature_description = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_id": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
}


@tf.function
def _decode_resize_preprocess_from_tfrecord(example_proto):
    ex = tf.io.parse_single_example(example_proto, feature_description)
    img = tf.io.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(
        img,
        [IMG_SIZE, IMG_SIZE],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    label = ex["label"]
    label_oh = tf.one_hot(label, NUM_CLASSES, dtype=tf.float32)
    return img, label_oh, ex["image_id"]


@tf.function
def _decode_resize_preprocess_no_label_from_tfrecord(example_proto):
    ex = tf.io.parse_single_example(
        example_proto,
        {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_id": tf.io.FixedLenFeature([], tf.string),
        },
    )
    img = tf.io.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(
        img,
        [IMG_SIZE, IMG_SIZE],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img, ex["image_id"]


def _make_train_val_datasets_from_tfrecords(files):
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTOTUNE).with_options(opts)
    ds = ds.map(_decode_resize_preprocess_from_tfrecord, num_parallel_calls=AUTOTUNE)

    def is_train(img, label_oh, image_id):
        return train_id_to_label.lookup(image_id) >= 0

    def is_val(img, label_oh, image_id):
        return val_id_to_label.lookup(image_id) >= 0

    train_ds_local = ds.filter(is_train).map(
        lambda img, label_oh, image_id: (img, label_oh), num_parallel_calls=AUTOTUNE
    )
    val_ds_local = ds.filter(is_val).map(
        lambda img, label_oh, image_id: (img, label_oh), num_parallel_calls=AUTOTUNE
    )

    SHUFFLE_BUFFER = min(len(train_image_ids), 8192)
    train_ds_local = train_ds_local.shuffle(
        buffer_size=SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=True
    )

    train_ds_local = train_ds_local.batch(BATCH_SIZE, drop_remainder=False)
    train_ds_local = train_ds_local.prefetch(AUTOTUNE)

    val_ds_local = val_ds_local.cache()
    val_ds_local = val_ds_local.batch(BATCH_SIZE, drop_remainder=False)
    val_ds_local = val_ds_local.prefetch(AUTOTUNE)
    return train_ds_local, val_ds_local


train_ds, val_ds = _make_train_val_datasets_from_tfrecords(train_tfrecord_files)

class_indices = {str(i): i for i in range(NUM_CLASSES)}
print("class_indices:", class_indices)
assert len(class_indices) == NUM_CLASSES, "Expected 5 classes from labels 0-4."



## === cell 2
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)

test_raw = tf.data.TFRecordDataset(
    test_tfrecord_files, num_parallel_reads=AUTOTUNE
).with_options(opts)
test_raw = test_raw.map(
    _decode_resize_preprocess_no_label_from_tfrecord, num_parallel_calls=AUTOTUNE
)

test_raw = test_raw.cache()

test_imgs_ds = test_raw.map(lambda img, image_id: img, num_parallel_calls=AUTOTUNE)
test_imgs_ds = test_imgs_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

test_image_ids = []
for _, image_id_batch in test_raw.batch(1024):
    test_image_ids.extend(image_id_batch.numpy().astype(str).tolist())
test_image_ids = np.array(test_image_ids, dtype=object)

print("Test TFRecord examples:", len(test_image_ids))
print("Sample submission rows:", len(sample_sub))
assert len(test_image_ids) == len(
    sample_sub
), "Test TFRecords count must match sample_submission."



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/2373900036.py in <cell line: 0>()
     19 # Materialize test image_ids in the same iteration order as predictions will be produced.
     20 test_image_ids = []
---> 21 for _, image_id_batch in test_raw.batch(1024):
     22     test_image_ids.extend(image_id_batch.numpy().astype(str).tolist())
     23 test_image_ids = np.array(test_image_ids, dtype=object)

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

InvalidArgumentError: {{function_node __wrapped__IteratorGetNext_output_types_2_device_/job:localhost/replica:0/task:0/device:CPU:0}} Error in user-defined function passed to ParallelMapDatasetV2:17 transformation with iterator: Iterator::Root::Prefetch::BatchV2::MemoryCacheImpl::ParallelMapV2: Feature: image_id (data type: string) is required but could not be found.
	 [[{{function_node __inference__decode_resize_preprocess_no_label_from_tfrecord_185}}{{node ParseSingleExample/ParseExample/ParseExampleV2}}]] [Op:IteratorGetNext] name: 

## === cell 3
from tensorflow.keras import layers, models

inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")

base = tf.keras.applications.ResNet50(
    include_top=False, weights="imagenet", input_tensor=inp, pooling="avg"
)
base.trainable = False

out = layers.Dense(NUM_CLASSES, activation="softmax", name="cassava_head")(base.output)
model = models.Model(inputs=inp, outputs=out)

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)

model.summary()



## === cell 4
EPOCHS = 3  # keep as-is to preserve runtime/approach

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
)

pred_test = model.predict(
    test_imgs_ds,
    verbose=1,
)
pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

order = {k: i for i, k in enumerate(test_image_ids.tolist())}
idx_map = np.fromiter(
    (order[i] for i in sample_sub["image_id"].values),
    dtype=np.int64,
    count=len(sample_sub),
)
pred_test_labels_ordered = pred_test_labels[idx_map]

sub = pd.DataFrame(
    {"image_id": sample_sub["image_id"].values, "label": pred_test_labels_ordered}
)

assert list(sub.columns) == ["image_id", "label"]
assert len(sub) == len(sample_sub)
assert sub["label"].between(0, NUM_CLASSES - 1).all()

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/4258193750.py in <cell line: 0>()
      1 EPOCHS = 3  # keep as-is to preserve runtime/approach
      2 
----> 3 history = model.fit(
      4     train_ds,
      5     validation_data=val_ds,

/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
     68             # To get the full stack trace, call:
     69             # `tf.debugging.disable_traceback_filtering()`
---> 70             raise e.with_traceback(filtered_tb) from None
     71         finally:
     72             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
     57       e.message += " name: " + name
     58     raise core._status_to_exception(e) from None
---> 59   except TypeError as e:
     60     keras_symbolic_tensors = [x for x in inputs if _is_keras_symbolic_tensor(x)]
     61     if keras_symbolic_tensors:

InvalidArgumentError: Graph execution error:

Detected at node ParseSingleExample/ParseExample/ParseExampleV2 defined at (most recent call last):
<stack traces unavailable>
Error in user-defined function passed to ParallelMapDatasetV2:3 transformation with iterator: Iterator::Root::Prefetch::BatchV2::Shuffle::Map::Filter::Map: Feature: image_id (data type: string) is required but could not be found.
	 [[{{node ParseSingleExample/ParseExample/ParseExampleV2}}]]
	 [[IteratorGetNext]] [Op:__inference_train_function_9111]
