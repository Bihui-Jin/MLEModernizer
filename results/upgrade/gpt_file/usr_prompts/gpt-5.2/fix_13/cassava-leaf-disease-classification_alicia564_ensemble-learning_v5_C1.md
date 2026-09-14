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

0.7733454215775158

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.12967) has done: 'The timeout is almost certainly dominated by JPEG decoding + resizing the full training set for every epoch (even with `.cache()` this spills to disk and still costs a lot) and by recomputing the `shuffle(buffer_size=len(train))` each epoch. I keep the exact same model, loss, optimizer, epochs, and dataset semantics, but switch training/validation input to the provided TFRecords (same images/labels) to avoid per-file JPEG overhead and leverage sequential reads. I also make caching explicitly file-backed (deterministic and stable) and keep deterministic options, while preserving the stratified split and label mapping exactly. Prediction remains on JPEGs (test TFRecords don’t include filenames), but input pipeline is tuned with parallelism and prefetch for speed.'
- What this solution (achieved 0.12967) has done: 'I first fix the TensorFlow import crash in cell 1 by setting the pure-Python protobuf implementation before importing TensorFlow (this resolves the `MessageFactory.GetPrototype` issue seen in some Kaggle images with newer Python/protobuf). Next, I fix the TFRecord parsing bug by using the correct feature key for the filename (the Cassava TFRecords store `image_name`, not `image_id`), and I still map labels via your existing CSV-driven `id_to_label` to preserve the original label encoding semantics. Finally, I ensure the TFRecord `image_name` matches your CSV `image_id` format by appending “.jpg” when needed, so the train/valid filtering and label lookup work correctly and training can run end-to-end to produce `submission.csv`.'

# 9. Code solution

## === cell 0
import os

for k in [
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION",
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION",
    "PROTOCOL_BUFFERS_PYTHON_DISABLE_C_DESCRIPTORS",
]:
    os.environ.pop(k, None)

import re
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

print("TF version:", tf.__version__)

SEED = 42
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

label_to_disease = pd.read_json(
    "/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json",
    typ="series",
)
train_csv = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/train.csv")

train_csv["disease"] = train_csv["label"].map(label_to_disease)
train_csv["path"] = (
    "/kaggle/input/cassava-leaf-disease-classification/train_images/"
    + train_csv["image_id"]
)

train_csv["disease"] = train_csv["disease"].astype(str)
train_csv["label"] = train_csv["label"].astype(str)

disease_categories = np.sort(train_csv["disease"].unique())
disease_to_int = {d: i for i, d in enumerate(disease_categories)}
train_csv["label_encoded"] = train_csv["disease"].map(disease_to_int).astype(np.int64)

rng = np.random.default_rng(SEED)
valid_frac = 0.2
valid_indices = []
for lbl, idx in train_csv.groupby("label", sort=True).indices.items():
    idx = np.asarray(idx, dtype=np.int64)
    rng.shuffle(idx)
    n_valid = int(np.round(len(idx) * valid_frac))
    valid_indices.append(idx[:n_valid])
valid_indices = np.concatenate(valid_indices)
valid_mask = np.zeros(len(train_csv), dtype=bool)
valid_mask[valid_indices] = True

valid = train_csv.loc[valid_mask].reset_index(drop=True)
train = train_csv.loc[~valid_mask].reset_index(drop=True)

id_to_label = dict(
    zip(
        train_csv["image_id"].values.tolist(),
        train_csv["label_encoded"].values.tolist(),
    )
)
train_ids = set(train["image_id"].values.tolist())
valid_ids = set(valid["image_id"].values.tolist())




## === cell 2
AUTOTUNE = tf.data.AUTOTUNE

TRAIN_TFREC_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_tfrecords"
train_tfrecords = sorted(tf.io.gfile.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec")))

options = tf.data.Options()
options.experimental_deterministic = True
try:
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.map_and_batch_fusion = True
    options.experimental_optimization.autotune_buffers = True
except Exception:
    pass

_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


def _ensure_jpg_py_arr(arr):
    arr = np.asarray(arr, dtype=object)
    out = np.empty(arr.shape, dtype=object)
    for i, v in enumerate(arr):
        s = "" if v is None else str(v).strip()
        sl = s.lower()
        if sl.endswith(".jpg"):
            out[i] = s
        else:
            out[i] = s + ".jpg"
    return out


train_keys_np = _ensure_jpg_py_arr(train["image_id"].to_numpy())
train_vals_np = train["label_encoded"].to_numpy(dtype=np.int64)
valid_keys_np = _ensure_jpg_py_arr(valid["image_id"].to_numpy())
valid_vals_np = valid["label_encoded"].to_numpy(dtype=np.int64)

all_keys_np = np.concatenate([train_keys_np, valid_keys_np], axis=0)
all_labels_np = np.concatenate([train_vals_np, valid_vals_np], axis=0)
all_splits_np = np.concatenate(
    [
        np.ones_like(train_vals_np, dtype=np.int32),
        np.full_like(valid_vals_np, 2, dtype=np.int32),
    ],
    axis=0,
)

all_keys = tf.constant(all_keys_np, dtype=tf.string)
all_vals = tf.stack(
    [
        tf.constant(all_labels_np, dtype=tf.int64),
        tf.constant(all_splits_np, dtype=tf.int32),
    ],
    axis=1,
)

labels_splits_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(keys=all_keys, values=all_vals),
    default_value=tf.constant([-1, 0], dtype=tf.int64),  # label=-1, split=0
)


@tf.function
def _ensure_jpg_tf(image_id: tf.Tensor) -> tf.Tensor:
    image_id = tf.strings.strip(image_id)
    lower = tf.strings.lower(image_id)
    has_jpg = tf.strings.length(lower) >= 4
    has_jpg = tf.logical_and(has_jpg, tf.equal(tf.strings.substr(lower, -4, 4), ".jpg"))
    return tf.where(has_jpg, image_id, tf.strings.join([image_id, ".jpg"]))


@tf.function
def _decode_and_preprocess(image_bytes: tf.Tensor) -> tf.Tensor:
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)
    return image


@tf.function
def _parse_tfrec(serialized):
    ex = tf.io.parse_single_example(serialized, _FEATURES)
    image_id = _ensure_jpg_tf(ex["image_name"])
    label_split = labels_splits_table.lookup(image_id)
    label = tf.cast(label_split[0], tf.int64)
    split = tf.cast(label_split[1], tf.int32)
    image = _decode_and_preprocess(ex["image"])
    return image, label, split


files_ds = tf.data.Dataset.from_tensor_slices(train_tfrecords).with_options(options)
raw_ds = files_ds.interleave(
    lambda fn: tf.data.TFRecordDataset(fn, num_parallel_reads=AUTOTUNE),
    cycle_length=min(len(train_tfrecords), 16),
    num_parallel_calls=AUTOTUNE,
    deterministic=True,
)

parsed = raw_ds.map(_parse_tfrec, num_parallel_calls=AUTOTUNE, deterministic=True)
parsed = parsed.filter(lambda image, label, split: tf.greater_equal(label, 0))

train_parsed = parsed.filter(lambda image, label, split: tf.equal(split, 1))
train_parsed = train_parsed.map(
    lambda image, label, split: (image, label),
    num_parallel_calls=AUTOTUNE,
    deterministic=True,
)

valid_parsed = parsed.filter(lambda image, label, split: tf.equal(split, 2))
valid_parsed = valid_parsed.map(
    lambda image, label, split: (image, label),
    num_parallel_calls=AUTOTUNE,
    deterministic=True,
)

train_cache_path = "/kaggle/working/train_cache.tfcache"
train_parsed = train_parsed.cache(train_cache_path)
valid_parsed = valid_parsed.cache()

_shuffle_buf = int(min(len(train), 8192))
BATCH_SIZE = 32

train_ds = (
    train_parsed.shuffle(
        buffer_size=_shuffle_buf, seed=SEED, reshuffle_each_iteration=True
    )
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

valid_ds = valid_parsed.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

steps_per_epoch = int((len(train) + BATCH_SIZE - 1) // BATCH_SIZE)
validation_steps = int((len(valid) + BATCH_SIZE - 1) // BATCH_SIZE)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/2246164409.py in <cell line: 0>()
     54 
     55 all_keys = tf.constant(all_keys_np, dtype=tf.string)
---> 56 all_vals = tf.stack(
     57     [
     58         tf.constant(all_labels_np, dtype=tf.int64),

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

InvalidArgumentError: cannot compute Pack as input #1(zero-based) was expected to be a int64 tensor but is a int32 tensor [Op:Pack] name: stack

## === cell 3
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback


class EarlyStoppingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if self.model.stop_training:
            print(f"Early stopping triggered at epoch {epoch + 1}.")


early_stopping = EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", patience=2, factor=0.5, min_lr=1e-6, verbose=1
)




## === cell 4
NUM_CLASSES = 5

base_model = EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3)
)
x = GlobalAveragePooling2D()(base_model.output)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=5,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[early_stopping, learning_rate_reduction],
    verbose=1,
)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3829149414.py in <cell line: 0>()
     15 
     16 history = model.fit(
---> 17     train_ds,
     18     validation_data=valid_ds,
     19     epochs=5,

NameError: name 'train_ds' is not defined

## === cell 5
import pandas as pd
import numpy as np
import os

sample_sub = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

TEST_TFREC_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_tfrecords"
test_tfrecords = sorted(tf.io.gfile.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec")))

_FEATURES_TEST = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


@tf.function
def _parse_test_tfrec(serialized):
    ex = tf.io.parse_single_example(serialized, _FEATURES_TEST)
    image_id = _ensure_jpg_tf(ex["image_name"])
    image = _decode_and_preprocess(ex["image"])
    return image_id, image


options = tf.data.Options()
options.experimental_deterministic = True
try:
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.map_and_batch_fusion = True
    options.experimental_optimization.autotune_buffers = True
except Exception:
    pass

test_files_ds = tf.data.Dataset.from_tensor_slices(test_tfrecords).with_options(options)
test_raw_ds = test_files_ds.interleave(
    lambda fn: tf.data.TFRecordDataset(fn, num_parallel_reads=AUTOTUNE),
    cycle_length=min(len(test_tfrecords), 16),
    num_parallel_calls=AUTOTUNE,
    deterministic=True,
)
test_parsed = test_raw_ds.map(
    _parse_test_tfrec, num_parallel_calls=AUTOTUNE, deterministic=True
)


@tf.function
def _predict_batch(batch_ids, batch_imgs):
    probs = model(batch_imgs, training=False)
    preds = tf.argmax(probs, axis=1, output_type=tf.int64)
    return batch_ids, preds


PRED_BATCH_SIZE = 128

test_pred_ds = (
    test_parsed.batch(PRED_BATCH_SIZE, drop_remainder=False)
    .map(_predict_batch, num_parallel_calls=AUTOTUNE, deterministic=True)
    .prefetch(AUTOTUNE)
)

test_ids_list = []
test_preds_list = []
for batch_ids, batch_preds in test_pred_ds:
    test_ids_list.append(batch_ids.numpy())
    test_preds_list.append(batch_preds.numpy())

test_ids = np.concatenate(test_ids_list, axis=0).astype("U")
test_preds = np.concatenate(test_preds_list, axis=0).astype(np.int64)

pred_map = pd.Series(test_preds, index=pd.Index(test_ids, name="image_id"))
submission_df = sample_sub.copy()
submission_df["label"] = pred_map.reindex(submission_df["image_id"]).values.astype(
    np.int64
)

submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)
print("Submission file created:", submission_path)
print(submission_df.head())
print("Rows:", len(submission_df), "Cols:", submission_df.shape[1])
print("Label value counts:\n", submission_df["label"].value_counts().sort_index())

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3853117050.py in <cell line: 0>()
     41     deterministic=True,
     42 )
---> 43 test_parsed = test_raw_ds.map(
     44     _parse_test_tfrec, num_parallel_calls=AUTOTUNE, deterministic=True
     45 )

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

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/tmp/__autograph_generated_file7frorfwo.py in tf___parse_test_tfrec(serialized)
      9                 retval_ = ag__.UndefinedReturnValue()
     10                 ex = ag__.converted_call(ag__.ld(tf).io.parse_single_example, (ag__.ld(serialized), ag__.ld(_FEATURES_TEST)), None, fscope)
---> 11                 image_id = ag__.converted_call(ag__.ld(_ensure_jpg_tf), (ag__.ld(ex)['image_name'],), None, fscope)
     12                 image = ag__.converted_call(ag__.ld(_decode_and_preprocess), (ag__.ld(ex)['image'],), None, fscope)
     13                 try:

NameError: in user code:

    File "/tmp/ipykernel_11/3853117050.py", line 21, in _parse_test_tfrec  *
        image_id = _ensure_jpg_tf(ex["image_name"])

    NameError: name '_ensure_jpg_tf' is not defined
