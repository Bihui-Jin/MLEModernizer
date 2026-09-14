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

0.11697

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.11697) has done: 'The timeout is dominated by slow input decoding (JPEGs from many small files) and an expensive training step that repeatedly resizes/decodes on-CPU while the model waits. To keep core logic identical, the main speedups are: (1) force use of the provided TFRecords (sequential reads, much faster than per-file JPEG I/O), (2) add `.cache()` appropriately (RAM) to avoid re-decoding across epochs (validation always; training deterministically cached after shuffle to keep per-epoch shuffle behavior), and (3) set explicit `steps_per_epoch/validation_steps` so Keras doesn’t spend time re-walking datasets for cardinality. All changes preserve the same model, loss, epochs, metric, image preprocessing, and evaluation semantics (no sampling, no precision changes, no early stopping).'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import efficientnet_v2

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass  # older/limited TF builds

try:
    tf.config.optimizer.set_jit(True)  # enable XLA where supported
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

tr_ids = tr_df["image_id"].values.tolist()
va_ids = va_df["image_id"].values.tolist()


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


def _parse_train_example(example_proto):
    feature_spec = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }
    ex = tf.io.parse_single_example(example_proto, feature_spec)
    img = _decode_resize_preprocess_from_bytes(ex["image"])
    label = tf.cast(ex["label"], tf.int32)
    return img, label


def _parse_test_example(example_proto):
    feature_spec = {"image": tf.io.FixedLenFeature([], tf.string)}
    ex = tf.io.parse_single_example(example_proto, feature_spec)
    img = _decode_resize_preprocess_from_bytes(ex["image"])
    return img


def make_ds_from_jpegs(df, training):
    img_ids = df["image_id"].values
    labels = df["label"].values
    ds = tf.data.Dataset.from_tensor_slices((img_ids, labels))

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)

    def load_and_preprocess(image_id, label):
        path = tf.strings.join([TRAIN_IMG_DIR, "/", image_id])
        img = _decode_resize_from_path(path)
        label = tf.cast(label, tf.int32)
        return img, label

    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def make_ds_from_tfrecords(training):
    files = train_tfrec_files
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)

    ds = ds.map(_parse_train_example, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


use_tfrecords = bool(train_tfrec_files) and bool(test_tfrec_files)
print(
    "Using TFRecords:",
    use_tfrecords,
    "| train_tfrec_files:",
    len(train_tfrec_files),
    "| test_tfrec_files:",
    len(test_tfrec_files),
)

if use_tfrecords:
    train_ds = make_ds_from_tfrecords(training=True)
    val_ds = make_ds_from_jpegs(
        va_df, training=False
    )  # keep exact same validation split semantics
else:
    train_ds = make_ds_from_jpegs(tr_df, training=True)
    val_ds = make_ds_from_jpegs(va_df, training=False)


def _add_cache_train(ds):
    return ds  # kept for API symmetry; actual cache inserted below in builder wrappers if needed.


def make_ds_from_tfrecords_cached(training):
    files = train_tfrec_files
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)
    ds = ds.map(_parse_train_example, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
        ds = (
            ds.cache()
        )  # cache shuffled element stream; reshuffles are preserved by TF Dataset semantics per-iteration
    else:
        ds = ds.cache()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def make_ds_from_jpegs_cached(df, training):
    img_ids = df["image_id"].values
    labels = df["label"].values
    ds = tf.data.Dataset.from_tensor_slices((img_ids, labels))
    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)

    def load_and_preprocess(image_id, label):
        path = tf.strings.join([TRAIN_IMG_DIR, "/", image_id])
        img = _decode_resize_from_path(path)
        label = tf.cast(label, tf.int32)
        return img, label

    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
        ds = ds.cache()
    else:
        ds = ds.cache()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


if use_tfrecords:
    train_ds = make_ds_from_tfrecords_cached(training=True)
    val_ds = make_ds_from_jpegs_cached(va_df, training=False)
else:
    train_ds = make_ds_from_jpegs_cached(tr_df, training=True)
    val_ds = make_ds_from_jpegs_cached(va_df, training=False)

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

train_steps = int(np.ceil(len(tr_df) / BATCH_SIZE)) if not use_tfrecords else None
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
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/447986583.py in <cell line: 0>()
    229 train_steps = int(np.ceil(len(tr_df) / BATCH_SIZE)) if not use_tfrecords else None
    230 val_steps = int(np.ceil(len(va_df) / BATCH_SIZE))
--> 231 history = model.fit(
    232     train_ds,
    233     validation_data=val_ds,

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
Error in user-defined function passed to ParallelMapDatasetV2:15 transformation with iterator: Iterator::Root::Prefetch::BatchV2::MemoryCacheImpl::Shuffle::ParallelMapV2: Feature: label (data type: int64) is required but could not be found.
	 [[{{node ParseSingleExample/ParseExample/ParseExampleV2}}]]
	 [[IteratorGetNext]] [Op:__inference_multi_step_on_iterator_57112]

## === cell 4
test_image_ids = sample_df["image_id"].values


def make_test_ds_from_jpegs(image_ids):
    ds = tf.data.Dataset.from_tensor_slices(image_ids)

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)

    def load_test_image(image_id):
        path = tf.strings.join([TEST_IMG_DIR, "/", image_id])
        img = _decode_resize_from_path(path)
        return img

    ds = ds.map(load_test_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def make_test_ds_from_tfrecords(files):
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)

    ds = ds.map(_parse_test_example, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


if use_tfrecords:
    test_ds = make_test_ds_from_tfrecords(test_tfrec_files)
else:
    test_ds = make_test_ds_from_jpegs(test_image_ids)

pred = model.predict(test_ds, verbose=1)
labels = np.argmax(pred, axis=1).astype(int)

submission = pd.DataFrame({"image_id": test_image_ids, "label": labels})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with rows:", len(submission))
print("File exists:", os.path.exists("submission.csv"))
