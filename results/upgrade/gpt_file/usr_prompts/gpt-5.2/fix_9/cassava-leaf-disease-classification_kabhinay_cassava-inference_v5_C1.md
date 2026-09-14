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

0.8260803868238138

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.13378) has done: 'I fix the environment-breaking TensorFlow import error by forcing the pure-Python protobuf implementation early (this is a common Kaggle issue causing the `MessageFactory.GetPrototype` crash). Next, I remove the dependency on a missing external SavedModel (`../input/gambler-s-loss-cassava/model`) by keeping your prediction pipeline intact but swapping in a standard Keras image classifier (EfficientNet) available in the default TF install, so the notebook can run end-to-end. I also ensure the data paths point at the actual provided dataset location and keep the submission format exactly as required. Finally, I preserve the “gambler head” semantics by outputting a 6-logit vector and using `argmax(pred[:,1:])` exactly as your existing code does.'
- What this solution (achieved 0.55942) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing protobuf’s pure-Python implementation earlier (before any TensorFlow-related import) and ensuring the environment variable `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION` is set as well, which is the common missing piece on Kaggle. Then I restore the intended core “gambler” training semantics by actually training your `gambler_like_efficientnetb0` model on `train.csv` images using your provided `loss_gambler`, so predictions aren’t essentially random (which explains the 0.13378 score). Finally, I keep the same submission logic (`argmax(pred[:,1:])`) and ensure the output `submission.csv` is written with the exact required columns and row order matching `sample_submission.csv`.'
- What this solution (achieved 0.46786) has done: 'I fix two execution blockers while keeping your model, gambler loss, and submission logic unchanged. First, the TensorFlow import crash (`MessageFactory.GetPrototype`) is resolved by pinning protobuf to the pure-Python implementation *and* forcing the compatible backend before any TF import (this is the common Kaggle workaround). Second, `flow_from_dataframe(..., class_mode="categorical")` requires string/list labels, so I cast `train_df["label"]` to `str` and explicitly set `classes=["0","1","2","3","4"]` to guarantee stable class-to-index mapping. These are correctness/stability fixes and should also improve score versus the currently failing pipeline by ensuring training actually runs as intended.'
- What this solution (achieved 0.16704) has done: 'I fix the two execution blockers that prevent training from running: (1) the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation *before* any TF import and (2) the `len(symbolic_tensor)` error inside `loss_gambler` by rewriting its loop using TensorFlow shapes/range so it works in graph mode. These are minimal changes that preserve your “gambler head” (6 logits with `argmax(pred[:,1:])`) and keep the same EfficientNetB0 backbone, training schedule, and data pipeline. Once training runs correctly, the model should learn meaningful weights and the accuracy should move up toward the target range. The script still write a correctly formatted `submission.csv` to `/kaggle/working/submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42)
np.random.seed(42)

print("TF version:", tf.__version__)
print("Keras version:", keras.__version__)

tf.config.experimental.enable_op_determinism()




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def acc_gambler(y_true, y_pred):
    y_temp = y_pred[:, 1:]
    count = tf.constant((0,))
    for i in range(len(y_true)):
        tf.autograph.experimental.set_loop_options(
            shape_invariants=[(count, tf.TensorShape([None]))]
        )
        if tf.math.argmax(y_temp[i]) == tf.math.argmax(y_true[i]):
            count = tf.math.add(count, 1)
    return float(count) / float(len(y_true))




## === cell 2
def loss_gambler(y_true, y_pred):
    """
    Bugfix: avoid iterating over a symbolic tensor (tf.range(num_classes)) in graph mode.
    Keeps same math/semantics as the original loop:
      sum_i [ (-1/n) * y_true[:,i] * log(y_temp[:,i] + f0/lamb) ]
    """
    y_temp = y_pred[:, 1:]  # (N, C)
    f0 = y_pred[:, 0:1]  # (N, 1) keep dims for broadcasting

    K = tf.keras.backend

    sum_y = K.sum(y_temp)
    sum_y2 = K.sum(tf.math.multiply(y_temp, y_temp))
    lamb = tf.math.divide(tf.math.multiply(sum_y, sum_y), sum_y2)

    n = tf.cast(tf.shape(y_true)[0], tf.float32)

    add_term = tf.cast(f0, tf.float32) / tf.cast(lamb, tf.float32)  # (N,1)
    inside = tf.cast(y_temp, tf.float32) + add_term  # (N,C)
    per_entry = (-1.0) * (1.0 / n) * tf.cast(y_true, tf.float32) * K.log(inside)
    loss = tf.reduce_sum(per_entry)

    return loss




## === cell 3
IMG_SIZE = 512
NUM_CLASSES = 5

base = keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling="avg",
)
base.trainable = False

inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
x = keras.applications.efficientnet.preprocess_input(inp)
x = base(x, training=False)
out = keras.layers.Dense(1 + NUM_CLASSES, name="gambler_logits")(x)

model_v3 = keras.Model(inputs=inp, outputs=out, name="gambler_like_efficientnetb0")
print("Built model:", model_v3.name)




## === cell 4
model_v3.summary()




## === cell 5
import matplotlib.pyplot as plt  # kept as in original environment




## === cell 6
BASE_PATH = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV_PATH = os.path.join(BASE_PATH, "train.csv")
TRAIN_DIR = os.path.join(BASE_PATH, "train_images")
TEST_DIR = os.path.join(BASE_PATH, "test_images")
SAMPLE_SUB_PATH = os.path.join(BASE_PATH, "sample_submission.csv")

TRAIN_TFREC_DIR = os.path.join(BASE_PATH, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(BASE_PATH, "test_tfrecords")

for p in [
    TRAIN_CSV_PATH,
    TRAIN_DIR,
    TEST_DIR,
    SAMPLE_SUB_PATH,
    TRAIN_TFREC_DIR,
    TEST_TFREC_DIR,
]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Required path not found: {p}")

train_df = pd.read_csv(TRAIN_CSV_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

train_df["label"] = train_df["label"].astype(str)

print("train.csv shape:", train_df.shape)
print("train.csv head:\n", train_df.head())
print("sample_submission shape:", sample_sub.shape)
print("sample_submission head:\n", sample_sub.head())




## === cell 7
AUTO = tf.data.AUTOTUNE


def _tfrecord_feature_description():
    return {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64),
    }


def _decode_and_resize(image_bytes):
    img = tf.io.decode_jpeg(image_bytes, channels=3)
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32)
    return img


def _augment_like_idg(img, seed_pair):
    s1, s2 = seed_pair[0], seed_pair[1]

    img = tf.image.stateless_random_flip_left_right(img, seed=[s1, s2])

    zoom = tf.random.stateless_uniform(
        [], seed=[s1, s2 + 1], minval=0.9, maxval=1.0, dtype=tf.float32
    )
    new_size = tf.cast(tf.cast(IMG_SIZE, tf.float32) * zoom, tf.int32)
    new_size = tf.maximum(new_size, 1)

    img_cropped = tf.image.stateless_random_crop(
        img, size=[new_size, new_size, 3], seed=[s1 + 1, s2 + 2]
    )
    img = tf.image.resize(
        img_cropped, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )

    pad = tf.cast(tf.round(tf.cast(IMG_SIZE, tf.float32) * 0.05), tf.int32)
    img = tf.pad(img, paddings=[[pad, pad], [pad, pad], [0, 0]], mode="REFLECT")
    img = tf.image.stateless_random_crop(
        img, size=[IMG_SIZE, IMG_SIZE, 3], seed=[s1 + 2, s2 + 3]
    )

    return img


def _parse_train(example_proto):
    ex = tf.io.parse_single_example(example_proto, _tfrecord_feature_description())
    img = _decode_and_resize(ex["image"])
    label = tf.cast(ex["target"], tf.int32)
    y = tf.one_hot(label, depth=NUM_CLASSES, dtype=tf.float32)
    return img, y


def _parse_test(example_proto):
    ex = tf.io.parse_single_example(
        example_proto,
        {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_name": tf.io.FixedLenFeature([], tf.string),
        },
    )
    img = _decode_and_resize(ex["image"])
    image_name = ex["image_name"]
    return img, image_name


_DATASET_OPTIONS = tf.data.Options()
_DATASET_OPTIONS.experimental_deterministic = True
_DATASET_OPTIONS.experimental_optimization.map_parallelization = True
_DATASET_OPTIONS.experimental_optimization.parallel_batch = True
_DATASET_OPTIONS.experimental_optimization.autotune_buffers = True


def build_train_ds(tfrecord_files, batch_size=16, seed=42):
    files = tf.data.Dataset.from_tensor_slices(tfrecord_files)
    files = files.shuffle(
        buffer_size=len(tfrecord_files), seed=seed, reshuffle_each_iteration=True
    )

    ds = files.interleave(
        lambda fn: tf.data.TFRecordDataset(fn, num_parallel_reads=AUTO),
        cycle_length=min(8, len(tfrecord_files)),
        num_parallel_calls=AUTO,
        deterministic=True,
    )
    ds = ds.map(_parse_train, num_parallel_calls=AUTO, deterministic=True)

    ds = ds.enumerate()

    def _apply_aug(i, data):
        img, y = data
        seed_pair = tf.stack([tf.cast(seed, tf.int32), tf.cast(i, tf.int32)])
        img = _augment_like_idg(img, seed_pair)
        return img, y

    ds = ds.map(_apply_aug, num_parallel_calls=AUTO, deterministic=True)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTO)

    ds = ds.with_options(_DATASET_OPTIONS)
    return ds


def build_test_ds(tfrecord_files, batch_size=16):
    files = tf.data.Dataset.from_tensor_slices(tfrecord_files)
    ds = files.interleave(
        lambda fn: tf.data.TFRecordDataset(fn, num_parallel_reads=AUTO),
        cycle_length=min(8, len(tfrecord_files)),
        num_parallel_calls=AUTO,
        deterministic=True,
    )
    ds = ds.map(_parse_test, num_parallel_calls=AUTO, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTO)
    ds = ds.with_options(_DATASET_OPTIONS)
    return ds


train_tfrec_files = sorted(
    [
        os.path.join(TRAIN_TFREC_DIR, f)
        for f in os.listdir(TRAIN_TFREC_DIR)
        if f.endswith(".tfrec")
    ]
)
test_tfrec_files = sorted(
    [
        os.path.join(TEST_TFREC_DIR, f)
        for f in os.listdir(TEST_TFREC_DIR)
        if f.endswith(".tfrec")
    ]
)

if len(train_tfrec_files) == 0 or len(test_tfrec_files) == 0:
    raise FileNotFoundError(
        "No TFRecord files found in train_tfrecords/test_tfrecords."
    )

BATCH_SIZE = 16
train_ds = build_train_ds(train_tfrec_files, batch_size=BATCH_SIZE, seed=42)

steps_per_epoch = max(1, int(np.ceil(len(train_df) / BATCH_SIZE)))
print("steps_per_epoch:", steps_per_epoch)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/283820250.py in <cell line: 0>()
     75 _DATASET_OPTIONS.experimental_optimization.map_parallelization = True
     76 _DATASET_OPTIONS.experimental_optimization.parallel_batch = True
---> 77 _DATASET_OPTIONS.experimental_optimization.autotune_buffers = True
     78 
     79 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/options.py in __setattr__(self, name, value)
     59       object.__setattr__(self, name, value)
     60     else:
---> 61       raise AttributeError("Cannot set the property {} on {}.".format(
     62           name,
     63           type(self).__name__))

AttributeError: Cannot set the property autotune_buffers on OptimizationOptions.

## === cell 8
model_v3.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=loss_gambler,
    steps_per_execution=max(1, min(64, steps_per_epoch)),
)

model_v3.fit(
    train_ds,
    epochs=3,
    steps_per_epoch=steps_per_epoch,
    verbose=1,
)

base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

model_v3.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss=loss_gambler,
    steps_per_execution=max(1, min(64, steps_per_epoch)),
)

model_v3.fit(
    train_ds,
    epochs=1,
    steps_per_epoch=steps_per_epoch,
    verbose=1,
)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1223173875.py in <cell line: 0>()
      2     optimizer=keras.optimizers.Adam(learning_rate=1e-3),
      3     loss=loss_gambler,
----> 4     steps_per_execution=max(1, min(64, steps_per_epoch)),
      5 )
      6 

NameError: name 'steps_per_epoch' is not defined

## === cell 9
test_ds_raw = build_test_ds(test_tfrec_files, batch_size=BATCH_SIZE)

test_ds_for_pred = test_ds_raw.map(
    lambda img, name: (img, tf.zeros([1], tf.float32)),
    num_parallel_calls=AUTO,
    deterministic=True,
)
test_ds_for_pred = test_ds_for_pred.with_options(_DATASET_OPTIONS)

pred_v3 = model_v3.predict(test_ds_for_pred, verbose=1)
pred_v3 = np.asarray(pred_v3)
print("Raw prediction shape:", pred_v3.shape)

if pred_v3.ndim != 2 or pred_v3.shape[1] < 2:
    raise ValueError(f"Unexpected prediction shape {pred_v3.shape}; expected (N, >=2).")

image_names = []
for _, batch_names in test_ds_raw.as_numpy_iterator():
    image_names.extend([n.decode("utf-8") for n in batch_names])

image_names = np.array(image_names)
if len(image_names) != pred_v3.shape[0]:
    m = min(len(image_names), pred_v3.shape[0])
    image_names = image_names[:m]
    pred_v3 = pred_v3[:m]

test_v3 = pd.DataFrame({"image_id": image_names})




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/705966905.py in <cell line: 0>()
      3 # We instead build ONE dataset that yields ((image, image_name), dummy_label) so Keras predict
      4 # consumes it while we still have a deterministic way to collect names in the same order.
----> 5 test_ds_raw = build_test_ds(test_tfrec_files, batch_size=BATCH_SIZE)
      6 
      7 # Dummy labels are ignored by predict(); structure is equivalent for prediction semantics.

NameError: name 'build_test_ds' is not defined

## === cell 10
predicted_class_indices_v3 = np.argmax(pred_v3[:, 1:], axis=1).astype(int)

submission_pred = pd.DataFrame(
    {"image_id": test_v3["image_id"].values, "label": predicted_class_indices_v3}
)

submission = sample_sub[["image_id"]].merge(submission_pred, on="image_id", how="left")
if submission["label"].isna().any():
    missing = submission[submission["label"].isna()]["image_id"].head(5).tolist()
    raise ValueError(f"Missing predictions for some test images (e.g., {missing}).")

submission["label"] = submission["label"].astype(int)

if submission.shape[0] != sample_sub.shape[0]:
    raise ValueError("Submission row count does not match sample_submission.")
if list(submission.columns) != ["image_id", "label"]:
    raise ValueError("Submission columns are incorrect.")
if submission["image_id"].isna().any():
    raise ValueError("Found NaN image_id in submission.")

out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote submission to:", out_path)
print(submission.head())

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/44955574.py in <cell line: 0>()
----> 1 predicted_class_indices_v3 = np.argmax(pred_v3[:, 1:], axis=1).astype(int)
      2 
      3 submission_pred = pd.DataFrame(
      4     {"image_id": test_v3["image_id"].values, "label": predicted_class_indices_v3}
      5 )

NameError: name 'pred_v3' is not defined
