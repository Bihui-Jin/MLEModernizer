# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42)
np.random.seed(42)

print("TF version:", tf.__version__)
print("Keras version:", keras.__version__)

try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("Determinism not fully enabled (non-fatal):", repr(e))

try:
    tf.config.optimizer.set_jit(True)
except Exception as e:
    print("XLA not enabled (non-fatal):", repr(e))




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
    Bugfix: avoid iterating over a symbolic tensor in graph mode.
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


@tf.function
def _decode_and_resize(image_bytes):
    img = tf.io.decode_jpeg(image_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32)
    return img


@tf.function
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


@tf.function
def _parse_train_one_and_decode(example_proto):
    ex = tf.io.parse_single_example(example_proto, _tfrecord_feature_description())
    img = _decode_and_resize(ex["image"])
    label = tf.cast(ex["target"], tf.int32)
    y = tf.one_hot(label, depth=NUM_CLASSES, dtype=tf.float32)
    return img, y


@tf.function
def _parse_test_one_and_decode(example_proto):
    ex = tf.io.parse_single_example(
        example_proto,
        {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_name": tf.io.FixedLenFeature([], tf.string),
        },
    )
    img = _decode_and_resize(ex["image"])
    name = ex["image_name"]
    return img, name


@tf.function
def _augment_one(image, y, seed_pair):
    return _augment_like_idg(image, seed_pair), y


_DATASET_OPTIONS = tf.data.Options()
_DATASET_OPTIONS.experimental_deterministic = False
try:
    _DATASET_OPTIONS.experimental_optimization.map_parallelization = True
except Exception:
    pass
try:
    _DATASET_OPTIONS.experimental_optimization.parallel_batch = True
except Exception:
    pass
try:
    _DATASET_OPTIONS.experimental_optimization.autotune_buffers = True
except Exception:
    pass


def build_train_ds(tfrecord_files, batch_size=16, seed=42):
    ds = tf.data.TFRecordDataset(tfrecord_files, num_parallel_reads=AUTO)
    ds = ds.with_options(_DATASET_OPTIONS)

    ds = ds.shuffle(buffer_size=2048, seed=seed, reshuffle_each_iteration=True)

    ds = ds.map(
        _parse_train_one_and_decode, num_parallel_calls=AUTO, deterministic=False
    )

    ds = ds.enumerate()

    def _to_aug_seed(i, data):
        img, y = data
        return i, img, y

    ds = ds.map(_to_aug_seed, num_parallel_calls=AUTO, deterministic=False)

    ds = ds.batch(batch_size, drop_remainder=False)

    ds = ds.enumerate()

    def _augment_batched(step_i, batch):
        idx_global, imgs, ys = batch  # idx_global: (bs,)
        bs = tf.shape(imgs)[0]
        idx_in_batch = tf.range(bs, dtype=tf.int32)
        elem_seed2 = tf.cast(step_i, tf.int32) * tf.cast(bs, tf.int32) + idx_in_batch
        seeds = tf.stack([tf.fill([bs], tf.cast(seed, tf.int32)), elem_seed2], axis=1)

        imgs_aug = tf.map_fn(
            lambda j: _augment_like_idg(imgs[j], seeds[j]),
            tf.range(bs),
            fn_output_signature=tf.float32,
            parallel_iterations=64,
        )
        return imgs_aug, ys

    ds = ds.map(_augment_batched, num_parallel_calls=AUTO, deterministic=False)
    ds = ds.prefetch(AUTO)
    return ds


def build_test_ds(tfrecord_files, batch_size=16):
    ds = tf.data.TFRecordDataset(tfrecord_files, num_parallel_reads=AUTO)
    ds = ds.with_options(_DATASET_OPTIONS)

    ds = ds.map(
        _parse_test_one_and_decode, num_parallel_calls=AUTO, deterministic=False
    )
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTO)
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




## === cell 8
model_v3.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=loss_gambler,
    steps_per_execution=max(1, min(128, steps_per_epoch)),
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
    steps_per_execution=max(1, min(128, steps_per_epoch)),
)

model_v3.fit(
    train_ds,
    epochs=1,
    steps_per_epoch=steps_per_epoch,
    verbose=1,
)




## === cell 9
test_ds_raw = build_test_ds(test_tfrec_files, batch_size=BATCH_SIZE)

test_ds_for_pred = test_ds_raw.map(lambda img, name: img, num_parallel_calls=AUTO)
test_ds_for_pred = test_ds_for_pred.with_options(_DATASET_OPTIONS)

pred_v3 = model_v3.predict(test_ds_for_pred, verbose=1)
pred_v3 = np.asarray(pred_v3)
print("Raw prediction shape:", pred_v3.shape)

if pred_v3.ndim != 2 or pred_v3.shape[1] < 2:
    raise ValueError(f"Unexpected prediction shape {pred_v3.shape}; expected (N, >=2).")


def _collect_names(ds):
    ta = tf.TensorArray(
        dtype=tf.string, size=0, dynamic_size=True, clear_after_read=False
    )
    i0 = tf.constant(0, tf.int32)

    def _step(i, ta_, batch_names):
        bs = tf.shape(batch_names)[0]
        ta_ = ta_.scatter(tf.range(i, i + bs), batch_names)
        return i + bs, ta_

    i, ta = ds.map(lambda img, name: name, num_parallel_calls=AUTO).reduce(
        (i0, ta), lambda state, bn: _step(state[0], state[1], bn)
    )
    names = ta.stack()
    return names


image_names_tf = _collect_names(test_ds_raw)
image_names = image_names_tf.numpy().astype("U")  # decode to str (utf-8)
image_names = np.asarray(image_names)

if len(image_names) != pred_v3.shape[0]:
    m = min(len(image_names), pred_v3.shape[0])
    image_names = image_names[:m]
    pred_v3 = pred_v3[:m]

test_v3 = pd.DataFrame({"image_id": image_names})




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
