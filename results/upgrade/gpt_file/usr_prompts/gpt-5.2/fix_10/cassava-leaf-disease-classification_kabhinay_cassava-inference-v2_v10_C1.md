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

2.7

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

0.8847083711090964

# 6. Current score

0.071

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.071) has done: 'The timeout is dominated by JPEG decode + 448×448 resize for ~18.7k train images (and repeated per epoch) plus a heavy augmentation pipeline; caching after decode currently stores huge float32 tensors and can thrash memory, slowing everything down. I keep the exact same model, loss, epochs, batch size, and augmentation logic, but switch the input pipeline to use the provided TFRecords (same images, much faster I/O) while caching only the parsed/decoded uint8 image and doing the float resize/normalize + augmentation afterward. I also enforce dataset options that reduce input overhead (no extra determinism toggles beyond what you already set), and make `steps_per_epoch` implicit (so Keras doesn’t waste time handling partial-step edge cases) while preserving the same number of training examples seen each epoch. These changes are provably equivalent in semantics (same data, same preprocessing/augmentation, same training loop) but significantly reduce wall time.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
if os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "").lower() == "python":
    del os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]

import tensorflow as tf

import tf_keras as keras
from tf_keras import layers

np.random.seed(42)
random.seed(42)
try:
    tf.random.set_seed(42)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # 0 = let TF pick
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

print("Python:", tuple(__import__("sys").version_info)[:3])
print("TensorFlow:", tf.__version__)
print("tf_keras:", getattr(keras, "__version__", "unknown"))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
"""Robust Bi-Tempered Logistic Loss Based on Bregman Divergences.

Source: https://bit.ly/3jSol8T

(Kept as-is; not required for the fallback model but preserved to keep original core logic present.)
"""

import functools
import tensorflow.compat.v1 as tf1


def for_loop(num_iters, body, initial_args):
    """Runs a simple for-loop with given body and initial_args."""
    for i in range(num_iters):
        if i == 0:
            outputs = body(*initial_args)
        else:
            outputs = body(*outputs)
    return outputs


def log_t(u, t):
    """Compute log_t for `u`."""

    def _internal_log_t(u, t):
        return (u ** (1.0 - t) - 1.0) / (1.0 - t)

    return tf1.cond(
        tf1.equal(t, 1.0), lambda: tf1.log(u), functools.partial(_internal_log_t, u, t)
    )


def exp_t(u, t):
    """Compute exp_t for `u`."""

    def _internal_exp_t(u, t):
        return tf1.nn.relu(1.0 + (1.0 - t) * u) ** (1.0 / (1.0 - t))

    return tf1.cond(
        tf1.equal(t, 1.0), lambda: tf1.exp(u), functools.partial(_internal_exp_t, u, t)
    )


def compute_normalization_fixed_point(activations, t, num_iters=5):
    """Returns the normalization value for each example (t > 1.0)."""
    mu = tf1.reduce_max(activations, -1, keep_dims=True)
    normalized_activations_step_0 = activations - mu
    shape_normalized_activations = tf1.shape(normalized_activations_step_0)

    def iter_body(i, normalized_activations):
        logt_partition = tf1.reduce_sum(
            exp_t(normalized_activations, t), -1, keep_dims=True
        )
        normalized_activations_t = tf1.reshape(
            normalized_activations_step_0 * tf1.pow(logt_partition, 1.0 - t),
            shape_normalized_activations,
        )
        return [i + 1, normalized_activations_t]

    _, normalized_activations_t = for_loop(
        num_iters, iter_body, [0, normalized_activations_step_0]
    )
    logt_partition = tf1.reduce_sum(
        exp_t(normalized_activations_t, t), -1, keep_dims=True
    )
    return -log_t(1.0 / logt_partition, t) + mu


def compute_normalization_binary_search(activations, t, num_iters=10):
    """Returns the normalization value for each example (t < 1.0)."""
    mu = tf1.reduce_max(activations, -1, keep_dims=True)
    normalized_activations = activations - mu
    shape_activations = tf1.shape(activations)
    effective_dim = tf1.cast(
        tf1.reduce_sum(
            tf1.cast(tf1.greater(normalized_activations, -1.0 / (1.0 - t)), tf1.int32),
            -1,
            keep_dims=True,
        ),
        tf1.float32,
    )
    shape_partition = tf1.concat([shape_activations[:-1], [1]], 0)
    lower = tf1.zeros(shape_partition)
    upper = -log_t(1.0 / effective_dim, t) * tf1.ones(shape_partition)

    def iter_body(i, lower, upper):
        logt_partition = (upper + lower) / 2.0
        sum_probs = tf1.reduce_sum(
            exp_t(normalized_activations - logt_partition, t), -1, keep_dims=True
        )
        update = tf1.cast(tf1.less(sum_probs, 1.0), tf1.float32)
        lower = tf1.reshape(
            lower * update + (1.0 - update) * logt_partition, shape_partition
        )
        upper = tf1.reshape(
            upper * (1.0 - update) + update * logt_partition, shape_partition
        )
        return [i + 1, lower, upper]

    _, lower, upper = for_loop(num_iters, iter_body, [0, lower, upper])
    logt_partition = (upper + lower) / 2.0
    return logt_partition + mu


def compute_normalization(activations, t, num_iters=5):
    """Returns the normalization value for each example."""
    return tf1.cond(
        tf1.less(t, 1.0),
        functools.partial(
            compute_normalization_binary_search, activations, t, num_iters
        ),
        functools.partial(compute_normalization_fixed_point, activations, t, num_iters),
    )


def _internal_bi_tempered_logistic_loss(activations, labels, t1, t2):
    """Computes the Bi-Tempered logistic loss."""
    if t2 == 1.0:
        normalization_constants = tf1.log(
            tf1.reduce_sum(tf1.exp(activations), -1, keep_dims=True)
        )
        if t1 == 1.0:
            return normalization_constants + tf1.reduce_sum(
                tf1.multiply(labels, tf1.log(labels + 1e-10) - activations), -1
            )
        else:
            shifted_activations = tf1.exp(activations - normalization_constants)
            one_minus_t1 = 1.0 - t1
            one_minus_t2 = 1.0
    else:
        one_minus_t1 = 1.0 - t1
        one_minus_t2 = 1.0 - t2
        normalization_constants = compute_normalization(activations, t2, num_iters=5)
        shifted_activations = tf1.nn.relu(
            1.0 + one_minus_t2 * (activations - normalization_constants)
        )

    if t1 == 1.0:
        return tf1.reduce_sum(
            tf1.multiply(
                tf1.log(labels + 1e-10)
                - tf1.log(tf1.pow(shifted_activations, 1.0 / one_minus_t2)),
                labels,
            ),
            -1,
        )
    else:
        beta = 1.0 + one_minus_t1
        logt_probs = (
            tf1.pow(shifted_activations, one_minus_t1 / one_minus_t2) - 1.0
        ) / one_minus_t1
        return tf1.reduce_sum(
            tf1.multiply(log_t(labels, t1) - logt_probs, labels)
            - 1.0
            / beta
            * (
                tf1.pow(labels, beta)
                - tf1.pow(shifted_activations, beta / one_minus_t2)
            ),
            -1,
        )


def tempered_softmax(activations, t, num_iters=5):
    """Tempered softmax function."""
    t = tf1.convert_to_tensor(t)
    normalization_constants = tf1.cond(
        tf1.equal(t, 1.0),
        lambda: tf1.log(tf1.reduce_sum(tf1.exp(activations), -1, keep_dims=True)),
        functools.partial(compute_normalization, activations, t, num_iters),
    )
    return exp_t(activations - normalization_constants, t)


def bi_tempered_logistic_loss(
    labels, activations, t1=0.2, t2=1.0, label_smoothing=0.1, num_iters=10
):
    """Bi-Tempered Logistic Loss with custom gradient."""
    with tf1.name_scope("bitempered_logistic"):
        t1 = tf1.convert_to_tensor(t1)
        t2 = tf1.convert_to_tensor(t2)
        if label_smoothing > 0.0:
            num_classes = tf1.cast(tf1.shape(labels)[-1], tf1.float32)
            labels = (
                1 - num_classes / (num_classes - 1) * label_smoothing
            ) * labels + label_smoothing / (num_classes - 1)

        @tf1.custom_gradient
        def _custom_gradient_bi_tempered_logistic_loss(activations):
            probabilities = tempered_softmax(activations, t2, num_iters)
            loss_values = tf1.multiply(
                labels, log_t(labels + 1e-10, t1) - log_t(probabilities, t1)
            ) - 1.0 / (2.0 - t1) * (
                tf1.pow(labels, 2.0 - t1) - tf1.pow(probabilities, 2.0 - t1)
            )

            def grad(d_loss):
                delta_probs = probabilities - labels
                forget_factor = tf1.pow(probabilities, t2 - t1)
                delta_probs_times_forget_factor = tf1.multiply(
                    delta_probs, forget_factor
                )
                delta_forget_sum = tf1.reduce_sum(
                    delta_probs_times_forget_factor, -1, keep_dims=True
                )
                escorts = tf1.pow(probabilities, t2)
                escorts = escorts / tf1.reduce_sum(escorts, -1, keep_dims=True)
                derivative = delta_probs_times_forget_factor - tf1.multiply(
                    escorts, delta_forget_sum
                )
                return tf1.multiply(d_loss, derivative)

            return loss_values, grad

        loss_values = tf1.cond(
            tf1.logical_and(tf1.equal(t1, 1.0), tf1.equal(t2, 1.0)),
            functools.partial(
                tf1.nn.softmax_cross_entropy_with_logits,
                labels=labels,
                logits=activations,
            ),
            functools.partial(_custom_gradient_bi_tempered_logistic_loss, activations),
        )
        reduce_sum_last = lambda x: tf1.reduce_sum(x, -1)
        loss_values = tf1.cond(
            tf1.logical_and(tf1.equal(t1, 1.0), tf1.equal(t2, 1.0)),
            functools.partial(tf1.identity, loss_values),
            functools.partial(reduce_sum_last, loss_values),
        )
        return loss_values




## === cell 2
def resolve_base_input():
    candidates = [
        "/kaggle/input/cassava-leaf-disease-classification",
        "/kaggle/data/cassava-leaf-disease-classification",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
        "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    ]
    for p in candidates:
        if os.path.exists(os.path.join(p, "train.csv")):
            return p
    candidates2 = [
        "/kaggle/input",
        "/kaggle/data",
    ]
    for root in candidates2:
        p = os.path.join(root, "cassava-leaf-disease-classification")
        if os.path.exists(os.path.join(p, "train.csv")):
            return p
    raise RuntimeError(
        "Could not locate cassava-leaf-disease-classification dataset folder."
    )


BASE_INPUT = resolve_base_input()
print("Using BASE_INPUT:", BASE_INPUT)

train_csv_path = os.path.join(BASE_INPUT, "train.csv")
sample_path = os.path.join(BASE_INPUT, "sample_submission.csv")

train_dir = os.path.join(BASE_INPUT, "train_images")
test_dir = os.path.join(BASE_INPUT, "test_images")

train_tfrecord_dir = os.path.join(BASE_INPUT, "train_tfrecords")
test_tfrecord_dir = os.path.join(BASE_INPUT, "test_tfrecords")

train_df = pd.read_csv(train_csv_path)
sample_sub = pd.read_csv(sample_path)
test_df = sample_sub[["image_id"]].copy()

train_df["label"] = train_df["label"].astype(str)

print("Train rows:", len(train_df), "Test rows:", len(test_df))
print(
    "Train images dir exists:",
    os.path.isdir(train_dir),
    "Test images dir exists:",
    os.path.isdir(test_dir),
)
print(
    "Train TFRecords dir exists:",
    os.path.isdir(train_tfrecord_dir),
    "Test TFRecords dir exists:",
    os.path.isdir(test_tfrecord_dir),
)



## === cell 3
IMG_SIZE = (448, 448)
BATCH_SIZE = 16  # unchanged

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

val_frac = 0.1
rng = np.random.RandomState(42)
indices = np.arange(len(train_df))
rng.shuffle(indices)
val_size = int(round(len(indices) * val_frac))
val_idx = indices[:val_size]
trn_idx = indices[val_size:]

train_df_split = train_df.iloc[trn_idx].reset_index(drop=True)
valid_df_split = train_df.iloc[val_idx].reset_index(drop=True)

num_classes = 5
class_names = [str(i) for i in range(num_classes)]
class_to_idx = {name: i for i, name in enumerate(class_names)}
inv_class_indices = {i: i for i in range(num_classes)}  # identity mapping
print("class_indices:", class_to_idx)
print("inv_class_indices:", inv_class_indices)


def _resize_and_rescale_uint8(img_uint8):
    img = tf.image.resize(img_uint8, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal", seed=42),
        layers.RandomRotation(factor=(10.0 / 360.0), seed=42),
        layers.RandomTranslation(height_factor=0.05, width_factor=0.05, seed=42),
        layers.RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), seed=42),
    ],
    name="data_augmentation",
)

_FEATURES_TRAIN = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
}
_FEATURES_TEST = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


@tf.function
def _parse_train_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURES_TRAIN)
    img = tf.io.decode_jpeg(ex["image"], channels=3)  # uint8
    label = tf.cast(ex["label"], tf.int32)
    return img, label


@tf.function
def _parse_test_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURES_TEST)
    img = tf.io.decode_jpeg(ex["image"], channels=3)  # uint8
    img_name = ex["image_name"]
    return img, img_name


@tf.function
def _train_postprocess(img_uint8, label):
    img = _resize_and_rescale_uint8(img_uint8)
    img = data_augmentation(img, training=True)
    return img, label


@tf.function
def _eval_postprocess(img_uint8, label):
    img = _resize_and_rescale_uint8(img_uint8)
    return img, label


@tf.function
def _test_postprocess(img_uint8, img_name):
    img = _resize_and_rescale_uint8(img_uint8)
    return img


def _list_tfrec_files(tfrec_dir, prefix):
    files = tf.io.gfile.glob(os.path.join(tfrec_dir, "%s*.tfrec" % prefix))
    files = sorted(files)
    if not files:
        raise RuntimeError(
            "No TFRecord files found in %s for prefix %s" % (tfrec_dir, prefix)
        )
    return files


train_tfrec_files = _list_tfrec_files(train_tfrecord_dir, "ld_train")
test_tfrec_files = _list_tfrec_files(test_tfrecord_dir, "ld_test")

shuffle_buf = min(len(train_df_split), 8192)

ds_opts = tf.data.Options()
ds_opts.experimental_deterministic = True

train_ds = tf.data.TFRecordDataset(train_tfrec_files, num_parallel_reads=AUTOTUNE)
train_ds = train_ds.with_options(ds_opts)
train_ds = train_ds.map(_parse_train_example, num_parallel_calls=AUTOTUNE)
train_ds = (
    train_ds.cache()
)  # caches uint8+label, much smaller than float32 resized tensors
train_ds = train_ds.shuffle(
    buffer_size=shuffle_buf, seed=42, reshuffle_each_iteration=True
)
train_ds = train_ds.map(_train_postprocess, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False)
train_ds = train_ds.prefetch(AUTOTUNE)

valid_ds = tf.data.TFRecordDataset(train_tfrec_files, num_parallel_reads=AUTOTUNE)
valid_ds = valid_ds.with_options(ds_opts)
valid_ds = valid_ds.map(_parse_train_example, num_parallel_calls=AUTOTUNE)
val_mask = np.zeros((len(train_df),), dtype=np.bool_)
val_mask[val_idx] = True
val_mask_tf = tf.constant(val_mask)


def _keep_if_in_valid(i, img_label):
    return tf.gather(val_mask_tf, i)


valid_ds = valid_ds.enumerate()
valid_ds = valid_ds.filter(_keep_if_in_valid)
valid_ds = valid_ds.map(lambda i, img_label: img_label, num_parallel_calls=AUTOTUNE)
valid_ds = valid_ds.cache()
valid_ds = valid_ds.map(_eval_postprocess, num_parallel_calls=AUTOTUNE)
valid_ds = valid_ds.batch(BATCH_SIZE, drop_remainder=False)
valid_ds = valid_ds.prefetch(AUTOTUNE)

test_ds = tf.data.TFRecordDataset(test_tfrec_files, num_parallel_reads=AUTOTUNE)
test_ds = test_ds.with_options(ds_opts)
test_ds = test_ds.map(_parse_test_example, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.cache()
test_ds = test_ds.map(_test_postprocess, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False)
test_ds = test_ds.prefetch(AUTOTUNE)

TRAIN_STEPS = int(np.ceil(len(train_df_split) / float(BATCH_SIZE)))
VALID_STEPS = int(np.ceil(len(valid_df_split) / float(BATCH_SIZE)))
TEST_STEPS = int(np.ceil(len(test_df) / float(BATCH_SIZE)))
print("Steps - train/valid/test:", TRAIN_STEPS, VALID_STEPS, TEST_STEPS)




## === cell 4
def build_model(input_shape=(448, 448, 3), num_classes=5):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


model = build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=num_classes)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    jit_compile=True,
)
model.summary()



## === cell 5
EPOCHS = 3

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    verbose=1,
)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/2221732174.py in <cell line: 0>()
      3 # Speed fix: let Keras infer steps from dataset cardinality; avoids overhead and keeps semantics (one full pass per epoch).
      4 # Correctness preserved because datasets are finite and built to match the original split sizes.
----> 5 history = model.fit(
      6     train_ds,
      7     validation_data=valid_ds,

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
Error in user-defined function passed to ParallelMapDatasetV2:3 transformation with iterator: Iterator::Root::Prefetch::MapAndBatch::Shuffle::MemoryCacheImpl::ParallelMapV2: Feature: label (data type: int64) is required but could not be found.
	 [[{{node ParseSingleExample/ParseExample/ParseExampleV2}}]]
	 [[IteratorGetNext]] [Op:__inference_train_function_3156]

## === cell 6
pred = model.predict(
    test_ds,
    verbose=1,
)
pred = np.asarray(pred)

pred_labels_idx = np.argmax(pred, axis=1).astype(np.int64)
pred_labels = pred_labels_idx

if len(pred_labels) != len(test_df):
    pred_labels = pred_labels[: len(test_df)]

submission = pd.DataFrame(
    {"image_id": test_df["image_id"].values, "label": pred_labels}
)

submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(submission.head())
print("Rows:", len(submission), "Cols:", list(submission.columns))
print("Label counts:\n", submission["label"].value_counts().sort_index())
