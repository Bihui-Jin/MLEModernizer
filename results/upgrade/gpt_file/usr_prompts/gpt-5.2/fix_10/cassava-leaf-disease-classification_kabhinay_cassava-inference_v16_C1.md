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

# 5. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import tensorflow as tf

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
try:
    tf.random.set_seed(SEED)
except Exception:
    pass

print("TF version:", tf.__version__)




## === cell 1
"""Robust Bi-Tempered Logistic Loss Based on Bregman Divergences.

Source: https://bit.ly/3jSol8T

NOTE: Kept as-is to preserve original notebook content; not required for the minimal baseline.
"""
import functools

import tensorflow.compat.v1 as tf1


def for_loop(num_iters, body, initial_args):
    for i in range(num_iters):
        if i == 0:
            outputs = body(*initial_args)
        else:
            outputs = body(*outputs)
    return outputs


def log_t(u, t):
    def _internal_log_t(u, t):
        return (u ** (1.0 - t) - 1.0) / (1.0 - t)

    return tf1.cond(
        tf1.equal(t, 1.0),
        lambda: tf1.log(u),
        functools.partial(_internal_log_t, u, t),
    )


def exp_t(u, t):
    def _internal_exp_t(u, t):
        return tf1.nn.relu(1.0 + (1.0 - t) * u) ** (1.0 / (1.0 - t))

    return tf1.cond(
        tf1.equal(t, 1.0),
        lambda: tf1.exp(u),
        functools.partial(_internal_exp_t, u, t),
    )


def compute_normalization_fixed_point(activations, t, num_iters=5):
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
    return tf1.cond(
        tf1.less(t, 1.0),
        functools.partial(
            compute_normalization_binary_search, activations, t, num_iters
        ),
        functools.partial(compute_normalization_fixed_point, activations, t, num_iters),
    )


def _internal_bi_tempered_logistic_loss(activations, labels, t1, t2):
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


def tempered_sigmoid(activations, t, num_iters=5):
    t = tf1.convert_to_tensor(t)
    input_shape = tf1.shape(activations)
    activations_2d = tf1.reshape(activations, [-1, 1])
    internal_activations = tf1.concat(
        [tf1.zeros_like(activations_2d), activations_2d], 1
    )
    normalization_constants = tf1.cond(
        tf1.equal(t, 1.0),
        lambda: tf1.log(
            tf1.reduce_sum(tf1.exp(internal_activations), -1, keep_dims=True)
        ),
        functools.partial(compute_normalization, internal_activations, t, num_iters),
    )
    internal_probabilities = exp_t(internal_activations - normalization_constants, t)
    one_class_probabilities = tf1.split(internal_probabilities, 2, axis=1)[1]
    return tf1.reshape(one_class_probabilities, input_shape)


def tempered_softmax(activations, t, num_iters=5):
    t = tf1.convert_to_tensor(t)
    normalization_constants = tf1.cond(
        tf1.equal(t, 1.0),
        lambda: tf1.log(tf1.reduce_sum(tf1.exp(activations), -1, keep_dims=True)),
        functools.partial(compute_normalization, activations, t, num_iters),
    )
    return exp_t(activations - normalization_constants, t)


def bi_tempered_binary_logistic_loss(
    activations, labels, t1, t2, label_smoothing=0.0, num_iters=5
):
    with tf1.name_scope("binary_bitempered_logistic"):
        t1 = tf1.convert_to_tensor(t1)
        t2 = tf1.convert_to_tensor(t2)
        out_shape = tf1.shape(labels)
        labels_2d = tf1.reshape(labels, [-1, 1])
        activations_2d = tf1.reshape(activations, [-1, 1])
        internal_labels = tf1.concat([1.0 - labels_2d, labels_2d], 1)
        internal_logits = tf1.concat(
            [tf1.zeros_like(activations_2d), activations_2d], 1
        )
        losses = bi_tempered_logistic_loss(
            internal_logits, internal_labels, t1, t2, label_smoothing, num_iters
        )
        return tf1.reshape(losses, out_shape)


def bi_tempered_logistic_loss(
    labels, activations, t1=0.2, t2=1.0, label_smoothing=0.1, num_iters=10
):
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
            with tf1.name_scope("gradient_bitempered_logistic"):
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


def sparse_bi_tempered_logistic_loss(activations, labels, t1, t2, num_iters=5):
    with tf1.name_scope("sparse_bitempered_logistic"):
        t1 = tf1.convert_to_tensor(t1)
        t2 = tf1.convert_to_tensor(t2)
        num_classes = tf1.shape(activations)[-1]

        @tf1.custom_gradient
        def _custom_gradient_sparse_bi_tempered_logistic_loss(activations):
            with tf1.name_scope("gradient_sparse_bitempered_logistic"):
                probabilities = tempered_softmax(activations, t2, num_iters)
                loss_values = -log_t(
                    tf1.reshape(
                        tf1.gather_nd(
                            probabilities, tf1.where(tf1.one_hot(labels, num_classes))
                        ),
                        tf1.shape(activations)[:-1],
                    ),
                    t1,
                ) - 1.0 / (2.0 - t1) * (
                    1.0 - tf1.reduce_sum(tf1.pow(probabilities, 2.0 - t1), -1)
                )

                def grad(d_loss):
                    delta_probs = probabilities - tf1.one_hot(labels, num_classes)
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
                tf1.nn.sparse_softmax_cross_entropy_with_logits,
                labels=labels,
                logits=activations,
            ),
            functools.partial(
                _custom_gradient_sparse_bi_tempered_logistic_loss, activations
            ),
        )
        return loss_values




## === cell 2
DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
if not os.path.exists(DATA_DIR):
    DATA_DIR = "/kaggle/data/cassava-leaf-disease-classification"

train_csv = os.path.join(DATA_DIR, "train.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")
train_dir = os.path.join(DATA_DIR, "train_images")
test_dir = os.path.join(DATA_DIR, "test_images")

assert os.path.exists(train_csv), train_csv
assert os.path.exists(sample_path), sample_path
assert os.path.isdir(train_dir), train_dir
assert os.path.isdir(test_dir), test_dir

train_df = pd.read_csv(train_csv)
sample_df = pd.read_csv(sample_path)
test_df = sample_df[["image_id"]].copy()

print("train_df:", train_df.shape, "test_df:", test_df.shape)




## === cell 3
IMG_SIZE = (448, 448)
BATCH_SIZE = 8

train_df_shuf = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
val_frac = 0.1
val_size = int(len(train_df_shuf) * val_frac)
val_df = train_df_shuf.iloc[:val_size].copy()
trn_df = train_df_shuf.iloc[val_size:].copy()

AUTOTUNE = tf.data.AUTOTUNE
IMG_H, IMG_W = IMG_SIZE

num_classes = int(train_df["label"].nunique())
print("num_classes:", num_classes)


def _build_paths_and_labels(df, directory, with_labels):
    image_ids = df["image_id"].values
    prefix = directory.rstrip("/") + "/"
    paths = np.char.add(prefix, image_ids.astype(np.str_)).astype(np.str_)
    if with_labels:
        labels = df["label"].values.astype(np.int64)
        return paths, labels
    return paths


trn_paths, trn_labels = _build_paths_and_labels(trn_df, train_dir, with_labels=True)
val_paths, val_labels = _build_paths_and_labels(val_df, train_dir, with_labels=True)
tst_paths = _build_paths_and_labels(test_df, test_dir, with_labels=False)


@tf.function
def _decode_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [IMG_H, IMG_W], method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0
    return img


@tf.function
def tfa_image_rotate(image, angle):
    angle = tf.cast(angle, tf.float32)
    cos_a = tf.cos(angle)
    sin_a = tf.sin(angle)
    cx = (tf.cast(IMG_W, tf.float32) - 1.0) / 2.0
    cy = (tf.cast(IMG_H, tf.float32) - 1.0) / 2.0
    a0 = cos_a
    a1 = -sin_a
    b0 = sin_a
    b1 = cos_a
    a2 = cx - a0 * cx - a1 * cy
    b2 = cy - b0 * cx - b1 * cy
    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])
    transform = tf.reshape(transform, [1, 8])
    image4 = tf.expand_dims(image, 0)
    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=image4,
        transforms=transform,
        output_shape=tf.constant([IMG_H, IMG_W], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )
    return tf.squeeze(out, 0)


@tf.function
def _augment(img):
    img = tf.image.random_flip_left_right(img, seed=SEED)

    shift_x = tf.cast(tf.round(0.05 * tf.cast(IMG_W, tf.float32)), tf.int32)
    shift_y = tf.cast(tf.round(0.05 * tf.cast(IMG_H, tf.float32)), tf.int32)
    pad = tf.maximum(shift_x, shift_y)
    img = tf.image.pad_to_bounding_box(img, pad, pad, IMG_H + 2 * pad, IMG_W + 2 * pad)
    img = tf.image.random_crop(img, size=[IMG_H, IMG_W, 3], seed=SEED)

    scale = tf.random.uniform([], 0.9, 1.1, seed=SEED)
    new_h = tf.cast(tf.round(scale * tf.cast(IMG_H, tf.float32)), tf.int32)
    new_w = tf.cast(tf.round(scale * tf.cast(IMG_W, tf.float32)), tf.int32)
    img2 = tf.image.resize(img, [new_h, new_w], method=tf.image.ResizeMethod.BILINEAR)
    img2 = tf.image.resize_with_crop_or_pad(img2, IMG_H, IMG_W)

    angle = tf.random.uniform([], -15.0, 15.0, seed=SEED) * (np.pi / 180.0)
    img2 = tfa_image_rotate(img2, angle)
    return img2


@tf.function
def _to_onehot(label):
    return tf.one_hot(tf.cast(label, tf.int32), depth=num_classes, dtype=tf.float32)


tfrecord_train_dir = os.path.join(DATA_DIR, "train_tfrecords")
tfrecord_test_dir = os.path.join(DATA_DIR, "test_tfrecords")


def _list_tfrec_files(directory):
    if not tf.io.gfile.exists(directory):
        return []
    files = tf.io.gfile.glob(os.path.join(directory, "*.tfrec"))
    return sorted(files)


train_tfrec_files = _list_tfrec_files(tfrecord_train_dir)
test_tfrec_files = _list_tfrec_files(tfrecord_test_dir)

_use_tfrec = (len(train_tfrec_files) > 0) and (len(test_tfrec_files) > 0)
print(
    "Using TFRecords:",
    _use_tfrec,
    "n_train_tfrec:",
    len(train_tfrec_files),
    "n_test_tfrec:",
    len(test_tfrec_files),
)

_tfrec_feature_desc = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


@tf.function
def _parse_tfrec(example_proto):
    ex = tf.io.parse_single_example(example_proto, _tfrec_feature_desc)
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(img, [IMG_H, IMG_W], method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0
    image_id = ex["image_name"]
    label = ex["target"]
    return image_id, img, label


def _make_split_tables_once():
    trn_ids = trn_df["image_id"].values.astype(np.str_)
    val_ids = val_df["image_id"].values.astype(np.str_)

    trn_keys = tf.constant(trn_ids)
    val_keys = tf.constant(val_ids)
    trn_vals = tf.ones([trn_keys.shape[0]], dtype=tf.bool)
    val_vals = tf.ones([val_keys.shape[0]], dtype=tf.bool)

    trn_table_local = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(trn_keys, trn_vals),
        default_value=False,
    )
    val_table_local = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(val_keys, val_vals),
        default_value=False,
    )
    return trn_table_local, val_table_local


trn_table, val_table = _make_split_tables_once()


def _dataset_options():
    opts = tf.data.Options()
    opts.experimental_deterministic = True  # preserve determinism semantics
    try:
        opts.experimental_optimization.apply_default_optimizations = True
        opts.experimental_optimization.map_parallelization = True
        opts.experimental_optimization.parallel_batch = True
    except Exception:
        pass
    return opts


def make_train_ds(paths, labels):
    if _use_tfrec:
        ds = tf.data.TFRecordDataset(train_tfrec_files, num_parallel_reads=AUTOTUNE)
        ds = ds.with_options(_dataset_options())

        ds = ds.map(_parse_tfrec, num_parallel_calls=AUTOTUNE)
        ds = ds.filter(lambda image_id, img, label: trn_table.lookup(image_id))

        ds = ds.cache()

        ds = ds.shuffle(buffer_size=8192, seed=SEED, reshuffle_each_iteration=True)

        @tf.function
        def _map(image_id, img, label):
            img = _augment(img)
            return img, _to_onehot(label)

        ds = ds.map(_map, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(BATCH_SIZE, drop_remainder=False)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.with_options(_dataset_options())
    ds = ds.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)

    @tf.function
    def _map(path, label):
        img = _decode_resize(path)
        img = _augment(img)
        return img, _to_onehot(label)

    ds = ds.map(_map, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_eval_ds(paths, labels=None):
    if labels is None:
        if _use_tfrec:
            ds = tf.data.TFRecordDataset(test_tfrec_files, num_parallel_reads=AUTOTUNE)
            ds = ds.with_options(_dataset_options())

            ds = ds.map(_parse_tfrec, num_parallel_calls=AUTOTUNE)
            ds = ds.map(lambda image_id, img, label: img, num_parallel_calls=AUTOTUNE)
            ds = ds.batch(BATCH_SIZE, drop_remainder=False)
            ds = ds.cache()
            ds = ds.prefetch(AUTOTUNE)
            return ds

        ds = tf.data.Dataset.from_tensor_slices(paths)
        ds = ds.with_options(_dataset_options())
        ds = ds.map(_decode_resize, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(BATCH_SIZE, drop_remainder=False)
        ds = ds.cache()
        ds = ds.prefetch(AUTOTUNE)
        return ds

    if _use_tfrec:
        ds = tf.data.TFRecordDataset(train_tfrec_files, num_parallel_reads=AUTOTUNE)
        ds = ds.with_options(_dataset_options())

        ds = ds.map(_parse_tfrec, num_parallel_calls=AUTOTUNE)
        ds = ds.filter(lambda image_id, img, label: val_table.lookup(image_id))

        ds = ds.cache()

        ds = ds.map(
            lambda image_id, img, label: (img, _to_onehot(label)),
            num_parallel_calls=AUTOTUNE,
        )
        ds = ds.batch(BATCH_SIZE, drop_remainder=False)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.with_options(_dataset_options())

    @tf.function
    def _map(path, label):
        img = _decode_resize(path)
        return img, _to_onehot(label)

    ds = ds.map(_map, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.cache()
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_train_ds(trn_paths, trn_labels)
val_ds = make_eval_ds(val_paths, val_labels)
test_ds = make_eval_ds(tst_paths, labels=None)

steps_per_epoch = None
val_steps = None

class_indices = {str(i): i for i in range(num_classes)}
print("class_indices:", class_indices)




## === cell 4
inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

EPOCHS = 3

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    verbose=1,
)




## === cell 5
pred = model.predict(test_ds, verbose=1)

n_test = test_df.shape[0]
pred = pred[:n_test]

idx_to_label_str = {v: k for k, v in class_indices.items()}
pred_idx = np.argmax(pred, axis=1).astype(np.int64)

lut = np.empty((num_classes,), dtype=np.int64)
for v, k in idx_to_label_str.items():
    lut[int(v)] = int(k)
pred_labels = lut[pred_idx]

submission = pd.DataFrame(
    {"image_id": test_df["image_id"].values, "label": pred_labels}
)

assert submission.shape[0] == sample_df.shape[0]
assert list(submission.columns) == ["image_id", "label"]

out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)
print(submission.head())
print("Wrote", out_path, "with shape:", submission.shape)
