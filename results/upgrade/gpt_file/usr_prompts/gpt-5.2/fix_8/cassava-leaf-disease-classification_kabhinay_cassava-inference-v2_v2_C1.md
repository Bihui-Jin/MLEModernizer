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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

np.random.seed(0)
tf.random.set_seed(0)

try:
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(
        max(1, (os.cpu_count() or 2) - 1)
    )
except Exception:
    pass

print("TF:", tf.__version__)
print("Eager:", tf.executing_eagerly())




## === cell 1
"""Robust Bi-Tempered Logistic Loss Based on Bregman Divergences.

Source: https://bit.ly/3jSol8T
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
        tf1.equal(t, 1.0), lambda: tf1.log(u), functools.partial(_internal_log_t, u, t)
    )


def exp_t(u, t):
    def _internal_exp_t(u, t):
        return tf1.nn.relu(1.0 + (1.0 - t) * u) ** (1.0 / (1.0 - t))

    return tf1.cond(
        tf1.equal(t, 1.0), lambda: tf1.exp(u), functools.partial(_internal_exp_t, u, t)
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


import tensorflow.keras.backend as K


def loss_gambler(label_smoothing=0.0):
    def loss_gamb(y_true, y_pred):
        y_true = tf.math.add(
            y_true,
            tf.math.add(
                tf.math.multiply(label_smoothing / 2.0, tf.math.add(1.0, -1 * y_true)),
                tf.math.multiply(-1 * label_smoothing / 2.0, y_true),
            ),
        )
        y_temp = y_pred[:, 1:]
        f0 = y_pred[:, 0]
        lamb = tf.math.divide(
            tf.math.multiply(K.sum(y_temp), K.sum(y_temp)),
            K.sum(tf.math.multiply(y_temp, y_temp)),
        )
        loss = tf.constant((0.0,))
        for i in range(len(y_true[0])):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[(loss, tf.TensorShape([None]))]
            )
            temp = tf.constant((0.0,))
            loss = tf.math.add(
                loss,
                tf.math.add(
                    temp,
                    (
                        -1.0
                        * (1 / float(len(y_true)))
                        * tf.math.multiply(
                            y_true[:, i], K.log(y_temp[:, i] + f0 / lamb)
                        )
                    ),
                ),
            )
        return tf.math.reduce_sum(loss)

    return loss_gamb




## === cell 3
def build_fallback_model(input_shape=(448, 448, 3), num_classes=5):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)


def try_load_savedmodel_as_keras_model(savedmodel_dir: str):
    if not savedmodel_dir or (not os.path.exists(savedmodel_dir)):
        return None
    try:
        layer = tf.keras.layers.TFSMLayer(
            savedmodel_dir, call_endpoint="serving_default"
        )
        inp = tf.keras.Input(shape=(448, 448, 3), dtype=tf.float32)
        out = layer(inp)
        if isinstance(out, dict):
            first_key = sorted(out.keys())[0]
            out = out[first_key]
        return tf.keras.Model(inputs=inp, outputs=out)
    except Exception as e:
        print(f"Could not load SavedModel at {savedmodel_dir}: {type(e).__name__}: {e}")
        return None


model_v1_path = "/kaggle/input/only-xception-with-cropping/saved-model-11-0.879"
model_v2_path = "/kaggle/input/efficientnet-with-cropping/saved-model-06-0.88"

model_v1 = try_load_savedmodel_as_keras_model(model_v1_path)
model_v2 = try_load_savedmodel_as_keras_model(model_v2_path)

if model_v1 is not None and model_v2 is not None:
    print("Loaded external models.")
else:
    print("External models not available; using fallback model(s).")
    model_v1 = build_fallback_model()
    model_v2 = build_fallback_model()

print("model_v1 output shape:", model_v1.output_shape)
print("model_v2 output shape:", model_v2.output_shape)




## === cell 4
def random_crop(img, random_crop_size):
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y : (y + dy), x : (x + dx), :]


def _batch_random_crop(batch_x, crop_length: int):
    b, h, w, c = batch_x.shape
    dy = dx = crop_length
    if h == dy and w == dx:
        return batch_x.astype(np.float32, copy=False)
    max_x = w - dx
    max_y = h - dy
    xs = np.random.randint(0, max_x + 1, size=b)
    ys = np.random.randint(0, max_y + 1, size=b)

    row_idx = ys[:, None] + np.arange(dy)[None, :]
    col_idx = xs[:, None] + np.arange(dx)[None, :]
    crops = batch_x[
        np.arange(b)[:, None, None], row_idx[:, :, None], col_idx[:, None, :], :
    ]
    return crops.astype(np.float32, copy=False)


def crop_generator(batches, crop_length):
    """Take as input a Keras ImageGen (Iterator) and generate random
    crops from the image batches generated by the original iterator.
    """
    while True:
        batch_x = next(batches)
        yield _batch_random_crop(batch_x, crop_length)




## === cell 5
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
train_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
assert os.path.exists(train_csv_path), f"Missing: {train_csv_path}"
assert os.path.exists(train_dir), f"Missing: {train_dir}"

train_df = pd.read_csv(train_csv_path)
train_df["label"] = train_df["label"].astype(str)

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.1,
    horizontal_flip=True,
    zoom_range=0.2,
)

train_gen = train_datagen.flow_from_dataframe(
    train_df,
    directory=train_dir,
    x_col="image_id",
    y_col="label",
    target_size=(448, 448),
    batch_size=16,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=0,
)

val_gen = train_datagen.flow_from_dataframe(
    train_df,
    directory=train_dir,
    x_col="image_id",
    y_col="label",
    target_size=(448, 448),
    batch_size=16,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
)

for m in (model_v1, model_v2):
    m.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

using_fallback = any(isinstance(l, tf.keras.layers.Rescaling) for l in model_v1.layers)

if using_fallback:
    steps_per_epoch = max(1, train_gen.samples // train_gen.batch_size)
    val_steps = max(1, val_gen.samples // val_gen.batch_size)
    model_v1.fit(
        train_gen,
        validation_data=val_gen,
        epochs=2,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        verbose=2,
    )
    model_v2.fit(
        train_gen,
        validation_data=val_gen,
        epochs=2,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        verbose=2,
    )




## === cell 6
labels = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4}

test_dir_v1 = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
assert os.path.exists(test_dir_v1), f"Missing: {test_dir_v1}"

test_image_ids = sorted(
    [f for f in os.listdir(test_dir_v1) if f.lower().endswith(".jpg")]
)
test_v1 = pd.DataFrame({"image_id": test_image_ids})

INFER_BATCH_SIZE = 16

AUTOTUNE = tf.data.AUTOTUNE
TEST_SIZE = (512, 512)
CROP_SIZE = 448


def make_base_test_ds_decoded_512(image_ids, base_dir):
    paths = tf.constant([os.path.join(base_dir, x) for x in image_ids])
    idxs = tf.range(tf.shape(paths)[0], dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, idxs))

    opts = tf.data.Options()
    opts.deterministic = True
    ds = ds.with_options(opts)

    @tf.function
    def _read_decode_resize(path, idx):
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
        img = tf.image.resize(img, TEST_SIZE, method="bilinear", antialias=False)
        img = tf.cast(img, tf.float32)
        img.set_shape([TEST_SIZE[0], TEST_SIZE[1], 3])
        return img, idx

    ds = ds.map(_read_decode_resize, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = (
        ds.cache()
    )  # caches decoded/resized tensors to reuse across TTA passes and models
    return ds


@tf.function
def _stateless_random_crop_448_from_512(img_512, seed2):
    max_off = TEST_SIZE[0] - CROP_SIZE  # 64
    offs = tf.random.stateless_uniform(
        [2], seed=seed2, minval=0, maxval=max_off + 1, dtype=tf.int32
    )
    y0 = offs[0]
    x0 = offs[1]
    return tf.slice(img_512, [y0, x0, 0], [CROP_SIZE, CROP_SIZE, 3])


@tf.function
def _augment_like_imagedatagen(img_512, zoom_range, hflip, seed2):
    if hflip:
        do_flip = tf.random.stateless_uniform(
            [], seed=seed2 + tf.constant([11, 17], tf.int32), minval=0.0, maxval=1.0
        )
        img_512 = tf.cond(
            do_flip < 0.5, lambda: tf.image.flip_left_right(img_512), lambda: img_512
        )

    if zoom_range and zoom_range > 0:
        z = tf.random.stateless_uniform(
            [],
            seed=seed2 + tf.constant([23, 29], tf.int32),
            minval=1.0 - float(zoom_range),
            maxval=1.0 + float(zoom_range),
        )
        new_size = tf.cast(tf.round(z * TEST_SIZE[0]), tf.int32)
        img_zoom = tf.image.resize(
            img_512, [new_size, new_size], method="bilinear", antialias=False
        )
        img_512 = tf.image.resize_with_crop_or_pad(img_zoom, TEST_SIZE[0], TEST_SIZE[1])
    return img_512


def make_test_ds_from_base(base_ds_decoded_512, batch_size, pass_id, zoom_range, hflip):
    @tf.function
    def _aug_and_crop(img_512, idx):
        seed2 = tf.stack([tf.constant(0, tf.int32) + pass_id, idx], axis=0)
        img = _augment_like_imagedatagen(
            img_512, zoom_range=zoom_range, hflip=hflip, seed2=seed2
        )
        img = _stateless_random_crop_448_from_512(
            img, seed2=seed2 + tf.constant([101, 131], tf.int32)
        )
        return img

    opts = tf.data.Options()
    opts.deterministic = True
    opts.experimental_deterministic = True
    ds = base_ds_decoded_512.with_options(opts)
    ds = ds.map(_aug_and_crop, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def predict_tta_4passes(model, base_ds_decoded_512, batch_size, zoom_range, hflip):
    pred_sum = None
    for pass_id in range(4):
        ds = make_test_ds_from_base(
            base_ds_decoded_512,
            batch_size=batch_size,
            pass_id=pass_id,
            zoom_range=zoom_range,
            hflip=hflip,
        )
        pred = model.predict(ds, verbose=1 if pass_id == 0 else 0)
        pred_sum = pred if pred_sum is None else (pred_sum + pred)
    return pred_sum / 4.0


base_test_ds = make_base_test_ds_decoded_512(test_image_ids, test_dir_v1)

pred_v1 = predict_tta_4passes(
    model_v1,
    base_test_ds,
    batch_size=INFER_BATCH_SIZE,
    zoom_range=0.4,
    hflip=True,
)
print("pred_v1 shape:", pred_v1.shape)




## === cell 7
test_dir_v2 = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
test_v2 = pd.DataFrame({"image_id": test_image_ids})

pred_v2 = predict_tta_4passes(
    model_v2,
    base_test_ds,
    batch_size=INFER_BATCH_SIZE,
    zoom_range=0.2,
    hflip=True,
)
print("pred_v2 shape:", pred_v2.shape)




## === cell 8
pred_new = pred_v1 + pred_v2
predicted_class_indices_new = np.argmax(pred_new, axis=1)
predictions_new = predicted_class_indices_new.astype(int)

sample_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
sample_sub = pd.read_csv(sample_path)

pred_df = pd.DataFrame(
    {"image_id": test_v2["image_id"].values, "label": predictions_new}
)

sub = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")
sub["label"] = sub["label"].fillna(0).astype(int)

out_path = "/kaggle/working/submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(sub.head())
print("Rows:", len(sub), "Cols:", sub.columns.tolist())
assert out_path.endswith(".csv") and os.path.exists(out_path)
assert list(sub.columns) == ["image_id", "label"]
assert len(sub) == len(sample_sub)
