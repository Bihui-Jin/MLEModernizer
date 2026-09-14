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
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ["PYTHONHASHSEED"] = "42"
random.seed(42)
np.random.seed(42)
try:
    tf.random.set_seed(42)
except Exception:
    pass

print("TF:", getattr(tf, "__version__", "unknown"))
print("Eager:", tf.executing_eagerly())

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.run_functions_eagerly(False)
except Exception:
    pass




## === cell 1
"""Robust Bi-Tempered Logistic Loss Based on Bregman Divergences.

Source: https://bit.ly/3jSol8T
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


def tempered_sigmoid(activations, t, num_iters=5):
    """Tempered sigmoid function."""
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
    """Tempered softmax function."""
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
    """Bi-Tempered binary logistic loss."""
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
    """Sparse Bi-Tempered Logistic Loss with custom gradient."""
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
import tensorflow.keras.backend as K


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
def load_savedmodel_as_keras_model(
    savedmodel_dir,
    input_shape=(448, 448, 3),
    dtype=tf.float32,
    call_endpoint="serving_default",
):
    if hasattr(keras.layers, "TFSMLayer"):
        layer = keras.layers.TFSMLayer(savedmodel_dir, call_endpoint=call_endpoint)
        inp = keras.Input(shape=input_shape, dtype=dtype, name="image")
        out = layer(inp)
        if isinstance(out, dict):
            out = out[sorted(out.keys())[0]]
        model = keras.Model(inp, out, name=os.path.basename(savedmodel_dir.rstrip("/")))
        return model
    else:
        return keras.models.load_model(savedmodel_dir)


def robust_load(savedmodel_dir, input_shape=(448, 448, 3)):
    if not (
        os.path.isdir(savedmodel_dir)
        and (
            os.path.exists(os.path.join(savedmodel_dir, "saved_model.pb"))
            or os.path.exists(os.path.join(savedmodel_dir, "saved_model.pbtxt"))
        )
    ):
        raise OSError(
            "SavedModel file does not exist at: %s/{saved_model.pbtxt|saved_model.pb}"
            % savedmodel_dir
        )

    last_err = None
    for ep in ["serving_default", "serve", "predict", "call"]:
        try:
            m = load_savedmodel_as_keras_model(
                savedmodel_dir, input_shape=input_shape, call_endpoint=ep
            )
            _ = m(tf.zeros((1,) + input_shape, dtype=tf.float32))
            print(
                "Loaded:",
                savedmodel_dir,
                "endpoint:",
                ep,
                "output_shape:",
                m.output_shape,
            )
            return m
        except Exception as e:
            last_err = e
    raise RuntimeError(
        "Failed to load SavedModel at %s. Last error: %r" % (savedmodel_dir, last_err)
    )


def build_fallback_classifier(input_shape=(448, 448, 3), n_classes=5):
    inp = keras.Input(shape=input_shape)
    x = keras.layers.Rescaling(1.0 / 255.0)(inp)
    x = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    out = keras.layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


model_path_v1 = "/kaggle/input/only-xception-with-cropping/saved-model-11-0.879"
model_path_v2 = "/kaggle/input/gambler-s-loss-cassava/saved-model-05-0.860"
model_path_v3 = (
    "/kaggle/input/bitempered-loss-only-xception-with-cropping/saved-model-15-0.839"
)

model_v1 = None
model_v2 = None
model_v3 = None

for name, path in [("v1", model_path_v1), ("v2", model_path_v2), ("v3", model_path_v3)]:
    try:
        m = robust_load(path, input_shape=(448, 448, 3))
        if name == "v1":
            model_v1 = m
        elif name == "v2":
            model_v2 = m
        else:
            model_v3 = m
    except Exception as e:
        print(
            "Model",
            name,
            "not available, will use fallback if needed. Reason:",
            repr(e),
        )




## === cell 4
def _random_crops_batch(imgs, crop_length):
    imgs = np.asarray(imgs)
    b, h, w, c = imgs.shape
    dy = dx = int(crop_length)
    if h == dy and w == dx:
        return imgs

    xs = np.random.randint(0, w - dx + 1, size=b)
    ys = np.random.randint(0, h - dy + 1, size=b)

    y_idx = ys[:, None] + np.arange(dy)[None, :]
    x_idx = xs[:, None] + np.arange(dx)[None, :]

    out = imgs[np.arange(b)[:, None, None], y_idx[:, :, None], x_idx[:, None, :], :]
    return out


def crop_generator(batches, crop_length):
    """Generate random crops from batches generated by an original iterator."""
    while True:
        batch_x = next(batches)
        yield _random_crops_batch(batch_x, crop_length)




## === cell 5
DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TRAIN_DIR = os.path.join(DATA_DIR, "train_images")
TEST_DIR = os.path.join(DATA_DIR, "test_images")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")

sub_df = pd.read_csv(SAMPLE_SUB)
test_v = pd.DataFrame({"image_id": sub_df["image_id"].astype(str).values})
print("Test rows:", len(test_v), "Example:", test_v["image_id"].iloc[0])

train_df = pd.read_csv(TRAIN_CSV)
train_df["image_id"] = train_df["image_id"].astype(str)
train_df["label"] = train_df["label"].astype(int)
print(
    "Train rows:", len(train_df), "labels:", sorted(train_df["label"].unique().tolist())
)

PRED_BATCH_SIZE = 32

_TEST_DATAGEN = ImageDataGenerator(zoom_range=0.4, horizontal_flip=True)


def _make_base_test_iter():
    return _TEST_DATAGEN.flow_from_dataframe(
        test_v,
        directory=TEST_DIR,
        x_col="image_id",
        target_size=(512, 512),
        batch_size=PRED_BATCH_SIZE,
        class_mode=None,
        shuffle=False,
    )


FINAL_SIZE = 448


def _build_test_base_dataset_with_index(image_ids, batch_size):
    img_paths = tf.strings.join([tf.constant(TEST_DIR + "/"), tf.constant(image_ids)])
    idxs = tf.range(tf.shape(img_paths)[0], dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((idxs, img_paths))

    def _decode_resize(i, path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3, dct_method="INTEGER_FAST")
        img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
        img = tf.image.resize(
            img, [FINAL_SIZE, FINAL_SIZE], method=tf.image.ResizeMethod.BILINEAR
        )
        return i, img

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)
    ds = ds.map(_decode_resize, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(2)
    return ds


_MAX_SCALE = 1.4
_MIN_SCALE = 0.6


@tf.function(reduce_retracing=True)
def _augment_batch_stateless(batch_imgs, base_index, seed_offset):
    batch_imgs = tf.convert_to_tensor(batch_imgs, dtype=tf.float32)
    bs = tf.shape(batch_imgs)[0]

    idx = tf.range(bs, dtype=tf.int32) + tf.cast(base_index, tf.int32)
    s = idx + tf.cast(seed_offset, tf.int32) * 100000

    seeds_flip = tf.stack([tf.fill([bs], 1234), s + 17], axis=1)
    flip_rnd = tf.random.stateless_uniform([bs], seed=seeds_flip)
    do_flip = flip_rnd < 0.5
    flipped = tf.where(
        do_flip[:, None, None, None], tf.reverse(batch_imgs, axis=[2]), batch_imgs
    )

    seeds_zoom = tf.stack([tf.fill([bs], 42), s], axis=1)
    scale = tf.random.stateless_uniform(
        [bs], seed=seeds_zoom, minval=_MIN_SCALE, maxval=_MAX_SCALE
    )

    inv = 1.0 / scale
    y1 = (1.0 - inv) * 0.5
    x1 = (1.0 - inv) * 0.5
    y2 = y1 + inv
    x2 = x1 + inv
    boxes = tf.stack([y1, x1, y2, x2], axis=1)
    box_ind = tf.range(bs, dtype=tf.int32)

    out = tf.image.crop_and_resize(
        flipped,
        boxes=boxes,
        box_ind=box_ind,
        crop_size=[FINAL_SIZE, FINAL_SIZE],
        method="bilinear",
        extrapolation_value=0.0,
    )
    return out


def _build_test_dataset_from_indexed_base(
    indexed_base_batched_ds, batch_size, seed_offset=0
):
    def _map_batch(batch_indices, batch_imgs):
        base_index = tf.reduce_min(tf.cast(batch_indices, tf.int32))
        return _augment_batch_stateless(
            batch_imgs, base_index=base_index, seed_offset=seed_offset
        )

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = indexed_base_batched_ds.with_options(opts)
    ds = ds.map(_map_batch, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(2)
    return ds


def tta_predict_multi(models_dict, n_tta=5):
    n = len(test_v)
    image_ids = test_v["image_id"].astype(str).values

    base_ds = _build_test_base_dataset_with_index(image_ids, batch_size=PRED_BATCH_SIZE)

    sums = {k: None for k in models_dict}
    model_items = [(k, models_dict[k]) for k in sorted(models_dict.keys())]

    for t in range(n_tta):
        ds = _build_test_dataset_from_indexed_base(
            base_ds, batch_size=PRED_BATCH_SIZE, seed_offset=t
        )
        for k, model in model_items:
            cur_np = model.predict(ds, verbose=0)
            if cur_np.shape[0] > n:
                cur_np = cur_np[:n]
            if sums[k] is None:
                sums[k] = cur_np.astype(np.float32, copy=False)
            else:
                sums[k] += cur_np.astype(np.float32, copy=False)

    avgs = {k: (sums[k] / float(n_tta)) for k in sums}
    return avgs




## === cell 6
if (model_v1 is None) and (model_v2 is None) and (model_v3 is None):
    print("No external models found. Training a fallback model locally...")

    train_gen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=15,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.10,
        horizontal_flip=True,
        validation_split=0.1,
    )

    train_it = train_gen.flow_from_dataframe(
        train_df,
        directory=TRAIN_DIR,
        x_col="image_id",
        y_col="label",
        target_size=(448, 448),
        batch_size=16,
        class_mode="raw",
        subset="training",
        shuffle=True,
        seed=42,
    )
    val_it = train_gen.flow_from_dataframe(
        train_df,
        directory=TRAIN_DIR,
        x_col="image_id",
        y_col="label",
        target_size=(448, 448),
        batch_size=16,
        class_mode="raw",
        subset="validation",
        shuffle=False,
        seed=42,
    )

    fallback = build_fallback_classifier(input_shape=(448, 448, 3), n_classes=5)
    fallback.fit(
        train_it,
        validation_data=val_it,
        epochs=3,
        verbose=1,
    )

    model_v1 = fallback
    model_v2 = fallback
    model_v3 = fallback




## === cell 7
models_to_run = {}
if model_v1 is not None:
    models_to_run["v1"] = model_v1
if model_v2 is not None:
    models_to_run["v2"] = model_v2
if model_v3 is not None:
    models_to_run["v3"] = model_v3
if not models_to_run:
    raise RuntimeError("No models available for prediction.")

tta_out = tta_predict_multi(models_to_run, n_tta=5)

pred_v1 = tta_out.get("v1", None)
pred_v2 = None
pred_v3 = tta_out.get("v3", None)

if "v2" in tta_out:
    temp_v2 = np.asarray(tta_out["v2"])
    if temp_v2.ndim == 2 and temp_v2.shape[1] == 6:
        pred_v2 = temp_v2[:, 1:]
    else:
        pred_v2 = temp_v2

preds = []
weights = []
if pred_v1 is not None:
    preds.append(pred_v1)
    weights.append(0.40)
if pred_v2 is not None:
    preds.append(pred_v2)
    weights.append(0.30)
if pred_v3 is not None:
    preds.append(pred_v3)
    weights.append(0.30)

if len(preds) == 0:
    raise RuntimeError("No models available for prediction.")

weights = np.asarray(weights, dtype=np.float32)
weights = weights / weights.sum()

pred_new = np.zeros_like(preds[0], dtype=np.float32)
for w, p in zip(weights, preds):
    pred_new += w * p.astype(np.float32)

predicted_class_indices_new = np.argmax(pred_new, axis=1).astype(int)

results_new = pd.DataFrame(
    {"image_id": test_v["image_id"].values, "label": predicted_class_indices_new}
)
assert results_new.shape[0] == sub_df.shape[0]
assert list(results_new.columns) == ["image_id", "label"]

out_path = "/kaggle/working/submission.csv"
results_new.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(results_new.head())
print(results_new["label"].value_counts().sort_index())
