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

0.8841039588999697

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import tensorflow.compat.v1 as tf1

tf1.disable_eager_execution()

import keras
from keras import layers, models

np.random.seed(42)
try:
    tf1.set_random_seed(42)
except Exception:
    pass

print("Using tf.compat.v1 version:", tf1.__version__)
print("Keras version:", keras.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
"""Robust Bi-Tempered Logistic Loss Based on Bregman Divergences.

Source: https://bit.ly/3jSol8T
"""

import functools


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
    mu = tf1.reduce_max(activations, -1, keepdims=True)
    normalized_activations_step_0 = activations - mu
    shape_normalized_activations = tf1.shape(normalized_activations_step_0)

    def iter_body(i, normalized_activations):
        logt_partition = tf1.reduce_sum(
            exp_t(normalized_activations, t), -1, keepdims=True
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
        exp_t(normalized_activations_t, t), -1, keepdims=True
    )
    return -log_t(1.0 / logt_partition, t) + mu


def compute_normalization_binary_search(activations, t, num_iters=10):
    """Returns the normalization value for each example (t < 1.0)."""
    mu = tf1.reduce_max(activations, -1, keepdims=True)
    normalized_activations = activations - mu
    shape_activations = tf1.shape(activations)
    effective_dim = tf1.cast(
        tf1.reduce_sum(
            tf1.cast(tf1.greater(normalized_activations, -1.0 / (1.0 - t)), tf1.int32),
            -1,
            keepdims=True,
        ),
        tf1.float32,
    )
    shape_partition = tf1.concat([shape_activations[:-1], [1]], 0)
    lower = tf1.zeros(shape_partition)
    upper = -log_t(1.0 / effective_dim, t) * tf1.ones(shape_partition)

    def iter_body(i, lower, upper):
        logt_partition = (upper + lower) / 2.0
        sum_probs = tf1.reduce_sum(
            exp_t(normalized_activations - logt_partition, t), -1, keepdims=True
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
            tf1.reduce_sum(tf1.exp(internal_activations), -1, keepdims=True)
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
        lambda: tf1.log(tf1.reduce_sum(tf1.exp(activations), -1, keepdims=True)),
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
                        delta_probs_times_forget_factor, -1, keepdims=True
                    )
                    escorts = tf1.pow(probabilities, t2)
                    escorts = escorts / tf1.reduce_sum(escorts, -1, keepdims=True)
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
                        delta_probs_times_forget_factor, -1, keepdims=True
                    )
                    escorts = tf1.pow(probabilities, t2)
                    escorts = escorts / tf1.reduce_sum(escorts, -1, keepdims=True)
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
import tensorflow.keras.backend as K  # uses TF backend available in the runtime


def acc_gambler(y_true, y_pred):
    y_temp = y_pred[:, 1:]
    count = tf1.constant((0,))
    for i in range(len(y_true)):
        if tf1.math.argmax(y_temp[i]) == tf1.math.argmax(y_true[i]):
            count = tf1.math.add(count, 1)
    return float(count) / float(len(y_true))


def loss_gambler(label_smoothing=0.0):
    def loss_gamb(y_true, y_pred):
        y_true = tf1.math.add(
            y_true,
            tf1.math.add(
                tf1.math.multiply(
                    label_smoothing / 2.0, tf1.math.add(1.0, -1 * y_true)
                ),
                tf1.math.multiply(-1 * label_smoothing / 2.0, y_true),
            ),
        )
        y_temp = y_pred[:, 1:]
        f0 = y_pred[:, 0]
        lamb = tf1.math.divide(
            tf1.math.multiply(K.sum(y_temp), K.sum(y_temp)),
            K.sum(tf1.math.multiply(y_temp, y_temp)),
        )
        loss = tf1.constant((0.0,))
        for i in range(len(y_true[0])):
            temp = tf1.constant((0.0,))
            loss = tf1.math.add(
                loss,
                tf1.math.add(
                    temp,
                    (
                        -1.0
                        * (1 / float(len(y_true)))
                        * tf1.math.multiply(
                            y_true[:, i], K.log(y_temp[:, i] + f0 / lamb)
                        )
                    ),
                ),
            )
        return tf1.math.reduce_sum(loss)

    return loss_gamb




## === cell 3


def _find_savedmodel_dir(root_dir):
    """Return a directory path that directly contains saved_model.pb(.txt), or None."""
    if not root_dir or not os.path.isdir(root_dir):
        return None
    if os.path.isfile(os.path.join(root_dir, "saved_model.pb")) or os.path.isfile(
        os.path.join(root_dir, "saved_model.pbtxt")
    ):
        return root_dir
    for name in sorted(os.listdir(root_dir)):
        p = os.path.join(root_dir, name)
        if os.path.isdir(p) and (
            os.path.isfile(os.path.join(p, "saved_model.pb"))
            or os.path.isfile(os.path.join(p, "saved_model.pbtxt"))
        ):
            return p
    return None


def _resolve_model_path(preferred_candidates):
    for cand in preferred_candidates:
        p = _find_savedmodel_dir(cand)
        if p is not None:
            return p
    return None


def load_savedmodel_as_keras_model(
    savedmodel_dir, input_shape=(448, 448, 3), call_endpoint="serving_default"
):
    """
    Wrap a TF SavedModel directory as a Keras Model using TFSMLayer.
    Assumes a single input tensor and a single output tensor.
    """
    tfsm_layer = keras.layers.TFSMLayer(savedmodel_dir, call_endpoint=call_endpoint)
    inp = keras.Input(shape=input_shape)
    out = tfsm_layer(inp)
    if isinstance(out, dict):
        out = out[sorted(out.keys())[0]]
    return keras.Model(inp, out)


BASE_INPUT = "/kaggle/input"

MODEL_V1_CANDIDATES = [
    os.path.join(BASE_INPUT, "only-xception-with-cropping", "saved-model-12-0.875"),
    os.path.join(BASE_INPUT, "only-xception-with-cropping"),
]
MODEL_V3_CANDIDATES = [
    os.path.join(BASE_INPUT, "gambler-s-loss-cassava", "saved-model-11-0.882"),
    os.path.join(BASE_INPUT, "gambler-s-loss-cassava"),
]
MODEL_V4_CANDIDATES = [
    os.path.join(BASE_INPUT, "efficientnet-with-cropping-v2", "saved-model-08-0.887"),
    os.path.join(BASE_INPUT, "efficientnet-with-cropping-v2"),
]

MODEL_V1_DIR = _resolve_model_path(MODEL_V1_CANDIDATES)
MODEL_V3_DIR = _resolve_model_path(MODEL_V3_CANDIDATES)
MODEL_V4_DIR = _resolve_model_path(MODEL_V4_CANDIDATES)

if MODEL_V1_DIR is None or MODEL_V3_DIR is None or MODEL_V4_DIR is None:
    found = []
    for d in sorted(os.listdir(BASE_INPUT)):
        root = os.path.join(BASE_INPUT, d)
        p = _find_savedmodel_dir(root)
        if p is not None:
            found.append((d, p))

    def _pick(keyword):
        for name, p in found:
            if keyword in name.lower():
                return p
        return None

    if MODEL_V1_DIR is None:
        MODEL_V1_DIR = _pick("xception")
    if MODEL_V3_DIR is None:
        MODEL_V3_DIR = _pick("gambler")
    if MODEL_V4_DIR is None:
        MODEL_V4_DIR = _pick("efficientnet")

print("Resolved model dirs:")
print("MODEL_V1_DIR:", MODEL_V1_DIR)
print("MODEL_V3_DIR:", MODEL_V3_DIR)
print("MODEL_V4_DIR:", MODEL_V4_DIR)

if MODEL_V1_DIR is None or MODEL_V3_DIR is None or MODEL_V4_DIR is None:
    raise OSError(
        "Could not resolve all SavedModel directories under /kaggle/input. "
        "Please ensure the three model datasets are attached to this notebook."
    )

model_v1 = load_savedmodel_as_keras_model(
    MODEL_V1_DIR, input_shape=(448, 448, 3), call_endpoint="serving_default"
)
model_v3 = load_savedmodel_as_keras_model(
    MODEL_V3_DIR, input_shape=(448, 448, 3), call_endpoint="serving_default"
)
model_v4 = load_savedmodel_as_keras_model(
    MODEL_V4_DIR, input_shape=(448, 448, 3), call_endpoint="serving_default"
)

print("Loaded models ok.")



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/651128709.py in <cell line: 0>()
     95 
     96 if MODEL_V1_DIR is None or MODEL_V3_DIR is None or MODEL_V4_DIR is None:
---> 97     raise OSError(
     98         "Could not resolve all SavedModel directories under /kaggle/input. "
     99         "Please ensure the three model datasets are attached to this notebook."

OSError: Could not resolve all SavedModel directories under /kaggle/input. Please ensure the three model datasets are attached to this notebook.

## === cell 4
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
test_dir = os.path.join(DATA_DIR, "test_images")

sample_path = os.path.join(DATA_DIR, "sample_submission.csv")
sample_sub = pd.read_csv(sample_path)
test_df = sample_sub[["image_id"]].copy()

test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow_from_dataframe(
    test_df,
    directory=test_dir,
    x_col="image_id",
    target_size=(448, 448),
    batch_size=1,
    class_mode=None,
    shuffle=False,
)
test_generator.reset()
print("Test samples:", len(test_df))



## === cell 5
steps = len(test_df)

test_generator.reset()
pred_v1 = model_v1.predict(test_generator, verbose=1, steps=steps)

test_generator.reset()
pred_v3 = model_v3.predict(test_generator, verbose=1, steps=steps)

test_generator.reset()
pred_v4 = model_v4.predict(test_generator, verbose=1, steps=steps)

pred_v1 = np.asarray(pred_v1)
pred_v3 = np.asarray(pred_v3)
pred_v4 = np.asarray(pred_v4)

if pred_v3.ndim == 2 and pred_v3.shape[1] == 6:
    pred_v3 = pred_v3[:, 1:]

print("Shapes:", pred_v1.shape, pred_v3.shape, pred_v4.shape)

if not (
    pred_v1.shape[0] == steps
    and pred_v3.shape[0] == steps
    and pred_v4.shape[0] == steps
):
    raise ValueError("Prediction row count mismatch with test set length.")

if not (pred_v1.ndim == 2 and pred_v3.ndim == 2 and pred_v4.ndim == 2):
    raise ValueError("Predictions must be 2D arrays.")

if not (pred_v1.shape[1] == 5 and pred_v3.shape[1] == 5 and pred_v4.shape[1] == 5):
    raise ValueError(
        "Each model must output 5 class scores/probabilities after normalization."
    )



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/202233290.py in <cell line: 0>()
      3 
      4 test_generator.reset()
----> 5 pred_v1 = model_v1.predict(test_generator, verbose=1, steps=steps)
      6 
      7 test_generator.reset()

NameError: name 'model_v1' is not defined

## === cell 6
pred_new = 0.33 * pred_v1 + 0.33 * pred_v3 + 0.34 * pred_v4
predicted_class_indices_new = np.argmax(pred_new, axis=1).astype(int)

submission = sample_sub.copy()
submission["label"] = predicted_class_indices_new.astype(int)

out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission.head())
print("Submission shape:", submission.shape)
print("Label value counts:\n", submission["label"].value_counts().sort_index())

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3558001598.py in <cell line: 0>()
      1 # Ensemble as in original code (weights unchanged).
----> 2 pred_new = 0.33 * pred_v1 + 0.33 * pred_v3 + 0.34 * pred_v4
      3 predicted_class_indices_new = np.argmax(pred_new, axis=1).astype(int)
      4 
      5 # Build submission matching sample_submission order exactly.

NameError: name 'pred_v1' is not defined
