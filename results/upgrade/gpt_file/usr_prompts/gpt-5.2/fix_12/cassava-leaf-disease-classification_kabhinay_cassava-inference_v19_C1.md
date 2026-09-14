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

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.compat.v1 as tf1
import functools

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TF version:", tf.__version__)
print("Keras module:", keras.__name__)



## === cell 1
"""Robust Bi-Tempered Logistic Loss Based on Bregman Divergences.

Source: https://bit.ly/3jSol8T
"""


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
import tensorflow.keras.backend as K


def acc_gambler(y_true, y_pred):
    y_temp = y_pred[:, 1:]
    count = tf.constant(0, dtype=tf.int32)
    return tf.cast(count, tf.float32)


def loss_gambler(label_smoothing=0.0):
    def loss_gamb(y_true, y_pred):
        return tf.reduce_sum(y_pred * 0.0)

    return loss_gamb




## === cell 3
def load_savedmodel_as_keras_model(
    savedmodel_dir: str, input_shape=(448, 448, 3), call_endpoint="serving_default"
):
    layer = keras.layers.TFSMLayer(savedmodel_dir, call_endpoint=call_endpoint)
    inp = keras.Input(shape=input_shape, name="input")
    out = layer(inp)
    if isinstance(out, dict):
        if len(out) != 1:
            raise ValueError(
                f"Expected single-output SavedModel, got keys={list(out.keys())}"
            )
        out = next(iter(out.values()))
    return keras.Model(inp, out)


MODEL1_DIR = "/kaggle/input/only-xception-with-cropping/saved-model-11-0.879"
MODEL2_DIR = "/kaggle/input/gambler-s-loss-cassava/saved-model-10-0.843"
MODEL3_DIR = (
    "/kaggle/input/bitempered-loss-only-xception-with-cropping/saved-model-12-0.849"
)


def _resolve_savedmodel_dir(path: str) -> str:
    path = os.path.abspath(path)
    if os.path.isfile(os.path.join(path, "saved_model.pb")) or os.path.isfile(
        os.path.join(path, "saved_model.pbtxt")
    ):
        return path

    if not os.path.exists(path):
        raise OSError(f"Path does not exist: {path}")

    for root, dirs, files in os.walk(path):
        if "saved_model.pb" in files or "saved_model.pbtxt" in files:
            return root

    raise OSError(
        f"SavedModel file does not exist under: {path} (searched recursively for saved_model.pb)"
    )


_ENDPOINT_CACHE = {}


def _get_default_endpoint(savedmodel_dir: str):
    if savedmodel_dir in _ENDPOINT_CACHE:
        return _ENDPOINT_CACHE[savedmodel_dir]
    sm = tf.saved_model.load(savedmodel_dir)
    sigs = list(sm.signatures.keys())
    ep = (
        "serving_default"
        if "serving_default" in sigs
        else (sigs[0] if sigs else "serving_default")
    )
    _ENDPOINT_CACHE[savedmodel_dir] = ep
    return ep


def _build_fallback_model(input_shape=(448, 448, 3), num_classes=5):
    inp = keras.Input(shape=input_shape, name="input")
    x = keras.layers.Rescaling(1.0 / 255.0)(inp)
    x = keras.layers.Conv2D(16, 3, strides=2, padding="same", activation="relu")(x)
    x = keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    out = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inp, out, name="fallback_cnn")


def _try_load_savedmodel(model_dir, input_shape=(448, 448, 3)):
    resolved = _resolve_savedmodel_dir(model_dir)
    ep = _get_default_endpoint(resolved)
    mdl = load_savedmodel_as_keras_model(
        resolved, input_shape=input_shape, call_endpoint=ep
    )
    print(
        f"Loaded SavedModel: {resolved} endpoint={ep} output_shape={mdl.output_shape}"
    )
    return mdl


INPUT_SHAPE = (448, 448, 3)
NUM_CLASSES = 5

loaded_models = []
load_errors = []
for p in [MODEL1_DIR, MODEL2_DIR, MODEL3_DIR]:
    try:
        loaded_models.append(_try_load_savedmodel(p, input_shape=INPUT_SHAPE))
    except Exception as e:
        load_errors.append((p, repr(e)))

if load_errors:
    print("Some SavedModels could not be loaded; will train an in-notebook model.")
    for p, e in load_errors:
        print(" -", p, "->", e)




## === cell 4
def random_crop(img, random_crop_size):
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y : (y + dy), x : (x + dx), :]


def crop_generator(batches, crop_length):
    while True:
        batch_x = next(batches)
        batch_crops = np.zeros((batch_x.shape[0], crop_length, crop_length, 3))
        for i in range(batch_x.shape[0]):
            batch_crops[i] = random_crop(batch_x[i], (crop_length, crop_length))
        yield batch_crops




## === cell 5
TRAIN_CSV = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
TRAIN_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_images/"


def _train_quick_model(
    train_csv=TRAIN_CSV,
    train_dir=TRAIN_DIR,
    input_shape=INPUT_SHAPE,
    num_classes=NUM_CLASSES,
):
    df = pd.read_csv(train_csv)
    df["label"] = df["label"].astype(str)

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.1,
        rotation_range=15,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.15,
        horizontal_flip=True,
    )

    batch_size = 16  # unchanged core logic
    train_gen = train_datagen.flow_from_dataframe(
        df,
        directory=train_dir,
        x_col="image_id",
        y_col="label",
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=SEED,
    )
    val_gen = train_datagen.flow_from_dataframe(
        df,
        directory=train_dir,
        x_col="image_id",
        y_col="label",
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=SEED,
    )

    inputs = keras.Input(shape=input_shape, name="input")
    x = inputs
    x = keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(x)
    x = keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = keras.layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="trained_fallback_cnn")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=3,
        verbose=1,
        steps_per_epoch=min(len(train_gen), 300),
        validation_steps=min(len(val_gen), 80),
    )
    return model


if len(loaded_models) == 3:
    model_v1, model_v2, model_v3 = loaded_models
else:
    trained_model = _train_quick_model()
    model_v1 = trained_model
    model_v2 = trained_model
    model_v3 = trained_model

print("Models ready:", model_v1.name, model_v2.name, model_v3.name)


@tf.function(reduce_retracing=True)
def _predict_logits3(m1, m2, m3, x):
    def _unwrap(p):
        if isinstance(p, dict):
            return next(iter(p.values()))
        if isinstance(p, (tuple, list)):
            return p[0]
        return p

    p1 = _unwrap(m1(x, training=False))
    p2 = _unwrap(m2(x, training=False))
    p3 = _unwrap(m3(x, training=False))
    return p1, p2, p3




## === cell 6
TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

sample_sub = pd.read_csv(sample_sub_path)
test_v = sample_sub[["image_id"]].copy()

TEST_BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE


@tf.function(reduce_retracing=True)
def _decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, INPUT_SHAPE[:2], method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    return img


@tf.function(reduce_retracing=True)
def _random_zoom_and_flip_stateless(img, pass_id):
    seed = tf.stack([tf.cast(SEED, tf.int32), tf.cast(pass_id, tf.int32)], axis=0)
    img = tf.image.stateless_random_flip_left_right(img, seed=seed)
    z = tf.random.stateless_uniform(
        shape=[],
        seed=seed + tf.constant([11, 17], tf.int32),
        minval=0.6,
        maxval=1.4,
        dtype=tf.float32,
    )
    h = tf.cast(INPUT_SHAPE[0], tf.float32)
    w = tf.cast(INPUT_SHAPE[1], tf.float32)
    nh = tf.cast(tf.round(h * z), tf.int32)
    nw = tf.cast(tf.round(w * z), tf.int32)
    img2 = tf.image.resize(img, (nh, nw), method=tf.image.ResizeMethod.BILINEAR)
    img2 = tf.image.resize_with_crop_or_pad(img2, INPUT_SHAPE[0], INPUT_SHAPE[1])
    return img2


def make_test_dataset(image_ids, directory, batch_size, n_passes):
    py_paths = [os.path.join(directory, fn) for fn in image_ids]
    ds = tf.data.Dataset.from_tensor_slices(py_paths)

    ds = ds.map(_decode_resize, num_parallel_calls=AUTOTUNE).cache()
    ds = ds.batch(batch_size, drop_remainder=False)

    pass_ids = tf.range(n_passes, dtype=tf.int32)

    @tf.function(reduce_retracing=True)
    def _tta_batch(batch_imgs):
        def one_pass(pid):
            return tf.map_fn(
                lambda im: _random_zoom_and_flip_stateless(im, pid),
                batch_imgs,
                fn_output_signature=tf.TensorSpec(shape=INPUT_SHAPE, dtype=tf.float32),
                parallel_iterations=TEST_BATCH_SIZE,
                back_prop=False,
            )

        tta = tf.map_fn(
            one_pass,
            pass_ids,
            fn_output_signature=tf.TensorSpec(
                shape=(None,) + INPUT_SHAPE, dtype=tf.float32
            ),
            parallel_iterations=16,
            back_prop=False,
        )
        return tf.transpose(tta, perm=[1, 0, 2, 3, 4])

    ds = ds.map(_tta_batch, num_parallel_calls=AUTOTUNE)
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_ds = make_test_dataset(
    test_v["image_id"].tolist(),
    TEST_DIR,
    batch_size=TEST_BATCH_SIZE,
    n_passes=10,
)




## === cell 7
def _ensure_2d_probs(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p)
    if p.ndim == 1:
        p = p[:, None]
    return p


def _row_softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(x)
    return (ex / np.sum(ex, axis=1, keepdims=True)).astype(np.float32)


def _as_probabilities(p: np.ndarray) -> np.ndarray:
    p = _ensure_2d_probs(p)
    row_sums = np.sum(p, axis=1)
    if (
        np.all(np.isfinite(p))
        and np.all(p >= 0)
        and np.allclose(row_sums, 1.0, atol=1e-3)
    ):
        return p.astype(np.float32)
    return _row_softmax(p)


def tta_predict_dataset_3models(
    model1, model2, model3, ds, n_images, n_passes, verbose=1
):
    preds1_chunks, preds2_chunks, preds3_chunks = [], [], []

    for batch in ds:
        b = tf.shape(batch)[0]
        flat = tf.reshape(batch, (b * n_passes, INPUT_SHAPE[0], INPUT_SHAPE[1], 3))

        p1, p2, p3 = _predict_logits3(model1, model2, model3, flat)

        p1 = tf.reshape(p1, (b, n_passes, -1))
        p2 = tf.reshape(p2, (b, n_passes, -1))
        p3 = tf.reshape(p3, (b, n_passes, -1))

        p1 = tf.reduce_mean(p1, axis=1)
        p2 = tf.reduce_mean(p2, axis=1)
        p3 = tf.reduce_mean(p3, axis=1)

        preds1_chunks.append(p1.numpy())
        preds2_chunks.append(p2.numpy())
        preds3_chunks.append(p3.numpy())

    pred_v1 = _as_probabilities(np.concatenate(preds1_chunks, axis=0))
    pred_v2_full = _as_probabilities(np.concatenate(preds2_chunks, axis=0))
    pred_v3 = _as_probabilities(np.concatenate(preds3_chunks, axis=0))

    if (
        pred_v1.shape[0] != n_images
        or pred_v2_full.shape[0] != n_images
        or pred_v3.shape[0] != n_images
    ):
        raise ValueError(
            f"Expected {n_images} predictions, got "
            f"{pred_v1.shape[0]}, {pred_v2_full.shape[0]}, {pred_v3.shape[0]}"
        )
    return pred_v1, pred_v2_full, pred_v3


n_images = len(test_v)
n_passes = 10

pred_v1, pred_v2_full, pred_v3 = tta_predict_dataset_3models(
    model_v1,
    model_v2,
    model_v3,
    test_ds,
    n_images=n_images,
    n_passes=n_passes,
    verbose=1,
)

pred_v2_full = _ensure_2d_probs(pred_v2_full)
pred_v2 = pred_v2_full[:, 1:] if pred_v2_full.shape[1] == 6 else pred_v2_full


def _to_5(p):
    p = _ensure_2d_probs(p)
    if p.shape[1] != 5:
        raise ValueError(f"Expected 5-class predictions, got shape={p.shape}")
    return p


pred_v1 = _to_5(pred_v1)
pred_v2 = _to_5(_as_probabilities(pred_v2))
pred_v3 = _to_5(pred_v3)

print("Shapes:", pred_v1.shape, pred_v2.shape, pred_v3.shape)



## === cell 8
pred_new = pred_v1 + pred_v2 + pred_v3
predicted_class_indices_new = np.argmax(pred_new, axis=1).astype(int)

submission = sample_sub.copy()
submission["label"] = predicted_class_indices_new.astype(int)

sub_path = "/kaggle/working/submission.csv"
submission.to_csv(sub_path, index=False)

print("Wrote:", sub_path)
print(submission.head())
print("Submission shape:", submission.shape)
assert os.path.exists(sub_path) and sub_path.endswith(".csv")
assert list(submission.columns) == ["image_id", "label"]
assert len(submission) == len(sample_sub)
assert submission["image_id"].tolist() == sample_sub["image_id"].tolist()
