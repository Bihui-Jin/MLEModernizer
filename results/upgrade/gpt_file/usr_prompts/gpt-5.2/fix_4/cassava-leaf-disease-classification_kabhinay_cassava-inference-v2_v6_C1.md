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

0.8819885161680266

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.61099) has done: 'I fix the immediate runtime/import failure caused by an incompatible protobuf implementation (the `MessageFactory.GetPrototype` error) by forcing TensorFlow to use the pure-Python protobuf backend before importing TensorFlow. Next, I make the SavedModel loading robust: those `/kaggle/input/...` model directories are not present in your provided dataset, so I detect missing model paths and fall back to a simple, deterministic baseline submission (majority class from `train.csv`) to ensure a valid `submission.csv` is always produced end-to-end. This keeps the core ensemble/prediction logic intact when the models exist, but prevents crashes when they don’t. The output always match the required submission format and `.csv` suffix.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

np.random.seed(42)
try:
    tf.random.set_seed(42)
except Exception:
    pass

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TEST_DIR = os.path.join(DATA_DIR, "test_images")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train.csv")

print("TF version:", tf.__version__)
print("tf.keras version:", getattr(keras, "__version__", "unknown"))
print(
    "Test dir exists:",
    os.path.isdir(TEST_DIR),
    "num_files:",
    len(os.listdir(TEST_DIR)) if os.path.isdir(TEST_DIR) else -1,
)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
def acc_gambler(y_true, y_pred):
    y_temp = y_pred[:, 1:]
    return tf.reduce_mean(
        tf.cast(
            tf.equal(tf.argmax(y_temp, axis=1), tf.argmax(y_true, axis=1)), tf.float32
        )
    )


def loss_gambler(label_smoothing=0.0):
    def loss_gamb(y_true, y_pred):
        y_true_s = y_true + (label_smoothing / 2.0) * (1.0 - 2.0 * y_true)
        y_temp = y_pred[:, 1:]
        f0 = y_pred[:, 0]
        sum_y = tf.reduce_sum(y_temp)
        sum_yy = tf.reduce_sum(y_temp * y_temp)
        lamb = (sum_y * sum_y) / (sum_yy + 1e-12)
        log_term = tf.math.log(
            y_temp + tf.expand_dims(f0 / (lamb + 1e-12), axis=1) + 1e-12
        )
        loss = -tf.reduce_mean(tf.reduce_sum(y_true_s * log_term, axis=1))
        return loss

    return loss_gamb




## === cell 3
def _resolve_savedmodel_dir(path):
    if not path or (not os.path.isdir(path)):
        return None

    def has_savedmodel(d):
        return os.path.isfile(os.path.join(d, "saved_model.pb")) or os.path.isfile(
            os.path.join(d, "saved_model.pbtxt")
        )

    if has_savedmodel(path):
        return path

    try:
        for name in sorted(os.listdir(path)):
            d = os.path.join(path, name)
            if os.path.isdir(d) and has_savedmodel(d):
                return d
    except Exception:
        pass

    for root, dirs, files in os.walk(path):
        if "saved_model.pb" in files or "saved_model.pbtxt" in files:
            return root
        if root.count(os.sep) - path.count(os.sep) >= 3:
            dirs[:] = []
    return path


def _build_inference_model_from_savedmodel(savedmodel_dir):
    savedmodel_dir = _resolve_savedmodel_dir(savedmodel_dir)
    if savedmodel_dir is None:
        raise IOError("SavedModel directory not found.")

    endpoints_to_try = ["serving_default", "predict", "call"]
    last_err = None
    for ep in endpoints_to_try:
        try:
            layer = keras.layers.TFSMLayer(savedmodel_dir, call_endpoint=ep)
            inp = keras.Input(shape=(448, 448, 3), dtype=tf.float32)
            out = layer(inp)
            if isinstance(out, dict):
                if "outputs" in out:
                    out = out["outputs"]
                else:
                    out = list(out.values())[0]
            return keras.Model(
                inp, out, name=os.path.basename(savedmodel_dir) + "_" + ep
            )
        except Exception as e:
            last_err = e
    raise RuntimeError(
        "Could not load SavedModel at %s via TFSMLayer. Last error: %r"
        % (savedmodel_dir, last_err)
    )




## === cell 4
MODEL_V1_DIR = "/kaggle/input/only-xception-with-cropping/saved-model-11-0.879"
MODEL_V2_DIR = "/kaggle/input/efficientnet-with-cropping/saved-model-06-0.88"
MODEL_V3_DIR = "/kaggle/input/gambler-s-loss-cassava/saved-model-10-0.843"

resolved_v1 = _resolve_savedmodel_dir(MODEL_V1_DIR)
resolved_v2 = _resolve_savedmodel_dir(MODEL_V2_DIR)
resolved_v3 = _resolve_savedmodel_dir(MODEL_V3_DIR)

print("Model v1 resolved:", resolved_v1)
print("Model v2 resolved:", resolved_v2)
print("Model v3 resolved:", resolved_v3)

models_available = (
    (resolved_v1 is not None)
    and (resolved_v2 is not None)
    and (resolved_v3 is not None)
)
print("All models available:", models_available)

model_v1 = None
model_v2 = None
model_v3 = None

if models_available:
    model_v1 = _build_inference_model_from_savedmodel(MODEL_V1_DIR)
    model_v2 = _build_inference_model_from_savedmodel(MODEL_V2_DIR)
    model_v3 = _build_inference_model_from_savedmodel(MODEL_V3_DIR)

    _ = model_v1(tf.zeros((1, 448, 448, 3), dtype=tf.float32))
    _ = model_v2(tf.zeros((1, 448, 448, 3), dtype=tf.float32))
    _ = model_v3(tf.zeros((1, 448, 448, 3), dtype=tf.float32))

    print("Loaded models OK:")
    print(" - v1 output shape:", model_v1.output_shape)
    print(" - v2 output shape:", model_v2.output_shape)
    print(" - v3 output shape:", model_v3.output_shape)




## === cell 5
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
        batch_crops = np.zeros(
            (batch_x.shape[0], crop_length, crop_length, 3), dtype=batch_x.dtype
        )
        for i in range(batch_x.shape[0]):
            batch_crops[i] = random_crop(batch_x[i], (crop_length, crop_length))
        yield batch_crops




## === cell 6
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_df = sample_sub[["image_id"]].copy()

test_generator = None
steps = None
if models_available:
    test_datagen = ImageDataGenerator()
    test_generator = test_datagen.flow_from_dataframe(
        test_df,
        directory=TEST_DIR,
        x_col="image_id",
        y_col=None,
        target_size=(448, 448),
        batch_size=16,  # keep as in provided script
        class_mode=None,
        shuffle=False,
    )
    steps = int(np.ceil(len(test_df) / float(test_generator.batch_size)))
    print(
        "Num test:",
        len(test_df),
        "batch_size:",
        test_generator.batch_size,
        "steps:",
        steps,
    )



## === cell 7
if models_available:
    pred_v1 = model_v1.predict(test_generator, verbose=1, steps=steps)
    test_generator.reset()
    pred_v2 = model_v2.predict(test_generator, verbose=1, steps=steps)
    test_generator.reset()
    pred_v3 = model_v3.predict(test_generator, verbose=1, steps=steps)

    if pred_v3.shape[-1] == 6:
        pred_v3 = pred_v3[:, 1:]
    elif pred_v3.shape[-1] != 5:
        raise ValueError(
            "Unexpected pred_v3 shape %r (expected last dim 6 or 5)." % (pred_v3.shape,)
        )

    for name, p in [("pred_v1", pred_v1), ("pred_v2", pred_v2), ("pred_v3", pred_v3)]:
        if p.shape[0] != len(test_df):
            raise ValueError(
                "%s rows %d != num test %d" % (name, p.shape[0], len(test_df))
            )
        if p.shape[-1] != 5:
            raise ValueError("%s last dim %d != 5" % (name, p.shape[-1]))

    pred_new = 0.3 * pred_v1 + 0.3 * pred_v2 + 0.4 * pred_v3
    predicted_class_indices_new = np.argmax(pred_new, axis=1).astype(np.int64)
else:
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    majority_label = int(train_df["label"].value_counts().idxmax())
    predicted_class_indices_new = np.full(
        (len(test_df),), majority_label, dtype=np.int64
    )
    print(
        "Models not found; using majority-label baseline. majority_label =",
        majority_label,
    )

submission = pd.DataFrame(
    {"image_id": test_df["image_id"].values, "label": predicted_class_indices_new}
)

assert submission.shape[0] == sample_sub.shape[0]
assert list(submission.columns) == ["image_id", "label"]

out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(submission.head())
