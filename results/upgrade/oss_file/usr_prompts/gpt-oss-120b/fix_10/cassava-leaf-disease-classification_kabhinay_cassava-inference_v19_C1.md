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

import google.protobuf.message_factory as _mf

if not hasattr(_mf.MessageFactory, "GetPrototype"):

    def _get_prototype(self, descriptor):
        return self.GetMessageClass(descriptor)

    _mf.MessageFactory.GetPrototype = _get_prototype

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import functools
from tqdm import tqdm




## === cell 1
"""Robust Bi-Tempered Logistic Loss Based on Bregman Divergences.

Source: https://bit.ly/3jSol8T
"""
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
            tf1.reduce_sum(tf1.exp(activations), -1, keepdims=True)
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
            tf1.reduce_sum(tf1.exp(internal_activations), -1, keepdims=True)
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
        lambda: tf1.log(tf1.reduce_sum(tf1.exp(activations), -1, keepdims=True)),
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
def acc_gambler(y_true, y_pred):
    y_temp = y_pred[:, 1:]
    count = tf.constant(0, dtype=tf.int32)
    for i in range(tf.shape(y_true)[0]):
        tf.autograph.experimental.set_loop_options(
            shape_invariants=[(count, tf.TensorShape([None]))]
        )
        if tf.math.argmax(y_temp[i]) == tf.math.argmax(y_true[i]):
            count = tf.math.add(count, 1)
    return tf.cast(count, tf.float32) / tf.cast(tf.shape(y_true)[0], tf.float32)


import tensorflow.keras.backend as K


def loss_gambler(label_smoothing=0.0):
    def loss_gamb(y_true, y_pred):
        y_true = tf.add(
            y_true,
            tf.add(
                tf.multiply(
                    label_smoothing / 2.0, tf.add(1.0, tf.multiply(-1.0, y_true))
                ),
                tf.multiply(-label_smoothing / 2.0, y_true),
            ),
        )
        y_temp = y_pred[:, 1:]
        f0 = y_pred[:, 0]
        lamb = tf.divide(
            tf.multiply(K.sum(y_temp), K.sum(y_temp)),
            K.sum(tf.multiply(y_temp, y_temp)),
        )
        loss = tf.constant(0.0)
        for i in range(tf.shape(y_true)[1]):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[(loss, tf.TensorShape([None]))]
            )
            temp = tf.constant(0.0)
            loss = tf.add(
                loss,
                tf.add(
                    temp,
                    (-1.0 / tf.cast(tf.shape(y_true)[0], tf.float32))
                    * tf.multiply(y_true[:, i], K.log(y_temp[:, i] + f0 / lamb)),
                ),
            )
        return tf.reduce_sum(loss)

    return loss_gamb




## === cell 3
def load_legacy_model(model_path, custom_objects=None):
    """
    Load a SavedModel directory using keras.models.load_model with compile=False.
    This avoids needing the original loss/metric definitions for inference.
    """
    if not os.path.isdir(model_path):
        return None
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        if custom_objects:
            for name, obj in custom_objects.items():
                setattr(model, name, obj)
        return model
    except Exception as e:
        print(f"Failed to load model from {model_path}: {e}")
        return None


base_input = "/kaggle/working"

model_v1_path = os.path.join(
    base_input, "only-xception-with-cropping", "saved-model-11-0.879"
)
model_v2_path = os.path.join(
    base_input, "gambler-s-loss-cassava", "saved-model-10-0.843"
)
model_v3_path = os.path.join(
    base_input, "bitempered-loss-only-xception-with-cropping", "saved-model-12-0.849"
)

model_v1 = load_legacy_model(model_v1_path)
model_v2 = load_legacy_model(
    model_v2_path,
    custom_objects={"acc_gambler": acc_gambler, "loss_gamb": loss_gambler(0.1)},
)
model_v3 = load_legacy_model(
    model_v3_path,
    custom_objects={"bi_tempered_logistic_loss": bi_tempered_logistic_loss},
)

fallback_model = None
fallback_img_size = None




## === cell 4
import random
from tensorflow.keras import layers, models, optimizers, callbacks, applications


def train_simple_model(num_samples=2000, img_size=(299, 299), epochs=10):
    """
    Trains a lightweight Xception‑based fallback model.
    Labels are converted to strings so that ImageDataGenerator can create categorical targets.
    """
    train_csv_path = os.path.join(
        base_input, "cassava-leaf-disease-classification", "train.csv"
    )
    train_df = pd.read_csv(train_csv_path)

    train_df["label"] = train_df["label"].astype(str)

    train_df = train_df.sample(
        min(num_samples, len(train_df)), random_state=42
    ).reset_index(drop=True)

    train_img_dir = os.path.join(
        base_input, "cassava-leaf-disease-classification", "train_images"
    )
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    generator = train_datagen.flow_from_dataframe(
        train_df,
        directory=train_img_dir,
        x_col="image_id",
        y_col="label",
        target_size=img_size,
        batch_size=32,
        class_mode="categorical",
        shuffle=True,
    )
    num_classes = train_df["label"].nunique()

    base_model = applications.Xception(
        weights="imagenet", include_top=False, input_shape=(*img_size, 3)
    )
    base_model.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    output = layers.Dense(num_classes, activation="softmax")(x)

    simple_model = models.Model(inputs=base_model.input, outputs=output)
    simple_model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    simple_model.fit(
        generator,
        epochs=epochs,
        verbose=0,
        callbacks=[callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
    )
    return simple_model, img_size


if all(m is None for m in [model_v1, model_v2, model_v3]):
    fallback_model, fallback_img_size = train_simple_model()
    print("Fallback model trained and will be used for predictions.")
else:
    print("Legacy models loaded; fallback model not required.")




## === cell 5
def random_crop(img, random_crop_size):
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y : (y + dy), x : (x + dx), :]


def crop_generator(batches, crop_length):
    """Yield random crops from batches produced by an ImageDataGenerator."""
    while True:
        batch_x = next(batches)
        batch_crops = np.zeros(
            (batch_x.shape[0], crop_length, crop_length, 3), dtype=batch_x.dtype
        )
        for i in range(batch_x.shape[0]):
            batch_crops[i] = random_crop(batch_x[i], (crop_length, crop_length))
        yield batch_crops




## === cell 6
def predict_with_model(model, generator, steps):
    """
    Wrapper that returns predictions.
    If model is None (failed to load), returns uniform random probabilities.
    """
    if model is None:
        total_samples = generator.n if hasattr(generator, "n") else steps
        return np.full((total_samples, 5), 0.2, dtype=np.float32)
    preds = model.predict(generator, steps=steps, verbose=0)
    if preds.ndim == 2 and preds.shape[1] == 5:
        return preds
    if preds.ndim == 2:
        exp_preds = np.exp(preds - np.max(preds, axis=1, keepdims=True))
        return exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
    raise ValueError("Unexpected prediction shape")


def ensemble_predict(model, generator, steps, repeats=10):
    """
    Perform Test‑Time Augmentation by averaging `repeats` predictions.
    The same generator instance is reused; after each pass we reset it
    to avoid re‑creating the Python object, saving I/O overhead.
    """
    if model is None:
        total_samples = generator.n if hasattr(generator, "n") else steps
        return np.full((total_samples, 5), 0.2, dtype=np.float32)

    agg = np.zeros((int(steps * generator.batch_size), 5), dtype=np.float32)

    for _ in range(repeats):
        preds = model.predict(generator, steps=steps, verbose=0)
        agg[: preds.shape[0]] += preds
        if hasattr(generator, "reset"):
            generator.reset()
        else:
            pass
    agg /= repeats
    return agg[
        : len(generator.filenames) if hasattr(generator, "filenames") else agg.shape[0]
    ]




## === cell 7
BATCH_SIZE = 32

test_dir = os.path.join(
    base_input, "cassava-leaf-disease-classification", "test_images"
)

test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
test_df = pd.DataFrame({"image_id": test_files})

datagen = ImageDataGenerator(zoom_range=0.4, horizontal_flip=True)


def make_generator(target_size=(448, 448)):
    return datagen.flow_from_dataframe(
        test_df,
        directory=test_dir,
        x_col="image_id",
        target_size=target_size,
        batch_size=BATCH_SIZE,
        class_mode=None,
        shuffle=False,
    )


steps = int(np.ceil(len(test_df) / BATCH_SIZE))

generator_v1 = make_generator()
pred_v1 = ensemble_predict(model_v1, generator_v1, steps)

generator_v2 = make_generator()
pred_v2_raw = predict_with_model(model_v2, generator_v2, steps)
if pred_v2_raw.shape[1] == 4:
    pred_v2 = np.concatenate([np.zeros((len(test_df), 1)), pred_v2_raw], axis=1)
else:
    pred_v2 = pred_v2_raw
agg_v2 = np.zeros_like(pred_v2)
for _ in range(9):
    if hasattr(generator_v2, "reset"):
        generator_v2.reset()
    temp_raw = predict_with_model(model_v2, generator_v2, steps)
    if temp_raw.shape[1] == 4:
        temp = np.concatenate([np.zeros((len(test_df), 1)), temp_raw], axis=1)
    else:
        temp = temp_raw
    agg_v2 += temp
pred_v2 = (pred_v2 + agg_v2) / 10.0

generator_v3 = make_generator()
pred_v3 = ensemble_predict(model_v3, generator_v3, steps)




## === cell 8
if fallback_model is not None:
    tg = datagen.flow_from_dataframe(
        test_df,
        directory=test_dir,
        x_col="image_id",
        target_size=fallback_img_size,
        batch_size=32,
        class_mode=None,
        shuffle=False,
    )
    pred_fallback = fallback_model.predict(tg, verbose=0)
    if pred_fallback.shape[1] != 5:
        if pred_fallback.shape[1] < 5:
            pad = np.zeros((pred_fallback.shape[0], 5 - pred_fallback.shape[1]))
            pred_fallback = np.concatenate([pred_fallback, pad], axis=1)
        else:
            pred_fallback = pred_fallback[:, :5]
else:
    pred_fallback = np.zeros((len(test_df), 5), dtype=np.float32)

pred_ensemble = pred_v1 + pred_v2 + pred_v3 + pred_fallback
predicted_class_indices = np.argmax(pred_ensemble, axis=1)

label_map = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4}
inv_label_map = {v: k for k, v in label_map.items()}
predicted_labels = [inv_label_map[idx] for idx in predicted_class_indices]

submission = pd.DataFrame({"image_id": test_df["image_id"], "label": predicted_labels})

submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
