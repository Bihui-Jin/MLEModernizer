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

0.8838017527954065

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'I fix the TensorFlow import crash by removing the protobuf “python” fallback (it triggers the `MessageFactory.GetPrototype` error with current protobuf/TensorFlow builds) and keep the rest of the environment setup intact. Then I fix the graph-construction error by ensuring `tf.cond` never receives a Python `bool` (use the `is_training` placeholder for all model calls), which also prevents downstream `NameError` from the graph not being built. Finally, I make validation/test iterators one-shot (no reinitialization needed) and ensure the script always writes `/kaggle/working/submission.csv` with the correct columns and row alignment.'
- What this solution (achieved 0.44021) has done: 'I fix the TensorFlow import crash by removing the forced protobuf CPP implementation (it breaks with the installed protobuf build) and letting TensorFlow/protobuf pick a working runtime. Then I make the code consistent with the actual Kaggle runtime (Python 3.x here, despite the “2.7” note) while keeping the same TF1-graph training logic, model, loss, and data pipeline intact. Finally, I ensure the graph builds successfully end-to-end and always writes `/kaggle/working/submission.csv` with the required `image_id,label` columns aligned to `sample_submission.csv`.'
- What this solution (achieved 0.11771) has done: 'I fix the TensorFlow import crash caused by an incompatible protobuf runtime by forcing the pure-Python protobuf implementation *before* importing TensorFlow (this is the specific combo that avoids the `MessageFactory.GetPrototype` failure in many Kaggle TF1-compat setups). I also add a safe fallback for the image resize API (`tf.image.resize` vs `tf.image.resize_images`) to keep the same preprocessing while preventing attribute errors across TF versions. Finally, I make the dataset `.map()` parallelism use `AUTOTUNE` when available (or a small constant otherwise) to stabilize performance without changing the model/training logic, and keep the exact required `/kaggle/working/submission.csv` output format.'
- What this solution (achieved 0.11809) has done: 'I fix the TensorFlow/protobuf import crash by avoiding the forced pure-Python protobuf runtime (it is what triggers the `MessageFactory.GetPrototype` AttributeError with the installed protobuf) and letting TF/protobuf use the default implementation. Then I fix the very low accuracy by correcting the dropout graph: the current `tf.cond` branch captures the wrong tensor (`x` is overwritten), causing dropout to be effectively skipped; this is a genuine logic bug that hurts training without changing the intended architecture. Finally, I keep the same data paths and training loop, and ensure the script still writes `/kaggle/working/submission.csv` with `image_id,label` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.60837) has done: 'I fix the TensorFlow import crash by forcing the pure-Python protobuf implementation *before* importing TensorFlow, which is the most common workaround for the `MessageFactory.GetPrototype` error in Kaggle TF1-compat environments. Then I keep the model/training logic the same, but ensure the graph builds reliably across TF versions by using a safe resize wrapper and making sure training/validation/test iterators are constructed consistently. Finally, I keep the existing submission creation logic but add a strict alignment to `sample_submission.csv` ordering to avoid accidental row-order mismatches (which can catastrophically drop accuracy).'

# 9. Code solution

## === cell 0
from __future__ import print_function
import os
import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

RANDOM_SEED = 1337
np.random.seed(RANDOM_SEED)

_tf_import_error = None
tf = None

try:
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"
    import tensorflow as tf  # noqa: E402
except Exception as e:
    _tf_import_error = e
    tf = None

if tf is None:
    try:
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        import importlib  # noqa: E402

        tf = importlib.import_module("tensorflow")
        _tf_import_error = None
    except Exception as e2:
        raise RuntimeError(
            "TensorFlow import failed with both protobuf runtimes. "
            "cpp_error=%r python_error=%r" % (_tf_import_error, e2)
        )

tf1 = tf.compat.v1
tf1.disable_eager_execution()

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

print("TF version:", getattr(tf, "__version__", "unknown"))


def _resize_bilinear(img, size_hw):
    if hasattr(tf1.image, "resize_images"):
        return tf1.image.resize_images(
            img, size_hw, method=tf1.image.ResizeMethod.BILINEAR
        )
    return tf.image.resize(img, size_hw, method="bilinear")


try:
    _AUTOTUNE = tf.data.experimental.AUTOTUNE
except Exception:
    _AUTOTUNE = 4



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/3626276349.py in <cell line: 0>()
     31 
---> 32         tf = importlib.import_module("tensorflow")
     33         _tf_import_error = None

/usr/lib/python3.11/importlib/__init__.py in import_module(name, package)
    125             level += 1
--> 126     return _bootstrap._gcd_import(name[level:], package, level)
    127 

/usr/lib/python3.11/importlib/_bootstrap.py in _gcd_import(name, package, level)

/usr/lib/python3.11/importlib/_bootstrap.py in _find_and_load(name, import_)

/usr/lib/python3.11/importlib/_bootstrap.py in _find_and_load_unlocked(name, import_)

/usr/lib/python3.11/importlib/_bootstrap.py in _load_unlocked(spec)

/usr/lib/python3.11/importlib/_bootstrap_external.py in exec_module(self, module)

/usr/lib/python3.11/importlib/_bootstrap.py in _call_with_frames_removed(f, *args, **kwds)

/usr/local/lib/python3.11/dist-packages/tensorflow/__init__.py in <module>
     48 
---> 49 from tensorflow._api.v2 import __internal__
     50 from tensorflow._api.v2 import __operators__

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/__init__.py in <module>
      7 
----> 8 from tensorflow._api.v2.__internal__ import autograph
      9 from tensorflow._api.v2.__internal__ import decorator

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/autograph/__init__.py in <module>
      7 
----> 8 from tensorflow.python.autograph.core.ag_ctx import control_status_ctx # line: 34
      9 from tensorflow.python.autograph.impl.api import tf_convert # line: 493

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/ag_ctx.py in <module>
     20 
---> 21 from tensorflow.python.autograph.utils import ag_logging
     22 from tensorflow.python.util.tf_export import tf_export

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/__init__.py in <module>
     16 
---> 17 from tensorflow.python.autograph.utils.context_managers import control_dependency_on_returns
     18 from tensorflow.python.autograph.utils.misc import alias_tensors

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/context_managers.py in <module>
     18 
---> 19 from tensorflow.python.framework import ops
     20 from tensorflow.python.ops import tensor_array_ops

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in <module>
     32 from google.protobuf import message
---> 33 from tensorflow.core.framework import attr_value_pb2
     34 from tensorflow.core.framework import full_type_pb2

/usr/local/lib/python3.11/dist-packages/tensorflow/core/framework/attr_value_pb2.py in <module>
      4 """Generated protocol buffer code."""
----> 5 from google.protobuf.internal import builder as _builder
      6 from google.protobuf import descriptor as _descriptor

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/builder.py in <module>
     17 from google.protobuf.internal import enum_type_wrapper
---> 18 from google.protobuf.internal import python_message
     19 from google.protobuf import message as _message

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/python_message.py in <module>
     37 
---> 38 from google.protobuf import descriptor as descriptor_mod
     39 from google.protobuf import message as message_mod

/usr/local/lib/python3.11/dist-packages/google/protobuf/descriptor.py in <module>
     28   if _message is None:
---> 29     from google.protobuf.pyext import _message
     30   _USE_C_DESCRIPTORS = True

ImportError: cannot import name '_message' from 'google.protobuf.pyext' (/usr/local/lib/python3.11/dist-packages/google/protobuf/pyext/__init__.py)

During handling of the above exception, another exception occurred:

RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/3626276349.py in <cell line: 0>()
     33         _tf_import_error = None
     34     except Exception as e2:
---> 35         raise RuntimeError(
     36             "TensorFlow import failed with both protobuf runtimes. "
     37             "cpp_error=%r python_error=%r" % (_tf_import_error, e2)

RuntimeError: TensorFlow import failed with both protobuf runtimes. cpp_error=ImportError("cannot import name '_message' from 'google.protobuf.pyext' (/usr/local/lib/python3.11/dist-packages/google/protobuf/pyext/__init__.py)") python_error=ImportError("cannot import name '_message' from 'google.protobuf.pyext' (/usr/local/lib/python3.11/dist-packages/google/protobuf/pyext/__init__.py)")

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

        loss_values = _custom_gradient_bi_tempered_logistic_loss(activations)
        loss_values = tf1.reduce_sum(loss_values, -1)
        return loss_values




## === cell 2
def acc_gambler(y_true, y_pred):
    y_true = tf1.convert_to_tensor(y_true)
    y_pred = tf1.convert_to_tensor(y_pred)
    return tf1.reduce_mean(
        tf1.cast(
            tf1.equal(tf1.argmax(y_true, axis=-1), tf1.argmax(y_pred[:, 1:], axis=-1)),
            tf1.float32,
        )
    )


def loss_gambler(label_smoothing=0.0):
    def loss_gamb(y_true, y_pred):
        y_temp = y_pred[:, 1:]
        y_true_sm = (1.0 - label_smoothing) * y_true + label_smoothing / tf1.cast(
            tf1.shape(y_true)[-1], tf1.float32
        )
        y_temp = tf1.clip_by_value(y_temp, 1e-7, 1.0)
        return -tf1.reduce_mean(tf1.reduce_sum(y_true_sm * tf1.log(y_temp), axis=-1))

    return loss_gamb




## === cell 3
def _infer_endpoint(saved_model_dir):
    return "serving_default"


def load_savedmodel_as_keras_model(saved_model_dir):
    raise OSError(
        "External SavedModel not available in this environment: %s" % saved_model_dir
    )




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
DATA_DIR = "../input/cassava-leaf-disease-classification/"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

train_df = pd.read_csv(TRAIN_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)
test_df = sample_sub[["image_id"]].copy()

print("train_df:", train_df.shape, "test_df:", test_df.shape)
print("Train image dir exists:", os.path.isdir(TRAIN_IMG_DIR))
print("Test image dir exists:", os.path.isdir(TEST_IMG_DIR))



## === cell 6
IMG_SIZE = 128  # keep small to ensure runtime within 600s
NUM_CLASSES = 5
BATCH_SIZE = 32
EPOCHS = 2  # keep as provided


def _build_dataset_from_df(df, img_dir, training):
    image_ids = df["image_id"].astype(str).values
    paths = [os.path.join(img_dir, x) for x in image_ids]

    if training:
        labels = df["label"].astype(np.int32).values
        ds = tf1.data.Dataset.from_tensor_slices((paths, labels))
    else:
        ds = tf1.data.Dataset.from_tensor_slices(paths)

    if training:
        ds = ds.shuffle(
            buffer_size=min(len(paths), 4096),
            seed=RANDOM_SEED,
            reshuffle_each_iteration=True,
        )

    def _load_img(path):
        img_bytes = tf1.read_file(path)
        img = tf1.image.decode_jpeg(img_bytes, channels=3)
        img = tf1.image.convert_image_dtype(img, tf1.float32)  # [0,1]
        img = _resize_bilinear(img, [IMG_SIZE, IMG_SIZE])
        return img

    if training:

        def _map_fn(path, label):
            img = _load_img(path)
            img = tf1.image.random_flip_left_right(img, seed=RANDOM_SEED)
            return img, label

        ds = ds.map(_map_fn, num_parallel_calls=_AUTOTUNE)
        ds = ds.batch(BATCH_SIZE).prefetch(1)
    else:

        def _map_fn_test(path):
            img = _load_img(path)
            return img

        ds = ds.map(_map_fn_test, num_parallel_calls=_AUTOTUNE)
        ds = ds.batch(BATCH_SIZE).prefetch(1)

    return ds


perm = np.random.RandomState(RANDOM_SEED).permutation(len(train_df))
val_size = int(0.1 * len(train_df))
val_idx = perm[:val_size]
trn_idx = perm[val_size:]

trn_df = train_df.iloc[trn_idx].reset_index(drop=True)
val_df = train_df.iloc[val_idx].reset_index(drop=True)



## === cell 7
tf1.reset_default_graph()
tf1.set_random_seed(RANDOM_SEED)

train_ds = _build_dataset_from_df(trn_df, TRAIN_IMG_DIR, training=True)

val_paths = [
    os.path.join(TRAIN_IMG_DIR, x) for x in val_df["image_id"].astype(str).values
]
val_labels = val_df["label"].astype(np.int32).values
val_ds = tf1.data.Dataset.from_tensor_slices((val_paths, val_labels))


def _val_map(path, label):
    img_bytes = tf1.read_file(path)
    img = tf1.image.decode_jpeg(img_bytes, channels=3)
    img = tf1.image.convert_image_dtype(img, tf1.float32)
    img = _resize_bilinear(img, [IMG_SIZE, IMG_SIZE])
    return img, label


val_ds = (
    val_ds.map(_val_map, num_parallel_calls=_AUTOTUNE).batch(BATCH_SIZE).prefetch(1)
)

test_ds = _build_dataset_from_df(test_df, TEST_IMG_DIR, training=False)

train_it = tf1.data.make_initializable_iterator(train_ds)
val_it = tf1.data.make_one_shot_iterator(val_ds)
test_it = tf1.data.make_one_shot_iterator(test_ds)

x_tr, y_tr = train_it.get_next()
x_va, y_va = val_it.get_next()
x_te = test_it.get_next()

is_training = tf1.placeholder_with_default(False, shape=())


def _model(x, training):
    with tf1.variable_scope("cnn", reuse=tf1.AUTO_REUSE):
        conv1 = tf1.keras.layers.Conv2D(
            32, 3, padding="same", activation=tf.nn.relu, name="conv1"
        )
        pool1 = tf1.keras.layers.MaxPool2D(pool_size=2, strides=2, name="pool1")
        conv2 = tf1.keras.layers.Conv2D(
            64, 3, padding="same", activation=tf.nn.relu, name="conv2"
        )
        pool2 = tf1.keras.layers.MaxPool2D(pool_size=2, strides=2, name="pool2")
        conv3 = tf1.keras.layers.Conv2D(
            128, 3, padding="same", activation=tf.nn.relu, name="conv3"
        )
        pool3 = tf1.keras.layers.MaxPool2D(pool_size=2, strides=2, name="pool3")
        flatten = tf1.keras.layers.Flatten(name="flatten")
        dense = tf1.keras.layers.Dense(NUM_CLASSES, name="fc")

        x = conv1(x)
        x = pool1(x)
        x = conv2(x)
        x = pool2(x)
        x = conv3(x)
        x = pool3(x)
        x = flatten(x)

        def _do_dropout(t):
            return tf1.nn.dropout(t, keep_prob=0.7)

        x = tf1.cond(
            tf1.cast(training, tf.bool),
            lambda: _do_dropout(x),
            lambda: tf1.identity(x),
        )

        logits = dense(x)
        return logits


logits_tr = _model(x_tr, is_training)
logits_va = _model(x_va, is_training)
logits_te = _model(x_te, is_training)

y_tr_oh = tf1.one_hot(y_tr, depth=NUM_CLASSES, dtype=tf1.float32)
loss_tr = tf1.reduce_mean(
    bi_tempered_logistic_loss(
        labels=y_tr_oh,
        activations=logits_tr,
        t1=0.2,
        t2=1.0,
        label_smoothing=0.1,
        num_iters=10,
    )
)

pred_tr = tf1.argmax(logits_tr, axis=1, output_type=tf1.int32)
acc_tr = tf1.reduce_mean(tf1.cast(tf1.equal(pred_tr, y_tr), tf1.float32))

pred_va = tf1.argmax(logits_va, axis=1, output_type=tf1.int32)
acc_va = tf1.reduce_mean(tf1.cast(tf1.equal(pred_va, y_va), tf1.float32))

global_step = tf1.train.get_or_create_global_step()
opt = tf1.train.AdamOptimizer(learning_rate=1e-3)
train_op = opt.minimize(loss_tr, global_step=global_step)

probs_te = tf1.nn.softmax(logits_te)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/764619776.py in <cell line: 0>()
----> 1 tf1.reset_default_graph()
      2 tf1.set_random_seed(RANDOM_SEED)
      3 
      4 train_ds = _build_dataset_from_df(trn_df, TRAIN_IMG_DIR, training=True)
      5 

NameError: name 'tf1' is not defined

## === cell 8
def _run_train_epoch(sess, init_op):
    sess.run(init_op)
    losses = []
    accs = []
    while True:
        try:
            l, a, _ = sess.run(
                [loss_tr, acc_tr, train_op], feed_dict={is_training: True}
            )
            losses.append(l)
            accs.append(a)
        except tf1.errors.OutOfRangeError:
            break
    return float(np.mean(losses)) if losses else np.nan, (
        float(np.mean(accs)) if accs else np.nan
    )


def _eval_acc(sess):
    accs = []
    while True:
        try:
            a = sess.run(acc_va, feed_dict={is_training: False})
            accs.append(a)
        except tf1.errors.OutOfRangeError:
            break
    return float(np.mean(accs)) if accs else np.nan


def _predict_test(sess):
    all_probs = []
    while True:
        try:
            p = sess.run(probs_te, feed_dict={is_training: False})
            all_probs.append(p)
        except tf1.errors.OutOfRangeError:
            break
    probs = (
        np.concatenate(all_probs, axis=0)
        if all_probs
        else np.zeros((0, NUM_CLASSES), dtype=np.float32)
    )
    return probs


with tf1.Session() as sess:
    sess.run(tf1.global_variables_initializer())
    sess.run(tf1.local_variables_initializer())

    for ep in range(EPOCHS):
        tr_loss, tr_acc = _run_train_epoch(sess, train_it.initializer)
        va_acc = _eval_acc(sess)
        print(
            "Epoch %d/%d - train_loss=%.4f train_acc=%.4f val_acc=%.4f"
            % (ep + 1, EPOCHS, tr_loss, tr_acc, va_acc)
        )

    test_probs = _predict_test(sess)

test_pred = np.argmax(test_probs, axis=1).astype(int)

submission = sample_sub[["image_id"]].copy()
submission["label"] = test_pred
submission = submission[["image_id", "label"]]
assert submission.shape[0] == sample_sub.shape[0]

out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)
print(submission.head())
print("Wrote %s with shape: %s" % (out_path, submission.shape))

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1454447634.py in <cell line: 0>()
     44 
     45 
---> 46 with tf1.Session() as sess:
     47     sess.run(tf1.global_variables_initializer())
     48     sess.run(tf1.local_variables_initializer())

NameError: name 'tf1' is not defined
