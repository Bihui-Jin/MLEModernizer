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

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

tf.compat.v1.disable_eager_execution()
tf1 = tf.compat.v1

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

RANDOM_SEED = 1337
np.random.seed(RANDOM_SEED)



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
IMG_SIZE = 128  # keep small to ensure runtime within 600s in this environment
NUM_CLASSES = 5
BATCH_SIZE = 32
EPOCHS = 2  # keep as in provided code to stay within runtime bounds


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
        img = tf1.image.resize_images(
            img, [IMG_SIZE, IMG_SIZE], method=tf1.image.ResizeMethod.BILINEAR
        )
        return img

    if training:

        def _map_fn(path, label):
            img = _load_img(path)
            img = tf1.image.random_flip_left_right(img, seed=RANDOM_SEED)
            return img, label

        ds = ds.map(_map_fn, num_parallel_calls=4)
        ds = ds.batch(BATCH_SIZE).prefetch(1)
    else:

        def _map_fn_test(path):
            img = _load_img(path)
            return img

        ds = ds.map(_map_fn_test, num_parallel_calls=4)
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
    img = tf1.image.resize_images(
        img, [IMG_SIZE, IMG_SIZE], method=tf1.image.ResizeMethod.BILINEAR
    )
    return img, label


val_ds = val_ds.map(_val_map, num_parallel_calls=4).batch(BATCH_SIZE).prefetch(1)

test_ds = _build_dataset_from_df(test_df, TEST_IMG_DIR, training=False)

train_it = tf1.data.make_initializable_iterator(train_ds)
val_it = tf1.data.make_initializable_iterator(val_ds)
test_it = tf1.data.make_initializable_iterator(test_ds)

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
        dropout = tf1.keras.layers.Dropout(rate=0.3, name="dropout")
        dense = tf1.keras.layers.Dense(NUM_CLASSES, name="fc")

        x = conv1(x)
        x = pool1(x)
        x = conv2(x)
        x = pool2(x)
        x = conv3(x)
        x = pool3(x)
        x = flatten(x)
        x = dropout(x, training=training)
        logits = dense(x)
        return logits


logits_tr = _model(x_tr, is_training)
logits_va = _model(x_va, False)
logits_te = _model(x_te, False)

loss_tr = tf1.reduce_mean(
    tf1.nn.sparse_softmax_cross_entropy_with_logits(labels=y_tr, logits=logits_tr)
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
OperatorNotAllowedInGraphError            Traceback (most recent call last)
/tmp/ipykernel_11/776004517.py in <cell line: 0>()
     68 
     69 
---> 70 logits_tr = _model(x_tr, is_training)
     71 logits_va = _model(x_va, False)
     72 logits_te = _model(x_te, False)

/tmp/ipykernel_11/776004517.py in _model(x, training)
     63         x = pool3(x)
     64         x = flatten(x)
---> 65         x = dropout(x, training=training)
     66         logits = dense(x)
     67         return logits

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/tensor.py in _disallow(self, task)
    301 
    302   def _disallow(self, task):
--> 303     raise errors.OperatorNotAllowedInGraphError(
    304         f"{task} is not allowed."
    305         " You can attempt the following resolutions to the problem:"

OperatorNotAllowedInGraphError: Exception encountered when calling Dropout.call().

Using a symbolic `tf.Tensor` as a Python `bool` is not allowed. You can attempt the following resolutions to the problem: If you are running in Graph mode, use Eager execution mode or decorate this function with @tf.function. If you are using AutoGraph, you can try decorating this function with @tf.function. If that does not work, then you may be using an unsupported feature or your source code may not be visible to AutoGraph. See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/limitations.md#access-to-source-code for more information.

Arguments received by Dropout.call():
  • inputs=tf.Tensor(shape=(None, 32768), dtype=float32)
  • training=tf.Tensor(shape=(), dtype=bool)

## === cell 8
def _run_epoch(sess, init_op, train_mode):
    sess.run(init_op)
    losses = []
    accs = []
    while True:
        try:
            if train_mode:
                l, a, _ = sess.run(
                    [loss_tr, acc_tr, train_op], feed_dict={is_training: True}
                )
            else:
                l, a = sess.run([loss_tr, acc_tr], feed_dict={is_training: False})
            losses.append(l)
            accs.append(a)
        except tf1.errors.OutOfRangeError:
            break
    return float(np.mean(losses)) if losses else np.nan, (
        float(np.mean(accs)) if accs else np.nan
    )


def _eval_acc(sess):
    sess.run(val_it.initializer)
    accs = []
    while True:
        try:
            a = sess.run(acc_va)
            accs.append(a)
        except tf1.errors.OutOfRangeError:
            break
    return float(np.mean(accs)) if accs else np.nan


def _predict_test(sess):
    sess.run(test_it.initializer)
    all_probs = []
    while True:
        try:
            p = sess.run(probs_te)
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
        tr_loss, tr_acc = _run_epoch(sess, train_it.initializer, train_mode=True)
        va_acc = _eval_acc(sess)
        print(
            "Epoch %d/%d - train_loss=%.4f train_acc=%.4f val_acc=%.4f"
            % (ep + 1, EPOCHS, tr_loss, tr_acc, va_acc)
        )

    test_probs = _predict_test(sess)

test_pred = np.argmax(test_probs, axis=1).astype(int)

submission = pd.DataFrame({"image_id": test_df["image_id"].values, "label": test_pred})
submission = submission[["image_id", "label"]]
assert submission.shape[0] == test_df.shape[0]

out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)
print(submission.head())
print("Wrote %s with shape: %s" % (out_path, submission.shape))

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4294777337.py in <cell line: 0>()
     54 
     55     for ep in range(EPOCHS):
---> 56         tr_loss, tr_acc = _run_epoch(sess, train_it.initializer, train_mode=True)
     57         va_acc = _eval_acc(sess)
     58         print(

/tmp/ipykernel_11/4294777337.py in _run_epoch(sess, init_op, train_mode)
      7             if train_mode:
      8                 l, a, _ = sess.run(
----> 9                     [loss_tr, acc_tr, train_op], feed_dict={is_training: True}
     10                 )
     11             else:

NameError: name 'loss_tr' is not defined
