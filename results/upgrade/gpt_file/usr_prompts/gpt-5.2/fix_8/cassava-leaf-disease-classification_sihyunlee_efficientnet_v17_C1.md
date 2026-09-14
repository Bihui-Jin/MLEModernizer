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

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import math
import json
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
)  # kept for compatibility with original imports

tf.random.set_seed(42)
np.random.seed(42)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.threading.set_inter_op_parallelism_threads(0)
    tf.config.threading.set_intra_op_parallelism_threads(0)
except Exception:
    pass

print("TF version:", tf.__version__)




## === cell 1
def draw_grpah(history):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, "bo", label="Training loss", markersize=1)
    plt.plot(epochs, val_loss, "b", label="Validation loss", markersize=1)
    plt.legend()
    plt.show()




## === cell 2
def stratifying_data(data):
    data.head()
    orderby_label = []
    sample_rate = [2.0, 1.5, 1.5, 0.3, 1.5]
    for i in range(5):
        orderby_label.append(
            data[data["label"] == i].sample(
                frac=sample_rate[i], replace=True, random_state=42
            )
        )
    stratified_data = pd.concat(orderby_label)
    return stratified_data




## === cell 3
def get_count_by_class(data):
    Count = []
    for i in range(5):
        Count.append((i, data[data["label"] == i].shape))
    return Count




## === cell 4
def stratified_split(df, stratify_col, test_size=0.05, random_state=42):
    rng = np.random.RandomState(random_state)
    val_parts = []
    train_parts = []
    for _, grp in df.groupby(stratify_col):
        n = len(grp)
        n_val = max(1, int(round(n * test_size)))
        idx = grp.index.to_numpy()
        rng.shuffle(idx)
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        val_parts.append(df.loc[val_idx])
        train_parts.append(df.loc[train_idx])
    train_df = (
        pd.concat(train_parts)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )
    val_df = (
        pd.concat(val_parts)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )
    return train_df, val_df




## === cell 5
PATH = "../input/cassava-leaf-disease-classification/"

data = pd.read_csv(PATH + "train.csv")
with open(PATH + "label_num_to_disease_map.json", "r") as f:
    real_labels = json.load(f)
real_labels = {int(k): v for k, v in real_labels.items()}

data["class_name"] = data.label.map(real_labels)
stratified_data = stratifying_data(data)
print(get_count_by_class(stratified_data))

train, val = stratified_split(
    stratified_data,
    stratify_col="class_name",
    test_size=0.05,
    random_state=42,
)

IMG_SIZE = 512

MODEL_PATH_EFF = "../input/efficientnet-day7/EfficientNet_day7.h5"
WILL_TRAIN = not os.path.isfile(MODEL_PATH_EFF)
BATCH_SIZE_TRAIN = (
    12 if WILL_TRAIN else 6
)  # training only; preserves exact data/epochs/steps semantics via recomputed steps

class_names = [real_labels[i] for i in range(5)]
class_to_index = {name: i for i, name in enumerate(class_names)}

train_paths = (PATH + "train_images/" + train["image_id"]).values.astype(str)
val_paths = (PATH + "train_images/" + val["image_id"]).values.astype(str)

train_labels_int = train["class_name"].map(class_to_index).values.astype(np.int32)
val_labels_int = val["class_name"].map(class_to_index).values.astype(np.int32)

AUTOTUNE = tf.data.AUTOTUNE


def _decode_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="nearest")
    img = tf.cast(img, tf.float32) / 255.0
    return img


BASE_SEED = tf.constant([42, 123], dtype=tf.int32)


def _augment(img, seed):
    img = tf.image.stateless_random_flip_left_right(img, seed=seed)
    img = tf.image.stateless_random_flip_up_down(
        img, seed=seed + tf.constant([1, 1], tf.int32)
    )

    angle = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([2, 2], tf.int32), minval=-45.0, maxval=45.0
    ) * (math.pi / 180.0)
    cos_a = tf.math.cos(angle)
    sin_a = tf.math.sin(angle)
    cx = (IMG_SIZE - 1) / 2.0
    cy = (IMG_SIZE - 1) / 2.0
    a0 = cos_a
    a1 = -sin_a
    a2 = cx - cos_a * cx + sin_a * cy
    b0 = sin_a
    b1 = cos_a
    b2 = cy - sin_a * cx - cos_a * cy
    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])[tf.newaxis, :]
    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[tf.newaxis, ...],
        transforms=transform,
        output_shape=[IMG_SIZE, IMG_SIZE],
        interpolation="NEAREST",
        fill_mode="NEAREST",
        fill_value=0.0,
    )[0]

    dx = (
        tf.random.stateless_uniform(
            [], seed=seed + tf.constant([3, 3], tf.int32), minval=-0.2, maxval=0.2
        )
        * IMG_SIZE
    )
    dy = (
        tf.random.stateless_uniform(
            [], seed=seed + tf.constant([4, 4], tf.int32), minval=-0.2, maxval=0.2
        )
        * IMG_SIZE
    )
    transform = tf.stack([1.0, 0.0, -dx, 0.0, 1.0, -dy, 0.0, 0.0])[tf.newaxis, :]
    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[tf.newaxis, ...],
        transforms=transform,
        output_shape=[IMG_SIZE, IMG_SIZE],
        interpolation="NEAREST",
        fill_mode="NEAREST",
        fill_value=0.0,
    )[0]

    z = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([5, 5], tf.int32), minval=0.8, maxval=1.2
    )
    transform = tf.stack([z, 0.0, cx - z * cx, 0.0, z, cy - z * cy, 0.0, 0.0])[
        tf.newaxis, :
    ]
    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[tf.newaxis, ...],
        transforms=transform,
        output_shape=[IMG_SIZE, IMG_SIZE],
        interpolation="NEAREST",
        fill_mode="NEAREST",
        fill_value=0.0,
    )[0]

    shear = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([6, 6], tf.int32), minval=-0.2, maxval=0.2
    )
    sh = tf.math.tan(shear)
    transform = tf.stack([1.0, -sh, sh * cy, 0.0, 1.0, 0.0, 0.0, 0.0])[tf.newaxis, :]
    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[tf.newaxis, ...],
        transforms=transform,
        output_shape=[IMG_SIZE, IMG_SIZE],
        interpolation="NEAREST",
        fill_mode="NEAREST",
        fill_value=0.0,
    )[0]

    return img


@tf.function(jit_compile=False)
def _aug_map_fn(i, img, lab):
    seed = BASE_SEED + tf.cast(tf.stack([i, i]), tf.int32)
    img = _augment(img, seed)
    y = tf.one_hot(lab, depth=5, dtype=tf.float32)
    return img, y


@tf.function(jit_compile=False)
def _val_map_fn(img, lab):
    y = tf.one_hot(lab, depth=5, dtype=tf.float32)
    return img, y


def _make_ds(paths_np, labels_np, training):
    options = tf.data.Options()
    options.deterministic = True  # preserve deterministic behavior

    try:
        options.experimental_optimization.apply_default_optimizations = True
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.parallel_batch = True
    except Exception:
        pass

    base = tf.data.Dataset.from_tensor_slices((paths_np, labels_np)).with_options(
        options
    )

    def _decode_map(path, lab):
        img = _decode_resize(path)
        return img, lab

    base = base.map(_decode_map, num_parallel_calls=AUTOTUNE, deterministic=True)
    base = base.apply(tf.data.experimental.ignore_errors())

    if training:
        cache_path = os.path.join(
            "/kaggle/working", f"cache_train_decoded_{IMG_SIZE}.tfdata"
        )
        base = base.cache(cache_path)
        base = base.shuffle(
            buffer_size=len(paths_np), seed=42, reshuffle_each_iteration=True
        )
        base = base.enumerate()
        ds = base.map(
            lambda i, img_lab: _aug_map_fn(i, img_lab[0], img_lab[1]),
            num_parallel_calls=AUTOTUNE,
            deterministic=True,
        )
    else:
        ds = base.map(
            lambda img, lab: _val_map_fn(img, lab),
            num_parallel_calls=AUTOTUNE,
            deterministic=True,
        )

    ds = ds.batch(BATCH_SIZE_TRAIN, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_set = _make_ds(train_paths, train_labels_int, training=True)
val_set = _make_ds(val_paths, val_labels_int, training=False)

train_steps = int(math.ceil(len(train_paths) / BATCH_SIZE_TRAIN))
val_steps = int(math.ceil(len(val_paths) / BATCH_SIZE_TRAIN))
print(train_steps)
print(val_steps)




## === cell 6
class SEUnit(keras.layers.Layer):
    def __init__(self, feature_map_len, se_ratio, **kwargs):
        super().__init__(**kwargs)
        self.feature_map_len = feature_map_len
        self.se_ratio = se_ratio
        self.global_avg_pool = keras.layers.GlobalAvgPool2D()
        self.reshape = keras.layers.Reshape((1, 1, feature_map_len))
        self.squeeze = keras.layers.Conv2D(
            max(1, int(feature_map_len * se_ratio)), kernel_size=1, activation="relu"
        )
        self.excitation = keras.layers.Conv2D(
            feature_map_len, kernel_size=1, activation="sigmoid"
        )

    def call(self, inputs):
        Z = inputs
        Z = self.global_avg_pool(Z)
        Z = self.reshape(Z)
        Z = self.squeeze(Z)
        excitation_vector = self.excitation(Z)
        excitation_vector = tf.reshape(
            excitation_vector, [-1, 1, 1, self.feature_map_len]
        )
        return inputs * excitation_vector

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "feature_map_len": self.feature_map_len,
            "se_ratio": self.se_ratio,
        }




## === cell 7
def SiLU(x):
    return x * tf.keras.backend.sigmoid(x)




## === cell 8
class MBConv_v2(keras.layers.Layer):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        multiplier,
        strides,
        name,
        dropout,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.multiplier = multiplier
        self.strides = strides
        self.dropout = dropout

        self.Name = name

        self.expansion_layers = []
        bn_axis = 3
        if multiplier != 1:
            self.expansion_layers = [
                keras.layers.Conv2D(
                    filters=in_channel * multiplier,
                    kernel_size=1,
                    padding="same",
                    use_bias=False,
                ),
                keras.layers.BatchNormalization(axis=bn_axis),
                keras.layers.Activation(SiLU),
            ]

        self.depthwise_layers = [
            keras.layers.DepthwiseConv2D(
                kernel_size=kernel_size, strides=strides, padding="same", use_bias=False
            ),
            keras.layers.BatchNormalization(axis=bn_axis),
            keras.layers.Activation(SiLU),
        ]
        se_ratio = 0.25 / multiplier
        self.se_unit = SEUnit(in_channel * multiplier, se_ratio)

        self.reduction_layers = [
            keras.layers.Conv2D(
                filters=out_channel, kernel_size=1, padding="same", use_bias=False
            ),
            keras.layers.BatchNormalization(axis=bn_axis),
        ]
        if dropout > 0:
            self.reduction_layers.append(
                keras.layers.Dropout(dropout, noise_shape=(None, 1, 1, 1))
            )

    def call(self, inputs):
        Z = inputs
        for layer in self.expansion_layers:
            Z = layer(Z)
        for layer in self.depthwise_layers:
            Z = layer(Z)
        Z = self.se_unit(Z)
        for layer in self.reduction_layers:
            Z = layer(Z)

        if self.strides == 1 and self.in_channel == self.out_channel:
            Z = Z + inputs
        return Z

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "in_channel": self.in_channel,
            "out_channel": self.out_channel,
            "kernel_size": self.kernel_size,
            "multiplier": self.multiplier,
            "strides": self.strides,
            "dropout": self.dropout,
        }




## === cell 9
default_efficient = [
    {
        "in_channel": 32,
        "out_channel": 16,
        "kernel_size": 3,
        "multiplier": 1,
        "strides": 1,
        "n_repeat": 1,
    },
    {
        "in_channel": 16,
        "out_channel": 24,
        "kernel_size": 3,
        "multiplier": 6,
        "strides": 2,
        "n_repeat": 2,
    },
    {
        "in_channel": 24,
        "out_channel": 40,
        "kernel_size": 5,
        "multiplier": 6,
        "strides": 2,
        "n_repeat": 2,
    },
    {
        "in_channel": 40,
        "out_channel": 80,
        "kernel_size": 3,
        "multiplier": 6,
        "strides": 2,
        "n_repeat": 3,
    },
    {
        "in_channel": 80,
        "out_channel": 112,
        "kernel_size": 5,
        "multiplier": 6,
        "strides": 1,
        "n_repeat": 3,
    },
    {
        "in_channel": 112,
        "out_channel": 192,
        "kernel_size": 5,
        "multiplier": 6,
        "strides": 2,
        "n_repeat": 4,
    },
    {
        "in_channel": 192,
        "out_channel": 320,
        "kernel_size": 3,
        "multiplier": 6,
        "strides": 1,
        "n_repeat": 1,
    },
]




## === cell 10
def rf(filters, width_coef):
    """Round number of filters based on width multiplier."""
    depth_divisor = 8
    filters *= width_coef
    new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)


def rd(repeat, depth_coef):
    return int(math.ceil(repeat * depth_coef))




## === cell 11
def condition_tensor(y, target_class):
    return tf.argmax(y, axis=-1) == target_class




## === cell 12
class condition_penalty_loss(tf.keras.losses.Loss):
    def __init__(self, condition_penalty, **kwargs):
        self.condition_penalty = condition_penalty
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

        class4 = condition_tensor(y_true, 4)
        predict0 = condition_tensor(y_pred, 0)
        predict4_0 = tf.cast(class4 & predict0, dtype=tf.float32)
        loss4_0 = loss * predict4_0 * 5

        class0123 = class4 == False
        predict4 = condition_tensor(y_pred, 4)
        predict0123_4 = tf.cast(class0123 & predict4, dtype=tf.float32)
        loss0123_4 = loss * predict0123_4 * 2

        class0 = condition_tensor(y_true, 0)
        predict0_4 = tf.cast(class0 & predict4, dtype=tf.float32)
        loss0_4 = loss * predict0_4 * 10

        return loss + loss0123_4 + loss4_0 + loss0_4

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "condition_penalty": self.condition_penalty}




## === cell 13
class EfficientNet(keras.models.Model):
    def __init__(
        self,
        default_efficient,
        width_coef,
        depth_coef,
        resolution,
        dropout,
        dropout_connect=0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        default_efficient = deepcopy(default_efficient)
        self.default_efficient = default_efficient
        self.width_coef = width_coef
        self.depth_coef = depth_coef
        self.resolution = resolution
        self.dropout = dropout
        self.dropout_connect = dropout_connect
        bn_axis = 3

        self.first_conv = [
            keras.layers.Conv2D(
                rf(32, width_coef),
                kernel_size=3,
                strides=2,
                padding="same",
                use_bias=False,
            ),
            keras.layers.BatchNormalization(axis=bn_axis),
            keras.layers.Activation(SiLU),
        ]

        self.MB_layers = []
        total_n_repeat = sum([block["n_repeat"] for block in default_efficient])
        block_num = 0
        name = "a"
        for block in default_efficient:
            block["in_channel"] = rf(block["in_channel"], width_coef)
            block["out_channel"] = rf(block["out_channel"], width_coef)
            block["n_repeat"] = rd(block["n_repeat"], depth_coef)
            block_args = dict(block)
            block_args.pop("n_repeat")

            drop_rate = dropout_connect * float(block_num) / total_n_repeat
            block_num += 1

            self.MB_layers.append(MBConv_v2(**block_args, dropout=drop_rate, name=name))
            name = chr(ord(name) + 1)
            if block["n_repeat"] > 1:
                block_args["in_channel"] = block_args["out_channel"]
                block_args["strides"] = 1
                for _ in range(block["n_repeat"] - 1):
                    drop_rate = dropout_connect * float(block_num) / total_n_repeat
                    self.MB_layers.append(
                        MBConv_v2(**block_args, dropout=drop_rate, name=name)
                    )
                    block_num += 1
                    name = chr(ord(name) + 1)

        self.last_conv = [
            keras.layers.Conv2D(
                rf(1280, width_coef),
                kernel_size=1,
                strides=1,
                padding="same",
                use_bias=False,
            ),
            keras.layers.BatchNormalization(axis=bn_axis),
            keras.layers.Activation(SiLU),
        ]

        self.top = [
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(5, activation="softmax"),
        ]

    def call(self, inputs):
        Z = inputs
        for layer in self.first_conv:
            Z = layer(Z)
        for layer in self.MB_layers:
            Z = layer(Z)
        for layer in self.last_conv:
            Z = layer(Z)
        for layer in self.top:
            Z = layer(Z)
        return Z

    def get_config(self):
        basic_config = super().get_config()
        return {
            **basic_config,
            "default_efficient": self.default_efficient,
            "width_coef": self.width_coef,
            "depth_coef": self.depth_coef,
            "resolution": self.resolution,
            "dropout": self.dropout,
            "dropout_connect": self.dropout_connect,
        }

    def model(self):
        x = tf.keras.layers.Input(shape=(self.resolution, self.resolution, 3))
        model = keras.models.Model(inputs=x, outputs=self.call(x))
        model.compile(
            optimizer=keras.optimizers.RMSprop(),
            loss=condition_penalty_loss(1),
            metrics=[keras.metrics.CategoricalAccuracy()],
        )
        return model




## === cell 14
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_categorical_accuracy", mode="max", patience=4, verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        "EfficientNet_best.h5", save_best_only=True, monitor="val_loss", mode="min"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=2, min_lr=1e-6, verbose=1
    ),
]




## === cell 15
class_weights = {}
count_by_class = get_count_by_class(train)

print(count_by_class)
mi = 987654321
for i in range(len(count_by_class)):
    class_weights[i] = (1 / count_by_class[i][1][0]) * len(train) / 2.0
    mi = min(mi, class_weights[i])
print(mi)
for k, v in class_weights.items():
    class_weights[k] /= mi
    class_weights[k] *= 1.5
class_weights[4] *= 4
print(class_weights)




## === cell 16
"""
Original alternative model block kept as comment (unchanged).
"""




## === cell 17
MODEL_PATH = "../input/efficientnet-day7/EfficientNet_day7.h5"
efficientNet_B3 = None

if os.path.isfile(MODEL_PATH):
    efficientNet_B3 = keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "condition_penalty_loss": condition_penalty_loss,
            "SiLU": SiLU,
            "MBConv_v2": MBConv_v2,
            "SEUnit": SEUnit,
        },
        compile=False,
    )
    try:
        efficientNet_B3.compile(
            optimizer=keras.optimizers.RMSprop(),
            loss=condition_penalty_loss(1),
            metrics=[keras.metrics.CategoricalAccuracy()],
        )
    except Exception:
        pass
    print("Using previous Model:", MODEL_PATH)
else:
    efficientNet_B3 = EfficientNet(default_efficient, 1.2, 1.4, 512, 0.3).model()
    print("Using New Model (will train) because pretrained file not found:", MODEL_PATH)

    history = efficientNet_B3.fit(
        train_set,
        validation_data=val_set,
        epochs=3,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
    )

    if os.path.isfile("EfficientNet_best.h5"):
        try:
            efficientNet_B3 = keras.models.load_model(
                "EfficientNet_best.h5",
                custom_objects={
                    "condition_penalty_loss": condition_penalty_loss,
                    "SiLU": SiLU,
                    "MBConv_v2": MBConv_v2,
                    "SEUnit": SEUnit,
                },
                compile=False,
            )
            efficientNet_B3.compile(
                optimizer=keras.optimizers.RMSprop(),
                loss=condition_penalty_loss(1),
                metrics=[keras.metrics.CategoricalAccuracy()],
            )
            print("Loaded best checkpoint: EfficientNet_best.h5")
        except Exception as e:
            print(
                "Could not reload EfficientNet_best.h5, using in-memory model. Error:",
                repr(e),
            )




## === cell 18
class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", use_se=False, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.strides = strides
        self.activation_name = activation
        self.use_se = use_se
        self.activation = keras.activations.get(activation)

        self.main_layer = [
            keras.layers.Conv2D(
                filters, 3, strides=strides, padding="same", use_bias=False
            ),
            keras.layers.BatchNormalization(),
            keras.layers.Activation(self.activation),
            keras.layers.Conv2D(filters, 3, strides=1, padding="same", use_bias=False),
            keras.layers.BatchNormalization(),
        ]
        if use_se is True:
            self.main_layer.append(SEUnit_Res(filters))

        self.skip_layer = []
        if strides > 1:
            self.skip_layer = [
                keras.layers.Conv2D(
                    filters, 1, strides=strides, padding="same", use_bias=False
                ),
                keras.layers.BatchNormalization(),
            ]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layer:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layer:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "filters": self.filters,
            "strides": self.strides,
            "activation": self.activation_name,
            "use_se": self.use_se,
        }


class SEUnit_Res(keras.layers.Layer):
    def __init__(self, feature_map_len, **kwargs):
        super().__init__(**kwargs)
        self.feature_map_len = feature_map_len
        self.global_avg_pool = keras.layers.GlobalAvgPool2D()
        self.squeeze = keras.layers.Dense(
            max(1, feature_map_len // 16), activation="relu"
        )
        self.excitation = keras.layers.Dense(feature_map_len, activation="sigmoid")

    def call(self, inputs):
        Z = inputs
        Z = self.global_avg_pool(Z)
        Z = self.squeeze(Z)
        excitation_vector = self.excitation(Z)
        excitation_vector = tf.reshape(
            excitation_vector, [-1, 1, 1, self.feature_map_len]
        )
        return inputs * excitation_vector

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "feature_map_len": self.feature_map_len}




## === cell 19
use_se = True
resnet = None
MODEL_PATH = "../input/weighted-se-resnet-16epochs/weighted_se_resent_16epochs.h5"

if os.path.isfile(MODEL_PATH):
    resnet = keras.models.load_model(
        MODEL_PATH,
        custom_objects={"ResidualUnit": ResidualUnit, "SEUnit_Res": SEUnit_Res},
        compile=False,
    )
    try:
        resnet.compile(
            optimizer=keras.optimizers.RMSprop(),
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=[keras.metrics.CategoricalAccuracy()],
        )
    except Exception:
        pass
    print("Using previous Model:", MODEL_PATH)
else:
    print("ResNet file not found; proceeding without ResNet ensemble:", MODEL_PATH)




## === cell 20
test_csv = pd.read_csv(PATH + "sample_submission.csv")

AUTOTUNE = tf.data.AUTOTUNE


def load_and_preprocess(image_id):
    img_bytes = tf.io.read_file(tf.strings.join([PATH, "test_images/", image_id]))
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [512, 512], method="nearest")
    img = tf.cast(img, tf.float32) / 255.0
    return img


batch_size = 64
all_ids = test_csv["image_id"].values

test_options = tf.data.Options()
test_options.deterministic = True
try:
    test_options.experimental_optimization.apply_default_optimizations = True
    test_options.experimental_optimization.map_parallelization = True
    test_options.experimental_optimization.parallel_batch = True
except Exception:
    pass

test_ds = (
    tf.data.Dataset.from_tensor_slices(all_ids)
    .with_options(test_options)
    .map(load_and_preprocess, num_parallel_calls=AUTOTUNE, deterministic=True)
    .batch(batch_size, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

preds_efficient = efficientNet_B3.predict(test_ds, verbose=0)

if resnet is not None:
    preds_resnet = resnet.predict(test_ds, verbose=0)
    blended = preds_efficient * 0.7 + preds_resnet * 0.3
else:
    blended = preds_efficient

test_csv["label"] = np.argmax(blended, axis=-1).astype(int)
test_csv = test_csv[["image_id", "label"]]
test_csv.to_csv("submission.csv", index=False)
print(test_csv.head())
print("Wrote submission.csv with shape:", test_csv.shape)




## === cell 21
"""
Original confusion-matrix analysis cells kept as comments in the source notebook;
omitted here as they are non-executing and not needed for submission generation.
"""
