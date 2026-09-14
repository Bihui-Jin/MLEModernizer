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
import sys

try:
    from google.protobuf import message_factory

    if not hasattr(message_factory.MessageFactory, "GetPrototype"):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import json
import math
from copy import deepcopy
import matplotlib.pyplot as plt

RESOLUTION = 224

tf.random.set_seed(42)
np.random.seed(42)




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
            data[data["label"] == i].sample(frac=sample_rate[i], replace=True)
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
PATH = "../input/cassava-leaf-disease-classification/"
data = pd.read_csv(PATH + "train.csv")
with open(PATH + "label_num_to_disease_map.json") as f:
    real_labels = json.load(f)
real_labels = {int(k): v for k, v in real_labels.items()}
data["class_name"] = data.label.map(real_labels)

stratified_data = stratifying_data(data)
print(get_count_by_class(stratified_data))

train, val = train_test_split(
    stratified_data, test_size=0.05, stratify=stratified_data["class_name"]
)

print(f"Train samples: {len(train)}, Validation samples: {len(val)}")




## === cell 5
class SEUnit(keras.layers.Layer):
    def __init__(self, feature_map_len, se_ratio, **kwargs):
        super().__init__(**kwargs)
        self.feature_map_len = feature_map_len
        self.se_ratio = se_ratio
        self.global_avg_pool = keras.layers.GlobalAvgPool2D()
        self.reshape = keras.layers.Reshape((1, 1, feature_map_len))
        squeeze_filters = max(1, int(feature_map_len * se_ratio))
        self.squeeze = keras.layers.Conv2D(
            squeeze_filters, kernel_size=1, activation="relu"
        )
        self.excitation = keras.layers.Conv2D(
            feature_map_len, kernel_size=1, activation="sigmoid"
        )

    def call(self, inputs):
        Z = self.global_avg_pool(inputs)
        Z = self.reshape(Z)
        Z = self.squeeze(Z)
        excitation_vector = self.excitation(Z)
        excitation_vector = tf.reshape(
            excitation_vector, [-1, 1, 1, self.feature_map_len]
        )
        return inputs * excitation_vector

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "feature_map_len": self.feature_map_len}




## === cell 6
def SiLU(x):
    return x * tf.keras.backend.sigmoid(x)




## === cell 7
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
        self.activation = keras.layers.Activation(SiLU)
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




## === cell 8
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




## === cell 9
def rf(filters, width_coef):
    depth_divisor = 8
    filters *= width_coef
    new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)


def rd(repeat, depth_coef):
    return int(math.ceil(repeat * depth_coef))




## === cell 10
def condition_tensor(y, target_class):
    return tf.argmax(y, axis=-1) == target_class




## === cell 11
class condition_penalty_loss(tf.keras.losses.Loss):
    def __init__(self, condition_penalty, **kwargs):
        self.condition_penalty = condition_penalty
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        class4 = condition_tensor(y_true, 4)
        predict0 = condition_tensor(y_pred, 0)
        loss4_0 = loss * tf.cast(class4 & predict0, tf.float32) * 5
        class0123 = ~class4
        predict4 = condition_tensor(y_pred, 4)
        loss0123_4 = loss * tf.cast(class0123 & predict4, tf.float32) * 2
        class0 = condition_tensor(y_true, 0)
        loss0_4 = loss * tf.cast(class0 & predict4, tf.float32) * 10
        return loss + loss0123_4 + loss4_0 + loss0_4

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "condition_penalty": self.condition_penalty}




## === cell 12
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
        self.activation = keras.layers.Activation(SiLU)
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
            block_args.pop("n_repeat", None)

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




## === cell 13
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



## === cell 14
class_weights = {}
count_by_class = get_count_by_class(train)
print(count_by_class)
mi = float("inf")
for i in range(len(count_by_class)):
    class_weights[i] = (1 / count_by_class[i][1][0]) * len(train) / 2.0
    mi = min(mi, class_weights[i])
for k, v in class_weights.items():
    class_weights[k] = (v / mi) * 1.5
class_weights[4] *= 4
print(class_weights)



## === cell 15
possible_paths = [
    "../input/efficientnet-day7/EfficientNet_day7.h5",
    "../input/efficientnet_day7/EfficientNet_day7.h5",
    "/kaggle/input/efficientnet-day7/EfficientNet_day7.h5",
    "/kaggle/input/efficientnet_day7/EfficientNet_day7.h5",
]
MODEL_PATH = next((p for p in possible_paths if os.path.isfile(p)), None)

if MODEL_PATH:
    efficientNet_B3 = keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "condition_penalty_loss": condition_penalty_loss,
            "SEUnit": SEUnit,
            "MBConv_v2": MBConv_v2,
            "EfficientNet": EfficientNet,
        },
    )
    print(f"Loaded EfficientNet from {MODEL_PATH}.")
else:
    print("Fine‑tuned checkpoint not found – building a fallback model (no training).")
    fallback = EfficientNet(
        default_efficient=default_efficient,
        width_coef=1.0,
        depth_coef=1.0,
        resolution=RESOLUTION,  # use the faster resolution
        dropout=0.2,
    )
    efficientNet_B3 = fallback.model()
    print("Fallback model built and ready for inference.")



## === cell 16
BATCH_SIZE = 64

train_paths = [os.path.join(PATH, "train_images", img) for img in train.image_id]
val_paths = [os.path.join(PATH, "train_images", img) for img in val.image_id]


def _load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [RESOLUTION, RESOLUTION])
    img = img / 255.0
    return img, label


train_labels = tf.keras.utils.to_categorical(train.label, 5)
val_labels = tf.keras.utils.to_categorical(val.label, 5)

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_ds = (
    train_ds.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)
    .cache()  # <-- cache after image decoding
    .shuffle(1024)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_ds = (
    val_ds.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)
    .cache()  # <-- cache after image decoding
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

print("Starting short training on the model (5 epochs).")
efficientNet_B3.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=2,
)
print("Training completed – proceeding to inference.")



## === cell 17
try:
    MODEL_PATH_RES = (
        "../input/weighted-se-resnet-16epochs/weighted_se_resent_16epochs.h5"
    )
    if os.path.isfile(MODEL_PATH_RES):
        resnet = keras.models.load_model(MODEL_PATH_RES, custom_objects={})
        print("Loaded ResNet‑SE model.")
    else:
        raise FileNotFoundError
except Exception as e:
    print(f"ResNet model not loaded ({e}); proceeding without it.")
    resnet = None



## === cell 18
import tensorflow.keras as tfk

test_csv = pd.read_csv(PATH + "sample_submission.csv")
batch_size = 32
preds_efficient = []
preds_resnet = []

image_paths = [
    os.path.join(PATH, "test_images", img_id) for img_id in test_csv.image_id
]


def _load_test_image(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [RESOLUTION, RESOLUTION])
    img = img / 255.0
    return img


for start in range(0, len(image_paths), batch_size):
    batch_paths = image_paths[start : start + batch_size]
    batch_imgs = [_load_test_image(p).numpy() for p in batch_paths]
    batch_array = np.stack(batch_imgs, axis=0)  # shape (B, RESOLUTION, RESOLUTION, 3)

    batch_pred_eff = efficientNet_B3.predict(
        batch_array, batch_size=len(batch_array), verbose=0
    )
    preds_efficient.extend(batch_pred_eff)

    if resnet is not None:
        batch_pred_res = resnet.predict(
            batch_array, batch_size=len(batch_array), verbose=0
        )
    else:
        batch_pred_res = np.zeros((len(batch_array), 5))
    preds_resnet.extend(batch_pred_res)

preds_efficient = np.array(preds_efficient)
preds_resnet = np.array(preds_resnet)

ensemble = preds_efficient * 0.7 + preds_resnet * 0.3
test_csv["label"] = np.argmax(ensemble, axis=1)
test_csv.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
print(test_csv.head())
