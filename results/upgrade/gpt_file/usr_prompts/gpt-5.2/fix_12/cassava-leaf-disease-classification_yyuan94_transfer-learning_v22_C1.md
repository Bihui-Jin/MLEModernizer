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

# 5. Target score

0.2311876699909338

# 6. Current score

0.12631

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.12631) has done: 'I fix the two blockers that prevent the notebook from running end-to-end: (1) the protobuf/TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing TensorFlow to use the pure-Python protobuf implementation, and (2) the missing EfficientNet weight file path by falling back to `weights="imagenet"` when the provided local `.h5` does not exist. These changes keep the same model architecture and training loop, but make them robust to the Kaggle environment. I also make the train/test image paths resilient to the dataset being mounted in either `/kaggle/input/...` or `/kaggle/data/...`, and ensure `submission.csv` is always written with the required columns and aligned to `sample_submission.csv`. No score-tuning beyond restoring the intended pretrained backbone is introduced; this should naturally move accuracy above the target band from “not yielded” to a valid baseline.'
- What this solution (achieved 0.12631) has done: 'The timeout is dominated by expensive JPEG decoding + 512→300 resizing done twice (512 cache write then 300 resize) and by spending time in plotting/summary/debug cells; the model training itself is already fairly optimized with `steps_per_execution`. I keep the exact same model, losses, epochs, and split/resampling logic, but make the input pipeline provably equivalent and faster by caching *final 300×300 float32 tensors* (no augmentation) and applying augmentation only on-the-fly from that cache, eliminating the extra resize and the heavy disk cache of 512×512 tensors. I also remove dataset warmup iterations and other non-essential visualization/printing work that costs time but doesn’t affect training or predictions. Finally, I switch the train/val TF cache from disk to in-memory (safe for these dataset sizes at 300×300) to avoid slow `/kaggle/working` I/O.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import json
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import tensorflow as tf
import tensorflow.keras.backend as K

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

print("TF version:", tf.__version__)

CANDIDATE_BASES = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
]
BASE_PATH = None
for p in CANDIDATE_BASES:
    if tf.io.gfile.exists(p):
        BASE_PATH = p
        break
if BASE_PATH is None:
    BASE_PATH = CANDIDATE_BASES[0]

TRAIN_CSV_PATH = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_PATH, "sample_submission.csv")
LABEL_MAP_PATH = os.path.join(BASE_PATH, "label_num_to_disease_map.json")
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train_images")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test_images")

TRAIN_TFREC_DIR = os.path.join(BASE_PATH, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(BASE_PATH, "test_tfrecords")

print("Using BASE_PATH:", BASE_PATH)
print("Train CSV exists:", tf.io.gfile.exists(TRAIN_CSV_PATH))
print("Train images dir exists:", tf.io.gfile.exists(TRAIN_IMG_DIR))
print("Test images dir exists:", tf.io.gfile.exists(TEST_IMG_DIR))
print("Train tfrecords dir exists:", tf.io.gfile.exists(TRAIN_TFREC_DIR))
print("Test tfrecords dir exists:", tf.io.gfile.exists(TEST_TFREC_DIR))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_csv = pd.read_csv(TRAIN_CSV_PATH)
print("Number of train images: {}".format(len(train_csv)))



## === cell 2
train_csv.head()



## === cell 3
with open(LABEL_MAP_PATH, "r") as fp:
    class_map = json.load(fp)
class_map



## === cell 4
pass



## === cell 5
pass



## === cell 6
pass



## === cell 7
input_shape = (300, 300, 3)
num_classes = 5  # fixed by competition

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomCrop(height=512, width=512),
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.25),
        tf.keras.layers.RandomZoom((-0.2, 0.0)),
        tf.keras.layers.RandomContrast((0.0, 0.2)),
    ]
)


def process_data(path, label, training=False):
    full_path = tf.strings.join([tf.constant(TRAIN_IMG_DIR + "/"), path])
    img = tf.io.read_file(full_path)
    img = tf.image.decode_jpeg(img, channels=3, dct_method="INTEGER_FAST")  # uint8

    img = tf.image.resize(img, [512, 512], method="bilinear")
    img_f = tf.cast(img, tf.float32)

    if training:
        img_f = data_augmentation(img_f, training=True)

    img_f = tf.image.resize(img_f, [input_shape[0], input_shape[1]], method="bilinear")
    return img_f, tf.one_hot(label, num_classes)




## === cell 8
train, val = train_test_split(
    train_csv, test_size=0.1, random_state=SEED, stratify=train_csv["label"]
)



## === cell 9
label_counts = train["label"].value_counts()
target_count = int(label_counts.loc[3] * 0.8)

oversampled_df = []
for i in range(len(class_map)):
    class_i = train[train.label == i]
    oversampled_df.append(class_i.sample(target_count, replace=True, random_state=SEED))



## === cell 10
resampled_train = pd.concat(oversampled_df, axis=0)
resampled_train.pivot_table(columns="label", aggfunc="size")



## === cell 11
resampled_train = resampled_train.sample(frac=1, random_state=SEED).reset_index(
    drop=True
)
resampled_train.head()



## === cell 12
batch_size = 8



## === cell 13
print("Planned batch_size:", batch_size)
print(
    "Train rows:",
    len(train),
    "Resampled train rows:",
    len(resampled_train),
    "Val rows:",
    len(val),
)




## === cell 14
def _set_if_exists(obj, name, value):
    try:
        setattr(obj, name, value)
        return True
    except Exception:
        return False


train_paths = (TRAIN_IMG_DIR + "/" + resampled_train.image_id.astype(str)).values
val_paths = (TRAIN_IMG_DIR + "/" + val.image_id.astype(str)).values

train_labels = resampled_train.label.values.astype(np.int32)
val_labels = val.label.values.astype(np.int32)

train_options = tf.data.Options()
train_options.experimental_deterministic = False
_set_if_exists(train_options, "deterministic", False)
_set_if_exists(train_options.experimental_optimization, "map_parallelization", True)
_set_if_exists(train_options.experimental_optimization, "parallel_batch", True)
_set_if_exists(train_options.experimental_optimization, "map_and_batch_fusion", True)
_set_if_exists(train_options.experimental_optimization, "autotune_buffers", True)

val_options = tf.data.Options()
val_options.experimental_deterministic = False
_set_if_exists(val_options, "deterministic", False)
_set_if_exists(val_options.experimental_optimization, "map_parallelization", True)
_set_if_exists(val_options.experimental_optimization, "parallel_batch", True)
_set_if_exists(val_options.experimental_optimization, "map_and_batch_fusion", True)
_set_if_exists(val_options.experimental_optimization, "autotune_buffers", True)


@tf.function
def decode_and_resize_to_300(full_path, label):
    img = tf.io.read_file(full_path)
    img = tf.image.decode_jpeg(img, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(
        img, [input_shape[0], input_shape[1]], method="bilinear", antialias=False
    )
    img = tf.cast(img, tf.float32)
    return img, label


@tf.function
def augment_300(img_f, label):
    img_f = data_augmentation(img_f, training=True)
    return img_f, tf.one_hot(label, num_classes)


@tf.function
def noaug_300(img_f, label):
    return img_f, tf.one_hot(label, num_classes)


train_base = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .with_options(train_options)
    .map(decode_and_resize_to_300, num_parallel_calls=tf.data.AUTOTUNE)
    .cache()  # in-memory cache (faster than disk for this size)
)

train_ds = (
    train_base.map(augment_300, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(buffer_size=2000, seed=SEED, reshuffle_each_iteration=True)
    .batch(batch_size, drop_remainder=False)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    .with_options(val_options)
    .map(decode_and_resize_to_300, num_parallel_calls=tf.data.AUTOTUNE)
    .cache()  # in-memory cache
    .map(noaug_300, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size, drop_remainder=False)
    .prefetch(tf.data.AUTOTUNE)
)



## === cell 15
TRAIN_TFRECS = sorted(tf.io.gfile.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec")))
TEST_TFRECS = sorted(tf.io.gfile.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec")))
print("Found train tfrecs:", len(TRAIN_TFRECS), "test tfrecs:", len(TEST_TFRECS))

train_id_set = tf.constant(resampled_train.image_id.astype(str).values)
val_id_set = tf.constant(val.image_id.astype(str).values)

train_id_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=train_id_set, values=tf.ones_like(train_id_set, dtype=tf.int64)
    ),
    default_value=tf.constant(0, tf.int64),
)
val_id_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=val_id_set, values=tf.ones_like(val_id_set, dtype=tf.int64)
    ),
    default_value=tf.constant(0, tf.int64),
)




## === cell 16
pass



## === cell 17
pass




## === cell 18
def plot_metrics(history, metrics=["loss", "accuracy"]):
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n + 1)
        plt.plot(history.epoch, history.history[metric], label="Train")
        plt.plot(
            history.epoch, history.history["val_" + metric], linestyle="--", label="Val"
        )
        plt.xlabel("Epoch")
        plt.ylabel(name)
        if metric == "loss":
            plt.ylim([0, plt.ylim()[1]])
        else:
            plt.ylim([0, 1])
    plt.legend()




## === cell 19
def plot_cm(labels, predictions):
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion matrix")
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")
    plt.show()




## === cell 20
def sigmoid_focal_crossentropy(
    y_true, y_pred, alpha=0.25, gamma=2.0, from_logits=False
):
    if gamma and gamma < 0:
        raise ValueError("Value of gamma should be greater than or equal to zero")

    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.convert_to_tensor(y_true, dtype=y_pred.dtype)

    ce = K.binary_crossentropy(y_true, y_pred, from_logits=from_logits)

    if from_logits:
        pred_prob = tf.sigmoid(y_pred)
    else:
        pred_prob = y_pred

    p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
    alpha_factor = 1.0
    modulating_factor = 1.0

    if alpha:
        alpha = tf.convert_to_tensor(alpha, dtype=K.floatx())
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)

    if gamma:
        gamma = tf.convert_to_tensor(gamma, dtype=K.floatx())
        modulating_factor = tf.pow((1.0 - p_t), gamma)

    return tf.reduce_sum(alpha_factor * modulating_factor * ce, axis=-1)




## === cell 21
def build_efficient_model(
    input_layer, input_shape, model_inputs, num_classes, dropout_rate=0.2
):
    model = tf.keras.applications.EfficientNetB3(
        weights="/kaggle/input/efficientnetb3notop/efficientnetb3_notop.h5",
        include_top=False,
        input_shape=input_shape,
    )

    model.trainable = False
    model_output = model(model_inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(model_output)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="pred")(x)
    model = tf.keras.Model(input_layer, outputs, name="EfficientNet")
    return model




## === cell 22
dropout_rate = 0.2
num_classes = len(class_map)



## === cell 23
EFFICIENTNET_LOCAL_WEIGHTS = "/kaggle/input/efficientnetb3notop/efficientnetb3_notop.h5"
effnet_weights = (
    EFFICIENTNET_LOCAL_WEIGHTS
    if tf.io.gfile.exists(EFFICIENTNET_LOCAL_WEIGHTS)
    else "imagenet"
)
print("EfficientNetB3 weights:", effnet_weights)

input_layer = tf.keras.layers.Input(shape=input_shape, dtype=tf.float32)
x = input_layer
base_model = tf.keras.applications.EfficientNetB3(
    weights=effnet_weights,
    include_top=False,
    input_shape=input_shape,
)
base_model.trainable = False

model_output = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(model_output)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="pred")(x)
model = tf.keras.Model(input_layer, outputs, name="EfficientNet")



## === cell 25
layers = [layer.name for layer in base_model.layers]
(layers.index("block7a_expand_conv"), len(layers))



## === cell 26
METRICS = [
    tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
    tf.keras.metrics.AUC(name="auc"),
]
optimizer = tf.keras.optimizers.Adam()



## === cell 27
model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=METRICS,
    steps_per_execution=64,
)



## === cell 28
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath="best_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=False,
        mode="max",
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=1, verbose=1
    ),
]

hist = model.fit(train_ds, epochs=5, validation_data=val_ds, callbacks=callbacks)



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1632000195.py in <cell line: 0>()
     13 ]
     14 
---> 15 hist = model.fit(train_ds, epochs=5, validation_data=val_ds, callbacks=callbacks)
     16 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/layers/input_spec.py in assert_input_compatibility(input_spec, inputs, layer_name)
    243                 if spec_dim is not None and dim is not None:
    244                     if spec_dim != dim:
--> 245                         raise ValueError(
    246                             f'Input {input_index} of layer "{layer_name}" is '
    247                             "incompatible with the layer: "

ValueError: Input 0 of layer "EfficientNet" is incompatible with the layer: expected shape=(None, 300, 300, 3), found shape=(None, 512, 512, 3)

## === cell 29
pass



## === cell 30
pass



## === cell 31
pass



## === cell 32
pass



## === cell 33
fine_tune_at = layers.index("block7a_expand_conv")
base_model.trainable = True
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-4, decay_steps=300 * 10
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(lr_schedule),
    loss="categorical_crossentropy",
    metrics=METRICS,
    steps_per_execution=64,
)



## === cell 34
fine_tune_epochs = 5
total_epochs = 5 + fine_tune_epochs

history_fine = model.fit(
    train_ds,
    epochs=total_epochs,
    initial_epoch=hist.epoch[-1] + 1,
    validation_data=val_ds,
    callbacks=callbacks,
)



## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3001517893.py in <cell line: 0>()
      5     train_ds,
      6     epochs=total_epochs,
----> 7     initial_epoch=hist.epoch[-1] + 1,
      8     validation_data=val_ds,
      9     callbacks=callbacks,

NameError: name 'hist' is not defined

## === cell 35
pass



## === cell 36
pass



## === cell 37
pass



## === cell 38
pass



## === cell 39
pass




## === cell 40
@tf.function
def process_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(
        img, [input_shape[0], input_shape[1]], method="bilinear", antialias=False
    )
    img = tf.cast(img, tf.float32)
    return img




## === cell 41
TEST_FILENAMES = tf.io.gfile.glob(os.path.join(TEST_IMG_DIR, "*.jpg"))
print("Found test images:", len(TEST_FILENAMES))



## === cell 42
test_options = tf.data.Options()
test_options.experimental_deterministic = False
_set_if_exists(test_options, "deterministic", False)
_set_if_exists(test_options.experimental_optimization, "map_parallelization", True)
_set_if_exists(test_options.experimental_optimization, "parallel_batch", True)
_set_if_exists(test_options.experimental_optimization, "map_and_batch_fusion", True)
_set_if_exists(test_options.experimental_optimization, "autotune_buffers", True)

test_ds = (
    tf.data.Dataset.from_tensor_slices(TEST_FILENAMES)
    .with_options(test_options)
    .map(process_img, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size, drop_remainder=False)
    .prefetch(tf.data.AUTOTUNE)
)

probabilities = model.predict(test_ds, verbose=0)
predictions = np.argmax(probabilities, axis=-1)

test_ids = np.array([os.path.basename(p) for p in TEST_FILENAMES], dtype=str)

submission = pd.DataFrame({"image_id": test_ids, "label": predictions})

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
submission = sample_sub[["image_id"]].merge(submission, on="image_id", how="left")
if submission["label"].isna().any():
    majority = int(train_csv["label"].mode().iloc[0])
    submission["label"] = submission["label"].fillna(majority).astype(int)
else:
    submission["label"] = submission["label"].astype(int)

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
submission.head()
