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

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
protobuf==6.33.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
tf_keras==2.18.0

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

0.0507

# 6. Current score

0.61061

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I fix the first crash by removing the incorrect hard-coded loop length and by mapping labels safely without changing the training objective. Then I resolve the Keras 3 / TF2.18 compatibility errors by consistently using `tf.keras` APIs (not standalone `keras`) and by replacing deprecated imports like `keras.preprocessing.image.ImageDataGenerator`. I also remove the TF-Hub online dependency (it cannot be used offline reliably on Kaggle) and ensure the EfficientNet model is actually trained (the original fit call was commented out), so the pipeline produces a meaningful submission and should move accuracy up toward the target band. Finally, I keep the same general modeling approach (transfer learning classifier + softmax) and write a valid `submission.csv` with `image_id,label`.'
- What this solution (achieved 0.10762) has done: 'I fix the two crashes that prevent any submission from being written: the Protobuf `MessageFactory.GetPrototype` issue caused by forcing the Python protobuf implementation, and the TFRecord parsing error caused by requiring an `image_id` feature that isn’t present in the provided TFRecords. To keep your core approach intact (TFRecords + EfficientNetB7 transfer learning + softmax), I make TFRecord parsing robust by treating `image_id` as optional and, for the test set, reading `image_id` from `sample_submission.csv` in the TFRecord file order (which is stable). Finally, I ensure `submission.csv` is always produced with the required columns and row count, and keep training/inference semantics unchanged apart from the necessary parsing fix.'
- What this solution (achieved 0.61061) has done: 'The timeout is dominated by extremely expensive input processing and model compute: 512×512 decoding/resizing with CPU-heavy augmentation plus EfficientNetB7 forward/backward over ~18k images for up to 10 epochs. To preserve the exact model, loss, and training loop semantics, the main speedups come from (1) removing redundant `.cache()` on the full decoded training set (it forces decoding all images upfront and can thrash disk/RAM), (2) fusing decode+augment into a single map and ensuring parallelism/prefetch are optimal, (3) enabling Keras graph compilation (`run_eagerly=False`) and XLA JIT (safe, numerically equivalent) for faster step execution, and (4) replacing the Python prediction loop with `model.predict(test_ds)` to eliminate per-batch Python overhead while keeping identical outputs. These changes keep the same data, augmentations, batch size, epochs, and architecture, but reduce constant-factor overhead enough to fit the 600s limit.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

import json
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

try:
    cpu = os.cpu_count() or 4
    tf.config.threading.set_intra_op_parallelism_threads(max(2, cpu // 2))
    tf.config.threading.set_inter_op_parallelism_threads(max(2, cpu // 4))
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

SEED = 1337
keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.data.experimental.enable_debug_mode(False)
except Exception:
    pass

print("TF:", tf.__version__)
print("Keras (tf.keras):", keras.__version__)

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = f"{DATA_DIR}/train.csv"
SAMPLE_SUB = f"{DATA_DIR}/sample_submission.csv"
TRAIN_IMG_DIR = f"{DATA_DIR}/train_images"
TEST_IMG_DIR = f"{DATA_DIR}/test_images"
LABEL_MAP_JSON = f"{DATA_DIR}/label_num_to_disease_map.json"
TRAIN_TFREC_DIR = f"{DATA_DIR}/train_tfrecords"
TEST_TFREC_DIR = f"{DATA_DIR}/test_tfrecords"

assert os.path.exists(TRAIN_CSV), TRAIN_CSV
assert os.path.exists(SAMPLE_SUB), SAMPLE_SUB
assert os.path.exists(TRAIN_IMG_DIR), TRAIN_IMG_DIR
assert os.path.exists(TEST_IMG_DIR), TEST_IMG_DIR
assert os.path.exists(LABEL_MAP_JSON), LABEL_MAP_JSON
assert os.path.exists(TRAIN_TFREC_DIR), TRAIN_TFREC_DIR
assert os.path.exists(TEST_TFREC_DIR), TEST_TFREC_DIR

train = pd.read_csv(TRAIN_CSV)
ss = pd.read_csv(SAMPLE_SUB)

print(train.shape, ss.shape)
train.head()



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
pass



## === cell 2
train.info()



## === cell 3
train["label"].unique()



## === cell 4
from PIL import Image

im = Image.open(f"{TRAIN_IMG_DIR}/999616605.jpg")
print("opened")
print(im.size)



## === cell 5
train_path = TRAIN_IMG_DIR
test_path = TEST_IMG_DIR



## === cell 6
with open(LABEL_MAP_JSON, "r") as f:
    json_data = json.load(f)
json_data



## === cell 7
json_data



## === cell 8
label_id_to_name = {
    0: "Cassava Bacterial Blight (CBB)",
    1: "Cassava Brown Streak Disease (CBSD)",
    2: "Cassava Green Mottle (CGM)",
    3: "Cassava Mosaic Disease (CMD)",
    4: "Healthy",
}
train_labels_named = train.copy()
train_labels_named["label"] = (
    train_labels_named["label"].map(label_id_to_name).astype(str)
)
train_labels_named.head()



## === cell 9
train_labels_named.head()



## === cell 10
from tensorflow.keras import layers
from tensorflow.keras import models



## === cell 11
size = 512
bat_size = 16
split = 0.31
epoch = 10



## === cell 12
pass



## === cell 13
AUTOTUNE = tf.data.AUTOTUNE

tfrec_train_files = tf.io.gfile.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec"))
tfrec_test_files = tf.io.gfile.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec"))
tfrec_train_files = sorted(tfrec_train_files)
tfrec_test_files = sorted(tfrec_test_files)

assert len(tfrec_train_files) > 0, "No train tfrecords found"
assert len(tfrec_test_files) > 0, "No test tfrecords found"

_TFREC_FEATURES_TRAIN = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
    "image_id": tf.io.FixedLenFeature([], tf.string, default_value=b""),
}
_TFREC_FEATURES_TEST = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_id": tf.io.FixedLenFeature([], tf.string, default_value=b""),
}


@tf.function
def _decode_and_resize(image_bytes):
    img = tf.image.decode_jpeg(image_bytes, channels=3)
    img = tf.image.resize(
        img, (size, size), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


@tf.function
def _augment(img, seed):
    img = tf.image.stateless_random_flip_left_right(img, seed=seed)
    img = tf.image.stateless_random_flip_up_down(
        img, seed=seed + tf.constant([1, 1], tf.int32)
    )

    b = tf.random.stateless_uniform(
        [],
        seed=seed + tf.constant([2, 2], tf.int32),
        minval=0.0,
        maxval=0.2,
        dtype=tf.float32,
    )
    img = tf.clip_by_value(img + b, 0.0, 1.0)

    return img


try:
    import tensorflow_addons as tfa  # type: ignore

    _HAS_TFA = True
except Exception:
    tfa = None
    _HAS_TFA = False


@tf.function
def _maybe_rotate(img, seed):
    angle = tf.random.stateless_uniform(
        [],
        seed=seed + tf.constant([3, 3], tf.int32),
        minval=-2.0,
        maxval=2.0,
        dtype=tf.float32,
    ) * (np.pi / 180.0)
    if _HAS_TFA:
        img = tfa.image.rotate(
            img, angles=angle, interpolation="NEAREST", fill_mode="nearest"
        )
    return img


@tf.function
def _zoom(img, seed):
    z = tf.random.stateless_uniform(
        [],
        seed=seed + tf.constant([4, 4], tf.int32),
        minval=0.9,
        maxval=1.1,
        dtype=tf.float32,
    )

    def _zoom_in():
        crop_frac = 1.0 / z
        crop_frac = tf.clip_by_value(crop_frac, 0.5, 1.0)
        cropped = tf.image.central_crop(img, crop_frac)
        return tf.image.resize(
            cropped, (size, size), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )

    def _zoom_out():
        pad_frac = (1.0 / z) - 1.0
        pad_frac = tf.clip_by_value(pad_frac, 0.0, 1.0)
        pad_h = tf.cast(tf.round(tf.cast(size, tf.float32) * pad_frac / 2.0), tf.int32)
        pad_w = tf.cast(tf.round(tf.cast(size, tf.float32) * pad_frac / 2.0), tf.int32)
        padded = tf.pad(img, [[pad_h, pad_h], [pad_w, pad_w], [0, 0]], mode="REFLECT")
        return tf.image.resize(
            padded, (size, size), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )

    img2 = tf.cond(z >= 1.0, _zoom_in, _zoom_out)
    return tf.clip_by_value(img2, 0.0, 1.0)


@tf.function
def _augment_full(img, seed):
    img = _augment(img, seed)
    img = _maybe_rotate(img, seed)
    img = _zoom(img, seed)
    return img


train_count = len(train)
val_count = int(round(train_count * split))
train_count_eff = train_count - val_count

labels_from_csv = tf.constant(train["label"].astype(np.int32).values)


@tf.function
def _parse_train_bytes_with_label_and_index(example_proto, idx):
    ex = tf.io.parse_single_example(example_proto, _TFREC_FEATURES_TRAIN)
    image_bytes = ex["image"]
    tfrec_label = tf.cast(ex["label"], tf.int32)
    csv_label = tf.gather(labels_from_csv, tf.cast(idx, tf.int32))
    final_label = tf.where(tfrec_label >= 0, tfrec_label, csv_label)
    y = tf.one_hot(final_label, 5, dtype=tf.float32)
    return image_bytes, y


@tf.function
def _parse_test_bytes(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TFREC_FEATURES_TEST)
    return ex["image"], ex["image_id"]


def _make_train_val_datasets():
    ds_all = tf.data.TFRecordDataset(tfrec_train_files, num_parallel_reads=AUTOTUNE)

    options = tf.data.Options()
    options.experimental_deterministic = True
    options.threading.private_threadpool_size = 0
    options.autotune.enabled = True
    ds_all = ds_all.with_options(options)

    ds_all = ds_all.enumerate()

    ds_all = ds_all.map(
        lambda idx, ex: _parse_train_bytes_with_label_and_index(ex, idx),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )

    val_ds_local = ds_all.take(val_count)
    train_ds_local = ds_all.skip(val_count)

    @tf.function
    def _decode_xy(image_bytes, y):
        x = _decode_and_resize(image_bytes)
        return x, y

    val_ds_local = val_ds_local.map(
        _decode_xy, num_parallel_calls=AUTOTUNE, deterministic=True
    )

    train_ds_local = train_ds_local.shuffle(
        8192, seed=SEED, reshuffle_each_iteration=True
    )

    counter = tf.data.experimental.Counter(start=0, step=1, dtype=tf.int64)
    train_ds_local = tf.data.Dataset.zip((train_ds_local, counter))

    @tf.function
    def _decode_aug_map(xy, idx):
        image_bytes, y = xy
        x = _decode_and_resize(image_bytes)
        seed = tf.stack(
            [
                tf.cast(idx % (2**31 - 1), tf.int32),
                tf.cast((idx // 997) % (2**31 - 1), tf.int32),
            ]
        )
        x = _augment_full(x, seed)
        return x, y

    train_ds_local = train_ds_local.map(
        _decode_aug_map, num_parallel_calls=AUTOTUNE, deterministic=True
    )

    train_ds_local = train_ds_local.batch(bat_size, drop_remainder=False).prefetch(
        AUTOTUNE
    )
    val_ds_local = val_ds_local.batch(bat_size, drop_remainder=False).prefetch(AUTOTUNE)
    return train_ds_local, val_ds_local


train_tfds, val_tfds = _make_train_val_datasets()

class_indices = {str(i): i for i in range(5)}
inv_class_indices = {v: k for k, v in class_indices.items()}



## === cell 14
train_steps = int(np.ceil(train_count_eff / bat_size))
val_steps = int(np.ceil(val_count / bat_size))
train_steps, val_steps



## === cell 15
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    BatchNormalization,
    Dropout,
)

cnn_model = Sequential()
cnn_model.add(
    Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(size, size, 3))
)
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(BatchNormalization())
cnn_model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(BatchNormalization())
cnn_model.add(Conv2D(96, kernel_size=(3, 3), activation="relu"))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(BatchNormalization())
cnn_model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(BatchNormalization())
cnn_model.add(Conv2D(256, kernel_size=(3, 3), activation="relu"))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(BatchNormalization())
cnn_model.add(Dropout(0.2))
cnn_model.add(Flatten())
cnn_model.add(Dense(64, activation="relu"))
cnn_model.add(Dense(64, activation="relu"))
cnn_model.add(
    Dense(
        128,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
        bias_regularizer=tf.keras.regularizers.l2(1e-4),
        activity_regularizer=tf.keras.regularizers.l2(1e-5),
    )
)
cnn_model.add(
    Dense(
        256,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
        bias_regularizer=tf.keras.regularizers.l2(1e-4),
        activity_regularizer=tf.keras.regularizers.l2(1e-5),
    )
)
cnn_model.add(Dropout(0.25))
cnn_model.add(Dense(5, activation="softmax"))

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
cnn_model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])



## === cell 16
callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    min_delta=0.0,
    patience=2,
    verbose=1,
    mode="max",
    restore_best_weights=True,
)



## === cell 17
pass



## === cell 18
from tensorflow.keras.applications import EfficientNetB7

backbone = EfficientNetB7(
    include_top=False, weights="imagenet", input_shape=(size, size, 3)
)
backbone.trainable = False

inputs = keras.Input(shape=(size, size, 3))
x = backbone(inputs, training=False)
x = keras.layers.Flatten()(x)
outputs = keras.layers.Dense(5, activation="softmax")(x)
model = keras.Model(inputs, outputs)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    run_eagerly=False,
)



## === cell 19
model.summary()



## === cell 20
history = model.fit(
    train_tfds,
    epochs=epoch,
    validation_data=val_tfds,
    verbose=1,
    callbacks=[callback],
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
)



## === cell 21
pass



## === cell 22
pass



## === cell 23
pass



## === cell 24
pass



## === cell 25
pass




## === cell 26
def _make_test_dataset():
    ds = tf.data.TFRecordDataset(tfrec_test_files, num_parallel_reads=AUTOTUNE)
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.autotune.enabled = True
    ds = ds.with_options(options)
    ds = ds.map(_parse_test_bytes, num_parallel_calls=AUTOTUNE, deterministic=True)

    @tf.function
    def _decode_test(image_bytes, image_id):
        x = _decode_and_resize(image_bytes)
        return x, image_id

    ds = ds.map(_decode_test, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(bat_size, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


test_tfds = _make_test_dataset()


def _split_test_ds(ds):
    xds = ds.map(lambda x, image_id: x, num_parallel_calls=AUTOTUNE, deterministic=True)
    ids = ds.map(
        lambda x, image_id: image_id, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    return xds, ids


test_xds, test_ids_ds = _split_test_ds(test_tfds)

probs = model.predict(test_xds, verbose=1)
test_ids = tf.concat(list(test_ids_ds.unbatch().batch(4096)), axis=0).numpy().tolist()

pred_class_idx = np.argmax(probs, axis=1).astype(np.int32)

test_ids = [
    x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in test_ids
]

if len(test_ids) != len(ss):
    raise RuntimeError(
        f"Mismatch between decoded test ids ({len(test_ids)}) "
        f"and sample_submission ({len(ss)})."
    )

decoded_nonempty = sum(1 for x in test_ids if x)
if decoded_nonempty == len(ss):
    test_image_ids = test_ids
else:
    test_image_ids = ss["image_id"].tolist()

pred_df = pd.DataFrame({"image_id": test_image_ids, "label": pred_class_idx.tolist()})

my_submission = ss[["image_id"]].merge(pred_df, on="image_id", how="left")
assert my_submission["label"].notna().all(), "Some test image_ids missing after merge"
my_submission["label"] = my_submission["label"].astype(int)

my_submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", my_submission.shape)
my_submission.head()



## === cell 27
my_submission



## === cell 28
assert os.path.exists("submission.csv")
sub_chk = pd.read_csv("submission.csv")
assert list(sub_chk.columns) == ["image_id", "label"]
assert len(sub_chk) == len(ss)
assert sub_chk["label"].between(0, 4).all()
sub_chk.head()



## === cell 29
model.save("efficientnetb7_cassava.keras")
print("Saved model to efficientnetb7_cassava.keras")
