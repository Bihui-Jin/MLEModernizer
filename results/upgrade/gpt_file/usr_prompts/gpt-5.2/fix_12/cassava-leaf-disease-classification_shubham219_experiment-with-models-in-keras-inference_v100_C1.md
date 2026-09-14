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

3.11

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
import os, glob, random

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"

print("TF:", tf.__version__)
print("Data root:", DATA_ROOT)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass



## === cell 1
import matplotlib.pyplot as plt
import tensorflow.keras.layers as tfl
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split



## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
train_df["path"] = TRAIN_IMG_DIR + "/" + train_df["image_id"].astype(str)
train_df["label"] = train_df["label"].astype(str)  # keep same semantics as original

train_df, val_df = train_test_split(
    train_df, test_size=0.1, random_state=SEED, stratify=train_df["label"]
)

IMG_SIZE = (256, 256)
BATCH_SIZE = 16
N_CLASSES = 5

preprocess = tf.keras.applications.resnet50.preprocess_input
AUTOTUNE = tf.data.AUTOTUNE


def _label_str_to_int(y):
    return tf.strings.to_number(y, out_type=tf.int32)


@tf.function
def _decode_resize_from_path(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(
        img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32)
    return img


rot_layer = tf.keras.layers.RandomRotation(
    factor=10.0 / 180.0, fill_mode="reflect", seed=SEED
)
trans_layer = tf.keras.layers.RandomTranslation(
    height_factor=0.05, width_factor=0.05, fill_mode="reflect", seed=SEED + 1
)
zoom_layer = tf.keras.layers.RandomZoom(
    height_factor=0.1, width_factor=0.1, fill_mode="reflect", seed=SEED + 2
)
flip_layer = tf.keras.layers.RandomFlip(mode="horizontal", seed=SEED + 3)


def _augment(img):
    img = flip_layer(img, training=True)
    img = rot_layer(img, training=True)
    img = trans_layer(img, training=True)
    img = zoom_layer(img, training=True)
    return img


def _prep_train_from_path(path, y):
    img = _decode_resize_from_path(path)
    img = _augment(img)
    img = preprocess(img)
    y_int = _label_str_to_int(y)
    y_oh = tf.one_hot(y_int, depth=N_CLASSES, dtype=tf.float32)
    return img, y_oh


def _prep_val_from_path(path, y):
    img = _decode_resize_from_path(path)
    img = preprocess(img)
    y_int = _label_str_to_int(y)
    y_oh = tf.one_hot(y_int, depth=N_CLASSES, dtype=tf.float32)
    return img, y_oh


train_paths = train_df["path"].to_numpy()
train_labels = train_df["label"].to_numpy()
val_paths = val_df["path"].to_numpy()
val_labels = val_df["label"].to_numpy()

ds_opts = tf.data.Options()
ds_opts.experimental_deterministic = True

try:
    ds_opts.experimental_optimization.apply_default_optimizations = True
except Exception:
    pass
for _name in [
    "map_parallelization",
    "map_and_batch_fusion",
    "parallel_batch",
    "filter_fusion",
    "map_fusion",
    "map_vectorization",
]:
    try:
        setattr(ds_opts.experimental_optimization, _name, True)
    except Exception:
        pass

try:
    ds_opts.experimental_slack = True
except Exception:
    pass

shuffle_buf = int(min(len(train_df), 4096))

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels)).with_options(
    ds_opts
)


@tf.function
def _decode_pair(path, y):
    return _decode_resize_from_path(path), y


train_ds = train_ds.map(_decode_pair, num_parallel_calls=AUTOTUNE)

train_ds = train_ds.cache()

train_ds = train_ds.shuffle(
    buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True
)


@tf.function
def _prep_train_from_decoded(img, y):
    img = _augment(img)
    img = preprocess(img)
    y_int = _label_str_to_int(y)
    y_oh = tf.one_hot(y_int, depth=N_CLASSES, dtype=tf.float32)
    return img, y_oh


train_ds = train_ds.map(_prep_train_from_decoded, num_parallel_calls=AUTOTUNE)

train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False)
train_ds = train_ds.prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels)).with_options(
    ds_opts
)
val_ds = val_ds.map(_decode_pair, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.cache()


@tf.function
def _prep_val_from_decoded(img, y):
    img = preprocess(img)
    y_int = _label_str_to_int(y)
    y_oh = tf.one_hot(y_int, depth=N_CLASSES, dtype=tf.float32)
    return img, y_oh


val_ds = val_ds.map(_prep_val_from_decoded, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

train_ds = train_ds.apply(tf.data.experimental.ignore_errors())
val_ds = val_ds.apply(tf.data.experimental.ignore_errors())

base = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
)
base.trainable = False  # keep minimal + stable

inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base(inputs, training=False)
x = tfl.GlobalAveragePooling2D()(x)
x = tfl.Dropout(0.2)(x)
outputs = tfl.Dense(N_CLASSES, activation="softmax")(x)
my_model = tf.keras.Model(inputs, outputs)

my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

EPOCHS = 3
steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
if (
    steps_per_epoch == tf.data.experimental.INFINITE_CARDINALITY
    or steps_per_epoch == tf.data.experimental.UNKNOWN_CARDINALITY
):
    steps_per_epoch = max(1, len(train_df) // BATCH_SIZE)
val_steps = tf.data.experimental.cardinality(val_ds).numpy()
if (
    val_steps == tf.data.experimental.INFINITE_CARDINALITY
    or val_steps == tf.data.experimental.UNKNOWN_CARDINALITY
):
    val_steps = max(1, len(val_df) // BATCH_SIZE)

history = my_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=int(steps_per_epoch),
    validation_steps=int(val_steps),
    verbose=1,
)



## === cell 3
test_images = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg")))
test_paths = np.array(test_images, dtype=object)


@tf.function
def _prep_test_from_path(path):
    img = _decode_resize_from_path(path)
    img = preprocess(img)
    return img


def make_test_ds(batch_size=16):
    ds = tf.data.Dataset.from_tensor_slices(test_paths).with_options(ds_opts)
    ds = ds.map(_prep_test_from_path, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    return ds




## === cell 4
test_ds = make_test_ds(batch_size=16)
pred_test = my_model.predict(test_ds, verbose=1)
pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

sample_sub = pd.read_csv(SAMPLE_SUB)
pred_map = {
    os.path.basename(p): int(lbl) for p, lbl in zip(test_images, pred_test_labels)
}
sample_sub["label"] = sample_sub["image_id"].map(pred_map).astype(int)

sample_sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sample_sub.shape)
sample_sub.head()
