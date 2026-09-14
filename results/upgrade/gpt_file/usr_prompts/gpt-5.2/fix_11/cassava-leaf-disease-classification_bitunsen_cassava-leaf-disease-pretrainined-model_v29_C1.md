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
import json
import numpy as np
import pandas as pd



## === cell 1
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images/"

TRAIN_TFREC_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_tfrecords/"
TEST_TFREC_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_tfrecords/"



## === cell 2
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
from PIL import Image

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

try:
    tf.config.optimizer.set_jit(False)  # keep determinism closer to baseline
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass



## === cell 3
with open(os.path.join(BASE_DIR, "label_num_to_disease_map.json")) as file:
    map_classes = json.loads(file.read())

print(json.dumps(map_classes, indent=2))



## === cell 4
label_list = [int(key) for key in map_classes.keys()]
print("Labels:", label_list)



## === cell 5
print("Train image dir:", os.path.join(BASE_DIR, "train_images"))



## === cell 6
IMG_HEIGHT = 400
IMG_WIDTH = 400
batch_size = 32

PRE_TRAINED_MODEL = (
    "../input/inceptionresnetv4/Cassava_Best_InceptionResNet_Model_V04.hdf5"
)



## === cell 7
train_df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
sample_sub = pd.read_csv(os.path.join(BASE_DIR, "sample_submission.csv"))

print(train_df.head(3))
print(sample_sub.head(3))
print("Train rows:", len(train_df), "Test rows:", len(sample_sub))



## === cell 8
from sklearn.model_selection import train_test_split

train_df["filepath"] = [os.path.join(TRAIN_DIR, x) for x in train_df["image_id"].values]

train_split, val_split = train_test_split(
    train_df, test_size=0.2, random_state=42, stratify=train_df["label"]
)

print("Train split:", train_split.shape, "Val split:", val_split.shape)



## === cell 9
from tensorflow.keras.applications.inception_resnet_v2 import (
    InceptionResNetV2,
    preprocess_input,
)

num_classes = 5

base_model = InceptionResNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    pooling="avg",
)
base_model.trainable = False

inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()



## === cell 10
AUTOTUNE = tf.data.AUTOTUNE


def _tfrecord_files_in_dir(directory):
    files = tf.io.gfile.glob(os.path.join(directory, "*.tfrec"))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No .tfrec files found in: {directory}")
    return files


_FEATURE_DESC_TRAIN = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}
_FEATURE_DESC_TEST = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


@tf.function
def _parse_train_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURE_DESC_TRAIN)
    img = tf.io.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH], method="bilinear")
    img = tf.cast(img, tf.float32)
    label = tf.cast(ex["target"], tf.int32)
    return img, label


@tf.function
def _parse_test_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURE_DESC_TEST)
    img = tf.io.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH], method="bilinear")
    img = tf.cast(img, tf.float32)
    image_id = ex["image_name"]
    return img, image_id


@tf.function
def _augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.10)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    return img, label


options = tf.data.Options()
try:
    options.experimental_deterministic = True
except Exception:
    pass
try:
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.autotune_buffers = True
except Exception:
    pass
try:
    options.threading.private_threadpool_size = 0
    options.threading.max_intra_op_parallelism = 0
except Exception:
    pass



## === cell 11
train_ids_set = set(train_split["image_id"].values.tolist())
val_ids_set = set(val_split["image_id"].values.tolist())

train_keys = tf.constant(list(train_ids_set), dtype=tf.string)
val_keys = tf.constant(list(val_ids_set), dtype=tf.string)

train_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        train_keys, tf.ones_like(train_keys, dtype=tf.int32)
    ),
    default_value=0,
)
val_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        val_keys, tf.ones_like(val_keys, dtype=tf.int32)
    ),
    default_value=0,
)


@tf.function
def _is_in_train(img, image_id, label):
    return tf.equal(train_table.lookup(image_id), 1)


@tf.function
def _is_in_val(img, image_id, label):
    return tf.equal(val_table.lookup(image_id), 1)


@tf.function
def _add_id_train(img, label, image_id):
    return img, label


@tf.function
def _parse_train_example_with_id(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURE_DESC_TRAIN)
    img = tf.io.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH], method="bilinear")
    img = tf.cast(img, tf.float32)
    label = tf.cast(ex["target"], tf.int32)
    image_id = ex["image_name"]
    return img, image_id, label


train_tfrec_files = _tfrecord_files_in_dir(TRAIN_TFREC_DIR)

raw_train = tf.data.TFRecordDataset(
    train_tfrec_files, num_parallel_reads=AUTOTUNE
).with_options(options)

parsed = raw_train.map(
    _parse_train_example_with_id, num_parallel_calls=AUTOTUNE, deterministic=True
)

train_ds = parsed.filter(_is_in_train).map(
    lambda img, image_id, label: (img, label),
    num_parallel_calls=AUTOTUNE,
    deterministic=True,
)
train_ds = train_ds.shuffle(4096, seed=42, reshuffle_each_iteration=True)
train_ds = train_ds.map(_augment, num_parallel_calls=AUTOTUNE, deterministic=True)
train_ds = train_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)

val_ds = parsed.filter(_is_in_val).map(
    lambda img, image_id, label: (img, label),
    num_parallel_calls=AUTOTUNE,
    deterministic=True,
)
val_ds = val_ds.cache(os.path.join("/kaggle/working", "val_cache.tf-data"))
val_ds = val_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)



## === cell 12
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3,
    verbose=1,
)

base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history_ft = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=2,
    verbose=1,
)



## === cell 13
try:
    RESAMPLE = Image.Resampling.LANCZOS  # Pillow>=9
except AttributeError:
    RESAMPLE = Image.LANCZOS



## === cell 14
test_df = sample_sub[["image_id"]].copy()
test_samples = test_df.shape[0]
print("Test samples:", test_samples)


@tf.function
def _batch_to_tta(batch_imgs):
    img0 = batch_imgs
    img1 = tf.image.flip_left_right(batch_imgs)
    img2 = tf.image.flip_up_down(batch_imgs)
    img3 = tf.image.rot90(batch_imgs, k=1)
    return tf.stack([img0, img1, img2, img3], axis=1)  # (B, 4, H, W, 3)


test_tfrec_files = _tfrecord_files_in_dir(TEST_TFREC_DIR)
raw_test = tf.data.TFRecordDataset(
    test_tfrec_files, num_parallel_reads=AUTOTUNE
).with_options(options)
test_parsed = raw_test.map(
    _parse_test_example, num_parallel_calls=AUTOTUNE, deterministic=True
)

test_ids_np = test_df["image_id"].values.astype("U")
id_to_index = {k: i for i, k in enumerate(test_ids_np)}

predict_batch_size = 128
test_parsed = test_parsed.cache(os.path.join("/kaggle/working", "test_cache.tf-data"))

test_batched = test_parsed.batch(predict_batch_size, drop_remainder=False)
tta_batched_ds = test_batched.map(
    lambda imgs, ids: (_batch_to_tta(imgs), ids),
    num_parallel_calls=AUTOTUNE,
    deterministic=True,
).prefetch(AUTOTUNE)

mean_probs_out = np.empty((test_samples, num_classes), dtype=np.float32)

for batch_tta, batch_ids in tta_batched_ds:
    b = int(batch_tta.shape[0])
    flat = tf.reshape(batch_tta, [b * 4, IMG_HEIGHT, IMG_WIDTH, 3])
    probs_flat = model.predict(
        flat, batch_size=predict_batch_size, verbose=0
    )  # (B*4, 5)
    probs = probs_flat.reshape((b, 4, num_classes))
    mean_probs = probs.mean(axis=1)  # (B,5)

    batch_ids_np = batch_ids.numpy().astype("U")
    for j in range(b):
        mean_probs_out[id_to_index[batch_ids_np[j]]] = mean_probs[j]

test_results = mean_probs_out.argmax(axis=1).astype(int).tolist()

submission = pd.DataFrame(
    {"image_id": test_df["image_id"].values, "label": test_results}
)
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())



## === cell 15
check = pd.read_csv("submission.csv")
assert list(check.columns) == ["image_id", "label"]
assert len(check) == len(sample_sub)
print("Submission OK:", check.shape)
print(check.head(3))
