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

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_DISABLE_CPP_IMPLEMENTATION", None)
os.environ.setdefault("PYTHONHASHSEED", "42")

import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import cv2

import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, GlobalAveragePooling2D

random.seed(42)
np.random.seed(42)
tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    for _gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(_gpu, True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

print("TF:", tf.__version__)
print("Visible devices:", tf.config.list_physical_devices())




## === cell 1
data_path = "../input/cassava-leaf-disease-classification/"
train_csv_data_path = data_path + "train.csv"
label_json_data_path = data_path + "label_num_to_disease_map.json"
images_dir_data_path = data_path + "train_images"

train_tfrecords_dir = data_path + "train_tfrecords/"
test_tfrecords_dir = data_path + "test_tfrecords/"




## === cell 2
train_csv = pd.read_csv(train_csv_data_path)
train_csv["label"] = train_csv["label"].astype("string")

label_class = pd.read_json(label_json_data_path, orient="index")
label_class = label_class.values.flatten().tolist()




## === cell 3
train_data_label_3 = train_csv[train_csv["label"] == "3"]
train_data_label_3 = shuffle(train_data_label_3, random_state=42)
train_data_label_3 = train_data_label_3[:3000]

train_data_label_not_3 = train_csv[train_csv["label"] != "3"]

train_csv = pd.concat([train_data_label_3, train_data_label_not_3], ignore_index=True)




## === cell 4
print("Label names :")
for i, label in enumerate(label_class):
    print(f" {i}. {label}")




## === cell 5
train_csv.head()




## === cell 6
BATCH_SIZE = 18
IMG_SIZE = 320
VAL_SPLIT = 0.15

train_csv_sorted = train_csv.sort_values("image_id").reset_index(drop=True)
n_total = len(train_csv_sorted)
n_val = int(np.floor(n_total * VAL_SPLIT))

val_df = train_csv_sorted.iloc[:n_val].copy()
train_df = train_csv_sorted.iloc[n_val:].copy()

train_paths = (images_dir_data_path + "/" + train_df["image_id"]).to_numpy()
val_paths = (images_dir_data_path + "/" + val_df["image_id"]).to_numpy()

train_labels_int = train_df["label"].astype(int).to_numpy()
val_labels_int = val_df["label"].astype(int).to_numpy()

NUM_CLASSES = 5




## === cell 7
AUTOTUNE = tf.data.AUTOTUNE


def _seed_from_id(image_id):
    h = tf.strings.to_hash_bucket_fast(image_id, 2**31 - 1)
    return tf.stack([tf.cast(h, tf.int32), tf.constant(42, tf.int32)], axis=0)


@tf.function
def _decode_resize_rescale_from_bytes(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img,
        [IMG_SIZE, IMG_SIZE],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


@tf.function
def _apply_projective(img, transform):
    transforms = tf.reshape(transform, [1, 8])
    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(img, 0),
        transforms=transforms,
        output_shape=tf.constant([IMG_SIZE, IMG_SIZE], tf.int32),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )
    return tf.squeeze(out, axis=0)


@tf.function
def _augment(img, seed):
    angle = tf.random.stateless_uniform(
        [], seed=seed, minval=-np.pi, maxval=np.pi, dtype=tf.float32
    )
    ca = tf.cos(angle)
    sa = tf.sin(angle)

    cx = (IMG_SIZE - 1) / 2.0
    cy = (IMG_SIZE - 1) / 2.0

    a0 = ca
    a1 = -sa
    b0 = sa
    b1 = ca
    a2 = cx - a0 * cx - a1 * cy
    b2 = cy - b0 * cx - b1 * cy
    rot_transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])
    img = _apply_projective(img, rot_transform)

    s2 = seed + tf.constant([1, 0], tf.int32)
    tx = (
        tf.random.stateless_uniform(
            [], seed=s2, minval=-0.1, maxval=0.1, dtype=tf.float32
        )
        * IMG_SIZE
    )
    s3 = seed + tf.constant([2, 0], tf.int32)
    ty = (
        tf.random.stateless_uniform(
            [], seed=s3, minval=-0.1, maxval=0.1, dtype=tf.float32
        )
        * IMG_SIZE
    )
    img = tf.roll(img, shift=tf.cast(tf.stack([ty, tx]), tf.int32), axis=[0, 1])

    s4 = seed + tf.constant([3, 0], tf.int32)
    b = tf.random.stateless_uniform(
        [], seed=s4, minval=0.1, maxval=0.9, dtype=tf.float32
    )
    img = tf.clip_by_value(img * b, 0.0, 1.0)

    s5 = seed + tf.constant([4, 0], tf.int32)
    shear = tf.random.stateless_uniform(
        [], seed=s5, minval=-25.0, maxval=25.0, dtype=tf.float32
    ) * (np.pi / 180.0)
    s6 = seed + tf.constant([5, 0], tf.int32)
    zoom = tf.random.stateless_uniform(
        [], seed=s6, minval=1.0 - 0.3, maxval=1.0 + 0.3, dtype=tf.float32
    )

    sh = tf.tan(shear)

    fa0 = zoom
    fa1 = zoom * sh
    fb0 = 0.0
    fb1 = zoom
    fa2 = cx - fa0 * cx - fa1 * cy
    fb2 = cy - fb0 * cx - fb1 * cy

    det = fa0 * fb1 - fa1 * fb0
    ia0 = fb1 / det
    ia1 = -fa1 / det
    ib0 = -fb0 / det
    ib1 = fa0 / det
    ia2 = -(ia0 * fa2 + ia1 * fb2)
    ib2 = -(ib0 * fa2 + ib1 * fb2)

    aff_transform = tf.stack([ia0, ia1, ia2, ib0, ib1, ib2, 0.0, 0.0])
    img = _apply_projective(img, aff_transform)

    s7 = seed + tf.constant([6, 0], tf.int32)
    cshift = tf.random.stateless_uniform(
        [1, 1, 3], seed=s7, minval=-0.1, maxval=0.1, dtype=tf.float32
    )
    img = tf.clip_by_value(img + cshift, 0.0, 1.0)

    s8 = seed + tf.constant([7, 0], tf.int32)
    do_h = tf.random.stateless_uniform([], seed=s8, minval=0.0, maxval=1.0) < 0.5
    img = tf.cond(do_h, lambda: tf.image.flip_left_right(img), lambda: img)

    s9 = seed + tf.constant([8, 0], tf.int32)
    do_v = tf.random.stateless_uniform([], seed=s9, minval=0.0, maxval=1.0) < 0.5
    img = tf.cond(do_v, lambda: tf.image.flip_up_down(img), lambda: img)

    return img


@tf.function
def _train_from_cached(img, image_id, y):
    seed = _seed_from_id(image_id)
    img = _augment(img, seed)
    y = tf.one_hot(tf.cast(y, tf.int32), NUM_CLASSES)
    return img, y


@tf.function
def _val_from_cached(img, y):
    y = tf.one_hot(tf.cast(y, tf.int32), NUM_CLASSES)
    return img, y


options = tf.data.Options()
options.experimental_deterministic = True
options.experimental_optimization.apply_default_optimizations = True
options.experimental_optimization.parallel_batch = True
options.experimental_optimization.map_parallelization = True
options.threading.private_threadpool_size = 0
options.threading.max_intra_op_parallelism = 0

train_id_to_label = dict(zip(train_df["image_id"].tolist(), train_labels_int.tolist()))
val_id_to_label = dict(zip(val_df["image_id"].tolist(), val_labels_int.tolist()))


def _make_label_table(dct):
    keys = tf.constant(list(dct.keys()), dtype=tf.string)
    vals = tf.constant(list(dct.values()), dtype=tf.int64)
    init = tf.lookup.KeyValueTensorInitializer(keys, vals)
    return tf.lookup.StaticHashTable(init, default_value=tf.constant(-1, tf.int64))


train_label_table = _make_label_table(train_id_to_label)
val_label_table = _make_label_table(val_id_to_label)


def _parse_train_tfrec(example_proto):
    feats = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
    }
    ex = tf.io.parse_single_example(example_proto, feats)
    img = _decode_resize_rescale_from_bytes(ex["image"])
    image_id = ex["image_name"]
    y = train_label_table.lookup(image_id)
    return img, image_id, y


def _parse_val_tfrec(example_proto):
    feats = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
    }
    ex = tf.io.parse_single_example(example_proto, feats)
    img = _decode_resize_rescale_from_bytes(ex["image"])
    y = val_label_table.lookup(ex["image_name"])
    return img, y


def _cleanup_cache_path(path):
    try:
        if tf.io.gfile.exists(path):
            tf.io.gfile.rmtree(path)
    except Exception:
        pass


run_tag = str(int(time.time()))
train_cache_path = f"/kaggle/working/train_decoded_cache_{run_tag}"
val_cache_path = f"/kaggle/working/val_decoded_cache_{run_tag}"
_cleanup_cache_path(train_cache_path)
_cleanup_cache_path(val_cache_path)

train_tfrec_files = sorted(tf.io.gfile.glob(train_tfrecords_dir + "*.tfrec"))
val_tfrec_files = train_tfrec_files  # same source; labels filtered via table

train_raw = tf.data.TFRecordDataset(train_tfrec_files, num_parallel_reads=AUTOTUNE)
val_raw = tf.data.TFRecordDataset(val_tfrec_files, num_parallel_reads=AUTOTUNE)

shuffle_buf = int(min(len(train_paths), 4096))

train_ds = (
    train_raw.map(_parse_train_tfrec, num_parallel_calls=AUTOTUNE)
    .filter(lambda img, image_id, y: y >= 0)
    .shuffle(buffer_size=shuffle_buf, seed=42, reshuffle_each_iteration=True)
    .cache(train_cache_path)
    .map(
        lambda img, image_id, y: _train_from_cached(img, image_id, y),
        num_parallel_calls=AUTOTUNE,
    )
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
    .with_options(options)
)

val_ds = (
    val_raw.map(_parse_val_tfrec, num_parallel_calls=AUTOTUNE)
    .filter(lambda img, y: y >= 0)
    .cache(val_cache_path)
    .map(_val_from_cached, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
    .with_options(options)
)




## === cell 8
for _x, _y in train_ds.take(1):
    images, labels = _x, _y
    break
print("Sanity check batch shapes:", images.shape, labels.shape)




## === cell 9
pass




## === cell 10
base = tf.keras.applications.ResNet152V2(
    include_top=False, weights="imagenet", input_shape=[IMG_SIZE, IMG_SIZE, 3]
)

base.trainable = False




## === cell 11
base.summary()




## === cell 12
model = tf.keras.Sequential()
model.add(base)
model.add(BatchNormalization(axis=-1))
model.add(GlobalAveragePooling2D())
model.add(Dense(5, activation="softmax"))




## === cell 13
model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adamax(learning_rate=0.01),
    metrics=["acc"],
)




## === cell 14
model.summary()




## === cell 15
history = model.fit(
    train_ds,
    epochs=20,
    validation_data=val_ds,
)




## === cell 16
pass




## === cell 17
pass




## === cell 18
pass




## === cell 19
ss = pd.read_csv(data_path + "sample_submission.csv")

test_tfrec_files = sorted(tf.io.gfile.glob(test_tfrecords_dir + "*.tfrec"))


def _parse_test_tfrec(example_proto):
    feats = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
    }
    ex = tf.io.parse_single_example(example_proto, feats)
    img = _decode_resize_rescale_from_bytes(ex["image"])
    return img, ex["image_name"]


test_raw = tf.data.TFRecordDataset(test_tfrec_files, num_parallel_reads=AUTOTUNE)

test_cache_path = f"/kaggle/working/test_decoded_cache_{run_tag}"
_cleanup_cache_path(test_cache_path)

test_ds = (
    test_raw.map(_parse_test_tfrec, num_parallel_calls=AUTOTUNE)
    .cache(test_cache_path)
    .batch(64, drop_remainder=False)
    .prefetch(AUTOTUNE)
    .with_options(options)
)

test_ids_list = []
preds_list = []

for batch_imgs, batch_ids in test_ds:
    batch_probs = model(batch_imgs, training=False).numpy()
    preds_list.append(batch_probs.argmax(axis=1).astype(np.int64))
    test_ids_list.append(batch_ids.numpy())

preds = np.concatenate(preds_list, axis=0)
test_ids = np.concatenate(test_ids_list, axis=0).astype("U")

pred_df = pd.DataFrame({"image_id": test_ids, "label": preds})
my_submission = ss[["image_id"]].merge(pred_df, on="image_id", how="left")
my_submission["label"] = my_submission["label"].astype(np.int64)

my_submission.to_csv("submission.csv", index=False)




## === cell 20
print("Submission File: \n---------------\n")
print(my_submission.head())  # Predicted Output
print("\nSaved to: /kaggle/working/submission.csv")
