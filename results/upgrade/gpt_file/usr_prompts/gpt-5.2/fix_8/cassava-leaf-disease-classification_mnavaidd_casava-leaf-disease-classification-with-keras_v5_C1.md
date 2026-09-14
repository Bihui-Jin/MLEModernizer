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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_PYTHON"] = "python"
os.environ.setdefault("PYTHONHASHSEED", "42")

import random
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



## === cell 1
data_path = "../input/cassava-leaf-disease-classification/"
train_csv_data_path = data_path + "train.csv"
label_json_data_path = data_path + "label_num_to_disease_map.json"
images_dir_data_path = data_path + "train_images"



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


def _seed_from_path(path):
    h = tf.strings.to_hash_bucket_fast(path, 2**31 - 1)
    return tf.stack([tf.cast(h, tf.int32), tf.constant(42, tf.int32)], axis=0)


@tf.function
def _decode_resize_rescale(path):
    img_bytes = tf.io.read_file(path)
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
def _rotate_projective(img, angle_rad):
    angle_rad = tf.cast(angle_rad, tf.float32)
    cos_a = tf.cos(angle_rad)
    sin_a = tf.sin(angle_rad)

    cx = (IMG_SIZE - 1) / 2.0
    cy = (IMG_SIZE - 1) / 2.0

    a0 = cos_a
    a1 = -sin_a
    b0 = sin_a
    b1 = cos_a

    a2 = cx - a0 * cx - a1 * cy
    b2 = cy - b0 * cx - b1 * cy

    det = a0 * b1 - a1 * b0
    ia0 = b1 / det
    ia1 = -a1 / det
    ib0 = -b0 / det
    ib1 = a0 / det
    ia2 = -(ia0 * a2 + ia1 * b2)
    ib2 = -(ib0 * a2 + ib1 * b2)

    transform = tf.stack([ia0, ia1, ia2, ib0, ib1, ib2, 0.0, 0.0])[None, :]
    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[None, ...],
        transforms=transform,
        output_shape=tf.constant([IMG_SIZE, IMG_SIZE], tf.int32),
        fill_value=0.0,
        interpolation="BILINEAR",
        fill_mode="REFLECT",
    )[0]
    return out


@tf.function
def _augment(img, seed):
    angle = tf.random.stateless_uniform(
        [], seed=seed, minval=-np.pi, maxval=np.pi, dtype=tf.float32
    )
    img = _rotate_projective(img, angle)

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

    cx = (IMG_SIZE - 1) / 2.0
    cy = (IMG_SIZE - 1) / 2.0

    sh = tf.tan(shear)

    a0 = zoom
    a1 = zoom * sh
    b0 = 0.0
    b1 = zoom

    a2 = cx - a0 * cx - a1 * cy
    b2 = cy - b0 * cx - b1 * cy

    det = a0 * b1 - a1 * b0
    ia0 = b1 / det
    ia1 = -a1 / det
    ib0 = -b0 / det
    ib1 = a0 / det
    ia2 = -(ia0 * a2 + ia1 * b2)
    ib2 = -(ib0 * a2 + ib1 * b2)

    transform = tf.stack([ia0, ia1, ia2, ib0, ib1, ib2, 0.0, 0.0])[None, :]
    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[None, ...],
        transforms=transform,
        output_shape=tf.constant([IMG_SIZE, IMG_SIZE], tf.int32),
        fill_value=0.0,
        interpolation="BILINEAR",
        fill_mode="REFLECT",
    )[0]

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
def _train_from_cached(img, path, y):
    seed = _seed_from_path(path)
    img = _augment(img, seed)
    y = tf.one_hot(tf.cast(y, tf.int32), NUM_CLASSES)
    return img, y


@tf.function
def _val_from_cached(img, y):
    y = tf.one_hot(tf.cast(y, tf.int32), NUM_CLASSES)
    return img, y


options = tf.data.Options()
options.experimental_deterministic = True

train_img_ds = (
    tf.data.Dataset.from_tensor_slices(train_paths)
    .map(_decode_resize_rescale, num_parallel_calls=AUTOTUNE)
    .cache()
)

val_img_ds = (
    tf.data.Dataset.from_tensor_slices(val_paths)
    .map(_decode_resize_rescale, num_parallel_calls=AUTOTUNE)
    .cache()
)

train_lbl_ds = tf.data.Dataset.from_tensor_slices(train_labels_int)
val_lbl_ds = tf.data.Dataset.from_tensor_slices(val_labels_int)

train_ds = tf.data.Dataset.zip(
    (train_img_ds, tf.data.Dataset.from_tensor_slices(train_paths), train_lbl_ds)
)
shuffle_buf = int(min(len(train_paths), 4096))
train_ds = train_ds.shuffle(
    buffer_size=shuffle_buf, seed=42, reshuffle_each_iteration=True
)
train_ds = train_ds.map(_train_from_cached, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
train_ds = train_ds.with_options(options)

val_ds = tf.data.Dataset.zip((val_img_ds, val_lbl_ds))
val_ds = val_ds.map(_val_from_cached, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
val_ds = val_ds.with_options(options)



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
test_paths = (data_path + "test_images/" + ss["image_id"]).to_numpy()


ds = tf.data.Dataset.from_tensor_slices(test_paths)
ds = ds.map(_decode_resize_rescale, num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.cache()
ds = ds.batch(64, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

probs = model.predict(ds, verbose=0)
preds = probs.argmax(axis=1).astype(np.int64)

my_submission = pd.DataFrame({"image_id": ss.image_id, "label": preds})
my_submission.to_csv("submission.csv", index=False)



## === cell 20
print("Submission File: \n---------------\n")
print(my_submission.head())  # Predicted Output
print("\nSaved to: /kaggle/working/submission.csv")
