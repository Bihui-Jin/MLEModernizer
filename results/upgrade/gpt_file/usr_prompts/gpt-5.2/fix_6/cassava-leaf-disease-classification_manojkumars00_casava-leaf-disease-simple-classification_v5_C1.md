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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D,
)

tf.keras.utils.set_random_seed(42)
np.random.seed(42)

AUTOTUNE = tf.data.AUTOTUNE
try:
    tf.config.optimizer.set_jit(
        False
    )  # preserve numerics; avoid potential compilation overhead
except Exception:
    pass



## === cell 1
train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
label_json_path = (
    "../input/cassava-leaf-disease-classification/label_num_to_disease_map.json"
)
images_dir_path = "../input/cassava-leaf-disease-classification/train_images"



## === cell 2
train_csv = pd.read_csv(train_csv_path)
train_csv["label"] = train_csv["label"].astype("string")

label_class = pd.read_json(label_json_path, orient="index")
label_class = label_class.values.flatten().tolist()



## === cell 3
print("Label names :")
for i, label in enumerate(label_class):
    print(f" {i}. {label}")



## === cell 4
train_csv.head()



## === cell 5
BATCH_SIZE = 24
IMG_SIZE = 320



## === cell 6
train_gen = ImageDataGenerator(
    rotation_range=270,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.1, 0.9],
    shear_range=25,
    zoom_range=0.3,
    channel_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1 / 255,
    validation_split=0.2,
)

valid_gen = ImageDataGenerator(rescale=1 / 255, validation_split=0.2)



## === cell 7
df = train_csv.copy()
n = len(df)
val_size = int(np.floor(0.2 * n))
perm = np.random.RandomState(42).permutation(n)
val_idx = perm[:val_size]
train_idx = perm[val_size:]

train_df = df.iloc[train_idx].reset_index(drop=True)
valid_df = df.iloc[val_idx].reset_index(drop=True)

class_names = sorted(df["label"].unique().tolist())
class_to_idx = {c: i for i, c in enumerate(class_names)}

train_paths = (images_dir_path.rstrip("/") + "/" + train_df["image_id"]).tolist()
valid_paths = (images_dir_path.rstrip("/") + "/" + valid_df["image_id"]).tolist()
train_labels = train_df["label"].map(class_to_idx).astype(np.int32).to_numpy()
valid_labels = valid_df["label"].map(class_to_idx).astype(np.int32).to_numpy()


def _read_decode_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32)
    return img


def _apply_projective(img, transform):
    transform = tf.reshape(transform, [1, 8])
    return tf.raw_ops.ImageProjectiveTransformV3(
        images=img[None, ...],
        transforms=transform,
        output_shape=[IMG_SIZE, IMG_SIZE],
        interpolation="BILINEAR",
        fill_mode="NEAREST",
        fill_value=0.0,
    )[0]


@tf.function
def _augment_train(img):
    img = img / 255.0

    angle = tf.random.uniform([], minval=-270.0, maxval=270.0, dtype=tf.float32) * (
        tf.constant(np.pi / 180.0, tf.float32)
    )
    c = tf.cos(angle)
    s = tf.sin(angle)
    cx = (tf.cast(IMG_SIZE, tf.float32) - 1.0) / 2.0
    cy = (tf.cast(IMG_SIZE, tf.float32) - 1.0) / 2.0
    rot = tf.stack(
        [c, -s, (1.0 - c) * cx + s * cy, s, c, (1.0 - c) * cy - s * cx, 0.0, 0.0],
        axis=0,
    )
    img = _apply_projective(img, rot)

    dx = tf.random.uniform([], -0.2, 0.2, dtype=tf.float32) * tf.cast(
        IMG_SIZE, tf.float32
    )
    dy = tf.random.uniform([], -0.2, 0.2, dtype=tf.float32) * tf.cast(
        IMG_SIZE, tf.float32
    )
    trans = tf.stack([1.0, 0.0, -dx, 0.0, 1.0, -dy, 0.0, 0.0], axis=0)
    img = _apply_projective(img, trans)

    b = tf.random.uniform([], 0.1, 0.9, dtype=tf.float32)
    img = tf.clip_by_value(img * b, 0.0, 1.0)

    shear = tf.random.uniform([], -25.0, 25.0, dtype=tf.float32) * (
        tf.constant(np.pi / 180.0, tf.float32)
    )
    sh = tf.tan(shear)
    shear_transform = tf.stack([1.0, sh, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], axis=0)
    img = _apply_projective(img, shear_transform)

    z = tf.random.uniform([], 0.7, 1.3, dtype=tf.float32)
    cx = (tf.cast(IMG_SIZE, tf.float32) - 1.0) / 2.0
    cy = (tf.cast(IMG_SIZE, tf.float32) - 1.0) / 2.0
    zoom_transform = tf.stack(
        [z, 0.0, (1.0 - z) * cx, 0.0, z, (1.0 - z) * cy, 0.0, 0.0], axis=0
    )
    img = _apply_projective(img, zoom_transform)

    cshift = tf.random.uniform([], -0.1, 0.1, dtype=tf.float32)
    img = tf.clip_by_value(img + cshift, 0.0, 1.0)

    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)

    return img


@tf.function
def _preprocess_valid(img):
    return img / 255.0


def _make_ds(paths, labels=None, training=False):
    paths = tf.constant(paths)
    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(paths)
        ds = ds.map(
            lambda p: _preprocess_valid(_read_decode_resize(p)),
            num_parallel_calls=AUTOTUNE,
        )
        ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
        return ds

    labels = tf.constant(labels)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        buffer_size = tf.minimum(
            tf.cast(tf.size(paths), tf.int64),
            tf.constant(4096, dtype=tf.int64),
        )
        ds = ds.shuffle(buffer_size=buffer_size, seed=42, reshuffle_each_iteration=True)

    def _map_fn(p, y):
        img = _read_decode_resize(p)
        img = _augment_train(img) if training else _preprocess_valid(img)
        y = tf.one_hot(y, depth=5, dtype=tf.float32)  # categorical mode
        return img, y

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


train_generator = _make_ds(train_paths, train_labels, training=True)
valid_generator = _make_ds(valid_paths, valid_labels, training=False)



## === cell 8
if False:
    batch = next(iter(train_generator))
    images = batch[0].numpy()
    labels = batch[1].numpy()

    plt.figure(figsize=(15, 9))
    for i, (img, label) in enumerate(zip(images, labels)):
        plt.subplot(5, 3, i % 15 + 1)
        plt.axis("off")
        plt.imshow(img)
        plt.title(label_class[np.argmax(label)])
        if i == 15:
            break
    plt.tight_layout()
    plt.show()



## === cell 9
base = applications.InceptionResNetV2(
    include_top=False, weights="imagenet", input_shape=[IMG_SIZE, IMG_SIZE, 3]
)



## === cell 10
model = tf.keras.Sequential()
model.add(base)
model.add(BatchNormalization(axis=-1))
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.5))
model.add(Dense(256, activation="relu"))
model.add(Dense(5, activation="softmax"))

model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    metrics=["acc"],
    run_eagerly=False,
)




## === cell 11
def scheduler(epoch, lr):
    if epoch > 2:
        return lr / 1.25
    else:
        return lr


callback = tf.keras.callbacks.LearningRateScheduler(scheduler)



## === cell 12
model_path = "./CasavaLeafDiseaseModel.h5"



## === cell 13
loaded = False
if os.path.exists(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        loaded = True
        print(f"Loaded existing model from {model_path}")
    except Exception as e:
        print(f"Could not load model from {model_path}: {e}")

if not loaded:
    EPOCHS = 6  # unchanged
    history = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=EPOCHS,
        callbacks=[callback],
        verbose=1,
    )
    model.save(model_path)
    print(f"Saved trained model to {model_path}")



## === cell 14
ss_preview = pd.read_csv(
    "../input/cassava-leaf-disease-classification/sample_submission.csv"
)
test_img_path = (
    "../input/cassava-leaf-disease-classification/test_images/"
    + ss_preview.image_id.iloc[0]
)

img = cv2.imread(test_img_path)
if img is None:
    raise FileNotFoundError(f"Could not read test image at path: {test_img_path}")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
resized_img = (
    cv2.resize(img, (IMG_SIZE, IMG_SIZE)).reshape(-1, IMG_SIZE, IMG_SIZE, 3) / 255.0
)

if False:
    plt.figure(figsize=(8, 4))
    plt.title("TEST IMAGE")
    plt.axis("off")
    plt.imshow(resized_img[0])
    plt.show()



## === cell 15
ss = pd.read_csv("../input/cassava-leaf-disease-classification/sample_submission.csv")
test_dir = "../input/cassava-leaf-disease-classification/test_images/"

test_paths = (test_dir.rstrip("/") + "/" + ss["image_id"]).tolist()
test_ds = _make_ds(test_paths, labels=None, training=False)

proba = model.predict(test_ds, verbose=0)
preds = np.argmax(proba, axis=1).astype(int).tolist()

ss["label"] = preds
ss.to_csv("./submission.csv", index=False)



## === cell 16
print("Submission File: \n---------------\n")
print(ss.head())
print("\nSaved to: ./submission.csv")
