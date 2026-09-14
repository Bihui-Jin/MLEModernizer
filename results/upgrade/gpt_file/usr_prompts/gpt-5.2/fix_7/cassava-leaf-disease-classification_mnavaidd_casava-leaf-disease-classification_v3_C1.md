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

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D,
)
from tensorflow.keras.models import Model

print("TF version:", tf.__version__)




## === cell 1
data_path = "../input/cassava-leaf-disease-classification/"
train_csv_data_path = data_path + "train.csv"
label_json_data_path = data_path + "label_num_to_disease_map.json"
images_dir_data_path = data_path + "train_images/"

test_images_dir_data_path = data_path + "test_images/"
sample_sub_path = data_path + "sample_submission.csv"




## === cell 2
train_csv = pd.read_csv(train_csv_data_path)
train_csv["label"] = (
    train_csv["label"].astype("int32").astype("string")
)  # keep original idea (string labels for flow_from_dataframe)

label_class = pd.read_json(label_json_data_path, orient="index")
label_class = label_class.values.flatten().tolist()




## === cell 3
print("Label names :")
for i, label in enumerate(label_class):
    print(f" {i}. {label}")




## === cell 4
train_csv.head()




## === cell 5
BATCH_SIZE = 18
IMG_SIZE = 224
NUM_CLASSES = 5
SEED = 42

tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE




## === cell 6
base_model = applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)

model_model = Model(inputs=base_model.input, outputs=outputs)

for layer in base_model.layers:
    layer.trainable = False

model_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model_model.summary()




## === cell 7
ss = pd.read_csv(sample_sub_path)
test_img_path = os.path.join(test_images_dir_data_path, ss.image_id.iloc[0])

img_bgr = cv2.imread(test_img_path)
if img_bgr is None:
    raise FileNotFoundError(f"Could not read test image at: {test_img_path}")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
resized_img = (
    cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE)).reshape(-1, IMG_SIZE, IMG_SIZE, 3) / 255.0
)

plt.figure(figsize=(8, 4))
plt.title(f"TEST IMAGE: {os.path.basename(test_img_path)}")
plt.imshow(resized_img[0])
plt.axis("off")
plt.show()




## === cell 8
VALIDATION_SPLIT = 0.1
rotation_range = 10
width_shift_range = 0.05
height_shift_range = 0.05
zoom_range = 0.1
horizontal_flip = True

n_total = len(train_csv)
n_train = int(np.ceil((1.0 - VALIDATION_SPLIT) * n_total))

train_df = train_csv.iloc[:n_train].copy()
val_df = train_csv.iloc[n_train:].copy()

train_paths = (images_dir_data_path + train_df["image_id"]).to_numpy()
train_labels = train_df["label"].astype("int32").to_numpy()

val_paths = (images_dir_data_path + val_df["image_id"]).to_numpy()
val_labels = val_df["label"].astype("int32").to_numpy()

_rotation_deg = float(rotation_range)
_wsr = float(width_shift_range)
_hsr = float(height_shift_range)
_zr = float(zoom_range)
_do_hflip = bool(horizontal_flip)

_rotation_factor = (
    _rotation_deg / 360.0
)  # RandomRotation expects fraction of a full turn
_rand_rotate = tf.keras.layers.RandomRotation(
    factor=_rotation_factor,
    fill_mode="reflect",
    interpolation="bilinear",
    seed=SEED,
)


@tf.function
def _read_bytes(path, label):
    img_bytes = tf.io.read_file(path)
    return img_bytes, tf.cast(label, tf.int32)


@tf.function
def _decode_resize_rescale(img_bytes, label):
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) / 255.0  # rescale
    return img, label


@tf.function
def _augment(img, label):
    if _rotation_deg > 0:
        img = _rand_rotate(tf.expand_dims(img, 0), training=True)[0]

    if _wsr > 0 or _hsr > 0:
        dx = tf.random.uniform([], -_wsr, _wsr, dtype=tf.float32) * tf.cast(
            IMG_SIZE, tf.float32
        )
        dy = tf.random.uniform([], -_hsr, _hsr, dtype=tf.float32) * tf.cast(
            IMG_SIZE, tf.float32
        )
        img = tf.raw_ops.ImageProjectiveTransformV3(
            images=tf.expand_dims(img, 0),
            transforms=tf.expand_dims(
                tf.stack([1.0, 0.0, -dx, 0.0, 1.0, -dy, 0.0, 0.0]), 0
            ),
            output_shape=tf.constant([IMG_SIZE, IMG_SIZE], dtype=tf.int32),
            interpolation="BILINEAR",
            fill_mode="REFLECT",
            fill_value=0.0,
        )[0]

    if _zr > 0:
        scale = tf.random.uniform([], 1.0 - _zr, 1.0 + _zr, dtype=tf.float32)
        cx = (tf.cast(IMG_SIZE, tf.float32) - 1.0) / 2.0
        cy = (tf.cast(IMG_SIZE, tf.float32) - 1.0) / 2.0
        a0 = 1.0 / scale
        a4 = 1.0 / scale
        a2 = cx - a0 * cx
        a5 = cy - a4 * cy
        img = tf.raw_ops.ImageProjectiveTransformV3(
            images=tf.expand_dims(img, 0),
            transforms=tf.expand_dims(
                tf.stack([a0, 0.0, a2, 0.0, a4, a5, 0.0, 0.0]), 0
            ),
            output_shape=tf.constant([IMG_SIZE, IMG_SIZE], dtype=tf.int32),
            interpolation="BILINEAR",
            fill_mode="REFLECT",
            fill_value=0.0,
        )[0]

    if _do_hflip:
        img = tf.image.random_flip_left_right(img, seed=SEED)

    return img, label


_ds_options = tf.data.Options()
_ds_options.experimental_deterministic = True
_ds_options.threading.private_threadpool_size = 16
_ds_options.threading.max_intra_op_parallelism = 0


def make_ds(paths, labels, training: bool):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.with_options(_ds_options)
    ds = ds.map(_read_bytes, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()  # caches bytes tuples, not decoded images
    ds = ds.map(_decode_resize_rescale, num_parallel_calls=AUTOTUNE)

    if training:
        ds = ds.shuffle(
            buffer_size=min(len(paths), 4096), seed=SEED, reshuffle_each_iteration=True
        )
        ds = ds.map(_augment, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_gen = make_ds(train_paths, train_labels, training=True)
val_gen = make_ds(val_paths, val_labels, training=False)

EPOCHS = 3
history = model_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    verbose=1,
)




## === cell 9
preds = []
ss = pd.read_csv(sample_sub_path)

test_paths = (test_images_dir_data_path + ss["image_id"]).to_numpy()


@tf.function
def _read_test_bytes(path):
    return tf.io.read_file(path)


@tf.function
def _decode_test(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


test_ds = tf.data.Dataset.from_tensor_slices(test_paths).with_options(_ds_options)
test_ds = test_ds.map(_read_test_bytes, num_parallel_calls=AUTOTUNE).cache()
test_ds = (
    test_ds.map(_decode_test, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

proba = model_model.predict(test_ds, verbose=0)
preds = np.argmax(proba, axis=1).astype(int).tolist()

my_submission = pd.DataFrame({"image_id": ss.image_id, "label": preds})
my_submission.to_csv("submission.csv", index=False)

print("Submission File: \n---------------\n")
print(my_submission.head())
print("\nWrote: submission.csv")
print("Rows:", len(my_submission), "Cols:", list(my_submission.columns))
