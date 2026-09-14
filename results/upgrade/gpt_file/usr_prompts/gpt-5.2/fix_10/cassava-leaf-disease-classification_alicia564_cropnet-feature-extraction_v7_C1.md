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

3.13

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
import re
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "").lower() == "python":
    os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import tensorflow as tf

print("TensorFlow:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))

SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")




## === cell 1
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder

label_to_disease = pd.read_json(
    "/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json",
    typ="series",
)
train_csv = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/train.csv")

train_csv["disease"] = train_csv["label"].map(label_to_disease)
train_csv["path"] = (
    "/kaggle/input/cassava-leaf-disease-classification/train_images/"
    + train_csv["image_id"]
)

train_csv["label_encoded"] = LabelEncoder().fit_transform(train_csv["disease"])

train_csv["disease"] = train_csv["disease"].astype(str)
train_csv["label"] = train_csv["label"].astype(str)

train, valid = train_test_split(
    train_csv, test_size=0.2, stratify=train_csv["label"], random_state=SEED
)

classes_sorted = sorted(train_csv["disease"].unique().tolist())
class_indices = {c: i for i, c in enumerate(classes_sorted)}

disease_to_label_id = {str(v): int(k) for k, v in label_to_disease.items()}
idx_to_disease = {idx: disease for disease, idx in class_indices.items()}
idx_to_label_id = {
    idx: disease_to_label_id[disease] for idx, disease in idx_to_disease.items()
}

print("class_indices (disease->idx):", class_indices)
print("idx_to_label_id:", idx_to_label_id)




## === cell 2
num_classes = len(class_indices)

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
outputs = Dense(num_classes, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()




## === cell 3
IMG_SIZE = 224
BATCH_SIZE = 32

_rng = tf.random.Generator.from_seed(SEED)

IMG_SIZE_F = tf.constant(float(IMG_SIZE), tf.float32)
PI_F = tf.constant(np.pi, tf.float32)


@tf.function
def _decode_resize(path):
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=3, dct_method="INTEGER_FAST")
    image = tf.image.resize(
        image, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    image = tf.cast(image, tf.float32)
    return image


@tf.function
def _random_affine_like_imagedatagen(image):
    angle = _rng.uniform([], minval=-45.0, maxval=45.0, dtype=tf.float32) * (
        PI_F / 180.0
    )
    tx = _rng.uniform([], minval=-0.2, maxval=0.2, dtype=tf.float32) * IMG_SIZE_F
    ty = _rng.uniform([], minval=-0.2, maxval=0.2, dtype=tf.float32) * IMG_SIZE_F
    shear = _rng.uniform([], minval=-0.2, maxval=0.2, dtype=tf.float32)
    zoom = _rng.uniform([], minval=0.8, maxval=1.2, dtype=tf.float32)

    cos_a = tf.cos(angle)
    sin_a = tf.sin(angle)

    a0 = cos_a / zoom
    a1 = -sin_a / zoom
    b0 = sin_a / zoom
    b1 = cos_a / zoom

    sh = tf.tan(shear)
    a1 = a1 + sh

    cx = (IMG_SIZE_F - 1.0) / 2.0
    cy = (IMG_SIZE_F - 1.0) / 2.0

    a2 = cx - a0 * cx - a1 * cy - tx
    b2 = cy - b0 * cx - b1 * cy - ty

    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])[tf.newaxis, :]

    image = tf.raw_ops.ImageProjectiveTransformV3(
        images=image[tf.newaxis, ...],
        transforms=transform,
        output_shape=[IMG_SIZE, IMG_SIZE],
        interpolation="BILINEAR",
        fill_mode="NEAREST",
        fill_value=0.0,
    )[0]
    return image


@tf.function
def _augment(image):
    image = _random_affine_like_imagedatagen(image)
    image = tf.cond(
        _rng.uniform([], dtype=tf.float32) < 0.5,
        lambda: tf.image.flip_left_right(image),
        lambda: image,
    )
    image = tf.cond(
        _rng.uniform([], dtype=tf.float32) < 0.5,
        lambda: tf.image.flip_up_down(image),
        lambda: image,
    )
    return image


@tf.function
def _to_onehot(label_idx):
    return tf.one_hot(tf.cast(label_idx, tf.int32), depth=num_classes, dtype=tf.float32)


@tf.function
def _train_map_from_path(path, label_idx):
    img = _decode_resize(path)
    img = _augment(img)
    img = preprocess_input(img)
    return img, _to_onehot(label_idx)


@tf.function
def _valid_map_from_path(path, label_idx):
    img = _decode_resize(path)
    img = preprocess_input(img)
    return img, _to_onehot(label_idx)


train_paths = train["path"].values.astype(str)
train_labels = train["label_encoded"].values.astype(np.int32)
valid_paths = valid["path"].values.astype(str)
valid_labels = valid["label_encoded"].values.astype(np.int32)

options = tf.data.Options()
options.experimental_deterministic = True
options.experimental_optimization.apply_default_optimizations = True
options.experimental_optimization.map_parallelization = True
options.experimental_optimization.parallel_batch = True
options.experimental_optimization.map_and_batch_fusion = True

cpu_cnt = os.cpu_count() or 8
options.threading.private_threadpool_size = min(32, max(8, cpu_cnt))
options.threading.max_intra_op_parallelism = 1

try:
    options.autotune.enabled = True
    options.autotune.cpu_budget = cpu_cnt
except Exception:
    pass

SHUFFLE_BUFFER = int(min(len(train_paths), 8192))

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels)).with_options(
    options
)
train_ds = train_ds.shuffle(
    buffer_size=SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=True
)

train_ds_decoded = train_ds.map(
    lambda p, y: (_decode_resize(p), y),
    num_parallel_calls=tf.data.AUTOTUNE,
    deterministic=True,
).cache()

train_ds = train_ds_decoded.map(
    lambda img, y: (preprocess_input(_augment(img)), _to_onehot(y)),
    num_parallel_calls=tf.data.AUTOTUNE,
    deterministic=True,
)
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

valid_ds = tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels)).with_options(
    options
)
valid_ds = valid_ds.map(
    _valid_map_from_path,
    num_parallel_calls=tf.data.AUTOTUNE,
    deterministic=True,
).cache()
valid_ds = valid_ds.batch(BATCH_SIZE, drop_remainder=False)
valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)




## === cell 4
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback


class EarlyStoppingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if self.model.stop_training:
            print(f"Early stopping triggered at epoch {epoch + 1}.")


early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
)

learning_rate_reduction = ReduceLROnPlateau(
    monitor="val_loss",
    patience=2,
    factor=0.5,
    min_lr=1e-6,
    verbose=1,
)

EPOCHS = 10

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping, learning_rate_reduction, EarlyStoppingCallback()],
    verbose=1,
)




## === cell 5
sample_sub = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"

test_paths = (image_dir + "/" + sample_sub["image_id"].values).astype(str)


@tf.function
def _test_map_fn(path):
    img = _decode_resize(path)
    img = preprocess_input(img)
    return img


test_ds = tf.data.Dataset.from_tensor_slices(test_paths).with_options(options)
test_ds = test_ds.map(
    _test_map_fn,
    num_parallel_calls=tf.data.AUTOTUNE,
    deterministic=True,
)
test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

probs = model.predict(test_ds, verbose=0)
pred_indices = np.argmax(probs, axis=1).astype(int)

map_arr = np.empty((num_classes,), dtype=np.int64)
for k, v in idx_to_label_id.items():
    map_arr[int(k)] = int(v)

pred_labels = map_arr[pred_indices].astype(int).tolist()

submission_df = pd.DataFrame(
    {"image_id": sample_sub["image_id"].values, "label": pred_labels}
)
submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)

print("Submission file created:", submission_path)
print(submission_df.head())
print("Submission shape:", submission_df.shape)
