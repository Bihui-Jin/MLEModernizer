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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
import math
import numpy as np
import pandas as pd
import tensorflow as tf

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import seaborn as sns

    sns.set()
except Exception:
    sns = None

from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

SEED = 42
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TF:", tf.__version__)
print("Keras:", keras.__version__)



## === cell 1
path = "../input/cassava-leaf-disease-classification"



## === cell 2
train_images_dir = os.path.join(path, "train_images")
test_images_dir = os.path.join(path, "test_images")



## === cell 3
with open(
    "../input/cassava-leaf-disease-classification/label_num_to_disease_map.json"
) as file:
    classes = json.loads(file.read())

print(json.dumps(classes, indent=4))



## === cell 4
df_train = pd.read_csv(os.path.join(path, "train.csv"))
df_train.head()



## === cell 5
df_train["class"] = df_train["label"].map({int(i): c for i, c in classes.items()})
df_train.head()



## === cell 6
_ = df_train["class"].value_counts()




## === cell 7
def plot(images, labels, predictions=None):
    pass




## === cell 8
pass



## === cell 9
pass



## === cell 10
pass



## === cell 11
pass



## === cell 12
pass



## === cell 13
pass



## === cell 14
pass



## === cell 15
pass



## === cell 16
pass



## === cell 17
pass



## === cell 18
pass



## === cell 19
train = df_train.astype({"label": str})
from sklearn.model_selection import train_test_split

train, valid = train_test_split(
    train, test_size=0.2, random_state=SEED, stratify=train["label"]
)



## === cell 20

img_size = 300
size = (img_size, img_size)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

class_names = [classes[str(i)] for i in range(5)]
class_to_index = {name: i for i, name in enumerate(class_names)}


def _read_decode_resize(image_path):
    img_bytes = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [img_size, img_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    img = tf.cast(img, tf.float32)
    return img


def _preprocess(img):
    return preprocess_input(img)


def _one_hot(label_idx):
    return tf.one_hot(label_idx, depth=5, dtype=tf.float32)


def _augment(img, seed):
    seeds = tf.random.experimental.stateless_split(seed, 7)
    s1 = seeds[0]
    s2 = seeds[1]
    s3 = seeds[2]
    s4 = seeds[3]
    s5 = seeds[4]
    s6 = seeds[5]
    s7 = seeds[6]

    img = tf.image.stateless_random_flip_left_right(img, seed=s1)
    img = tf.image.stateless_random_flip_up_down(img, seed=s2)

    angle = tf.random.stateless_uniform([], seed=s3, minval=-45.0, maxval=45.0) * (
        math.pi / 180.0
    )
    zoom = tf.random.stateless_uniform([], seed=s4, minval=0.8, maxval=1.2)

    tx = tf.random.stateless_uniform([], seed=s5, minval=-0.2, maxval=0.2) * tf.cast(
        img_size, tf.float32
    )
    ty = tf.random.stateless_uniform([], seed=s6, minval=-0.2, maxval=0.2) * tf.cast(
        img_size, tf.float32
    )
    shear = tf.random.stateless_uniform([], seed=s7, minval=-0.2, maxval=0.2)

    cos_a = tf.cos(angle)
    sin_a = tf.sin(angle)

    r00 = cos_a
    r01 = -sin_a
    r10 = sin_a
    r11 = cos_a

    sh = tf.tan(shear)
    s00 = 1.0
    s01 = sh
    s10 = 0.0
    s11 = 1.0

    z00 = 1.0 / zoom
    z11 = 1.0 / zoom

    a00 = z00 * (s00 * r00 + s01 * r10)
    a01 = z00 * (s00 * r01 + s01 * r11)
    b00 = z11 * (s10 * r00 + s11 * r10)
    b01 = z11 * (s10 * r01 + s11 * r11)

    cx = (img_size - 1) / 2.0
    cy = (img_size - 1) / 2.0

    a02 = cx - a00 * cx - a01 * cy - tx
    b02 = cy - b00 * cx - b01 * cy - ty

    transform = tf.stack([a00, a01, a02, b00, b01, b02, 0.0, 0.0])[tf.newaxis, :]

    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[tf.newaxis, ...],
        transforms=transform,
        output_shape=[img_size, img_size],
        interpolation="NEAREST",
        fill_mode="REFLECT",
        fill_value=0.0,
    )[0]
    return img


def make_train_ds(df):
    file_paths = tf.constant(
        [os.path.join(train_images_dir, f) for f in df["image_id"].values]
    )
    label_idx = tf.constant(
        [class_to_index[c] for c in df["class"].values], dtype=tf.int32
    )

    ds = tf.data.Dataset.from_tensor_slices((file_paths, label_idx))
    ds = ds.shuffle(buffer_size=len(df), seed=SEED, reshuffle_each_iteration=True)

    def _map_fn(p, y):
        img = _read_decode_resize(p)
        h = tf.strings.to_hash_bucket_fast(p, 2**31 - 1)
        seed = tf.stack([tf.cast(SEED, tf.int64), tf.cast(h, tf.int64)])
        seed = tf.cast(seed, tf.int32)
        img = _augment(img, seed)
        img = _preprocess(img)
        y = _one_hot(y)
        return img, y

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_valid_ds(df):
    file_paths = tf.constant(
        [os.path.join(train_images_dir, f) for f in df["image_id"].values]
    )
    label_idx = tf.constant(
        [class_to_index[c] for c in df["class"].values], dtype=tf.int32
    )

    ds = tf.data.Dataset.from_tensor_slices((file_paths, label_idx))

    def _map_fn(p, y):
        img = _read_decode_resize(p)
        img = _preprocess(img)
        y = _one_hot(y)
        return img, y

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.cache()
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_train_ds(train)
valid_ds = make_valid_ds(valid)



## === cell 21
step_size_train = max(1, len(train) // BATCH_SIZE)
step_size_valid = max(1, len(valid) // BATCH_SIZE)
step_size_train, step_size_valid




## === cell 22
def modelTransf():
    model = models.Sequential()
    model.add(
        EfficientNetB3(
            input_shape=(img_size, img_size, 3), include_top=False, weights="imagenet"
        )
    )
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(5, activation="softmax"))
    return model




## === cell 23
model = modelTransf()



## === cell 24
model.summary()



## === cell 25
model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=0.001),
    metrics=["accuracy"],
)



## === cell 26
early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, mode="min", restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "modelB3.keras", monitor="val_loss", verbose=1, mode="min", save_best_only=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=10, min_lr=0.001, mode="min", verbose=1
)



## === cell 27
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=30,
    steps_per_epoch=step_size_train,
    validation_steps=step_size_valid,
    callbacks=[early_stopping, checkpoint, reduce_lr],
)



## === cell 28
best_model_path = "modelB3.keras"
if os.path.exists(best_model_path):
    model_trained = keras.models.load_model(best_model_path)
else:
    model_trained = model



## === cell 29
from sklearn.metrics import accuracy_score

val_probs = model_trained.predict(
    valid_ds,
    steps=math.ceil(len(valid) / BATCH_SIZE),
    verbose=0,
)
val_pred = np.argmax(val_probs, axis=1)[: len(valid)]
val_true = valid["class"].map(class_to_index).values
print("Validation accuracy:", accuracy_score(val_true, val_pred))



## === cell 30
submission_file = pd.read_csv(
    "../input/cassava-leaf-disease-classification/sample_submission.csv"
)
submission_file.head()



## === cell 31
test_df = submission_file.copy()

test_paths = tf.constant(
    [os.path.join(test_images_dir, f) for f in test_df["image_id"].values]
)
test_ds = tf.data.Dataset.from_tensor_slices(test_paths)


def _map_test(p):
    img = _read_decode_resize(p)
    img = _preprocess(img)
    return img


test_ds = test_ds.map(_map_test, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False)
test_ds = test_ds.prefetch(AUTOTUNE)



## === cell 32
test_probs = model_trained.predict(
    test_ds,
    steps=math.ceil(len(test_df) / BATCH_SIZE),
    verbose=1,
)
test_pred = np.argmax(test_probs, axis=1)[: len(test_df)]
len(test_pred), len(test_df)



## === cell 33
submission = submission_file.copy()
submission["label"] = test_pred.astype(int)
submission.head()



## === cell 34
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print("Saved at:", os.path.abspath("submission.csv"))
