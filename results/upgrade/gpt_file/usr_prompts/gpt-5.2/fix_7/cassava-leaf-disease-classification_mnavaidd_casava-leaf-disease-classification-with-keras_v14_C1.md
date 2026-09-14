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
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import cv2

import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D,
)
from tensorflow.keras import layers

tf.config.run_functions_eagerly(False)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass




## === cell 1
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)




## === cell 2
data_path = "../input/cassava-leaf-disease-classification/"
train_csv_data_path = data_path + "train.csv"
label_json_data_path = data_path + "label_num_to_disease_map.json"
images_dir_data_path = data_path + "train_images"

test_images_dir_data_path = data_path + "test_images"
sample_sub_path = data_path + "sample_submission.csv"




## === cell 3
train_csv = pd.read_csv(train_csv_data_path)

train_csv["label"] = train_csv["label"].astype("string")

label_class = pd.read_json(label_json_data_path, orient="index")
label_class = label_class.values.flatten().tolist()




## === cell 4
assert os.path.exists(train_csv_data_path), f"Missing: {train_csv_data_path}"
assert os.path.isdir(images_dir_data_path), f"Missing dir: {images_dir_data_path}"
assert os.path.isdir(
    test_images_dir_data_path
), f"Missing dir: {test_images_dir_data_path}"
assert os.path.exists(sample_sub_path), f"Missing: {sample_sub_path}"




## === cell 5
print("Label names :")
for i, label in enumerate(label_class):
    print(f" {i}. {label}")




## === cell 6
train_csv.head()




## === cell 7
pass




## === cell 8
BATCH_SIZE = 18
IMG_SIZE = 224




## === cell 9
VAL_SPLIT = 0.15

n_total = len(train_csv)
n_val = int(np.floor(n_total * VAL_SPLIT))
n_train = n_total - n_val

train_df = train_csv.iloc[:n_train].reset_index(drop=True)
valid_df = train_csv.iloc[n_train:].reset_index(drop=True)

class_names = sorted(train_csv["label"].unique().tolist())
class_to_index = {name: i for i, name in enumerate(class_names)}
NUM_CLASSES = len(class_names)

train_paths = (images_dir_data_path + "/" + train_df["image_id"].values).astype(str)
valid_paths = (images_dir_data_path + "/" + valid_df["image_id"].values).astype(str)

train_labels = train_df["label"].map(class_to_index).astype(np.int32).values
valid_labels = valid_df["label"].map(class_to_index).astype(np.int32).values

AUTOTUNE = tf.data.AUTOTUNE

ds_options = tf.data.Options()
ds_options.experimental_deterministic = True
ds_options.autotune.enabled = True

try:
    ds_options.experimental_optimization.map_parallelization = True
    ds_options.experimental_optimization.map_fusion = True
except Exception:
    pass
try:
    ds_options.experimental_prefetch_to_device = "/GPU:0"
except Exception:
    pass


@tf.function
def _decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) / 255.0  # identical to rescale=1/255
    return img


@tf.function
def _augment(img, path):
    h = tf.strings.to_hash_bucket_fast(path, 2**31 - 1)
    seed0 = tf.stack([tf.cast(SEED, tf.int32), tf.cast(h, tf.int32)], axis=0)
    seed1 = tf.stack(
        [tf.cast(SEED + 1, tf.int32), tf.cast(h ^ 0x9E3779B9, tf.int32)], axis=0
    )
    seed2 = tf.stack(
        [tf.cast(SEED + 2, tf.int32), tf.cast(h ^ 0x7F4A7C15, tf.int32)], axis=0
    )

    img = tf.image.stateless_random_flip_left_right(img, seed=seed0)
    img = tf.image.stateless_random_flip_up_down(img, seed=seed1)
    scale = tf.random.stateless_uniform([], seed=seed2, minval=0.1, maxval=0.9)
    img = tf.clip_by_value(img * scale, 0.0, 1.0)
    return img


data_augmentation = tf.keras.Sequential(
    [
        layers.RandomRotation(factor=1.0, fill_mode="nearest", seed=SEED),
        layers.RandomTranslation(
            height_factor=0.1, width_factor=0.1, fill_mode="nearest", seed=SEED
        ),
        layers.RandomZoom(
            height_factor=(-0.3, 0.3),
            width_factor=(-0.3, 0.3),
            fill_mode="nearest",
            seed=SEED,
        ),
    ],
    name="data_augmentation",
)


@tf.function
def _train_map(path, label):
    img = _decode_resize(path)
    img = data_augmentation(img, training=True)
    img = _augment(img, path)
    y = tf.one_hot(label, depth=NUM_CLASSES, dtype=tf.float32)
    return img, y


@tf.function
def _valid_map(path, label):
    img = _decode_resize(path)
    y = tf.one_hot(label, depth=NUM_CLASSES, dtype=tf.float32)
    return img, y


train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels)).with_options(
    ds_options
)
train_ds = train_ds.shuffle(
    buffer_size=min(len(train_df), 8192), seed=SEED, reshuffle_each_iteration=True
)
train_ds = train_ds.map(_train_map, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False)
train_ds = train_ds.prefetch(AUTOTUNE)

valid_ds = tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels)).with_options(
    ds_options
)
valid_ds = valid_ds.map(_valid_map, num_parallel_calls=AUTOTUNE)
valid_ds = valid_ds.cache()  # safe: deterministic, no augmentation
valid_ds = valid_ds.batch(BATCH_SIZE, drop_remainder=False)
valid_ds = valid_ds.prefetch(AUTOTUNE)

class_indices = class_to_index




## === cell 10
pass




## === cell 11
pass




## === cell 12
base_model = applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)




## === cell 13
model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adamax(learning_rate=0.01),
    metrics=["acc"],
)

model.summary()




## === cell 14
model_save = tf.keras.callbacks.ModelCheckpoint(
    filepath="best_model.weights.h5",
    save_best_only=True,
    save_weights_only=True,
    monitor="val_loss",
    mode="min",
    verbose=1,
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.001,
    patience=5,
    mode="min",
    verbose=1,
    restore_best_weights=True,
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=2,
    min_delta=0.001,
    mode="min",
    verbose=1,
)




## === cell 15
EPOCHS = 3
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    callbacks=[model_save, early_stop, reduce_lr],
    verbose=1,
)




## === cell 16
if os.path.exists("best_model.weights.h5"):
    model.load_weights("best_model.weights.h5")




## === cell 17
pass




## === cell 18
pass




## === cell 19
ss = pd.read_csv(sample_sub_path)
test_paths = (test_images_dir_data_path + "/" + ss["image_id"].values).astype(str)


@tf.function
def _test_map(path):
    img = _decode_resize(path)
    return img


test_ds = tf.data.Dataset.from_tensor_slices(test_paths).with_options(ds_options)
test_ds = test_ds.map(_test_map, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.cache()  # safe: pure decode/resize only
test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False)
test_ds = test_ds.prefetch(AUTOTUNE)

test_probs = model.predict(test_ds, verbose=1)
preds = np.argmax(test_probs, axis=1).astype(int)

my_submission = pd.DataFrame({"image_id": ss["image_id"].values, "label": preds})
my_submission.to_csv("submission.csv", index=False)

print("Submission File: \n---------------\n")
print(my_submission.head())
print("\nWrote: submission.csv")
