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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras.layers import Dense, Dropout

print("TF version:", tf.__version__)



## === cell 1
train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
label_json_path = (
    "../input/cassava-leaf-disease-classification/label_num_to_disease_map.json"
)
images_dir_path = "../input/cassava-leaf-disease-classification/train_images"
test_images_dir_path = "../input/cassava-leaf-disease-classification/test_images"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"

train_tfrecords_dir = "../input/cassava-leaf-disease-classification/train_tfrecords"
test_tfrecords_dir = "../input/cassava-leaf-disease-classification/test_tfrecords"



## === cell 2
train_csv = pd.read_csv(train_csv_path)

train_csv["label"] = train_csv["label"].astype(np.int32)

label_class = pd.read_json(label_json_path, orient="index")
label_class = label_class.values.flatten().tolist()

print(train_csv.head())
print("Num classes from json:", len(label_class), label_class)



## === cell 3
IMG_SIZE = 288
BATCH_SIZE = 12
EPOCHS = 32
lr = 1e-5
SEED = 42

tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE



## === cell 4
n = len(train_csv)
idx = np.arange(n)
rng_np = np.random.RandomState(SEED)
rng_np.shuffle(idx)

split = int(0.8 * n)
train_idx = idx[:split]
valid_idx = idx[split:]

train_df = train_csv.iloc[train_idx].reset_index(drop=True)
valid_df = train_csv.iloc[valid_idx].reset_index(drop=True)

print("Train/Valid sizes:", len(train_df), len(valid_df))


def _paths_and_labels(df):
    paths = df["image_id"].apply(lambda x: os.path.join(images_dir_path, x)).values
    labels = df["label"].astype(np.int32).values
    return paths, labels


train_paths, train_labels_int = _paths_and_labels(train_df)
valid_paths, valid_labels_int = _paths_and_labels(valid_df)


def _one_hot(y):
    return tf.one_hot(tf.cast(y, tf.int32), depth=5)


train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels_int))
train_ds = train_ds.shuffle(
    buffer_size=len(train_df), seed=SEED, reshuffle_each_iteration=True
)

valid_ds = tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels_int))
print("Class indices:", {str(i): i for i in range(5)})




## === cell 5
@tf.function
def F1_score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_pred_labels = tf.argmax(y_pred, axis=-1)
    y_pred_oh = tf.one_hot(
        tf.cast(y_pred_labels, tf.int32), depth=tf.shape(y_pred)[-1], dtype=tf.float32
    )

    tp = tf.reduce_sum(y_true * y_pred_oh)
    fp = tf.reduce_sum((1.0 - y_true) * y_pred_oh)
    fn = tf.reduce_sum(y_true * (1.0 - y_pred_oh))

    precision = tp / (tp + fp + 1e-23)
    recall = tp / (tp + fn + 1e-23)
    return 2.0 * (precision * recall) / (precision + recall + 1e-23)




## === cell 6
augment = tf.keras.Sequential(
    [
        tf.keras.layers.Rescaling(1.0 / 255.0),
        tf.keras.layers.RandomRotation(factor=270.0 / 360.0, seed=SEED),
        tf.keras.layers.RandomTranslation(
            height_factor=0.2, width_factor=0.2, seed=SEED
        ),
        tf.keras.layers.RandomBrightness(
            factor=0.4, seed=SEED
        ),  # approximates [0.1, 0.9] range
        tf.keras.layers.RandomShear(
            x_factor=25.0 * np.pi / 180.0, y_factor=0.0, seed=SEED
        ),
        tf.keras.layers.RandomZoom(
            height_factor=(-0.3, 0.3), width_factor=(-0.3, 0.3), seed=SEED
        ),
        tf.keras.layers.RandomFlip(mode="horizontal_and_vertical", seed=SEED),
    ],
    name="augmentation",
)


@tf.function
def channel_shift_batch(x, rng):
    batch = tf.shape(x)[0]
    shifts = rng.uniform(shape=(batch, 1, 1, 3), minval=-0.1, maxval=0.1, dtype=x.dtype)
    x = x + shifts
    return tf.clip_by_value(x, 0.0, 1.0)


rng = tf.random.Generator.from_seed(SEED)


@tf.function
def _decode_resize_jpeg(bytestr):
    img = tf.image.decode_jpeg(bytestr, channels=3)
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    return img


@tf.function
def _decode_resize_from_path(path):
    img_bytes = tf.io.read_file(path)
    return _decode_resize_jpeg(img_bytes)


ds_options = tf.data.Options()
try:
    ds_options.experimental_optimization.map_parallelization = True
except Exception:
    pass
try:
    ds_options.experimental_optimization.parallel_batch = True
except Exception:
    pass
try:
    ds_options.experimental_optimization.autotune_buffers = True
except Exception:
    pass
try:
    ds_options.deterministic = True
except Exception:
    pass


@tf.function
def train_map_from_path(path, label_int):
    img = _decode_resize_from_path(path)
    img = tf.cast(img, tf.float32)  # augment includes rescaling
    labels = _one_hot(label_int)
    return img, labels


@tf.function
def valid_map_from_path(path, label_int):
    img = _decode_resize_from_path(path)
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    labels = _one_hot(label_int)
    return img, labels


@tf.function
def train_batch_map(imgs, labels):
    imgs = augment(imgs, training=True)
    imgs = channel_shift_batch(imgs, rng)
    return imgs, labels


train_ds_opt = (
    train_ds.map(train_map_from_path, num_parallel_calls=AUTOTUNE)
    .cache()
    .batch(BATCH_SIZE, drop_remainder=False)
    .map(train_batch_map, num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
).with_options(ds_options)

valid_ds_opt = (
    valid_ds.map(valid_map_from_path, num_parallel_calls=AUTOTUNE)
    .cache()
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
).with_options(ds_options)


BASE0 = applications.MobileNet(
    include_top=False,
    input_shape=[IMG_SIZE, IMG_SIZE, 3],
    weights=None,
    pooling="max",
)


def build_model(input_size=[IMG_SIZE, IMG_SIZE, 3]):
    model = tf.keras.Sequential()
    model.add(BASE0)
    model.add(Dropout(0.5))
    model.add(Dense(5, activation="softmax"))

    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9),
        metrics=["accuracy", F1_score],
        run_eagerly=False,
    )
    return model


model0 = build_model()
model0.summary()



## === cell 7
callback0 = tf.keras.callbacks.ModelCheckpoint(
    "CasavaLeafDiseaseModel.h5",
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=False,
    verbose=1,
)



## === cell 8
history = model0.fit(
    train_ds_opt,
    validation_data=valid_ds_opt,
    epochs=EPOCHS,
    callbacks=[callback0],
    verbose=1,
)



## === cell 9
if os.path.exists("CasavaLeafDiseaseModel.h5"):
    model0 = tf.keras.models.load_model(
        "CasavaLeafDiseaseModel.h5", custom_objects={"F1_score": F1_score}
    )



## === cell 10
ss = pd.read_csv(sample_sub_path)

test_df = ss[["image_id"]].copy()

test_paths = tf.constant(
    [os.path.join(test_images_dir_path, f) for f in test_df["image_id"].values]
)


@tf.function
def load_and_preprocess(path):
    img = _decode_resize_from_path(path)
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


test_ds = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    .cache()
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
).with_options(ds_options)

probs = model0.predict(test_ds, verbose=1)
preds = np.argmax(probs, axis=1).astype(int)

my_submission = pd.DataFrame({"image_id": ss["image_id"].values, "label": preds})
my_submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", my_submission.shape)



## === cell 11
print("Submission File: \n---------------\n")
print(my_submission.head())
print("\nSaved to: submission.csv")
