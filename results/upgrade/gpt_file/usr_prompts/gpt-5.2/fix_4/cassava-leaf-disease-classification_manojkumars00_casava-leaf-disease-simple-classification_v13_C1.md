# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

# 5. Target score

0.8735267452402539

# 6. Current score

0.05531

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.05531) has done: 'I fix the protobuf/TensorFlow import crash by forcing the pure-Python protobuf implementation early, which avoids the `MessageFactory.GetPrototype` error. Then I fix the dataset construction bug: `image_dataset_from_directory` can’t work here because the images are not arranged in per-class subfolders, so I switch to building a `tf.data.Dataset` from `train.csv` image paths/labels (same images, same 80/20 split semantics via a seeded shuffle). Finally, I keep your model/augmentations/training loop intact and ensure `model0` is always defined so prediction runs and a correctly-formatted `submission.csv` is written.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras.layers import Dense, Dropout

print("TF version:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
label_json_path = (
    "../input/cassava-leaf-disease-classification/label_num_to_disease_map.json"
)
images_dir_path = "../input/cassava-leaf-disease-classification/train_images"
test_images_dir_path = "../input/cassava-leaf-disease-classification/test_images"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"



## === cell 2
train_csv = pd.read_csv(train_csv_path)
train_csv["label"] = train_csv["label"].astype(str)

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
PRECISION = tf.keras.metrics.Precision()
RECALL = tf.keras.metrics.Recall()


def F1_score(y_true, y_pred):
    if hasattr(PRECISION, "reset_state"):
        PRECISION.reset_state()
        RECALL.reset_state()
    else:
        PRECISION.reset_states()
        RECALL.reset_states()

    PRECISION.update_state(y_true, y_pred)
    precision_out = PRECISION.result()

    RECALL.update_state(y_true, y_pred)
    recall_out = RECALL.result()

    return 2 * ((precision_out * recall_out) / (precision_out + recall_out + 1e-23))




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
def channel_shift(x, rng):
    batch = tf.shape(x)[0]
    shifts = rng.uniform(shape=(batch, 1, 1, 3), minval=-0.1, maxval=0.1, dtype=x.dtype)
    x = x + shifts
    return tf.clip_by_value(x, 0.0, 1.0)


rng = tf.random.Generator.from_seed(SEED)


def _decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    return img


def train_map(path, label_int):
    images = _decode_resize(path)
    images = tf.cast(images, tf.float32)
    images = augment(images, training=True)
    images = channel_shift(images, rng)
    labels = _one_hot(label_int)
    return images, labels


def valid_map(path, label_int):
    images = _decode_resize(path)
    images = tf.cast(images, tf.float32) * (1.0 / 255.0)
    labels = _one_hot(label_int)
    return images, labels


train_ds_opt = (
    train_ds.map(train_map, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

valid_ds_opt = (
    valid_ds.map(valid_map, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

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



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2284462062.py in <cell line: 0>()
----> 1 history = model0.fit(
      2     train_ds_opt,
      3     validation_data=valid_ds_opt,
      4     epochs=EPOCHS,
      5     callbacks=[callback0],

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/models/functional.py in _adjust_input_rank(self, flat_inputs)
    270                     adjusted.append(ops.expand_dims(x, axis=-1))
    271                     continue
--> 272             raise ValueError(
    273                 f"Invalid input shape for input {x}. Expected shape "
    274                 f"{ref_shape}, but input has incompatible shape {x.shape}"

ValueError: Exception encountered when calling Sequential.call().

Invalid input shape for input Tensor("data:0", shape=(None, 288, 288, 288, 3), dtype=float32). Expected shape (None, 288, 288, 3), but input has incompatible shape (None, 288, 288, 288, 3)

Arguments received by Sequential.call():
  • inputs=tf.Tensor(shape=(None, 288, 288, 288, 3), dtype=float32)
  • training=True
  • mask=None

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


def load_and_preprocess(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


test_ds = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

probs = model0.predict(test_ds, verbose=1)
preds = np.argmax(probs, axis=1).astype(int)

my_submission = pd.DataFrame({"image_id": ss["image_id"].values, "label": preds})
my_submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", my_submission.shape)



## === cell 11
print("Submission File: \n---------------\n")
print(my_submission.head())
print("\nSaved to: submission.csv")
