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
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
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
tf_keras==2.18.0

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
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

import json
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

try:
    cpu = os.cpu_count() or 4
    tf.config.threading.set_intra_op_parallelism_threads(max(2, cpu // 2))
    tf.config.threading.set_inter_op_parallelism_threads(max(2, cpu // 4))
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

SEED = 1337
keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TF:", tf.__version__)
print("Keras (tf.keras):", keras.__version__)

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = f"{DATA_DIR}/train.csv"
SAMPLE_SUB = f"{DATA_DIR}/sample_submission.csv"
TRAIN_IMG_DIR = f"{DATA_DIR}/train_images"
TEST_IMG_DIR = f"{DATA_DIR}/test_images"
LABEL_MAP_JSON = f"{DATA_DIR}/label_num_to_disease_map.json"

assert os.path.exists(TRAIN_CSV), TRAIN_CSV
assert os.path.exists(SAMPLE_SUB), SAMPLE_SUB
assert os.path.exists(TRAIN_IMG_DIR), TRAIN_IMG_DIR
assert os.path.exists(TEST_IMG_DIR), TEST_IMG_DIR
assert os.path.exists(LABEL_MAP_JSON), LABEL_MAP_JSON

train = pd.read_csv(TRAIN_CSV)
ss = pd.read_csv(SAMPLE_SUB)

print(train.shape, ss.shape)
train.head()



## === cell 1
pass



## === cell 2
train.info()



## === cell 3
train["label"].unique()



## === cell 4
from PIL import Image

im = Image.open(f"{TRAIN_IMG_DIR}/999616605.jpg")
print("opened")
print(im.size)



## === cell 5
train_path = TRAIN_IMG_DIR
test_path = TEST_IMG_DIR



## === cell 6
with open(LABEL_MAP_JSON, "r") as f:
    json_data = json.load(f)
json_data



## === cell 7
json_data



## === cell 8
label_id_to_name = {
    0: "Cassava Bacterial Blight (CBB)",
    1: "Cassava Brown Streak Disease (CBSD)",
    2: "Cassava Green Mottle (CGM)",
    3: "Cassava Mosaic Disease (CMD)",
    4: "Healthy",
}
train_labels_named = train.copy()
train_labels_named["label"] = (
    train_labels_named["label"].map(label_id_to_name).astype(str)
)
train_labels_named.head()



## === cell 9
train_labels_named.head()



## === cell 10
from tensorflow.keras import layers
from tensorflow.keras import models



## === cell 11
size = 512
bat_size = 16
split = 0.31
epoch = 10



## === cell 12
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-6,
    rotation_range=2,
    width_shift_range=0.0,
    height_shift_range=0.0,
    brightness_range=[0.0, 0.2],
    shear_range=0.0,
    zoom_range=0.1,
    channel_shift_range=0.0,
    fill_mode="nearest",
    cval=0.0,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=split,
)



## === cell 13
train_for_gen = train.copy()
train_for_gen["label"] = train_for_gen["label"].astype(str)

train_generator = train_datagen.flow_from_dataframe(
    train_for_gen,
    directory=train_path,
    x_col="image_id",
    y_col="label",
    target_size=(size, size),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=bat_size,
    shuffle=True,
    subset="training",
    interpolation="nearest",
    validate_filenames=False,
    seed=SEED,
)
train_ds = train_generator


def _gen_train():
    for batch in train_generator:
        yield batch


train_tfds = tf.data.Dataset.from_generator(
    _gen_train,
    output_signature=(
        tf.TensorSpec(shape=(None, size, size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 5), dtype=tf.float32),
    ),
)
AUTOTUNE = tf.data.AUTOTUNE
train_tfds = train_tfds.prefetch(AUTOTUNE)



## === cell 14
validation_generator = train_datagen.flow_from_dataframe(
    train_for_gen,
    directory=train_path,
    x_col="image_id",
    y_col="label",
    target_size=(size, size),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=bat_size,
    shuffle=True,
    subset="validation",
    interpolation="nearest",
    validate_filenames=False,
    seed=SEED,
)
val_ds = validation_generator


def _gen_val():
    for batch in validation_generator:
        yield batch


val_tfds = tf.data.Dataset.from_generator(
    _gen_val,
    output_signature=(
        tf.TensorSpec(shape=(None, size, size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 5), dtype=tf.float32),
    ),
).prefetch(AUTOTUNE)



## === cell 15
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    BatchNormalization,
    Dropout,
)

cnn_model = Sequential()
cnn_model.add(
    Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(size, size, 3))
)
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(BatchNormalization())
cnn_model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(BatchNormalization())
cnn_model.add(Conv2D(96, kernel_size=(3, 3), activation="relu"))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(BatchNormalization())
cnn_model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(BatchNormalization())
cnn_model.add(Conv2D(256, kernel_size=(3, 3), activation="relu"))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(BatchNormalization())
cnn_model.add(Dropout(0.2))
cnn_model.add(Flatten())
cnn_model.add(Dense(64, activation="relu"))
cnn_model.add(Dense(64, activation="relu"))
cnn_model.add(
    Dense(
        128,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
        bias_regularizer=tf.keras.regularizers.l2(1e-4),
        activity_regularizer=tf.keras.regularizers.l2(1e-5),
    )
)
cnn_model.add(
    Dense(
        256,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
        bias_regularizer=tf.keras.regularizers.l2(1e-4),
        activity_regularizer=tf.keras.regularizers.l2(1e-5),
    )
)
cnn_model.add(Dropout(0.25))
cnn_model.add(Dense(5, activation="softmax"))

opt = tf.keras.optimizers.Adam(learning_rate=0.001)
cnn_model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])



## === cell 16
callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    min_delta=0.0,
    patience=2,
    verbose=1,
    mode="max",
    restore_best_weights=True,
)



## === cell 17
pass



## === cell 18
from tensorflow.keras.applications import EfficientNetB7

backbone = EfficientNetB7(
    include_top=False, weights="imagenet", input_shape=(size, size, 3)
)
backbone.trainable = False

inputs = keras.Input(shape=(size, size, 3))
x = backbone(inputs, training=False)
x = keras.layers.Flatten()(x)
outputs = keras.layers.Dense(5, activation="softmax")(x)
model = keras.Model(inputs, outputs)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
)



## === cell 19
model.summary()



## === cell 20
history = model.fit(
    train_tfds,
    epochs=epoch,
    validation_data=val_tfds,
    verbose=1,
    callbacks=[callback],
    steps_per_epoch=len(train_generator),
    validation_steps=len(validation_generator),
)



## === cell 21
pass



## === cell 22
pass



## === cell 23
pass



## === cell 24
pass



## === cell 25
pass



## === cell 26
pass



## === cell 27
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_df = ss.copy()

test_generator = test_datagen.flow_from_dataframe(
    test_df,
    directory=TEST_IMG_DIR,
    x_col="image_id",
    y_col=None,
    target_size=(size, size),
    color_mode="rgb",
    class_mode=None,
    batch_size=bat_size,
    shuffle=False,  # critical to preserve ss order
    interpolation="nearest",
    validate_filenames=False,
)


def _gen_test():
    for batch in test_generator:
        yield batch


test_tfds = tf.data.Dataset.from_generator(
    _gen_test,
    output_signature=tf.TensorSpec(shape=(None, size, size, 3), dtype=tf.float32),
).prefetch(AUTOTUNE)

probs = model.predict(
    test_tfds,
    verbose=1,
    steps=len(test_generator),
)
pred_class_idx = np.argmax(probs, axis=1)

inv_class_indices = {v: k for k, v in train_generator.class_indices.items()}
pred_labels = [int(inv_class_indices[int(i)]) for i in pred_class_idx]

my_submission = pd.DataFrame({"image_id": ss.image_id, "label": pred_labels})
my_submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", my_submission.shape)
my_submission.head()



## === cell 28
my_submission



## === cell 29
assert os.path.exists("submission.csv")
sub_chk = pd.read_csv("submission.csv")
assert list(sub_chk.columns) == ["image_id", "label"]
assert len(sub_chk) == len(ss)
assert sub_chk["label"].between(0, 4).all()
sub_chk.head()



## === cell 30
model.save("efficientnetb7_cassava.keras")
print("Saved model to efficientnetb7_cassava.keras")
