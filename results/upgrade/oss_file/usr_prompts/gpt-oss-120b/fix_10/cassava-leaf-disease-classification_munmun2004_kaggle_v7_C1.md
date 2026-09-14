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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Flatten,
)
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import random
import warnings
import matplotlib.pyplot as plt

try:
    from tensorflow.keras import mixed_precision

    mixed_precision.set_global_policy("mixed_float16")
except Exception:
    pass




## === cell 1
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
if tf.config.list_physical_devices("GPU"):
    print("Yes, there is GPU")
tf.debugging.set_log_device_placement(True)




## === cell 2
work_dir = "/kaggle/input/cassava-leaf-disease-classification/"
train_path = "/kaggle/input/cassava-leaf-disease-classification/train_images"
print(os.listdir(work_dir)[:3])  # sanity check




## === cell 3
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


seed = 21
seed_everything(seed)
warnings.filterwarnings("ignore")




## === cell 4
data = pd.read_csv(os.path.join(work_dir, "train.csv"))
print(data["label"].value_counts())




## === cell 5
import json

with open(os.path.join(work_dir, "label_num_to_disease_map.json")) as f:
    real_labels = json.load(f)
    real_labels = {int(k): v for k, v in real_labels.items()}
data["class_name"] = data["label"].map(real_labels)




## === cell 6
from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(
    data, test_size=0.05, random_state=42, stratify=data["class_name"]
)




## === cell 7
IMG_SIZE = 456
size = (IMG_SIZE, IMG_SIZE)
n_CLASS = 5
BATCH_SIZE = 48




## === cell 8
datagen_train = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest",
)

datagen_val = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
)




## === cell 9
train_set = datagen_train.flow_from_dataframe(
    train_df,
    directory=train_path,
    seed=42,
    x_col="image_id",
    y_col="class_name",
    target_size=size,
    class_mode="categorical",
    interpolation="nearest",
    shuffle=True,
    batch_size=BATCH_SIZE,
)

val_set = datagen_val.flow_from_dataframe(
    val_df,
    directory=train_path,
    seed=42,
    x_col="image_id",
    y_col="class_name",
    target_size=size,
    class_mode="categorical",
    interpolation="nearest",
    shuffle=False,
    batch_size=BATCH_SIZE,
)




## === cell 10
def create_model():
    model = Sequential()
    model.add(
        EfficientNetB3(
            input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet"
        )
    )
    model.add(GlobalAveragePooling2D())
    model.add(Flatten())
    model.add(
        Dense(
            256,
            activation="relu",
            bias_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001),
        )
    )
    model.add(Dropout(0.5))
    model.add(Dense(n_CLASS, activation="softmax"))
    return model


leaf_model = create_model()
leaf_model.summary()




## === cell 11
loss_fn = tf.keras.losses.CategoricalCrossentropy(
    from_logits=False, label_smoothing=0.0001, name="categorical_crossentropy"
)

leaf_model.compile(
    optimizer=Adam(learning_rate=1e-3), loss=loss_fn, metrics=["categorical_accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_loss", mode="min", patience=3, restore_best_weights=True, verbose=1
)

model_save = ModelCheckpoint(
    filepath="best_model.h5",
    save_best_only=True,
    monitor="val_loss",
    mode="min",
    verbose=1,
    save_weights_only=False,
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6, mode="min", verbose=1
)

EPOCHS = 10  # increased epochs for better accuracy

import math

steps_per_epoch = math.ceil(train_set.n / BATCH_SIZE)
validation_steps = math.ceil(val_set.n / BATCH_SIZE)

history = leaf_model.fit(
    train_set,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_set,
    validation_steps=validation_steps,
    epochs=EPOCHS,
    callbacks=[early_stop, model_save, reduce_lr],
    verbose=2,
)




## === cell 12
if "history" in locals():
    plt.figure(figsize=(8, 4))
    plt.plot(history.epoch, history.history["loss"], "-o", label="train loss")
    plt.plot(history.epoch, history.history["val_loss"], "-o", label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()




## === cell 13
if "history" in locals():
    plt.figure(figsize=(8, 4))
    plt.plot(
        history.epoch, history.history["categorical_accuracy"], "-o", label="train acc"
    )
    plt.plot(
        history.epoch,
        history.history["val_categorical_accuracy"],
        "-o",
        label="val acc",
    )
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()




## === cell 14
if os.path.exists("best_model.h5"):
    trained_model = tf.keras.models.load_model("best_model.h5")
    print("Loaded model from best_model.h5")
else:
    trained_model = leaf_model
    print("Using the model from the current session (no checkpoint found).")




## === cell 15
sample_sub = pd.read_csv(os.path.join(work_dir, "sample_submission.csv"))
test_images_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"

test_df = pd.DataFrame({"image_id": sample_sub["image_id"]})

datagen_test = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
)

test_set = datagen_test.flow_from_dataframe(
    test_df,
    directory=test_images_dir,
    x_col="image_id",
    y_col=None,
    target_size=size,
    class_mode=None,
    interpolation="nearest",
    shuffle=False,
    batch_size=BATCH_SIZE,
)

preds_probs = trained_model.predict(test_set, verbose=0)
preds = np.argmax(preds_probs, axis=1)

submission = pd.DataFrame(
    {"image_id": sample_sub["image_id"], "label": preds.astype(int)}
)
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")
print(submission.head())
