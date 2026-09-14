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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import mixed_precision
from tensorflow.keras.preprocessing import image

try:
    from tensorflow.keras.applications.efficientnet import (
        EfficientNetB0,
        preprocess_input,
    )
except Exception:
    from tensorflow.keras.applications.mobilenet_v2 import (
        MobileNetV2 as EfficientNetB0,
        preprocess_input,
    )

mixed_precision.set_global_policy("float32")

tf.random.set_seed(42)
np.random.seed(42)
os.environ["PYTHONHASHSEED"] = "42"

label_to_disease = pd.read_json(
    "/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json",
    typ="series",
)

train_csv = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/train.csv")
train_csv["path"] = (
    "/kaggle/input/cassava-leaf-disease-classification/train_images/"
    + train_csv["image_id"]
)
train_csv["label"] = train_csv["label"].astype(str)

train_df, valid_df = train_test_split(
    train_csv,
    test_size=0.2,
    stratify=train_csv["label"],
    random_state=42,
)

BATCH_SIZE = 64
NUM_CLASSES = 5
IMG_SIZE = (224, 224)


class DataGenerator(tf.keras.utils.Sequence):
    """Keras Sequence that loads images from disk, applies preprocessing and optional augmentation."""

    def __init__(self, df, batch_size, is_training):
        self.paths = df["path"].values
        self.labels = tf.one_hot(
            df["label"].astype(int).values, depth=NUM_CLASSES, dtype=tf.float32
        ).numpy()
        self.batch_size = batch_size
        self.is_training = is_training
        self.indexes = np.arange(len(self.paths))
        if self.is_training:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_paths = self.paths[batch_idx]

        batch_images = []
        for p in batch_paths:
            img = image.load_img(p, target_size=IMG_SIZE)
            img = image.img_to_array(img)
            img = preprocess_input(img)

            if self.is_training:
                if np.random.rand() < 0.5:
                    img = np.fliplr(img)
                if np.random.rand() < 0.5:
                    img = np.flipud(img)

            batch_images.append(img)

        x = np.stack(batch_images, axis=0).astype(np.float32)
        y = self.labels[batch_idx]
        return x, y

    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.indexes)


train_generator = DataGenerator(train_df, batch_size=BATCH_SIZE, is_training=True)
valid_generator = DataGenerator(valid_df, batch_size=BATCH_SIZE, is_training=False)




## === cell 1
try:
    base_model = EfficientNetB0(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )
except Exception:
    base_model = EfficientNetB0(
        weights=None, include_top=False, input_shape=(224, 224, 3)
    )

base_model.trainable = True

x = GlobalAveragePooling2D()(base_model.output)
output = Dense(NUM_CLASSES, activation="softmax", dtype="float32")(
    x
)  # ensure float32 output
model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
)
lr_reduction = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
)




## === cell 2
model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=30,
    callbacks=[early_stopping, lr_reduction],
    verbose=2,
)




## === cell 3
sample_sub = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
sample_sub["path"] = test_dir + "/" + sample_sub["image_id"]


class TestGenerator(tf.keras.utils.Sequence):
    """Generator for test images – returns only preprocessed image batches."""

    def __init__(self, df, batch_size):
        self.paths = df["path"].values
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.paths))

    def __len__(self):
        return int(np.ceil(len(self.paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_paths = self.paths[batch_idx]

        batch_images = []
        for p in batch_paths:
            img = image.load_img(p, target_size=IMG_SIZE)
            img = image.img_to_array(img)
            img = preprocess_input(img)
            batch_images.append(img)

        x = np.stack(batch_images, axis=0).astype(np.float32)
        return x


test_generator = TestGenerator(sample_sub, batch_size=BATCH_SIZE)

pred_probs = model.predict(test_generator, verbose=2)
pred_classes = np.argmax(pred_probs, axis=1).astype(int)

submission_df = pd.DataFrame(
    {"image_id": sample_sub["image_id"], "label": pred_classes}
)

submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission file created: {submission_path}")
print(submission_df.head())
