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

# 5. Target score

0.7039891205802358

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.13341) has done: 'I added the missing imports, defined the TensorFlow model and its training callbacks, and ensured the data pipelines and submission creation use the correct variables. This fixes the NameError issues, allows the model to train, and writes a properly‑formatted `submission.csv` so the notebook runs end‑to‑end and can achieve the target accuracy.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

np.random.seed(42)
tf.random.set_seed(42)

BATCH_SIZE = 256

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

le = LabelEncoder()
train_csv["label_encoded"] = le.fit_transform(train_csv["label"].astype(str))

train_csv["disease"] = train_csv["disease"].astype(str)
train_csv["label"] = train_csv["label"].astype(str)

train_df, valid_df = train_test_split(
    train_csv, test_size=0.2, stratify=train_csv["label"], random_state=42
)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
AUTOTUNE = tf.data.AUTOTUNE
NUM_CLASSES = 5


def _load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = preprocess_input(image)
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label


def _load_image_test(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = preprocess_input(image)
    return image


train_paths = train_df["path"].values
train_labels = train_df["label_encoded"].values
train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_ds = train_ds.shuffle(buffer=len(train_paths), seed=42)
train_ds = train_ds.map(_load_image, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

valid_paths = valid_df["path"].values
valid_labels = valid_df["label_encoded"].values
valid_ds = tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels))
valid_ds = valid_ds.map(_load_image, num_parallel_calls=AUTOTUNE)
valid_ds = valid_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/527071709.py in <cell line: 0>()
     24 train_labels = train_df["label_encoded"].values
     25 train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
---> 26 train_ds = train_ds.shuffle(buffer=len(train_paths), seed=42)
     27 train_ds = train_ds.map(_load_image, num_parallel_calls=AUTOTUNE)
     28 train_ds = train_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

TypeError: DatasetV2.shuffle() got an unexpected keyword argument 'buffer'

## === cell 2
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(mode="horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(factor=0.125),  # ~45°
        tf.keras.layers.RandomTranslation(height_factor=0.2, width_factor=0.2),
        tf.keras.layers.RandomZoom(height_factor=0.2, width_factor=0.2),
        tf.keras.layers.RandomShear(factor=0.2),
    ]
)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), weights="imagenet", include_top=False
)
base_model.trainable = True

inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = base_model(x, training=True)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

early_stop = EarlyStopping(patience=3, restore_best_weights=True, verbose=1)
lr_reduce = ReduceLROnPlateau(patience=2, factor=0.5, verbose=1)

model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=12,
    callbacks=[early_stop, lr_reduce],
    verbose=2,
)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3951269755.py in <cell line: 0>()
      6         tf.keras.layers.RandomTranslation(height_factor=0.2, width_factor=0.2),
      7         tf.keras.layers.RandomZoom(height_factor=0.2, width_factor=0.2),
----> 8         tf.keras.layers.RandomShear(factor=0.2),
      9     ]
     10 )

/usr/local/lib/python3.11/dist-packages/keras/src/layers/preprocessing/image_preprocessing/random_shear.py in __init__(self, x_factor, y_factor, interpolation, fill_mode, fill_value, data_format, seed, **kwargs)
     81         **kwargs,
     82     ):
---> 83         super().__init__(data_format=data_format, **kwargs)
     84         self.x_factor = self._set_factor_with_name(x_factor, "x_factor")
     85         self.y_factor = self._set_factor_with_name(y_factor, "y_factor")

/usr/local/lib/python3.11/dist-packages/keras/src/layers/preprocessing/image_preprocessing/base_image_preprocessing_layer.py in __init__(self, factor, bounding_box_format, data_format, **kwargs)
     22             self._set_factor(factor)
     23         elif factor is not None:
---> 24             raise ValueError(
     25                 f"Layer {self.__class__.__name__} does not take "
     26                 f"a `factor` argument. Received: factor={factor}"

ValueError: Layer RandomShear does not take a `factor` argument. Received: factor=0.2

## === cell 3
test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
test_files = [f for f in os.listdir(test_image_dir) if f.lower().endswith(".jpg")]
test_df = pd.DataFrame(
    {
        "image_id": test_files,
        "path": [os.path.join(test_image_dir, f) for f in test_files],
    }
)

test_paths = test_df["path"].values
test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = test_ds.map(_load_image_test, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

pred_probs = model.predict(test_ds, verbose=0)
pred_classes = np.argmax(pred_probs, axis=1)

pred_label_str = le.inverse_transform(pred_classes)
pred_label_int = pred_label_str.astype(int)

submission_df = pd.DataFrame({"image_id": test_df["image_id"], "label": pred_label_int})
submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Submission file created: {submission_path}")
print(submission_df.head())

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3699868088.py in <cell line: 0>()
     13 test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
     14 
---> 15 pred_probs = model.predict(test_ds, verbose=0)
     16 pred_classes = np.argmax(pred_probs, axis=1)
     17 

NameError: name 'model' is not defined
