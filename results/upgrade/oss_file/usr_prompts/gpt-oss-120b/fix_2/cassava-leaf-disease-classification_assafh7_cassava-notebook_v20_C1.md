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

0.837413115744938

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
import os
import json
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    Dropout,
    Flatten,
    Dense,
    Conv2D,
    MaxPooling2D,
    GlobalAveragePooling2D,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
import warnings

warnings.simplefilter("ignore")


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        pass  # Suppressed output for brevity


## === cell 2
general_path = "../input/cassava-leaf-disease-classification/"


## === cell 3
with open(os.path.join(general_path, "label_num_to_disease_map.json")) as file:
    map_classes = json.loads(file.read())
    map_classes = {int(k): v for k, v in map_classes.items()}


## === cell 4
train = pd.read_csv(os.path.join(general_path, "train.csv"))
train["label"] = train["label"].astype(str)  # ensure string labels for generators
train["class_name"] = train["label"].map(map_classes)


## === cell 5
plt.figure(figsize=(8, 4))
sns.countplot(y="class_name", data=train)


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/518712277.py in <cell line: 0>()
      1 plt.figure(figsize=(8, 4))
----> 2 sns.countplot(y="class_name", data=train)

/usr/local/lib/python3.11/dist-packages/seaborn/categorical.py in countplot(data, x, y, hue, order, hue_order, orient, color, palette, saturation, width, dodge, ax, **kwargs)
   2941         raise ValueError("Cannot pass values for both `x` and `y`")
   2942 
-> 2943     plotter = _CountPlotter(
   2944         x, y, hue, data, order, hue_order,
   2945         estimator, errorbar, n_boot, units, seed,

/usr/local/lib/python3.11/dist-packages/seaborn/categorical.py in __init__(self, x, y, hue, data, order, hue_order, estimator, errorbar, n_boot, units, seed, orient, color, palette, saturation, width, errcolor, errwidth, capsize, dodge)
   1530         self.establish_variables(x, y, hue, data, orient,
   1531                                  order, hue_order, units)
-> 1532         self.establish_colors(color, palette, saturation)
   1533         self.estimate_statistic(estimator, errorbar, n_boot, seed)
   1534 

/usr/local/lib/python3.11/dist-packages/seaborn/categorical.py in establish_colors(self, color, palette, saturation)
    705         # Determine the gray color to use for the lines framing the plot
    706         light_vals = [rgb_to_hls(*c)[1] for c in rgb_colors]
--> 707         lum = min(light_vals) * .6
    708         gray = mpl.colors.rgb2hex((lum, lum, lum))
    709 

ValueError: min() arg is an empty sequence

## === cell 6
img_width, img_height = 224, 224
batch_size = 64


## === cell 7
datagen = ImageDataGenerator(
    validation_split=0.2,
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
)

train_gen = datagen.flow_from_dataframe(
    dataframe=train,
    directory=os.path.join(general_path, "train_images"),
    x_col="image_id",
    y_col="label",
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical",
    subset="training",
    shuffle=True,
)

valid_gen = datagen.flow_from_dataframe(
    dataframe=train,
    directory=os.path.join(general_path, "train_images"),
    x_col="image_id",
    y_col="label",
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
)


## === cell 8
x_batch, y_batch = next(train_gen)
plt.figure(figsize=(12, 6))
for i in range(min(5, x_batch.shape[0])):
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_batch[i])
    plt.title(f"Class {np.argmax(y_batch[i])}")
    plt.axis("off")
plt.show()


## === cell 9
num_classes = train_gen.num_classes
model = Sequential(
    [
        Conv2D(32, (3, 3), activation="relu", input_shape=(img_width, img_height, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu"),
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(128, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ]
)
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3072472510.py in <cell line: 0>()
      1 # Build a small CNN model
----> 2 num_classes = train_gen.num_classes
      3 model = Sequential(
      4     [
      5         Conv2D(32, (3, 3), activation="relu", input_shape=(img_width, img_height, 3)),

AttributeError: 'DataFrameIterator' object has no attribute 'num_classes'

## === cell 10
epochs = 3
model.fit(
    train_gen,
    steps_per_epoch=train_gen.samples // batch_size,
    validation_data=valid_gen,
    validation_steps=valid_gen.samples // batch_size,
    epochs=epochs,
    verbose=2,
)


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3304413048.py in <cell line: 0>()
      1 # Train the model (few epochs for speed)
      2 epochs = 3
----> 3 model.fit(
      4     train_gen,
      5     steps_per_epoch=train_gen.samples // batch_size,

NameError: name 'model' is not defined

## === cell 11
sample_submission = pd.read_csv(os.path.join(general_path, "sample_submission.csv"))
test_gen = ImageDataGenerator(rescale=1.0 / 255).flow_from_dataframe(
    dataframe=sample_submission,
    directory=os.path.join(general_path, "test_images"),
    x_col="image_id",
    y_col=None,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False,
)


## === cell 12
pred_probs = model.predict(
    test_gen, steps=np.ceil(sample_submission.shape[0] / batch_size), verbose=0
)
pred_indices = np.argmax(pred_probs, axis=1)
idx_to_class = {v: k for k, v in train_gen.class_indices.items()}
pred_labels = [idx_to_class[idx] for idx in pred_indices]


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/284136034.py in <cell line: 0>()
      1 # Predict on test set
----> 2 pred_probs = model.predict(
      3     test_gen, steps=np.ceil(sample_submission.shape[0] / batch_size), verbose=0
      4 )
      5 pred_indices = np.argmax(pred_probs, axis=1)

NameError: name 'model' is not defined

## === cell 13
submission = pd.DataFrame(
    {"image_id": sample_submission["image_id"], "label": pred_labels}
)
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2176969820.py in <cell line: 0>()
      1 # Create submission file
      2 submission = pd.DataFrame(
----> 3     {"image_id": sample_submission["image_id"], "label": pred_labels}
      4 )
      5 submission_path = "submission.csv"

NameError: name 'pred_labels' is not defined
