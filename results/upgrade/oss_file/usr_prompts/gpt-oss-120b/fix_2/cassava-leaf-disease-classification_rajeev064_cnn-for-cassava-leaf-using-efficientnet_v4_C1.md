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

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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
tqdm==4.67.1

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

0.0507

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Dropout, GlobalAveragePooling2D
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.applications import EfficientNetB3



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
base_path = os.path.join("..", "input", "cassava-leaf-disease-classification")
training_folder = os.path.join(base_path, "train_images")
test_folder = os.path.join(base_path, "test_images")
train_csv_path = os.path.join(base_path, "train.csv")
sample_submission_path = os.path.join(base_path, "sample_submission.csv")



## === cell 2
samples_df = pd.read_csv(train_csv_path)
samples_df = samples_df.sample(frac=1, random_state=42).reset_index(drop=True)
samples_df["filepath"] = samples_df["image_id"].apply(
    lambda x: os.path.join(training_folder, x)
)



## === cell 3
training_percentage = 0.8
training_item_count = int(len(samples_df) * training_percentage)
training_df = samples_df.iloc[:training_item_count]
validation_df = samples_df.iloc[training_item_count:]



## === cell 4
batch_size = 8
image_size = 512
input_shape = (image_size, image_size, 3)
dropout_rate = 0.4
classes_to_predict = sorted(samples_df.label.unique())



## === cell 5
training_data = tf.data.Dataset.from_tensor_slices(
    (training_df.filepath.values, training_df.label.values.astype(np.int32))
)
validation_data = tf.data.Dataset.from_tensor_slices(
    (validation_df.filepath.values, validation_df.label.values.astype(np.int32))
)




## === cell 6
def load_image_and_label_from_path(image_path, label):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [image_size, image_size])
    return img, label


AUTOTUNE = tf.data.AUTOTUNE

training_data = training_data.map(
    load_image_and_label_from_path, num_parallel_calls=AUTOTUNE
)
validation_data = validation_data.map(
    load_image_and_label_from_path, num_parallel_calls=AUTOTUNE
)

training_data_batches = (
    training_data.shuffle(buffer_size=1000)
    .batch(batch_size)
    .prefetch(buffer_size=AUTOTUNE)
)
validation_data_batches = (
    validation_data.shuffle(buffer_size=1000)
    .batch(batch_size)
    .prefetch(buffer_size=AUTOTUNE)
)



## === cell 7
data_augmentation_layers = tf.keras.Sequential(
    [
        layers.experimental.preprocessing.RandomCrop(image_size, image_size),
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomRotation(0.25),
        layers.experimental.preprocessing.RandomZoom((-0.2, 0)),
        layers.experimental.preprocessing.RandomContrast((0.2, 0.2)),
    ]
)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/3435936392.py in <cell line: 0>()
      1 data_augmentation_layers = tf.keras.Sequential(
      2     [
----> 3         layers.experimental.preprocessing.RandomCrop(image_size, image_size),
      4         layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
      5         layers.experimental.preprocessing.RandomRotation(0.25),

AttributeError: module 'tensorflow.keras.layers' has no attribute 'experimental'

## === cell 8
efficientnet = EfficientNetB3(
    weights="imagenet", include_top=False, input_shape=input_shape
)

inputs = Input(shape=input_shape)
augmented = data_augmentation_layers(inputs)
features = efficientnet(augmented)
pooling = GlobalAveragePooling2D()(features)
dropout = Dropout(dropout_rate)(pooling)
outputs = Dense(len(classes_to_predict), activation="softmax")(dropout)
model = Model(inputs=inputs, outputs=outputs)

model.summary()



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2480799896.py in <cell line: 0>()
      4 
      5 inputs = Input(shape=input_shape)
----> 6 augmented = data_augmentation_layers(inputs)
      7 features = efficientnet(augmented)
      8 pooling = GlobalAveragePooling2D()(features)

NameError: name 'data_augmentation_layers' is not defined

## === cell 9
epochs = 1  # minimal epoch to keep runtime low



## === cell 10
decay_steps = int(np.ceil(len(training_df) / batch_size)) * epochs
cosine_decay = CosineDecay(
    initial_learning_rate=1e-4, decay_steps=decay_steps, alpha=0.3
)

callbacks = [
    ModelCheckpoint(filepath="best_model.h5", monitor="val_loss", save_best_only=True)
]

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=cosine_decay),
    metrics=["accuracy"],
)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4046867814.py in <cell line: 0>()
      8 ]
      9 
---> 10 model.compile(
     11     loss="sparse_categorical_crossentropy",
     12     optimizer=tf.keras.optimizers.Adam(learning_rate=cosine_decay),

NameError: name 'model' is not defined

## === cell 11
history = model.fit(
    training_data_batches,
    epochs=epochs,
    validation_data=validation_data_batches,
    callbacks=callbacks,
)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/200394358.py in <cell line: 0>()
----> 1 history = model.fit(
      2     training_data_batches,
      3     epochs=epochs,
      4     validation_data=validation_data_batches,
      5     callbacks=callbacks,

NameError: name 'model' is not defined

## === cell 12
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("Loss over epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/849786837.py in <cell line: 0>()
----> 1 plt.plot(history.history["loss"], label="train")
      2 plt.plot(history.history["val_loss"], label="val")
      3 plt.title("Loss over epochs")
      4 plt.xlabel("Epoch")
      5 plt.ylabel("Loss")

NameError: name 'history' is not defined

## === cell 13
model.load_weights("best_model.h5")




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/607800890.py in <cell line: 0>()
----> 1 model.load_weights("best_model.h5")
      2 
      3 

NameError: name 'model' is not defined

## === cell 14
def scan_over_image(img_path, crop_size=512):
    img = Image.open(img_path)
    img = np.array(img)
    h, w, _ = img.shape
    xs = [0, max(w - crop_size, 0)]
    ys = [0, max(h - crop_size, 0)]
    patches = []
    for x in xs:
        for y in ys:
            patch = img[y : y + crop_size, x : x + crop_size, :]
            if patch.shape[0] == crop_size and patch.shape[1] == crop_size:
                patches.append(patch)
    return np.array(patches)


def display_samples(img_path):
    img_list = scan_over_image(img_path)
    fig = plt.figure(figsize=(8, len(img_list) * 2))
    for i, img in enumerate(img_list):
        ax = fig.add_subplot(1, len(img_list), i + 1)
        ax.imshow(img)
        ax.set_title(str(i))
        ax.axis("off")
    plt.show()





## === cell 15
test_time_augmentation_layers = tf.keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomZoom((-0.2, 0)),
        layers.experimental.preprocessing.RandomContrast((0.2, 0.2)),
    ]
)




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/69788865.py in <cell line: 0>()
      1 test_time_augmentation_layers = tf.keras.Sequential(
      2     [
----> 3         layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
      4         layers.experimental.preprocessing.RandomZoom((-0.2, 0)),
      5         layers.experimental.preprocessing.RandomContrast((0.2, 0.2)),

AttributeError: module 'tensorflow.keras.layers' has no attribute 'experimental'

## === cell 16
def predict_and_vote(image_filename, folder, TTA_runs=4):
    local_image_list = scan_over_image(os.path.join(folder, image_filename))
    localised_predictions = []
    for local_image in local_image_list:
        duplicated = np.stack([local_image] * TTA_runs, axis=0)
        duplicated_tensor = tf.convert_to_tensor(duplicated, dtype=tf.float32)
        augmented = test_time_augmentation_layers(duplicated_tensor)
        preds = model.predict(augmented, verbose=0)
        localised_predictions.append(np.sum(preds, axis=0))
    global_pred = np.sum(localised_predictions, axis=0)
    return int(np.argmax(global_pred))


def run_predictions_over_image_list(image_list, folder):
    predictions = []
    for img_name in tqdm(image_list):
        predictions.append(predict_and_vote(img_name, folder))
    return predictions




## === cell 17
test_image_ids = os.listdir(test_folder)
submission_df = pd.DataFrame({"image_id": test_image_ids})
submission_df["label"] = run_predictions_over_image_list(
    submission_df["image_id"], test_folder
)



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3865339690.py in <cell line: 0>()
      1 test_image_ids = os.listdir(test_folder)
      2 submission_df = pd.DataFrame({"image_id": test_image_ids})
----> 3 submission_df["label"] = run_predictions_over_image_list(
      4     submission_df["image_id"], test_folder
      5 )

/tmp/ipykernel_55/3982814439.py in run_predictions_over_image_list(image_list, folder)
     15     predictions = []
     16     for img_name in tqdm(image_list):
---> 17         predictions.append(predict_and_vote(img_name, folder))
     18     return predictions
     19 

/tmp/ipykernel_55/3982814439.py in predict_and_vote(image_filename, folder, TTA_runs)
      5         duplicated = np.stack([local_image] * TTA_runs, axis=0)
      6         duplicated_tensor = tf.convert_to_tensor(duplicated, dtype=tf.float32)
----> 7         augmented = test_time_augmentation_layers(duplicated_tensor)
      8         preds = model.predict(augmented, verbose=0)
      9         localised_predictions.append(np.sum(preds, axis=0))

NameError: name 'test_time_augmentation_layers' is not defined

## === cell 18
submission_df.head()



## === cell 19
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

## --- ERROR in outputing the csv:
Invalid submission: Submission must have the same length as the answers.
