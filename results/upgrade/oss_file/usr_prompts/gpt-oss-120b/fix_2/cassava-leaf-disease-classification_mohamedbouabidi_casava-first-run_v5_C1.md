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
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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

# 5. Target score

0.0016

# 6. Current score

0.10874

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.10874) has done: 'I guard all TensorFlow/Keras imports and model‑training code so that they are only executed when the libraries load successfully; otherwise the script falls back to a simple baseline that predicts the most frequent label from the training set. This removes the import‑related errors, prevents the ModelCheckpoint filename error, and guarantees that a valid `submission.csv` with the correct columns is written. The changes are minimal and keep the original workflow structure while ensuring the notebook runs end‑to‑end.'

# 9. Code solution

## === cell 0
import os, json, numpy as np, pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import (
        ImageDataGenerator,
        load_img,
        img_to_array,
    )
    from tensorflow.keras.callbacks import (
        ModelCheckpoint,
        EarlyStopping,
        ReduceLROnPlateau,
    )
    from tensorflow.keras.applications import EfficientNetB7
    from tensorflow import keras
    from keras import layers, models
except Exception as e:
    print("TensorFlow import failed; proceeding with a fallback baseline.")
    tf = None
    ImageDataGenerator = None
    ModelCheckpoint = EarlyStopping = ReduceLROnPlateau = None
    EfficientNetB7 = None
    keras = None
    layers = models = None



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
base_dir = "../input/cassava-leaf-disease-classification"
print("Base directory contents:", os.listdir(base_dir)[:5])



## === cell 2
train_labels = pd.read_csv(os.path.join(base_dir, "train.csv"))
train_labels.head()



## === cell 3
BATCH_SIZE = 20
TARGET_SIZE = 224
EPOCHS = 20
STEPS_PER_EPOCH = len(train_labels) * 0.8 / BATCH_SIZE
VALIDATION_STEPS = len(train_labels) * 0.2 / BATCH_SIZE



## === cell 4
if ImageDataGenerator is not None:
    train_labels["label"] = train_labels["label"].astype(str)

    train_datagen = ImageDataGenerator(
        validation_split=0.2,
        rotation_range=40,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest",
        shear_range=0.2,
        height_shift_range=0.2,
        width_shift_range=0.2,
    )

    train_generator = train_datagen.flow_from_dataframe(
        train_labels,
        directory=os.path.join(base_dir, "train_images"),
        subset="training",
        x_col="image_id",
        y_col="label",
        target_size=(TARGET_SIZE, TARGET_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="sparse",
    )

    validation_datagen = ImageDataGenerator(validation_split=0.2)
    validation_generator = validation_datagen.flow_from_dataframe(
        train_labels,
        directory=os.path.join(base_dir, "train_images"),
        subset="validation",
        x_col="image_id",
        y_col="label",
        target_size=(TARGET_SIZE, TARGET_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="sparse",
    )
else:
    train_generator = validation_generator = None



## === cell 5
if tf is not None and EfficientNetB7 is not None:
    eff_base = EfficientNetB7(
        include_top=False, weights="imagenet", input_shape=(TARGET_SIZE, TARGET_SIZE, 3)
    )
    model = models.Sequential(
        [
            eff_base,
            layers.GlobalAveragePooling2D(),
            layers.Dense(5, activation="softmax", name="Output"),
        ]
    )
    model.compile(
        optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["acc"]
    )
else:
    model = None



## === cell 6
if ModelCheckpoint is not None and model is not None:
    model_save = ModelCheckpoint(
        "./EffNetB7_best_weights.weights.h5",  # must end with .weights.h5 when save_weights_only=True
        save_best_only=True,
        save_weights_only=True,
        monitor="val_loss",
        mode="min",
        verbose=1,
    )
    early_stop = EarlyStopping(
        monitor="val_loss",
        min_delta=0.001,
        patience=5,
        mode="min",
        verbose=1,
        restore_best_weights=True,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_delta=0.001,
        mode="min",
        verbose=1,
    )
    callbacks = [model_save, early_stop, reduce_lr]
else:
    callbacks = []



## === cell 7
if model is not None and train_generator is not None:
    history = model.fit(
        train_generator,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=VALIDATION_STEPS,
        callbacks=callbacks,
    )
else:
    history = None



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1856573485.py in <cell line: 0>()
      1 # Train only if everything is set up.
      2 if model is not None and train_generator is not None:
----> 3     history = model.fit(
      4         train_generator,
      5         steps_per_epoch=STEPS_PER_EPOCH,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/trainers/epoch_iterator.py in _enumerate_iterator(self)
    102                 self._current_iterator = iter(self._get_iterator())
    103                 self._steps_seen = 0
--> 104             for step in range(0, steps_per_epoch, self.steps_per_execution):
    105                 if self._num_batches and self._steps_seen >= self._num_batches:
    106                     if self.steps_per_epoch:

TypeError: 'float' object cannot be interpreted as an integer

## === cell 8
if history is not None:
    acc = history.history["acc"]
    val_acc = history.history["val_acc"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    sns.set_style("white")
    plt.suptitle("Train history", size=15)

    ax1.plot(epochs_range, acc, "bo", label="Training acc")
    ax1.plot(epochs_range, val_acc, "b", label="Validation acc")
    ax1.set_title("Training and validation acc")
    ax1.legend()

    ax2.plot(epochs_range, loss, "bo", label="Training loss", color="red")
    ax2.plot(epochs_range, val_loss, "b", label="Validation loss", color="red")
    ax2.set_title("Training and validation loss")
    ax2.legend()
    plt.show()
else:
    print("Training was skipped; no history to plot.")



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2369509820.py in <cell line: 0>()
      1 # Plot training history if available.
----> 2 if history is not None:
      3     acc = history.history["acc"]
      4     val_acc = history.history["val_acc"]
      5     loss = history.history["loss"]

NameError: name 'history' is not defined

## === cell 9
sub = pd.read_csv(os.path.join(base_dir, "sample_submission.csv"))
sub.head()



## === cell 10
most_common_label = train_labels["label"].astype(str).mode()[0]
print(f"Fallback label (most common in training): {most_common_label}")



## === cell 11
preds = []
if model is not None:
    for image_id in sub["image_id"]:
        img_path = os.path.join(base_dir, "test_images", image_id)
        img = Image.open(img_path).convert("RGB")
        img = img.resize((TARGET_SIZE, TARGET_SIZE))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
        pred = np.argmax(model.predict(img_array, verbose=0))
        preds.append(str(pred))
else:
    preds = [most_common_label] * len(sub)

sub["label"] = preds
sub.head()



## === cell 12
output_path = "submission.csv"
sub.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")
