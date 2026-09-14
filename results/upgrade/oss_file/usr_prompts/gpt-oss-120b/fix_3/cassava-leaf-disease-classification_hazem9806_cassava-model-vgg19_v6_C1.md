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
import numpy as np
import pandas as pd
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt

try:
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    from keras.callbacks import EarlyStopping
    from tensorflow.keras.applications import VGG19
    from tensorflow.keras.applications.vgg19 import preprocess_input
except Exception as e:
    tf = None
    ImageDataGenerator = None
    print("TensorFlow import failed; model training will be skipped.", e)



## === cell 1
main_directory = "/kaggle/input/cassava-leaf-disease-classification/"
training_images_path = os.path.join(main_directory, "train_images")
print("List of top‑level files in main directory:\n", os.listdir(main_directory))



## === cell 2
image_label_data = pd.read_csv(os.path.join(main_directory, "train.csv"))
labels = pd.read_json(
    os.path.join(main_directory, "label_num_to_disease_map.json"), typ="series"
)
print("Cassava Leaf Disease Classification Labels are:\n", dict(labels))
image_label_data["disease_name"] = image_label_data["label"].map(labels)



## === cell 3
most_common_label = image_label_data["label"].mode()[0]
print(f"Most common label (fallback): {most_common_label}")



## === cell 4
if tf is not None:
    from sklearn.model_selection import train_test_split

    TEST_PERCENTAGE = 0.05
    train_set_splitted, validation_set_splitted = train_test_split(
        image_label_data,
        test_size=TEST_PERCENTAGE,
        random_state=42,
        stratify=image_label_data["disease_name"],
    )
else:
    train_set_splitted = validation_set_splitted = None



## === cell 5
if ImageDataGenerator is not None and tf is not None:
    IMAGE_WIDTH = 224
    IMAGE_HEIGHT = 224
    IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
    BATCH_SIZE = 20

    TrainingImageGenerator = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest",
    )
    ValidationImageGenerator = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    training_dataset = TrainingImageGenerator.flow_from_dataframe(
        train_set_splitted,
        directory=training_images_path,
        seed=9806,
        x_col="image_id",
        y_col="disease_name",
        target_size=IMAGE_SIZE,
        class_mode="categorical",
        interpolation="nearest",
        shuffle=True,
        batch_size=BATCH_SIZE,
        workers=4,
        use_multiprocessing=True,
    )
    validation_dataset = ValidationImageGenerator.flow_from_dataframe(
        validation_set_splitted,
        directory=training_images_path,
        seed=9806,
        x_col="image_id",
        y_col="disease_name",
        target_size=IMAGE_SIZE,
        class_mode="categorical",
        interpolation="nearest",
        shuffle=False,
        batch_size=BATCH_SIZE,
        workers=4,
        use_multiprocessing=True,
    )
else:
    training_dataset = validation_dataset = None



## === cell 6
if tf is not None:
    LOAD_MODEL = False
    IMAGE_WIDTH = 224
    IMAGE_HEIGHT = 224
    NO_OF_CLASSES = 5

    if not LOAD_MODEL:
        CassavaDisease_model = Sequential(name="Cassava_Neural_Network")
        CassavaDisease_model.add(
            VGG19(
                input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3),
                include_top=False,
                weights="imagenet",
            )
        )
        CassavaDisease_model.add(tf.keras.layers.GlobalAveragePooling2D())
        CassavaDisease_model.add(tf.keras.layers.Flatten())
        CassavaDisease_model.add(Dense(256, activation="relu"))
        CassavaDisease_model.add(tf.keras.layers.BatchNormalization())
        CassavaDisease_model.add(Dense(NO_OF_CLASSES, activation="softmax"))
        CassavaDisease_model.compile(
            loss="categorical_crossentropy",
            optimizer=Adam(learning_rate=0.001),
            metrics=["accuracy"],
        )
    else:
        CassavaDisease_model = Sequential()
else:
    CassavaDisease_model = None



## === cell 7
if tf is not None and training_dataset is not None:
    EPOCHES = 1  # minimal training to keep runtime short
    EARLY_STOP = EarlyStopping(
        monitor="val_accuracy", patience=3, restore_best_weights=True
    )
    BEST_MODEL_REACHED = tf.keras.callbacks.ModelCheckpoint(
        filepath="./Cassava_best_Model_Reached_best_model.h5",
        save_best_only=True,
        monitor="val_loss",
        mode="min",
    )
    REDUCE_LR = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6, mode="min", verbose=1
    )
    CassavaDisease_model.fit(
        training_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHES,
        callbacks=[EARLY_STOP, BEST_MODEL_REACHED, REDUCE_LR],
        verbose=1,
    )
else:
    print("Skipping model training due to missing TensorFlow or data generators.")



## === cell 8
TEST_DIR = os.path.join(main_directory, "test_images")
test_images = sorted(os.listdir(TEST_DIR))
print(f"Number of test images: {len(test_images)}")



## === cell 9
if (
    tf is not None
    and CassavaDisease_model is not None
    and hasattr(CassavaDisease_model, "predict")
):
    size = (IMAGE_WIDTH, IMAGE_HEIGHT)
    img_list = []
    for img_name in test_images:
        img_path = os.path.join(TEST_DIR, img_name)
        img = Image.open(img_path).convert("RGB").resize(size)
        img_list.append(np.array(img))
    img_batch = np.stack(img_list, axis=0).astype(np.float32)
    img_batch = preprocess_input(img_batch)
    preds = CassavaDisease_model.predict(img_batch, batch_size=BATCH_SIZE, verbose=0)
    predictions = preds.argmax(axis=1).astype(int).tolist()
else:
    predictions = [int(most_common_label)] * len(test_images)

print(f"Predictions generated: {len(predictions)} items.")



## === cell 10
submission = pd.DataFrame({"image_id": test_images, "label": predictions})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
