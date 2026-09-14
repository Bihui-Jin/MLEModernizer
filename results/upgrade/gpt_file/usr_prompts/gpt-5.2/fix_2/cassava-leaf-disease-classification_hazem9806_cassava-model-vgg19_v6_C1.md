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
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))



## === cell 1
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Flatten,
    BatchNormalization,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import VGG19



## === cell 2
main_directory = "/kaggle/input/cassava-leaf-disease-classification/"
if not os.path.exists(main_directory):
    main_directory = "/kaggle/data/cassava-leaf-disease-classification/"
if not os.path.exists(main_directory):
    main_directory = "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/"

training_images_path = os.path.join(main_directory, "train_images")
test_images_path = os.path.join(main_directory, "test_images")

print("main_directory:", main_directory)
print("train_images exists:", os.path.exists(training_images_path))
print("test_images exists:", os.path.exists(test_images_path))
print("List of Files in main_directory:\n", os.listdir(main_directory)[:20])



## === cell 3
image_label_data = pd.read_csv(os.path.join(main_directory, "train.csv"))
labels = pd.read_json(
    os.path.join(main_directory, "label_num_to_disease_map.json"), typ="series"
)

print("Cassava Leaf Disease Classification Labels are:\n", dict(labels))
print(image_label_data.head())



## === cell 4
print(Counter(image_label_data["label"]))
print(image_label_data["label"].value_counts(normalize=True))



## === cell 5
image_label_data["disease_name"] = image_label_data.label.map(labels)
print(image_label_data.head())



## === cell 6
from sklearn.model_selection import train_test_split

TEST_PERCENTAGE = 0.05
train_set_splitted, validation_set_splitted = train_test_split(
    image_label_data,
    test_size=TEST_PERCENTAGE,
    random_state=42,
    stratify=image_label_data["disease_name"],
)

print(
    "Train size:", len(train_set_splitted), "Valid size:", len(validation_set_splitted)
)



## === cell 7
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
NO_OF_CLASSES = 5
BATCH_SIZE = 20

TrainingImageGenerator = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg19.preprocess_input,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest",
)

ValidatonImageGenerator = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg19.preprocess_input
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
)

validation_dataset = ValidatonImageGenerator.flow_from_dataframe(
    validation_set_splitted,
    directory=training_images_path,
    seed=9806,
    x_col="image_id",
    y_col="disease_name",
    target_size=IMAGE_SIZE,
    class_mode="categorical",
    interpolation="nearest",
    shuffle=True,
    batch_size=BATCH_SIZE,
)

print("Class indices:", training_dataset.class_indices)



## === cell 8
LOAD_MODEL = True
external_model_path = "../input/84-percentage-model/Cassava_best_Model_Reached_best_model_7_Jan_04_acc_is_86.h5"

CassaveDisease_model = None
if LOAD_MODEL and os.path.exists(external_model_path):
    CassaveDisease_model = tf.keras.models.load_model(external_model_path)
    print("External model loaded successfully:", external_model_path)
else:
    if LOAD_MODEL:
        print(
            "External model not found; building VGG19 model from scratch and training instead."
        )
    CassaveDisease_model = Sequential(name="Cassava_Neural_Network")
    CassaveDisease_model.add(
        VGG19(
            input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3),
            include_top=False,
            weights="imagenet",
        )
    )
    CassaveDisease_model.add(GlobalAveragePooling2D())
    CassaveDisease_model.add(Flatten())
    CassaveDisease_model.add(Dense(256, activation="relu"))
    CassaveDisease_model.add(BatchNormalization())
    CassaveDisease_model.add(Dense(NO_OF_CLASSES, activation="softmax"))

Adam_Optimizer = Adam(learning_rate=0.001)
CassaveDisease_model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam_Optimizer,
    metrics=["accuracy"],
)

CassaveDisease_model.summary()



## === cell 9
EPOCHES = 2

EPOCHES_TO_WAIT_WITH_NO_IMPORVEMENT = 3
EARLY_STOP = EarlyStopping(
    monitor="val_accuracy",
    patience=EPOCHES_TO_WAIT_WITH_NO_IMPORVEMENT,
    restore_best_weights=True,
)

BEST_MODEL_REACHED = ModelCheckpoint(
    filepath="./Cassava_best_Model_Reached_best_model_8_Jan_03.h5",
    save_best_only=True,
    monitor="val_loss",
    mode="min",
)

REDUCE_LR = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    min_lr=1e-6,
    mode="min",
    verbose=1,
)

Trained_Model = CassaveDisease_model.fit(
    training_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHES,
    callbacks=[EARLY_STOP, BEST_MODEL_REACHED, REDUCE_LR],
    verbose=1,
)

CassaveDisease_model.save("./Cassava_best_Model_Reached_8_jan_03.h5")
print("Model Saved Successfully!")



## === cell 10
print(Trained_Model.history.keys())




## === cell 11
def Train_Val_Plot(acc, val_acc, loss, val_loss):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
    fig.suptitle(" MODEL'S METRICS VISUALIZATION ", fontsize=20)

    ax1.plot(range(1, len(acc) + 1), acc)
    ax1.plot(range(1, len(val_acc) + 1), val_acc)
    ax1.set_title("History of Accuracy", fontsize=15)
    ax1.set_xlabel("Epochs", fontsize=15)
    ax1.set_ylabel("Accuracy", fontsize=15)
    ax1.legend(["training", "validation"])

    ax2.plot(range(1, len(loss) + 1), loss)
    ax2.plot(range(1, len(val_loss) + 1), val_loss)
    ax2.set_title("History of Loss", fontsize=15)
    ax2.set_xlabel("Epochs", fontsize=15)
    ax2.set_ylabel("Loss", fontsize=15)
    ax2.legend(["training", "validation"])
    plt.show()




## === cell 12
if "accuracy" in Trained_Model.history and len(Trained_Model.history["accuracy"]) > 0:
    Train_Val_Plot(
        Trained_Model.history.get("accuracy", []),
        Trained_Model.history.get("val_accuracy", []),
        Trained_Model.history.get("loss", []),
        Trained_Model.history.get("val_loss", []),
    )



## === cell 13
sample_sub_path = os.path.join(main_directory, "sample_submission.csv")
sample_sub = pd.read_csv(sample_sub_path)
test_images = sample_sub["image_id"].tolist()

predictions = []
size = (IMAGE_WIDTH, IMAGE_HEIGHT)

for image_name in test_images:
    img_path = os.path.join(test_images_path, image_name)
    img = Image.open(img_path).convert("RGB")
    img = img.resize(size)
    img_arr = np.array(img, dtype=np.float32)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = tf.keras.applications.vgg19.preprocess_input(img_arr)
    pred = CassaveDisease_model.predict(img_arr, verbose=0).argmax(axis=1)[0]
    predictions.append(int(pred))

print("Num test images:", len(test_images), "Num predictions:", len(predictions))
print("First 10 preds:", predictions[:10])



## === cell 14
sub = pd.DataFrame({"image_id": test_images, "label": predictions})
print(sub.head())
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
