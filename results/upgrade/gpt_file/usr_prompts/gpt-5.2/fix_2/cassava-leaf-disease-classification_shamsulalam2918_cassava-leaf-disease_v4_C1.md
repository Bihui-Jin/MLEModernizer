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

INPUT_DIR = "/kaggle/input/cassava-leaf-disease-classification"
WORKING_DIR = "/kaggle/working"
os.makedirs(WORKING_DIR, exist_ok=True)

print("INPUT_DIR exists:", os.path.exists(INPUT_DIR))
print("Listing input root:")
print(os.listdir("/kaggle/input")[:10])



## === cell 1
import numpy as np
import pandas as pd

for dirname, _, filenames in os.walk(
    "/kaggle/input/cassava-leaf-disease-classification"
):
    if "train.csv" in filenames or "sample_submission.csv" in filenames:
        print("Found in:", dirname)
        print(
            "  files:",
            [f for f in filenames if f.endswith(".csv") or f.endswith(".json")][:10],
        )
        break



## === cell 2
import glob
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    Callback,
    ReduceLROnPlateau,
    ModelCheckpoint,
    TensorBoard,
)
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras import Input, Model
from tensorflow.keras.applications import InceptionResNetV2

plt.rcParams["figure.figsize"] = (17, 6)

print("TensorFlow:", tf.__version__)



## === cell 3
TRAINING_DIR = os.path.join(INPUT_DIR, "train_images")
TRAINING_CSV = os.path.join(INPUT_DIR, "train.csv")
JSON_LABELS = os.path.join(INPUT_DIR, "label_num_to_disease_map.json")
TEST_DIR = os.path.join(INPUT_DIR, "test_images")
SAMPLE_SUB = os.path.join(INPUT_DIR, "sample_submission.csv")

assert os.path.exists(TRAINING_DIR), TRAINING_DIR
assert os.path.exists(TRAINING_CSV), TRAINING_CSV
assert os.path.exists(TEST_DIR), TEST_DIR
assert os.path.exists(SAMPLE_SUB), SAMPLE_SUB



## === cell 4
train_df = pd.read_csv(TRAINING_CSV)
train_df["label"] = train_df["label"].astype("string")  # for flow_from_dataframe
train_df.head()



## === cell 5
total_images_count = len(train_df.index)
total_train_img_count = int(total_images_count * 0.8)
total_val_img_count = total_images_count - total_train_img_count

print("Expected images counts:")
print(f"Total Images from original directory: {total_images_count}")
print(f"Training Images: {total_train_img_count}")
print(f"Validation Images: {total_val_img_count}")



## === cell 6
label_df = pd.read_json(JSON_LABELS, orient="index")
label_names = label_df.values.flatten().tolist()
print("Label names:", label_names)



## === cell 7
training_images = glob.glob(os.path.join(TRAINING_DIR, "*.jpg"))
print("Num training jpgs:", len(training_images))

plt.figure(figsize=(12, 12))
for i in range(1, 10):
    training_image = np.random.choice(training_images)
    img_rgb = cv2.imread(training_image)[..., ::-1]
    plt.subplot(3, 3, i)
    plt.imshow(img_rgb)
    plt.axis("off")
plt.show()



## === cell 8
training_datagen = ImageDataGenerator(
    rescale=1 / 255,
    rotation_range=100,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    brightness_range=[0.7, 1.4],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest",
    validation_split=0.2,
)

validation_datagen = ImageDataGenerator(
    rescale=1 / 255,
    validation_split=0.2,
)



## === cell 9
BATCH_SIZE = 24
IMG_WIDTH = 300
IMG_HEIGHT = 300
CHANNEL = 3

print("\nTraining Dataset")
train_ds = training_datagen.flow_from_dataframe(
    train_df,
    TRAINING_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    x_col="image_id",
    y_col="label",
    shuffle=True,
    subset="training",
)

print("\nValidation Dataset")
validation_ds = validation_datagen.flow_from_dataframe(
    train_df,
    TRAINING_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    x_col="image_id",
    y_col="label",
    shuffle=False,
    subset="validation",
)

print("\nClass Indices:")
print(train_ds.class_indices)




## === cell 10
class theCallBacks(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if (logs.get("val_accuracy", 0) > 0.92) and (logs.get("accuracy", 0) > 0.92):
            print(
                "\nTraining Accuracy > 0.92 & Validation Accuracy > 0.92\nCancelling training!"
            )
            self.model.stop_training = True


callback_on_metrics = theCallBacks()

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    verbose=1,
    cooldown=1,
    min_lr=0.0001,
)



## === cell 11
import datetime


class LearningRateLogger(Callback):
    def __init__(self):
        super().__init__()
        self._supports_tf_logs = True

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if "learning_rate" in logs:
            return
        lr = getattr(self.model.optimizer, "learning_rate", None)
        try:
            logs["learning_rate"] = float(tf.keras.backend.get_value(lr))
        except Exception:
            pass


log_dir = os.path.join(
    WORKING_DIR, "logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
)
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=0)



## === cell 12
new_input = Input(shape=(IMG_WIDTH, IMG_HEIGHT, CHANNEL))

base_model = InceptionResNetV2(
    include_top=False,
    weights="imagenet",
    input_tensor=new_input,
    pooling="avg",
)

for layer in base_model.layers:
    layer.trainable = False

x = Flatten(name="flatten")(base_model.output)
x = Dropout(0.5)(x)
x = Dense(4096, activation="relu", name="fc6")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu", name="fc7")(x)
x = Dropout(0.5)(x)
out = Dense(5, activation="softmax", name="classifier")(x)

model = Model(inputs=base_model.input, outputs=out)

sgd = tf.keras.optimizers.SGD(
    learning_rate=0.01, momentum=0.9, decay=0.0001, nesterov=True
)
model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()



## === cell 13
num_epochs = 10
steps_per_epoch = max(1, total_train_img_count // BATCH_SIZE)
validation_steps = max(1, total_val_img_count // BATCH_SIZE)

model_checkpoint_path = os.path.join(WORKING_DIR, "cassava_best.keras")
checkpoint = ModelCheckpoint(
    filepath=model_checkpoint_path,
    monitor="val_accuracy",
    verbose=1,
    save_best_only=True,
    mode="max",
)

history = model.fit(
    train_ds,
    epochs=num_epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_ds,
    validation_steps=validation_steps,
    callbacks=[
        checkpoint,
        reduce_lr,
        callback_on_metrics,
        tensorboard_callback,
        LearningRateLogger(),
    ],
    verbose=1,
)

print("Best model saved to:", model_checkpoint_path)



## === cell 14
best_model = tf.keras.models.load_model(model_checkpoint_path, compile=False)

sample_submission = pd.read_csv(SAMPLE_SUB)
assert {"image_id", "label"}.issubset(sample_submission.columns)

predicted = []
for image_id in sample_submission["image_id"].values:
    img_path = os.path.join(TEST_DIR, image_id)
    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.preprocessing.image.smart_resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = tf.reshape(img, (-1, IMG_WIDTH, IMG_HEIGHT, CHANNEL))
    probs = best_model.predict(img / 255.0, verbose=0)
    predicted.append(int(np.argmax(probs, axis=1)[0]))

submission = pd.DataFrame(
    {"image_id": sample_submission["image_id"], "label": predicted}
)
submission_path = os.path.join(WORKING_DIR, "submission.csv")
submission.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(submission.head())
print("Rows:", len(submission))
assert submission_path.endswith(".csv") and os.path.exists(submission_path)
