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

0.6128739800543971

# 6. Current score

0.13528

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.13528) has done: 'The changes speed up data loading by using multiple workers for the generators, enable parallelism during model fitting, and replace the per‑image prediction loop with a single batched prediction, dramatically reducing I/O and TensorFlow call overhead while keeping the model architecture, training regime, and prediction logic unchanged.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf
import numpy as np
import pandas as pd
import json
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

print("TensorFlow version:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
base_path = "/kaggle/input/cassava-leaf-disease-classification"

with open(os.path.join(base_path, "label_num_to_disease_map.json")) as f:
    classes = json.load(f)

print("Class map:", json.dumps(classes, indent=2))

train_df = pd.read_csv(os.path.join(base_path, "train.csv"))
train_df["class"] = train_df["label"].map({int(i): c for i, c in classes.items()})

print("Training set preview:")
print(train_df.head())



## === cell 2
plt.figure(figsize=(8, 5))
ax = sns.countplot(
    x="class", data=train_df, order=train_df["class"].value_counts().index
)
for p in ax.patches:
    ax.annotate(f"{p.get_height()}", (p.get_x() + 0.3, p.get_height()))
plt.xticks(rotation=45)
plt.title("Class distribution")
plt.show()




## === cell 3
def plot_images(class_id, label, images_number, verbose=0):
    img_ids = (
        train_df[train_df["label"] == class_id]
        .sample(images_number)["image_id"]
        .tolist()
    )
    if verbose:
        print(img_ids)
    size = int(np.ceil(np.sqrt(images_number)))
    plt.figure(figsize=(20, 20))
    for i, img_id in enumerate(img_ids):
        img_path = os.path.join(base_path, "train_images", img_id)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(size, size, i + 1)
        plt.imshow(img)
        plt.title(label, fontsize=12)
        plt.axis("off")
    plt.show()




## === cell 4
TARGET_SIZE = (380, 380)
BATCH_SIZE = 16
STEPS_PER_EPOCH = int(len(train_df) * 0.8 // BATCH_SIZE)
VALIDATION_STEPS = int(len(train_df) * 0.2 // BATCH_SIZE)
EPOCHS = 5  # modest number to fit within resource limits



## === cell 5
train_df["label"] = train_df["label"].astype(str)

train_datagen = ImageDataGenerator(
    validation_split=0.2,
    rotation_range=45,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest",
    height_shift_range=0.2,
    width_shift_range=0.2,
    preprocessing_function=preprocess_input,  # EfficientNet preprocessing
)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    directory=os.path.join(base_path, "train_images"),
    subset="training",
    x_col="image_id",
    y_col="label",
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    seed=2021,
    shuffle=True,
    workers=4,  # <-- parallel workers
    use_multiprocessing=True,
    max_queue_size=10,
)

validation_datagen = ImageDataGenerator(
    validation_split=0.2, preprocessing_function=preprocess_input
)

validation_generator = validation_datagen.flow_from_dataframe(
    train_df,
    directory=os.path.join(base_path, "train_images"),
    subset="validation",
    x_col="image_id",
    y_col="label",
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    seed=2021,
    shuffle=False,
    workers=4,  # <-- parallel workers
    use_multiprocessing=True,
    max_queue_size=10,
)



## === cell 6
base_model = EfficientNetB0(
    weights="imagenet", include_top=False, input_shape=TARGET_SIZE + (3,)
)

x = layers.GlobalAveragePooling2D()(base_model.output)
output = layers.Dense(5, activation="softmax")(x)
model = keras.Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)



## === cell 7
checkpoint_cb = ModelCheckpoint(
    "best_weights.h5", save_best_only=True, monitor="val_loss", mode="min", verbose=1
)
reduce_lr_cb = ReduceLROnPlateau(
    monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6, mode="min", verbose=1
)
early_stop_cb = EarlyStopping(
    monitor="val_loss", patience=3, mode="min", verbose=1, restore_best_weights=True
)

history = model.fit(
    train_generator,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=VALIDATION_STEPS,
    callbacks=[checkpoint_cb, reduce_lr_cb, early_stop_cb],
    verbose=2,
    workers=4,  # <-- parallel workers for fit
    use_multiprocessing=True,
)

model.save("model.h5")



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/649635737.py in <cell line: 0>()
      9 )
     10 
---> 11 history = model.fit(
     12     train_generator,
     13     steps_per_epoch=STEPS_PER_EPOCH,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    117             return fn(*args, **kwargs)
    118         except Exception as e:
--> 119             filtered_tb = _process_traceback_frames(e.__traceback__)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`

TypeError: TensorFlowTrainer.fit() got an unexpected keyword argument 'workers'

## === cell 8
sample_sub_path = os.path.join(base_path, "sample_submission.csv")
submission_df = pd.read_csv(sample_sub_path)

test_dir = os.path.join(base_path, "test_images")

test_images = []
for img_name in submission_df["image_id"]:
    img_path = os.path.join(test_dir, img_name)
    img = Image.open(img_path).convert("RGB")
    img = img.resize(TARGET_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)
    test_images.append(img_array)

test_batch = np.stack(test_images, axis=0)  # shape (N, H, W, 3)

pred_probs = model.predict(test_batch, batch_size=BATCH_SIZE, verbose=0)
pred_classes = np.argmax(pred_probs, axis=1).astype(str)

submission_df["label"] = pred_classes
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")
