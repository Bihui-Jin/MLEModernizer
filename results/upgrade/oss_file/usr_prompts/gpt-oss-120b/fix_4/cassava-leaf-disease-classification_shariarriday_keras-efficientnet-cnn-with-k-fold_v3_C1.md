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

0.8559987911755818

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import cast, float32, numpy_function
from tensorflow.data import Dataset
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (
    LearningRateScheduler,
    ModelCheckpoint,
    TensorBoard,
)
from tensorflow.keras.layers import (
    Conv2D,
    Input,
    Dense,
    GlobalAveragePooling2D,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import StratifiedKFold

BATCH_SIZE = 64
ROW = 224
COL = 224

train_csv_loc = "../input/cassava-leaf-disease-classification/train.csv"
train_location = "../input/cassava-leaf-disease-classification/train_images/"
test_location = "../input/cassava-leaf-disease-classification/test_images"

if os.path.isdir(os.path.join(test_location, "test_images")):
    test_location = os.path.join(test_location, "test_images")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def augmentations(file):
    file = tf.io.read_file(file)
    img = tf.image.decode_jpeg(file, channels=3)
    img = tf.cast(img, tf.float32)

    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)

    img = tf.image.resize(img, [ROW, COL])
    img = preprocess_input(img)
    return img




## === cell 2
def fetch_image_without_aug(filename, label):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [ROW, COL])
    img = preprocess_input(img)
    return img, label




## === cell 3
def fetch_image_with_aug(filename, label):
    aug_img = numpy_function(func=augmentations, inp=[filename], Tout=float32)
    return aug_img, label




## === cell 4
def getDatasetFromDataframe(train_files, train_labels, val_files, val_labels):
    train_ds = Dataset.from_tensor_slices((train_files, train_labels))
    train_ds = train_ds.shuffle(len(train_files))
    train_ds = train_ds.map(fetch_image_with_aug, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_ds = Dataset.from_tensor_slices((val_files, val_labels))
    val_ds = val_ds.shuffle(len(val_files))
    val_ds = val_ds.map(fetch_image_without_aug, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds




## === cell 5
def scheduler(epoch, lr):
    if epoch < 8:
        return lr
    else:
        return lr * np.exp(-0.05)




## === cell 6
def create_callbacks(folder_name):
    base_path = os.path.join("Weights")
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(os.path.join(base_path, "logs", folder_name), exist_ok=True)

    lr_scheduler = LearningRateScheduler(scheduler)

    weight_save = ModelCheckpoint(
        filepath=os.path.join(base_path, f"{folder_name}.h5"),
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
    )

    weight_save_only = ModelCheckpoint(
        filepath=os.path.join(base_path, f"{folder_name}_weights.weights.h5"),
        monitor="val_accuracy",
        verbose=0,
        save_best_only=True,
        save_weights_only=True,
    )

    tensorboard = TensorBoard(
        log_dir=os.path.join(base_path, "logs", folder_name), histogram_freq=1
    )

    callbacks = [lr_scheduler, weight_save, tensorboard, weight_save_only]
    return callbacks




## === cell 7
def create_model(training=True, weights="imagenet"):
    base_model = EfficientNetB0(
        weights=weights, include_top=False, input_shape=(ROW, COL, 3)
    )
    x = Input(shape=(ROW, COL, 3))
    out = base_model(x, training=training)
    out = GlobalAveragePooling2D(name="encoding")(out)
    out = Dropout(0.5)(out)
    output = Dense(5, activation="softmax")(out)
    model = Model(inputs=x, outputs=output)
    return model




## === cell 8
train_csv = pd.read_csv(train_csv_loc)
train_csv["image_id"] = train_csv["image_id"].map(
    lambda x: os.path.join(train_location, x)
)

Folder_name_base = "Exp_"
create_k_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_number = 1
trained_models = []

for train_idx, val_idx in create_k_folds.split(
    train_csv["image_id"], train_csv["label"]
):
    print(f"Fold Number {fold_number} is starting its training")
    folder_name = f"{Folder_name_base}{fold_number}"
    callbacks = create_callbacks(folder_name)

    X_train = train_csv["image_id"].iloc[train_idx].values
    y_train = train_csv["label"].iloc[train_idx].values
    X_val = train_csv["image_id"].iloc[val_idx].values
    y_val = train_csv["label"].iloc[val_idx].values

    train_ds, val_ds = getDatasetFromDataframe(X_train, y_train, X_val, y_val)

    model = create_model()
    model.compile(
        optimizer=RMSprop(learning_rate=1e-4),
        loss=SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    try:
        model.fit(
            train_ds,
            epochs=15,
            validation_data=val_ds,
            callbacks=callbacks,
            workers=4,
            use_multiprocessing=False,
        )
    except Exception as e:
        print("Training error:", e)

    trained_models.append(model)
    fold_number += 1




## === cell 9
weight_files = glob.glob(os.path.join("Weights", "Exp_*.h5"))
models = []
for wf in weight_files:
    m = create_model(training=False, weights=None)
    m.load_weights(wf)
    models.append(m)

if not models:
    models = trained_models

valid_exts = (".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG")
images = [f for f in os.listdir(test_location) if f.lower().endswith(valid_exts)]
results = []

for img_name in images:
    img_path = os.path.join(test_location, img_name)
    img = load_img(img_path, target_size=(ROW, COL))
    img_arr = img_to_array(img)
    img_arr = preprocess_input(img_arr)
    batch = np.expand_dims(img_arr, axis=0)

    preds = [m.predict(batch, verbose=0) for m in models]
    avg_pred = np.mean(preds, axis=0)
    label = int(np.argmax(avg_pred, axis=1)[0])
    results.append({"image_id": img_name, "label": label})




## === cell 10
submission = pd.DataFrame(results)
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}, rows: {len(submission)}")

## --- ERROR in outputing the csv:
Invalid submission: Submission must have the same length as the answers.
