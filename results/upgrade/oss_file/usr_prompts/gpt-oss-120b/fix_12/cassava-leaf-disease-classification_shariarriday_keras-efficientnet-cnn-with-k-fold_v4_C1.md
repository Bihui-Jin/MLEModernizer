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
from sklearn.model_selection import StratifiedKFold

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import (
    Input,
    Dense,
    GlobalAveragePooling2D,
    Dropout,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import (
    LearningRateScheduler,
    ModelCheckpoint,
)
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.preprocessing.image import load_img, img_to_array

if tf.config.list_physical_devices("GPU"):
    mixed_policy = tf.keras.mixed_precision.Policy("mixed_float16")
    tf.keras.mixed_precision.set_global_policy(mixed_policy)

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

tf.random.set_seed(42)
np.random.seed(42)

tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

BATCH_SIZE = 256
ROW = 300
COL = 300

train_csv_loc = "../input/cassava-leaf-disease-classification/train.csv"
train_location = "../input/cassava-leaf-disease-classification/train_images/"
test_location = "../input/cassava-leaf-disease-classification/test_images"




## === cell 1
def augmentations(file_path):
    """Simple TensorFlow‑based augmentations (original version, kept for compatibility)."""
    img_bytes = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.resize(img, [ROW, COL])
    img = preprocess_input(img)
    return img




## === cell 2
def fetch_image_without_aug(filename, label):
    img_bytes = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = preprocess_input(img)
    img = tf.image.resize(img, [ROW, COL])
    return img, label




## === cell 3
def augment_tensor(image):
    """Random flips applied to an already decoded & resized image."""
    img = tf.image.random_flip_left_right(image)
    img = tf.image.random_flip_up_down(img)
    img.set_shape([ROW, COL, 3])
    return img




## === cell 4
def getDatasetFromDataframe(train_files, train_labels, val_files, val_labels):
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_ds = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    train_ds = train_ds.shuffle(1000, seed=42, reshuffle_each_iteration=True)
    train_ds = train_ds.map(fetch_image_without_aug, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.cache()  # <-- cache after decoding/resizing
    train_ds = train_ds.map(
        lambda img, lbl: (augment_tensor(img), lbl), num_parallel_calls=AUTOTUNE
    )
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_files, val_labels))
    val_ds = val_ds.map(fetch_image_without_aug, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.cache()
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    options = tf.data.Options()
    options.experimental_deterministic = False
    train_ds = train_ds.with_options(options)
    val_ds = val_ds.with_options(options)

    return train_ds, val_ds




## === cell 5
def scheduler(epoch, lr):
    if epoch < 8:
        return lr
    else:
        return lr * np.exp(-0.05)




## === cell 6
def create_callbacks(folder_name):
    weights_dir = os.path.join("Weights", folder_name)
    logs_dir = os.path.join("Weights", "logs", folder_name)
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    lr_scheduler = LearningRateScheduler(scheduler)

    weight_save = ModelCheckpoint(
        filepath=os.path.join(weights_dir, "best_full.h5"),
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
    )

    weight_save_only = ModelCheckpoint(
        filepath=os.path.join(weights_dir, "best_weights.weights.h5"),
        monitor="val_accuracy",
        verbose=0,
        save_best_only=True,
        save_weights_only=True,
    )

    callbacks = [lr_scheduler, weight_save, weight_save_only]
    return callbacks




## === cell 7
def create_model(training=True, weights="imagenet"):
    base_model = EfficientNetB3(
        weights=weights, include_top=False, input_shape=(ROW, COL, 3)
    )
    base_model.trainable = False
    inputs = Input(shape=(ROW, COL, 3))
    x = base_model(inputs, training=training)
    x = GlobalAveragePooling2D(name="encoding")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(5, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model




## === cell 8
train_csv = pd.read_csv(train_csv_loc)
train_csv["image_id"] = train_csv["image_id"].map(
    lambda x: os.path.join(train_location, x)
)

create_k_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_number = 1
Models = []  # store trained models for inference

for train_idx, val_idx in create_k_folds.split(
    train_csv["image_id"], train_csv["label"]
):
    print(f"\n=== Fold {fold_number} ===")
    folder_name = f"Exp_{fold_number}"
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
            train_ds, epochs=15, validation_data=val_ds, callbacks=callbacks, verbose=2
        )
    except Exception as e:
        print("Training error:", e)

    best_path = os.path.join("Weights", folder_name, "best_full.h5")
    if os.path.exists(best_path):
        model.load_weights(best_path)
    Models.append(model)

    tf.keras.backend.clear_session()
    fold_number += 1



## === cell 9
test_files = [
    os.path.join(test_location, f)
    for f in os.listdir(test_location)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
    and os.path.isfile(os.path.join(test_location, f))
]

AUTOTUNE = tf.data.experimental.AUTOTUNE


def load_and_preprocess(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [ROW, COL])
    img = preprocess_input(img)
    return img


test_ds = tf.data.Dataset.from_tensor_slices(test_files)
test_ds = test_ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

preds_per_model = []
for m in Models:
    preds = m.predict(test_ds, verbose=0)
    preds_per_model.append(preds)

avg_preds = np.mean(preds_per_model, axis=0)

results = []
for img_path, pred in zip(test_files, avg_preds):
    label = int(np.argmax(pred))
    results.append({"image_id": os.path.basename(img_path), "label": label})



## === cell 10
submission = pd.DataFrame(results)
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path} with {len(submission)} rows.")
