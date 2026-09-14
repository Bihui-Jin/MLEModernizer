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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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

0.8856149894227864

# 6. Current score

0.33894

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I replace the broken imports and missing model file with a fresh EfficientNet‑B0 model built from TensorFlow‑Keras, create a lightweight training pipeline using the provided JPEG images, train it for a few epochs, and then generate predictions for the test set. The script now reads the train and test CSVs, builds TensorFlow datasets from the image folders, trains the model, and finally writes a correctly‑formatted `submission.csv` file, fixing all previous NameError and file‑not‑found issues while keeping the core EfficientNet architecture.'
- What this solution (achieved 0.29821) has done: 'The fix adds a protobuf compatibility flag before importing TensorFlow to stop the `MessageFactory` attribute error, reduces the image size to the standard 224×224 for EfficientNetB0, unfreezes the base model and uses a smaller learning‑rate, and trains for more epochs to raise validation accuracy toward the target. All other logic and file handling remain unchanged.'
- What this solution (achieved 0.61286) has done: 'The changes enable TensorFlow’s XLA JIT compiler and set optimal threading, which speeds up graph execution without altering the model, data pipeline, or training logic. These adjustments keep the same architecture, epochs, and preprocessing while providing a noticeable runtime reduction, allowing the script to complete within the 600‑second limit.'
- What this solution (achieved 0.44843) has done: 'I remove the redundant second TensorFlow import that triggers the protobuf error, add a few extra augmentations, freeze the EfficientNet backbone for an initial warm‑up phase, then unfreeze it for fine‑tuning with a lower learning rate and class‑weighted loss. These minimal changes keep the original architecture while improving validation accuracy, moving the score nearer the target.'
- What this solution (achieved 0.41405) has done: 'The fix moves the protobuf‑compatibility flag to the very start, wraps the TensorFlow import in a fallback that disables TF‑v2 if the protobuf error occurs, and keeps the rest of the pipeline unchanged. This resolves the import crash while preserving the original EfficientNet‑B0 model and training logic, letting the script run end‑to‑end and produce a correct `submission.csv` that can be scored toward the target.'
- What this solution (achieved 0.33894) has done: 'The update speeds up the pipeline by increasing the batch size (halving the number of training steps per epoch) and moving data directly onto the GPU with `prefetch_to_device`, which reduces host‑GPU transfer overhead. All other model definitions, training epochs, loss, optimizer, and data‑augmentation layers stay exactly the same, preserving the original learning behavior and final accuracy.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()

try:
    tf.config.optimizer.set_jit(True)  # activate XLA
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.efficientnet import preprocess_input

tf.random.set_seed(42)

if tf.config.list_physical_devices("GPU"):
    from tensorflow.keras import mixed_precision

    mixed_precision.set_global_policy("mixed_float16")

BASE_PATH = "/kaggle/input/cassava-leaf-disease-classification"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 256
NUM_CLASSES = 5
AUTOTUNE = tf.data.AUTOTUNE


def image_path(folder, image_id):
    return os.path.join(BASE_PATH, folder, image_id)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_df = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))
train_df["label"] = train_df["label"].astype(int)
train_df["image_path"] = train_df["image_id"].apply(
    lambda x: image_path("train_images", x)
)

train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_df["image_path"].values,
    train_df["label"].values,
    test_size=0.1,
    stratify=train_df["label"],
    random_state=42,
)


def _parse_image(filename, label):
    img_raw = tf.io.read_file(filename)
    img = tf.io.decode_jpeg(img_raw, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)  # EfficientNet preprocessing
    label_onehot = tf.one_hot(label, NUM_CLASSES)
    return img, label_onehot


def make_dataset(paths, labels, ordered=False):
    """
    Build a tf.data pipeline with in‑memory caching and GPU prefetch.
    - For training (ordered=False) we shuffle after caching; for validation we keep order.
    """
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(_parse_image, num_parallel_calls=AUTOTUNE, deterministic=False)
    ds = ds.cache()  # cache processed images in RAM
    if not ordered:
        ds = ds.shuffle(1024)  # shuffle only for training
    ds = ds.batch(BATCH_SIZE)
    ds = ds.apply(tf.data.experimental.prefetch_to_device("/GPU:0"))
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_dataset(
    train_paths,
    train_labels,
    ordered=False,
)

val_ds = make_dataset(
    val_paths,
    val_labels,
    ordered=True,  # preserve order for validation
)




## === cell 2
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(*IMAGE_SIZE, 3)
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
x = tf.keras.layers.RandomFlip(mode="horizontal")(inputs)
x = tf.keras.layers.RandomRotation(factor=0.2)(x)
x = tf.keras.layers.RandomContrast(factor=0.2)(x)
x = tf.keras.layers.RandomZoom(height_factor=0.2, width_factor=0.2)(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

unique, counts = np.unique(train_labels, return_counts=True)
class_weights = {
    int(u): float(len(train_labels) / (NUM_CLASSES * c)) for u, c in zip(unique, counts)
}

lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_accuracy",
    factor=0.5,
    patience=3,
    verbose=1,
    min_lr=1e-6,
)

ckpt_path = "best_model.h5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    verbose=2,
    callbacks=[lr_reduce, checkpoint],
    class_weight=class_weights,
)

base_model.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=35,
    verbose=2,
    callbacks=[lr_reduce, checkpoint],
    class_weight=class_weights,
)

model.load_weights(ckpt_path)




## === cell 3
test_df = pd.read_csv(os.path.join(BASE_PATH, "sample_submission.csv"))
test_paths = test_df["image_id"].apply(lambda x: image_path("test_images", x)).values


def _parse_test_image(filename):
    img_raw = tf.io.read_file(filename)
    img = tf.io.decode_jpeg(img_raw, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img


test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = test_ds.map(
    _parse_test_image, num_parallel_calls=AUTOTUNE, deterministic=False
)
test_ds = test_ds.cache()  # cache test set in memory (small enough)
test_ds = test_ds.batch(BATCH_SIZE)
test_ds = test_ds.apply(tf.data.experimental.prefetch_to_device("/GPU:0"))
test_ds = test_ds.prefetch(AUTOTUNE)

probabilities = model.predict(test_ds, verbose=0)
predictions = np.argmax(probabilities, axis=-1)

submission = pd.DataFrame({"image_id": test_df["image_id"], "label": predictions})
submission.to_csv("submission.csv", index=False)
print("submission.csv written –", submission.shape[0], "rows")
