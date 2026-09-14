# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

3.13

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

0.8859171955273496

# 6. Current score

0.73281

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I replace the failing model‑loading and prediction code with a simple, robust baseline that reads the training labels, determines the most frequent class, and assigns that class to every test image listed in the provided sample_submission.csv. This eliminates missing‑file errors, ensures the length of predictions matches the test set, and guarantees a valid submission.csv is written.'
- What this solution (achieved 0.70254) has done: 'I replace the constant‑majority prediction with a small image classifier built on TensorFlow’s MobileNetV2 backbone. The model is trained for a few quick epochs on the training images (resized 128×128) and then used to predict the test set, producing a valid submission.csv. This change is allowed because the current score is >30 % away from the target, and the new approach should raise accuracy toward the desired 0.886 while still finishing within the runtime limit.'
- What this solution (achieved 0.70254) has done: 'The fix adds a compatibility shim for the protobuf MessageFactory before importing TensorFlow, which prevents the `'MessageFactory' object has no attribute 'GetPrototype'` error that stopped the notebook from running. No other logic is changed, so the MobileNetV2 model, training, and prediction pipeline remain intact, allowing the score to improve toward the target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.59828) has done: 'Implemented a protobuf compatibility fix by setting the appropriate environment variable before importing TensorFlow and simplified the shim. Added a two‑stage training schedule: first train the frozen MobileNetV2 backbone for a few epochs, then unfreeze it and fine‑tune with a lower learning rate for additional epochs. This boosts model capacity and pushes validation accuracy nearer the target while keeping the original architecture and data pipeline intact.'
- What this solution (achieved 0.75075) has done: 'The fix adds a robust protobuf shim that safely assigns `GetPrototype` only when the alternate `GetMessageClass` exists, preventing the import‑time AttributeError.  
Training hyper‑parameters are modestly increased (more epochs and a slightly higher initial learning rate) and a simple augmentation (random horizontal flip) is added to the preprocessing pipeline to improve model accuracy while keeping the original architecture unchanged. The script now runs end‑to‑end and writes a valid `submission.csv`.'
- What this solution (achieved 0.7429) has done: 'I add dataset caching to avoid repeatedly decoding the same images each epoch and increase the batch size (64) to halve the number of training steps per epoch. Caching after the image‑preprocessing map stores the decoded tensors in memory, which drastically reduces I/O overhead while preserving exactly the same data and model logic. The larger batch size speeds up GPU utilization without altering the architecture, loss, or training schedule, so the final predictions remain unchanged.'
- What this solution (achieved 0.71039) has done: 'Implemented a robust protobuf shim to prevent the `MessageFactory` attribute error before importing TensorFlow, and modestly extended the training schedule (initial 20 epochs, fine‑tune 10 epochs) to gain a small accuracy boost while preserving the original model architecture and pipeline. The script now runs end‑to‑end and writes a valid `submission.csv`.'
- What this solution (achieved 0.77093) has done: 'Implemented a robust base‑directory lookup so the script works whether the data lives under `data/`, `input/` or the typical Kaggle `/kaggle/input/` path. Added a fallback that raises a clear error if none are found. Increased training epochs (10 + 5 fine‑tune) to improve accuracy while keeping the original MobileNetV2 architecture and preprocessing unchanged. All other logic remains the same, and the script now reliably writes a valid `submission.csv`.'
- What this solution (achieved 0.7799) has done: 'Implemented a protobuf compatibility shim before importing TensorFlow to prevent the `MessageFactory` AttributeError. Adjusted the training schedule by extending the initial training to 15 epochs and fine‑tuning to 10 epochs, giving the model more learning capacity while retaining the original architecture and preprocessing logic.'
- What this solution (achieved 0.73281) has done: 'Implemented two modest refinements to boost validation accuracy while keeping the original architecture unchanged:  
1. Computed class‑frequency based weights and supplied them via the `class_weight` argument during both the frozen‑base and fine‑tuning training phases to mitigate label imbalance.  
2. Extended the training schedule slightly (initial 20 epochs, fine‑tune 15 epochs) to give the model more learning opportunity without altering the core model definition.  

These changes are minimal, preserve the existing pipeline, and are aimed at moving the score closer to the target.'

# 9. Code solution

## === cell 0
import os
import pathlib
import pandas as pd
import numpy as np

try:
    from google.protobuf import message_factory

    if not hasattr(message_factory.MessageFactory, "GetPrototype"):

        def _get_prototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        message_factory.MessageFactory.GetPrototype = _get_prototype
except Exception:
    pass

possible_dirs = [
    "data/cassava-leaf-disease-classification",
    "input/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification",
]
BASE_DIR = None
for d in possible_dirs:
    if pathlib.Path(d).exists():
        BASE_DIR = pathlib.Path(d)
        break
if BASE_DIR is None:
    raise RuntimeError(
        "Could not locate cassava-leaf-disease-classification data directory. "
        f"Tried: {possible_dirs}"
    )

TRAIN_CSV = BASE_DIR / "train.csv"
TEST_IMAGES_DIR = BASE_DIR / "test_images"
TRAIN_IMAGES_DIR = BASE_DIR / "train_images"
SAMPLE_SUBMISSION = BASE_DIR / "sample_submission.csv"
SUBMISSION_PATH = pathlib.Path("submission.csv")

train_df = pd.read_csv(TRAIN_CSV)
if (
    train_df.empty
    or "label" not in train_df.columns
    or "image_id" not in train_df.columns
):
    raise RuntimeError("train.csv must contain 'image_id' and 'label' columns.")

train_df["filepath"] = train_df["image_id"].apply(lambda x: str(TRAIN_IMAGES_DIR / x))

test_sub = pd.read_csv(SAMPLE_SUBMISSION)
if test_sub.empty or "image_id" not in test_sub.columns:
    raise RuntimeError("sample_submission.csv must contain 'image_id' column.")
test_image_paths = (
    test_sub["image_id"].apply(lambda x: str(TEST_IMAGES_DIR / x)).tolist()
)

NUM_CLASSES = train_df["label"].nunique()
IMG_SIZE = 224  # MobileNetV2 native resolution
BATCH_SIZE = 32

import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE




## === cell 1
def preprocess_image(path, label=None):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    if label is not None:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    if label is None:
        return img
    else:
        return img, tf.one_hot(label, NUM_CLASSES)


train_paths = train_df["filepath"].values
train_labels = train_df["label"].values.astype(np.int32)

train_ds_full = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_ds_full = train_ds_full.map(preprocess_image, num_parallel_calls=AUTOTUNE)
train_ds_full = train_ds_full.cache()
train_ds_full = train_ds_full.shuffle(
    buffer_size=1000, seed=42, reshuffle_each_iteration=False
)

val_size = int(0.1 * len(train_paths))
val_ds = train_ds_full.take(val_size).batch(BATCH_SIZE).prefetch(AUTOTUNE)
train_ds = (
    train_ds_full.skip(val_size)
    .shuffle(1000, seed=42)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)




## === cell 2
class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
total_samples = len(train_labels)
class_weights = {
    i: total_samples / (NUM_CLASSES * count)
    for i, count in enumerate(class_counts)
    if count > 0
}

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False  # freeze base initially

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")

model = tf.keras.Sequential([base_model, global_average_layer, prediction_layer])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"],
)

EPOCHS = 20  # extended initial training
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    verbose=2,
)

base_model.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"],
)

FINE_TUNE_EPOCHS = 15  # extended fine‑tuning
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    class_weight=class_weights,
    verbose=2,
)




## === cell 3
test_ds = tf.data.Dataset.from_tensor_slices(test_image_paths)
test_ds = test_ds.map(lambda x: preprocess_image(x), num_parallel_calls=AUTOTUNE)
test_ds = test_ds.cache()
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

pred_probs = model.predict(test_ds, verbose=0)
pred_labels = np.argmax(pred_probs, axis=1)

submission = pd.DataFrame({"image_id": test_sub["image_id"], "label": pred_labels})
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission written to {SUBMISSION_PATH} with {len(submission)} rows.")
