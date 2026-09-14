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

# 5. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input, EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import mixed_precision

tf.config.optimizer.set_jit(True)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

if gpus:
    mixed_precision.set_global_policy("mixed_float16")
else:
    mixed_precision.set_global_policy("float32")

tf.random.set_seed(42)
np.random.seed(42)

cpu_cnt = os.cpu_count() or 1
tf.config.threading.set_inter_op_parallelism_threads(cpu_cnt)
tf.config.threading.set_intra_op_parallelism_threads(cpu_cnt)




## === cell 1
label_to_disease = pd.read_json(
    "/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json",
    typ="series",
)
train_csv = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/train.csv")

train_csv["disease"] = train_csv["label"].map(label_to_disease)
train_csv["path"] = (
    "/kaggle/input/cassava-leaf-disease-classification/train_images/"
    + train_csv["image_id"]
)

label_encoder = LabelEncoder()
train_csv["label_enc"] = label_encoder.fit_transform(train_csv["disease"])

train_df, valid_df = train_test_split(
    train_csv,
    test_size=0.2,
    stratify=train_csv["label"],
    random_state=42,
)

BATCH_SIZE = 1024
NUM_WORKERS = min(8, os.cpu_count() or 1)
IMG_SIZE = (224, 224)


def _load_and_preprocess(path, label):
    """Read, decode (fast DCT), resize and preprocess an image."""
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3, dct_method="INTEGER_FAST")
    image = tf.image.resize(image, IMG_SIZE)
    image = preprocess_input(image)  # float32, EfficientNet preprocessing
    return image, tf.one_hot(label, depth=5)


rotation_layer = tf.keras.layers.RandomRotation(0.125, fill_mode="nearest")
translation_layer = tf.keras.layers.RandomTranslation(0.2, 0.2, fill_mode="nearest")
zoom_layer = tf.keras.layers.RandomZoom(0.2, 0.2, fill_mode="nearest")
flip_layer = tf.keras.layers.RandomFlip("horizontal_and_vertical")


def _augment(image, label):
    """Apply the same augmentations as the original ImageDataGenerator."""
    image = rotation_layer(image)
    image = translation_layer(image)
    image = zoom_layer(image)
    image = flip_layer(image)
    return image, label


train_dataset = (
    tf.data.Dataset.from_tensor_slices(
        (train_df["path"].values, train_df["label_enc"].values)
    )
    .map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .cache()  # cache preprocessed images in memory
    .shuffle(buffer_size=1024, seed=42, reshuffle_each_iteration=True)
    .map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

valid_dataset = (
    tf.data.Dataset.from_tensor_slices(
        (valid_df["path"].values, valid_df["label_enc"].values)
    )
    .map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .cache()  # cache in memory
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)




## === cell 2
base_model = EfficientNetB0(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
output = Dense(5, activation="softmax", dtype="float32")(x)

model = Model(inputs=base_model.input, outputs=output)

for layer in base_model.layers:
    layer.trainable = False
for layer in base_model.layers[-20:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

early_stop = EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
)
lr_reduce = ReduceLROnPlateau(
    monitor="val_loss", patience=2, factor=0.5, min_lr=1e-6, verbose=1
)




## === cell 3
epochs = 20
model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=epochs,
    callbacks=[early_stop, lr_reduce],
    verbose=2,
)




## === cell 4
test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
test_files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")])
test_paths = [os.path.join(test_dir, fn) for fn in test_files]

test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(lambda p: tf.io.read_file(p), num_parallel_calls=tf.data.AUTOTUNE)
    .map(
        lambda b: tf.image.decode_jpeg(b, channels=3, dct_method="INTEGER_FAST"),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    .map(
        lambda img: tf.image.resize(img, IMG_SIZE), num_parallel_calls=tf.data.AUTOTUNE
    )
    .map(preprocess_input, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

probs = model.predict(test_dataset, verbose=1)
pred_indices = np.argmax(probs, axis=1)

disease_to_numeric = {v: int(k) for k, v in label_to_disease.items()}
pred_diseases = label_encoder.inverse_transform(pred_indices)
pred_labels = [disease_to_numeric[d] for d in pred_diseases]

submission_df = pd.DataFrame({"image_id": test_files, "label": pred_labels})
submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Submission file created: {submission_path}")
print(submission_df.head())
