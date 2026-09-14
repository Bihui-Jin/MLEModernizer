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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
try:
    from google.protobuf.message_factory import MessageFactory

    if not hasattr(MessageFactory, "GetPrototype"):
        MessageFactory.GetPrototype = lambda self, descriptor: self.GetMessageClass(
            descriptor
        )
except Exception:
    pass  # If protobuf is not available, TensorFlow will raise its own error later

import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.applications import EfficientNetB4
from tqdm import tqdm
import math

tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)

if tf.config.list_physical_devices("GPU"):
    from tensorflow.keras import mixed_precision

    mixed_precision.set_global_policy("mixed_float16")




## === cell 1
IMAGE_SIZE = 380
CLASSES = 5
SEED = 99
BATCH_SIZE = 32
TTA_STEPS = 2  # enable test‑time augmentation with 2 augmented copies


def seed_everything(seed):
    import random, os

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


seed_everything(SEED)


AUTOTUNE = tf.data.experimental.AUTOTUNE

options = tf.data.Options()
options.experimental_optimization.apply_default_optimizations = True
options.experimental_deterministic = False


def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    return image


def data_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    return image, label


def resize_image(image):
    return tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])


def process_path(file_path):
    img = tf.io.read_file(file_path)
    img = decode_image(img)
    return img


def make_dataset(file_paths, labels, shuffled=False, tta=False, extension="jpg"):
    """
    Build a tf.data pipeline from raw file paths and label arrays.
    """
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds = ds.map(lambda x, y: (process_path(x), y), num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda img, y: (resize_image(img), y), num_parallel_calls=AUTOTUNE)
    if tta:
        ds = ds.map(
            lambda img, _: (img, tf.zeros([], dtype=tf.int32)),
            num_parallel_calls=AUTOTUNE,
        )
        ds = ds.map(data_augment, num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda img, _: img, num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.map(data_augment, num_parallel_calls=AUTOTUNE)
    if shuffled:
        ds = ds.shuffle(1024, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    ds = ds.with_options(options)
    return ds


def get_dataset(files_path, shuffled=False, tta=False, extension="jpg"):
    """
    Used only for test‑time augmentation where we only need images.
    """
    dataset = tf.data.Dataset.list_files(f"{files_path}*{extension}", shuffle=shuffled)
    dataset = dataset.map(lambda x: process_path(x), num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(resize_image, num_parallel_calls=AUTOTUNE)
    if tta:
        dataset = dataset.map(
            lambda img: (img, tf.zeros([], dtype=tf.int32)), num_parallel_calls=AUTOTUNE
        )
        dataset = dataset.map(data_augment, num_parallel_calls=AUTOTUNE)
        dataset = dataset.map(lambda img, _: img, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    dataset = dataset.with_options(options)
    return dataset




## === cell 2
def model_fn(input_shape, n_class, weights="imagenet"):
    inputs = L.Input(shape=input_shape, name="input_image")
    base_model = EfficientNetB4(
        input_tensor=inputs, include_top=False, weights=weights, pooling="avg"
    )
    base_model.trainable = True
    x = L.Dropout(0.3)(base_model.output)
    output = L.Dense(n_class, activation="softmax", name="output")(x)
    model = Model(inputs=inputs, outputs=output)
    return model




## === cell 3
train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
train_images_path = "../input/cassava-leaf-disease-classification/train_images/"

df = pd.read_csv(train_csv_path)
file_paths = np.array(
    [os.path.join(train_images_path, img) for img in df["image_id"].values]
)
labels = df["label"].values.astype(np.int32)

np.random.seed(SEED)
perm = np.random.permutation(len(file_paths))
val_count = int(0.1 * len(file_paths))
val_idx = perm[:val_count]
train_idx = perm[val_count:]

train_paths, train_labels = file_paths[train_idx], labels[train_idx]
val_paths, val_labels = file_paths[val_idx], labels[val_idx]

train_ds = make_dataset(train_paths, train_labels, shuffled=True, tta=False)
val_ds = make_dataset(val_paths, val_labels, shuffled=False, tta=False)

n_train = len(train_paths)
n_val = len(val_paths)

model = model_fn((IMAGE_SIZE, IMAGE_SIZE, 3), CLASSES, weights="imagenet")
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-4
    ),  # slightly higher LR for fine‑tuning
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

best_weights_path = "best_cassava_efficientnetb4.weights.h5"
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    best_weights_path,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=0,
)
earlystop_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=5, restore_best_weights=True, mode="max", verbose=0
)

model.fit(
    train_ds,
    epochs=30,
    steps_per_epoch=math.ceil(n_train / BATCH_SIZE),
    validation_data=val_ds,
    validation_steps=math.ceil(n_val / BATCH_SIZE),
    verbose=1,
    callbacks=[checkpoint_cb, earlystop_cb],
)

trained_weights_path = best_weights_path
model.save_weights(trained_weights_path)




## === cell 4
test_path = "../input/cassava-leaf-disease-classification/test_images/"
test_files = sorted(glob.glob(os.path.join(test_path, "*.jpg")))
test_size = len(test_files)

model_paths = [trained_weights_path] if os.path.exists(trained_weights_path) else []

test_preds = np.zeros((test_size, CLASSES), dtype=np.float32)

if not model_paths:
    final_labels = np.zeros(test_size, dtype=int)
else:
    for model_path in tqdm(model_paths, desc="Ensembling models"):
        K.clear_session()
        model = model_fn((IMAGE_SIZE, IMAGE_SIZE, 3), CLASSES, weights="imagenet")
        model.load_weights(model_path)

        tta_accumulator = np.zeros((test_size, CLASSES), dtype=np.float32)
        for _ in range(TTA_STEPS):
            ds = get_dataset(test_path, shuffled=False, tta=True)
            preds = model.predict(ds, verbose=0)
            preds = preds[:test_size]
            tta_accumulator += preds
        tta_accumulator /= TTA_STEPS

        test_preds += tta_accumulator / len(model_paths)

    final_labels = np.argmax(test_preds, axis=1)




## === cell 5
submission = pd.DataFrame(
    {"image_id": [os.path.basename(p) for p in test_files], "label": final_labels}
)
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
