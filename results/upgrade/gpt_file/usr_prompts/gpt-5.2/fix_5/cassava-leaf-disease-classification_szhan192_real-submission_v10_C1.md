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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
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

0.8058325778180719

# 6. Current score

0.46674

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11809) has done: 'I fix the runtime crash caused by an incompatibility between TensorFlow 2.18 and protobuf 6 by forcing the pure-Python protobuf implementation before importing TensorFlow (this avoids the `MessageFactory.GetPrototype` error). I also make model loading robust by adding `compile=False` and a clear fallback path so the notebook still produces a valid `submission.csv` even if the external model file isn’t available. Finally, I replace the slow per-image `model.predict` loop with a batched `tf.data` pipeline over the same test images (same preprocessing semantics: resize to 256 and scale to [0,1]) so it finishes within the time limit and writes a correctly formatted CSV.'
- What this solution (achieved 0.65321) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *and* ensuring it is applied before TensorFlow (or anything that pulls protobuf) is imported, which requires moving that environment setting to the very top of the script. To improve the score toward your target (the current 0.118 is essentially random), I keep the same inference core (load pretrained model if available, otherwise fallback) but also add a minimal, standard training fallback on `train.csv` using the same image resizing/scaling pipeline so the notebook can learn a non-random classifier when the external `.h5` isn’t present. This preserves the overall approach (image classification with a simple Keras model) while fixing the runtime error and making the output meaningful. The script still write `/kaggle/working/submission.csv` with the required `image_id,label` columns.'
- What this solution (achieved 0.65957) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before any protobuf/TensorFlow import* and also setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2`, which is the known stable combination for TF 2.18 in Kaggle. Then I keep your existing training/inference logic intact, but make the TFRecord-free image pipeline more robust (ensure `preprocess_image` returns a fixed shape) to avoid any occasional shape-related graph issues. Finally, I keep the same submission writing code and guarantee a valid `/kaggle/working/submission.csv` is always produced.'
- What this solution (achieved 0.46674) has done: 'I fix the TensorFlow/protobuf crash by ensuring the pure-Python protobuf environment variables are set before any TensorFlow-related import in a way that’s robust in Kaggle’s execution order (including clearing already-imported protobuf modules if present). Then I make the fallback training slightly stronger (while keeping the same simple CNN training approach) by adding standard, light image augmentation and class-weighting to address cassava label imbalance; this should move accuracy up toward your 0.8058 target from 0.6596. I also ensure the preprocessing is consistent and stable (fixed shapes/dtypes) and that the submission is always written as `/kaggle/working/submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf"):
        del sys.modules[m]

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42)
np.random.seed(42)

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

TARGET_SIZE = 256
BATCH_SIZE = 32
NUM_CLASSES = 5

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_paths = [
    os.path.join(TEST_IMG_DIR, img_id) for img_id in sample_sub["image_id"].tolist()
]

train_df = pd.read_csv(TRAIN_CSV_PATH)
train_paths = [
    os.path.join(TRAIN_IMG_DIR, img_id) for img_id in train_df["image_id"].tolist()
]
train_labels = train_df["label"].astype(np.int32).values

print("Train rows:", len(train_paths), "Test rows:", len(test_paths))
print("TensorFlow:", tf.__version__)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def build_fallback_model(input_size=TARGET_SIZE, num_classes=NUM_CLASSES):
    aug = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.05),
            keras.layers.RandomZoom(0.10),
            keras.layers.RandomContrast(0.10),
        ],
        name="augment",
    )

    inputs = keras.Input(shape=(input_size, input_size, 3))
    x = aug(inputs)
    x = keras.layers.Rescaling(1.0)(x)  # identity: data already scaled to [0,1]
    x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPool2D()(x)
    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPool2D()(x)
    x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    return model


@tf.function
def preprocess_image(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [TARGET_SIZE, TARGET_SIZE], method="bilinear", antialias=True
    )
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape([TARGET_SIZE, TARGET_SIZE, 3])
    return img


def make_labeled_ds(paths, labels, batch_size=BATCH_SIZE, training=True):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(min(len(paths), 8192), seed=42, reshuffle_each_iteration=True)

    def _map_fn(p, y):
        img = preprocess_image(p)
        y = tf.cast(y, tf.int32)
        return img, y

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def make_test_ds(paths, batch_size=BATCH_SIZE):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def compute_class_weight(labels, num_classes=NUM_CLASSES):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    total = counts.sum()
    weights = total / (num_classes * np.maximum(counts, 1.0))
    return {i: float(weights[i]) for i in range(num_classes)}




## === cell 2
model_path = "../input/densenet22/densenet_22.h5"
model = None

try:
    model = keras.models.load_model(model_path, compile=False)
    print(f"Loaded pretrained model: {model_path}")
except Exception as e:
    print(f"WARNING: Could not load model at {model_path}: {repr(e)}")
    print("Training a lightweight fallback model to produce a stronger submission.")

    n = len(train_paths)
    idx = np.arange(n)
    rng = np.random.RandomState(42)
    rng.shuffle(idx)

    val_size = int(0.1 * n)
    val_idx = idx[:val_size]
    tr_idx = idx[val_size:]

    tr_paths = [train_paths[i] for i in tr_idx]
    tr_labels = train_labels[tr_idx]
    val_paths = [train_paths[i] for i in val_idx]
    val_labels = train_labels[val_idx]

    train_ds = make_labeled_ds(tr_paths, tr_labels, training=True)
    val_ds = make_labeled_ds(val_paths, val_labels, training=False)

    model = build_fallback_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )

    class_weight = compute_class_weight(tr_labels, num_classes=NUM_CLASSES)
    print("Using class_weight:", class_weight)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5,
        verbose=2,
        class_weight=class_weight,
    )



## === cell 3
test_ds = make_test_ds(test_paths, batch_size=BATCH_SIZE)

probs = model.predict(test_ds, verbose=1)
preds = np.argmax(probs, axis=1).astype(int)

my_submission = pd.DataFrame(
    {"image_id": sample_sub["image_id"].values, "label": preds}
)
out_path = "/kaggle/working/submission.csv"
my_submission.to_csv(out_path, index=False)

print("Wrote", out_path, "with shape:", my_submission.shape)
print(my_submission.head())
print(
    "Label distribution:", my_submission["label"].value_counts().sort_index().to_dict()
)
