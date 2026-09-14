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

0.8176186158960411

# 6. Current score

0.64462

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61809) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation before importing TensorFlow, which resolves the `MessageFactory.GetPrototype` error in this environment. I also fix the `model.predict()` XLA failure by ensuring the test dataset fed into `predict()` contains only images (no string IDs), while still extracting IDs separately for the submission file. Finally, I make the submission-writing cell robust by defining `predictions` unconditionally and aligning predictions to `sample_submission.csv` order so a valid `submission.csv` is always produced.'
- What this solution (achieved 0.6633) has done: 'I fix the protobuf/TensorFlow crash by forcing the pure-Python protobuf runtime *and* disabling the C++ implementation explicitly before importing TensorFlow (TF 2.18 + protobuf 6 needs this in many Kaggle images). Then I keep your training/inference logic the same, but correct the image resize order to `(IMG_WIDTH, IMG_HEIGHT)` (TensorFlow uses `[height, width]`, and your variables were swapped), which is a minimal logic bugfix that should materially improve accuracy toward the target. Finally, I keep the robust submission alignment to `sample_submission.csv` and ensure IDs decode cleanly to strings so a valid `submission.csv` is always written.'
- What this solution (achieved 0.6349) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf runtime *and* disabling the C++ implementation before importing TensorFlow, which resolves the `MessageFactory.GetPrototype` error in this Kaggle environment. I keep your model and training loop unchanged, but I correct the image resize call to use `[height, width]` consistent with your `IMG_HEIGHT/IMG_WIDTH` variables (a small preprocessing bug that should improve accuracy toward the target). I also keep the safe inference path that feeds only images into `model.predict()` and separately extracts IDs for the submission, ensuring alignment to `sample_submission.csv`. Finally, I make sure a valid `submission.csv` is always written with the exact required columns.'
- What this solution (achieved 0.64462) has done: 'You’re currently crashing immediately on importing TensorFlow due to an incompatibility between TF 2.18 and protobuf 6 where `MessageFactory.GetPrototype` is missing; the environment variables alone aren’t reliably applied early enough for this Kaggle image. I fix this by forcing the pure-Python protobuf implementation *before* TensorFlow import via a safe `google.protobuf` monkeypatch fallback when needed, keeping your model/training/inference logic unchanged. I also add a small safety fallback for `CLASS_NAMES` reading (some Kaggle copies store the mapping as a plain JSON dict) and ensure the submission is always written as `submission.csv` with correct types/order. No changes are made to architecture, loss, optimizer, epochs, or dataset logic, so any score change should be negligible and the main goal is end-to-end stability.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_DISABLE_CPP_IMPLEMENTATION"] = "1"

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import message_factory as _message_factory

    if hasattr(_message_factory, "MessageFactory") and not hasattr(
        _message_factory.MessageFactory, "GetPrototype"
    ):
        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        _message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

import re
import numpy as np
import pandas as pd
import tensorflow as tf

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
AUTOTUNE = tf.data.AUTOTUNE

FILENAMES = tf.io.gfile.glob(DATA_DIR + "train_tfrecords/*.tfrec")
FILENAMES = sorted(FILENAMES)

split_ind = int(0.9 * len(FILENAMES))
TRAINING_FILENAMES, VALID_FILENAMES = FILENAMES[:split_ind], FILENAMES[split_ind:]

try:
    CLASS_NAMES = pd.read_json(DATA_DIR + "label_num_to_disease_map.json", typ="series")
except ValueError:
    import json

    with tf.io.gfile.GFile(DATA_DIR + "label_num_to_disease_map.json", "r") as f:
        CLASS_NAMES = pd.Series(json.load(f))

BATCH_SIZE = 16
IMG_HEIGHT = 400
IMG_WIDTH = 300
IMAGE_SIZE = (IMG_HEIGHT, IMG_WIDTH)

SEED = 42
tf.keras.utils.set_random_seed(SEED)

print("TF version:", tf.__version__)
print(
    "Num train tfrecs:",
    len(TRAINING_FILENAMES),
    "Num valid tfrecs:",
    len(VALID_FILENAMES),
)
print("Class names:", CLASS_NAMES.to_dict())



## === cell 1
from tensorflow.keras import layers, Model

inp = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3), dtype=tf.float32, name="image")
x = layers.Rescaling(1.0 / 255.0)(inp)
x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
out = layers.Dense(5, activation="softmax")(x)

model = Model(inputs=inp, outputs=out, name="cassava_cnn")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()




## === cell 2
def count_data_items(filenames):
    n = [
        int(re.compile(r"-([0-9]*)\.").search(filename).group(1))
        for filename in filenames
    ]
    return int(np.sum(n))


TEST_FILENAMES = sorted(tf.io.gfile.glob(DATA_DIR + "test_tfrecords/*.tfrec"))
NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)

print("Num test tfrecs:", len(TEST_FILENAMES), "Num test images:", NUM_TEST_IMAGES)



## === cell 3
from functools import partial


def decode_image(image_bytes):
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = tf.cast(image, tf.float32)
    return image


def read_tfrecord(example, labeled):
    feature_description = (
        {
            "target": tf.io.FixedLenFeature([], tf.int64),
            "image_name": tf.io.FixedLenFeature([], tf.string),
            "image": tf.io.FixedLenFeature([], tf.string),
        }
        if labeled
        else {
            "image_name": tf.io.FixedLenFeature([], tf.string),
            "image": tf.io.FixedLenFeature([], tf.string),
        }
    )
    example = tf.io.parse_single_example(example, feature_description)
    image = decode_image(example["image"])

    if labeled:
        label = tf.cast(example["target"], tf.int32)
        label = tf.one_hot(label, depth=5, dtype=tf.float32)
        return image, label

    image_id = example["image_name"]
    return image, image_id


def load_dataset(filenames, labeled=True, ordered=False):
    opts = tf.data.Options()
    if not ordered:
        opts.experimental_deterministic = False
    ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    ds = ds.with_options(opts)
    ds = ds.map(partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE)
    return ds


def get_training_dataset(filenames):
    ds = load_dataset(filenames, labeled=True, ordered=False)
    ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


def get_validation_dataset(filenames):
    ds = load_dataset(filenames, labeled=True, ordered=True)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


def get_test_dataset(filenames):
    ds = load_dataset(filenames, labeled=False, ordered=True)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


train_dataset = get_training_dataset(TRAINING_FILENAMES)
valid_dataset = get_validation_dataset(VALID_FILENAMES)
test_dataset = get_test_dataset(TEST_FILENAMES)



## === cell 4
EPOCHS = 5
history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=EPOCHS,
    verbose=1,
)



## === cell 5
test_images_ds = test_dataset.map(
    lambda image, image_id: image, num_parallel_calls=AUTOTUNE
)

probabilities = model.predict(test_images_ds, verbose=1)

probabilities = np.asarray(probabilities)
if probabilities.ndim != 2 or probabilities.shape[1] != 5:
    raise ValueError(
        f"Unexpected model output shape: {probabilities.shape}; expected (N, 5)"
    )

predictions = np.argmax(probabilities, axis=1).astype(np.int64)
print("Pred shape:", predictions.shape, "Unique:", np.unique(predictions))



## === cell 6
test_dataset_id = test_dataset.map(
    lambda image, idnum: idnum, num_parallel_calls=AUTOTUNE
).unbatch()
test_ids = next(iter(test_dataset_id.batch(NUM_TEST_IMAGES))).numpy()

test_ids = np.array(
    [
        x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)
        for x in test_ids
    ],
    dtype=object,
)

if len(test_ids) != len(predictions):
    raise ValueError(f"Mismatch: {len(test_ids)} ids vs {len(predictions)} predictions")

pred_df = pd.DataFrame({"image_id": test_ids, "label": predictions})

sample_path = DATA_DIR + "sample_submission.csv"
sample = pd.read_csv(sample_path)

sub = sample[["image_id"]].merge(pred_df, on="image_id", how="left")

if sub["label"].isna().any():
    sub["label"] = (
        sub["label"].fillna(int(pred_df["label"].mode().iloc[0])).astype(np.int64)
    )
else:
    sub["label"] = sub["label"].astype(np.int64)

sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)

assert list(sub.columns) == ["image_id", "label"]
assert sub["image_id"].dtype == object
assert sub["label"].between(0, 4).all()
assert len(sub) == len(sample)
