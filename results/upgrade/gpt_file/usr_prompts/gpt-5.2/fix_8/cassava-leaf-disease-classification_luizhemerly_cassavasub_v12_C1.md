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

0.8679359323058325

# 6. Current score

0.63565

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61622) has done: 'I remove the protobuf “python” fallback that’s triggering the `MessageFactory.GetPrototype` crash under TF 2.18 + protobuf 6, so TensorFlow can import cleanly. Then I fix the test `tf.data` pipeline so it doesn’t return a string tensor alongside the image during `model.predict` (XLA can’t compile DT_STRING inputs), while still collecting `image_id`s for the submission. Finally, I ensure the submission is aligned to the sample submission order and always writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.61846) has done: 'I fix the TensorFlow import crash caused by the protobuf 6 API change by pinning protobuf to the pure-Python implementation via environment variables before importing TensorFlow. Then I make the TFRecord parsing more robust by supporting both possible label feature names (`target` and `label`) so training always uses the correct labels (this should materially improve accuracy toward your target without changing the model/training loop). Finally, I keep the existing “predict images only / collect ids separately” logic and ensure the submission is aligned to `sample_submission.csv` and written as `submission.csv`.'
- What this solution (achieved 0.61958) has done: 'I fix the TensorFlow import crash by removing the protobuf pure-Python fallback (it’s what triggers the `MessageFactory.GetPrototype` error under TF 2.18 + protobuf 6 in Kaggle). Then I keep your TFRecord parsing/training logic the same, but make the label feature selection robust without introducing invalid labels (so training isn’t silently corrupted). Finally, I ensure test IDs are decoded cleanly and the written `submission.csv` is aligned to `sample_submission.csv` with the exact required columns.'
- What this solution (achieved 0.63378) has done: 'I fix the TensorFlow import crash by removing the forced protobuf “cpp” setting that is incompatible with the provided protobuf build in this environment, and instead force the pure-Python protobuf implementation before importing TensorFlow. This unblock execution of the remaining cells (which currently fail only because cell 0 never completes, leaving variables like `tf` and `TRAINING_FILENAMES` undefined). I keep your TFRecord parsing, model, and training loop intact, only adding a small safety fallback to locate `DATA_DIR` reliably across the mirrored Kaggle paths. Finally, I ensure test ID decoding is robust and that a valid `submission.csv` with the required columns is always written aligned to `sample_submission.csv`.'
- What this solution (achieved 0.63565) has done: 'The crash happens before any training because TensorFlow 2.18 is incompatible with forcing protobuf’s pure-Python backend in this Kaggle image; removing that environment override allows TF to import cleanly and the rest of your pipeline to run. I keep your TFRecord parsing, model, and training loop unchanged, only adjusting the protobuf env handling to avoid the `MessageFactory.GetPrototype` AttributeError. I also make test id decoding simpler/safer by directly decoding the batched string tensors (no codepoint conversion), which is score-neutral but prevents subtle id corruption that can hurt submission alignment. The script still write a valid `submission.csv` with the required columns and align to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import re
import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

_CANDIDATE_DIRS = [
    "/kaggle/input/cassava-leaf-disease-classification/",
    "/kaggle/data/cassava-leaf-disease-classification/",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/",
    "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification/",
]
DATA_DIR = None
for d in _CANDIDATE_DIRS:
    if tf.io.gfile.exists(d) and tf.io.gfile.exists(os.path.join(d, "train_tfrecords")):
        DATA_DIR = d
        break
if DATA_DIR is None:
    DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification/"

AUTOTUNE = tf.data.AUTOTUNE

TRAIN_FILENAMES = sorted(tf.io.gfile.glob(DATA_DIR + "train_tfrecords/*.tfrec"))
TEST_FILENAMES = sorted(tf.io.gfile.glob(DATA_DIR + "test_tfrecords/*.tfrec"))

split_ind = int(0.9 * len(TRAIN_FILENAMES))
TRAINING_FILENAMES, VALID_FILENAMES = (
    TRAIN_FILENAMES[:split_ind],
    TRAIN_FILENAMES[split_ind:],
)

CLASS_NAMES = pd.read_json(DATA_DIR + "label_num_to_disease_map.json", typ="series")
NUM_CLASSES = len(CLASS_NAMES)

BATCH_SIZE = 16
IMG_HEIGHT = 400
IMG_WIDTH = 300
IMAGE_SIZE = (IMG_HEIGHT, IMG_WIDTH)

print("TF:", tf.__version__)
print("DATA_DIR:", DATA_DIR)
print(
    "Train tfrecs:",
    len(TRAIN_FILENAMES),
    "-> train/valid:",
    len(TRAINING_FILENAMES),
    len(VALID_FILENAMES),
)
print("Test tfrecs:", len(TEST_FILENAMES))
print("Num classes:", NUM_CLASSES)


def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(fn).group(1)) for fn in filenames]
    return int(np.sum(n))


NUM_TRAIN_IMAGES = count_data_items(TRAIN_FILENAMES) if TRAIN_FILENAMES else 0
NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES) if TEST_FILENAMES else 0
print("NUM_TRAIN_IMAGES:", NUM_TRAIN_IMAGES)
print("NUM_TEST_IMAGES:", NUM_TEST_IMAGES)

assert (
    len(TRAIN_FILENAMES) > 0
), f"No train TFRecords found under {DATA_DIR}train_tfrecords/"
assert (
    len(TEST_FILENAMES) > 0
), f"No test TFRecords found under {DATA_DIR}test_tfrecords/"



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
from functools import partial


def decode_image(image_bytes):
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH], method="bilinear")
    image = tf.cast(image, tf.float32) / 255.0
    return image


def _augment_image(image):
    image = tf.image.random_flip_left_right(image, seed=SEED)
    image = tf.image.random_brightness(image, max_delta=0.10)
    image = tf.image.random_contrast(image, lower=0.90, upper=1.10)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def read_tfrecord(example, labeled):
    if labeled:
        feature_description = {
            "image_name": tf.io.FixedLenFeature([], tf.string),
            "image": tf.io.FixedLenFeature([], tf.string),
            "target": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
            "label": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
        }
    else:
        feature_description = {
            "image_name": tf.io.FixedLenFeature([], tf.string),
            "image": tf.io.FixedLenFeature([], tf.string),
        }

    example = tf.io.parse_single_example(example, feature_description)
    image = decode_image(example["image"])

    if labeled:
        image = _augment_image(image)

        target = tf.cast(example["target"], tf.int32)
        label = tf.cast(example["label"], tf.int32)

        y = tf.where(label >= 0, label, target)
        y = tf.where(y >= 0, y, tf.zeros_like(y))

        y = tf.one_hot(y, depth=NUM_CLASSES, dtype=tf.float32)
        return image, y

    idnum = example["image_name"]
    return image, idnum


def load_dataset(filenames, labeled=True, ordered=False):
    opts = tf.data.Options()
    if not ordered:
        opts.experimental_deterministic = False
    ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE).with_options(
        opts
    )
    ds = ds.map(partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE)
    return ds


def get_training_dataset(filenames):
    ds = load_dataset(filenames, labeled=True, ordered=False)
    ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def get_validation_dataset(filenames):
    ds = load_dataset(filenames, labeled=True, ordered=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def get_test_dataset_for_predict(filenames):
    ds = load_dataset(filenames, labeled=False, ordered=True)  # yields (image, id)
    ds = ds.map(lambda image, idnum: image, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def get_test_dataset_for_ids(filenames):
    ds = load_dataset(filenames, labeled=False, ordered=True)  # yields (image, id)
    ds = ds.map(lambda image, idnum: idnum, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_dataset = get_training_dataset(TRAINING_FILENAMES)
valid_dataset = get_validation_dataset(VALID_FILENAMES)
test_dataset_pred = get_test_dataset_for_predict(TEST_FILENAMES)
test_dataset_ids = get_test_dataset_for_ids(TEST_FILENAMES)




## === cell 2
def build_model():
    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    return model


model = build_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc")],
)
model.summary()



## === cell 3
EPOCHS = 3

history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=EPOCHS,
    verbose=2,
)



## === cell 4
probabilities = model.predict(test_dataset_pred, verbose=1)
predictions = np.argmax(probabilities, axis=1).astype(np.int64)
print("probabilities:", probabilities.shape, "predictions:", predictions.shape)
print("predictions[:10]:", predictions[:10])

test_ids = np.concatenate(
    [batch.numpy().astype("U") for batch in test_dataset_ids], axis=0
)

assert len(test_ids) == len(predictions), (len(test_ids), len(predictions))

submission = pd.DataFrame({"image_id": test_ids, "label": predictions})

sample_path = DATA_DIR + "sample_submission.csv"
if tf.io.gfile.exists(sample_path):
    sample = pd.read_csv(sample_path)
    submission = sample[["image_id"]].merge(submission, on="image_id", how="left")
    submission["label"] = submission["label"].fillna(0).astype(np.int64)

submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote:", "submission.csv", "rows:", len(submission))
print("label value counts:\n", submission["label"].value_counts().sort_index())
