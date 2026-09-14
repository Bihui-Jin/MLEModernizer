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

0.8928679359323058

# 6. Current score

0.11584

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.73094) has done: 'I remove the failing external EfficientNet/kerasapplications imports that crash under the current protobuf/TensorFlow stack, since they are not needed for TFRecord reading or for a simple baseline model. Because the referenced pretrained `.h5` files are not available in this environment, I replace that block with a minimal, standard TF/Keras EfficientNet model trained on the provided `train_tfrecords` and then used to predict on `test_tfrecords`. I also fix the TFRecord parsing to correctly decode JPEGs (without forcing an incorrect reshape) and ensure the test IDs align exactly with predictions. Finally, I write a valid `submission.csv` with the required `image_id,label` columns.'
- What this solution (achieved 0.53288) has done: 'I fix the protobuf/TensorFlow import crash (`MessageFactory.GetPrototype`), which is caused by an incompatible protobuf runtime for TF 2.18 in this environment; the minimal safe workaround is to force the pure-Python protobuf implementation before importing TensorFlow. I also keep the existing TFRecord parsing/model/training logic intact, but switch EfficientNet to use ImageNet weights (available inside `tf.keras.applications`) to legitimately improve accuracy toward your target without changing architecture/training loop. Finally, I make TFRecord filename parsing more robust and ensure deterministic file ordering for stable ID/prediction alignment, while still producing a valid `submission.csv`.'
- What this solution (achieved 0.08894) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *and* ensuring it is applied before any protobuf/TensorFlow-related imports take effect in this notebook process. I also add a small, safe compatibility fallback that patches the missing `MessageFactory.GetPrototype` symbol when the runtime protobuf version removed it (this is score-neutral but unblocks execution). Finally, I keep your data pipeline/model/training logic unchanged and only add minor safety checks so the TFRecord counts and submission alignment don’t break if filename-count parsing fails.'
- What this solution (achieved 0.11584) has done: 'I remove the protobuf monkey-patch that is triggering the `MessageFactory` AttributeError in this environment and rely only on the safe `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` switch (set before importing TensorFlow), which is sufficient to unblock TF 2.18 here. I keep your TFRecord parsing, EfficientNetB0(ImageNet) model, training loop, and submission-writing logic unchanged to preserve core semantics and move accuracy back toward your target (your current 0.08894 is far below target, consistent with the current crash/partial run issues). I also add a tiny safety fallback for test-id extraction so the submission is always produced even if TFRecord filename counts don’t parse (score-neutral, stability-only). All paths and the required `submission.csv` format remain the same.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import re
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from functools import partial
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TF version:", tf.__version__)

BASE_PATH = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(SAMPLE_SUB)
print("train_df:", train_df.shape, "test_df:", test_df.shape)

AUTOTUNE = tf.data.AUTOTUNE

IMAGE_SIZE = (512, 512)
NUM_CLASSES = 5
BATCH_SIZE = 32


def dataset_sizes(filenames):
    out = []
    pat = re.compile(r"-([0-9]+)\.tfrec$")
    for fn in filenames:
        m = pat.search(fn)
        out.append(int(m.group(1)) if m else 0)
    return int(np.sum(out))


TRAIN_FILENAMES = sorted(
    tf.io.gfile.glob(os.path.join(BASE_PATH, "train_tfrecords", "ld_train*.tfrec"))
)
TEST_FILENAMES = sorted(
    tf.io.gfile.glob(os.path.join(BASE_PATH, "test_tfrecords", "ld_test*.tfrec"))
)

NUM_TRAIN_IMAGES = dataset_sizes(TRAIN_FILENAMES)
NUM_TEST_IMAGES = dataset_sizes(TEST_FILENAMES)

print("TFRecord files:", len(TRAIN_FILENAMES), "train,", len(TEST_FILENAMES), "test")
print(
    "Image counts (from filenames):",
    NUM_TRAIN_IMAGES,
    "train,",
    NUM_TEST_IMAGES,
    "test",
)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def decode_img(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE, method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    return img


def read_tfrecord(example, labeled):
    if labeled:
        tfrec_format = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "target": tf.io.FixedLenFeature([], tf.int64),
        }
    else:
        tfrec_format = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_name": tf.io.FixedLenFeature([], tf.string),
        }
    ex = tf.io.parse_single_example(example, tfrec_format)
    img = decode_img(ex["image"])
    if labeled:
        label = tf.cast(ex["target"], tf.int32)
        return img, label
    else:
        image_id = ex["image_name"]
        return img, image_id


def load_dataset(filenames, labeled=True, ordered=False):
    opts = tf.data.Options()
    if not ordered:
        opts.experimental_deterministic = False
    ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    ds = ds.with_options(opts)
    ds = ds.map(partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE)
    return ds


train_files, valid_files = train_test_split(
    TRAIN_FILENAMES, test_size=0.1, random_state=SEED
)
print(
    "Split TFRecords:",
    len(train_files),
    "train files,",
    len(valid_files),
    "valid files",
)


def get_training_dataset():
    ds = load_dataset(train_files, labeled=True, ordered=False)
    ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def get_validation_dataset():
    ds = load_dataset(valid_files, labeled=True, ordered=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def get_test_dataset(ordered=True):
    ds = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = get_training_dataset()
valid_ds = get_validation_dataset()
test_ds = get_test_dataset(ordered=True)

inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
backbone = tf.keras.applications.EfficientNetB0(
    include_top=False, weights="imagenet", input_tensor=inputs, pooling="avg"
)
x = tf.keras.layers.Dropout(0.2)(backbone.output)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
)

model.summary()



## === cell 2
EPOCHS = 3

print("Training...")
history = model.fit(train_ds, validation_data=valid_ds, epochs=EPOCHS, verbose=2)



## === cell 3
print("Computing predictions...")
test_images_ds = test_ds.map(lambda img, image_id: img, num_parallel_calls=AUTOTUNE)
probs = model.predict(test_images_ds, verbose=1)
predictions = np.argmax(probs, axis=-1).astype(int)

print("Predictions shape:", predictions.shape, "unique:", np.unique(predictions))



## === cell 4
print("Generating submission.csv file...")
test_ids_ds = test_ds.map(
    lambda img, image_id: image_id, num_parallel_calls=AUTOTUNE
).unbatch()

expected_n = int(NUM_TEST_IMAGES) if int(NUM_TEST_IMAGES) > 0 else int(len(test_df))

try:
    test_ids = next(iter(test_ids_ds.batch(expected_n))).numpy().astype("U")
except Exception:
    test_ids = None

if test_ids is None or len(test_ids) != len(predictions):
    test_ids = np.array([x.numpy().decode("utf-8") for x in test_ids_ds], dtype="U")

if len(test_ids) != len(predictions):
    raise RuntimeError(
        f"Mismatch: {len(test_ids)} test_ids vs {len(predictions)} predictions"
    )

sub = pd.DataFrame({"image_id": test_ids, "label": predictions})

sub = test_df[["image_id"]].merge(sub, on="image_id", how="left")
if sub["label"].isna().any():
    sub["label"] = sub["label"].fillna(0).astype(int)
else:
    sub["label"] = sub["label"].astype(int)

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
