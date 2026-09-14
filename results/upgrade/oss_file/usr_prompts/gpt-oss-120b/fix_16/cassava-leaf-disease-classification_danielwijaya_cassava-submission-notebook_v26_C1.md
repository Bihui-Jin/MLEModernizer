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

# 5. Code solution

## === cell 0
import os, random, sys, re, tempfile

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    from google.protobuf import message_factory as _mf

    if not hasattr(_mf.MessageFactory, "GetPrototype"):
        prototype = getattr(_mf.MessageFactory, "GetMessageClass", None)
        if prototype is not None:
            setattr(_mf.MessageFactory, "GetPrototype", prototype)
except Exception:
    pass  # If protobuf is unavailable, TensorFlow will raise later if required

import numpy as np
import pandas as pd
import tensorflow as tf

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

tf.config.threading.set_intra_op_parallelism_threads(min(8, os.cpu_count() or 4))
tf.config.threading.set_inter_op_parallelism_threads(4)

tf.config.optimizer.set_jit(True)

from functools import partial
from sklearn.model_selection import train_test_split




## === cell 1
if tf.config.list_physical_devices("GPU"):
    from tensorflow.keras.mixed_precision import Policy, set_global_policy

    policy = Policy("mixed_float16")
    set_global_policy(policy)

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 16 * 8
IMAGE_SIZE = [512, 512]
NUM_CLASSES = 5

DATA_ROOT = "../input/cassava-leaf-disease-classification"
if not os.path.isdir(DATA_ROOT):
    fallback_root = "/kaggle/input/cassava-leaf-disease-classification"
    if os.path.isdir(fallback_root):
        DATA_ROOT = fallback_root


def dataset_sizes(filenames):
    """Return an approximate total number of images from TFRecord filenames."""
    if not filenames:
        return 0
    nums = [int(re.search(r"-([0-9]+)\.", f).group(1)) for f in filenames]
    return int(np.sum(nums))


def decode_img(img_bytes):
    """Decode JPEG bytes to a normalized float32 tensor."""
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.image.resize(img, IMAGE_SIZE)
    return img


def read_tfrecord(example, labeled):
    """Parse a single TFRecord example."""
    if labeled:
        fmt = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "target": tf.io.FixedLenFeature([], tf.int64),
        }
    else:
        fmt = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_name": tf.io.FixedLenFeature([], tf.string),
        }
    parsed = tf.io.parse_single_example(example, fmt)
    image = decode_img(parsed["image"])
    if labeled:
        label = tf.cast(parsed["target"], tf.int32)
        return image, label
    else:
        return image, parsed["image_name"]


def load_dataset(filenames, labeled=True, ordered=False):
    """Create a tf.data.Dataset from TFRecord files without disk caching."""
    opts = tf.data.Options()
    if not ordered:
        opts.experimental_deterministic = False
    ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    ds = ds.with_options(opts)
    ds = ds.map(partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE)
    return ds




## === cell 2
train_filenames = tf.io.gfile.glob(
    os.path.join(DATA_ROOT, "train_tfrecords", "ld_train*.tfrec")
)
test_filenames = tf.io.gfile.glob(
    os.path.join(DATA_ROOT, "test_tfrecords", "ld_test*.tfrec")
)

train_filenames, val_filenames = train_test_split(
    train_filenames, test_size=0.1, random_state=42
)

NUM_TRAIN_IMAGES = dataset_sizes(train_filenames)
NUM_VAL_IMAGES = dataset_sizes(val_filenames)
NUM_TEST_IMAGES = dataset_sizes(test_filenames)

train_ds = load_dataset(train_filenames, labeled=True, ordered=False)
train_cache_path = os.path.join(tempfile.gettempdir(), "train_cache")
train_ds = train_ds.cache(train_cache_path)  # cache decoded records
train_ds = train_ds.shuffle(1024).batch(BATCH_SIZE).prefetch(AUTOTUNE)

val_ds = load_dataset(val_filenames, labeled=True, ordered=False)
val_cache_path = os.path.join(tempfile.gettempdir(), "val_cache")
val_ds = val_ds.cache(val_cache_path)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

test_ds_ordered = load_dataset(test_filenames, labeled=False, ordered=True)
test_ds_ordered = test_ds_ordered.batch(BATCH_SIZE).prefetch(AUTOTUNE)




## === cell 3
base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(*IMAGE_SIZE, 3), weights="imagenet", include_top=False, pooling="avg"
)

inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
x = tf.keras.layers.RandomFlip(mode="horizontal_and_vertical")(inputs)
x = tf.keras.layers.RandomRotation(factor=0.2)(x)
x = base_model(x, training=False)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

steps_per_epoch = max(1, NUM_TRAIN_IMAGES // BATCH_SIZE)
validation_steps = max(1, NUM_VAL_IMAGES // BATCH_SIZE)

model.fit(
    train_ds,
    epochs=30,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=validation_steps,
    verbose=0,
)




## === cell 4
test_images = test_ds_ordered.map(lambda img, _: img)

probabilities = model.predict(test_images, verbose=0)
predictions = np.argmax(probabilities, axis=-1).astype(int)




## === cell 5
sample_sub_path = os.path.join(DATA_ROOT, "sample_submission.csv")
submission_df = pd.read_csv(sample_sub_path)

assert len(predictions) == len(submission_df), "Prediction count mismatch."

submission_df["label"] = predictions

output_path = "submission.csv"
submission_df.to_csv(output_path, index=False)

print(f"Submission file written to {output_path}")
