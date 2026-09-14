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
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
try:
    from google.protobuf.message_factory import MessageFactory

    if not hasattr(MessageFactory, "GetPrototype"):

        def _get_prototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        MessageFactory.GetPrototype = _get_prototype
except Exception:
    pass  # TensorFlow will raise later if needed

import json
import numpy as np
import pandas as pd
import tensorflow as tf
import tempfile  # used for a single disk cache path

tf.config.optimizer.set_jit(True)

num_threads = os.cpu_count() or 1
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.threading.set_inter_op_parallelism_threads(num_threads)

from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 16 * 8  # frozen‑stage batch size
FINE_BATCH_SIZE = 16  # fine‑tuning batch size
IMAGE_SIZE = [512, 512]
NUM_CLASSES = 5
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

possible_paths = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "./input/cassava-leaf-disease-classification",
    "./data/cassava-leaf-disease-classification",
    "./working/input/cassava-leaf-disease-classification",
]
BASE_PATH = next((p for p in possible_paths if os.path.isdir(p)), None)
if BASE_PATH is None:
    raise FileNotFoundError(
        "Could not locate cassava-leaf-disease-classification data directory."
    )

TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train_images")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test_images")
SAMPLE_SUBMISSION = os.path.join(BASE_PATH, "sample_submission.csv")
SUBMISSION_PATH = "submission.csv"




## === cell 1
train_df = pd.read_csv(TRAIN_CSV)
train_df["filepath"] = train_df["image_id"].apply(
    lambda x: os.path.join(TRAIN_IMG_DIR, x)
)

train_df = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
val_split = int(0.9 * len(train_df))
train_paths = train_df["filepath"][:val_split].values
train_labels = train_df["label"][:val_split].values
val_paths = train_df["filepath"][val_split:].values
val_labels = train_df["label"][val_split:].values

test_filenames = sorted(
    [f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(".jpg")]
)
test_paths = [os.path.join(TEST_IMG_DIR, f) for f in test_filenames]




## === cell 2
def decode_and_preprocess(path, label=None):
    """Read an image file, decode JPEG, resize, and normalize."""
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    if label is None:
        return image
    return image, label


def augment(image):
    """Simple random augmentations for training."""
    image = tf.image.random_flip_left_right(image, seed=SEED)
    image = tf.image.random_flip_up_down(image, seed=SEED + 1)
    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32, seed=SEED + 2)
    image = tf.image.rot90(image, k)
    return image


def prepare_dataset(
    paths,
    labels=None,
    batch_size=BATCH_SIZE,
    shuffle=False,
    augment_flag=False,
    cache=False,
    cache_path=None,
):
    """
    Create a tf.data pipeline.

    Parameters
    ----------
    cache : bool
        Whether to cache the dataset. If `cache_path` is provided the cache is
        stored on disk (avoiding RAM pressure while still saving decode time).
    """
    if labels is not None:
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    else:
        ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(decode_and_preprocess, num_parallel_calls=AUTOTUNE)
    if augment_flag:
        ds = ds.map(lambda img, lbl: (augment(img), lbl), num_parallel_calls=AUTOTUNE)
    if cache:
        if cache_path:
            ds = ds.cache(cache_path)  # disk‑based cache
        else:
            ds = ds.cache()  # in‑memory cache
    if shuffle:
        ds = ds.shuffle(buffer_size=1024, seed=SEED)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


train_cache_file = tempfile.NamedTemporaryFile(delete=False).name

train_ds = prepare_dataset(
    train_paths,
    train_labels,
    batch_size=BATCH_SIZE,
    shuffle=True,
    augment_flag=True,
    cache=True,
    cache_path=train_cache_file,  # disk cache to avoid OOM and speed up epochs
)
val_ds = prepare_dataset(
    val_paths,
    val_labels,
    batch_size=BATCH_SIZE,
    shuffle=False,
    augment_flag=False,
    cache=True,  # in‑memory cache is fine for the smaller validation set
)

test_ds = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(lambda p: decode_and_preprocess(p), num_parallel_calls=AUTOTUNE)
    .cache()
    .batch(FINE_BATCH_SIZE)
    .prefetch(AUTOTUNE)
)




## === cell 3
base = tf.keras.applications.EfficientNetB5(
    include_top=False, weights="imagenet", input_shape=(*IMAGE_SIZE, 3)
)
base.trainable = False  # frozen stage

inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
x = base(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

model.fit(train_ds, validation_data=val_ds, epochs=3, verbose=2)




## === cell 4
base.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

train_ds_fine = prepare_dataset(
    train_paths,
    train_labels,
    batch_size=FINE_BATCH_SIZE,
    shuffle=True,
    augment_flag=True,
    cache=True,
    cache_path=train_cache_file,  # reuse the existing disk cache
)
val_ds_fine = prepare_dataset(
    val_paths,
    val_labels,
    batch_size=FINE_BATCH_SIZE,
    shuffle=False,
    augment_flag=False,
    cache=True,  # in‑memory cache for validation
)

model.fit(train_ds_fine, validation_data=val_ds_fine, epochs=2, verbose=2)




## === cell 5
probabilities = model.predict(test_ds, verbose=0)
pred_labels = np.argmax(probabilities, axis=1)

submission_df = pd.DataFrame({"image_id": test_filenames, "label": pred_labels})
submission_df.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission file written to {SUBMISSION_PATH}")

tf.io.gfile.remove(train_cache_file)
