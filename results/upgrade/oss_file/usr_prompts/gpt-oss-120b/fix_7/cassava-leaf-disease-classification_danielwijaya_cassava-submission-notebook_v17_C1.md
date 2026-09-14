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

0.8821396192203083

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'The fix removes the incompatible external EfficientNet import, replaces the missing pretrained model file with a fresh EfficientNetB5 model built from TensorFlow‑Keras, adds a quick training loop on the provided TFRecord training data, and correctly creates the submission CSV using the test IDs and model predictions. This resolves the import error, the missing model file error, and the undefined‑variable errors while keeping the original data‑pipeline logic.'
- What this solution (achieved 0.0852) has done: 'I set the protobuf implementation flag before importing TensorFlow to avoid the `MessageFactory` attribute error, replace the TFRecord‑based pipeline with a straightforward image‑file pipeline (using the provided CSV for labels), and add a brief fine‑tuning stage (unfreeze the EfficientNet backbone after two epochs) while keeping the original model architecture. This fixes the runtime crash and should raise validation accuracy toward the target score, and finally writes a correctly‑formatted `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import json
import numpy as np
import pandas as pd

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import tensorflow as tf

tf.config.optimizer.set_jit(True)

num_threads = os.cpu_count() or 1
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.threading.set_inter_op_parallelism_threads(num_threads)

from tensorflow.keras.mixed_precision import experimental as mixed_precision

mixed_precision.set_policy("mixed_float16")

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 16 * 8  # frozen training batch size
FINE_BATCH_SIZE = 16  # fine‑tuning batch size
IMAGE_SIZE = [512, 512]
NUM_CLASSES = 5
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

BASE_PATH = os.path.abspath("../input/cassava-leaf-disease-classification")
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train_images")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test_images")
SAMPLE_SUBMISSION = os.path.join(BASE_PATH, "sample_submission.csv")
SUBMISSION_PATH = "submission.csv"




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3248857809.py in <cell line: 0>()
----> 1 train_df = pd.read_csv(TRAIN_CSV)
      2 train_df["filepath"] = train_df["image_id"].apply(
      3     lambda x: os.path.join(TRAIN_IMG_DIR, x)
      4 )
      5 

NameError: name 'TRAIN_CSV' is not defined

## === cell 2
def decode_and_preprocess(path, label=None):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0  # normalize to [0,1]
    if label is None:
        return image
    else:
        return image, label


def prepare_dataset(paths, labels=None, batch_size=BATCH_SIZE, shuffle=False):
    """
    Build a tf.data pipeline with caching to avoid re‑reading and re‑processing
    images on every epoch. Caching occurs after decoding/pre‑processing and
    before shuffling (so shuffling still varies each epoch).
    """
    ds = (
        tf.data.Dataset.from_tensor_slices((paths, labels))
        if labels is not None
        else tf.data.Dataset.from_tensor_slices(paths)
    )
    ds = ds.map(decode_and_preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()  # <-- cache pre‑processed tensors
    if shuffle:
        ds = ds.shuffle(buffer_size=1024, seed=SEED)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


train_ds = prepare_dataset(
    train_paths, train_labels, batch_size=BATCH_SIZE, shuffle=True
)
val_ds = prepare_dataset(val_paths, val_labels, batch_size=BATCH_SIZE, shuffle=False)
test_ds = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(lambda p: decode_and_preprocess(p), num_parallel_calls=AUTOTUNE)
    .cache()  # cache test images as well
    .batch(FINE_BATCH_SIZE)
    .prefetch(AUTOTUNE)
)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3152144324.py in <cell line: 0>()
     10 
     11 
---> 12 def prepare_dataset(paths, labels=None, batch_size=BATCH_SIZE, shuffle=False):
     13     """
     14     Build a tf.data pipeline with caching to avoid re‑reading and re‑processing

NameError: name 'BATCH_SIZE' is not defined

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




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/604078108.py in <cell line: 0>()
      1 base = tf.keras.applications.EfficientNetB5(
----> 2     include_top=False, weights="imagenet", input_shape=(*IMAGE_SIZE, 3)
      3 )
      4 base.trainable = False  # frozen stage
      5 

NameError: name 'IMAGE_SIZE' is not defined

## === cell 4
model.fit(train_ds, validation_data=val_ds, epochs=5, verbose=2)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3900139454.py in <cell line: 0>()
----> 1 model.fit(train_ds, validation_data=val_ds, epochs=5, verbose=2)
      2 
      3 

NameError: name 'model' is not defined

## === cell 5
base.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

train_ds_fine = prepare_dataset(
    train_paths, train_labels, batch_size=FINE_BATCH_SIZE, shuffle=True
)
val_ds_fine = prepare_dataset(
    val_paths, val_labels, batch_size=FINE_BATCH_SIZE, shuffle=False
)

model.fit(train_ds_fine, validation_data=val_ds_fine, epochs=3, verbose=2)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4002125778.py in <cell line: 0>()
----> 1 base.trainable = True
      2 model.compile(
      3     optimizer=tf.keras.optimizers.Adam(1e-4),
      4     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      5     metrics=["accuracy"],

NameError: name 'base' is not defined

## === cell 6
probabilities = model.predict(test_ds, verbose=0)
pred_labels = np.argmax(probabilities, axis=1)

submission_df = pd.DataFrame({"image_id": test_filenames, "label": pred_labels})
submission_df.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission file written to {SUBMISSION_PATH}")

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2539156712.py in <cell line: 0>()
----> 1 probabilities = model.predict(test_ds, verbose=0)
      2 pred_labels = np.argmax(probabilities, axis=1)
      3 
      4 submission_df = pd.DataFrame({"image_id": test_filenames, "label": pred_labels})
      5 submission_df.to_csv(SUBMISSION_PATH, index=False)

NameError: name 'model' is not defined
