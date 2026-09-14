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

0.8898458748866727

# 6. Current score

0.11584

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11697) has done: 'I remove the failing dependencies (`kaggle_datasets`, external EfficientNet/keras_applications paths, and missing pretrained .h5 loads) that prevent the notebook from running under your current TensorFlow/Protobuf setup. Then I keep your core inference logic (TFRecord input pipeline + simple averaged ensemble + argmax) but build two equivalent EfficientNet models directly from `tf.keras.applications` and run them with ImageNet weights, which is a minimal, legitimate replacement for the missing models that yields a reasonable accuracy baseline. Finally, I fix the TFRecord decode bug (JPEG decode then forced reshape) by resizing properly, and ensure the generated `submission.csv` has the exact required columns and row alignment with `sample_submission.csv`.'
- What this solution (achieved 0.05531) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype` from an incompatible protobuf runtime) by forcing the pure-Python protobuf implementation before TensorFlow is imported, which is the minimal change to unblock execution in this environment. Then I keep your exact inference core (TFRecords → EfficientNetB6/B4 with ImageNet weights → average probs → argmax) but correct a logic issue that can silently scramble predictions: your image and id pipelines are iterated separately while `experimental_deterministic=False` is set in `load_dataset` (and only partially overridden), which can desynchronize IDs from images. I make the test dataset strictly deterministic/ordered end-to-end and derive `test_ids` and `predictions` from the same iteration order to align outputs with `sample_submission.csv`, which should move accuracy sharply upward toward the target without changing model architecture or training. Finally, I keep the submission format exactly as required and ensure `submission.csv` is always written.'
- What this solution (achieved 0.11584) has done: 'I fix the TensorFlow import crash by forcing the pure-Python protobuf implementation *before* any TensorFlow import, which resolves the `MessageFactory.GetPrototype` error in this environment. Then I fix the dataset construction error by removing the unsupported `deterministic=` argument to `TFRecordDataset` (TF 2.18 uses `tf.data.Options().experimental_deterministic` instead). Finally, I keep your exact core inference/ensemble logic but ensure IDs and images are read in the same deterministic dataset pass so predictions align with `sample_submission.csv`, and always write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))



## === cell 1
import tensorflow as tf
import matplotlib.pyplot as plt
from functools import partial
import re
import random

print("TensorFlow:", tf.__version__)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
from tensorflow.keras import layers

NUM_CLASSES = 5
IMAGE_SIZE = [512, 512]


def build_effnet(model_name: str, image_size=(512, 512), num_classes=5):
    inputs = tf.keras.Input(shape=(image_size[0], image_size[1], 3))
    x = layers.Lambda(
        tf.keras.applications.efficientnet.preprocess_input,
        name=f"{model_name}_preprocess",
    )(inputs)

    if model_name == "B6":
        base = tf.keras.applications.EfficientNetB6(
            include_top=False, weights="imagenet", input_tensor=x, pooling="avg"
        )
    elif model_name == "B4":
        base = tf.keras.applications.EfficientNetB4(
            include_top=False, weights="imagenet", input_tensor=x, pooling="avg"
        )
    else:
        raise ValueError("Unsupported model_name")

    outputs = layers.Dense(
        num_classes, activation="softmax", name=f"{model_name}_pred"
    )(base.output)
    model = tf.keras.Model(inputs, outputs, name=f"EfficientNet{model_name}_cassava")
    return model


model_15 = build_effnet("B6", image_size=tuple(IMAGE_SIZE), num_classes=NUM_CLASSES)
model_17 = build_effnet("B4", image_size=tuple(IMAGE_SIZE), num_classes=NUM_CLASSES)



## === cell 3
print(model_15.name, "params:", model_15.count_params())
print(model_17.name, "params:", model_17.count_params())



## === cell 4
test_df = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
print(test_df.head())

AUTOTUNE = tf.data.experimental.AUTOTUNE
GCS_PATH = "/kaggle/input/cassava-leaf-disease-classification"
BATCH_SIZE = 16 * 8
CLASSES = ["0", "1", "2", "3", "4"]


def dataset_sizes(filenames):
    n = [
        int(re.compile(r"-([0-9]*)\.").search(filename).group(1))
        for filename in filenames
    ]
    return int(np.sum(n))


TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + "/test_tfrecords/ld_test*.tfrec")
TEST_FILENAMES = sorted(TEST_FILENAMES)  # keep order stable across runs
NUM_TEST_IMAGES = dataset_sizes(TEST_FILENAMES)
print("Test TFRecords:", len(TEST_FILENAMES), "NUM_TEST_IMAGES:", NUM_TEST_IMAGES)




## === cell 5
def to_float32(image, label):
    return tf.cast(image, tf.float32), label


def decode_img(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0
    return img


def read_tfrecord(example, labeled):
    if labeled:
        TFREC_FORMAT = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "target": tf.io.FixedLenFeature([], tf.int64),
        }
    else:
        TFREC_FORMAT = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_name": tf.io.FixedLenFeature([], tf.string),
        }
    example = tf.io.parse_single_example(example, TFREC_FORMAT)
    img = decode_img(example["image"])
    if labeled:
        label = tf.cast(example["target"], tf.int32)
        return img, label
    else:
        id_num = example["image_name"]
        return img, id_num


def load_dataset(filenames, labeled=True, ordered=False):
    options = tf.data.Options()
    options.experimental_deterministic = bool(ordered)

    dataset = tf.data.TFRecordDataset(
        filenames,
        num_parallel_reads=1 if ordered else AUTOTUNE,
    )
    dataset = dataset.with_options(options)

    dataset = dataset.map(
        partial(read_tfrecord, labeled=labeled),
        num_parallel_calls=1 if ordered else AUTOTUNE,
        deterministic=bool(ordered),
    )
    return dataset


def get_test_data(ordered=False):
    dataset = load_dataset(filenames=TEST_FILENAMES, labeled=False, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=False)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset




## === cell 6
test_ds = get_test_data(ordered=True)
test_ds = test_ds.map(to_float32, num_parallel_calls=AUTOTUNE)

print("Computing predictions...")

image_ds = test_ds.map(lambda image, idnum: image, num_parallel_calls=AUTOTUNE)

prob1 = model_15.predict(image_ds, verbose=1)
prob2 = model_17.predict(image_ds, verbose=1)

probabilities = 0.5 * (prob1 + prob2)
predictions = np.argmax(probabilities, axis=-1).astype(np.int64)

test_ids = []
for _, id_batch in test_ds:
    test_ids.extend(id_batch.numpy().astype("U"))
test_ids = np.array(test_ids)

print("Predictions shape:", predictions.shape, "unique:", np.unique(predictions))
print("IDs shape:", test_ids.shape)

if len(test_ids) != len(predictions):
    raise RuntimeError(
        f"Length mismatch: ids={len(test_ids)} vs preds={len(predictions)}"
    )

if len(test_ids) != len(test_df):
    raise RuntimeError(
        f"ID count mismatch: TFRecords={len(test_ids)} vs sample_submission={len(test_df)}"
    )



## === cell 7
print("Generating submission.csv file...")

sub_pred = pd.DataFrame({"image_id": test_ids, "label": predictions})
sub = test_df[["image_id"]].merge(sub_pred, on="image_id", how="left")

if sub["label"].isna().any():
    mode_label = int(pd.Series(predictions).mode().iloc[0])
    sub["label"] = sub["label"].fillna(mode_label)

sub["label"] = sub["label"].astype(int)

out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print(sub.head())
print("Wrote:", out_path, "rows:", len(sub))
with open(out_path, "r") as f:
    for _ in range(5):
        print(f.readline().rstrip())
