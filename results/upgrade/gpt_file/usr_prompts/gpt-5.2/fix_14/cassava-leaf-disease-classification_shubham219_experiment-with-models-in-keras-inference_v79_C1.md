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
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys
import subprocess
import pandas as pd

SEED = 42
DEBUG = False

BASE_INPUT = "/kaggle/input/cassava-leaf-disease-classification"
TEST_DIR = os.path.join(BASE_INPUT, "test_images")
SAMPLE_SUB_PATH = os.path.join(BASE_INPUT, "sample_submission.csv")
TRAIN_CSV_PATH = os.path.join(BASE_INPUT, "train.csv")
TRAIN_DIR = os.path.join(BASE_INPUT, "train_images")

assert os.path.exists(BASE_INPUT), f"Dataset base path not found: {BASE_INPUT}"
assert os.path.exists(
    SAMPLE_SUB_PATH
), f"sample_submission.csv not found: {SAMPLE_SUB_PATH}"
assert os.path.exists(TEST_DIR), f"test_images dir not found: {TEST_DIR}"
assert os.path.exists(TRAIN_CSV_PATH), f"train.csv not found: {TRAIN_CSV_PATH}"
assert os.path.exists(TRAIN_DIR), f"train_images dir not found: {TRAIN_DIR}"

runner = r"""
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"]="3"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","2")

import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras import layers, Model

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

BASE_INPUT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_DIR = os.path.join(BASE_INPUT, "train_images")
TEST_DIR = os.path.join(BASE_INPUT, "test_images")
TRAIN_CSV_PATH = os.path.join(BASE_INPUT, "train.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_INPUT, "sample_submission.csv")

# Change: use a smaller image size for feasible head-training within time limits; still same model family.
IMG_SIZE = (300, 300)
NUM_CLASSES = 5

# Determinism knobs (best-effort on CPU/GPU).
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

preprocess_input = tf.keras.applications.efficientnet.preprocess_input

train_df = pd.read_csv(TRAIN_CSV_PATH)
train_df["image_id"] = train_df["image_id"].astype(str)
train_df["label"] = train_df["label"].astype(str)  # flow_from_dataframe expects strings for categorical

# Build model: same architecture as you had (EfficientNetB3 frozen + Dropout + Dense softmax head).
base = EfficientNetB3(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    pooling="avg",
)
base.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base(inputs, training=False)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
my_model = Model(inputs, outputs)

# Change: compile and train ONLY the head (base frozen). Loss/metric aligns with accuracy metric.
my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Generators
train_idg = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.1,
)

BATCH_SIZE = 32

train_gen = train_idg.flow_from_dataframe(
    dataframe=train_df,
    directory=TRAIN_DIR,
    x_col="image_id",
    y_col="label",
    subset="training",
    seed=SEED,
    shuffle=True,
    class_mode="categorical",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

val_gen = train_idg.flow_from_dataframe(
    dataframe=train_df,
    directory=TRAIN_DIR,
    x_col="image_id",
    y_col="label",
    subset="validation",
    seed=SEED,
    shuffle=False,
    class_mode="categorical",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

# Change: a small fixed number of epochs to materially improve over random head, without chasing SOTA.
EPOCHS = 4
my_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    verbose=2,
)

# Refit on full training data for best generalization under same head-only training regime.
full_idg = ImageDataGenerator(preprocessing_function=preprocess_input)
full_gen = full_idg.flow_from_dataframe(
    dataframe=train_df,
    directory=TRAIN_DIR,
    x_col="image_id",
    y_col="label",
    seed=SEED,
    shuffle=True,
    class_mode="categorical",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

my_model.fit(
    full_gen,
    epochs=1,
    verbose=2,
)

# Test inference and submission writing (same semantics: argmax over softmax).
test_images = sorted(glob.glob(os.path.join(TEST_DIR, "*.jpg")))
if len(test_images) == 0:
    raise FileNotFoundError(f"No test images found in: {TEST_DIR}")

df_test = pd.DataFrame({"image_id": [os.path.basename(p) for p in test_images]})

test_idg = ImageDataGenerator(preprocessing_function=preprocess_input)
test_gen = test_idg.flow_from_dataframe(
    dataframe=df_test,
    directory=TEST_DIR,
    x_col="image_id",
    y_col=None,
    batch_size=64,
    seed=SEED,
    shuffle=False,
    class_mode=None,
    target_size=IMG_SIZE,
)

pred_test = my_model.predict(test_gen, verbose=1)
pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

final_submission = df_test.copy()
final_submission["label"] = pred_test_labels
final_csv = final_submission[["image_id", "label"]]

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
final_csv = sample_sub[["image_id"]].merge(final_csv, on="image_id", how="left")

if final_csv["label"].isna().any():
    fill_label = int(pd.Series(pred_test_labels).mode().iloc[0])
    final_csv["label"] = final_csv["label"].fillna(fill_label).astype(int)
else:
    final_csv["label"] = final_csv["label"].astype(int)

final_csv.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", final_csv.shape)
print(final_csv.head())
"""

proc = subprocess.run([sys.executable, "-c", runner], capture_output=True, text=True)
print(proc.stdout)
if proc.returncode != 0:
    print(proc.stderr)
    raise RuntimeError("Subprocess TensorFlow runner failed; see stderr above.")

assert os.path.exists(
    "submission.csv"
), "submission.csv was not created by the subprocess runner."



## === cell 1
final_csv = pd.read_csv("submission.csv")
assert list(final_csv.columns) == [
    "image_id",
    "label",
], f"Unexpected columns: {final_csv.columns.tolist()}"
assert len(final_csv) == 2676, f"Unexpected row count: {len(final_csv)}"
final_csv["label"] = final_csv["label"].astype(int)
final_csv.head()



## === cell 2
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
assert sample_sub["image_id"].equals(
    final_csv["image_id"]
), "Submission image_id ordering/content mismatch vs sample_submission.csv"
assert final_csv["label"].notna().all(), "Found NaN labels in submission"
final_csv["label"].value_counts().head()



## === cell 3
print("Ready: submission.csv")
