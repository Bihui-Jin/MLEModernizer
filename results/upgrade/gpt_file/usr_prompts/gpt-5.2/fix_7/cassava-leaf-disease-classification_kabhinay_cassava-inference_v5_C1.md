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
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow import keras
except AttributeError as e:
    raise RuntimeError(
        "TensorFlow failed to import due to a protobuf compatibility issue. "
        "The notebook sets PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python early, "
        "but this runtime still raised an AttributeError during TF import."
    ) from e

tf.random.set_seed(42)
np.random.seed(42)

print("TF version:", tf.__version__)
print("Keras version:", keras.__version__)




## === cell 1
def acc_gambler(y_true, y_pred):
    y_temp = y_pred[:, 1:]
    count = tf.constant((0,))
    for i in range(len(y_true)):
        tf.autograph.experimental.set_loop_options(
            shape_invariants=[(count, tf.TensorShape([None]))]
        )
        if tf.math.argmax(y_temp[i]) == tf.math.argmax(y_true[i]):
            count = tf.math.add(count, 1)
    return float(count) / float(len(y_true))




## === cell 2
def loss_gambler(y_true, y_pred):
    """
    Bugfix: avoid iterating over a symbolic tensor (tf.range(num_classes)) in graph mode.
    Keeps same math/semantics as the original loop:
      sum_i [ (-1/n) * y_true[:,i] * log(y_temp[:,i] + f0/lamb) ]
    """
    y_temp = y_pred[:, 1:]  # (N, C)
    f0 = y_pred[:, 0:1]  # (N, 1) keep dims for broadcasting

    K = tf.keras.backend

    sum_y = K.sum(y_temp)
    sum_y2 = K.sum(tf.math.multiply(y_temp, y_temp))
    lamb = tf.math.divide(tf.math.multiply(sum_y, sum_y), sum_y2)

    n = tf.cast(tf.shape(y_true)[0], tf.float32)

    add_term = tf.cast(f0, tf.float32) / tf.cast(lamb, tf.float32)  # (N,1)
    inside = tf.cast(y_temp, tf.float32) + add_term  # (N,C)
    per_entry = (-1.0) * (1.0 / n) * tf.cast(y_true, tf.float32) * K.log(inside)
    loss = tf.reduce_sum(per_entry)

    return loss




## === cell 3
IMG_SIZE = 512
NUM_CLASSES = 5

base = keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling="avg",
)
base.trainable = False

inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
x = keras.applications.efficientnet.preprocess_input(inp)
x = base(x, training=False)
out = keras.layers.Dense(1 + NUM_CLASSES, name="gambler_logits")(x)

model_v3 = keras.Model(inputs=inp, outputs=out, name="gambler_like_efficientnetb0")
print("Built model:", model_v3.name)



## === cell 4
model_v3.summary()



## === cell 5
import matplotlib.pyplot as plt  # kept as in original environment
from tensorflow.keras.preprocessing.image import ImageDataGenerator



## === cell 6
BASE_PATH = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV_PATH = os.path.join(BASE_PATH, "train.csv")
TRAIN_DIR = os.path.join(BASE_PATH, "train_images")
TEST_DIR = os.path.join(BASE_PATH, "test_images")
SAMPLE_SUB_PATH = os.path.join(BASE_PATH, "sample_submission.csv")

for p in [TRAIN_CSV_PATH, TRAIN_DIR, TEST_DIR, SAMPLE_SUB_PATH]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Required path not found: {p}")

train_df = pd.read_csv(TRAIN_CSV_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

train_df["label"] = train_df["label"].astype(str)

print("train.csv shape:", train_df.shape)
print("train.csv head:\n", train_df.head())
print("sample_submission shape:", sample_sub.shape)
print("sample_submission head:\n", sample_sub.head())



## === cell 7
datagen_train = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.05,
    height_shift_range=0.05,
)

class_list = [str(i) for i in range(NUM_CLASSES)]

train_flow = datagen_train.flow_from_dataframe(
    train_df,
    directory=TRAIN_DIR,
    x_col="image_id",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=16,
    class_mode="categorical",
    classes=class_list,
    shuffle=True,
    seed=42,
)

model_v3.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=loss_gambler,
)

steps_per_epoch = max(1, int(np.ceil(train_flow.n / train_flow.batch_size)))
model_v3.fit(
    train_flow,
    epochs=3,
    steps_per_epoch=steps_per_epoch,
    verbose=1,
)

base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

model_v3.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss=loss_gambler,
)

model_v3.fit(
    train_flow,
    epochs=1,
    steps_per_epoch=steps_per_epoch,
    verbose=1,
)



## === cell 8
test_v3 = sample_sub[["image_id"]].copy()

test_datagen_v3 = ImageDataGenerator()

test_generator_v3 = test_datagen_v3.flow_from_dataframe(
    test_v3,
    directory=TEST_DIR,
    x_col="image_id",
    y_col=None,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=16,
    class_mode=None,
    shuffle=False,
)



## === cell 9
pred_v3 = model_v3.predict(
    test_generator_v3, verbose=1, steps=int(np.ceil(len(test_v3) / 16))
)

pred_v3 = np.asarray(pred_v3)
print("Raw prediction shape:", pred_v3.shape)

if pred_v3.ndim != 2 or pred_v3.shape[1] < 2:
    raise ValueError(f"Unexpected prediction shape {pred_v3.shape}; expected (N, >=2).")

if pred_v3.shape[0] != len(test_v3):
    pred_v3 = pred_v3[: len(test_v3)]
    print("Trimmed predictions to:", pred_v3.shape)



## === cell 10
predicted_class_indices_v3 = np.argmax(pred_v3[:, 1:], axis=1).astype(int)

submission = pd.DataFrame(
    {"image_id": test_v3["image_id"].values, "label": predicted_class_indices_v3}
)

if submission.shape[0] != sample_sub.shape[0]:
    raise ValueError("Submission row count does not match sample_submission.")
if list(submission.columns) != ["image_id", "label"]:
    raise ValueError("Submission columns are incorrect.")
if submission["image_id"].isna().any():
    raise ValueError("Found NaN image_id in submission.")

out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote submission to:", out_path)
print(submission.head())
