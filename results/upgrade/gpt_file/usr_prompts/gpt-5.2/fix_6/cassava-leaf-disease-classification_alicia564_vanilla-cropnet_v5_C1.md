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

3.13

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

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

print("TF version:", tf.__version__)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE




## === cell 1
from sklearn.model_selection import train_test_split

from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

BASE_PATH = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV_PATH = f"{BASE_PATH}/train.csv"
TRAIN_IMG_DIR = f"{BASE_PATH}/train_images"
SAMPLE_SUB_PATH = f"{BASE_PATH}/sample_submission.csv"
TEST_IMG_DIR = f"{BASE_PATH}/test_images"
LABEL_MAP_PATH = f"{BASE_PATH}/label_num_to_disease_map.json"

label_to_disease = pd.read_json(LABEL_MAP_PATH, typ="series")
train_csv = pd.read_csv(TRAIN_CSV_PATH)

train_csv["disease"] = train_csv["label"].map(label_to_disease).astype(str)
train_csv["path"] = TRAIN_IMG_DIR.rstrip("/") + "/" + train_csv["image_id"]
train_csv["label"] = train_csv["label"].astype(str)

train, valid = train_test_split(
    train_csv,
    test_size=0.2,
    stratify=train_csv["label"],
    random_state=SEED,
)

classes = sorted(train_csv["disease"].unique().tolist())
class_indices = {c: i for i, c in enumerate(classes)}
NUM_CLASSES = len(classes)

print("Classes found (tf.data):", class_indices)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def _read_and_decode(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)  # cassava images are jpg
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    return img


def _augment_like_imagedatagenerator(img):

    img = tf.image.stateless_random_flip_left_right(img, seed=(SEED, 1))
    img = tf.image.stateless_random_flip_up_down(img, seed=(SEED, 2))

    angle = tf.random.stateless_uniform(
        [], seed=(SEED, 3), minval=-45.0, maxval=45.0
    ) * (np.pi / 180.0)

    tx = tf.random.stateless_uniform(
        [], seed=(SEED, 4), minval=-0.2, maxval=0.2
    ) * tf.cast(IMG_SIZE[0], tf.float32)
    ty = tf.random.stateless_uniform(
        [], seed=(SEED, 5), minval=-0.2, maxval=0.2
    ) * tf.cast(IMG_SIZE[1], tf.float32)

    shear = tf.random.stateless_uniform([], seed=(SEED, 6), minval=-0.2, maxval=0.2)

    zoom = tf.random.stateless_uniform([], seed=(SEED, 7), minval=0.8, maxval=1.2)

    cos_a = tf.math.cos(angle) / zoom
    sin_a = tf.math.sin(angle) / zoom

    shear_mat = tf.stack([1.0, tf.math.tan(shear), 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    rot = tf.stack([cos_a, -sin_a, 0.0, sin_a, cos_a, 0.0, 0.0, 0.0])

    def to33(t8):
        a0, a1, a2, b0, b1, b2, c0, c1 = tf.unstack(t8)
        return tf.reshape(tf.stack([a0, a1, a2, b0, b1, b2, c0, c1, 1.0]), (3, 3))

    def to8(m33):
        a0, a1, a2, b0, b1, b2, c0, c1, _ = tf.unstack(tf.reshape(m33, (-1,)))
        return tf.stack([a0, a1, a2, b0, b1, b2, c0, c1])

    m = tf.linalg.matmul(to33(rot), to33(shear_mat))
    t = to8(m)

    cx = (IMG_SIZE[1] - 1) / 2.0
    cy = (IMG_SIZE[0] - 1) / 2.0

    a0, a1, a2, b0, b1, b2, c0, c1 = tf.unstack(t)
    a2 = a2 + cx - (a0 * cx + a1 * cy) - ty  # note swapped because x/y axis handling
    b2 = b2 + cy - (b0 * cx + b1 * cy) - tx

    t = tf.stack([a0, a1, a2, b0, b1, b2, c0, c1])[None, :]

    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[None, ...],
        transforms=t,
        output_shape=tf.constant([IMG_SIZE[0], IMG_SIZE[1]], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="NEAREST",
        fill_value=0.0,
    )[0]
    return img


def _preprocess(img):
    return preprocess_input(img)


def make_dataset(df, training):
    paths = df["path"].values
    diseases = df["disease"].values

    y = np.array([class_indices[d] for d in diseases], dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, y))
    if training:
        ds = ds.shuffle(buffer_size=len(df), seed=SEED, reshuffle_each_iteration=True)

    def _decode(path, label):
        img = _read_and_decode(path)
        return img, label

    ds = ds.map(_decode, num_parallel_calls=AUTOTUNE).cache()

    if training:

        def _aug(img, label):
            img = _augment_like_imagedatagenerator(img)
            return img, label

        ds = ds.map(_aug, num_parallel_calls=AUTOTUNE)

    def _prep(img, label):
        img = _preprocess(img)
        label_oh = tf.one_hot(label, NUM_CLASSES, dtype=tf.float32)
        return img, label_oh

    ds = ds.map(_prep, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_dataset(train, training=True)
valid_ds = make_dataset(valid, training=False)

steps_per_epoch = (len(train) + BATCH_SIZE - 1) // BATCH_SIZE
validation_steps = (len(valid) + BATCH_SIZE - 1) // BATCH_SIZE




## === cell 2
base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
preds = Dense(NUM_CLASSES, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=preds)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

EPOCHS = 3

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)




## === cell 3
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
sample_sub["path"] = TEST_IMG_DIR.rstrip("/") + "/" + sample_sub["image_id"]

test_paths = sample_sub["path"].values

test_ds = tf.data.Dataset.from_tensor_slices(test_paths)


def _decode_test(path):
    img = _read_and_decode(path)
    img = _preprocess(img)
    return img


test_ds = (
    test_ds.map(_decode_test, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

probs = model.predict(
    test_ds,
    verbose=1,
)

inv_class_indices = {v: k for k, v in class_indices.items()}  # idx -> disease_str
disease_to_labelid = {
    str(v): int(k) for k, v in label_to_disease.to_dict().items()
}  # disease_str -> label_id

pred_class_idx = np.argmax(probs, axis=1).astype(int)
pred_disease = [inv_class_indices[i] for i in pred_class_idx]
pred_labels = np.array([disease_to_labelid[str(d)] for d in pred_disease], dtype=int)

submission_df = pd.DataFrame(
    {
        "image_id": sample_sub["image_id"].values,
        "label": pred_labels,
    }
)

submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)
print("Submission file created:", submission_path)
print(submission_df.head())
print("Submission shape:", submission_df.shape)
print(
    "Label distribution:\n",
    submission_df["label"].value_counts(dropna=False).sort_index(),
)
