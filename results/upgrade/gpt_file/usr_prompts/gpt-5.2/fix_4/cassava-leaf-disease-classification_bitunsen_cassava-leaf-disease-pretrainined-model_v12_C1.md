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
import json
import random
import numpy as np
import pandas as pd

from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

print("TF:", tf.__version__)
print("Keras:", keras.__version__)



## === cell 1
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images/"

TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")
MAP_JSON = os.path.join(BASE_DIR, "label_num_to_disease_map.json")

assert os.path.exists(TRAIN_CSV), TRAIN_CSV
assert os.path.exists(SAMPLE_SUB), SAMPLE_SUB
assert os.path.exists(TRAIN_DIR), TRAIN_DIR
assert os.path.exists(TEST_DIR), TEST_DIR



## === cell 2
with open(MAP_JSON, "r") as f:
    map_classes = json.load(f)
print(json.dumps(map_classes, indent=2))
label_list = sorted([int(k) for k in map_classes.keys()])
NUM_CLASSES = len(label_list)
print("NUM_CLASSES:", NUM_CLASSES, "labels:", label_list)



## === cell 3
IMG_HEIGHT = 500
IMG_WIDTH = 500
batch_size = 16

PRE_TRAINED_MODEL = "../input/unitedmodelv01/Cassava_Best_UnitedModel_V01.hdf5"
print("Pretrained exists?", os.path.exists(PRE_TRAINED_MODEL), PRE_TRAINED_MODEL)



## === cell 4
train_df = pd.read_csv(TRAIN_CSV)
sample_sub_df = pd.read_csv(SAMPLE_SUB)

assert set(train_df.columns) == {"image_id", "label"}
assert set(sample_sub_df.columns) == {"image_id", "label"}

print("Train rows:", len(train_df), "Test rows:", len(sample_sub_df))

missing = 0
for fn in train_df["image_id"].head(50):
    if not os.path.exists(os.path.join(TRAIN_DIR, fn)):
        missing += 1
print("Missing in first 50 train images:", missing)



## === cell 5
_RESAMPLE = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS


def load_single_image(data_type, image_id):
    if data_type == "TEST_DATA":
        image_path = os.path.join(TEST_DIR, image_id)
    else:
        image_path = os.path.join(TRAIN_DIR, image_id)
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT), _RESAMPLE)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # float32 0..1
    return arr




## === cell 6
def _tf_load_image_from_path(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)  # dataset is .jpg
    img = tf.image.resize(
        img,
        [IMG_HEIGHT, IMG_WIDTH],
        method=tf.image.ResizeMethod.LANCZOS3,
        antialias=True,
    )
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape([IMG_HEIGHT, IMG_WIDTH, 3])
    return img


def _tf_read_train_example(image_id, label):
    path = tf.strings.join([tf.constant(TRAIN_DIR), image_id])
    x = _tf_load_image_from_path(path)
    y = tf.cast(label, tf.int32)
    y.set_shape([])
    return x, y


@tf.function
def _augment(x, y):
    x = tf.image.random_flip_left_right(x, seed=SEED)
    x = tf.image.random_flip_up_down(x, seed=SEED)
    x = tf.image.random_brightness(x, max_delta=0.15, seed=SEED)
    x = tf.clip_by_value(x, 0.0, 1.0)
    return x, y


train_df_shuf = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
val_frac = 0.1
val_size = int(len(train_df_shuf) * val_frac)
val_df = train_df_shuf.iloc[:val_size].reset_index(drop=True)
trn_df = train_df_shuf.iloc[val_size:].reset_index(drop=True)
print("Train split:", len(trn_df), "Val split:", len(val_df))

trn_ds = tf.data.Dataset.from_tensor_slices(
    (trn_df["image_id"].astype(str).values, trn_df["label"].values)
)
val_ds = tf.data.Dataset.from_tensor_slices(
    (val_df["image_id"].astype(str).values, val_df["label"].values)
)

AUTOTUNE = tf.data.AUTOTUNE

trn_ds = (
    trn_ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    .map(_tf_read_train_example, num_parallel_calls=AUTOTUNE)
    .map(_augment, num_parallel_calls=AUTOTUNE)
    .batch(batch_size, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

val_ds = (
    val_ds.map(_tf_read_train_example, num_parallel_calls=AUTOTUNE)
    .cache()
    .batch(batch_size, drop_remainder=False)
    .prefetch(AUTOTUNE)
)



## === cell 7
inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2, seed=SEED)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()



## === cell 8
EPOCHS = 3
history = model.fit(trn_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)



## === cell 9
from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_datagen = ImageDataGenerator(
    rotation_range=45,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest",
    shear_range=0.1,
    height_shift_range=0.1,
    width_shift_range=0.1,
)


def get_augmented_images(image_id, n_aug=5):
    x = load_single_image("TEST_DATA", image_id)
    x = np.expand_dims(x, axis=0)  # (1,H,W,3)

    it = test_datagen.flow(x, batch_size=1, shuffle=False, seed=SEED)
    image_list = [x]
    for _ in range(n_aug):
        batch = next(it)
        batch = np.clip(batch.astype(np.float32), 0.0, 1.0)
        image_list.append(batch)
    return image_list




## === cell 10
test_ids = sample_sub_df["image_id"].astype(str).values
n_test = len(test_ids)
n_aug = 5
tta_total = 1 + n_aug

X0 = np.empty((n_test, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
for i, image_id in enumerate(test_ids):
    X0[i] = load_single_image("TEST_DATA", image_id)
    if (i + 1) % 500 == 0:
        print(f"Loaded {i+1}/{n_test} test images")

probs_sum = np.zeros((n_test, NUM_CLASSES), dtype=np.float32)

PRED_BS = 32

probs_sum += model.predict(X0, batch_size=PRED_BS, verbose=0).astype(np.float32)

GEN_BS = 32
for k in range(n_aug):
    gen = test_datagen.flow(X0, batch_size=GEN_BS, shuffle=False, seed=SEED + k)
    yk = model.predict(gen, steps=int(np.ceil(n_test / GEN_BS)), verbose=0).astype(
        np.float32
    )
    yk = yk[:n_test]
    probs_sum += yk
    print(f"Augmented pass {k+1}/{n_aug} done")

probs_mean = probs_sum / float(tta_total)
test_results = probs_mean.argmax(axis=1).astype(int).tolist()

submission = pd.DataFrame({"image_id": test_ids, "label": test_results})
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())



## === cell 11
sub_check = pd.read_csv("submission.csv")
assert list(sub_check.columns) == ["image_id", "label"]
assert len(sub_check) == len(sample_sub_df)
assert sub_check["label"].between(0, NUM_CLASSES - 1).all()
print("submission.csv OK")
