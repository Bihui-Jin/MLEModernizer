# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

# 5. Target score

0.8584164400120883

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import json
import random
import numpy as np
import pandas as pd



## === cell 1
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images/"

assert os.path.isdir(BASE_DIR), f"BASE_DIR not found: {BASE_DIR}"
assert os.path.isdir(TRAIN_DIR), f"TRAIN_DIR not found: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"TEST_DIR not found: {TEST_DIR}"



## === cell 2
from PIL import Image

import tensorflow as tf
import keras
from keras import layers

keras.backend.clear_session()
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass



## === cell 3
with open(os.path.join(BASE_DIR, "label_num_to_disease_map.json")) as file:
    map_classes = json.loads(file.read())
print(json.dumps(map_classes, indent=2))

label_list = sorted([int(k) for k in map_classes.keys()])
NUM_CLASSES = len(label_list)
assert NUM_CLASSES == 5, f"Expected 5 classes, got {NUM_CLASSES}"



## === cell 4
input_files = os.listdir(TRAIN_DIR)
print(f"Number of train images: {len(input_files)}")



## === cell 5
IMG_HEIGHT = 300
IMG_WIDTH = 300
batch_size = 16

PRE_TRAINED_MODEL = "../input/unionmodelv05/Cassava_Best_UnitedModel_V05.hdf5"
print("Pretrained model exists?:", os.path.exists(PRE_TRAINED_MODEL))

RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS



## === cell 6
train_df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
train_df["filepath"] = train_df["image_id"].apply(lambda x: os.path.join(TRAIN_DIR, x))
train_df["label"] = train_df["label"].astype(int)

exists_mask = [os.path.exists(p) for p in train_df["filepath"].values.tolist()]
train_df = train_df.loc[exists_mask].reset_index(drop=True)

print(
    "Train rows:",
    len(train_df),
    "Unique labels:",
    sorted(train_df["label"].unique().tolist()),
)



## === cell 7
sample_sub = pd.read_csv(os.path.join(BASE_DIR, "sample_submission.csv"))
test_df = sample_sub[["image_id"]].copy()
test_df["filepath"] = test_df["image_id"].apply(lambda x: os.path.join(TEST_DIR, x))

exists_mask = [os.path.exists(p) for p in test_df["filepath"].values.tolist()]
test_df = test_df.loc[exists_mask].reset_index(drop=True)

print("Test rows:", len(test_df))




## === cell 8
def decode_and_resize(path, label=None, training=False):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH], method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0

    if training:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, max_delta=0.12)
        img = tf.image.random_contrast(img, lower=0.85, upper=1.15)

    if label is None:
        return img
    return img, tf.cast(label, tf.int32)


from sklearn.model_selection import train_test_split

train_paths = train_df["filepath"].values
train_labels = train_df["label"].values
x_tr, x_va, y_tr, y_va = train_test_split(
    train_paths, train_labels, test_size=0.2, random_state=42, stratify=train_labels
)

AUTOTUNE = tf.data.AUTOTUNE

ds_train = tf.data.Dataset.from_tensor_slices((x_tr, y_tr))
ds_train = ds_train.shuffle(2048, seed=42, reshuffle_each_iteration=True)

ds_train = ds_train.map(
    lambda p, y: (decode_and_resize(p, y, training=False)[0], y),
    num_parallel_calls=AUTOTUNE,
).cache()
ds_train = ds_train.map(
    lambda img, y: (
        decode_and_resize(tf.constant("", dtype=tf.string), y, training=True)[0]
        if False
        else (
            tf.image.random_contrast(
                tf.image.random_brightness(
                    tf.image.random_flip_up_down(tf.image.random_flip_left_right(img)),
                    max_delta=0.12,
                ),
                lower=0.85,
                upper=1.15,
            ),
            tf.cast(y, tf.int32),
        )
    ),
    num_parallel_calls=AUTOTUNE,
)
ds_train = ds_train.batch(batch_size).prefetch(AUTOTUNE)

ds_valid = tf.data.Dataset.from_tensor_slices((x_va, y_va))
ds_valid = ds_valid.map(
    lambda p, y: decode_and_resize(p, y, training=False), num_parallel_calls=AUTOTUNE
).cache()
ds_valid = ds_valid.batch(batch_size).prefetch(AUTOTUNE)



## === cell 9
inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()



## === cell 10
EPOCHS = 8
history = model.fit(ds_train, validation_data=ds_valid, epochs=EPOCHS, verbose=2)




## === cell 11
def load_single_image_np(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT), RESAMPLE)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def augment_np(img):
    out = img
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.2:
        out = np.flipud(out)
    if random.random() < 0.5:
        b = (random.random() * 0.24) - 0.12
        out = np.clip(out + b, 0.0, 1.0)
    if random.random() < 0.5:
        c = 1.0 + (random.random() * 0.30 - 0.15)
        mean = out.mean(axis=(0, 1), keepdims=True)
        out = np.clip((out - mean) * c + mean, 0.0, 1.0)
    return out


def get_augmented_images(image_id, n_aug=5):
    image_path = os.path.join(TEST_DIR, image_id)
    base = load_single_image_np(image_path)
    image_list = [base]
    for _ in range(n_aug):
        image_list.append(augment_np(base))
    return [np.expand_dims(im.astype(np.float32), axis=0) for im in image_list]




## === cell 12
GLOBAL_SEED = 42
N_AUG = 5
N_VIEWS = 1 + N_AUG


def _tta_views_tf(img, idx):
    views = [img]

    for j in range(N_AUG):
        seed_base = tf.stack(
            [tf.cast(GLOBAL_SEED, tf.int32), tf.cast(idx * 1000 + j, tf.int32)], axis=0
        )

        out = img

        r = tf.random.stateless_uniform(
            [], seed=seed_base + tf.constant([1, 11], tf.int32)
        )
        out = tf.cond(r < 0.5, lambda: tf.image.flip_left_right(out), lambda: out)

        r = tf.random.stateless_uniform(
            [], seed=seed_base + tf.constant([2, 22], tf.int32)
        )
        out = tf.cond(r < 0.2, lambda: tf.image.flip_up_down(out), lambda: out)

        r = tf.random.stateless_uniform(
            [], seed=seed_base + tf.constant([3, 33], tf.int32)
        )
        delta = tf.random.stateless_uniform(
            [],
            seed=seed_base + tf.constant([4, 44], tf.int32),
            minval=-0.12,
            maxval=0.12,
        )
        out = tf.cond(
            r < 0.5, lambda: tf.clip_by_value(out + delta, 0.0, 1.0), lambda: out
        )

        r = tf.random.stateless_uniform(
            [], seed=seed_base + tf.constant([5, 55], tf.int32)
        )
        c = tf.random.stateless_uniform(
            [],
            seed=seed_base + tf.constant([6, 66], tf.int32),
            minval=0.85,
            maxval=1.15,
        )

        def _apply_contrast():
            mean = tf.reduce_mean(out, axis=[0, 1], keepdims=True)
            return tf.clip_by_value((out - mean) * c + mean, 0.0, 1.0)

        out = tf.cond(r < 0.5, _apply_contrast, lambda: out)

        views.append(out)

    return tf.stack(views, axis=0)  # [6,H,W,3]


test_image_ids = test_df["image_id"].values.tolist()
test_paths = test_df["filepath"].values.tolist()

ds_test = tf.data.Dataset.from_tensor_slices((test_image_ids, test_paths))
ds_test = ds_test.enumerate()  # (idx, (image_id, path))


def _load_and_tta(idx, pair):
    image_id, path = pair
    img = decode_and_resize(path, label=None, training=False)  # [H,W,3]
    views = _tta_views_tf(img, idx)  # [6,H,W,3]
    return image_id, views


ds_test = ds_test.map(_load_and_tta, num_parallel_calls=AUTOTUNE)
ds_test = ds_test.batch(8).prefetch(AUTOTUNE)  # batch of images; each has 6 views

test_results_list = []
for batch in ds_test:
    image_ids_b, views_b = batch  # image_ids_b: [B], views_b: [B,6,H,W,3]
    b = tf.shape(views_b)[0]
    views_flat = tf.reshape(views_b, [b * N_VIEWS, IMG_HEIGHT, IMG_WIDTH, 3])

    preds_flat = model.predict(views_flat, verbose=0)  # [B*6, NUM_CLASSES]
    preds = preds_flat.reshape((-1, N_VIEWS, NUM_CLASSES))  # [B,6,C]
    preds_mean = preds.mean(axis=1)  # [B,C]
    labels = preds_mean.argmax(axis=1).astype(int)

    ids_np = image_ids_b.numpy()
    for i in range(len(labels)):
        image_id = (
            ids_np[i].decode("utf-8")
            if isinstance(ids_np[i], (bytes, np.bytes_))
            else str(ids_np[i])
        )
        test_results_list.append({"image_id": image_id, "label": int(labels[i])})

submission = pd.DataFrame(test_results_list)

submission = sample_sub[["image_id"]].merge(submission, on="image_id", how="left")
assert submission["label"].notnull().all(), "Some test image_ids were not predicted."
submission["label"] = submission["label"].astype(int)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(submission.head())
print(submission.shape)



## === cell 13
chk = pd.read_csv("submission.csv")
assert list(chk.columns) == ["image_id", "label"], chk.columns
assert len(chk) == len(sample_sub), (len(chk), len(sample_sub))
chk.head(3)
