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
import numpy as np
import pandas as pd



## === cell 1
import json
from PIL import Image

if hasattr(Image, "Resampling"):
    PIL_RESAMPLE = Image.Resampling.LANCZOS
else:
    PIL_RESAMPLE = Image.LANCZOS



## === cell 2
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images/"



## === cell 3
with open(os.path.join(BASE_DIR, "label_num_to_disease_map.json")) as file:
    map_classes = json.loads(file.read())

print(json.dumps(map_classes, indent=2))



## === cell 4
label_list = [int(key) for key in map_classes.keys()]
label_list



## === cell 5
input_files = os.listdir(os.path.join(BASE_DIR, "train_images"))
print(f"Number of train images: {len(input_files)}")



## === cell 6
IMG_HEIGHT = 300
IMG_WIDTH = 300
batch_size = 16
PRE_TRAINED_MODEL = "../input/unionmodelv08/Cassava_Best_UnitedModel_V08.hdf5"



## === cell 7
from albumentations import (
    Compose,
    HorizontalFlip,
    VerticalFlip,
    RandomBrightnessContrast,
    CenterCrop,
    ShiftScaleRotate,
    ToFloat,
)

AUGMENTATIONS_TRAIN = Compose(
    [
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        CenterCrop(p=1.0, height=IMG_HEIGHT, width=IMG_WIDTH),
        ShiftScaleRotate(
            p=0.5,
            shift_limit=0.0,
            scale_limit=(0.5, 1.50),
            rotate_limit=15,
            interpolation=0,
            border_mode=0,
        ),
        ToFloat(max_value=255.0),
    ]
)

AUGMENTATIONS_TEST = Compose([ToFloat(max_value=255.0)])



## === cell 8
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_inter_op_parallelism_threads(0)
    tf.config.threading.set_intra_op_parallelism_threads(0)
except Exception:
    pass



## === cell 9
from tensorflow.keras.utils import Sequence
import random


def load_single_image(data_type, image_id):
    if data_type == "TEST_DATA":
        image_path = os.path.join(TEST_DIR, image_id)
    else:
        image_path = os.path.join(TRAIN_DIR, image_id)

    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)  # uint8
    img = tf.image.resize(
        img,
        [IMG_HEIGHT, IMG_WIDTH],
        method=tf.image.ResizeMethod.LANCZOS3,
        antialias=True,
    )  # float32
    return np.ascontiguousarray(img.numpy())


class AugmentedImageSequence(Sequence):
    def __init__(self, mode, data_set_type, x_set, y_set, batch_size, augmentations):
        self.mode = mode
        self.data_type = data_set_type
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.augment = augmentations

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]

        if self.mode == "TEST":
            batch_y = []
        else:
            batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]

        bs = len(batch_x)
        img_array = np.empty((bs, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)

        for i, x in enumerate(batch_x):
            if self.data_type == "TRAIN_DATA":
                random_num = random.uniform(0, 1)
                if random_num > 0.5:
                    img_data = self.augment(image=load_single_image(self.data_type, x))[
                        "image"
                    ]
                else:
                    img_data = (
                        load_single_image(self.data_type, x).astype(np.float32) / 255.0
                    )
            else:
                if self.data_type == "VALIDATE_DATA":
                    img_data = self.augment(image=load_single_image(self.data_type, x))[
                        "image"
                    ]
                else:
                    img_data = (
                        load_single_image(self.data_type, x).astype(np.float32) / 255.0
                    )

            img_data = img_data.astype(np.float32, copy=False)
            if img_data.max() > 1.5:
                img_data = img_data / 255.0
            img_array[i] = img_data

        return img_array, np.array(batch_y)




## === cell 10
test_filenames = os.listdir(TEST_DIR)
test_df = pd.DataFrame({"image_id": test_filenames})
test_samples = test_df.shape[0]
test_samples



## === cell 11
test_gen = AugmentedImageSequence(
    "TEST",
    "TEST_DATA",
    test_df["image_id"].values,
    None,
    batch_size,
    augmentations=AUGMENTATIONS_TEST,
)



## === cell 12
from tensorflow.keras.models import load_model


def build_fallback_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), n_classes=5):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


model = None
if os.path.exists(PRE_TRAINED_MODEL):
    model = load_model(PRE_TRAINED_MODEL)
    model.summary()
else:
    print(f"Pretrained model not found at: {PRE_TRAINED_MODEL}")
    print(
        "Training a small fallback CNN so we can generate a valid submission end-to-end."
    )
    train_csv_path = os.path.join(BASE_DIR, "train.csv")
    train_df = pd.read_csv(train_csv_path)

    train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    val_frac = 0.1
    n_val = int(len(train_df) * val_frac)
    val_df = train_df.iloc[:n_val].reset_index(drop=True)
    trn_df = train_df.iloc[n_val:].reset_index(drop=True)

    trn_gen = AugmentedImageSequence(
        "TRAIN",
        "TRAIN_DATA",
        trn_df["image_id"].values,
        trn_df["label"].values,
        batch_size=batch_size,
        augmentations=AUGMENTATIONS_TRAIN,
    )
    val_gen = AugmentedImageSequence(
        "VALIDATE",
        "VALIDATE_DATA",
        val_df["image_id"].values,
        val_df["label"].values,
        batch_size=batch_size,
        augmentations=AUGMENTATIONS_TEST,
    )

    model = build_fallback_model()
    model.fit(trn_gen, validation_data=val_gen, epochs=3, verbose=1)




## === cell 13
def old_predict():
    test_results_list = []
    for image_id in test_df["image_id"]:
        image_path = os.path.join(TEST_DIR, image_id)
        image_data = Image.open(image_path).convert("RGB")
        image_data = image_data.resize((IMG_HEIGHT, IMG_WIDTH), PIL_RESAMPLE)
        image_data = np.asarray(image_data).astype(np.float32) / 255.0
        image_data = np.expand_dims(image_data, axis=0)
        predict_class = model.predict(image_data, verbose=0)
        test_results_list.append(
            {"image_id": image_id, "label": int(np.argmax(predict_class))}
        )

    test_results_df = pd.DataFrame(test_results_list)
    test_results_df.to_csv("submission.csv", index=False)




## === cell 14
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    RandomFlip,
    RandomRotation,
    RandomZoom,
    RandomTranslation,
    RandomShear,
)

_TTA_AUG = Sequential(
    [
        RandomFlip(mode="horizontal_and_vertical", seed=42),
        RandomRotation(factor=45.0 / 360.0, fill_mode="nearest", seed=42),
        RandomZoom(
            height_factor=(-0.4, 0.4),
            width_factor=(-0.4, 0.4),
            fill_mode="nearest",
            seed=42,
        ),
        RandomTranslation(
            height_factor=0.1, width_factor=0.1, fill_mode="nearest", seed=42
        ),
        RandomShear(x_factor=0.1, y_factor=0.1, fill_mode="nearest", seed=42),
    ],
    name="tta_aug",
)

AUTOTUNE = tf.data.AUTOTUNE


def _make_test_ds(image_ids, batch):
    ids = tf.constant(image_ids)
    ds = tf.data.Dataset.from_tensor_slices(ids)

    def _read_decode_resize(iid):
        path = tf.strings.join([tf.constant(TEST_DIR), iid])
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)  # uint8
        img = tf.image.resize(
            img,
            [IMG_HEIGHT, IMG_WIDTH],
            method=tf.image.ResizeMethod.LANCZOS3,
            antialias=True,
        )  # float32
        img = img / 255.0
        return iid, img

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)
    ds = ds.map(_read_decode_resize, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    ds = ds.batch(batch, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


@tf.function(reduce_retracing=True)
def _tta_augment_5_stateless(x, seed0, seed1):
    n = tf.shape(x)[0]
    x5 = tf.repeat(x, repeats=5, axis=0)  # (5N,H,W,3)

    idx = tf.range(n * 5, dtype=tf.int32)
    s0 = seed0 + idx * tf.constant(997, tf.int32)
    s1 = seed1 + idx * tf.constant(7919, tf.int32)
    seeds = tf.stack([s0, s1], axis=1)  # (5N,2)

    x5 = tf.image.stateless_random_flip_left_right(x5, seed=seeds)
    x5 = _TTA_AUG(x5, training=True)
    return x5




## === cell 15
tta_k = 6  # 1 original + 5 augmented

PRED_GROUP = 64
PRED_BATCH_SIZE = 256  # only affects internal predict batching; outputs identical

image_ids_all = test_df["image_id"].values
test_ds = _make_test_ds(image_ids_all, batch=PRED_GROUP)

all_ids = []
all_labels = []

for batch_index, (ids_tf, x0_tf) in enumerate(test_ds):
    seed0 = tf.constant(12345 + batch_index * PRED_GROUP, dtype=tf.int32)
    seed1 = tf.constant(54321 + batch_index * PRED_GROUP, dtype=tf.int32)

    x5 = _tta_augment_5_stateless(x0_tf, seed0, seed1)  # (5N,H,W,3)
    big_x = tf.concat([x0_tf, x5], axis=0)  # (6N,H,W,3)

    preds = model.predict(big_x, verbose=0, batch_size=PRED_BATCH_SIZE)  # (6N,C) numpy
    preds = (
        preds.reshape(tta_k, int(preds.shape[0] // tta_k), -1)
        .transpose(1, 0, 2)
        .mean(axis=1)
    )
    labels = np.argmax(preds, axis=1).astype(np.int32)

    ids_np = ids_tf.numpy().astype(str)
    all_ids.append(ids_np)
    all_labels.append(labels)

all_ids = np.concatenate(all_ids, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

test_results_df = pd.DataFrame({"image_id": all_ids, "label": all_labels.astype(int)})

sample_path = os.path.join(BASE_DIR, "sample_submission.csv")
sample_sub = pd.read_csv(sample_path)
test_results_df = sample_sub[["image_id"]].merge(
    test_results_df, on="image_id", how="left"
)
test_results_df["label"] = test_results_df["label"].fillna(0).astype(int)

test_results_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", test_results_df.shape)
test_results_df.head()



## === cell 16
submission = pd.read_csv("submission.csv")
print(submission.head(3))
print(submission.columns)
print("Unique labels:", submission["label"].unique()[:10])
