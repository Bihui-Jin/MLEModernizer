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

0.8735267452402539

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras.layers import Dense, Dropout

print("TF version:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
label_json_path = (
    "../input/cassava-leaf-disease-classification/label_num_to_disease_map.json"
)
images_dir_path = "../input/cassava-leaf-disease-classification/train_images"
test_images_dir_path = "../input/cassava-leaf-disease-classification/test_images"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"



## === cell 2
train_csv = pd.read_csv(train_csv_path)
train_csv["label"] = train_csv["label"].astype(str)

label_class = pd.read_json(label_json_path, orient="index")
label_class = label_class.values.flatten().tolist()

print(train_csv.head())
print("Num classes from json:", len(label_class), label_class)



## === cell 3
IMG_SIZE = 288
BATCH_SIZE = 12
EPOCHS = 32
lr = 1e-5
SEED = 42

tf.keras.utils.set_random_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE



## === cell 4
train_ds = tf.keras.utils.image_dataset_from_directory(
    images_dir_path,
    labels="inferred",
    label_mode="categorical",
    class_names=[str(i) for i in range(5)],  # preserve label->index mapping (0..4)
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
    validation_split=0.2,
    subset="training",
)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    images_dir_path,
    labels="inferred",
    label_mode="categorical",
    class_names=[str(i) for i in range(5)],  # preserve label->index mapping (0..4)
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False,
    seed=SEED,
    validation_split=0.2,
    subset="validation",
)

print("Class indices:", {str(i): i for i in range(5)})



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/545885803.py in <cell line: 0>()
      2 # This preserves the same data split semantics (80/20 via seed) and same target_size/batch_size.
      3 # It also enables parallel decode + prefetch to keep the accelerator/CPU busy without changing model logic.
----> 4 train_ds = tf.keras.utils.image_dataset_from_directory(
      5     images_dir_path,
      6     labels="inferred",

/usr/local/lib/python3.11/dist-packages/keras/src/utils/image_dataset_utils.py in image_dataset_from_directory(directory, labels, label_mode, class_names, color_mode, batch_size, image_size, shuffle, seed, validation_split, subset, interpolation, follow_links, crop_to_aspect_ratio, pad_to_aspect_ratio, data_format, verbose)
    230     if seed is None:
    231         seed = np.random.randint(1e6)
--> 232     image_paths, labels, class_names = dataset_utils.index_directory(
    233         directory,
    234         labels,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/dataset_utils.py in index_directory(directory, labels, formats, class_names, shuffle, seed, follow_links, verbose)
    536         if class_names is not None:
    537             if not set(class_names).issubset(set(subdirs)):
--> 538                 raise ValueError(
    539                     "The `class_names` passed did not match the "
    540                     "names of the subdirectories of the target directory. "

ValueError: The `class_names` passed did not match the names of the subdirectories of the target directory. Expected: ['train_images'] (or a subset of it), but received: class_names=['0', '1', '2', '3', '4']

## === cell 5
PRECISION = tf.keras.metrics.Precision()
RECALL = tf.keras.metrics.Recall()


def F1_score(y_true, y_pred):
    if hasattr(PRECISION, "reset_state"):
        PRECISION.reset_state()
        RECALL.reset_state()
    else:
        PRECISION.reset_states()
        RECALL.reset_states()

    PRECISION.update_state(y_true, y_pred)
    precision_out = PRECISION.result()

    RECALL.update_state(y_true, y_pred)
    recall_out = RECALL.result()

    return 2 * ((precision_out * recall_out) / (precision_out + recall_out + 1e-23))




## === cell 6
augment = tf.keras.Sequential(
    [
        tf.keras.layers.Rescaling(1.0 / 255.0),
        tf.keras.layers.RandomRotation(factor=270.0 / 360.0, seed=SEED),
        tf.keras.layers.RandomTranslation(
            height_factor=0.2, width_factor=0.2, seed=SEED
        ),
        tf.keras.layers.RandomBrightness(
            factor=0.4, seed=SEED
        ),  # approximates [0.1, 0.9] range
        tf.keras.layers.RandomShear(
            x_factor=25.0 * np.pi / 180.0, y_factor=0.0, seed=SEED
        ),
        tf.keras.layers.RandomZoom(
            height_factor=(-0.3, 0.3), width_factor=(-0.3, 0.3), seed=SEED
        ),
        tf.keras.layers.RandomFlip(mode="horizontal_and_vertical", seed=SEED),
    ],
    name="augmentation",
)


@tf.function
def channel_shift(x, rng):
    batch = tf.shape(x)[0]
    shifts = rng.uniform(shape=(batch, 1, 1, 3), minval=-0.1, maxval=0.1, dtype=x.dtype)
    x = x + shifts
    return tf.clip_by_value(x, 0.0, 1.0)


rng = tf.random.Generator.from_seed(SEED)


def train_map(images, labels):
    images = tf.cast(images, tf.float32)
    images = augment(images, training=True)
    images = channel_shift(images, rng)
    return images, labels


def valid_map(images, labels):
    images = tf.cast(images, tf.float32) * (1.0 / 255.0)
    return images, labels


train_ds_opt = train_ds.map(train_map, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

valid_ds_opt = valid_ds.map(valid_map, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

BASE0 = applications.MobileNet(
    include_top=False,
    input_shape=[IMG_SIZE, IMG_SIZE, 3],
    weights=None,
    pooling="max",
)


def build_model(input_size=[IMG_SIZE, IMG_SIZE, 3]):
    model = tf.keras.Sequential()
    model.add(BASE0)
    model.add(Dropout(0.5))
    model.add(Dense(5, activation="softmax"))

    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9),
        metrics=["accuracy", F1_score],
        run_eagerly=False,
    )
    return model


model0 = build_model()
model0.summary()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1260909939.py in <cell line: 0>()
     55 # SPEED: Cache (in memory) after decoding to avoid repeated JPEG decode cost across epochs.
     56 # This does not change data or labels; only avoids redundant work.
---> 57 train_ds_opt = train_ds.map(train_map, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
     58 
     59 valid_ds_opt = valid_ds.map(valid_map, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

NameError: name 'train_ds' is not defined

## === cell 7
callback0 = tf.keras.callbacks.ModelCheckpoint(
    "CasavaLeafDiseaseModel.h5",
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=False,
    verbose=1,
)



## === cell 8
history = model0.fit(
    train_ds_opt,
    validation_data=valid_ds_opt,
    epochs=EPOCHS,
    callbacks=[callback0],
    verbose=1,
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2451621326.py in <cell line: 0>()
      1 # SPEED: Use the optimized tf.data pipelines (parallel decode/map + prefetch).
----> 2 history = model0.fit(
      3     train_ds_opt,
      4     validation_data=valid_ds_opt,
      5     epochs=EPOCHS,

NameError: name 'model0' is not defined

## === cell 9
if os.path.exists("CasavaLeafDiseaseModel.h5"):
    model0 = tf.keras.models.load_model(
        "CasavaLeafDiseaseModel.h5", custom_objects={"F1_score": F1_score}
    )



## === cell 10
ss = pd.read_csv(sample_sub_path)

test_df = ss[["image_id"]].copy()

test_paths = tf.constant(
    [os.path.join(test_images_dir_path, f) for f in test_df["image_id"].values]
)


def load_and_preprocess(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


test_ds = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

probs = model0.predict(test_ds, verbose=1)
preds = np.argmax(probs, axis=1).astype(int)

my_submission = pd.DataFrame({"image_id": ss["image_id"].values, "label": preds})
my_submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", my_submission.shape)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4040437527.py in <cell line: 0>()
     27 )
     28 
---> 29 probs = model0.predict(test_ds, verbose=1)
     30 preds = np.argmax(probs, axis=1).astype(int)
     31 

NameError: name 'model0' is not defined

## === cell 11
print("Submission File: \n---------------\n")
print(my_submission.head())
print("\nSaved to: submission.csv")

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1961544400.py in <cell line: 0>()
      1 print("Submission File: \n---------------\n")
----> 2 print(my_submission.head())
      3 print("\nSaved to: submission.csv")

NameError: name 'my_submission' is not defined
