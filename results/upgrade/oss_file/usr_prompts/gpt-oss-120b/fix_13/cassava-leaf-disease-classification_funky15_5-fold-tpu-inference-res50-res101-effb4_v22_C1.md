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

0.8824418253248716

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11024) has done: 'The changes remove the failing imports, replace the missing pretrained checkpoint with a fresh EfficientNetB0 model built from ImageNet weights, add a lightweight training loop using the provided `train.csv`, and simplify the prediction utilities so they work with the new model. This resolves the import and file‑not‑found errors, ensures a valid `submission.csv` is written, and, with a modest amount of fine‑tuning, moves the accuracy toward the target score while keeping the original workflow intact.'
- What this solution (achieved 0.61099) has done: 'We replace the slower `ImageDataGenerator` pipelines with a native `tf.data` pipeline that does the same resizing, rescaling and comparable augmentations, and we turn off unnecessary multiprocessing overhead. This keeps the model architecture and training schedule unchanged while dramatically reducing the per‑epoch data‑loading cost, allowing the whole script to finish well under the 600 s limit.'

# 9. Code solution

## === cell 0
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("GPU memory growth error:", e)

import os, glob, math, re, random

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
from PIL import Image

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

tf.config.optimizer.set_jit(True)

mixed_precision.set_global_policy("mixed_float16")

print("Tensorflow version", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1532233509.py in <cell line: 0>()
      1 # Added GPU detection and memory‑growth configuration to ensure TensorFlow uses an available GPU,
      2 # which dramatically reduces training time compared to CPU execution.
----> 3 gpus = tf.config.list_physical_devices("GPU")
      4 if gpus:
      5     try:

NameError: name 'tf' is not defined

## === cell 1
IMAGE_SIZE = 224  # smaller size for faster training
BATCH_SIZE = 256  # larger batch reduces steps per epoch
NUM_CLASSES = 5
EPOCHS = 12  # increased epochs for better learning
FINE_TUNE_EPOCHS = 6  # increased fine‑tuning epochs

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUBMIT = os.path.join(DATA_ROOT, "sample_submission.csv")

CACHE_DIR = "/tmp/tf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3601028550.py in <cell line: 0>()
      6 
      7 DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
----> 8 TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
      9 TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
     10 TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")

NameError: name 'os' is not defined

## === cell 2
df = pd.read_csv(TRAIN_CSV)
train_df, val_df = train_test_split(
    df, test_size=0.1, stratify=df["label"], random_state=42
)

train_df["filepath"] = train_df["image_id"].apply(
    lambda x: os.path.join(TRAIN_IMG_DIR, x)
)
val_df["filepath"] = val_df["image_id"].apply(lambda x: os.path.join(TRAIN_IMG_DIR, x))



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3852992607.py in <cell line: 0>()
----> 1 df = pd.read_csv(TRAIN_CSV)
      2 train_df, val_df = train_test_split(
      3     df, test_size=0.1, stratify=df["label"], random_state=42
      4 )
      5 

NameError: name 'pd' is not defined

## === cell 3
AUTOTUNE = tf.data.AUTOTUNE

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip(mode="horizontal_and_vertical"),
        layers.RandomRotation(0.111),  # ~20 degrees
        layers.RandomZoom(height_factor=0.1, width_factor=0.1),
    ]
)


def _decode_resize_normalize(filepath):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    img = img / 255.0
    img = tf.cast(img, tf.float16)
    return img


def _parse_function(filepath, label, augment=False):
    img = _decode_resize_normalize(filepath)
    if augment:
        img = data_augmentation(img, training=True)
    return img, label


def make_dataset(
    df, shuffle=False, augment=False, cache_path=None, in_memory_cache=False
):
    """
    Build a tf.data pipeline.
    cache_path: optional file path for on‑disk caching.
    in_memory_cache: if True, cache in RAM (fastest) – used for training set.
    """
    ds = tf.data.Dataset.from_tensor_slices((df["filepath"].values, df["label"].values))
    if shuffle:
        ds = ds.shuffle(len(df), reshuffle_each_iteration=True)
    ds = ds.map(
        lambda f, l: (_decode_resize_normalize(f), tf.cast(l, tf.int32)),
        num_parallel_calls=AUTOTUNE,
    )
    if in_memory_cache:
        ds = ds.cache()  # cache in RAM
    else:
        ds = ds.cache(cache_path) if cache_path else ds.cache()
    if augment:
        ds = ds.map(
            lambda img, l: (data_augmentation(img, training=True), l),
            num_parallel_calls=AUTOTUNE,
        )
    options = tf.data.Options()
    options.experimental_deterministic = False
    ds = ds.with_options(options)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


train_ds = make_dataset(
    train_df,
    shuffle=True,
    augment=True,
    cache_path=os.path.join(CACHE_DIR, "train_cache"),
    in_memory_cache=False,  # use disk cache to keep RAM usage modest
)

val_ds = make_dataset(
    val_df,
    shuffle=False,
    augment=False,
    cache_path=os.path.join(CACHE_DIR, "val_cache"),
    in_memory_cache=False,
)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1608179197.py in <cell line: 0>()
----> 1 AUTOTUNE = tf.data.AUTOTUNE
      2 
      3 data_augmentation = tf.keras.Sequential(
      4     [
      5         layers.RandomFlip(mode="horizontal_and_vertical"),

NameError: name 'tf' is not defined

## === cell 4
def build_model():
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    )
    base.trainable = False  # freeze ImageNet weights initially
    x = layers.GlobalAveragePooling2D()(base.output)
    output = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(
        x
    )  # force float32 for stability
    model = tf.keras.Model(inputs=base.input, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


model = build_model()
model.summary()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/571009018.py in <cell line: 0>()
     17 
     18 
---> 19 model = build_model()
     20 model.summary()
     21 

/tmp/ipykernel_11/571009018.py in build_model()
      1 def build_model():
----> 2     base = tf.keras.applications.EfficientNetB0(
      3         include_top=False, weights="imagenet", input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
      4     )
      5     base.trainable = False  # freeze ImageNet weights initially

NameError: name 'tf' is not defined

## === cell 5
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True, monitor="val_accuracy"),
    ReduceLROnPlateau(patience=2, factor=0.5, monitor="val_accuracy"),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=2,
)

model.layers[0].trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=callbacks,
    verbose=2,
)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1647822510.py in <cell line: 0>()
      1 callbacks = [
----> 2     EarlyStopping(patience=3, restore_best_weights=True, monitor="val_accuracy"),
      3     ReduceLROnPlateau(patience=2, factor=0.5, monitor="val_accuracy"),
      4 ]
      5 

NameError: name 'EarlyStopping' is not defined

## === cell 6
def predict_on_folder(image_dir, model_obj, batch_size=64):
    img_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    ids = [os.path.basename(p) for p in img_paths]

    ds = tf.data.Dataset.from_tensor_slices(img_paths)
    ds = ds.map(
        lambda p: _decode_resize_normalize(p),
        num_parallel_calls=AUTOTUNE,
    )
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)

    preds = model_obj.predict(ds, verbose=0)
    preds = np.argmax(preds, axis=1)

    return pd.DataFrame({"image_id": ids, "label": preds.astype(int)})




## === cell 7
predict_df = predict_on_folder(TEST_IMG_DIR, model)
predict_df.to_csv("submission.csv", index=False)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2674484475.py in <cell line: 0>()
----> 1 predict_df = predict_on_folder(TEST_IMG_DIR, model)
      2 predict_df.to_csv("submission.csv", index=False)
      3 

NameError: name 'TEST_IMG_DIR' is not defined

## === cell 8
print(predict_df.head())

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2160939148.py in <cell line: 0>()
----> 1 print(predict_df.head())

NameError: name 'predict_df' is not defined
