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

0.8942278634028408

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import re
import numpy as np
import pandas as pd
import random
import math
from sklearn import metrics
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow_addons as tfa
from kaggle_datasets import KaggleDatasets
import glob

from tensorflow.keras.applications import EfficientNetB5



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Running on TPU ", tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)



## === cell 2
AUTO = tf.data.experimental.AUTOTUNE

EPOCHS = 20
BATCH_SIZE = 32 * strategy.num_replicas_in_sync
IMAGE_SIZE = [512, 512]
SEED = 123
LR = 0.0001
TTA = 10
VERBOSE = 2
N_CLASSES = 5

TEST_FILENAMES = "../input/cassava-leaf-disease-classification/test_images/*.jpg"




## === cell 3
def data_augment(image, image_name):
    p_spatial = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_rotate = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_pixel_1 = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_pixel_2 = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_pixel_3 = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_crop = tf.random.uniform([], 0, 1.0, dtype=tf.float32)

    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    if p_spatial > 0.75:
        image = tf.image.transpose(image)

    if p_rotate > 0.75:
        image = tf.image.rot90(image, k=3)  # rotate 270º
    elif p_rotate > 0.5:
        image = tf.image.rot90(image, k=2)  # rotate 180º
    elif p_rotate > 0.25:
        image = tf.image.rot90(image, k=1)  # rotate 90º

    if p_pixel_1 >= 0.4:
        image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
    if p_pixel_2 >= 0.4:
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    if p_pixel_3 >= 0.4:
        image = tf.image.random_brightness(image, max_delta=0.1)

    if p_crop > 0.7:
        if p_crop > 0.9:
            image = tf.image.central_crop(image, central_fraction=0.7)
        elif p_crop > 0.8:
            image = tf.image.central_crop(image, central_fraction=0.8)
        else:
            image = tf.image.central_crop(image, central_fraction=0.9)
    elif p_crop > 0.4:
        crop_size = tf.random.uniform(
            [], int(IMAGE_SIZE[0] * 0.8), IMAGE_SIZE[0], dtype=tf.int32
        )
        image = tf.image.random_crop(image, size=[crop_size, crop_size, 3])

    image = tf.image.resize(image, size=IMAGE_SIZE)
    image = tf.reshape(image, [*IMAGE_SIZE, 3])

    return image, image_name


def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image


def get_image_name(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    image_name = parts[-1]
    return image_name


def read_image(file_path):
    image_name = get_image_name(file_path)
    image = tf.io.read_file(file_path)
    image = decode_image(image)
    return image, image_name


def get_test_dataset(filenames, tta=False):
    dataset = tf.data.Dataset.list_files(filenames, shuffle=False)
    dataset = dataset.map(read_image, num_parallel_calls=AUTO)
    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    if tta:
        dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    return dataset


NUM_TESTING_IMAGES = len(
    os.listdir("../input/cassava-leaf-disease-classification/test_images/")
)




## === cell 4
def get_model():
    with strategy.scope():
        inp = tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3))

        x = EfficientNetB5(weights=None, include_top=False)(inp)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        output = tf.keras.layers.Dense(N_CLASSES, activation="softmax")(x)

        model = tf.keras.models.Model(inputs=[inp], outputs=[output])

        opt = tf.keras.optimizers.Adam(learning_rate=LR)
        model.compile(
            optimizer=opt,
            loss=[tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.4)],
            metrics=[tf.keras.metrics.CategoricalAccuracy()],
        )
        return model


def inference(model_paths):
    if not model_paths:
        raise FileNotFoundError(
            "No model .h5 files found in ../input/cassava-models/. "
            "Please ensure the dataset containing trained weights is attached."
        )

    prediction = np.zeros((NUM_TESTING_IMAGES, N_CLASSES), dtype=np.float32)

    print("Extracting test image names...")
    test_dataset = get_test_dataset(TEST_FILENAMES, tta=False)
    image_name = test_dataset.map(lambda image, image_name: image_name).unbatch()
    image_name = next(iter(image_name.batch(NUM_TESTING_IMAGES))).numpy().astype("U")
    print("Test image names completed...")

    for fold, model_path in enumerate(model_paths):
        print("\n" + "-" * 50)
        print(f"Predicting fold {fold + 1} | loading: {model_path}")
        K.clear_session()
        model = get_model()
        model.load_weights(model_path)

        steps = TTA * ((NUM_TESTING_IMAGES / BATCH_SIZE) + 1)
        test_dataset = get_test_dataset(TEST_FILENAMES, tta=True)
        image = test_dataset.map(lambda image, image_name: image)

        probabilities = model.predict(image, steps=steps, verbose=VERBOSE)[
            : TTA * NUM_TESTING_IMAGES
        ]
        probabilities = np.mean(
            probabilities.reshape((NUM_TESTING_IMAGES, TTA, N_CLASSES), order="F"),
            axis=1,
        )
        prediction += probabilities / len(model_paths)

    sub = pd.DataFrame(
        {
            "image_id": image_name,
            "label": np.argmax(prediction, axis=-1).astype(np.int64),
        }
    )
    sub.to_csv("submission.csv", index=False)
    return image_name, prediction, sub


model_paths = sorted(glob.glob("../input/cassava-models/*.h5"))
image_name, prediction, sub = inference(model_paths)
sub

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3039149021.py in <cell line: 0>()
     67 
     68 # Get pretrained models list for inference (glob import ensured above to prevent NameError).
---> 69 model_paths = sorted(glob.glob("../input/cassava-models/*.h5"))
     70 image_name, prediction, sub = inference(model_paths)
     71 sub

NameError: name 'glob' is not defined
