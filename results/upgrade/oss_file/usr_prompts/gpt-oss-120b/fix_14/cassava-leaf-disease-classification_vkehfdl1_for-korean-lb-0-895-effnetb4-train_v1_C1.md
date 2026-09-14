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
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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
tqdm==4.67.1

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

0.0507

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.utils import shuffle
import random
import warnings

warnings.filterwarnings("ignore")

import tensorflow as tf

tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

tf.config.optimizer.set_jit(True)

try:
    from tensorflow.keras.mixed_precision import Policy, set_global_policy

    set_global_policy(Policy("mixed_float16"))
    print("Mixed precision enabled.")
except Exception as e:
    print(f"Mixed precision not set due to: {e}")

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, GlobalAveragePooling2D, Dropout
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
training_folder = "../input/cassava-leaf-disease-classification/train_images/"



## === cell 2
samples_df = pd.read_csv("../input/cassava-leaf-disease-classification/train.csv")
samples_df = shuffle(samples_df, random_state=42)
samples_df["filepath"] = training_folder + samples_df["image_id"]
samples_df.head()



## === cell 3
training_df = samples_df[:1000]
validation_df = samples_df[1000:1200]



## === cell 4
batch_size = 64  # was 32
image_size = 512
input_shape = (image_size, image_size, 3)
dropout_rate = 0.4
classes_to_predict = list(range(5))



## === cell 5
training_data = tf.data.Dataset.from_tensor_slices(
    (training_df.filepath.values, training_df.label.values.astype(np.int32))
)
validation_data = tf.data.Dataset.from_tensor_slices(
    (validation_df.filepath.values, validation_df.label.values.astype(np.int32))
)


def load_image_and_label_from_path(image_path, label):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [image_size, image_size])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


AUTOTUNE = tf.data.AUTOTUNE

training_data = (
    training_data.map(load_image_and_label_from_path, num_parallel_calls=AUTOTUNE)
    .cache()
    .prefetch(AUTOTUNE)
)

validation_data = (
    validation_data.map(load_image_and_label_from_path, num_parallel_calls=AUTOTUNE)
    .cache()
    .prefetch(AUTOTUNE)
)



## === cell 6
training_data_batches = (
    training_data.shuffle(buffer_size=len(training_df), reshuffle_each_iteration=True)
    .batch(batch_size)
    .prefetch(buffer_size=AUTOTUNE)
)
validation_data_batches = validation_data.batch(batch_size).prefetch(
    buffer_size=AUTOTUNE
)



## === cell 7
data_augmentation_layers = tf.keras.Sequential(
    [
        layers.RandomCrop(height=image_size, width=image_size),
        layers.RandomFlip(mode="horizontal_and_vertical"),
        layers.RandomRotation(factor=0.25),
        layers.RandomZoom(height_factor=(-0.2, 0.0), width_factor=(-0.2, 0.0)),
        layers.RandomContrast(factor=0.2),
    ]
)


def simple_cnn_base(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    return tf.keras.Model(inputs, x)


efficientnet_base = simple_cnn_base(
    input_shape
)  # keep variable name for downstream code
efficientnet_base.trainable = False

inputs = Input(shape=input_shape)
x = data_augmentation_layers(inputs)
x = efficientnet_base(x)
x = Dropout(dropout_rate)(x)
outputs = Dense(len(classes_to_predict), activation="softmax", dtype="float32")(x)

model = Model(inputs=inputs, outputs=outputs)
model.summary()



## === cell 8
epochs = 15
steps_per_epoch = int(round(len(training_df) / batch_size))
decay_steps = steps_per_epoch * epochs
cosine_decay = CosineDecay(
    initial_learning_rate=1e-4, decay_steps=decay_steps, alpha=0.3
)

callbacks = [
    ModelCheckpoint(filepath="best_model.h5", monitor="val_loss", save_best_only=True)
]

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=Adam(learning_rate=cosine_decay),
    metrics=["accuracy"],
)

history = model.fit(
    training_data_batches,
    epochs=epochs,
    validation_data=validation_data_batches,
    callbacks=callbacks,
)




## === cell 9
def scan_over_image(img_path, crop_size=512):
    """
    Extract up to four overlapping crops of size `crop_size` from the image.
    Uses Pillow (PIL) for robust JPEG decoding.
    """
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        img_np = np.array(img)

    h, w, _ = img_np.shape
    if h < crop_size or w < crop_size:
        pad_h = max(crop_size - h, 0)
        pad_w = max(crop_size - w, 0)
        img_np = np.pad(
            img_np,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        h, w, _ = img_np.shape

    x_origins = [0, max(w - crop_size, 0)]
    y_origins = [0, max(h - crop_size, 0)]
    crops = []
    for x in x_origins:
        for y in y_origins:
            crop = img_np[y : y + crop_size, x : x + crop_size, :]
            crops.append(crop)
    return np.array(crops, dtype=np.uint8)




## === cell 10
test_time_augmentation_layers = tf.keras.Sequential(
    [
        layers.RandomFlip(mode="horizontal_and_vertical"),
        layers.RandomZoom(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2)),
        layers.RandomContrast(factor=0.2),
    ]
)




## === cell 11
def predict_and_vote(image_filename, folder, TTA_runs=4):
    """
    Predict the class for a single image using TTA and multiple crops.
    All crops and all TTA augmentations are batched together so the model
    is called only once per image.
    """
    crops_np = scan_over_image(os.path.join(folder, image_filename))
    crops_tf = tf.convert_to_tensor(crops_np, dtype=tf.float32) / 255.0  # (C, H, W, 3)

    repeated = tf.repeat(crops_tf, repeats=TTA_runs, axis=0)

    augmented = test_time_augmentation_layers(repeated)

    preds = model(augmented, training=False)  # (C*TTA, num_classes)

    C = tf.shape(crops_tf)[0]
    preds = tf.reshape(preds, (C, TTA_runs, -1))

    avg_per_crop = tf.reduce_mean(preds, axis=1)  # (C, num_classes)

    global_pred = tf.reduce_mean(avg_per_crop, axis=0)  # (num_classes,)

    final_prediction = int(tf.argmax(global_pred).numpy())
    return final_prediction




## === cell 12
def run_predictions_over_image_list(image_list, folder, TTA_runs=4):
    """
    Predict over many images sequentially.
    """
    predictions = []
    for img_name in image_list:
        pred = predict_and_vote(img_name, folder, TTA_runs=TTA_runs)
        predictions.append(pred)
    return predictions




## === cell 13
test_folder = "../input/cassava-leaf-disease-classification/test_images/test_images/"

image_files = [f for f in os.listdir(test_folder) if f.lower().endswith(".jpg")]
submission_df = pd.DataFrame(
    {
        "image_id": sorted(image_files),
        "label": [0] * len(image_files),  # placeholder, will be replaced
    }
)



## === cell 14
submission_df["label"] = run_predictions_over_image_list(
    submission_df["image_id"].tolist(), test_folder
)



## === cell 15
with open(
    "../input/cassava-leaf-disease-classification/label_num_to_disease_map.json"
) as f:
    label_map = f.read()
print(label_map[:200])



## === cell 16
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

## --- ERROR in outputing the csv:
Invalid submission: Submission must have the same length as the answers.
