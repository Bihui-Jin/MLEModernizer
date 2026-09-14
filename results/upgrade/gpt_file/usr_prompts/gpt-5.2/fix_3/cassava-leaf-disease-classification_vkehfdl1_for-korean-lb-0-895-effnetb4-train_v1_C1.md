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

# 5. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import random
import warnings

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import EfficientNetB0

from tensorflow.keras.optimizers.schedules import CosineDecay

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TF:", tf.__version__)



## === cell 1
training_folder = "../input/cassava-leaf-disease-classification/train_images/"



## === cell 2
samples_df = pd.read_csv("../input/cassava-leaf-disease-classification/train.csv")
samples_df = shuffle(samples_df, random_state=SEED).reset_index(drop=True)
samples_df["filepath"] = training_folder + samples_df["image_id"]
samples_df.head()



## === cell 3
training_df = samples_df[:1000].copy()
validation_df = samples_df[1000:1200].copy()



## === cell 4
batch_size = 8
image_size = 512
input_shape = (image_size, image_size, 3)
dropout_rate = 0.4

num_classes = 5

print(
    "Train rows:",
    len(training_df),
    "Val rows:",
    len(validation_df),
    "Classes:",
    num_classes,
)



## === cell 5
training_data = tf.data.Dataset.from_tensor_slices(
    (training_df.filepath.values, training_df.label.values)
)
validation_data = tf.data.Dataset.from_tensor_slices(
    (validation_df.filepath.values, validation_df.label.values)
)




## === cell 6
def load_image_and_label_from_path(image_path, label):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (image_size, image_size), method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    return img, tf.cast(label, tf.int32)


AUTOTUNE = tf.data.AUTOTUNE

training_data = training_data.map(
    load_image_and_label_from_path, num_parallel_calls=AUTOTUNE
)
validation_data = validation_data.map(
    load_image_and_label_from_path, num_parallel_calls=AUTOTUNE
)



## === cell 7
training_data_batches = (
    training_data.shuffle(buffer_size=1000, seed=SEED)
    .batch(batch_size)
    .prefetch(buffer_size=AUTOTUNE)
)
validation_data_batches = (
    validation_data.shuffle(buffer_size=1000, seed=SEED)
    .batch(batch_size)
    .prefetch(buffer_size=AUTOTUNE)
)



## === cell 8
data_augmentation_layers = tf.keras.Sequential(
    [
        layers.RandomCrop(height=image_size, width=image_size),
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.25),
        layers.RandomZoom((-0.2, 0.0)),
        layers.RandomContrast((0.2, 0.2)),
    ],
    name="train_aug",
)



## === cell 9
efficientnet_backbone = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=input_shape,
)

inputs = Input(shape=input_shape)
augmented = data_augmentation_layers(inputs)
features = efficientnet_backbone(augmented, training=True)
pooling = layers.GlobalAveragePooling2D()(features)
dropout = layers.Dropout(dropout_rate)(pooling)
outputs = Dense(num_classes, activation="softmax")(dropout)
model = Model(inputs=inputs, outputs=outputs)

model.summary()



## === cell 10
epochs = 15
decay_steps = int(round(len(training_df) / batch_size)) * epochs
cosine_decay = CosineDecay(
    initial_learning_rate=1e-4, decay_steps=decay_steps, alpha=0.3
)

callbacks = [
    ModelCheckpoint(
        filepath="best_model.keras", monitor="val_loss", save_best_only=True
    )
]

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=cosine_decay),
    metrics=["accuracy"],
)



## === cell 11
history = model.fit(
    training_data_batches,
    epochs=epochs,
    validation_data=validation_data_batches,
    callbacks=callbacks,
)

model = tf.keras.models.load_model("best_model.keras")




## === cell 12
def scan_over_image(img_path, crop_size=512):
    """
    Extract 512x512 images covering the whole original image (4 crops).
    """
    img = Image.open(img_path).convert("RGB")
    img_width, img_height = img.size
    img = np.array(img)

    pad_h = max(0, crop_size - img_height)
    pad_w = max(0, crop_size - img_width)
    if pad_h > 0 or pad_w > 0:
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        img_height, img_width = img.shape[0], img.shape[1]

    x_img_origins = [0, img_width - crop_size]
    y_img_origins = [0, img_height - crop_size]

    img_list = []
    for x in x_img_origins:
        for y in y_img_origins:
            img_list.append(img[y : y + crop_size, x : x + crop_size, :])

    return np.array(img_list)




## === cell 13
test_time_augmentation_layers = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomZoom((-0.2, 0.2)),
        layers.RandomContrast((0.2, 0.2)),
    ],
    name="tta_aug",
)




## === cell 14
def predict_and_vote(image_filename, folder, TTA_runs=4):
    """
    Run the model over 4 local areas of the given image, with light TTA, then vote by summed probabilities.
    """
    localised_predictions = []
    local_image_list = scan_over_image(
        os.path.join(folder, image_filename), crop_size=image_size
    )

    for local_image in local_image_list:
        local_image = tf.cast(local_image, tf.float32) / 255.0
        local_image = tf.expand_dims(local_image, 0)  # (1, H, W, C)

        probs_sum = None
        for _ in range(TTA_runs):
            aug = test_time_augmentation_layers(local_image, training=True)
            preds = model.predict(aug, verbose=0)[0]  # (num_classes,)
            probs_sum = preds if probs_sum is None else (probs_sum + preds)

        localised_predictions.append(probs_sum)

    global_predictions = np.sum(np.array(localised_predictions), axis=0)
    final_prediction = int(np.argmax(global_predictions))
    return final_prediction




## === cell 15
def run_predictions_over_image_list(image_list, folder):
    predictions = []
    with tqdm(total=len(image_list)) as pbar:
        for image_filename in image_list:
            predictions.append(predict_and_vote(image_filename, folder))
            pbar.update(1)
    return predictions




## === cell 16
map_path = "../input/cassava-leaf-disease-classification/label_num_to_disease_map.json"
with open(map_path, "r") as f:
    print(f.read())



## === cell 17
test_folder = "../input/cassava-leaf-disease-classification/test_images/"

test_files = sorted([f for f in os.listdir(test_folder) if f.lower().endswith(".jpg")])

submission_df = pd.DataFrame({"image_id": test_files})
submission_df["label"] = 0
submission_df.head()



## === cell 18
submission_df["label"] = run_predictions_over_image_list(
    submission_df["image_id"].tolist(), test_folder
)

submission_df["label"] = submission_df["label"].astype(int)
submission_df = submission_df[["image_id", "label"]]

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print("Wrote", submission_path, "with shape", submission_df.shape)
print(submission_df.head())
