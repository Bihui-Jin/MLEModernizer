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
Detect apple diseases from images.

## Metric
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Target score

0.5422

# 6. Current score

0.81772

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.81772) has done: 'Implemented fixes to unblock the pipeline and produce a valid submission:
- Corrected filename creation by applying `os.path.join` per row.
- Fixed label extraction to use the proper column order (`healthy, multiple_diseases, rust, scab`) and kept numeric class IDs.
- Replaced the failing `ImageDataGenerator` workflow with a lightweight `tf.data` pipeline that loads, rescales, and batches images, removing the protobuf conflict.
- Adjusted dataset preparation, training, validation, and test inference to work with the new pipeline.
- Ensured the final CSV matches the required submission format (`image_id, healthy, multiple_diseases, rust, scab`).'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_csv_path = "/kaggle/input/plant-pathology-2020-fgvc7/train.csv"
test_csv_path = "/kaggle/input/plant-pathology-2020-fgvc7/test.csv"
folder_images = "/kaggle/input/plant-pathology-2020-fgvc7/images"




## === cell 2
def add_filename(file_path):
    df = pd.read_csv(file_path)
    df["filename"] = df["image_id"].apply(
        lambda x: os.path.join(folder_images, f"{x}.jpg")
    )
    return df


def prepare_data(file_path):
    df = pd.read_csv(file_path)
    y = df[["healthy", "multiple_diseases", "rust", "scab"]].values
    categories = y.argmax(axis=1)  # 0‑healthy,1‑multiple,2‑rust,3‑scab
    df["filename"] = df["image_id"].apply(
        lambda x: os.path.join(folder_images, f"{x}.jpg")
    )
    df["category"] = categories
    return df




## === cell 3
train_df = prepare_data(train_csv_path)
test_df = add_filename(test_csv_path)




## === cell 4
train_df, validate_df = train_test_split(train_df, test_size=0.20, random_state=1)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)




## === cell 5
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
IMAGE_CHANNELS = 3
OUTPUT = 4
batch_size = 5
AUTOTUNE = tf.data.AUTOTUNE




## === cell 6
def decode_image(filename, label=None):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=IMAGE_CHANNELS)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = img / 255.0  # rescale
    if label is None:
        return img
    label = tf.one_hot(label, depth=OUTPUT)
    return img, label


def make_dataset(df, training=True):
    ds = tf.data.Dataset.from_tensor_slices(
        (df["filename"].values, df["category"].values)
    )
    ds = ds.map(decode_image, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


train_dataset = make_dataset(train_df, training=True)
validate_dataset = make_dataset(validate_df, training=False)




## === cell 7
model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(
            256,
            (3, 3),
            strides=2,
            activation="relu",
            input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(256, (3, 3), strides=2, activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(256, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(OUTPUT, activation="softmax"),
    ]
)

model.compile(
    loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"]
)




## === cell 8
earlystop = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_accuracy", patience=2, factor=0.5, min_lr=1e-5, verbose=1
)
callbacks = [earlystop, lr_reduce]




## === cell 9
epochs = 5
model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=validate_dataset,
    callbacks=callbacks,
    verbose=2,
)




## === cell 10
def make_test_dataset(df):
    ds = tf.data.Dataset.from_tensor_slices(df["filename"].values)
    ds = ds.map(lambda x: decode_image(x, label=None), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


test_dataset = make_test_dataset(test_df)




## === cell 11
predictions = model.predict(test_dataset, verbose=2)




## === cell 12
submission = pd.read_csv(test_csv_path)  # contains only image_id
submission["healthy"] = predictions[:, 0]
submission["multiple_diseases"] = predictions[:, 1]
submission["rust"] = predictions[:, 2]
submission["scab"] = predictions[:, 3]

submission.to_csv("submission.csv", index=False)
