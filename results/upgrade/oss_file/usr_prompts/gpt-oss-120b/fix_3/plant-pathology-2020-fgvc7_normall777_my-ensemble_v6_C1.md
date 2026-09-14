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

3.9

# 3. Installed packages

No external packages required in the script and installed.

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

0.9682332315790864

# 6. Current score

0.46692

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.62799) has done: 'Implemented fixes to eliminate runtime errors and produce a valid `submission.csv`:
- Replaced unavailable `KaggleDatasets` calls with local dataset paths.
- Defined `AUTO` correctly and simplified TPU handling.
- Corrected path formatting and ensured all variables are defined before use.
- Consolidated model creation to a single EfficientNetB0 architecture with ImageNet weights (no external weight files).
- Added a brief training loop (5 epochs) suitable for the small dataset.
- Adjusted prediction aggregation and saved the submission with proper columns.'
- What this solution (achieved 0.46692) has done: 'Implemented three key fixes: (1) set the protobuf implementation environment variable before importing TensorFlow to prevent the import‑time AttributeError; (2) corrected the model for multi‑label classification by using a sigmoid output, `binary_crossentropy` loss, and a binary‑accuracy metric; (3) kept the rest of the pipeline unchanged so the script runs end‑to‑end and writes a proper `submission.csv`. These changes resolve the runtime crash and align the training objective with the ROC‑AUC metric, boosting the score toward the target.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np, pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
from sklearn.model_selection import train_test_split

print("Tensorflow version:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_PATH = "/kaggle/input/plant-pathology-2020-fgvc7"
IMG_DIR = os.path.join(BASE_PATH, "images")
train_csv = os.path.join(BASE_PATH, "train.csv")
test_csv = os.path.join(BASE_PATH, "test.csv")
sample_sub_csv = os.path.join(BASE_PATH, "sample_submission.csv")

train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)
sub_df = pd.read_csv(sample_sub_csv)


def format_path(st):
    return os.path.join(IMG_DIR, f"{st}.jpg")


train_paths = train_df["image_id"].apply(format_path).values
test_paths = test_df["image_id"].apply(format_path).values
train_labels = train_df.loc[:, "healthy":].values  # columns order matches submission

train_paths, valid_paths, train_labels, valid_labels = train_test_split(
    train_paths, train_labels, test_size=0.15, random_state=2020, stratify=train_labels
)



## === cell 2
img_size = 224
AUTO = tf.data.AUTOTUNE
BATCH_SIZE = 32


def decode_image(filename, label=None, image_size=(img_size, img_size)):
    bits = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(bits, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # 0‑1
    img = tf.image.resize(img, image_size)
    if label is None:
        return img
    return img, label


def data_augment(image, label=None):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    if label is None:
        return image
    return image, label


train_ds = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .map(decode_image, num_parallel_calls=AUTO)
    .map(data_augment, num_parallel_calls=AUTO)
    .shuffle(1024)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_ds = (
    tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

test_ds = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(lambda x: decode_image(x), num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)




## === cell 3
def get_model(num_classes):
    base = EfficientNetB0(
        weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3)
    )
    x = GlobalAveragePooling2D()(base.output)
    output = Dense(num_classes, activation="sigmoid")(x)
    model = Model(inputs=base.input, outputs=output)
    model.compile(
        optimizer="nadam",
        loss="binary_crossentropy",
        metrics=["binary_accuracy"],
    )
    return model


num_classes = train_labels.shape[1]
model = get_model(num_classes)



## === cell 4
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5, verbose=1),
    tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True, verbose=1),
]
model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=5,
    steps_per_epoch=len(train_paths) // BATCH_SIZE,
    validation_steps=len(valid_paths) // BATCH_SIZE,
    callbacks=callbacks,
    verbose=2,
)



## === cell 5
probabilities = model.predict(test_ds, verbose=1)

sub_df.loc[:, "healthy":] = probabilities
sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print(f"Submission saved to {sub_path}")
sub_df.head()
