# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import random
import warnings

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from sklearn.model_selection import train_test_split



## === cell 1
data_path = "/kaggle/input/plant-pathology-2021-fgvc8"
labels_file_path = os.path.join(data_path, "train.csv")
train_images_path = os.path.join(data_path, "train_images")
test_images_path = os.path.join(data_path, "test_images")

submission = pd.read_csv(os.path.join(data_path, "sample_submission.csv"))
train_df = pd.read_csv(labels_file_path)

train_df.head()



## === cell 2
possible = [
    "healthy",
    "scab",
    "frog_eye_leaf_spot",
    "complex",
    "rust",
    "powdery_mildew",
]

labels = np.zeros((train_df.shape[0], len(possible)), dtype=np.float32)
labels = pd.DataFrame(columns=possible, data=labels)

for i in range(train_df.shape[0]):
    full_lab = str(train_df.loc[i, "labels"])
    for lab in possible:
        if lab in full_lab.split():
            labels.loc[i, lab] = 1.0

labels.head()



## === cell 3
labels.index = train_df.index
train_df = train_df.drop("labels", axis=1)



## === cell 4
data = pd.concat([train_df, labels], axis=1)
data.head()



## === cell 5
train_split, valid_split = train_test_split(
    data,
    test_size=0.2,
    random_state=SEED,
    shuffle=True,
)

train_split = train_split.reset_index(drop=True)
valid_split = valid_split.reset_index(drop=True)

y_cols = possible



## === cell 6
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    samplewise_center=True,
    samplewise_std_normalization=True,
)

valid_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    samplewise_center=True,
    samplewise_std_normalization=True,
)

INPUT_SIZE = (256, 256, 3)
BATCH_SIZE = 16

train_generator = train_datagen.flow_from_dataframe(
    train_split,
    directory=train_images_path,
    x_col="image",
    y_col=y_cols,
    class_mode="raw",
    target_size=INPUT_SIZE[:2],
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
)

valid_generator = valid_datagen.flow_from_dataframe(
    valid_split,
    directory=train_images_path,
    x_col="image",
    y_col=y_cols,
    class_mode="raw",
    target_size=INPUT_SIZE[:2],
    batch_size=BATCH_SIZE,
    shuffle=False,
)



## === cell 7
convnet = Sequential(
    [
        layers.Input(shape=INPUT_SIZE),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(len(possible), activation="sigmoid"),
    ]
)

convnet.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

EPOCHS = 3
history = convnet.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS,
    verbose=1,
)



## === cell 8
convnet.summary()



## === cell 9
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

test_generator = test_datagen.flow_from_dataframe(
    submission,
    directory=test_images_path,
    x_col="image",
    y_col=None,
    class_mode=None,
    target_size=INPUT_SIZE[:2],
    batch_size=BATCH_SIZE,
    shuffle=False,
)



## === cell 10
preds = convnet.predict(test_generator, verbose=1)

pred_idx = np.argmax(preds, axis=1).astype(int)
labeltest = [possible[i] for i in pred_idx]

len(labeltest), submission.shape



## === cell 11
submission["labels"] = labeltest
submission.head()



## === cell 12
submission.head()



## === cell 13
assert list(submission.columns) == ["image", "labels"]
assert submission["labels"].notna().all()
submission.head()



## === cell 14
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
