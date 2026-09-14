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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

tf.random.set_seed(42)




## === cell 1
data_path = "/kaggle/input/plant-pathology-2021-fgvc8"
train_csv_path = os.path.join(data_path, "train.csv")
train_images_dir = os.path.join(data_path, "train_images")
test_images_dir = os.path.join(data_path, "test_images")
sample_submission_path = os.path.join(data_path, "sample_submission.csv")

train_df = pd.read_csv(train_csv_path)
submission = pd.read_csv(sample_submission_path)

train_df["primary_label"] = train_df["labels"].apply(lambda x: x.split(" ")[0])

class_names = sorted(train_df["primary_label"].unique())
num_classes = len(class_names)




## === cell 2
train_df, val_df = train_test_split(
    train_df, test_size=0.2, stratify=train_df["primary_label"], random_state=42
)

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    horizontal_flip=True,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
)

INPUT_SIZE = (256, 256)
BATCH_SIZE = 512  # increased from 256
NUM_WORKERS = (os.cpu_count() or 4) * 2  # retain parallel loading

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_images_dir,
    x_col="image",
    y_col="primary_label",
    target_size=INPUT_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    classes=class_names,
    shuffle=True,
    seed=42,
    workers=NUM_WORKERS,
    use_multiprocessing=True,
    max_queue_size=32,  # smaller queue reduces memory pressure
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=train_images_dir,
    x_col="image",
    y_col="primary_label",
    target_size=INPUT_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    classes=class_names,
    shuffle=False,
    workers=NUM_WORKERS,
    use_multiprocessing=True,
    max_queue_size=32,
)

from sklearn.utils import class_weight

class_weights_array = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=train_generator.classes,
)
class_weights = dict(enumerate(class_weights_array))

model = models.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(*INPUT_SIZE, 3)),
        layers.MaxPool2D(2, 2),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPool2D(2, 2),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPool2D(2, 2),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    verbose=1,
    class_weight=class_weights,
)




## === cell 3
import gc

gc.collect()

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=submission,
    directory=test_images_dir,
    x_col="image",
    y_col=None,
    target_size=INPUT_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=None,
    shuffle=False,
    workers=NUM_WORKERS,
    use_multiprocessing=True,
    max_queue_size=32,
)

preds = model.predict(
    test_generator,
    verbose=0,
)

pred_indices = np.argmax(preds, axis=1)
pred_labels = [class_names[idx] for idx in pred_indices]

submission["labels"] = pred_labels




## === cell 4
output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission file written to {output_path}")
