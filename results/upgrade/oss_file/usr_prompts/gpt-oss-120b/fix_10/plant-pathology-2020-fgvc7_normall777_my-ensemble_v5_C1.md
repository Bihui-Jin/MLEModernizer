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

# 5. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPooling2D,
    Flatten,
    GlobalAveragePooling2D,
)
from sklearn.model_selection import train_test_split

tf.keras.mixed_precision.set_global_policy("mixed_float16")

tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)

print("TensorFlow version:", tf.__version__)




## === cell 1
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except Exception:
    strategy = tf.distribute.get_strategy()  # CPU or single‑GPU

print("Strategy replicas:", strategy.num_replicas_in_sync)

AUTO = tf.data.AUTOTUNE
BATCH_SIZE = 256 * strategy.num_replicas_in_sync
img_size = 224  # image size unchanged

BASE_IMG_PATH = "/kaggle/input/plant-pathology-2020-fgvc7/images"




## === cell 2
def format_path(st):
    return os.path.join(BASE_IMG_PATH, st + ".jpg")


train = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/train.csv")
test = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/test.csv")
sub = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv")

train_paths = train["image_id"].apply(format_path).values
test_paths = test["image_id"].apply(format_path).values

train_labels = train.loc[:, "healthy":].values

train_paths, valid_paths, train_labels, valid_labels = train_test_split(
    train_paths, train_labels, test_size=0.15, random_state=2020, stratify=train_labels
)




## === cell 3
def preprocess_image(path):
    """Read, decode, resize and normalize a JPEG file."""
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # scales to [0,1]
    img = tf.image.resize(img, (img_size, img_size))
    return img


def data_augment(image, label=None, seed=2020):
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.random_flip_up_down(image, seed=seed)
    return (image, label) if label is not None else image


train_dataset = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .map(lambda p, l: (preprocess_image(p), l), num_parallel_calls=AUTO)
    .map(data_augment, num_parallel_calls=AUTO)
    .shuffle(1024, seed=2020)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels))
    .map(lambda p, l: (preprocess_image(p), l), num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(preprocess_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)




## === cell 4
def get_model(num_classes):
    """
    Build a model. Try EfficientNetB0 with ImageNet weights; if that fails,
    fall back to a small custom CNN to keep the pipeline functional.
    """
    try:
        from tensorflow.keras.applications import EfficientNetB0

        base = EfficientNetB0(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(img_size, img_size, 3),
        )
        x = base.output
    except Exception as e:
        print("EfficientNetB0 import or weight loading failed:", e)
        print("Falling back to a lightweight CNN.")
        inputs = tf.keras.Input(shape=(img_size, img_size, 3))
        x = Conv2D(32, (3, 3), activation="relu")(inputs)
        x = MaxPooling2D()(x)
        x = Conv2D(64, (3, 3), activation="relu")(x)
        x = GlobalAveragePooling2D()(x)

    predictions = Dense(num_classes, activation="softmax", dtype="float32")(x)
    model = Model(
        inputs=base.input if "base" in locals() else inputs, outputs=predictions
    )
    model.compile(
        optimizer="nadam",
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )
    return model


num_classes = train_labels.shape[1]

print("Training model...")
with strategy.scope():
    model = get_model(num_classes)

import math

steps_per_epoch = math.ceil(len(train_paths) / BATCH_SIZE)
validation_steps = math.ceil(len(valid_paths) / BATCH_SIZE)

model.fit(
    train_dataset,
    epochs=20,
    steps_per_epoch=steps_per_epoch,
    validation_data=valid_dataset,
    validation_steps=validation_steps,
    verbose=0,  # suppress per‑batch logging for speed
)

print("Generating predictions on test set...")
probs = model.predict(test_dataset, verbose=0)

sub.iloc[:, 1:] = probs.astype(np.float32)
sub.to_csv("submission.csv", index=False)
print("Saved submission.csv")
sub.head()
