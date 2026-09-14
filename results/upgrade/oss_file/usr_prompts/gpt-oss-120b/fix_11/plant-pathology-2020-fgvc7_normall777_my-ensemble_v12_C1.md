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

0.9711393062766024

# 6. Current score

0.4761

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.54596) has done: 'The update reduces the image resolution from 768 → 224 and enlarges the batch size, which dramatically cuts the amount of data each GPU/CPU has to process while keeping the same model architecture, training loop, and label handling. A `.cache()` call is added to the test dataset so images are decoded only once. These changes keep the deterministic seeds and overall logic intact, allowing the whole pipeline to finish well within the 600‑second limit.'
- What this solution (achieved 0.49845) has done: 'The fix adds a protobuf compatibility shim before importing TensorFlow to stop the initialization error, increases training epochs, and adds a checkpoint callback so the best‑validation‑AUC model is used for predictions—these changes should raise the ROC‑AUC toward the target while keeping the original architecture and workflow.'
- What this solution (achieved 0.51047) has done: 'We will (1) avoid the unnecessary distribution strategy and (2) use mixed‑precision only when a GPU is present, eliminating its CPU overhead. (3) Increase the batch size modestly (to 512) to cut the number of training steps while keeping the same number of epochs, which speeds up training without altering the model architecture or loss. All other logic, paths, and evaluation remain unchanged.'
- What this solution (achieved 0.4761) has done: 'I add a more robust protobuf compatibility shim before any TensorFlow import by setting both required environment variables and clearing any cached protobuf C‑extension modules. This resolves the `MessageFactory` AttributeError, allowing the model to train and produce predictions, which should raise the ROC‑AUC toward the target while keeping the original pipeline unchanged.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import sys

sys.modules.pop("google.protobuf.pyext._message", None)

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import mixed_precision
from sklearn.model_selection import train_test_split

if tf.config.list_physical_devices("GPU"):
    mixed_precision.set_global_policy("mixed_float16")
else:
    mixed_precision.set_global_policy("float32")

print("TensorFlow version:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_ROOT = "/kaggle/input/plant-pathology-2020-fgvc7"
GCS_DS_PATH = DATA_ROOT  # used by format_path helper

EPOCHS = 10  # keep original training schedule
BATCH_SIZE = 512
AUTO = tf.data.experimental.AUTOTUNE
img_size = 160




## === cell 2
def format_path(st):
    """Convert image_id to absolute file path."""
    return os.path.join(GCS_DS_PATH, "images", f"{st}.jpg")




## === cell 3
train = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"))
test = pd.read_csv(os.path.join(DATA_ROOT, "test.csv"))
sub = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))

train_paths = train["image_id"].apply(format_path).values
test_paths = test["image_id"].apply(format_path).values

label_cols = ["healthy", "multiple_diseases", "rust", "scab"]
train_labels = train[label_cols].values.astype(np.float32)

train_paths, valid_paths, train_labels, valid_labels = train_test_split(
    train_paths, train_labels, test_size=0.15, random_state=2020
)




## === cell 4
def decode_image(filename, label=None, image_size=(img_size, img_size)):
    bits = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(bits, channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.image.resize(img, image_size)
    return (img, label) if label is not None else img


def data_augment(image, label):
    image = tf.image.random_flip_left_right(image, seed=2020)
    image = tf.image.random_flip_up_down(image, seed=2020)
    return image, label


train_dataset = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .map(decode_image, num_parallel_calls=AUTO, deterministic=False)
    .cache()  # cache decoded images in memory
    .map(data_augment, num_parallel_calls=AUTO, deterministic=False)
    .shuffle(1024, seed=2020, reshuffle_each_iteration=True)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels))
    .map(decode_image, num_parallel_calls=AUTO, deterministic=False)
    .cache()
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(lambda x: decode_image(x), num_parallel_calls=AUTO, deterministic=False)
    .cache()
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)




## === cell 5
def get_model(base_cls):
    base = base_cls(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(img_size, img_size, 3),
    )
    base.trainable = False
    x = base.output
    outputs = Dense(len(label_cols), activation="sigmoid")(x)
    model = Model(inputs=base.input, outputs=outputs)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return model


model = get_model(EfficientNetB0)



## === cell 6
import math

steps_per_epoch = math.ceil(len(train_paths) / BATCH_SIZE)
validation_steps = math.ceil(len(valid_paths) / BATCH_SIZE)

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "best_weights.weights.h5",
    monitor="val_auc",
    mode="max",
    save_best_only=True,
    save_weights_only=True,
    verbose=0,
)

model.fit(
    train_dataset,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=valid_dataset,
    validation_steps=validation_steps,
    callbacks=[checkpoint_cb],
    verbose=2,
)

model.load_weights("best_weights.weights.h5")



## === cell 7
probabilities = model.predict(test_dataset, verbose=1)

sub.loc[:, label_cols] = probabilities
submission_path = "submission.csv"
sub.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
print(sub.head())
