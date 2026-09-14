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

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
tf_keras==2.18.0

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

0.96833

# 6. Current score

0.50692

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.56927) has done: 'The changes enable XLA compilation, freeze the EfficientNetB7 backbone to avoid costly weight updates, and add an EarlyStopping callback so training can finish early if validation loss stops improving. These optimizations keep the same model architecture and loss, only reducing unnecessary computation, which preserves the original training semantics while fitting within the 600‑second limit.'
- What this solution (achieved 0.51635) has done: 'Implemented mixed‑precision training and doubled the batch size (while keeping the same model, augmentations, and training loop). Mixed‑precision speeds up the large EfficientNetB7 forward/backward passes on GPU without altering the model architecture or loss, preserving result accuracy. Increasing batch size reduces the number of steps per epoch, cutting overall runtime while maintaining the same number of epochs and early‑stopping logic. No other code parts were changed.'
- What this solution (achieved 0.50692) has done: 'I added a small environment‑variable fix to avoid the protobuf import error, switched the EfficientNetB7 import to the plain Keras version, froze the backbone so only the head is trained (which greatly improves generalisation on this small dataset), and enabled the early‑stopping callback to restore the best weights. These changes fix the runtime crash and are expected to raise the validation ROC‑AUC toward the target while keeping the original model architecture and training logic intact.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model, mixed_precision
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler,
)
from sklearn.model_selection import train_test_split

from keras.applications import EfficientNetB7

mixed_precision.set_global_policy("mixed_float16")

strategy = tf.distribute.get_strategy()
print("REPLICAS:", strategy.num_replicas_in_sync)

AUTO = tf.data.experimental.AUTOTUNE
EPOCHS = 40
BATCH_SIZE = 16 * strategy.num_replicas_in_sync



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
LOCAL_DATA_PATH = "/kaggle/input/plant-pathology-2020-fgvc7"


def format_path(st):
    return f"{LOCAL_DATA_PATH}/images/{st}.jpg"




## === cell 2
train_df = pd.read_csv(f"{LOCAL_DATA_PATH}/train.csv")
test_df = pd.read_csv(f"{LOCAL_DATA_PATH}/test.csv")
sub_df = pd.read_csv(f"{LOCAL_DATA_PATH}/sample_submission.csv")

all_paths = train_df["image_id"].apply(format_path).values
all_labels = train_df.loc[:, "healthy":].values  # 4‑column multi‑label matrix

train_paths, valid_paths, train_labels, valid_labels = train_test_split(
    all_paths,
    all_labels,
    test_size=0.06,
    random_state=2020,
    stratify=all_labels.argmax(axis=1),
)

test_paths = test_df["image_id"].apply(format_path).values




## === cell 3
IMG_SIZE = 768


def decode_image(filename, label=None, image_size=(IMG_SIZE, IMG_SIZE)):
    bits = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(bits, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # already 0‑1
    img = tf.image.resize(img, image_size)
    return (img, label) if label is not None else img


def augment(image, label=None, seed=2020):
    img = tf.image.random_flip_left_right(image, seed=seed)
    img = tf.image.random_flip_up_down(image, seed=seed)
    return (img, label) if label is not None else img


train_ds = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .map(decode_image, num_parallel_calls=AUTO)
    .map(augment, num_parallel_calls=AUTO)
    .cache()
    .shuffle(1024)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_ds = (
    tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels))
    .map(decode_image, num_parallel_calls=AUTO)
    .cache()
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

test_ds = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(lambda x: decode_image(x), num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)



## === cell 4
LR_START = 1e-5
LR_MAX = 1e-4 * strategy.num_replicas_in_sync
LR_MIN = 1e-5
LR_RAMPUP_EPOCHS = 5
LR_SUSTAIN_EPOCHS = 0
LR_EXP_DECAY = 0.8


def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        return (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        return LR_MAX
    else:
        return (LR_MAX - LR_MIN) * LR_EXP_DECAY ** (
            epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS
        ) + LR_MIN


lr_callback = LearningRateScheduler(lrfn, verbose=True)




## === cell 5
def build_model():
    base = EfficientNetB7(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base.trainable = False
    x = base.output
    out = layers.Dense(train_labels.shape[1], activation="sigmoid")(x)
    return Model(inputs=base.input, outputs=out)


with strategy.scope():
    model = build_model()
    model.compile(
        optimizer="nadam",
        loss="binary_crossentropy",
        metrics=["binary_accuracy"],
    )
model.summary()



## === cell 6
checkpoint = ModelCheckpoint(
    filepath="best_efnet_b7.h5",
    monitor="val_loss",
    save_best_only=True,
    mode="min",
    verbose=1,
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,  # ensure the best model is used later
    mode="min",
    verbose=1,
)

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    steps_per_epoch=len(train_paths) // BATCH_SIZE,
    validation_data=valid_ds,
    callbacks=[lr_callback, checkpoint, early_stop],
)



## === cell 7
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["binary_accuracy"], label="train acc")
plt.plot(history.history["val_binary_accuracy"], label="val acc")
plt.xlabel("epoch")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.xlabel("epoch")
plt.legend()
plt.show()



## === cell 8
model.load_weights("best_efnet_b7.h5")
probs = model.predict(test_ds, verbose=1)

sub_df.loc[:, "healthy":] = probs
sub_df.to_csv("submission.csv", index=False)
print("Submission written to submission.csv")
