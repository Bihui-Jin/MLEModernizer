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
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                sample_submission.csv (184 lines)
                ... and 2 other files
                images/
                    Train_744.jpg (218.7 kB)
                    Train_541.jpg (194.0 kB)
                    ... and 1819 other files
        input/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                sample_submission.csv (184 lines)
                ... and 2 other files
                images/
                    Train_744.jpg (218.7 kB)
                    Train_541.jpg (194.0 kB)
                    ... and 1819 other files
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                sample_submission.csv (184 lines)
                ... and 2 other files
                images/
                    Train_744.jpg (218.7 kB)
                    Train_541.jpg (194.0 kB)
                    ... and 1819 other files
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
Here is some information about the columns:
healthy (float64) has 1 unique values: [0.25]
image_id (object) has 183 unique values. Some example values: ['Test_0', 'Test_115', 'Test_117', 'Test_118']
multiple_diseases (float64) has 1 unique values: [0.25]
rust (float64) has 1 unique values: [0.25]
scab (float64) has 1 unique values: [0.25]

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
Here is some information about the columns:
image_id (object) has 183 unique values. Some example values: ['Test_0', 'Test_115', 'Test_117', 'Test_118']

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
Here is some information about the columns:
healthy (int64) has 2 unique values: [0, 1]
image_id (object) has 1638 unique values. Some example values: ['Train_0', 'Train_1088', 'Train_1098', 'Train_1097']
multiple_diseases (int64) has 2 unique values: [0, 1]
rust (int64) has 2 unique values: [1, 0]
scab (int64) has 2 unique values: [0, 1]

-> input/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
Here is some information about the columns:
healthy (float64) has 1 unique values: [0.25]
image_id (object) has 183 unique values. Some example values: ['Test_0', 'Test_115', 'Test_117', 'Test_118']
multiple_diseases (float64) has 1 unique values: [0.25]
rust (float64) has 1 unique values: [0.25]
scab (float64) has 1 unique values: [0.25]

-> input/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
Here is some information about the columns:
image_id (object) has 183 unique values. Some example values: ['Test_0', 'Test_115', 'Test_117', 'Test_118']

-> input/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
Here is some information about the columns:
healthy (int64) has 2 unique values: [0, 1]
image_id (object) has 1638 unique values. Some example values: ['Train_0', 'Train_1088', 'Train_1098', 'Train_1097']
multiple_diseases (int64) has 2 unique values: [0, 1]
rust (int64) has 2 unique values: [1, 0]
scab (int64) has 2 unique values: [0, 1]

-> working/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
Here is some information about the columns:
healthy (float64) has 1 unique values: [0.25]
image_id (object) has 183 unique values. Some example values: ['Test_0', 'Test_115', 'Test_117', 'Test_118']
multiple_diseases (float64) has 1 unique values: [0.25]
rust (float64) has 1 unique values: [0.25]
scab (float64) has 1 unique values: [0.25]

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler,
)
from sklearn.model_selection import train_test_split

from tensorflow.keras.applications import EfficientNetB7

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except Exception:
    strategy = tf.distribute.get_strategy()

print("REPLICAS:", strategy.num_replicas_in_sync)

AUTO = tf.data.experimental.AUTOTUNE
EPOCHS = 40
BATCH_SIZE = 8 * strategy.num_replicas_in_sync




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
    img = tf.image.random_flip_up_down(img, seed=seed)
    return (img, label) if label is not None else img


train_ds = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .map(decode_image, num_parallel_calls=AUTO)
    .map(augment, num_parallel_calls=AUTO)
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

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    steps_per_epoch=len(train_paths) // BATCH_SIZE,
    validation_data=valid_ds,
    callbacks=[lr_callback, checkpoint],
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
