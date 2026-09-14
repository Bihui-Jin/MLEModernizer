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

3.9

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
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import multiprocessing

tf.config.threading.set_intra_op_parallelism_threads(multiprocessing.cpu_count())
tf.config.threading.set_inter_op_parallelism_threads(multiprocessing.cpu_count())

from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy(mixed_precision.Policy("mixed_float16"))
tf.config.optimizer.set_jit(True)  # XLA acceleration
tf.random.set_seed(42)  # ensure deterministic behavior



## === cell 1
DATA_ROOT = "../input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_SUBMISSION_CSV = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")

train_df = pd.read_csv(TRAIN_CSV)
submissions = pd.read_csv(TEST_SUBMISSION_CSV)



## === cell 2
IMG_H, IMG_W = 256, 256
BATCH_SIZE = 128  # larger batch reduces steps per epoch
THRESH = 0.25
EPOCHS = 5
RANDOM_STATE = 42



## === cell 3
label_lists = train_df["labels"].apply(lambda x: x.split()).tolist()
mlb = MultiLabelBinarizer()
mlb.fit(label_lists)
num_classes = len(mlb.classes_)

train_labels = mlb.transform(label_lists)



## === cell 4
train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.1,
    random_state=RANDOM_STATE,
    stratify=train_labels.argmax(axis=1),  # simple stratification
)

train_paths = (
    train_df.loc[train_idx, "image"]
    .apply(lambda x: os.path.join(TRAIN_IMG_DIR, x))
    .values
)
val_paths = (
    train_df.loc[val_idx, "image"]
    .apply(lambda x: os.path.join(TRAIN_IMG_DIR, x))
    .values
)

train_y = train_labels[train_idx]
val_y = train_labels[val_idx]




## === cell 5
def _parse_image(filename, label):
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.resize(image, [IMG_H, IMG_W])
    image = tf.image.convert_image_dtype(image, tf.float16)
    return image, tf.cast(label, tf.float32)  # loss expects float32 labels


train_ds = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_y))
    .map(_parse_image, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    .shuffle(buffer=1024, seed=RANDOM_STATE, reshuffle_each_iteration=False)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((val_paths, val_y))
    .map(_parse_image, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)



## === cell 6
base_model = keras.applications.EfficientNetB4(
    include_top=False, weights="imagenet", input_shape=(IMG_H, IMG_W, 3), pooling="avg"
)

inputs = keras.Input(shape=(IMG_H, IMG_W, 3))
x = base_model(inputs, training=False)
outputs = keras.layers.Dense(num_classes, activation="sigmoid", dtype="float32")(
    x
)  # keep final logits in float32
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
)



## === cell 7
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2)



## === cell 8
test_paths = submissions["image"].apply(lambda x: os.path.join(TEST_IMG_DIR, x)).values

test_ds = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(
        lambda fp: _parse_image(fp, tf.zeros([num_classes]))[0],
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)



## === cell 9
preds = model.predict(test_ds, verbose=1)



## === cell 10
pred_labels = []
for prob_vec in preds:
    idx = np.where(prob_vec >= THRESH)[0]
    if len(idx) == 0:
        idx = [np.argmax(prob_vec)]
    lbls = mlb.classes_[idx]
    pred_labels.append(" ".join(lbls))

submissions["labels"] = pred_labels
submissions.to_csv("submission.csv", index=False)



## === cell 11
submissions.head()
