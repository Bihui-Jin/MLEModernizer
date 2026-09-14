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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import random
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

try:
    import tensorflow as tf
    from tensorflow import keras
except Exception as e:
    tf = None
    keras = None
    print("TensorFlow import failed, falling back to dummy predictions:", e)

random.seed(42)
np.random.seed(42)
if tf is not None:
    tf.random.set_seed(42)
    tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count())
    tf.config.threading.set_inter_op_parallelism_threads(os.cpu_count())



## === cell 1
train_df = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")



## === cell 2
submissions = pd.read_csv("../input/plant-pathology-2021-fgvc8/sample_submission.csv")
submissions.head()



## === cell 3
label_lists = train_df["labels"].apply(lambda x: x.split())
mlb = MultiLabelBinarizer()
multihot = mlb.fit_transform(label_lists)

train_df[mlb.classes_] = multihot
num_classes = len(mlb.classes_)



## === cell 4
if tf is not None:
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 512  # larger batch reduces number of steps per epoch
    TRAIN_DIR = "../input/plant-pathology-2021-fgvc8/train_images"
    TEST_DIR = "../input/plant-pathology-2021-fgvc8/test_images"

    train_paths = train_df["image"].apply(lambda x: os.path.join(TRAIN_DIR, x)).tolist()
    train_labels = multihot.astype(np.float32)

    def _load_and_preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_ds = train_ds.map(
        _load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE
    ).cache()

    augmentation = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.04),  # ~15 degrees
            keras.layers.RandomTranslation(0.1, 0.1),
            keras.layers.RandomZoom(0.1, 0.1),
        ]
    )

    train_ds = train_ds.map(
        lambda img, lbl: (augmentation(img, training=True), lbl),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    train_ds = train_ds.shuffle(buffer_size=20000, seed=42)
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    test_paths = (
        submissions["image"].apply(lambda x: os.path.join(TEST_DIR, x)).tolist()
    )
    test_ds = tf.data.Dataset.from_tensor_slices(test_paths)

    def _load_test(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        return img

    test_ds = test_ds.map(_load_test, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(128).prefetch(tf.data.AUTOTUNE)
else:
    train_ds = None
    test_ds = None



## === cell 5
if tf is not None:
    base_model = keras.applications.MobileNetV2(
        weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,)
    )
    base_model.trainable = False

    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = base_model(inputs, training=False)  # feature extractor
    x = keras.layers.GlobalAveragePooling2D()(x)  # pool
    outputs = keras.layers.Dense(num_classes, activation="sigmoid")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
    )
else:
    model = None



## === cell 6
EPOCHS = 3
if tf is not None and train_ds is not None:
    steps_per_epoch = math.ceil(len(train_paths) / BATCH_SIZE)
    model.fit(
        train_ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        verbose=1,
    )
else:
    print("Skipping model training due to missing TensorFlow or dataset.")



## === cell 7
if tf is not None and model is not None and test_ds is not None:
    preds = model.predict(test_ds, verbose=1)
else:
    preds = np.zeros((len(submissions), num_classes))



## === cell 8
thresh = 0.5
pred_labels = []
for prob in preds:
    idxs = np.where(prob >= thresh)[0]
    if len(idxs) == 0:
        idxs = [np.argmax(prob)]
    pred_labels.append(" ".join(mlb.classes_[idxs]))

submissions["labels"] = pred_labels
submissions.to_csv("submission.csv", index=False)



## === cell 9
submissions.head()
