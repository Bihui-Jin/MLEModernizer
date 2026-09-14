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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.7

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
pillow==11.3.0
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
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Target score

0.9646

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt

BASE_DIR = "/kaggle/input/aerial-cactus-identification"
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test")
AUTOTUNE = tf.data.AUTOTUNE

df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_df, val_df = train_test_split(
    df, train_size=0.70, random_state=0, stratify=df["has_cactus"]
)
n_training_items = len(train_df)
n_validation_items = len(val_df)




## === cell 2
def img_path(fname, is_test=False):
    """Return full path for a given filename."""
    return (
        os.path.join(TEST_IMG_DIR, fname)
        if is_test
        else os.path.join(TRAIN_IMG_DIR, fname)
    )


train_image_paths = [img_path(f) for f in train_df["id"]]
train_image_labels = train_df["has_cactus"].astype("int32").tolist()

val_image_paths = [img_path(f) for f in val_df["id"]]
val_image_labels = val_df["has_cactus"].astype("int32").tolist()

test_filenames = sorted(os.listdir(TEST_IMG_DIR))
test_image_paths = [img_path(f, is_test=True) for f in test_filenames]




## === cell 3
def load_and_preprocess_image(path):
    """Read an image file, decode, resize to 32x32 and normalize."""
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [32, 32])
    image = image / 255.0  # scale to [0,1]
    return image


def load_image_with_label(path, label):
    return load_and_preprocess_image(path), label




## === cell 4
train_ds = tf.data.Dataset.from_tensor_slices((train_image_paths, train_image_labels))
val_ds = tf.data.Dataset.from_tensor_slices((val_image_paths, val_image_labels))
test_ds = tf.data.Dataset.from_tensor_slices(test_image_paths)

BATCH_SIZE = 32

train_ds = (
    train_ds.shuffle(buffer_size=n_training_items, reshuffle_each_iteration=True)
    .repeat()
    .map(load_image_with_label, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

val_ds = (
    val_ds.map(load_image_with_label, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

test_ds = (
    test_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)




## === cell 5
model = keras.Sequential(
    [
        layers.Conv2D(
            filters=12,
            kernel_size=3,
            strides=1,
            padding="valid",
            activation=tf.nn.leaky_relu,
            input_shape=(32, 32, 3),
        ),
        layers.Conv2D(
            filters=32, kernel_size=3, padding="same", activation=tf.nn.leaky_relu
        ),
        layers.Conv2D(
            filters=32, kernel_size=3, padding="same", activation=tf.nn.leaky_relu
        ),
        layers.AveragePooling2D(pool_size=3, strides=2),
        layers.Conv2D(
            filters=64, kernel_size=3, padding="same", activation=tf.nn.leaky_relu
        ),
        layers.Conv2D(
            filters=64, kernel_size=3, padding="same", activation=tf.nn.leaky_relu
        ),
        layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        layers.Flatten(),
        layers.Dense(units=2, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
model.summary()




## === cell 6
steps_per_epoch = math.ceil(n_training_items / BATCH_SIZE)
validation_steps = math.ceil(n_validation_items / BATCH_SIZE)

model.fit(
    train_ds,
    epochs=10,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=validation_steps,
)




## === cell 7
val_loss, val_acc = model.evaluate(val_ds, steps=validation_steps)
print(f"Validation loss: {val_loss:.4f}, accuracy: {val_acc:.4f}")




## === cell 8
pred_probs = model.predict(test_ds, steps=math.ceil(len(test_image_paths) / BATCH_SIZE))
has_cactus_probs = pred_probs[:, 1]




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
FailedPreconditionError                   Traceback (most recent call last)
/tmp/ipykernel_55/3084076784.py in <cell line: 0>()
      1 # Predict probabilities for the test set
----> 2 pred_probs = model.predict(test_ds, steps=math.ceil(len(test_image_paths) / BATCH_SIZE))
      3 # Take probability of class 1 (has_cactus)
      4 has_cactus_probs = pred_probs[:, 1]
      5 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

FailedPreconditionError: {{function_node __wrapped__IteratorGetNext_output_types_1_device_/job:localhost/replica:0/task:0/device:CPU:0}} /kaggle/input/aerial-cactus-identification/test/test; Is a directory
	 [[{{node ReadFile}}]] [Op:IteratorGetNext] name: 

## === cell 9
submission_df = pd.DataFrame({"id": test_filenames, "has_cactus": has_cactus_probs})
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/942777046.py in <cell line: 0>()
      1 # Build submission DataFrame and write to CSV
----> 2 submission_df = pd.DataFrame({"id": test_filenames, "has_cactus": has_cactus_probs})
      3 submission_path = "submission.csv"
      4 submission_df.to_csv(submission_path, index=False)
      5 print(f"Submission written to {submission_path}")

NameError: name 'has_cactus_probs' is not defined
