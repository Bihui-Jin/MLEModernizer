# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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

0.4861

# 6. Current score

0.99881

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.9984) has done: 'I first fix the environment-breaking TensorFlow/protobuf import issue by removing the unnecessary IPython display and GridSearch imports and forcing a safe protobuf implementation before importing TensorFlow. Then I correct dataset paths to the actual Kaggle folder structure and repair the augmentation/parsing functions so `tf.data` pipelines work without undefined variables or invalid ranges. Next I fix Keras/TensorFlow API changes (`predict_generator`, history keys, and weights filename suffix) so training and inference run end-to-end. Finally I generate predictions for the real test set in the exact `id,has_cactus` format with the correct row count (no thresholding to 0/1, since AUC expects probabilities) and write `submission.csv`.'
- What this solution (achieved 0.99881) has done: 'I fix the immediate runtime crash caused by an incompatibility between TensorFlow 2.18 and the installed protobuf 6.x by downgrading protobuf to a TF-compatible version at runtime before importing TensorFlow. This is a minimal environment fix that preserves your model/data pipeline logic and should restore end-to-end execution. I also keep the rest of your training/inference code intact, only adding small safety checks for input paths so the notebook fails clearly if the dataset root differs. The submission writing remains unchanged and still produce a valid `submission.csv` with `id,has_cactus` probabilities.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess


def _ensure_compatible_protobuf():
    try:
        import google.protobuf as gp  # noqa: F401
        from google.protobuf import __version__ as pb_ver

        if pb_ver.startswith("6.") or pb_ver.startswith("5."):
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
            )
            for k in list(sys.modules.keys()):
                if k.startswith("google.protobuf"):
                    del sys.modules[k]
    except Exception as e:
        print("Warning: protobuf compatibility step failed:", repr(e))


_ensure_compatible_protobuf()

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split



## === cell 1
print("TensorFlow:", tf.__version__)



## === cell 2
BASE_PATH = "/kaggle/input/aerial-cactus-identification"
TRAIN_CSV_PATH = os.path.join(BASE_PATH, "train.csv")
TRAIN_DIR = os.path.join(BASE_PATH, "train")
TEST_DIR = os.path.join(BASE_PATH, "test")
SAMPLE_SUB_PATH = os.path.join(BASE_PATH, "sample_submission.csv")

if not os.path.exists(TRAIN_CSV_PATH):
    alt_base = "/kaggle/input/aerial-cactus-identification/aerial-cactus-identification"
    if os.path.exists(os.path.join(alt_base, "train.csv")):
        BASE_PATH = alt_base
        TRAIN_CSV_PATH = os.path.join(BASE_PATH, "train.csv")
        TRAIN_DIR = os.path.join(BASE_PATH, "train")
        TEST_DIR = os.path.join(BASE_PATH, "test")
        SAMPLE_SUB_PATH = os.path.join(BASE_PATH, "sample_submission.csv")

train_csv = pd.read_csv(TRAIN_CSV_PATH)
print(train_csv.describe(include="all"))
print(train_csv.head())
print("Train images dir exists:", os.path.isdir(TRAIN_DIR))
print("Test images dir exists:", os.path.isdir(TEST_DIR))



## === cell 3
filenames = [os.path.join(TRAIN_DIR, fname) for fname in train_csv["id"].tolist()]
labels = train_csv["has_cactus"].astype(np.int32).tolist()

train_filenames, val_filenames, train_labels, val_labels = train_test_split(
    filenames,
    labels,
    train_size=0.9,
    random_state=420,
    stratify=labels,  # score-stable & avoids degenerate splits for AUC
)

size_train = len(train_filenames)
size_val = len(val_filenames)
print("Train size:", size_train, "Val size:", size_val)



## === cell 4
IMAGE_SIZE = 96
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE


def _parse_fn(filename, label):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = (tf.cast(img, tf.float32) / 127.5) - 1.0
    label = tf.cast(label, tf.float32)
    return img, label




## === cell 5
def _augment_(image, label):
    image = tf.image.random_flip_left_right(image)
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    return image, label




## === cell 6
train_data = (
    tf.data.Dataset.from_tensor_slices(
        (tf.constant(train_filenames), tf.constant(train_labels))
    )
    .map(_parse_fn, num_parallel_calls=AUTOTUNE)
    .map(_augment_, num_parallel_calls=AUTOTUNE)
    .shuffle(buffer_size=10000, reshuffle_each_iteration=True)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

val_data = (
    tf.data.Dataset.from_tensor_slices(
        (tf.constant(val_filenames), tf.constant(val_labels))
    )
    .map(_parse_fn, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)



## === cell 7
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

modelMNV2 = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights="imagenet",
)
modelMNV2.trainable = False



## === cell 8
pool_layer = tf.keras.layers.GlobalMaxPooling2D()
dense_layer = tf.keras.layers.Dense(1, activation="sigmoid")



## === cell 9
model = tf.keras.Sequential([modelMNV2, pool_layer, dense_layer])
model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()



## === cell 10
num_epochs = 1
steps_per_epoch = int(np.ceil(size_train / BATCH_SIZE))
val_steps = int(np.ceil(size_val / BATCH_SIZE))
print("steps_per_epoch:", steps_per_epoch, "val_steps:", val_steps)



## === cell 11
history = model.fit(
    train_data,
    epochs=num_epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_data,
    validation_steps=val_steps,
)



## === cell 12
model.save_weights("weights_epoch_1.weights.h5")



## === cell 13
acc = history.history.get("accuracy", [])
val_acc = history.history.get("val_accuracy", [])
loss = history.history.get("loss", [])
val_loss = history.history.get("val_loss", [])
print(
    "Last train acc:",
    acc[-1] if acc else None,
    "Last val acc:",
    val_acc[-1] if val_acc else None,
)



## === cell 14
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_ids = sample_sub["id"].tolist()
test_paths = [os.path.join(TEST_DIR, tid) for tid in test_ids]


def _parse_test(filename):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = (tf.cast(img, tf.float32) / 127.5) - 1.0
    return img


test_ds = (
    tf.data.Dataset.from_tensor_slices(tf.constant(test_paths))
    .map(_parse_test, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)



## === cell 15
pred = model.predict(test_ds, verbose=1).reshape(-1)
pred = np.clip(pred, 0.0, 1.0)

print("Pred shape:", pred.shape, "Expected:", len(test_ids))



## === cell 16
submission_df = pd.DataFrame({"id": test_ids, "has_cactus": pred.astype(np.float32)})
assert (
    submission_df.shape[0] == sample_sub.shape[0]
), "Row count mismatch vs sample submission."
submission_df.to_csv("submission.csv", index=False)
print(submission_df.head())
print("Wrote submission.csv with rows:", len(submission_df))



## === cell 17
print("submission.csv exists:", os.path.exists("submission.csv"))
print("submission.csv preview:")
with open("submission.csv", "r") as f:
    for _ in range(5):
        print(f.readline().strip())
