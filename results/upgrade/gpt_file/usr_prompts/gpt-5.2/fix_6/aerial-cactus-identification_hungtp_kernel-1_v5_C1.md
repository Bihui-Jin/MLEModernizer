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
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
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

0.8791

# 6. Current score

0.99886

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.99888) has done: 'I fix the TensorFlow import crash caused by an incompatibility between TensorFlow 2.18 and protobuf 6 by forcing the pure-Python protobuf implementation before importing TensorFlow. Then I robustly resolve the train/test image directories by probing the real folder structure (your current code mistakenly builds `/train/train` and `/test/test`, causing the `ReadFile` NOT_FOUND errors). Finally, I keep the same preprocessing/model/training logic but ensure prediction runs and a correctly formatted `submission.csv` is always written with the required columns and row alignment.'
- What this solution (achieved 0.99686) has done: 'I fix the TensorFlow import crash by pinning protobuf to the compatible pure-Python implementation *and* forcing the C++ implementation off before importing TensorFlow (this resolves the `MessageFactory.GetPrototype` issue seen with TF 2.18 + protobuf 6). I keep the rest of your pipeline (data discovery, tf.data input pipeline, CNN architecture, training loop, and submission formatting) unchanged to avoid unnecessary score changes since your current score is already far above the target. I also keep the robust dataset path probing exactly as-is so image directories resolve correctly and the submission file is always written. The result run end-to-end and produce `submission.csv` with the required columns and row alignment.'
- What this solution (achieved 0.99886) has done: 'I fix the runtime crash by setting protobuf environment variables to force the pure-Python implementation before importing TensorFlow (TF 2.18 + protobuf 6 otherwise fails to import). I also keep your original CNN, tf.data pipeline, training loop, and submission formatting unchanged, only adding robust dataset path resolution so `TRAIN_DIR`/`TEST_DIR` always point to the actual image folders and file reads don’t fail. Finally, I ensure the script always writes `submission.csv` with exactly the required `id,has_cactus` columns and correct row alignment. These changes are necessary for end-to-end execution and should yield a strong AUC (likely well above the target since your core model is already capable).'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ["PROTOCOL_BUFFERS_PYTHON_DISABLE_CPP_IMPLEMENTATION"] = "1"

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.models as km
import tensorflow.keras.layers as kl

print("TensorFlow:", tf.__version__)

DATA_ROOT_CANDIDATES = [
    "/kaggle/input/aerial-cactus-identification",
    "/kaggle/data/aerial-cactus-identification",
    "/kaggle/input",
    "/kaggle/data",
    "../input/aerial-cactus-identification",
    "../input",
]

DATA_ROOT = None
for p in DATA_ROOT_CANDIDATES:
    if not os.path.exists(p):
        continue
    if os.path.exists(os.path.join(p, "train.csv")) and os.path.exists(
        os.path.join(p, "train")
    ):
        DATA_ROOT = p
        break
    if os.path.exists(os.path.join(p, "aerial-cactus-identification", "train.csv")):
        DATA_ROOT = os.path.join(p, "aerial-cactus-identification")
        break

if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate aerial-cactus-identification dataset folder."
    )

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")


def _resolve_image_dir(root, split):
    cands = [
        os.path.join(root, split),
        os.path.join(root, split, split),  # some copies are nested
        os.path.join(root, "aerial-cactus-identification", split),
        os.path.join(root, "aerial-cactus-identification", split, split),
    ]
    for d in cands:
        if os.path.isdir(d):
            try:
                for fn in os.listdir(d)[:100]:
                    if fn.lower().endswith(".jpg"):
                        return d
            except Exception:
                pass
    for d in cands:
        if os.path.isdir(d):
            return d
    raise FileNotFoundError(
        f"Could not find image directory for split='{split}' under: {root}"
    )


TRAIN_DIR = _resolve_image_dir(DATA_ROOT, "train")
TEST_DIR = _resolve_image_dir(DATA_ROOT, "test")

print("DATA_ROOT:", DATA_ROOT)
print("TRAIN_DIR:", TRAIN_DIR)
print("TEST_DIR :", TEST_DIR)

train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(SAMPLE_SUB)

AUTOTUNE = tf.data.AUTOTUNE



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_image_names = train["id"].astype(str)
ytrain = train["has_cactus"].astype(np.int32).values



## === cell 2
train_image_names.values[:5], ytrain[:5]



## === cell 3
train_image_paths = np.array(
    [os.path.join(TRAIN_DIR, fn) for fn in train_image_names.values]
)



## === cell 4
train_image_paths[:3]



## === cell 5
image = train_image_paths[0]
image



## === cell 6
ytrain[:10]




## === cell 7
def preprocess_image(image_bytes):
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, [32, 32])
    image = tf.cast(image, tf.float32) / 255.0
    return image




## === cell 8
def load_and_preprocess_image(path):
    image_bytes = tf.io.read_file(path)
    return preprocess_image(image_bytes)




## === cell 9
path_ds = tf.data.Dataset.from_tensor_slices(train_image_paths)



## === cell 10
ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)



## === cell 11
labels_ds = tf.data.Dataset.from_tensor_slices(ytrain)



## === cell 12
ds_label_ds = tf.data.Dataset.zip((ds, labels_ds))



## === cell 13
ds_label_ds = ds_label_ds.shuffle(
    buffer_size=len(train), reshuffle_each_iteration=True
).repeat()



## === cell 14
BATCH_SIZE = 30
ds_label_ds = ds_label_ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)



## === cell 15
ds_label_ds



## === cell 16
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
)  # kept to preserve original imports/intent



## === cell 17
model = km.Sequential(
    [
        kl.Conv2D(
            filters=32,
            kernel_size=3,
            padding="same",
            input_shape=(32, 32, 3),
            activation=tf.nn.relu,
        ),
    ]
)



## === cell 18
model.add(kl.Conv2D(32, (3, 3)))
model.add(kl.Activation("relu"))
model.add(kl.MaxPooling2D(pool_size=(2, 2)))
model.add(kl.Dropout(0.25))

model.add(kl.Conv2D(64, (3, 3), padding="same"))
model.add(kl.Activation("relu"))
model.add(kl.Conv2D(64, (3, 3)))
model.add(kl.Activation("relu"))
model.add(kl.MaxPooling2D(pool_size=(2, 2)))
model.add(kl.Dropout(0.25))

model.add(kl.Conv2D(64, (3, 3), padding="same"))
model.add(kl.Activation("relu"))
model.add(kl.Conv2D(64, (3, 3)))
model.add(kl.Activation("relu"))
model.add(kl.MaxPooling2D(pool_size=(2, 2)))
model.add(kl.Dropout(0.25))

model.add(kl.Flatten())
model.add(kl.Dense(512))
model.add(kl.Activation("relu"))
model.add(kl.Dropout(0.5))

model.add(kl.Dense(1))
model.add(kl.Activation("sigmoid"))



## === cell 19
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=["accuracy"],
)



## === cell 20
EPOCHS = 5
steps_per_epoch = int(np.ceil(len(train) / BATCH_SIZE))
history = model.fit(ds_label_ds, epochs=EPOCHS, steps_per_epoch=steps_per_epoch)



## === cell 21
test.shape



## === cell 22
test_image_names = test["id"].astype(str)



## === cell 23
test_image_paths = np.array(
    [os.path.join(TEST_DIR, fn) for fn in test_image_names.values]
)



## === cell 24
test_image_paths[:5]



## === cell 25
test_path_ds = tf.data.Dataset.from_tensor_slices(test_image_paths)



## === cell 26
test_img_ds = test_path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)



## === cell 27
test_ds = test_img_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)



## === cell 28
pre = model.predict(test_ds, verbose=1)



## === cell 29
pre.shape



## === cell 30
pre_prob = pre.reshape(-1).astype(np.float32)
pre_prob[:10]



## === cell 31
submission = pd.DataFrame({"id": test_image_names.values, "has_cactus": pre_prob})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with", len(submission), "rows")
assert (
    submission.shape[0] == test.shape[0]
), "Submission row count must match sample_submission."
assert list(submission.columns) == [
    "id",
    "has_cactus",
], "Submission columns must be: id,has_cactus"
