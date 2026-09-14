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
tqdm==4.67.1

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

0.5

# 6. Current score

0.98972

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.98985) has done: 'I fix the runtime crash caused by an incompatibility between TensorFlow 2.18 and protobuf 6.x by forcing the pure-Python protobuf implementation before importing TensorFlow. I also correct the Kaggle input paths to point at the actual dataset folder structure and ensure the test directory join is correct. Finally, I make the submission filename end with `.csv` and add a small safety clip to keep probabilities in a valid range; the model/training core logic remains unchanged (so the score should stay similarly high, even though it’s above your target).'
- What this solution (achieved 0.99063) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf runtime *and* avoiding the C++ backend explicitly before TensorFlow is imported (this is the root cause of the `MessageFactory`/`GetPrototype` error in many Kaggle images). I also make the dataset base directory resolution robust to either `../input/...` or `/kaggle/input/...` so it runs reliably. Since your current score (0.98985) is far above the target (0.5) and higher-is-better, I not change model/training logic or calibration; the goal here is just to run end-to-end and produce a valid `submission.csv`. The output submission format and `.csv` suffix be preserved exactly.'
- What this solution (achieved 0.98972) has done: 'You’re hitting the known TensorFlow 2.18 + protobuf 6.x incompatibility where TensorFlow imports protobuf descriptors expecting APIs like `MessageFactory.GetPrototype`. The minimal reliable fix is to force the pure-Python protobuf runtime *and* disable the C++ protobuf backend **before any protobuf/tensorflow import** (and proactively import `google.protobuf` first to lock the implementation). I also keep your model/training logic unchanged and only add small robustness around image RGB conversion to avoid occasional grayscale/alpha-channel shape issues that can crash training/prediction. The submission writing stays identical and still produces `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_CPP"] = "1"

import google.protobuf  # noqa: F401

from glob import glob
import shutil
from tqdm import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

_CANDIDATES = [
    "../input/aerial-cactus-identification",
    "/kaggle/input/aerial-cactus-identification",
    "../input/aerial-cactus-identification/aerial-cactus-identification",
    "/kaggle/input/aerial-cactus-identification/aerial-cactus-identification",
]
BASE_DIR = None
for c in _CANDIDATES:
    if os.path.exists(os.path.join(c, "train.csv")) and os.path.isdir(
        os.path.join(c, "train")
    ):
        BASE_DIR = c
        break
if BASE_DIR is None:
    BASE_DIR = "../input/aerial-cactus-identification"

train_dir = os.path.join(BASE_DIR, "train")
test_dir = os.path.join(BASE_DIR, "test")

input_shape = (32, 32, 3)
batch_size = 32
num_classes = 2
num_epochs = 1
data_augmentation = True

learning_rate = 0.001


def build_model(input_shape):
    inputs = layers.Input(input_shape)
    net = layers.Conv2D(32, (3, 3), padding="same")(inputs)
    net = layers.Activation("relu")(net)
    net = layers.Conv2D(32, (3, 3))(net)
    net = layers.Activation("relu")(net)
    net = layers.MaxPooling2D(pool_size=(2, 2))(net)
    net = layers.Dropout(0.25)(net)

    net = layers.Conv2D(64, (3, 3), padding="same")(net)
    net = layers.Activation("relu")(net)
    net = layers.Conv2D(64, (3, 3))(net)
    net = layers.Activation("relu")(net)
    net = layers.MaxPooling2D(pool_size=(2, 2))(net)
    net = layers.Dropout(0.25)(net)

    net = layers.Flatten()(net)
    net = layers.Dense(512)(net)
    net = layers.Activation("relu")(net)
    net = layers.Dropout(0.5)(net)
    net = layers.Dense(num_classes)(net)
    net = layers.Activation("softmax")(net)

    return tf.keras.Model(inputs=inputs, outputs=net)


model = build_model(input_shape)
model.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    metrics=["accuracy"],
)

df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))

datasets = []
for fname, cls in zip(df["id"], df["has_cactus"]):
    datasets.append((fname, cls))

np.random.shuffle(datasets)
train_paths = datasets[: int(len(datasets) * 0.8)]
test_paths = datasets[int(len(datasets) * 0.8) :]


def _load_rgb_image(path):
    img = Image.open(path).convert("RGB")
    return np.array(img)


def batch_dataset(batch_paths):
    batch_images = []
    batch_labels = []

    for fname, cls in batch_paths:
        img_path = os.path.join(train_dir, fname)
        image = _load_rgb_image(img_path)
        label = np.array(np.array([0, 1]) == cls).astype(np.uint8)
        batch_images.append(image)
        batch_labels.append(label)

    batch_images = np.array(batch_images)
    batch_labels = np.array(batch_labels)
    return batch_images, batch_labels


images, labels = batch_dataset(train_paths[:4])
_ = (images.shape, labels.shape)

for epoch in range(num_epochs):
    np.random.shuffle(train_paths)
    batches_per_epoch = len(train_paths) // batch_size
    for step in tqdm(range(batches_per_epoch)):
        images, labels = batch_dataset(
            train_paths[step * batch_size : (step + 1) * batch_size]
        )
        model.fit(images, labels, batch_size=batch_size, verbose=0)

test_df = pd.read_csv(os.path.join(BASE_DIR, "sample_submission.csv"))

answers = []
for fname in test_df["id"]:
    path = os.path.join(test_dir, fname)
    image = _load_rgb_image(path)
    image = np.expand_dims(image, 0)
    logit = model.predict(image, verbose=0)
    ans = float(logit[0, 1])
    ans = float(np.clip(ans, 0.0, 1.0))
    answers.append(ans)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
submit_data = {"id": test_df["id"], "has_cactus": answers}

submit_df = pd.DataFrame(submit_data)

submit_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submit_df.shape)
print(submit_df.head())
