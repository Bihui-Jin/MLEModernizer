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
Given a dataset of images of dogs and cats, predict if an image is a dog or a cat.

## Metric
Log loss.

## Submission Format
For each image in the test set, you must submit a probability that image is a dog. The file should have a header and be in the following format:

```
id,label
1,0.5
2,0.5
3,0.5
...
```

## Dataset
The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id.

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
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
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
        input/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
        working/
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
```

-> data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> working/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

# 5. Target score

0.16863

# 6. Current score

0.0486

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.04923) has done: 'The changes fix the import errors, correctly unzip (only if needed), gather image files recursively, safely load and resize images, correct the loss configuration, replace the nonexistent `predict_proba` with `predict`, and generate a proper `submission.csv` with the required columns. All modifications keep the original ResNet101‑based model untouched while making the notebook runnable end‑to‑end.'
- What this solution (achieved 0.0486) has done: 'I add a small monkey‑patch before importing TensorFlow that restores the missing `GetPrototype` method in protobuf’s `MessageFactory`, allowing TensorFlow to load without the AttributeError. The rest of the notebook remains unchanged, preserving the ResNet101 model and training procedure, so the current very good log‑loss score stays valid while fixing the runtime crash.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    from google.protobuf.message_factory import MessageFactory

    if not hasattr(MessageFactory, "GetPrototype"):
        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass  # If protobuf is not available, let TensorFlow raise its own error later

import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import zipfile
import glob
import matplotlib.pyplot as plt




## === cell 1
TEST_ZIP = "../input/dogs-vs-cats-redux-kernels-edition/test.zip"
TRAIN_ZIP = "../input/dogs-vs-cats-redux-kernels-edition/train.zip"

if not os.path.isdir("train"):
    with zipfile.ZipFile(TRAIN_ZIP, "r") as z:
        z.extractall()
if not os.path.isdir("test"):
    with zipfile.ZipFile(TEST_ZIP, "r") as z:
        z.extractall()

TRAIN_ROOT = (
    "train"
    if os.path.isdir("train")
    else os.path.join("dogs-vs-cats-redux-kernels-edition", "train")
)
TEST_ROOT = (
    "test"
    if os.path.isdir("test")
    else os.path.join("dogs-vs-cats-redux-kernels-edition", "test")
)




## === cell 2
train_images = sorted(
    glob.glob(os.path.join(TRAIN_ROOT, "**", "*.jpg"), recursive=True)
)
test_images = sorted(glob.glob(os.path.join(TEST_ROOT, "**", "*.jpg"), recursive=True))

limit = int(0.8 * len(train_images))
train_list = train_images[:limit]
val_list = train_images[limit:]




## === cell 3
rows, cols = 160, 160


def load_images(file_list):
    data = np.ndarray((len(file_list), rows, cols, 3), dtype=np.uint8)
    for idx, fp in enumerate(file_list):
        img = cv2.imread(fp)
        if img is None:  # guard against read failures
            img = np.zeros((rows, cols, 3), dtype=np.uint8)
        else:
            img = cv2.resize(img, (rows, cols), interpolation=cv2.INTER_CUBIC)
        data[idx] = img
    return data


train_data = load_images(train_list)
val_data = load_images(val_list)
test_data = load_images(test_images)




## === cell 4
train_labels = np.array(
    [1 if "dog" in os.path.basename(p).lower() else 0 for p in train_list],
    dtype=np.float32,
)
val_labels = np.array(
    [1 if "dog" in os.path.basename(p).lower() else 0 for p in val_list],
    dtype=np.float32,
)




## === cell 5
image_shape = (rows, rows, 3)

base_model = tf.keras.applications.ResNet101(
    weights="imagenet", include_top=False, input_shape=image_shape
)
base_model.trainable = False

model = tf.keras.Sequential(
    [
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.BinaryCrossentropy(),  # sigmoid output → from_logits=False
    metrics=["accuracy"],
)




## === cell 6
model.fit(
    x=train_data,
    y=train_labels,
    validation_data=(val_data, val_labels),
    batch_size=128,
    epochs=5,
    shuffle=True,
    verbose=1,
)




## === cell 7
predictions = model.predict(
    test_data, batch_size=128, verbose=1
).squeeze()  # shape (N,)




## === cell 8
test_ids = [os.path.splitext(os.path.basename(p))[0] for p in test_images]
submission = pd.DataFrame({"id": test_ids, "label": predictions})
submission.to_csv("submission.csv", index=False, header=True)

print(submission.head())
