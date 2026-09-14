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

0.5125

# 6. Current score

0.68299

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.71774) has done: 'I fix the runtime crash by removing `tensorflow.python` imports (they trigger the protobuf `MessageFactory.GetPrototype` error under TF 2.18 / protobuf 6) and using the public `tf.keras` API instead. I also correct the Kaggle input paths to the actual dataset folder (`../input/aerial-cactus-identification/...`) so images and CSVs are found reliably. To ensure a valid submission for AUC, I output the positive-class probability (`has_cactus=1`) rather than an argmax class label, and I align prediction order to `sample_submission.csv` ids to avoid id/pred mismatch. Finally, I replace deprecated `fit_generator/predict_generator` calls with `fit/predict` (same semantics) so the notebook runs end-to-end and writes `submission.csv`.'
- What this solution (achieved 0.76806) has done: 'I fix the TensorFlow/protobuf crash by removing the forced `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` setting (it commonly triggers `MessageFactory.GetPrototype` issues under TF 2.18 + protobuf 6) and by keeping imports strictly on the public `tf.keras` API. I also harden dataset path resolution so `train.csv` and `sample_submission.csv` are found reliably in Kaggle’s `/kaggle/input/...` layout, which fixes the `FileNotFoundError` and prevents downstream `NameError`s. Finally, I ensure the submission is written as `submission.csv` with `id` aligned exactly to `sample_submission.csv` order and `has_cactus` as a probability (positive class), which is required for AUC.'
- What this solution (achieved 0.68299) has done: 'The crash happens before training because TensorFlow 2.18 with protobuf 6 can error if `tensorflow` is imported before ensuring the protobuf runtime uses the pure-Python implementation. I fix this by setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` **before** importing TensorFlow (and removing the later `pop()`), which is the minimal change to make the notebook run end-to-end. I also keep everything else (model, augmentation, training loop, and submission logic) identical so the score behavior stays essentially the same (already well above the target). Finally, I keep the same dataset path resolution and ensure `submission.csv` is produced in the required format.'

# 9. Code solution

## === cell 0
import os
from os.path import join

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing.image import (
    load_img,
    img_to_array,
    ImageDataGenerator,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split

np.random.seed(1)
tf.random.set_seed(1)

BASE_CANDIDATES = [
    "/kaggle/input/aerial-cactus-identification/aerial-cactus-identification",
    "/kaggle/input/aerial-cactus-identification",
    "../input/aerial-cactus-identification/aerial-cactus-identification",
    "../input/aerial-cactus-identification",
]
BASE_DIR = None
for c in BASE_CANDIDATES:
    if os.path.exists(join(c, "train.csv")) and os.path.exists(
        join(c, "sample_submission.csv")
    ):
        if os.path.isdir(join(c, "train")) and os.path.isdir(join(c, "test")):
            BASE_DIR = c
            break

if BASE_DIR is None:
    raise FileNotFoundError(
        "Could not locate aerial-cactus-identification dataset. Tried: "
        + str(BASE_CANDIDATES)
    )

train_img_dir = join(BASE_DIR, "train")
test_img_dir = join(BASE_DIR, "test")
train_csv_path = join(BASE_DIR, "train.csv")
sample_sub_path = join(BASE_DIR, "sample_submission.csv")

train_data = pd.read_csv(train_csv_path)

img_size = 32


def prep_imgs(img_paths, img_height=img_size, img_width=img_size):
    imgs = [load_img(img, target_size=(img_height, img_width)) for img in img_paths]
    img_arr = np.array([img_to_array(img) for img in imgs], dtype=np.float32) / 255.0
    return img_arr


train_img_paths = [join(train_img_dir, img_id) for img_id in train_data["id"].values]
X = prep_imgs(train_img_paths)
y = train_data["has_cactus"].values.astype(np.int64)  # sparse labels (0/1)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)

model = Sequential()
model.add(
    Conv2D(
        25,
        kernel_size=2,
        strides=2,
        activation="relu",
        input_shape=(img_size, img_size, 3),
    )
)
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Conv2D(25, kernel_size=2, strides=2, activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Flatten())
model.add(Dense(250, activation="relu"))
model.add(Dense(2, activation="softmax"))

model.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)

batch_size = 32

train_datagen = ImageDataGenerator(
    featurewise_center=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.2, 0.5],
    horizontal_flip=True,
)
train_datagen.fit(X_train)

model.summary()



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_generator = train_datagen.flow(
    X_train, y_train, batch_size=batch_size, shuffle=True
)

history = model.fit(
    train_generator,
    steps_per_epoch=max(1, len(X_train) // batch_size),
    validation_data=(X_val, y_val),
    epochs=1,
    verbose=2,
)



## === cell 2
sample_sub = pd.read_csv(sample_sub_path)
test_ids = sample_sub["id"].values
test_img_paths = [join(test_img_dir, img_id) for img_id in test_ids]

X_test = prep_imgs(test_img_paths)

preds_proba = model.predict(X_test, batch_size=256, verbose=0)[:, 1]

output = pd.DataFrame({"id": test_ids, "has_cactus": preds_proba.astype(np.float32)})
output.to_csv("submission.csv", index=False)

print("BASE_DIR:", BASE_DIR)
print("Wrote submission.csv with shape:", output.shape)
print(output.head())
