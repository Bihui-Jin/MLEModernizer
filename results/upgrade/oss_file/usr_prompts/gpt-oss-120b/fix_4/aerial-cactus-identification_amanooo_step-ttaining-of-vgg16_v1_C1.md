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
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tf_keras==2.18.0

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

0.9928

# 6. Current score

0.15655

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.15655) has done: 'I replace the faulty tf_keras imports with standard tensorflow.keras imports, fix missing imports, filter the test directory to only include *.jpg files (removing stray entries that caused a length mismatch), and ensure all referenced objects are defined. These changes resolve the runtime errors and produce a correctly‑sized submission CSV while keeping the original model architecture and training strategy.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization
from tensorflow.keras.applications import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

np.random.seed(2)
tf.random.set_seed(2)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
possible_dirs = [
    "/kaggle/input/aerial-cactus-identification/",
    "./input/aerial-cactus-identification/",
    "./kaggle/input/aerial-cactus-identification/",
    "./data/aerial-cactus-identification/",
]
DIRin = None
for p in possible_dirs:
    if os.path.isdir(p):
        DIRin = p
        break
if DIRin is None:
    raise FileNotFoundError("Could not locate the dataset folder.")
print("Using data folder:", DIRin)



## === cell 2
labels_path = os.path.join(DIRin, "train.csv")
labels = pd.read_csv(labels_path)
labels["has_cactus"] = labels["has_cactus"].astype(int)
print("Train labels shape:", labels.shape)



## === cell 3
test_dir = os.path.join(DIRin, "test")
test_ids = [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
tests = pd.DataFrame(test_ids, columns=["id"])
print("Test set size:", tests.shape[0])




## === cell 4
def show_image(in_set="train", n=10):
    df = labels if in_set == "train" else tests
    fig = plt.figure(figsize=(10, (n // 5) * 2))
    for idx, img_name in enumerate(np.random.choice(df["id"], n)):
        ax = fig.add_subplot(n // 5, 5, idx + 1, xticks=[], yticks=[])
        img_path = os.path.join(DIRin, in_set, img_name)
        img = Image.open(img_path)
        ax.imshow(img)
        if in_set == "train":
            lbl = df.loc[df["id"] == img_name, "has_cactus"].values[0]
            ax.set_title(f"Label: {lbl}")




## === cell 5
train_imgs = []
train_labels = []
train_dir = os.path.join(DIRin, "train")
for img_id, lab in zip(labels["id"].values, labels["has_cactus"].values):
    path = os.path.join(train_dir, img_id)
    img = cv2.imread(path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        train_imgs.append(img)
        train_labels.append(lab)
X = np.asarray(train_imgs, dtype="float32") / 255.0
y = np.asarray(train_labels, dtype="int")
print("X shape:", X.shape, "y shape:", y.shape)



## === cell 6
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=2, stratify=y
)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4019736755.py in <cell line: 0>()
----> 1 X_train, X_val, y_train, y_val = train_test_split(
      2     X, y, test_size=0.2, random_state=2, stratify=y
      3 )
      4 

NameError: name 'train_test_split' is not defined

## === cell 7
test_imgs = []
for img_id in tests["id"].values:
    path = os.path.join(test_dir, img_id)
    img = cv2.imread(path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        test_imgs.append(img)
X_test = np.asarray(test_imgs, dtype="float32") / 255.0
print("Test set shape:", X_test.shape)



## === cell 8
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=5,
    zoom_range=0.1,
    channel_shift_range=0.01,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest",
)
datagen.fit(X_train)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2106811246.py in <cell line: 0>()
----> 1 datagen = ImageDataGenerator(
      2     rotation_range=10,
      3     width_shift_range=0.1,
      4     height_shift_range=0.1,
      5     shear_range=5,

NameError: name 'ImageDataGenerator' is not defined

## === cell 9
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(32, 32, 3))
base_model.trainable = False



## === cell 10
model = Sequential(
    [
        base_model,
        Flatten(),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dense(1, activation="sigmoid"),
    ]
)
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.summary()



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2765261899.py in <cell line: 0>()
     11 )
     12 model.compile(
---> 13     optimizer=Adam(learning_rate=1e-4),
     14     loss="binary_crossentropy",
     15     metrics=["accuracy"],

NameError: name 'Adam' is not defined

## === cell 11
reduce_lr = ReduceLROnPlateau(
    monitor="val_accuracy", factor=0.5, patience=3, verbose=1, min_lr=1e-6
)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1455718057.py in <cell line: 0>()
----> 1 reduce_lr = ReduceLROnPlateau(
      2     monitor="val_accuracy", factor=0.5, patience=3, verbose=1, min_lr=1e-6
      3 )
      4 

NameError: name 'ReduceLROnPlateau' is not defined

## === cell 12
model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    epochs=12,
    validation_data=(X_val, y_val),
    callbacks=[reduce_lr],
    verbose=2,
)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1640088108.py in <cell line: 0>()
      1 # Phase 1 – train classifier head only
      2 model.fit(
----> 3     datagen.flow(X_train, y_train, batch_size=64),
      4     epochs=12,
      5     validation_data=(X_val, y_val),

NameError: name 'datagen' is not defined

## === cell 13
base_model.trainable = True
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1151927546.py in <cell line: 0>()
      2 base_model.trainable = True
      3 model.compile(
----> 4     optimizer=Adam(learning_rate=1e-5),
      5     loss="binary_crossentropy",
      6     metrics=["accuracy"],

NameError: name 'Adam' is not defined

## === cell 14
model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    epochs=12,
    validation_data=(X_val, y_val),
    callbacks=[reduce_lr],
    verbose=2,
)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2581596217.py in <cell line: 0>()
      1 model.fit(
----> 2     datagen.flow(X_train, y_train, batch_size=64),
      3     epochs=12,
      4     validation_data=(X_val, y_val),
      5     callbacks=[reduce_lr],

NameError: name 'datagen' is not defined

## === cell 15
pred_probs = model.predict(X_test, batch_size=64).flatten()
submission = pd.DataFrame({"id": tests["id"], "has_cactus": pred_probs})
submission_path = "solution_01.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}, shape:", submission.shape)
