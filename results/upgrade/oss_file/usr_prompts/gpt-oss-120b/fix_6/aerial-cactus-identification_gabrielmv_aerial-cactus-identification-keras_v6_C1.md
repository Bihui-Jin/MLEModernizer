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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tf_keras==2.18.0
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

0.9991

# 6. Current score

0.5023

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.49014) has done: 'Fix protobuf import error, correctly load only image files from the test directory, and ensure the test predictions are generated before creating the submission file.'
- What this solution (achieved 0.49921) has done: 'I add a TensorFlow random seed for reproducibility, import the missing `ImageDataGenerator` and `MaxPooling2D`, and expand the CNN with larger filter sizes and pooling layers to give the model more capacity. I also introduce modest data augmentation and train for more epochs (80) using the generator, which should raise the validation AUC toward the target while keeping the overall workflow unchanged.'
- What this solution (achieved 0.49993) has done: 'I modestly adjust the model and training to improve its ability to learn the data while keeping the overall structure unchanged. The changes add an AUC metric, slightly reduce dropout rates, compute class‑weights to handle label imbalance, and train a few more epochs using those weights. These tweaks should lift the validation AUC from the current ~0.50 toward the target without overhauling the core architecture.'
- What this solution (achieved 0.5023) has done: 'I adjust the training callbacks to monitor the AUC metric (the target evaluation) instead of accuracy/loss, and after training reload the best‑AUC weights before computing validation and test predictions. This small change keeps the model architecture untouched while steering training toward higher AUC, moving the score nearer the target.'

# 9. Code solution

## === cell 0
import sys
import types
import google.protobuf.message_factory as mf

if not hasattr(mf.MessageFactory, "GetPrototype"):

    def _GetPrototype(self, descriptor):
        return self.GetMessageClass(descriptor)

    mf.MessageFactory.GetPrototype = _GetPrototype

import os
import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm, tqdm_notebook

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

import tensorflow as tf

tf.random.set_seed(42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    BatchNormalization,
    Dropout,
    GlobalAveragePooling2D,
    MaxPooling2D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

np.random.seed(42)




## === cell 1
train_path = "../input/train/train/"
test_path = "../input/test/test/"
train_csv_path = "../input/train.csv"
sample_submission_path = "../input/sample_submission.csv"




## === cell 2
train_data = pd.read_csv(train_csv_path)




## === cell 3
def create_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, activation="relu", input_shape=(32, 32, 3)))
    model.add(Conv2D(32, kernel_size=3, activation="relu"))
    model.add(MaxPooling2D())
    model.add(BatchNormalization())
    model.add(Dropout(0.20))  # reduced dropout
    model.add(Conv2D(64, kernel_size=3, activation="relu"))
    model.add(Conv2D(64, kernel_size=3, activation="relu"))
    model.add(MaxPooling2D())
    model.add(BatchNormalization())
    model.add(Dropout(0.20))  # reduced dropout
    model.add(Conv2D(128, kernel_size=3, activation="relu"))
    model.add(Conv2D(128, kernel_size=3, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.20))  # reduced dropout
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.30))  # reduced dropout
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],  # added AUC metric
    )
    return model




## === cell 4
def plot_training_curves(history):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, "r", label="Training loss")
    plt.plot(epochs, val_loss, "g", label="Validation loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, "r", label="Training acc")
    plt.plot(epochs, val_acc, "g", label="Validation acc")
    plt.title("Accuracy")
    plt.legend()
    plt.show()




## === cell 5
checkpoint_path = "weights-aerial-cactus.h5"
callbacks = [
    ModelCheckpoint(
        checkpoint_path,
        monitor="val_auc",  # monitor AUC instead of accuracy
        verbose=1,
        save_best_only=True,
        mode="max",
    ),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=3, verbose=1, mode="min", min_lr=1e-5
    ),
    EarlyStopping(
        monitor="val_auc",  # stop based on AUC improvement
        min_delta=1e-4,
        patience=8,
        verbose=1,
        mode="max",
        restore_best_weights=False,  # we will reload from checkpoint explicitly
    ),
]




## === cell 6
images_train = []
labels_train = []

image_ids = train_data["id"].values
for img_id in tqdm_notebook(image_ids, desc="Loading train images"):
    img = cv.imread(os.path.join(train_path, img_id))
    if img is None:
        raise FileNotFoundError(f"Image {img_id} not found in {train_path}")
    images_train.append(img)
    label = train_data.loc[train_data["id"] == img_id, "has_cactus"].values[0]
    labels_train.append(label)

images_train = np.array(images_train, dtype="float32") / 255.0
labels_train = np.array(labels_train, dtype="float32")




## === cell 7
x_train, x_val, y_train, y_val = train_test_split(
    images_train, labels_train, test_size=0.15, stratify=labels_train, random_state=42
)




## === cell 8
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    fill_mode="nearest",
)
datagen.fit(x_train)




## === cell 9
unique, counts = np.unique(y_train, return_counts=True)
class_weights = {int(u): (len(y_train) / (2 * c)) for u, c in zip(unique, counts)}

model = create_model()
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    steps_per_epoch=len(x_train) // 32,
    epochs=120,  # extended epochs
    validation_data=(x_val, y_val),
    class_weight=class_weights,  # apply class weights
    callbacks=callbacks,
    verbose=1,
)

model.load_weights(checkpoint_path)




## === cell 10
val_pred_prob = model.predict(x_val, verbose=0).ravel()
val_auc = roc_auc_score(y_val, val_pred_prob)
print(f"Validation AUC: {val_auc:.5f}")




## === cell 11
test_ids = []
for fname in sorted(os.listdir(test_path)):
    if fname.lower().endswith(".jpg"):
        test_ids.append(fname)

images_test = []
for img_id in tqdm_notebook(test_ids, desc="Loading test images"):
    img = cv.imread(os.path.join(test_path, img_id))
    if img is None:
        raise FileNotFoundError(f"Test image {img_id} not found in {test_path}")
    images_test.append(img)

images_test = np.array(images_test, dtype="float32") / 255.0

test_preds = model.predict(images_test, verbose=1).ravel()




## === cell 12
submission = pd.read_csv(sample_submission_path)
submission["has_cactus"] = test_preds
submission.to_csv("submission.csv", index=False)
print("Submission file written to submission.csv")
