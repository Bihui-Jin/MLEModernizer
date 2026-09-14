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

0.9944

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.51148) has done: 'The changes import the TensorFlow Keras version, add missing imports, correct variable assignments, update callbacks and metric keys, replace deprecated `predict_proba`/`predict_classes` with `predict`, and ensure the submission CSV is written with the required columns. These fixes unblock the entire pipeline and let the model train and generate a valid `aerial-cactus-submission.csv` while preserving the original CNN architecture.'
- What this solution (achieved 0.50588) has done: 'The fix replaces the failing `tensorflow.keras` imports with the compatible `keras` package (which is installed) and lowers the optimizer learning rate from 0.01 to 0.001 to stabilize training and raise AUC. No core architecture changes are made, and the script now runs end‑to‑end and writes a proper `aerial-cactus-submission.csv` file.'

# 9. Code solution

## === cell 0
import os
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Flatten,
    BatchNormalization,
    LeakyReLU,
    Dropout,
    GlobalAveragePooling2D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

tf.random.set_seed(42)
np.random.seed(42)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
print("Input dirs:", os.listdir("../input"))




## === cell 2
train_data = pd.read_csv("../input/train.csv")
print("Train rows:", len(train_data))




## === cell 3
def create_model():
    model = Sequential()

    model.add(
        Conv2D(filters=16, kernel_size=3, activation="relu", input_shape=(32, 32, 3))
    )
    model.add(Conv2D(filters=16, kernel_size=3, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=32, kernel_size=3, activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=3, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=3, activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=3, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=128, kernel_size=3, activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=3, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model




## === cell 4
def plot_training_curves(history):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    auc = history.history["auc"]
    val_auc = history.history["val_auc"]

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, loss, "r", label="Training loss")
    plt.plot(epochs, val_loss, "g", label="Validation loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, acc, "r", label="Training acc")
    plt.plot(epochs, val_acc, "g", label="Validation acc")
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, auc, "r", label="Training AUC")
    plt.plot(epochs, val_auc, "g", label="Validation AUC")
    plt.title("AUC")
    plt.legend()

    plt.show()




## === cell 5
BASE_INPUT = "../input"
TRAIN_IMG_DIR = os.path.join(BASE_INPUT, "train")
if not os.path.isdir(TRAIN_IMG_DIR):
    TRAIN_IMG_DIR = os.path.join(BASE_INPUT, "aerial-cactus-identification", "train")

TEST_IMG_DIR = os.path.join(BASE_INPUT, "test")
if not os.path.isdir(TEST_IMG_DIR):
    TEST_IMG_DIR = os.path.join(BASE_INPUT, "aerial-cactus-identification", "test")

file_path = "weights-aerial-cactus.h5"

callbacks = [
    ModelCheckpoint(
        file_path,
        monitor="val_auc",
        verbose=1,
        save_best_only=True,
        mode="max",
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=3,
        verbose=1,
        mode="min",
        min_lr=1e-5,
    ),
    EarlyStopping(
        monitor="val_auc",
        min_delta=1e-4,
        patience=7,
        verbose=1,
        mode="max",
        restore_best_weights=True,
    ),
]



## === cell 6
images_train = []
labels_train = []

for image_id in tqdm(train_data["id"].values, desc="Loading train images"):
    img_path = os.path.join(TRAIN_IMG_DIR, image_id)
    img = cv.imread(img_path)
    if img is None:
        continue
    images_train.append(img)
    label = train_data.loc[train_data["id"] == image_id, "has_cactus"].values[0]
    labels_train.append(label)

images_train = np.asarray(images_train, dtype="float32") / 255.0
labels_train = np.asarray(labels_train, dtype="float32")
print(f"Loaded {len(images_train)} training images.")




## === cell 7
x_tr = images_train
y_tr = labels_train




## === cell 8
test_images_names = sorted(os.listdir(TEST_IMG_DIR))

images_test = []
for image_id in tqdm(test_images_names, desc="Loading test images"):
    img_path = os.path.join(TEST_IMG_DIR, image_id)
    img = cv.imread(img_path)
    if img is None:
        continue
    images_test.append(img)

images_test = np.asarray(images_test, dtype="float32") / 255.0
print(f"Loaded {len(images_test)} test images.")




## === cell 9
x_train, x_val, y_train, y_val = train_test_split(
    x_tr,
    y_tr,
    test_size=0.15,
    stratify=y_tr,
    random_state=42,
)
print(f"Train/val split: {x_train.shape[0]} / {x_val.shape[0]}")




## === cell 10
model = create_model()
model.summary()




## === cell 11
history = model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=80,
    validation_data=(x_val, y_val),
    verbose=1,
    callbacks=callbacks,
)




## === cell 12
model.load_weights(file_path)




## === cell 13
pred_test = model.predict(images_test, verbose=1).flatten()




## === cell 14
submission = pd.read_csv("../input/sample_submission.csv")
submission = submission.set_index("id").loc[test_images_names].reset_index()
submission["has_cactus"] = pred_test
submission.to_csv("aerial-cactus-submission.csv", index=False)
print("Submission saved to aerial-cactus-submission.csv")




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_56/1447115562.py in <cell line: 0>()
      1 submission = pd.read_csv("../input/sample_submission.csv")
      2 # Ensure the order matches the test image names
----> 3 submission = submission.set_index("id").loc[test_images_names].reset_index()
      4 submission["has_cactus"] = pred_test
      5 submission.to_csv("aerial-cactus-submission.csv", index=False)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in __getitem__(self, key)
   1189             maybe_callable = com.apply_if_callable(key, self.obj)
   1190             maybe_callable = self._check_deprecated_callable_usage(key, maybe_callable)
-> 1191             return self._getitem_axis(maybe_callable, axis=axis)
   1192 
   1193     def _is_scalar_access(self, key: tuple):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_axis(self, key, axis)
   1418                     raise ValueError("Cannot index with multidimensional key")
   1419 
-> 1420                 return self._getitem_iterable(key, axis=axis)
   1421 
   1422             # nested tuple slicing

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_iterable(self, key, axis)
   1358 
   1359         # A collection of keys
-> 1360         keyarr, indexer = self._get_listlike_indexer(key, axis)
   1361         return self.obj._reindex_with_indexers(
   1362             {axis: [keyarr, indexer]}, copy=True, allow_dups=True

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _get_listlike_indexer(self, key, axis)
   1556         axis_name = self.obj._get_axis_name(axis)
   1557 
-> 1558         keyarr, indexer = ax._get_indexer_strict(key, axis_name)
   1559 
   1560         return keyarr, indexer

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['test'] not in index"

## === cell 15
plot_training_curves(history)




## === cell 16
val_pred = model.predict(x_val, verbose=0).flatten()
auc = roc_auc_score(y_val, val_pred)
print(f"Validation AUC: {auc:.4f}")

conf_matrix = confusion_matrix(y_val, (val_pred > 0.5).astype(int))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    xticklabels=["0", "1"],
    yticklabels=["0", "1"],
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
