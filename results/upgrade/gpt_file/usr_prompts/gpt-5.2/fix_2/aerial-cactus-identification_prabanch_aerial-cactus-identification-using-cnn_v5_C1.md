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

0.9146

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import cv2

import tf_keras as keras
from tf_keras.models import Sequential
from tf_keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Conv2D,
    MaxPooling2D,
    Activation,
    BatchNormalization,
)
from tf_keras.optimizers import SGD
from tf_keras.preprocessing.image import ImageDataGenerator
from tf_keras.preprocessing import image as kimage

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix

np.random.seed(42)
try:
    keras.utils.set_random_seed(42)
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_PATH = "/kaggle/input/aerial-cactus-identification/"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")
TRAIN_DIR = os.path.join(BASE_PATH, "train", "train")
TEST_DIR = os.path.join(BASE_PATH, "test", "test")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_DIR), f"Missing dir: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing dir: {TEST_DIR}"

train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(SAMPLE_SUB)

train.head(), test.head()



## === cell 2
y = train["has_cactus"].astype(int)
print(y.value_counts())




## === cell 3
def load_images(img_dir, df):
    images = []
    for i in tqdm(range(df.shape[0])):
        img_path = os.path.join(img_dir, df["id"].iloc[i])
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[0] != 32 or img.shape[1] != 32:
            img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
        img = kimage.img_to_array(img).astype("float32") / 255.0
        images.append(img)
    return np.stack(images, axis=0)


X = load_images(TRAIN_DIR, train)
test_images = load_images(TEST_DIR, test)

X.shape, test_images.shape



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/800495237.py in <cell line: 0>()
     16 
     17 
---> 18 X = load_images(TRAIN_DIR, train)
     19 test_images = load_images(TEST_DIR, test)
     20 

/tmp/ipykernel_11/800495237.py in load_images(img_dir, df)
      6         img = cv2.imread(img_path)
      7         if img is None:
----> 8             raise FileNotFoundError(f"Could not read image: {img_path}")
      9         # Ensure 32x32 and RGB order expected by common Keras pipelines
     10         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

FileNotFoundError: Could not read image: /kaggle/input/aerial-cactus-identification/train/train/2de8f189f1dce439766637e75df0ee27.jpg

## === cell 4
X_train, X_val, y_train_raw, y_val_raw = train_test_split(
    X, y, random_state=42, test_size=0.2, stratify=y
)

y_train = pd.get_dummies(y_train_raw).values
y_val = pd.get_dummies(y_val_raw).values

y_train.shape, y_val.shape



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1720598647.py in <cell line: 0>()
      1 # Split
      2 X_train, X_val, y_train_raw, y_val_raw = train_test_split(
----> 3     X, y, random_state=42, test_size=0.2, stratify=y
      4 )
      5 

NameError: name 'X' is not defined

## === cell 5
classes = np.unique(y_train_raw)
cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_raw)
class_weights = {int(c): float(w) for c, w in zip(classes, cw)}
class_weights



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/170614098.py in <cell line: 0>()
      1 # Bugfix: sklearn>=1.2 requires keyword-only args for compute_class_weight
----> 2 classes = np.unique(y_train_raw)
      3 cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_raw)
      4 class_weights = {int(c): float(w) for c, w in zip(classes, cw)}
      5 class_weights

NameError: name 'y_train_raw' is not defined

## === cell 6
model = Sequential()
model.add(
    Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=(32, 32, 3))
)
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(128, activation="relu"))
model.add(Dropout(0.25))

model.add(Dense(64, activation="relu"))
model.add(Dropout(0.25))

model.add(Dense(2, activation="softmax"))

model.summary()



## === cell 7
opt = SGD(learning_rate=1e-3, momentum=0.9)

model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])



## === cell 8
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)
datagen.fit(X_train)
print("augmentation ready")



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3826184508.py in <cell line: 0>()
      9     fill_mode="nearest",
     10 )
---> 11 datagen.fit(X_train)
     12 print("augmentation ready")
     13 

NameError: name 'X_train' is not defined

## === cell 9
batch_size = 100

results = model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True),
    epochs=1,
    steps_per_epoch=max(1, X_train.shape[0] // batch_size),
    validation_data=(X_val, y_val),
    class_weight=class_weights,
    verbose=2,
)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2578628531.py in <cell line: 0>()
      3 
      4 results = model.fit(
----> 5     datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True),
      6     epochs=1,
      7     steps_per_epoch=max(1, X_train.shape[0] // batch_size),

NameError: name 'X_train' is not defined

## === cell 10
fig, ax = plt.subplots(2, 1, figsize=(8, 8))

ax[0].plot(results.history.get("loss", []), color="b", label="Training loss")
ax[0].plot(results.history.get("val_loss", []), color="r", label="Validation loss")
ax[0].legend(loc="best", shadow=True)

ax[1].plot(results.history.get("accuracy", []), color="b", label="Training accuracy")
ax[1].plot(
    results.history.get("val_accuracy", []), color="r", label="Validation accuracy"
)
ax[1].legend(loc="best", shadow=True)

plt.tight_layout()
plt.show()



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/530092186.py in <cell line: 0>()
      2 fig, ax = plt.subplots(2, 1, figsize=(8, 8))
      3 
----> 4 ax[0].plot(results.history.get("loss", []), color="b", label="Training loss")
      5 ax[0].plot(results.history.get("val_loss", []), color="r", label="Validation loss")
      6 ax[0].legend(loc="best", shadow=True)

NameError: name 'results' is not defined

## === cell 11
final_loss, final_acc = model.evaluate(X_val, y_val, verbose=0)
print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2185963129.py in <cell line: 0>()
----> 1 final_loss, final_acc = model.evaluate(X_val, y_val, verbose=0)
      2 print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))
      3 

NameError: name 'X_val' is not defined

## === cell 12
val_proba = model.predict(X_val, verbose=0)
val_pred = np.argmax(val_proba, axis=1)

cm = confusion_matrix(y_val_raw, val_pred)
cm



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/494547896.py in <cell line: 0>()
      1 # Bugfix: predict_classes removed; compute class preds from probabilities
----> 2 val_proba = model.predict(X_val, verbose=0)
      3 val_pred = np.argmax(val_proba, axis=1)
      4 
      5 cm = confusion_matrix(y_val_raw, val_pred)

NameError: name 'X_val' is not defined

## === cell 13
test_proba = model.predict(test_images, verbose=0)[:, 1]
submission = test.copy()
submission["has_cactus"] = test_proba.astype("float64")

submission = submission[["id", "has_cactus"]]
submission.to_csv("submission.csv", index=False)

submission.head(), submission.shape

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/442730187.py in <cell line: 0>()
      1 # Submission: for ROC-AUC we must submit probabilities for has_cactus=1
----> 2 test_proba = model.predict(test_images, verbose=0)[:, 1]
      3 submission = test.copy()
      4 submission["has_cactus"] = test_proba.astype("float64")
      5 

NameError: name 'test_images' is not defined
