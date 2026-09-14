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
imageio==2.37.0
imageio-ffmpeg==0.6.0
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

0.9918

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.50708) has done: 'The fix replaces the outdated Keras imports, removes the `%matplotlib inline` magic, corrects the deprecated `np.float` usage, and ensures all variables are defined in the proper order. The script now loads and preprocesses the images, builds a simple CNN with TensorFlow‑Keras, trains it, and creates a valid `submission.csv` containing the required “id,has_cactus” columns.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, callbacks, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.utils import class_weight

print("TensorFlow version:", tf.__version__)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def load_images(glob_path):
    imgs, names = [], []
    for p in glob.glob(glob_path):
        names.append(os.path.basename(p))
        img = cv2.imread(p, cv2.IMREAD_COLOR)  # shape (32,32,3)
        if img is None:
            continue
        imgs.append(img)
    return np.array(imgs), np.array(names)


train_dir = os.path.abspath("../input/train/")
test_dir = os.path.abspath("../input/test/")

train_meta = pd.read_csv("../input/train.csv")
train_imgs, train_names = load_images(os.path.join(train_dir, "*.jpg"))
print("Loaded train images:", train_imgs.shape)

label_map = dict(zip(train_meta.id.astype(str), train_meta.has_cactus))
train_labels = np.array([label_map.get(name, 0) for name in train_names], dtype=float)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4202200791.py in <cell line: 0>()
     14 
     15 train_meta = pd.read_csv("../input/train.csv")
---> 16 train_imgs, train_names = load_images(os.path.join(train_dir, "*.jpg"))
     17 print("Loaded train images:", train_imgs.shape)
     18 

/tmp/ipykernel_55/4202200791.py in load_images(glob_path)
      1 def load_images(glob_path):
      2     imgs, names = [], []
----> 3     for p in glob.glob(glob_path):
      4         names.append(os.path.basename(p))
      5         img = cv2.imread(p, cv2.IMREAD_COLOR)  # shape (32,32,3)

NameError: name 'glob' is not defined

## === cell 2
train_imgs = train_imgs.astype(np.float32) / 255.0
print("Train data shape:", train_imgs.shape, "Labels shape:", train_labels.shape)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1524876188.py in <cell line: 0>()
----> 1 train_imgs = train_imgs.astype(np.float32) / 255.0
      2 print("Train data shape:", train_imgs.shape, "Labels shape:", train_labels.shape)
      3 

NameError: name 'train_imgs' is not defined

## === cell 3
train_x, val_x, train_y, val_y = train_test_split(
    train_imgs, train_labels, test_size=0.2, random_state=7, stratify=train_labels
)
print("Train split:", train_x.shape, "Validation split:", val_x.shape)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1123294785.py in <cell line: 0>()
      1 train_x, val_x, train_y, val_y = train_test_split(
----> 2     train_imgs, train_labels, test_size=0.2, random_state=7, stratify=train_labels
      3 )
      4 print("Train split:", train_x.shape, "Validation split:", val_x.shape)
      5 

NameError: name 'train_imgs' is not defined

## === cell 4
datagen = preprocessing.image.ImageDataGenerator(
    rotation_range=60, zoom_range=0.2, horizontal_flip=True, vertical_flip=True
)
datagen.fit(train_x)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/915318939.py in <cell line: 0>()
      2     rotation_range=60, zoom_range=0.2, horizontal_flip=True, vertical_flip=True
      3 )
----> 4 datagen.fit(train_x)
      5 

NameError: name 'train_x' is not defined

## === cell 5
input_shape = train_x.shape[1:]  # (32,32,3)
model = models.Sequential(
    [
        layers.Conv2D(32, (3, 3), input_shape=input_shape, use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)
model.compile(
    optimizer=optimizers.Adam(),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
)
model.summary()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1279008515.py in <cell line: 0>()
----> 1 input_shape = train_x.shape[1:]  # (32,32,3)
      2 model = models.Sequential(
      3     [
      4         layers.Conv2D(32, (3, 3), input_shape=input_shape, use_bias=False),
      5         layers.BatchNormalization(),

NameError: name 'train_x' is not defined

## === cell 6
weights = class_weight.compute_class_weight(
    class_weight="balanced", classes=np.unique(train_y), y=train_y
)
class_weights = {i: w for i, w in enumerate(weights)}

ckpt = callbacks.ModelCheckpoint(
    "best_model.keras", monitor="val_loss", save_best_only=True, verbose=1
)
es = callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1)

history = model.fit(
    datagen.flow(train_x, train_y, batch_size=32),
    steps_per_epoch=len(train_x) // 32,
    epochs=30,
    validation_data=(val_x, val_y),
    callbacks=[ckpt, es],
    class_weight=class_weights,
    verbose=2,
)

model.load_weights("best_model.keras")



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/86986718.py in <cell line: 0>()
      1 # Compute class weights to handle imbalance
      2 weights = class_weight.compute_class_weight(
----> 3     class_weight="balanced", classes=np.unique(train_y), y=train_y
      4 )
      5 class_weights = {i: w for i, w in enumerate(weights)}

NameError: name 'train_y' is not defined

## === cell 7
test_imgs, test_names = load_images(os.path.join(test_dir, "*.jpg"))
test_imgs = test_imgs.astype(np.float32) / 255.0
preds = model.predict(test_imgs, batch_size=64, verbose=0).ravel()
submission = pd.DataFrame({"id": test_names, "has_cactus": preds})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print("Submission written to", submission_path, "rows:", len(submission))

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2882450115.py in <cell line: 0>()
----> 1 test_imgs, test_names = load_images(os.path.join(test_dir, "*.jpg"))
      2 test_imgs = test_imgs.astype(np.float32) / 255.0
      3 preds = model.predict(test_imgs, batch_size=64, verbose=0).ravel()
      4 submission = pd.DataFrame({"id": test_names, "has_cactus": preds})
      5 submission_path = "submission.csv"

/tmp/ipykernel_55/4202200791.py in load_images(glob_path)
      1 def load_images(glob_path):
      2     imgs, names = [], []
----> 3     for p in glob.glob(glob_path):
      4         names.append(os.path.basename(p))
      5         img = cv2.imread(p, cv2.IMREAD_COLOR)  # shape (32,32,3)

NameError: name 'glob' is not defined
