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

0.592

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 1
import os
import sys
import subprocess

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as _pb_ver
except Exception:
    _pb_ver = "unknown"


def _major(ver):
    try:
        return int(str(ver).split(".")[0])
    except Exception:
        return None


if _major(_pb_ver) is None or _major(_pb_ver) >= 5:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<5"])



## === cell 2
import pandas as pd

BASE = "../input/aerial-cactus-identification"
TRAIN_CSV = os.path.join(BASE, "train.csv")
SAMPLE_SUB = os.path.join(BASE, "sample_submission.csv")
TRAIN_DIR = os.path.join(BASE, "train", "train")
TEST_DIR = os.path.join(BASE, "test", "test")

train_ds = pd.read_csv(TRAIN_CSV, dtype={"id": str, "has_cactus": str})
train_ds.head()



## === cell 3
train_dir = TRAIN_DIR
print("train_dir:", train_dir)
print("total training images:", len(os.listdir(train_dir)))

train_files = os.listdir(train_dir)
print(train_files[:10])



## === cell 4
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

pic_index = 2
next_img = [
    os.path.join(train_dir, fname) for fname in train_files[pic_index - 2 : pic_index]
]

for img_path in next_img:
    print(img_path)
    img = mpimg.imread(img_path)
    print(img.shape)
    plt.imshow(img)
    plt.axis("off")
    plt.show()



## === cell 5
import numpy as np
import tensorflow as tf

from keras_preprocessing.image import ImageDataGenerator

tf.random.set_seed(42)
np.random.seed(42)

TRAINING_DIR = TRAIN_DIR

training_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.25,
)

train_generator = training_datagen.flow_from_dataframe(
    dataframe=train_ds,
    directory=TRAINING_DIR,
    shuffle=True,
    x_col="id",
    y_col="has_cactus",
    target_size=(32, 32),
    class_mode="categorical",
    subset="training",
    batch_size=32,
)

validation_generator = training_datagen.flow_from_dataframe(
    dataframe=train_ds,
    directory=TRAINING_DIR,
    shuffle=True,
    x_col="id",
    y_col="has_cactus",
    target_size=(32, 32),
    class_mode="categorical",
    subset="validation",
    batch_size=32,
)

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(2, activation="softmax"),
    ]
)

model.summary()
model.compile(
    loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"]
)

history = model.fit(
    train_generator,
    epochs=2,
    validation_data=validation_generator,
    verbose=1,
)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/3925096007.py in <cell line: 0>()
      3 
      4 # Ensure Keras preprocessing is available (keep core data pipeline using ImageDataGenerator)
----> 5 from keras_preprocessing.image import ImageDataGenerator
      6 
      7 # Reproducibility (score-neutral; avoids run-to-run variance)

ModuleNotFoundError: No module named 'keras_preprocessing'

## === cell 6
import matplotlib.pyplot as plt

acc = history.history.get("accuracy", [])
val_acc = history.history.get("val_accuracy", [])
loss = history.history.get("loss", [])
val_loss = history.history.get("val_loss", [])

epochs = range(len(acc))

plt.plot(epochs, acc, "r", label="Training accuracy")
plt.plot(epochs, val_acc, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend(loc=0)
plt.figure()

plt.plot(epochs, loss, "r", label="Loss")
plt.plot(epochs, val_loss, "b", label="Validation Loss")
plt.title("Training and validation Loss")
plt.legend(loc=0)
plt.figure()
plt.show()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1897692598.py in <cell line: 0>()
      2 
      3 # Bugfix: TF2 uses 'accuracy'/'val_accuracy' keys (not 'acc'/'val_acc')
----> 4 acc = history.history.get("accuracy", [])
      5 val_acc = history.history.get("val_accuracy", [])
      6 loss = history.history.get("loss", [])

NameError: name 'history' is not defined

## === cell 7
testdf = pd.read_csv(SAMPLE_SUB, dtype={"id": str, "has_cactus": str})

test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=testdf,
    directory=TEST_DIR,
    target_size=(32, 32),
    x_col="id",
    y_col=None,
    class_mode=None,
    shuffle=False,
    batch_size=32,
)

test_generator.reset()

pred = model.predict(test_generator, verbose=1)

class_indices = train_generator.class_indices  # e.g. {'0':0,'1':1} or reversed
pos_index = class_indices.get("1", 1)
has_cactus_prob = pred[:, pos_index].astype(np.float64)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2705896389.py in <cell line: 0>()
      1 testdf = pd.read_csv(SAMPLE_SUB, dtype={"id": str, "has_cactus": str})
      2 
----> 3 test_datagen = ImageDataGenerator(rescale=1.0 / 255)
      4 test_generator = test_datagen.flow_from_dataframe(
      5     dataframe=testdf,

NameError: name 'ImageDataGenerator' is not defined

## === cell 8
results = pd.DataFrame({"id": testdf["id"].values, "has_cactus": has_cactus_prob})

results["has_cactus"] = results["has_cactus"].clip(0.0, 1.0)

results.to_csv("submission.csv", index=False)
print(results.head())
print("Wrote submission.csv with shape:", results.shape)

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4238801136.py in <cell line: 0>()
      1 # Submission must use the id values from sample_submission.csv in the same order.
      2 # Use testdf['id'] rather than generator filenames to avoid path-prefix issues.
----> 3 results = pd.DataFrame({"id": testdf["id"].values, "has_cactus": has_cactus_prob})
      4 
      5 # Safety: clip to [0,1]

NameError: name 'has_cactus_prob' is not defined
