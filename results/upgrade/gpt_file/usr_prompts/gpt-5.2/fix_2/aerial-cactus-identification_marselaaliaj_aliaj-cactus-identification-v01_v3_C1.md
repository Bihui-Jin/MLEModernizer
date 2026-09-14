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

3.9

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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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

0.9576

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense

from tf_keras.preprocessing.image import ImageDataGenerator

np.random.seed(1)
keras.utils.set_random_seed(1)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_DIR = "/kaggle/input/aerial-cactus-identification"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

train = pd.read_csv(TRAIN_CSV, dtype={"id": "string", "has_cactus": "string"})
test = pd.read_csv(SAMPLE_SUB, dtype={"id": "string", "has_cactus": "string"})

train.head()



## === cell 2
test.head()



## === cell 3
train["has_cactus"].value_counts()



## === cell 4
cmap = plt.get_cmap("Blues")
colors = [cmap(i) for i in np.linspace(0, 0.7, train["has_cactus"].nunique())]

plt.title("Сlass distribution")
train["has_cactus"].value_counts().plot(
    kind="pie", figsize=(6, 6), autopct="%1.2f%%", shadow=True, colors=colors
)
plt.show()



## === cell 5
print("Train dir:", TRAIN_DIR)
print("Test dir: ", TEST_DIR)
print("Training Images:", len(os.listdir(TRAIN_DIR)))
print("Testing Images: ", len(os.listdir(TEST_DIR)))



## === cell 6
submission = pd.read_csv(SAMPLE_SUB)
submission.head()



## === cell 7
train_datagen = ImageDataGenerator(rescale=1 / 255.0, validation_split=0.20)
test_datagen = ImageDataGenerator(rescale=1 / 255.0)



## === cell 8
bs = 64

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train,
    directory=TRAIN_DIR,
    x_col="id",
    y_col="has_cactus",
    subset="training",
    batch_size=bs,
    shuffle=True,
    class_mode="categorical",
    target_size=(32, 32),
)

valid_generator = train_datagen.flow_from_dataframe(
    dataframe=train,
    directory=TRAIN_DIR,
    x_col="id",
    y_col="has_cactus",
    subset="validation",
    batch_size=bs,
    shuffle=True,
    class_mode="categorical",
    target_size=(32, 32),
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test,
    directory=TEST_DIR,
    x_col="id",
    y_col=None,
    batch_size=bs,
    seed=1,
    shuffle=False,
    class_mode=None,
    target_size=(32, 32),
)



## === cell 9
tr_steps = int(math.ceil(train_generator.samples / bs))
va_steps = int(math.ceil(valid_generator.samples / bs))
te_steps = int(math.ceil(test_generator.samples / bs))

tr_steps, va_steps, te_steps




## === cell 10
def training_images(seed):
    np.random.seed(seed)
    train_generator.reset()
    imgs, labels = next(train_generator)

    plt.figure(figsize=(14, 14))
    for i in range(min(36, imgs.shape[0])):
        plt.subplot(6, 6, i + 1)
        plt.imshow(imgs[i])
        if labels[i, 0] == 1:
            plt.text(0, -2, "Negative", color="r")
        else:
            plt.text(0, -2, "Positive", color="b")
        plt.axis("off")
    plt.show()


training_images(5)



## === cell 11
np.random.seed(1)

cnn = Sequential()

cnn.add(Conv2D(16, (3, 3), activation="relu", padding="same", input_shape=(32, 32, 3)))
cnn.add(Conv2D(16, (3, 3), activation="relu", padding="same"))
cnn.add(MaxPooling2D(2, 2))
cnn.add(BatchNormalization())

cnn.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
cnn.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
cnn.add(MaxPooling2D(2, 2))
cnn.add(BatchNormalization())

cnn.add(Flatten())
cnn.add(Dense(64, activation="relu"))
cnn.add(BatchNormalization())

cnn.add(Dense(2, activation="softmax"))

cnn.summary()



## === cell 12
opt = keras.optimizers.Adam(0.0001)
cnn.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

h1 = cnn.fit(
    train_generator,
    steps_per_epoch=tr_steps,
    epochs=20,
    validation_data=valid_generator,
    validation_steps=va_steps,
    verbose=1,
)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_10/1202362429.py in <cell line: 0>()
      3 cnn.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
      4 
----> 5 h1 = cnn.fit(
      6     train_generator,
      7     steps_per_epoch=tr_steps,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/trainers/data_adapters/__init__.py in get_data_adapter(x, y, sample_weight, batch_size, steps_per_epoch, shuffle, class_weight)
    123         # )
    124     else:
--> 125         raise ValueError(f"Unrecognized data type: x={x} (of type {type(x)})")
    126 
    127 

ValueError: Unrecognized data type: x=<tf_keras.src.preprocessing.image.DataFrameIterator object at 0x7f09339ce4d0> (of type <class 'tf_keras.src.preprocessing.image.DataFrameIterator'>)

## === cell 13
start = 1
ep_rng = np.arange(start, len(h1.history["accuracy"]))

plt.figure(figsize=[12, 6])
plt.subplot(1, 2, 1)
plt.plot(ep_rng, h1.history["accuracy"][start:], label="Training Accuracy")
plt.plot(ep_rng, h1.history["val_accuracy"][start:], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(ep_rng, h1.history["loss"][start:], label="Training Loss")
plt.plot(ep_rng, h1.history["val_loss"][start:], label="Validation Loss")
plt.xlabel("Epoch")
plt.legend()

plt.show()



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/3345722290.py in <cell line: 0>()
      1 start = 1
----> 2 ep_rng = np.arange(start, len(h1.history["accuracy"]))
      3 
      4 plt.figure(figsize=[12, 6])
      5 plt.subplot(1, 2, 1)

NameError: name 'h1' is not defined

## === cell 14
test_generator.reset()
test_pred = cnn.predict(test_generator, steps=te_steps, verbose=1)

class_indices = train_generator.class_indices  # e.g. {'0': 0, '1': 1}
pos_index = class_indices.get("1", 1)

has_cactus_proba = test_pred[:, pos_index].astype(float)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_10/1786038488.py in <cell line: 0>()
      1 # Keras 3: use predict() instead of deprecated predict_generator()
      2 test_generator.reset()
----> 3 test_pred = cnn.predict(test_generator, steps=te_steps, verbose=1)
      4 
      5 # Convert softmax -> probability for "has_cactus"=1

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/trainers/data_adapters/__init__.py in get_data_adapter(x, y, sample_weight, batch_size, steps_per_epoch, shuffle, class_weight)
    123         # )
    124     else:
--> 125         raise ValueError(f"Unrecognized data type: x={x} (of type {type(x)})")
    126 
    127 

ValueError: Unrecognized data type: x=<tf_keras.src.preprocessing.image.DataFrameIterator object at 0x7f09be16ebd0> (of type <class 'tf_keras.src.preprocessing.image.DataFrameIterator'>)

## === cell 15
sub = pd.read_csv(SAMPLE_SUB)
sub["has_cactus"] = has_cactus_proba[: len(sub)]

sub["id"] = sub["id"].astype(str)
sub.to_csv("submission.csv", index=False)

sub.head()



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/1725791669.py in <cell line: 0>()
      1 # Use the sample submission IDs to guarantee correct order/format.
      2 sub = pd.read_csv(SAMPLE_SUB)
----> 3 sub["has_cactus"] = has_cactus_proba[: len(sub)]
      4 
      5 # Ensure id column is exactly the filename (not a path)

NameError: name 'has_cactus_proba' is not defined

## === cell 16
import shutil

for p in ("/kaggle/working/train", "/kaggle/working/test"):
    if os.path.isdir(p):
        shutil.rmtree(p)
print("Wrote submission.csv with", len(sub), "rows")
