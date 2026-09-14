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
Classify plant seedlings into their respective species.

## Metric
Micro-averaged F1-score.

## Submission Format
For each `file` in the test set, you must predict a probability for the `species` variable. The file should contain a header and have the following format:

```
file,species
0021e90e4.png,Maize
003d61042.png,Sugar beet
007b3da8b.png,Common wheat
etc.
```

## Dataset
The list of species is as follows:

```
Black-grass
Charlock
Cleavers
Common Chickweed
Common wheat
Fat Hen
Loose Silky-bent
Maize
Scentless Mayweed
Shepherds Purse
Small-flowered Cranesbill
Sugar beet
```

- **train.csv** - the training set, with plant species organized by folder
- **test.csv** - the test set, you need to predict the species of each image
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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
sklearn-pandas==2.2.0
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        input/
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        working/
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
```

-> data/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> data/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> working/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

# 5. Target score

0.05919

# 6. Current score

0.12162

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.04204) has done: 'I fix the runtime-breaking API issues (Pillow resize filter, Keras 3 imports, and BatchNormalization import path) and correct the label encoding bug that was producing invalid multi-hot vectors. I also align the training objective with the multi-class problem by switching from `binary_crossentropy` to `categorical_crossentropy` while keeping the same CNN architecture and training loop, which is necessary for the model to train at all. Finally, I ensure test files are read and written in the exact order required by `sample_submission.csv`, producing a valid `submission.csv` with columns `file,species`. These changes are minimal, unblock end-to-end execution, and should yield a non-trivial F1 score (well above random), moving toward your target.'
- What this solution (achieved 0.12162) has done: 'I fix the runtime error coming from `tf_keras` imports by switching to the Kaggle-installed `keras` (Keras 3) API while keeping the exact same Sequential CNN architecture and training loop. I also add a small compatibility shim for `BatchNormalization` import and make the run deterministic to stabilize the score (this is score-neutral to slightly positive, not a modeling change). Finally, I keep the submission generation aligned to `sample_submission.csv` order and ensure `submission.csv` is written with the required `file,species` columns.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

from PIL import Image
import os
from random import shuffle
import matplotlib.pyplot as plt

import random

random.seed(1987)
np.random.seed(1987)



## === cell 1
CLASS = [
    "Black-grass",
    "Charlock",
    "Cleavers",
    "Common Chickweed",
    "Common wheat",
    "Fat Hen",
    "Loose Silky-bent",
    "Maize",
    "Scentless Mayweed",
    "Shepherds Purse",
    "Small-flowered Cranesbill",
    "Sugar beet",
]



## === cell 2
SAMPLE_PER_CATEGORY = 200
SEED = 1987

data_dir = "/kaggle/input/plant-seedlings-classification/"

train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
sample_submission = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))

print("train_dir:", train_dir)
print("test_dir :", test_dir)
print(sample_submission.head(2))



## === cell 3
import imageio


def get_size_statistics():
    heights = []
    widths = []
    img_count = 0
    for category in CLASS:
        cat_dir = os.path.join(train_dir, category)
        for img in os.listdir(cat_dir):
            path = os.path.join(cat_dir, img)
            data = np.array(Image.open(path))
            heights.append(data.shape[0])
            widths.append(data.shape[1])
            img_count += 1
    avg_height = sum(heights) / len(heights)
    avg_width = sum(widths) / len(widths)
    print("Image count:", img_count)
    print("Average Height:", avg_height)
    print("Max Height:", max(heights))
    print("Min Height:", min(heights))
    print()
    print("Average Width:", avg_width)
    print("Max Width:", max(widths))
    print("Min Width:", min(widths))


get_size_statistics()




## === cell 4
def label_img_from_dir(dir_path):
    category = os.path.basename(dir_path.rstrip("/"))
    idx = CLASS.index(category)
    y = np.zeros(len(CLASS), dtype=np.float32)
    y[idx] = 1.0
    return y


RESAMPLE = Image.Resampling.LANCZOS

IMG_SIZE = 300
train_data = []
test_data = []


def load_training_data(DIR):
    files = os.listdir(DIR)
    files = sorted(files)
    rnd = np.random.RandomState(SEED)
    rnd.shuffle(files)

    label = label_img_from_dir(DIR)
    split = int(len(files) * 2 / 3)

    for index, fname in enumerate(files):
        path = os.path.join(DIR, fname)
        img = Image.open(path).convert("L").resize((IMG_SIZE, IMG_SIZE), RESAMPLE)
        arr = np.array(img, dtype=np.uint8)
        if index < split:
            train_data.append([arr, label])
        else:
            test_data.append([arr, label])


for category in CLASS:
    load_training_data(os.path.join(train_dir, category))

shuffle(train_data)
shuffle(test_data)

print("Train samples:", len(train_data), "Test/val samples:", len(test_data))



## === cell 5
if len(train_data) > 2:
    plt.imshow(train_data[2][0], cmap="gist_gray")
    plt.title("Example training image")
    plt.axis("off")
    plt.show()



## === cell 6
trainImages = (
    np.array([i[0] for i in train_data], dtype=np.float32).reshape(
        -1, IMG_SIZE, IMG_SIZE, 1
    )
    / 255.0
)
trainLabels = np.array([i[1] for i in train_data], dtype=np.float32)

testImages = (
    np.array([i[0] for i in test_data], dtype=np.float32).reshape(
        -1, IMG_SIZE, IMG_SIZE, 1
    )
    / 255.0
)
testLabels = np.array([i[1] for i in test_data], dtype=np.float32)

print("trainImages:", trainImages.shape, "trainLabels:", trainLabels.shape)
print("testImages :", testImages.shape, "testLabels :", testLabels.shape)



## === cell 7
target_data = []


def load_target_data(test_dir, files_in_order):
    out = []
    for fname in files_in_order:
        path = os.path.join(test_dir, fname)
        img = Image.open(path).convert("L").resize((IMG_SIZE, IMG_SIZE), RESAMPLE)
        out.append(np.array(img, dtype=np.uint8))
    return out


test_files_ordered = sample_submission["file"].tolist()
target_data = load_target_data(test_dir, test_files_ordered)

targetImages = (
    np.array(target_data, dtype=np.float32).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
)
print("targetImages:", targetImages.shape)



## === cell 8
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

try:
    from keras.layers import BatchNormalization
except Exception:
    from keras.layers.normalization import (
        BatchNormalization,
    )  # fallback for older layouts

try:
    keras.utils.set_random_seed(SEED)
except Exception:
    pass



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 9
model = Sequential()
model.add(
    Conv2D(
        32, kernel_size=(3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 1)
    )
)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(96, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(96, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(128, activation="relu"))
model.add(Dense(12, activation="softmax"))

model.summary()



## === cell 10
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])



## === cell 11
history = model.fit(trainImages, trainLabels, batch_size=50, epochs=5, verbose=1)



## === cell 12
loss, acc = model.evaluate(testImages, testLabels, verbose=0)
print("Validation accuracy (%):", acc * 100)



## === cell 13
y_prob = model.predict(targetImages, verbose=0)
y_classes = y_prob.argmax(axis=1)
print("Pred classes shape:", y_classes.shape)



## === cell 14
pred_species = [CLASS[i] for i in y_classes]
print(pred_species[:5])

submission = pd.DataFrame(
    {"file": sample_submission["file"].values, "species": pred_species}
)
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
