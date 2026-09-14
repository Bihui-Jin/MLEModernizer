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

0.0961

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.04354) has done: 'I fix the path handling so the script points to the correct dataset directories, make the image‑loading function skip sub‑directories, replace the incompatible `keras` import with `tensorflow.keras`, and adjust the test‑set loading to create `targetImages` correctly. These changes resolve the IsADirectoryError and the protobuf import error, enable the model to train, and ensure a valid `submission.csv` is written.'
- What this solution (achieved 0.08108) has done: 'I set the protobuf implementation to the pure‑Python version before any TensorFlow import to avoid the `MessageFactory` error, and I increased the training epochs from 5 to 10 to raise the validation performance modestly toward the target score. No other logic was altered.'
- What this solution (achieved 0.0961) has done: 'I replace the TensorFlow‑based Keras imports with the lightweight `tf_keras` package, which avoids the protobuf MessageFactory error while keeping the exact model architecture and training unchanged. This minimal change restores execution so the script can train, evaluate, and write a valid `submission.csv` without altering scoring logic.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from random import shuffle

base_dir = os.path.abspath("../input")
candidate = os.path.join(base_dir, "plant-seedlings-classification")
if os.path.isdir(os.path.join(candidate, "train")):
    base_dir = candidate

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
IMG_SIZE = 300

train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")
sample_submission = pd.read_csv(os.path.join(base_dir, "sample_submission.csv"))




## === cell 1
def one_hot(label_name):
    """Return a 12‑dim one‑hot vector for the given class name."""
    vec = np.zeros(len(CLASS), dtype=np.float32)
    vec[CLASS.index(label_name)] = 1.0
    return vec


def load_images_from_folder(folder_path, label_vec=None):
    """Load images from a folder, ignore sub‑directories, convert to grayscale,
    resize, and return a list of (image_array, label) tuples (or just arrays)."""
    data = []
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        if not os.path.isfile(fpath):  # skip directories
            continue
        img = Image.open(fpath).convert("L")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        if label_vec is not None:
            data.append([arr, label_vec])
        else:
            data.append(arr)
    return data




## === cell 2
train_data = []
val_data = []

for cat in CLASS:
    folder = os.path.join(train_dir, cat)
    all_imgs = load_images_from_folder(folder, one_hot(cat))
    split_idx = int(len(all_imgs) * 2 / 3)
    train_data.extend(all_imgs[:split_idx])
    val_data.extend(all_imgs[split_idx:])

shuffle(train_data)
shuffle(val_data)

trainImages = np.array([x[0] for x in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
trainLabels = np.array([x[1] for x in train_data])

valImages = np.array([x[0] for x in val_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
valLabels = np.array([x[1] for x in val_data])




## === cell 3
test_filenames = [
    f for f in sorted(os.listdir(test_dir)) if os.path.isfile(os.path.join(test_dir, f))
]
target_data = load_images_from_folder(
    test_dir
)  # no labels, directories already ignored
targetImages = np.array(target_data).reshape(-1, IMG_SIZE, IMG_SIZE, 1)




## === cell 4
from tf_keras.models import Sequential
from tf_keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tf_keras.layers import Dropout, Flatten, Dense




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 5
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
model.add(Dense(len(CLASS), activation="softmax"))




## === cell 6
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(
    trainImages,
    trainLabels,
    validation_data=(valImages, valLabels),
    batch_size=50,
    epochs=10,
    verbose=1,
)




## === cell 7
val_loss, val_acc = model.evaluate(valImages, valLabels, verbose=0)
print(f"Validation accuracy: {val_acc * 100:.2f}%")




## === cell 8
y_prob = model.predict(targetImages)
y_pred_idx = np.argmax(y_prob, axis=1)
pred_species = [CLASS[i] for i in y_pred_idx]

submission = pd.DataFrame({"file": test_filenames, "species": pred_species})

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")
