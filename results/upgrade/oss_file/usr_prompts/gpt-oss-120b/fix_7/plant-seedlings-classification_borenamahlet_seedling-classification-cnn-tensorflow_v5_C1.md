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

3.6

# 3. Installed packages

geopandas==0.14.4
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
tqdm==4.67.1

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

0.35075

# 6. Current score

0.68318

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.13514) has done: 'I correct the data paths to point to the actual Kaggle input directory, remove the TensorFlow import (which caused a protobuf error), and replace the model‑based pipeline with a simple baseline that predicts the most frequent training class for every test image. This fixes the FileNotFound and import errors and guarantees a non‑empty submission CSV, moving the solution from “no score” toward the target.'
- What this solution (achieved 0.66216) has done: 'I replace the trivial “most‑common class” baseline with a lightweight TensorFlow CNN that reads the resized images, trains for a few epochs on the full training set, and then predicts the most‑likely species for each test image. This modest model upgrade is expected to raise the micro‑averaged F1 from 0.135 toward the target 0.351 while keeping the overall pipeline structure intact.'
- What this solution (achieved 0.70871) has done: 'The fix adds an environment variable before importing TensorFlow to avoid the protobuf `MessageFactory` error, allowing the model to be built, trained, and predictions to be generated. No other logic changes are made, preserving the original workflow and scoring behavior.'
- What this solution (achieved 0.68318) has done: 'I keep the original workflow but add a small deterministic ordering for the test filenames (sorting the list) and clear the training data from memory after fitting to avoid unnecessary RAM use. These fixes prevent any hidden nondeterminism and ensure the script runs end‑to‑end, producing a valid `submission.csv` while preserving the existing model and score (which is already above the target).'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf

BASE_DIR = "/kaggle/input/plant-seedlings-classification"
train_dir = os.path.join(BASE_DIR, "train")
test_dir = os.path.join(BASE_DIR, "test")
IMG_SIZE = 50
LR = 1e-3
MODEL_NAME = f"plantclassification-{LR}-2conv-basic.h5"



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
CATEGORIES = [
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
NUM_CATEGORIES = len(CATEGORIES)
print("Number of categories:", NUM_CATEGORIES)




## === cell 2
def label_img(word_label):
    idx = CATEGORIES.index(word_label)
    label = [0] * NUM_CATEGORIES
    label[idx] = 1
    return label




## === cell 3
def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = img / 255.0  # normalize
    return img


def load_training_data():
    X = []
    y = []
    for cat in CATEGORIES:
        cat_path = os.path.join(train_dir, cat)
        if not os.path.isdir(cat_path):
            continue
        for fname in os.listdir(cat_path):
            fpath = os.path.join(cat_path, fname)
            if os.path.isfile(fpath):
                X.append(load_image(fpath))
                y.append(CATEGORIES.index(cat))
    X = tf.stack(X)
    y = tf.keras.utils.to_categorical(y, NUM_CATEGORIES)
    return X, y




## === cell 4
def create_test_filenames():
    test_filenames = []
    for img_name in tqdm(os.listdir(test_dir), desc="loading test"):
        path = os.path.join(test_dir, img_name)
        if os.path.isfile(path):
            test_filenames.append(img_name)
    test_filenames.sort()
    return test_filenames




## === cell 5
print("Loading training data…")
X_train, y_train = load_training_data()
print("Training samples:", X_train.shape[0])




## === cell 6
def build_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


print("Building and training model…")
model = build_model()
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)
model.save(MODEL_NAME)

del X_train, y_train
tf.keras.backend.clear_session()




## === cell 7
def load_test_images(filenames):
    imgs = []
    for fname in filenames:
        fpath = os.path.join(test_dir, fname)
        imgs.append(load_image(fpath))
    return tf.stack(imgs)


test_filenames = create_test_filenames()
print("Loading test images…")
X_test = load_test_images(test_filenames)

print("Predicting on test set…")
pred_probs = model.predict(X_test, batch_size=32, verbose=0)
pred_indices = np.argmax(pred_probs, axis=1)
pred_species = [CATEGORIES[i] for i in pred_indices]



## === cell 8
submission_path = "submission.csv"
with open(submission_path, "w") as f:
    f.write("file,species\n")
    for img_name, species in zip(test_filenames, pred_species):
        f.write(f"{img_name},{species}\n")
print(f"Submission written to {submission_path}")
