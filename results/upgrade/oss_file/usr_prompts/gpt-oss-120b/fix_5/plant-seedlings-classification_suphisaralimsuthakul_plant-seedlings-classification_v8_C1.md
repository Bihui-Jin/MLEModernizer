# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

3.11

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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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
tf_keras==2.18.0
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

# 5. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd
import cv2
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import multiprocessing as mp
import warnings
import gc

warnings.filterwarnings("ignore", category=FutureWarning)

np.random.seed(42)


def _init_worker():
    np.random.seed(42)




## === cell 1
TRAIN_DIR = "/kaggle/input/plant-seedlings-classification/train"
img_size = 128


def _load_and_resize(path):
    """Read an image, resize to (img_size, img_size), and return it."""
    img = cv2.resize(cv2.imread(path), (img_size, img_size))
    return img  # returns None if read fails


train_paths = glob.glob(os.path.join(TRAIN_DIR, "*", "*.png"))
train_labels = [os.path.basename(os.path.dirname(p)) for p in train_paths]

with mp.Pool(mp.cpu_count(), initializer=_init_worker) as pool:
    loaded = pool.map(_load_and_resize, train_paths, chunksize=1000)

valid_imgs = []
valid_labels = []
for img, lbl in zip(loaded, train_labels):
    if img is not None:
        valid_imgs.append(img)
        valid_labels.append(lbl)

train_X = np.stack(valid_imgs, axis=0)  # shape (n_samples, 128, 128, 3)
train_Y = np.array(valid_labels)

le = LabelEncoder()
y_int = le.fit_transform(train_Y)  # integer labels 0‑11
_ = LabelBinarizer().fit_transform(y_int)  # retained for compatibility

del valid_imgs, valid_labels, loaded
gc.collect()



## === cell 2
X_train, X_val, y_train, y_val = train_test_split(
    train_X, y_int, test_size=0.2, random_state=7, stratify=y_int
)

X_train_flat = X_train.astype(np.float32, copy=False).reshape(len(X_train), -1)
X_train_flat *= 1.0 / 255.0

X_val_flat = X_val.astype(np.float32, copy=False).reshape(len(X_val), -1)
X_val_flat *= 1.0 / 255.0



## === cell 3
clf = LogisticRegression(
    multi_class="multinomial",
    solver="saga",
    max_iter=200,
    n_jobs=-1,
    C=10.0,
    random_state=42,
)
clf.fit(X_train_flat, y_train)

val_pred = clf.predict(X_val_flat)
val_f1 = f1_score(y_val, val_pred, average="micro")
print(f"Validation micro‑averaged F1: {val_f1:.5f}")



## === cell 4
TEST_DIR = "/kaggle/input/plant-seedlings-classification/test"


def _load_test_image(path):
    img = cv2.resize(cv2.imread(path), (img_size, img_size))
    return img, os.path.basename(path)


test_paths = glob.glob(os.path.join(TEST_DIR, "*.png"))

with mp.Pool(mp.cpu_count(), initializer=_init_worker) as pool:
    test_results = pool.map(_load_test_image, test_paths, chunksize=1000)

test_images = []
test_files = []
for img, fname in test_results:
    if img is not None:
        test_images.append(img)
        test_files.append(fname)

test_X = np.stack(test_images, axis=0)
test_X_flat = test_X.astype(np.float32, copy=False).reshape(len(test_X), -1)
test_X_flat *= 1.0 / 255.0

test_pred_int = clf.predict(test_X_flat)
test_pred_str = le.inverse_transform(test_pred_int)

submission = pd.DataFrame({"file": test_files, "species": test_pred_str})
submission = submission.sort_values("file").reset_index(drop=True)

submission_path = "Plant-Seedlings-Classification.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file saved to {submission_path}")
