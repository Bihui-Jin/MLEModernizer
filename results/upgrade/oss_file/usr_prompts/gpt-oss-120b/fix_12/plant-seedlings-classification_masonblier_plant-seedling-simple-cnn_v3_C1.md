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
import random
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
import concurrent.futures
import gc

print("Input directories:", os.listdir("../input"))




## === cell 1
def loadImagesData(glob_path):
    """Load and resize images in parallel, returning a stacked NumPy array and filenames."""
    img_paths = glob.glob(glob_path)
    if not img_paths:
        return np.empty((0, 100, 100, 3), dtype=np.uint8), []

    def read_and_resize(p):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            return None, None
        img_resized = cv2.resize(img, (100, 100), interpolation=cv2.INTER_CUBIC)
        return img_resized, os.path.basename(p)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(read_and_resize, img_paths))

    imgs, names = zip(*[(img, name) for img, name in results if img is not None])
    stacked = np.stack(imgs)  # shape (N,100,100,3)
    return stacked, list(names)


train_base = "../input/plant-seedlings-classification/train"




## === cell 2
all_images = []
all_labels = []

for label in os.listdir(train_base):
    imgs, _ = loadImagesData(os.path.join(train_base, label, "*.png"))
    if imgs.shape[0] == 0:
        continue
    all_images.append(imgs)
    all_labels.extend([label] * imgs.shape[0])

train_images = np.concatenate(all_images, axis=0)  # (N,100,100,3)
train_labels = np.array(all_labels)

np.random.seed(7)
perm = np.random.permutation(train_images.shape[0])
train_images = train_images[perm]
train_labels = train_labels[perm]

all_x = train_images.astype(np.float32, copy=False) / 255.0

del train_images, all_images
gc.collect()




## === cell 3
le = LabelEncoder()
le.fit(np.unique(train_labels))
int_y = le.transform(train_labels)  # integer labels for scikit‑learn




## === cell 4
train_x, val_x, train_y_int, val_y_int = train_test_split(
    all_x, int_y, test_size=0.2, random_state=7, stratify=int_y
)
print("Train shape:", train_x.shape, "Validation shape:", val_x.shape)




## === cell 5
train_x_flat = train_x.reshape(train_x.shape[0], -1)
val_x_flat = val_x.reshape(val_x.shape[0], -1)

scaler = StandardScaler(copy=False)
train_x_scaled = scaler.fit_transform(train_x_flat)
val_x_scaled = scaler.transform(val_x_flat)

pca = PCA(n_components=0.99, random_state=42)
train_x_pca = pca.fit_transform(train_x_scaled)
val_x_pca = pca.transform(val_x_scaled)

rf = RandomForestClassifier(
    n_estimators=1500,
    max_depth=None,
    max_features=None,
    n_jobs=-1,
    random_state=42,
    class_weight="balanced",
)
rf.fit(train_x_pca, train_y_int)

val_pred = rf.predict(val_x_pca)
f1 = f1_score(val_y_int, val_pred, average="micro")
print(f"Micro‑averaged F1 on validation split: {f1:.5f}")

del (
    train_x,
    val_x,
    train_x_flat,
    val_x_flat,
    train_x_scaled,
    val_x_scaled,
    train_x_pca,
    val_x_pca,
)
gc.collect()




## === cell 6
test_base = "../input/plant-seedlings-classification/test"
test_images, test_names = loadImagesData(os.path.join(test_base, "*.png"))
if test_images.shape[0] == 0:
    raise RuntimeError("No test images were loaded; check the input path.")

test_flat = (
    test_images.astype(np.float32, copy=False).reshape(test_images.shape[0], -1) / 255.0
)

test_flat_scaled = scaler.transform(test_flat)
test_flat_pca = pca.transform(test_flat_scaled)

test_pred_int = rf.predict(test_flat_pca)
test_pred_labels = le.inverse_transform(test_pred_int)

submission_df = pd.DataFrame({"file": test_names, "species": test_pred_labels})
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path} – rows:", len(submission_df))
