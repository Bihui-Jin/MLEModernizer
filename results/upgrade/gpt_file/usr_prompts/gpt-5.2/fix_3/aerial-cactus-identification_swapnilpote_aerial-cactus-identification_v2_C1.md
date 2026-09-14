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
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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
pillow==11.3.0
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

0.7998

# 6. Current score

0.92504

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.92504) has done: 'I fix the import crash by avoiding the `tf_keras` package (it triggers the protobuf `MessageFactory.GetPrototype` error in this environment) and using standard `tensorflow.keras` instead, keeping the exact same model/training code. I also fix the test file listing so it only includes actual `.jpg` files (your current `os.listdir(TEST_DIR)` includes a nested `test/` directory entry, which causes the `Could not read image: .../test` failure). Finally, I ensure the submission is created even if there are any unexpected missing predictions by merging onto `sample_submission.csv` and filling defaults, writing `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
from tensorflow import keras

np.random.seed(42)
keras.utils.set_random_seed(42)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_CANDIDATES = [
    "/kaggle/input/aerial-cactus-identification",
    "/kaggle/data/aerial-cactus-identification",
    "../input/aerial-cactus-identification",
    "../input",
]
BASE_DIR = None
for c in BASE_CANDIDATES:
    if os.path.exists(c):
        if os.path.exists(os.path.join(c, "train.csv")) and os.path.isdir(
            os.path.join(c, "train")
        ):
            BASE_DIR = c
            break
        if os.path.isdir(c):
            inner = os.path.join(c, "aerial-cactus-identification")
            if os.path.exists(os.path.join(inner, "train.csv")) and os.path.isdir(
                os.path.join(inner, "train")
            ):
                BASE_DIR = inner
                break

if BASE_DIR is None:
    raise FileNotFoundError(
        "Could not locate aerial-cactus-identification dataset directory."
    )

TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

print("BASE_DIR:", BASE_DIR)
print("Train CSV exists:", os.path.exists(TRAIN_CSV))
print(
    "Train dir exists:",
    os.path.isdir(TRAIN_DIR),
    "n_files:",
    len(os.listdir(TRAIN_DIR)),
)
print(
    "Test dir exists:", os.path.isdir(TEST_DIR), "n_files:", len(os.listdir(TEST_DIR))
)



## === cell 2
dataset = pd.read_csv(TRAIN_CSV)
dataset.head()



## === cell 3
grouped_dataset = dataset.groupby("has_cactus")
grouped_dataset.count()



## === cell 4
dataset.count()




## === cell 5
def datagen(dataset=dataset, path=TRAIN_DIR):
    n = len(dataset)
    x = np.empty((n, 32, 32, 3), dtype=np.uint8)
    y = np.empty((n,), dtype=np.float32)

    for i, (img_id, label) in enumerate(dataset[["id", "has_cactus"]].values):
        img_path = os.path.join(path, img_id)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img = cv2.resize(img, (32, 32))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x[i] = img
        y[i] = float(label)

    permutation = np.random.permutation(n)
    x = x[permutation]
    y = y[permutation]

    return x, y




## === cell 6
X, Y = datagen()
X.shape, Y.shape, Y.mean()



## === cell 7
fig, axs = plt.subplots(1, 5, figsize=(25, 5))
for ax, img, label in zip(axs, X[10:15], Y[10:15]):
    ax.set_title("cactus" if label == 1.0 else "No cactus")
    ax.imshow(img)
    ax.axis("off")
plt.show()



## === cell 8
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(10, 5, padding="valid", input_shape=(32, 32, 3)))
model.add(keras.layers.MaxPooling2D(2, 2))
model.add(keras.layers.ReLU())
model.add(keras.layers.Conv2D(20, 5, padding="valid"))
model.add(keras.layers.SpatialDropout2D(0.5))
model.add(keras.layers.MaxPooling2D(2, 2))
model.add(keras.layers.ReLU())

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(320, activation="relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=0.00001),
    metrics=["accuracy"],
)



## === cell 9
history = model.fit(X, Y, batch_size=256, epochs=50, verbose=1, validation_split=0.2)



## === cell 10
test_imgs = sorted(
    f
    for f in os.listdir(TEST_DIR)
    if os.path.isfile(os.path.join(TEST_DIR, f)) and f.lower().endswith(".jpg")
)
print(len(test_imgs), test_imgs[:5])




## === cell 11
def test_pred(test_imgs=test_imgs, path=TEST_DIR):
    results = []
    for rec in test_imgs:
        img_path = os.path.join(path, rec)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img = cv2.resize(img, (32, 32))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.reshape(img, (1, 32, 32, 3))
        prob = float(model.predict(img, batch_size=1, verbose=0).ravel()[0])
        results.append([rec, float(np.clip(prob, 0.005, 0.995))])
    return results




## === cell 12
predictions = test_pred()
predictions = pd.DataFrame(predictions, columns=["id", "has_cactus"])
predictions.head()



## === cell 13
sample = pd.read_csv(SAMPLE_SUB)
sub = sample[["id"]].merge(predictions, on="id", how="left")

sub["has_cactus"] = sub["has_cactus"].astype(float).fillna(0.5)

sub_path = "submission.csv"
sub.to_csv(sub_path, index=False)
print("Wrote:", sub_path, "shape:", sub.shape)
print(sub.head())
