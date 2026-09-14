# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Detect apple diseases from images.

## Metric
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.12

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
pillow==11.3.0
plotly==5.24.1
plotly-express==0.4.1
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Target score

0.85263

# 6. Current score

0.76198

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.754) has done: 'I fix the TensorFlow import crash caused by an incompatible protobuf/TF stack by forcing the pure-Python protobuf implementation before importing TensorFlow. I also ensure the code runs in Kaggle’s `/kaggle/input/...` paths (your current `../input/...` fail here) and keep everything else (data prep, model, training loops) the same. To move the score toward the target, I stop rounding probabilities to 2 decimals in the submission (rounding hurts ROC AUC calibration) while preserving the model and training exactly. Finally, I make sure the submission has the exact required column names and a `.csv` suffix.'
- What this solution (achieved 0.76198) has done: 'I fix the TensorFlow/protobuf crash by forcing a compatible protobuf version at runtime (without changing your model/training logic) and add a safe fallback to load TensorFlow after that patch. I also correct the dataset path to the actual Kaggle directory you have (`/kaggle/data/...`) so CSVs and images are found reliably. To move the score toward your target (0.85263) with minimal semantic change, I fix the label mapping: `np.argmax(x[1:])` assumes column order and can silently mismatch classes; instead we compute labels from the explicit target column list in the correct order matching the submission columns. Finally, I ensure the submission columns and order exactly match `sample_submission.csv` and write `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault(
    "CUDA_VISIBLE_DEVICES", ""
)  # keep CPU-only stable in Kaggle-like env

import sys
import subprocess


def _ensure_compatible_protobuf():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver

        major = int(pb_ver.split(".")[0])
    except Exception:
        major = None

    if major is None or major >= 5:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
        )
        for m in list(sys.modules.keys()):
            if m.startswith("google.protobuf"):
                del sys.modules[m]


_ensure_compatible_protobuf()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Sequential

import cv2
import plotly.express as px
from PIL import Image

np.random.seed(123)
tf.random.set_seed(123)

print("Python:", sys.version)
print("TensorFlow:", tf.__version__)



## === cell 1
DATA_DIR = "/kaggle/data/plant-pathology-2020-fgvc7"

train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
test_df = pd.read_csv(f"{DATA_DIR}/test.csv")

print("train_df:", train_df.shape, "test_df:", test_df.shape)
train_df.head()



## === cell 2
TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]
train_df["label"] = train_df[TARGET_COLS].values.argmax(axis=1)
train_df.head()



## === cell 3
img = Image.open(f"{DATA_DIR}/images/Train_0.jpg")
nparr = np.asarray(img)
nparr.shape



## === cell 4
label_encode = {0: "healthy", 1: "multiple_diseases", 2: "rust", 3: "scab"}
train_df.shape



## === cell 5
train_df.iloc[0]["image_id"]



## === cell 6
from sklearn.model_selection import train_test_split

train_df, validate_df = train_test_split(
    train_df, test_size=0.2, random_state=123, shuffle=True
)

train_df.shape, validate_df.shape, test_df.shape



## === cell 7
new_img_size = (224, 224)

img = Image.open(f'{DATA_DIR}/images/{train_df.iloc[0]["image_id"]}.jpg')
img = img.resize(new_img_size)
images_train = np.array([np.asarray(img)])

for i in range(1, train_df.shape[0]):
    img = Image.open(f'{DATA_DIR}/images/{train_df.iloc[i]["image_id"]}.jpg')
    img = img.resize(new_img_size)
    images_train = np.concatenate((images_train, np.array([np.asarray(img)])), axis=0)



## === cell 8
img = Image.open(f'{DATA_DIR}/images/{validate_df.iloc[0]["image_id"]}.jpg')
img = img.resize(new_img_size)
images_validate = np.array([np.asarray(img)])

for i in range(1, validate_df.shape[0]):
    img = Image.open(f'{DATA_DIR}/images/{validate_df.iloc[i]["image_id"]}.jpg')
    img = img.resize(new_img_size)
    images_validate = np.concatenate(
        (images_validate, np.array([np.asarray(img)])), axis=0
    )



## === cell 9
img = Image.open(f'{DATA_DIR}/images/{test_df.iloc[0]["image_id"]}.jpg')
img = img.resize(new_img_size)
images_test = np.array([np.asarray(img)])

for i in range(1, test_df.shape[0]):
    img = Image.open(f'{DATA_DIR}/images/{test_df.iloc[i]["image_id"]}.jpg')
    img = img.resize(new_img_size)
    images_test = np.concatenate((images_test, np.array([np.asarray(img)])), axis=0)

print("Training set is of shape : ", images_train.shape)
print("Validation set is of shape : ", images_validate.shape)
print("Test set is of shape : ", images_test.shape)



## === cell 10
leaf_fig = px.imshow(cv2.resize(images_train[42], (300, 200)))
leaf_fig.show()




## === cell 11
def edge_and_cut(img):
    emb_img = img.copy()
    edges = cv2.Canny(img, 100, 200)
    edge_coors = []
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            if edges[i][j] != 0:
                edge_coors.append((i, j))

    if len(edge_coors) == 0:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
        ax[0].imshow(img)
        ax[0].set_title("original", fontsize=24)
        ax[1].imshow(edges, cmap="gray")
        ax[1].set_title("canny edge (empty)", fontsize=24)
        plt.show()
        return

    row_min = edge_coors[np.argsort([coor[0] for coor in edge_coors])[0]][0]
    row_max = edge_coors[np.argsort([coor[0] for coor in edge_coors])[-1]][0]
    col_min = edge_coors[np.argsort([coor[1] for coor in edge_coors])[0]][1]
    col_max = edge_coors[np.argsort([coor[1] for coor in edge_coors])[-1]][1]
    new_img = img[row_min:row_max, col_min:col_max]

    emb_img[max(row_min - 10, 0) : row_min + 10, col_min:col_max] = [255, 0, 0]
    emb_img[max(row_max - 10, 0) : row_max + 10, col_min:col_max] = [255, 0, 0]
    emb_img[row_min:row_max, max(col_min - 10, 0) : col_min + 10] = [255, 0, 0]
    emb_img[row_min:row_max, max(col_max - 10, 0) : col_max + 10] = [255, 0, 0]

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 20))
    ax[0].imshow(img, cmap="gray")
    ax[0].set_title("original", fontsize=24)
    ax[1].imshow(edges, cmap="gray")
    ax[1].set_title("canny edge", fontsize=24)
    ax[2].imshow(emb_img, cmap="gray")
    ax[2].set_title("trimming", fontsize=24)
    plt.show()




## === cell 12
edge_and_cut(images_train[42])
edge_and_cut(images_train[25])
edge_and_cut(images_train[31])



## === cell 13
leaf_fig = px.imshow(cv2.resize(images_train[42], (200, 200)))
leaf_fig.show()




## === cell 14
def invert(img):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 20))
    ax[0].imshow(img)
    ax[0].set_title("original", fontsize=24)
    ax[1].imshow(cv2.flip(img, 0))
    ax[1].set_title("upside down", fontsize=24)
    ax[2].imshow(cv2.flip(img, 1))
    ax[2].set_title("horizontal flip", fontsize=24)
    plt.show()




## === cell 15
invert(images_train[42])
invert(images_train[25])
invert(images_train[31])




## === cell 16
def conv(img):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 20))
    kernel = np.ones((7, 7), np.float32) / 25
    conv_img = cv2.filter2D(img, -1, kernel)
    ax[0].imshow(img)
    ax[0].set_title("original", fontsize=24)
    ax[1].imshow(conv_img)
    ax[1].set_title("convolved", fontsize=24)
    plt.show()




## === cell 17
conv(images_train[42])
conv(images_train[25])
conv(images_train[31])




## === cell 18
def blur(img):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 20))
    ax[0].imshow(img)
    ax[0].set_title("original", fontsize=24)
    ax[1].imshow(cv2.blur(img, (100, 100)))
    ax[1].set_title("blurred", fontsize=24)
    plt.show()




## === cell 19
blur(images_train[42])
blur(images_train[25])
blur(images_train[31])



## === cell 20
model = Sequential(
    [
        layers.Rescaling(1.0 / 255.0, input_shape=(224, 224, 3)),
        layers.Conv2D(8, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.2),
        layers.Conv2D(16, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.2),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(4, activation="softmax"),
    ]
)
model.build()



## === cell 21
model.compile(
    optimizer="adam",
    loss=tf.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)



## === cell 22
batch_size = 32
epochs = 15

model.fit(
    images_train,
    train_df["label"],
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(images_validate, validate_df["label"]),
)



## === cell 23
data_augmentaion = Sequential(
    [
        layers.RandomRotation(factor=(-0.2, 0.2), seed=123),
        layers.RandomZoom(0.1),
    ]
)



## === cell 24
model = Sequential(
    [
        layers.Rescaling(1.0 / 255.0, input_shape=(224, 224, 3)),
        data_augmentaion,
        layers.Conv2D(8, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.2),
        layers.Conv2D(16, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.2),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.4),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(4, activation="softmax"),
    ]
)
model.build()



## === cell 25
model.compile(
    optimizer="adam",
    loss=tf.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

batch_size = 32
epochs = 15

history = model.fit(
    images_train,
    train_df["label"],
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(images_validate, validate_df["label"]),
)



## === cell 26
train_acc = history.history["accuracy"]
validate_acc = history.history["val_accuracy"]
epoch_ = range(1, epochs + 1)

plt.plot(epoch_, train_acc, "r", label="Training")
plt.plot(epoch_, validate_acc, "b", label="Validation")
plt.legend()
plt.show()



## === cell 27
predictions = model.predict(images_test)
predictions.shape



## === cell 28
predictions_df = pd.DataFrame(predictions, columns=[label_encode[i] for i in range(4)])
submission_df = pd.concat(
    [
        test_df[["image_id"]].reset_index(drop=True),
        predictions_df.reset_index(drop=True),
    ],
    axis=1,
)

submission_df.head()



## === cell 29
sample_sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

submission_df = submission_df[sample_sub.columns]

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

print("Wrote", submission_path, "with shape:", submission_df.shape)
print(submission_df.head())
