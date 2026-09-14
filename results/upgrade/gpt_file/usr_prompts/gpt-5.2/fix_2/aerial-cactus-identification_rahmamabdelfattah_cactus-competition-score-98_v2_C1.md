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

3.13

# 3. Installed packages

No external packages required in the script and installed.

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

0.5

# 6. Current score

0.99945

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.99945) has done: 'I fix the import/runtime crash caused by mixing standalone `keras` with `tensorflow.keras` (protobuf MessageFactory error) by using `tf.keras` callbacks consistently. I also fix the missing extracted image paths by extracting `train.zip`/`test.zip` into a known folder and pointing `train_path`/`test_path` to the actual extracted directories. Next, I correct the data pipeline so images become numeric `float32` arrays in `[0,1]` with shape `(32,32,3)`, which prevents model training/inference shape/type issues. Finally, I ensure test predictions are probabilities (not thresholded labels) and that the submission rows align with `sample_submission.csv` order, producing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import glob
from zipfile import ZipFile

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import cv2
from PIL import Image

import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    Flatten,
    Dense,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

np.random.seed(42)
tf.random.set_seed(42)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
path = "/kaggle/input/aerial-cactus-identification/"
workdir = "/kaggle/working"

train_csv_path = os.path.join(path, "train.csv")
sample_sub_path = os.path.join(path, "sample_submission.csv")

train_zip_path = os.path.join(path, "train.zip")
test_zip_path = os.path.join(path, "test.zip")



## === cell 2
train_labels = pd.read_csv(train_csv_path)
train_labels.head()



## === cell 3
class_names = ["Has cactus", "Hasn't cactus"]
class_names_label = {class_name: i for i, class_name in enumerate(class_names)}
nb_classes = len(class_names)
class_names_label



## === cell 4
extract_root = os.path.join(workdir, "aerial_cactus_extracted")
os.makedirs(extract_root, exist_ok=True)

train_extract_dir = os.path.join(extract_root, "train")
test_extract_dir = os.path.join(extract_root, "test")
os.makedirs(train_extract_dir, exist_ok=True)
os.makedirs(test_extract_dir, exist_ok=True)

with ZipFile(train_zip_path) as zipper:
    zipper.extractall(train_extract_dir)

with ZipFile(test_zip_path) as zipper:
    zipper.extractall(test_extract_dir)


def resolve_image_dir(root_dir, expected_leaf):
    candidate = os.path.join(root_dir, expected_leaf)
    if (
        os.path.isdir(candidate)
        and len(glob.glob(os.path.join(candidate, "*.jpg"))) > 0
    ):
        return candidate
    if len(glob.glob(os.path.join(root_dir, "*.jpg"))) > 0:
        return root_dir
    matches = []
    for d, _, _ in os.walk(root_dir):
        if len(glob.glob(os.path.join(d, "*.jpg"))) > 0:
            matches.append(d)
    if not matches:
        raise FileNotFoundError(f"No jpg images found under {root_dir}")
    return sorted(
        matches, key=lambda p: len(glob.glob(os.path.join(p, "*.jpg"))), reverse=True
    )[0]


train_path = resolve_image_dir(train_extract_dir, "train")
test_path = resolve_image_dir(test_extract_dir, "test")

print("Resolved train_path:", train_path)
print("Resolved test_path :", test_path)
print("Train jpgs:", len(glob.glob(os.path.join(train_path, "*.jpg"))))
print("Test  jpgs:", len(glob.glob(os.path.join(test_path, "*.jpg"))))



## === cell 5
pass




## === cell 6
def load_data(train_labels_df, train_img_dir):
    x = np.empty((len(train_labels_df), 32, 32, 3), dtype=np.float32)
    y = np.empty((len(train_labels_df),), dtype=np.int32)

    for idx in range(len(train_labels_df)):
        img_name = train_labels_df.iloc[idx, 0]
        img_path = os.path.join(train_img_dir, img_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing image: {img_path}")

        img = cv2.imread(img_path)  # BGR uint8
        if img is None:
            raise ValueError(f"cv2.imread failed for: {img_path}")
        img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        x[idx] = img
        y[idx] = int(train_labels_df.iloc[idx, 1])

    return x, y




## === cell 7
x_train, y_train = load_data(train_labels, train_path)
x_train.shape, y_train.shape, x_train.dtype, y_train.dtype



## === cell 8
pass




## === cell 9
def display_examples(class_names, images, labels):
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle("Some examples of images of the dataset", fontsize=16)
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
        plt.xlabel(class_names[int(labels[i])])
    plt.show()




## === cell 10
display_examples(class_names, x_train, y_train)



## === cell 11
pass



## === cell 12
unique_labels, train_counts = np.unique(y_train, return_counts=True)
print(f"{unique_labels[0]}: {train_counts[0]}\n{unique_labels[1]}: {train_counts[1]}")



## === cell 13
plt.figure(figsize=(4, 4))
plt.bar(unique_labels, train_counts, color="skyblue", edgecolor="black")
plt.xlabel("Class Labels")
plt.ylabel("Number of Samples")
plt.title("Training Set Class Distribution - Bar Chart")
plt.xticks(unique_labels)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()



## === cell 14
plt.figure(figsize=(4, 4))
plt.pie(
    train_counts,
    labels=unique_labels,
    autopct="%1.1f%%",
    startangle=90,
    colors=plt.cm.Pastel1.colors,
)
plt.title("Training Set Class Distribution - Pie Chart")
plt.axis("equal")
plt.tight_layout()
plt.show()



## === cell 15
pass



## === cell 16
x_train = np.asarray(x_train, dtype="float32")
y_train = np.asarray(y_train, dtype="int32")



## === cell 17
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)



## === cell 18
n_train = len(x_train)
n_val = len(x_val)

print("Number of training examples:", n_train)
print("Number of validation examples:", n_val)



## === cell 19
pass



## === cell 20
model = Sequential(
    [
        Conv2D(32, 3, activation="relu", input_shape=(32, 32, 3)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Conv2D(64, 3, activation="relu"),
        BatchNormalization(),
        Conv2D(128, 3, activation="relu"),
        BatchNormalization(),
        Flatten(),
        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid"),
    ]
)



## === cell 21
early_stop = EarlyStopping(
    monitor="val_accuracy", mode="max", patience=5, restore_best_weights=True, verbose=1
)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, verbose=1)



## === cell 22
model.summary()



## === cell 23
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])



## === cell 24
class_weights_arr = compute_class_weight(
    class_weight="balanced", classes=np.unique(y_train), y=y_train
)
class_weights = {
    int(c): float(w) for c, w in zip(np.unique(y_train), class_weights_arr)
}
class_weights



## === cell 25
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=30,
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weights,
    verbose=2,
)



## === cell 26
pass



## === cell 27
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over Epochs")
plt.legend()
plt.grid()
plt.show()



## === cell 28
preds = model.predict(x_val, verbose=0)
preds_labels = (preds > 0.5).astype(int).flatten()
print(classification_report(y_val, preds_labels, digits=4))



## === cell 29
conf_matrix = confusion_matrix(y_val, preds_labels)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



## === cell 30
pass



## === cell 31
sample_sub = pd.read_csv(sample_sub_path)
test_ids = sample_sub["id"].tolist()



## === cell 32
x_test = np.empty((len(test_ids), 32, 32, 3), dtype=np.float32)

for i, img_name in enumerate(test_ids):
    img_path = os.path.join(test_path, img_name)
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Missing test image: {img_path}")

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"cv2.imread failed for: {img_path}")
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    x_test[i] = img

x_test.shape



## === cell 33
pred_probs = model.predict(x_test, verbose=0).reshape(-1).astype(np.float64)

pred_probs = np.clip(pred_probs, 0.0, 1.0)

pred_probs[:5], pred_probs.shape



## === cell 34
df_submission = pd.DataFrame({"id": test_ids, "has_cactus": pred_probs})

df_submission = df_submission[["id", "has_cactus"]]
df_submission.to_csv("submission.csv", index=False)

print(df_submission.head())
print("Wrote submission.csv with rows:", len(df_submission))
print("Saved at:", os.path.join(os.getcwd(), "submission.csv"))
