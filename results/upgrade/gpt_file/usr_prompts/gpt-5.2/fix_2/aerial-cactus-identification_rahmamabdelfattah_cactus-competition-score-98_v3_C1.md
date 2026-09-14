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

0.9815

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'I fix the import/runtime crash by removing incompatible/unused imports (the `MessageFactory.GetPrototype` error is a known protobuf/tf/keras mismatch trigger) and using only `tensorflow.keras` callbacks. Then I fix the data extraction/paths so `train.zip`/`test.zip` extract into the expected folders and the loader can actually find images. Next I correct image loading to produce numeric arrays of shape `(32,32,3)` and normalize consistently for both train/val and test, keeping the same model and training loop. Finally, I ensure test files are loaded in the exact `sample_submission.csv` order and write a valid `submission.csv` with probabilities (not hard labels), which is appropriate for ROC-AUC.'

# 9. Code solution

## === cell 0
import os
import glob
from zipfile import ZipFile

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

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
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report

np.random.seed(42)
tf.random.set_seed(42)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
path = "/kaggle/input/aerial-cactus-identification/"
work_dir = "/kaggle/working"

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
os.makedirs(work_dir, exist_ok=True)

with ZipFile(train_zip_path) as zipper:
    zipper.extractall(work_dir)

with ZipFile(test_zip_path) as zipper:
    zipper.extractall(work_dir)

train_path = os.path.join(work_dir, "train")
test_path = os.path.join(work_dir, "test")

assert os.path.isdir(train_path), f"train_path not found: {train_path}"
assert os.path.isdir(test_path), f"test_path not found: {test_path}"



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_11/2348950815.py in <cell line: 0>()
     12 test_path = os.path.join(work_dir, "test")
     13 
---> 14 assert os.path.isdir(train_path), f"train_path not found: {train_path}"
     15 assert os.path.isdir(test_path), f"test_path not found: {test_path}"
     16 

AssertionError: train_path not found: /kaggle/working/train

## === cell 5
print("Num train images found:", len(glob.glob(os.path.join(train_path, "*.jpg"))))
print("Num test images found:", len(glob.glob(os.path.join(test_path, "*.jpg"))))




## === cell 6
def load_data(train_labels_df, train_dir):
    x = np.zeros((len(train_labels_df), 32, 32, 3), dtype=np.float32)
    y = np.zeros((len(train_labels_df),), dtype=np.int32)

    for idx in range(len(train_labels_df)):
        img_name = train_labels_df.iloc[idx, 0]
        img_path = os.path.join(train_dir, img_name)

        image = load_img(img_path, target_size=(32, 32))
        img_array = img_to_array(image).astype(np.float32)

        x[idx] = img_array
        y[idx] = int(train_labels_df.iloc[idx, 1])

    return x, y




## === cell 7
x_train, y_train = load_data(train_labels, train_path)
print("x_train:", x_train.shape, x_train.dtype)
print("y_train:", y_train.shape, y_train.dtype, "positives:", int(y_train.sum()))



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3662589903.py in <cell line: 0>()
----> 1 x_train, y_train = load_data(train_labels, train_path)
      2 print("x_train:", x_train.shape, x_train.dtype)
      3 print("y_train:", y_train.shape, y_train.dtype, "positives:", int(y_train.sum()))
      4 

/tmp/ipykernel_11/2348552235.py in load_data(train_labels_df, train_dir)
      9 
     10         # load and convert to array
---> 11         image = load_img(img_path, target_size=(32, 32))
     12         img_array = img_to_array(image).astype(np.float32)
     13 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/image_utils.py in load_img(path, color_mode, target_size, interpolation, keep_aspect_ratio)
    233         if isinstance(path, pathlib.Path):
    234             path = str(path.resolve())
--> 235         with open(path, "rb") as f:
    236             img = pil_image.open(io.BytesIO(f.read()))
    237     else:

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/train/2de8f189f1dce439766637e75df0ee27.jpg'

## === cell 8
x_train = x_train / 255.0




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3950548009.py in <cell line: 0>()
      1 # Normalize inputs (score-positive and standard; does not change core modeling approach)
----> 2 x_train = x_train / 255.0
      3 
      4 

NameError: name 'x_train' is not defined

## === cell 9
def display_examples(class_names_list, images, labels):
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle("Some examples of images of the dataset", fontsize=16)
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
        plt.xlabel(class_names_list[int(labels[i])])
    plt.show()




## === cell 10
display_examples(class_names, x_train, y_train)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1994255891.py in <cell line: 0>()
----> 1 display_examples(class_names, x_train, y_train)
      2 

NameError: name 'x_train' is not defined

## === cell 11
unique_labels, train_counts = np.unique(y_train, return_counts=True)
print(f"{unique_labels[0]}: {train_counts[0]}\n{unique_labels[1]}: {train_counts[1]}")



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/976855323.py in <cell line: 0>()
      1 # (Optional) distribution checks
----> 2 unique_labels, train_counts = np.unique(y_train, return_counts=True)
      3 print(f"{unique_labels[0]}: {train_counts[0]}\n{unique_labels[1]}: {train_counts[1]}")
      4 

NameError: name 'y_train' is not defined

## === cell 12
plt.figure(figsize=(4, 4))
plt.bar(unique_labels, train_counts, color="skyblue", edgecolor="black")
plt.xlabel("Class Labels")
plt.ylabel("Number of Samples")
plt.title("Training Set Class Distribution - Bar Chart")
plt.xticks(unique_labels)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3738049679.py in <cell line: 0>()
      1 plt.figure(figsize=(4, 4))
----> 2 plt.bar(unique_labels, train_counts, color="skyblue", edgecolor="black")
      3 plt.xlabel("Class Labels")
      4 plt.ylabel("Number of Samples")
      5 plt.title("Training Set Class Distribution - Bar Chart")

NameError: name 'unique_labels' is not defined

## === cell 13
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



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2691473930.py in <cell line: 0>()
      1 plt.figure(figsize=(4, 4))
      2 plt.pie(
----> 3     train_counts,
      4     labels=unique_labels,
      5     autopct="%1.1f%%",

NameError: name 'train_counts' is not defined

## === cell 14
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)

n_train = len(x_train)
n_val = len(x_val)
print("Number of training examples:", n_train)
print("Number of validation examples:", n_val)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3046205486.py in <cell line: 0>()
      1 x_train, x_val, y_train, y_val = train_test_split(
----> 2     x_train, y_train, test_size=0.2, stratify=y_train, random_state=42
      3 )
      4 
      5 n_train = len(x_train)

NameError: name 'x_train' is not defined

## === cell 15
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



## === cell 16
early_stop = EarlyStopping(
    monitor="val_accuracy", mode="max", patience=5, restore_best_weights=True, verbose=1
)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, verbose=1)

model.summary()



## === cell 17
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])



## === cell 18
classes = np.unique(y_train)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weights = {int(c): float(w) for c, w in zip(classes, weights)}
class_weights



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1226165986.py in <cell line: 0>()
      1 # Fix: compute_class_weight returns weights aligned with 'classes' values; map them correctly.
----> 2 classes = np.unique(y_train)
      3 weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
      4 class_weights = {int(c): float(w) for c, w in zip(classes, weights)}
      5 class_weights

NameError: name 'y_train' is not defined

## === cell 19
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=30,
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weights,
    verbose=2,
)



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2972992768.py in <cell line: 0>()
      1 history = model.fit(
----> 2     x_train,
      3     y_train,
      4     validation_data=(x_val, y_val),
      5     epochs=30,

NameError: name 'x_train' is not defined

## === cell 20
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over Epochs")
plt.legend()
plt.grid()
plt.show()



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/817662460.py in <cell line: 0>()
----> 1 plt.plot(history.history["accuracy"], label="Training Accuracy")
      2 plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
      3 plt.xlabel("Epoch")
      4 plt.ylabel("Accuracy")
      5 plt.title("Accuracy over Epochs")

NameError: name 'history' is not defined

## === cell 21
preds = model.predict(x_val, verbose=0)
preds_labels = (preds > 0.5).astype(int).flatten()
print(classification_report(y_val, preds_labels, digits=4))

conf_matrix = confusion_matrix(y_val, preds_labels)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/369839987.py in <cell line: 0>()
----> 1 preds = model.predict(x_val, verbose=0)
      2 preds_labels = (preds > 0.5).astype(int).flatten()
      3 print(classification_report(y_val, preds_labels, digits=4))
      4 
      5 conf_matrix = confusion_matrix(y_val, preds_labels)

NameError: name 'x_val' is not defined

## === cell 22
sample_sub = pd.read_csv(sample_sub_path)
test_ids = sample_sub["id"].tolist()

x_test = np.zeros((len(test_ids), 32, 32, 3), dtype=np.float32)
missing = 0
for i, img_name in enumerate(test_ids):
    img_path = os.path.join(test_path, img_name)
    if not os.path.exists(img_path):
        missing += 1
        continue
    img = load_img(img_path, target_size=(32, 32))
    x_test[i] = img_to_array(img).astype(np.float32)

if missing:
    print("Warning: missing test images:", missing)

x_test = x_test / 255.0
print("x_test:", x_test.shape, x_test.dtype)



## === cell 23
pred_probs = model.predict(x_test, verbose=0).reshape(-1)

df_submission = pd.DataFrame({"id": test_ids, "has_cactus": pred_probs})
df_submission.to_csv("submission.csv", index=False)

print(df_submission.head())
print("Wrote submission.csv with shape:", df_submission.shape)
