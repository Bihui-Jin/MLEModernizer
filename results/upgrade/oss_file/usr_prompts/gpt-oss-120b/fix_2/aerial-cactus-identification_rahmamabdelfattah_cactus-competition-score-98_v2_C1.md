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

0.99773

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.99773) has done: 'I fixed the import conflict that caused the protobuf error, corrected the paths where the zip files are extracted, and updated the data‑loading function to read images as normalized NumPy arrays (so the model receives the right shape and scale). The test‑set path is also fixed, and the submission now writes the raw predicted probabilities (the required format) instead of hard‑coded class labels. All changes are minimal and keep the original model architecture and training logic intact.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
from zipfile import ZipFile
import glob
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
path = "/kaggle/input/aerial-cactus-identification/"
train_labels = pd.read_csv(os.path.join(path, "train.csv"))
train_labels.head()



## === cell 2
class_names = ["Has cactus", "Hasn't cactus"]
class_names_label = {class_name: i for i, class_name in enumerate(class_names)}
nb_classes = len(class_names)
class_names_label



## === cell 3
with ZipFile(os.path.join(path, "train.zip")) as zipper:
    zipper.extractall()
with ZipFile(os.path.join(path, "test.zip")) as zipper:
    zipper.extractall()



## === cell 4
base_dir = "/kaggle/working/aerial-cactus-identification"
train_path = os.path.join(base_dir, "train")
test_path = os.path.join(base_dir, "test")




## === cell 5
def load_data(train_labels, train_path):
    x_train = []
    y_train = []
    for idx in range(len(train_labels)):
        img_name = train_labels.iloc[idx, 0]
        img_path = os.path.join(train_path, img_name)
        image = Image.open(img_path).convert("RGB")
        arr = img_to_array(image) / 255.0  # shape (32,32,3), float32
        label = train_labels.iloc[idx, 1]
        x_train.append(arr)
        y_train.append(label)
    return np.array(x_train, dtype="float32"), np.array(y_train, dtype="int32")




## === cell 6
x_train, y_train = load_data(train_labels, train_path)




## === cell 7
def display_examples(class_names, images, labels):
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle("Some examples of images of the dataset", fontsize=16)
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
        plt.xlabel(class_names[labels[i]])
    plt.show()




## === cell 8
display_examples(class_names, x_train, y_train)



## === cell 9
unique_labels, train_counts = np.unique(y_train, return_counts=True)
print(f"{unique_labels[0]}: {train_counts[0]}\n{unique_labels[1]}: {train_counts[1]}")



## === cell 10
plt.figure(figsize=(4, 4))
plt.bar(unique_labels, train_counts, color="skyblue", edgecolor="black")
plt.xlabel("Class Labels")
plt.ylabel("Number of Samples")
plt.title("Training Set Class Distribution - Bar Chart")
plt.xticks(unique_labels)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()



## === cell 11
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



## === cell 12
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)



## === cell 13
n_train = len(x_train)
n_val = len(x_val)
print("Number of training examples:", n_train)
print("Number of validation examples:", n_val)



## === cell 14
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



## === cell 15
early_stop = EarlyStopping(
    monitor="val_accuracy", mode="max", patience=5, restore_best_weights=True, verbose=1
)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, verbose=1)



## === cell 16
model.summary()



## === cell 17
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])



## === cell 18
class_weights_arr = compute_class_weight(
    class_weight="balanced", classes=np.unique(y_train), y=y_train
)
class_weights = dict(enumerate(class_weights_arr))
class_weights



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



## === cell 20
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over Epochs")
plt.legend()
plt.grid()
plt.show()



## === cell 21
preds = model.predict(x_val)
preds_labels = (preds > 0.5).astype(int).flatten()
print(classification_report(y_val, preds_labels, digits=4))



## === cell 22
conf_matrix = confusion_matrix(y_val, preds_labels)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



## === cell 23
test_images = glob.glob(os.path.join(test_path, "*.jpg"))



## === cell 24
x_test = []
for p in test_images:
    img = cv2.imread(p)
    img = cv2.resize(img, (32, 32))
    img = img / 255.0
    x_test.append(img)
x_test = np.array(x_test, dtype="float32")



## === cell 25
pred_probs = model.predict(x_test).flatten()  # probabilities required for submission



## === cell 26
image_names = [os.path.basename(p) for p in test_images]
df_submission = pd.DataFrame({"id": image_names, "has_cactus": pred_probs})
df_submission.to_csv("submission.csv", index=False)
