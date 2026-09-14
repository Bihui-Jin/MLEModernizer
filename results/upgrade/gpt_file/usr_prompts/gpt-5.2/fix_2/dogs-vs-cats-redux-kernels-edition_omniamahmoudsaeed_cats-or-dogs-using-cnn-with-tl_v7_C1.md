# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given a dataset of images of dogs and cats, predict if an image is a dog or a cat.

## Metric
Log loss.

## Submission Format
For each image in the test set, you must submit a probability that image is a dog. The file should have a header and be in the following format:

```
id,label
1,0.5
2,0.5
3,0.5
...
```

## Dataset
The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id.

# 2. Python version

3.13

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
        input/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
        working/
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
```

-> data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> working/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

# 5. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import zipfile
from random import shuffle

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



## === cell 1
TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 64
NO_EPOCHS = 20
NUM_CLASSES = 2
SAMPLE_SIZE = 20000
IMG_SIZE = 128

TRAIN_FOLDER = "/kaggle/working/train"
TEST_FOLDER = "/kaggle/working/test"
PATH_TRAIN = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip"
PATH_TEST = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip"



## === cell 2
os.makedirs(TRAIN_FOLDER, exist_ok=True)
os.makedirs(TEST_FOLDER, exist_ok=True)

with zipfile.ZipFile(PATH_TRAIN, "r") as zip_ref:
    zip_ref.extractall(TRAIN_FOLDER)
with zipfile.ZipFile(PATH_TEST, "r") as zip_ref:
    zip_ref.extractall(TEST_FOLDER)


def find_images_dir(root_dir: str, kind: str) -> str:
    """
    Find directory containing jpg files for train/test after extraction.
    kind: 'train' or 'test' (used only as a hint).
    """
    candidates = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        jpgs = [f for f in filenames if f.lower().endswith(".jpg")]
        if jpgs:
            candidates.append((dirpath, len(jpgs)))
    if not candidates:
        raise FileNotFoundError(f"No .jpg files found under: {root_dir}")

    candidates.sort(key=lambda x: (kind not in x[0].lower(), -x[1], len(x[0])))
    return candidates[0][0]


train_inner_folder = find_images_dir(TRAIN_FOLDER, "train")
test_inner_folder = find_images_dir(TEST_FOLDER, "test")

train_image_list = os.listdir(train_inner_folder)
train_image_list = [f for f in train_image_list if f.lower().endswith(".jpg")][
    :SAMPLE_SIZE
]

test_image_list = os.listdir(test_inner_folder)
test_image_list = [f for f in test_image_list if f.lower().endswith(".jpg")]

print("train_inner_folder:", train_inner_folder)
print("test_inner_folder :", test_inner_folder)
print("Found train images:", len(os.listdir(train_inner_folder)))
print("Using SAMPLE_SIZE:", len(train_image_list))
print("Found test images:", len(test_image_list))




## === cell 3
def label_pet_image_one_hot_encoder(img_filename):
    pet = img_filename.split(".")[0]
    if pet == "cat":
        return [1, 0]
    elif pet == "dog":
        return [0, 1]
    return [1, 0]




## === cell 4
def process_data(data_image_list, DATA_FOLDER, isTrain=True):
    data_df = []
    for img in data_image_list:
        path = os.path.join(DATA_FOLDER, img)
        if isTrain:
            label = label_pet_image_one_hot_encoder(img)
        else:
            label = img
        img_array = cv2.imread(path)
        if img_array is None:
            continue
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        data_df.append([np.array(img_array), np.array(label)])
    shuffle(data_df)
    return data_df




## === cell 5
def plot_image_list_count(data_image_list):
    labels = []
    for img in data_image_list:
        labels.append(img.split(".")[0])
    plt.figure(figsize=(6, 4))
    sns.countplot(x=labels)
    plt.title("Cats and Dogs")
    plt.show()




## === cell 6
plot_image_list_count(train_image_list)



## === cell 7
train = process_data(train_image_list, train_inner_folder, True)




## === cell 8
def show_images(data, isTest=False):
    f, ax = plt.subplots(5, 5, figsize=(12, 12))
    for i, item in enumerate(data[:25]):
        img_data = item[0]
        img_num = item[1]
        if isTest:
            str_label = "None"
        else:
            label = np.argmax(img_num)
            str_label = "Dog" if label == 1 else "Cat"
        ax[i // 5, i % 5].imshow(img_data.astype(np.uint8))
        ax[i // 5, i % 5].axis("off")
        ax[i // 5, i % 5].set_title("Label: {}".format(str_label))
    plt.tight_layout()
    plt.show()


show_images(train)



## === cell 9
test = process_data(test_image_list, test_inner_folder, False)
show_images(test, True)



## === cell 10
X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y = np.array([i[1] for i in train])
print("X shape:", X.shape, "y shape:", y.shape)



## === cell 11
base = ResNet50(
    include_top=False,
    pooling="max",
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)
model = Sequential([base, Dense(NUM_CLASSES, activation="softmax")])
base.trainable = False
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()



## === cell 12
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
train_model = model.fit(
    X_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=NO_EPOCHS,
    verbose=1,
    validation_data=(X_val, y_val),
)




## === cell 13
def plot_accuracy_and_loss(train_model):
    hist = train_model.history
    acc = hist["accuracy"] if "accuracy" in hist else hist.get("acc", [])
    val_acc = (
        hist["val_accuracy"] if "val_accuracy" in hist else hist.get("val_acc", [])
    )
    loss = hist["loss"]
    val_loss = hist["val_loss"]
    epochs = range(len(acc))
    f, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].plot(epochs, acc, "g", label="Training accuracy")
    ax[0].plot(epochs, val_acc, "r", label="Validation accuracy")
    ax[0].set_title("Training and validation accuracy")
    ax[0].legend()
    ax[1].plot(epochs, loss, "g", label="Training loss")
    ax[1].plot(epochs, val_loss, "r", label="Validation loss")
    ax[1].set_title("Training and validation loss")
    ax[1].legend()
    plt.show()




## === cell 14
plot_accuracy_and_loss(train_model)



## === cell 15
score = model.evaluate(X_val, y_val, verbose=0)
print("Validation loss:", score[0])
print("Validation accuracy:", score[1])



## === cell 16
y_pred_proba = model.predict(X_val)
predicted_classes = np.argmax(y_pred_proba, axis=1)
y_true = np.argmax(y_val, axis=1)



## === cell 17
correct = np.nonzero(predicted_classes == y_true)[0]
incorrect = np.nonzero(predicted_classes != y_true)[0]
print("Correct:", len(correct), "Incorrect:", len(incorrect))



## === cell 18
target_names = ["Class 0 (cat)", "Class 1 (dog)"]
print(classification_report(y_true, predicted_classes, target_names=target_names))



## === cell 19
ss = pd.read_csv(
    "/kaggle/input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv"
)



## === cell 20
ss.head()



## === cell 21
X_test = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
test_filenames = [str(i[1]) for i in test]
test_ids = [int(os.path.splitext(f)[0]) for f in test_filenames]

y_pred_proba_test = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
dog_proba = y_pred_proba_test[:, 1].astype(
    float
)  # class 1 = dog based on label_pet_image_one_hot_encoder

pred_df = pd.DataFrame({"id": test_ids, "label": dog_proba})

submission = ss[["id"]].merge(pred_df, on="id", how="left")

submission["label"] = submission["label"].fillna(0.5).clip(1e-7, 1 - 1e-7)

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)



## === cell 22
submission.head()



## === cell 23
submission.shape
