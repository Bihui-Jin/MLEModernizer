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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tf_keras==2.18.0
tqdm==4.67.1

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

0.5055

# 6. Current score

0.98982

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.98982) has done: 'I fix the environment/runtime crash by importing TensorFlow before `tf_keras`, which avoids the protobuf `MessageFactory.GetPrototype` incompatibility seen in Kaggle images. Then I correct the dataset paths (your `train/train` and `test/test` subfolders don’t exist here) so image loading works and the empty-directory/`FileNotFoundError` chain disappears. I also make the visualization cell robust (skip plotting if the folder is empty) without changing training logic. Finally, I ensure inference runs on a non-empty test array and always writes a valid `submission.csv` with the required `id,has_cactus` columns.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import tensorflow as tf  # noqa: F401

from tf_keras.layers import (
    Dense,
    Flatten,
    Conv2D,
    MaxPool2D,
    Dropout,
    LeakyReLU,
    BatchNormalization,
)
from tf_keras.preprocessing.image import ImageDataGenerator
from tf_keras.callbacks import EarlyStopping
from tf_keras.models import Sequential
from tf_keras import optimizers

np.random.seed(42)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
HEIGHT = 32
WIDTH = 32

FULL_PATH = "/kaggle/input/aerial-cactus-identification"


def _pick_dir(*candidates):
    for d in candidates:
        if os.path.isdir(d) and len(os.listdir(d)) > 0:
            return d
    for d in candidates:
        if os.path.isdir(d):
            return d
    return candidates[0]


TRAIN_DIR = _pick_dir(
    os.path.join(FULL_PATH, "train"),
    os.path.join(FULL_PATH, "train", "train"),
)
TEST_DIR = _pick_dir(
    os.path.join(FULL_PATH, "test"),
    os.path.join(FULL_PATH, "test", "test"),
)

LABELS = os.path.join(FULL_PATH, "train.csv")
SAMPLE_SUB = _pick_dir(FULL_PATH)  # just to keep same base; file below
SAMPLE_SUB = os.path.join(FULL_PATH, "sample_submission.csv")

assert os.path.exists(LABELS), f"Missing labels file: {LABELS}"
assert os.path.isdir(TRAIN_DIR), f"Missing train dir: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing test dir: {TEST_DIR}"
assert os.path.exists(SAMPLE_SUB), f"Missing sample submission file: {SAMPLE_SUB}"

print("Using TRAIN_DIR:", TRAIN_DIR)
print("Using TEST_DIR :", TEST_DIR)




## === cell 2
def process_image(img, width=WIDTH, height=HEIGHT):
    proc_img = (
        Image.open(img).resize((width, height), Image.Resampling.LANCZOS).convert("RGB")
    )
    return np.asarray(proc_img)




## === cell 3
def plot_loss_accuracy(history):
    plt.plot(history.history.get("loss", []))
    plt.plot(history.history.get("val_loss", []))
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Val"], loc="upper left")
    plt.show()




## === cell 4
train = pd.read_csv(LABELS)
assert {"id", "has_cactus"}.issubset(train.columns)
train["has_cactus"] = train["has_cactus"].astype(np.float32)

missing = 0
for _id in train["id"].head(50).values:
    if not os.path.exists(os.path.join(TRAIN_DIR, _id)):
        missing += 1
print(f"Sanity check: {missing}/50 first train files missing (should be 0).")



## === cell 5
fig = plt.figure(figsize=(25, 8))
train_imgs = [f for f in os.listdir(TRAIN_DIR) if f.lower().endswith(".jpg")]

if len(train_imgs) >= 1:
    n_show = min(20, len(train_imgs))
    for idx, img in enumerate(np.random.choice(train_imgs, n_show, replace=False)):
        ax = fig.add_subplot(4, max(1, n_show // 4), idx + 1, xticks=[], yticks=[])
        im = Image.open(os.path.join(TRAIN_DIR, img))
        plt.imshow(im)
        row = train.loc[train["id"] == img, "has_cactus"].values
        lab = row[0] if len(row) else -1
        ax.set_title(f"Label: {int(lab)}")
    plt.tight_layout()
    plt.show()
else:
    print("TRAIN_DIR appears empty; skipping sample plot.")



## === cell 6
images = []
for img in tqdm(train["id"].values, desc="Loading train images"):
    path = os.path.join(TRAIN_DIR, img)
    images.append(process_image(path))

trainX = np.asarray(images, dtype=np.float32)
trainY = train["has_cactus"].values.astype(np.float32)

print("trainX:", trainX.shape, trainX.dtype)
print("trainY:", trainY.shape, trainY.dtype)



## === cell 7
x_train, x_test, y_train, y_test = train_test_split(
    trainX, trainY, stratify=trainY, test_size=0.2, random_state=42
)

print("x_train:", x_train.shape, "x_test:", x_test.shape)



## === cell 8
model = Sequential()

model.add(Conv2D(64, (5, 5), activation="relu", input_shape=(HEIGHT, WIDTH, 3)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.3))

model.add(Conv2D(64, (5, 5)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.3))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(128, (5, 5)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.3))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.3))

model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.3))
model.add(LeakyReLU(alpha=0.3))
model.add(Dense(1, activation="sigmoid"))



## === cell 9
datagen = ImageDataGenerator(rescale=1.0 / 255.0)
datagen.fit(x_train)



## === cell 10
opt = optimizers.RMSprop(learning_rate=0.001)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])



## === cell 11
callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)
]



## === cell 12
epochs = 30
batch_size = 64

history = model.fit(
    datagen.flow(x_train, y_train, batch_size=batch_size, shuffle=True),
    epochs=epochs,
    callbacks=callbacks,
    steps_per_epoch=max(1, x_train.shape[0] // 8),
    validation_data=(x_test / 255.0, y_test),
    verbose=1,
)



## === cell 13
plot_loss_accuracy(history)



## === cell 14
loss, accuracy = model.evaluate(x_test / 255.0, y_test, verbose=0)
print("Test Set Accuracy: ", str(accuracy * 100), "%")



## === cell 15
submission = pd.read_csv(SAMPLE_SUB)
assert "id" in submission.columns and "has_cactus" in submission.columns

images_test = []
for filename in tqdm(submission["id"].values, desc="Loading test images"):
    images_test.append(process_image(os.path.join(TEST_DIR, filename)))

images_test = np.asarray(images_test, dtype=np.float32)
images_test /= 255.0

print("images_test:", images_test.shape, images_test.dtype)



## === cell 16
assert (
    images_test.shape[0] == submission.shape[0] and images_test.shape[0] > 0
), f"Empty or misaligned test array: images_test={images_test.shape}, submission={submission.shape}"

prediction = model.predict(images_test, verbose=0).reshape(-1)
prediction = np.clip(prediction, 0.0, 1.0)



## === cell 17
submission["has_cactus"] = prediction.astype(np.float32)
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
