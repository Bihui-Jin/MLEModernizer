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

3.9

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

0.9987

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import sys

try:
    import google.protobuf  # noqa: F401
    from packaging.version import parse as _vparse
    import google.protobuf as _pb

    if _vparse(_pb.__version__) >= _vparse("5.0.0"):
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
        )
        os.execv(sys.executable, [sys.executable] + sys.argv)
except Exception:
    pass

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import zipfile
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow import keras

from tensorflow.keras.layers import Conv2D, Dense, Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

print(os.listdir("../input/aerial-cactus-identification/"))



## === cell 1
BASE_DIR = "/kaggle/temp"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_ROOT_DIR = os.path.join(BASE_DIR, "test")  # root for flow_from_directory
TEST_CLASS_DIR = os.path.join(
    TEST_ROOT_DIR, "test"
)  # subdir required by flow_from_directory

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_CLASS_DIR, exist_ok=True)

with zipfile.ZipFile("../input/aerial-cactus-identification/train.zip", "r") as z:
    z.extractall(BASE_DIR)  # yields /kaggle/temp/train/*.jpg

with zipfile.ZipFile("../input/aerial-cactus-identification/test.zip", "r") as z:
    z.extractall(TEST_CLASS_DIR)  # yields /kaggle/temp/test/test/*.jpg

print("train images:", len(os.listdir(TRAIN_DIR)))
print("test images:", len(os.listdir(TEST_CLASS_DIR)))



## === cell 2
train_dir = TRAIN_DIR
test_dir = TEST_ROOT_DIR

labels = pd.read_csv("../input/aerial-cactus-identification/train.csv")
labels.has_cactus = labels.has_cactus.astype(
    str
)  # required by flow_from_dataframe with class_mode='binary'
print(labels["has_cactus"].value_counts())



## === cell 3
rand_images = random.sample(os.listdir(train_dir), 16)

fig = plt.figure(figsize=(16, 4))
for i, im in enumerate(rand_images):
    plt.subplot(2, 8, i + 1)
    im_arr = cv2.imread(os.path.join(train_dir, im))
    plt.imshow(im_arr)
    plt.axis("off")
plt.show()



## === cell 4
rng = np.random.RandomState(42)
train_frac = 0.8

idxs_train = np.zeros(len(labels), dtype=bool)
for cls in labels["has_cactus"].unique():
    cls_idx = np.where(labels["has_cactus"].values == cls)[0]
    cls_perm = rng.permutation(cls_idx)
    n_train = int(round(train_frac * len(cls_perm)))
    idxs_train[cls_perm[:n_train]] = True

train_labels = labels[idxs_train].reset_index(drop=True)
val_labels = labels[~idxs_train].reset_index(drop=True)
print(len(train_labels), len(val_labels))



## === cell 5
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1 / 255, horizontal_flip=True, vertical_flip=True
)

batch_size = 128

train_generator = train_datagen.flow_from_dataframe(
    train_labels,
    directory=train_dir,
    x_col="id",
    y_col="has_cactus",
    class_mode="binary",
    batch_size=batch_size,
    target_size=(32, 32),
    shuffle=True,
    seed=42,  # deterministic ordering/augmentation for repeatability
)

val_generator = train_datagen.flow_from_dataframe(
    val_labels,
    directory=train_dir,
    x_col="id",
    y_col="has_cactus",
    class_mode="binary",
    batch_size=batch_size,
    target_size=(32, 32),
    shuffle=False,
)



## === cell 6
input_shape = (32, 32, 3)

model = keras.models.Sequential()

model.add(
    Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=input_shape)
)
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.summary()



## === cell 7
model.compile(loss=keras.losses.binary_crossentropy, optimizer="adam", metrics=["acc"])

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=20,
        verbose=1,
        restore_best_weights=True,
    ),
    ReduceLROnPlateau(patience=10, verbose=1),
]



## === cell 8
epochs = 100

history = model.fit(
    train_generator,
    epochs=epochs,
    verbose=1,
    callbacks=callbacks,
    validation_data=val_generator,
)



## === cell 9
val_acc_key = "val_acc" if "val_acc" in history.history else "val_accuracy"
acc_key = "acc" if "acc" in history.history else "accuracy"

idx = int(np.argmax(history.history[val_acc_key]))
print(history.history["val_loss"][idx], history.history[val_acc_key][idx])

idx = int(np.argmin(history.history["val_loss"]))
print(history.history["val_loss"][idx], history.history[val_acc_key][idx])



## === cell 10
plt.figure(figsize=(16, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history[acc_key], label="training accuracy")
plt.xlabel("# epochs")
plt.ylabel("Accuracy")

plt.plot(history.history[val_acc_key], label="validation accuracy")
plt.title("Accuracy evolution")
plt.legend()
plt.ylim(0.9, 1.01)

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="training loss")
plt.xlabel("# epochs")
plt.ylabel("Loss - Binary Cross Entropy")

plt.plot(history.history["val_loss"], label="validation loss")
plt.title("Loss evolution")
plt.legend()
plt.ylim(-0.01, 0.1)

plt.show()



## === cell 11
test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255)

test_generator = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(32, 32),
    batch_size=1,
    class_mode=None,
    shuffle=False,
)

probabilities = model.predict(test_generator, verbose=1)



## === cell 12
sample_submission = pd.read_csv(
    "../input/aerial-cactus-identification/sample_submission.csv"
)

probs_1d = probabilities.reshape(-1)

pred_files = [os.path.basename(p) for p in test_generator.filenames]
if len(pred_files) != len(probs_1d):
    raise ValueError(
        f"filenames length ({len(pred_files)}) != predictions length ({len(probs_1d)})"
    )

pred_df = pd.DataFrame({"id": pred_files, "has_cactus": probs_1d})

df = sample_submission[["id"]].merge(pred_df, on="id", how="left")

if df["has_cactus"].isna().any():
    missing = df.loc[df["has_cactus"].isna(), "id"].head(5).tolist()
    raise ValueError(f"Some test IDs missing predictions (showing up to 5): {missing}")

df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df.shape)
print(df.head())
