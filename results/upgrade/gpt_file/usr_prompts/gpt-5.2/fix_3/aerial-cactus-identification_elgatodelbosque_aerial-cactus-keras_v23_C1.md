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

0.9998

# 6. Current score

0.48758

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fix the Keras import/runtime crash by switching from the incompatible standalone `keras` package to `tf_keras` (which matches the installed environment), keeping the same model architecture and training loop. I also correct the dataset paths to the actual competition folder under `../input/aerial-cactus-identification/`, ensuring images and CSVs load in the correct order. For AUC, I fix the submission logic to write a single probability for the positive class (`has_cactus=1`) instead of a 2-column softmax array, which is the main cause of the very low score. Finally, I make the history plotting robust to metric key names so it doesn’t crash after training.'
- What this solution (achieved 0.48758) has done: 'I fix the two blockers that prevent the notebook from running: (1) the incorrect dataset subfolder paths (`train/train` and `test/test`) by auto-detecting the correct image directories, and (2) the Keras import crash by switching from `tf_keras` to `tensorflow.keras` (which is the most stable backend in Kaggle TF environments). These changes are execution/stability fixes and do not alter your model architecture, loss, optimizer, or training loop semantics. Once data loads and training runs, the existing submission logic (using the positive-class probability `[:, 1]`) produce a valid `submission.csv` with the required columns. This should also move the score sharply upward from 0.5 because the previous run never trained/predicted successfully.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd
import cv2
import random

random.seed(42)
np.random.seed(42)

print("Listing ../input:")
print(os.listdir("../input")[:20])



## === cell 1
DATA_ROOT_CANDIDATES = [
    "../input/aerial-cactus-identification",
    "../input/aerial-cactus-identification/aerial-cactus-identification",
]

path = None
for p in DATA_ROOT_CANDIDATES:
    if os.path.exists(p):
        path = p
        break
if path is None:
    raise FileNotFoundError(
        f"Could not find dataset root in candidates: {DATA_ROOT_CANDIDATES}"
    )


def find_image_dir(root, split):
    cands = [
        os.path.join(root, split),
        os.path.join(root, split, split),
    ]
    for c in cands:
        if os.path.isdir(c) and len(glob.glob(os.path.join(c, "*.jpg"))) > 0:
            return c
    for c in glob.glob(os.path.join(root, "**", split), recursive=True):
        if os.path.isdir(c) and len(glob.glob(os.path.join(c, "*.jpg"))) > 0:
            return c
    return None


train_dir = find_image_dir(path, "train")
test_dir = find_image_dir(path, "test")
if train_dir is None or test_dir is None:
    raise FileNotFoundError(
        f"Could not locate train/test image directories under: {path}"
    )

train_path = train_dir + ("" if train_dir.endswith(os.sep) else os.sep)
test_path = test_dir + ("" if test_dir.endswith(os.sep) else os.sep)

print("dataset root path:", path)
print("train_path:", train_path)
print("test_path:", test_path)
print("Num train jpg:", len(glob.glob(train_path + "*.jpg")))
print("Num test jpg:", len(glob.glob(test_path + "*.jpg")))



## === cell 2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

print("TensorFlow version:", tf.__version__)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
files = sorted(glob.glob(train_path + "*.jpg"))
if len(files) == 0:
    raise FileNotFoundError(f"No training images found in {train_path}")

train_raw = np.array([cv2.imread(image) for image in files], dtype="uint8")
if train_raw.ndim != 4 or train_raw.shape[-1] != 3:
    raise ValueError(f"Unexpected train_raw shape from cv2.imread: {train_raw.shape}")
print("train_raw shape:", train_raw.shape, "dtype:", train_raw.dtype)



## === cell 4
test_files = sorted(glob.glob(test_path + "*.jpg"))
if len(test_files) == 0:
    raise FileNotFoundError(f"No test images found in {test_path}")

test_raw = np.array([cv2.imread(image) for image in test_files], dtype="uint8")
if test_raw.ndim != 4 or test_raw.shape[-1] != 3:
    raise ValueError(f"Unexpected test_raw shape from cv2.imread: {test_raw.shape}")
print("test_raw shape:", test_raw.shape, "dtype:", test_raw.dtype)



## === cell 5
train_images = train_raw.astype("float32") / 255.0
test_images = test_raw.astype("float32") / 255.0

print("train_images range:", float(train_images.min()), float(train_images.max()))
print("test_images range:", float(test_images.min()), float(test_images.max()))



## === cell 6
train_csv_path = os.path.join(path, "train.csv")
if not os.path.exists(train_csv_path):
    alt = os.path.join("../input/aerial-cactus-identification", "train.csv")
    if os.path.exists(alt):
        train_csv_path = alt
    else:
        raise FileNotFoundError(
            f"train.csv not found at {os.path.join(path,'train.csv')} or {alt}"
        )

train_set = pd.read_csv(train_csv_path)
train_set = train_set.set_index("id")

train_ids = [os.path.basename(f) for f in files]
missing = [i for i in train_ids if i not in train_set.index]
if missing:
    raise ValueError(
        f"{len(missing)} training ids missing from train.csv, e.g. {missing[:3]}"
    )

train_labels = train_set.loc[train_ids, "has_cactus"].astype(int).values
train_labels = to_categorical(train_labels, num_classes=2)
print("train_labels shape:", train_labels.shape)



## === cell 7
model = models.Sequential()
model.add(
    layers.Conv2D(
        16, (3, 3), activation="relu", input_shape=(32, 32, 3), padding="same"
    )
)
model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same"))
model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (2, 2), activation="relu", padding="same"))
model.add(layers.Conv2D(128, (2, 2), activation="relu", padding="same"))
model.add(layers.Conv2D(64, (2, 2), activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(16))
model.add(layers.Dense(16))
model.add(layers.Dense(16))
model.add(layers.Dense(2, activation="softmax"))



## === cell 8
model.summary()



## === cell 9
cbl = [
    EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,  # score-neutral, avoids needing to load checkpoint
    ),
    ModelCheckpoint(
        filepath="best_model.h5",
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
    ),
    ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=10),
]



## === cell 10
model.compile(optimizer="adadelta", loss="binary_crossentropy", metrics=["accuracy"])



## === cell 11
import time

t0 = time.time()

history = model.fit(
    train_images,
    train_labels,
    validation_split=0.01,
    epochs=100,
    batch_size=32,
    callbacks=cbl,
    verbose=2,
)

print("Training time (s):", round(time.time() - t0, 2))



## === cell 12
history_df = pd.DataFrame(history.history)
print("History keys:", list(history_df.columns))

try:
    import matplotlib.pyplot as plt

    ax = history_df[["loss", "val_loss"]].plot(title="Loss")
    plt.show()

    acc_key = (
        "accuracy"
        if "accuracy" in history_df.columns
        else ("acc" if "acc" in history_df.columns else None)
    )
    val_acc_key = (
        "val_accuracy"
        if "val_accuracy" in history_df.columns
        else ("val_acc" if "val_acc" in history_df.columns else None)
    )
    if acc_key and val_acc_key:
        history_df[[acc_key, val_acc_key]].plot(title="Accuracy")
        plt.show()
except Exception as e:
    print("Plotting skipped due to:", repr(e))



## === cell 13
predicted_data = model.predict(test_images, batch_size=256, verbose=0)
print("predicted_data shape:", predicted_data.shape)



## === cell 14
sample_sub_path = os.path.join(path, "sample_submission.csv")
if not os.path.exists(sample_sub_path):
    alt = os.path.join("../input/aerial-cactus-identification", "sample_submission.csv")
    if os.path.exists(alt):
        sample_sub_path = alt
    else:
        raise FileNotFoundError(
            f"sample_submission.csv not found at {os.path.join(path,'sample_submission.csv')} or {alt}"
        )

test_data = pd.read_csv(sample_sub_path)
if len(test_data) != len(test_files):
    print(
        "Warning: sample_submission rows != number of test images:",
        len(test_data),
        len(test_files),
    )

test_data["has_cactus"] = predicted_data[:, 1].astype(float)



## === cell 15
submission_path = "submission.csv"
test_data.to_csv(submission_path, index=False)
print("Wrote:", submission_path, "shape:", test_data.shape)
print(test_data.head())

check = pd.read_csv(submission_path)
print(check.dtypes)
print(
    "has_cactus min/max:",
    float(check["has_cactus"].min()),
    float(check["has_cactus"].max()),
)
print("Rows with ~0.5 prob:", check.query("0.4 <= has_cactus <= 0.6").shape)
