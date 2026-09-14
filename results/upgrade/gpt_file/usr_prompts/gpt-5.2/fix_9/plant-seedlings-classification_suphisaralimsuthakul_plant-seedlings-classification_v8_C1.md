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

3.11

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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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
tf_keras==2.18.0
tqdm==4.67.1

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

# 5. Target score

0.61209

# 6. Current score

0.7027

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.07357) has done: 'I fix the notebook-breaking import/runtime issues without changing your core CNN architecture or training loop. Specifically, I remove the IPython-only `%matplotlib inline`, fix the TensorFlow/Keras/protobuf crash by avoiding standalone `keras` (Keras 3) and using `tf.keras` consistently, and restore missing imports (`glob`). I also fix wrong constants (your `N_CLASSES` must be 12, not 42) and ensure labels/classes are aligned deterministically so the predicted class index maps back to the correct species name. Finally, I ensure test files are predicted in the same order as `sample_submission.csv` and write a valid `.csv` submission with required columns.'
- What this solution (achieved 0.71471) has done: 'I fix the TensorFlow/protobuf crash by pinning protobuf’s Python implementation via environment variables before importing TensorFlow (this is a common Kaggle TF2.18+protobuf 6 issue). Then I fix the label/model shape mismatch causing `(None, 12)` vs `(None, 13)` by ensuring the one-hot encoder always produces exactly `N_CLASSES` columns, aligned to `CLASSES`, and by asserting the model’s output dimension matches that. Finally, I ensure the test prediction→label mapping uses the same `CLASSES` ordering used during training and that the submission is written as a valid `.csv` with the required `file,species` columns in the same order as `sample_submission.csv`—these are correctness fixes and should substantially improve score from the current near-random level.'
- What this solution (achieved 0.72072) has done: 'I fix the TensorFlow/protobuf crash by also forcing the pure-Python protobuf backend at import time (not just via env vars), which resolves the `MessageFactory.GetPrototype` error in TF2.18+ with protobuf 6. Then I fix the “13 classes” issue by filtering out non-class folders (notably the nested `train/train` directory) when building `CLASSES`, so `N_CLASSES` becomes 12 and aligns with the labels. Finally, because your current score (0.71471) is already above the target (0.61209), I keep the model/training logic unchanged and only make these correctness/stability fixes; this should keep performance in the same ballpark while producing a valid submission CSV.'
- What this solution (achieved 0.7012) has done: 'I fix the protobuf/TensorFlow import crash that prevents the notebook from running by forcing the pure-Python protobuf backend before importing TensorFlow and by patching `MessageFactory.GetPrototype` to call `GetMessageClass` when needed (this is the exact root cause of your traceback on TF2.18 + protobuf 6). I keep your CNN, augmentation, training loop, and submission logic unchanged so the score should remain in the same general range (and not be intentionally optimized further since you’re already above the target band). I also make the matplotlib/seaborn imports non-fatal in case they’re unavailable in some Kaggle runtimes, without changing any training/inference behavior. The output submission file name and format remain the same and a valid `.csv` be written.'
- What this solution (achieved 0.72523) has done: 'I fix the runtime crash in the protobuf compatibility patch: the current code calls `hasattr(MessageFactory, "GetPrototype")` on the class, but in protobuf 6 this method may be missing on the instance, which triggers the exact `AttributeError` you saw; I patch the instance method safely instead. I keep the CNN, training loop, augmentation, and submission logic unchanged to avoid intentionally improving further (your current score is already above the target band). I also make the TensorFlow import/protobuf backend forcing more robust in TF2.18/protobuf6 by applying the patch before importing TensorFlow and avoiding brittle attribute checks. The result run end-to-end and still write a valid `Plant-Seedlings-Classification.csv` submission with the required `file,species` columns.'
- What this solution (achieved 0.71922) has done: 'I fix the protobuf/TensorFlow crash happening before any training by making the `MessageFactory` patch instance-safe: in protobuf 6 the instance may not have `GetPrototype`, and directly checking/setting it can raise `AttributeError`. The rest of your pipeline (data loading, CNN architecture, training loop, and submission creation) remain unchanged to avoid intentionally optimizing further since your current score is already above the target band. I also keep the environment forcing for the pure-Python protobuf backend, but apply the patch in a way that cannot fail at import time. The script then run end-to-end and still write `Plant-Seedlings-Classification.csv` with the required `file,species` columns.'
- What this solution (achieved 0.74324) has done: 'I fix the protobuf/TensorFlow crash in the first cell by removing the brittle `MessageFactory.GetPrototype` monkeypatch that can still raise `AttributeError` on some protobuf 6 builds, while keeping the same “force python protobuf backend” behavior. This change is runtime/stability-only and does not alter your CNN, training loop, augmentation, or submission formatting, so it should keep performance in the same general range (you’re already above the target band). I also keep all paths and the required `file,species` submission schema unchanged and ensure the script runs end-to-end to write `Plant-Seedlings-Classification.csv`.'
- What this solution (achieved 0.7027) has done: 'I fix the protobuf/TensorFlow import crash by forcing the pure-Python protobuf backend *before* importing TensorFlow and by defensively removing any brittle `MessageFactory.GetPrototype` usage that can trigger `AttributeError` on protobuf 6. I keep your CNN architecture, training loop, augmentation, and prediction logic unchanged to avoid intentionally optimizing further (your current score is already above the target). I also renumber the cells starting at 1 (your provided script starts at cell 0) so the notebook-style runner can execute cleanly. The script still write a valid `Plant-Seedlings-Classification.csv` with the required `file,species` columns and test-file ordering matching `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

try:
    from google.protobuf.internal import api_implementation

    if hasattr(api_implementation, "_implementation_type"):
        api_implementation._implementation_type = "python"
except Exception:
    pass

import random
from glob import glob

import numpy as np
import pandas as pd

import tensorflow as tf
import cv2
import imageio

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    plt = None
    sns = None

from tqdm import tqdm

SEED = 7
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow:", tf.__version__)
print("OpenCV:", cv2.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
TRAIN_GLOB = "/kaggle/input/plant-seedlings-classification/train/*/*.png"
TEST_DIR = "/kaggle/input/plant-seedlings-classification/test"
SAMPLE_SUB_PATH = "/kaggle/input/plant-seedlings-classification/sample_submission.csv"

img_size = 128

images = sorted(glob(TRAIN_GLOB))
assert len(images) > 0, f"No training images found with glob: {TRAIN_GLOB}"

train_images = []
train_labels = []
for p in tqdm(images, desc="Loading train images"):
    im = cv2.imread(p)
    if im is None:
        raise ValueError(f"Failed to read image: {p}")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (img_size, img_size))
    train_images.append(im)
    train_labels.append(os.path.basename(os.path.dirname(p)))

train_X = np.asarray(train_images, dtype=np.uint8)
train_Y = pd.DataFrame(train_labels, columns=["species"])

print("train_X:", train_X.shape, train_X.dtype)
print("train_Y:", train_Y.shape)
print("Unique species:", train_Y["species"].nunique())



## === cell 2
species_names = sorted(train_Y["species"].unique().tolist())
train_count = train_Y["species"].value_counts().reindex(species_names).values

df = pd.DataFrame({"Train": train_count, "Name": species_names})
df



## === cell 3
if plt is not None and sns is not None:
    plt.figure(figsize=(10, 5))
    chart = sns.countplot(data=train_Y, x="species", order=species_names)
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
    plt.tight_layout()
    plt.show()



## === cell 4
TRAIN_DIR = "/kaggle/input/plant-seedlings-classification/train"

CLASSES = []
for folder in sorted(glob(TRAIN_DIR + "/*")):
    if not os.path.isdir(folder):
        continue
    pngs = glob(os.path.join(folder, "*.png"))
    if len(pngs) == 0:
        continue
    CLASSES.append(os.path.basename(folder))

TARGET_SIZE = (64, 64)
TARGET_DIMS = (64, 64, 3)  # add channel for RGB
N_CLASSES = len(CLASSES)  # should be 12
VALIDATION_SPLIT = 0.1
BATCH_SIZE = 64

print("Detected classes:", CLASSES)
print("N_CLASSES:", N_CLASSES)

assert N_CLASSES == 12, f"Expected 12 classes for this competition, got {N_CLASSES}"




## === cell 5
def plot_one_sample_of_each(base_path):
    if plt is None:
        return
    cols = 4
    rows = int(np.ceil(len(CLASSES) / cols))
    plt.figure(figsize=(16, 20))

    for i, cls in enumerate(CLASSES):
        img_path = os.path.join(base_path, cls, "*")
        path_contents = glob(img_path)
        if not path_contents:
            continue
        img_file = random.choice(path_contents)

        sp = plt.subplot(rows, cols, i + 1)
        plt.imshow(imageio.imread(img_file))
        plt.title(cls)
        sp.axis("off")

    plt.tight_layout()
    plt.show()




## === cell 6
plot_one_sample_of_each(TRAIN_DIR)



## === cell 7
class_to_idx = {c: i for i, c in enumerate(CLASSES)}
y_idx = train_Y["species"].map(class_to_idx)

assert (
    y_idx.notna().all()
), "Found unmapped labels; CLASSES and train_Y species mismatch."

y_idx = y_idx.astype(int).to_numpy()

train_label = tf.keras.utils.to_categorical(y_idx, num_classes=N_CLASSES).astype(
    np.float32
)

print("train_label:", train_label.shape, train_label.dtype)
assert train_label.shape[1] == N_CLASSES
assert set(train_Y["species"].unique()) == set(CLASSES)

df = pd.DataFrame(
    {
        "Train": train_Y["species"].value_counts().reindex(CLASSES).values,
        "Name": CLASSES,
    }
)



## === cell 8
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    train_X,
    train_label,
    test_size=0.3,
    random_state=SEED,
    stratify=y_idx,  # stratify by class index to match one-hot labels
)

print("X_train:", X_train.shape, "X_test:", X_test.shape)



## === cell 9
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0



## === cell 10
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=180,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
)
datagen.fit(X_train)



## === cell 11
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping



## === cell 12
model0 = Sequential(
    [
        Conv2D(
            32,
            kernel_size=(3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            input_shape=(128, 128, 3),
        ),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),
        Conv2D(128, (3, 3), activation="relu"),
        Dropout(0.40),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(N_CLASSES, activation="softmax"),
    ]
)

assert model0.output_shape[-1] == N_CLASSES, (model0.output_shape, N_CLASSES)



## === cell 13
model0.summary()



## === cell 14
checkpoint = ModelCheckpoint(
    "plant_classifier.h5",
    save_best_only=True,
    monitor="val_accuracy",
    mode="max",
    verbose=1,
)

early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)



## === cell 15
model0.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

batch_size = 32
epochs = 30



## === cell 16
history = model0.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, checkpoint],
    verbose=1,
)



## === cell 17
if plt is not None:
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    plt.show()



## === cell 18
if plt is not None:
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    plt.show()



## === cell 19
loss, acc = model0.evaluate(X_test, y_test, verbose=0)
loss1, acc1 = model0.evaluate(X_train, y_train, verbose=0)
print("Test loss:", loss, "   Test accuracy:", acc)
print("Train loss:", loss1, "   Train accuracy:", acc1)



## === cell 20
predictions = model0.predict(X_test, verbose=0)




## === cell 21
def plot_image(i, predictions_array, true_label, img):
    if plt is None:
        return
    true_label_i, img_i = int(np.argmax(true_label[i])), img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img_i)

    predicted_label = int(np.argmax(predictions_array))
    color = "blue" if predicted_label == true_label_i else "red"

    plt.xlabel(
        "{} {:2.0f}% \n({})".format(
            CLASSES[predicted_label],
            100 * float(np.max(predictions_array)),
            CLASSES[true_label_i],
        ),
        color=color,
    )




## === cell 22
if plt is not None:
    fig = plt.figure(figsize=(16, 20))
    rows, cols = 3, 4
    for i in range(0, min(cols * rows, len(X_test))):
        fig.add_subplot(rows, cols, i + 1)
        plot_image(i, predictions[i], y_test, X_test)
        plt.subplots_adjust(hspace=-0.5)
    plt.show()



## === cell 23
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_files = sample_sub["file"].tolist()

test_images_arr = []
for fn in tqdm(test_files, desc="Loading test images"):
    p = os.path.join(TEST_DIR, fn)
    im = cv2.imread(p)
    if im is None:
        raise ValueError(f"Failed to read test image: {p}")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (img_size, img_size))
    test_images_arr.append(im)

test_X = np.asarray(test_images_arr, dtype=np.float32) / 255.0
print("test_X:", test_X.shape, test_X.dtype)



## === cell 24
predictions_test = model0.predict(test_X, verbose=0)
preds = np.argmax(predictions_test, axis=1)
pred_str = np.array(CLASSES, dtype=object)[preds]

final_predictions = pd.DataFrame({"file": test_files, "species": pred_str})
final_predictions.head()



## === cell 25
out_path = "Plant-Seedlings-Classification.csv"
final_predictions.to_csv(out_path, index=False)
print(
    "Wrote:",
    out_path,
    "rows:",
    len(final_predictions),
    "cols:",
    list(final_predictions.columns),
)

assert list(final_predictions.columns) == ["file", "species"]
assert len(final_predictions) == len(sample_sub)
assert final_predictions["file"].tolist() == sample_sub["file"].tolist()



## === cell 26
if plt is not None:
    fig = plt.figure(figsize=(16, 20))
    rows, cols = 3, 4
    for i in range(0, min(cols * rows, len(test_X))):
        fig.add_subplot(rows, cols, i + 1)
        plt.title(final_predictions["species"].iloc[i])
        plt.imshow(test_X[i])
        plt.axis("off")
        plt.subplots_adjust(hspace=-0.5)

    plt.show()
