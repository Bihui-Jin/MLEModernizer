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

0.71599

# 6. Current score

0.11712

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.11712) has done: 'I remove the notebook-only `%matplotlib inline` (it breaks in a .py/script run) and fix the TensorFlow/Protobuf import crash by forcing the Python protobuf implementation before importing TensorFlow. I also fix missing imports (`glob`) and update the Keras 3 `ImageDataGenerator` import path while keeping the same augmentation logic. Finally, I correct constants/paths that were inconsistent (e.g., `N_CLASSES=42` should be 12) and ensure the submission is aligned to `sample_submission.csv` order and written as a valid `.csv` with columns `file,species`.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import pandas as pd

import tensorflow as tf

from tqdm import tqdm
import cv2
import imageio
import random
from glob import glob

import matplotlib.pyplot as plt
import seaborn as sns


SEED = 7
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TF:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
images_path = "/kaggle/input/plant-seedlings-classification/train/*/*.png"
images = glob(images_path)

img_size = 128
train_images = []
train_labels = []
for i in images:
    img = cv2.imread(i)
    if img is None:
        continue
    train_images.append(cv2.resize(img, (img_size, img_size)))
    train_labels.append(i.split("/")[-2])

train_X = np.asarray(train_images)
train_Y = pd.DataFrame(train_labels)

print("Train images:", train_X.shape, "Train labels:", train_Y.shape)



## === cell 2
train_Y.rename(columns={0: "species"}, inplace=True)
_, train_count = np.unique(train_Y, return_counts=True)
df = pd.DataFrame(data=train_count)
a = train_Y["species"].unique().tolist()
a.sort()
df["Index"] = a
df.columns = ["Train", "Name"]
df



## === cell 3
plt.figure(figsize=(10, 5))
chart = sns.countplot(data=train_Y, x="species")
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)



## === cell 4
TRAIN_DIR = "/kaggle/input/plant-seedlings-classification/train"
CLASSES = [folder[len(TRAIN_DIR) + 1 :] for folder in glob(TRAIN_DIR + "/*")]
CLASSES.sort()

TARGET_SIZE = (64, 64)
TARGET_DIMS = (64, 64, 3)  # add channel for RGB

N_CLASSES = len(CLASSES)

VALIDATION_SPLIT = 0.1
BATCH_SIZE = 64

print("N_CLASSES:", N_CLASSES)
print("CLASSES:", CLASSES)




## === cell 5
def plot_one_sample_of_each(base_path):
    cols = 4
    rows = int(np.ceil(len(CLASSES) / cols))
    fig = plt.figure(figsize=(16, 20))

    for i in range(len(CLASSES)):
        cls = CLASSES[i]
        img_path = base_path + "/" + cls + "/*"
        path_contents = glob(img_path)
        if len(path_contents) == 0:
            continue

        imgs = random.sample(path_contents, 1)

        sp = plt.subplot(rows, cols, i + 1)
        plt.imshow(imageio.imread(imgs[0]))
        plt.title(cls)
        sp.axis("off")

    plt.show()




## === cell 6
plot_one_sample_of_each(TRAIN_DIR)



## === cell 7
from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()
y = lb.fit_transform(train_Y.species)

if list(lb.classes_) != CLASSES:
    cls_to_idx = {c: i for i, c in enumerate(CLASSES)}
    y_fixed = np.zeros((len(train_Y), len(CLASSES)), dtype=np.float32)
    for i, c in enumerate(train_Y.species.values):
        y_fixed[i, cls_to_idx[c]] = 1.0
    train_label = y_fixed
else:
    train_label = np.array(y, dtype=np.float32)

print("Label matrix:", train_label.shape)



## === cell 8
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    train_X,
    train_label,
    test_size=0.3,
    random_state=SEED,
    stratify=train_Y["species"].values,
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
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D



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
        Dense(12, activation="softmax"),
    ]
)



## === cell 13
model0.summary()



## === cell 14
from keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint = ModelCheckpoint(
    "plant_classifier.keras",
    save_best_only=True,
    monitor="val_accuracy",
    mode="max",
    verbose=1,
)

early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=False
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
    datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True),
    epochs=epochs,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, checkpoint],
    verbose=1,
)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2508001130.py in <cell line: 0>()
      1 # Keep core training loop; use datagen for augmentation as originally intended.
----> 2 history = model0.fit(
      3     datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True),
      4     epochs=epochs,
      5     validation_data=(X_test, y_test),

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/nn.py in categorical_crossentropy(target, output, from_logits, axis)
    658     for e1, e2 in zip(target.shape, output.shape):
    659         if e1 is not None and e2 is not None and e1 != e2:
--> 660             raise ValueError(
    661                 "Arguments `target` and `output` must have the same shape. "
    662                 "Received: "

ValueError: Arguments `target` and `output` must have the same shape. Received: target.shape=(None, 13), output.shape=(None, 12)

## === cell 17
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper left")
plt.show()



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1149848261.py in <cell line: 0>()
----> 1 plt.plot(history.history["accuracy"])
      2 plt.plot(history.history["val_accuracy"])
      3 plt.title("model accuracy")
      4 plt.ylabel("accuracy")
      5 plt.xlabel("epoch")

NameError: name 'history' is not defined

## === cell 18
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper left")
plt.show()



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3377530796.py in <cell line: 0>()
----> 1 plt.plot(history.history["loss"])
      2 plt.plot(history.history["val_loss"])
      3 plt.title("model loss")
      4 plt.ylabel("loss")
      5 plt.xlabel("epoch")

NameError: name 'history' is not defined

## === cell 19
loss, acc = model0.evaluate(X_test, y_test, verbose=0)
loss1, acc1 = model0.evaluate(X_train, y_train, verbose=0)
print("Test loss:", loss, "   Test accuracy:", acc)
print("Train loss:", loss1, "   Train accuracy:", acc1)



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2656261412.py in <cell line: 0>()
----> 1 loss, acc = model0.evaluate(X_test, y_test, verbose=0)
      2 loss1, acc1 = model0.evaluate(X_train, y_train, verbose=0)
      3 print("Test loss:", loss, "   Test accuracy:", acc)
      4 print("Train loss:", loss1, "   Train accuracy:", acc1)
      5 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/nn.py in categorical_crossentropy(target, output, from_logits, axis)
    658     for e1, e2 in zip(target.shape, output.shape):
    659         if e1 is not None and e2 is not None and e1 != e2:
--> 660             raise ValueError(
    661                 "Arguments `target` and `output` must have the same shape. "
    662                 "Received: "

ValueError: Arguments `target` and `output` must have the same shape. Received: target.shape=(None, 13), output.shape=(None, 12)

## === cell 20
predictions = model0.predict(X_test, verbose=0)




## === cell 21
def plot_image(i, predictions_array, true_label, img):
    true_label_i, img_i = np.argmax(true_label[i]), img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img_i)

    predicted_label = np.argmax(predictions_array)
    color = "blue" if predicted_label == true_label_i else "red"

    plt.xlabel(
        "{} {:2.0f}% \n({})".format(
            np.array(df.Name)[predicted_label],
            100 * np.max(predictions_array),
            np.array(df.Name)[true_label_i],
        ),
        color=color,
    )




## === cell 22
fig = plt.figure(figsize=(16, 20))
rows, cols = 3, 4
nshow = min(cols * rows, len(X_test))
for i in range(0, nshow):
    fig.add_subplot(rows, cols, i + 1)
    plot_image(i, predictions[i], y_test, X_test)
    plt.subplots_adjust(hspace=-0.5)
plt.show()



## === cell 23
test_images_path = "/kaggle/input/plant-seedlings-classification/test/*.png"
test_images = glob(test_images_path)

test_images_arr = []
test_files = []

for img_path in test_images:
    img = cv2.imread(img_path)
    if img is None:
        continue
    test_images_arr.append(cv2.resize(img, (128, 128)))
    test_files.append(os.path.basename(img_path))

test_X = np.asarray(test_images_arr).astype("float32") / 255.0

print("Test:", test_X.shape, "files:", len(test_files))



## === cell 24
predictions = model0.predict(test_X, verbose=0)
preds = np.argmax(predictions, axis=1)
pred_str = np.array(CLASSES)[preds]

final_predictions = pd.DataFrame({"file": test_files, "species": pred_str})

sample_path = "/kaggle/input/plant-seedlings-classification/sample_submission.csv"
sample = pd.read_csv(sample_path)
final_predictions = sample[["file"]].merge(final_predictions, on="file", how="left")

if final_predictions["species"].isna().any():
    fallback = pd.Series(train_Y["species"].values).value_counts().idxmax()
    final_predictions["species"] = final_predictions["species"].fillna(fallback)

final_predictions.head()



## === cell 25
final_predictions.to_csv("Plant-Seedlings-Classification.csv", index=False)
print(
    "Wrote submission:",
    "Plant-Seedlings-Classification.csv",
    "rows:",
    len(final_predictions),
)



## === cell 26
fig = plt.figure(figsize=(16, 20))
rows, cols = 3, 4
nshow = min(cols * rows, len(test_X))
for i in range(0, nshow):
    fig.add_subplot(rows, cols, i + 1)
    plt.title(final_predictions.species.iloc[i])
    plt.imshow(test_X[i])
    plt.axis("off")
    plt.subplots_adjust(hspace=-0.5)

plt.show()
