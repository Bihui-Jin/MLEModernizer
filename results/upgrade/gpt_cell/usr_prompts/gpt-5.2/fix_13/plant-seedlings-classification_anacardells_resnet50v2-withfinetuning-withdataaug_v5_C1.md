# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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
protobuf==6.33.0
scikit-image==0.25.2
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
tf_keras==2.18.0

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

# 5. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

for _dirname, _sub, _files in os.walk("/kaggle/input"):
    pass



## === cell 1
import csv

csv_trainfile = "/kaggle/working/train.csv"

with open(csv_trainfile, "w") as file:
    for dirname, _, filenames in os.walk(
        "/kaggle/input/plant-seedlings-classification/train"
    ):
        for filename in filenames:
            class_name = dirname
            class_name = class_name.replace(
                "/kaggle/input/plant-seedlings-classification/train/", ""
            )
            row = dirname + "/" + filename + ";" + class_name + ";" + filename
            file.write(row + "\n")



## === cell 2
column_names = ["path", "specie", "file"]
dataFrameTrain = pd.read_csv(csv_trainfile, delimiter=";", header=None)
dataFrameTrain.columns = column_names

print(dataFrameTrain.shape)
print(dataFrameTrain.head())



## === cell 3
print(dataFrameTrain.describe())  # Verify there are no NaNs



## === cell 4
classes = dataFrameTrain["specie"].unique()
print(f"Number of classes: {len(classes)}")

datos_classes = dataFrameTrain.groupby("specie").count()
print(datos_classes)



## === cell 5
SKIP_EDA_PLOTS = True
if not SKIP_EDA_PLOTS:
    plot = datos_classes.plot.pie(y="file", figsize=(5, 5), legend=False)



## === cell 6
if not SKIP_EDA_PLOTS:
    import random
    from skimage import io
    import matplotlib.pyplot as plt

    fig = plt.figure()
    plt.figure(figsize=(15, 11))

    for i in range(12):
        plt.subplot(3, 4, i + 1)
        num = random.randint(0, len(dataFrameTrain) - 1)
        file = dataFrameTrain["path"][num]
        img = io.imread(file)
        plt.imshow(img)
        plt.xlabel(dataFrameTrain["specie"][num])
    plt.show()

    print("Image Shape: ", img.shape)
    print("Pixel value: ", img[0][0])



## === cell 7
if not SKIP_EDA_PLOTS:
    from skimage import io
    import matplotlib.pyplot as plt

    width = []
    for file_num in range(500):
        path = dataFrameTrain["path"][file_num]
        img = io.imread(path)
        width.append(img.shape[0])

    plt.hist(width, bins=50, range=(0, 1024))
    plt.xlabel("Size of images (width)")
    plt.ylabel("Frecuency")
    plt.show()



## === cell 8
import os
import sys
import subprocess

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "protobuf==4.25.3"]
)

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"

import random
import numpy as np
import tensorflow as tf

seed = 42
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Activation,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
    MaxPooling2D,
)
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt
from tensorflow.keras import layers

from tensorflow.keras.callbacks import (
    LearningRateScheduler,
    EarlyStopping,
    ModelCheckpoint,
)

from math import exp

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet_v2 import preprocess_input


## === cell 9
file = "/kaggle/working/pretrained_ResNet50V2_vFINAL"
batch_size = 32
val_split = 0.2
image_size = (256, 256)
PROYECT_FOLDER_TRAIN = "/kaggle/input/plant-seedlings-classification/train/"

train_ds_raw = tf.keras.utils.image_dataset_from_directory(
    PROYECT_FOLDER_TRAIN,
    labels="inferred",
    label_mode="categorical",
    validation_split=val_split,
    subset="training",
    seed=seed,
    image_size=image_size,
    batch_size=batch_size,
    shuffle=True,
)

val_ds_raw = tf.keras.utils.image_dataset_from_directory(
    PROYECT_FOLDER_TRAIN,
    labels="inferred",
    label_mode="categorical",
    validation_split=val_split,
    subset="validation",
    seed=seed,
    image_size=image_size,
    batch_size=batch_size,
    shuffle=True,
)

class_names = list(train_ds_raw.class_names)
num_classes = len(class_names)

_random_translate = tf.keras.layers.RandomTranslation(
    height_factor=0.2, width_factor=0.2, fill_mode="reflect", seed=seed
)
_random_rotate = tf.keras.layers.RandomRotation(
    factor=30.0 / 360.0, fill_mode="reflect", seed=seed
)
_random_zoom = tf.keras.layers.RandomZoom(
    height_factor=(-0.2, 0.2),
    width_factor=(-0.2, 0.2),
    fill_mode="reflect",
    seed=seed,
)
_random_flip = tf.keras.layers.RandomFlip(mode="horizontal_and_vertical", seed=seed)


@tf.function
def _augment_and_preprocess(x, y):
    x = tf.cast(x, tf.float32)

    x = _random_translate(x, training=True)
    x = _random_rotate(x, training=True)
    x = _random_zoom(x, training=True)
    x = _random_flip(x, training=True)

    b = tf.random.uniform([], 0.7, 1.3, seed=seed)
    x = tf.clip_by_value(x * b, 0.0, 255.0)

    x = x * 0.9

    x = preprocess_input(x)
    return x, y


@tf.function
def _preprocess_only(x, y):
    x = tf.cast(x, tf.float32)
    x = preprocess_input(x)
    return x, y


AUTOTUNE = tf.data.AUTOTUNE

train_ds = (
    train_ds_raw.cache()
    .map(_augment_and_preprocess, num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)
val_ds = (
    val_ds_raw.cache()
    .map(_preprocess_only, num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)


## === cell 10
input_shape_c = tuple((image_size[0], image_size[1], 3))
print(input_shape_c)

base_model = ResNet50V2(
    weights="imagenet", include_top=False, input_shape=input_shape_c
)  # It should have exactly 3 inputs channels, and width and height should be no smaller than 32.



## === cell 11
for layer in base_model.layers:
    if layer.name == "conv5_block1_1_conv":
        break
    layer.trainable = False

pre_trained_model = Sequential()
pre_trained_model.add(base_model)
pre_trained_model.add(layers.Flatten())

pre_trained_model.add(layers.Dense(512, activation="relu"))

pre_trained_model.add(Dropout(0.5))
pre_trained_model.add(BatchNormalization())

pre_trained_model.add(layers.Dense(12, activation="softmax"))
pre_trained_model.summary()



## === cell 12
epochs = 200

print("[INFO]: Compiling the model...")

val_num_classes = len(val_ds_raw.class_names)
assert (
    num_classes == val_num_classes
), f"Train/val datasets disagree on num_classes: train={num_classes}, val={val_num_classes}"

if hasattr(pre_trained_model, "layers") and len(pre_trained_model.layers) > 0:
    last_layer = pre_trained_model.layers[-1]
    if getattr(last_layer, "units", None) != num_classes:
        pre_trained_model.pop()
        pre_trained_model.add(layers.Dense(num_classes, activation="softmax"))

pre_trained_model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=1e-3),
    metrics=["accuracy"],
)


def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * exp(-0.1)


annealer = LearningRateScheduler(scheduler)

earlystop = EarlyStopping(
    patience=5,
    monitor="val_loss",
)

modelsave = ModelCheckpoint(
    filepath=file + ".h5",
    save_best_only=True,
    verbose=1,
)

ckpt_path = file + ".h5"
if os.path.exists(ckpt_path):
    print(
        f"[INFO]: Found existing checkpoint at {ckpt_path}. Loading weights and skipping training..."
    )
    pre_trained_model.load_weights(ckpt_path)
    H_pre = type(
        "History",
        (),
        {"history": {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}},
    )()
else:
    print("[INFO]: Entrenando la red...")
    H_pre = pre_trained_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[annealer, earlystop, modelsave],
        verbose=2,
    )
    if os.path.exists(ckpt_path):
        pre_trained_model.load_weights(ckpt_path)



## === cell 13
print("[INFO]: Evaluating the model...")

if (
    getattr(H_pre, "history", None)
    and len(H_pre.history.get("loss", [])) > 0
    and not SKIP_EDA_PLOTS
):
    num_epochs = len(H_pre.history["loss"])
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, num_epochs), H_pre.history["loss"], label="train_loss")
    plt.plot(np.arange(0, num_epochs), H_pre.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, num_epochs), H_pre.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, num_epochs), H_pre.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()



## === cell 14
csv_testfile = "/kaggle/working/test.csv"

with open(csv_testfile, "w") as file:
    for dirname, _, filenames in os.walk(
        "/kaggle/input/plant-seedlings-classification/test"
    ):
        for filename in filenames:
            row = dirname + "/" + filename + ";" + filename
            file.write(row + "\n")



## === cell 15
column_names = ["path", "file"]
dataFrameTest = pd.read_csv(csv_testfile, delimiter=";", header=None)
dataFrameTest.columns = column_names

print(dataFrameTest.shape)
print(dataFrameTest.head())
print(dataFrameTest["path"][0])



## === cell 16
batch_size = 32
image_size = (256, 256)
PROYECT_FOLDER_TEST = "/kaggle/input/plant-seedlings-classification/"

test_ds_raw = tf.keras.utils.image_dataset_from_directory(
    PROYECT_FOLDER_TEST,
    labels=None,
    image_size=image_size,
    batch_size=batch_size,
    shuffle=False,
    classes=["test"],
)


@tf.function
def _preprocess_only_x(x):
    x = tf.cast(x, tf.float32)
    x = preprocess_input(x)
    return x


test_ds = test_ds_raw.map(_preprocess_only_x, num_parallel_calls=AUTOTUNE).prefetch(
    AUTOTUNE
)



## === cell 17
predicted_class = pre_trained_model.predict(test_ds, verbose=0)



## === cell 18
print(len(predicted_class[0]))
print(len(predicted_class))
predicted_class_number = np.argmax(predicted_class, axis=1)
print(len(predicted_class_number))

classes = np.array(class_names)
print(classes)



## === cell 19
csv_resultsfile = "/kaggle/working/submission.csv"

file_series = dataFrameTest["file"].astype(str)
species_series = classes[predicted_class_number]

submission = pd.DataFrame({"file": file_series.values, "species": species_series})
submission.to_csv(csv_resultsfile, index=False)



## === cell 20
dataFrameResults = pd.read_csv(csv_resultsfile, delimiter=",")
print(dataFrameResults.shape)
print(dataFrameResults.head())



## === cell 21
pass



## === cell 22
pass

## --- ERROR in outputing the csv:
Invalid submission: Submission must have 'file' column
