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
import numpy as np
import pandas as pd
import os

PRINT_INPUT_TREE = False

if PRINT_INPUT_TREE:
    for dirname, _, filenames in os.walk("/kaggle/input"):
        for filename in filenames:
            print(os.path.join(dirname, filename))



## === cell 1
import csv

csv_trainfile = "/kaggle/working/train.csv"

if not os.path.exists(csv_trainfile):
    with open(csv_trainfile, "w") as file:
        for dirname, _, filenames in os.walk(
            "/kaggle/input/plant-seedlings-classification/train"
        ):
            class_name = dirname.replace(
                "/kaggle/input/plant-seedlings-classification/train/", ""
            )
            for filename in sorted(filenames):
                row = dirname + "/" + filename + ";" + class_name + ";" + filename
                file.write(row + "\n")



## === cell 2
column_names = ["path", "specie", "file"]
dataFrameTrain = pd.read_csv(csv_trainfile, delimiter=";", header=None)
dataFrameTrain.columns = column_names

print(dataFrameTrain.shape)
print(dataFrameTrain.head())



## === cell 3
DO_EDA = False

if DO_EDA:
    print(dataFrameTrain.describe())  # Verify there are no NaNs



## === cell 4
classes = dataFrameTrain["specie"].unique()
print(f"Number of classes: {len(classes)}")

datos_classes = dataFrameTrain.groupby("specie").count()
print(datos_classes)



## === cell 5
if DO_EDA:
    plot = datos_classes.plot.pie(y="file", figsize=(5, 5), legend=False)



## === cell 6
if DO_EDA:
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
if DO_EDA:
    from skimage import io
    import matplotlib.pyplot as plt

    width = []
    for file_num in range(min(500, len(dataFrameTrain))):
        path = dataFrameTrain["path"][file_num]
        img = io.imread(path)
        width.append(img.shape[0])

    plt.hist(width, bins=50, range=(0, 1024))
    plt.xlabel("Size of images (width)")
    plt.ylabel("Frecuency")
    plt.show()



## === cell 8
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"

import google.protobuf  # noqa: F401
from google.protobuf import message_factory as _message_factory

if hasattr(_message_factory, "MessageFactory") and not hasattr(
    _message_factory.MessageFactory, "GetPrototype"
):

    def _GetPrototype(self, descriptor):
        return self.GetMessageClass(descriptor)

    _message_factory.MessageFactory.GetPrototype = _GetPrototype

import numpy as np
import tensorflow as tf
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
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from math import exp

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet_v2 import preprocess_input

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass


## === cell 9
file = "/kaggle/working/pretrained_ResNet50V2_vFINAL"
batch_size = 32
val_split = 0.2
image_size = (256, 256)
PROYECT_FOLDER_TRAIN = "/kaggle/input/plant-seedlings-classification/train/"

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # Standandardize for Resnet
    rotation_range=30,  # Int. Degree range for random rotations.
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.7, 1.3],
    rescale=0.9,
    vertical_flip=True,
    horizontal_flip=True,
    validation_split=val_split,
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=val_split,
)

train_generator = train_datagen.flow_from_directory(
    PROYECT_FOLDER_TRAIN,  # It should contain one subdirectory per class.
    target_size=image_size,  # Dims to which all images found will be resized.
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",  # Type of label arrays that are returned
    subset="training",
    seed=seed,
    shuffle=True,
)

val_generator = val_datagen.flow_from_directory(
    PROYECT_FOLDER_TRAIN,
    target_size=image_size,
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
    seed=seed,
    shuffle=True,
)

FIT_WORKERS = max(1, (os.cpu_count() or 2) - 1)
FIT_USE_MULTIPROCESSING = False
FIT_MAX_QUEUE_SIZE = 64



## === cell 10
input_shape_c = tuple((image_size[0], image_size[1], 3))
print(input_shape_c)

base_model = ResNet50V2(
    weights="imagenet", include_top=False, input_shape=input_shape_c
)



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

num_classes = int(
    getattr(
        train_generator,
        "num_classes",
        len(getattr(train_generator, "class_indices", {})),
    )
)
if pre_trained_model.layers and isinstance(pre_trained_model.layers[-1], layers.Dense):
    if pre_trained_model.layers[-1].units != num_classes:
        pre_trained_model.pop()
        pre_trained_model.add(layers.Dense(num_classes, activation="softmax"))

print("[INFO]: Compiling the model...")

pre_trained_model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=1e-3),
    metrics=["accuracy"],
    steps_per_execution=10,
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

modelsave = ModelCheckpoint(filepath=file + ".h5", save_best_only=True, verbose=1)

print("[INFO]: Entrenando la red...")
H_pre = pre_trained_model.fit(
    train_generator,
    validation_data=val_generator,
    steps_per_epoch=train_generator.n // train_generator.batch_size,
    validation_steps=val_generator.n // val_generator.batch_size,
    epochs=epochs,
    callbacks=[annealer, earlystop, modelsave],
    verbose=1,
)


## === cell 13
if DO_EDA:
    print("[INFO]: Evaluating the model...")
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

if not os.path.exists(csv_testfile):
    with open(csv_testfile, "w") as file:
        for dirname, _, filenames in os.walk(
            "/kaggle/input/plant-seedlings-classification/test"
        ):
            for filename in sorted(filenames):
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

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_directory(
    PROYECT_FOLDER_TEST,
    target_size=image_size,
    color_mode="rgb",
    batch_size=batch_size,
    classes=["test"],
    shuffle=False,
)



## === cell 17
list_of_files = test_generator.filenames
print(list_of_files[0])



## === cell 18
best_path = file + ".h5"
if os.path.exists(best_path):
    pre_trained_model.load_weights(best_path)

predicted_class = pre_trained_model.predict(
    test_generator,
    workers=FIT_WORKERS,
    use_multiprocessing=FIT_USE_MULTIPROCESSING,
    max_queue_size=FIT_MAX_QUEUE_SIZE,
    verbose=1,
)



## === cell 19
predicted_class_number = np.argmax(predicted_class, axis=1)

inv_class_indices = {v: k for k, v in train_generator.class_indices.items()}
classes_by_index = [inv_class_indices[i] for i in range(len(inv_class_indices))]

print(classes_by_index[:12])
print(predicted_class_number[:20])



## === cell 20
submission_path = "/kaggle/working/submission.csv"
sample_path = "/kaggle/input/plant-seedlings-classification/sample_submission.csv"

file_names = [f.replace("test/", "") for f in list_of_files]
species_pred = [classes_by_index[i] for i in predicted_class_number]

pred_df = pd.DataFrame({"file": file_names, "species": species_pred})

sample_df = pd.read_csv(sample_path)
submission_df = sample_df[["file"]].merge(pred_df, on="file", how="left")

if submission_df["species"].isna().any():
    submission_df["species"] = submission_df["species"].fillna(classes_by_index[0])

submission_df.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(submission_df.shape)
print(submission_df.head())

## --- ERROR in outputing the csv:
Invalid submission: Submission must have 'file' column
