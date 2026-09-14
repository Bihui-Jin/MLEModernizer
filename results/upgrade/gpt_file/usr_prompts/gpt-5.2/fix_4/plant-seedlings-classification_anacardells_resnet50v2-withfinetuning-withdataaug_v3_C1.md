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

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))



## === cell 1
import csv

BASE = "/kaggle/input/plant-seedlings-classification"
TRAIN_DIR = os.path.join(BASE, "train")
TEST_DIR = os.path.join(BASE, "test")

assert os.path.isdir(TRAIN_DIR), f"TRAIN_DIR not found: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"TEST_DIR not found: {TEST_DIR}"

csv_trainfile = "/kaggle/working/train.csv"

with open(csv_trainfile, "w") as f:
    for dirname, _, filenames in os.walk(TRAIN_DIR):
        for filename in filenames:
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            class_name = dirname.replace(TRAIN_DIR + "/", "")
            row = dirname + "/" + filename + ";" + class_name + ";" + filename
            f.write(row + "\n")

print("Wrote:", csv_trainfile, "size(bytes)=", os.path.getsize(csv_trainfile))



## === cell 2
column_names = ["path", "specie", "file"]
dataFrameTrain = pd.read_csv(csv_trainfile, delimiter=";", header=None)
dataFrameTrain.columns = column_names

print(dataFrameTrain.shape)
print(dataFrameTrain.head())



## === cell 3
print(dataFrameTrain.describe(include="all"))  # Verify there are no NaNs



## === cell 4
classes = dataFrameTrain["specie"].unique()
print(f"Number of classes: {len(classes)}")

datos_classes = dataFrameTrain.groupby("specie").count()
print(datos_classes)



## === cell 5
import matplotlib.pyplot as plt

try:
    plot = datos_classes.plot.pie(y="file", figsize=(5, 5), legend=False)
    plt.show()
except Exception as e:
    print("Skipping pie plot due to:", repr(e))



## === cell 6
import random
from skimage import io

try:
    fig = plt.figure()
    plt.figure(figsize=(15, 11))

    n = len(dataFrameTrain)
    for i in range(12):
        plt.subplot(3, 4, i + 1)
        num = random.randint(0, n - 1)
        file_path = dataFrameTrain["path"].iloc[num]
        img = io.imread(file_path)
        plt.imshow(img)
        plt.xlabel(dataFrameTrain["specie"].iloc[num])
    plt.show()

    print("Image Shape: ", img.shape)
    print("Pixel value: ", img[0][0])
except Exception as e:
    print("Skipping sample image plot due to:", repr(e))



## === cell 7
try:
    width = []
    upto = min(500, len(dataFrameTrain))
    for file_num in range(upto):
        path = dataFrameTrain["path"].iloc[file_num]
        img = io.imread(path)
        width.append(img.shape[0])

    plt.hist(width, bins=50, range=(0, 1024))
    plt.xlabel("Size of images (width)")
    plt.ylabel("Frecuency")
    plt.show()
except Exception as e:
    print("Skipping histogram due to:", repr(e))



## === cell 8
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import tensorflow as tf
import tf_keras as keras
from tf_keras import layers
from tf_keras.layers import Dropout, BatchNormalization
from tf_keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from tf_keras.models import Sequential
from tf_keras.optimizers import Adam
from tf_keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from tf_keras.preprocessing.image import ImageDataGenerator
from math import exp

print("TensorFlow:", tf.__version__)
print("tf_keras:", keras.__version__)



## === cell 9
file = "/kaggle/working/pretrained_ResNet50V2_vFINAL"
batch_size = 32
seed = 42
val_split = 0.2
image_size = (256, 256)
PROYECT_FOLDER_TRAIN = TRAIN_DIR

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # Standardize for ResNetV2
    rotation_range=30,
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
    PROYECT_FOLDER_TRAIN,
    target_size=image_size,
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
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

idx_to_class = {v: k for k, v in train_generator.class_indices.items()}
print("Classes (generator order):", [idx_to_class[i] for i in range(len(idx_to_class))])
print("num_classes from generator:", len(train_generator.class_indices))



## === cell 10
input_shape_c = (image_size[0], image_size[1], 3)
print(input_shape_c)

base_model = ResNet50V2(
    weights="imagenet", include_top=False, input_shape=input_shape_c
)



## === cell 11
num_classes = len(train_generator.class_indices)

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
pre_trained_model.add(layers.Dense(num_classes, activation="softmax"))
pre_trained_model.summary()



## === cell 12
epochs = 200

print("[INFO]: Compiling the model...")
pre_trained_model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=1e-3),
    metrics=["accuracy"],
)


def scheduler(epoch, lr):
    if epoch < 5:
        return lr
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

print("[INFO]: Entrenando la red...")
H_pre = pre_trained_model.fit(
    train_generator,
    validation_data=val_generator,
    steps_per_epoch=train_generator.n // train_generator.batch_size,
    validation_steps=val_generator.n // val_generator.batch_size,
    epochs=epochs,
    callbacks=[annealer, earlystop, modelsave],
)



## === cell 13
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

with open(csv_testfile, "w") as f:
    for dirname, _, filenames in os.walk(TEST_DIR):
        for filename in filenames:
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            row = dirname + "/" + filename + ";" + filename
            f.write(row + "\n")

print("Wrote:", csv_testfile, "size(bytes)=", os.path.getsize(csv_testfile))



## === cell 15
column_names = ["path", "file"]
dataFrameTest = pd.read_csv(csv_testfile, delimiter=";", header=None)
dataFrameTest.columns = column_names

print(dataFrameTest.shape)
print(dataFrameTest.head())
print(dataFrameTest["path"].iloc[0])



## === cell 16
batch_size = 32
image_size = (256, 256)
PROYECT_FOLDER_TEST = BASE  # flow_from_directory expects a base with subfolder "test"

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_directory(
    PROYECT_FOLDER_TEST,
    target_size=image_size,
    color_mode="rgb",
    batch_size=batch_size,
    classes=["test"],
    class_mode=None,  # no labels for test
    shuffle=False,  # preserve filename order for submission
)

print("Test samples found:", test_generator.n)



## === cell 17
best_path = file + ".h5"
if os.path.exists(best_path):
    pre_trained_model = keras.models.load_model(best_path, compile=False)
    print("Loaded best checkpoint:", best_path)
else:
    print("Checkpoint not found, using in-memory model weights:", best_path)

predicted_class = pre_trained_model.predict(test_generator, verbose=1)
print("Pred shape:", predicted_class.shape)

predicted_class_number = np.argmax(predicted_class, axis=1)
print("Num predictions:", len(predicted_class_number))



## === cell 18
pred_files = [os.path.basename(f) for f in test_generator.filenames]
pred_species = [idx_to_class[i] for i in predicted_class_number]

submission = pd.DataFrame({"file": pred_files, "species": pred_species})

print(submission.shape)
print(submission.head())

csv_resultsfile = "/kaggle/working/submission.csv"

sample_path = os.path.join(BASE, "sample_submission.csv")
sample = pd.read_csv(sample_path)

print("Sample shape:", sample.shape, "Raw submission shape:", submission.shape)
print("Columns OK:", list(submission.columns))

missing = set(sample["file"]) - set(submission["file"])
extra = set(submission["file"]) - set(sample["file"])
print("Missing files:", len(missing), "Extra files:", len(extra))

submission = sample[["file"]].merge(submission, on="file", how="left")
assert (
    submission["species"].isna().sum() == 0
), "Some test files did not get predictions."

submission.to_csv(csv_resultsfile, index=False)
print("Saved ordered submission to:", csv_resultsfile)
print(submission.head())
print("Submission columns:", submission.columns.tolist(), "rows:", len(submission))

## --- ERROR in outputing the csv:
Invalid submission: Submission must have 'file' column
