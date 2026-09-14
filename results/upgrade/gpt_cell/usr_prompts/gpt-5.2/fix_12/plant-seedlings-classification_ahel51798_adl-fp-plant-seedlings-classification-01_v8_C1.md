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

3.12

# 3. Installed packages

geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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
protobuf==6.33.0
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

print(
    "Train dir listing (first 20):",
    os.listdir("/kaggle/input/plant-seedlings-classification/train")[:20],
)



## === cell 1
from datetime import datetime, timedelta
import gc
import random
import pickle

import sys
import subprocess

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint,
)

from tensorflow.keras.applications.inception_v3 import (
    preprocess_input as inception_preprocess_input,
)

start_time = datetime.now()
print("Time now is", start_time)
end_training_by_tdelta = timedelta(seconds=8400)
this_run_file_prefix = start_time.strftime("%Y%m%d_%H%M_")
print("this_run_file_prefix", this_run_file_prefix)


## === cell 2
print("TensorFlow version:", tf.__version__)

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Running on TPU ", tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)




## === cell 3
def visualize_class_images(base_directory, rows=2, cols=6):
    class_directories = [
        d
        for d in os.listdir(base_directory)
        if os.path.isdir(os.path.join(base_directory, d))
    ]

    images_to_display = []
    for class_dir in class_directories:
        class_dir_path = os.path.join(base_directory, class_dir)
        class_images = os.listdir(class_dir_path)

        if not class_images:
            continue

        selected_image = random.choice(class_images)
        images_to_display.append(
            (os.path.join(class_dir_path, selected_image), class_dir)
        )

    random.shuffle(images_to_display)
    images_to_display = images_to_display[: rows * cols]

    plt.figure(figsize=(15, 6))
    for i, (img_path, class_name) in enumerate(images_to_display):
        img = Image.open(img_path)
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(class_name)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


train_directory = "/kaggle/input/plant-seedlings-classification/train"
validation_directory = "/kaggle/input/plant-seedlings-classification/train"
test_directory = "/kaggle/input/plant-seedlings-classification/test"
visualize_class_images(train_directory)


## === cell 4
IMAGE_SIZE = [299, 299]  # InceptionV3 default
width = 299
height = 299
num_classes = 12
CLASSES = [
    "Black-grass",
    "Charlock",
    "Cleavers",
    "Common Chickweed",
    "Common wheat",
    "Fat Hen",
    "Loose Silky-bent",
    "Maize",
    "Scentless Mayweed",
    "Shepherds Purse",
    "Small-flowered Cranesbill",
    "Sugar beet",
]
BATCH_SIZE = 16 * strategy.num_replicas_in_sync


def define_generators(train_directory, validation_directory, test_directory):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=inception_preprocess_input,
        rotation_range=360,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.5,
        vertical_flip=True,
        horizontal_flip=True,
        validation_split=0.15,
    )

    train_generator = train_datagen.flow_from_directory(
        directory=train_directory,
        target_size=(width, height),
        batch_size=BATCH_SIZE,
        color_mode="rgb",
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=42,
    )

    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=inception_preprocess_input,
        validation_split=0.15,
    )

    validation_generator = validation_datagen.flow_from_directory(
        directory=validation_directory,
        target_size=(width, height),
        batch_size=BATCH_SIZE,
        color_mode="rgb",
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=42,
    )

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=inception_preprocess_input
    )

    test_generator = test_datagen.flow_from_directory(
        directory=os.path.dirname(test_directory),
        classes=["test"],
        target_size=(width, height),
        batch_size=1,
        color_mode="rgb",
        shuffle=False,
        class_mode=None,
    )

    return train_generator, validation_generator, test_generator


def visualize_generator_samples(generator, rows=2, cols=6):
    images, labels = next(generator)
    class_labels = list(generator.class_indices.keys())

    plt.figure(figsize=(15, 6))
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        img = images[i]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        plt.imshow(img)
        plt.title(class_labels[np.argmax(labels[i])])
        plt.axis("off")
    plt.tight_layout()
    plt.show()


train_generator, validation_generator, test_generator = define_generators(
    train_directory, validation_directory, test_directory
)

len_train_generator = len(train_generator)
len_validation_generator = len(validation_generator)
len_test_generator = len(test_generator)

print(f"\nLength of Train Generator: {len_train_generator}")
print(f"Length of Validation Generator: {len_validation_generator}")
print(f"Length of Test Generator: {len_test_generator}")

images, labels = next(train_generator)
print(f"\nShape of images in the first batch: {images.shape}")
print(f"Shape of labels in the first batch: {labels.shape}")

visualize_generator_samples(train_generator)




## === cell 5
def create_ResNet50_model():
    pretrained_model = tf.keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_shape=[*IMAGE_SIZE, 3]
    )
    pretrained_model.trainable = True
    model = tf.keras.Sequential(
        [
            pretrained_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(len(CLASSES), activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def create_ResNet101V2_model():
    pretrained_model = tf.keras.applications.ResNet101V2(
        weights="imagenet", include_top=False, input_shape=[*IMAGE_SIZE, 3]
    )
    pretrained_model.trainable = True
    model = tf.keras.Sequential(
        [
            pretrained_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(len(CLASSES), activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def create_VGG16_model():
    pretrained_model = tf.keras.applications.VGG16(
        weights="imagenet", include_top=False, input_shape=[*IMAGE_SIZE, 3]
    )
    pretrained_model.trainable = True
    model = tf.keras.Sequential(
        [
            pretrained_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(len(CLASSES), activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def create_Xception_model():
    pretrained_model = tf.keras.applications.Xception(
        include_top=False, input_shape=[*IMAGE_SIZE, 3]
    )
    pretrained_model.trainable = True
    model = tf.keras.Sequential(
        [
            pretrained_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(len(CLASSES), activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def create_DenseNet_model():
    pretrained_model = tf.keras.applications.DenseNet201(
        weights="imagenet", include_top=False, input_shape=[*IMAGE_SIZE, 3]
    )
    pretrained_model.trainable = True
    model = tf.keras.Sequential(
        [
            pretrained_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(len(CLASSES), activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def create_EfficientNet_model():
    raise ImportError("efficientnet package not available in this environment")


def create_InceptionV3_model():
    pretrained_model = tf.keras.applications.InceptionV3(
        weights="imagenet", include_top=False, input_shape=[*IMAGE_SIZE, 3]
    )
    pretrained_model.trainable = True
    model = tf.keras.Sequential(
        [
            pretrained_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(len(CLASSES), activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def create_ResNet152_model():
    pretrained_model = tf.keras.applications.ResNet152V2(
        weights="imagenet", include_top=False, input_shape=[*IMAGE_SIZE, 3]
    )
    pretrained_model.trainable = True
    model = tf.keras.Sequential(
        [
            pretrained_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(len(CLASSES), activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def create_MobileNetV2_model():
    pretrained_model = tf.keras.applications.MobileNetV2(
        weights="imagenet", include_top=False, input_shape=[*IMAGE_SIZE, 3]
    )
    pretrained_model.trainable = True
    model = tf.keras.Sequential(
        [
            pretrained_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(len(CLASSES), activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def create_InceptionResNetV2_model():
    pretrained_model = tf.keras.applications.InceptionResNetV2(
        weights="imagenet", include_top=False, input_shape=[*IMAGE_SIZE, 3]
    )
    pretrained_model.trainable = True
    model = tf.keras.Sequential(
        [
            pretrained_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(len(CLASSES), activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def write_history(j, filename):
    history_dict = [0] * no_of_models
    for i in range(j + 1):
        if historys[i] != 0:
            history_dict[i] = historys[i].history
    filename = filename + ".pkl"
    with open(filename, "ab") as pklfile:
        pickle.dump(history_dict, pklfile)


def load_history(filename):
    with open(filename, "rb") as file:
        history_dict = pickle.load(file)
    return history_dict


def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["accuracy"], label="Accuracy")
    plt.plot(history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["Train", "Validation"], loc="upper left")

    plt.subplot(1, 2, 2)
    plt.plot(history["loss"], label="Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Train", "Validation"], loc="upper left")
    plt.show()




## === cell 6
def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = LR_START + (epoch * (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS)
    elif epoch < (LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS):
        lr = LR_MAX
    else:
        lr = LR_MIN + (LR_MAX - LR_MIN) * LR_EXP_DECAY ** (
            epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS
        )
    return lr


LR_START = 0.00001
LR_MAX = 0.00005 * strategy.num_replicas_in_sync
LR_MIN = LR_START
LR_RAMPUP_EPOCHS = 5
LR_SUSTAIN_EPOCHS = 0
LR_EXP_DECAY = 0.80

lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)

rng = [i for i in range(30)]
y = [lrfn(x) for x in rng]
plt.plot(rng, y)
plt.title("Learning Rate over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.show()

print("lrfn y:", y)



## === cell 7
no_of_models = 1
models = [0] * no_of_models
start_model = 0
end_model = 1
model_indx_0 = start_model
model_indx_1 = start_model + 1

EPOCHS = 50
historys = [0] * no_of_models
finished_models = 0

checkpoint_path = "/kaggle/working/best_model.keras"

early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)
lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)
learning_rate_reduction = ReduceLROnPlateau(
    monitor="val_accuracy", patience=2, verbose=1, factor=0.5, min_lr=0.00001
)
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    save_best_only=True,
    monitor="val_loss",
    mode="min",
    verbose=1,
)

val_probabilities = [0] * no_of_models
test_probabilities = [0] * no_of_models
all_probabilities = [0] * no_of_models

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=inception_preprocess_input,
    rotation_range=360,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.5,
    vertical_flip=True,
    horizontal_flip=True,
    validation_split=0.15,
)
train_generator = train_datagen.flow_from_directory(
    directory=train_directory,
    classes=CLASSES,
    target_size=(width, height),
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=42,
)

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=inception_preprocess_input,
    validation_split=0.15,
)
validation_generator = validation_datagen.flow_from_directory(
    directory=validation_directory,
    classes=CLASSES,
    target_size=(width, height),
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    seed=42,
)

with strategy.scope():
    for j in range(no_of_models):
        models[j] = create_InceptionV3_model()

models[0].summary()

for j in range(start_model, end_model):
    start_training = datetime.now()
    print(start_training)
    time_from_start_program_tdelta = start_training - start_time
    if time_from_start_program_tdelta > end_training_by_tdelta:
        print(j, "time limit for doing training over, get out")
        break
    print("LR_EXP_DECAY:", LR_EXP_DECAY, ". LR_MAX:", LR_MAX)

    historys[j] = models[j].fit(
        train_generator,
        epochs=EPOCHS,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        callbacks=[
            learning_rate_reduction,
            early_stopping,
            lr_callback,
            model_checkpoint_callback,
        ],
    )

    filename = this_run_file_prefix + "models_" + str(j)
    write_history(j, filename)
    models[j].save(filename)

    gc.collect()
    finished_models = j + 1

print(datetime.now())


## === cell 8
history_to_analyze = historys[0].history
plot_history(history_to_analyze)

best_model = load_model("/kaggle/working/best_model.keras")




## === cell 9
def plot_test_images_with_predictions(generator, predictions, num_images):
    num_images = min(num_images, generator.samples)
    plt.figure(figsize=(15, 15))

    for i in range(num_images):
        img_path = os.path.join(generator.directory, generator.filenames[i])
        img = Image.open(img_path)
        predicted_class = predictions[i]
        plt.subplot(max(1, num_images // 5), 5, i + 1)
        plt.imshow(img)
        plt.title(f"Predicted: {predicted_class}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


predictions = best_model.predict(
    test_generator, steps=test_generator.samples, verbose=1
)
class_list = [CLASSES[int(np.argmax(pred, axis=-1))] for pred in predictions]

plot_test_images_with_predictions(test_generator, class_list, num_images=25)



## === cell 10
sample_sub_path = "/kaggle/input/plant-seedlings-classification/sample_submission.csv"
sample = pd.read_csv(sample_sub_path)

pred_df = pd.DataFrame(
    {
        "file": pd.Series(test_generator.filenames).str.replace(
            "test/", "", regex=False
        ),
        "species": class_list,
    }
)

submission = sample[["file"]].merge(pred_df, on="file", how="left")

if submission["species"].isna().any():
    submission["species"] = submission["species"].fillna(
        pred_df["species"].mode().iloc[0]
    )

assert submission.shape[0] == sample.shape[0]
assert submission["species"].isna().sum() == 0

submission.to_csv("submission.csv", index=False)
print(
    "Submission file generated at /kaggle/working/submission.csv with shape:",
    submission.shape,
)
print(submission.head())
