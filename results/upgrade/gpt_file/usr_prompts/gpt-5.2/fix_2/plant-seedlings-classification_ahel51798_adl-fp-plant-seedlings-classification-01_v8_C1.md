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

# 5. Target score

0.98488

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O

print("Listing train directory (may be empty if path is wrong):")
print(os.listdir("/kaggle/input/plant-seedlings-classification")[:20])



## === cell 1
from datetime import datetime, timedelta
import gc
import random
import pickle

import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint,
)

start_time = datetime.now()
print("Time now is", start_time)
end_training_by_tdelta = timedelta(seconds=8400)
this_run_file_prefix = start_time.strftime("%Y%m%d_%H%M_")
print("this_run_file_prefix", this_run_file_prefix)

print("TensorFlow version:", tf.__version__)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Running on TPU ", tpu.master())
except Exception:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)



## === cell 3
DATA_ROOT = (
    "/kaggle/input/plant-seedlings-classification/plant-seedlings-classification"
)
train_directory = os.path.join(DATA_ROOT, "train")
validation_directory = os.path.join(
    DATA_ROOT, "train"
)  # unchanged core logic (still same as train)
test_directory = os.path.join(DATA_ROOT, "test")

print("Train dir exists:", os.path.exists(train_directory), train_directory)
print("Test  dir exists:", os.path.exists(test_directory), test_directory)


def visualize_class_images(base_directory, rows=2, cols=6):
    class_directories = [
        d
        for d in os.listdir(base_directory)
        if os.path.isdir(os.path.join(base_directory, d))
    ]

    images_to_display = []
    for class_dir in class_directories:
        class_dir_path = os.path.join(base_directory, class_dir)
        class_images = [
            f
            for f in os.listdir(class_dir_path)
            if os.path.isfile(os.path.join(class_dir_path, f))
        ]
        if not class_images:
            continue
        selected_image = random.choice(class_images)
        images_to_display.append(
            (os.path.join(class_dir_path, selected_image), class_dir)
        )

    if not images_to_display:
        print("No images found to visualize. Check the directory path:", base_directory)
        return

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


visualize_class_images(train_directory)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1718273823.py in <cell line: 0>()
     54 
     55 
---> 56 visualize_class_images(train_directory)
     57 
     58 

/tmp/ipykernel_11/1718273823.py in visualize_class_images(base_directory, rows, cols)
     16     class_directories = [
     17         d
---> 18         for d in os.listdir(base_directory)
     19         if os.path.isdir(os.path.join(base_directory, d))
     20     ]

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/plant-seedlings-classification/plant-seedlings-classification/train'

## === cell 4
def define_generators(train_directory, validation_directory, test_directory):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=360,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.5,
        vertical_flip=True,
        horizontal_flip=True,
    )

    train_generator = train_datagen.flow_from_directory(
        directory=train_directory,
        target_size=(width, height),
        batch_size=BATCH_SIZE,
        color_mode="rgb",
        class_mode="categorical",
    )

    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

    validation_generator = validation_datagen.flow_from_directory(
        directory=validation_directory,
        target_size=(width, height),
        batch_size=BATCH_SIZE,
        color_mode="rgb",
        class_mode="categorical",
    )

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

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
    n = min(rows * cols, images.shape[0])
    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i].astype("uint8"))
        plt.title(class_labels[int(np.argmax(labels[i]))])
        plt.axis("off")
    plt.tight_layout()
    plt.show()




## === cell 5
IMAGE_SIZE = [299, 299]  # kept as in original
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
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




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2883100902.py in <cell line: 0>()
     19 ]
     20 
---> 21 train_generator, validation_generator, test_generator = define_generators(
     22     train_directory, validation_directory, test_directory
     23 )

/tmp/ipykernel_11/824974485.py in define_generators(train_directory, validation_directory, test_directory)
     10     )
     11 
---> 12     train_generator = train_datagen.flow_from_directory(
     13         directory=train_directory,
     14         target_size=(width, height),

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/preprocessing/image.py in flow_from_directory(self, directory, target_size, color_mode, classes, class_mode, batch_size, shuffle, seed, save_to_dir, save_prefix, save_format, follow_links, subset, interpolation, keep_aspect_ratio)
   1136         keep_aspect_ratio=False,
   1137     ):
-> 1138         return DirectoryIterator(
   1139             directory,
   1140             self,

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/preprocessing/image.py in __init__(self, directory, image_data_generator, target_size, color_mode, classes, class_mode, batch_size, shuffle, seed, data_format, save_to_dir, save_prefix, save_format, follow_links, subset, interpolation, keep_aspect_ratio, dtype)
    451         if not classes:
    452             classes = []
--> 453             for subdir in sorted(os.listdir(directory)):
    454                 if os.path.isdir(os.path.join(directory, subdir)):
    455                     classes.append(subdir)

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/plant-seedlings-classification/plant-seedlings-classification/train'

## === cell 6
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




## === cell 7
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



## === cell 8
no_of_models = 1
models = [0] * no_of_models
start_model = 0
end_model = 1

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
    models[j].save(filename + ".keras")

    gc.collect()
    finished_models = j + 1

print(datetime.now())



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3885047242.py in <cell line: 0>()
     46     print("LR_EXP_DECAY:", LR_EXP_DECAY, ". LR_MAX:", LR_MAX)
     47     historys[j] = models[j].fit(
---> 48         train_generator,
     49         epochs=EPOCHS,
     50         steps_per_epoch=train_generator.samples // BATCH_SIZE,

NameError: name 'train_generator' is not defined

## === cell 9
if historys[0] != 0:
    history_to_analyze = historys[0].history
    plot_history(history_to_analyze)
else:
    print(
        "Training did not run (history missing). Proceeding to load checkpoint if available."
    )

best_model = load_model("/kaggle/working/best_model.keras")




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3842186719.py in <cell line: 0>()
      8     )
      9 
---> 10 best_model = load_model("/kaggle/working/best_model.keras")
     11 
     12 

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    198         )
    199     elif str(filepath).endswith(".keras"):
--> 200         raise ValueError(
    201             f"File not found: filepath={filepath}. "
    202             "Please ensure the file is an accessible `.keras` "

ValueError: File not found: filepath=/kaggle/working/best_model.keras. Please ensure the file is an accessible `.keras` zip file.

## === cell 10
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



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/327061055.py in <cell line: 0>()
     15 
     16 
---> 17 predictions = best_model.predict(
     18     test_generator, steps=test_generator.samples, verbose=1
     19 )

NameError: name 'best_model' is not defined

## === cell 11
submission = pd.DataFrame()
submission["file"] = test_generator.filenames
submission["file"] = submission["file"].str.replace("test/", "", regex=False)
submission["species"] = class_list

sample_path = os.path.join(DATA_ROOT, "sample_submission.csv")
if os.path.exists(sample_path):
    sample = pd.read_csv(sample_path)
    submission = sample[["file"]].merge(submission, on="file", how="left")
    submission["species"] = submission["species"].fillna(CLASSES[0])

submission.to_csv("submission.csv", index=False)
print("Submission file generated:", os.path.abspath("submission.csv"))
print(submission.head())
print("Rows:", len(submission), "Cols:", submission.columns.tolist())

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3189145937.py in <cell line: 0>()
      1 submission = pd.DataFrame()
----> 2 submission["file"] = test_generator.filenames
      3 submission["file"] = submission["file"].str.replace("test/", "", regex=False)
      4 submission["species"] = class_list
      5 

NameError: name 'test_generator' is not defined
