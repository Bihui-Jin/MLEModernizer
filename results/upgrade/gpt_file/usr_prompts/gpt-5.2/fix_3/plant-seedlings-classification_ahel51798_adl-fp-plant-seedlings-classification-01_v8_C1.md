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

0.12162

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.12162) has done: 'I fix the TensorFlow/protobuf crash by removing the incompatible `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` override (it triggers the `MessageFactory.GetPrototype` error in this environment). I also correct the dataset root path so the code can actually find `train/` and `test/`, and add a small, safe fallback that searches known Kaggle locations if the preferred path isn’t present. Finally, I make training and inference robust: ensure generators are always defined before fitting, load the best checkpoint if it exists (otherwise use the in-memory model), and always write a valid `submission.csv` with the required `file,species` columns aligned to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import numpy as np
import pandas as pd

print("Listing /kaggle/input/plant-seedlings-classification (if present):")
if os.path.exists("/kaggle/input/plant-seedlings-classification"):
    print(os.listdir("/kaggle/input/plant-seedlings-classification")[:30])
else:
    print("Path not found.")



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
CANDIDATE_ROOTS = [
    "/kaggle/input/plant-seedlings-classification",  # common Kaggle mount
    "/kaggle/input/plant-seedlings-classification/plant-seedlings-classification",
    "/kaggle/data/plant-seedlings-classification",
    "/kaggle/data/plant-seedlings-classification/plant-seedlings-classification",
]


def pick_data_root(candidates):
    for root in candidates:
        tr = os.path.join(root, "train")
        te = os.path.join(root, "test")
        if os.path.isdir(tr) and os.path.isdir(te):
            return root
    base = "/kaggle/input"
    if os.path.isdir(base):
        for name in os.listdir(base):
            root = os.path.join(base, name)
            tr = os.path.join(root, "train")
            te = os.path.join(root, "test")
            if os.path.isdir(tr) and os.path.isdir(te):
                return root
            nested = os.path.join(root, name)
            tr2 = os.path.join(nested, "train")
            te2 = os.path.join(nested, "test")
            if os.path.isdir(tr2) and os.path.isdir(te2):
                return nested
    raise FileNotFoundError(
        "Could not locate dataset root containing train/ and test/."
    )


DATA_ROOT = pick_data_root(CANDIDATE_ROOTS)

train_directory = os.path.join(DATA_ROOT, "train")
validation_directory = os.path.join(
    DATA_ROOT, "train"
)  # unchanged core logic (still same as train)
test_directory = os.path.join(DATA_ROOT, "test")

print("DATA_ROOT:", DATA_ROOT)
print("Train dir exists:", os.path.exists(train_directory), train_directory)
print("Test  dir exists:", os.path.exists(test_directory), test_directory)


def visualize_class_images(base_directory, rows=2, cols=6):
    if not os.path.isdir(base_directory):
        print("No directory found to visualize:", base_directory)
        return

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
        shuffle=True,
    )

    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

    validation_generator = validation_datagen.flow_from_directory(
        directory=validation_directory,
        target_size=(width, height),
        batch_size=BATCH_SIZE,
        color_mode="rgb",
        class_mode="categorical",
        shuffle=False,
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
len_validation_generator = (
    len(validationation_generator) if False else len(validation_generator)
)
len_test_generator = len(test_generator)

print(f"\nLength of Train Generator: {len_train_generator}")
print(f"Length of Validation Generator: {len_validation_generator}")
print(f"Length of Test Generator: {len_test_generator}")

images, labels = next(train_generator)
print(f"\nShape of images in the first batch: {images.shape}")
print(f"Shape of labels in the first batch: {labels.shape}")

visualize_generator_samples(train_generator)




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
        steps_per_epoch=max(1, train_generator.samples // BATCH_SIZE),
        validation_data=validation_generator,
        validation_steps=max(1, validation_generator.samples // BATCH_SIZE),
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
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2688483801.py in <cell line: 0>()
     44 
     45     print("LR_EXP_DECAY:", LR_EXP_DECAY, ". LR_MAX:", LR_MAX)
---> 46     historys[j] = models[j].fit(
     47         train_generator,
     48         epochs=EPOCHS,

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

## === cell 9
if historys[0] != 0:
    history_to_analyze = historys[0].history
    plot_history(history_to_analyze)
else:
    print("Training did not run (history missing). Proceeding without plot.")

if os.path.exists(checkpoint_path):
    best_model = load_model(checkpoint_path)
    print("Loaded best model from checkpoint:", checkpoint_path)
else:
    best_model = models[0]
    print("Checkpoint not found; using in-memory trained model.")




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



## === cell 11
submission = pd.DataFrame()
submission["file"] = test_generator.filenames
submission["file"] = submission["file"].str.replace("test/", "", regex=False)
submission["species"] = class_list

sample_path = os.path.join(DATA_ROOT, "sample_submission.csv")
if os.path.exists(sample_path):
    sample = pd.read_csv(sample_path)
else:
    fallback_sample = (
        "/kaggle/input/plant-seedlings-classification/sample_submission.csv"
    )
    sample = pd.read_csv(fallback_sample) if os.path.exists(fallback_sample) else None

if sample is not None:
    submission = sample[["file"]].merge(submission, on="file", how="left")
    submission["species"] = submission["species"].fillna(CLASSES[0])

submission.to_csv("submission.csv", index=False)
print("Submission file generated:", os.path.abspath("submission.csv"))
print(submission.head())
print("Rows:", len(submission), "Cols:", submission.columns.tolist())
assert list(submission.columns) == ["file", "species"]
assert len(submission) == 666
assert submission["species"].notna().all()
