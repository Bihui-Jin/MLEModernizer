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

3.8

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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
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

# 5. Target score

0.14105

# 6. Current score

0.17417

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0991) has done: 'I fixed the import errors (switched to tf.keras to avoid protobuf issues), added missing os, numpy, pandas imports, corrected the callback monitor name, updated deprecated fit_generator/evaluate_generator/predict_generator calls to the current fit/evaluate/predict API, ensured the model is saved with a “.h5” extension, and adjusted the submission generation so the file column contains only the filename. These changes let the notebook run end‑to‑end and write a valid submission.csv while keeping the original model architecture and training logic.'
- What this solution (achieved 0.17417) has done: 'I set the protobuf implementation environment variable before importing TensorFlow to avoid the import error, increase the per‑category sample limit to use more training data, compute steps per epoch with a ceiling so all samples are used, and raise the training epochs from 4 to 8 to boost the micro‑F1 score toward the target while keeping the original model architecture unchanged.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    load_img,
    img_to_array,
)
from tensorflow.keras.applications import VGG19, ResNet50
from tensorflow.keras.applications.vgg19 import (
    preprocess_input as vgg19_preprocess_input,
)
from tensorflow.keras.applications.resnet50 import (
    preprocess_input as resnet50_preprocess_input,
)
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from sklearn.model_selection import StratifiedKFold

plt.rcParams["figure.figsize"] = [16, 10]
plt.rcParams["font.size"] = 16

SAMPLE_PER_CATEGORY = 500  # use more images per class for better performance
SEED = 42
WIDTH = 128
HEIGHT = 128
DEPTH = 3
INPUT_SHAPE = (WIDTH, HEIGHT, DEPTH)

DATA_DIR = "/kaggle/input/plant-seedlings-classification"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
SAMPLE_SUBMISSION = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

np.random.seed(SEED)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
CATEGORIES = [
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
NUM_CATEGORIES = len(CATEGORIES)
NUM_CATEGORIES



## === cell 2
train_rows = []
for cat_id, cat in enumerate(CATEGORIES):
    cat_path = os.path.join(TRAIN_DIR, cat)
    for fname in os.listdir(cat_path):
        train_rows.append([f"train/{cat}/{fname}", cat_id, cat])
train = pd.DataFrame(train_rows, columns=["file", "category_id", "category"])
train.shape



## === cell 3
train = pd.concat(
    [train[train["category"] == c][:SAMPLE_PER_CATEGORY] for c in CATEGORIES]
)
train = train.sample(frac=1, random_state=SEED).reset_index(drop=True)
train.shape



## === cell 4
test_rows = []
for fname in os.listdir(TEST_DIR):
    test_rows.append([f"test/{fname}", fname])
test = pd.DataFrame(test_rows, columns=["filepath", "file"])
test.shape




## === cell 5
def setTrainableLayersVGG(vgg_model):
    set_trainable = False
    for layer in vgg_model.layers:
        if layer.name in ["block5_conv1", "block4_conv1"]:
            set_trainable = True
        layer.trainable = set_trainable
    return vgg_model


def setTrainableLayersResNet(resnet_model):
    set_trainable = False
    for layer in resnet_model.layers:
        if layer.name in ["res5c_branch2b", "res5c_branch2c", "activation_97"]:
            set_trainable = True
        layer.trainable = set_trainable
    return resnet_model


def printHistory(history, title, epochs):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    f.suptitle(title, fontsize=12)
    epoch_list = list(range(1, epochs + 1))
    ax1.plot(epoch_list, history.history["accuracy"], label="Train Accuracy")
    ax1.plot(epoch_list, history.history["val_accuracy"], label="Validation Accuracy")
    ax1.set_xticks(np.arange(0, epochs + 1, max(1, epochs // 5)))
    ax1.set_ylabel("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax2.plot(epoch_list, history.history["loss"], label="Train Loss")
    ax2.plot(epoch_list, history.history["val_loss"], label="Validation Loss")
    ax2.set_xticks(np.arange(0, epochs + 1, max(1, epochs // 5)))
    ax2.set_ylabel("Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend()


def createModel(
    pretrainedModel,
    fineTune,
    number_of_hidden_layers,
    activation,
    optimizer,
    learning_rate,
    epochs,
):
    if pretrainedModel == "ResNet-50":
        base = ResNet50(weights="imagenet", include_top=False, input_shape=INPUT_SHAPE)
        if fineTune:
            base = setTrainableLayersResNet(base)
        else:
            base.trainable = False
    elif pretrainedModel == "VGG-19":
        base = VGG19(weights="imagenet", include_top=False, input_shape=INPUT_SHAPE)
        if fineTune:
            base = setTrainableLayersVGG(base)
        else:
            base.trainable = False
    else:
        raise ValueError("Unsupported pretrained model")

    x = Flatten()(base.output)
    model = Model(base.input, x)

    seq = Sequential()
    seq.add(model)

    for _ in range(number_of_hidden_layers):
        seq.add(Dense(512))
        seq.add(Activation(activation))
        seq.add(Dropout(0.3))

    seq.add(Dense(NUM_CATEGORIES, activation="softmax"))

    if optimizer == "SGD":
        opt = SGD(learning_rate=learning_rate, decay=learning_rate / epochs)
    elif optimizer == "Adam":
        opt = Adam(learning_rate=learning_rate, decay=learning_rate / epochs)
    else:
        raise ValueError("Unsupported optimizer")

    seq.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return seq


def get_callbacks(patience):
    lr_reduce = ReduceLROnPlateau(
        monitor="val_accuracy", factor=0.1, min_delta=1e-5, patience=patience, verbose=1
    )
    return [lr_reduce, EarlyStopping(patience=patience, restore_best_weights=True)]


def trainFinalModel(
    images,
    pretrainedModel,
    fineTune,
    epochs,
    batch_size,
    learning_rate,
    activation,
    number_of_hidden_layers,
    optimizer,
):
    datagen = ImageDataGenerator(rescale=1.0 / 255)

    model = createModel(
        pretrainedModel,
        fineTune,
        number_of_hidden_layers,
        activation,
        optimizer,
        learning_rate,
        epochs,
    )

    train_gen = datagen.flow_from_dataframe(
        dataframe=images,
        directory=DATA_DIR,
        x_col="file",
        y_col="category",
        batch_size=batch_size,
        seed=SEED,
        shuffle=True,
        class_mode="categorical",
        classes=CATEGORIES,
        target_size=(HEIGHT, WIDTH),
    )

    steps_per_epoch = int(np.ceil(train_gen.n / train_gen.batch_size))

    model.fit(train_gen, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=1)

    model_path = "/kaggle/working/best_model.h5"
    model.save(model_path)
    return train_gen.class_indices


def predict_createSubmission(class_indices):
    datagen = ImageDataGenerator(rescale=1.0 / 255)

    test_gen = datagen.flow_from_dataframe(
        dataframe=test,
        directory=DATA_DIR,
        x_col="filepath",
        y_col=None,
        batch_size=1,
        seed=SEED,
        shuffle=False,
        class_mode=None,
        classes=CATEGORIES,
        target_size=(HEIGHT, WIDTH),
    )

    model = load_model("/kaggle/working/best_model.h5")
    nb_samples = len(test_gen.filenames)

    predictions = model.predict(test_gen, steps=nb_samples, verbose=0)
    pred_indices = np.argmax(predictions, axis=1)

    idx_to_label = {v: k for k, v in class_indices.items()}
    pred_labels = [idx_to_label[idx] for idx in pred_indices]

    files = [os.path.basename(f) for f in test_gen.filenames]

    results = pd.DataFrame({"file": files, "species": pred_labels})
    results.to_csv("submission.csv", index=False)
    print("Submission file written to submission.csv")




## === cell 6
class_indices = trainFinalModel(
    images=train,
    pretrainedModel="ResNet-50",
    fineTune=True,
    epochs=8,  # increased epochs for better learning
    batch_size=32,
    learning_rate=0.001,
    activation="relu",
    number_of_hidden_layers=1,
    optimizer="Adam",
)

predict_createSubmission(class_indices)
