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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from sklearn.model_selection import StratifiedKFold

import tf_keras as keras
from tf_keras.models import Sequential, Model, load_model
from tf_keras.layers import Dense, Dropout, Activation, Flatten
from tf_keras.layers import (
    Conv2D,
    MaxPooling2D,
)  # kept for compatibility with original imports
from tf_keras.preprocessing.image import ImageDataGenerator
from tf_keras.preprocessing import image
from tf_keras.optimizers import Adam, SGD
from tf_keras.callbacks import ReduceLROnPlateau, EarlyStopping

from tf_keras.applications import VGG19
from tf_keras.applications.resnet50 import ResNet50

plt.rcParams["figure.figsize"] = [16, 10]
plt.rcParams["font.size"] = 16

SAMPLE_PER_CATEGORY = 200
SEED = 42
WIDTH = 128
HEIGHT = 128
DEPTH = 3
INPUT_SHAPE = (WIDTH, HEIGHT, DEPTH)

data_dir = "/kaggle/input/plant-seedlings-classification/"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
sample_submission = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))

np.random.seed(SEED)
try:
    import tensorflow as tf

    tf.random.set_seed(SEED)
except Exception:
    pass

print("Data dir:", data_dir)
print("Train dir exists:", os.path.isdir(train_dir))
print("Test dir exists:", os.path.isdir(test_dir))
print("Sample submission shape:", sample_submission.shape)



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
for category in CATEGORIES:
    print(
        "{} {} images".format(
            category, len(os.listdir(os.path.join(train_dir, category)))
        )
    )




## === cell 3
def read_img(filepath, size):
    img = image.load_img(os.path.join(data_dir, filepath), target_size=size)
    img = image.img_to_array(img)
    return img




## === cell 4
train = []
for category_id, category in enumerate(CATEGORIES):
    for file in os.listdir(os.path.join(train_dir, category)):
        train.append([f"train/{category}/{file}", category_id, category])
train = pd.DataFrame(train, columns=["file", "category_id", "category"])
train.shape



## === cell 5
train.head(2)



## === cell 6
train = pd.concat(
    [train[train["category"] == c].iloc[:SAMPLE_PER_CATEGORY] for c in CATEGORIES],
    axis=0,
)
train = train.sample(frac=1, random_state=SEED).reset_index(drop=True)
train.shape



## === cell 7
train.head()



## === cell 8
test = []
for file in os.listdir(test_dir):
    test.append([f"test/{file}", file])
test = pd.DataFrame(test, columns=["file", "file_id"])
test.shape



## === cell 9
test.head(2)



## === cell 10
try:
    fig = plt.figure(1, figsize=(NUM_CATEGORIES, NUM_CATEGORIES))
    grid = ImageGrid(
        fig, 111, nrows_ncols=(NUM_CATEGORIES, NUM_CATEGORIES), axes_pad=0.05
    )
    i = 0
    for category_id, category in enumerate(CATEGORIES):
        for filepath in train[train["category"] == category]["file"].values[
            :NUM_CATEGORIES
        ]:
            ax = grid[i]
            img = read_img(filepath, (WIDTH, HEIGHT))
            ax.imshow(img / 255.0)
            ax.axis("off")
            if i % NUM_CATEGORIES == NUM_CATEGORIES - 1:
                ax.text(250, 112, filepath.split("/")[1], verticalalignment="center")
            i += 1
    plt.show()
except Exception as e:
    print("Visualization skipped:", repr(e))



## === cell 11
np.random.seed(seed=SEED)




## === cell 12
def setTrainableLayersVGG(vgg_model):
    set_trainable = False
    for layer in vgg_model.layers:
        if layer.name in ["block5_conv1", "block4_conv1"]:
            set_trainable = True

        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False
    return vgg_model




## === cell 13
vgg = VGG19(include_top=False, weights="imagenet", input_shape=INPUT_SHAPE)

output = vgg.layers[-1].output
output = keras.layers.Flatten()(output)
vgg_model = Model(vgg.input, output)

vgg_model = setTrainableLayersVGG(vgg_model)

pd.set_option("display.max_colwidth", None)
layers = [
    (layer.__class__.__name__, layer.name, layer.trainable)
    for layer in vgg_model.layers
]
pd.DataFrame(layers, columns=["Layer Type", "Layer Name", "Layer Trainable"]).head(10)




## === cell 14
def setTrainableLayersResNet(resnet_model):
    set_trainable = False
    for layer in resnet_model.layers:
        if layer.name in ["res5c_branch2b", "res5c_branch2c", "activation_97"]:
            set_trainable = True

        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False
    return resnet_model




## === cell 15
resnet = ResNet50(include_top=False, weights="imagenet", input_shape=INPUT_SHAPE)

output = resnet.layers[-1].output
output = keras.layers.Flatten()(output)
resnet_model = Model(resnet.input, output)

setTrainableLayersResNet(resnet_model)

layers = [
    (layer.__class__.__name__, layer.name, layer.trainable)
    for layer in resnet_model.layers
]
pd.DataFrame(layers, columns=["Layer Type", "Layer Name", "Layer Trainable"]).head(10)




## === cell 16
def printHistory(history, title, epochs):
    try:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        f.suptitle(title, fontsize=12)
        f.subplots_adjust(top=0.85, wspace=0.3)

        epoch_list = list(range(1, epochs + 1))
        ax1.plot(
            epoch_list, history.history.get("accuracy", []), label="Train Accuracy"
        )
        ax1.plot(
            epoch_list,
            history.history.get("val_accuracy", []),
            label="Validation Accuracy",
        )
        ax1.set_xticks(np.arange(0, epochs + 1, 1 if epochs < 10 else 5))
        ax1.set_ylabel("Accuracy Value")
        ax1.set_xlabel("Epoch")
        ax1.set_title("Accuracy")
        ax1.legend(loc="best")

        ax2.plot(epoch_list, history.history.get("loss", []), label="Train Loss")
        ax2.plot(
            epoch_list, history.history.get("val_loss", []), label="Validation Loss"
        )
        ax2.set_xticks(np.arange(0, epochs + 1, 1 if epochs < 10 else 5))
        ax2.set_ylabel("Loss Value")
        ax2.set_xlabel("Epoch")
        ax2.set_title("Loss")
        ax2.legend(loc="best")
        plt.show()
    except Exception as e:
        print("History plot skipped:", repr(e))




## === cell 17
def createModel(
    pretrainedModel,
    fineTune,
    number_of_hidden_layers,
    activation,
    optimizer,
    learning_rate,
    epochs,
):
    print("Create Model")

    tranfer_model = None

    if pretrainedModel == "ResNet-50":
        tranfer_model = ResNet50(
            weights="imagenet", input_shape=INPUT_SHAPE, include_top=False
        )
        if fineTune is True:
            tranfer_model = setTrainableLayersResNet(tranfer_model)
        else:
            for layer in tranfer_model.layers:
                layer.trainable = False
    elif pretrainedModel == "VGG-19":
        tranfer_model = VGG19(
            weights="imagenet", input_shape=INPUT_SHAPE, include_top=False
        )
        if fineTune is True:
            tranfer_model = setTrainableLayersVGG(tranfer_model)
        else:
            for layer in tranfer_model.layers:
                layer.trainable = False

    output = tranfer_model.layers[-1].output
    output = keras.layers.Flatten()(output)
    trans_model = Model(tranfer_model.input, output)

    model = Sequential()
    model.add(trans_model)

    for _ in range(0, number_of_hidden_layers):
        model.add(Dense(512))
        model.add(Activation(activation))
        model.add(Dropout(0.3))

    model.add(Dense(12, activation="softmax"))

    if optimizer == "SGD":
        opt = SGD(learning_rate=learning_rate, decay=learning_rate / epochs)
    elif optimizer == "Adam":
        opt = Adam(learning_rate=learning_rate, decay=learning_rate / epochs)
    else:
        opt = Adam(learning_rate=learning_rate, decay=learning_rate / epochs)

    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model




## === cell 18
def get_callbacks(patience):
    print("Get Callbacks")
    lr_reduce = ReduceLROnPlateau(
        monitor="val_accuracy", factor=0.1, min_delta=1e-5, patience=patience, verbose=1
    )
    return [lr_reduce, EarlyStopping()]




## === cell 19
def trainModelDF(
    images,
    pretrainedModel,
    fineTune,
    epochs,
    batch_size,
    learning_rate,
    cross_validation_folds,
    activation,
    number_of_hidden_layers,
    optimizer,
):
    print("Train Model")

    datagen_train = ImageDataGenerator(rescale=1.0 / 255)
    datagen_valid = ImageDataGenerator(rescale=1.0 / 255)

    print("Cross validation")
    kfold = StratifiedKFold(
        n_splits=cross_validation_folds, shuffle=True, random_state=SEED
    )
    cvscores = []
    iteration = 1

    t = images.category_id

    for train_index, test_index in kfold.split(np.zeros(len(t)), t):
        print("======================================")
        print("Iteration = ", iteration)
        iteration += 1

        train_fold = images.iloc[train_index]
        valid_fold = images.iloc[test_index]

        print("======================================")

        model = createModel(
            pretrainedModel,
            fineTune,
            number_of_hidden_layers,
            activation,
            optimizer,
            learning_rate,
            epochs,
        )

        print("======================================")

        train_generator = datagen_train.flow_from_dataframe(
            dataframe=train_fold,
            directory="/kaggle/input/plant-seedlings-classification/",
            x_col="file",
            y_col="category",
            batch_size=batch_size,
            seed=SEED,
            shuffle=True,
            class_mode="categorical",
            classes=CATEGORIES,
            target_size=(HEIGHT, WIDTH),
        )

        valid_generator = datagen_valid.flow_from_dataframe(
            dataframe=valid_fold,
            directory="/kaggle/input/plant-seedlings-classification/",
            x_col="file",
            y_col="category",
            batch_size=batch_size,
            seed=SEED,
            shuffle=False,
            class_mode="categorical",
            classes=CATEGORIES,
            target_size=(HEIGHT, WIDTH),
        )

        STEP_SIZE_TRAIN = max(1, train_generator.n // train_generator.batch_size)
        STEP_SIZE_VALID = max(1, valid_generator.n // valid_generator.batch_size)

        history = model.fit(
            train_generator,
            validation_data=valid_generator,
            steps_per_epoch=STEP_SIZE_TRAIN,
            validation_steps=STEP_SIZE_VALID,
            epochs=epochs,
            verbose=1,
        )

        scores = model.evaluate(valid_generator, steps=STEP_SIZE_VALID, verbose=0)
        print("Accuarcy %s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        cvscores.append(scores[1] * 100)

        printHistory(history, pretrainedModel, epochs)

    accuracy = np.mean(cvscores)
    std = np.std(cvscores)
    print("Accuracy: %.2f%% (+/- %.2f%%)" % (accuracy, std))
    return accuracy, std




## === cell 20
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
    print("Train Model")

    datagen_train = ImageDataGenerator(rescale=1.0 / 255)

    print("======================================")
    model = createModel(
        pretrainedModel,
        fineTune,
        number_of_hidden_layers,
        activation,
        optimizer,
        learning_rate,
        epochs,
    )
    print("======================================")

    train_generator = datagen_train.flow_from_dataframe(
        dataframe=images,
        directory="/kaggle/input/plant-seedlings-classification/",
        x_col="file",
        y_col="category",
        batch_size=batch_size,
        seed=SEED,
        shuffle=True,
        class_mode="categorical",
        classes=CATEGORIES,
        target_size=(HEIGHT, WIDTH),
    )

    print(train_generator.class_indices)

    STEP_SIZE_TRAIN = max(1, train_generator.n // train_generator.batch_size)

    model.fit(
        train_generator, steps_per_epoch=STEP_SIZE_TRAIN, epochs=epochs, verbose=1
    )

    model.save("/kaggle/working/best_model.keras")

    return train_generator.class_indices




## === cell 21
def predict_createSubmission(class_indices):
    print("Predicting......")

    datagen_test = ImageDataGenerator(rescale=1.0 / 255)

    test_generator = datagen_test.flow_from_dataframe(
        dataframe=test,
        directory="/kaggle/input/plant-seedlings-classification/",
        x_col="file",
        y_col=None,
        batch_size=1,
        seed=SEED,
        shuffle=False,
        class_mode=None,
        target_size=(HEIGHT, WIDTH),
    )

    model = load_model("/kaggle/working/best_model.keras")
    filenames = test_generator.filenames
    nb_samples = len(filenames)

    predictions = model.predict(test_generator, steps=nb_samples, verbose=1)

    predicted_class_indices = np.argmax(predictions, axis=1)

    labels = dict((v, k) for k, v in class_indices.items())
    predicted_labels = [labels[k] for k in predicted_class_indices]

    submission_files = [os.path.basename(f) for f in filenames]

    results = pd.DataFrame({"file": submission_files, "species": predicted_labels})

    results = results.sort_values("file").reset_index(drop=True)
    sample_sorted = sample_submission.sort_values("file").reset_index(drop=True)
    if len(results) == len(sample_sorted) and set(results["file"]) == set(
        sample_sorted["file"]
    ):
        results = results.set_index("file").loc[sample_sorted["file"]].reset_index()

    results.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", results.shape)
    print(results.head())




## === cell 22
class_indices = trainFinalModel(
    train,
    pretrainedModel="ResNet-50",
    fineTune=True,
    batch_size=32,
    learning_rate=0.001,
    activation="relu",
    number_of_hidden_layers=1,
    optimizer="Adam",
    epochs=4,
)



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3901548160.py in <cell line: 0>()
      1 # Train final model (keeps original hyperparameters/logic)
----> 2 class_indices = trainFinalModel(
      3     train,
      4     pretrainedModel="ResNet-50",
      5     fineTune=True,

/tmp/ipykernel_11/1253292966.py in trainFinalModel(images, pretrainedModel, fineTune, epochs, batch_size, learning_rate, activation, number_of_hidden_layers, optimizer)
     16 
     17     print("======================================")
---> 18     model = createModel(
     19         pretrainedModel,
     20         fineTune,

/tmp/ipykernel_11/3334537354.py in createModel(pretrainedModel, fineTune, number_of_hidden_layers, activation, optimizer, learning_rate, epochs)
     49         opt = SGD(learning_rate=learning_rate, decay=learning_rate / epochs)
     50     elif optimizer == "Adam":
---> 51         opt = Adam(learning_rate=learning_rate, decay=learning_rate / epochs)
     52     else:
     53         opt = Adam(learning_rate=learning_rate, decay=learning_rate / epochs)

/usr/local/lib/python3.11/dist-packages/tf_keras/src/optimizers/adam.py in __init__(self, learning_rate, beta_1, beta_2, epsilon, adaptive_epsilon, amsgrad, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, jit_compile, name, **kwargs)
    112         **kwargs
    113     ):
--> 114         super().__init__(
    115             name=name,
    116             weight_decay=weight_decay,

/usr/local/lib/python3.11/dist-packages/tf_keras/src/optimizers/optimizer.py in __init__(self, name, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, jit_compile, **kwargs)
   1161         mesh = kwargs.pop("mesh", None)
   1162         self._mesh = mesh
-> 1163         super().__init__(
   1164             name,
   1165             weight_decay,

/usr/local/lib/python3.11/dist-packages/tf_keras/src/optimizers/optimizer.py in __init__(self, name, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, jit_compile, **kwargs)
    108         self._sharded_variable_builders = self._no_dependency({})
    109         self._create_iteration_variable()
--> 110         self._process_kwargs(kwargs)
    111 
    112     def _create_iteration_variable(self):

/usr/local/lib/python3.11/dist-packages/tf_keras/src/optimizers/optimizer.py in _process_kwargs(self, kwargs)
    137         for k in kwargs:
    138             if k in legacy_kwargs:
--> 139                 raise ValueError(
    140                     f"{k} is deprecated in the new TF-Keras optimizer, please "
    141                     "check the docstring for valid arguments, or use the "

ValueError: decay is deprecated in the new TF-Keras optimizer, please check the docstring for valid arguments, or use the legacy optimizer, e.g., tf.keras.optimizers.legacy.Adam.

## === cell 23
predict_createSubmission(class_indices)

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2808298795.py in <cell line: 0>()
----> 1 predict_createSubmission(class_indices)

NameError: name 'class_indices' is not defined
