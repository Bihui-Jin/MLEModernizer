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
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.0262298038148007

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the TensorFlow/Keras import errors by switching from `tensorflow.python.keras` to the supported public `tensorflow.keras` API, which resolves the `MessageFactory.GetPrototype` and missing `BatchNormalization` issues. Then I ensure the data generators are created successfully so `test_gen` exists and the pipeline runs end-to-end. Finally, I replace the placeholder random predictions with real model inference (loading existing weights if present; otherwise using the untrained model), and write `submission.csv` with the exact required columns and `id_code` format. These changes preserve your model/training logic while making the script executable and producing a valid submission file.'
- What this solution (achieved 0.00958) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation before TensorFlow is imported, which is a common Kaggle TF1/TF2 + protobuf mismatch issue. I also ensure TensorFlow/Keras is imported only after that environment setting, and make the generators/model creation robust so `submission.csv` is always produced. Finally, I make the submission `id_code` come strictly from `test.csv` (not generator filenames) to avoid any path/filename mismatch that can silently create misaligned predictions and hurt kappa; this is score-improving but does not change the model itself.'
- What this solution (achieved -0.03563) has done: 'I fix the TensorFlow/protobuf crash by moving the environment variables to the very top (before any TensorFlow/Keras-related import can occur) and by enforcing `protobuf<4`-compatible behavior via the pure-Python implementation plus version sanity prints. Then I make the Keras generator and prediction steps deterministic and correctly sized (using `ceil` steps and trimming predictions to exactly `len(test)`), which is score-neutral but prevents subtle misalignment/length issues. Finally, I ensure the submission `id_code` comes from `test.csv` (not `sample_submission.csv`) and is written with the exact required columns and `.csv` suffix, which can improve kappa if any prior row-order mismatch existed.'
- What this solution (achieved 0.0) has done: 'I fix two execution blockers: the TensorFlow/protobuf crash and the generator dtype crash during rescaling. First, I force the pure-Python protobuf implementation *before any TensorFlow import* and delay TensorFlow-related imports until after that, which resolves the `MessageFactory.GetPrototype` error in this environment. Second, I make `preprocess_image` return `float32` in `[0,1]` (not `uint8`) so that `ImageDataGenerator(rescale=...)` doesn’t try to multiply into a `uint8` array and fail. These changes are score-neutral (they don’t alter model architecture/training semantics) but ensure the notebook runs end-to-end and writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

TRAINING = False



## === cell 1
TRAIN_CSV_PATH = "../input/aptos2019-blindness-detection/train.csv"
TEST_CSV_PATH = "../input/aptos2019-blindness-detection/test.csv"
TRAIN_IMG_DIR = "../input/aptos2019-blindness-detection/train_images/"
TEST_IMG_DIR = "../input/aptos2019-blindness-detection/test_images/"



## === cell 2
for p in [TRAIN_CSV_PATH, TEST_CSV_PATH, TRAIN_IMG_DIR, TEST_IMG_DIR]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Expected path not found: {p}")



## === cell 3
train = pd.read_csv(TRAIN_CSV_PATH)
test = pd.read_csv(TEST_CSV_PATH)

print("Number of train samples: ", train.shape[0])
print("Number of test samples: ", test.shape[0])
print(train.head())
print(test.head())



## === cell 4
train["id_code"] = train["id_code"].astype(str)
test["id_code"] = test["id_code"].astype(str)

train["id_code_png"] = train["id_code"].apply(lambda x: x + ".png")
test["id_code_png"] = test["id_code"].apply(lambda x: x + ".png")

train["diagnosis"] = train["diagnosis"].astype(str)



## === cell 5
import cv2

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("TensorFlow version:", tf.__version__)
try:
    import google.protobuf

    print("protobuf version:", google.protobuf.__version__)
except Exception as e:
    print("Could not import protobuf version:", repr(e))

IMG_SIZE = 224
NB_CHANNELS = 3
NB_CLASSES = 5  # 0, 1, 2, 3, 4
BATCH_SIZE = 32
TEST_BATCH_SIZE = 1



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 6
"""
crops black parts around the image (intensity is <= tol)
"""


def crop_image(img, tol=10):
    def crop_image_1(img2d):
        mask = img2d > tol
        if mask.any():
            return img2d[np.ix_(mask.any(1), mask.any(0))]
        return img2d

    if img.ndim == 2:
        return crop_image_1(img)

    elif img.ndim == 3:
        h, w, _ = img.shape
        img1 = cv2.resize(crop_image_1(img[:, :, 0]), (w, h))
        img2 = cv2.resize(crop_image_1(img[:, :, 1]), (w, h))
        img3 = cv2.resize(crop_image_1(img[:, :, 2]), (w, h))

        img = img.copy()
        img[:, :, 0] = img1
        img[:, :, 1] = img2
        img[:, :, 2] = img3
        return img

    return img


"""
crops black parts and enhances image (Ben Graham's method)

Bugfix: This function must be compatible with ImageDataGenerator(rescale=1/255).
Keras applies preprocessing_function BEFORE rescale, and expects the output to remain
a float array it can multiply by `rescale`. Returning uint8 causes:
UFuncTypeError: cannot cast multiply output to uint8.

So: convert to uint8 for OpenCV ops, then return float32 in [0,1] so rescale is safe.
"""


def preprocess_image(img):
    if img is None:
        return img

    if img.dtype != np.uint8:
        if np.max(img) <= 1.0:
            img_u8 = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img_u8 = img.clip(0, 255).astype(np.uint8)
    else:
        img_u8 = img

    img_u8 = crop_image(img_u8)
    img_u8 = cv2.resize(img_u8, (IMG_SIZE, IMG_SIZE))
    img_u8 = cv2.addWeighted(
        img_u8, 4, cv2.GaussianBlur(img_u8, (0, 0), IMG_SIZE / 10), -4, 128
    )

    return img_u8.astype(np.float32) / 255.0




## === cell 7
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    horizontal_flip=True,
    preprocessing_function=preprocess_image,
)

train_gen = train_datagen.flow_from_dataframe(
    dataframe=train,
    directory=TRAIN_IMG_DIR,
    x_col="id_code_png",
    y_col="diagnosis",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    target_size=(IMG_SIZE, IMG_SIZE),
    subset="training",
    shuffle=True,
    seed=42,
)

val_gen = train_datagen.flow_from_dataframe(
    dataframe=train,
    directory=TRAIN_IMG_DIR,
    x_col="id_code_png",
    y_col="diagnosis",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    target_size=(IMG_SIZE, IMG_SIZE),
    subset="validation",
    shuffle=True,
    seed=42,
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255, preprocessing_function=preprocess_image
)

test_gen = test_datagen.flow_from_dataframe(
    dataframe=test,
    directory=TEST_IMG_DIR,
    x_col="id_code_png",
    batch_size=TEST_BATCH_SIZE,
    class_mode=None,
    target_size=(IMG_SIZE, IMG_SIZE),
    shuffle=False,
)



## === cell 8
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    BatchNormalization,
    Conv2D,
    MaxPooling2D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications.resnet50 import ResNet50

MODEL_NAME = "conv1"

NB_WARMUP_EPOCHS = 2
NB_EPOCHS = 30
INITIAL_LR = 1e-3

weights_path_template = os.path.join(
    "../input/aptos-2019-conv1-weights/", "{}_weights.hdf5"
)
log_path_template = os.path.join("logs/", "{}_training_log.csv")



## === cell 9
os.makedirs("weights", exist_ok=True)
os.makedirs("logs", exist_ok=True)

if not os.path.isdir(os.path.dirname(weights_path_template)):
    weights_path_template = os.path.join("weights", "{}_weights.hdf5")



## === cell 10
"""
ResNet50 based model
"""


def get_resnet50(input_shape, nb_out):
    inputs = Input(shape=input_shape)
    base_model = ResNet50(weights="imagenet", include_top=False, input_tensor=inputs)

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)

    x = Dense(2048, activation="relu")(x)
    x = Dropout(0.5)(x)

    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)

    output = Dense(nb_out, activation="softmax", name="final_output")(x)

    model = Model(inputs, output)
    return model




## === cell 11
"""
simple CNN
"""


def get_conv1(input_shape, nb_out):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, (7, 7), activation="relu")(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (7, 7), activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (5, 5), activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(256, (3, 3), activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(512, (3, 3), activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)

    x = Dense(2048, activation="relu")(x)
    x = Dropout(0.5)(x)

    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)

    output = Dense(nb_out, activation="softmax", name="final_output")(x)

    model = Model(inputs, output)
    return model




## === cell 12
"""
returns model
"""


def get_model(name, input_shape, nb_out):
    models = {
        "resnet50": get_resnet50,
        "conv1": get_conv1,
    }

    if name not in models:
        print(f"No model named '{name}'")
        return None

    model = models[name](input_shape, nb_out)

    weights_path = weights_path_template.format(name)
    if os.path.isfile(weights_path):
        model.load_weights(weights_path)
        print(f"loaded model from {weights_path}")
    else:
        print(f"no weights found at {weights_path}; using randomly initialized weights")

    return model




## === cell 13
"""
trains a ResNet50-based model
"""


def train_resnet50(model, train_generator, val_generator, weights_path, log_path):
    for i in range(len(model.layers)):
        model.layers[i].trainable = False

    for i in range(-5, 0):
        model.layers[i].trainable = True

    metrics_list = ["accuracy"]
    optimizer = Adam(learning_rate=INITIAL_LR)

    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=metrics_list
    )

    mc = ModelCheckpoint(weights_path, monitor="val_loss", save_best_only=True)
    es = EarlyStopping(
        monitor="val_loss", mode="min", restore_best_weights=True, verbose=1
    )
    cl = CSVLogger(log_path)

    STEP_SIZE_TRAIN = max(1, train_generator.n // train_generator.batch_size)
    STEP_SIZE_VAL = max(1, val_generator.n // val_generator.batch_size)

    model.fit(
        train_generator,
        steps_per_epoch=STEP_SIZE_TRAIN,
        validation_data=val_generator,
        validation_steps=STEP_SIZE_VAL,
        epochs=NB_WARMUP_EPOCHS,
        callbacks=[mc, cl],
        verbose=1,
    )

    train_generator.reset()
    val_generator.reset()

    for i in range(len(model.layers)):
        model.layers[i].trainable = True

    optimizer = Adam(learning_rate=INITIAL_LR)

    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=metrics_list
    )

    mc = ModelCheckpoint(weights_path, monitor="val_loss", save_best_only=True)
    es = EarlyStopping(
        monitor="val_loss", mode="min", restore_best_weights=True, verbose=1
    )
    cl = CSVLogger(log_path)

    STEP_SIZE_TRAIN = max(1, train_generator.n // train_generator.batch_size)
    STEP_SIZE_VAL = max(1, val_generator.n // val_generator.batch_size)

    model.fit(
        train_generator,
        steps_per_epoch=STEP_SIZE_TRAIN,
        validation_data=val_generator,
        validation_steps=STEP_SIZE_VAL,
        epochs=NB_WARMUP_EPOCHS,
        callbacks=[mc, cl],
        verbose=1,
    )




## === cell 14
"""
trains the simple CNN
"""


def train_conv1(model, train_generator, val_generator, weights_path, log_path):
    metrics_list = ["accuracy"]
    optimizer = Adam(learning_rate=INITIAL_LR)

    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=metrics_list
    )

    mc = ModelCheckpoint(weights_path, monitor="val_loss", save_best_only=True)
    es = EarlyStopping(
        monitor="val_loss", mode="min", restore_best_weights=True, verbose=1
    )
    cl = CSVLogger(log_path)

    STEP_SIZE_TRAIN = max(1, train_generator.n // train_generator.batch_size)
    STEP_SIZE_VAL = max(1, val_generator.n // val_generator.batch_size)

    model.fit(
        train_generator,
        steps_per_epoch=STEP_SIZE_TRAIN,
        validation_data=val_generator,
        validation_steps=STEP_SIZE_VAL,
        epochs=NB_EPOCHS,
        callbacks=[mc, cl],
        verbose=1,
    )




## === cell 15
def train_model(name, input_shape, nb_out, train_generator, val_generator):
    model = get_model(name, input_shape, nb_out)

    trainers = {"resnet50": train_resnet50, "conv1": train_conv1}

    if name not in trainers:
        print(f"No model named '{name}'")
        return

    trainers[name](
        model,
        train_generator,
        val_generator,
        weights_path_template.format(name),
        log_path_template.format(name),
    )




## === cell 16
if TRAINING:
    train_model(
        MODEL_NAME, (IMG_SIZE, IMG_SIZE, NB_CHANNELS), NB_CLASSES, train_gen, val_gen
    )



## === cell 17
model = get_model(MODEL_NAME, (IMG_SIZE, IMG_SIZE, NB_CHANNELS), NB_CLASSES)
if model is None:
    raise RuntimeError("Model could not be created.")

test_gen.reset()
STEP_SIZE_TEST = int(math.ceil(test_gen.n / test_gen.batch_size))

probs = model.predict(test_gen, steps=STEP_SIZE_TEST, verbose=1)
predictions = np.argmax(probs, axis=1).astype(int)
predictions = predictions[: len(test)]

sub = test[["id_code"]].copy()
sub["diagnosis"] = predictions.astype(int)

sub.to_csv("submission.csv", index=False)
print(sub.head(10))
print("Wrote submission.csv with shape:", sub.shape)
print(
    "Diagnosis value counts:\n",
    sub["diagnosis"].value_counts(dropna=False).sort_index(),
)
