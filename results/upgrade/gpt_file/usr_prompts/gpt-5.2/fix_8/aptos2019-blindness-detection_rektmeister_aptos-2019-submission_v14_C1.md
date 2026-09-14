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

0.8348565227449207

# 6. Current score

-0.00171

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the import/runtime issues preventing the data generators and TensorFlow/Keras from loading by switching to the built-in `tensorflow.keras.preprocessing.image.ImageDataGenerator` and forcing the pure-Python protobuf implementation to avoid the `MessageFactory.GetPrototype` crash. Then I ensure the generator cell completes so `test_gen` is defined for inference, and add a small, safe fallback to the alternate dataset path in case the expected `../input/aptos2019-blindness-detection/...` path is not present. Finally, I keep the exact same model/inference logic, but make the script always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow/protobuf crash causing the `MessageFactory.GetPrototype` error by importing TensorFlow only after forcing the pure-Python protobuf implementation and by clearing any already-imported `google.protobuf` modules before the TF import. I also make the input-path detection robust for this environment by including `/kaggle/input/...` and `/kaggle/data/...` candidates (without changing how files are read once found). These changes are runtime/stability fixes; they keep the same generators, model, and prediction logic, but they should move the score up from 0.0 by allowing the script to actually run and produce non-empty predictions/submission. Finally, I ensure the submission CSV is always written with the exact required columns and correct `id_code` formatting.'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation *before any protobuf/TensorFlow import*, and by preventing accidental prior protobuf imports from lingering in `sys.modules`. I also make the TensorFlow import happen only once (to avoid re-triggering the same crash later) and keep all generators/model code unchanged in behavior. Finally, I add a small safeguard so inference still writes a valid `submission.csv` even if the weight file is missing (it still run, but score be poor without weights). These are runtime/stability fixes that should move the score up from 0.0 by enabling the notebook to actually run end-to-end and generate non-empty predictions.'
- What this solution (achieved -0.02735) has done: 'I fix the runtime crash coming from the TensorFlow/protobuf incompatibility that currently stops execution at the generator creation step, because that is why you are getting a 0.0 score (no valid predictions). The fix is to force the pure-Python protobuf implementation *and* prevent any `google.protobuf` C++ backend from being used by setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before imports, and by importing protobuf once (python impl) before importing TensorFlow. I also add a small compatibility fallback: if TensorFlow still fails to import in this environment, the script still write a valid `submission.csv` (score be poor, but it run end-to-end). These changes keep the same data pipeline, model, and prediction post-processing logic; they only unblock execution so your score can move up toward the target.'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow/protobuf crash that stops execution in the generator/model steps by ensuring the pure-Python protobuf implementation is enforced *before any protobuf/TensorFlow import* and by removing the “pre-import protobuf then import TF” pattern that triggers the `MessageFactory.GetPrototype` mismatch in this environment. I keep your model, generators, preprocessing, and prediction-to-class logic unchanged, only adjusting import order and module cleanup so the pipeline runs end-to-end. This should move the score up substantially from the current negative kappa (which is consistent with random/fallback predictions when TF fails) toward your target by allowing the intended DenseNet121 weights to load and real predictions to be made. The script still always write a valid `submission.csv` with the required columns.'
- What this solution (achieved -0.00171) has done: 'I fix the TensorFlow/protobuf crash that stops execution at generator creation by enforcing the pure-Python protobuf backend before any TensorFlow import and by importing TensorFlow in a fresh process state (clearing any previously loaded protobuf modules). I also ensure `TF_AVAILABLE` correctly reflects whether TensorFlow actually imported, so downstream cells don’t reference undefined generators/models. These are runtime/stability fixes that preserve your exact model, preprocessing, generators, and prediction post-processing; they should move your score up from 0.0 by allowing the intended DenseNet121 weights (if present) to load and real predictions to be generated. The script still always write a valid `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os
import math
import sys
import numpy as np
import pandas as pd

os.environ["PYTHONHASHSEED"] = "0"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf"):
        del sys.modules[m]

TRAINING = False
np.random.seed(0)



## === cell 1
WEIGHTS_DIR_CANDIDATES = [
    "/kaggle/input/aptos-2019-densenet121-weights",
    "/kaggle/input/aptos2019-densenet121-weights",
    "/kaggle/input/aptos2019-blindness-detection",
    "../input/aptos-2019-densenet121-weights",
    "../input/aptos2019-densenet121-weights",
    "../input/aptos2019-blindness-detection",
]
weights_dir = None
for d in WEIGHTS_DIR_CANDIDATES:
    if os.path.isdir(d):
        weights_dir = d
        break

if weights_dir is None:
    print(
        "No weights directory found in expected locations; will run with random weights if training is off."
    )
else:
    print("Using weights dir:", weights_dir)
    try:
        print("Contents:", os.listdir(weights_dir)[:20])
    except Exception as e:
        print("Could not list weights dir:", e)



## === cell 2
INPUT_ROOT_CANDIDATES = [
    "/kaggle/input/aptos2019-blindness-detection",
    "/kaggle/input/aptos2019-blindness-detection/aptos2019-blindness-detection",
    "/kaggle/data/aptos2019-blindness-detection",
    "/kaggle/data/aptos2019-blindness-detection/aptos2019-blindness-detection",
    "../input/aptos2019-blindness-detection",
    "../input/aptos2019-blindness-detection/aptos2019-blindness-detection",
]
INPUT_ROOT = None
for p in INPUT_ROOT_CANDIDATES:
    if os.path.isfile(os.path.join(p, "train.csv")) and os.path.isfile(
        os.path.join(p, "test.csv")
    ):
        INPUT_ROOT = p
        break

if INPUT_ROOT is None:
    INPUT_ROOT = "../input/aptos2019-blindness-detection"

train = pd.read_csv(os.path.join(INPUT_ROOT, "train.csv"))
test = pd.read_csv(os.path.join(INPUT_ROOT, "test.csv"))

print("Using INPUT_ROOT:", INPUT_ROOT)
print("Number of train samples: ", train.shape[0])
print("Number of test samples: ", test.shape[0])



## === cell 3
train["id_code"] = train["id_code"].apply(lambda x: str(x) + ".png")
test["id_code"] = test["id_code"].apply(lambda x: str(x) + ".png")
train["diagnosis"] = train["diagnosis"].astype(str)

label_cols = ["lbl_0", "lbl_1", "lbl_2", "lbl_3", "lbl_4"]
label_mat = np.zeros((train.shape[0], len(label_cols)), dtype=np.int32)

for i in range(train.shape[0]):
    for j in range(int(train["diagnosis"][i]) + 1):
        label_mat[i, j] = 1

train = pd.concat([train, pd.DataFrame(label_mat, columns=label_cols)], axis=1)
print(train.head(10))



## === cell 4
import cv2

IMG_SIZE = 224
NB_CHANNELS = 3
NB_CLASSES = 5  # 0, 1, 2, 3, 4
BATCH_SIZE = 32
TEST_BATCH_SIZE = 1



## === cell 5
"""
crops black parts around the image (intensity is <= tol)
"""


def crop_image(img, tol=10):
    def crop_image_1(img_2d):
        mask = img_2d > tol
        return img_2d[np.ix_(mask.any(1), mask.any(0))]

    if img.ndim == 2:
        return crop_image_1(img)

    elif img.ndim == 3:
        try:
            img_cpy = img.copy()
            h, w, _ = img.shape
            img1 = cv2.resize(crop_image_1(img[:, :, 0]), (w, h))
            img2 = cv2.resize(crop_image_1(img[:, :, 1]), (w, h))
            img3 = cv2.resize(crop_image_1(img[:, :, 2]), (w, h))

            img[:, :, 0] = img1
            img[:, :, 1] = img2
            img[:, :, 2] = img3
        except Exception:
            return img_cpy

        return img


"""
crops black parts and enhances image (Ben Graham's method)
"""


def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_image(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), IMG_SIZE / 10), -4, 128)
    return img




## === cell 6
TF_AVAILABLE = True
try:
    import tensorflow as tf  # noqa: F401
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
except Exception as e:
    TF_AVAILABLE = False
    print(
        "ERROR: TensorFlow import failed; will write a fallback submission. Error was:",
        repr(e),
    )

train_images_dir = os.path.join(INPUT_ROOT, "train_images")
test_images_dir = os.path.join(INPUT_ROOT, "test_images")

if not os.path.isdir(train_images_dir):
    alt = os.path.join(INPUT_ROOT, "aptos2019-blindness-detection", "train_images")
    if os.path.isdir(alt):
        train_images_dir = alt

if not os.path.isdir(test_images_dir):
    alt = os.path.join(INPUT_ROOT, "aptos2019-blindness-detection", "test_images")
    if os.path.isdir(alt):
        test_images_dir = alt

print("Train images dir:", train_images_dir, "exists:", os.path.isdir(train_images_dir))
print("Test images dir:", test_images_dir, "exists:", os.path.isdir(test_images_dir))

if TF_AVAILABLE:
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        horizontal_flip=True,
        preprocessing_function=preprocess_image,
    )

    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train,
        directory=train_images_dir,
        x_col="id_code",
        y_col=label_cols,
        batch_size=BATCH_SIZE,
        class_mode="other",
        target_size=(IMG_SIZE, IMG_SIZE),
        subset="training",
        shuffle=True,
    )

    val_gen = train_datagen.flow_from_dataframe(
        dataframe=train,
        directory=train_images_dir,
        x_col="id_code",
        y_col=label_cols,
        batch_size=BATCH_SIZE,
        class_mode="other",
        target_size=(IMG_SIZE, IMG_SIZE),
        subset="validation",
        shuffle=True,
    )

    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255, preprocessing_function=preprocess_image
    )

    test_gen = test_datagen.flow_from_dataframe(
        dataframe=test,
        directory=test_images_dir,
        x_col="id_code",
        batch_size=TEST_BATCH_SIZE,
        class_mode=None,
        target_size=(IMG_SIZE, IMG_SIZE),
        shuffle=False,
    )



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 7
if TF_AVAILABLE:
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
    from tensorflow.keras.applications.densenet import DenseNet121

MODEL_NAME = "densenet121"

NB_WARMUP_EPOCHS = 2
NB_EPOCHS = 30
INITIAL_LR = 1e-3

_default_weights_dir = "../input/aptos-2019-densenet121-weights/"
if weights_dir is not None:
    _default_weights_dir = weights_dir

weights_path_template = os.path.join(_default_weights_dir, "{}_weights.hdf5")
log_path_template = os.path.join("logs/", "{}_training_log.csv")



## === cell 8
os.makedirs("weights", exist_ok=True)
os.makedirs("logs", exist_ok=True)



## === cell 9
"""
ResNet50 based model
"""
if TF_AVAILABLE:

    def get_resnet50(input_shape, nb_out):
        inputs = Input(shape=input_shape)
        base_model = ResNet50(
            weights="imagenet", include_top=False, input_tensor=inputs
        )

        x = GlobalAveragePooling2D()(base_model.output)
        x = Dropout(0.5)(x)

        x = Dense(2048, activation="relu")(x)
        x = Dropout(0.5)(x)

        x = Dense(1024, activation="relu")(x)
        x = Dropout(0.5)(x)

        output = Dense(nb_out, activation="softmax", name="final_output")(x)
        model = Model(inputs, output)
        return model




## === cell 10
if TF_AVAILABLE:

    def get_densenet121(input_shape, nb_out):
        inputs = Input(shape=input_shape)
        base_model = DenseNet121(weights=None, include_top=False, input_tensor=inputs)

        x = GlobalAveragePooling2D()(base_model.output)
        x = Dropout(0.5)(x)

        x = Dense(1024, activation="relu")(x)
        output = Dense(nb_out, activation="sigmoid")(x)

        model = Model(inputs, output)
        return model




## === cell 11
"""
simple CNN (conv1)
"""
if TF_AVAILABLE:

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
simple CNN v2 (conv2)
"""
if TF_AVAILABLE:

    def get_conv2(input_shape, nb_out):
        inputs = Input(shape=input_shape)

        x = Conv2D(32, (7, 7), activation="relu")(inputs)
        x = MaxPooling2D((2, 2))(x)
        x = BatchNormalization()(x)

        x = Conv2D(64, (5, 5), activation="relu")(x)
        x = MaxPooling2D((2, 2))(x)
        x = BatchNormalization()(x)

        x = Conv2D(128, (3, 3), activation="relu")(x)
        x = MaxPooling2D((2, 2))(x)
        x = BatchNormalization()(x)

        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)

        x = Dense(1024, activation="relu")(x)
        x = Dropout(0.5)(x)

        output = Dense(nb_out, activation="softmax")(x)
        model = Model(inputs, output)
        return model




## === cell 13
"""
returns model
"""
if TF_AVAILABLE:

    def get_model(name, input_shape, nb_out):
        models = {
            "resnet50": get_resnet50,
            "conv1": get_conv1,
            "conv2": get_conv2,
            "densenet121": get_densenet121,
        }

        if name not in models:
            print(f"No model named '{name}'")
            return None

        model = models[name](input_shape, nb_out)

        weights_path = weights_path_template.format(name)
        if os.path.isfile(weights_path):
            model.load_weights(weights_path)
            print(f"loaded model weights from {weights_path}")
        else:
            print(f"WARNING: weights file not found at {weights_path}.")
            print(
                "If TRAINING=False and no weights are available, predictions will be random and score will be poor."
            )

        return model




## === cell 14
"""
trains a ResNet50-based model
"""
if TF_AVAILABLE:

    def train_resnet50(model, train_generator, val_generator, weights_path, log_path):
        for i in range(len(model.layers)):
            model.layers[i].trainable = False
        for i in range(-5, 0):
            model.layers[i].trainable = True

        metrics_list = ["accuracy"]
        optimizer = Adam(lr=INITIAL_LR)
        model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=metrics_list
        )

        mc = ModelCheckpoint(
            weights_path, monitor="val_loss", save_best_only=True, verbose=1
        )
        es = EarlyStopping(
            monitor="val_loss", mode="min", restore_best_weights=True, verbose=1
        )
        cl = CSVLogger(log_path)

        STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
        STEP_SIZE_VAL = val_generator.n // val_generator.batch_size

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

        optimizer = Adam(lr=INITIAL_LR)
        model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=metrics_list
        )

        mc = ModelCheckpoint(
            weights_path, monitor="val_loss", save_best_only=True, verbose=1
        )
        es = EarlyStopping(
            monitor="val_loss", mode="min", restore_best_weights=True, verbose=1
        )
        cl = CSVLogger(log_path)

        STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
        STEP_SIZE_VAL = val_generator.n // val_generator.batch_size

        model.fit(
            train_generator,
            steps_per_epoch=STEP_SIZE_TRAIN,
            validation_data=val_generator,
            validation_steps=STEP_SIZE_VAL,
            epochs=NB_WARMUP_EPOCHS,
            callbacks=[mc, cl],
            verbose=1,
        )




## === cell 15
if TF_AVAILABLE:

    def train_densenet121(
        model, train_generator, val_generator, weights_path, log_path
    ):
        metrics_list = ["accuracy"]
        optimizer = Adam(lr=INITIAL_LR)
        model.compile(
            optimizer=optimizer, loss="binary_crossentropy", metrics=metrics_list
        )

        mc = ModelCheckpoint(
            weights_path, monitor="val_loss", save_best_only=True, verbose=1
        )
        es = EarlyStopping(
            monitor="val_loss", mode="min", restore_best_weights=True, verbose=1
        )
        cl = CSVLogger(log_path)

        STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
        STEP_SIZE_VAL = val_generator.n // val_generator.batch_size

        model.fit(
            train_generator,
            steps_per_epoch=STEP_SIZE_TRAIN,
            validation_data=val_generator,
            validation_steps=STEP_SIZE_VAL,
            epochs=NB_EPOCHS,
            callbacks=[mc, cl],
            verbose=1,
        )




## === cell 16
"""
trains the simple CNN
"""
if TF_AVAILABLE:

    def train_conv1(model, train_generator, val_generator, weights_path, log_path):
        metrics_list = ["accuracy"]
        optimizer = Adam(lr=INITIAL_LR)
        model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=metrics_list
        )

        mc = ModelCheckpoint(
            weights_path, monitor="val_loss", save_best_only=True, verbose=1
        )
        es = EarlyStopping(
            monitor="val_loss", mode="min", restore_best_weights=True, verbose=1
        )
        cl = CSVLogger(log_path)

        STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
        STEP_SIZE_VAL = val_generator.n // val_generator.batch_size

        model.fit(
            train_generator,
            steps_per_epoch=STEP_SIZE_TRAIN,
            validation_data=val_generator,
            validation_steps=STEP_SIZE_VAL,
            epochs=NB_EPOCHS,
            callbacks=[mc, cl],
            verbose=1,
        )




## === cell 17
if TF_AVAILABLE:

    def train_conv2(model, train_generator, val_generator, weights_path, log_path):
        metrics_list = ["accuracy"]
        optimizer = Adam(lr=INITIAL_LR)
        model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=metrics_list
        )

        mc = ModelCheckpoint(
            weights_path, monitor="val_loss", save_best_only=True, verbose=1
        )
        es = EarlyStopping(
            monitor="val_loss", mode="min", restore_best_weights=True, verbose=1
        )
        cl = CSVLogger(log_path)

        STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
        STEP_SIZE_VAL = val_generator.n // val_generator.batch_size

        model.fit(
            train_generator,
            steps_per_epoch=STEP_SIZE_TRAIN,
            validation_data=val_generator,
            validation_steps=STEP_SIZE_VAL,
            epochs=NB_EPOCHS,
            callbacks=[mc, cl],
            verbose=1,
        )




## === cell 18
if TF_AVAILABLE:

    def train_model(name, input_shape, nb_out, train_generator, val_generator):
        model = get_model(name, input_shape, nb_out)

        trainers = {
            "resnet50": train_resnet50,
            "conv1": train_conv1,
            "conv2": train_conv2,
            "densenet121": train_densenet121,
        }

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




## === cell 19
if TF_AVAILABLE and TRAINING:
    train_model(
        MODEL_NAME, (IMG_SIZE, IMG_SIZE, NB_CHANNELS), NB_CLASSES, train_gen, val_gen
    )



## === cell 20
if TF_AVAILABLE:
    model = get_model(MODEL_NAME, (IMG_SIZE, IMG_SIZE, NB_CHANNELS), NB_CLASSES)
    test_gen.reset()

    preds = model.predict(test_gen, verbose=1)
    preds = preds > 0.5
    predictions = preds.astype(int).sum(axis=1) - 1
    predictions = np.clip(predictions, 0, 4).astype(int)
else:
    predictions = np.zeros((test.shape[0],), dtype=int)

results = pd.DataFrame(
    {
        "id_code": test["id_code"].str.replace(".png", "", regex=False),
        "diagnosis": predictions,
    }
)

results = results[["id_code", "diagnosis"]]
results.to_csv("submission.csv", index=False)
print(results.head())
print("Wrote submission.csv with shape:", results.shape)
print("Columns:", list(results.columns))
