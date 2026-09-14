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

0.7333452683918757

# 6. Current score

-0.0544

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the TensorFlow/Keras import errors by switching from `tensorflow.python.keras.*` to the stable `tf.keras` API, which resolves the `MessageFactory/GetPrototype` and `BatchNormalization` import issues in this environment. I also restore missing/incorrect variable definitions and make prediction use the non-deprecated `model.predict(...)` while keeping the same model logic. Finally, I ensure the inference path works even when `TRAINING=False` by loading available weights if present and always writing a correctly formatted `submission.csv` with `id_code,diagnosis`. These changes are execution/stability focused and should let you obtain a valid Kaggle score (and then iterate toward the target if needed).'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing TensorFlow to use the pure-Python protobuf implementation before importing TF, which is a common Kaggle workaround for this exact error. I also make the weight-loading path robust by checking a couple of likely input locations so the model doesn’t run with random weights (which explains the 0.0 score) when pretrained competition weights are available. Finally, I keep the model/training logic identical and ensure the script always writes a correctly formatted `submission.csv` with `id_code,diagnosis`.'
- What this solution (achieved 0.0) has done: 'You’re currently crashing at the TensorFlow import due to a protobuf/TensorFlow incompatibility (`MessageFactory.GetPrototype`), so the pipeline never trains or predicts, which explains the 0.0 score. I fix this by forcing a compatible protobuf implementation *before any TF-related imports* and by importing `cv2` early so preprocessing doesn’t fail later. To move the score toward the target (and away from “all-random” predictions), I also robustly look for the pretrained weights directory by scanning `/kaggle/input/*` for a matching dataset name, without changing the model/training logic. Finally, I keep the submission formatting identical but add a strict sanity check to ensure the output rows align with `test.csv`.'
- What this solution (achieved 0.0) has done: 'You’re failing at `import tensorflow as tf` with a protobuf incompatibility, so the pipeline never reaches inference and you end up with a useless (or missing) submission; I fix that by forcing the pure-Python protobuf implementation early and (critically) also forcing the Python “api” implementation, which is the common missing piece for this exact `MessageFactory.GetPrototype` crash. Next, because `TRAINING=False`, your score remain ~0 if no pretrained weights are found; I keep the same model and prediction logic but make weight discovery robust by scanning all `/kaggle/input/**` subfolders for the expected `*_weights.hdf5` file and loading it if present. Finally, I keep submission formatting identical but add a fallback to produce a valid, non-empty `submission.csv` even if weights truly don’t exist (so you always get a valid file), while still preferring loaded weights to move score toward the target.'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by ensuring the protobuf environment variables are set before any library imports and by adding a safe fallback to force the pure-Python protobuf implementation at runtime if the first TF import fails. I also remove unused TensorFlow callbacks that could trigger extra imports and keep the rest of the model/training/inference logic unchanged. Finally, I keep the submission formatting identical but add a small safety guard to always produce a valid `submission.csv` even if TF cannot be loaded (so you don’t end up with a broken/empty submission again), while still preferring real model predictions when TF works to move the score up from 0.0.'
- What this solution (achieved 0.04576) has done: 'Your run fails before training/inference because TensorFlow crashes on import with a protobuf incompatibility (`MessageFactory.GetPrototype`), which prevents any real predictions and leads to a 0.0 score. I fix this by (1) ensuring the protobuf settings are applied before *any* TF-related import, (2) proactively downgrading the protobuf runtime to the compatible 3.20.x series using the offline wheels available in `/kaggle/input` (a common Kaggle-only fix), and (3) adding a strict TF import fallback that retries after the fix. With TensorFlow successfully importing, your existing model/weight-loading/prediction logic run unchanged and produce a proper `submission.csv`, which should increase score substantially toward the target.'
- What this solution (achieved -0.0544) has done: 'I fix the TensorFlow/protobuf crash by ensuring protobuf is made compatible *before* importing TensorFlow, and by forcing a runtime restart of the protobuf module if we install a different wheel (otherwise the old incompatible version stays in-memory). This unblocks the existing TF/Keras pipeline so it can actually load weights and run inference rather than falling back to a dummy constant prediction (which explains the very low kappa). I also make the protobuf wheel search slightly more robust (including common offline wheel locations) while keeping your model/training/prediction logic unchanged. Finally, I keep the submission formatting identical and always write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_PYTHON_API", "python")
os.environ.setdefault("PYTHONHASHSEED", "0")

import math
import sys
import glob
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(0)

TRAINING = False



## === cell 1
INPUT_DIR = "../input/aptos2019-blindness-detection"
if not os.path.exists(INPUT_DIR):
    alt = "/kaggle/input/aptos2019-blindness-detection"
    if os.path.exists(alt):
        INPUT_DIR = alt

TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train_images")
TEST_IMG_DIR = os.path.join(INPUT_DIR, "test_images")



## === cell 2
import cv2


def _ensure_compatible_protobuf():
    """
    Bugfix: TF import fails with AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
    Root cause is typically an incompatible protobuf runtime already installed.
    We try to install protobuf==3.20.* from offline wheels if available (no internet).
    """
    try:
        from google.protobuf import __version__ as pb_ver  # noqa

        if str(pb_ver).startswith("3.20."):
            return True
    except Exception:
        pass

    wheel_candidates = []
    search_patterns = [
        "/kaggle/input/**/protobuf-3.20*.whl",
        "/kaggle/input/**/protobuf-3.19*.whl",
        "/kaggle/input/**/protobuf-3.20*.tar.gz",
        "/kaggle/input/**/protobuf-3.19*.tar.gz",
        "../input/**/protobuf-3.20*.whl",
        "../input/**/protobuf-3.19*.whl",
        "../input/**/protobuf-3.20*.tar.gz",
        "../input/**/protobuf-3.19*.tar.gz",
    ]
    for pattern in search_patterns:
        wheel_candidates.extend(glob.glob(pattern, recursive=True))

    if not wheel_candidates:
        return False

    def _rank(p):
        b = os.path.basename(p)
        return (
            0 if "3.20" in b else 1,
            0 if b.endswith(".whl") else 1,
            b,
        )

    wheel_candidates = sorted(set(wheel_candidates), key=_rank)

    try:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            "--upgrade",
            wheel_candidates[0],
        ]
        subprocess.check_call(cmd)
        return True
    except Exception:
        return False


_ensure_compatible_protobuf()

import importlib

for mod in list(sys.modules.keys()):
    if mod.startswith("google.protobuf"):
        del sys.modules[mod]
importlib.invalidate_caches()

tf = None
_tf_import_error = None

try:
    import tensorflow as tf  # noqa: F401
except Exception as e:
    _tf_import_error = e
    tf = None

if tf is None:
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_PYTHON_API"] = "python"

    for mod in list(sys.modules.keys()):
        if mod.startswith("google.protobuf"):
            del sys.modules[mod]
    importlib.invalidate_caches()

    try:
        import tensorflow as tf  # noqa: F401

        _tf_import_error = None
    except Exception as e2:
        _tf_import_error = e2
        tf = None

IMG_SIZE = 224
NB_CHANNELS = 3
NB_CLASSES = 5  # 0, 1, 2, 3, 4
BATCH_SIZE = 32
TEST_BATCH_SIZE = 1

if tf is not None:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
    from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
    from tensorflow.keras.applications.resnet50 import ResNet50

    print("TensorFlow version:", tf.__version__)
else:
    print("WARNING: TensorFlow failed to import; will fall back to a dummy submission.")
    print("TF import error was:", repr(_tf_import_error))



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)

print("Number of train samples: ", train.shape[0])
print("Number of test samples: ", test.shape[0])



## === cell 4
train["id_code"] = train["id_code"].apply(lambda x: str(x) + ".png")
test["id_code"] = test["id_code"].apply(lambda x: str(x) + ".png")
train["diagnosis"] = train["diagnosis"].astype(str)



## === cell 5
"""
crops black parts around the image (intensity is <= tol)
"""


def crop_image(img, tol=10):

    def crop_image_1(img2d):
        mask = img2d > tol
        return img2d[np.ix_(mask.any(1), mask.any(0))]

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
    img = crop_image(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), IMG_SIZE / 10), -4, 128)
    return img




## === cell 6
if tf is not None:
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        horizontal_flip=True,
        preprocessing_function=preprocess_image,
    )

    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train,
        directory=TRAIN_IMG_DIR,
        x_col="id_code",
        y_col="diagnosis",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        target_size=(IMG_SIZE, IMG_SIZE),
        subset="training",
        shuffle=True,
    )

    val_gen = train_datagen.flow_from_dataframe(
        dataframe=train,
        directory=TRAIN_IMG_DIR,
        x_col="id_code",
        y_col="diagnosis",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        target_size=(IMG_SIZE, IMG_SIZE),
        subset="validation",
        shuffle=True,
    )

    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255, preprocessing_function=preprocess_image
    )

    test_gen = test_datagen.flow_from_dataframe(
        dataframe=test,
        directory=TEST_IMG_DIR,
        x_col="id_code",
        batch_size=TEST_BATCH_SIZE,
        class_mode=None,
        target_size=(IMG_SIZE, IMG_SIZE),
        shuffle=False,
    )



## === cell 7
MODEL_NAME = "conv1"

NB_WARMUP_EPOCHS = 2
NB_EPOCHS = 30
INITIAL_LR = 1e-3

weights_path_template = os.path.join(
    "../input/aptos-2019-conv1-weights/", "{}_weights.hdf5"
)
log_path_template = os.path.join("logs/", "{}_training_log.csv")



## === cell 8
os.makedirs("weights", exist_ok=True)
os.makedirs("logs", exist_ok=True)



## === cell 9
"""
ResNet50 based model
"""
if tf is not None:

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
"""
simple CNN
"""
if tf is not None:

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




## === cell 11
"""
returns model
"""
if tf is not None:

    def _scan_kaggle_input_for_weights(fname: str):
        base = "/kaggle/input"
        out = []
        if not os.path.isdir(base):
            return out
        for root, dirs, files in os.walk(base):
            if fname in files:
                out.append(os.path.join(root, fname))
        return out

    def _candidate_weight_paths(name: str):
        fname = f"{name}_weights.hdf5"
        candidates = [
            weights_path_template.format(name),
            os.path.join("/kaggle/input/aptos-2019-conv1-weights", fname),
            os.path.join("../input/aptos-2019-conv1-weights", fname),
            os.path.join("/kaggle/input", "aptos-2019-conv1-weights", fname),
            os.path.join("weights", fname),
        ]
        candidates.extend(_scan_kaggle_input_for_weights(fname))

        seen = set()
        out = []
        for p in candidates:
            if p not in seen:
                out.append(p)
                seen.add(p)
        return out

    def get_model(name, input_shape, nb_out):

        models = {
            "resnet50": get_resnet50,
            "conv1": get_conv1,
        }

        if name not in models:
            print(f"No model named '{name}'")
            return None

        model = models[name](input_shape, nb_out)

        loaded = False
        for wp in _candidate_weight_paths(name):
            if os.path.isfile(wp):
                try:
                    model.load_weights(wp)
                    print(f"loaded model from {wp}")
                    loaded = True
                    break
                except Exception as e:
                    print(f"found weights at {wp} but failed to load: {e}")

        if not loaded:
            print(
                "weights not found in any candidate location (will run with randomly initialized weights):\n"
                + "\n".join(_candidate_weight_paths(name))
            )

        return model




## === cell 12
"""
trains a ResNet50-based model
"""
if tf is not None:

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

        mc = ModelCheckpoint(
            weights_path, monitor="val_loss", save_best_only=True, verbose=1
        )
        cl = CSVLogger(log_path)

        step_size_train = max(1, train_generator.n // train_generator.batch_size)
        step_size_val = max(1, val_generator.n // val_generator.batch_size)

        model.fit(
            train_generator,
            steps_per_epoch=step_size_train,
            validation_data=val_generator,
            validation_steps=step_size_val,
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

        mc = ModelCheckpoint(
            weights_path, monitor="val_loss", save_best_only=True, verbose=1
        )
        cl = CSVLogger(log_path)

        model.fit(
            train_generator,
            steps_per_epoch=step_size_train,
            validation_data=val_generator,
            validation_steps=step_size_val,
            epochs=NB_WARMUP_EPOCHS,
            callbacks=[mc, cl],
            verbose=1,
        )




## === cell 13
"""
trains the simple CNN
"""
if tf is not None:

    def train_conv1(model, train_generator, val_generator, weights_path, log_path):

        metrics_list = ["accuracy"]
        optimizer = Adam(learning_rate=INITIAL_LR)

        model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=metrics_list
        )

        mc = ModelCheckpoint(
            weights_path, monitor="val_loss", save_best_only=True, verbose=1
        )
        cl = CSVLogger(log_path)

        step_size_train = max(1, train_generator.n // train_generator.batch_size)
        step_size_val = max(1, val_generator.n // val_generator.batch_size)

        model.fit(
            train_generator,
            steps_per_epoch=step_size_train,
            validation_data=val_generator,
            validation_steps=step_size_val,
            epochs=NB_EPOCHS,
            callbacks=[mc, cl],
            verbose=1,
        )




## === cell 14
if tf is not None:

    def train_model(name, input_shape, nb_out, train_generator, val_generator):
        model = get_model(name, input_shape, nb_out)

        trainers = {
            "resnet50": train_resnet50,
            "conv1": train_conv1,
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




## === cell 15
if tf is not None and TRAINING:
    train_model(
        MODEL_NAME, (IMG_SIZE, IMG_SIZE, NB_CHANNELS), NB_CLASSES, train_gen, val_gen
    )



## === cell 16
if tf is not None:
    model = get_model(MODEL_NAME, (IMG_SIZE, IMG_SIZE, NB_CHANNELS), NB_CLASSES)
    if model is None:
        raise RuntimeError("Model could not be created; check MODEL_NAME and imports.")

    test_gen.reset()
    preds = model.predict(test_gen, verbose=1)
    predictions = np.argmax(preds, axis=1).astype(int)

    if len(predictions) != len(test):
        raise RuntimeError(
            f"Prediction length mismatch: got {len(predictions)} preds for {len(test)} test rows"
        )
else:
    most_common = int(train["diagnosis"].astype(int).value_counts().idxmax())
    predictions = np.full((len(test),), most_common, dtype=int)

results = pd.DataFrame(
    {
        "id_code": test["id_code"].str.replace(".png", "", regex=False),
        "diagnosis": predictions,
    }
)

results["id_code"] = results["id_code"].astype(str)
results["diagnosis"] = results["diagnosis"].astype(int).clip(0, NB_CLASSES - 1)

results.to_csv("submission.csv", index=False)

print(results.head(10))
print("Wrote submission.csv with shape:", results.shape)
print("diagnosis value counts:\n", results["diagnosis"].value_counts().sort_index())
