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

0.0

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
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow import crash by forcing a protobuf 3.20.x runtime *before* importing anything that might pull in `google.protobuf`, and retrying the TF import after cleaning already-loaded protobuf modules. This unblocks the real inference path (instead of falling back to a constant-prediction dummy submission), which should substantially improve QWK toward your 0.733 target from the current -0.0544. I also ensure we always load the best available weights when `TRAINING=False` by searching common Kaggle input locations, but otherwise keep the model, preprocessing, training loops, and prediction semantics unchanged. Finally, the script always write a valid `submission.csv` with the required columns aligned to `test.csv`.'
- What this solution (achieved 0.00793) has done: 'You’re crashing on `import tensorflow as tf` due to an incompatible protobuf runtime (`MessageFactory.GetPrototype`), so the model never runs and you end up with effectively useless predictions. I make the protobuf fix deterministic by (1) forcing the pure-Python protobuf implementation early and (2) if needed, installing a compatible protobuf 3.20.* wheel from any offline wheels found under `/kaggle/input/**` and then retrying the TF import after purging already-imported `google.protobuf` modules. Once TensorFlow imports, I keep your model/training/inference logic unchanged, but I also make the `weights_path_template` point to `weights/` (a writable location) so training can save weights properly if you enable `TRAINING=True`. Finally, the script always write a correctly formatted `submission.csv` aligned to `test.csv`.'
- What this solution (achieved 0.0) has done: 'We need to unblock TensorFlow import first, because your current run stops at the protobuf `MessageFactory.GetPrototype` crash and then ends up producing effectively random/constant predictions (hence the 0.00793). The minimal reliable fix in Kaggle for this specific error is to force a protobuf 3.20.x runtime *and* force TF to use the pure-Python protobuf implementation, doing so before importing TensorFlow and purging already-imported `google.protobuf` modules. Once TF imports, I keep your model/data pipeline identical, but ensure inference actually uses trained weights by loading the best available weights file (prefer local `weights/`, otherwise scan `/kaggle/input/**`) and failing over cleanly only if none exist. Finally, I keep submission formatting the same but add a strict check that the output is aligned to `test.csv` and always writes `submission.csv`.'
- What this solution (achieved -0.00046) has done: 'The immediate blocker is the TensorFlow import crash coming from an incompatible protobuf runtime (`MessageFactory.GetPrototype`). I make the protobuf fix deterministic by forcing a protobuf 3.20.* install from offline wheels (if available) and purging already-imported `google.protobuf` modules *before* importing TensorFlow, then retrying the TF import once. Next, to move the score up from 0.0 without changing your model/training logic, I ensure inference actually uses trained weights by also loading the “best” checkpoint saved by `ModelCheckpoint` when present and by expanding the weight-file search to include both `.hdf5` and `.h5` (common variants). Finally, I keep your submission formatting but add a hard guarantee that the output is aligned to `test.csv` and always writes `submission.csv`.'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow import crash by ensuring protobuf is pinned to a TF-compatible 3.20.* **before** anything imports `google.protobuf`, and by purging already-loaded protobuf modules before retrying the TF import. This unblocks the real model inference path (instead of failing early), which is necessary to move the score up from ~0 toward your 0.733 target. I also make the TF import robust by cleanly falling back to a dummy submission only if TF still cannot import after the protobuf fix, while keeping your model/training/prediction logic unchanged. Finally, I keep the submission formatting and alignment checks, ensuring `submission.csv` is always produced.'

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


def _pip_install_offline(path):
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--no-index",
        "--upgrade",
        path,
    ]
    subprocess.check_call(cmd)


def _purge_google_protobuf_modules():
    import importlib

    for mod in list(sys.modules.keys()):
        if mod == "google" or mod.startswith("google."):
            if mod.startswith("google.protobuf"):
                del sys.modules[mod]
    importlib.invalidate_caches()


def _ensure_protobuf_320():
    """
    Bugfix for: AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
    Root cause: protobuf>=4 incompatibility with older TF builds commonly used on Kaggle.
    Solution: ensure protobuf==3.20.* is installed (offline if needed), then purge modules.
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
        "/kaggle/input/**/protobuf-3.20*.tar.gz",
        "/kaggle/input/**/protobuf-3.20*.zip",
        "../input/**/protobuf-3.20*.whl",
        "../input/**/protobuf-3.20*.tar.gz",
        "../input/**/protobuf-3.20*.zip",
    ]
    for pattern in search_patterns:
        wheel_candidates.extend(glob.glob(pattern, recursive=True))

    if not wheel_candidates:
        return False

    def _rank(p):
        b = os.path.basename(p)
        return (0 if b.endswith(".whl") else 1, b)

    wheel_candidates = sorted(set(wheel_candidates), key=_rank)

    try:
        _pip_install_offline(wheel_candidates[0])
        _purge_google_protobuf_modules()
        return True
    except Exception:
        return False


_ensure_protobuf_320()
_purge_google_protobuf_modules()

tf = None
_tf_import_error = None

try:
    import tensorflow as tf  # noqa: F401

    _tf_import_error = None
except Exception as e:
    _tf_import_error = e
    tf = None

if tf is None:
    _ensure_protobuf_320()
    _purge_google_protobuf_modules()
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

weights_path_template = os.path.join("weights", "{}_weights.hdf5")
log_path_template = os.path.join("logs", "{}_training_log.csv")


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

    def _scan_kaggle_input_for_weights(fnames):
        base = "/kaggle/input"
        out = []
        if not os.path.isdir(base):
            return out
        fnames = set(fnames)
        for root, dirs, files in os.walk(base):
            for f in files:
                if f in fnames:
                    out.append(os.path.join(root, f))
        return out

    def _candidate_weight_paths(name: str):
        fnames = [
            f"{name}_weights.hdf5",
            f"{name}_weights.h5",
            f"{name}_best.hdf5",
            f"{name}_best.h5",
            f"{name}.hdf5",
            f"{name}.h5",
        ]

        candidates = []
        for f in fnames:
            candidates.extend(
                [
                    os.path.join("weights", f),
                    os.path.join("/kaggle/input/aptos-2019-conv1-weights", f),
                    os.path.join("../input/aptos-2019-conv1-weights", f),
                    os.path.join("/kaggle/input", "aptos-2019-conv1-weights", f),
                ]
            )

        candidates.extend(_scan_kaggle_input_for_weights(fnames))

        def _rank(p):
            base = os.path.basename(p)
            is_local = (
                0 if os.path.abspath(p).startswith(os.path.abspath("weights")) else 1
            )
            exact = 0 if base == f"{name}_weights.hdf5" else 1
            return (is_local, exact, len(p))

        seen = set()
        out = []
        for p in sorted(set(candidates), key=_rank):
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
        tried = []
        for wp in _candidate_weight_paths(name):
            if os.path.isfile(wp):
                tried.append(wp)
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

expected_ids = pd.read_csv(TEST_CSV)["id_code"].astype(str).values
if not np.array_equal(results["id_code"].values, expected_ids):
    tmp = results.set_index("id_code").reindex(expected_ids).reset_index()
    if tmp["diagnosis"].isna().any():
        raise RuntimeError(
            "Submission alignment failed: missing predictions for some test ids."
        )
    results = tmp

results.to_csv("submission.csv", index=False)

print(results.head(10))
print("Wrote submission.csv with shape:", results.shape)
print("diagnosis value counts:\n", results["diagnosis"].value_counts().sort_index())
