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

0.06709

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the TensorFlow/Keras import errors by switching from `tensorflow.python.keras` to the supported `tensorflow.keras` API, which also resolves the `BatchNormalization` and `ImageDataGenerator` failures. I also remove notebook-only shell commands (`!mkdir`) and replace them with `os.makedirs` so the script runs as a plain Python file in Kaggle. To ensure inference works even when no external weight file is available, I make weight loading optional and fall back to a freshly initialized model (still producing a valid `submission.csv`). Finally, I fix a few execution-order/name issues (like `MODEL_NAME` not being defined due to earlier import failures) and update deprecated `.fit_generator/.predict_generator` to `.fit/.predict` for compatibility.'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow import crash (`MessageFactory GetPrototype`) by forcing the pure-Python protobuf implementation before importing TensorFlow, which is a common Kaggle runtime incompatibility. Then I fix the score=0.0 issue by ensuring inference uses a real trained model: automatically locate and load the provided conv1 weight file from any attached input dataset (instead of only one hardcoded directory), while keeping the architecture/training logic unchanged. I also make the submission alignment robust by using `test["id_code"]` order (not generator filenames) and writing integer diagnoses 0–4 with the exact required columns. These changes are minimal, execution-blocking/score-critical, and keep the rest of the pipeline intact.'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow/protobuf crash by setting the protobuf environment variables before any TensorFlow import (the current placement is too late), which unblocks the whole pipeline. Then I keep the same model/inference logic but make prediction robust by explicitly limiting prediction steps to the generator length so it can’t hang or mismatch. Finally, I keep the submission formatting/alignment identical to the sample submission and ensure the CSV is always written with the required columns and correct row count, which should move the score up from 0.0 to a meaningful value (assuming weights are found/loaded).'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow/protobuf crash by moving the protobuf environment-variable settings to the very top of the script (before any TensorFlow-related import), which is the root cause of the `MessageFactory.GetPrototype` error. I also make the image directory resolution robust (some Kaggle setups have the files under `/kaggle/input/...` instead of `../input/...`) without changing any modeling/training logic. Finally, I keep the same weight-loading logic but ensure the code always finds the correct dataset root and writes a correctly formatted `submission.csv` aligned to `test.csv`, which should move the score up from 0.0 (random/unrun) toward the target if pretrained weights are present.'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow/protobuf crash that stops execution by setting the protobuf env vars before any TensorFlow-related import and by forcing a compatible import order. Then I keep the exact same data pipeline/model code, but add a safe fallback to load a matching local weights file if the hardcoded dataset isn’t attached (otherwise you keep getting effectively-random predictions and ~0.0 kappa). Finally, I ensure prediction length and submission alignment always match `test.csv`/`sample_submission.csv`, and that `submission.csv` is written with the required columns and row count.'
- What this solution (achieved 0.0) has done: 'I fix the execution-blocking TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation *before* any TensorFlow import (the current `cpp` setting triggers the `_message` ImportError in this environment). Then I ensure TensorFlow/Keras-dependent cells run by moving the TF import and Keras imports into a single place after the env vars, which also resolves the downstream `ImageDataGenerator`/`MODEL_NAME` NameErrors caused by earlier crashes. Finally, I keep the same model/data pipeline and prediction logic, but make submission writing unconditional and aligned to `test.csv` so a valid `submission.csv` is always produced.'
- What this solution (achieved 0.0) has done: 'We fix the execution-blocking TensorFlow/protobuf incompatibility that triggers `MessageFactory.GetPrototype` by ensuring the protobuf implementation environment variables are set before any TensorFlow/protobuf import and by importing `google.protobuf` before `tensorflow` to lock in the pure-Python backend. Then we keep your model/data pipeline unchanged but make weight discovery robust to multiple common filename variants/locations so it actually loads pretrained conv1 weights when present (avoiding near-random predictions and ~0.0 kappa). Finally, we keep the same prediction and submission logic while ensuring the submission is always written as `submission.csv` with the correct columns and row count aligned to `test.csv`.'
- What this solution (achieved 0.0) has done: 'We fix the TensorFlow/protobuf crash by setting the required environment variables *before any protobuf/TensorFlow import* and by explicitly importing `google.protobuf` first, which prevents the `MessageFactory.GetPrototype` AttributeError. We keep the model, generators, and training/inference logic unchanged, but make the import order safe and add a defensive fallback to alternate TF keras preprocessing import in case the first path fails. Finally, we ensure the script always reaches submission writing and produces a valid `submission.csv` aligned to `test.csv`/`sample_submission.csv`, which should move the score up from 0.0 once weights load successfully.'
- What this solution (achieved -0.00591) has done: 'I fix the TensorFlow/protobuf import crash by moving and tightening the protobuf environment configuration to occur before any `google.protobuf` or `tensorflow` import, and by avoiding importing `google.protobuf` prior to TensorFlow (which can lock in an incompatible implementation). This unblocks the pipeline so the model can be built and predictions can be generated. I also add a safe fallback so if TensorFlow still cannot import in this environment, the script still produce a valid `submission.csv` (score-neutral fallback, but ensures end-to-end execution). Finally, I keep the model/generator/prediction logic unchanged and preserve the existing robust weight-file discovery so the score can move up from 0.0 when pretrained weights are available.'
- What this solution (achieved 0.06709) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by configuring protobuf-related environment variables before any TensorFlow/protobuf import and by forcing TensorFlow to use the pure-Python protobuf implementation reliably in this Kaggle Python 3.7 environment. Then I keep your model, generators, and weight-loading logic unchanged, but make the TensorFlow availability check robust so the pipeline actually runs with TF instead of falling back to all-zero predictions (which is causing the very low kappa). Finally, I keep the submission formatting identical while ensuring prediction length matches `test.csv` exactly and that `submission.csv` is always written correctly.'

# 9. Code solution

## === cell 0
import os
import math
import warnings

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.random.seed(42)

TRAINING = False




## === cell 1
def _resolve_input_dir(preferred_rel_path="../input/aptos2019-blindness-detection"):
    candidates = [
        preferred_rel_path,
        "/kaggle/input/aptos2019-blindness-detection",
        "/kaggle/data/aptos2019-blindness-detection",
        "../input",
        "/kaggle/input",
        "/kaggle/data",
    ]
    for p in candidates:
        p = os.path.abspath(p)
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, "train.csv")):
            return p

    for base in ["/kaggle/input", "/kaggle/data", os.path.abspath("../input")]:
        if os.path.isdir(base):
            for root, _, files in os.walk(base):
                if "train.csv" in files and "test.csv" in files:
                    return root
    return os.path.abspath(preferred_rel_path)


INPUT_DIR = _resolve_input_dir("../input/aptos2019-blindness-detection")
TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train_images")
TEST_IMG_DIR = os.path.join(INPUT_DIR, "test_images")

print("Using INPUT_DIR:", INPUT_DIR)
print("TRAIN_CSV exists:", os.path.isfile(TRAIN_CSV))
print("TEST_CSV exists:", os.path.isfile(TEST_CSV))
print("TRAIN_IMG_DIR exists:", os.path.isdir(TRAIN_IMG_DIR))
print("TEST_IMG_DIR exists:", os.path.isdir(TEST_IMG_DIR))



## === cell 2
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)

print("Number of train samples: ", train.shape[0])
print("Number of test samples: ", test.shape[0])



## === cell 3
train["id_code"] = train["id_code"].apply(lambda x: str(x) + ".png")
test["id_code"] = test["id_code"].apply(lambda x: str(x) + ".png")
train["diagnosis"] = train["diagnosis"].astype(str)



## === cell 4
import cv2

TF_AVAILABLE = True
try:
    import tensorflow as tf

    try:
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
    except Exception:
        from keras.preprocessing.image import ImageDataGenerator  # type: ignore

    print("TensorFlow version:", tf.__version__)
except Exception as e:
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = repr(e)
    print("WARNING: TensorFlow failed to import; will write a fallback submission.")
    print("TF import error:", TF_IMPORT_ERROR)

IMG_SIZE = 224
NB_CHANNELS = 3
NB_CLASSES = 5  # 0, 1, 2, 3, 4
BATCH_SIZE = 32
TEST_BATCH_SIZE = 1



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
        img_cpy = img.copy()
        try:
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
if TF_AVAILABLE:
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
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
        rescale=1.0 / 255.0, preprocessing_function=preprocess_image
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
"""
simple CNN
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




## === cell 11
"""
returns model

Score fix: ensure we actually load the pre-trained weights if they exist anywhere under ../input or /kaggle/input.
Core model logic is unchanged; only weight path discovery is made robust.
"""


def _find_weight_file_any(possible_filenames, search_roots):
    for base in search_roots:
        for fn in possible_filenames:
            cand = os.path.join(base, fn)
            if os.path.isfile(cand):
                return cand

    for base_dir in search_roots:
        if not os.path.isdir(base_dir):
            continue
        for root, _, files in os.walk(base_dir):
            file_set = set(files)
            for fn in possible_filenames:
                if fn in file_set:
                    return os.path.join(root, fn)
    return None


if TF_AVAILABLE:

    def get_model(name, input_shape, nb_out):
        models = {
            "resnet50": get_resnet50,
            "conv1": get_conv1,
        }

        if name not in models:
            raise ValueError(f"No model named '{name}'")

        model = models[name](input_shape, nb_out)

        expected_weights_path = weights_path_template.format(name)
        expected_fn = os.path.basename(expected_weights_path)

        possible_fns = [
            expected_fn,  # conv1_weights.hdf5
            f"{name}_weights.h5",  # conv1_weights.h5
            f"{name}.h5",  # conv1.h5
            f"{name}.hdf5",  # conv1.hdf5
            "conv1_weights.hdf5",
            "conv1_weights.h5",
        ]

        if os.path.isfile(expected_weights_path):
            weights_path = expected_weights_path
        else:
            weights_path = _find_weight_file_any(
                possible_filenames=possible_fns,
                search_roots=[
                    "../input/aptos-2019-conv1-weights",
                    "/kaggle/input/aptos-2019-conv1-weights",
                    "../input",
                    "/kaggle/input",
                    "/kaggle/data",
                ],
            )

        if weights_path and os.path.isfile(weights_path):
            model.load_weights(weights_path)
            print(f"loaded model from {weights_path}")
        else:
            print("weights not found; using randomly initialized weights")

        return model




## === cell 12
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
        optimizer = Adam(learning_rate=INITIAL_LR)

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
        es = EarlyStopping(
            monitor="val_loss", mode="min", restore_best_weights=True, verbose=1
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




## === cell 13
"""
trains the simple CNN
"""
if TF_AVAILABLE:

    def train_conv1(model, train_generator, val_generator, weights_path, log_path):
        metrics_list = ["accuracy"]
        optimizer = Adam(learning_rate=INITIAL_LR)

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
if TF_AVAILABLE:

    def train_model(name, input_shape, nb_out, train_generator, val_generator):
        model = get_model(name, input_shape, nb_out)

        trainers = {"resnet50": train_resnet50, "conv1": train_conv1}

        if name not in trainers:
            raise ValueError(f"No model named '{name}'")

        trainers[name](
            model,
            train_generator,
            val_generator,
            weights_path_template.format(name),
            log_path_template.format(name),
        )




## === cell 15
if TF_AVAILABLE and TRAINING:
    train_model(
        MODEL_NAME, (IMG_SIZE, IMG_SIZE, NB_CHANNELS), NB_CLASSES, train_gen, val_gen
    )



## === cell 16
if TF_AVAILABLE:
    model = get_model(MODEL_NAME, (IMG_SIZE, IMG_SIZE, NB_CHANNELS), NB_CLASSES)

    test_gen.reset()
    pred_steps = len(test_gen)

    preds = model.predict(test_gen, steps=pred_steps, verbose=1)
    predictions = np.argmax(preds, axis=1).astype(int)

    predictions = predictions[: len(test)]

    results = pd.DataFrame(
        {
            "id_code": test["id_code"].str.replace(".png", "", regex=False),
            "diagnosis": predictions,
        }
    )
else:
    results = pd.DataFrame(
        {
            "id_code": test["id_code"].str.replace(".png", "", regex=False),
            "diagnosis": 0,
        }
    )

sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")
if os.path.isfile(sample_sub_path):
    sample_sub = pd.read_csv(sample_sub_path)
    if "id_code" in sample_sub.columns and len(sample_sub) == len(results):
        results = sample_sub[["id_code"]].merge(results, on="id_code", how="left")
        results["diagnosis"] = results["diagnosis"].fillna(0).astype(int)

results["diagnosis"] = results["diagnosis"].clip(0, 4).astype(int)
results.to_csv("submission.csv", index=False)
print(results.head(10))
print("Wrote submission.csv with shape:", results.shape)
