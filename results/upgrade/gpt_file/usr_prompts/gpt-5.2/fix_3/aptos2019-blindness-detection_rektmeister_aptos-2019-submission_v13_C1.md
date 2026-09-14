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

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the TensorFlow/Keras import crash by avoiding the problematic `tensorflow.keras.preprocessing.image.ImageDataGenerator` in this environment and replacing it with a small, equivalent `tf.keras.utils.Sequence` generator that preserves the same preprocessing/augmentation logic. I also fix inference by replacing the deprecated `predict_generator` call with `model.predict`, and make the code robust to the missing `../input/aptos-2019-densenet121-weights` directory by falling back to random weights (still producing a valid submission). Finally, I ensure the submission has exactly the required columns (`id_code,diagnosis`) and that `id_code` matches `test.csv` ordering (no directory prefixes).'
- What this solution (achieved 0.0) has done: 'The crash happens at `import tensorflow as tf` due to an incompatibility between TensorFlow and the `protobuf` version in this environment (the `MessageFactory.GetPrototype` error). The minimal, robust fix is to force protobuf to use the pure-Python implementation *before* importing TensorFlow, which avoids that failing C++ path. Once TensorFlow imports, the rest of your pipeline can run unchanged and produce a valid `submission.csv`. Since your current score is 0.0 due to the runtime crash (no valid model inference), this fix move the score upward toward the target by enabling end-to-end execution.'

# 9. Code solution

## === cell 0
import os
import math
import random

import numpy as np
import pandas as pd

TRAINING = False

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")



## === cell 1
BASE_CANDIDATES = [
    "../input/aptos2019-blindness-detection",
    "../input/aptos2019-blindness-detection/aptos2019-blindness-detection",
    "/kaggle/input/aptos2019-blindness-detection",
    "/kaggle/input/aptos2019-blindness-detection/aptos2019-blindness-detection",
]
DATA_DIR = None
for p in BASE_CANDIDATES:
    if os.path.exists(os.path.join(p, "train.csv")):
        DATA_DIR = p
        break

if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not locate aptos2019-blindness-detection dataset directory."
    )

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

print("Using DATA_DIR:", DATA_DIR)
print("Train images dir exists:", os.path.isdir(TRAIN_IMG_DIR))
print("Test images dir exists:", os.path.isdir(TEST_IMG_DIR))



## === cell 2
WEIGHTS_DIR = "../input/aptos-2019-densenet121-weights"
if os.path.isdir(WEIGHTS_DIR):
    print("Found weights dir:", WEIGHTS_DIR)
    try:
        print("Files:", os.listdir(WEIGHTS_DIR)[:10])
    except Exception as e:
        print("Could not list weights dir:", repr(e))
else:
    print(
        "Weights dir not found (will run with randomly initialized weights):",
        WEIGHTS_DIR,
    )



## === cell 3
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)

print("Number of train samples: ", train.shape[0])
print("Number of test samples: ", test.shape[0])
print(train.head())
print(test.head())



## === cell 4
train = train.copy()
test = test.copy()

train["id_code"] = train["id_code"].apply(lambda x: str(x) + ".png")
test["id_code"] = test["id_code"].apply(lambda x: str(x) + ".png")
train["diagnosis"] = train["diagnosis"].astype(str)

label_cols = ["lbl_0", "lbl_1", "lbl_2", "lbl_3", "lbl_4"]
label_mat = np.zeros((train.shape[0], len(label_cols)), dtype=np.int32)

for i in range(train.shape[0]):
    d = int(train["diagnosis"].iloc[i])
    for j in range(d + 1):
        label_mat[i, j] = 1

train = pd.concat([train, pd.DataFrame(label_mat, columns=label_cols)], axis=1)
print(train.head(3))



## === cell 5
import cv2
import tensorflow as tf

IMG_SIZE = 224
NB_CHANNELS = 3
NB_CLASSES = 5  # 0, 1, 2, 3, 4
BATCH_SIZE = 32
TEST_BATCH_SIZE = 1

try:
    tf.random.set_seed(SEED)
except Exception:
    pass



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
    else:
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




## === cell 7
class DataFrameSequence(tf.keras.utils.Sequence):
    def __init__(
        self,
        df,
        directory,
        x_col,
        y_cols=None,
        batch_size=32,
        shuffle=True,
        rescale=1.0 / 255.0,
        preprocess_fn=None,
        horizontal_flip=False,
    ):
        self.df = df.reset_index(drop=True)
        self.directory = directory
        self.x_col = x_col
        self.y_cols = y_cols
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rescale = rescale
        self.preprocess_fn = preprocess_fn
        self.horizontal_flip = horizontal_flip

        self.indices = np.arange(len(self.df))
        self.on_epoch_end()

    @property
    def n(self):
        return len(self.df)

    @property
    def filenames(self):
        return self.df[self.x_col].tolist()

    def __len__(self):
        return int(math.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def reset(self):
        self.on_epoch_end()

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_idx]

        X = np.zeros((len(batch_df), IMG_SIZE, IMG_SIZE, NB_CHANNELS), dtype=np.float32)

        for i, fname in enumerate(batch_df[self.x_col].tolist()):
            fpath = os.path.join(self.directory, fname)
            img = cv2.imread(fpath)
            if img is None:
                img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            if self.preprocess_fn is not None:
                img = self.preprocess_fn(img)
            else:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if self.horizontal_flip and (np.random.rand() < 0.5):
                img = np.ascontiguousarray(img[:, ::-1, :])

            img = img.astype(np.float32) * self.rescale
            X[i] = img

        if self.y_cols is None:
            return X
        y = batch_df[self.y_cols].values.astype(np.float32)
        return X, y




## === cell 8
val_fraction = 0.2
n_total = len(train)
n_val = int(round(n_total * val_fraction))

perm = np.random.RandomState(SEED).permutation(n_total)
val_idx = perm[:n_val]
train_idx = perm[n_val:]

train_df = train.iloc[train_idx].reset_index(drop=True)
val_df = train.iloc[val_idx].reset_index(drop=True)

train_gen = DataFrameSequence(
    df=train_df,
    directory=TRAIN_IMG_DIR,
    x_col="id_code",
    y_cols=label_cols,
    batch_size=BATCH_SIZE,
    shuffle=True,
    rescale=1.0 / 255.0,
    preprocess_fn=preprocess_image,
    horizontal_flip=True,
)

val_gen = DataFrameSequence(
    df=val_df,
    directory=TRAIN_IMG_DIR,
    x_col="id_code",
    y_cols=label_cols,
    batch_size=BATCH_SIZE,
    shuffle=False,
    rescale=1.0 / 255.0,
    preprocess_fn=preprocess_image,
    horizontal_flip=False,
)

test_gen = DataFrameSequence(
    df=test,
    directory=TEST_IMG_DIR,
    x_col="id_code",
    y_cols=None,
    batch_size=TEST_BATCH_SIZE,
    shuffle=False,
    rescale=1.0 / 255.0,
    preprocess_fn=preprocess_image,
    horizontal_flip=False,
)

print(
    "train batches:",
    len(train_gen),
    "val batches:",
    len(val_gen),
    "test batches:",
    len(test_gen),
)



## === cell 9
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

from sklearn.metrics import cohen_kappa_score

MODEL_NAME = "densenet121"

NB_WARMUP_EPOCHS = 2
NB_EPOCHS = 30
INITIAL_LR = 1e-3

weights_path_template = os.path.join(WEIGHTS_DIR, "{}_weights.hdf5")
log_path_template = os.path.join("logs/", "{}_training_log.csv")



## === cell 10
os.makedirs("weights", exist_ok=True)
os.makedirs("logs", exist_ok=True)



## === cell 11
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




## === cell 12
def get_densenet121(input_shape, nb_out):
    inputs = Input(shape=input_shape)
    base_model = DenseNet121(weights=None, include_top=False, input_tensor=inputs)

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)

    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)

    output = Dense(nb_out, activation="sigmoid")(x)
    model = Model(inputs, output)
    return model




## === cell 13
"""
simple CNN (conv1)
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




## === cell 14
"""
simple CNN v2 (conv2)
"""


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




## === cell 15
"""
returns model
"""


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
        print(
            f"weights not found at {weights_path} (continuing without pretrained weights)"
        )

    return model




## === cell 16
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

    mc = ModelCheckpoint(
        weights_path, monitor="val_loss", save_best_only=True, verbose=1
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

    optimizer = Adam(learning_rate=INITIAL_LR)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=metrics_list
    )

    mc = ModelCheckpoint(
        weights_path, monitor="val_loss", save_best_only=True, verbose=1
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




## === cell 17
def train_densenet121(model, train_generator, val_generator, weights_path, log_path):
    metrics_list = ["accuracy"]
    optimizer = Adam(learning_rate=INITIAL_LR)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=metrics_list)

    mc = ModelCheckpoint(
        weights_path, monitor="val_loss", save_best_only=True, verbose=1
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
"""
trains the simple CNN
"""


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




## === cell 19
def train_conv2(model, train_generator, val_generator, weights_path, log_path):
    metrics_list = ["accuracy"]
    optimizer = Adam(learning_rate=INITIAL_LR)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=metrics_list
    )

    mc = ModelCheckpoint(
        weights_path, monitor="val_loss", save_best_only=True, verbose=1
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




## === cell 20
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

    weights_path = weights_path_template.format(name)
    if not os.path.isdir(os.path.dirname(weights_path)):
        weights_path = os.path.join("weights", f"{name}_weights.hdf5")

    trainers[name](
        model,
        train_generator,
        val_generator,
        weights_path,
        log_path_template.format(name),
    )




## === cell 21
if TRAINING:
    train_model(
        MODEL_NAME, (IMG_SIZE, IMG_SIZE, NB_CHANNELS), NB_CLASSES, train_gen, val_gen
    )



## === cell 22
model = get_model(MODEL_NAME, (IMG_SIZE, IMG_SIZE, NB_CHANNELS), NB_CLASSES)
test_gen.reset()

preds = model.predict(test_gen, verbose=1)

preds_bin = preds > 0.5
predictions = preds_bin.astype(int).sum(axis=1) - 1
predictions = np.clip(predictions, 0, 4).astype(int)

id_codes = pd.read_csv(TEST_CSV)["id_code"].astype(str).values
submission = pd.DataFrame({"id_code": id_codes, "diagnosis": predictions})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("submission.csv exists:", os.path.isfile("submission.csv"))
