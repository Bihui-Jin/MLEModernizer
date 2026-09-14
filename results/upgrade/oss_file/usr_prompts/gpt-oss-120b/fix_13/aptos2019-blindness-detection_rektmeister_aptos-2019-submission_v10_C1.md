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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'Implemented faster image loading and preprocessing while preserving the original training and inference logic.

Key improvements:
- Optimized `crop_image` to compute a single mask for 3‑channel images, avoiding repeated per‑channel operations.
- Replaced costly `ImageDataGenerator` pipelines with in‑memory NumPy arrays and custom `tf.keras.utils.Sequence` classes that load and preprocess each image only once, then serve batches efficiently.
- Added deterministic shuffling and optional horizontal‑flip augmentation to match the original augmentation behavior.
- Provided required attributes (`n`, `batch_size`, `reset`) so existing training code works unchanged.

These changes significantly reduce I/O and per‑epoch preprocessing, allowing the script to complete within the 600‑second limit without altering model architecture, loss, or evaluation semantics.'
- What this solution (achieved 0.0) has done: 'I wrap the TensorFlow imports in a safe try/except block, remove the unused ImageDataGenerator import that caused an uncaught error, and guard any TensorFlow‑only calls (like tf.random.set_seed) so the script falls back to the RandomForest model when TensorFlow cannot be loaded. This fixes the runtime crash and ensures a valid submission.csv is produced, moving the solution toward the target score.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image  # added Pillow for image handling

TRAINING = True




## === cell 1
train = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
test = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")

train["filename"] = train["id_code"] + ".png"
test["filename"] = test["id_code"] + ".png"

train["diagnosis"] = train["diagnosis"].astype(str)

print("Number of train samples: ", train.shape[0])
print("Number of test samples: ", test.shape[0])




## === cell 2
try:
    import tensorflow as tf

    TF_AVAILABLE = True
except Exception as e:
    print("TensorFlow import failed:", e)
    TF_AVAILABLE = False

try:
    import cv2

    _cv2_available = True
except Exception:
    _cv2_available = False

TF_AVAILABLE = False

IMG_SIZE = 224
NB_CHANNELS = 3
NB_CLASSES = 5  # 0, 1, 2, 3, 4
BATCH_SIZE = 64
TEST_BATCH_SIZE = 1
NUM_WORKERS = min(4, os.cpu_count() or 1)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
def crop_image(img, tol=10):
    if img.ndim == 2:  # grayscale
        mask = img > tol
        if not mask.any():
            return img
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:  # color
        mask = (img > tol).any(axis=2)
        if not mask.any():
            return img
        rows = np.where(mask.any(1))[0]
        cols = np.where(mask.any(0))[0]
        return img[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1, :]
    else:
        return img  # fallback unchanged


def preprocess_image(img):
    img = crop_image(img)
    if _cv2_available:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.addWeighted(
            img, 4, cv2.GaussianBlur(img, (0, 0), IMG_SIZE / 10), -4, 128
        )
    elif TF_AVAILABLE:
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE)).numpy().astype(np.uint8)
    else:
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        img = np.array(pil_img)
    return img




## === cell 4
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
if TF_AVAILABLE:
    tf.random.set_seed(SEED)


def load_and_preprocess_image(path):
    if _cv2_available:
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE, NB_CHANNELS), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        try:
            pil_img = Image.open(path).convert("RGB")
            img = np.array(pil_img)
        except Exception:
            img = np.zeros((IMG_SIZE, IMG_SIZE, NB_CHANNELS), dtype=np.uint8)
    img = preprocess_image(img)
    img = (
        img.astype(np.float32) / 255.0
    )  # same scaling as ImageDataGenerator rescale=1/255
    return img


class ImageSequence(Sequence):
    def __init__(self, images, labels, batch_size, augment=False, shuffle=False):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.augment = augment  # only horizontal flip
        self.shuffle = shuffle
        self.indices = np.arange(len(self.images))
        if self.shuffle:
            np.random.shuffle(self.indices)

        self.n = len(self.images)  # attribute expected by training code

    def __len__(self):
        return int(np.ceil(self.n / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_x = self.images[batch_idx]
        batch_y = self.labels[batch_idx]

        if self.augment:
            flip_mask = np.random.rand(len(batch_x)) < 0.5
            batch_x[flip_mask] = batch_x[flip_mask, :, ::-1, :]
        return batch_x, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def reset(self):
        self.indices = np.arange(len(self.images))


class TestSequence(Sequence):
    def __init__(self, images, batch_size):
        self.images = images
        self.batch_size = batch_size
        self.n = len(self.images)

    def __len__(self):
        return int(np.ceil(self.n / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.images[idx * self.batch_size : (idx + 1) * self.batch_size]
        return batch_x

    def reset(self):
        pass  # nothing needed


if TF_AVAILABLE:
    train_image_paths = [
        os.path.join("../input/aptos2019-blindness-detection/train_images/", fn)
        for fn in train["filename"]
    ]
    all_train_imgs = np.stack([load_and_preprocess_image(p) for p in train_image_paths])
    all_train_labels = pd.get_dummies(train["diagnosis"]).values  # one‑hot

    X_train, X_val, y_train, y_val = train_test_split(
        all_train_imgs,
        all_train_labels,
        test_size=0.2,
        stratify=train["diagnosis"],
        random_state=SEED,
    )

    train_gen = ImageSequence(
        X_train, y_train, batch_size=BATCH_SIZE, augment=True, shuffle=True
    )
    val_gen = ImageSequence(
        X_val, y_val, batch_size=BATCH_SIZE, augment=False, shuffle=False
    )

    test_image_paths = [
        os.path.join("../input/aptos2019-blindness-detection/test_images/", fn)
        for fn in test["filename"]
    ]
    test_imgs = np.stack([load_and_preprocess_image(p) for p in test_image_paths])
    test_gen = TestSequence(test_imgs, batch_size=TEST_BATCH_SIZE)
else:
    train_gen = val_gen = test_gen = None




## === cell 5
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
    from tensorflow.keras.applications import ResNet50

    MODEL_NAME = "resnet50"

    NB_WARMUP_EPOCHS = 2
    NB_EPOCHS = 5
    INITIAL_LR = 1e-3

    weights_path_template = os.path.join("weights", "{}_weights.keras")
    log_path_template = os.path.join("logs", "{}_training_log.csv")
else:
    from sklearn.ensemble import RandomForestClassifier

    MODEL_NAME = "rf_fallback"
    NB_WARMUP_EPOCHS = NB_EPOCHS = 0  # not used
    INITIAL_LR = None

    weights_path_template = None
    log_path_template = None




## === cell 6
if TF_AVAILABLE:
    os.makedirs("weights", exist_ok=True)
    os.makedirs("logs", exist_ok=True)




## === cell 7
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




## === cell 8
if TF_AVAILABLE:

    def get_model(name, input_shape, nb_out):
        models = {"resnet50": get_resnet50, "conv1": get_conv1}
        if name not in models:
            print(f"No model named '{name}'")
            return None
        model = models[name](input_shape, nb_out)
        weights_path = weights_path_template.format(name)
        if os.path.isfile(weights_path):
            model.load_weights(weights_path)
            print(f"loaded model from {weights_path}")
        return model




## === cell 9
if TF_AVAILABLE:

    def train_resnet50(model, train_generator, val_generator, weights_path, log_path):
        for i in range(len(model.layers)):
            model.layers[i].trainable = False
        for i in range(-5, 0):
            model.layers[i].trainable = True

        optimizer = Adam(learning_rate=INITIAL_LR)
        model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
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
            callbacks=[mc, es, cl],
            verbose=1,
        )

        train_generator.reset()
        val_generator.reset()

        for i in range(len(model.layers)):
            model.layers[i].trainable = True

        optimizer = Adam(learning_rate=INITIAL_LR)
        model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
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
            callbacks=[mc, es, cl],
            verbose=1,
        )




## === cell 10
if TF_AVAILABLE:

    def train_conv1(model, train_generator, val_generator, weights_path, log_path):
        optimizer = Adam(learning_rate=INITIAL_LR)
        model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
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
            callbacks=[mc, es, cl],
            verbose=1,
        )




## === cell 11
def train_model(name, input_shape, nb_out, train_generator, val_generator):
    if not TF_AVAILABLE:
        print("TensorFlow not available – using RandomForest fallback.")

        def load_images(df, folder):
            imgs = []
            for fn in df["filename"]:
                path = os.path.join(folder, fn)
                if _cv2_available:
                    img = cv2.imread(path)
                    if img is None:
                        img = np.zeros(
                            (IMG_SIZE, IMG_SIZE, NB_CHANNELS), dtype=np.uint8
                        )
                    else:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    try:
                        pil_img = Image.open(path).convert("RGB")
                        img = np.array(pil_img)
                    except Exception:
                        img = np.zeros(
                            (IMG_SIZE, IMG_SIZE, NB_CHANNELS), dtype=np.uint8
                        )
                img = (
                    cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    if _cv2_available
                    else Image.fromarray(img).resize(
                        (IMG_SIZE, IMG_SIZE), Image.BILINEAR
                    )
                )
                imgs.append(img.flatten())
            return np.array(imgs)

        X = load_images(train, "../input/aptos2019-blindness-detection/train_images/")
        y = train["diagnosis"].astype(int).values
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        rf = RandomForestClassifier(
            n_estimators=400, max_depth=None, n_jobs=NUM_WORKERS, random_state=42
        )
        rf.fit(X_train, y_train)
        global fallback_model
        fallback_model = rf
        return

    model = get_model(name, input_shape, nb_out)
    trainers = {"resnet50": train_resnet50, "conv1": train_conv1}
    if name not in trainers:
        print(f"No trainer for model '{name}'")
        return
    trainers[name](
        model,
        train_generator,
        val_generator,
        weights_path_template.format(name),
        log_path_template.format(name),
    )




## === cell 12
if TRAINING:
    train_model(
        MODEL_NAME,
        (IMG_SIZE, IMG_SIZE, NB_CHANNELS),
        NB_CLASSES,
        train_gen,
        val_gen,
    )




## === cell 13
if TF_AVAILABLE:
    model = get_model(MODEL_NAME, (IMG_SIZE, IMG_SIZE, NB_CHANNELS), NB_CLASSES)
    test_gen.reset()
    preds = model.predict(test_gen, verbose=1)
    predictions = np.argmax(preds, axis=1)
else:
    predictions = fallback_model.predict(
        test_imgs.reshape(len(test_imgs), -1)  # flatten for RF
    )

filenames = test["id_code"].values
results = pd.DataFrame({"id_code": filenames, "diagnosis": predictions})
results.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2937852206.py in <cell line: 0>()
      6 else:
      7     predictions = fallback_model.predict(
----> 8         test_imgs.reshape(len(test_imgs), -1)  # flatten for RF
      9     )
     10 

NameError: name 'test_imgs' is not defined
