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

3.9

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

0.5341472935256788

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'We fix the environment-breaking import error by avoiding the standalone `keras` backend (which triggers the `MessageFactory` protobuf issue) and using `tf.keras.backend` consistently. Since the referenced pre-trained model file doesn’t exist in your input folders, we replace the missing `load_model()` step with a minimal training/inference pipeline that uses the same image preprocessing and produces valid 5-class predictions for the required submission format. We also fix deprecated/removed APIs (`predict_generator`) and ensure paths point to the provided dataset directory so the generator can find images. Finally, we always write `submission.csv` with columns `id_code,diagnosis` and the correct row order.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

import cv2
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models

from tensorflow.keras import backend as K

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_PATH = "/kaggle/input/aptos2019-blindness-detection/"

DIM_X = 256
DIM_Y = 256
BATCH_SIZE = 32

assert os.path.exists(
    os.path.join(DATA_PATH, "train.csv")
), "train.csv not found at DATA_PATH"
assert os.path.exists(
    os.path.join(DATA_PATH, "test.csv")
), "test.csv not found at DATA_PATH"
assert os.path.isdir(
    os.path.join(DATA_PATH, "train_images")
), "train_images dir not found"
assert os.path.isdir(
    os.path.join(DATA_PATH, "test_images")
), "test_images dir not found"


def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:  # image too dark; return original
            return img
        img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
        img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
        img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
        return np.stack([img1, img2, img3], axis=-1)


def circle_crop_v2(img):
    height, width, depth = img.shape
    largest_side = np.max((height, width))
    img = cv2.resize(img, (largest_side, largest_side))
    height, width, depth = img.shape
    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x, y))
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    return img


def preprocess_image(image, sigmaX=25, DIM_X=256, DIM_Y=256):
    if image is None:
        return np.zeros((DIM_Y, DIM_X, 3), dtype=np.uint8)
    if image.shape[-1] == 3:
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception:
            pass
    image = crop_image_from_gray(image)
    image = circle_crop_v2(image)
    image = cv2.resize(image, (DIM_X, DIM_Y))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image


def kappa_keras(y_true, y_pred):
    y_true = K.cast(K.argmax(y_true, axis=-1), dtype="int32")
    y_pred = K.cast(K.argmax(y_pred, axis=-1), dtype="int32")

    min_rating = K.minimum(K.min(y_true), K.min(y_pred))
    max_rating = K.maximum(K.max(y_true), K.max(y_pred))

    y_true = tf.map_fn(lambda y: y - min_rating, y_true, fn_output_signature=tf.int32)
    y_pred = tf.map_fn(lambda y: y - min_rating, y_pred, fn_output_signature=tf.int32)

    num_ratings = max_rating - min_rating + 1
    num_ratings = K.cast(num_ratings, "int32")

    observed = tf.math.confusion_matrix(y_true, y_pred, num_classes=num_ratings)
    num_scored_items = K.shape(y_true)[0]

    r = K.cast(K.arange(num_ratings), dtype="float64")
    weights = K.expand_dims(r, axis=-1) - K.expand_dims(r, axis=0)
    weights = K.cast(K.pow(weights, 2), dtype="float64")

    hist_true = tf.math.bincount(y_true, minlength=num_ratings, maxlength=num_ratings)
    hist_pred = tf.math.bincount(y_pred, minlength=num_ratings, maxlength=num_ratings)
    hist_true = K.cast(hist_true, "float64") / K.cast(num_scored_items, "float64")
    hist_pred = K.cast(hist_pred, "float64") / K.cast(num_scored_items, "float64")

    expected = K.dot(
        K.expand_dims(hist_true, axis=-1), K.expand_dims(hist_pred, axis=0)
    )
    observed = K.cast(observed, "float64") / K.cast(num_scored_items, "float64")

    score = K.sum(weights * observed) / (K.sum(weights * expected) + K.epsilon())
    return 1.0 - score


def create_kappa_loss(bsize, eps=1e-10, N=5):
    def kappa_loss(y_true, y_pred):
        y_true = tf.cast(y_true, dtype="float32")
        y_pred = tf.cast(y_pred, dtype="float32")

        repeat_op = tf.cast(
            tf.tile(tf.reshape(tf.range(0, N), [N, 1]), [1, N]), dtype="float32"
        )
        weights = tf.square(repeat_op - tf.transpose(repeat_op)) / tf.cast(
            (N - 1) ** 2, dtype="float32"
        )

        pred_ = tf.square(y_pred)
        pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, axis=1), [-1, 1]))

        hist_rater_a = tf.reduce_sum(pred_norm, axis=0)
        hist_rater_b = tf.reduce_sum(y_true, axis=0)

        conf_mat = tf.matmul(tf.transpose(pred_norm), y_true)

        nom = tf.reduce_sum(weights * conf_mat)

        b = tf.cast(tf.shape(y_true)[0], dtype="float32")
        denom = tf.reduce_sum(
            weights
            * tf.matmul(
                tf.reshape(hist_rater_a, [N, 1]), tf.reshape(hist_rater_b, [1, N])
            )
            / (b + eps)
        )
        return nom / (denom + eps)

    return kappa_loss


KAPPA_LOSS = create_kappa_loss(BATCH_SIZE)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def build_model(input_shape=(DIM_Y, DIM_X, 3), num_classes=5):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPool2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    return model


model = build_model()
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy", kappa_keras],
)
model.summary()



## === cell 2
train_df = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))

train_df["filename"] = train_df["id_code"].astype(str) + ".png"
test_df["filename"] = test_df["id_code"].astype(str) + ".png"

train_df["diagnosis"] = train_df["diagnosis"].astype(str)

idx = np.arange(len(train_df))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
split = int(0.9 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]

tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
va_df = train_df.iloc[va_idx].reset_index(drop=True)

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_image,
    rescale=1.0 / 255.0,
    rotation_range=10,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.05,
)

valid_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_image,
    rescale=1.0 / 255.0,
)

train_gen = train_datagen.flow_from_dataframe(
    dataframe=tr_df,
    directory=os.path.join(DATA_PATH, "train_images"),
    x_col="filename",
    y_col="diagnosis",
    target_size=(DIM_Y, DIM_X),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=SEED,
    validate_filenames=False,
)

valid_gen = valid_datagen.flow_from_dataframe(
    dataframe=va_df,
    directory=os.path.join(DATA_PATH, "train_images"),
    x_col="filename",
    y_col="diagnosis",
    target_size=(DIM_Y, DIM_X),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
    validate_filenames=False,
)

assert set(train_gen.class_indices.keys()) == set(
    [str(i) for i in range(5)]
), train_gen.class_indices



## === cell 3
EPOCHS = 3
history = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=EPOCHS,
    verbose=1,
)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
OperatorNotAllowedInGraphError            Traceback (most recent call last)
/tmp/ipykernel_11/2898332636.py in <cell line: 0>()
      2 # No early stopping or sampling; just a small fixed number of epochs.
      3 EPOCHS = 3
----> 4 history = model.fit(
      5     train_gen,
      6     validation_data=valid_gen,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_11/1048150816.py in kappa_keras(y_true, y_pred)
    108     num_scored_items = K.shape(y_true)[0]
    109 
--> 110     r = K.cast(K.arange(num_ratings), dtype="float64")
    111     weights = K.expand_dims(r, axis=-1) - K.expand_dims(r, axis=0)
    112     weights = K.cast(K.pow(weights, 2), dtype="float64")

OperatorNotAllowedInGraphError: Using a symbolic `tf.Tensor` as a Python `bool` is not allowed. You can attempt the following resolutions to the problem: If you are running in Graph mode, use Eager execution mode or decorate this function with @tf.function. If you are using AutoGraph, you can try decorating this function with @tf.function. If that does not work, then you may be using an unsupported feature or your source code may not be visible to AutoGraph. See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/limitations.md#access-to-source-code for more information.

## === cell 4
submission_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_image,
    rescale=1.0 / 255.0,
)

test_gen = submission_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=os.path.join(DATA_PATH, "test_images"),
    x_col="filename",
    y_col=None,
    target_size=(DIM_Y, DIM_X),
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_mode=None,
    validate_filenames=False,
)

pred_proba = model.predict(test_gen, verbose=1)
pred = np.argmax(pred_proba, axis=1).astype(int)



## === cell 5
submission = pd.read_csv(os.path.join(DATA_PATH, "sample_submission.csv"))
pred_map = dict(zip(test_df["id_code"].values, pred))
submission["diagnosis"] = submission["id_code"].map(pred_map).astype(int)
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
