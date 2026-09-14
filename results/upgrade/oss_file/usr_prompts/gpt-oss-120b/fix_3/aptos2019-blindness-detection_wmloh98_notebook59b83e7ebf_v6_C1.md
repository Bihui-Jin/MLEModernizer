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

0.8016449638822514

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I replace the failed model‑loading step with a small EfficientNet‑B0 model built directly in the script, fix the TensorFlow/Keras imports, and use the modern `model.predict` API. The script now creates a training generator, trains the model for a few epochs, predicts on the test set, converts predictions to class labels, and writes a proper `submission.csv`. All paths are corrected, the custom dropout and kappa loss are retained, and the code runs end‑to‑end producing a valid submission file.'
- What this solution (achieved 0.0) has done: 'The fix adds a protobuf‑compatible environment setting before importing TensorFlow, converts the `diagnosis` labels to strings so Keras can build a categorical generator, and ensures the generators and model are defined after these corrections. This resolves the import error and the `class_mode="categorical"` type error, allowing the script to run end‑to‑end and produce a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam

BASE_PATH = "/kaggle/input/aptos2019-blindness-detection/"
DIM_X, DIM_Y = 256, 256
BATCH_SIZE = 32
NUM_CLASSES = 5




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


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


def preprocess_image(image, sigmaX=25, dim_x=DIM_X, dim_y=DIM_Y):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = circle_crop_v2(image)
    image = cv2.resize(image, (dim_x, dim_y))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image


class FixedDropout(Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        symbolic_shape = K.shape(inputs)
        return tuple(
            symbolic_shape[axis] if shape is None else shape
            for axis, shape in enumerate(self.noise_shape)
        )




## === cell 2
def kappa_keras(y_true, y_pred):
    y_true = K.cast(K.argmax(y_true, axis=-1), dtype="int32")
    y_pred = K.cast(K.argmax(y_pred, axis=-1), dtype="int32")
    min_rating = K.minimum(K.min(y_true), K.min(y_pred))
    max_rating = K.maximum(K.max(y_true), K.max(y_pred))
    y_true = K.map_fn(lambda y: y - min_rating, y_true, dtype="int32")
    y_pred = K.map_fn(lambda y: y - min_rating, y_pred, dtype="int32")
    num_ratings = max_rating - min_rating + 1
    observed = tf.math.confusion_matrix(y_true, y_pred, num_classes=num_ratings)
    num_scored_items = K.shape(y_true)[0]
    weights = K.expand_dims(K.arange(num_ratings), axis=-1) - K.expand_dims(
        K.arange(num_ratings), axis=0
    )
    weights = K.cast(K.pow(weights, 2), dtype="float64")
    hist_true = tf.math.bincount(y_true, minlength=num_ratings)[:num_ratings] / tf.cast(
        num_scored_items, tf.float32
    )
    hist_pred = tf.math.bincount(y_pred, minlength=num_ratings)[:num_ratings] / tf.cast(
        num_scored_items, tf.float32
    )
    expected = K.dot(
        K.expand_dims(hist_true, axis=-1), K.expand_dims(hist_pred, axis=0)
    )
    observed = observed / tf.cast(num_scored_items, tf.float32)
    score = tf.where(
        K.any(K.not_equal(weights, 0)),
        K.sum(weights * observed) / K.sum(weights * expected),
        0,
    )
    return 1.0 - score


def create_kappa_loss(bsize, eps=1e-10, N=NUM_CLASSES):
    def kappa_loss(y_true, y_pred):
        y_true = tf.cast(y_true, dtype="float")
        repeat_op = tf.cast(
            tf.tile(tf.reshape(tf.range(0, N), [N, 1]), [1, N]), dtype="float"
        )
        repeat_op_sq = tf.square((repeat_op - tf.transpose(repeat_op)))
        weights = repeat_op_sq / tf.cast((N - 1) ** 2, dtype="float")
        pred_ = y_pred**2
        pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [-1, 1]))
        hist_rater_a = tf.reduce_sum(pred_norm, 0)
        hist_rater_b = tf.reduce_sum(y_true, 0)
        conf_mat = tf.matmul(tf.transpose(pred_norm), y_true)
        nom = tf.reduce_sum(weights * conf_mat)
        denom = tf.reduce_sum(
            weights
            * tf.matmul(
                tf.reshape(hist_rater_a, [N, 1]), tf.reshape(hist_rater_b, [1, N])
            )
            / tf.cast(bsize, dtype="float")
        )
        return nom / (denom + eps)

    return kappa_loss


KAPPA_LOSS = create_kappa_loss(BATCH_SIZE)



## === cell 3
train_df = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))
train_df["diagnosis"] = train_df["diagnosis"].astype(str)
train_df["filename"] = train_df["id_code"].astype(str) + ".png"

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_image,
    rescale=1 / 255.0,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20,
    zoom_range=0.2,
)

train_gen = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=os.path.join(BASE_PATH, "train_images"),
    x_col="filename",
    y_col="diagnosis",
    target_size=(DIM_X, DIM_Y),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    validate_filenames=False,
)

val_size = int(0.1 * len(train_df))
val_gen = train_datagen.flow_from_dataframe(
    dataframe=train_df.iloc[-val_size:],
    directory=os.path.join(BASE_PATH, "train_images"),
    x_col="filename",
    y_col="diagnosis",
    target_size=(DIM_X, DIM_Y),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
    validate_filenames=False,
)



## === cell 4
base_model = EfficientNetB0(
    weights="imagenet", include_top=False, input_shape=(DIM_X, DIM_Y, 3), pooling="avg"
)
base_model.trainable = True  # fine‑tune whole model

inputs = tf.keras.Input(shape=(DIM_X, DIM_Y, 3))
x = base_model(inputs, training=True)
x = FixedDropout(0.3)(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=Adam(learning_rate=1e-4), loss=KAPPA_LOSS, metrics=[kappa_keras]
)

model.fit(train_gen, epochs=3, validation_data=val_gen, verbose=1)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
OperatorNotAllowedInGraphError            Traceback (most recent call last)
/tmp/ipykernel_11/3335664247.py in <cell line: 0>()
     16 
     17 # Train briefly (adjust epochs as needed)
---> 18 model.fit(train_gen, epochs=3, validation_data=val_gen, verbose=1)
     19 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_11/1192231480.py in kappa_keras(y_true, y_pred)
      9     observed = tf.math.confusion_matrix(y_true, y_pred, num_classes=num_ratings)
     10     num_scored_items = K.shape(y_true)[0]
---> 11     weights = K.expand_dims(K.arange(num_ratings), axis=-1) - K.expand_dims(
     12         K.arange(num_ratings), axis=0
     13     )

OperatorNotAllowedInGraphError: Using a symbolic `tf.Tensor` as a Python `bool` is not allowed. You can attempt the following resolutions to the problem: If you are running in Graph mode, use Eager execution mode or decorate this function with @tf.function. If you are using AutoGraph, you can try decorating this function with @tf.function. If that does not work, then you may be using an unsupported feature or your source code may not be visible to AutoGraph. See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/limitations.md#access-to-source-code for more information.

## === cell 5
test_df = pd.read_csv(os.path.join(BASE_PATH, "test.csv"))
test_df["filename"] = test_df["id_code"].astype(str) + ".png"

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_image, rescale=1 / 255.0
)

test_gen = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=os.path.join(BASE_PATH, "test_images"),
    x_col="filename",
    y_col=None,
    target_size=(DIM_X, DIM_Y),
    batch_size=BATCH_SIZE,
    class_mode=None,
    shuffle=False,
    validate_filenames=False,
)

pred_probs = model.predict(test_gen, verbose=1)
pred_labels = np.argmax(pred_probs, axis=1)

submission = pd.read_csv(os.path.join(BASE_PATH, "sample_submission.csv"))
submission["diagnosis"] = pred_labels
submission.to_csv("submission.csv", index=False)

print("Submission file saved as submission.csv")
