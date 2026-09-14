# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import gc, random
from sklearn.model_selection import train_test_split
import concurrent.futures

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("GPU memory growth error:", e)

tf.random.set_seed(2)
np.random.seed(0)
random.seed(0)




## === cell 1
"""
    Preprocessing using Ben Graham's method (Last competition's winner) 
    https://www.kaggle.com/ratthachat/aptos-updatedv14-preprocessing-ben-s-cropping
"""


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


def load_ben_color(image, IMG_SIZE, sigmaX=10):
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image




## === cell 2
"""
    Define a lightweight CNN model (replaces EfficientNet to avoid protobuf issues)
"""


def output_relu(x):
    return K.relu(x, max_value=4)


def get_model(version, IMG_SIZE):
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)
    x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)
    x = Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1, activation=output_relu, kernel_initializer="he_normal")(x)
    model = Model(inputs=inputs, outputs=x)
    return model




## === cell 3
"""
    Initialize and compile model
"""
IMG_SIZE = 300
model = get_model(3, IMG_SIZE)
model.compile(optimizer=Adam(learning_rate=1e-4), loss="mse")
TRAIN_BATCH_SIZE = 64




## === cell 4
"""
    Optimized Rounder
    https://www.kaggle.com/abhishek/optimizer-for-quadratic-weighted-kappa
    Objective: Minimizes mse between predictions and ground-truth labels
"""
import scipy.optimize as opt
from functools import partial
from sklearn import metrics


class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = None

    def _mse_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred < coef[1]:
                X_p[i] = 1
            elif pred < coef[2]:
                X_p[i] = 2
            elif pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return metrics.mean_squared_error(y, X_p)

    def fit(self, X, y):
        loss_partial = partial(self._mse_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = opt.minimize(loss_partial, initial_coef, method="nelder-mead")
        return self

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred < coef[1]:
                X_p[i] = 1
            elif pred < coef[2]:
                X_p[i] = 2
            elif pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def coefficients(self):
        if self.coef_ is None:
            return [0.5, 1.5, 2.5, 3.5]
        return self.coef_["x"]




## === cell 5
"""
    Build tf.data pipelines that read and preprocess images lazily.
    Added a cache to the validation pipeline to avoid loading the same
    images twice, and reuse the in‑memory label list for the OptimizedRounder
    to eliminate an extra pass over the validation set.
"""

train_csv_path = "/kaggle/input/aptos2019-blindness-detection/train.csv"
train_img_dir = "/kaggle/input/aptos2019-blindness-detection/train_images/"

train_df = pd.read_csv(train_csv_path)

train_ids, val_ids = train_test_split(
    train_df["id_code"].values,
    test_size=0.1,
    random_state=42,
    stratify=train_df["diagnosis"],
)

label_dict = dict(zip(train_df["id_code"], train_df["diagnosis"]))

train_paths = [os.path.join(train_img_dir, f"{idx}.png") for idx in train_ids]
train_labels = [label_dict.get(idx, np.nan) for idx in train_ids]

val_paths = [os.path.join(train_img_dir, f"{idx}.png") for idx in val_ids]
val_labels = [label_dict.get(idx, np.nan) for idx in val_ids]
val_labels_np = np.array(val_labels, dtype=np.float32)  # reuse instead of re‑reading


def _load_and_preprocess(path):
    """Read image from disk, apply Ben Graham preprocessing, convert to RGB and normalize."""
    img = cv2.imread(path.decode())  # path comes as bytes from tf.string
    if img is None:
        img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    img = load_ben_color(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    return img


def tf_preprocess(path, label):
    """TensorFlow wrapper around the numpy preprocessing."""
    img = tf.numpy_function(_load_and_preprocess, [path], tf.float32)
    img.set_shape([IMG_SIZE, IMG_SIZE, 3])
    return img, tf.cast(label, tf.float32)


train_ds = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .shuffle(buffer_size=len(train_labels), reshuffle_each_iteration=True)
    .map(tf_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(TRAIN_BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    .map(tf_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .cache()  # cache after preprocessing to avoid repeat I/O
    .batch(TRAIN_BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

del (
    train_df,
    train_ids,
    val_ids,
    label_dict,
    train_paths,
    train_labels,
    val_paths,
    val_labels,
)
gc.collect()

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3,
    verbose=2,
)

val_pred = model.predict(val_ds, batch_size=TRAIN_BATCH_SIZE).ravel()
optR = OptimizedRounder()
optR.fit(val_pred, val_labels_np)  # use pre‑stored labels, no extra dataset pass
best_coef = optR.coefficients()
print("Optimized coefficients:", best_coef)




## === cell 6
"""
    Predict on test set using a tf.data pipeline (lazy loading) to avoid
    holding all test images in memory at once.
"""

test_csv_path = "/kaggle/input/aptos2019-blindness-detection/test.csv"
test_img_dir = "/kaggle/input/aptos2019-blindness-detection/test_images/"
test_df = pd.read_csv(test_csv_path)
id_codes = test_df["id_code"].values

test_paths = [os.path.join(test_img_dir, f"{code}.png") for code in id_codes]

test_ds = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(
        lambda p: tf.numpy_function(_load_and_preprocess, [p], tf.float32),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    .batch(TRAIN_BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

batch_preds = model.predict(test_ds, verbose=0).ravel()
test_preds = batch_preds.astype("float32")  # already aligned with id_codes order




## === cell 7
"""
    Apply OptimizedRounder and write submission file.
"""
if best_coef is None or len(best_coef) != 4:
    best_coef = [0.5, 1.5, 2.5, 3.5]

final_pred = optR.predict(test_preds, best_coef).astype("int")
submission = pd.DataFrame({"id_code": id_codes, "diagnosis": final_pred})
submission_path = "./submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
unique, counts = np.unique(final_pred, return_counts=True)
print(dict(zip(unique, counts)))
