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

0.8878958060904764

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import *
from tensorflow.keras.applications import (
    EfficientNetB0,
    EfficientNetB1,
    EfficientNetB2,
    EfficientNetB3,
    EfficientNetB4,
)
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os, gc, random
from sklearn.model_selection import train_test_split
import concurrent.futures  # will be used for parallel image loading

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



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
    Define model
"""


def output_relu(x):
    return K.relu(x, max_value=4)


def get_model(version, IMG_SIZE):
    if version == 0:
        base_model = EfficientNetB0(
            weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
    elif version == 1:
        base_model = EfficientNetB1(
            weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
    elif version == 2:
        base_model = EfficientNetB2(
            weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
    elif version == 3:
        base_model = EfficientNetB3(
            weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
    elif version == 4:
        base_model = EfficientNetB4(
            weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
    else:
        return None

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1, activation=output_relu, kernel_initializer="he_normal")(x)
    model = Model(inputs=base_model.input, outputs=x)
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
        return self.coef_["x"]




## === cell 5
"""
    Define a Keras Sequence for efficient batch‑wise image loading.
    This avoids loading the entire training set into memory up‑front.
"""


class TrainSequence(tf.keras.utils.Sequence):
    def __init__(
        self, id_list, label_dict, img_dir, img_size, batch_size, shuffle=True
    ):
        self.id_list = np.array(id_list)
        self.label_dict = label_dict
        self.img_dir = img_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.id_list) / self.batch_size))

    def __getitem__(self, idx):
        batch_ids = self.id_list[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_images = np.empty(
            (len(batch_ids), self.img_size, self.img_size, 3), dtype=np.float32
        )
        batch_labels = np.empty((len(batch_ids),), dtype=np.float32)
        for i, code in enumerate(batch_ids):
            img_path = os.path.join(self.img_dir, f"{code}.png")
            img = cv2.imread(img_path)
            img = load_ben_color(img, self.img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            batch_images[i] = img
            batch_labels[i] = self.label_dict.get(code, np.nan)
        return batch_images, batch_labels

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.id_list)




## === cell 6
"""
    Train the model on the provided training data and fit the OptimizedRounder.
    Training data is now streamed via TrainSequence to reduce load time.
"""
train_csv_path = "../input/aptos2019-blindness-detection/train.csv"
train_img_dir = "../input/aptos2019-blindness-detection/train_images/"

train_df = pd.read_csv(train_csv_path)

train_ids, val_ids = train_test_split(
    train_df["id_code"].values,
    test_size=0.1,
    random_state=42,
    stratify=train_df["diagnosis"],
)

label_dict = dict(zip(train_df["id_code"], train_df["diagnosis"]))


def _process_image(idx):
    """Helper for parallel image loading; returns (image array, label)."""
    img_path = os.path.join(train_img_dir, f"{idx}.png")
    img = cv2.imread(img_path)
    if img is None:
        return None, None
    img = load_ben_color(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    return img, label_dict.get(idx, np.nan)


def load_dataset_parallel(id_list):
    """Loads images and labels using a process pool, preserving order."""
    images = []
    labels = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for img, lbl in executor.map(_process_image, id_list, chunksize=16):
            if img is None:
                continue
            images.append(img)
            labels.append(lbl)
    return np.array(images), np.array(labels, dtype="float32")


X_val, y_val = load_dataset_parallel(val_ids)

train_seq = TrainSequence(
    train_ids, label_dict, train_img_dir, IMG_SIZE, TRAIN_BATCH_SIZE, shuffle=True
)

model.fit(
    train_seq,
    validation_data=(X_val, y_val),
    epochs=5,
    verbose=2,
    workers=4,
    use_multiprocessing=True,
)

val_pred = model.predict(X_val, batch_size=TRAIN_BATCH_SIZE).ravel()
optR = OptimizedRounder()
optR.fit(val_pred, y_val)
best_coef = optR.coefficients()
print("Optimized coefficients:", best_coef)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3434289841.py in <cell line: 0>()
     51 )
     52 
---> 53 model.fit(
     54     train_seq,
     55     validation_data=(X_val, y_val),

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    117             return fn(*args, **kwargs)
    118         except Exception as e:
--> 119             filtered_tb = _process_traceback_frames(e.__traceback__)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`

TypeError: TensorFlowTrainer.fit() got an unexpected keyword argument 'workers'

## === cell 7
"""
    Predict on test set using the trained model.
    Image loading is parallelized with a process pool (same improvement as training).
"""
test_csv_path = "../input/aptos2019-blindness-detection/sample_submission.csv"
test_img_dir = "../input/aptos2019-blindness-detection/test_images/"
test_df = pd.read_csv(test_csv_path)
id_codes = test_df["id_code"].values


def _process_test_image(code):
    img_path = os.path.join(test_img_dir, f"{code}.png")
    img = cv2.imread(img_path)
    if img is None:
        return None, None
    img = load_ben_color(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    return img, code


test_images = []
valid_indices = []  # keep track of which ids were successfully loaded

with concurrent.futures.ProcessPoolExecutor() as executor:
    for img, code in executor.map(_process_test_image, id_codes, chunksize=16):
        if img is None:
            continue
        test_images.append(img)
        valid_indices.append(np.where(id_codes == code)[0][0])

test_images_arr = np.array(test_images)
batch_preds = model.predict(test_images_arr, batch_size=TRAIN_BATCH_SIZE).ravel()

test_preds = np.zeros(len(id_codes), dtype="float32")
test_preds[valid_indices] = batch_preds



## === cell 8
"""
    Apply OptimizedRounder and write submission file.
"""
final_pred = optR.predict(test_preds, best_coef).astype("int")
submission = pd.DataFrame({"id_code": id_codes, "diagnosis": final_pred})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
unique, counts = np.unique(final_pred, return_counts=True)
print(dict(zip(unique, counts)))

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1999876243.py in <cell line: 0>()
      2     Apply OptimizedRounder and write submission file.
      3 """
----> 4 final_pred = optR.predict(test_preds, best_coef).astype("int")
      5 submission = pd.DataFrame({"id_code": id_codes, "diagnosis": final_pred})
      6 submission_path = "submission.csv"

NameError: name 'optR' is not defined
