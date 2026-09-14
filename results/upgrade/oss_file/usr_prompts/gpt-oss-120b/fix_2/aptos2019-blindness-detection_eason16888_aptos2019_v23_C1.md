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

3.10

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

0.7657290256664656

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from keras.models import Sequential
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Dropout,
    Flatten,
    Dense,
)
from keras.callbacks import ReduceLROnPlateau, Callback, ModelCheckpoint
from sklearn.metrics import cohen_kappa_score
from tensorflow.keras.utils import to_categorical
from keras import backend as K
from keras import regularizers
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle, class_weight
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from tensorflow.keras.applications import DenseNet121, EfficientNetB2
from tqdm import tqdm
import os
import gc
import matplotlib.pyplot as plt



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
"""
    Config
"""
IMG_SIZE = 224
BATCH_SIZE = 16


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


def load_ben_color(image, sigmaX=10):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image).astype("uint8")
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image


def preprocessing(image, sigmaX=10):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = crop_image_from_gray(image).astype("uint8")
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image.astype("float32") / 255.0




## === cell 2
try:
    base_model = tf.keras.applications.efficientnet.EfficientNetB2(
        include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
except Exception:
    base_model = tf.keras.applications.efficientnet.EfficientNetB2(
        include_top=False, weights=None, input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )



## === cell 3
flatten_layer = tf.keras.layers.Flatten()
dense_layer_1 = tf.keras.layers.Dense(4096, activation="relu")
Dropout_1 = tf.keras.layers.Dropout(0.6)
dense_layer_2 = tf.keras.layers.Dense(2048, activation="relu")
Dropout_2 = tf.keras.layers.Dropout(0.6)
dense_layer_3 = tf.keras.layers.Dense(1024, activation="relu")
prediction_layer = tf.keras.layers.Dense(5, activation="softmax")

model = tf.keras.models.Sequential(
    [
        base_model,
        flatten_layer,
        dense_layer_1,
        Dropout_1,
        dense_layer_2,
        Dropout_2,
        dense_layer_3,
        prediction_layer,
    ]
)

weight_path = "../input/eff-b2-model/eff_b2_model.h5"
if os.path.exists(weight_path):
    try:
        model.load_weights(weight_path)
        print("Loaded pretrained weights.")
    except Exception as e:
        print(f"Could not load weights from {weight_path}: {e}")
else:
    print("Weight file not found; using randomly initialized model.")



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/582517119.py in <cell line: 0>()
     22 # Attempt to load pre‑trained weights if they exist; otherwise continue with random weights.
     23 weight_path = "../input/eff-b2-model/eff_b2_model.h5"
---> 24 if os.path.exists(weight_path):
     25     try:
     26         model.load_weights(weight_path)

NameError: name 'os' is not defined

## === cell 4
test_csv_path = "../input/aptos2019-blindness-detection/test.csv"
if not os.path.exists(test_csv_path):
    test_csv_path = (
        "../input/aptos2019-blindness-detection/sample_submission.csv"  # fallback
    )
test_csv = pd.read_csv(test_csv_path)
id_code = test_csv["id_code"].values
test_prediction = np.empty(len(id_code), dtype="int64")



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/935137375.py in <cell line: 0>()
      1 # Prepare test IDs
      2 test_csv_path = "../input/aptos2019-blindness-detection/test.csv"
----> 3 if not os.path.exists(test_csv_path):
      4     test_csv_path = (
      5         "../input/aptos2019-blindness-detection/sample_submission.csv"  # fallback

NameError: name 'os' is not defined

## === cell 5
for i, img_id in enumerate(tqdm(id_code, desc="Predicting")):
    img_path = f"../input/aptos2019-blindness-detection/test_images/{img_id}.png"
    img = cv2.imread(img_path)
    if img is None:
        img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype="uint8")
    img = load_ben_color(img)
    X = np.expand_dims(img, axis=0)  # shape (1, IMG_SIZE, IMG_SIZE, 3)
    pred = model.predict(X, verbose=0)
    test_prediction[i] = np.argmax(pred).astype("int64")



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3904562274.py in <cell line: 0>()
      1 # Predict on test images
----> 2 for i, img_id in enumerate(tqdm(id_code, desc="Predicting")):
      3     img_path = f"../input/aptos2019-blindness-detection/test_images/{img_id}.png"
      4     img = cv2.imread(img_path)
      5     if img is None:

NameError: name 'tqdm' is not defined

## === cell 6
submission = pd.DataFrame({"id_code": id_code, "diagnosis": test_prediction})
submission.to_csv("submission.csv", index=False)
unique, counts = np.unique(test_prediction, return_counts=True)
print(dict(zip(unique, counts)))
print("Done!")

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4042606391.py in <cell line: 0>()
      1 # Create submission file
----> 2 submission = pd.DataFrame({"id_code": id_code, "diagnosis": test_prediction})
      3 submission.to_csv("submission.csv", index=False)
      4 unique, counts = np.unique(test_prediction, return_counts=True)
      5 print(dict(zip(unique, counts)))

NameError: name 'id_code' is not defined
