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

-0.012600939397712

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
DATA_PATH = "/kaggle/input/aptos2019-blindness-detection/"



## === cell 1
import os
import numpy as np
import pandas as pd
from PIL import Image

import cv2

import tensorflow as tf

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

STANDARDIZE_CROP_RATIO = 0.792
OVERCROP_THRESHOLD = 25
BLUE_LAYER_IDX = 2
RED_LAYER_IDX = 0


def autocrop_scale(
    path, IMG_DIM=(512, 512), EPSILON=7, standardize_crop=False, return_crop_only=False
):
    """
    Loads image from `path`, auto-crops away dark background, optionally resizes.
    Returns:
      - np.ndarray if return_crop_only=True
      - PIL.Image otherwise
    """
    data = cv2.imread(path, cv2.IMREAD_COLOR)
    if data is None:
        raise FileNotFoundError(f"Could not read image at: {path}")
    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

    gray_data = data.mean(axis=2)

    limit_h = np.where(gray_data.mean(axis=0) >= EPSILON)[0]
    limit_v = np.where(gray_data.mean(axis=1) >= EPSILON)[0]

    if len(limit_h) == 0 or len(limit_v) == 0:
        new_data = data
    else:
        horizontal = (limit_h[0], limit_h[-1])

        if standardize_crop and (
            abs((limit_h[-1] - limit_h[0]) - (limit_v[-1] - limit_v[0]))
            <= OVERCROP_THRESHOLD
        ):
            crop_v = STANDARDIZE_CROP_RATIO * (limit_v[-1] - limit_v[0]) / 2
            center = (limit_v[-1] + limit_v[0]) / 2
            vertical = (int(center - crop_v), int(center + crop_v))
        else:
            vertical = (limit_v[0], limit_v[-1])

        new_data = data[
            vertical[0] : vertical[1] + 1, horizontal[0] : horizontal[1] + 1, :
        ]

        if new_data.shape[0] < 100 or new_data.shape[1] < 100:
            new_data = data

    if return_crop_only:
        return new_data

    processed_img = Image.fromarray(new_data)
    if IMG_DIM is not None:
        processed_img = processed_img.resize(IMG_DIM)

    return processed_img


def contrast_enhance(img, sigma=10, gray=False):
    """
    Contrast enhancement. Expects img as np.ndarray in RGB.
    Returns np.ndarray in RGB.
    """
    if gray:
        img = img.mean(axis=2)
        img = np.dstack((img, img, img)).astype(np.uint8)
    return cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigma), -4, 128)


def standard_crop(
    path, IMG_DIM=(512, 512), ratio=4 / 3, cratio=592 / 386, contrast_fnc=None, **kwargs
):
    """
    Standardized crop; returns a 3-channel uint8 image array (H,W,3).
    """
    data = autocrop_scale(path, return_crop_only=True)  # np array RGB
    h, l = data.shape[0], data.shape[1]

    if h < 10 or l < 10:
        data = cv2.resize(data, dsize=IMG_DIM, interpolation=cv2.INTER_AREA)
        return data

    sample_column = data[:, 1, :]
    denom = max(sample_column.mean(axis=1).std(), 1e-6)
    edge = np.where((data.mean(axis=2)[:, 0] > sample_column.mean() + 5 / denom))
    edge = edge[0][edge[0] > h / 8]
    if len(edge) > 0:
        r = np.sqrt((edge[0] - h / 2) ** 2 + (l / 2) ** 2)
    else:
        r = l / 2

    delta_h = r - np.sqrt(max(r**2 - (cratio * r / 2) ** 2, 0.0))
    top = max(int(delta_h - (2 * r - h) / 2), 0)
    bottom = h - top
    if bottom > top + 5:
        data = data[top:bottom, ...]

    h, l = data.shape[0], data.shape[1]
    delta_l = int(max(l - h * ratio, 0) / 2)
    if l - 2 * delta_l > 5:
        data = data[:, delta_l : l - delta_l, :]

    data = cv2.resize(data, dsize=IMG_DIM, interpolation=cv2.INTER_AREA)

    if contrast_fnc is not None:
        data = contrast_fnc(data, **kwargs)

    return data




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
from tensorflow.keras.preprocessing.image import ImageDataGenerator



## === cell 3
from tensorflow.keras import layers, models

IMG_SIZE = (128, 128)

model = models.Sequential(
    [
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dense(5, activation="softmax"),
    ]
)



## === cell 4
submission_df = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))
submission_df["filename"] = submission_df["id_code"].astype(str) + ".png"

test_images_dir = os.path.join(DATA_PATH, "test_images")


def _preprocess_batch(x):
    x = x.astype(np.uint8)
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        if x.shape[-1] == 1:
            rgb = np.repeat(x[i], 3, axis=2)
        else:
            rgb = x[i]
        rgb = contrast_enhance(rgb, sigma=10, gray=False)
        gray = rgb.mean(axis=2, keepdims=True).astype(np.uint8)
        out[i] = gray
    return out


test_datagen = ImageDataGenerator(preprocessing_function=_preprocess_batch)

gen = test_datagen.flow_from_dataframe(
    dataframe=submission_df,
    directory=test_images_dir,
    x_col="filename",
    y_col=None,
    color_mode="grayscale",
    batch_size=32,
    shuffle=False,
    class_mode=None,
    target_size=IMG_SIZE,
    validate_filenames=True,
)

pred = model.predict(gen, verbose=1)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AxisError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4009040309.py in <cell line: 0>()
     39 
     40 # BUGFIX: predict_generator is deprecated/removed; use model.predict
---> 41 pred = model.predict(gen, verbose=1)
     42 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_11/4009040309.py in _preprocess_batch(x)
     14     for i in range(x.shape[0]):
     15         if x.shape[-1] == 1:
---> 16             rgb = np.repeat(x[i], 3, axis=2)
     17         else:
     18             rgb = x[i]

/usr/local/lib/python3.11/dist-packages/numpy/core/fromnumeric.py in repeat(a, repeats, axis)
    464 
    465     """
--> 466     return _wrapfunc(a, 'repeat', repeats, axis=axis)
    467 
    468 

/usr/local/lib/python3.11/dist-packages/numpy/core/fromnumeric.py in _wrapfunc(obj, method, *args, **kwds)
     58     try:
     59         return bound(*args, **kwds)
---> 60     except TypeError:
     61         # A TypeError occurs if the object does have such a method in its
     62         # class, but its signature is not identical to that of NumPy's. This

AxisError: axis 2 is out of bounds for array of dimension 2

## === cell 5
pred_labels = np.argmax(pred, axis=1).astype(int)
submission_out = submission_df[["id_code"]].copy()
submission_out["diagnosis"] = pred_labels

submission_out.to_csv("submission.csv", index=False)

print(submission_out.head())
print("Wrote submission.csv with shape:", submission_out.shape)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1106169105.py in <cell line: 0>()
----> 1 pred_labels = np.argmax(pred, axis=1).astype(int)
      2 submission_out = submission_df[["id_code"]].copy()
      3 submission_out["diagnosis"] = pred_labels
      4 
      5 # Ensure valid submission file with .csv suffix in working directory

NameError: name 'pred' is not defined

## === cell 6
try:
    submission_out.to_csv("../submission.csv", index=False)
except Exception as e:
    print("Could not write ../submission.csv:", repr(e))



## === cell 7
submission_out



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4058231654.py in <cell line: 0>()
----> 1 submission_out
      2 

NameError: name 'submission_out' is not defined

## === cell 8
submission_out.shape

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/639057193.py in <cell line: 0>()
----> 1 submission_out.shape

NameError: name 'submission_out' is not defined
