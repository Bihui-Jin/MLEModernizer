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
import os, sys

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")



## === cell 1
import gc
import cv2
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

tf.random.set_seed(2)
np.random.seed(0)

print("TF version:", tf.__version__)



## === cell 2
try:
    from tensorflow.keras.applications import (
        EfficientNetB0,
        EfficientNetB1,
        EfficientNetB2,
        EfficientNetB3,
        EfficientNetB4,
    )
except Exception:
    from tensorflow.keras.applications.efficientnet import (
        EfficientNetB0,
        EfficientNetB1,
        EfficientNetB2,
        EfficientNetB3,
        EfficientNetB4,
    )

try:
    from tensorflow.keras.applications.efficientnet import (
        preprocess_input as eff_preprocess,
    )
except Exception:
    from tensorflow.keras.applications import preprocess_input as eff_preprocess



## === cell 3
"""
    Preprocessing using Ben Graham's method (Last competition's winner) 
    https://www.kaggle.com/ratthachat/aptos-updatedv14-preprocessing-ben-s-cropping
"""


def crop_image_from_gray(img, tol=7):
    if img is None:
        return None
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
    return img


def load_ben_color(image, IMG_SIZE, sigmaX=10):
    image = crop_image_from_gray(image)
    if image is None:
        return None
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image




## === cell 4
"""
    Define model
"""


def output_relu(x):
    return K.relu(x, max_value=4)


def get_model(version, IMG_SIZE, weights=None):
    base_model = None
    if version == 0:
        base_model = EfficientNetB0(
            weights=weights, include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
    elif version == 1:
        base_model = EfficientNetB1(
            weights=weights, include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
    elif version == 2:
        base_model = EfficientNetB2(
            weights=weights, include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
    elif version == 3:
        base_model = EfficientNetB3(
            weights=weights, include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
    elif version == 4:
        base_model = EfficientNetB4(
            weights=weights, include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
    else:
        return None

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1, activation=output_relu, kernel_initializer="he_normal")(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model




## === cell 5
"""
    Initialize model
"""
IMG_SIZE = 300

model = get_model(3, IMG_SIZE, weights=None)

weight_candidates = [
    "/kaggle/input/pretrained-weights/model_b3.h5",
    "../input/pretrained-weights/model_b3.h5",
    "/kaggle/input/aptos2019-blindness-detection/model_b3.h5",
    "../input/aptos2019-blindness-detection/model_b3.h5",
]
loaded = False
for wpath in weight_candidates:
    if os.path.exists(wpath):
        model.load_weights(wpath)
        print("Loaded weights from:", wpath)
        loaded = True
        break

if not loaded:
    print(
        "WARNING: Pretrained weights not found. Falling back to ImageNet backbone weights."
    )
    model = get_model(3, IMG_SIZE, weights="imagenet")

model.compile(optimizer="adam", loss="mse")



## === cell 6
"""
    Optimized Rounder
    https://www.kaggle.com/abhishek/optimizer-for-quadratic-weighted-kappa
    Objective: Minimizes mse between predictions and ground-truth labels
"""
import scipy as sp
from functools import partial
from sklearn import metrics


class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _mse_loss(self, coef, X, y):
        bins = np.asarray(coef, dtype=np.float32)
        X_p = np.digitize(X, bins, right=False).astype(np.float32)
        ll = metrics.mean_squared_error(y, X_p)
        return ll

    def fit(self, X, y):
        loss_partial = partial(self._mse_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(
            loss_partial, initial_coef, method="nelder-mead"
        )

    def predict(self, X, coef):
        bins = np.asarray(coef, dtype=np.float32)
        return np.digitize(X, bins, right=False).astype(np.float32)

    def coefficients(self):
        return self.coef_["x"]




## === cell 7
INPUT_CANDIDATES = [
    "/kaggle/input/aptos2019-blindness-detection",
    "../input/aptos2019-blindness-detection",
    "/kaggle/data/aptos2019-blindness-detection",
]
INPUT_DIR = None
for p in INPUT_CANDIDATES:
    if os.path.exists(p):
        INPUT_DIR = p
        break
if INPUT_DIR is None:
    raise FileNotFoundError(
        f"Could not find aptos2019-blindness-detection in {INPUT_CANDIDATES}"
    )

train_csv_path = os.path.join(INPUT_DIR, "train.csv")
train_img_dir = os.path.join(INPUT_DIR, "train_images")
test_csv_path = os.path.join(INPUT_DIR, "test.csv")
test_img_dir = os.path.join(INPUT_DIR, "test_images")

for req_path, is_dir in [
    (train_csv_path, False),
    (test_csv_path, False),
    (train_img_dir, True),
    (test_img_dir, True),
]:
    if is_dir and not os.path.isdir(req_path):
        raise FileNotFoundError(f"Missing directory at: {req_path}")
    if (not is_dir) and (not os.path.exists(req_path)):
        raise FileNotFoundError(f"Missing file at: {req_path}")



## === cell 8
BATCH_SIZE = 16  # safe, deterministic; adjust only for memory if needed.


def batched_predict(ids, img_dir, model, img_size, batch_size):
    n = len(ids)
    out = np.empty(n, dtype="float32")
    batch_x = np.empty((batch_size, img_size, img_size, 3), dtype="float32")
    batch_idx = np.empty((batch_size,), dtype=np.int32)

    bi = 0
    for i, idv in enumerate(ids):
        img_path = os.path.join(img_dir, "{}.png".format(idv))
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            out[i] = 0.0
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_proc = load_ben_color(img_rgb, img_size)
        if img_proc is None:
            out[i] = 0.0
            continue

        batch_x[bi] = img_proc.astype("float32", copy=False)
        batch_idx[bi] = i
        bi += 1

        if bi == batch_size:
            X = eff_preprocess(batch_x.copy())  # copy to keep buffer reusable safely
            pred = model.predict(X, verbose=0).reshape(-1)
            out[batch_idx[:bi]] = pred[:bi].astype("float32", copy=False)
            bi = 0

        if (i + 1) % 512 == 0:
            gc.collect()

    if bi > 0:
        X = eff_preprocess(batch_x[:bi].copy())
        pred = model.predict(X, verbose=0).reshape(-1)
        out[batch_idx[:bi]] = pred[:bi].astype("float32", copy=False)

    return out


train_df = pd.read_csv(train_csv_path)
if not {"id_code", "diagnosis"}.issubset(train_df.columns):
    raise ValueError("train.csv must contain id_code and diagnosis columns")

train_ids = train_df["id_code"].astype(str).values
train_y = train_df["diagnosis"].astype(np.int64).values

train_pred = batched_predict(train_ids, train_img_dir, model, IMG_SIZE, BATCH_SIZE)

optR = OptimizedRounder()
optR.fit(train_pred, train_y)
coefficients = optR.coefficients()
coefficients = np.sort(np.asarray(coefficients, dtype="float32")).tolist()
print("Fitted coefficients:", coefficients)



## === cell 9
test_df = pd.read_csv(test_csv_path)
if "id_code" not in test_df.columns:
    raise ValueError("test.csv must contain id_code column")

id_code = test_df["id_code"].astype(str).values
test_prediction = batched_predict(id_code, test_img_dir, model, IMG_SIZE, BATCH_SIZE)

test_prediction_round = optR.predict(test_prediction, coefficients)
test_prediction_round = np.clip(np.rint(test_prediction_round), 0, 4).astype("uint8")

sub_df = pd.DataFrame(
    {"id_code": id_code, "diagnosis": test_prediction_round.astype(int)}
)
sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)

unique, counts = np.unique(test_prediction_round, return_counts=True)
tmp = dict(zip(unique.tolist(), counts.tolist()))
print("Prediction label distribution:", tmp)
print("Wrote", sub_path, "with shape:", sub_df.shape)
print("Submission head:\n", sub_df.head())
