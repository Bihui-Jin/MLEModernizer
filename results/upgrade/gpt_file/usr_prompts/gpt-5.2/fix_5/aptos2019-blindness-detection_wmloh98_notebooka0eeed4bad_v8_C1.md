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

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the TensorFlow import crash by forcing the pure-Python protobuf implementation before importing TensorFlow, which resolves the `MessageFactory.GetPrototype` error in Kaggle environments. Then I fix the `ImageDataGenerator` preprocessing function: Keras passes a single image (H,W,C) to `preprocessing_function`, not a batch, which caused the axis error; I rewrite it to handle both single images and batches safely without changing the intended preprocessing. Finally, I compile the model (required by some TF/Keras versions for `predict`) and ensure prediction completes so `submission.csv` is always written with the correct columns and row order.'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow/protobuf crash that prevents the notebook from importing TF by forcing the pure-Python protobuf implementation *and* setting the version flag early enough to take effect in Kaggle. I keep the model and preprocessing logic unchanged, but add a safe fallback so that if TensorFlow still can’t import in this environment, the script still produce a valid `submission.csv` with the correct format (score-neutral vs. your current 0.0 baseline). I also ensure paths resolve correctly and that the submission row order aligns exactly with `test.csv`.'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation **before any TensorFlow-related import** and by proactively removing already-imported `google.protobuf` modules so the env vars actually take effect in the Kaggle runtime. I also make the TensorFlow import more robust (still minimal) and keep your fallback path that writes a valid `submission.csv` if TF still can’t load, so you always get a submission file. The model, preprocessing, and prediction logic remain the same; changes are strictly to unblock execution and ensure the pipeline completes end-to-end. This should also allow real model predictions (instead of the all-zeros fallback), which is expected to move the score above your current 0.0 baseline.'

# 9. Code solution

## === cell 0
DATA_PATH = "/kaggle/input/aptos2019-blindness-detection/"



## === cell 1
import os
import sys

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

for k in list(sys.modules.keys()):
    if k.startswith("google.protobuf"):
        del sys.modules[k]

import numpy as np
import pandas as pd
from PIL import Image

import cv2

SEED = 42
np.random.seed(SEED)

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




## === cell 2
TF_AVAILABLE = True
try:
    import tensorflow as tf

    tf.random.set_seed(SEED)
except Exception as e:
    TF_AVAILABLE = False
    tf_import_error = repr(e)
    print(
        "WARNING: TensorFlow failed to import; will write fallback submission. Error:",
        tf_import_error,
    )



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
if TF_AVAILABLE:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator



## === cell 4
if TF_AVAILABLE:
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

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
else:
    IMG_SIZE = (128, 128)
    model = None



## === cell 5
submission_df = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))
submission_df["filename"] = submission_df["id_code"].astype(str) + ".png"

test_images_dir = os.path.join(DATA_PATH, "test_images")


def _preprocess_batch_or_single(x):
    x = np.asarray(x)

    def _process_one(img):
        img = img.astype(np.uint8)
        if img.ndim == 2:
            rgb = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[-1] == 1:
            rgb = np.repeat(img, 3, axis=-1)
        else:
            rgb = img
        rgb = contrast_enhance(rgb, sigma=10, gray=False)
        gray = rgb.mean(axis=2, keepdims=True).astype(np.uint8)
        return gray

    if x.ndim == 4:
        out = np.empty((x.shape[0], x.shape[1], x.shape[2], 1), dtype=np.uint8)
        for i in range(x.shape[0]):
            out[i] = _process_one(x[i])
        return out
    else:
        return _process_one(x)




## === cell 6
if TF_AVAILABLE:
    test_datagen = ImageDataGenerator(
        preprocessing_function=_preprocess_batch_or_single
    )

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

    steps = int(np.ceil(gen.n / gen.batch_size))
    pred = model.predict(gen, steps=steps, verbose=1)
else:
    pred = None



## === cell 7
if TF_AVAILABLE:
    pred_labels = np.argmax(pred, axis=1).astype(int)
    pred_labels = pred_labels[: len(submission_df)]
else:
    pred_labels = np.zeros(len(submission_df), dtype=int)

submission_out = submission_df[["id_code"]].copy()
submission_out["diagnosis"] = pred_labels

submission_out.to_csv("submission.csv", index=False)

print(submission_out.head())
print("Wrote submission.csv with shape:", submission_out.shape)



## === cell 8
try:
    submission_out.to_csv("../submission.csv", index=False)
except Exception as e:
    print("Could not write ../submission.csv:", repr(e))



## === cell 9
submission_out



## === cell 10
submission_out.shape
