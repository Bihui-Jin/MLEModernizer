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

0.8733014191044104

# 6. Current score

0.09955

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.09955) has done: 'I fixed the import that caused the protobuf error, defined the `INPUT_FOLDER` early so it is available everywhere, removed the unused `ray` import, added a lightweight training pipeline that uses a pretrained ImageNet DenseNet121 (the original model architecture), trains it on the provided data for a few epochs, and then generates predictions for the test set.  The predictions are converted with `argmax` to the required label column and saved to `submission.csv` with the correct format.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import cv2
import gc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

import keras
from keras.preprocessing import image
from keras.models import Model
from keras.applications import DenseNet121
from keras.layers import GlobalAveragePooling2D, Dropout, Dense, Input
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

IMG_DIM = 256
CHANNELS = 3
NUM_CLASSES = 5  # diagnoses 0‑4
BATCH_SIZE = 32
EPOCHS = 5  # few epochs to stay within execution limits
INPUT_FOLDER = "../input/aptos2019-blindness-detection/"

print("Folders checked:")
print("cwd:", os.listdir("."))
print("input:", os.listdir("../input"))
print("aptos folder:", os.listdir(INPUT_FOLDER))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def crop(gray, img, percent_smaller):
    thresh = 8
    top = left = 0
    bottom = gray.shape[0] - 1
    right = gray.shape[1] - 1

    middleCol = gray[:, int(gray.shape[1] / 2)] > thresh
    while middleCol[top] == 0:
        top += 1
    while middleCol[bottom] == 0:
        bottom -= 1

    middleRow = gray[int(gray.shape[0] / 2)] > thresh
    while middleRow[left] == 0:
        left += 1
    while middleRow[right] == 0:
        right -= 1

    height = bottom - top
    width = right - left

    bottom -= int(percent_smaller * height)
    top += int(percent_smaller * height)
    right -= int(percent_smaller * width)
    left += int(percent_smaller * width)

    if height < 100 or width < 100:
        return img
    return img[top:bottom, left:right]


def benYCC(bgr, weight=4, gamma=20):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)
    y = cv2.addWeighted(y, weight, cv2.GaussianBlur(y, (0, 0), gamma), -weight, 128)
    ycc_mod = cv2.merge((y, cr, cb))
    return cv2.cvtColor(ycc_mod, cv2.COLOR_YCrCb2BGR)


def reflectAndSquareUp(img):
    h, w = img.shape[:2]
    if h > w:
        offset = (h - w) // 2
        return img[offset : offset + w]
    else:
        new_img = (
            np.zeros((w, w, img.shape[2]), np.uint8)
            if img.ndim == 3
            else np.zeros((w, w), np.uint8)
        )
        h1 = (w - h) // 2
        h2 = h1 + h
        new_img[h1:h2, :] = img
        new_img[:h1, :] = img[:h1, ::-1]
        new_img[h2:, :] = img[-(w - h2) :, ::-1]
        return new_img


def adjust_gamma(img, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype(
        "uint8"
    )
    return cv2.LUT(img, table)


def process_image(bgr):
    green = bgr[:, :, 1]
    if bgr.shape != (480, 640, 3):
        cropped = crop(green, bgr, 0.02)
        width = int(cropped.shape[1] * 0.9)
        height = int(width * 480 / 640)
        if height > cropped.shape[0]:
            height = cropped.shape[0] - 2
        h = (cropped.shape[0] - height) // 2
        w = (cropped.shape[1] - width) // 2
        bgr = cropped[h : h + height, w : w + width, :]
    reflected = reflectAndSquareUp(bgr)
    resized = cv2.resize(reflected, (IMG_DIM, IMG_DIM), interpolation=cv2.INTER_AREA)
    equalised = adjust_gamma(resized, 1 + np.log(90) - np.log(np.median(resized)))
    bens = benYCC(equalised, weight=3, gamma=20)
    return cv2.cvtColor(bens, cv2.COLOR_BGR2RGB)




## === cell 2
def data_generator(df, images_dir, batch_size, jitter=0.0, shuffle=True):
    """Yield batches of (images, labels) using the custom preprocessing."""
    indices = np.arange(len(df))
    if shuffle:
        np.random.shuffle(indices)

    while True:
        for start in range(0, len(df), batch_size):
            end = min(start + batch_size, len(df))
            batch_idx = indices[start:end]
            batch_images = np.empty(
                (len(batch_idx), IMG_DIM, IMG_DIM, CHANNELS), dtype=np.float32
            )
            batch_labels = np.empty((len(batch_idx),), dtype=np.int32)

            for i, idx in enumerate(batch_idx):
                fname = df.iloc[idx].id_code + ".png"
                path = os.path.join(images_dir, fname)
                bgr = cv2.imread(path)
                if bgr is None:
                    batch_images[i] = np.full(
                        (IMG_DIM, IMG_DIM, CHANNELS), 128, dtype=np.float32
                    )
                else:
                    img = process_image(bgr)
                    batch_images[i] = img.astype(np.float32) / 255.0
                batch_labels[i] = df.iloc[idx].diagnosis

            yield batch_images, batch_labels




## === cell 3
train_df = pd.read_csv(os.path.join(INPUT_FOLDER, "train.csv"))
train_df["diagnosis"] = train_df["diagnosis"].astype(int)

train_split, val_split = train_test_split(
    train_df, test_size=0.1, stratify=train_df["diagnosis"], random_state=42
)

train_gen = data_generator(
    train_split,
    os.path.join(INPUT_FOLDER, "train_images"),
    BATCH_SIZE,
    jitter=0.1,
    shuffle=True,
)
val_gen = data_generator(
    val_split,
    os.path.join(INPUT_FOLDER, "train_images"),
    BATCH_SIZE,
    jitter=0.0,
    shuffle=False,
)

steps_per_epoch = len(train_split) // BATCH_SIZE
validation_steps = len(val_split) // BATCH_SIZE



## === cell 4
base = DenseNet121(
    weights="imagenet", include_top=False, input_shape=(IMG_DIM, IMG_DIM, CHANNELS)
)
x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.5)(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)
model = Model(inputs=base.input, outputs=outputs)

model.compile(
    optimizer=Adam(lr=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

ckpt_path = "best_model.h5"
checkpoint = ModelCheckpoint(
    ckpt_path, monitor="val_accuracy", save_best_only=True, mode="max"
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_accuracy", factor=0.5, patience=2, mode="max", verbose=1
)

model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=validation_steps,
    epochs=EPOCHS,
    callbacks=[checkpoint, reduce_lr],
    verbose=2,
)

model.load_weights(ckpt_path)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3362783599.py in <cell line: 0>()
      9 
     10 model.compile(
---> 11     optimizer=Adam(lr=1e-4),
     12     loss="sparse_categorical_crossentropy",
     13     metrics=["accuracy"],

/usr/local/lib/python3.11/dist-packages/keras/src/optimizers/adam.py in __init__(self, learning_rate, beta_1, beta_2, epsilon, amsgrad, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, **kwargs)
     60         **kwargs,
     61     ):
---> 62         super().__init__(
     63             learning_rate=learning_rate,
     64             name=name,

/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/optimizer.py in __init__(self, *args, **kwargs)
     19 class TFOptimizer(KerasAutoTrackable, base_optimizer.BaseOptimizer):
     20     def __init__(self, *args, **kwargs):
---> 21         super().__init__(*args, **kwargs)
     22         self._distribution_strategy = tf.distribute.get_strategy()
     23 

/usr/local/lib/python3.11/dist-packages/keras/src/optimizers/base_optimizer.py in __init__(self, learning_rate, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, **kwargs)
     88             )
     89         if kwargs:
---> 90             raise ValueError(f"Argument(s) not recognized: {kwargs}")
     91 
     92         if name is None:

ValueError: Argument(s) not recognized: {'lr': 0.0001}

## === cell 5
def predict_dataset(dset_name, processing_function):
    images_dir = os.path.join(INPUT_FOLDER, f"{dset_name}_images")
    df = pd.read_csv(os.path.join(INPUT_FOLDER, f"{dset_name}.csv"))
    df["id_code"] = df["id_code"].astype(str)

    batch_size = BATCH_SIZE
    total = len(df)
    predictions = np.empty((total, NUM_CLASSES), dtype=np.float32)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        block = np.empty((end - start, IMG_DIM, IMG_DIM, CHANNELS), dtype=np.float32)
        for i, idx in enumerate(range(start, end)):
            fname = df.iloc[idx].id_code + ".png"
            path = os.path.join(images_dir, fname)
            bgr = cv2.imread(path)
            if bgr is None:
                block[i] = (
                    np.full((IMG_DIM, IMG_DIM, CHANNELS), 128, dtype=np.float32) / 255.0
                )
            else:
                img = processing_function(bgr)
                block[i] = img.astype(np.float32) / 255.0
        preds = model.predict(block, verbose=0)
        predictions[start:end] = preds
    return predictions




## === cell 6
test_preds = predict_dataset("test", process_image)
test_classes = np.argmax(test_preds, axis=1)

test_df = pd.read_csv(os.path.join(INPUT_FOLDER, "test.csv"))
test_df["diagnosis"] = test_classes
submission_path = "submission.csv"
test_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
