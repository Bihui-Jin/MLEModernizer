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

0.8189281547268961

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import gc
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, confusion_matrix

import tensorflow as tf

tf.random.set_seed(42)

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Input,
    Dropout,
    Flatten,
    Dense,
    BatchNormalization,
)
from tensorflow.keras.callbacks import (
    Callback,
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)
from tensorflow.keras.optimizers import Adam

IMG_DIM = 256
BATCH_SIZE = 32
CHANNEL_SIZE = 3
NUM_CLASSES = 5

INPUT_FOLDER = "../input/aptos2019-blindness-detection/"
TEST_IMAGES_DIR = os.path.join(INPUT_FOLDER, "test_images/")

print("Root:", os.listdir("../"))
print("Input:", os.listdir("../input/"))
print("Dataset folder:", os.listdir(INPUT_FOLDER))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
test_df = pd.read_csv(os.path.join(INPUT_FOLDER, "test.csv"))
test_df["id_code"] = test_df["id_code"].apply(lambda x: f"{x}.png")




## === cell 2
def label_convert(y_val):
    y_val = y_val.astype(int).sum(axis=1) - 1
    return y_val


def crop(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    thresh = 5

    rowMaxes = gray.max(axis=1)
    top = 0
    while top < len(rowMaxes) and rowMaxes[top] < thresh:
        top += 1
    bottom = len(rowMaxes) - 1
    while bottom > 0 and rowMaxes[bottom] < thresh:
        bottom -= 1

    middleRow = gray[int((bottom - top) / 2)]
    left = 0
    while left < len(middleRow) and middleRow[left] < thresh:
        left += 1
    right = len(middleRow) - 1
    while right > 0 and middleRow[right] < thresh:
        right -= 1

    height = bottom - top
    width = right - left

    if height < 100 or width < 100:
        return bgr
    return bgr[top:bottom, left:right]


def colourfulEyes(bgr, weight=4, gamma=15):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)
    y = cv2.addWeighted(y, weight, cv2.GaussianBlur(y, (0, 0), gamma), -weight, 128)
    ycc_modified = cv2.merge((y, cr, cb))
    return cv2.cvtColor(ycc_modified, cv2.COLOR_YCrCb2BGR)


def processImageBgrToRgb(bgr):
    if bgr is None:
        bgr = np.full((IMG_DIM, IMG_DIM, 3), 128, dtype=np.uint8)
    modified = crop(bgr)
    modified = cv2.resize(modified, (IMG_DIM, IMG_DIM))
    modified = colourfulEyes(modified)
    return cv2.cvtColor(modified, cv2.COLOR_BGR2RGB)




## === cell 3
def dataGenerator(jitter=0.1):
    datagen = image.ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=bool(jitter > 0.01),
        vertical_flip=bool(jitter > 0.01),
        rotation_range=int(800 * jitter),
        brightness_range=[1 - jitter, 1 + jitter],
        channel_shift_range=int(30 * jitter),
        zoom_range=[(1 - jitter), (1 + jitter / 2)],
        fill_mode="reflect",
    )
    return datagen




## === cell 4
def create_model():
    model = Sequential()
    model.add(
        DenseNet121(
            weights=None,
            include_top=False,
            input_shape=(IMG_DIM, IMG_DIM, CHANNEL_SIZE),
        )
    )
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation="sigmoid"))
    return model


model = create_model()

weight_path = os.path.join("../input/densenetmulti/dense-multi-2015-run.h5")
if os.path.exists(weight_path):
    model.load_weights(weight_path)
    print("Loaded pretrained weights.")
else:
    print("Weight file not found; using randomly initialized model.")

model.compile(
    optimizer=Adam(learning_rate=5e-5), loss="binary_crossentropy", metrics=["accuracy"]
)
gc.collect()




## === cell 5
block_size = 500
total = test_df.shape[0]
y_pred_list = np.zeros(total, dtype=np.int64)

for start in range(0, total, block_size):
    gc.collect()
    end = min(start + block_size, total)

    img_batch = np.empty(
        (end - start, IMG_DIM, IMG_DIM, CHANNEL_SIZE), dtype=np.float32
    )

    for i, filename in enumerate(test_df.iloc[start:end]["id_code"]):
        img_path = os.path.join(TEST_IMAGES_DIR, filename)
        bgr = cv2.imread(img_path)
        if bgr is None:
            bgr = np.full((IMG_DIM, IMG_DIM, 3), 128, dtype=np.uint8)
        img_batch[i] = processImageBgrToRgb(bgr)

    img_batch = img_batch / 255.0

    preds = model.predict(img_batch, batch_size=BATCH_SIZE, verbose=0)
    preds_binary = preds > 0.5
    y_pred_list[start:end] = label_convert(preds_binary)

    print(f"{start} - {end} finished")

test_df["diagnosis"] = y_pred_list
submission_path = "submission.csv"
test_df[["id_code", "diagnosis"]].to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

## --- ERROR in outputing the csv:
Invalid submission: Submission must have the same id_codes as answers
