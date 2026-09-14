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

0.9067101186443274

# 6. Current score

0.01983

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the environment-breaking TensorFlow/Keras import that triggers the protobuf `MessageFactory` error by switching to `tensorflow.keras` consistently and removing the deprecated `tensorflow.set_random_seed` import. I also make the input paths robust (avoid referencing a missing `../input/densenetmulti` directory) and ensure `INPUT_FOLDER`, `test_df`, and the model are defined in the right order so later cells can run. Finally, I update the deprecated `predict_generator` call to `predict`, keep the same inference logic/thresholding, and always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.01983) has done: 'I fix the environment-breaking protobuf/TensorFlow import crash that prevents the notebook from running at all, by forcing the pure-Python protobuf implementation before importing TensorFlow (a common Kaggle workaround for the `MessageFactory.GetPrototype` error). I also make the input path resolution robust for both `/kaggle/input/...` and relative `../input/...` layouts, so `train_images/` and `test_images/` are actually found. These changes are execution/stability fixes only; the model architecture, TTA inference, thresholding, and submission formatting remain identical so the score should move from 0.0 (no valid run) toward the expected level when the weights file is available. The script always write a valid `submission.csv` with the required `id_code,diagnosis` columns.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import gc
import numpy as np
import pandas as pd
import cv2

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.optimizers import Adam

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMG_DIM = 256
BATCH_SIZE = 32
CHANNEL_SIZE = 3
NUM_CLASSES = 5

candidate_inputs = [
    "/kaggle/input/aptos2019-blindness-detection",
    "../input/aptos2019-blindness-detection",
    "/kaggle/input",
    "../input",
]
INPUT_FOLDER = None
for p in candidate_inputs:
    if os.path.exists(p):
        INPUT_FOLDER = p
        break
if INPUT_FOLDER is None:
    INPUT_FOLDER = "."

TEST_IMAGES_DIR = os.path.join(INPUT_FOLDER, "test_images")
TRAIN_IMAGES_DIR = os.path.join(INPUT_FOLDER, "train_images")

print("INPUT_FOLDER:", INPUT_FOLDER)
print("Has test_images:", os.path.exists(TEST_IMAGES_DIR))
print("Has train_images:", os.path.exists(TRAIN_IMAGES_DIR))

WEIGHTS_PATH = None
candidate_paths = [
    "../input/densenetmulti/dense-multi-second-0.9038.h5",
    "../input/densenetmulti/dense-multi-second-0.9038.hdf5",
    os.path.join(INPUT_FOLDER, "dense-multi-second-0.9038.h5"),
]
for p in candidate_paths:
    if os.path.exists(p):
        WEIGHTS_PATH = p
        break

if WEIGHTS_PATH is None and os.path.exists("../input"):
    for root, _, files in os.walk("../input"):
        for f in files:
            if f == "dense-multi-second-0.9038.h5":
                WEIGHTS_PATH = os.path.join(root, f)
                break
        if WEIGHTS_PATH is not None:
            break

if WEIGHTS_PATH is None and os.path.exists("/kaggle/input"):
    for root, _, files in os.walk("/kaggle/input"):
        for f in files:
            if f == "dense-multi-second-0.9038.h5":
                WEIGHTS_PATH = os.path.join(root, f)
                break
        if WEIGHTS_PATH is not None:
            break

print("WEIGHTS_PATH:", WEIGHTS_PATH)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
test_csv_path = os.path.join(INPUT_FOLDER, "test.csv")
if not os.path.exists(test_csv_path):
    alt = os.path.join(INPUT_FOLDER, "aptos2019-blindness-detection", "test.csv")
    if os.path.exists(alt):
        test_csv_path = alt
        INPUT_FOLDER = os.path.dirname(test_csv_path)
        TEST_IMAGES_DIR = os.path.join(INPUT_FOLDER, "test_images")
        TRAIN_IMAGES_DIR = os.path.join(INPUT_FOLDER, "train_images")

test_df = pd.read_csv(test_csv_path)
test_df["id_code_png"] = test_df["id_code"].astype(str) + ".png"
print(test_df.head())
print("Using TEST_IMAGES_DIR:", TEST_IMAGES_DIR)




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
    while bottom >= 0 and rowMaxes[bottom] < thresh:
        bottom -= 1

    if top >= bottom:
        return bgr

    middleRow = gray[int((bottom - top) / 2)]
    left = 0
    while left < len(middleRow) and middleRow[left] < thresh:
        left += 1
    right = len(middleRow) - 1
    while right >= 0 and middleRow[right] < thresh:
        right -= 1

    height = bottom - top
    width = right - left

    if height < 100 or width < 100 or left >= right:
        return bgr

    return bgr[top:bottom, left:right]


def colourfulEyes(bgr, weight=4, gamma=15):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)
    y = cv2.addWeighted(y, weight, cv2.GaussianBlur(y, (0, 0), gamma), -weight, 128)
    ycc_modified = cv2.merge((y, cr, cb))
    modified = cv2.cvtColor(ycc_modified, cv2.COLOR_YCrCb2BGR)
    return modified


def processImageBgrToRgb(bgr):
    modified = crop(bgr)
    modified = cv2.resize(modified, (IMG_DIM, IMG_DIM))
    modified = colourfulEyes(modified)
    modified = cv2.cvtColor(modified, cv2.COLOR_BGR2RGB)
    return modified




## === cell 3
def dataGenerator(jitter=0.1):
    datagen = image.ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True and (jitter > 0.01),
        vertical_flip=True and (jitter > 0.01),
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

if WEIGHTS_PATH is not None:
    model.load_weights(WEIGHTS_PATH)
else:
    print("WARNING: pretrained weights not found; submission will be untrained.")

model.compile(
    optimizer=Adam(learning_rate=0.00005),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
gc.collect()



## === cell 5
block_size = 500
total = test_df.index.size

y_pred_list = np.zeros(total, dtype=int)

for start in range(0, total, block_size):
    gc.collect()

    end = min(start + block_size, total)

    img_list = np.empty((end - start, IMG_DIM, IMG_DIM, 3), dtype=np.float32)
    for i, filename in enumerate(test_df.iloc[start:end]["id_code_png"].values):
        path = os.path.join(TEST_IMAGES_DIR, filename)
        try:
            bgr = cv2.imread(path)
            if bgr is None:
                raise ValueError(f"cv2.imread failed for {path}")
            img_list[i, :, :, :] = processImageBgrToRgb(bgr)
        except Exception:
            img_list[i, :, :, :] = 128.0

    num = 7
    prediction_lists = np.zeros((len(img_list), num, 5), dtype=np.float32)

    for i in range(num):
        datagen = dataGenerator(0.03).flow(
            img_list, shuffle=False, batch_size=BATCH_SIZE
        )
        prediction_lists[:, i] = model.predict(datagen, steps=len(datagen), verbose=1)

    gc.collect()

    predictions = np.median(prediction_lists, axis=1)
    y_pred_list[start:end] = label_convert(predictions > 0.5)

    print(f"{start} - {end} finished")



## === cell 6
submission = pd.DataFrame(
    {"id_code": test_df["id_code"].values, "diagnosis": y_pred_list.astype(int)}
)
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
