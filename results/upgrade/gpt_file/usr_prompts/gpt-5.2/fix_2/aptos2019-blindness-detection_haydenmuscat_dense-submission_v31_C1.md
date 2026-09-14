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

0.8787278172684493

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
import psutil
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, confusion_matrix


import keras
from keras.preprocessing import image
from keras.models import Sequential
from keras.applications import DenseNet121
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dropout, Dense
from keras.optimizers import Adam

IMG_DIM = 224
BATCH_SIZE = 32
CHANNELS = 3
NUM_CLASSES = 5

_CANDIDATE_INPUT_FOLDERS = [
    "../input/aptos2019-blindness-detection/",
    "/kaggle/input/aptos2019-blindness-detection/",
    "../input/",
    "/kaggle/input/",
]

INPUT_FOLDER = None
for p in _CANDIDATE_INPUT_FOLDERS:
    if os.path.exists(os.path.join(p, "train.csv")) and os.path.exists(
        os.path.join(p, "test.csv")
    ):
        INPUT_FOLDER = p if p.endswith("/") else (p + "/")
        break

if INPUT_FOLDER is None:
    for base in ["../input", "/kaggle/input"]:
        if os.path.exists(base):
            for root, dirs, files in os.walk(base):
                if "train.csv" in files and "test.csv" in files:
                    INPUT_FOLDER = root if root.endswith("/") else (root + "/")
                    break
            if INPUT_FOLDER is not None:
                break

if INPUT_FOLDER is None:
    raise FileNotFoundError(
        "Could not locate aptos2019-blindness-detection dataset folder containing train.csv/test.csv"
    )

print("INPUT_FOLDER =", INPUT_FOLDER)
print("CPU count:", psutil.cpu_count())
print("Listing INPUT_FOLDER:", os.listdir(INPUT_FOLDER)[:20])

TRAIN_IMG_DIR = os.path.join(INPUT_FOLDER, "train_images")
TEST_IMG_DIR = os.path.join(INPUT_FOLDER, "test_images")

if not os.path.isdir(TRAIN_IMG_DIR) or not os.path.isdir(TEST_IMG_DIR):
    nested = os.path.join(INPUT_FOLDER, "aptos2019-blindness-detection")
    if os.path.exists(nested):
        INPUT_FOLDER = nested if nested.endswith("/") else (nested + "/")
        TRAIN_IMG_DIR = os.path.join(INPUT_FOLDER, "train_images")
        TEST_IMG_DIR = os.path.join(INPUT_FOLDER, "test_images")

print("TRAIN_IMG_DIR exists:", os.path.isdir(TRAIN_IMG_DIR), TRAIN_IMG_DIR)
print("TEST_IMG_DIR exists:", os.path.isdir(TEST_IMG_DIR), TEST_IMG_DIR)

gc.collect()




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def crop(gray, img, percent_smaller):
    thresh = 8

    top = 0
    left = 0
    bottom = gray.shape[0] - 1
    right = gray.shape[1] - 1

    middleCol = gray[:, int(gray.shape[1] / 2)] > thresh
    while top < bottom and middleCol[top] == 0:
        top += 1
    while bottom > top and middleCol[bottom] == 0:
        bottom -= 1

    middleRow = gray[int(gray.shape[0] / 2)] > thresh
    while left < right and middleRow[left] == 0:
        left += 1
    while right > left and middleRow[right] == 0:
        right -= 1

    height = bottom - top
    width = right - left

    bottom -= int(percent_smaller * height)
    top += int(percent_smaller * height)
    right -= int(percent_smaller * width)
    left += int(percent_smaller * width)

    if height < 100 or width < 100 or top >= bottom or left >= right:
        return img

    return img[top:bottom, left:right]


def benSimple(img, weight=4, gamma=20):
    bens = cv2.addWeighted(
        img, weight, cv2.GaussianBlur(img, (0, 0), gamma), -weight, 128
    )
    return bens


def reflectAndSquareUp(img):
    height = img.shape[0]
    width = img.shape[1]

    if height > width:
        offset = int((height - width) / 2)
        return img[offset : offset + width]
    else:
        if len(img.shape) == 3:
            new_img = np.zeros((width, width, img.shape[2]), np.uint8)
        else:
            new_img = np.zeros((width, width), np.uint8)

        h1 = int((width - height) / 2)
        h2 = h1 + height
        new_img[h1:h2, :] = img

        for i in range(h1):
            new_img[h1 - i] = img[i]
        for i in range(width - h2):
            new_img[h2 + i] = img[height - i - 1]

        return new_img


def circleMask(img):
    if img.shape[0] != img.shape[1]:
        return img

    dim = img.shape[0]
    half = int(dim / 2)

    circle_mask = np.zeros((dim, dim), np.uint8)
    circle_mask = cv2.circle(circle_mask, (half, half), half, 1, thickness=-1)

    return cv2.bitwise_and(img, img, mask=circle_mask)


def adjust_gamma(image_in, gamma=1.0):
    gamma = float(gamma)
    if not np.isfinite(gamma) or gamma <= 0:
        gamma = 1.0
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    return cv2.LUT(image_in, table)


def processBenWeird(bgr):
    if bgr is None:
        return np.full((IMG_DIM, IMG_DIM, 3), 128, dtype=np.uint8)

    green = bgr[:, :, 1]  # use green as a greyscale
    cropped = crop(green, bgr, 0.02)
    squared = reflectAndSquareUp(cropped)
    resized = cv2.resize(squared, (2 * IMG_DIM, 2 * IMG_DIM))
    circled = circleMask(resized)

    med = np.median(circled)
    if med <= 0:
        g = 1.0
    else:
        g = 1 + np.log(90) - np.log(med)
    equalised = adjust_gamma(circled, g)

    resized_again = cv2.resize(benSimple(equalised), (IMG_DIM, IMG_DIM))
    return cv2.cvtColor(resized_again, cv2.COLOR_BGR2RGB)




## === cell 2
def dataGenerator(jitter=0.1):
    datagen = image.ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True and (jitter > 0.01),
        vertical_flip=True and (jitter > 0.01),
        rotation_range=int(800 * jitter),
        brightness_range=[1 - jitter, 1],
        channel_shift_range=int(30 * jitter),
        zoom_range=[(1 - jitter), (1 + jitter / 2)],
        fill_mode="reflect",
    )
    return datagen




## === cell 3
figure = plt.figure(figsize=(22, 20))


def test_datagen_plot():
    sample_df = pd.read_csv(f"{INPUT_FOLDER}test.csv")
    sample_df.id_code = sample_df.id_code.apply(lambda x: x + ".png")

    img_list = np.empty((32, IMG_DIM, IMG_DIM, 3), dtype=np.uint8)
    for i, filename in enumerate(sample_df[:32].id_code):
        bgr = cv2.imread(os.path.join(TEST_IMG_DIR, filename))
        img_list[i, :, :, :] = processBenWeird(bgr)

    datagen_sample = dataGenerator(0.03).flow(img_list, shuffle=True)

    for x in datagen_sample:
        for j in range(16):
            ax = figure.add_subplot(4, 4, j + 1)
            img = np.clip(x[j], 0, 1)
            plt.imshow(img)
            plt.axis("off")
        break


gc.collect()



## === cell 4
train_df = pd.read_csv(f"{INPUT_FOLDER}train.csv")
test_df = pd.read_csv(f"{INPUT_FOLDER}test.csv")

train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.15,
    random_state=42,
    stratify=train_df["diagnosis"].values,
)

train_split = train_df.iloc[train_idx].reset_index(drop=True)
val_split = train_df.iloc[val_idx].reset_index(drop=True)


def to_onehot(y, num_classes=NUM_CLASSES):
    y = np.asarray(y).astype(int)
    oh = np.zeros((len(y), num_classes), dtype=np.float32)
    oh[np.arange(len(y)), y] = 1.0
    return oh


y_train_oh = to_onehot(train_split["diagnosis"].values)
y_val_oh = to_onehot(val_split["diagnosis"].values)

print("Train size:", len(train_split), "Val size:", len(val_split))
gc.collect()




## === cell 5
def create_model():
    model = Sequential()
    model.add(
        DenseNet121(
            weights=None, include_top=False, input_shape=(IMG_DIM, IMG_DIM, CHANNELS)
        )
    )
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation="sigmoid"))
    return model


model = create_model()

model.compile(
    optimizer=Adam(learning_rate=0.00005),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
gc.collect()




## === cell 6
class BenWeirdSequence(keras.utils.Sequence):
    def __init__(
        self, df, y_onehot=None, images_dir="", batch_size=32, jitter=0.0, shuffle=False
    ):
        self.df = df.reset_index(drop=True)
        self.y_onehot = y_onehot
        self.images_dir = images_dir
        self.batch_size = batch_size
        self.jitter = float(jitter)
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))
        self.datagen = dataGenerator(self.jitter)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_inds = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_inds]

        x = np.empty((len(batch_df), IMG_DIM, IMG_DIM, CHANNELS), dtype=np.uint8)
        for i, id_code in enumerate(batch_df["id_code"].values):
            fn = id_code + ".png" if not str(id_code).endswith(".png") else id_code
            bgr = cv2.imread(os.path.join(self.images_dir, fn))
            x[i] = processBenWeird(bgr)

        x = self.datagen.standardize(x.astype(np.float32))

        if self.y_onehot is None:
            return x
        y = self.y_onehot[batch_inds]
        return x, y


train_seq = BenWeirdSequence(
    train_split,
    y_train_oh,
    images_dir=TRAIN_IMG_DIR,
    batch_size=BATCH_SIZE,
    jitter=0.10,
    shuffle=True,
)
val_seq = BenWeirdSequence(
    val_split,
    y_val_oh,
    images_dir=TRAIN_IMG_DIR,
    batch_size=BATCH_SIZE,
    jitter=0.00,
    shuffle=False,
)

history = model.fit(
    train_seq,
    validation_data=val_seq,
    epochs=2,
    workers=2,
    use_multiprocessing=False,
    verbose=1,
)
gc.collect()




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3955866222.py in <cell line: 0>()
     41 # Train for a small number of epochs to ensure runtime safety while still producing meaningful predictions.
     42 # (No early stopping added; fixed epochs to satisfy constraints.)
---> 43 train_seq = BenWeirdSequence(
     44     train_split,
     45     y_train_oh,

/tmp/ipykernel_11/3955866222.py in __init__(self, df, y_onehot, images_dir, batch_size, jitter, shuffle)
     11         self.shuffle = shuffle
     12         self.indices = np.arange(len(self.df))
---> 13         self.datagen = dataGenerator(self.jitter)
     14         self.on_epoch_end()
     15 

/tmp/ipykernel_11/1490733791.py in dataGenerator(jitter)
      1 def dataGenerator(jitter=0.1):
----> 2     datagen = image.ImageDataGenerator(
      3         rescale=1.0 / 255,
      4         horizontal_flip=True and (jitter > 0.01),
      5         vertical_flip=True and (jitter > 0.01),

AttributeError: module 'keras.api.preprocessing.image' has no attribute 'ImageDataGenerator'

## === cell 7
def make_predictions(d_set, jitters=5):
    if d_set == "test":
        images_dir = TEST_IMG_DIR
        df = pd.read_csv(f"{INPUT_FOLDER}test.csv")
    elif d_set == "train":
        images_dir = TRAIN_IMG_DIR
        df = pd.read_csv(f"{INPUT_FOLDER}train.csv")[["id_code"]]
    else:
        raise ValueError("d_set must be 'test' or 'train'")

    block_size = 256
    total = df.index.size
    predictions = np.zeros((total, NUM_CLASSES), dtype=np.float32)

    print(f"Making predictions on the {d_set} dataset. Total: {total}")

    for start in range(0, total, block_size):
        end = min(start + block_size, total)

        img_block = np.empty((end - start, IMG_DIM, IMG_DIM, CHANNELS), dtype=np.uint8)
        for i, id_code in enumerate(df.iloc[start:end].id_code.values):
            filename = id_code + ".png"
            bgr = cv2.imread(os.path.join(images_dir, filename))
            img_block[i] = processBenWeird(bgr)

        prediction_jitters = np.zeros(
            (len(img_block), jitters, NUM_CLASSES), dtype=np.float32
        )
        jit = 0.0
        for j in range(jitters):
            datagen = dataGenerator(jit).flow(
                img_block, shuffle=False, batch_size=BATCH_SIZE
            )
            pred = model.predict(datagen, steps=len(datagen), verbose=0)
            prediction_jitters[:, j] = pred
            gc.collect()
            jit += 0.02

        predictions[start:end] = np.median(prediction_jitters, axis=1)
        print(f"{start} - {end} finished")
        gc.collect()

    return predictions




## === cell 8
def label_convert(preds):
    y_val = preds > 0.5
    out = y_val.astype(int).sum(axis=1) - 1
    out = np.clip(out, 0, 4).astype(int)
    return out


val_preds = model.predict(val_seq, steps=len(val_seq), verbose=0)
val_classes_pred = label_convert(val_preds)
val_true = val_split["diagnosis"].values.astype(int)
print("Val QWK:", cohen_kappa_score(val_true, val_classes_pred, weights="quadratic"))
print("Val confusion matrix:\n", confusion_matrix(val_true, val_classes_pred))

gc.collect()



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/232348118.py in <cell line: 0>()
      9 
     10 # Optional: quick sanity-check QWK on validation (not used for submission directly)
---> 11 val_preds = model.predict(val_seq, steps=len(val_seq), verbose=0)
     12 val_classes_pred = label_convert(val_preds)
     13 val_true = val_split["diagnosis"].values.astype(int)

NameError: name 'val_seq' is not defined

## === cell 9
test_predictions = make_predictions("test", jitters=5)
test_classes = label_convert(test_predictions)

print(test_predictions[:2])
print(test_classes[:10])

sub_df = pd.read_csv(f"{INPUT_FOLDER}test.csv")
sub_df["diagnosis"] = test_classes.astype(int)
sub_df.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head())
gc.collect()

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3977352947.py in <cell line: 0>()
----> 1 test_predictions = make_predictions("test", jitters=5)
      2 test_classes = label_convert(test_predictions)
      3 
      4 print(test_predictions[:2])
      5 print(test_classes[:10])

/tmp/ipykernel_11/59464019.py in make_predictions(d_set, jitters)
     29         jit = 0.0
     30         for j in range(jitters):
---> 31             datagen = dataGenerator(jit).flow(
     32                 img_block, shuffle=False, batch_size=BATCH_SIZE
     33             )

/tmp/ipykernel_11/1490733791.py in dataGenerator(jitter)
      1 def dataGenerator(jitter=0.1):
----> 2     datagen = image.ImageDataGenerator(
      3         rescale=1.0 / 255,
      4         horizontal_flip=True and (jitter > 0.01),
      5         vertical_flip=True and (jitter > 0.01),

AttributeError: module 'keras.api.preprocessing.image' has no attribute 'ImageDataGenerator'
