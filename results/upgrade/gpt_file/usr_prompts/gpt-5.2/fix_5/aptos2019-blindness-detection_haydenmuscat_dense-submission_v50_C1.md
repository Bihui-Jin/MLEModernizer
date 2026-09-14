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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import gc
import numpy as np
import pandas as pd
import cv2

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

IMG_DIM = 256
BATCH_SIZE = 32
CHANNELS = 3
NUM_CLASSES = 3  # 2 stage ordinal via multi-label

NORMAL_WEIGHTS = "../input/densenetmulti/0.8822791912279122.h5"
STAGE_1 = "../input/densenetmulti/stage_1.h5"
STAGE_2 = "../input/densenetmulti/stage_2.h5"

DEFAULT_INPUT_FOLDER = "../input/aptos2019-blindness-detection/"
ALT_INPUT_FOLDER = "/kaggle/input/aptos2019-blindness-detection/"
if os.path.exists(DEFAULT_INPUT_FOLDER):
    INPUT_FOLDER = DEFAULT_INPUT_FOLDER
elif os.path.exists(ALT_INPUT_FOLDER):
    INPUT_FOLDER = ALT_INPUT_FOLDER
else:
    INPUT_FOLDER = "/kaggle/data/aptos2019-blindness-detection/"

print("Using INPUT_FOLDER:", INPUT_FOLDER)
print("INPUT_FOLDER exists:", os.path.exists(INPUT_FOLDER))
print(
    "Listing INPUT_FOLDER (first 20):",
    (
        sorted(os.listdir(INPUT_FOLDER))[:20]
        if os.path.exists(INPUT_FOLDER)
        else "MISSING"
    ),
)


def _safe_listdir(p):
    try:
        return os.listdir(p)
    except Exception as e:
        return f"ERROR: {e}"


print("Listing ../input (if any):", _safe_listdir("../input"))

np.random.seed(42)
tf.random.set_seed(42)
try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass
try:
    cv2.setNumThreads(0)
except Exception:
    pass

gc.collect()




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

    if height < 100 or width < 100 or bottom <= top or right <= left:
        return img

    return img[top:bottom, left:right]


def benYCC(bgr, weight=4, gamma=20):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)
    y = cv2.addWeighted(y, weight, cv2.GaussianBlur(y, (0, 0), gamma), -weight, 128)
    ycc_modified = cv2.merge((y, cr, cb))
    bens = cv2.cvtColor(ycc_modified, cv2.COLOR_YCrCb2BGR)
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
            if i < height:
                new_img[h1 - i] = img[i]

        for i in range(width - h2):
            if (height - i - 1) >= 0:
                new_img[h2 + i] = img[height - i - 1]

        return new_img


from functools import lru_cache


@lru_cache(maxsize=2048)
def _gamma_lut(gamma):
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    return table


def adjust_gamma(image_, gamma=1.0):
    return cv2.LUT(image_, _gamma_lut(float(gamma)))


def processBenNormal(bgr):
    if bgr is None:
        return np.full((IMG_DIM, IMG_DIM, CHANNELS), 128, dtype=np.uint8)

    green = bgr[:, :, 1]  # use green as a greyscale

    if bgr.shape != (480, 640, 3):
        cropped = crop(green, bgr, 0.02)
        width = int(cropped.shape[1] * 0.9)
        height = int(width * 480 / 640)
        if height > cropped.shape[0]:
            height = cropped.shape[0] - 2
        h = int((cropped.shape[0] - height) / 2)
        w = int((cropped.shape[1] - width) / 2)
        test_crop = cropped[h : height + h, w : width + w, :]
    else:
        test_crop = bgr

    reflected = reflectAndSquareUp(test_crop)
    resized = cv2.resize(reflected, (IMG_DIM, IMG_DIM), interpolation=cv2.INTER_AREA)

    med = np.median(resized)
    if med <= 0:
        med = 1.0
    equalised = adjust_gamma(resized, 1 + np.log(90) - np.log(med))
    bens = benYCC(equalised, weight=3, gamma=20)
    return cv2.cvtColor(bens, cv2.COLOR_BGR2RGB)


pre_process_function = processBenNormal




## === cell 2
def dataGenerator(jitter=0.1):
    datagen = image.ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True and (jitter > 0.01),
        vertical_flip=True and (jitter > 0.01),
        zoom_range=[max(0.8, 1 - 5 * jitter), 1],
        rotation_range=int(600 * jitter),
        brightness_range=[1 - jitter / 3, 1 + jitter / 3],
        fill_mode="mirror",
        channel_shift_range=int(30 * jitter),
    )
    return datagen




## === cell 3
RUN_PLOTS = False


def test_datagen_plot(processing_function, jitter=0.3):
    images_dir = f"{INPUT_FOLDER}test_images/"
    df = pd.read_csv(f"{INPUT_FOLDER}test.csv")
    df.id_code = df.id_code.apply(lambda x: x + ".png")

    img_block = np.empty((100, IMG_DIM, IMG_DIM, CHANNELS), dtype=np.uint8)
    for i, filename in enumerate(df[:100].id_code):
        bgr = cv2.imread(images_dir + filename)
        img_block[i, :, :, :] = processing_function(bgr)

    datagen_sample = dataGenerator(jitter).flow(img_block)
    fig = plt.figure(figsize=(22, 20))
    for x in datagen_sample:
        for j in range(16):
            ax = fig.add_subplot(4, 4, j + 1)
            ax.imshow(x[j])
            ax.axis("off")
        break
    plt.show()


if RUN_PLOTS:
    test_datagen_plot(pre_process_function)
gc.collect()




## === cell 4
def create_model(weights=None):
    model = Sequential()
    model.add(
        DenseNet121(
            weights=None, include_top=False, input_shape=(IMG_DIM, IMG_DIM, CHANNELS)
        )
    )
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation="sigmoid"))

    if weights is not None and os.path.exists(weights):
        model.load_weights(weights)

    model.compile(
        optimizer=Adam(learning_rate=0.00005),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _ordinal_targets(y):
    y = np.asarray(y).astype(int)
    t = np.zeros((len(y), NUM_CLASSES), dtype=np.float32)
    t[:, 0] = (y >= 1).astype(np.float32)
    t[:, 1] = (y >= 2).astype(np.float32)
    t[:, 2] = (y >= 3).astype(np.float32)
    return t




## === cell 5
def make_predictions(d_set, processing_function, model, jitters=5):
    images_dir = f"{INPUT_FOLDER}{d_set}_images/"
    df = pd.read_csv(f"{INPUT_FOLDER}{d_set}.csv")
    df.id_code = df.id_code.astype(str) + ".png"

    block_size = 512
    total = df.index.size
    predictions = np.zeros((total, NUM_CLASSES), dtype=np.float32)

    workers = max(1, min(4, (os.cpu_count() or 2) // 2))

    print(f"Making predictions on the {d_set} dataset. Total: {total}")

    id_codes = df.id_code.values  # avoid repeated pandas slicing overhead

    for start in range(0, total, block_size):
        end = min(start + block_size, total)
        cur_n = end - start

        img_block = np.empty((cur_n, IMG_DIM, IMG_DIM, CHANNELS), dtype=np.uint8)
        for i, filename in enumerate(id_codes[start:end]):
            bgr = cv2.imread(images_dir + filename)
            img_block[i, :, :, :] = processing_function(bgr)

        prediction_jitters = np.empty((cur_n, jitters, NUM_CLASSES), dtype=np.float32)

        jit = 0.0
        for j in range(jitters):
            datagen = dataGenerator(jit).flow(
                img_block, shuffle=False, batch_size=BATCH_SIZE
            )
            steps = len(datagen)  # compute once
            prediction_jitters[:, j] = model.predict(
                datagen,
                steps=steps,
                verbose=0,
                workers=workers,
                use_multiprocessing=True,
                max_queue_size=16,
            )
            jit += 0.02

        predictions[start:end] = np.median(prediction_jitters, axis=1)
        print(f"{start} - {end} finished")
        gc.collect()

    return predictions




## === cell 6
def label_convert_two_stage(stage_1_preds, stage_2_preds):
    thresh_1 = stage_1_preds > 0.5
    thresh_2 = stage_2_preds > 0.5

    y_val = thresh_1.astype(np.int32).sum(axis=1) - 1
    y_val_2 = thresh_2.astype(np.int32).sum(axis=1) + 1

    mask = y_val == 2
    y_val = y_val.astype(np.int32, copy=False)
    y_val[mask] = y_val_2[mask]
    return y_val




## === cell 7
HAVE_EXTERNAL = os.path.exists(STAGE_1) and os.path.exists(STAGE_2)
print("External weights available:", HAVE_EXTERNAL)

if not HAVE_EXTERNAL:
    train_df = pd.read_csv(f"{INPUT_FOLDER}train.csv")
    train_df["filename"] = train_df["id_code"].astype(str) + ".png"
    img_dir = f"{INPUT_FOLDER}train_images/"

    trn_df, val_df = train_test_split(
        train_df, test_size=0.15, random_state=42, stratify=train_df["diagnosis"]
    )

    def _load_block(df_slice):
        x = np.empty((len(df_slice), IMG_DIM, IMG_DIM, CHANNELS), dtype=np.uint8)
        y = _ordinal_targets(df_slice["diagnosis"].values)
        fns = df_slice["filename"].values
        for i, fn in enumerate(fns):
            bgr = cv2.imread(os.path.join(img_dir, fn))
            x[i] = processBenNormal(bgr)
        return x, y

    trn_take = min(1200, len(trn_df))
    val_take = min(400, len(val_df))
    trn_small = trn_df.sample(trn_take, random_state=42)
    val_small = val_df.sample(val_take, random_state=42)

    x_trn, y_trn = _load_block(trn_small)
    x_val, y_val = _load_block(val_small)

    datagen = dataGenerator(jitter=0.15)

    stage1_model = create_model(weights=None)

    ckpt1_path = "stage_1_trained.weights.h5"
    ckpt1 = ModelCheckpoint(
        ckpt1_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )
    rlr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=1, min_lr=1e-6, verbose=1
    )

    stage1_model.fit(
        datagen.flow(x_trn, y_trn, batch_size=BATCH_SIZE, shuffle=True),
        validation_data=(x_val.astype(np.float32) / 255.0, y_val),
        epochs=3,
        callbacks=[ckpt1, rlr],
        verbose=2,
    )
    stage1_model.load_weights(ckpt1_path)

    stage2_model = create_model(weights=None)
    ckpt2_path = "stage_2_trained.weights.h5"
    ckpt2 = ModelCheckpoint(
        ckpt2_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )
    stage2_model.fit(
        datagen.flow(x_trn, y_trn, batch_size=BATCH_SIZE, shuffle=True),
        validation_data=(x_val.astype(np.float32) / 255.0, y_val),
        epochs=3,
        callbacks=[ckpt2, rlr],
        verbose=2,
    )
    stage2_model.load_weights(ckpt2_path)

    val_p1 = stage1_model.predict(
        (x_val.astype(np.float32) / 255.0), batch_size=BATCH_SIZE, verbose=0
    )
    val_p2 = stage2_model.predict(
        (x_val.astype(np.float32) / 255.0), batch_size=BATCH_SIZE, verbose=0
    )
    val_pred = label_convert_two_stage(val_p1, val_p2)
    kappa = cohen_kappa_score(
        val_small["diagnosis"].values, val_pred, weights="quadratic"
    )
    print("Validation quadratic kappa (sanity):", kappa)

    del x_trn, y_trn, x_val, y_val
    gc.collect()
else:
    stage1_model = create_model(STAGE_1)
    stage2_model = create_model(STAGE_2)




## === cell 8
preds_1 = make_predictions("test", processBenNormal, stage1_model, jitters=5)
preds_2 = make_predictions("test", processBenNormal, stage2_model, jitters=5)

test_classes = label_convert_two_stage(preds_1, preds_2).astype(int)
test_classes = np.clip(test_classes, 0, 4)

print("Sample preds_1:", preds_1[:3])
print("Sample preds_2:", preds_2[:3])
print("Sample classes:", test_classes[:10])

test_df = pd.read_csv(INPUT_FOLDER + "test.csv")
test_df["diagnosis"] = test_classes.astype(int)

test_df = test_df[["id_code", "diagnosis"]]
test_df.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", test_df.shape)
print(test_df.head())
gc.collect()
