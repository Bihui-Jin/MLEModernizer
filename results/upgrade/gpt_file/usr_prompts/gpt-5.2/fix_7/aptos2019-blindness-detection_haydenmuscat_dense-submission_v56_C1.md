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
import gc
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import cohen_kappa_score

IMG_DIM = 256
BATCH_SIZE = 32
CHANNELS = 3
NUM_CLASSES = 5

NORMAL_WEIGHTS = "/kaggle/working/normal_trained.weights.h5"
FALLBACK_INPUT_WEIGHTS = "/kaggle/input/densenetmulti/normal.h5"

INPUT_FOLDER = "../input/aptos2019-blindness-detection/"

SEED = 1337
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("INPUT_FOLDER exists:", os.path.exists(INPUT_FOLDER))
print("Files in INPUT_FOLDER:", os.listdir(INPUT_FOLDER)[:10])
print("Has train_images:", os.path.exists(os.path.join(INPUT_FOLDER, "train_images")))
print("Has test_images:", os.path.exists(os.path.join(INPUT_FOLDER, "test_images")))
print("Has fallback pretrained weights:", os.path.exists(FALLBACK_INPUT_WEIGHTS))

try:
    cv2.setNumThreads(0)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

TRAIN_IMG_DIR = os.path.join(INPUT_FOLDER, "train_images") + "/"
TEST_IMG_DIR = os.path.join(INPUT_FOLDER, "test_images") + "/"

IMREAD_FLAGS = cv2.IMREAD_COLOR




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

        if h1 > 0:
            new_img[:h1, :] = img[:h1][::-1, :]
        tail = width - h2
        if tail > 0:
            new_img[h2:, :] = img[height - tail : height][::-1, :]

        return new_img


def circleMask(img):
    if img.shape[0] != img.shape[1]:
        return img

    dim = img.shape[0]
    half = int(dim / 2)

    circle_mask = np.zeros((dim, dim), np.uint8)
    circle_mask = cv2.circle(circle_mask, (half, half), half, 1, thickness=-1)

    return cv2.bitwise_and(img, img, mask=circle_mask)


def clahe_gray(gray, clipLimit=3.5, grid=4):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(grid, grid))
    return clahe.apply(gray)


_GAMMA_LUT_CACHE = {}


def adjust_gamma(image, gamma=1.0):
    lut = _GAMMA_LUT_CACHE.get(gamma)
    if lut is None:
        invGamma = 1.0 / gamma
        lut = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype(
            "uint8"
        )
        _GAMMA_LUT_CACHE[gamma] = lut
    return cv2.LUT(image, lut)


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
        if height <= 0 or width <= 0:
            test_crop = bgr
        else:
            h = int((cropped.shape[0] - height) / 2)
            w = int((cropped.shape[1] - width) / 2)
            test_crop = cropped[h : height + h, w : width + w, :]
            if test_crop.size == 0:
                test_crop = bgr
    else:
        test_crop = bgr

    resized = cv2.resize(test_crop, (IMG_DIM, IMG_DIM), interpolation=cv2.INTER_AREA)
    bens = benYCC(resized, weight=3, gamma=15)
    return cv2.cvtColor(bens, cv2.COLOR_BGR2RGB)


pre_process_function = processBenNormal




## === cell 2
_DATAGEN_CACHE = {}


def dataGenerator(jitter=0.1):
    key = float(jitter)
    dg = _DATAGEN_CACHE.get(key)
    if dg is None:
        dg = image.ImageDataGenerator(
            rescale=1.0 / 255,
            horizontal_flip=True and (jitter > 0.01),
            vertical_flip=True and (jitter > 0.01),
            zoom_range=[max(0.8, 1 - 5 * jitter), 1],
            rotation_range=int(600 * jitter),
            brightness_range=[1 - jitter / 3, 1 + jitter / 3],
            fill_mode="mirror",
            channel_shift_range=int(30 * jitter),
        )
        _DATAGEN_CACHE[key] = dg
    return dg




## === cell 3
def test_datagen_plot(processing_function, jitter=0.03):
    images_dir = f"{INPUT_FOLDER}test_images/"
    df = pd.read_csv(f"{INPUT_FOLDER}test.csv")
    df.id_code = df.id_code.apply(lambda x: x + ".png")

    figure = plt.figure(figsize=(10, 10))

    img_block = np.empty((100, IMG_DIM, IMG_DIM, CHANNELS), dtype=np.float32)
    for i, filename in enumerate(df[:100].id_code):
        try:
            bgr = cv2.imread(images_dir + filename)
            img_block[i, :, :, :] = processing_function(bgr)
        except Exception:
            img_block[i, :, :, :] = 128.0

    datagen_sample = dataGenerator(jitter).flow(img_block)
    for x in datagen_sample:
        for j in range(16):
            ax = figure.add_subplot(4, 4, j + 1)
            plt.imshow(x[j])
            plt.axis("off")
        break




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


def _make_train_generator(processing_function, jitter=0.1):
    train_df = pd.read_csv(f"{INPUT_FOLDER}train.csv").copy()
    train_df["filename"] = train_df["id_code"].astype(str) + ".png"
    train_df["diagnosis"] = train_df["diagnosis"].astype(int)

    def encode_multi(d):
        y = np.zeros(NUM_CLASSES, dtype=np.float32)
        y[: d + 1] = 1.0
        return y

    y = np.stack(train_df["diagnosis"].apply(encode_multi).values)

    images_dir = f"{INPUT_FOLDER}train_images/"

    datagen = dataGenerator(jitter)

    def gen():
        idx = np.arange(len(train_df))
        while True:
            np.random.shuffle(idx)
            for start in range(0, len(idx), BATCH_SIZE):
                batch_ids = idx[start : start + BATCH_SIZE]
                bs = len(batch_ids)
                x_batch = np.empty((bs, IMG_DIM, IMG_DIM, CHANNELS), dtype=np.uint8)
                y_batch = y[batch_ids]
                for i, ridx in enumerate(batch_ids):
                    bgr = cv2.imread(images_dir + train_df.iloc[ridx]["filename"])
                    x_batch[i] = processing_function(bgr)
                aug = datagen.flow(x_batch, y_batch, batch_size=bs, shuffle=False)
                xb, yb = next(aug)
                yield xb, yb

    steps = int(np.ceil(len(train_df) / BATCH_SIZE))
    return gen(), steps


def train_and_save_if_needed(weights_path):
    if os.path.exists(weights_path):
        print("Found existing weights:", weights_path)
        return

    if os.path.exists(FALLBACK_INPUT_WEIGHTS):
        print(
            "Copying pretrained weights from input to working:", FALLBACK_INPUT_WEIGHTS
        )
        with open(FALLBACK_INPUT_WEIGHTS, "rb") as fsrc, open(
            weights_path, "wb"
        ) as fdst:
            fdst.write(fsrc.read())
        return

    print("Training weights locally (pretrained weights not found)...")
    model = create_model(weights=None)

    train_gen, steps = _make_train_generator(processBenNormal, jitter=0.08)

    ckpt = ModelCheckpoint(
        weights_path,
        monitor="loss",
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        verbose=1,
    )
    rlrop = ReduceLROnPlateau(
        monitor="loss", factor=0.5, patience=1, min_lr=1e-6, verbose=1
    )

    EPOCHS = 2
    model.fit(
        train_gen,
        steps_per_epoch=steps,
        epochs=EPOCHS,
        callbacks=[ckpt, rlrop],
        verbose=1,
    )

    if not os.path.exists(weights_path):
        model.save_weights(weights_path)

    del model
    gc.collect()




## === cell 5
def _rng_for_jitter(jitter, block_start):
    seed = (
        SEED * 1000003 + int(jitter * 1e6) * 9176 + int(block_start) * 97
    ) & 0xFFFFFFFF
    return np.random.RandomState(seed)


def _aug_params_from_jitter(jitter):
    zoom_min = max(0.8, 1 - 5 * jitter)
    zoom_max = 1.0
    rot_deg = int(600 * jitter)
    bright_min = 1 - jitter / 3
    bright_max = 1 + jitter / 3
    ch_shift = int(30 * jitter)
    do_flip = jitter > 0.01
    return zoom_min, zoom_max, rot_deg, bright_min, bright_max, ch_shift, do_flip


def _precompute_tta_params(n, jitter, rng):
    zoom_min, zoom_max, rot_deg, bright_min, bright_max, ch_shift, do_flip = (
        _aug_params_from_jitter(jitter)
    )

    if do_flip:
        hflip = rng.rand(n) < 0.5
        vflip = rng.rand(n) < 0.5
    else:
        hflip = np.zeros(n, dtype=bool)
        vflip = np.zeros(n, dtype=bool)

    if rot_deg > 0:
        angles = rng.uniform(-rot_deg, rot_deg, size=n).astype(np.float32, copy=False)
    else:
        angles = np.zeros(n, dtype=np.float32)

    if zoom_max > zoom_min:
        zooms = rng.uniform(zoom_min, zoom_max, size=n).astype(np.float32, copy=False)
    else:
        zooms = np.full(n, zoom_max, dtype=np.float32)

    if bright_max > bright_min:
        brights = rng.uniform(bright_min, bright_max, size=n).astype(
            np.float32, copy=False
        )
    else:
        brights = np.ones(n, dtype=np.float32)

    if ch_shift > 0:
        shifts = rng.uniform(-ch_shift, ch_shift, size=(n, 3)).astype(
            np.float32, copy=False
        )
    else:
        shifts = np.zeros((n, 3), dtype=np.float32)

    return hflip, vflip, angles, zooms, brights, shifts


def _apply_tta_batch_uint8_to_float_with_params(
    imgs_uint8, jitter, params_slice, out_float
):
    hflip, vflip, angles, zooms, brights, shifts = params_slice
    n, h, w, c = imgs_uint8.shape
    _, _, rot_deg, bright_min, bright_max, ch_shift, _ = _aug_params_from_jitter(jitter)

    center = (w * 0.5, h * 0.5)
    cv2_flip = cv2.flip
    cv2_getRot = cv2.getRotationMatrix2D
    cv2_warp = cv2.warpAffine
    np_clip = np.clip

    for i in range(n):
        img = imgs_uint8[i]
        if hflip[i]:
            img = cv2_flip(img, 1)
        if vflip[i]:
            img = cv2_flip(img, 0)

        if rot_deg > 0 or zooms[i] != 1.0:
            M = cv2_getRot(center, float(angles[i]), float(zooms[i]))
            img = cv2_warp(
                img,
                M,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )

        img_f = img.astype(np.float32, copy=False)

        if ch_shift > 0:
            img_f = img_f + shifts[i][None, None, :]

        if bright_max != 1.0 or bright_min != 1.0:
            img_f = img_f * brights[i]

        np_clip(img_f, 0.0, 255.0, out=img_f)
        out_float[i, :, :, :] = img_f * (1.0 / 255.0)

    return out_float[:n]


@tf.function(reduce_retracing=True)
def _model_forward(model, xb):
    return model(xb, training=False)


def make_predictions(d_set, processing_function, model, jitters=5):
    images_dir = TRAIN_IMG_DIR if d_set == "train" else TEST_IMG_DIR
    df = pd.read_csv(f"{INPUT_FOLDER}{d_set}.csv")
    filenames = (df["id_code"].astype(str) + ".png").values

    block_size = 256
    total = filenames.shape[0]
    predictions = np.zeros((total, NUM_CLASSES), dtype=np.float32)

    print(f"Making predictions on the {d_set} dataset. Total: {total}")

    jitters_list = [0.0075 * i for i in range(jitters)]

    max_block_imgs = np.empty((block_size, IMG_DIM, IMG_DIM, CHANNELS), dtype=np.uint8)
    max_batch_float = np.empty(
        (BATCH_SIZE, IMG_DIM, IMG_DIM, CHANNELS), dtype=np.float32
    )
    prediction_jitters = np.empty((block_size, jitters, NUM_CLASSES), dtype=np.float32)
    out = np.empty((block_size, NUM_CLASSES), dtype=np.float32)

    imread = cv2.imread
    ceil = np.ceil
    tf_const = tf.constant  # slightly cheaper path when input is a NumPy array

    for start in range(0, total, block_size):
        end = min(start + block_size, total)
        n = end - start

        img_block = max_block_imgs[:n]
        for i, fn in enumerate(filenames[start:end]):
            bgr = imread(images_dir + fn, IMREAD_FLAGS)
            img_block[i, :, :, :] = processing_function(bgr)

        for ji, jitter in enumerate(jitters_list):
            rng = _rng_for_jitter(jitter, start)
            params = _precompute_tta_params(n, jitter, rng)

            steps = int(ceil(n / BATCH_SIZE))
            for s in range(steps):
                b0 = s * BATCH_SIZE
                b1 = min(b0 + BATCH_SIZE, n)
                bs = b1 - b0

                params_slice = (
                    params[0][b0:b1],
                    params[1][b0:b1],
                    params[2][b0:b1],
                    params[3][b0:b1],
                    params[4][b0:b1],
                    params[5][b0:b1],
                )
                xb_np = _apply_tta_batch_uint8_to_float_with_params(
                    img_block[b0:b1], jitter, params_slice, max_batch_float
                )

                xb = tf_const(xb_np, dtype=tf.float32)
                pred = _model_forward(model, xb)
                out[b0:b1] = pred.numpy().astype(np.float32, copy=False)

            prediction_jitters[:n, ji, :] = out[:n]

        predictions[start:end] = np.median(prediction_jitters[:n], axis=1)
        print(f"{start} - {end} finished")

        if (start // block_size) % 4 == 3:
            gc.collect()

    return predictions




## === cell 6
def prediction_convert_sum(predictions, thresholds):
    thr = np.asarray(thresholds, dtype=predictions.dtype)[None, :]
    thresholded = predictions > thr
    y_val = thresholded.astype(np.int32).sum(axis=1) - 1
    return y_val


def prediction_convert_highest(predictions, thresholds):
    thr = np.asarray(thresholds, dtype=predictions.dtype)[None, :]
    thresholded = predictions > thr  # (n,5) bool

    any_true = thresholded.any(axis=1)
    rev_idx = thresholded[:, ::-1].argmax(axis=1)
    y_val = np.zeros((predictions.shape[0],), dtype=np.int32)
    y_val[any_true] = (NUM_CLASSES - 1) - rev_idx[any_true]
    return y_val


def find_best_thresholds(train_predictions):
    print("Finding best thresholds...")
    prediction_convert = prediction_convert_sum

    train_df = pd.read_csv(f"{INPUT_FOLDER}train.csv")
    y_actual = train_df.diagnosis.astype(int).values

    thresholds = [0.5 for _ in range(NUM_CLASSES)]
    d_thresh = 0.25

    for sweep in range(5):
        for label in range(5):
            currKappa = cohen_kappa_score(
                y_actual,
                prediction_convert(train_predictions, thresholds),
                weights="quadratic",
            )

            thresholds[label] += d_thresh
            kappaUp = cohen_kappa_score(
                y_actual,
                prediction_convert(train_predictions, thresholds),
                weights="quadratic",
            )

            thresholds[label] -= 2 * d_thresh
            kappaDown = cohen_kappa_score(
                y_actual,
                prediction_convert(train_predictions, thresholds),
                weights="quadratic",
            )

            thresholds[label] += d_thresh

            if kappaUp > currKappa:
                thresholds[label] += d_thresh
            elif kappaDown > currKappa:
                thresholds[label] -= d_thresh

        d_thresh /= 2

    gc.collect()
    return thresholds




## === cell 7
train_and_save_if_needed(NORMAL_WEIGHTS)

model = create_model(NORMAL_WEIGHTS)

_dummy = tf.constant(np.zeros((1, IMG_DIM, IMG_DIM, CHANNELS), dtype=np.float32))
_ = _model_forward(model, _dummy).numpy()

preds = make_predictions("test", processBenNormal, model, jitters=5)

thresholds = [0.5 for _ in range(NUM_CLASSES)]
test_classes = prediction_convert_highest(preds, thresholds)

test_df = pd.read_csv(INPUT_FOLDER + "test.csv")
test_df["diagnosis"] = test_classes.astype(int)
test_df.to_csv("submission.csv", index=False)

print("Wrote submission.csv")
print(test_df.head())
print("submission.csv size:", os.path.getsize("submission.csv"), "bytes")
