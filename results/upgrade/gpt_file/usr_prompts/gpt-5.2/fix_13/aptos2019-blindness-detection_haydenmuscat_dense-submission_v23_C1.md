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

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import gc
import numpy as np
import pandas as pd
import cv2
import psutil
import matplotlib.pyplot as plt

from sklearn.metrics import cohen_kappa_score, confusion_matrix  # kept (may be unused)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.optimizers import Adam

IMG_DIM = 364
BATCH_SIZE = 16
CHANNEL_SIZE = 3
NUM_CLASSES = 5

SEED = 1337
os.environ.setdefault("PYTHONHASHSEED", str(SEED))
try:
    tf.keras.utils.set_random_seed(SEED)
except Exception:
    pass
try:
    tf.config.threading.set_intra_op_parallelism_threads(
        max(1, (psutil.cpu_count() or 2) // 2)
    )
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

try:
    cv2.setNumThreads(max(1, min(4, (psutil.cpu_count() or 4) // 2)))
except Exception:
    pass


def _find_input_folder():
    """
    Robustly locate the competition dataset folder.
    """
    candidates = [
        "../input/aptos2019-blindness-detection/",
        "/kaggle/input/aptos2019-blindness-detection/",
        "../kaggle/input/aptos2019-blindness-detection/",
        "/kaggle/data/aptos2019-blindness-detection/",
        "../kaggle/data/aptos2019-blindness-detection/",
    ]
    for p in candidates:
        if os.path.exists(os.path.join(p, "train.csv")) and os.path.exists(
            os.path.join(p, "test.csv")
        ):
            return p if p.endswith("/") else p + "/"

    roots = [
        "../input",
        "/kaggle/input",
        "../kaggle/input",
        "../kaggle/data",
        "/kaggle/data",
    ]
    for r in roots:
        cand = os.path.join(r, "aptos2019-blindness-detection")
        if os.path.exists(os.path.join(cand, "train.csv")) and os.path.exists(
            os.path.join(cand, "test.csv")
        ):
            return cand + "/"

    raise FileNotFoundError(
        "Could not locate aptos2019-blindness-detection folder with train.csv/test.csv"
    )


INPUT_FOLDER = _find_input_folder()

print("TF version:", tf.__version__)
print("INPUT_FOLDER =", INPUT_FOLDER)
print("cpu_count =", psutil.cpu_count())
print("Listing INPUT_FOLDER:", os.listdir(INPUT_FOLDER)[:20])
gc.collect()




## === cell 1
def crop(gray, img, percent_smaller):

    thresh = 8

    h, w = gray.shape[:2]
    top = 0
    left = 0
    bottom = h - 1
    right = w - 1

    middleCol = gray[:, w // 2] > thresh
    if not middleCol.any():
        return img
    top = int(np.argmax(middleCol))
    bottom = int(h - 1 - np.argmax(middleCol[::-1]))
    if top >= bottom:
        return img

    middleRow = gray[h // 2, :] > thresh
    if not middleRow.any():
        return img
    left = int(np.argmax(middleRow))
    right = int(w - 1 - np.argmax(middleRow[::-1]))
    if left >= right:
        return img

    height = bottom - top
    width = right - left

    bottom -= int(percent_smaller * height)
    top += int(percent_smaller * height)
    right -= int(percent_smaller * width)
    left += int(percent_smaller * width)

    if height < 100 or width < 100:
        print("Error: squareUp: bottom:", bottom, "top:", top)
        print("Error: squareUp: right:", right, "left:", left)
        return img

    return img[top:bottom, left:right]


def bensYCC(bgr, weight=4, gamma=10):

    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)

    y = cv2.addWeighted(y, weight, cv2.GaussianBlur(y, (0, 0), gamma), -weight, 128)

    ycc_modified = cv2.merge((y, cr, cb))
    bens = cv2.cvtColor(ycc_modified, cv2.COLOR_YCrCb2BGR)

    return bens


def bensGray(gray, weight=4, gamma=10):

    bens = cv2.addWeighted(
        gray, weight, cv2.GaussianBlur(gray, (0, 0), gamma), -weight, 128
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
            new_img[:h1, :] = img[:h1, :][::-1, :]

        bottom_pad = width - h2
        if bottom_pad > 0:
            new_img[h2:, :] = img[height - bottom_pad : height, :][::-1, :]

        return new_img


def circleMask(img):

    if img.shape[0] != img.shape[1]:
        print("Error: circle mask assumes square image")
        return img

    dim = img.shape[0]
    half = int(dim / 2)

    circle_mask = np.zeros((dim, dim), np.uint8)
    circle_mask = cv2.circle(circle_mask, (half, half), half, 1, thickness=-1)

    return cv2.bitwise_and(img, img, mask=circle_mask)


def clahe_gray(gray, clipLimit=4.0, grid=8):

    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(grid, grid))
    return clahe.apply(gray)


_GAMMA_LUT_CACHE = {}


def adjust_gamma(image_, gamma=1.0):
    invGamma = 1.0 / gamma
    table = _GAMMA_LUT_CACHE.get(invGamma)
    if table is None:
        x = np.arange(256, dtype=np.float32) / 255.0
        table = np.clip((x**invGamma) * 255.0, 0, 255).astype(np.uint8)
        _GAMMA_LUT_CACHE[invGamma] = table
    return cv2.LUT(image_, table)


def processBensColor(bgr):

    green = bgr[:, :, 1]  # use green as a greyscale
    cropped = crop(green, bgr, 0.02)

    squared = reflectAndSquareUp(cropped)
    resized = cv2.resize(squared, (IMG_DIM, IMG_DIM))
    circled = circleMask(resized)

    med = float(np.median(circled))
    if med <= 0:
        med = 1.0
    equalised = adjust_gamma(circled, 1 + np.log(100) - np.log(med))

    return cv2.cvtColor(bensYCC(equalised), cv2.COLOR_BGR2RGB)




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
            rotation_range=int(800 * jitter),
            brightness_range=[1 - jitter, 1],
            channel_shift_range=int(30 * jitter),
            zoom_range=[(1 - jitter), (1 + jitter / 2)],
            fill_mode="reflect",
        )
        _DATAGEN_CACHE[key] = dg
    return dg




## === cell 3
figure = plt.figure(figsize=(22, 20))

RUN_PLOTTING = False


def test_datagen_plot():
    sample_df = pd.read_csv(f"{INPUT_FOLDER}test.csv")
    sample_df.id_code = sample_df.id_code.apply(lambda x: x + ".png")

    img_list = np.empty((32, IMG_DIM, IMG_DIM, 3), dtype=np.uint8)
    for i, filename in enumerate(sample_df[:32].id_code):
        try:
            bgr = cv2.imread(f"{INPUT_FOLDER}test_images/{filename}")
            if bgr is None:
                raise ValueError("cv2.imread returned None")
            img_list[i, :, :, :] = processBensColor(bgr)
        except Exception as e:
            img_list[i, :, :, :] = 128

    datagen_sample = dataGenerator(0.03).flow(img_list, shuffle=True, batch_size=16)

    for x in datagen_sample:
        for j in range(16):
            ax = figure.add_subplot(4, 4, j + 1)
            img_ = np.clip(x[j], 0, 1)
            plt.imshow(img_)
        break


if RUN_PLOTTING:
    try:
        test_datagen_plot()
    except Exception as e:
        print("Plotting skipped due to:", repr(e))

gc.collect()




## === cell 4
def _find_weights_file(preferred_rel_path):
    """
    Optimization: avoid an expensive full-disk os.walk which can cost many seconds.
    Correctness preserved because we still honor the exact preferred path, and if absent
    we safely fall back to ImageNet weights (same as previous behavior when not found).
    """
    if preferred_rel_path and os.path.exists(preferred_rel_path):
        return preferred_rel_path
    return None


WEIGHTS_PATH = _find_weights_file("../input/densenetmulti/ben_colour_-0.9126.h5")
print("Using WEIGHTS_PATH =", WEIGHTS_PATH)




## === cell 5
def create_model(dims, channels, weightsFile):
    imagenet_backbone = weightsFile is None

    model = Sequential()
    model.add(
        DenseNet121(
            weights="imagenet" if imagenet_backbone else None,
            include_top=False,
            input_shape=(dims, dims, channels),
        )
    )
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation="sigmoid"))

    if weightsFile is not None:
        model.load_weights(weightsFile)
        print("Loaded custom weights:", weightsFile)
    else:
        print(
            "Custom weights not found; using ImageNet-initialized DenseNet121 backbone."
        )

    return model


model = create_model(IMG_DIM, 3, WEIGHTS_PATH)

model.compile(
    optimizer=Adam(learning_rate=0.00005),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

gc.collect()




## === cell 6
from concurrent.futures import ThreadPoolExecutor

_CACHE_DIR = "./cache_benscolor"
os.makedirs(_CACHE_DIR, exist_ok=True)


def _memmap_paths(d_set):
    x_path = os.path.join(_CACHE_DIR, f"{d_set}_benscolor_dim{IMG_DIM}_uint8.mmap")
    ok_path = os.path.join(_CACHE_DIR, f"{d_set}_benscolor_dim{IMG_DIM}_uint8.ok")
    return x_path, ok_path


_IMREAD_REDUCED_FLAG = getattr(cv2, "IMREAD_REDUCED_COLOR_2", cv2.IMREAD_COLOR)


def _build_or_load_memmap(d_set, max_workers=None):
    images_dir = f"{INPUT_FOLDER}{d_set}_images/"
    df = pd.read_csv(f"{INPUT_FOLDER}{d_set}.csv")
    df["filename"] = df["id_code"].astype(str) + ".png"
    n = df.shape[0]

    x_path, ok_path = _memmap_paths(d_set)
    shape = (n, IMG_DIM, IMG_DIM, 3)

    if os.path.exists(ok_path) and os.path.exists(x_path):
        try:
            mm = np.memmap(x_path, mode="r", dtype=np.uint8, shape=shape)
            return df, mm
        except Exception:
            pass

    if os.path.exists(x_path):
        try:
            os.remove(x_path)
        except Exception:
            pass
    if os.path.exists(ok_path):
        try:
            os.remove(ok_path)
        except Exception:
            pass

    mm = np.memmap(x_path, mode="w+", dtype=np.uint8, shape=shape)

    if max_workers is None:
        max_workers = max(2, min(12, (psutil.cpu_count() or 4)))

    def _process_idx(i):
        fn = df.iloc[i]["filename"]
        bgr = cv2.imread(images_dir + fn, _IMREAD_REDUCED_FLAG)
        if bgr is None:
            return i, None
        try:
            arr = processBensColor(bgr)
            return i, arr
        except Exception:
            return i, None

    print(f"Building memmap cache for {d_set}: {n} images -> {x_path}")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, arr in ex.map(_process_idx, range(n), chunksize=32):
            if arr is None:
                mm[i, :, :, :] = 128
            else:
                mm[i, :, :, :] = arr

    mm.flush()
    with open(ok_path, "w") as f:
        f.write("ok\n")

    mm = np.memmap(x_path, mode="r", dtype=np.uint8, shape=shape)
    return df, mm


def _predict_block_with_jitters_reuse_iterator(
    dg, img_block_uint8, base_seed, jitters, predict_batch_size
):
    n = img_block_uint8.shape[0]
    aug_flow_bs = max(n, 64)

    x_cat = np.empty((jitters * n, IMG_DIM, IMG_DIM, 3), dtype=np.float32)
    for j in range(jitters):
        it = dg.flow(
            img_block_uint8,
            batch_size=aug_flow_bs,
            shuffle=False,
            seed=int(base_seed + j * 100_000),
        )
        x_aug = next(it)
        if x_aug.shape[0] != n:
            x_aug = x_aug[:n]
        x_cat[j * n : (j + 1) * n, :, :, :] = x_aug

    p_cat = model.predict(x_cat, batch_size=predict_batch_size, verbose=0)
    p_cat = p_cat.reshape((jitters, n, NUM_CLASSES)).transpose(1, 0, 2)
    return np.median(p_cat, axis=1)


def make_predictions(d_set, jitters=5):
    df, mm = _build_or_load_memmap(d_set)

    block_size = 512
    total = df.shape[0]
    predictions = np.zeros((total, NUM_CLASSES), dtype=np.float32)

    predict_batch_size = max(64, BATCH_SIZE * 8)
    print(f"Making predictions on the {d_set} dataset. Total: {total}")

    dg = dataGenerator(0.03)

    for start in range(0, total, block_size):
        end = min(start + block_size, total)

        img_block = mm[start:end]

        base_seed = SEED + (0 if d_set == "train" else 10_000_000) + start * 97
        predictions[start:end] = _predict_block_with_jitters_reuse_iterator(
            dg,
            img_block,
            base_seed=base_seed,
            jitters=jitters,
            predict_batch_size=predict_batch_size,
        )

        print(f"{start} - {end} finished")
        if (start // block_size) % 3 == 2:
            gc.collect()

    return predictions


def label_convert(preds, thr=0.5):
    y_val = preds > thr
    labels = y_val.astype(int).sum(axis=1) - 1
    return np.clip(labels, 0, NUM_CLASSES - 1)


def _tune_threshold_on_train_from_preds(train_preds, y_true):
    flat = np.unique(train_preds.ravel())
    if flat.size == 0:
        return 0.5

    vals = np.concatenate(([-1.0], flat, [2.0])).astype(np.float64, copy=False)
    vals.sort()
    thr_grid = (
        vals[:-1] + vals[1:]
    ) * 0.5  # midpoints; never equals a prediction exactly

    best_thr = 0.5
    best_kappa = -1e9

    P = train_preds.astype(np.float32, copy=False)

    chunk = 512  # thresholds per chunk
    for s in range(0, thr_grid.size, chunk):
        t = thr_grid[s : s + chunk].astype(np.float32, copy=False)  # (k,)
        cnt = (
            (P[:, None, :] > t[None, :, None]).sum(axis=2).astype(np.int16, copy=False)
        )
        labels = cnt - 1
        labels = np.clip(labels, 0, NUM_CLASSES - 1).astype(np.int8, copy=False)

        for j in range(labels.shape[1]):
            k = cohen_kappa_score(y_true, labels[:, j], weights="quadratic")
            if k > best_kappa:
                best_kappa = k
                best_thr = float(t[j])

    print(f"Best threshold on train: {best_thr:.6f} with QWK={best_kappa:.6f}")
    return best_thr


def _tune_threshold_on_train(jitters=3):
    train_df = pd.read_csv(INPUT_FOLDER + "train.csv")
    y_true = train_df["diagnosis"].values.astype(int)

    cache_path = f"train_preds_j{jitters}_img{IMG_DIM}_b{BATCH_SIZE}.npy"
    if os.path.exists(cache_path):
        train_preds = np.load(cache_path, allow_pickle=False)
    else:
        train_preds = make_predictions("train", jitters=jitters)
        np.save(cache_path, train_preds, allow_pickle=False)

    return _tune_threshold_on_train_from_preds(train_preds, y_true)


best_thr = _tune_threshold_on_train(jitters=3)

test_predictions = make_predictions("test", 5)
test_classes = label_convert(test_predictions, thr=best_thr)

print(test_predictions[:5])
print(test_classes[:5])

test_df = pd.read_csv(INPUT_FOLDER + "test.csv")
test_df["diagnosis"] = test_classes.astype(int)

test_df = test_df[["id_code", "diagnosis"]]
test_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", test_df.shape)
print(test_df.head())
