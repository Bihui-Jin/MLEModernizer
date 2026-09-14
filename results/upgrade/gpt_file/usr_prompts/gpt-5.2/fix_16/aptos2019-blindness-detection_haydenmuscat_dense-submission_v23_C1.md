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

_IMREAD_REDUCED_FLAG = getattr(cv2, "IMREAD_REDUCED_COLOR_2", cv2.IMREAD_COLOR)


def _read_and_process_one(path_bytes):
    path = path_bytes.decode("utf-8")
    bgr = cv2.imread(path, _IMREAD_REDUCED_FLAG)
    if bgr is None:
        return (np.full((IMG_DIM, IMG_DIM, 3), 128, dtype=np.uint8),)
    try:
        arr = processBensColor(bgr)
        if arr is None or arr.shape != (IMG_DIM, IMG_DIM, 3):
            arr = np.full((IMG_DIM, IMG_DIM, 3), 128, dtype=np.uint8)
        return (arr.astype(np.uint8, copy=False),)
    except Exception:
        return (np.full((IMG_DIM, IMG_DIM, 3), 128, dtype=np.uint8),)


def _make_preproc_dataset(file_paths, parallel_calls=None):
    if parallel_calls is None:
        parallel_calls = max(2, min(12, (psutil.cpu_count() or 4)))

    ds = tf.data.Dataset.from_tensor_slices(file_paths)

    def _map_fn(p):
        x = tf.numpy_function(_read_and_process_one, [p], Tout=[tf.uint8])[0]
        x.set_shape((IMG_DIM, IMG_DIM, 3))
        return x

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)

    ds = ds.map(_map_fn, num_parallel_calls=parallel_calls, deterministic=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def _predict_block_with_jitters_fast(
    dg, img_block_uint8, base_seed, jitters, predict_batch_size
):
    n = img_block_uint8.shape[0]
    preds = np.empty((jitters, n, NUM_CLASSES), dtype=np.float32)

    for j in range(jitters):
        seed = int(base_seed + j * 100_000)
        flow = dg.flow(
            img_block_uint8,
            batch_size=predict_batch_size,
            shuffle=False,
            seed=seed,
        )
        preds[j] = model.predict(
            flow, steps=1 + (n - 1) // predict_batch_size, verbose=0
        ).astype(np.float32, copy=False)[:n]

    return np.median(preds, axis=0)


def make_predictions(d_set, jitters=5):
    images_dir = f"{INPUT_FOLDER}{d_set}_images/"
    df = pd.read_csv(f"{INPUT_FOLDER}{d_set}.csv")
    df["filename"] = df["id_code"].astype(str) + ".png"
    file_paths = (images_dir + df["filename"]).values.astype(str)

    total = df.shape[0]
    predictions = np.zeros((total, NUM_CLASSES), dtype=np.float32)

    block_size = 256
    predict_batch_size = max(64, BATCH_SIZE * 8)
    dg = dataGenerator(0.03)

    print(f"Making predictions on the {d_set} dataset. Total: {total}")

    ds = _make_preproc_dataset(file_paths)
    ds = ds.batch(block_size, drop_remainder=False)

    idx = 0
    for block_i, x_block in enumerate(ds):
        img_block = x_block.numpy()  # uint8, shape (n, IMG_DIM, IMG_DIM, 3)
        n = img_block.shape[0]
        start, end = idx, idx + n

        base_seed = SEED + (0 if d_set == "train" else 10_000_000) + start * 97
        predictions[start:end] = _predict_block_with_jitters_fast(
            dg,
            img_block,
            base_seed=base_seed,
            jitters=jitters,
            predict_batch_size=predict_batch_size,
        )

        idx = end
        print(f"{start} - {end} finished")
        if block_i % 4 == 3:
            gc.collect()

    return predictions


def label_convert(preds, thr=0.5):
    y_val = preds > thr
    labels = y_val.astype(int).sum(axis=1) - 1
    return np.clip(labels, 0, NUM_CLASSES - 1)


def _qwk_from_confmat(O):
    O = O.astype(np.float64, copy=False)
    N = O.shape[0]
    n = O.sum()
    if n <= 0:
        return 0.0

    act_hist = O.sum(axis=1)
    pred_hist = O.sum(axis=0)
    E = np.outer(act_hist, pred_hist) / n

    W = np.zeros((N, N), dtype=np.float64)
    denom = float((N - 1) ** 2)
    for i in range(N):
        di = (np.arange(N) - i) ** 2
        W[i, :] = di / denom

    num = (W * O).sum()
    den = (W * E).sum()
    if den == 0:
        return 0.0
    return 1.0 - (num / den)


def _tune_threshold_on_train_from_preds(train_preds, y_true):
    flat = np.unique(train_preds.ravel())
    if flat.size == 0:
        return 0.5

    vals = np.concatenate(([-1.0], flat, [2.0])).astype(np.float64, copy=False)
    vals.sort()
    thr_grid = (vals[:-1] + vals[1:]) * 0.5  # midpoints

    best_thr = 0.5
    best_kappa = -1e9

    P = train_preds.astype(np.float32, copy=False)
    y_true = y_true.astype(np.int64, copy=False)
    n = y_true.shape[0]

    chunk = 256
    for s in range(0, thr_grid.size, chunk):
        t = thr_grid[s : s + chunk].astype(np.float32, copy=False)  # (k,)
        cnt = (
            (P[:, None, :] > t[None, :, None]).sum(axis=2).astype(np.int16, copy=False)
        )
        labels = np.clip(cnt - 1, 0, NUM_CLASSES - 1).astype(
            np.int64, copy=False
        )  # (n,k)
        k = labels.shape[1]

        O = np.zeros((k, NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        jj = np.repeat(np.arange(k, dtype=np.int64), n)
        yy = np.tile(y_true, k)
        ll = labels.T.reshape(-1)
        np.add.at(O, (jj, yy, ll), 1)

        for j in range(k):
            qwk = _qwk_from_confmat(O[j])
            if qwk > best_kappa:
                best_kappa = qwk
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
