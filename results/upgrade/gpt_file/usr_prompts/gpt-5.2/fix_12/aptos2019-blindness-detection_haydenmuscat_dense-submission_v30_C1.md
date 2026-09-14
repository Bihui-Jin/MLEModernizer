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

0.8464910054382703

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -0.07122) has done: 'I fix the runtime errors by switching the deprecated/absent `ImageDataGenerator` import to the supported `keras.preprocessing.image.ImageDataGenerator`, and by avoiding the protobuf-related crash caused by importing standalone `keras` (use `tf.keras` consistently instead). I also correct `NUM_CLASSES` to 5 (labels are 0–4) so the model output shape matches the competition’s 5-class target and so predictions map correctly to `diagnosis`. These changes keep the same core DenseNet121 + GAP + Dropout + Dense(sigmoid) inference logic and the same threshold-sum label conversion, but make the pipeline run end-to-end and write a valid `submission.csv`. Paths and I/O remain the same, and the script still load external weights if present.'
- What this solution (achieved -0.01475) has done: 'I fix the protobuf crash that happens at import time by forcing TensorFlow to use the pure-Python protobuf implementation (a common Kaggle TF1/TF2 + protobuf mismatch) and by setting it before importing TensorFlow. Then I ensure the image folder paths are constructed robustly with `os.path.join` (avoids missing/duplicate slashes) while keeping the exact same preprocessing, model, jitters, and threshold-to-class conversion logic. Finally, I keep the submission generation identical but add a small safety check to guarantee `diagnosis` is within 0–4 and the CSV is written correctly.'
- What this solution (achieved -0.00198) has done: 'I fix the import-time protobuf crash that prevents TensorFlow from loading by forcing a compatible protobuf runtime setting and (if needed) downgrading protobuf within the notebook environment before importing TensorFlow. Then I keep your model, preprocessing, TTA/jitter prediction, and threshold-to-class conversion logic intact, but add a small safety fallback so the script can still run even if TensorFlow cannot be imported (it then produce a valid CSV with a neutral prediction rather than crashing). This should both unblock end-to-end execution and restore meaningful predictions (instead of a broken run), which is necessary to move the QWK score up toward the target. All paths and submission formatting remain unchanged.'
- What this solution (achieved 0.05723) has done: 'Your current score is far below the target, so we should make a minimal change that improves agreement with the ordinal 0–4 labels without changing the model or training loop. The biggest issue is the post-processing: summing 5 independent sigmoid outputs at a fixed 0.5 threshold is a poor fit for an ordinal 5-class task and often collapses predictions, hurting QWK. Keeping the exact same model and weights, we switch only the label conversion to a standard approach for 5-class sigmoid heads: take `argmax` over the 5 outputs (still produces 0–4) and optionally apply a tiny, deterministic class-bias calibration computed from the training label distribution to avoid degenerate outputs. This preserves architecture, preprocessing, and inference/TTA, but should move the kappa substantially upward toward your target.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

import sys
import gc
import math
import numpy as np
import pandas as pd
import cv2
import psutil
from concurrent.futures import ThreadPoolExecutor

try:
    cv2.setNumThreads(0)
except Exception:
    pass

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

TF_AVAILABLE = True
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.applications import DenseNet121
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
    from tensorflow.keras.optimizers import Adam

    try:
        os.environ.setdefault("PYTHONHASHSEED", "123")
        tf.random.set_seed(123)
        np.random.seed(123)
        try:
            tf.config.experimental.enable_op_determinism(True)
        except Exception:
            pass
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception:
        pass
except Exception as e:
    TF_AVAILABLE = False
    tf = None
    print("TensorFlow import failed; will fall back to safe submission generation.")
    print("TF import error:", repr(e))

IMG_DIM = 224
BATCH_SIZE = 16
CHANNELS = 3
NUM_CLASSES = 5  # labels are 0..4

CANDIDATE_INPUT_FOLDERS = [
    "/kaggle/input/aptos2019-blindness-detection/",
    "/kaggle/data/aptos2019-blindness-detection/",
    "../input/aptos2019-blindness-detection/",
]
INPUT_FOLDER = None
for p in CANDIDATE_INPUT_FOLDERS:
    if os.path.exists(p) and os.path.exists(os.path.join(p, "train.csv")):
        INPUT_FOLDER = p
        break
if INPUT_FOLDER is None:
    raise FileNotFoundError(
        "Could not locate aptos2019-blindness-detection folder in expected paths: "
        + ", ".join(CANDIDATE_INPUT_FOLDERS)
    )

MODEL_WEIGHTS = "../input/densenetmulti/ben_colour_-0.8746.h5"

print("INPUT_FOLDER:", INPUT_FOLDER)
print("CPU count:", psutil.cpu_count())
if TF_AVAILABLE:
    print("TF version:", tf.__version__)
print("Sample input folder listing:", os.listdir(INPUT_FOLDER)[:20])

CACHE_DIR = "/kaggle/working/preprocessed_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

CPU_WORKERS = max(1, min(8, (psutil.cpu_count() or 2)))




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

    if height < 100 or width < 100 or bottom <= top or right <= left:
        return img

    return img[top:bottom, left:right]


def bensYCC(bgr, weight=4, gamma=8):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)
    y = cv2.addWeighted(y, weight, cv2.GaussianBlur(y, (0, 0), gamma), -weight, 128)
    ycc_modified = cv2.merge((y, cr, cb))
    bens = cv2.cvtColor(ycc_modified, cv2.COLOR_YCrCb2BGR)
    return bens


def bensGray(gray, weight=4, gamma=8):
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


_GAMMA_LUT_CACHE = {}


def adjust_gamma(image_, gamma=1.0):
    key = float(np.round(gamma, 6))
    table = _GAMMA_LUT_CACHE.get(key)
    if table is None:
        invGamma = 1.0 / float(key)
        x = (np.arange(256, dtype=np.float32) / 255.0) ** invGamma
        table = np.clip(x * 255.0, 0, 255).astype(np.uint8)
        _GAMMA_LUT_CACHE[key] = table
    return cv2.LUT(image_, table)


def processBensColor(bgr):
    if bgr is None:
        return np.full((IMG_DIM, IMG_DIM, CHANNELS), 128, dtype=np.uint8)

    green = bgr[:, :, 1]  # use green as a greyscale
    cropped = crop(green, bgr, 0.02)

    squared = reflectAndSquareUp(cropped)
    resized = cv2.resize(squared, (IMG_DIM, IMG_DIM))
    circled = circleMask(resized)

    med = np.median(circled)
    med = max(med, 1.0)
    equalised = adjust_gamma(circled, 1 + np.log(90) - np.log(med))

    return cv2.cvtColor(bensYCC(equalised), cv2.COLOR_BGR2RGB)




## === cell 2
def dataGenerator(jitter=0.1):
    datagen = image.ImageDataGenerator(
        rescale=1.0 / 255.0,
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
def create_model():
    model = Sequential()
    base = DenseNet121(
        weights="imagenet", include_top=False, input_shape=(IMG_DIM, IMG_DIM, CHANNELS)
    )
    model.add(base)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation="sigmoid"))

    if MODEL_WEIGHTS and os.path.exists(MODEL_WEIGHTS):
        model.load_weights(MODEL_WEIGHTS)
        print("Loaded external weights:", MODEL_WEIGHTS)
    else:
        print("External weights not found; using ImageNet-initialized DenseNet121.")

    return model


model = None
if TF_AVAILABLE:
    model = create_model()
    model.compile(
        optimizer=Adam(learning_rate=0.00005),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
gc.collect()




## === cell 4
def _cache_path_for_set(d_set: str) -> str:
    return os.path.join(CACHE_DIR, f"{d_set}_ben224_uint8.npy")


def _load_or_build_preprocessed_block(d_set: str):
    images_dir = os.path.join(INPUT_FOLDER, f"{d_set}_images")
    df = pd.read_csv(os.path.join(INPUT_FOLDER, f"{d_set}.csv"))
    df.id_code = df.id_code.astype(str) + ".png"
    cache_path = _cache_path_for_set(d_set)

    if os.path.exists(cache_path):
        arr = np.load(cache_path, mmap_mode="r")
        if arr.shape[0] == len(df) and arr.shape[1:] == (IMG_DIM, IMG_DIM, CHANNELS):
            print(f"Loaded cached preprocessed {d_set} array:", cache_path, arr.shape)
            return df, arr
        else:
            print("Cache shape mismatch; rebuilding:", cache_path)

    filenames = df.id_code.values
    total = len(filenames)
    arr = np.empty((total, IMG_DIM, IMG_DIM, CHANNELS), dtype=np.uint8)

    join = os.path.join
    imread = cv2.imread
    proc = processBensColor

    print(
        f"Building preprocessed cache for {d_set}: {total} images with {CPU_WORKERS} workers"
    )

    def _work(i_fn):
        i, fn = i_fn
        bgr = imread(join(images_dir, fn))
        return i, proc(bgr)

    with ThreadPoolExecutor(max_workers=CPU_WORKERS) as ex:
        for k, (i, out) in enumerate(
            ex.map(_work, enumerate(filenames), chunksize=64), 1
        ):
            arr[i] = out
            if k % 512 == 0:
                print(f"  processed {k}/{total}")

    np.save(cache_path, arr)
    del arr
    gc.collect()
    arr = np.load(cache_path, mmap_mode="r")
    print(f"Saved and reloaded cache {d_set}:", cache_path, arr.shape)
    return df, arr


def _make_tta_dataset(arr_uint8, seed, jitter_amount, batch_size):
    rot_deg = int(800 * jitter_amount)
    bmin, bmax = (1.0 - jitter_amount), 1.0

    ds = tf.data.Dataset.from_tensor_slices(arr_uint8)

    def _augment(x):
        x = tf.cast(x, tf.float32) / 255.0  # rescale exactly as before

        if jitter_amount > 0.01:
            return x  # placeholder, replaced below

        return x

    if jitter_amount > 0.01:
        ds = ds.enumerate()

        def _aug_enum(i, x):
            s = tf.stack([tf.cast(seed, tf.int32), tf.cast(i, tf.int32)])
            x = tf.cast(x, tf.float32) / 255.0

            x = tf.image.stateless_random_flip_left_right(x, seed=s)
            x = tf.image.stateless_random_flip_up_down(x, seed=s + 1)

            br = tf.random.stateless_uniform([], seed=s + 2, minval=bmin, maxval=bmax)
            x = tf.clip_by_value(x * br, 0.0, 1.0)

            if rot_deg > 0:
                ang_deg = tf.random.stateless_uniform(
                    [], seed=s + 3, minval=-float(rot_deg), maxval=float(rot_deg)
                )
                ang = ang_deg * (math.pi / 180.0)
                x = tf.image.rotate(
                    x, ang, interpolation="BILINEAR", fill_mode="reflect"
                )

            zoom_lo, zoom_hi = (1.0 - jitter_amount), (1.0 + jitter_amount / 2.0)
            zx = tf.random.stateless_uniform(
                [], seed=s + 4, minval=zoom_lo, maxval=zoom_hi
            )
            zy = tf.random.stateless_uniform(
                [], seed=s + 5, minval=zoom_lo, maxval=zoom_hi
            )

            h = tf.shape(x)[0]
            w = tf.shape(x)[1]
            new_h = tf.cast(tf.cast(h, tf.float32) / zy, tf.int32)
            new_w = tf.cast(tf.cast(w, tf.float32) / zx, tf.int32)
            new_h = tf.clip_by_value(new_h, 1, h)
            new_w = tf.clip_by_value(new_w, 1, w)

            offset_h = (h - new_h) // 2
            offset_w = (w - new_w) // 2
            x = tf.image.crop_to_bounding_box(x, offset_h, offset_w, new_h, new_w)
            x = tf.image.resize(x, [IMG_DIM, IMG_DIM], method="bilinear")
            x = tf.clip_by_value(x, 0.0, 1.0)
            return x

        ds = ds.map(_aug_enum, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(
            lambda x: tf.cast(x, tf.float32) / 255.0,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds


def _predict_jitter_median(
    arr_uint8, jitters, jitter_amount=0.03, batch_size=BATCH_SIZE, base_seed=123
):
    n = int(arr_uint8.shape[0])
    pred_jitters = np.empty((n, jitters, NUM_CLASSES), dtype=np.float32)

    for j in range(jitters):
        ds = _make_tta_dataset(
            arr_uint8,
            seed=base_seed + j,
            jitter_amount=jitter_amount,
            batch_size=batch_size,
        )
        pred = model.predict(ds, verbose=0)
        pred_jitters[:, j, :] = pred.astype(np.float32, copy=False)

    return np.median(pred_jitters, axis=1)


def make_predictions(d_set, jitters=5):
    df, arr_uint8 = _load_or_build_preprocessed_block(d_set)
    total = len(df)
    print(f"Making predictions on the {d_set} dataset. Total: {total}")

    preds = _predict_jitter_median(
        arr_uint8,
        jitters=jitters,
        jitter_amount=0.03,
        batch_size=BATCH_SIZE,
        base_seed=123,
    )
    return preds


def quadratic_weighted_kappa(y_true, y_pred, num_classes=5):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    y_true = np.clip(y_true, 0, num_classes - 1)
    y_pred = np.clip(y_pred, 0, num_classes - 1)

    idx = y_true * num_classes + y_pred
    O = (
        np.bincount(idx, minlength=num_classes * num_classes)
        .reshape(num_classes, num_classes)
        .astype(np.float64)
    )

    act_hist = np.bincount(y_true, minlength=num_classes).astype(np.float64)
    pred_hist = np.bincount(y_pred, minlength=num_classes).astype(np.float64)
    E = np.outer(act_hist, pred_hist)
    if E.sum() > 0:
        E = E / E.sum() * O.sum()

    i = np.arange(num_classes, dtype=np.float64)
    W = ((i[:, None] - i[None, :]) ** 2) / ((num_classes - 1) ** 2)

    denom = (W * E).sum()
    if denom == 0:
        return 0.0
    return 1.0 - (W * O).sum() / denom


def preds_to_severity_score(preds):
    preds = np.asarray(preds, dtype=np.float32)
    class_ids = np.arange(preds.shape[1], dtype=np.float32)[None, :]
    return (preds * class_ids).sum(axis=1)


def apply_thresholds(scores, thresholds):
    t = np.asarray(thresholds, dtype=np.float32)
    t = np.sort(t)
    return np.sum(scores[:, None] > t[None, :], axis=1).astype(int)


def fit_thresholds_for_qwk(
    scores, y_true, init_thresholds=None, iters=2, grid_step=0.02
):
    scores = np.asarray(scores, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=np.int64)

    if init_thresholds is None:
        qs = [0.2, 0.4, 0.6, 0.8]
        init_thresholds = [float(np.quantile(scores, q)) for q in qs]
    best_t = np.array(init_thresholds, dtype=np.float32)

    smin = float(scores.min())
    smax = float(scores.max())

    best_k = quadratic_weighted_kappa(
        y_true, apply_thresholds(scores, best_t), num_classes=NUM_CLASSES
    )

    for _ in range(iters):
        for k in range(4):
            lo = smin if k == 0 else float(best_t[k - 1] + 1e-4)
            hi = smax if k == 3 else float(best_t[k + 1] - 1e-4)
            if hi <= lo:
                continue

            candidates = np.arange(lo, hi, grid_step, dtype=np.float32)
            local_best_tk = float(best_t[k])
            local_best_kappa = float(best_k)

            for cand in candidates:
                trial = best_t.copy()
                trial[k] = cand
                pred = apply_thresholds(scores, trial)
                kappa = quadratic_weighted_kappa(y_true, pred, num_classes=NUM_CLASSES)
                if kappa > local_best_kappa:
                    local_best_kappa = float(kappa)
                    local_best_tk = float(cand)

            best_t[k] = local_best_tk
            best_k = local_best_kappa

    return best_t, float(best_k)


test_df = pd.read_csv(os.path.join(INPUT_FOLDER, "test.csv"))

if TF_AVAILABLE and model is not None:
    thresholds = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)

    test_predictions = make_predictions("test", 5)
    test_scores = preds_to_severity_score(test_predictions)
    test_classes = apply_thresholds(test_scores, thresholds)

    print("Using fixed thresholds:", thresholds.tolist())
    print("Pred head:\n", test_predictions[:5])
    print("Score head:\n", test_scores[:5])
    print("Class head:\n", test_classes[:5])

    test_df["diagnosis"] = np.clip(test_classes.astype(int), 0, 4)
else:
    print("WARNING: Using fallback predictions (all zeros) due to missing TensorFlow.")
    test_df["diagnosis"] = 0

submission = test_df[["id_code", "diagnosis"]]
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2086401258.py in <cell line: 0>()
    263     thresholds = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)
    264 
--> 265     test_predictions = make_predictions("test", 5)
    266     test_scores = preds_to_severity_score(test_predictions)
    267     test_classes = apply_thresholds(test_scores, thresholds)

/tmp/ipykernel_11/2086401258.py in make_predictions(d_set, jitters)
    160     print(f"Making predictions on the {d_set} dataset. Total: {total}")
    161 
--> 162     preds = _predict_jitter_median(
    163         arr_uint8,
    164         jitters=jitters,

/tmp/ipykernel_11/2086401258.py in _predict_jitter_median(arr_uint8, jitters, jitter_amount, batch_size, base_seed)
    143 
    144     for j in range(jitters):
--> 145         ds = _make_tta_dataset(
    146             arr_uint8,
    147             seed=base_seed + j,

/tmp/ipykernel_11/2086401258.py in _make_tta_dataset(arr_uint8, seed, jitter_amount, batch_size)
    125             return x
    126 
--> 127         ds = ds.map(_aug_enum, num_parallel_calls=tf.data.AUTOTUNE)
    128     else:
    129         ds = ds.map(

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in map(self, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)
   2339     from tensorflow.python.data.ops import map_op
   2340 
-> 2341     return map_op._map_v2(
   2342         self,
   2343         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py in _map_v2(input_dataset, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)
     55           num_parallel_calls,
     56       )
---> 57     return _ParallelMapDataset(
     58         input_dataset,
     59         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py in __init__(self, input_dataset, map_func, num_parallel_calls, deterministic, use_inter_op_parallelism, preserve_cardinality, use_legacy_function, use_unbounded_threadpool, name)
    200     self._input_dataset = input_dataset
    201     self._use_inter_op_parallelism = use_inter_op_parallelism
--> 202     self._map_func = structured_function.StructuredFunctionWrapper(
    203         map_func,
    204         self._transformation_name(),

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in __init__(self, func, transformation_name, dataset, input_classes, input_shapes, input_types, input_structure, add_to_graph, use_legacy_function, defun_kwargs)
    263         fn_factory = trace_tf_function(defun_kwargs)
    264 
--> 265     self._function = fn_factory()
    266     # There is no graph to add in eager mode.
    267     add_to_graph &= not context.executing_eagerly()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in get_concrete_function(self, *args, **kwargs)
   1249   def get_concrete_function(self, *args, **kwargs):
   1250     # Implements PolymorphicFunction.get_concrete_function.
-> 1251     concrete = self._get_concrete_function_garbage_collected(*args, **kwargs)
   1252     concrete._garbage_collector.release()  # pylint: disable=protected-access
   1253     return concrete

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in _get_concrete_function_garbage_collected(self, *args, **kwargs)
   1219       if self._variable_creation_config is None:
   1220         initializers = []
-> 1221         self._initialize(args, kwargs, add_initializers_to=initializers)
   1222         self._initialize_uninitialized_variables(initializers)
   1223 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in _initialize(self, args, kwds, add_initializers_to)
    694     )
    695     # Force the definition of the function for these arguments
--> 696     self._concrete_variable_creation_fn = tracing_compilation.trace_function(
    697         args, kwds, self._variable_creation_config
    698     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in trace_function(args, kwargs, tracing_options)
    176       kwargs = {}
    177 
--> 178     concrete_function = _maybe_define_function(
    179         args, kwargs, tracing_options
    180     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in _maybe_define_function(args, kwargs, tracing_options)
    281         else:
    282           target_func_type = lookup_func_type
--> 283         concrete_function = _create_concrete_function(
    284             target_func_type, lookup_func_context, func_graph, tracing_options
    285         )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in _create_concrete_function(function_type, type_context, func_graph, tracing_options)
    308       attributes_lib.DISABLE_ACD, False
    309   )
--> 310   traced_func_graph = func_graph_module.func_graph_from_py_func(
    311       tracing_options.name,
    312       tracing_options.python_function,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/func_graph.py in func_graph_from_py_func(name, python_func, args, kwargs, signature, func_graph, add_control_dependencies, arg_names, op_return_value, collections, capture_by_value, create_placeholders)
   1057 
   1058     _, original_func = tf_decorator.unwrap(python_func)
-> 1059     func_outputs = python_func(*func_args, **func_kwargs)
   1060 
   1061     # invariant: `func_outputs` contains only Tensors, CompositeTensors,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in wrapped_fn(*args, **kwds)
    597         # the function a weak reference to itself to avoid a reference cycle.
    598         with OptionalXlaContext(compile_with_xla):
--> 599           out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    600         return out
    601 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in wrapped_fn(*args)
    229       # Note: wrapper_helper will apply autograph based on context.
    230       def wrapped_fn(*args):  # pylint: disable=missing-docstring
--> 231         ret = wrapper_helper(*args)
    232         ret = structure.to_tensor_list(self._output_structure, ret)
    233         return [ops.convert_to_tensor(t) for t in ret]

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in wrapper_helper(*args)
    159       if not _should_unpack(nested_args):
    160         nested_args = (nested_args,)
--> 161       ret = autograph.tf_convert(self._func, ag_ctx)(*nested_args)
    162       ret = variable_utils.convert_variables_to_tensors(ret)
    163       if _should_pack(ret):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in wrapper(*args, **kwargs)
    691       except Exception as e:  # pylint:disable=broad-except
    692         if hasattr(e, 'ag_error_metadata'):
--> 693           raise e.ag_error_metadata.to_exception(e)
    694         else:
    695           raise

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in wrapper(*args, **kwargs)
    688       try:
    689         with conversion_ctx:
--> 690           return converted_call(f, args, kwargs, options=options)
    691       except Exception as e:  # pylint:disable=broad-except
    692         if hasattr(e, 'ag_error_metadata'):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    437     try:
    438       if kwargs is not None:
--> 439         result = converted_f(*effective_args, **kwargs)
    440       else:
    441         result = converted_f(*effective_args)

/tmp/__autograph_generated_file5etx5_lv.py in tf___aug_enum(i, x)
     38                 ang_deg = ag__.Undefined('ang_deg')
     39                 ang = ag__.Undefined('ang')
---> 40                 ag__.if_stmt(ag__.ld(rot_deg) > 0, if_body, else_body, get_state, set_state, ('x',), 1)
     41                 zoom_lo, zoom_hi = (1.0 - ag__.ld(jitter_amount), 1.0 + ag__.ld(jitter_amount) / 2.0)
     42                 zx = ag__.converted_call(ag__.ld(tf).random.stateless_uniform, ([],), dict(seed=ag__.ld(s) + 4, minval=ag__.ld(zoom_lo), maxval=ag__.ld(zoom_hi)), fscope)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/operators/control_flow.py in if_stmt(cond, body, orelse, get_state, set_state, symbol_names, nouts)
   1215     _tf_if_stmt(cond, body, orelse, get_state, set_state, symbol_names, nouts)
   1216   else:
-> 1217     _py_if_stmt(cond, body, orelse)
   1218 
   1219 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/operators/control_flow.py in _py_if_stmt(cond, body, orelse)
   1268 def _py_if_stmt(cond, body, orelse):
   1269   """Overload of if_stmt that executes a Python if statement."""
-> 1270   return body() if cond else orelse()

/tmp/__autograph_generated_file5etx5_lv.py in if_body()
     31                     ang_deg = ag__.converted_call(ag__.ld(tf).random.stateless_uniform, ([],), dict(seed=ag__.ld(s) + 3, minval=-ag__.converted_call(ag__.ld(float), (ag__.ld(rot_deg),), None, fscope), maxval=ag__.converted_call(ag__.ld(float), (ag__.ld(rot_deg),), None, fscope)), fscope)
     32                     ang = ag__.ld(ang_deg) * (ag__.ld(math).pi / 180.0)
---> 33                     x = ag__.converted_call(ag__.ld(tf).image.rotate, (ag__.ld(x), ag__.ld(ang)), dict(interpolation='BILINEAR', fill_mode='reflect'), fscope)
     34 
     35                 def else_body():

AttributeError: in user code:

    File "/tmp/ipykernel_11/2086401258.py", line 96, in _aug_enum  *
        x = tf.image.rotate(

    AttributeError: module 'tensorflow._api.v2.image' has no attribute 'rotate'
