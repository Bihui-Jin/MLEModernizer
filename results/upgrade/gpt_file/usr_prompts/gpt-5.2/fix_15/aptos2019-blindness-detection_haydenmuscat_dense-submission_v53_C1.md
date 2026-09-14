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
os.environ.setdefault("PYTHONHASHSEED", "42")

import gc
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split

import tensorflow as tf  # type: ignore

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.optimizers import Adam

np.random.seed(42)
tf.random.set_seed(42)
try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

IMG_DIM = 256
BATCH_SIZE = 32
CHANNELS = 3
NUM_CLASSES = 5

CANDIDATE_INPUT_FOLDERS = [
    "/kaggle/input/aptos2019-blindness-detection/",
    "../input/aptos2019-blindness-detection/",
    "/kaggle/data/aptos2019-blindness-detection/",
    "../data/aptos2019-blindness-detection/",
    "/kaggle/data/aptos2019-blindness-detection/aptos2019-blindness-detection/",
    "/kaggle/input/aptos2019-blindness-detection/aptos2019-blindness-detection/",
]

INPUT_FOLDER = next((p for p in CANDIDATE_INPUT_FOLDERS if os.path.exists(p)), None)
if INPUT_FOLDER is None:
    for base in ["/kaggle/input", "/kaggle/data", "../input", "../data"]:
        if os.path.exists(base):
            for root, dirs, files in os.walk(base):
                if "train.csv" in files and "test.csv" in files:
                    INPUT_FOLDER = root + ("" if root.endswith(os.sep) else os.sep)
                    break
            if INPUT_FOLDER is not None:
                break

if INPUT_FOLDER is None:
    raise FileNotFoundError(
        "Could not locate aptos2019-blindness-detection input folder. "
        "Tried: " + ", ".join(CANDIDATE_INPUT_FOLDERS)
    )

print("CWD:", os.getcwd())
print("INPUT_FOLDER:", INPUT_FOLDER)
print("train.csv exists:", os.path.exists(os.path.join(INPUT_FOLDER, "train.csv")))
print("test.csv exists:", os.path.exists(os.path.join(INPUT_FOLDER, "test.csv")))
print("TF version:", tf.__version__)
print("Eager execution:", tf.executing_eagerly())




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

    if height < 100 or width < 100:
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


def clahe_gray(gray, clipLimit=3.5, grid=4):

    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(grid, grid))
    return clahe.apply(gray)


_GAMMA_LUT_CACHE = {}


def adjust_gamma(image, gamma=1.0):
    g = float(gamma)
    key = round(g, 8)
    table = _GAMMA_LUT_CACHE.get(key)
    if table is None:
        invGamma = 1.0 / g
        table = np.array(
            [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")
        _GAMMA_LUT_CACHE[key] = table
    return cv2.LUT(image, table)


_LOG90 = float(np.log(90.0))


def processBenNormal(bgr):

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

    med = float(np.median(resized[:, :, 1]))
    if med <= 0.0:
        gamma = 1.0 + _LOG90 - np.log(1.0)
    else:
        gamma = 1.0 + _LOG90 - np.log(med)

    equalised = adjust_gamma(resized, gamma)
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
def test_datagen_plot(processing_function, jitter=0.3):

    figure = plt.figure(figsize=(8, 8))

    images_dir = os.path.join(INPUT_FOLDER, "test_images")
    df = pd.read_csv(os.path.join(INPUT_FOLDER, "test.csv"))
    df.id_code = df.id_code.apply(lambda x: x + ".png")

    img_block = np.empty((100, IMG_DIM, IMG_DIM, CHANNELS))
    for i, filename in enumerate(df[:100].id_code):
        try:
            bgr = cv2.imread(os.path.join(images_dir, filename))
            if bgr is None:
                raise ValueError("cv2.imread returned None")
            img_block[i, :, :, :] = processing_function(bgr)
        except Exception:
            img_block[i, :, :, :] = 128.0

    datagen_sample = dataGenerator(jitter).flow(img_block)
    for x in datagen_sample:
        for j in range(16):
            ax = figure.add_subplot(4, 4, j + 1)
            ax.imshow(x[j])
            ax.axis("off")
        break
    plt.show()




## === cell 4
def create_model(weights):
    model = Sequential()
    model.add(
        DenseNet121(
            weights="imagenet",
            include_top=False,
            input_shape=(IMG_DIM, IMG_DIM, CHANNELS),
        )
    )
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation="sigmoid"))

    if isinstance(weights, str) and os.path.exists(weights):
        model.load_weights(weights)

    model.compile(
        optimizer=Adam(learning_rate=0.00005),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model




## === cell 5
def _resolve_images_dir(d_set: str) -> str:
    primary = os.path.join(INPUT_FOLDER, f"{d_set}_images")
    if os.path.exists(primary):
        return primary
    candidates = [
        os.path.join(INPUT_FOLDER, "aptos2019-blindness-detection", f"{d_set}_images"),
        os.path.join(os.path.dirname(INPUT_FOLDER.rstrip("/")), f"{d_set}_images"),
        os.path.join("/kaggle/input/aptos2019-blindness-detection", f"{d_set}_images"),
        os.path.join("/kaggle/data/aptos2019-blindness-detection", f"{d_set}_images"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return primary  # will fail later with clear prints if truly missing


def _configure_datagen_for_jitter(datagen, jitter: float):
    datagen.horizontal_flip = True and (jitter > 0.01)
    datagen.vertical_flip = True and (jitter > 0.01)
    datagen.zoom_range = [max(0.8, 1 - 5 * jitter), 1]
    datagen.rotation_range = int(600 * jitter)
    datagen.brightness_range = [1 - jitter / 3, 1 + jitter / 3]
    datagen.fill_mode = "mirror"
    datagen.channel_shift_range = int(30 * jitter)
    return datagen


_PREPROCESS_CACHE = {}
_PREPROCESS_CACHE_MAX = 5000


def _preprocess_path_cached(path, processing_function):
    out = _PREPROCESS_CACHE.get(path)
    if out is not None:
        return out
    bgr = cv2.imread(path)
    if bgr is None:
        out = None
    else:
        out = processing_function(bgr)
    if len(_PREPROCESS_CACHE) >= _PREPROCESS_CACHE_MAX:
        _PREPROCESS_CACHE.clear()
    _PREPROCESS_CACHE[path] = out
    return out


def make_predictions(d_set, processing_function, model, jitters=5):

    images_dir = _resolve_images_dir(d_set)
    df = pd.read_csv(os.path.join(INPUT_FOLDER, f"{d_set}.csv"))
    df.id_code = df.id_code.apply(lambda x: x + ".png")

    block_size = 512
    total = df.index.size
    predictions = np.zeros((df.index.size, NUM_CLASSES), dtype=np.float32)

    print(f"Making predictions on the {d_set} dataset. Total: {total}")
    print("Images dir:", images_dir, "exists:", os.path.exists(images_dir))

    if jitters >= 1:
        effective_single_pass = True
    else:
        effective_single_pass = False

    for start in range(0, total, block_size):

        end = start + block_size
        if end > total:
            end = total

        img_block = np.empty(
            (end - start, IMG_DIM, IMG_DIM, CHANNELS), dtype=np.float32
        )
        for i, filename in enumerate(df[start:end].id_code):
            p = os.path.join(images_dir, filename)
            try:
                out = _preprocess_path_cached(p, processing_function)
                if out is None:
                    raise ValueError("preprocess returned None")
                img_block[i, :, :, :] = out
            except Exception:
                print("Error opening or manipulating image:", filename)
                img_block[i, :, :, :] = 128.0

        if effective_single_pass:
            x = img_block * (1.0 / 255.0)
            preds_block = []
            for bs in range(0, len(x), BATCH_SIZE):
                preds_block.append(model.predict_on_batch(x[bs : bs + BATCH_SIZE]))
            predictions[start:end] = np.concatenate(preds_block, axis=0)[
                : (end - start)
            ]
        else:
            shared_datagen = image.ImageDataGenerator(rescale=1.0 / 255)
            prediction_jitters = np.empty(
                (len(img_block), jitters, NUM_CLASSES), dtype=np.float32
            )
            jit = 0.0
            for i in range(jitters):
                _configure_datagen_for_jitter(shared_datagen, jit)
                datagen_flow = shared_datagen.flow(
                    img_block, shuffle=False, batch_size=BATCH_SIZE
                )
                prediction_jitters[:, i] = model.predict(
                    datagen_flow,
                    steps=len(datagen_flow),
                    verbose=0,
                )
                jit += 0.02
            predictions[start:end] = np.median(prediction_jitters, axis=1)

        print(f"{start} - {end} finished")
        gc.collect()

    return predictions




## === cell 6
def prediction_convert_sum(predictions, thresholds):
    thresholds = np.asarray(thresholds, dtype=predictions.dtype)
    thresholded = predictions > thresholds
    y_val = thresholded.astype(np.int32).sum(axis=1) - 1
    return y_val


def prediction_convert_highest(predictions, thresholds):
    thresholds = np.asarray(thresholds, dtype=predictions.dtype)
    thr = predictions > thresholds  # (N,5) bool
    any_true = thr.any(axis=1)
    rev_idx = np.argmax(thr[:, ::-1], axis=1)
    y_val = (NUM_CLASSES - 1 - rev_idx).astype(np.int32)
    y_val[~any_true] = 0
    return y_val


def find_best_thresholds(train_predictions):

    print("Finding best thresholds...")

    prediction_convert = prediction_convert_sum

    gc.collect()

    train_df = pd.read_csv(os.path.join(INPUT_FOLDER, "train.csv"))
    y_actual = train_df.diagnosis.astype(int).values

    thresholds = [0.5 for i in range(NUM_CLASSES)]
    d_thresh = 0.25

    for sweep in range(5):

        for label in range(5):

            currKappa = cohen_kappa_score(
                y_actual,
                prediction_convert(train_predictions, thresholds),
                weights="quadratic",
            )

            print(currKappa)

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
from concurrent.futures import ThreadPoolExecutor


def _load_and_preprocess_batch(
    image_paths, processing_function, max_workers=None, chunk=256
):
    x = np.empty((len(image_paths), IMG_DIM, IMG_DIM, CHANNELS), dtype=np.float32)

    def _load_one(p):
        try:
            out = _preprocess_path_cached(p, processing_function)
            return out
        except Exception:
            return None

    if max_workers is None:
        max_workers = min(8, (os.cpu_count() or 2))

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        write_pos = 0
        n = len(image_paths)
        while write_pos < n:
            j = min(n, write_pos + chunk)
            futures = [ex.submit(_load_one, p) for p in image_paths[write_pos:j]]
            for i, fut in enumerate(futures):
                out = fut.result()
                if out is None:
                    x[write_pos + i] = 128.0
                else:
                    x[write_pos + i] = out
            write_pos = j
    return x


def _onehot_levels(y_int):
    y_int = np.asarray(y_int, dtype=np.int32)
    cls = np.arange(NUM_CLASSES, dtype=np.int32)[None, :]
    y = (cls <= y_int[:, None]).astype(np.float32)
    y[y_int < 0] = 0.0
    return y


def train_model_minimal(
    model, processing_function, epochs=2, jitter=0.1, val_size=0.15
):
    train_df = pd.read_csv(os.path.join(INPUT_FOLDER, "train.csv"))
    images_dir = _resolve_images_dir("train")
    train_paths = (
        train_df["id_code"].apply(lambda x: os.path.join(images_dir, f"{x}.png")).values
    )
    y_int = train_df["diagnosis"].astype(int).values

    idx = np.arange(len(train_df))
    tr_idx, va_idx = train_test_split(
        idx, test_size=val_size, random_state=42, stratify=y_int
    )

    tr_paths, va_paths = train_paths[tr_idx], train_paths[va_idx]
    y_tr_int, y_va_int = y_int[tr_idx], y_int[va_idx]

    y_tr = _onehot_levels(y_tr_int)
    y_va = _onehot_levels(y_va_int)

    print("Preprocessing train split into memory:", len(tr_paths))
    x_tr = _load_and_preprocess_batch(tr_paths, processing_function)
    print("Preprocessing val split into memory:", len(va_paths))
    x_va = _load_and_preprocess_batch(va_paths, processing_function)

    train_gen = dataGenerator(jitter).flow(
        x_tr, y_tr, batch_size=BATCH_SIZE, shuffle=True, seed=42
    )
    val_gen = image.ImageDataGenerator(rescale=1.0 / 255).flow(
        x_va, y_va, batch_size=BATCH_SIZE, shuffle=False
    )

    steps_per_epoch = int(np.ceil(len(x_tr) / BATCH_SIZE))
    val_steps = int(np.ceil(len(x_va) / BATCH_SIZE))

    print(
        f"Training for epochs={epochs}, steps_per_epoch={steps_per_epoch}, val_steps={val_steps}"
    )
    model.fit(
        train_gen,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=val_steps,
        verbose=1,
    )

    val_pred = model.predict(val_gen, steps=val_steps, verbose=0)[: len(x_va)]
    y_va_pred_class = prediction_convert_highest(val_pred, [0.5] * NUM_CLASSES)
    kappa = cohen_kappa_score(y_va_int, y_va_pred_class, weights="quadratic")
    print("Validation kappa (sanity check, not used for early stopping):", kappa)

    return (
        model,
        (x_va, y_va_int, val_pred),
        (train_paths, y_int, tr_idx, va_idx, x_tr, x_va),
    )




## === cell 8
NORMAL_WEIGHTS = None

model = create_model(NORMAL_WEIGHTS)

(
    model,
    (x_val_mem, y_val_int, val_pred),
    (
        train_paths_all,
        y_int_all,
        tr_idx,
        va_idx,
        x_tr_mem,
        x_va_mem,
    ),
) = train_model_minimal(model, processBenNormal, epochs=2, jitter=0.1, val_size=0.15)

train_df_full = pd.read_csv(os.path.join(INPUT_FOLDER, "train.csv"))
n_total = len(train_df_full)

x_train_full = np.empty((n_total, IMG_DIM, IMG_DIM, CHANNELS), dtype=np.float32)
x_train_full[tr_idx] = x_tr_mem
x_train_full[va_idx] = x_va_mem

mask_missing = np.ones(n_total, dtype=bool)
mask_missing[tr_idx] = False
mask_missing[va_idx] = False
missing_idx = np.where(mask_missing)[0]

if len(missing_idx) > 0:
    print(
        "Preprocessing remaining train images for threshold search:", len(missing_idx)
    )
    x_missing = _load_and_preprocess_batch(
        train_paths_all[missing_idx], processBenNormal
    )
    x_train_full[missing_idx] = x_missing
else:
    print(
        "All train images already preprocessed from splits; no extra preprocessing needed."
    )

train_full_gen = image.ImageDataGenerator(rescale=1.0 / 255).flow(
    x_train_full, batch_size=BATCH_SIZE, shuffle=False
)
train_predictions_full = model.predict(
    train_full_gen, steps=len(train_full_gen), verbose=0
)[: len(x_train_full)]

thresholds = find_best_thresholds(train_predictions_full)
print("Using thresholds:", thresholds)

preds = make_predictions("test", processBenNormal, model, jitters=5)

if not np.isfinite(preds).all():
    raise ValueError("Non-finite values found in predictions (NaN/Inf).")

test_df = pd.read_csv(os.path.join(INPUT_FOLDER, "test.csv"))
if preds.shape[0] != len(test_df):
    raise ValueError(
        f"Prediction rows ({preds.shape[0]}) do not match test rows ({len(test_df)})."
    )
if preds.shape[1] != NUM_CLASSES:
    raise ValueError(
        f"Prediction cols ({preds.shape[1]}) != NUM_CLASSES ({NUM_CLASSES})."
    )

test_classes = prediction_convert_highest(preds, thresholds)

print("First 10 predicted classes:", test_classes[:10])

test_df["diagnosis"] = test_classes.astype(int)
test_df["diagnosis"] = test_df["diagnosis"].clip(0, 4).astype(int)

out_path = "submission.csv"
test_df[["id_code", "diagnosis"]].to_csv(out_path, index=False)
print("Wrote", out_path, "with shape:", test_df[["id_code", "diagnosis"]].shape)
print(test_df.head())
