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
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import sys
import subprocess


def _ensure_compatible_protobuf():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver

        major = int(str(pb_ver).split(".")[0])
        if major >= 4:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf==3.20.3"]
            )
            for k in list(sys.modules.keys()):
                if k.startswith("google.protobuf"):
                    del sys.modules[k]
    except Exception:
        pass


_ensure_compatible_protobuf()

import gc
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from sklearn.metrics import cohen_kappa_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.densenet import (
    preprocess_input as densenet_preprocess,
)

IMG_DIM = 256
BATCH_SIZE = 32
CHANNELS = 3
NUM_CLASSES = 5


def _pick_input_folder():
    candidates = [
        "../input/aptos2019-blindness-detection/",
        "/kaggle/input/aptos2019-blindness-detection/",
        "/kaggle/data/aptos2019-blindness-detection/",
        "../input/",
        "/kaggle/input/",
        "/kaggle/data/",
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "train.csv")) and os.path.exists(
            os.path.join(c, "test.csv")
        ):
            return c

    for c in [
        "../input/aptos2019-blindness-detection/aptos2019-blindness-detection/",
        "/kaggle/input/aptos2019-blindness-detection/aptos2019-blindness-detection/",
        "/kaggle/data/aptos2019-blindness-detection/aptos2019-blindness-detection/",
    ]:
        if os.path.exists(os.path.join(c, "train.csv")) and os.path.exists(
            os.path.join(c, "test.csv")
        ):
            return c

    raise FileNotFoundError(
        "Could not locate train.csv/test.csv under expected Kaggle input/data paths."
    )


INPUT_FOLDER = _pick_input_folder()

print("Using INPUT_FOLDER:", INPUT_FOLDER)
print("train.csv exists:", os.path.exists(os.path.join(INPUT_FOLDER, "train.csv")))
print("test.csv exists:", os.path.exists(os.path.join(INPUT_FOLDER, "test.csv")))
print(
    "train_images exists:", os.path.exists(os.path.join(INPUT_FOLDER, "train_images"))
)
print("test_images exists:", os.path.exists(os.path.join(INPUT_FOLDER, "test_images")))

SEED = 1337
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    import multiprocessing as _mp

    _CPU = _mp.cpu_count()
except Exception:
    _CPU = 4
_TF_THREADS = max(1, min(4, _CPU))
try:
    tf.config.threading.set_intra_op_parallelism_threads(_TF_THREADS)
    tf.config.threading.set_inter_op_parallelism_threads(_TF_THREADS)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

from concurrent.futures import ThreadPoolExecutor


def _default_worker_count():
    try:
        import multiprocessing as mp

        c = mp.cpu_count()
    except Exception:
        c = 4
    return max(2, min(8, c))


IMG_WORKERS = _default_worker_count()
print("IMG_WORKERS:", IMG_WORKERS)

PRED_CACHE_DIR = "./_pred_cache"
os.makedirs(PRED_CACHE_DIR, exist_ok=True)

_IMG_CACHE = {}
_IMG_CACHE_ORDER = []
_IMG_CACHE_MAX = 1024  # bounded; images are 256x256x3 uint8/float32, size managed.


def _img_cache_get(key):
    return _IMG_CACHE.get(key)


def _img_cache_put(key, value):
    if key in _IMG_CACHE:
        return
    _IMG_CACHE[key] = value
    _IMG_CACHE_ORDER.append(key)
    if len(_IMG_CACHE_ORDER) > _IMG_CACHE_MAX:
        old = _IMG_CACHE_ORDER.pop(0)
        _IMG_CACHE.pop(old, None)




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


_GAMMA_LUT_CACHE = {}


def adjust_gamma(image_, gamma=1.0):
    lut = _GAMMA_LUT_CACHE.get(gamma)
    if lut is None:
        invGamma = 1.0 / gamma
        lut = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype(
            "uint8"
        )
        _GAMMA_LUT_CACHE[gamma] = lut
    return cv2.LUT(image_, lut)


def bensYCC(bgr, weight=4, gamma=15):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)
    y = cv2.addWeighted(y, weight, cv2.GaussianBlur(y, (0, 0), gamma), -weight, 128)
    ycc_modified = cv2.merge((y, cr, cb))
    bens = cv2.cvtColor(ycc_modified, cv2.COLOR_YCrCb2BGR)
    return bens


_CLAHE_CACHE = {}


def claheYCC(bgr, clipLimit=5, grid=8):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)

    key = (clipLimit, grid)
    clahe = _CLAHE_CACHE.get(key)
    if clahe is None:
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(grid, grid))
        _CLAHE_CACHE[key] = clahe

    y = clahe.apply(y)
    med = np.median(y)
    med = med if med > 0 else 1.0
    y = adjust_gamma(y, 1 + np.log(110) - np.log(med))

    ycc_modified = cv2.merge((y, cr, cb))
    img = cv2.cvtColor(ycc_modified, cv2.COLOR_YCrCb2BGR)
    return img


def bensSimple(bgr, weight=4, gamma=15):
    img = cv2.addWeighted(
        bgr, weight, cv2.GaussianBlur(bgr, (0, 0), gamma), -weight, 128
    )
    return img


def process(bgr, model):
    if bgr is None:
        return np.zeros((IMG_DIM, IMG_DIM, 3), dtype=np.uint8)

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

    if model == "normal":
        colouring_fn = bensYCC
    elif model == "weird":
        colouring_fn = bensSimple
    elif model == "clahe":
        colouring_fn = claheYCC
    else:
        colouring_fn = bensYCC  # safe default

    resized = cv2.resize(test_crop, (IMG_DIM, IMG_DIM), interpolation=cv2.INTER_AREA)
    img_ = colouring_fn(resized)

    return cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)




## === cell 2
_DATAGEN_CACHE = {}


def dataGenerator(jitter=0.1):
    jitter = float(jitter)
    datagen = _DATAGEN_CACHE.get(jitter)
    if datagen is None:
        datagen = image.ImageDataGenerator(
            preprocessing_function=densenet_preprocess,
            horizontal_flip=True and (jitter > 0.01),
            vertical_flip=True and (jitter > 0.01),
            zoom_range=[max(0.8, 1 - 5 * jitter), 1],
            rotation_range=int(600 * jitter),
            brightness_range=[1 - jitter / 3, 1 + jitter / 3],
            fill_mode="mirror",
            channel_shift_range=int(30 * jitter),
        )
        _DATAGEN_CACHE[jitter] = datagen
    return datagen




## === cell 3
RUN_PLOTS = False


def test_datagen_plot(processing_function, jitter=0.03):
    images_dir = f"{INPUT_FOLDER}test_images/"
    df = pd.read_csv(f"{INPUT_FOLDER}test.csv")
    df["id_code"] = df["id_code"].apply(lambda x: x + ".png")

    img_block = np.empty((100, IMG_DIM, IMG_DIM, CHANNELS), dtype=np.float32)
    j = 1
    for i, filename in enumerate(df[:100].id_code):
        bgr = cv2.imread(os.path.join(images_dir, filename))
        img_block[i, :, :, :] = process(bgr, processing_function)
        if bgr is not None:
            if bgr.shape != (480, 640, 3) and j <= 8:
                ax = figure.add_subplot(4, 4, j)
                plt.imshow(img_block[i, :, :, :] / 255.0)
                j += 1
            elif bgr.shape == (480, 640, 3) and j > 8:
                ax = figure.add_subplot(4, 4, j)
                plt.imshow(img_block[i, :, :, :] / 255.0)
                j += 1
                if j > 16:
                    return

    datagen_sample = dataGenerator(jitter).flow(img_block)
    for x in datagen_sample:
        for j in range(16):
            ax = figure.add_subplot(4, 4, j + 1)
            vis = x[j].copy()
            vis = (vis - vis.min()) / (vis.max() - vis.min() + 1e-8)
            plt.imshow(vis)
        break


if RUN_PLOTS:
    figure = plt.figure(figsize=(22, 20))
    test_datagen_plot("clahe")
    gc.collect()




## === cell 4
def _find_weights_file():
    candidates = [
        "../input/densenetmulti/dense-0.800.h5",
        "/kaggle/input/densenetmulti/dense-0.800.h5",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    search_roots = ["../input", "/kaggle/input", "/kaggle/data"]
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith(".h5") and (
                    "dense" in fn.lower() or "densenet" in fn.lower()
                ):
                    return os.path.join(dirpath, fn)
    return None


WEIGHTS_FILE = _find_weights_file()
print("WEIGHTS_FILE:", WEIGHTS_FILE)




## === cell 5
_MODEL_CACHE = {}


def load_network(network_name):
    cached = _MODEL_CACHE.get(network_name)
    if cached is not None:
        return cached

    model = Sequential()
    if WEIGHTS_FILE is None:
        backbone = DenseNet121(
            weights="imagenet",
            include_top=False,
            input_shape=(IMG_DIM, IMG_DIM, CHANNELS),
        )
    else:
        backbone = DenseNet121(
            weights=None, include_top=False, input_shape=(IMG_DIM, IMG_DIM, CHANNELS)
        )

    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation="sigmoid"))

    if WEIGHTS_FILE is not None:
        model.load_weights(WEIGHTS_FILE)

    model.compile(
        optimizer=Adam(learning_rate=0.00005),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    _MODEL_CACHE[network_name] = model
    return model




## === cell 6
def prediction_convert_highest(predictions, thresholds):
    thr = np.asarray(thresholds, dtype=predictions.dtype)[None, :]
    mask = predictions > thr  # (N,5) bool
    rev = mask[:, ::-1]
    any_true = rev.any(axis=1)
    idx_from_end = rev.argmax(axis=1)
    y = (NUM_CLASSES - 1 - idx_from_end).astype(np.int64)
    y[~any_true] = 0
    return y


def _resolve_images_dir(d_set):
    candidates = [
        os.path.join(INPUT_FOLDER, f"{d_set}_images"),
        os.path.join("/kaggle/data/aptos2019-blindness-detection", f"{d_set}_images"),
        os.path.join("/kaggle/input/aptos2019-blindness-detection", f"{d_set}_images"),
        os.path.join("/kaggle/data", f"{d_set}_images"),
        os.path.join("/kaggle/input", f"{d_set}_images"),
    ]
    for c in candidates:
        if os.path.isdir(c) and len(os.listdir(c)) > 0:
            return c
    return candidates[0]


def _predict_no_aug_tfdata(neural_net, img_block, batch_size=BATCH_SIZE):
    x = tf.convert_to_tensor(img_block, dtype=tf.float32)
    ds = tf.data.Dataset.from_tensor_slices(x)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.map(lambda b: densenet_preprocess(b), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return neural_net.predict(ds, verbose=0)


class _TTASequence(keras.utils.Sequence):
    def __init__(self, datagen, x, seed, batch_size):
        self.datagen = datagen
        self.x = x
        self.seed = int(seed) & 0x7FFFFFFF
        self.batch_size = int(batch_size)
        self.n = int(x.shape[0])

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(start + self.batch_size, self.n)
        batch = self.x[start:end]
        out = np.empty_like(batch, dtype=np.float32)

        for j in range(end - start):
            abs_i = start + j
            rs = np.random.RandomState((self.seed + abs_i * 1000003) & 0x7FFFFFFF)
            out[j] = self.datagen.random_transform(
                batch[j], seed=rs.randint(0, 2**31 - 1)
            )
            out[j] = self.datagen.standardize(out[j])
        return out


def _predict_with_tta_sequence(
    neural_net, datagen, img_block, seed, batch_size=BATCH_SIZE
):
    seq = _TTASequence(datagen, img_block, seed=seed, batch_size=batch_size)
    return neural_net.predict(seq, verbose=0)


def _load_and_process_block(filepaths, model_name):
    n = len(filepaths)
    out = np.empty((n, IMG_DIM, IMG_DIM, CHANNELS), dtype=np.float32)

    def _one(i_fp):
        i, fp = i_fp
        key = (fp, model_name)
        cached = _img_cache_get(key)
        if cached is not None:
            return i, cached
        bgr = cv2.imread(fp)
        arr = process(bgr, model_name).astype(np.float32, copy=False)
        _img_cache_put(key, arr)
        return i, arr

    if IMG_WORKERS <= 1 or n < 8:
        for i, fp in enumerate(filepaths):
            key = (fp, model_name)
            cached = _img_cache_get(key)
            if cached is None:
                bgr = cv2.imread(fp)
                cached = process(bgr, model_name).astype(np.float32, copy=False)
                _img_cache_put(key, cached)
            out[i] = cached
        return out

    with ThreadPoolExecutor(max_workers=IMG_WORKERS) as ex:
        for i, arr in ex.map(_one, enumerate(filepaths), chunksize=16):
            out[i] = arr
    return out


def _pred_cache_path(d_set, models, jitter_amounts, block_size):
    key = f"{d_set}__models={'-'.join(models)}__img={IMG_DIM}__bs={block_size}__jit={','.join(map(str,jitter_amounts))}__seed={SEED}"
    safe = "".join([c if (c.isalnum() or c in "._-=,") else "_" for c in key])
    return os.path.join(PRED_CACHE_DIR, safe + ".npy")


def make_predictions(d_set, models):
    images_dir = _resolve_images_dir(d_set)
    print("Using images_dir:", images_dir)

    df = pd.read_csv(f"{INPUT_FOLDER}{d_set}.csv")
    df["id_code"] = df["id_code"].apply(lambda x: x + ".png")
    filepaths = [os.path.join(images_dir, fn) for fn in df["id_code"].tolist()]

    block_size = 512
    total = df.index.size

    jitter_amounts = [
        0,
        0.01,
        0.01,
        0.01,
        0.02,
        0.02,
        0.02,
        0.05,
        0.05,
        0.05,
        0.2,
        0.2,
        0.2,
    ]

    cache_path = _pred_cache_path(d_set, models, jitter_amounts, block_size)
    if os.path.exists(cache_path):
        print(f"Loading cached predictions from: {cache_path}")
        return np.load(cache_path)

    final_preds = np.empty((total, NUM_CLASSES), dtype=np.float32)

    num_tta = len(jitter_amounts) * len(models)

    for start in range(0, total, block_size):
        end = min(start + block_size, total)

        acc = np.empty((end - start, num_tta, NUM_CLASSES), dtype=np.float32)

        base_seed = (
            SEED * 1000003 + start * 9176 + (0 if d_set == "train" else 1) * 31337
        ) & 0x7FFFFFFF

        for m, model_name in enumerate(models):
            print(f"Predicting block {start}:{end} with model={model_name} on {d_set}")
            neural_net = load_network(model_name)

            img_block = _load_and_process_block(filepaths[start:end], model_name)

            tta_col0 = m * len(jitter_amounts)
            for i, jit in enumerate(jitter_amounts):
                if jit == 0:
                    preds = _predict_no_aug_tfdata(neural_net, img_block, BATCH_SIZE)
                else:
                    datagen = dataGenerator(jit)
                    preds = _predict_with_tta_sequence(
                        neural_net,
                        datagen,
                        img_block,
                        seed=(base_seed + i * 1315423911) & 0x7FFFFFFF,
                        batch_size=BATCH_SIZE,
                    )
                acc[:, tta_col0 + i, :] = preds

        final_preds[start:end] = np.median(acc, axis=1)

        gc.collect()

    np.save(cache_path, final_preds)
    print(f"Saved cached predictions to: {cache_path}")
    return final_preds




## === cell 7
def find_best_thresholds(train_predictions):
    print("Finding best thresholds...")

    gc.collect()

    train_df = pd.read_csv(f"{INPUT_FOLDER}train.csv")
    y_actual = train_df.diagnosis.astype(np.int64).values

    thresholds = [0.5 for _ in range(NUM_CLASSES)]
    d_thresh = 0.25

    preds = np.asarray(train_predictions, dtype=np.float32)

    def _kappa_for_thresholds(thr_list):
        y_pred = prediction_convert_highest(preds, thr_list)
        return cohen_kappa_score(y_actual, y_pred, weights="quadratic")

    for _ in range(5):
        base_kappa = _kappa_for_thresholds(thresholds)
        for label in range(5):
            currKappa = base_kappa

            thresholds[label] += d_thresh
            kappaUp = _kappa_for_thresholds(thresholds)

            thresholds[label] -= 2 * d_thresh
            kappaDown = _kappa_for_thresholds(thresholds)

            thresholds[label] += d_thresh

            if kappaUp > currKappa:
                thresholds[label] += d_thresh
                base_kappa = kappaUp
            elif kappaDown > currKappa:
                thresholds[label] -= d_thresh
                base_kappa = kappaDown
            else:
                base_kappa = currKappa

        d_thresh /= 2

    gc.collect()
    return thresholds


train_predictions = make_predictions("train", ["normal"])

thresholds = find_best_thresholds(train_predictions)
thresholds = [float(np.clip(t, 0.0, 1.0)) for t in thresholds]
print("Optimized thresholds:", thresholds)

train_df = pd.read_csv(os.path.join(INPUT_FOLDER, "train.csv"))
y_true = train_df["diagnosis"].astype(int).values
y_hat = prediction_convert_highest(train_predictions, thresholds)
print(
    "Train QWK (for sanity):",
    cohen_kappa_score(y_true, y_hat, weights="quadratic"),
)

predictions = make_predictions("test", ["normal"])
as_classes = prediction_convert_highest(predictions, thresholds)
print("First 10 predictions:", as_classes[:10])

test_df = pd.read_csv(os.path.join(INPUT_FOLDER, "test.csv"))
test_df["diagnosis"] = as_classes.astype(int)

test_df = test_df[["id_code", "diagnosis"]]
test_df.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", test_df.shape)
print(test_df.head())
