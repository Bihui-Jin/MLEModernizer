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

0.8977084505420068

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.22769) has done: 'I fix the Keras import/API issues causing the `MessageFactory/GetPrototype` crash and the missing `ImageDataGenerator` attribute by switching to `tensorflow.keras`, which is the supported Keras backend in Kaggle’s TF environment. I also make the generator use `tf.keras.preprocessing.image.ImageDataGenerator` so `flow()` works, while preserving your preprocessing, model architecture, and prediction logic. Finally, I make the weights-path fallback robust (so the model still runs end-to-end if the external weights aren’t present) and keep the submission writing unchanged so it always produces `submission.csv` with the correct columns.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import gc
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import cohen_kappa_score

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.optimizers import Adam

IMG_DIM = 256
BATCH_SIZE = 32
CHANNELS = 3
NUM_CLASSES = 5

NORMAL_WEIGHTS = "../input/densenetmulti/dense-0.800.h5"

CANDIDATE_INPUT_FOLDERS = [
    "../input/aptos2019-blindness-detection/",
    "/kaggle/input/aptos2019-blindness-detection/",
    "/kaggle/data/aptos2019-blindness-detection/",
]
INPUT_FOLDER = None
for p in CANDIDATE_INPUT_FOLDERS:
    if os.path.exists(os.path.join(p, "train.csv")) and os.path.exists(
        os.path.join(p, "test.csv")
    ):
        INPUT_FOLDER = p if p.endswith("/") else p + "/"
        break
if INPUT_FOLDER is None:
    if os.path.exists("/kaggle/data/train.csv") and os.path.exists(
        "/kaggle/data/test.csv"
    ):
        INPUT_FOLDER = "/kaggle/data/"
    else:
        raise FileNotFoundError(
            "Could not locate aptos2019-blindness-detection input folder with train.csv/test.csv."
        )

print("Resolved INPUT_FOLDER:", INPUT_FOLDER)
print("Listing INPUT_FOLDER:", os.listdir(INPUT_FOLDER)[:20])
print(
    "train_images exists:", os.path.exists(os.path.join(INPUT_FOLDER, "train_images"))
)
print("test_images exists:", os.path.exists(os.path.join(INPUT_FOLDER, "test_images")))

RESOLVED_WEIGHTS = NORMAL_WEIGHTS if os.path.exists(NORMAL_WEIGHTS) else None
if RESOLVED_WEIGHTS is None:
    print("WARNING: External weights not found:", NORMAL_WEIGHTS)
    print(
        "         Will fall back to DenseNet121(weights='imagenet') so code runs end-to-end."
    )

os.environ.setdefault("PYTHONHASHSEED", "42")
np.random.seed(42)
tf.random.set_seed(42)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    cv2.setNumThreads(0)
except Exception:
    pass

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass




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
    height, width = img.shape[:2]

    if height > width:
        offset = (height - width) // 2
        return img[offset : offset + width]
    else:
        if height == width:
            return img

        out = np.zeros((width, width) + img.shape[2:], dtype=img.dtype)
        h1 = (width - height) // 2
        h2 = h1 + height

        out[h1:h2, :] = img

        if h1 > 0:
            out[1 : h1 + 1, :] = img[:h1][::-1]

        tail = width - h2
        if tail > 0:
            out[h2 : h2 + tail, :] = img[height - tail : height][::-1]

        return out


def circleMask(img):
    if img.shape[0] != img.shape[1]:
        return img

    dim = img.shape[0]
    half = int(dim / 2)

    circle_mask = np.zeros((dim, dim), np.uint8)
    circle_mask = cv2.circle(circle_mask, (half, half), half, 1, thickness=-1)

    return cv2.bitwise_and(img, img, mask=circle_mask)


_CLAHE_CACHE = {}


def clahe_gray(gray, clipLimit=3.5, grid=4):
    key = (float(clipLimit), int(grid))
    clahe = _CLAHE_CACHE.get(key)
    if clahe is None:
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(grid, grid))
        _CLAHE_CACHE[key] = clahe
    return clahe.apply(gray)


_GAMMA_LUT_CACHE = {}


def _gamma_lut(gamma):
    key = float(np.round(gamma, 6))
    lut = _GAMMA_LUT_CACHE.get(key)
    if lut is None:
        invGamma = 1.0 / key
        lut = np.array(
            [((i / 255.0) ** invGamma) * 255 for i in range(256)], dtype=np.uint8
        )
        _GAMMA_LUT_CACHE[key] = lut
    return lut


def adjust_gamma(image_, gamma=1.0):
    return cv2.LUT(image_, _gamma_lut(gamma))


def processBenNormal(bgr):
    if bgr is None:
        return np.full((IMG_DIM, IMG_DIM, CHANNELS), 128, dtype=np.uint8)

    green = bgr[:, :, 1]

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
        equalised = resized
    else:
        equalised = adjust_gamma(resized, 1 + np.log(90) - np.log(med))

    bens = benYCC(equalised, weight=3, gamma=20)
    return cv2.cvtColor(bens, cv2.COLOR_BGR2RGB)


pre_process_function = processBenNormal




## === cell 2
_BASE_DATAGEN = image.ImageDataGenerator(
    rescale=1.0 / 255,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=[0.8, 1.0],
    rotation_range=0,
    brightness_range=[1.0, 1.0],
    fill_mode="mirror",
    channel_shift_range=0.0,
)


def _set_datagen_jitter(datagen, jitter):
    if jitter > 0.01:
        datagen.horizontal_flip = True
        datagen.vertical_flip = True
    else:
        datagen.horizontal_flip = False
        datagen.vertical_flip = False

    datagen.zoom_range = [max(0.8, 1 - 5 * jitter), 1]
    datagen.rotation_range = int(600 * jitter)
    datagen.brightness_range = [1 - jitter / 3, 1 + jitter / 3]
    datagen.channel_shift_range = float(int(30 * jitter))


def dataGenerator(jitter=0.1):
    _set_datagen_jitter(_BASE_DATAGEN, jitter)
    return _BASE_DATAGEN




## === cell 3
def test_datagen_plot(processing_function, jitter=0.3):
    images_dir = f"{INPUT_FOLDER}test_images/"
    df = pd.read_csv(f"{INPUT_FOLDER}test.csv")
    df.id_code = df.id_code.apply(lambda x: x + ".png")

    img_block = np.empty((100, IMG_DIM, IMG_DIM, CHANNELS), dtype=np.uint8)
    for i, filename in enumerate(df[:100].id_code):
        bgr = cv2.imread(os.path.join(images_dir, filename))
        img_block[i, :, :, :] = processing_function(bgr)

    figure = plt.figure(figsize=(8, 8))
    datagen_sample = dataGenerator(jitter).flow(img_block)
    for x in datagen_sample:
        for j in range(16):
            ax = figure.add_subplot(4, 4, j + 1)
            plt.imshow(x[j])
            ax.axis("off")
        break
    plt.show()




## === cell 4
def create_model(weights_path=None):
    backbone_weights = None if weights_path is not None else "imagenet"

    model = Sequential()
    model.add(
        DenseNet121(
            weights=backbone_weights,
            include_top=False,
            input_shape=(IMG_DIM, IMG_DIM, CHANNELS),
        )
    )
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation="sigmoid"))

    if weights_path is not None:
        model.load_weights(weights_path)

    model.compile(
        optimizer=Adam(learning_rate=0.00005),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model




## === cell 5
import multiprocessing as mp


def _cache_path_for_dataset(d_set):
    return os.path.join("/kaggle/working", f"cache_{d_set}_{IMG_DIM}.npy")


_GLOBAL_PATHS = None


def _init_pool(paths):
    global _GLOBAL_PATHS
    _GLOBAL_PATHS = paths
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass


def _preprocess_one_idx(i):
    path = _GLOBAL_PATHS[i]
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    return i, processBenNormal(bgr)


def _load_or_build_preprocessed(d_set, processing_function):
    images_dir = f"{INPUT_FOLDER}{d_set}_images/"
    df = pd.read_csv(f"{INPUT_FOLDER}{d_set}.csv")
    df.id_code = df.id_code.apply(lambda x: x + ".png")

    cache_path = _cache_path_for_dataset(d_set)
    if os.path.exists(cache_path):
        arr = np.load(cache_path, mmap_mode="r")
        if (
            arr.shape[0] == len(df)
            and arr.shape[1:] == (IMG_DIM, IMG_DIM, CHANNELS)
            and arr.dtype == np.uint8
        ):
            return df, arr

    total = len(df)
    tmp_path = cache_path + ".tmp"
    if os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    arr_mm = np.lib.format.open_memmap(
        tmp_path, mode="w+", dtype=np.uint8, shape=(total, IMG_DIM, IMG_DIM, CHANNELS)
    )

    paths = [os.path.join(images_dir, fn) for fn in df.id_code.values]

    workers = max(1, min(8, (os.cpu_count() or 2)))
    chunksize = 4096  # larger chunks reduces IPC overhead; output is indexed so order doesn't matter.
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=workers, initializer=_init_pool, initargs=(paths,)) as pool:
        done = 0
        for i, out in pool.imap_unordered(
            _preprocess_one_idx, range(total), chunksize=chunksize
        ):
            arr_mm[i] = out
            done += 1
            if done % 16384 == 0:
                arr_mm.flush()
    arr_mm.flush()
    del arr_mm
    gc.collect()

    if os.path.exists(cache_path):
        try:
            os.remove(cache_path)
        except Exception:
            pass
    os.replace(tmp_path, cache_path)

    arr = np.load(cache_path, mmap_mode="r")
    return df, arr


def _jitter_params(jitter):
    zoom_min = max(0.8, 1.0 - 5.0 * jitter)
    rot_deg = int(600.0 * jitter)
    bmin, bmax = (1.0 - jitter / 3.0), (1.0 + jitter / 3.0)
    cshift = float(int(30.0 * jitter))
    do_flip = jitter > 0.01
    return zoom_min, rot_deg, bmin, bmax, cshift, do_flip


def _fill_mirror_pad(img, out_h, out_w):
    ih = tf.shape(img)[0]
    iw = tf.shape(img)[1]
    ph = tf.maximum(0, out_h - ih)
    pw = tf.maximum(0, out_w - iw)
    pad_top = ph // 2
    pad_bottom = ph - pad_top
    pad_left = pw // 2
    pad_right = pw - pad_left
    return tf.pad(
        img, [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode="REFLECT"
    )


@tf.function(reduce_retracing=True)
def _augment_one_stateless(img_u8, seed, jitter):
    img = tf.cast(img_u8, tf.float32) / 255.0

    zoom_min, rot_deg, bmin, bmax, cshift, do_flip = _jitter_params(jitter)

    if jitter > 0.0:
        b = tf.random.stateless_uniform(
            [], seed=seed + tf.constant([11, 17], tf.int32), minval=bmin, maxval=bmax
        )
        img = tf.clip_by_value(img * b, 0.0, 1.0)

    if cshift > 0.0:
        shift = tf.random.stateless_uniform(
            [3],
            seed=seed + tf.constant([23, 29], tf.int32),
            minval=-cshift,
            maxval=cshift,
        )
        img = tf.clip_by_value(img + shift / 255.0, 0.0, 1.0)

    if do_flip:
        r = tf.random.stateless_uniform(
            [2], seed=seed + tf.constant([31, 37], tf.int32)
        )
        img = tf.cond(r[0] < 0.5, lambda: tf.image.flip_left_right(img), lambda: img)
        img = tf.cond(r[1] < 0.5, lambda: tf.image.flip_up_down(img), lambda: img)

    if jitter > 0.0:
        z = tf.random.stateless_uniform(
            [], seed=seed + tf.constant([41, 43], tf.int32), minval=zoom_min, maxval=1.0
        )
        in_h = tf.shape(img)[0]
        in_w = tf.shape(img)[1]
        new_h = tf.cast(tf.cast(in_h, tf.float32) * z, tf.int32)
        new_w = tf.cast(tf.cast(in_w, tf.float32) * z, tf.int32)
        img2 = tf.image.resize(img, [new_h, new_w], method="bilinear", antialias=False)
        img2 = _fill_mirror_pad(img2, in_h, in_w)
        off_h = (tf.shape(img2)[0] - in_h) // 2
        off_w = (tf.shape(img2)[1] - in_w) // 2
        img = tf.image.crop_to_bounding_box(img2, off_h, off_w, in_h, in_w)

    if rot_deg > 0:
        ang = tf.random.stateless_uniform(
            [],
            seed=seed + tf.constant([47, 53], tf.int32),
            minval=-float(rot_deg),
            maxval=float(rot_deg),
        )
        rad = ang * (np.pi / 180.0)
        pad = tf.cast(tf.cast(tf.shape(img)[0], tf.float32) * 0.25, tf.int32)
        imgp = tf.pad(img, [[pad, pad], [pad, pad], [0, 0]], mode="REFLECT")
        imgp = tf.raw_ops.ImageProjectiveTransformV3(
            images=tf.expand_dims(imgp, 0),
            transforms=tf.expand_dims(
                tf.stack(
                    [
                        tf.cos(rad),
                        -tf.sin(rad),
                        0.0,
                        tf.sin(rad),
                        tf.cos(rad),
                        0.0,
                        0.0,
                        0.0,
                    ]
                ),
                0,
            ),
            output_shape=tf.shape(imgp)[:2],
            fill_value=0.0,
            interpolation="BILINEAR",
            fill_mode="REFLECT",
        )[0]
        img = tf.image.crop_to_bounding_box(imgp, pad, pad, IMG_DIM, IMG_DIM)

    return img


def _predict_block_tf(model, img_block_u8, jitter, seed_base):
    n = img_block_u8.shape[0]
    ds = tf.data.Dataset.from_tensor_slices(img_block_u8)

    def _map_fn(i, x):
        seed = tf.stack([tf.cast(seed_base, tf.int32), tf.cast(i, tf.int32)], axis=0)
        return _augment_one_stateless(x, seed, tf.constant(jitter, tf.float32))

    ds = ds.enumerate()
    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    preds = model.predict(ds, verbose=0)
    return preds[:n]


def make_predictions(d_set, processing_function, model, jitters=5):
    df, preprocessed = _load_or_build_preprocessed(d_set, processing_function)

    block_size = 512  # speed-only: larger blocks reduce Python loop overhead; does not change results.
    total = df.index.size
    predictions = np.zeros((total, NUM_CLASSES), dtype=np.float32)

    print(f"Making predictions on the {d_set} dataset. Total: {total}")

    base_seed = 12345
    pred_jitters_buf = np.empty((block_size, jitters, NUM_CLASSES), dtype=np.float32)

    for start in range(0, total, block_size):
        end = min(start + block_size, total)
        n = end - start
        img_block = np.asarray(
            preprocessed[start:end]
        )  # ensure contiguous for fast TF ingestion

        jit = 0.0
        for i in range(jitters):
            pred = _predict_block_tf(
                model, img_block, jitter=jit, seed_base=int(base_seed + i)
            )
            pred_jitters_buf[:n, i, :] = pred
            jit += 0.02

        predictions[start:end] = np.median(pred_jitters_buf[:n, :, :], axis=1)
        print(f"{start} - {end} finished")

    return predictions




## === cell 6
def prediction_convert_sum(predictions, thresholds):
    thr = np.asarray(thresholds, dtype=predictions.dtype)[None, :]
    y_val = (predictions > thr).sum(axis=1).astype(np.int32) - 1
    return y_val


def prediction_convert_highest(predictions, thresholds):
    thr = np.asarray(thresholds, dtype=predictions.dtype)[None, :]
    mask = predictions > thr  # (n,5) bool

    any_true = mask.any(axis=1)
    idx = mask.shape[1] - 1 - np.argmax(mask[:, ::-1], axis=1)
    y_val = np.where(any_true, idx, 0).astype(np.int32)
    return y_val


def find_best_thresholds(train_predictions):
    print("Finding best thresholds...")
    train_df = pd.read_csv(f"{INPUT_FOLDER}train.csv")
    y_actual = train_df.diagnosis.astype(int).values

    thresholds = [0.5 for _ in range(NUM_CLASSES)]
    d_thresh = 0.25

    preds = train_predictions
    for sweep in range(5):
        thr = np.asarray(thresholds, dtype=preds.dtype)[None, :]
        base_mask = preds > thr  # (n,5) bool

        for label in range(5):
            y_curr = base_mask.sum(axis=1).astype(np.int32) - 1
            currKappa = cohen_kappa_score(y_actual, y_curr, weights="quadratic")

            thr_up = thresholds[label] + d_thresh
            col_up = preds[:, label] > thr_up
            mask_up = base_mask.copy()
            mask_up[:, label] = col_up
            y_up = mask_up.sum(axis=1).astype(np.int32) - 1
            kappaUp = cohen_kappa_score(y_actual, y_up, weights="quadratic")

            thr_down = thresholds[label] - d_thresh
            col_down = preds[:, label] > thr_down
            mask_down = base_mask.copy()
            mask_down[:, label] = col_down
            y_down = mask_down.sum(axis=1).astype(np.int32) - 1
            kappaDown = cohen_kappa_score(y_actual, y_down, weights="quadratic")

            if kappaUp > currKappa:
                thresholds[label] += d_thresh
                base_mask[:, label] = col_up
            elif kappaDown > currKappa:
                thresholds[label] -= d_thresh
                base_mask[:, label] = col_down

        d_thresh /= 2

    return thresholds




## === cell 7
model = create_model(RESOLVED_WEIGHTS)

train_preds = make_predictions("train", processBenNormal, model, jitters=3)
thresholds = find_best_thresholds(train_preds)
print("Optimized thresholds:", thresholds)

test_preds = make_predictions("test", processBenNormal, model, jitters=5)
test_classes = prediction_convert_highest(test_preds, thresholds)
test_classes = np.clip(test_classes, 0, 4).astype(int)

print("First 10 predictions:", test_classes[:10])

test_df = pd.read_csv(os.path.join(INPUT_FOLDER, "test.csv"))
test_df["diagnosis"] = test_classes
test_df.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", test_df.shape)
print(test_df.head())

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
OperatorNotAllowedInGraphError            Traceback (most recent call last)
/tmp/ipykernel_11/4229627568.py in <cell line: 0>()
      1 model = create_model(RESOLVED_WEIGHTS)
      2 
----> 3 train_preds = make_predictions("train", processBenNormal, model, jitters=3)
      4 thresholds = find_best_thresholds(train_preds)
      5 print("Optimized thresholds:", thresholds)

/tmp/ipykernel_11/3860344897.py in make_predictions(d_set, processing_function, model, jitters)
    232         jit = 0.0
    233         for i in range(jitters):
--> 234             pred = _predict_block_tf(
    235                 model, img_block, jitter=jit, seed_base=int(base_seed + i)
    236             )

/tmp/ipykernel_11/3860344897.py in _predict_block_tf(model, img_block_u8, jitter, seed_base)
    204 
    205     ds = ds.enumerate()
--> 206     ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    207     ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    208     ds = ds.prefetch(tf.data.AUTOTUNE)

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

OperatorNotAllowedInGraphError: in user code:

    File "/tmp/ipykernel_11/3860344897.py", line 203, in _map_fn  *
        return _augment_one_stateless(x, seed, tf.constant(jitter, tf.float32))
    File "/tmp/ipykernel_11/3860344897.py", line 114, in _augment_one_stateless  *
        zoom_min, rot_deg, bmin, bmax, cshift, do_flip = _jitter_params(jitter)
    File "/tmp/ipykernel_11/3860344897.py", line 87, in _jitter_params  *
        zoom_min = max(0.8, 1.0 - 5.0 * jitter)

    OperatorNotAllowedInGraphError: Using a symbolic `tf.Tensor` as a Python `bool` is not allowed. You can attempt the following resolutions to the problem: If you are running in Graph mode, use Eager execution mode or decorate this function with @tf.function. If you are using AutoGraph, you can try decorating this function with @tf.function. If that does not work, then you may be using an unsupported feature or your source code may not be visible to AutoGraph. See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/limitations.md#access-to-source-code for more information.
