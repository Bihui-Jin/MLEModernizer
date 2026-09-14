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

0.8904328752378834

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.09862) has done: 'I fix the two runtime errors blocking execution: the Keras `ImageDataGenerator` import/usage mismatch and the early `MessageFactory.GetPrototype` crash caused by importing standalone `keras` in this environment. To keep core logic identical, I switch to `tensorflow.keras` equivalents (same layers/architecture, same preprocessing semantics) and keep the rest of the pipeline unchanged. I also make weight-file existence checks robust and ensure the script always reaches the submission-writing cell, producing `submission.csv` with the required columns. These changes are necessary for a valid end-to-end run; score then depend on whether the provided external weights are available, otherwise it fall back to ImageNet weights as already intended.'
- What this solution (achieved -0.01721) has done: 'The crash in cell 0 is coming from a protobuf/TensorFlow compatibility issue that can happen when importing TensorFlow after other packages; the minimal reliable fix in Kaggle is to force the pure-Python protobuf implementation before importing TensorFlow. I move that environment setting to the very top (before `import tensorflow as tf`) to prevent the `MessageFactory.GetPrototype` AttributeError. Then I keep the rest of the pipeline and model logic unchanged, only adding small safety guards for image reading and prediction-to-class conversion to avoid invalid `-1` labels (which can severely hurt kappa and matches your very low current score). The submission writing stays the same and always produce `submission.csv` with the required columns.'
- What this solution (achieved -0.03857) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *and* disabling the C++ implementation before any TensorFlow-related imports, which is the most reliable workaround for the `MessageFactory.GetPrototype` error in Kaggle. I keep the model/prediction logic unchanged, but make the environment/seed setup deterministic and stable so the run completes consistently. I also add a small guard to ensure `submission.csv` is always written with the correct columns and integer labels, without changing the underlying prediction semantics. No changes are made to the architecture, preprocessing functions, augmentation design, or prediction aggregation beyond what’s necessary for stability.'
- What this solution (achieved -0.08247) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation *before any TensorFlow-related import* and by removing imports that can indirectly trigger the bad proto path in this Kaggle image. Then I keep your model/prediction pipeline unchanged, but add a tiny, score-positive correctness fix in `label_convert`: mapping sigmoid multi-label outputs to ordinal classes should not subtract 1 (that shifts everything down and can yield pathological kappas); instead it should count positives directly and clamp to 0–4. Finally, I add a minimal safety check that the located input folder actually contains the expected `*_images` directories used later, so the run reliably reaches the submission-writing cell and produces a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_CPP", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import gc
import random
import numpy as np
import pandas as pd
import cv2
import psutil
import matplotlib.pyplot as plt

from sklearn.metrics import cohen_kappa_score, confusion_matrix  # noqa: F401

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.optimizers import Adam

SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass
try:
    tf.config.threading.set_intra_op_parallelism_threads(
        max(1, psutil.cpu_count(logical=True) // 2)
    )
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

try:
    cv2.setNumThreads(max(1, psutil.cpu_count(logical=True) // 2))
except Exception:
    pass

IMG_DIM = 224
BATCH_SIZE = 32
CHANNELS = 3
NUM_CLASSES = 5

NORMAL_WEIGHTS = "../input/densenetmulti/ben_normal_-0.9021.h5"
WEIRD_WEIGHTS = "../input/densenetmulti/ben_weird_-0.9048.h5"


def _pick_input_folder():
    """
    Robustly select the dataset folder that actually exists in this environment.
    Keeps I/O semantics identical (reads train/test CSVs and images from INPUT_FOLDER).
    """
    candidates = [
        "../input/aptos2019-blindness-detection/",
        "/kaggle/input/aptos2019-blindness-detection/",
        "/kaggle/data/aptos2019-blindness-detection/",
        "../input/",
        "/kaggle/input/",
        "/kaggle/data/",
    ]
    for c in candidates:
        c2 = c if c.endswith("/") else c + "/"
        if os.path.exists(os.path.join(c2, "train.csv")) and os.path.exists(
            os.path.join(c2, "test.csv")
        ):
            return c2

    for base in ["../input", "/kaggle/input", "/kaggle/data"]:
        if os.path.isdir(base):
            for name in os.listdir(base):
                cand = os.path.join(base, name)
                if (
                    os.path.isdir(cand)
                    and os.path.exists(os.path.join(cand, "train.csv"))
                    and os.path.exists(os.path.join(cand, "test.csv"))
                ):
                    return cand + ("" if cand.endswith("/") else "/")

    raise FileNotFoundError(
        "Could not locate APTOS input folder containing train.csv and test.csv"
    )


INPUT_FOLDER = _pick_input_folder()

for needed in ["train_images", "test_images"]:
    if not os.path.isdir(os.path.join(INPUT_FOLDER, needed)):
        nested = os.path.join(INPUT_FOLDER, "aptos2019-blindness-detection")
        if os.path.isdir(os.path.join(nested, needed)) and os.path.exists(
            os.path.join(nested, "train.csv")
        ):
            INPUT_FOLDER = nested + ("" if nested.endswith("/") else "/")
            break

print("INPUT_FOLDER =", INPUT_FOLDER)
print("CPU count =", psutil.cpu_count())
print("TF version =", tf.__version__)

for p in ["../", "../input/", INPUT_FOLDER]:
    try:
        print(p, "->", os.listdir(p)[:10])
    except Exception as e:
        print(p, "-> (unavailable)", repr(e))

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
    invGamma = 1.0 / gamma
    key = float(invGamma)
    table = _GAMMA_LUT_CACHE.get(key)
    if table is None:
        table = np.array(
            [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")
        _GAMMA_LUT_CACHE[key] = table
    return cv2.LUT(image, table)


def _ben_shared_equalised(bgr):
    green = bgr[:, :, 1]
    cropped = crop(green, bgr, 0.02)
    squared = reflectAndSquareUp(cropped)
    resized = cv2.resize(squared, (2 * IMG_DIM, 2 * IMG_DIM))
    circled = circleMask(resized)
    med = float(np.median(circled))
    med = med if med > 1e-6 else 1e-6
    equalised = adjust_gamma(circled, 1 + np.log(90) - np.log(med))
    return equalised


def processBenNormal(bgr):
    equalised = _ben_shared_equalised(bgr)
    resized_again = cv2.resize(benYCC(equalised), (IMG_DIM, IMG_DIM))
    return cv2.cvtColor(resized_again, cv2.COLOR_BGR2RGB)


def processBenWeird(bgr):
    equalised = _ben_shared_equalised(bgr)
    resized_again = cv2.resize(benSimple(equalised), (IMG_DIM, IMG_DIM))
    return cv2.cvtColor(resized_again, cv2.COLOR_BGR2RGB)


gc.collect()




## === cell 2
def dataGenerator(jitter=0.1):
    datagen = ImageDataGenerator(
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


gc.collect()



## === cell 3
figure = plt.figure(figsize=(22, 20))


def test_datagen_plot():
    sample_df = pd.read_csv(f"{INPUT_FOLDER}test.csv")
    sample_df.id_code = sample_df.id_code.apply(lambda x: x + ".png")

    img_list = np.empty((32, IMG_DIM, IMG_DIM, 3), dtype=np.float32)
    for i, filename in enumerate(sample_df[:32].id_code):
        try:
            bgr = cv2.imread(f"{INPUT_FOLDER}test_images/{filename}")
            if bgr is None:
                raise ValueError("cv2.imread returned None")
            img_list[i, :, :, :] = processBenNormal(bgr).astype(np.float32)
        except Exception:
            img_list[i, :, :, :] = 128.0

    datagen_sample = dataGenerator(0.03).flow(img_list, shuffle=True, batch_size=32)

    for x in datagen_sample:
        for j in range(16):
            ax = figure.add_subplot(4, 4, j + 1)
            img = np.clip(x[j], 0, 1)
            ax.imshow(img)
            ax.axis("off")
        break


gc.collect()




## === cell 4
def _weights_exist(path):
    return path is not None and os.path.exists(path) and os.path.isfile(path)


print("NORMAL_WEIGHTS exists:", _weights_exist(NORMAL_WEIGHTS), NORMAL_WEIGHTS)
print("WEIRD_WEIGHTS exists:", _weights_exist(WEIRD_WEIGHTS), WEIRD_WEIGHTS)

gc.collect()




## === cell 5
def create_model(weights):
    """
    If external weight files are missing, fall back to ImageNet backbone weights
    so the pipeline runs end-to-end and yields a valid submission.csv.
    Core architecture remains: DenseNet121 (no top) -> GAP -> Dropout -> Dense(sigmoid).
    """
    model = Sequential()
    if _weights_exist(weights):
        backbone = DenseNet121(
            weights=None, include_top=False, input_shape=(IMG_DIM, IMG_DIM, CHANNELS)
        )
    else:
        backbone = DenseNet121(
            weights="imagenet",
            include_top=False,
            input_shape=(IMG_DIM, IMG_DIM, CHANNELS),
        )

    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation="sigmoid"))

    if _weights_exist(weights):
        model.load_weights(weights)

    model.compile(
        optimizer=Adam(learning_rate=0.00005),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


gc.collect()



## === cell 6
model = None
gc.collect()



## === cell 7
CACHE_DIR = "/kaggle/working/aptos_cache_preprocessed/"
os.makedirs(CACHE_DIR, exist_ok=True)

_AUTOTUNE = tf.data.experimental.AUTOTUNE

_ID_CACHE = {}  # d_set -> np.ndarray of filenames with ".png"


def _get_ids_png(d_set):
    ids = _ID_CACHE.get(d_set)
    if ids is not None:
        return ids
    df = pd.read_csv(f"{INPUT_FOLDER}{d_set}.csv", usecols=["id_code"])
    ids = (df["id_code"].astype(str) + ".png").values
    _ID_CACHE[d_set] = ids
    return ids


def _cache_path(d_set, proc_name, img_dim=IMG_DIM):
    return os.path.join(CACHE_DIR, f"{d_set}_{proc_name}_{img_dim}.npy")


def _pred_cache_path(d_set, proc_name, weights_tag):
    safe_tag = str(weights_tag).replace("/", "_").replace("..", "__")
    return os.path.join(CACHE_DIR, f"pred_{d_set}_{proc_name}_{safe_tag}.npy")


def _shared_cache_path(d_set, img_dim=IMG_DIM):
    return os.path.join(CACHE_DIR, f"{d_set}_BENSHARED_equalised_{2*img_dim}.npy")


from concurrent.futures import ThreadPoolExecutor

_N_WORKERS = max(1, min(8, (psutil.cpu_count(logical=True) or 2) // 2))


def _load_or_build_shared_equalised(d_set):
    cache_path = _shared_cache_path(d_set, IMG_DIM)
    if os.path.exists(cache_path):
        return np.load(cache_path, mmap_mode="r")

    images_dir = f"{INPUT_FOLDER}{d_set}_images/"
    ids = _get_ids_png(d_set)
    total = int(ids.shape[0])

    fp = np.lib.format.open_memmap(
        cache_path,
        mode="w+",
        dtype=np.uint8,
        shape=(total, 2 * IMG_DIM, 2 * IMG_DIM, CHANNELS),
    )

    imread = cv2.imread

    def _worker(i):
        filename = ids[i]
        try:
            bgr = imread(images_dir + filename)
            if bgr is None:
                raise ValueError("cv2.imread returned None")
            return i, _ben_shared_equalised(bgr).astype(np.uint8, copy=False)
        except Exception:
            return i, None

    with ThreadPoolExecutor(max_workers=_N_WORKERS) as ex:
        for i, out in ex.map(_worker, range(total), chunksize=32):
            if out is None:
                fp[i] = 128
            else:
                fp[i] = out

    fp.flush()
    del fp
    return np.load(cache_path, mmap_mode="r")


def _final_from_equalised(equalised_uint8, proc_name):
    if proc_name == "processBenNormal":
        out = cv2.resize(benYCC(equalised_uint8), (IMG_DIM, IMG_DIM))
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    elif proc_name == "processBenWeird":
        out = cv2.resize(benSimple(equalised_uint8), (IMG_DIM, IMG_DIM))
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("Unknown proc_name: %r" % proc_name)


def _load_or_build_preprocessed(d_set, processing_function, proc_name):
    cache_path = _cache_path(d_set, proc_name, IMG_DIM)
    if os.path.exists(cache_path):
        return np.load(cache_path, mmap_mode="r")

    ids = _get_ids_png(d_set)
    total = int(ids.shape[0])

    if proc_name in ("processBenNormal", "processBenWeird"):
        shared = _load_or_build_shared_equalised(d_set)

        fp = np.lib.format.open_memmap(
            cache_path,
            mode="w+",
            dtype=np.uint8,
            shape=(total, IMG_DIM, IMG_DIM, CHANNELS),
        )

        def _worker(i):
            try:
                return i, _final_from_equalised(shared[i], proc_name).astype(
                    np.uint8, copy=False
                )
            except Exception:
                return i, None

        with ThreadPoolExecutor(max_workers=_N_WORKERS) as ex:
            for i, out in ex.map(_worker, range(total), chunksize=64):
                if out is None:
                    fp[i] = 128
                else:
                    fp[i] = out

        fp.flush()
        del fp
        return np.load(cache_path, mmap_mode="r")

    images_dir = f"{INPUT_FOLDER}{d_set}_images/"
    fp = np.lib.format.open_memmap(
        cache_path,
        mode="w+",
        dtype=np.uint8,
        shape=(total, IMG_DIM, IMG_DIM, CHANNELS),
    )

    imread = cv2.imread
    proc = processing_function

    def _worker(i):
        filename = ids[i]
        try:
            bgr = imread(images_dir + filename)
            if bgr is None:
                raise ValueError("cv2.imread returned None")
            return i, proc(bgr).astype(np.uint8, copy=False)
        except Exception:
            return i, None

    with ThreadPoolExecutor(max_workers=_N_WORKERS) as ex:
        for i, out in ex.map(_worker, range(total), chunksize=32):
            if out is None:
                fp[i] = 128
            else:
                fp[i] = out

    fp.flush()
    del fp
    return np.load(cache_path, mmap_mode="r")


class _MemmapBatchSeq(tf.keras.utils.Sequence):
    def __init__(self, memmap_uint8, batch_size):
        self.x = memmap_uint8
        self.bs = int(batch_size)
        self.n = int(len(memmap_uint8))

    def __len__(self):
        return (self.n + self.bs - 1) // self.bs

    def __getitem__(self, idx):
        start = idx * self.bs
        end = min(start + self.bs, self.n)
        batch = np.asarray(self.x[start:end], dtype=np.float32)
        batch *= 1.0 / 255.0
        return batch


_PRED_CACHE = {}  # (d_set, proc_name, weights_path)-> np.ndarray float32


def make_predictions(d_set, processing_function, jitters=5, weights_tag=None):
    proc_name = processing_function.__name__
    cache_key = (d_set, proc_name, str(weights_tag), int(jitters))
    if cache_key in _PRED_CACHE:
        return _PRED_CACHE[cache_key]

    disk_pred_path = _pred_cache_path(d_set, proc_name, weights_tag)
    if os.path.exists(disk_pred_path):
        preds = np.load(disk_pred_path, mmap_mode="r")
        preds = np.asarray(preds, dtype=np.float32)
        _PRED_CACHE[cache_key] = preds
        return preds

    preprocessed = _load_or_build_preprocessed(d_set, processing_function, proc_name)
    total = int(len(preprocessed))

    print(f"Making predictions on the {d_set} dataset. Total: {total}")

    seq = _MemmapBatchSeq(preprocessed, BATCH_SIZE)

    predictions = model.predict(
        seq,
        verbose=0,
        workers=max(1, min(4, _N_WORKERS)),
        use_multiprocessing=False,
        max_queue_size=16,
    )
    predictions = predictions.astype(np.float32, copy=False)

    np.save(disk_pred_path, predictions)
    _PRED_CACHE[cache_key] = predictions
    return predictions


gc.collect()




## === cell 8
def _fit_thresholds_by_distribution(train_preds, y_true, num_classes=5):
    """
    Find per-output thresholds t_k so that the predicted cumulative positives
    produce a class distribution similar to y_true.
    For k=0..num_classes-1, threshold t_k is chosen so that:
      P(pred[:,k] > t_k) ~= P(y_true >= k+1)
    """
    y_true = np.asarray(y_true).astype(int)
    thresholds = np.zeros(num_classes, dtype=np.float32)
    for k in range(num_classes):
        target_rate = float(np.mean(y_true >= (k + 1)))
        if target_rate <= 0.0:
            thresholds[k] = 1.0 + 1e-6
        elif target_rate >= 1.0:
            thresholds[k] = -1e-6
        else:
            thresholds[k] = float(np.quantile(train_preds[:, k], 1.0 - target_rate))
    return thresholds


def label_convert(preds, thresholds=None):
    preds = np.asarray(preds)
    if thresholds is None:
        y_val = preds > 0.5
    else:
        thr = np.asarray(thresholds).reshape(1, -1)
        y_val = preds > thr
    labels = y_val.astype(int).sum(axis=1)
    return np.clip(labels, 0, NUM_CLASSES - 1).astype(np.int64)


gc.collect()



## === cell 9
model = create_model(NORMAL_WEIGHTS)
normal_test_preds = make_predictions(
    "test", processBenNormal, weights_tag=NORMAL_WEIGHTS
)

model = create_model(WEIRD_WEIGHTS)
weird_test_preds = make_predictions("test", processBenWeird, weights_tag=WEIRD_WEIGHTS)

test_predictions = (normal_test_preds + weird_test_preds) / 2.0

thresholds = None
try:
    train_df = pd.read_csv(INPUT_FOLDER + "train.csv", usecols=["diagnosis"])
    y_train = train_df["diagnosis"].values.astype(int)

    model = create_model(NORMAL_WEIGHTS)
    normal_train_preds = make_predictions(
        "train", processBenNormal, weights_tag=NORMAL_WEIGHTS
    )

    model = create_model(WEIRD_WEIGHTS)
    weird_train_preds = make_predictions(
        "train", processBenWeird, weights_tag=WEIRD_WEIGHTS
    )

    train_predictions = (normal_train_preds + weird_train_preds) / 2.0
    thresholds = _fit_thresholds_by_distribution(
        train_predictions, y_train, NUM_CLASSES
    )

    train_classes = label_convert(train_predictions, thresholds=thresholds)
    kappa = cohen_kappa_score(y_train, train_classes, weights="quadratic")
    print("Fitted thresholds:", thresholds)
    print("Train QWK (distribution-calibrated):", kappa)
except Exception as e:
    print("Threshold calibration failed; falling back to fixed 0.5. Error:", repr(e))
    thresholds = None

test_classes = label_convert(test_predictions, thresholds=thresholds)

print(test_predictions[:3])
print(test_classes[:10])

gc.collect()



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1421689722.py in <cell line: 0>()
      1 model = create_model(NORMAL_WEIGHTS)
----> 2 normal_test_preds = make_predictions(
      3     "test", processBenNormal, weights_tag=NORMAL_WEIGHTS
      4 )
      5 

/tmp/ipykernel_11/864247147.py in make_predictions(d_set, processing_function, jitters, weights_tag)
    203     # --- Speed: allow Keras to prefetch batches from Sequence in parallel.
    204     # Preserves semantics: same batches/order; only overlaps CPU-to-model feeding.
--> 205     predictions = model.predict(
    206         seq,
    207         verbose=0,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    117             return fn(*args, **kwargs)
    118         except Exception as e:
--> 119             filtered_tb = _process_traceback_frames(e.__traceback__)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`

TypeError: TensorFlowTrainer.predict() got an unexpected keyword argument 'workers'

## === cell 10
test_df = pd.read_csv(INPUT_FOLDER + "test.csv", usecols=["id_code"])
test_df["diagnosis"] = test_classes.astype(np.int64)

test_df = test_df[["id_code", "diagnosis"]]
test_df.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", test_df.shape)
print(test_df.head())
print(
    "submission.csv exists:",
    os.path.exists("submission.csv"),
    "size:",
    os.path.getsize("submission.csv") if os.path.exists("submission.csv") else None,
)

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/176683561.py in <cell line: 0>()
      1 test_df = pd.read_csv(INPUT_FOLDER + "test.csv", usecols=["id_code"])
----> 2 test_df["diagnosis"] = test_classes.astype(np.int64)
      3 
      4 test_df = test_df[["id_code", "diagnosis"]]
      5 test_df.to_csv("submission.csv", index=False)

NameError: name 'test_classes' is not defined
