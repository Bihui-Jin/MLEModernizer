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

try:
    from google.protobuf import message_factory as _message_factory  # noqa: E402

    MF = getattr(_message_factory, "MessageFactory", None)
    if MF is not None and not hasattr(MF, "GetPrototype"):
        if hasattr(MF, "GetMessageClass"):

            def _GetPrototype(self, descriptor):
                return self.GetMessageClass(descriptor)

            MF.GetPrototype = _GetPrototype
        else:

            def _GetPrototype(self, descriptor):
                return self._InternalGetPrototype(descriptor)

            MF.GetPrototype = _GetPrototype
except Exception:
    pass

import gc
import math
import numpy as np
import pandas as pd
import cv2
from concurrent.futures import ThreadPoolExecutor

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.optimizers import Adam

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

IMG_DIM = 256
BATCH_SIZE = 32
CHANNEL_SIZE = 3
NUM_CLASSES = 5

candidate_roots = [
    "/kaggle/input/aptos2019-blindness-detection",
    "/kaggle/input",
    "../input/aptos2019-blindness-detection",
    "../input",
    "/kaggle/data/input/aptos2019-blindness-detection",
    "/kaggle/data/input",
    "/kaggle/data/aptos2019-blindness-detection",
    "/kaggle/data",
]

INPUT_FOLDER = None
for root in candidate_roots:
    if not os.path.exists(root):
        continue
    if os.path.exists(os.path.join(root, "train_images")) and os.path.exists(
        os.path.join(root, "test_images")
    ):
        INPUT_FOLDER = root
        break
    sub = os.path.join(root, "aptos2019-blindness-detection")
    if (
        os.path.exists(sub)
        and os.path.exists(os.path.join(sub, "train_images"))
        and os.path.exists(os.path.join(sub, "test_images"))
    ):
        INPUT_FOLDER = sub
        break

if INPUT_FOLDER is None:
    INPUT_FOLDER = "."

TEST_IMAGES_DIR = os.path.join(INPUT_FOLDER, "test_images")
TRAIN_IMAGES_DIR = os.path.join(INPUT_FOLDER, "train_images")

print("INPUT_FOLDER:", INPUT_FOLDER)
print("Has test_images:", os.path.exists(TEST_IMAGES_DIR))
print("Has train_images:", os.path.exists(TRAIN_IMAGES_DIR))

WEIGHTS_PATH = None
preferred_names = [
    "dense-multi-second-0.9038.h5",
    "dense-multi-second-0.9038.hdf5",
    "dense-multi-second-0.9038.weights.h5",
    "model.h5",
    "weights.h5",
]

candidate_paths = [
    "../input/densenetmulti/dense-multi-second-0.9038.h5",
    "../input/densenetmulti/dense-multi-second-0.9038.hdf5",
    os.path.join(INPUT_FOLDER, "dense-multi-second-0.9038.h5"),
]
for p in candidate_paths:
    if os.path.exists(p):
        WEIGHTS_PATH = p
        break


def _walk_find_weight(search_root, names):
    for root, _, files in os.walk(search_root):
        for n in names:
            if n in files:
                return os.path.join(root, n)
    return None


if WEIGHTS_PATH is None and os.path.exists("../input"):
    WEIGHTS_PATH = _walk_find_weight("../input", preferred_names)

if WEIGHTS_PATH is None and os.path.exists("/kaggle/input"):
    WEIGHTS_PATH = _walk_find_weight("/kaggle/input", preferred_names)

if WEIGHTS_PATH is None and os.path.exists("/kaggle/data/input"):
    WEIGHTS_PATH = _walk_find_weight("/kaggle/data/input", preferred_names)

print("WEIGHTS_PATH:", WEIGHTS_PATH)



## === cell 1
test_csv_path = os.path.join(INPUT_FOLDER, "test.csv")
if not os.path.exists(test_csv_path):
    alt = "/kaggle/input/aptos2019-blindness-detection/test.csv"
    alt2 = "../input/aptos2019-blindness-detection/test.csv"
    alt3 = "/kaggle/data/aptos2019-blindness-detection/test.csv"
    alt4 = "/kaggle/data/input/aptos2019-blindness-detection/test.csv"
    for a in (alt, alt2, alt3, alt4):
        if os.path.exists(a):
            test_csv_path = a
            INPUT_FOLDER = os.path.dirname(test_csv_path)
            TEST_IMAGES_DIR = os.path.join(INPUT_FOLDER, "test_images")
            TRAIN_IMAGES_DIR = os.path.join(INPUT_FOLDER, "train_images")
            break

test_df = pd.read_csv(test_csv_path)
test_df["id_code_png"] = test_df["id_code"].astype(str) + ".png"
print(test_df.head())
print("Using TEST_IMAGES_DIR:", TEST_IMAGES_DIR)

if not os.path.exists(TEST_IMAGES_DIR):
    nested = os.path.join(INPUT_FOLDER, "aptos2019-blindness-detection", "test_images")
    if os.path.exists(nested):
        TEST_IMAGES_DIR = nested
print("Final TEST_IMAGES_DIR:", TEST_IMAGES_DIR)




## === cell 2
def label_convert(y_val):
    y_val = y_val.astype(int).sum(axis=1) - 1
    return y_val


def crop(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    thresh = 5

    rowMaxes = gray.max(axis=1)
    top = 0
    while top < len(rowMaxes) and rowMaxes[top] < thresh:
        top += 1
    bottom = len(rowMaxes) - 1
    while bottom >= 0 and rowMaxes[bottom] < thresh:
        bottom -= 1

    if top >= bottom:
        return bgr

    middleRow = gray[int((bottom - top) / 2)]
    left = 0
    while left < len(middleRow) and middleRow[left] < thresh:
        left += 1
    right = len(middleRow) - 1
    while right >= 0 and middleRow[right] < thresh:
        right -= 1

    height = bottom - top
    width = right - left

    if height < 100 or width < 100 or left >= right:
        return bgr

    return bgr[top:bottom, left:right]


def colourfulEyes(bgr, weight=4, gamma=15):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)
    y = cv2.addWeighted(y, weight, cv2.GaussianBlur(y, (0, 0), gamma), -weight, 128)
    ycc_modified = cv2.merge((y, cr, cb))
    modified = cv2.cvtColor(ycc_modified, cv2.COLOR_YCrCb2BGR)
    return modified


def processImageBgrToRgb(bgr):
    modified = crop(bgr)
    modified = cv2.resize(modified, (IMG_DIM, IMG_DIM))
    modified = colourfulEyes(modified)
    modified = cv2.cvtColor(modified, cv2.COLOR_BGR2RGB)
    return modified




## === cell 3
def dataGenerator(jitter=0.1):
    datagen = image.ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True and (jitter > 0.01),
        vertical_flip=True and (jitter > 0.01),
        rotation_range=int(800 * jitter),
        brightness_range=[1 - jitter, 1 + jitter],
        channel_shift_range=int(30 * jitter),
        zoom_range=[(1 - jitter), (1 + jitter / 2)],
        fill_mode="reflect",
    )
    return datagen




## === cell 4
def create_model():
    model = Sequential()
    model.add(
        DenseNet121(
            weights=None,
            include_top=False,
            input_shape=(IMG_DIM, IMG_DIM, CHANNEL_SIZE),
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

if WEIGHTS_PATH is not None:
    print("Loading pretrained weights from:", WEIGHTS_PATH)
    model.load_weights(WEIGHTS_PATH)
else:
    print(
        "Pretrained weights not found. Falling back to training on train.csv/train_images "
        "using the same model/loss so a valid submission can still score reasonably."
    )

    train_csv_path = os.path.join(INPUT_FOLDER, "train.csv")
    if not os.path.exists(train_csv_path):
        for a in (
            "/kaggle/input/aptos2019-blindness-detection/train.csv",
            "../input/aptos2019-blindness-detection/train.csv",
            "/kaggle/data/aptos2019-blindness-detection/train.csv",
            "/kaggle/data/input/aptos2019-blindness-detection/train.csv",
        ):
            if os.path.exists(a):
                train_csv_path = a
                break
    train_df = pd.read_csv(train_csv_path)
    train_df["id_code_png"] = train_df["id_code"].astype(str) + ".png"

    if not os.path.exists(TRAIN_IMAGES_DIR):
        nested = os.path.join(
            INPUT_FOLDER, "aptos2019-blindness-detection", "train_images"
        )
        if os.path.exists(nested):
            TRAIN_IMAGES_DIR = nested
    print("Using TRAIN_IMAGES_DIR:", TRAIN_IMAGES_DIR)

    def to_multi_label(diag_int):
        y = np.zeros((NUM_CLASSES,), dtype=np.float32)
        y[: int(diag_int) + 1] = 1.0
        return y

    y_multi = np.stack(
        [to_multi_label(v) for v in train_df["diagnosis"].values], axis=0
    )

    idx = np.arange(len(train_df))
    rng = np.random.RandomState(SEED)
    rng.shuffle(idx)
    split = int(0.9 * len(idx))
    tr_idx, va_idx = idx[:split], idx[split:]

    tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
    va_df = train_df.iloc[va_idx].reset_index(drop=True)
    y_tr = y_multi[tr_idx]
    y_va = y_multi[va_idx]

    def _load_batch(df_slice):
        n = len(df_slice)
        x = np.empty((n, IMG_DIM, IMG_DIM, 3), dtype=np.float32)
        for i, fn in enumerate(df_slice["id_code_png"].values):
            p = os.path.join(TRAIN_IMAGES_DIR, fn)
            bgr = cv2.imread(p)
            if bgr is None:
                x[i] = np.full((IMG_DIM, IMG_DIM, 3), 128.0, dtype=np.float32)
            else:
                x[i] = processImageBgrToRgb(bgr).astype(np.float32)
        return x

    train_datagen = dataGenerator(0.1)
    val_datagen = image.ImageDataGenerator(rescale=1.0 / 255)

    x_tr = _load_batch(tr_df)
    x_va = _load_batch(va_df)

    train_flow = train_datagen.flow(
        x_tr, y_tr, batch_size=BATCH_SIZE, shuffle=True, seed=SEED
    )
    val_flow = val_datagen.flow(x_va, y_va, batch_size=BATCH_SIZE, shuffle=False)

    model.fit(
        train_flow,
        validation_data=val_flow,
        epochs=3,
        verbose=1,
    )

gc.collect()



## === cell 5
block_size = 500
total = test_df.index.size

y_pred_list = np.zeros(total, dtype=int)

tta_num = 7
tta_jitter = 0.03
base_tta_seed = SEED

max_workers = min(8, (os.cpu_count() or 4))


def _read_and_process_one(path):
    bgr = cv2.imread(path)
    if bgr is None:
        return np.full((IMG_DIM, IMG_DIM, 3), 128.0, dtype=np.float32)
    try:
        return processImageBgrToRgb(bgr).astype(np.float32, copy=False)
    except Exception:
        return np.full((IMG_DIM, IMG_DIM, 3), 128.0, dtype=np.float32)


def _apply_tta_batch_uint8like(x01, rng: np.random.RandomState, jitter: float):
    n, h, w, _ = x01.shape
    out = x01.copy()

    if jitter > 0.01:
        do_h = rng.rand(n) < 0.5
        do_v = rng.rand(n) < 0.5
        for i in range(n):
            if do_h[i]:
                out[i] = out[i, :, ::-1, :]
            if do_v[i]:
                out[i] = out[i, ::-1, :, :]

    rot_deg = int(800 * jitter)
    if rot_deg > 0:
        angles = rng.uniform(-rot_deg, rot_deg, size=n).astype(np.float32)
        center = (w * 0.5, h * 0.5)
        for i in range(n):
            M = cv2.getRotationMatrix2D(center, float(angles[i]), 1.0)
            out[i] = cv2.warpAffine(
                out[i],
                M,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,  # reflect-like to match "reflect"
            )

    zmin, zmax = (1.0 - jitter), (1.0 + jitter / 2.0)
    if jitter > 0.0:
        scales = rng.uniform(zmin, zmax, size=n).astype(np.float32)
        for i in range(n):
            s = float(scales[i])
            if abs(s - 1.0) < 1e-7:
                continue
            new_w = max(1, int(round(w * s)))
            new_h = max(1, int(round(h * s)))
            resized = cv2.resize(out[i], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            if s >= 1.0:
                x0 = (new_w - w) // 2
                y0 = (new_h - h) // 2
                out[i] = resized[y0 : y0 + h, x0 : x0 + w, :]
            else:
                pad_x = w - new_w
                pad_y = h - new_h
                left = pad_x // 2
                right = pad_x - left
                top = pad_y // 2
                bottom = pad_y - top
                out[i] = cv2.copyMakeBorder(
                    resized,
                    top,
                    bottom,
                    left,
                    right,
                    borderType=cv2.BORDER_REFLECT_101,
                )

    if jitter > 0.0:
        bmin, bmax = (1.0 - jitter), (1.0 + jitter)
        br = rng.uniform(bmin, bmax, size=n).astype(np.float32)
        out *= br[:, None, None, None]

    cshift = int(30 * jitter)
    if cshift > 0:
        shifts = rng.uniform(-cshift, cshift, size=(n, 3)).astype(np.float32) / 255.0
        out += shifts[:, None, None, :]

    np.clip(out, 0.0, 1.0, out=out)
    return out


def _predict_batches(x01):
    n = x01.shape[0]
    preds = np.empty((n, NUM_CLASSES), dtype=np.float32)
    for s in range(0, n, BATCH_SIZE):
        e = min(s + BATCH_SIZE, n)
        preds[s:e] = model.predict_on_batch(x01[s:e])
    return preds


max_block = min(block_size, total)
img_block = np.empty((max_block, IMG_DIM, IMG_DIM, 3), dtype=np.float32)  # 0..255
pred_block = np.empty((max_block, tta_num, NUM_CLASSES), dtype=np.float32)

with ThreadPoolExecutor(max_workers=max_workers) as ex:
    for start in range(0, total, block_size):
        end = min(start + block_size, total)
        filenames = test_df.iloc[start:end]["id_code_png"].values
        n_block = end - start

        for i, arr in enumerate(
            ex.map(
                _read_and_process_one,
                (os.path.join(TEST_IMAGES_DIR, fn) for fn in filenames),
            )
        ):
            img_block[i] = arr

        base_x01 = img_block[:n_block] * (1.0 / 255.0)

        for t in range(tta_num):
            rng = np.random.RandomState(int(base_tta_seed + t))
            aug_x01 = _apply_tta_batch_uint8like(base_x01, rng, tta_jitter)
            pred_block[:n_block, t, :] = _predict_batches(aug_x01)

        predictions = np.median(pred_block[:n_block], axis=1)
        y_pred_list[start:end] = label_convert(predictions > 0.5)

        if (start // block_size) % 2 == 1:
            gc.collect()

        print(f"{start} - {end} finished")



## === cell 6
submission = pd.DataFrame(
    {"id_code": test_df["id_code"].values, "diagnosis": y_pred_list.astype(int)}
)
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
