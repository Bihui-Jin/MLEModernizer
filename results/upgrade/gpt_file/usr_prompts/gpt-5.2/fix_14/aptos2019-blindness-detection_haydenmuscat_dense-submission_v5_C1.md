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

0.8360317782738592

# 6. Current score

0.07501

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the initial import/runtime crash by removing the incompatible `tensorflow.set_random_seed` import and instead setting seeds via `numpy`/`random`/`tf.random.set_seed`, which avoids the protobuf `MessageFactory.GetPrototype` error in this environment. I also make the input-path discovery robust (choose the existing APTOS folder and the DenseNet weights folder if present) so `INPUT_FOLDER` is always defined and the images/weights load correctly. Next, I restore missing imports (so `Sequential` is defined) and update deprecated Keras calls (`predict_generator`, `lr` argument) to current equivalents without changing the model or prediction logic. Finally, I ensure `test_df` is created before it’s used and that a valid `submission.csv` with `id_code,diagnosis` is always written.'
- What this solution (achieved 0.07501) has done: 'I remove the biggest avoidable runtime cost: caching **all** preprocessed training images in RAM (which forces thousands of expensive OpenCV transforms up front and can trigger heavy GC/memory pressure). Instead, the training `Sequence` load/process images on-demand per batch (same preprocessing), while enabling Keras multiprocessing prefetch (`workers`/`use_multiprocessing`/`max_queue_size`) to overlap CPU image work with GPU/TF execution. For test-time TTA, I keep the exact same augmentation logic but avoid the per-block `np.stack` and extra copies by filling the preallocated block buffer directly from threaded results. These changes preserve the exact model, loss, epochs, TTA count, preprocessing, and evaluation semantics—only removing redundant work and improving pipeline overlap.'

# 9. Code solution

## === cell 0
import os
import gc
import random
import numpy as np
import pandas as pd
import cv2

import matplotlib.pyplot as plt  # noqa: F401

from sklearn.model_selection import train_test_split  # noqa: F401
from sklearn.metrics import cohen_kappa_score, confusion_matrix  # noqa: F401


IMG_DIM = 256
BATCH_SIZE = 32
CHANNEL_SIZE = 3
NUM_CLASSES = 5

SEED = 1337
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

try:
    cv2.setNumThreads(max(1, (os.cpu_count() or 4)))
except Exception:
    pass

BASE_INPUT = "../input"
CANDIDATE_APTOS = [
    os.path.join(BASE_INPUT, "aptos2019-blindness-detection"),
    os.path.join(
        BASE_INPUT, "aptos2019-blindness-detection", "aptos2019-blindness-detection"
    ),
]
INPUT_FOLDER = None
for p in CANDIDATE_APTOS:
    if os.path.exists(os.path.join(p, "train.csv")) and os.path.exists(
        os.path.join(p, "test.csv")
    ):
        INPUT_FOLDER = p.rstrip("/") + "/"
        break

if INPUT_FOLDER is None:
    raise FileNotFoundError(
        "Could not locate aptos2019-blindness-detection folder under ../input"
    )

TEST_IMAGES_DIR = os.path.join(INPUT_FOLDER, "test_images") + "/"
TRAIN_IMAGES_DIR = os.path.join(INPUT_FOLDER, "train_images") + "/"

print("Resolved INPUT_FOLDER:", INPUT_FOLDER)
print("TRAIN_IMAGES_DIR exists:", os.path.exists(TRAIN_IMAGES_DIR))
print("TEST_IMAGES_DIR exists:", os.path.exists(TEST_IMAGES_DIR))
print("List ../input:", os.listdir(BASE_INPUT)[:50])



## === cell 1
test_df = pd.read_csv(os.path.join(INPUT_FOLDER, "test.csv"))
test_df["id_code"] = test_df["id_code"].astype(str)
test_df["filename"] = test_df["id_code"].apply(lambda x: x + ".png")
test_df.head()




## === cell 2
def label_convert(y_val):
    y_val = y_val.astype(int).sum(axis=1) - 1
    y_val = np.clip(y_val, 0, 4)
    return y_val


def crop(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    thresh = 5

    rowMaxes = gray.max(axis=1)
    rows = np.flatnonzero(rowMaxes >= thresh)
    if rows.size == 0:
        return bgr
    top = int(rows[0])
    bottom = int(rows[-1])
    if top >= bottom:
        return bgr

    middleRow = gray[int((bottom - top) / 2)]
    cols = np.flatnonzero(middleRow >= thresh)
    if cols.size == 0:
        return bgr
    left = int(cols[0])
    right = int(cols[-1])

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


def load_image_rgb_from_path(path):
    bgr = cv2.imread(path)
    if bgr is None:
        raise ValueError(f"cv2.imread returned None for {path}")
    return processImageBgrToRgb(bgr)




## === cell 3
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.applications import DenseNet121
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
    from tensorflow.keras.optimizers import Adam

    try:
        tf.random.set_seed(SEED)
    except Exception:
        pass

    try:
        tf.config.optimizer.set_jit(True)
    except Exception:
        pass

    try:
        tf.config.run_functions_eagerly(False)
    except Exception:
        pass

    try:
        tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF choose
        tf.config.threading.set_inter_op_parallelism_threads(0)
    except Exception:
        pass

except Exception as e:
    raise RuntimeError(
        "TensorFlow failed to import in this environment. "
        "This notebook requires TensorFlow/Keras to build the model."
    ) from e


def create_model():
    model = Sequential()
    model.add(
        DenseNet121(
            weights="imagenet",
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

train_df = pd.read_csv(os.path.join(INPUT_FOLDER, "train.csv"))
train_df["id_code"] = train_df["id_code"].astype(str)
train_df["filename"] = train_df["id_code"].apply(lambda x: x + ".png")


def diagnosis_to_multilabel(d):
    d = int(d)
    return np.array([1, d >= 1, d >= 2, d >= 3, d >= 4], dtype=np.float32)


y_multi = np.stack(
    [diagnosis_to_multilabel(d) for d in train_df["diagnosis"].values], axis=0
)

idx = np.arange(len(train_df))
np.random.shuffle(idx)
split = int(0.9 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]

train_df_tr = train_df.iloc[tr_idx].reset_index(drop=True)
train_df_va = train_df.iloc[va_idx].reset_index(drop=True)
y_tr = y_multi[tr_idx]
y_va = y_multi[va_idx]


def _load_train_img_u8_from_filename(filename):
    path = os.path.join(TRAIN_IMAGES_DIR, filename)
    try:
        img = load_image_rgb_from_path(path).astype(np.uint8, copy=False)
    except Exception:
        img = np.full((IMG_DIM, IMG_DIM, 3), 128, dtype=np.uint8)
    return img


class SimpleSequence(tf.keras.utils.Sequence):
    def __init__(self, df, y, images_dir, batch_size=32, shuffle=True):
        self.df = df
        self.y = y
        self.images_dir = images_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.filenames = df["filename"].values
        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx_batch):
        sl = slice(idx_batch * self.batch_size, (idx_batch + 1) * self.batch_size)
        batch_ids = self.indexes[sl]
        bs = len(batch_ids)
        X = np.empty((bs, IMG_DIM, IMG_DIM, 3), dtype=np.float32)
        Y = self.y[batch_ids].astype(np.float32, copy=False)
        for j, ridx in enumerate(batch_ids):
            fn = self.filenames[ridx]
            img_u8 = _load_train_img_u8_from_filename(fn)
            X[j] = img_u8
        X *= 1.0 / 255.0
        return X, Y


train_seq = SimpleSequence(
    train_df_tr, y_tr, TRAIN_IMAGES_DIR, batch_size=BATCH_SIZE, shuffle=True
)
val_seq = SimpleSequence(
    train_df_va, y_va, TRAIN_IMAGES_DIR, batch_size=BATCH_SIZE, shuffle=False
)

_workers = min(4, max(1, (os.cpu_count() or 2) - 1))
model.fit(
    train_seq,
    validation_data=val_seq,
    epochs=1,
    verbose=1,
    workers=_workers,
    use_multiprocessing=(_workers > 1),
    max_queue_size=10,
)

gc.collect()




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4
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




## === cell 5
import concurrent.futures as _fut

_MAX_WORKERS = min(32, (os.cpu_count() or 4))
_TPE = _fut.ThreadPoolExecutor(max_workers=_MAX_WORKERS) if _MAX_WORKERS > 1 else None


def _precompute_tta_params(bs, h, w, jitter, seed_base):
    """Precompute all randomness and affine matrices once per (block, tta)."""
    do_flip = jitter > 0.01
    rotation_range = int(800 * jitter)  # degrees
    bright_lo, bright_hi = (1 - jitter), (1 + jitter)
    ch_shift = float(int(30 * jitter))
    zoom_lo, zoom_hi = (1 - jitter), (1 + jitter / 2.0)

    rng = np.random.RandomState(seed_base)

    if do_flip:
        hflip = rng.rand(bs) < 0.5
        vflip = rng.rand(bs) < 0.5
    else:
        hflip = None
        vflip = None

    if rotation_range > 0:
        angles = rng.uniform(-rotation_range, rotation_range, size=bs).astype(
            np.float32
        )
    else:
        angles = None

    if zoom_lo != 1.0 or zoom_hi != 1.0:
        zooms = rng.uniform(zoom_lo, zoom_hi, size=bs).astype(np.float32)
    else:
        zooms = None

    if jitter > 0:
        brights = rng.uniform(bright_lo, bright_hi, size=bs).astype(np.float32)
    else:
        brights = None

    if ch_shift != 0.0:
        shifts = rng.uniform(-ch_shift, ch_shift, size=(bs, 3)).astype(np.float32)
    else:
        shifts = None

    if angles is not None or zooms is not None:
        if angles is None:
            angles = np.zeros(bs, dtype=np.float32)
        if zooms is None:
            zooms = np.ones(bs, dtype=np.float32)
        cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
        Ms = np.empty((bs, 2, 3), dtype=np.float32)
        for i in range(bs):
            Ms[i] = cv2.getRotationMatrix2D((cx, cy), float(angles[i]), float(zooms[i]))
    else:
        Ms = None

    return do_flip, hflip, vflip, Ms, brights, shifts


def _augment_batch_opencv_u8_to_f32_with_params(
    imgs_u8, do_flip, hflip, vflip, Ms, brights, shifts, out_f32, tmp_u8
):
    bs, h, w, _ = imgs_u8.shape

    def _warp_one(i):
        img = imgs_u8[i]
        if do_flip:
            if hflip[i]:
                img = cv2.flip(img, 1)
            if vflip[i]:
                img = cv2.flip(img, 0)

        if Ms is not None:
            cv2.warpAffine(
                img,
                Ms[i],
                (w, h),
                dst=tmp_u8[i],
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
        else:
            tmp_u8[i, :, :, :] = img

    if bs >= 16 and _TPE is not None:
        list(_TPE.map(_warp_one, range(bs)))
    else:
        for i in range(bs):
            _warp_one(i)

    out_f32[:bs] = tmp_u8[:bs].astype(np.float32)
    if brights is not None:
        out_f32[:bs] *= brights[:bs, None, None, None]
    if shifts is not None:
        out_f32[:bs] += shifts[:bs, None, None, :]
    np.clip(out_f32[:bs], 0.0, 255.0, out=out_f32[:bs])
    out_f32[:bs] *= 1.0 / 255.0
    return out_f32


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None, IMG_DIM, IMG_DIM, 3], dtype=tf.float32)
    ],
    reduce_retracing=True,
)
def _predict_batch_tensor(x_tensor):
    return model(x_tensor, training=False)


def _load_one_test_u8(filename):
    try:
        img = load_image_rgb_from_path(os.path.join(TEST_IMAGES_DIR, filename))
        return img.astype(np.uint8, copy=False)
    except Exception:
        return np.full((IMG_DIM, IMG_DIM, 3), 128, dtype=np.uint8)


block_size = 128
total = test_df.index.size
y_pred_list = np.zeros(total, dtype=int)

tta_num = 7
jitter = 0.03

test_filenames = test_df["filename"].values

x_aug_buf = np.empty((block_size, IMG_DIM, IMG_DIM, 3), dtype=np.float32)
tmp_u8_buf = np.empty((block_size, IMG_DIM, IMG_DIM, 3), dtype=np.uint8)
img_block_u8_buf = np.empty((block_size, IMG_DIM, IMG_DIM, 3), dtype=np.uint8)

for start in range(0, total, block_size):
    end = min(start + block_size, total)
    bs = end - start
    fns_block = test_filenames[start:end]

    if _TPE is not None and bs >= 16:
        for i, img in enumerate(_TPE.map(_load_one_test_u8, fns_block)):
            img_block_u8_buf[i] = img
    else:
        for i, filename in enumerate(fns_block):
            img_block_u8_buf[i] = _load_one_test_u8(filename)

    img_block_u8 = img_block_u8_buf[:bs]
    prediction_lists = np.empty((bs, tta_num, NUM_CLASSES), dtype=np.float32)

    for t in range(tta_num):
        seed_base = SEED + (t + 1) * 1000003 + start * 1009
        do_flip, hflip, vflip, Ms, brights, shifts = _precompute_tta_params(
            bs, IMG_DIM, IMG_DIM, jitter=jitter, seed_base=seed_base
        )

        _augment_batch_opencv_u8_to_f32_with_params(
            img_block_u8,
            do_flip=do_flip,
            hflip=hflip,
            vflip=vflip,
            Ms=Ms,
            brights=brights,
            shifts=shifts,
            out_f32=x_aug_buf,
            tmp_u8=tmp_u8_buf,
        )

        preds = _predict_batch_tensor(x_aug_buf[:bs]).numpy()
        prediction_lists[:, t, :] = preds

    predictions = np.median(prediction_lists, axis=1)
    y_pred_list[start:end] = label_convert(predictions > 0.5)

    print(f"{start} - {end} finished")

if _TPE is not None:
    _TPE.shutdown(wait=True)

gc.collect()



## === cell 6
sub_df = pd.read_csv(os.path.join(INPUT_FOLDER, "test.csv"))
sub_df["id_code"] = sub_df["id_code"].astype(str)
sub_df["diagnosis"] = y_pred_list.astype(int)

sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print("Wrote:", sub_path)
print(sub_df.head())
print("diagnosis value counts:\n", sub_df["diagnosis"].value_counts().sort_index())
