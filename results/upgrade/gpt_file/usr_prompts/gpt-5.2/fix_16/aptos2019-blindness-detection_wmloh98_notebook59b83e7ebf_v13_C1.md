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

3.9

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

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import random
import numpy as np
import pandas as pd
import tensorflow as tf

import cv2

from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    if hasattr(tf.config, "experimental") and hasattr(
        tf.config.experimental, "enable_op_determinism"
    ):
        tf.config.experimental.enable_op_determinism()
except Exception:
    pass

DATA_PATH = "/kaggle/input/aptos2019-blindness-detection/"
TRAIN_CSV = os.path.join(DATA_PATH, "train.csv")
TEST_CSV = os.path.join(DATA_PATH, "test.csv")
TRAIN_DIR = os.path.join(DATA_PATH, "train_images")
TEST_DIR = os.path.join(DATA_PATH, "test_images")

DIM_X = 256
DIM_Y = 256
BATCH_SIZE = 32

CACHE_DIR = "/kaggle/working/preprocessed_cache_256"
TRAIN_CACHE_DIR = os.path.join(CACHE_DIR, "train")
TEST_CACHE_DIR = os.path.join(CACHE_DIR, "test")
os.makedirs(TRAIN_CACHE_DIR, exist_ok=True)
os.makedirs(TEST_CACHE_DIR, exist_ok=True)

try:
    cv2.setNumThreads(max(1, min(4, (os.cpu_count() or 2) // 2)))
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass


def crop_image_from_gray(img, tol=7):
    if img is None:
        return img
    if img.ndim == 2:
        mask = img > tol
    elif img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray > tol
    else:
        return img

    if not mask.any():
        return img

    ys = np.where(mask.any(axis=1))[0]
    xs = np.where(mask.any(axis=0))[0]
    y0, y1 = ys[0], ys[-1] + 1
    x0, x1 = xs[0], xs[-1] + 1
    return img[y0:y1, x0:x1]


def circle_crop_v2(img):
    height, width, depth = img.shape
    largest_side = int(max(height, width))
    if height != largest_side or width != largest_side:
        img = cv2.resize(
            img, (largest_side, largest_side), interpolation=cv2.INTER_LINEAR
        )

    h, w, _ = img.shape
    x = w // 2
    y = h // 2
    r = int(min(x, y))

    circle_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circle_mask, (x, y), r, 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_mask)

    img = crop_image_from_gray(img)
    return img


def preprocess_image(image, sigmaX=25, DIM_X=256, DIM_Y=256):
    """
    Keep core preprocessing identical; return float32 so downstream rescale is safe.
    """
    if image is None:
        return np.zeros((DIM_Y, DIM_X, 3), dtype=np.float32)
    image = image.astype(np.uint8, copy=False)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = circle_crop_v2(image)
    image = cv2.resize(image, (DIM_X, DIM_Y), interpolation=cv2.INTER_LINEAR)
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image.astype(np.float32, copy=False)


def _cache_path(cache_dir, filename):
    base = os.path.splitext(os.path.basename(filename))[0]
    return os.path.join(cache_dir, base + ".npy")


def _preprocess_and_save_one(args):
    inp, outp, dim_x, dim_y = args
    if os.path.isfile(outp):
        return 1
    img_bgr = cv2.imread(inp, cv2.IMREAD_COLOR)
    arr = preprocess_image(img_bgr, DIM_X=dim_x, DIM_Y=dim_y)  # float32 HWC
    np.save(outp, arr, allow_pickle=False)
    return 1


def ensure_preprocessed_cache(df, image_dir, cache_dir, filename_col="filename"):
    import concurrent.futures

    fns = df[filename_col].astype(str).to_numpy()
    in_paths = [os.path.join(image_dir, fn) for fn in fns]
    out_paths = [_cache_path(cache_dir, fn) for fn in fns]

    missing = [
        (inp, outp, DIM_X, DIM_Y)
        for inp, outp in zip(in_paths, out_paths)
        if not os.path.isfile(outp)
    ]
    if not missing:
        return

    cpu = os.cpu_count() or 2
    max_workers = min(8, max(2, cpu // 2))
    chunksize = 32

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
        for _ in ex.map(_preprocess_and_save_one, missing, chunksize=chunksize):
            pass




## === cell 1
class FixedDropout(tf.keras.layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        symbolic_shape = K.shape(inputs)
        noise_shape = [
            symbolic_shape[axis] if shape is None else shape
            for axis, shape in enumerate(self.noise_shape)
        ]
        return tuple(noise_shape)




## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

train_df["filename"] = train_df["id_code"].astype(str) + ".png"
test_df["filename"] = test_df["id_code"].astype(str) + ".png"

ensure_preprocessed_cache(train_df, TRAIN_DIR, TRAIN_CACHE_DIR, filename_col="filename")
ensure_preprocessed_cache(test_df, TEST_DIR, TEST_CACHE_DIR, filename_col="filename")


class CachedNpySequence(tf.keras.utils.Sequence):
    def __init__(
        self,
        df,
        cache_dir,
        datagen,
        x_col="filename",
        y_col=None,
        batch_size=32,
        shuffle=False,
        seed=42,
        rescale=None,
        num_classes=None,
        lru_cache_size=512,
    ):
        self.df = df.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.datagen = datagen
        self.x_col = x_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.indices = np.arange(len(self.df), dtype=np.int32)
        self.rescale = rescale
        self.num_classes = num_classes

        self._x = self.df[self.x_col].astype(str).to_numpy()
        self._cache_paths = [_cache_path(self.cache_dir, fn) for fn in self._x]
        self._y = (
            None if self.y_col is None else self.df[self.y_col].to_numpy(dtype=np.int64)
        )

        from collections import OrderedDict

        self._lru = OrderedDict()
        self._lru_max = int(lru_cache_size)

        self._do_aug = False
        if self.datagen is not None:
            self._do_aug = bool(
                self.datagen.rotation_range
                or self.datagen.zoom_range
                or self.datagen.horizontal_flip
                or self.datagen.vertical_flip
                or self.datagen.width_shift_range
                or self.datagen.height_shift_range
                or self.datagen.shear_range
                or getattr(self.datagen, "brightness_range", None)
                or self.datagen.channel_shift_range
            )

        self.on_epoch_end()

    def __len__(self):
        return (len(self.indices) + self.batch_size - 1) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def _load_npy(self, path):
        arr = self._lru.get(path, None)
        if arr is not None:
            self._lru.move_to_end(path, last=True)
            return arr

        arr = np.load(path, allow_pickle=False)
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)
        arr = arr.astype(np.float32, copy=False)

        self._lru[path] = arr
        if len(self._lru) > self._lru_max:
            self._lru.popitem(last=False)
        return arr

    def __getitem__(self, idx):
        sl = slice(idx * self.batch_size, (idx + 1) * self.batch_size)
        batch_ids = self.indices[sl]
        bs = len(batch_ids)

        bx = np.empty((bs, DIM_Y, DIM_X, 3), dtype=np.float32)
        by = None if self._y is None else self._y[batch_ids]

        for i, j in enumerate(batch_ids):
            bx[i] = self._load_npy(self._cache_paths[int(j)])

        if self.rescale is not None:
            bx *= self.rescale

        if self._do_aug:
            for i in range(bs):
                params = self.datagen.get_random_transform(bx[i].shape, seed=None)
                bx[i] = self.datagen.apply_transform(bx[i], params)

        if by is None:
            return bx
        return bx, by


train_datagen = ImageDataGenerator(
    preprocessing_function=None,
    rescale=None,  # applied in sequence to avoid double-scaling
    validation_split=0.15,
    rotation_range=15,
    horizontal_flip=True,
    vertical_flip=False,
    zoom_range=0.1,
)

valid_datagen = ImageDataGenerator(
    preprocessing_function=None,
    rescale=None,
    validation_split=0.15,
)

train_df_shuf = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
num = len(train_df_shuf)
split = int(np.floor(0.15 * num))
valid_df = train_df_shuf.iloc[:split].copy()
train_df_split = train_df_shuf.iloc[split:].copy()

train_gen = CachedNpySequence(
    df=train_df_split,
    cache_dir=TRAIN_CACHE_DIR,
    datagen=train_datagen,
    x_col="filename",
    y_col="diagnosis",
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
    rescale=(1.0 / 255.0),
)

valid_gen = CachedNpySequence(
    df=valid_df,
    cache_dir=TRAIN_CACHE_DIR,
    datagen=valid_datagen,
    x_col="filename",
    y_col="diagnosis",
    batch_size=BATCH_SIZE,
    shuffle=False,
    seed=SEED,
    rescale=(1.0 / 255.0),
)

num_classes = 5
model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(DIM_Y, DIM_X, 3)),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

EPOCHS = 6

history = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=EPOCHS,
    verbose=1,
)



## === cell 3
test_pred_gen = CachedNpySequence(
    df=test_df,
    cache_dir=TEST_CACHE_DIR,
    datagen=None,
    x_col="filename",
    y_col=None,
    batch_size=64,
    shuffle=False,
    seed=SEED,
    rescale=(1.0 / 255.0),
)

pred_proba = model.predict(
    test_pred_gen,
    verbose=1,
)

pred = np.argmax(pred_proba, axis=1).astype(np.int64)
pred = np.clip(pred, 0, 4).astype(int)



## === cell 4
submission_df = test_df[["id_code"]].copy()
submission_df["diagnosis"] = pred
submission_df.to_csv("submission.csv", index=False)

print(submission_df.head())
print("Saved submission.csv with shape:", submission_df.shape)

from collections import Counter

cnt = Counter(submission_df["diagnosis"].tolist())
print(cnt)
