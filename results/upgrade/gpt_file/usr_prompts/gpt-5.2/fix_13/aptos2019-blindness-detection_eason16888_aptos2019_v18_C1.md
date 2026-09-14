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

3.10

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

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import gc
import math
import hashlib
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import DenseNet121

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

BASE_PATH = "/kaggle/input/aptos2019-blindness-detection"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TEST_CSV = os.path.join(BASE_PATH, "test.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train_images")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test_images")

print("TensorFlow:", tf.__version__)
print("Train CSV exists:", os.path.exists(TRAIN_CSV))
print("Test CSV exists:", os.path.exists(TEST_CSV))
print("Train images dir exists:", os.path.isdir(TRAIN_IMG_DIR))
print("Test images dir exists:", os.path.isdir(TEST_IMG_DIR))



## === cell 1
"""
    Config
"""
IMG_SIZE = 224
BATCH_SIZE = 16
N_CLASSES = 5

try:
    cv2.setUseOptimized(True)
except Exception:
    pass


def crop_image_from_gray(img, tol=7):
    if img is None:
        return img
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


def load_ben_color(image, sigmaX=10):
    if image is None:
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype="float32")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image.astype("float32")


def preprocessing(image, sigmaX=10):
    if image is None:
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

    if image.dtype != np.uint8:
        image_u8 = image.astype(np.uint8, copy=False)
    else:
        image_u8 = image

    gray = cv2.cvtColor(image_u8, cv2.COLOR_RGB2GRAY)
    mask = gray > 7
    if mask.any():
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        y0, y1 = int(rows[0]), int(rows[-1]) + 1
        x0, x1 = int(cols[0]), int(cols[-1]) + 1
        image_u8 = image_u8[y0:y1, x0:x1]

    bgr = cv2.cvtColor(image_u8, cv2.COLOR_RGB2BGR)
    bgr = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    bgr = cv2.addWeighted(bgr, 4, cv2.GaussianBlur(bgr, (0, 0), sigmaX), -4, 128)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype("float32") / 255.0


CACHE_DIR = "/kaggle/working/preprocessed_cache"
TRAIN_CACHE_DIR = os.path.join(CACHE_DIR, f"train_{IMG_SIZE}")
TEST_CACHE_DIR = os.path.join(CACHE_DIR, f"test_{IMG_SIZE}")
os.makedirs(TRAIN_CACHE_DIR, exist_ok=True)
os.makedirs(TEST_CACHE_DIR, exist_ok=True)


def _cache_key(path: str) -> str:
    base = os.path.basename(path)
    h = hashlib.md5(path.encode("utf-8")).hexdigest()[:10]
    return f"{os.path.splitext(base)[0]}_{h}.npy"


def _preprocess_and_cache_png_to_npy(src_path: str, dst_path: str):
    img_bgr = cv2.imread(src_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        arr = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    else:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        arr = preprocessing(img_rgb)  # float32 in [0,1]
    tmp_path = dst_path + ".tmp.npy"
    np.save(tmp_path, arr, allow_pickle=False)
    os.replace(tmp_path, dst_path)


def _cache_worker(args):
    sp, cp = args
    if not os.path.exists(cp):
        _preprocess_and_cache_png_to_npy(sp, cp)
    return 1


def ensure_cache(
    df: pd.DataFrame, img_dir: str, out_dir: str, max_workers: int = 8
) -> pd.DataFrame:
    from concurrent.futures import ProcessPoolExecutor

    df = df.copy()
    src_paths = [os.path.join(img_dir, fn) for fn in df["filename"].tolist()]
    cached_paths = [os.path.join(out_dir, _cache_key(p)) for p in src_paths]
    df["cached_path"] = cached_paths

    missing = [
        (sp, cp) for sp, cp in zip(src_paths, cached_paths) if not os.path.exists(cp)
    ]
    if missing:
        n_workers = min(max_workers, (os.cpu_count() or 2))
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                list(ex.map(_cache_worker, missing, chunksize=32))
        except Exception as e:
            print(
                "Parallel cache build failed, falling back to sequential. Error:",
                repr(e),
            )
            for t in missing:
                _cache_worker(t)
    return df




## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

train_df = train_df.copy()
test_df = test_df.copy()
train_df["filename"] = train_df["id_code"].astype(str) + ".png"
test_df["filename"] = test_df["id_code"].astype(str) + ".png"

train_df["diagnosis"] = train_df["diagnosis"].astype(int)

trn_df, val_df = train_test_split(
    train_df, test_size=0.15, random_state=SEED, stratify=train_df["diagnosis"]
)

trn_df = trn_df.copy()
val_df = val_df.copy()
trn_df["diagnosis_str"] = trn_df["diagnosis"].astype(str)
val_df["diagnosis_str"] = val_df["diagnosis"].astype(str)

trn_df = ensure_cache(trn_df, TRAIN_IMG_DIR, TRAIN_CACHE_DIR)
val_df = ensure_cache(val_df, TRAIN_IMG_DIR, TRAIN_CACHE_DIR)
test_df = ensure_cache(test_df, TEST_IMG_DIR, TEST_CACHE_DIR)

AUTO = tf.data.AUTOTUNE
MAP_PARALLEL = AUTO

data_opts = tf.data.Options()
data_opts.deterministic = True


def _load_npy_tf(path):
    def _np_load(p):
        fp = p.decode("utf-8") if isinstance(p, (bytes, bytearray)) else str(p)
        arr = np.load(fp, allow_pickle=False)
        return arr.astype(np.float32, copy=False)

    img = tf.numpy_function(_np_load, [path], Tout=tf.float32)
    img.set_shape((IMG_SIZE, IMG_SIZE, 3))
    return img


_aug_layer = keras.Sequential(
    [
        keras.layers.RandomRotation(15.0 / 360.0, fill_mode="constant", seed=SEED),
        keras.layers.RandomFlip(mode="horizontal", seed=SEED),
        keras.layers.RandomZoom(
            height_factor=(-0.10, 0.10),
            width_factor=(-0.10, 0.10),
            fill_mode="constant",
            seed=SEED,
        ),
        keras.layers.RandomBrightness(factor=0.2, value_range=(0.0, 1.0), seed=SEED),
    ],
    name="augmentation",
)


def make_train_ds(df: pd.DataFrame):
    paths = df["cached_path"].values.astype(str)
    labels = keras.utils.to_categorical(
        df["diagnosis"].values.astype(np.int32), N_CLASSES
    )
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.with_options(data_opts)
    ds = ds.shuffle(buffer_size=len(df), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: (_load_npy_tf(p), y), num_parallel_calls=MAP_PARALLEL)

    def _aug_map(x, y):
        x = _aug_layer(x, training=True)
        return x, y

    ds = ds.map(_aug_map, num_parallel_calls=MAP_PARALLEL)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTO)
    return ds


def make_eval_ds(df: pd.DataFrame, with_labels: bool):
    paths = df["cached_path"].values.astype(str)
    if with_labels:
        labels = keras.utils.to_categorical(
            df["diagnosis"].values.astype(np.int32), N_CLASSES
        )
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(lambda p, y: (_load_npy_tf(p), y), num_parallel_calls=MAP_PARALLEL)
    else:
        ds = tf.data.Dataset.from_tensor_slices(paths)
        ds = ds.map(_load_npy_tf, num_parallel_calls=MAP_PARALLEL)

    ds = ds.with_options(data_opts)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTO)
    return ds


train_ds = make_train_ds(trn_df)
val_ds = make_eval_ds(val_df, with_labels=True)
test_ds = make_eval_ds(test_df, with_labels=False)

cw = class_weight.compute_class_weight(
    class_weight="balanced", classes=np.arange(N_CLASSES), y=trn_df["diagnosis"].values
)
class_weights = {i: float(w) for i, w in enumerate(cw)}
print("Class weights:", class_weights)

STEPS_PER_EPOCH = int(math.ceil(len(trn_df) / BATCH_SIZE))
VAL_STEPS = int(math.ceil(len(val_df) / BATCH_SIZE))
TEST_STEPS = int(math.ceil(len(test_df) / BATCH_SIZE))
print(
    "steps_per_epoch:",
    STEPS_PER_EPOCH,
    "val_steps:",
    VAL_STEPS,
    "test_steps:",
    TEST_STEPS,
)



## === cell 3
LOCAL_MODEL_PATH = "/kaggle/working/model.h5"


def build_model():
    inp = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    base = DenseNet121(include_top=False, weights="imagenet", input_tensor=inp)
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    out = Dense(N_CLASSES, activation="softmax")(x)
    model = Model(inputs=inp, outputs=out)
    return model


def _fit_with_loader_kwargs(model, *args, **kwargs):
    return model.fit(*args, **kwargs)


if os.path.exists(LOCAL_MODEL_PATH):
    model = keras.models.load_model(LOCAL_MODEL_PATH)
    print("Loaded existing model:", LOCAL_MODEL_PATH)
else:
    model = build_model()

    for layer in model.layers:
        if layer.name.startswith("densenet"):
            layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    ckpt = ModelCheckpoint(
        LOCAL_MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )
    rlrop = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
    )

    EPOCHS_FROZEN = 2
    _fit_with_loader_kwargs(
        model,
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FROZEN,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VAL_STEPS,
        class_weight=class_weights,
        callbacks=[ckpt, rlrop],
        verbose=1,
    )

    for layer in model.layers:
        layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    EPOCHS_FINETUNE = 3
    _fit_with_loader_kwargs(
        model,
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINETUNE,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VAL_STEPS,
        class_weight=class_weights,
        callbacks=[ckpt, rlrop],
        verbose=1,
    )

    model = keras.models.load_model(LOCAL_MODEL_PATH)
    print("Trained and loaded best model:", LOCAL_MODEL_PATH)




## === cell 4
def _predict_with_loader_kwargs(model, *args, **kwargs):
    return model.predict(*args, **kwargs)


pred_proba = _predict_with_loader_kwargs(
    model,
    test_ds,
    steps=TEST_STEPS,
    verbose=1,
)
test_prediction = np.argmax(pred_proba, axis=1).astype("int64")

sub = pd.read_csv(SAMPLE_SUB)

if len(test_prediction) != len(sub):
    raise ValueError(
        f"Prediction length {len(test_prediction)} does not match sample_submission rows {len(sub)}"
    )

sub["diagnosis"] = test_prediction

sub_path = "submission.csv"
sub.to_csv(sub_path, index=False)

unique, counts = np.unique(test_prediction, return_counts=True)
print(dict(zip(unique.tolist(), counts.tolist())))
print("Wrote:", sub_path, "rows:", len(sub))
print(sub.head())

gc.collect()
