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

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as _pb_ver
except Exception:
    _pb_ver = None


def _version_tuple(v):
    try:
        return tuple(int(x) for x in v.split(".")[:3])
    except Exception:
        return (0, 0, 0)


if _pb_ver is None or _version_tuple(_pb_ver) >= (5, 0, 0):
    import sys
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<5"])

import gc
import random
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf

print("TF version:", tf.__version__)

SEED = 1337
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(
        max(1, (os.cpu_count() or 2) // 2)
    )
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass



## === cell 1
"""
    Config
"""
IMG_SIZE = 224
BATCH_SIZE = 16  # keep training semantics; for test inference we'll use a larger batch to reduce TF call overhead.


def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = (img > tol).astype(np.uint8)
        if mask.max() == 0:
            return img
        x, y, w, h = cv2.boundingRect(mask)
        return img[y : y + h, x : x + w]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = (gray_img > tol).astype(np.uint8)
        if mask.max() == 0:
            return img
        x, y, w, h = cv2.boundingRect(mask)
        return img[y : y + h, x : x + w]
    return img


def load_ben_color(image, sigmaX=10):
    """
    OpenCV reads BGR; convert to RGB, Ben Graham preprocessing, resize.
    Return float32 processed with EfficientNet's preprocess_input.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image).astype("uint8")
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)

    image = image.astype("float32")
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image


"""
    Preprocessing for ImageDataGenerator since ImageDataGenerator reads images in rgb mode, while opencv in bgr
"""


def preprocessing(image, sigmaX=10):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = crop_image_from_gray(image).astype("uint8")
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)

    image = image.astype("float32")
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image




## === cell 2
def _pick_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of the expected paths exist: {paths}")


BASE_DIR = _pick_existing(
    "/kaggle/input/aptos2019-blindness-detection",
    "/kaggle/data/aptos2019-blindness-detection",
)

TEST_CSV_PATH = _pick_existing(
    os.path.join(BASE_DIR, "test.csv"),
)

TRAIN_CSV_PATH = _pick_existing(
    os.path.join(BASE_DIR, "train.csv"),
)

TEST_IMG_DIR = _pick_existing(
    os.path.join(BASE_DIR, "test_images"),
)

TRAIN_IMG_DIR = _pick_existing(
    os.path.join(BASE_DIR, "train_images"),
)

base_model = None

print("BASE_DIR:", BASE_DIR)
print("TRAIN_CSV_PATH:", TRAIN_CSV_PATH)
print("TEST_CSV_PATH:", TEST_CSV_PATH)
print("TRAIN_IMG_DIR:", TRAIN_IMG_DIR)
print("TEST_IMG_DIR:", TEST_IMG_DIR)




## === cell 3
def build_head_layers():
    flatten_layer = tf.keras.layers.Flatten()
    dense_layer_1 = tf.keras.layers.Dense(4096, activation="relu")
    Dropout_1 = tf.keras.layers.Dropout(0.6)
    dense_layer_2 = tf.keras.layers.Dense(2048, activation="relu")
    Dropout_2 = tf.keras.layers.Dropout(0.5)
    dense_layer_3 = tf.keras.layers.Dense(1024, activation="relu")
    Dropout_3 = tf.keras.layers.Dropout(0.3)
    dense_layer_4 = tf.keras.layers.Dense(512, activation="relu")
    prediction_layer = tf.keras.layers.Dense(5, activation="softmax")
    return [
        flatten_layer,
        dense_layer_1,
        Dropout_1,
        dense_layer_2,
        Dropout_2,
        dense_layer_3,
        Dropout_3,
        dense_layer_4,
        prediction_layer,
    ]




## === cell 4
base_model = tf.keras.applications.efficientnet.EfficientNetB0(
    include_top=False, weights=None, input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

model = tf.keras.models.Sequential([base_model] + build_head_layers())




## === cell 5
def _find_weights_file():
    direct_candidates = [
        "../input/eff-b0-model/eff_b0_model_224.h5",
        "/kaggle/input/eff-b0-model/eff_b0_model_224.h5",
    ]
    for c in direct_candidates:
        if os.path.exists(c):
            return c
    return None


weights_path = _find_weights_file()
USED_FALLBACK_CALIBRATION = False

if weights_path is not None:
    model.load_weights(weights_path)
    print(f"Loaded competition weights: {weights_path}")
else:
    print(
        "WARNING: Could not find eff_b0_model_224.h5 (trained weights not available).\n"
        "Falling back to EfficientNetB0 ImageNet weights + minimal calibration on train.csv.\n"
        "This preserves architecture/loss/loop but will score substantially below the target."
    )
    base_model_imagenet = tf.keras.applications.efficientnet.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    model = tf.keras.models.Sequential([base_model_imagenet] + build_head_layers())
    USED_FALLBACK_CALIBRATION = True

model.trainable = False




## === cell 6
def _make_dataset(df, img_dir, shuffle=False):
    paths = (img_dir + "/" + df["id_code"].astype(str).values + ".png").astype(str)

    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    if shuffle:
        path_ds = path_ds.shuffle(
            buffer_size=min(len(df), 1024), seed=SEED, reshuffle_each_iteration=False
        )

    def _load_np(pbytes):
        p = pbytes.decode("utf-8")
        img = cv2.imread(p)
        if img is None:
            x = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        else:
            x = load_ben_color(img)
        return x

    def _load_tf(p):
        x = tf.numpy_function(_load_np, [p], Tout=tf.float32)
        x.set_shape((IMG_SIZE, IMG_SIZE, 3))
        return x

    x_ds = path_ds.map(_load_tf, num_parallel_calls=tf.data.AUTOTUNE)

    if "diagnosis" in df.columns:
        y = df["diagnosis"].astype(np.int64).values
        y = tf.one_hot(y, depth=5)
        y_ds = tf.data.Dataset.from_tensor_slices(tf.cast(y, tf.float32))
        ds = tf.data.Dataset.zip((x_ds, y_ds))
    else:
        ds = x_ds

    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds


def _quadratic_weighted_kappa(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    assert y_true.shape == y_pred.shape

    idx = y_true * n_classes + y_pred
    O = (
        np.bincount(idx, minlength=n_classes * n_classes)
        .reshape(n_classes, n_classes)
        .astype(np.float64)
    )

    act_hist = np.bincount(y_true, minlength=n_classes).astype(np.float64)
    pred_hist = np.bincount(y_pred, minlength=n_classes).astype(np.float64)
    E = np.outer(act_hist, pred_hist)
    Esum = E.sum()
    if Esum != 0:
        E = E / Esum * O.sum()

    r = np.arange(n_classes, dtype=np.float64)
    W = (r[:, None] - r[None, :]) ** 2 / ((n_classes - 1) ** 2)

    num = (W * O).sum()
    den = (W * E).sum()
    return 1.0 - num / den if den != 0 else 0.0


def _fit_thresholds(y_true_int, y_score_cont):
    y_true_int = np.asarray(y_true_int, dtype=np.int64)
    y_score_cont = np.asarray(y_score_cont, dtype=np.float64)

    qs = np.quantile(y_score_cont, [0.2, 0.4, 0.6, 0.8])
    thr = qs.astype(np.float64)

    def _apply(th):
        return np.digitize(y_score_cont, th, right=False)

    best = _quadratic_weighted_kappa(y_true_int, _apply(thr))
    for _ in range(12):
        improved = False
        for k in range(4):
            base = thr[k]
            steps = np.array(
                [-0.25, -0.15, -0.08, -0.03, 0.03, 0.08, 0.15, 0.25], dtype=np.float64
            )
            candidates = base + steps
            for c in candidates:
                th2 = thr.copy()
                th2[k] = c
                th2 = np.sort(th2)
                q = _quadratic_weighted_kappa(y_true_int, _apply(th2))
                if q > best:
                    best = q
                    thr = th2
                    improved = True
        if not improved:
            break
    return thr, best


train_df = pd.read_csv(TRAIN_CSV_PATH)
test_df = pd.read_csv(TEST_CSV_PATH)

if "id_code" not in test_df.columns:
    raise ValueError(
        f"Expected 'id_code' column in {TEST_CSV_PATH}, got {test_df.columns.tolist()}"
    )
if not {"id_code", "diagnosis"}.issubset(train_df.columns):
    raise ValueError(
        f"Expected columns ['id_code','diagnosis'] in {TRAIN_CSV_PATH}, got {train_df.columns.tolist()}"
    )

default_thresholds = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64)
thresholds = default_thresholds.copy()

idx = np.arange(len(train_df))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)

n_val = int(0.15 * len(train_df))
val_idx = idx[:n_val]
trn_idx = idx[n_val:]

trn_df = train_df.iloc[trn_idx].reset_index(drop=True)
val_df = train_df.iloc[val_idx].reset_index(drop=True)

if USED_FALLBACK_CALIBRATION:
    backbone = model.layers[0]
    backbone.trainable = True

    for lyr in backbone.layers[:-30]:
        lyr.trainable = False
    for lyr in model.layers[1:]:
        lyr.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    trn_ds = _make_dataset(trn_df, TRAIN_IMG_DIR, shuffle=True)
    val_ds = _make_dataset(val_df, TRAIN_IMG_DIR, shuffle=False)

    model.fit(trn_ds, validation_data=val_ds, epochs=6, verbose=2)

    val_probs = model.predict(val_ds.map(lambda x, y: x), verbose=0)
    val_score = (val_probs * np.arange(5, dtype=np.float32)[None, :]).sum(axis=1)
    thresholds, best_qwk = _fit_thresholds(val_df["diagnosis"].values, val_score)
    print(
        "Fitted thresholds (fallback, OOF val):",
        thresholds,
        "Val QWK (proxy):",
        best_qwk,
    )

    model.trainable = False
else:
    max_val = 768
    if len(val_df) > max_val:
        val_df = val_df.iloc[:max_val].reset_index(drop=True)

    val_ds = _make_dataset(val_df, TRAIN_IMG_DIR, shuffle=False)
    val_probs = model.predict(val_ds.map(lambda x, y: x), verbose=0)
    val_score = (val_probs * np.arange(5, dtype=np.float32)[None, :]).sum(axis=1)
    thresholds, best_qwk = _fit_thresholds(val_df["diagnosis"].values, val_score)
    print(
        "Fitted thresholds (competition weights, OOF val):",
        thresholds,
        "Val QWK (proxy):",
        best_qwk,
    )


def _valid_thresholds(th):
    th = np.asarray(th, dtype=np.float64)
    if th.shape != (4,):
        return False
    if not np.all(np.isfinite(th)):
        return False
    if not np.all(np.diff(th) > 1e-6):
        return False
    return True


if not _valid_thresholds(thresholds):
    print(
        "WARNING: Invalid fitted thresholds; reverting to defaults:", default_thresholds
    )
    thresholds = default_thresholds.copy()

gc.collect()



## === cell 7
id_code = test_df["id_code"].astype(str).values
test_paths = (TEST_IMG_DIR + "/" + id_code + ".png").astype(str)

INFER_BATCH_SIZE = 64


def _load_np_test(pbytes):
    p = pbytes.decode("utf-8")
    img = cv2.imread(p)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {p}")
    return load_ben_color(img)


def _load_tf_test(p):
    x = tf.numpy_function(_load_np_test, [p], Tout=tf.float32)
    x.set_shape((IMG_SIZE, IMG_SIZE, 3))
    return x


test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = test_ds.map(_load_tf_test, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(INFER_BATCH_SIZE, drop_remainder=False).prefetch(
    tf.data.AUTOTUNE
)

probs = model.predict(test_ds, verbose=0)

if USED_FALLBACK_CALIBRATION:
    scores = (probs * np.arange(5, dtype=np.float32)[None, :]).sum(axis=1)
    test_prediction = np.digitize(scores, thresholds, right=False).astype(np.int64)
else:
    test_prediction = np.argmax(probs, axis=1).astype(np.int64)



## === cell 8
test_df["diagnosis"] = test_prediction.astype(np.int64)

sub = test_df[["id_code", "diagnosis"]].copy()
sub.to_csv("submission.csv", index=False)

unique, counts = np.unique(test_prediction, return_counts=True)
print("USED_FALLBACK_CALIBRATION:", USED_FALLBACK_CALIBRATION)
print("thresholds (only used in fallback):", thresholds)
print(dict(zip(unique.tolist(), counts.tolist())))
print("Wrote submission.csv with shape:", sub.shape)
print("Done!")
