# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.8878958060904764

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the environment/import issues that currently prevent TensorFlow/Keras and EfficientNet from loading, by using `tf.keras` (avoids the protobuf `MessageFactory` crash) and by switching to the built-in `tf.keras.applications.EfficientNetB3` (removes the missing `efficientnet` package dependency). I keep the same model structure (EfficientNetB3 backbone → GAP → Dense(1) with relu cap at 4) and the same inference loop, but make the preprocessing consistent with the provided `load_ben_color()` (previously defined but not used), which is a minimal logic correction that should substantially improve score vs the current broken/uncalibrated pipeline. I also make weight loading robust to the Kaggle filesystem (try likely paths; if not found, continue rather than crash), and ensure we always write a valid `submission.csv` with the correct columns. Finally, I add a safe fallback rounding to 0–4 if coefficients aren’t fitted (your code currently never fits them), while preserving the current behavior (fixed thresholds) by default.'
- What this solution (achieved 0.0) has done: 'We fix the TensorFlow/protobuf crash that prevents the notebook from running by setting a safe protobuf implementation *before* importing TensorFlow (this is the root cause of the `MessageFactory` error). We also make the input path resolution robust (prefer `/kaggle/input/...` first, but fall back to `../input/...`) so image/CSV loading doesn’t silently fail in different Kaggle directory layouts. Finally, we keep your exact model/inference/rounding logic, but add a clear hard-fail if the test image directory is missing (to avoid generating an all-zeros submission that scores ~0.0), ensuring a valid `submission.csv` is produced from real predictions.'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow/protobuf `MessageFactory` crash by setting both protobuf env vars *before* importing TensorFlow and by falling back to a compatible TF import path if needed, without changing your model/inference logic. I also make the EfficientNet import robust across TF versions (some TF 2.x builds don’t expose all EfficientNet variants at that path), while keeping EfficientNetB3 as the chosen backbone. To avoid a guaranteed ~0.0 score from random weights, I additionally load ImageNet weights when your custom `model_b3.h5` isn’t found (same architecture, just sensible initialization). Finally, I ensure we always read IDs from `test.csv` (not `sample_submission.csv`) and still output a valid `submission.csv` with the required columns.'
- What this solution (achieved -0.03024) has done: 'We fix the TensorFlow/protobuf `MessageFactory` crash by forcing the pure-Python protobuf runtime before TensorFlow import (and doing it at the very top, in the same cell) and by avoiding any accidental earlier TF import. We also ensure the script doesn’t silently run with uninitialized/random head weights when custom weights aren’t found: we keep your exact EfficientNetB3→GAP→Dense(1, relu capped at 4) architecture, but compile and load ImageNet weights into the backbone as you intended for a sane fallback. Finally, we make inference preprocessing consistent and stable (Ben preprocessing + EfficientNet `preprocess_input`), and ensure the submission rows align exactly with `test.csv` and always write a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'We fix the TensorFlow import crash by setting protobuf to the Python implementation before any TF-related import and by falling back safely if determinism isn’t available, so the pipeline can start. Then we fix the `OptimizedRounder` failure by enforcing monotonic thresholds during optimization (sorting the candidate bins inside the loss), which removes the `np.digitize` “bins must be monotonic” error without changing the overall rounding approach. Finally, we make submission generation robust by defining a safe default threshold set if fitting fails for any reason, ensuring `coefficients` always exists and a valid `submission.csv` is written.'
- What this solution (achieved 0.0) has done: 'We fix the TensorFlow import crash (`MessageFactory` missing `GetPrototype`) by forcing a compatible protobuf runtime and, if needed, safely downgrading protobuf at runtime *before* importing TensorFlow (this is the root blocker preventing any real model inference and causing the 0.0 score). We also make OpenCV optional by providing a Pillow-based image loader fallback (Kaggle images can still be read without cv2), keeping the same Ben-style preprocessing and EfficientNet preprocessing semantics. Finally, we keep your model and rounding logic intact, but harden the prediction pipeline so it never silently produces all-zero images/predictions due to missing readers/paths, ensuring a valid `submission.csv` is always written from real predictions.'

# 9. Code solution

## === cell 0
import os, sys, subprocess

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as _pb_ver

    def _ver_tuple(v):
        parts = []
        for p in v.split("."):
            try:
                parts.append(int(p))
            except Exception:
                parts.append(0)
        return tuple(parts + [0, 0, 0])

    if _ver_tuple(_pb_ver) >= (4, 21, 0):
        print(
            "Detected protobuf",
            _pb_ver,
            "-> attempting runtime pin to protobuf<4 for TF compat.",
        )
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf<4"]
        )
        import importlib
        import google.protobuf as _gp

        importlib.reload(_gp)
        from google.protobuf import __version__ as _pb_ver2

        print("protobuf after pin:", _pb_ver2)
except Exception as _e:
    print("WARNING: protobuf compatibility check/pin failed; continuing:", repr(_e))




## === cell 1
import gc

import numpy as np
import pandas as pd

try:
    import cv2  # type: ignore

    _HAS_CV2 = True
except Exception:
    cv2 = None
    _HAS_CV2 = False
    from PIL import Image

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

tf.random.set_seed(2)
np.random.seed(0)

print("TF version:", tf.__version__)
print("Has cv2:", _HAS_CV2)




## === cell 2
try:
    from tensorflow.keras.applications import (
        EfficientNetB0,
        EfficientNetB1,
        EfficientNetB2,
        EfficientNetB3,
        EfficientNetB4,
    )
except Exception:
    from tensorflow.keras.applications.efficientnet import (  # type: ignore
        EfficientNetB0,
        EfficientNetB1,
        EfficientNetB2,
        EfficientNetB3,
        EfficientNetB4,
    )

try:
    from tensorflow.keras.applications.efficientnet import (
        preprocess_input as eff_preprocess,
    )
except Exception:
    from tensorflow.keras.applications import preprocess_input as eff_preprocess




## === cell 3
"""
Preprocessing using Ben Graham's method.
Keep core logic; add PIL fallback if cv2 is unavailable.
"""


def crop_image_from_gray(img, tol=7):
    if img is None:
        return None
    if img.ndim == 2:
        mask = img > tol
        if not mask.any():
            return img
        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        return img[y0:y1, x0:x1]
    elif img.ndim == 3:
        if _HAS_CV2:
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray_img = (
                0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
            ).astype(img.dtype)
        mask = gray_img > tol
        if not mask.any():
            return img
        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        return img[y0:y1, x0:x1, :]
    return img


def load_ben_color(image, IMG_SIZE, sigmaX=10):
    image = crop_image_from_gray(image)
    if image is None:
        return None
    if _HAS_CV2:
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        image = cv2.addWeighted(
            image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128
        )
        return image
    else:
        im = Image.fromarray(image.astype(np.uint8))
        im = im.resize((IMG_SIZE, IMG_SIZE), resample=Image.BILINEAR)
        arr = np.asarray(im).astype(np.float32)

        try:
            from PIL import ImageFilter

            blurred = im.filter(ImageFilter.GaussianBlur(radius=float(sigmaX)))
            blur_arr = np.asarray(blurred).astype(np.float32)
            arr = 4.0 * arr + (-4.0) * blur_arr + 128.0
        except Exception:
            arr = arr  # fallback: no blur
        return np.clip(arr, 0, 255).astype(np.float32)




## === cell 4
"""
Define model (unchanged core logic).
"""


def output_relu(x):
    return K.relu(x, max_value=4)


def get_model(version, IMG_SIZE, weights=None):
    base_model = None
    if version == 0:
        base_model = EfficientNetB0(
            weights=weights, include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
    elif version == 1:
        base_model = EfficientNetB1(
            weights=weights, include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
    elif version == 2:
        base_model = EfficientNetB2(
            weights=weights, include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
    elif version == 3:
        base_model = EfficientNetB3(
            weights=weights, include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
    elif version == 4:
        base_model = EfficientNetB4(
            weights=weights, include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
    else:
        return None

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1, activation=output_relu, kernel_initializer="he_normal")(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model




## === cell 5
"""
Initialize model.
"""
IMG_SIZE = 300

model = get_model(3, IMG_SIZE, weights=None)

weight_candidates = [
    "/kaggle/input/pretrained-weights/model_b3.h5",
    "../input/pretrained-weights/model_b3.h5",
    "/kaggle/input/aptos2019-blindness-detection/model_b3.h5",
    "../input/aptos2019-blindness-detection/model_b3.h5",
]
loaded = False
for wpath in weight_candidates:
    if os.path.exists(wpath):
        model.load_weights(wpath)
        print("Loaded weights from:", wpath)
        loaded = True
        break

if not loaded:
    print(
        "WARNING: Pretrained weights not found. Falling back to ImageNet backbone weights."
    )
    model = get_model(3, IMG_SIZE, weights="imagenet")

model.compile(optimizer="adam", loss="mse")


@tf.function(reduce_retracing=True)
def _infer_fn(x):
    return model(x, training=False)




## === cell 6
"""
Optimized Rounder (core logic preserved).
"""
import scipy as sp
from functools import partial
from sklearn import metrics


class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _mse_loss(self, coef, X, y):
        bins = np.sort(np.asarray(coef, dtype=np.float32))
        X_p = np.digitize(X, bins, right=False).astype(np.float32)
        ll = metrics.mean_squared_error(y, X_p)
        return ll

    def fit(self, X, y):
        loss_partial = partial(self._mse_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(
            loss_partial, initial_coef, method="nelder-mead"
        )

    def predict(self, X, coef):
        bins = np.sort(np.asarray(coef, dtype=np.float32))
        return np.digitize(X, bins, right=False).astype(np.float32)

    def coefficients(self):
        return self.coef_["x"]




## === cell 7
INPUT_CANDIDATES = [
    "/kaggle/input/aptos2019-blindness-detection",
    "../input/aptos2019-blindness-detection",
    "/kaggle/data/aptos2019-blindness-detection",
]
INPUT_DIR = None
for p in INPUT_CANDIDATES:
    if os.path.exists(p):
        INPUT_DIR = p
        break
if INPUT_DIR is None:
    raise FileNotFoundError(
        f"Could not find aptos2019-blindness-detection in {INPUT_CANDIDATES}"
    )

train_csv_path = os.path.join(INPUT_DIR, "train.csv")
train_img_dir = os.path.join(INPUT_DIR, "train_images")
test_csv_path = os.path.join(INPUT_DIR, "test.csv")
test_img_dir = os.path.join(INPUT_DIR, "test_images")

for req_path, is_dir in [
    (train_csv_path, False),
    (test_csv_path, False),
    (train_img_dir, True),
    (test_img_dir, True),
]:
    if is_dir and not os.path.isdir(req_path):
        raise FileNotFoundError(f"Missing directory at: {req_path}")
    if (not is_dir) and (not os.path.exists(req_path)):
        raise FileNotFoundError(f"Missing file at: {req_path}")

print("Using INPUT_DIR:", INPUT_DIR)




## === cell 8
"""
Batched prediction with tf.data + py_function preprocessing.
"""

BATCH_SIZE = 32  # keep as originally intended


def _as_py_str(x):
    if isinstance(x, tf.Tensor):
        x = x.numpy()
    if isinstance(x, (np.bytes_, bytes, bytearray)):
        return bytes(x).decode("utf-8")
    return str(x)


def _read_image_rgb(path):
    if _HAS_CV2:
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                return np.asarray(im)
        except Exception:
            return None


def _py_preprocess_one(idv_bytes, img_dir_bytes, img_size):
    idv = _as_py_str(idv_bytes)
    img_dir = _as_py_str(img_dir_bytes)
    img_size = int(img_size)

    img_path = os.path.join(img_dir, "{}.png".format(idv))

    img_rgb = _read_image_rgb(img_path)
    if img_rgb is None:
        return np.zeros((img_size, img_size, 3), dtype=np.float32)

    img_proc = load_ben_color(img_rgb, img_size)
    if img_proc is None:
        return np.zeros((img_size, img_size, 3), dtype=np.float32)

    return img_proc.astype(np.float32, copy=False)


def batched_predict(ids, img_dir, batch_size):
    ids = np.asarray(ids, dtype=object)
    img_dir_bytes = tf.constant(img_dir.encode("utf-8"))
    img_size = int(IMG_SIZE)

    ds = tf.data.Dataset.from_tensor_slices(ids.astype("S"))
    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)

    def _map_fn(idv):
        x = tf.py_function(
            func=_py_preprocess_one,
            inp=[idv, img_dir_bytes, tf.constant(img_size, dtype=tf.int32)],
            Tout=tf.float32,
        )
        x.set_shape((img_size, img_size, 3))
        return x

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    preds = np.empty((len(ids),), dtype=np.float32)
    offset = 0
    for xb in ds:
        xb = eff_preprocess(xb)
        yb = _infer_fn(xb)
        yb = tf.reshape(yb, (-1,))
        y_np = yb.numpy().astype(np.float32, copy=False)
        preds[offset : offset + y_np.shape[0]] = y_np
        offset += y_np.shape[0]

    return preds


train_df = pd.read_csv(train_csv_path)
if not {"id_code", "diagnosis"}.issubset(train_df.columns):
    raise ValueError("train.csv must contain id_code and diagnosis columns")

train_ids = train_df["id_code"].astype(str).values
train_y = train_df["diagnosis"].astype(np.int64).values

cache_pred_path = os.path.join("/kaggle/working", f"train_pred_b3_{IMG_SIZE}.npy")
cache_y_path = os.path.join("/kaggle/working", f"train_y_{IMG_SIZE}.npy")

if os.path.exists(cache_pred_path) and os.path.exists(cache_y_path):
    train_pred = np.load(cache_pred_path)
    cached_y = np.load(cache_y_path)
    if (
        train_pred.shape[0] == train_ids.shape[0]
        and cached_y.shape[0] == train_y.shape[0]
    ):
        train_y = cached_y.astype(np.int64, copy=False)
        print("Loaded cached train predictions:", cache_pred_path)
    else:
        train_pred = batched_predict(train_ids, train_img_dir, BATCH_SIZE)
        np.save(cache_pred_path, train_pred)
        np.save(cache_y_path, train_y)
else:
    train_pred = batched_predict(train_ids, train_img_dir, BATCH_SIZE)
    np.save(cache_pred_path, train_pred)
    np.save(cache_y_path, train_y)

default_coefficients = [0.5, 1.5, 2.5, 3.5]
optR = OptimizedRounder()
try:
    optR.fit(train_pred, train_y)
    coefficients = optR.coefficients()
    coefficients = np.sort(np.asarray(coefficients, dtype="float32")).tolist()
    print("Fitted coefficients:", coefficients)
except Exception as e:
    print("WARNING: OptimizedRounder.fit failed, using default coefficients.", repr(e))
    coefficients = default_coefficients

gc.collect()




## === cell 9
test_df = pd.read_csv(test_csv_path)
if "id_code" not in test_df.columns:
    raise ValueError("test.csv must contain id_code column")

id_code = test_df["id_code"].astype(str).values
test_prediction = batched_predict(id_code, test_img_dir, BATCH_SIZE)

test_prediction_round = optR.predict(test_prediction, coefficients)
test_prediction_round = np.clip(np.rint(test_prediction_round), 0, 4).astype("uint8")

sub_df = pd.DataFrame(
    {"id_code": id_code, "diagnosis": test_prediction_round.astype(int)}
)
sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)

unique, counts = np.unique(test_prediction_round, return_counts=True)
tmp = dict(zip(unique.tolist(), counts.tolist()))
print("Prediction label distribution:", tmp)
print("Wrote", sub_path, "with shape:", sub_df.shape)
print("Submission head:\n", sub_df.head())
