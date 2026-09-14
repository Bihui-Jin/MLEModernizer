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

# 5. Target score

0.8452750103900721

# 6. Current score

-0.12028

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.01069) has done: 'I remove the `pip install tensorflow-addons` (it triggers the protobuf `MessageFactory.GetPrototype` crash) and also remove the unused `tensorflow_addons` import so the notebook can import TensorFlow/Keras cleanly. Then I fix the cell ordering/numbering and ensure `tf`, `np`, and `pd` are defined before they’re used, so the model can be built and inference can run. Finally, I fix test-time preprocessing to match training-time scaling (the current code feeds 0–255 into a model that expects 0–1), and I make the script robust to Kaggle’s input path by auto-detecting whether data lives under `/kaggle/input/...` or `/kaggle/data/...`, ensuring a valid `submission.csv` is always produced.'
- What this solution (achieved 0.0) has done: 'You’re hitting the known TensorFlow/protobuf `MessageFactory.GetPrototype` crash at import-time, so the first fix is to force TensorFlow to use the pure-Python protobuf implementation *before* importing `tensorflow`. Next, your model currently uses `weights=None` and often can’t find the external `.h5` weights, which makes predictions essentially random (hence the negative kappa); the minimal, core-logic-preserving fix is to use ImageNet pretrained weights for the EfficientNet backbone when the external weights aren’t available. Finally, I keep your preprocessing consistent and ensure the script always writes `submission.csv` with the required `id_code,diagnosis` columns.'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation early and (crucially) disabling the C++ protobuf fast-path before importing TensorFlow. Then I fix the dataset path resolver so it correctly finds the existing `sample_submission.csv` and `test_images/` directory without accidentally appending a duplicated `aptos2019-blindness-detection` subfolder. Finally, I keep your model and preprocessing logic the same, but make submission generation robust by reading `test.csv` (or falling back to `sample_submission.csv`) and writing a valid `submission.csv` with `id_code,diagnosis`.'
- What this solution (achieved 0.10147) has done: 'The TensorFlow import crash is happening before any modeling code runs, so the main fix is to set protobuf-related environment variables in the same process *before* any TensorFlow/protobuf-dependent import and to add a safe fallback that removes the problematic `google.protobuf` C++ implementation from `sys.modules` if it was preloaded. Then, to move your score up from 0.0 toward the target, the smallest legitimate improvement is to ensure the EfficientNet backbone preprocessing matches what it expects: keep your Ben Graham style enhancement but apply EfficientNet’s `preprocess_input` (instead of simple `/255.0`) consistently at test-time. Finally, the script keeps the same architecture/weights-loading logic and still writes a valid `submission.csv` with `id_code,diagnosis`.'
- What this solution (achieved -0.00631) has done: 'You’re still crashing at `import tensorflow as tf` due to an incompatibility between the protobuf runtime in this Kaggle image and the TensorFlow build (`MessageFactory.GetPrototype` missing). The minimal, execution-unblocking fix is to force a protobuf version that provides `GetPrototype` (protobuf 3.20.x) before importing TensorFlow; this is a targeted environment fix and doesn’t change your model logic. After TensorFlow imports cleanly, the rest of your pipeline can run unchanged and produce `submission.csv`. This should also materially improve the score versus essentially-random predictions from a non-running/partially-running pipeline, while keeping the same EfficientNetB2 + head and preprocessing semantics intact.'
- What this solution (achieved 0.02995) has done: 'Your current score is far below the target, so we should make the smallest changes that legitimately improve quadratic weighted kappa without changing the model architecture or training loop (there is no training here). The biggest issue is that you are using classification argmax directly, while QWK is typically improved by converting class-probabilities into an “ordinal” scalar and then mapping to 0–4 via tuned thresholds. I add a tiny validation-based threshold calibration step using the provided `train.csv` and `train_images/` (no new packages), keeping the same preprocessing and model, then apply those thresholds to test predictions. This preserves the core inference logic (EfficientNetB2 + same head + same weights) and only changes the post-processing to better match the metric, and still writes a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current score is far below the target, so we keep your same EfficientNetB2 model/inference logic but make the threshold calibration more reliable (less noisy) and better aligned to QWK. Concretely, we (1) tune thresholds on a larger, stratified subset of the training set (instead of a small random sample) and (2) run threshold tuning in a simple out-of-fold way to reduce overfitting to the calibration subset, then average thresholds. Finally, we make inference much faster (and deterministic) by batching predictions for both tuning and test, without changing any model layers or loss—this also lets us afford the larger calibration size within the time limit. These are minimal, metric-aligned post-processing changes and should move QWK substantially upward from ~0.03 toward your target band.'
- What this solution (achieved -0.12028) has done: 'I fix the runtime `IndexError` in `thresholds_from_train_distribution` by ensuring it always returns exactly 4 monotonically-increasing thresholds, even when quantile targets contain duplicates or edge cases that can make `np.quantile` return fewer usable cut points. This is a minimal, score-aligned bug fix that restores the intended “train-label-distribution-based thresholding” fallback path when external weights aren’t found. I also add a tiny safety fallback so that if quantile computation yields invalid/degenerate thresholds, we revert to the default `[0.5, 1.5, 2.5, 3.5]` rather than crashing. No model/architecture/training logic is changed; this only ensures end-to-end execution and a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import subprocess
import sys


def ensure_protobuf_320():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver

        if not pb_ver.startswith("3.20."):
            raise RuntimeError(f"protobuf version {pb_ver} is not 3.20.*")
    except Exception:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==3.20.3"]
        )
        import importlib

        for m in list(sys.modules.keys()):
            if m.startswith("google.protobuf"):
                sys.modules.pop(m, None)
        importlib.invalidate_caches()


ensure_protobuf_320()



## === cell 1
import gc

import cv2
import numpy as np
import pandas as pd

import tensorflow as tf

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

print("TF version:", tf.__version__)



## === cell 2
"""
    Config
"""
IMG_SIZE = 224
BATCH_SIZE = 16


def crop_image_from_gray(img, tol=7):
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
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image).astype("uint8")
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image


"""
Preprocessing for ImageDataGenerator since ImageDataGenerator reads images in rgb mode, while opencv in bgr
"""


def preprocessing(image, sigmaX=10):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = crop_image_from_gray(image).astype("uint8")
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image.astype("float32") / 255.0


eff_preprocess = tf.keras.applications.efficientnet.preprocess_input




## === cell 3
def resolve_comp_dir():
    candidates = [
        "/kaggle/input/aptos2019-blindness-detection",
        "/kaggle/data/aptos2019-blindness-detection",
        "../input/aptos2019-blindness-detection",
        "../data/aptos2019-blindness-detection",
        "/kaggle/input/aptos2019-blindness-detection/aptos2019-blindness-detection",
        "/kaggle/data/aptos2019-blindness-detection/aptos2019-blindness-detection",
    ]

    def looks_like_comp_dir(p):
        return (
            os.path.exists(os.path.join(p, "sample_submission.csv"))
            and os.path.exists(os.path.join(p, "test.csv"))
            and os.path.exists(os.path.join(p, "test_images"))
        )

    for c in candidates:
        if os.path.exists(c) and looks_like_comp_dir(c):
            return c

    roots = ["/kaggle/input", "/kaggle/data", "../input", "../data"]
    for r in roots:
        if not os.path.exists(r):
            continue
        for sub in ["aptos2019-blindness-detection"]:
            p = os.path.join(r, sub)
            if os.path.exists(p) and looks_like_comp_dir(p):
                return p
            p2 = os.path.join(r, sub, "aptos2019-blindness-detection")
            if os.path.exists(p2) and looks_like_comp_dir(p2):
                return p2

    raise FileNotFoundError(
        "Could not find aptos2019-blindness-detection directory containing sample_submission.csv/test.csv/test_images."
    )


COMP_DIR = resolve_comp_dir()
TEST_IMG_DIR = os.path.join(COMP_DIR, "test_images")
TRAIN_IMG_DIR = os.path.join(COMP_DIR, "train_images")
SAMPLE_SUB_PATH = os.path.join(COMP_DIR, "sample_submission.csv")
TEST_CSV_PATH = os.path.join(COMP_DIR, "test.csv")
TRAIN_CSV_PATH = os.path.join(COMP_DIR, "train.csv")

print("COMP_DIR:", COMP_DIR)
print("TEST_IMG_DIR exists:", os.path.exists(TEST_IMG_DIR))
print("TRAIN_IMG_DIR exists:", os.path.exists(TRAIN_IMG_DIR))
print("SAMPLE_SUB_PATH exists:", os.path.exists(SAMPLE_SUB_PATH))
print("TEST_CSV_PATH exists:", os.path.exists(TEST_CSV_PATH))
print("TRAIN_CSV_PATH exists:", os.path.exists(TRAIN_CSV_PATH))



## === cell 4
base_model = tf.keras.applications.efficientnet.EfficientNetB2(
    include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

flatten_layer = tf.keras.layers.Flatten()
dense_layer_1 = tf.keras.layers.Dense(4096, activation="relu")
Dropout_1 = tf.keras.layers.Dropout(0.6)
dense_layer_2 = tf.keras.layers.Dense(2048, activation="relu")
Dropout_2 = tf.keras.layers.Dropout(0.5)
dense_layer_3 = tf.keras.layers.Dense(1024, activation="relu")
Dropout_3 = tf.keras.layers.Dropout(0.3)
dense_layer_4 = tf.keras.layers.Dense(512, activation="relu")
prediction_layer = tf.keras.layers.Dense(5, activation="softmax")

model = tf.keras.models.Sequential(
    [
        base_model,
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
)

_ = model(tf.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32))
model.summary()



## === cell 5
WEIGHTS_CANDIDATES = [
    "../input/eff-balance-b1-model-224/eff_balance_b1_model_224.h5",
    "/kaggle/input/eff-balance-b1-model-224/eff_balance_b1_model_224.h5",
]
weights_loaded = False
for wpath in WEIGHTS_CANDIDATES:
    if os.path.exists(wpath):
        model.load_weights(wpath)
        weights_loaded = True
        print("Loaded weights:", wpath)
        break

if not weights_loaded:
    print(
        "WARNING: External pretrained weights not found; using ImageNet backbone weights only."
    )




## === cell 6
def quadratic_weighted_kappa(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    assert y_true.shape == y_pred.shape

    O = np.zeros((n_classes, n_classes), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        if 0 <= a < n_classes and 0 <= b < n_classes:
            O[a, b] += 1.0

    act_hist = O.sum(axis=1)
    pred_hist = O.sum(axis=0)
    E = np.outer(act_hist, pred_hist)
    if E.sum() > 0:
        E = E * (O.sum() / E.sum())

    W = np.zeros((n_classes, n_classes), dtype=np.float64)
    for i in range(n_classes):
        for j in range(n_classes):
            W[i, j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)

    denom = (W * E).sum()
    if denom == 0:
        return 0.0
    return 1.0 - (W * O).sum() / denom


def scores_to_labels(scores, thresholds):
    t0, t1, t2, t3 = thresholds
    scores = np.asarray(scores, dtype=np.float32)
    return np.digitize(scores, bins=[t0, t1, t2, t3]).astype(np.int64)


def tune_thresholds_bruteforce(scores, y_true, init=None, step=0.05, iters=2):
    scores = np.asarray(scores, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=np.int64)

    if init is None:
        init = [0.5, 1.5, 2.5, 3.5]
    th = np.array(init, dtype=np.float32)

    best_k = quadratic_weighted_kappa(y_true, scores_to_labels(scores, th))
    for _ in range(iters):
        for i in range(4):
            candidates = []
            for delta in np.arange(-5, 6) * step:
                cand = th.copy()
                cand[i] = cand[i] + delta
                if not (0.0 <= cand[0] < cand[1] < cand[2] < cand[3] <= 4.0):
                    continue
                k = quadratic_weighted_kappa(y_true, scores_to_labels(scores, cand))
                candidates.append((k, cand))
            if candidates:
                kmax, thmax = max(candidates, key=lambda x: x[0])
                if kmax >= best_k:
                    best_k = kmax
                    th = thmax
        step = step / 2.0
    return th.tolist(), float(best_k)


def predict_scores_for_ids(img_dir, img_ids, batch_size=BATCH_SIZE):
    n = len(img_ids)
    scores = np.empty(n, dtype=np.float32)
    cls = np.arange(5, dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        X = np.empty((end - start, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        for k, img_id in enumerate(img_ids[start:end]):
            img_path = os.path.join(img_dir, f"{img_id}.png")
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Could not read image: {img_path}")
            img = load_ben_color(img).astype("float32")
            img = eff_preprocess(img)
            X[k] = img
        p = model.predict(X, verbose=0).astype(np.float32)  # (B,5)
        scores[start:end] = (p * cls.reshape(1, -1)).sum(axis=1)
    return scores


def stratified_sample_indices(y, n_total, seed=SEED):
    y = np.asarray(y, dtype=np.int64)
    rng = np.random.RandomState(seed)
    idx_by_c = [np.where(y == c)[0] for c in range(5)]
    for c in range(5):
        rng.shuffle(idx_by_c[c])

    counts = np.array([len(v) for v in idx_by_c], dtype=np.int64)
    probs = counts / max(1, counts.sum())
    n_per_c = np.maximum(1, np.floor(probs * n_total).astype(np.int64))
    while n_per_c.sum() < n_total:
        c = int(rng.choice(5, p=probs))
        if n_per_c[c] < counts[c]:
            n_per_c[c] += 1
    while n_per_c.sum() > n_total:
        c = int(rng.choice(5, p=probs))
        if n_per_c[c] > 1:
            n_per_c[c] -= 1

    selected = []
    for c in range(5):
        take = min(n_per_c[c], counts[c])
        selected.append(idx_by_c[c][:take])
    sel = np.concatenate(selected)
    rng.shuffle(sel)
    return sel


def thresholds_from_train_distribution(train_labels, scores):
    y = np.asarray(train_labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float32)

    if s.size < 10 or not np.all(np.isfinite(s)):
        return [0.5, 1.5, 2.5, 3.5]

    hist = np.bincount(y, minlength=5).astype(np.float64)
    if hist.sum() <= 0:
        return [0.5, 1.5, 2.5, 3.5]
    p = hist / hist.sum()
    cdf = np.cumsum(p)

    q = [float(cdf[0]), float(cdf[1]), float(cdf[2]), float(cdf[3])]
    q = [min(max(x, 1e-3), 1.0 - 1e-3) for x in q]

    for i in range(1, 4):
        if q[i] <= q[i - 1]:
            q[i] = min(1.0 - 1e-3, q[i - 1] + 1e-3)

    try:
        th_raw = np.quantile(s, q)
    except Exception:
        return [0.5, 1.5, 2.5, 3.5]

    th_raw = np.asarray(th_raw, dtype=np.float32).reshape(-1)
    if th_raw.shape[0] != 4 or (not np.all(np.isfinite(th_raw))):
        return [0.5, 1.5, 2.5, 3.5]

    th = []
    prev = 0.0
    for i in range(4):
        val = float(np.clip(th_raw[i], 0.0, 4.0))
        if i > 0 and val <= prev:
            val = min(4.0, prev + 1e-3)
        th.append(val)
        prev = val

    if not (0.0 <= th[0] < th[1] < th[2] < th[3] <= 4.0):
        return [0.5, 1.5, 2.5, 3.5]
    return th


use_thresholds = False
thresholds = [0.5, 1.5, 2.5, 3.5]  # safe default

if os.path.exists(TRAIN_CSV_PATH) and os.path.exists(TRAIN_IMG_DIR):
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    y_all = train_df["diagnosis"].astype(np.int64).values
    ids_all = train_df["id_code"].astype(str).values

    if weights_loaded:
        tune_n = min(1200, len(train_df))
        sel = stratified_sample_indices(y_all, tune_n, seed=SEED)
        tune_ids = ids_all[sel].tolist()
        tune_y = y_all[sel]

        tune_scores = predict_scores_for_ids(
            TRAIN_IMG_DIR, tune_ids, batch_size=BATCH_SIZE
        )

        rng = np.random.RandomState(SEED)
        perm = rng.permutation(tune_n)
        folds = np.array_split(perm, 3)

        th_list = []
        k_list = []
        for f in range(3):
            val_idx = folds[f]
            th_f, k_f = tune_thresholds_bruteforce(
                tune_scores[val_idx],
                tune_y[val_idx],
                init=thresholds,
                step=0.1,
                iters=2,
            )
            th_list.append(th_f)
            k_list.append(k_f)

        thresholds = np.mean(np.array(th_list, dtype=np.float32), axis=0).tolist()
        best_k = quadratic_weighted_kappa(
            tune_y, scores_to_labels(tune_scores, thresholds)
        )
        use_thresholds = True
        print("Fold tuned thresholds:", th_list)
        print("Fold kappas:", k_list)
        print("Averaged thresholds:", thresholds, "calib QWK:", best_k)
    else:
        print(
            "INFO: External weights not loaded; will use train-distribution-based thresholds on test scores."
        )
else:
    print("WARNING: train.csv/train_images not found; falling back to argmax labels.")



## === cell 7
if os.path.exists(TEST_CSV_PATH):
    test_df = pd.read_csv(TEST_CSV_PATH)
elif os.path.exists(SAMPLE_SUB_PATH):
    test_df = pd.read_csv(SAMPLE_SUB_PATH)[["id_code"]]
else:
    raise FileNotFoundError(
        "Neither test.csv nor sample_submission.csv could be found."
    )

id_code = test_df["id_code"].astype(str).tolist()

test_scores = predict_scores_for_ids(TEST_IMG_DIR, id_code, batch_size=BATCH_SIZE)

if use_thresholds:
    test_prediction = scores_to_labels(test_scores, thresholds).astype(np.int64)
else:
    if os.path.exists(TRAIN_CSV_PATH) and (not weights_loaded):
        train_df = pd.read_csv(TRAIN_CSV_PATH)
        y_all = train_df["diagnosis"].astype(np.int64).values
        thresholds = thresholds_from_train_distribution(y_all, test_scores)
        test_prediction = scores_to_labels(test_scores, thresholds).astype(np.int64)
        print("Used train-distribution thresholds:", thresholds)
    else:
        test_prediction = np.empty(len(id_code), dtype=np.int64)
        for start in range(0, len(id_code), BATCH_SIZE):
            end = min(len(id_code), start + BATCH_SIZE)
            X = np.empty((end - start, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
            for k, img_id in enumerate(id_code[start:end]):
                img_path = os.path.join(TEST_IMG_DIR, f"{img_id}.png")
                img = cv2.imread(img_path)
                if img is None:
                    raise FileNotFoundError(f"Could not read image: {img_path}")
                img = load_ben_color(img).astype("float32")
                img = eff_preprocess(img)
                X[k] = img
            pred = model.predict(X, verbose=0)
            test_prediction[start:end] = np.argmax(pred, axis=1).astype(np.int64)

sub = pd.DataFrame({"id_code": id_code, "diagnosis": test_prediction.astype(np.int64)})
sub.to_csv("submission.csv", index=False)

unique, counts = np.unique(test_prediction, return_counts=True)
print(dict(zip(unique.tolist(), counts.tolist())))
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
gc.collect()
