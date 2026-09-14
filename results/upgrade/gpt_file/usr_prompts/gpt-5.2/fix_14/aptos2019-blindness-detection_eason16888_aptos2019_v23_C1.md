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

0.7657290256664656

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the failing `tensorflow-addons` install/import (it’s not used for inference here and is triggering the protobuf `MessageFactory` error), and consolidate imports so `tf`, `pd`, etc. are always defined before later cells run. I also fix the image normalization mismatch: your model is fed raw 0–255 images while EfficientNet expects normalized inputs; we reuse your existing `preprocessing()` logic for test-time so predictions are meaningful (this should improve score vs. random). Finally, I make file paths robust to this environment by using the provided `/kaggle/data/aptos2019-blindness-detection/...` paths (with a safe fallback), ensure weights loading is optional (so the notebook still runs even if the weights file isn’t present), and always write a valid `submission.csv` with the required columns.'
- What this solution (achieved -0.15492) has done: 'The crash in your first cell is happening before your code runs due to an incompatibility between the installed protobuf runtime and TensorFlow’s internal proto usage; we can fix this safely by forcing the pure-Python protobuf implementation *before* importing TensorFlow. After that, the rest of your pipeline can run end-to-end and always write a valid `submission.csv`. To move the score up from 0.0 toward your target, the smallest legitimate improvement is to ensure EfficientNetB2 receives inputs in the expected scale and normalization: keep your existing BEN preprocessing but additionally apply `tf.keras.applications.efficientnet.preprocess_input` (this is consistent with EfficientNet evaluation semantics and doesn’t change your model architecture). I also keep your weight-loading logic but make it robust (if weights exist, they be used; otherwise it still runs and submits).'
- What this solution (achieved -0.05813) has done: 'We fix the crash happening before your code runs by forcing TensorFlow to use the pure-Python protobuf implementation *and* disabling the C++ implementation via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2` before importing TensorFlow (this directly targets the `MessageFactory.GetPrototype` error). Next, we make the EfficientNet preprocessing consistent and non-destructive by avoiding the redundant RGB↔BGR conversion mismatch (so the BEN preprocessing is applied correctly) and then applying `efficientnet.preprocess_input` exactly once in the expected scale. Finally, we keep your weight-loading logic but expand it to also search common Kaggle input directories so it actually finds the provided model file when present, which should move your score up substantially toward the target without changing the model architecture.'
- What this solution (achieved 0.02537) has done: 'We fix the TensorFlow import crash by ensuring the protobuf runtime is compatible before TensorFlow loads, using a safe monkey-patch for the missing `MessageFactory.GetPrototype` symbol (this is the direct cause of your current failure). Then we keep your model and preprocessing logic unchanged, but make the environment and paths deterministic and robust so the notebook runs end-to-end and always writes `submission.csv`. Finally, we keep the existing weight-search logic (so you can actually use the trained weights when present), which is the main lever to move the score up toward the target without changing core modeling.'
- What this solution (achieved -0.03952) has done: 'The current failure happens before your code runs because TensorFlow’s protobuf expects `MessageFactory.GetPrototype` but in this environment it’s missing; your patch only adds it to the class, while TensorFlow can be calling it on an already-created instance. I make the protobuf compatibility patch robust by adding `GetPrototype` to both the class and any existing `MessageFactory` instances (and fall back safely if protobuf internals differ), which should unblock the TensorFlow import. After that, I keep your model, preprocessing, and inference logic unchanged, only ensuring the submission is always written as `submission.csv` with the required columns. This is score-neutral in intent (it mainly fixes the crash) and allows your existing weight-loading logic (the main score driver) to actually run.'
- What this solution (achieved 0.0) has done: 'Your current score (-0.03952) is far below the target (0.7657), so we should improve predictions without changing the model architecture or training logic. The biggest score issue here is likely a preprocessing mismatch: you apply BEN-style preprocessing twice (once in `load_ben_color`, then again in `preprocessing`), which can distort images and make the trained weights (if found) behave poorly. I make inference preprocessing consistent by running BEN preprocessing exactly once (keep your existing function), then apply EfficientNet’s `preprocess_input` once in the expected scale. I also read `test.csv` (instead of `sample_submission.csv`) to guarantee the exact test ordering used by Kaggle, while still writing the required `id_code,diagnosis` submission.'
- What this solution (achieved 0.15524) has done: 'Your current 0.0 score strongly suggests the model is effectively untrained at inference time (random weights not found/loaded), so the smallest legitimate score-improving change is to reliably locate and load the provided `eff_b2_model.h5` weights. I keep your model architecture and single-image prediction loop intact, but expand the weight search to also scan `/kaggle/data` and common nested competition folders (the dataset is present there in your environment). I also add a strict check that the weights file is non-trivially sized before “accepting” it, to avoid accidentally loading an empty/placeholder file and still getting near-random predictions. Everything else (BEN preprocessing + EfficientNet preprocess + submission writing) is preserved.'
- What this solution (achieved 0.0) has done: 'Your gap to the target is large (0.15524 vs 0.76573), so we should improve score with minimal, metric-aligned changes while preserving your model and inference loop. The biggest likely issue now is that you’re using argmax of softmax class probabilities, which is not aligned with quadratic weighted kappa on an ordinal 0–4 scale; a tiny post-processing change is to use the expected value (probability-weighted class index) and then round+clip to 0–4. This keeps the same model, same weights, same preprocessing, and only changes the final mapping from probabilities to labels in a way that usually improves QWK for ordinal tasks. I also keep the submission row order tied to `test.csv` ids (and still write `submission.csv` with the required columns).'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score indicates the model is effectively not using meaningful trained weights (random predictions), so the smallest change that can move you toward the 0.7657 target is to make weight loading actually succeed in this environment. I keep your model and preprocessing/inference loop intact, but expand the weight search to include any `.h5` file in Kaggle inputs/data and pick the best candidate by file size (common when the exact filename differs). I also fail fast if no viable weights are found (instead of silently producing a near-random submission), because a valid-but-random submission stay far from the target. Submission format/order remains tied to `test.csv`, and the expected-value rounding post-processing is preserved.'
- What this solution (achieved 0.00773) has done: 'The notebook currently stops because it hard-fails when no external `.h5` weights are found; in this environment the weights file is not present under `/kaggle/input` or `/kaggle/data`. To make it run end-to-end and still aim for a non-random score increase toward your target, I keep the exact same model and inference loop but switch to a safe fallback: initialize the EfficientNetB2 backbone with ImageNet weights when custom weights are unavailable (architecture unchanged). I also keep your BEN preprocessing + EfficientNet `preprocess_input` path intact and ensure we always write a valid `submission.csv` with `id_code,diagnosis` aligned to `test.csv` order. These changes are minimal, unblock execution, and should improve the score versus random initialization without altering the core approach.'
- What this solution (achieved 0.0) has done: 'We’re far below the target (0.00773 vs 0.7657), so the minimal way to move the score upward without changing your model architecture or training loop is to fix inference-time behavior that strongly harms ordinal QWK. I (1) make dropout inactive by forcing `model.eval()` equivalent in Keras (`training=False` via `model.predict` is already inference-safe, but we also explicitly set `model.trainable=False` and ensure no accidental training mode), and (2) add a tiny, metric-aligned post-processing calibration: search a single scalar temperature on the test probabilities to reduce overconfidence, then apply the same expected-value rounding you already use. This keeps the same model and preprocessing, only adjusts probability calibration (not labels) in a way that typically improves kappa when the head is weak/random. Submission reading/writing and paths remain unchanged, and it still always produces `submission.csv`.'
- What this solution (achieved -0.02912) has done: 'Your current 0.0 score is far below the 0.7657 target, and the biggest likely cause (without changing your model/training core) is that you’re not actually using the trained competition weights—your fallback leaves the classification head randomly initialized, which typically produces near-random ordinal predictions. I make the smallest change that reliably searches for and loads a valid `.h5` weights file from the provided dataset directories (including the nested `aptos2019-blindness-detection/...` locations) and only fall back to ImageNet+random head if none exists. I also keep your preprocessing and expected-value rounding intact, but switch submission construction to be driven by `test.csv` order directly (still valid Kaggle format), eliminating any possible id/order mismatch from mapping through `sample_submission.csv`. These changes preserve the same architecture and inference approach while materially increasing the chance of meaningful predictions and moving the score upward toward the target.'
- What this solution (achieved 0.0) has done: 'We’re far below the target (current -0.02912 vs target 0.7657, higher-is-better), so the smallest likely score-moving fix is to make your inference preprocessing match what a trained EfficientNetB2 head would have seen: right now you use `load_ben_color()` but you never apply your own `preprocessing()` normalization pipeline, and your BEN function differs from it slightly. I unify inference to use your existing `preprocessing()` output (float32 in [0,1]) and then apply EfficientNet’s `preprocess_input` in the correct expected scale, which is a minimal change that preserves the model and loop but should improve signal. I also remove the test-time temperature scaling (set it to 1.0) because without validation-based tuning it can easily hurt QWK and move you away from the target. Everything else (architecture, weights search/loading, expected-value rounding, submission format and ordering) stays the same.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("PYTHONHASHSEED", "0")

try:
    from google.protobuf import message_factory as _message_factory

    def _patch_message_factory_getprototype():
        MF = getattr(_message_factory, "MessageFactory", None)
        if MF is None:
            return

        def _GetPrototype(self, descriptor):
            if hasattr(self, "GetMessageClass"):
                return self.GetMessageClass(descriptor)
            raise AttributeError(
                "MessageFactory has no GetMessageClass; cannot emulate GetPrototype"
            )

        if not hasattr(MF, "GetPrototype"):
            setattr(MF, "GetPrototype", _GetPrototype)

        for attr in ("_DEFAULT", "default_pool", "_FACTORY", "_factory", "factory"):
            obj = getattr(_message_factory, attr, None)
            if (
                obj is not None
                and hasattr(obj, "__class__")
                and obj.__class__.__name__.endswith("MessageFactory")
            ):
                if not hasattr(obj, "GetPrototype") and hasattr(obj, "GetMessageClass"):
                    try:
                        setattr(
                            obj,
                            "GetPrototype",
                            _GetPrototype.__get__(obj, obj.__class__),
                        )
                    except Exception:
                        pass

    _patch_message_factory_getprototype()
except Exception:
    pass

import gc
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tqdm import tqdm

tf.random.set_seed(0)
np.random.seed(0)




## === cell 1
"""
    Config
"""
IMG_SIZE = 224
BATCH_SIZE = 16  # kept for compatibility; not used in this single-image loop


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


def load_ben_color(image_bgr, sigmaX=10):
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image).astype("uint8")
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image


def preprocessing(image_rgb_uint8, sigmaX=10):
    image = crop_image_from_gray(image_rgb_uint8).astype("uint8")
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image.astype("float32") / 255.0




## === cell 2
base_model = tf.keras.applications.efficientnet.EfficientNetB2(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)

flatten_layer = tf.keras.layers.Flatten()
dense_layer_1 = tf.keras.layers.Dense(4096, activation="relu")
Dropout_1 = tf.keras.layers.Dropout(0.6)
dense_layer_2 = tf.keras.layers.Dense(2048, activation="relu")
Dropout_2 = tf.keras.layers.Dropout(0.6)
dense_layer_3 = tf.keras.layers.Dense(1024, activation="relu")
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
        prediction_layer,
    ]
)

_ = model(tf.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32))

model.trainable = False




## === cell 3
def _iter_weight_candidates():
    yield from [
        "../input/eff-b2-model/eff_b2_model.h5",
        "/kaggle/input/eff-b2-model/eff_b2_model.h5",
        "/kaggle/data/eff-b2-model/eff_b2_model.h5",
        "/kaggle/input/eff-b2-model/eff_b2_model/eff_b2_model.h5",
        "/kaggle/input/eff-b2-model/eff_b2_model.h5/eff_b2_model.h5",
        "/kaggle/input/aptos2019-blindness-detection/eff_b2_model.h5",
        "/kaggle/input/aptos2019-blindness-detection/aptos2019-blindness-detection/eff_b2_model.h5",
        "/kaggle/data/aptos2019-blindness-detection/eff_b2_model.h5",
        "/kaggle/data/aptos2019-blindness-detection/aptos2019-blindness-detection/eff_b2_model.h5",
        "/kaggle/working/eff_b2_model.h5",
    ]

    scan_roots = [
        "/kaggle/input",
        "/kaggle/data",
        "/kaggle/data/aptos2019-blindness-detection",
        "/kaggle/input/aptos2019-blindness-detection",
    ]
    for base in scan_roots:
        try:
            if not os.path.exists(base):
                continue
            for root, _, files in os.walk(base):
                for fn in files:
                    if fn.lower().endswith(".h5"):
                        yield os.path.join(root, fn)
        except Exception:
            pass


def _is_valid_weights_file(p: str) -> bool:
    try:
        return os.path.exists(p) and os.path.getsize(p) > 1_000_000
    except Exception:
        return False


seen = set()
candidates = []
for p in _iter_weight_candidates():
    if p in seen:
        continue
    seen.add(p)
    if _is_valid_weights_file(p):
        try:
            sz = os.path.getsize(p)
            name_bonus = (
                1
                if os.path.basename(p).lower() in ("eff_b2_model.h5", "eff-b2-model.h5")
                else 0
            )
            candidates.append((name_bonus, sz, p))
        except Exception:
            pass

candidates.sort(reverse=True)  # (name_bonus desc, size desc)
weight_path = candidates[0][2] if candidates else None

if weight_path is not None:
    model.load_weights(weight_path)
    print(
        f"Loaded custom weights: {weight_path} (size={os.path.getsize(weight_path)} bytes)"
    )
else:
    print(
        "No viable custom .h5 weights found (>1MB). Using ImageNet backbone weights; head remains randomly initialized."
    )




## === cell 4
def resolve_path(*candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


test_csv_path = resolve_path(
    "../input/aptos2019-blindness-detection/test.csv",
    "/kaggle/input/aptos2019-blindness-detection/test.csv",
    "/kaggle/data/aptos2019-blindness-detection/test.csv",
    "/kaggle/data/test.csv",
)
test_img_dir = resolve_path(
    "../input/aptos2019-blindness-detection/test_images",
    "/kaggle/input/aptos2019-blindness-detection/test_images",
    "/kaggle/data/aptos2019-blindness-detection/test_images",
    "/kaggle/data/test_images",
)

if test_csv_path is None:
    raise FileNotFoundError("Could not find test.csv in expected Kaggle paths.")
if test_img_dir is None:
    raise FileNotFoundError(
        "Could not find test_images directory in expected Kaggle paths."
    )

test_df = pd.read_csv(test_csv_path)
id_code = test_df["id_code"].astype(str).values
print(f"Loaded {len(id_code)} test ids from: {test_csv_path}")
print(f"Using test image dir: {test_img_dir}")




## === cell 5
eff_preprocess = tf.keras.applications.efficientnet.preprocess_input
class_idx = np.arange(5, dtype=np.float32)


def _apply_temperature_to_proba(proba: np.ndarray, T: float) -> np.ndarray:
    proba = np.asarray(proba, dtype=np.float32)
    proba = np.clip(proba, 1e-7, 1.0)  # avoid log(0)
    logits = np.log(proba)
    logits = logits / np.float32(T)
    logits = logits - np.max(logits)  # for numerical stability
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits)


TEMPERATURE = 1.0

test_prediction = np.empty(len(id_code), dtype=np.int64)

for i, img_id in enumerate(tqdm(id_code, total=len(id_code))):
    img_path = os.path.join(test_img_dir, f"{img_id}.png")
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        test_prediction[i] = 0
        continue

    img_rgb_uint8 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_01 = preprocessing(img_rgb_uint8)  # float32 in [0,1], cropped/resized/sharpened
    img_255 = img_01 * 255.0  # EfficientNet preprocess expects [0,255]
    img_input = eff_preprocess(img_255)  # EfficientNet normalized

    X = np.expand_dims(img_input, axis=0)
    proba = model.predict(X, verbose=0)[0].astype(np.float32)

    proba = _apply_temperature_to_proba(proba, TEMPERATURE)
    exp_label = float(np.dot(proba, class_idx))
    test_prediction[i] = int(np.clip(np.rint(exp_label), 0, 4))




## === cell 6
sub = pd.DataFrame({"id_code": id_code})
sub["diagnosis"] = test_prediction.astype(np.int64)

sub.to_csv("submission.csv", index=False)

unique, counts = np.unique(sub["diagnosis"].values, return_counts=True)
print(dict(zip(unique.tolist(), counts.tolist())))
print("Wrote submission.csv")

gc.collect()
