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

0.8726249846688257

# 6. Current score

0.22873

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.01969) has done: 'I fix the two runtime-breaking issues: (1) the protobuf/Keras import crash caused by using standalone `keras` in this environment, and (2) the missing `ImageDataGenerator` path by switching to `tf.keras.preprocessing.image.ImageDataGenerator`. These are minimal, compatibility-focused changes that keep the same model, preprocessing, and prediction logic intact. I also add a safe fallback for the pretrained weights path so the notebook doesn’t fail if that external dataset isn’t mounted. Finally, I ensure the submission file is written as `submission.csv` with the required columns and correct row alignment.'
- What this solution (achieved -0.14714) has done: 'I fix the TensorFlow import crash caused by an incompatible protobuf runtime (the `MessageFactory.GetPrototype` error) by forcing the pure-Python protobuf implementation before importing TensorFlow, which is a standard Kaggle workaround and is score-neutral. Then I address the main scoring issue: your current label conversion can output `-1` (when no sigmoid outputs exceed 0.5), which produces invalid class labels and tanks QWK; I clamp predictions into the valid 0–4 range while keeping the same “count-thresholded sigmoid” core logic. Finally, I make prediction deterministic and ensure the submission is always aligned to `test.csv` and written to `submission.csv` with the required columns.'
- What this solution (achieved 0.25967) has done: 'We fix the protobuf/TensorFlow crash by ensuring the pure-Python protobuf implementation is forced *before* any protobuf/TensorFlow-related imports and by setting the additional `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` environment variable early. Then we make the input path resolution more robust so it consistently finds the dataset under `/kaggle/input/...` in Kaggle notebooks. Finally, to move the score up toward the target (your current negative QWK strongly suggests a label-mapping mismatch), we keep your exact “count sigmoid > 0.5 then minus 1” core logic but add a small data-driven calibration step: optimize the per-class sigmoid thresholds on a validation split using QWK, and apply those thresholds at test time; this changes only the post-processing calibration, not the model.'
- What this solution (achieved -0.07265) has done: 'I fix the runtime-breaking protobuf/TensorFlow import error by forcing the pure-Python protobuf implementation *and* ensuring an older-compatible protobuf version is used before importing TensorFlow (this is a standard Kaggle fix for the `MessageFactory.GetPrototype` crash). I keep your model, preprocessing, and prediction logic identical, and only adjust the environment/bootstrap so it runs end-to-end. To move the score upward toward the target, I keep your existing threshold-tuning calibration but make it deterministic and ensure it always produces valid 0–4 labels aligned exactly to `test.csv`. Finally, I ensure `submission.csv` is always written with the required columns and row count.'
- What this solution (achieved 0.23848) has done: 'Your negative QWK strongly suggests the post-processing is still mismatched to the label semantics used by this pretrained multi-label sigmoid model: the current “count(sigmoid>thr)-1” mapping can easily invert ordering and collapse classes. To move the score up toward the target with minimal disruption, I keep your model, preprocessing, TTA/jitter, and threshold-tuning loop, but switch only the final label conversion to an ordinal-consistent mapping: convert the 5 sigmoid outputs into 4 cumulative probabilities (P(y>k)) and count how many exceed tuned thresholds to get classes 0–4. I also tune only 4 thresholds (for k=0..3) on the validation split using the same QWK grid search, which is a small calibration change (not a model change). This should eliminate the pathological label inversion and typically moves QWK from negative/low into a much more reasonable range while keeping the rest identical, and still writes a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current gap to the target is large (0.23848 → 0.87262), so we need a small but high-impact correction that preserves the model and preprocessing while aligning post-processing to the ordinal nature of the task. The current `label_convert` uses the first 4 sigmoid outputs directly as cumulative probabilities, but for many pretrained “multi-label” DR models the 5 sigmoid heads represent per-class likelihoods, not monotonic cumulative heads—this breaks ordinal decoding and depresses QWK. I keep your architecture, weights, TTA/jitter, and threshold tuning loop intact, and change only the conversion to build cumulative probabilities from the 5 class heads (P(y>k)=sum_{c>k} p_c / sum p), then tune/apply thresholds on those 4 cumulative values. This is a minimal, metric-aligned calibration fix and should move the score substantially toward the target while still writing a valid `submission.csv`.'
- What this solution (achieved 0.03018) has done: 'The pipeline fails because the pretrained weights file isn’t available in this environment, which causes `create_model()` to raise and prevents `model` from being defined, cascading into `NameError` later. I make weight-loading robust by (a) searching common Kaggle input locations, and (b) if not found, continuing with ImageNet-initialized DenseNet121 (same architecture) while warning that the score be far from the target. I also guard the threshold-tuning step so it runs only when a model exists, and ensure we always produce a valid `submission.csv` with correct length/columns aligned to `test.csv`. These changes are minimal and focused on unblocking end-to-end execution and producing a valid submission file.'
- What this solution (achieved 0.0) has done: 'I remove the hard failure when the external pretrained `.h5` weights aren’t present, because it currently stops execution and prevents any `submission.csv` from being written. To keep the core model and preprocessing intact while improving the chance of finding weights, I broaden the weight-file search to look for any `.h5/.hdf5` under `/kaggle/input` that matches the expected DenseNet multi-label head shape (5 sigmoid outputs). If no compatible weights are found, the script proceed with ImageNet backbone + randomly initialized head (likely low score, but it run end-to-end), and it still tune thresholds on a small validation split and produce a valid submission. Finally, I make model creation robust so later cells don’t crash with `NameError: model is not defined`.'
- What this solution (achieved 0.22873) has done: 'Your current 0.0 score is most consistent with the model not actually using the intended pretrained DR weights (so predictions are essentially uninformative), rather than a small calibration issue. To move the score up toward the 0.8726 target with minimal logic change, I (1) stop the overly-broad “any compatible .h5” weight search (it can accidentally load unrelated weights and yield garbage), and instead only accept weight files whose Dense layer exactly matches `(1024 -> 5)` with a nontrivial magnitude; and (2) if no such weights are found, fail fast with a clear message instead of silently producing a near-random submission that scores ~0.0. I also make the validation threshold tuning conditional on `weights_loaded` to avoid “tuning noise” when the model is effectively random, while leaving the model architecture, preprocessing, TTA/jitter prediction loop, and QWK-based threshold tuning logic intact. The script still always produce a valid `submission.csv` when weights are successfully loaded.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

try:
    import subprocess, sys

    def _get_dist_version(dist_name):
        try:
            import pkg_resources

            return pkg_resources.get_distribution(dist_name).version
        except Exception:
            return None

    pb_ver = _get_dist_version("protobuf")
    if pb_ver is not None:
        major = int(str(pb_ver).split(".")[0])
        if major >= 4:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf==3.20.3"]
            )
            import importlib

            importlib.invalidate_caches()
except Exception as e:
    print("Warning: protobuf pin attempt failed; continuing. Error:", repr(e))

import gc
import numpy as np
import pandas as pd
import cv2
import psutil
import matplotlib.pyplot as plt

from sklearn.metrics import cohen_kappa_score

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.optimizers import Adam

np.random.seed(42)
tf.random.set_seed(42)

IMG_DIM = 224
BATCH_SIZE = 32
CHANNELS = 3
NUM_CLASSES = 5

MODEL_WEIGHTS = "../input/densenetmulti/ben_normal_-0.9021.h5"


def resolve_input_folder():
    candidates = [
        "/kaggle/input/aptos2019-blindness-detection/",
        "/kaggle/input/aptos2019-blindness-detection/aptos2019-blindness-detection/",
        "/kaggle/data/aptos2019-blindness-detection/",
        "/kaggle/data/aptos2019-blindness-detection/aptos2019-blindness-detection/",
        "../input/aptos2019-blindness-detection/",
        "../input/aptos2019-blindness-detection/aptos2019-blindness-detection/",
        "../data/aptos2019-blindness-detection/",
        "../data/aptos2019-blindness-detection/aptos2019-blindness-detection/",
        "/kaggle/input/",
        "../input/",
    ]
    for c in candidates:
        try:
            c2 = c if c.endswith("/") else (c + "/")
            if os.path.exists(os.path.join(c2, "train.csv")) and os.path.exists(
                os.path.join(c2, "test.csv")
            ):
                return c2
        except Exception:
            pass
    return "./"


def _dense_kernel_shape_and_l1(model):
    """
    Score-relevant safeguard: verify a candidate weights file matches the expected head
    (Dense kernel shape (1024,5)) and has non-trivial magnitude. This prevents accidentally
    loading unrelated .h5 files that "load" but produce garbage predictions (often scoring ~0.0).
    """
    try:
        dense_layer = model.layers[-1]
        w = dense_layer.get_weights()
        if not w or len(w) < 2:
            return None, None
        kernel = w[0]
        l1 = float(np.mean(np.abs(kernel)))
        return tuple(kernel.shape), l1
    except Exception:
        return None, None


def _try_load_weights_into_model_strict(model, weights_path):
    """
    Score-relevant change: only accept weights that (a) load, (b) produce the expected Dense kernel
    shape (1024->5 for DenseNet121+GAP), and (c) have nontrivial magnitude. This is a minimal
    change that directly targets the current 0.0 score failure mode (wrong/missing weights).
    """
    if not weights_path or (not os.path.exists(weights_path)):
        return False
    try:
        model.load_weights(weights_path)
    except Exception:
        return False

    kshape, l1 = _dense_kernel_shape_and_l1(model)
    if kshape != (1024, NUM_CLASSES):
        return False
    if l1 is None or l1 < 1e-4:
        return False
    return True


def resolve_model_weights(path):
    """
    Score-relevant change: remove overly-broad "any .h5 under /kaggle/input" acceptance by using
    a strict compatibility check. This increases the chance we actually load the intended DR weights,
    which is necessary to approach the target QWK.
    """
    env_p = os.environ.get("KAGGLE_MODEL_WEIGHTS", "").strip()
    candidates = []
    if path:
        candidates.append(path)
    if env_p:
        candidates.append(env_p)

    candidates += [
        "/kaggle/input/densenetmulti/ben_normal_-0.9021.h5",
        "../input/densenetmulti/ben_normal_-0.9021.h5",
        "/kaggle/input/densenet-multi/ben_normal_-0.9021.h5",
        "../input/densenet-multi/ben_normal_-0.9021.h5",
        "/kaggle/input/densenetmulti/ben_normal_-0.9021.hdf5",
        "../input/densenetmulti/ben_normal_-0.9021.hdf5",
        "/kaggle/input/densenetmulti/ben_normal_-0.9021.weights.h5",
        "../input/densenetmulti/ben_normal_-0.9021.weights.h5",
    ]

    for p in candidates:
        try:
            if p and os.path.exists(p):
                return p
        except Exception:
            pass

    try:
        base = "/kaggle/input"
        if os.path.isdir(base):
            likely_tokens = (
                "aptos",
                "blind",
                "retina",
                "densenet",
                "ben",
                "normal",
                "dr",
            )
            for root, _, files in os.walk(base):
                for fn in sorted(files):
                    low = fn.lower()
                    if not low.endswith((".h5", ".hdf5")):
                        continue
                    if not any(tok in low for tok in likely_tokens):
                        continue
                    p = os.path.join(root, fn)
                    return p
    except Exception as e:
        print("Warning: weights search failed:", repr(e))

    return None


INPUT_FOLDER = resolve_input_folder()
MODEL_WEIGHTS_RESOLVED = resolve_model_weights(MODEL_WEIGHTS)

print("Resolved INPUT_FOLDER:", INPUT_FOLDER)
print("Resolved MODEL_WEIGHTS (candidate):", MODEL_WEIGHTS_RESOLVED)
print("CPU count:", psutil.cpu_count())
print("TensorFlow version:", tf.__version__)

gc.collect()




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


def adjust_gamma(image_in, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    return cv2.LUT(image_in, table)


def processBenNormal(bgr):
    if bgr is None:
        raise ValueError("cv2.imread returned None")

    green = bgr[:, :, 1]
    cropped = crop(green, bgr, 0.02)
    squared = reflectAndSquareUp(cropped)
    resized = cv2.resize(squared, (2 * IMG_DIM, 2 * IMG_DIM))
    circled = circleMask(resized)

    med = np.median(circled)
    med = max(med, 1.0)
    equalised = adjust_gamma(circled, 1 + np.log(90) - np.log(med))

    resized_again = cv2.resize(benYCC(equalised), (IMG_DIM, IMG_DIM))
    return cv2.cvtColor(resized_again, cv2.COLOR_BGR2RGB)


def processBenWeird(bgr):
    if bgr is None:
        raise ValueError("cv2.imread returned None")

    green = bgr[:, :, 1]
    cropped = crop(green, bgr, 0.02)
    squared = reflectAndSquareUp(cropped)
    resized = cv2.resize(squared, (2 * IMG_DIM, 2 * IMG_DIM))
    circled = circleMask(resized)

    med = np.median(circled)
    med = max(med, 1.0)
    equalised = adjust_gamma(circled, 1 + np.log(90) - np.log(med))

    resized_again = cv2.resize(benSimple(equalised), (IMG_DIM, IMG_DIM))
    return cv2.cvtColor(resized_again, cv2.COLOR_BGR2RGB)




## === cell 2
def dataGenerator(jitter=0.1):
    datagen = image.ImageDataGenerator(
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




## === cell 3
DO_PLOT_AUGMENTATION = False
figure = plt.figure(figsize=(22, 20))


def test_datagen_plot():
    sample_df = pd.read_csv(f"{INPUT_FOLDER}test.csv")
    sample_df.id_code = sample_df.id_code.apply(lambda x: x + ".png")

    img_list = np.empty((32, IMG_DIM, IMG_DIM, 3), dtype=np.float32)
    for i, filename in enumerate(sample_df[:32].id_code):
        try:
            bgr = cv2.imread(f"{INPUT_FOLDER}test_images/{filename}")
            img_list[i, :, :, :] = processBenNormal(bgr)
        except Exception:
            img_list[i, :, :, :] = 128.0

    datagen_sample = dataGenerator(0.03).flow(img_list, shuffle=True, batch_size=32)

    for x in datagen_sample:
        for j in range(16):
            ax = figure.add_subplot(4, 4, j + 1)
            img_ = np.clip(x[j], 0, 1)
            plt.imshow(img_)
            ax.axis("off")
        break


if DO_PLOT_AUGMENTATION:
    test_datagen_plot()
    plt.show()

gc.collect()




## === cell 4
def create_model():
    model = Sequential()
    base = DenseNet121(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_DIM, IMG_DIM, CHANNELS),
    )
    model.add(base)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation="sigmoid"))

    loaded = False

    if MODEL_WEIGHTS_RESOLVED and os.path.exists(MODEL_WEIGHTS_RESOLVED):
        print("Trying to load pretrained weights (strict):", MODEL_WEIGHTS_RESOLVED)
        loaded = _try_load_weights_into_model_strict(model, MODEL_WEIGHTS_RESOLVED)
        if loaded:
            try:
                base.trainable = False
            except Exception:
                pass
            kshape, l1 = _dense_kernel_shape_and_l1(model)
            print("Loaded weights OK. Dense kernel shape:", kshape, "mean|w|:", l1)
        else:
            print(
                "Warning: weights candidate exists but failed strict compatibility checks:",
                MODEL_WEIGHTS_RESOLVED,
            )
    else:
        print("Warning: pretrained weights candidate path not found.")

    return model, loaded


model, weights_loaded = create_model()
model.compile(
    optimizer=Adam(learning_rate=0.00005),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
gc.collect()




## === cell 5
def make_predictions(d_set, jitters=5):
    images_dir = f"{INPUT_FOLDER}{d_set}_images/"
    df = pd.read_csv(f"{INPUT_FOLDER}{d_set}.csv")
    df.id_code = df.id_code.apply(lambda x: x + ".png")

    block_size = 1024
    total = df.index.size
    predictions = np.zeros((df.index.size, NUM_CLASSES), dtype=np.float32)

    print(f"Making predictions on the {d_set} dataset. Total: {total}")
    print("Images dir:", images_dir)

    for start in range(0, total, block_size):
        end = min(start + block_size, total)

        img_block = np.empty(
            (end - start, IMG_DIM, IMG_DIM, CHANNELS), dtype=np.float32
        )
        for i, filename in enumerate(df.iloc[start:end].id_code):
            try:
                bgr = cv2.imread(os.path.join(images_dir, filename))
                img_block[i, :, :, :] = processBenNormal(bgr)
            except Exception:
                img_block[i, :, :, :] = 128.0

        prediction_jitters = np.zeros(
            (len(img_block), jitters, NUM_CLASSES), dtype=np.float32
        )
        jit = 0.0
        for i in range(jitters):
            datagen = dataGenerator(jit).flow(
                img_block, shuffle=False, batch_size=BATCH_SIZE
            )
            pred = model.predict(datagen, steps=len(datagen), verbose=0)
            prediction_jitters[:, i] = pred
            gc.collect()
            jit += 0.02

        predictions[start:end] = np.median(prediction_jitters, axis=1)
        print(f"{start} - {end} finished")
        gc.collect()

    return predictions




## === cell 6
def _to_cumulative_probs_from_class_heads(preds, eps=1e-6):
    preds = np.asarray(preds, dtype=np.float32)
    preds = np.clip(preds, 0.0, 1.0)
    row_sum = preds.sum(axis=1, keepdims=True)
    probs = preds / (row_sum + eps)
    cum = np.cumsum(probs[:, ::-1], axis=1)[:, ::-1]
    cum_gt = cum[:, 1:]  # P(y>0..3)
    return cum_gt


def label_convert(preds, thresholds=None):
    preds = np.asarray(preds, dtype=np.float32)
    if preds.ndim != 2 or preds.shape[1] != NUM_CLASSES:
        raise ValueError(f"preds must have shape (n,{NUM_CLASSES}); got {preds.shape}")

    cum = _to_cumulative_probs_from_class_heads(preds)

    if thresholds is None:
        thresholds = np.full((NUM_CLASSES - 1,), 0.5, dtype=np.float32)
    thresholds = np.asarray(thresholds, dtype=np.float32).reshape((1, -1))
    if thresholds.shape[1] != (NUM_CLASSES - 1):
        raise ValueError(
            f"thresholds must have length {NUM_CLASSES-1}; got {thresholds.shape}"
        )

    cls = (cum > thresholds).astype(np.int32).sum(axis=1)
    return np.clip(cls, 0, NUM_CLASSES - 1)


def load_processed_images(id_codes, images_dir):
    x = np.empty((len(id_codes), IMG_DIM, IMG_DIM, CHANNELS), dtype=np.float32)
    for i, idc in enumerate(id_codes):
        filename = idc + ".png"
        try:
            bgr = cv2.imread(os.path.join(images_dir, filename))
            x[i, :, :, :] = processBenNormal(bgr)
        except Exception:
            x[i, :, :, :] = 128.0
    return x


def predict_on_array(img_array, jitters=3):
    prediction_jitters = np.zeros(
        (len(img_array), jitters, NUM_CLASSES), dtype=np.float32
    )
    jit = 0.0
    for i in range(jitters):
        datagen = dataGenerator(jit).flow(
            img_array, shuffle=False, batch_size=BATCH_SIZE
        )
        pred = model.predict(datagen, steps=len(datagen), verbose=0)
        prediction_jitters[:, i] = pred
        gc.collect()
        jit += 0.02
    return np.median(prediction_jitters, axis=1)


def tune_thresholds_qwk(preds, y_true, init=0.5, iters=2, grid=None):
    if grid is None:
        grid = np.array([0.25, 0.35, 0.45, 0.5, 0.55, 0.65, 0.75], dtype=np.float32)

    thresholds = np.full((NUM_CLASSES - 1,), float(init), dtype=np.float32)
    best = cohen_kappa_score(
        y_true, label_convert(preds, thresholds), weights="quadratic"
    )

    for _ in range(iters):
        improved = False
        for k in range(NUM_CLASSES - 1):
            best_k = thresholds[k]
            best_score_k = best
            for t in grid:
                cand = thresholds.copy()
                cand[k] = t
                y_pred = label_convert(preds, cand)
                score = cohen_kappa_score(y_true, y_pred, weights="quadratic")
                if score > best_score_k:
                    best_score_k = score
                    best_k = t
            if best_score_k > best + 1e-12:
                thresholds[k] = best_k
                best = best_score_k
                improved = True
        if not improved:
            break

    return thresholds, best


best_thresholds = None

if not weights_loaded:
    raise RuntimeError(
        "No compatible pretrained weights were loaded. With this architecture and no training here, "
        "predictions will be near-random and score ~0.0. "
        "Attach the intended weights dataset (e.g., /kaggle/input/densenetmulti/...) or set env var "
        "KAGGLE_MODEL_WEIGHTS to the correct .h5 path, then rerun."
    )

train_df = pd.read_csv(INPUT_FOLDER + "train.csv")
train_images_dir = INPUT_FOLDER + "train_images/"
train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

val_frac = 0.15
val_n = int(len(train_df) * val_frac)
val_df = train_df.iloc[:val_n].copy()

print("Preparing validation set of size:", len(val_df))
val_x = load_processed_images(val_df["id_code"].tolist(), train_images_dir)
val_preds = predict_on_array(val_x, jitters=3)
val_y = val_df["diagnosis"].values.astype(int)

base_qwk = cohen_kappa_score(val_y, label_convert(val_preds), weights="quadratic")
print("Validation QWK with default threshold=0.5:", base_qwk)

best_thresholds, tuned_qwk = tune_thresholds_qwk(val_preds, val_y, init=0.5, iters=2)
print("Tuned thresholds:", best_thresholds)
print("Validation QWK after tuning:", tuned_qwk)

del val_x
gc.collect()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/3287796246.py in <cell line: 0>()
     93 # which is far from the target. Fail fast so the user attaches the correct weights (minimal behavior change).
     94 if not weights_loaded:
---> 95     raise RuntimeError(
     96         "No compatible pretrained weights were loaded. With this architecture and no training here, "
     97         "predictions will be near-random and score ~0.0. "

RuntimeError: No compatible pretrained weights were loaded. With this architecture and no training here, predictions will be near-random and score ~0.0. Attach the intended weights dataset (e.g., /kaggle/input/densenetmulti/...) or set env var KAGGLE_MODEL_WEIGHTS to the correct .h5 path, then rerun.

## === cell 7
test_predictions = make_predictions("test", 5)

if best_thresholds is not None:
    test_classes = label_convert(test_predictions, best_thresholds)
else:
    test_classes = label_convert(test_predictions)

print(test_predictions[:5])
print(test_classes[:5])

test_df = pd.read_csv(INPUT_FOLDER + "test.csv")
if len(test_classes) != len(test_df):
    raise RuntimeError(
        f"Prediction length mismatch: got {len(test_classes)} preds for {len(test_df)} rows"
    )

test_df["diagnosis"] = test_classes.astype(int)
test_df = test_df[["id_code", "diagnosis"]]
test_df.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", test_df.shape)
print(test_df.head())
print("submission.csv exists:", os.path.exists("submission.csv"))
print(
    "submission.csv size:",
    os.path.getsize("submission.csv") if os.path.exists("submission.csv") else None,
)
