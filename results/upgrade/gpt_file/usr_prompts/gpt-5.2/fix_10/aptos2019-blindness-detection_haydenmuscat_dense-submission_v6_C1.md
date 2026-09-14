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

0.8189281547268961

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the import/runtime issues preventing the notebook from running (the protobuf/`MessageFactory` crash triggered by mixing `keras` and `tensorflow`, missing `Sequential` due to the crashed first cell, and a bad `../input/densenetmulti` reference). I keep the exact same model architecture and preprocessing logic, but switch to `tf.keras` consistently so it works in the Kaggle TensorFlow environment and can load weights if present. I also make the dataset/weights paths robust by using the known APTOS input folder and conditionally loading weights only if the file exists, ensuring the script always produces a valid `submission.csv` with the required columns. These changes are primarily correctness/stability; if the weights file is available, score should move toward the target, and if not, it still run end-to-end and generate a valid submission.'
- What this solution (achieved 0.0) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation before TensorFlow is imported (this is a known Kaggle TF/protobuf compatibility issue). I also make `cv2` optional and fall back to TensorFlow image decoding if OpenCV isn’t available, so the script reliably runs in the stated “no external packages required” environment. Finally, I keep the exact model/preprocessing/prediction logic intact, but make paths a bit more robust and ensure we always write a valid `submission.csv` with the required columns.'
- What this solution (achieved -0.01903) has done: 'The crash happens before any model code runs because TensorFlow is importing an incompatible protobuf runtime; setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` is not sufficient alone, so we also force `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=3` before importing TensorFlow to prevent the `MessageFactory.GetPrototype` error. The current pipeline can also silently fail when `cv2` is unavailable because several preprocessing functions unconditionally call `cv2`; I add safe fallbacks so the script runs in a “no external packages required” environment. Finally, to move the score up from 0.0 toward the target, I make the weights path search robust by scanning the Kaggle input directory for the expected `.h5` file (without changing the model), so if the weights exist in the dataset they be loaded and predictions won’t be random. The submission writing is kept the same but made robust to length mismatches.'
- What this solution (achieved 0.0) has done: 'You’re hitting the TensorFlow/protobuf incompatibility before any model code runs, so the main fix is to pin protobuf to the pure-Python implementation *before* TensorFlow is imported and to avoid importing any TF/Keras submodules until after that. Next, the non-OpenCV path currently crashes because `processImageBgrToRgb` always calls `cv2.resize/cvtColor`; I make it fully safe when `cv2` isn’t available. Finally, your very low score strongly suggests the weight file isn’t being loaded (random predictions); I keep the same architecture and weight filename but expand the search to `/kaggle/input/**` so the weights are found if present, which should move score toward the target without changing the model logic.'
- What this solution (achieved 0.00263) has done: 'You’re currently crashing before any model code runs due to a TensorFlow↔protobuf incompatibility (`MessageFactory.GetPrototype`). I fix that by avoiding TensorFlow/Keras entirely (since you don’t train here) and running inference with the same DenseNet121+GAP+Dropout+Dense(sigmoid) architecture using PyTorch/torchvision, which is available on Kaggle and not affected by protobuf. I keep your preprocessing logic and label conversion semantics the same, and I make the weights-file search robust; if the expected weights aren’t found, the script still run end-to-end and write a valid `submission.csv`. This should move the score up from 0.0 toward your target when the weights are present, while remaining stable and producing a valid submission regardless.'
- What this solution (achieved 0.0) has done: 'Your current score is extremely low because the model is almost certainly running with random weights (the found `.h5` weights aren’t loaded in PyTorch), so the smallest change that can realistically move QWK toward the target is to correctly load the provided Keras/TensorFlow `.h5` weights by running inference with `tf.keras` using the exact same DenseNet121+GAP+Dropout+Dense(sigmoid) architecture and the same preprocessing + `>0.5` multi-label threshold + `label_convert` rule. To keep runtime stable and avoid the protobuf crash you previously hit, this forces the pure-Python protobuf implementation **before** importing TensorFlow. The rest of the pipeline (paths, preprocessing functions, block inference, and submission writing) is kept semantically identical, just swapping the inference backend so weights can actually be used. If no `.h5` weights are found, it still produce a valid `submission.csv` (but score likely remain low).'
- What this solution (achieved 0.0) has done: 'We fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation *before any protobuf/TensorFlow import happens*, and by ensuring `google.protobuf` isn’t imported early via `cv2`/other libs. Next, we make the `processImageBgrToRgb` path safe when `cv2` is unavailable (it currently calls `cv2.resize/cvtColor` unconditionally). Finally, we keep the exact model/prediction logic the same but make the weight-file search and `.h5` loading robust so the model is actually weighted when the file exists, which should move the score up from ~0.0 toward the target.'
- What this solution (achieved 0.0) has done: 'We fix the TensorFlow/protobuf crash (`MessageFactory` missing `GetPrototype`) by forcing the pure-Python protobuf implementation **and** importing `google.protobuf` early before TensorFlow is imported, which avoids the incompatible C++ protobuf path in this Kaggle environment. We keep your model architecture, preprocessing, and prediction-to-label conversion identical, only adjusting import order and environment variables to ensure the notebook runs end-to-end. We also add a defensive fallback so that if TensorFlow still cannot import for any reason, the script still produce a valid `submission.csv` (with deterministic baseline predictions) rather than crashing before file creation. These changes are primarily runtime/stability fixes; if weights are found/loaded, score should move up from ~0.0 toward your target.'
- What this solution (achieved 0.0) has done: 'You’re crashing on TensorFlow import due to the protobuf `MessageFactory.GetPrototype` incompatibility, so the main fix is to prevent TensorFlow from ever importing the problematic compiled protobuf by forcing the pure-Python protobuf implementation and importing `google.protobuf` before TensorFlow. If TensorFlow still fails, the code deterministically fall back to a safe baseline submission (so you always get a valid `submission.csv` instead of a crash). When TensorFlow does import, the model architecture, preprocessing, and `preds>0.5` + `label_convert` logic are kept identical; we only make the import order and weight-file discovery more robust so pretrained weights can actually be loaded when present (which is necessary to move score up from ~0.0 toward your target). Finally, we ensure the test image directory and submission rows align and always write a correct CSV with required columns.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_CPP"] = "1"

import gc
import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

try:
    import cv2  # type: ignore

    HAS_CV2 = True
except Exception:
    cv2 = None
    HAS_CV2 = False

IMG_DIM = 256
BATCH_SIZE = 32
CHANNEL_SIZE = 3
NUM_CLASSES = 5

INPUT_FOLDER_CANDIDATES = [
    "/kaggle/input/aptos2019-blindness-detection/",
    "../input/aptos2019-blindness-detection/",
    "/kaggle/data/aptos2019-blindness-detection/",
]
INPUT_FOLDER = next(
    (p for p in INPUT_FOLDER_CANDIDATES if os.path.exists(p)),
    INPUT_FOLDER_CANDIDATES[0],
)

TEST_IMAGES_DIR = os.path.join(INPUT_FOLDER, "test_images")
if not os.path.isdir(TEST_IMAGES_DIR):
    nested = os.path.join(INPUT_FOLDER, "aptos2019-blindness-detection", "test_images")
    if os.path.isdir(nested):
        TEST_IMAGES_DIR = nested
TEST_IMAGES_DIR = TEST_IMAGES_DIR + os.sep

print("INPUT_FOLDER:", INPUT_FOLDER)
print("Has train.csv:", os.path.exists(os.path.join(INPUT_FOLDER, "train.csv")))
print("Has test.csv:", os.path.exists(os.path.join(INPUT_FOLDER, "test.csv")))
print("Has test_images dir:", os.path.isdir(TEST_IMAGES_DIR))
print("HAS_CV2:", HAS_CV2)

gc.collect()



## === cell 1
test_df = pd.read_csv(os.path.join(INPUT_FOLDER, "test.csv"))
test_df["id_code_png"] = test_df["id_code"].astype(str) + ".png"
test_df.head()




## === cell 2
def label_convert(y_val):
    y_val = y_val.astype(int).sum(axis=1) - 1
    return y_val


def crop(bgr):
    if not HAS_CV2:
        return bgr
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
    if not HAS_CV2:
        return bgr
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)
    y = cv2.addWeighted(y, weight, cv2.GaussianBlur(y, (0, 0), gamma), -weight, 128)
    ycc_modified = cv2.merge((y, cr, cb))
    modified = cv2.cvtColor(ycc_modified, cv2.COLOR_YCrCb2BGR)
    return modified


def processImageBgrToRgb(bgr):
    if not HAS_CV2:
        raise RuntimeError(
            "OpenCV not available; use processImagePathToRgbNoCv2 instead."
        )

    modified = crop(bgr)
    modified = cv2.resize(modified, (IMG_DIM, IMG_DIM))
    modified = colourfulEyes(modified)
    modified = cv2.cvtColor(modified, cv2.COLOR_BGR2RGB)
    return modified.astype(np.float32)


def processImagePathToRgbNoCv2(path):
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        im = im.resize((IMG_DIM, IMG_DIM), resample=Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32)
    return arr




## === cell 3
try:
    import google.protobuf  # noqa: F401

    from google.protobuf import descriptor_pool  # noqa: F401
except Exception as e:
    print("WARNING: could not import google.protobuf early:", repr(e))

TF_AVAILABLE = True
try:
    import tensorflow as tf

    tf.random.set_seed(SEED)
    print("TensorFlow:", tf.__version__)
except Exception as e:
    TF_AVAILABLE = False
    tf = None
    print(
        "WARNING: TensorFlow failed to import; will fall back to baseline predictions."
    )
    print("TF import error:", repr(e))


def _find_weight_file():
    candidates = [
        "../input/densenetmulti/dense-multi-2015-run.h5",  # original reference
        os.path.join(INPUT_FOLDER, "dense-multi-2015-run.h5"),
        os.path.join(
            INPUT_FOLDER, "aptos2019-blindness-detection", "dense-multi-2015-run.h5"
        ),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    target_names = ["dense-multi-2015-run.h5"]
    for base in ["/kaggle/input", INPUT_FOLDER]:
        if os.path.isdir(base):
            for root, _, files in os.walk(base):
                for t in target_names:
                    if t in files:
                        return os.path.join(root, t)
    return None


def build_tf_model(img_dim=IMG_DIM, num_classes=NUM_CLASSES, dropout=0.5):
    inp = tf.keras.Input(shape=(img_dim, img_dim, 3))
    base = tf.keras.applications.DenseNet121(
        include_top=False,
        weights=None,
        input_tensor=inp,
        pooling=None,
    )
    x = base.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(rate=dropout)(x)
    x = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=inp, outputs=x)
    return model


weights_path = _find_weight_file()
tf_model = None
if TF_AVAILABLE:
    tf_model = build_tf_model()
    if weights_path is not None and weights_path.endswith(".h5"):
        print("Found weights:", weights_path)
        try:
            tf_model.load_weights(weights_path)
            print("Loaded .h5 weights into tf.keras model.")
        except Exception as e:
            print("WARNING: failed to load weights; proceeding without them.")
            print("Weight load error:", repr(e))
    else:
        print(
            "WARNING: weights file not found. Proceeding without loading pretrained weights."
        )

    tf_model.trainable = False

gc.collect()



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4
block_size = 128
total = test_df.shape[0]
y_pred_list = np.zeros(total, dtype=int)

if not TF_AVAILABLE or tf_model is None:
    y_pred_list[:] = 0
    print("Using fallback predictions (all zeros) due to unavailable TensorFlow model.")
else:
    for start in range(0, total, block_size):
        gc.collect()
        end = min(start + block_size, total)

        img_list = np.empty((end - start, IMG_DIM, IMG_DIM, 3), dtype=np.float32)

        for i, filename in enumerate(test_df.iloc[start:end]["id_code_png"].values):
            full_path = TEST_IMAGES_DIR + filename
            if not os.path.exists(full_path):
                img_list[i, :, :, :] = 128.0
                continue

            if HAS_CV2:
                bgr = cv2.imread(full_path)
                if bgr is None:
                    img_list[i, :, :, :] = 128.0
                else:
                    try:
                        img_list[i, :, :, :] = processImageBgrToRgb(bgr)
                    except Exception:
                        img_list[i, :, :, :] = 128.0
            else:
                try:
                    img_list[i, :, :, :] = processImagePathToRgbNoCv2(full_path)
                except Exception:
                    img_list[i, :, :, :] = 128.0

        img_list = img_list * (1.0 / 255.0)

        preds = tf_model(img_list, training=False).numpy()  # (N,5) in [0,1]
        preds_bool = preds > 0.5
        y_pred_list[start:end] = label_convert(preds_bool)

        print(f"{start} - {end} finished")



## === cell 5
submission = pd.read_csv(os.path.join(INPUT_FOLDER, "test.csv"))

n = min(len(submission), len(y_pred_list))
submission = submission.iloc[:n].copy()
submission["diagnosis"] = y_pred_list[:n].astype(int)
submission["diagnosis"] = submission["diagnosis"].clip(0, 4).astype(int)

submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
