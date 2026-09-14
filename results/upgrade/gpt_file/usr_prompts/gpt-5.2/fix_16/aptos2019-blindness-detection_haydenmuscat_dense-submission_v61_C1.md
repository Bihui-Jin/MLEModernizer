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

0.9089048795530408

# 6. Current score

-0.11588

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -0.00867) has done: 'I fix the TensorFlow/protobuf crash by removing the forced `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` setting, which is incompatible in this environment and triggers the `MessageFactory.GetPrototype` error before anything runs. Then I fix the Keras `model.predict()` call by removing unsupported `workers`/`use_multiprocessing` arguments (they are not accepted by the current TF/Keras trainer API), keeping prediction logic identical otherwise. Finally, I add a safe fallback for weights: if the external `.h5` is missing, the model run with ImageNet weights so you still get a valid `submission.csv` end-to-end.'
- What this solution (achieved 0.08257) has done: 'I fix the immediate TensorFlow/protobuf crash by removing the problematic environment override and forcing a compatible protobuf implementation before importing TensorFlow. Then I make the input path resolution robust to this dataset’s nested folder layout so the code always finds `train.csv/test.csv` and the `*_images/` directories. Finally, I keep the model/prediction logic intact but add a safe, deterministic fallback: if the external `.h5` weights aren’t present, we use ImageNet weights and produce a valid `submission.csv` (this should also improve your current negative kappa toward the target compared to random/uninitialized behavior).'
- What this solution (achieved 0.27347) has done: 'I fix the TensorFlow/protobuf crash that happens before any training/inference by forcing TensorFlow to use the pure-Python protobuf implementation (and removing any conflicting env settings) *before* importing TensorFlow. I keep your model, preprocessing, TTA/jitter prediction loop, and threshold-to-class conversion unchanged, only making this environment fix plus a small path fallback so the script consistently finds the dataset folder in this Kaggle layout. These changes are required for the notebook to run end-to-end and write a valid `submission.csv`, and they should also restore normal inference behavior (instead of crashing), which is necessary to move your score toward the target. No architecture, loss, or inference semantics are changed.'
- What this solution (achieved 0.38881) has done: 'I fix the TensorFlow/protobuf crash by removing the forced `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"` setting, which is what triggers the `MessageFactory.GetPrototype` AttributeError in this environment. I keep the rest of your pipeline (preprocessing, DenseNet121 model, TTA/jitter loop, threshold-to-class conversion) unchanged so evaluation semantics remain the same. I also make the input folder detection slightly more robust for this specific Kaggle directory layout (without changing I/O paths you already use) to ensure the script always finds `train.csv/test.csv` and image folders. These changes are primarily to restore end-to-end execution and produce a valid `submission.csv`; any score change should come only from the model actually running instead of crashing.'
- What this solution (achieved -0.00591) has done: 'I fix the immediate TensorFlow/protobuf crash by forcing TensorFlow to use the pure-Python protobuf implementation *before* importing TensorFlow (this resolves the `MessageFactory.GetPrototype` AttributeError in this environment). I keep your model, preprocessing, TTA/jitter prediction loop, and threshold-to-class conversion unchanged so the core logic and evaluation semantics remain the same. I also make the dataset root selection prefer `/kaggle/input` (the standard Kaggle path) to avoid accidentally picking an incompatible folder first, while keeping your existing fallback logic. With these changes the notebook should run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved -0.03739) has done: 'I fix the TensorFlow/protobuf crash by removing the incompatible forced `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"` setting (it triggers the `MessageFactory.GetPrototype` error in this environment before anything can run). I keep your model, preprocessing, and TTA/jitter prediction logic intact, but make weight loading robust by locating the provided `densenetmulti` weights anywhere under `/kaggle/input` (so you don’t silently fall back to ImageNet and get near-random kappa). Finally, I add a tiny safety guard so predictions are always valid integers in `[0,4]` and ensure `submission.csv` is written in the required format.'
- What this solution (achieved -0.04438) has done: 'You’re crashing before any training/inference due to an incompatibility between TensorFlow and the installed protobuf runtime; the current code only unsets env vars but doesn’t ensure a compatible protobuf implementation is used before importing TF. I fix this by forcing the pure-Python protobuf implementation early (and keeping your existing pops) so TF imports reliably in this Kaggle image. I also make the external weight search safer by not accidentally picking arbitrary `.h5` files (which can silently load wrong weights and tank kappa), while preserving the same “use external if found else ImageNet” behavior. These changes are execution/stability fixes and should move your score up from the current negative kappa by ensuring the intended weights are loaded (or a consistent fallback is used) and the pipeline runs end-to-end to write `submission.csv`.'
- What this solution (achieved -0.00065) has done: 'I fix the TensorFlow import crash by forcing protobuf to use the pure-Python implementation (the current attempt to force `cpp` fails because the compiled `_message` extension isn’t available). That allow cell 1+ to run and ensures `WEIRD_WEIGHTS` is defined so the later `NameError` disappears. I also update the script to match the provided “cell 0..8” into the required “cell 1..N” format while keeping your model, preprocessing, TTA/jitter inference, and thresholding unchanged. Finally, I keep the existing weights fallback logic and ensure a valid `submission.csv` is always written with the required columns.'
- What this solution (achieved 0.01679) has done: 'I fix the TensorFlow/protobuf import crash by removing the forced pure-Python protobuf setting that triggers the `MessageFactory.GetPrototype` error in this environment, keeping the rest of the pipeline intact. I also add a tiny safety shim to ensure TensorFlow imports cleanly by preferring the default/compiled protobuf when available, without changing model logic. Since your current score is far below target, I additionally make the script reliably load the intended external weights by searching for a `.h5` under `/kaggle/input` (not just in a single folder), because falling back to ImageNet-only classification is the main reason for near-random/negative kappa. All other model architecture, preprocessing, TTA/jitter prediction, and thresholding are preserved, and the script still always write a valid `submission.csv`.'
- What this solution (achieved -0.28004) has done: 'I fix the TensorFlow/protobuf import crash that prevents the notebook from running at all by forcing the compatible pure-Python protobuf implementation *before* importing TensorFlow (and keeping any conflicting env vars cleared). I also make the weight-file search faster and more reliable by restricting it to the APTOS dataset folder first (then falling back to `/kaggle/input`), so you’re much more likely to actually load the intended `.h5` weights instead of silently using ImageNet-only weights (which explains the very low current kappa). Finally, I keep your model, preprocessing, TTA/jitter prediction, and thresholding unchanged, and ensure the script always writes a valid `submission.csv` with the required columns.'
- What this solution (achieved -0.04017) has done: 'The crash happens before any training/inference because forcing `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"` is incompatible with this Kaggle TensorFlow/protobuf build and triggers the `MessageFactory.GetPrototype` AttributeError on `import tensorflow`. I remove that forced override (while still clearing any conflicting protobuf env vars) so TensorFlow imports cleanly and the pipeline can run end-to-end. I keep the model, preprocessing, TTA/jitter prediction loop, and thresholding exactly the same, only adding a small safety fix to ensure the image generator yields an integer number of steps and that labels stay in `[0,4]`. This should both fix the runtime error and restore the intended inference behavior, moving the score up from the current negative kappa.'
- What this solution (achieved -0.11588) has done: 'I fix the TensorFlow/protobuf import crash (`MessageFactory.GetPrototype` AttributeError) by forcing the pure-Python protobuf implementation *before* importing TensorFlow, which is the minimal change that unblocks the entire pipeline. I also add a deterministic seed setup (score-neutral for expected performance but improves reproducibility) and keep your model, preprocessing, TTA/jitter loop, thresholding, and submission formatting unchanged. Finally, I make the dataset root selection prefer the standard Kaggle `/kaggle/input/aptos2019-blindness-detection` folder first (without changing any I/O semantics) to avoid accidentally picking an incompatible root. These changes should restore intended inference (instead of crashing) and thereby move the score up from the current negative kappa toward the target.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import gc
import math
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.optimizers import Adam

np.random.seed(0)
tf.random.set_seed(0)

IMG_DIM = 256
BATCH_SIZE = 32
CHANNELS = 3
NUM_CLASSES = 5

CANDIDATE_INPUT_ROOTS = [
    "/kaggle/input",
    "../input",
    "./input",
    "/kaggle/data",
]
INPUT_ROOT = None
for r in CANDIDATE_INPUT_ROOTS:
    if os.path.isdir(r):
        INPUT_ROOT = r
        break
if INPUT_ROOT is None:
    raise FileNotFoundError(
        "Could not find Kaggle input root among: " + str(CANDIDATE_INPUT_ROOTS)
    )

DATASET_SLUG = "aptos2019-blindness-detection"
CANDIDATE_DATASET_FOLDERS = [
    "/kaggle/input/aptos2019-blindness-detection",
    os.path.join(INPUT_ROOT, DATASET_SLUG),
    os.path.join(INPUT_ROOT, "kaggle", "data", DATASET_SLUG),
    os.path.join(INPUT_ROOT, "data", DATASET_SLUG),
    "/kaggle/data/aptos2019-blindness-detection",
]
INPUT_FOLDER = None
for cand in CANDIDATE_DATASET_FOLDERS:
    if os.path.isfile(os.path.join(cand, "train.csv")) and os.path.isfile(
        os.path.join(cand, "test.csv")
    ):
        INPUT_FOLDER = cand
        break

if INPUT_FOLDER is None and os.path.isdir(INPUT_ROOT):
    for name in os.listdir(INPUT_ROOT):
        cand = os.path.join(INPUT_ROOT, name)
        if (
            os.path.isdir(cand)
            and os.path.isfile(os.path.join(cand, "train.csv"))
            and os.path.isfile(os.path.join(cand, "test.csv"))
        ):
            INPUT_FOLDER = cand
            break

if INPUT_FOLDER is None:
    if os.path.isfile(os.path.join(INPUT_ROOT, "train.csv")) and os.path.isfile(
        os.path.join(INPUT_ROOT, "test.csv")
    ):
        INPUT_FOLDER = INPUT_ROOT
    else:
        raise FileNotFoundError(
            "Could not locate dataset folder containing train.csv/test.csv under INPUT_ROOT="
            + str(INPUT_ROOT)
        )

INPUT_FOLDER = INPUT_FOLDER.rstrip("/") + "/"

print("TF version:", tf.__version__)
print("INPUT_ROOT:", INPUT_ROOT)
print("INPUT_FOLDER:", INPUT_FOLDER)
print("Found train.csv:", os.path.isfile(os.path.join(INPUT_FOLDER, "train.csv")))
print("Found test.csv:", os.path.isfile(os.path.join(INPUT_FOLDER, "test.csv")))


def _find_weights_file():
    """
    Score-critical: external weights likely exist in this Kaggle dataset; failing to load them
    often yields near-random kappa. Minimal change: search deterministically but more reliably,
    preferring the current dataset folder first to avoid scanning all of /kaggle/input.
    """

    def scan_for_h5(scan_root):
        preferred = []
        densenet_named = []
        for root, dirs, files in os.walk(scan_root):
            base = os.path.basename(root).lower()
            for f in files:
                fl = f.lower()
                if not fl.endswith(".h5"):
                    continue
                full = os.path.join(root, f)
                if "densenetmulti" in base:
                    preferred.append(full)
                if "densenet" in fl or "densenet" in root.lower():
                    densenet_named.append(full)
        preferred = sorted(set(preferred))
        densenet_named = sorted(set(densenet_named))
        return preferred, densenet_named

    if os.path.isdir(INPUT_FOLDER):
        p, d = scan_for_h5(INPUT_FOLDER)
        if p:
            return p[0]
        if d:
            return d[0]

    if os.path.isdir(INPUT_ROOT):
        p, d = scan_for_h5(INPUT_ROOT)
        if p:
            return p[0]
        if d:
            return d[0]

    return os.path.join(INPUT_ROOT, "densenetmulti", "weird.h5")


WEIRD_WEIGHTS = _find_weights_file()
print("Using WEIRD_WEIGHTS:", WEIRD_WEIGHTS)
print("WEIRD_WEIGHTS exists:", os.path.isfile(WEIRD_WEIGHTS))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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


def bensYCC(bgr, weight=4, gamma=15):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)

    y = cv2.addWeighted(y, weight, cv2.GaussianBlur(y, (0, 0), gamma), -weight, 128)

    ycc_modified = cv2.merge((y, cr, cb))
    bens = cv2.cvtColor(ycc_modified, cv2.COLOR_YCrCb2BGR)

    return bens


def bensSimple(img, weight=4, gamma=15):
    bens = cv2.addWeighted(
        img, weight, cv2.GaussianBlur(img, (0, 0), gamma), -weight, 128
    )
    return bens


def adjust_gamma(image_arr, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    return cv2.LUT(image_arr, table)


def process(bgr, final_function=bensYCC):
    if bgr is None:
        return np.full((IMG_DIM, IMG_DIM, CHANNELS), 128, dtype=np.uint8)

    green = bgr[:, :, 1]  # use green as a greyscale

    if bgr.shape != (480, 640, 3):
        cropped = crop(green, bgr, 0.02)
        width = int(cropped.shape[1] * 0.9)
        height = int(width * 480 / 640)
        if height > cropped.shape[0]:
            height = cropped.shape[0] - 2
        h = int((cropped.shape[0] - height) / 2)
        w = int((cropped.shape[1] - width) / 2)
        test_crop = cropped[h : height + h, w : width + w, :]
    else:
        test_crop = bgr

    resized = cv2.resize(test_crop, (IMG_DIM, IMG_DIM), interpolation=cv2.INTER_AREA)
    bens = final_function(resized, weight=3, gamma=15)
    return cv2.cvtColor(bens, cv2.COLOR_BGR2RGB)




## === cell 2
def dataGenerator(jitter=0.1):
    datagen = image.ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True and (jitter > 0.01),
        vertical_flip=True and (jitter > 0.01),
        zoom_range=[max(0.8, 1 - 5 * jitter), 1],
        rotation_range=int(600 * jitter),
        brightness_range=[1 - jitter / 3, 1 + jitter / 3],
        fill_mode="mirror",
        channel_shift_range=int(30 * jitter),
    )
    return datagen




## === cell 3
def test_datagen_plot(processing_function, jitter=0.03):
    images_dir = f"{INPUT_FOLDER}test_images/"
    df = pd.read_csv(f"{INPUT_FOLDER}test.csv")
    df.id_code = df.id_code.apply(lambda x: x + ".png")

    img_block = np.empty(
        (min(100, len(df)), IMG_DIM, IMG_DIM, CHANNELS), dtype=np.uint8
    )
    j = 1
    for i, filename in enumerate(df[: img_block.shape[0]].id_code):
        try:
            bgr = cv2.imread(images_dir + filename)
            img_block[i, :, :, :] = process(bgr, processing_function)
            if bgr is not None:
                if bgr.shape != (480, 640, 3) and j <= 8:
                    ax = figure.add_subplot(4, 4, j)
                    plt.imshow(img_block[i, :, :, :] / 255.0)
                    j += 1
                elif bgr.shape == (480, 640, 3) and j > 8:
                    ax = figure.add_subplot(4, 4, j)
                    plt.imshow(img_block[i, :, :, :] / 255.0)
                    j += 1
                    if j > 16:
                        return
        except Exception:
            img_block[i, :, :, :] = 128

    datagen_sample = dataGenerator(jitter).flow(img_block)
    for x in datagen_sample:
        for j in range(min(16, x.shape[0])):
            ax = figure.add_subplot(4, 4, j + 1)
            plt.imshow(x[j])
        break


try:
    figure = plt.figure(figsize=(22, 20))
    test_datagen_plot(bensSimple)
    gc.collect()
except Exception as e:
    print("Skipping plot due to:", repr(e))




## === cell 4
def create_model(weights):
    use_external = isinstance(weights, str) and os.path.isfile(weights)

    model = Sequential()
    model.add(
        DenseNet121(
            weights=None if use_external else "imagenet",
            include_top=False,
            input_shape=(IMG_DIM, IMG_DIM, CHANNELS),
        )
    )
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation="sigmoid"))

    if use_external:
        model.load_weights(weights)
        print("Loaded external weights from:", weights)
    else:
        if isinstance(weights, str):
            print(
                f"External weights not found at {weights}; using ImageNet base weights."
            )

    model.compile(
        optimizer=Adam(learning_rate=0.00005),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model




## === cell 5
def make_predictions(d_set, processing_function, model, jitters=7):
    images_dir = f"{INPUT_FOLDER}{d_set}_images/"
    df = pd.read_csv(f"{INPUT_FOLDER}{d_set}.csv")
    df.id_code = df.id_code.apply(lambda x: x + ".png")

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Could not find images directory: {images_dir}")

    block_size = 1024
    total = df.index.size
    predictions = np.zeros((df.index.size, NUM_CLASSES), dtype=np.float32)

    print(f"Making predictions on the {d_set} dataset. Total: {total}")

    for start in range(0, total, block_size):
        end = min(start + block_size, total)

        img_block = np.empty((end - start, IMG_DIM, IMG_DIM, CHANNELS), dtype=np.uint8)
        for i, filename in enumerate(df[start:end].id_code):
            bgr = cv2.imread(images_dir + filename)
            img_block[i, :, :, :] = process(bgr, processing_function)

        prediction_jitters = np.zeros(
            (len(img_block), jitters, NUM_CLASSES), dtype=np.float32
        )
        jit = 0.0
        for i in range(jitters):
            datagen = dataGenerator(jit).flow(
                img_block, shuffle=False, batch_size=BATCH_SIZE
            )

            steps = int(math.ceil(len(img_block) / float(BATCH_SIZE)))

            prediction_jitters[:, i] = model.predict(
                datagen,
                steps=steps,
                verbose=1,
            )
            gc.collect()
            jit += 0.0075

        predictions[start:end] = np.median(prediction_jitters, axis=1)
        print(f"{start} - {end} finished")
        gc.collect()

    return predictions




## === cell 6
def prediction_convert_sum(predictions, thresholds):
    thresholded = np.zeros(predictions.shape, dtype=np.int32)
    for i in range(NUM_CLASSES):
        thresholded[:, i] = (predictions[:, i] > thresholds[i]).astype(np.int32)
    y_val = thresholded.sum(axis=1) - 1
    return y_val


def prediction_convert_highest(predictions, thresholds):
    thresholded = np.zeros(predictions.shape, dtype=np.int32)
    for i in range(NUM_CLASSES):
        thresholded[:, i] = (predictions[:, i] > thresholds[i]).astype(np.int32)

    y_val = np.zeros((predictions.shape[0],), dtype=int)
    for i in range(predictions.shape[0]):
        for j in range(4, -1, -1):
            if thresholded[i][j]:
                y_val[i] = j
                break
    return y_val




## === cell 7
def label_convert(preds):
    y_val = preds > 0.5
    return y_val.astype(int).sum(axis=1) - 1




## === cell 8
model = create_model(WEIRD_WEIGHTS)
preds = make_predictions("test", bensSimple, model)

thresholds = [0.5 for _ in range(NUM_CLASSES)]
test_classes = prediction_convert_sum(preds, thresholds)

test_classes = np.clip(test_classes, 0, 4).astype(int)

test_df = pd.read_csv(INPUT_FOLDER + "test.csv")
test_df["diagnosis"] = test_classes
test_df.to_csv("submission.csv", index=False)

print(test_df.head())
print("Wrote submission.csv with shape:", test_df.shape)
print("submission.csv columns:", list(test_df.columns))
print("submission.csv path:", os.path.abspath("submission.csv"))
assert os.path.isfile("submission.csv") and os.path.getsize("submission.csv") > 0
assert list(test_df.columns) == ["id_code", "diagnosis"]
