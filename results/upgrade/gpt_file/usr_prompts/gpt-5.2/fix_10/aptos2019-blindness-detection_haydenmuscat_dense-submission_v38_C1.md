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

0.8503383060270171

# 6. Current score

0.03059

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The main blockers are (1) a TensorFlow/protobuf incompatibility causing `MessageFactory.GetPrototype` errors on import, and (2) missing external `.h5` weights (`../input/densenetmulti/...`) which prevents the model from running at all. I fix the protobuf crash by forcing the pure-Python protobuf implementation before importing TensorFlow (a common Kaggle workaround), and I make the weights handling robust: if no weights are found, the code proceed with random initialization so it can still generate a valid `submission.csv`. These changes keep the architecture/inference logic identical; they only unblock execution and ensure a submission file is always produced.'
- What this solution (achieved -0.00363) has done: 'I fix the TensorFlow/protobuf import crash that’s currently preventing the notebook from running by forcing the pure-Python protobuf runtime *and* disabling the C++ protobuf implementation before TensorFlow is imported. I also make the environment deterministic and reduce the chance of other TF startup issues without changing the model or inference logic. Finally, I keep the existing robust weights-path resolution (and fallback to random weights if none exist) so the script always completes and writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.18049) has done: 'The TensorFlow import is still crashing due to a protobuf runtime mismatch, so I harden the “force python protobuf” workaround by setting the environment variables *before any protobuf/tensorflow-related import* and by explicitly importing `google.protobuf` once to lock the runtime. Next, because your current score is far below target, I ensure the code reliably loads ImageNet weights for DenseNet121 when the custom `.h5` isn’t present (this keeps the same architecture and inference pipeline but provides meaningful features instead of random weights). Finally, I keep the existing submission formatting but make path resolution more robust across `/kaggle/input` vs relative paths so the script always writes a valid `submission.csv`.'
- What this solution (achieved -0.00245) has done: 'I fix the runtime crash caused by the protobuf/TensorFlow incompatibility by pinning protobuf to the pure-Python implementation **before** any protobuf/TensorFlow import and by additionally disabling the C++ protobuf backend. I also make TensorFlow import more robust by ensuring `google.protobuf.message_factory` has the expected `GetPrototype` attribute (older TF expects it), via a minimal compatibility shim; this unblocks execution without changing the model’s logic. I keep your model architecture, preprocessing, and prediction logic unchanged, and ensure the script always writes a valid `submission.csv` with the required columns. No score-tuning changes are introduced beyond making the pipeline actually run reliably (which is prerequisite to improving toward the target).'
- What this solution (achieved -0.21429) has done: 'The crash is happening before your compatibility shim runs because it checks for `GetPrototype` on the class, but the AttributeError is raised from a `MessageFactory` instance during TensorFlow’s import path. I fix this by patching `google.protobuf.message_factory.MessageFactory` at the module level (and also patching an already-created default factory if present) *before* importing TensorFlow, so TF sees the expected `GetPrototype` attribute. This is a minimal runtime-compatibility fix that unblocks end-to-end execution without changing the model architecture or prediction logic. Once TF imports reliably, the rest of the pipeline run and produce a valid `submission.csv` in the required format.'
- What this solution (achieved 0.03185) has done: 'The current failure happens before TensorFlow can import because TensorFlow expects `MessageFactory.GetPrototype` but newer protobuf versions removed it; your patch runs too late for some import paths and doesn’t cover the exact factory object used during TF import. I harden the protobuf compatibility shim by patching both `MessageFactory` and `symbol_database.Default()`’s internal factory *before* importing TensorFlow, ensuring any instance accessed during import has `GetPrototype`. I also make the TF import order strictly “protobuf shim first, TF second” and keep your model/prediction logic unchanged so score changes only come from actually running reliably (and using your existing ImageNet fallback if the `.h5` is absent). Finally, I keep the same submission writing but add one small safety check to ensure the submission has the expected row count and columns.'
- What this solution (achieved 0.03059) has done: 'I fix the TensorFlow/protobuf crash by moving the `GetPrototype` compatibility shim to patch *any* `MessageFactory` instance via `__getattr__`, which covers the exact object TensorFlow touches during import (your current patch still misses some paths). This is a pure runtime-compatibility change and does not alter your model/prediction logic. Once TF imports, the rest of the pipeline run unchanged, still using your existing “custom .h5 if found else ImageNet DenseNet121” behavior (which should improve score vs near-random outputs). Finally, I keep the same submission formatting but add a small safeguard to ensure the `submission.csv` is always written with correct columns and row count.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_DISABLE_CPP_IMPLEMENTATION", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import google.protobuf  # noqa: F401
from google.protobuf import message_factory as _message_factory
from google.protobuf import symbol_database as _symbol_database


def _patch_message_factory_getprototype():
    """
    Provide MessageFactory.GetPrototype in a way that is robust even if TF touches
    a MessageFactory instance created through a different import path during TF import.
    """
    MF = _message_factory.MessageFactory

    if (not hasattr(MF, "GetPrototype")) and hasattr(MF, "GetMessageClass"):

        def GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        MF.GetPrototype = GetPrototype

    if not hasattr(MF, "__getattr__"):

        def __getattr__(self, name):
            if name == "GetPrototype" and hasattr(self, "GetMessageClass"):
                return lambda descriptor: self.GetMessageClass(descriptor)
            raise AttributeError(
                f"{type(self).__name__!s} object has no attribute {name}"
            )

        MF.__getattr__ = __getattr__

    for fac in (
        getattr(_message_factory, "_DEFAULT", None),
        getattr(getattr(_symbol_database.Default(), "pool", None), "_factory", None),
    ):
        if fac is None:
            continue
        if (not hasattr(fac, "GetPrototype")) and hasattr(fac, "GetMessageClass"):
            fac.GetPrototype = lambda descriptor, _fac=fac: _fac.GetMessageClass(
                descriptor
            )


_patch_message_factory_getprototype()

import gc
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.optimizers import Adam

np.random.seed(42)
tf.random.set_seed(42)

IMG_DIM = 256
BATCH_SIZE = 32
CHANNELS = 3
NUM_CLASSES = 5

NORMAL_WEIGHTS = "../input/densenetmulti/0.8822791912279122.h5"
INPUT_FOLDER = "../input/aptos2019-blindness-detection/"


def _resolve_input_root() -> str:
    if os.path.isdir("/kaggle/input"):
        return "/kaggle/input"
    if os.path.isdir("../input"):
        return "../input"
    return "../input"


def _resolve_competition_folder(input_root: str) -> str:
    cand = os.path.join(input_root, "aptos2019-blindness-detection")
    if os.path.isdir(cand):
        return cand + "/"
    return input_root.rstrip("/") + "/"


def _resolve_weights_path(path_hint: str):
    """
    Make weights path robust across Kaggle path layouts.
    If nothing is found, return None (caller will fall back to ImageNet weights).
    """
    if path_hint and os.path.exists(path_hint):
        return path_hint

    input_root = _resolve_input_root()
    candidates = []
    if os.path.isdir(input_root):
        for root, _, files in os.walk(input_root):
            for fn in files:
                if fn.lower().endswith(".h5"):
                    candidates.append(os.path.join(root, fn))

    preferred = [c for c in candidates if "densenetmulti" in c.lower()]
    if preferred:
        preferred.sort(key=lambda x: (len(x), x))
        return preferred[0]

    if candidates:
        candidates.sort(key=lambda x: (len(x), x))
        return candidates[0]

    return None


_input_root = _resolve_input_root()
INPUT_FOLDER = _resolve_competition_folder(_input_root)
NORMAL_WEIGHTS = _resolve_weights_path(NORMAL_WEIGHTS)

print("Using INPUT_FOLDER:", INPUT_FOLDER)
print("Using NORMAL_WEIGHTS:", NORMAL_WEIGHTS)
print("Listing INPUT_FOLDER (head):", os.listdir(INPUT_FOLDER)[:10])




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/863254685.py in __getattr__(self, name)
     37             if name == "GetPrototype" and hasattr(self, "GetMessageClass"):
     38                 return lambda descriptor: self.GetMessageClass(descriptor)
---> 39             raise AttributeError(
     40                 f"{type(self).__name__!s} object has no attribute {name}"
     41             )

AttributeError: MessageFactory object has no attribute GetPrototype

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

    if height < 100 or width < 100:
        print("Error: squareUp: bottom:", bottom, "top:", top)
        print("Error: squareUp: right:", right, "left:", left)
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
        print("Error: circle mask assumes square image")
        return img

    dim = img.shape[0]
    half = int(dim / 2)

    circle_mask = np.zeros((dim, dim), np.uint8)
    circle_mask = cv2.circle(circle_mask, (half, half), half, 1, thickness=-1)

    return cv2.bitwise_and(img, img, mask=circle_mask)


def clahe_gray(gray, clipLimit=3.5, grid=4):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(grid, grid))
    return clahe.apply(gray)


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")

    return cv2.LUT(image, table)


def processBenNormal(bgr):
    green = bgr[:, :, 1]  # use green as a greyscale
    cropped = crop(green, bgr, 0.02)
    squared = reflectAndSquareUp(cropped)
    resized = cv2.resize(squared, (2 * IMG_DIM, 2 * IMG_DIM))
    circled = circleMask(resized)

    med = np.median(circled)
    if med <= 0:
        med = 1.0
    equalised = adjust_gamma(circled, 1 + np.log(90) - np.log(med))

    resized_again = cv2.resize(benYCC(equalised), (IMG_DIM, IMG_DIM))
    return cv2.cvtColor(resized_again, cv2.COLOR_BGR2RGB)


pre_process_function = processBenNormal




## === cell 2
def dataGenerator(jitter=0.1):
    datagen = image.ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True and (jitter > 0.01),
        vertical_flip=True and (jitter > 0.01),
        zoom_range=[max(0.7, 1 - 5 * jitter), 1],
        rotation_range=int(600 * jitter),
        brightness_range=[1 - jitter / 3, 1 + jitter / 3],
        fill_mode="constant",
        cval=128.0,
        channel_shift_range=int(30 * jitter),
    )
    return datagen




## === cell 3
def test_datagen_plot(processing_function, jitter=0.03):
    images_dir = f"{INPUT_FOLDER}train_images/"
    df = pd.read_csv(f"{INPUT_FOLDER}train.csv")
    df.id_code = df.id_code.apply(lambda x: x + ".png")

    img_block = np.empty((32, IMG_DIM, IMG_DIM, CHANNELS))
    for i, filename in enumerate(df[:32].id_code):
        try:
            bgr = cv2.imread(images_dir + filename)
            img_block[i, :, :, :] = processing_function(bgr)
        except Exception as e:
            print("Error opening or manipulating image:", e)
            img_block[i, :, :, :] = 128.0

    datagen_sample = dataGenerator(jitter).flow(img_block)

    figure = plt.figure(figsize=(10, 10))
    for x in datagen_sample:
        for j in range(16):
            ax = figure.add_subplot(4, 4, j + 1)
            plt.imshow(x[j])
            ax.axis("off")
        break
    plt.show()




## === cell 4
def create_model(weights):
    """
    If custom .h5 weights aren't available, fall back to ImageNet backbone weights.
    Keeps the same model architecture (DenseNet121 backbone + GAP + Dropout + Dense(5,sigmoid)).
    """
    model = Sequential()
    backbone_weights = None  # default: random
    if weights is None or (isinstance(weights, str) and not os.path.exists(weights)):
        backbone_weights = "imagenet"

    model.add(
        DenseNet121(
            weights=backbone_weights,
            include_top=False,
            input_shape=(IMG_DIM, IMG_DIM, CHANNELS),
        )
    )
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation="sigmoid"))

    if weights is not None and os.path.exists(weights):
        model.load_weights(weights)
        print("Loaded weights:", weights)
    else:
        if backbone_weights == "imagenet":
            print(
                "WARNING: No valid .h5 weights found. Using ImageNet DenseNet121 weights."
            )
        else:
            print(
                "WARNING: No valid .h5 weights found. Running with random initialization."
            )

    model.compile(
        optimizer=Adam(learning_rate=0.00005),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model




## === cell 5
def make_predictions(d_set, processing_function, model, jitters=5):
    images_dir = f"{INPUT_FOLDER}{d_set}_images/"
    df = pd.read_csv(f"{INPUT_FOLDER}{d_set}.csv")
    df.id_code = df.id_code.apply(lambda x: x + ".png")

    block_size = 512
    total = df.index.size
    predictions = np.zeros((df.index.size, NUM_CLASSES), dtype=np.float32)

    print(f"Making predictions on the {d_set} dataset. Total: {total}")

    for start in range(0, total, block_size):
        end = min(start + block_size, total)

        img_block = np.empty((end - start, IMG_DIM, IMG_DIM, CHANNELS), dtype=np.uint8)
        for i, filename in enumerate(df.iloc[start:end].id_code):
            try:
                bgr = cv2.imread(images_dir + filename)
                if bgr is None:
                    raise ValueError("cv2.imread returned None")
                img_block[i, :, :, :] = processing_function(bgr)
            except Exception as e:
                print("Error opening or manipulating image:", filename, e)
                img_block[i, :, :, :] = 128

        prediction_jitters = np.zeros(
            (len(img_block), jitters, NUM_CLASSES), dtype=np.float32
        )
        jit = 0.0
        for i in range(jitters):
            datagen = dataGenerator(jit).flow(
                img_block, shuffle=False, batch_size=BATCH_SIZE
            )

            prediction_jitters[:, i] = model.predict(
                datagen, steps=len(datagen), verbose=1
            )

            gc.collect()
            jit += 0.02

        predictions[start:end] = np.median(prediction_jitters, axis=1)

        print(f"{start} - {end} finished")
        gc.collect()

    return predictions




## === cell 6
def label_convert(preds):
    y_val = preds > 0.5
    return y_val.astype(int).sum(axis=1) - 1




## === cell 7
model = create_model(NORMAL_WEIGHTS)
test_predictions = make_predictions("test", processBenNormal, model, jitters=5)

test_classes = label_convert(test_predictions).astype(int)
test_classes = np.clip(test_classes, 0, 4)

print("Sample raw preds:", test_predictions[:2])
print("Sample classes:", test_classes[:10])

test_df = pd.read_csv(INPUT_FOLDER + "test.csv")
test_df["diagnosis"] = test_classes
test_df = test_df[["id_code", "diagnosis"]]

if list(test_df.columns) != ["id_code", "diagnosis"]:
    raise ValueError("Submission columns are incorrect: %r" % (list(test_df.columns),))
if len(test_df) != len(pd.read_csv(INPUT_FOLDER + "test.csv")):
    raise ValueError("Submission row count mismatch.")

test_df.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", test_df.shape)
print(test_df.head())
