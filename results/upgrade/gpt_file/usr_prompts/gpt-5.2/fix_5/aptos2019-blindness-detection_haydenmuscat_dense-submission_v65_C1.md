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

0.9016780557601428

# 6. Current score

0.02594

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -0.09392) has done: 'I fix the notebook so it runs end-to-end in the provided environment and always writes a valid `submission.csv`. The main blockers are (1) an early crash due to a Keras/protobuf compatibility issue and a missing `../input/densenetmulti` directory, (2) `INPUT_FOLDER` not being defined because the earlier cell failed, and (3) outdated Keras prediction API usage and a deprecated NumPy dtype. To keep core logic intact, I (a) switch imports to `tensorflow.keras` (more stable on Kaggle), (b) auto-detect the correct dataset folder under `../input/aptos2019-blindness-detection/`, and (c) load the provided pretrained weights only if present, otherwise fall back to ImageNet weights so a submission is still produced. I also ensure predictions are clipped to valid class range [0,4] and that the CSV has the required columns.'
- What this solution (achieved -0.00055) has done: 'We fix the immediate TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation before importing TensorFlow, which resolves the `MessageFactory.GetPrototype` error on Kaggle’s Py3.7 stacks. Then we correct the model head to match the competition’s 5-class ordinal label setup (use `softmax` + `categorical_crossentropy`) while keeping the same DenseNet121 backbone, pooling, dropout, image preprocessing, and test-time augmentation loop intact. Finally, we convert predictions using `argmax` (score-improving and metric-consistent) and still write a correctly formatted `submission.csv` with `id_code,diagnosis`.'
- What this solution (achieved 0.02352) has done: 'We fix the TensorFlow/protobuf import crash by setting both protobuf environment variables *before* importing TensorFlow and by proactively removing the incompatible `protobuf` package from `sys.path` if present, forcing TensorFlow to use its bundled compatible version (this is the direct cause of the `MessageFactory.GetPrototype` error). We keep your model, preprocessing, TTA, and argmax conversion unchanged to preserve core logic and evaluation semantics. We also add a tiny safety fallback to try alternate TensorFlow import paths if the first import still fails, and ensure the submission is always written as `submission.csv` with the required columns.'
- What this solution (achieved 0.02594) has done: 'We fix the remaining TensorFlow/protobuf import crash by ensuring TensorFlow always uses the pure-Python protobuf runtime and by removing *any* incompatible system protobuf from `sys.path` (not just paths containing “protobuf”), which is the root cause of the `MessageFactory.GetPrototype` error. Then we make the TensorFlow import more robust by retrying with a clean import state. These changes are execution-stability focused and keep your model, preprocessing, TTA, and argmax-based submission logic unchanged (so scoring behavior is preserved, just unblocked). The script still always write a valid `submission.csv` with `id_code,diagnosis`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import sys
import gc
import importlib
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt


def _remove_incompatible_protobuf_from_syspath():
    """
    Root cause: an incompatible 'google.protobuf' from the environment can be imported
    instead of the one TensorFlow expects, causing:
      AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

    Fix: remove suspect site-packages paths that contain google/protobuf and also any
    path that explicitly mentions protobuf, so TF uses a compatible bundled version.
    """
    removed = []
    new_path = []
    for p in sys.path:
        low = (p or "").lower()
        suspect = False
        if "site-packages" in low or "dist-packages" in low:
            if (
                ("protobuf" in low)
                or (low.rstrip("/\\").endswith("google"))
                or ("/google/" in low)
                or ("\\google\\" in low)
            ):
                suspect = True
        if suspect:
            removed.append(p)
            continue
        new_path.append(p)

    if removed:
        sys.path[:] = new_path
        print(
            "Removed suspect protobuf/google paths from sys.path to avoid TF/protobuf incompatibility:"
        )
        for r in removed:
            print(" -", r)


_remove_incompatible_protobuf_from_syspath()

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf") or m == "google" or m.startswith("protobuf"):
        sys.modules.pop(m, None)

try:
    import tensorflow as tf
except Exception as e:
    print("First TensorFlow import failed, retrying after cleanup. Error was:", repr(e))
    gc.collect()
    for m in list(sys.modules.keys()):
        if (
            m.startswith("tensorflow")
            or m.startswith("google.protobuf")
            or m == "google"
        ):
            sys.modules.pop(m, None)
    import tensorflow as tf

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.optimizers import Adam

IMG_DIM = 256
BATCH_SIZE = 32
CHANNELS = 3
NUM_CLASSES = 5

CANDIDATE_INPUT_FOLDERS = [
    "../input/aptos2019-blindness-detection/",
    "/kaggle/input/aptos2019-blindness-detection/",
]
INPUT_FOLDER = None
for p in CANDIDATE_INPUT_FOLDERS:
    if os.path.exists(os.path.join(p, "train.csv")):
        INPUT_FOLDER = p
        break
if INPUT_FOLDER is None:
    raise FileNotFoundError(
        f"Could not find aptos2019-blindness-detection data in: {CANDIDATE_INPUT_FOLDERS}"
    )

print("Using INPUT_FOLDER:", INPUT_FOLDER)
print("Files:", sorted(os.listdir(INPUT_FOLDER))[:25])


def _resolve_images_dir(split):
    candidates = [
        os.path.join(INPUT_FOLDER, f"{split}_images"),
        os.path.join(INPUT_FOLDER, "aptos2019-blindness-detection", f"{split}_images"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c + "/"  # keep existing behavior of path concatenation
    for root, dirs, files in os.walk(INPUT_FOLDER):
        if os.path.basename(root) == f"{split}_images":
            return root + "/"
    raise FileNotFoundError(
        f"Could not locate {split}_images directory under {INPUT_FOLDER}"
    )


TRAIN_IMAGES_DIR = _resolve_images_dir("train")
TEST_IMAGES_DIR = _resolve_images_dir("test")
print("Resolved TRAIN_IMAGES_DIR:", TRAIN_IMAGES_DIR)
print("Resolved TEST_IMAGES_DIR :", TEST_IMAGES_DIR)

np.random.seed(1337)
tf.random.set_seed(1337)




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

    if height < 100 or width < 100:
        return img

    return img[top:bottom, left:right]


def bensYCC(bgr, weight=4, gamma=15):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)
    y = cv2.addWeighted(y, weight, cv2.GaussianBlur(y, (0, 0), gamma), -weight, 128)
    ycc_modified = cv2.merge((y, cr, cb))
    bens = cv2.cvtColor(ycc_modified, cv2.COLOR_YCrCb2BGR)
    return bens


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    return cv2.LUT(image, table)


def claheYCC(bgr, clipLimit=5, grid=8):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)

    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(grid, grid))
    y = clahe.apply(y)
    y = adjust_gamma(y, 1 + np.log(110) - np.log(np.median(y)))

    ycc_modified = cv2.merge((y, cr, cb))
    img = cv2.cvtColor(ycc_modified, cv2.COLOR_YCrCb2BGR)
    return img


def bensSimple(img, weight=4, gamma=15):
    bens = cv2.addWeighted(
        img, weight, cv2.GaussianBlur(img, (0, 0), gamma), -weight, 128
    )
    return bens


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
    img = final_function(resized)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)




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
    images_dir = TEST_IMAGES_DIR
    df = pd.read_csv(f"{INPUT_FOLDER}test.csv")
    df.id_code = df.id_code.apply(lambda x: x + ".png")

    img_block = np.empty(
        (min(100, len(df)), IMG_DIM, IMG_DIM, CHANNELS), dtype=np.uint8
    )
    j = 1
    for i, filename in enumerate(df[: img_block.shape[0]].id_code):
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

    datagen_sample = dataGenerator(jitter).flow(img_block)
    for x in datagen_sample:
        for j in range(16):
            ax = figure.add_subplot(4, 4, j + 1)
            plt.imshow(x[j])
        break


RUN_PLOTS = False
if RUN_PLOTS:
    figure = plt.figure(figsize=(22, 20))
    test_datagen_plot(claheYCC)
    gc.collect()




## === cell 4
def create_model(network_name):
    weights_path = f"../input/densenetmulti/{network_name}.h5"

    model = Sequential()
    base_weights = None if os.path.exists(weights_path) else "imagenet"
    model.add(
        DenseNet121(
            weights=base_weights,
            include_top=False,
            input_shape=(IMG_DIM, IMG_DIM, CHANNELS),
        )
    )
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))

    model.add(Dense(NUM_CLASSES, activation="softmax"))

    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        print("Loaded custom weights:", weights_path)
    else:
        print("WARNING: Custom weights not found:", weights_path)
        print(
            "Falling back to DenseNet121(weights='imagenet'). Submission will be valid but score may be lower."
        )

    model.compile(
        optimizer=Adam(learning_rate=0.00005),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model




## === cell 5
def make_predictions(d_set, processing_function, model, jitters=7):
    images_dir = TEST_IMAGES_DIR if d_set == "test" else TRAIN_IMAGES_DIR
    df = pd.read_csv(f"{INPUT_FOLDER}{d_set}.csv")
    df.id_code = df.id_code.apply(lambda x: x + ".png")

    block_size = 256  # safer memory footprint; does not change core logic (still block inference)
    total = df.index.size
    predictions = np.zeros((df.index.size, NUM_CLASSES), dtype=np.float32)

    print(f"Making predictions on the {d_set} dataset. Total: {total}")

    jitter_amounts = [0, 0.01, 0.01, 0.1, 0.1, 0.4, 0.4]

    for start in range(0, total, block_size):
        end = min(start + block_size, total)

        img_block = np.empty((end - start, IMG_DIM, IMG_DIM, CHANNELS), dtype=np.uint8)
        for i, filename in enumerate(df.iloc[start:end].id_code):
            bgr = cv2.imread(images_dir + filename)
            img_block[i, :, :, :] = process(bgr, processing_function)

        prediction_jitters = np.zeros(
            (len(img_block), len(jitter_amounts), NUM_CLASSES), dtype=np.float32
        )
        for i, jit in enumerate(jitter_amounts):
            datagen = dataGenerator(jit).flow(
                img_block, shuffle=False, batch_size=BATCH_SIZE
            )
            prediction_jitters[:, i] = model.predict(
                datagen, steps=len(datagen), verbose=1
            )
            gc.collect()

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
model = create_model("clahe")
preds = make_predictions("test", claheYCC, model)

test_classes = np.argmax(preds, axis=1).astype(int)
test_classes = np.clip(test_classes, 0, 4).astype(int)

print("First 10 predicted classes:", test_classes[:10])

test_df = pd.read_csv(INPUT_FOLDER + "test.csv")
test_df["diagnosis"] = test_classes
test_df.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", test_df.shape)
print(test_df.head())
