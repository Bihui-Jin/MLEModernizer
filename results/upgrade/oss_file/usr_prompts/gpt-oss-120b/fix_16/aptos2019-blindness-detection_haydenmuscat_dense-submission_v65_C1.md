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

# 5. Code solution

## === cell 0
import os, random, gc
import cv2, numpy as np, pandas as pd
import matplotlib.pyplot as plt

try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.applications import DenseNet121
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.callbacks import (
        Callback,
        ModelCheckpoint,
        EarlyStopping,
        ReduceLROnPlateau,
    )
    from tensorflow.keras.optimizers import Adam

    TF_AVAILABLE = True
except Exception as e:
    print("TensorFlow import failed:", e)
    TF_AVAILABLE = False

from concurrent.futures import ThreadPoolExecutor
from functools import partial

if TF_AVAILABLE:
    tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count())
    tf.config.threading.set_inter_op_parallelism_threads(os.cpu_count())
    tf.random.set_seed(42)
else:
    np.random.seed(42)

random.seed(42)
np.random.seed(42)

IMG_DIM = 256
BATCH_SIZE = 32
CHANNELS = 3
NUM_CLASSES = 5
INPUT_FOLDER = "../input/aptos2019-blindness-detection/"




## === cell 1
def crop(gray, img, percent_smaller):
    """Vectorised cropping that replicates the original loop logic."""
    thresh = 8
    middle_col = gray[:, gray.shape[1] // 2] > thresh
    rows = np.where(middle_col)[0]
    top, bottom = rows[0], rows[-1]
    middle_row = gray[gray.shape[0] // 2, :] > thresh
    cols = np.where(middle_row)[0]
    left, right = cols[0], cols[-1]

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


def bensYCC(bgr, weight=4, gamma=15):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)
    y = cv2.addWeighted(y, weight, cv2.GaussianBlur(y, (0, 0), gamma), -weight, 128)
    ycc_modified = cv2.merge((y, cr, cb))
    return cv2.cvtColor(ycc_modified, cv2.COLOR_YCrCb2BGR)


def claheYCC(bgr, clipLimit=5, grid=8):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(grid, grid))
    y = clahe.apply(y)
    y = adjust_gamma(y, 1 + np.log(110) - np.log(np.median(y)))
    ycc_modified = cv2.merge((y, cr, cb))
    return cv2.cvtColor(ycc_modified, cv2.COLOR_YCrCb2BGR)


def bensSimple(img, weight=4, gamma=15):
    return cv2.addWeighted(
        img, weight, cv2.GaussianBlur(img, (0, 0), gamma), -weight, 128
    )


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")
    return cv2.LUT(image, table)


def process(bgr, final_function=bensYCC):
    if bgr is None:
        return np.full((IMG_DIM, IMG_DIM, CHANNELS), 128, dtype=np.uint8)
    green = bgr[:, :, 1]  # use green as grayscale
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
PROCESSING_FUNC = None


def _init_worker(func):
    """Initializer for each thread to set the global processing function."""
    global PROCESSING_FUNC
    PROCESSING_FUNC = func


def _load_and_process_path(path):
    """Loads a single image file and applies the global processing function."""
    bgr = cv2.imread(path)
    img = process(bgr, PROCESSING_FUNC)
    return img.astype(np.float32) / 255.0


def _cache_path(func_name):
    return os.path.join(INPUT_FOLDER, f"train_images_{func_name}.npy")


def load_dataset(df, processing_function):
    """Load and process every image in df once, now using threads, pre‑allocation and on‑disk caching."""
    cache_file = _cache_path(processing_function.__name__)
    if os.path.exists(cache_file):
        imgs = np.load(cache_file)
        if TF_AVAILABLE:
            labels = tf.keras.utils.to_categorical(df.diagnosis.values, NUM_CLASSES)
        else:
            labels = np.eye(NUM_CLASSES)[df.diagnosis.values]
        return imgs, labels

    global PROCESSING_FUNC
    PROCESSING_FUNC = processing_function

    image_dir = f"{INPUT_FOLDER}train_images/"
    filenames = df.id_code.apply(lambda x: f"{x}.png").tolist()
    paths = [os.path.join(image_dir, f) for f in filenames]

    max_workers = min(os.cpu_count(), 8)

    imgs = np.empty((len(paths), IMG_DIM, IMG_DIM, CHANNELS), dtype=np.float32)
    with ThreadPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(processing_function,),
    ) as executor:
        for idx, img in enumerate(
            executor.map(_load_and_process_path, paths, chunksize=64)
        ):
            imgs[idx] = img

    gc.collect()
    np.save(cache_file, imgs)  # cache for future runs

    if TF_AVAILABLE:
        labels = tf.keras.utils.to_categorical(df.diagnosis.values, NUM_CLASSES)
    else:
        labels = np.eye(NUM_CLASSES)[df.diagnosis.values]
    return imgs, labels


def train_model(model, processing_function, epochs=3):
    train_df = pd.read_csv(os.path.join(INPUT_FOLDER, "train.csv"))
    imgs, labels = load_dataset(train_df, processing_function)
    if TF_AVAILABLE:
        model.fit(imgs, labels, batch_size=BATCH_SIZE, epochs=epochs, verbose=1)
    else:
        model.fit(imgs, labels)
    return model




## === cell 3
_JITTER_AMOUNTS = [0, 0.01, 0.1, 0.4]


def make_predictions(d_set, processing_function, model, jitters=7):
    images_dir = f"{INPUT_FOLDER}{d_set}_images/"
    df = pd.read_csv(f"{INPUT_FOLDER}{d_set}.csv")
    df.id_code = df.id_code.apply(lambda x: x + ".png")
    total = df.shape[0]
    predictions = np.zeros((total, NUM_CLASSES))
    block_size = 1024
    print(f"Making predictions on the {d_set} dataset. Total: {total}")

    max_workers = min(os.cpu_count(), 8)

    with ThreadPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(processing_function,),
    ) as executor:
        for start in range(0, total, block_size):
            end = min(start + block_size, total)
            filenames = df[start:end].id_code.tolist()
            paths = [os.path.join(images_dir, f) for f in filenames]

            img_block = np.empty(
                (len(paths), IMG_DIM, IMG_DIM, CHANNELS), dtype=np.float32
            )
            for idx, img in enumerate(
                executor.map(_load_and_process_path, paths, chunksize=64)
            ):
                img_block[idx] = img

            jitter_amounts = _JITTER_AMOUNTS
            prediction_jitters = np.zeros(
                (len(img_block), len(jitter_amounts), NUM_CLASSES)
            )

            for i, jit in enumerate(jitter_amounts):
                if jit == 0 or not TF_AVAILABLE:
                    preds = model.predict(img_block, verbose=0)
                else:
                    datagen = dataGenerator(jit).flow(img_block, shuffle=False)
                    preds = model.predict(datagen, verbose=0)
                prediction_jitters[:, i] = preds

            median_preds = np.median(prediction_jitters, axis=1)
            predictions[start:end] = median_preds
            print(f"{start} - {end} finished")
    return predictions




## === cell 4
def prediction_to_class(predictions):
    """Convert probability vectors to class indices via arg‑max."""
    return np.argmax(predictions, axis=1)




## === cell 5
def dataGenerator(jitter=0.1):
    if not TF_AVAILABLE:
        raise RuntimeError("DataGenerator requires TensorFlow")
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


def create_model(network_name):
    if TF_AVAILABLE:
        base = DenseNet121(
            weights="imagenet",
            include_top=False,
            input_shape=(IMG_DIM, IMG_DIM, CHANNELS),
        )
        base.trainable = False
        model = Sequential(
            [
                base,
                GlobalAveragePooling2D(),
                Dropout(0.5),
                Dense(NUM_CLASSES, activation="softmax"),
            ]
        )
        model.compile(
            optimizer=Adam(learning_rate=0.00005),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model
    else:

        class SimplePrototypeModel:
            def __init__(self):
                self.prototypes = None

            def fit(self, imgs, labels):
                self.prototypes = []
                for c in range(NUM_CLASSES):
                    class_imgs = imgs[labels[:, c] == 1]
                    if len(class_imgs) == 0:
                        self.prototypes.append(np.zeros_like(imgs[0]))
                    else:
                        prototype = class_imgs.mean(axis=0)
                        self.prototypes.append(prototype)
                self.prototypes = np.stack(self.prototypes, axis=0)

            def predict(self, imgs):
                N = imgs.shape[0]
                flat_imgs = imgs.reshape(N, -1)
                flat_proto = self.prototypes.reshape(NUM_CLASSES, -1)
                dists = np.linalg.norm(
                    flat_imgs[:, None, :] - flat_proto[None, :, :], axis=2
                )
                scores = -dists
                exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
                probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
                return probs

        return SimplePrototypeModel()


model = create_model("clahe")
model = train_model(model, claheYCC, epochs=3)

preds = make_predictions("test", claheYCC, model)
test_classes = prediction_to_class(preds)

test_df = pd.read_csv(os.path.join(INPUT_FOLDER, "test.csv"))
test_df["diagnosis"] = test_classes
submission_path = "submission.csv"
test_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
