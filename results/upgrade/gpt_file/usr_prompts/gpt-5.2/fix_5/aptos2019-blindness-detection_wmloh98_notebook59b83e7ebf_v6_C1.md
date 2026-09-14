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

3.9

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

0.8016449638822514

# 6. Current score

0.04631

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.04631) has done: 'The timeout is dominated by expensive OpenCV preprocessing (crop/circle crop/blur) being executed inside `ImageDataGenerator` on the fly for every batch and every epoch, repeatedly decoding the same PNGs. To preserve the exact model/training logic and augmentations, the main speedup is to precompute the *deterministic* preprocessing once per image (crop/circle/resize/sharpen) and save the result to a cache directory, then let `ImageDataGenerator` only handle cheap augmentations and rescaling. Additionally, we enable TF graph optimizations and ensure OpenCV uses all CPU threads while avoiding per-call overhead. This keeps the same image transformation semantics (same processed pixels entering the augmenter), but removes redundant repeated work across epochs and between train/val/test.'
- What this solution (achieved 0.04631) has done: 'The immediate runtime failure comes from forcing `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`, which triggers an incompatibility with the protobuf version bundled in the Kaggle TensorFlow environment (hence the `MessageFactory.GetPrototype` error). I remove those two environment overrides (they are not needed here) and instead keep only deterministic seeding/thread settings. Then I ensure the pipeline still builds the preprocessing cache, trains, predicts, and writes `submission.csv` with the required `id_code,diagnosis` columns. No model/training/augmentation logic is changed beyond unblocking TensorFlow import and execution.'

# 9. Code solution

## === cell 0
import os

for k in [
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION",
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION",
]:
    if k in os.environ:
        del os.environ[k]

import random
import numpy as np
import pandas as pd

import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass
try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    cv2.setUseOptimized(True)
    cv2.setNumThreads(0)
except Exception:
    pass

DATA_PATH = "../input/aptos2019-blindness-detection/"

DIM_X = 256
DIM_Y = 256
BATCH_SIZE = 32

TRAIN_CSV = os.path.join(DATA_PATH, "train.csv")
TEST_CSV = os.path.join(DATA_PATH, "test.csv")
TRAIN_DIR = os.path.join(DATA_PATH, "train_images")
TEST_DIR = os.path.join(DATA_PATH, "test_images")

print("TensorFlow:", tf.__version__)
print("Train CSV exists:", os.path.exists(TRAIN_CSV))
print("Test CSV exists:", os.path.exists(TEST_CSV))
print("Train dir exists:", os.path.isdir(TRAIN_DIR))
print("Test dir exists:", os.path.isdir(TEST_DIR))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        if mask.any():
            return img[np.ix_(mask.any(1), mask.any(0))]
        return img
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
        img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
        img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
        return np.stack([img1, img2, img3], axis=-1)
    return img


def circle_crop_v2(img):
    height, width, depth = img.shape
    largest_side = int(np.max((height, width)))
    img = cv2.resize(img, (largest_side, largest_side))

    height, width, depth = img.shape
    x = int(width / 2)
    y = int(height / 2)
    r = int(np.amin((x, y)))

    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), r, 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    return img


def preprocess_image(image, sigmaX=25, DIM_X=256, DIM_Y=256):
    if image is None:
        image = np.zeros((DIM_Y, DIM_X, 3), dtype=np.uint8)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = circle_crop_v2(image)
    image = cv2.resize(image, (DIM_X, DIM_Y))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image


CACHE_ROOT = os.path.join("/kaggle/working", "preprocessed_cache_256")
TRAIN_CACHE_DIR = os.path.join(CACHE_ROOT, "train_images")
TEST_CACHE_DIR = os.path.join(CACHE_ROOT, "test_images")
os.makedirs(TRAIN_CACHE_DIR, exist_ok=True)
os.makedirs(TEST_CACHE_DIR, exist_ok=True)


def _preprocess_to_cache(src_path: str, dst_path: str) -> None:
    if os.path.exists(dst_path):
        return
    img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
    img = preprocess_image(img, DIM_X=DIM_X, DIM_Y=DIM_Y)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(dst_path, img_bgr)


def build_preprocessed_cache(file_list, src_dir, dst_dir, max_workers=None):
    from concurrent.futures import ThreadPoolExecutor

    if max_workers is None:
        max_workers = min(8, (os.cpu_count() or 4))

    src_paths = [os.path.join(src_dir, f) for f in file_list]
    dst_paths = [os.path.join(dst_dir, f) for f in file_list]

    tasks = [(s, d) for s, d in zip(src_paths, dst_paths) if not os.path.exists(d)]
    if not tasks:
        return

    def _worker(t):
        s, d = t
        _preprocess_to_cache(s, d)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(ex.map(_worker, tasks))




## === cell 2
def kappa_keras(y_true, y_pred):
    y_true = K.cast(K.argmax(y_true, axis=-1), dtype="int32")
    y_pred = K.cast(K.argmax(y_pred, axis=-1), dtype="int32")

    min_rating = K.minimum(K.min(y_true), K.min(y_pred))
    max_rating = K.maximum(K.max(y_true), K.max(y_pred))

    y_true = K.map_fn(lambda y: y - min_rating, y_true, dtype="int32")
    y_pred = K.map_fn(lambda y: y - min_rating, y_pred, dtype="int32")

    num_ratings = max_rating - min_rating + 1
    observed = tf.math.confusion_matrix(y_true, y_pred, num_classes=num_ratings)
    num_scored_items = K.shape(y_true)[0]

    weights = K.expand_dims(K.arange(num_ratings), axis=-1) - K.expand_dims(
        K.arange(num_ratings), axis=0
    )
    weights = K.cast(K.pow(weights, 2), dtype="float64")

    hist_true = tf.math.bincount(y_true, minlength=num_ratings)
    hist_true = hist_true[:num_ratings] / tf.cast(num_scored_items, hist_true.dtype)
    hist_pred = tf.math.bincount(y_pred, minlength=num_ratings)
    hist_pred = hist_pred[:num_ratings] / tf.cast(num_scored_items, hist_pred.dtype)
    expected = K.dot(
        K.expand_dims(hist_true, axis=-1), K.expand_dims(hist_pred, axis=0)
    )

    observed = tf.cast(observed, tf.float64) / tf.cast(num_scored_items, tf.float64)

    score = tf.where(
        K.any(K.not_equal(weights, 0)),
        K.sum(weights * observed) / K.sum(weights * expected),
        0.0,
    )

    return 1.0 - score


def create_kappa_loss(bsize, eps=1e-10, N=5):
    def kappa_loss(y_true, y_pred):
        y_true = tf.cast(y_true, dtype="float32")
        y_pred = tf.cast(y_pred, dtype="float32")
        repeat_op = tf.cast(
            tf.tile(tf.reshape(tf.range(0, N), [N, 1]), [1, N]), dtype="float32"
        )
        repeat_op_sq = tf.square((repeat_op - tf.transpose(repeat_op)))
        weights = repeat_op_sq / tf.cast((N - 1) ** 2, dtype="float32")

        pred_ = y_pred**2
        pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, axis=1), [-1, 1]))

        hist_rater_a = tf.reduce_sum(pred_norm, axis=0)
        hist_rater_b = tf.reduce_sum(y_true, axis=0)

        conf_mat = tf.matmul(tf.transpose(pred_norm), y_true)

        nom = tf.reduce_sum(weights * conf_mat)
        denom = tf.reduce_sum(
            weights
            * tf.matmul(
                tf.reshape(hist_rater_a, [N, 1]), tf.reshape(hist_rater_b, [1, N])
            )
            / tf.cast(bsize, dtype="float32")
        )

        return nom / (denom + eps)

    return kappa_loss


KAPPA_LOSS = create_kappa_loss(BATCH_SIZE)



## === cell 3
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

train_df["filename"] = train_df["id_code"].astype(str) + ".png"
test_df["filename"] = test_df["id_code"].astype(str) + ".png"

train_df["diagnosis_str"] = train_df["diagnosis"].astype(str)

from sklearn.model_selection import train_test_split

train_split, val_split = train_test_split(
    train_df,
    test_size=0.15,
    random_state=SEED,
    stratify=train_df["diagnosis"],
)

build_preprocessed_cache(train_df["filename"].tolist(), TRAIN_DIR, TRAIN_CACHE_DIR)
build_preprocessed_cache(test_df["filename"].tolist(), TEST_DIR, TEST_CACHE_DIR)

train_aug = ImageDataGenerator(
    rescale=1 / 255.0,
    rotation_range=15,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.10,
)

val_aug = ImageDataGenerator(
    rescale=1 / 255.0,
)

train_gen = train_aug.flow_from_dataframe(
    dataframe=train_split,
    directory=TRAIN_CACHE_DIR,
    x_col="filename",
    y_col="diagnosis_str",
    class_mode="categorical",
    target_size=(DIM_Y, DIM_X),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
    validate_filenames=False,
)

val_gen = val_aug.flow_from_dataframe(
    dataframe=val_split,
    directory=TRAIN_CACHE_DIR,
    x_col="filename",
    y_col="diagnosis_str",
    class_mode="categorical",
    target_size=(DIM_Y, DIM_X),
    batch_size=BATCH_SIZE,
    shuffle=False,
    validate_filenames=False,
)

num_classes = len(train_gen.class_indices)
print("Detected classes:", train_gen.class_indices)
print("num_classes:", num_classes)



## === cell 4
from tensorflow.keras import layers, models

inputs = layers.Input(shape=(DIM_Y, DIM_X, 3))
x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

EPOCHS = 3
steps_per_epoch = int(np.ceil(train_gen.samples / BATCH_SIZE))
val_steps = int(np.ceil(val_gen.samples / BATCH_SIZE))

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=val_steps,
    verbose=1,
)



## === cell 5
submission_df = pd.read_csv(TEST_CSV)
submission_df["filename"] = submission_df["id_code"].astype(str) + ".png"

test_aug = ImageDataGenerator(
    rescale=1 / 255.0,
)

test_gen = test_aug.flow_from_dataframe(
    dataframe=submission_df,
    directory=TEST_CACHE_DIR,
    x_col="filename",
    class_mode=None,
    target_size=(DIM_Y, DIM_X),
    batch_size=BATCH_SIZE,
    shuffle=False,
    validate_filenames=False,
)

pred_proba = model.predict(test_gen, verbose=1)
pred = np.argmax(pred_proba, axis=1).astype(int)

out_df = submission_df[["id_code"]].copy()
out_df["diagnosis"] = pred
out_path = "submission.csv"
out_df.to_csv(out_path, index=False)
print("Wrote:", out_path, "shape:", out_df.shape)
print(out_df.head())



## === cell 6
from collections import Counter

cnt = Counter(out_df["diagnosis"].tolist())
print(cnt)
