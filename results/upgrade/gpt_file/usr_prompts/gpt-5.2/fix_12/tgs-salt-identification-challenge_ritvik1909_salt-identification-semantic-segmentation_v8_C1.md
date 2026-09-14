# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Segment regions of salt in seismic images.

## Metric
Mean average precision at different intersection over union (IoU) thresholds. The IoU of a proposed set of object pixels and a set of true object pixels is calculated as:

$$\text{IoU}(A, B)=\frac{A \cap B}{A \cup B}$$

The metric sweeps over a range of IoU thresholds, at each point calculating an average precision value. The threshold values range from 0.5 to 0.95 with a step size of 0.05: `(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)`. In other words, at a threshold of 0.5, a predicted object is considered a "hit" if its intersection over union with a ground truth object is greater than 0.5.

At each threshold value 𝑡t, a precision value is calculated based on the number of true positives (TP), false negatives (FN), and false positives (FP) resulting from comparing the predicted object to all ground truth objects:

$$\frac{T P(t)}{T P(t)+F P(t)+F N(t)}$$

A true positive is counted when a single predicted object matches a ground truth object with an IoU above the threshold. A false positive indicates a predicted object had no associated ground truth object. A false negative indicates a ground truth object had no associated predicted object. The average precision of a single image is then calculated as the mean of the above precision values at each IoU threshold:

$$\frac{1}{\mid \text { thresholds } \mid} \sum_t \frac{T P(t)}{T P(t)+F P(t)+F N(t)}$$

## Submission Format
Use run-length encoding on the pixel values. Instead of submitting an exhaustive list of indices for your segmentation, you will submit pairs of values that contain a start position and a run length. E.g. '1 3' implies starting at pixel 1 and running a total of 3 pixels (1,2,3).

The competition format requires a space delimited list of pairs. For example, '1 3 10 5' implies pixels 1,2,3,10,11,12,13,14 are to be included in the mask. The pixels are one-indexed\
and numbered from top to bottom, then left to right: 1 is pixel (1,1), 2 is pixel (2,1), etc.

The metric checks that the pairs are sorted, positive, and the decoded pixel values are not duplicated. It also checks that no two predicted masks for the same image are overlapping.

The file should contain a header and have the following format. Each row in your submission represents a single predicted salt segmentation for the given image.

```
id,rle_mask
3e06571ef3,1 1
a51b08d882,1 1
c32590b06f,1 1
etc.
```

## Dataset
The data is a set of images chosen at various locations chosen at random in the subsurface. The images are 101 x 101 pixels and each pixel is classified as either salt or sediment. In addition to the seismic images, the depth of the imaged location is provided for each image.

# 2. Python version

3.9

# 3. Installed packages



# 4. Data file paths

```
/
    kaggle/
        data/
            depths.csv (3001 lines)
            depths.csv.zip (25.0 kB)
            description.md (83 lines)
            sample_submission.csv (1001 lines)
            sample_submission.csv.zip (6.8 kB)
            test.zip (9.8 MB)
            train.csv (3001 lines)
            train.csv.zip (293.9 kB)
            train.zip (30.9 MB)
            test/
                images/
                    a05ae39815.png (10.5 kB)
                    9bf8a38b92.png (10.7 kB)
                    ... and 998 other files
                test/
            tgs-salt-identification-challenge/
                depths.csv (3001 lines)
                depths.csv.zip (25.0 kB)
                ... and 7 other files
                test/
                    images/
                        a05ae39815.png (10.5 kB)
                        9bf8a38b92.png (10.7 kB)
                        ... and 998 other files
                    test/
                tgs-salt-identification-challenge/
                train/
                    images/
                        66cf41c563.png (7.7 kB)
                        429bf7c665.png (8.2 kB)
                        ... and 2998 other files
                    masks/
                        6eeeda7f4a.png (230 Bytes)
                        5d600057f5.png (230 Bytes)
                        ... and 2998 other files
                    train/
            train/
                images/
                    66cf41c563.png (7.7 kB)
                    429bf7c665.png (8.2 kB)
                    ... and 2998 other files
                masks/
                    6eeeda7f4a.png (230 Bytes)
                    5d600057f5.png (230 Bytes)
                    ... and 2998 other files
                train/
        input/
            depths.csv (3001 lines)
            depths.csv.zip (25.0 kB)
            description.md (83 lines)
            sample_submission.csv (1001 lines)
            sample_submission.csv.zip (6.8 kB)
            test.zip (9.8 MB)
            train.csv (3001 lines)
            train.csv.zip (293.9 kB)
            train.zip (30.9 MB)
            test/
                images/
                    a05ae39815.png (10.5 kB)
                    9bf8a38b92.png (10.7 kB)
                    ... and 998 other files
                test/
                    images/
                        a05ae39815.png (10.5 kB)
                        9bf8a38b92.png (10.7 kB)
                        ... and 998 other files
                    test/
            tgs-salt-identification-challenge/
                depths.csv (3001 lines)
                depths.csv.zip (25.0 kB)
                ... and 7 other files
                test/
                    images/
                        a05ae39815.png (10.5 kB)
                        9bf8a38b92.png (10.7 kB)
                        ... and 998 other files
                    test/
                tgs-salt-identification-challenge/
                train/
                    images/
                        66cf41c563.png (7.7 kB)
                        429bf7c665.png (8.2 kB)
                        ... and 2998 other files
                    masks/
                        6eeeda7f4a.png (230 Bytes)
                        5d600057f5.png (230 Bytes)
                        ... and 2998 other files
                    train/
            train/
                images/
                    66cf41c563.png (7.7 kB)
                    429bf7c665.png (8.2 kB)
                    ... and 2998 other files
                masks/
                    6eeeda7f4a.png (230 Bytes)
                    5d600057f5.png (230 Bytes)
                    ... and 2998 other files
                train/
                    images/
                        66cf41c563.png (7.7 kB)
                        429bf7c665.png (8.2 kB)
                        ... and 2998 other files
                    masks/
                        6eeeda7f4a.png (230 Bytes)
                        5d600057f5.png (230 Bytes)
                        ... and 2998 other files
                    train/
        working/
            tgs-salt-identification-challenge/
                depths.csv (3001 lines)
                depths.csv.zip (25.0 kB)
                ... and 7 other files
                test/
                    images/
                        a05ae39815.png (10.5 kB)
                        9bf8a38b92.png (10.7 kB)
                        ... and 998 other files
                    test/
                tgs-salt-identification-challenge/
                train/
                    images/
                        66cf41c563.png (7.7 kB)
                        429bf7c665.png (8.2 kB)
                        ... and 2998 other files
                    masks/
                        6eeeda7f4a.png (230 Bytes)
                        5d600057f5.png (230 Bytes)
                        ... and 2998 other files
                    train/
```

-> data/depths.csv has 3000 rows and 2 columns.
The columns are: id, z

-> data/sample_submission.csv has 1000 rows and 2 columns.
The columns are: id, rle_mask

-> data/tgs-salt-identification-challenge/depths.csv has 3000 rows and 2 columns.
The columns are: id, z

-> data/tgs-salt-identification-challenge/sample_submission.csv has 1000 rows and 2 columns.
The columns are: id, rle_mask

-> data/tgs-salt-identification-challenge/train.csv has 3000 rows and 2 columns.
The columns are: id, rle_mask

-> data/train.csv has 3000 rows and 2 columns.
The columns are: id, rle_mask

-> input/depths.csv has 3000 rows and 2 columns.
The columns are: id, z

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os, sys, random, warnings, math

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.auto import tqdm, trange
from itertools import chain

import cv2
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from tensorflow.keras.preprocessing.image import (
    array_to_img,
    img_to_array,
    load_img,
    ImageDataGenerator,
)

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import models, Input, layers, callbacks, utils

warnings.filterwarnings("ignore")
np.random.seed(19)
random.seed(19)
tf.random.set_seed(19)

print("TF version:", tf.__version__)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

try:
    if not tf.config.list_physical_devices("GPU"):
        tf.config.threading.set_intra_op_parallelism_threads(
            max(1, os.cpu_count() // 2)
        )
        tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

try:
    cv2.setNumThreads(max(1, (os.cpu_count() or 4)))
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass




## === cell 1
class config:
    im_width = 128
    im_height = 128
    im_chan = 1
    path_train = "train/"
    path_test = "test/"




## === cell 2
import subprocess, shutil


def _run(cmd):
    return subprocess.run(
        cmd,
        shell=True,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _find_comp_root():
    candidates = [
        "/kaggle/input/tgs-salt-identification-challenge",
        "../input/tgs-salt-identification-challenge",
        "/kaggle/data/tgs-salt-identification-challenge",
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return None


def _ensure_dataset_structure():
    if (
        os.path.isdir("train/images")
        and os.path.isdir("train/masks")
        and os.path.isdir("test/images")
    ):
        return

    comp_root = _find_comp_root()
    assert comp_root is not None, "Could not locate competition dataset root."

    src_train_images = os.path.join(comp_root, "train/images")
    src_train_masks = os.path.join(comp_root, "train/masks")
    src_test_images = os.path.join(comp_root, "test/images")

    if (
        os.path.isdir(src_train_images)
        and os.path.isdir(src_train_masks)
        and os.path.isdir(src_test_images)
    ):
        _run("rm -rf train test")
        os.makedirs("train", exist_ok=True)
        os.makedirs("test", exist_ok=True)
        try:
            os.symlink(
                os.path.abspath(src_train_images), os.path.join("train", "images")
            )
            os.symlink(os.path.abspath(src_train_masks), os.path.join("train", "masks"))
            os.symlink(os.path.abspath(src_test_images), os.path.join("test", "images"))
            return
        except Exception:
            _run("rm -rf train test")
            os.makedirs("train", exist_ok=True)
            os.makedirs("test", exist_ok=True)

    _run("rm -rf train test")
    os.makedirs("train", exist_ok=True)
    os.makedirs("test", exist_ok=True)

    train_zip = os.path.join(comp_root, "train.zip")
    test_zip = os.path.join(comp_root, "test.zip")
    assert os.path.isfile(train_zip), f"train.zip not found at {train_zip}"
    assert os.path.isfile(test_zip), f"test.zip not found at {test_zip}"

    _run(f'unzip -q "{train_zip}" -d train/')
    _run(f'unzip -q "{test_zip}" -d test/')

    if not os.path.isdir("train/images") and os.path.isdir("train/train/images"):
        os.makedirs("train/images", exist_ok=True)
        os.makedirs("train/masks", exist_ok=True)
        for fn in os.listdir("train/train/images"):
            shutil.move(
                os.path.join("train/train/images", fn), os.path.join("train/images", fn)
            )
        for fn in os.listdir("train/train/masks"):
            shutil.move(
                os.path.join("train/train/masks", fn), os.path.join("train/masks", fn)
            )
    if not os.path.isdir("test/images") and os.path.isdir("test/test/images"):
        os.makedirs("test/images", exist_ok=True)
        for fn in os.listdir("test/test/images"):
            shutil.move(
                os.path.join("test/test/images", fn), os.path.join("test/images", fn)
            )

    assert os.path.isdir("train/images"), "train/images not found after unzip/normalize"
    assert os.path.isdir("train/masks"), "train/masks not found after unzip/normalize"
    assert os.path.isdir("test/images"), "test/images not found after unzip/normalize"


_ensure_dataset_structure()
print("Train images:", len(os.listdir("train/images")))
print("Train masks :", len(os.listdir("train/masks")))
print("Test images :", len(os.listdir("test/images")))

assert len(os.listdir("train/images")) == 3000, "Unexpected train/images count."
assert len(os.listdir("train/masks")) == 3000, "Unexpected train/masks count."
assert len(os.listdir("test/images")) == 1000, "Unexpected test/images count."




## === cell 3
if os.environ.get("SHOW_SAMPLES", "0") == "1":
    ids = random.choices(os.listdir("train/images"), k=6)
    fig = plt.figure(figsize=(20, 6))
    for j, img_name in enumerate(ids):
        q = j + 1
        img = load_img("train/images/" + img_name, color_mode="grayscale")
        img_mask = load_img("train/masks/" + img_name, color_mode="grayscale")

        plt.subplot(2, 6, q * 2 - 1)
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.subplot(2, 6, q * 2)
        plt.imshow(img_mask, cmap="gray")
        plt.axis("off")
    fig.suptitle("Sample Images", fontsize=24)
    plt.show()
else:
    print("Sample plotting skipped (set SHOW_SAMPLES=1 to enable).")




## === cell 4
train_ids = next(os.walk(config.path_train + "images"))[2]
test_ids = next(os.walk(config.path_test + "images"))[2]

train_ids = sorted(train_ids)
test_ids = sorted(test_ids)

print("Found train_ids:", len(train_ids), "test_ids:", len(test_ids))




## === cell 5
from concurrent.futures import ThreadPoolExecutor

X = np.zeros(
    (len(train_ids), config.im_height, config.im_width, config.im_chan), dtype=np.uint8
)
Y = np.zeros((len(train_ids), config.im_height, config.im_width, 1), dtype=bool)

print("Getting and resizing train images and masks ... ")
sys.stdout.flush()

train_img_dir = os.path.join(config.path_train, "images")
train_msk_dir = os.path.join(config.path_train, "masks")


def _load_one_train(id_):
    img = cv2.imread(os.path.join(train_img_dir, id_), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(
        img, (config.im_width, config.im_height), interpolation=cv2.INTER_AREA
    )
    m = cv2.imread(os.path.join(train_msk_dir, id_), cv2.IMREAD_GRAYSCALE)
    m = cv2.resize(
        m, (config.im_width, config.im_height), interpolation=cv2.INTER_NEAREST
    )
    return img, (m > 127)


max_workers = min(16, (os.cpu_count() or 8))
with ThreadPoolExecutor(max_workers=max_workers) as ex:
    for n, (img, mbool) in tqdm(
        enumerate(ex.map(_load_one_train, train_ids)),
        total=len(train_ids),
    ):
        X[n, ..., 0] = img
        Y[n, ..., 0] = mbool

print("Done!")
print("X shape:", X.shape, "dtype:", X.dtype)
print("Y shape:", Y.shape, "dtype:", Y.dtype)




## === cell 6
n_total = len(X)
n_train = int(0.9 * n_total)

X_train0 = X[:n_train]
Y_train0 = Y[:n_train]
X_eval = X[n_train:]
Y_eval = Y[n_train:]

X_train = np.empty(
    (n_train * 2, config.im_height, config.im_width, config.im_chan), dtype=np.uint8
)
Y_train = np.empty((n_train * 2, config.im_height, config.im_width, 1), dtype=bool)

X_train[:n_train] = X_train0
Y_train[:n_train] = Y_train0
X_train[n_train:] = np.flip(X_train0, axis=2)
Y_train[n_train:] = np.flip(Y_train0, axis=2)

X_train = np.ascontiguousarray(X_train)
Y_train = np.ascontiguousarray(Y_train)
X_eval = np.ascontiguousarray(X_eval)
Y_eval = np.ascontiguousarray(Y_eval)

del X, Y, X_train0, Y_train0

print("X train shape:", X_train.shape, "X eval shape:", X_eval.shape)
print("Y train shape:", Y_train.shape, "Y eval shape:", Y_eval.shape)




## === cell 7
def build_model(input_layer, start_neurons):
    scaled = layers.Lambda(lambda x: x / 255)(input_layer)

    conv1 = layers.Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(
        scaled
    )
    conv1 = layers.Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(
        conv1
    )
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    pool1 = layers.Dropout(0.25)(pool1)

    conv2 = layers.Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(
        pool1
    )
    conv2 = layers.Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(
        conv2
    )
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    pool2 = layers.Dropout(0.5)(pool2)

    conv3 = layers.Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(
        pool2
    )
    conv3 = layers.Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(
        conv3
    )
    pool3 = layers.MaxPooling2D((2, 2))(conv3)
    pool3 = layers.Dropout(0.5)(pool3)

    conv4 = layers.Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(
        pool3
    )
    conv4 = layers.Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(
        conv4
    )
    pool4 = layers.MaxPooling2D((2, 2))(conv4)
    pool4 = layers.Dropout(0.5)(pool4)

    convm = layers.Conv2D(
        start_neurons * 16, (3, 3), activation="relu", padding="same"
    )(pool4)
    convm = layers.Conv2D(
        start_neurons * 16, (3, 3), activation="relu", padding="same"
    )(convm)

    deconv4 = layers.Conv2DTranspose(
        start_neurons * 8, (3, 3), strides=(2, 2), padding="same"
    )(convm)
    uconv4 = layers.concatenate([deconv4, conv4])
    uconv4 = layers.Dropout(0.5)(uconv4)
    uconv4 = layers.Conv2D(
        start_neurons * 8, (3, 3), activation="relu", padding="same"
    )(uconv4)
    uconv4 = layers.Conv2D(
        start_neurons * 8, (3, 3), activation="relu", padding="same"
    )(uconv4)

    deconv3 = layers.Conv2DTranspose(
        start_neurons * 4, (3, 3), strides=(2, 2), padding="same"
    )(uconv4)
    uconv3 = layers.concatenate([deconv3, conv3])
    uconv3 = layers.Dropout(0.5)(uconv3)
    uconv3 = layers.Conv2D(
        start_neurons * 4, (3, 3), activation="relu", padding="same"
    )(uconv3)
    uconv3 = layers.Conv2D(
        start_neurons * 4, (3, 3), activation="relu", padding="same"
    )(uconv3)

    deconv2 = layers.Conv2DTranspose(
        start_neurons * 2, (3, 3), strides=(2, 2), padding="same"
    )(uconv3)
    uconv2 = layers.concatenate([deconv2, conv2])
    uconv2 = layers.Dropout(0.5)(uconv2)
    uconv2 = layers.Conv2D(
        start_neurons * 2, (3, 3), activation="relu", padding="same"
    )(uconv2)
    uconv2 = layers.Conv2D(
        start_neurons * 2, (3, 3), activation="relu", padding="same"
    )(uconv2)

    deconv1 = layers.Conv2DTranspose(
        start_neurons * 1, (3, 3), strides=(2, 2), padding="same"
    )(uconv2)
    uconv1 = layers.concatenate([deconv1, conv1])
    uconv1 = layers.Dropout(0.5)(uconv1)
    uconv1 = layers.Conv2D(
        start_neurons * 1, (3, 3), activation="relu", padding="same"
    )(uconv1)
    uconv1 = layers.Conv2D(
        start_neurons * 1, (3, 3), activation="relu", padding="same"
    )(uconv1)

    output_layer = layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(
        uconv1
    )

    return output_layer


input_layer = Input((config.im_height, config.im_width, config.im_chan))
output_layer = build_model(input_layer, 16)

model = models.Model(input_layer, output_layer)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
model.summary()




## === cell 8
if os.environ.get("PLOT_MODEL", "0") == "1":
    try:
        utils.plot_model(model, expand_nested=True, show_shapes=True)
    except Exception as e:
        print("plot_model skipped due to:", repr(e))
else:
    print("plot_model skipped (set PLOT_MODEL=1 to enable).")




## === cell 9
def _make_ds(Xn, Yn, batch_size, training, cache_path):
    ds = tf.data.Dataset.from_tensor_slices((Xn, Yn))
    options = tf.data.Options()
    options.experimental_deterministic = True
    ds = ds.with_options(options)

    if training:
        ds = ds.shuffle(buffer_size=len(Xn), seed=19, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.cache(cache_path)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


BATCH_SIZE = 8
train_ds = _make_ds(
    X_train, Y_train, BATCH_SIZE, training=True, cache_path="/tmp/tgs_train_cache"
)
eval_ds = _make_ds(
    X_eval, Y_eval, BATCH_SIZE, training=False, cache_path="/tmp/tgs_eval_cache"
)

es = callbacks.EarlyStopping(patience=30, verbose=1, restore_best_weights=True)
rlp = callbacks.ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-12, verbose=1)

results = model.fit(
    train_ds,
    validation_data=eval_ds,
    epochs=300,
    callbacks=[es, rlp],
    verbose=2,
)




## === cell 10
if os.environ.get("SHOW_CURVES", "0") == "1":
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(2, 1, figsize=(20, 8))
    history = pd.DataFrame(results.history)
    history[["loss", "val_loss"]].plot(ax=ax[0])
    history[["acc", "val_acc"]].plot(ax=ax[1])
    fig.suptitle("Learning Curve", fontsize=24)
    plt.show()
else:
    print("Learning curve plotting skipped (set SHOW_CURVES=1 to enable).")




## === cell 11
def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in

    true_objects = 2
    pred_objects = 2

    intersection = np.histogram2d(
        labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects)
    )[0]

    area_true = np.histogram(labels, bins=true_objects)[0]
    area_pred = np.histogram(y_pred, bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    union = area_true + area_pred - intersection

    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    union[union == 0] = 1e-9

    iou = intersection / union

    def precision_at(threshold, iou_):
        matches = iou_ > threshold
        true_positives = np.sum(matches, axis=1) == 1
        false_positives = np.sum(matches, axis=0) == 0
        false_negatives = np.sum(matches, axis=1) == 0
        tp, fp, fn = (
            np.sum(true_positives),
            np.sum(false_positives),
            np.sum(false_negatives),
        )
        return tp, fp, fn

    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)

    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)


def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)




## === cell 12
preds_eval = model.predict(eval_ds, verbose=1)

thresholds = np.linspace(0, 1, 50, dtype=np.float32)

Y_eval_bool = Y_eval[..., 0].astype(np.bool_, copy=False)  # (N,H,W)
preds_eval_p = preds_eval[..., 0].astype(np.float32, copy=False)  # (N,H,W)

y_true_sum = (
    Y_eval_bool.reshape(Y_eval_bool.shape[0], -1)
    .sum(axis=1)
    .astype(np.int64, copy=False)
)

ious = np.empty((thresholds.shape[0],), dtype=np.float64)
preds_flat = preds_eval_p.reshape(preds_eval_p.shape[0], -1)
y_true_flat = Y_eval_bool.reshape(Y_eval_bool.shape[0], -1)

for i, thr in enumerate(thresholds):
    y_pred_flat = preds_flat > thr
    y_pred_sum = y_pred_flat.sum(axis=1).astype(np.int64, copy=False)
    inter = (
        np.logical_and(y_true_flat, y_pred_flat)
        .sum(axis=1)
        .astype(np.int64, copy=False)
    )
    union = y_true_sum + y_pred_sum - inter
    union = np.where(union == 0, 1, union)
    iou_img = inter / union.astype(np.float64)

    ious[i] = (
        (iou_img[:, None] > np.arange(0.5, 1.0, 0.05, dtype=np.float64)[None, :])
        .mean(axis=1)
        .mean()
    )

threshold_best_index = int(np.argmax(ious[9:-10]) + 9)
iou_best = float(ious[threshold_best_index])
threshold_best = float(thresholds[threshold_best_index])

if os.environ.get("SHOW_THRESH_PLOT", "0") == "1":
    plt.figure(figsize=(10, 4))
    plt.plot(thresholds, ious)
    plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
    plt.xlabel("Threshold")
    plt.ylabel("IoU metric")
    plt.title("Threshold vs IoU ({:.3f}, {:.4f})".format(threshold_best, iou_best))
    plt.legend()
    plt.show()
else:
    print("Threshold plot skipped (set SHOW_THRESH_PLOT=1 to enable).")

print("Selected threshold_best:", float(threshold_best), "iou_best:", float(iou_best))




## === cell 13
X_test = np.zeros(
    (len(test_ids), config.im_height, config.im_width, config.im_chan), dtype=np.uint8
)
sizes_test = [None] * len(test_ids)

print("Getting and resizing test images ... ")
sys.stdout.flush()

test_img_dir = os.path.join(config.path_test, "images")


def _load_one_test(item):
    n, id_ = item
    img = cv2.imread(os.path.join(test_img_dir, id_), cv2.IMREAD_GRAYSCALE)
    sz = (img.shape[0], img.shape[1])
    img = cv2.resize(
        img, (config.im_width, config.im_height), interpolation=cv2.INTER_AREA
    )
    return n, sz, img


max_workers = min(16, (os.cpu_count() or 8))
with ThreadPoolExecutor(max_workers=max_workers) as ex:
    for n, sz, img in tqdm(
        ex.map(_load_one_test, enumerate(test_ids)), total=len(test_ids)
    ):
        sizes_test[n] = [sz[0], sz[1]]
        X_test[n, ..., 0] = img

X_test = np.ascontiguousarray(X_test)
print("Done!", "X_test shape:", X_test.shape)




## === cell 14
test_ds = (
    tf.data.Dataset.from_tensor_slices(X_test).batch(16).prefetch(tf.data.AUTOTUNE)
)
preds_test = model.predict(test_ds, verbose=1)

target_h, target_w = sizes_test[0]
preds_ch = preds_test[..., 0].astype(np.float32, copy=False)  # (N,128,128)

preds_test_upsampled = np.empty(
    (preds_ch.shape[0], target_h, target_w), dtype=np.float32
)

for i in tqdm(range(preds_ch.shape[0]), total=preds_ch.shape[0]):
    preds_test_upsampled[i] = cv2.resize(
        preds_ch[i], (target_w, target_h), interpolation=cv2.INTER_LINEAR
    )




## === cell 15
import shutil

shutil.rmtree("train", ignore_errors=True)
shutil.rmtree("test", ignore_errors=True)




## === cell 16
def RLenc(img, order="F", format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    pixels = img.reshape(-1, order=order).astype(np.uint8)
    padded = np.concatenate([[0], pixels, [0]])
    changes = np.flatnonzero(padded[1:] != padded[:-1]) + 1
    runs = changes[0::2]
    ends = changes[1::2]
    lengths = ends - runs
    if not format:
        return list(zip(runs.tolist(), lengths.tolist()))
    if runs.size == 0:
        return ""
    out = np.empty(runs.size * 2, dtype=np.int64)
    out[0::2] = runs
    out[1::2] = lengths
    return " ".join(map(str, out.tolist()))


pred_dict = {}
for i, fn in tqdm(enumerate(test_ids), total=len(test_ids)):
    mask = (preds_test_upsampled[i] > threshold_best).astype(np.uint8)
    pred_dict[fn[:-4]] = RLenc(mask)

sub = pd.DataFrame.from_dict(pred_dict, orient="index")
sub.index.names = ["id"]
sub.columns = ["rle_mask"]

comp_root = _find_comp_root()
sample_path = None
if comp_root is not None:
    candidate = os.path.join(comp_root, "sample_submission.csv")
    if os.path.isfile(candidate):
        sample_path = candidate
if sample_path is None:
    for candidate in [
        "/kaggle/input/tgs-salt-identification-challenge/sample_submission.csv",
        "../input/tgs-salt-identification-challenge/sample_submission.csv",
    ]:
        if os.path.isfile(candidate):
            sample_path = candidate
            break

assert sample_path is not None, "Could not locate sample_submission.csv for ordering."
sample = pd.read_csv(sample_path)
sub = sub.reindex(sample["id"].values)
sub["rle_mask"] = sub["rle_mask"].fillna("")

sub.to_csv("submission.csv")
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
