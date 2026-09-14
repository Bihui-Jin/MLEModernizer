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
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import sys, random, warnings, math
from concurrent.futures import ThreadPoolExecutor

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
from tensorflow.keras import models, Input, layers, callbacks, utils, optimizers

warnings.filterwarnings("ignore")
np.random.seed(19)
random.seed(19)
tf.random.set_seed(19)

try:
    cv2.setNumThreads(1)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(
        max(1, (os.cpu_count() or 4) // 2)
    )
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(False)
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
import zipfile

train_zip = "/kaggle/input/tgs-salt-identification-challenge/train.zip"
test_zip = "/kaggle/input/tgs-salt-identification-challenge/test.zip"

os.makedirs("train", exist_ok=True)
os.makedirs("test", exist_ok=True)


def _maybe_extract(zip_path: str, out_dir: str, sentinel_paths):
    if all(os.path.exists(p) for p in sentinel_paths):
        return
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)


_maybe_extract(
    train_zip,
    "train",
    sentinel_paths=[os.path.join("train", "images"), os.path.join("train", "masks")],
)
_maybe_extract(
    test_zip,
    "test",
    sentinel_paths=[os.path.join("test", "images")],
)


def _ensure_expected_layout(base_dir: str, expected_subdirs):
    if all(os.path.isdir(os.path.join(base_dir, sd)) for sd in expected_subdirs):
        return
    for root, dirs, files in os.walk(base_dir):
        if root == base_dir:
            continue
        if all(os.path.isdir(os.path.join(root, sd)) for sd in expected_subdirs):
            for sd in expected_subdirs:
                src = os.path.join(root, sd)
                dst = os.path.join(base_dir, sd)
                if os.path.exists(dst):
                    continue
                os.rename(src, dst)
            return


_ensure_expected_layout("train", ["images", "masks"])
_ensure_expected_layout("test", ["images"])

assert os.path.isdir("train/images") and os.path.isdir(
    "train/masks"
), "Train images/masks folders not found after unzip."
assert os.path.isdir("test/images"), "Test images folder not found after unzip."

print(
    "Train images:",
    len(os.listdir("train/images")),
    "Train masks:",
    len(os.listdir("train/masks")),
)
print("Test images:", len(os.listdir("test/images")))



## === cell 3
SHOW_SAMPLES = False

if SHOW_SAMPLES:
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



## === cell 4
train_ids = next(os.walk(config.path_train + "images"))[2]
test_ids = next(os.walk(config.path_test + "images"))[2]

train_ids = sorted(train_ids)
test_ids = sorted(test_ids)

print("N train:", len(train_ids), "N test:", len(test_ids))




## === cell 5
def _read_gray_uint8(path: str) -> np.ndarray:
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(path)
    return im


def _resize_to_model(im2d_uint8: np.ndarray, is_mask: bool = False) -> np.ndarray:
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.resize(
        im2d_uint8, (config.im_width, config.im_height), interpolation=interp
    )


train_img_paths = [os.path.join(config.path_train, "images", fn) for fn in train_ids]
train_msk_paths = [os.path.join(config.path_train, "masks", fn) for fn in train_ids]

X = np.empty(
    (len(train_ids), config.im_height, config.im_width, config.im_chan), dtype=np.uint8
)
Y = np.empty((len(train_ids), config.im_height, config.im_width, 1), dtype=np.bool_)

print("Getting and resizing train images and masks ... ")
sys.stdout.flush()


def _load_one_train(idx: int):
    img = _read_gray_uint8(train_img_paths[idx])
    msk = _read_gray_uint8(train_msk_paths[idx])
    img_r = _resize_to_model(img, is_mask=False)
    msk_r = _resize_to_model(msk, is_mask=True)
    return idx, img_r, (msk_r > 127)


max_workers = min(8, (os.cpu_count() or 4))
with ThreadPoolExecutor(max_workers=max_workers) as ex:
    for idx, img_r, msk_b in tqdm(
        ex.map(_load_one_train, range(len(train_ids))), total=len(train_ids)
    ):
        X[idx, :, :, 0] = img_r
        Y[idx, :, :, 0] = msk_b

print("Done!")
print("X shape:", X.shape)
print("Y shape:", Y.shape, Y.dtype)



## === cell 6
n_train = int(0.9 * len(X))
X_train = X[:n_train]
Y_train = Y[:n_train]
X_eval = X[n_train:]
Y_eval = Y[n_train:]

X_lr = X_train[:, :, ::-1, :]
Y_lr = Y_train[:, :, ::-1, :]
X_ud = X_train[:, ::-1, :, :]
Y_ud = Y_train[:, ::-1, :, :]

X_train = np.concatenate([X_train, X_lr, X_ud], axis=0)
Y_train = np.concatenate([Y_train, Y_lr, Y_ud], axis=0)

del X, Y, X_lr, Y_lr, X_ud, Y_ud

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
PLOT_MODEL = False
if PLOT_MODEL:
    try:
        utils.plot_model(model, expand_nested=True, show_shapes=True)
    except Exception as e:
        print("plot_model skipped due to:", repr(e))



## === cell 9
es = callbacks.EarlyStopping(patience=30, verbose=1, restore_best_weights=True)
rlp = callbacks.ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-12, verbose=1)

BATCH_SIZE = 16

X_train_np = np.ascontiguousarray(X_train, dtype=np.float32)
Y_train_np = np.ascontiguousarray(Y_train.astype(np.float32), dtype=np.float32)
X_eval_np = np.ascontiguousarray(X_eval, dtype=np.float32)
Y_eval_np = np.ascontiguousarray(Y_eval.astype(np.float32), dtype=np.float32)

del X_train, Y_train, X_eval, Y_eval  # free RAM early; does not affect correctness

results = model.fit(
    X_train_np,
    Y_train_np,
    validation_data=(X_eval_np, Y_eval_np),
    epochs=300,
    batch_size=BATCH_SIZE,
    callbacks=[es, rlp],
    verbose=2,
    shuffle=False,
)



## === cell 10
PLOT_HISTORY = False
if PLOT_HISTORY:
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(2, 1, figsize=(20, 8))
    history = pd.DataFrame(results.history)
    history[["loss", "val_loss"]].plot(ax=ax[0])
    history[["acc", "val_acc"]].plot(ax=ax[1])
    fig.suptitle("Learning Curve", fontsize=24)
    plt.show()




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

    def precision_at(threshold, iou):
        matches = iou > threshold
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
preds_eval = model.predict(X_eval_np, batch_size=64, verbose=0)

y_true_flat = Y_eval_np.astype(np.uint8).reshape(Y_eval_np.shape[0], -1)
true_sum = y_true_flat.sum(axis=1).astype(np.int64)

preds_eval_flat = preds_eval.reshape(preds_eval.shape[0], -1).astype(np.float32)

thresholds = np.linspace(0, 1, 50, dtype=np.float32)
tvals = np.arange(0.5, 1.0, 0.05, dtype=np.float32)

B, P = preds_eval_flat.shape
T = thresholds.size

order = np.argsort(preds_eval_flat, axis=1)
pred_sorted = np.take_along_axis(preds_eval_flat, order, axis=1)
true_sorted = np.take_along_axis(y_true_flat, order, axis=1).astype(np.int32)

true_suffix = np.cumsum(true_sorted[:, ::-1], axis=1)[:, ::-1]

idx = np.searchsorted(pred_sorted, thresholds[None, :], side="right")
pred_cnt = (P - idx).astype(np.int64)  # (B,T)

idx_clip = np.clip(idx, 0, P - 1)
gathered = np.take_along_axis(true_suffix, idx_clip, axis=1).astype(np.int64)
inter = np.where(idx == P, 0, gathered)

union = true_sum[:, None] + pred_cnt - inter
union_safe = np.where(union == 0, 1, union)
iou_per_img = inter / union_safe.astype(np.float64)  # (B, T)

ap_per_img = (iou_per_img[:, :, None] > tvals[None, None, :]).mean(axis=2)  # (B, T)
ious = ap_per_img.mean(axis=0)  # (T,)

threshold_best_index = int(np.argmax(ious[9:-10]) + 9)
iou_best = float(ious[threshold_best_index])
threshold_best = float(thresholds[threshold_best_index])

PLOT_THRESH = False
if PLOT_THRESH:
    plt.figure(figsize=(10, 4))
    plt.plot(thresholds, ious)
    plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
    plt.xlabel("Threshold")
    plt.ylabel("IoU metric (validation proxy)")
    plt.title(
        "Threshold vs IoU ({:.3f}, {:.4f})".format(
            float(threshold_best), float(iou_best)
        )
    )
    plt.legend()
    plt.show()

print("Chosen threshold_best:", float(threshold_best))



## === cell 13
test_img_paths = [os.path.join(config.path_test, "images", fn) for fn in test_ids]

X_test = np.empty(
    (len(test_ids), config.im_height, config.im_width, config.im_chan), dtype=np.uint8
)
sizes_test = np.empty((len(test_ids), 2), dtype=np.int32)

print("Getting and resizing test images ... ")
sys.stdout.flush()


def _load_one_test(idx: int):
    img = _read_gray_uint8(test_img_paths[idx])
    h, w = img.shape[:2]
    img_r = _resize_to_model(img, is_mask=False)
    return idx, h, w, img_r


max_workers = min(8, (os.cpu_count() or 4))
with ThreadPoolExecutor(max_workers=max_workers) as ex:
    for idx, h, w, img_r in tqdm(
        ex.map(_load_one_test, range(len(test_ids))), total=len(test_ids)
    ):
        sizes_test[idx, 0] = h
        sizes_test[idx, 1] = w
        X_test[idx, :, :, 0] = img_r

print("Done! X_test:", X_test.shape)



## === cell 14
preds_test = model.predict(X_test.astype(np.float32), batch_size=64, verbose=0)

n_test = preds_test.shape[0]
preds_test_upsampled = [None] * n_test


def _upsample_one(i: int):
    h = int(sizes_test[i, 0])
    w = int(sizes_test[i, 1])
    up = cv2.resize(preds_test[i, :, :, 0], (w, h), interpolation=cv2.INTER_LINEAR)
    return i, up


max_workers = min(8, (os.cpu_count() or 4))
with ThreadPoolExecutor(max_workers=max_workers) as ex:
    for i, up in tqdm(ex.map(_upsample_one, range(n_test)), total=n_test):
        preds_test_upsampled[i] = up




## === cell 15
def RLenc(img, order="F", format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    pixels = img.reshape(-1, order=order).astype(np.uint8)
    padded = np.concatenate([[0], pixels, [0]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    starts = changes[0::2] + 1  # 1-indexed starts in original pixels
    ends = changes[1::2] + 1
    lengths = ends - starts
    if not format:
        return list(zip(starts.tolist(), lengths.tolist()))
    if starts.size == 0:
        return ""
    out = np.empty((starts.size * 2,), dtype=np.int64)
    out[0::2] = starts
    out[1::2] = lengths
    return " ".join(map(str, out.tolist()))


thr = float(threshold_best)
ids_noext = [fn[:-4] for fn in test_ids]
rles = [""] * len(test_ids)

for i in tqdm(range(len(test_ids)), total=len(test_ids)):
    rles[i] = RLenc((preds_test_upsampled[i] > thr).view(np.uint8))

sub = pd.DataFrame({"id": ids_noext, "rle_mask": rles})

sample_path = "/kaggle/input/tgs-salt-identification-challenge/sample_submission.csv"
sample = pd.read_csv(sample_path)
sub = sample[["id"]].merge(sub, on="id", how="left")[["id", "rle_mask"]]
sub["rle_mask"] = sub["rle_mask"].fillna("")

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
