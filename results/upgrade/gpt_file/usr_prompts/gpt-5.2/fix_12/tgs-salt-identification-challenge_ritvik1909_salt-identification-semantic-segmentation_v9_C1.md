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
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

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
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(min(8, os.cpu_count() or 8))
    tf.config.threading.set_inter_op_parallelism_threads(min(2, os.cpu_count() or 2))
except Exception:
    pass

try:
    cv2.setNumThreads(0)
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
train_zip = "../input/tgs-salt-identification-challenge/train.zip"
test_zip = "../input/tgs-salt-identification-challenge/test.zip"
if not os.path.exists(train_zip):
    train_zip = "../input/train.zip"
if not os.path.exists(test_zip):
    test_zip = "../input/test.zip"

os.makedirs("train", exist_ok=True)
os.makedirs("test", exist_ok=True)

if not (os.path.exists("train/images") and os.path.exists("train/masks")) and not (
    os.path.exists("train/train/images") and os.path.exists("train/train/masks")
):
    get_ipython().system(f'unzip -q -o "{train_zip}" -d train/')
if not os.path.exists("test/images") and not os.path.exists("test/test/images"):
    get_ipython().system(f'unzip -q -o "{test_zip}" -d test/')


def _ensure_link(src, dst):
    if os.path.exists(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.symlink(src, dst)
    except Exception:
        get_ipython().system(f'cp -r "{src}" "{dst}"')


if not os.path.exists("train/images") and os.path.exists("train/train/images"):
    _ensure_link(os.path.abspath("train/train/images"), os.path.abspath("train/images"))
if not os.path.exists("train/masks") and os.path.exists("train/train/masks"):
    _ensure_link(os.path.abspath("train/train/masks"), os.path.abspath("train/masks"))
if not os.path.exists("test/images") and os.path.exists("test/test/images"):
    _ensure_link(os.path.abspath("test/test/images"), os.path.abspath("test/images"))

print(
    "train/images exists:",
    os.path.exists("train/images"),
    "count:",
    len(os.listdir("train/images")) if os.path.exists("train/images") else 0,
)
print(
    "train/masks exists:",
    os.path.exists("train/masks"),
    "count:",
    len(os.listdir("train/masks")) if os.path.exists("train/masks") else 0,
)
print(
    "test/images exists:",
    os.path.exists("test/images"),
    "count:",
    len(os.listdir("test/images")) if os.path.exists("test/images") else 0,
)




## === cell 3
if False:
    random.seed(19)
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
train_img_path = os.path.join(config.path_train, "images")
test_img_path = os.path.join(config.path_test, "images")

train_ids = sorted([e.name for e in os.scandir(train_img_path) if e.is_file()])
test_ids = sorted([e.name for e in os.scandir(test_img_path) if e.is_file()])

print("n_train:", len(train_ids), "n_test:", len(test_ids))




## === cell 5
from concurrent.futures import ThreadPoolExecutor


def _read_gray_cv2(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def _resize_img_cv2(
    img: np.ndarray, out_hw=(config.im_height, config.im_width), is_mask=False
) -> np.ndarray:
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    return cv2.resize(img, (out_hw[1], out_hw[0]), interpolation=interp)


X = np.zeros(
    (len(train_ids), config.im_height, config.im_width, config.im_chan), dtype=np.uint8
)
Y = np.zeros((len(train_ids), config.im_height, config.im_width, 1), dtype=np.bool_)

print("Getting and resizing train images and masks ... ")
sys.stdout.flush()

train_img_dir = os.path.join(config.path_train, "images")
train_msk_dir = os.path.join(config.path_train, "masks")


def _load_pair(args):
    n, id_ = args
    img = _read_gray_cv2(os.path.join(train_img_dir, id_))
    img = _resize_img_cv2(img, is_mask=False)
    msk = _read_gray_cv2(os.path.join(train_msk_dir, id_))
    msk = _resize_img_cv2(msk, is_mask=True)
    return n, img, (msk > 127)


max_workers = min(16, (os.cpu_count() or 4))
with ThreadPoolExecutor(max_workers=max_workers) as ex:
    for n, img, msk_bool in tqdm(
        ex.map(_load_pair, enumerate(train_ids)), total=len(train_ids)
    ):
        X[n, ..., 0] = img
        Y[n, ..., 0] = msk_bool

print("Done!")
print("X shape:", X.shape, X.dtype)
print("Y shape:", Y.shape, Y.dtype)




## === cell 6
split = int(0.9 * len(X))
X_train = X[:split]
Y_train = Y[:split]
X_eval = X[split:]
Y_eval = Y[split:]

print("X train shape:", X_train.shape, "X eval shape:", X_eval.shape)
print("Y train shape:", Y_train.shape, "Y eval shape:", Y_eval.shape)




## === cell 7
def build_model(input_layer, start_neurons):
    scaled = layers.Lambda(lambda x: x)(input_layer)

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

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["acc"],
    steps_per_execution=16,
)
model.summary()




## === cell 8
if False:
    try:
        utils.plot_model(model, expand_nested=True, show_shapes=True)
    except Exception as e:
        print("plot_model skipped:", repr(e))




## === cell 9
def _make_ds_uint8(X_arr_u8, Y_arr_bool, batch_size, training: bool):
    ds = tf.data.Dataset.from_tensor_slices((X_arr_u8, Y_arr_bool))

    def _cast_and_scale(x, y):
        x = tf.cast(x, tf.float32) / 255.0
        y = tf.cast(y, tf.float32)
        return x, y

    ds = ds.map(_cast_and_scale, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()

    if training:
        buf = min(int(X_arr_u8.shape[0]), 1024)
        ds = ds.shuffle(buffer_size=buf, seed=19, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size, drop_remainder=False)

    opts = tf.data.Options()
    opts.deterministic = True
    ds = ds.with_options(opts)

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


es = callbacks.EarlyStopping(patience=30, verbose=1, restore_best_weights=True)
rlp = callbacks.ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-12, verbose=1)

batch_size = 8
train_ds = _make_ds_uint8(X_train, Y_train, batch_size=batch_size, training=True)
val_ds = _make_ds_uint8(X_eval, Y_eval, batch_size=batch_size, training=False)

results = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=80,
    callbacks=[es, rlp],
    verbose=2,
)




## === cell 10
if False:
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
eval_ds_pred = (
    tf.data.Dataset.from_tensor_slices(X_eval)
    .map(lambda x: tf.cast(x, tf.float32) / 255.0, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(256)
    .prefetch(tf.data.AUTOTUNE)
)
preds_eval = model.predict(eval_ds_pred, verbose=0)

y_true_flat = Y_eval.astype(np.bool_).reshape(Y_eval.shape[0], -1)
true_area = y_true_flat.sum(axis=1).astype(np.float64)

preds_flat = preds_eval.reshape(preds_eval.shape[0], -1).astype(np.float32, copy=False)

thresholds = np.linspace(0, 1, 50, dtype=np.float32)
iou_thresholds = np.arange(0.5, 1.0, 0.05, dtype=np.float64)

preds_u8 = np.clip((preds_flat * 255.0).round(), 0, 255).astype(np.uint8, copy=False)
y_true_u8 = y_true_flat.astype(np.uint8, copy=False)

N, P = preds_u8.shape
B = 256

offset = (np.arange(N, dtype=np.int64)[:, None] * B).astype(np.int64, copy=False)
idx_flat = (preds_u8.astype(np.int64, copy=False) + offset).ravel()
hist_pred = (
    np.bincount(idx_flat, minlength=N * B).reshape(N, B).astype(np.int32, copy=False)
)

idx_inter_flat = idx_flat
w_flat = y_true_u8.ravel().astype(np.int32, copy=False)
hist_inter = (
    np.bincount(idx_inter_flat, weights=w_flat, minlength=N * B)
    .reshape(N, B)
    .astype(np.int32, copy=False)
)

suf_pred = np.cumsum(hist_pred[:, ::-1], axis=1)[:, ::-1].astype(np.float64, copy=False)
suf_inter = np.cumsum(hist_inter[:, ::-1], axis=1)[:, ::-1].astype(
    np.float64, copy=False
)

t_bin = np.clip((thresholds * 255.0).round().astype(np.int32), 0, 255)
idx = np.minimum(t_bin + 1, 255)

pred_area = suf_pred[:, idx].T
inter = suf_inter[:, idx].T

union = true_area[None, :] + pred_area - inter
union = np.where(union == 0.0, 1e-9, union)
iou_per_img = inter / union

prec_per_img = (iou_per_img[:, :, None] > iou_thresholds[None, None, :]).mean(axis=2)
ious = prec_per_img.mean(axis=1)

threshold_best_index = int(np.argmax(ious[9:-10]) + 9)
iou_best = float(ious[threshold_best_index])
threshold_best = float(thresholds[threshold_best_index])

if False:
    plt.figure(figsize=(10, 4))
    plt.plot(thresholds, ious)
    plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
    plt.xlabel("Threshold")
    plt.ylabel("IoU metric")
    plt.title("Threshold vs IoU ({:.3f}, {:.4f})".format(threshold_best, iou_best))
    plt.legend()
    plt.show()

print("Best threshold:", threshold_best, "Best IoU metric:", iou_best)




## === cell 13
lr_last = None
if (
    isinstance(results, tf.keras.callbacks.History)
    and "lr" in results.history
    and len(results.history["lr"]) > 0
):
    lr_last = float(results.history["lr"][-1])

es2 = callbacks.EarlyStopping(
    patience=30, verbose=1, restore_best_weights=True, monitor="loss"
)
rlp2 = callbacks.ReduceLROnPlateau(
    factor=0.1, patience=5, min_lr=1e-12, verbose=1, monitor="loss"
)

if lr_last is not None and np.isfinite(lr_last) and lr_last > 0:
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr_last),
        loss="binary_crossentropy",
        metrics=["acc"],
        steps_per_execution=16,
    )
else:
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["acc"],
        steps_per_execution=16,
    )


def _make_full_aug_ds_uint8_on_the_fly(X_arr_u8, Y_arr_bool, batch_size):
    base = tf.data.Dataset.from_tensor_slices((X_arr_u8, Y_arr_bool))

    def _cast_and_scale(x, y):
        x = tf.cast(x, tf.float32) / 255.0
        y = tf.cast(y, tf.float32)
        return x, y

    base = base.map(_cast_and_scale, num_parallel_calls=tf.data.AUTOTUNE).cache()

    def _flip_pair(x, y):
        return tf.reverse(x, axis=[0]), tf.reverse(y, axis=[0])

    flipped = base.map(_flip_pair, num_parallel_calls=tf.data.AUTOTUNE)

    ds = base.concatenate(flipped)

    buf = min(int(2 * X_arr_u8.shape[0]), 1024)
    ds = ds.shuffle(buffer_size=buf, seed=19, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=False)

    opts = tf.data.Options()
    opts.deterministic = True
    ds = ds.with_options(opts)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


full_ds = _make_full_aug_ds_uint8_on_the_fly(X, Y, batch_size=8)

results2 = model.fit(
    full_ds, batch_size=None, epochs=80, callbacks=[es2, rlp2], verbose=2
)




## === cell 14
if False:
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(2, 1, figsize=(20, 8))
    history2 = pd.DataFrame(results2.history)
    history2[["loss"]].plot(ax=ax[0])
    history2[["acc"]].plot(ax=ax[1])
    fig.suptitle("Learning Curve (full data)", fontsize=24)
    plt.show()




## === cell 15
X_test = np.zeros(
    (len(test_ids), config.im_height, config.im_width, config.im_chan), dtype=np.uint8
)
sizes_test = np.zeros((len(test_ids), 2), dtype=np.int32)

print("Getting and resizing test images ... ")
sys.stdout.flush()

test_img_dir = os.path.join(config.path_test, "images")


def _load_test(args):
    n, id_ = args
    img = _read_gray_cv2(os.path.join(test_img_dir, id_))
    h, w = img.shape[:2]
    img_r = _resize_img_cv2(img, is_mask=False)
    return n, h, w, img_r


max_workers = min(16, (os.cpu_count() or 4))
with ThreadPoolExecutor(max_workers=max_workers) as ex:
    for n, h, w, img_r in tqdm(
        ex.map(_load_test, enumerate(test_ids)), total=len(test_ids)
    ):
        sizes_test[n, 0] = h
        sizes_test[n, 1] = w
        X_test[n, ..., 0] = img_r

print("Done!", X_test.shape)




## === cell 16
test_ds_pred = (
    tf.data.Dataset.from_tensor_slices(X_test)
    .map(lambda x: tf.cast(x, tf.float32) / 255.0, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(256)
    .prefetch(tf.data.AUTOTUNE)
)
preds_test = model.predict(test_ds_pred, verbose=0)

h0, w0 = int(sizes_test[0, 0]), int(sizes_test[0, 1])
if np.all(sizes_test[:, 0] == h0) and np.all(sizes_test[:, 1] == w0):
    preds_test_upsampled = tf.image.resize(
        preds_test, size=(h0, w0), method="area"
    ).numpy()[..., 0]
else:
    preds_test_upsampled = [None] * len(preds_test)
    for i in trange(len(preds_test)):
        h, w = int(sizes_test[i, 0]), int(sizes_test[i, 1])
        preds_test_upsampled[i] = cv2.resize(
            preds_test[i, ..., 0], (w, h), interpolation=cv2.INTER_AREA
        )




## === cell 17
get_ipython().system("rm -rf train || true")
get_ipython().system("rm -rf test || true")




## === cell 18
def RLenc(img, order="F", format=True):
    pixels = img.reshape(-1, order=order).astype(np.uint8, copy=False)
    if pixels.size == 0:
        return "" if format else []
    padded = np.empty(pixels.size + 2, dtype=np.uint8)
    padded[0] = 0
    padded[-1] = 0
    padded[1:-1] = pixels
    changes = np.flatnonzero(padded[1:] != padded[:-1]) + 1
    starts = changes[::2]
    ends = changes[1::2]
    lengths = ends - starts
    if not format:
        return list(zip(starts.tolist(), lengths.tolist()))
    if starts.size == 0:
        return ""
    s = starts.astype(str)
    l = lengths.astype(str)
    out = np.empty(starts.size * 2, dtype=object)
    out[0::2] = s
    out[1::2] = l
    return " ".join(out.tolist())


pred_dict = {}
if isinstance(preds_test_upsampled, np.ndarray):
    for i, fn in tqdm(enumerate(test_ids), total=len(test_ids)):
        pred_dict[fn[:-4]] = RLenc(
            (preds_test_upsampled[i] > threshold_best).astype(np.uint8)
        )
else:
    for i, fn in tqdm(enumerate(test_ids), total=len(test_ids)):
        pred_dict[fn[:-4]] = RLenc(
            (preds_test_upsampled[i] > threshold_best).astype(np.uint8)
        )




## === cell 19
sample_path = "../input/tgs-salt-identification-challenge/sample_submission.csv"
if not os.path.exists(sample_path):
    sample_path = "../input/sample_submission.csv"

sample = pd.read_csv(sample_path)
sample["rle_mask"] = sample["id"].map(pred_dict).fillna("")
sample.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sample.shape)
print(sample.head())
