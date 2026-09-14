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

No external packages required in the script and installed.

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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

try:
    from google.protobuf import message_factory as _message_factory

    if not hasattr(_message_factory.MessageFactory, "GetPrototype"):

        def _fallback_get_prototype(self, descriptor):
            return None

        _message_factory.MessageFactory.GetPrototype = _fallback_get_prototype
except Exception:
    pass

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

SEED = 19
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
warnings.filterwarnings("ignore")

try:
    tf.config.threading.set_intra_op_parallelism_threads(max(1, os.cpu_count() // 2))
    tf.config.threading.set_inter_op_parallelism_threads(max(1, os.cpu_count() // 2))
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism(True)
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
train_check = os.path.join(config.path_train, "images")
test_check = os.path.join(config.path_test, "images")
if not os.path.isdir(train_check) and not os.path.isdir(
    os.path.join(config.path_train, "train", "images")
):
    os.system("unzip -q ../input/tgs-salt-identification-challenge/train.zip -d train/")
if not os.path.isdir(test_check) and not os.path.isdir(
    os.path.join(config.path_test, "test", "images")
):
    os.system("unzip -q ../input/tgs-salt-identification-challenge/test.zip -d test/")




## === cell 3
img_dir = os.path.join(config.path_train, "images")
mask_dir = os.path.join(config.path_train, "masks")
if not os.path.isdir(img_dir):
    img_dir = os.path.join(config.path_train, "train", "images")
    mask_dir = os.path.join(config.path_train, "train", "masks")

DO_PLOTS = False
if DO_PLOTS:
    random.seed(19)
    ids = random.choices(os.listdir(img_dir), k=6)
    fig = plt.figure(figsize=(20, 6))
    for j, img_name in enumerate(ids):
        q = j + 1

        img = load_img(os.path.join(img_dir, img_name))
        img_mask = load_img(os.path.join(mask_dir, img_name))

        plt.subplot(2, 6, q * 2 - 1)
        plt.imshow(img)
        plt.subplot(2, 6, q * 2)
        plt.imshow(img_mask)
    fig.suptitle("Sample Images", fontsize=24)




## === cell 4
train_images_dir = os.path.join(config.path_train, "images")
if not os.path.isdir(train_images_dir):
    train_images_dir = os.path.join(config.path_train, "train", "images")

test_images_dir = os.path.join(config.path_test, "images")
if not os.path.isdir(test_images_dir):
    test_images_dir = os.path.join(config.path_test, "test", "images")

train_ids = next(os.walk(train_images_dir))[2]
test_ids = next(os.walk(test_images_dir))[2]

train_ids = sorted(train_ids)
test_ids = sorted(test_ids)




## === cell 5
print("Preparing tf.data pipelines for train/val ... ")
sys.stdout.flush()

_train_img_dir = os.path.join(config.path_train, "images")
_train_msk_dir = os.path.join(config.path_train, "masks")
if not os.path.isdir(_train_img_dir):
    _train_img_dir = os.path.join(config.path_train, "train", "images")
    _train_msk_dir = os.path.join(config.path_train, "train", "masks")

train_files = np.array(train_ids, dtype=object)

n_total = len(train_files) * 2
n_val = int(round(n_total * 0.1))
n_train = n_total - n_val

n_orig = len(train_files)


def _read_resize_pair_py(img_path, msk_path):
    img = cv2.imread(img_path.decode("utf-8"), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(img_path.decode("utf-8"))
    img = cv2.resize(
        img, (config.im_width, config.im_height), interpolation=cv2.INTER_LINEAR
    )

    msk = cv2.imread(msk_path.decode("utf-8"), cv2.IMREAD_GRAYSCALE)
    if msk is None:
        raise FileNotFoundError(msk_path.decode("utf-8"))
    msk = cv2.resize(
        msk, (config.im_width, config.im_height), interpolation=cv2.INTER_NEAREST
    )
    msk = (msk > 0).astype(
        np.uint8
    )  # keep mask binary; will cast to float32/bool later as needed

    img = img[..., None]
    msk = msk[..., None]
    return img, msk


def _tf_read_resize_pair(img_path, msk_path):
    img, msk = tf.numpy_function(
        _read_resize_pair_py, [img_path, msk_path], Tout=[tf.uint8, tf.uint8]
    )
    img.set_shape([config.im_height, config.im_width, config.im_chan])
    msk.set_shape([config.im_height, config.im_width, 1])
    return img, tf.cast(msk, tf.bool)


def _make_ds(indices):
    idx = tf.convert_to_tensor(indices, dtype=tf.int32)

    def _to_paths_and_flip(i):
        flip = i >= n_orig
        base_i = tf.where(flip, i - n_orig, i)
        fn = tf.gather(train_files, base_i)
        img_path = tf.strings.join([_train_img_dir, "/", fn])
        msk_path = tf.strings.join([_train_msk_dir, "/", fn])
        return img_path, msk_path, flip

    ds = tf.data.Dataset.from_tensor_slices(idx).map(
        _to_paths_and_flip, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
    )
    ds = ds.map(
        lambda img_p, msk_p, flip: (_tf_read_resize_pair(img_p, msk_p), flip),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )

    def _apply_flip(pair, flip):
        img, msk = pair
        img = tf.cond(flip, lambda: tf.image.flip_left_right(img), lambda: img)
        msk = tf.cond(flip, lambda: tf.image.flip_left_right(msk), lambda: msk)
        return img, msk

    ds = ds.map(_apply_flip, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    return ds


all_indices = np.arange(n_total, dtype=np.int32)
tr_indices = all_indices[:n_train]
va_indices = all_indices[n_train:]

options = tf.data.Options()
options.experimental_deterministic = True

batch_size = 8

shuffle_buf = min(n_train, 2048)

ds_tr = (
    _make_ds(tr_indices)
    .with_options(options)
    .cache()
    .shuffle(buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True)
    .batch(batch_size, drop_remainder=False)
    .prefetch(tf.data.AUTOTUNE)
)
ds_va = (
    _make_ds(va_indices)
    .with_options(options)
    .cache()
    .batch(batch_size, drop_remainder=False)
    .prefetch(tf.data.AUTOTUNE)
)




## === cell 6
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




## === cell 7
if DO_PLOTS:
    utils.plot_model(model, expand_nested=True, show_shapes=True)




## === cell 8
es = callbacks.EarlyStopping(patience=30, verbose=1, restore_best_weights=True)
rlp = callbacks.ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-12, verbose=1)

results = model.fit(
    ds_tr,
    validation_data=ds_va,
    epochs=300,
    callbacks=[es, rlp],
    verbose=2,
)




## === cell 9
if DO_PLOTS:
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(2, 1, figsize=(20, 8))
    history = pd.DataFrame(results.history)
    history[["loss", "val_loss"]].plot(ax=ax[0])
    history[["acc", "val_acc"]].plot(ax=ax[1])
    fig.suptitle("Learning Curve", fontsize=24)




## === cell 10
print("Preparing tf.data pipeline for test ... ")
sys.stdout.flush()

_test_img_dir = os.path.join(config.path_test, "images")
if not os.path.isdir(_test_img_dir):
    _test_img_dir = os.path.join(config.path_test, "test", "images")

test_files = np.array(test_ids, dtype=object)


def _read_resize_img_py(img_path):
    img = cv2.imread(img_path.decode("utf-8"), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(img_path.decode("utf-8"))
    img = cv2.resize(
        img, (config.im_width, config.im_height), interpolation=cv2.INTER_LINEAR
    )
    img = img[..., None]
    return img


def _tf_read_resize_img(img_path):
    img = tf.numpy_function(_read_resize_img_py, [img_path], Tout=tf.uint8)
    img.set_shape([config.im_height, config.im_width, config.im_chan])
    return img


ds_test = (
    tf.data.Dataset.from_tensor_slices(
        tf.strings.join([_test_img_dir, "/", test_files])
    )
    .with_options(options)
    .map(_tf_read_resize_img, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    .batch(32, drop_remainder=False)
    .prefetch(tf.data.AUTOTUNE)
)




## === cell 11
preds_test = model.predict(ds_test, verbose=1)  # shape: (N, 128, 128, 1)

preds_test_upsampled = []
for i in range(preds_test.shape[0]):
    p = preds_test[i, :, :, 0]
    p_up = cv2.resize(p, (101, 101), interpolation=cv2.INTER_LINEAR)
    preds_test_upsampled.append(p_up)




## === cell 12
os.system("rm -rf train")
os.system("rm -rf test")




## === cell 13
def RLenc(img, order="F", format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    if order != "F":
        bytes_ = img.reshape(img.shape[0] * img.shape[1], order=order)
    else:
        bytes_ = np.asarray(img, dtype=np.uint8).reshape(-1, order="F")

    if bytes_.size == 0:
        return "" if format else []

    padded = np.concatenate(([0], bytes_, [0]))
    changes = np.flatnonzero(padded[1:] != padded[:-1]) + 1
    runs = changes.reshape(-1, 2)
    runs[:, 1] = runs[:, 1] - runs[:, 0]

    if format:
        if runs.size == 0:
            return ""
        return " ".join(map(str, runs.reshape(-1).tolist()))
    else:
        return [tuple(x) for x in runs.tolist()]


pred_dict = {}
for i, fn in tqdm(list(enumerate(test_ids)), total=len(test_ids)):
    pred_dict[fn[:-4]] = RLenc(np.round(preds_test_upsampled[i]))




## === cell 14
sub = pd.DataFrame.from_dict(pred_dict, orient="index")
sub.index.names = ["id"]
sub.columns = ["rle_mask"]
sub.to_csv("submission.csv")
