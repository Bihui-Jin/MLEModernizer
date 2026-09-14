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

3.7

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
import os
import sys
import random
import numpy as np
import pandas as pd

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import tensorflow as tf

from skimage.transform import resize  # kept for exact upsample step later (submission)
from tqdm import tqdm

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    concatenate,
    Dropout,
)
from tensorflow.keras.optimizers import Adam

SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(max(1, os.cpu_count() // 2))
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

BASE_INPUT = "/kaggle/input/tgs-salt-identification-challenge"
if not os.path.exists(BASE_INPUT):
    alt = "/kaggle/input"
    if os.path.exists(alt):
        candidates = [
            os.path.join(alt, d)
            for d in os.listdir(alt)
            if "tgs-salt-identification-challenge" in d
        ]
        if len(candidates) > 0:
            BASE_INPUT = candidates[0]
assert os.path.exists(BASE_INPUT), f"Expected Kaggle dataset at {BASE_INPUT}"

print("TensorFlow:", tf.__version__)
print("Eager:", tf.executing_eagerly())
print("BASE_INPUT:", BASE_INPUT)



## === cell 1
test_path = os.path.join(BASE_INPUT, "test")
test_images_dir = os.path.join(test_path, "images")
test_ids = sorted(next(os.walk(test_images_dir))[2])
print(f"# of Test images: {len(test_ids)}")

IMG_SIZE = 128
N_CHANNELS = 3


def _load_png_3ch_uint8(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_png(img_bytes, channels=0)  # keep native channels
    img.set_shape([None, None, None])  # ensure rank-3 known

    img = tf.cond(
        tf.equal(tf.shape(img)[-1], 1),
        lambda: tf.repeat(img, repeats=3, axis=-1),
        lambda: img[:, :, :3],
    )
    img.set_shape([None, None, 3])
    img = tf.cast(img, tf.uint8)
    return img


def _resize_uint8_to_128(img_uint8):
    img_uint8.set_shape([None, None, 3])
    img_f = tf.cast(img_uint8, tf.float32)
    img_r = tf.image.resize(
        img_f, [IMG_SIZE, IMG_SIZE], method="bilinear", antialias=False
    )
    img_u8 = tf.cast(tf.clip_by_value(tf.round(img_r), 0.0, 255.0), tf.uint8)
    img_u8.set_shape([IMG_SIZE, IMG_SIZE, 3])
    return img_u8


test_files = [os.path.join(test_images_dir, fn) for fn in test_ids]

print("Getting and resizing test images ... ")
sys.stdout.flush()

AUTOTUNE = tf.data.AUTOTUNE

test_ds_native = tf.data.Dataset.from_tensor_slices(test_files).map(
    _load_png_3ch_uint8, num_parallel_calls=AUTOTUNE
)

sizes_test = []
for img in test_ds_native:
    sizes_test.append([int(img.shape[0]), int(img.shape[1])])

test_ds_resized = (
    tf.data.Dataset.from_tensor_slices(test_files)
    .map(
        lambda p: _resize_uint8_to_128(_load_png_3ch_uint8(p)),
        num_parallel_calls=AUTOTUNE,
    )
    .batch(128)
)

X_test = np.zeros((len(test_ids), IMG_SIZE, IMG_SIZE, N_CHANNELS), dtype=np.uint8)
offset = 0
for batch in test_ds_resized:
    b = batch.numpy()
    X_test[offset : offset + b.shape[0]] = b
    offset += b.shape[0]

print("Done!")




## === cell 2
def mean_iou(y_true, y_pred):
    y_true = tf.cast(y_true > 0.5, tf.int32)
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_t = tf.cast(y_pred > t, tf.int32)
        iou_obj = tf.keras.metrics.MeanIoU(num_classes=2)
        iou_obj.update_state(y_true, y_pred_t)
        prec.append(iou_obj.result())
    return tf.reduce_mean(tf.stack(prec), axis=0)


def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(tf.cast(y_pred, tf.float32))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)




## === cell 3
train_path = os.path.join(BASE_INPUT, "train")
train_images_dir = os.path.join(train_path, "images")
train_masks_dir = os.path.join(train_path, "masks")

train_ids = sorted(next(os.walk(train_images_dir))[2])
print(f"# of Train images: {len(train_ids)}")


def _load_mask_uint8(path):
    img_bytes = tf.io.read_file(path)
    m = tf.image.decode_png(img_bytes, channels=1)  # force single channel
    m.set_shape([None, None, 1])
    m = tf.cast(m, tf.uint8)
    return m


def _resize_mask_to_128(mask_uint8):
    mask_uint8.set_shape([None, None, 1])
    m_f = tf.cast(mask_uint8, tf.float32)
    m_r = tf.image.resize(m_f, [IMG_SIZE, IMG_SIZE], method="bilinear", antialias=False)
    m_u8 = tf.cast(tf.clip_by_value(tf.round(m_r), 0.0, 255.0), tf.uint8)
    m_bin = tf.cast(m_u8 > 127, tf.uint8)
    m_bin.set_shape([IMG_SIZE, IMG_SIZE, 1])
    return m_bin


train_img_files = [os.path.join(train_images_dir, fn) for fn in train_ids]
train_msk_files = [os.path.join(train_masks_dir, fn) for fn in train_ids]

print("Getting and resizing train images and masks ...")
sys.stdout.flush()

img_ds = tf.data.Dataset.from_tensor_slices(train_img_files).map(
    lambda p: _resize_uint8_to_128(_load_png_3ch_uint8(p)),
    num_parallel_calls=AUTOTUNE,
)
msk_ds = tf.data.Dataset.from_tensor_slices(train_msk_files).map(
    lambda p: _resize_mask_to_128(_load_mask_uint8(p)),
    num_parallel_calls=AUTOTUNE,
)

train_ds = tf.data.Dataset.zip((img_ds, msk_ds)).batch(64)

X_train = np.zeros((len(train_ids), IMG_SIZE, IMG_SIZE, N_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_SIZE, IMG_SIZE, 1), dtype=np.uint8)

offset = 0
for xb, yb in train_ds:
    xb = xb.numpy()
    yb = yb.numpy()
    bs = xb.shape[0]
    X_train[offset : offset + bs] = xb
    Y_train[offset : offset + bs] = yb
    offset += bs

print("Done!")

X_train_f = X_train.astype(np.float32) / 255.0
X_test_f = X_test.astype(np.float32) / 255.0
Y_train_f = Y_train.astype(np.float32)

idx = np.arange(len(train_ids))
np.random.shuffle(idx)
split = int(0.9 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]

X_tr, Y_tr = X_train_f[tr_idx], Y_train_f[tr_idx]
X_va, Y_va = X_train_f[va_idx], Y_train_f[va_idx]

print("Train:", X_tr.shape, Y_tr.shape, "Val:", X_va.shape, Y_va.shape)




## === cell 4
def build_unet(input_shape=(IMG_SIZE, IMG_SIZE, N_CHANNELS)):
    inputs = Input(input_shape)

    c1 = Conv2D(16, (3, 3), activation="relu", padding="same")(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation="relu", padding="same")(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation="relu", padding="same")(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation="relu", padding="same")(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation="relu", padding="same")(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation="relu", padding="same")(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation="relu", padding="same")(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation="relu", padding="same")(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation="relu", padding="same")(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation="relu", padding="same")(c5)

    u6 = UpSampling2D((2, 2))(c5)
    u6 = Conv2D(128, (2, 2), activation="relu", padding="same")(u6)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation="relu", padding="same")(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation="relu", padding="same")(c6)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = Conv2D(64, (2, 2), activation="relu", padding="same")(u7)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation="relu", padding="same")(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation="relu", padding="same")(c7)

    u8 = UpSampling2D((2, 2))(c7)
    u8 = Conv2D(32, (2, 2), activation="relu", padding="same")(u8)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation="relu", padding="same")(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation="relu", padding="same")(c8)

    u9 = UpSampling2D((2, 2))(c8)
    u9 = Conv2D(16, (2, 2), activation="relu", padding="same")(u9)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(16, (3, 3), activation="relu", padding="same")(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation="relu", padding="same")(c9)

    outputs = Conv2D(1, (1, 1), activation="sigmoid")(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model


model = build_unet()
model.compile(optimizer=Adam(1e-3), loss=dice_loss, metrics=[dice_coef])
model.summary()

BATCH_SIZE = 16
train_tfds = (
    tf.data.Dataset.from_tensor_slices((X_tr, Y_tr))
    .shuffle(buffer_size=len(X_tr), seed=SEED, reshuffle_each_iteration=True)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)
val_tfds = (
    tf.data.Dataset.from_tensor_slices((X_va, Y_va))
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

history = model.fit(train_tfds, validation_data=val_tfds, epochs=10, verbose=1)



## === cell 5
preds_val = model.predict(X_va, verbose=0)


def _dice_np(y_true, y_pred_bin, smooth=1.0):
    y_true_f = y_true.reshape(-1).astype(np.float32)
    y_pred_f = y_pred_bin.reshape(-1).astype(np.float32)
    inter = (y_true_f * y_pred_f).sum()
    return (2.0 * inter + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)


thresholds = np.round(np.arange(0.30, 0.71, 0.05), 2)
best_t = 0.5
best_score = -1.0
for t in thresholds:
    yb = (preds_val > t).astype(np.uint8)
    dices = []
    for i in range(yb.shape[0]):
        dices.append(_dice_np(Y_va[i, ..., 0], yb[i, ..., 0]))
    s = float(np.mean(dices))
    if s > best_score:
        best_score = s
        best_t = float(t)

print("Chosen threshold:", best_t, "val_mean_dice:", best_score)



## === cell 6
preds_test = model.predict(X_test_f, verbose=1)

preds_test_upsampled = []
for i in range(len(preds_test)):
    up = resize(
        np.squeeze(preds_test[i]),
        (sizes_test[i][0], sizes_test[i][1]),
        mode="constant",
        preserve_range=True,
        anti_aliasing=False,
    )
    preds_test_upsampled.append(up)

print("Preds upsampled:", len(preds_test_upsampled))




## === cell 7
def RLenc(img, order="F", format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    pixels = img.reshape(-1, order=order).astype(np.uint8)
    pads = np.concatenate([[0], pixels, [0]])
    changes = np.where(pads[1:] != pads[:-1])[0] + 1  # 1-indexed
    starts = changes[0::2]
    ends = changes[1::2]
    lengths = ends - starts
    if format:
        if len(starts) == 0:
            return ""
        out = " ".join(map(str, np.column_stack((starts, lengths)).ravel()))
        return out
    else:
        return list(zip(starts.tolist(), lengths.tolist()))


pred_dict = {
    fn[:-4]: RLenc((preds_test_upsampled[i] > best_t).astype(np.uint8))
    for i, fn in tqdm(list(enumerate(test_ids)), total=len(test_ids))
}

sample_path = os.path.join(BASE_INPUT, "sample_submission.csv")
sample = pd.read_csv(sample_path)

sub = sample.copy()
sub["rle_mask"] = sub["id"].map(pred_dict).fillna("")

out_path = "submission_02.csv"
sub.to_csv(out_path, index=False)
print("Wrote", out_path, "with shape", sub.shape)
print(sub.head())
