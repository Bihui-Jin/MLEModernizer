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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

os.environ.pop("TF_XLA_FLAGS", None)
os.environ.pop("TF_XLA_ENABLE_XLA_DEVICES", None)
os.environ.pop("XLA_FLAGS", None)
os.environ.pop("TF_XLA_PERSISTENT_CACHE_DIR", None)

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from tqdm.auto import tqdm

import cv2
from skimage.transform import resize  # kept for strict-compat fallback paths if needed

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import models, Input, layers, callbacks, utils

SEED = 19
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    cv2.setNumThreads(0)
except Exception:
    pass

from concurrent.futures import ThreadPoolExecutor

N_WORKERS = max(2, min(8, (os.cpu_count() or 4)))

print("TF version:", tf.__version__)
print("OpenCV version:", cv2.__version__)
print("N_WORKERS:", N_WORKERS)




## === cell 1
class config:
    im_width = 128
    im_height = 128
    im_chan = 1
    path_train = "train/"
    path_test = "test/"




## === cell 2
train_zip_candidates = [
    "/kaggle/input/tgs-salt-identification-challenge/train.zip",
    "/kaggle/data/tgs-salt-identification-challenge/train.zip",
    "../input/tgs-salt-identification-challenge/train.zip",
]
test_zip_candidates = [
    "/kaggle/input/tgs-salt-identification-challenge/test.zip",
    "/kaggle/data/tgs-salt-identification-challenge/test.zip",
    "../input/tgs-salt-identification-challenge/test.zip",
]

train_zip = next((p for p in train_zip_candidates if os.path.exists(p)), None)
test_zip = next((p for p in test_zip_candidates if os.path.exists(p)), None)

if train_zip is None or test_zip is None:
    raise FileNotFoundError(
        f"Could not find train.zip/test.zip. Tried: {train_zip_candidates} and {test_zip_candidates}"
    )

os.makedirs("train", exist_ok=True)
os.makedirs("test", exist_ok=True)


def _dir_has_pngs(p):
    return os.path.isdir(p) and any(fn.endswith(".png") for fn in os.listdir(p))


need_unzip_train = not (_dir_has_pngs("train/images") and _dir_has_pngs("train/masks"))
need_unzip_test = not _dir_has_pngs("test/images")

if need_unzip_train:
    os.system(f"unzip -q -o '{train_zip}' -d train/")
if need_unzip_test:
    os.system(f"unzip -q -o '{test_zip}' -d test/")

if not os.path.isdir("train/images") or not os.path.isdir("train/masks"):
    if os.path.isdir("train/train/images") and os.path.isdir("train/train/masks"):
        os.makedirs("train/images", exist_ok=True)
        os.makedirs("train/masks", exist_ok=True)
        if not _dir_has_pngs("train/images"):
            os.system("cp -r train/train/images/* train/images/ 2>/dev/null")
        if not _dir_has_pngs("train/masks"):
            os.system("cp -r train/train/masks/* train/masks/ 2>/dev/null")

if not os.path.isdir("test/images"):
    if os.path.isdir("test/test/images"):
        os.makedirs("test/images", exist_ok=True)
        if not _dir_has_pngs("test/images"):
            os.system("cp -r test/test/images/* test/images/ 2>/dev/null")

assert os.path.isdir("train/images"), "train/images not found after unzip"
assert os.path.isdir("train/masks"), "train/masks not found after unzip"
assert os.path.isdir("test/images"), "test/images not found after unzip"

print("Unzip OK:")
print(" train/images:", len(os.listdir("train/images")))
print(" train/masks :", len(os.listdir("train/masks")))
print(" test/images :", len(os.listdir("test/images")))



## === cell 3
pass



## === cell 4
train_ids = next(os.walk(config.path_train + "images"))[2]
test_ids = next(os.walk(config.path_test + "images"))[2]

train_ids = sorted(train_ids)
test_ids = sorted(test_ids)

print("n_train:", len(train_ids), "n_test:", len(test_ids))



## === cell 5
cache_train_npz = f"cache_train_{config.im_height}x{config.im_width}_seed{SEED}.npz"

if os.path.exists(cache_train_npz):
    d = np.load(cache_train_npz, allow_pickle=False)
    X = d["X"]
    Y = d["Y"]
    print("Loaded cached train arrays:", X.shape, Y.shape)
else:
    X = np.empty(
        (len(train_ids), config.im_height, config.im_width, config.im_chan),
        dtype=np.uint8,
    )
    Y = np.empty((len(train_ids), config.im_height, config.im_width, 1), dtype=np.uint8)

    print("Getting and resizing train images and masks ... ")
    sys.stdout.flush()

    h, w = config.im_height, config.im_width
    img_dir = os.path.join(config.path_train, "images")
    msk_dir = os.path.join(config.path_train, "masks")

    def _load_pair(n_id):
        n, id_ = n_id
        img_path = os.path.join(img_dir, id_)
        msk_path = os.path.join(msk_dir, id_)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(img_path)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

        msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        if msk is None:
            raise FileNotFoundError(msk_path)
        msk = cv2.resize(msk, (w, h), interpolation=cv2.INTER_NEAREST)
        return n, img, (msk > 127).astype(np.uint8)

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        for n, img, msku8 in tqdm(
            ex.map(_load_pair, enumerate(train_ids)), total=len(train_ids)
        ):
            X[n, :, :, 0] = img
            Y[n, :, :, 0] = msku8

    np.savez_compressed(cache_train_npz, X=X, Y=Y)
    print("Saved cache:", cache_train_npz)

print("Done!")
print("X shape:", X.shape)
print("Y shape:", Y.shape)



## === cell 6
split = int(0.9 * len(X))
X_train0 = X[:split]
Y_train0 = Y[:split]
X_eval = X[split:]
Y_eval = Y[split:]

n0 = X_train0.shape[0]
X_train = np.empty((n0 * 3, *X_train0.shape[1:]), dtype=np.uint8)
Y_train = np.empty((n0 * 3, *Y_train0.shape[1:]), dtype=np.uint8)

X_train[:n0] = X_train0
Y_train[:n0] = Y_train0
X_train[n0 : 2 * n0] = X_train0[:, :, ::-1, :]
Y_train[n0 : 2 * n0] = Y_train0[:, :, ::-1, :]
X_train[2 * n0 :] = X_train0[:, ::-1, :, :]
Y_train[2 * n0 :] = Y_train0[:, ::-1, :, :]

del X, Y, X_train0, Y_train0

X_eval = X_eval.astype(np.uint8, copy=False)
Y_train = Y_train.astype(np.float32, copy=False)
Y_eval = Y_eval.astype(np.float32, copy=False)

print("X train shape:", X_train.shape, "X eval shape:", X_eval.shape)
print("Y train shape:", Y_train.shape, "Y eval shape:", Y_eval.shape)




## === cell 7
def BatchActivate(x):
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def convolution_block(
    x, filters, size, strides=(1, 1), padding="same", activation=True
):
    x = layers.Conv2D(filters, size, strides=strides, padding=padding)(x)
    if activation == True:
        x = BatchActivate(x)
    return x


def residual_block(blockInput, num_filters=16, batch_activate=False):
    x = BatchActivate(blockInput)
    x = convolution_block(x, num_filters, (3, 3))
    x = convolution_block(x, num_filters, (3, 3), activation=False)
    x = layers.Add()([x, blockInput])
    if batch_activate:
        x = BatchActivate(x)
    return x




## === cell 8
def build_model(input_layer, start_neurons, DropoutRatio=0.5):
    scaled = layers.Lambda(lambda x: x / 255)(input_layer)

    conv1 = layers.Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(
        scaled
    )
    conv1 = residual_block(conv1, start_neurons * 1)
    conv1 = residual_block(conv1, start_neurons * 1, True)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    pool1 = layers.Dropout(DropoutRatio / 2)(pool1)

    conv2 = layers.Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(
        pool1
    )
    conv2 = residual_block(conv2, start_neurons * 2)
    conv2 = residual_block(conv2, start_neurons * 2, True)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    pool2 = layers.Dropout(DropoutRatio)(pool2)

    conv3 = layers.Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(
        pool2
    )
    conv3 = residual_block(conv3, start_neurons * 4)
    conv3 = residual_block(conv3, start_neurons * 4, True)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)
    pool3 = layers.Dropout(DropoutRatio)(pool3)

    conv4 = layers.Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(
        pool3
    )
    conv4 = residual_block(conv4, start_neurons * 8)
    conv4 = residual_block(conv4, start_neurons * 8, True)
    pool4 = layers.MaxPooling2D((2, 2))(conv4)
    pool4 = layers.Dropout(DropoutRatio)(pool4)

    convm = layers.Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(
        pool4
    )
    convm = residual_block(convm, start_neurons * 16)
    convm = residual_block(convm, start_neurons * 16, True)

    deconv4 = layers.Conv2DTranspose(
        start_neurons * 8, (3, 3), strides=(2, 2), padding="same"
    )(convm)
    uconv4 = layers.concatenate([deconv4, conv4])
    uconv4 = layers.Dropout(DropoutRatio)(uconv4)

    uconv4 = layers.Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(
        uconv4
    )
    uconv4 = residual_block(uconv4, start_neurons * 8)
    uconv4 = residual_block(uconv4, start_neurons * 8, True)

    deconv3 = layers.Conv2DTranspose(
        start_neurons * 4, (3, 3), strides=(2, 2), padding="same"
    )(uconv4)
    uconv3 = layers.concatenate([deconv3, conv3])
    uconv3 = layers.Dropout(DropoutRatio)(uconv3)

    uconv3 = layers.Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(
        uconv3
    )
    uconv3 = residual_block(uconv3, start_neurons * 4)
    uconv3 = residual_block(uconv3, start_neurons * 4, True)

    deconv2 = layers.Conv2DTranspose(
        start_neurons * 2, (3, 3), strides=(2, 2), padding="same"
    )(uconv3)
    uconv2 = layers.concatenate([deconv2, conv2])

    uconv2 = layers.Dropout(DropoutRatio)(uconv2)
    uconv2 = layers.Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(
        uconv2
    )
    uconv2 = residual_block(uconv2, start_neurons * 2)
    uconv2 = residual_block(uconv2, start_neurons * 2, True)

    deconv1 = layers.Conv2DTranspose(
        start_neurons * 1, (3, 3), strides=(2, 2), padding="same"
    )(uconv2)
    uconv1 = layers.concatenate([deconv1, conv1])

    uconv1 = layers.Dropout(DropoutRatio)(uconv1)
    uconv1 = layers.Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(
        uconv1
    )
    uconv1 = residual_block(uconv1, start_neurons * 1)
    uconv1 = residual_block(uconv1, start_neurons * 1, True)

    output_layer_noActi = layers.Conv2D(1, (1, 1), padding="same", activation=None)(
        uconv1
    )
    output_layer = layers.Activation("sigmoid")(output_layer_noActi)

    return output_layer


input_layer = Input((config.im_height, config.im_width, config.im_chan))
output_layer = build_model(input_layer, 16)

model = models.Model(input_layer, output_layer)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["acc"],
    jit_compile=False,
    steps_per_execution=64,
)

model.summary()



## === cell 9
try:
    pass
except Exception as e:
    print("plot_model skipped:", repr(e))



## === cell 10
es = callbacks.EarlyStopping(patience=30, verbose=1, restore_best_weights=True)
rlp = callbacks.ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-12, verbose=1)

BATCH_SIZE = 8

options = tf.data.Options()
options.deterministic = True
try:
    options.threading.private_threadpool_size = 0
    options.threading.max_intra_op_parallelism = 0
except Exception:
    pass

shuffle_buf = min(len(X_train), 4096)

base_train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, Y_train)).with_options(options).cache()
)

train_ds = (
    base_train_ds.shuffle(
        buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True
    )
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(tf.data.AUTOTUNE)
)

eval_ds = (
    tf.data.Dataset.from_tensor_slices((X_eval, Y_eval))
    .with_options(options)
    .cache()
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(tf.data.AUTOTUNE)
)

results = model.fit(
    train_ds,
    validation_data=eval_ds,
    epochs=300,
    callbacks=[es, rlp],
    verbose=2,
)



## === cell 11
pass




## === cell 12
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




## === cell 13
eval_pred_ds = (
    tf.data.Dataset.from_tensor_slices(X_eval)
    .batch(64, drop_remainder=False)
    .prefetch(tf.data.AUTOTUNE)
)
preds_eval = model.predict(eval_pred_ds, verbose=0)

thresholds = np.linspace(0, 1, 50, dtype=np.float32)

y_true_bin = (Y_eval[..., 0] > 0.5).astype(np.uint8, copy=False)
preds_eval_f = preds_eval[..., 0].astype(np.float32, copy=False)

scores = np.empty((thresholds.shape[0],), dtype=np.float32)
for i, thr in enumerate(thresholds):
    y_pred_bin = (preds_eval_f > thr).astype(np.uint8, copy=False)
    scores[i] = iou_metric_batch(y_true_bin, y_pred_bin)

threshold_best_index = int(np.argmax(scores))
iou_best = float(scores[threshold_best_index])
threshold_best = float(thresholds[threshold_best_index])

print("Best threshold:", threshold_best, "Best IoU metric:", iou_best)



## === cell 14
cache_test_npz = f"cache_test_{config.im_height}x{config.im_width}_seed{SEED}.npz"

if os.path.exists(cache_test_npz):
    d = np.load(cache_test_npz, allow_pickle=False)
    X_test = d["X_test"]
    sizes_test = d["sizes_test"]
    print("Loaded cached test arrays:", X_test.shape, sizes_test.shape)
else:
    X_test = np.empty(
        (len(test_ids), config.im_height, config.im_width, config.im_chan),
        dtype=np.uint8,
    )
    sizes_test = np.empty((len(test_ids), 2), dtype=np.int32)

    print("Getting and resizing test images ... ")
    sys.stdout.flush()

    test_img_dir = os.path.join(config.path_test, "images")
    h, w = config.im_height, config.im_width

    def _load_test(n_id):
        n, id_ = n_id
        img_path = os.path.join(test_img_dir, id_)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(img_path)
        sh, sw = img.shape[0], img.shape[1]
        img_r = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        return n, sh, sw, img_r

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        for n, sh, sw, img_r in tqdm(
            ex.map(_load_test, enumerate(test_ids)), total=len(test_ids)
        ):
            sizes_test[n, 0] = sh
            sizes_test[n, 1] = sw
            X_test[n, :, :, 0] = img_r

    np.savez_compressed(cache_test_npz, X_test=X_test, sizes_test=sizes_test)
    print("Saved cache:", cache_test_npz)

print("Done! X_test shape:", X_test.shape)



## === cell 15
test_pred_ds = (
    tf.data.Dataset.from_tensor_slices(X_test)
    .batch(64, drop_remainder=False)
    .prefetch(tf.data.AUTOTUNE)
)
preds_test = model.predict(test_pred_ds, verbose=0)

preds_test_upsampled = [None] * len(preds_test)


def _upsample_one(i):
    ph, pw = int(sizes_test[i, 0]), int(sizes_test[i, 1])
    up = cv2.resize(preds_test[i, :, :, 0], (pw, ph), interpolation=cv2.INTER_LINEAR)
    return i, up


with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
    for i, up in tqdm(
        ex.map(_upsample_one, range(len(preds_test))), total=len(preds_test)
    ):
        preds_test_upsampled[i] = up



## === cell 16
pass




## === cell 17
def RLenc(img, order="F", format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    """
    pixels = img.reshape(-1, order=order).astype(np.uint8, copy=False)
    padded = np.concatenate(([0], pixels, [0]))
    changes = np.flatnonzero(padded[1:] != padded[:-1]) + 1
    runs = changes[::2]
    lengths = changes[1::2] - runs
    if format:
        if runs.size == 0:
            return ""
        return " ".join(map(str, np.column_stack((runs, lengths)).ravel()))
    else:
        return list(zip(runs.tolist(), lengths.tolist()))


pred_dict = {}
thr = float(threshold_best)

for i, fn in tqdm(enumerate(test_ids), total=len(test_ids)):
    mask = (preds_test_upsampled[i] > thr).astype(np.uint8, copy=False)
    pred_dict[fn[:-4]] = RLenc(mask)



## === cell 18
sample_path_candidates = [
    "/kaggle/input/tgs-salt-identification-challenge/sample_submission.csv",
    "/kaggle/data/tgs-salt-identification-challenge/sample_submission.csv",
    "../input/tgs-salt-identification-challenge/sample_submission.csv",
]
sample_path = next((p for p in sample_path_candidates if os.path.exists(p)), None)
if sample_path is None:
    sample_path = "sample_submission.csv"

sample = pd.read_csv(sample_path)
sample["rle_mask"] = sample["id"].map(pred_dict).fillna("")
sample.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sample.shape)
print(sample.head())
