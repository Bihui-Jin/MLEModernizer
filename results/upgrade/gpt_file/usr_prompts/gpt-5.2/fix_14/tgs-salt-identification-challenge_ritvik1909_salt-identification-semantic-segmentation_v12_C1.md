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

# 5. Target score

0.77948

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, sys, random, warnings, math

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.auto import tqdm, trange

import cv2
from tensorflow.keras.preprocessing.image import load_img

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import (
    models,
    Input,
    layers,
    callbacks,
    utils,
    applications,
)

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
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

print("TF:", tf.__version__)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
class config:
    im_width = 128
    im_height = 128
    im_chan = 1
    path_train = "train/"
    path_test = "test/"




## === cell 2
def _find_input_file(rel_path_candidates):
    for p in rel_path_candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Could not find any of: {rel_path_candidates}")


train_zip = _find_input_file(
    [
        "../input/tgs-salt-identification-challenge/train.zip",
        "/kaggle/input/tgs-salt-identification-challenge/train.zip",
    ]
)
test_zip = _find_input_file(
    [
        "../input/tgs-salt-identification-challenge/test.zip",
        "/kaggle/input/tgs-salt-identification-challenge/test.zip",
    ]
)

import zipfile

os.makedirs("train", exist_ok=True)
os.makedirs("test", exist_ok=True)


def _looks_extracted_fast(root):
    if os.path.isdir(os.path.join(root, "images")):
        return True
    if os.path.isdir(os.path.join(root, "train", "images")):
        return True
    if os.path.isdir(os.path.join(root, "test", "images")):
        return True
    try:
        for name in os.listdir(root):
            if name.endswith(".png"):
                return True
            p = os.path.join(root, name)
            if os.path.isdir(p) and os.path.isdir(os.path.join(p, "images")):
                return True
    except FileNotFoundError:
        return False
    return False


if not _looks_extracted_fast("train"):
    with zipfile.ZipFile(train_zip, "r") as z:
        z.extractall("train")
if not _looks_extracted_fast("test"):
    with zipfile.ZipFile(test_zip, "r") as z:
        z.extractall("test")


def _ensure_expected_structure(root, expected_subdir="images"):
    if os.path.isdir(os.path.join(root, expected_subdir)):
        return root
    for name in os.listdir(root):
        cand = os.path.join(root, name)
        if os.path.isdir(cand) and os.path.isdir(os.path.join(cand, expected_subdir)):
            return cand
    for dirpath, dirnames, _ in os.walk(root):
        if expected_subdir in dirnames:
            return dirpath
    raise FileNotFoundError(f"Could not find '{expected_subdir}' under '{root}'")


train_root = _ensure_expected_structure("train", "images")
test_root = _ensure_expected_structure("test", "images")

config.path_train = train_root + ("" if train_root.endswith("/") else "/")
config.path_test = test_root + ("" if test_root.endswith("/") else "/")

print("Using train root:", config.path_train)
print("Using test root:", config.path_test)
print(
    "Train images dir exists:", os.path.isdir(os.path.join(config.path_train, "images"))
)
print(
    "Train masks dir exists:", os.path.isdir(os.path.join(config.path_train, "masks"))
)
print(
    "Test images dir exists:", os.path.isdir(os.path.join(config.path_test, "images"))
)



## === cell 3
img_dir = os.path.join(config.path_train, "images")
mask_dir = os.path.join(config.path_train, "masks")

random.seed(19)
all_imgs = sorted(os.listdir(img_dir))
k = min(6, len(all_imgs))
ids = random.choices(all_imgs, k=k) if k > 0 else []

SHOW_PLOTS = False

if SHOW_PLOTS and k > 0:
    fig = plt.figure(figsize=(20, 6))
    for j, img_name in enumerate(ids):
        img = load_img(os.path.join(img_dir, img_name))
        img_mask = load_img(os.path.join(mask_dir, img_name))
        plt.subplot(2, 6, j * 2 + 1)
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.subplot(2, 6, j * 2 + 2)
        plt.imshow(img_mask, cmap="gray")
        plt.axis("off")
    fig.suptitle("Sample Images", fontsize=24)
    plt.show()




## === cell 4
def _strip_png(fn):
    return fn[:-4] if fn.lower().endswith(".png") else fn


train_filenames = sorted(next(os.walk(os.path.join(config.path_train, "images")))[2])
test_filenames = sorted(next(os.walk(os.path.join(config.path_test, "images")))[2])

train_ids = [_strip_png(f) for f in train_filenames]
test_ids = [_strip_png(f) for f in test_filenames]

print("Num train:", len(train_ids), "Num test:", len(test_ids))
print("First train id:", train_ids[0] if train_ids else None)
print("First test id:", test_ids[0] if test_ids else None)




## === cell 5
def _read_gray_uint8(path):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(path)
    return im


def _resize_gray_to128(im, h, w):
    return cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)


def _resize_mask_to128(im, h, w):
    return cv2.resize(im, (w, h), interpolation=cv2.INTER_NEAREST)


CACHE_DIR = "cache_arrays"
os.makedirs(CACHE_DIR, exist_ok=True)
cache_train_npz = os.path.join(
    CACHE_DIR,
    f"train_XY_{len(train_ids)}_{config.im_height}x{config.im_width}_seed19.npz",
)

if os.path.exists(cache_train_npz):
    d = np.load(cache_train_npz, allow_pickle=False)
    X = d["X"]
    Y = d["Y"]
    print("Loaded cached train arrays:", cache_train_npz, X.shape, Y.shape)
else:
    X = np.zeros(
        (len(train_ids), config.im_height, config.im_width, config.im_chan),
        dtype=np.uint8,
    )
    Y = np.zeros((len(train_ids), config.im_height, config.im_width, 1), dtype=bool)

    print("Getting and resizing train images and masks ... ")
    sys.stdout.flush()

    img_base = os.path.join(config.path_train, "images")
    mask_base = os.path.join(config.path_train, "masks")
    _join = os.path.join
    _read = _read_gray_uint8
    _r_img = _resize_gray_to128
    _r_msk = _resize_mask_to128

    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        im = _read(_join(img_base, id_ + ".png"))
        im = _r_img(im, config.im_height, config.im_width)
        X[n, :, :, 0] = im

        m = _read(_join(mask_base, id_ + ".png"))
        m = _r_msk(m, config.im_height, config.im_width)
        Y[n, :, :, 0] = m > 127

    np.savez_compressed(cache_train_npz, X=X, Y=Y)
    print("Saved cached train arrays:", cache_train_npz)

print("Done!")
print("X shape:", X.shape)
print("Y shape:", Y.shape)



## === cell 6
rng = np.random.RandomState(19)
idx = np.arange(len(X))
rng.shuffle(idx)
X = X[idx]
Y = Y[idx]

split = int(0.9 * len(X))
X_train = X[:split]
Y_train = Y[:split]
X_eval = X[split:]
Y_eval = Y[split:]

del X, Y

print("X train shape:", X_train.shape, "X eval shape:", X_eval.shape)
print("Y train shape:", Y_train.shape, "Y eval shape:", Y_eval.shape)




## === cell 7
def convolution_block(
    x, filters, size, strides=(1, 1), padding="same", activation=True
):
    x = layers.Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    if activation == True:
        x = layers.LeakyReLU(alpha=0.1)(x)
    return x


def residual_block(blockInput, num_filters=16):
    x = layers.LeakyReLU(alpha=0.1)(blockInput)
    x = layers.BatchNormalization()(x)
    blockInput = layers.BatchNormalization()(blockInput)
    x = convolution_block(x, num_filters, (3, 3))
    x = convolution_block(x, num_filters, (3, 3), activation=False)
    x = layers.Add()([x, blockInput])
    return x




## === cell 8
def UXception(input_shape=(None, None, 3)):

    backbone = applications.Xception(
        input_shape=input_shape, weights="imagenet", include_top=False
    )

    backbone.trainable = False

    input_layer = backbone.input
    start_neurons = 16

    conv4 = backbone.layers[121].output
    conv4 = layers.LeakyReLU(alpha=0.1)(conv4)
    pool4 = layers.MaxPooling2D((2, 2))(conv4)
    pool4 = layers.Dropout(0.1)(pool4)

    convm = layers.Conv2D(start_neurons * 32, (3, 3), activation=None, padding="same")(
        pool4
    )
    convm = residual_block(convm, start_neurons * 32)
    convm = residual_block(convm, start_neurons * 32)
    convm = layers.LeakyReLU(alpha=0.1)(convm)

    deconv4 = layers.Conv2DTranspose(
        start_neurons * 16, (3, 3), strides=(2, 2), padding="same"
    )(convm)
    uconv4 = layers.concatenate([deconv4, conv4])
    uconv4 = layers.Dropout(0.1)(uconv4)

    uconv4 = layers.Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(
        uconv4
    )
    uconv4 = residual_block(uconv4, start_neurons * 16)
    uconv4 = residual_block(uconv4, start_neurons * 16)
    uconv4 = layers.LeakyReLU(alpha=0.1)(uconv4)

    deconv3 = layers.Conv2DTranspose(
        start_neurons * 8, (3, 3), strides=(2, 2), padding="same"
    )(uconv4)
    conv3 = backbone.layers[31].output
    uconv3 = layers.concatenate([deconv3, conv3])
    uconv3 = layers.Dropout(0.1)(uconv3)

    uconv3 = layers.Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(
        uconv3
    )
    uconv3 = residual_block(uconv3, start_neurons * 8)
    uconv3 = residual_block(uconv3, start_neurons * 8)
    uconv3 = layers.LeakyReLU(alpha=0.1)(uconv3)

    deconv2 = layers.Conv2DTranspose(
        start_neurons * 4, (3, 3), strides=(2, 2), padding="same"
    )(uconv3)
    conv2 = backbone.layers[21].output
    conv2 = layers.ZeroPadding2D(((1, 0), (1, 0)))(conv2)
    uconv2 = layers.concatenate([deconv2, conv2])

    uconv2 = layers.Dropout(0.1)(uconv2)
    uconv2 = layers.Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(
        uconv2
    )
    uconv2 = residual_block(uconv2, start_neurons * 4)
    uconv2 = residual_block(uconv2, start_neurons * 4)
    uconv2 = layers.LeakyReLU(alpha=0.1)(uconv2)

    deconv1 = layers.Conv2DTranspose(
        start_neurons * 2, (3, 3), strides=(2, 2), padding="same"
    )(uconv2)
    conv1 = backbone.layers[11].output
    conv1 = layers.ZeroPadding2D(((3, 0), (3, 0)))(conv1)
    uconv1 = layers.concatenate([deconv1, conv1])

    uconv1 = layers.Dropout(0.1)(uconv1)
    uconv1 = layers.Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(
        uconv1
    )
    uconv1 = residual_block(uconv1, start_neurons * 2)
    uconv1 = residual_block(uconv1, start_neurons * 2)
    uconv1 = layers.LeakyReLU(alpha=0.1)(uconv1)

    uconv0 = layers.Conv2DTranspose(
        start_neurons * 1, (3, 3), strides=(2, 2), padding="same"
    )(uconv1)
    uconv0 = layers.Dropout(0.1)(uconv0)
    uconv0 = layers.Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(
        uconv0
    )
    uconv0 = residual_block(uconv0, start_neurons * 1)
    uconv0 = residual_block(uconv0, start_neurons * 1)
    uconv0 = layers.LeakyReLU(alpha=0.1)(uconv0)

    uconv0 = layers.Dropout(0.1 / 2)(uconv0)
    output_layer = layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(
        uconv0
    )

    model = models.Model(input_layer, output_layer)

    return model




## === cell 9
input_layer = Input(shape=(config.im_height, config.im_width, 1))
scaled = layers.Lambda(lambda x: x / 255)(input_layer)
x = layers.Conv2D(3, 1, activation="relu", padding="same")(scaled)
output_layer = UXception(input_shape=(config.im_height, config.im_width, 3))(x)

model = models.Model(input_layer, output_layer)

model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["acc"], jit_compile=True
)
model.summary()



## === cell 10
if SHOW_PLOTS:
    try:
        utils.plot_model(model, expand_nested=True, show_shapes=True)
    except Exception as e:
        print("plot_model skipped:", repr(e))



## === cell 11
es = callbacks.EarlyStopping(patience=30, verbose=1, restore_best_weights=True)
rlp = callbacks.ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-12, verbose=1)

BATCH_SIZE = 8

base_train_steps = int(math.ceil(len(X_train) / BATCH_SIZE))
train_steps = int(math.ceil((3 * len(X_train)) / BATCH_SIZE))  # one pass over 3x aug
eval_steps = int(math.ceil(len(X_eval) / BATCH_SIZE))

opts = tf.data.Options()
opts.experimental_deterministic = True  # keep determinism


def _cast_xy_np(x, y):
    return tf.cast(x, tf.float32), tf.cast(y, tf.bool)


def _augment_threeways(x, y):
    x0, y0 = x, y
    x1, y1 = tf.reverse(x, axis=[1]), tf.reverse(y, axis=[1])  # width flip
    x2, y2 = tf.reverse(x, axis=[0]), tf.reverse(y, axis=[0])  # height flip
    return tf.data.Dataset.from_tensor_slices(((x0, y0), (x1, y1), (x2, y2)))


train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    .with_options(opts)
    .cache()
    .shuffle(
        buffer_size=min(len(X_train), 4096), seed=19, reshuffle_each_iteration=True
    )
    .flat_map(_augment_threeways)
    .map(_cast_xy_np, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .repeat()
    .prefetch(tf.data.AUTOTUNE)
)

eval_ds = (
    tf.data.Dataset.from_tensor_slices((X_eval, Y_eval))
    .with_options(opts)
    .cache()
    .map(_cast_xy_np, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(tf.data.AUTOTUNE)
)

tf.config.run_functions_eagerly(False)

results = model.fit(
    train_ds,
    validation_data=eval_ds,
    steps_per_epoch=train_steps,
    validation_steps=eval_steps,
    epochs=300,
    callbacks=[es, rlp],
)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2119607025.py in <cell line: 0>()
     36     )
     37     .flat_map(_augment_threeways)
---> 38     .map(_cast_xy_np, num_parallel_calls=tf.data.AUTOTUNE)
     39     .batch(BATCH_SIZE, drop_remainder=False)
     40     .repeat()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in map(self, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)
   2339     from tensorflow.python.data.ops import map_op
   2340 
-> 2341     return map_op._map_v2(
   2342         self,
   2343         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py in _map_v2(input_dataset, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)
     55           num_parallel_calls,
     56       )
---> 57     return _ParallelMapDataset(
     58         input_dataset,
     59         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py in __init__(self, input_dataset, map_func, num_parallel_calls, deterministic, use_inter_op_parallelism, preserve_cardinality, use_legacy_function, use_unbounded_threadpool, name)
    200     self._input_dataset = input_dataset
    201     self._use_inter_op_parallelism = use_inter_op_parallelism
--> 202     self._map_func = structured_function.StructuredFunctionWrapper(
    203         map_func,
    204         self._transformation_name(),

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in __init__(self, func, transformation_name, dataset, input_classes, input_shapes, input_types, input_structure, add_to_graph, use_legacy_function, defun_kwargs)
    263         fn_factory = trace_tf_function(defun_kwargs)
    264 
--> 265     self._function = fn_factory()
    266     # There is no graph to add in eager mode.
    267     add_to_graph &= not context.executing_eagerly()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in get_concrete_function(self, *args, **kwargs)
   1249   def get_concrete_function(self, *args, **kwargs):
   1250     # Implements PolymorphicFunction.get_concrete_function.
-> 1251     concrete = self._get_concrete_function_garbage_collected(*args, **kwargs)
   1252     concrete._garbage_collector.release()  # pylint: disable=protected-access
   1253     return concrete

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in _get_concrete_function_garbage_collected(self, *args, **kwargs)
   1219       if self._variable_creation_config is None:
   1220         initializers = []
-> 1221         self._initialize(args, kwargs, add_initializers_to=initializers)
   1222         self._initialize_uninitialized_variables(initializers)
   1223 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in _initialize(self, args, kwds, add_initializers_to)
    694     )
    695     # Force the definition of the function for these arguments
--> 696     self._concrete_variable_creation_fn = tracing_compilation.trace_function(
    697         args, kwds, self._variable_creation_config
    698     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in trace_function(args, kwargs, tracing_options)
    176       kwargs = {}
    177 
--> 178     concrete_function = _maybe_define_function(
    179         args, kwargs, tracing_options
    180     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in _maybe_define_function(args, kwargs, tracing_options)
    281         else:
    282           target_func_type = lookup_func_type
--> 283         concrete_function = _create_concrete_function(
    284             target_func_type, lookup_func_context, func_graph, tracing_options
    285         )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in _create_concrete_function(function_type, type_context, func_graph, tracing_options)
    308       attributes_lib.DISABLE_ACD, False
    309   )
--> 310   traced_func_graph = func_graph_module.func_graph_from_py_func(
    311       tracing_options.name,
    312       tracing_options.python_function,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/func_graph.py in func_graph_from_py_func(name, python_func, args, kwargs, signature, func_graph, add_control_dependencies, arg_names, op_return_value, collections, capture_by_value, create_placeholders)
   1057 
   1058     _, original_func = tf_decorator.unwrap(python_func)
-> 1059     func_outputs = python_func(*func_args, **func_kwargs)
   1060 
   1061     # invariant: `func_outputs` contains only Tensors, CompositeTensors,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in wrapped_fn(*args, **kwds)
    597         # the function a weak reference to itself to avoid a reference cycle.
    598         with OptionalXlaContext(compile_with_xla):
--> 599           out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    600         return out
    601 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in wrapped_fn(*args)
    229       # Note: wrapper_helper will apply autograph based on context.
    230       def wrapped_fn(*args):  # pylint: disable=missing-docstring
--> 231         ret = wrapper_helper(*args)
    232         ret = structure.to_tensor_list(self._output_structure, ret)
    233         return [ops.convert_to_tensor(t) for t in ret]

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in wrapper_helper(*args)
    159       if not _should_unpack(nested_args):
    160         nested_args = (nested_args,)
--> 161       ret = autograph.tf_convert(self._func, ag_ctx)(*nested_args)
    162       ret = variable_utils.convert_variables_to_tensors(ret)
    163       if _should_pack(ret):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in wrapper(*args, **kwargs)
    691       except Exception as e:  # pylint:disable=broad-except
    692         if hasattr(e, 'ag_error_metadata'):
--> 693           raise e.ag_error_metadata.to_exception(e)
    694         else:
    695           raise

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in wrapper(*args, **kwargs)
    688       try:
    689         with conversion_ctx:
--> 690           return converted_call(f, args, kwargs, options=options)
    691       except Exception as e:  # pylint:disable=broad-except
    692         if hasattr(e, 'ag_error_metadata'):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    437     try:
    438       if kwargs is not None:
--> 439         result = converted_f(*effective_args, **kwargs)
    440       else:
    441         result = converted_f(*effective_args)

TypeError: in user code:


    TypeError: outer_factory.<locals>.inner_factory.<locals>.tf___cast_xy_np() takes 2 positional arguments but 3 were given


## === cell 12
if SHOW_PLOTS:
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(2, 1, figsize=(20, 8))
    history = pd.DataFrame(results.history)
    history[["loss", "val_loss"]].plot(ax=ax[0])
    history[["acc", "val_acc"]].plot(ax=ax[1])
    fig.suptitle("Learning Curve", fontsize=24)
    plt.show()




## === cell 13
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




## === cell 14
pred_eval_ds = (
    tf.data.Dataset.from_tensor_slices(X_eval)
    .with_options(opts)
    .cache()
    .map(lambda x: tf.cast(x, tf.float32), num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

preds_eval = model.predict(pred_eval_ds, verbose=1)

thresholds = np.linspace(0, 1, 50).astype(np.float32)
Y_eval_int = np.int32(Y_eval[..., 0])  # (N,H,W)
preds_eval_f = preds_eval[..., 0].astype(np.float32)  # (N,H,W)

N = Y_eval_int.shape[0]
P = Y_eval_int.shape[1] * Y_eval_int.shape[2]

true_flat = Y_eval_int.reshape(N, P).astype(np.uint8)
pred_flat_f = preds_eval_f.reshape(N, P)

true_bool = true_flat.astype(bool)
true_area = true_flat.sum(axis=1, dtype=np.int64).astype(np.float64)  # (N,)

pred_sorted = np.sort(pred_flat_f, axis=1)  # (N,P)
idx_right = np.searchsorted(pred_sorted, thresholds[:, None], side="right")  # (T,N)
pred_area_all = (P - idx_right).astype(np.float64)  # (T,N)

T = thresholds.shape[0]
inter_all = np.zeros((T, N), dtype=np.float64)
for i in range(N):
    vals = pred_flat_f[i][true_bool[i]]
    if vals.size == 0:
        continue
    vals = np.sort(vals)
    inter_all[:, i] = (
        vals.size - np.searchsorted(vals, thresholds, side="right")
    ).astype(np.float64)

ious = np.empty((T,), dtype=np.float64)
for ti in range(T):
    pred_area = pred_area_all[ti]
    inter = inter_all[ti]
    union = true_area + pred_area - inter
    union[union == 0.0] = 1e-9
    ious[ti] = float(np.mean(inter / union))

threshold_best_index = int(np.argmax(ious[9:-10]) + 9)
iou_best = float(ious[threshold_best_index])
threshold_best = float(thresholds[threshold_best_index])

if SHOW_PLOTS:
    plt.figure(figsize=(10, 4))
    plt.plot(thresholds, ious)
    plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
    plt.xlabel("Threshold")
    plt.ylabel("IoU")
    plt.title("Threshold vs IoU ({}, {})".format(threshold_best, iou_best))
    plt.legend()
    plt.show()
else:
    print("Best threshold:", threshold_best, "IoU:", iou_best)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2888024758.py in <cell line: 0>()
     27 # pred area for each threshold via sort + searchsorted (already vectorized)
     28 pred_sorted = np.sort(pred_flat_f, axis=1)  # (N,P)
---> 29 idx_right = np.searchsorted(pred_sorted, thresholds[:, None], side="right")  # (T,N)
     30 pred_area_all = (P - idx_right).astype(np.float64)  # (T,N)
     31 

/usr/local/lib/python3.11/dist-packages/numpy/core/fromnumeric.py in searchsorted(a, v, side, sorter)
   1398 
   1399     """
-> 1400     return _wrapfunc(a, 'searchsorted', v, side=side, sorter=sorter)
   1401 
   1402 

/usr/local/lib/python3.11/dist-packages/numpy/core/fromnumeric.py in _wrapfunc(obj, method, *args, **kwds)
     57 
     58     try:
---> 59         return bound(*args, **kwds)
     60     except TypeError:
     61         # A TypeError occurs if the object does have such a method in its

ValueError: object too deep for desired array

## === cell 15
cache_test_npz = os.path.join(
    CACHE_DIR, f"test_X_{len(test_ids)}_{config.im_height}x{config.im_width}.npz"
)

if os.path.exists(cache_test_npz):
    d = np.load(cache_test_npz, allow_pickle=False)
    X_test = d["X_test"]
    sizes_test = d["sizes_test"].astype(np.int32).tolist()
    print("Loaded cached test arrays:", cache_test_npz, X_test.shape, len(sizes_test))
else:
    X_test = np.zeros(
        (len(test_ids), config.im_height, config.im_width, config.im_chan),
        dtype=np.uint8,
    )
    sizes_test = []
    print("Getting and resizing test images ... ")
    sys.stdout.flush()

    test_img_base = os.path.join(config.path_test, "images")
    _join = os.path.join
    _read = _read_gray_uint8
    _r_img = _resize_gray_to128

    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        im = _read(_join(test_img_base, id_ + ".png"))
        sizes_test.append([im.shape[0], im.shape[1]])
        im = _r_img(im, config.im_height, config.im_width)
        X_test[n, :, :, 0] = im

    np.savez_compressed(
        cache_test_npz, X_test=X_test, sizes_test=np.array(sizes_test, dtype=np.int16)
    )
    print("Saved cached test arrays:", cache_test_npz)

print("Done!")
print("X_test shape:", X_test.shape)



## === cell 16
pred_test_ds = (
    tf.data.Dataset.from_tensor_slices(X_test)
    .cache()
    .map(lambda x: tf.cast(x, tf.float32), num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

preds_test = model.predict(pred_test_ds, verbose=1)

preds_test_2d = preds_test[..., 0]

preds_test_upsampled = [None] * len(preds_test_2d)
_resize = cv2.resize
_interp = cv2.INTER_LINEAR
for i in trange(len(preds_test_2d)):
    h, w = sizes_test[i]
    preds_test_upsampled[i] = _resize(preds_test_2d[i], (w, h), interpolation=_interp)



## === cell 17
import shutil

try:
    shutil.rmtree("train", ignore_errors=True)
    shutil.rmtree("test", ignore_errors=True)
except Exception as e:
    print("Cleanup skipped:", repr(e))




## === cell 18
def RLenc(img, order="F", format=True):
    img = (img > 0).astype(np.uint8)
    bytes_ = img.reshape(img.shape[0] * img.shape[1], order=order).astype(np.uint8)
    if bytes_.size == 0:
        return "" if format else []

    padded = np.concatenate(([0], bytes_, [0]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    starts = changes[0::2] + 1  # 1-indexed
    ends = changes[1::2]
    lengths = ends - changes[0::2]

    if not format:
        return list(zip(starts.tolist(), lengths.tolist()))
    if starts.size == 0:
        return ""
    return " ".join(map(str, np.column_stack((starts, lengths)).ravel()))


rles = [None] * len(test_ids)
thr = float(threshold_best)

_rlenc = RLenc
for i in tqdm(range(len(test_ids))):
    rles[i] = _rlenc(preds_test_upsampled[i] > thr)

sub = pd.DataFrame({"id": test_ids, "rle_mask": rles})

sample_path = None
for p in [
    "../input/tgs-salt-identification-challenge/sample_submission.csv",
    "/kaggle/input/tgs-salt-identification-challenge/sample_submission.csv",
    "../input/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
]:
    if os.path.exists(p):
        sample_path = p
        break

if sample_path is not None:
    sample = pd.read_csv(sample_path)
    sub = sample[["id"]].merge(sub, on="id", how="left")
    sub["rle_mask"] = sub["rle_mask"].fillna("")
    sub.to_csv("submission.csv", index=False)
else:
    sub = sub.sort_values("id").reset_index(drop=True)
    sub["rle_mask"] = sub["rle_mask"].fillna("")
    sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
print(
    "submission.csv exists:",
    os.path.exists("submission.csv"),
    "size:",
    os.path.getsize("submission.csv") if os.path.exists("submission.csv") else None,
)

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2364191262.py in <cell line: 0>()
     19 
     20 rles = [None] * len(test_ids)
---> 21 thr = float(threshold_best)
     22 
     23 _rlenc = RLenc

NameError: name 'threshold_best' is not defined
