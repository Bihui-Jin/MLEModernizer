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
import os, sys, random, warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.auto import tqdm

import cv2
from skimage.transform import resize  # kept for correctness fallback if needed
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

random.seed(19)
np.random.seed(19)
tf.random.set_seed(19)

try:
    tf.config.threading.set_intra_op_parallelism_threads(
        max(1, (os.cpu_count() or 2) - 1)
    )
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    cv2.setNumThreads(max(1, (os.cpu_count() or 2) - 1))
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")




## === cell 1
class config:
    im_width = 128
    im_height = 128
    im_chan = 1
    path_train = "train/"
    path_test = "test/"




## === cell 2
train_images_probe = os.path.join("train", "images")
test_images_probe = os.path.join("test", "images")


def _dir_has_files(p):
    try:
        with os.scandir(p) as it:
            for e in it:
                if e.is_file():
                    return True
    except FileNotFoundError:
        return False
    return False


if not (_dir_has_files(train_images_probe) and _dir_has_files(test_images_probe)):
    os.system("unzip -q ../input/tgs-salt-identification-challenge/train.zip -d train/")
    os.system("unzip -q ../input/tgs-salt-identification-challenge/test.zip -d test/")




## === cell 3
def _resolve_images_dir(base_path: str) -> str:
    cand = os.path.join(base_path, "images")
    if os.path.isdir(cand):
        return cand
    base_name = os.path.basename(os.path.normpath(base_path))
    cand2 = os.path.join(base_path, base_name, "images")
    if os.path.isdir(cand2):
        return cand2
    raise FileNotFoundError(
        f"Could not locate an 'images' directory under '{base_path}'."
    )


def _resolve_masks_dir(base_path: str) -> str:
    cand = os.path.join(base_path, "masks")
    if os.path.isdir(cand):
        return cand
    base_name = os.path.basename(os.path.normpath(base_path))
    cand2 = os.path.join(base_path, base_name, "masks")
    if os.path.isdir(cand2):
        return cand2
    raise FileNotFoundError(
        f"Could not locate a 'masks' directory under '{base_path}'."
    )


train_images_dir = _resolve_images_dir(config.path_train)
train_masks_dir = _resolve_masks_dir(config.path_train)
test_images_dir = _resolve_images_dir(config.path_test)


def _sorted_filenames(dir_path: str):
    with os.scandir(dir_path) as it:
        names = [e.name for e in it if e.is_file()]
    names.sort()
    return names


train_ids = _sorted_filenames(train_images_dir)
test_ids = _sorted_filenames(test_images_dir)



## === cell 4
if False:
    ids = random.choices(train_ids, k=6)
    fig = plt.figure(figsize=(20, 6))
    for j, img_name in enumerate(ids):
        q = j + 1
        img = cv2.imread(os.path.join(train_images_dir, img_name), cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(os.path.join(train_masks_dir, img_name), cv2.IMREAD_GRAYSCALE)
        plt.subplot(2, 6, q * 2 - 1)
        plt.imshow(img, cmap="gray")
        plt.subplot(2, 6, q * 2)
        plt.imshow(msk, cmap="gray")
    fig.suptitle("Sample Images", fontsize=24)



## === cell 5
from concurrent.futures import ThreadPoolExecutor

X_train = np.zeros(
    (len(train_ids), config.im_height, config.im_width, config.im_chan), dtype=np.uint8
)
Y_train = np.zeros((len(train_ids), config.im_height, config.im_width, 1), dtype=bool)

print("Getting and resizing train images and masks ... ")
sys.stdout.flush()

_imread = cv2.imread
_resize = cv2.resize
IMREAD_GRAYSCALE = cv2.IMREAD_GRAYSCALE
INTER_LINEAR = cv2.INTER_LINEAR
INTER_NEAREST = cv2.INTER_NEAREST
t_wh = (config.im_width, config.im_height)


def _load_train_pair(idx_and_name):
    n, id_ = idx_and_name
    img_path = os.path.join(train_images_dir, id_)
    mask_path = os.path.join(train_masks_dir, id_)

    x = _imread(img_path, IMREAD_GRAYSCALE)
    if x is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")
    x = _resize(x, t_wh, interpolation=INTER_LINEAR)

    m = _imread(mask_path, IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Failed to read mask: {mask_path}")
    m = _resize(m, t_wh, interpolation=INTER_NEAREST)
    mb = m > 127
    return n, x, mb


_workers = max(2, min(8, (os.cpu_count() or 4)))
with ThreadPoolExecutor(max_workers=_workers) as ex:
    for n, x, mb in tqdm(
        ex.map(_load_train_pair, enumerate(train_ids), chunksize=256),
        total=len(train_ids),
    ):
        X_train[n, :, :, 0] = x
        Y_train[n, :, :, 0] = mb

print("Done!")

X_train_f = np.ascontiguousarray(X_train, dtype=np.float32)
Y_train_f = np.ascontiguousarray(Y_train.astype(np.float32, copy=False))



## === cell 6
inputs = Input((config.im_height, config.im_width, config.im_chan))
s = layers.Lambda(lambda x: x / 255)(inputs)

c1 = layers.Conv2D(8, (3, 3), activation="relu", padding="same")(s)
c1 = layers.Conv2D(8, (3, 3), activation="relu", padding="same")(c1)
p1 = layers.MaxPooling2D((2, 2))(c1)

c2 = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(p1)
c2 = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(c2)
p2 = layers.MaxPooling2D((2, 2))(c2)

c3 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(p2)
c3 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(c3)
p3 = layers.MaxPooling2D((2, 2))(c3)

c4 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(p3)
c4 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(c4)
p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)

c5 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(p4)
c5 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(c5)

u6 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c5)
u6 = layers.concatenate([u6, c4])
c6 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(u6)
c6 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(c6)

u7 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c6)
u7 = layers.concatenate([u7, c3])
c7 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(u7)
c7 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(c7)

u8 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(c7)
u8 = layers.concatenate([u8, c2])
c8 = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(u8)
c8 = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(c8)

u9 = layers.Conv2DTranspose(8, (2, 2), strides=(2, 2), padding="same")(c8)
u9 = layers.concatenate([u9, c1], axis=3)
c9 = layers.Conv2D(8, (3, 3), activation="relu", padding="same")(u9)
c9 = layers.Conv2D(8, (3, 3), activation="relu", padding="same")(c9)

outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(c9)

model = models.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
model.summary()



## === cell 7
pass



## === cell 8
es = callbacks.EarlyStopping(patience=30, verbose=1, restore_best_weights=True)
rlp = callbacks.ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-12, verbose=1)

n_total = X_train_f.shape[0]
n_val = int(round(n_total * 0.1))
n_train = n_total - n_val

X_tr, X_va = X_train_f[:n_train], X_train_f[n_train:]
Y_tr, Y_va = Y_train_f[:n_train], Y_train_f[n_train:]

BATCH = 8

cache_train_path = os.path.join(".", "tf_cache_train")
cache_val_path = os.path.join(".", "tf_cache_val")
for p in (cache_train_path, cache_val_path):
    try:
        if os.path.exists(p):
            os.remove(p)
    except Exception:
        pass

opts = tf.data.Options()
opts.deterministic = True
opts.experimental_optimization.apply_default_optimizations = True
opts.experimental_optimization.map_parallelization = True
opts.experimental_optimization.parallel_batch = True

train_ds = (
    tf.data.Dataset.from_tensor_slices((X_tr, Y_tr))
    .with_options(opts)
    .cache(cache_train_path)
    .shuffle(buffer_size=min(len(X_tr), 1024), seed=19, reshuffle_each_iteration=True)
    .batch(BATCH, drop_remainder=False)
    .prefetch(AUTOTUNE)
)
val_ds = (
    tf.data.Dataset.from_tensor_slices((X_va, Y_va))
    .with_options(opts)
    .cache(cache_val_path)
    .batch(BATCH, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

steps_per_epoch = int(np.ceil(len(X_tr) / BATCH))
validation_steps = int(np.ceil(len(X_va) / BATCH))

results = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=300,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[es, rlp],
    verbose=1,
)



## === cell 9
if False:
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(2, 1, figsize=(20, 8))
    history = pd.DataFrame(results.history)
    history[["loss", "val_loss"]].plot(ax=ax[0])
    history[["acc", "val_acc"]].plot(ax=ax[1])
    fig.suptitle("Learning Curve", fontsize=24)



## === cell 10
X_test = np.zeros(
    (len(test_ids), config.im_height, config.im_width, config.im_chan), dtype=np.uint8
)
sizes_test = np.zeros((len(test_ids), 2), dtype=np.int32)

print("Getting and resizing test images ... ")
sys.stdout.flush()

_imread = cv2.imread
_resize = cv2.resize
IMREAD_GRAYSCALE = cv2.IMREAD_GRAYSCALE
INTER_LINEAR = cv2.INTER_LINEAR
t_wh = (config.im_width, config.im_height)


def _load_test(idx_and_name):
    n, id_ = idx_and_name
    img_path = os.path.join(test_images_dir, id_)
    x = _imread(img_path, IMREAD_GRAYSCALE)
    if x is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")
    h, w = x.shape[:2]
    xr = _resize(x, t_wh, interpolation=INTER_LINEAR)
    return n, h, w, xr


_workers = max(2, min(8, (os.cpu_count() or 4)))
with ThreadPoolExecutor(max_workers=_workers) as ex:
    for n, h, w, xr in tqdm(
        ex.map(_load_test, enumerate(test_ids), chunksize=256), total=len(test_ids)
    ):
        sizes_test[n, 0], sizes_test[n, 1] = h, w
        X_test[n, :, :, 0] = xr

print("Done!")

X_test_f = np.ascontiguousarray(X_test, dtype=np.float32)



## === cell 11
test_ds = (
    tf.data.Dataset.from_tensor_slices(X_test_f)
    .with_options(tf.data.Options())
    .cache()
    .batch(64, drop_remainder=False)
    .prefetch(AUTOTUNE)
)
preds_test = model.predict(test_ds, verbose=1)

sizes_unique, inv = np.unique(sizes_test, axis=0, return_inverse=True)
preds_test_upsampled = [None] * len(preds_test)

preds_test_4d = preds_test.astype(np.float32, copy=False)  # (N,128,128,1)
for g, (h, w) in enumerate(sizes_unique):
    idxs = np.nonzero(inv == g)[0]
    if idxs.size == 0:
        continue
    resized = tf.image.resize(
        preds_test_4d[idxs],
        size=(int(h), int(w)),
        method=tf.image.ResizeMethod.BILINEAR,
    ).numpy()
    for j, k in enumerate(idxs):
        preds_test_upsampled[int(k)] = resized[j, :, :, 0]



## === cell 12
os.system("rm -rf train")
os.system("rm -rf test")




## === cell 13
def RLenc(img, order="F", format=True):
    bytes_ = img.reshape(img.shape[0] * img.shape[1], order=order).astype(
        np.uint8, copy=False
    )
    padded = np.concatenate(([0], bytes_, [0]))
    changes = np.flatnonzero(padded[1:] != padded[:-1]) + 1
    starts = changes[0::2]
    ends = changes[1::2]
    lengths = ends - starts
    if not format:
        return list(zip(starts.tolist(), lengths.tolist()))
    if starts.size == 0:
        return ""
    s = starts.astype(str)
    l = lengths.astype(str)
    out = np.empty((s.size * 2,), dtype=object)
    out[0::2] = s
    out[1::2] = l
    return " ".join(out.tolist())


preds_rounded = np.rint(np.asarray(preds_test_upsampled, dtype=np.float32)).astype(
    np.uint8, copy=False
)

ids_out = [fn[:-4] for fn in test_ids]

_rle = RLenc
rle_out = [None] * len(test_ids)
for i in tqdm(range(len(test_ids)), total=len(test_ids)):
    rle_out[i] = _rle(preds_rounded[i])

sub = pd.DataFrame({"id": ids_out, "rle_mask": rle_out})
sub.to_csv("submission.csv", index=False)
