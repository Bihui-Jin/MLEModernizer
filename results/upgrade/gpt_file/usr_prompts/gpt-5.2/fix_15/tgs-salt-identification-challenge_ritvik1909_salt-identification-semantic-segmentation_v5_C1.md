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

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PYTHONHASHSEED", "19")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import sys, random, warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.auto import tqdm

import cv2
from skimage.transform import resize  # kept for equivalence where still used
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

try:
    tf.config.threading.set_intra_op_parallelism_threads(
        max(1, (os.cpu_count() or 2) // 2)
    )
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

random.seed(19)
np.random.seed(19)
tf.random.set_seed(19)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

print("TF version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))




## === cell 1
class config:
    im_width = 128
    im_height = 128
    im_chan = 1
    path_train = "train/"
    path_test = "test/"




## === cell 2
import shutil
from pathlib import Path


def _ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def _looks_ready():
    return (
        Path("train/images").is_dir()
        and Path("train/masks").is_dir()
        and Path("test/images").is_dir()
        and any(Path("train/images").glob("*.png"))
        and any(Path("train/masks").glob("*.png"))
        and any(Path("test/images").glob("*.png"))
    )


def _rm_rf(p: Path):
    """Remove file/dir/symlink if it exists."""
    try:
        if not p.exists() and not p.is_symlink():
            return
        if p.is_symlink() or p.is_file():
            p.unlink()
        else:
            shutil.rmtree(p)
    except Exception:
        pass


def _link_or_copy_tree(src: Path, dst: Path):
    _rm_rf(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(str(src), str(dst), target_is_directory=True)
    except Exception:
        shutil.copytree(str(src), str(dst), dirs_exist_ok=True)


def _ensure_tree_from_candidates(expected: str, candidates):
    expected = Path(expected)
    if expected.exists() and any(expected.glob("*.png")):
        return True
    for c in candidates:
        c = Path(c)
        if c.exists() and c.is_dir() and any(c.glob("*.png")):
            _link_or_copy_tree(c, expected)
            return True
    return False


def _find_dataset_root():
    candidates = [
        Path("/kaggle/input/tgs-salt-identification-challenge"),
        Path("../input/tgs-salt-identification-challenge"),
        Path("/kaggle/data/tgs-salt-identification-challenge"),
        Path("../data/tgs-salt-identification-challenge"),
        Path("/kaggle/input"),
        Path("../input"),
        Path("/kaggle/data"),
        Path("../data"),
    ]
    for c in candidates:
        if (c / "train.zip").exists() and (c / "test.zip").exists():
            return c
        if (
            (c / "train" / "images").is_dir()
            and (c / "train" / "masks").is_dir()
            and (c / "test" / "images").is_dir()
        ):
            return c
        if (c / "tgs-salt-identification-challenge" / "train.zip").exists():
            return c / "tgs-salt-identification-challenge"
        if (c / "tgs-salt-identification-challenge" / "train" / "images").is_dir():
            return c / "tgs-salt-identification-challenge"
    return None


if not _looks_ready():
    ds_root = _find_dataset_root()
    if ds_root is None:
        raise FileNotFoundError(
            "Could not locate tgs-salt-identification-challenge dataset root under /kaggle/input or ../input"
        )

    direct_train_images = ds_root / "train" / "images"
    direct_train_masks = ds_root / "train" / "masks"
    direct_test_images = ds_root / "test" / "images"

    for p in ["train/images", "train/masks", "test/images", "train", "test"]:
        _rm_rf(Path(p))

    _ensure_dir("train")
    _ensure_dir("test")

    if (
        direct_train_images.exists()
        and direct_train_masks.exists()
        and direct_test_images.exists()
    ):
        _link_or_copy_tree(direct_train_images, Path("train/images"))
        _link_or_copy_tree(direct_train_masks, Path("train/masks"))
        _link_or_copy_tree(direct_test_images, Path("test/images"))
    else:
        train_zip = ds_root / "train.zip"
        test_zip = ds_root / "test.zip"
        if not train_zip.exists() or not test_zip.exists():
            raise FileNotFoundError(
                f"Could not find train.zip/test.zip in dataset root: {ds_root}"
            )
        os.system(f"unzip -q {train_zip} -d train/")
        os.system(f"unzip -q {test_zip} -d test/")

_ensure_tree_from_candidates(
    "train/images",
    [
        "train/images",
        "train/train/images",
        "train/tgs-salt-identification-challenge/train/images",
        "/kaggle/input/tgs-salt-identification-challenge/train/images",
        "../input/tgs-salt-identification-challenge/train/images",
    ],
)
_ensure_tree_from_candidates(
    "train/masks",
    [
        "train/masks",
        "train/train/masks",
        "train/tgs-salt-identification-challenge/train/masks",
        "/kaggle/input/tgs-salt-identification-challenge/train/masks",
        "../input/tgs-salt-identification-challenge/train/masks",
    ],
)
_ensure_tree_from_candidates(
    "test/images",
    [
        "test/images",
        "test/test/images",
        "test/tgs-salt-identification-challenge/test/images",
        "/kaggle/input/tgs-salt-identification-challenge/test/images",
        "../input/tgs-salt-identification-challenge/test/images",
    ],
)

assert Path("train/images").exists(), "train/images not found after unzip/normalize"
assert Path("train/masks").exists(), "train/masks not found after unzip/normalize"
assert Path("test/images").exists(), "test/images not found after unzip/normalize"

print("Dataset ready.")
print("train/images exists:", Path("train/images").exists())
print("train/masks  exists:", Path("train/masks").exists())
print("test/images  exists:", Path("test/images").exists())



## === cell 3
if bool(int(os.environ.get("SHOW_SAMPLES", "0"))):
    random.seed(19)
    ids = random.choices(os.listdir("train/images"), k=6)
    fig = plt.figure(figsize=(20, 6))
    for j, img_name in enumerate(ids):
        q = j + 1

        img = load_img("train/images/" + img_name)
        img_mask = load_img("train/masks/" + img_name)

        plt.subplot(2, 6, q * 2 - 1)
        plt.imshow(img)
        plt.axis("off")
        plt.subplot(2, 6, q * 2)
        plt.imshow(img_mask)
        plt.axis("off")
    fig.suptitle("Sample Images", fontsize=24)
else:
    print("Sample visualization skipped (set SHOW_SAMPLES=1 to enable).")



## === cell 4
train_img_path = os.path.join(config.path_train, "images")
test_img_path = os.path.join(config.path_test, "images")

train_ids = sorted(
    [f for f in os.listdir(train_img_path) if f.lower().endswith(".png")]
)
test_ids = sorted([f for f in os.listdir(test_img_path) if f.lower().endswith(".png")])

print("n_train:", len(train_ids), "n_test:", len(test_ids))
assert (
    len(train_ids) > 0 and len(test_ids) > 0
), "No PNGs found in train/test image directories."



## === cell 5
AUTOTUNE = tf.data.AUTOTUNE
options = tf.data.Options()
options.deterministic = True

train_img_dir = os.path.join(config.path_train, "images")
train_mask_dir = os.path.join(config.path_train, "masks")
test_img_dir = os.path.join(config.path_test, "images")

train_img_paths = [os.path.join(train_img_dir, f) for f in train_ids]
train_mask_paths = [os.path.join(train_mask_dir, f) for f in train_ids]
test_img_paths = [os.path.join(test_img_dir, f) for f in test_ids]


def _decode_png_gray(path):
    b = tf.io.read_file(path)
    im = tf.image.decode_png(b, channels=1)  # uint8, [H,W,1]
    return im


def _parse_train(img_path, mask_path):
    img = _decode_png_gray(img_path)
    mask = _decode_png_gray(mask_path)

    img = tf.image.resize(img, [config.im_height, config.im_width], method="bilinear")
    img = tf.cast(tf.round(img), tf.uint8)  # keep uint8-like input pipeline

    mask = tf.image.resize(mask, [config.im_height, config.im_width], method="nearest")
    mask = tf.cast(mask > 0, tf.bool)
    return img, mask


def _parse_test(img_path):
    img = _decode_png_gray(img_path)
    shp = tf.shape(img)
    h = tf.cast(shp[0], tf.int32)
    w = tf.cast(shp[1], tf.int32)
    img_r = tf.image.resize(img, [config.im_height, config.im_width], method="bilinear")
    img_r = tf.cast(tf.round(img_r), tf.uint8)
    return img_r, tf.stack([h, w], axis=0)


print("Building tf.data datasets (train/val/test) ...")
sys.stdout.flush()

n = len(train_img_paths)
val_n = int(round(n * 0.1))
train_n = n - val_n

train_ds_full = tf.data.Dataset.from_tensor_slices(
    (train_img_paths, train_mask_paths)
).with_options(options)

train_ds_full = train_ds_full.map(_parse_train, num_parallel_calls=AUTOTUNE).cache()

train_ds = train_ds_full.take(train_n).batch(8, drop_remainder=False).prefetch(AUTOTUNE)
val_ds = train_ds_full.skip(train_n).batch(8, drop_remainder=False).prefetch(AUTOTUNE)

test_ds_sizes = tf.data.Dataset.from_tensor_slices(test_img_paths).with_options(options)
test_ds_sizes = test_ds_sizes.map(_parse_test, num_parallel_calls=AUTOTUNE).cache()
test_ds = test_ds_sizes.map(lambda x, sz: x).batch(32).prefetch(AUTOTUNE)

sizes_test = np.empty((len(test_ids), 2), dtype=np.int16)
for i, (_, sz) in enumerate(test_ds_sizes):
    sizes_test[i, 0] = int(sz[0].numpy())
    sizes_test[i, 1] = int(sz[1].numpy())

print("Datasets ready. train_n:", train_n, "val_n:", val_n)



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
if bool(int(os.environ.get("PLOT_MODEL", "0"))):
    try:
        utils.plot_model(model, show_shapes=True)
    except Exception as e:
        print("plot_model skipped due to:", repr(e))
else:
    print("plot_model skipped (set PLOT_MODEL=1 to enable).")



## === cell 8
es = callbacks.EarlyStopping(patience=30, verbose=1, restore_best_weights=True)
rlp = callbacks.ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-12, verbose=1)

MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", "80"))

results = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=MAX_EPOCHS,
    callbacks=[es, rlp],
    verbose=1,
)



## === cell 9
if bool(int(os.environ.get("PLOT_HISTORY", "0"))):
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(2, 1, figsize=(20, 8))
    history = pd.DataFrame(results.history)
    history[["loss", "val_loss"]].plot(ax=ax[0])
    history[["acc", "val_acc"]].plot(ax=ax[1])
    fig.suptitle("Learning Curve", fontsize=24)
else:
    print("History plotting skipped (set PLOT_HISTORY=1 to enable).")



## === cell 10
preds_test = model.predict(test_ds, verbose=1)

preds_test_hw = preds_test[..., 0].astype(np.float32, copy=False)  # (N, 128, 128)
sizes_hw = sizes_test.astype(np.int32, copy=False)  # (N, 2) [h, w]

if np.all(sizes_hw[:, 0] == sizes_hw[0, 0]) and np.all(
    sizes_hw[:, 1] == sizes_hw[0, 1]
):
    h = int(sizes_hw[0, 0])
    w = int(sizes_hw[0, 1])

    preds_ch_last = np.transpose(preds_test_hw, (1, 2, 0))  # (128,128,N)
    preds_up_ch_last = cv2.resize(
        preds_ch_last, (w, h), interpolation=cv2.INTER_LINEAR
    )  # (h,w,N)
    preds_test_upsampled = np.transpose(preds_up_ch_last, (2, 0, 1))  # (N,h,w)
else:
    preds_test_upsampled = np.empty(
        (preds_test_hw.shape[0], 101, 101), dtype=np.float32
    )
    for i in range(preds_test_hw.shape[0]):
        h = int(sizes_hw[i, 0])
        w = int(sizes_hw[i, 1])
        preds_test_upsampled[i] = cv2.resize(
            preds_test_hw[i], (w, h), interpolation=cv2.INTER_LINEAR
        )



## === cell 11
print("Skipping cleanup of train/test folders to save time.")




## === cell 12
def RLenc_from_flat_fortran(pixels_uint8, format=True):
    """
    pixels_uint8: 1D uint8 array of a 2D mask flattened in Fortran order (down-then-right).
    """
    padded = np.empty((pixels_uint8.size + 2,), dtype=np.uint8)
    padded[0] = 0
    padded[-1] = 0
    padded[1:-1] = pixels_uint8

    changes = np.where(padded[1:] != padded[:-1])[0] + 1
    starts = changes[0::2]
    ends = changes[1::2]
    lengths = ends - starts

    if not format:
        return list(zip(starts.tolist(), lengths.tolist()))

    if starts.size == 0:
        return ""
    out = np.empty((starts.size * 2,), dtype=np.int64)
    out[0::2] = starts
    out[1::2] = lengths
    return " ".join(map(str, out))


masks_bin = np.round(preds_test_upsampled).astype(np.uint8, copy=False)  # (N,101,101)

ids_out = [fn[:-4] for fn in test_ids]

rle_out = [None] * len(test_ids)
for i in range(len(test_ids)):
    flat_f = np.ravel(masks_bin[i], order="F")
    rle_out[i] = RLenc_from_flat_fortran(flat_f, format=True)

sub = pd.DataFrame({"id": ids_out, "rle_mask": rle_out})
sub = sub[["id", "rle_mask"]]
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
