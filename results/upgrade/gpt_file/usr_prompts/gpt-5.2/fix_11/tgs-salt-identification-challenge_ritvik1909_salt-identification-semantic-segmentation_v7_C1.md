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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.auto import tqdm, trange
from itertools import chain

import cv2
from skimage.transform import (
    resize,
)  # kept for strict compatibility if needed elsewhere
from skimage.morphology import label

import tensorflow as tf
from tensorflow.keras.preprocessing.image import (
    array_to_img,
    img_to_array,
    load_img,
    ImageDataGenerator,
)
from tensorflow.keras import backend as K
from tensorflow.keras import models, Input, layers, callbacks, utils

warnings.filterwarnings("ignore")
np.random.seed(19)
random.seed(19)
tf.random.set_seed(19)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TF version:", tf.__version__)

try:
    tf.get_logger().setLevel("ERROR")
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE




## === cell 1
class config:
    im_width = 128
    im_height = 128
    im_chan = 1
    path_train = "train/"
    path_test = "test/"




## === cell 2
INPUT_CANDIDATES = [
    "../input/tgs-salt-identification-challenge",
    "../input/tgs-salt-identification-challenge/tgs-salt-identification-challenge",
    "../input",
]


def _find_dataset_root():
    for p in INPUT_CANDIDATES:
        if os.path.exists(os.path.join(p, "train.zip")) and os.path.exists(
            os.path.join(p, "test.zip")
        ):
            return p
    if os.path.isdir("../input"):
        for root, dirs, files in os.walk("../input"):
            if "train.zip" in files and "test.zip" in files:
                return root
    raise FileNotFoundError(
        "Could not locate dataset root containing train.zip and test.zip under ../input"
    )


DATA_ROOT = _find_dataset_root()
print("Using DATA_ROOT:", DATA_ROOT)

os.makedirs("train", exist_ok=True)
os.makedirs("test", exist_ok=True)


def _maybe_unzip(zip_path, out_dir, expected_rel):
    expected_path = os.path.join(out_dir, expected_rel)
    if os.path.exists(expected_path):
        return 0
    return os.system(f'unzip -q -n "{zip_path}" -d "{out_dir}/"')


ret1 = _maybe_unzip(f"{DATA_ROOT}/train.zip", "train", os.path.join("train", "images"))
ret2 = _maybe_unzip(
    f"{DATA_ROOT}/test.zip", "test", os.path.join("test", "test", "images")
)
if ret1 != 0 and not any(
    os.path.isdir(os.path.join("train", d))
    for d in ["train", "images", "tgs-salt-identification-challenge"]
):
    raise RuntimeError(f"Unzip failed with code train={ret1}")
if ret2 != 0 and not any(
    os.path.isdir(os.path.join("test", d))
    for d in ["test", "images", "tgs-salt-identification-challenge"]
):
    raise RuntimeError(f"Unzip failed with code test={ret2}")


def _resolve_images_masks_root(extract_base: str, want_masks: bool):
    """
    Return directory that directly contains images/(and masks/ if want_masks).
    Handles zips that extract as:
      train/images, train/masks
      train/train/images, train/train/masks
      train/tgs-salt-identification-challenge/train/images, ...
    """
    candidates = [
        extract_base,
        os.path.join(extract_base, "train"),
        os.path.join(extract_base, "test"),
        os.path.join(extract_base, "tgs-salt-identification-challenge", "train"),
        os.path.join(extract_base, "tgs-salt-identification-challenge", "test"),
        os.path.join(extract_base, "tgs-salt-identification-challenge"),
        os.path.join(
            extract_base,
            "tgs-salt-identification-challenge",
            "tgs-salt-identification-challenge",
            "train",
        ),
        os.path.join(
            extract_base,
            "tgs-salt-identification-challenge",
            "tgs-salt-identification-challenge",
            "test",
        ),
    ]
    for c in candidates:
        img_dir = os.path.join(c, "images")
        msk_dir = os.path.join(c, "masks")
        if os.path.isdir(img_dir) and (not want_masks or os.path.isdir(msk_dir)):
            return c

    for root, dirs, files in os.walk(extract_base):
        if os.path.basename(root) == "images":
            if any(f.lower().endswith(".png") for f in files):
                parent = os.path.dirname(root)
                if not want_masks:
                    return parent
                if os.path.isdir(os.path.join(parent, "masks")):
                    return parent

    raise FileNotFoundError(
        f"Could not find a valid extracted structure under {extract_base}"
    )


train_root = _resolve_images_masks_root("train", want_masks=True)
test_root = _resolve_images_masks_root("test", want_masks=False)

config.path_train = train_root + ("" if train_root.endswith(os.sep) else os.sep)
config.path_test = test_root + ("" if test_root.endswith(os.sep) else os.sep)

assert os.path.isdir(
    os.path.join(config.path_train, "images")
), "train images dir missing"
assert os.path.isdir(
    os.path.join(config.path_train, "masks")
), "train masks dir missing"
assert os.path.isdir(
    os.path.join(config.path_test, "images")
), "test images dir missing"

print("Resolved train root:", config.path_train)
print("Resolved test root :", config.path_test)




## === cell 3
random.seed(19)
train_images_dir = os.path.join(config.path_train, "images")
train_masks_dir = os.path.join(config.path_train, "masks")

ids_all = [f for f in os.listdir(train_images_dir) if f.lower().endswith(".png")]
k = min(6, len(ids_all))
ids = random.sample(ids_all, k=k) if k > 0 else []

fig = plt.figure(figsize=(20, 6))
for j, img_name in enumerate(ids):
    q = j + 1
    img = load_img(os.path.join(train_images_dir, img_name), color_mode="grayscale")
    img_mask = load_img(os.path.join(train_masks_dir, img_name), color_mode="grayscale")

    plt.subplot(2, 6, q * 2 - 1)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.subplot(2, 6, q * 2)
    plt.imshow(img_mask, cmap="gray")
    plt.axis("off")
fig.suptitle("Sample Images", fontsize=24)
plt.show()




## === cell 4
train_ids = sorted(
    [
        f
        for f in os.listdir(os.path.join(config.path_train, "images"))
        if f.lower().endswith(".png")
    ]
)
test_ids = sorted(
    [
        f
        for f in os.listdir(os.path.join(config.path_test, "images"))
        if f.lower().endswith(".png")
    ]
)

print("Train images:", len(train_ids))
print("Test images:", len(test_ids))
print("Example train id:", train_ids[0] if train_ids else None)
print("Example test id:", test_ids[0] if test_ids else None)




## === cell 5
def _tf_read_png_grayscale(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_png(img_bytes, channels=1, dtype=tf.uint8)  # [H,W,1]
    return img


def _tf_resize_image_uint8(img_uint8):
    x = tf.image.resize(
        img_uint8, [config.im_height, config.im_width], method="bilinear"
    )
    x = tf.cast(tf.round(x), tf.uint8)
    return x


def _tf_resize_mask_to_bool(mask_uint8):
    y = tf.image.resize(
        mask_uint8, [config.im_height, config.im_width], method="nearest"
    )
    y = tf.cast(y > 127, tf.bool)
    return y


def make_train_dataset(train_ids, batch_size=8, val_split=0.1, seed=19):
    n = len(train_ids)
    n_val = int(round(n * val_split))
    n_train = n - n_val

    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    train_files = [train_ids[i] for i in train_idx]
    val_files = [train_ids[i] for i in val_idx]

    def _ds_from_files(files, shuffle, cache_path):
        img_paths = tf.constant(
            [os.path.join(config.path_train, "images", f) for f in files]
        )
        msk_paths = tf.constant(
            [os.path.join(config.path_train, "masks", f) for f in files]
        )

        ds = tf.data.Dataset.from_tensor_slices((img_paths, msk_paths))
        ds = ds.enumerate()

        def _load_resize(i, paths):
            ip, mp = paths
            x = _tf_read_png_grayscale(ip)
            y = _tf_read_png_grayscale(mp)
            x = _tf_resize_image_uint8(x)
            y = _tf_resize_mask_to_bool(y)
            return i, x, y

        ds = ds.map(_load_resize, num_parallel_calls=AUTOTUNE)
        ds = ds.cache(cache_path)

        if shuffle:
            ds = ds.shuffle(len(files), seed=seed, reshuffle_each_iteration=True)

            def _augment(i, x, y):
                r = tf.random.stateless_uniform(
                    [], seed=tf.stack([tf.cast(seed, tf.int32), tf.cast(i, tf.int32)])
                )
                do_flip = r < 0.5
                x = tf.cond(do_flip, lambda: tf.image.flip_left_right(x), lambda: x)
                y = tf.cond(do_flip, lambda: tf.image.flip_left_right(y), lambda: y)
                return x, y

            ds = ds.map(_augment, num_parallel_calls=AUTOTUNE)
        else:
            ds = ds.map(lambda i, x, y: (x, y), num_parallel_calls=AUTOTUNE)

        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(AUTOTUNE)

        opts = tf.data.Options()
        opts.experimental_optimization.apply_default_optimizations = True
        opts.experimental_deterministic = True
        ds = ds.with_options(opts)
        return ds

    train_cache = os.path.join(os.getcwd(), "cache_train_det.tfcache")
    val_cache = os.path.join(os.getcwd(), "cache_val_det.tfcache")
    return _ds_from_files(
        train_files, shuffle=True, cache_path=train_cache
    ), _ds_from_files(val_files, shuffle=False, cache_path=val_cache)




## === cell 6
train_ds, val_ds = make_train_dataset(train_ids, batch_size=8, val_split=0.1, seed=19)

print(
    "Prepared tf.data train/val datasets (decode+resize cached; deterministic stateless flip augmentation after cache)."
)




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

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["acc"],
    run_eagerly=False,
)
model.summary()




## === cell 8
DO_PLOT_MODEL = False
if DO_PLOT_MODEL:
    try:
        utils.plot_model(model, expand_nested=True, show_shapes=True)
    except Exception as e:
        print("plot_model skipped due to:", repr(e))




## === cell 9
es = callbacks.EarlyStopping(patience=30, verbose=1, restore_best_weights=True)
rlp = callbacks.ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-12, verbose=1)

results = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=300,
    callbacks=[es, rlp],
    verbose=2,
)




## === cell 10
sns.set_style("darkgrid")
fig, ax = plt.subplots(2, 1, figsize=(20, 8))
history = pd.DataFrame(results.history)
history[["loss", "val_loss"]].plot(ax=ax[0])
history[["acc", "val_acc"]].plot(ax=ax[1])
fig.suptitle("Learning Curve", fontsize=24)
plt.show()




## === cell 11
test_img_dir = os.path.join(config.path_test, "images")
test_paths = [os.path.join(test_img_dir, f) for f in test_ids]


def _tf_load_test_with_size(img_path):
    x = _tf_read_png_grayscale(img_path)  # [H,W,1] uint8
    shape = tf.shape(x)
    size = tf.stack([shape[0], shape[1]], axis=0)  # int32
    x = _tf_resize_image_uint8(x)  # [128,128,1] uint8
    size = tf.cast(size, tf.int32)
    return x, size


test_ds = tf.data.Dataset.from_tensor_slices(tf.constant(test_paths))
test_ds = test_ds.map(_tf_load_test_with_size, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.cache(os.path.join(os.getcwd(), "cache_test.tfcache"))
test_ds = test_ds.batch(32, drop_remainder=False).prefetch(AUTOTUNE)

opts = tf.data.Options()
opts.experimental_optimization.apply_default_optimizations = True
opts.experimental_deterministic = True
test_ds = test_ds.with_options(opts)

print("Prepared cached tf.data test dataset.")




## === cell 12
sizes_list = []
preds_list = []

for x_b, size_b in test_ds:
    preds_b = model(x_b, training=False).numpy()
    preds_list.append(preds_b)
    sizes_list.append(size_b.numpy())

preds_test = np.concatenate(preds_list, axis=0)
sizes_test = np.concatenate(sizes_list, axis=0).astype(np.int32)

print(
    "Done! preds_test shape:", preds_test.shape, "sizes_test shape:", sizes_test.shape
)




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
        bytes_ = img.T.reshape(-1)  # faster equivalent to reshape(order='F')

    bytes_ = bytes_.astype(np.uint8)
    if bytes_.size == 0:
        return "" if format else []

    padded = np.concatenate([[0], bytes_, [0]])
    changes = np.flatnonzero(padded[1:] != padded[:-1]) + 1
    runs = changes.reshape(-1, 2)
    runs[:, 1] = runs[:, 1] - runs[:, 0]
    runs[:, 0] = runs[:, 0]  # already 1-indexed due to padding scheme

    if not format:
        return [tuple(x) for x in runs.tolist()]

    if runs.size == 0:
        return ""
    return " ".join(map(str, runs.reshape(-1).tolist()))


pred_dict = {}
for i, fn in tqdm(enumerate(test_ids), total=len(test_ids)):
    h, w = int(sizes_test[i, 0]), int(sizes_test[i, 1])
    p = preds_test[i, :, :, 0]
    up = cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)
    m = (up >= 0.5).astype(np.uint8)
    pred_dict[fn[:-4]] = RLenc(m)




## === cell 14
sub = pd.DataFrame.from_dict(pred_dict, orient="index")
sub.index.names = ["id"]
sub.columns = ["rle_mask"]
sub = sub.reset_index()

sample_path = os.path.join(DATA_ROOT, "sample_submission.csv")
if not os.path.exists(sample_path):
    sample_path = "../input/sample_submission.csv"
sample = pd.read_csv(sample_path)

sub = sample[["id"]].merge(sub, on="id", how="left")
sub["rle_mask"] = sub["rle_mask"].fillna("")

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
print("submission.csv exists:", os.path.exists("submission.csv"))
