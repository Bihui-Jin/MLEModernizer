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

0.73727

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

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

    candidates = []
    comp_root = _find_comp_root()
    if comp_root is not None:
        candidates.append(comp_root)
    candidates.extend(
        [
            "/kaggle/data/tgs-salt-identification-challenge",
            "/kaggle/data",
            "/kaggle/input",
        ]
    )

    src_train_images = src_train_masks = src_test_images = None

    for root in candidates:
        if not root or not os.path.isdir(root):
            continue
        ti = os.path.join(root, "train/images")
        tm = os.path.join(root, "train/masks")
        te = os.path.join(root, "test/images")
        if os.path.isdir(ti) and os.path.isdir(tm) and os.path.isdir(te):
            src_train_images, src_train_masks, src_test_images = ti, tm, te
            break
        ti = os.path.join(root, "tgs-salt-identification-challenge/train/images")
        tm = os.path.join(root, "tgs-salt-identification-challenge/train/masks")
        te = os.path.join(root, "tgs-salt-identification-challenge/test/images")
        if os.path.isdir(ti) and os.path.isdir(tm) and os.path.isdir(te):
            src_train_images, src_train_masks, src_test_images = ti, tm, te
            break

    assert (
        src_train_images and src_train_masks and src_test_images
    ), "Could not locate extracted train/test folders."

    _run("rm -rf train test")
    os.makedirs("train", exist_ok=True)
    os.makedirs("test", exist_ok=True)

    try:
        os.symlink(os.path.abspath(src_train_images), os.path.join("train", "images"))
        os.symlink(os.path.abspath(src_train_masks), os.path.join("train", "masks"))
        os.symlink(os.path.abspath(src_test_images), os.path.join("test", "images"))
    except Exception:
        _run("rm -rf train test")
        shutil.copytree(src_train_images, os.path.join("train", "images"))
        shutil.copytree(src_train_masks, os.path.join("train", "masks"))
        shutil.copytree(src_test_images, os.path.join("test", "images"))

    assert os.path.isdir("train/images"), "train/images not found after linking/copy"
    assert os.path.isdir("train/masks"), "train/masks not found after linking/copy"
    assert os.path.isdir("test/images"), "test/images not found after linking/copy"


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
AUTOTUNE = tf.data.AUTOTUNE

train_img_dir = os.path.join(config.path_train, "images")
train_msk_dir = os.path.join(config.path_train, "masks")
test_img_dir = os.path.join(config.path_test, "images")

n_total = len(train_ids)
n_train = int(0.9 * n_total)

train_ids_train = train_ids[:n_train]
train_ids_eval = train_ids[n_train:]


def _read_png_grayscale(path):
    b = tf.io.read_file(path)
    img = tf.io.decode_png(b, channels=1, dtype=tf.uint8)
    return img


def _resize_image_uint8(img):
    img = tf.image.resize(img, (config.im_height, config.im_width), method="area")
    img = tf.cast(tf.round(img), tf.uint8)
    return img


def _resize_mask_bool(mask):
    m = tf.image.resize(mask, (config.im_height, config.im_width), method="nearest")
    m = tf.cast(m > 127, tf.bool)
    return m


def _load_train_pair(id_bytes):
    id_str = tf.compat.as_str_any(id_bytes)
    img_path = tf.strings.join([train_img_dir, "/", id_str])
    msk_path = tf.strings.join([train_msk_dir, "/", id_str])
    img = _read_png_grayscale(img_path)
    msk = _read_png_grayscale(msk_path)
    img = _resize_image_uint8(img)
    msk = _resize_mask_bool(msk)
    return img, msk


def _flip_lr(img, msk):
    return tf.image.flip_left_right(img), tf.image.flip_left_right(msk)


def _make_train_ds_from_ids(
    ids_list, batch_size, training, cache_in_memory=True, add_flip=False
):
    ids_tensor = tf.constant(ids_list)
    ds = tf.data.Dataset.from_tensor_slices(ids_tensor)
    options = tf.data.Options()
    options.experimental_deterministic = True
    ds = ds.with_options(options)

    if training:
        ds = ds.shuffle(
            buffer_size=len(ids_list), seed=19, reshuffle_each_iteration=True
        )

    ds = ds.map(_load_train_pair, num_parallel_calls=AUTOTUNE)

    if add_flip:
        ds_flip = ds.map(_flip_lr, num_parallel_calls=AUTOTUNE)
        ds = ds.concatenate(ds_flip)
        if training:
            ds = ds.shuffle(
                buffer_size=len(ids_list) * 2, seed=19, reshuffle_each_iteration=True
            )

    if cache_in_memory:
        ds = ds.cache()

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


BATCH_SIZE = 8
train_ds = _make_train_ds_from_ids(
    train_ids_train, BATCH_SIZE, training=True, cache_in_memory=True, add_flip=True
)
eval_ds = _make_train_ds_from_ids(
    train_ids_eval, BATCH_SIZE, training=False, cache_in_memory=True, add_flip=False
)

steps_per_epoch = int(math.ceil((len(train_ids_train) * 2) / BATCH_SIZE))
validation_steps = int(math.ceil(len(train_ids_eval) / BATCH_SIZE))

print("Train ids:", len(train_ids_train), "Eval ids:", len(train_ids_eval))
print("steps_per_epoch:", steps_per_epoch, "validation_steps:", validation_steps)




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

model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["acc"], jit_compile=True
)
model.summary()



## === cell 7
if os.environ.get("PLOT_MODEL", "0") == "1":
    try:
        utils.plot_model(model, expand_nested=True, show_shapes=True)
    except Exception as e:
        print("plot_model skipped due to:", repr(e))
else:
    print("plot_model skipped (set PLOT_MODEL=1 to enable).")



## === cell 8
es = callbacks.EarlyStopping(patience=30, verbose=1, restore_best_weights=True)
rlp = callbacks.ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-12, verbose=1)

results = model.fit(
    train_ds,
    validation_data=eval_ds,
    epochs=300,
    callbacks=[es, rlp],
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=2,
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NotFoundError                             Traceback (most recent call last)
/tmp/ipykernel_11/90976728.py in <cell line: 0>()
      4 rlp = callbacks.ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-12, verbose=1)
      5 
----> 6 results = model.fit(
      7     train_ds,
      8     validation_data=eval_ds,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
     57       e.message += " name: " + name
     58     raise core._status_to_exception(e) from None
---> 59   except TypeError as e:
     60     keras_symbolic_tensors = [x for x in inputs if _is_keras_symbolic_tensor(x)]
     61     if keras_symbolic_tensors:

NotFoundError: Graph execution error:

Detected at node ReadFile defined at (most recent call last):
<stack traces unavailable>
Error in user-defined function passed to ParallelMapDatasetV2:3 transformation with iterator: Iterator::Root::Prefetch::BatchV2::MemoryCacheImpl::Shuffle::Concatenate[0]::ParallelMapV2: train/images/Tensor("args_0:0", shape=(), dtype=string); No such file or directory
	 [[{{node ReadFile}}]]
	 [[IteratorGetNext]] [Op:__inference_multi_step_on_iterator_8913]

## === cell 9
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




## === cell 10
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




## === cell 11
preds_eval = model.predict(eval_ds, verbose=1)

Y_eval_list = []
for _, yb in eval_ds:
    Y_eval_list.append(yb.numpy())
Y_eval = np.concatenate(Y_eval_list, axis=0)

thresholds = np.linspace(0, 1, 50, dtype=np.float32)
thr_grid = np.arange(0.5, 1.0, 0.05, dtype=np.float64)[None, :]  # (1,10)

Y_eval_bool = Y_eval[..., 0].astype(np.bool_, copy=False)  # (N,H,W)
preds_eval_p = preds_eval[..., 0].astype(np.float32, copy=False)  # (N,H,W)

y_true_flat = Y_eval_bool.reshape(Y_eval_bool.shape[0], -1)
preds_flat = preds_eval_p.reshape(preds_eval_p.shape[0], -1)

y_true_sum = y_true_flat.sum(axis=1).astype(np.int64, copy=False)  # (N,)


def _score_thresholds_in_chunks(thresholds_arr, chunk=10):
    ious_out = np.empty((len(thresholds_arr),), dtype=np.float64)
    for s in range(0, len(thresholds_arr), chunk):
        thr_chunk = thresholds_arr[s : s + chunk]
        y_pred = preds_flat[None, ...] > thr_chunk[:, None, None]
        y_pred_sum = y_pred.sum(axis=2).astype(np.int64, copy=False)  # (tc,N)
        inter = (
            np.logical_and(y_true_flat[None, ...], y_pred)
            .sum(axis=2)
            .astype(np.int64, copy=False)
        )  # (tc,N)
        union = y_true_sum[None, :] + y_pred_sum - inter
        union = np.where(union == 0, 1, union)
        iou_img = inter / union.astype(np.float64)  # (tc,N)
        ious_out[s : s + len(thr_chunk)] = (
            (iou_img[:, :, None] > thr_grid[None, None, :]).mean(axis=2).mean(axis=1)
        )
    return ious_out


ious = _score_thresholds_in_chunks(thresholds, chunk=10)

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




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NotFoundError                             Traceback (most recent call last)
/tmp/ipykernel_11/230908186.py in <cell line: 0>()
      1 # CHANGE (timeout fix): materialize eval tensors once via the existing eval_ds pipeline to avoid keeping X/Y arrays.
      2 # Why it speeds up: avoids Python-side image loading duplication and keeps I/O efficient/deterministic.
----> 3 preds_eval = model.predict(eval_ds, verbose=1)
      4 
      5 # Collect Y_eval from eval_ds deterministically (same order, no shuffle, cached).

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

NotFoundError: {{function_node __wrapped__IteratorGetNext_output_types_2_device_/job:localhost/replica:0/task:0/device:CPU:0}} Error in user-defined function passed to ParallelMapDatasetV2:12 transformation with iterator: Iterator::Root::Prefetch::BatchV2::MemoryCacheImpl::ParallelMapV2: train/images/Tensor("args_0:0", shape=(), dtype=string); No such file or directory
	 [[{{node ReadFile}}]] [Op:IteratorGetNext] name: 

## === cell 12
def _load_test_with_size(id_bytes):
    id_str = tf.compat.as_str_any(id_bytes)
    img_path = tf.strings.join([test_img_dir, "/", id_str])
    img = _read_png_grayscale(img_path)  # uint8 [H,W,1]
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    img_r = _resize_image_uint8(img)
    return id_str, tf.stack([h, w]), img_r


test_ids_tensor = tf.constant(test_ids)
test_meta_ds = tf.data.Dataset.from_tensor_slices(test_ids_tensor)
options = tf.data.Options()
options.experimental_deterministic = True
test_meta_ds = test_meta_ds.with_options(options)
test_meta_ds = test_meta_ds.map(
    _load_test_with_size, num_parallel_calls=AUTOTUNE
).cache()

ids_out = []
sizes_test = []
X_test_list = []
for id_str, sz, img in test_meta_ds:
    ids_out.append(id_str.numpy().decode("utf-8"))
    sizes_test.append(sz.numpy().tolist())
    X_test_list.append(img.numpy())
X_test = np.stack(X_test_list, axis=0).astype(np.uint8, copy=False)

assert ids_out == test_ids, "Test id order mismatch."

X_test = np.ascontiguousarray(X_test)
print("Done!", "X_test shape:", X_test.shape)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NotFoundError                             Traceback (most recent call last)
/tmp/ipykernel_11/3250415921.py in <cell line: 0>()
     25 sizes_test = []
     26 X_test_list = []
---> 27 for id_str, sz, img in test_meta_ds:
     28     ids_out.append(id_str.numpy().decode("utf-8"))
     29     sizes_test.append(sz.numpy().tolist())

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in __next__(self)
    824   def __next__(self):
    825     try:
--> 826       return self._next_internal()
    827     except errors.OutOfRangeError:
    828       raise StopIteration

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in _next_internal(self)
    774     # to communicate that there is no more data to iterate over.
    775     with context.execution_mode(context.SYNC):
--> 776       ret = gen_dataset_ops.iterator_get_next(
    777           self._iterator_resource,
    778           output_types=self._flat_output_types,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_dataset_ops.py in iterator_get_next(iterator, output_types, output_shapes, name)
   3084       return _result
   3085     except _core._NotOkStatusException as e:
-> 3086       _ops.raise_from_not_ok_status(e, name)
   3087     except _core._FallbackException:
   3088       pass

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

NotFoundError: {{function_node __wrapped__IteratorGetNext_output_types_3_device_/job:localhost/replica:0/task:0/device:CPU:0}} Error in user-defined function passed to ParallelMapDatasetV2:20 transformation with iterator: Iterator::Root::Prefetch::MemoryCacheImpl::ParallelMapV2: test/images/Tensor("args_0:0", shape=(), dtype=string); No such file or directory
	 [[{{node ReadFile}}]] [Op:IteratorGetNext] name: 

## === cell 13
test_ds = (
    tf.data.Dataset.from_tensor_slices(X_test).batch(16).prefetch(tf.data.AUTOTUNE)
)
preds_test = model.predict(test_ds, verbose=1)

target_h, target_w = sizes_test[0]
preds_ch = preds_test[..., 0].astype(np.float32, copy=False)  # (N,128,128)

preds_test_upsampled = np.empty(
    (preds_ch.shape[0], target_h, target_w), dtype=np.float32
)

for i in trange(preds_ch.shape[0], desc="Upsampling preds"):
    preds_test_upsampled[i] = cv2.resize(
        preds_ch[i], (target_w, target_h), interpolation=cv2.INTER_LINEAR
    )



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4179984429.py in <cell line: 0>()
      1 test_ds = (
----> 2     tf.data.Dataset.from_tensor_slices(X_test).batch(16).prefetch(tf.data.AUTOTUNE)
      3 )
      4 preds_test = model.predict(test_ds, verbose=1)
      5 

NameError: name 'X_test' is not defined

## === cell 14
import shutil

shutil.rmtree("train", ignore_errors=True)
shutil.rmtree("test", ignore_errors=True)




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

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1560817980.py in <cell line: 0>()
     25 pred_dict = {}
     26 for i, fn in tqdm(enumerate(test_ids), total=len(test_ids)):
---> 27     mask = (preds_test_upsampled[i] > threshold_best).astype(np.uint8)
     28     pred_dict[fn[:-4]] = RLenc(mask)
     29 

NameError: name 'preds_test_upsampled' is not defined
