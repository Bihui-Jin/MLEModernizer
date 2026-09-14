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

0.71819

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, sys, random, warnings

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import models, Input, layers, callbacks, utils
from tensorflow.keras.preprocessing.image import (
    array_to_img,
    img_to_array,
    load_img,
    ImageDataGenerator,
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.auto import tqdm, trange
from itertools import chain

import cv2
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.morphology import label

print("TensorFlow:", tf.__version__)

SEED = 19
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
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
INPUT_BASE = "/kaggle/input/tgs-salt-identification-challenge"
if not os.path.exists(INPUT_BASE):
    INPUT_BASE = "../input/tgs-salt-identification-challenge"

train_zip = os.path.join(INPUT_BASE, "train.zip")
test_zip = os.path.join(INPUT_BASE, "test.zip")

print("Using INPUT_BASE:", INPUT_BASE)
print("train_zip exists:", os.path.exists(train_zip))
print("test_zip exists:", os.path.exists(test_zip))

if not os.path.exists(config.path_train):
    os.makedirs(config.path_train, exist_ok=True)
if not os.path.exists(config.path_test):
    os.makedirs(config.path_test, exist_ok=True)

if not os.path.exists(os.path.join(config.path_train, "train")) and not os.path.exists(
    os.path.join(config.path_train, "images")
):
    os.system(f'unzip -q "{train_zip}" -d "{config.path_train}"')
if not os.path.exists(os.path.join(config.path_test, "test")) and not os.path.exists(
    os.path.join(config.path_test, "images")
):
    os.system(f'unzip -q "{test_zip}" -d "{config.path_test}"')


def resolve_dir(base_dir, expected_leaf):
    """
    Return a directory that contains expected_leaf as a direct child, trying a few common layouts.
    """
    candidates = [
        base_dir,
        os.path.join(
            base_dir, os.path.basename(base_dir.rstrip("/"))
        ),  # e.g., train/train
        os.path.join(base_dir, "train"),
        os.path.join(base_dir, "test"),
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, expected_leaf)):
            return c
    return base_dir


TRAIN_ROOT = resolve_dir(config.path_train, "images")
TEST_ROOT = resolve_dir(config.path_test, "images")

print("Resolved TRAIN_ROOT:", TRAIN_ROOT)
print("Resolved TEST_ROOT:", TEST_ROOT)
print("Train images dir exists:", os.path.exists(os.path.join(TRAIN_ROOT, "images")))
print("Train masks dir exists:", os.path.exists(os.path.join(TRAIN_ROOT, "masks")))
print("Test images dir exists:", os.path.exists(os.path.join(TEST_ROOT, "images")))



## === cell 3
train_images_dir = os.path.join(TRAIN_ROOT, "images")
train_masks_dir = os.path.join(TRAIN_ROOT, "masks")

DO_PLOTS = False
if DO_PLOTS and os.path.exists(train_images_dir) and os.path.exists(train_masks_dir):
    ids = random.choices(os.listdir(train_images_dir), k=6)
    fig = plt.figure(figsize=(20, 6))
    for j, img_name in enumerate(ids):
        q = j + 1
        img = load_img(os.path.join(train_images_dir, img_name))
        img_mask = load_img(os.path.join(train_masks_dir, img_name))

        plt.subplot(2, 6, q * 2 - 1)
        plt.imshow(img)
        plt.axis("off")
        plt.subplot(2, 6, q * 2)
        plt.imshow(img_mask)
        plt.axis("off")
    fig.suptitle("Sample Images", fontsize=24)
    plt.show()
else:
    print("Skipping sample plot.")



## === cell 4
train_images_path = os.path.join(TRAIN_ROOT, "images")
test_images_path = os.path.join(TEST_ROOT, "images")

if not os.path.exists(train_images_path):
    raise FileNotFoundError(f"Train images path not found: {train_images_path}")
if not os.path.exists(test_images_path):
    raise FileNotFoundError(f"Test images path not found: {test_images_path}")

train_ids = sorted([f for f in os.listdir(train_images_path) if f.endswith(".png")])
test_ids = sorted([f for f in os.listdir(test_images_path) if f.endswith(".png")])

print("Num train images:", len(train_ids))
print("Num test images:", len(test_ids))
print("Example train id:", train_ids[0] if train_ids else None)
print("Example test id:", test_ids[0] if test_ids else None)



## === cell 5
from concurrent.futures import ThreadPoolExecutor

CACHE_DIR = "./cache_preproc"
os.makedirs(CACHE_DIR, exist_ok=True)
train_cache_npz = os.path.join(
    CACHE_DIR, f"train_{config.im_height}x{config.im_width}_g1.npz"
)


def _read_train_pair(id_):
    img_path = os.path.join(TRAIN_ROOT, "images", id_)
    msk_path = os.path.join(TRAIN_ROOT, "masks", id_)

    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    x = img_bgr[:, :, 1]
    x = cv2.resize(
        x, (config.im_width, config.im_height), interpolation=cv2.INTER_LINEAR
    ).astype(np.uint8, copy=False)

    mask_bgr = cv2.imread(msk_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    mask = mask_bgr[:, :, 1]
    mask = cv2.resize(
        mask, (config.im_width, config.im_height), interpolation=cv2.INTER_NEAREST
    )
    y = mask > 127

    return x, y


if os.path.exists(train_cache_npz):
    with np.load(train_cache_npz) as z:
        X_train = z["X_train"]
        Y_train = z["Y_train"].astype(bool, copy=False)
    print("Loaded cached train arrays:", X_train.shape, Y_train.shape)
else:
    X_train = np.zeros(
        (len(train_ids), config.im_height, config.im_width, config.im_chan),
        dtype=np.uint8,
    )
    Y_train = np.zeros(
        (len(train_ids), config.im_height, config.im_width, 1), dtype=bool
    )

    print("Getting and resizing train images and masks ... ")
    sys.stdout.flush()

    max_workers = min(8, (os.cpu_count() or 2))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for n, (x, y) in tqdm(
            enumerate(ex.map(_read_train_pair, train_ids)), total=len(train_ids)
        ):
            X_train[n, ..., 0] = x
            Y_train[n, ..., 0] = y

    np.savez_compressed(
        train_cache_npz, X_train=X_train, Y_train=Y_train.astype(np.uint8)
    )
    print("Cached train arrays to:", train_cache_npz)
    print("Done!")



## === cell 6
X_flip = X_train[:, :, ::-1, :]
Y_flip = Y_train[:, :, ::-1, :]
X_train = np.concatenate([X_train, X_flip], axis=0)
Y_train = np.concatenate([Y_train, Y_flip], axis=0)

print("Augmented X_train:", X_train.shape)
print("Augmented Y_train:", Y_train.shape)



## === cell 7
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

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["acc"],
    jit_compile=True,
    steps_per_execution=16,
)
model.summary()



## === cell 8
DO_PLOT_MODEL = False
if DO_PLOT_MODEL:
    try:
        utils.plot_model(model, expand_nested=True, show_shapes=True)
    except Exception as e:
        print("Skipping plot_model due to:", repr(e))



## === cell 9
es = callbacks.EarlyStopping(patience=30, verbose=1, restore_best_weights=True)
rlp = callbacks.ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-12, verbose=1)

n = X_train.shape[0]
val_size = int(round(0.1 * n))
train_size = n - val_size

X_tr, Y_tr = X_train[:train_size], Y_train[:train_size]
X_va, Y_va = X_train[train_size:], Y_train[train_size:]

BATCH = 8


class NumpySequence(utils.Sequence):
    def __init__(self, X, Y, batch_size, shuffle, seed):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.indices = np.arange(len(X), dtype=np.int32)
        self.on_epoch_end()

    def __len__(self):
        return (len(self.indices) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, i):
        sl = slice(i * self.batch_size, (i + 1) * self.batch_size)
        idx = self.indices[sl]
        return self.X[idx], self.Y[idx]

    def on_epoch_end(self):
        if self.shuffle:
            rng = np.random.RandomState(self.seed + self.epoch)
            rng.shuffle(self.indices)
        self.epoch += 1


train_seq = NumpySequence(X_tr, Y_tr, batch_size=BATCH, shuffle=True, seed=SEED)
val_seq = NumpySequence(X_va, Y_va, batch_size=BATCH, shuffle=False, seed=SEED)

results = model.fit(
    train_seq,
    validation_data=val_seq,
    epochs=300,
    callbacks=[es, rlp],
    verbose=1,
    workers=1,  # keep deterministic ordering / no MP nondeterminism
    use_multiprocessing=False,
)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/68507020.py in <cell line: 0>()
     43 val_seq = NumpySequence(X_va, Y_va, batch_size=BATCH, shuffle=False, seed=SEED)
     44 
---> 45 results = model.fit(
     46     train_seq,
     47     validation_data=val_seq,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    117             return fn(*args, **kwargs)
    118         except Exception as e:
--> 119             filtered_tb = _process_traceback_frames(e.__traceback__)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`

TypeError: TensorFlowTrainer.fit() got an unexpected keyword argument 'workers'

## === cell 10
DO_PLOTS = False
if DO_PLOTS:
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(2, 1, figsize=(20, 8))
    history = pd.DataFrame(results.history)
    history[["loss", "val_loss"]].plot(ax=ax[0])
    history[["acc", "val_acc"]].plot(ax=ax[1])
    fig.suptitle("Learning Curve", fontsize=24)
    plt.show()
else:
    print("Skipping learning curve plot.")



## === cell 11
test_cache_npz = os.path.join(
    CACHE_DIR, f"test_{config.im_height}x{config.im_width}_g1.npz"
)


def _read_test_img(id_):
    img_path = os.path.join(TEST_ROOT, "images", id_)
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    x = img_bgr[:, :, 1]
    size = (x.shape[0], x.shape[1])
    x = cv2.resize(
        x, (config.im_width, config.im_height), interpolation=cv2.INTER_LINEAR
    ).astype(np.uint8, copy=False)
    return x, size


if os.path.exists(test_cache_npz):
    with np.load(test_cache_npz, allow_pickle=True) as z:
        X_test = z["X_test"]
        sizes_test = z["sizes_test"].tolist()
    print("Loaded cached test arrays:", X_test.shape, len(sizes_test))
else:
    X_test = np.zeros(
        (len(test_ids), config.im_height, config.im_width, config.im_chan),
        dtype=np.uint8,
    )
    sizes_test = []
    print("Getting and resizing test images ... ")
    sys.stdout.flush()

    max_workers = min(8, (os.cpu_count() or 2))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for n, (x, size) in tqdm(
            enumerate(ex.map(_read_test_img, test_ids)), total=len(test_ids)
        ):
            X_test[n, ..., 0] = x
            sizes_test.append([size[0], size[1]])

    np.savez_compressed(
        test_cache_npz, X_test=X_test, sizes_test=np.array(sizes_test, dtype=np.int16)
    )
    print("Cached test arrays to:", test_cache_npz)
    print("Done!")




## === cell 12
class NumpyPredictSequence(utils.Sequence):
    def __init__(self, X, batch_size):
        self.X = X
        self.batch_size = batch_size

    def __len__(self):
        return (len(self.X) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, i):
        sl = slice(i * self.batch_size, (i + 1) * self.batch_size)
        return self.X[sl]


preds_test = model.predict(
    NumpyPredictSequence(X_test, batch_size=64),
    verbose=1,
    workers=1,
    use_multiprocessing=False,
)

sizes_arr = np.asarray(sizes_test, dtype=np.int32)
orig_h, orig_w = int(sizes_arr[0, 0]), int(sizes_arr[0, 1])
all_same = bool(np.all(sizes_arr[:, 0] == orig_h) and np.all(sizes_arr[:, 1] == orig_w))

preds2d = np.squeeze(preds_test, axis=-1)  # (N,128,128)

if all_same:
    preds_test_upsampled = np.empty(
        (preds2d.shape[0], orig_h, orig_w), dtype=preds2d.dtype
    )
    for i in range(preds2d.shape[0]):
        preds_test_upsampled[i] = cv2.resize(
            preds2d[i], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
        )
else:
    preds_test_upsampled = [None] * preds2d.shape[0]
    for i in trange(preds2d.shape[0]):
        h, w = int(sizes_arr[i, 0]), int(sizes_arr[i, 1])
        preds_test_upsampled[i] = cv2.resize(
            preds2d[i], (w, h), interpolation=cv2.INTER_LINEAR
        )



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2591160214.py in <cell line: 0>()
     13 
     14 
---> 15 preds_test = model.predict(
     16     NumpyPredictSequence(X_test, batch_size=64),
     17     verbose=1,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    117             return fn(*args, **kwargs)
    118         except Exception as e:
--> 119             filtered_tb = _process_traceback_frames(e.__traceback__)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`

TypeError: TensorFlowTrainer.predict() got an unexpected keyword argument 'workers'

## === cell 13
os.system("rm -rf train")
os.system("rm -rf test")




## === cell 14
def RLenc(img, order="F", format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    returns run length as an array or string (if format is True)
    """
    x = np.asarray(img)
    if x.ndim != 2:
        x = np.squeeze(x)
    pixels = x.reshape(-1, order=order).astype(np.uint8, copy=False)

    pads = np.zeros(pixels.size + 2, dtype=np.uint8)
    pads[1:-1] = pixels
    changes = np.flatnonzero(pads[1:] != pads[:-1])

    starts = changes[0::2] + 1  # 1-indexed positions in flattened mask
    ends = changes[1::2]
    lengths = ends - changes[0::2]

    if not format:
        return list(zip(starts.tolist(), lengths.tolist()))
    if starts.size == 0:
        return ""
    out = np.empty(starts.size * 2, dtype=object)
    out[0::2] = starts.astype(str)
    out[1::2] = lengths.astype(str)
    return " ".join(out.tolist())


pred_dict = {}
round_ = np.round

if isinstance(preds_test_upsampled, np.ndarray):
    for i, fn in enumerate(test_ids):
        pred_dict[fn[:-4]] = RLenc(round_(preds_test_upsampled[i]))
else:
    for i, fn in enumerate(test_ids):
        pred_dict[fn[:-4]] = RLenc(round_(preds_test_upsampled[i]))

sample_path = os.path.join(INPUT_BASE, "sample_submission.csv")
if not os.path.exists(sample_path):
    sample_path = "../input/tgs-salt-identification-challenge/sample_submission.csv"

sample = pd.read_csv(sample_path)
sample["rle_mask"] = sample["id"].map(pred_dict).fillna("")

sample.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sample.shape)
print(sample.head())

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1796661818.py in <cell line: 0>()
     32 
     33 # Handle preds_test_upsampled as either ndarray (common) or list (mixed sizes)
---> 34 if isinstance(preds_test_upsampled, np.ndarray):
     35     for i, fn in enumerate(test_ids):
     36         pred_dict[fn[:-4]] = RLenc(round_(preds_test_upsampled[i]))

NameError: name 'preds_test_upsampled' is not defined
