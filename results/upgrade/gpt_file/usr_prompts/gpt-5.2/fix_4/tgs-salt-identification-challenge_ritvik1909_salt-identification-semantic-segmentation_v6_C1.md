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
X_train = np.zeros(
    (len(train_ids), config.im_height, config.im_width, config.im_chan), dtype=np.uint8
)
Y_train = np.zeros((len(train_ids), config.im_height, config.im_width, 1), dtype=bool)

print("Getting and resizing train images and masks ... ")
sys.stdout.flush()

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    img_path = os.path.join(TRAIN_ROOT, "images", id_)
    msk_path = os.path.join(TRAIN_ROOT, "masks", id_)

    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    x = img_bgr[:, :, 1]
    x = cv2.resize(
        x, (config.im_width, config.im_height), interpolation=cv2.INTER_LINEAR
    )
    X_train[n, ..., 0] = x.astype(np.uint8, copy=False)

    mask_bgr = cv2.imread(msk_path, cv2.IMREAD_COLOR)
    mask = mask_bgr[:, :, 1]
    mask = cv2.resize(
        mask, (config.im_width, config.im_height), interpolation=cv2.INTER_NEAREST
    )
    Y_train[n, ..., 0] = mask > 127  # keep boolean mask

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
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
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

results = model.fit(
    X_train,
    Y_train,
    validation_split=0.1,
    batch_size=8,
    epochs=300,
    callbacks=[es, rlp],
)




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
X_test = np.zeros(
    (len(test_ids), config.im_height, config.im_width, config.im_chan), dtype=np.uint8
)
sizes_test = []
print("Getting and resizing test images ... ")
sys.stdout.flush()

for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    img_path = os.path.join(TEST_ROOT, "images", id_)
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    x = img_bgr[:, :, 1]
    sizes_test.append([x.shape[0], x.shape[1]])
    x = cv2.resize(
        x, (config.im_width, config.im_height), interpolation=cv2.INTER_LINEAR
    )
    X_test[n, ..., 0] = x.astype(np.uint8, copy=False)

print("Done!")




## === cell 12
preds_test = model.predict(X_test, verbose=1, batch_size=32)

orig_h, orig_w = sizes_test[0]
if not all((h == orig_h and w == orig_w) for h, w in sizes_test):
    preds_test_upsampled = []
    for i in trange(len(preds_test)):
        pred = np.squeeze(preds_test[i])
        up = cv2.resize(
            pred, (sizes_test[i][1], sizes_test[i][0]), interpolation=cv2.INTER_LINEAR
        )
        preds_test_upsampled.append(up)
else:
    preds2d = np.squeeze(preds_test, axis=-1)  # (N,128,128)
    preds_test_upsampled = [
        cv2.resize(preds2d[i], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        for i in range(preds2d.shape[0])
    ]




## === cell 13
os.system("rm -rf train")
os.system("rm -rf test")




## === cell 14
def RLenc(img, order="F", format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes_ = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []
    r = 0
    pos = 1
    for c in bytes_:
        if c == 0:
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ""
        for rr in runs:
            z += "{} {} ".format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs


pred_dict = {
    fn[:-4]: RLenc(np.round(preds_test_upsampled[i]))
    for i, fn in tqdm(enumerate(test_ids), total=len(test_ids))
}

sample_path = os.path.join(INPUT_BASE, "sample_submission.csv")
if not os.path.exists(sample_path):
    sample_path = "../input/tgs-salt-identification-challenge/sample_submission.csv"

sample = pd.read_csv(sample_path)
sample["rle_mask"] = sample["id"].map(pred_dict).fillna("")

sample.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sample.shape)
print(sample.head())
