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
import os, sys, random, warnings, math, zipfile

warnings.filterwarnings("ignore")

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

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
from tensorflow.keras import models, Input, layers, callbacks, utils, optimizers




## === cell 1
class config:
    im_width = 128
    im_height = 128
    im_chan = 1
    path_train = "kaggle/data/train/"
    path_test = "kaggle/data/test/"




## === cell 2
if not os.path.isdir(config.path_train):
    os.makedirs(config.path_train, exist_ok=True)
    with zipfile.ZipFile(
        "../input/tgs-salt-identification-challenge/train.zip", "r"
    ) as z:
        z.extractall("kaggle/data/")
if not os.path.isdir(config.path_test):
    os.makedirs(config.path_test, exist_ok=True)
    with zipfile.ZipFile(
        "../input/tgs-salt-identification-challenge/test.zip", "r"
    ) as z:
        z.extractall("kaggle/data/")



## === cell 3
random.seed(19)
train_img_dir = os.path.join(config.path_train, "images")
train_mask_dir = os.path.join(config.path_train, "masks")
test_img_dir = os.path.join(config.path_test, "images")

sample_ids = random.choices(os.listdir(train_img_dir), k=6)
fig = plt.figure(figsize=(20, 6))
for j, img_name in enumerate(sample_ids):
    img = load_img(os.path.join(train_img_dir, img_name))
    img_mask = load_img(os.path.join(train_mask_dir, img_name))
    plt.subplot(2, 6, j * 2 + 1)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.subplot(2, 6, j * 2 + 2)
    plt.imshow(img_mask, cmap="gray")
    plt.axis("off")
fig.suptitle("Sample Images", fontsize=24)
plt.show()



## === cell 4
train_ids = os.listdir(train_img_dir)
test_ids = os.listdir(test_img_dir)




## === cell 5
def load_resize_gray(path, target_h, target_w):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # shape (h, w)
    img = cv2.resize(
        img, (target_w, target_h), interpolation=cv2.INTER_LINEAR  # cv2 uses (w, h)
    )
    img = img[..., np.newaxis]  # add channel dim
    return img


X = np.empty(
    (len(train_ids), config.im_height, config.im_width, config.im_chan), dtype=np.uint8
)
Y = np.empty((len(train_ids), config.im_height, config.im_width, 1), dtype=bool)

print("Loading and resizing train images and masks...")
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    img_path = os.path.join(train_img_dir, id_)
    mask_path = os.path.join(train_mask_dir, id_)
    X[n] = load_resize_gray(img_path, config.im_height, config.im_width).astype(
        np.uint8
    )
    m = load_resize_gray(mask_path, config.im_height, config.im_width)
    Y[n] = (m > 127).astype(bool)  # binarize mask
print("Done!")
print("X shape:", X.shape)
print("Y shape:", Y.shape)



## === cell 6
split_idx = int(0.9 * len(X))
X_train_orig = X[:split_idx]
Y_train_orig = Y[:split_idx]
X_eval = X[split_idx:]
Y_eval = Y[split_idx:]

X_hflip = np.flip(X_train_orig, axis=2)  # horizontal flip (width axis)
Y_hflip = np.flip(Y_train_orig, axis=2)

X_vflip = np.flip(X_train_orig, axis=1)  # vertical flip (height axis)
Y_vflip = np.flip(Y_train_orig, axis=1)

X_train = np.concatenate(
    [
        X_train_orig,
        X_hflip,
        X_vflip,
        np.flip(np.concatenate([X_train_orig, X_hflip]), axis=1),
    ],
    axis=0,
)
Y_train = np.concatenate(
    [
        Y_train_orig,
        Y_hflip,
        Y_vflip,
        np.flip(np.concatenate([Y_train_orig, Y_hflip]), axis=1),
    ],
    axis=0,
)

print("After augmentation:")
print("X_train shape:", X_train.shape, "X_eval shape:", X_eval.shape)
print("Y_train shape:", Y_train.shape, "Y_eval shape:", Y_eval.shape)




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
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()



## === cell 8
es = callbacks.EarlyStopping(patience=10, verbose=1, restore_best_weights=True)
rlp = callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, verbose=1)

results = model.fit(
    X_train,
    Y_train,
    validation_data=(X_eval, Y_eval),
    batch_size=8,
    epochs=30,
    callbacks=[es, rlp],
    verbose=2,
)



## === cell 9
sns.set_style("darkgrid")
fig, ax = plt.subplots(2, 1, figsize=(20, 8))
history = pd.DataFrame(results.history)
history[["loss", "val_loss"]].plot(ax=ax[0])
history[["accuracy", "val_accuracy"]].plot(ax=ax[1])
fig.suptitle("Learning Curve", fontsize=24)
plt.show()




## === cell 10
def iou_metric(y_true, y_pred, print_table=False):
    true_objects = 2
    pred_objects = 2

    intersection = np.histogram2d(
        y_true.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects)
    )[0]

    area_true = np.histogram(y_true, bins=true_objects)[0]
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
        tp = np.sum(true_positives)
        fp = np.sum(false_positives)
        fn = np.sum(false_negatives)
        return tp, fp, fn

    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        p = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        if print_table:
            print(f"{t:.3f}\t{tp}\t{fp}\t{fn}\t{p:.3f}")
        prec.append(p)
    if print_table:
        print(f"AP\t-\t-\t-\t{np.mean(prec):.3f}")
    return np.mean(prec)


def iou_metric_batch(y_true_batch, y_pred_batch):
    return np.mean(
        [
            iou_metric(y_true, y_pred)
            for y_true, y_pred in zip(y_true_batch, y_pred_batch)
        ]
    )




## === cell 11
preds_eval = model.predict(X_eval, verbose=0)

thresholds = np.linspace(0, 1, 50)
ious = []
for thr in tqdm(thresholds):
    ious.append(iou_metric_batch(Y_eval, (preds_eval > thr).astype(np.uint8)))
ious = np.array(ious)

if len(ious) > 0:
    best_idx = np.argmax(ious[9:-10]) + 9
    iou_best = ious[best_idx]
    threshold_best = thresholds[best_idx]
else:
    threshold_best = 0.5
    iou_best = 0.0

plt.plot(thresholds, ious)
plt.plot(threshold_best, iou_best, "xr", label="Best threshold")
plt.xlabel("Threshold")
plt.ylabel("IoU")
plt.title(f"Threshold vs IoU (best={threshold_best:.2f}, IoU={iou_best:.3f})")
plt.legend()
plt.show()



## === cell 12
X_test = np.empty(
    (len(test_ids), config.im_height, config.im_width, config.im_chan), dtype=np.uint8
)
sizes_test = []
print("Loading and resizing test images...")
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    img_path = os.path.join(test_img_dir, id_)
    x = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    sizes_test.append([x.shape[0], x.shape[1]])
    x = cv2.resize(
        x, (config.im_width, config.im_height), interpolation=cv2.INTER_LINEAR
    )
    X_test[n] = x[..., np.newaxis].astype(np.uint8)
print("Done!")



## === cell 13
preds_test = model.predict(X_test, verbose=0)
preds_test_upsampled = []
for i in trange(len(preds_test)):
    up = cv2.resize(
        np.squeeze(preds_test[i]),
        (sizes_test[i][1], sizes_test[i][0]),  # width, height order for cv2
        interpolation=cv2.INTER_LINEAR,
    )
    preds_test_upsampled.append(up)




## === cell 14
def RLenc(img, order="F", format=True):
    """
    img: binary mask (2D array)
    Returns run‑length encoding as required by the competition.
    """
    bytes = img.reshape(-1, order=order)
    runs = []
    r = 0
    pos = 1
    for c in bytes:
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
    if format:
        return " ".join(f"{p} {l}" for p, l in runs)
    return runs


pred_dict = {
    fn[:-4]: RLenc(np.round(preds_test_upsampled[i] > threshold_best))
    for i, fn in enumerate(test_ids)
}



## === cell 15
sub = pd.DataFrame.from_dict(pred_dict, orient="index")
sub.index.name = "id"
sub.columns = ["rle_mask"]
sub.to_csv("submission.csv")
print("Submission file written to submission.csv")
