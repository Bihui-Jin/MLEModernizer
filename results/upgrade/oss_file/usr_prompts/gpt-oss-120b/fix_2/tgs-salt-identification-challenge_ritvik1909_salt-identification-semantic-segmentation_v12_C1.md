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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = (
    "python"  # fix TensorFlow protobuf issue
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.auto import tqdm, trange
from itertools import chain
import zipfile

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
from tensorflow.keras import (
    models,
    Input,
    layers,
    callbacks,
    utils,
    optimizers,
    applications,
)




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
def unzip_to(dest_path, zip_path):
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_path)


zip_train = "../input/tgs-salt-identification-challenge/train.zip"
zip_test = "../input/tgs-salt-identification-challenge/test.zip"

if not os.path.isfile(zip_train):
    zip_train = "train.zip"
if not os.path.isfile(zip_test):
    zip_test = "test.zip"

unzip_to("train", zip_train)
unzip_to("test", zip_test)




## === cell 3
train_ids = next(os.walk(os.path.join(config.path_train, "images")))[2]
test_ids = next(os.walk(os.path.join(config.path_test, "images")))[2]

print(f"Found {len(train_ids)} training images and {len(test_ids)} test images.")




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
StopIteration                             Traceback (most recent call last)
/tmp/ipykernel_11/1765043256.py in <cell line: 0>()
      1 # collect image filenames
----> 2 train_ids = next(os.walk(os.path.join(config.path_train, "images")))[2]
      3 test_ids = next(os.walk(os.path.join(config.path_test, "images")))[2]
      4 
      5 print(f"Found {len(train_ids)} training images and {len(test_ids)} test images.")

StopIteration: 

## === cell 4
random.seed(19)
sample_ids = random.choices(train_ids, k=min(6, len(train_ids)))
fig = plt.figure(figsize=(20, 6))
for j, img_name in enumerate(sample_ids):
    img = load_img(os.path.join(config.path_train, "images", img_name))
    img_mask = load_img(os.path.join(config.path_train, "masks", img_name))
    plt.subplot(2, 6, j * 2 + 1)
    plt.imshow(img)
    plt.subplot(2, 6, j * 2 + 2)
    plt.imshow(img_mask)
fig.suptitle("Sample Images", fontsize=24)
plt.close(fig)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/654274528.py in <cell line: 0>()
      1 # optional visual check – will not raise if any image is missing
      2 random.seed(19)
----> 3 sample_ids = random.choices(train_ids, k=min(6, len(train_ids)))
      4 fig = plt.figure(figsize=(20, 6))
      5 for j, img_name in enumerate(sample_ids):

NameError: name 'train_ids' is not defined

## === cell 5
X = np.zeros(
    (len(train_ids), config.im_height, config.im_width, config.im_chan), dtype=np.uint8
)
Y = np.zeros((len(train_ids), config.im_height, config.im_width, 1), dtype=bool)

print("Loading and resizing training images and masks ...")
for n, img_id in tqdm(enumerate(train_ids), total=len(train_ids)):
    img = img_to_array(
        load_img(
            os.path.join(config.path_train, "images", img_id), color_mode="grayscale"
        )
    )
    img = resize(
        img,
        (config.im_height, config.im_width, 1),
        mode="constant",
        preserve_range=True,
    )
    X[n] = img

    mask = img_to_array(
        load_img(
            os.path.join(config.path_train, "masks", img_id), color_mode="grayscale"
        )
    )
    mask = resize(
        mask,
        (config.im_height, config.im_width, 1),
        mode="constant",
        preserve_range=True,
    )
    Y[n] = mask.astype(bool)

print("Done loading data.")
print("X shape:", X.shape, "Y shape:", Y.shape)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2108559803.py in <cell line: 0>()
      1 X = np.zeros(
----> 2     (len(train_ids), config.im_height, config.im_width, config.im_chan), dtype=np.uint8
      3 )
      4 Y = np.zeros((len(train_ids), config.im_height, config.im_width, 1), dtype=bool)
      5 

NameError: name 'train_ids' is not defined

## === cell 6
split_idx = int(0.9 * len(X))
X_train, X_eval = X[:split_idx], X[split_idx:]
Y_train, Y_eval = Y[:split_idx], Y[split_idx:]

X_train = np.append(X_train, [np.fliplr(x) for x in X_train], axis=0)
Y_train = np.append(Y_train, [np.fliplr(y) for y in Y_train], axis=0)
X_train = np.append(X_train, [np.flipud(x) for x in X_train], axis=0)
Y_train = np.append(Y_train, [np.flipud(y) for y in Y_train], axis=0)

del X, Y
print("After augmentation -> X_train:", X_train.shape, "Y_train:", Y_train.shape)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1369928541.py in <cell line: 0>()
      1 # split 90/10 for train/validation
----> 2 split_idx = int(0.9 * len(X))
      3 X_train, X_eval = X[:split_idx], X[split_idx:]
      4 Y_train, Y_eval = Y[:split_idx], Y[split_idx:]
      5 

NameError: name 'X' is not defined

## === cell 7
def convolution_block(
    x, filters, size, strides=(1, 1), padding="same", activation=True
):
    x = layers.Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    if activation:
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


def UXception(input_shape=(None, None, 3)):
    backbone = applications.Xception(
        input_shape=input_shape, weights="imagenet", include_top=False
    )
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

    uconv0 = layers.Dropout(0.05)(uconv0)
    output_layer = layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(
        uconv0
    )

    model = models.Model(inputs=input_layer, outputs=output_layer)
    return model




## === cell 8
input_layer = Input(shape=(config.im_height, config.im_width, config.im_chan))
scaled = layers.Lambda(lambda x: x / 255.0)(input_layer)
x = layers.Conv2D(3, 1, activation="relu", padding="same")(scaled)
output_layer = UXception(input_shape=(config.im_height, config.im_width, 3))(x)

model = models.Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()




## === cell 9
es = callbacks.EarlyStopping(patience=10, verbose=1, restore_best_weights=True)
rlp = callbacks.ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-12, verbose=1)

history = model.fit(
    X_train,
    Y_train,
    validation_data=(X_eval, Y_eval),
    batch_size=8,
    epochs=20,  # reduced from 300 to keep runtime reasonable
    callbacks=[es, rlp],
    verbose=2,
)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2526200779.py in <cell line: 0>()
      4 
      5 history = model.fit(
----> 6     X_train,
      7     Y_train,
      8     validation_data=(X_eval, Y_eval),

NameError: name 'X_train' is not defined

## === cell 10
threshold_best = 0.5




## === cell 11
X_test = np.zeros(
    (len(test_ids), config.im_height, config.im_width, config.im_chan), dtype=np.uint8
)
sizes_test = []
print("Loading and resizing test images ...")
for n, img_id in tqdm(enumerate(test_ids), total=len(test_ids)):
    img = img_to_array(
        load_img(
            os.path.join(config.path_test, "images", img_id), color_mode="grayscale"
        )
    )
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(
        img,
        (config.im_height, config.im_width, 1),
        mode="constant",
        preserve_range=True,
    )
    X_test[n] = img
print("Done.")




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3911184475.py in <cell line: 0>()
      1 # Prepare test data
      2 X_test = np.zeros(
----> 3     (len(test_ids), config.im_height, config.im_width, config.im_chan), dtype=np.uint8
      4 )
      5 sizes_test = []

NameError: name 'test_ids' is not defined

## === cell 12
preds_test = model.predict(X_test, verbose=1)

preds_test_upsampled = []
for i in trange(len(preds_test)):
    up = resize(
        np.squeeze(preds_test[i]),
        (sizes_test[i][0], sizes_test[i][1]),
        mode="constant",
        preserve_range=True,
    )
    preds_test_upsampled.append(up)




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2159659007.py in <cell line: 0>()
      1 # Predict on test set
----> 2 preds_test = model.predict(X_test, verbose=1)
      3 
      4 # Upsample predictions back to original size
      5 preds_test_upsampled = []

NameError: name 'X_test' is not defined

## === cell 13
def RLenc(img, order="F", format=True):
    """Run‑length encoding for binary mask."""
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
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
    for i, fn in tqdm(enumerate(test_ids), total=len(test_ids))
}




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1417003167.py in <cell line: 0>()
     24 pred_dict = {
     25     fn[:-4]: RLenc(np.round(preds_test_upsampled[i] > threshold_best))
---> 26     for i, fn in tqdm(enumerate(test_ids), total=len(test_ids))
     27 }
     28 

NameError: name 'test_ids' is not defined

## === cell 14
sub = pd.DataFrame.from_dict(pred_dict, orient="index")
sub.index.name = "id"
sub.columns = ["rle_mask"]
submission_path = "submission.csv"
sub.to_csv(submission_path, index=True)
print(f"Submission written to {submission_path}")

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/417958716.py in <cell line: 0>()
      1 # Create submission file
----> 2 sub = pd.DataFrame.from_dict(pred_dict, orient="index")
      3 sub.index.name = "id"
      4 sub.columns = ["rle_mask"]
      5 submission_path = "submission.csv"

NameError: name 'pred_dict' is not defined
