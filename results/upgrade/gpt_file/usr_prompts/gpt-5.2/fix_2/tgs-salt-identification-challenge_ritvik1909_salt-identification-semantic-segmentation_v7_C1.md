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

0.71573

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
get_ipython().run_line_magic(
    "bash",
    '-lc "unzip -q -n ../input/tgs-salt-identification-challenge/train.zip -d train/"',
)
get_ipython().run_line_magic(
    "bash",
    '-lc "unzip -q -n ../input/tgs-salt-identification-challenge/test.zip -d test/"',
)

if not os.path.isdir(os.path.join(config.path_train, "images")):
    nested = os.path.join(config.path_train, "train", "images")
    if os.path.isdir(nested):
        get_ipython().run_line_magic(
            "bash", '-lc "mv -n train/train/* train/ 2>/dev/null || true"'
        )

if not os.path.isdir(os.path.join(config.path_test, "images")):
    nested = os.path.join(config.path_test, "test", "images")
    if os.path.isdir(nested):
        get_ipython().run_line_magic(
            "bash", '-lc "mv -n test/test/* test/ 2>/dev/null || true"'
        )

assert os.path.isdir(
    os.path.join(config.path_train, "images")
), "train/images not found after unzip"
assert os.path.isdir(
    os.path.join(config.path_train, "masks")
), "train/masks not found after unzip"
assert os.path.isdir(
    os.path.join(config.path_test, "images")
), "test/images not found after unzip"



## === cell 3
random.seed(19)
ids = random.choices(os.listdir("train/images"), k=6)
fig = plt.figure(figsize=(20, 6))
for j, img_name in enumerate(ids):
    q = j + 1

    img = load_img("train/images/" + img_name)
    img_mask = load_img("train/masks/" + img_name)

    plt.subplot(2, 6, q * 2 - 1)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.subplot(2, 6, q * 2)
    plt.imshow(img_mask, cmap="gray")
    plt.axis("off")
fig.suptitle("Sample Images", fontsize=24)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/959905413.py in <cell line: 0>()
      1 random.seed(19)
----> 2 ids = random.choices(os.listdir("train/images"), k=6)
      3 fig = plt.figure(figsize=(20, 6))
      4 for j, img_name in enumerate(ids):
      5     q = j + 1

FileNotFoundError: [Errno 2] No such file or directory: 'train/images'

## === cell 4
train_ids = next(os.walk(config.path_train + "images"))[2]
test_ids = next(os.walk(config.path_test + "images"))[2]

print("Train images:", len(train_ids))
print("Test images:", len(test_ids))




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
StopIteration                             Traceback (most recent call last)
/tmp/ipykernel_11/1408676892.py in <cell line: 0>()
----> 1 train_ids = next(os.walk(config.path_train + "images"))[2]
      2 test_ids = next(os.walk(config.path_test + "images"))[2]
      3 
      4 print("Train images:", len(train_ids))
      5 print("Test images:", len(test_ids))

StopIteration: 

## === cell 5
def load_grayscale_as_1ch(path):
    img = img_to_array(load_img(path))  # typically float32
    if img.ndim == 2:
        img = img[..., None]
    if img.shape[-1] == 3:
        img = img[:, :, 1:2]
    elif img.shape[-1] == 4:
        img = img[:, :, 1:2]
    elif img.shape[-1] == 1:
        pass
    else:
        img = img[:, :, :1]
    return img




## === cell 6
X_train = np.zeros(
    (len(train_ids), config.im_height, config.im_width, config.im_chan), dtype=np.uint8
)
Y_train = np.zeros((len(train_ids), config.im_height, config.im_width, 1), dtype=bool)

print("Getting and resizing train images and masks ... ")
sys.stdout.flush()

for n, id_ in tqdm(list(enumerate(train_ids)), total=len(train_ids)):
    img = load_grayscale_as_1ch(os.path.join(config.path_train, "images", id_))
    x = resize(
        img,
        (config.im_height, config.im_width, 1),
        mode="constant",
        preserve_range=True,
    )
    X_train[n] = x.astype(np.uint8)

    mask = load_grayscale_as_1ch(os.path.join(config.path_train, "masks", id_))
    mask = resize(
        mask,
        (config.im_height, config.im_width, 1),
        mode="constant",
        preserve_range=True,
    )
    Y_train[n] = mask > 127

print("Done!")



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2555032753.py in <cell line: 0>()
      1 X_train = np.zeros(
----> 2     (len(train_ids), config.im_height, config.im_width, config.im_chan), dtype=np.uint8
      3 )
      4 Y_train = np.zeros((len(train_ids), config.im_height, config.im_width, 1), dtype=bool)
      5 

NameError: name 'train_ids' is not defined

## === cell 7
X_train = np.append(
    X_train, [np.fliplr(x) for x in tqdm(X_train, desc="Flip X")], axis=0
)
Y_train = np.append(
    Y_train, [np.fliplr(x) for x in tqdm(Y_train, desc="Flip Y")], axis=0
)

print("Augmented X_train:", X_train.shape, "Y_train:", Y_train.shape)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1859684515.py in <cell line: 0>()
      1 # Simple augmentation consistent with original code (horizontal flip)
      2 X_train = np.append(
----> 3     X_train, [np.fliplr(x) for x in tqdm(X_train, desc="Flip X")], axis=0
      4 )
      5 Y_train = np.append(

NameError: name 'X_train' is not defined

## === cell 8
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
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
model.summary()



## === cell 9
try:
    utils.plot_model(model, expand_nested=True, show_shapes=True)
except Exception as e:
    print("plot_model skipped due to:", repr(e))



## === cell 10
es = callbacks.EarlyStopping(patience=30, verbose=1, restore_best_weights=True)
rlp = callbacks.ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-12, verbose=1)

results = model.fit(
    X_train,
    Y_train,
    validation_split=0.1,
    batch_size=8,
    epochs=300,
    callbacks=[es, rlp],
    verbose=2,
)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2420774403.py in <cell line: 0>()
      3 
      4 results = model.fit(
----> 5     X_train,
      6     Y_train,
      7     validation_split=0.1,

NameError: name 'X_train' is not defined

## === cell 11
sns.set_style("darkgrid")
fig, ax = plt.subplots(2, 1, figsize=(20, 8))
history = pd.DataFrame(results.history)
history[["loss", "val_loss"]].plot(ax=ax[0])
history[["acc", "val_acc"]].plot(ax=ax[1])
fig.suptitle("Learning Curve", fontsize=24)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3568918997.py in <cell line: 0>()
      1 sns.set_style("darkgrid")
      2 fig, ax = plt.subplots(2, 1, figsize=(20, 8))
----> 3 history = pd.DataFrame(results.history)
      4 history[["loss", "val_loss"]].plot(ax=ax[0])
      5 history[["acc", "val_acc"]].plot(ax=ax[1])

NameError: name 'results' is not defined

## === cell 12
X_test = np.zeros(
    (len(test_ids), config.im_height, config.im_width, config.im_chan), dtype=np.uint8
)
sizes_test = []
print("Getting and resizing test images ... ")
sys.stdout.flush()

for n, id_ in tqdm(list(enumerate(test_ids)), total=len(test_ids)):
    img = load_grayscale_as_1ch(os.path.join(config.path_test, "images", id_))
    sizes_test.append([img.shape[0], img.shape[1]])
    x = resize(
        img,
        (config.im_height, config.im_width, 1),
        mode="constant",
        preserve_range=True,
    )
    X_test[n] = x.astype(np.uint8)

print("Done!")



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/175601786.py in <cell line: 0>()
      1 X_test = np.zeros(
----> 2     (len(test_ids), config.im_height, config.im_width, config.im_chan), dtype=np.uint8
      3 )
      4 sizes_test = []
      5 print("Getting and resizing test images ... ")

NameError: name 'test_ids' is not defined

## === cell 13
preds_test = model.predict(X_test, verbose=1)
preds_test_upsampled = []
for i in trange(len(preds_test)):
    preds_test_upsampled.append(
        resize(
            np.squeeze(preds_test[i]),
            (sizes_test[i][0], sizes_test[i][1]),
            mode="constant",
            preserve_range=True,
        )
    )



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/704596821.py in <cell line: 0>()
----> 1 preds_test = model.predict(X_test, verbose=1)
      2 preds_test_upsampled = []
      3 for i in trange(len(preds_test)):
      4     preds_test_upsampled.append(
      5         resize(

NameError: name 'X_test' is not defined

## === cell 14
pass




## === cell 15
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
    fn[:-4]: RLenc(np.round(preds_test_upsampled[i]).astype(np.uint8))
    for i, fn in tqdm(list(enumerate(test_ids)), total=len(test_ids))
}



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1795458703.py in <cell line: 0>()
     38 pred_dict = {
     39     fn[:-4]: RLenc(np.round(preds_test_upsampled[i]).astype(np.uint8))
---> 40     for i, fn in tqdm(list(enumerate(test_ids)), total=len(test_ids))
     41 }
     42 

NameError: name 'test_ids' is not defined

## === cell 16
sub = pd.DataFrame.from_dict(pred_dict, orient="index")
sub.index.names = ["id"]
sub.columns = ["rle_mask"]
sub = sub.reset_index()

sample_path = "../input/tgs-salt-identification-challenge/sample_submission.csv"
if os.path.exists(sample_path):
    sample = pd.read_csv(sample_path)
    sub = sample[["id"]].merge(sub, on="id", how="left")
    sub["rle_mask"] = sub["rle_mask"].fillna("")

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/239978213.py in <cell line: 0>()
----> 1 sub = pd.DataFrame.from_dict(pred_dict, orient="index")
      2 sub.index.names = ["id"]
      3 sub.columns = ["rle_mask"]
      4 sub = sub.reset_index()
      5 

NameError: name 'pred_dict' is not defined
