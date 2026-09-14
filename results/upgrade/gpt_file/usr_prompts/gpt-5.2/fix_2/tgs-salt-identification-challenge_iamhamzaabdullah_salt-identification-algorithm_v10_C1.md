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

3.7

# 3. Installed packages

geopandas==0.14.4
imageio==2.37.0
imageio-ffmpeg==0.6.0
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-image==0.25.2
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
tf_keras==2.18.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1

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

0.61732

# 6. Current score

0.0291

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.0291) has done: 'I fix the environment/runtime issues caused by mixing legacy `keras` imports with TensorFlow 2.18/Keras 3 (which triggers the protobuf `MessageFactory` error) by switching to `tf.keras` equivalents while keeping the same U-Net architecture, loss, and training loop. I also fix missing symbols (`load_img`, `Lambda`, `EarlyStopping`, etc.), deprecated NumPy dtypes (`np.bool`), and `tf.metrics` / session-based `mean_iou` code that no longer works in eager mode. Finally, I correct the input pipeline to produce the intended 2-channel tensors (original + cumsum), ensure file paths match the provided `/kaggle/input/...` layout, and generate a properly formatted `submission.csv` with `id,rle_mask` for all test images.'

# 9. Code solution

## === cell 0
import os
import sys
import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import imageio

from tqdm import tqdm
from itertools import chain

from skimage.transform import resize

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input,
    Lambda,
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    concatenate,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import img_to_array, load_img

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

BASE_INPUT = "/kaggle/input/tgs-salt-identification-challenge"
if not os.path.exists(BASE_INPUT):
    BASE_INPUT = "/kaggle/input"  # fallback for alternate mounts



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
import torch
from torch.utils import data


class TGSSaltDataset(data.Dataset):
    def __init__(self, root_path, file_list):
        self.root_path = root_path
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        if index not in range(0, len(self.file_list)):
            return self.__getitem__(np.random.randint(0, self.__len__()))
        file_id = self.file_list[index]
        image_folder = os.path.join(self.root_path, "images")
        image_path = os.path.join(image_folder, file_id + ".png")
        mask_folder = os.path.join(self.root_path, "masks")
        mask_path = os.path.join(mask_folder, file_id + ".png")
        image = np.array(imageio.imread(image_path), dtype=np.uint8)
        mask = np.array(imageio.imread(mask_path), dtype=np.uint8)
        return image, mask




## === cell 2
train_mask = pd.read_csv(os.path.join(BASE_INPUT, "train.csv"))
depth = pd.read_csv(os.path.join(BASE_INPUT, "depths.csv"))
train_path = os.path.join(BASE_INPUT, "train")

file_list = list(train_mask["id"].values)
dataset = TGSSaltDataset(train_path, file_list)




## === cell 3
def plot2x2Array(image, mask):
    f, axarr = plt.subplots(1, 2, figsize=(8, 4))
    axarr[0].imshow(image, cmap="gray")
    axarr[1].imshow(mask, cmap="gray")
    axarr[0].grid(False)
    axarr[1].grid(False)
    axarr[0].set_title("Image")
    axarr[1].set_title("Mask")
    plt.show()




## === cell 4
for i in range(2):
    image, mask = dataset[np.random.randint(0, len(dataset))]
    plot2x2Array(image, mask)



## === cell 5
plt.figure(figsize=(6, 4))
plt.hist(depth["z"], bins=50)
plt.title("Depth distribution")
plt.show()




## === cell 6
def rleToMask(rleString, height, width):
    rows, cols = height, width
    try:
        rleNumbers = [int(numstring) for numstring in rleString.split(" ")]
        rlePairs = np.array(rleNumbers).reshape(-1, 2)
        img = np.zeros(rows * cols, dtype=np.uint8)
        for index, length in rlePairs:
            index -= 1
            img[index : index + length] = 255
        img = img.reshape(cols, rows)
        img = img.T
    except Exception:
        img = np.zeros((cols, rows), dtype=np.uint8)
    return img


def salt_proportion(imgArray):
    try:
        unique, counts = np.unique(imgArray, return_counts=True)
        return counts[1] / 10201.0
    except Exception:
        return 0.0




## === cell 7
merged = train_mask.merge(depth, how="left", on="id")
merged.head()



## === cell 8
im_width = 128
im_height = 128
border = 5
im_chan = 2  # original + cumsum
n_features = 1
path_train = os.path.join(BASE_INPUT, "train") + "/"
path_test = os.path.join(BASE_INPUT, "test") + "/"



## === cell 9
ids = ["1f1cc6b3a4", "5b7c160d0d", "6c40978ddf", "7dfdf6eeb8", "7e5a6e5013"]
plt.figure(figsize=(18, 8))
for j, img_name in enumerate(ids):
    img = load_img(
        os.path.join(path_train, "images", img_name + ".png"), color_mode="grayscale"
    )
    img_mask = load_img(
        os.path.join(path_train, "masks", img_name + ".png"), color_mode="grayscale"
    )

    img = np.array(img)
    img_cumsum = (np.float32(img) - img.mean()).cumsum(axis=0)
    img_mask = np.array(img_mask)

    plt.subplot(3, len(ids), j + 1)
    plt.imshow(img, cmap="seismic")
    plt.axis("off")
    plt.title("img")

    plt.subplot(3, len(ids), len(ids) + j + 1)
    plt.imshow(img_cumsum, cmap="seismic")
    plt.axis("off")
    plt.title("cumsum")

    plt.subplot(3, len(ids), 2 * len(ids) + j + 1)
    plt.imshow(img_mask, cmap="gray")
    plt.axis("off")
    plt.title("mask")
plt.tight_layout()
plt.show()



## === cell 10
train_ids = next(os.walk(path_train + "images"))[2]
test_ids = next(os.walk(path_test + "images"))[2]



## === cell 11
X_train = np.zeros((len(train_ids), im_height, im_width, im_chan), dtype=np.float32)
Y_train = np.zeros((len(train_ids), im_height, im_width, 1), dtype=bool)

print("Getting and resizing train images and masks ... ")
sys.stdout.flush()

for n, id_ in tqdm(list(enumerate(train_ids)), total=len(train_ids)):
    img = load_img(path_train + "images/" + id_, color_mode="grayscale")
    x = img_to_array(img)[:, :, 0].astype(np.float32)  # (101,101)
    x_mean = x.mean()
    x0 = x  # original
    x1 = (x - x_mean).cumsum(axis=0)  # cumsum feature

    x0 = resize(
        x0,
        (im_height, im_width),
        mode="constant",
        preserve_range=True,
        anti_aliasing=False,
    )
    x1 = resize(
        x1,
        (im_height, im_width),
        mode="constant",
        preserve_range=True,
        anti_aliasing=False,
    )
    X_train[n, :, :, 0] = x0
    X_train[n, :, :, 1] = x1

    mask = load_img(path_train + "masks/" + id_, color_mode="grayscale")
    y = img_to_array(mask)[:, :, 0]
    y = resize(
        y,
        (im_height, im_width),
        mode="constant",
        preserve_range=True,
        anti_aliasing=False,
    )
    Y_train[n, :, :, 0] = y > 127

print("Done!")




## === cell 12
def mean_iou(y_true, y_pred):
    y_true = tf.cast(y_true > 0.5, tf.int32)
    prec = []
    for t in tf.range(0.5, 1.0, 0.05):
        y_pred_t = tf.cast(y_pred > t, tf.int32)

        intersection = tf.reduce_sum(
            tf.cast(y_true * y_pred_t, tf.float32), axis=[1, 2, 3]
        )
        union = tf.reduce_sum(
            tf.cast(tf.cast(y_true + y_pred_t, tf.bool), tf.float32), axis=[1, 2, 3]
        )

        iou = tf.math.divide_no_nan(intersection, union)
        prec.append(tf.reduce_mean(iou))
    return tf.reduce_mean(tf.stack(prec), axis=0)




## === cell 13
inputs = Input((im_height, im_width, im_chan))
s = Lambda(lambda x: x / 255.0)(inputs)

c1 = Conv2D(8, (3, 3), activation="relu", padding="same")(s)
c1 = Conv2D(8, (3, 3), activation="relu", padding="same")(c1)
p1 = MaxPooling2D((2, 2))(c1)

c2 = Conv2D(16, (3, 3), activation="relu", padding="same")(p1)
c2 = Conv2D(16, (3, 3), activation="relu", padding="same")(c2)
p2 = MaxPooling2D((2, 2))(c2)

c3 = Conv2D(32, (3, 3), activation="relu", padding="same")(p2)
c3 = Conv2D(32, (3, 3), activation="relu", padding="same")(c3)
p3 = MaxPooling2D((2, 2))(c3)

c4 = Conv2D(64, (3, 3), activation="relu", padding="same")(p3)
c4 = Conv2D(64, (3, 3), activation="relu", padding="same")(c4)
p4 = MaxPooling2D(pool_size=(2, 2))(c4)

c5 = Conv2D(128, (3, 3), activation="relu", padding="same")(p4)
c5 = Conv2D(128, (3, 3), activation="relu", padding="same")(c5)

u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(64, (3, 3), activation="relu", padding="same")(u6)
c6 = Conv2D(64, (3, 3), activation="relu", padding="same")(c6)

u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(32, (3, 3), activation="relu", padding="same")(u7)
c7 = Conv2D(32, (3, 3), activation="relu", padding="same")(c7)

u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(16, (3, 3), activation="relu", padding="same")(u8)
c8 = Conv2D(16, (3, 3), activation="relu", padding="same")(c8)

u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding="same")(c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(8, (3, 3), activation="relu", padding="same")(u9)
c9 = Conv2D(8, (3, 3), activation="relu", padding="same")(c9)

outputs = Conv2D(1, (1, 1), activation="sigmoid")(c9)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[mean_iou])
model.summary()



## === cell 14
callback = EarlyStopping(patience=12, verbose=1, restore_best_weights=False)
checkpointer = ModelCheckpoint("model-tgs-salt-1.keras", verbose=1, save_best_only=True)
results = model.fit(
    X_train,
    Y_train.astype(np.float32),
    validation_split=0.1,
    batch_size=32,
    epochs=20,
    callbacks=[callback, checkpointer],
    verbose=2,
)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
OperatorNotAllowedInGraphError            Traceback (most recent call last)
/tmp/ipykernel_11/3274835338.py in <cell line: 0>()
      2 callback = EarlyStopping(patience=12, verbose=1, restore_best_weights=False)
      3 checkpointer = ModelCheckpoint("model-tgs-salt-1.keras", verbose=1, save_best_only=True)
----> 4 results = model.fit(
      5     X_train,
      6     Y_train.astype(np.float32),

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_11/456041318.py in mean_iou(y_true, y_pred)
      4     y_true = tf.cast(y_true > 0.5, tf.int32)
      5     prec = []
----> 6     for t in tf.range(0.5, 1.0, 0.05):
      7         y_pred_t = tf.cast(y_pred > t, tf.int32)
      8 

OperatorNotAllowedInGraphError: Iterating over a symbolic `tf.Tensor` is not allowed. You can attempt the following resolutions to the problem: If you are running in Graph mode, use Eager execution mode or decorate this function with @tf.function. If you are using AutoGraph, you can try decorating this function with @tf.function. If that does not work, then you may be using an unsupported feature or your source code may not be visible to AutoGraph. See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/limitations.md#access-to-source-code for more information.

## === cell 15
X_test = np.zeros((len(test_ids), im_height, im_width, im_chan), dtype=np.float32)
sizes_test = []
print("Getting and resizing test images ... ")
sys.stdout.flush()

for n, id_ in tqdm(list(enumerate(test_ids)), total=len(test_ids)):
    img = load_img(path_test + "images/" + id_, color_mode="grayscale")
    x = img_to_array(img)[:, :, 0].astype(np.float32)
    sizes_test.append([x.shape[0], x.shape[1]])

    x0 = x
    x1 = (x - x.mean()).cumsum(axis=0)

    x0 = resize(
        x0,
        (im_height, im_width),
        mode="constant",
        preserve_range=True,
        anti_aliasing=False,
    )
    x1 = resize(
        x1,
        (im_height, im_width),
        mode="constant",
        preserve_range=True,
        anti_aliasing=False,
    )

    X_test[n, :, :, 0] = x0
    X_test[n, :, :, 1] = x1

print("Done!")



## === cell 16
model_path = "model-tgs-salt-1.keras"
if os.path.exists(model_path):
    model = load_model(model_path, custom_objects={"mean_iou": mean_iou}, compile=False)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[mean_iou])

preds_train = model.predict(X_train[: int(X_train.shape[0] * 0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0] * 0.9) :], verbose=1)
preds_test = model.predict(X_test, verbose=1)

preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)



## === cell 17
preds_test_upsampled = []
for i in range(len(preds_test)):
    preds_test_upsampled.append(
        resize(
            np.squeeze(preds_test[i]),
            (sizes_test[i][0], sizes_test[i][1]),
            mode="constant",
            preserve_range=True,
            anti_aliasing=False,
        )
    )

preds_test_upsampled[0].shape



## === cell 18
ix = random.randint(0, len(preds_train_t) - 1)
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(X_train[ix, :, :, 0], cmap="gray")
plt.title("Train image (ch0)")
plt.axis("off")
plt.subplot(1, 3, 2)
plt.imshow(np.squeeze(Y_train[ix]).astype(np.float32), cmap="gray")
plt.title("True mask")
plt.axis("off")
plt.subplot(1, 3, 3)
plt.imshow(np.squeeze(preds_train_t[ix]).astype(np.float32), cmap="gray")
plt.title("Pred mask")
plt.axis("off")
plt.tight_layout()
plt.show()




## === cell 19
def RLenc(img, order="F", format=True):
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
        return " ".join([f"{rr[0]} {rr[1]}" for rr in runs])
    return runs


pred_dict = {}
for i, fn in tqdm(list(enumerate(test_ids)), total=len(test_ids)):
    img = (preds_test_upsampled[i] > 0.5).astype(np.uint8)
    pred_dict[fn[:-4]] = RLenc(img)



## === cell 20
sub = pd.DataFrame({"id": list(pred_dict.keys()), "rle_mask": list(pred_dict.values())})

sample_path = os.path.join(BASE_INPUT, "sample_submission.csv")
sample = pd.read_csv(sample_path)
sub = sample[["id"]].merge(sub, on="id", how="left")
sub["rle_mask"] = sub["rle_mask"].fillna("")

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
