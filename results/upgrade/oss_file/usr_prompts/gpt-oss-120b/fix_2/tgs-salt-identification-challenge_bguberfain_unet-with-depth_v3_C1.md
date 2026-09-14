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
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
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

0.68137

# 6. Current score

0.1091

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.1091) has done: 'I fixed the import errors, replaced the old Keras imports with `tensorflow.keras`, corrected the missing `load_img` and tqdm utilities, added safe handling for missing depth values, and ensured the model, training, prediction, and submission steps all run sequentially and produce a valid `submission.csv` file.'

# 9. Code solution

## === cell 0
import os
import sys
import random
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import cv2
from tqdm import tqdm

from skimage.transform import resize

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    concatenate,
    RepeatVector,
    Reshape,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import backend as K




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
im_width = 128
im_height = 128
border = 5
im_chan = 2  # original + cumulative sum channel
n_features = 1  # depth feature

path_train = "../input/train/"
path_test = "../input/test/"




## === cell 2
df_depths = pd.read_csv("../input/depths.csv", index_col="id")
df_depths.head()




## === cell 3
ids = ["1f1cc6b3a4", "5b7c160d0d", "6c40978ddf", "7dfdf6eeb8", "7e5a6e5013"]
plt.figure(figsize=(30, 15))
for j, img_name in enumerate(ids):
    q = j + 1
    img = load_img(
        os.path.join(path_train, "images", img_name + ".png"), color_mode="grayscale"
    )
    img_mask = load_img(
        os.path.join(path_train, "masks", img_name + ".png"), color_mode="grayscale"
    )
    img = np.array(img)
    img_cumsum = (np.float32(img) - img.mean()).cumsum(axis=0)
    img_mask = np.array(img_mask)

    plt.subplot(1, 3 * (1 + len(ids)), q * 3 - 2)
    plt.imshow(img, cmap="seismic")
    plt.subplot(1, 3 * (1 + len(ids)), q * 3 - 1)
    plt.imshow(img_cumsum, cmap="seismic")
    plt.subplot(1, 3 * (1 + len(ids)), q * 3)
    plt.imshow(img_mask, cmap="gray")
plt.show()




## === cell 4
train_ids = next(os.walk(os.path.join(path_train, "images")))[2]
test_ids = next(os.walk(os.path.join(path_test, "images")))[2]




## === cell 5
X = np.zeros((len(train_ids), im_height, im_width, im_chan), dtype=np.float32)
y = np.zeros((len(train_ids), im_height, im_width, 1), dtype=np.float32)
X_feat = np.zeros((len(train_ids), n_features), dtype=np.float32)

print("Getting and resizing train images and masks ...")
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    depth_val = (
        df_depths.loc[id_.replace(".png", ""), "z"]
        if id_.replace(".png", "") in df_depths.index
        else 0.0
    )
    X_feat[n, 0] = depth_val

    img = load_img(os.path.join(path_train, "images", id_), color_mode="grayscale")
    x_img = img_to_array(img)  # (h, w, 1)
    x_img = resize(
        x_img, (im_height, im_width, 1), mode="constant", preserve_range=True
    )

    x_center = x_img[border:-border, border:-border, 0]
    x_center_mean = x_center.mean()
    x_csum = (np.float32(x_img) - x_center_mean).cumsum(axis=0)
    x_csum -= x_csum[border:-border, border:-border, 0].mean()
    x_csum /= max(1e-3, x_csum[border:-border, border:-border, 0].std())

    mask = load_img(os.path.join(path_train, "masks", id_), color_mode="grayscale")
    mask = img_to_array(mask)
    mask = resize(mask, (im_height, im_width, 1), mode="constant", preserve_range=True)

    X[n, ..., 0] = x_img.squeeze() / 255.0
    X[n, ..., 1] = x_csum.squeeze()
    y[n] = mask / 255.0
print("Done!")




## === cell 6
X_train, X_valid, X_feat_train, X_feat_valid, y_train, y_valid = train_test_split(
    X, X_feat, y, test_size=0.15, random_state=42
)




## === cell 7
x_feat_mean = X_feat_train.mean(axis=0, keepdims=True)
x_feat_std = X_feat_train.std(axis=0, keepdims=True) + 1e-8

X_feat_train = (X_feat_train - x_feat_mean) / x_feat_std
X_feat_valid = (X_feat_valid - x_feat_mean) / x_feat_std




## === cell 8
ix = random.randint(0, len(X_train) - 1)
has_mask = y_train[ix].max() > 0

fig, ax = plt.subplots(1, 3, figsize=(20, 10))
ax[0].imshow(X_train[ix, ..., 0], cmap="seismic")
if has_mask:
    ax[0].contour(y_train[ix].squeeze(), colors="k", levels=[0.5])
ax[0].set_title("Seismic")

ax[1].imshow(X_train[ix, ..., 1], cmap="seismic")
if has_mask:
    ax[1].contour(y_train[ix].squeeze(), colors="k", levels=[0.5])
ax[1].set_title("Seismic cumsum")

ax[2].imshow(y_train[ix].squeeze(), cmap="gray")
ax[2].set_title("Salt")
plt.show()




## === cell 9
def mean_iou_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.int32)
    metric = tf.keras.metrics.MeanIoU(num_classes=2)
    metric.update_state(y_true, y_pred)
    return metric.result()




## === cell 10
input_img = Input((im_height, im_width, im_chan), name="img")
input_feat = Input((n_features,), name="feat")

c1 = Conv2D(8, (3, 3), activation="relu", padding="same")(input_img)
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
p4 = MaxPooling2D((2, 2))(c4)

f_repeat = RepeatVector(8 * 8)(input_feat)
f_reshape = Reshape((8, 8, n_features))(f_repeat)

p4_feat = concatenate([p4, f_reshape], axis=-1)

c5 = Conv2D(128, (3, 3), activation="relu", padding="same")(p4_feat)
c5 = Conv2D(128, (3, 3), activation="relu", padding="same")(c5)

u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c5)
u6 = concatenate([u6, c4], axis=-1)
c6 = Conv2D(64, (3, 3), activation="relu", padding="same")(u6)
c6 = Conv2D(64, (3, 3), activation="relu", padding="same")(c6)

u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c6)
u7 = concatenate([u7, c3], axis=-1)
c7 = Conv2D(32, (3, 3), activation="relu", padding="same")(u7)
c7 = Conv2D(32, (3, 3), activation="relu", padding="same")(c7)

u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(c7)
u8 = concatenate([u8, c2], axis=-1)
c8 = Conv2D(16, (3, 3), activation="relu", padding="same")(u8)
c8 = Conv2D(16, (3, 3), activation="relu", padding="same")(c8)

u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding="same")(c8)
u9 = concatenate([u9, c1], axis=-1)
c9 = Conv2D(8, (3, 3), activation="relu", padding="same")(u9)
c9 = Conv2D(8, (3, 3), activation="relu", padding="same")(c9)

output = Conv2D(1, (1, 1), activation="sigmoid")(c9)

model = Model(inputs=[input_img, input_feat], outputs=output)
model.compile(optimizer="adam", loss="binary_crossentropy")
model.summary()




## === cell 11
callbacks = [
    EarlyStopping(patience=5, verbose=1, restore_best_weights=True),
    ReduceLROnPlateau(patience=3, verbose=1),
    ModelCheckpoint(
        "model-tgs-salt-1.h5", verbose=1, save_best_only=True, save_weights_only=True
    ),
]

results = model.fit(
    {"img": X_train, "feat": X_feat_train},
    y_train,
    batch_size=16,
    epochs=50,
    callbacks=callbacks,
    validation_data=({"img": X_valid, "feat": X_feat_valid}, y_valid),
)




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/1851721414.py in <cell line: 0>()
      2     EarlyStopping(patience=5, verbose=1, restore_best_weights=True),
      3     ReduceLROnPlateau(patience=3, verbose=1),
----> 4     ModelCheckpoint(
      5         "model-tgs-salt-1.h5", verbose=1, save_best_only=True, save_weights_only=True
      6     ),

/usr/local/lib/python3.11/dist-packages/keras/src/callbacks/model_checkpoint.py in __init__(self, filepath, monitor, verbose, save_best_only, save_weights_only, mode, save_freq, initial_value_threshold)
    182         if save_weights_only:
    183             if not self.filepath.endswith(".weights.h5"):
--> 184                 raise ValueError(
    185                     "When using `save_weights_only=True` in `ModelCheckpoint`"
    186                     ", the filepath provided must end in `.weights.h5` "

ValueError: When using `save_weights_only=True` in `ModelCheckpoint`, the filepath provided must end in `.weights.h5` (Keras weights format). Received: filepath=model-tgs-salt-1.h5

## === cell 12
X_test = np.zeros((len(test_ids), im_height, im_width, im_chan), dtype=np.float32)
X_feat_test = np.zeros((len(test_ids), n_features), dtype=np.float32)
sizes_test = []

print("Getting and resizing test images ...")
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    depth_val = (
        df_depths.loc[id_.replace(".png", ""), "z"]
        if id_.replace(".png", "") in df_depths.index
        else 0.0
    )
    X_feat_test[n, 0] = depth_val

    img = load_img(os.path.join(path_test, "images", id_), color_mode="grayscale")
    x = img_to_array(img)
    sizes_test.append([x.shape[0], x.shape[1]])

    x = resize(x, (im_height, im_width, 1), mode="constant", preserve_range=True)

    x_center = x[border:-border, border:-border, 0]
    x_center_mean = x_center.mean()
    x_csum = (np.float32(x) - x_center_mean).cumsum(axis=0)
    x_csum -= x_csum[border:-border, border:-border, 0].mean()
    x_csum /= max(1e-3, x_csum[border:-border, border:-border, 0].std())

    X_test[n, ..., 0] = x.squeeze() / 255.0
    X_test[n, ..., 1] = x_csum.squeeze()
print("Done!")




## === cell 13
X_feat_test = (X_feat_test - x_feat_mean) / x_feat_std




## === cell 14
val_score = model.evaluate({"img": X_valid, "feat": X_feat_valid}, y_valid, verbose=1)
print("Validation loss:", val_score)




## === cell 15
preds_train = model.predict({"img": X_train, "feat": X_feat_train}, verbose=0)
preds_val = model.predict({"img": X_valid, "feat": X_feat_valid}, verbose=0)
preds_test = model.predict({"img": X_test, "feat": X_feat_test}, verbose=0)




## === cell 16
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)




## === cell 17
preds_test_upsampled = []
for i in tqdm(range(len(preds_test_t))):
    up = resize(
        np.squeeze(preds_test_t[i]),
        (sizes_test[i][0], sizes_test[i][1]),
        mode="constant",
        preserve_range=True,
    )
    preds_test_upsampled.append(up.astype(np.uint8))
print("Upsampling done.")




## === cell 18
def RLenc(img, order="F", format=True):
    """Run‑length encoding.  img must be binary (0 / 1)."""
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []
    r = 0
    pos = 1  # 1‑based indexing
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

    if not format:
        return runs
    return " ".join(f"{p} {l}" for p, l in runs)


pred_dict = {}
for i, fn in enumerate(test_ids):
    mask = preds_test_upsampled[i]
    rle = RLenc(mask)
    pred_dict[fn[:-4]] = rle




## === cell 19
sub = pd.DataFrame.from_dict(pred_dict, orient="index")
sub.index.name = "id"
sub.columns = ["rle_mask"]
sub.to_csv("submission.csv")
print("Submission file written to submission.csv")
