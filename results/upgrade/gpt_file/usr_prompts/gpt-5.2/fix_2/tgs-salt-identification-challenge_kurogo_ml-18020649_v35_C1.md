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

0.8134314149718772

# 6. Current score

0.1561

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.1561) has done: 'I fix the environment-breaking TensorFlow/Keras import issue by using `tf.keras` consistently (this avoids the protobuf `MessageFactory.GetPrototype` crash). I also correct the dataset paths (your `ROOT_DATA_DIR` points to a non-existent folder) and make the code robust by auto-detecting the correct Kaggle input directory that contains `train/images`, `train/masks`, and `test/images`. Then I ensure the train/valid split and training run end-to-end by removing the invalid `steps_per_epoch` usage for in-memory numpy arrays (which can cause shape/step mismatches) while keeping the same training objective and model. Finally, I generate a properly formatted `submission.csv` with `id` and `rle_mask` for all test images.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import tensorflow as tf

np.random.seed(1234)
tf.random.set_seed(1234)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    concatenate,
    Dropout,
    BatchNormalization,
    Activation,
    Input,
    Add,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import load_img, img_to_array



## === cell 2
CANDIDATE_ROOTS = [
    "/kaggle/input/tgs-salt-identification-challenge/",
    "/kaggle/input/tgs-salt-identification-challenge/tgs-salt-identification-challenge/",
    "/kaggle/data/tgs-salt-identification-challenge/",
    "/kaggle/data/",
    "/kaggle/input/",
]


def _find_dataset_root(candidates):
    for root in candidates:
        train_img = os.path.join(root, "train", "images")
        train_msk = os.path.join(root, "train", "masks")
        test_img = os.path.join(root, "test", "images")
        if (
            os.path.isdir(train_img)
            and os.path.isdir(train_msk)
            and os.path.isdir(test_img)
        ):
            return root
    for base in ["/kaggle/input", "/kaggle/data"]:
        if not os.path.isdir(base):
            continue
        for name in os.listdir(base):
            root = os.path.join(base, name)
            train_img = os.path.join(root, "train", "images")
            train_msk = os.path.join(root, "train", "masks")
            test_img = os.path.join(root, "test", "images")
            if (
                os.path.isdir(train_img)
                and os.path.isdir(train_msk)
                and os.path.isdir(test_img)
            ):
                return root
    raise FileNotFoundError(
        "Could not locate dataset root containing train/images, train/masks, test/images"
    )


ROOT_DATA_DIR = _find_dataset_root(CANDIDATE_ROOTS)
TRAIN_MASK_DIR = os.path.join(ROOT_DATA_DIR, "train", "masks") + os.sep
TRAIN_IMAGE_DIR = os.path.join(ROOT_DATA_DIR, "train", "images") + os.sep
TEST_IMAGE_DIR = os.path.join(ROOT_DATA_DIR, "test", "images") + os.sep

ROOT_DATA_DIR, TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, TEST_IMAGE_DIR




## === cell 3
def _find_file(possible_paths):
    for p in possible_paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"File not found in any of: {possible_paths}")


train_csv = _find_file(
    [
        os.path.join(ROOT_DATA_DIR, "train.csv"),
        "/kaggle/input/tgs-salt-identification-challenge/train.csv",
        "/kaggle/data/train.csv",
        "/kaggle/input/train.csv",
    ]
)
sample_csv = _find_file(
    [
        os.path.join(ROOT_DATA_DIR, "sample_submission.csv"),
        "/kaggle/input/tgs-salt-identification-challenge/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
    ]
)
depths_csv = _find_file(
    [
        os.path.join(ROOT_DATA_DIR, "depths.csv"),
        "/kaggle/input/tgs-salt-identification-challenge/depths.csv",
        "/kaggle/data/depths.csv",
        "/kaggle/input/depths.csv",
    ]
)

train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(sample_csv)
depths_df = pd.read_csv(depths_csv)

train_df.shape, test_df.shape, depths_df.shape




## === cell 4
def get_coverage(rle_mask):
    if pd.isna(rle_mask):
        return 0
    arr = rle_mask.split()
    coverage = sum(int(x) for x in arr[1::2]) / (101**2)
    return np.round(coverage, 1)


train_df["coverage"] = train_df["rle_mask"].map(get_coverage)
train_df.head()



## === cell 5
train_imgs = [
    load_img(TRAIN_IMAGE_DIR + image_name + ".png", color_mode="grayscale")
    for image_name in tqdm(train_df["id"], desc="Loading train images")
]
train_masks = [
    load_img(TRAIN_MASK_DIR + image_name + ".png", color_mode="grayscale")
    for image_name in tqdm(train_df["id"], desc="Loading train masks")
]

train_imgs = np.array(
    [img_to_array(img, dtype=np.uint8) / 255.0 for img in train_imgs], dtype=np.float32
)
train_masks = np.array(
    [img_to_array(mask, dtype=np.uint8) // 255 for mask in train_masks], dtype=np.uint8
)

train_imgs.shape, train_masks.shape, train_imgs.dtype, train_masks.dtype



## === cell 6
x_train, x_valid, y_train, y_valid = train_test_split(
    train_imgs,
    train_masks,
    test_size=0.2,
    stratify=train_df.coverage,
    random_state=1234,
)

x_train = np.append(x_train, np.array([np.fliplr(x) for x in x_train]), axis=0)
y_train = np.append(y_train, np.array([np.fliplr(x) for x in y_train]), axis=0)

x_train.shape, x_valid.shape, y_train.shape, y_valid.shape




## === cell 7
def iou_vector(trues, preds):
    SMOOTH = 1e-10
    batch_size = trues.shape[0]
    metric = []
    for idx in range(batch_size):
        true, pred = trues[idx], preds[idx]
        intersection = np.logical_and(true, pred)
        union = np.logical_or(true, pred)
        iou = (np.sum(intersection) + SMOOTH) / (np.sum(union) + SMOOTH)
        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))
    return np.mean(metric)


def my_iou_metric(label, pred):
    return tf.py_function(iou_vector, [label, pred > 0.5], tf.float64)




## === cell 8
def convolution_block(
    x, filters, size, strides=(1, 1), padding="same", activation=True
):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    if activation:
        x = Activation("relu")(x)
    return x


def residual_block(blockInput, num_filters=16):
    x = convolution_block(blockInput, num_filters, (3, 3))
    x = convolution_block(x, num_filters, (3, 3), activation=False)
    x = Add()([x, blockInput])
    x = Activation("relu")(x)
    return x


def build_model(input_shape):
    start_feature = 32
    DropoutRatio = 0.5
    skip_connections = []

    inputs = Input(input_shape)
    x = inputs

    for i in range(4):
        x = convolution_block(x, start_feature * (2**i), 3)
        x = residual_block(x, start_feature * (2**i))
        x = residual_block(x, start_feature * (2**i))
        skip_connections.append(x)
        x = MaxPooling2D((2, 2))(x)

    x = convolution_block(x, start_feature * (2**4), 3)
    x = residual_block(x, start_feature * (2**4))
    x = residual_block(x, start_feature * (2**4))

    for i in reversed(range(4)):
        if x.shape[2] * 2 != skip_connections[i].shape[2]:
            x = Conv2DTranspose(
                start_feature * (2**i), (3, 3), strides=(2, 2), padding="valid"
            )(x)
        else:
            x = Conv2DTranspose(
                start_feature * (2**i), (3, 3), strides=(2, 2), padding="same"
            )(x)

        x = concatenate([x, skip_connections[i]])
        x = Dropout(DropoutRatio)(x)
        x = convolution_block(x, start_feature * (2**i), 3)
        x = residual_block(x, start_feature * (2**i))
        x = residual_block(x, start_feature * (2**i))

    output = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(x)
    model = Model(inputs, output)
    return model


model = build_model((101, 101, 1))
model.compile(loss="binary_crossentropy", optimizer="Adam", metrics=[my_iou_metric])
model.summary()



## === cell 9
early_stopping = EarlyStopping(
    monitor="val_my_iou_metric", mode="max", patience=15, verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_my_iou_metric",
    mode="max",
    factor=0.5,
    patience=5,
    min_lr=0.0001,
    verbose=1,
)

epochs = 100

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_valid, y_valid),
    epochs=epochs,
    callbacks=[early_stopping, reduce_lr],
    verbose=1,
)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2601405195.py in <cell line: 0>()
     14 epochs = 100
     15 
---> 16 history = model.fit(
     17     x_train,
     18     y_train,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/metrics/reduction_metrics.py in reduce_to_samplewise_values(values, sample_weight, reduce_fn, dtype)
     39             )
     40 
---> 41     values_ndim = len(values.shape)
     42     if values_ndim > 1:
     43         values = reduce_fn(values, axis=list(range(1, values_ndim)))

ValueError: Cannot take the length of shape with unknown rank.

## === cell 10
fig, (ax_loss, ax_score) = plt.subplots(1, 2, figsize=(15, 5))
ax_loss.set_ylim(0, 1)
ax_loss.plot(history.epoch, history.history["loss"], label="Train loss")
ax_loss.plot(history.epoch, history.history["val_loss"], label="Validation loss")
ax_score.plot(history.epoch, history.history["my_iou_metric"], label="Train score")
ax_score.plot(
    history.epoch, history.history["val_my_iou_metric"], label="Validation score"
)
ax_loss.legend()
ax_score.legend()
plt.show()



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3706628140.py in <cell line: 0>()
      2 fig, (ax_loss, ax_score) = plt.subplots(1, 2, figsize=(15, 5))
      3 ax_loss.set_ylim(0, 1)
----> 4 ax_loss.plot(history.epoch, history.history["loss"], label="Train loss")
      5 ax_loss.plot(history.epoch, history.history["val_loss"], label="Validation loss")
      6 ax_score.plot(history.epoch, history.history["my_iou_metric"], label="Train score")

NameError: name 'history' is not defined

## === cell 11
test_imgs = [
    load_img(TEST_IMAGE_DIR + image_name + ".png", color_mode="grayscale")
    for image_name in tqdm(test_df["id"], desc="Loading test images")
]
test_imgs = np.array(
    [img_to_array(img, dtype=np.uint8) / 255.0 for img in test_imgs], dtype=np.float32
)

pred_test = model.predict(test_imgs, verbose=1)

test_flip = np.array([np.fliplr(x) for x in test_imgs])
pred_test_flip = model.predict(test_flip, verbose=1)
pred_test += np.array([np.fliplr(x) for x in pred_test_flip])
pred_test /= 2.0

pred_test.shape




## === cell 12
def mask2rle(mask):
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask = (mask > 0.5).astype(np.uint8)

    pixels = mask.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


rle_masks = [mask2rle(x) for x in pred_test]

test_df["rle_mask"] = rle_masks
test_df.to_csv("submission.csv", index=False)

test_df.head()



## === cell 13
assert os.path.exists("submission.csv")
sub = pd.read_csv("submission.csv")
sub.columns.tolist(), sub.shape, sub.head()
