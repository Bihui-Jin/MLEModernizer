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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
protobuf==6.33.0
scipy==1.15.3
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

0.73762

# 6. Current score

0.5221

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5221) has done: 'I fix the broken array shaping logic by stacking images/masks correctly based on the actual number of training/test samples instead of hardcoded sizes (the root cause of the reshape/concatenate errors). I also fix the Keras 3 incompatibilities by switching `ImageDataGenerator` to `tf.keras.preprocessing.image.ImageDataGenerator` and replacing deprecated `tf.py_func` with `tf.numpy_function`, and avoid the protobuf-related crash by using `tf.keras` consistently. Finally, I ensure the test set is built from `sample_submission.csv` (guaranteed 1000 IDs) and generate a valid `submission.csv` with correct RLE encoding (1-indexed, column-major), so the notebook runs end-to-end and yields a valid submission.'
- What this solution (achieved 0.5221) has done: 'I fix the TensorFlow/protobuf runtime crash happening at import time (the `MessageFactory.GetPrototype` error) by pinning protobuf’s Python implementation and avoiding the mixed `keras`/`tf.keras` stack in this script. Then I make a minimal score-improving change that preserves the core model/training logic: add the already-defined `my_iou_metric` to `model.compile` (metric-aligned monitoring) and calibrate the prediction thresholding to use a standard `> 0.5` cutoff instead of the current `round(y_pred - 0.25)` heuristic. Finally, I keep submission formatting identical but ensure RLE encoding always receives a clean 2D binary mask.'
- What this solution (achieved 0.5221) has done: 'We fix the protobuf-related TensorFlow import crash by forcing the pure-Python protobuf implementation before TensorFlow loads and by using a consistent `tf.keras` stack only. Then we fix the `my_iou_metric` “unknown rank” error by wrapping it as a proper Keras `Metric` that always returns a scalar float32 and has a defined shape, keeping the same underlying IoU logic. Finally, we ensure the depth input matches the model’s expected `(50,50,1)` shape for both train/test and keep the existing `>0.5` thresholding and RLE submission generation intact so the notebook runs end-to-end and writes `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import numpy as np
import pandas as pd
from PIL import Image

INPUT_ROOT = "/kaggle/input/tgs-salt-identification-challenge"

print("Listing:", INPUT_ROOT)
print(os.listdir(INPUT_ROOT))

train_csv_path = os.path.join(INPUT_ROOT, "train.csv")
depths_csv_path = os.path.join(INPUT_ROOT, "depths.csv")
sample_sub_path = os.path.join(INPUT_ROOT, "sample_submission.csv")

train_img_dir = os.path.join(INPUT_ROOT, "train", "images")
train_mask_dir = os.path.join(INPUT_ROOT, "train", "masks")
test_img_dir = os.path.join(INPUT_ROOT, "test", "images")

df_train = pd.read_csv(train_csv_path)
print(df_train.head())
print("\nTrain files shape is", df_train.shape)

df_depths = pd.read_csv(depths_csv_path)
df_depths["z"] = df_depths["z"] / df_depths["z"].max()
print(df_depths.head())
print("\nDepths files shape is", df_depths.shape)

df_train = df_train.merge(df_depths, on="id", how="inner")
df_train = df_train.rename(columns={"z": "depth"})
print(df_train.head())
print("Merged train shape:", df_train.shape)



## === cell 1
df_train["images"] = [
    np.array(Image.open(os.path.join(train_img_dir, f"{idx}.png")))
    for idx in df_train["id"]
]
print("Sample Image Shape is", df_train["images"].iloc[0].shape)


def to_single_channel(img):
    if img.ndim == 3:
        return img[:, :, :1]
    return img[:, :, None]


df_train["images"] = df_train["images"].map(to_single_channel)
print("After optimization, Image Shape is", df_train["images"].iloc[0].shape)

df_train["images"] = df_train["images"].map(lambda x: x.astype(np.float32) / 255.0)
print("After normalization, pixel value is", df_train["images"].iloc[0][1, 0, 0])
print("No. of train images are", len(df_train))

df_train["masks"] = [
    np.array(Image.open(os.path.join(train_mask_dir, f"{idx}.png")))
    for idx in df_train["id"]
]
df_train["masks"] = df_train["masks"].map(to_single_channel)
print("\nSample Mask Shape is", df_train["masks"].iloc[0].shape)
print("No. of mask images are", len(df_train))
print("Before normalization, pixel value is", df_train["masks"].iloc[15][10, 0, 0])

print("train df columns are", df_train.columns.tolist())



## === cell 2
train_x = np.stack(df_train["images"].values, axis=0).astype(np.float32)
train_y = np.stack(df_train["masks"].values, axis=0).astype(np.float32)

train_y = train_y / (train_y.max() if train_y.max() > 0 else 1.0)
train_y = np.round(train_y).astype(np.float32)

print("Train Shape =", train_x.shape)
print("Mask Shape =", train_y.shape)
print("Sample pixel value (x)", train_x[0, 1, 0, 0])
print("Sample pixel value (y)", train_y[15, 10, 0, 0])



## === cell 3
import scipy.signal as sg
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import backend as K

Height = 101
Width = 101

import random as rn

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(7)
rn.seed(12345)
tf.random.set_seed(12345)

print("TensorFlow:", tf.__version__)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4
num_train_images = train_x.shape[0]

df_train["depth_image"] = df_train["depth"].map(
    lambda x: np.full(shape=(50, 50, 1), fill_value=x, dtype=np.float32)
)
depth_np = np.stack(df_train["depth_image"].values, axis=0).astype(np.float32)

depth_np_new = (
    df_train["depth"].values.astype(np.float32).reshape((num_train_images, 1, 1, 1))
)

print("depth (first 5):", depth_np_new[:5].reshape(-1))
print("depth_np_new shape:", depth_np_new.shape)
print("depth_np shape:", depth_np.shape)
print("df_train columns:", df_train.columns.tolist())



## === cell 5
Activation1 = "elu"
Activation2 = "tanh"


def convolution_block(
    x, filters, size, strides=(1, 1), padding="same", activation=True
):
    x = layers.Conv2D(filters, size, strides=strides, padding=padding)(x)
    if activation:
        x = layers.BatchNormalization()(x)
        x = layers.Activation(Activation1)(x)
    return x


def residual_block(blockInput, num_filters=16, activate=True):
    x = convolution_block(blockInput, num_filters, (3, 3))
    x = convolution_block(x, num_filters, (3, 3), activation=False)
    x = layers.Add()([x, blockInput])
    x = layers.BatchNormalization()(x)
    if activate:
        x = layers.Activation(Activation2)(x)
    return x




## === cell 6
img_input = layers.Input(shape=(Height, Width, 1), name="img_input")
depth_input = layers.Input(shape=(50, 50, 1), name="depth_input")
depth_input_new = layers.Input(shape=(1, 1, 1), name="depth_input_new")

print(img_input)
print(depth_input)

print("101->100")
x = layers.Conv2D(filters=16, kernel_size=(2, 2), padding="valid", activation="elu")(
    img_input
)
print(x)

x_depth = layers.Conv2D(
    filters=16, kernel_size=(2, 2), padding="valid", activation="elu"
)(depth_input)
print(x_depth)

print("100 -> 50")
x_16_pool = layers.MaxPooling2D(pool_size=(2, 2))(x)
x_16_pool = layers.BatchNormalization()(x_16_pool)
print(x)

x_16_pool_depth = layers.AveragePooling2D(pool_size=(2, 2))(x_depth)
x_16_pool_depth = layers.BatchNormalization()(x_16_pool_depth)
x_16_pool_depth = layers.Dropout(0.2)(x_16_pool_depth)
print(x_16_pool_depth)

print("50-> 48")
x = layers.Conv2D(32, 3, padding="valid", activation="elu")(x_16_pool)
x2 = layers.Conv2D(32, 3, padding="same", activation="tanh")(x)
x = layers.add([x, x2])
x = layers.BatchNormalization()(x)
print(x)

x_depth = layers.Conv2D(32, 3, padding="valid", activation="elu")(x_16_pool_depth)
x2_depth = layers.Conv2D(32, 3, padding="same", activation="tanh")(x_depth)
x_depth = layers.add([x_depth, x2_depth])
x_depth = layers.BatchNormalization()(x_depth)
print(x_depth)

print("48-> 24")
x_32_pool = layers.MaxPooling2D(2)(x)
print(x_32_pool)

x_32_pool_depth = layers.AveragePooling2D(2)(x_depth)
print(x_32_pool_depth)

x = layers.Dropout(0.25)(x_32_pool)
x_depth = layers.Dropout(0.2)(x_32_pool_depth)

x = layers.Conv2D(64, 3, padding="same", activation="elu")(x)
x3 = layers.Conv2D(64, 3, padding="same", activation="elu")(x)
x = layers.add([x, x3])
x = layers.BatchNormalization()(x)
print(x)

x_depth = layers.Conv2D(64, 3, padding="same", activation="elu")(x_32_pool_depth)
x2_depth = layers.Conv2D(64, 3, padding="same", activation="tanh")(x_depth)
x_depth = layers.add([x_depth, x2_depth])
x_depth = layers.BatchNormalization()(x_depth)
print(x_depth)

print("24->12")
x_64_pool = layers.MaxPooling2D(2)(x)
x_64_pool_depth = layers.AveragePooling2D(2)(x_depth)

print(x)
x_64_pool = layers.Dropout(0.25)(x_64_pool)
x_64_pool_depth = layers.Dropout(0.2)(x_64_pool_depth)

x = layers.Conv2D(128, 3, padding="same", activation="elu")(x_64_pool)
x2 = layers.Conv2D(128, 3, padding="same", activation="elu")(x)
x = layers.add([x, x2])

print("12->6")
x_128_pool = layers.MaxPooling2D(2)(x)
print(x_128_pool)

x_128_pool = layers.Dropout(0.25)(x_128_pool)

x = layers.Conv2D(192, 3, padding="same", activation="elu")(x_128_pool)
print(x)
x_192_pool = layers.MaxPooling2D(2)(x)
print(x_192_pool)

x_192_pool = layers.Dropout(0.25)(x_192_pool)

x = layers.Conv2D(192, 2, padding="valid", activation="elu")(x_192_pool)
print(x)
x = layers.MaxPooling2D(2)(x)
print(x)

x = layers.Dropout(0.25)(x)



## === cell 7
inverse = layers.Conv2DTranspose(
    filters=192, kernel_size=(3, 3), strides=(2, 2), padding="valid", activation="elu"
)(x)
print(inverse)

inverse = layers.Concatenate()([inverse, x_192_pool])

inverse = layers.Conv2D(
    filters=192, kernel_size=(2, 2), padding="same", activation="elu"
)(inverse)
print(inverse)
inverse = layers.BatchNormalization()(inverse)

inverse = layers.Dropout(0.25)(inverse)
print(inverse)

inverse = layers.Conv2DTranspose(
    filters=128, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation="elu"
)(inverse)
print("ok", inverse)

inverse = layers.Concatenate()([inverse, x_128_pool])
print("concatination", inverse)
inverse = layers.Conv2D(
    filters=128, kernel_size=(3, 3), padding="same", activation="elu"
)(inverse)
inverse = layers.BatchNormalization()(inverse)
inverse = layers.Dropout(0.25)(inverse)

inverse = layers.Conv2DTranspose(
    filters=64, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation="elu"
)(inverse)
print(inverse)

inverse = layers.Concatenate()([inverse, x_64_pool])

inverse = layers.Conv2D(
    filters=64, kernel_size=(3, 3), padding="same", activation="elu"
)(inverse)
inverse = layers.BatchNormalization()(inverse)
inverse = layers.Dropout(0.25)(inverse)

inverse = layers.Conv2DTranspose(
    filters=32, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation="elu"
)(inverse)
print(inverse)

inverse = layers.Conv2D(
    filters=32, kernel_size=(3, 3), padding="same", activation="elu"
)(inverse)
inverse = layers.Concatenate()([inverse, x_32_pool])
inverse = layers.BatchNormalization()(inverse)
inverse = layers.Dropout(0.25)(inverse)

inverse = layers.Conv2DTranspose(
    filters=16, kernel_size=(4, 4), strides=(2, 2), padding="valid", activation="elu"
)(inverse)
print(inverse)

inverse = layers.Concatenate()([inverse, x_16_pool])
inverse = layers.Concatenate()([inverse, depth_input])

inverse = layers.Conv2D(
    filters=16, kernel_size=(3, 3), padding="same", activation="elu"
)(inverse)
inverse = layers.BatchNormalization()(inverse)
inverse = layers.Dropout(0.25)(inverse)

print(inverse)

inverse2 = layers.Conv2DTranspose(
    filters=1, kernel_size=(3, 3), strides=(2, 2), padding="valid", activation="sigmoid"
)(inverse)
print(inverse2)

model = Model([img_input, depth_input], inverse2)
model.summary()



## === cell 8
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

Batch_size = 96
gen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)


def gen_flow_for_two_inputs(X1, X2, y):
    genX1 = gen.flow(X1, y, batch_size=Batch_size, seed=1234, shuffle=True)
    genX2 = gen.flow(X2, X2, batch_size=Batch_size, seed=1234, shuffle=True)
    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        yield [X1i[0], X2i[0]], X1i[1]


def get_iou(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    y_pred = np.round(y_pred)
    intersect = np.sum(np.logical_and(y_true == 1, y_pred == 1))
    union = np.sum(np.logical_or(y_true == 1, y_pred == 1))
    return (intersect + 1e-9) / (union + 1e-9)


class MeanIoUThreshold05(tf.keras.metrics.Metric):
    def __init__(self, name="my_iou_metric", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(
            name="total", initializer="zeros", dtype=tf.float32
        )
        self.count = self.add_weight(
            name="count", initializer="zeros", dtype=tf.float32
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)

        ious = tf.numpy_function(
            func=lambda yt, yp: np.array(
                [get_iou(yt[i], yp[i]) for i in range(yt.shape[0])], dtype=np.float32
            ),
            inp=[y_true, y_pred_bin],
            Tout=tf.float32,
        )
        ious.set_shape([None])  # known rank: vector

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            sample_weight = tf.reshape(sample_weight, [-1])
            ious = ious * sample_weight
            denom = tf.reduce_sum(sample_weight)
        else:
            denom = tf.cast(tf.size(ious), tf.float32)

        self.total.assign_add(tf.reduce_sum(ious))
        self.count.assign_add(denom)

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)

    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)


my_iou_metric = MeanIoUThreshold05()



## === cell 9
import tensorflow.keras.losses as losses


def dice_coeff(y_true, y_pred):
    smooth = 1.0
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )
    return score


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coeff(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    return losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)




## === cell 10
epochs = 60

model.compile(optimizer="adam", loss=bce_dice_loss, metrics=[dice_loss, my_iou_metric])

early_stopping = EarlyStopping(
    monitor="dice_loss", patience=3, mode="min", restore_best_weights=True
)

history = model.fit(
    [train_x, depth_np],
    train_y,
    batch_size=Batch_size,
    epochs=epochs,
    callbacks=[early_stopping],
    verbose=2,
)

model.save("model_7.h5")



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/97965847.py in <cell line: 0>()
      7 )
      8 
----> 9 history = model.fit(
     10     [train_x, depth_np],
     11     train_y,

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

InvalidArgumentError: Graph execution error:

Detected at node PyFunc defined at (most recent call last):
<stack traces unavailable>
Detected at node PyFunc defined at (most recent call last):
<stack traces unavailable>
Detected unsupported operations when trying to compile graph __inference_one_step_on_data_11802[] on XLA_GPU_JIT: PyFunc (No registered 'PyFunc' OpKernel for XLA_GPU_JIT devices compatible with node {{node PyFunc}}){{node PyFunc}}
The op is created at: 
File "<frozen runpy>", line 198, in _run_module_as_main
File "<frozen runpy>", line 88, in _run_code
File "/usr/local/lib/python3.11/dist-packages/colab_kernel_launcher.py", line 37, in <module>
File "/usr/local/lib/python3.11/dist-packages/traitlets/config/application.py", line 992, in launch_instance
File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelapp.py", line 712, in start
File "/usr/local/lib/python3.11/dist-packages/tornado/platform/asyncio.py", line 211, in start
File "/usr/lib/python3.11/asyncio/base_events.py", line 608, in run_forever
File "/usr/lib/python3.11/asyncio/base_events.py", line 1936, in _run_once
File "/usr/lib/python3.11/asyncio/events.py", line 84, in _run
File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 510, in dispatch_queue
File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 499, in process_one
File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 406, in dispatch_shell
File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 730, in execute_request
File "/usr/local/lib/python3.11/dist-packages/ipykernel/ipkernel.py", line 383, in do_execute
File "/usr/local/lib/python3.11/dist-packages/ipykernel/zmqshell.py", line 528, in run_cell
File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 2975, in run_cell
File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3030, in _run_cell
File "/usr/local/lib/python3.11/dist-packages/IPython/core/async_helpers.py", line 78, in _pseudo_sync_runner
File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3257, in run_cell_async
File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3473, in run_ast_nodes
File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3553, in run_code
File "/tmp/ipykernel_11/97965847.py", line 9, in <cell line: 0>
File "/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py", line 117, in error_handler
File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 371, in fit
File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 219, in function
File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 132, in multi_step_on_iterator
File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 113, in one_step_on_data
File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 84, in train_step
File "/usr/local/lib/python3.11/dist-packages/keras/src/trainers/trainer.py", line 490, in compute_metrics
File "/usr/local/lib/python3.11/dist-packages/keras/src/trainers/compile_utils.py", line 334, in update_state
File "/usr/local/lib/python3.11/dist-packages/keras/src/trainers/compile_utils.py", line 21, in update_state
File "/tmp/ipykernel_11/3777252264.py", line 43, in update_state
	tf2xla conversion failed while converting __inference_one_step_on_data_11802[]. Run with TF_DUMP_GRAPH_PREFIX=/path/to/dump/dir and --vmodule=xla_compiler=2 to obtain a dump of the compiled functions.
	 [[StatefulPartitionedCall]] [Op:__inference_multi_step_on_iterator_12212]

## === cell 11
df_sample = pd.read_csv(sample_sub_path)
df_test = df_sample[["id"]].merge(df_depths, on="id", how="left")  # adds normalized z
print("Length of df_test is", len(df_test))
print(df_test.head())



## === cell 12
df_test["images"] = [
    np.array(Image.open(os.path.join(test_img_dir, f"{idx}.png")))
    for idx in df_test["id"]
]



## === cell 13
print("Sample Image Shape is", df_test["images"].iloc[10].shape)



## === cell 14
df_test["images"] = df_test["images"].map(to_single_channel)
print("After optimization, Image Shape is", df_test["images"].iloc[0].shape)
print("No. of test images are", len(df_test))
print("Before normalization, pixel value is", df_test["images"].iloc[10][2, 0, 0])

df_test["images"] = df_test["images"].map(lambda x: x.astype(np.float32) / 255.0)
print("After normalization, pixel value is", df_test["images"].iloc[10][2, 0, 0])

test_x = np.stack(df_test["images"].values, axis=0).astype(np.float32)
print("test_x max/min:", test_x.max(), test_x.min())
print("train_x max/min:", train_x.max(), train_x.min())
print("Test Shape =", test_x.shape)
print("Sample Pixel Value", test_x[10, 2, 0, 0])



## === cell 15
df_test = df_test.rename(columns={"z": "depth"})  # align naming
df_test["depth_image"] = df_test["depth"].map(
    lambda x: np.full(shape=(50, 50, 1), fill_value=x, dtype=np.float32)
)
depth_test_np = np.stack(df_test["depth_image"].values, axis=0).astype(np.float32)

print(df_test.columns.tolist())
print("depth_test_np shape:", depth_test_np.shape)




## === cell 16
def rle_encode(mask):
    """
    mask: 2D numpy array of 0/1
    returns: run length encoding string
    """
    mask = (mask > 0).astype(np.uint8)
    pixels = mask.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] = runs[1::2] - runs[::2]
    if len(runs) == 0:
        return ""
    return " ".join(str(int(x)) for x in runs)


def post_process(x):
    x = x.astype(np.uint8)
    out = []
    for i in range(x.shape[0]):
        out.append(rle_encode(x[i, :, :, 0]))
    return out




## === cell 17
print("Predicting now.....")
y_pred = model.predict([test_x, depth_test_np], batch_size=Batch_size, verbose=1)
print("Prediction Complete. y_pred shape:", y_pred.shape)



## === cell 18
y_pred_bin = (y_pred > 0.5).astype(np.uint8)

print("Y Prediction shape =", y_pred_bin.shape)
print("min/max:", y_pred_bin.min(), y_pred_bin.max())

new_y_pred = post_process(y_pred_bin)
print("Y Prediction length =", len(new_y_pred))



## === cell 19
output = pd.DataFrame({"id": df_test["id"].values, "rle_mask": new_y_pred})
output.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", output.shape)
print(output.head())



## === cell 20
import time

print("Done at:", time.time())
print(np.random.rand(10))
