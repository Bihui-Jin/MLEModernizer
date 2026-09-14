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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

# 5. Code solution

## === cell 0
import os
import sys
import subprocess

try:
    import google.protobuf  # noqa: F401

    pb_ver = None
    try:
        from importlib.metadata import version

        pb_ver = version("protobuf")
    except Exception:
        pb_ver = None

    if pb_ver is not None:
        major = int(pb_ver.split(".")[0])
        if major >= 6:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
            )
            os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
except Exception:
    pass

import random as rn
import numpy as np
import pandas as pd
from PIL import Image

import tensorflow as tf
from keras import backend as K

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(1234)
rn.seed(1234)
tf.random.set_seed(1234)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

print("TensorFlow:", tf.__version__)
print(
    "Keras:", tf.keras.__version__ if hasattr(tf.keras, "__version__") else "tf.keras"
)
print("Listing /kaggle/input:", os.listdir("/kaggle/input")[:10])

DATA_ROOT = "/kaggle/input/tgs-salt-identification-challenge"
if not os.path.exists(DATA_ROOT):
    DATA_ROOT = "/kaggle/input"

train_csv_path = os.path.join(DATA_ROOT, "train.csv")
depths_csv_path = os.path.join(DATA_ROOT, "depths.csv")

df_train = pd.read_csv(train_csv_path)
df_depths = pd.read_csv(depths_csv_path)

depth_z_max = float(np.max(df_depths["z"].values))
df_depths["z_norm"] = df_depths["z"].astype(np.float32) / (depth_z_max + 1e-9)

df_train = df_train.merge(
    df_depths[["id", "z_norm"]].rename(columns={"z_norm": "depth"}),
    on="id",
    how="inner",
)

print("\nTrain files shape:", df_train.shape)
print("Depths files shape:", df_depths.shape)
print("Train df columns:", df_train.columns.tolist())
print("depth_z_max:", depth_z_max)



## === cell 1
train_img_dir = os.path.join(DATA_ROOT, "train", "images")
train_mask_dir = os.path.join(DATA_ROOT, "train", "masks")

ids_train = df_train["id"].values
num_train_images = len(ids_train)


def _read_png_101(path):
    b = tf.io.read_file(path)
    img = tf.io.decode_png(b, channels=1)  # always 1 channel
    img = tf.ensure_shape(img, [101, 101, 1])
    return img


def _load_train_pair(img_path, mask_path):
    img = tf.cast(_read_png_101(img_path), tf.float32) / 255.0
    mask = tf.cast(_read_png_101(mask_path), tf.float32)
    return img, mask


img_paths = tf.constant(
    [os.path.join(train_img_dir, f"{idx}.png") for idx in ids_train]
)
mask_paths = tf.constant(
    [os.path.join(train_mask_dir, f"{idx}.png") for idx in ids_train]
)

ds = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
ds = ds.map(_load_train_pair, num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.batch(256).prefetch(tf.data.AUTOTUNE)

train_x = np.empty((num_train_images, 101, 101, 1), dtype=np.float32)
train_y = np.empty((num_train_images, 101, 101, 1), dtype=np.float32)

offset = 0
for bx, by in ds:
    n = bx.shape[0]
    train_x[offset : offset + n] = bx.numpy()
    train_y[offset : offset + n] = by.numpy()
    offset += n

print("Sample Image Shape is", train_x[0].shape)
print("After optimization, Image Shape is", train_x[0].shape)
print("No. of train images are", num_train_images)
print("After normalization, pixel value is", train_x[0][1, 0, 0])

print("\nSample Mask Shape is", train_y[0].shape)
print("No. of mask images are", num_train_images)
print(
    "Before normalization, pixel value is",
    train_y[15][10, 0, 0] if num_train_images > 15 else train_y[0][10, 0, 0],
)

print("train df columns are", df_train.columns.tolist())



## === cell 2
train_y = train_y / (train_y.max() + 1e-9)
train_y = np.round(train_y)

print("Train Shape =", train_x.shape)
print("Sample Pixel Value", train_x[0, 1, 0, 0])
print("Mask Shape =", train_y.shape)
print(
    "Sample Pixel Value",
    train_y[15, 10, 0, 0] if num_train_images > 15 else train_y[0, 10, 0, 0],
)



## === cell 3
print(train_x.shape)
print(train_y.shape)



## === cell 4
import scipy.signal as sg  # kept (used nowhere but preserve original imports)
from keras.models import Model
from keras import layers, regularizers

Height = 101
Width = 101



## === cell 5
depth_vals = df_train["depth"].astype(np.float32).values
depth_np = depth_vals[:, None, None, None] * np.ones((1, 50, 50, 1), dtype=np.float32)

depth_train_max = float(np.max(depth_vals))

depth_np_new = depth_vals.astype(np.float32).reshape((num_train_images, 1, 1, 1))
print("depth", depth_np_new[0:5, 0, 0, 0])
print("depth shape", depth_np_new.shape)
print("df_train columns:", df_train.columns.tolist())
print("depth_np shape", depth_np.shape)
print("depth_train_max", depth_train_max)



## === cell 6
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




## === cell 7
reg = regularizers.l2(0.01)
img_input = layers.Input(shape=(Height, Width, 1), name="img_input")
depth_input = layers.Input(shape=(50, 50, 1), name="depth_input")
depth_input_new = layers.Input(shape=(1, 1, 1), name="depth_input_new")  # kept unused

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
x2 = layers.Conv2D(32, 3, padding="same", activation="tanh", kernel_regularizer=reg)(x)
x = layers.add([x, x2])
x = layers.BatchNormalization()(x)
print(x)

x_depth = layers.Conv2D(32, 3, padding="valid", activation="elu")(x_16_pool_depth)
x2_depth = layers.Conv2D(
    32, 3, padding="same", activation="tanh", kernel_regularizer=reg
)(x_depth)
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
x3 = layers.Conv2D(64, 3, padding="same", activation="elu", kernel_regularizer=reg)(x)
x = layers.add([x, x3])
x = layers.BatchNormalization()(x)
print(x)

x_depth = layers.Conv2D(64, 3, padding="same", activation="elu")(x_32_pool_depth)
x2_depth = layers.Conv2D(
    64, 3, padding="same", activation="tanh", kernel_regularizer=reg
)(x_depth)
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
x2 = layers.Conv2D(128, 3, padding="same", activation="elu", kernel_regularizer=reg)(x)
x = layers.add([x, x2])

print("12->6")
x_128_pool = layers.MaxPooling2D(2)(x)
print(x_128_pool)

x_128_pool = layers.Dropout(0.25)(x_128_pool)

x = layers.Conv2D(192, 3, padding="same", activation="elu", kernel_regularizer=reg)(
    x_128_pool
)
print(x)
x_192_pool = layers.MaxPooling2D(2)(x)
print(x_192_pool)

x_192_pool = layers.Dropout(0.25)(x_192_pool)

x = layers.Conv2D(
    192, 2, padding="valid", activation="elu", kernel_regularizer=regularizers.l2(0.01)
)(x_192_pool)
print(x)
x = layers.MaxPooling2D(2)(x)
print(x)

x = layers.Dropout(0.25)(x)



## === cell 8
inverse = layers.Conv2DTranspose(
    filters=192,
    kernel_size=(3, 3),
    strides=(2, 2),
    padding="valid",
    activation="elu",
    kernel_regularizer=reg,
)(x)
print(inverse)

inverse = layers.Concatenate()([inverse, x_192_pool])

inverse = layers.Conv2D(
    filters=192,
    kernel_size=(2, 2),
    padding="same",
    activation="elu",
    kernel_regularizer=reg,
)(inverse)
print(inverse)
inverse = layers.BatchNormalization()(inverse)

inverse = layers.Dropout(0.25)(inverse)

print(inverse)

inverse = layers.Conv2DTranspose(
    filters=128,
    kernel_size=(2, 2),
    strides=(2, 2),
    padding="valid",
    kernel_regularizer=reg,
    activation="elu",
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
    filters=64,
    kernel_size=(3, 3),
    padding="same",
    activation="elu",
    kernel_regularizer=reg,
)(inverse)
inverse = layers.BatchNormalization()(inverse)

inverse = layers.Dropout(0.25)(inverse)

inverse = layers.Conv2DTranspose(
    filters=32, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation="elu"
)(inverse)
print(inverse)

inverse = layers.Conv2D(
    filters=32,
    kernel_size=(3, 3),
    padding="same",
    activation="elu",
    kernel_regularizer=reg,
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



## === cell 9
import keras.losses as losses


def dice_coeff(y_true, y_pred):
    smooth = 0.00001
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
    return 0.2 * losses.binary_crossentropy(y_true, y_pred) + 0.8 * dice_loss(
        y_true, y_pred
    )




## === cell 10
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

train_x1, val_x, depth_np1, val_depth, train_y1, val_y = train_test_split(
    train_x, depth_np, train_y, test_size=0.25, random_state=1234
)

Batch_size = 96
epochs = 60

model.compile(optimizer="adam", loss=bce_dice_loss, metrics=[dice_coeff, "acc"])

early_stopping = EarlyStopping(
    monitor="val_dice_coeff", patience=3, mode="max", restore_best_weights=True
)

history = model.fit(
    [train_x1, depth_np1],
    train_y1,
    validation_data=([val_x, val_depth], val_y),
    batch_size=Batch_size,
    epochs=epochs,
    callbacks=[early_stopping],
    verbose=2,
)

model.save("model_7.h5")



## === cell 11
print("Predicting the validation set....")
val_x_pred = model.predict([val_x, val_depth], batch_size=Batch_size, verbose=0)
print("Validation set prediction completed.", val_x_pred.shape)




## === cell 12
def iou_numpy(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0  # both empty => perfect
    return inter / union


def mean_ap_iou_objects_single(gt_mask, pr_mask, thresholds=np.arange(0.5, 1.0, 0.05)):
    gt = gt_mask.astype(bool)
    pr = pr_mask.astype(bool)

    try:
        import scipy.ndimage as ndi
    except Exception:
        gt_any = gt.sum() > 0
        pr_any = pr.sum() > 0
        if (not gt_any) and (not pr_any):
            return 1.0
        if gt_any != pr_any:
            return 0.0
        iou = iou_numpy(gt, pr)
        return float(np.mean([1.0 if iou > t else 0.0 for t in thresholds]))

    gt_lab, gt_n = ndi.label(gt)
    pr_lab, pr_n = ndi.label(pr)

    if gt_n == 0 and pr_n == 0:
        return 1.0
    if gt_n == 0 and pr_n > 0:
        return 0.0
    if gt_n > 0 and pr_n == 0:
        return 0.0

    ious = np.zeros((gt_n, pr_n), dtype=np.float32)
    for gi in range(1, gt_n + 1):
        g = gt_lab == gi
        for pi in range(1, pr_n + 1):
            p = pr_lab == pi
            ious[gi - 1, pi - 1] = iou_numpy(g, p)

    aps = []
    for t in thresholds:
        matches = ious > t
        tp = 0
        matched_gt = np.zeros(gt_n, dtype=bool)
        matched_pr = np.zeros(pr_n, dtype=bool)
        if matches.any():
            pairs = np.argwhere(matches)
            pair_ious = ious[pairs[:, 0], pairs[:, 1]]
            order = np.argsort(-pair_ious)
            for k in order:
                gidx, pidx = int(pairs[k, 0]), int(pairs[k, 1])
                if (not matched_gt[gidx]) and (not matched_pr[pidx]):
                    matched_gt[gidx] = True
                    matched_pr[pidx] = True
                    tp += 1
        fn = int((~matched_gt).sum())
        fp = int((~matched_pr).sum())
        denom = tp + fp + fn
        aps.append(0.0 if denom == 0 else tp / denom)
    return float(np.mean(aps))


def _cleanup_mask_single(mask_bool, min_size=15, hole_size=15):
    m = mask_bool.astype(bool)

    try:
        import scipy.ndimage as ndi

        labeled, n = ndi.label(m)
        if n > 0:
            sizes = ndi.sum(m, labeled, index=np.arange(1, n + 1))
            keep = np.zeros(n + 1, dtype=bool)
            keep[1:] = sizes >= min_size
            m = keep[labeled]
        inv = ~m
        labeled_h, nh = ndi.label(inv)
        if nh > 0:
            hsizes = ndi.sum(inv, labeled_h, index=np.arange(1, nh + 1))
            fill = np.zeros(nh + 1, dtype=bool)
            fill[1:] = hsizes <= hole_size
            holes_to_fill = fill[labeled_h] & inv
            m = m | holes_to_fill
    except Exception:
        pass
    return m


def _morph_smooth_single(mask_bool, close_iter=1, open_iter=1):
    try:
        import scipy.ndimage as ndi

        m = mask_bool.astype(bool)
        struct = np.ones((3, 3), dtype=bool)
        if close_iter and close_iter > 0:
            m = ndi.binary_closing(m, structure=struct, iterations=int(close_iter))
        if open_iter and open_iter > 0:
            m = ndi.binary_opening(m, structure=struct, iterations=int(open_iter))
        return m
    except Exception:
        return mask_bool.astype(bool)


def _keep_largest_cc(mask_bool):
    m = mask_bool.astype(bool)
    if not m.any():
        return m
    try:
        import scipy.ndimage as ndi

        lab, n = ndi.label(m)
        if n <= 1:
            return m
        sizes = ndi.sum(m, lab, index=np.arange(1, n + 1))
        keep_idx = int(np.argmax(sizes)) + 1
        return lab == keep_idx
    except Exception:
        return m


def apply_threshold_and_empty_gate(
    y_prob,
    thr,
    empty_frac_thr=None,
    empty_mean_thr=None,
    do_cleanup=False,
    min_size=15,
    hole_size=15,
    do_morph=False,
    close_iter=1,
    open_iter=1,
    keep_largest=False,
):
    y_prob2 = y_prob[..., 0]
    y_pred_bin = y_prob2 > thr

    if empty_mean_thr is not None:
        means = y_prob2.reshape((y_prob2.shape[0], -1)).mean(axis=1)
        y_pred_bin[means < empty_mean_thr] = False

    if empty_frac_thr is not None:
        frac = y_pred_bin.reshape((y_pred_bin.shape[0], -1)).mean(axis=1)
        y_pred_bin[frac < empty_frac_thr] = False

    if do_cleanup or do_morph or keep_largest:
        for i in range(y_pred_bin.shape[0]):
            if y_pred_bin[i].any():
                if do_morph:
                    y_pred_bin[i] = _morph_smooth_single(
                        y_pred_bin[i], close_iter=close_iter, open_iter=open_iter
                    )
                if do_cleanup and y_pred_bin[i].any():
                    y_pred_bin[i] = _cleanup_mask_single(
                        y_pred_bin[i], min_size=min_size, hole_size=hole_size
                    )
                if keep_largest and y_pred_bin[i].any():
                    y_pred_bin[i] = _keep_largest_cc(y_pred_bin[i])

    return y_pred_bin


_THRESHOLDS = np.arange(0.5, 1.0, 0.05).astype(np.float32)


def mean_ap_iou_objects_batch(y_true_bin, y_pred_bin, thresholds=_THRESHOLDS):
    scores = []
    for i in range(y_true_bin.shape[0]):
        scores.append(
            mean_ap_iou_objects_single(y_true_bin[i], y_pred_bin[i], thresholds)
        )
    return float(np.mean(scores))


thr_grid_coarse = list(np.round(np.arange(0.10, 0.61, 0.02), 3))
empty_frac_grid = list(np.round(np.arange(0.000, 0.021, 0.002), 3))
empty_mean_grid = list(np.round(np.arange(0.000, 0.071, 0.01), 3))

do_cleanup = True
min_size = 15
hole_size = 15

morph_grid = [False, True]
close_iter = 1
open_iter = 1

largest_cc_grid = [False, True]

best_score = -1.0
best_thr = 0.25
best_empty_frac = None
best_empty_mean = None
best_do_morph = False
best_keep_largest = False

val_y_true_bin = val_y[..., 0] > 0.5
val_prob2 = val_x_pred[..., 0].astype(np.float32, copy=False)
val_means = (
    val_prob2.reshape((val_prob2.shape[0], -1))
    .mean(axis=1)
    .astype(np.float32, copy=False)
)


def _tune_over_thr_grid(thr_grid, do_morph_flag, keep_largest_flag, best_pack):
    (
        best_score_local,
        best_thr_local,
        best_ef_local,
        best_em_local,
        best_morph_local,
        best_keep_largest_local,
    ) = best_pack

    cached_base_masks = {}  # thr -> (mask_bool_after_post, frac_after_post)
    for thr in thr_grid:
        base = apply_threshold_and_empty_gate(
            val_x_pred,
            thr=float(thr),
            empty_frac_thr=None,
            empty_mean_thr=None,
            do_cleanup=do_cleanup,
            min_size=min_size,
            hole_size=hole_size,
            do_morph=do_morph_flag,
            close_iter=close_iter,
            open_iter=open_iter,
            keep_largest=keep_largest_flag,
        )
        frac = base.reshape((base.shape[0], -1)).mean(axis=1).astype(np.float32)
        cached_base_masks[float(thr)] = (base, frac)

    for thr in thr_grid:
        base_mask, base_frac = cached_base_masks[float(thr)]
        for ef in empty_frac_grid:
            ef_use = None if ef == 0.0 else float(ef)
            gate_ef = None if ef_use is None else (base_frac < ef_use)

            for em in empty_mean_grid:
                em_use = None if em == 0.0 else float(em)
                gate_em = None if em_use is None else (val_means < em_use)

                if gate_ef is None and gate_em is None:
                    pred = base_mask
                else:
                    pred = base_mask.copy()
                    if gate_em is not None:
                        pred[gate_em] = False
                    if gate_ef is not None:
                        pred[gate_ef] = False

                s = float(mean_ap_iou_objects_batch(val_y_true_bin, pred))
                if s > best_score_local:
                    best_score_local = s
                    best_thr_local = float(thr)
                    best_ef_local = ef_use
                    best_em_local = em_use
                    best_morph_local = bool(do_morph_flag)
                    best_keep_largest_local = bool(keep_largest_flag)

        print(
            f"Finished morph={do_morph_flag} keep_largest={keep_largest_flag} thr={thr:.3f} "
            f"(best_score={best_score_local:.5f}, best_thr={best_thr_local}, best_empty_frac={best_ef_local}, best_empty_mean={best_em_local})"
        )

    return (
        best_score_local,
        best_thr_local,
        best_ef_local,
        best_em_local,
        best_morph_local,
        best_keep_largest_local,
    )


best_pack = (
    best_score,
    best_thr,
    best_empty_frac,
    best_empty_mean,
    best_do_morph,
    best_keep_largest,
)
for dm in morph_grid:
    for kl in largest_cc_grid:
        best_pack = _tune_over_thr_grid(thr_grid_coarse, dm, kl, best_pack)

(
    best_score,
    best_thr,
    best_empty_frac,
    best_empty_mean,
    best_do_morph,
    best_keep_largest,
) = best_pack

thr_grid_fine = list(
    np.round(
        np.clip(np.arange(best_thr - 0.03, best_thr + 0.031, 0.005), 0.05, 0.95), 3
    )
)
thr_grid_fine = sorted(set([float(t) for t in thr_grid_fine]))

best_pack = (
    best_score,
    best_thr,
    best_empty_frac,
    best_empty_mean,
    best_do_morph,
    best_keep_largest,
)
best_pack = _tune_over_thr_grid(
    thr_grid_fine, best_do_morph, best_keep_largest, best_pack
)
(
    best_score,
    best_thr,
    best_empty_frac,
    best_empty_mean,
    best_do_morph,
    best_keep_largest,
) = best_pack

print("Best on validation (object-based proxy):")
print("  best_score      =", best_score)
print("  best_thr        =", best_thr)
print("  best_empty_frac =", best_empty_frac)
print("  best_empty_mean =", best_empty_mean)
print(
    "  do_morph        =",
    best_do_morph,
    "close_iter=",
    close_iter,
    "open_iter=",
    open_iter,
)
print("  keep_largest    =", best_keep_largest)
print("  cleanup         =", do_cleanup, "min_size=", min_size, "hole_size=", hole_size)



## === cell 13
sample_sub_path = os.path.join(DATA_ROOT, "sample_submission.csv")
df_test = pd.read_csv(sample_sub_path)[["id"]].merge(
    df_depths[["id", "z_norm"]], on="id", how="left"
)
df_test = df_test.rename(columns={"z_norm": "depth"})

print("Length of df_test is", len(df_test))
print(df_test.head())



## === cell 14
test_img_dir = os.path.join(DATA_ROOT, "test", "images")

ids_test = df_test["id"].values
num_test = len(ids_test)
test_x = np.empty((num_test, 101, 101, 1), dtype=np.float32)

test_paths = tf.constant([os.path.join(test_img_dir, f"{idx}.png") for idx in ids_test])


def _load_test_img(path):
    img = tf.cast(_read_png_101(path), tf.float32) / 255.0
    return img


ds_test = tf.data.Dataset.from_tensor_slices(test_paths)
ds_test = ds_test.map(_load_test_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(256).prefetch(tf.data.AUTOTUNE)

offset = 0
for bx in ds_test:
    n = bx.shape[0]
    test_x[offset : offset + n] = bx.numpy()
    offset += n

print("Sample Image Shape is", test_x[10].shape if num_test > 10 else test_x[0].shape)



## === cell 15
print("After optimization, Image Shape is", test_x[0].shape)
print("No. of test images are", num_test)
print(
    "Sample Pixel Value", test_x[10, 2, 0, 0] if num_test > 10 else test_x[0, 2, 0, 0]
)
print("Test Shape =", test_x.shape)



## === cell 16
depth_test_vals = df_test["depth"].astype(np.float32).values
depth_test_np = depth_test_vals[:, None, None, None] * np.ones(
    (1, 50, 50, 1), dtype=np.float32
)

print("df_test columns:", df_test.columns.tolist())
print("depth_test_np shape", depth_test_np.shape)




## === cell 17
def post_process(x):
    num_files = x.shape[0]
    main_list = []
    pic_size = Height * Width
    for i in range(num_files):
        pic = x[i, :, :, 0].flatten(order="F")
        s_parts = []
        length = 0
        start = 0
        for j in range(pic_size):
            if pic[j] == 1:
                length += 1
                if length == 1:
                    start = j + 1  # 1-indexed
            if (pic[j] == 0 and length > 0) or (j == pic_size - 1 and length > 0):
                s_parts.append(str(start))
                s_parts.append(str(length))
                length = 0
        main_list.append(" ".join(s_parts) if len(s_parts) > 0 else "")
    return main_list


def rle_encode_batch(masks_bool):
    N, H, W = masks_bool.shape
    flat = (
        masks_bool.transpose(0, 2, 1).reshape(N, -1).astype(np.uint8)
    )  # Fortran-like flatten
    out = []
    for i in range(N):
        pixels = flat[i]
        padded = np.concatenate(([0], pixels, [0]))
        changes = np.flatnonzero(padded[1:] != padded[:-1])
        runs = changes + 1
        runs[1::2] -= runs[::2]
        if runs.size == 0:
            out.append("")
        else:
            out.append(" ".join(map(str, runs.tolist())))
    return out




## === cell 18
print("Predicting now.....")
y_pred = model.predict([test_x, depth_test_np], batch_size=Batch_size, verbose=2)
print("Prediction Complete", y_pred.shape)



## === cell 19
y_pred_bin = apply_threshold_and_empty_gate(
    y_pred,
    thr=best_thr,
    empty_frac_thr=best_empty_frac,
    empty_mean_thr=best_empty_mean,
    do_cleanup=do_cleanup,
    min_size=min_size,
    hole_size=hole_size,
    do_morph=best_do_morph,
    close_iter=close_iter,
    open_iter=open_iter,
    keep_largest=best_keep_largest,  # apply tuned single-object constraint at test time
)

new_y_pred = rle_encode_batch(y_pred_bin.astype(bool))

submission = pd.DataFrame({"id": df_test["id"].values, "rle_mask": new_y_pred})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print("Wrote", submission_path, "with shape", submission.shape)
print(submission.head())
