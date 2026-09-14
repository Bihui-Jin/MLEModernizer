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

0.74891

# 6. Current score

0.5221

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5221) has done: 'I update the code to be compatible with TensorFlow 2.18 / Keras 3 by removing the deprecated TF1 session/ConfigProto usage that currently crashes imports, while keeping the same model and training flow. I also fix hard-coded image counts/shapes (4000/18000) to use the actual dataset sizes (3000 train / 1000 test), and correct the join with depths so it doesn’t rely on column name collisions. Finally, I replace the broken `keras.preprocessing.image.ImageDataGenerator` import with `tf.keras.preprocessing.image.ImageDataGenerator` (same functionality), ensure the test set comes from `sample_submission.csv` (correct IDs), and write a valid `submission.csv` with the required `id,rle_mask` columns.'
- What this solution (achieved 0.5221) has done: 'I first fix the TensorFlow/Keras import crash caused by an incompatible protobuf version by forcing the pure-Python protobuf implementation before importing TensorFlow (this is a runtime stability fix). Next, I fix the Keras 3 `EarlyStopping` error by explicitly setting `mode='min'` for the monitored `dice_loss` metric, which restores the intended training flow without changing the model. Finally, I keep the existing training/inference/post-processing logic intact so the pipeline runs end-to-end and reliably writes a valid `submission.csv` with `id,rle_mask`. These changes are expected to improve the score versus the current broken/short-circuited training, moving it toward the target.'
- What this solution (achieved 0.5221) has done: 'I first fix the crash happening before training by making TensorFlow/Keras imports robust to the protobuf `MessageFactory.GetPrototype` incompatibility: we force the pure-Python protobuf implementation and, if needed, monkey-patch `GetPrototype` to `GetMessageClass` before importing TensorFlow. Next, I correct the IoU helper bug (it currently computes “accuracy”, not IoU) even though it’s not used for training; this is a logic fix and score-neutral. Finally, to move the score upward toward your target with minimal change, I adjust the final binarization step from the current hard-coded `round(pred - 0.25)` to a standard `pred > 0.5` threshold (same model, same training), which typically improves Kaggle mAP-IoU for this competition and should reduce the gap versus 0.74891.'
- What this solution (achieved 0.5221) has done: 'I fix the protobuf monkey-patch condition that currently triggers the AttributeError before any training starts (it should only patch when `GetPrototype` exists on the class but is missing on the instance), which restores end-to-end execution in this Kaggle TensorFlow/Keras environment. I also make `gen_flow_for_two_inputs()` correctly augment both inputs by using `X2` for the second generator (the current code mistakenly uses `X1`, so depth is not aligned/augmented), and then actually use this generator during `model.fit` so the intended augmentation logic takes effect (same model/loss, just fixes a training data bug). Finally, I keep the submission format unchanged but ensure the RLE encoder outputs strictly 1-indexed starts and sorted pairs with no zero-length runs, producing a valid `submission.csv`.'
- What this solution (achieved 0.5221) has done: 'I fix the protobuf monkey-patch so it doesn’t trigger the `GetPrototype` AttributeError before TensorFlow imports, restoring end-to-end execution without changing the model. Then I fix the `model.fit()` crash by wrapping the existing Python generator in a `tf.data.Dataset` with an explicit `output_signature` (Keras 3 requirement), keeping the same augmentation logic and training loop semantics. Finally, I keep prediction, thresholding, and RLE submission writing intact so a valid `submission.csv` is always produced. These changes are primarily runtime/logic fixes; the corrected training input pipeline should also improve score versus a run that fails to train properly.'
- What this solution (achieved 0.5221) has done: 'I fix the protobuf monkey-patch condition that currently triggers an AttributeError before TensorFlow can import, which is preventing end-to-end execution. Then I keep the model/training logic intact but add a score-improving (and competition-standard) test-time augmentation by averaging predictions over the original + flipped versions before thresholding, which typically improves mAP-IoU without changing the architecture or training loop. Finally, I make depth normalization for test consistent with train (use the same max-z scaling) and keep the same RLE encoder while ensuring the submission is written as a valid `submission.csv` with `id,rle_mask`.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

try:
    from google.protobuf import message_factory as _message_factory  # noqa: E402

    if (
        hasattr(_message_factory, "MessageFactory")
        and hasattr(_message_factory.MessageFactory, "GetMessageClass")
        and not hasattr(_message_factory.MessageFactory, "GetPrototype")
    ):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        _message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception as _e:
    print("Warning: protobuf monkey-patch skipped due to:", repr(_e))

import random as rn
import numpy as np
import pandas as pd
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from keras import backend as K

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(1234)
rn.seed(1234)
tf.random.set_seed(1234)

INPUT_DIR = "../input"
if not os.path.exists(INPUT_DIR):
    if os.path.exists("/kaggle/input"):
        cand = "/kaggle/input/tgs-salt-identification-challenge"
        INPUT_DIR = cand if os.path.exists(cand) else "/kaggle/input"
    elif os.path.exists("/kaggle/data/tgs-salt-identification-challenge"):
        INPUT_DIR = "/kaggle/data/tgs-salt-identification-challenge"

print("Using INPUT_DIR:", INPUT_DIR)
print("INPUT_DIR contents (top):", os.listdir(INPUT_DIR)[:20])

df_train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
df_depths = pd.read_csv(os.path.join(INPUT_DIR, "depths.csv"))

DEPTHS_MAX_Z = float(np.max(df_depths["z"].values))
df_depths["z"] = df_depths["z"] / (DEPTHS_MAX_Z if DEPTHS_MAX_Z != 0 else 1.0)

df_train = df_train.merge(df_depths, on="id", how="inner")
df_train = df_train.rename(columns={"z": "depth"})
df_train = df_train[["id", "rle_mask", "depth"]]

print(df_train.head())
print("\nTrain shape:", df_train.shape)
print(df_depths.head())
print("\nDepths shape:", df_depths.shape)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_img_dir = os.path.join(INPUT_DIR, "train", "images")
train_mask_dir = os.path.join(INPUT_DIR, "train", "masks")

df_train["images"] = [
    np.array(Image.open(os.path.join(train_img_dir, f"{idx}.png")))
    for idx in df_train["id"]
]
print("Sample Image Shape is ", df_train["images"].iloc[0].shape)

df_train["images"] = pd.Series(
    map(lambda x: np.delete(x, np.s_[1:], 2), df_train["images"])
)
print("After optimization, Image Shape is ", df_train["images"].iloc[0].shape)
print("No. of train images are ", len(df_train))

df_train["images"] = df_train["images"] / 255.0
print("After normalization, pixel value is ", df_train["images"].iloc[0][1, 0])

df_train["masks"] = [
    np.array(Image.open(os.path.join(train_mask_dir, f"{idx}.png")))
    for idx in df_train["id"]
]
print("\nSample Mask Shape is ", df_train["masks"].iloc[0].shape)
print("No. of mask images are ", len(df_train))
print("Before normalization, pixel value is ", df_train["masks"].iloc[15][10, 0])
print("train df columns are ", df_train.columns)



## === cell 2
num_train_images = len(df_train)

train_x = np.stack(df_train["images"].values, axis=0).astype(np.float32)
train_x = train_x.reshape((num_train_images, 101, 101, 1))
print("Train Shape = ", train_x.shape)
print("Sample Pixel Value", train_x[0, 1, 0, 0])

train_y = np.stack(df_train["masks"].values, axis=0).astype(np.float32)
if train_y.ndim == 3:
    train_y = train_y[..., None]
train_y = train_y.reshape((num_train_images, 101, 101, 1))
train_y = train_y / (train_y.max() if train_y.max() != 0 else 1.0)
train_y = np.round(train_y).astype(np.float32)
print("Mask Shape = ", train_y.shape)
print("Sample Mask Pixel Value", train_y[15, 10, 0, 0])



## === cell 3
print(train_x.shape)
print(train_y.shape)



## === cell 4
import scipy.signal as sg
from keras.models import Model
from keras import layers

Height = 101
Width = 101



## === cell 5
df_train["depth_image"] = pd.Series(
    map(
        lambda x: np.full(shape=(50, 50, 1), fill_value=x, dtype=np.float32),
        df_train["depth"],
    )
)

depth_np = np.stack(df_train["depth_image"].values, axis=0).astype(np.float32)
depth_np = depth_np.reshape((num_train_images, 50, 50, 1))

depth_np_new = np.array(df_train["depth"], dtype=np.float32)
print("depth", depth_np_new[0:5])
print(depth_np_new.shape)
depth_np_new = depth_np_new.reshape((num_train_images, 1, 1, 1))
print("depth shape ", depth_np_new.shape)
print(df_train.columns)
print(depth_np.shape)



## === cell 6
Activation1 = "elu"
Activation2 = "tanh"


def convolution_block(
    x, filters, size, strides=(1, 1), padding="same", activation=True
):
    x = layers.Conv2D(filters, size, strides=strides, padding=padding)(x)
    if activation is True:
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
print(x_16_pool)

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

inverse = layers.Conv2DTranspose(
    filters=128, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation="elu"
)(inverse)
print("ok ", inverse)

inverse = layers.Concatenate()([inverse, x_128_pool])
print("concatination ", inverse)
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
from keras.callbacks import EarlyStopping

Batch_size = 96
gen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)


def gen_flow_for_two_inputs(X1, X2, y):
    genX1 = gen.flow(X1, y, batch_size=Batch_size, seed=1234, shuffle=True)
    genX2 = gen.flow(X2, X2, batch_size=Batch_size, seed=1234, shuffle=True)
    while True:
        X1i = next(genX1)  # (batch_X1, batch_y)
        X2i = next(genX2)  # (batch_X2, batch_X2)
        yield (X1i[0], X2i[0]), X1i[1]


def make_tf_dataset(X1, X2, y):
    output_signature = (
        (
            tf.TensorSpec(shape=(None, 101, 101, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 50, 50, 1), dtype=tf.float32),
        ),
        tf.TensorSpec(shape=(None, 101, 101, 1), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(
        lambda: gen_flow_for_two_inputs(X1, X2, y),
        output_signature=output_signature,
    )
    return ds.prefetch(tf.data.AUTOTUNE)


def get_iou(y_true, y_pred):
    y_true = y_true.astype(np.bool_).flatten()
    y_pred = np.round(y_pred).astype(np.bool_).flatten()
    inter = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    return (inter + 1e-9) / (union + 1e-9)


def my_iou_metric(label, pred):
    return tf.numpy_function(get_iou, [label, pred > 0.5], tf.float64)




## === cell 9
import keras.losses as losses


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
    return 0.2 * losses.binary_crossentropy(y_true, y_pred) + 0.8 * dice_loss(
        y_true, y_pred
    )




## === cell 10
epochs = 60

model.compile(optimizer="adam", loss=bce_dice_loss, metrics=[dice_loss, "acc"])

early_stopping = EarlyStopping(
    monitor="dice_loss", mode="min", patience=3, restore_best_weights=False
)

steps_per_epoch = int(np.ceil(num_train_images / Batch_size))
train_ds = make_tf_dataset(train_x, depth_np, train_y)

history = model.fit(
    train_ds,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    callbacks=[early_stopping],
    verbose=1,
)

model.save("model_7.h5")



## === cell 11
df_test = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))[["id"]].copy()

df_depths_test_raw = pd.read_csv(os.path.join(INPUT_DIR, "depths.csv"))
df_depths_test_raw["z"] = df_depths_test_raw["z"] / (
    DEPTHS_MAX_Z if DEPTHS_MAX_Z != 0 else 1.0
)

df_test = df_test.merge(df_depths_test_raw, on="id", how="left")
print("Length of df_test is ", len(df_test))
print(df_test.head())



## === cell 12
test_img_dir = os.path.join(INPUT_DIR, "test", "images")
df_test["images"] = [
    np.array(Image.open(os.path.join(test_img_dir, f"{idx}.png")))
    for idx in df_test["id"]
]



## === cell 13
print("Sample Image Shape is ", df_test["images"].iloc[10].shape)



## === cell 14
df_test["images"] = pd.Series(
    map(lambda x: np.delete(x, np.s_[1:], 2), df_test["images"])
)
print("After optimization, Image Shape is ", df_test["images"].iloc[0].shape)
print("No. of test images are ", len(df_test))
print("Before normalization, pixel value is ", df_test["images"].iloc[10][2, 0])

df_test["images"] = df_test["images"] / 255.0
print("After normalization, pixel value is ", df_test["images"].iloc[10][2, 0])

print(df_test.columns)

num_test_images = len(df_test)
test_x = (
    np.stack(df_test["images"].values, axis=0)
    .astype(np.float32)
    .reshape((num_test_images, 101, 101, 1))
)

print(test_x.max())
print(test_x.min())
print(train_x.max())
print(train_x.min())
print("Test Shape = ", test_x.shape)
print("Sample Pixel Value", test_x[10, 2, 0, 0])
print(df_test.columns)



## === cell 15
df_test["depth_image"] = pd.Series(
    map(
        lambda x: np.full(shape=(50, 50, 1), fill_value=x, dtype=np.float32),
        df_test["z"].values.astype(np.float32),
    )
)

depth_test_np = (
    np.stack(df_test["depth_image"].values, axis=0)
    .astype(np.float32)
    .reshape((num_test_images, 50, 50, 1))
)

print(df_test.columns)
print(depth_test_np.shape)




## === cell 16
def post_process(x):
    num_files = x.shape[0]
    main_list = []
    pic_size = Height * Width
    for i in range(num_files):
        pic = x[i].T.flatten()
        runs = []
        run_len = 0
        run_start = 0
        for j in range(pic_size):
            if pic[j] == 1:
                if run_len == 0:
                    run_start = j + 1  # 1-indexed start
                run_len += 1
            else:
                if run_len > 0:
                    runs.append(str(run_start))
                    runs.append(str(run_len))
                    run_len = 0
        if run_len > 0:
            runs.append(str(run_start))
            runs.append(str(run_len))
        main_list.append(" ".join(runs))
    return main_list




## === cell 17
def predict_with_tta(model, x_img, x_depth, batch_size=96):
    p0 = model.predict([x_img, x_depth], batch_size=batch_size, verbose=1)

    x_h = x_img[:, :, ::-1, :]  # horizontal flip (width axis)
    p_h = model.predict([x_h, x_depth], batch_size=batch_size, verbose=0)
    p_h = p_h[:, :, ::-1, :]

    x_v = x_img[:, ::-1, :, :]  # vertical flip (height axis)
    p_v = model.predict([x_v, x_depth], batch_size=batch_size, verbose=0)
    p_v = p_v[:, ::-1, :, :]

    x_hv = x_img[:, ::-1, ::-1, :]
    p_hv = model.predict([x_hv, x_depth], batch_size=batch_size, verbose=0)
    p_hv = p_hv[:, ::-1, ::-1, :]

    return (p0 + p_h + p_v + p_hv) / 4.0


print("Predicting now.....")
y_pred = predict_with_tta(model, test_x, depth_test_np, batch_size=Batch_size)
print("Prediction Complete", y_pred.shape)



## === cell 18
y_pred = (y_pred > 0.5).astype(np.uint8)
print("Y Prediction shape =", y_pred.shape)
print(y_pred.min(), y_pred.max())

new_y_pred = post_process(y_pred)
print("Y Prediction length =", len(new_y_pred))



## === cell 19
output = pd.DataFrame(data={"id": df_test["id"].values, "rle_mask": new_y_pred})
output.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", output.shape)
print(output.head())



## === cell 20
import time

print(time.time())
print(np.random.rand(10))
