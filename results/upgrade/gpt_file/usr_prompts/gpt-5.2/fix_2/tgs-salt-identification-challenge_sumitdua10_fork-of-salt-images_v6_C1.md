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

0.72268

# 6. Current score

0.643

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.643) has done: 'I fix the shape/indexing bugs by building `train_x/train_y/depth_np` using `np.stack` with the actual number of training rows (3000) instead of hardcoded 4000, and similarly build `test_x/depth_test_np` from the sample submission ids (1000) to avoid empty test sets. I remove the broken `ImageDataGenerator` import (Keras 3 no longer exposes it there) and keep the same direct `model.fit` training approach. I also replace the protobuf-related crash by using `tf.keras` APIs consistently for model/layers/callbacks. Finally, I ensure the submission is written as a valid `submission.csv` with correct `id,rle_mask` columns and deterministic ordering.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from PIL import Image

INPUT_DIR = "../input"

CANDIDATES = [
    os.path.join(INPUT_DIR, "tgs-salt-identification-challenge"),
    INPUT_DIR,
]
BASE_PATH = None
for c in CANDIDATES:
    if os.path.exists(os.path.join(c, "train", "images")) and os.path.exists(
        os.path.join(c, "test", "images")
    ):
        BASE_PATH = c
        break
if BASE_PATH is None:
    BASE_PATH = INPUT_DIR

print("Using BASE_PATH:", BASE_PATH)
print("Listing input dir:", os.listdir(INPUT_DIR)[:20])

train_csv_path = os.path.join(BASE_PATH, "train.csv")
depths_csv_path = os.path.join(BASE_PATH, "depths.csv")
sample_sub_path = os.path.join(BASE_PATH, "sample_submission.csv")

df_train = pd.read_csv(train_csv_path)
print(df_train.head())
print("\nTrain files shape is", df_train.shape)

df_depths = pd.read_csv(depths_csv_path)
df_depths["z"] = df_depths["z"] / df_depths["z"].max()
print(df_depths.head())
print("\nDepths files shape is", df_depths.shape)

df_train = df_train.merge(df_depths, on="id", how="inner")
df_train = df_train.rename(columns={"z": "depth"})
df_train = df_train[["id", "rle_mask", "depth"]]
print(df_train.head())



## === cell 1
train_img_dir = os.path.join(BASE_PATH, "train", "images")
train_mask_dir = os.path.join(BASE_PATH, "train", "masks")


def load_grayscale_png(path):
    arr = np.array(Image.open(path))
    if arr.ndim == 2:
        arr = arr[..., None]
    elif arr.ndim == 3:
        arr = arr[:, :, :1]
    return arr


df_train["images"] = [
    load_grayscale_png(os.path.join(train_img_dir, f"{idx}.png"))
    for idx in df_train["id"]
]
print("Sample Image Shape is", df_train["images"].iloc[0].shape)

df_train["images"] = df_train["images"].apply(lambda x: x.astype(np.float32) / 255.0)
print("After normalization, pixel value is", df_train["images"].iloc[0][1, 0, 0])

df_train["masks"] = [
    load_grayscale_png(os.path.join(train_mask_dir, f"{idx}.png"))
    for idx in df_train["id"]
]
print("Sample Mask Shape is", df_train["masks"].iloc[0].shape)
print(
    "Before normalization, pixel value is",
    (
        df_train["masks"].iloc[15][10, 0, 0]
        if len(df_train) > 15
        else df_train["masks"].iloc[0][10, 0, 0]
    ),
)

print("train df columns are", df_train.columns.tolist())
print("No. of train images are", len(df_train))



## === cell 2
train_x = np.stack(df_train["images"].to_list(), axis=0).astype(
    np.float32
)  # (N,101,101,1)
train_y = np.stack(df_train["masks"].to_list(), axis=0).astype(
    np.float32
)  # (N,101,101,1)

mx = train_y.max()
if mx > 0:
    train_y = train_y / mx
train_y = np.round(train_y).astype(np.int32)

print("Train Shape =", train_x.shape)
print("Mask Shape  =", train_y.shape)
print("Sample Pixel Value X", train_x[0, 1, 0, 0])
print("Sample Pixel Value Y", train_y[min(15, len(train_y) - 1), 10, 0, 0])



## === cell 3
import random as rn
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(7)
rn.seed(12345)
tf.random.set_seed(7)

Height = 101
Width = 101



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4
num_train_images = len(df_train)

df_train["depth_image"] = df_train["depth"].apply(
    lambda x: np.full(shape=(50, 50, 1), fill_value=x, dtype=np.float32)
)
depth_np = np.stack(df_train["depth_image"].to_list(), axis=0).astype(
    np.float32
)  # (N,50,50,1)

depth_np_new = (
    df_train["depth"].to_numpy(dtype=np.float32).reshape((num_train_images, 1, 1, 1))
)

print("depth", depth_np_new[:5].reshape(-1))
print("depth_np_new shape", depth_np_new.shape)
print("depth_np shape", depth_np.shape)



## === cell 5
img_input = layers.Input(shape=(Height, Width, 1), name="img_input")
depth_input = layers.Input(shape=(50, 50, 1), name="depth_input")

x = layers.Conv2D(filters=16, kernel_size=(2, 2), padding="valid", activation="elu")(
    img_input
)
x_depth = layers.Conv2D(
    filters=16, kernel_size=(2, 2), padding="valid", activation="elu"
)(depth_input)

x_16_pool = layers.MaxPooling2D(pool_size=(2, 2))(x)
x_16_pool = layers.BatchNormalization()(x_16_pool)
x_16_pool = layers.Dropout(0.25)(x_16_pool)

x_16_pool_depth = layers.AveragePooling2D(pool_size=(2, 2))(x_depth)
x_16_pool_depth = layers.BatchNormalization()(x_16_pool_depth)
x_16_pool_depth = layers.Dropout(0.2)(x_16_pool_depth)

x = layers.Conv2D(32, 3, padding="valid", activation="elu")(x_16_pool)
x2 = layers.Conv2D(32, 3, padding="same", activation="tanh")(x)
x = layers.add([x, x2])
x = layers.BatchNormalization()(x)

x_depth = layers.Conv2D(32, 3, padding="valid", activation="elu")(x_16_pool_depth)
x2_depth = layers.Conv2D(32, 3, padding="same", activation="tanh")(x_depth)
x_depth = layers.add([x_depth, x2_depth])
x_depth = layers.BatchNormalization()(x_depth)

x_32_pool = layers.MaxPooling2D(2)(x)
x_32_pool_depth = layers.AveragePooling2D(2)(x_depth)

x = layers.Dropout(0.25)(x_32_pool)
x_depth = layers.Dropout(0.2)(x_32_pool_depth)

x = layers.Conv2D(64, 3, padding="same", activation="elu")(x)
x3 = layers.Conv2D(64, 3, padding="same", activation="elu")(x)
x = layers.add([x, x3])
x = layers.BatchNormalization()(x)

x_depth = layers.Conv2D(64, 3, padding="same", activation="elu")(x_32_pool_depth)
x2_depth = layers.Conv2D(64, 3, padding="same", activation="tanh")(x_depth)
x_depth = layers.add([x_depth, x2_depth])
x_depth = layers.BatchNormalization()(x_depth)

x_64_pool = layers.MaxPooling2D(2)(x)
x_64_pool_depth = layers.AveragePooling2D(2)(x_depth)

x_64_pool = layers.Dropout(0.25)(x_64_pool)
x_64_pool_depth = layers.Dropout(0.2)(x_64_pool_depth)

x = layers.Conv2D(128, 3, padding="same", activation="elu")(x_64_pool)
x2 = layers.Conv2D(128, 3, padding="same", activation="elu")(x)
x = layers.add([x, x2])
x_128_pool = layers.MaxPooling2D(2)(x)
x_128_pool = layers.Dropout(0.25)(x_128_pool)

x = layers.Conv2D(192, 3, padding="same", activation="elu")(x_128_pool)
x_192_pool = layers.MaxPooling2D(2)(x)
x_192_pool = layers.Dropout(0.25)(x_192_pool)

x = layers.Conv2D(192, 2, padding="valid", activation="elu")(x_192_pool)
x = layers.MaxPooling2D(2)(x)
x = layers.Dropout(0.25)(x)



## === cell 6
inverse = layers.Conv2DTranspose(
    filters=192, kernel_size=(3, 3), strides=(2, 2), padding="valid", activation="elu"
)(x)
inverse = layers.Concatenate()([inverse, x_192_pool])
inverse = layers.Conv2D(
    filters=192, kernel_size=(2, 2), padding="same", activation="elu"
)(inverse)
inverse = layers.BatchNormalization()(inverse)
inverse = layers.Dropout(0.25)(inverse)

inverse = layers.Conv2DTranspose(
    filters=128, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation="elu"
)(inverse)
inverse = layers.Concatenate()([inverse, x_128_pool])
inverse = layers.Conv2D(
    filters=128, kernel_size=(3, 3), padding="same", activation="elu"
)(inverse)
inverse = layers.BatchNormalization()(inverse)
inverse = layers.Dropout(0.25)(inverse)

inverse = layers.Conv2DTranspose(
    filters=64, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation="elu"
)(inverse)
inverse = layers.Concatenate()([inverse, x_64_pool])
inverse = layers.Conv2D(
    filters=64, kernel_size=(3, 3), padding="same", activation="elu"
)(inverse)
inverse = layers.BatchNormalization()(inverse)
inverse = layers.Dropout(0.25)(inverse)

inverse = layers.Conv2DTranspose(
    filters=32, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation="elu"
)(inverse)
inverse = layers.Conv2D(
    filters=32, kernel_size=(3, 3), padding="same", activation="elu"
)(inverse)
inverse = layers.Concatenate()([inverse, x_32_pool])
inverse = layers.BatchNormalization()(inverse)
inverse = layers.Dropout(0.25)(inverse)

inverse = layers.Conv2DTranspose(
    filters=16, kernel_size=(4, 4), strides=(2, 2), padding="valid", activation="elu"
)(inverse)
inverse = layers.Concatenate()([inverse, x_16_pool])
inverse = layers.Concatenate()([inverse, depth_input])

inverse = layers.Conv2D(
    filters=16, kernel_size=(3, 3), padding="same", activation="elu"
)(inverse)
inverse = layers.BatchNormalization()(inverse)
inverse = layers.Dropout(0.25)(inverse)

inverse2 = layers.Conv2DTranspose(
    filters=1, kernel_size=(3, 3), strides=(2, 2), padding="valid", activation="sigmoid"
)(inverse)

model = Model([img_input, depth_input], inverse2)
model.summary()



## === cell 7
Batch_size = 96



## === cell 8
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

early_stopping = EarlyStopping(
    monitor="accuracy", patience=3, mode="max", restore_best_weights=False
)

model.fit(
    [train_x, depth_np],
    train_y,
    epochs=50,
    verbose=1,
    batch_size=96,
    callbacks=[early_stopping],
)

model.save("model_7.h5")



## === cell 9
df_sample = pd.read_csv(sample_sub_path)
df_test = df_sample[["id"]].copy()
df_test = df_test.merge(df_depths, on="id", how="left")  # z is already normalized
print("Length of df_test is", len(df_test))
print(df_test.head())



## === cell 10
test_img_dir = os.path.join(BASE_PATH, "test", "images")
df_test["images"] = [
    load_grayscale_png(os.path.join(test_img_dir, f"{idx}.png"))
    for idx in df_test["id"]
]



## === cell 11
print("Sample Image Shape is", df_test["images"].iloc[min(10, len(df_test) - 1)].shape)



## === cell 12
df_test["images"] = df_test["images"].apply(lambda x: x.astype(np.float32) / 255.0)
print(
    "After normalization, pixel value is",
    df_test["images"].iloc[min(10, len(df_test) - 1)][2, 0, 0],
)

test_x = np.stack(df_test["images"].to_list(), axis=0).astype(np.float32)
print("Test Shape =", test_x.shape)
print("Test min/max", test_x.min(), test_x.max())
print("Train min/max", train_x.min(), train_x.max())



## === cell 13
df_test["depth_image"] = (
    df_test["z"]
    .fillna(0.0)
    .apply(lambda x: np.full(shape=(50, 50, 1), fill_value=x, dtype=np.float32))
)
depth_test_np = np.stack(df_test["depth_image"].to_list(), axis=0).astype(np.float32)

print(df_test.columns.tolist())
print("depth_test_np shape", depth_test_np.shape)




## === cell 14
def post_process(x):
    num_files = x.shape[0]
    main_list = []
    pic_size = Height * Width
    for i in range(num_files):
        pic = x[
            i
        ].T.flatten()  # column-major numbering (top-to-bottom then left-to-right)
        s_parts = []
        length = 0
        start = 0

        for j in range(pic_size):
            if pic[j] == 1:
                length += 1
                if length == 1:
                    start = j + 1  # 1-indexed
            if pic[j] == 0 and length > 0:
                s_parts.append(str(start))
                s_parts.append(str(length))
                length = 0

        if length > 0:
            s_parts.append(str(start))
            s_parts.append(str(length))

        main_list.append(" ".join(s_parts))
    return main_list




## === cell 15
print("Predicting now.....")
y_pred = model.predict([test_x, depth_test_np], batch_size=96, verbose=1)
print("Prediction Complete:", y_pred.shape)



## === cell 16
y_pred_bin = np.round(y_pred - 0.25).astype(np.int32)
print("Y Prediction shape =", y_pred_bin.shape)
print("min/max", y_pred_bin.min(), y_pred_bin.max())

new_y_pred = post_process(y_pred_bin)
print("Y Prediction length =", len(new_y_pred))



## === cell 17
output = pd.DataFrame({"id": df_test["id"].values, "rle_mask": new_y_pred})
output.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape", output.shape)
print(output.head())



## === cell 18
import time

print("Done at:", time.time())
print(np.random.rand(10))
