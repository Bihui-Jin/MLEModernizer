# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

# 5. Target score

0.68244

# 6. Current score

0.5221

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5221) has done: 'I fix the runtime-breaking incompatibilities caused by using TF1-style sessions/ConfigProto/tf.set_random_seed and legacy Keras imports in a TF 2.18 + Keras 3 environment. I also correct hardcoded image counts/shapes (4000/18000) to use the actual dataset sizes (3000 train, 1000 test) so reshapes don’t crash and predictions align with ids. Finally, I keep the model architecture and training loop semantics intact, but update the data generator import (or remove it since it isn’t used) and replace deprecated `tf.py_func` usage to avoid execution errors; the script always write a valid `submission.csv` with the required `id,rle_mask` columns.'
- What this solution (achieved 0.5221) has done: 'I fix the runtime-crashing `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` by pinning `protobuf` to a compatible version (this is a known TF/Keras + protobuf 6.x break) before importing TensorFlow. Then I fix a logic bug where you create a train/val split but still train on the full dataset (data leakage that hurts real generalization and Kaggle score); training use the split arrays while keeping the same model, loss, and epochs. Finally, I ensure the EarlyStopping callback monitors the correct validation metric (`val_dice_coeff`) and that depth normalization for test uses the same scale as train, which is a score-relevant calibration fix without changing the core architecture.'
- What this solution (achieved 0.5221) has done: 'You’re currently leaving score on the table mostly in the *prediction-to-RLE* stage: the mask threshold is hardcoded via `np.round(y_pred - shift)` and you pick `i=6` without validating what threshold best matches the competition’s IoU-sweeping metric. I keep your model/training exactly as-is, but add a small “threshold tuning” step using your existing validation split: evaluate a handful of fixed thresholds on the validation set with the official-ish mAP@IoU(0.50:0.95) formulation (single object per image), then use the best threshold to binarize the test predictions. I also fix a subtle RLE edge case where the last pixel uses `j` instead of `j+1`, which can corrupt encodings for masks ending on the final pixel (hurting score). These are minimal, score-relevant changes that typically move results upward without changing architecture or training semantics.'
- What this solution (achieved 0.5221) has done: 'I keep your model and training loop intact and focus on two score-relevant issues that can realistically move 0.5221 toward 0.68244 without architectural changes: (1) fix depth normalization mismatch between train and test (currently test depth is divided by `depth_train_max` while train depth was normalized by max z, which hurts generalization), and (2) make threshold selection better aligned with the competition metric by searching a slightly finer threshold grid and using the best one from the validation split. I also make a minimal correction to the RLE encoder to robustly close a run at the final pixel (this can affect a subset of masks and hurt score). These changes preserve evaluation semantics and only affect calibration/post-processing, which is the smallest lever likely to improve Kaggle mAP here.'
- What this solution (achieved 0.5221) has done: 'Your current gap to the target is +0.16034 (0.5221 → 0.68244), so we should improve score with the smallest, safest changes that don’t alter the model/training core. The biggest score-relevant issue left is that your depth channel preprocessing is inconsistent between train and test (you renormalize test depth by its own max), which shifts the depth distribution at inference and typically hurts generalization; we make test depth normalization match train exactly. Next, we make threshold tuning slightly more robust (still minimal) by (a) setting `restore_best_weights=True` so the model used for predictions corresponds to the best validation dice during training, and (b) searching a small but slightly wider threshold grid to better align binarization to mAP@IoU. These changes keep architecture, loss, and training loop semantics intact and only affect calibration/preprocessing/post-processing.'
- What this solution (achieved 0.5221) has done: 'We keep your model/training loop intact and focus on small, score-relevant fixes in post-processing that affect the Kaggle metric: (1) align RLE encoding orientation with the competition’s expected “top-to-bottom then left-to-right” (column-major) order by switching from `pic = x[i].T.flatten()` to `pic = x[i].flatten(order="F")` (equivalent intent, safer and less error-prone), and (2) add the standard “empty mask gating” used in this competition by tuning a per-image “empty threshold” on the validation set to decide when to output an empty RLE (this reduces false positives and typically improves mAP@IoU). We also ensure test depth normalization matches train exactly (your current code is close, but we make it strictly consistent by using the same normalized `depth` definition for both). These are minimal calibration/encoding changes that commonly move scores upward without changing architecture or training semantics. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.5221) has done: 'I keep your model, loss, and training loop intact and focus only on score-relevant post-processing. The main change is to tune the binarization threshold using a validation proxy that more closely matches the competition’s “mask IoU mAP” behavior by applying the same “empty-mask gating” logic during scoring and by using mean probability (not max) as the emptiness signal, which is usually more stable for this dataset. I also make the empty-RLE output explicit (use empty string) and ensure the validation scoring uses the exact same binarization pipeline as the test-time pipeline. These minimal calibration/post-processing changes are the smallest levers likely to move 0.5221 upward toward 0.68244 without changing core architecture or training semantics.'
- What this solution (achieved 0.5221) has done: 'We keep your model/training exactly the same and only adjust post-processing to better match the competition’s mAP@IoU behavior, since your current gap to the target (0.5221 → 0.68244) suggests calibration/post-processing is the safest lever. Specifically, we fix a subtle but important edge case in the mAP proxy scoring: when both GT and prediction are empty, the per-threshold precision should be 1.0 (currently it becomes 0.0 due to denom logic), which can mis-tune thresholds and hurt public LB. Then we tune the “empty-mask gate” using a more appropriate emptiness signal for this dataset (mean probability plus an alternate “fraction of pixels above threshold” signal) while keeping the same threshold grid size to stay minimal and fast. Finally, we ensure train/test depth scaling is strictly consistent by using the same `z_norm`-based depth image construction for test without any extra rescaling (your test currently divides by `depth_train_max`, which can introduce a distribution shift).'
- What this solution (achieved 0.5221) has done: 'We keep your model, loss, and training loop intact and only adjust the validation-based threshold/empty-mask tuning so it better matches Kaggle’s mAP@IoU for this single-object-per-image task. The current proxy scorer treats “prediction non-empty but tiny IoU” as an FP even when GT is empty; we switch to the standard competition-like confusion logic where emptiness is handled explicitly (empty/empty is perfect; empty/nonnull is FP; nonempty/empty is FN), which improves threshold selection without changing predictions themselves. Then we make the empty-mask gate use a single, stable signal (mean probability) and search a small joint grid efficiently to avoid overfitting noise from two simultaneous gates. Finally, we keep your RLE encoding and submission writing unchanged, ensuring a valid `submission.csv`.'
- What this solution (achieved 0.5221) has done: 'I keep your model and training loop unchanged and only adjust post-processing to better match the competition metric, since your current gap (0.5221 → 0.68244) is best addressed by thresholding/gating. Specifically, I tune the empty-mask gate using a more reliable “emptiness” signal (fraction of pixels above threshold) instead of mean probability, because mean can be biased by widespread low-confidence noise and can suppress real masks. I also change the validation proxy scoring to the standard single-object competition logic (empty/empty=1, empty/nonempty=0, nonempty/empty=0, otherwise IoU-threshold sweep), which avoids mis-tuning due to FP/FN bookkeeping artifacts. Finally, I keep your RLE encoder and submission format intact, only ensuring the tuned parameters are applied consistently at test time.'

# 9. Code solution

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

df_train["images"] = [
    np.array(Image.open(os.path.join(train_img_dir, f"{idx}.png")))
    for idx in df_train["id"]
]
print("Sample Image Shape is", df_train["images"].iloc[0].shape)

df_train["images"] = pd.Series(
    map(lambda x: np.delete(x, np.s_[1:], 2), df_train["images"])
)
print("After optimization, Image Shape is", df_train["images"].iloc[0].shape)
print("No. of train images are", len(df_train))

df_train["images"] = df_train["images"] / 255.0
print("After normalization, pixel value is", df_train["images"].iloc[0][1, 0])

df_train["masks"] = [
    np.array(Image.open(os.path.join(train_mask_dir, f"{idx}.png")))
    for idx in df_train["id"]
]
print("\nSample Mask Shape is", df_train["masks"].iloc[0].shape)
print("No. of mask images are", len(df_train))
print("Before normalization, pixel value is", df_train["masks"].iloc[15][10, 0])

print("train df columns are", df_train.columns.tolist())



## === cell 2
num_train_images = len(df_train)

train_x = np.array(df_train["images"].to_list(), dtype=np.float32)  # (N,101,101,1)
train_x = np.reshape(train_x, (num_train_images, 101, 101, 1))
print("Train Shape =", train_x.shape)
print("Sample Pixel Value", train_x[0, 1, 0, 0])

train_y = np.array(df_train["masks"].to_list(), dtype=np.float32)  # (N,101,101)
train_y = np.reshape(train_y, (num_train_images, 101, 101, 1))
train_y = train_y / (train_y.max() + 1e-9)
train_y = np.round(train_y)
print("Mask Shape =", train_y.shape)
print("Sample Pixel Value", train_y[15, 10, 0, 0])



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
df_train["depth_image"] = pd.Series(
    map(
        lambda x: np.full(shape=(50, 50, 1), fill_value=x, dtype=np.float32),
        df_train["depth"],
    )
)

depth_np = np.array(df_train["depth_image"].to_list(), dtype=np.float32)
depth_np = np.reshape(depth_np, (num_train_images, 50, 50, 1))

depth_train_max = float(np.max(df_train["depth"].values))

depth_np_new = np.array(df_train["depth"].to_list(), dtype=np.float32).reshape(
    (num_train_images, 1, 1, 1)
)
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
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(16, 8))
columns = 3
rows = 4
fig, axs = plt.subplots(rows, columns, figsize=(16, 12))
index = np.random.randint(0, val_x.shape[0], rows)

val_x_pred_new = (val_x_pred > 0.25).astype(np.float32)

print("Sample indices:", index)

for i in range(rows):
    axs[i, 0].imshow(val_x[index[i], :, :, 0], cmap="gray")
    axs[i, 0].set_title("image")
    axs[i, 0].axis("off")
    axs[i, 1].imshow(val_y[index[i], :, :, 0], cmap="gray")
    axs[i, 1].set_title("mask")
    axs[i, 1].axis("off")
    axs[i, 2].imshow(val_x_pred_new[index[i], :, :, 0], cmap="gray")
    axs[i, 2].set_title("pred (thr=0.25)")
    axs[i, 2].axis("off")

plt.tight_layout()
plt.show()




## === cell 13
def iou_numpy(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0  # both empty => perfect
    return inter / union


def mean_ap_iou_single(gt, pr, thresholds=np.arange(0.5, 1.0, 0.05)):
    gt = gt.astype(bool)
    pr = pr.astype(bool)

    gt_any = gt.sum() > 0
    pr_any = pr.sum() > 0

    if (not gt_any) and (not pr_any):
        return 1.0
    if gt_any != pr_any:
        return 0.0

    iou = iou_numpy(gt, pr)
    return float(np.mean([1.0 if iou > t else 0.0 for t in thresholds]))


def apply_threshold_and_empty_gate(y_prob, thr, empty_frac_thr=None):
    y_prob2 = y_prob[..., 0]
    y_pred_bin = y_prob2 > thr

    if empty_frac_thr is not None:
        frac = y_pred_bin.reshape((y_pred_bin.shape[0], -1)).mean(axis=1)
        for i in range(y_pred_bin.shape[0]):
            if frac[i] < empty_frac_thr:
                y_pred_bin[i, :, :] = False

    return y_pred_bin


def score_threshold_on_val(y_true, y_prob, thr, empty_frac_thr=None):
    y_true_bin = y_true[..., 0] > 0.5
    y_pred_bin = apply_threshold_and_empty_gate(
        y_prob, thr=thr, empty_frac_thr=empty_frac_thr
    )
    scores = [
        mean_ap_iou_single(y_true_bin[i], y_pred_bin[i]) for i in range(y_true.shape[0])
    ]
    return float(np.mean(scores))


thr_grid = list(np.round(np.arange(0.10, 0.61, 0.02), 3))
empty_frac_grid = list(np.round(np.arange(0.000, 0.021, 0.001), 3))  # 0.0 => no gate

best_score = -1.0
best_thr = 0.25
best_empty_frac = None

for thr in thr_grid:
    for ef in empty_frac_grid:
        ef_use = None if ef == 0.0 else float(ef)
        s = score_threshold_on_val(
            val_y, val_x_pred, thr=float(thr), empty_frac_thr=ef_use
        )
        if s > best_score:
            best_score = s
            best_thr = float(thr)
            best_empty_frac = ef_use
    print(
        f"Finished thr={thr:.3f} (current best_score={best_score:.5f}, best_thr={best_thr}, best_empty_frac={best_empty_frac})"
    )

print("Best on validation:")
print("  best_score      =", best_score)
print("  best_thr        =", best_thr)
print("  best_empty_frac =", best_empty_frac)



## === cell 14
sample_sub_path = os.path.join(DATA_ROOT, "sample_submission.csv")
df_test = pd.read_csv(sample_sub_path)[["id"]].merge(
    df_depths[["id", "z_norm"]], on="id", how="left"
)
df_test = df_test.rename(columns={"z_norm": "depth"})

print("Length of df_test is", len(df_test))
print(df_test.head())



## === cell 15
test_img_dir = os.path.join(DATA_ROOT, "test", "images")
df_test["images"] = [
    np.array(Image.open(os.path.join(test_img_dir, f"{idx}.png")))
    for idx in df_test["id"]
]



## === cell 16
print("Sample Image Shape is", df_test["images"].iloc[10].shape)



## === cell 17
df_test["images"] = pd.Series(
    map(lambda x: np.delete(x, np.s_[1:], 2), df_test["images"])
)
print("After optimization, Image Shape is", df_test["images"].iloc[0].shape)
print("No. of test images are", len(df_test))
print("Before normalization, pixel value is", df_test["images"].iloc[10][2, 0])

df_test["images"] = df_test["images"] / 255.0
print("After normalization, pixel value is", df_test["images"].iloc[10][2, 0])

test_x = np.array(df_test["images"].to_list(), dtype=np.float32)
test_x = np.reshape(test_x, (len(df_test), 101, 101, 1))
print("Test Shape =", test_x.shape)
print("Sample Pixel Value", test_x[10, 2, 0, 0])



## === cell 18
df_test["depth_image"] = pd.Series(
    map(
        lambda x: np.full(shape=(50, 50, 1), fill_value=x, dtype=np.float32),
        df_test["depth"],
    )
)
depth_test_np = np.array(df_test["depth_image"].to_list(), dtype=np.float32)
depth_test_np = np.reshape(depth_test_np, (len(df_test), 50, 50, 1))

print("df_test columns:", df_test.columns.tolist())
print("depth_test_np shape", depth_test_np.shape)




## === cell 19
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




## === cell 20
print("Predicting now.....")
y_pred = model.predict([test_x, depth_test_np], batch_size=Batch_size, verbose=2)
print("Prediction Complete", y_pred.shape)



## === cell 21
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8, 8))
rows = 4
print("Pred mask shape:", y_pred[0, :, :, 0].shape)

y_pred_bin = apply_threshold_and_empty_gate(
    y_pred,
    thr=best_thr,
    empty_frac_thr=best_empty_frac,
)
y_pred_new = y_pred_bin[..., None].astype(np.float32)

for i in range(1, rows):
    fig.add_subplot(rows, 1, i)
    xidx = np.random.randint(0, len(df_test), 1)[0]
    print("plot idx:", xidx)
    plt.imshow(y_pred_new[xidx, :, :, 0], cmap="gray")
    plt.axis("off")
plt.show()



## === cell 22
y_pred_bin = apply_threshold_and_empty_gate(
    y_pred,
    thr=best_thr,
    empty_frac_thr=best_empty_frac,
)
new_y_pred = post_process(y_pred_bin[..., None].astype(np.uint8))

submission = pd.DataFrame({"id": df_test["id"].values, "rle_mask": new_y_pred})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print("Wrote", submission_path, "with shape", submission.shape)
print(submission.head())
