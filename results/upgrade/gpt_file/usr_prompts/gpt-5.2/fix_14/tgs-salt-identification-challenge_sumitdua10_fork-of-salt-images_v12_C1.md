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

0.4425

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5221) has done: 'I fix the broken array shaping logic by stacking images/masks correctly based on the actual number of training/test samples instead of hardcoded sizes (the root cause of the reshape/concatenate errors). I also fix the Keras 3 incompatibilities by switching `ImageDataGenerator` to `tf.keras.preprocessing.image.ImageDataGenerator` and replacing deprecated `tf.py_func` with `tf.numpy_function`, and avoid the protobuf-related crash by using `tf.keras` consistently. Finally, I ensure the test set is built from `sample_submission.csv` (guaranteed 1000 IDs) and generate a valid `submission.csv` with correct RLE encoding (1-indexed, column-major), so the notebook runs end-to-end and yields a valid submission.'
- What this solution (achieved 0.5221) has done: 'I fix the TensorFlow/protobuf runtime crash happening at import time (the `MessageFactory.GetPrototype` error) by pinning protobuf’s Python implementation and avoiding the mixed `keras`/`tf.keras` stack in this script. Then I make a minimal score-improving change that preserves the core model/training logic: add the already-defined `my_iou_metric` to `model.compile` (metric-aligned monitoring) and calibrate the prediction thresholding to use a standard `> 0.5` cutoff instead of the current `round(y_pred - 0.25)` heuristic. Finally, I keep submission formatting identical but ensure RLE encoding always receives a clean 2D binary mask.'
- What this solution (achieved 0.5221) has done: 'We fix the protobuf-related TensorFlow import crash by forcing the pure-Python protobuf implementation before TensorFlow loads and by using a consistent `tf.keras` stack only. Then we fix the `my_iou_metric` “unknown rank” error by wrapping it as a proper Keras `Metric` that always returns a scalar float32 and has a defined shape, keeping the same underlying IoU logic. Finally, we ensure the depth input matches the model’s expected `(50,50,1)` shape for both train/test and keep the existing `>0.5` thresholding and RLE submission generation intact so the notebook runs end-to-end and writes `submission.csv`.'
- What this solution (achieved 0.5221) has done: 'We fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by ensuring protobuf uses the pure-Python implementation before TensorFlow loads and by avoiding any mixed `keras` vs `tf.keras` imports. Then we fix the training crash caused by `tf.numpy_function` inside a metric (not XLA compatible) by rewriting the IoU metric to be pure-TensorFlow (same IoU@0.5 intent, but no PyFunc), which is score-neutral except for enabling training to run. Finally, we keep the model, loss, data pipeline, and thresholding logic the same, and ensure the submission CSV is written with the required columns and valid RLE.'
- What this solution (achieved 0.026) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing protobuf to use the pure-Python implementation *before* any TensorFlow-related import and by also setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` at the process level via `sys.argv`-safe environment settings. Then I keep the existing model and training logic unchanged, but make the training generator actually used (it’s currently defined but not used), which is a minimal, legitimate change that usually improves generalization for this segmentation task without changing the architecture/loss. Finally, I ensure depth normalization is consistent for test rows with missing depths (fill with 0.0) and keep the exact same RLE encoding/submission format so a valid `submission.csv` is always produced.'
- What this solution (achieved 0.026) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation before TensorFlow is imported, and by removing the unused standalone `keras` stack so we only use `tf.keras`. Then I fix the data generator runtime error by replacing the removed `.next()` call with the correct Python iterator usage (`next(iterator)`), keeping the same augmentation logic and training loop semantics. Finally, I keep the same prediction thresholding and RLE formatting, ensuring the notebook runs end-to-end and reliably writes a valid `submission.csv` in the required format; these fixes should also materially improve the score versus the currently broken/ineffective training run.'
- What this solution (achieved 0.2017) has done: 'We fix the TensorFlow/protobuf import crash by moving the protobuf environment variables to the very top of the script and ensuring no TensorFlow/Keras modules are imported before that. Next, we fix the `model.fit` crash by making the two-input generator yield a tuple of inputs (not a Python list) and by providing an explicit `output_signature` via `tf.data.Dataset.from_generator`, which Keras 3 requires for generator-based training. Finally, we keep the model architecture/loss/thresholding/RLE logic unchanged, but ensure training actually runs end-to-end and a valid `submission.csv` is always written.'
- What this solution (achieved 0.2516) has done: 'I fix the TensorFlow import crash (`MessageFactory` / protobuf incompatibility) by setting the protobuf implementation environment variables *before any protobuf/tensorflow-related imports* and by importing TensorFlow via `tf.keras` only. Then I keep your architecture/training loop intact, but correct the generator’s second branch so the depth augmentation stays aligned with the image/mask augmentation (it currently augments depths using depths as labels, which can silently misalign batches and hurt score). Finally, I keep the same prediction thresholding and RLE encoding, and ensure a valid `submission.csv` is always written with the required columns.'
- What this solution (achieved 0.4971) has done: 'I fix the immediate runtime crash by ensuring the protobuf environment variables are set before *any* TensorFlow/protobuf-related import happens, and by forcing TensorFlow to use the legacy pure-Python protobuf implementation path. Then I keep your model, loss, generator, and thresholding intact, but make the training generator correctly apply the *same* augmentation transform to both image and depth branches by augmenting the depth input using the image-derived random transform parameters (instead of independently sampling transforms per branch). Finally, I keep the submission formatting identical while ensuring the script always writes `submission.csv` with the required `id,rle_mask` columns.'
- What this solution (achieved 0.3721) has done: 'We fix the immediate runtime crash in the TensorFlow import (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation at the very top and (critically) ensuring no standalone `keras` package gets imported indirectly; everything consistently use `tf.keras` only. Then we correct one depth-merge issue that can silently drop test rows: the test merge currently uses the normalized `df_depths` but then renames `z` later—this is kept but made robust so all 1000 IDs remain and depths are always float32. These changes are score-neutral except enabling the script to actually train/predict end-to-end reliably; with training actually running (instead of crashing), the score should move back toward your target.'
- What this solution (achieved 0.531) has done: 'You’re hitting the TensorFlow/protobuf `MessageFactory.GetPrototype` crash because TF 2.18 + protobuf 6.x can still load the C++ protobuf backend even when `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` is set, depending on import order; we force the Python protobuf backend *before any TF-related import* and also prevent Keras 3 standalone imports from being pulled in indirectly. After that, the script run end-to-end and write a valid `submission.csv`. To nudge score upward toward your target (current 0.3721 is far below 0.73762), we make a minimal, metric-aligned inference calibration change: apply a small fixed test-time augmentation (horizontal flip) and average predictions, without changing the model or training procedure. Everything else (architecture, loss, generator, training loop, RLE format) stays intact.'
- What this solution (achieved 0.168) has done: 'We fix the TensorFlow import crash (`MessageFactory` / protobuf) by forcing the pure-Python protobuf implementation *before* any TensorFlow-related import and by also disabling the C++ protobuf backend via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` early in the process. Then we keep your model/training/inference logic unchanged, but ensure the environment variables are set at the very top (cell 1) and remove any chance of importing TensorFlow before that. This is primarily a correctness/stability fix so the notebook runs end-to-end and writes `submission.csv`; it should be score-neutral relative to your current logic, while enabling the existing TTA (horizontal flip averaging) to actually execute.'
- What this solution (achieved 0.4425) has done: 'We fix the TensorFlow import crash (`MessageFactory` / protobuf) by forcing the pure-Python protobuf backend *before any TensorFlow-related imports* and by avoiding any standalone `keras` imports that can pull in incompatible protobuf bindings. Then we keep your model/training/inference logic unchanged, but make the environment setup robust and fail-fast so the pipeline always reaches training and submission writing. Finally, we ensure the submission is always valid by guaranteeing RLE encoding receives a clean 2D binary mask per test image and that `submission.csv` is written with the required columns.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

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
    rng = np.random.RandomState(1234)
    n = X1.shape[0]
    while True:
        idx = rng.choice(n, size=Batch_size, replace=(Batch_size > n))
        X1b = X1[idx].copy()
        X2b = X2[idx].copy()
        yb = y[idx].copy()

        for i in range(X1b.shape[0]):
            params = gen.get_random_transform(img_shape=X1b[i].shape, seed=None)
            X1b[i] = gen.apply_transform(X1b[i], params)
            yb[i] = gen.apply_transform(yb[i], params)

            params_depth = dict(params)
            for k in ["tx", "ty", "shear", "zx", "zy"]:
                if k in params_depth:
                    params_depth[k] = 0
            X2b[i] = gen.apply_transform(X2b[i], params_depth)

        yield (X1b, X2b), yb


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
        y_true = tf.cast(y_true > 0.5, tf.float32)
        y_pred = tf.cast(y_pred > 0.5, tf.float32)

        axes = [1, 2, 3]
        intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
        union = tf.reduce_sum(tf.cast((y_true + y_pred) > 0.0, tf.float32), axis=axes)
        ious = (intersection + 1e-9) / (union + 1e-9)

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

steps_per_epoch = int(np.ceil(train_x.shape[0] / float(Batch_size)))

output_signature = (
    (
        tf.TensorSpec(shape=(None, Height, Width, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 50, 50, 1), dtype=tf.float32),
    ),
    tf.TensorSpec(shape=(None, Height, Width, 1), dtype=tf.float32),
)

train_ds = tf.data.Dataset.from_generator(
    lambda: gen_flow_for_two_inputs(train_x, depth_np, train_y),
    output_signature=output_signature,
)

history = model.fit(
    train_ds,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    callbacks=[early_stopping],
    verbose=2,
)

model.save("model_7.h5")



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
df_test["depth"] = df_test["depth"].astype(np.float32).fillna(0.0)

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
y_pred_0 = model.predict([test_x, depth_test_np], batch_size=Batch_size, verbose=1)

test_x_h = test_x[:, :, ::-1, :]
y_pred_h = model.predict([test_x_h, depth_test_np], batch_size=Batch_size, verbose=1)
y_pred_h = y_pred_h[:, :, ::-1, :]  # unflip

y_pred = 0.5 * (y_pred_0 + y_pred_h)
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
