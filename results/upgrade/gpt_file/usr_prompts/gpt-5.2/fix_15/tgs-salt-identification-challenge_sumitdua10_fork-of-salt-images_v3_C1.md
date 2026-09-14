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

0.63693

# 6. Current score

0.5221

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5221) has done: 'I fix the shape/reshape bugs by building `train_x/train_y/test_x/depth_np/depth_test_np` from the actual dataframe lengths instead of hard-coded (4000/18000), and by stacking arrays correctly without flattening the dataset dimension. I fix the Keras import/runtime issue by using `tf.keras` (compatible with TF 2.18 + protobuf), while keeping the exact same model layers/architecture and training loop semantics. I fix pandas indexing bugs (`Series[10]` vs `.iloc[10]`) and ensure the test set is read from the sample submission IDs (so it’s never empty). Finally, I produce a valid `submission.csv` with columns `id,rle_mask` and correct 1-indexed RLE encoding.'
- What this solution (achieved 0.5221) has done: 'I fix the runtime crash caused by the protobuf 6.x incompatibility with TF’s default compiled protobuf by forcing the pure-Python protobuf implementation *before* importing TensorFlow, which resolves the `MessageFactory.GetPrototype` error without changing model logic. I also make the input directory selection robust to both `/kaggle/input/...` and `../input` layouts so the notebook reliably finds the dataset in your environment. Finally, I keep the same model/training loop, but adjust only the prediction binarization threshold to a more standard 0.5 (instead of `round(x-0.25)` which effectively thresholds at 0.75 and tends to under-segment), which should increase IoU/AP toward your target score while preserving the overall approach. The code still write a valid `submission.csv` with `id,rle_mask`.'
- What this solution (achieved 0.5221) has done: 'I fix the TensorFlow/protobuf crash by ensuring the environment variables that force pure-Python protobuf are set before any TensorFlow import (currently they’re set too late). I also keep the same model/training loop and data pipeline, only moving the TF import and seed setup into an earlier cell so execution proceeds to training and inference. Finally, I add a small safety fallback so if TF import still fails due to protobuf, the notebook prints a clear error instead of silently stopping, and it still write a valid `submission.csv` when training/prediction succeed.'
- What this solution (achieved 0.5221) has done: 'I fix the TensorFlow/protobuf crash by moving the protobuf environment-variable settings into the very first cell (before any TensorFlow import can happen) and forcing a clean TensorFlow import path. I keep the same model, data pipeline, training loop, and thresholding, only making import-order and stability fixes so the notebook runs end-to-end. I also make the input-path detection robust for your provided `/kaggle/data/...` layout (currently it only checks `/kaggle/input/...` and `../input/...`). Finally, I ensure a valid `submission.csv` is always written with the required `id,rle_mask` columns.'
- What this solution (achieved 0.5221) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation before any TensorFlow import and by importing `tf_keras` (the installed, TF-compatible Keras) instead of `tensorflow.keras`, which avoids the `MessageFactory.GetPrototype` error in this environment. I keep the model architecture, data pipeline, training loop, and inference logic intact, only adjusting the import plumbing so everything runs end-to-end. I also add a small, safe fallback path for Keras imports so the script doesn’t crash if one backend isn’t available. The submission writing remains unchanged and still produce a valid `submission.csv` with `id,rle_mask`.'
- What this solution (achieved 0.5221) has done: 'I fix the TensorFlow/protobuf crash by moving the protobuf environment-variable settings into the very first cell *before any TensorFlow-related import can occur* (the current failure indicates TF is still loading the compiled protobuf path). I also make sure no Keras/TensorFlow modules are imported before that cell executes, and keep the rest of the pipeline (data loading, model architecture, training loop, thresholding at 0.5, and RLE encoding) unchanged. Finally, I keep the same output schema and ensure `submission.csv` is always written with `id,rle_mask` and the test IDs aligned to `sample_submission.csv`, which is score-neutral but prevents invalid submissions.'
- What this solution (achieved 0.5221) has done: 'I fix the TensorFlow/protobuf import crash by ensuring the protobuf “python” implementation is forced before *any* TensorFlow/Keras-related import occurs and by also setting `TF_USE_LEGACY_KERAS=1` to avoid Keras 3 / TF integration pitfalls in this environment. I keep your model architecture, data pipeline, training loop, and 0.5 thresholding unchanged, only adjusting import plumbing and adding a safe fallback to `tf_keras` if needed. I also make the input directory resolution slightly more robust for both `/kaggle/data/...` and `/kaggle/input/...` layouts without changing which files are used. The script still run end-to-end and write a valid `submission.csv` with the required `id,rle_mask` columns.'
- What this solution (achieved 0.5221) has done: 'I fix the TensorFlow/protobuf crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation *and* `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before any TensorFlow-related import, and also explicitly importing `google.protobuf` once to ensure the correct implementation is loaded before TF initializes. I keep your model architecture/training loop unchanged, but adjust only the runtime plumbing around imports so the notebook runs end-to-end. To move the score up toward the target with minimal, metric-aligned change, I keep the 0.5 binarization threshold and additionally apply a tiny post-processing cleanup (remove extremely small predicted blobs) before RLE encoding, which tends to reduce false positives for this competition without changing the model. Finally, I ensure `submission.csv` is always written with the required `id,rle_mask` columns aligned to `sample_submission.csv`.'
- What this solution (achieved 0.5221) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation earlier and pinning a compatible protobuf runtime (>=3.20,<5) before importing TensorFlow, which resolves the `MessageFactory.GetPrototype` error in TF 2.18 environments. I keep your model architecture, training loop, and post-processing logic intact, only adjusting the import/bootstrap order and adding a safe fallback if pip install is not needed. I also make path resolution slightly more robust for the provided `/kaggle/data/...` layout without changing which dataset files are used. These changes are runtime/stability fixes and should allow the existing thresholding + small-component removal to run end-to-end and produce a valid `submission.csv`.'
- What this solution (achieved 0.5221) has done: 'You’re currently below the target (0.5221 vs 0.63693), so we should make a small, metric-aligned improvement without changing the model/training core. The biggest low-risk gain here is calibrating the binarization threshold using a small validation split from the existing training data, because mAP@IoU is very sensitive to under/over-segmentation and a fixed 0.5 often isn’t optimal for this architecture. I keep the same data pipeline, model, loss, and fit call semantics, but add a tiny held-out split and pick the threshold that maximizes the competition’s IoU-sweep AP on that split, then use that threshold at test-time. I also fix an off-by-one in RLE encoding at the last pixel (currently it can emit start index 10200 for a 101x101 image), which can silently hurt score due to invalid/shifted masks.'
- What this solution (achieved 0.5221) has done: 'We’re currently below the target (0.5221 vs 0.63693), so the smallest score-aligned change is to make the validation threshold selection match the competition metric more faithfully. Your current `ap_iou_sweep` incorrectly treats the task as a single IoU hit/miss (no FP/FN accounting and no “both empty” handling per Kaggle’s mAP), which can select a suboptimal threshold and depress the leaderboard score. I replace only the threshold-selection metric with the standard Salt “mean precision at IoU thresholds” computation (TP/FP/FN for the single-object mask case), keeping the model/training and post-processing the same. Everything else (data loading, architecture, early stopping, prediction, RLE submission writing) stays intact and still produces `submission.csv`.'
- What this solution (achieved 0.5221) has done: 'We’re below the target (0.5221 vs 0.63693), so the smallest likely gain without changing your model/training core is to fix two metric-aligned issues in post-processing and threshold selection. First, your RLE encoder currently flattens with `x[i].T.flatten()` which swaps row/column order and can significantly hurt leaderboard score even if masks look reasonable; we encode in Kaggle’s required column-major order directly (no transpose). Second, we select the binarization threshold using out-of-fold predictions (using the same model and fit, but predicting on a held-out split trained without those samples), which makes the chosen threshold far less biased than evaluating on training-trained samples; this typically improves mAP for this competition with minimal risk. Everything else (data loading, model architecture, compile/fit call, small-component removal, submission schema/paths) stays the same and still writes `submission.csv`.'
- What this solution (achieved 0.5221) has done: 'You’re below the target (0.5221 vs 0.63693), so we should make the smallest metric-aligned changes that can legitimately lift mAP without changing the model or training loop. The biggest likely hidden issue is that your current depth merge for test uses an unnormalized `df_depths` (because it was loaded before normalization), causing a train/test feature scale mismatch that depresses predictions; I normalize depths consistently and use that same normalized table for both train and test merges. Next, your RLE encoder should strictly emit column-major order without any reshape ambiguity; I replace `post_process` with a standard, proven Kaggle RLE that avoids edge-case off-by-one and ensures sorted runs. Everything else (data loading, augmentation, architecture, compile/fit, threshold selection, small-component removal) stays the same.'
- What this solution (achieved 0.5221) has done: 'We’re below the target (0.5221 vs 0.63693), so the smallest score-relevant lift is to reduce systematic submission/feature mismatches rather than changing the model. I (1) ensure test depths are normalized the same way as training depths (right now test uses raw `z`, which creates a train/test scale mismatch for the depth branch), and (2) fix the horizontal flip augmentation axis: you currently flip axis=2 (height) but comment says left-right; this inconsistency weakens augmentation and can reduce generalization. Everything else (model architecture, loss, fit procedure, threshold selection logic, and RLE schema) stays the same, and it still write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["PYTHONHASHSEED"] = "0"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

CANDIDATES = [
    "/kaggle/data/tgs-salt-identification-challenge",
    "/kaggle/data",
    "/kaggle/input/tgs-salt-identification-challenge",
    "/kaggle/input",
    "../input/tgs-salt-identification-challenge",
    "../input",
]
INPUT_DIR = None
for c in CANDIDATES:
    if os.path.isdir(c) and (
        os.path.exists(os.path.join(c, "train.csv"))
        or os.path.exists(
            os.path.join(c, "tgs-salt-identification-challenge", "train.csv")
        )
    ):
        INPUT_DIR = c
        break

if INPUT_DIR is None:
    INPUT_DIR = "../input"

if os.path.exists(
    os.path.join(INPUT_DIR, "tgs-salt-identification-challenge", "train.csv")
):
    INPUT_DIR = os.path.join(INPUT_DIR, "tgs-salt-identification-challenge")

print("Using INPUT_DIR:", INPUT_DIR)
print("INPUT_DIR contents (first 20):", sorted(os.listdir(INPUT_DIR))[:20])



## === cell 1
import sys
import random as rn
import numpy as np
import subprocess

np.random.seed(7)
rn.seed(12345)


def _ensure_compatible_protobuf():
    try:
        import google.protobuf
        from google.protobuf import __version__ as pv

        print("protobuf version (pre-check):", pv)
        major = int(pv.split(".")[0])
        if major >= 5:
            print("Attempting to install compatible protobuf (<5) to avoid TF crash...")
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-q",
                    "--no-deps",
                    "protobuf>=3.20.3,<5",
                ]
            )
            import importlib

            importlib.invalidate_caches()
            print("protobuf install attempted.")
    except Exception as e:
        print("protobuf compatibility step skipped/failed (continuing):", repr(e))


_ensure_compatible_protobuf()

import google.protobuf
from google.protobuf import __version__ as protobuf_version

print("protobuf version:", protobuf_version)

import tensorflow as tf

tf.random.set_seed(12345)

try:
    import tf_keras as keras  # TF-compatible Keras
except Exception:
    from tensorflow import keras

layers = keras.layers
Model = keras.models.Model

print("TF version:", tf.__version__)
print("Keras module:", keras.__name__)



## === cell 2
import pandas as pd
from PIL import Image

df_train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
print(df_train.head())
print("\nTrain files shape is", df_train.shape)

df_depths = pd.read_csv(os.path.join(INPUT_DIR, "depths.csv"))
_depth_max = float(df_depths["z"].max())
if _depth_max != 0.0:
    df_depths["z"] = df_depths["z"] / _depth_max
print(df_depths.head())
print("\nDepths files shape is", df_depths.shape)

df_train = df_train.merge(df_depths, on="id", how="inner")
df_train = df_train.rename(columns={"z": "depth"})
print(df_train.head())
print("\nMerged train shape is", df_train.shape)



## === cell 3
train_img_dir = os.path.join(INPUT_DIR, "train", "images")
train_mask_dir = os.path.join(INPUT_DIR, "train", "masks")

df_train["images"] = [
    np.array(Image.open(os.path.join(train_img_dir, f"{idx}.png")))
    for idx in df_train["id"]
]
print("Sample Image Shape is", df_train["images"].iloc[0].shape)


def to_single_channel(x):
    x = np.asarray(x)
    if x.ndim == 2:
        x = x[..., None]
    if x.shape[-1] != 1:
        x = x[..., :1]
    return x


df_train["images"] = df_train["images"].map(to_single_channel)
print("After optimization, Image Shape is", df_train["images"].iloc[0].shape)
print("No. of train images are", len(df_train))

df_train["images"] = df_train["images"].map(lambda x: x.astype(np.float32) / 255.0)
print("After normalization, pixel value is", df_train["images"].iloc[0][1, 0, 0])

df_train["masks"] = [
    np.array(Image.open(os.path.join(train_mask_dir, f"{idx}.png")))
    for idx in df_train["id"]
]
print("\nSample Mask Shape is", df_train["masks"].iloc[0].shape)
print("No. of mask images are", len(df_train))
print("Before normalization, pixel value is", df_train["masks"].iloc[15][10, 0])

df_train["masks"] = df_train["masks"].map(to_single_channel)
df_train["masks"] = df_train["masks"].map(lambda x: (x.astype(np.float32) / 255.0))
df_train["masks"] = df_train["masks"].map(lambda x: np.round(x).astype(np.int32))

print("train df columns are", df_train.columns.tolist())



## === cell 4
train_x = np.stack(df_train["images"].to_list(), axis=0).astype(np.float32)
train_y = np.stack(df_train["masks"].to_list(), axis=0).astype(np.int32)

print("Train Shape =", train_x.shape)
print("Mask Shape =", train_y.shape)
print("Sample Pixel Value (x)", train_x[0, 1, 0, 0])
print("Sample Pixel Value (y)", train_y[15, 10, 0, 0])



## === cell 5
train_x_hflip = np.flip(
    train_x, axis=1
)  # flip left-right (width axis for (H,W,1) is axis=1)
train_y_hflip = np.flip(train_y, axis=1)

train_x = np.concatenate((train_x, train_x_hflip), axis=0)
train_y = np.concatenate((train_y, train_y_hflip), axis=0)

print("Augmented train_x:", train_x.shape)
print("Augmented train_y:", train_y.shape)



## === cell 6
Height = 101
Width = 101

df_train["depth_image"] = df_train["depth"].map(
    lambda x: np.full(shape=(50, 50, 1), fill_value=x, dtype=np.float32)
)

depth_np = np.stack(df_train["depth_image"].to_list(), axis=0).astype(np.float32)
depth_np = np.concatenate(
    (depth_np, depth_np), axis=0
)  # duplicate to match augmentation

print("depth_np shape:", depth_np.shape)
print("train_x shape:", train_x.shape, "train_y shape:", train_y.shape)

assert (
    train_x.shape[0] == train_y.shape[0] == depth_np.shape[0]
), "Mismatch in augmented training lengths."



## === cell 7
img_input = layers.Input(shape=(Height, Width, 1), name="img_input")
depth_input = layers.Input(shape=(50, 50, 1), name="depth_input")
depth_input_new = layers.Input(shape=(1, 1, 1), name="depth_input_new")  # unused, kept

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



## === cell 8
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



## === cell 9
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

EarlyStopping = keras.callbacks.EarlyStopping
early_stopping = EarlyStopping(
    monitor="accuracy", patience=3, restore_best_weights=True
)

model.fit(
    [train_x, depth_np],
    train_y,
    epochs=50,
    verbose=2,
    batch_size=96,
    callbacks=[early_stopping],
)

model.save("model_7.h5")



## === cell 10
import scipy.ndimage as ndi


def remove_small_components(batch_mask, min_size=20):
    out = np.zeros_like(batch_mask, dtype=np.int32)
    for i in range(batch_mask.shape[0]):
        m = batch_mask[i, :, :, 0].astype(bool)
        labeled, n = ndi.label(m)
        if n == 0:
            continue
        counts = np.bincount(labeled.ravel())
        keep = np.zeros_like(counts, dtype=bool)
        keep[1:] = counts[1:] >= min_size
        out[i, :, :, 0] = keep[labeled].astype(np.int32)
    return out


def _iou_single_mask(y_true_2d, y_pred_2d):
    y_true_2d = y_true_2d.astype(bool)
    y_pred_2d = y_pred_2d.astype(bool)
    inter = np.logical_and(y_true_2d, y_pred_2d).sum()
    union = np.logical_or(y_true_2d, y_pred_2d).sum()
    if union == 0:
        return 0.0
    return inter / union


def salt_mean_precision(y_true, y_pred_bin):
    thresholds = np.arange(0.5, 1.0, 0.05)
    precisions = []
    for i in range(y_true.shape[0]):
        t = y_true[i, :, :, 0].astype(np.int32)
        p = y_pred_bin[i, :, :, 0].astype(np.int32)

        t_empty = t.sum() == 0
        p_empty = p.sum() == 0

        if t_empty and p_empty:
            precisions.append(1.0)
            continue
        if t_empty != p_empty:
            precisions.append(0.0)
            continue

        iou = _iou_single_mask(t, p)
        precisions.append(float(np.mean([(iou > thr) for thr in thresholds])))
    return float(np.mean(precisions))


rng = np.random.RandomState(12345)
n = train_x.shape[0]
idx = np.arange(n)
rng.shuffle(idx)
val_n = max(1, int(0.1 * n))
val_idx = idx[:val_n]
tr_idx = idx[val_n:]

model_thr = keras.models.clone_model(model)
model_thr.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

early_stopping_thr = EarlyStopping(
    monitor="accuracy", patience=3, restore_best_weights=True
)

model_thr.fit(
    [train_x[tr_idx], depth_np[tr_idx]],
    train_y[tr_idx],
    epochs=50,
    verbose=0,
    batch_size=96,
    callbacks=[early_stopping_thr],
)

y_val_pred = model_thr.predict(
    [train_x[val_idx], depth_np[val_idx]], batch_size=96, verbose=0
)

cand_thresholds = [0.35, 0.40, 0.45, 0.50, 0.55]
best_t = 0.50
best_ap = -1.0
for t in cand_thresholds:
    yb = (y_val_pred >= t).astype(np.int32)
    yb = remove_small_components(yb, min_size=20)
    ap = salt_mean_precision(train_y[val_idx], yb)
    print(f"Val mAP@IoU(thresholds) at threshold {t:.2f}: {ap:.5f}")
    if ap > best_ap:
        best_ap = ap
        best_t = t

print("Selected threshold:", best_t, "with val mAP:", best_ap)



## === cell 11
df_sub = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))

df_test = df_sub.merge(df_depths, on="id", how="left")

print("Length of df_test is", len(df_test))
print(df_test.head())



## === cell 12
test_img_dir = os.path.join(INPUT_DIR, "test", "images")
df_test["images"] = [
    np.array(Image.open(os.path.join(test_img_dir, f"{idx}.png")))
    for idx in df_test["id"]
]



## === cell 13
print("Sample Image Shape is", df_test["images"].iloc[0].shape)



## === cell 14
df_test["images"] = df_test["images"].map(to_single_channel)
print("After optimization, Image Shape is", df_test["images"].iloc[0].shape)
print("No. of test images are", len(df_test))

df_test["images"] = df_test["images"].map(lambda x: x.astype(np.float32) / 255.0)

test_x = np.stack(df_test["images"].to_list(), axis=0).astype(np.float32)
print("test_x shape:", test_x.shape)
print("test_x max/min:", test_x.max(), test_x.min())
print("train_x max/min:", train_x.max(), train_x.min())



## === cell 15
df_test["depth_image"] = df_test["z"].map(
    lambda x: np.full(shape=(50, 50, 1), fill_value=x, dtype=np.float32)
)
depth_test_np = np.stack(df_test["depth_image"].to_list(), axis=0).astype(np.float32)

print(df_test.columns.tolist())
print("depth_test_np shape:", depth_test_np.shape)
print(
    "depth_test_np min/max:",
    float(np.nanmin(depth_test_np)),
    float(np.nanmax(depth_test_np)),
)
print("depth_np min/max:", float(np.min(depth_np)), float(np.max(depth_np)))




## === cell 16
def rle_encode(mask_2d):
    """
    mask_2d: (H, W) binary {0,1}. Kaggle expects column-major (top-to-bottom, then left-to-right).
    """
    pixels = mask_2d.T.flatten()  # column-major as required
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def post_process(batch_mask):
    out = []
    for i in range(batch_mask.shape[0]):
        m = batch_mask[i, :, :, 0].astype(np.uint8)
        out.append(rle_encode(m) if m.sum() > 0 else "")
    return out




## === cell 17
print("Predicting now.....")
y_pred = model.predict([test_x, depth_test_np], batch_size=96, verbose=1)
print("Prediction Complete. y_pred shape:", y_pred.shape)



## === cell 18
y_pred_bin = (y_pred >= best_t).astype(np.int32)
y_pred_bin = remove_small_components(y_pred_bin, min_size=20)

print("Y Prediction shape =", y_pred_bin.shape)
print("min/max:", y_pred_bin.min(), y_pred_bin.max())

new_y_pred = post_process(y_pred_bin)
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
