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

0.5221

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5221) has done: 'I fix the notebook-to-script issues (the `%matplotlib inline` magic and deprecated/removed Keras imports) by switching to `tf.keras` equivalents that work in the Kaggle TF 2.18 + Keras 3 environment. I also fix the depth feature lookup for test images (test ids are not in `depths.csv`, so we must join depth by using `train.csv` as the mapping source) to remove the `KeyError`. Finally, I keep the same U-Net-like model and training loop, ensure `ModelCheckpoint` writes weights correctly, and generate a correctly formatted `submission.csv` with `id,rle_mask` for all test images.'
- What this solution (achieved 0.5221) has done: 'We fix the immediate runtime crash caused by an incompatibility between TensorFlow 2.18 and the installed protobuf 6.x by pinning protobuf to a compatible 3.20.x version at runtime before importing TensorFlow. Then we keep the model/training logic intact, but correct the submission-side postprocessing to better match the competition metric by using a slightly higher mask threshold and removing tiny predicted components (this is minimal, inference-only calibration that typically improves MAP@IoU for this task). We also make the run-length encoding robust for empty masks and ensure IDs align exactly with `sample_submission.csv` ordering. Finally, we ensure the script always writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.5221) has done: 'We keep your U-Net model and training loop identical, but fix two inference-time issues that typically suppress MAP@IoU for TGS: (1) use the **true per-image depth feature for test** by reading the competition’s `depths.csv` (it contains both train+test), and (2) tune **mask binarization/postprocessing** in a minimal way to better match the metric (choose threshold from a small set using your existing validation split, and apply a light morphological close before small-component filtering). These are small, semantics-preserving changes (no architecture/loss/training rewrite) and should move the score upward toward your 0.68137 target. The submission writing and ordering remain aligned to `sample_submission.csv`, producing a valid `submission.csv`.'
- What this solution (achieved 0.5221) has done: 'Your current gap to the target is large (0.5221 → 0.68137), so the most “minimal but high-leverage” improvement is to keep the same model/training, but make the inference post-processing better match the competition’s object-level mAP@IoU. I (1) replace the current proxy threshold selection (which incorrectly treats per-image IoU as “precision across thresholds”) with a proper Kaggle-style mAP computation on your existing validation split, and pick the best threshold from a small grid; (2) tune the small-component filtering by selecting `min_area` jointly with threshold on validation (still tiny search, inference-only); and (3) ensure the same postprocess is applied consistently to validation and test before RLE. This keeps architecture/loss/training unchanged and typically lifts scores for this competition without adding approximations or changing convergence.'

# 9. Code solution

## === cell 0
import os
import sys
import random
import warnings
import subprocess

try:
    import google.protobuf  # noqa: F401
    import importlib
    import pkg_resources

    pb_ver = pkg_resources.get_distribution("protobuf").version
    if pb_ver.startswith(("6.", "5.", "4.")):
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==3.20.3"]
        )
        importlib.invalidate_caches()
        if "google.protobuf" in sys.modules:
            del sys.modules["google.protobuf"]
except Exception as e:
    print("Warning: protobuf pin attempt failed:", repr(e))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from tqdm import tqdm
from skimage.transform import resize
from sklearn.model_selection import train_test_split

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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import img_to_array, load_img

print("TF version:", tf.__version__)



## === cell 1
im_width = 128
im_height = 128
border = 5
im_chan = 2  # Number of channels: first is original and second cumsum(axis=0)
n_features = 1  # Number of extra features, like depth

path_train = "../input/train/"
path_test = "../input/test/"

if not os.path.exists(os.path.join(path_train, "images")):
    path_train = "../input/tgs-salt-identification-challenge/train/"
if not os.path.exists(os.path.join(path_test, "images")):
    path_test = "../input/tgs-salt-identification-challenge/test/"

print("Using path_train:", path_train)
print("Using path_test :", path_test)



## === cell 2
depths_path = "../input/depths.csv"
train_csv_path = "../input/train.csv"
if not os.path.exists(depths_path):
    depths_path = "../input/tgs-salt-identification-challenge/depths.csv"
if not os.path.exists(train_csv_path):
    train_csv_path = "../input/tgs-salt-identification-challenge/train.csv"

df_depths = pd.read_csv(depths_path)  # columns: id,z (train+test ids)
df_train = pd.read_csv(train_csv_path)  # columns: id,rle_mask (train ids)

depth_map_all = df_depths.set_index("id")["z"].to_dict()

missing_train_depths = df_train["id"].map(depth_map_all).isna().sum()
print("Missing depths for train ids (should be 0):", int(missing_train_depths))

_ = pd.Series(df_train["id"].map(depth_map_all).dropna().values).hist(bins=30)
plt.title("Depth (z) distribution (train)")
plt.show()



## === cell 3
ids = ["1f1cc6b3a4", "5b7c160d0d", "6c40978ddf", "7dfdf6eeb8", "7e5a6e5013"]
plt.figure(figsize=(30, 10))
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
    if j == 0:
        plt.ylabel("image")

    plt.subplot(3, len(ids), len(ids) + j + 1)
    plt.imshow(img_cumsum, cmap="seismic")
    plt.axis("off")
    if j == 0:
        plt.ylabel("cumsum")

    plt.subplot(3, len(ids), 2 * len(ids) + j + 1)
    plt.imshow(img_mask, cmap="gray")
    plt.axis("off")
    if j == 0:
        plt.ylabel("mask")
plt.tight_layout()
plt.show()



## === cell 4
train_ids = next(os.walk(os.path.join(path_train, "images")))[2]
test_ids = next(os.walk(os.path.join(path_test, "images")))[2]

train_ids = sorted(train_ids)
test_ids = sorted(test_ids)

print("Train images:", len(train_ids))
print("Test images :", len(test_ids))



## === cell 5
X = np.zeros((len(train_ids), im_height, im_width, im_chan), dtype=np.float32)
y = np.zeros((len(train_ids), im_height, im_width, 1), dtype=np.float32)
X_feat = np.zeros((len(train_ids), n_features), dtype=np.float32)

print("Getting and resizing train images and masks ...")
sys.stdout.flush()

train_depth_values = []
for n, fn in tqdm(list(enumerate(train_ids)), total=len(train_ids)):
    id_ = fn.replace(".png", "")
    z = depth_map_all.get(id_, np.nan)
    train_depth_values.append(z)
default_depth = float(np.nanmean(train_depth_values))

for n, fn in tqdm(list(enumerate(train_ids)), total=len(train_ids)):
    id_ = fn.replace(".png", "")
    z = depth_map_all.get(id_, np.nan)
    if np.isnan(z):
        z = default_depth
    X_feat[n, 0] = z

    img = load_img(os.path.join(path_train, "images", fn), color_mode="grayscale")
    x_img = img_to_array(img)  # (H,W,1)
    x_img = resize(
        x_img,
        (im_height, im_width, 1),
        mode="constant",
        preserve_range=True,
        anti_aliasing=False,
    )

    x_center_mean = x_img[border:-border, border:-border].mean()
    x_csum = (np.float32(x_img) - x_center_mean).cumsum(axis=0)
    x_csum -= x_csum[border:-border, border:-border].mean()
    x_csum /= max(1e-3, x_csum[border:-border, border:-border].std())

    mask = img_to_array(
        load_img(os.path.join(path_train, "masks", fn), color_mode="grayscale")
    )
    mask = resize(
        mask,
        (im_height, im_width, 1),
        mode="constant",
        preserve_range=True,
        anti_aliasing=False,
    )

    X[n, ..., 0] = x_img.squeeze() / 255.0
    X[n, ..., 1] = x_csum.squeeze()
    y[n] = mask / 255.0

print("Done!")



## === cell 6
X_train, X_valid, X_feat_train, X_feat_valid, y_train, y_valid = train_test_split(
    X, X_feat, y, test_size=0.15, random_state=SEED
)
print(X_train.shape, X_valid.shape, X_feat_train.shape, y_train.shape)



## === cell 7
x_feat_mean = X_feat_train.mean(axis=0, keepdims=True)
x_feat_std = X_feat_train.std(axis=0, keepdims=True) + 1e-7

X_feat_train = (X_feat_train - x_feat_mean) / x_feat_std
X_feat_valid = (X_feat_valid - x_feat_mean) / x_feat_std



## === cell 8
ix = random.randint(0, len(X_train) - 1)
has_mask = y_train[ix].max() > 0

fig, ax = plt.subplots(1, 3, figsize=(18, 6))
ax[0].imshow(X_train[ix, ..., 0], cmap="seismic", interpolation="bilinear")
if has_mask:
    ax[0].contour(y_train[ix].squeeze(), colors="k", levels=[0.5])
ax[0].set_title("Seismic")

ax[1].imshow(X_train[ix, ..., 1], cmap="seismic", interpolation="bilinear")
if has_mask:
    ax[1].contour(y_train[ix].squeeze(), colors="k", levels=[0.5])
ax[1].set_title("Seismic cumsum")

ax[2].imshow(y_train[ix].squeeze(), interpolation="bilinear", cmap="gray")
ax[2].set_title("Salt mask")
plt.show()




## === cell 9
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.cast(y_pred > t, tf.int32)
        score = tf.numpy_function(
            lambda yt, yp: np.mean((yt == yp).astype(np.float32)),
            [y_true, y_pred_],
            tf.float32,
        )
        prec.append(score)
    return tf.reduce_mean(tf.stack(prec), axis=0)




## === cell 10
input_img = Input((im_height, im_width, im_chan), name="img")
input_features = Input((n_features,), name="feat")

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
p4 = MaxPooling2D(pool_size=(2, 2))(c4)

f_repeat = RepeatVector(8 * 8)(input_features)
f_conv = Reshape((8, 8, n_features))(f_repeat)
p4_feat = concatenate([p4, f_conv], axis=-1)

c5 = Conv2D(128, (3, 3), activation="relu", padding="same")(p4_feat)
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

model = Model(inputs=[input_img, input_features], outputs=[outputs])
model.compile(optimizer="adam", loss="binary_crossentropy")
model.summary()



## === cell 11
checkpoint_path = "model-tgs-salt-1.weights.h5"
callbacks = [
    EarlyStopping(patience=5, verbose=1, restore_best_weights=False),
    ReduceLROnPlateau(patience=3, verbose=1),
    ModelCheckpoint(
        checkpoint_path, verbose=1, save_best_only=True, save_weights_only=True
    ),
]

results = model.fit(
    {"img": X_train, "feat": X_feat_train},
    y_train,
    batch_size=16,
    epochs=50,
    callbacks=callbacks,
    validation_data=({"img": X_valid, "feat": X_feat_valid}, y_valid),
    verbose=2,
)



## === cell 12
X_test = np.zeros((len(test_ids), im_height, im_width, im_chan), dtype=np.float32)
X_feat_test = np.zeros((len(test_ids), n_features), dtype=np.float32)
sizes_test = []

test_depths = []
for fn in test_ids:
    id_ = fn.replace(".png", "")
    test_depths.append(depth_map_all.get(id_, np.nan))
default_test_depth = (
    float(np.nanmean(np.array(test_depths, dtype=np.float32)))
    if np.any(~np.isnan(test_depths))
    else default_depth
)

print("Getting and resizing test images ...")
sys.stdout.flush()
for n, fn in tqdm(list(enumerate(test_ids)), total=len(test_ids)):
    id_ = fn.replace(".png", "")
    z = depth_map_all.get(id_, np.nan)
    if np.isnan(z):
        z = default_test_depth
    X_feat_test[n, 0] = z

    img = load_img(os.path.join(path_test, "images", fn), color_mode="grayscale")
    x = img_to_array(img)
    sizes_test.append([x.shape[0], x.shape[1]])
    x = resize(
        x,
        (im_height, im_width, 1),
        mode="constant",
        preserve_range=True,
        anti_aliasing=False,
    )

    x_center_mean = x[border:-border, border:-border].mean()
    x_csum = (np.float32(x) - x_center_mean).cumsum(axis=0)
    x_csum -= x_csum[border:-border, border:-border].mean()
    x_csum /= max(1e-3, x_csum[border:-border, border:-border].std())

    X_test[n, ..., 0] = x.squeeze() / 255.0
    X_test[n, ..., 1] = x_csum.squeeze()

print("Done!")



## === cell 13
X_feat_test = (X_feat_test - x_feat_mean) / x_feat_std



## === cell 14
if os.path.exists(checkpoint_path):
    model.load_weights(checkpoint_path)
    print("Loaded best weights from:", checkpoint_path)
else:
    print("Checkpoint not found; using current model weights.")



## === cell 15
val_loss = model.evaluate({"img": X_valid, "feat": X_feat_valid}, y_valid, verbose=0)
print("Validation loss:", float(val_loss))



## === cell 16
preds_train = model.predict({"img": X_train, "feat": X_feat_train}, verbose=0)
preds_val = model.predict({"img": X_valid, "feat": X_feat_valid}, verbose=0)
preds_test = model.predict({"img": X_test, "feat": X_feat_test}, verbose=0)

print("Pred shapes:", preds_test.shape)



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
print("Upsampled[0] shape:", preds_test_upsampled[0].shape)



## === cell 18
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plt.imshow(
    np.squeeze(
        img_to_array(
            load_img(
                os.path.join(path_test, "images", test_ids[0]), color_mode="grayscale"
            )
        )
    ),
    cmap="seismic",
)
plt.title("Test image")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(preds_test_upsampled[0], cmap="gray", vmin=0, vmax=1)
plt.title("Pred mask (prob)")
plt.axis("off")
plt.show()



## === cell 19
ix = random.randint(0, len(X_train) - 1)
has_mask = y_train[ix].max() > 0

fig, ax = plt.subplots(1, 4, figsize=(20, 8))
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

ax[3].imshow(preds_train[ix].squeeze(), vmin=0, vmax=1, cmap="gray")
if has_mask:
    ax[3].contour(y_train[ix].squeeze(), colors="r", levels=[0.5])
ax[3].set_title("Salt Pred (prob)")
plt.show()




## === cell 20
def mean_iou_np(labels, predictions, n_classes):
    miou = 0.0
    seen_classes = n_classes
    for c in range(n_classes):
        labels_c = labels == c
        pred_c = predictions == c
        labels_c_sum = labels_c.sum()
        pred_c_sum = pred_c.sum()

        if (labels_c_sum == 0) and (pred_c_sum == 0):
            seen_classes -= 1
            continue

        intersect = np.logical_and(labels_c, pred_c).sum()
        union = labels_c_sum + pred_c_sum - intersect
        if union > 1e-12:
            miou += intersect / union

    return miou / max(seen_classes, 1)


preds_val_t_05 = (preds_val > 0.5).astype(np.uint8)
miou_val = mean_iou_np(
    y_valid.ravel().astype(np.uint8), preds_val_t_05.ravel().astype(np.uint8), 2
)
print("Mean IoU (threshold=0.5 binarized) on valid:", float(miou_val))



## === cell 21
import cv2


def RLenc(img, order="F", format=True):
    if img is None:
        return "" if format else []
    if img.max() == 0:
        return "" if format else []

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

    if format:
        return " ".join([f"{rr[0]} {rr[1]}" for rr in runs])
    else:
        return runs


def postprocess_mask(prob, thr=0.5, min_area=20, do_close=True):
    m = (prob > thr).astype(np.uint8)
    if m.max() == 0:
        return m
    if do_close:
        kernel = np.ones((3, 3), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m, dtype=np.uint8)
    for lab in range(1, num_labels):
        area = stats[lab, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == lab] = 1
    return out


def kaggle_map_iou(y_true_bin, y_pred_bin, thresholds=np.arange(0.5, 1.0, 0.05)):
    """
    y_true_bin, y_pred_bin: (H,W) uint8 {0,1} for single image.
    TGS has one foreground class; this reduces to intersection/union with empty-mask handling.
    """
    y_true_bin = (y_true_bin > 0).astype(np.uint8)
    y_pred_bin = (y_pred_bin > 0).astype(np.uint8)

    inter = np.logical_and(y_true_bin, y_pred_bin).sum()
    union = np.logical_or(y_true_bin, y_pred_bin).sum()
    iou = 1.0 if union == 0 else inter / union

    return float(np.mean([1.0 if iou > t else 0.0 for t in thresholds]))


def kaggle_map_iou_batch(y_true, y_pred_bin):
    scores = []
    for i in range(y_true.shape[0]):
        scores.append(
            kaggle_map_iou(y_true[i, ..., 0] > 0.5, y_pred_bin[i, ..., 0] > 0)
        )
    return float(np.mean(scores))


thr_grid = [0.40, 0.45, 0.50, 0.55, 0.60]
min_area_grid = [0, 10, 20, 30, 40]
do_close = True

best_thr = 0.5
best_min_area = 20
best_score = -1.0

for thr in thr_grid:
    for min_area in min_area_grid:
        pv = np.zeros(
            (len(preds_val), y_valid.shape[1], y_valid.shape[2], 1), dtype=np.uint8
        )
        for i in range(len(preds_val)):
            pv[i, ..., 0] = postprocess_mask(
                preds_val[i, ..., 0], thr=thr, min_area=min_area, do_close=do_close
            )
        score = kaggle_map_iou_batch(y_valid, pv)
        if score > best_score:
            best_score = score
            best_thr = thr
            best_min_area = min_area

print(
    f"Chosen postprocess on validation: thr={best_thr}, min_area={best_min_area}, mAP@IoU proxy={best_score:.4f}"
)

pred_dict = {}
for i, fn in enumerate(test_ids):
    id_ = fn[:-4]
    prob = preds_test_upsampled[i]
    pmask = postprocess_mask(
        prob, thr=best_thr, min_area=best_min_area, do_close=do_close
    )
    pred_dict[id_] = RLenc(pmask)



## === cell 22
sub = pd.DataFrame({"id": list(pred_dict.keys()), "rle_mask": list(pred_dict.values())})
sample_sub_path = "../input/sample_submission.csv"
if not os.path.exists(sample_sub_path):
    sample_sub_path = "../input/tgs-salt-identification-challenge/sample_submission.csv"
sample = pd.read_csv(sample_sub_path)

sub = sample[["id"]].merge(sub, on="id", how="left")
sub["rle_mask"] = sub["rle_mask"].fillna("")

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
