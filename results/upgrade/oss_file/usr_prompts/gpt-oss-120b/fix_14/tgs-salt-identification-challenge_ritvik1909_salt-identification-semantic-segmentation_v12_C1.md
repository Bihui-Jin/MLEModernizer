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

0.77948

# 6. Current score

0.1301

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0033) has done: 'Implemented missing imports, corrected mask path resolution, added fallback Otsu implementation using scikit‑image when opencv is unavailable, and ensured all progress‑bar utilities are available. Fixed the creation of training placeholders and streamlined test‑image loading and prediction loops. The script now runs end‑to‑end and writes a proper `submission.csv` with RLE masks.'
- What this solution (achieved 0.0012) has done: 'I fixed the import error for `binary_fill_holes` by adding a safe fallback, ensured the training and test image directories point to the correct `images` sub‑folders after unzipping, and kept all original logic intact. These changes unblock the script, let it load images/masks, run the Otsu‑based prediction, encode the masks, and finally write a proper `submission.csv` file.'
- What this solution (achieved 0.1072) has done: 'The changes keep the overall Otsu‑based pipeline but improve the mask post‑processing: instead of discarding all but the largest connected component, we now keep every component and simply fill holes, which preserves true salt regions that were previously removed. This minimal tweak is expected to raise the validation metric toward the target without altering the core model logic.'
- What this solution (achieved 0.1158) has done: 'The changes lower the global threshold used when converting the up‑sampled Otsu masks to binary (from 0.5 to 0.3) and add a light Gaussian blur before Otsu thresholding, which helps produce cleaner masks and typically raises the mean‑average‑precision toward the target value while keeping the original pipeline intact.'
- What this solution (achieved 0.0461) has done: 'The fix addresses the OpenCV threshold error caused by multi‑channel images and ensures the processing loop completes so that the RLE encoding step receives a mask for every test image. By converting any RGB(A) image to a single‑channel grayscale before applying Otsu, the script runs end‑to‑end and produces a valid `submission.csv`.'
- What this solution (achieved 0.1301) has done: 'I load the training images and masks, build a tiny convolutional network with tf.keras to learn a simple segmentation model, and use it to predict the test masks instead of the pure Otsu baseline.  The model is deliberately small (few Conv‑layers and up‑sampling) so it trains quickly while still providing a sizable lift in mean‑average‑precision.  After training I threshold the predictions (0.5) and resize them back to the original 101×101 size before run‑length encoding, ensuring a valid `submission.csv` is written.'
- What this solution (achieved 0.1301) has done: 'The fix adds a compatibility shim for the protobuf MessageFactory so TensorFlow can be imported without the `GetPrototype` error, and extends the training to 10 epochs (instead of 3) to give the tiny CNN a bit more learning capacity, nudging the validation score toward the target while keeping the original pipeline unchanged.'
- What this solution (achieved 0.1301) has done: 'I add a lightweight validation‑threshold search after training the tiny CNN so the model uses the threshold that gives the highest pixel‑wise IoU on the held‑out validation set, and I increase the number of training epochs slightly (to 20) to let the model learn a bit better. These changes keep the original architecture and workflow intact while nudging the validation metric upward, which should move the Kaggle score closer to the target.'

# 9. Code solution

## === cell 0
import os
import zipfile
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from skimage.io import imread
from skimage.transform import resize
from skimage import filters
from skimage.measure import label

try:
    from google.protobuf import message_factory

    if not hasattr(message_factory.MessageFactory, "GetPrototype"):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

try:
    from skimage.morphology import binary_fill_holes
except Exception:
    try:
        from scipy.ndimage import binary_fill_holes
    except Exception:

        def binary_fill_holes(mask):
            return mask


try:
    import cv2

    _has_cv2 = True
except Exception:
    _has_cv2 = False

try:
    import tensorflow as tf

    _has_tf = True
except Exception:
    _has_tf = False


class config:
    im_width = 128
    im_height = 128
    im_chan = 1
    path_train = "train/"
    path_test = "test/"




## === cell 1
def unzip_to(dest_path, zip_path):
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_path)


zip_train = "../input/tgs-salt-identification-challenge/train.zip"
zip_test = "../input/tgs-salt-identification-challenge/test.zip"
if not os.path.isfile(zip_train):
    zip_train = "train.zip"
if not os.path.isfile(zip_test):
    zip_test = "test.zip"

unzip_to("train", zip_train)
unzip_to("test", zip_test)

if os.path.isdir(os.path.join(config.path_train, "images")):
    config.path_train = os.path.join(config.path_train, "images")
else:
    for root, dirs, _ in os.walk(config.path_train):
        if "images" in dirs:
            config.path_train = os.path.join(root, "images")
            break

if os.path.isdir(os.path.join(config.path_test, "images")):
    config.path_test = os.path.join(config.path_test, "images")
else:
    for root, dirs, _ in os.walk(config.path_test):
        if "images" in dirs:
            config.path_test = os.path.join(root, "images")
            break


def collect_ids(img_dir_path):
    return sorted([f for f in os.listdir(img_dir_path) if f.lower().endswith(".png")])


train_ids = collect_ids(config.path_train)
test_ids = collect_ids(config.path_test)

print(f"Found {len(train_ids)} training images and {len(test_ids)} test images.")



## === cell 2
X = []
Y = []
mask_dir = os.path.join(os.path.dirname(config.path_train), "masks")
for img_name in tqdm(train_ids, desc="Loading train data"):
    img_path = os.path.join(config.path_train, img_name)
    mask_path = os.path.join(mask_dir, img_name)

    img = imread(img_path, as_gray=True)  # ensure grayscale
    mask = imread(mask_path, as_gray=True)

    img_resized = resize(
        img,
        (config.im_height, config.im_width),
        preserve_range=True,
        anti_aliasing=True,
    )
    mask_resized = resize(
        mask,
        (config.im_height, config.im_width),
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )

    img_resized = img_resized.astype(np.float32) / 255.0
    mask_bin = (mask_resized > 0.5).astype(np.float32)

    X.append(img_resized[..., np.newaxis])  # add channel dim
    Y.append(mask_bin[..., np.newaxis])

X = np.stack(X, axis=0)
Y = np.stack(Y, axis=0)

print("Training data loaded:")
print("X shape:", X.shape, "Y shape:", Y.shape)



## === cell 3
val_fraction = 0.1
split_idx = int((1 - val_fraction) * len(X))
X_train, X_val = X[:split_idx], X[split_idx:]
Y_train, Y_val = Y[:split_idx], Y[split_idx:]

print(f"Split -> X_train: {X_train.shape}, X_val: {X_val.shape}")



## === cell 4
if _has_tf:
    inputs = tf.keras.layers.Input(
        shape=(config.im_height, config.im_width, config.im_chan)
    )
    x = tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same")(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)

    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same")(x)
    outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid", padding="same")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy")
    print(model.summary())
else:
    model = None
    print("TensorFlow not available – will fall back to Otsu baseline.")



## === cell 5
if _has_tf:
    model.fit(
        X_train,
        Y_train,
        validation_data=(X_val, Y_val),
        epochs=20,  # increased from 10
        batch_size=32,
        verbose=1,
    )

    preds_val = model.predict(X_val, batch_size=32, verbose=0)

    def _pixel_iou(th):
        """Mean pixel‑wise IoU for a given threshold."""
        pred_bin = (preds_val > th).astype(np.float32)
        intersection = np.sum(pred_bin * Y_val, axis=(1, 2, 3))
        union = np.sum(np.clip(pred_bin + Y_val, 0, 1), axis=(1, 2, 3))
        return np.mean(intersection / (union + 1e-7))

    thresholds = np.arange(0.30, 0.71, 0.05)
    ious = [_pixel_iou(t) for t in thresholds]
    threshold_best = thresholds[np.argmax(ious)]
    print(f"Optimal threshold on validation set: {threshold_best:.3f}")
else:
    threshold_best = 0.3



## === cell 6
preds_test_upsampled = []
print("Loading and processing test images ...")
for img_id in tqdm(test_ids, desc="Processing test"):
    img_path = os.path.join(config.path_test, img_id)
    img = imread(img_path, as_gray=True)

    img_resized = resize(
        img,
        (config.im_height, config.im_width),
        preserve_range=True,
        anti_aliasing=True,
    )
    img_norm = img_resized.astype(np.float32) / 255.0
    img_input = img_norm[..., np.newaxis]  # add channel

    if _has_tf:
        pred = model.predict(img_input[np.newaxis, ...], verbose=0)[0, ..., 0]
    else:
        if img.dtype != np.uint8:
            img_uint8 = (img * 255).astype(np.uint8)
        else:
            img_uint8 = img
        img_blurred = filters.gaussian(img_uint8, sigma=0.5, preserve_range=True)
        img_uint8 = img_blurred.astype(np.uint8)
        if _has_cv2:
            if img_uint8.ndim == 3:
                img_gray = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = img_uint8
            _, mask = cv2.threshold(img_gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            pred = mask.astype(np.float32)
        else:
            thresh = filters.threshold_otsu(img_uint8)
            pred = (img_uint8 > thresh).astype(np.float32)

    mask_bin = (pred > 0).astype(np.uint8)
    filled = binary_fill_holes(mask_bin).astype(np.uint8)

    mask_resized_back = resize(
        filled,
        (img.shape[0], img.shape[1]),
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )
    mask_resized_back = (mask_resized_back > 0.5).astype(np.uint8)

    preds_test_upsampled.append(mask_resized_back)

print("Done processing test data.")




## === cell 7
def RLenc(img, order="F", format=True):
    """Run‑length encoding for binary mask."""
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []
    r = 0
    pos = 1
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
    if format:
        return " ".join(f"{p} {l}" for p, l in runs)
    return runs


pred_dict = {
    fn[:-4]: RLenc(np.round(preds_test_upsampled[i] > threshold_best))
    for i, fn in tqdm(enumerate(test_ids), total=len(test_ids), desc="Encoding RLE")
}



## === cell 8
sub = pd.DataFrame.from_dict(pred_dict, orient="index")
sub.index.name = "id"
sub.columns = ["rle_mask"]
submission_path = "submission.csv"
sub.to_csv(submission_path, index=True)
print(f"Submission written to {submission_path}")
