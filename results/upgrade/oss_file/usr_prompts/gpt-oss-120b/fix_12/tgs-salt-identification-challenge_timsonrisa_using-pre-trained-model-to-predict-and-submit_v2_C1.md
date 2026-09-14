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

0.5795153613154457

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.026) has done: 'I replace the broken TensorFlow/Keras imports and the missing model file with a lightweight end‑to‑end pipeline that reads the training images/masks, trains a tiny convolutional network for a few epochs, predicts on the test set, resizes the predictions back to the original image size, encodes them with run‑length encoding and finally writes a valid `submission.csv`. The changes fix the import errors, remove the nonexistent model path, and add the missing variables so the notebook runs from start to finish and produces a proper submission file.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from PIL import Image

try:
    import tensorflow as tf
    from tensorflow.keras.layers import (
        Input,
        Conv2D,
        MaxPooling2D,
        Conv2DTranspose,
        concatenate,
    )
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
except Exception:
    tf = None


def imread(path):
    """Read an image and return a numpy array (H,W,3)."""
    img = Image.open(path).convert("RGB")
    return np.array(img)


def read_mask(path):
    """Read a mask image (assumed single‑channel) and return a binary numpy array."""
    mask = Image.open(path).convert("L")
    arr = np.array(mask)
    return (arr > 127).astype(np.uint8)


def resize(img, size, mode="constant", preserve_range=True):
    """Resize a numpy image to (height, width) using Pillow."""
    h, w = size
    pil_img = Image.fromarray(img)
    pil_resized = pil_img.resize((w, h), Image.BILINEAR)
    arr = np.array(pil_resized)
    if not preserve_range:
        arr = arr.astype(np.float32) / 255.0
    return arr


BASE_DIR = "/kaggle/working"
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train", "images")
TRAIN_MASK_DIR = os.path.join(BASE_DIR, "train", "masks")
TEST_CSV = os.path.join(BASE_DIR, "sample_submission.csv")  # contains test ids
TEST_IMG_DIR = os.path.join(BASE_DIR, "test", "images")




## === cell 1
train_df = pd.read_csv(TRAIN_CSV)
train_ids = train_df["id"].values
print(f"# train images: {len(train_ids)}")

IMG_SIZE = 101
X = np.zeros((len(train_ids), IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
Y = np.zeros((len(train_ids), IMG_SIZE, IMG_SIZE, 1), dtype=np.uint8)

print("Reading and resizing training images / masks...")
for i, img_id in enumerate(tqdm(train_ids)):
    img_path = os.path.join(TRAIN_IMG_DIR, f"{img_id}.png")
    mask_path = os.path.join(TRAIN_MASK_DIR, f"{img_id}.png")
    img = imread(img_path)[:, :, :3]
    mask = read_mask(mask_path)
    img_resized = (
        resize(img, (IMG_SIZE, IMG_SIZE), mode="constant", preserve_range=True).astype(
            np.float32
        )
        / 255.0
    )
    mask_resized = resize(
        mask, (IMG_SIZE, IMG_SIZE), mode="constant", preserve_range=True
    )
    mask_resized = (mask_resized > 0).astype(np.uint8)  # binary
    X[i] = img_resized
    Y[i, :, :, 0] = mask_resized

val_split = 0.05
val_idx = int(len(train_ids) * (1 - val_split))
X_train, X_val = X[:val_idx], X[val_idx:]
Y_train, Y_val = Y[:val_idx], Y[val_idx:]

avg_mask = Y_train.mean(axis=0).squeeze()  # shape (IMG_SIZE, IMG_SIZE)




## === cell 2
def build_simple_unet(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    inputs = Input(shape=input_shape)

    c1 = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(64, (3, 3), activation="relu", padding="same")(p1)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(128, (3, 3), activation="relu", padding="same")(p2)

    u1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c3)
    concat1 = concatenate([u1, c2])
    c4 = Conv2D(64, (3, 3), activation="relu", padding="same")(concat1)

    u2 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c4)
    concat2 = concatenate([u2, c1])
    c5 = Conv2D(32, (3, 3), activation="relu", padding="same")(concat2)

    outputs = Conv2D(1, (1, 1), activation="sigmoid")(c5)

    model = Model(inputs, outputs)
    return model


if tf is not None:
    print("Building and training a lightweight UNet model...")
    model = build_simple_unet()
    model.compile(
        optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"]
    )
    callbacks = [
        EarlyStopping(patience=12, restore_best_weights=True, monitor="val_loss")
    ]
    model.fit(
        X_train,
        Y_train,
        validation_data=(X_val, Y_val),
        epochs=150,
        batch_size=16,
        callbacks=callbacks,
        verbose=2,
    )

    preds_val = model.predict(X_val, batch_size=16).squeeze(-1)  # (N, H, W)

    def compute_map(y_true, y_pred_bin):
        """Mean Average Precision over IoU thresholds 0.5‑0.95."""
        iou_thresholds = np.arange(0.5, 1.0, 0.05)
        precisions = []
        for thr in iou_thresholds:
            intersection = (y_pred_bin & y_true).sum(axis=(1, 2))
            union = (y_pred_bin | y_true).sum(axis=(1, 2)) + 1e-7
            iou = intersection / union
            tp = (iou > thr).sum()
            fp = ((iou <= thr) & (y_pred_bin.sum(axis=(1, 2)) > 0)).sum()
            fn = ((iou <= thr) & (y_true.sum(axis=(1, 2)) > 0)).sum()
            precision = tp / (tp + fp + fn + 1e-7)
            precisions.append(precision)
        return np.mean(precisions)

    candidate_thr = np.arange(0.30, 0.71, 0.01)
    best_thr = 0.5
    best_map = -1.0
    Y_val_bin = Y_val.squeeze(-1)  # (N, H, W)

    for t in candidate_thr:
        bin_pred = (preds_val > t).astype(np.uint8)
        map_score = compute_map(Y_val_bin, bin_pred)
        if map_score > best_map:
            best_map = map_score
            best_thr = t

    BEST_THR = best_thr
    print(f"Selected BEST_THR = {BEST_THR:.3f} (validation MAP = {best_map:.4f})")
else:
    print("TensorFlow unavailable – training will be skipped.")
    BEST_THR = 0.5




## === cell 3
test_df = pd.read_csv(TEST_CSV)
test_ids = test_df["id"].values
print(f"# test images: {len(test_ids)}")

X_test = np.zeros((len(test_ids), IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
orig_sizes = []

print("Reading and resizing test images...")
for i, img_id in enumerate(tqdm(test_ids)):
    img_path = os.path.join(TEST_IMG_DIR, f"{img_id}.png")
    img = imread(img_path)[:, :, :3]
    orig_sizes.append(img.shape[:2])  # (height, width)
    img_resized = (
        resize(img, (IMG_SIZE, IMG_SIZE), mode="constant", preserve_range=True).astype(
            np.float32
        )
        / 255.0
    )
    X_test[i] = img_resized

if tf is not None and "model" in globals():
    preds_resized = model.predict(X_test, batch_size=16).squeeze(-1)  # (N, H, W)
else:
    preds_resized = np.stack([avg_mask] * len(test_ids), axis=0)

preds_upsampled = []
print("Upsampling predictions to original image sizes...")
for i, (h, w) in enumerate(tqdm(orig_sizes)):
    up = resize(
        preds_resized[i],
        (h, w),
        mode="constant",
        preserve_range=True,
    )
    mask_bin = (up > BEST_THR).astype(np.uint8)
    preds_upsampled.append(mask_bin)




## === cell 4
def RLenc(img, order="F"):
    """
    Run‑length encoding for binary mask.
    img: 2‑D binary array (1 = mask, 0 = background)
    order: 'F' for column‑wise (Fortran) order required by the competition.
    Returns a space‑separated string.
    """
    pixels = img.flatten(order=order)
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] = runs[1::2] - runs[::2]
    return " ".join(str(x) for x in runs)


pred_dict = {}
print("Encoding predictions to RLE...")
for i, img_id in enumerate(tqdm(test_ids)):
    mask = preds_upsampled[i]
    rle = RLenc(mask)
    pred_dict[img_id] = rle

submission = pd.DataFrame.from_dict(pred_dict, orient="index", columns=["rle_mask"])
submission.index.name = "id"
submission.reset_index(inplace=True)
submission.to_csv("submission.csv", index=False)
print('Submission file "submission.csv" written with', len(submission), "rows.")
