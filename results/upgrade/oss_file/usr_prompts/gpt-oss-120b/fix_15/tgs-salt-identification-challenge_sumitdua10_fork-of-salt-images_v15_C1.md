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

0.74891

# 6. Current score

0.5221

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5221) has done: 'Implemented fixes to enable end‑to‑end execution and generate a valid submission CSV.  
- Added `mode='min'` to the EarlyStopping callback to resolve the monitor metric error.  
- Rewrote test‑set construction: list image files in the test directory, merge with depths, and build proper arrays, preventing empty‑array stacking errors.  
- Adjusted variable names to keep consistency for downstream prediction and submission steps.'
- What this solution (achieved 0.5221) has done: 'I add a simple train‑validation split, use the validation set to pick a better mask‑threshold (instead of the low 0.25 that caused many false positives), and raise the training epochs modestly so the model can learn a bit more. These tweaks keep the original architecture and loss unchanged while moving the score upward toward the target.'
- What this solution (achieved 0.5221) has done: 'Implemented a modest training boost and a more appropriate early‑stopping monitor.  
- Increased `EPOCHS` to 20 and patience to 6 so the model can train longer while still avoiding over‑fit.  
- Switched EarlyStopping to watch the **validation dice loss** (`val_dice_loss`) which better reflects generalisation.  
- Expanded the threshold sweep to include a slightly lower value (0.25) and a higher one (0.70) for finer selection of the optimal mask cutoff.  

These lightweight tweaks keep the original architecture unchanged while steering the validation performance upward, moving the score closer to the target.'
- What this solution (achieved 0.5221) has done: 'I add the missing imports, define BASE_PATH so all file paths resolve, and ensure the required TensorFlow/Keras objects are available. This fixes the NameError crashes and lets the pipeline run end‑to‑end, producing a valid CSV submission while keeping the original model and training logic unchanged.'
- What this solution (achieved 0.5221) has done: 'Implemented fixes to resolve the protobuf import error by forcing the pure‑Python implementation, increased training epochs and early‑stopping patience for better model convergence, broadened the threshold search, and switched the validation‑threshold selection to use IoU‑based scoring (which aligns with the competition metric) instead of Dice. These changes keep the original architecture unchanged while addressing runtime failures and nudging the validation performance toward the target score.'
- What this solution (achieved 0.5221) has done: 'Implemented a protobuf monkey‑patch to avoid the `MessageFactory` attribute error, added test‑time augmentation (horizontal & vertical flips) for more robust predictions, extended the validation threshold search to the full competition range, and replaced the custom RLE logic with a standard, reliable encoder. These fixes ensure the script runs end‑to‑end, produces a valid CSV submission, and modestly improves validation performance, moving the score closer to the target.'
- What this solution (achieved 0.5221) has done: 'We boost the validation performance (and thus the final score) by expanding the training augmentation to include the combined horizontal‑vertical flip, raising the training budget, and letting early‑stopping monitor run a bit longer. The TTA routine is also extended to average the prediction from this additional augmentation, which modestly improves the mask quality while keeping the original model architecture unchanged.'
- What this solution (achieved 0.5221) has done: 'I increase the early‑stopping patience to let the model train a bit longer and replace the simple IoU‑based threshold search with a proper competition‑style metric (mean average precision across IoU thresholds). This keeps the original architecture untouched while selecting a mask‑threshold that better matches the evaluation, which should raise the validation score toward the target.'
- What this solution (achieved 0.5221) has done: 'I add a small connected‑component filter in the post‑processing step to drop tiny spurious masks (which often cause false‑positives and lower the MAP score) and switch early‑stopping to monitor the overall validation loss (`val_loss`) so the model can train a bit longer before stopping. These minimal tweaks keep the architecture unchanged while aiming to raise the validation‑based competition score toward the target.'
- What this solution (achieved 0.5221) has done: 'I adjust the early‑stopping to monitor validation Dice loss (which aligns better with the segmentation goal) and refine the mask‑threshold search with a finer grid around the best coarse value, then tighten the post‑processing component size filter slightly. These small tweaks keep the original architecture intact while aiming to raise the validation MAP score toward the target.'
- What this solution (achieved 0.5221) has done: 'I slightly extend the training budget (more epochs and a longer early‑stopping patience) so the model can converge a bit better, and I make the post‑processing a little stricter by raising the minimum component size from 20 to 30 pixels to cut down false‑positive tiny blobs. These tiny adjustments keep the original architecture and loss unchanged while aiming to lift the validation MAP score toward the target.'

# 9. Code solution

## === cell 0
import os

try:
    import google.protobuf.message_factory as mf

    if not hasattr(mf.MessageFactory, "GetPrototype"):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        mf.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import time
import numpy as np
import pandas as pd
from PIL import Image

from scipy.ndimage import label as nd_label

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

BASE_PATH = "/kaggle/input/tgs-salt-identification-challenge"




## === cell 1
df_train = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))
df_depths = pd.read_csv(os.path.join(BASE_PATH, "depths.csv"))
df_depths["z"] = df_depths["z"] / df_depths["z"].max()  # normalise depth
df_train = df_train.merge(df_depths, on="id", how="inner")
df_train = df_train[["id", "rle_mask", "z"]].rename(columns={"z": "depth"})
print("train df head:\n", df_train.head())
print("train shape:", df_train.shape)




## === cell 2
def load_image(path):
    return np.array(Image.open(path))


imgs = []
msks = []
depth_imgs = []

for idx, depth in zip(df_train["id"], df_train["depth"]):
    img_path = os.path.join(BASE_PATH, "train/images/{}.png".format(idx))
    mask_path = os.path.join(BASE_PATH, "train/masks/{}.png".format(idx))
    img = load_image(img_path)
    msk = load_image(mask_path)

    img = img[:, :, 0] if img.ndim == 3 else img
    msk = msk[:, :, 0] if msk.ndim == 3 else msk

    imgs.append(img.astype(np.float32) / 255.0)  # normalise image
    msks.append((msk > 127).astype(np.float32))  # binary mask
    depth_imgs.append(np.full((50, 50, 1), depth, dtype=np.float32))

train_x_full = np.expand_dims(np.stack(imgs, axis=0), -1)  # (N,101,101,1)
train_y_full = np.expand_dims(np.stack(msks, axis=0), -1)  # (N,101,101,1)
depth_np_full = np.stack(depth_imgs, axis=0)  # (N,50,50,1)

print("train_x shape:", train_x_full.shape)
print("train_y shape:", train_y_full.shape)
print("depth_np shape:", depth_np_full.shape)




## === cell 3
val_fraction = 0.1
num_samples = train_x_full.shape[0]
val_size = int(num_samples * val_fraction)

indices = np.arange(num_samples)
np.random.shuffle(indices)
val_idx = indices[:val_size]
train_idx = indices[val_size:]

train_x = train_x_full[train_idx]
train_y = train_y_full[train_idx]
depth_np = depth_np_full[train_idx]

val_x = train_x_full[val_idx]
val_y = train_y_full[val_idx]
val_depth = depth_np_full[val_idx]

np.random.seed(1234)

horiz_x = np.flip(train_x, axis=2)
horiz_y = np.flip(train_y, axis=2)
horiz_depth = depth_np.copy()

vert_x = np.flip(train_x, axis=1)
vert_y = np.flip(train_y, axis=1)
vert_depth = depth_np.copy()

both_x = np.flip(np.flip(train_x, axis=2), axis=1)
both_y = np.flip(np.flip(train_y, axis=2), axis=1)
both_depth = depth_np.copy()

train_x = np.concatenate([train_x, horiz_x, vert_x, both_x], axis=0)
train_y = np.concatenate([train_y, horiz_y, vert_y, both_y], axis=0)
depth_np = np.concatenate([depth_np, horiz_depth, vert_depth, both_depth], axis=0)

print("After augmentation - train_x shape:", train_x.shape)




## === cell 4
def dice_coeff(y_true, y_pred, smooth=1.0):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def dice_loss(y_true, y_pred):
    return 1 - dice_coeff(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return 0.2 * bce + 0.8 * dice_loss(y_true, y_pred)


Height, Width = 101, 101

img_input = layers.Input(shape=(Height, Width, 1), name="img_input")
depth_input = layers.Input(shape=(50, 50, 1), name="depth_input")

depth_up = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(
    depth_input
)  # 50→100
depth_up = layers.Resizing(Height, Width, interpolation="bilinear")(depth_up)  # 100→101

x = layers.Concatenate()([img_input, depth_up])  # (101,101,2)

x = layers.Conv2D(32, 3, padding="same", activation="elu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(2)(x)  # 101→50

x = layers.Conv2D(64, 3, padding="same", activation="elu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(2)(x)  # 50→25

x = layers.Conv2D(128, 3, padding="same", activation="elu")(x)
x = layers.BatchNormalization()(x)

x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="elu")(
    x
)  # 25→50
x = layers.BatchNormalization()(x)

x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="elu")(
    x
)  # 50→100
x = layers.Resizing(Height, Width, interpolation="bilinear")(x)  # 100→101
x = layers.BatchNormalization()(x)

output = layers.Conv2D(1, 1, activation="sigmoid")(x)

model = Model(inputs=[img_input, depth_input], outputs=output)
model.summary()




## === cell 5
model.compile(optimizer=Adam(), loss=bce_dice_loss, metrics=[dice_loss, "accuracy"])

early_stop = EarlyStopping(
    monitor="val_dice_loss", patience=30, mode="min", restore_best_weights=True
)

BATCH_SIZE = 32
EPOCHS = 120  # give the model more epochs; early stopping will cut off if not needed.

history = model.fit(
    [train_x, depth_np],
    train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[early_stop],
    verbose=2,
    validation_data=([val_x, val_depth], val_y),
)

model.save("model_7.h5")




## === cell 6
def tta_predict(model, imgs, depths, batch_size):
    pred_orig = model.predict([imgs, depths], batch_size=batch_size, verbose=0)

    horiz_imgs = np.flip(imgs, axis=2)
    pred_h = model.predict([horiz_imgs, depths], batch_size=batch_size, verbose=0)
    pred_h = np.flip(pred_h, axis=2)

    vert_imgs = np.flip(imgs, axis=1)
    pred_v = model.predict([vert_imgs, depths], batch_size=batch_size, verbose=0)
    pred_v = np.flip(pred_v, axis=1)

    both_imgs = np.flip(np.flip(imgs, axis=2), axis=1)
    pred_b = model.predict([both_imgs, depths], batch_size=batch_size, verbose=0)
    pred_b = np.flip(np.flip(pred_b, axis=2), axis=1)

    return (pred_orig + pred_h + pred_v + pred_b) / 4.0


val_pred = tta_predict(model, val_x, val_depth, batch_size=BATCH_SIZE)


def competition_score(preds, trues, mask_thresh):
    """Mean average precision across IoU thresholds 0.5‑0.95 for a single mask per image."""
    preds_bin = (preds.squeeze() > mask_thresh).astype(np.uint8)
    ious = []
    for i in range(len(preds_bin)):
        pred = preds_bin[i]
        true = trues[i].squeeze().astype(np.uint8)
        if pred.sum() == 0 and true.sum() == 0:
            ious.append(1.0)
        else:
            inter = np.logical_and(pred, true).sum()
            union = np.logical_or(pred, true).sum()
            ious.append(inter / union if union > 0 else 0.0)
    ious = np.array(ious)

    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    precisions = []
    for t in iou_thresholds:
        tp = (ious > t).sum()
        fp = ((preds_bin.sum(axis=(1, 2)) > 0) & (ious <= t)).sum()
        fn = ((trues.squeeze().sum(axis=(1, 2)) > 0) & (ious <= t)).sum()
        denom = tp + fp + fn
        precisions.append(tp / denom if denom > 0 else 1.0)
    return np.mean(precisions)


best_thresh = 0.5
best_score = -1.0
for t in np.arange(0.05, 0.96, 0.05):
    score = competition_score(val_pred, val_y, t)
    print(f"Coarse mask thresh {t:.2f} – competition score: {score:.4f}")
    if score > best_score:
        best_score = score
        best_thresh = t

fine_start = max(0.05, best_thresh - 0.04)
fine_end = min(0.95, best_thresh + 0.04)
for t in np.arange(fine_start, fine_end + 0.001, 0.01):
    score = competition_score(val_pred, val_y, t)
    print(f"Fine mask thresh {t:.2f} – competition score: {score:.4f}")
    if score > best_score:
        best_score = score
        best_thresh = t

print(f"Selected mask threshold: {best_thresh:.2f} (score {best_score:.4f})")




## === cell 7
test_image_dir = os.path.join(BASE_PATH, "test/images")
test_ids = [
    fname.split(".")[0]
    for fname in os.listdir(test_image_dir)
    if fname.lower().endswith(".png")
]
df_test = pd.DataFrame({"id": test_ids})
df_test = df_test.merge(df_depths, on="id", how="left")
df_test.rename(columns={"z": "depth"}, inplace=True)

print("Test set size:", df_test.shape[0])

test_imgs = []
test_depths = []

for idx, depth in zip(df_test["id"], df_test["depth"]):
    img_path = os.path.join(test_image_dir, f"{idx}.png")
    img = load_image(img_path)
    img = img[:, :, 0] if img.ndim == 3 else img
    test_imgs.append(img.astype(np.float32) / 255.0)
    test_depths.append(np.full((50, 50, 1), depth, dtype=np.float32))

test_x = np.expand_dims(np.stack(test_imgs, axis=0), -1)  # (N,101,101,1)
depth_test_np = np.stack(test_depths, axis=0)  # (N,50,50,1)

print("test_x shape:", test_x.shape)
print("depth_test_np shape:", depth_test_np.shape)




## === cell 8
print("Predicting now…")
y_pred = tta_predict(model, test_x, depth_test_np, batch_size=BATCH_SIZE)
print("Prediction complete. Shape:", y_pred.shape)




## === cell 9
def rle_encode(mask):
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] = runs[1::2] - runs[::2]
    return " ".join(str(x) for x in runs)


def post_process(preds, threshold=0.5, min_size=30):
    """
    Binarise predictions, remove tiny connected components (size < min_size),
    and encode remaining masks with RLE.
    """
    preds_bin = (preds.squeeze() > threshold).astype(np.uint8)
    encodings = []
    for mask in preds_bin:
        if mask.sum() == 0:
            encodings.append("")
            continue

        labeled, num = nd_label(mask)
        keep_mask = np.zeros_like(mask, dtype=np.uint8)

        for comp_id in range(1, num + 1):
            component = labeled == comp_id
            if component.sum() >= min_size:
                keep_mask |= component

        if keep_mask.sum() == 0:
            encodings.append("")
        else:
            encodings.append(rle_encode(keep_mask))
    return encodings


new_y_pred = post_process(y_pred, threshold=best_thresh)
print("Encoded masks count:", len(new_y_pred))




## === cell 10
submission = pd.DataFrame({"id": df_test["id"], "rle_mask": new_y_pred})
submission_path = "output_6.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")




## === cell 11
print("Finished at", time.time())
