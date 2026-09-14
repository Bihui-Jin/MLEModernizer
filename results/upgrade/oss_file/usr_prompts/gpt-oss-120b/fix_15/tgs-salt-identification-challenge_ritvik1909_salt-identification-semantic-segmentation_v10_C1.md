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

0.72102

# 6. Current score

0.0759

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.089) has done: 'The update adds a missing import, replaces the expensive per‑threshold IoU loop with a fully vectorised version (computing intersection/union per image in bulk), and keeps the original helper functions unchanged to preserve semantics.'
- What this solution (achieved 0.0665) has done: 'The changes disable TensorFlow (which fails to import in this environment) and ensure all downstream code gracefully falls back to the simple Otsu predictor. Guard clauses prevent references to undefined variables (`build_model`, `results`, `model`) and keep the original workflow otherwise unchanged.'
- What this solution (achieved 0.0759) has done: 'The changes add a lightweight post‑processing step that keeps only the largest connected component of each Otsu mask (the typical shape of the salt region) and use nearest‑neighbor up‑sampling when resizing predictions back to the original image size. This modest refinement sharpens the binary masks and raises the validation IoU, moving the Kaggle score closer to the target without altering the core workflow.'

# 9. Code solution

## === cell 0
import os
import sys
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import seaborn as sns

tf_available = False
tf = None  # placeholder to avoid NameError later


class config:
    im_width = 128
    im_height = 128
    im_chan = 1
    base_path = "/kaggle/input/tgs-salt-identification-challenge"
    path_train = os.path.join(base_path, "train")
    path_test = os.path.join(base_path, "test")


train_csv = os.path.join(config.base_path, "train.csv")
train_df = pd.read_csv(train_csv)
train_ids = train_df["id"].astype(str).values
train_img_dir = os.path.join(config.path_train, "images")
train_mask_dir = os.path.join(config.path_train, "masks")

sample_sub_path = os.path.join(config.base_path, "sample_submission.csv")
sample_sub = pd.read_csv(sample_sub_path)
test_ids = sample_sub["id"].astype(str).values
test_img_dir = os.path.join(config.path_test, "images")




## === cell 1
def load_resize_gray(path, target_h, target_w):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # (h, w)
    img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    img = img[..., np.newaxis]  # add channel dim
    return img


X = np.empty(
    (len(train_ids), config.im_height, config.im_width, config.im_chan), dtype=np.uint8
)
Y = np.empty((len(train_ids), config.im_height, config.im_width, 1), dtype=bool)

print("Loading and resizing train images and masks...")
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    img_path = os.path.join(train_img_dir, f"{id_}.png")
    mask_path = os.path.join(train_mask_dir, f"{id_}.png")
    X[n] = load_resize_gray(img_path, config.im_height, config.im_width).astype(
        np.uint8
    )
    m = load_resize_gray(mask_path, config.im_height, config.im_width)
    Y[n] = (m > 127).astype(bool)  # binarize mask
print("Done!")
print("X shape:", X.shape)
print("Y shape:", Y.shape)




## === cell 2
split_idx = int(0.9 * len(X))
X_train_orig = X[:split_idx]
Y_train_orig = Y[:split_idx]
X_eval = X[split_idx:]
Y_eval = Y[split_idx:]

X_hflip = np.flip(X_train_orig, axis=2)  # horizontal flip
Y_hflip = np.flip(Y_train_orig, axis=2)

X_train = np.concatenate([X_train_orig, X_hflip], axis=0)
Y_train = np.concatenate([Y_train_orig, Y_hflip], axis=0)

print("After augmentation:")
print("X_train shape:", X_train.shape, "X_eval shape:", X_eval.shape)
print("Y_train shape:", Y_train.shape, "Y_eval shape:", Y_eval.shape)




## === cell 3
def _keep_largest_component(mask):
    """
    Keep only the largest connected component in a binary mask.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if num_labels <= 1:
        return mask
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    cleaned = (labels == largest_label).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cleaned


def simple_otsu_predict(images):
    """
    images: uint8 array of shape (N, H, W, 1)
    returns binary masks of same shape (N, H, W, 1) as uint8 (0/1)
    """
    preds = np.empty_like(images, dtype=np.uint8)
    for i in range(images.shape[0]):
        img = images[i, :, :, 0]
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        _, mask = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = _keep_largest_component(mask)
        preds[i, :, :, 0] = mask
    return preds


if tf_available:
    pass
else:
    preds_eval = simple_otsu_predict(X_eval)




## === cell 4
sns.set_style("darkgrid")
if tf_available and "results" in globals():
    fig, ax = plt.subplots(2, 1, figsize=(20, 8))
    history = pd.DataFrame(results.history)
    history[["loss", "val_loss"]].plot(ax=ax[0])
    history[["accuracy", "val_accuracy"]].plot(ax=ax[1])
    fig.suptitle("Learning Curve", fontsize=24)
    plt.show()
else:
    print("Training history not available (fallback mode).")


def iou_metric(y_true, y_pred, print_table=False):
    true_objects = 2
    pred_objects = 2

    intersection = np.histogram2d(
        y_true.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects)
    )[0]

    area_true = np.histogram(y_true, bins=true_objects)[0]
    area_pred = np.histogram(y_pred, bins=pred_objects)[0]

    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    union = area_true + area_pred - intersection
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    union[union == 0] = 1e-9
    iou = intersection / union

    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1
        false_positives = np.sum(matches, axis=0) == 0
        false_negatives = np.sum(matches, axis=1) == 0
        tp = np.sum(true_positives)
        fp = np.sum(false_positives)
        fn = np.sum(false_negatives)
        return tp, fp, fn

    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        p = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        if print_table:
            print(f"{t:.3f}\t{tp}\t{fp}\t{fn}\t{p:.3f}")
        prec.append(p)
    if print_table:
        print(f"AP\t-\t-\t-\t{np.mean(prec):.3f}")
    return np.mean(prec)


def iou_metric_batch(y_true_batch, y_pred_batch):
    return np.mean(
        [
            iou_metric(y_true, y_pred)
            for y_true, y_pred in zip(y_true_batch, y_pred_batch)
        ]
    )


Y_eval_uint = Y_eval.astype(np.uint8)
preds_eval_uint = preds_eval.astype(np.uint8)

inter = np.sum(Y_eval_uint * preds_eval_uint, axis=(1, 2, 3))
union = np.sum(np.clip(Y_eval_uint + preds_eval_uint, 0, 1), axis=(1, 2, 3))
iou_per_image = inter / (union + 1e-9)
print(f"Mean IoU on validation split: {iou_per_image.mean():.4f}")




## === cell 5
X_test = np.empty(
    (len(test_ids), config.im_height, config.im_width, config.im_chan), dtype=np.uint8
)
sizes_test = []
print("Loading and resizing test images...")
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    img_path = os.path.join(test_img_dir, f"{id_}.png")
    x = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    sizes_test.append([x.shape[0], x.shape[1]])  # original height, width
    x = cv2.resize(
        x, (config.im_width, config.im_height), interpolation=cv2.INTER_LINEAR
    )
    X_test[n] = x[..., np.newaxis].astype(np.uint8)
print("Done!")

if tf_available:
    preds_test = model.predict(X_test, verbose=0)
else:
    preds_test = simple_otsu_predict(X_test)

preds_test_upsampled = []
for i in trange(len(preds_test)):
    up = cv2.resize(
        np.squeeze(preds_test[i]),
        (sizes_test[i][1], sizes_test[i][0]),  # width, height
        interpolation=cv2.INTER_NEAREST,
    )
    up = _keep_largest_component(up)
    preds_test_upsampled.append(up)




## === cell 6
def RLenc(img, order="F", format=True):
    """
    img: binary mask (2D array)
    Returns run‑length encoding as required by the competition.
    """
    bytes = img.reshape(-1, order=order)
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


threshold_best = 0.5

pred_dict = {
    fn: RLenc(np.round(preds_test_upsampled[i] > threshold_best))
    for i, fn in enumerate(test_ids)
}




## === cell 7
sub = pd.DataFrame.from_dict(pred_dict, orient="index")
sub.index.name = "id"
sub.columns = ["rle_mask"]
sub.to_csv("submission.csv", index=True)
print("Submission file written to submission.csv")
