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

0.72826

# 6. Current score

0.0268

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0677) has done: 'I added robust path detection for the training and test folders, guarded the unzip steps so they only run when the zip files actually exist, and fixed the order of cell execution so that the data arrays are created before they are used. These changes eliminate the FileNotFoundError and NameError, ensuring the script runs end‑to‑end and writes a valid `submission.csv` file.'
- What this solution (achieved 0.0755) has done: 'I add a definition for `val_ids` after the train‑validation split so the validation loop can locate the corresponding mask files, and keep the existing logic that selects the best threshold (`threshold_best`). This resolves the NameError in cells 5 and 7, allowing the notebook to run end‑to‑end and write a proper `submission.csv` file, while preserving the original model and processing pipeline.'
- What this solution (achieved 0.0305) has done: 'I make two lightweight tweaks that keep the original pipeline but give the mask predictions a better chance of matching the ground‑truth. First, the dummy model now smooths each image with a small Gaussian blur before Otsu thresholding and automatically flips the binary mask if it appears to predict the background instead of salt (i.e., when > 50 % of pixels are foreground). Second, the validation step searches a broader range of thresholds (0.1 → 0.9) so the optimal cut‑off can be found more reliably. These changes retain the overall structure and training loop while nudging the IoU toward the target score.'
- What this solution (achieved 0.0268) has done: 'I keep the overall pipeline unchanged but improve the handling of binary masks during up‑sampling. The original code used `skimage.transform.resize` with the default linear interpolation, which blurs the 0/1 predictions and hurts IoU. By switching to nearest‑neighbor interpolation (`order=0`) and disabling anti‑aliasing for the mask up‑sampling steps in both validation and test processing, the binary masks stay crisp, which should raise the IoU and move the score much closer to the target. The only modifications are these two `resize` calls.'

# 9. Code solution

## === cell 0
import os, warnings, numpy as np, pandas as pd, cv2
from tqdm.auto import tqdm
from skimage.io import imread, imshow
from skimage.transform import resize

warnings.filterwarnings("ignore")


class config:
    im_width = 128
    im_height = 128
    im_chan = 1

    _candidates_train = [
        "input/train",
        "train",
        "input/tgs-salt-identification-challenge/train",
        "tgs-salt-identification-challenge/train",
        "working/tgs-salt-identification-challenge/train",
    ]
    _candidates_test = [
        "input/test",
        "test",
        "input/tgs-salt-identification-challenge/test",
        "tgs-salt-identification-challenge/test",
        "working/tgs-salt-identification-challenge/test",
    ]

    for cand in _candidates_train:
        if os.path.isdir(os.path.join(cand, "images")) and os.path.isdir(
            os.path.join(cand, "masks")
        ):
            path_train = cand
            break
    else:
        path_train = "train"  # fallback (may fail later)

    for cand in _candidates_test:
        if os.path.isdir(os.path.join(cand, "images")):
            path_test = cand
            break
    else:
        path_test = "test"  # fallback (may fail later)




## === cell 1
train_zip = "../input/tgs-salt-identification-challenge/train.zip"
test_zip = "../input/tgs-salt-identification-challenge/test.zip"
if os.path.isfile(train_zip) and not os.path.isdir(
    os.path.join(config.path_train, "images")
):
    os.system(f"unzip -q {train_zip} -d {config.path_train}")
if os.path.isfile(test_zip) and not os.path.isdir(
    os.path.join(config.path_test, "images")
):
    os.system(f"unzip -q {test_zip} -d {config.path_test}")




## === cell 2
train_dir = os.path.join(config.path_train, "images")
mask_dir = os.path.join(config.path_train, "masks")
train_ids = sorted([f for f in os.listdir(train_dir) if f.lower().endswith(".png")])
print(f"Found {len(train_ids)} training images")

X = np.zeros(
    (len(train_ids), config.im_height, config.im_width, config.im_chan), dtype=np.uint8
)
Y = np.zeros((len(train_ids), config.im_height, config.im_width, 1), dtype=bool)
sizes_train = []  # keep original (height, width) for later up‑sampling

print("Loading and resizing training data...")
for n, img_id in enumerate(tqdm(train_ids)):
    img_path = os.path.join(train_dir, img_id)
    mask_path = os.path.join(mask_dir, img_id)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    sizes_train.append((img.shape[0], img.shape[1]))

    img_resized = resize(
        img,
        (config.im_height, config.im_width),
        mode="constant",
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.uint8)
    X[n] = img_resized[..., None]

    mask_resized = resize(
        mask,
        (config.im_height, config.im_width),
        mode="constant",
        preserve_range=True,
        anti_aliasing=False,
    )
    Y[n] = (mask_resized > 127).astype(bool)[..., None]

print("Done loading training data.")




## === cell 3
split_idx = int(0.9 * len(X))
X_train, X_val = X[:split_idx], X[split_idx:]
Y_train, Y_val = Y[:split_idx], Y[split_idx:]
sizes_val = sizes_train[split_idx:]  # original sizes for validation images
val_ids = train_ids[split_idx:]  # filenames corresponding to validation set

print("X_train shape:", X_train.shape, "X_val shape:", X_val.shape)




## === cell 4
class DummyModel:
    def fit(self, *args, **kwargs):
        return None

    def predict(self, X, verbose=0):
        preds = []
        for img in tqdm(X, disable=verbose == 0):
            img_uint8 = img.squeeze()
            if img_uint8.max() > 1:
                img_uint8 = img_uint8.astype(np.uint8)
            else:
                img_uint8 = (img_uint8 * 255).astype(np.uint8)

            blurred = cv2.GaussianBlur(img_uint8, (5, 5), 0)

            _, th = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            if (th > 0).mean() > 0.5:
                th = cv2.bitwise_not(th)

            preds.append((th / 255.0).reshape(config.im_height, config.im_width, 1))
        return np.array(preds, dtype=np.float32)


model = DummyModel()
model.fit(
    X_train, Y_train, validation_data=(X_val, Y_val), epochs=1, batch_size=8, verbose=0
)




## === cell 5
def iou_score(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    if union == 0:
        return 1.0
    return intersection / union


val_preds = model.predict(X_val, verbose=0)
threshold_grid = np.arange(0.1, 0.91, 0.05)
best_thresh = 0.5
best_iou = -1.0

for th in threshold_grid:
    ious = []
    for i in range(len(val_preds)):
        pred_up = resize(
            np.squeeze(val_preds[i]),
            sizes_val[i],
            order=0,
            mode="constant",
            preserve_range=True,
            anti_aliasing=False,
        )
        bin_pred = (pred_up > th).astype(bool)
        true_mask = cv2.imread(os.path.join(mask_dir, val_ids[i]), cv2.IMREAD_GRAYSCALE)
        true_mask = (true_mask > 127).astype(bool)
        ious.append(iou_score(true_mask, bin_pred))
    mean_iou = np.mean(ious)
    if mean_iou > best_iou:
        best_iou = mean_iou
        best_thresh = th

threshold_best = best_thresh
print(f"Selected threshold_best = {threshold_best:.3f} (mean IoU ≈ {best_iou:.4f})")




## === cell 6
test_dir = os.path.join(config.path_test, "images")
test_ids = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(".png")])
X_test = np.zeros(
    (len(test_ids), config.im_height, config.im_width, config.im_chan), dtype=np.uint8
)
sizes_test = []

print("Loading and resizing test data...")
for n, img_id in enumerate(tqdm(test_ids)):
    img_path = os.path.join(test_dir, img_id)
    sizes_test.append(
        (img.shape[0], img.shape[1])
        if (img := cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)) is not None
        else (0, 0)
    )
    img_resized = resize(
        img,
        (config.im_height, config.im_width),
        mode="constant",
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.uint8)
    X_test[n] = img_resized[..., None]
print("Done loading test data.")




## === cell 7
preds_test = model.predict(X_test, verbose=1)


def RLenc(img, order="F", format=True):
    """Run‑length encoding for binary mask."""
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
        return " ".join(f"{s} {l}" for s, l in runs)
    return runs


pred_dict = {}
kernel = np.ones((3, 3), np.uint8)  # morphological kernel for closing
for i, fn in enumerate(tqdm(test_ids)):
    mask_up = resize(
        np.squeeze(preds_test[i]),
        sizes_test[i],
        order=0,
        mode="constant",
        preserve_range=True,
        anti_aliasing=False,
    )
    bin_mask = (mask_up > threshold_best).astype(np.uint8)
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)
    pred_dict[fn[:-4]] = RLenc(bin_mask)

sub = pd.DataFrame.from_dict(pred_dict, orient="index")
sub.index.name = "id"
sub.columns = ["rle_mask"]
sub.to_csv("submission.csv")
print("Submission file saved as submission.csv")
