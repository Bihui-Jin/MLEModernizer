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
scikit-image==0.25.2
sklearn-pandas==2.2.0
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

0.8018520121159664

# 6. Current score

0.1301

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5221) has done: 'I make the notebook run in this Kaggle environment by removing the unavailable `pydensecrf` dependency and replacing it with a small, deterministic post-processing step using only installed `scikit-image` (this preserves the “post-process predicted masks” core intent without changing the upstream model). I also remove IPython-only magic (`%matplotlib inline`) and fix the broken external submission path by loading `sample_submission.csv` from the provided dataset so a valid `submission.csv` is always produced. Finally, I make NaN/empty-mask handling correct (using `pd.isna`) and ensure the output CSV has exactly the required `id,rle_mask` columns and `.csv` suffix.'
- What this solution (achieved 0.0001) has done: 'The crash comes from looking up test IDs in `depths_df` loaded from the wrong CSV: the `depths.csv` at `DATA_ROOT` contains only train IDs, so `.loc[test_ids]` raises a `KeyError`. I fix this by loading depths from the top-level `/kaggle/input/depths.csv` (which includes both train+test), and by making the test-depth lookup robust via a merge so missing IDs (if any) get a safe fill value instead of crashing. These changes are score-neutral (they only unblock inference) and preserve your existing “depth prior + simple mask cleanup” core logic. Finally, the submission-writing cell work because `sub_df` always be created.'
- What this solution (achieved 0.1301) has done: 'Your current 0.0001 score strongly suggests the submission masks are mostly empty or otherwise badly miscalibrated for the IoU-sweep metric. I keep your “depth prior + simple morphological cleanup” core logic identical, but tune only two calibration knobs that directly affect mask size/quality: (1) lower the depth-prior threshold (so predictions aren’t empty), and (2) relax the morphology cleanup slightly (so thin/fragmented salt regions aren’t deleted). I also ensure the depth binning used for train/test is consistent by reusing the same quantile edges for both, which avoids systematic bin misassignment that can degrade predictions. These changes are small, deterministic, and should move the score upward toward your target without changing the overall approach.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.io import imread
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    binary_opening,
    binary_closing,
    disk,
)

DATA_ROOT = "/kaggle/input/tgs-salt-identification-challenge"
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test", "images")
TRAIN_CSV_PATH = os.path.join(DATA_ROOT, "train.csv")

DEPTHS_PATH = "/kaggle/input/depths.csv"

SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")




## === cell 1
def rle_decode(rle_mask):
    """
    rle_mask: run-length as string formatted (start length)
    Returns numpy array (101,101), 1 - mask, 0 - background

    Note: competition encoding/decoding is in column-major order (Fortran).
    """
    if rle_mask is None or (isinstance(rle_mask, float) and np.isnan(rle_mask)):
        return np.zeros((101, 101), dtype=np.uint8)
    s = str(rle_mask).strip()
    if s == "":
        return np.zeros((101, 101), dtype=np.uint8)

    s = s.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(101 * 101, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape((101, 101), order="F")




## === cell 2
def crf(original_image, mask_img):
    """
    Replacement for DenseCRF post-processing (pydensecrf not available).
    Keeps intent: refine a predicted binary mask using image-independent cleanup.

    Input:
      original_image: unused (kept for API compatibility with original code)
      mask_img: (101,101) mask array-like

    Output:
      (101,101) uint8 array in {0,1}
    """
    m = (np.asarray(mask_img) > 0).astype(bool)

    se = disk(1)
    m = binary_closing(m, se)
    m = binary_opening(m, se)
    m = remove_small_holes(m, area_threshold=8)
    m = remove_small_objects(m, min_size=8)

    return m.astype(np.uint8)




## === cell 3
def rle_encode(im):
    """
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted.

    Kaggle expects column-major order (top-to-bottom, then left-to-right),
    which corresponds to flattening in Fortran order.
    """
    im = np.asarray(im).astype(np.uint8)

    pixels = im.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]

    if len(runs) == 0:
        return ""
    return " ".join(str(x) for x in runs)




## === cell 4
assert os.path.isdir(TEST_IMG_DIR), f"Test image directory not found: {TEST_IMG_DIR}"
assert os.path.isfile(TRAIN_CSV_PATH), f"Train CSV not found: {TRAIN_CSV_PATH}"
assert os.path.isfile(DEPTHS_PATH), f"Depths CSV not found: {DEPTHS_PATH}"
assert os.path.isfile(
    SAMPLE_SUB_PATH
), f"Sample submission not found: {SAMPLE_SUB_PATH}"

train_df = pd.read_csv(TRAIN_CSV_PATH)
depths_df = pd.read_csv(DEPTHS_PATH)

assert set(["id", "rle_mask"]).issubset(train_df.columns), "train.csv format unexpected"
assert set(["id", "z"]).issubset(depths_df.columns), "depths.csv format unexpected"

train_df["rle_mask"] = train_df["rle_mask"].replace(
    {"nan": np.nan, "NaN": np.nan, "None": np.nan}
)

train_merged = train_df.merge(depths_df, on="id", how="left")
if train_merged["z"].isna().any():
    train_merged["z"] = train_merged["z"].fillna(train_merged["z"].median())

train_merged.head()



## === cell 5
BIN_COUNT = 50  # keep same core choice
z_train = train_merged["z"].values.astype(np.float32)

quantiles = np.linspace(0.0, 1.0, BIN_COUNT + 1)
edges = np.quantile(z_train, quantiles)
for i in range(1, len(edges)):
    if edges[i] <= edges[i - 1]:
        edges[i] = edges[i - 1] + 1e-6


def z_to_bin(z_val):
    b = int(np.digitize([z_val], edges[1:-1], right=True)[0])
    if b < 0:
        b = 0
    if b >= BIN_COUNT:
        b = BIN_COUNT - 1
    return b


train_merged["z_bin"] = np.array([z_to_bin(float(z)) for z in z_train], dtype=np.int32)
n_bins = BIN_COUNT

sum_masks = np.zeros((n_bins, 101, 101), dtype=np.float32)
cnt_masks = np.zeros((n_bins,), dtype=np.int32)

for _, row in tqdm(
    train_merged.iterrows(), total=train_merged.shape[0], desc="Building depth priors"
):
    zb = int(row["z_bin"])
    m = rle_decode(row["rle_mask"]).astype(np.float32)
    sum_masks[zb] += m
    cnt_masks[zb] += 1

mean_masks = np.zeros_like(sum_masks, dtype=np.float32)
for b in range(n_bins):
    if cnt_masks[b] > 0:
        mean_masks[b] = sum_masks[b] / float(cnt_masks[b])

PRIOR_THRESH = 0.20
prior_masks = (mean_masks >= PRIOR_THRESH).astype(np.uint8)

for b in range(n_bins):
    prior_masks[b] = crf(None, prior_masks[b])

prior_masks.shape, int(cnt_masks.min()), int(cnt_masks.max())



## === cell 6
sample_df = pd.read_csv(SAMPLE_SUB_PATH)
assert set(["id", "rle_mask"]).issubset(
    sample_df.columns
), "sample_submission.csv format unexpected"

test_ids = sample_df["id"].astype(str).values

test_depths_df = pd.DataFrame({"id": test_ids}).merge(
    depths_df[["id", "z"]], on="id", how="left"
)
if test_depths_df["z"].isna().any():
    test_depths_df["z"] = test_depths_df["z"].fillna(train_merged["z"].median())
test_depths = test_depths_df["z"].values.astype(np.float32)

pred_rles = []
for img_id, z_val in tqdm(
    zip(test_ids, test_depths), total=len(test_ids), desc="Predicting test masks"
):
    b = z_to_bin(float(z_val))
    mask = prior_masks[b]

    img_path = os.path.join(TEST_IMG_DIR, img_id + ".png")
    orig_img = imread(img_path)
    refined = crf(orig_img, mask)

    pred_rles.append(rle_encode(refined))

sub_df = pd.DataFrame({"id": test_ids, "rle_mask": pred_rles})
sub_df.head()



## === cell 7
sub_df = sub_df[["id", "rle_mask"]]
sub_df.to_csv("submission.csv", index=False)

print("Wrote:", os.path.abspath("submission.csv"))
print(sub_df.head())
print(
    "Non-empty masks:",
    int((sub_df["rle_mask"].astype(str).str.len() > 0).sum()),
    "/",
    sub_df.shape[0],
)
