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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.5221) has done: 'I make the notebook run in this Kaggle environment by removing the unavailable `pydensecrf` dependency and replacing it with a small, deterministic post-processing step using only installed `scikit-image` (this preserves the “post-process predicted masks” core intent without changing the upstream model). I also remove IPython-only magic (`%matplotlib inline`) and fix the broken external submission path by loading `sample_submission.csv` from the provided dataset so a valid `submission.csv` is always produced. Finally, I make NaN/empty-mask handling correct (using `pd.isna`) and ensure the output CSV has exactly the required `id,rle_mask` columns and `.csv` suffix.'

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
DEPTHS_PATH = os.path.join(DATA_ROOT, "depths.csv")
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
    m = remove_small_holes(m, area_threshold=16)
    m = remove_small_objects(m, min_size=16)

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

BIN_COUNT = 50  # small enough for stability, large enough to capture depth trend
train_merged["z_bin"] = pd.qcut(
    train_merged["z"], q=BIN_COUNT, labels=False, duplicates="drop"
)
n_bins = int(train_merged["z_bin"].max() + 1)

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

PRIOR_THRESH = 0.35
prior_masks = (mean_masks >= PRIOR_THRESH).astype(np.uint8)

for b in range(n_bins):
    prior_masks[b] = crf(None, prior_masks[b])

prior_masks.shape, cnt_masks.min(), cnt_masks.max()



## === cell 6

sample_df = pd.read_csv(SAMPLE_SUB_PATH)
assert set(["id", "rle_mask"]).issubset(
    sample_df.columns
), "sample_submission.csv format unexpected"

test_ids = sample_df["id"].values
test_depths = depths_df.set_index("id").loc[test_ids, "z"].values

z_train = train_merged["z"].values.astype(np.float32)
quantiles = np.linspace(0.0, 1.0, n_bins + 1)
edges = np.quantile(z_train, quantiles)
for i in range(1, len(edges)):
    if edges[i] <= edges[i - 1]:
        edges[i] = edges[i - 1] + 1e-6


def z_to_bin(z_val):
    b = int(np.digitize([z_val], edges[1:-1], right=True)[0])
    if b < 0:
        b = 0
    if b >= n_bins:
        b = n_bins - 1
    return b


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



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1808548452.py in <cell line: 0>()
      8 
      9 test_ids = sample_df["id"].values
---> 10 test_depths = depths_df.set_index("id").loc[test_ids, "z"].values
     11 
     12 # Use the same binning scheme: compute bin edges from train z distribution for deterministic mapping.

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in __getitem__(self, key)
   1182             if self._is_scalar_access(key):
   1183                 return self.obj._get_value(*key, takeable=self._takeable)
-> 1184             return self._getitem_tuple(key)
   1185         else:
   1186             # we by definition only have the 0th axis

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_tuple(self, tup)
   1366         with suppress(IndexingError):
   1367             tup = self._expand_ellipsis(tup)
-> 1368             return self._getitem_lowerdim(tup)
   1369 
   1370         # no multi-index, so validate all of the indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_lowerdim(self, tup)
   1087                     return section
   1088                 # This is an elided recursive call to iloc/loc
-> 1089                 return getattr(section, self.name)[new_key]
   1090 
   1091         raise IndexingError("not applicable")

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in __getitem__(self, key)
   1189             maybe_callable = com.apply_if_callable(key, self.obj)
   1190             maybe_callable = self._check_deprecated_callable_usage(key, maybe_callable)
-> 1191             return self._getitem_axis(maybe_callable, axis=axis)
   1192 
   1193     def _is_scalar_access(self, key: tuple):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_axis(self, key, axis)
   1418                     raise ValueError("Cannot index with multidimensional key")
   1419 
-> 1420                 return self._getitem_iterable(key, axis=axis)
   1421 
   1422             # nested tuple slicing

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_iterable(self, key, axis)
   1358 
   1359         # A collection of keys
-> 1360         keyarr, indexer = self._get_listlike_indexer(key, axis)
   1361         return self.obj._reindex_with_indexers(
   1362             {axis: [keyarr, indexer]}, copy=True, allow_dups=True

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _get_listlike_indexer(self, key, axis)
   1556         axis_name = self.obj._get_axis_name(axis)
   1557 
-> 1558         keyarr, indexer = ax._get_indexer_strict(key, axis_name)
   1559 
   1560         return keyarr, indexer

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6247         if nmissing:
   6248             if nmissing == len(indexer):
-> 6249                 raise KeyError(f"None of [{key}] are in the [{axis_name}]")
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())

KeyError: "None of [Index(['003c477d7c', '0108518d1e', '010ee525b6', '01323211a0', '026b509cd4',\n       '0280deb8ae', '02d40993ab', '03a9a9f615', '03c9b5ffe8', '04234be388',\n       ...\n       'fdb7d132be', 'fe801cdc47', 'fe9e558605', 'ff038e87b1', 'ff532274cd',\n       'ff89e5e3cd', 'ff9b7d0d2e', 'ffce5bbb9a', 'ffe228a7e3', 'fff2abcaf9'],\n      dtype='object', name='id', length=1000)] are in the [index]"

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

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2441169613.py in <cell line: 0>()
      1 # Why change: ensure we always write a valid Kaggle submission with required columns and .csv suffix.
----> 2 sub_df = sub_df[["id", "rle_mask"]]
      3 sub_df.to_csv("submission.csv", index=False)
      4 
      5 print("Wrote:", os.path.abspath("submission.csv"))

NameError: name 'sub_df' is not defined
