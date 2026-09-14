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

0.81087840761575

# 6. Current score

0.5221

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The crash happens because the notebook expects four prior submission CSVs in `../input/submission/`, but that directory/files don’t exist in your environment, so nothing downstream can run or write a submission. I keep your blending logic intact, but add a safe fallback that uses the competition’s `sample_submission.csv` as the base `df1` and copies it into `df2/df3/df4` when the external files are missing, so the pipeline runs end-to-end and always produces a valid `.csv`. I also fix the loop to iterate over the actual number of rows (1000) instead of a hardcoded 18000, and replace incorrect NaN-string checks with proper `pd.notna` checks to avoid logic errors. Finally, I write the output submission to a `.csv` file (`blend_ez.csv`) with the required columns.'
- What this solution (achieved 0.0) has done: 'I fix the crash in the depth/coverage fallback by ensuring missing `cov_pred` values are filled before converting to integers, which prevents the `NaN`→`int` error. Then I harden the external-submission loading so `df1..df4` are always valid DataFrames (never `None`), unblocking the merge and subsequent blending loop. I also keep your blending logic intact but make the merge robust by using `how="outer"` and re-aligning to `sample_submission` ids for a guaranteed 1000-row submission. Finally, the script always write a valid `blend_ez.csv` with `id,rle_mask` in the working directory.'
- What this solution (achieved 0.5221) has done: 'Your 0.0 score is coming from the “missing external submissions” fallback: it generates centered rectangles based on depth/coverage heuristics, which produces essentially random masks and tanks mAP. To move score toward your 0.8109 target with minimal change and without altering your blending core, I keep your 4-way majority-vote logic exactly as-is but change the fallback to a legitimate, competition-safe baseline: predict empty masks for all test images when the external submission files aren’t available. This produce a non-zero Kaggle score (typical empty-mask baseline is ~0.2–0.4 on this competition) and be strictly closer to the target than 0.0, while still generating a valid `blend_ez.csv` every run.'
- What this solution (achieved 0.5221) has done: 'Your current 0.5221 score is mainly limited by the “strict 3-of-4” majority vote (it drops many borderline-correct pixels), plus a likely row misalignment bug: you sort `df` but then still index it with `.loc[i]`, which is label-based and can silently pull the wrong row after sorting. I make two minimal, metric-relevant fixes: (1) iterate with `.iloc[i]` so the 4 masks and the output row are correctly aligned, and (2) relax the ensemble threshold from 3-of-4 to 2-of-4 (standard majority vote) to improve recall and typically increase mAP for this competition without changing the overall blending approach. I also ensure the merged dataframe is explicitly aligned to `sample_submission` ids before blending, preventing any accidental id-order drift. These are small changes that should move your score upward toward 0.8109 while preserving your core blending logic and producing the same `blend_ez.csv` submission.'
- What this solution (achieved 0.5221) has done: 'Your current 0.5221 score suggests the ensemble is running but the final binary masks are likely underperforming due to a fixed 0.5-ish decision boundary implicit in the source submissions. A minimal, metric-aligned improvement (without changing your ensemble/blending core) is to tune the final binarization threshold on train data using your existing RLE encode/decode: treat the 4-way vote sum (0–4) as a “confidence” map and choose the best cutoff among {1,2,3,4} by maximizing mean IoU/AP proxy on train. Then apply that single chosen cutoff to test for submission. This keeps the same decoding/encoding and majority-vote approach, but calibrates the vote threshold to this dataset, which should move the score upward toward 0.8109. I also keep the previous safety/alignment fixes intact and ensure a valid `blend_ez.csv` is always written.'
- What this solution (achieved 0.5221) has done: 'Your blending is currently limited by the vote-threshold tuning: it optimizes mean IoU on train, which is not well-aligned with the competition’s mAP over IoU thresholds and can pick a suboptimal cutoff. I keep your 4-mask vote-sum ensemble exactly the same, but tune the single `k` using a lightweight approximation of the official metric (image-wise AP averaged over IoU thresholds 0.5–0.95), computed on train for the same candidate cutoffs {1,2,3,4}. I also make one correctness fix during tuning: treat missing external predictions as empty masks (instead of skipping those rows), so the selected `k` is stable and reflects how the ensemble behaves at inference. Everything else (RLE decode/encode, merging, submission writing) stays intact and it still write `blend_ez.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import sys
import pandas as pd
from skimage.io import imread, imsave
from skimage.color import gray2rgb
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from functools import reduce
import os

print(
    os.listdir("../input")
    if os.path.exists("../input")
    else os.listdir("/kaggle/input")
)




## === cell 1
def rle_encode(im):
    """
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels = im.flatten(order="F")  # Kaggle expects top-to-bottom then left-to-right
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(rle_mask):
    """
    rle_mask: run-length as string formated (start length)
    Returns numpy array, 1 - mask, 0 - background
    """
    s = str(rle_mask).split()
    if len(s) == 0:
        return np.zeros((101, 101), dtype=np.uint8)
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(101 * 101, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape((101, 101), order="F")




## === cell 2
def merge_dataframes(dfs, merge_keys):
    dfs_merged = reduce(
        lambda left, right: pd.merge(left, right, on=merge_keys, how="outer"), dfs
    )
    return dfs_merged




## === cell 3
SAMPLE_PATHS = [
    "../input/sample_submission.csv",
    "../input/tgs-salt-identification-challenge/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
    "/kaggle/input/tgs-salt-identification-challenge/sample_submission.csv",
]
DEPTHS_PATHS = [
    "../input/depths.csv",
    "../input/tgs-salt-identification-challenge/depths.csv",
    "/kaggle/input/depths.csv",
    "/kaggle/input/tgs-salt-identification-challenge/depths.csv",
]
TRAINCSV_PATHS = [
    "../input/train.csv",
    "../input/tgs-salt-identification-challenge/train.csv",
    "/kaggle/input/train.csv",
    "/kaggle/input/tgs-salt-identification-challenge/train.csv",
]

sample_path = next((p for p in SAMPLE_PATHS if os.path.exists(p)), None)
depths_path = next((p for p in DEPTHS_PATHS if os.path.exists(p)), None)
traincsv_path = next((p for p in TRAINCSV_PATHS if os.path.exists(p)), None)

if sample_path is None:
    raise FileNotFoundError(
        "Could not find sample_submission.csv in expected Kaggle input paths."
    )
if depths_path is None:
    raise FileNotFoundError("Could not find depths.csv in expected Kaggle input paths.")
if traincsv_path is None:
    raise FileNotFoundError("Could not find train.csv in expected Kaggle input paths.")


def _safe_read_csv(path, rename_to=None):
    if os.path.exists(path):
        df = pd.read_csv(path)
        if rename_to is not None and "rle_mask" in df.columns:
            df = df.rename(columns={"rle_mask": rename_to})
        return df
    return None


df1 = _safe_read_csv("../input/submission/crf_c_1.csv", rename_to="rle_mask")
df2 = _safe_read_csv("../input/submission/crf_c.csv", rename_to="rle_mask1")
df3 = _safe_read_csv("../input/submission/crf_c_2.csv", rename_to="rle_mask2")
df4 = _safe_read_csv("../input/submission/sub.csv", rename_to="rle_mask3")

base = pd.read_csv(sample_path)[["id", "rle_mask"]].copy()
depths = pd.read_csv(depths_path)[["id", "z"]].copy()
train_df = pd.read_csv(traincsv_path)[["id", "rle_mask"]].copy()

missing_external = (df1 is None) and (df2 is None) and (df3 is None) and (df4 is None)

if missing_external:
    pseudo = base[["id"]].copy()
    pseudo["rle_mask"] = ""  # empty mask baseline

    df1 = pseudo[["id", "rle_mask"]].copy()
    df2 = pseudo.rename(columns={"rle_mask": "rle_mask1"})[["id", "rle_mask1"]].copy()
    df3 = pseudo.rename(columns={"rle_mask": "rle_mask2"})[["id", "rle_mask2"]].copy()
    df4 = pseudo.rename(columns={"rle_mask": "rle_mask3"})[["id", "rle_mask3"]].copy()
else:
    if df1 is None:
        df1 = base.copy()  # keep column name rle_mask
    if df2 is None:
        df2 = base.rename(columns={"rle_mask": "rle_mask1"}).copy()
    if df3 is None:
        df3 = base.rename(columns={"rle_mask": "rle_mask2"}).copy()
    if df4 is None:
        df4 = base.rename(columns={"rle_mask": "rle_mask3"}).copy()

df1 = df1[["id", "rle_mask"]].copy()
df2 = df2[["id", "rle_mask1"]].copy()
df3 = df3[["id", "rle_mask2"]].copy()
df4 = df4[["id", "rle_mask3"]].copy()



## === cell 4
dfs = [df1, df2, df3, df4]
merge_keys = ["id"]
df = merge_dataframes(dfs, merge_keys=merge_keys)

sample_ids = pd.read_csv(sample_path)[["id"]].copy()
df = sample_ids.merge(df, on="id", how="left").reset_index(drop=True)

df1_sorted = sample_ids.merge(df1, on="id", how="left").reset_index(drop=True)

df.head()




## === cell 5
def iou_binary(y_true, y_pred):
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    inter = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    if union == 0:
        return 1.0  # both empty
    return inter / union


def ap_like_single_mask(gt_mask, pred_mask, thresholds=np.arange(0.5, 1.0, 0.05)):
    """
    Minimal, competition-aligned approximation for this dataset setup (single binary mask):
    for each IoU threshold, precision is 1 if IoU>t else 0, except both-empty => precision 1.
    Then average across thresholds.
    """
    gt_any = bool(gt_mask.sum() > 0)
    pr_any = bool(pred_mask.sum() > 0)
    if (not gt_any) and (not pr_any):
        return 1.0
    if gt_any != pr_any:
        return 0.0
    iou = iou_binary(gt_mask, pred_mask)
    return float(np.mean([(1.0 if iou > t else 0.0) for t in thresholds]))


def mean_ap_over_train(thresh_k, train_merged):
    aps = []
    for _, row in train_merged.iterrows():
        gt = row["rle_mask_gt"]
        if not (pd.notna(gt) and str(gt).strip() != ""):
            gt_mask = np.zeros((101, 101), dtype=np.uint8)
        else:
            gt_mask = rle_decode(str(gt))

        v0, v1, v2, v3 = (
            row.get("rle_mask", ""),
            row.get("rle_mask1", ""),
            row.get("rle_mask2", ""),
            row.get("rle_mask3", ""),
        )

        def _to_mask(v):
            if pd.notna(v) and str(v).strip() != "":
                return rle_decode(str(v))
            return np.zeros((101, 101), dtype=np.uint8)

        m1 = _to_mask(v0)
        m2 = _to_mask(v1)
        m3 = _to_mask(v2)
        m4 = _to_mask(v3)

        vote_sum = m1 + m2 + m3 + m4
        pred = (vote_sum >= thresh_k).astype(np.uint8)

        aps.append(ap_like_single_mask(gt_mask, pred))

    if len(aps) == 0:
        return -1.0
    return float(np.mean(aps))


train_ids = train_df[["id"]].copy()
train_gt = train_df.rename(columns={"rle_mask": "rle_mask_gt"})[
    ["id", "rle_mask_gt"]
].copy()

train_pred = merge_dataframes([df1, df2, df3, df4], merge_keys=["id"])
train_merged = train_ids.merge(train_gt, on="id", how="left").merge(
    train_pred, on="id", how="left"
)

best_k = 2  # default majority
best_score = -1e9
for k in [1, 2, 3, 4]:
    s = mean_ap_over_train(k, train_merged)
    if s > best_score:
        best_score = s
        best_k = k

print(
    "Tuned vote threshold k (vote_sum>=k):",
    best_k,
    "mean AP-like proxy on train:",
    best_score,
)



## === cell 6
res = df1_sorted.copy()

n = len(df)
for i in range(n):
    v0 = df.iloc[i]["rle_mask"]
    v1 = df.iloc[i]["rle_mask1"]
    v2 = df.iloc[i]["rle_mask2"]
    v3 = df.iloc[i]["rle_mask3"]

    if (
        pd.notna(v0)
        and pd.notna(v1)
        and pd.notna(v2)
        and pd.notna(v3)
        and str(v0).strip() != ""
        and str(v1).strip() != ""
        and str(v2).strip() != ""
        and str(v3).strip() != ""
    ):
        decoded_mask1 = rle_decode(str(v0))
        decoded_mask2 = rle_decode(str(v1))
        decoded_mask3 = rle_decode(str(v2))
        decoded_mask4 = rle_decode(str(v3))

        decoded_mask_all1 = (
            decoded_mask1 + decoded_mask2 + decoded_mask3 + decoded_mask4
        )

        decoded_mask_all1 = (decoded_mask_all1 >= best_k).astype(np.uint8)

        mask = rle_encode(decoded_mask_all1.astype(np.uint8))
        res.iloc[i, res.columns.get_loc("rle_mask")] = mask
    else:
        res.iloc[i, res.columns.get_loc("rle_mask")] = df1_sorted.iloc[i]["rle_mask"]



## === cell 7
sub = pd.read_csv(sample_path)[["id"]].merge(res, on="id", how="left")
sub["rle_mask"] = sub["rle_mask"].fillna("")

sub.to_csv("blend_ez.csv", index=False)
print(
    "Wrote submission:", "blend_ez.csv", "rows:", len(sub), "cols:", list(sub.columns)
)
print("External submissions missing -> used empty-mask fallback:", missing_external)



## === cell 8
dft = sub
i = 0
plt.figure(figsize=(25, 12))
plt.subplots_adjust(bottom=0.3, top=0.9, hspace=0.3)
j = 0
while True:
    if i >= len(dft):
        break
    if pd.notna(dft.loc[i, "rle_mask"]) and str(dft.loc[i, "rle_mask"]).strip() != "":
        decoded_mask = rle_decode(str(dft.loc[i, "rle_mask"]))
        plt.subplot(1, 7, j + 1)
        plt.imshow(decoded_mask, cmap="gray")
        plt.title(" ID: " + str(dft.loc[i, "id"]))
        j = j + 1
        if j > 6:
            break
    i = i + 1
plt.show()
