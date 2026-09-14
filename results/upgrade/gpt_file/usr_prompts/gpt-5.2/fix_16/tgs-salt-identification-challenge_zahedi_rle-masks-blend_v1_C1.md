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
- What this solution (achieved 0.5221) has done: 'Your score is capped because the code is currently tuning the vote threshold on train using *test* predictions (your external submissions are for test ids), so `train_merged` has mostly missing predictions and the tuned `best_k` becomes unreliable; this also means the blend loop often falls back to `df1_sorted` rather than a true ensemble. I make two minimal, metric-relevant fixes: (1) tune `best_k` directly on the same dataframe you actually predict on (the merged test dataframe `df`) by using a tiny held-out subset of train images as “pseudo-test” via their ground-truth masks and the same ensemble logic, and (2) remove the overly-strict “all four must be non-empty” gating so we can still ensemble when some sources are empty (treat missing/empty as empty masks). These changes keep your core 4-mask vote-sum ensemble and RLE encode/decode intact, but should increase recall and improve mAP toward the 0.8109 target while still producing a valid `blend_ez.csv` end-to-end.'
- What this solution (achieved 0.5221) has done: 'Your current 0.5221 score is far below the 0.8109 target, so we should cautiously improve without changing the core “4-mask vote-sum ensemble + single cutoff k + RLE encode/decode” logic. The biggest issue is that your auto-tuning of `best_k` is effectively disabled (it tunes only if test ids overlap train ids, which they don’t), so you always use the default `k=2` even if `k=1` (more recall) would score better for your ensemble sources. I tune `best_k` properly on the *training set* by reading each source’s **train** predictions from standard locations if available, or by safely defaulting to empty masks if not; then apply the chosen `best_k` to the test merge exactly as you already do. This is a minimal, metric-aligned change (just making tuning actually use train ground truth) and should move your score upward toward the target while still writing a valid `blend_ez.csv`.'
- What this solution (achieved 0.5221) has done: 'Your current 0.5221 is far below the 0.8109 target (gap −0.2888), so we should make a small, metric-aligned improvement without changing your core “4-mask vote-sum ensemble + single cutoff k + RLE encode/decode” logic. The biggest issue is that your `best_k` tuning is usually ineffective because it depends on optional `*_train.csv` files that are typically missing, causing the proxy-score to be meaningless and defaulting you to `k=2`. I keep the same tuning mechanism, but add a safe fallback that tunes `best_k` directly on the available train ground truth by constructing 4 “pseudo-predictors” from the GT mask via fixed, light morphological perturbations (no new model, still binary masks), so the chosen `k` is stable and typically favors recall (often `k=1`) which should move mAP upward. I also add one post-processing step that is standard for this competition and consistent with your semantics: remove tiny predicted components (small-area filter) to reduce false positives before RLE encoding.'
- What this solution (achieved 0.5221) has done: 'Your current gap to target is large (0.5221 → 0.8109), so we should make a small, metric-aligned improvement without changing the ensemble’s core “4 decoded masks → vote_sum → (vote_sum>=k) → RLE” logic. The biggest easy win here is calibrating the final probability-to-mask decision using a second hyperparameter: the small-component filter size, which strongly affects FP/FN balance for this mAP metric. I keep your tuning framework intact but extend it to jointly tune `(k, min_size)` on the same AP-like proxy you already implemented, then use those tuned values at inference. This is a minimal, safe change (no new models, no new data, same blending semantics) and should move score upward toward the target.'
- What this solution (achieved 0.5221) has done: 'Your current 0.5221 is far below the 0.8109 target, so the smallest meaningful gain is to stop “tuning” on synthetic/morphed ground-truth masks (which can pick an over-aggressive `min_size` and hurt real test performance) and instead use a stable, competition-standard postprocessing: keep the tuned vote cutoff `k`, but set `min_size=0` (no component filtering) unless real train prediction files exist. This preserves your core 4-mask vote-sum ensemble and your AP-like tuning framework, but prevents the fallback tuning from optimizing the wrong objective proxy and overfitting the morphology heuristic. I also add one tiny but metric-relevant improvement: if the final predicted mask is empty, write an empty string instead of encoding to an empty RLE via `rle_encode` (which can produce non-empty artifacts if anything non-binary slips in). The script still runs end-to-end and writes `blend_ez.csv` with the required columns.'
- What this solution (achieved 0.5221) has done: 'Your current 0.5221 is far below the 0.8109 target, so we should cautiously improve without changing the core “4 decoded masks → vote_sum → (vote_sum>=k) → RLE” ensemble logic. The most likely low-risk gain is post-processing: use the standard “fill small holes + remove tiny islands” cleanup on the final binary mask, and tune those two small hyperparameters on a proxy closer to the competition metric. I keep your existing tuning framework and add (1) a hole-filling step, (2) a slightly more realistic AP proxy that penalizes false positives when GT is empty by using connected-components matching, and (3) a small grid search over `hole_area` and `min_size` (only when real train prediction files exist; otherwise defaults stay conservative). This stays within your semantics (binary-mask ensembling) and should improve mAP by reducing spurious fragments and improving mask solidity.'
- What this solution (achieved 0.5221) has done: 'Your current score (0.5221) is far below the target (0.8109), so we should make a small, legitimate change that increases mAP without changing your core “4 decoded masks → vote_sum → (vote_sum>=k) → RLE” ensemble logic. The biggest likely gain is fixing the tuning stage so it does not “learn” postprocessing parameters from synthetic GT-morphology (which can pick harmful settings) when real train prediction files are absent; instead, we use a stable default in that case (no postprocess, k=2) and only tune when real train prediction CSVs exist. Additionally, we make the AP-like tuning closer to the competition by binarizing predictions per-image (already) but ensuring train ids are aligned and treated consistently as strings to avoid silent mismatches. These are minimal changes that should move the score upward toward the target while keeping your ensemble/blending approach intact and still writing a valid `blend_ez.csv`.'
- What this solution (achieved 0.5221) has done: 'Your current 0.5221 is far below the 0.8109 target, so we should increase score with the smallest metric-relevant changes while preserving your “4 decoded masks → vote_sum → (>=k) → RLE” core. The main low-risk improvement is to tune only what matters at submission time even when no train-pred CSVs exist: we keep `best_k` logic intact but add a tiny calibration step on a small deterministic train split to select a better global `k` (and optionally postprocess) **only when external submissions are present**; otherwise we keep your stable defaults. We also add a standard, minimal RLE safety fix: ensure the mask is strictly {0,1} and contiguous-type before encoding to avoid any accidental non-binary artifacts. Finally, we keep your alignment safeguards and ensure `blend_ez.csv` is always produced with exactly 1000 rows in sample order.'

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
    im = (im > 0).astype(np.uint8)

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


def ap_official_like_single_object(
    gt_mask, pred_mask, thresholds=np.arange(0.5, 1.0, 0.05)
):
    gt_any = bool(gt_mask.sum() > 0)
    pr_any = bool(pred_mask.sum() > 0)

    if (not gt_any) and (not pr_any):
        return 1.0

    if gt_any and pr_any:
        iou = iou_binary(gt_mask, pred_mask)
        return float(np.mean([(1.0 if iou > t else 0.0) for t in thresholds]))

    return 0.0


def _to_mask(v):
    if pd.notna(v) and str(v).strip() != "":
        return rle_decode(str(v))
    return np.zeros((101, 101), dtype=np.uint8)


from skimage.measure import label


def remove_small_components(mask01, min_size=15):
    m = (mask01 > 0).astype(np.uint8)
    if m.sum() == 0:
        return m
    lab = label(m, connectivity=1)
    if lab.max() == 0:
        return m
    out = np.zeros_like(m)
    for cc in range(1, lab.max() + 1):
        area = int((lab == cc).sum())
        if area >= min_size:
            out[lab == cc] = 1
    return out.astype(np.uint8)


from skimage.morphology import remove_small_holes


def postprocess_mask(mask01, min_size=0, hole_area=0):
    m = (mask01 > 0).astype(bool)
    if m.sum() == 0:
        return mask01.astype(np.uint8)
    if hole_area and int(hole_area) > 0:
        m = remove_small_holes(m, area_threshold=int(hole_area))
    out = m.astype(np.uint8)
    if min_size and int(min_size) > 0:
        out = remove_small_components(out, min_size=int(min_size))
    return out.astype(np.uint8)


def mean_ap_over_train_for_params(
    thresh_k, min_size, hole_area, df_with_preds, gt_by_id
):
    aps = []
    for _, row in df_with_preds.iterrows():
        img_id = str(row["id"])
        gt_rle = gt_by_id.get(img_id, "")
        gt_mask = _to_mask(gt_rle)

        m1 = _to_mask(row.get("rle_mask", ""))
        m2 = _to_mask(row.get("rle_mask1", ""))
        m3 = _to_mask(row.get("rle_mask2", ""))
        m4 = _to_mask(row.get("rle_mask3", ""))

        vote_sum = m1 + m2 + m3 + m4
        pred = (vote_sum >= thresh_k).astype(np.uint8)

        pred = postprocess_mask(pred, min_size=int(min_size), hole_area=int(hole_area))
        aps.append(ap_official_like_single_object(gt_mask, pred))
    if len(aps) == 0:
        return -1.0
    return float(np.mean(aps))


train_pred_candidates = {
    "rle_mask": [
        "../input/submission/crf_c_1_train.csv",
        "../working/crf_c_1_train.csv",
    ],
    "rle_mask1": ["../input/submission/crf_c_train.csv", "../working/crf_c_train.csv"],
    "rle_mask2": [
        "../input/submission/crf_c_2_train.csv",
        "../working/crf_c_2_train.csv",
    ],
    "rle_mask3": ["../input/submission/sub_train.csv", "../working/sub_train.csv"],
}


def _first_existing(paths):
    return next((p for p in paths if os.path.exists(p)), None)


train_df["id"] = train_df["id"].astype(str)
train_ids_df = train_df[["id"]].copy()

train_pred_dfs = []
n_found = 0
for col, paths in train_pred_candidates.items():
    p = _first_existing(paths)
    if p is not None:
        d = pd.read_csv(p)[["id", "rle_mask"]].copy()
        d["id"] = d["id"].astype(str)
        d = d.rename(columns={"rle_mask": col})
        n_found += 1
    else:
        d = train_ids_df.copy()
        d[col] = ""
    train_pred_dfs.append(d)

gt_by_id = dict(
    zip(train_df["id"].astype(str), train_df["rle_mask"].fillna("").astype(str))
)

if (not missing_external) and (n_found == 0):
    best_k = 2
    best_min_size = 0
    best_hole_area = 0
    best_score = float("nan")
elif n_found > 0:
    df_train_like = merge_dataframes([train_ids_df] + train_pred_dfs, merge_keys=["id"])
    df_train_like = df_train_like[
        ["id", "rle_mask", "rle_mask1", "rle_mask2", "rle_mask3"]
    ].copy()

    best_k = 2  # default majority
    best_min_size = 0
    best_hole_area = 0
    best_score = -1e9

    min_size_grid = [0, 5, 15, 30]
    hole_area_grid = [0, 8, 20, 50]

    for k in [1, 2, 3, 4]:
        for ms in min_size_grid:
            for ha in hole_area_grid:
                s = mean_ap_over_train_for_params(
                    k,
                    ms,
                    ha,
                    df_train_like[
                        ["id", "rle_mask", "rle_mask1", "rle_mask2", "rle_mask3"]
                    ],
                    gt_by_id,
                )
                if s > best_score:
                    best_score = s
                    best_k = k
                    best_min_size = ms
                    best_hole_area = ha
else:
    best_k = 2
    best_min_size = 0
    best_hole_area = 0
    best_score = float("nan")

print(
    "Tuned params:",
    "k (vote_sum>=k) =",
    best_k,
    ", min_size =",
    best_min_size,
    ", hole_area =",
    best_hole_area,
    ", mean AP-like proxy on train =",
    best_score,
    ", train_pred_files_found =",
    n_found,
)



## === cell 6
res = df1_sorted.copy()

n = len(df)
for i in range(n):
    v0 = df.iloc[i]["rle_mask"]
    v1 = df.iloc[i]["rle_mask1"]
    v2 = df.iloc[i]["rle_mask2"]
    v3 = df.iloc[i]["rle_mask3"]

    decoded_mask1 = _to_mask(v0)
    decoded_mask2 = _to_mask(v1)
    decoded_mask3 = _to_mask(v2)
    decoded_mask4 = _to_mask(v3)

    decoded_mask_all1 = decoded_mask1 + decoded_mask2 + decoded_mask3 + decoded_mask4
    decoded_mask_all1 = (decoded_mask_all1 >= best_k).astype(np.uint8)

    decoded_mask_all1 = postprocess_mask(
        decoded_mask_all1, min_size=int(best_min_size), hole_area=int(best_hole_area)
    )

    if decoded_mask_all1.sum() == 0:
        mask = ""
    else:
        mask = rle_encode(decoded_mask_all1)

    res.iloc[i, res.columns.get_loc("rle_mask")] = mask



## === cell 7
sub = pd.read_csv(sample_path)[["id"]].merge(res, on="id", how="left")
sub["rle_mask"] = sub["rle_mask"].fillna("")

sub.to_csv("blend_ez.csv", index=False)
print(
    "Wrote submission:", "blend_ez.csv", "rows:", len(sub), "cols:", list(sub.columns)
)
print("External submissions missing -> used empty-mask fallback:", missing_external)
print(
    "Using tuned params:",
    "k (vote_sum>=k) =",
    best_k,
    ", min_size =",
    best_min_size,
    ", hole_area =",
    best_hole_area,
)



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
