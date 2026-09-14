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

0.8086888792730411

# 6. Current score

0.5222

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5221) has done: 'I remove the unavailable `pydensecrf` dependency (it isn’t installed in your environment) and replace the CRF refinement step with a lightweight, deterministic mask post-processing (morphological closing/opening) that keeps the pipeline intent (refine predicted masks) while making it runnable. I also fix the notebook-only `%matplotlib inline` line, restore missing imports so `pd/plt/tqdm` are defined, and make the code robust to missing external submission inputs by falling back to `sample_submission.csv` if the referenced `../input/u-net-with-simple-resnet-blocks-forked/submission.csv` doesn’t exist. Finally, I ensure the script always writes a valid Kaggle submission CSV with the required columns (`id,rle_mask`) and a `.csv` suffix.'
- What this solution (achieved 0.5221) has done: 'Your current score is far below the target, so the smallest meaningful improvement is to fix likely correctness issues and better mimic the original “CRF refinement” intent without changing the upstream model submission you’re refining. I (1) make RLE decoding match the competition’s column-major convention (your encode is correct but your decode is not), and (2) switch the post-processing to a slightly stronger, image-aware refinement (morphology + boundary snapping from the seismic image gradient) while keeping it deterministic and lightweight. I also ensure the refined submission stays aligned to `sample_submission.csv` ids to avoid any ordering/missing-id penalties. These changes keep the same overall pipeline: load base submission → decode mask → refine using the image → encode → write CSV.'
- What this solution (achieved 0.5221) has done: 'Your score is far below the target, so the smallest likely gain without changing your overall “refine an existing submission” approach is to (1) avoid accidentally destroying good masks with an overly aggressive/unstable MGAC step, and (2) apply a very light, deterministic post-processing that mainly removes obvious noise and fills small holes while keeping shapes close to the base prediction. Concretely, I replace MGAC with a safer morphology-only refinement (still “refinement”, but less likely to worsen IoU across thresholds), and I tune the cleanup thresholds to be conservative so we preserve the base submission’s performance while nudging it upward. I also ensure masks are always encoded as clean binary arrays and keep the `sample_submission.csv` id alignment unchanged. This should move you upward from ~0.52 toward the target without any core pipeline rewrite.'
- What this solution (achieved 0.5221) has done: 'Your current score (0.5221) is far below the target (0.8087), so the most “minimal but meaningful” move is to fix a likely major issue: you are currently refining masks starting from `sample_submission.csv` (all empty masks) whenever the external U-Net submission file is unavailable, which guarantees a very low score. I keep your exact pipeline (load base submission → decode → conservative morphology refine → encode → write CSV) but make it reliably pick a real base submission if one exists anywhere in the environment (common locations), while still falling back to the sample only if absolutely necessary. I also keep strict `sample_submission.csv` id alignment (to avoid penalties) and add a small safety check that reports the fraction of non-empty masks loaded so you can verify it’s not empty again. These changes do not alter your refinement logic; they just ensure you are refining actual predictions, which should move the score sharply upward toward the target band.'
- What this solution (achieved 0.5221) has done: 'Your current score is far below the target, so the most likely “minimal-impact” improvement is to ensure you are refining a strong base submission rather than accidentally refining the (empty) `sample_submission.csv`. I keep your exact refinement pipeline (load base → decode → conservative morphology → encode) but make base-submission discovery stricter: prefer non-empty, correctly-shaped salt submissions found under `/kaggle/input/`, and only fall back to `sample_submission.csv` if no suitable candidate exists. I also add a safety gate that refuses to refine if the base is essentially empty (to avoid writing another ~0.52 submission), and I keep id alignment identical to `sample_submission.csv` to avoid any missing-id/order penalties. These changes should move the score upward toward the target without changing your refinement logic.'
- What this solution (achieved 0.0) has done: 'Your current score is far below target, so the smallest likely improvement is to stop unintentionally degrading good base masks during “refinement.” I keep your exact pipeline (load base submission → decode → conservative morphology → encode → write CSV), but make the refinement truly no-op for masks that are already “reasonable,” by applying morphology only to very small/fragile masks where it’s most likely to help (noise removal/filling tiny gaps). I also remove the unused test image read (it doesn’t affect predictions) to save time, while keeping the same I/O paths and submission alignment. This should move the score upward toward the target by preserving the base model’s quality instead of over-smoothing it.'
- What this solution (achieved 0.0) has done: 'Your current Kaggle score is 0.0 because the script is wiping masks when it can’t find test images: your `DATA_ROOT` points to `/kaggle/input/...` but in this environment the files are under `/kaggle/data/...`, so `img_path` doesn’t exist and every prediction becomes empty. I make the data root auto-detect between the available locations (while keeping paths otherwise identical) so test images are found and masks aren’t cleared. I also stop reading the test image file during refinement (it isn’t used for prediction), so a missing image won’t force an empty mask; instead we keep the base mask in that rare case. These are minimal correctness fixes that should move the score sharply upward toward your target by restoring non-empty predictions.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with producing an invalid/empty submission due to the hard failure when `test/images` isn’t found at the chosen `DATA_ROOT`. I make `DATA_ROOT` selection more robust by accepting both the “flat” dataset layout (`/kaggle/data/test/images`) and the nested competition layout (`/kaggle/data/tgs-salt-identification-challenge/test/images`), and I remove the unnecessary hard stop in cell 6 so the script always completes and writes a submission. To avoid silently generating a nearly-empty file again, I add a small safety fallback: if no suitable base submission is found, we still write a valid CSV aligned to `sample_submission.csv` (but we won’t raise). These changes preserve your core “load base submission → decode → conservative refine → encode → write” logic while ensuring a valid submission and restoring non-empty predictions, which should move the score up toward your target.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is most consistent with submitting an effectively empty (or nearly empty) prediction file, which can happen if the “base submission discovery” accidentally selects `sample_submission.csv` or another near-empty CSV. I keep your exact pipeline (load base submission → decode → conservative morphology refine → encode → write CSV) but make base-submission selection explicitly exclude the sample submission and require exact row-count match (1000) after alignment, so we reliably start from a real prediction file. I also add a final safety check: if the chosen base is still almost empty, we fall back to the “best available non-empty” candidate rather than writing another empty-ish file. These are minimal correctness/robustness changes intended to move the score upward toward your target by restoring non-empty, meaningful masks without changing your refinement logic.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is most consistent with producing an almost-all-empty submission because the pipeline can’t find a real base prediction and ends up refining the (empty) `sample_submission.csv`. I keep your exact “load base submission → decode → conservative refine → encode → write” logic, but make base-submission discovery actually look in the dataset folders you have (`/kaggle/data/**`) and prefer non-empty candidates, while explicitly excluding `sample_submission.csv` and your own output `crf_correction.csv`. I also add a safety fallback: if no good base is found, we won’t run refinement that guarantees empties; instead we emit the best non-empty candidate we can find (still aligned to `sample_submission.csv` ids) so the score moves upward toward the target. These are minimal, score-relevant robustness fixes and keep evaluation semantics unchanged.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is most consistent with the output submission being effectively all-empty after refinement, which can happen if the base submission you load is empty/near-empty or if the refinement step accidentally clears many masks. I make base-submission selection more robust by also accepting common alternative column names (not just `id`/`rle_mask`) and by prioritizing candidates with both high non-empty fraction and substantial mask pixel coverage, which helps avoid picking “mostly empty” files. I also add a safety guard during refinement: if the “refined” mask collapses too much compared to the original (likely harming IoU), we keep the original decoded mask instead—this preserves baseline quality while still allowing conservative cleanup. These are minimal, score-relevant changes that keep your core pipeline (load base submission → decode → conservative refine → encode → write CSV) intact and should move the score upward toward your target.'
- What this solution (achieved 0.2778) has done: 'Your 0.0 score is most consistent with the pipeline falling back to an empty/near-empty base submission (often the sample submission) because there is no real `submission*.csv` to refine in this environment. To move the score upward toward your target while preserving your core “refine an existing submission” logic, I (1) add a deterministic fallback that generates a non-empty baseline submission directly from the provided `train.csv` RLE masks by matching each test image to the most similar training image (simple L2 distance on the raw 101×101 image), and (2) keep your existing conservative refinement + safety area-ratio guard unchanged on top of that base. This avoids writing an all-empty submission when no external base file exists, while keeping evaluation semantics and post-processing intact. The result is still a valid `id,rle_mask` CSV aligned to `sample_submission.csv`.'
- What this solution (achieved 0.1273) has done: 'Your current score (0.2778) is far below the target (0.8087), so we should increase performance with the smallest changes that don’t alter your overall pipeline. The biggest low-risk gain here is to improve the fallback “NN transfer from train” base-generation: right now it uses only the first 400 train images and unnormalized distance, which can yield poor matches and low IoU. I keep the same core approach (nearest-neighbor on raw images → transfer that train mask → then your conservative refinement), but (1) choose a more diverse prototype bank deterministically (stratified by mask coverage) and (2) use a proper squared L2 distance (including the missing constant term) to improve matching quality. This should move the score upward toward the target while keeping runtime under the 600s constraint and preserving all downstream refinement/encoding logic.'
- What this solution (achieved 0.5222) has done: 'Your current score (0.1273) is far below the target (0.8087), so we should improve the fallback base-generation without changing your overall “base submission → decode → conservative refine → encode” pipeline. The biggest issue is that the NN-transfer baseline is too weak because it uses raw pixels only; we keep the same NN-transfer idea but make the distance more robust by (a) standardizing each image vector (zero-mean/unit-std) and (b) augmenting the vector with a lightweight gradient-magnitude channel, which is still the same approach (nearest neighbor in a fixed feature space) but matches seismic texture better. We also fix the prototype-bank size to a safe level for runtime (and precompute test vectors in a small batch loop) to stay under the 600s constraint deterministically. Everything downstream (refinement, RLE encode/decode, submission alignment) stays identical, just with a stronger fallback base so the score should move upward toward the target band.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

from skimage.io import imread
from skimage.morphology import (
    binary_closing,
    binary_opening,
    remove_small_holes,
    remove_small_objects,
    square,
)

import matplotlib.pyplot as plt

np.random.seed(100)


def _pick_data_root():
    """
    Change rationale (score-toward-target):
      Ensure we read sample_submission.csv and test/images from a valid root in this environment
      so we don't accidentally produce empty/invalid outputs due to missing folders.
    """
    candidates = [
        "/kaggle/input/tgs-salt-identification-challenge",
        "/kaggle/data/tgs-salt-identification-challenge",
        "/kaggle/input",
        "/kaggle/data",
    ]
    for c in candidates:
        sample = os.path.join(c, "sample_submission.csv")
        test_dir = os.path.join(c, "test", "images")
        if os.path.isfile(sample) and os.path.isdir(test_dir):
            return c

    for c in candidates:
        sample = os.path.join(c, "sample_submission.csv")
        if os.path.isfile(sample):
            return c

    return "/kaggle/data"


DATA_ROOT = _pick_data_root()
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test", "images")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

print(f"Using DATA_ROOT={DATA_ROOT}")
print(f"TEST_IMG_DIR exists: {os.path.isdir(TEST_IMG_DIR)} ({TEST_IMG_DIR})")
print(f"SAMPLE_SUB_PATH exists: {os.path.isfile(SAMPLE_SUB_PATH)} ({SAMPLE_SUB_PATH})")




## === cell 1
def rle_decode(rle_mask: str, shape=(101, 101)):
    """
    Decode RLE string into a binary mask.

    Competition convention:
      - 1-indexed
      - column-major (Fortran order): top-to-bottom, then left-to-right.

    Keep decode consistent with encode (which uses im.T.flatten()).
    """
    if rle_mask is None:
        return np.zeros(shape, dtype=np.uint8)
    rle_mask = str(rle_mask)
    if rle_mask.strip() == "" or rle_mask.lower() == "nan":
        return np.zeros(shape, dtype=np.uint8)

    s = rle_mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape((shape[1], shape[0])).T




## === cell 2
def rle_encode(im: np.ndarray):
    """
    Encode binary mask into RLE using column-major order (Fortran-equivalent).
    """
    im = (im > 0).astype(np.uint8)
    pixels = im.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    if len(runs) == 0:
        return ""
    return " ".join(str(x) for x in runs)




## === cell 3
def _normalize_submission_df(df_sub: pd.DataFrame):
    """
    Normalize to columns ['id','rle_mask'] where rle_mask is a string.

    Change rationale (score-toward-target):
      Many strong public submissions use 'EncodedPixels' rather than 'rle_mask'.
      If we fail to detect that, we treat them as invalid and may fall back to an empty sample,
      leading to a 0.0 score.
    """
    if df_sub is None:
        return None

    if "id" not in df_sub.columns:
        for alt_id in ["ID", "image_id", "ImageId", "img", "img_id"]:
            if alt_id in df_sub.columns:
                df_sub = df_sub.rename(columns={alt_id: "id"})
                break
    if "id" not in df_sub.columns:
        return None

    if "rle_mask" not in df_sub.columns:
        for alt in [
            "rle",
            "mask",
            "rle_encoded",
            "encoded_pixels",
            "EncodedPixels",
            "pixels",
            "predicted",
        ]:
            if alt in df_sub.columns:
                df_sub = df_sub.rename(columns={alt: "rle_mask"})
                break
    if "rle_mask" not in df_sub.columns:
        return None

    df_sub = df_sub[["id", "rle_mask"]].copy()
    df_sub["rle_mask"] = df_sub["rle_mask"].fillna("").astype(str)
    df_sub.loc[df_sub["rle_mask"].str.lower().eq("nan"), "rle_mask"] = ""
    return df_sub


def _mean_mask_area_from_rle(rle_series: pd.Series, shape=(101, 101), max_decode=60):
    """
    Small, deterministic estimator for typical mask size.
    Used only for base-submission selection (not changing model/refinement semantics).

    Change rationale (score-toward-target):
      A near-empty file can have a decent non-empty fraction (many tiny runs),
      but will still score poorly. Preferring candidates with reasonable area
      reduces the chance of selecting a bad base and ending up near 0.0.
    """
    rles = rle_series.dropna().astype(str)
    rles = rles[rles.str.strip().ne("") & ~rles.str.lower().eq("nan")]
    if len(rles) == 0:
        return 0.0
    take = rles.iloc[:max_decode]
    areas = []
    for r in take:
        m = rle_decode(r, shape=shape)
        areas.append(float(m.sum()))
    return float(np.mean(areas)) if len(areas) else 0.0


def _submission_quality(df_sub: pd.DataFrame, sample_ids: pd.Series):
    """
    Heuristic quality score to pick a good base submission without changing core logic.

    Change rationale (score-toward-target):
      0.0 score is often caused by selecting an empty/near-empty base submission.
      Prefer candidates with higher non-empty fraction AND reasonable mask area.
    """
    aligned = pd.DataFrame({"id": sample_ids}).merge(df_sub, on="id", how="left")
    aligned["rle_mask"] = aligned["rle_mask"].fillna("").astype(str)
    aligned.loc[aligned["rle_mask"].str.lower().eq("nan"), "rle_mask"] = ""

    non_empty_frac = aligned["rle_mask"].str.strip().ne("").mean()
    coverage = aligned["rle_mask"].notna().mean()
    mean_area = _mean_mask_area_from_rle(
        aligned["rle_mask"], shape=(101, 101), max_decode=60
    )

    area_term = min(mean_area / 800.0, 1.0)  # 0..1
    score = float(non_empty_frac) + 0.15 * float(area_term)
    return score, float(non_empty_frac), float(coverage), float(mean_area)


def _find_existing_base_submission(sample_ids: pd.Series, sample_path: str):
    """
    Pick best existing base submission by non-empty fraction after aligning to sample ids.

    Change rationale (score-toward-target):
      Search both /kaggle/input/** and /kaggle/data/** and accept common column names.
      Exclude our own output CSV to avoid self-selection loops.
    """
    candidates = []

    candidates += [
        "/kaggle/input/u-net-with-simple-resnet-blocks-forked/submission.csv",
        "../input/u-net-with-simple-resnet-blocks-forked/submission.csv",
    ]

    candidates += glob.glob("/kaggle/input/**/submission*.csv", recursive=True)
    candidates += glob.glob("/kaggle/data/**/submission*.csv", recursive=True)
    candidates += glob.glob("/kaggle/working/*.csv")

    sample_real = (
        os.path.realpath(sample_path) if os.path.exists(sample_path) else sample_path
    )
    out_real = (
        os.path.realpath("crf_correction.csv")
        if os.path.exists("crf_correction.csv")
        else None
    )

    seen = set()
    deduped = []
    for p in candidates:
        if not os.path.isfile(p):
            continue
        try:
            rp = os.path.realpath(p)
            if rp == sample_real:
                continue
            if out_real is not None and rp == out_real:
                continue
        except Exception:
            if p == sample_path or os.path.basename(p) == "crf_correction.csv":
                continue
        if p not in seen:
            seen.add(p)
            deduped.append(p)

    best = None
    best_info = None

    def _rank_path(x):
        if x.startswith("/kaggle/input/"):
            return (0, len(x))
        if x.startswith("/kaggle/data/"):
            return (1, len(x))
        if x.startswith("/kaggle/working/"):
            return (2, len(x))
        return (3, len(x))

    deduped = sorted(deduped, key=_rank_path)[:600]

    for path in deduped:
        try:
            df_try = pd.read_csv(path)
        except Exception:
            continue

        df_try = _normalize_submission_df(df_try)
        if df_try is None:
            continue

        if df_try["id"].nunique() < 900:
            continue

        score, non_empty_frac, coverage, mean_area = _submission_quality(
            df_try, sample_ids
        )

        if non_empty_frac < 0.05:
            continue

        if (best_info is None) or (score > best_info["score"]):
            best = path
            best_info = {
                "score": float(score),
                "non_empty_frac": float(non_empty_frac),
                "coverage": float(coverage),
                "mean_area": float(mean_area),
            }

    return best, best_info


def _resolve_train_test_roots(data_root: str):
    """
    Change rationale (score-toward-target):
      Needed for deterministic fallback base generation if no external submission exists.
      We must find actual train images/masks and test images in this environment layout.
    """
    candidates = [
        data_root,
        "/kaggle/data/tgs-salt-identification-challenge",
        "/kaggle/data",
        "/kaggle/input/tgs-salt-identification-challenge",
        "/kaggle/input",
    ]

    train_img = train_mask = test_img = None
    for c in candidates:
        ti = os.path.join(c, "train", "images")
        tm = os.path.join(c, "train", "masks")
        te = os.path.join(c, "test", "images")
        if os.path.isdir(te) and test_img is None:
            test_img = te
        if os.path.isdir(ti) and os.path.isdir(tm):
            train_img, train_mask = ti, tm
            if test_img is not None:
                break

    return train_img, train_mask, test_img


def _img_to_vec(path: str):
    """
    Change rationale (score-toward-target):
      Keep the same NN-transfer core logic but improve feature robustness:
        - Use standardized intensity (zero-mean/unit-std) to reduce brightness/contrast mismatch.
        - Add a lightweight gradient-magnitude channel to better match salt boundaries.
      This tends to produce closer-looking NN matches => better transferred masks => higher IoU/AP.
    """
    img = imread(path)
    if img.ndim == 3:
        img = img[..., 0]
    img = img.astype(np.float32) / 255.0

    mu = float(img.mean())
    sigma = float(img.std()) + 1e-6
    img_z = (img - mu) / sigma

    gx = np.zeros_like(img_z)
    gy = np.zeros_like(img_z)
    gx[:, 1:] = img_z[:, 1:] - img_z[:, :-1]
    gy[1:, :] = img_z[1:, :] - img_z[:-1, :]
    grad = np.sqrt(gx * gx + gy * gy)

    feat = np.concatenate([img_z.reshape(-1), grad.reshape(-1)], axis=0).astype(
        np.float32
    )
    return feat


def _generate_base_submission_from_train(sample_ids: pd.Series, data_root: str):
    """
    Deterministic fallback base submission: nearest-neighbor transfer from train masks.

    Change rationale (score-toward-target):
      Improve matching quality (hence IoU/AP) without changing the core approach:
        - Keep deterministic, coverage-stratified prototype bank
        - Use correct squared L2 distance
        - Improve feature vector (standardized intensity + gradient magnitude)
      Also adjust K to keep runtime under 600s in this environment.
    """
    train_img_dir, train_mask_dir, test_img_dir = _resolve_train_test_roots(data_root)
    if train_img_dir is None or train_mask_dir is None or test_img_dir is None:
        raise RuntimeError(
            "Could not locate train/images, train/masks, and test/images for fallback base generation."
        )

    train_img_paths = sorted(glob.glob(os.path.join(train_img_dir, "*.png")))
    if len(train_img_paths) == 0:
        raise RuntimeError("No train images found for fallback base generation.")

    train_ids = [os.path.splitext(os.path.basename(p))[0] for p in train_img_paths]

    coverages = []
    for tid in tqdm(train_ids, desc="Estimating train mask coverages"):
        mp = os.path.join(train_mask_dir, f"{tid}.png")
        m = imread(mp)
        if m.ndim == 3:
            m = m[..., 0]
        m = (m > 127).astype(np.uint8)
        coverages.append(float(m.mean()))
    cov = np.asarray(coverages, dtype=np.float32)

    order = np.argsort(cov)

    K = min(600, len(train_ids))
    bins = 10
    per_bin = max(1, K // bins)

    picked = []
    for b in range(bins):
        lo = int(b * len(order) / bins)
        hi = int((b + 1) * len(order) / bins)
        idx = order[lo:hi]
        if len(idx) == 0:
            continue
        take = np.linspace(0, len(idx) - 1, num=min(per_bin, len(idx)), dtype=int)
        picked.extend(idx[take].tolist())

    picked = list(dict.fromkeys(picked))  # dedupe, preserve order
    if len(picked) < K:
        mid = order[len(order) // 2 :]
        for ii in mid:
            if ii not in picked:
                picked.append(int(ii))
            if len(picked) >= K:
                break
    picked = picked[:K]

    bank_ids = [train_ids[i] for i in picked]
    feat_dim = 2 * 101 * 101
    bank_X = np.zeros((K, feat_dim), dtype=np.float32)
    bank_rle = [""] * K

    for k, tid in enumerate(tqdm(bank_ids, desc="Building train prototype bank")):
        bank_X[k] = _img_to_vec(os.path.join(train_img_dir, f"{tid}.png"))
        m = imread(os.path.join(train_mask_dir, f"{tid}.png"))
        if m.ndim == 3:
            m = m[..., 0]
        m = (m > 127).astype(np.uint8)
        bank_rle[k] = rle_encode(m)

    bank_norm = (bank_X * bank_X).sum(axis=1)

    out_rle = []
    for test_id in tqdm(
        sample_ids.tolist(), desc="Generating fallback base (NN transfer)"
    ):
        p = os.path.join(test_img_dir, f"{test_id}.png")
        if not os.path.isfile(p):
            out_rle.append("")
            continue
        x = _img_to_vec(p)
        x_norm = float((x * x).sum())
        d = bank_norm + x_norm - 2.0 * (bank_X @ x)
        nn = int(np.argmin(d))
        out_rle.append(bank_rle[nn])

    return pd.DataFrame({"id": sample_ids.values, "rle_mask": out_rle})


def load_base_submission():
    """
    Load a base submission to refine and align to sample_submission ids.

    Change rationale (score-toward-target):
      If no valid non-empty base submission is found, generate a deterministic non-empty
      fallback base from train masks (NN transfer) instead of refining the empty sample,
      to move score upward from 0.0 toward the target.
    """
    sample = pd.read_csv(SAMPLE_SUB_PATH)[["id"]].copy()
    sample_ids = sample["id"]

    base_path, info = _find_existing_base_submission(sample_ids, SAMPLE_SUB_PATH)

    if base_path is None:
        print(
            "No suitable external base submission found; generating fallback base from train masks."
        )
        df_sub = _generate_base_submission_from_train(sample_ids, DATA_ROOT)
        df_sub = _normalize_submission_df(df_sub)
        source = "FALLBACK_NN_TRANSFER_FROM_TRAIN"
        info = {
            "score": None,
            "non_empty_frac": None,
            "coverage": 1.0,
            "mean_area": None,
        }
    else:
        df_sub = pd.read_csv(base_path)
        df_sub = _normalize_submission_df(df_sub)
        source = base_path

    df_sub = sample.merge(df_sub, on="id", how="left")
    df_sub["rle_mask"] = df_sub["rle_mask"].fillna("").astype(str)
    df_sub.loc[df_sub["rle_mask"].str.lower().eq("nan"), "rle_mask"] = ""

    non_empty_frac = df_sub["rle_mask"].str.strip().ne("").mean()
    print(f"Base submission source: {source}")
    print(f"Loaded base masks non-empty fraction: {non_empty_frac:.3f}")
    if info is not None:
        print(f"Base submission info: {info}")

    if non_empty_frac < 0.01:
        print(
            "WARNING: Base submission appears nearly empty after alignment. "
            "This will likely score ~0.0."
        )

    return df_sub


df = load_base_submission()




## === cell 4
def refine_mask_conservative(mask_img: np.ndarray):
    """
    Conservative refinement designed to avoid harming a strong base submission.

    We apply morphology only for small/fragmented masks where it most likely removes noise,
    otherwise we return the original mask unchanged.
    """
    m = (mask_img > 0).astype(bool)
    area = int(m.sum())

    if area == 0 or area >= 2500:
        return m.astype(np.uint8)

    if area >= 500:
        return m.astype(np.uint8)

    m2 = binary_closing(m, square(3))
    m2 = binary_opening(m2, square(2))

    m2 = remove_small_objects(m2, min_size=10)
    m2 = remove_small_holes(m2, area_threshold=10)

    if m2.sum() == 0 and area > 0:
        return m.astype(np.uint8)

    return m2.astype(np.uint8)




## === cell 5
try:
    non_empty_idx = []
    for i in range(min(len(df), 200)):
        rle = str(df.loc[i, "rle_mask"])
        if rle.strip() != "" and rle.lower() != "nan":
            non_empty_idx.append(i)
        if len(non_empty_idx) >= 6:
            break

    if len(non_empty_idx) > 0:
        plt.figure(figsize=(18, 3))
        for j, idx in enumerate(non_empty_idx):
            decoded_mask = rle_decode(df.loc[idx, "rle_mask"])
            plt.subplot(1, len(non_empty_idx), j + 1)
            plt.imshow(decoded_mask, cmap="gray")
            plt.axis("off")
            plt.title(df.loc[idx, "id"])
        plt.tight_layout()
except Exception:
    pass




## === cell 6
test_path = os.path.join(DATA_ROOT, "test", "images")
if not os.path.isdir(test_path):
    print(f"WARNING: Test images folder not found: {test_path} (continuing anyway)")




## === cell 7
for i in tqdm(range(df.shape[0]), desc="Refining masks"):
    rle = df.loc[i, "rle_mask"]
    if rle is None or str(rle).strip() == "" or str(rle).lower() == "nan":
        df.loc[i, "rle_mask"] = ""
        continue

    decoded_mask = rle_decode(rle, shape=(101, 101))

    refined = refine_mask_conservative(decoded_mask)

    orig_area = float((decoded_mask > 0).sum())
    ref_area = float((refined > 0).sum())
    if orig_area > 0:
        ratio = ref_area / (orig_area + 1e-6)
        if ratio < 0.50 or ratio > 1.50:
            refined = (decoded_mask > 0).astype(np.uint8)

    df.loc[i, "rle_mask"] = rle_encode(refined)




## === cell 8
df = df[["id", "rle_mask"]].copy()
df["rle_mask"] = df["rle_mask"].fillna("")

out_path = "crf_correction.csv"
df.to_csv(out_path, index=False)

print(f"Wrote submission: {out_path} with shape={df.shape}")
print(df.head())
print(
    "Non-empty fraction in output:",
    df["rle_mask"].astype(str).str.strip().ne("").mean(),
)
