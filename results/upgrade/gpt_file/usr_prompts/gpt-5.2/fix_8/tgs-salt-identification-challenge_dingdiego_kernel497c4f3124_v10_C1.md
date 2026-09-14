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

0.8219558632626561

# 6. Current score

0.026

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5221) has done: 'I remove the unavailable `pydensecrf` dependency (it’s not installed in this environment) and replace the CRF step with a tiny, deterministic post-processing that keeps the same “refine predicted mask then re-encode to RLE” core workflow. I also remove notebook-only magics (`%matplotlib inline`) and the external input dependency on `../input/k-fold-merger/submission.csv` by instead reading the competition’s `sample_submission.csv` as the base (so the pipeline always produces a valid submission). Finally, I fix the NaN checks and missing imports so the script runs end-to-end and writes a proper `.csv` submission file with the required columns.'
- What this solution (achieved 0.0728) has done: 'Your current pipeline is refining the *sample_submission’s dummy masks* (mostly empty) instead of generating masks from the seismic images, so the score can’t improve much beyond near-random/empty predictions. To move the score toward the 0.82 target while keeping the same overall “predict mask → refine → RLE encode → write CSV” semantics, I add a tiny deterministic, image-driven baseline mask generator (Otsu threshold on the image) and feed that into your existing `crf()` refinement. I also fix the RLE encode/decode to use the competition’s required **column-major (Fortran) pixel order**, which is a correctness bug that can significantly depress score. These are minimal, no-training changes that keep runtime well under the limit and should raise the score materially toward the target.'
- What this solution (achieved 0.029) has done: 'Your score is far below target, and the smallest legitimate way to move it upward (without changing your overall “image → initial mask → refine → RLE” pipeline) is to make the initial mask substantially more aligned with salt structure. I keep your refinement (`crf`) and RLE logic intact, but improve `initial_mask_from_image` using a deterministic, training-free heuristic that’s known to work better on this dataset: combine Otsu with a depth/brightness prior by using both raw intensity and a lightly smoothed version, then apply a conservative threshold and fill/clean with the same morphology family you already use. I also ensure the mask polarity is chosen correctly by checking which side of the threshold produces more plausible salt coverage (salt is often darker, but not always), picking the smaller connected foreground as a prior to avoid predicting “all salt”. These are minimal changes localized to initial mask generation and should move performance materially toward the 0.82 target without altering the rest of your workflow or adding training.'
- What this solution (achieved 0.027) has done: 'Your score is far below the target, so we should make a small but meaningful improvement to the *initial mask* while keeping the same “image → initial mask → refine (crf) → RLE → CSV” pipeline unchanged. The biggest low-risk gain here is to use the provided `depths.csv` as a deterministic prior: deeper images tend to have different intensity distributions, so we adjust the Otsu threshold slightly as a function of normalized depth (no training, no new model). To avoid catastrophic all-salt/empty masks, we also add a tiny safety clamp on predicted area before refinement, but keep the same morphology family and `crf()` step. Finally, we disable plotting by default (it can waste time) while preserving the code.'
- What this solution (achieved 0.0262) has done: 'Your current score is far below the target, so the smallest legitimate way to move it upward (without changing your overall “image → initial mask → refine → RLE → CSV” pipeline) is to make the initial mask slightly more aligned with salt texture while keeping everything deterministic and training-free. I keep `crf()` and the RLE functions unchanged, but adjust `initial_mask_from_image()` to use a second, edge-aware cue (local contrast via a light high-pass) blended with the smoothed intensity before the same Otsu+depth shift decision. I also add a tiny, deterministic per-image threshold tweak based on image mean/variance to reduce catastrophic empty/all-salt cases while preserving the existing area-safety logic. These are localized changes meant to raise mAP materially toward the 0.82 target without introducing training, new dependencies, or changing the core refinement/encoding semantics.'
- What this solution (achieved 0.026) has done: 'Your current score is far below the target, so we should make a small but meaningful boost without changing the overall “image → initial mask → refine → RLE → CSV” flow. The biggest likely blocker is that the initial mask is still too weak/unstable, so I keep your Otsu+depth structure but add a very small, deterministic multi-threshold ensembling (3 close thresholds) and then select the best candidate by a simple, safe area prior; this often improves IoU robustness across images without any training. I also add a tiny border-clearing step (removing a 1px frame) before refinement to reduce false positives that hurt precision at higher IoU thresholds. Everything remains deterministic, uses only existing packages, and still writes the same valid `crf_correction.csv` submission.'
- What this solution (achieved 0.026) has done: 'Your score is extremely far below the target, so the likely issue is not “small tuning” but a correctness mismatch with the competition’s *instance-based* metric: it expects **separate objects**, while your pipeline predicts one big semantic mask and encodes it as-is, which can score near-zero. To move substantially toward the target without changing your core “image → initial mask → refine → RLE” workflow, I add a minimal connected-components step that converts the refined binary mask into non-overlapping object instances and encodes them in a single RLE string per image (still one row per id) in a deterministic way. I also add a small probability thresholding fix by ensuring the refined mask is strictly binary uint8 and apply a tiny hole/object cleanup after labeling to avoid fragment noise hurting precision across IoU thresholds. Everything else (Otsu+depth init, morphology-based `crf`, RLE Fortran order, I/O paths) stays the same.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.morphology import (
    remove_small_holes,
    remove_small_objects,
    binary_opening,
    binary_closing,
    disk,
)
from skimage.filters import threshold_otsu, gaussian
from skimage.measure import label
from tqdm import tqdm

import matplotlib.pyplot as plt

np.random.seed(42)




## === cell 1
def rle_decode(rle_mask, shape=(101, 101)):
    """
    Kaggle TGS Salt RLE decode.

    IMPORTANT: Competition RLE is in column-major order (top-to-bottom, then left-to-right),
    which corresponds to Fortran order flatten/reshape.
    """
    if (
        rle_mask is None
        or (isinstance(rle_mask, float) and np.isnan(rle_mask))
        or str(rle_mask).strip() == ""
    ):
        return np.zeros(shape, dtype=np.uint8)

    s = str(rle_mask).split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape, order="F")


def rle_encode(im):
    """
    Kaggle TGS Salt RLE encode (column-major / Fortran order).
    """
    im = (im > 0).astype(np.uint8)
    pixels = im.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_encode_instances(mask, min_instance_size=24):
    mask = (mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        return ""

    lab = label(mask, connectivity=2)
    if lab.max() == 0:
        return ""

    out = np.zeros_like(mask, dtype=np.uint8)
    for k in range(1, lab.max() + 1):
        comp = lab == k
        if int(comp.sum()) >= int(min_instance_size):
            out[comp] = 1

    if out.sum() == 0:
        return ""

    return rle_encode(out)




## === cell 2
BASE = "../input/tgs-salt-identification-challenge"
if not os.path.exists(BASE):
    BASE = "../input"

sample_path = os.path.join(BASE, "sample_submission.csv")
test_img_dir = os.path.join(BASE, "test", "images")
depths_path = os.path.join(BASE, "depths.csv")

if not os.path.exists(sample_path):
    alt_base = os.path.join(BASE, "tgs-salt-identification-challenge")
    sample_path = os.path.join(alt_base, "sample_submission.csv")
    test_img_dir = os.path.join(alt_base, "test", "images")
    depths_path = os.path.join(alt_base, "depths.csv")

assert os.path.exists(
    sample_path
), f"Could not find sample_submission.csv at {sample_path}"
assert os.path.exists(
    test_img_dir
), f"Could not find test images directory at {test_img_dir}"
assert os.path.exists(depths_path), f"Could not find depths.csv at {depths_path}"

df = pd.read_csv(sample_path)
assert (
    "id" in df.columns and "rle_mask" in df.columns
), "sample_submission missing required columns"

depths = pd.read_csv(depths_path)
depths = depths[["id", "z"]].copy()
z_min, z_max = float(depths["z"].min()), float(depths["z"].max())
if z_max > z_min:
    depths["z_norm"] = (depths["z"] - z_min) / (z_max - z_min)
else:
    depths["z_norm"] = 0.5

z_map = dict(zip(depths["id"].values, depths["z_norm"].values))



## === cell 3
"""
Refinement function (kept as core post-processing step).
"""


def crf(original_image, mask_img):
    if len(mask_img.shape) >= 3:
        mask = mask_img[..., 0] > 0
    else:
        mask = mask_img > 0

    mask = remove_small_holes(mask, area_threshold=16)
    mask = remove_small_objects(mask, min_size=16)
    se = disk(1)
    mask = binary_closing(mask, footprint=se)
    mask = binary_opening(mask, footprint=se)

    return mask.astype(np.uint8)


"""
Score-relevant improvement while preserving the same core logic:
- Keep: Otsu-based mask init + depth-conditioned shift + same morphology family + crf refine.
- Change (minimal): build 3 very-close thresholds around the adjusted Otsu threshold and select
  the best candidate via a conservative area prior; this stabilizes predictions and typically
  improves IoU robustness without training.
- Change (minimal): clear a 1px border in the initial mask to reduce border false positives,
  which can otherwise lower AP at higher IoU thresholds.
"""


def initial_mask_from_image(orig_img, z_norm=0.5):
    if orig_img.ndim == 3:
        orig_img = orig_img[..., 0]

    img = orig_img.astype(np.float32)

    mn, mx = float(img.min()), float(img.max())
    if mx > mn:
        img01 = (img - mn) / (mx - mn)
    else:
        img01 = img * 0.0

    img_s = gaussian(img01, sigma=1.0, preserve_range=True)
    img_blur2 = gaussian(img01, sigma=2.0, preserve_range=True)
    hp = img_s - img_blur2  # high-pass (signed)

    hp_abs_mean = float(np.mean(np.abs(hp))) + 1e-6
    hp_n = np.clip(hp / (4.0 * hp_abs_mean), -1.0, 1.0)
    feat = np.clip(img_s + 0.15 * hp_n, 0.0, 1.0)

    t = threshold_otsu(feat)

    mu = float(feat.mean())
    sigma = float(feat.std())
    stat_shift = np.clip((0.12 - sigma) * 0.08 + (mu - 0.5) * 0.02, -0.03, 0.03)

    t_adj = np.clip(t + (z_norm - 0.5) * 0.06 + stat_shift, 0.0, 1.0)

    t_candidates = [
        np.clip(t_adj - 0.02, 0.0, 1.0),
        t_adj,
        np.clip(t_adj + 0.02, 0.0, 1.0),
    ]

    def score_area(a):
        if a < 0.003:
            return -10.0
        if a > 0.75:
            return -10.0
        return -abs(a - 0.20)

    best_mask = None
    best_score = -1e9

    for tc in t_candidates:
        m_bright = feat > tc
        m_dark = feat < tc

        area_b = float(m_bright.mean())
        area_d = float(m_dark.mean())

        if score_area(area_d) >= score_area(area_b):
            mask = m_dark
            area = area_d
        else:
            mask = m_bright
            area = area_b

        if area > 0.65:
            mask = ~mask

        mask = mask.copy()
        mask[0, :] = 0
        mask[-1, :] = 0
        mask[:, 0] = 0
        mask[:, -1] = 0

        mask = remove_small_objects(mask, min_size=8)
        mask = remove_small_holes(mask, area_threshold=8)

        sc = score_area(float(mask.mean()))
        if sc > best_score:
            best_score = sc
            best_mask = mask

    return best_mask.astype(np.uint8)




## === cell 4
"""
(Optional) quick visualization sanity check on a few images.
Disabled by default to save time; enable if needed.
"""
DO_PLOT = False

if DO_PLOT:
    nImgs = 3
    idxs = np.random.choice(df.index.values, size=min(nImgs, len(df)), replace=False)

    plt.figure(figsize=(12, 8))
    for j, i in enumerate(idxs, start=1):
        img_id = df.loc[i, "id"]
        img_path = os.path.join(test_img_dir, f"{img_id}.png")
        orig_img = imread(img_path)

        z_norm = float(z_map.get(img_id, 0.5))
        init_mask = initial_mask_from_image(orig_img, z_norm=z_norm)
        refined = crf(orig_img, init_mask)

        plt.subplot(nImgs, 3, 3 * j - 2)
        plt.imshow(orig_img, cmap="gray")
        plt.title(f"{img_id} image (z={z_norm:.2f})")
        plt.axis("off")

        plt.subplot(nImgs, 3, 3 * j - 1)
        plt.imshow(init_mask, cmap="gray")
        plt.title("initial mask")
        plt.axis("off")

        plt.subplot(nImgs, 3, 3 * j)
        plt.imshow(refined, cmap="gray")
        plt.title("refined mask")
        plt.axis("off")

    plt.tight_layout()



## === cell 5
"""
Apply image-driven initial mask + refinement and write submission.
"""
for i in tqdm(range(df.shape[0])):
    img_id = df.loc[i, "id"]
    img_path = os.path.join(test_img_dir, f"{img_id}.png")
    orig_img = imread(img_path)

    z_norm = float(z_map.get(img_id, 0.5))
    init_mask = initial_mask_from_image(orig_img, z_norm=z_norm)
    refined_mask = crf(orig_img, init_mask)

    refined_mask = (refined_mask > 0).astype(np.uint8)
    refined_mask = remove_small_objects(refined_mask.astype(bool), min_size=20).astype(
        np.uint8
    )
    refined_mask = remove_small_holes(
        refined_mask.astype(bool), area_threshold=20
    ).astype(np.uint8)

    df.loc[i, "rle_mask"] = rle_encode_instances(refined_mask, min_instance_size=24)



## === cell 6
df = df[["id", "rle_mask"]]
out_path = "crf_correction.csv"
df.to_csv(out_path, index=False)

print(f"Wrote submission: {out_path}")
print(df.head())
