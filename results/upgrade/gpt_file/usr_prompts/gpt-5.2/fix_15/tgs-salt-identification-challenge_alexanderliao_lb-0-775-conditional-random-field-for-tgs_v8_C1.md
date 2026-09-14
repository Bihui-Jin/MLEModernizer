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

0.5221

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5221) has done: 'I make the notebook run in this Kaggle environment by removing the unavailable `pydensecrf` dependency and replacing it with a small, deterministic post-processing step using only installed `scikit-image` (this preserves the “post-process predicted masks” core intent without changing the upstream model). I also remove IPython-only magic (`%matplotlib inline`) and fix the broken external submission path by loading `sample_submission.csv` from the provided dataset so a valid `submission.csv` is always produced. Finally, I make NaN/empty-mask handling correct (using `pd.isna`) and ensure the output CSV has exactly the required `id,rle_mask` columns and `.csv` suffix.'
- What this solution (achieved 0.0001) has done: 'The crash comes from looking up test IDs in `depths_df` loaded from the wrong CSV: the `depths.csv` at `DATA_ROOT` contains only train IDs, so `.loc[test_ids]` raises a `KeyError`. I fix this by loading depths from the top-level `/kaggle/input/depths.csv` (which includes both train+test), and by making the test-depth lookup robust via a merge so missing IDs (if any) get a safe fill value instead of crashing. These changes are score-neutral (they only unblock inference) and preserve your existing “depth prior + simple mask cleanup” core logic. Finally, the submission-writing cell work because `sub_df` always be created.'
- What this solution (achieved 0.1301) has done: 'Your current 0.0001 score strongly suggests the submission masks are mostly empty or otherwise badly miscalibrated for the IoU-sweep metric. I keep your “depth prior + simple morphological cleanup” core logic identical, but tune only two calibration knobs that directly affect mask size/quality: (1) lower the depth-prior threshold (so predictions aren’t empty), and (2) relax the morphology cleanup slightly (so thin/fragmented salt regions aren’t deleted). I also ensure the depth binning used for train/test is consistent by reusing the same quantile edges for both, which avoids systematic bin misassignment that can degrade predictions. These changes are small, deterministic, and should move the score upward toward your target without changing the overall approach.'
- What this solution (achieved 0.1301) has done: 'Your current score suggests the depth-prior masks are still systematically mis-sized for the IoU-sweep metric, so I keep your exact “depth-bin mean mask → threshold → morphology cleanup → RLE” pipeline but calibrate only the two most sensitive knobs: the prior threshold and the cleanup aggressiveness. Specifically, I slightly lower `PRIOR_THRESH` to reduce empty/under-segmented predictions, and I soften `remove_small_objects/remove_small_holes` so true salt regions aren’t erased. I also add a tiny “empty-mask safeguard” that prevents morphology from deleting the entire mask when the prior is already very small (this preserves semantics but avoids catastrophic all-empty outputs). These changes are deterministic, minimal, and aimed at moving the score upward toward your 0.80 target.'
- What this solution (achieved 0.5221) has done: 'The crash comes from reading PNGs that sometimes have 4 channels (RGBA), so `orig_img` becomes shape `(101,101,4)` and cannot be multiplied with the `(101,101)` prior. I fix this by converting any multi-channel image to a single grayscale channel deterministically (use the first channel, which is sufficient for these seismic images), preserving the existing “prior * image-derived gate” logic. I also make cell numbering start at 1 (your current script starts at cell 0) and keep submission creation unchanged so `submission.csv` is always written. These changes are execution-unblocking and should restore your intended behavior and score.'
- What this solution (achieved 0.5221) has done: 'Your current pipeline is a depth-bin mean-mask prior gated by an image-derived sigmoid, then a single global threshold and light morphology before RLE. To move the score upward toward 0.80 with minimal risk and without changing the core approach, I only (1) calibrate the single most sensitive knob (`PROB_THRESH`) using a small train-split validation search for best mean AP@IoU-sweep, and (2) slightly align the morphology to preserve connected salt regions while avoiding deletion of true positives. The trained prior construction, gating function, cleanup intent, and RLE semantics remain the same; we just pick a better threshold deterministically from data. The script still runs end-to-end and always writes a valid `submission.csv`.'
- What this solution (achieved 0.5221) has done: 'Your current score (0.5221) is far below the target (0.80185), so we should improve performance with minimal, low-risk changes that preserve your core “depth-bin mean-mask prior × image gate → single threshold → morphology → RLE” pipeline. The biggest likely regression is that the threshold search evaluates the *post-processed binary* masks, but Kaggle performance is very sensitive to the threshold and the morphology settings; we keep the approach identical but (1) make the validation split deterministic and stratified by depth-bin to better match test distribution, and (2) widen and slightly lower the threshold candidate grid so we don’t get stuck in an over-conservative (too-empty) regime. Additionally, we make the image gating scale (`0.10`) itself selectable from a tiny fixed grid during the same validation loop (this keeps the same gating function and semantics, only calibrates its steepness), which is often a major calibration knob for this heuristic method. All changes remain deterministic, keep I/O paths unchanged, and still produce a valid `submission.csv`.'
- What this solution (achieved 0.5221) has done: 'Your current heuristic pipeline is already stable and producing a valid submission, but it’s leaving a lot of score on the table because it predicts **one connected mask per image** while the competition metric is **instance-based** (multiple objects). To move the score upward toward your target without changing the “depth-bin mean-mask prior × image gate → threshold → morphology → RLE” core logic, I keep the prior/gating/thresholding identical and only change post-processing to (a) split predictions into connected components and (b) keep only the best-matching single component on validation when selecting threshold/gate, then apply the same “keep-largest-component” rule at test time. This is a minimal, deterministic change that aligns predictions with the evaluation semantics and usually yields a sizable jump for this competition. I also cache loaded PNGs during threshold search to keep runtime under the limit while preserving the same computations.'
- What this solution (achieved 0.5221) has done: 'We keep your exact depth-bin prior × image-gate → threshold → morphology → RLE pipeline, but fix one metric-misalignment that’s likely capping your score: the current validation metric treats each image as a single-object IoU, while Kaggle scores **instance AP** (connected components matching). I replace only the validation scorer with an instance-style mAP@IoU-sweep computed from connected components, so the chosen `PROB_THRESH`/`GATE_SCALE` are tuned for the real leaderboard metric (no change to prediction core). I also adjust post-processing minimally by making “keep largest component” conditional (only when the prediction is highly fragmented), preserving your intent but avoiding deleting legitimate multi-blob salt regions that help AP. These are small, deterministic changes that should move your 0.5221 score upward toward the 0.8019 target without altering the main approach.'
- What this solution (achieved 0.5221) has done: 'Your current score (0.5221) is far below the target (0.80185), so we should improve toward the target with the smallest, safest calibration changes while preserving the exact “depth-bin prior × image sigmoid gate → threshold → morphology → RLE” pipeline. The biggest likely score limiter now is calibration: using `prob = prior * g` compresses probabilities and tends to under-segment, so I keep the same elements but only allow a tiny, validated exponent reweighting of the gate (`g**p`) to better match the metric without changing the model structure. I also slightly extend the threshold and gate-scale candidate grids (still small) so the deterministic validation search can find a less conservative setting, and I keep the same instance-style validation metric you already use. These changes are minimal, deterministic, and should move the leaderboard score upward toward your target band while still writing a valid `submission.csv`.'
- What this solution (achieved 0.5221) has done: 'Your current score (0.5221) is far below the target (0.80185), so we should improve toward it with the smallest changes that keep your exact “depth-bin prior × image sigmoid gate → global threshold → morphology → RLE” pipeline intact. The biggest likely limiter is that the learned prior is built from **raw, unrefined** train masks, while at inference you always apply the `crf()` morphology; this train/test mismatch makes the prior systematically noisier than your final predictions. I therefore build the depth-bin mean-mask prior from the **same refined masks** (using `crf()` on the decoded train masks), which preserves your approach but aligns the prior with the post-processing you actually use. Everything else (validation search, prediction, RLE, paths, output format) remains the same so it still runs end-to-end and writes `submission.csv`.'
- What this solution (achieved 0.5221) has done: 'Your current score (0.5221) is far below the 0.80185 target, so we should improve toward it with the smallest changes that keep your exact “depth-bin mean-mask prior × image sigmoid gate → global threshold → morphology → RLE” pipeline intact. The most likely score limiter now is that `keep_largest_component_if_fragmented()` can delete valid multi-blob salt structure and hurt the instance-based AP metric; we make that step less aggressive by allowing more components before collapsing to the largest one. To avoid overfitting the threshold search to this changed post-process, we re-run the same deterministic validation grid-search unchanged in structure, just with the updated component rule. No I/O paths change, and the script still runs end-to-end and writes a valid `submission.csv`.'

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
from skimage.measure import label

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

    Change (score-relevant, minimal): slightly favor connectivity (closing) and
    avoid erasing small true salt blobs too aggressively.
    """
    m = (np.asarray(mask_img) > 0).astype(bool)

    small_mask = m.sum() < 12

    se = disk(1)
    m = binary_closing(m, se)
    if not small_mask:
        m = binary_opening(m, se)

    if not small_mask:
        m = remove_small_holes(m, area_threshold=8)
        m = remove_small_objects(m, min_size=3)

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

dummy_img = np.zeros((101, 101), dtype=np.float32)

for _, row in tqdm(
    train_merged.iterrows(), total=train_merged.shape[0], desc="Building depth priors"
):
    zb = int(row["z_bin"])
    m = rle_decode(row["rle_mask"]).astype(np.uint8)
    m_ref = crf(dummy_img, m).astype(np.float32)
    sum_masks[zb] += m_ref
    cnt_masks[zb] += 1

mean_masks = np.zeros_like(sum_masks, dtype=np.float32)
for b in range(n_bins):
    if cnt_masks[b] > 0:
        mean_masks[b] = sum_masks[b] / float(cnt_masks[b])

prior_probs = mean_masks.copy()

prior_probs.shape, int(cnt_masks.min()), int(cnt_masks.max())




## === cell 6
def load_grayscale_0to1_png(path):
    img = imread(path)
    img = np.asarray(img)
    if img.ndim == 3:
        img = img[..., 0]
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    return img


_IMG_CACHE = {}


def get_image_by_id(img_id):
    if img_id in _IMG_CACHE:
        return _IMG_CACHE[img_id]
    train_img_path = os.path.join(DATA_ROOT, "train", "images", img_id + ".png")
    if os.path.isfile(train_img_path):
        img_path = train_img_path
    else:
        img_path = os.path.join(TEST_IMG_DIR, img_id + ".png")
    orig_img = load_grayscale_0to1_png(img_path)
    _IMG_CACHE[img_id] = orig_img
    return orig_img


def keep_largest_component(mask_u8):
    m = np.asarray(mask_u8) > 0
    if m.sum() == 0:
        return mask_u8.astype(np.uint8)
    lab = label(m, connectivity=1)
    if lab.max() == 0:
        return mask_u8.astype(np.uint8)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    k = int(np.argmax(sizes))
    return (lab == k).astype(np.uint8)


def keep_largest_component_if_fragmented(mask_u8, max_components=4):
    """
    Change (score-relevant, minimal): your current setting (max_components=2) tends
    to collapse legitimate multi-blob salt into a single blob, which can reduce the
    instance-based AP metric. We keep the same idea (collapse only when very noisy),
    but allow a few more components before collapsing.

    This does not change the core pipeline (prior × gate → threshold → morphology);
    it only relaxes a post-process decision that is tightly coupled to the metric.
    """
    m = np.asarray(mask_u8) > 0
    if m.sum() == 0:
        return mask_u8.astype(np.uint8)
    lab = label(m, connectivity=1)
    n_comp = int(lab.max())
    if n_comp <= int(max_components):
        return mask_u8.astype(np.uint8)
    return keep_largest_component(mask_u8)


def predict_mask_for_id(img_id, z_val, prob_thresh, gate_scale, gate_power):
    b = z_to_bin(float(z_val))
    prior = prior_probs[b].astype(np.float32)

    orig_img = get_image_by_id(img_id)

    g = 1.0 / (1.0 + np.exp(-(orig_img - orig_img.mean()) / float(gate_scale)))
    g = np.power(g, float(gate_power)).astype(np.float32)

    prob = prior * g
    mask = (prob >= float(prob_thresh)).astype(np.uint8)
    refined = crf(orig_img, mask)

    refined = keep_largest_component_if_fragmented(refined, max_components=4)
    return refined


def _components_from_mask(mask_u8):
    m = (np.asarray(mask_u8) > 0).astype(np.uint8)
    if m.sum() == 0:
        return []
    lab = label(m.astype(bool), connectivity=1)
    comps = []
    for k in range(1, int(lab.max()) + 1):
        comp = lab == k
        if comp.sum() > 0:
            comps.append(comp)
    return comps


def _iou(a_bool, b_bool):
    inter = float(np.logical_and(a_bool, b_bool).sum())
    union = float(np.logical_or(a_bool, b_bool).sum())
    return inter / union if union > 0 else 0.0


def map_iou_sweep_instance(y_true_batch, y_pred_batch):
    """
    Validation scorer aligned with Kaggle's instance AP over IoU thresholds.
    """
    thresholds = np.arange(0.5, 1.0, 0.05, dtype=np.float32)
    N = y_true_batch.shape[0]
    aps = np.zeros((N,), dtype=np.float32)

    for i in range(N):
        t_comps = _components_from_mask(y_true_batch[i])
        p_comps = _components_from_mask(y_pred_batch[i])

        if len(t_comps) == 0 and len(p_comps) == 0:
            aps[i] = 1.0
            continue
        if len(t_comps) == 0 and len(p_comps) > 0:
            aps[i] = 0.0
            continue
        if len(t_comps) > 0 and len(p_comps) == 0:
            aps[i] = 0.0
            continue

        iou_mat = np.zeros((len(t_comps), len(p_comps)), dtype=np.float32)
        for ti, tc in enumerate(t_comps):
            for pj, pc in enumerate(p_comps):
                iou_mat[ti, pj] = _iou(tc, pc)

        precs = []
        for thr in thresholds:
            used_t = np.zeros((len(t_comps),), dtype=bool)
            used_p = np.zeros((len(p_comps),), dtype=bool)

            flat = []
            for ti in range(iou_mat.shape[0]):
                for pj in range(iou_mat.shape[1]):
                    v = float(iou_mat[ti, pj])
                    if v > float(thr):
                        flat.append((v, ti, pj))
            flat.sort(reverse=True, key=lambda x: x[0])

            tp = 0
            for _, ti, pj in flat:
                if (not used_t[ti]) and (not used_p[pj]):
                    used_t[ti] = True
                    used_p[pj] = True
                    tp += 1

            fp = int((~used_p).sum())
            fn = int((~used_t).sum())
            denom = tp + fp + fn
            precs.append(tp / denom if denom > 0 else 0.0)

        aps[i] = float(np.mean(precs, dtype=np.float32))

    return float(aps.mean())


rng = np.random.RandomState(1337)
val_frac = 0.2

val_indices = []
trn_indices = []
for zb, grp in train_merged.groupby("z_bin").groups.items():
    idx = np.array(list(grp), dtype=np.int64)
    idx = rng.permutation(idx)
    vn = int(round(val_frac * len(idx)))
    val_indices.append(idx[:vn])
    trn_indices.append(idx[vn:])

val_idx = (
    np.concatenate(val_indices) if len(val_indices) else np.array([], dtype=np.int64)
)
trn_idx = (
    np.concatenate(trn_indices) if len(trn_indices) else np.array([], dtype=np.int64)
)

val_df = train_merged.iloc[val_idx].reset_index(drop=True)

y_true = np.zeros((len(val_df), 101, 101), dtype=np.uint8)
for i, rle in enumerate(val_df["rle_mask"].values):
    y_true[i] = rle_decode(rle)

cand_thresholds = np.array(
    [
        0.10,
        0.12,
        0.14,
        0.16,
        0.18,
        0.20,
        0.22,
        0.24,
        0.26,
        0.28,
        0.30,
        0.32,
        0.35,
        0.38,
        0.40,
        0.42,
        0.45,
        0.48,
    ],
    dtype=np.float32,
)

cand_gate_scales = np.array([0.06, 0.07, 0.08, 0.10, 0.12, 0.14], dtype=np.float32)
cand_gate_powers = np.array([0.80, 1.00, 1.20], dtype=np.float32)

best_t = 0.45
best_gate = 0.10
best_pow = 1.00
best_score = -1.0

for img_id in val_df["id"].astype(str).values:
    _ = get_image_by_id(img_id)

for gate_scale in cand_gate_scales:
    for gate_power in cand_gate_powers:
        for t in cand_thresholds:
            y_pred = np.zeros_like(y_true)
            for i, (img_id, z_val) in enumerate(
                zip(
                    val_df["id"].astype(str).values,
                    val_df["z"].values.astype(np.float32),
                )
            ):
                y_pred[i] = predict_mask_for_id(
                    img_id, z_val, t, gate_scale, gate_power
                )

            s = map_iou_sweep_instance(y_true, y_pred)
            if s > best_score:
                best_score = s
                best_t = float(t)
                best_gate = float(gate_scale)
                best_pow = float(gate_power)

PROB_THRESH = best_t
GATE_SCALE = best_gate
GATE_POWER = best_pow
print(
    "Chosen PROB_THRESH:",
    PROB_THRESH,
    "Chosen GATE_SCALE:",
    GATE_SCALE,
    "Chosen GATE_POWER:",
    GATE_POWER,
    "val instance-mAP:",
    best_score,
)



## === cell 7
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
    list(zip(test_ids, test_depths)), total=len(test_ids), desc="Predicting test masks"
):
    refined = predict_mask_for_id(img_id, z_val, PROB_THRESH, GATE_SCALE, GATE_POWER)
    pred_rles.append(rle_encode(refined))

sub_df = pd.DataFrame({"id": test_ids, "rle_mask": pred_rles})
sub_df.head()



## === cell 8
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
