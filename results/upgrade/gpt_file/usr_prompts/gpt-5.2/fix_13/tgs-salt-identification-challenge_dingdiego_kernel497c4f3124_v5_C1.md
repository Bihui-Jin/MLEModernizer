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

0.7782604932929458

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5221) has done: 'I make the notebook run in the Kaggle environment by (1) removing the unavailable `pydensecrf` dependency and the Jupyter-only `%matplotlib inline`, and (2) fixing the NaN checks and missing imports that currently prevent `df` from being created and used. To preserve the core intent (post-processing a precomputed submission with a “CRF-like correction”), I keep the same I/O flow and mask encode/decode logic, but implement a safe fallback post-processing step using only installed libraries (a small morphological cleanup) when CRF isn’t available. I also ensure the script doesn’t rely on an external dataset submission path that may not exist by falling back to `sample_submission.csv` if the baseline submission file is missing. Finally, the code always write a valid `crf_correction.csv` with the required `id,rle_mask` columns.'
- What this solution (achieved 0.5221) has done: 'Your current score (0.5221) is far below the target (0.7783), so we need a meaningful lift without changing the external model/core approach (we still post-process an existing submission). The biggest likely issue is that the RLE decode/encode is using row-major order, while this competition expects column-major (top-to-bottom then left-to-right), which can severely harm IoU; fixing this is a minimal, correctness-preserving change that should boost score substantially. I also apply the same small morphological cleanup to *all* masks (including empty ones) in a controlled way, but keep it conservative so it doesn’t over-alter predictions. The script still fall back to `sample_submission.csv` if the baseline file is absent and always write a valid `crf_correction.csv`.'
- What this solution (achieved 0.0) has done: 'Your current score (0.5221) is far below the target (0.7783), so we need a real lift but with minimal changes and without changing the overall “post-process an existing submission” approach. The biggest low-risk gain is to tune the post-processing to be less destructive: your current opening/closing + small-object/hole removal can erase thin salt structures and hurt IoU, so I reduce its aggressiveness and make it conditional on mask size. I also fix a correctness bug in the loop: using `df.loc[i]` with a non-RangeIndex can misalign rows; switching to `iloc` ensures we update the right `id`/mask pairs. Finally, I keep the RLE order consistent with the competition convention and ensure the output CSV remains valid.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score strongly suggests the submission file is invalid for the competition (most commonly: wrong column name, wrong RLE convention, or IDs not matching the sample submission order). I keep your exact “post-process an existing submission” approach and the same CRF-like morphological cleanup, but (1) force the output `id` list to match `sample_submission.csv` exactly (and align predictions to it), (2) harden RLE decode/encode to the exact 101×101 column-major convention and guarantee sorted runs, and (3) remove the plotting cell’s side effects and ensure we never accidentally write non-string/`nan` into `rle_mask`. These are minimal correctness fixes that should move the score up from 0.0 without changing your model/post-processing core logic.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 strongly indicates the submission is being rejected/parsed incorrectly rather than the masks being “bad”, so the smallest move toward the target is to harden submission validity. I keep your exact post-processing approach, but (1) guarantee the output `id` order matches `sample_submission.csv` exactly without any merge reordering surprises, and (2) make RLE encode/decode strictly follow the competition’s column-major convention using `reshape(..., order='F')`/`flatten(order='F')` to eliminate any subtle transpose mistakes. Finally, I ensure we never write the literal string `"nan"` (or non-string types) into `rle_mask`, and we clip any decoded runs to image size to avoid invalid RLE. These are minimal correctness fixes that should lift the score off 0.0 and toward your target without changing the core logic.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with an invalid submission format rather than “bad masks”, so the smallest move toward the target is to harden submission validity while keeping your exact post-processing approach. I (1) make `rle_encode` return an empty string for empty masks (instead of returning `"1 0"`), which Kaggle often treats as malformed RLE, and (2) ensure we always write strictly valid strings in `rle_mask`. I also keep the sample-submission ID alignment you already added and leave the CRF-like morphology and overall workflow unchanged. These changes should lift you off 0.0 toward your previous ~0.52+ range without altering the core logic.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly coming from a submission-parsing issue rather than model quality, so the smallest move toward the target is to harden RLE correctness and ensure every `rle_mask` is strictly valid for Kaggle’s checker. I keep your exact workflow (load baseline → decode → morphological “CRF-like” cleanup → re-encode), but fix the RLE encoder to the canonical Kaggle TGS implementation that guarantees sorted, non-overlapping runs in column-major order and avoids edge-case invalid strings. I also add a fast validity check that decodes the produced RLE back and asserts it matches the binary mask size-wise, so we don’t silently write malformed masks. These are minimal correctness changes that should lift the score off 0.0 back toward your prior ~0.52+ range without changing the core approach.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score indicates Kaggle is likely treating the submission as malformed/empty for many rows rather than merely “low quality”, so the smallest move toward the target is to make the RLE handling strictly match the competition’s required pixel order and accepted empty-mask convention. I keep your exact workflow (load baseline→decode→morphology “CRF-like” cleanup→encode) but harden the encoder/decoder to the canonical TGS implementation that guarantees sorted runs in column-major order and never emits invalid placeholders like `"1 0"`. I also make the sanity check non-fatal (log and fall back) so the script always writes a valid CSV instead of potentially erroring mid-loop and producing a partial/invalid file. Finally, I ensure the output IDs are aligned to `sample_submission.csv` exactly (already mostly done) and that `rle_mask` is always a proper string.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly coming from Kaggle treating many/all rows as invalid/empty RLE due to a subtle convention mismatch, not from the morphology itself. I keep your exact workflow (load baseline → decode → CRF-like morphology → encode) but fix the single highest-impact correctness issue: TGS expects RLE to be encoded over the transposed (width,height) flattening in Fortran order, i.e., `mask.T.flatten()` behavior, not `mask.flatten(order='F')` on (H,W) directly. I update `rle_encode`/`rle_decode` to the canonical, competition-safe implementation (still 101×101, still returns '' for empty), and keep your roundtrip check so we always emit a valid `crf_correction.csv`. No changes to the post-processing logic or loop structure beyond using the corrected RLE functions.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with Kaggle rejecting/parsing the submission rather than the masks being genuinely awful, so the smallest move toward the target is to harden RLE validity without changing your “baseline → decode → CRF-like morphology → encode” core logic. I fix the single biggest correctness risk: your encoder currently emits “start positions” in 1-based indexing but your `changes` array is computed in 0-based; this produces systematically wrong RLE and can yield near-empty/invalid interpretations. I replace `rle_encode`/`rle_decode` with the canonical TGS implementation (column-major via transpose, correct 1-based starts, and empty string for empty masks) and keep your roundtrip check to guarantee we only write valid strings. Everything else (paths, morphology, loop, output columns/order) stays the same.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with Kaggle not accepting/parsing many of your masks as valid RLE, so the smallest improvement toward the target is to make RLE generation strictly canonical for this competition and to ensure we never silently blank out masks due to a too-strict roundtrip check. I keep your exact workflow (baseline submission → decode → same CRF-like morphology → encode) but update the encoder/decoder to the well-known TGS implementation (Fortran-order flattening with 1-based starts) and add a lightweight RLE sanitizer that fixes edge cases (odd token counts, non-positive lengths) instead of turning them into empty masks. I also change the “roundtrip failure” behavior to fall back to the original (pre-CRF) RLE for that image, which preserves prediction quality while still guaranteeing a valid CSV. These minimal correctness fixes should move the score off 0.0 and back toward your prior ~0.5+ range without altering the core post-processing logic.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly due to Kaggle rejecting/parsing the submission RLE rather than the post-processing quality, so the smallest move toward the target is to make the RLE convention strictly canonical for TGS. I change only the RLE encode/decode to the well-known “mask.T.flatten()” convention (this fixes a subtle but critical orientation mismatch), and I keep your existing CRF-like morphology and loop intact. I also harden `_sanitize_rle` to enforce sorted, non-overlapping runs (by rebuilding from a decoded mask), which avoids invalid/overlapping runs that can trigger a 0.0. The output CSV path/columns/order remain identical.'

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

import matplotlib.pyplot as plt
from tqdm import tqdm

np.random.seed(42)




## === cell 1
def rle_decode(rle_mask, shape=(101, 101)):
    """
    Decode RLE to mask (H, W) using the canonical TGS Salt convention:
    - RLE is over mask.T.flatten() (i.e., column-major over (W,H) effectively)
    - 1-based starts
    """
    h, w = shape

    if rle_mask is None:
        return np.zeros((h, w), dtype=np.uint8)
    if isinstance(rle_mask, float) and np.isnan(rle_mask):
        return np.zeros((h, w), dtype=np.uint8)

    s = str(rle_mask).strip()
    if s == "" or s.lower() == "nan":
        return np.zeros((h, w), dtype=np.uint8)

    tokens = s.split()
    if len(tokens) < 2 or (len(tokens) % 2 != 0):
        return np.zeros((h, w), dtype=np.uint8)

    starts = np.asarray(tokens[0::2], dtype=np.int64) - 1
    lengths = np.asarray(tokens[1::2], dtype=np.int64)
    lengths = np.maximum(lengths, 0)
    ends = starts + lengths

    img = np.zeros(h * w, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        lo = int(max(lo, 0))
        hi = int(min(hi, h * w))
        if hi > lo:
            img[lo:hi] = 1

    return img.reshape((w, h), order="F").T


def rle_encode(mask):
    """
    Encode mask (H, W) to RLE using the canonical TGS Salt convention:
    - encode mask.T.flatten() in Fortran order
    - return '' for empty
    """
    m = (mask > 0).astype(np.uint8)
    if m.sum() == 0:
        return ""

    pixels = m.T.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]]).astype(np.uint8)
    changes = np.where(pixels[1:] != pixels[:-1])[0] + 1  # 1-based positions
    changes[1::2] -= changes[::2]  # lengths

    return " ".join(str(int(x)) for x in changes)


def _sanitize_rle(rle_str, shape=(101, 101)):
    """
    Ensure RLE is safely parseable *and* valid for the competition checker.
    Minimal but robust approach: decode -> re-encode canonically.
    This enforces:
      - even token count
      - positive lengths
      - sorted/non-overlapping runs
      - correct pixel order convention
    """
    if rle_str is None:
        return ""
    if isinstance(rle_str, float) and np.isnan(rle_str):
        return ""

    s = str(rle_str).strip()
    if s == "" or s.lower() == "nan":
        return ""

    toks = s.split()
    if len(toks) % 2 != 0:
        return ""

    clean = []
    for a, b in zip(toks[0::2], toks[1::2]):
        try:
            st = int(a)
            ln = int(b)
        except Exception:
            continue
        if st <= 0 or ln <= 0:
            continue
        clean.extend([str(st), str(ln)])

    if len(clean) == 0:
        return ""

    m = rle_decode(" ".join(clean), shape=shape)
    return rle_encode(m)


def _rle_roundtrip_ok(mask, rle, shape=(101, 101)):
    dec = rle_decode(rle, shape=shape)
    m = (mask > 0).astype(np.uint8)
    return (dec.shape == m.shape) and np.array_equal(dec, m)




## === cell 2
BASE_INPUT = "../input/tgs-salt-identification-challenge"
test_path = os.path.join(BASE_INPUT, "test", "images")

baseline_submission_path = (
    "../input/baseline-0-731-0-158-5-and-more-fold/submission.csv"
)
sample_submission_path = os.path.join(BASE_INPUT, "sample_submission.csv")

sample_df = pd.read_csv(sample_submission_path)
if "id" not in sample_df.columns or "rle_mask" not in sample_df.columns:
    raise ValueError(
        f"sample_submission.csv must contain columns ['id','rle_mask'], got {sample_df.columns.tolist()}"
    )

if os.path.exists(baseline_submission_path):
    df = pd.read_csv(baseline_submission_path)
else:
    df = sample_df.copy()

if "id" not in df.columns or "rle_mask" not in df.columns:
    raise ValueError(
        f"Submission file must contain columns ['id','rle_mask'], got {df.columns.tolist()}"
    )

df["id"] = df["id"].astype(str)
df["rle_mask"] = df["rle_mask"].where(df["rle_mask"].notna(), "")
df["rle_mask"] = df["rle_mask"].astype(str)
df.loc[df["rle_mask"].str.lower().eq("nan"), "rle_mask"] = ""

df["rle_mask"] = df["rle_mask"].map(lambda x: _sanitize_rle(x, shape=(101, 101)))

pred_map = dict(zip(df["id"].values, df["rle_mask"].values))
df = sample_df.copy()
df["id"] = df["id"].astype(str)
df["rle_mask"] = df["id"].map(pred_map).fillna("").astype(str)
df.loc[df["rle_mask"].str.lower().eq("nan"), "rle_mask"] = ""
df["rle_mask"] = df["rle_mask"].map(lambda x: _sanitize_rle(x, shape=(101, 101)))

df.head()



## === cell 3
"""
Fallback post-processing (CRF-like cleanup without pydensecrf).
Core logic unchanged; we only fix RLE correctness/robustness to avoid 0.0 invalid submissions.
"""


def crf(original_image, mask_img):
    mask = mask_img
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask = (mask > 0).astype(bool)

    area = int(mask.sum())
    if area == 0:
        return mask.astype(np.uint8)

    if area >= 40:
        se = disk(1)
        mask = binary_closing(mask, se)
        mask = binary_opening(mask, se)

    if area >= 80:
        mask = remove_small_objects(mask, min_size=6)
        mask = remove_small_holes(mask, area_threshold=6)

    return mask.astype(np.uint8)




## === cell 4
"""
Optional visualization: keep it non-blocking and avoid any dependency on interactive backends.
"""
try:
    non_empty_idx = df.index[df["rle_mask"].astype(str).str.len() > 0].tolist()
    if len(non_empty_idx) == 0:
        vis_idx = list(
            np.random.choice(df.index.values, size=min(3, len(df)), replace=False)
        )
    else:
        vis_idx = non_empty_idx[:3]

    plt.figure(figsize=(12, 9))
    for k, i in enumerate(vis_idx, start=1):
        img_id = df.loc[i, "id"]
        rle = df.loc[i, "rle_mask"]
        decoded_mask = rle_decode(rle)

        img_path = os.path.join(test_path, f"{img_id}.png")
        orig_img = (
            imread(img_path)
            if os.path.exists(img_path)
            else np.zeros((101, 101), dtype=np.uint8)
        )

        post = crf(orig_img, decoded_mask)

        plt.subplot(len(vis_idx), 3, 3 * (k - 1) + 1)
        plt.imshow(orig_img, cmap="gray")
        plt.title(f"{img_id} image")
        plt.axis("off")

        plt.subplot(len(vis_idx), 3, 3 * (k - 1) + 2)
        plt.imshow(decoded_mask, cmap="gray")
        plt.title("input mask")
        plt.axis("off")

        plt.subplot(len(vis_idx), 3, 3 * (k - 1) + 3)
        plt.imshow(post, cmap="gray")
        plt.title("post-processed")
        plt.axis("off")

    plt.tight_layout()
except Exception as e:
    print(f"Visualization skipped due to: {e}")



## === cell 5
"""
Applying fallback post-processing on the predicted masks.

Keep the loop/training approach identical; only RLE correctness changes above.
We keep the roundtrip check non-fatal so we always finish and write a complete submission.
"""
rle_col = df.columns.get_loc("rle_mask")
bad_roundtrip = 0
fallback_to_input = 0

for row_i in tqdm(range(df.shape[0])):
    img_id = df.iloc[row_i]["id"]
    rle_in = df.iloc[row_i]["rle_mask"]

    decoded_mask = rle_decode(rle_in)

    img_path = os.path.join(test_path, f"{img_id}.png")
    orig_img = (
        imread(img_path)
        if os.path.exists(img_path)
        else np.zeros((101, 101), dtype=np.uint8)
    )

    crf_output = crf(orig_img, decoded_mask)

    enc = rle_encode(crf_output)
    enc = _sanitize_rle(enc, shape=(101, 101))

    if not _rle_roundtrip_ok(crf_output, enc, shape=(101, 101)):
        bad_roundtrip += 1
        enc = _sanitize_rle(rle_in, shape=(101, 101))
        fallback_to_input += 1

    df.iat[row_i, rle_col] = enc

df["rle_mask"] = df["rle_mask"].where(df["rle_mask"].notna(), "").astype(str)
df.loc[df["rle_mask"].str.lower().eq("nan"), "rle_mask"] = ""
df["rle_mask"] = df["rle_mask"].map(lambda x: _sanitize_rle(x, shape=(101, 101)))

if bad_roundtrip > 0:
    print(
        f"Warning: {bad_roundtrip} masks failed post-process RLE roundtrip; "
        f"fell back to input RLE for {fallback_to_input} rows."
    )



## === cell 6
df = df[["id", "rle_mask"]].copy()
df["id"] = df["id"].astype(str)
df["rle_mask"] = df["rle_mask"].fillna("").astype(str)
df.loc[df["rle_mask"].str.lower().eq("nan"), "rle_mask"] = ""
df["rle_mask"] = df["rle_mask"].map(lambda x: _sanitize_rle(x, shape=(101, 101)))

sample_ids = sample_df["id"].astype(str).values
pred_map_final = dict(zip(df["id"].values, df["rle_mask"].values))
out_df = pd.DataFrame({"id": sample_ids})
out_df["rle_mask"] = out_df["id"].map(pred_map_final).fillna("").astype(str)
out_df.loc[out_df["rle_mask"].str.lower().eq("nan"), "rle_mask"] = ""
out_df["rle_mask"] = out_df["rle_mask"].map(
    lambda x: _sanitize_rle(x, shape=(101, 101))
)

out_path = "crf_correction.csv"
out_df.to_csv(out_path, index=False)

print(f"Wrote submission to: {out_path}")
print(out_df.head())
print("Rows:", len(out_df), "Unique ids:", out_df["id"].nunique())
assert len(out_df) == len(
    sample_df
), "Submission row count must match sample_submission.csv"
assert (
    out_df["id"].values == sample_ids
).all(), "Submission id order must match sample_submission.csv exactly"
assert set(out_df.columns.tolist()) == {
    "id",
    "rle_mask",
}, "Submission must have exactly columns ['id','rle_mask']"
assert (
    out_df["rle_mask"].map(lambda x: isinstance(x, str)).all()
), "All rle_mask must be strings"
