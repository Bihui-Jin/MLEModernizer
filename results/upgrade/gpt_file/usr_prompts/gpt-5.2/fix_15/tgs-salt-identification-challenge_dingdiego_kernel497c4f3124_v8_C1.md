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

0.8134746862829938

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5221) has done: 'I remove the unavailable `pydensecrf` dependency (it’s causing the first hard failure) and replace the CRF refinement step with a lightweight, deterministic post-processing that keeps the same intent: refine predicted binary masks using the original image. I also convert the notebook-only commands (`%matplotlib inline`, `ls`) into Kaggle-script-safe code, and ensure all required imports are available in every executed cell. Finally, I fix NaN/empty-mask handling (the current `str(np.nan)` check is incorrect) and guarantee a valid `submission.csv` is written with the required `id,rle_mask` columns.'
- What this solution (achieved 0.5221) has done: 'Your current script only post-processes an existing submission (sample_submission or a found `k-fold-merger/submission.csv`) and can’t materially improve mAP unless that input submission already comes from a strong model. To move your score upward toward 0.813, the smallest “core-logic-preserving” improvement is to (1) automatically pick the best available base submission among common locations in `../input/` instead of defaulting to `sample_submission.csv`, and (2) tune the refinement strength very slightly to avoid over-eroding masks (which often hurts IoU on this competition). These changes keep the same approach (decode → refine using image → encode) and only adjust input selection + conservative post-processing so you can recover performance closer to the intended baseline. The code still runs end-to-end and always writes a valid `submission.csv`.'
- What this solution (achieved 0.5221) has done: 'Your current pipeline can only “refine” an existing submission, and with the current gating/morphology it’s likely over-eroding true positives, which depresses IoU and mAP. To move the score upward toward 0.813 with minimal core-logic change, I make the refinement more conservative: use softer image-based gating (keep more of the original mask), reduce hole/object removal aggressiveness, and apply a slightly gentler closing. I also ensure we preserve empty masks correctly and avoid creating invalid/degenerate RLE outputs by returning empty strings when the refined mask becomes empty. These changes keep the same decode → refine using image → encode approach and should improve recall/IoU, nudging the score closer to the target.'
- What this solution (achieved 0.0) has done: 'Your current score is far below the target, and the biggest likely issue is that the “refinement” step is suppressing too many positive pixels (hurting recall and IoU), which is especially damaging under mAP@IoU thresholds. I keep the exact same pipeline (load base submission → RLE decode → image-guided refine → RLE encode → write submission.csv), but make the refinement strictly more conservative by (1) keeping the original mask by default and only removing pixels that are *very* unlikely given local intensity, and (2) reducing morphological deletions that can remove thin structures. I also add a small safety rule to avoid turning near-empty masks into empty ones after refinement (another recall-killer), while preserving empty masks as empty. These are minimal, targeted changes intended to move the score upward toward your 0.813 target without changing the overall approach.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score strongly suggests a submission-format/alignment problem (e.g., wrong flattening order in RLE decode/encode consistency, or mismatched test image IDs/order). I keep your same “base submission → decode → image-guided refine → encode” pipeline, but fix the RLE decoding to match Kaggle’s required Fortran-order convention (your encoder is Fortran-order but your decoder currently reshapes in C-order, which breaks masks and can crater mAP). I also enforce that the output rows align exactly to `sample_submission.csv` ids (same ordering and complete coverage), while still sourcing the base masks from the best-found submission. These are minimal changes that preserve your core approach and should move the score up toward the target band.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with a submission-format rule violation rather than model quality: in this competition, RLE must be generated in column-major (Fortran) pixel order *and* the run pairs must be strictly increasing and non-overlapping—small encoding mistakes can lead to a near-zero/zero score. I keep your exact pipeline (base submission → RLE decode → image-guided refine → RLE encode), but I harden the RLE encoder/decoder to fully conform to Kaggle’s convention (including correct indexing and run construction) and add a lightweight sanitizer that re-decodes/re-encodes each output mask once to guarantee sorted, valid runs. This is a minimal change focused on turning an invalid/garbled submission into a valid one and should move the score up toward your target without changing your refinement logic. The output remain aligned exactly to `sample_submission.csv` ids and always write a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly coming from an RLE convention mismatch: in this competition, RLE runs are in column-major (Fortran) order, so the canonical approach is to **reshape in C order** after writing runs into a flat Fortran-indexed vector. Your current decoder reshapes with `order="F"`, which is the common “gotcha” that produces transposed/garbled masks and can crater the score even if the submission is formatted correctly. I make the smallest change: fix `rle_decode` to reshape in default C order (while keeping your encoder Fortran-flattening), and keep your existing sanitize + alignment pipeline unchanged. This preserves your core decode → refine → encode logic, but makes decode/encode consistent with Kaggle’s definition so the refinement isn’t operating on a wrong mask.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with a submission that decodes/encodes masks in an inconsistent pixel order, which can silently produce near-random masks after refinement even if the CSV format looks valid. I make the smallest, core-logic-preserving fix by correcting `rle_decode` to match the same Fortran-order convention used by `rle_encode` (reshape with `order="F"`). I also keep your existing “sanitize by decode→encode” step (now consistent) so every output RLE is canonical and competition-safe, and keep the same alignment to `sample_submission.csv` ids to avoid any ordering/id mismatch. These changes should move the score upward toward your target without changing the overall pipeline (base submission → decode → image-guided refine → encode).'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly coming from the decoder/encoder convention mismatch: your `rle_encode` is Fortran-order (correct), but the safest way to ensure decode/encode consistency is to make `rle_decode` explicitly reconstruct the flat vector in Fortran index space and then reshape in **C order** (the common Kaggle TGS implementation), otherwise the refinement step operates on a transposed/garbled mask and can collapse the score to ~0. I make that single, minimal change and keep the rest of your pipeline (base submission selection → decode → refine using image → encode → sanitize → align to sample ids) identical. I also keep the sanitize step so every output RLE is canonical after the fix. This should turn the submission from effectively-random masks into properly aligned masks, moving the score upward toward your target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with a broken RLE decode/encode convention (masks get effectively scrambled before/after refinement), so I make the smallest possible correction: decode explicitly reshape back with `order="F"` to match your Fortran-order encoder. I keep your overall pipeline identical (pick base submission → sanitize → image-guided refine → encode → sanitize → align to sample ids), but also harden the “sanitize” step to always canonicalize through the corrected decode/encode pair. These changes are directly aimed at turning the submission from invalid/garbled masks into properly aligned ones, which should move the score upward toward your 0.813 target. The script still runs end-to-end and writes a valid `submission.csv` with `id,rle_mask`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with a *mask orientation/order mismatch* rather than model quality: even if the CSV format is valid, a wrong RLE decode/encode convention scramble masks and crater mAP. I make the minimal, core-logic-preserving fix by changing `rle_decode` to the canonical TGS inverse of a Fortran-flattened RLE (build a flat vector in Fortran index space, then reshape in C order), while keeping your encoder and overall decode→refine→encode pipeline intact. I also keep the sanitize step (decode→encode) so outputs are guaranteed canonical under the corrected convention, and keep alignment to `sample_submission.csv` ids unchanged to avoid any id/order issues. This should move the score up from 0.0 toward your target without changing the approach.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with a broken RLE convention: your encoder uses Fortran order (correct for this competition), but your decoder currently reshapes in C order, so masks get effectively scrambled before refinement and re-encoding. I make the smallest core-logic-preserving fix by decoding with `reshape(..., order="F")` so decode/encode are true inverses. I keep your same pipeline (base submission selection → sanitize → refine using image → encode → sanitize → align to sample ids) unchanged otherwise, because fixing mask orientation alone should move the score up substantially toward the target. This still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 strongly indicates a mask/RLE convention mismatch rather than “model quality”, so the smallest score-improving change is to fix RLE decoding to be the true inverse of your Fortran-order encoder (the prior decode used an inconsistent reshape convention that can scramble masks). I keep the exact same pipeline (base submission → sanitize → image-guided refine → encode → sanitize → align to sample ids), but correct `rle_decode` to the canonical TGS inverse: write runs into a flat Fortran-indexed vector and then reshape in C order. This ensures the refinement operates on the intended mask orientation and the final RLEs match Kaggle’s expected pixel indexing, which should move the score up from 0.0 toward your target. Everything still runs end-to-end and writes a valid `submission.csv` with the required columns and ordering.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with a submission alignment/format bug rather than model quality, and the biggest red flag here is that your `rle_decode` isn’t the true inverse of your Fortran-order `rle_encode` (you fill a flat Fortran-indexed vector, but reshape in default C order). I make the minimal fix: reshape with `order="F"` so decode/encode are consistent with the competition’s column-major RLE convention, while keeping your overall pipeline (base submission → sanitize → image-guided refine → encode → sanitize → align) unchanged. I also add one small safety check to ensure `df` row order matches `sample_submission.csv` exactly (to avoid any accidental index/order drift). These changes are directly aimed at moving the score up from 0.0 toward your target by producing correctly oriented masks and a properly aligned submission.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

from skimage.io import imread
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import remove_small_holes, remove_small_objects, closing, disk
from tqdm import tqdm




## === cell 1
def rle_decode(rle_mask, shape=(101, 101)):
    """
    rle_mask: run-length as string formatted (start length), 1-indexed, Fortran-flattened indexing.
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    Change (score-improving toward target, core-logic-preserving):
    - Make rle_decode the true inverse of the Fortran-order rle_encode used below by reshaping with
      order="F". Without this, decoded masks can be transposed/scrambled, and the refine step will
      operate on the wrong pixels, often collapsing mAP to ~0.
    """
    if rle_mask is None or (isinstance(rle_mask, float) and np.isnan(rle_mask)):
        return np.zeros(shape, dtype=np.uint8)
    rle_mask = str(rle_mask).strip()
    if rle_mask == "" or rle_mask.lower() == "nan":
        return np.zeros(shape, dtype=np.uint8)

    s = rle_mask.split()
    if len(s) % 2 != 0:
        return np.zeros(shape, dtype=np.uint8)

    starts = np.asarray(s[0::2], dtype=np.int64) - 1  # to 0-index
    lengths = np.asarray(s[1::2], dtype=np.int64)
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        lo = int(max(lo, 0))
        hi = int(min(hi, img.size))
        if hi > lo:
            img[lo:hi] = 1

    return img.reshape(shape, order="F")




## === cell 2
def rle_encode(im):
    """
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted (start length) in Fortran order.

    Kept as-is (already canonical Fortran-order encoding).
    """
    im = (im > 0).astype(np.uint8)
    if im.sum() == 0:
        return ""

    pixels = im.flatten(order="F")
    padded = np.concatenate([[0], pixels, [0]])
    changes = np.where(padded[1:] != padded[:-1])[0]
    starts = changes[0::2] + 1  # to 1-indexed
    ends = changes[1::2] + 1
    lengths = ends - starts
    rle = np.empty((starts.size * 2,), dtype=np.int64)
    rle[0::2] = starts
    rle[1::2] = lengths
    return " ".join(map(str, rle.tolist()))


def _sanitize_rle(rle_str, shape=(101, 101)):
    """
    Keep sanitize step to guarantee sorted/non-overlapping canonical runs and consistent indexing.
    With the corrected rle_decode, this now reliably canonicalizes without scrambling masks.
    """
    if rle_str is None:
        return ""
    rle_str = str(rle_str).strip()
    if rle_str == "" or rle_str.lower() == "nan":
        return ""
    m = rle_decode(rle_str, shape=shape)
    return rle_encode(m)




## === cell 3
"""
Replacement for unavailable DenseCRF:
Refine a predicted mask using the original grayscale image via edge-preserving smoothing
and small morphological cleanups.

Core logic unchanged.
"""


def refine_mask_with_image(original_image, mask_img):
    if original_image.ndim == 3:
        original_image = original_image[..., 0]
    img = original_image.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0

    mask = (mask_img > 0).astype(bool)
    orig_sum = int(mask.sum())
    if orig_sum == 0:
        return mask.astype(np.uint8)

    sm = gaussian(img, sigma=0.5, preserve_range=True)

    try:
        t = float(threshold_otsu(sm))
    except Exception:
        t = float(np.mean(sm))

    gated = mask & (sm >= t * 0.20)

    gated = closing(gated, disk(1))

    gated = remove_small_holes(gated, area_threshold=2)
    gated = remove_small_objects(gated, min_size=2)

    new_sum = int(gated.sum())
    if new_sum == 0:
        return mask.astype(np.uint8)
    if new_sum < int(0.35 * orig_sum):
        return mask.astype(np.uint8)

    return gated.astype(np.uint8)




## === cell 4
BASE = "../input/tgs-salt-identification-challenge"
test_path = os.path.join(BASE, "test", "images")
sample_path = os.path.join(BASE, "sample_submission.csv")


def _find_best_base_submission():
    candidates = []

    known = "../input/k-fold-merger/submission.csv"
    if os.path.exists(known):
        candidates.append(known)

    root = "../input"
    if os.path.isdir(root):
        for d1 in sorted(os.listdir(root)):
            p1 = os.path.join(root, d1)
            if not os.path.isdir(p1):
                continue
            p_sub = os.path.join(p1, "submission.csv")
            if os.path.exists(p_sub):
                candidates.append(p_sub)
            try:
                for d2 in sorted(os.listdir(p1)):
                    p2 = os.path.join(p1, d2)
                    if not os.path.isdir(p2):
                        continue
                    p_sub2 = os.path.join(p2, "submission.csv")
                    if os.path.exists(p_sub2):
                        candidates.append(p_sub2)
            except Exception:
                pass

    candidates.append(sample_path)

    seen = set()
    uniq = []
    for c in candidates:
        if c not in seen:
            uniq.append(c)
            seen.add(c)

    scored = []
    for c in uniq:
        try:
            tmp = pd.read_csv(c)
            ok = ("id" in tmp.columns) and ("rle_mask" in tmp.columns)
            n = tmp.shape[0] if ok else -1
            non_empty = 0
            if ok and n > 0:
                non_empty = int(
                    (
                        ~tmp["rle_mask"].isna()
                        & (tmp["rle_mask"].astype(str).str.strip() != "")
                        & (tmp["rle_mask"].astype(str).str.lower().str.strip() != "nan")
                    ).sum()
                )
            scored.append((n, non_empty, c))
        except Exception:
            scored.append((-1, -1, c))

    scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return scored[0][2]


sub_path = _find_best_base_submission()
base_df = pd.read_csv(sub_path)

if "id" not in base_df.columns:
    raise ValueError(f"Input submission at {sub_path} must contain 'id' column.")
if "rle_mask" not in base_df.columns:
    raise ValueError(f"Input submission at {sub_path} must contain 'rle_mask' column.")

sample_df = pd.read_csv(sample_path)[["id"]]

df = sample_df.merge(base_df[["id", "rle_mask"]], on="id", how="left")
df["rle_mask"] = df["rle_mask"].fillna("")

df = df.set_index("id").loc[sample_df["id"]].reset_index()

df["rle_mask"] = df["rle_mask"].map(lambda x: _sanitize_rle(x, shape=(101, 101)))

print("Using base submission:", sub_path, "shape:", base_df.shape)
print("Aligned to sample_submission ids, shape:", df.shape)



## === cell 5
out_rle = []
missing_images = 0
refined_to_empty = 0
sanitized_changed = 0

for i in tqdm(range(df.shape[0]), total=df.shape[0]):
    img_id = df.loc[i, "id"]
    rle = df.loc[i, "rle_mask"]

    is_empty = (
        pd.isna(rle) or str(rle).strip() == "" or str(rle).lower().strip() == "nan"
    )
    if is_empty:
        out_rle.append("")  # empty mask
        continue

    img_path = os.path.join(test_path, f"{img_id}.png")
    if not os.path.exists(img_path):
        missing_images += 1
        out_rle.append(str(rle))
        continue

    orig_img = imread(img_path)
    decoded_mask = rle_decode(rle, shape=(101, 101))
    refined = refine_mask_with_image(orig_img, decoded_mask)

    if refined.sum() == 0:
        refined_to_empty += 1
        out_rle.append("")
    else:
        enc = rle_encode(refined)
        enc2 = _sanitize_rle(enc, shape=(101, 101))
        if enc2 != enc:
            sanitized_changed += 1
        out_rle.append(enc2)

df["rle_mask"] = out_rle



## === cell 6
df = df[["id", "rle_mask"]]
df.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", df.shape)
if missing_images:
    print("Warning: missing images encountered:", missing_images)
if refined_to_empty:
    print("Info: refined masks that became empty:", refined_to_empty)
if sanitized_changed:
    print("Info: RLE strings canonicalized (changed):", sanitized_changed)
print(df.head())
