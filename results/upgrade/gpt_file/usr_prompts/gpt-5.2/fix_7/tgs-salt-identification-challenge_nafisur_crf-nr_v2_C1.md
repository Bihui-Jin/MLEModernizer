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

0.77029

# 6. Current score

0.084

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5221) has done: 'Your notebook currently fails because it depends on `pydensecrf` (not installed) and on a missing input file (`../input/subnr1/submission.csv`), so it never reaches the CSV-writing step. I remove the unavailable CRF dependency while preserving the core intent (post-processing a predicted mask) by replacing CRF with a lightweight morphological cleanup using `scikit-image`, which is installed. I also switch the input to `sample_submission.csv` so the pipeline always runs end-to-end and produces a valid `submission.csv` with the required `id,rle_mask` columns. Finally, I ensure all plotting and notebook-magics are optional/removed so the script runs in a plain Kaggle execution environment.'
- What this solution (achieved 0.5221) has done: 'Your current score (0.5221) is far below the target (0.77029), so we should improve performance with minimal, safe changes that keep your core “refine an initial mask using the original image” logic intact. The biggest issue is that you’re refining the *sample submission* masks (essentially empty/dummy), so your Otsu fallback is generating noisy, untrained segmentations; we can instead build a much stronger “initial mask” from the image itself using a deterministic depth-aware threshold (depths.csv is available) and then apply the same morphological cleanup. To better match the IoU-sweep metric, we also tune the cleanup parameters slightly to reduce false positives (which heavily hurt precision across thresholds). Finally, we keep the same submission writing and RLE conventions, ensuring a valid `submission.csv`.'
- What this solution (achieved 0.5221) has done: 'We keep your current “image → initial mask (Otsu + depth tweak) → morphological cleanup → RLE” core logic intact, but make two minimal scoring-oriented fixes: (1) correct the run-length decoding reshape to Fortran order to match the competition’s RLE convention (this alone can significantly lift IoU/AP if the decoded mask was effectively transposed), and (2) add a tiny, deterministic “empty mask suppression” after refinement to reduce false positives (which heavily penalize AP across IoU thresholds). These changes preserve the same approach and semantics (still deterministic post-processing of each image) while making the encoding/decoding consistent and reducing FP noise. The script still run end-to-end and write a valid `submission.csv` with `id,rle_mask`.'
- What this solution (achieved 0.0771) has done: 'We need to raise the score from 0.5221 toward 0.77029 (higher-is-better), so the smallest safe gains come from better mask post-processing and metric-aligned binarization while keeping your current “image→initial mask (Otsu+depth tweak)→morph cleanup→RLE” core logic. I add a single deterministic test-time augmentation (horizontal flip) and average the two probability-like masks before thresholding, which usually improves IoU without changing the training approach (there is none). I also switch the refinement pipeline to operate on a soft mask (float 0..1) and apply a fixed threshold tuned to reduce false positives (which hurt AP across IoU thresholds), while preserving the same morphology steps. Finally, I add a tiny “edge padding” before morphology to avoid border artifacts from operations near image edges, then crop back—this often stabilizes objects touching borders.'
- What this solution (achieved 0.0789) has done: 'Your current score (0.0771) is far below the target (0.77029), so we need a real uplift while keeping your “image → Otsu(+depth) → morph cleanup → RLE” core logic unchanged. The biggest likely cause of the collapse is that the pipeline thresholds a *soft* mask with a fixed cutoff (0.55) without calibrating it to this heuristic method, which can massively under/over-segment; I switch to a per-image threshold derived from the soft mask itself (Otsu on the soft mask), which preserves the same logic but makes binarization adaptive and typically much closer to a reasonable mask. I also add a small deterministic “empty image” guard: if the image is nearly constant (common in this dataset), predict empty to avoid false positives that heavily hurt AP across IoU thresholds. Finally, I keep your TTA and morphology, but make the “empty suppression” depend on mask area ratio (not a fixed pixel count) to be scale-consistent and reduce noisy positives.'
- What this solution (achieved 0.084) has done: 'Your score is far below target, so we should improve without changing your core “image → (Otsu+depth) soft mask → threshold → morphology → RLE” pipeline. The biggest low-risk gain is to tune the final binarization threshold to reduce systematic over/under-segmentation: we compute a per-image threshold from the soft mask but bias it slightly upward (more conservative) and make it depth-aware, which typically reduces false positives and improves mean AP across IoU thresholds. We also add a minimal “empty if too small OR too large” area guard (large masks are common failure cases for this heuristic) while keeping all morphology and TTA intact. Finally, we make output deterministic and ensure IDs are processed in submission order as-is.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(0)

CANDIDATE_INPUT_ROOTS = [
    "/kaggle/input/tgs-salt-identification-challenge",
    "/kaggle/data/tgs-salt-identification-challenge",
    "/kaggle/input",
    "/kaggle/data",
    "../input/tgs-salt-identification-challenge",
    "../input",
    "../data/tgs-salt-identification-challenge",
    "../data",
]


def find_existing_path(rel_path):
    for root in CANDIDATE_INPUT_ROOTS:
        p = os.path.join(root, rel_path)
        if os.path.exists(p):
            return p
    return None


sample_sub_path = find_existing_path("sample_submission.csv")
if sample_sub_path is None:
    sample_sub_path = find_existing_path(
        "tgs-salt-identification-challenge/sample_submission.csv"
    )

test_images_dir = find_existing_path("test/images")
if test_images_dir is None:
    test_images_dir = find_existing_path(
        "tgs-salt-identification-challenge/test/images"
    )

depths_path = find_existing_path("depths.csv")
if depths_path is None:
    depths_path = find_existing_path("tgs-salt-identification-challenge/depths.csv")

print("sample_submission.csv:", sample_sub_path)
print("test/images dir:", test_images_dir)
print("depths.csv:", depths_path)
print(
    "Listing a likely input dir:",
    os.listdir(os.path.dirname(sample_sub_path)) if sample_sub_path else "NOT FOUND",
)



## === cell 1
from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    binary_closing,
    binary_opening,
    disk,
)


def rle_decode(rle_mask):
    """
    rle_mask: run-length as string formatted (start length)
    Returns numpy array, 1 - mask, 0 - background, shape (101, 101)

    Decode uses Fortran order to match the competition’s column-major indexing,
    consistent with rle_encode(flatten(order="F")).
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


def rle_encode(im):
    """
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    """
    im = (im > 0).astype(np.uint8)
    pixels = im.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 2
if sample_sub_path is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv in any known input roots."
    )

df = pd.read_csv(sample_sub_path)
assert set(["id", "rle_mask"]).issubset(
    df.columns
), "sample_submission.csv must have columns: id,rle_mask"
df.head()



## === cell 3
if depths_path is None:
    raise FileNotFoundError("Could not locate depths.csv in any known input roots.")
depths_df = pd.read_csv(depths_path)
assert set(["id", "z"]).issubset(
    depths_df.columns
), "depths.csv must have columns: id,z"
depth_map = dict(zip(depths_df["id"].values, depths_df["z"].values))

z_values = depths_df["z"].values.astype(np.float32)
z_min, z_max = float(np.min(z_values)), float(np.max(z_values))
z_rng = (z_max - z_min) if (z_max > z_min) else 1.0



## === cell 4
"""
Score-oriented minimal adjustments while preserving core logic:
- Keep: image -> initial soft mask via Otsu(+depth) -> threshold -> morphology -> RLE, with same TTA.
- Change (small, score-relevant): make the adaptive threshold slightly more conservative and depth-aware.
  This tends to reduce false positives, which heavily hurt mean AP across IoU thresholds.
- Change (small, score-relevant): add a guard for implausibly large predicted masks (heuristic methods often
  "flood fill" large areas), setting them to empty to avoid catastrophic FP-heavy cases.
"""


def _img_to_01(img):
    if img.ndim == 3:
        img = img[..., 0]
    img = img.astype(np.float32)
    mx = float(img.max()) if img.size else 0.0
    if mx > 1.5:
        img01 = img / 255.0
    else:
        img01 = img.copy()
    return np.clip(img01, 0.0, 1.0)


def _initial_softmask_from_image(img, z_value=None):
    """
    Same intent as before (Otsu + depth tweak), returning a *soft* mask in [0,1].
    """
    img01 = _img_to_01(img)

    try:
        thr = float(threshold_otsu(img01))
    except Exception:
        thr = float(img01.mean())

    if z_value is not None:
        z_norm = (float(z_value) - z_min) / z_rng  # 0..1
        thr = thr + (z_norm - 0.5) * 0.06  # keep same adjustment scale

    thr = float(np.clip(thr, 0.05, 0.95))

    k = 12.0
    soft = 1.0 / (1.0 + np.exp(-k * (img01 - thr)))
    return soft.astype(np.float32)


def _morph_refine(binary_mask):
    """
    Morphology unchanged in spirit; pad by 1px to reduce border artifacts.
    """
    m = binary_mask.astype(bool)
    m = np.pad(m, ((1, 1), (1, 1)), mode="edge")

    m = binary_opening(m, footprint=disk(1))
    m = binary_closing(m, footprint=disk(1))

    m = remove_small_objects(m, min_size=30)
    m = remove_small_holes(m, area_threshold=30)

    m = m[1:-1, 1:-1]
    return m.astype(np.uint8)


def _adaptive_soft_threshold(soft, z_value=None):
    """
    (score-relevant) Adaptive thresholding of the soft mask.

    Minimal change vs your current approach:
    - still Otsu on the soft mask,
    - but apply a small upward bias (more conservative),
    - and a small depth-dependent bias to reduce FP on deeper/shallower extremes.
    """
    soft = np.clip(soft.astype(np.float32), 0.0, 1.0)
    try:
        t = float(threshold_otsu(soft))
    except Exception:
        t = float(np.mean(soft))

    t = t + 0.03

    if z_value is not None:
        z_norm = (float(z_value) - z_min) / z_rng  # 0..1
        t = t + (z_norm - 0.5) * 0.02

    return float(np.clip(t, 0.40, 0.80))


def crf(original_image, mask_img, z_value=None):
    """
    Deterministic refinement: if an input mask exists, use it; otherwise derive from image.
    Then apply threshold + morphology cleanup.
    """
    if mask_img is None:
        mask_sum = 0
    else:
        mask_sum = (
            float(np.sum(mask_img > 0.5))
            if mask_img.dtype != np.uint8
            else int(mask_img.sum())
        )

    img01 = _img_to_01(original_image)

    if float(img01.std()) < 0.02:
        return np.zeros((101, 101), dtype=np.uint8)

    if mask_sum == 0:
        soft = _initial_softmask_from_image(original_image, z_value=z_value)
    else:
        soft = mask_img.astype(np.float32)
        if soft.max() > 1.5:
            soft = soft / 255.0
        soft = np.clip(soft, 0.0, 1.0)

    t = _adaptive_soft_threshold(soft, z_value=z_value)
    bin_mask = soft > t

    refined = _morph_refine(bin_mask)

    area = float(refined.sum()) / float(refined.size)

    if area < 0.0015:
        refined[:] = 0

    if area > 0.65:
        refined[:] = 0

    return refined.astype(np.uint8)


def predict_with_tta(img, z_value=None):
    """
    Horizontal-flip TTA on the *soft* initial mask, then threshold+refine once.
    """
    soft1 = _initial_softmask_from_image(img, z_value=z_value)
    img_f = np.fliplr(img)
    soft2 = _initial_softmask_from_image(img_f, z_value=z_value)
    soft2 = np.fliplr(soft2)

    soft_mean = 0.5 * (soft1 + soft2)

    return crf(img, soft_mean, z_value=z_value)




## === cell 5
if test_images_dir is None:
    raise FileNotFoundError(
        "Could not locate test/images directory in any known input roots."
    )
test_path = (
    test_images_dir if test_images_dir.endswith("/") else (test_images_dir + "/")
)



## === cell 6
try:
    import matplotlib.pyplot as plt

    DO_PLOT = False
except Exception:
    DO_PLOT = False

if DO_PLOT:
    nImgs = 3
    ids = df["id"].iloc[:nImgs].tolist()
    plt.figure(figsize=(12, 4 * nImgs))
    for j, img_id in enumerate(ids, start=1):
        decoded_mask = rle_decode(df.loc[df["id"] == img_id, "rle_mask"].values[0])
        orig_img = imread(test_path + img_id + ".png")
        out = predict_with_tta(orig_img, z_value=depth_map.get(img_id, None))
        plt.subplot(nImgs, 3, 3 * j - 2)
        plt.imshow(orig_img, cmap="gray")
        plt.title(f"{img_id} - image")
        plt.axis("off")
        plt.subplot(nImgs, 3, 3 * j - 1)
        plt.imshow(decoded_mask, cmap="gray")
        plt.title("input mask")
        plt.axis("off")
        plt.subplot(nImgs, 3, 3 * j)
        plt.imshow(out, cmap="gray")
        plt.title("refined mask (TTA)")
        plt.axis("off")
    plt.tight_layout()



## === cell 7
from tqdm import tqdm



## === cell 8
"""
Apply the refinement on each test image and write back RLE.
This produces a valid submission.csv with the required columns and .csv suffix.
"""
for i in tqdm(range(df.shape[0]), desc="Refining masks"):
    img_id = df.loc[i, "id"]

    img_path = test_path + img_id + ".png"
    orig_img = imread(img_path)

    refined = predict_with_tta(orig_img, z_value=depth_map.get(img_id, None))
    df.loc[i, "rle_mask"] = rle_encode(refined)



## === cell 9
out_path = "submission.csv"
df.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(df.head())
