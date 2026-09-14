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

0.808913890090869

# 6. Current score

0.0739

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I make the notebook run in this Kaggle environment by removing the hard dependency on `pydensecrf` (it’s not installed) and adding a safe fallback so the pipeline still produces a submission CSV. I also fix the broken/irrelevant input CSV path by loading the provided `sample_submission.csv` (or another existing submission if you point to one), and correct the `NaN` checks so empty masks are handled properly. The CRF step be skipped automatically when `pydensecrf` is unavailable, keeping the rest of the logic (RLE decode/encode and per-image processing loop) intact and ensuring `crf_correction_unet_bn_diceloss.csv` is always written. This yield a valid submission file end-to-end; score changes are not meaningfully optimizable here without an actual base prediction file, so the focus is on correctness and producing a valid CSV.'
- What this solution (achieved 0.0598) has done: 'Your current 0.0 score is consistent with submitting essentially the `sample_submission.csv`, which is not a real prediction and score near-zero; the minimal way to move toward your target is to generate a legitimate mask per test image instead of relying on the sample file’s dummy RLEs. I keep your pipeline structure (load IDs → per-image loop → optional CRF → RLE encode → write CSV) but switch the input `rle_mask` source to a simple, deterministic image-based baseline (Otsu threshold on normalized grayscale), which is a legitimate segmentation heuristic and should lift the score well above 0.0. I also fix the RLE encode/decode to use the competition’s expected column-major (Fortran) order to avoid format-related score loss. The CRF step remain optional and run only when `pydensecrf` exists; otherwise it safely skip as before, and the script still write `crf_correction_unet_bn_diceloss.csv`.'
- What this solution (achieved 0.0612) has done: 'Your current score (0.0598) is far below the target (0.8089), so we should improve legitimate mask quality while keeping your existing pipeline (per-image heuristic mask → optional CRF → RLE encode → write CSV) intact. The biggest low-risk gain here is fixing the RLE encode/decode to the competition’s required column-major (Fortran) order (your current encode uses a transposed C-flatten that can corrupt masks). Then, to raise baseline segmentation quality without changing the “simple image-threshold baseline” core idea, we keep Otsu thresholding but add small, deterministic morphological cleanup (remove tiny noise + fill tiny holes) and a simple border-cropping prior that reduces common edge false-positives in this dataset. These changes should move the score substantially upward toward the target while staying lightweight and within time limits, and still produce the required `crf_correction_unet_bn_diceloss.csv`.'
- What this solution (achieved 0.0462) has done: 'Your score is far below the target (0.0612 vs 0.8089; higher-is-better), so we need a legitimate uplift while keeping your existing “Otsu threshold → morphology cleanup → optional CRF → RLE” pipeline unchanged in spirit. The biggest low-risk issue is that raw Otsu on these images often needs a simple per-image contrast normalization; adding deterministic percentile stretching typically improves separability without changing the core approach. Next, your border-zeroing is very aggressive (kills true salt that touches borders); we replace it with a much lighter “corner suppression” prior that only clears a few pixels in the four corners. Finally, we do a tiny bit of threshold calibration (a small bias on Otsu) and slightly stronger morphology parameters; these are minimal tweaks that usually move mAP up materially while keeping runtime well under the limit and still writing the same submission CSV.'
- What this solution (achieved 0.0533) has done: 'Your current score (0.0462) is far below the target (0.8089), so we should increase it with minimal, low-risk changes that keep your existing “per-image Otsu threshold → morphology cleanup → optional CRF → RLE” pipeline intact. The biggest likely issue is mask polarity: depending on image intensity convention, salt may be brighter in many tiles, so always using `x < thr` can systematically invert predictions; I make the polarity deterministic by evaluating both `x < thr` and `x > thr` and picking the one that looks more plausible using a simple, safe prior (salt area shouldn’t be extreme). I also add a tiny connected-component keep-largest step (after cleanup) to reduce false positives without changing the overall approach. Everything else (paths, loop, optional CRF, RLE Fortran order, output filename) stays the same and still produces `crf_correction_unet_bn_diceloss.csv`.'
- What this solution (achieved 0.0563) has done: 'Your current score (0.0533) is far below the target (0.8089), so we should improve mask quality while keeping your existing “per-image Otsu threshold → morphology cleanup → optional CRF → RLE” pipeline intact. The biggest low-risk gain is to stop picking polarity based on a fixed target area (which can be wrong per image) and instead pick the polarity that best matches a simple, deterministic “salt prior” derived from the training set: salt tends to appear brighter than background on average, and its typical area fraction is learnable. I compute (1) whether salt is usually brighter than background and (2) the distribution of mask area fractions from `train.csv` + train masks, then use that to choose between `x<thr` and `x>thr` per image. I also calibrate the small Otsu bias from training (rather than a hard-coded -0.02) but keep the same core thresholding logic, morphology, and RLE encoding, and still write `crf_correction_unet_bn_diceloss.csv`.'
- What this solution (achieved 0.0527) has done: 'We keep your exact pipeline (per-image Otsu → morphology cleanup → optional CRF → RLE) but make two minimal, score-relevant fixes: (1) use a larger/safer training subset for learning the priors (area_mu/area_sigma, salt-brighter, thr_bias) so polarity/threshold bias selection is less noisy, and (2) replace the “keep only largest component” step with a “keep components up to a cumulative area fraction” rule so images with multiple salt blobs aren’t wrongly collapsed to one blob (a common AP killer). Both changes preserve the core logic and run fast, but should materially raise mAP from the current ~0.056 toward your target. Output path, submission schema, RLE Fortran order, and optional CRF behavior remain unchanged.'
- What this solution (achieved 0.0764) has done: 'We keep your exact pipeline (per-image Otsu threshold → morphology cleanup → optional CRF → RLE) but tighten two score-critical places that can heavily suppress mAP at IoU thresholds: (1) replace the “area fraction closeness” polarity choice with a training-calibrated “which side is more salt-like” score that uses both learned salt-vs-bg intensity direction and typical salt intensity separation, and (2) add a tiny, deterministic mask probability smoothing step (Gaussian on the normalized image, then threshold) which often stabilizes Otsu on this dataset without changing the core approach. We also add a simple depth prior (available in `depths.csv`) to slightly adjust the threshold bias by depth percentile, which is a minimal, legitimate calibration and typically improves generalization. Output filename, submission schema, RLE Fortran order, and optional CRF behavior remain unchanged, and it still runs within the time limit.'
- What this solution (achieved 0.0739) has done: 'We keep your exact heuristic pipeline (per-image Otsu → morphology cleanup → optional CRF → RLE) but make two minimal, score-relevant calibrations that typically help mAP a lot on this dataset: (1) learn and apply a global probability threshold (instead of implicitly using Otsu’s per-image threshold as the final decision), and (2) learn a depth-conditioned adjustment for that threshold using `depths.csv` (your current depth tweak is fixed and may be mis-scaled). This preserves your core logic (still uses Otsu and the same postprocessing), but improves calibration/consistency across images, which is critical for IoU-sweep AP. We also keep your Fortran-order RLE, optional CRF behavior, paths, and output filename unchanged, and ensure runtime remains under the limit by calibrating on a capped subset.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.io import imread
from skimage.color import gray2rgb
from skimage.filters import threshold_otsu
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    binary_opening,
    binary_closing,
    disk,
)
from skimage.measure import label
from skimage.filters import gaussian

_HAS_PYDENSECRF = False
try:
    import pydensecrf.densecrf as dcrf  # noqa: F401
    from pydensecrf.utils import unary_from_labels  # noqa: F401

    _HAS_PYDENSECRF = True
except ModuleNotFoundError:
    _HAS_PYDENSECRF = False

print("pydensecrf available:", _HAS_PYDENSECRF)

BASE_INPUT = "/kaggle/input/tgs-salt-identification-challenge"
TEST_IMG_DIR = os.path.join(BASE_INPUT, "test", "images")
SAMPLE_SUB_PATH = os.path.join(BASE_INPUT, "sample_submission.csv")

TRAIN_CSV_PATH = os.path.join(BASE_INPUT, "train.csv")
TRAIN_IMG_DIR = os.path.join(BASE_INPUT, "train", "images")
TRAIN_MASK_DIR = os.path.join(BASE_INPUT, "train", "masks")
DEPTHS_PATH = os.path.join(BASE_INPUT, "depths.csv")

assert os.path.isdir(TEST_IMG_DIR), f"Missing test image dir: {TEST_IMG_DIR}"
assert os.path.isfile(SAMPLE_SUB_PATH), f"Missing sample submission: {SAMPLE_SUB_PATH}"
assert os.path.isfile(TRAIN_CSV_PATH), f"Missing train.csv: {TRAIN_CSV_PATH}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing train image dir: {TRAIN_IMG_DIR}"
assert os.path.isdir(TRAIN_MASK_DIR), f"Missing train mask dir: {TRAIN_MASK_DIR}"
assert os.path.isfile(DEPTHS_PATH), f"Missing depths.csv: {DEPTHS_PATH}"




## === cell 1
def rle_decode(rle_mask: str, shape=(101, 101)):
    """
    rle_mask: run-length as string formatted (start length), 1-indexed
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    if rle_mask is None:
        return np.zeros(shape, dtype=np.uint8)
    if isinstance(rle_mask, float) and np.isnan(rle_mask):
        return np.zeros(shape, dtype=np.uint8)
    rle_mask = str(rle_mask).strip()
    if rle_mask == "" or rle_mask.lower() == "nan":
        return np.zeros(shape, dtype=np.uint8)

    s = rle_mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape, order="F")


def rle_encode(im: np.ndarray):
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
def crf(original_image: np.ndarray, mask_img: np.ndarray):
    """
    Apply DenseCRF post-processing if pydensecrf is available.
    If not available, return the input mask unchanged.
    """
    if not _HAS_PYDENSECRF:
        return (mask_img > 0).astype(np.uint8)

    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_labels

    if len(mask_img.shape) < 3:
        mask_img = gray2rgb(mask_img)

    annotated_label = (
        mask_img[:, :, 0] + (mask_img[:, :, 1] << 8) + (mask_img[:, :, 2] << 16)
    )
    _, labels = np.unique(annotated_label, return_inverse=True)

    n_labels = 2
    d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)

    U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(
        sxy=(3, 3),
        compat=3,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC,
    )

    Q = d.inference(10)
    MAP = np.argmax(Q, axis=0)
    return MAP.reshape((original_image.shape[0], original_image.shape[1])).astype(
        np.uint8
    )




## === cell 3
df = pd.read_csv(SAMPLE_SUB_PATH)
print("Loaded sample_submission shape:", df.shape)
print(df.head())

required_cols = {"id", "rle_mask"}
missing = required_cols - set(df.columns)
assert not missing, f"Submission missing columns: {missing}"

missing_imgs = 0
for _id in df["id"].iloc[:20]:
    if not os.path.isfile(os.path.join(TEST_IMG_DIR, f"{_id}.png")):
        missing_imgs += 1
print("Missing images among first 20 ids:", missing_imgs)


## === cell 4
train_df = pd.read_csv(TRAIN_CSV_PATH)
depths_df = pd.read_csv(DEPTHS_PATH)
depth_map = dict(zip(depths_df["id"].values, depths_df["z"].values.astype(np.float32)))

area_fracs = []
delta_means = []  # mean(salt_pixels) - mean(bg_pixels) on normalized image
thr_residuals = []  # (midpoint(salt,bg)) - otsu threshold
sep_strength = []  # |mean_s - mean_b|, used to choose polarity more robustly

p_thresh_list = []
z_list = []
p_list = []
y_list = []

train_ids = train_df["id"].values

max_calib = min(2000, len(train_ids))
if len(train_ids) > max_calib:
    step = max(1, len(train_ids) // max_calib)
    calib_ids = train_ids[::step][:max_calib]
else:
    calib_ids = train_ids

z_vals = depths_df["z"].values.astype(np.float32)
z_p10, z_p90 = np.percentile(z_vals, [10, 90])
z_scale = float(max(1e-6, z_p90 - z_p10))

for img_id in tqdm(calib_ids, desc="Calibrating priors from train (subset)"):
    img_path = os.path.join(TRAIN_IMG_DIR, f"{img_id}.png")
    msk_path = os.path.join(TRAIN_MASK_DIR, f"{img_id}.png")
    if (not os.path.isfile(img_path)) or (not os.path.isfile(msk_path)):
        continue

    img = imread(img_path)
    if img.ndim == 3:
        img = img[..., 0]
    x = img.astype(np.float32)
    if x.max() > 1.0:
        x /= 255.0

    x = gaussian(x, sigma=0.7, preserve_range=True).astype(np.float32)
    p2, p98 = np.percentile(x, [2, 98])
    if p98 > p2 + 1e-6:
        x = np.clip((x - p2) / (p98 - p2), 0.0, 1.0)

    mask = imread(msk_path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    m = mask > 127

    frac = float(m.mean())
    area_fracs.append(frac)

    if (m.sum() > 0) and ((~m).sum() > 0):
        mean_s = float(x[m].mean())
        mean_b = float(x[~m].mean())
        d = mean_s - mean_b
        delta_means.append(d)
        sep_strength.append(abs(d))

        try:
            thr = float(threshold_otsu(x))
            midpoint = 0.5 * (mean_s + mean_b)
            thr_residuals.append(midpoint - thr)
        except Exception:
            pass

    try:
        thr_img = float(threshold_otsu(x))
    except Exception:
        continue
    k = 12.0
    p = 1.0 / (1.0 + np.exp(-k * (x - thr_img)))

    y_mean = float(m.mean())
    p_mean = float(p.mean())
    p_s = float(p[m].mean()) if m.any() else p_mean
    p_b = float(p[~m].mean()) if (~m).any() else p_mean
    bright_is_salt = p_s >= p_b
    if not bright_is_salt:
        p = 1.0 - p

    p_sub = p.ravel()[::29]
    y_sub = m.astype(np.uint8).ravel()[::29]
    p_list.append(p_sub)
    y_list.append(y_sub)
    if img_id in depth_map:
        z = float(depth_map[img_id])
        z_norm = float(np.clip((z - z_p10) / z_scale, 0.0, 1.0))
        z_list.append(z_norm)
    else:
        z_list.append(0.5)

if len(area_fracs) == 0:
    area_mu, area_sigma = 0.15, 0.10
else:
    area_mu = float(np.mean(area_fracs))
    area_sigma = float(np.std(area_fracs) + 1e-6)

if len(delta_means) == 0:
    salt_brighter_prior = True
    sep_mu = 0.10
else:
    salt_brighter_prior = float(np.mean(delta_means)) >= 0.0
    sep_mu = float(np.median(sep_strength))

if len(thr_residuals) == 0:
    thr_bias = -0.02
else:
    thr_bias = float(np.median(thr_residuals))
    thr_bias = float(np.clip(thr_bias, -0.08, 0.08))

if len(p_list) == 0:
    p_thresh = 0.5
else:
    P = np.concatenate(p_list)
    Y = np.concatenate(y_list)
    y_target = float(Y.mean()) if Y.size else area_mu
    candidates = np.linspace(0.35, 0.75, 41)
    best_t, best_err = 0.5, 1e9
    for t in candidates:
        err = abs(float((P >= t).mean()) - y_target)
        if err < best_err:
            best_err, best_t = err, float(t)
    p_thresh = float(best_t)

if len(z_list) > 10 and len(area_fracs) > 10:
    z_arr = np.asarray(z_list[: len(area_fracs)], dtype=np.float32)
    a_arr = np.asarray(area_fracs[: len(z_arr)], dtype=np.float32)
    zc = z_arr - float(np.mean(z_arr))
    ac = a_arr - float(np.mean(a_arr))
    cov = float(np.mean(zc * ac))
    p_depth_coef = float(np.clip(0.06 * np.sign(cov), -0.06, 0.06))
else:
    p_depth_coef = 0.0

print(
    "Learned priors:",
    {
        "area_mu": round(area_mu, 4),
        "area_sigma": round(area_sigma, 4),
        "salt_brighter": salt_brighter_prior,
        "thr_bias": round(thr_bias, 4),
        "sep_mu": round(sep_mu, 4),
        "z_p10": float(z_p10),
        "z_p90": float(z_p90),
        "p_thresh": round(p_thresh, 4),
        "p_depth_coef": round(p_depth_coef, 4),
        "calib_n": int(len(calib_ids)),
    },
)




## === cell 5
def _keep_main_components(mask: np.ndarray, keep_area_frac: float = 0.98) -> np.ndarray:
    """
    Keep the largest components until they cover a high fraction of the predicted positive area.
    """
    lab = label(mask)
    n = int(lab.max())
    if n <= 1:
        return mask

    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    total = float(sizes.sum())
    if total <= 0:
        return mask

    order = np.argsort(sizes)[::-1]
    order = order[order != 0]

    kept = []
    acc = 0.0
    for k in order:
        if sizes[k] <= 0:
            continue
        kept.append(k)
        acc += float(sizes[k])
        if acc / total >= keep_area_frac:
            break

    out = np.isin(lab, kept)
    return out


def _postprocess_mask(mask: np.ndarray) -> np.ndarray:
    """
    morphology cleanup + component filtering to suppress scattered FP.
    """
    se = disk(2)
    mask = binary_opening(mask, se)
    mask = binary_closing(mask, se)

    mask = remove_small_objects(mask, min_size=60)
    mask = remove_small_holes(mask, area_threshold=60)

    mask = _keep_main_components(mask, keep_area_frac=0.98)

    mask = mask.astype(np.uint8)

    k = 3
    mask[:k, :k] = 0
    mask[:k, -k:] = 0
    mask[-k:, :k] = 0
    mask[-k:, -k:] = 0

    return mask


def baseline_mask_from_image(img: np.ndarray, img_id: str = None) -> np.ndarray:
    if img.ndim == 3:
        img = img[..., 0]
    x = img.astype(np.float32)
    if x.max() > 1.0:
        x /= 255.0

    x = gaussian(x, sigma=0.7, preserve_range=True).astype(np.float32)

    p2, p98 = np.percentile(x, [2, 98])
    if p98 > p2 + 1e-6:
        x = np.clip((x - p2) / (p98 - p2), 0.0, 1.0)

    thr = float(threshold_otsu(x))

    thr = float(np.clip(thr + thr_bias, 0.0, 1.0))

    k = 12.0
    p = 1.0 / (1.0 + np.exp(-k * (x - thr)))

    p_bright = p
    p_dark = 1.0 - p

    db = 0.0
    if img_id is not None and img_id in depth_map:
        z = float(depth_map[img_id])
        z_norm = float(np.clip((z - z_p10) / z_scale, 0.0, 1.0))
        db = float(p_depth_coef * (z_norm - 0.5))

    t_img = float(np.clip(p_thresh + db, 0.25, 0.85))

    m_dark = p_dark >= t_img
    m_bright = p_bright >= t_img

    m_dark = _postprocess_mask(m_dark)
    m_bright = _postprocess_mask(m_bright)

    frac_dark = float(m_dark.mean())
    frac_bright = float(m_bright.mean())

    def _score(mask_bin: np.ndarray, frac: float) -> float:
        mask_bool = mask_bin.astype(bool)
        if mask_bool.sum() == 0 or (~mask_bool).sum() == 0:
            sep = 0.0
            dmean = 0.0
        else:
            mean_s = float(x[mask_bool].mean())
            mean_b = float(x[~mask_bool].mean())
            dmean = mean_s - mean_b
            sep = abs(dmean)

        dir_score = 1.0 if ((dmean >= 0.0) == salt_brighter_prior) else -1.0
        sep_score = sep / float(sep_mu + 1e-6)
        area_pen = abs(frac - area_mu) / float(area_sigma + 1e-6)

        return (1.2 * dir_score * sep_score) - (0.25 * area_pen)

    s_dark = _score(m_dark, frac_dark)
    s_bright = _score(m_bright, frac_bright)

    mask = m_bright if (s_bright >= s_dark) else m_dark
    return mask.astype(np.uint8)




## === cell 6
for i in tqdm(range(df.shape[0]), desc="Predict + (optional) CRF postprocess"):
    img_id = df.loc[i, "id"]
    img_path = os.path.join(TEST_IMG_DIR, f"{img_id}.png")
    if not os.path.isfile(img_path):
        df.loc[i, "rle_mask"] = ""
        continue

    orig_img = imread(img_path)
    pred_mask = baseline_mask_from_image(orig_img, img_id=img_id)

    crf_output = crf(orig_img, pred_mask)

    df.loc[i, "rle_mask"] = rle_encode(crf_output)


## === cell 7
df["rle_mask"] = df["rle_mask"].fillna("").astype(str)

out_path = "crf_correction_unet_bn_diceloss.csv"
df.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(df))
print(df.head())
