# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

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

np.random.seed(123)
_SE_DISK2 = disk(2)




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
    starts = np.asarray(s[0:][::2], dtype=np.int64)
    lengths = np.asarray(s[1:][::2], dtype=np.int64)
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

    Change (timeout-relevant, correctness-preserving): keep the same logic, but avoid any
    extra imports/work when CRF isn't available (common on Kaggle images).
    """
    if not _HAS_PYDENSECRF:
        return (mask_img > 0).astype(np.uint8)

    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_labels

    img = original_image
    if img.ndim == 2:
        img = gray2rgb(img)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    m = (mask_img > 0).astype(np.uint8)

    n_labels = 2
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)

    U = unary_from_labels(
        m.ravel().astype(np.int32), n_labels, gt_prob=0.7, zero_unsure=False
    )
    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(
        sxy=(3, 3),
        compat=3,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC,
    )

    Q = d.inference(10)
    MAP = np.argmax(Q, axis=0).astype(np.uint8)
    return MAP.reshape((img.shape[0], img.shape[1]))




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
sep_strength = []  # |mean_s - mean_b|, used to choose polarity more robustly

train_ids = train_df["id"].values

max_calib = min(1200, len(train_ids))
if len(train_ids) > max_calib:
    step = max(1, len(train_ids) // max_calib)
    calib_ids = train_ids[::step][:max_calib]
else:
    calib_ids = train_ids

z_vals = depths_df["z"].values.astype(np.float32)
z_p10, z_p90 = np.percentile(z_vals, [10, 90])
z_scale = float(max(1e-6, z_p90 - z_p10))


def _norm_and_smooth(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = img[..., 0]
    x = img.astype(np.float32, copy=False)
    if x.max() > 1.0:
        x = x / 255.0
    x = gaussian(x, sigma=0.7, preserve_range=True).astype(np.float32, copy=False)
    p2, p98 = np.percentile(x, [2, 98])
    if p98 > p2 + 1e-6:
        x = np.clip((x - p2) / (p98 - p2), 0.0, 1.0, out=x)
    return x


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool, copy=False)
    b = b.astype(bool, copy=False)
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    if union <= 0:
        return 1.0 if inter <= 0 else 0.0
    return inter / union


_THRESHOLDS = np.arange(0.5, 1.0, 0.05).astype(np.float32)


def _ap_like_single(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    iou = _iou(y_pred, y_true)
    return float(np.mean(iou > _THRESHOLDS))


for img_id in tqdm(calib_ids, desc="Calibrating basic priors from train (subset)"):
    img_path = os.path.join(TRAIN_IMG_DIR, f"{img_id}.png")
    msk_path = os.path.join(TRAIN_MASK_DIR, f"{img_id}.png")
    if (not os.path.isfile(img_path)) or (not os.path.isfile(msk_path)):
        continue

    x = _norm_and_smooth(imread(img_path))

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

calib_records = []
for img_id in tqdm(calib_ids[:600], desc="Preparing metric calibration samples"):
    img_path = os.path.join(TRAIN_IMG_DIR, f"{img_id}.png")
    msk_path = os.path.join(TRAIN_MASK_DIR, f"{img_id}.png")
    if (not os.path.isfile(img_path)) or (not os.path.isfile(msk_path)):
        continue
    x = _norm_and_smooth(imread(img_path))
    mask = imread(msk_path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    y = (mask > 127).astype(np.uint8, copy=False)
    try:
        thr0 = float(threshold_otsu(x))
    except Exception:
        continue
    z = float(depth_map.get(img_id, (z_p10 + z_p90) * 0.5))
    z_norm = float(np.clip((z - z_p10) / z_scale, 0.0, 1.0))
    calib_records.append((x, y, thr0, z_norm))

thr_bias_candidates = np.linspace(-0.06, 0.06, 13).astype(np.float32)

p_thresh_candidates = np.linspace(0.20, 0.85, 27).astype(np.float32)
k_candidates = np.asarray([8.0, 10.0, 12.0, 14.0, 16.0], dtype=np.float32)

best_score = -1.0
best_thr_bias = -0.02
best_p_thresh = 0.5
best_k_sig = 12.0




## === cell 5
def _keep_main_components(mask: np.ndarray, keep_area_frac: float = 0.98) -> np.ndarray:
    """
    Keep the largest components until they cover a high fraction of the predicted positive area.

    Optimization (provably equivalent): replace np.isin(lab, kept) (slow, allocates) with a
    boolean lookup-table indexed by labels (O(HW) with tiny extra memory).
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

    acc = 0.0
    kept = []
    for k in order:
        sz = int(sizes[k])
        if sz <= 0:
            continue
        kept.append(k)
        acc += float(sz)
        if acc / total >= keep_area_frac:
            break

    keep_lut = np.zeros(n + 1, dtype=bool)
    keep_lut[np.asarray(kept, dtype=np.int32)] = True
    out = keep_lut[lab]
    return out


def _postprocess_mask(mask: np.ndarray) -> np.ndarray:
    """
    morphology cleanup + component filtering to suppress scattered FP.
    """
    mask = binary_opening(mask, _SE_DISK2)
    mask = binary_closing(mask, _SE_DISK2)

    mask = remove_small_objects(mask, min_size=60)
    mask = remove_small_holes(mask, area_threshold=60)

    mask = _keep_main_components(mask, keep_area_frac=0.98)

    mask = mask.astype(np.uint8, copy=False)

    k = 3
    mask[:k, :k] = 0
    mask[:k, -k:] = 0
    mask[-k:, :k] = 0
    mask[-k:, -k:] = 0

    return mask


def _aplike_from_prob_and_side(
    x: np.ndarray, y: np.ndarray, thr: float, k_sig: float, pt: float, use_bright: bool
) -> float:
    p = 1.0 / (1.0 + np.exp(-float(k_sig) * (x - float(thr))))
    p_use = p if use_bright else (1.0 - p)
    m = p_use >= float(pt)
    m = _postprocess_mask(m)
    return _ap_like_single(y, m)


if len(calib_records) >= 50:
    pt_arr = p_thresh_candidates.astype(np.float32, copy=False)
    tb_arr = thr_bias_candidates.astype(np.float32, copy=False)
    k_arr = k_candidates.astype(np.float32, copy=False)

    y_trues = []
    y_true_sums = []
    xs = []
    thr0s = []
    for x, y, thr0, _zn in calib_records:
        xb = x.astype(np.float32, copy=False)
        yb = y > 0
        xs.append(xb)
        thr0s.append(float(thr0))
        y_trues.append(yb)
        y_true_sums.append(float(yb.sum()))
    y_true_sums = np.asarray(y_true_sums, dtype=np.float32)
    n_samples = len(y_trues)

    best_score = -1.0
    threshs = _THRESHOLDS  # (10,)

    for k_sig in tqdm(k_arr, desc="Grid-search k (logistic slope)"):
        for tb in tb_arr:
            ap_sum_by_pt = np.zeros(pt_arr.size, dtype=np.float64)

            for si in range(n_samples):
                x = xs[si]
                yb = y_trues[si]
                y_sum = y_true_sums[si]

                thr = float(np.clip(thr0s[si] + float(tb), 0.0, 1.0))
                p = 1.0 / (1.0 + np.exp(-float(k_sig) * (x - thr)))

                p_use = p if salt_brighter_prior else (1.0 - p)

                for pti, pt in enumerate(pt_arr):
                    pred = _postprocess_mask(p_use >= float(pt)).astype(
                        bool, copy=False
                    )
                    inter = float(np.logical_and(pred, yb).sum())
                    union = float(pred.sum() + y_sum - inter)
                    iou = (
                        (inter / union)
                        if union > 0.0
                        else (1.0 if inter <= 0.0 else 0.0)
                    )
                    ap = float(np.mean(iou > threshs))
                    ap_sum_by_pt[pti] += ap

            mean_ap_by_pt = (ap_sum_by_pt / float(n_samples)).astype(
                np.float32, copy=False
            )
            best_pti = int(np.argmax(mean_ap_by_pt))
            m_ap = float(mean_ap_by_pt[best_pti])

            if m_ap > best_score:
                best_score = m_ap
                best_thr_bias = float(tb)
                best_p_thresh = float(pt_arr[best_pti])
                best_k_sig = float(k_sig)
else:
    best_thr_bias = -0.02
    best_p_thresh = 0.5
    best_k_sig = 12.0

thr_bias = float(best_thr_bias)
p_thresh = float(best_p_thresh)
k_sig = float(best_k_sig)

if len(area_fracs) > 20:
    z_norms = []
    areas = []
    for img_id in calib_ids[: min(len(calib_ids), 800)]:
        img_path = os.path.join(TRAIN_IMG_DIR, f"{img_id}.png")
        if not os.path.isfile(img_path):
            continue
        msk_path = os.path.join(TRAIN_MASK_DIR, f"{img_id}.png")
        if not os.path.isfile(msk_path):
            continue
        mask = imread(msk_path)
        if mask.ndim == 3:
            mask = mask[..., 0]
        a = float((mask > 127).mean())
        z = float(depth_map.get(img_id, (z_p10 + z_p90) * 0.5))
        zn = float(np.clip((z - z_p10) / z_scale, 0.0, 1.0))
        z_norms.append(zn)
        areas.append(a)
    z_arr = np.asarray(z_norms, dtype=np.float32)
    a_arr = np.asarray(areas, dtype=np.float32)
    if z_arr.size > 30:
        x_ = z_arr - float(np.mean(z_arr))
        y_ = a_arr - float(np.mean(a_arr))
        denom = float(np.mean(x_ * x_) + 1e-6)
        slope = float(np.mean(x_ * y_) / denom)  # d(area)/d(z_norm)
        p_depth_coef = float(np.clip(-0.35 * slope, -0.06, 0.06))
    else:
        p_depth_coef = 0.0
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
        "k_sig": round(k_sig, 2),
        "calib_n": int(len(calib_ids)),
        "metric_calib_n": int(len(calib_records)),
        "best_mean_aplike": round(float(best_score), 4),
    },
)




## === cell 6
def baseline_mask_from_image(img: np.ndarray, img_id: str = None) -> np.ndarray:
    x = _norm_and_smooth(img)

    thr = float(threshold_otsu(x))
    thr = float(np.clip(thr + thr_bias, 0.0, 1.0))

    p = 1.0 / (1.0 + np.exp(-float(k_sig) * (x - thr)))

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
        mask_bool = mask_bin.astype(bool, copy=False)
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
    return mask.astype(np.uint8, copy=False)




## === cell 7
ids = df["id"].values
rles = [""] * len(ids)

_imread = imread
_join = os.path.join
_isfile = os.path.isfile
_rle_encode = rle_encode
_baseline = baseline_mask_from_image
_crf = crf

for i in tqdm(range(len(ids)), desc="Predict + (optional) CRF postprocess"):
    img_id = ids[i]
    img_path = _join(TEST_IMG_DIR, f"{img_id}.png")
    if not _isfile(img_path):
        rles[i] = ""
        continue

    orig_img = _imread(img_path)
    pred_mask = _baseline(orig_img, img_id=img_id)

    crf_output = _crf(orig_img, pred_mask)

    rles[i] = _rle_encode(crf_output)

df["rle_mask"] = pd.Series(rles, index=df.index)



## === cell 8
df["rle_mask"] = df["rle_mask"].fillna("").astype(str)

out_path = "crf_correction_unet_bn_diceloss.csv"
df.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(df))
print(df.head())
