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
from tqdm import tqdm
from skimage.io import imread
from skimage.color import gray2rgb
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.transform import resize

import matplotlib.pyplot as plt

CRF_AVAILABLE = False
try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_labels

    CRF_AVAILABLE = True
except ModuleNotFoundError:
    CRF_AVAILABLE = False

INPUT_ROOT = "/kaggle/input/tgs-salt-identification-challenge"
test_path = os.path.join(INPUT_ROOT, "test", "images")
train_img_path = os.path.join(INPUT_ROOT, "train", "images")
train_csv_path = os.path.join(INPUT_ROOT, "train.csv")
sample_sub_path = os.path.join(INPUT_ROOT, "sample_submission.csv")

np.random.seed(42)




## === cell 1
def rle_decode(rle_mask):
    """
    rle_mask: run-length as string formatted (start length)
    Returns numpy array, 1 - mask, 0 - background, shape (101, 101)
    """
    if rle_mask is None:
        return np.zeros((101, 101), dtype=np.uint8)
    rle_mask = str(rle_mask)
    if rle_mask.strip() == "" or rle_mask.lower() == "nan":
        return np.zeros((101, 101), dtype=np.uint8)

    s = rle_mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(101 * 101, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(101, 101)


def rle_encode(im):
    """
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted.
    """
    pixels = im.astype(np.uint8).flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 2
def crf(original_image, mask_img):
    """
    Return the labelled image after applying CRF.

    If pydensecrf isn't installed, return the input mask unchanged.
    """
    if not CRF_AVAILABLE:
        return (mask_img > 0).astype(np.uint8)

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
    MAP = np.argmax(Q, axis=0).astype(np.uint8)
    return MAP.reshape((original_image.shape[0], original_image.shape[1]))




## === cell 3
df_sub = pd.read_csv(sample_sub_path)
assert "id" in df_sub.columns and "rle_mask" in df_sub.columns

df_train = pd.read_csv(train_csv_path)
assert "id" in df_train.columns and "rle_mask" in df_train.columns

for _id in df_sub["id"].head(5).tolist():
    p = os.path.join(test_path, f"{_id}.png")
    if not os.path.exists(p):
        raise FileNotFoundError("Test images not found at expected path: " + test_path)

for _id in df_train["id"].head(5).tolist():
    p = os.path.join(train_img_path, f"{_id}.png")
    if not os.path.exists(p):
        raise FileNotFoundError(
            "Train images not found at expected path: " + train_img_path
        )

print("Test rows:", len(df_sub), "Train rows:", len(df_train))
df_sub.head()




## === cell 4
def _ensure_gray01(img):
    if img.ndim == 3:
        img = img[:, :, 0]
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32)


def img_to_feat(img_101x101, out_size=26):
    """
    Keep same core logic: downsampled pixel feature.
    Use anti-aliased resize to make kNN distances stable.
    """
    img = _ensure_gray01(img_101x101)
    small = resize(
        img,
        (out_size, out_size),
        order=1,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float32, copy=False)
    feat = small.reshape(-1)
    if feat.shape[0] != out_size * out_size:
        raise ValueError(
            f"Unexpected feature length {feat.shape[0]} for out_size={out_size}"
        )
    return feat


def img_to_feat_multi(img_101x101, out_sizes=(20, 26, 32)):
    feats = [img_to_feat(img_101x101, out_size=s) for s in out_sizes]
    return np.concatenate(feats, axis=0).astype(np.float32, copy=False)


_THRESHOLDS = np.arange(0.5, 1.0, 0.05, dtype=np.float32)


def map_iou_tgs(y_true_batch, y_pred_batch):
    """
    Vectorized TGS Salt metric for single-object masks (same semantics as original).
    """
    thresholds = _THRESHOLDS  # (T,)
    yt = (y_true_batch > 0).reshape(y_true_batch.shape[0], -1)
    yp = (y_pred_batch > 0).reshape(y_pred_batch.shape[0], -1)

    yt_sum = yt.sum(axis=1)
    yp_sum = yp.sum(axis=1)
    yt_has = yt_sum > 0
    yp_has = yp_sum > 0

    inter = np.logical_and(yt, yp).sum(axis=1).astype(np.float32)
    union = np.logical_or(yt, yp).sum(axis=1).astype(np.float32)

    iou = np.zeros_like(union, dtype=np.float32)
    nz = union > 0
    iou[nz] = inter[nz] / union[nz]

    scores = np.zeros((yt.shape[0], thresholds.shape[0]), dtype=np.float32)

    both_empty = (~yt_has) & (~yp_has)
    exactly_one_empty = yt_has ^ yp_has
    normal = ~(both_empty | exactly_one_empty)

    if both_empty.any():
        scores[both_empty, :] = 1.0
    if normal.any():
        scores[normal, :] = (iou[normal, None] > thresholds[None, :]).astype(np.float32)

    return float(scores.mean())


def postprocess_mask(mask01, min_obj=12, min_hole=12):
    m = (mask01 > 0).astype(bool)
    if min_obj > 0:
        m = remove_small_objects(m, min_size=int(min_obj), connectivity=1)
    if min_hole > 0:
        m = remove_small_holes(m, area_threshold=int(min_hole), connectivity=1)
    return m.astype(np.uint8)


out_size = 26
feat_scales = (20, 26, 32)
D = int(sum(s * s for s in feat_scales))

df_train_by_id = df_train.set_index("id")

train_ids = df_train["id"].values
train_feats = np.empty((len(train_ids), D), dtype=np.float32)
train_masks = np.empty((len(train_ids), 101, 101), dtype=np.uint8)

for i, tid in enumerate(tqdm(train_ids, desc="Loading train feats+masks")):
    img = imread(os.path.join(train_img_path, tid + ".png"), as_gray=True)
    train_feats[i] = img_to_feat_multi(img, out_sizes=feat_scales)
    train_masks[i] = rle_decode(df_train_by_id.loc[tid, "rle_mask"])

print("train_feats shape:", train_feats.shape, "train_masks shape:", train_masks.shape)

N = len(train_ids)
rng = np.random.RandomState(42)
perm = rng.permutation(N)
val_n = int(0.15 * N)
val_idx = perm[:val_n]
tr_idx = perm[val_n:]

tr_feats = train_feats[tr_idx]
tr_masks = train_masks[tr_idx]
val_feats = train_feats[val_idx]
val_masks = train_masks[val_idx]

print("Train split:", tr_feats.shape[0], "Val split:", val_feats.shape[0])




## === cell 5
def _pairwise_dist2_precomp(A, A2, b):
    b = b.astype(np.float32, copy=False)
    b2 = float(np.dot(b, b))
    Ab = A @ b
    dist2 = A2 - 2.0 * Ab + b2
    np.maximum(dist2, 0.0, out=dist2)
    return dist2.astype(np.float32, copy=False)


def predict_soft_mask_from_cached(nn_idx, nn_dist2, ref_masks, k=5, temperature=1.0):
    k = int(min(k, nn_idx.shape[0]))
    idx = nn_idx[:k]
    d = nn_dist2[:k]
    dmin = float(d.min()) if k > 0 else 0.0
    denom_temp = max(float(temperature), 1e-6)
    w = np.exp(-(d - dmin) / denom_temp).astype(np.float32, copy=False)
    w = w / (w.sum() + 1e-9)

    masks = ref_masks[idx].astype(np.float32, copy=False)  # (k,101,101)
    soft = (masks * w[:, None, None]).sum(axis=0)
    return soft.astype(np.float32, copy=False)


BASE_K = 7
BASE_TEMP = 0.7

K_CANDIDATES = [5, 7, 9, 11]
TEMP_CANDIDATES = [0.4, 0.6, 0.8, 1.0]
thr_grid = np.linspace(0.20, 0.80, 13).astype(np.float32)

MIN_OBJ_CAND = [0, 12, 24]
MIN_HOLE_CAND = [0, 12, 24]

AREA_GATE_CAND = [0, 20, 40, 60, 80, 120]  # pixels

CONF_GATE_CAND = [
    0.0,
    0.20,
    0.25,
    0.30,
    0.35,
]  # min max(soft) required to allow non-empty

best_params = (
    BASE_K,
    BASE_TEMP,
    0.5,
    0,
    0,
    0,
    0.0,
)  # (K, TEMP, thr, min_obj, min_hole, area_gate, conf_gate)
best_score = -1.0

maxK = int(max(K_CANDIDATES))
val_n = val_feats.shape[0]
nn_idx_all = np.empty((val_n, maxK), dtype=np.int32)
nn_dist2_all = np.empty((val_n, maxK), dtype=np.float32)

A = tr_feats.astype(np.float32, copy=False)
A2 = np.einsum("ij,ij->i", A, A).astype(np.float32, copy=False)

for i in range(val_n):
    dist2 = _pairwise_dist2_precomp(A, A2, val_feats[i])
    idx = np.argpartition(dist2, kth=maxK - 1)[:maxK]
    ordk = np.argsort(dist2[idx], kind="mergesort")
    idx = idx[ordk]
    nn_idx_all[i, :] = idx.astype(np.int32, copy=False)
    nn_dist2_all[i, :] = dist2[idx].astype(np.float32, copy=False)

for K_NEIGHBORS in K_CANDIDATES:
    for TEMP in TEMP_CANDIDATES:
        val_soft = np.empty((val_n, 101, 101), dtype=np.float32)
        val_soft_max = np.empty((val_n,), dtype=np.float32)

        for i in range(val_n):
            sm = predict_soft_mask_from_cached(
                nn_idx_all[i],
                nn_dist2_all[i],
                tr_masks,
                k=K_NEIGHBORS,
                temperature=TEMP,
            )
            val_soft[i] = sm
            val_soft_max[i] = float(sm.max())

        raw_pred_by_thr = [
            (val_soft >= float(thr)).astype(np.uint8) for thr in thr_grid
        ]

        conf_zero_idx_by_gate = {}
        for conf_gate in CONF_GATE_CAND:
            if conf_gate > 0.0:
                conf_zero_idx_by_gate[float(conf_gate)] = val_soft_max < float(
                    conf_gate
                )
            else:
                conf_zero_idx_by_gate[0.0] = None

        base_pp_cache = {}
        areas_cache = {}
        area_zero_idx_by_gate = {}  # keyed by (key, area_gate) -> boolean idx or None

        for t_i, thr in enumerate(thr_grid):
            raw_pred = raw_pred_by_thr[t_i]
            for min_obj in MIN_OBJ_CAND:
                for min_hole in MIN_HOLE_CAND:
                    key = (t_i, int(min_obj), int(min_hole))
                    if min_obj == 0 and min_hole == 0:
                        base_pp = raw_pred
                    else:
                        base_pp = np.empty_like(raw_pred)
                        for j in range(val_n):
                            base_pp[j] = postprocess_mask(
                                raw_pred[j], min_obj=min_obj, min_hole=min_hole
                            )
                    base_pp_cache[key] = base_pp
                    areas = base_pp.reshape(val_n, -1).sum(axis=1)
                    areas_cache[key] = areas

                    for area_gate in AREA_GATE_CAND:
                        if area_gate > 0:
                            area_zero_idx_by_gate[(key, int(area_gate))] = areas < int(
                                area_gate
                            )
                        else:
                            area_zero_idx_by_gate[(key, 0)] = None

        for t_i, thr in enumerate(thr_grid):
            for min_obj in MIN_OBJ_CAND:
                for min_hole in MIN_HOLE_CAND:
                    key = (t_i, int(min_obj), int(min_hole))
                    base_pp = base_pp_cache[key]

                    for area_gate in AREA_GATE_CAND:
                        az = (
                            area_zero_idx_by_gate[(key, int(area_gate))]
                            if area_gate > 0
                            else None
                        )
                        az_any = bool(az.any()) if az is not None else False

                        for conf_gate in CONF_GATE_CAND:
                            cz = (
                                conf_zero_idx_by_gate[float(conf_gate)]
                                if conf_gate > 0.0
                                else None
                            )
                            cz_any = bool(cz.any()) if cz is not None else False

                            if not az_any and not cz_any:
                                val_pred = base_pp
                            else:
                                val_pred = base_pp.copy()
                                if az_any:
                                    val_pred[az] = 0
                                if cz_any:
                                    val_pred[cz] = 0

                            s = map_iou_tgs(val_masks, val_pred)
                            if s > best_score:
                                best_score = s
                                best_params = (
                                    int(K_NEIGHBORS),
                                    float(TEMP),
                                    float(thr),
                                    int(min_obj),
                                    int(min_hole),
                                    int(area_gate),
                                    float(conf_gate),
                                )

val_soft_base = np.empty((val_n, 101, 101), dtype=np.float32)
for i in range(val_n):
    val_soft_base[i] = predict_soft_mask_from_cached(
        nn_idx_all[i], nn_dist2_all[i], tr_masks, k=BASE_K, temperature=BASE_TEMP
    )

base_best_thr = 0.5
base_best_score = -1.0
for thr in thr_grid:
    s = map_iou_tgs(val_masks, (val_soft_base >= float(thr)).astype(np.uint8))
    if s > base_best_score:
        base_best_score = s
        base_best_thr = float(thr)

print(
    f"Baseline proxy best (TGS mAP): mAP={base_best_score:.4f} @ thr={base_best_thr:.3f} (K={BASE_K}, TEMP={BASE_TEMP})"
)
print(
    f"Tuned proxy best (TGS mAP):    mAP={best_score:.4f} @ thr={best_params[2]:.3f} "
    f"(K={best_params[0]}, TEMP={best_params[1]}, min_obj={best_params[3]}, min_hole={best_params[4]}, area_gate={best_params[5]}, conf_gate={best_params[6]:.2f})"
)

if best_score + 1e-8 < base_best_score:
    K_NEIGHBORS, TEMP, best_thr, MIN_OBJ, MIN_HOLE, AREA_GATE, CONF_GATE = (
        BASE_K,
        BASE_TEMP,
        base_best_thr,
        0,
        0,
        0,
        0.0,
    )
    print("Tuning did not beat baseline on TGS proxy; falling back to baseline params.")
else:
    K_NEIGHBORS, TEMP, best_thr, MIN_OBJ, MIN_HOLE, AREA_GATE, CONF_GATE = best_params
    print("Using tuned params (TGS-proxy-improving).")




## === cell 6
def predict_soft_masks_batched(
    test_feats, ref_feats, ref_masks, k=5, temperature=1.0, batch=64
):
    ref_feats = ref_feats.astype(np.float32, copy=False)
    ref2 = np.einsum("ij,ij->i", ref_feats, ref_feats).astype(np.float32, copy=False)

    k = int(min(k, ref_feats.shape[0]))
    out = np.empty((test_feats.shape[0], 101, 101), dtype=np.float32)
    out_max = np.empty((test_feats.shape[0],), dtype=np.float32)

    denom_temp = max(float(temperature), 1e-6)

    for i0 in range(0, test_feats.shape[0], int(batch)):
        i1 = min(test_feats.shape[0], i0 + int(batch))
        B = test_feats[i0:i1].astype(np.float32, copy=False)
        B2 = np.einsum("ij,ij->i", B, B).astype(np.float32, copy=False)

        AB = ref_feats @ B.T  # (n_ref, b)
        dist2 = ref2[:, None] - 2.0 * AB + B2[None, :]
        np.maximum(dist2, 0.0, out=dist2)

        idx_part = np.argpartition(dist2, kth=k - 1, axis=0)[:k, :]  # (k, b)
        d_part = np.take_along_axis(dist2, idx_part, axis=0)  # (k, b)
        ordk = np.argsort(d_part, axis=0, kind="mergesort")
        idx = np.take_along_axis(idx_part, ordk, axis=0)  # (k, b)
        d = np.take_along_axis(d_part, ordk, axis=0)  # (k, b)

        dmin = d.min(axis=0, keepdims=True)
        w = np.exp(-(d - dmin) / denom_temp).astype(np.float32, copy=False)  # (k,b)
        w /= w.sum(axis=0, keepdims=True) + 1e-9

        masks_kb = ref_masks[idx.T].transpose(1, 0, 2, 3).astype(np.float32, copy=False)
        soft_b = (masks_kb * w.T[:, :, None, None]).sum(axis=1)  # (b,101,101)

        out[i0:i1] = soft_b.astype(np.float32, copy=False)
        out_max[i0:i1] = (
            soft_b.reshape(soft_b.shape[0], -1).max(axis=1).astype(np.float32)
        )

    return out, out_max


test_ids = df_sub["id"].values
pred_rles = []

ref_feats = tr_feats.astype(np.float32, copy=False)
ref_masks = tr_masks

test_feats = np.empty((len(test_ids), D), dtype=np.float32)

test_imgs = []
for i, tid in enumerate(tqdm(test_ids, desc="Loading test feats+imgs")):
    test_img = imread(os.path.join(test_path, tid + ".png"), as_gray=True)
    test_imgs.append(test_img)
    test_feats[i] = img_to_feat_multi(test_img, out_sizes=feat_scales)

soft_all, soft_max_all = predict_soft_masks_batched(
    test_feats, ref_feats, ref_masks, k=K_NEIGHBORS, temperature=TEMP, batch=64
)

for i, tid in enumerate(tqdm(test_ids, desc="Postprocess+CRF+RLE")):
    soft = soft_all[i]
    mask_pred = (soft >= float(best_thr)).astype(np.uint8)

    if MIN_OBJ > 0 or MIN_HOLE > 0:
        mask_pred = postprocess_mask(mask_pred, min_obj=MIN_OBJ, min_hole=MIN_HOLE)

    if AREA_GATE > 0:
        if int(mask_pred.sum()) < int(AREA_GATE):
            mask_pred[:] = 0

    if CONF_GATE > 0.0:
        if float(soft_max_all[i]) < float(CONF_GATE):
            mask_pred[:] = 0

    test_img = test_imgs[i]
    mask_pred = crf(test_img, mask_pred)

    pred_rles.append(rle_encode(mask_pred))

df_sub["rle_mask"] = pred_rles
df_sub.head()



## === cell 7
nImgs = 2
i0 = np.random.randint(0, len(df_sub))
shown = 0

plt.figure(figsize=(10, 6))
plt.subplots_adjust(wspace=0.2, hspace=0.2)
i = i0
while shown < nImgs and i < len(df_sub):
    decoded_mask = rle_decode(df_sub.loc[i, "rle_mask"])
    orig_img = imread(
        os.path.join(test_path, df_sub.loc[i, "id"] + ".png"), as_gray=True
    )
    crf_output = crf(orig_img, decoded_mask)

    plt.subplot(nImgs, 3, shown * 3 + 1)
    plt.imshow(orig_img, cmap="gray")
    plt.title("Original")

    plt.subplot(nImgs, 3, shown * 3 + 2)
    plt.imshow(decoded_mask, cmap="gray")
    plt.title("Pred mask (decoded)")

    plt.subplot(nImgs, 3, shown * 3 + 3)
    plt.imshow(crf_output, cmap="gray")
    plt.title("After CRF (or passthrough)")

    shown += 1
    i += 1

plt.show()



## === cell 8
out_path = "crf_correction_unet_bn_diceloss.csv"
df_sub[["id", "rle_mask"]].to_csv(out_path, index=False)

print("Wrote:", out_path, "rows:", len(df_sub))
print(df_sub.head())
