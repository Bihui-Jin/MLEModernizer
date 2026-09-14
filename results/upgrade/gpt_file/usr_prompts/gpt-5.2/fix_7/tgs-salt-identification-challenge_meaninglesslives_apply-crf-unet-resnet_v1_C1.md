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

0.8018520121159672

# 6. Current score

0.3788

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on `pydensecrf` (it’s not installed in this environment) by making CRF post-processing optional: if `pydensecrf` is missing, the code safely skip CRF and still generate a valid `.csv` submission. I also remove notebook-only syntax (`%matplotlib inline`) and fix the undefined variables caused by the import cell failing, so later cells can run. Since the referenced baseline submission path (`../input/u-net-bn-aug-strat-dice/submission.csv`) does not exist here, I instead load the provided `sample_submission.csv` (valid format) so the pipeline completes end-to-end. The output always be written as `crf_correction_unet_bn_diceloss.csv` with the required `id,rle_mask` columns.'
- What this solution (achieved 0.0) has done: 'Your current score is 0.0 because the code is not generating any meaningful masks: it takes `sample_submission.csv` (which is essentially empty predictions) and optionally “CRF-corrects” those empty masks, so the output stays empty and scores near zero. To move toward the target score with minimal changes and without changing your RLE/CRF utilities, I replace the prediction source with a legitimate baseline: build masks from the provided `train.csv` RLEs by matching each test image to the most similar training image via a fast, downsampled pixel MSE, then optionally apply your CRF step and encode. This keeps the “post-process + write submission” core flow, but swaps the non-predictive input with an actual (simple) prediction mechanism so the score increases substantially from 0.0. I also keep everything deterministic and within the 600s budget by downsampling and using vectorized computations.'
- What this solution (achieved 0.0) has done: 'I fix the feature-dimension mismatch that causes the broadcast errors by making `img_to_feat` produce a deterministic, fixed-size (26×26) downsample for both train and test, matching the preallocated `train_feats` shape. I also make the train mask lookup robust by indexing `df_train` by `id` (instead of relying on positional `.loc[i]`), preventing accidental misalignment if the CSV order changes. These changes keep your core “nearest-neighbor in downsampled pixel space + optional CRF + RLE” logic identical, but make it run end-to-end and generate a meaningful submission (instead of crashing). The submission file name/path and required columns remain unchanged.'
- What this solution (achieved 0.253) has done: 'I fix the feature-size mismatch that causes the broadcast errors by making `img_to_feat` always return a fixed-length (26×26) vector for both train and test, regardless of image channel shape. I also make image loading explicitly grayscale and robust to occasional RGB reads so the downsample is consistent and deterministic. These changes are score-positive (your current 0.0 is from a crash/no valid submission) while preserving your core nearest-neighbor-on-downsampled-pixels + optional CRF + RLE pipeline. The script then run end-to-end and write a valid `crf_correction_unet_bn_diceloss.csv` submission.'
- What this solution (achieved 0.3702) has done: 'I keep your nearest-neighbor-on-downsampled-pixels pipeline intact, but make two minimal, metric-relevant upgrades that typically lift this baseline toward your target: (1) use a distance-weighted average of the top‑K nearest training masks (soft mask) instead of a single nearest neighbor, and (2) choose a binarization threshold on a small deterministic validation split by directly maximizing the competition’s mean AP over IoU thresholds. These changes don’t alter the overall approach (still downsample → compute distances → retrieve train masks → (optional) CRF → RLE), but they greatly reduce brittleness from one-neighbor noise and align the final thresholding with the evaluation metric. I also keep everything deterministic and within time by precomputing train–val features and only searching a small fixed grid of thresholds. The script still write `crf_correction_unet_bn_diceloss.csv` with `id,rle_mask`.'
- What this solution (achieved 0.3788) has done: 'I keep your downsampled-pixel nearest-neighbor + soft top‑K mask averaging + optional CRF + RLE pipeline intact, but make two metric-aligned adjustments that should lift score toward your target: (1) tune the soft-kNN hyperparameters (K and temperature) together with the binarization threshold on the same deterministic validation split, because these strongly control mask quality and calibration, and (2) add a tiny post-processing step that removes very small predicted components and fills very small holes, which typically improves IoU-sweep mAP by reducing noisy speckles (without changing the modeling approach). The search is kept small and deterministic to stay within the 600s budget, and if the tuned parameters don’t beat the current baseline on the validation proxy, the code falls back to your original (K=7, TEMP=0.7) to avoid regressions. Output format and the submission filename remain unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.io import imread
from skimage.color import gray2rgb
from skimage.morphology import remove_small_objects, remove_small_holes

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
def _ensure_gray(img):
    if img.ndim == 3:
        img = img[:, :, 0]
    return img.astype(np.float32) / 255.0


def img_to_feat(img_101x101, out_size=26):
    img = _ensure_gray(img_101x101)
    idx = np.linspace(0, img.shape[0] - 1, out_size).round().astype(np.int32)
    small = img[np.ix_(idx, idx)]
    feat = small.reshape(-1)
    if feat.shape[0] != out_size * out_size:
        raise ValueError(
            f"Unexpected feature length {feat.shape[0]} for out_size={out_size}"
        )
    return feat


def iou_metric_batch(y_true, y_pred, eps=1e-9):
    y_true = (y_true > 0).astype(np.uint8)
    y_pred = (y_pred > 0).astype(np.uint8)
    intersection = np.sum(y_true * y_pred, axis=(1, 2)).astype(np.float32)
    union = np.sum((y_true + y_pred) > 0, axis=(1, 2)).astype(np.float32)
    return (intersection + eps) / (union + eps)


def map_iou(y_true, y_pred):
    thresholds = np.arange(0.5, 1.0, 0.05)
    ious = iou_metric_batch(y_true, y_pred)
    precisions = []
    y_pred_has = y_pred.reshape(y_pred.shape[0], -1).sum(axis=1) > 0
    y_true_has = y_true.reshape(y_true.shape[0], -1).sum(axis=1) > 0
    for t in thresholds:
        tp = (ious > t).astype(np.float32)
        fp = (y_pred_has & (ious <= t)).astype(np.float32)
        fn = (y_true_has & (ious <= t)).astype(np.float32)
        precisions.append(tp / (tp + fp + fn + 1e-9))
    return float(np.mean(np.stack(precisions, axis=1)))


def postprocess_mask(mask01, min_obj=12, min_hole=12):
    m = (mask01 > 0).astype(bool)
    if min_obj > 0:
        m = remove_small_objects(m, min_size=int(min_obj), connectivity=1)
    if min_hole > 0:
        m = remove_small_holes(m, area_threshold=int(min_hole), connectivity=1)
    return m.astype(np.uint8)


out_size = 26  # keep same core logic: downsampled pixels
D = out_size * out_size

df_train_by_id = df_train.set_index("id")

train_ids = df_train["id"].values
train_feats = np.empty((len(train_ids), D), dtype=np.float32)
train_masks = []

for i, tid in enumerate(tqdm(train_ids, desc="Loading train feats+masks")):
    img = imread(os.path.join(train_img_path, tid + ".png"), as_gray=True)
    train_feats[i] = img_to_feat(img, out_size=out_size)
    train_masks.append(rle_decode(df_train_by_id.loc[tid, "rle_mask"]))

train_masks = np.stack(train_masks, axis=0).astype(np.uint8)
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
def predict_soft_mask_from_feats(feat, ref_feats, ref_masks, k=5, temperature=1.0):
    dist2 = np.sum((ref_feats - feat[None, :]) ** 2, axis=1).astype(np.float32)
    k = int(min(k, dist2.shape[0]))
    nn = np.argpartition(dist2, kth=k - 1)[:k]
    d = dist2[nn]
    w = np.exp(-(d - d.min()) / max(float(temperature), 1e-6)).astype(np.float32)
    w = w / (w.sum() + 1e-9)
    soft = np.tensordot(w, ref_masks[nn].astype(np.float32), axes=(0, 0))
    return soft  # (101,101) float32


BASE_K = 7
BASE_TEMP = 0.7

K_CANDIDATES = [5, 7, 9]
TEMP_CANDIDATES = [0.5, 0.7, 1.0]
thr_grid = np.linspace(0.25, 0.75, 11).astype(np.float32)

MIN_OBJ_CAND = [0, 12]
MIN_HOLE_CAND = [0, 12]

best_params = (BASE_K, BASE_TEMP, 0.5, 0, 0)
best_score = -1.0

for K_NEIGHBORS in K_CANDIDATES:
    for TEMP in TEMP_CANDIDATES:
        val_soft = np.empty((val_feats.shape[0], 101, 101), dtype=np.float32)
        for i in range(val_feats.shape[0]):
            val_soft[i] = predict_soft_mask_from_feats(
                val_feats[i], tr_feats, tr_masks, k=K_NEIGHBORS, temperature=TEMP
            )

        for thr in thr_grid:
            raw_pred = (val_soft >= float(thr)).astype(np.uint8)
            for min_obj in MIN_OBJ_CAND:
                for min_hole in MIN_HOLE_CAND:
                    if min_obj == 0 and min_hole == 0:
                        val_pred = raw_pred
                    else:
                        val_pred = np.empty_like(raw_pred)
                        for j in range(raw_pred.shape[0]):
                            val_pred[j] = postprocess_mask(
                                raw_pred[j], min_obj=min_obj, min_hole=min_hole
                            )
                    s = map_iou(val_masks, val_pred)
                    if s > best_score:
                        best_score = s
                        best_params = (
                            K_NEIGHBORS,
                            TEMP,
                            float(thr),
                            int(min_obj),
                            int(min_hole),
                        )

val_soft_base = np.empty((val_feats.shape[0], 101, 101), dtype=np.float32)
for i in range(val_feats.shape[0]):
    val_soft_base[i] = predict_soft_mask_from_feats(
        val_feats[i], tr_feats, tr_masks, k=BASE_K, temperature=BASE_TEMP
    )
base_best_thr = 0.5
base_best_score = -1.0
for thr in thr_grid:
    s = map_iou(val_masks, (val_soft_base >= float(thr)).astype(np.uint8))
    if s > base_best_score:
        base_best_score = s
        base_best_thr = float(thr)

print(
    f"Baseline proxy best: mAP={base_best_score:.4f} @ thr={base_best_thr:.3f} (K={BASE_K}, TEMP={BASE_TEMP})"
)
print(
    f"Tuned proxy best:    mAP={best_score:.4f} @ thr={best_params[2]:.3f} (K={best_params[0]}, TEMP={best_params[1]}, min_obj={best_params[3]}, min_hole={best_params[4]})"
)

if best_score + 1e-8 < base_best_score:
    K_NEIGHBORS, TEMP, best_thr, MIN_OBJ, MIN_HOLE = (
        BASE_K,
        BASE_TEMP,
        base_best_thr,
        0,
        0,
    )
    print("Tuning did not beat baseline on proxy; falling back to baseline params.")
else:
    K_NEIGHBORS, TEMP, best_thr, MIN_OBJ, MIN_HOLE = best_params
    print("Using tuned params (proxy-improving).")



## === cell 6
test_ids = df_sub["id"].values
pred_rles = []

ref_feats = train_feats
ref_masks = train_masks

for tid in tqdm(test_ids, desc="Predicting test masks"):
    test_img = imread(os.path.join(test_path, tid + ".png"), as_gray=True)
    tf = img_to_feat(test_img, out_size=out_size)

    soft = predict_soft_mask_from_feats(
        tf, ref_feats, ref_masks, k=K_NEIGHBORS, temperature=TEMP
    )
    mask_pred = (soft >= float(best_thr)).astype(np.uint8)

    if MIN_OBJ > 0 or MIN_HOLE > 0:
        mask_pred = postprocess_mask(mask_pred, min_obj=MIN_OBJ, min_hole=MIN_HOLE)

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
