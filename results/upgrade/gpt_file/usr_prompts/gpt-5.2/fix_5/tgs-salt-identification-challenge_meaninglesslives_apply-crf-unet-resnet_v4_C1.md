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

0.0462

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I make the notebook run in this Kaggle environment by removing the hard dependency on `pydensecrf` (it’s not installed) and adding a safe fallback so the pipeline still produces a submission CSV. I also fix the broken/irrelevant input CSV path by loading the provided `sample_submission.csv` (or another existing submission if you point to one), and correct the `NaN` checks so empty masks are handled properly. The CRF step be skipped automatically when `pydensecrf` is unavailable, keeping the rest of the logic (RLE decode/encode and per-image processing loop) intact and ensuring `crf_correction_unet_bn_diceloss.csv` is always written. This yield a valid submission file end-to-end; score changes are not meaningfully optimizable here without an actual base prediction file, so the focus is on correctness and producing a valid CSV.'
- What this solution (achieved 0.0598) has done: 'Your current 0.0 score is consistent with submitting essentially the `sample_submission.csv`, which is not a real prediction and score near-zero; the minimal way to move toward your target is to generate a legitimate mask per test image instead of relying on the sample file’s dummy RLEs. I keep your pipeline structure (load IDs → per-image loop → optional CRF → RLE encode → write CSV) but switch the input `rle_mask` source to a simple, deterministic image-based baseline (Otsu threshold on normalized grayscale), which is a legitimate segmentation heuristic and should lift the score well above 0.0. I also fix the RLE encode/decode to use the competition’s expected column-major (Fortran) order to avoid format-related score loss. The CRF step remain optional and run only when `pydensecrf` exists; otherwise it safely skip as before, and the script still write `crf_correction_unet_bn_diceloss.csv`.'
- What this solution (achieved 0.0612) has done: 'Your current score (0.0598) is far below the target (0.8089), so we should improve legitimate mask quality while keeping your existing pipeline (per-image heuristic mask → optional CRF → RLE encode → write CSV) intact. The biggest low-risk gain here is fixing the RLE encode/decode to the competition’s required column-major (Fortran) order (your current encode uses a transposed C-flatten that can corrupt masks). Then, to raise baseline segmentation quality without changing the “simple image-threshold baseline” core idea, we keep Otsu thresholding but add small, deterministic morphological cleanup (remove tiny noise + fill tiny holes) and a simple border-cropping prior that reduces common edge false-positives in this dataset. These changes should move the score substantially upward toward the target while staying lightweight and within time limits, and still produce the required `crf_correction_unet_bn_diceloss.csv`.'
- What this solution (achieved 0.0462) has done: 'Your score is far below the target (0.0612 vs 0.8089; higher-is-better), so we need a legitimate uplift while keeping your existing “Otsu threshold → morphology cleanup → optional CRF → RLE” pipeline unchanged in spirit. The biggest low-risk issue is that raw Otsu on these images often needs a simple per-image contrast normalization; adding deterministic percentile stretching typically improves separability without changing the core approach. Next, your border-zeroing is very aggressive (kills true salt that touches borders); we replace it with a much lighter “corner suppression” prior that only clears a few pixels in the four corners. Finally, we do a tiny bit of threshold calibration (a small bias on Otsu) and slightly stronger morphology parameters; these are minimal tweaks that usually move mAP up materially while keeping runtime well under the limit and still writing the same submission CSV.'

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

assert os.path.isdir(TEST_IMG_DIR), f"Missing test image dir: {TEST_IMG_DIR}"
assert os.path.isfile(SAMPLE_SUB_PATH), f"Missing sample submission: {SAMPLE_SUB_PATH}"




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
def baseline_mask_from_image(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = img[..., 0]
    x = img.astype(np.float32)
    if x.max() > 1.0:
        x /= 255.0

    p2, p98 = np.percentile(x, [2, 98])
    if p98 > p2 + 1e-6:
        x = np.clip((x - p2) / (p98 - p2), 0.0, 1.0)

    thr = threshold_otsu(x)
    thr = float(np.clip(thr - 0.02, 0.0, 1.0))
    mask = x < thr

    se = disk(2)
    mask = binary_opening(mask, se)
    mask = binary_closing(mask, se)

    mask = remove_small_objects(mask, min_size=60)
    mask = remove_small_holes(mask, area_threshold=60)

    mask = mask.astype(np.uint8)

    k = 3
    mask[:k, :k] = 0
    mask[:k, -k:] = 0
    mask[-k:, :k] = 0
    mask[-k:, -k:] = 0

    return mask




## === cell 5
for i in tqdm(range(df.shape[0]), desc="Predict + (optional) CRF postprocess"):
    img_id = df.loc[i, "id"]
    img_path = os.path.join(TEST_IMG_DIR, f"{img_id}.png")
    if not os.path.isfile(img_path):
        df.loc[i, "rle_mask"] = ""
        continue

    orig_img = imread(img_path)
    pred_mask = baseline_mask_from_image(orig_img)

    crf_output = crf(orig_img, pred_mask)

    df.loc[i, "rle_mask"] = rle_encode(crf_output)



## === cell 6
df["rle_mask"] = df["rle_mask"].fillna("").astype(str)

out_path = "crf_correction_unet_bn_diceloss.csv"
df.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(df))
print(df.head())
