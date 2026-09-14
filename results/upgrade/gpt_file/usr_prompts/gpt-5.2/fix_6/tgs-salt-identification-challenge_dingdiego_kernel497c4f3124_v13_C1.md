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

0.7726698398961486

# 6. Current score

0.1958

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on `pydensecrf` (not installed in this Kaggle environment) by providing a safe fallback that keeps the pipeline running and still produces a valid submission CSV. I also remove notebook-only magic (`%matplotlib inline`) and fix the `pd/plt/tqdm` “not defined” errors by ensuring imports execute and by making the baseline submission path robust (use `sample_submission.csv` if the referenced `baseline-v4/submission.csv` doesn’t exist). Finally, I fix the NaN/empty-mask checks so we don’t try to decode invalid RLE strings, and I always write a `.csv` with the required `id,rle_mask` columns.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is coming from an invalid submission encoding: this competition’s RLE is **column-major (Fortran order)**, but your `rle_encode`/`rle_decode` use row-major, which makes masks decode incorrectly and tanks IoU/AP. I minimally fix both functions to use Fortran order and add a safety guard for truly empty masks to emit an empty string, matching common TGS conventions. This preserves your core approach (post-processing an existing submission with optional CRF) while making the produced CSV semantically correct for the metric. Everything else (paths, loop, CRF fallback) stays the same so it runs end-to-end and writes `crf_correction.csv`.'
- What this solution (achieved 0.0996) has done: 'Your 0.0 score is consistent with producing a valid-looking CSV that is nevertheless *semantically wrong for this competition*: the sample submission contains placeholder masks (“1 1”), and your script only modifies rows where `rle_mask` is non-empty—so you end up submitting mostly (or entirely) those placeholders, which score ~0. I minimally switch the input source from `sample_submission.csv` to a real baseline if present, and if not present I instead run a simple, deterministic segmentation directly from the test images (Otsu thresholding) so every row gets a plausible mask. This keeps your core flow (read ids → decode/produce mask → optional CRF → RLE encode → write CSV) and preserves the CRF fallback behavior, but avoids the guaranteed-bad placeholder submission. I also make the test-image directory robust by checking alternate provided paths so we don’t silently skip all images.'
- What this solution (achieved 0.1007) has done: 'Your current score is far below the target, so the smallest safe way to improve is to keep your exact pipeline (Otsu → optional CRF fallback → RLE) but make the initial mask less noisy and more “salt-shaped” using lightweight, deterministic morphology on the Otsu mask. This doesn’t change the model/approach (still pure image-threshold segmentation), but it reduces scattered pixels and fills small holes, which typically increases IoU and AP for this competition. I also ensure every row is processed even if an image is missing by explicitly writing an empty mask for those ids (prevents stale placeholders from remaining). Finally, I keep your Fortran-order RLE and output format unchanged.'
- What this solution (achieved 0.1958) has done: 'Your current score (0.1007) is far below the target (0.7727), so we should improve the *same* Otsu→(light morphology)→CRF-fallback→RLE pipeline rather than change approaches. The most score-critical minimal fix here is to stop the “choose the bigger side of Otsu” heuristic, which often predicts the background as salt; instead we consistently take the darker-than-threshold side (salt is typically darker in this dataset) and add a very light border-clear step to remove frequent frame artifacts. To keep changes minimal and deterministic, I’m only adjusting the mask selection and adding `clear_border`, leaving CRF fallback, RLE (Fortran order), paths, and submission writing intact. This should move IoU/AP substantially upward toward your target without changing the overall logic or adding any new dependencies beyond scikit-image you already use.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.color import gray2rgb
from skimage.filters import threshold_otsu

from skimage.morphology import remove_small_objects, remove_small_holes, closing, disk
from skimage.segmentation import (
    clear_border,
)  # score-relevant: remove common border/frame false positives

from tqdm import tqdm

_HAS_DCRF = False
try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_labels

    _HAS_DCRF = True
except Exception:
    _HAS_DCRF = False

BASE_INPUT = "../input/tgs-salt-identification-challenge"

_TEST_DIR_CANDIDATES = [
    os.path.join(BASE_INPUT, "test", "images"),
    "../input/test/images",
    "../input/tgs-salt-identification-challenge/test/images",
    "../input/kaggle/input/tgs-salt-identification-challenge/test/images",
    "../kaggle/data/tgs-salt-identification-challenge/test/images",
]
TEST_IMG_DIR = next(
    (p for p in _TEST_DIR_CANDIDATES if os.path.exists(p)), _TEST_DIR_CANDIDATES[0]
)




## === cell 1
def rle_decode(rle_mask, shape=(101, 101)):
    """
    Kaggle TGS Salt uses column-major (Fortran) order for RLE.
    Fixing decode order is directly score-critical (otherwise masks are scrambled -> near-zero IoU/AP).
    """
    if rle_mask is None:
        return np.zeros(shape, dtype=np.uint8)
    s = str(rle_mask).strip()
    if s == "" or s.lower() == "nan":
        return np.zeros(shape, dtype=np.uint8)

    s = s.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape, order="F")


def rle_encode(im):
    """
    Kaggle TGS Salt uses column-major (Fortran) order for RLE.
    Fixing encode order is directly score-critical (otherwise submitted masks are wrong).
    """
    im = (im > 0).astype(np.uint8)

    if im.sum() == 0:
        return ""

    pixels = im.reshape(-1, order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def _is_valid_rle(x):
    if x is None:
        return False
    if isinstance(x, float) and np.isnan(x):
        return False
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return False
    return True


def _is_placeholder_rle(x):
    s = str(x).strip()
    return s == "1 1"




## === cell 2
baseline_path = "../input/baseline-v4/submission.csv"
sample_path = os.path.join(BASE_INPUT, "sample_submission.csv")

if os.path.exists(baseline_path):
    df = pd.read_csv(baseline_path)
    _INPUT_SOURCE = "baseline-v4"
else:
    df = pd.read_csv(sample_path)
    _INPUT_SOURCE = "sample_submission"

if "id" not in df.columns or "rle_mask" not in df.columns:
    raise ValueError(
        f"Submission must have columns ['id','rle_mask'], got: {df.columns.tolist()}"
    )

df["id"] = df["id"].astype(str)



## === cell 3
"""
Function which returns the labelled image after applying CRF.
If pydensecrf is unavailable, return the input mask unchanged (keeps pipeline running).
"""


def crf(original_image, mask_img):
    if not _HAS_DCRF:
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
    MAP = np.argmax(Q, axis=0)
    return MAP.reshape((original_image.shape[0], original_image.shape[1])).astype(
        np.uint8
    )




## === cell 4
try:
    n_show = 6
    shown = 0
    plt.figure(figsize=(18, 4))
    for idx in range(len(df)):
        if shown >= n_show:
            break
        rle = df.loc[idx, "rle_mask"]
        if _is_valid_rle(rle) and not _is_placeholder_rle(rle):
            m = rle_decode(rle)
            plt.subplot(1, n_show, shown + 1)
            plt.imshow(m, cmap="gray")
            plt.title(df.loc[idx, "id"])
            plt.axis("off")
            shown += 1
    plt.close()
except Exception:
    pass



## === cell 5
"""
Applying CRF on the predicted mask.

Score-relevant minimal change (keeps same core approach):
- Otsu mask selection: use img < t consistently (salt tends to be darker). The prior "pick larger side"
  often predicts background as salt -> very low IoU/AP.
- Add clear_border on the binary mask to remove common edge/frame artifacts (reduces false positives).

Everything else (morphology cleanup, CRF fallback, Fortran RLE, paths) stays the same.
"""
missing_imgs = 0
processed = 0
generated_from_image = 0
used_existing_rle = 0

_ignore_placeholders_globally = _INPUT_SOURCE == "sample_submission"

_MIN_OBJ = 20  # remove components smaller than this
_MIN_HOLE = 20  # fill holes smaller than this
_CLOSING_RADIUS = 1  # very light closing

selem = disk(_CLOSING_RADIUS)

for i in tqdm(range(df.shape[0])):
    img_id = df.loc[i, "id"]
    img_path = os.path.join(TEST_IMG_DIR, f"{img_id}.png")
    if not os.path.exists(img_path):
        df.loc[i, "rle_mask"] = ""
        missing_imgs += 1
        continue

    orig_img = imread(img_path)

    rle = df.loc[i, "rle_mask"]
    can_use_rle = (
        _is_valid_rle(rle)
        and (not _is_placeholder_rle(rle))
        and (not _ignore_placeholders_globally)
    )

    if can_use_rle:
        decoded_mask = rle_decode(rle)
        used_existing_rle += 1
    else:
        img = orig_img.astype(np.float32)
        if img.ndim == 3:
            img = img[..., 0]
        img = img / 255.0 if img.max() > 1.5 else img

        try:
            t = threshold_otsu(img)
        except Exception:
            t = float(np.mean(img))

        decoded_mask = (img < t).astype(bool)

        try:
            decoded_mask = clear_border(decoded_mask)

            decoded_mask = remove_small_objects(decoded_mask, min_size=_MIN_OBJ)
            decoded_mask = remove_small_holes(decoded_mask, area_threshold=_MIN_HOLE)
            decoded_mask = closing(decoded_mask, selem)
        except Exception:
            pass

        decoded_mask = decoded_mask.astype(np.uint8)
        generated_from_image += 1

    crf_output = crf(orig_img, decoded_mask)
    df.loc[i, "rle_mask"] = rle_encode(crf_output)
    processed += 1



## === cell 6
out_path = "crf_correction.csv"
df[["id", "rle_mask"]].to_csv(out_path, index=False)

print(f"Wrote submission: {out_path}")
print(
    f"Rows: {len(df)}, processed: {processed}, generated_from_image: {generated_from_image}, "
    f"used_existing_rle: {used_existing_rle}, missing_images: {missing_imgs}, "
    f"has_dcrf: {_HAS_DCRF}, input_source: {_INPUT_SOURCE}, test_img_dir: {TEST_IMG_DIR}"
)
