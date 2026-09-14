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

0.8218693206404148

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on `pydensecrf` (not installed in this Kaggle environment) by adding a safe fallback that keeps the pipeline running end-to-end while preserving the same “CRF post-process” call sites. I also fix notebook-only syntax (`%matplotlib inline`, `ls`) and missing imports that caused `NameError`s. Because the referenced `../input/k-fold-merger/submission.csv` dataset is not available, I instead read `sample_submission.csv` and generate a valid submission by applying the same loop logic over all test ids (leaving empty masks empty). Finally, I ensure the output is written as a proper `.csv` with the required `id,rle_mask` columns.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

from skimage.io import imread
from skimage.color import gray2rgb

import matplotlib.pyplot as plt
from tqdm import tqdm

_HAS_DCRF = False
try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_labels

    _HAS_DCRF = True
except ModuleNotFoundError:
    _HAS_DCRF = False

np.random.seed(42)




## === cell 1
def rle_decode(rle_mask):
    """
    rle_mask: run-length as string formatted (start length)
    Returns numpy array shape (101,101), 1 - mask, 0 - background
    """
    if rle_mask is None:
        return np.zeros((101, 101), dtype=np.uint8)
    if isinstance(rle_mask, float) and np.isnan(rle_mask):
        return np.zeros((101, 101), dtype=np.uint8)

    rle_mask = str(rle_mask).strip()
    if rle_mask == "" or rle_mask.lower() == "nan":
        return np.zeros((101, 101), dtype=np.uint8)

    s = rle_mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(101 * 101, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(101, 101)




## === cell 2
BASE_INPUT = "../input/tgs-salt-identification-challenge"
test_path = os.path.join(BASE_INPUT, "test", "images")

sample_path = os.path.join(BASE_INPUT, "sample_submission.csv")
if not os.path.exists(sample_path):
    sample_path = "../input/sample_submission.csv"

df = pd.read_csv(sample_path)
assert "id" in df.columns and "rle_mask" in df.columns, "Unexpected submission format."

df["rle_mask"] = df["rle_mask"].astype(object)




## === cell 3
def crf(original_image, mask_img):
    """
    Return labelled image after applying CRF.
    If pydensecrf is not available, return the input mask unchanged (score-neutral vs not running).
    """
    if len(mask_img.shape) < 3:
        mask_img_rgb = gray2rgb(mask_img)
    else:
        mask_img_rgb = mask_img

    if not _HAS_DCRF:
        if len(mask_img_rgb.shape) == 3:
            out = (mask_img_rgb[:, :, 0] > 0).astype(np.uint8)
        else:
            out = (mask_img_rgb > 0).astype(np.uint8)
        return out

    annotated_label = (
        mask_img_rgb[:, :, 0].astype(np.uint32)
        + (mask_img_rgb[:, :, 1].astype(np.uint32) << 8)
        + (mask_img_rgb[:, :, 2].astype(np.uint32) << 16)
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




## === cell 4
def rle_encode(im):
    """
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted.

    Note: Competition expects 1-indexed runs in column-major order (top->bottom, left->right).
    Flattening in Fortran order satisfies that.
    """
    im = (im > 0).astype(np.uint8)
    pixels = im.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 5
try:
    nImgs = 3
    start_i = np.random.randint(0, len(df))
    shown = 0
    plt.figure(figsize=(12, 10))
    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    i = start_i
    while shown < nImgs and i < len(df):
        img_id = df.loc[i, "id"]
        img_file = os.path.join(test_path, f"{img_id}.png")
        if os.path.exists(img_file):
            decoded_mask = rle_decode(df.loc[i, "rle_mask"])
            orig_img = imread(img_file)
            crf_output = crf(orig_img, decoded_mask)

            r = shown + 1
            plt.subplot(nImgs, 3, 3 * shown + 1)
            plt.imshow(orig_img, cmap="gray")
            plt.title(f"Original: {img_id}")
            plt.axis("off")

            plt.subplot(nImgs, 3, 3 * shown + 2)
            plt.imshow(decoded_mask, cmap="gray")
            plt.title("Input mask")
            plt.axis("off")

            plt.subplot(nImgs, 3, 3 * shown + 3)
            plt.imshow(crf_output, cmap="gray")
            plt.title("After CRF (or fallback)")
            plt.axis("off")

            shown += 1
        i += 1
    plt.close()
except Exception:
    pass



## === cell 6
"""
Applying CRF on the predicted mask.

We don't have an upstream model submission available in this environment,
so we start from sample_submission and apply the same post-process loop.
This guarantees a valid submission file is produced end-to-end.
"""
for i in tqdm(range(df.shape[0]), desc="Post-processing masks"):
    rle = df.loc[i, "rle_mask"]
    if (
        rle is None
        or (isinstance(rle, float) and np.isnan(rle))
        or str(rle).strip() == ""
    ):
        df.loc[i, "rle_mask"] = ""
        continue

    decoded_mask = rle_decode(rle)
    img_id = df.loc[i, "id"]
    img_file = os.path.join(test_path, f"{img_id}.png")

    if not os.path.exists(img_file):
        continue

    orig_img = imread(img_file)
    crf_output = crf(orig_img, decoded_mask)
    df.loc[i, "rle_mask"] = rle_encode(crf_output)



## === cell 7
out_path = "crf_correction.csv"
df.to_csv(out_path, index=False)

assert out_path.endswith(".csv")
assert df.shape[0] > 0
assert list(df.columns) == ["id", "rle_mask"]
print(f"Wrote submission: {out_path} (rows={len(df)})")
