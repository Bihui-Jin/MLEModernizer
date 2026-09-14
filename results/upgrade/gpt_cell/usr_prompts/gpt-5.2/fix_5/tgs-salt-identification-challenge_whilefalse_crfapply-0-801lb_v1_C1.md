# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.7

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import numpy as np

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral

    _HAS_DENSECRF = True
except ModuleNotFoundError:
    dcrf = None
    unary_from_labels = None
    create_pairwise_bilateral = None
    _HAS_DENSECRF = False

from skimage.io import imread
from skimage.color import gray2rgb
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os


def rle_decode(mask_rle, shape=(101, 101)):
    """
    mask_rle: run-length as string formatted (start length), 1-indexed, column-major.
    Returns numpy array, 1 - mask, 0 - background (shape 101x101)
    """
    if mask_rle is None:
        return np.zeros(shape, dtype=np.uint8)
    if not isinstance(mask_rle, str):
        return np.zeros(shape, dtype=np.uint8)
    mask_rle = mask_rle.strip()
    if mask_rle == "":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape, order="F")


def rle_encode(im):
    """
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted, column-major, 1-indexed.
    """
    im = (im > 0).astype(np.uint8)
    pixels = im.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def crf(original_image, mask_img):
    if not _HAS_DENSECRF:
        if len(mask_img.shape) >= 3:
            mask2d = mask_img[:, :, 0]
        else:
            mask2d = mask_img
        return (mask2d > 0).astype(np.uint8)

    if len(mask_img.shape) < 3:
        mask_img = gray2rgb(mask_img)

    annotated_label = (
        mask_img[:, :, 0] + (mask_img[:, :, 1] << 8) + (mask_img[:, :, 2] << 16)
    )
    colors, labels = np.unique(annotated_label, return_inverse=True)
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
    return MAP.reshape((original_image.shape[0], original_image.shape[1]))


candidate_paths = [
    "../input/candsub/blend_ez.csv",  # original intended input
    "../input/tgs-salt-identification-challenge/blend_ez.csv",
    "../input/blend_ez.csv",
]
df = None
used_path = None
for p in candidate_paths:
    if os.path.exists(p):
        df = pd.read_csv(p)
        used_path = p
        break

if df is None:
    raise FileNotFoundError(
        "No prediction submission found. Expected one of:\n"
        + "\n".join(candidate_paths)
        + "\n\n"
        "Refusing to run with sample_submission.csv because it produces near-empty masks and a ~0 score."
    )

required_cols = {"id", "rle_mask"}
if not required_cols.issubset(df.columns):
    raise ValueError(
        f"Submission input must have columns {required_cols}, got {set(df.columns)}"
    )

print(f"Loaded candidate submission: {used_path} (rows={len(df)})")

i = 0
j = 0
plt.figure(figsize=(30, 15))
plt.subplots_adjust(bottom=0.2, top=0.8, hspace=0.2)

max_scan = min(df.shape[0], 200)
while i < max_scan and j < 6:
    rle = df.loc[i, "rle_mask"]
    if isinstance(rle, str) and len(rle.strip()) > 0:
        decoded_mask = rle_decode(rle)
        plt.subplot(1, 6, j + 1)
        plt.imshow(decoded_mask)
        plt.title("ID: " + str(df.loc[i, "id"]))
        j += 1
    i += 1

test_img_dir = "../input/tgs-salt-identification-challenge/test/images/"

for i in tqdm(range(df.shape[0])):
    rle = df.loc[i, "rle_mask"]

    if isinstance(rle, str) and len(rle.strip()) > 0:
        img_path = os.path.join(test_img_dir, str(df.loc[i, "id"]) + ".png")

        if not os.path.exists(img_path):
            continue

        decoded_mask = rle_decode(rle)
        orig_img = imread(img_path)
        crf_output = crf(orig_img, decoded_mask)
        df.loc[i, "rle_mask"] = rle_encode(crf_output)

out_path = "crf_correction_unet_bn_diceloss.csv"
df.to_csv(out_path, index=False)
print(f"Wrote submission: {out_path} (rows={len(df)})")

## --- ERROR in cell 0, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/4257647655.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m    108[0m [0;34m[0m[0m
[1;32m    109[0m [0;32mif[0m [0mdf[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 110[0;31m     raise FileNotFoundError(
[0m[1;32m    111[0m         [0;34m"No prediction submission found. Expected one of:\n"[0m[0;34m[0m[0;34m[0m[0m
[1;32m    112[0m         [0;34m+[0m [0;34m"\n"[0m[0;34m.[0m[0mjoin[0m[0;34m([0m[0mcandidate_paths[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mFileNotFoundError[0m: No prediction submission found. Expected one of:
../input/candsub/blend_ez.csv
../input/tgs-salt-identification-challenge/blend_ez.csv
../input/blend_ez.csv

Refusing to run with sample_submission.csv because it produces near-empty masks and a ~0 score.
