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

0.8086888792730411

# 6. Current score

0.5221

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5221) has done: 'I remove the unavailable `pydensecrf` dependency (it isn’t installed in your environment) and replace the CRF refinement step with a lightweight, deterministic mask post-processing (morphological closing/opening) that keeps the pipeline intent (refine predicted masks) while making it runnable. I also fix the notebook-only `%matplotlib inline` line, restore missing imports so `pd/plt/tqdm` are defined, and make the code robust to missing external submission inputs by falling back to `sample_submission.csv` if the referenced `../input/u-net-with-simple-resnet-blocks-forked/submission.csv` doesn’t exist. Finally, I ensure the script always writes a valid Kaggle submission CSV with the required columns (`id,rle_mask`) and a `.csv` suffix.'
- What this solution (achieved 0.5221) has done: 'Your current score is far below the target, so the smallest meaningful improvement is to fix likely correctness issues and better mimic the original “CRF refinement” intent without changing the upstream model submission you’re refining. I (1) make RLE decoding match the competition’s column-major convention (your encode is correct but your decode is not), and (2) switch the post-processing to a slightly stronger, image-aware refinement (morphology + boundary snapping from the seismic image gradient) while keeping it deterministic and lightweight. I also ensure the refined submission stays aligned to `sample_submission.csv` ids to avoid any ordering/missing-id penalties. These changes keep the same overall pipeline: load base submission → decode mask → refine using the image → encode → write CSV.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from skimage.io import imread
from skimage.morphology import (
    binary_closing,
    binary_opening,
    remove_small_holes,
    remove_small_objects,
    square,
)
from skimage.filters import sobel
from skimage.segmentation import (
    morphological_geodesic_active_contour,
    inverse_gaussian_gradient,
)

import matplotlib.pyplot as plt

np.random.seed(100)

DATA_ROOT = "/kaggle/input/tgs-salt-identification-challenge"
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test", "images")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")




## === cell 1
def rle_decode(rle_mask: str, shape=(101, 101)):
    """
    Decode RLE string into a binary mask.

    IMPORTANT for this competition:
      - RLE is 1-indexed
      - Flattening order is column-major (Fortran order): pixels are numbered top-to-bottom, then left-to-right.
    So decoding must reconstruct in that same (column-major) convention.

    Minimal change, but critical for correctness: this fixes a mismatch where encode was column-major
    but decode was effectively row-major.
    """
    if rle_mask is None:
        return np.zeros(shape, dtype=np.uint8)
    rle_mask = str(rle_mask)
    if rle_mask.strip() == "" or rle_mask.lower() == "nan":
        return np.zeros(shape, dtype=np.uint8)

    s = rle_mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape((shape[1], shape[0])).T




## === cell 2
def rle_encode(im: np.ndarray):
    """
    Encode binary mask into RLE.

    Kaggle TGS Salt expects column-major order (Fortran order) when flattening.
    """
    im = (im > 0).astype(np.uint8)
    pixels = im.T.flatten()  # column-major / Fortran-equivalent
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    if len(runs) == 0:
        return ""
    return " ".join(str(x) for x in runs)




## === cell 3
def load_base_submission():
    """
    Keep the same preference for an external base submission if present,
    but always fall back to the competition sample submission so we can run end-to-end.

    Additionally, align to sample_submission ids to avoid any ordering/missing-id penalties.
    """
    preferred = "../input/u-net-with-simple-resnet-blocks-forked/submission.csv"
    if os.path.exists(preferred):
        df_sub = pd.read_csv(preferred)
    else:
        df_sub = pd.read_csv(SAMPLE_SUB_PATH)

    if "id" not in df_sub.columns:
        raise ValueError("Submission-like file must contain an 'id' column.")
    if "rle_mask" not in df_sub.columns:
        for alt in ["rle", "mask", "rle_encoded", "encoded_pixels"]:
            if alt in df_sub.columns:
                df_sub = df_sub.rename(columns={alt: "rle_mask"})
                break
    if "rle_mask" not in df_sub.columns:
        df_sub["rle_mask"] = ""

    df_sub = df_sub[["id", "rle_mask"]].copy()
    df_sub["rle_mask"] = df_sub["rle_mask"].fillna("")

    sample = pd.read_csv(SAMPLE_SUB_PATH)[["id"]].copy()
    df_sub = sample.merge(df_sub, on="id", how="left")
    df_sub["rle_mask"] = df_sub["rle_mask"].fillna("")
    return df_sub


df = load_base_submission()




## === cell 4
def refine_mask_with_morphology_and_edges(
    original_image: np.ndarray, mask_img: np.ndarray
):
    """
    Deterministic refinement that stays in the same "refine predicted masks using the image" spirit:
      1) Clean mask with mild morphology (close/open, remove tiny objects/holes)
      2) If mask is non-empty, snap boundaries toward image edges using a few iterations
         of morphological geodesic active contour (MGAC) driven by the image gradient.

    This is still lightweight (101x101, few iterations) and avoids unavailable CRF dependencies,
    but tends to improve boundary alignment vs pure morphology.
    """
    if original_image.ndim == 3:
        img = original_image[:, :, 0]
    else:
        img = original_image

    img = img.astype(np.float32)
    img_min, img_max = float(np.min(img)), float(np.max(img))
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = np.zeros_like(img, dtype=np.float32)

    m = (mask_img > 0).astype(bool)

    m = binary_closing(m, square(3))
    m = binary_opening(m, square(3))
    m = remove_small_objects(m, min_size=20)
    m = remove_small_holes(m, area_threshold=20)

    if m.sum() > 0:
        gimage = inverse_gaussian_gradient(img, alpha=100.0, sigma=1.0)

        init_ls = m.astype(np.int8)

        try:
            refined = morphological_geodesic_active_contour(
                gimage,
                num_iter=12,
                init_level_set=init_ls,
                smoothing=1,
                balloon=0,
                threshold="auto",
            ).astype(bool)
            m = refined
        except Exception:
            pass

    return m.astype(np.uint8)




## === cell 5
try:
    non_empty_idx = []
    for i in range(min(len(df), 200)):
        rle = str(df.loc[i, "rle_mask"])
        if rle.strip() != "" and rle.lower() != "nan":
            non_empty_idx.append(i)
        if len(non_empty_idx) >= 6:
            break

    if len(non_empty_idx) > 0:
        plt.figure(figsize=(18, 3))
        for j, idx in enumerate(non_empty_idx):
            decoded_mask = rle_decode(df.loc[idx, "rle_mask"])
            plt.subplot(1, len(non_empty_idx), j + 1)
            plt.imshow(decoded_mask, cmap="gray")
            plt.axis("off")
            plt.title(df.loc[idx, "id"])
        plt.tight_layout()
except Exception:
    pass



## === cell 6
test_path = os.path.join(DATA_ROOT, "test", "images") + "/"
if not os.path.isdir(test_path):
    raise FileNotFoundError(f"Test images folder not found: {test_path}")



## === cell 7
for i in tqdm(range(df.shape[0]), desc="Refining masks"):
    rle = df.loc[i, "rle_mask"]
    if rle is None or str(rle).strip() == "" or str(rle).lower() == "nan":
        df.loc[i, "rle_mask"] = ""
        continue

    img_id = df.loc[i, "id"]
    img_path = test_path + img_id + ".png"
    if not os.path.exists(img_path):
        df.loc[i, "rle_mask"] = ""
        continue

    decoded_mask = rle_decode(rle, shape=(101, 101))
    orig_img = imread(img_path)

    refined = refine_mask_with_morphology_and_edges(orig_img, decoded_mask)
    df.loc[i, "rle_mask"] = rle_encode(refined)



## === cell 8
df = df[["id", "rle_mask"]].copy()
df["rle_mask"] = df["rle_mask"].fillna("")

out_path = "crf_correction.csv"
df.to_csv(out_path, index=False)

print(f"Wrote submission: {out_path} with shape={df.shape}")
print(df.head())
