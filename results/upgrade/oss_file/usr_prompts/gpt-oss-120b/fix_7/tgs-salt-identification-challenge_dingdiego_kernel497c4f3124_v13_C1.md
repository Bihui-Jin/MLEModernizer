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

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The fixes import only the packages that are available, replace the unavailable CRF library with a simple identity function, correctly read the provided sample submission, and ensure all variables (`pd`, `plt`, `tqdm`, etc.) are defined so the script runs end‑to‑end and writes a valid `crf_correction.csv` file.'
- What this solution (achieved 0.0) has done: 'Implemented a simple fallback mask generation for rows where the original submission had missing RLE masks.  
- If `rle_mask` is NaN, the code now creates a basic binary mask by thresholding the average pixel intensity of the original image (values > 0.5 become salt).  
- This mask is then encoded back to RLE, ensuring every entry has a valid mask, which moves the Kaggle score away from 0.0 toward the target.  
- The rest of the pipeline remains unchanged, preserving the original logic and output format.'
- What this solution (achieved 0.0) has done: 'I fix the image path handling, replace the fixed 0.5 threshold with an adaptive Otsu threshold, and clean each mask with a small‑hole removal step so the generated masks more closely resemble the true salt shapes. These modest adjustments keep the original CRF‑identity logic while improving the quality of the submitted masks, which should raise the score toward the target without over‑hauling the pipeline.'
- What this solution (achieved 0.0) has done: 'I add a few lightweight morphological clean‑up steps (small‑object removal and binary opening) after the existing hole‑filling, which modestly improves mask quality and should raise the mean‑average‑precision toward the target without altering the core pipeline. I also import the needed functions at the top.'
- What this solution (achieved 0.0) has done: 'The fix updates the morphological opening call to use the correct `footprint` argument (instead of the removed `selem`) and streamlines its usage. Small robustness tweaks are added: building image paths with `os.path.join`, ensuring the mask is non‑empty (set a single pixel if all zeros) and simplifying type conversions. These changes resolve the runtime error and produce a valid RLE submission, moving the score from 0.0 toward the target.'
- What this solution (achieved 0.0) has done: 'The changes add robust handling for empty or missing RLE strings, ensure the sample submission is loaded from any typical input location, and adjust the mask‑generation loop to treat blank masks like NaNs. This produces non‑empty masks for all rows, yielding a valid `crf_correction.csv` and moving the score away from 0.0 toward the target.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.io import imread
from skimage.color import gray2rgb
import os
from skimage.filters import threshold_otsu
from skimage.morphology import (
    remove_small_holes,
    remove_small_objects,
    binary_opening,
    disk,
)




## === cell 1
def rle_decode(rle_mask):
    """
    Decode a run‑length encoded mask string into a 101×101 binary array.
    Handles empty or NaN inputs by returning an all‑zero mask.
    """
    if pd.isna(rle_mask) or not isinstance(rle_mask, str) or rle_mask.strip() == "":
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
_possible_submission_paths = [
    "../input/sample_submission.csv",
    "./input/sample_submission.csv",
    "sample_submission.csv",
]
df = None
for p in _possible_submission_paths:
    if os.path.isfile(p):
        df = pd.read_csv(p)
        print(f"Loaded submission from: {p}")
        break
if df is None:
    raise FileNotFoundError("sample_submission.csv not found in expected locations.")
print(f"Loaded submission with {len(df)} rows.")



## === cell 3
plt.figure(figsize=(15, 3))
for idx in range(min(5, len(df))):
    if pd.isna(df.loc[idx, "rle_mask"]) or df.loc[idx, "rle_mask"].strip() == "":
        continue
    mask = rle_decode(df.loc[idx, "rle_mask"])
    plt.subplot(1, 5, idx + 1)
    plt.imshow(mask, cmap="gray")
    plt.title(df.loc[idx, "id"])
    plt.axis("off")
plt.show()



## === cell 4
_possible_paths = [
    "../input/tgs-salt-identification-challenge/test/images/",
    "../input/test/images/",
    "test/images/",
]
test_path = None
for p in _possible_paths:
    if os.path.isdir(p):
        test_path = p
        break
if test_path is None:
    raise FileNotFoundError("Test image directory not found.")
print(f"Using test images from: {test_path}")




## === cell 5
def crf(original_image, mask_img):
    """
    Placeholder CRF function.
    The original code used pydensecrf, which is unavailable.
    We simply return the input mask unchanged (identity operation),
    preserving shape and type.
    """
    if mask_img.shape != original_image.shape[:2]:
        mask_img = np.resize(mask_img, original_image.shape[:2])
    return mask_img.astype(np.uint8)




## === cell 6
n_imgs = 3
i = np.random.randint(0, len(df))
j = 1
plt.figure(figsize=(15, 5))
while j <= n_imgs:
    if pd.isna(df.loc[i, "rle_mask"]) or df.loc[i, "rle_mask"].strip() == "":
        i = (i + 1) % len(df)
        continue
    decoded_mask = rle_decode(df.loc[i, "rle_mask"])
    orig_img = imread(os.path.join(test_path, df.loc[i, "id"] + ".png"))
    crf_output = crf(orig_img, decoded_mask)

    plt.subplot(n_imgs, 3, (j - 1) * 3 + 1)
    plt.imshow(orig_img)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(n_imgs, 3, (j - 1) * 3 + 2)
    plt.imshow(decoded_mask, cmap="gray")
    plt.title("Decoded Mask")
    plt.axis("off")

    plt.subplot(n_imgs, 3, (j - 1) * 3 + 3)
    plt.imshow(crf_output, cmap="gray")
    plt.title("CRF Output")
    plt.axis("off")

    j += 1
    i = (i + 1) % len(df)

plt.tight_layout()
plt.show()




## === cell 7
def rle_encode(im):
    """
    Encode a binary mask (numpy array) to run‑length encoding string.
    """
    pixels = im.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 8
for idx in tqdm(range(len(df)), desc="Applying CRF"):
    img_path = os.path.join(test_path, df.loc[idx, "id"] + ".png")
    orig_img = imread(img_path)
    if orig_img.ndim == 3:
        gray = orig_img.mean(axis=2)
    else:
        gray = orig_img

    mask_str = df.loc[idx, "rle_mask"]
    if pd.isna(mask_str) or not isinstance(mask_str, str) or mask_str.strip() == "":
        thresh = threshold_otsu(gray)
        mask = (gray > thresh).astype(np.uint8)
    else:
        decoded_mask = rle_decode(mask_str)
        mask = crf(orig_img, decoded_mask)

    mask = remove_small_holes(mask.astype(bool), area_threshold=64)
    mask = remove_small_objects(mask, min_size=64)
    mask = binary_opening(mask, footprint=disk(3))
    mask = mask.astype(np.uint8)

    if mask.shape != (101, 101):
        mask = np.resize(mask, (101, 101))

    if mask.sum() == 0:
        mask[0, 0] = 1

    df.loc[idx, "rle_mask"] = rle_encode(mask)



## === cell 9
df.to_csv("crf_correction.csv", index=False)
print("Saved corrected submission to crf_correction.csv")
