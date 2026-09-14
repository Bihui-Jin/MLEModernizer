# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

3.9

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

0.71819

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, zipfile, warnings
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from skimage.transform import resize
from skimage.filters import threshold_otsu
from PIL import Image

warnings.filterwarnings("ignore")




## === cell 1
def safe_unzip(zip_path, extract_to):
    """Extract zip only if it exists and target folder is missing."""
    if not os.path.isdir(extract_to) and os.path.isfile(zip_path):
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_to)


base_dir = os.path.join(".", "data")

safe_unzip(os.path.join(base_dir, "train.zip"), os.path.join(base_dir, "train"))
safe_unzip(os.path.join(base_dir, "test.zip"), os.path.join(base_dir, "test"))


class config:
    im_width = 128
    im_height = 128
    im_chan = 1
    path_train = os.path.join(base_dir, "train")
    path_test = os.path.join(base_dir, "test")




## === cell 2
sample_sub_path = os.path.join(base_dir, "sample_submission.csv")
if not os.path.isfile(sample_sub_path):
    raise FileNotFoundError(f"sample_submission.csv not found at {sample_sub_path}")

test_ids = pd.read_csv(sample_sub_path)["id"].astype(str).tolist()
print(f"Found {len(test_ids)} test ids from sample_submission.csv.")

test_images_dir = os.path.join(config.path_test, "images")
if not os.path.isdir(test_images_dir):
    raise FileNotFoundError(f"Test images directory not found: {test_images_dir}")



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/832659389.py in <cell line: 0>()
      2 sample_sub_path = os.path.join(base_dir, "sample_submission.csv")
      3 if not os.path.isfile(sample_sub_path):
----> 4     raise FileNotFoundError(f"sample_submission.csv not found at {sample_sub_path}")
      5 
      6 test_ids = pd.read_csv(sample_sub_path)["id"].astype(str).tolist()

FileNotFoundError: sample_submission.csv not found at ./data/sample_submission.csv

## === cell 3
X_test = np.zeros(
    (len(test_ids), config.im_height, config.im_width, config.im_chan), dtype=np.uint8
)
sizes_test = []

print("Loading and resizing test images...")
for n, img_id in tqdm(enumerate(test_ids), total=len(test_ids)):
    img_path = os.path.join(test_images_dir, f"{img_id}.png")
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")

    img = Image.open(img_path).convert("L")
    x = np.array(img)  # shape (H, W)
    sizes_test.append((x.shape[0], x.shape[1]))  # original (height, width)

    x_resized = resize(
        x, (config.im_height, config.im_width), mode="constant", preserve_range=True
    )
    X_test[n, :, :, 0] = x_resized.astype(np.uint8)

print("Done loading test images.")



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/704208665.py in <cell line: 0>()
      1 # Prepare array for resized test images and keep original sizes
      2 X_test = np.zeros(
----> 3     (len(test_ids), config.im_height, config.im_width, config.im_chan), dtype=np.uint8
      4 )
      5 sizes_test = []

NameError: name 'test_ids' is not defined

## === cell 4
preds_resized = np.zeros_like(X_test, dtype=np.uint8)

print("Generating simple threshold‑based masks...")
for i in trange(len(X_test)):
    img = X_test[i, :, :, 0]
    thresh = threshold_otsu(img)
    mask = (img > thresh).astype(np.uint8)
    preds_resized[i, :, :, 0] = mask
print("Mask generation complete.")

preds_test_upsampled = []
print("Upsampling masks to original sizes...")
for i in trange(len(preds_resized)):
    mask_small = preds_resized[i, :, :, 0]
    up_mask = resize(
        mask_small,
        sizes_test[i],
        mode="constant",
        preserve_range=True,
        anti_aliasing=False,
    )
    up_mask = (up_mask > 0.5).astype(np.uint8)
    preds_test_upsampled.append(up_mask)
print("Upsampling done.")




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3528004693.py in <cell line: 0>()
      1 # Create a very simple prediction: threshold each resized image at its mean intensity.
      2 # This yields a binary mask (0/1) for each image.
----> 3 preds_resized = np.zeros_like(X_test, dtype=np.uint8)
      4 
      5 print("Generating simple threshold‑based masks...")

NameError: name 'X_test' is not defined

## === cell 5
def RLenc(img, order="F", format=True):
    """
    Run-length encoding.
    img: binary mask (2D numpy array)
    order: 'F' for Fortran order (column‑major) as required by competition
    format: if True, returns a space‑separated string; otherwise a list of tuples
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []
    r = 0
    pos = 1
    for c in bytes:
        if c == 0:
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1
    if r != 0:
        runs.append((pos, r))

    if format:
        return " ".join(f"{start} {length}" for start, length in runs)
    return runs


pred_dict = {
    img_id: RLenc(np.round(preds_test_upsampled[i]).astype(np.uint8))
    for i, img_id in tqdm(enumerate(test_ids), total=len(test_ids))
}



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3902154389.py in <cell line: 0>()
     30 pred_dict = {
     31     img_id: RLenc(np.round(preds_test_upsampled[i]).astype(np.uint8))
---> 32     for i, img_id in tqdm(enumerate(test_ids), total=len(test_ids))
     33 }
     34 

NameError: name 'test_ids' is not defined

## === cell 6
sub = pd.DataFrame.from_dict(pred_dict, orient="index", columns=["rle_mask"])
sub.index.name = "id"
submission_path = "submission.csv"
sub.to_csv(submission_path)
print(f"Submission file written to {submission_path}")

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/265328258.py in <cell line: 0>()
      1 # Create submission DataFrame and write to CSV
----> 2 sub = pd.DataFrame.from_dict(pred_dict, orient="index", columns=["rle_mask"])
      3 sub.index.name = "id"
      4 submission_path = "submission.csv"
      5 sub.to_csv(submission_path)

NameError: name 'pred_dict' is not defined
