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


def find_root_dir():
    candidates = [
        os.path.join(".", "kaggle", "data"),
        os.path.join(".", "data"),
        os.path.join(".", "input"),
    ]
    for cand in candidates:
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError("No data root directory found among expected locations.")


root_dir = find_root_dir()
base_dir_data = root_dir  # zip files and extracted train/test
base_dir_input = root_dir  # sample_submission.csv etc.


def safe_unzip(zip_path, extract_to):
    """Extract zip only if it exists and target folder is missing."""
    if not os.path.isdir(extract_to) and os.path.isfile(zip_path):
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_to)


safe_unzip(
    os.path.join(base_dir_data, "train.zip"), os.path.join(base_dir_data, "train")
)
safe_unzip(os.path.join(base_dir_data, "test.zip"), os.path.join(base_dir_data, "test"))


class config:
    im_width = 128
    im_height = 128
    im_chan = 1
    path_train = os.path.join(base_dir_data, "train")
    path_test = os.path.join(base_dir_data, "test")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/447040936.py in <cell line: 0>()
     24 
     25 
---> 26 root_dir = find_root_dir()
     27 base_dir_data = root_dir  # zip files and extracted train/test
     28 base_dir_input = root_dir  # sample_submission.csv etc.

/tmp/ipykernel_11/447040936.py in find_root_dir()
     21         if os.path.isdir(cand):
     22             return cand
---> 23     raise FileNotFoundError("No data root directory found among expected locations.")
     24 
     25 

FileNotFoundError: No data root directory found among expected locations.

## === cell 1
possible_paths = [
    os.path.join(base_dir_input, "sample_submission.csv"),
]
sample_sub_path = next((p for p in possible_paths if os.path.isfile(p)), None)
if sample_sub_path is None:
    raise FileNotFoundError("sample_submission.csv not found in expected locations.")

test_ids = pd.read_csv(sample_sub_path)["id"].astype(str).tolist()
print(f"Found {len(test_ids)} test ids from sample_submission.csv.")

test_images_dir = os.path.join(config.path_test, "images")
if not os.path.isdir(test_images_dir):
    raise FileNotFoundError(f"Test images directory not found: {test_images_dir}")



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2550641625.py in <cell line: 0>()
      1 # locate sample_submission.csv
      2 possible_paths = [
----> 3     os.path.join(base_dir_input, "sample_submission.csv"),
      4 ]
      5 sample_sub_path = next((p for p in possible_paths if os.path.isfile(p)), None)

NameError: name 'base_dir_input' is not defined

## === cell 2
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



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1347837563.py in <cell line: 0>()
      1 X_test = np.zeros(
----> 2     (len(test_ids), config.im_height, config.im_width, config.im_chan), dtype=np.uint8
      3 )
      4 sizes_test = []
      5 

NameError: name 'test_ids' is not defined

## === cell 3
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


def RLenc(img, order="F", format=True):
    """
    Run-length encoding.
    img: binary mask (2D numpy array)
    order: 'F' for Fortran order (column‑major) as required by competition
    format: if True, returns a space‑separated string; otherwise a list of tuples
    """
    bytes_flat = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []
    r = 0
    pos = 1
    for c in bytes_flat:
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



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3860888214.py in <cell line: 0>()
----> 1 preds_resized = np.zeros_like(X_test, dtype=np.uint8)
      2 
      3 print("Generating simple threshold‑based masks...")
      4 for i in trange(len(X_test)):
      5     img = X_test[i, :, :, 0]

NameError: name 'X_test' is not defined

## === cell 4
sub = pd.DataFrame.from_dict(pred_dict, orient="index", columns=["rle_mask"])
sub.index.name = "id"
submission_path = "submission.csv"
sub.to_csv(submission_path)
print(f"Submission file written to {submission_path}")

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3626281432.py in <cell line: 0>()
----> 1 sub = pd.DataFrame.from_dict(pred_dict, orient="index", columns=["rle_mask"])
      2 sub.index.name = "id"
      3 submission_path = "submission.csv"
      4 sub.to_csv(submission_path)
      5 print(f"Submission file written to {submission_path}")

NameError: name 'pred_dict' is not defined
