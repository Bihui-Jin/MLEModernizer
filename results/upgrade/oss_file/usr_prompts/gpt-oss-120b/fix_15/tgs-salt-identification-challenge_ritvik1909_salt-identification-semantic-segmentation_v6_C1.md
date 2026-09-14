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

0.0688

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0677) has done: 'The fix expands the search for the data root directory (including the typical Kaggle `/kaggle/input` path), ensures the input and data directories are set correctly, and renumbers the cells so they start at 1 as required. With these adjustments the script can locate the CSV files and images, generate simple Otsu‑threshold masks, encode them in RLE, and write a valid `submission.csv`.'
- What this solution (achieved 0.0652) has done: 'I keep the same Otsu‑threshold approach but avoid the unnecessary resize‑up‑sample cycle, which was degrading mask quality. The config is changed to use the native 101 × 101 image size, the test images are loaded directly without resizing, and the up‑sampling step is removed. This small adjustment preserves the core logic while giving the model sharper masks, which should raise the mean AP markedly toward the target score. The script also continues to write a proper `submission.csv`.'
- What this solution (achieved 0.0272) has done: 'The update adds a few simple post‑processing steps to the Otsu‑threshold masks: it selects the polarity (greater‑than or less‑than) that yields the smaller foreground area (salt tends to occupy less than half the image), fills interior holes, and removes tiny isolated objects. These cheap operations keep the original Otsu‑based approach while producing cleaner masks, which should raise the mean AP and move the score closer to the target.'
- What this solution (achieved 0.0374) has done: 'Implemented a fix for the `binary_dilation` call, replacing the deprecated `selem` argument with the correct `footprint` parameter. This resolves the TypeError that halted mask generation, allowing `pred_dict` to be created and the submission CSV to be written successfully.'
- What this solution (achieved 0.0359) has done: 'The plan adds two lightweight improvements that keep the original Otsu‑based pipeline but fix the RLE encoding (so masks are read correctly) and keep only the largest connected component, which usually corresponds to the salt body. Both changes are small, preserve the core logic, and are expected to raise the mean AP noticeably toward the target score.'
- What this solution (achieved 0.1172) has done: 'I rename the cells to start at 1, add a light Gaussian blur before Otsu, and select the larger‑area polarity (salt typically occupies more than half the image) instead of the smaller one. I also tighten the small‑object removal and make the RLE encoder safely return an empty string for empty masks. These minimal yet targeted changes keep the overall Otsu‑based pipeline while improving mask quality, which should raise the mean AP toward the target score.'
- What this solution (achieved 0.0389) has done: 'I adjust the mask‑selection logic to favor the smaller‑area polarity (salt usually occupies a minority of the image) and tune the post‑processing parameters (smaller object removal size and a slightly larger dilation kernel). These lightweight changes keep the Otsu‑based pipeline intact while improving mask quality, which should raise the mean AP toward the target score.'
- What this solution (achieved 0.0509) has done: 'I add a lightweight training‑phase that scans the provided training masks to compute the typical proportion of salt pixels, then use that statistic to pick the Otsu polarity (greater‑than or less‑than) that best matches the expected salt area for each test image. I also tighten the post‑processing (larger dilation disk, slightly larger object‑removal size) which together raise the mean AP while keeping the original Otsu‑based pipeline unchanged.'
- What this solution (achieved 0.0688) has done: 'I add a simple depth‑based estimate of the expected salt proportion and use it to choose the Otsu polarity for each test image, then tighten the post‑processing (smaller object removal size and a slightly larger dilation disk). These changes keep the original Otsu‑threshold pipeline while giving a better per‑image expectation, which should raise the mean AP toward the target score.'

# 9. Code solution

## === cell 0
import os, zipfile, warnings, glob
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import remove_small_objects, binary_dilation, disk
from skimage.measure import label
from scipy.ndimage import binary_fill_holes
from PIL import Image

warnings.filterwarnings("ignore")


def find_root_dir():
    """Return the first existing directory among typical Kaggle locations."""
    candidates = [
        os.path.join(".", "kaggle", "data"),
        os.path.join(".", "data"),
        os.path.join(".", "input"),
        os.path.join("/", "kaggle", "input"),
        os.path.abspath("."),
    ]
    for cand in candidates:
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError("No data root directory found among expected locations.")


root_dir = find_root_dir()
base_dir_data = root_dir
base_dir_input = root_dir


def safe_unzip(zip_path, extract_to):
    if not os.path.isdir(extract_to) and os.path.isfile(zip_path):
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_to)


safe_unzip(
    os.path.join(base_dir_data, "train.zip"), os.path.join(base_dir_data, "train")
)
safe_unzip(os.path.join(base_dir_data, "test.zip"), os.path.join(base_dir_data, "test"))


class config:
    im_height = 101
    im_width = 101
    im_chan = 1
    path_train = os.path.join(base_dir_data, "train")
    path_test = os.path.join(base_dir_data, "test")


train_masks_dir = os.path.join(config.path_train, "masks")
mask_paths = glob.glob(os.path.join(train_masks_dir, "*.png"))
salt_areas = []
salt_area_by_id = {}

print("Scanning training masks to estimate average salt area...")
for mp in tqdm(mask_paths, desc="train masks"):
    mask_img = Image.open(mp).convert("L")
    mask_arr = np.array(mask_img)
    mask_bin = (mask_arr > 127).astype(np.uint8)
    area = mask_bin.mean()
    salt_areas.append(area)
    img_id = os.path.splitext(os.path.basename(mp))[0]
    salt_area_by_id[img_id] = area

if len(salt_areas) == 0:
    avg_salt_area = 0.2
else:
    avg_salt_area = np.mean(salt_areas)

print(f"Estimated average salt proportion (train): {avg_salt_area:.4f}")

depth_path = os.path.join(base_dir_input, "depths.csv")
if os.path.isfile(depth_path):
    depth_df = pd.read_csv(depth_path).astype({"id": str})
    train_ids = list(salt_area_by_id.keys())
    train_depths = depth_df[depth_df["id"].isin(train_ids)]["z"].values
    train_areas = np.array([salt_area_by_id[i] for i in train_ids])
    if len(train_depths) > 1:
        slope, intercept = np.polyfit(train_depths, train_areas, 1)
        depth_coeff = (slope, intercept)
        print(f"Depth regression: area = {slope:.4e} * depth + {intercept:.4f}")
    else:
        depth_coeff = None
else:
    depth_coeff = None




## === cell 1
sample_sub_path = os.path.join(base_dir_input, "sample_submission.csv")
if not os.path.isfile(sample_sub_path):
    raise FileNotFoundError("sample_submission.csv not found.")
test_ids = pd.read_csv(sample_sub_path)["id"].astype(str).tolist()
print(f"Found {len(test_ids)} test ids from sample_submission.csv.")

test_images_dir = os.path.join(config.path_test, "images")
if not os.path.isdir(test_images_dir):
    raise FileNotFoundError(f"Test images directory not found: {test_images_dir}")

X_test = np.zeros(
    (len(test_ids), config.im_height, config.im_width, config.im_chan), dtype=np.uint8
)

print("Loading test images...")
for n, img_id in tqdm(enumerate(test_ids), total=len(test_ids)):
    img_path = os.path.join(test_images_dir, f"{img_id}.png")
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")
    img = Image.open(img_path).convert("L")
    x = np.array(img)  # (101,101)
    X_test[n, :, :, 0] = x.astype(np.uint8)
print("Done loading test images.")

test_depths = {}
if depth_coeff is not None:
    depth_df = pd.read_csv(depth_path).astype({"id": str})
    test_depths = dict(zip(depth_df["id"], depth_df["z"]))




## === cell 2
preds_resized = np.zeros_like(X_test, dtype=np.uint8)

print("Generating Otsu‑threshold masks with refined post‑processing...")
for i in trange(len(X_test)):
    img = X_test[i, :, :, 0]

    img_blur = gaussian(img, sigma=1, preserve_range=True).astype(np.uint8)

    thresh = threshold_otsu(img_blur)

    mask_gt = (img_blur > thresh).astype(np.uint8)
    mask_lt = (img_blur < thresh).astype(np.uint8)

    area_gt = mask_gt.sum() / mask_gt.size
    area_lt = mask_lt.sum() / mask_lt.size

    img_id = test_ids[i]
    if depth_coeff is not None and img_id in test_depths:
        slope, intercept = depth_coeff
        expected_prop = slope * test_depths[img_id] + intercept
        expected_prop = np.clip(expected_prop, 0.0, 1.0)
    else:
        expected_prop = avg_salt_area

    if abs(area_gt - expected_prop) < abs(area_lt - expected_prop):
        mask = mask_gt
    else:
        mask = mask_lt

    mask_filled = binary_fill_holes(mask).astype(np.uint8)

    mask_dilated = binary_dilation(mask_filled, footprint=disk(5)).astype(np.uint8)

    mask_clean = remove_small_objects(mask_dilated.astype(bool), min_size=5).astype(
        np.uint8
    )

    labeled = label(mask_clean)
    if labeled.max() > 0:
        largest_cc = np.argmax(np.bincount(labeled.flat)[1:]) + 1
        mask_clean = (labeled == largest_cc).astype(np.uint8)

    preds_resized[i, :, :, 0] = mask_clean
print("Mask generation complete.")

preds_test_upsampled = [preds_resized[i, :, :, 0] for i in range(len(preds_resized))]


def RLenc(img, order="F"):
    """
    Run‑length encoding compatible with the competition.
    Returns a space‑separated string of start positions (1‑based) and lengths.
    """
    flat = img.reshape(-1, order=order)
    padded = np.concatenate([[0], flat, [0]])
    changes = np.where(padded[1:] != padded[:-1])[0] + 1
    if len(changes) == 0:
        return ""  # empty mask
    runs = []
    for start, end in zip(changes[0::2], changes[1::2]):
        runs.append(f"{start} {end - start}")
    return " ".join(runs)


pred_dict = {
    img_id: RLenc(np.round(preds_test_upsampled[i]).astype(np.uint8))
    for i, img_id in tqdm(enumerate(test_ids), total=len(test_ids))
}




## === cell 3
sub = pd.DataFrame.from_dict(pred_dict, orient="index", columns=["rle_mask"])
sub.index.name = "id"
submission_path = "submission.csv"
sub.to_csv(submission_path)
print(f"Submission file written to {submission_path}")
