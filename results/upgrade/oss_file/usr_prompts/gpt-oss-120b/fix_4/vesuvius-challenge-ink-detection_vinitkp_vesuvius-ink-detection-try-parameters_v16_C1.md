# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Overview
Detect the presence of ink from 3d x-ray scans of detached fragments of ancient papyrus scrolls.

## Metric
We evaluate how well your output image matches our reference image using a modified version of the [Sørensen--Dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient), where instead of using the F1 score, we are using the F0.5 score. The F0.5 score is given by:

$$
\frac{\left(1+\beta^2\right) p r}{\beta^2 p+r} \text { where } p=\frac{t p}{t p+f p}, r=\frac{t p}{t p+f n}, \beta=0.5
$$

The F0.5 score weights precision higher than recall, which improves the ability to form coherent characters out of detected ink areas.

In order to reduce the submission file size, our metric uses run-length encoding on the pixel values. Instead of submitting an exhaustive list of indices for your segmentation, you will submit pairs of values that contain a start position and a run length. E.g. '1 3' implies starting at pixel 1 and running a total of 3 pixels (1,2,3).

Note that, at the time of encoding, the output should be binary, with 0 indicating "no ink" and 1 indicating "ink".

The competition format requires a space delimited list of pairs. For example, '1 3 10 5' implies pixels 1,2,3,10,11,12,13,14 are to be included in the mask. The metric checks that the pairs are sorted, positive, and the decoded pixel values are not duplicated. The pixels are numbered from left to right, then top to bottom: 1 is pixel (1,1), 2 is pixel (1,2), etc.

Your output should be a single file, **submission.csv**, with this run-length encoded information. This should have a header with two columns, `Id` and `Predicted`, and with one row for every directory under **test/**. For example:

```
Id,Predicted
a,1 1 5 1 etc.
b,10 20 etc.
```

For a real-world example of what these files look like, see `inklabels_rce.csv` in the data directories, which have been generated with [this script](https://gist.github.com/janpaul123/ca3477c1db6de4346affca37e0e3d5b0).


## Data
- **[train/test]/[fragment_id]/surface_volume/[image_id].tif** slices from the 3d x-ray [surface volume](https://scrollprize.org/tutorial1#3-surface-volumes). Each file contains a greyscale slice in the z-direction. Each fragment contains 65 slices. Combined this image stack gives us `width * height * 65` number of voxels per fragment. You can expect two fragments in the hidden test set, which together are roughly the same size as a single training fragment. The sample slices available to download in the test folders are simply copied from training fragment one, but when you submit your notebook they will be substituted with the real test data.
- **[train/test]/[fragment_id]/mask.png** --- a binary mask of which pixels contain data.
- **train/[fragment_id]/inklabels.png** --- a binary mask of the ink vs no-ink labels.
- **train/[fragment_id]/inklabels_rle.csv** --- a run-length-encoded version of the labels, generated using [this script](https://gist.github.com/janpaul123/ca3477c1db6de4346affca37e0e3d5b0). This is the same format as you should make your submission in.
- **train/[fragment_id]/ir.png** --- the infrared photo on which the binary mask is based.
- **sample_submission.csv**, an example of a submission file in the correct format. You need to output the following file in the home directory: **submission.csv**.

# 2. Python version

3.11

# 3. Installed packages

geopandas==0.14.4
ipywidgets==8.1.5
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
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (135 lines)
            sample_submission.csv (2 lines)
            sample_submission.csv.zip (215 Bytes)
            test.zip (3.0 GB)
            train.zip (15.5 GB)
            test/
                a/
                    mask.png (40.7 kB)
                    surface_volume/
                        06.tif (79.8 MB)
                        01.tif (79.8 MB)
                        ... and 63 other files
                test/
            train/
                1/
                    inklabels.png (92.6 kB)
                    inklabels_rle.csv (2 lines)
                    ... and 2 other files
                    surface_volume/
                        42.tif (103.6 MB)
                        45.tif (103.6 MB)
                        ... and 63 other files
                2/
                    inklabels.png (294.3 kB)
                    inklabels_rle.csv (2 lines)
                    ... and 2 other files
                    surface_volume/
                        10.tif (281.9 MB)
                        17.tif (281.9 MB)
                        ... and 63 other files
                train/
            vesuvius-challenge-ink-detection/
                description.md (135 lines)
                sample_submission.csv (2 lines)
                ... and 3 other files
                test/
                    a/
                        mask.png (40.7 kB)
                        surface_volume/
                            ... (max depth reached)
                    test/
                train/
                    1/
                        inklabels.png (92.6 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    2/
                        inklabels.png (294.3 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    train/
                vesuvius-challenge-ink-detection/
        input/
            description.md (135 lines)
            sample_submission.csv (2 lines)
            sample_submission.csv.zip (215 Bytes)
            test.zip (3.0 GB)
            train.zip (15.5 GB)
            test/
                a/
                    mask.png (40.7 kB)
                    surface_volume/
                        06.tif (79.8 MB)
                        01.tif (79.8 MB)
                        ... and 63 other files
                test/
                    a/
                        mask.png (40.7 kB)
                        surface_volume/
                            ... (max depth reached)
                    test/
            train/
                1/
                    inklabels.png (92.6 kB)
                    inklabels_rle.csv (2 lines)
                    ... and 2 other files
                    surface_volume/
                        42.tif (103.6 MB)
                        45.tif (103.6 MB)
                        ... and 63 other files
                2/
                    inklabels.png (294.3 kB)
                    inklabels_rle.csv (2 lines)
                    ... and 2 other files
                    surface_volume/
                        10.tif (281.9 MB)
                        17.tif (281.9 MB)
                        ... and 63 other files
                train/
                    1/
                        inklabels.png (92.6 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    2/
                        inklabels.png (294.3 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    train/
            vesuvius-challenge-ink-detection/
                description.md (135 lines)
                sample_submission.csv (2 lines)
                ... and 3 other files
                test/
                    a/
                        mask.png (40.7 kB)
                        surface_volume/
                            ... (max depth reached)
                    test/
                train/
                    1/
                        inklabels.png (92.6 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    2/
                        inklabels.png (294.3 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    train/
                vesuvius-challenge-ink-detection/
        working/
            vesuvius-challenge-ink-detection/
                description.md (135 lines)
                sample_submission.csv (2 lines)
                ... and 3 other files
                test/
                    a/
                        mask.png (40.7 kB)
                        surface_volume/
                            ... (max depth reached)
                    test/
                train/
                    1/
                        inklabels.png (92.6 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    2/
                        inklabels.png (294.3 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    train/
                vesuvius-challenge-ink-detection/
```

-> data/sample_submission.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> data/train/1/inklabels_rle.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> data/train/2/inklabels_rle.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> data/vesuvius-challenge-ink-detection/sample_submission.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> data/vesuvius-challenge-ink-detection/train/1/inklabels_rle.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> data/vesuvius-challenge-ink-detection/train/2/inklabels_rle.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> input/sample_submission.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> (stopped after 10 files for performance)

# 5. Target score

0.012406

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, glob, numpy as np, torch
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

BASE_PATH = "/kaggle/input"


def safe_load_image(path, mode=None):
    if not os.path.exists(path):
        return None
    img = Image.open(path)
    if mode:
        img = img.convert(mode)
    return img


train_frag_paths = sorted(glob.glob(os.path.join(BASE_PATH, "*", "train", "*")))
if not train_frag_paths:
    raise RuntimeError("No training fragment directories found under /kaggle/input")
TRAIN_PREFIX = os.path.join(train_frag_paths[0], "")
ir_img = safe_load_image(os.path.join(TRAIN_PREFIX, "ir.png"))
if ir_img is not None:
    plt.imshow(ir_img, cmap="gray")
    plt.title("Sample IR Image (training)")
    plt.axis("off")
    plt.show()




## === cell 1
def rle_encode(mask: np.ndarray) -> str:
    """
    Encode a 2‑D binary mask to the competition's space‑delimited RLE format.
    The mask is assumed to be 0/1 with shape (H, W) and uses 1‑based indexing.
    Returns an empty string for an all‑zero mask.
    """
    flat = mask.flatten().astype(np.uint8)

    padded = np.concatenate([[0], flat, [0]])
    changes = np.where(padded[1:] != padded[:-1])[0] + 1  # 1‑based positions

    runs = []
    for start, end in zip(changes[::2], changes[1::2]):
        runs.extend([start, end - start])
    return " ".join(map(str, runs))




## === cell 2
beta = 0.5
beta_sq = beta**2


def fbeta_score(tp, fp, fn, beta_sq=beta_sq):
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


train_frag_dirs = sorted(
    [
        d
        for d in glob.glob(os.path.join(BASE_PATH, "*", "train", "*"))
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "inklabels.png"))
    ]
)

ir_arrays = []
label_arrays = []
for frag_dir in tqdm(train_frag_dirs, desc="Loading training data"):
    ir_path = os.path.join(frag_dir, "ir.png")
    label_path = os.path.join(frag_dir, "inklabels.png")
    ir_img = safe_load_image(ir_path, mode="L")
    label_img = safe_load_image(label_path, mode="L")
    if ir_img is None or label_img is None:
        continue
    ir_np = np.array(ir_img).astype(np.uint8)
    label_np = (np.array(label_img) > 0).astype(np.uint8)
    ir_arrays.append(ir_np.ravel())
    label_arrays.append(label_np.ravel())

if not ir_arrays:
    raise RuntimeError("No training IR images with labels found.")

ir_all = np.concatenate(ir_arrays)
label_all = np.concatenate(label_arrays)

best_thresh = 0
best_fbeta = -1.0
for thresh in range(0, 256):
    pred = (ir_all > thresh).astype(np.uint8)
    tp = np.sum((pred == 1) & (label_all == 1))
    fp = np.sum((pred == 1) & (label_all == 0))
    fn = np.sum((pred == 0) & (label_all == 1))
    score = fbeta_score(tp, fp, fn)
    if score > best_fbeta:
        best_fbeta = score
        best_thresh = thresh

print(f"Optimal intensity threshold: {best_thresh} (F0.5 ≈ {best_fbeta:.5f})")




## === cell 3
test_root_candidates = sorted(glob.glob(os.path.join(BASE_PATH, "*", "test", "*")))
test_frag_dirs = [
    p
    for p in test_root_candidates
    if os.path.isdir(p) and os.path.exists(os.path.join(p, "mask.png"))
]

if not test_frag_dirs:
    raise RuntimeError("No test fragment directories with mask.png found.")

submission_path = "submission.csv"
with open(submission_path, "w") as f:
    f.write("Id,Predicted\n")
    for frag_dir in tqdm(test_frag_dirs, desc="Creating submission"):
        frag_id = os.path.basename(frag_dir)
        mask_path = os.path.join(frag_dir, "mask.png")
        ir_path = os.path.join(frag_dir, "ir.png")

        mask_img = safe_load_image(mask_path, mode="L")
        if mask_img is None:
            f.write(f"{frag_id},\n")
            continue
        mask_np = np.array(mask_img) > 0

        ir_img = safe_load_image(ir_path, mode="L")
        if ir_img is None:
            pred_np = mask_np  # fallback: predict everywhere mask is present
        else:
            ir_np = np.array(ir_img).astype(np.uint8)
            pred_np = (ir_np > best_thresh) & mask_np  # apply threshold & mask

        rle_str = rle_encode(pred_np.astype(np.uint8))
        f.write(f"{frag_id},{rle_str}\n")

print(f"Submission file written to {submission_path}")




## === cell 4
first_test_dir = test_frag_dirs[0]
first_mask = safe_load_image(os.path.join(first_test_dir, "mask.png"), mode="L")
first_ir = safe_load_image(os.path.join(first_test_dir, "ir.png"), mode="L")
if first_mask is not None and first_ir is not None:
    mask_np = np.array(first_mask) > 0
    pred_np = (np.array(first_ir).astype(np.uint8) > best_thresh) & mask_np
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(first_ir, cmap="gray")
    plt.title("IR image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(pred_np, cmap="gray")
    plt.title("Predicted ink mask")
    plt.axis("off")
    plt.show()
