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

catboost==1.2.8
geopandas==0.14.4
ipywidgets==8.1.5
joblib==1.5.2
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
sklearn-pandas==2.2.0
tifffile==2025.6.11
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

0.0968707289588988

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd
import cv2
import tifffile
from scipy.ndimage import binary_opening

target_score = 0.0968707289588988


def rle_encode(mask):
    """Convert binary mask to RLE string using row‑major order (left‑to‑right, top‑to‑bottom)."""
    mask = mask.astype(np.uint8)
    pixels = mask.ravel(order="C")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] = runs[1::2] - runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(rle_str, shape):
    """Decode RLE string (as produced by rle_encode) to binary mask with given shape."""
    if pd.isna(rle_str) or rle_str == "":
        return np.zeros(shape, dtype=np.uint8)
    s = list(map(int, rle_str.strip().split()))
    starts = s[0::2]
    lengths = s[1::2]
    starts = [x - 1 for x in starts]  # convert to zero‑based
    flat = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, length in zip(starts, lengths):
        flat[start : start + length] = 1
    return flat.reshape(shape, order="C")


def generate_mask(fragment_path, stack_count=3, threshold=None):
    """
    Create a binary mask from the average of the first *stack_count* slices.

    If ``threshold`` is ``None`` an adaptive Otsu threshold is used,
    otherwise the provided fixed threshold (found on the training set) is applied.
    """
    slice_paths = sorted(
        glob.glob(os.path.join(fragment_path, "surface_volume", "*.tif"))
    )
    if len(slice_paths) == 0:
        return None
    slice_paths = slice_paths[:stack_count]
    volume = []
    for sp in slice_paths:
        img = tifffile.imread(sp).astype(np.float32)
        if img.max() > 1:
            img = img / 65535.0
        volume.append(img)
    vol_stack = np.stack(volume, axis=0)  # (C, H, W)
    avg_img = vol_stack.mean(axis=0)  # (H, W)

    mn, mx = avg_img.min(), avg_img.max()
    if mx - mn > 1e-8:
        avg_norm = (avg_img - mn) / (mx - mn)
    else:
        avg_norm = avg_img

    if threshold is None:
        img_uint8 = (avg_norm * 255).astype(np.uint8)
        _, mask = cv2.threshold(img_uint8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        mask = (avg_norm > threshold).astype(np.uint8)

    mask = binary_opening(mask, structure=np.ones((3, 3))).astype(np.uint8)
    return mask


def f0_5_score(y_true, y_pred):
    """Compute the F0.5 score (precision‑weighted)."""
    tp = np.logical_and(y_true == 1, y_pred == 1).sum()
    fp = np.logical_and(y_true == 0, y_pred == 1).sum()
    fn = np.logical_and(y_true == 1, y_pred == 0).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    beta = 0.5
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)


def build_submission(test_root):
    """Generate a DataFrame with Id and RLE‑encoded predictions for every fragment."""
    fragment_dirs = [
        d for d in os.listdir(test_root) if os.path.isdir(os.path.join(test_root, d))
    ]
    rows = []
    for frag in sorted(fragment_dirs):
        frag_path = os.path.join(test_root, frag)
        mask = generate_mask(frag_path, stack_count=stack_count, threshold=threshold)
        rle = "" if mask is None else rle_encode(mask)
        rows.append({"Id": frag, "Predicted": rle})
    return pd.DataFrame(rows)




## === cell 1
train_root = "/kaggle/input/vesuvius-challenge-ink-detection/train"
fragment_dirs = [
    d for d in os.listdir(train_root) if os.path.isdir(os.path.join(train_root, d))
]

stack_count = 1


def evaluate_threshold(th):
    """Return mean F0.5 over the training fragments for a given threshold."""
    scores = []
    for frag in sorted(fragment_dirs):
        frag_path = os.path.join(train_root, frag)
        pred_mask = generate_mask(frag_path, stack_count=stack_count, threshold=th)
        if pred_mask is None:
            continue
        rle_path = os.path.join(frag_path, "inklabels_rle.csv")
        if not os.path.isfile(rle_path):
            continue
        df = pd.read_csv(rle_path)
        true_rle = df.at[0, "Predicted"]
        true_mask = rle_decode(true_rle, pred_mask.shape)
        scores.append(f0_5_score(true_mask, pred_mask))
    return np.mean(scores) if scores else 0.0


candidate_thresholds = np.linspace(0.0, 1.0, 21)

threshold_scores = []
for th in candidate_thresholds:
    sc = evaluate_threshold(th)
    threshold_scores.append((th, sc))
    print(f"Threshold {th:.2f} -> mean F0.5 {sc:.6f}")

best_th, best_score = min(threshold_scores, key=lambda ts: abs(ts[1] - target_score))

print(f"\nSelected threshold: {best_th:.2f} (mean F0.5 {best_score:.6f})")
threshold = best_th  # global variable used by build_submission




## === cell 2
test_root = "/kaggle/input/vesuvius-challenge-ink-detection/test"
submission_df = build_submission(test_root)

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}, rows: {len(submission_df)}")
