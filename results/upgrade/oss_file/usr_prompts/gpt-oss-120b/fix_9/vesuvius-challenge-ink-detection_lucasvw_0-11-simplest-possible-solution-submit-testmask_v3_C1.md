# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

fastai==2.8.5
geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

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

0.128913

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.15335) has done: 'I add a quick evaluation on the training data to pick an IR threshold that gives a better F0.5 score, then use that threshold for the test predictions. The change keeps the original rule‑based logic but makes the threshold data‑driven, which should move the score toward the target while preserving the core approach and still producing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
from fastai.vision.all import *
import numpy as np, pandas as pd
from collections import defaultdict
from pathlib import Path
from PIL import Image



## === cell 1
base_path = Path("/kaggle/input/vesuvius-challenge-ink-detection")

train_path = base_path / "train"
test_path = base_path / "test"

if (train_path / "train").is_dir():
    train_path = train_path / "train"
if (test_path / "test").is_dir():
    test_path = test_path / "test"




## === cell 2
def encode_rle(binary_arr):
    """Standard run‑length encoding for a 2‑D binary mask."""
    flat = binary_arr.ravel(order="C")
    flat = np.concatenate([[0], flat, [0]])
    runs = np.where(flat[1:] != flat[:-1])[0] + 1
    runs[1::2] = runs[1::2] - runs[::2]
    return " ".join(str(x) for x in runs) if runs.size else ""


def decode_rle(rle_str, shape):
    """Decode a RLE string back to a binary mask of given shape."""
    if not rle_str:
        return np.zeros(shape, dtype=np.uint8)
    s = list(map(int, rle_str.split()))
    starts, lengths = s[0::2], s[1::2]
    starts = np.array(starts) - 1  # zero‑based
    ends = starts + lengths
    flat = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        flat[lo:hi] = 1
    return flat.reshape(shape, order="C")


def fbeta_score(y_true, y_pred, beta=0.5, eps=1e-7):
    """Fβ score where β < 1 weights precision higher (β=0.5)."""
    tp = (y_true * y_pred).sum()
    fp = ((1 - y_true) * y_pred).sum()
    fn = (y_true * (1 - y_pred)).sum()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    beta2 = beta**2
    return (1 + beta2) * precision * recall / (beta2 * precision + recall + eps)




## === cell 3
mask_threshold = 0.7  # fallback behaviour
candidate_thresholds = list(range(10, 51, 5))  # 10,15,…,50
best_thr = 20  # fallback if something goes wrong
best_score = -1.0

train_fragments = [p for p in train_path.iterdir() if p.is_dir()]
train_fragments.sort()

for thr in candidate_thresholds:
    scores = []
    for frag in train_fragments:
        mask_path = frag / "mask.png"
        mask_arr = np.array(Image.open(mask_path).convert("L"))
        mask_bin = (mask_arr > 0).astype(np.uint8)

        ir_path = frag / "ir.png"
        if ir_path.exists():
            ir_arr = np.array(Image.open(ir_path).convert("L"))
            pred = ((ir_arr < thr) & (mask_bin == 1)).astype(np.uint8)
        else:
            pred = (mask_arr > mask_threshold).astype(np.uint8)

        ink_rle_path = frag / "inklabels_rle.csv"
        gt_rle = pd.read_csv(ink_rle_path).loc[0, "Predicted"]
        gt_mask = decode_rle(gt_rle, mask_bin.shape)

        scores.append(fbeta_score(gt_mask, pred, beta=0.5))
    avg_score = np.mean(scores)
    if avg_score > best_score:
        best_score = avg_score
        best_thr = thr

IR_THRESHOLD = best_thr
print(f"Selected IR_THRESHOLD = {IR_THRESHOLD} with estimated F0.5 ≈ {best_score:.5f}")



## === cell 4
submission = defaultdict(list)

test_fragments = [p for p in test_path.iterdir() if p.is_dir()]
test_fragments.sort()

if not test_fragments:
    raise RuntimeError("No test fragments found – check the dataset path.")

for fragment in test_fragments:
    mask_path = fragment / "mask.png"
    mask_arr = np.array(Image.open(mask_path).convert("L"))
    mask_bin = (mask_arr > 0).astype(np.uint8)

    ir_path = fragment / "ir.png"
    if ir_path.exists():
        ir_arr = np.array(Image.open(ir_path).convert("L"))
        ink_pred = ((ir_arr < IR_THRESHOLD) & (mask_bin == 1)).astype(np.uint8)
    else:
        ink_pred = (mask_arr > mask_threshold).astype(np.uint8)

    submission["Id"].append(fragment.name)
    submission["Predicted"].append(encode_rle(ink_pred))



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/249266960.py in <cell line: 0>()
      5 
      6 if not test_fragments:
----> 7     raise RuntimeError("No test fragments found – check the dataset path.")
      8 
      9 for fragment in test_fragments:

RuntimeError: No test fragments found – check the dataset path.

## === cell 5
df = pd.DataFrame(submission)
df.to_csv("/kaggle/working/submission.csv", index=False)
print("Submission written to /kaggle/working/submission.csv")

## --- ERROR in outputing the csv:
Invalid submission: Expected 1 rows in the submission DataFrame, but got 0 rows.
