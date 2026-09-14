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

0.128913

# 6. Current score

0.16321

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.15335) has done: 'I fix the path enumeration bug that accidentally includes the nested `test/` directory, causing `.../test/test/mask.png` lookups and the FileNotFoundError. I also correct the submission column name to exactly `Id` (the competition requirement), since using `ID` would produce an invalid submission even if the code runs. To keep runtime stable and within limits, I remove plotting during submission generation (it’s not needed and can slow/timeout). The core logic (RLE over the provided mask) is preserved so it run end-to-end and write `submission.csv`.'
- What this solution (achieved 0.15356) has done: 'Your current score (0.15335) is better than the target (0.128913), so to move closer we should intentionally (but legitimately) reduce performance with the smallest, safest change. Since the core logic is “submit the mask as prediction”, we can slightly degrade the predicted mask before RLE by applying a tiny morphological erosion, which typically reduces recall (and thus F0.5) while keeping output valid. I implement a simple binary erosion using only NumPy (no new packages) and gate it behind a single constant so it’s easy to tune. Everything else (paths, Id/Predicted schema, RLE semantics, output file) stays unchanged.'
- What this solution (achieved 0.15377) has done: 'Your current score (0.15356) is higher than the target (0.128913), so we should make the smallest legitimate change that is likely to reduce the score toward the target band without breaking the submission. The safest lever in your current “mask-as-prediction” core logic is the amount of erosion: increasing it reduce predicted ink area, typically lowering recall and thus lowering F0.5. I only adjust `ERODE_ITERS` upward slightly (from 1 to 2) and keep everything else identical (paths, RLE, schema, and submission generation). This should move the score downward toward the target while preserving validity and runtime.'
- What this solution (achieved 0.15397) has done: 'Your current score (0.15377) is higher than the target (0.128913), so we should make a minimal, legitimate change that slightly *reduces* performance toward the target band rather than improving it. Since your core logic is “use the provided mask as the prediction,” the smallest stable lever is to increase the amount of binary erosion, which tends to reduce recall and lower F0.5. I only adjust `ERODE_ITERS` upward from 2 to 3 and keep everything else (paths, RLE function, submission schema/merge, and runtime behavior) identical. This preserves valid submission generation while nudging the score downward toward the target.'
- What this solution (achieved 0.15417) has done: 'Your current score (0.15397) is higher than the target (0.128913), so we should intentionally reduce performance slightly to move closer to the target band with minimal risk. The smallest stable lever in your existing “mask-as-prediction” core logic is to increase the erosion strength, which usually reduces recall and lowers F0.5. I only change `ERODE_ITERS` from 3 to 4 and keep everything else (paths, RLE encoding, submission schema/merge, and runtime) identical to preserve submission validity and stability. This should nudge the score downward without changing the overall approach.'
- What this solution (achieved 0.15437) has done: 'Your current score (0.15417) is above the target (0.128913), so to move closer we should slightly and legitimately reduce performance without changing the “mask-as-prediction then RLE” core logic. The smallest stable lever is the erosion strength, which reduces predicted ink area and typically lowers recall (and therefore F0.5). I only increase `ERODE_ITERS` by 1 (from 4 to 5) and keep all paths, RLE encoding, submission schema, and runtime behavior identical. This should nudge the score downward toward the target band while still producing a valid `submission.csv`.'
- What this solution (achieved 0.15457) has done: 'Your current score (0.15437) is above the target (0.128913), so to move closer we should make the smallest legitimate change that is likely to *reduce* F0.5 while keeping the same “mask-as-prediction → (optional erosion) → RLE” core logic intact. The simplest stable lever is increasing erosion slightly, which generally reduces recall and therefore lowers F0.5. I only bump `ERODE_ITERS` by +1 (5 → 6) and keep paths, RLE encoding, schema, and file writing identical to preserve submission validity and runtime. This should nudge the score downward toward the target tolerance band without altering the overall approach.'
- What this solution (achieved 0.15477) has done: 'Your current score (0.15457) is higher than the target (0.128913), so to move closer we should make a minimal, legitimate change that slightly reduces F0.5 rather than improving it. The smallest stable lever in your existing “mask-as-prediction → erosion → RLE” pipeline is to increase erosion strength a bit more, which typically reduces recall and lowers the score. I only bump `ERODE_ITERS` up by +1 and keep everything else (paths, RLE logic, submission schema/merge, and file writing) identical to preserve validity and runtime. This should nudge the score downward toward the target band.'
- What this solution (achieved 0.15497) has done: 'Your current score (0.15477) is higher than the target (0.128913), so we should make the smallest legitimate change that is likely to reduce F0.5 toward the target band without changing the overall “mask → (optional erosion) → RLE → submission.csv” pipeline. The safest lever is still the erosion strength; increasing it shrinks predicted positives, typically reducing recall and thus lowering F0.5. I only bump `ERODE_ITERS` up by +1 and keep everything else (paths, RLE logic, schema, and file writing) identical for stability. This preserves end-to-end execution and a valid `submission.csv`.'
- What this solution (achieved 0.15536) has done: 'Your current score (0.15497) is higher than the target (0.128913), so we should make a minimal, legitimate change that is likely to reduce performance toward the target band without changing the overall “mask → erosion → RLE → submission.csv” pipeline. The smallest stable lever remains erosion strength; increasing it shrinks predicted positives, typically reducing recall and thus lowering the F0.5 score. I only increase `ERODE_ITERS` by +2 (8 → 10) to make a slightly stronger, but still simple and fast, degradation. All paths, RLE encoding, submission schema, and file writing remain identical to preserve validity and runtime.'
- What this solution (achieved 0.15613) has done: 'Your current score (0.15536) is above the target (0.128913), so we should make the smallest legitimate change that is likely to reduce F0.5 toward the target band without changing the overall “mask → erosion → RLE → submission.csv” logic. The most stable lever in your existing pipeline is still erosion strength; increasing it shrinks predicted positives and typically reduces recall, lowering F0.5. I only bump `ERODE_ITERS` up slightly (10 → 14) and keep paths, RLE encoding, schema, and file writing identical to preserve validity and runtime.'
- What this solution (achieved 0.15762) has done: 'Your current score (0.15613) is above the target (0.128913), so we should legitimately reduce performance to move closer rather than improve it. Since your core logic is “use the provided mask as prediction,” the smallest stable lever is to increase erosion strength to shrink predicted positives, reducing recall and typically lowering F0.5. I only increase `ERODE_ITERS` (14 → 22) and keep everything else (paths, RLE encoding, schema, and file writing) identical for stability. This should move the score downward toward the target tolerance band while still producing a valid `submission.csv`.'
- What this solution (achieved 0.15994) has done: 'Your current score (0.15762) is above the target (0.128913), so to move closer we should legitimately reduce performance with the smallest, safest adjustment while keeping the same “mask → (optional erosion) → RLE → submission.csv” pipeline. The most stable lever you already introduced is erosion strength; increasing it shrinks predicted positives, typically reducing recall and lowering F0.5. I only increase `ERODE_ITERS` moderately (22 → 35) and keep all paths, RLE encoding, schema, and file writing identical to preserve validity and runtime. This should nudge the score downward toward the target band without changing the core logic.'
- What this solution (achieved 0.16321) has done: 'Your current score (0.15994) is higher than the target (0.128913), so we should make a minimal, legitimate change that is likely to reduce performance toward the target band without changing the overall “mask → erosion → RLE → submission.csv” logic. The safest lever you already use is erosion strength, which shrinks predicted positives and typically reduces recall and thus lowers F0.5. I only increase `ERODE_ITERS` moderately (35 → 55) and keep paths, RLE encoding, schema, and file writing identical to preserve validity and runtime. This should move the score downward toward the target while still producing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import gc
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import PIL.Image as Image
import matplotlib.pyplot as plt

import glob
import torch.nn as nn
from tqdm import tqdm
import torch.utils.data as data

BUFFER = 30
WINDOW = 1
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_path = "/kaggle/input/vesuvius-challenge-ink-detection/test"

ERODE_ITERS = 55  # was 35; set to 0 to revert to original behavior




## === cell 1
def rle(img):
    pixels = img.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return " ".join(str(x) for x in runs)




## === cell 2
def binary_erosion_np(mask_bool: np.ndarray, iters: int = 1) -> np.ndarray:
    m = mask_bool.astype(bool, copy=False)
    if iters <= 0:
        return m
    for _ in range(iters):
        p = np.pad(m, ((1, 1), (1, 1)), mode="constant", constant_values=False)
        m = (
            p[0:-2, 0:-2]
            & p[0:-2, 1:-1]
            & p[0:-2, 2:]
            & p[1:-1, 0:-2]
            & p[1:-1, 1:-1]
            & p[1:-1, 2:]
            & p[2:, 0:-2]
            & p[2:, 1:-1]
            & p[2:, 2:]
        )
    return m




## === cell 3
base_path = Path("/kaggle/input/vesuvius-challenge-ink-detection/test")

test_fragments = sorted(
    [p for p in base_path.iterdir() if p.is_dir() and (p / "mask.png").exists()],
    key=lambda p: p.name,
)

submission_dict = {"Id": [], "Predicted": []}




## === cell 4
for frag_path in test_fragments:
    fragment_name = frag_path.name
    prefix = str(frag_path) + "/"
    print(prefix)

    mask = np.array(Image.open(prefix + "mask.png").convert("1"))
    mask_bool = mask.astype(bool)
    mask_bool = binary_erosion_np(mask_bool, iters=ERODE_ITERS)
    rle_output = rle(mask_bool.astype(np.uint8))

    submission_dict["Id"].append(fragment_name)
    submission_dict["Predicted"].append(rle_output)

submission = pd.DataFrame(data=submission_dict)

sample_path = Path(
    "/kaggle/input/vesuvius-challenge-ink-detection/sample_submission.csv"
)
if sample_path.exists():
    sample = pd.read_csv(sample_path)
    submission = sample[["Id"]].merge(submission, on="Id", how="left")
    submission["Predicted"] = submission["Predicted"].fillna("")

submission.to_csv("submission.csv", index=False)
print(submission)
print("Wrote submission.csv")
