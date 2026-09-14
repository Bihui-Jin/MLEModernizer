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
seaborn==0.12.2
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

0.1290207398594625

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.09649) has done: 'I replace the placeholder “1 1” predictions with a simple heuristic that re‑uses the real RLE mask from the first training fragment. This keeps the core pipeline unchanged, guarantees a valid submission.csv, and should raise the F0.5 score toward the target without altering any model logic.'
- What this solution (achieved 0.09649) has done: 'I keep the existing pipeline but improve the heuristic used for the predictions.  
Instead of assigning the same training RLE to every test fragment, I look at each fragment’s mask image: if the mask contains very little data (below a small threshold) I output an empty prediction, otherwise I fall back to the training RLE. This simple rule should raise precision (the F0.5 metric favours precision) and move the score closer to the target while preserving the original model‑free workflow.'
- What this solution (achieved 0.15335) has done: 'Implemented a more precise prediction heuristic: instead of blindly using the training RLE for every test fragment, the script now encodes the actual mask image to run‑length format when the mask’s foreground density is sufficiently high (≥ 0.5). Low‑density masks receive an empty prediction, preserving precision which the F0.5 metric rewards. A small `rle_encode` utility handles the conversion, and missing masks still fall back to the default training RLE to guarantee a valid submission.'
- What this solution (achieved 0.15335) has done: 'I added the missing imports, defined the required paths, handled the optional `monai` import, and fixed variable names so the script can run end‑to‑end. The heuristic now encodes a mask’s RLE when its foreground density ≥ 0.5 (as in the previous improvement that already surpassed the target score) and falls back to an empty prediction otherwise, guaranteeing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

try:
    import monai
except Exception:
    monai = None
try:
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader
except Exception:
    pl = None
    DataLoader = None




## === cell 1
KAGGLE_DIR = Path("/") / "kaggle"
INPUT_DIR = KAGGLE_DIR / "input"
COMPETITION_DATA_DIR = INPUT_DIR / "vesuvius-challenge-ink-detection"

DOWNSAMPLING = 0.5




## === cell 2
def create_df_from_mask_paths(mask_paths, train=True):
    df = pd.DataFrame({"mask_png": mask_paths})
    df["mask_png"] = df["mask_png"].astype(str)
    df["stage"] = df["mask_png"].str.split("/").str[-3]
    df["fragmet_id"] = df["mask_png"].str.split("/").str[-2]

    df["mask_npy"] = df["mask_png"].str.replace(
        "train", f"train_{DOWNSAMPLING}", regex=False
    )
    df["mask_npy"] = df["mask_npy"].str.replace("input", "working", regex=False)
    df["mask_npy"] = df["mask_npy"].str.replace(".png", ".npy", regex=False)

    if train:
        df["label_png"] = df["mask_png"].str.replace("mask", "inklabels", regex=False)
        df["label_npy"] = df["mask_npy"].str.replace("mask", "inklabels", regex=False)

    df["volumes_dir"] = df["mask_png"].str.replace(
        "mask.png", "surface_volume", regex=False
    )
    df["volume_npy"] = df["mask_npy"].str.replace("mask", "volume", regex=False)
    return df


test_mask_paths = sorted(COMPETITION_DATA_DIR.glob("test/*/mask.png"))
test_df = create_df_from_mask_paths(test_mask_paths, train=False)
test_df.to_csv("test.csv", index=False)




## === cell 3
submission_df = pd.read_csv(COMPETITION_DATA_DIR / "sample_submission.csv")


def rle_encode(mask: np.ndarray) -> str:
    """
    Encode a binary mask (numpy bool/uint8) into run‑length encoding
    using 1‑based indexing and column‑major order as required
    by the competition.
    """
    flat = mask.ravel(order="C")  # row‑major (left‑to‑right, top‑to‑bottom)
    padded = np.pad(flat, (1, 1), mode="constant", constant_values=0)
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    lengths = ends - starts
    rle = []
    for s, l in zip(starts, lengths):
        rle.extend([s, l])
    return " ".join(map(str, rle))


fallback_rle_path = COMPETITION_DATA_DIR / "train" / "1" / "inklabels_rle.csv"
try:
    fallback_rle = pd.read_csv(fallback_rle_path)["Predicted"].iloc[0]
except Exception:
    fallback_rle = ""  # safest fallback if the file cannot be read

mask_path_map = dict(zip(test_df["fragmet_id"], test_df["mask_png"]))

DENSITY_THRESHOLD = 0.2

predictions = []
for fragment_id in submission_df["Id"]:
    mask_path = mask_path_map.get(fragment_id)
    if mask_path and Path(mask_path).exists():
        try:
            mask_img = Image.open(mask_path).convert("L")
            mask_arr = np.array(mask_img) > 0  # binary mask
            density = mask_arr.mean()
            if density >= DENSITY_THRESHOLD:
                predictions.append(rle_encode(mask_arr))
                continue
            else:
                predictions.append(fallback_rle)
                continue
        except Exception:
            pass
    predictions.append(fallback_rle)

submission_df["Predicted"] = predictions
submission_df.to_csv("submission.csv", index=False)
print("Submission file 'submission.csv' created with", len(submission_df), "rows.")
