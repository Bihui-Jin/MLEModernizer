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

0.0001022775828751

# 6. Current score

0.04838

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the unavailable `monai` dependency, define the needed paths and constants directly, and simplify the pipeline to generate a valid `submission.csv` using zero‑mask predictions. The fix ensures all variables (`COMPETITION_DATA_DIR`, `DOWNSAMPLING`, etc.) are defined, avoids the missing `load_label_npy` function, and replaces the RLE routine with a safe fallback that produces a placeholder “1 1” for each test image, guaranteeing a correctly formatted CSV.'
- What this solution (achieved 0.10877) has done: 'I replace the dummy‑prediction function with a very light “average‑mask” predictor that resizes the training ink‑label masks to each test fragment size and uses the mean mask as a probability map. This keeps the pipeline unchanged, still writes a valid `submission.csv`, and adds just enough signal to move the F0.5 score above zero and closer to the tiny target.'
- What this solution (achieved 0.04838) has done: 'I lower the prediction threshold to an extreme value and change the RLE helper so that when a mask contains no foreground it returns an empty string instead of the placeholder “1 1”. This make the submission predict almost nothing, driving the F0.5 score down from the current 0.10877 toward the tiny target value while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.04899) has done: 'I lower the prediction threshold from 1.0 to 0.9 so that only pixels that appear ink in **both** training masks after resizing are kept. This adds a tiny amount of foreground to the otherwise empty predictions, moving the F0.5 score upward toward the tiny target value without drastically changing the pipeline.'
- What this solution (achieved 0.04873) has done: 'I adjust the prediction threshold so that only the most confident pixels (where all training masks agree) are kept. This yields a very small amount of foreground, moving the F0.5 score from 0 toward the tiny target 0.000102 without altering the core pipeline.'
- What this solution (achieved 0.04873) has done: 'I keep the overall pipeline unchanged but lower the prediction threshold from an impossible 1.5 to 0.95 so that a few high‑confidence pixels from the averaged training masks are kept. This introduces a small amount of foreground in the submission, moving the F0.5 score from 0 up toward the tiny target 0.000102 without drastically changing the model or data handling.'
- What this solution (achieved 0.04845) has done: 'We lower the binary‑mask threshold so that almost no pixels are kept, drastically reducing the amount of predicted foreground and therefore moving the F0.5 score down toward the tiny target value. By increasing the threshold from 0.95 to 0.99 the average‑mask predictions almost always be below the cutoff, yielding mostly empty RLE strings and a score much closer to 0.000102.'
- What this solution (achieved 0.04838) has done: 'The update dynamically sets the RLE threshold to just below the highest predicted probability across all test fragments, ensuring only a tiny amount of foreground pixels are encoded. This produces a very small, non‑zero F0.5 score that moves the result toward the target 0.000102 without altering the core averaging logic.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm

KAGGLE_DIR = Path("/") / "kaggle"
INPUT_DIR = KAGGLE_DIR / "input"
COMPETITION_DATA_DIR = INPUT_DIR / "vesuvius-challenge-ink-detection"

DOWNSAMPLING = 1.0  # no down‑sampling for dummy run
Z_START = 27
Z_DIM = 16




## === cell 1
def create_df_from_mask_paths(mask_paths, train=True):
    df = pd.DataFrame({"mask_png": mask_paths})
    df["mask_png"] = df["mask_png"].astype(str)
    df["stage"] = df["mask_png"].str.split("/").str[-3]
    df["fragmet_id"] = df["mask_png"].str.split("/").str[-2]
    return df


test_mask_paths = sorted(COMPETITION_DATA_DIR.glob("test/*/mask.png"))
test_df = create_df_from_mask_paths(test_mask_paths, train=False)




## === cell 2
def average_mask_predictions(df):
    """
    Load all training ink label masks, then for each test mask size
    resize every training mask to that size, average them, and use the
    mean image as a probability map.
    """
    train_mask_paths = sorted(COMPETITION_DATA_DIR.glob("train/*/inklabels.png"))
    train_masks = []
    for p in train_mask_paths:
        img = Image.open(p).convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0  # binary 0/1
        train_masks.append(arr)

    preds = []
    for mask_path in tqdm(df["mask_png"], desc="Generating average predictions"):
        mask_img = Image.open(mask_path)
        w, h = mask_img.size

        resized_sum = np.zeros((h, w), dtype=np.float32)
        for tm in train_masks:
            resized = (
                np.array(
                    Image.fromarray((tm * 255).astype(np.uint8)).resize(
                        (w, h), resample=Image.BILINEAR
                    )
                ).astype(np.float32)
                / 255.0
            )
            resized_sum += resized

        avg_pred = resized_sum / max(1, len(train_masks))
        preds.append(avg_pred)

    return preds


predictions = average_mask_predictions(test_df)




## === cell 3
def fast_rle(prediction_resized, threshold):
    """
    Convert a binary mask to run‑length encoding.
    If no foreground pixels are present, return an empty string.
    """
    binary = (prediction_resized > threshold).astype(np.uint8)
    flat = binary.flatten()
    if flat.sum() == 0:
        return ""  # no RLE entries for empty mask
    starts = np.where((flat[:-1] == 0) & (flat[1:] == 1))[0] + 2
    ends = np.where((flat[:-1] == 1) & (flat[1:] == 0))[0] + 2
    if flat[0] == 1:
        starts = np.insert(starts, 0, 1)
    if flat[-1] == 1:
        ends = np.append(ends, len(flat) + 1)
    lengths = ends - starts
    rle_array = np.stack([starts, lengths], axis=1).flatten()
    return " ".join(map(str, rle_array.astype(int)))


submission_df = pd.read_csv(COMPETITION_DATA_DIR / "sample_submission.csv")

max_pred = max(p.max() for p in predictions)
threshold = max_pred - 1e-4  # tiny epsilon to keep only the highest‑probability pixels
if threshold < 0:
    threshold = 0.0

predictions_rle = []

for mask_png_path, pred in zip(test_df["mask_png"].values, predictions):
    mask_img = Image.open(mask_png_path)
    pred_uint8 = (pred * 255).astype(np.uint8)
    prediction_resized = (
        np.array(
            Image.fromarray(pred_uint8).resize(mask_img.size, resample=Image.NEAREST)
        )
        / 255.0
    )
    rle = fast_rle(prediction_resized, threshold)
    predictions_rle.append(rle)

while len(predictions_rle) < len(submission_df):
    predictions_rle.append("")

submission_df["Predicted"] = predictions_rle
submission_df.to_csv("submission.csv", index=False)
print("Submission file written to submission.csv")
