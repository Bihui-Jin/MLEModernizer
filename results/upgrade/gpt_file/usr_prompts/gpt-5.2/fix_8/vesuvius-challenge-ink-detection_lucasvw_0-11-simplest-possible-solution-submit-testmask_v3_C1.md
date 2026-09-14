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

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.15335) has done: 'I fix the test fragment discovery so it only includes real fragment folders (and not the stray nested `test/` directory), which is the root cause of the missing `mask.png` error. Then I make the prediction-building loop robust by skipping any folder without `mask.png`, ensuring `pred_images` and `Id` stay aligned and the DataFrame builds correctly. Finally, I ensure we always write a valid `submission.csv` with the exact required columns in `/kaggle/working/`. These changes are score-neutral (the “predict mask as ink” baseline remains unchanged) but make the notebook run end-to-end and yield a valid submission file.'

# 9. Code solution

## === cell 0
from fastai.vision.all import *
import numpy as np
import pandas as pd
from PIL import Image



## === cell 1
DATA_ROOT = Path("/kaggle/input/vesuvius-challenge-ink-detection")
test_path = DATA_ROOT / "test"
sample_path = DATA_ROOT / "sample_submission.csv"

assert test_path.exists(), f"Missing test path: {test_path}"
assert sample_path.exists(), f"Missing sample submission: {sample_path}"

sample_df = pd.read_csv(sample_path)
sample_ids = sample_df["Id"].astype(str).tolist()

print("Sample submission Ids:", sample_ids)




## === cell 2
def is_valid_fragment_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    if p.name.startswith("."):
        return False
    if not (p / "mask.png").exists():
        return False
    sv = p / "surface_volume"
    if not sv.exists():
        return False
    if len(list(sv.glob("*.tif"))) == 0:
        return False
    return True


test_fragments = sorted(
    [p for p in test_path.iterdir() if is_valid_fragment_dir(p)], key=lambda x: x.name
)
test_fragment_ids = [p.name for p in test_fragments]

print("Discovered test fragments:", test_fragment_ids)



## === cell 3
if len(test_fragments) == 0:
    print(
        f"Warning: No valid fragment folders found under: {test_path}. Falling back to sample_submission.csv Ids."
    )
else:
    _ = Image.open(test_fragments[0] / "mask.png")
    print("First fragment mask opened OK:", test_fragments[0].name)




## === cell 4
def rle_from_binary_mask(mask_2d: np.ndarray) -> str:
    """
    Kaggle Vesuvius expects RLE over a flattened mask (row-major), 1-indexed starts.
    Correctly handles runs touching the first/last pixel.
    """
    pixels = mask_2d.astype(np.uint8).reshape(-1)
    pixels = np.concatenate([[0], pixels, [0]])
    changes = np.where(pixels[1:] != pixels[:-1])[0] + 1
    if changes.size == 0:
        return ""
    runs = changes.reshape(-1, 2)
    starts = runs[:, 0]
    ends = runs[:, 1]
    lengths = ends - starts
    out = []
    for s, l in zip(starts, lengths):
        out.append(str(int(s)))
        out.append(str(int(l)))
    return " ".join(out)




## === cell 5
def binary_erosion_numpy(mask01: np.ndarray, k: int = 9, iters: int = 1) -> np.ndarray:
    """
    Simple binary erosion using sliding-window min implemented with numpy.
    mask01: uint8 array with values in {0,1}.
    k: odd kernel size.
    iters: number of erosion iterations.
    """
    assert k % 2 == 1, "k must be odd"
    m = mask01.astype(np.uint8)
    pad = k // 2
    for _ in range(iters):
        padded = np.pad(m, ((pad, pad), (pad, pad)), mode="constant", constant_values=0)
        out = np.ones_like(m, dtype=np.uint8)
        for dy in range(k):
            ys = dy
            ye = dy + m.shape[0]
            for dx in range(k):
                xs = dx
                xe = dx + m.shape[1]
                out &= padded[ys:ye, xs:xe].astype(np.uint8)
                if out.sum() == 0:  # speed-up; semantics unchanged
                    break
            if out.sum() == 0:
                break
        m = out
    return m


pred_by_id = {}

MASK_BIN_THRESHOLD = 200  # preserves original interpretation of 0/255 mask.png
ERODE_KERNEL = 31  # conservative shrink to boost precision (F0.5)
ERODE_ITERS = 1

for frag in test_fragments:
    mask_path = frag / "mask.png"
    if not mask_path.exists():
        continue
    m = np.array(Image.open(mask_path))
    base_mask = (m > MASK_BIN_THRESHOLD).astype(np.uint8)  # 1 where data exists
    pred = binary_erosion_numpy(base_mask, k=ERODE_KERNEL, iters=ERODE_ITERS).astype(
        np.uint8
    )
    pred_by_id[frag.name] = rle_from_binary_mask(pred)

print("Built predictions for ids:", list(pred_by_id.keys()))



## === cell 6
submission_ids = test_fragment_ids if len(test_fragment_ids) > 0 else sample_ids

if (
    (len(test_fragment_ids) > 0)
    and (len(sample_ids) == 1)
    and (sample_ids[0] not in test_fragment_ids)
):
    print(
        "Note: sample_submission.csv appears to be a placeholder; using discovered test fragment IDs instead."
    )
    submission_ids = test_fragment_ids

predicted_col = [pred_by_id.get(_id, "") for _id in submission_ids]
df = pd.DataFrame({"Id": submission_ids, "Predicted": predicted_col})

print("Submission shape:", df.shape)
print(df.head())



## === cell 7
out_path = Path("/kaggle/working/submission.csv")
df.to_csv(out_path, index=False)

print(f"Wrote: {out_path}  rows={len(df)}")
print(df.head(10))
print("Any empty predictions:", int((df["Predicted"] == "").sum()))
print("Ids used:", submission_ids)
