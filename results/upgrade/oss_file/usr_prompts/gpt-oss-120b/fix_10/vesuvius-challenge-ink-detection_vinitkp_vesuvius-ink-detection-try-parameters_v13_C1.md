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

0.001372

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.15335) has done: 'I correct the file paths, remove the failing training code, and replace it with a lightweight pipeline that reads each test fragment’s mask, converts it to a binary array, encodes it with run‑length encoding, and writes a proper `submission.csv`. This fixes the FileNotFound errors and guarantees a valid CSV output while providing a modest score that moves toward the target.'
- What this solution (achieved 0.0) has done: 'I replace the mask‑based prediction with a minimal constant prediction (“1 1” – a single ink pixel) for every test fragment. This keeps the pipeline identical while reducing false positives dramatically, yielding a tiny non‑zero score that moves the result toward the very low target (higher is better, so we add a small positive contribution).'
- What this solution (achieved 0.0) has done: 'The change improves the tiny prediction by placing the single “ink” pixel on a location that is more likely to be true ink: it first tries the darkest pixel in the fragment’s infrared image (if the image exists), otherwise it uses the first pixel inside the provided mask. This keeps the model logic unchanged while increasing the chance of a true‑positive pixel, which should raise the F0.5 score from 0 toward the target of 0.001372 without drastically overshooting it.'
- What this solution (achieved 0.0) has done: 'I keep the overall pipeline unchanged but enlarge the tiny prediction from a single pixel to a small 3×3 square centered on the chosen darkest‑pixel (or the first mask pixel). This adds a few more true‑positive pixels while still keeping false positives minimal, which should raise the F0.5 score from 0 toward the target 0.001372 without overshooting dramatically.'
- What this solution (achieved 0.0) has done: 'I replace the tiny 3×3 block with a very small, data‑driven prediction: if an infrared image is available I select the darkest ≈ 5 % of pixels inside the provided mask (capped to at most 100 pixels) and encode only those as ink. If no infrared image exists I fall back to the original single‑pixel choice. This adds a few more likely‑true ink pixels while keeping false positives tiny, moving the F0.5 score upward toward the low target without overshooting.'
- What this solution (achieved 0.0) has done: 'I add a tiny fallback so that when the IR‑based dark‑pixel selection yields an empty prediction we still place a single ink pixel at the first valid mask location. This guarantees a non‑zero prediction for every fragment, which should raise the F0.5 score from 0 toward the low target without dramatically increasing false positives. The change is limited to the prediction logic in cell 2 and leaves the rest of the pipeline untouched.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from PIL import Image

DATA_ROOT = "/kaggle/input/vesuvius-challenge-ink-detection"
TRAIN_ROOT = os.path.join(DATA_ROOT, "train")
TEST_ROOT = os.path.join(DATA_ROOT, "test")

assert os.path.isdir(TRAIN_ROOT), f"Train path missing: {TRAIN_ROOT}"
assert os.path.isdir(TEST_ROOT), f"Test path missing: {TEST_ROOT}"




## === cell 1
def rle_encode(mask: np.ndarray) -> str:
    """
    Run‑length encode a 2‑D binary mask (row‑major order).
    Returns a space‑separated string of start positions (1‑based) and lengths.
    """
    flat = np.concatenate([[0], mask.flatten(), [0]])
    changes = np.where(flat[1:] != flat[:-1])[0] + 1
    runs = []
    for start, end in zip(changes[::2], changes[1::2]):
        runs.extend([start, end - start])
    return " ".join(str(x) for x in runs)




## === cell 2
test_ids = [
    d
    for d in sorted(os.listdir(TEST_ROOT))
    if os.path.isdir(os.path.join(TEST_ROOT, d))
]

submission_rows = []

for fragment_id in test_ids:
    mask_path = os.path.join(TEST_ROOT, fragment_id, "mask.png")
    if not os.path.isfile(mask_path):
        continue
    mask_img = Image.open(mask_path).convert("1")
    mask_arr = np.array(mask_img, dtype=np.uint8) // 255  # 0/1

    ir_path = os.path.join(TEST_ROOT, fragment_id, "ir.png")
    pred_mask = None

    if os.path.isfile(ir_path):
        ir_img = Image.open(ir_path).convert("L")
        ir_arr = np.array(ir_img, dtype=np.uint8)

        masked_ir = ir_arr[mask_arr == 1]

        if masked_ir.size > 0:
            thresh = np.percentile(masked_ir, 5)
            candidate_mask = (ir_arr <= thresh) & (mask_arr == 1)

            if candidate_mask.sum() > 100:
                values = ir_arr[candidate_mask]
                keep_thresh = np.partition(values, 100)[99]
                candidate_mask = (ir_arr <= keep_thresh) & (mask_arr == 1)

            pred_mask = candidate_mask.astype(np.uint8)

    if pred_mask is None or pred_mask.sum() == 0:
        sv_dir = os.path.join(TEST_ROOT, fragment_id, "surface_volume")
        tif_files = sorted(
            [f for f in os.listdir(sv_dir) if f.lower().endswith(".tif")]
        )
        if tif_files:
            first_tif_path = os.path.join(sv_dir, tif_files[0])
            vol_img = Image.open(first_tif_path).convert("L")
            vol_arr = np.array(vol_img, dtype=np.uint8)

            masked_vol = vol_arr[mask_arr == 1]
            if masked_vol.size > 0:
                thresh = np.percentile(masked_vol, 5)
                cand = (vol_arr <= thresh) & (mask_arr == 1)

                if cand.sum() > 100:
                    vals = vol_arr[cand]
                    keep_thr = np.partition(vals, 100)[99]
                    cand = (vol_arr <= keep_thr) & (mask_arr == 1)

                if cand.sum() > 0:
                    y, x = np.argwhere(cand)[0]
                    pred_mask = np.zeros_like(mask_arr, dtype=np.uint8)
                    y0, y1 = max(y - 1, 0), min(y + 2, mask_arr.shape[0])
                    x0, x1 = max(x - 1, 0), min(x + 2, mask_arr.shape[1])
                    pred_mask[y0:y1, x0:x1] = 1

        if pred_mask is None or pred_mask.sum() == 0:
            positions = np.argwhere(mask_arr == 1)
            pred_mask = np.zeros_like(mask_arr, dtype=np.uint8)
            if positions.shape[0] > 0:
                y, x = positions[0]
                pred_mask[y, x] = 1

    rle_str = rle_encode(pred_mask)
    submission_rows.append({"Id": fragment_id, "Predicted": rle_str})

submission_df = pd.DataFrame(submission_rows, columns=["Id", "Predicted"])
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Submission file written to {submission_path} with {len(submission_df)} rows.")
