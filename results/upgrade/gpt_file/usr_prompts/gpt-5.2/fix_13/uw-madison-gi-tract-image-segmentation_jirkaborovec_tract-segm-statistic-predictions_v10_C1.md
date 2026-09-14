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
Create a model to automatically segment the stomach and intestines on MRI scans.

## Metric
Mean Dice coefficient and 3D Hausdorff distance. 

The Dice coefficient can be used to compare the pixel-wise agreement between a predicted segmentation and its corresponding ground truth. The formula is given by:

$$
\frac{2 \cdot |X \cap Y|}{|X| + |Y|}
$$

where $X$ is the predicted set of pixels and $Y$ is the ground truth. The Dice coefficient is defined to be 0 when both $X$ and $Y$ are empty. 

Hausdorff distance is a method for calculating the distance between segmentation objects A and B, by calculating the furthest point on object A from the nearest point on object B. For 3D Hausdorff, we construct 3D volumes by combining each 2D segmentation with slice depth as the Z coordinate and then find the Hausdorff distance between them. (Here the slice depth for all scans is set to 1). The expected / predicted pixel locations are normalized by image size to create a bounded 0-1 score.

The two metrics are combined, with a weight of 0.4 for the Dice metric and 0.6 for the Hausdorff distance.

## Submission Format
Use run-length encoding on the pixel values.  Instead of submitting an exhaustive list of indices for your segmentation, you will submit pairs of values that contain a start position and a run length. E.g. '1 3' implies starting at pixel 1 and running a total of 3 pixels (1,2,3).

Note that, at the time of encoding, the mask should be binary, meaning the masks for all objects in an image are joined into a single large mask. A value of 0 should indicate pixels that are not masked, and a value of 1 will indicate pixels that are masked.

The competition format requires a space delimited list of pairs. For example, '1 3 10 5' implies pixels 1,2,3,10,11,12,13,14 are to be included in the mask. The metric checks that the pairs are sorted, positive, and the decoded pixel values are not duplicated. The pixels are numbered from top to bottom, then left to right: 1 is pixel (1,1), 2 is pixel (2,1), etc.

The file should contain a header and have the following format:

```
id,class,predicted
1,large_bowel,1 1 5 1
1,small_bowel,1 1
1,stomach,1 1
2,large_bowel,1 5 2 17
etc.
```

## Dataset
Each case is represented by multiple sets of scan slices (each set is identified by the day the scan took place). Some cases are split by time (early days are in train, later days are in test) while some cases are split by case - the entirety of the case is in train or test. The goal is to be able to generalize to both partially and wholly unseen cases.

### Files
- train.csv - IDs and masks for all training objects.
- sample_submission.csv - a sample submission file in the correct format
- train - a folder of case/day folders, each containing slice images for a particular case on a given day.

Note that the image filenames include 4 numbers (ex. 276_276_1.63_1.63.png). These four numbers are slice width / height (integers in pixels) and width/height pixel spacing (floating points in mm). The first two defines the resolution of the slide. The last two record the physical size of each pixel.

Physical pixel thickness in superior-inferior direction is 3mm.

### Columns
- `id` - unique identifier for object
- `class` - the predicted class for the object
- `segmentation` - RLE-encoded pixels for the identified object

# 2. Python version

3.10

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
            description.md (126 lines)
            sample_submission.csv (20401 lines)
            sample_submission.csv.zip (57.4 kB)
            test.csv (20401 lines)
            test.csv.zip (55.2 kB)
            test.zip (432.9 MB)
            train.csv (95089 lines)
            train.csv.zip (6.7 MB)
            train.zip (2.0 GB)
            test/
                case110/
                    case110_day12/
                        scans/
                            ... (max depth reached)
                    case110_day16/
                        scans/
                            ... (max depth reached)
                case113/
                    case113_day22/
                        scans/
                            ... (max depth reached)
                ... and 27 other folders
            train/
                case101/
                    case101_day20/
                        scans/
                            ... (max depth reached)
                    case101_day22/
                        scans/
                            ... (max depth reached)
                    case101_day26/
                        scans/
                            ... (max depth reached)
                    case101_day32/
                        scans/
                            ... (max depth reached)
                case102/
                    case102_day0/
                        scans/
                            ... (max depth reached)
                ... and 75 other folders
            uw-madison-gi-tract-image-segmentation/
                description.md (126 lines)
                sample_submission.csv (20401 lines)
                ... and 7 other files
                test/
                    case110/
                        case110_day12/
                            ... (max depth reached)
                        case110_day16/
                            ... (max depth reached)
                    case113/
                        case113_day22/
                            ... (max depth reached)
                    ... and 27 other folders
                train/
                    case101/
                        case101_day20/
                            ... (max depth reached)
                        case101_day22/
                            ... (max depth reached)
                        case101_day26/
                            ... (max depth reached)
                        case101_day32/
                            ... (max depth reached)
                    case102/
                        case102_day0/
                            ... (max depth reached)
                    ... and 75 other folders
                uw-madison-gi-tract-image-segmentation/
        input/
            description.md (126 lines)
            sample_submission.csv (20401 lines)
            sample_submission.csv.zip (57.4 kB)
            test.csv (20401 lines)
            test.csv.zip (55.2 kB)
            test.zip (432.9 MB)
            train.csv (95089 lines)
            train.csv.zip (6.7 MB)
            train.zip (2.0 GB)
            test/
                case110/
                    case110_day12/
                        scans/
                            ... (max depth reached)
                    case110_day16/
                        scans/
                            ... (max depth reached)
                case113/
                    case113_day22/
                        scans/
                            ... (max depth reached)
                ... and 27 other folders
            train/
                case101/
                    case101_day20/
                        scans/
                            ... (max depth reached)
                    case101_day22/
                        scans/
                            ... (max depth reached)
                    case101_day26/
                        scans/
                            ... (max depth reached)
                    case101_day32/
                        scans/
                            ... (max depth reached)
                case102/
                    case102_day0/
                        scans/
                            ... (max depth reached)
                ... and 75 other folders
            uw-madison-gi-tract-image-segmentation/
                description.md (126 lines)
                sample_submission.csv (20401 lines)
                ... and 7 other files
                test/
                    case110/
                        case110_day12/
                            ... (max depth reached)
                        case110_day16/
                            ... (max depth reached)
                    case113/
                        case113_day22/
                            ... (max depth reached)
                    ... and 27 other folders
                train/
                    case101/
                        case101_day20/
                            ... (max depth reached)
                        case101_day22/
                            ... (max depth reached)
                        case101_day26/
                            ... (max depth reached)
                        case101_day32/
                            ... (max depth reached)
                    case102/
                        case102_day0/
                            ... (max depth reached)
                    ... and 75 other folders
                uw-madison-gi-tract-image-segmentation/
        working/
            uw-madison-gi-tract-image-segmentation/
                description.md (126 lines)
                sample_submission.csv (20401 lines)
                ... and 7 other files
                test/
                    case110/
                        case110_day12/
                            ... (max depth reached)
                        case110_day16/
                            ... (max depth reached)
                    case113/
                        case113_day22/
                            ... (max depth reached)
                    ... and 27 other folders
                train/
                    case101/
                        case101_day20/
                            ... (max depth reached)
                        case101_day22/
                            ... (max depth reached)
                        case101_day26/
                            ... (max depth reached)
                        case101_day32/
                            ... (max depth reached)
                    case102/
                        case102_day0/
                            ... (max depth reached)
                    ... and 75 other folders
                uw-madison-gi-tract-image-segmentation/
```

-> data/sample_submission.csv has 20400 rows and 3 columns.
The columns are: id, class, predicted

-> data/test.csv has 20400 rows and 2 columns.
The columns are: id, class

-> data/train.csv has 95088 rows and 3 columns.
The columns are: id, class, segmentation

-> data/uw-madison-gi-tract-image-segmentation/sample_submission.csv has 20400 rows and 3 columns.
The columns are: id, class, predicted

-> data/uw-madison-gi-tract-image-segmentation/test.csv has 20400 rows and 2 columns.
The columns are: id, class

-> data/uw-madison-gi-tract-image-segmentation/train.csv has 95088 rows and 3 columns.
The columns are: id, class, segmentation

-> input/sample_submission.csv has 20400 rows and 3 columns.
The columns are: id, class, predicted

-> (stopped after 10 files for performance)

# 5. Target score

0.55248

# 6. Current score

0.48817

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.48817) has done: 'I remove (or gate) the interactive plotting/widget cells and any `display()` usage that wastes time in a non-notebook scoring run, since they don’t affect the algorithm or submission. The main timeout bottleneck is the Python-level loops during mask accumulation and submission generation, so I replace `iterrows()` loops with vectorized/NumPy-indexed loops over arrays while keeping the exact same logic (same per-slice RLE decode, same averaging per day, same argmax segmentation and thresholding, same resize). I also avoid repeatedly building the same dictionaries and repeatedly converting arrays between NumPy↔Torch inside inner loops by caching mappings and reusing preallocated arrays where provably equivalent. Finally, I speed up RLE encode by using a standard vectorized run-detection implementation (still correct 1-based, Fortran-order) and avoid redundant work (like encoding empty masks).'
- What this solution (achieved 0.48817) has done: 'We’re currently below the target (0.48817 vs 0.55248, higher-is-better), so the smallest safe way to improve is to fix a correctness issue in how slices are filled. Right now, per Case_Day you allocate the volume as `n_slices` deep but then index it by the original slice number (`Slice-1`), which can exceed `n_slices-1` when slices are not contiguous; this silently breaks mask construction (or misaligns) and hurts score. I minimally change the training-volume construction to allocate depth as `max_slice` and fill by true slice index, while preserving the rest of your logic (same AVG_VOLUME_SIZE, same averaging per day, same argmax+thresholding, same RLE). I also make the analogous test-time slice indexing robust by selecting the actual class-rows used for `ids/slices` explicitly and sorting by Slice to ensure consistent indexing.'
- What this solution (achieved 0.48817) has done: 'Your score is below target (0.48817 vs 0.55248, higher-is-better), so we should make a small, correctness-focused change that’s likely to improve segmentation alignment without changing the overall method. The biggest remaining mismatch is in test-time volume resizing: you currently set the output depth to `len(dfgg)` where `dfgg` contains only one class’s rows (and assumes slices are contiguous/complete), which can misalign masks across slice indices and hurt Dice/Hausdorff. I minimally change test-time depth handling to use the true max slice index for each Case_Day (robust even with missing/non-contiguous slices), and then index into that resized volume by the actual slice number; this preserves your same “average-by-day template → argmax labels → resize → RLE” core logic. I also avoid `iterrows()` in the final per-slice loop to reduce overhead (no semantic change).'
- What this solution (achieved 0.48906) has done: 'Your current approach is close but still loses score due to a remaining slice-depth/indexing mismatch at test-time: you resize the predicted 3D label volume to `depth=max(Slice)` but then index it by `Slice-1`, which requires the resized depth to be `max(Slice)+1` to keep the last slice in-bounds and correctly aligned. I make the smallest correction by setting `depth = slices_all.max() + 1` and keeping everything else (day interpolation, argmax over class templates, nearest resize, and per-slice class mask + RLE) identical. I also reuse the existing cached `lb_to_idx` outside the group loop to avoid rebuilding it (no semantic change), and keep submission formatting exactly as required.'
- What this solution (achieved 0.48817) has done: 'Your current score (0.48906) is below the target (0.55248), so we should make a small correctness fix that improves alignment without changing the underlying “average template per day → argmax class volume → resize → per-slice RLE” logic. The biggest remaining issue is a depth/index mismatch at inference: you resize the predicted label volume to `depth=max(Slice)+1` but the *actual number of rows to predict* for that Case_Day is `len(dfg)`, and those two only coincide if slices are contiguous from 1..max; when they’re not, your per-row indexing can pull masks from empty/unintended slices. I minimally change inference to resize to `depth = len(dfg)` and to build a deterministic `slice->position` mapping from the sorted unique slice numbers, so each submission row gets the correct slice plane (still nearest-neighbor, still same templates/argmax). This should move Dice/Hausdorff upward toward your target while keeping everything else identical.'
- What this solution (achieved 0.48817) has done: 'Your current score is below target (0.48817 vs 0.55248, higher-is-better), so we make a small correctness fix that improves slice-to-volume alignment at inference without changing your core “average template per day → argmax class volume → resize → per-slice RLE” method. Right now you resize to `depth=len(unique_slices)` and then map each row’s `Slice` to a compact position; this can misalign the predicted masks relative to the true physical slice index when there are missing/non-contiguous slices, which hurts Dice/Hausdorff. The minimal fix is to resize the label volume to `depth=max_slice` and index it by `Slice-1` directly (robust and consistent with how training volumes were built). I also switch the per-row loop from `zip(...)` to NumPy arrays (no semantic change) and keep the submission merge/format identical.'
- What this solution (achieved 0.48906) has done: 'Your current score is below target (0.48817 vs 0.55248, higher-is-better), so we should make a minimal correctness fix that improves the slice-to-volume alignment at inference without changing the overall “average-by-day template → argmax → resize → per-slice class mask → RLE” method. Right now you resize the predicted 3D label volume to `depth = max(Slice)`, but then index with `Slice-1`, which makes the last slice (`Slice=max`) out-of-bounds logically and effectively shifts/misassigns masks for that slice (and can silently hurt Dice/Hausdorff). The smallest fix is to resize to `depth = max(Slice) + 1`, preserving your direct physical indexing with `Slice-1`. I also add a safety clip for any unexpected slice index (shouldn’t trigger, but avoids rare crashes) while keeping submission formatting identical.'
- What this solution (achieved 0.48817) has done: 'Your current score is below the target, so we make one small, correctness-focused improvement that should increase Dice/Hausdorff without changing the overall “average template per day → argmax volume → resize → per-slice RLE” logic. The main issue is that inference currently resizes the predicted 3D label volume to `depth=max(Slice)+1` and then indexes by `Slice-1`, which creates empty planes for missing/non-contiguous slices and tends to hurt alignment; we instead resize to `depth=len(unique_slices)` and map each slice number to a compact index deterministically. This preserves nearest-neighbor resizing, preserves the argmax labeling, and only changes the slice-to-plane mapping to be consistent for each Case_Day. Submission generation and file paths stay the same and it still write `submission.csv`.'
- What this solution (achieved 0.48817) has done: 'We’re below target (0.48817 vs 0.55248; higher-is-better), so the smallest likely gain is to fix one remaining correctness mismatch: at inference you resize the 3D label volume to `depth=len(unique_slices)` and then map each slice number to a compact index, which can distort the true physical z-positions when slices are missing/non-contiguous and hurts Dice/Hausdorff. I keep your exact “average template per day → argmax volume → nearest resize → per-slice class mask → RLE” logic, but change inference to resize to `depth = max_slice` (i.e., `max(Slice)`), and index directly by `Slice-1` so slice planes align to their original indices. I also make this consistent by building `vol_lbl` once per Case_Day and clipping the computed index to bounds (safety only). Submission format, paths, thresholds, and all training/template code remain unchanged.'
- What this solution (achieved 0.48906) has done: 'Your current score is below target (0.48817 vs 0.55248; higher-is-better), so the smallest likely gain is a correctness fix in inference: you resize to `depth=max_slice` but index with `Slice-1`, which makes the last slice out-of-bounds logically and forces a clip that can duplicate the last plane for some rows. I change inference depth to `max_slice + 1` so `idx = Slice-1` is always valid and aligned, matching the way training volumes are built (`depth = slices.max()+1`). I also keep everything else identical (same templates, same argmax, same nearest interpolation, same RLE), only removing the now-unnecessary index clipping (kept as a safety clip but it should no longer trigger). This should improve slice-to-volume alignment and move Dice/Hausdorff upward toward the target.'
- What this solution (achieved 0.48817) has done: 'Your current score is below target, so the safest way to move upward is to fix a remaining correctness issue in inference slice-depth alignment without changing your “average template per day → argmax label volume → nearest resize → per-slice class mask → RLE” core logic. Right now you resize the 3D label volume to `depth=max_slice+1`, which creates empty planes when slices are non-contiguous; that tends to hurt Dice/Hausdorff because predictions get misaligned in z. I instead resize to `depth=len(unique_slices)` and map each row’s `Slice` to a compact, deterministic index based on sorted unique slice numbers (nearest-neighbor semantics preserved). Everything else (templates, thresholds, argmax, interpolation mode, RLE format, output CSV schema/path) stays the same.'

# 9. Code solution

## === cell 0
import os, glob
import numpy as np
import pandas as pd

DATASET_FOLDER = "/kaggle/input/uw-madison-gi-tract-image-segmentation"



## === cell 1
df_train = pd.read_csv(os.path.join(DATASET_FOLDER, "train.csv"))
print(f"train size: {len(df_train)}")
print(df_train.head(3))

df_ssub = pd.read_csv(os.path.join(DATASET_FOLDER, "sample_submission.csv"))
print(f"sample_submission size: {len(df_ssub)}")
print(df_ssub.head(3))




## === cell 2
def enrich_data(df, sdir="train"):
    imgs = glob.glob(
        os.path.join(DATASET_FOLDER, sdir, "case*", "case*_day*", "scans", "*.png")
    )

    folders = [os.path.dirname(p).split(os.path.sep) for p in imgs]
    names = [os.path.splitext(os.path.basename(p))[0].split("_") for p in imgs]
    keys = [f"{f[-2]}_slice_{n[1]}" for f, n in zip(folders, names)]

    key_to_path = dict(zip(keys, imgs))
    key_to_caseday = {k: f[-2] for k, f in zip(keys, folders)}
    key_to_slice = {k: int(n[1]) for k, n in zip(keys, names)}
    key_to_w = {k: int(n[2]) for k, n in zip(keys, names)}
    key_to_h = {k: int(n[3]) for k, n in zip(keys, names)}
    key_to_s1 = {k: float(n[4]) for k, n in zip(keys, names)}
    key_to_s2 = {k: float(n[5]) for k, n in zip(keys, names)}

    df["img_path"] = df["id"].map(key_to_path)
    df["Case_Day"] = df["id"].map(key_to_caseday)

    s = df["id"].astype(str)
    df["Case"] = (
        s.str.split("_", n=2, expand=True)[0]
        .str.replace("case", "", regex=False)
        .astype(int)
    )
    df["Day"] = (
        s.str.split("_", n=2, expand=True)[1]
        .str.replace("day", "", regex=False)
        .astype(int)
    )

    df["Slice"] = df["id"].map(key_to_slice)
    df["width"] = df["id"].map(key_to_w)
    df["height"] = df["id"].map(key_to_h)
    df["spacing1"] = df["id"].map(key_to_s1)
    df["spacing2"] = df["id"].map(key_to_s2)




## === cell 3
enrich_data(df_train, "train")
print(df_train.head())



## === cell 4
enrich_data(df_ssub, "test")
print(df_ssub.head())




## === cell 5
def rle_decode(rle, img=None, label=1):
    if img is None:
        raise ValueError("img must be provided to infer shape")
    if (
        rle is None
        or (isinstance(rle, float) and np.isnan(rle))
        or (isinstance(rle, str) and len(rle.strip()) == 0)
    ):
        return img

    s = rle.split()
    starts = np.asarray(s[0::2], dtype=np.int64) - 1
    lengths = np.asarray(s[1::2], dtype=np.int64)
    ends = starts + lengths

    h, w = img.shape
    flat = img.reshape(-1, order="F")
    for lo, hi in zip(starts, ends):
        flat[lo:hi] = label
    return flat.reshape((h, w), order="F")




## === cell 6
import torch
import torch.nn.functional as F


def interpolate_volume(volume, vol_size):
    vol_shape = tuple(volume.shape)
    if not vol_size:
        d_new = min(vol_shape[:2])
        vol_size = (vol_shape[0], vol_shape[1], d_new)
    if vol_shape == vol_size:
        return volume
    vol = F.interpolate(volume.unsqueeze(0).unsqueeze(0), size=vol_size, mode="nearest")
    return vol[0, 0]




## === cell 7
from tqdm.auto import tqdm

AVG_VOLUME_SIZE = (144, 266, 266)
segms, counts = {}, {}

for lb, dfg in df_train.groupby("class", sort=False):
    print(lb)
    vol_acc, nb_acc = {}, {}

    for cd, dfgg in tqdm(
        dfg.groupby("Case_Day", sort=False), total=dfg["Case_Day"].nunique()
    ):
        day = int(dfgg["Day"].iloc[0])
        h = int(dfgg["height"].iloc[0])
        w = int(dfgg["width"].iloc[0])

        slices = dfgg["Slice"].to_numpy(dtype=np.int64) - 1
        segs = dfgg["segmentation"].to_numpy(object)

        depth = int(slices.max() + 1)  # ensures idx is always in-bounds and aligned
        vol = np.zeros((depth, h, w), dtype=np.uint8)

        for i in range(len(dfgg)):
            rle = segs[i]
            if isinstance(rle, str) and rle:
                idx = int(slices[i])
                vol[idx, :, :] = rle_decode(rle, img=vol[idx, :, :], label=1)

        vol_t = torch.from_numpy(vol)
        vol_rs = interpolate_volume(vol_t, vol_size=AVG_VOLUME_SIZE).numpy()

        if day not in vol_acc:
            vol_acc[day] = vol_rs.astype(np.float32, copy=False)
            nb_acc[day] = 1
        else:
            vol_acc[day] += vol_rs
            nb_acc[day] += 1

    for d in list(vol_acc.keys()):
        vol_acc[d] = vol_acc[d] / float(nb_acc[d])

    segms[lb] = vol_acc
    counts[lb] = nb_acc



## === cell 8
days = sorted(segms["stomach"].keys())
print(days)



## === cell 9
pass



## === cell 10
pass



## === cell 11
LABELS = sorted(df_train["class"].unique())



## === cell 12
pass




## === cell 13
def rle_encode(mask, bg=0) -> str:
    pixels = np.asarray(mask, dtype=np.uint8).reshape(-1, order="F")
    if pixels.sum() == 0:
        return ""

    pads = np.empty(pixels.size + 2, dtype=np.uint8)
    pads[0] = bg
    pads[-1] = bg
    pads[1:-1] = pixels

    changes = np.nonzero(pads[1:] != pads[:-1])[0] + 1
    runs = changes.copy()
    runs[1::2] -= runs[0::2]
    return " ".join(map(str, runs.tolist()))




## === cell 14
def _interpolate_day(segm_lb, day):
    if day in segm_lb:
        return segm_lb[day]
    keys = sorted(segm_lb.keys())
    day_last = max(d for d in keys if d < day)
    next_days = [d for d in keys if d > day]
    if not next_days:
        return segm_lb[day_last]
    day_next = min(next_days)
    dn = (day - day_last) / (day_next - day_last)
    dl = (day_next - day) / (day_next - day_last)
    return dl * segm_lb[day_last] + dn * segm_lb[day_next]


def _process_vol(dfgg, segm, thr, lb):
    day = int(dfgg[["Day"]].iloc[0])
    h, w = dfgg[["height", "width"]].iloc[0]
    vol = (
        interpolate_volume(
            torch.tensor(_interpolate_day(segm, day) > thr, dtype=torch.float32),
            vol_size=(len(dfgg), h, w),
        )
        .numpy()
        .astype(np.uint8)
    )
    rows = []
    for _, row in dfgg.iterrows():
        idx = int(row["Slice"]) - 1
        mask = vol[idx, :, :]
        rle = rle_encode(mask) if np.sum(mask) > 0 else ""
        rows.append({"id": row["id"], "class": lb, "predicted": rle})
    return rows




## === cell 15
segm_thr = 0.05

df_pred_base = df_ssub.copy()
preds = []

lb_to_idx = {lb: i for i, lb in enumerate(LABELS, start=1)}

for _, dfg in tqdm(
    df_pred_base.groupby("Case_Day", sort=False),
    total=df_pred_base["Case_Day"].nunique(),
):
    day = int(dfg["Day"].iloc[0])

    segms_lb = [np.ones(AVG_VOLUME_SIZE, dtype=np.float32) * segm_thr]
    for lb in LABELS:
        segms_lb.append(_interpolate_day(segms[lb], day).astype(np.float32, copy=False))

    segm_sc = np.argmax(segms_lb, axis=0).astype(np.uint8)  # 0=bg, 1..3=classes

    h = int(dfg["height"].iloc[0])
    w = int(dfg["width"].iloc[0])

    slices_all = dfg["Slice"].to_numpy(np.int64)
    uniq_slices = np.unique(slices_all)
    uniq_slices.sort()
    slice_to_pos = {int(s): i for i, s in enumerate(uniq_slices)}
    depth = int(len(uniq_slices))

    vol_lbl = (
        interpolate_volume(
            torch.from_numpy(segm_sc.astype(np.float32, copy=False)),
            vol_size=(depth, h, w),
        )
        .numpy()
        .astype(np.uint8, copy=False)
    )

    ids = dfg["id"].to_numpy(object)
    classes = dfg["class"].to_numpy(object)

    max_idx = depth - 1

    for i in range(len(dfg)):
        img_id = ids[i]
        sl = int(slices_all[i])
        cl = classes[i]

        idx = slice_to_pos.get(sl, 0)
        if idx < 0:
            idx = 0
        elif idx > max_idx:
            idx = max_idx

        mask_lbl = vol_lbl[idx, :, :]  # {0,1,2,3}
        class_idx = lb_to_idx[cl]
        mask = (mask_lbl == class_idx).astype(np.uint8, copy=False)
        rle = rle_encode(mask)
        preds.append({"id": img_id, "class": cl, "predicted": rle})

df_pred = pd.DataFrame(preds)
print(df_pred[df_pred["predicted"] != ""].head(10))

sub = df_ssub.drop(columns=["predicted"]).merge(df_pred, on=["id", "class"], how="left")
sub["predicted"] = sub["predicted"].fillna("")
sub[["id", "class", "predicted"]].to_csv("submission.csv", index=False)

with open("submission.csv", "r", encoding="utf-8") as f:
    for _ in range(5):
        print(f.readline().rstrip("\n"))
print("submission rows:", len(sub), "non-empty:", int((sub["predicted"] != "").sum()))
