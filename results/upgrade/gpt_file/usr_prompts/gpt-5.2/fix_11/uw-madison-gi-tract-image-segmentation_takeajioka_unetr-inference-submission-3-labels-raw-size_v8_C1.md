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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
nibabel==5.3.2
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

0.8333879605969338

# 6. Current score

0.0309

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00373) has done: 'I fix the UNet forward runtime error by making the upsampled feature maps match the encoder feature sizes (handling odd image dimensions) before concatenation; this keeps the same architecture and avoids shape mismatches. I also ensure inference runs deterministically and that `pred_df` is always created so the submission-writing cell cannot fail. Finally, I keep the existing “no-trained-weights” core logic but make the pipeline complete end-to-end and write a valid `submission.csv` with the exact required columns and row count.'
- What this solution (achieved 0.00407) has done: 'Your current score is extremely low because the UNet is never trained (random weights), so predictions are essentially noise; the smallest legitimate improvement toward your target is to train this exact same UNet on the provided `train.csv` masks and corresponding `train/` PNG slices. I keep the architecture, loss (BCEWithLogits), and simple epoch-based training loop minimal, add a train dataset that loads slices + 3-class masks from RLE, and then run the same inference/RLE submission pipeline as you already have. I also make sure the slice indexing is correct by mapping `slice` values to the sorted scan PNG list per volume (instead of `slice-1`), which fixes a common misalignment that can destroy Dice/Hausdorff. These changes are directly aimed at moving the score upward toward your target while staying within Kaggle constraints and producing a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'I fix the DataLoader crash by ensuring all training samples in a batch have the same spatial size, using a minimal padding-based collate function (no resizing, so the model and training semantics stay intact). I also make the training dataset use the correct (H,W) per slice by reading it from each PNG instead of assuming a volume-wide constant, which prevents silent mask/image shape mismatches. These changes unblock end-to-end training/inference and should significantly improve score versus effectively-untrained/random behavior, while preserving your UNet, loss, optimizer, and epoch loop. The submission-writing logic remains the same and still output a valid `submission.csv` with the correct columns/row count.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with a “mostly empty / badly misaligned” submission; the fastest legitimate way to move upward toward the 0.833 target (without changing your UNet, loss, optimizer, or overall pipeline) is to (1) train on substantially more slices (still the same slice-wise supervision), (2) ensure each sample’s image and masks use the exact same (H,W) by reading from the PNG that corresponds to that `id` (not just any slice in the same volume), and (3) reduce false positives at inference with a slightly higher fixed threshold (calibration change only). These are minimal, metric-relevant changes that preserve your core approach while greatly reducing the chance of an all-zero-like submission. The code below keeps your architecture and training loop intact, but fixes the biggest data-alignment pitfall and uses more of the available training data within the runtime budget. It still writes a valid `submission.csv` with the required columns and row count.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with producing invalid/near-empty masks due to a train/test normalization mismatch and an overly strict inference threshold. I keep your UNet, BCEWithLogitsLoss, optimizer, and epoch loop unchanged, but make training use the same per-volume normalization you use at inference (instead of per-slice), which is a minimal, metric-relevant alignment fix. I also slightly lower the fixed inference threshold (calibration-only change) to reduce the chance of all-empty predictions, which is the most common cause of 0.0 here. Finally, I add a lightweight sanity check to ensure we’re actually emitting non-empty RLEs for at least some rows before writing `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with a submission that decodes to (nearly) all-empty masks; the quickest legitimate way to move upward toward your target while keeping the same UNet, loss, and training loop is to (1) fix the train/test slice-index mismatch (your training uses `SliceInt` but your scan filenames’ leading numbers are not the same as `SliceInt`), and (2) set a slightly less-strict inference threshold so the model emits some positives instead of collapsing to empty. I keep your architecture, BCEWithLogitsLoss, Adam, and epoch loop unchanged, but rebuild the scan index by using the scan list order (1..N) rather than parsing the filename prefix, which aligns with `slice_####` in the IDs. Finally, I add a tiny safety check to ensure `pred_df` covers all (id,class) pairs before writing `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the submission decoding to invalid/degenerate masks because the per-slice IDs are not aligned to the model’s predicted slice index: you infer in simple sorted scan order (0..z-1) but then later index with `SliceInt` assuming it maps to that order, while `df_test` per-volume slice ordering can differ if not explicitly enforced. I make the slice-index mapping deterministic and identical for both train and test by building an explicit per-(case,day) DataFrame that assigns `slice_idx` from the sorted scan list and then joining it to `df_train/df_test` (minimal data-alignment fix; model/loss/training loop unchanged). I also remove the redundant `load_image_volume_with_paths(...); build_scan_index(...)` double-scan in inference and ensure we always use the same scan list for normalization and indexing. These changes are directly aimed at producing non-empty, correctly-aligned masks so the metric is no longer 0.0, moving the score upward toward your target.'
- What this solution (achieved 0.43512) has done: 'Your 0.0 score is most likely coming from a submission that decodes to mostly-empty or badly-calibrated masks, not from CSV formatting (your submission schema is correct). To move the score upward toward the 0.833 target with minimal changes and without altering the core UNet/training loop/loss, I (1) use a slightly less strict inference threshold (calibration-only) to avoid all-empty outputs, and (2) ensure the model is trained on a bit more data within the same slice-wise pipeline by increasing `max_volumes` modestly (still the same dataset logic). I also add a tiny safety fallback: if a volume produces near-all-empty predictions, rerun that volume once with a lower threshold (still deterministic, no new model, just post-processing) to prevent catastrophic 0.0. These changes are directly aimed at producing non-degenerate masks and improving overlap metrics while keeping the rest intact.'
- What this solution (achieved 0.0309) has done: 'I keep your exact UNet, loss (BCEWithLogits), and epoch-based training loop intact, but reduce the train/test mismatch by training with the same thresholded binarization semantics you use at inference (i.e., mild label smoothing via `pos_weight` is avoided; instead we correct the padded-area influence). The main score lift comes from masking padded pixels out of the loss (so the model doesn’t learn “padding=background” artifacts) and from using per-class thresholds that better match class prevalence while staying a pure post-processing calibration change. Finally, I add a deterministic per-volume threshold fallback that triggers per-class (not only when the whole volume is empty), which prevents catastrophic all-empty predictions for a specific organ without changing the model.'

# 9. Code solution

## === cell 0
import os, glob, gc
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 1
DATASET_FOLDER = "../input/uw-madison-gi-tract-image-segmentation"
SAMPLE_SUB_PATH = os.path.join(DATASET_FOLDER, "sample_submission.csv")
TEST_CSV_PATH = os.path.join(DATASET_FOLDER, "test.csv")
TRAIN_CSV_PATH = os.path.join(DATASET_FOLDER, "train.csv")
TEST_FOLDER = os.path.join(DATASET_FOLDER, "test")
TRAIN_FOLDER = os.path.join(DATASET_FOLDER, "train")

sub_df = pd.read_csv(SAMPLE_SUB_PATH)
df_test = pd.read_csv(TEST_CSV_PATH)
df_train = pd.read_csv(TRAIN_CSV_PATH)

print(
    "sample_submission:",
    sub_df.shape,
    "test.csv:",
    df_test.shape,
    "train.csv:",
    df_train.shape,
)
print(df_test.head())
print(df_train.head())




## === cell 2
def extract_details(id_):
    id_fields = id_.split("_")
    case = id_fields[0].replace("case", "")
    day = id_fields[1].replace("day", "")
    slice_id = id_fields[3]
    return {"Case": int(case), "Day": int(day), "Slice": slice_id}


df_test[["Case", "Day", "Slice"]] = df_test["id"].apply(
    lambda x: pd.Series(extract_details(x))
)
df_train[["Case", "Day", "Slice"]] = df_train["id"].apply(
    lambda x: pd.Series(extract_details(x))
)

df_test["SliceInt"] = df_test["Slice"].astype(int)
df_train["SliceInt"] = df_train["Slice"].astype(int)

print(df_test.head())
print(df_train.head())



## === cell 3
test_overview = []
for (case, day), dfg in df_test.groupby(["Case", "Day"], sort=True):
    test_overview.append({"Day": int(day), "Case": int(case), "Slices": len(dfg)})
df_overview = (
    pd.DataFrame(test_overview).sort_values(["Case", "Day"]).reset_index(drop=True)
)
print(df_overview.head(), "num test volumes:", len(df_overview))

train_overview = []
for (case, day), dfg in df_train.groupby(["Case", "Day"], sort=True):
    train_overview.append({"Day": int(day), "Case": int(case), "Slices": len(dfg)})
df_train_overview = (
    pd.DataFrame(train_overview).sort_values(["Case", "Day"]).reset_index(drop=True)
)
print(df_train_overview.head(), "num train volumes:", len(df_train_overview))




## === cell 4
def build_scan_list(img_dir):
    scans = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    if len(scans) == 0:
        raise FileNotFoundError(f"No png files in {img_dir}")
    return scans


def attach_slice_idx(df, root_folder):
    out_parts = []
    for (case, day), dfg in df.groupby(["Case", "Day"], sort=True):
        img_dir = os.path.join(
            root_folder, f"case{int(case)}", f"case{int(case)}_day{int(day)}", "scans"
        )
        scans = build_scan_list(img_dir)
        n = len(scans)
        mapper = pd.DataFrame(
            {
                "Case": int(case),
                "Day": int(day),
                "SliceInt": np.arange(1, n + 1, dtype=np.int64),
                "slice_idx": np.arange(0, n, dtype=np.int64),
                "scan_path": scans,
            }
        )
        out_parts.append(dfg.merge(mapper, on=["Case", "Day", "SliceInt"], how="left"))
    return pd.concat(out_parts, axis=0, ignore_index=True)


df_test = attach_slice_idx(df_test, TEST_FOLDER)
df_train = attach_slice_idx(df_train, TRAIN_FOLDER)

print("df_test w/ slice_idx null fraction:", df_test["slice_idx"].isna().mean())
print("df_train w/ slice_idx null fraction:", df_train["slice_idx"].isna().mean())
print(df_test.head())




## === cell 5
def load_volume_from_scans(scans, quant=0.01):
    arrs = [np.array(Image.open(p)) for p in scans]
    vol = np.stack(arrs, axis=0).astype(np.float32)  # (z,h,w)
    if quant:
        q_low, q_high = np.percentile(vol, [quant * 100, (1 - quant) * 100])
        vol = np.clip(vol, q_low, q_high)
    v_min, v_max = float(np.min(vol)), float(np.max(vol))
    if v_max > v_min:
        vol = (vol - v_min) / (v_max - v_min)
    else:
        vol = np.zeros_like(vol, dtype=np.float32)
    return vol.astype(np.float32), (v_min, v_max)


def load_image_volume_with_paths(img_dir, quant=0.01):
    scans = build_scan_list(img_dir)
    vol, _ = load_volume_from_scans(scans, quant=quant)
    return vol, scans


def load_image_volume(img_dir, quant=0.01):
    vol, _ = load_image_volume_with_paths(img_dir, quant=quant)
    return vol




## === cell 6
def rle_encode(mask2d: np.ndarray) -> str:
    """
    mask2d: 2D boolean/0-1 array (H,W). Kaggle GI Tract uses RLE over pixels in column-major order
    by flattening the transpose.
    """
    if mask2d.dtype != np.uint8:
        mask2d = mask2d.astype(np.uint8)
    pixels = mask2d.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    if len(runs) == 0:
        return ""
    return " ".join(str(x) for x in runs)


def rle_decode(rle: str, shape):
    """
    rle: run-length string (start length ...)
    shape: (H,W)
    Returns uint8 mask (H,W) with 0/1.
    """
    h, w = shape
    if rle is None or (isinstance(rle, float) and np.isnan(rle)) or rle == "":
        return np.zeros((h, w), dtype=np.uint8)
    s = rle.strip().split()
    starts = np.asarray(s[0::2], dtype=np.int64) - 1
    lengths = np.asarray(s[1::2], dtype=np.int64)
    ends = starts + lengths
    img = np.zeros(h * w, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape((w, h)).T




## === cell 7
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet2D(nn.Module):
    def __init__(self, in_ch=1, out_ch=3, base=32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base * 4, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.head = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        if d3.shape[-2:] != e3.shape[-2:]:
            d3 = F.interpolate(
                d3, size=e3.shape[-2:], mode="bilinear", align_corners=False
            )
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = F.interpolate(
                d2, size=e2.shape[-2:], mode="bilinear", align_corners=False
            )
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(
                d1, size=e1.shape[-2:], mode="bilinear", align_corners=False
            )
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.head(d1)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(
                out, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet2D(in_ch=1, out_ch=3, base=32).to(device)
print("Using device:", device)



## === cell 8
CLASSES = ["large_bowel", "small_bowel", "stomach"]
class_to_ch = {c: i for i, c in enumerate(CLASSES)}


class TrainSliceDataset(Dataset):
    def __init__(self, df_train, quant=0.01, max_volumes=90):
        self.items = []
        self.quant = quant

        vols = df_train.groupby(["Case", "Day"], sort=True)
        sel = list(vols.groups.keys())[:max_volumes]

        for case, day in sel:
            dfv = df_train[(df_train["Case"] == case) & (df_train["Day"] == day)].copy()
            dfv = dfv[dfv["scan_path"].notna()].copy()
            if len(dfv) == 0:
                continue

            scans = (
                dfv[["slice_idx", "scan_path"]]
                .drop_duplicates()
                .sort_values("slice_idx")["scan_path"]
                .tolist()
            )
            vol_raw, (v_min, v_max) = load_volume_from_scans(scans, quant=self.quant)

            dfv = dfv.sort_values(["SliceInt", "class"])
            for sid, dfs in dfv.groupby("SliceInt", sort=True):
                scan_path = dfs["scan_path"].iloc[0]
                tmp = np.array(Image.open(scan_path))
                h, w = int(tmp.shape[0]), int(tmp.shape[1])

                mask = np.zeros((3, h, w), dtype=np.uint8)
                for _, r in dfs.iterrows():
                    ch = class_to_ch[r["class"]]
                    mask[ch] = rle_decode(r["segmentation"], (h, w))

                self.items.append((scan_path, mask, v_min, v_max))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, mask, v_min, v_max = self.items[i]
        img = np.array(Image.open(p)).astype(np.float32)

        if self.quant:
            q_low, q_high = np.percentile(
                img, [self.quant * 100, (1 - self.quant) * 100]
            )
            img = np.clip(img, q_low, q_high)

        if v_max > v_min:
            img = (img - v_min) / (v_max - v_min)
        else:
            img = np.zeros_like(img, dtype=np.float32)

        x = torch.from_numpy(img).unsqueeze(0)  # (1,h,w)
        y = torch.from_numpy(mask.astype(np.float32))  # (3,h,w)
        return x, y


def pad_collate(batch):
    xs, ys = zip(*batch)
    max_h = max(x.shape[-2] for x in xs)
    max_w = max(x.shape[-1] for x in xs)

    xb, yb, mb = [], [], []
    for x, y in zip(xs, ys):
        _, h, w = x.shape
        pad_h = max_h - h
        pad_w = max_w - w

        xb.append(F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0))
        yb.append(F.pad(y, (0, pad_w, 0, pad_h), mode="constant", value=0.0))

        m = torch.ones((1, h, w), dtype=torch.float32)
        m = F.pad(
            m, (0, pad_w, 0, pad_h), mode="constant", value=0.0
        )  # 1 on real pixels
        mb.append(m)

    return torch.stack(xb, 0), torch.stack(yb, 0), torch.stack(mb, 0)


train_ds = TrainSliceDataset(df_train, quant=0.01, max_volumes=90)
print("Train slices:", len(train_ds))

train_dl = DataLoader(
    train_ds,
    batch_size=8,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
    collate_fn=pad_collate,
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss(reduction="none")

model.train()
epochs = 2
for ep in range(epochs):
    losses = []
    for xb, yb, mb in train_dl:
        xb = xb.to(device, non_blocking=True).float()
        yb = yb.to(device, non_blocking=True).float()
        mb = mb.to(device, non_blocking=True).float()  # (b,1,H,W)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)

        per_pix = criterion(logits, yb)  # (b,3,H,W)
        per_pix = per_pix * mb  # broadcast to (b,3,H,W)
        denom = mb.sum() * yb.shape[1] + 1e-6
        loss = per_pix.sum() / denom

        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    print(f"epoch {ep+1}/{epochs} loss={np.mean(losses):.4f}")

model.eval()
gc.collect()




## === cell 9
class SliceDataset(Dataset):
    def __init__(self, vol: np.ndarray):
        self.vol = vol

    def __len__(self):
        return self.vol.shape[0]

    def __getitem__(self, idx):
        img = self.vol[idx]  # (h,w)
        img = torch.from_numpy(img).unsqueeze(0)  # (1,h,w)
        return img, idx


def infer_volume_to_masks(vol: np.ndarray, batch_size=8, thr=(0.25, 0.25, 0.25)):
    if isinstance(thr, (float, int)):
        thr = (float(thr), float(thr), float(thr))
    thr = np.asarray(thr, dtype=np.float32).reshape(3, 1, 1)

    ds = SliceDataset(vol)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    z, h, w = vol.shape
    out = np.zeros((3, z, h, w), dtype=np.uint8)

    with torch.no_grad():
        for xb, idxb in dl:
            xb = xb.to(device, non_blocking=True).float()
            logits = model(xb)  # (b,3,h,w)
            probs = torch.sigmoid(logits).cpu().numpy()  # (b,3,h,w)
            preds = (probs > thr[None, ...]).astype(np.uint8)
            for bi, zi in enumerate(idxb.numpy().tolist()):
                out[:, zi, :, :] = preds[bi]
    return out




## === cell 10
pred_rows = []

DEFAULT_THR = (0.28, 0.22, 0.22)
FALLBACK_THR = (0.22, 0.16, 0.16)

for i in range(len(df_overview)):
    CASE = int(df_overview.loc[i, "Case"])
    DAY = int(df_overview.loc[i, "Day"])
    img_dir = os.path.join(TEST_FOLDER, f"case{CASE}", f"case{CASE}_day{DAY}", "scans")

    scans = build_scan_list(img_dir)
    vol, _ = load_volume_from_scans(scans, quant=0.01)
    z, h, w = vol.shape
    print(f"Volume case{CASE}_day{DAY} shape:", vol.shape)

    segm = infer_volume_to_masks(vol, batch_size=8, thr=DEFAULT_THR)  # (3,z,h,w)

    per_class_nonempty = segm.reshape(3, -1).sum(axis=1) > 0
    if not bool(per_class_nonempty.all()):
        segm_fb = infer_volume_to_masks(vol, batch_size=8, thr=FALLBACK_THR)
        for ch in range(3):
            if not per_class_nonempty[ch]:
                segm[ch] = segm_fb[ch]
        del segm_fb

    df_vol = df_test[(df_test["Case"] == CASE) & (df_test["Day"] == DAY)].copy()
    df_vol = df_vol.sort_values(["slice_idx", "class"]).reset_index(drop=True)

    for _, row in df_vol.iterrows():
        cls = row["class"]
        ch = class_to_ch[cls]
        if pd.isna(row["slice_idx"]):
            mask = np.zeros((h, w), dtype=np.uint8)
        else:
            slice_idx = int(row["slice_idx"])
            if 0 <= slice_idx < z:
                mask = segm[ch, slice_idx]
            else:
                mask = np.zeros((h, w), dtype=np.uint8)
        pred_rows.append({"id": row["id"], "class": cls, "predicted": rle_encode(mask)})

    del vol, segm, df_vol
    gc.collect()

pred_df = pd.DataFrame(pred_rows, columns=["id", "class", "predicted"])
print("pred_df:", pred_df.shape)
print(pred_df.head())

non_empty = (pred_df["predicted"].astype(str).str.len() > 0).mean()
print(f"Non-empty predicted RLE fraction: {non_empty:.4f}")

assert pred_df.shape[0] == df_test.shape[0], (pred_df.shape, df_test.shape)



## === cell 11
sub_df = pd.read_csv(SAMPLE_SUB_PATH)
sub_df = sub_df[["id", "class"]].merge(pred_df, on=["id", "class"], how="left")
sub_df["predicted"] = sub_df["predicted"].fillna("")
assert sub_df.shape[0] == pd.read_csv(SAMPLE_SUB_PATH).shape[0]

sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print("Wrote:", sub_path, "shape:", sub_df.shape)
print(sub_df.head())
