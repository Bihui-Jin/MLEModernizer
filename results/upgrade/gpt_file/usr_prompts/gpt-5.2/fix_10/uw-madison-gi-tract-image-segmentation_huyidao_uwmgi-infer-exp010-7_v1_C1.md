# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

No external packages required in the script and installed.

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

0.8419090350968389

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0054) has done: 'I remove the hard dependency on `segmentation_models_pytorch` by implementing a small PyTorch U-Net fallback that matches the same “U-Net-style” core logic (encoder–decoder with skip connections) so inference can run in this environment. I also fix RLE encoding to use the competition’s required Fortran-order flattening (top-to-bottom then left-to-right), and ensure empty masks serialize to an empty string. Finally, I make the checkpoint loading robust (handles missing file / unexpected key prefixes) and guarantee that `submission.csv` is always written with exactly the required columns.'
- What this solution (achieved 0.0054) has done: 'Your code currently can’t yield a meaningful Kaggle score because it is very likely running with randomly initialized weights (the referenced checkpoint path probably doesn’t exist in this environment), which produces near-empty/garbage masks. The smallest score-improving change is therefore to make checkpoint discovery deterministic and robust by preferentially searching common Kaggle input locations for a valid `.ckpt/.pth/.pt` and then loading it (without changing the model core logic). I also fix a subtle but important bug in `load_slice`: it parses the slice number from the wrong underscore field, which can break 5-slice stacking and hurt predictions even with a good checkpoint. These changes preserve your architecture/inference pipeline and only address correctness and weight-loading so you can generate a valid submission with a materially higher score toward your target.'
- What this solution (achieved 0.0) has done: 'I fix the dataframe parsing bug that breaks `extract_metadata_from_path` (file names sometimes don’t split into exactly 5 parts), which currently prevents `image_path` from being created and causes downstream `KeyError`s in the Dataset/DataLoader. I make the path metadata extraction robust by using `Path.parts` + regex on the scan filename, ensuring we always recover `case/day/slice` and keep `image_path` aligned. Then I ensure inference always runs (even if some rows are missing paths) and that the final `submission.csv` is written with exactly `id,class,predicted` columns by building it from `sample_submission.csv` and filling missing predictions with empty strings. These changes are execution-unblocking and score-positive (they restore correct test image mapping and 5-slice stacking) without changing the model architecture or inference semantics.'
- What this solution (achieved 0.0) has done: 'I fix the preprocessing so `test_df` is never empty by correcting how the slice number is parsed from the `id` (it contains `slice_XXXX` as the third token, not a separate token) and by making path parsing robust to both `.../caseXXX/caseXXX_dayYY/scans/*.png` and any unexpected but similar nesting. Then I ensure the submission schema is exactly what Kaggle expects by sanitizing column names/types (strip whitespace, force exact `id,class,predicted` order) before writing `submission.csv`. These changes unblock end-to-end execution and produce a valid `.csv` submission without altering the model/inference core logic.'
- What this solution (achieved 0.0) has done: 'The empty `test_df` comes from a mismatch between the slice index encoded in `id` (1-based, `slice_0001`) and the slice index in filenames (0-based in many cases), plus a fragile merge that drops all rows when paths don’t match. I fix this by extracting slice as before but performing a robust join that, for each `(case, day)`, maps IDs to scan files by sorted order and falls back to an off-by-one match when needed—this preserves your 5-slice stacking/inference logic and just repairs the test image mapping. I also make the hidden-test detection reliable (sample_submission is never empty) and ensure we always write a valid `submission.csv` with the required schema even if some images can’t be resolved. These changes are execution-unblocking and should materially improve score versus predicting on an empty/incorrect test set, without changing model architecture or the inference pipeline.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")



## === cell 1
from pathlib import Path
from typing import List, Tuple
import re
import warnings

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm.notebook import tqdm

try:
    import cupy as cp  # type: ignore
except Exception:
    cp = None  # fallback handled in mask2rle

try:
    import segmentation_models_pytorch as smp  # type: ignore
except Exception as e:
    smp = None
    _smp_import_error = e

warnings.filterwarnings("ignore")
torch.set_grad_enabled(False)



## === cell 2
KAGGLE_DIR = Path("/") / "kaggle"
INPUT_DIR = KAGGLE_DIR / "input"
OUTPUT_DIR = KAGGLE_DIR / "working"

INPUT_DATA_DIR = INPUT_DIR / "uw-madison-gi-tract-image-segmentation"
INPUT_DATA_NPY_DIR = (
    INPUT_DIR / "uw-madison-gi-tract-image-segmentation-masks"
)  # unused but preserved

IMG_SIZE = 356
CROP_SIZE = 320
USE_AUGS = True
BATCH_SIZE = 32
NUM_WORKERS = 2
ENCODER_NAME = "efficientnet-b3"
GPUS = 1
CHANNELS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THR = 0.45

DEBUG = False  # Debug complete pipeline

print("DEVICE:", DEVICE)



## === cell 3
transforms_val = A.Compose(
    [A.CenterCrop(CROP_SIZE, CROP_SIZE, p=1), ToTensorV2(transpose_mask=True)]
)




## === cell 4
class UWDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def resize(self, img, interp):
        return cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=interp)

    def load_slice(self, img_file, diff):
        """
        Scan filenames are like:
          {slice}_{W}_{H}_{spacingW}_{spacingH}.png
        We adjust the first underscore field only to load neighbors.
        """
        base = os.path.basename(img_file)
        stem = base.replace(".png", "")
        parts = stem.split("_")
        if len(parts) < 5:
            return None

        try:
            slice_idx = int(parts[0])
        except Exception:
            return None

        target_idx = slice_idx + diff
        if target_idx < 0:
            return None

        parts2 = parts[:]
        parts2[0] = str(int(target_idx))
        filename = os.path.join(os.path.dirname(img_file), "_".join(parts2) + ".png")
        if os.path.exists(filename):
            return cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        return None

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        img_path = row["image_path"]
        imgs = [self.load_slice(img_path, i) for i in range(-2, 3)]
        if imgs[2] is None:
            imgs[2] = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if imgs[3] is None:
            imgs[3] = imgs[2]
        if imgs[4] is None:
            imgs[4] = imgs[3]
        if imgs[1] is None:
            imgs[1] = imgs[2]
        if imgs[0] is None:
            imgs[0] = imgs[1]

        image = np.stack(imgs, axis=2).astype(np.float32)  # (H,W,5)
        h, w = image.shape[:2]
        max_val = float(image.max())
        if max_val != 0:
            image /= max_val

        image = self.resize(image, cv2.INTER_AREA)

        if self.transforms:
            data = self.transforms(image=image)
            image = data["image"]  # torch tensor (C,H,W)

        return {"image": image, "id": row["id"], "h": h, "w": w}




## === cell 5
def extract_metadata_from_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    ids look like:
      case123_day45_slice_0001
    Splitting with n=2 yields:
      ['case123', 'day45', 'slice_0001']
    """
    df = df.copy()
    parts = df["id"].astype(str).str.split("_", n=2, expand=True)
    df["case"] = parts[0].str.replace("case", "", regex=False).astype(int)
    df["day"] = parts[1].str.replace("day", "", regex=False).astype(int)
    df["slice"] = (
        parts[2].astype(str).str.extract(r"slice[_]?(\d+)", expand=False).astype(int)
    )
    return df


def extract_metadata_from_path(path_df: pd.DataFrame) -> pd.DataFrame:
    """
    Robust parsing of:
      .../case{case}/case{case}_day{day}/scans/{slice}_{W}_{H}_{spW}_{spH}.png
    """
    rows = []
    scan_re = re.compile(
        r"^(?P<slice>\d+)_(?P<width>\d+)_(?P<height>\d+)_(?P<spw>[\d.]+)_(?P<sph>[\d.]+)\.png$"
    )
    case_day_re = re.compile(r"^case(?P<case>\d+)_day(?P<day>\d+)$")

    for p in path_df["image_path"].astype(str).tolist():
        pp = Path(p)
        m = scan_re.match(pp.name)
        if m is None:
            continue

        case = day = None
        for parent in [pp.parent, pp.parent.parent, pp.parent.parent.parent]:
            md = case_day_re.match(parent.name)
            if md is not None:
                case = int(md.group("case"))
                day = int(md.group("day"))
                break
        if case is None or day is None:
            continue

        rows.append(
            {
                "image_path": p,
                "case": case,
                "day": day,
                "slice": int(m.group("slice")),
                "width": int(m.group("width")),
                "height": int(m.group("height")),
                "spacing": float(m.group("spw")),
            }
        )

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return pd.DataFrame(
            columns=["image_path", "case", "day", "slice", "width", "height", "spacing"]
        )
    return out


def attach_image_paths_by_case_day_slice(
    ids_df: pd.DataFrame, paths_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Bugfix (score-critical): id slice indices are often 1-based (slice_0001..),
    while filename slice indices are often 0-based (0_...png..). A direct merge can
    drop everything. We:
      1) try direct merge on (case,day,slice)
      2) try off-by-one merge (slice-1)
      3) fallback: for each (case,day), map ids to scans by sorted order (k-th id -> k-th file)
    This preserves core inference logic; only restores correct test image mapping.
    """
    out = ids_df.copy()

    merged = out.merge(
        paths_df[["case", "day", "slice", "image_path", "width", "height", "spacing"]],
        on=["case", "day", "slice"],
        how="left",
    )
    out = merged

    need = out["image_path"].isna()
    if need.any():
        tmp = out.loc[need, ["id", "case", "day", "slice"]].copy()
        tmp["slice"] = tmp["slice"] - 1
        tmp = tmp.merge(
            paths_df[
                ["case", "day", "slice", "image_path", "width", "height", "spacing"]
            ],
            on=["case", "day", "slice"],
            how="left",
        ).rename(
            columns={
                "image_path": "image_path_fix",
                "width": "width_fix",
                "height": "height_fix",
                "spacing": "spacing_fix",
            }
        )
        out.loc[need, "image_path"] = tmp["image_path_fix"].values
        out.loc[need, "width"] = tmp["width_fix"].values
        out.loc[need, "height"] = tmp["height_fix"].values
        out.loc[need, "spacing"] = tmp["spacing_fix"].values

    need = out["image_path"].isna()
    if need.any():
        g = paths_df.sort_values(["case", "day", "slice", "image_path"]).groupby(
            ["case", "day"], sort=False
        )
        scan_map = {}
        for (c, d), pdf in g:
            scan_map[(c, d)] = pdf.reset_index(drop=True)

        unresolved = out.loc[need, ["id", "case", "day", "slice"]].copy()
        unresolved["rank"] = (
            unresolved.groupby(["case", "day"])["slice"]
            .rank(method="first")
            .astype(int)
            - 1
        )

        image_path_fix = []
        width_fix = []
        height_fix = []
        spacing_fix = []
        for _, r in unresolved.iterrows():
            key = (int(r["case"]), int(r["day"]))
            k = int(r["rank"])
            pdf = scan_map.get(key, None)
            if pdf is None or k < 0 or k >= len(pdf):
                image_path_fix.append(np.nan)
                width_fix.append(np.nan)
                height_fix.append(np.nan)
                spacing_fix.append(np.nan)
            else:
                image_path_fix.append(pdf.loc[k, "image_path"])
                width_fix.append(pdf.loc[k, "width"])
                height_fix.append(pdf.loc[k, "height"])
                spacing_fix.append(pdf.loc[k, "spacing"])

        unresolved["image_path_fix"] = image_path_fix
        unresolved["width_fix"] = width_fix
        unresolved["height_fix"] = height_fix
        unresolved["spacing_fix"] = spacing_fix

        out.loc[need, "image_path"] = unresolved["image_path_fix"].values
        out.loc[need, "width"] = unresolved["width_fix"].values
        out.loc[need, "height"] = unresolved["height_fix"].values
        out.loc[need, "spacing"] = unresolved["spacing_fix"].values

    return out




## === cell 6
sub_df = pd.read_csv(INPUT_DATA_DIR / "sample_submission.csv")

test_csv_path = INPUT_DATA_DIR / "test.csv"
test_set_hidden = not test_csv_path.exists()

if test_set_hidden:
    test_df = pd.read_csv(INPUT_DATA_DIR / "train.csv")[: 1000 * 3]
    test_df = test_df.drop(columns=["class", "segmentation"]).drop_duplicates()
    image_paths = [str(path) for path in (INPUT_DATA_DIR / "train").rglob("*.png")]
else:
    test_df = pd.read_csv(test_csv_path)
    test_df = test_df.drop(columns=["class"]).drop_duplicates()
    image_paths = [str(path) for path in (INPUT_DATA_DIR / "test").rglob("*.png")]

test_df = extract_metadata_from_id(test_df)

path_df = pd.DataFrame(image_paths, columns=["image_path"])
path_df = extract_metadata_from_path(path_df)

test_df = attach_image_paths_by_case_day_slice(test_df, path_df)

missing = int(test_df["image_path"].isna().sum())
print("test_df:", len(test_df))
print("Missing image_path rows after robust attach:", missing)
print(test_df.head())



## === cell 7
test_df.to_csv("test_preprocessed.csv", index=False)



## === cell 8
td = UWDataset(
    test_df.dropna(subset=["image_path"]).reset_index(drop=True),
    transforms=transforms_val,
)
if len(td) == 0:
    raise RuntimeError(
        "Preprocessing produced an empty test_df after attempting robust mapping. "
        "This indicates test/ scan paths could not be discovered under INPUT_DATA_DIR."
    )
print(td[0]["image"].shape)
print(td[0]["id"], td[0]["h"], td[0]["w"])




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/1015603292.py in <cell line: 0>()
      4 )
      5 if len(td) == 0:
----> 6     raise RuntimeError(
      7         "Preprocessing produced an empty test_df after attempting robust mapping. "
      8         "This indicates test/ scan paths could not be discovered under INPUT_DATA_DIR."

RuntimeError: Preprocessing produced an empty test_df after attempting robust mapping. This indicates test/ scan paths could not be discovered under INPUT_DATA_DIR.

## === cell 9
def discover_checkpoints(
    search_root: Path, exts=(".ckpt", ".pth", ".pt"), max_files: int = 400
) -> List[str]:
    found = []
    if not search_root.exists():
        return found
    for ext in exts:
        for p in search_root.rglob(f"*{ext}"):
            found.append(str(p))
            if len(found) >= max_files:
                return found
    return found


def _score_ckpt_path(p: str) -> Tuple[int, int, int, str]:
    lp = p.lower()
    good = 0
    good += 6 if ("best" in lp) else 0
    good += 4 if ("fold" in lp or "cv" in lp) else 0
    good += 2 if ("dice" in lp or "iou" in lp or "loss" in lp) else 0
    good += 2 if (lp.endswith(".ckpt")) else 0
    try:
        sz = os.path.getsize(p)
    except Exception:
        sz = 0
    return (good, sz, -len(lp), p)


model_pths = ["../input/exp01017/expexp010-bestloss-fold0-7.ckpt"]

exists = os.path.exists(model_pths[0])
print("Checkpoint exists:", exists, model_pths[0])

if not exists:
    candidates = []
    candidates += discover_checkpoints(INPUT_DIR)
    candidates += discover_checkpoints(Path("/kaggle/data"))
    candidates += discover_checkpoints(OUTPUT_DIR)

    seen = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]

    if len(candidates):
        candidates_sorted = sorted(candidates, key=_score_ckpt_path, reverse=True)
        model_pths = [candidates_sorted[0]]
        print("Using discovered checkpoint:", model_pths[0])
        if len(candidates_sorted) > 1:
            print("Other checkpoint candidates (top 5):")
            for c in candidates_sorted[1:6]:
                print(" -", c)
    else:
        print(
            "Warning: no checkpoint found under input/data/working roots. Inference will use randomly initialized weights."
        )




## === cell 10
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


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = torch.nn.functional.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class SimpleUNet(nn.Module):
    def __init__(self, in_channels=5, classes=3, base=32):
        super().__init__()
        self.inc = DoubleConv(in_channels, base)
        self.down1 = Down(base, base * 2)
        self.down2 = Down(base * 2, base * 4)
        self.down3 = Down(base * 4, base * 8)
        self.down4 = Down(base * 8, base * 16)

        self.up1 = Up(base * 16, base * 8, base * 8)
        self.up2 = Up(base * 8, base * 4, base * 4)
        self.up3 = Up(base * 4, base * 2, base * 2)
        self.up4 = Up(base * 2, base, base)

        self.outc = nn.Conv2d(base, classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


def build_model():
    if smp is not None:
        model = smp.Unet(
            encoder_name=ENCODER_NAME,
            encoder_weights=None,
            in_channels=CHANNELS,
            classes=3,
            activation=None,
            decoder_use_batchnorm=True,
            decoder_attention_type="scse",
        )
        return model
    return SimpleUNet(in_channels=CHANNELS, classes=3, base=32)


def _strip_prefix_from_state_dict(
    state_dict, prefixes=("model.", "net.", "module.", "seg_model.")
):
    out = {}
    for k, v in state_dict.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p) :]
        out[nk] = v
    return out


def load_model(path):
    model = build_model()

    if not os.path.exists(path):
        print(
            f"Warning: checkpoint not found at {path}. Using randomly initialized weights."
        )
        model.to(DEVICE).eval()
        return model

    ckpt = torch.load(path, map_location="cpu")
    state = (
        ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    )

    state = _strip_prefix_from_state_dict(state)
    try:
        model.load_state_dict(state, strict=True)
    except Exception as e:
        print(
            f"Warning: strict checkpoint load failed ({type(e).__name__}: {e}). Retrying with strict=False."
        )
        model.load_state_dict(state, strict=False)

    model.to(DEVICE)
    model.eval()
    return model




## === cell 11
def mask2rle(mask: np.ndarray) -> str:
    """
    UW-Madison GI tract expects RLE with pixels numbered top-to-bottom then left-to-right,
    which corresponds to Fortran-order flattening of (H,W) arrays.
    Return '' for empty mask.
    """
    if mask is None:
        return ""
    if mask.max() == 0:
        return ""

    if cp is not None:
        m = cp.array(mask, dtype=cp.uint8)
        pixels = m.T.flatten()
        pad = cp.array([0], dtype=cp.uint8)
        pixels = cp.concatenate([pad, pixels, pad])
        runs = cp.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        runs = cp.asnumpy(runs)
        return " ".join(str(int(x)) for x in runs)

    pixels = mask.astype(np.uint8).T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(int(x)) for x in runs)


def pad_mask(mask):
    padded = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=mask.dtype)
    dh = IMG_SIZE - mask.shape[0]
    dw = IMG_SIZE - mask.shape[1]
    top = dh // 2
    left = dw // 2
    padded[top : top + mask.shape[0], left : left + mask.shape[1], :] = mask
    return padded


def resize_mask(mask, height, width):
    msk = np.zeros((height, width, 3), dtype=mask.dtype)
    msk[:, :, 0] = cv2.resize(
        mask[:, :, 0], (width, height), interpolation=cv2.INTER_NEAREST
    )
    msk[:, :, 1] = cv2.resize(
        mask[:, :, 1], (width, height), interpolation=cv2.INTER_NEAREST
    )
    msk[:, :, 2] = cv2.resize(
        mask[:, :, 2], (width, height), interpolation=cv2.INTER_NEAREST
    )
    return msk


def masks2rles(masks, ids, heights, widths):
    pred_strings = []
    pred_ids = []
    pred_classes = []

    for idx in range(masks.shape[0]):
        mask = pad_mask(masks[idx])  # crop_size -> img_size
        mask = resize_mask(mask, int(heights[idx]), int(widths[idx]))  # -> original

        rle = [None] * 3
        for midx in [0, 1, 2]:
            rle[midx] = mask2rle(mask[..., midx])

        pred_strings.extend(rle)
        pred_ids.extend([ids[idx]] * len(rle))
        pred_classes.extend(["large_bowel", "small_bowel", "stomach"])

    return pred_strings, pred_ids, pred_classes


@torch.no_grad()
def infer(model_paths, thr):
    infer_df = test_df.dropna(subset=["image_path"]).reset_index(drop=True)
    test_set = UWDataset(infer_df, transforms=transforms_val)

    test_dataloader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        drop_last=False,
    )

    pred_strings = []
    pred_ids = []
    pred_classes = []

    models = [load_model(p) for p in model_paths]

    for r in tqdm(test_dataloader):
        imgs, ids, heights, widths = r["image"], r["id"], r["h"], r["w"]
        imgs = imgs.to(DEVICE, dtype=torch.float)

        size = imgs.size()
        masks = torch.zeros(
            (size[0], 3, size[2], size[3]), device=DEVICE, dtype=torch.float32
        )

        for model in models:
            out = model(imgs)
            out = torch.sigmoid(out)
            masks += out / len(models)

        masks = (masks.permute((0, 2, 3, 1)) > thr).to(torch.uint8).cpu().numpy()

        result = masks2rles(masks, ids, heights, widths)
        pred_strings.extend(result[0])
        pred_ids.extend(result[1])
        pred_classes.extend(result[2])

    pred_df = pd.DataFrame(
        {"id": pred_ids, "class": pred_classes, "predicted": pred_strings}
    )
    return pred_df




## === cell 12
pred_df = infer(model_pths, THR)
print(pred_df.head())



## === cell 13
sub_df = pd.read_csv(INPUT_DATA_DIR / "sample_submission.csv").copy()
sub_df.columns = [c.strip() for c in sub_df.columns]

pred_df = pred_df.copy()
pred_df.columns = [c.strip() for c in pred_df.columns]

sub_df = sub_df[["id", "class"]].merge(pred_df, on=["id", "class"], how="left")
sub_df["predicted"] = sub_df["predicted"].fillna("").astype(str)
sub_df = sub_df[["id", "class", "predicted"]]

sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)

print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head(5))
print("submission.csv exists:", os.path.exists(sub_path))
print("Columns:", list(sub_df.columns))
