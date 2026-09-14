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

albumentations==2.0.8
cupy-cuda12x==13.6.0
fastai==2.8.5
more-itertools==10.7.0
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-image==0.25.2
scipy==1.15.3
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124

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

0.8398563920497558

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on missing Kaggle input folders and missing third‑party modules (`segmentation_models_pytorch`, `fast_ai_utils`, `uwm.*`) so the notebook can run in your provided environment. I keep the same inference semantics (fastai `load_learner` + sigmoid thresholding + optional morphology + RLE) but add safe fallbacks when the pretrained `.pkl` model isn’t available, so a valid `submission.csv` is always produced. I also fix the notebook/Jupyter magics (`!cat ...`) and missing imports (`pandas`, fastai `Transform`) that currently stop execution. Finally, I ensure IDs/classes align exactly to `sample_submission.csv` and every row has a (possibly empty) RLE string, which is required for a valid submission.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is coming from submitting (effectively) empty masks because the pretrained fastai `.pkl` isn’t present at `/kaggle/input/uwm-models/…`. To move the score upward with minimal logic change, I (1) auto-discover any available `.pkl` under `/kaggle/input` and load it if found, otherwise keep the empty-mask fallback, (2) fix a key mismatch between `sample_submission` ids and `METADATA` keys by building `METADATA` keyed by the exact `id` format used in the CSV, and (3) ensure RLE encoding uses the competition’s required column-major (Fortran) flatten order (top-to-bottom then left-to-right). These are small, execution-safe fixes that preserve your inference semantics (fastai `load_learner` + sigmoid threshold + morphology + RLE) but turn the pipeline from “mostly missing IDs / empty output” into “real predictions aligned to submission rows,” which should increase the score toward your target.'
- What this solution (achieved 0.0) has done: 'I remove the hard failure when the pretrained fastai `.pkl` isn’t available and instead fall back to a deterministic “empty-mask” predictor so the pipeline always runs end-to-end and writes `submission.csv`. I also make the inference loop robust to that fallback (no `learn.no_bar()` when `learn` is `None`) and ensure `df_preds` always has the required `id/class/predicted` columns even if no predictions were generated (fixes your merge `KeyError: 'id'`). Finally, I keep your existing RLE encoding semantics (including the important transpose/ordering) and preserve all core inference/postprocessing logic when a model is found.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is because the code usually can’t find a trained `.pkl`, so it falls back to predicting empty masks everywhere; that yields essentially zero Dice. To move toward your target with minimal core-logic change, I (1) robustly locate and load the intended fastai `.pkl` by searching both `/kaggle/input` and the provided dataset directory, and (2) make the test image discovery align to the IDs in `sample_submission.csv` so we actually run inference for every required id/class row. I also add a safe CPU-only fallback for RLE encoding when CuPy/GPU isn’t available, keeping the exact same Fortran-order encoding semantics. These are execution- and alignment-fixes that should turn the submission from empty masks into real model predictions, increasing the score substantially toward the target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is caused by submitting empty masks because no trained fastai `.pkl` is being loaded; the smallest change that should move score toward your target is to reliably load a model from your available inputs rather than falling back. I keep your inference pipeline exactly the same (fastai `load_learner` → `get_preds` logits → sigmoid threshold → resize/morphology → RLE), but improve the `.pkl` discovery to prioritize UWM GI-tract models and avoid accidentally picking unrelated `.pkl` files. I also switch `F.sigmoid` to `torch.sigmoid` (numerically equivalent, but avoids deprecated usage) and add a safety check that the loaded model outputs 3 channels; otherwise we still fall back to empty masks to keep submissions valid. These changes directly increase the chance of producing non-empty, correctly-aligned predictions while preserving your core logic and output format.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is coming from (mostly) empty predictions: the inference loop only predicts for the “middle slice” of each 5-slice pack, so most `id` rows in `sample_submission.csv` never get a non-empty mask and remain blank after the merge. I keep your model/inference semantics (fastai `load_learner` + sigmoid threshold + opening + RLE) but change the packing to `step_size=1` with `n_slices_to_merge=3` so the middle slice advances by 1 and covers every slice id exactly once. I also add a tiny safety step that forces exact alignment to the sample submission (deduplicate by `id,class` and reindex), preventing accidental missing rows from silently becoming empty masks. These minimal changes should move the score upward toward your target by producing predictions for all required IDs without altering the core modeling logic.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is because the code typically falls back to empty masks when no fastai `.pkl` is found, so the smallest improvement toward your target is to make model discovery more reliable in *your provided dataset layout* and ensure inference actually covers every test slice id. I (1) search for `.pkl` files under both `/kaggle/input` and `/kaggle/data` (your environment shows data there), prioritizing files that look like UWM GI-tract models, and (2) fix `DATA_DIR` auto-detection so it points to an existing directory rather than assuming only the standard Kaggle path. I also (3) make `METADATA` and `packs` be built from *all actual test scan PNGs* (not just those whose ids appear in sample_submission) and then reindex back to `sample_submission.csv` at the end, preventing silent “missing id -> empty” predictions due to any id/path mismatch. These are minimal changes that preserve your inference logic (fastai `load_learner` → sigmoid threshold → morphology → RLE) but should move the score up from 0.0 by producing real non-empty predictions when a model exists.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with producing mostly-empty masks, which in this code happens whenever the model isn’t loaded (or predictions don’t align to the exact `id` rows in `sample_submission.csv`). I keep your inference pipeline intact (fastai `load_learner` → `get_preds` logits → `sigmoid` threshold → resize → opening → RLE), but make two minimal fixes that directly increase the chance of getting non-empty, correctly-aligned predictions: (1) ensure we infer over *all* test scan PNGs and then reindex to `sample_submission.csv` (so no slice IDs get skipped due to metadata/packs mismatch), and (2) make the `.pkl` discovery prioritize UWM GI-tract models inside your provided dataset paths, avoiding accidental loading of irrelevant `.pkl` files. These changes should move the score upward toward your target without changing architecture/loss/training logic. The script still always write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import logging
import sys
from pathlib import Path

lib_path = Path("/kaggle/input/uwm-libs")
if lib_path.exists():
    for fn in lib_path.iterdir():
        if fn.is_dir():
            sys.path.insert(0, str(fn))
    sys.path.insert(0, str(lib_path))

logging.captureWarnings(True)



## === cell 1
import gc
from dataclasses import dataclass
import re
from collections import defaultdict
from functools import partial

import cv2 as cv
import numpy as np
import pandas as pd
import torch

from fastai.vision.all import (
    Transform,
    ItemTransform,
    TensorImage,
    TensorMask,
    load_learner,
    progress_bar,
    noop,
)

from more_itertools import windowed, chunked
from skimage.morphology import disk
from scipy.ndimage import binary_opening


def on_kaggle() -> bool:
    return Path("/kaggle").exists()


def get_size_from_filename(p: Path):
    parts = p.stem.split("_")
    w = int(parts[0])
    h = int(parts[1])
    return h, w


def get_case_day(s):
    return re.search(r"case\d+_day\d+", str(s)).group()


def get_slice(s, as_number=False):
    m = re.search(r"slice_\d{4}", str(s))
    if m:
        slice_no = m.group()
        return int(slice_no.split("_")[-1]) if as_number else slice_no
    parts = Path(s).stem.split("_")
    if len(parts) >= 3 and parts[2].isdigit():
        return int(parts[2]) if as_number else f"slice_{int(parts[2]):04d}"
    return 0 if as_number else "slice_0000"


def get_sample_id_from_path(p: Path):
    return f"{get_case_day(p)}_{get_slice(p)}"


def get_submission_id_from_path(p: Path):
    """
    IMPORTANT: Submission/test.csv ids are like:
      case123_day20_slice_0001
    so we must build ids in exactly that format.
    """
    return f"{get_case_day(p)}_{get_slice(p)}"


def group_case_day_from_files(image_files):
    groups = defaultdict(list)
    for fn in image_files:
        groups[get_case_day(fn)].append(fn)
    groups = {
        k: sorted(v, key=partial(get_slice, as_number=True)) for k, v in groups.items()
    }
    return groups


def packed(groups, n_slices_to_merge=3, step_size=1):
    assert n_slices_to_merge % 2 != 0
    chunks = []
    for case_day, files in groups.items():
        files = [None] + files + [None]
        for pack in windowed(files, n=n_slices_to_merge, step=step_size):
            pack = list(pack)
            last_not_none = [i for i, x in enumerate(pack) if x is not None][-1]
            if last_not_none != (len(pack) - 1):
                for i in range(last_not_none, len(pack)):
                    pack[i] = pack[last_not_none]
            first_not_none = [i for i, x in enumerate(pack) if x is not None][0]
            if first_not_none != 0:
                for i in range(0, first_not_none):
                    pack[i] = pack[first_not_none]
            chunks.append(pack)
    return chunks




## === cell 2
@dataclass
class Metadata:
    sample_id: str
    full_path: str
    h: int
    w: int

    @classmethod
    def extract(cls, path: Path) -> "Metadata":
        sample_id = get_submission_id_from_path(path)
        img = cv.imread(str(path), cv.IMREAD_UNCHANGED)
        if img is None:
            h, w = get_size_from_filename(path)
        else:
            h, w = img.shape[:2]
        return Metadata(sample_id=sample_id, full_path=str(path), h=int(h), w=int(w))




## === cell 3
CANDIDATE_DATA_DIRS = [
    Path("/kaggle/input/uw-madison-gi-tract-image-segmentation/"),
    Path("/kaggle/data/uw-madison-gi-tract-image-segmentation/"),
    Path("/kaggle/input"),
    Path("/kaggle/data"),
]

DATA_DIR = None
for d in CANDIDATE_DATA_DIRS:
    if (d / "sample_submission.csv").exists() and (d / "test").exists():
        DATA_DIR = d
        break
if DATA_DIR is None:
    DATA_DIR = Path("/kaggle/input/uw-madison-gi-tract-image-segmentation/")

DEBUG = False

sample_sub_path = DATA_DIR / "sample_submission.csv"
test_csv_path = DATA_DIR / "test.csv"
train_csv_path = DATA_DIR / "train.csv"

df_test_ids = pd.read_csv(train_csv_path if DEBUG else sample_sub_path)
TEST_IDS = df_test_ids["id"].drop_duplicates().tolist()

scan_root = DATA_DIR / ("train" if DEBUG else "test")
TEST_FILES = sorted(scan_root.glob("**/scans/*.png"))

METADATA = {m.sample_id: m for m in map(Metadata.extract, TEST_FILES)}

print("DATA_DIR:", DATA_DIR)
print(
    "len(TEST_IDS):",
    len(TEST_IDS),
    "len(TEST_FILES):",
    len(TEST_FILES),
    "len(METADATA):",
    len(METADATA),
)



## === cell 4
model_name = "5ch_e10_step1_bce_dice"




## === cell 5
class CreateSample(Transform):
    def encodes(self, pack):
        imgs = []
        q = 0.01
        for fn in pack:
            img = cv.imread(str(fn), cv.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(f"Failed to read image: {fn}")
            lo, hi = np.percentile(img, [q * 100, (1 - q) * 100])
            img = np.clip(img, lo, hi)
            v_min, v_max = float(np.min(img)), float(np.max(img))
            if v_max > v_min:
                img = (img - v_min) / (v_max - v_min)
            else:
                img = np.zeros_like(img, dtype=np.float32)
            img = (img * 255).astype(np.uint8)
            imgs.append(img)
        merged = np.stack(imgs, axis=-1)  # H,W,C
        return merged


class TensorImageNChannels(TensorImage):
    def show(self, ctx=None, channels=(0, 1, 2), **kwargs):
        assert len(channels) == 3
        visible_image = TensorImage(
            torch.cat([self[..., c, None] for c in channels], dim=-1)
        )
        from fastai.vision.all import show_image

        return show_image(visible_image, ctx=ctx, **kwargs)


class AugBase(ItemTransform):
    def __init__(self, aug):
        self.aug = aug

    def encodes(self, x):
        if len(x) == 2:
            img, mask = x
            result = self.aug(image=img, mask=mask)
            return TensorImageNChannels(result["image"]), TensorMask(result["mask"])
        else:
            (img,) = x
            return (TensorImageNChannels(self.aug(image=img)["image"]),)


class AugValid(AugBase):
    split_idx, order = 1, 2


def valid_aug():
    import albumentations as A

    return AugValid(A.Compose([A.Resize(320, 320), A.CenterCrop(288, 288)]))




## === cell 6
if DEBUG:
    predicted_ids = [
        id_
        for id_ in TEST_IDS
        if id_.startswith("case123_day20") or id_.startswith("case77_day20")
    ]
else:
    predicted_ids = sorted(METADATA.keys())

print("len(predicted_ids):", len(predicted_ids))



## === cell 7
paths = []
missing = 0
for id_ in predicted_ids:
    m = METADATA.get(id_)
    if m is None:
        missing += 1
        continue
    paths.append(Path(m.full_path))

print("missing:", missing, "len(paths):", len(paths))



## === cell 8
packs = packed(group_case_day_from_files(paths), n_slices_to_merge=3, step_size=1)
print("len(packs):", len(packs), "pack size:", len(packs[0]) if packs else None)



## === cell 9
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device



## === cell 10
try:
    import cupy as cp  # type: ignore

    _HAS_CUPY = True
except Exception:
    cp = None
    _HAS_CUPY = False


def mask2rle(mask):
    """
    mask: 2D numpy array, 1 - mask, 0 - background

    IMPORTANT: Competition expects pixels numbered top-to-bottom then left-to-right,
    which corresponds to Fortran/column-major flattening for a (H,W) array.
    """
    mask_u8 = mask.astype(np.uint8)
    if _HAS_CUPY and torch.cuda.is_available():
        mask_c = cp.asarray(mask_u8)
        pixels = mask_c.T.flatten()
        pad = cp.array([0], dtype=pixels.dtype)
        pixels = cp.concatenate([pad, pixels, pad])
        runs = cp.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        if runs.size == 0:
            return ""
        return " ".join(str(int(x)) for x in runs.get().tolist())
    else:
        pixels = mask_u8.T.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        if runs.size == 0:
            return ""
        return " ".join(str(int(x)) for x in runs.tolist())


def rle2mask(mask_rle, shape):
    s = np.asarray(mask_rle.split(), dtype=int)
    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


def pad_mask(mask, image_size):
    padded = np.zeros((image_size, image_size), dtype=mask.dtype)
    dh = image_size - mask.shape[0]
    dw = image_size - mask.shape[1]
    top = dh // 2
    left = dw // 2
    padded[top : top + mask.shape[0], left : left + mask.shape[1]] = mask
    return padded




## === cell 11
model_path = Path(f"/kaggle/input/uwm-models/{model_name}.pkl")


def find_any_fastai_pkl():
    """
    Change: restrict/score candidates to prefer GI-tract/UWM models in the provided dataset roots.
    Why it helps score: reduces chance of loading an unrelated fastai .pkl that outputs wrong channels,
    which otherwise triggers fallback-to-empty (0.0 score).
    """
    roots = [
        Path("/kaggle/input/uwm-models"),
        Path("/kaggle/input"),
        Path("/kaggle/data"),
        DATA_DIR,
        Path("/kaggle/working"),
    ]
    candidates = []
    for root in roots:
        if root.exists():
            candidates.extend(list(root.glob("**/*.pkl")))
    if not candidates:
        return None

    def score_candidate(p: Path) -> tuple:
        name = p.name.lower()
        path = str(p).lower()
        name_match = int(model_name.lower() in name)
        uwm_hint = int(
            ("uwm" in name)
            or ("gi" in name)
            or ("tract" in name)
            or ("bowel" in name)
            or ("stomach" in name)
            or ("madison" in name)
            or ("gi-tract" in path)
            or ("uw-madison" in path)
            or ("gi_tract" in path)
        )
        bad_hint = int(
            ("mnist" in name)
            or ("cifar" in name)
            or ("imagenet" in name)
            or ("bert" in name)
            or ("nlp" in name)
            or ("tabular" in name)
        )
        models_hint = int(("model" in path) or ("models" in path))
        mtime = p.stat().st_mtime if p.exists() else 0.0
        size = p.stat().st_size if p.exists() else 0
        return (name_match, uwm_hint, models_hint, -bad_hint, size, mtime)

    candidates = sorted(candidates, key=score_candidate, reverse=True)
    return candidates[0]


learn = None
chosen_model_path = model_path if model_path.exists() else find_any_fastai_pkl()

if chosen_model_path is not None and chosen_model_path.exists():
    print("Loading model:", chosen_model_path)
    learn = load_learner(str(chosen_model_path), cpu=(device.type == "cpu"))
    learn.dls.to(device)
    learn.eval()
else:
    print(
        "WARNING: No fastai .pkl model found under available inputs. "
        "Proceeding with empty-mask fallback to generate a valid submission.csv."
    )



## === cell 12
preds = []

batch_size = 64
chunks = list(chunked(packs, n=batch_size))

_model_ok = True
if learn is not None and packs:
    try:
        xb = learn.dls.test_dl(
            [packs[0]],
            batch_size=1,
            device=device,
            after_item=[CreateSample(), valid_aug()],
        ).one_batch()[0]
        with torch.no_grad():
            out = learn.model(xb)
        if out.ndim < 4 or out.shape[1] != 3:
            print(
                f"WARNING: Model output shape {tuple(out.shape)} is not (N,3,H,W). Falling back to empty masks."
            )
            _model_ok = False
    except Exception as e:
        print(
            f"WARNING: Could not validate model output; falling back to empty masks. Error: {e}"
        )
        _model_ok = False

if learn is None or not _model_ok or not packs:
    for subset in progress_bar(chunks):
        for pack in subset:
            mid_path = pack[len(pack) // 2]
            test_id = get_submission_id_from_path(mid_path)
            for name in ("large_bowel", "small_bowel", "stomach"):
                preds.append({"id": test_id, "class": name, "predicted": ""})
else:
    with learn.no_bar():
        for subset in progress_bar(chunks):
            test_dl = learn.dls.test_dl(
                subset,
                batch_size=batch_size,
                device=device,
                after_item=[CreateSample(), valid_aug()],
            )
            logits, *_ = learn.get_preds(dl=test_dl, act=noop)

            labels = (
                (torch.sigmoid(logits) >= 0.6).detach().cpu().numpy().astype(np.uint8)
            )

            for pack, mask in zip(subset, labels):
                mid_path = pack[len(pack) // 2]
                test_id = get_submission_id_from_path(mid_path)

                m = METADATA[test_id]
                h, w = m.h, m.w

                for i, name in enumerate(("large_bowel", "small_bowel", "stomach")):
                    cls_mask = pad_mask(mask[i], 320)
                    cls_mask = cv.resize(
                        cls_mask, (w, h), interpolation=cv.INTER_NEAREST
                    )
                    cls_mask = binary_opening(cls_mask, structure=disk(5)).astype(
                        np.uint8
                    )
                    rle = mask2rle(cls_mask)
                    preds.append({"id": test_id, "class": name, "predicted": rle})

            del logits, labels
            gc.collect()

print("len(preds):", len(preds), "example:", (preds[0] if preds else None))



## === cell 13
df_preds = pd.DataFrame(preds, columns=["id", "class", "predicted"])
df_preds.head()



## === cell 14
df_submit = pd.read_csv(sample_sub_path)

df_preds = df_preds.drop_duplicates(subset=["id", "class"], keep="last")
df_preds = df_submit[["id", "class"]].merge(df_preds, on=["id", "class"], how="left")

df_preds["predicted"] = df_preds["predicted"].fillna("").astype(str)

assert df_preds.shape[0] == df_submit.shape[0]
assert list(df_preds.columns) == ["id", "class", "predicted"]

out_path = Path("submission.csv")
df_preds.to_csv(out_path, index=False)
print("Wrote:", out_path.resolve(), "rows:", len(df_preds))
df_preds.head(10)
