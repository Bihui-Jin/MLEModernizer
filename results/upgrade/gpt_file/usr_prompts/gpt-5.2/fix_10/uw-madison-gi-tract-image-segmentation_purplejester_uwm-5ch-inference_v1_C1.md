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

0.8427207142594262

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the metadata parsing so it correctly reads image height/width from the real scan filenames (which start with `slice_...._...png`, not `W_H_...png`) and builds `METADATA` without crashing. I also make model loading robust by falling back to a simple “all-empty masks” submission when the external `uwm-models` dataset is not available, ensuring a valid `submission.csv` is always produced end-to-end. Finally, I correct the image stacking shape in `CreateSample` (H/W were swapped), which is required for consistent inference if a model is present, while keeping the rest of the core inference logic unchanged.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is coming from producing an all-empty submission because the model pickle is missing, so the minimal score-improving change is to ensure a real model is actually loaded at inference time. I keep your inference pipeline (packing slices, fastai `get_preds`, thresholding, resizing, morphology, and RLE) the same, but make the model path discovery robust by searching the available `/kaggle/input` folders for the requested `{model_name}.pkl` and loading it if found. I also add a small sanity check that the produced predictions cover exactly the sample submission ids/classes; if not, we fall back only for the missing rows (still valid submission, but avoids accidental 0.0 from mismatched ids). These changes are directly aimed at moving the score upward toward the target by preventing the empty-mask fallback when a model file exists somewhere in the provided dataset mounts.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score strongly suggests the submission is effectively empty (either because no model is being loaded and you fall back to empty masks, or because the produced `id` values don’t match the competition `id` format and everything becomes blank after merging into `sample_submission.csv`). I make the `id` generation consistent with `test.csv`/`sample_submission.csv` by using the exact `case_day_slice_####` key derived from the scan path (instead of `case_day_slice_0001` without the `slice_` token in `METADATA` but with it in predictions), which prevents the merge from wiping predictions. I also keep your inference core intact (same `packed()`, same fastai `get_preds`, same threshold/morphology), but make `predicted_ids` driven by the actual scan metadata keys so we don’t silently drop most test slices. These minimal fixes should move the score upward toward the target by producing non-empty, correctly-aligned predictions.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with the submission being effectively all-empty after the final merge, which can happen if your predicted `id` strings don’t exactly match `test.csv`/`sample_submission.csv`. I make the `test_id` used during inference come directly from the already-validated `METADATA` key for the center slice (instead of recomputing via `get_sample_id`), eliminating subtle parsing mismatches. I also fix RLE encoding to the competition’s required top-to-bottom then left-to-right order (column-major/Fortran order), which is score-critical for this dataset. Finally, I keep the model/inference pipeline unchanged otherwise, but add a strict post-merge sanity check so we fail fast if ids/classes still don’t align (preventing another silent 0.0).'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with a submission that decodes to (almost) all-empty masks, which can happen even with a loaded model if the threshold/post-processing is too aggressive or the RLE encoding/shape alignment is wrong. I keep your exact inference pipeline (fastai `get_preds`, sigmoid+threshold, padding, resize, binary opening, and RLE) but make two minimal, score-relevant fixes: (1) use a per-class threshold vector (still “thresholding”, not a new method) that is less aggressive for typically small/low-confidence classes, and (2) avoid over-eroding small predicted regions by reducing the opening kernel size slightly (still the same morphological opening step). I also ensure we never silently drop rows by building `test_id` directly from the already-built `METADATA` for the center slice path, and I keep the strict post-merge sanity check so we don’t regress back to an all-empty submission.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with producing (nearly) all-empty masks due to a too-high threshold and too-strong morphology, even when the model loads and ids align. I keep your exact inference pipeline (fastai `get_preds`, sigmoid, thresholding, pad->resize, binary opening, RLE) but make minimal post-processing adjustments that are known to recover small bowel/stomach regions: slightly lower per-class thresholds and a smaller opening radius to avoid erasing thin structures. I also stop re-parsing the center slice id during inference (which can still subtly mismatch) and instead use the already-built `METADATA` mapping for the center slice path to guarantee exact `id` strings. These changes should move the score up from 0.0 toward the target by producing non-empty, correctly-aligned masks without changing the model or training logic.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with a submission that decodes incorrectly (bad RLE orientation) and/or predictions being wiped by subtle `id` mismatches. I keep your inference core the same, but (1) stop re-parsing `test_id` during inference and instead use the already-built `METADATA` mapping to guarantee exact id alignment, and (2) make RLE encoding explicitly match the competition requirement (top-to-bottom then left-to-right) by encoding in row-major order. These are minimal, score-critical correctness fixes that should move your score up toward the target without changing the model, training, or overall post-processing steps. I also add a strict post-merge assertion that predicted ids/classes cover the expected set (fail fast rather than silently scoring 0.0).'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most likely caused by an incorrect RLE encoding order: this competition expects pixels numbered top-to-bottom then left-to-right, which corresponds to Fortran-order flattening of the transposed mask (or equivalently flattening with `order="F"` on the transposed array), not simple C-order flattening. I make the minimal, score-critical fix by changing `mask2rle`/`rle2mask` to the canonical UW-Madison RLE convention while leaving your model, inference, thresholds, resizing, and morphology unchanged. I also ensure the mask is strictly binary (0/1) before encoding to avoid any unexpected behavior. These changes should move the score up from 0.0 toward your target without altering the core approach.'

# 9. Code solution

## === cell 0
import logging
import sys
from pathlib import Path

logging.captureWarnings(True)

LIB_PATH = Path("/kaggle/input/uwm-libs")
if LIB_PATH.exists():
    for fn in LIB_PATH.iterdir():
        if fn.is_dir():
            sys.path.insert(0, str(fn))
    sys.path.insert(0, str(LIB_PATH))



## === cell 1
import gc
import re
from dataclasses import dataclass
from collections import defaultdict
from functools import partial

import albumentations as A
import cv2 as cv
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_opening
from skimage.morphology import disk
from more_itertools import windowed, chunked

from fastai.vision.all import (
    Transform,
    ItemTransform,
    TensorImage,
    TensorMask,
    show_image,
    show_images,
    get_image_files,
    load_learner,
    progress_bar,
    noop,
)




## === cell 2
def on_kaggle() -> bool:
    return Path("/kaggle").exists()


def get_case_day(s) -> str:
    return re.search(r"case\d+_day\d+", str(s)).group()


def get_slice(s, as_number: bool = False):
    m = re.search(r"slice_\d{4}", str(s))
    if m is None:
        raise ValueError(f"Could not parse slice_#### from path: {s}")
    slice_no = m.group()
    return int(slice_no.split("_")[-1]) if as_number else slice_no


def get_sample_id(s) -> str:
    """
    Competition id format is 'caseXXX_dayYY_slice_####' (matches test.csv).
    """
    return f"{get_case_day(s)}_{get_slice(s)}"


def get_size_from_scan_png(p: Path):
    """
    scan filenames are like: slice_0001_266_266_1.50_1.50.png
    Parse the two integer tokens after the slice number as width/height.
    """
    parts = p.stem.split("_")
    if len(parts) < 4 or parts[0] != "slice":
        raise ValueError(f"Unexpected scan filename: {p.name}")
    try:
        w = int(parts[2])
        h = int(parts[3])
    except Exception as e:
        raise ValueError(f"Could not parse W/H from scan filename: {p.name}") from e
    return (h, w)


def group_case_day_from_files(image_files):
    groups = defaultdict(list)
    for fn in image_files:
        groups[get_case_day(fn)].append(fn)
    groups = {
        k: sorted(v, key=partial(get_slice, as_number=True)) for k, v in groups.items()
    }
    return groups


def packed(groups, n_slices_to_merge=3, step_size=2):
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




## === cell 3
@dataclass
class Metadata:
    sample_id: str
    full_path: str
    h: int
    w: int

    @classmethod
    def extract(cls, path: Path) -> "Metadata":
        """
        Ensure sample_id matches test.csv exactly: 'caseXXX_dayYY_slice_####'
        """
        case_and_day = path.parents[1].stem  # "caseXXX_dayYY"
        slice_no = get_slice(path, as_number=True)
        h, w = get_size_from_scan_png(path)
        sample_id = f"{case_and_day}_slice_{int(slice_no):04d}"
        return Metadata(sample_id, str(path), int(h), int(w))




## === cell 4
DATA_DIR = Path("/kaggle/input/uw-madison-gi-tract-image-segmentation/")

DEBUG = False

TEST_IDS = (
    pd.read_csv(DATA_DIR / ("train.csv" if DEBUG else "test.csv"))["id"]
    .drop_duplicates()
    .tolist()
)

TEST_FILES = get_image_files(DATA_DIR / ("train" if DEBUG else "test"))

_meta_list = [Metadata.extract(p) for p in TEST_FILES]
METADATA = {m.sample_id: m for m in _meta_list}

print("n unique ids in csv:", len(TEST_IDS))
print("n slice metadata from files:", len(METADATA))

csv_set = set(TEST_IDS)
meta_set = set(METADATA.keys())
overlap = len(csv_set & meta_set)
print("id overlap (csv ∩ metadata):", overlap, "/", len(csv_set))
if overlap == 0:
    raise RuntimeError(
        "No overlap between test.csv ids and parsed scan ids; predictions would all be blank."
    )



## === cell 5
model_name = "5ch_e10_step1_bce_dice"




## === cell 6
class CreateSample(Transform):
    def encodes(self, pack):
        h, w = get_size_from_scan_png(pack[0])
        merged = np.ndarray((h, w, len(pack)), dtype=np.uint8)
        q = 0.01
        for i, fn in enumerate(pack):
            img = cv.imread(str(fn), cv.IMREAD_UNCHANGED)
            lo, hi = np.percentile(img, [q * 100, (1 - q) * 100])
            img = np.clip(img, lo, hi)
            v_min, v_max = np.min(img), np.max(img)
            img = (img - v_min) / float(v_max - v_min + 1e-8)
            img = (img * 255).astype(np.uint8)
            merged[:, :, i] = img
        return merged


class TensorImageNChannels(TensorImage):
    def show(self, ctx=None, channels=(0, 1, 2), **kwargs):
        assert len(channels) == 3
        visible_image = TensorImage(
            torch.cat([self[..., c, None] for c in channels], dim=-1)
        )
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


class AugTrain(AugBase):
    split_idx, order = 0, 2


class AugValid(AugBase):
    split_idx, order = 1, 2


class ChannelsFirst(ItemTransform):
    def encodes(self, x):
        return tuple(t.permute(0, 3, 1, 2) for t in x)

    def decodes(self, x):
        return tuple(t.permute(0, 2, 3, 1) for t in x)


class FloatMask(Transform):
    order = 99

    def encodes(self, x: TensorMask):
        return TensorImage(x.float())

    def decodes(self, x: TensorMask):
        return TensorMask(x.long())


def train_aug():
    return AugTrain(
        A.Compose(
            [
                A.Resize(320, 320),
                A.CoarseDropout(
                    min_holes=1,
                    max_holes=8,
                    min_height=4,
                    max_height=288 // 10,
                    min_width=4,
                    max_width=288 // 10,
                    mask_fill_value=0,
                    p=0.1,
                ),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.2,
                    rotate_limit=25,
                    interpolation=cv.INTER_AREA,
                    p=0.2,
                ),
                A.RandomCrop(288, 288),
                A.OneOf(
                    [
                        A.HorizontalFlip(p=1),
                        A.VerticalFlip(p=0.3),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.MotionBlur(p=0.2),
                        A.MedianBlur(p=0.2),
                        A.Blur(blur_limit=1, p=0.1),
                    ],
                    p=0.2,
                ),
                A.Perspective(p=0.3),
                A.GaussNoise(var_limit=0.001, p=0.2),
                A.OneOf(
                    [
                        A.OpticalDistortion(p=0.3),
                        A.GridDistortion(p=0.2),
                        A.PiecewiseAffine(p=0.3),
                    ],
                    p=0.2,
                ),
                A.OneOf(
                    [
                        A.Sharpen(p=0.1),
                        A.Emboss(p=0.1),
                        A.RandomBrightnessContrast(p=0.1),
                    ]
                ),
                A.Cutout(p=0.3),
            ]
        )
    )


def valid_aug():
    return AugValid(A.Compose([A.Resize(320, 320), A.CenterCrop(288, 288)]))




## === cell 7
def find_model_pkl(model_name: str) -> Path | None:
    expected = f"{model_name}.pkl"
    p0 = Path(f"/kaggle/input/uwm-models/{expected}")
    if p0.exists():
        return p0

    base = Path("/kaggle/input")
    if not base.exists():
        return None

    candidates = []
    for d in base.iterdir():
        if d.is_dir():
            p = d / expected
            if p.exists():
                candidates.append(p)
            p2 = d / "models" / expected
            if p2.exists():
                candidates.append(p2)

    if candidates:
        return sorted(candidates, key=lambda x: len(str(x)))[0]

    try:
        for p in base.rglob(expected):
            if p.is_file():
                return p
    except Exception:
        return None

    return None


learn = None
learn_path = find_model_pkl(model_name)
if learn_path is not None and learn_path.exists():
    learn = load_learner(learn_path)
    print("Loaded model:", learn_path)
else:
    print(
        "WARNING: Model file not found anywhere under /kaggle/input; will create an empty-mask submission."
    )
    print("Tried model_name:", model_name)



## === cell 8
if DEBUG:
    predicted_ids = [id_ for id_ in TEST_IDS if id_.startswith("case123_day20")]
else:
    predicted_ids = [id_ for id_ in TEST_IDS if id_ in METADATA]

print(
    "n predicted ids (csv filtered by metadata):",
    len(predicted_ids),
    "/",
    len(TEST_IDS),
)
if len(predicted_ids) == 0:
    raise RuntimeError("No test ids found in METADATA; cannot run inference.")



## === cell 9
paths = [Path(METADATA[id_].full_path) for id_ in predicted_ids]
packs = packed(group_case_day_from_files(paths), n_slices_to_merge=5, step_size=1)
print("n packs:", len(packs))



## === cell 10
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device




## === cell 11
def mask2rle(mask: np.ndarray) -> str:
    """
    Score-critical fix: UW-Madison GI Tract uses RLE with pixels counted
    top-to-bottom then left-to-right, which corresponds to flattening the
    transpose in Fortran order (common Kaggle convention for this dataset).

    Minimal change: only fix encoding order; keep downstream semantics identical.
    """
    mask = (mask > 0).astype(np.uint8)
    pixels = mask.T.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]]).astype(np.uint8)
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle2mask(mask_rle: str, shape):
    s = np.asarray(mask_rle.split(), dtype=int)
    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape((shape[1], shape[0]), order="F").T


def pad_mask(mask, image_size):
    padded = np.zeros((image_size, image_size), dtype=mask.dtype)
    dh = image_size - mask.shape[0]
    dw = image_size - mask.shape[1]
    top = dh // 2
    left = dw // 2
    padded[top : top + mask.shape[0], left : left + mask.shape[1]] = mask
    return padded




## === cell 12
CLASS_NAMES = ("large_bowel", "small_bowel", "stomach")

CLASS_THRESH = {
    "large_bowel": 0.35,
    "small_bowel": 0.30,
    "stomach": 0.30,
}

OPENING_RADIUS = 2

preds = []

if learn is not None:
    learn.dls.to(device)
    learn.model.to(device)
    learn.eval()

    batch_size = 64
    chunks = list(chunked(packs, n=batch_size))

    with learn.no_bar(), torch.no_grad():
        for subset in progress_bar(chunks):
            test_dl = learn.dls.test_dl(subset, batch_size=batch_size, device=device)

            logits, *_ = learn.get_preds(dl=test_dl, act=noop)
            probs = torch.sigmoid(logits).cpu().numpy()  # (bs, 3, H, W)

            for pack, prob in zip(subset, probs):
                center_path = Path(pack[len(pack) // 2])
                center_id = get_sample_id(center_path)  # should match METADATA keys
                m = METADATA.get(center_id, None)
                if m is None:
                    continue
                test_id = m.sample_id  # canonical id exactly as in test.csv

                h, w = cv.imread(m.full_path, cv.IMREAD_UNCHANGED).shape[:2]

                for i, name in enumerate(CLASS_NAMES):
                    thr = CLASS_THRESH[name]
                    cls_mask = (prob[i] >= thr).astype(np.uint8)

                    cls_mask = pad_mask(cls_mask, 320)
                    cls_mask = cv.resize(cls_mask, (w, h), cv.INTER_NEAREST)
                    cls_mask = binary_opening(
                        cls_mask.astype(bool), structure=disk(OPENING_RADIUS)
                    ).astype(np.uint8)

                    rle = mask2rle(cls_mask)
                    preds.append({"id": test_id, "class": name, "predicted": rle})

            del logits, probs
            gc.collect()
else:
    test_df = pd.read_csv(DATA_DIR / ("train.csv" if DEBUG else "test.csv"))
    preds = [
        {"id": r.id, "class": r["class"], "predicted": ""}
        for _, r in test_df.iterrows()
    ]

print("n pred rows:", len(preds))



## === cell 13
df_preds = pd.DataFrame(preds)
print(df_preds.head())



## === cell 14
df_submit = pd.read_csv(DATA_DIR / "sample_submission.csv")

df_preds = df_preds.drop_duplicates(subset=["id", "class"], keep="first")

if learn is not None:
    bad_ids = set(df_preds["id"].unique()) - set(df_submit["id"].unique())
    if len(bad_ids) > 0:
        raise RuntimeError(
            f"Predictions contain ids not present in sample_submission/test.csv (n={len(bad_ids)}). "
            "This will cause merge/drop and likely score ~0.0. Example: "
            + ", ".join(list(sorted(bad_ids))[:5])
        )

df_submit = df_submit.drop(columns="predicted").merge(
    df_preds, on=["id", "class"], how="left"
)
df_submit["predicted"] = df_submit["predicted"].fillna("")

if learn is not None:
    frac_non_empty = (df_submit["predicted"].str.len() > 0).mean()
    print("fraction non-empty RLE rows (post-merge):", frac_non_empty)
    if frac_non_empty == 0.0:
        raise RuntimeError(
            "Post-merge submission is entirely empty despite having a model; "
            "this indicates an id/class alignment issue that would score ~0.0."
        )

out_path = Path("submission.csv")
df_submit.to_csv(out_path, index=False)

print("Wrote:", out_path.resolve())
print("shape:", df_submit.shape)
print("columns:", df_submit.columns.tolist())
print(df_submit.head(10))

assert out_path.suffix == ".csv"
assert list(df_submit.columns) == ["id", "class", "predicted"]
assert len(df_submit) == len(pd.read_csv(DATA_DIR / "sample_submission.csv"))

non_empty = (df_submit["predicted"].str.len() > 0).mean()
print("fraction non-empty RLE rows:", non_empty)
