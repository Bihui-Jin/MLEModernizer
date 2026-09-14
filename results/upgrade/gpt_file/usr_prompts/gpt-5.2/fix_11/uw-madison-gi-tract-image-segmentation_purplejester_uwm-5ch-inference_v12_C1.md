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

0.8582719138113138

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the metadata/id parsing so it matches the dataset’s actual filename pattern (`slice_XXXX_H_W_psX_psY.png`) and directory layout, which currently prevents building the test-file index and breaks the inference pipeline. I also make the model-loading step robust: if the external `.pkl` is missing (as in your error), the script still run end-to-end by emitting a valid all-empty-mask submission (score be low but valid), rather than crashing. Finally, I ensure the inferred IDs align exactly with `sample_submission.csv` IDs and that the script always writes `submission.csv` with the required columns and 20400 rows.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is coming from producing an all-empty submission because the model file isn’t found at `/kaggle/input/uwm-models/dataset_norm_e10_e15.pkl`. The smallest change that moves the score toward your target is to (1) automatically discover and load an available `.pkl` model from `/kaggle/input` (including nested folders) instead of hard-failing into empty masks, and (2) keep the rest of your inference/RLE pipeline identical so evaluation semantics don’t change. If no model is found, the script still fall back to a valid empty submission (so it always runs end-to-end). This should immediately improve the score from 0.0 to a non-trivial value (assuming any valid trained `.pkl` exists in your Kaggle inputs).'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is coming from an all-empty submission, which happens whenever no `.pkl` is found/loaded. The smallest legitimate improvement toward your target is to (1) reliably find and load a learner `.pkl` from `/kaggle/input` (including common fastai export patterns), and (2) ensure inference uses the same preprocessing expected by the exported model (adding `valid_aug()` and `NormalizeSample()` to the test `dl` only if they are not already present in the exported `dls`). These changes keep your model/inference core logic intact while turning the submission from “empty masks” into real predictions, which should move the score upward substantially if any valid model exists. If no model is found, the script still produces a valid `submission.csv` with the correct schema.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with silently producing empty masks whenever the `.pkl` isn’t found/loaded (or when predictions don’t align back to the 20400 `id,class` rows). I keep your model/inference pipeline intact, but make model discovery more reliable (also checking `/kaggle/input` and `/kaggle/data`), and ensure test-time DataLoader applies the exact same item transforms your `test_dl` expects (including `CreateSample` and `ChannelsFirst`) so the learner actually receives the right tensor shape. Finally, I fix the pack-to-id mapping so predictions are keyed by the center slice id that exactly matches `sample_submission.csv`, and add a hard assertion that all 20400 rows are present after merge (preventing accidental mass-NaNs that become empty RLEs). These minimal changes should move the score upward toward your target when a valid exported fastai learner exists in the environment, while still producing a valid `submission.csv` even if no model is available.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is caused by generating an all-empty submission when no exported fastai model is found/loaded, so the smallest improvement toward your target is to reliably locate and load a usable exported learner from the available dataset folder tree (including `/kaggle/input/.../uw-madison-gi-tract-image-segmentation/`). I keep your inference/RLE pipeline and thresholding the same, but make model discovery more robust and add a safety check that the loaded object is actually a fastai `Learner` with a model before running inference (otherwise we still fall back to a valid empty submission). I also ensure the test DataLoader always applies `CreateSample -> valid_aug -> ChannelsFirst -> NormalizeSample` in the correct order when missing, because a mismatch here can silently produce near-empty masks even with a model. These changes are minimal, keep your core logic intact, and should move the score upward substantially if any valid `.pkl` exists in the environment.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is coming from producing an all-empty submission because no usable exported fastai learner is being found/loaded at runtime, so the smallest improvement toward the target is to reliably locate and load a valid `.pkl` (if present) rather than falling back. I keep your entire preprocessing/inference/RLE logic the same, but tighten model discovery to prioritize likely UW-Madison GI exported learners and verify the loaded object is actually a `Learner`. I also add a deterministic safety check to ensure we’re not silently using an empty/invalid model and that we still always write a valid `submission.csv`. These changes should move the score upward from 0.0 toward your target whenever any valid learner export exists in the available Kaggle inputs, without changing your model architecture or inference semantics.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly because you’re not loading any real trained model (so you submit empty masks), or you’re loading a learner but the test DataLoader transforms don’t match what the exported model expects (leading to near-empty outputs). I keep your inference/RLE logic intact, but make model discovery load the *best available* exported fastai learner by actually validating candidate `.pkl` files instead of picking a path heuristically. Then I force the test `after_item` pipeline to include the required transforms in a safe way that doesn’t depend on fragile `dls.after_item` introspection (so the model receives correctly shaped/normalized tensors). These are minimal changes aimed at moving the score upward from 0.0 toward your target, while still falling back to a valid empty submission if absolutely no usable model exists.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with your `HAS_MODEL=False` branch (all-empty masks) or with a silent transform mismatch that makes the model output near-empty masks; the smallest meaningful step toward your target is to ensure a real model is actually found/loaded and that test-time preprocessing matches what the learner expects. I keep your model/inference/RLE logic intact, but (1) improve `.pkl` discovery by also checking common fastai export locations under the competition dataset folder and validating candidates, and (2) only override `after_item` for `test_dl` when necessary (otherwise use the learner’s own `after_item`), which avoids accidentally double-applying resize/crop/normalization and producing empty outputs. I also add hard checks to prevent producing an all-empty submission unnoticed when a model is loaded, while still guaranteeing a valid `submission.csv` is written. These are minimal changes aimed specifically at moving the score up from 0.0 toward your target without altering architecture/loss/training loops.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with producing mostly/entirely empty masks due to a test-time transform mismatch: you only pass `after_item` when `learn.dls.after_item` is empty, but exported fastai learners often have an `after_item` that is non-empty yet *not compatible* with your `packs` input (so the model receives wrong-shaped tensors and outputs near-empty masks). I make a minimal change to always use your explicit test-time `after_item` pipeline for `test_dl` (CreateSample → valid_aug → ChannelsFirst → NormalizeSample), preserving your core model/inference logic while ensuring consistent preprocessing. I also add a tiny “sanity check” to detect a near-all-empty output when a model is loaded, and in that case retry once with the explicit pipeline (still minimal and deterministic). This should move the score upward toward your target without changing architecture/loss/training loops, and the script still always write a valid `submission.csv`.'

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
from pathlib import Path

import albumentations as A
import cv2 as cv
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from fastai.vision.all import (
    Transform,
    ItemTransform,
    TensorImage,
    TensorMask,
    show_image,
    show_images,
    get_image_files,
    progress_bar,
    load_learner,
    noop,
)
from fastai.learner import Learner


def on_kaggle() -> bool:
    return Path("/kaggle").exists()


def equalize(img: np.ndarray) -> np.ndarray:
    if img is None:
        return img
    if img.dtype != np.uint8:
        return img
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)




## === cell 2
import re


@dataclass
class Metadata:
    sample_id: str
    full_path: str
    h: int
    w: int

    @classmethod
    def extract(cls, path: Path) -> "Metadata":
        case_and_day = path.parents[1].stem  # "case110_day12"
        m = re.match(r"slice_(\d+)_([0-9]+)_([0-9]+)_.+", path.stem)
        if m is None:
            m2 = re.search(r"slice_(\d+)", path.stem)
            if m2 is None:
                raise ValueError(f"Unrecognized scan filename pattern: {path.name}")
            slice_no = int(m2.group(1))
            h = -1
            w = -1
        else:
            slice_no = int(m.group(1))
            h = int(m.group(2))
            w = int(m.group(3))

        sample_id = f"{case_and_day}_slice_{slice_no:04d}"
        return Metadata(sample_id=sample_id, full_path=str(path), h=h, w=w)




## === cell 3
import re
from collections import defaultdict
from functools import partial
from more_itertools import windowed


def get_size(s: Path):
    img = cv.imread(str(s), cv.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {s}")
    h, w = img.shape[:2]
    return (h, w)


def get_case_day(s):
    return re.search(r"case\d+_day\d+", str(s)).group()


def get_slice(s, as_number=False):
    m = re.search(r"slice_\d{4}", str(s))
    if m:
        slice_no = m.group()
        return int(slice_no.split("_")[-1]) if as_number else slice_no
    m2 = re.search(r"slice_(\d+)", str(Path(s).stem))
    if m2:
        slice_no = int(m2.group(1))
        return slice_no if as_number else f"slice_{slice_no:04d}"
    slice_no = int(Path(s).stem)
    return slice_no if as_number else f"slice_{slice_no:04d}"


def get_sample_id(s):
    return f"{get_case_day(s)}_{get_slice(s)}"


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




## === cell 4
DATA_DIR = Path("/kaggle/input/uw-madison-gi-tract-image-segmentation/")

DEBUG = False

TEST_IDS = (
    pd.read_csv(DATA_DIR / "sample_submission.csv")["id"].drop_duplicates().tolist()
)

TEST_FILES = get_image_files(DATA_DIR / ("train" if DEBUG else "test"))

METADATA = {m.sample_id: m for m in TEST_FILES.map(Metadata.extract)}

len(TEST_IDS), len(TEST_FILES), len(METADATA)




## === cell 5
class CreateSample(Transform):
    def encodes(self, pack):
        h, w = get_size(pack[0])
        merged = np.ndarray((h, w, len(pack)), dtype=np.uint8)
        q = 0.01
        for i, fn in enumerate(pack):
            img = cv.imread(str(fn), cv.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(f"Failed to read image: {fn}")
            lo, hi = np.percentile(img, [q * 100, (1 - q) * 100])
            img = np.clip(img, lo, hi)
            v_min, v_max = float(np.min(img)), float(np.max(img))
            if v_max > v_min:
                img = (img - v_min) / (v_max - v_min)
            else:
                img = img * 0.0
            img = (img * 255).astype(np.uint8)
            merged[:, :, i] = img
        return merged


class CreateTarget(Transform):
    def __init__(self, codes=(1, 2, 3)):
        super().__init__()
        self.codes = codes

    def encodes(self, pack):
        raise NotImplementedError("CreateTarget is not used for test-time inference.")


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


class NormalizeSample(Transform):
    order = 99

    def setups(self, *args, **kwargs):
        self.mean, self.std = 0.18161897, 0.257913

    def encodes(self, x: TensorImageNChannels):
        return (x - self.mean) / self.std

    def decodes(self, x: TensorImageNChannels):
        return x * self.std + self.mean


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




## === cell 6
def find_and_load_learner(preferred_stem: str) -> tuple[Learner | None, Path | None]:
    preferred = Path(f"/kaggle/input/uwm-models/{preferred_stem}.pkl")

    roots = [
        Path("/kaggle/input"),
        Path("/kaggle/data"),
        Path("/kaggle/working"),
        Path("/kaggle/input/uw-madison-gi-tract-image-segmentation"),
        Path(
            "/kaggle/input/uw-madison-gi-tract-image-segmentation/uw-madison-gi-tract-image-segmentation"
        ),
        Path("/kaggle/working/uw-madison-gi-tract-image-segmentation"),
        Path(
            "/kaggle/working/uw-madison-gi-tract-image-segmentation/uw-madison-gi-tract-image-segmentation"
        ),
    ]

    candidates: list[Path] = []
    if preferred.exists():
        candidates.append(preferred)

    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.pkl"):
            if p not in candidates:
                candidates.append(p)

    if not candidates:
        return None, None

    def rank_key(p: Path):
        s = p.as_posix().lower()
        stem_match = int(p.stem == preferred_stem)
        kw_good = [
            "export",
            "learner",
            "fastai",
            "uw",
            "uwm",
            "madison",
            "gi-tract",
            "gi_tract",
            "unet",
            "seg",
            "model",
        ]
        kw_bad = ["fold", "oof", "tmp", "debug", "cache"]
        good = sum(k in s for k in kw_good)
        bad = sum(k in s for k in kw_bad)
        return (-stem_match, -good, bad, len(s))

    candidates = sorted(candidates, key=rank_key)

    max_tries = min(30, len(candidates))
    for p in candidates[:max_tries]:
        try:
            obj = load_learner(p, cpu=not torch.cuda.is_available())
            if (
                isinstance(obj, Learner)
                and hasattr(obj, "model")
                and obj.model is not None
            ):
                return obj, p
        except Exception:
            continue

    return None, None


model_name = "dataset_norm_e10_e15"
learn, model_path = find_and_load_learner(model_name)
HAS_MODEL = learn is not None

if HAS_MODEL:
    print(f"Loaded Learner from: {model_path}")
else:
    print(
        "WARNING: No usable exported fastai Learner (.pkl) could be loaded from /kaggle/input, /kaggle/data, or /kaggle/working. "
        "Proceeding with an all-empty-mask submission (valid format, low score)."
    )



## === cell 7
if DEBUG:
    predicted_ids = [
        id_
        for id_ in TEST_IDS
        if id_.startswith("case123_day20") or id_.startswith("case77_day20")
    ]
else:
    predicted_ids = TEST_IDS

missing = [id_ for id_ in predicted_ids if id_ not in METADATA]
if missing:
    example_files = sorted([Path(p).name for p in TEST_FILES[:5]])
    raise KeyError(
        f"{len(missing)} ids not found in METADATA. Example missing id: {missing[0]}. "
        f"Example scan filenames: {example_files}"
    )

paths = [Path(METADATA[id_].full_path) for id_ in predicted_ids]
len(paths)



## === cell 8
packs = packed(group_case_day_from_files(paths), n_slices_to_merge=5, step_size=1)
len(packs), packs[0][len(packs[0]) // 2]



## === cell 9
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if HAS_MODEL:
    learn.dls.to(device)
    learn.model.to(device)
    learn.eval()
device




## === cell 10
def mask2rle(mask: np.ndarray) -> str:
    """
    mask: 2D numpy array, 1 - mask, 0 - background
    Returns run-length as string formatted: start length start length ...
    """
    if mask is None:
        return ""
    mask = (mask > 0).astype(np.uint8)
    pixels = mask.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
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
    return img.reshape(shape, order="F")


def pad_mask(mask, image_size):
    padded = np.zeros((image_size, image_size), dtype=mask.dtype)
    dh = image_size - mask.shape[0]
    dw = image_size - mask.shape[1]
    top = dh // 2
    left = dw // 2
    padded[top : top + mask.shape[0], left : left + mask.shape[1]] = mask
    return padded




## === cell 11
from more_itertools import chunked

preds = []

if HAS_MODEL:
    batch_size = 64
    chunks = list(chunked(packs, n=batch_size))

    explicit_after_item = [
        CreateSample(),
        valid_aug(),
        ChannelsFirst(),
        NormalizeSample(),
    ]

    def run_inference(use_explicit: bool):
        _preds = []
        with learn.no_bar():
            for subset in progress_bar(chunks):
                test_dl = learn.dls.test_dl(
                    subset,
                    batch_size=batch_size,
                    device=device,
                    after_item=(explicit_after_item if use_explicit else None),
                )
                logits, *_ = learn.get_preds(dl=test_dl, act=noop)

                labels = (
                    (torch.sigmoid(logits) >= 0.4)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )

                for pack, mask in zip(subset, labels):
                    center_path = pack[len(pack) // 2]
                    test_id = get_sample_id(center_path)
                    if test_id not in METADATA:
                        raise KeyError(
                            f"Pack center id not found in METADATA: {test_id}"
                        )
                    m = METADATA[test_id]

                    img_hw = cv.imread(m.full_path, cv.IMREAD_UNCHANGED)
                    if img_hw is None:
                        raise FileNotFoundError(f"Failed to read: {m.full_path}")
                    h, w = img_hw.shape[:2]

                    for i, name in enumerate(("large_bowel", "small_bowel", "stomach")):
                        cls_mask = pad_mask(mask[i], 320)
                        cls_mask = cv.resize(cls_mask, (w, h), cv.INTER_NEAREST)
                        rle = mask2rle(cls_mask)
                        _preds.append({"id": test_id, "class": name, "predicted": rle})

                del logits, labels, test_dl
                gc.collect()
        return _preds

    preds = run_inference(use_explicit=True)

    non_empty = sum(
        1 for d in preds if isinstance(d["predicted"], str) and len(d["predicted"]) > 0
    )
    frac_non_empty = non_empty / max(1, len(preds))
    print(
        f"Non-empty RLE rows (pass1 explicit_after_item=True): {non_empty} / {len(preds)} ({frac_non_empty:.6f})"
    )

    if frac_non_empty < 1e-4:
        preds2 = run_inference(use_explicit=False)
        non_empty2 = sum(
            1
            for d in preds2
            if isinstance(d["predicted"], str) and len(d["predicted"]) > 0
        )
        frac_non_empty2 = non_empty2 / max(1, len(preds2))
        print(
            f"Non-empty RLE rows (pass2 explicit_after_item=False): {non_empty2} / {len(preds2)} ({frac_non_empty2:.6f})"
        )
        if frac_non_empty2 > frac_non_empty:
            preds = preds2

else:
    for test_id in predicted_ids:
        for name in ("large_bowel", "small_bowel", "stomach"):
            preds.append({"id": test_id, "class": name, "predicted": ""})

df_preds = pd.DataFrame(preds)
df_preds.head()



## === cell 12
df_submit = pd.read_csv(DATA_DIR / "sample_submission.csv")

df_submit = df_submit.drop(columns="predicted").merge(
    df_preds, on=["id", "class"], how="left", validate="one_to_one"
)

missing_rows = int(df_submit["predicted"].isna().sum())
if missing_rows:
    print(
        f"WARNING: {missing_rows} rows missing predictions after merge; filling with empty RLE."
    )
df_submit["predicted"] = df_submit["predicted"].fillna("")

assert len(df_submit) == 20400
assert int(df_submit["predicted"].isna().sum()) == 0

out_path = Path("submission.csv")
df_submit.to_csv(out_path, index=False)

check = pd.read_csv(out_path)
assert list(check.columns) == ["id", "class", "predicted"]
assert len(check) == 20400
check.head(10)
