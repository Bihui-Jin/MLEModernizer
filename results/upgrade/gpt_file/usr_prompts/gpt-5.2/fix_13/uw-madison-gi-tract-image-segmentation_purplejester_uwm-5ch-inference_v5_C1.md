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

0.8416054666904719

# 6. Current score

0.10014

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on non-existent Kaggle datasets/modules (`/kaggle/input/uwm-libs`, `segmentation_models_pytorch`, `fast_ai_utils`, and `uwm.*`) that currently prevent the notebook from running at all. To preserve the original inference semantics as closely as possible, I keep the same multi-slice packing + thresholding + per-class postprocessing + RLE pipeline, but replace the missing model/learner with a safe, deterministic fallback that still produces correctly formatted predictions. I also fix missing imports (`pandas`, `fastai` symbols usage), remove IPython `!` shell usage, and ensure `mask2rle` returns a proper string (and isn’t accidentally joining CuPy scalars). Finally, I guarantee that `submission.csv` is written with the exact required columns/row count by merging into `sample_submission.csv` and filling missing predictions with empty strings.'
- What this solution (achieved 0.0) has done: 'I fix the crash in `CreateSample.encodes` by allocating the merged volume using the actual image shape (H,W) instead of the filename-parsed (W,H), which currently causes a broadcast error on slices where width/height are swapped. I also make `get_size` return `(h,w)` consistently (OpenCV order) so downstream logic stays coherent and avoids unnecessary transposes. These changes are execution-critical and keep the rest of the pipeline (multi-slice packing, resizing/cropping, thresholding, per-class morphology, and RLE submission formatting) identical. The script then run end-to-end and write a valid `submission.csv` with the required 20400 rows/3 columns.'
- What this solution (achieved 0.02426) has done: 'Your current 0.0 score is consistent with predicting (almost) all-empty masks due to the dummy model outputting very negative logits, so the smallest score-improving change is to replace that dummy with a lightweight, deterministic intensity-based segmentation heuristic that produces non-empty masks. I keep the exact same multi-slice packing, resizing/cropping, morphology, and RLE submission pipeline; only the “model” forward pass be swapped to generate plausible logits from the input image intensities. I also add a small, fixed per-class bias so different organs get different mask sizes without changing any downstream semantics. This should raise the score from 0.0 toward your target while keeping runtime under the limit and preserving the overall structure.'
- What this solution (achieved 0.38922) has done: 'Your current score is far below the target (0.02426 vs 0.8416, higher-is-better), so we need a real boost while keeping your pipeline intact. The smallest high-impact fix is to make the heuristic “model” output more organ-like masks by using deterministic, per-class intensity thresholds (and mild smoothing) instead of a near-random z-score with small biases. This preserves your core inference structure (multi-slice packing, same resizing/cropping, same sigmoid+0.5 thresholding, same morphology, same RLE formatting) but yields substantially more plausible non-empty segmentations. I also make `mask2rle` robust to missing CuPy (fallback to NumPy) without changing outputs when CuPy is available.'
- What this solution (achieved 0.28259) has done: 'Your current score (0.38922) is far below the target (0.8416), so we need a modest-but-real boost without changing your overall pipeline structure. The biggest low-risk gain here is to fix two inference mismatches that make masks systematically worse: (1) you currently create packs using `case_day_slice` IDs but later write predictions using `case_day_slice_####` IDs, which silently misaligns/empties merges; and (2) you pad to 320 even though your model outputs 288×288 center-crop space, which shifts masks and hurts Dice/Hausdorff. I make `get_sample_id()` return the exact same ID format as `Metadata.extract()` and pad in the correct (288) space before resizing back to original size, keeping the heuristic model, thresholding, morphology, and RLE submission logic intact. These are minimal, execution-safe changes that should move the score upward toward the target.'
- What this solution (achieved 0.28069) has done: 'Your current score (0.28259) is far below the target (0.8416), so we should push performance upward with minimal, low-risk changes that keep your overall heuristic pipeline intact. The biggest issue is a spatial misalignment: you center-crop to 288×288 but then “pad” back into 288 (no-op) before resizing to original size, which effectively discards the crop offset and harms Dice/Hausdorff; I instead correctly un-crop by padding the 288 crop back to the 320 resize canvas before final resize. Second, I keep your same heuristic model and thresholding semantics but make morphology less destructive (disk(5) is aggressive and can erase thin bowel), switching to a smaller opening kernel and adding a tiny hole-fill to stabilize masks without changing the overall approach. These changes are directly aligned with the metric (better spatial alignment + less over-erosion) and should move the score upward toward your target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.28227) has done: 'Your score gap is large (0.28069 vs target 0.8416, higher-is-better), so the smallest safe way to move upward without changing your pipeline is to fix a key semantic mismatch: the competition requires a single *binary union mask* shared across all classes per image, but your current code predicts per-class masks independently, which can be heavily penalized under the metric/submission expectations. I keep your exact packing/augmentation/heuristic-logits/threshold/morphology/RLE mechanics, but after producing the three class masks I compute their union and submit the same union RLE for each class row of that image. This is a minimal post-processing change (no model/training changes) that directly aligns submission semantics with the stated requirement and is likely to increase Dice/Hausdorff substantially toward your target. I also avoid re-reading the same PNG per-sample by using metadata H/W to keep results identical while reducing overhead.'
- What this solution (achieved 0.28227) has done: 'Your current score is far below the target (0.28227 vs 0.8416), so we should improve predictions while keeping your inference pipeline intact. The biggest remaining correctness issue is that you’re generating IDs from `get_sample_id(mid_path)` that don’t match the competition/sample_submission ID format (you’re missing the `_slice_####` part), which causes many predictions to be dropped and filled with empty strings. I make `get_sample_id()` return the exact same ID format as `Metadata.extract()` so merges align, and I also make pack grouping use only case_day (not slice) to avoid fragmented packs. These are minimal, semantic-preserving fixes that should materially increase the number of non-empty, correctly aligned masks and move the score upward toward the target.'
- What this solution (achieved 0.10023) has done: 'The current score (0.28227) is far below the target (0.8416, higher-is-better), so we should improve mask quality with minimal, low-risk changes that keep your exact inference pipeline intact. The most direct gain (without changing model/training/core flow) is to (1) make the heuristic logits use all 5 packed slices with a center-weighted average instead of collapsing channels by mean (keeps the same heuristic model idea but uses the packed information you already compute), and (2) switch morphology from opening-only (which can erase thin bowel) to a light closing + hole-fill, which typically improves Dice/Hausdorff for GI masks. I also keep the “union mask for all classes” submission behavior you already have, but make the per-class thresholds slightly less strict to reduce false-empty predictions. These are confined to the heuristic forward pass and post-processing; packing, resize/crop/uncrop, thresholding at 0.5, and RLE submission formatting stay the same.'
- What this solution (achieved 0.09999) has done: 'Your current score (0.10023) is far below the target (0.8416, higher-is-better), so we should push predictions upward with small, safe fixes that keep your heuristic model + resize/crop/uncrop + sigmoid@0.5 + morphology + RLE pipeline intact. The biggest correctness issue is that you only generate predictions for the “mid-slice” of each 5-slice pack, so most `id`s in `sample_submission.csv` never get a prediction and are filled with empty masks, which tanks Dice/Hausdorff. I keep the same packed inference, but also emit a prediction for every slice in each pack (and ensure later predictions overwrite earlier ones deterministically), so all 20400 test rows get a non-empty/empty decision from the same core heuristic. This is a minimal change (only bookkeeping/output coverage) and should move the score substantially toward your target without changing the model architecture or post-processing semantics.'
- What this solution (achieved 0.10095) has done: 'Your current score is far below the target, so the smallest likely high-impact fix is to ensure every test slice gets a prediction rather than inheriting an empty mask when it was never the “mid” slice of any pack. I keep the same packing, resizing/cropping, heuristic logits, sigmoid@0.5, morphology, union-mask submission, and RLE formatting, but change the bookkeeping so each pack emits predictions for all its member slices (and we deterministically avoid overwriting an already-filled prediction). I also make the pack generation use `TEST_FILES` grouped by `case_day` directly (instead of only those paths found via `METADATA` lookups), so we don’t accidentally skip slices due to any ID/path mismatch. These changes increase coverage/alignment without changing the core inference semantics, and should move the score upward toward your target.'
- What this solution (achieved 0.10014) has done: 'Your current score is far below the target, so we should increase it with the smallest changes that improve mask correctness/coverage without changing the overall pipeline. The biggest low-risk issue is that you skip packs for slices missing in `METADATA` (which can silently reduce coverage and fill many rows with empty masks), even though you can derive `h,w` directly from the PNG when needed; I add a safe fallback metadata extraction so every slice in every pack can be predicted. Next, your “don’t overwrite existing prediction” rule makes early (often less-centered) pack predictions stick for a slice; switching to “always overwrite” makes each slice end up with the most recent (more centered) prediction, improving spatial quality while keeping the same heuristic/model/postprocess. Finally, I keep the union-mask submission semantics but make postprocessing slightly less likely to create false positives by removing tiny components after resizing to the original shape (minimal extra cleanup aligned with Dice/Hausdorff).'

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
    get_image_files,
    show_image,
    show_images,
    progress_bar,
    noop,
)

import re
from collections import defaultdict
from functools import partial
from more_itertools import windowed, chunked


def get_size(p: Path):
    parts = p.stem.split("_")
    img = cv.imread(str(p), cv.IMREAD_UNCHANGED)
    if img is None:
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            w, h = int(parts[0]), int(parts[1])
            return (h, w)
        return (0, 0)
    h, w = img.shape[:2]
    return (h, w)


def get_case_day(s):
    return re.search(r"case\d+_day\d+", str(s)).group()


def get_slice(s, as_number=False):
    m = re.search(r"slice_\d\d\d\d", str(s))
    if m is None:
        return 0 if as_number else "slice_0000"
    slice_no = m.group()
    return int(slice_no.split("_")[-1]) if as_number else slice_no


def get_sample_id(s):
    return f"{get_case_day(s)}_slice_{int(get_slice(s, as_number=True)):04d}"


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
    mid_idx = n_slices_to_merge // 2
    chunks = []
    for case_day, files in groups.items():
        pad = [None] * (n_slices_to_merge // 2)
        files = pad + files + pad
        for pack in windowed(files, n=n_slices_to_merge, step=step_size):
            pack = list(pack)
            mid = pack[mid_idx]
            if mid is None:
                continue
            not_none = [i for i, x in enumerate(pack) if x is not None]
            if not not_none:
                continue
            first_not_none, last_not_none = not_none[0], not_none[-1]
            for i in range(0, first_not_none):
                pack[i] = pack[first_not_none]
            for i in range(last_not_none + 1, len(pack)):
                pack[i] = pack[last_not_none]
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
        case_and_day = path.parents[1].stem  # e.g. case110_day12
        slice_no = get_slice(path, as_number=True)
        sample_id = f"{case_and_day}_slice_{int(slice_no):04d}"
        h, w = get_size(path)
        return Metadata(sample_id, str(path), int(h), int(w))




## === cell 3
DATA_DIR = Path("/kaggle/input/uw-madison-gi-tract-image-segmentation/")

DEBUG = False

TEST_IDS = (
    pd.read_csv(DATA_DIR / "sample_submission.csv")["id"].drop_duplicates().tolist()
)

TEST_FILES = get_image_files(DATA_DIR / "test")
METADATA = {m.sample_id: m for m in TEST_FILES.map(Metadata.extract)}

len(TEST_IDS), len(TEST_FILES), len(METADATA)




## === cell 4
class CreateSample(Transform):
    def encodes(self, pack):
        first = None
        for fn in pack:
            if fn is not None:
                first = fn
                break
        if first is None:
            return np.zeros((0, 0, len(pack)), dtype=np.uint8)

        first_img = cv.imread(str(first), cv.IMREAD_UNCHANGED)
        h, w = first_img.shape[:2]
        merged = np.empty((h, w, len(pack)), dtype=np.uint8)

        q = 0.01
        for i, fn in enumerate(pack):
            img = cv.imread(str(fn), cv.IMREAD_UNCHANGED)
            if img is None:
                merged[..., i] = 0
                continue

            if img.shape[0] != h or img.shape[1] != w:
                img = cv.resize(img, (w, h), interpolation=cv.INTER_LINEAR)

            lo, hi = np.percentile(img, [q * 100, (1 - q) * 100])
            img = np.clip(img, lo, hi)
            v_min, v_max = np.min(img), np.max(img)
            denom = float(v_max - v_min) if (v_max - v_min) != 0 else 1.0
            img = (img - v_min) / denom
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


class AugValid(AugBase):
    split_idx, order = 1, 2


def valid_aug():
    return AugValid(
        A.Compose(
            [
                A.Resize(320, 320),
                A.CenterCrop(288, 288),
            ]
        )
    )




## === cell 5
class HeuristicSegModel(torch.nn.Module):
    def __init__(self, n_classes=3, out_h=288, out_w=288, n_in_slices=5):
        super().__init__()
        self.n_classes = n_classes
        self.out_h = out_h
        self.out_w = out_w
        self.n_in_slices = n_in_slices

        self.register_buffer(
            "_thresholds",
            torch.tensor([0.56, 0.50, 0.46], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "_scales",
            torch.tensor([14.0, 12.0, 10.0], dtype=torch.float32).view(1, 3, 1, 1),
        )

        w = torch.tensor([0.10, 0.20, 0.40, 0.20, 0.10], dtype=torch.float32)
        w = w / w.sum()
        self.register_buffer("_slice_w", w.view(1, -1, 1, 1))  # [1,C,1,1]

    def forward(self, x):
        if x.shape[1] == self.n_in_slices:
            w = self._slice_w.to(device=x.device, dtype=x.dtype)
            xw = (x * w).sum(dim=1, keepdim=True)  # [B,1,H,W]
        else:
            xw = x.mean(dim=1, keepdim=True)

        xs = F.avg_pool2d(xw, kernel_size=5, stride=1, padding=2)

        x3 = xs.repeat(1, 3, 1, 1)
        thr = self._thresholds.to(device=x.device, dtype=x.dtype)
        scl = self._scales.to(device=x.device, dtype=x.dtype)

        logits = (x3 - thr) * scl

        if logits.shape[-2:] != (self.out_h, self.out_w):
            logits = F.interpolate(
                logits,
                size=(self.out_h, self.out_w),
                mode="bilinear",
                align_corners=False,
            )
        return logits


dummy_model = HeuristicSegModel()



## === cell 6
if DEBUG:
    predicted_ids = [
        id_
        for id_ in TEST_IDS
        if id_.startswith("case123_day20") or id_.startswith("case77_day20")
    ]
else:
    predicted_ids = TEST_IDS

len(predicted_ids)



## === cell 7
all_groups = group_case_day_from_files(TEST_FILES)
packs = packed(all_groups, n_slices_to_merge=5, step_size=1)
len(packs), packs[0][len(packs[0]) // 2]



## === cell 8
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dummy_model = dummy_model.to(device).eval()
device



## === cell 9
try:
    import cupy as cp  # type: ignore

    _HAS_CUPY = True
except Exception:
    cp = None
    _HAS_CUPY = False


def mask2rle(mask):
    """
    mask: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    """
    if _HAS_CUPY:
        mask = cp.asarray(mask, dtype=cp.uint8)
        pixels = mask.flatten()
        pad = cp.asarray([0], dtype=cp.uint8)
        pixels = cp.concatenate([pad, pixels, pad])
        runs = cp.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        runs = runs.astype(cp.int64)
        runs_list = cp.asnumpy(runs).tolist()
        return " ".join(str(x) for x in runs_list)

    mask = np.asarray(mask, dtype=np.uint8).ravel()
    pixels = np.concatenate([[0], mask, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(int(x)) for x in runs)


def rle2mask(mask_rle, shape):
    s = np.asarray(mask_rle.split(), dtype=int)
    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


def uncrop_to_canvas(mask_288, canvas_size=320):
    mask_288 = np.asarray(mask_288)
    h, w = mask_288.shape[:2]
    assert h == 288 and w == 288
    out = np.zeros((canvas_size, canvas_size), dtype=mask_288.dtype)
    top = (canvas_size - h) // 2
    left = (canvas_size - w) // 2
    out[top : top + h, left : left + w] = mask_288
    return out




## === cell 10
from skimage.morphology import disk
from scipy.ndimage import binary_closing, binary_fill_holes, label as ndi_label

factory = CreateSample()
val_aug = valid_aug()

preds_map = {}  # (id, class) -> rle

batch_size = 64
chunks = list(chunked(packs, n=batch_size))

MODEL_OUT_SIZE = 288
AUG_CANVAS_SIZE = 320


def get_meta_for_path(fn: Path) -> Metadata:
    sid = get_sample_id(fn)
    m = METADATA.get(sid)
    if m is not None:
        return m
    h, w = get_size(fn)
    return Metadata(sample_id=sid, full_path=str(fn), h=int(h), w=int(w))


def remove_small_components(mask01: np.ndarray, min_area: int = 64) -> np.ndarray:
    mask01 = (mask01 > 0).astype(np.uint8)
    if mask01.sum() < min_area:
        return mask01
    cc, n = ndi_label(mask01)
    if n <= 1:
        return mask01
    out = np.zeros_like(mask01)
    for i in range(1, n + 1):
        comp = cc == i
        if int(comp.sum()) >= min_area:
            out[comp] = 1
    return out


with torch.no_grad():
    for subset in progress_bar(chunks):
        ims = []
        subset_slice_ids = []
        subset_slice_metas = []

        for pack in subset:
            slice_ids = []
            slice_metas = []
            for fn in pack:
                if fn is None:
                    continue
                meta = get_meta_for_path(fn)
                slice_ids.append(meta.sample_id)
                slice_metas.append(meta)

            if not slice_ids:
                continue

            img = factory(pack)  # uint8 (H,W,C=5)
            img = val_aug.aug(image=img)["image"]  # HWC uint8, now 288x288

            t = torch.from_numpy(img).to(torch.float32) / 255.0  # HWC
            t = t.permute(2, 0, 1)  # CHW
            ims.append(t)

            subset_slice_ids.append(slice_ids)
            subset_slice_metas.append(slice_metas)

        if not ims:
            continue

        x = torch.stack(ims, dim=0).to(device)  # [B,C,H,W]
        logits = dummy_model(x)
        labels = (torch.sigmoid(logits) >= 0.5).to(torch.uint8).cpu().numpy()

        for slice_ids, slice_metas, mask in zip(
            subset_slice_ids, subset_slice_metas, labels
        ):
            cls_final_masks = []
            for i in range(3):
                cls_mask = uncrop_to_canvas(mask[i], canvas_size=AUG_CANVAS_SIZE)
                closed = binary_closing(cls_mask.astype(bool), structure=disk(2))
                filled = binary_fill_holes(closed)
                final_320 = filled.astype(np.uint8)
                cls_final_masks.append(final_320)

            union_320 = (
                cls_final_masks[0] | cls_final_masks[1] | cls_final_masks[2]
            ).astype(np.uint8)

            for test_id, meta in zip(slice_ids, slice_metas):
                h, w = int(meta.h), int(meta.w)
                union_mask = cv.resize(union_320, (w, h), cv.INTER_NEAREST)
                union_mask = remove_small_components(union_mask, min_area=64)

                rle = mask2rle(union_mask) if union_mask.any() else ""
                for name in ("large_bowel", "small_bowel", "stomach"):
                    preds_map[(test_id, name)] = rle

        del x, logits, labels
        gc.collect()

len(preds_map)



## === cell 11
df_preds = pd.DataFrame(
    [{"id": k[0], "class": k[1], "predicted": v} for k, v in preds_map.items()]
)
df_preds.head(), df_preds.shape



## === cell 12
df_submit = pd.read_csv(DATA_DIR / "sample_submission.csv")
df_submit = df_submit.drop(columns="predicted").merge(
    df_preds, on=["id", "class"], how="left"
)
df_submit["predicted"] = df_submit["predicted"].fillna("")
df_submit.to_csv("submission.csv", index=False)

print(df_submit.shape)
print(df_submit.columns.tolist())
print(pd.read_csv("submission.csv").head(5))
print("Filled preds:", (df_submit["predicted"] != "").sum(), "of", len(df_submit))
