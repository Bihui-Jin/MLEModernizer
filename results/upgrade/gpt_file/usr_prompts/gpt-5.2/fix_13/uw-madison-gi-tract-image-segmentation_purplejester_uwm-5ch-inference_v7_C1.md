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

0.8416672559477131

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the dependency on missing external libraries (`/kaggle/input/uwm-libs`, `segmentation_models_pytorch`, `fast_ai_utils`, and `uwm.*`) by replacing them with small local equivalents that preserve the same execution flow (pack slices → build test_dl → load exported fastai learner → predict → threshold → postprocess → RLE). I also fix notebook-to-script issues (cell numbering, `!` shell usage, missing `pandas` import, and undefined fastai symbols like `Transform`/`load_learner`) so the pipeline runs end-to-end. Finally, I make submission generation robust by merging against `sample_submission.csv`, filling missing predictions with empty strings, and writing `submission.csv` with correct columns and row count.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is coming from producing (effectively) empty/invalid masks: the model file is not present at `/kaggle/input/uwm-models/norm_full.pkl`, so the code falls back to empty RLE for every row. The smallest score-improving change is to (1) auto-discover a `.pkl` model inside the provided dataset directory (and common Kaggle locations) and load it if available, instead of hard-failing to the empty-mask path. Additionally, to avoid a subtle but common cause of “all-empty” RLEs even when a model loads, I fix RLE encoding order to the competition’s expected column-major (Fortran) flattening. These changes preserve your core inference pipeline (pack slices → fastai test_dl → predict → threshold → postprocess → RLE → merge with sample_submission) while making the submission non-degenerate and correctly encoded.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the submission being effectively “all empty masks” due to model loading issues and/or a test-id ↔ image-path mismatch that prevents filling predictions for the required rows. I make two minimal, score-relevant fixes: (1) ensure `Metadata.extract()` builds `sample_id` that exactly matches the competition `id` format (case/day + `slice_XXXX`), and (2) make inference cover every test image (not just packed mid-slices) by packing with `n_slices_to_merge=3` and `step_size=1`, then explicitly generate predictions only for the center slice of each pack (unique by id). This preserves your core pipeline (pack slices → fastai test_dl → predict → threshold → postprocess → RLE → merge with sample_submission) while preventing missing/empty rows that yield near-zero. The submission writing stays unchanged and still produces a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with generating mostly-empty masks even when a learner loads; the two most common causes here are (1) model/device mismatch at inference time (fastai’s `test_dl` applies transforms but your input packs are raw Paths, so the model may not be seeing the same preprocessing as training), and (2) overly aggressive post-processing (a disk(5) opening can erase small/medium structures, collapsing to empty RLE). I keep your core pipeline unchanged (packs → `test_dl` → `get_preds` → sigmoid+threshold → resize → RLE → merge) but make two minimal score-directed changes: ensure the test dataloader receives the same item type as training by explicitly applying `CreateSample` before `test_dl`, and make the morphological cleanup less destructive (smaller structuring element) to avoid wiping predictions to empty. I also add a safe assert-based sanity check to detect if we’re producing near-all-empty RLEs (without changing outputs), so you can catch regressions quickly.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly coming from producing an all-empty (or near-all-empty) submission due to an inference mismatch: you’re building `test_dl` from already-materialized numpy arrays (`subset_x`) without applying the same dataloader transforms/normalization that the exported fastai learner expects. I make the smallest change that keeps your pipeline intact but ensures inference uses the learner’s item transforms by passing the original `packs` into `test_dl` and letting `after_item` run `CreateSample`, then I add `ChannelsFirst` + `NormalizeSample` at inference time if they are not already present. I also keep your RLE encoding order (Fortran) and lightly reduce the chance of wiping masks to empty by making the opening structuring element smaller (disk(1)), which is a minimal postprocess tweak aimed at moving score up from 0 toward the target. The script still write a valid `submission.csv` with the exact required columns and row count.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with either (a) producing an almost-all-empty submission because the learner isn’t actually generating usable masks, or (b) producing masks but mapping them to the wrong `id` due to a subtle `case_day` parsing issue. I make two minimal, score-directed fixes: (1) make `Metadata.extract()` and `get_case_day()` robust to the folder structure so `sample_id` always matches `test.csv`/`sample_submission.csv` ids, and (2) ensure inference IDs come from the center slice’s metadata (not a regex on the path string) to avoid any mismatch. These preserve your core pipeline (packs → fastai `test_dl` → `get_preds` → threshold → postprocess → RLE → merge) but prevent the “valid CSV yet wrong/empty alignment” failure mode that commonly yields 0.0.'
- What this solution (achieved 0.0) has done: 'The 0.0 score is most consistent with (1) test `id` ↔ image-path mismatch (so your predictions never get merged onto the required rows) and/or (2) duplicate/unstable IDs produced during packing causing many rows to remain empty after the merge. I make two minimal, score-directed fixes: build `packs` directly from the actual `DATA_DIR/test/**/scans/*.png` files (so every center slice maps to a real test image), and ensure the `test_id` used for each prediction is computed from the center slice path using the exact same parsing rules as `Metadata.extract()` (no re-reading the image). I also add a small guard to force creation of predictions for *all* `TEST_IDS` (missing ones become empty, but now it’s only for truly missing files), and keep your model, threshold, postprocess, and RLE encoding semantics unchanged. This should move the score upward toward the target by fixing alignment/coverage rather than “optimizing” the model.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with producing mostly-empty or mis-decoded masks even though the CSV is valid, so the smallest score-directed changes are to (1) stop re-normalizing each slice independently inside `CreateSample` (which often destroys the intensity distribution the trained fastai model expects) and instead apply a simple fixed uint16→uint8 scaling, and (2) make the final binary mask less likely to be wiped out by postprocessing by gating `binary_opening` to only run on sufficiently large predicted areas. These keep your core pipeline identical (packs → fastai `test_dl` → `get_preds` → threshold → postprocess → RLE → merge) while making the produced masks non-degenerate. I also ensure `predicted_ids` actually controls which packs are inferred (so DEBUG and coverage behave as intended) without changing inference semantics for the full test set.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most likely because the submission is effectively empty or misaligned: right now you only predict center-slices that appear in `packs`, but `packs` (with padding and windowing) can skip/duplicate center slices and your `seen_ids` logic can silently drop required ids, leaving many rows empty after the merge. I make a minimal change to build `packs` directly from the exact `test.csv` ids (so coverage and id-alignment are guaranteed), while keeping the same model, transforms, threshold (0.4), postprocess, and RLE encoding logic. I also make `METADATA` built from the same ids-to-path mapping (no reliance on regex parent search), and add a small assert-style check that we produced predictions for ~all ids (without changing outputs). These changes should move the score up from 0.0 toward your target by fixing coverage/alignment rather than “optimizing” the model.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 is most consistent with producing a largely empty or misaligned submission because `predicted_ids` comes from `sample_submission.csv` but `packs` are built only for ids that exist in `id2path`; any id/path parsing mismatch silently drops those ids and leaves empty rows after the merge. I make the smallest score-improving change by building `id2path` directly from `test/**/scans/*.png` using `Metadata.extract()` (single canonical id parser), then constructing `packs` from the actual `test.csv` ids to guarantee coverage and correct alignment. I also ensure we add exactly one prediction per `(id,class)` by deduplicating at the source (one center slice per id), while keeping the same model loading, fastai inference flow, threshold, postprocess, and RLE (Fortran order) encoding. These changes should move the score up from 0.0 toward your target by fixing the “valid CSV but wrong/empty alignment” failure mode rather than tuning the model.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with a silent ID↔image mismatch that leaves most/all rows empty after the merge, even though the CSV is valid. I make ID extraction deterministic by building `id2path/METADATA` from the official `test.csv` IDs and resolving each ID to an existing `test/**/scans/*.png` path (instead of relying on parsing every file and hoping it matches). I also ensure we always create exactly one pack per `test.csv` id (center slice + nearest neighbors within the same case/day), so prediction coverage is ~100% and aligned to the required submission rows. These changes keep your core inference logic (fastai learner → test_dl → sigmoid/threshold → postprocess → Fortran RLE → merge with sample_submission) identical, but remove the main failure mode that yields an all-empty submission.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly coming from a submission that evaluates as “all empty / wrong masks” due to an inference mismatch, not from the CSV format itself. I keep your exact model/inference/postprocess logic, but make two minimal, score-relevant fixes: (1) ensure inference uses the learner’s stored validation pipeline (`dl=learn.dls.valid`) so the same `after_item/after_batch` transforms and normalization are applied as in training, and (2) ensure the predicted mask is decoded to the correct spatial size by using the learner’s actual output mask shape (rather than assuming 288→pad to 320) before resizing back to the original `(h,w)`. These changes are small, deterministic, and aimed at moving you up from 0 toward the target by preventing “valid but mis-scaled/misaligned” masks. The script still run end-to-end and write a valid `submission.csv` with the required columns and 20400 rows.'

# 9. Code solution

## === cell 0
import gc
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from functools import partial

import numpy as np
import pandas as pd
import cv2 as cv
import torch
import torch.nn.functional as F

import albumentations as A
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
from more_itertools import windowed, chunked

logging.captureWarnings(True)
pd.set_option("display.max_colwidth", 200)

DATA_DIR = Path("/kaggle/input/uw-madison-gi-tract-image-segmentation/")
if not DATA_DIR.exists():
    DATA_DIR = Path("/kaggle/data/uw-madison-gi-tract-image-segmentation/")

assert DATA_DIR.exists(), f"DATA_DIR not found: {DATA_DIR}"




## === cell 1
@dataclass
class Metadata:
    sample_id: str
    full_path: str
    h: int
    w: int

    @classmethod
    def extract(cls, path: Path) -> "Metadata":
        case_and_day = None
        for p in path.parents:
            if re.fullmatch(r"case\d+_day\d+", p.name):
                case_and_day = p.name
                break
        if case_and_day is None:
            m = re.search(r"case\d+_day\d+", str(path))
            case_and_day = m.group(0) if m else path.parents[1].stem

        parts = path.stem.split("_")
        if len(parts) >= 3:
            slice_no = int(parts[2])
        else:
            m = re.findall(r"\d+", path.stem)
            slice_no = int(m[-1]) if m else 0

        img = cv.imread(str(path), cv.IMREAD_UNCHANGED)
        h, w = img.shape[:2]
        sample_id = f"{case_and_day}_slice_{int(slice_no):04d}"
        return Metadata(sample_id, str(path), int(h), int(w))




## === cell 2
DEBUG = False

TEST_IDS = pd.read_csv(DATA_DIR / "test.csv")["id"].drop_duplicates().tolist()

TEST_FILES = get_image_files(DATA_DIR / ("train" if DEBUG else "test"))


def _parse_case_day_and_slice_from_id(sid: str):
    m = re.match(r"^(case\d+_day\d+)_slice_(\d{4})$", sid)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def _parse_case_day_and_slice_from_path(p: Path):
    case_and_day = None
    for parent in p.parents:
        if re.fullmatch(r"case\d+_day\d+", parent.name):
            case_and_day = parent.name
            break
    if case_and_day is None:
        m = re.search(r"case\d+_day\d+", str(p))
        case_and_day = m.group(0) if m else None

    parts = p.stem.split("_")
    slice_no = None
    if len(parts) >= 3:
        try:
            slice_no = int(parts[2])
        except Exception:
            slice_no = None
    if slice_no is None:
        m = re.findall(r"\d+", p.stem)
        slice_no = int(m[-1]) if m else None
    return case_and_day, slice_no


case_day_to_slices = defaultdict(dict)
for p in TEST_FILES:
    p = Path(p)
    cd, sl = _parse_case_day_and_slice_from_path(p)
    if cd is None or sl is None:
        continue
    if sl not in case_day_to_slices[cd]:
        case_day_to_slices[cd][sl] = p


def _resolve_id_to_path_and_meta(sid: str):
    cd, sl = _parse_case_day_and_slice_from_id(sid)
    if cd is None or sl is None:
        return None, None
    d = case_day_to_slices.get(cd, {})
    if not d:
        return None, None
    if sl in d:
        p = d[sl]
    else:
        nearest = min(d.keys(), key=lambda x: abs(x - sl))
        p = d[nearest]
    md = Metadata.extract(p)
    md = Metadata(sample_id=sid, full_path=str(p), h=md.h, w=md.w)
    return p, md


id2path = {}
METADATA = {}
missing = []
for sid in TEST_IDS:
    p, md = _resolve_id_to_path_and_meta(sid)
    if p is None:
        missing.append(sid)
        continue
    id2path[sid] = p
    METADATA[sid] = md

if missing:
    logging.warning(
        f"Could not resolve {len(missing)} TEST_IDS to png paths (showing up to 10): {missing[:10]}"
    )
    logging.warning("Unresolved ids will remain empty after merge and may hurt score.")

logging.warning(f"Resolved {len(METADATA)} / {len(TEST_IDS)} test ids to image paths.")




## === cell 3
def get_case_day(s) -> str:
    m = re.search(r"case\d+_day\d+", str(s))
    return m.group(0) if m else ""


def get_slice(s, as_number: bool = False):
    m = re.search(r"slice_(\d{4})", str(s))
    if m:
        slice_no = int(m.group(1))
    else:
        parts = Path(s).stem.split("_")
        slice_no = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else 0
    return slice_no if as_number else f"slice_{slice_no:04d}"


def get_sample_id(s) -> str:
    cd = get_case_day(s)
    return f"{cd}_{get_slice(s)}" if cd else f"{get_slice(s)}"


def get_size_from_path(p: Path):
    img = cv.imread(str(p), cv.IMREAD_UNCHANGED)
    return img.shape[:2]


def group_case_day_from_files(image_files):
    groups = defaultdict(list)
    for fn in image_files:
        cd = get_case_day(fn)
        if not cd:
            continue
        groups[cd].append(fn)
    groups = {
        k: sorted(v, key=partial(get_slice, as_number=True)) for k, v in groups.items()
    }
    return groups


def packed(groups, n_slices_to_merge=3, step_size=1):
    assert n_slices_to_merge % 2 != 0
    mid_idx = n_slices_to_merge // 2
    chunks = []
    for _, files in groups.items():
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
            assert pack[mid_idx] is not None
            chunks.append(pack)
    return chunks




## === cell 4
class CreateSample(Transform):
    def encodes(self, pack):
        h, w = get_size_from_path(pack[0])
        merged = np.zeros((h, w, len(pack)), dtype=np.uint8)
        for i, fn in enumerate(pack):
            img = cv.imread(str(fn), cv.IMREAD_UNCHANGED)
            if img is None:
                merged[:, :, i] = 0
                continue
            if img.dtype == np.uint16:
                merged[:, :, i] = (img / 257.0).clip(0, 255).astype(np.uint8)
            else:
                merged[:, :, i] = np.clip(img, 0, 255).astype(np.uint8)
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
        return tuple(t.permute(2, 0, 1) for t in x)

    def decodes(self, x):
        return tuple(t.permute(1, 2, 0) for t in x)


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
            ]
        )
    )


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
model_name = "norm_full"


def find_model_path(preferred: Path) -> Path | None:
    if preferred.exists():
        return preferred
    search_roots = [
        Path("/kaggle/input"),
        Path("/kaggle/data"),
        Path("/kaggle/working"),
        DATA_DIR,
    ]
    seen = set()
    candidates = []
    for root in search_roots:
        if not root.exists():
            continue
        rp = str(root.resolve())
        if rp in seen:
            continue
        seen.add(rp)
        for p in root.rglob("*.pkl"):
            candidates.append(p)
    if not candidates:
        return None
    exact = [p for p in candidates if p.stem == preferred.stem]
    if exact:
        return sorted(exact, key=lambda x: len(str(x)))[0]
    return sorted(candidates, key=lambda x: len(str(x)))[0]


MODEL_PATH = Path(f"/kaggle/input/uwm-models/{model_name}.pkl")
MODEL_PATH = find_model_path(MODEL_PATH)

learn = None
if MODEL_PATH is not None and MODEL_PATH.exists():
    logging.warning(f"Loading model: {MODEL_PATH}")
    learn = load_learner(MODEL_PATH)
else:
    logging.warning(
        "Model not found anywhere under common Kaggle folders. "
        "Will create an empty-mask submission (valid but low score)."
    )




## === cell 6
predicted_ids = (
    TEST_IDS
    if not DEBUG
    else [
        id_
        for id_ in TEST_IDS
        if id_.startswith("case123_day20") or id_.startswith("case77_day20")
    ]
)
len(predicted_ids)




## === cell 7
def _get_neighbor_path_from_map(cd: str, slice_no: int, fallback: Path) -> Path:
    d = case_day_to_slices.get(cd, {})
    if slice_no in d:
        return d[slice_no]
    if not d:
        return fallback
    nearest = min(d.keys(), key=lambda x: abs(x - slice_no))
    return d[nearest]


packs = []
pack_center_ids = []
skipped = 0

for sid in predicted_ids:
    cd, sl = _parse_case_day_and_slice_from_id(sid)
    if cd is None or sl is None:
        skipped += 1
        continue
    center = id2path.get(sid, None)
    if center is None:
        skipped += 1
        continue

    left = _get_neighbor_path_from_map(cd, sl - 1, center)
    right = _get_neighbor_path_from_map(cd, sl + 1, center)

    packs.append([left, center, right])
    pack_center_ids.append(sid)

logging.warning(
    f"Built packs for {len(packs)} ids; skipped {skipped} ids due to unresolved ids."
)
len(packs), len(predicted_ids)




## === cell 8
import cupy as cp


def mask2rle(mask):
    """
    mask: numpy array, 1 - mask, 0 - background

    Encode in column-major (Fortran) order to match the competition's
    top-to-bottom then left-to-right indexing.
    """
    if mask is None:
        return ""
    mask = cp.asarray(mask.astype(np.uint8))
    pixels = mask.reshape(-1, order="F")
    pad = cp.asarray([0], dtype=pixels.dtype)
    pixels = cp.concatenate([pad, pixels, pad])
    runs = cp.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    if runs.size == 0:
        return ""
    return " ".join(str(int(x)) for x in cp.asnumpy(runs))


def rle2mask(mask_rle, shape):
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




## === cell 9
from skimage.morphology import disk
from scipy.ndimage import binary_opening

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

preds = []
if learn is None:
    for test_id in predicted_ids:
        for name in ("large_bowel", "small_bowel", "stomach"):
            preds.append({"id": test_id, "class": name, "predicted": ""})
else:
    learn.dls.to(device)
    learn.model.to(device)
    learn.eval()

    base_dl = getattr(learn.dls, "valid", None)
    if base_dl is None:
        base_dl = learn.dls[1] if len(learn.dls.loaders) > 1 else learn.dls.train

    with learn.no_bar():
        batch_size = 32  # keep memory stable
        chunks = list(chunked(list(zip(packs, pack_center_ids)), n=batch_size))

        for subset in progress_bar(chunks):
            subset_packs = [x[0] for x in subset]
            subset_ids = [x[1] for x in subset]

            test_dl = base_dl.test_dl(
                subset_packs, batch_size=batch_size, device=device
            )

            logits, *_ = learn.get_preds(dl=test_dl, act=noop)

            labels = (
                (torch.sigmoid(logits) >= 0.4).detach().cpu().numpy().astype(np.uint8)
            )

            for test_id, mask in zip(subset_ids, labels):
                if test_id not in METADATA:
                    continue
                m = METADATA[test_id]
                h, w = m.h, m.w

                ph, pw = int(mask.shape[-2]), int(mask.shape[-1])

                for i, name in enumerate(("large_bowel", "small_bowel", "stomach")):
                    cls_mask = mask[i].astype(np.uint8)

                    if (pw != w) or (ph != h):
                        cls_mask = cv.resize(cls_mask, (w, h), cv.INTER_NEAREST)

                    if int(cls_mask.sum()) >= 32:
                        cls_mask = binary_opening(
                            cls_mask.astype(bool), structure=disk(1)
                        ).astype(np.uint8)

                    rle = mask2rle(cls_mask)
                    preds.append({"id": test_id, "class": name, "predicted": rle})

            del logits, labels
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()




## === cell 10
df_preds = pd.DataFrame(preds)

df_preds = df_preds.groupby(["id", "class"], as_index=False)["predicted"].first()

print(df_preds.shape)
print(df_preds.head())

empty_rate = (df_preds["predicted"].fillna("") == "").mean() if len(df_preds) else 1.0
print(f"Empty-RLE rate in produced preds: {empty_rate:.4f}")

pred_id_count = df_preds["id"].nunique() if len(df_preds) else 0
print(f"Unique ids predicted: {pred_id_count} / {len(TEST_IDS)}")

missing_pred_ids = sorted(set(TEST_IDS) - set(df_preds["id"].unique()))
if len(missing_pred_ids) > 0:
    logging.warning(
        f"Missing predictions for {len(missing_pred_ids)} ids (showing up to 10): {missing_pred_ids[:10]}"
    )
if pred_id_count < int(0.98 * len(TEST_IDS)):
    logging.warning(
        "Predictions cover <98% of TEST_IDS; remaining rows will be empty after merge and may hurt score."
    )




## === cell 11
df_submit = pd.read_csv(DATA_DIR / "sample_submission.csv")
df_submit = df_submit.drop(columns="predicted").merge(
    df_preds, on=["id", "class"], how="left"
)

df_submit["predicted"] = df_submit["predicted"].fillna("")

df_submit.to_csv("submission.csv", index=False)

out = pd.read_csv("submission.csv")
print(out.shape)
print(out.head(10))
print("Wrote submission.csv")
