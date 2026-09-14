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

0.838288417791989

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the submission merge crash by ensuring `df_preds` always has the exact expected columns (`id`, `class`, `predicted`) and by defensively handling cases where prediction generation yields an empty/incorrect DataFrame. I also fix an ID mismatch bug: you were using `sample_submission.csv` to build `TEST_IDS`, which are *row ids* (like `case123_dayXX_slice_YYYY_class`) and don’t match the `METADATA` keys; instead we must use `test.csv` (or parse ids consistently) so every test slice gets predicted. These changes are score-positive (they enable non-empty predictions and correct alignment) while preserving your model/inference core logic, and they guarantee a valid `submission.csv` is written.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with the model not being found, which makes the code emit empty masks for every row; the smallest score-positive change is to load the learner from any available `.pkl` under the Kaggle inputs rather than a single hardcoded path. I keep your inference, thresholding, resizing, and post-processing exactly the same, and only add a defensive model-discovery fallback plus a CPU-safe `mask2rle` fallback if CuPy is unavailable (to avoid runtime failures that would again produce an all-empty submission). This should move the score upward toward the target as soon as a compatible exported fastai learner exists in the environment, while still always producing a valid `submission.csv`. Paths and submission formatting stay unchanged.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with producing an all-empty submission, which in this pipeline happens when the learner isn’t loaded or when inference yields no usable predictions. I make the smallest score-positive change by ensuring the code actually finds and loads the intended model file first (including common Kaggle input locations) and by failing hard if no learner is found (so you don’t unknowingly submit blanks again). I also add a minimal safety fix to guarantee every `(id,class)` pair is filled exactly once by reindexing predictions to `sample_submission` before writing, avoiding silent mismatches/duplicates that can zero out masks. Core inference logic (packing, resizing, sigmoid+0.5 threshold, opening, RLE) stays identical.'
- What this solution (achieved 0.0) has done: 'I fix the execution blockers so the notebook always runs end-to-end and writes a valid `submission.csv`. The root cause of the 0.0 is that no exported fastai learner `.pkl` exists in this Kaggle environment, so `learn` is `None` and inference crashes; I replace the hard fail with a safe fallback that still produces a correctly formatted submission (empty masks) rather than erroring. To avoid producing misaligned/partial predictions when running the fallback, I build the submission directly from `sample_submission.csv` and only attempt inference if a learner is successfully loaded. These changes are minimal and preserve your existing inference logic when a model is available; without a model present, score cannot be improved, but the submission be valid and non-crashing.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly coming from an all-empty submission, which in your current pipeline happens when no exported fastai learner is found/loaded. The smallest score-positive change is to ensure a learner is actually available by (1) loading from a local `/kaggle/working` cache if present and (2) optionally downloading your specified model from a Kaggle Dataset via `kagglehub` (common in Kaggle notebooks) into that cache—without changing your model/inference logic. I also add a hard check that, if a learner is loaded but yields zero predictions due to an ID/pack mismatch, we fail early instead of silently writing blanks again. These changes keep your architecture/inference/thresholding/post-processing identical, but make it far more likely you generate non-empty masks and move the score upward toward the 0.838 target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with a logically invalid submission for this metric: the competition expects **per-slice 2D RLEs** for each `(id,class)` row in `test.csv`, but your current code predicts **per-volume “sample_id”** like `case110_day12_slice_0001`, which won’t align to `test.csv` row ids and gets reindexed to all-empty. I keep your model and inference exactly the same, but change the ID used in `preds` to match `test.csv`’s `id` by reading it from `test.csv` and generating one prediction per test row. Concretely, I build a mapping from each packed center slice to all corresponding `test.csv` row-ids (one per class) and fill `preds` with those exact ids so the merge/reindex no longer wipes everything to blanks. This is the smallest change that should move the score upward toward the target because it turns your non-empty masks into correctly addressed predictions instead of empty strings.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is caused by an ID alignment bug: `test.csv` IDs are per-slice and include the class suffix (e.g. `case110_day12_slice_0001_large_bowel`), but your `METADATA` keys and `center_sample_id` omit the class suffix; this makes `predicted_sample_ids` get dropped and/or predictions get written under wrong IDs, resulting in all-empty after reindexing to `sample_submission`. I minimally fix this by parsing `test.csv` IDs into a `base_id` (no class suffix) and using that for `METADATA` lookup + pack creation, while still writing predictions back to the original full `id` required by submission (one row per `(id,class)`). I also fix `sample_to_rowid` to map `base_id -> {class: full_row_id}` (your current code incorrectly maps full row id to itself), preserving your model, thresholding, resizing, and post-processing. These changes should convert your existing non-empty masks into correctly-addressed per-row RLEs, moving the score upward toward the target without changing the model’s core behavior.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with the submission being effectively all-empty due to an ID mismatch: `df_submit` uses `sample_submission.csv` ids, while your predictions are keyed to `test.csv` ids (different schema), so the reindex fills everything with `""`. I keep your model/inference/post-processing untouched, but switch the submission base to `test.csv` (the file Kaggle evaluates against) so predicted `(id,class)` pairs align 1:1 with what’s scored. I also ensure predicted rows always use the exact `test.csv` `id` for the center slice (not a synthetic fallback), and deduplicate safely, so non-empty masks survive into `submission.csv`. These are the smallest changes expected to move the score upward toward the 0.838 target while preserving your core logic.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score strongly suggests the submission is effectively empty (or mostly empty) because only *center slices of packs* get predictions, while `test.csv` expects predictions for **every slice**. I keep your exact model, thresholding, resizing, and post-processing, but change only the inference-to-submission mapping: generate a prediction for each pack, then **propagate that prediction to all slice-ids covered by the pack window** (center ±2), so every `base_id` in `test.csv` receives an RLE. I also build packs from the full ordered slice list (not a sparse list of base_ids) to avoid missing slices due to step/window effects. These are minimal alignment/coverage fixes that should move the score up toward the target without altering your core inference semantics.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the submission being effectively empty due to incorrect RLE encoding order and inconsistent mask padding: this competition expects RLEs in **column-major (Fortran) order**, while your `mask2rle` currently flattens in row-major order, which can zero out the score even if masks look reasonable. I minimally fix `mask2rle` (and `rle2mask` for consistency) to use Fortran-order flattening, and I also remove the hardcoded `pad_mask(..., 320)` in favor of padding to the actual predicted mask size to avoid shifting/cropping errors. These changes keep your model, thresholding, resizing, and post-processing intact, but make the encoded submission match Kaggle’s expected decoding so the score should move upward toward the 0.838 target. The script still run end-to-end and always write a valid `submission.csv`.'

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
import re
import os
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
from more_itertools import windowed, chunked

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


def on_kaggle() -> bool:
    return Path("/kaggle").exists()




## === cell 2
@dataclass
class Metadata:
    sample_id: str
    full_path: str
    h: int
    w: int

    @classmethod
    def extract(cls, path: Path) -> "Metadata":
        case_and_day = path.parents[2].stem  # e.g., case110_day12
        parts = path.stem.split("_")
        slice_no = int(parts[2])
        sample_id = f"{case_and_day}_slice_{slice_no:04d}"

        img = cv.imread(str(path), cv.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        h, w = img.shape[:2]
        return Metadata(sample_id, str(path), int(h), int(w))




## === cell 3
DATA_DIR = Path("/kaggle/input/uw-madison-gi-tract-image-segmentation/")

DEBUG = False

df_test = pd.read_csv(DATA_DIR / ("train.csv" if DEBUG else "test.csv"))[
    ["id", "class"]
].copy()


def _base_id_from_row_id(row_id: str) -> str:
    m = re.match(
        r"^(case\d+_day\d+_slice_\d{4})(?:_(large_bowel|small_bowel|stomach))?$",
        str(row_id),
    )
    if m:
        return m.group(1)
    return str(row_id)


df_test["base_id"] = df_test["id"].astype(str).map(_base_id_from_row_id)
TEST_BASE_IDS = df_test["base_id"].drop_duplicates().tolist()

TEST_FILES = get_image_files(DATA_DIR / ("train" if DEBUG else "test"))
METADATA = {m.sample_id: m for m in TEST_FILES.map(Metadata.extract)}

len(TEST_BASE_IDS), len(TEST_FILES), len(METADATA)




## === cell 4
def get_size_from_filename(s: Path):
    parts = s.stem.split("_")
    w = int(parts[0])
    h = int(parts[1])
    return (h, w)


def get_case_day(s):
    return re.search(r"case\d+_day\d+", str(s)).group()


def get_slice(s, as_number=False):
    m = re.search(r"slice_\d\d\d\d", str(s))
    if m:
        slice_no = m.group()
        return int(slice_no.split("_")[-1]) if as_number else slice_no
    parts = Path(s).stem.split("_")
    slice_no = int(parts[2])
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




## === cell 5
class CreateSample(Transform):
    def encodes(self, pack):
        h, w = get_size_from_filename(pack[0])
        merged = np.ndarray((h, w, len(pack)), dtype=np.uint8)
        q = 0.01
        for i, fn in enumerate(pack):
            img = cv.imread(str(fn), cv.IMREAD_UNCHANGED)
            lo, hi = np.percentile(img, [q * 100, (1 - q) * 100])
            img = np.clip(img, lo, hi)
            v_min, v_max = np.min(img), np.max(img)
            if v_max > v_min:
                img = (img - v_min) / float(v_max - v_min)
            else:
                img = img * 0.0
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


class NormalizeSample(Transform):
    order = 99

    def setups(self, *args, **kwargs):
        self.mean, self.std = 0.18161897, 0.257913

    def encodes(self, x: TensorImageNChannels):
        return (x - self.mean) / self.std

    def decodes(self, x: TensorImageNChannels):
        return x * self.std + self.mean


def valid_aug():
    return AugValid(A.Compose([A.Resize(320, 320), A.CenterCrop(288, 288)]))




## === cell 6
model_name = "norm_full"
model_path = Path(f"/kaggle/input/uwm-models/{model_name}.pkl")
working_cache_dir = Path("/kaggle/working/uwm_models_cache")
working_cache_dir.mkdir(parents=True, exist_ok=True)


def _find_any_learner_pkl(preferred_name: str) -> Path | None:
    roots = [
        Path("/kaggle/input/uwm-models"),
        Path("/kaggle/input"),
        Path("/kaggle/working"),
    ]
    candidates = []
    for r in roots:
        if r.exists():
            candidates.extend(list(r.rglob("*.pkl")))
    if not candidates:
        return None
    for p in candidates:
        if p.name == f"{preferred_name}.pkl":
            return p
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _try_kagglehub_download_to_cache() -> list[Path]:
    try:
        import kagglehub  # type: ignore
    except Exception:
        return []

    dataset = os.environ.get("UWM_MODEL_DATASET", "").strip()
    if not dataset:
        return []

    try:
        downloaded_path = Path(kagglehub.dataset_download(dataset))
        pkls = list(downloaded_path.rglob("*.pkl"))
        copied = []
        for p in pkls:
            dst = working_cache_dir / p.name
            if not dst.exists():
                dst.write_bytes(p.read_bytes())
            copied.append(dst)
        return copied
    except Exception as e:
        print(f"WARNING: kagglehub download failed: {e}")
        return []


learn = None
chosen_path = None

if model_path.exists():
    chosen_path = model_path
else:
    _ = _try_kagglehub_download_to_cache()
    chosen_path = _find_any_learner_pkl(model_name)

if chosen_path is not None and Path(chosen_path).exists():
    learn = load_learner(chosen_path)
    print(f"Loaded learner from: {chosen_path}")
else:
    print(
        f"WARNING: No exported fastai learner (.pkl) found. Looked for {model_path}, /kaggle/working, and under /kaggle/input.\n"
        "Will write a valid submission with empty masks for all rows."
    )



## === cell 7
sample_to_rowid = defaultdict(dict)
for full_id, cls_name, base_id in zip(
    df_test["id"].astype(str).tolist(),
    df_test["class"].astype(str).tolist(),
    df_test["base_id"].astype(str).tolist(),
):
    sample_to_rowid[base_id][cls_name] = full_id

test_groups_all = group_case_day_from_files(TEST_FILES)

if DEBUG:
    dbg_keys = [
        k
        for k in test_groups_all.keys()
        if ("case123_day20" in k or "case77_day20" in k)
    ]
    test_groups = {k: test_groups_all[k] for k in dbg_keys}
else:
    test_groups = test_groups_all

packs = packed(test_groups, n_slices_to_merge=5, step_size=1)

len(TEST_FILES), len(packs), len(test_groups)



## === cell 8
try:
    import cupy as cp  # type: ignore

    _HAS_CUPY = True
except Exception:
    cp = None
    _HAS_CUPY = False

from skimage.morphology import disk
from scipy.ndimage import binary_opening

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if learn is not None:
    learn.dls.to(device)
    learn.model.to(device)
    learn.eval()


def mask2rle(mask: np.ndarray) -> str:
    mask = mask.astype(np.uint8)
    if _HAS_CUPY:
        m = cp.asarray(mask)
        pixels = m.reshape((-1,), order="F")
        pad = cp.array([0], dtype=pixels.dtype)
        pixels = cp.concatenate([pad, pixels, pad])
        runs = cp.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return " ".join(str(int(x)) for x in runs.get())
    else:
        pixels = mask.reshape((-1,), order="F")
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return " ".join(str(int(x)) for x in runs)


def rle2mask(mask_rle: str, shape: tuple[int, int]) -> np.ndarray:
    s = np.asarray(mask_rle.split(), dtype=int)
    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")


def pad_mask(mask: np.ndarray, image_size: int) -> np.ndarray:
    padded = np.zeros((image_size, image_size), dtype=mask.dtype)
    dh = image_size - mask.shape[0]
    dw = image_size - mask.shape[1]
    top = dh // 2
    left = dw // 2
    padded[top : top + mask.shape[0], left : left + mask.shape[1]] = mask
    return padded




## === cell 9
preds = []
if learn is not None:
    with learn.no_bar():
        batch_size = 64
        chunks = list(chunked(packs, n=batch_size))

        for subset in progress_bar(chunks):
            test_dl = learn.dls.test_dl(subset, batch_size=batch_size, device=device)
            logits, *_ = learn.get_preds(dl=test_dl, act=noop)
            labels = (torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(np.uint8)

            for pack, mask in zip(subset, labels):
                pack_base_ids = [get_sample_id(p) for p in pack]
                for base_id in pack_base_ids:
                    if base_id not in METADATA:
                        continue
                    m = METADATA[base_id]
                    h, w = m.h, m.w

                    rowid_by_class = sample_to_rowid.get(base_id, {})

                    for i, name in enumerate(("large_bowel", "small_bowel", "stomach")):
                        pred_hw = int(mask[i].shape[-1])
                        cls_mask = pad_mask(mask[i], pred_hw)

                        cls_mask = cv.resize(cls_mask, (w, h), cv.INTER_NEAREST)

                        opened = binary_opening(
                            cls_mask.astype(bool), structure=disk(5)
                        ).astype(np.uint8)
                        rle = mask2rle(opened) if opened.any() else ""

                        out_id = rowid_by_class.get(name, None)
                        if out_id is None:
                            continue
                        preds.append({"id": out_id, "class": name, "predicted": rle})

            del logits, labels
            gc.collect()

if learn is not None and len(preds) == 0:
    raise RuntimeError(
        "Learner was loaded but no predictions were generated. "
        "This would produce an all-empty submission (likely 0.0). "
        "Check test.csv id parsing / METADATA alignment and pack generation."
    )

len(preds), (preds[0] if len(preds) else None)



## === cell 10
df_submit = pd.read_csv(DATA_DIR / ("train.csv" if DEBUG else "test.csv"))[
    ["id", "class"]
].copy()

if len(preds) == 0:
    df_submit["predicted"] = ""
else:
    df_preds = pd.DataFrame(preds)
    for col in ["id", "class", "predicted"]:
        if col not in df_preds.columns:
            df_preds[col] = ""
    df_preds = df_preds[["id", "class", "predicted"]].copy()
    df_preds["id"] = df_preds["id"].astype(str)
    df_preds["class"] = df_preds["class"].astype(str)
    df_preds["predicted"] = df_preds["predicted"].fillna("").astype(str)

    submit_index = pd.MultiIndex.from_frame(df_submit[["id", "class"]])
    pred_index = pd.MultiIndex.from_frame(df_preds[["id", "class"]])

    df_preds = df_preds.loc[~pred_index.duplicated(keep="first")].copy()
    pred_index = pd.MultiIndex.from_frame(df_preds[["id", "class"]])

    pred_series = pd.Series(df_preds["predicted"].to_numpy(), index=pred_index)
    df_submit["predicted"] = pred_series.reindex(submit_index).fillna("").to_numpy()

out_path = Path("submission.csv")
df_submit.to_csv(out_path, index=False)

print(out_path, df_submit.shape)
print(df_submit.head(10).to_string(index=False))
print("Empty predicted rows:", int((df_submit["predicted"] == "").sum()))
print("Non-empty predicted rows:", int((df_submit["predicted"] != "").sum()))
print("Unique ids in submit:", df_submit["id"].nunique())
