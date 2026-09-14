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

0.8426629557225797

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.34195) has done: 'The crash comes from batching variable-sized images: different slices have different (H,W), so fastai’s default collate can’t stack them. I add a minimal after-item transform that pads/crops every merged 5-slice sample to a fixed 320×320 so batching works, and I ensure the same transform is applied both in the fallback learner and when creating the test dataloader from a loaded `.pkl`. Finally, I keep the rest of the inference/RLE logic intact and make sure `df_preds` is always created so `submission.csv` is written successfully.'
- What this solution (achieved 0.22099) has done: 'Your current score gap is large (0.34195 vs target 0.84266), and the biggest issue is that the inference pipeline is not using a real segmentation model when the `.pkl` is missing (it falls back to a tiny dummy conv net), which cap performance far below target. I keep your overall packing/5-slice stacking, resizing, and RLE submission logic intact, but I (1) reliably load the competition’s common public baseline weights if available, and (2) fix one post-processing bug: you’re padding masks to 320 even though they are already 320 after `PadCropToSquare`, which can distort predictions when resizing back to (h,w). These are minimal, evaluation-relevant changes that should substantially increase Dice/Hausdorff toward the target without changing the training approach (still inference-only here) or the core data flow.'
- What this solution (achieved 0.0) has done: 'Your current gap to the target is very large (0.22099 vs 0.84266, higher-is-better), and the biggest likely score killer in this inference-only pipeline is incorrect probability handling: this is a 3-class *mutually exclusive* segmentation in most fastai baselines (softmax over 3 channels), but the code always applies `sigmoid`, which tends to over/under-segment and hurts both Dice and Hausdorff. I make a minimal, evaluation-relevant change to use `softmax` when the model output looks like a 3-channel multiclass logit map, while keeping `sigmoid` as a fallback for true multilabel heads. I also make the morphological opening less aggressive (disk radius 2 instead of 5) to avoid erasing thin structures, which should improve Dice without changing your core model/inference flow. Everything else (5-slice packing, 320 pad/crop, resizing back to original (h,w), RLE formatting, and submission merge) stays the same and still writes `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is most consistent with an invalid/incompatible prediction format for this competition: the metric expects a **single binary mask per image per class**, but your code thresholds **per-class probabilities** and RLE-encodes each class separately without ensuring binary encoding consistency and correct per-image alignment. I make two minimal, evaluation-relevant fixes: (1) ensure the predicted masks are **strictly binary {0,1}** after resizing and morphology, and (2) ensure every `(id,class)` pair appears exactly once and in the same order as `sample_submission.csv` (preventing any silent misalignment that can yield a 0). I not change your model, packing, transforms, or inference loop structure—only the post-processing and submission assembly to produce a valid, correctly-aligned submission.'
- What this solution (achieved 0.0) has done: 'Your pipeline crashes because it hard-requires an external `/kaggle/input/uwm-models/*.pkl` that is not present, so `learn` stays `None` and everything downstream fails (no preds, no submission). I keep your exact inference data flow (5-slice packing, 320 pad/crop, per-class thresholding, morphology, RLE, and merge with `sample_submission.csv`) but make the model-loading robust: first try to load a learner `.pkl` from common locations; if none exists, fall back to a simple, deterministic baseline that still produces valid masks and a valid `submission.csv` (score be low but not a runtime failure). I also fix the `learn.no_bar()` usage to work even when `learn` is absent and ensure `df_preds` is always created so the final CSV is always written. These changes are minimal and focused on unblocking end-to-end execution and producing a valid submission file.'
- What this solution (achieved 0.03276) has done: 'Your current 0.0 score is most consistent with a submission that’s *syntactically valid* but *semantically empty* (all masks blank) because the code falls back to empty predictions whenever the exported learner `.pkl` isn’t found. To move the score upward toward the target with minimal changes and without changing your core inference flow, I make the fallback use a simple, deterministic, image-derived mask (Otsu threshold + light morphology) instead of always-empty masks, while keeping the same 5-slice packing and per-(id,class) RLE submission assembly. I also ensure the `(id,class)` alignment is exactly the `sample_submission.csv` order (already mostly done) and keep the existing model path-loading logic intact. This should move you from 0.0 to a non-trivial score while staying within Kaggle constraints and still producing `submission.csv` end-to-end.'
- What this solution (achieved 0.0) has done: 'The crash happens because `_find_image_path_from_id` tries to locate training images at `.../scans/slice_XXXX.png`, but the dataset’s actual filenames are not `slice_*.png` (they’re like `276_276_1.63_1.63.png`), so most paths become `None` and fastai’s `Datasets` setup breaks. I fix this minimally by building an index of available scan files per `case_day` and using the slice order to map `..._slice_####` to the correct filename, keeping the same 5-slice packing and the same U-Net learner/training approach. I also make `get_size_from_filename` robust by reading image shape directly (so resizing back uses the true (h,w) even when filename parsing is wrong). Finally, I ensure `df_preds` is always defined (even if something goes wrong) so `submission.csv` is always written in the correct format.'
- What this solution (achieved 0.0) has done: 'The crash is caused by the training `DataBlock` applying `CreateSample()` to both `x` (a list of Paths) and `y` (a `TensorMask`), because `TransformBlock` lets item transforms run on all components; `CreateSample` then receives a `PILMask`/mask object and fails when indexing. I fix this minimally by making the `x` block explicitly an `ImageBlock(cls=TensorImageNChannels)` and ensuring `CreateSample` returns a `TensorImageNChannels`, so `CreateSample/PadCropToSquare/ChannelsFirst` only apply to the image input. I also make `PadCropToSquare` accept torch tensors (converting to numpy) so it works consistently across both training and inference pipelines. Finally, I keep your inference + softmax/sigmoid logic intact and ensure `submission.csv` is always written with exact `sample_submission.csv` ordering/alignment.'

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
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd
import cv2 as cv
import torch
import torch.nn as nn

from fastai.vision.all import *
from more_itertools import windowed, chunked


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
        case_and_day = path.parents[1].stem  # e.g. case110_day12
        parts = path.stem.split("_")
        if len(parts) >= 4:
            w, h = int(parts[0]), int(parts[1])
        else:
            img = cv.imread(str(path), cv.IMREAD_UNCHANGED)
            h, w = img.shape[:2]
        m = re.search(r"slice_(\d+)", str(path))
        slice_no = int(m.group(1)) if m else 0
        sample_id = f"{case_and_day}_slice_{slice_no:04d}"
        return Metadata(sample_id, str(path), int(h), int(w))




## === cell 3
DATA_DIR = Path("/kaggle/input/uw-madison-gi-tract-image-segmentation/")

DEBUG = False

TEST_IDS = (
    pd.read_csv(DATA_DIR / "sample_submission.csv")["id"].drop_duplicates().tolist()
)

TEST_FILES = get_image_files(DATA_DIR / "test")

METADATA_BY_PATH = {
    str(p): (cv.imread(str(p), cv.IMREAD_UNCHANGED).shape[:2], p) for p in TEST_FILES
}


def get_case_day_from_path(s):
    return re.search(r"case\d+_day\d+", str(s)).group()


def get_size_from_filename(p: Path):
    img = cv.imread(str(p), cv.IMREAD_UNCHANGED)
    if img is None:
        return (320, 320)
    return img.shape[:2]




## === cell 4
def get_case_day(s):
    return re.search(r"case\d+_day\d+", str(s)).group()


def _slice_no_from_path(p: Path) -> int:
    m = re.search(r"slice_(\d+)", str(p))
    return int(m.group(1)) if m else -1


def group_case_day_from_files(image_files):
    groups = defaultdict(list)
    for fn in image_files:
        groups[get_case_day(fn)].append(fn)
    groups = {k: sorted(v, key=_slice_no_from_path) for k, v in groups.items()}
    return groups


def packed(groups, n_slices_to_merge=5, step_size=1):
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


def get_sample_id_from_pack(pack, groups_index_map):
    mid = pack[len(pack) // 2]
    case_day = get_case_day(mid)
    slice_idx = groups_index_map[case_day][str(mid)]
    return f"{case_day}_slice_{slice_idx:04d}"




## === cell 5
class CreateSample(Transform):
    def encodes(self, pack):
        img0 = cv.imread(str(pack[0]), cv.IMREAD_UNCHANGED)
        h, w = img0.shape[:2]
        merged = np.empty((h, w, len(pack)), dtype=np.uint8)
        q = 0.01
        for i, fn in enumerate(pack):
            img = cv.imread(str(fn), cv.IMREAD_UNCHANGED)
            lo, hi = np.percentile(img, [q * 100, (1 - q) * 100])
            img = np.clip(img, lo, hi)
            v_min, v_max = float(np.min(img)), float(np.max(img))
            if v_max > v_min:
                img = (img - v_min) / (v_max - v_min)
            else:
                img = img * 0.0
            img = (img * 255.0).astype(np.uint8)
            merged[:, :, i] = img
        return TensorImageNChannels(merged)


class TensorImageNChannels(TensorImage):
    pass


class ChannelsFirst(ItemTransform):
    def encodes(self, x):
        return x.permute(2, 0, 1)  # HWC->CHW

    def decodes(self, x):
        return x.permute(1, 2, 0)


class NormalizeSample(Transform):
    order = 99

    def setups(self, *args, **kwargs):
        self.mean, self.std = 0.18161897, 0.257913

    def encodes(self, x: TensorImageNChannels):
        return (x - self.mean) / self.std

    def decodes(self, x: TensorImageNChannels):
        return x * self.std + self.mean


class PadCropToSquare(Transform):
    def __init__(self, size=320, pad_mode=cv.BORDER_CONSTANT, pad_value=0):
        self.size = int(size)
        self.pad_mode = pad_mode
        self.pad_value = pad_value

    def encodes(self, x):
        is_tensor = torch.is_tensor(x)
        if is_tensor:
            x_np = x.detach().cpu().numpy()
        else:
            x_np = x

        h, w = x_np.shape[:2]
        target = self.size

        if h > target:
            top = (h - target) // 2
            x_np = x_np[top : top + target, :, :]
            h = target
        if w > target:
            left = (w - target) // 2
            x_np = x_np[:, left : left + target, :]
            w = target

        dh = target - h
        dw = target - w
        if dh > 0 or dw > 0:
            top = dh // 2
            bottom = dh - top
            left = dw // 2
            right = dw - left
            x_np = cv.copyMakeBorder(
                x_np,
                top,
                bottom,
                left,
                right,
                borderType=self.pad_mode,
                value=self.pad_value,
            )

        if is_tensor:
            return type(x)(torch.from_numpy(x_np))
        return x_np




## === cell 6
model_name = "new"

candidate_pkls = [
    Path(f"/kaggle/input/uwm-models/{model_name}.pkl"),
    Path("/kaggle/input/uwm-models/model.pkl"),
    Path("/kaggle/input/uwm-models/best.pkl"),
    Path("/kaggle/input/uwm-models/export.pkl"),
    Path("/kaggle/input/uw-madison-gi-tract-image-segmentation/export.pkl"),
    Path("/kaggle/input/uw-madison-gi-tract-image-segmentation/model.pkl"),
    Path("/kaggle/working/export.pkl"),
    Path("/kaggle/working/model.pkl"),
    Path("/kaggle/working/best.pkl"),
]

for p in list(DATA_DIR.glob("*.pkl")) + list((DATA_DIR / "models").glob("*.pkl")):
    candidate_pkls.append(p)

learn = None
loaded_from = None
for pkl_path in candidate_pkls:
    if pkl_path.exists():
        try:
            learn = load_learner(pkl_path)
            loaded_from = str(pkl_path)
            break
        except Exception as e:
            print(f"Failed to load {pkl_path}: {type(e).__name__}: {e}")

print(
    "Loaded learner:",
    (
        loaded_from
        if loaded_from
        else "None (will train a small baseline learner from train.csv)"
    ),
)



## === cell 7
groups = group_case_day_from_files(TEST_FILES)

groups_index_map = {}
for case_day, files in groups.items():
    groups_index_map[case_day] = {str(p): i for i, p in enumerate(files)}

packs = packed(groups, n_slices_to_merge=5, step_size=1)

all_pack_ids = [get_sample_id_from_pack(pack, groups_index_map) for pack in packs]
wanted_ids = set(TEST_IDS)
packs = [pack for pack, sid in zip(packs, all_pack_ids) if sid in wanted_ids]

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if learn is not None:
    learn.to(device)
    learn.model.eval()

len(packs), device



## === cell 8
from skimage.morphology import disk
from scipy.ndimage import binary_opening, binary_closing


def mask2rle(mask: np.ndarray) -> str:
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    pixels = mask.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]]).astype(np.uint8)
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(map(str, runs.tolist()))


def rle2mask(mask_rle, shape):
    s = np.asarray(mask_rle.split(), dtype=int)
    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


def pad_mask(mask, image_size):
    padded = np.zeros((image_size, image_size), dtype=mask.dtype)
    dh = image_size - mask.shape[0]
    dw = image_size - mask.shape[1]
    top = dh // 2
    left = dw // 2
    padded[top : top + mask.shape[0], left : left + mask.shape[1]] = mask
    return padded




## === cell 9
def build_scan_index(split: str):
    base = DATA_DIR / split
    groups = defaultdict(list)
    if not base.exists():
        return {}
    for p in get_image_files(base):
        if p.parent.name != "scans":
            continue
        case_day = get_case_day(p)
        groups[case_day].append(p)
    return {k: sorted(v) for k, v in groups.items()}


TRAIN_SCAN_INDEX = build_scan_index("train")


def _find_image_path_from_id(sample_id: str, split: str) -> Path:
    m = re.match(r"(case\d+_day\d+)_slice_(\d+)", sample_id)
    if m is None:
        return None
    case_day, slice_no = m.group(1), int(m.group(2))

    if split == "train":
        files = TRAIN_SCAN_INDEX.get(case_day, None)
        if files is None or len(files) == 0:
            return None
        if 0 <= slice_no < len(files):
            return files[slice_no]
        return None

    base = DATA_DIR / split / case_day.split("_")[0] / case_day / "scans"
    if not base.exists():
        return None
    files = sorted(get_image_files(base))
    if 0 <= slice_no < len(files):
        return files[slice_no]
    return None


class RLEtoMask(Transform):
    def __init__(self, class_name: str, size=320):
        self.class_name = class_name
        self.size = int(size)

    def encodes(self, rle: str):
        if (
            rle is None
            or (isinstance(rle, float) and np.isnan(rle))
            or str(rle).strip() == ""
        ):
            m = np.zeros((self.size, self.size), dtype=np.uint8)
        else:
            m = rle2mask(str(rle), (self.size, self.size)).astype(np.uint8)
        return TensorMask(m)


def _build_train_learner(train_n_images=800, bs=8):
    df = pd.read_csv(DATA_DIR / "train.csv")
    df["path"] = df["id"].apply(lambda x: _find_image_path_from_id(x, "train"))
    df = df[df["path"].notna()].copy()

    pivot = df.pivot_table(
        index=["id", "path"], columns="class", values="segmentation", aggfunc="first"
    ).reset_index()
    for c in ["large_bowel", "small_bowel", "stomach"]:
        if c not in pivot.columns:
            pivot[c] = ""
        pivot[c] = pivot[c].fillna("")

    pivot = pivot.sort_values("id").head(int(train_n_images)).reset_index(drop=True)

    def get_x(row):
        sid = row["id"]
        m = re.match(r"(case\d+_day\d+)_slice_(\d+)", sid)
        case_day, slice_no = m.group(1), int(m.group(2))
        files = TRAIN_SCAN_INDEX.get(case_day, None)
        if files is None or len(files) == 0:
            return [row["path"]] * 5

        center_idx = min(max(slice_no, 0), len(files) - 1)
        center = files[center_idx]

        pack = []
        for d in [-2, -1, 0, 1, 2]:
            idx = center_idx + d
            if 0 <= idx < len(files):
                pack.append(files[idx])
            else:
                pack.append(center)
        return pack

    def get_y(row):
        m = np.zeros((320, 320), dtype=np.uint8)
        for idx, cls in enumerate(["large_bowel", "small_bowel", "stomach"], start=1):
            rle = row[cls]
            if isinstance(rle, str) and rle.strip() != "":
                mm = rle2mask(rle, (320, 320)).astype(np.uint8)
                m[mm > 0] = idx
        return TensorMask(m)

    create = CreateSample()
    fixsz = PadCropToSquare(320)

    batch_tfms = [IntToFloatTensor(), Normalize.from_stats(0.18161897, 0.257913)]

    dblock = DataBlock(
        blocks=(
            ImageBlock(cls=TensorImageNChannels),
            MaskBlock(codes=["background", "large_bowel", "small_bowel", "stomach"]),
        ),
        get_x=get_x,
        get_y=get_y,
        splitter=RandomSplitter(valid_pct=0.1, seed=42),
        item_tfms=[create, fixsz, ChannelsFirst()],
        batch_tfms=batch_tfms,
    )
    dls = dblock.dataloaders(pivot.to_dict(orient="records"), bs=bs, num_workers=0)

    learn = unet_learner(
        dls, resnet34, n_out=4, pretrained=True, loss_func=CrossEntropyLossFlat(axis=1)
    )
    learn.to(device)
    return learn


preds = []
df_preds = pd.DataFrame(columns=["id", "class", "predicted"])

if learn is None:
    learn = _build_train_learner(train_n_images=800, bs=8)
    learn.fine_tune(2, base_lr=2e-3)
    learn.model.eval()

batch_size = 64

create = CreateSample()
fixsz = PadCropToSquare(320)

norm = None
if learn is not None and hasattr(learn, "dls") and hasattr(learn.dls, "after_batch"):
    for tfm in listify(learn.dls.after_batch.fs):
        if isinstance(tfm, Normalize):
            norm = tfm
            break
if norm is None:
    norm = Normalize.from_stats(0.18161897, 0.257913)


def _to_chw_float(b):
    if isinstance(b, (tuple, list)):
        b = b[0]
    b = b.float() / 255.0
    if b.ndim == 4:  # (bs,H,W,C) -> (bs,C,H,W)
        b = b.permute(0, 3, 1, 2).contiguous()
    return b


CLASS_THRESH = {"large_bowel": 0.35, "small_bowel": 0.35, "stomach": 0.35}

with learn.no_bar():
    for subset in progress_bar(list(chunked(packs, n=batch_size))):
        test_dl = learn.dls.test_dl(
            subset,
            batch_size=batch_size,
            device=device,
            num_workers=0,
            after_item=[create, fixsz],
            after_batch=[_to_chw_float, norm],
        )
        logits, *_ = learn.get_preds(dl=test_dl, act=noop)

        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if logits.ndim == 3:
            logits = logits.unsqueeze(1)

        if logits.ndim == 4 and logits.shape[1] == 4:
            p = torch.softmax(logits, dim=1)[:, 1:4]  # drop background
            probs = p
        elif logits.ndim == 4 and logits.shape[1] == 3:
            probs = torch.softmax(logits, dim=1)
        else:
            probs = torch.sigmoid(logits)

        if probs.ndim == 3:
            probs = probs.unsqueeze(1)
        if probs.shape[1] != 3:
            if probs.shape[1] > 3:
                probs = probs[:, :3]
            else:
                probs = probs.repeat(1, 3 // probs.shape[1] + 1, 1, 1)[:, :3]

        probs_np = probs.detach().cpu().numpy()

        for pack, prob_map in zip(subset, probs_np):
            test_id = get_sample_id_from_pack(pack, groups_index_map)
            mid_path = pack[len(pack) // 2]
            h, w = get_size_from_filename(mid_path)

            for i, name in enumerate(("large_bowel", "small_bowel", "stomach")):
                thr = float(CLASS_THRESH[name])

                cls_mask = (prob_map[i] >= thr).astype(np.uint8)

                cls_mask = cv.resize(cls_mask, (w, h), interpolation=cv.INTER_NEAREST)
                cls_mask = (cls_mask > 0).astype(np.uint8)

                opened = binary_opening(cls_mask.astype(bool), structure=disk(2))
                opened = (opened.astype(np.uint8) > 0).astype(np.uint8)

                rle = mask2rle(opened) if opened.any() else ""
                preds.append({"id": test_id, "class": name, "predicted": rle})

        del logits, probs, probs_np
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

df_preds = pd.DataFrame(preds)
df_preds.head()



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_56/354777513.py in <cell line: 0>()
    129 
    130 if learn is None:
--> 131     learn = _build_train_learner(train_n_images=800, bs=8)
    132     learn.fine_tune(2, base_lr=2e-3)
    133     learn.model.eval()

/tmp/ipykernel_56/354777513.py in _build_train_learner(train_n_images, bs)
    107     dblock = DataBlock(
    108         blocks=(
--> 109             ImageBlock(cls=TensorImageNChannels),
    110             MaskBlock(codes=["background", "large_bowel", "small_bowel", "stomach"]),
    111         ),

/usr/local/lib/python3.11/dist-packages/fastai/vision/data.py in ImageBlock(cls)
     82 def ImageBlock(cls:PILBase=PILImage):
     83     "A `TransformBlock` for images of `cls`"
---> 84     return TransformBlock(type_tfms=cls.create, batch_tfms=IntToFloatTensor)
     85 
     86 # %% ../../nbs/08_vision.data.ipynb 21

AttributeError: type object 'TensorImageNChannels' has no attribute 'create'

## === cell 10
df_submit = pd.read_csv(DATA_DIR / "sample_submission.csv")

df_preds = df_preds.drop_duplicates(subset=["id", "class"], keep="last")

df_submit = df_submit.drop(columns="predicted").merge(
    df_preds, on=["id", "class"], how="left", validate="one_to_one"
)
df_submit["predicted"] = df_submit["predicted"].fillna("")

assert df_submit.shape[0] == pd.read_csv(DATA_DIR / "sample_submission.csv").shape[0]
assert list(df_submit.columns) == ["id", "class", "predicted"]

df_submit.to_csv("submission.csv", index=False)

print(df_submit.shape)
print(df_submit.head(3))
print(Path("submission.csv").resolve())
