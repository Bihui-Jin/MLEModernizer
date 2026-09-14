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

0.8376129787881307

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on the missing `/kaggle/input/uwm-libs` and missing third-party modules (`segmentation_models_pytorch`, `fast_ai_utils`, and `uwm.*`) so the notebook can run in your listed Kaggle environment. I keep the same inference core idea (load a FastAI `Learner` and predict masks → threshold at 0.5 → resize back → light morphological opening → RLE) and only add small compatibility shims (minimal `on_kaggle/equalize` fallbacks, robust DEBUG flag, missing imports). I also make the pipeline robust if the external model pickle is unavailable by producing an all-empty-mask submission (score be low, but it “yield” a valid submission as requested). Finally, I ensure `submission.csv` is written with the exact required columns and non-null `predicted` strings aligned to `sample_submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly coming from a submission/id misalignment: `get_sample_id()` currently produces IDs like `case110_day12_slice_0000`, but the competition/test.csv IDs include additional scan metadata (typically `_h_w_spacing1_spacing2`) and must match *exactly* for the evaluator to find your predictions. I minimally change the ID construction so it uses the true ID format derived from the image path (case/day + `slice_####` + the filename stem metadata), and I also ensure the pack’s center slice ID is computed from that same rule. This keeps your model/inference/post-processing intact but makes the predictions land on the correct rows, which should move the score up toward the target. I also add the missing `import os` so DEBUG/env handling is deterministic and doesn’t silently default.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with “mostly-empty predictions” caused by a pack→id mismatch: you’re only predicting one `id` per 5-slice pack (the center slice), but the submission expects predictions for every slice `id`. I keep your model, thresholding, resizing, morphology, and RLE exactly the same, but change inference to write predictions for *each* slice in the pack by reusing the appropriate channel from the 5-channel output (so every `id` in `sample_submission.csv` gets a non-missing prediction). I also make the mapping robust by building an `id->path` lookup from the actual file stems, and I preserve your empty-model fallback. This should move the score up substantially toward the target because the evaluator now receive aligned predictions for all required ids/classes instead of only a sparse subset.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with an ID-format mismatch: `get_sample_id()` currently duplicates `p.stem` after `slice_####`, producing ids that won’t match `test.csv/sample_submission.csv`, so almost all predictions get merged as empty. I minimally fix `get_sample_id()` to exactly match the competition’s true `id` format derived from the folder (`caseXXX_dayYY`) plus the PNG stem (`slice_####_H_W_s1_s2`) only once. To keep core inference identical but improve stability and avoid accidental empty predictions, I also build `ID2PATH/METADATA` from files and then restrict `paths` to those ids that appear in `sample_submission.csv` (ensuring 1:1 alignment). Everything else (model loading, 5-slice packing, threshold=0.5, resize/morphology, and RLE encoding) remains unchanged.'
- What this solution (achieved 0.0) has done: 'Your 0.0 is most likely because the 5-slice pack → per-slice prediction mapping is off: `labels` coming from the model are being indexed as if the slice dimension is axis=1, but with FastAI the batch dimension is first, so you need to pick the k-th slice **within** the correct axes (usually `labels[b, :, k, :, :]` not `labels[:, k, :, :]`). I make a minimal fix to correctly extract the per-slice, per-class mask for each slice in the pack while keeping your model, threshold=0.5, resizing, morphology, and RLE logic unchanged. I also ensure pack masks are iterated per item in the batch (so each pack uses its own prediction), which avoids silently reusing/reshaping wrong slices that can yield mostly-empty/incorrect masks. This should move the score up substantially toward the target without changing the core approach.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with predictions not being generated for most test IDs because `METADATA/ID2PATH` are built from `get_image_files(...)` which may miss nested `scans/` PNGs; then `available_ids` becomes small and the merge fills the rest with empty strings. I minimally change file discovery to explicitly glob `**/scans/*.png` (keeping the same ID construction, 5-slice packing, model inference, thresholding, morphology, and RLE), so every `sample_submission.csv` id can be found and predicted. I also avoid re-reading each image during inference by using the already-stored `h,w` from `METADATA` (no semantic change), which reduces overhead and helps finish within the time limit. Everything else (model, transforms, post-processing, CSV schema) is preserved.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly because you’re generating *per-class* RLEs, but the competition requires a *single binary mask per slice shared across all three class rows* (“masks for all objects… joined into a single large mask”). I keep your model/inference loop, thresholding (0.5), resizing, morphology, and RLE code intact, and only change the post-processing so the three class masks are unioned into one binary mask and then written to all three classes for that `id`. I also add a small safety fix so pack-mask axis handling is robust for both `[B,C,H,W]` and `[B,C,S,H,W]` outputs without changing the core computation. These minimal changes should move you off 0.0 and substantially closer to the target score by matching the evaluation semantics.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with a submission that’s effectively empty or malformed relative to what this competition expects: the leaderboard metric is computed per-class and per-slice, and it does not want you to union all three classes into one mask and paste it into every class row. I keep your exact inference core (FastAI learner → sigmoid → threshold 0.5 → pad to 320 → resize back → binary opening → RLE) but change only the post-processing/output so it emits one RLE per class (not unioned) and writes that class RLE only to the matching class row. I also make the prediction-shape handling deterministic for common FastAI segmentation outputs (supporting `[B,3,H,W]` and `[B,3,S,H,W]` with `S=5`) so per-slice masks are taken from the correct slice index `k`. These minimal changes should move the score substantially upward toward your target because the evaluator finally receive correctly-classed, correctly-aligned masks rather than “same union mask for all classes.”'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still consistent with “model not actually used” (missing `/kaggle/input/uwm-models/...pkl`) or a silent train/test path mismatch that leaves most IDs without real predictions. I keep your exact inference/post-processing logic, but make two minimal, score-relevant fixes: (1) automatically fall back to searching for the `.pkl` in common Kaggle input locations so inference actually runs when the file exists, and (2) ensure we generate predictions for *every* `id` in `sample_submission.csv` by building an `ID2PATH` lookup and iterating IDs directly (not via `packs` that can miss slices). This preserves the 5-slice packing approach (still predicts on 5-slice packs), but guarantees full coverage and correct alignment to the submission rows. The rest (threshold=0.5, padding to 320, resize back, binary opening, per-class RLE) stays unchanged.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the model never being loaded (so you submit all-empty masks), since the code only looks in a few fixed locations for the `.pkl`. I make a minimal, score-relevant change to search all `/kaggle/input/**` subfolders for the expected `{model_name}.pkl` and load it if found, without changing your inference/post-processing logic. Additionally, I ensure inference covers every `id` in `sample_submission.csv` by still iterating `TEST_IDS` but keeping the existing per-id 5-slice pack creation (no change to architecture/threshold/morphology/RLE). If the model still isn’t present, the fallback remains an all-empty submission (so it still runs end-to-end).'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the model producing predictions at 288×288 (due to your `valid_aug()` CenterCrop) but the code currently pads those predictions into a 320×320 canvas as if they were 320×320, which shifts them and can destroy overlap after resizing back—yielding near-empty/wrong masks and effectively a 0 score. I keep your exact model, thresholding, morphology, and RLE logic, but make the padding size match the *actual* model output size dynamically (and only pad if needed), so the masks stay correctly centered. I also keep the rest of your pipeline intact (IDs, packs, per-class output, CSV merge), so this is a minimal semantic fix expected to move the score upward toward the target. If the model pickle is still missing, the empty-mask fallback remains unchanged.'

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
import os  # required for DEBUG env flag
import gc
from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import cv2 as cv
import numpy as np
import pandas as pd
import torch

from fastai.vision.all import (
    Transform,
    ItemTransform,
    TensorImage,
    TensorMask,
    show_image,
    progress_bar,
    load_learner,
    noop,
)

import re
from collections import defaultdict
from functools import partial
from more_itertools import windowed, chunked


def on_kaggle() -> bool:
    return Path("/kaggle").exists()


def equalize(x: np.ndarray) -> np.ndarray:
    return x


def get_case_day(s) -> str:
    return re.search(r"case\d+_day\d+", str(s)).group()


def get_slice(s, as_number: bool = False):
    m = re.search(r"slice_\d\d\d\d", str(s)).group()
    return int(m.split("_")[-1]) if as_number else m


def get_sample_id(s) -> str:
    """
    Ensure IDs match Kaggle's test.csv/sample_submission.csv exactly.
    Format: "{caseXXX_dayYY}_{png_stem}" where png_stem includes "slice_####_H_W_s1_s2".
    """
    p = Path(s)
    case_day = get_case_day(p)
    return f"{case_day}_{p.stem}"


def get_size_from_filename(fn: Path):
    parts = fn.stem.split("_")
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        w, h = int(parts[0]), int(parts[1])
        return h, w
    img = cv.imread(str(fn), cv.IMREAD_UNCHANGED)
    return img.shape[:2]


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
        files = [None] + list(files) + [None]
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
        h, w = get_size_from_filename(path)
        sample_id = get_sample_id(path)
        return Metadata(sample_id, str(path), int(h), int(w))




## === cell 3
DATA_DIR = Path("/kaggle/input/uw-madison-gi-tract-image-segmentation/")

DEBUG = bool(int(os.environ.get("DEBUG_UWM", "0")))

ids_source = "train.csv" if DEBUG else "sample_submission.csv"
TEST_IDS = pd.read_csv(DATA_DIR / ids_source)["id"].drop_duplicates().tolist()

scan_root = DATA_DIR / ("train" if DEBUG else "test")

TEST_FILES = sorted(scan_root.glob("**/scans/*.png"))

ID2PATH = {get_sample_id(p): p for p in TEST_FILES}
METADATA = {k: Metadata.extract(v) for k, v in ID2PATH.items()}

print(
    "n_TEST_IDS:",
    len(TEST_IDS),
    "n_TEST_FILES:",
    len(TEST_FILES),
    "n_metadata:",
    len(METADATA),
    "n_ids_with_metadata:",
    sum(1 for i in TEST_IDS if i in METADATA),
)




## === cell 4
class CreateSample(Transform):
    def encodes(self, pack):
        h, w = get_size_from_filename(pack[0])
        merged = np.ndarray((h, w, len(pack)), dtype=np.uint8)
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
    return AugTrain(A.Compose([A.Resize(320, 320), A.RandomCrop(288, 288)]))


def valid_aug():
    return AugValid(A.Compose([A.Resize(320, 320), A.CenterCrop(288, 288)]))




## === cell 5
model_name = "5ch_1c_e1_cos_e12_bce_dice"
model_path = Path(f"/kaggle/input/uwm-models/{model_name}.pkl")

candidate_model_paths = [
    model_path,
    Path(f"/kaggle/input/{model_name}/{model_name}.pkl"),
    Path(f"/kaggle/input/{model_name}.pkl"),
    Path(f"/kaggle/input/uw-madison-gi-tract-image-segmentation/{model_name}.pkl"),
]

input_root = Path("/kaggle/input")
if input_root.exists():
    try:
        for p in input_root.rglob(f"{model_name}.pkl"):
            candidate_model_paths.append(p)
    except Exception as e:
        print("WARNING: rglob model search failed:", repr(e))

_seen = set()
candidate_model_paths = [
    p for p in candidate_model_paths if not (str(p) in _seen or _seen.add(str(p)))
]

found_model_path = next((p for p in candidate_model_paths if p.exists()), None)

learn = None
if found_model_path is not None:
    print("Loading model from:", found_model_path)
    learn = load_learner(found_model_path)
else:
    print(
        f"WARNING: model not found. Tried: {[str(p) for p in candidate_model_paths]}. "
        "Will create empty-mask submission."
    )



## === cell 6
if DEBUG:
    predicted_ids = [
        id_
        for id_ in TEST_IDS
        if id_.startswith("case123_day20") or id_.startswith("case77_day20")
    ]
else:
    predicted_ids = TEST_IDS

print("predicted_ids:", len(predicted_ids))



## === cell 7
available_ids = [id_ for id_ in predicted_ids if id_ in ID2PATH]
missing_ids = [id_ for id_ in predicted_ids if id_ not in ID2PATH]
print("available ids:", len(available_ids), "missing ids:", len(missing_ids))

paths_for_available = [ID2PATH[i] for i in available_ids]
groups = group_case_day_from_files(paths_for_available)

FILE2IDX = {}
for cd, files in groups.items():
    for j, p in enumerate(files):
        FILE2IDX[p] = j


def make_pack_for_path(p: Path, n=5):
    assert n % 2 == 1
    cd = get_case_day(p)
    files = groups[cd]
    mid = FILE2IDX[p]
    r = n // 2
    pack = []
    for t in range(mid - r, mid + r + 1):
        tt = min(max(t, 0), len(files) - 1)
        pack.append(files[tt])
    return pack


packs_by_id = {id_: make_pack_for_path(ID2PATH[id_], n=5) for id_ in available_ids}
print("packs_by_id:", len(packs_by_id))



## === cell 8
import cupy as cp


def mask2rle(mask):
    """
    mask: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted (1-indexed, column-major as required by competition).
    """
    mask = np.asarray(mask, dtype=np.uint8)
    mask_f = np.asfortranarray(mask)
    pixels = cp.asarray(mask_f).reshape(-1, order="F")
    pad = cp.array([0], dtype=pixels.dtype)
    pixels = cp.concatenate([pad, pixels, pad])
    runs = cp.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(map(str, cp.asnumpy(runs).tolist()))


def rle2mask(mask_rle, shape):
    s = np.asarray(mask_rle.split(), dtype=int)
    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")


def pad_mask_to_size(mask, image_size: int):
    if mask.shape[0] == image_size and mask.shape[1] == image_size:
        return mask
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
class_names = ("large_bowel", "small_bowel", "stomach")

if learn is not None:
    learn.dls.to(device)
    learn.model.to(device)
    learn.eval()

    with learn.no_bar():
        batch_size = 64
        id_list = list(packs_by_id.keys())
        for id_chunk in progress_bar(list(chunked(id_list, n=batch_size))):
            subset_packs = [packs_by_id[i] for i in id_chunk]
            test_dl = learn.dls.test_dl(
                subset_packs, batch_size=batch_size, device=device
            )
            logits, *_ = learn.get_preds(dl=test_dl, act=noop)

            labels = (
                (torch.sigmoid(logits) >= 0.5).detach().cpu().numpy().astype(np.uint8)
            )

            for test_id, pack, pack_mask in zip(id_chunk, subset_packs, labels):
                if test_id not in METADATA:
                    continue
                m = METADATA[test_id]
                h, w = m.h, m.w

                k = len(pack) // 2

                if pack_mask.ndim == 3:
                    cls_stack = pack_mask  # [3,H,W]
                elif pack_mask.ndim == 4:
                    if pack_mask.shape[0] == 3 and pack_mask.shape[1] == len(pack):
                        cls_stack = pack_mask[:, k, :, :]  # [3,H,W]
                    elif pack_mask.shape[1] == 3 and pack_mask.shape[0] == len(pack):
                        cls_stack = pack_mask[k, :, :, :]  # [3,H,W]
                    else:
                        raise RuntimeError(
                            f"Unexpected 4D prediction shape {pack_mask.shape}"
                        )
                else:
                    raise RuntimeError(f"Unexpected prediction ndim={pack_mask.ndim}")

                out_h, out_w = int(cls_stack.shape[-2]), int(cls_stack.shape[-1])
                assert (
                    out_h == out_w
                ), f"Expected square model output, got {cls_stack.shape}"

                for i, name in enumerate(class_names):
                    cls_mask = pad_mask_to_size(cls_stack[i], out_h)
                    cls_mask = cv.resize(cls_mask, (w, h), cv.INTER_NEAREST)
                    cls_mask = binary_opening(
                        cls_mask.astype(bool), structure=disk(5)
                    ).astype(np.uint8)
                    rle = mask2rle(cls_mask) if cls_mask.any() else ""
                    preds.append({"id": test_id, "class": name, "predicted": rle})

            del logits, labels
            gc.collect()
else:
    for test_id in predicted_ids:
        for name in class_names:
            preds.append({"id": test_id, "class": name, "predicted": ""})

df_preds = pd.DataFrame(preds)
print(df_preds.head(), df_preds.shape)



## === cell 10
df_submit = pd.read_csv(DATA_DIR / "sample_submission.csv")

df_preds = df_preds.drop_duplicates(subset=["id", "class"], keep="last")

df_submit = df_submit.drop(columns=["predicted"]).merge(
    df_preds, on=["id", "class"], how="left"
)
df_submit["predicted"] = df_submit["predicted"].fillna("")

out_path = Path("submission.csv")
df_submit.to_csv(out_path, index=False)

print("Wrote:", out_path.resolve())
print(df_submit.head(10))
print("Rows:", len(df_submit), "Cols:", df_submit.columns.tolist())
assert out_path.exists() and out_path.suffix == ".csv"
assert list(df_submit.columns) == ["id", "class", "predicted"]
assert len(df_submit) == 20400
assert df_submit["predicted"].isna().sum() == 0
