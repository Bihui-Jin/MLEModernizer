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

0.8426658047635575

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on the missing `uwm` package and the missing `/kaggle/input/uwm-models/*.pkl` by providing small local replacements for the needed functions and switching to a deterministic, valid “empty-mask” submission fallback when a model can’t be loaded. I also fix the `Metadata.extract` parsing bug that crashes on filenames like `slice_XXXX_...png` by handling both naming patterns robustly. Finally, I make sure the pipeline always builds `METADATA`, aligns predictions with `sample_submission.csv`, and writes a correctly formatted `submission.csv` with the required columns and a `.csv` suffix.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is coming from always falling back to an all-empty submission because the model file isn’t found/loaded, so the smallest legitimate way to move toward the target is to actually run inference with a model checkpoint if it exists somewhere in the provided input tree. I keep your inference pipeline and post-processing intact, but make model discovery more robust by searching `/kaggle/input/**/` for `{model_name}.pkl` and loading the first match (still using `load_learner`). I also ensure the prediction ids are exactly those in `sample_submission.csv` (not train ids) and deduplicate any accidental repeats so the merge cannot create missing/duplicated rows, producing a valid `submission.csv`. These changes are minimal, preserve the core logic, and should raise the score above 0.0 when a compatible `.pkl` is present.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly because the model isn’t being loaded (so you submit all-empty masks), not because of the post-processing itself. I keep your inference pipeline and thresholding/morphology intact, but make checkpoint loading robust by searching for both `.pkl` and `.pth` under `/kaggle/input/**` and supporting loading a Torch `state_dict` into the existing learner architecture when only `.pth` is available. I also fix a critical submission-ID mismatch: your `get_sample_id()` currently produces `caseXX_dayYY_slice_0001` but the competition ids are `caseXX_dayYY_slice_0001` (with `slice_` already in `get_slice()`), so we ensure it exactly matches `Metadata.extract()` and `sample_submission.csv` ids. These minimal fixes should move the score up from 0.0 toward the target by enabling real predictions and correct row alignment.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with the code producing an all-empty submission because `learn` stays `None` when no compatible exported learner is found/loaded. The smallest legitimate change to move toward the target is to reliably locate and load the correct exported `.pkl` for `model_name` (instead of reconstructing from an arbitrary first `.pkl`, which usually fails silently or yields wrong outputs). I tighten model discovery to (1) prefer an exact `{model_name}.pkl` anywhere under `/kaggle/input`, (2) only fall back to state_dict loading when a matching architecture export is available, and (3) fail loudly enough to avoid accidentally submitting empties when a model exists. This preserves your inference pipeline, thresholding, morphology, and submission formatting, but should move the score up from 0.0 by enabling real predictions.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with an “all-empty” submission (or effectively empty due to ID mismatches), so the smallest change to move toward the target is to (1) ensure the predicted `id` strings *exactly* match `sample_submission.csv`, and (2) ensure we generate one prediction per required `(id, class)` row (not per merged-slice pack). I keep your same model loading, transforms, thresholding (0.4), and morphology, but change pack creation so it yields exactly one pack per slice (centered window with edge padding) and uses `Metadata.sample_id` for the `id`. This preserves your inference approach (multi-slice merge) while fixing the alignment that otherwise collapses coverage and tanks the score. Finally, I add a safety reindex against `sample_submission.csv` to guarantee full row coverage and valid CSV output.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the submission being effectively empty/incorrect due to a mismatch between the model’s expected input pipeline and the inference `test_dl` pipeline (your exported learner likely expects `CreateSample`/`valid_aug`/`ChannelsFirst` transforms, but `test_dl(subset)` is currently fed raw lists of Paths). I make the smallest change to ensure inference uses the learner’s own `after_item/after_batch` pipeline by passing a `Datasets` built from your packs with `CreateSample()` applied, so the model receives the same shaped tensors it was trained on. I also preserve your thresholding (0.4), morphology, and submission alignment, and keep the empty-mask fallback unchanged in case the model still can’t be loaded. These changes should move the score up from 0.0 toward the target by enabling real, non-empty predictions when the checkpoint is present.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with the model producing predictions that are never credited (most commonly because the RLE encoding is in the wrong pixel order for this competition). I keep your model loading, inference, thresholding (0.4), and morphology unchanged, and only fix the RLE encoder to use the competition’s required **column-major (Fortran) order** (top-to-bottom then left-to-right). This is a minimal, directly score-impacting change that typically moves a “looks fine locally but scores 0.0” submission into a reasonable range. I also keep the rest of your submission alignment logic intact so the CSV remains valid.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the submission encoding being invalid for the competition (wrong flattening/pixel order), or with the model never being used due to missing/failed export loading. Your code already includes a Fortran-order RLE encoder and robust submission alignment, so the smallest score-moving change is to ensure the **inference data pipeline exactly matches what the exported fastai Learner expects** (so predictions aren’t garbage/empty even when the model loads). I keep your model, threshold (0.4), morphology, pack-building, and submission merge logic intact, but change the test dataloader creation to use the learner’s own `test_dl(subset)` with your `CreateSample()` transform applied as `after_item`, ensuring proper tensor shape/channel order. I also add a strict check that we produced predictions for all required ids/classes when `learn` is loaded; if not, we fail over to empty strings to keep the CSV valid (but avoid silent partial coverage).'
- What this solution (achieved 0.0) has done: 'Your 0.0 is most consistent with the inference pipeline not matching what the exported fastai Learner was trained with: you’re currently feeding raw `np.uint8` images without ensuring tensor conversion, channel-first layout, and normalization compatible with the model, which often yields near-empty/garbage masks and effectively a zero score. I make the smallest change that preserves your core logic (same learner, same pack building, same 0.4 threshold, same morphology, same RLE) by building the test dataloader using the learner’s own `after_item/after_batch` pipeline (so transforms like `IntToFloatTensor/Normalize/ChannelsFirst` are applied as in training). I also stop recomputing `Metadata.extract()` from the file path during prediction (which is error-prone) and instead use the already-known `center_id` from the pack-building step, guaranteeing exact ID alignment with `sample_submission.csv`. Finally, I keep the empty-mask fallback unchanged so you still always produce a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the submission being rejected by the scorer due to an invalid RLE (common causes: including an explicit “0 0” run for empty masks, or producing masks with runs that aren’t strictly valid/positive), even though the CSV format itself looks fine. I make one minimal, directly score-impacting change: ensure `mask2rle()` returns an empty string for truly empty masks (and also enforces binary input), which guarantees the encoded runs are valid and avoids “invalid submission -> 0.0” outcomes. I also add a tiny guard so morphology output is converted back to `uint8` 0/1 before encoding (some scipy/skimage ops yield boolean), without changing your model/inference logic, threshold, resizing, or pack-building. The rest of your pipeline (model loading, fastai `test_dl`, threshold=0.4, opening, and sample_submission alignment) stays the same.'

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
from pathlib import Path

import albumentations as A
import cv2 as cv
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from fastai.vision.all import *

try:
    import segmentation_models_pytorch as smp  # noqa: F401
except ModuleNotFoundError:
    smp = None


def on_kaggle() -> bool:
    return Path("/kaggle").exists()


def uwm_get_size(s: Path):
    """
    Returns (h, w) from filename.
    Expected filename pattern: 'slice_0001_266_266_1.63_1.63.png' (h,w after slice index)
    """
    parts = s.stem.split("_")
    if len(parts) >= 4 and parts[0] == "slice":
        h, w = int(parts[2]), int(parts[3])
    else:
        h, w = int(parts[-4]), int(parts[-3])
    return (h, w)


def uwm_get_case_day(s: Path):
    return re.search(r"case\d+_day\d+", str(s)).group()


def uwm_get_slice(s: Path, as_number=False):
    m = re.search(r"slice_\d+", s.stem)
    if m:
        slice_tok = m.group()  # slice_0001
        return int(slice_tok.split("_")[-1]) if as_number else slice_tok
    tok = s.stem.split("_")[0]
    return int(tok) if as_number else f"slice_{int(tok):04d}"


def get_y(pack):
    raise RuntimeError(
        "Training target loading is not supported in this inference-only notebook."
    )




## === cell 2
@dataclass
class Metadata:
    sample_id: str
    full_path: str
    h: int
    w: int

    @classmethod
    def extract(cls, path: Path) -> "Metadata":
        case_and_day = path.parents[1].stem  # caseXXX_dayYY
        parts = path.stem.split("_")

        if len(parts) >= 4 and parts[0] == "slice":
            slice_no = int(parts[1])
            h = int(parts[2])
            w = int(parts[3])
        else:
            slice_no = int(parts[0])
            if len(parts) >= 5:
                h = int(parts[1])
                w = int(parts[2])
            else:
                h = int(parts[-4])
                w = int(parts[-3])

        sample_id = f"{case_and_day}_slice_{slice_no:04d}"
        return Metadata(sample_id, str(path), h, w)




## === cell 3
DATA_DIR = Path("/kaggle/input/uw-madison-gi-tract-image-segmentation/")

DEBUG = False
try:
    n_lines = sum(
        1 for _ in open(DATA_DIR / "sample_submission.csv", "r", encoding="utf-8")
    )
    DEBUG = n_lines <= 1
except FileNotFoundError:
    DEBUG = False

SUB_DF = pd.read_csv(DATA_DIR / ("train.csv" if DEBUG else "sample_submission.csv"))
TEST_IDS = SUB_DF["id"].drop_duplicates().tolist()

TEST_FILES = get_image_files(DATA_DIR / ("train" if DEBUG else "test"))
METADATA = {m.sample_id: m for m in L(TEST_FILES).map(Metadata.extract)}




## === cell 4
class CreateSample(Transform):
    def encodes(self, pack):
        h, w = uwm_get_size(pack[0])
        merged = np.ndarray((w, h, len(pack)), dtype=np.uint8)
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


class CreateTarget(Transform):
    def __init__(self, codes=(1, 2, 3)):
        super().__init__()
        self.codes = codes

    def encodes(self, pack):
        fn = get_y(pack)
        mask = cv.imread(str(fn), cv.IMREAD_UNCHANGED)
        channels = [(mask == c).astype(np.uint8) for c in self.codes]
        mask_rgb = np.dstack(channels)
        return mask_rgb

    def decodes(self, mask):
        return mask * 255


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
                A.OneOf([A.HorizontalFlip(p=1), A.VerticalFlip(p=0.3)], p=0.5),
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
                A.CoarseDropout(
                    min_holes=1,
                    max_holes=1,
                    min_height=16,
                    max_height=48,
                    min_width=16,
                    max_width=48,
                    p=0.3,
                ),
            ]
        )
    )


def valid_aug():
    return AugValid(A.Compose([A.Resize(320, 320), A.CenterCrop(288, 288)]))




## === cell 5
model_name = "5ch_cos_e100_bce_dice"




## === cell 6
def find_model_paths(model_name: str) -> dict[str, list[Path]]:
    """
    Robustly discover the intended export for this exact model_name.
    """
    root = Path("/kaggle/input")
    out = {"pkl": [], "pth": [], "pt": []}
    if not root.exists():
        return out

    preferred_dir = Path("/kaggle/input/uwm-models")
    if preferred_dir.exists():
        for ext in ("pkl", "pth", "pt"):
            p = preferred_dir / f"{model_name}.{ext}"
            if p.exists():
                out[ext].append(p)

    for ext in ("pkl", "pth", "pt"):
        out[ext].extend(sorted(root.rglob(f"{model_name}.{ext}")))

    for k in out:
        seen = set()
        uniq = []
        for p in out[k]:
            sp = str(p)
            if sp not in seen:
                seen.add(sp)
                uniq.append(p)
        out[k] = uniq
    return out


def load_model_for_inference(model_name: str):
    paths = find_model_paths(model_name)

    if paths["pkl"]:
        pkl_path = paths["pkl"][0]
        print(f"Loading learner from: {pkl_path}")
        return load_learner(pkl_path)

    if paths["pth"] or paths["pt"]:
        ckpt_path = (paths["pth"] + paths["pt"])[0]
        raise FileNotFoundError(
            f"Found checkpoint {ckpt_path} but no matching {model_name}.pkl export to reconstruct the model."
        )

    raise FileNotFoundError(
        f"Model file not found for name '{model_name}' under /kaggle/input."
    )


learn = None
try:
    learn = load_model_for_inference(model_name)
except Exception as e:
    print(f"WARNING: {e}")
    print("Will create an empty-mask submission (this will likely score near 0).")
    learn = None



## === cell 7
if DEBUG:
    predicted_ids = [
        id_
        for id_ in TEST_IDS
        if id_.startswith("case123_day20") or id_.startswith("case77_day20")
    ]
else:
    predicted_ids = TEST_IDS

predicted_ids = list(pd.unique(pd.Series(predicted_ids)))
len(predicted_ids)



## === cell 8
from collections import defaultdict
from more_itertools import chunked


def group_case_day_from_ids(ids: list[str]) -> dict[str, list[str]]:
    groups = defaultdict(list)
    for id_ in ids:
        m = re.search(r"(case\d+_day\d+)_slice_(\d+)", id_)
        if m is None:
            continue
        groups[m.group(1)].append(id_)
    for k in list(groups.keys()):
        groups[k] = sorted(groups[k], key=lambda s: int(s.split("_slice_")[-1]))
    return dict(groups)


def build_packs_from_ids(
    ids: list[str], n_slices_to_merge: int = 5
) -> list[tuple[str, list[Path]]]:
    assert n_slices_to_merge % 2 == 1
    half = n_slices_to_merge // 2

    groups = group_case_day_from_ids(ids)
    packs: list[tuple[str, list[Path]]] = []
    for _, id_list in groups.items():
        for idx, center_id in enumerate(id_list):
            if center_id not in METADATA:
                continue

            pack_paths: list[Path] = []
            for rel in range(-half, half + 1):
                j = idx + rel
                if j < 0:
                    use_id = id_list[0]
                elif j >= len(id_list):
                    use_id = id_list[-1]
                else:
                    use_id = id_list[j]

                if use_id in METADATA:
                    pack_paths.append(Path(METADATA[use_id].full_path))
                else:
                    pack_paths.append(Path(METADATA[center_id].full_path))

            packs.append((center_id, pack_paths))
    return packs


packs = build_packs_from_ids(predicted_ids, n_slices_to_merge=5)
len(packs)



## === cell 9
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device



## === cell 10
try:
    import cupy as cp
except Exception:
    cp = None


def mask2rle(mask: np.ndarray) -> str:
    """
    Minimal score-critical fix:
    - Kaggle expects empty masks encoded as an empty string, NOT '0 0'.
    - Ensure strict binary (0/1) before encoding so runs are valid.
    - Keep required Fortran/column-major flattening ("top to bottom, then left to right").
    """
    mask = (mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        return ""

    if cp is not None:
        m = cp.asarray(mask, dtype=cp.uint8)
        pixels = m.reshape((-1,), order="F")
        pad = cp.array([0], dtype=cp.uint8)
        pixels = cp.concatenate([pad, pixels, pad])
        runs = cp.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        runs = cp.asnumpy(runs)
    else:
        pixels = mask.reshape(-1, order="F")
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]

    return " ".join(str(int(x)) for x in runs)


def rle2mask(mask_rle, shape):
    s = np.asarray(mask_rle.split(), dtype=int)
    if len(s) == 0:
        return np.zeros(shape, dtype=np.uint8)
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
from skimage.morphology import disk
from scipy.ndimage import binary_opening

preds = []

if learn is None:
    for test_id in predicted_ids:
        for name in ("large_bowel", "small_bowel", "stomach"):
            preds.append({"id": test_id, "class": name, "predicted": ""})
else:
    learn.dls.to(device)
    learn.model.to(device)
    learn.eval()

    tfm_sample = CreateSample()

    with learn.no_bar():
        batch_size = 32  # keep safe for GPU memory with 5ch 288x288
        chunks = list(chunked(packs, n=batch_size))

        for subset in progress_bar(chunks):
            center_ids = [cid for cid, _ in subset]
            subset_packs = [pk for _, pk in subset]

            test_dl = learn.dls.test_dl(
                subset_packs,
                bs=batch_size,
                device=device,
                after_item=[tfm_sample] + list(getattr(learn.dls, "after_item", [])),
                after_batch=list(getattr(learn.dls, "after_batch", [])),
            )

            logits, *_ = learn.get_preds(dl=test_dl, act=noop)

            labels = (F.sigmoid(logits) >= 0.4).cpu().numpy().astype(np.uint8)

            for center_id, mask in zip(center_ids, labels):
                if center_id not in METADATA:
                    continue

                m = METADATA[center_id]
                h, w = cv.imread(m.full_path, cv.IMREAD_UNCHANGED).shape[:2]

                for i, name in enumerate(("large_bowel", "small_bowel", "stomach")):
                    cls_mask = pad_mask(mask[i], 320)
                    cls_mask = cv.resize(cls_mask, (w, h), cv.INTER_NEAREST)

                    opened = binary_opening(cls_mask.astype(bool), structure=disk(5))
                    cls_mask_bin = opened.astype(np.uint8)

                    rle = mask2rle(cls_mask_bin)
                    preds.append({"id": center_id, "class": name, "predicted": rle})

            del logits, labels, test_dl
            gc.collect()



## === cell 12
df_preds = pd.DataFrame(preds)
df_preds.head()



## === cell 13
if DEBUG and len(df_preds):
    from skimage.color import label2rgb

    id_ = df_preds["id"].iloc[0]
    factory = CreateSample()
    metadata = METADATA[id_]
    pairs, overlaid = [], []

    for i in range(3):
        rle = df_preds[df_preds.id == id_].iloc[i].predicted
        mask = rle2mask(rle, (metadata.h, metadata.w))
        sample = factory([Path(metadata.full_path)]).squeeze()
        pairs.extend([sample, mask])
        overlaid.append(label2rgb(mask, image=sample))

    show_images(pairs)
    show_images(overlaid)



## === cell 14
df_submit = pd.read_csv(DATA_DIR / "sample_submission.csv")

df_preds = df_preds.drop_duplicates(subset=["id", "class"], keep="last")

df_submit = df_submit.drop(columns="predicted").merge(
    df_preds, on=["id", "class"], how="left", validate="one_to_one"
)
df_submit["predicted"] = df_submit["predicted"].fillna("")

df_submit = (
    df_submit.set_index(["id", "class"])
    .reindex(
        pd.read_csv(DATA_DIR / "sample_submission.csv").set_index(["id", "class"]).index
    )
    .reset_index()
)
df_submit["predicted"] = df_submit["predicted"].fillna("")

df_submit.to_csv("submission.csv", index=False)

print(df_submit.shape)
print(df_submit.head(3))
print("Wrote submission.csv")
