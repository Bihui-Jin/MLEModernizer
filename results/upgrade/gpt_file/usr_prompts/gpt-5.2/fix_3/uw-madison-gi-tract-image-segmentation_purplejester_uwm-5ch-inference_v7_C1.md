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
        case_and_day = path.parents[1].stem
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

TEST_IDS = (
    pd.read_csv(DATA_DIR / "sample_submission.csv")["id"].drop_duplicates().tolist()
)
TEST_FILES = get_image_files(DATA_DIR / ("train" if DEBUG else "test"))

METADATA = {m.sample_id: m for m in TEST_FILES.map(Metadata.extract)}

missing = [i for i in TEST_IDS[:100] if i not in METADATA]
if missing:
    logging.warning(
        f"Some test ids missing from metadata (showing up to 5): {missing[:5]}"
    )




## === cell 3
def get_case_day(s) -> str:
    return re.search(r"case\d+_day\d+", str(s)).group()


def get_slice(s, as_number: bool = False):
    m = re.search(r"slice_(\d{4})", str(s))
    if m:
        slice_no = int(m.group(1))
    else:
        parts = Path(s).stem.split("_")
        slice_no = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else 0
    return slice_no if as_number else f"slice_{slice_no:04d}"


def get_sample_id(s) -> str:
    return f"{get_case_day(s)}_{get_slice(s)}"


def get_size_from_path(p: Path):
    img = cv.imread(str(p), cv.IMREAD_UNCHANGED)
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
        q = 0.01
        for i, fn in enumerate(pack):
            img = cv.imread(str(fn), cv.IMREAD_UNCHANGED)
            lo, hi = np.percentile(img, [q * 100, (1 - q) * 100])
            img = np.clip(img, lo, hi)
            v_min, v_max = np.min(img), np.max(img)
            if float(v_max - v_min) < 1e-8:
                img = np.zeros_like(img, dtype=np.uint8)
            else:
                img = (img - v_min) / float(v_max - v_min)
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
paths = [Path(METADATA[id_].full_path) for id_ in predicted_ids if id_ in METADATA]
packs = packed(group_case_day_from_files(paths), n_slices_to_merge=5, step_size=1)
len(packs), len(paths)




## === cell 8
import cupy as cp


def mask2rle(mask):
    """
    mask: numpy array, 1 - mask, 0 - background

    Change: encode in column-major (Fortran) order to match the competition's
    "top-to-bottom then left-to-right" indexing used by this dataset's RLE.
    This prevents effectively-wrong masks that can score near 0 even if predictions are non-empty.
    """
    if mask is None:
        return ""
    mask = cp.asarray(mask.astype(np.uint8))
    pixels = mask.reshape(-1, order="F")  # critical: Fortran order
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

    with learn.no_bar():
        batch_size = 64
        chunks = list(chunked(packs, n=batch_size))

        for subset in progress_bar(chunks):
            test_dl = learn.dls.test_dl(subset, batch_size=batch_size, device=device)
            logits, *_ = learn.get_preds(dl=test_dl, act=noop)

            labels = (
                (torch.sigmoid(logits) >= 0.4).detach().cpu().numpy().astype(np.uint8)
            )

            for pack, mask in zip(subset, labels):
                test_id = get_sample_id(pack[len(pack) // 2])
                if test_id not in METADATA:
                    continue
                m = METADATA[test_id]
                h, w = m.h, m.w

                for i, name in enumerate(("large_bowel", "small_bowel", "stomach")):
                    cls_mask = pad_mask(mask[i], 320)  # -> 320x320
                    cls_mask = cv.resize(cls_mask, (w, h), cv.INTER_NEAREST)
                    cls_mask = binary_opening(
                        cls_mask.astype(bool), structure=disk(5)
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
