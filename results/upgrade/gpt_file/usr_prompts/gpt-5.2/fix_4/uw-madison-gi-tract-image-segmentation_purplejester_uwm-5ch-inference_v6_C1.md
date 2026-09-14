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
def find_model_path(model_name: str) -> Path | None:
    preferred = Path(f"/kaggle/input/uwm-models/{model_name}.pkl")
    if preferred.exists():
        return preferred
    root = Path("/kaggle/input")
    if not root.exists():
        return None
    matches = sorted(root.rglob(f"{model_name}.pkl"))
    return matches[0] if matches else None


model_path = find_model_path(model_name)
learn = None
if model_path is not None and model_path.exists():
    try:
        print(f"Loading learner from: {model_path}")
        learn = load_learner(model_path)
    except Exception as e:
        print(f"WARNING: Failed to load learner from {model_path}: {e}")
        learn = None
else:
    print(
        f"WARNING: Model file not found for name '{model_name}'. Will create an empty-mask submission."
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

predicted_ids = list(pd.unique(pd.Series(predicted_ids)))
len(predicted_ids)



## === cell 8
from collections import defaultdict
from functools import partial

from more_itertools import windowed


def get_case_day(s):
    return re.search(r"case\d+_day\d+", str(s)).group()


def get_slice(s, as_number=False):
    m = re.search(r"slice_\d+", str(s))
    if m:
        slice_no = m.group()
        return int(slice_no.split("_")[-1]) if as_number else slice_no
    tok = Path(s).stem.split("_")[0]
    return int(tok) if as_number else f"slice_{int(tok):04d}"


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




## === cell 9
missing = [id_ for id_ in predicted_ids if id_ not in METADATA]
if len(missing):
    print(f"WARNING: {len(missing)} ids missing from METADATA; they will be skipped.")
paths = [Path(METADATA[id_].full_path) for id_ in predicted_ids if id_ in METADATA]

packs = packed(group_case_day_from_files(paths), n_slices_to_merge=5, step_size=1)
len(packs)



## === cell 10
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device



## === cell 11
try:
    import cupy as cp
except Exception:
    cp = None


def mask2rle(mask):
    """
    mask: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    """
    if cp is not None:
        m = cp.asarray(mask, dtype=cp.uint8)
        pixels = m.flatten()
        pad = cp.array([0], dtype=cp.uint8)
        pixels = cp.concatenate([pad, pixels, pad])
        runs = cp.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        runs = cp.asnumpy(runs)
    else:
        pixels = mask.astype(np.uint8).ravel()
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
    return img.reshape(shape)


def pad_mask(mask, image_size):
    padded = np.zeros((image_size, image_size), dtype=mask.dtype)
    dh = image_size - mask.shape[0]
    dw = image_size - mask.shape[1]
    top = dh // 2
    left = dw // 2
    padded[top : top + mask.shape[0], left : left + mask.shape[1]] = mask
    return padded




## === cell 12
from skimage.morphology import disk
from scipy.ndimage import binary_opening
from more_itertools import chunked

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

            labels = (F.sigmoid(logits) >= 0.4).cpu().numpy().astype(np.uint8)

            for pack, mask in zip(subset, labels):
                test_id = get_sample_id(pack[len(pack) // 2])
                m = METADATA[test_id]
                h, w = cv.imread(m.full_path, cv.IMREAD_UNCHANGED).shape[:2]

                for i, name in enumerate(("large_bowel", "small_bowel", "stomach")):
                    cls_mask = pad_mask(mask[i], 320)
                    cls_mask = cv.resize(cls_mask, (w, h), cv.INTER_NEAREST)
                    rle = mask2rle(binary_opening(cls_mask, structure=disk(5)))
                    preds.append({"id": test_id, "class": name, "predicted": rle})

            del logits, labels
            gc.collect()



## === cell 13
df_preds = pd.DataFrame(preds)
df_preds.head()



## === cell 14
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



## === cell 15
df_submit = pd.read_csv(DATA_DIR / "sample_submission.csv")
df_submit = df_submit.drop(columns="predicted").merge(
    df_preds, on=["id", "class"], how="left"
)
df_submit["predicted"] = df_submit["predicted"].fillna("")

df_submit.to_csv("submission.csv", index=False)

print(df_submit.shape)
print(df_submit.head(3))
print("Wrote submission.csv")
