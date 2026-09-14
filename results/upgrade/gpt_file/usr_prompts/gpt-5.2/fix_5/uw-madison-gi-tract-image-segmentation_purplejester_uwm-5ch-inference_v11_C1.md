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

0.8582293164570355

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the immediate runtime blocker by making model loading robust: if the external `/kaggle/input/uwm-models/*.pkl` isn’t available, the script fall back to a valid baseline submission (all-empty masks) so a `.csv` is always produced end-to-end. I also correct a shape bug in `CreateSample` where the merged array used `(w,h,channels)` instead of `(h,w,channels)`, which would break inference if a learner is available. Finally, I make the inference cell conditional on a successfully loaded learner, preventing the downstream `NameError` and ensuring the submission format matches `sample_submission.csv` exactly.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is coming from the fallback path producing an all-empty submission because the model file is missing at `/kaggle/input/uwm-models/{model_name}.pkl`. To move the score toward the 0.858 target with minimal change, I make model discovery robust by searching `/kaggle/input/**` for a matching `.pkl` (and optionally accept an environment override), so inference actually runs when the notebook has the model attached under a different dataset name/path. I also ensure the test set file listing matches `sample_submission.csv` IDs (filtering to `scans/` PNGs and building packs from those), preventing silent ID/path mismatches that can yield empty/incorrect predictions even with a loaded model. Core model/inference logic, thresholds, and RLE encoding remain unchanged.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly because the fallback path is being used (no `.pkl` found), producing an all-empty submission. The smallest change to move toward the 0.858 target is to make model discovery more robust (also searching for any `.pkl` that looks like your model name, and allowing a directory override), so inference actually runs when a model dataset is attached under an unexpected path/name. In addition, I add a strict ID/file alignment check so we only predict for IDs present in `sample_submission.csv` and report how many are missing, preventing silent misalignment that can yield near-zero scores even with a loaded model. Core architecture/inference (threshold=0.5, padding/resize, RLE) is unchanged.'

# 9. Code solution

## === cell 0
import os
import re
import sys
import gc
import logging
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
    load_learner,
    progress_bar,
    noop,
)

logging.captureWarnings(True)


def on_kaggle() -> bool:
    return os.path.exists("/kaggle")


def equalize(img: np.ndarray) -> np.ndarray:
    if img is None:
        return img
    if img.dtype != np.uint8:
        x = img.astype(np.float32)
        x = x - x.min()
        mx = x.max()
        if mx > 0:
            x = x / mx
        img8 = (x * 255).astype(np.uint8)
    else:
        img8 = img
    return cv.equalizeHist(img8)




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
        if len(parts) >= 4 and parts[0] == "slice":
            slice_no = parts[1]
            h, w = parts[2], parts[3]
        else:
            img = cv.imread(str(path), cv.IMREAD_UNCHANGED)
            h, w = img.shape[:2]
            slice_no = (
                re.findall(r"\d+", path.stem)[0]
                if re.findall(r"\d+", path.stem)
                else "0"
            )

        sample_id = f"{case_and_day}_slice_{int(slice_no):04d}"
        return Metadata(sample_id, str(path), int(h), int(w))




## === cell 2
DATA_DIR = Path("/kaggle/input/uw-madison-gi-tract-image-segmentation/")

DEBUG = False

TEST_IDS = (
    pd.read_csv(DATA_DIR / ("train.csv" if DEBUG else "sample_submission.csv"))["id"]
    .drop_duplicates()
    .tolist()
)

TEST_FILES = [
    p
    for p in get_image_files(DATA_DIR / ("train" if DEBUG else "test"))
    if "/scans/" in str(p)
]
METADATA = {m.sample_id: m for m in pd.Series(TEST_FILES).map(Metadata.extract)}

len(TEST_IDS), len(TEST_FILES), len(METADATA)




## === cell 3
def get_size_from_scan_path(s: Path):
    img = cv.imread(str(s), cv.IMREAD_UNCHANGED)
    h, w = img.shape[:2]
    return (h, w)


class CreateSample(Transform):
    def encodes(self, pack):
        h, w = get_size_from_scan_path(pack[0])
        merged = np.ndarray((h, w, len(pack)), dtype=np.uint8)

        q = 0.01
        for i, fn in enumerate(pack):
            img = cv.imread(str(fn), cv.IMREAD_UNCHANGED)
            lo, hi = np.percentile(img, [q * 100, (1 - q) * 100])
            img = np.clip(img, lo, hi)
            v_min, v_max = np.min(img), np.max(img)
            if float(v_max - v_min) > 0:
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


class FloatMask(Transform):
    order = 99

    def encodes(self, x: TensorMask):
        return TensorImage(x.float())

    def decodes(self, x: TensorMask):
        return TensorMask(x.long())


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
                A.Cutout(p=0.3),
            ]
        )
    )


def valid_aug():
    return AugValid(A.Compose([A.Resize(320, 320), A.CenterCrop(288, 288)]))




## === cell 4
model_name = "dataset_norm_e10_e15"




## === cell 5
def find_model_path(model_name: str) -> Path | None:
    env_p = os.environ.get("UWM_MODEL_PATH", "").strip()
    if env_p:
        p = Path(env_p)
        if p.exists() and p.suffix == ".pkl":
            return p

    env_dir = os.environ.get("UWM_MODEL_DIR", "").strip()
    if env_dir:
        d = Path(env_dir)
        if d.exists() and d.is_dir():
            candidates = sorted(d.rglob("*.pkl"), key=lambda p: str(p))
            for p in candidates:
                if p.name == f"{model_name}.pkl":
                    return p
            for p in candidates:
                if model_name in p.name:
                    return p
            if candidates:
                return candidates[0]

    p0 = Path(f"/kaggle/input/uwm-models/{model_name}.pkl")
    if p0.exists():
        return p0

    root = Path("/kaggle/input")
    if root.exists():
        exact_hits = list(root.rglob(f"{model_name}.pkl"))
        if exact_hits:
            exact_hits = sorted(exact_hits, key=lambda p: (len(p.parts), str(p)))
            return exact_hits[0]

        partial_hits = [p for p in root.rglob("*.pkl") if model_name in p.name]
        if partial_hits:
            partial_hits = sorted(partial_hits, key=lambda p: (len(p.parts), str(p)))
            return partial_hits[0]
    return None


MODEL_PATH = find_model_path(model_name)
learn = None
if MODEL_PATH is not None and MODEL_PATH.exists():
    logging.warning(f"Loading model from: {MODEL_PATH}")
    learn = load_learner(MODEL_PATH)
else:
    logging.warning(
        f"Model '{model_name}.pkl' not found under /kaggle/input. Falling back to empty-mask submission."
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

len(predicted_ids)




## === cell 7
def get_case_day(s: Path) -> str:
    return re.search(r"case\d+_day\d+", str(s)).group()


def get_slice(s: Path, as_number=False):
    m = re.search(r"slice_\d\d\d\d", str(s))
    if m is None:
        nums = re.findall(r"\d+", s.stem)
        val = int(nums[0]) if nums else 0
        return val if as_number else f"slice_{val:04d}"
    slice_no = m.group()
    return int(slice_no.split("_")[-1]) if as_number else slice_no


def get_sample_id(s: Path) -> str:
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




## === cell 8
id_to_path = {}
for p in TEST_FILES:
    sid = get_sample_id(p)
    id_to_path[sid] = p

missing = [sid for sid in predicted_ids if sid not in id_to_path]
if len(missing) > 0:
    logging.warning(
        f"Missing {len(missing)} ids in scan files (showing up to 5): {missing[:5]}"
    )

paths = [id_to_path[sid] for sid in predicted_ids if sid in id_to_path]

len(paths), (paths[0] if len(paths) else None)



## === cell 9
packs = packed(group_case_day_from_files(paths), n_slices_to_merge=5, step_size=1)
len(packs), len(packs[0]), packs[0][len(packs[0]) // 2]



## === cell 10
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device




## === cell 11
def mask2rle(mask: np.ndarray) -> str:
    """
    mask: 2D numpy array, 1 - mask, 0 - background
    RLE encoding with pixels numbered top-to-bottom then left-to-right.
    Achieved by flattening mask.T in C order.
    """
    if mask is None:
        return ""
    mask = mask.astype(np.uint8)
    pixels = mask.T.flatten()
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
    return img.reshape((shape[1], shape[0])).T


def pad_mask(mask, image_size):
    padded = np.zeros((image_size, image_size), dtype=mask.dtype)
    dh = image_size - mask.shape[0]
    dw = image_size - mask.shape[1]
    top = dh // 2
    left = dw // 2
    padded[top : top + mask.shape[0], left : left + mask.shape[1]] = mask
    return padded




## === cell 12
preds = []

if learn is not None:
    learn.dls.to(device)
    learn.eval()

    with learn.no_bar():
        batch_size = 64
        chunks = list(chunked(packs, n=batch_size))

        for subset in progress_bar(chunks):
            test_dl = learn.dls.test_dl(subset, batch_size=batch_size, device=device)
            logits, *_ = learn.get_preds(dl=test_dl, act=noop)

            labels = (
                (torch.sigmoid(logits) >= 0.5).detach().cpu().numpy().astype(np.uint8)
            )

            for pack, mask in zip(subset, labels):
                test_id = get_sample_id(pack[len(pack) // 2])
                m = METADATA[test_id]
                h, w = cv.imread(m.full_path, cv.IMREAD_UNCHANGED).shape[:2]

                for i, name in enumerate(("large_bowel", "small_bowel", "stomach")):
                    cls_mask = pad_mask(mask[i], 320)
                    cls_mask = cv.resize(cls_mask, (w, h), cv.INTER_NEAREST)
                    rle = mask2rle(cls_mask)
                    preds.append({"id": test_id, "class": name, "predicted": rle})

            del logits, labels
            gc.collect()
else:
    df_submit = pd.read_csv(DATA_DIR / "sample_submission.csv")
    preds = df_submit.assign(predicted="").to_dict("records")

len(preds), preds[0] if len(preds) else None



## === cell 13
df_preds = pd.DataFrame(preds)

df_submit = pd.read_csv(DATA_DIR / "sample_submission.csv")
df_submit = df_submit.drop(columns="predicted").merge(
    df_preds, on=["id", "class"], how="left"
)
df_submit["predicted"] = df_submit["predicted"].fillna("")

out_path = Path("submission.csv")
df_submit.to_csv(out_path, index=False)

df_submit.head(), out_path, out_path.exists(), out_path.stat().st_size
