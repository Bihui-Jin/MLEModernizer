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

0.841204308399207

# 6. Current score

0.02211

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the metadata/sample-id parsing so it matches the competition’s `id` values from `test.csv/sample_submission.csv`, which removes the `ValueError: 'slice'` issue and makes `METADATA` build correctly. Since your external pretrained `.pkl` is missing, I replace that hard dependency with a minimal, deterministic fallback that generates valid binary masks (empty) so the notebook runs end-to-end and always writes a correct `submission.csv`. I also fix the submission merge logic so it cannot crash when `preds` is empty (or missing columns), ensuring the file has exactly `id,class,predicted` and 20400 rows. These changes are execution-critical and keep evaluation semantics intact (RLE encoding, per-class rows), while giving you a guaranteed valid submission to obtain a baseline score.'
- What this solution (achieved 0.01225) has done: 'Your current 0.0 score is caused by the fallback path producing empty masks for every image/class, which yields an all-empty submission. Since the external `.pkl` model is missing, the smallest legitimate improvement is to replace the “always empty” fallback with a deterministic, image-driven baseline that creates non-empty binary masks using simple intensity thresholding and light morphology, then RLE-encodes per class. This keeps the same submission semantics (per `id,class` RLE; same packing/metadata; no training; no architecture/loss changes) while producing a meaningful (non-zero) score. I also keep the existing learned-model path unchanged, so if the `.pkl` is present the original behavior stays intact.'
- What this solution (achieved 0.02211) has done: 'Your current score is far below target, so the smallest safe way to move it upward (without changing the model/training logic) is to make the fallback predictions less noisy and more “organ-like” when the `.pkl` model is missing. I keep your exact pipeline and submission semantics, but improve the fallback by (1) using a more stable foreground extraction (adaptive + Otsu union), (2) explicitly removing border-connected air/background regions, and (3) selecting up to 3 plausible components and mapping them deterministically to the 3 classes (largest=large_bowel, next=small_bowel, next=stomach), instead of deriving all three from the same mask via different openings. This should increase Dice substantially versus the current one-mask-for-all-classes behavior, while remaining deterministic and fast. The learned-model path is untouched.'

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
from functools import partial

import numpy as np
import pandas as pd
import cv2 as cv
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_opening
from skimage.morphology import disk

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


def on_kaggle() -> bool:
    return Path("/kaggle").exists()


def equalize(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv.equalizeHist(img)
    out = img.copy()
    for c in range(out.shape[-1]):
        out[..., c] = cv.equalizeHist(out[..., c])
    return out




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

        if parts[0] == "slice":
            slice_no = int(parts[1])
            w = int(parts[2])
            h = int(parts[3])
        else:
            w = int(parts[0])
            h = int(parts[1])
            slice_no = int(parts[2])

        sample_id = f"{case_and_day}_slice_{slice_no:04d}"
        return Metadata(sample_id, str(path), h, w)




## === cell 3
DATA_DIR = Path("/kaggle/input/uw-madison-gi-tract-image-segmentation/")

DEBUG = False

IDS_SOURCE = "test.csv" if not DEBUG else "train.csv"
TEST_IDS = pd.read_csv(DATA_DIR / IDS_SOURCE)["id"].drop_duplicates().tolist()

TEST_FILES = get_image_files(DATA_DIR / ("train" if DEBUG else "test"))
METADATA = {m.sample_id: m for m in TEST_FILES.map(Metadata.extract)}

missing = sum(1 for x in TEST_IDS if x not in METADATA)
len(TEST_IDS), len(TEST_FILES), len(METADATA), missing



## === cell 4
model_name = "5ch_e10_step1_bce_dice"




## === cell 5
def get_size_from_path(s: Path):
    parts = s.stem.split("_")
    if parts[0] == "slice":
        w, h = int(parts[2]), int(parts[3])
    else:
        w, h = int(parts[0]), int(parts[1])
    return h, w  # (h, w)


def get_case_day(s: Path) -> str:
    return re.search(r"case\d+_day\d+", str(s)).group()


def get_slice_from_sample_id(sample_id: str, as_number=False):
    m = re.search(r"slice_\d{4}", sample_id)
    if m is None:
        raise ValueError(f"Bad sample_id: {sample_id}")
    slice_tag = m.group()
    return int(slice_tag.split("_")[-1]) if as_number else slice_tag


def get_slice_from_path(s: Path, as_number=False):
    parts = s.stem.split("_")
    if parts[0] == "slice":
        slice_no = int(parts[1])
    else:
        slice_no = int(parts[2])
    return slice_no if as_number else f"slice_{slice_no:04d}"


def get_sample_id_from_path(s: Path) -> str:
    return f"{get_case_day(s)}_{get_slice_from_path(s)}"


def group_case_day_from_files(image_files):
    groups = defaultdict(list)
    for fn in image_files:
        groups[get_case_day(fn)].append(fn)
    groups = {
        k: sorted(v, key=partial(get_slice_from_path, as_number=True))
        for k, v in groups.items()
    }
    return groups


def packed(groups, n_slices_to_merge=3, step_size=2):
    assert n_slices_to_merge % 2 != 0
    chunks = []
    for _, files in groups.items():
        pad = n_slices_to_merge // 2
        files = [None] * pad + files + [None] * pad
        for pack in windowed(files, n=n_slices_to_merge, step=step_size):
            pack = list(pack)
            last_not_none = [i for i, x in enumerate(pack) if x is not None][-1]
            for i in range(last_not_none + 1, len(pack)):
                pack[i] = pack[last_not_none]
            first_not_none = [i for i, x in enumerate(pack) if x is not None][0]
            for i in range(0, first_not_none):
                pack[i] = pack[first_not_none]
            chunks.append(pack)
    return chunks




## === cell 6
class CreateSample(Transform):
    def encodes(self, pack):
        h, w = get_size_from_path(pack[0])
        merged = np.ndarray((w, h, len(pack)), dtype=np.uint8)
        q = 0.01
        for i, fn in enumerate(pack):
            img = cv.imread(str(fn), cv.IMREAD_UNCHANGED)
            lo, hi = np.percentile(img, [q * 100, (1 - q) * 100])
            img = np.clip(img, lo, hi)
            v_min, v_max = np.min(img), np.max(img)
            if v_max == v_min:
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


class ChannelsFirst(ItemTransform):
    def encodes(self, x):
        return tuple(t.permute(0, 3, 1, 2) for t in x)

    def decodes(self, x):
        return tuple(t.permute(0, 2, 3, 1) for t in x)




## === cell 7
model_path = Path(f"/kaggle/input/uwm-models/{model_name}.pkl")
learn = None
if model_path.exists():
    learn = load_learner(model_path)
else:
    learn = None  # fallback used downstream

model_path.exists()



## === cell 8
if DEBUG:
    predicted_ids = [
        id_
        for id_ in TEST_IDS
        if id_.startswith("case123_day20") or id_.startswith("case77_day20")
    ]
else:
    predicted_ids = TEST_IDS

len(predicted_ids)



## === cell 9
paths = [Path(METADATA[id_].full_path) for id_ in predicted_ids if id_ in METADATA]
packs = packed(group_case_day_from_files(paths), n_slices_to_merge=5, step_size=1)
len(packs)



## === cell 10
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if learn is not None:
    learn.dls.to(device)
    learn.model.to(device)
    learn.eval()
device



## === cell 11
import cupy as cp


def mask2rle(mask: np.ndarray) -> str:
    """
    mask: numpy array, 1 - mask, 0 - background
    RLE is done in column-major order (top-to-bottom, then left-to-right).
    """
    if mask is None:
        return ""
    mask = mask.astype(np.uint8)
    if mask.sum() == 0:
        return ""

    mask_cp = cp.asarray(mask, dtype=cp.uint8)
    pixels = mask_cp.T.flatten()  # column-major
    pad = cp.asarray([0], dtype=cp.uint8)
    pixels = cp.concatenate([pad, pixels, pad])
    runs = cp.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    runs = cp.asnumpy(runs).astype(np.int64)
    return " ".join(str(x) for x in runs)


def rle2mask(mask_rle: str, shape):
    if mask_rle is None or mask_rle == "":
        return np.zeros(shape, dtype=np.uint8)
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
def fallback_predict_rles_for_id(test_id: str):
    m = METADATA[test_id]
    img = cv.imread(m.full_path, cv.IMREAD_UNCHANGED)
    if img is None:
        return {"large_bowel": "", "small_bowel": "", "stomach": ""}

    img = img.astype(np.float32)
    lo, hi = np.percentile(img, [1, 99])
    img = np.clip(img, lo, hi)
    vmin, vmax = float(img.min()), float(img.max())
    if vmax <= vmin:
        return {"large_bowel": "", "small_bowel": "", "stomach": ""}

    img8 = ((img - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
    blur = cv.GaussianBlur(img8, (5, 5), 0)

    otsu_thr, _ = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    mask_otsu = (blur > otsu_thr).astype(np.uint8)

    mask_adapt = cv.adaptiveThreshold(
        blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 51, -2
    )
    mask_adapt = (mask_adapt > 0).astype(np.uint8)

    mask = ((mask_otsu | mask_adapt) > 0).astype(np.uint8)

    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, k, iterations=1)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, k, iterations=2)

    h, w = mask.shape
    inv = (1 - mask).astype(np.uint8) * 255
    ff = inv.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv.floodFill(ff, flood_mask, (0, 0), 255)
    bg = ff == 255  # background reachable from border in inv-space
    mask = (mask & (~bg).astype(np.uint8)).astype(np.uint8)

    num, labels, stats, centroids = cv.connectedComponentsWithStats(
        mask, connectivity=8
    )
    comps = []
    center = np.array([w / 2.0, h / 2.0], dtype=np.float32)

    for i in range(1, num):
        area = int(stats[i, cv.CC_STAT_AREA])
        if area < 800:
            continue
        c = centroids[i].astype(np.float32)
        dist = float(np.linalg.norm(c - center))
        score = area - 0.5 * dist
        comps.append((score, area, i))

    comps.sort(reverse=True)  # best first
    chosen = [ci for _, _, ci in comps[:3]]

    if len(chosen) == 0:
        return {"large_bowel": "", "small_bowel": "", "stomach": ""}

    masks = [(labels == ci).astype(np.uint8) for ci in chosen]

    areas = [int(msk.sum()) for msk in masks]
    order = np.argsort(areas)[::-1]

    lb = masks[order[0]]
    sb = masks[order[1]] if len(order) > 1 else np.zeros_like(lb)
    st = masks[order[2]] if len(order) > 2 else np.zeros_like(lb)

    lb = binary_opening(lb.astype(bool), structure=disk(3)).astype(np.uint8)
    sb = binary_opening(sb.astype(bool), structure=disk(2)).astype(np.uint8)
    st = binary_opening(st.astype(bool), structure=disk(2)).astype(np.uint8)

    return {
        "large_bowel": mask2rle(lb),
        "small_bowel": mask2rle(sb),
        "stomach": mask2rle(st),
    }


preds = []

if learn is None:
    for test_id in predicted_ids:
        rles = fallback_predict_rles_for_id(test_id)
        for name in ("large_bowel", "small_bowel", "stomach"):
            preds.append({"id": test_id, "class": name, "predicted": rles[name]})
else:
    batch_size = 64
    chunks = list(chunked(packs, n=batch_size))

    with learn.no_bar():
        for subset in progress_bar(chunks):
            test_dl = learn.dls.test_dl(subset, batch_size=batch_size, device=device)

            logits, *_ = learn.get_preds(dl=test_dl, act=noop)

            probs = torch.sigmoid(logits).detach().cpu()
            labels = (probs >= 0.5).numpy().astype(np.uint8)

            for pack, mask in zip(subset, labels):
                test_id = get_sample_id_from_path(pack[len(pack) // 2])
                m = METADATA[test_id]
                h, w = cv.imread(m.full_path, cv.IMREAD_UNCHANGED).shape[:2]

                for i, name in enumerate(("large_bowel", "small_bowel", "stomach")):
                    cls_mask = pad_mask(mask[i], 320)
                    cls_mask = cv.resize(cls_mask, (w, h), cv.INTER_NEAREST)

                    cls_mask = binary_opening(
                        cls_mask.astype(bool), structure=disk(5)
                    ).astype(np.uint8)

                    rle = mask2rle(cls_mask)
                    preds.append({"id": test_id, "class": name, "predicted": rle})

            del logits, probs, labels
            gc.collect()

len(preds)



## === cell 13
df_preds = pd.DataFrame(preds)
for col in ["id", "class", "predicted"]:
    if col not in df_preds.columns:
        df_preds[col] = pd.Series(dtype="object")
df_preds = df_preds[["id", "class", "predicted"]]
df_preds.head(), df_preds.shape



## === cell 14
df_submit = pd.read_csv(DATA_DIR / "sample_submission.csv")

df_submit = df_submit.merge(
    df_preds, on=["id", "class"], how="left", suffixes=("", "_pred")
)

if "predicted_pred" in df_submit.columns:
    df_submit["predicted"] = df_submit["predicted_pred"]
    df_submit = df_submit.drop(columns=["predicted_pred"])

df_submit["predicted"] = df_submit["predicted"].fillna("")

out_path = Path("submission.csv")
df_submit.to_csv(out_path, index=False)

assert out_path.exists() and out_path.suffix == ".csv"
assert list(df_submit.columns) == ["id", "class", "predicted"]
assert len(df_submit) == 20400
df_submit.head(10)
