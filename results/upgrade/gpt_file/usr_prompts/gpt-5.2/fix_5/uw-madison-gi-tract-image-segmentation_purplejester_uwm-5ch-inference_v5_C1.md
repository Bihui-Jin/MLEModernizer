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

0.38922

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on non-existent Kaggle datasets/modules (`/kaggle/input/uwm-libs`, `segmentation_models_pytorch`, `fast_ai_utils`, and `uwm.*`) that currently prevent the notebook from running at all. To preserve the original inference semantics as closely as possible, I keep the same multi-slice packing + thresholding + per-class postprocessing + RLE pipeline, but replace the missing model/learner with a safe, deterministic fallback that still produces correctly formatted predictions. I also fix missing imports (`pandas`, `fastai` symbols usage), remove IPython `!` shell usage, and ensure `mask2rle` returns a proper string (and isn’t accidentally joining CuPy scalars). Finally, I guarantee that `submission.csv` is written with the exact required columns/row count by merging into `sample_submission.csv` and filling missing predictions with empty strings.'
- What this solution (achieved 0.0) has done: 'I fix the crash in `CreateSample.encodes` by allocating the merged volume using the actual image shape (H,W) instead of the filename-parsed (W,H), which currently causes a broadcast error on slices where width/height are swapped. I also make `get_size` return `(h,w)` consistently (OpenCV order) so downstream logic stays coherent and avoids unnecessary transposes. These changes are execution-critical and keep the rest of the pipeline (multi-slice packing, resizing/cropping, thresholding, per-class morphology, and RLE submission formatting) identical. The script then run end-to-end and write a valid `submission.csv` with the required 20400 rows/3 columns.'
- What this solution (achieved 0.02426) has done: 'Your current 0.0 score is consistent with predicting (almost) all-empty masks due to the dummy model outputting very negative logits, so the smallest score-improving change is to replace that dummy with a lightweight, deterministic intensity-based segmentation heuristic that produces non-empty masks. I keep the exact same multi-slice packing, resizing/cropping, morphology, and RLE submission pipeline; only the “model” forward pass be swapped to generate plausible logits from the input image intensities. I also add a small, fixed per-class bias so different organs get different mask sizes without changing any downstream semantics. This should raise the score from 0.0 toward your target while keeping runtime under the limit and preserving the overall structure.'
- What this solution (achieved 0.38922) has done: 'Your current score is far below the target (0.02426 vs 0.8416, higher-is-better), so we need a real boost while keeping your pipeline intact. The smallest high-impact fix is to make the heuristic “model” output more organ-like masks by using deterministic, per-class intensity thresholds (and mild smoothing) instead of a near-random z-score with small biases. This preserves your core inference structure (multi-slice packing, same resizing/cropping, same sigmoid+0.5 thresholding, same morphology, same RLE formatting) but yields substantially more plausible non-empty segmentations. I also make `mask2rle` robust to missing CuPy (fallback to NumPy) without changing outputs when CuPy is available.'

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
    def __init__(self, n_classes=3, out_h=288, out_w=288):
        super().__init__()
        self.n_classes = n_classes
        self.out_h = out_h
        self.out_w = out_w
        self.register_buffer(
            "_thresholds",
            torch.tensor([0.58, 0.52, 0.48], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "_scales",
            torch.tensor([14.0, 12.0, 10.0], dtype=torch.float32).view(1, 3, 1, 1),
        )

    def forward(self, x):
        xmean = x.mean(dim=1, keepdim=True)  # [B,1,H,W]

        xs = F.avg_pool2d(xmean, kernel_size=5, stride=1, padding=2)

        x3 = xs.repeat(1, 3, 1, 1)
        thr = self._thresholds.to(device=x.device, dtype=x.dtype)
        scl = self._scales.to(device=x.device, dtype=x.dtype)

        logits = (x3 - thr) * scl  # >0 when intensity above class threshold

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
paths = [Path(METADATA[id_].full_path) for id_ in predicted_ids if id_ in METADATA]
missing = [id_ for id_ in predicted_ids if id_ not in METADATA]
len(paths), len(missing)



## === cell 8
packs = packed(group_case_day_from_files(paths), n_slices_to_merge=5, step_size=1)
len(packs), packs[0][len(packs[0]) // 2]



## === cell 9
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dummy_model = dummy_model.to(device).eval()
device



## === cell 10
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

factory = CreateSample()
val_aug = valid_aug()

preds = []
batch_size = 64
chunks = list(chunked(packs, n=batch_size))

with torch.no_grad():
    for subset in progress_bar(chunks):
        ims = []
        mids = []
        for pack in subset:
            mid = pack[len(pack) // 2]
            mids.append(mid)

            img = factory(pack)  # uint8 (H,W,C)
            img = val_aug.aug(image=img)["image"]  # HWC uint8

            t = torch.from_numpy(img).to(torch.float32) / 255.0  # HWC
            t = t.permute(2, 0, 1)  # CHW
            ims.append(t)

        x = torch.stack(ims, dim=0).to(device)  # [B,C,H,W]
        logits = dummy_model(x)
        labels = (torch.sigmoid(logits) >= 0.5).to(torch.uint8).cpu().numpy()

        for pack, mask in zip(subset, labels):
            test_id = get_sample_id(pack[len(pack) // 2])
            m = METADATA[test_id]
            h, w = cv.imread(m.full_path, cv.IMREAD_UNCHANGED).shape[:2]

            for i, name in enumerate(("large_bowel", "small_bowel", "stomach")):
                cls_mask = pad_mask(mask[i], 320)
                cls_mask = cv.resize(cls_mask, (w, h), cv.INTER_NEAREST)
                opened = binary_opening(
                    cls_mask.astype(bool), structure=disk(5)
                ).astype(np.uint8)
                rle = mask2rle(opened) if opened.any() else ""
                preds.append({"id": test_id, "class": name, "predicted": rle})

        del x, logits, labels
        gc.collect()

len(preds), preds[0]



## === cell 12
df_preds = pd.DataFrame(preds)
df_preds.head(), df_preds.shape



## === cell 13
df_submit = pd.read_csv(DATA_DIR / "sample_submission.csv")
df_submit = df_submit.drop(columns="predicted").merge(
    df_preds, on=["id", "class"], how="left"
)
df_submit["predicted"] = df_submit["predicted"].fillna("")
df_submit.to_csv("submission.csv", index=False)

print(df_submit.shape)
print(df_submit.columns.tolist())
print(pd.read_csv("submission.csv").head(5))
