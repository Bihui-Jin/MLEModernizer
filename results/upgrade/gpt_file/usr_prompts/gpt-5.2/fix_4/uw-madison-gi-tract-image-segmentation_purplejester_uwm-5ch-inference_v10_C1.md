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

0.8426629557225797

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the metadata parsing so it correctly extracts image height/width from the actual scan filenames (not the folder name), which is what caused the `ValueError: invalid literal for int() with base 10: 'slice'`. Because the provided external model file is missing, I add a safe, minimal fallback path that produces a valid submission by outputting empty masks (this is score-poor but guarantees a valid `.csv` end-to-end in the current environment). I also make the inference code conditional on having a loaded learner, preventing the downstream `NameError: learn is not defined`. Finally, I ensure `df_preds` always has the required columns so the merge into `sample_submission.csv` cannot fail.'
- What this solution (achieved 0.0) has done: 'I fix the filename/metadata parsing so it doesn’t assume the scan filename begins with `H_W_...`, which is what’s triggering `ValueError: invalid literal for int() with base 10: 'slice'`. The safest minimal fix is to extract `(h,w)` from the *actual* scan filename using a regex that finds the first two integers anywhere in the stem, and to fall back to reading the image shape if parsing fails. I also make `get_size()` consistent with the same robust parsing so `CreateSample` can’t break on unexpected names. These changes are execution/stability fixes and preserve your existing inference/submission logic.'

# 9. Code solution

## === cell 0
import os
import sys
import gc
import re
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

from fastai.vision.all import (
    Transform,
    ItemTransform,
    TensorImage,
    TensorMask,
    get_image_files,
    progress_bar,
    load_learner,
    noop,
)
from more_itertools import windowed, chunked
from skimage.morphology import disk
from scipy.ndimage import binary_opening

logging.captureWarnings(True)


def on_kaggle() -> bool:
    return True


def equalize(x):
    return x




## === cell 1
@dataclass
class Metadata:
    sample_id: str
    full_path: str
    h: int
    w: int

    @staticmethod
    def _parse_hw_from_stem(stem: str):
        """
        Bugfix: some environments/files may not start the stem with 'H_W_...'.
        Robustly extract the first two integers found anywhere in the stem.
        """
        nums = re.findall(r"\d+", stem)
        if len(nums) >= 2:
            return int(nums[0]), int(nums[1])
        return None

    @classmethod
    def extract(cls, path: Path) -> "Metadata":
        hw = cls._parse_hw_from_stem(path.stem)
        if hw is None:
            img = cv.imread(str(path), cv.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Could not read scan image to infer size: {path}")
            h, w = img.shape[:2]
        else:
            h, w = hw
        return Metadata(sample_id="", full_path=str(path), h=h, w=w)




## === cell 2
DATA_DIR = Path("/kaggle/input/uw-madison-gi-tract-image-segmentation")

DEBUG = False

df_test = pd.read_csv(DATA_DIR / ("train.csv" if DEBUG else "test.csv"))
TEST_IDS = df_test["id"].drop_duplicates().tolist()

TEST_FILES = get_image_files(DATA_DIR / ("train" if DEBUG else "test"))


def get_case_day_from_path(p: Path) -> str:
    return re.search(r"case\d+_day\d+", str(p)).group()


groups = defaultdict(list)
for fn in TEST_FILES:
    groups[get_case_day_from_path(fn)].append(fn)

groups = {k: sorted(v, key=lambda x: x.name) for k, v in groups.items()}

METADATA = {}
for case_day, files in groups.items():
    for i, fn in enumerate(files, start=1):
        sample_id = f"{case_day}_slice_{i:04d}"
        m = Metadata.extract(fn)
        m.sample_id = sample_id
        METADATA[sample_id] = m

missing = sum(1 for _id in TEST_IDS if _id not in METADATA)
print(
    f"DEBUG={DEBUG} | unique ids={len(TEST_IDS)} | metadata ids={len(METADATA)} | missing={missing}"
)




## === cell 3
class CreateSample(Transform):
    def encodes(self, pack):
        h, w = get_size(pack[0])
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


class CreateTarget(Transform):
    def __init__(self, codes=(1, 2, 3)):
        super().__init__()
        self.codes = codes

    def encodes(self, pack):
        raise NotImplementedError("Targets are not used for test-time inference.")


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
        return x


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




## === cell 4
model_name = "dataset_norm_e10_e15"




## === cell 5
learn = None
model_path = Path(f"/kaggle/input/uwm-models/{model_name}.pkl")
if model_path.exists():
    learn = load_learner(model_path)
    print(f"Loaded model: {model_path}")
else:
    print(
        f"WARNING: Missing model file: {model_path}. "
        "Will generate a valid submission with empty predictions."
    )




## === cell 6
predicted_ids = TEST_IDS
print("n predicted ids:", len(predicted_ids))




## === cell 7
def get_size(s: Path):
    hw = Metadata._parse_hw_from_stem(s.stem)
    if hw is not None:
        return hw
    img = cv.imread(str(s), cv.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read scan image to infer size: {s}")
    h, w = img.shape[:2]
    return (h, w)


def get_case_day(s: Path) -> str:
    return re.search(r"case\d+_day\d+", str(s)).group()


def get_slice(s: str, as_number=False):
    slice_no = re.search(r"slice_\d\d\d\d", str(s)).group()
    return int(slice_no.split("_")[-1]) if as_number else slice_no


def get_sample_id_from_mid_slice_path(p: Path, slice_index_1based: int) -> str:
    return f"{get_case_day(p)}_slice_{slice_index_1based:04d}"


def group_case_day_from_files(image_files):
    groups2 = defaultdict(list)
    for fn in image_files:
        groups2[get_case_day(fn)].append(fn)
    groups2 = {k: sorted(v, key=lambda x: x.name) for k, v in groups2.items()}
    return groups2


def packed(groups2, n_slices_to_merge=3, step_size=2):
    assert n_slices_to_merge % 2 != 0
    chunks = []
    for case_day, files in groups2.items():
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
paths = []
for id_ in predicted_ids:
    m = METADATA.get(id_)
    if m is None:
        continue
    paths.append(Path(m.full_path))

print("paths found:", len(paths), "missing:", len(predicted_ids) - len(paths))




## === cell 9
packs = packed(group_case_day_from_files(paths), n_slices_to_merge=5, step_size=1)
print("packs:", len(packs))




## === cell 10
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if learn is not None:
    learn.dls.to(device)
    learn.model.to(device)
    learn.eval()
device




## === cell 11
def mask2rle(mask: np.ndarray) -> str:
    mask = (mask > 0).astype(np.uint8)
    pixels = mask.T.flatten()  # Fortran-like order
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle2mask(mask_rle, shape):
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

if learn is None:
    df_submit_template = pd.read_csv(DATA_DIR / "sample_submission.csv")
    preds = df_submit_template.assign(predicted="")
    df_preds = preds[["id", "class", "predicted"]].copy()
    print("Model missing -> generating empty predictions:", len(df_preds))
else:
    batch_size = 64
    chunks = list(chunked(packs, n=batch_size))

    with learn.no_bar():
        for subset in progress_bar(chunks):
            test_dl = learn.dls.test_dl(subset, batch_size=batch_size, device=device)

            logits, *_ = learn.get_preds(dl=test_dl, act=noop)

            labels = (torch.sigmoid(logits) >= 0.5).to("cpu").numpy().astype(np.uint8)

            for pack, mask in zip(subset, labels):
                mid_path = pack[len(pack) // 2]
                case_day = get_case_day(mid_path)
                files = groups[case_day]
                slice_index = files.index(mid_path) + 1
                test_id = f"{case_day}_slice_{slice_index:04d}"

                m = METADATA.get(test_id)
                if m is None:
                    continue

                h, w = cv.imread(m.full_path, cv.IMREAD_UNCHANGED).shape[:2]

                for i, name in enumerate(("large_bowel", "small_bowel", "stomach")):
                    cls_mask = pad_mask(mask[i], 320)
                    cls_mask = cv.resize(cls_mask, (w, h), cv.INTER_NEAREST)
                    cls_mask = binary_opening(
                        cls_mask.astype(bool), structure=disk(5)
                    ).astype(np.uint8)
                    rle = mask2rle(cls_mask)
                    preds.append({"id": test_id, "class": name, "predicted": rle})

            del logits, labels
            gc.collect()

    df_preds = pd.DataFrame(preds, columns=["id", "class", "predicted"])
    print("pred rows:", len(df_preds))




## === cell 13
if not isinstance(df_preds, pd.DataFrame):
    df_preds = pd.DataFrame(df_preds)
for col in ["id", "class", "predicted"]:
    if col not in df_preds.columns:
        df_preds[col] = pd.Series(dtype="object")

print(df_preds.head())
print(
    "unique ids predicted:",
    df_preds["id"].nunique(),
    "unique classes:",
    df_preds["class"].nunique(),
)




## === cell 14
df_submit = pd.read_csv(DATA_DIR / "sample_submission.csv")
df_submit = df_submit.drop(columns="predicted").merge(
    df_preds[["id", "class", "predicted"]], on=["id", "class"], how="left"
)
df_submit["predicted"] = df_submit["predicted"].fillna("")
df_submit.to_csv("submission.csv", index=False)

check = pd.read_csv("submission.csv")
assert list(check.columns) == ["id", "class", "predicted"]
assert len(check) == len(pd.read_csv(DATA_DIR / "sample_submission.csv"))
print(check.head(10))
print("Wrote submission.csv")
