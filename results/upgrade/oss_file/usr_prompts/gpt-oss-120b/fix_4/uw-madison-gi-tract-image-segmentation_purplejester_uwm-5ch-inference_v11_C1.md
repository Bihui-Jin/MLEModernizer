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

0.00457

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.00457) has done: 'I make the metadata extraction robust by reading image dimensions directly and generating sample IDs that match the expected pattern. I also guard against missing height/width values when resizing masks, and I ensure the submission merge works even if the predictions dataframe is initially empty.'

# 9. Code solution

## === cell 0
import logging
import sys
import re
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass

logging.captureWarnings(True)



## === cell 1
import gc
import numpy as np
import pandas as pd
import torch
import cv2 as cv
from fastai.vision.all import *
from more_itertools import windowed, chunked




## === cell 2
@dataclass
class Metadata:
    sample_id: str
    full_path: str
    h: int = None
    w: int = None

    @classmethod
    def extract(cls, path: Path) -> "Metadata":
        case_day = re.search(r"case\d+_day\d+", str(path)).group()
        slice_match = re.search(r"slice_(\d{4})", str(path))
        slice_part = f"slice_{slice_match.group(1)}" if slice_match else ""
        sample_id = f"{case_day}_{slice_part}" if slice_part else case_day

        img = cv.imread(str(path), cv.IMREAD_UNCHANGED)
        if img is not None:
            h, w = img.shape[:2]
        else:
            h = w = None
        return cls(sample_id, str(path), h, w)




## === cell 3
DATA_DIR = Path("/kaggle/input/uw-madison-gi-tract-image-segmentation")

TEST_IDS = pd.read_csv(DATA_DIR / "test.csv")["id"].drop_duplicates().tolist()

TEST_FILES = get_image_files(DATA_DIR / "test", recurse=True)

METADATA = {m.sample_id: m for m in L(TEST_FILES).map(Metadata.extract)}




## === cell 4
def get_case_day(s: Path) -> str:
    """Extract 'caseXXX_dayYY' from a path."""
    return re.search(r"case\d+_day\d+", str(s)).group()


def get_slice(s: Path, as_number=False):
    """Extract slice number from filename."""
    match = re.search(r"slice_(\d{4})", str(s))
    if not match:
        return None
    return int(match.group(1)) if as_number else f"slice_{match.group(1)}"


def get_sample_id(s: Path) -> str:
    return f"{get_case_day(s)}_{get_slice(s)}"


def group_case_day_from_files(image_files):
    groups = defaultdict(list)
    for fn in image_files:
        groups[get_case_day(fn)].append(fn)
    groups = {
        k: sorted(v, key=lambda p: int(re.search(r"slice_(\d{4})", str(p)).group(1)))
        for k, v in groups.items()
    }
    return groups


def packed(groups, n_slices_to_merge=3, step_size=2):
    """Create overlapping packs of file paths for the model."""
    assert n_slices_to_merge % 2 != 0, "n_slices_to_merge must be odd"
    chunks = []
    for case_day, files in groups.items():
        files = [None] + files + [None]  # pad for edge handling
        for pack in windowed(files, n=n_slices_to_merge, step=step_size):
            pack = list(pack)
            last_not_none = max(i for i, x in enumerate(pack) if x is not None)
            for i in range(last_not_none + 1, len(pack)):
                pack[i] = pack[last_not_none]
            first_not_none = min(i for i, x in enumerate(pack) if x is not None)
            for i in range(first_not_none):
                pack[i] = pack[first_not_none]
            chunks.append(pack)
    return chunks




## === cell 5
class CreateSample(Transform):
    def encodes(self, pack):
        h, w = get_size(pack[0])
        merged = np.ndarray((w, h, len(pack)), dtype=np.uint8)
        q = 0.01
        for i, fn in enumerate(pack):
            img = cv.imread(str(fn), cv.IMREAD_UNCHANGED)
            lo, hi = np.percentile(img, [q * 100, (1 - q) * 100])
            img = np.clip(img, lo, hi)
            v_min, v_max = img.min(), img.max()
            img = (img - v_min) / float(v_max - v_min + 1e-8)
            img = (img * 255).astype(np.uint8)
            merged[:, :, i] = img
        return merged




## === cell 6
class DummyLearner:
    def __init__(self):
        self.dls = self

    def test_dl(self, items, batch_size=1, device=None):
        return DataLoader(items, batch_size=batch_size, shuffle=False, drop_last=False)

    def to(self, *args, **kwargs):
        return self

    def get_preds(self, dl, act=None):
        batch = len(dl.dataset)
        logits = torch.zeros((batch, 3, 320, 320), dtype=torch.float32)
        return (logits,), None


learn = DummyLearner()



## === cell 7
predicted_ids = TEST_IDS  # all test ids are required



## === cell 8
paths = [Path(METADATA[id_].full_path) for id_ in predicted_ids]



## === cell 9
packs = packed(group_case_day_from_files(paths), n_slices_to_merge=5, step_size=1)




## === cell 10
def mask2rle(mask):
    """Convert a binary mask (numpy 2‑D) to a run‑length encoded string."""
    pixels = mask.flatten()
    pads = np.array([0])
    pixels = np.concatenate([pads, pixels, pads])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def pad_mask(mask, image_size):
    """Pad mask to square size expected by the model output (320)."""
    padded = np.zeros((image_size, image_size), dtype=mask.dtype)
    dh = image_size - mask.shape[0]
    dw = image_size - mask.shape[1]
    top = dh // 2
    left = dw // 2
    padded[top : top + mask.shape[0], left : left + mask.shape[1]] = mask
    return padded




## === cell 11
preds = []
batch_size = 64
chunks = list(chunked(packs, batch_size))

for subset in progress_bar(chunks):
    dummy_logits = torch.zeros((len(subset), 3, 320, 320), dtype=torch.float32)
    labels = (torch.sigmoid(dummy_logits) >= 0.5).cpu().numpy().astype(np.uint8)

    for pack, mask in zip(subset, labels):
        centre_path = pack[len(pack) // 2]
        test_id = get_sample_id(centre_path)
        meta = METADATA.get(test_id)

        if meta is None:
            continue
        if meta.h is None or meta.w is None:
            img = cv.imread(meta.full_path, cv.IMREAD_UNCHANGED)
            if img is not None:
                meta.h, meta.w = img.shape[:2]
            else:
                meta.h = meta.w = 320

        h, w = meta.h, meta.w

        for i, name in enumerate(("large_bowel", "small_bowel", "stomach")):
            cls_mask = pad_mask(mask[i], 320)
            cls_mask = cv.resize(cls_mask, (w, h), interpolation=cv.INTER_NEAREST)
            rle = mask2rle(cls_mask)
            preds.append({"id": test_id, "class": name, "predicted": rle})

    del dummy_logits, labels
    gc.collect()



## === cell 12
df_preds = pd.DataFrame(preds)

df_submit = pd.read_csv(DATA_DIR / "sample_submission.csv")
df_submit.columns = [c.lower() for c in df_submit.columns]

if "predicted" in df_submit.columns:
    df_submit = df_submit.drop(columns=["predicted"])

df_submit = df_submit.merge(df_preds, on=["id", "class"], how="left")
df_submit["predicted"] = df_submit["predicted"].fillna("")
df_submit.to_csv("submission.csv", index=False)

pd.read_csv("submission.csv").head(10)
