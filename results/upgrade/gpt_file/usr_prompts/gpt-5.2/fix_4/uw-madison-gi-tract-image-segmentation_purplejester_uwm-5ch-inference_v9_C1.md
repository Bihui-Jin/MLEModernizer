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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

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
    parts = p.stem.split("_")
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        return (int(parts[1]), int(parts[0]))  # (h,w)
    img = cv.imread(str(p), cv.IMREAD_UNCHANGED)
    return img.shape[:2]




## === cell 4
def get_case_day(s):
    return re.search(r"case\d+_day\d+", str(s)).group()


def group_case_day_from_files(image_files):
    groups = defaultdict(list)
    for fn in image_files:
        groups[get_case_day(fn)].append(fn)
    groups = {k: sorted(v, key=lambda x: x.name) for k, v in groups.items()}
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
        return merged


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




## === cell 6
model_name = "new"

pkl_path = Path(f"/kaggle/input/uwm-models/{model_name}.pkl")
learn = None

if pkl_path.exists():
    learn = load_learner(pkl_path)
else:
    learn = None




## === cell 7
class DummySegModel(nn.Module):
    def __init__(self, in_ch=5, out_ch=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_ch, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


def build_fallback_learner(bs=2):
    create = CreateSample()

    def _to_chw_float(b):
        if isinstance(b, (tuple, list)):
            b = b[0]
        b = b.float() / 255.0
        if b.ndim == 4:  # (bs,H,W,C) -> (bs,C,H,W)
            b = b.permute(0, 3, 1, 2).contiguous()
        return b

    norm = Normalize.from_stats(0.18161897, 0.257913)

    groups = group_case_day_from_files(TEST_FILES)
    packs0 = packed(groups, n_slices_to_merge=5, step_size=1)
    if len(packs0) == 0:
        raise RuntimeError("No test packs found; check DATA_DIR and TEST_FILES.")

    items = packs0[: max(2, bs)]  # a few items are enough to build dls

    dblock = DataBlock(
        blocks=(TransformBlock,),
        get_items=noop,
        splitter=FuncSplitter(lambda o: True),  # everything goes to valid
        get_x=noop,
        item_tfms=[create],
        batch_tfms=[_to_chw_float, norm],
    )
    dls = dblock.dataloaders(items, bs=bs, shuffle=False)
    model = DummySegModel(in_ch=5, out_ch=3)
    return Learner(dls, model)


if learn is None:
    learn = build_fallback_learner(bs=2)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_55/98840427.py in <cell line: 0>()
     50 
     51 if learn is None:
---> 52     learn = build_fallback_learner(bs=2)
     53 

/tmp/ipykernel_55/98840427.py in build_fallback_learner(bs)
     44         batch_tfms=[_to_chw_float, norm],
     45     )
---> 46     dls = dblock.dataloaders(items, bs=bs, shuffle=False)
     47     model = DummySegModel(in_ch=5, out_ch=3)
     48     return Learner(dls, model)

/usr/local/lib/python3.11/dist-packages/fastai/data/block.py in dataloaders(self, source, path, verbose, **kwargs)
    155         **kwargs
    156     ) -> DataLoaders:
--> 157         dsets = self.datasets(source, verbose=verbose)
    158         kwargs = {**self.dls_kwargs, **kwargs, 'verbose': verbose}
    159         return dsets.dataloaders(path=path, after_item=self.item_tfms, after_batch=self.batch_tfms, **kwargs)

/usr/local/lib/python3.11/dist-packages/fastai/data/block.py in datasets(self, source, verbose)
    147         splits = (self.splitter or RandomSplitter())(items)
    148         pv(f"{len(splits)} datasets of sizes {','.join([str(len(s)) for s in splits])}", verbose)
--> 149         return Datasets(items, tfms=self._combine_type_tfms(), splits=splits, dl_type=self.dl_type, n_inp=self.n_inp, verbose=verbose)
    150 
    151     def dataloaders(self, 

/usr/local/lib/python3.11/dist-packages/fastai/data/core.py in __init__(self, items, tfms, tls, n_inp, dl_type, **kwargs)
    448     ):
    449         super().__init__(dl_type=dl_type)
--> 450         self.tls = L(tls if tls else [TfmdLists(items, t, **kwargs) for t in L(ifnone(tfms,[None]))])
    451         self.n_inp = ifnone(n_inp, max(1, len(self.tls)-1))
    452 

/usr/local/lib/python3.11/dist-packages/fastai/data/core.py in <listcomp>(.0)
    448     ):
    449         super().__init__(dl_type=dl_type)
--> 450         self.tls = L(tls if tls else [TfmdLists(items, t, **kwargs) for t in L(ifnone(tfms,[None]))])
    451         self.n_inp = ifnone(n_inp, max(1, len(self.tls)-1))
    452 

/usr/local/lib/python3.11/dist-packages/fastcore/foundation.py in __call__(cls, x, *args, **kwargs)
    103     def __call__(cls, x=None, *args, **kwargs):
    104         if not args and not kwargs and x is not None and isinstance(x,cls): return x
--> 105         return super().__call__(x, *args, **kwargs)
    106 
    107 # %% ../nbs/02_foundation.ipynb

/usr/local/lib/python3.11/dist-packages/fastai/data/core.py in __init__(self, items, tfms, use_list, do_setup, split_idx, train_setup, splits, types, verbose, dl_type)
    362         if do_setup:
    363             pv(f"Setting up {self.tfms}", verbose)
--> 364             self.setup(train_setup=train_setup)
    365 
    366     def _new(self, items, split_idx=None, **kwargs):

/usr/local/lib/python3.11/dist-packages/fastai/data/core.py in setup(self, train_setup)
    385         self.tfms.setup(self, train_setup)
    386         if len(self) != 0:
--> 387             x = super().__getitem__(0) if self.splits is None else super().__getitem__(self.splits[0])[0]
    388             self.types = []
    389             for f in self.tfms.fs:

/usr/local/lib/python3.11/dist-packages/fastcore/foundation.py in __getitem__(self, idx)
    118     def _new(self, items, *args, **kwargs): return type(self)(items, *args, use_list=None, **kwargs)
    119     def __getitem__(self, idx):
--> 120         if isinstance(idx,int) and not hasattr(self.items,'iloc'): return self.items[idx]
    121         return self._get(idx) if is_indexer(idx) else L(self._get(idx), use_list=None)
    122     def copy(self): return self._new(self.items.copy())

IndexError: list index out of range

## === cell 8
groups = group_case_day_from_files(TEST_FILES)

groups_index_map = {}
for case_day, files in groups.items():
    groups_index_map[case_day] = {str(p): i for i, p in enumerate(files)}

packs = packed(groups, n_slices_to_merge=5, step_size=1)

all_pack_ids = [get_sample_id_from_pack(pack, groups_index_map) for pack in packs]
wanted_ids = set(TEST_IDS)
packs = [pack for pack, sid in zip(packs, all_pack_ids) if sid in wanted_ids]

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
learn.dls.to(device)
learn.model.to(device)
learn.model.eval()

len(packs), device



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/988192878.py in <cell line: 0>()
     12 
     13 device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
---> 14 learn.dls.to(device)
     15 learn.model.to(device)
     16 learn.model.eval()

AttributeError: 'NoneType' object has no attribute 'dls'

## === cell 9
from skimage.morphology import disk
from scipy.ndimage import binary_opening


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




## === cell 10
preds = []
batch_size = 64

with learn.no_bar():
    for subset in progress_bar(list(chunked(packs, n=batch_size))):
        test_dl = learn.dls.test_dl(subset, batch_size=batch_size, device=device)
        logits, *_ = learn.get_preds(dl=test_dl, act=noop)

        probs = torch.sigmoid(logits)
        if probs.ndim == 3:
            probs = probs.unsqueeze(1)
        if probs.shape[1] != 3:
            if probs.shape[1] > 3:
                probs = probs[:, :3]
            else:
                probs = probs.repeat(1, 3 // probs.shape[1] + 1, 1, 1)[:, :3]

        labels = (probs >= 0.5).detach().cpu().numpy().astype(np.uint8)

        for pack, mask in zip(subset, labels):
            test_id = get_sample_id_from_pack(pack, groups_index_map)
            mid_path = pack[len(pack) // 2]
            h, w = get_size_from_filename(mid_path)

            for i, name in enumerate(("large_bowel", "small_bowel", "stomach")):
                cls_mask = pad_mask(mask[i], 320)
                cls_mask = cv.resize(cls_mask, (w, h), interpolation=cv.INTER_NEAREST)
                opened = binary_opening(
                    cls_mask.astype(bool), structure=disk(5)
                ).astype(np.uint8)
                rle = mask2rle(opened) if opened.any() else ""
                preds.append({"id": test_id, "class": name, "predicted": rle})

        del logits, probs, labels
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

df_preds = pd.DataFrame(preds)
df_preds.head()



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/1460225312.py in <cell line: 0>()
      2 batch_size = 64
      3 
----> 4 with learn.no_bar():
      5     for subset in progress_bar(list(chunked(packs, n=batch_size))):
      6         test_dl = learn.dls.test_dl(subset, batch_size=batch_size, device=device)

AttributeError: 'NoneType' object has no attribute 'no_bar'

## === cell 11
df_submit = pd.read_csv(DATA_DIR / "sample_submission.csv")
df_submit = df_submit.drop(columns="predicted").merge(
    df_preds, on=["id", "class"], how="left"
)
df_submit["predicted"] = df_submit["predicted"].fillna("")
df_submit.to_csv("submission.csv", index=False)

print(df_submit.shape)
print(df_submit.head(3))
print(Path("submission.csv").resolve())

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/754680419.py in <cell line: 0>()
      1 df_submit = pd.read_csv(DATA_DIR / "sample_submission.csv")
      2 df_submit = df_submit.drop(columns="predicted").merge(
----> 3     df_preds, on=["id", "class"], how="left"
      4 )
      5 df_submit["predicted"] = df_submit["predicted"].fillna("")

NameError: name 'df_preds' is not defined
