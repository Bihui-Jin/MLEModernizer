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

3.13

# 3. Installed packages

No external packages required in the script and installed.

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

0.8184877883067355

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
from glob import glob

import numpy as np
import pandas as pd

try:
    import cv2  # type: ignore

    CV2_AVAILABLE = True
except Exception:
    cv2 = None
    CV2_AVAILABLE = False

try:
    from PIL import Image
except Exception as e:
    raise RuntimeError(
        "PIL (Pillow) is required for fallback image IO but is missing."
    ) from e

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.colors import ListedColormap
    import seaborn as sns
except Exception:
    plt = None
    Rectangle = None
    ListedColormap = None
    sns = None

try:
    from tqdm.notebook import tqdm
except Exception:
    from tqdm import tqdm

try:
    tqdm.pandas()
except Exception:
    pass

from sklearn.model_selection import StratifiedGroupKFold

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

try:
    import albumentations as A  # type: ignore
    from albumentations.pytorch import ToTensorV2  # type: ignore

    ALBU_AVAILABLE = True
except Exception:
    A = None
    ToTensorV2 = None
    ALBU_AVAILABLE = False

try:
    import segmentation_models_pytorch as smp  # type: ignore

    SMP_AVAILABLE = True
except Exception:
    smp = None
    SMP_AVAILABLE = False

MONAI_AVAILABLE = False


def _imread_gray_float01(path: str) -> np.ndarray:
    """Read a grayscale image as float32 in [0,1]. Uses cv2 if available; otherwise PIL."""
    if CV2_AVAILABLE:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"cv2.imread failed for: {path}")
        img = img.astype("float32")
    else:
        im = Image.open(path).convert("I")  # 32-bit signed integer pixels
        img = np.array(im, dtype=np.float32)

    mx = float(np.max(img)) if img.size else 0.0
    if mx > 0:
        img = img / mx
    return img.astype(np.float32)


def _resize_nearest(img: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Nearest neighbor resize for HxW or HxWxC."""
    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    if CV2_AVAILABLE:
        return cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    if img.ndim == 2:
        im = Image.fromarray(img)
        im = im.resize((out_w, out_h), resample=Image.NEAREST)
        return np.array(im)
    chs = []
    for c in range(img.shape[2]):
        im = Image.fromarray(img[..., c])
        im = im.resize((out_w, out_h), resample=Image.NEAREST)
        chs.append(np.array(im))
    return np.stack(chs, axis=-1)


class _ComposeFallback:
    """Minimal albumentations-like Compose that supports Resize + Normalize + ToTensor."""

    def __init__(self, resize_hw, mean, std, max_pixel_value=1.0, transpose_mask=True):
        self.resize_hw = resize_hw
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.max_pixel_value = float(max_pixel_value)
        self.transpose_mask = bool(transpose_mask)

    def __call__(self, image, mask=None):
        image = _resize_nearest(image, tuple(self.resize_hw)).astype(np.float32)
        if mask is not None:
            mask = _resize_nearest(mask, tuple(self.resize_hw)).astype(np.uint8)

        image = image / self.max_pixel_value
        image = (image - self.mean) / self.std

        if image.ndim == 2:
            image = image[..., None]
        image_t = torch.from_numpy(image.transpose(2, 0, 1)).float().contiguous()

        out = {"image": image_t}
        if mask is not None:
            if self.transpose_mask and mask.ndim == 3:
                mask_t = torch.from_numpy(mask.transpose(2, 0, 1)).long().contiguous()
            else:
                mask_t = torch.from_numpy(mask).long().contiguous()
            out["mask"] = mask_t
        return out


class _FallbackSMP:
    class Unet(nn.Module):
        def __init__(
            self, encoder_name=None, encoder_weights=None, in_channels=3, classes=3
        ):
            super().__init__()
            self.enc1 = nn.Sequential(
                nn.Conv2d(in_channels, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            self.pool1 = nn.MaxPool2d(2)
            self.enc2 = nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
            self.pool2 = nn.MaxPool2d(2)

            self.bottleneck = nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )

            self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
            self.dec2 = nn.Sequential(
                nn.Conv2d(128, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
            self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
            self.dec1 = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )

            self.out_conv = nn.Conv2d(32, classes, kernel_size=1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool1(e1))
            b = self.bottleneck(self.pool2(e2))

            d2 = self.up2(b)
            d2 = torch.cat([d2, e2], dim=1)
            d2 = self.dec2(d2)

            d1 = self.up1(d2)
            d1 = torch.cat([d1, e1], dim=1)
            d1 = self.dec1(d1)

            return self.out_conv(d1)

    class losses:
        class SoftBCEWithLogitsLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.bce = nn.BCEWithLogitsLoss()

            def forward(self, y_pred, y_true):
                return self.bce(y_pred, y_true)

        class DiceLoss(nn.Module):
            def __init__(self, mode="multilabel", eps=1e-6):
                super().__init__()
                self.eps = eps

            def forward(self, y_pred, y_true):
                y_pred = torch.sigmoid(y_pred)
                dims = (2, 3)
                intersection = (y_pred * y_true).sum(dims)
                union = y_pred.sum(dims) + y_true.sum(dims)
                dice = (2.0 * intersection + self.eps) / (union + self.eps)
                return 1.0 - dice.mean()


if not SMP_AVAILABLE:
    smp = _FallbackSMP()
    print(
        "segmentation_models_pytorch not found; using local fallback UNet + losses (score may differ)."
    )



## === cell 1
print(f"Number of available CPUs: {os.cpu_count()}")
print(f"Number of available GPUs: {torch.cuda.device_count()}")



## === cell 2
DIR_PATH = "/kaggle/input/uw-madison-gi-tract-image-segmentation/"

pd.set_option("display.max_colwidth", 400)

if ListedColormap is not None:
    CMAP1 = ListedColormap([[0, 0, 0, 0], [1, 0, 0, 1]])
    CMAP2 = ListedColormap([[0, 0, 0, 0], [0, 1, 0, 1]])
    CMAP3 = ListedColormap([[0, 0, 0, 0], [0, 0, 1, 1]])
else:
    CMAP1 = CMAP2 = CMAP3 = None

RANDOM_SEED = 0

IMAGE_NORMALIZE_MEAN = (0.485, 0.456, 0.406)
IMAGE_NORMALIZE_SD = (0.229, 0.224, 0.225)

IMAGE_RESIZE = [288, 288]

BATCH_SIZE_TRAIN = 32
BATCH_SIZE_VALID = BATCH_SIZE_TRAIN * 2
BATCH_SIZE_TEST = BATCH_SIZE_TRAIN * 2

_CPU = os.cpu_count() or 4
DATA_LOADER_NUM_WORKERS = min(12, max(2, _CPU // 2))
DATA_LOADER_PREFETCH_FACTOR = 4

NUM_CLASSES = 3
CLASS_NAMES = ["large_bowel", "small_bowel", "stomach"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 5

MODEL_PARAMS_FILE_NAME = "GIT-Seg-efficientnet-b1.pth"
MODEL_PARAMS_LOAD_FILE_PATH = (
    "/kaggle/input/git-seg/pytorch/default/2/GIT-Seg-efficientnet-b1.pth"
)

TRAIN_VALID_SPLIT = True
TEST_PREDICT = True

SAVE_TRAIN_VALID_MODEL = True
LOAD_MODEL_FOR_TEST_PREDICT = True



## === cell 3
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass


def _seed_worker(worker_id: int):
    seed = RANDOM_SEED + worker_id
    random.seed(seed)
    np.random.seed(seed)


_DATALOADER_GENERATOR = torch.Generator()
_DATALOADER_GENERATOR.manual_seed(RANDOM_SEED)



## === cell 4
data = pd.read_csv(DIR_PATH + "train.csv")
data.head()



## === cell 5
data_nonnaseg = data.loc[data.segmentation.notna(), :]
data_nonnaseg.head()



## === cell 6
data[["case", "day", "slice"]] = data["id"].str.extract(
    r"case(\d+)_day(\d+)_slice_(\d+)"
)
data




## === cell 7
def _id_to_case_day_slice(id_str: str) -> tuple[int, int, int]:
    a, b, c = id_str.split("_")
    case = int(a[4:])
    day = int(b[3:])
    slice_idx = int(c.split("slice_")[1])
    return case, day, slice_idx


def _find_slice_png_path(base_dir: str, train: bool, id_str: str) -> str:
    case, day, slice_idx = _id_to_case_day_slice(id_str)
    scans_dir = os.path.join(
        base_dir,
        "train" if train else "test",
        f"case{case}",
        f"case{case}_day{day}",
        "scans",
    )
    pattern = os.path.join(scans_dir, f"slice_{slice_idx}_*.png")
    matches = glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No scan found for id={id_str} using pattern: {pattern}"
        )
    matches.sort()
    return matches[0]


def _read_hw_from_png(path: str) -> tuple[int, int]:
    if CV2_AVAILABLE:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"cv2.imread failed for: {path}")
        h, w = img.shape[:2]
        return int(h), int(w)
    with Image.open(path) as im:
        w, h = im.size
        return int(h), int(w)


def get_path_df(train=True):
    base = "train" if train else "test"
    paths = glob(os.path.join(DIR_PATH, base, "case*/case*_day*/scans/*.png"))
    path_df = pd.DataFrame(paths, columns=["image_path"])
    path_df[["case", "day", "slice", "slice_w", "slice_h", "px_w", "px_h"]] = (
        path_df.image_path.str.extract(
            r".*/case(\d+)_day(\d+)/scans/slice_(\d+)_(\d+)_(\d+)_([0-9]+(?:\.[0-9]+)?)_([0-9]+(?:\.[0-9]+)?)\.png"
        )
    )
    return path_df


path_df = pd.DataFrame({"image_path": []})



## === cell 8
data.info()



## === cell 9
path_df.info()



## === cell 10
_unique_ids = data["id"].drop_duplicates().tolist()
_id_to_path = {}
_id_to_hw = {}
for _id in tqdm(_unique_ids, desc="Mapping train id->image_path"):
    p = _find_slice_png_path(DIR_PATH, True, _id)
    _id_to_path[_id] = p
    _id_to_hw[_id] = _read_hw_from_png(p)

data["image_path"] = data["id"].map(_id_to_path)
data["slice_h"] = data["id"].map(lambda x: _id_to_hw[x][0]).astype(np.uint32)
data["slice_w"] = data["id"].map(lambda x: _id_to_hw[x][1]).astype(np.uint32)

data["px_w"] = np.float32(0.0)
data["px_h"] = np.float32(0.0)

data



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3703601205.py in <cell line: 0>()
      5 _id_to_hw = {}
      6 for _id in tqdm(_unique_ids, desc="Mapping train id->image_path"):
----> 7     p = _find_slice_png_path(DIR_PATH, True, _id)
      8     _id_to_path[_id] = p
      9     _id_to_hw[_id] = _read_hw_from_png(p)

/tmp/ipykernel_55/837926081.py in _find_slice_png_path(base_dir, train, id_str)
     13 
     14 def _find_slice_png_path(base_dir: str, train: bool, id_str: str) -> str:
---> 15     case, day, slice_idx = _id_to_case_day_slice(id_str)
     16     scans_dir = os.path.join(
     17         base_dir,

/tmp/ipykernel_55/837926081.py in _id_to_case_day_slice(id_str)
      5 def _id_to_case_day_slice(id_str: str) -> tuple[int, int, int]:
      6     # id format: case{case}_day{day}_slice_{slice}
----> 7     a, b, c = id_str.split("_")
      8     case = int(a[4:])
      9     day = int(b[3:])

ValueError: too many values to unpack (expected 3)

## === cell 11
data.info()



## === cell 12
data.px_w.unique(), data.px_h.unique()



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/3700244779.py in <cell line: 0>()
----> 1 data.px_w.unique(), data.px_h.unique()
      2 

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'px_w'

## === cell 13
data.case.unique(), data.day.unique(), data.slice.unique(), data.slice_w.unique(), data.slice_h.unique()



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/28830184.py in <cell line: 0>()
----> 1 data.case.unique(), data.day.unique(), data.slice.unique(), data.slice_w.unique(), data.slice_h.unique()
      2 

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'slice_w'

## === cell 14
int_cols = ["case", "day", "slice", "slice_w", "slice_h"]
for c in int_cols:
    data[c] = pd.to_numeric(data[c], errors="coerce")
data[int_cols] = data[int_cols].fillna(0).astype(np.uint32)

float_cols = ["px_w", "px_h"]
for c in float_cols:
    data[c] = pd.to_numeric(data[c], errors="coerce")
data[float_cols] = data[float_cols].fillna(0).astype(np.float32)

data.info()




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'slice_w'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/3481517056.py in <cell line: 0>()
      1 int_cols = ["case", "day", "slice", "slice_w", "slice_h"]
      2 for c in int_cols:
----> 3     data[c] = pd.to_numeric(data[c], errors="coerce")
      4 data[int_cols] = data[int_cols].fillna(0).astype(np.uint32)
      5 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'slice_w'

## === cell 15
def rle_decode(mask_rle, shape):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height,width) of array to return

    Competition pixels are numbered from top-to-bottom, then left-to-right.
    That corresponds to flattening in Fortran order (column-major) on (H, W).
    """
    s = np.asarray(mask_rle.split(), dtype=np.int64)
    starts = s[0::2] - 1
    lengths = s[1::2]
    if starts.size == 0:
        return np.zeros(shape, dtype=np.uint8)
    ends = starts + lengths

    flat = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    diff = np.zeros(flat.size + 1, dtype=np.int32)
    np.add.at(diff, starts, 1)
    np.add.at(diff, ends, -1)
    flat[:] = (np.cumsum(diff[:-1]) > 0).astype(np.uint8)
    return flat.reshape(shape, order="F")


def rle_encode(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted.
    Competition expects empty string for empty mask.

    IMPORTANT: encode using Fortran order (column-major) to match the competition.
    """
    img = (img > 0).astype(np.uint8)

    pixels = img.flatten(order="F")
    if pixels.max() == 0:
        return ""
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 16
_ID_CACHE_BUILT = False
_ID_TO_IMAGE_PATH = {}
_ID_TO_HW = {}
_ID_TO_RLES = {}  # id -> dict{class_name: rle_or_nan}


def _build_id_cache(df: pd.DataFrame):
    global _ID_CACHE_BUILT, _ID_TO_IMAGE_PATH, _ID_TO_HW, _ID_TO_RLES
    _ID_TO_IMAGE_PATH = df.drop_duplicates("id").set_index("id")["image_path"].to_dict()
    hw_df = df.drop_duplicates("id").set_index("id")[["slice_h", "slice_w"]]
    _ID_TO_HW = {k: (int(v[0]), int(v[1])) for k, v in hw_df.to_dict("index").items()}
    seg_piv = df.pivot_table(
        index="id", columns="class", values="segmentation", aggfunc="first"
    )
    _ID_TO_RLES = seg_piv.to_dict(orient="index")
    _ID_CACHE_BUILT = True


def get_mask(id_, data_):
    if not _ID_CACHE_BUILT:
        _build_id_cache(data_)
    hw = _ID_TO_HW.get(id_)
    if hw is None:
        return None
    h, w = hw
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    rles = _ID_TO_RLES.get(id_)
    if not rles:
        return mask
    for i, class_ in enumerate(CLASS_NAMES):
        rle = rles.get(class_)
        if rle is not None and not pd.isna(rle):
            mask[..., i] = rle_decode(rle, (h, w))
    return mask




## === cell 17
example_path = data["image_path"].iloc[0] if len(data) else None
print("Example image path:", example_path)
print(
    "CV2_AVAILABLE:",
    CV2_AVAILABLE,
    "| ALBU_AVAILABLE:",
    ALBU_AVAILABLE,
    "| SMP_AVAILABLE:",
    SMP_AVAILABLE,
)




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'image_path'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/809217740.py in <cell line: 0>()
----> 1 example_path = data["image_path"].iloc[0] if len(data) else None
      2 print("Example image path:", example_path)
      3 print(
      4     "CV2_AVAILABLE:",
      5     CV2_AVAILABLE,

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'image_path'

## === cell 18
def load_image(id_, data_):
    if not _ID_CACHE_BUILT:
        _build_id_cache(data_)
    p = _ID_TO_IMAGE_PATH.get(id_)
    if not isinstance(p, str) or (not os.path.exists(p)):
        raise FileNotFoundError(f"image_path missing for id={id_}: {p}")
    return _imread_gray_float01(p)




## === cell 19
def display_image(
    id_,
    data_,
    pred_mask=None,
    apply_CLAHE=False,
    show_orig_img=True,
    show_true_mask=True,
    show_pred_mask=False,
):
    if plt is None:
        print("Matplotlib not available; skipping display.")
        return

    img = load_image(id_, data_)
    img_u8 = (img * 255).astype(np.uint8)

    if apply_CLAHE and CV2_AVAILABLE:
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2, 2))
        img_u8 = clahe.apply(img_u8)

    mask = get_mask(id_, data_)

    plt.figure(figsize=(9, 3))

    i = 1
    if show_orig_img:
        plt.subplot(1, 3, i)
        i += 1
        plt.imshow(img_u8, cmap="bone")
        plt.title(f"{id_} image")
        plt.axis("off")

    if show_true_mask and mask is not None:
        plt.subplot(1, 3, i)
        i += 1
        plt.imshow(img_u8, cmap="bone")
        plt.title("Image with true mask")
        if CMAP1 is not None:
            plt.imshow(mask[..., 0], cmap=CMAP1)
            plt.imshow(mask[..., 1], cmap=CMAP2)
            plt.imshow(mask[..., 2], cmap=CMAP3)
        plt.axis("off")

        if Rectangle is not None and CMAP1 is not None:
            handles = [
                Rectangle((0, 0), 1, 1, color=CMAP1(1.0)),
                Rectangle((0, 0), 1, 1, color=CMAP2(1.0)),
                Rectangle((0, 0), 1, 1, color=CMAP3(1.0)),
            ]
            labels = ["Large Bowel", "Small Bowel", "Stomach"]
            plt.legend(
                handles,
                labels,
                bbox_to_anchor=(1.0, -0.4),
                loc="lower right",
                borderaxespad=0.0,
            )

    if show_pred_mask and pred_mask is not None:
        plt.subplot(1, 3, i)
        plt.imshow(img_u8, cmap="bone")
        plt.title("Image with predicted mask")
        if CMAP1 is not None:
            plt.imshow(pred_mask[..., 0], cmap=CMAP1)
            plt.imshow(pred_mask[..., 1], cmap=CMAP2)
            plt.imshow(pred_mask[..., 2], cmap=CMAP3)
        plt.axis("off")

    plt.tight_layout()
    plt.show()




## === cell 20
if False:
    some_id = data["id"].iloc[0]
    display_image(some_id, data)



## === cell 21
if False:
    some_id = data["id"].iloc[0]
    display_image(some_id, data, apply_CLAHE=True)



## === cell 22
if False:
    example_ids = data["id"].unique()[:3]
    for _id in example_ids:
        display_image(_id, data, apply_CLAHE=True)



## === cell 23
if False:
    example_ids = data["id"].unique()[:3]
    for _id in example_ids:
        display_image(_id, data, apply_CLAHE=True)




## === cell 24
def display_multiple_slices(
    id_array, data_, apply_CLAHE=False, show_pred_mask=False, pred_mask_array=None
):
    if plt is None:
        print("Matplotlib not available; skipping display.")
        return

    l = len(id_array)
    if l == 0:
        print("No ids to display.")
        return
    rows = int(np.ceil(l / 5))
    max_cols = 5
    data_subset = data_.loc[data_.id.isin(id_array), :].copy()

    plt.figure(figsize=(max_cols * 3, rows * 3))

    for i in range(l):
        id_ = id_array[i]
        data_one = data_subset.loc[data_subset.id == id_]
        if len(data_one) == 0:
            continue
        p = data_one.image_path.iloc[0]
        if not isinstance(p, str) or (not os.path.exists(p)):
            continue

        img = _imread_gray_float01(p)
        if apply_CLAHE and CV2_AVAILABLE:
            clahe_local = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2, 2))
            img_u8 = (img * 255).astype(np.uint8)
            img_u8 = clahe_local.apply(img_u8)
            img_disp = img_u8
        else:
            img_disp = img

        if show_pred_mask and pred_mask_array is not None:
            mask = pred_mask_array[i]
        else:
            mask = get_mask(id_, data_)

        plt.subplot(rows, max_cols, i + 1)
        plt.imshow(img_disp, cmap="bone")
        plt.title(id_)
        if mask is not None and CMAP1 is not None:
            plt.imshow(mask[..., 0], cmap=CMAP1)
            plt.imshow(mask[..., 1], cmap=CMAP2)
            plt.imshow(mask[..., 2], cmap=CMAP3)
        plt.axis("off")

    plt.tight_layout()
    plt.show()




## === cell 25
if False:
    display_multiple_slices(
        data.query(
            "case == 123 and day == 20 and slice >= 63 and slice <= 70"
        ).id.unique(),
        data,
        apply_CLAHE=True,
    )



## === cell 26
if False:
    display_multiple_slices(
        data.query(
            "case == 131 and day == 0 and slice > 55 and slice <= 70"
        ).id.unique(),
        data,
        apply_CLAHE=True,
    )



## === cell 27
data.loc[data.segmentation.isna(), :].head()



## === cell 28
data.isna().sum()



## === cell 29
print(
    f"Num cases : {len(data.case.unique())}         Num unique days : {len(data.day.unique())}           Num unique slices : {len(data.slice.unique())}"
)



## === cell 30
count_df = (
    data[["id", "slice_w", "slice_h"]]
    .drop_duplicates()[["slice_w", "slice_h"]]
    .value_counts()
    .reset_index(name="count")
)
count_df["percent"] = count_df["count"] * 100 / sum(count_df["count"])
print(sum(count_df["count"]))
count_df



## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/4032712873.py in <cell line: 0>()
      1 count_df = (
----> 2     data[["id", "slice_w", "slice_h"]]
      3     .drop_duplicates()[["slice_w", "slice_h"]]
      4     .value_counts()
      5     .reset_index(name="count")

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['slice_w', 'slice_h'] not in index"

## === cell 31
count_df = (
    data[["id", "px_w", "px_h"]]
    .drop_duplicates()[["px_w", "px_h"]]
    .value_counts()
    .reset_index(name="count")
)
count_df["percent"] = count_df["count"] * 100 / sum(count_df["count"])
print(sum(count_df["count"]))
count_df



## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/2581462106.py in <cell line: 0>()
      1 count_df = (
----> 2     data[["id", "px_w", "px_h"]]
      3     .drop_duplicates()[["px_w", "px_h"]]
      4     .value_counts()
      5     .reset_index(name="count")

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['px_w', 'px_h'] not in index"

## === cell 32
if False and sns is not None and plt is not None:
    day_dist = (
        data[["case", "day"]]
        .drop_duplicates()["case"]
        .value_counts()
        .reset_index(name="num_days")
    )

    sns.histplot(
        data=day_dist,
        x="num_days",
        bins=range(1, int(day_dist["num_days"].max()) + 1),
        discrete=True,
    )
    plt.xlabel("Number of Days per Case")
    plt.ylabel("Number of Cases")
    plt.title("Distribution of Days per Case")
    plt.show()



## === cell 33
if False and sns is not None and plt is not None:
    slice_dist = (
        data[["case", "day", "slice"]]
        .drop_duplicates()[["case", "day"]]
        .value_counts()
        .reset_index(name="num_slices")
    )

    sns.histplot(
        data=slice_dist,
        x="num_slices",
        bins=range(1, int(slice_dist["num_slices"].max()) + 1),
        discrete=True,
    )
    plt.xlabel("Number of slices per case-days")
    plt.ylabel("Number of specific case-days")
    plt.title("Distribution of slices per case-day")
    plt.show()



## === cell 34
slice_dist = (
    data[["case", "day", "slice"]]
    .drop_duplicates()[["case", "day"]]
    .value_counts()
    .reset_index(name="num_slices")
)
slice_dist.loc[slice_dist.num_slices == 80, :]



## === cell 35
case_day_slice_df = data[
    ["case", "day", "slice", "slice_w", "slice_h"]
].drop_duplicates()
case_day_slice_df.merge(case_day_slice_df, on=["case", "day"]).query(
    "(slice_w_x != slice_w_y) | (slice_h_x != slice_h_y)"
)



## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/89756251.py in <cell line: 0>()
----> 1 case_day_slice_df = data[
      2     ["case", "day", "slice", "slice_w", "slice_h"]
      3 ].drop_duplicates()
      4 case_day_slice_df.merge(case_day_slice_df, on=["case", "day"]).query(
      5     "(slice_w_x != slice_w_y) | (slice_h_x != slice_h_y)"

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['slice_w', 'slice_h'] not in index"

## === cell 36
case_day_slice_df = data[
    ["case", "day", "slice", "slice_w", "slice_h"]
].drop_duplicates()
case_day_slice_df.merge(case_day_slice_df, on=["case"]).query(
    "(slice_w_x != slice_w_y) | (slice_h_x != slice_h_y)"
)



## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/2670137713.py in <cell line: 0>()
----> 1 case_day_slice_df = data[
      2     ["case", "day", "slice", "slice_w", "slice_h"]
      3 ].drop_duplicates()
      4 case_day_slice_df.merge(case_day_slice_df, on=["case"]).query(
      5     "(slice_w_x != slice_w_y) | (slice_h_x != slice_h_y)"

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['slice_w', 'slice_h'] not in index"

## === cell 37
case_day_slice_df = data[["case", "day", "slice", "px_w", "px_h"]].drop_duplicates()
case_day_slice_df.merge(case_day_slice_df, on=["case", "day"]).query(
    "(px_w_x != px_w_y) | (px_h_x != px_h_y)"
)



## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/128668739.py in <cell line: 0>()
----> 1 case_day_slice_df = data[["case", "day", "slice", "px_w", "px_h"]].drop_duplicates()
      2 case_day_slice_df.merge(case_day_slice_df, on=["case", "day"]).query(
      3     "(px_w_x != px_w_y) | (px_h_x != px_h_y)"
      4 )
      5 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['px_w', 'px_h'] not in index"

## === cell 38
case_day_slice_df = data[["case", "day", "slice", "px_w", "px_h"]].drop_duplicates()
case_day_slice_df.merge(case_day_slice_df, on=["case"]).query(
    "(px_w_x != px_w_y) | (px_h_x != px_h_y)"
)



## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/1032694782.py in <cell line: 0>()
----> 1 case_day_slice_df = data[["case", "day", "slice", "px_w", "px_h"]].drop_duplicates()
      2 case_day_slice_df.merge(case_day_slice_df, on=["case"]).query(
      3     "(px_w_x != px_w_y) | (px_h_x != px_h_y)"
      4 )
      5 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['px_w', 'px_h'] not in index"

## === cell 39
num_missing_seg_masks = data.segmentation.isna().sum()
print(
    f"Missing Seg Mask \n count = {num_missing_seg_masks}\n percentage = {num_missing_seg_masks/len(data)*100}"
)



## === cell 40
data["class"].value_counts()



## === cell 41
if False and sns is not None and plt is not None:
    na_counts = (
        data.groupby("class")["segmentation"]
        .apply(lambda s: s.isna().sum())
        .reset_index(name="count")
    )
    na_counts["percent"] = (
        100 * na_counts["count"] / data.groupby("class")["segmentation"].size().values
    )

    sns.set_style("whitegrid")
    ax = sns.barplot(data=na_counts, x="class", y="percent")

    for i, row in na_counts.iterrows():
        ax.text(
            i,
            row["percent"] + 1,
            f"{row['percent']:.2f}% ({row['count']})",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.ylabel("Percentage")
    plt.xlabel("Segmentation Class")
    plt.title("Missing Segmentation Masks")
    plt.show()



## === cell 42
if False and sns is not None and plt is not None:
    case_day_seg_missing = (
        data[["case", "day", "class", "segmentation"]]
        .groupby(["case", "day", "class"])["segmentation"]
        .apply(lambda s: s.isna().sum())
        .reset_index(name="count")
        .sort_values(by="count", ascending=False)
    )

    sns.boxplot(
        data=case_day_seg_missing,
        x="class",
        y="count",
    )
    sns.stripplot(
        data=case_day_seg_missing,
        x="class",
        y="count",
        color="black",
        size=3,
        jitter=True,
        alpha=0.4,
    )
    plt.ylabel("Missing Mask Count")
    plt.xlabel("Segmentation Class")
    plt.title("Distribution of Missing Masks per Class (by Case-Day)")
    plt.show()



## === cell 43
if False:
    for _id in data["id"].unique()[:4]:
        display_image(_id, data, apply_CLAHE=True)



## === cell 44
if False:
    for _id in data["id"].unique()[4:8]:
        display_image(_id, data, apply_CLAHE=True)



## === cell 45
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
index_train, index_valid = next(
    sgkf.split(data.id, data.segmentation.isna(), data.case)
)



## === cell 46
len(index_train), len(index_valid)



## === cell 47
data_train = data.iloc[index_train, :]
data_valid = data.iloc[index_valid, :]



## === cell 48
data_train.head()



## === cell 49
data_valid.head()



## === cell 50
print(len(data_train.case.unique()), len(data_valid.case.unique()))



## === cell 51
data_train_sub = data_train.drop_duplicates("id").reset_index(drop=True)
data_valid_sub = data_valid.drop_duplicates("id").reset_index(drop=True)

print("Using full split (unique ids for Dataset):")
print(
    "train rows (orig):",
    len(data_train),
    "unique ids:",
    data_train["id"].nunique(),
    "cases:",
    data_train["case"].nunique(),
)
print(
    "valid rows (orig):",
    len(data_valid),
    "unique ids:",
    data_valid["id"].nunique(),
    "cases:",
    data_valid["case"].nunique(),
)
print(
    "train dataset rows:",
    len(data_train_sub),
    "| valid dataset rows:",
    len(data_valid_sub),
)



## === cell 52
missing_masks_train = data_train.segmentation.isna().sum()
missing_masks_valid = data_valid.segmentation.isna().sum()
print(missing_masks_train, missing_masks_train * 100 / len(data_train))
print(missing_masks_valid, missing_masks_valid * 100 / len(data_valid))



## === cell 53
na_counts_train = (
    data_train.groupby("class")["segmentation"]
    .apply(lambda s: s.isna().sum())
    .reset_index(name="count")
)
na_counts_train["percent"] = (
    100
    * na_counts_train["count"]
    / data_train.groupby("class")["segmentation"].size().values
)

na_counts_valid = (
    data_valid.groupby("class")["segmentation"]
    .apply(lambda s: s.isna().sum())
    .reset_index(name="count")
)
na_counts_valid["percent"] = (
    100
    * na_counts_valid["count"]
    / data_valid.groupby("class")["segmentation"].size().values
)

na_counts_train, na_counts_valid



## === cell 54
_RLE_DECODE_CACHE = {}


def _rle_decode_cached(mask_rle: str, shape_hw: tuple[int, int]) -> np.ndarray:
    key = (mask_rle, int(shape_hw[0]), int(shape_hw[1]))
    m = _RLE_DECODE_CACHE.get(key)
    if m is None:
        m = rle_decode(mask_rle, shape_hw)
        _RLE_DECODE_CACHE[key] = m
    return m


def _build_id_to_rles(df_full: pd.DataFrame) -> dict:
    seg_piv = df_full.pivot_table(
        index="id", columns="class", values="segmentation", aggfunc="first"
    )
    return seg_piv.to_dict(orient="index")


_ID_TO_RLES_TRAIN = _build_id_to_rles(data_train)
_ID_TO_RLES_VALID = _build_id_to_rles(data_valid)


class GITractDataset(Dataset):
    def __init__(self, df_unique_ids, is_train=True, transforms=None, id_to_rles=None):
        self.df = df_unique_ids
        self.id_ = df_unique_ids["id"].to_numpy()
        self.is_train = is_train
        self.transforms = transforms

        dd = df_unique_ids.set_index("id")
        self.id_to_path = dd["image_path"].to_dict()
        self.id_to_hw = {
            k: (int(v["slice_h"]), int(v["slice_w"]))
            for k, v in dd[["slice_h", "slice_w"]].to_dict("index").items()
        }
        self.id_to_rles = id_to_rles if (is_train and id_to_rles is not None) else {}

        self._cache_img = None
        self._cache_mask = None

    def __len__(self):
        return self.id_.shape[0]

    def __getitem__(self, idx):
        id_ = self.id_[idx]
        h, w = self.id_to_hw[id_]

        p = self.id_to_path.get(id_)
        if not isinstance(p, str) or (not os.path.exists(p)):
            raise FileNotFoundError(f"image_path missing for id={id_}: {p}")

        img = _imread_gray_float01(p)
        img = np.tile(img[..., None], [1, 1, 3])

        if self.is_train:
            mask = np.zeros((h, w, NUM_CLASSES), dtype=np.uint8)
            rles = self.id_to_rles.get(id_, {})
            for i, class_ in enumerate(CLASS_NAMES):
                rle = rles.get(class_)
                if rle is not None and not pd.isna(rle):
                    mask[..., i] = _rle_decode_cached(str(rle), (h, w))

            if self.transforms:
                augmented = self.transforms(image=img, mask=mask)
                img_t = augmented["image"]
                mask_t = augmented["mask"]
            else:
                img_t = torch.from_numpy(img.transpose(2, 0, 1)).float().contiguous()
                mask_t = torch.from_numpy(mask.transpose(2, 0, 1)).long().contiguous()

            return img_t, mask_t, id_
        else:
            if self.transforms:
                augmented = self.transforms(image=img)
                img_t = augmented["image"]
            else:
                img_t = torch.from_numpy(img.transpose(2, 0, 1)).float().contiguous()

            return img_t, id_, h, w




## === cell 55
if ALBU_AVAILABLE:
    transform_train = A.Compose(
        [
            A.Resize(
                IMAGE_RESIZE[0],
                IMAGE_RESIZE[1],
                interpolation=cv2.INTER_LINEAR if CV2_AVAILABLE else 1,
                mask_interpolation=cv2.INTER_NEAREST if CV2_AVAILABLE else 0,
            ),
            A.Normalize(
                mean=IMAGE_NORMALIZE_MEAN, std=IMAGE_NORMALIZE_SD, max_pixel_value=1.0
            ),
            ToTensorV2(transpose_mask=True),
        ]
    )

    transform_valid = A.Compose(
        [
            A.Resize(
                IMAGE_RESIZE[0],
                IMAGE_RESIZE[1],
                interpolation=cv2.INTER_LINEAR if CV2_AVAILABLE else 1,
                mask_interpolation=cv2.INTER_NEAREST if CV2_AVAILABLE else 0,
            ),
            A.Normalize(
                mean=IMAGE_NORMALIZE_MEAN, std=IMAGE_NORMALIZE_SD, max_pixel_value=1.0
            ),
            ToTensorV2(transpose_mask=True),
        ]
    )
else:
    transform_train = _ComposeFallback(
        resize_hw=IMAGE_RESIZE,
        mean=IMAGE_NORMALIZE_MEAN,
        std=IMAGE_NORMALIZE_SD,
        max_pixel_value=1.0,
        transpose_mask=True,
    )
    transform_valid = _ComposeFallback(
        resize_hw=IMAGE_RESIZE,
        mean=IMAGE_NORMALIZE_MEAN,
        std=IMAGE_NORMALIZE_SD,
        max_pixel_value=1.0,
        transpose_mask=True,
    )



## === cell 56
dataset_train = GITractDataset(
    data_train_sub, transforms=transform_train, id_to_rles=_ID_TO_RLES_TRAIN
)
dataset_valid = GITractDataset(
    data_valid_sub, transforms=transform_valid, id_to_rles=_ID_TO_RLES_VALID
)

dataloader_train = DataLoader(
    dataset_train,
    batch_size=BATCH_SIZE_TRAIN,
    shuffle=True,
    drop_last=True,
    num_workers=DATA_LOADER_NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(DATA_LOADER_NUM_WORKERS > 0),
    prefetch_factor=(
        DATA_LOADER_PREFETCH_FACTOR if DATA_LOADER_NUM_WORKERS > 0 else None
    ),
    worker_init_fn=_seed_worker,
    generator=_DATALOADER_GENERATOR,
)
dataloader_valid = DataLoader(
    dataset_valid,
    batch_size=BATCH_SIZE_VALID,
    shuffle=False,
    num_workers=DATA_LOADER_NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(DATA_LOADER_NUM_WORKERS > 0),
    prefetch_factor=(
        DATA_LOADER_PREFETCH_FACTOR if DATA_LOADER_NUM_WORKERS > 0 else None
    ),
    worker_init_fn=_seed_worker,
    generator=_DATALOADER_GENERATOR,
)



## --- ERROR in cell 56, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'image_path'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/1755646580.py in <cell line: 0>()
----> 1 dataset_train = GITractDataset(
      2     data_train_sub, transforms=transform_train, id_to_rles=_ID_TO_RLES_TRAIN
      3 )
      4 dataset_valid = GITractDataset(
      5     data_valid_sub, transforms=transform_valid, id_to_rles=_ID_TO_RLES_VALID

/tmp/ipykernel_55/625800517.py in __init__(self, df_unique_ids, is_train, transforms, id_to_rles)
     31 
     32         dd = df_unique_ids.set_index("id")
---> 33         self.id_to_path = dd["image_path"].to_dict()
     34         self.id_to_hw = {
     35             k: (int(v["slice_h"]), int(v["slice_w"]))

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'image_path'

## === cell 57
dataset = next(iter(dataloader_train))
img, mask, id_ = dataset
print(img.shape, mask.shape, len(id_))



## --- ERROR in cell 57, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1272461141.py in <cell line: 0>()
----> 1 dataset = next(iter(dataloader_train))
      2 img, mask, id_ = dataset
      3 print(img.shape, mask.shape, len(id_))
      4 

NameError: name 'dataloader_train' is not defined

## === cell 58
idx = 0
np.max(img[idx].numpy()), np.min(img[idx].numpy())



## --- ERROR in cell 58, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/132881774.py in <cell line: 0>()
      1 idx = 0
----> 2 np.max(img[idx].numpy()), np.min(img[idx].numpy())
      3 

NameError: name 'img' is not defined

## === cell 59
type(img[idx].numpy()[0, 0, 0]), type(mask[idx].numpy()[0, 0, 0])




## --- ERROR in cell 59, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4141723581.py in <cell line: 0>()
----> 1 type(img[idx].numpy()[0, 0, 0]), type(mask[idx].numpy()[0, 0, 0])
      2 
      3 

NameError: name 'img' is not defined

## === cell 60
def display_dataset(
    dataset_batch,
    display_orig=False,
    num_images=None,
    denormalize=False,
    apply_CLAHE=False,
):
    if plt is None:
        print("Matplotlib not available; skipping display.")
        return

    img_arr, mask_arr, id_arr = dataset_batch
    if num_images is None:
        num_images = len(img_arr)
    max_cols = 5

    if display_orig:
        num_images = 5
        rows = 2
        plt.figure(figsize=(max_cols * 3, rows * 3))
        ids_shown = []
    else:
        rows = int(np.ceil(num_images / max_cols))
        plt.figure(figsize=(max_cols * 3, rows * 3))

    for idx in range(num_images):
        img, mask, id_ = img_arr[idx], mask_arr[idx], id_arr[idx]
        img = img.permute(1, 2, 0)
        if denormalize:
            img = img * torch.tensor(IMAGE_NORMALIZE_SD) + torch.tensor(
                IMAGE_NORMALIZE_MEAN
            )
            img = img.clamp(0, 1)
        img = img.cpu().numpy()
        img = (img * 255).astype(np.uint8)

        mask = mask.permute(1, 2, 0).cpu().numpy()

        plt.subplot(rows, max_cols, idx + 1)
        plt.imshow(img[:, :, 0], cmap="bone")
        plt.title(f"{idx} : {id_}")

        if CMAP1 is not None:
            plt.imshow(mask[..., 0], cmap=CMAP1)
            plt.imshow(mask[..., 1], cmap=CMAP2)
            plt.imshow(mask[..., 2], cmap=CMAP3)
        plt.axis("off")

        if display_orig:
            ids_shown.append(id_)

    plt.tight_layout()
    plt.show()




## === cell 61
if False:
    display_dataset(dataset, num_images=5, denormalize=True)



## === cell 62
if False:
    display_dataset(dataset, display_orig=True, denormalize=True, apply_CLAHE=True)



## === cell 63
smp_encoder_weights = (
    None if TEST_PREDICT and LOAD_MODEL_FOR_TEST_PREDICT else "imagenet"
)
model = smp.Unet(
    encoder_name="efficientnet-b1",
    encoder_weights=smp_encoder_weights,
    in_channels=3,
    classes=NUM_CLASSES,
)
model.to(DEVICE)



## === cell 64
optimizer = optim.Adam(model.parameters(), lr=1e-3)



## === cell 65
dice_loss = smp.losses.DiceLoss(mode="multilabel")
BCE_loss = smp.losses.SoftBCEWithLogitsLoss()


def loss_fn(y_pred, y_true, loss_wt=0.5):
    return dice_loss(y_pred, y_true) * loss_wt + BCE_loss(y_pred, y_true) * (
        1 - loss_wt
    )




## === cell 66
class DiceScoreCustom:
    def __init__(self, num_classes, eps=1e-6):
        self.num_classes = num_classes
        self.eps = eps
        self.reset()

    def reset(self):
        self.dice_sum = 0.0
        self.image_count = 0
        self.organ_dice_sum = torch.zeros(self.num_classes)
        self.organ_count = torch.zeros(self.num_classes)

    def update(self, preds, targets):
        I = (targets & preds).sum((2, 3))
        U = (targets | preds).sum((2, 3))
        dice = (2 * I) / (U + I + self.eps)
        non_empty = U > 0
        organ_counts = non_empty.sum(dim=1)
        dice_per_image = dice.sum(dim=1) / organ_counts.clamp(min=1)

        self.dice_sum += dice_per_image.sum().item()
        self.image_count += dice_per_image.numel()
        self.organ_dice_sum += dice.sum(dim=0).detach().cpu()
        self.organ_count += non_empty.sum(dim=0).detach().cpu()

    def compute(self):
        overall = (
            torch.tensor(self.dice_sum / self.image_count)
            if self.image_count > 0
            else torch.tensor(0.0)
        )
        per_organ = torch.where(
            self.organ_count > 0,
            self.organ_dice_sum / self.organ_count,
            torch.tensor(0.0),
        )
        return overall, per_organ




## === cell 67
class HausdorffDistanceCustom:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.h3d_sum = 0.0
        self.image3d_count = 0
        self.organ_h3d_sum = np.zeros(self.num_classes)
        self.organ_count_sum = np.zeros(self.num_classes)

    def update(self, preds, targets):
        return

    def compute(self):
        overall = 0.0
        per_organ = np.zeros(self.num_classes)
        return overall, per_organ




## === cell 68
dice_score_obj = DiceScoreCustom(num_classes=NUM_CLASSES)
hausdorff_obj = HausdorffDistanceCustom(num_classes=NUM_CLASSES)




## === cell 69
def one_epoch_train(epoch):
    model.train()
    running_loss = 0.0

    loop = tqdm(dataloader_train, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for data_batch in loop:
        imgs, masks, ids = data_batch
        imgs = imgs.to(DEVICE, dtype=torch.float, non_blocking=True)
        masks = masks.to(DEVICE, dtype=torch.float, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        pred_masks = model(imgs)

        loss = loss_fn(pred_masks, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(dataloader_train)
    return avg_loss




## === cell 70
slices80_casedays = set(
    data_valid[["case", "day", "slice"]]
    .drop_duplicates()
    .value_counts(["case", "day"])
    .loc[lambda s: s == 80]
    .index
)




## === cell 71
def one_epoch_valid():
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        for data_batch in dataloader_valid:
            imgs, masks, ids = data_batch
            imgs = imgs.to(DEVICE, dtype=torch.float, non_blocking=True)
            masks = masks.to(DEVICE, dtype=torch.float, non_blocking=True)
            pred_masks = model(imgs)
            loss = loss_fn(pred_masks, masks)
            running_loss += loss.item()

            pred_masks_bin = (torch.sigmoid(pred_masks) > 0.5).int()
            masks_bin = masks.int()
            dice_score_obj.update(pred_masks_bin, masks_bin)

        avg_loss = running_loss / len(dataloader_valid)
        epoch_dice_score = dice_score_obj.compute()
        dice_score_obj.reset()
        epoch_hausdorff = hausdorff_obj.compute()
        hausdorff_obj.reset()

    return avg_loss, epoch_dice_score, epoch_hausdorff




## === cell 72
if TRAIN_VALID_SPLIT:
    for epoch in range(EPOCHS):
        loss_train = one_epoch_train(epoch)
        loss_valid, dice_score, hausdorff = one_epoch_valid()
        dice_overall, dice_per_organ = dice_score
        hausdorff_overall, hausdorff_per_organ = hausdorff
        combined_metric = 0.4 * dice_overall + 0.6 * (1 - float(hausdorff_overall))
        print(
            f"Epoch {epoch+1} | "
            f"Train Loss: {loss_train:.3f} | Valid Loss: {loss_valid:.3f} | "
            f"Combined metric: {combined_metric:.3f} | "
            f"Dice: {dice_overall:.3f} (LB {dice_per_organ[0]:.3f}, SB {dice_per_organ[1]:.3f}, S {dice_per_organ[2]:.3f}) | "
            f"Hausdorff: {hausdorff_overall:.3f}"
        )



## --- ERROR in cell 72, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1139777262.py in <cell line: 0>()
      1 if TRAIN_VALID_SPLIT:
      2     for epoch in range(EPOCHS):
----> 3         loss_train = one_epoch_train(epoch)
      4         loss_valid, dice_score, hausdorff = one_epoch_valid()
      5         dice_overall, dice_per_organ = dice_score

/tmp/ipykernel_55/1503873278.py in one_epoch_train(epoch)
      3     running_loss = 0.0
      4 
----> 5     loop = tqdm(dataloader_train, desc=f"Epoch {epoch+1}/{EPOCHS}")
      6     for data_batch in loop:
      7         imgs, masks, ids = data_batch

NameError: name 'dataloader_train' is not defined

## === cell 73
if TRAIN_VALID_SPLIT and SAVE_TRAIN_VALID_MODEL:
    torch.save(model.state_dict(), MODEL_PARAMS_FILE_NAME)
    print("Saved trained model to:", os.path.abspath(MODEL_PARAMS_FILE_NAME))



## === cell 74
if TEST_PREDICT:
    load_path = None
    if os.path.exists(MODEL_PARAMS_FILE_NAME):
        load_path = MODEL_PARAMS_FILE_NAME
    elif LOAD_MODEL_FOR_TEST_PREDICT and os.path.exists(MODEL_PARAMS_LOAD_FILE_PATH):
        load_path = MODEL_PARAMS_LOAD_FILE_PATH

    if load_path is not None:
        state = torch.load(load_path, map_location=DEVICE)
        try:
            model.load_state_dict(state)
        except Exception:
            model.load_state_dict(state, strict=False)
        print(f"Loaded model weights from: {load_path}")
    else:
        print(
            f"No checkpoint found at {MODEL_PARAMS_FILE_NAME} or {MODEL_PARAMS_LOAD_FILE_PATH}.\nProceeding with current model weights."
        )

    model.eval()

    sample_sub = pd.read_csv(DIR_PATH + "sample_submission.csv")
    data_test = sample_sub[["id", "class"]].copy()

    data_test[["case", "day", "slice"]] = data_test["id"].str.extract(
        r"case(\d+)_day(\d+)_slice_(\d+)"
    )

    _test_unique_ids = data_test["id"].drop_duplicates().tolist()
    _test_id_to_path = {}
    _test_id_to_hw = {}
    for _id in tqdm(_test_unique_ids, desc="Mapping test id->image_path"):
        p = _find_slice_png_path(DIR_PATH, False, _id)
        _test_id_to_path[_id] = p
        _test_id_to_hw[_id] = _read_hw_from_png(p)

    data_test["image_path"] = data_test["id"].map(_test_id_to_path)
    data_test["slice_h"] = (
        data_test["id"].map(lambda x: _test_id_to_hw[x][0]).astype(np.uint32)
    )
    data_test["slice_w"] = (
        data_test["id"].map(lambda x: _test_id_to_hw[x][1]).astype(np.uint32)
    )
    data_test["px_w"] = np.float32(0.0)
    data_test["px_h"] = np.float32(0.0)

    int_cols = ["case", "day", "slice", "slice_w", "slice_h"]
    for c in int_cols:
        data_test[c] = pd.to_numeric(data_test[c], errors="coerce")
    data_test[int_cols] = data_test[int_cols].fillna(0).astype(np.uint32)

    float_cols = ["px_w", "px_h"]
    for c in float_cols:
        data_test[c] = pd.to_numeric(data_test[c], errors="coerce")
    data_test[float_cols] = data_test[float_cols].fillna(0).astype(np.float32)

    if data_test["image_path"].isna().any():
        missing = data_test.loc[data_test["image_path"].isna(), "id"].head(5).tolist()
        raise RuntimeError(
            f"Missing image_path after merge for some test ids, e.g.: {missing}"
        )

    if ALBU_AVAILABLE:
        transform_test = A.Compose(
            [
                A.Resize(
                    IMAGE_RESIZE[0],
                    IMAGE_RESIZE[1],
                    interpolation=cv2.INTER_LINEAR if CV2_AVAILABLE else 1,
                ),
                A.Normalize(
                    mean=IMAGE_NORMALIZE_MEAN,
                    std=IMAGE_NORMALIZE_SD,
                    max_pixel_value=1.0,
                ),
                ToTensorV2(transpose_mask=False),
            ]
        )
    else:
        transform_test = _ComposeFallback(
            resize_hw=IMAGE_RESIZE,
            mean=IMAGE_NORMALIZE_MEAN,
            std=IMAGE_NORMALIZE_SD,
            max_pixel_value=1.0,
            transpose_mask=False,
        )

    data_test_unique = data_test.drop_duplicates("id").reset_index(drop=True)

    dataset_test = GITractDataset(
        data_test_unique, is_train=False, transforms=transform_test
    )
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=BATCH_SIZE_TEST,
        shuffle=False,
        num_workers=DATA_LOADER_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(DATA_LOADER_NUM_WORKERS > 0),
        prefetch_factor=(
            DATA_LOADER_PREFETCH_FACTOR if DATA_LOADER_NUM_WORKERS > 0 else None
        ),
        worker_init_fn=_seed_worker,
        generator=_DATALOADER_GENERATOR,
    )



## --- ERROR in cell 74, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2335819405.py in <cell line: 0>()
     32     _test_id_to_hw = {}
     33     for _id in tqdm(_test_unique_ids, desc="Mapping test id->image_path"):
---> 34         p = _find_slice_png_path(DIR_PATH, False, _id)
     35         _test_id_to_path[_id] = p
     36         _test_id_to_hw[_id] = _read_hw_from_png(p)

/tmp/ipykernel_55/837926081.py in _find_slice_png_path(base_dir, train, id_str)
     13 
     14 def _find_slice_png_path(base_dir: str, train: bool, id_str: str) -> str:
---> 15     case, day, slice_idx = _id_to_case_day_slice(id_str)
     16     scans_dir = os.path.join(
     17         base_dir,

/tmp/ipykernel_55/837926081.py in _id_to_case_day_slice(id_str)
      5 def _id_to_case_day_slice(id_str: str) -> tuple[int, int, int]:
      6     # id format: case{case}_day{day}_slice_{slice}
----> 7     a, b, c = id_str.split("_")
      8     case = int(a[4:])
      9     day = int(b[3:])

ValueError: too many values to unpack (expected 3)

## === cell 75
if TEST_PREDICT:
    out_rows = []
    with torch.no_grad():
        for imgs, ids, heights, widths in tqdm(dataloader_test, desc="Test inference"):
            imgs = imgs.to(DEVICE, dtype=torch.float, non_blocking=True)
            pred_masks = model(imgs)
            pred_masks = (torch.sigmoid(pred_masks) > 0.5).to(torch.uint8)
            pred_masks = pred_masks.permute(0, 2, 3, 1).cpu().numpy()  # (B,288,288,3)

            for mask288, id_, h, w in zip(pred_masks, ids, heights, widths):
                h = int(h)
                w = int(w)
                mask_orig_size = _resize_nearest(mask288, (h, w)).astype(np.uint8)
                out_rows.append(
                    (id_, CLASS_NAMES[0], rle_encode(mask_orig_size[..., 0]))
                )
                out_rows.append(
                    (id_, CLASS_NAMES[1], rle_encode(mask_orig_size[..., 1]))
                )
                out_rows.append(
                    (id_, CLASS_NAMES[2], rle_encode(mask_orig_size[..., 2]))
                )

    pred_df = pd.DataFrame(out_rows, columns=["id", "class", "predicted"])

    out = sample_sub[["id", "class"]].merge(
        pred_df, on=["id", "class"], how="left", sort=False
    )
    out["predicted"] = out["predicted"].fillna("")

    out = (
        out[["id", "class", "predicted"]]
        .sort_values(["id", "class"])
        .reset_index(drop=True)
    )

    out.to_csv("submission.csv", index=False)

    print(out.head())
    print("Saved submission.csv with shape:", out.shape)
    print("Saved to:", os.path.abspath("submission.csv"))
    assert list(out.columns) == ["id", "class", "predicted"]
    assert len(out) == 20400

## --- ERROR in cell 75, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4217635283.py in <cell line: 0>()
      3     out_rows = []
      4     with torch.no_grad():
----> 5         for imgs, ids, heights, widths in tqdm(dataloader_test, desc="Test inference"):
      6             imgs = imgs.to(DEVICE, dtype=torch.float, non_blocking=True)
      7             pred_masks = model(imgs)

NameError: name 'dataloader_test' is not defined
