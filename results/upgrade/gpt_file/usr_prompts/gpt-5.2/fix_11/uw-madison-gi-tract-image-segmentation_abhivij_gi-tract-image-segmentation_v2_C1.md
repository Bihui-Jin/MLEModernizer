# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

albumentations==2.0.8
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1

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

# 5. Code solution

## === cell 0
import os, sys, warnings

warnings.filterwarnings("ignore")
os.environ.setdefault(
    "CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "")
)

import numpy as np
import pandas as pd

import random
import re

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap

import seaborn as sns

from glob import glob

from tqdm.auto import tqdm

from sklearn.model_selection import StratifiedGroupKFold

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2


def _has_module(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


SMP_AVAILABLE = _has_module("segmentation_models_pytorch")
MONAI_AVAILABLE = _has_module("monai")




## === cell 1
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetSmall(nn.Module):
    """
    Minimal UNet-like network with logits output of shape (B, classes, H, W).
    This is only used because segmentation_models_pytorch isn't installed.
    """

    def __init__(self, in_channels=3, classes=3, base=32):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base * 4, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.out = nn.Conv2d(base, classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)


class DiceLossMultilabel(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        dims = (0, 2, 3)
        intersection = (probs * targets).sum(dims)
        cardinality = probs.sum(dims) + targets.sum(dims)
        dice = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        return 1.0 - dice.mean()




## === cell 2
print(f"Number of available CPUs: {os.cpu_count()}")
print(f"Number of available GPUs: {torch.cuda.device_count()}")



## === cell 3
DIR_CANDIDATES = [
    "/kaggle/input/uw-madison-gi-tract-image-segmentation/",
    "/kaggle/data/uw-madison-gi-tract-image-segmentation/",
    "/kaggle/input/",
]
DIR_PATH = None
for c in DIR_CANDIDATES:
    if os.path.exists(os.path.join(c, "train.csv")):
        DIR_PATH = c if c.endswith("/") else c + "/"
        break
if DIR_PATH is None:
    raise FileNotFoundError(
        "Could not locate dataset directory containing train.csv in known Kaggle paths."
    )



## === cell 4
pd.set_option("display.max_colwidth", 400)

CMAP1 = ListedColormap([[0, 0, 0, 0], [1, 0, 0, 1]])  # black transparent, red opaque
CMAP2 = ListedColormap([[0, 0, 0, 0], [0, 1, 0, 1]])  # black transparent, green opaque
CMAP3 = ListedColormap([[0, 0, 0, 0], [0, 0, 1, 1]])  # black transparent, blue opaque

RANDOM_SEED = 0

IMAGE_NORMALIZE_MEAN = (0.485, 0.456, 0.406)
IMAGE_NORMALIZE_SD = (0.229, 0.224, 0.225)

IMAGE_RESIZE = [288, 288]

BATCH_SIZE_TRAIN = 32
BATCH_SIZE_VALID = BATCH_SIZE_TRAIN * 2
BATCH_SIZE_TEST = BATCH_SIZE_TRAIN * 2

DATA_LOADER_NUM_WORKERS = 4

NUM_CLASSES = 3
CLASS_NAMES = ["large_bowel", "small_bowel", "stomach"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 5

MODEL_PARAMS_FILE_NAME = "GIT-Seg-efficientnet-b1-subset.pth"
MODEL_PARAMS_LOAD_FILE_PATH = (
    "/kaggle/input/git-seg/pytorch/default/1/GIT-Seg-efficientnet-b1-subset.pth"
)

TRAIN_VALID_SPLIT = False
TEST_PREDICT = True

SAVE_TRAIN_VALID_MODEL = False
LOAD_MODEL_FOR_TEST_PREDICT = True



## === cell 5
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass



## === cell 6
if LOAD_MODEL_FOR_TEST_PREDICT and not os.path.exists(MODEL_PARAMS_LOAD_FILE_PATH):
    print(
        f"Warning: MODEL_PARAMS_LOAD_FILE_PATH not found: {MODEL_PARAMS_LOAD_FILE_PATH}"
    )
    print(
        "Proceeding without loading external weights (will use randomly initialized model)."
    )
    LOAD_MODEL_FOR_TEST_PREDICT = False



## === cell 7
data = pd.read_csv(DIR_PATH + "train.csv")
data.head()



## === cell 8
data_nonnaseg = data.loc[data.segmentation.notna(), :]
data_nonnaseg.head()



## === cell 9
data[["case", "day", "slice"]] = data["id"].str.extract(
    r"case(\d+)_day(\d+)_slice_(\d+)"
)
data




## === cell 10
def _parse_scan_filename_fast(name: str):
    if not name.endswith(".png") or not name.startswith("slice_"):
        return None
    base = name[:-4]
    parts = base.split("_")
    if len(parts) != 6:
        return None
    return parts[1], parts[2], parts[3], parts[4], parts[5]


def _path_cache_file(train: bool) -> str:
    base = "train" if train else "test"
    return os.path.join("/kaggle/working", f"_scan_index_cache_{base}.parquet")


def get_path_df(train=True):
    cache_fp = _path_cache_file(train)
    if os.path.exists(cache_fp):
        try:
            return pd.read_parquet(cache_fp)
        except Exception:
            pass  # fall back to rebuild

    base = "train" if train else "test"
    root = os.path.join(DIR_PATH, base)
    if not os.path.exists(root):
        return pd.DataFrame(
            columns=[
                "image_path",
                "case",
                "day",
                "slice",
                "slice_w",
                "slice_h",
                "px_w",
                "px_h",
            ]
        )

    pattern = os.path.join(root, "case*", "case*_day*", "scans", "slice_*.png")
    files = glob(pattern)
    rows = []
    for fp in files:
        fn = os.path.basename(fp)
        parsed = _parse_scan_filename_fast(fn)
        if parsed is None:
            continue
        slice_, slice_w, slice_h, px_w, px_h = parsed
        scans_dir = os.path.dirname(fp)  # .../scans
        day_dir = os.path.dirname(scans_dir)
        case_dir = os.path.dirname(day_dir)
        case_name = os.path.basename(case_dir)
        day_name = os.path.basename(day_dir)
        if not case_name.startswith("case") or "_day" not in day_name:
            continue
        case_num = case_name[4:]
        day_num = day_name.split("_day", 1)[1]
        rows.append((fp, case_num, day_num, slice_, slice_w, slice_h, px_w, px_h))

    df = pd.DataFrame(
        rows,
        columns=[
            "image_path",
            "case",
            "day",
            "slice",
            "slice_w",
            "slice_h",
            "px_w",
            "px_h",
        ],
    )
    try:
        df.to_parquet(cache_fp, index=False)
    except Exception:
        pass
    return df


path_df = get_path_df(train=True)
path_df.head()



## === cell 11
data.info()



## === cell 12
path_df.info()



## === cell 13
if len(path_df) == 0:
    raise RuntimeError(
        "No scan image paths found. Check DIR_PATH and get_path_df glob pattern."
    )



## === cell 14
data = data.merge(path_df, on=["case", "day", "slice"])
data



## === cell 15
data.info()



## === cell 16
data.px_w.unique(), data.px_h.unique()



## === cell 17
data.case.unique(), data.day.unique(), data.slice.unique(), data.slice_w.unique(), data.slice_h.unique()



## === cell 18
int_cols = ["case", "day", "slice", "slice_w", "slice_h"]
data[int_cols] = data[int_cols].astype(np.uint32)

float_cols = ["px_w", "px_h"]
data[float_cols] = data[float_cols].astype(np.float32)

data.info()



## === cell 19
clahe3 = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2, 2))



## === cell 20
_rle_decode_cache = {}
_RLE_DECODE_CACHE_MAX = 20000  # bounded to avoid unbounded memory growth


def rle_decode(mask_rle, shape):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    if (
        mask_rle is None
        or mask_rle == ""
        or (isinstance(mask_rle, float) and np.isnan(mask_rle))
    ):
        return np.zeros(shape, dtype=np.uint8)

    key = (mask_rle, int(shape[0]), int(shape[1]))
    cached = _rle_decode_cache.get(key)
    if cached is not None:
        return cached

    s = np.fromstring(mask_rle, sep=" ", dtype=np.int64)
    if s.size == 0:
        out = np.zeros(shape, dtype=np.uint8)
        if len(_rle_decode_cache) < _RLE_DECODE_CACHE_MAX:
            _rle_decode_cache[key] = out
        return out

    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    diff = np.zeros(img.size + 1, dtype=np.int32)
    np.add.at(diff, starts, 1)
    np.add.at(diff, ends, -1)
    img[:] = (np.cumsum(diff[:-1]) > 0).astype(np.uint8)
    out = img.reshape(shape)

    if len(_rle_decode_cache) < _RLE_DECODE_CACHE_MAX:
        _rle_decode_cache[key] = out
    return out


def rle_encode(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels = np.asarray(img, dtype=np.uint8).reshape(-1)
    if pixels.size == 0:
        return ""
    p = np.empty(pixels.size + 2, dtype=np.uint8)
    p[0] = 0
    p[-1] = 0
    p[1:-1] = pixels
    changes = np.flatnonzero(p[1:] != p[:-1]) + 1
    if changes.size == 0:
        return ""
    runs = changes.copy()
    runs[1::2] -= runs[::2]
    return " ".join(runs.astype(str))


def rle_encode_fast(img2d: np.ndarray) -> str:
    pixels = np.asarray(img2d, dtype=np.uint8).reshape(-1)
    if pixels.size == 0:
        return ""
    p = np.empty(pixels.size + 2, dtype=np.uint8)
    p[0] = 0
    p[-1] = 0
    p[1:-1] = pixels
    idx = np.flatnonzero(p[1:] != p[:-1]) + 1
    if idx.size == 0:
        return ""
    runs = idx.copy()
    runs[1::2] -= runs[::2]
    return " ".join(map(str, runs.tolist()))




## === cell 21
from functools import lru_cache


@lru_cache(maxsize=4096)
def _imread_grayscale_16bit_cached(path: str):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(
            f"Failed to read image (cv2 returned None). Path: {path}"
        )
    return img


def imread_grayscale_16bit(path: str):
    return _imread_grayscale_16bit_cached(path)




## === cell 22
def get_mask(id_, data):
    data_subset_id = data.loc[data["id"] == id_]
    slice_dim = data_subset_id[["slice_h", "slice_w"]].iloc[0]
    shape = (int(slice_dim.slice_h), int(slice_dim.slice_w), 3)
    mask = np.zeros(shape, dtype=np.uint8)
    for i, class_ in enumerate(CLASS_NAMES):
        data_subset_class = data_subset_id[data_subset_id["class"] == class_]
        rle = data_subset_class.segmentation.squeeze()
        if not pd.isna(rle):
            mask[..., i] = rle_decode(rle, shape[:2])
    return mask




## === cell 23
pass



## === cell 24
pass




## === cell 25
def get_existing_id(case=None, day=None):
    if case is None:
        return data.id.iloc[0]
    q = data.query("case == @case and day == @day")
    if len(q) == 0:
        return data.id.iloc[0]
    return q.id.iloc[0]




## === cell 26
id_example = get_existing_id(case=int(data.case.iloc[0]), day=int(data.day.iloc[0]))
id_example




## === cell 27
def load_image(id_, data):
    data_subset = data.loc[data.id == id_]
    if len(data_subset) == 0:
        raise KeyError(f"id not found in dataframe: {id_}")
    path = data_subset.image_path.iloc[0]
    img = imread_grayscale_16bit(path).astype("float32")
    mx = float(img.max())
    if mx > 0:
        img /= mx
    return img




## === cell 28
def display_image(
    id_,
    data,
    pred_mask=None,
    apply_CLAHE=False,
    show_orig_img=True,
    show_true_mask=True,
    show_pred_mask=False,
):
    img = load_image(id_, data)
    img = (img * 255).astype(np.uint8)
    if apply_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2, 2))
        img = clahe.apply(img)

    mask = get_mask(id_, data)

    plt.figure(figsize=(9, 3))

    i = 1
    if show_orig_img:
        plt.subplot(1, 3, i)
        i += 1
        plt.imshow(img, cmap="bone")
        plt.title(f"{id_} image")
        plt.axis("off")

    if show_true_mask:
        plt.subplot(1, 3, i)
        i += 1
        plt.imshow(img, cmap="bone")
        plt.title("Image with true mask")
        plt.imshow(mask[..., 0], cmap=CMAP1)
        plt.imshow(mask[..., 1], cmap=CMAP2)
        plt.imshow(mask[..., 2], cmap=CMAP3)

        handles = [
            Rectangle((0, 0), 1, 1, color=CMAP1(1.0)),
            Rectangle((0, 0), 1, 1, color=CMAP2(1.0)),
            Rectangle((0, 0), 1, 1, color=CMAP3(1.0)),
        ]
        labels = ["Large Bowel", "Small Bowel", "Stomach"]
        plt.axis("off")
        plt.legend(
            handles,
            labels,
            bbox_to_anchor=(1.0, -0.4),
            loc="lower right",
            borderaxespad=0.0,
        )

    if show_pred_mask and pred_mask is not None:
        plt.subplot(1, 3, i)
        plt.imshow(img, cmap="bone")
        plt.title("Image with predicted mask")
        plt.imshow(pred_mask[..., 0], cmap=CMAP1)
        plt.imshow(pred_mask[..., 1], cmap=CMAP2)
        plt.imshow(pred_mask[..., 2], cmap=CMAP3)

        handles = [
            Rectangle((0, 0), 1, 1, color=CMAP1(1.0)),
            Rectangle((0, 0), 1, 1, color=CMAP2(1.0)),
            Rectangle((0, 0), 1, 1, color=CMAP3(1.0)),
        ]
        labels = ["Large Bowel", "Small Bowel", "Stomach"]
        plt.axis("off")
        plt.legend(
            handles,
            labels,
            bbox_to_anchor=(1.0, -0.4),
            loc="lower right",
            borderaxespad=0.0,
        )

    plt.tight_layout()
    plt.show()




## === cell 29
pass



## === cell 30
pass




## === cell 31
def display_multiple_slices(
    id_array, data, apply_CLAHE=False, show_pred_mask=False, pred_mask_array=None
):
    l = len(id_array)
    rows = np.ceil(l / 5).astype(int)
    max_cols = 5
    data_subset = data.loc[data.id.isin(id_array),].copy()

    plt.figure(figsize=(max_cols * 3, rows * 3))

    id_to_path = (
        data_subset.drop_duplicates("id").set_index("id")["image_path"].to_dict()
    )

    for i in range(l):
        id_ = id_array[i]
        path = id_to_path.get(id_, None)
        if path is None:
            continue

        img = imread_grayscale_16bit(path).astype("float32")
        mx = float(img.max())
        if mx > 0:
            img /= mx

        if apply_CLAHE:
            img_u8 = (img * 255).astype(np.uint8)
            img_u8 = clahe3.apply(img_u8)
            img_show = img_u8
        else:
            img_show = img

        if show_pred_mask and pred_mask_array is not None:
            mask = pred_mask_array[i]
        else:
            mask = get_mask(id_, data)

        plt.subplot(rows, max_cols, i + 1)
        plt.imshow(img_show, cmap="bone")
        plt.title(id_)
        plt.imshow(mask[..., 0], cmap=CMAP1)
        plt.imshow(mask[..., 1], cmap=CMAP2)
        plt.imshow(mask[..., 2], cmap=CMAP3)
        plt.axis("off")

        if i == 0:
            handles = [
                Rectangle((0, 0), 1, 1, color=CMAP1(1.0)),
                Rectangle((0, 0), 1, 1, color=CMAP2(1.0)),
                Rectangle((0, 0), 1, 1, color=CMAP3(1.0)),
            ]
            labels = ["Large Bowel", "Small Bowel", "Stomach"]
            plt.legend(
                handles,
                labels,
                bbox_to_anchor=(0.0, 1.5),
                loc="upper left",
                borderaxespad=0.0,
            )

    plt.tight_layout()
    plt.show()




## === cell 32
pass



## === cell 33
data.loc[data.segmentation.isna(), :].head()



## === cell 34
data.isna().sum()



## === cell 35
print(
    f"Num cases : {len(data.case.unique())}         Num unique days : {len(data.day.unique())}   \
        Num unique slices : {len(data.slice.unique())}"
)



## === cell 36
count_df = (
    data[["id", "slice_w", "slice_h"]]
    .drop_duplicates()[["slice_w", "slice_h"]]
    .value_counts()
    .reset_index(name="count")
)
count_df["percent"] = count_df["count"] * 100 / sum(count_df["count"])
print(sum(count_df["count"]))
count_df.head()



## === cell 37
count_df = (
    data[["id", "px_w", "px_h"]]
    .drop_duplicates()[["px_w", "px_h"]]
    .value_counts()
    .reset_index(name="count")
)
count_df["percent"] = count_df["count"] * 100 / sum(count_df["count"])
print(sum(count_df["count"]))
count_df.head()



## === cell 38
day_dist = (
    data[["case", "day"]]
    .drop_duplicates()["case"]
    .value_counts()
    .reset_index(name="num_days")
)
display(day_dist.head())



## === cell 39
slice_dist = (
    data[["case", "day", "slice"]]
    .drop_duplicates()[["case", "day"]]
    .value_counts()
    .reset_index(name="num_slices")
)
display(slice_dist.head())



## === cell 40
num_missing_seg_masks = data.segmentation.isna().sum()
print(
    f"Missing Seg Mask \n count = {num_missing_seg_masks}\n percentage = {num_missing_seg_masks/len(data)*100:.2f}%"
)



## === cell 41
data["class"].value_counts()



## === cell 42
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
index_train, index_valid = next(
    sgkf.split(data.id, data.segmentation.isna(), data.case)
)



## === cell 43
len(index_train), len(index_valid)



## === cell 44
data_train = data.iloc[index_train, :]
data_valid = data.iloc[index_valid, :]



## === cell 45
print(len(data_train.case.unique()), len(data_valid.case.unique()))



## === cell 46
data_train_sub = data_train.loc[data_train.case.isin(data_train.case.unique()[:11]), :]
data_valid_sub = data_valid.loc[data_valid.case.isin(data_valid.case.unique()[:2]), :]

print(
    len(data_train_sub), len(data_valid_sub), len(data_train_sub) / len(data_valid_sub)
)



## === cell 47
missing_masks_train = data_train_sub.segmentation.isna().sum()
missing_masks_valid = data_valid_sub.segmentation.isna().sum()
print(missing_masks_train, missing_masks_train * 100 / len(data_train_sub))
print(missing_masks_valid, missing_masks_valid * 100 / len(data_valid_sub))



## === cell 48
data_train_sub = data_train_sub.reset_index(drop=True)
data_valid_sub = data_valid_sub.reset_index(drop=True)




## === cell 49
def build_id_index(df: pd.DataFrame, with_masks: bool):
    meta = df.drop_duplicates("id")[["id", "image_path", "slice_h", "slice_w"]].copy()
    id_to_meta = meta.set_index("id")[["image_path", "slice_h", "slice_w"]].to_dict(
        "index"
    )

    if not with_masks:
        return id_to_meta, None

    tmp = df[["id", "class", "segmentation"]].copy()
    tmp["class"] = pd.Categorical(tmp["class"], categories=CLASS_NAMES, ordered=True)
    wide = (
        tmp.drop_duplicates(["id", "class"])
        .set_index(["id", "class"])["segmentation"]
        .unstack("class")
    )
    wide = wide.reindex(columns=CLASS_NAMES)
    id_to_rles = {idx: tuple(row.values.tolist()) for idx, row in wide.iterrows()}
    return id_to_meta, id_to_rles


def build_id_to_mask_cache(id_to_meta, id_to_rles):
    id_to_mask = {}
    _rle_decode = rle_decode
    for id_, meta in id_to_meta.items():
        h = int(meta["slice_h"])
        w = int(meta["slice_w"])
        mask = np.zeros((h, w, NUM_CLASSES), dtype=np.uint8)
        rles = id_to_rles.get(id_, (np.nan, np.nan, np.nan))
        for i in range(NUM_CLASSES):
            rle = rles[i]
            if not pd.isna(rle):
                mask[..., i] = _rle_decode(rle, (h, w))
        id_to_mask[id_] = mask
    return id_to_mask




## === cell 50
class GITractDataset(Dataset):
    def __init__(self, df, is_train=True, transforms=None):
        self.df = df
        self.id_ = df["id"].unique()
        self.is_train = is_train
        self.transforms = transforms

        self.id_to_meta, self.id_to_rles = build_id_index(df, with_masks=is_train)

        self.id_to_mask = None
        if is_train:
            self.id_to_mask = build_id_to_mask_cache(self.id_to_meta, self.id_to_rles)

    def __len__(self):
        return len(self.id_)

    def __getitem__(self, idx):
        id_ = self.id_[idx]
        meta = self.id_to_meta[id_]
        path = meta["image_path"]
        h = int(meta["slice_h"])
        w = int(meta["slice_w"])

        img = imread_grayscale_16bit(path).astype("float32")
        mx = float(img.max())
        if mx > 0:
            img /= mx
        img = np.repeat(img[..., None], 3, axis=2)

        if self.is_train:
            mask = self.id_to_mask[id_]  # uint8 HWC, cached
            if self.transforms:
                augmented = self.transforms(image=img, mask=mask)
                img = augmented["image"]
                mask = augmented["mask"]
            return img, mask, id_
        else:
            if self.transforms:
                augmented = self.transforms(image=img)
                img = augmented["image"]
            return img, id_, h, w




## === cell 51
transform_train = A.Compose(
    [
        A.Resize(
            IMAGE_RESIZE[0],
            IMAGE_RESIZE[1],
            interpolation=cv2.INTER_NEAREST,
            mask_interpolation=cv2.INTER_NEAREST,
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
            interpolation=cv2.INTER_NEAREST,
            mask_interpolation=cv2.INTER_NEAREST,
        ),
        A.Normalize(
            mean=IMAGE_NORMALIZE_MEAN, std=IMAGE_NORMALIZE_SD, max_pixel_value=1.0
        ),
        ToTensorV2(transpose_mask=True),
    ]
)



## === cell 52
dataset_train = GITractDataset(data_train_sub, transforms=transform_train)
dataset_valid = GITractDataset(data_valid_sub, transforms=transform_valid)

_cpu = os.cpu_count() or 4
_DATA_LOADER_NUM_WORKERS_EFF = max(DATA_LOADER_NUM_WORKERS, min(8, max(2, _cpu // 2)))

_common_loader_kwargs = dict(
    num_workers=_DATA_LOADER_NUM_WORKERS_EFF,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_DATA_LOADER_NUM_WORKERS_EFF > 0),
    prefetch_factor=8 if _DATA_LOADER_NUM_WORKERS_EFF > 0 else None,
)

dataloader_train = DataLoader(
    dataset_train,
    batch_size=BATCH_SIZE_TRAIN,
    shuffle=True,
    **{k: v for k, v in _common_loader_kwargs.items() if v is not None},
)
dataloader_valid = DataLoader(
    dataset_valid,
    batch_size=BATCH_SIZE_VALID,
    shuffle=False,
    **{k: v for k, v in _common_loader_kwargs.items() if v is not None},
)



## === cell 53
dataset = next(iter(dataloader_train))
img, mask, id_ = dataset
print(img.shape, mask.shape, len(id_))



## === cell 54
if SMP_AVAILABLE:
    import segmentation_models_pytorch as smp  # noqa

    smp_encoder_weights = (
        None if TEST_PREDICT and LOAD_MODEL_FOR_TEST_PREDICT else "imagenet"
    )
    model = smp.Unet(
        encoder_name="efficientnet-b1",
        encoder_weights=smp_encoder_weights,
        in_channels=3,
        classes=NUM_CLASSES,
    )
else:
    model = UNetSmall(in_channels=3, classes=NUM_CLASSES, base=32)

model.to(DEVICE)

if torch.cuda.is_available():
    model = model.to(memory_format=torch.channels_last)



## === cell 55
optimizer = optim.Adam(model.parameters(), lr=1e-3)



## === cell 56
if SMP_AVAILABLE:
    dice_loss = smp.losses.DiceLoss(mode="multilabel")
    BCE_loss = smp.losses.SoftBCEWithLogitsLoss()
else:
    dice_loss = DiceLossMultilabel()
    BCE_loss = nn.BCEWithLogitsLoss()


def loss_fn(y_pred, y_true, loss_wt=0.5):
    return dice_loss(y_pred, y_true) * loss_wt + BCE_loss(y_pred, y_true) * (
        1 - loss_wt
    )




## === cell 57
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
        non_empty = U > 0  # [B, C]
        organ_counts = non_empty.sum(dim=1)  # [B]
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




## === cell 58
class HausdorffDistanceCustom:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.h3d_sum = 0.0
        self.image3d_count = 0
        self.organ_h3d_sum = np.zeros(self.num_classes, dtype=np.float32)
        self.organ_count_sum = np.zeros(self.num_classes, dtype=np.float32)

    def update(self, preds, targets):
        U = (targets | preds).sum((1, 2, 3))
        non_empty = U > 0
        if np.all(preds == targets):
            hausdorff = np.zeros(self.num_classes, dtype=np.float32)
        else:
            hausdorff = np.ones(self.num_classes, dtype=np.float32)
        organ_count = non_empty.sum()
        if organ_count != 0:
            hausdorff_per_3dimage = hausdorff[non_empty].mean()
            self.h3d_sum += float(hausdorff_per_3dimage)
            self.image3d_count += 1
        self.organ_h3d_sum += hausdorff
        self.organ_count_sum += non_empty.astype(np.float32)

    def compute(self):
        overall = (self.h3d_sum / self.image3d_count) if self.image3d_count > 0 else 0.0
        per_organ = np.divide(
            self.organ_h3d_sum, np.maximum(self.organ_count_sum, 1.0), dtype=np.float32
        )
        return overall, per_organ




## === cell 59
dice_score_obj = DiceScoreCustom(num_classes=NUM_CLASSES)
hausdorff_obj = HausdorffDistanceCustom(num_classes=NUM_CLASSES)




## === cell 60
def one_epoch_train(epoch):
    model.train()
    running_loss = 0.0

    loop = tqdm(dataloader_train, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
    for batch in loop:
        imgs, masks, ids = batch
        if torch.cuda.is_available():
            imgs = imgs.to(DEVICE, dtype=torch.float, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )
            masks = masks.to(DEVICE, dtype=torch.float, non_blocking=True)
        else:
            imgs, masks = imgs.to(DEVICE, dtype=torch.float), masks.to(
                DEVICE, dtype=torch.float
            )

        optimizer.zero_grad(set_to_none=True)
        pred_masks = model(imgs)
        loss = loss_fn(pred_masks, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=float(loss.item()))

    avg_loss = running_loss / len(dataloader_train)
    return avg_loss




## === cell 61
slices80_casedays = set(
    data_valid[["case", "day", "slice"]]
    .drop_duplicates()
    .value_counts(["case", "day"])
    .loc[lambda s: s == 80]
    .index
)



## === cell 62
_id_triplet_cache = {}
_triplet_re = re.compile(r"case(\d+)_day(\d+)_slice_(\d+)")


def parse_id_triplet(id_):
    t = _id_triplet_cache.get(id_)
    if t is not None:
        return t
    m = _triplet_re.match(id_)
    if not m:
        _id_triplet_cache[id_] = None
        return None
    t = tuple(map(int, m.groups()))
    _id_triplet_cache[id_] = t
    return t


def one_epoch_valid():
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        pred_masks_dict, masks_dict = {}, {}
        for batch in dataloader_valid:
            imgs, masks, ids = batch
            if torch.cuda.is_available():
                imgs = imgs.to(DEVICE, dtype=torch.float, non_blocking=True).contiguous(
                    memory_format=torch.channels_last
                )
                masks = masks.to(DEVICE, dtype=torch.float, non_blocking=True)
            else:
                imgs, masks = imgs.to(DEVICE, dtype=torch.float), masks.to(
                    DEVICE, dtype=torch.float
                )

            pred_masks = model(imgs)
            loss = loss_fn(pred_masks, masks)
            running_loss += loss.item()

            pred_bin = (torch.sigmoid(pred_masks) > 0.5).int()
            masks_bin = masks.int()
            dice_score_obj.update(pred_bin, masks_bin)

            for p, m, id_ in zip(pred_bin, masks_bin, ids):
                triplet = parse_id_triplet(id_)
                if triplet is None:
                    continue
                caseid, dayid, sliceid = triplet
                casedayid = (caseid, dayid)
                pred_masks_dict.setdefault(casedayid, []).append((sliceid, p))
                masks_dict.setdefault(casedayid, []).append((sliceid, m))

                if (len(pred_masks_dict[casedayid]) == 144) or (
                    casedayid in slices80_casedays
                    and len(pred_masks_dict[casedayid]) == 80
                ):
                    pred_sorted = [
                        pp.cpu().numpy()
                        for sid, pp in sorted(
                            pred_masks_dict[casedayid], key=lambda x: x[0]
                        )
                    ]
                    mask_sorted = [
                        mm.cpu().numpy()
                        for sid, mm in sorted(masks_dict[casedayid], key=lambda x: x[0])
                    ]
                    pred_vol = np.stack(pred_sorted, axis=1)
                    mask_vol = np.stack(mask_sorted, axis=1)
                    hausdorff_obj.update(pred_vol, mask_vol)
                    del pred_masks_dict[casedayid], masks_dict[casedayid]

        avg_loss = running_loss / len(dataloader_valid)
        epoch_dice_score = dice_score_obj.compute()
        dice_score_obj.reset()
        epoch_hausdorff = hausdorff_obj.compute()
        hausdorff_obj.reset()
    return avg_loss, epoch_dice_score, epoch_hausdorff




## === cell 63
if TRAIN_VALID_SPLIT:
    for epoch in range(EPOCHS):
        loss_train = one_epoch_train(epoch)
        loss_valid, dice_score, hausdorff = one_epoch_valid()
        dice_overall, dice_per_organ = dice_score
        hausdorff_overall, hausdorff_per_organ = hausdorff
        combined_metric = 0.4 * float(dice_overall) + 0.6 * (
            1 - float(hausdorff_overall)
        )
        print(
            f"Epoch {epoch+1} | "
            f"Train Loss: {loss_train:.3f} | Valid Loss: {loss_valid:.3f} | "
            f"Combined metric: {combined_metric:.3f} | "
            f"Dice: {float(dice_overall):.3f} (LB {float(dice_per_organ[0]):.3f}, SB {float(dice_per_organ[1]):.3f}, S {float(dice_per_organ[2]):.3f}) | "
            f"Hausdorff: {float(hausdorff_overall):.3f} (LB {float(hausdorff_per_organ[0]):.3f}, SB {float(hausdorff_per_organ[1]):.3f}, S {float(hausdorff_per_organ[2]):.3f})"
        )



## === cell 64
if TRAIN_VALID_SPLIT and SAVE_TRAIN_VALID_MODEL:
    torch.save(model.state_dict(), MODEL_PARAMS_FILE_NAME)



## === cell 65
if TEST_PREDICT:
    if LOAD_MODEL_FOR_TEST_PREDICT:
        state = torch.load(MODEL_PARAMS_LOAD_FILE_PATH, map_location=DEVICE)
        try:
            model.load_state_dict(state, strict=True)
        except Exception:
            if isinstance(state, dict) and "state_dict" in state:
                model.load_state_dict(state["state_dict"], strict=False)
            else:
                model.load_state_dict(state, strict=False)
    model.eval()

    data_test = pd.read_csv(DIR_PATH + "sample_submission.csv")
    test_set_hidden = not bool(len(data_test))
    if test_set_hidden:
        data_test = data_valid_sub.copy()
    else:
        data_test[["case", "day", "slice"]] = data_test["id"].str.extract(
            r"case(\d+)_day(\d+)_slice_(\d+)"
        )
        path_df_test = get_path_df(train=False)
        if len(path_df_test) == 0:
            raise RuntimeError(
                "No test scan image paths found. Check dataset extraction and get_path_df(train=False)."
            )

        data_test = data_test.merge(
            path_df_test, on=["case", "day", "slice"], how="left"
        )
        if data_test.image_path.isna().any():
            missing = data_test.loc[data_test.image_path.isna(), "id"].iloc[0]
            raise RuntimeError(
                f"Some test rows could not be matched to image paths. Example missing id: {missing}"
            )

        int_cols = ["case", "day", "slice", "slice_w", "slice_h"]
        data_test[int_cols] = data_test[int_cols].astype(np.uint32)
        float_cols = ["px_w", "px_h"]
        data_test[float_cols] = data_test[float_cols].astype(np.float32)

    transform_test = A.Compose(
        [
            A.Resize(IMAGE_RESIZE[0], IMAGE_RESIZE[1], interpolation=cv2.INTER_NEAREST),
            A.Normalize(
                mean=IMAGE_NORMALIZE_MEAN, std=IMAGE_NORMALIZE_SD, max_pixel_value=1.0
            ),
            ToTensorV2(transpose_mask=False),
        ]
    )
    dataset_test = GITractDataset(data_test, is_train=False, transforms=transform_test)
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=BATCH_SIZE_TEST,
        shuffle=False,
        **{k: v for k, v in _common_loader_kwargs.items() if v is not None},
    )



## === cell 66
if TEST_PREDICT:
    test_ids, test_class, test_pred_RLE = [], [], []

    with torch.inference_mode():
        _resize = cv2.resize
        _interp = cv2.INTER_NEAREST
        _rle = rle_encode_fast

        for imgs, ids, heights, widths in tqdm(dataloader_test, desc="Test inference"):
            if torch.cuda.is_available():
                imgs = imgs.to(DEVICE, dtype=torch.float, non_blocking=True).contiguous(
                    memory_format=torch.channels_last
                )
            else:
                imgs = imgs.to(DEVICE, dtype=torch.float)

            pred_masks = model(imgs)
            pred_masks = (torch.sigmoid(pred_masks) > 0.5).to(torch.uint8)  # [B,C,H,W]
            pm = pred_masks.cpu().numpy()  # uint8 [B,C,H,W]

            heights_np = np.asarray(heights, dtype=np.int32)
            widths_np = np.asarray(widths, dtype=np.int32)

            sizes = np.stack([heights_np, widths_np], axis=1)
            uniq, inv = np.unique(sizes, axis=0, return_inverse=True)

            for gi, (h, w) in enumerate(uniq.tolist()):
                idxs = np.flatnonzero(inv == gi)
                if idxs.size == 0:
                    continue

                for j in idxs.tolist():
                    id_ = ids[j]
                    pm_hwc = np.transpose(
                        pm[j], (1, 2, 0)
                    )  # [H,W,C] at resized resolution
                    pm_resized = _resize(
                        pm_hwc, dsize=(int(w), int(h)), interpolation=_interp
                    )
                    if pm_resized.ndim == 2:
                        pm_resized = pm_resized[..., None]

                    rles0 = _rle(pm_resized[..., 0])
                    rles1 = _rle(pm_resized[..., 1])
                    rles2 = _rle(pm_resized[..., 2])

                    test_ids.extend([id_] * NUM_CLASSES)
                    test_class.extend(CLASS_NAMES)
                    test_pred_RLE.extend([rles0, rles1, rles2])

    submission_df = pd.DataFrame(
        {"id": test_ids, "class": test_class, "predicted": test_pred_RLE}
    )
    submission_df.to_csv("submission.csv", index=False)

    print("Wrote submission.csv")
    print(submission_df.head())
    assert submission_df.shape[1] == 3
    assert set(submission_df.columns) == {"id", "class", "predicted"}
    assert (
        submission_df.shape[0]
        == pd.read_csv(DIR_PATH + "sample_submission.csv").shape[0]
    )
    print("submission.csv looks valid (row count matches sample_submission).")
