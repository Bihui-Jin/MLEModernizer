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
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

# 5. Target score

0.5833560557254097

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'The merge error happens because `pred_df` contains duplicate `(id, class)` rows: your inference dataset is built from unique slices, but the submission requires one row per `(id, class)` and many different slices share the same `id`. I fix this by generating predictions per slice and then aggregating (union) masks across all slices for each `id` and class before RLE encoding, so keys become unique and the submission matches the competition format. I also ensure missing/empty predictions are encoded as an empty string and keep the model/inference logic intact. These changes are necessary for a valid submission and should improve score vs. the current (non-submittable) pipeline.'

# 9. Code solution

## === cell 0
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser



## === cell 1
import os, sys, subprocess, textwrap, pathlib



## === cell 2
import warnings

warnings.filterwarnings("ignore")

import gc
import time
import copy
from glob import glob
from collections import defaultdict
from typing import List, Tuple, Dict

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import albumentations as A

from tqdm.auto import tqdm

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
tqdm.pandas()




## === cell 3
class CFG:
    seed = 42
    debug = False
    model_name = "UNET"
    encoder_name = "resnet50"
    encoder_weights = "imagenet"
    train_batch_size = 32
    val_batch_size = 32
    img_size = (224, 224)
    scheduler = "CosineAnnealingLR"
    epochs = 22
    lr = 2e-3
    min_lr = 1e-6
    weight_decay = 1e-6
    T_max = int(30000 / train_batch_size * epochs) + 50
    num_classes = 3
    val_split_percentage = 0.2
    n_accumulate = max(1, 32 // train_batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    thr = 0.45




## === cell 4
TRAIN_METADATA_FILE = "../input/uw-madison-gi-tract-image-segmentation/train.csv"
TRAIN_DIR = "../input/uw-madison-gi-tract-image-segmentation/train/"
BASE_PATH = "../input/uw-madison-gi-tract-image-segmentation"
CKPT_DIR = "../input/resnet50best"
SAMPLE_SUBMISSION_CSV_PATH = (
    "../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv"
)
TRAIN_FILE_CSV_PATH = TRAIN_METADATA_FILE
TEST_CSV_PATH = "../input/uw-madison-gi-tract-image-segmentation/test.csv"




## === cell 5
def seed_everything(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(CFG.seed)


def get_metadata(df):
    df = df.copy()
    df["case"] = df["id"].apply(lambda x: int(x.split("_")[0].split("case")[1]))
    df["day"] = df["id"].apply(lambda x: int(x.split("_")[1].split("day")[1]))
    df["slice"] = df["id"].apply(lambda x: int(x.split("_")[-1]))
    return df


def path2info(row):
    """
    Robustly parse scan filenames and skip non-image/stray entries.
    Expected scan filename format:
      {slice}_{width}_{height}_{spacingx}_{spacingy}.png
    Example:
      86_266_266_1.63_1.63.png
    """
    path = row["image_path"]
    fname = os.path.basename(path)
    if not fname.lower().endswith(".png"):
        row["height"] = np.nan
        row["width"] = np.nan
        row["case"] = np.nan
        row["day"] = np.nan
        row["slice"] = np.nan
        return row

    stem = fname[:-4]
    toks = stem.split("_")
    if len(toks) < 3:
        row["height"] = np.nan
        row["width"] = np.nan
        row["case"] = np.nan
        row["day"] = np.nan
        row["slice"] = np.nan
        return row

    try:
        slice_ = int(toks[0])
        width = int(toks[1])
        height = int(toks[2])
    except Exception:
        row["height"] = np.nan
        row["width"] = np.nan
        row["case"] = np.nan
        row["day"] = np.nan
        row["slice"] = np.nan
        return row

    parts = path.split(os.sep)
    case_day = None
    for p in reversed(parts):
        if p.startswith("case") and "_day" in p:
            case_day = p
            break
    if case_day is None:
        row["height"] = np.nan
        row["width"] = np.nan
        row["case"] = np.nan
        row["day"] = np.nan
        row["slice"] = np.nan
        return row

    case = int(case_day.split("_")[0].replace("case", ""))
    day = int(case_day.split("_")[1].replace("day", ""))

    row["height"] = height
    row["width"] = width
    row["case"] = case
    row["day"] = day
    row["slice"] = slice_
    return row




## === cell 6
def load_image(path):
    return Image.open(path).convert("RGB")


def load_msk(path):
    msk = np.load(path)
    msk = msk.astype("float32")
    msk /= 255.0
    return msk


def show_img(img, mask=None):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    plt.imshow(img, cmap="bone")
    if mask is not None:
        plt.imshow(mask, alpha=0.5)
        handles = [
            Rectangle((0, 0), 1, 1, color=_c)
            for _c in [(0.667, 0.0, 0.0), (0.0, 0.667, 0.0), (0.0, 0.0, 0.667)]
        ]
        labels = ["Large Bowel", "Small Bowel", "Stomach"]
        plt.legend(handles, labels)
    plt.axis("off")




## === cell 7
test_pairs = pd.read_csv(TEST_CSV_PATH)
sub_template = pd.read_csv(SAMPLE_SUBMISSION_CSV_PATH)

test_pairs["id"] = test_pairs["id"].astype(str)
test_pairs["class"] = test_pairs["class"].astype(str)
sub_template["id"] = sub_template["id"].astype(str)
sub_template["class"] = sub_template["class"].astype(str)

sub_df = get_metadata(test_pairs.drop_duplicates().copy())
sub_df.head()



## === cell 8
paths = glob(os.path.join(BASE_PATH, "test", "**", "scans", "*.png"), recursive=True)
path_df = pd.DataFrame(paths, columns=["image_path"])
path_df = path_df.progress_apply(path2info, axis=1)
path_df = path_df.dropna(subset=["case", "day", "slice"]).copy()
path_df[["case", "day", "slice", "height", "width"]] = path_df[
    ["case", "day", "slice", "height", "width"]
].astype(int)
path_df.head()



## === cell 9
test_df = sub_df.merge(path_df, on=["case", "day", "slice"], how="left")
missing = int(test_df["image_path"].isna().sum())
if missing:
    print(f"Warning: {missing} rows missing image_path after merge.")
test_df.head()




## === cell 10
class TestDataset(Dataset):
    def __init__(self, df, transforms=None, label=None):
        self.df = df
        self.label = label
        self.img_paths = df["image_path"].tolist()
        self.ids = df["id"].tolist()
        self.transforms = transforms
        self.heights = df.get("height", pd.Series([np.nan] * len(df))).tolist()
        self.widths = df.get("width", pd.Series([np.nan] * len(df))).tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        id_ = self.ids[index]

        if not isinstance(img_path, str) or not os.path.exists(img_path):
            h = self.heights[index]
            w = self.widths[index]
            if not (
                isinstance(h, (int, np.integer)) and isinstance(w, (int, np.integer))
            ):
                h, w = CFG.img_size
            dummy = np.zeros((CFG.img_size[0], CFG.img_size[1], 3), dtype=np.uint8)
            if self.transforms:
                data = self.transforms(image=dummy)
                dummy = data["image"]
            dummy = np.transpose(dummy, (2, 0, 1))
            return (
                torch.tensor(np.ascontiguousarray(dummy), dtype=torch.float32),
                id_,
                int(h),
                int(w),
            )

        img = load_image(img_path)
        img = np.array(img)
        h, w = img.shape[:2]

        if self.transforms:
            data = self.transforms(image=img)
            img = data["image"]
        img = np.transpose(img, (2, 0, 1))
        return (
            torch.tensor(np.ascontiguousarray(img), dtype=torch.float32),
            id_,
            int(h),
            int(w),
        )




## === cell 11
test_transforms = {
    "test": A.Compose(
        [
            A.Resize(CFG.img_size[0], CFG.img_size[1], interpolation=cv2.INTER_LINEAR),
        ],
        p=1.0,
    )
}




## === cell 12
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


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        if diffX != 0 or diffY != 0:
            x1 = F.pad(
                x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
            )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, classes=3, base=32):
        super().__init__()
        self.inc = DoubleConv(in_channels, base)
        self.down1 = Down(base, base * 2)
        self.down2 = Down(base * 2, base * 4)
        self.down3 = Down(base * 4, base * 8)
        self.down4 = Down(base * 8, base * 8)

        self.up1 = Up(base * 16, base * 4)
        self.up2 = Up(base * 8, base * 2)
        self.up3 = Up(base * 4, base)
        self.up4 = Up(base * 2, base)
        self.outc = nn.Conv2d(base, classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


def build_model():
    return SimpleUNet(in_channels=3, classes=CFG.num_classes, base=32)


def load_model(path):
    model = build_model()
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    cleaned = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        cleaned[nk] = v
    model.load_state_dict(cleaned, strict=False)
    model.eval()
    return model




## === cell 13
def mask2rle(msk):
    msk = np.array(msk, dtype=np.uint8)
    pixels = msk.flatten(order="F")  # column-major for this competition
    pad = np.array([0], dtype=np.uint8)
    pixels = np.concatenate([pad, pixels, pad])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def masks2rles(msks, ids, heights, widths):
    pred_strings = []
    pred_ids = []
    pred_classes = []

    for idx in range(msks.shape[0]):
        height = int(heights[idx])
        width = int(widths[idx])

        m = msks[idx].astype(np.uint8)  # Hs,Ws,3
        if m.shape[0] != height or m.shape[1] != width:
            m = cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST)

        rle = [None] * 3
        for midx in [0, 1, 2]:
            rle[midx] = mask2rle(m[..., midx])
        pred_strings.extend(rle)
        pred_ids.extend([ids[idx]] * 3)
        pred_classes.extend(["large_bowel", "small_bowel", "stomach"])
    return pred_strings, pred_ids, pred_classes




## === cell 14
def _find_model_paths(ckpt_dir: str):
    candidates = []

    patterns = []
    if ckpt_dir and os.path.isdir(ckpt_dir):
        patterns += [
            os.path.join(ckpt_dir, "*.bin"),
            os.path.join(ckpt_dir, "*.pth"),
            os.path.join(ckpt_dir, "*.pt"),
        ]

    patterns += [
        os.path.join("/kaggle/input", "**", "*.bin"),
        os.path.join("/kaggle/input", "**", "*.pth"),
        os.path.join("/kaggle/input", "**", "*.pt"),
    ]

    for pat in patterns:
        candidates.extend(glob(pat, recursive=True))

    best = [p for p in candidates if "best" in os.path.basename(p).lower()]
    chosen = sorted(best) if len(best) else sorted(candidates)
    return chosen[:5]


@torch.no_grad()
def infer_and_aggregate_to_submission(
    model_paths: List[str], test_loader: DataLoader, thr: float = CFG.thr
):
    agg: Dict[str, Dict[str, np.ndarray]] = {}
    agg_hw: Dict[str, Tuple[int, int]] = {}  # (h,w) per id

    class_names = ["large_bowel", "small_bowel", "stomach"]
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    if not model_paths:
        return agg, agg_hw

    models = []
    for path in model_paths:
        m = load_model(path).to(CFG.device)
        models.append(m)

    for img, ids, heights, widths in tqdm(
        test_loader, total=len(test_loader), desc="Infer+Aggregate"
    ):
        img = img.to(CFG.device, dtype=torch.float32)

        size = img.size()
        msk = torch.zeros(
            (size[0], 3, size[2], size[3]), device=CFG.device, dtype=torch.float32
        )

        for model in models:
            out = model(img)
            out = torch.sigmoid(out)
            msk += out / len(models)

        msk = (msk.permute((0, 2, 3, 1)) > thr).to(torch.uint8).cpu().numpy()

        bs = msk.shape[0]
        for i in range(bs):
            id_ = str(ids[i])
            h = int(heights[i])
            w = int(widths[i])

            mi = msk[i]  # (Hs,Ws,3)
            if mi.shape[0] != h or mi.shape[1] != w:
                mi = cv2.resize(mi, (w, h), interpolation=cv2.INTER_NEAREST)

            if id_ not in agg:
                agg[id_] = {c: np.zeros((h, w), dtype=np.uint8) for c in class_names}
                agg_hw[id_] = (h, w)
            else:
                hh, ww = agg_hw[id_]
                if (h, w) != (hh, ww):
                    mi = cv2.resize(mi, (ww, hh), interpolation=cv2.INTER_NEAREST)
                    h, w = hh, ww

            for c in class_names:
                ci = class_to_idx[c]
                agg[id_][c] = np.maximum(agg[id_][c], mi[..., ci])

        del img, msk, out
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return agg, agg_hw




## === cell 15
test_dataset = TestDataset(test_df, transforms=test_transforms["test"])
test_loader = DataLoader(
    test_dataset,
    batch_size=CFG.val_batch_size,
    num_workers=2,
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
)

model_paths = _find_model_paths(CKPT_DIR)
print(f"Found {len(model_paths)} checkpoints (showing up to 5 used):")
for p in model_paths[:5]:
    print(" -", p)

agg_masks, agg_hw = infer_and_aggregate_to_submission(
    model_paths, test_loader, thr=CFG.thr
)



## === cell 16
if len(agg_masks):
    first_id = next(iter(agg_masks.keys()))
    m0 = np.stack(
        [
            agg_masks[first_id]["large_bowel"],
            agg_masks[first_id]["small_bowel"],
            agg_masks[first_id]["stomach"],
        ],
        axis=-1,
    )
    plt.figure(figsize=(10, 4))
    plt.imshow(m0.sum(axis=-1), cmap="gray")
    plt.title(f"Aggregated mask sum (id={first_id})")
    plt.axis("off")
    plt.show()



## === cell 17
class_names = ["large_bowel", "small_bowel", "stomach"]

pred_rows = []
for id_, cls in zip(sub_template["id"].tolist(), sub_template["class"].tolist()):
    id_ = str(id_)
    cls = str(cls)
    if id_ in agg_masks and cls in agg_masks[id_]:
        rle = mask2rle(agg_masks[id_][cls])
        pred_rows.append((id_, cls, rle))
    else:
        pred_rows.append((id_, cls, ""))

sub_out = pd.DataFrame(pred_rows, columns=["id", "class", "predicted"])
sub_out.to_csv("submission.csv", index=False)

print(sub_out.head())
print("Saved submission.csv with shape:", sub_out.shape)
print("Empty predicted rows:", int((sub_out["predicted"] == "").sum()))
print("submission.csv path:", os.path.abspath("submission.csv"))
