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

0.5711150869012114

# 6. Current score

0.24608

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.24608) has done: 'I fix the path parsing so it correctly extracts `case/day/slice` and `width/height` from the actual UW-Madison folder and filename conventions (the current code crashes on filenames like `slice_0001_...png`). Then I rebuild `path_df` deterministically by using the numeric `slice_####` in filenames, ensuring `sub_df` merges to image paths without missing rows. I also fix a small dataset bug where `msk_path` vs `mask_path` mismatched, and make the RLE padding logic robust to any off-by-one differences after cropping. These changes are execution-stability focused and should allow the notebook to run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.24608) has done: 'Your current score is far below the target (0.24608 vs 0.5711; higher is better), and the most likely cause is a submission-encoding mismatch rather than the model itself. I make two minimal, metric-aligned fixes: (1) generate RLE in the competition’s required pixel order (column-major / Fortran order) instead of row-major flattening, and (2) correctly “uncrop” the predicted mask back to the original image size by padding or center-cropping as needed (your current code only pads, which can misalign masks when the original is smaller than the crop). These changes preserve the model and inference logic, but fix decoding/geometry so predictions land on the right pixels, which should move the score substantially toward the target. The output remains a valid `submission.csv` with the required columns and row count.'
- What this solution (achieved 0.24608) has done: 'Your current score is far below the target (0.24608 vs 0.5711; higher is better), so we should improve the geometric/encoding correctness rather than change the model. The biggest remaining mismatch is that the notebook uses `CenterCrop(224,224)` at inference but then “restores” to the original size by center padding/cropping, which is not the inverse of a center-crop when the original image is larger—this shifts masks into the wrong location. I make a minimal change to keep the model input exactly the same (224×224), but additionally return the crop coordinates (y0/x0) from the dataset and use them to paste predictions back into the correct location in the full-resolution canvas before RLE encoding. I keep the RLE Fortran-order encoding and submission formatting intact, and still write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser



## === cell 1
import os
import gc
import time
import copy
import warnings
import re
from glob import glob
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import pandas as pd
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import albumentations as A
import torchvision

warnings.filterwarnings("ignore")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

try:
    from pandarallel import pandarallel  # type: ignore

    pandarallel.initialize(progress_bar=True)
except Exception:
    pandarallel = None

try:
    tqdm.pandas()
except Exception:
    pass


def df_apply_with_progress(df, func, axis=1, desc="apply"):
    """Robust apply with optional progress bar."""
    try:
        return df.progress_apply(func, axis=axis)
    except Exception:
        return df.apply(func, axis=axis)


class _ConvRelu(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = _ConvRelu(in_ch + skip_ch, out_ch)
        self.conv2 = _ConvRelu(out_ch, out_ch)

    def forward(self, x, skip):
        x = nn.functional.interpolate(
            x, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SimpleResNet50UNet(nn.Module):
    def __init__(self, classes=3):
        super().__init__()
        enc = torchvision.models.resnet50(weights=None)

        self.layer0 = nn.Sequential(enc.conv1, enc.bn1, enc.relu)  # /2
        self.pool = enc.maxpool  # /4
        self.layer1 = enc.layer1  # /4
        self.layer2 = enc.layer2  # /8
        self.layer3 = enc.layer3  # /16
        self.layer4 = enc.layer4  # /32

        self.center = nn.Sequential(_ConvRelu(2048, 512), _ConvRelu(512, 512))

        self.up4 = _UpBlock(512, 1024, 256)
        self.up3 = _UpBlock(256, 512, 128)
        self.up2 = _UpBlock(128, 256, 64)
        self.up1 = _UpBlock(64, 64, 32)

        self.final = nn.Conv2d(32, classes, kernel_size=1)

    def forward(self, x):
        x0 = self.layer0(x)  # 64, /2
        x1 = self.pool(x0)  # 64, /4
        x1 = self.layer1(x1)  # 256, /4
        x2 = self.layer2(x1)  # 512, /8
        x3 = self.layer3(x2)  # 1024, /16
        x4 = self.layer4(x3)  # 2048, /32

        c = self.center(x4)  # 512, /32
        d4 = self.up4(c, x3)  # 256, /16
        d3 = self.up3(d4, x2)  # 128, /8
        d2 = self.up2(d3, x1)  # 64, /4
        d1 = self.up1(d2, x0)  # 32, /2

        out = nn.functional.interpolate(
            d1, scale_factor=2.0, mode="bilinear", align_corners=False
        )  # back to input
        out = self.final(out)
        return out




## === cell 2
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


def seed_everything(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


seed_everything(CFG.seed)



## === cell 3
BASE_PATH = "/kaggle/input/uw-madison-gi-tract-image-segmentation"
TRAIN_METADATA_FILE = f"{BASE_PATH}/train.csv"
TRAIN_DIR = f"{BASE_PATH}/train/"
SAMPLE_SUBMISSION_CSV_PATH = f"{BASE_PATH}/sample_submission.csv"
TRAIN_FILE_CSV_PATH = TRAIN_METADATA_FILE

CKPT_DIR = "/kaggle/input/bestepoch152nditer"




## === cell 4
def get_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["case"] = df["id"].apply(lambda x: int(x.split("_")[0].split("case")[1]))
    df["day"] = df["id"].apply(lambda x: int(x.split("_")[1].split("day")[1]))
    df["slice"] = df["id"].apply(lambda x: int(x.split("_")[-1]))
    return df


def path2info(row: pd.Series) -> pd.Series:
    path = row["image_path"]
    parts = path.split("/")
    fname = os.path.basename(path).replace(".png", "")

    case_day = parts[-3]  # ".../caseXXX_dayYY/scans/<file>"
    if "_day" not in case_day:
        raise ValueError(
            f"Unexpected folder format (cannot find case_day): {case_day} for path={path}"
        )
    case_str, day_str = case_day.split("_")
    case = int(case_str.replace("case", ""))
    day = int(day_str.replace("day", ""))

    tokens = fname.split("_")
    if len(tokens) < 4:
        raise ValueError(f"Unexpected filename format: {os.path.basename(path)}")

    if tokens[0] == "slice" and tokens[1].isdigit():
        slice_idx = int(tokens[1])
        width = int(tokens[2])
        height = int(tokens[3])
    else:
        m = re.search(r"slice_(\d+)", fname)
        slice_idx = int(m.group(1)) if m else np.nan
        ints = [int(t) for t in tokens if t.isdigit()]
        if len(ints) >= 2:
            width, height = ints[0], ints[1]
        else:
            raise ValueError(
                f"Cannot parse width/height from filename: {os.path.basename(path)}"
            )

    row["height"] = int(height)
    row["width"] = int(width)
    row["case"] = int(case)
    row["day"] = int(day)
    row["slice"] = int(slice_idx) if not pd.isna(slice_idx) else np.nan
    return row




## === cell 5
def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_msk(path: str) -> np.ndarray:
    msk = np.load(path)
    msk = msk.astype("float32")
    msk /= 255.0
    return msk


def show_img(img, mask=None):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if img.ndim == 3:
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        g = img
    g = clahe.apply(g)
    plt.imshow(g, cmap="bone")

    if mask is not None:
        plt.imshow(mask, alpha=0.5)
        handles = [
            Rectangle((0, 0), 1, 1, color=_c)
            for _c in [(0.667, 0.0, 0.0), (0.0, 0.667, 0.0), (0.0, 0.0, 0.667)]
        ]
        labels = ["Large Bowel", "Small Bowel", "Stomach"]
        plt.legend(handles, labels)
    plt.axis("off")




## === cell 6
sub_df = pd.read_csv(SAMPLE_SUBMISSION_CSV_PATH)
if not len(sub_df):
    debug = True
    sub_df = pd.read_csv(TRAIN_FILE_CSV_PATH)[: 1000 * 3]
    sub_df = sub_df.drop(columns=["class", "segmentation"]).drop_duplicates()
else:
    debug = False
    sub_df = sub_df.drop(columns=["class", "predicted"]).drop_duplicates()

sub_df = get_metadata(sub_df)



## === cell 7
if debug:
    paths = glob(BASE_PATH + "/train/**/*png", recursive=True)
else:
    paths = glob(BASE_PATH + "/test/**/*png", recursive=True)

path_df = pd.DataFrame(paths, columns=["image_path"])
path_df = df_apply_with_progress(path_df, path2info, axis=1, desc="parse_paths")

path_df = path_df.dropna(subset=["case", "day", "slice", "width", "height"]).copy()
path_df["case"] = path_df["case"].astype(int)
path_df["day"] = path_df["day"].astype(int)
path_df["slice"] = path_df["slice"].astype(int)
path_df["fname"] = path_df["image_path"].apply(lambda p: os.path.basename(p))

path_df = path_df.sort_values(["case", "day", "slice"]).reset_index(drop=True)

required_cols = ["case", "day", "slice", "width", "height", "image_path"]
missing = [c for c in required_cols if c not in path_df.columns]
if missing:
    raise RuntimeError(f"path_df is missing columns after parsing: {missing}")

path_df.head()



## === cell 8
test_df = sub_df.merge(path_df, on=["case", "day", "slice"], how="left")

if test_df["image_path"].isna().any():
    n_missing = int(test_df["image_path"].isna().sum())
    sample_missing = test_df.loc[test_df["image_path"].isna(), "id"].head(5).tolist()
    raise RuntimeError(
        f"Failed to match {n_missing} rows to image paths. Sample missing ids: {sample_missing}. "
        f"Check slice mapping between ids and scan filenames."
    )

test_df.head()




## === cell 9
class TestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transforms=None, label=None):
        self.df = df.reset_index(drop=True)
        self.label = label
        self.img_paths = self.df["image_path"].tolist()
        self.ids = self.df["id"].tolist()
        if "mask_path" in self.df.columns:
            self.msk_paths = self.df["mask_path"].tolist()
        else:
            self.msk_paths = None
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        id_ = self.ids[index]

        img = load_image(img_path)
        img = np.array(img)
        h, w = img.shape[:2]

        ch, cw = CFG.img_size
        y0 = max(0, (h - ch) // 2)
        x0 = max(0, (w - cw) // 2)

        if self.label:
            msk_path = self.msk_paths[index]
            msk = load_msk(msk_path)
            if self.transforms:
                data = self.transforms(image=img, mask=msk)
                img = data["image"]
                msk = data["mask"]
            img = np.transpose(img, (2, 0, 1))
            msk = np.transpose(msk, (2, 0, 1))
            return torch.tensor(img), torch.tensor(msk)
        else:
            if self.transforms:
                data = self.transforms(image=img)
                img = data["image"]
            img = np.transpose(img, (2, 0, 1))
            return (
                torch.tensor(img),
                id_,
                torch.tensor(h),
                torch.tensor(w),
                torch.tensor(y0),
                torch.tensor(x0),
            )




## === cell 10
test_transforms = {
    "test": A.Compose(
        [
            A.CenterCrop(*CFG.img_size),
        ],
        p=1.0,
    )
}




## === cell 11
def build_model():
    model = SimpleResNet50UNet(classes=CFG.num_classes)
    return model


def load_model(path: str):
    model = build_model()
    state = torch.load(path, map_location="cpu")

    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]

    cleaned = {}
    for k, v in state.items():
        nk = k
        for prefix in ("module.", "model."):
            if nk.startswith(prefix):
                nk = nk[len(prefix) :]
        cleaned[nk] = v

    model.load_state_dict(cleaned, strict=False)
    model.eval()
    return model




## === cell 12
def mask2rle(msk, thr=0.5):
    msk = np.asarray(msk)
    msk = (msk > 0).astype(np.uint8)
    pixels = msk.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def _paste_crop_to_canvas(
    crop_hw3: np.ndarray, height: int, width: int, y0: int, x0: int
) -> np.ndarray:
    ch, cw = crop_hw3.shape[:2]
    canvas = np.zeros((height, width, 3), dtype=crop_hw3.dtype)
    y1 = min(height, y0 + ch)
    x1 = min(width, x0 + cw)
    canvas[y0:y1, x0:x1, :] = crop_hw3[: (y1 - y0), : (x1 - x0), :]
    return canvas


def masks2rles(msks, ids, heights, widths, y0s, x0s):
    pred_strings = []
    pred_ids = []
    pred_classes = []
    for idx in range(msks.shape[0]):
        height = (
            int(heights[idx].item())
            if torch.is_tensor(heights[idx])
            else int(heights[idx])
        )
        width = (
            int(widths[idx].item())
            if torch.is_tensor(widths[idx])
            else int(widths[idx])
        )
        y0 = int(y0s[idx].item()) if torch.is_tensor(y0s[idx]) else int(y0s[idx])
        x0 = int(x0s[idx].item()) if torch.is_tensor(x0s[idx]) else int(x0s[idx])

        msk_restored = _paste_crop_to_canvas(msks[idx], height, width, y0, x0)

        rle = [None] * 3
        for midx in [0, 1, 2]:
            rle[midx] = mask2rle(msk_restored[..., midx])
        pred_strings.extend(rle)
        pred_ids.extend([ids[idx]] * len(rle))
        pred_classes.extend(["large_bowel", "small_bowel", "stomach"])
    return pred_strings, pred_ids, pred_classes




## === cell 13
@torch.no_grad()
def infer(model_paths, test_loader, num_log=1, thr=CFG.thr):
    msks = []
    imgs = []
    pred_strings = []
    pred_ids = []
    pred_classes = []

    if len(model_paths) == 0:
        for idx, (img, ids, heights, widths, y0s, x0s) in enumerate(
            tqdm(test_loader, total=len(test_loader), desc="Infer (no ckpt)")
        ):
            bs = img.shape[0]
            msk = np.zeros((bs, CFG.img_size[0], CFG.img_size[1], 3), dtype=np.uint8)
            result = masks2rles(msk, ids, heights, widths, y0s, x0s)
            pred_strings.extend(result[0])
            pred_ids.extend(result[1])
            pred_classes.extend(result[2])
        return pred_strings, pred_ids, pred_classes, imgs, msks

    models = []
    for path in model_paths:
        m = load_model(path).to(CFG.device)
        m.eval()
        models.append(m)

    for idx, (img, ids, heights, widths, y0s, x0s) in enumerate(
        tqdm(test_loader, total=len(test_loader), desc="Infer")
    ):
        img = img.to(CFG.device, dtype=torch.float)

        size = img.size()
        msk = torch.zeros(
            (size[0], 3, size[2], size[3]), device=CFG.device, dtype=torch.float32
        )

        for model in models:
            out = model(img)
            out = torch.sigmoid(out)
            msk += out / len(models)

        msk = (msk.permute((0, 2, 3, 1)) > thr).to(torch.uint8).cpu().numpy()
        result = masks2rles(msk, ids, heights, widths, y0s, x0s)
        pred_strings.extend(result[0])
        pred_ids.extend(result[1])
        pred_classes.extend(result[2])

        if idx < num_log:
            img_np = img.permute((0, 2, 3, 1)).cpu().numpy()
            imgs.append(img_np[:10])
            msks.append(msk[:10])

        del img, msk, result
        gc.collect()
        torch.cuda.empty_cache()

    for m in models:
        del m
    gc.collect()
    torch.cuda.empty_cache()

    return pred_strings, pred_ids, pred_classes, imgs, msks




## === cell 14
test_dataset = TestDataset(test_df, transforms=test_transforms["test"])
test_loader = DataLoader(
    test_dataset,
    batch_size=CFG.val_batch_size,
    num_workers=2,
    shuffle=False,
    pin_memory=False,
)

model_paths = glob(f"{CKPT_DIR}/best_epoch*.bin")

if len(model_paths) == 0:
    model_paths = sorted(glob("/kaggle/input/**/best_epoch*.bin", recursive=True))[:5]

pred_strings, pred_ids, pred_classes, imgs, msks = infer(model_paths, test_loader)



## === cell 15
if "imgs" in globals() and "msks" in globals() and len(imgs) > 0 and len(msks) > 0:
    for img, msk in zip(imgs[0][:2], msks[0][:2]):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(img, cmap="bone")
        plt.axis("OFF")
        plt.title("image")

        plt.subplot(1, 3, 2)
        plt.imshow(msk * 255)
        plt.axis("OFF")
        plt.title("mask")

        plt.subplot(1, 3, 3)
        plt.imshow(img, cmap="bone")
        plt.imshow(msk * 255, alpha=0.4)
        plt.axis("OFF")
        plt.title("overlay")

        plt.tight_layout()
        plt.show()



## === cell 16
try:
    del imgs, msks
except Exception:
    pass
gc.collect()



## === cell 17
pred_df = pd.DataFrame(
    {"id": pred_ids, "class": pred_classes, "predicted": pred_strings}
)

sub_template = pd.read_csv(SAMPLE_SUBMISSION_CSV_PATH)
sub = sub_template.merge(pred_df, on=["id", "class"], how="left", suffixes=("", "_y"))
sub["predicted"] = sub["predicted"].fillna("")
sub = sub[["id", "class", "predicted"]]

sub.to_csv("submission.csv", index=False)

print(sub.head(5))
print("Wrote submission.csv with shape:", sub.shape)
print("Num empty preds:", int((sub["predicted"] == "").sum()))
print("Used ckpts:", model_paths[:5], "..." if len(model_paths) > 5 else "")
