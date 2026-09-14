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

0.3229766746089432

# 6. Current score

0.12231

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on `segmentation_models_pytorch` (it isn’t installed) by replacing it with a minimal U-Net implementation that preserves the same “U-Net with ResNet34-style encoder” intent and keeps the rest of the pipeline unchanged. I also fix the path parsing bug in `path2info` by robustly extracting width/height from the *last two numeric tokens* in the filename, which avoids crashing on unexpected prefixes like `slice_...png`. Finally, I ensure `path_df` contains `case/day/slice/height/width/image_path` so the merge builds `test_df` correctly, and I make submission writing unconditional so a valid `submission.csv` is always produced end-to-end.'
- What this solution (achieved 0.12231) has done: 'Your 0.0 score is most consistent with “empty masks everywhere” (common when no checkpoint files are found, so the code averages over an empty model list and leaves `msk` as all zeros). I keep your core pipeline intact, but make two minimal changes that legitimately move the score upward toward the target: (1) ensure `CKPT_DIR` points to the actual competition dataset folder (so weights can be found if they exist there), and (2) if no weights are found, fall back to a deterministic, simple non-empty prediction based on image intensity inside the same center-crop (so the submission is not all-empty). I also fix a small but critical RLE bug: your `mask2rle` ignores the `thr` argument and never binarizes, so I apply thresholding explicitly to guarantee valid binary masks.'

# 9. Code solution

## === cell 0
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser



## === cell 1
import os
import gc
import time
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

import albumentations as A
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"




## === cell 2
class CFG:
    seed = 42
    debug = False
    model_name = "UNET"
    encoder_name = "resnet34"
    encoder_weights = "imagenet"
    train_batch_size = 32
    val_batch_size = 32
    img_size = (224, 224)
    scheduler = "CosineAnnealingLR"
    epochs = 22
    lr = 2e-3
    min_lr = 1e-6
    weight_decay = 1e-6
    T_max = int(30000 / train_batch_size * epochs) + 50  # max iterations for scheduler
    num_classes = 3
    val_split_percentage = 0.2
    n_accumulate = max(1, 32 // train_batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    thr = 0.45


torch.manual_seed(CFG.seed)
np.random.seed(CFG.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CFG.seed)



## === cell 3
TRAIN_METADATA_FILE = "../input/uw-madison-gi-tract-image-segmentation/train.csv"
BASE_PATH = "../input/uw-madison-gi-tract-image-segmentation"
CKPT_DIR = "../input/uw-madison-gi-tract-image-segmentation"
SAMPLE_SUBMISSION_CSV_PATH = (
    "../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv"
)
TRAIN_FILE_CSV_PATH = TRAIN_METADATA_FILE

if not exists(SAMPLE_SUBMISSION_CSV_PATH):
    BASE_PATH = "/kaggle/input/uw-madison-gi-tract-image-segmentation"
    TRAIN_METADATA_FILE = join(BASE_PATH, "train.csv")
    SAMPLE_SUBMISSION_CSV_PATH = join(BASE_PATH, "sample_submission.csv")
    TRAIN_FILE_CSV_PATH = TRAIN_METADATA_FILE
    CKPT_DIR = BASE_PATH  # same logic for non-relative Kaggle paths

if exists("../input/weights"):
    CKPT_DIR_FALLBACK = "../input/weights"
elif exists("/kaggle/input/weights"):
    CKPT_DIR_FALLBACK = "/kaggle/input/weights"
else:
    CKPT_DIR_FALLBACK = None




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
    fname = parts[-1]
    case_day_dir = parts[-3]  # 'case123_day45'
    case_dir = parts[-4]  # 'case123'

    case = int(case_dir.replace("case", ""))
    day = int(case_day_dir.split("_")[1].replace("day", ""))

    stem = os.path.splitext(fname)[0]
    tokens = stem.split("_")

    ints = []
    for t in tokens:
        try:
            ints.append(int(t))
        except Exception:
            continue

    if len(ints) >= 2:
        width, height = ints[-2], ints[-1]
    else:
        img = Image.open(path)
        width, height = img.size

    row["height"] = int(height)
    row["width"] = int(width)
    row["case"] = int(case)
    row["day"] = int(day)
    row["slice"] = -1
    return row




## === cell 5
def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


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

CFG.debug = debug



## === cell 7
if CFG.debug:
    paths = glob(BASE_PATH + "/train/**/*png", recursive=True)
else:
    paths = glob(BASE_PATH + "/test/**/*png", recursive=True)

path_df = pd.DataFrame(paths, columns=["image_path"])
path_df = path_df.apply(path2info, axis=1)

path_df["fname"] = path_df["image_path"].apply(lambda p: os.path.basename(p))
path_df = path_df.sort_values(["case", "day", "fname"]).reset_index(drop=True)
path_df["slice"] = path_df.groupby(["case", "day"]).cumcount()

path_df = path_df.drop(columns=["fname"])
path_df.head()



## === cell 8
test_df = sub_df.merge(path_df, on=["case", "day", "slice"], how="left")
missing = int(test_df["image_path"].isna().sum())
if missing:
    test_df = test_df.dropna(subset=["image_path"]).reset_index(drop=True)

for c in ["height", "width"]:
    if c not in test_df.columns or test_df[c].isna().any():
        tmp = test_df["image_path"].apply(lambda p: Image.open(p).size)
        test_df["width"] = [s[0] for s in tmp]
        test_df["height"] = [s[1] for s in tmp]

test_df.head()




## === cell 9
class TestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transforms=None, label=None):
        self.df = df
        self.label = label
        self.img_paths = df["image_path"].tolist()
        self.ids = df["id"].tolist()
        self.heights = df["height"].tolist()
        self.widths = df["width"].tolist()
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        img_path = self.img_paths[index]
        id_ = self.ids[index]
        img = np.array(load_image(img_path))
        h = int(self.heights[index])
        w = int(self.widths[index])

        if self.transforms:
            data = self.transforms(image=img)
            img = data["image"]

        img = np.transpose(img, (2, 0, 1))
        return torch.tensor(img), id_, torch.tensor(h), torch.tensor(w)




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
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(
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
        self.down4 = Down(base * 8, base * 16)
        self.up1 = Up(base * 16, base * 8)
        self.up2 = Up(base * 8, base * 4)
        self.up3 = Up(base * 4, base * 2)
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
    model = SimpleUNet(in_channels=3, classes=CFG.num_classes, base=32)
    return model


def load_model(path: str):
    model = build_model()
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model




## === cell 12
def mask2rle(msk, thr=0.5):
    msk = (np.array(msk) > thr).astype(np.uint8)
    pixels = msk.flatten()
    pad = np.array([0], dtype=pixels.dtype)
    pixels = np.concatenate([pad, pixels, pad])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(int(x)) for x in runs)


def masks2rles(msks, ids, heights, widths):
    pred_strings = []
    pred_ids = []
    pred_classes = []
    for idx in range(msks.shape[0]):
        height = int(
            heights[idx].item() if hasattr(heights[idx], "item") else heights[idx]
        )
        width = int(widths[idx].item() if hasattr(widths[idx], "item") else widths[idx])

        left = (width - msks[idx].shape[0]) // 2
        right = left
        top = (height - msks[idx].shape[1]) // 2
        bottom = top

        msk = cv2.copyMakeBorder(
            msks[idx], top, bottom, left, right, cv2.BORDER_CONSTANT, 0
        )

        rle = [None] * 3
        for midx in [0, 1, 2]:
            rle[midx] = mask2rle(msk[..., midx], thr=0.5)
        pred_strings.extend(rle)
        pred_ids.extend([ids[idx]] * len(rle))
        pred_classes.extend(["large_bowel", "small_bowel", "stomach"])
    return pred_strings, pred_ids, pred_classes




## === cell 13
@torch.no_grad()
def infer(model_paths, test_loader, num_log=1, thr=CFG.thr):
    models = []
    for p in model_paths:
        m = load_model(p).to(CFG.device)
        models.append(m)

    pred_strings = []
    pred_ids = []
    pred_classes = []
    imgs = []
    msks = []

    for idx, (img, ids, heights, widths) in enumerate(
        tqdm(test_loader, total=len(test_loader), desc="Infer")
    ):
        img = img.to(CFG.device, dtype=torch.float)

        size = img.size()
        msk = torch.zeros(
            (size[0], 3, size[2], size[3]), device=CFG.device, dtype=torch.float32
        )

        if len(models) == 0:
            gray = img.mean(dim=1, keepdim=True)  # (B,1,H,W)
            g_min = gray.amin(dim=(2, 3), keepdim=True)
            g_max = gray.amax(dim=(2, 3), keepdim=True)
            g = (gray - g_min) / (g_max - g_min + 1e-6)
            msk[:, 0:1] = (g > 0.55).float()
            msk[:, 1:2] = (g > 0.60).float()
            msk[:, 2:3] = (g > 0.50).float()
        else:
            for model in models:
                out = model(img)
                out = torch.sigmoid(out)
                msk += out / len(models)

        msk = (msk.permute((0, 2, 3, 1)) > thr).to(torch.uint8).cpu().numpy()
        result = masks2rles(msk, ids, heights, widths)

        pred_strings.extend(result[0])
        pred_ids.extend(result[1])
        pred_classes.extend(result[2])

        if idx < num_log:
            imgs.append(img.permute((0, 2, 3, 1)).cpu().numpy()[:10])
            msks.append(msk[:10])

        del img, msk
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del models
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pred_strings, pred_ids, pred_classes, imgs, msks




## === cell 14
test_dataset = TestDataset(test_df, transforms=test_transforms["test"])
test_loader = DataLoader(
    test_dataset,
    batch_size=CFG.val_batch_size,
    num_workers=2,
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
)

model_paths = sorted(glob(f"{CKPT_DIR}/**/best_epoch*.bin", recursive=True))
if len(model_paths) == 0 and CKPT_DIR_FALLBACK is not None:
    model_paths = sorted(
        glob(f"{CKPT_DIR_FALLBACK}/**/best_epoch*.bin", recursive=True)
    )

pred_strings, pred_ids, pred_classes, imgs, msks = infer(model_paths, test_loader)

len(pred_strings), len(pred_ids), len(pred_classes)



## === cell 15
if len(imgs) > 0 and len(msks) > 0:
    for img, msk in zip(imgs[0][:2], msks[0][:2]):
        plt.figure(figsize=(12, 7))
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
pred_df = pd.DataFrame(
    {"id": pred_ids, "class": pred_classes, "predicted": pred_strings}
)

sub_df_full = pd.read_csv(join(BASE_PATH, "sample_submission.csv"))
sub_df_full = sub_df_full.drop(columns=["predicted"])
sub_df_full = sub_df_full.merge(pred_df, on=["id", "class"], how="left")
sub_df_full["predicted"] = sub_df_full["predicted"].fillna("")

sub_df_full.to_csv("submission.csv", index=False)
print(sub_df_full.head(5))
print("Saved submission.csv with shape:", sub_df_full.shape)
print(
    "Num model weights found:",
    len(model_paths),
    "searched in:",
    CKPT_DIR,
    "fallback:",
    CKPT_DIR_FALLBACK,
)
