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

0.5796554874810033

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the missing dependency on `segmentation_models_pytorch` by providing a tiny local Unet implementation that preserves the same “Unet with ResNet50 encoder” core idea and can load your checkpoints with `strict=False`. I also fix the pandas `progress_apply` error by using `tqdm.pandas()` correctly and falling back to plain `apply` if needed, and I make the image-path parsing robust so `case/day/slice/width/height` columns always exist for the merge. Finally, I ensure the pipeline always reaches the submission-writing cell and produces a valid `submission.csv` with the required columns and row order taken from `sample_submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with “all-empty masks” submissions, which in your code can happen if (a) no checkpoints are found/loaded correctly or (b) your post-processing produces invalid/empty RLE due to a shape/padding bug. I make two minimal, score-directed fixes: (1) correct the crop→pad logic so the predicted mask is resized back to the original (height,width) reliably (instead of attempting negative/incorrect padding), and (2) make RLE encoding column-major (Fortran order), which is the required convention for this competition and can otherwise crush the score. I also add a safe fallback to use the competition’s official input directory if your CKPT_DIR is missing, without changing any model logic. These are small changes that should move you up from 0.0 toward your target without altering the core model/ensemble approach.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most likely coming from effectively “no useful prediction”: the current pipeline (a) uses a center-crop at test time without a reliable inverse mapping, which can destroy spatial alignment and lead to near-empty/incorrect masks after resizing, and (b) very likely never actually loads compatible checkpoints because your local `resnet50(weights=None)` cannot match ResNet50-encoder checkpoints trained with ImageNet weights/normalization. I keep the same core Unet(ResNet50) idea and inference loop, but make two minimal, score-directed fixes: switch test preprocessing from `CenterCrop` to `Resize` (so we have a deterministic mapping back to original size) and load the encoder with ImageNet weights when requested to improve checkpoint compatibility. I also ensure the checkpoints are found more robustly (while keeping the same ensemble logic) and keep the RLE encoding in the required Fortran order. These changes should move you up from 0.0 toward your target without changing your model family or training approach (only fixing inference correctness/compatibility).'

# 9. Code solution

## === cell 0
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser



## === cell 1
import os, sys

print("Python:", sys.version)
print("Working dir:", os.getcwd())



## === cell 2
import os
import gc
import warnings
from glob import glob
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import albumentations as A

warnings.filterwarnings("ignore")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

try:
    tqdm.pandas()
except Exception as e:
    print("WARNING: tqdm.pandas() failed; will use plain apply. Error:", repr(e))




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


def set_seed(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(CFG.seed)



## === cell 4
TRAIN_METADATA_FILE = "../input/uw-madison-gi-tract-image-segmentation/train.csv"
BASE_PATH = "../input/uw-madison-gi-tract-image-segmentation"
CKPT_DIR = "../input/res50augupdated2iter"
SAMPLE_SUBMISSION_CSV_PATH = (
    "../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv"
)
TRAIN_FILE_CSV_PATH = TRAIN_METADATA_FILE

print("BASE_PATH exists:", os.path.exists(BASE_PATH))
print("CKPT_DIR exists:", os.path.exists(CKPT_DIR))
print("Sample submission exists:", os.path.exists(SAMPLE_SUBMISSION_CSV_PATH))

if (not os.path.exists(CKPT_DIR)) or (len(glob(f"{CKPT_DIR}/best_epoch*.bin")) == 0):
    candidates = sorted(glob("../input/**/best_epoch*.bin", recursive=True))
    if len(candidates):
        CKPT_DIR = os.path.dirname(candidates[0])
        print("CKPT_DIR fallback selected:", CKPT_DIR)
    else:
        print(
            "WARNING: No checkpoints found under ../input; will run empty-mask fallback."
        )




## === cell 5
def get_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["case"] = df["id"].apply(lambda x: int(x.split("_")[0].split("case")[1]))
    df["day"] = df["id"].apply(lambda x: int(x.split("_")[1].split("day")[1]))
    df["slice"] = df["id"].apply(lambda x: int(x.split("_")[-1]))
    return df


def path2info(row: pd.Series) -> pd.Series:
    """
    Robust parse for:
      folder .../caseXXX/caseXXX_dayYY/scans/{slice}_{width}_{height}_{sx}_{sy}.png
    """
    path = row["image_path"]
    parts = path.split("/")
    if len(parts) < 4:
        row["height"] = 0
        row["width"] = 0
        row["case"] = -1
        row["day"] = -1
        row["slice"] = -1
        return row

    folder = parts[-3]  # caseXXX_dayYY
    fn = parts[-1]

    try:
        case_str, day_str = folder.split("_")
        case = int(case_str.replace("case", ""))
        day = int(day_str.replace("day", ""))
    except Exception:
        case, day = -1, -1

    try:
        fn_parts = fn.replace(".png", "").split("_")
        slice_ = int(fn_parts[0])
        width = int(fn_parts[1])
        height = int(fn_parts[2])
    except Exception:
        slice_, width, height = -1, 0, 0

    row["height"] = int(height)
    row["width"] = int(width)
    row["case"] = int(case)
    row["day"] = int(day)
    row["slice"] = int(slice_)
    return row




## === cell 6
sub_df = pd.read_csv(SAMPLE_SUBMISSION_CSV_PATH)
if not len(sub_df):
    debug = True
    sub_df = pd.read_csv(TRAIN_FILE_CSV_PATH)[: 1000 * 3]
    sub_df = sub_df.drop(columns=["class", "segmentation"]).drop_duplicates()
else:
    debug = False
    sub_df = sub_df.drop(columns=["class", "predicted"]).drop_duplicates()

CFG.debug = debug
sub_df = get_metadata(sub_df)
print("debug:", debug, "sub_df rows:", len(sub_df))
sub_df.head()



## === cell 7
if debug:
    paths = glob(BASE_PATH + "/train/**/*png", recursive=True)
else:
    paths = glob(BASE_PATH + "/test/**/*png", recursive=True)

path_df = pd.DataFrame(paths, columns=["image_path"])

if hasattr(path_df, "progress_apply"):
    path_df = path_df.progress_apply(path2info, axis=1)
else:
    path_df = path_df.apply(path2info, axis=1)

for c in ["case", "day", "slice", "width", "height"]:
    if c not in path_df.columns:
        path_df[c] = -1 if c in ["case", "day", "slice"] else 0

print("Found images:", len(path_df))
path_df.head()



## === cell 8
test_df = sub_df.merge(path_df, on=["case", "day", "slice"], how="left")

missing = test_df["image_path"].isna().sum()
print("Merged test_df rows:", len(test_df), "missing image_path:", missing)
if missing:
    test_df["image_path"] = test_df["image_path"].fillna("")

for c in ["width", "height"]:
    if c not in test_df.columns:
        test_df[c] = 0
    test_df[c] = test_df[c].fillna(0).astype(int)

test_df.head()




## === cell 9
def load_image(path: str):
    return Image.open(path).convert("RGB")


class TestDataset(Dataset):
    def __init__(self, df, transforms=None, label=None):
        self.df = df.reset_index(drop=True)
        self.label = label
        self.img_paths = self.df["image_path"].tolist()
        self.ids = self.df["id"].tolist()
        self.heights = (
            self.df["height"].tolist()
            if "height" in self.df.columns
            else [0] * len(self.df)
        )
        self.widths = (
            self.df["width"].tolist()
            if "width" in self.df.columns
            else [0] * len(self.df)
        )
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        id_ = self.ids[index]

        if not img_path or (not os.path.exists(img_path)):
            h = int(self.heights[index]) if self.heights[index] else CFG.img_size[0]
            w = int(self.widths[index]) if self.widths[index] else CFG.img_size[1]
            img = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            img = np.array(load_image(img_path))
            h, w = img.shape[:2]

        if self.transforms:
            data = self.transforms(image=img)
            img = data["image"]

        img = np.transpose(img, (2, 0, 1))
        return torch.tensor(img), id_, h, w




## === cell 10
COLOR_MEAN: float = 0.349977
COLOR_STD: float = 0.215829

test_transforms = {
    "test": A.Compose(
        [
            A.Resize(*CFG.img_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=COLOR_MEAN, std=COLOR_STD, max_pixel_value=255),
        ],
        p=1.0,
    )
}




## === cell 11
class ConvRelu(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = ConvRelu(in_ch + skip_ch, out_ch)
        self.conv2 = ConvRelu(out_ch, out_ch)

    def forward(self, x, skip):
        x = torch.nn.functional.interpolate(
            x, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResNet50UNet(nn.Module):
    def __init__(self, classes=3, in_channels=3):
        super().__init__()
        if CFG.encoder_weights == "imagenet":
            weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        else:
            weights = None

        backbone = torchvision.models.resnet50(weights=weights)
        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        self.encoder0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.encoder1 = backbone.layer1
        self.encoder2 = backbone.layer2
        self.encoder3 = backbone.layer3
        self.encoder4 = backbone.layer4

        self.center = nn.Sequential(
            ConvRelu(2048, 512),
            ConvRelu(512, 512),
        )

        self.dec4 = DecoderBlock(512, 1024, 256)
        self.dec3 = DecoderBlock(256, 512, 128)
        self.dec2 = DecoderBlock(128, 256, 64)
        self.dec1 = DecoderBlock(64, 64, 32)

        self.final = nn.Conv2d(32, classes, kernel_size=1)

    def forward(self, x):
        e0 = self.encoder0(x)
        x = self.pool(e0)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        c = self.center(e4)
        d4 = self.dec4(c, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, e0)
        out = self.final(d1)
        return out


def build_model():
    model = ResNet50UNet(classes=CFG.num_classes, in_channels=3)
    return model


def load_model(path):
    model = build_model()
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("model."):
                nk = nk[len("model.") :]
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            new_state[nk] = v
        state = new_state
    model.load_state_dict(state, strict=False)
    model.eval()
    return model




## === cell 12
def mask2rle(msk):
    msk = np.asarray(msk, dtype=np.uint8)
    pixels = msk.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def masks2rles(msks, ids, heights, widths):
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
        if height <= 0 or width <= 0:
            height, width = CFG.img_size

        m = msks[idx]  # (Hc, Wc, 3) uint8
        if (m.shape[0] != height) or (m.shape[1] != width):
            m_resized = np.zeros((height, width, 3), dtype=np.uint8)
            for c in range(3):
                m_resized[..., c] = cv2.resize(
                    m[..., c],
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
            m = m_resized

        rle = [mask2rle(m[..., 0]), mask2rle(m[..., 1]), mask2rle(m[..., 2])]
        pred_strings.extend(rle)
        pred_ids.extend([ids[idx]] * 3)
        pred_classes.extend(["large_bowel", "small_bowel", "stomach"])
    return pred_strings, pred_ids, pred_classes




## === cell 13
@torch.no_grad()
def infer(model_paths, test_loader, num_log=1, thr=CFG.thr):
    if len(model_paths) == 0:
        pred_strings = []
        pred_ids = []
        pred_classes = []
        for _, ids, _, _ in tqdm(
            test_loader, total=len(test_loader), desc="Infer (no ckpt)"
        ):
            for id_ in ids:
                pred_ids.extend([id_] * 3)
                pred_classes.extend(["large_bowel", "small_bowel", "stomach"])
                pred_strings.extend(["", "", ""])
        return pred_strings, pred_ids, pred_classes, [], []

    models = []
    for path in model_paths:
        m = load_model(path).to(CFG.device)
        m.eval()
        models.append(m)

    msks_log = []
    imgs_log = []
    pred_strings = []
    pred_ids = []
    pred_classes = []

    for idx, (img, ids, heights, widths) in enumerate(
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

        msk_np = (msk.permute((0, 2, 3, 1)) > thr).to(torch.uint8).cpu().numpy()
        result = masks2rles(msk_np, ids, heights, widths)
        pred_strings.extend(result[0])
        pred_ids.extend(result[1])
        pred_classes.extend(result[2])

        if idx < num_log:
            img_np = img.permute((0, 2, 3, 1)).cpu().numpy()
            imgs_log.append(img_np[:10])
            msks_log.append(msk_np[:10])

        del img, msk, out, result, msk_np
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pred_strings, pred_ids, pred_classes, imgs_log, msks_log




## === cell 14
test_dataset = TestDataset(test_df, transforms=test_transforms["test"])
test_loader = DataLoader(
    test_dataset,
    batch_size=CFG.val_batch_size,
    num_workers=2,
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
)

model_paths = sorted(glob(f"{CKPT_DIR}/best_epoch*.bin"))
print("Num checkpoints found:", len(model_paths))
if len(model_paths) > 0:
    print("First ckpt:", model_paths[0])

pred_strings, pred_ids, pred_classes, imgs, msks = infer(model_paths, test_loader)

print("Pred rows:", len(pred_ids), len(pred_classes), len(pred_strings))



## === cell 15
if len(imgs) and len(msks):
    for img, msk in zip(imgs[0][:2], msks[0][:2]):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(img.astype(np.float32))
        plt.axis("off")
        plt.title("image")
        plt.subplot(1, 3, 2)
        plt.imshow(msk * 255)
        plt.axis("off")
        plt.title("mask")
        plt.subplot(1, 3, 3)
        plt.imshow(img.astype(np.float32))
        plt.imshow(msk * 255, alpha=0.4)
        plt.axis("off")
        plt.title("overlay")
        plt.tight_layout()
        plt.show()



## === cell 16
pred_df = pd.DataFrame(
    {"id": pred_ids, "class": pred_classes, "predicted": pred_strings}
)

sub_template = pd.read_csv(SAMPLE_SUBMISSION_CSV_PATH)  # has id,class,predicted
sub_out = sub_template.drop(columns=["predicted"]).merge(
    pred_df, on=["id", "class"], how="left"
)
sub_out["predicted"] = sub_out["predicted"].fillna("")

sub_out.to_csv("submission.csv", index=False)
print(sub_out.head())
print("Saved submission.csv with rows:", len(sub_out))
assert os.path.exists("submission.csv") and os.path.getsize("submission.csv") > 0
assert list(sub_out.columns) == ["id", "class", "predicted"]
assert len(sub_out) == len(sub_template)
