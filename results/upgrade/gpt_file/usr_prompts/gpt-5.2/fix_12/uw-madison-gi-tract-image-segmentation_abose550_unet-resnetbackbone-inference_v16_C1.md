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
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the “no checkpoints found → empty-mask fallback” path being taken, which guarantees a zero-like submission regardless of the rest of the pipeline. I make a minimal, score-directed change to robustly discover `best_epoch*.bin` checkpoints anywhere under `../input` (including nested competition folders) and use all of them for the existing ensemble inference, without changing the model architecture or inference logic. I also add a tiny safeguard that prints the final resolved checkpoint list and errors out early if the list is empty (so you don’t silently submit all-empty masks again). These changes should move your score up toward your target by ensuring real model predictions are produced.'
- What this solution (achieved 0.0) has done: 'I remove the hard failure when no checkpoints are found so the notebook always runs end-to-end and writes a valid `submission.csv` (this fixes the current runtime error and prevents downstream `NameError`s). Since you don’t have model checkpoints available in this environment, I keep your existing “no ckpt → empty masks” inference path, which produces a valid submission (score be low, but it be yielded). I also make the visualization and submission cells robust by initializing `imgs/msks/pred_*` even when inference falls back, without changing the model/inference logic when checkpoints do exist. All paths and the overall pipeline structure are preserved.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly because the notebook never finds any `best_epoch*.bin` checkpoints, so it falls back to all-empty RLEs (valid CSV but zero-like submission). I make a minimal, score-directed change to locate checkpoints more robustly by also scanning common weight extensions (`.pth`, `.pt`) and common filename patterns (`best*`, `fold*`, `checkpoint*`) under `../input`, while keeping the exact same model and inference logic. To avoid silently submitting empties again, I additionally hard-fail if no checkpoints are found (since a valid-but-empty submission cannot move you toward the 0.5796 target). No model architecture, loss, or inference semantics change—only checkpoint discovery and a safety guard.'
- What this solution (achieved 0.0) has done: 'I remove the hard failure when no checkpoints are found so the notebook always runs end-to-end and writes a valid `submission.csv` (fixing the current runtime error). I keep your existing inference logic unchanged when checkpoints do exist, but if none are available in this environment, it fall back to the already-implemented “no ckpt → empty RLEs” path to guarantee a valid submission file. I also make the checkpoint discovery slightly more robust by additionally scanning the current working directory (`/kaggle/working`) in case weights were copied there, without changing model logic. These changes are correctness/stability focused; without actual weights, the score cannot be meaningfully improved from 0.0 in this environment.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is coming from the “no checkpoints found → empty-mask predictions” branch, which guarantees an all-empty submission. To move the score upward toward your 0.5796 target without changing the model/inference core logic, I (1) expand checkpoint discovery to include common Kaggle weight locations/patterns (including `/kaggle/input` and nested competition folders) and (2) hard-stop if still no checkpoints are found so you don’t accidentally submit another 0.0 again. I also (3) make `mask2rle` return an empty string for empty masks (some scorers are picky) while keeping the required Fortran-order encoding. Everything else (architecture, transforms, thresholding, ensemble averaging) stays the same.'
- What this solution (achieved 0.0) has done: 'I remove the hard failure when checkpoints are missing so the pipeline always runs end-to-end and writes a valid `submission.csv` (fixing the current runtime error). To still move the score upward from 0.0 toward your target without changing your model architecture or inference semantics, I add a minimal fallback that trains the same ResNet50-UNet for a very small number of steps on the provided `train/` images (only if no checkpoints are found) and then uses it for test inference. This keeps the same core model/loss/training-loop approach (simple supervised segmentation) and avoids an all-empty submission. I also keep the existing correct Fortran-order RLE and resizing back to original (height, width) unchanged.'

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
import math
import random
import warnings
from glob import glob
from typing import List, Tuple, Optional

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
    train_batch_size = 8
    val_batch_size = 32
    img_size = (224, 224)
    scheduler = "CosineAnnealingLR"
    epochs = 22
    lr = 2e-3
    min_lr = 1e-6
    weight_decay = 1e-6
    T_max = int(30000 / max(1, train_batch_size) * epochs) + 50
    num_classes = 3
    val_split_percentage = 0.2
    n_accumulate = max(1, 32 // train_batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    thr = 0.45

    fallback_max_steps = 250  # small but non-trivial; within time budget
    fallback_num_train_images = 600  # cap to keep runtime bounded
    fallback_num_workers = 2


def set_seed(seed: int = 42):
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


def find_ckpt_paths(preferred_dir: str) -> List[str]:
    patterns = [
        os.path.join(preferred_dir, "best_epoch*.bin"),
        os.path.join(preferred_dir, "best*.bin"),
        os.path.join(preferred_dir, "*.bin"),
        os.path.join(preferred_dir, "*.pth"),
        os.path.join(preferred_dir, "*.pt"),
    ]
    found = []
    for p in patterns:
        found.extend(glob(p))
    found = sorted(set(found))
    if len(found):
        return found

    recursive_patterns = [
        "../input/**/best_epoch*.bin",
        "../input/**/best*.bin",
        "../input/**/fold*.bin",
        "../input/**/checkpoint*.bin",
        "../input/**/*.bin",
        "../input/**/*.pth",
        "../input/**/*.pt",
        "../working/**/best_epoch*.bin",
        "../working/**/best*.bin",
        "../working/**/fold*.bin",
        "../working/**/checkpoint*.bin",
        "../working/**/*.bin",
        "../working/**/*.pth",
        "../working/**/*.pt",
        "/kaggle/working/**/best_epoch*.bin",
        "/kaggle/working/**/best*.bin",
        "/kaggle/working/**/fold*.bin",
        "/kaggle/working/**/checkpoint*.bin",
        "/kaggle/working/**/*.bin",
        "/kaggle/working/**/*.pth",
        "/kaggle/working/**/*.pt",
        "/kaggle/input/**/best_epoch*.bin",
        "/kaggle/input/**/best*.bin",
        "/kaggle/input/**/fold*.bin",
        "/kaggle/input/**/checkpoint*.bin",
        "/kaggle/input/**/*.bin",
        "/kaggle/input/**/*.pth",
        "/kaggle/input/**/*.pt",
    ]
    candidates = []
    for rp in recursive_patterns:
        candidates.extend(glob(rp, recursive=True))
    candidates = sorted(set(candidates))

    filtered = []
    for p in candidates:
        try:
            if os.path.isfile(p) and os.path.getsize(p) > 50_000:  # ~50KB minimum
                filtered.append(p)
        except Exception:
            continue
    return filtered


model_paths_initial = find_ckpt_paths(CKPT_DIR)
print("Initial resolved ckpt count:", len(model_paths_initial))
if len(model_paths_initial):
    print("First ckpt:", model_paths_initial[0])




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
            if h <= 0:
                h = CFG.img_size[0]
            if w <= 0:
                w = CFG.img_size[1]
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
    if msk.max() == 0:
        return ""
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
def rle_decode(mask_rle: str, shape: Tuple[int, int]) -> np.ndarray:
    """
    Decode RLE string into binary mask.
    Competition uses column-major (Fortran) order.
    """
    h, w = shape
    if not isinstance(mask_rle, str) or mask_rle.strip() == "":
        return np.zeros((h, w), dtype=np.uint8)
    s = mask_rle.strip().split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(h * w, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        if lo >= 0 and hi > lo:
            img[lo:hi] = 1
    return img.reshape((h, w), order="F")


def build_train_dataframe_for_fallback(
    base_path: str, max_images: int = 600, seed: int = 42
) -> pd.DataFrame:
    """
    Minimal fallback: sample train slices and create per-slice 3-channel masks from train.csv RLE.
    This is only used if no model checkpoints are available.
    """
    train_csv = pd.read_csv(os.path.join(base_path, "train.csv"))
    train_csv = get_metadata(train_csv)

    img_paths = glob(base_path + "/train/**/*png", recursive=True)
    dfp = pd.DataFrame(img_paths, columns=["image_path"])
    if hasattr(dfp, "progress_apply"):
        dfp = dfp.progress_apply(path2info, axis=1)
    else:
        dfp = dfp.apply(path2info, axis=1)

    dfp = dfp[(dfp["case"] >= 0) & (dfp["day"] >= 0) & (dfp["slice"] >= 0)]
    dfp = dfp[["image_path", "case", "day", "slice", "width", "height"]]

    merged = dfp.merge(
        train_csv[["id", "class", "segmentation", "case", "day", "slice"]],
        on=["case", "day", "slice"],
        how="inner",
    )

    piv = merged.pivot_table(
        index=["case", "day", "slice", "image_path", "width", "height"],
        columns="class",
        values="segmentation",
        aggfunc="first",
    ).reset_index()

    for c in ["large_bowel", "small_bowel", "stomach"]:
        if c not in piv.columns:
            piv[c] = ""
        piv[c] = piv[c].fillna("")

    rng = np.random.default_rng(seed)
    if len(piv) > max_images:
        idx = rng.choice(len(piv), size=max_images, replace=False)
        piv = piv.iloc[idx].reset_index(drop=True)

    return piv


class FallbackTrainDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transforms):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = row["image_path"]
        img = np.array(load_image(path))
        h0, w0 = img.shape[:2]

        m_lb = rle_decode(row["large_bowel"], (h0, w0))
        m_sb = rle_decode(row["small_bowel"], (h0, w0))
        m_st = rle_decode(row["stomach"], (h0, w0))
        mask = np.stack([m_lb, m_sb, m_st], axis=-1).astype(np.uint8)

        data = self.transforms(image=img, mask=mask)
        img_t = data["image"]
        msk_t = data["mask"]

        img_t = np.transpose(img_t, (2, 0, 1))
        msk_t = np.transpose(msk_t, (2, 0, 1)).astype(np.float32)

        return torch.tensor(img_t), torch.tensor(msk_t)


train_transforms_fallback = A.Compose(
    [
        A.Resize(*CFG.img_size, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=COLOR_MEAN, std=COLOR_STD, max_pixel_value=255),
    ],
    p=1.0,
)


def fallback_train_one_model() -> nn.Module:
    """
    Bug fix + score-directed: if no checkpoints exist, train the same core UNet model briefly
    so we avoid a guaranteed all-empty submission.
    """
    df_train = build_train_dataframe_for_fallback(
        BASE_PATH, max_images=CFG.fallback_num_train_images, seed=CFG.seed
    )
    if len(df_train) == 0:
        print(
            "WARNING: Fallback training found 0 train samples; will output empty masks."
        )
        return None

    ds = FallbackTrainDataset(df_train, transforms=train_transforms_fallback)
    dl = DataLoader(
        ds,
        batch_size=CFG.train_batch_size,
        shuffle=True,
        num_workers=CFG.fallback_num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True if len(ds) >= CFG.train_batch_size else False,
    )

    model = build_model().to(CFG.device)
    model.train()

    opt = torch.optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    bce = nn.BCEWithLogitsLoss()

    step = 0
    pbar = tqdm(total=CFG.fallback_max_steps, desc="Fallback training", leave=False)
    while step < CFG.fallback_max_steps:
        for xb, yb in dl:
            xb = xb.to(CFG.device, dtype=torch.float32)
            yb = yb.to(CFG.device, dtype=torch.float32)

            logits = model(xb)
            loss = bce(logits, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            step += 1
            pbar.update(1)
            if step >= CFG.fallback_max_steps:
                break
        if len(dl) == 0:
            break

    pbar.close()
    model.eval()
    return model


@torch.no_grad()
def infer(model_paths, test_loader, num_log=1, thr=CFG.thr):
    if len(model_paths) == 0:
        fb_model = fallback_train_one_model()
        if fb_model is None:
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

        msks_log = []
        imgs_log = []
        pred_strings = []
        pred_ids = []
        pred_classes = []

        for idx, (img, ids, heights, widths) in enumerate(
            tqdm(test_loader, total=len(test_loader), desc="Infer (fallback model)")
        ):
            img = img.to(CFG.device, dtype=torch.float)
            out = torch.sigmoid(fb_model(img))
            msk_np = (out.permute((0, 2, 3, 1)) > thr).to(torch.uint8).cpu().numpy()

            result = masks2rles(msk_np, ids, heights, widths)
            pred_strings.extend(result[0])
            pred_ids.extend(result[1])
            pred_classes.extend(result[2])

            if idx < num_log:
                img_np = img.permute((0, 2, 3, 1)).cpu().numpy()
                imgs_log.append(img_np[:10])
                msks_log.append(msk_np[:10])

            del img, out, result, msk_np
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return pred_strings, pred_ids, pred_classes, imgs_log, msks_log

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

model_paths = model_paths_initial
if len(model_paths) == 0:
    model_paths = find_ckpt_paths(CKPT_DIR)

print("Num checkpoints found:", len(model_paths))
if len(model_paths) > 0:
    print("First ckpt:", model_paths[0])
else:
    print(
        "WARNING: No model checkpoints found. Will train a small fallback model from train.csv/train images."
    )

pred_strings, pred_ids, pred_classes, imgs, msks = infer(model_paths, test_loader)

print("Pred rows:", len(pred_ids), len(pred_classes), len(pred_strings))



## === cell 15
if "imgs" in globals() and "msks" in globals() and len(imgs) and len(msks):
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
if "pred_ids" not in globals():
    pred_ids, pred_classes, pred_strings = [], [], []

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
