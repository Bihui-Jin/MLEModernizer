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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser



## === cell 1
import os, sys

print("Python:", sys.version)
print("Working dir:", os.getcwd())



## === cell 2
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
import os
import torch
from glob import glob
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import numpy as np
import cv2
import gc
import torch.nn as nn
import albumentations as A
import segmentation_models_pytorch as smp
import warnings

warnings.filterwarnings("ignore")

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

tqdm.pandas()




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_55/1236356351.py in <cell line: 0>()
     13 import torch.nn as nn
     14 import albumentations as A
---> 15 import segmentation_models_pytorch as smp
     16 import warnings
     17 

ModuleNotFoundError: No module named 'segmentation_models_pytorch'

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




## === cell 5
def get_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["case"] = df["id"].apply(lambda x: int(x.split("_")[0].split("case")[1]))
    df["day"] = df["id"].apply(lambda x: int(x.split("_")[1].split("day")[1]))
    df["slice"] = df["id"].apply(lambda x: int(x.split("_")[-1]))
    return df


def path2info(row: pd.Series) -> pd.Series:
    """
    BUGFIX: original indices were wrong for case/day, and width/height parsing used wrong tokens.
    File name pattern: {slice}_{width}_{height}_{spacingx}_{spacingy}.png
    Folder: .../caseXXX/caseXXX_dayYY/scans/...
    """
    path = row["image_path"]
    parts = path.split("/")
    folder = parts[-3]  # caseXXX_dayYY
    fn = parts[-1]
    case_str, day_str = folder.split("_")  # ["case110","day12"]
    case = int(case_str.replace("case", ""))
    day = int(day_str.replace("day", ""))
    fn_parts = fn.replace(".png", "").split("_")
    slice_ = int(fn_parts[0])
    width = int(fn_parts[1])
    height = int(fn_parts[2])

    row["height"] = height
    row["width"] = width
    row["case"] = case
    row["day"] = day
    row["slice"] = slice_
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
path_df = path_df.progress_apply(path2info, axis=1)
print("Found images:", len(path_df))
path_df.head()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/3686234232.py in <cell line: 0>()
      6 
      7 path_df = pd.DataFrame(paths, columns=["image_path"])
----> 8 path_df = path_df.progress_apply(path2info, axis=1)
      9 print("Found images:", len(path_df))
     10 path_df.head()

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'progress_apply'

## === cell 8
test_df = sub_df.merge(path_df, on=["case", "day", "slice"], how="left")

missing = test_df["image_path"].isna().sum()
print("Merged test_df rows:", len(test_df), "missing image_path:", missing)
if missing:
    test_df["image_path"] = test_df["image_path"].fillna("")

test_df.head()




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/1612746196.py in <cell line: 0>()
----> 1 test_df = sub_df.merge(path_df, on=["case", "day", "slice"], how="left")
      2 
      3 # Sanity check for missing image paths
      4 missing = test_df["image_path"].isna().sum()
      5 print("Merged test_df rows:", len(test_df), "missing image_path:", missing)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in merge(self, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
  10830         from pandas.core.reshape.merge import merge
  10831 
> 10832         return merge(
  10833             self,
  10834             right,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in merge(left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
    168         )
    169     else:
--> 170         op = _MergeOperation(
    171             left_df,
    172             right_df,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in __init__(self, left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, indicator, validate)
    792             left_drop,
    793             right_drop,
--> 794         ) = self._get_merge_keys()
    795 
    796         if left_drop:

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in _get_merge_keys(self)
   1295                         rk = cast(Hashable, rk)
   1296                         if rk is not None:
-> 1297                             right_keys.append(right._get_label_or_level_values(rk))
   1298                         else:
   1299                             # work-around for merge_asof(right_index=True)

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _get_label_or_level_values(self, key, axis)
   1909             values = self.axes[axis].get_level_values(key)._values
   1910         else:
-> 1911             raise KeyError(key)
   1912 
   1913         # Check for duplicates

KeyError: 'case'

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
            A.CenterCrop(*CFG.img_size),
            A.Normalize(mean=COLOR_MEAN, std=COLOR_STD, max_pixel_value=255),
        ],
        p=1.0,
    )
}




## === cell 11
def build_model():
    model = smp.Unet(
        encoder_name=CFG.encoder_name,
        encoder_weights=None,  # keep as in original inference code
        in_channels=3,
        classes=CFG.num_classes,
    )
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
def mask2rle(msk, thr=0.5):
    msk = np.array(msk)
    pixels = msk.flatten()
    pad = np.array([0])
    pixels = np.concatenate([pad, pixels, pad])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def masks2rles(msks, ids, heights, widths):
    pred_strings = []
    pred_ids = []
    pred_classes = []
    for idx in range(msks.shape[0]):
        height = (
            int(heights[idx])
            if hasattr(heights[idx], "__int__")
            else int(heights[idx].item())
        )
        width = (
            int(widths[idx])
            if hasattr(widths[idx], "__int__")
            else int(widths[idx].item())
        )

        left = max(0, (width - msks[idx].shape[0]) // 2)
        right = left
        top = max(0, (height - msks[idx].shape[1]) // 2)
        bottom = top

        msk = cv2.copyMakeBorder(
            msks[idx], top, bottom, left, right, cv2.BORDER_CONSTANT, 0
        )
        rle = [None] * 3
        for midx in [0, 1, 2]:
            rle[midx] = mask2rle(msk[..., midx])
        pred_strings.extend(rle)
        pred_ids.extend([ids[idx]] * len(rle))
        pred_classes.extend(["large_bowel", "small_bowel", "stomach"])
    return pred_strings, pred_ids, pred_classes




## === cell 13
@torch.no_grad()
def infer(model_paths, test_loader, num_log=1, thr=CFG.thr):
    """
    BUGFIX: load models once outside the batch loop (original loaded per batch -> extremely slow / may timeout).
    Core inference remains: average sigmoid outputs across checkpoints, then threshold.
    """
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



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2254059390.py in <cell line: 0>()
----> 1 test_dataset = TestDataset(test_df, transforms=test_transforms["test"])
      2 test_loader = DataLoader(
      3     test_dataset,
      4     batch_size=CFG.val_batch_size,
      5     num_workers=2,

NameError: name 'test_df' is not defined

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



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1067464779.py in <cell line: 0>()
      1 # Optional visualization (won't crash if no logs)
----> 2 if len(imgs) and len(msks):
      3     for img, msk in zip(imgs[0][:2], msks[0][:2]):
      4         plt.figure(figsize=(12, 5))
      5         plt.subplot(1, 3, 1)

NameError: name 'imgs' is not defined

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

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1009249504.py in <cell line: 0>()
      1 # Build submission ensuring exact required format/order from sample_submission.csv
      2 pred_df = pd.DataFrame(
----> 3     {"id": pred_ids, "class": pred_classes, "predicted": pred_strings}
      4 )
      5 

NameError: name 'pred_ids' is not defined
