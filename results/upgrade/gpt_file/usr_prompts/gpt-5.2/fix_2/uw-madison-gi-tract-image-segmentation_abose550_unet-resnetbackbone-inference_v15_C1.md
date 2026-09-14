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

0.5882940727366334

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
import sys, os, glob

print("Python:", sys.version)
print("CWD:", os.getcwd())



## === cell 2
print("Skipping external modellibs setup (not present / not needed).")



## === cell 3
print("Skipping efficientnet_pytorch manual install/copy (not required).")



## === cell 4
print("Skipping timm manual install/copy (not required).")



## === cell 5
print("Skipping pretrained-models manual install/copy (not required).")



## === cell 6
print(
    "Skipping segmentation_models_pytorch wheel install; using environment package if available."
)



## === cell 7
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
from albumentations.pytorch import ToTensorV2

import warnings

warnings.filterwarnings("ignore")

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

PANDARALLEL_AVAILABLE = False
try:
    from pandarallel import pandarallel  # noqa: F401

    pandarallel.initialize(progress_bar=True)
    PANDARALLEL_AVAILABLE = True
except Exception as e:
    print("pandarallel not available; falling back to pandas apply. Reason:", repr(e))

try:
    tqdm.pandas()
except Exception as e:
    print("tqdm.pandas() not available; will not use progress_apply. Reason:", repr(e))

import segmentation_models_pytorch as smp

print("Torch:", torch.__version__, "CUDA available:", torch.cuda.is_available())




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_55/2376228098.py in <cell line: 0>()
     37     print("tqdm.pandas() not available; will not use progress_apply. Reason:", repr(e))
     38 
---> 39 import segmentation_models_pytorch as smp
     40 
     41 print("Torch:", torch.__version__, "CUDA available:", torch.cuda.is_available())

ModuleNotFoundError: No module named 'segmentation_models_pytorch'

## === cell 8
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




## === cell 9
TRAIN_METADATA_FILE = "../input/uw-madison-gi-tract-image-segmentation/train.csv"
TRAIN_DIR = "../input/uw-madison-gi-tract-image-segmentation/train/"
BASE_PATH = "../input/uw-madison-gi-tract-image-segmentation"
CKPT_DIR = "../input/resnet50aug-updated"
SAMPLE_SUBMISSION_CSV_PATH = (
    "../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv"
)
TRAIN_FILE_CSV_PATH = TRAIN_METADATA_FILE

print("BASE_PATH exists:", os.path.exists(BASE_PATH))
print("CKPT_DIR exists:", os.path.exists(CKPT_DIR))
print("Sample submission exists:", os.path.exists(SAMPLE_SUBMISSION_CSV_PATH))




## === cell 10
def get_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["case"] = df["id"].apply(lambda x: int(x.split("_")[0].split("case")[1]))
    df["day"] = df["id"].apply(lambda x: int(x.split("_")[1].split("day")[1]))
    df["slice"] = df["id"].apply(lambda x: int(x.split("_")[-1]))
    return df


def path2info(row: pd.Series) -> pd.Series:
    path = row["image_path"]
    data = path.split("/")
    slice_ = int(data[-1].split("_")[1])
    case = int(data[-3].split("_")[0].replace("case", ""))
    day = int(data[-3].split("_")[1].replace("day", ""))
    width = int(data[-1].split("_")[2])
    height = int(data[-1].split("_")[3])
    row["height"] = height
    row["width"] = width
    row["case"] = case
    row["day"] = day
    row["slice"] = slice_
    return row




## === cell 11
def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")




## === cell 12
sub_df = pd.read_csv(SAMPLE_SUBMISSION_CSV_PATH)
if not len(sub_df):
    debug = True
    sub_df = pd.read_csv(TRAIN_FILE_CSV_PATH)[: 1000 * 3]
    sub_df = sub_df.drop(columns=["class", "segmentation"]).drop_duplicates()
else:
    debug = False
    sub_df = sub_df.drop(columns=["class", "predicted"]).drop_duplicates()

sub_df = get_metadata(sub_df)
print("debug:", debug, "sub_df shape:", sub_df.shape)
sub_df.head()



## === cell 13
if debug:
    paths = glob(BASE_PATH + "/train/**/*png", recursive=True)
else:
    paths = glob(BASE_PATH + "/test/**/*png", recursive=True)

path_df = pd.DataFrame(paths, columns=["image_path"])

if hasattr(path_df, "progress_apply"):
    path_df = path_df.progress_apply(path2info, axis=1)
else:
    path_df = path_df.apply(path2info, axis=1)

print("Found images:", len(path_df))
path_df.head()



## === cell 14
test_df = sub_df.merge(path_df, on=["case", "day", "slice"], how="left")
missing = test_df["image_path"].isna().sum()
print("Merged test_df:", test_df.shape, "missing image_path:", missing)
if missing > 0:
    test_df_missing = test_df[test_df["image_path"].isna()].head(5)
    raise RuntimeError(
        f"Some ids could not be matched to image paths. Examples:\n{test_df_missing}"
    )

test_df.head()




## === cell 15
class TestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transforms=None, label=None):
        self.df = df
        self.label = label
        self.img_paths = df["image_path"].tolist()
        self.ids = df["id"].tolist()
        if "mask_path" in df.columns:
            self.msk_paths = df["mask_path"].tolist()
        else:
            self.msk_paths = None
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        id_ = self.ids[index]

        img = np.array(load_image(img_path))
        h, w = img.shape[:2]

        if self.label:
            msk_path = self.msk_paths[index]
            msk = np.load(msk_path).astype("float32") / 255.0
            if self.transforms:
                data = self.transforms(image=img, mask=msk)
                img = data["image"]
                msk = data["mask"]
            if isinstance(img, torch.Tensor):
                pass
            else:
                img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
            if isinstance(msk, torch.Tensor):
                pass
            else:
                msk = torch.from_numpy(np.transpose(msk, (2, 0, 1))).float()
            return img, msk
        else:
            if self.transforms:
                data = self.transforms(image=img)
                img = data["image"]
            if isinstance(img, torch.Tensor):
                img_t = img
            else:
                img_t = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
            return img_t, id_, torch.tensor(h), torch.tensor(w)




## === cell 16
COLOR_MEAN: float = 0.349977
COLOR_STD: float = 0.215829

test_transforms = {
    "test": A.Compose(
        [
            A.CenterCrop(*CFG.img_size),
            A.Normalize(mean=COLOR_MEAN, std=COLOR_STD, max_pixel_value=255),
            ToTensorV2(transpose_mask=True),
        ],
        p=1.0,
    )
}




## === cell 17
def build_model():
    model = smp.Unet(
        encoder_name=CFG.encoder_name,
        encoder_weights=None,  # keep as in original inference (weights loaded from ckpt)
        in_channels=3,
        classes=CFG.num_classes,
    )
    return model


def load_model(path: str):
    model = build_model()
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
        new_state = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("model."):
                nk = nk[len("model.") :]
            new_state[nk] = v
        state = new_state
    elif isinstance(ckpt, dict) and any(
        k.startswith(("encoder.", "decoder.", "segmentation_head."))
        for k in ckpt.keys()
    ):
        state = ckpt
    else:
        state = ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model




## === cell 18
def mask2rle(msk, thr=0.5):
    msk = np.array(msk)
    pixels = msk.flatten()
    pad = np.array([0], dtype=pixels.dtype)
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

        crop_h, crop_w = msks[idx].shape[0], msks[idx].shape[1]

        left = max(0, (width - crop_w) // 2)
        right = width - crop_w - left
        top = max(0, (height - crop_h) // 2)
        bottom = height - crop_h - top

        msk_full = cv2.copyMakeBorder(
            msks[idx], top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
        )

        rle = [None] * 3
        for midx in [0, 1, 2]:
            rle[midx] = mask2rle(msk_full[..., midx])
        pred_strings.extend(rle)
        pred_ids.extend([ids[idx]] * 3)
        pred_classes.extend(["large_bowel", "small_bowel", "stomach"])
    return pred_strings, pred_ids, pred_classes




## === cell 19
@torch.no_grad()
def infer(model_paths, test_loader, num_log=1, thr=CFG.thr):
    msks_log = []
    imgs_log = []
    pred_strings = []
    pred_ids = []
    pred_classes = []

    if len(model_paths) == 0:
        raise RuntimeError(
            f"No model checkpoints found under: {CKPT_DIR}. Expected pattern best_epoch*.bin"
        )

    models = []
    for path in model_paths:
        m = load_model(path).to(CFG.device)
        models.append(m)

    sigmoid = nn.Sigmoid()

    for bidx, (img, ids, heights, widths) in enumerate(
        tqdm(test_loader, total=len(test_loader), desc="Infer")
    ):
        img = img.to(CFG.device, dtype=torch.float32)

        size = img.size()
        msk = torch.zeros(
            (size[0], 3, size[2], size[3]), device=CFG.device, dtype=torch.float32
        )

        for m in models:
            out = m(img)
            out = sigmoid(out)
            msk += out / len(models)

        msk_np = (msk.permute((0, 2, 3, 1)) > thr).to(torch.uint8).cpu().numpy()
        result = masks2rles(msk_np, ids, heights, widths)
        pred_strings.extend(result[0])
        pred_ids.extend(result[1])
        pred_classes.extend(result[2])

        if bidx < num_log:
            imgs_log.append(img.permute((0, 2, 3, 1)).cpu().numpy()[:10])
            msks_log.append(msk_np[:10])

        del img, msk, out, result, msk_np
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pred_strings, pred_ids, pred_classes, imgs_log, msks_log




## === cell 20
test_dataset = TestDataset(test_df, transforms=test_transforms["test"])
test_loader = DataLoader(
    test_dataset,
    batch_size=CFG.val_batch_size,
    num_workers=2,
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
)

model_paths = sorted(glob(f"{CKPT_DIR}/best_epoch*.bin"))
print("Num model checkpoints:", len(model_paths))
pred_strings, pred_ids, pred_classes, imgs, msks = infer(model_paths, test_loader)

print("Pred rows:", len(pred_strings))



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/2954603421.py in <cell line: 0>()
     10 model_paths = sorted(glob(f"{CKPT_DIR}/best_epoch*.bin"))
     11 print("Num model checkpoints:", len(model_paths))
---> 12 pred_strings, pred_ids, pred_classes, imgs, msks = infer(model_paths, test_loader)
     13 
     14 print("Pred rows:", len(pred_strings))

/usr/local/lib/python3.11/dist-packages/torch/utils/_contextlib.py in decorate_context(*args, **kwargs)
    114     def decorate_context(*args, **kwargs):
    115         with ctx_factory():
--> 116             return func(*args, **kwargs)
    117 
    118     return decorate_context

/tmp/ipykernel_55/657647518.py in infer(model_paths, test_loader, num_log, thr)
      8 
      9     if len(model_paths) == 0:
---> 10         raise RuntimeError(
     11             f"No model checkpoints found under: {CKPT_DIR}. Expected pattern best_epoch*.bin"
     12         )

RuntimeError: No model checkpoints found under: ../input/resnet50aug-updated. Expected pattern best_epoch*.bin

## === cell 21
if len(imgs) > 0:
    for img, msk in zip(imgs[0][:2], msks[0][:2]):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.axis("OFF")
        plt.title("image")
        plt.subplot(1, 3, 2)
        plt.imshow(msk * 255)
        plt.axis("OFF")
        plt.title("mask")
        plt.subplot(1, 3, 3)
        plt.imshow(img)
        plt.imshow(msk * 255, alpha=0.4)
        plt.axis("OFF")
        plt.title("overlay")
        plt.tight_layout()
        plt.show()



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4031313674.py in <cell line: 0>()
      1 # Optional visualization (kept; safe even if running headless)
----> 2 if len(imgs) > 0:
      3     for img, msk in zip(imgs[0][:2], msks[0][:2]):
      4         plt.figure(figsize=(12, 6))
      5         plt.subplot(1, 3, 1)

NameError: name 'imgs' is not defined

## === cell 22
try:
    del imgs, msks
except Exception:
    pass
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()



## === cell 23
pred_df = pd.DataFrame(
    {"id": pred_ids, "class": pred_classes, "predicted": pred_strings}
)

if not debug:
    sub_df = pd.read_csv(BASE_PATH + "/sample_submission.csv")
    sub_df = sub_df.drop(columns=["predicted"])
else:
    sub_df = pd.read_csv(BASE_PATH + "/train.csv")[: 1000 * 3]
    sub_df = sub_df.drop(columns=["segmentation"])

sub_df = sub_df.merge(pred_df, on=["id", "class"], how="left")
sub_df["predicted"] = sub_df["predicted"].fillna("")
sub_df.to_csv("submission.csv", index=False)

print(sub_df.head(5))
print("Saved submission.csv with shape:", sub_df.shape)
print("Submission columns:", sub_df.columns.tolist())

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2649096478.py in <cell line: 0>()
      1 pred_df = pd.DataFrame(
----> 2     {"id": pred_ids, "class": pred_classes, "predicted": pred_strings}
      3 )
      4 
      5 if not debug:

NameError: name 'pred_ids' is not defined
