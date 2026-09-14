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

- What this solution (achieved 0.18972) has done: 'I fixed the missing `segmentation_models_pytorch` import by adding a safe fallback and switched the model builder to use a torchvision segmentation model. I also added a fallback inference path that creates simple Otsu‑threshold masks when no checkpoint files are found, ensuring the pipeline always produces predictions and writes a valid `submission.csv`. These changes resolve the import errors, the undefined transform variable, and guarantee a runnable end‑to‑end script that outputs the required CSV.'

# 9. Code solution

## === cell 0
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser




## === cell 1
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
import os
import torch
from glob import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import numpy as np
from torchvision import transforms as T
from torchvision.utils import make_grid
import cv2
import copy
from matplotlib.patches import Rectangle
from sklearn.model_selection import train_test_split
from collections import defaultdict

try:
    import segmentation_models_pytorch as smp
except ModuleNotFoundError:
    smp = None
    import torchvision.models.segmentation as seg_models
from torch.cuda import amp
import torch.nn as nn
import torch.optim as optim
import time
import gc
from torch.optim import lr_scheduler
import albumentations as A
from albumentations.pytorch import ToTensorV2

import warnings

warnings.filterwarnings("ignore")

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
tqdm.pandas()  # enables tqdm progress_apply if needed later




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
    T_max = int(30000 / train_batch_size * epochs) + 50  # max iterations for scheduler
    num_classes = 3
    val_split_percentage = 0.2
    n_accumulate = max(1, 32 // train_batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    thr = 0.45




## === cell 3
TRAIN_METADATA_FILE = "../input/uw-madison-gi-tract-image-segmentation/train.csv"
TRAIN_DIR = "../input/uw-madison-gi-tract-image-segmentation/train/"
BASE_PATH = "../input/uw-madison-gi-tract-image-segmentation"
CKPT_DIR = "../input/resnet50aug-updated"
SAMPLE_SUBMISSION_CSV_PATH = (
    "../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv"
)
TRAIN_FILE_CSV_PATH = TRAIN_METADATA_FILE




## === cell 4
def get_metadata(df):
    df["case"] = df["id"].apply(lambda x: int(x.split("_")[0].split("case")[1]))
    df["day"] = df["id"].apply(lambda x: int(x.split("_")[1].split("day")[1]))
    df["slice"] = df["id"].apply(lambda x: int(x.split("_")[-1]))
    return df


def path2info(row):
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




## === cell 5
def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = np.expand_dims(img, axis=2)
    img = img.astype("float32")  # original is uint16
    mx = np.max(img)
    if mx:
        img /= mx  # scale image to [0, 1]
    return img


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
path_df = path_df.apply(path2info, axis=1)
path_df.head()




## === cell 8
sub_df.head()




## === cell 9
test_df = sub_df.merge(path_df, on=["case", "day", "slice"], how="left")
test_df.head()




## === cell 10
class TestDataset(Dataset):
    def __init__(self, df, transforms=None, label=None):
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
        img = load_image(img_path)
        img = np.array(img)
        h, w = img.shape[:2]
        if self.label and self.msk_paths is not None:
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
            return torch.tensor(img), id_, h, w




## === cell 11
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




## === cell 12
def build_model():
    """
    Build a segmentation model.
    If SMP is available we keep the original UNet but now pass the
    encoder_weights defined in CFG (pre‑trained ImageNet weights).
    If we fall back to torchvision we enable pretrained weights,
    giving a much stronger baseline for the no‑checkpoint path.
    """
    if smp is not None:
        model = smp.Unet(
            encoder_name=CFG.encoder_name,
            encoder_weights=CFG.encoder_weights,  # use pretrained encoder
            in_channels=3,
            classes=CFG.num_classes,
        )
    else:
        model = seg_models.deeplabv3_resnet50(
            pretrained=True,  # <- enable pretrained weights
            progress=False,
            num_classes=CFG.num_classes,
        )
    return model


def load_model(path):
    model = build_model()
    model.load_state_dict(torch.load(path, map_location=CFG.device))
    model.eval()
    return model




## === cell 13
def mask2rle(msk, thr=0.5):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    """
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
        height = heights[idx].item()
        width = widths[idx].item()
        left = (width - msks[idx].shape[0]) // 2
        right = left
        top = (height - msks[idx].shape[1]) // 2
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




## === cell 14
@torch.no_grad()
def infer(model_paths, test_loader, num_log=1, thr=CFG.thr):
    """
    Run inference.
    - If checkpoint files are found, we ensemble them (original behaviour).
    - If no checkpoints exist, we now run the pretrained model
      (instead of the pure Otsu threshold) to obtain a stronger baseline.
    """
    pred_strings = []
    pred_ids = []
    pred_classes = []
    imgs = []
    msks = []
    for idx, (img, ids, heights, widths) in enumerate(
        tqdm(test_loader, total=len(test_loader), desc="Infer ")
    ):
        img = img.to(CFG.device, dtype=torch.float)
        batch_size = img.size(0)

        if model_paths:
            msk_acc = torch.zeros(
                (batch_size, 3, img.size(2), img.size(3)),
                device=CFG.device,
                dtype=torch.float32,
            )
            for path in model_paths:
                model = load_model(path)
                model = model.to(CFG.device)
                out = model(img)
                if isinstance(out, dict) and "out" in out:
                    out = out["out"]
                out = torch.sigmoid(out)
                msk_acc += out / len(model_paths)
                del model, out
                torch.cuda.empty_cache()
            msk_bin = (msk_acc.permute(0, 2, 3, 1) > thr).to(torch.uint8).cpu().numpy()
        else:
            model = build_model()
            model = model.to(CFG.device)
            model.eval()
            out = model(img)
            if isinstance(out, dict) and "out" in out:
                out = out["out"]
            out = torch.sigmoid(out)
            msk_bin = (out.permute(0, 2, 3, 1) > thr).to(torch.uint8).cpu().numpy()
            del model, out
            torch.cuda.empty_cache()

        result = masks2rles(msk_bin, ids, heights, widths)
        pred_strings.extend(result[0])
        pred_ids.extend(result[1])
        pred_classes.extend(result[2])

        if idx < num_log:
            imgs.append(img.permute(0, 2, 3, 1).cpu().numpy())
            msks.append(msk_bin)
        del img, msk_bin, result
        gc.collect()
        torch.cuda.empty_cache()
    return pred_strings, pred_ids, pred_classes, imgs, msks




## === cell 15
test_dataset = TestDataset(test_df, transforms=test_transforms["test"])
test_loader = DataLoader(
    test_dataset,
    batch_size=CFG.val_batch_size,
    num_workers=4,
    shuffle=False,
    pin_memory=False,
)
model_paths = glob(f"{CKPT_DIR}/best_epoch*.bin")
pred_strings, pred_ids, pred_classes, imgs, msks = infer(model_paths, test_loader)




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_56/3562476530.py in <cell line: 0>()
      8 )
      9 model_paths = glob(f"{CKPT_DIR}/best_epoch*.bin")
---> 10 pred_strings, pred_ids, pred_classes, imgs, msks = infer(model_paths, test_loader)
     11 
     12 

/usr/local/lib/python3.11/dist-packages/torch/utils/_contextlib.py in decorate_context(*args, **kwargs)
    114     def decorate_context(*args, **kwargs):
    115         with ctx_factory():
--> 116             return func(*args, **kwargs)
    117 
    118     return decorate_context

/tmp/ipykernel_56/2050645128.py in infer(model_paths, test_loader, num_log, thr)
     37         else:
     38             # No checkpoints: use the pretrained model directly
---> 39             model = build_model()
     40             model = model.to(CFG.device)
     41             model.eval()

/tmp/ipykernel_56/186726764.py in build_model()
     16     else:
     17         # Use a pretrained backbone for better zero‑shot performance
---> 18         model = seg_models.deeplabv3_resnet50(
     19             pretrained=True,  # <- enable pretrained weights
     20             progress=False,

/usr/local/lib/python3.11/dist-packages/torchvision/models/_utils.py in wrapper(*args, **kwargs)
    140             kwargs.update(keyword_only_kwargs)
    141 
--> 142         return fn(*args, **kwargs)
    143 
    144     return wrapper

/usr/local/lib/python3.11/dist-packages/torchvision/models/_utils.py in inner_wrapper(*args, **kwargs)
    226                 kwargs[weights_param] = default_weights_arg
    227 
--> 228             return builder(*args, **kwargs)
    229 
    230         return inner_wrapper

/usr/local/lib/python3.11/dist-packages/torchvision/models/segmentation/deeplabv3.py in deeplabv3_resnet50(weights, progress, num_classes, aux_loss, weights_backbone, **kwargs)
    267     if weights is not None:
    268         weights_backbone = None
--> 269         num_classes = _ovewrite_value_param("num_classes", num_classes, len(weights.meta["categories"]))
    270         aux_loss = _ovewrite_value_param("aux_loss", aux_loss, True)
    271     elif num_classes is None:

/usr/local/lib/python3.11/dist-packages/torchvision/models/_utils.py in _ovewrite_value_param(param, actual, expected)
    244     if actual is not None:
    245         if actual != expected:
--> 246             raise ValueError(f"The parameter '{param}' expected value {expected} but got {actual} instead.")
    247     return expected
    248 

ValueError: The parameter 'num_classes' expected value 21 but got 3 instead.

## === cell 16
for img, msk in zip(imgs[0][:5], msks[0][:5]):
    plt.figure(figsize=(12, 7))
    plt.subplot(1, 3, 1)
    plt.imshow(img.astype(np.uint8), cmap="bone")
    plt.axis("off")
    plt.title("image")
    plt.subplot(1, 3, 2)
    plt.imshow(msk * 255)
    plt.axis("off")
    plt.title("mask")
    plt.subplot(1, 3, 3)
    plt.imshow(img.astype(np.uint8), cmap="bone")
    plt.imshow(msk * 255, alpha=0.4)
    plt.axis("off")
    plt.title("overlay")
    plt.tight_layout()
    plt.show()




## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/473235542.py in <cell line: 0>()
----> 1 for img, msk in zip(imgs[0][:5], msks[0][:5]):
      2     plt.figure(figsize=(12, 7))
      3     plt.subplot(1, 3, 1)
      4     plt.imshow(img.astype(np.uint8), cmap="bone")
      5     plt.axis("off")

NameError: name 'imgs' is not defined

## === cell 17
del imgs, msks
gc.collect()




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/3082903294.py in <cell line: 0>()
----> 1 del imgs, msks
      2 gc.collect()
      3 
      4 

NameError: name 'imgs' is not defined

## === cell 18
pred_df = pd.DataFrame(
    {"id": pred_ids, "class": pred_classes, "predicted": pred_strings}
)
if not debug:
    sub_df = pd.read_csv(BASE_PATH + "/sample_submission.csv")
    del sub_df["predicted"]
else:
    sub_df = pd.read_csv(BASE_PATH + "/train.csv")[: 1000 * 3]
    del sub_df["segmentation"]

sub_df = sub_df.merge(pred_df, on=["id", "class"])
sub_df.to_csv("submission.csv", index=False)
print(sub_df.head(5))

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/2410237748.py in <cell line: 0>()
      1 pred_df = pd.DataFrame(
----> 2     {"id": pred_ids, "class": pred_classes, "predicted": pred_strings}
      3 )
      4 if not debug:
      5     sub_df = pd.read_csv(BASE_PATH + "/sample_submission.csv")

NameError: name 'pred_ids' is not defined
