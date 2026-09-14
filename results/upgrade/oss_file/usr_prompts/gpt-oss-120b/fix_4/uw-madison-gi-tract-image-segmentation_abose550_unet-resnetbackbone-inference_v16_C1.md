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

0.1141

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I add a safe fallback for the missing `segmentation_models_pytorch` package by defining a minimal dummy UNet that returns zero‑filled logits, and I ensure the Albumentations alias `A` is imported correctly. This resolves the import errors, lets the transforms be built, and allows inference to run (producing empty masks) so a valid `submission.csv` file is written. No other logic is altered.'
- What this solution (achieved 0.1141) has done: 'I add a lightweight fallback inside the inference function so that when no pretrained checkpoint is found the code produces a simple binary mask derived from the normalized input image (using the same threshold that was originally applied to model outputs). This gives non‑empty predictions, moving the Dice/Hausdorff score away from 0 while preserving the original pipeline and all existing logic. The change is confined to the `infer` function in cell 22 and leaves the rest of the script untouched.'

# 9. Code solution

## === cell 0
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch
from glob import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import numpy as np
import cv2
import copy
from matplotlib.patches import Rectangle
from sklearn.model_selection import train_test_split
from collections import defaultdict
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings

warnings.filterwarnings("ignore")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
tqdm.pandas()

try:
    import segmentation_models_pytorch as smp
except ModuleNotFoundError:
    import torch.nn as nn

    class _DummyUNet(nn.Module):
        def __init__(self, in_channels=3, classes=3):
            super().__init__()
            self.in_channels = in_channels
            self.classes = classes

        def forward(self, x):
            batch, _, h, w = x.shape
            return torch.zeros(
                batch, self.classes, h, w, device=x.device, dtype=x.dtype
            )

    class smp:
        @staticmethod
        def Unet(encoder_name=None, encoder_weights=None, in_channels=3, classes=3):
            return _DummyUNet(in_channels=in_channels, classes=classes)


from torch import nn
import gc
from torch.optim import lr_scheduler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings




## === cell 1
pass




## === cell 2
pass




## === cell 3
pass




## === cell 4
pass




## === cell 5
pass




## === cell 6
pass




## === cell 7
pass




## === cell 8
pass




## === cell 9
pass




## === cell 10
pass




## === cell 11
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




## === cell 12
TRAIN_METADATA_FILE = "../input/uw-madison-gi-tract-image-segmentation/train.csv"
BASE_PATH = "../input/uw-madison-gi-tract-image-segmentation"
CKPT_DIR = "../input/res50augupdated2iter"
SAMPLE_SUBMISSION_CSV_PATH = f"{BASE_PATH}/sample_submission.csv"
TRAIN_FILE_CSV_PATH = TRAIN_METADATA_FILE




## === cell 13
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




## === cell 14
def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = np.expand_dims(img, axis=2)
    img = img.astype("float32")
    mx = np.max(img)
    if mx:
        img /= mx
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




## === cell 15
sub_df = pd.read_csv(SAMPLE_SUBMISSION_CSV_PATH)
if not len(sub_df):
    debug = True
    sub_df = pd.read_csv(TRAIN_FILE_CSV_PATH)[: 1000 * 3]
    sub_df = sub_df.drop(columns=["class", "segmentation"]).drop_duplicates()
else:
    debug = False
    sub_df = sub_df.drop(columns=["class", "predicted"]).drop_duplicates()
sub_df = get_metadata(sub_df)




## === cell 16
if debug:
    paths = glob(f"{BASE_PATH}/train/**/*png", recursive=True)
else:
    paths = glob(f"{BASE_PATH}/test/**/*png", recursive=True)
path_df = pd.DataFrame(paths, columns=["image_path"])
path_df = path_df.apply(path2info, axis=1)




## === cell 17
test_df = sub_df.merge(path_df, on=["case", "day", "slice"], how="left")
test_df = test_df.dropna(subset=["image_path"]).reset_index(drop=True)




## === cell 18
class TestDataset(Dataset):
    def __init__(self, df, transforms=None, label=False):
        self.df = df
        self.label = label
        self.img_paths = df["image_path"].tolist()
        self.ids = df["id"].tolist()
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        id_ = self.ids[index]
        img = np.array(load_image(img_path))
        h, w = img.shape[:2]

        if self.transforms:
            transformed = self.transforms(image=img)
            img = transformed["image"]

        if isinstance(img, np.ndarray):
            img = np.transpose(img, (2, 0, 1))
            img = torch.from_numpy(img).float()
        else:  # already a torch tensor from ToTensorV2
            img = img.float()

        if self.label:
            raise NotImplementedError("TestDataset is used only for inference.")
        else:
            return img, id_, h, w




## === cell 19
COLOR_MEAN = (0.349977, 0.349977, 0.349977)
COLOR_STD = (0.215829, 0.215829, 0.215829)
test_transforms = {
    "test": A.Compose(
        [
            A.CenterCrop(*CFG.img_size),
            A.Normalize(mean=COLOR_MEAN, std=COLOR_STD, max_pixel_value=255),
            ToTensorV2(),
        ],
        p=1.0,
    )
}




## === cell 20
def build_model():
    model = smp.Unet(
        encoder_name=CFG.encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=CFG.num_classes,
    )
    return model


def load_model(path):
    model = build_model()
    model.load_state_dict(torch.load(path, map_location=CFG.device))
    model.eval()
    return model




## === cell 21
def mask2rle(msk, thr=0.5):
    msk = np.array(msk).astype(np.uint8)
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
        top = (height - msks[idx].shape[1]) // 2
        msk = cv2.copyMakeBorder(
            msks[idx], top, top, left, left, cv2.BORDER_CONSTANT, 0
        )
        rle = [mask2rle(msk[..., c]) for c in range(3)]
        pred_strings.extend(rle)
        pred_ids.extend([ids[idx]] * 3)
        pred_classes.extend(["large_bowel", "small_bowel", "stomach"])
    return pred_strings, pred_ids, pred_classes




## === cell 22
@torch.no_grad()
def infer(model_paths, test_loader, num_log=1, thr=CFG.thr):
    pred_strings = []
    pred_ids = []
    pred_classes = []
    imgs = []
    msks = []

    for idx, (img, ids, heights, widths) in enumerate(
        tqdm(test_loader, total=len(test_loader), desc="Infer")
    ):
        img = img.to(CFG.device, dtype=torch.float)
        batch_size = img.size(0)
        agg_msk = torch.zeros(
            (batch_size, 3, img.size(2), img.size(3)), device=CFG.device
        )

        if model_paths:
            for path in model_paths:
                model = load_model(path).to(CFG.device)
                out = model(img)
                out = torch.sigmoid(out)
                agg_msk += out / len(model_paths)
                del model, out
                torch.cuda.empty_cache()
        else:
            pred = (img > thr).float()  # (B, C, H, W)
            agg_msk = pred  # use as the aggregated mask

        pred = (agg_msk.permute(0, 2, 3, 1) > thr).cpu().numpy().astype(np.uint8)
        strings, ids_out, classes_out = masks2rles(pred, ids, heights, widths)
        pred_strings.extend(strings)
        pred_ids.extend(ids_out)
        pred_classes.extend(classes_out)

        if idx < num_log:
            imgs.append(img.permute(0, 2, 3, 1).cpu().numpy())
            msks.append(pred)

        del img, agg_msk, pred
        gc.collect()
        torch.cuda.empty_cache()

    return pred_strings, pred_ids, pred_classes, imgs, msks




## === cell 23
test_dataset = TestDataset(test_df, transforms=test_transforms["test"], label=False)
test_loader = DataLoader(
    test_dataset,
    batch_size=CFG.val_batch_size,
    num_workers=4,
    shuffle=False,
    pin_memory=False,
)
model_paths = glob(f"{CKPT_DIR}/best_epoch*.bin")
pred_strings, pred_ids, pred_classes, imgs, msks = infer(model_paths, test_loader)




## === cell 24
pred_df = pd.DataFrame(
    {"id": pred_ids, "class": pred_classes, "predicted": pred_strings}
)
if not CFG.debug:
    sub_template = pd.read_csv(f"{BASE_PATH}/sample_submission.csv")
    sub_template = sub_template.drop(columns=["predicted"])
else:
    sub_template = pd.read_csv(TRAIN_FILE_CSV_PATH)[: 1000 * 3].drop(
        columns=["segmentation"]
    )

submission = sub_template.merge(pred_df, on=["id", "class"], how="left")
submission.to_csv("submission.csv", index=False)
print(submission.head())
