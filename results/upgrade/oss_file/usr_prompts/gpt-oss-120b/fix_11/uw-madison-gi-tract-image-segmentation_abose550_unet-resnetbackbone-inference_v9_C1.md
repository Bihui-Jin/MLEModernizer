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

0.0065

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The changes add a safe fallback for the missing `segmentation_models_pytorch` library, make the path‑parsing robust, ensure the test dataframe is correctly built even when the merge fails, and replace the model inference with a zero‑mask baseline (avoiding the need for a trained model). These fixes eliminate the import error, the ValueError during metadata extraction, and the subsequent NameErrors, allowing the script to run end‑to‑end and generate a valid `submission.csv` file.'
- What this solution (achieved 0.00728) has done: 'I replace the dummy zero‑mask inference with a simple intensity‑based mask: each image is converted to a grayscale intensity, thresholded (using a low value to capture more foreground), and the same binary mask is copied to all three classes. This keeps the overall pipeline unchanged while providing non‑empty predictions, which should raise the Dice component and therefore move the score toward the target.'
- What this solution (achieved 0.00737) has done: 'I replace the fixed‑threshold mask in the inference step with an Otsu‑based per‑image threshold, which usually yields a better binary segmentation and therefore a higher Dice score while keeping the rest of the pipeline unchanged. This small change is expected to move the current score (≈0.007) toward the target (~0.32) without altering the core model logic.'
- What this solution (achieved 0.00724) has done: 'I make two small, targeted adjustments: (1) in the inference step, lower the Otsu‑derived threshold slightly and apply a modest dilation to each binary mask so the predictions cover a bit more area, which should raise the Dice component; (2) fix the padding logic in `masks2rles` so the masks are correctly centered according to the original image dimensions, preventing unnecessary cropping that hurts the score. These changes keep the overall pipeline untouched while nudging the metric toward the target.'
- What this solution (achieved 0.00725) has done: 'I slightly increase the predicted mask size by using the raw Otsu threshold (removing the 0.8 scaling) and applying a two‑iteration dilation. This modest change should enlarge the binary masks, raise the Dice overlap, and move the score closer to the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.00722) has done: 'I improve the inference step by generating a separate binary mask for each RGB channel (instead of a single grayscale mask) and slightly lower the Otsu‑derived threshold (by a factor of 0.8). This modest change keeps the overall pipeline unchanged while providing more realistic, larger masks for each organ class, which should raise the Dice component and move the score toward the target.'
- What this solution (achieved 0.00722) has done: 'I load a (potentially pretrained) UNet model if a checkpoint exists and use its predictions for the masks instead of the purely intensity‑based Otsu masks. When no checkpoint is found the code falls back to the previous Otsu + dilation logic, preserving the original behaviour. This modest change keeps the overall pipeline intact while providing far more informative segmentations, moving the Dice component—and thus the combined score—closer to the target.'
- What this solution (achieved 0.00717) has done: 'I only adjust the inference step: use a single Otsu mask computed on the average of the three RGB channels (instead of per‑channel masks) and then copy that mask for all three classes, while also enlarging the mask with a larger dilation kernel (5×5) and more iterations (4). This keeps the overall pipeline unchanged but should give substantially larger, more consistent predictions, raising the Dice component and moving the score nearer to the target.'
- What this solution (achieved 0.0065) has done: 'I fix the RLE encoding to use column‑major (Fortran) order, which matches the competition’s expected pixel ordering. This small change should make the predicted masks line up with the ground‑truth masks and improve the Dice component, moving the score closer to the target without altering the core model or inference logic.'

# 9. Code solution

## === cell 0
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
import os
import torch
from glob import glob

from tqdm.notebook import tqdm
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
import warnings

try:
    import segmentation_models_pytorch as smp
except ModuleNotFoundError:
    import torch.nn as nn

    class DummyUNet(nn.Module):
        def __init__(self, in_channels=3, classes=3):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, classes, kernel_size=1)

        def forward(self, x):
            return self.conv(x)

    smp = type(
        "smp",
        (),
        {
            "Unet": lambda *args, **kwargs: DummyUNet(
                in_channels=kwargs.get("in_channels", 3),
                classes=kwargs.get("classes", 3),
            )
        },
    )

from torch.cuda import amp
import torch.nn as nn
import torch.optim as optim
import time
import gc
from torch.optim import lr_scheduler
import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings("ignore")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
tqdm.pandas()




## === cell 1
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




## === cell 2
TRAIN_METADATA_FILE = "../input/uw-madison-gi-tract-image-segmentation/train.csv"
TRAIN_DIR = "../input/uw-madison-gi-tract-image-segmentation/train/"
BASE_PATH = "../input/uw-madison-gi-tract-image-segmentation"
CKPT_DIR = "../input/weights"
SAMPLE_SUBMISSION_CSV_PATH = (
    "../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv"
)
TRAIN_FILE_CSV_PATH = TRAIN_METADATA_FILE




## === cell 3
def get_metadata(df):
    df["case"] = df["id"].apply(lambda x: int(x.split("_")[0].split("case")[1]))
    df["day"] = df["id"].apply(lambda x: int(x.split("_")[1].split("day")[1]))
    df["slice"] = df["id"].apply(lambda x: int(x.split("_")[-1]))
    return df


def path2info(row):
    path = row["image_path"]
    parts = path.replace("\\", "/").split("/")
    if len(parts) < 4:
        return row
    case_dir = parts[-4] if parts[-4].startswith("case") else parts[-3]
    day_dir = parts[-3] if "_" in parts[-3] else parts[-2]
    try:
        case = int(case_dir.replace("case", ""))
    except Exception:
        case = 0
    try:
        day = int(day_dir.split("_")[1].replace("day", ""))
    except Exception:
        day = 0
    fname = os.path.splitext(parts[-1])[0]
    tokens = fname.split("_")
    slice_num = 0
    for tok in reversed(tokens):
        if tok.isdigit():
            slice_num = int(tok)
            break
    width = int(tokens[0]) if len(tokens) > 0 and tokens[0].isdigit() else 0
    height = int(tokens[1]) if len(tokens) > 1 and tokens[1].isdigit() else 0
    row["case"] = case
    row["day"] = day
    row["slice"] = slice_num
    row["width"] = width
    row["height"] = height
    return row




## === cell 4
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




## === cell 5
sub_df = pd.read_csv(SAMPLE_SUBMISSION_CSV_PATH)
if not len(sub_df):
    debug = True
    sub_df = pd.read_csv(TRAIN_FILE_CSV_PATH)[: 1000 * 3]
    sub_df = sub_df.drop(columns=["class", "segmentation"]).drop_duplicates()
else:
    debug = False
    sub_df = sub_df.drop(columns=["class", "predicted"]).drop_duplicates()
sub_df = get_metadata(sub_df)



## === cell 6
if debug:
    paths = glob(BASE_PATH + "/train/**/*png", recursive=True)
else:
    paths = glob(BASE_PATH + "/test/**/*png", recursive=True)
path_df = pd.DataFrame(paths, columns=["image_path"])
path_df = path_df.apply(path2info, axis=1)



## === cell 7
test_df = sub_df.merge(path_df, on=["case", "day", "slice"], how="left")
if test_df["image_path"].isnull().any():
    fallback = (
        path_df.groupby(["case", "day"])["image_path"]
        .first()
        .reset_index()
        .rename(columns={"image_path": "fallback_path"})
    )
    test_df = test_df.drop(columns=["image_path"]).merge(
        fallback, on=["case", "day"], how="left"
    )
    test_df["image_path"] = test_df["fallback_path"]
    test_df = test_df.drop(columns=["fallback_path"])
test_df = test_df.dropna(subset=["image_path"]).reset_index(drop=True)




## === cell 8
class TestDataset(Dataset):
    def __init__(self, df, transforms=None, label=None):
        self.df = df
        self.label = label
        self.img_paths = df["image_path"].tolist()
        self.ids = df["id"].tolist()
        if "msk_path" in df.columns:
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
            return torch.tensor(img), id_, h, w




## === cell 9
test_transforms = {
    "test": A.Compose(
        [
            A.CenterCrop(*CFG.img_size),
        ],
        p=1.0,
    )
}




## === cell 10
def build_model():
    model = smp.Unet(
        encoder_weights=None,
        in_channels=3,
        classes=CFG.num_classes,
    )
    return model


def load_model(path):
    model = build_model()
    try:
        model.load_state_dict(torch.load(path, map_location=CFG.device))
    except Exception:
        pass
    model.to(CFG.device)
    model.eval()
    return model




## === cell 11
def mask2rle(msk, thr=0.5):
    """
    Convert a binary mask to run‑length encoding using column‑major order,
    which matches the competition specification.
    """
    msk = (msk > thr).astype(np.uint8)
    pixels = msk.flatten(order="F")
    pad = np.array([0], dtype=np.uint8)
    pixels = np.concatenate([pad, pixels, pad])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def masks2rles(msks, ids, heights, widths):
    """
    Pad each predicted mask back to the original image size and convert each
    channel to RLE using the corrected column‑major mask2rle.
    """
    pred_strings = []
    pred_ids = []
    pred_classes = []
    for idx in range(msks.shape[0]):
        height = heights[idx].item()
        width = widths[idx].item()
        mask_h, mask_w = msks[idx].shape[:2]
        left = (width - mask_w) // 2
        right = left
        top = (height - mask_h) // 2
        bottom = top
        msk = cv2.copyMakeBorder(
            msks[idx], top, bottom, left, right, cv2.BORDER_CONSTANT, 0
        )
        rle = [None] * 3
        for midx in [0, 1, 2]:
            rle[midx] = mask2rle(msk[..., midx])
        pred_strings.extend(rle)
        pred_ids.extend([ids[idx]] * 3)
        pred_classes.extend(["large_bowel", "small_bowel", "stomach"])
    return pred_strings, pred_ids, pred_classes




## === cell 12
@torch.no_grad()
def infer(model_paths, test_loader, num_log=1, thr=CFG.thr):
    """
    Perform inference using a pretrained UNet if a checkpoint is found.
    When no checkpoint is available, fall back to a single Otsu mask computed on
    the average of the three RGB channels and then copy it to all classes.
    The mask is dilated with a larger kernel and more iterations to increase
    coverage, which should raise the Dice component and move the score toward the
    target.
    """
    pred_strings = []
    pred_ids = []
    pred_classes = []
    imgs = []
    msks = []

    model = None
    if model_paths:
        try:
            model = load_model(model_paths[0])
        except Exception:
            model = None

    kernel = np.ones((5, 5), np.uint8)
    dilation_iters = 4
    thr_scale = 1.0  # no scaling; use Otsu threshold directly

    for idx, (img, ids, heights, widths) in enumerate(
        tqdm(test_loader, total=len(test_loader), desc="Infer ")
    ):
        img = img.to(CFG.device, dtype=torch.float)
        batch_size = img.size(0)

        if model is not None:
            logits = model(img)  # (B, C, H, W)
            probs = torch.sigmoid(logits)
            binary = (probs > thr).cpu().numpy().astype(np.uint8)  # (B, C, H, W)
        else:
            binary = []
            for b in range(batch_size):
                ch_np = img[b].cpu().numpy()  # (C, H, W)
                mean_img = np.mean(ch_np, axis=0)  # (H, W)
                mean_uint8 = (mean_img * 255).astype(np.uint8)
                _, otsu_thr = cv2.threshold(
                    mean_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                thr_norm = (otsu_thr / 255.0) * thr_scale
                mask = (mean_img > thr_norm).astype(np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=dilation_iters)
                mask_stack = np.stack([mask, mask, mask], axis=-1)  # (H, W, 3)
                binary.append(mask_stack)
            binary = np.stack(binary, axis=0)  # (B, H, W, 3)

        if binary.ndim == 4:
            msk_bin = binary  # already (B, H, W, C)
        else:
            msk_bin = np.stack(binary, axis=0)

        result = masks2rles(msk_bin, ids, heights, widths)
        pred_strings.extend(result[0])
        pred_ids.extend(result[1])
        pred_classes.extend(result[2])

        if idx < num_log:
            img_np = img.permute(0, 2, 3, 1).cpu().numpy()
            imgs.append(img_np[:10])
            msks.append(msk_bin[:10])

        del img, binary, msk_bin, result
        gc.collect()
        torch.cuda.empty_cache()
    return pred_strings, pred_ids, pred_classes, imgs, msks




## === cell 13
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



## === cell 14
if imgs and msks:
    for img_batch, msk_batch in zip(imgs[0][:5], msks[0][:5]):
        plt.figure(figsize=(12, 7))
        plt.subplot(1, 3, 1)
        plt.imshow(img_batch, cmap="bone")
        plt.axis("off")
        plt.title("image")
        plt.subplot(1, 3, 2)
        plt.imshow(msk_batch * 255)
        plt.axis("off")
        plt.title("mask")
        plt.subplot(1, 3, 3)
        plt.imshow(img_batch, cmap="bone")
        plt.imshow(msk_batch * 255, alpha=0.4)
        plt.axis("off")
        plt.title("overlay")
        plt.tight_layout()
        plt.show()



## === cell 15
del imgs, msks
gc.collect()



## === cell 16
pred_df = pd.DataFrame(
    {"id": pred_ids, "class": pred_classes, "predicted": pred_strings}
)
if not CFG.debug:
    sub_df_template = pd.read_csv(BASE_PATH + "/sample_submission.csv")
    del sub_df_template["predicted"]
else:
    sub_df_template = pd.read_csv(BASE_PATH + "/train.csv")[: 1000 * 3]
    del sub_df_template["segmentation"]

sub_df = sub_df_template.merge(pred_df, on=["id", "class"], how="left")
sub_df.to_csv("submission.csv", index=False)
print(sub_df.head(5))
