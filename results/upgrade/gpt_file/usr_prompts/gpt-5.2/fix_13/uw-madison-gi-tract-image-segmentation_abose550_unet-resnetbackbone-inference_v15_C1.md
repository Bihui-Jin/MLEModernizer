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

0.5882940727366334

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the pipeline so it runs end-to-end in this Kaggle environment by removing the hard dependency on `segmentation_models_pytorch` (not installed) and replacing it with a minimal UNet implemented in pure PyTorch that preserves the same inference semantics (3-channel sigmoid outputs + thresholding + RLE). I also fix the missing checkpoint issue by falling back to a deterministic “empty mask” submission when no compatible checkpoints are available at `CKPT_DIR`, which guarantees a valid `submission.csv` is produced (this is score-poor but unblocks submission generation). Finally, I make the optional visualization and submission-building cells robust to earlier failures by ensuring required variables are always defined.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is coming from always falling back to an all-empty submission because `CKPT_DIR` doesn’t exist, so no model weights are loaded and every mask is predicted as zeros. To move the score toward your ~0.588 target with minimal disruption, I keep your exact inference pipeline and UNet fallback, but add a tiny, legitimate training step that trains the same UNetFallback on the provided `train.csv` masks (decoded from RLE) for a small, deterministic subset and then uses that trained checkpoint for test inference. I also fix one correctness issue in `mask2rle` by forcing Fortran-order flattening (the competition’s required convention), which can materially improve score without changing model semantics. Finally, I keep the submission format/paths unchanged and ensure the script always writes `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most likely caused by producing invalid (non-binary / non-thresholded) RLE strings because `mask2rle()` currently ignores `thr` and encodes *all* pixel transitions (including zeros/ones mixes) rather than encoding a proper binary mask; this can collapse the leaderboard score to 0 even if the model predicts something. I make the smallest correctness fix: apply thresholding inside `mask2rle()` (still Fortran-order) and ensure it returns `""` for an empty mask, which matches the competition’s expectations. To move the score upward toward your ~0.588 target without changing the core modeling approach, I also make the fallback training consistent with the test preprocessing by resizing (not cropping) both train and test to a fixed size and then resizing predictions back to the original image size before RLE; this avoids the current “center-crop then pad with zeros” artifact that systematically erases anatomy near borders. Everything else (UNetFallback, BCEWithLogitsLoss training loop, sigmoid+threshold inference, submission schema/path) stays the same.'
- What this solution (achieved 0.0) has done: 'I fix the image-path merge failure by parsing the `id` slice correctly (it’s zero-padded like `slice_0001`, but your current metadata parses it as `1`), and by extracting the slice number from filenames correctly (your current `path2info` mistakenly treats the first token as slice when it’s actually width). Next, I make the fallback training robust when no training images can be merged (so it won’t crash with `num_samples=0`) and cleanly fall back to an empty-mask submission if needed. Finally, I ensure inference always defines `pred_ids/pred_strings/...` so submission creation cannot NameError, and I keep the existing UNet + sigmoid+threshold + RLE semantics unchanged.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the model never producing meaningful masks (either because training is too short/too little data, or because the train IDs are not correctly matched to image paths so training is effectively on nothing/near-nothing). I keep your exact UNetFallback + BCEWithLogitsLoss training loop and the same sigmoid+threshold+RLE inference, but (1) fix `path2info()` slice parsing (it currently reads the width as slice), (2) make the train/test merge use the same correct slice key so training actually sees correct images, and (3) scale the fallback training up just enough (still deterministic, no new methods) so the score moves upward toward your ~0.588 target rather than staying near empty-mask behavior. I also ensure the RLE encoding remains competition-correct (Fortran order, empty string for empty mask), and keep output as `submission.csv` with the required columns.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most likely coming from an invalid id↔image merge: `path2info()` is currently parsing the PNG filename tokens incorrectly (it treats width/height as slice), so many/most `image_path` become missing and the pipeline predicts zeros for those rows. I make the smallest correctness fix by parsing `slice` from the actual leading numeric token in the filename and reading `width,height` from tokens 1 and 2, matching the dataset naming convention (`{slice}_{width}_{height}_{...}.png`). Then your fallback training finally see real images/masks, and test inference run on real images instead of blank arrays, which should lift the score toward your target without changing the model, loss, or inference semantics. I also keep the RLE encoding as Fortran-order binary masks (already correct) and ensure `submission.csv` is written as before.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 is most consistent with training almost never happening effectively (too slow/too little signal) and/or producing near-empty masks, even though the RLE formatting looks correct. To move upward toward ~0.588 with minimal disruption, I keep your exact UNetFallback + BCEWithLogitsLoss training loop and the same sigmoid→threshold→RLE inference, but make training actually learn by (1) training on a smaller, balanced set of slices that contain foreground (otherwise the network learns “all background”), and (2) using `pos_weight` in BCE to counter extreme class imbalance without changing the loss type. I also make the threshold used for encoding consistent with `CFG.thr` (you were using `thr=0.5` inside `masks2rles`, which can unnecessarily zero-out predictions). Everything still runs end-to-end and writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly because the script is still producing (near-)empty predictions: the fallback training is weak for this task, and you additionally average probabilities and then threshold, which tends to suppress small structures. To move the score upward toward your 0.588 target without changing the core model/loss/training loop, I make two minimal, metric-relevant fixes: (1) train longer but on fewer images (so training actually completes and learns within the time budget), and (2) ensemble by averaging logits (not post-sigmoid probabilities), which is standard for BCEWithLogits-trained models and improves calibration while keeping the same architecture and thresholding semantics. I also ensure the threshold used for binarization/RLE is applied exactly once (in `infer`) and keep `mask2rle` strictly binary/Fortran-order as required. Everything still runs end-to-end and writes a valid `submission.csv`.'

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
    "Skipping segmentation_models_pytorch wheel install; using a local PyTorch UNet fallback if SMP is unavailable."
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

SMP_AVAILABLE = False
try:
    import segmentation_models_pytorch as smp  # noqa: F401

    SMP_AVAILABLE = True
    print("segmentation_models_pytorch available:", True)
except Exception as e:
    print(
        "segmentation_models_pytorch not available; using local UNet. Reason:", repr(e)
    )

print("Torch:", torch.__version__, "CUDA available:", torch.cuda.is_available())




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
    df["case"] = df["id"].apply(lambda x: int(str(x).split("_")[0].split("case")[1]))
    df["day"] = df["id"].apply(lambda x: int(str(x).split("_")[1].split("day")[1]))

    def _parse_slice(x):
        tok = str(x).split("_")[-1]  # "slice_0001"
        digits = "".join([c for c in tok if c.isdigit()])
        return int(digits) if digits else -1

    df["slice"] = df["id"].apply(_parse_slice)
    return df


def path2info(row: pd.Series) -> pd.Series:
    path = row["image_path"]
    fname = os.path.basename(path)
    stem = fname.replace(".png", "")
    parts = stem.split("_")

    try:
        slice_ = int(parts[0])
    except Exception:
        digits = "".join([c for c in str(parts[0]) if c.isdigit()])
        slice_ = int(digits) if digits else -1

    try:
        width = int(parts[1])
        height = int(parts[2])
    except Exception:
        width, height = -1, -1

    case_day_dir = os.path.basename(os.path.dirname(path))  # "scans"
    if case_day_dir == "scans":
        case_day = os.path.basename(
            os.path.dirname(os.path.dirname(path))
        )  # caseXXX_dayYY
        case_dir = os.path.basename(
            os.path.dirname(os.path.dirname(os.path.dirname(path)))
        )  # caseXXX
    else:
        case_day = case_day_dir
        case_dir = os.path.basename(os.path.dirname(os.path.dirname(path)))

    try:
        case = int(case_dir.replace("case", ""))
    except Exception:
        case = int(case_day.split("_")[0].replace("case", ""))

    day = int(case_day.split("_")[1].replace("day", ""))

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

req_cols = ["case", "day", "slice", "width", "height"]
missing_cols = [c for c in req_cols if c not in path_df.columns]
if missing_cols:
    raise RuntimeError(f"path_df missing required columns: {missing_cols}")

path_df = path_df[path_df["slice"] >= 0].reset_index(drop=True)

print("Found images:", len(path_df))
path_df.head()



## === cell 14
test_df = sub_df.merge(path_df, on=["case", "day", "slice"], how="left")
missing = test_df["image_path"].isna().sum()
print("Merged test_df:", test_df.shape, "missing image_path:", missing)

if missing > 0:
    print(
        "Warning: Some ids could not be matched to image paths; will keep them with empty predictions."
    )
test_df.head()




## === cell 15
class TestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transforms=None, label=None):
        self.df = df.reset_index(drop=True)
        self.label = label
        self.img_paths = (
            df["image_path"].tolist()
            if "image_path" in df.columns
            else [None] * len(df)
        )
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

        if (
            img_path is None
            or (isinstance(img_path, float) and np.isnan(img_path))
            or (isinstance(img_path, str) and img_path.strip() == "")
        ):
            img = np.zeros((CFG.img_size[0], CFG.img_size[1], 3), dtype=np.uint8)
            h, w = CFG.img_size
        else:
            img = np.array(load_image(img_path))
            h, w = img.shape[:2]

        if self.label:
            msk_path = self.msk_paths[index]
            msk = np.load(msk_path).astype("float32") / 255.0
            if self.transforms:
                data = self.transforms(image=img, mask=msk)
                img = data["image"]
                msk = data["mask"]
            if not isinstance(img, torch.Tensor):
                img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
            if not isinstance(msk, torch.Tensor):
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
            A.Resize(CFG.img_size[0], CFG.img_size[1], interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=COLOR_MEAN, std=COLOR_STD, max_pixel_value=255),
            ToTensorV2(transpose_mask=True),
        ],
        p=1.0,
    )
}




## === cell 17
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
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        if diffY != 0 or diffX != 0:
            x1 = nn.functional.pad(
                x1,
                [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2],
            )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNetFallback(nn.Module):
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
    if SMP_AVAILABLE:
        import segmentation_models_pytorch as smp  # local import

        model = smp.Unet(
            encoder_name=CFG.encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=CFG.num_classes,
        )
        return model
    return UNetFallback(in_channels=3, classes=CFG.num_classes, base=32)


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
    msk = np.asarray(msk)
    if msk.dtype != np.uint8:
        msk = (msk > thr).astype(np.uint8)
    else:
        msk = (msk > 0).astype(np.uint8)

    if msk.max() == 0:
        return ""

    pixels = msk.flatten(order="F")
    pad = np.array([0], dtype=pixels.dtype)
    pixels = np.concatenate([pad, pixels, pad])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def masks2rles(msks, ids, heights, widths, thr=CFG.thr):
    pred_strings = []
    pred_ids = []
    pred_classes = []
    for idx in range(msks.shape[0]):
        height = int(heights[idx])
        width = int(widths[idx])

        msk_resized = cv2.resize(
            msks[idx].astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )

        rle = [None] * 3
        for midx in [0, 1, 2]:
            rle[midx] = mask2rle(msk_resized[..., midx], thr=thr)
        pred_strings.extend(rle)
        pred_ids.extend([ids[idx]] * 3)
        pred_classes.extend(["large_bowel", "small_bowel", "stomach"])
    return pred_strings, pred_ids, pred_classes




## === cell 19
def rle_decode(mask_rle: str, shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    if mask_rle is None:
        return np.zeros((h, w), dtype=np.uint8)
    s = str(mask_rle)
    if s.strip() == "" or s.strip().lower() == "nan":
        return np.zeros((h, w), dtype=np.uint8)
    rle = np.asarray(s.split(), dtype=int)
    starts = rle[0::2] - 1
    lengths = rle[1::2]
    ends = starts + lengths
    img = np.zeros(h * w, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape((w, h)).T


class TrainDataset(Dataset):
    def __init__(self, df_img: pd.DataFrame, df_masks: pd.DataFrame, transforms=None):
        self.df_img = df_img.reset_index(drop=True)
        self.transforms = transforms

        self.seg_map = {
            (r["id"], r["class"]): r["segmentation"] for _, r in df_masks.iterrows()
        }
        self.classes = ["large_bowel", "small_bowel", "stomach"]

    def __len__(self):
        return len(self.df_img)

    def __getitem__(self, idx):
        row = self.df_img.iloc[idx]
        img_path = row["image_path"]
        img = np.array(load_image(img_path))
        h0, w0 = img.shape[:2]

        mask = np.zeros((h0, w0, 3), dtype=np.uint8)
        for cidx, c in enumerate(self.classes):
            rle = self.seg_map.get((row["id"], c), "")
            mask[..., cidx] = rle_decode(rle, (h0, w0))

        if self.transforms:
            data = self.transforms(image=img, mask=mask)
            img_t = data["image"]
            msk_t = data["mask"]
        else:
            img_t = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
            msk_t = torch.from_numpy(np.transpose(mask, (2, 0, 1))).float()

        if isinstance(msk_t, torch.Tensor):
            msk_t = (msk_t > 0.5).float()
        else:
            msk_t = torch.from_numpy(np.transpose(msk_t, (2, 0, 1))).float()
            msk_t = (msk_t > 0.5).float()

        return img_t, msk_t


train_transforms = {
    "train": A.Compose(
        [
            A.Resize(CFG.img_size[0], CFG.img_size[1], interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=COLOR_MEAN, std=COLOR_STD, max_pixel_value=255),
            ToTensorV2(transpose_mask=True),
        ],
        p=1.0,
    )
}


def seed_everything(seed=42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _select_training_subset(
    train_img_df: pd.DataFrame, train_csv: pd.DataFrame, max_images: int
) -> pd.DataFrame:
    df = train_img_df.copy()

    seg_nonempty = (
        train_csv.assign(
            _nonempty=train_csv["segmentation"].fillna("").astype(str).str.len() > 0
        )
        .groupby("id")["_nonempty"]
        .any()
        .rename("has_fg")
        .reset_index()
    )
    df = df.merge(seg_nonempty, on="id", how="left")
    df["has_fg"] = df["has_fg"].fillna(False)

    fg = df[df["has_fg"]].copy()
    bg = df[~df["has_fg"]].copy()

    fg = fg.sort_values(["case", "day", "slice"]).reset_index(drop=True)
    bg = bg.sort_values(["case", "day", "slice"]).reset_index(drop=True)

    fg_target = min(len(fg), int(max_images * 0.75))
    bg_target = min(len(bg), max_images - fg_target)

    out = pd.concat([fg.iloc[:fg_target], bg.iloc[:bg_target]], axis=0).reset_index(
        drop=True
    )
    out = out.sort_values(["case", "day", "slice"]).reset_index(drop=True)
    return out.drop(columns=["has_fg"], errors="ignore")


def train_fallback_model_if_needed() -> List[str]:
    model_paths = sorted(glob(f"{CKPT_DIR}/best_epoch*.bin"))
    if len(model_paths) > 0:
        print("Using existing checkpoints from CKPT_DIR:", CKPT_DIR)
        return model_paths

    seed_everything(CFG.seed)

    train_csv = pd.read_csv(TRAIN_FILE_CSV_PATH)
    train_csv = train_csv.dropna(subset=["id", "class"]).reset_index(drop=True)

    train_ids = train_csv[["id"]].drop_duplicates()
    train_ids = get_metadata(train_ids)

    train_paths = glob(BASE_PATH + "/train/**/*png", recursive=True)
    train_path_df = pd.DataFrame(train_paths, columns=["image_path"])
    if hasattr(train_path_df, "progress_apply"):
        train_path_df = train_path_df.progress_apply(path2info, axis=1)
    else:
        train_path_df = train_path_df.apply(path2info, axis=1)
    train_path_df = train_path_df[train_path_df["slice"] >= 0].reset_index(drop=True)

    train_img_df = train_ids.merge(
        train_path_df, on=["case", "day", "slice"], how="left"
    )
    train_img_df = train_img_df.dropna(subset=["image_path"]).reset_index(drop=True)

    if len(train_img_df) == 0:
        print(
            "Warning: No training images matched to train.csv ids; skipping training and using empty-mask fallback."
        )
        return []

    max_train_images = 1800
    train_img_df = _select_training_subset(
        train_img_df, train_csv, max_images=max_train_images
    )

    print(
        "Training fallback model on images:",
        len(train_img_df),
        "from train.csv rows:",
        len(train_csv),
    )

    train_ds = TrainDataset(
        train_img_df, train_csv, transforms=train_transforms["train"]
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.train_batch_size,
        num_workers=2,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    model = UNetFallback(in_channels=3, classes=CFG.num_classes, base=32).to(CFG.device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )

    pos_weight = torch.tensor(
        [6.0, 6.0, 6.0], device=CFG.device, dtype=torch.float32
    ).view(1, 3, 1, 1)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    train_epochs = 12
    for ep in range(train_epochs):
        pbar = tqdm(train_loader, desc=f"TrainFallback ep {ep+1}/{train_epochs}")
        for img, msk in pbar:
            img = img.to(CFG.device, dtype=torch.float32)
            msk = msk.to(CFG.device, dtype=torch.float32)

            logits = model(img)
            loss = bce(logits, msk)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            pbar.set_postfix(loss=float(loss.detach().cpu().item()))

    ckpt_path = "fallback_unet.bin"
    torch.save(model.state_dict(), ckpt_path)
    print("Saved fallback checkpoint:", os.path.abspath(ckpt_path))
    return [ckpt_path]




## === cell 20
@torch.no_grad()
def infer(model_paths, test_loader, num_log=1, thr=CFG.thr):
    msks_log = []
    imgs_log = []
    pred_strings = []
    pred_ids = []
    pred_classes = []

    if len(model_paths) == 0:
        for bidx, (img, ids, heights, widths) in enumerate(
            tqdm(test_loader, total=len(test_loader), desc="Infer(empty-fallback)")
        ):
            bs = img.shape[0]
            msk_np = np.zeros((bs, CFG.img_size[0], CFG.img_size[1], 3), dtype=np.uint8)
            result = masks2rles(msk_np, ids, heights, widths, thr=thr)
            pred_strings.extend(result[0])
            pred_ids.extend(result[1])
            pred_classes.extend(result[2])
            if bidx < num_log:
                imgs_log.append(img.permute((0, 2, 3, 1)).cpu().numpy()[:10])
                msks_log.append(msk_np[:10])
        return pred_strings, pred_ids, pred_classes, imgs_log, msks_log

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
        logits_ens = torch.zeros(
            (size[0], 3, size[2], size[3]), device=CFG.device, dtype=torch.float32
        )

        for m in models:
            out_logits = m(img)
            logits_ens += out_logits / len(models)

        prob = sigmoid(logits_ens)
        msk_np = (prob.permute((0, 2, 3, 1)) > thr).to(torch.uint8).cpu().numpy()
        result = masks2rles(msk_np, ids, heights, widths, thr=thr)
        pred_strings.extend(result[0])
        pred_ids.extend(result[1])
        pred_classes.extend(result[2])

        if bidx < num_log:
            imgs_log.append(img.permute((0, 2, 3, 1)).cpu().numpy()[:10])
            msks_log.append(msk_np[:10])

        del img, logits_ens, out_logits, prob, result, msk_np
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pred_strings, pred_ids, pred_classes, imgs_log, msks_log




## === cell 21
test_dataset = TestDataset(test_df, transforms=test_transforms["test"])
test_loader = DataLoader(
    test_dataset,
    batch_size=CFG.val_batch_size,
    num_workers=2,
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
)

model_paths = train_fallback_model_if_needed()
print("Num model checkpoints:", len(model_paths), "paths:", model_paths[:3])

pred_strings, pred_ids, pred_classes, imgs, msks = infer(model_paths, test_loader)

print("Pred rows:", len(pred_strings))



## === cell 22
if "imgs" in globals() and isinstance(imgs, list) and len(imgs) > 0:
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



## === cell 23
try:
    del imgs, msks
except Exception:
    pass
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()



## === cell 24
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
print("submission.csv path:", os.path.abspath("submission.csv"))
