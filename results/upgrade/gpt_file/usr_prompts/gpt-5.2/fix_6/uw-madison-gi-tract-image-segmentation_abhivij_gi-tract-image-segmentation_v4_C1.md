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

3.13

# 3. Installed packages

No external packages required in the script and installed.

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

0.8184877883067355

# 6. Current score

None

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import socket


def internet_on(host="8.8.8.8", port=53, timeout=3):
    """
    Host: 8.8.8.8 (Google DNS)
    Open socket to test connectivity
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except Exception:
        return False


if internet_on():
    try:
        import segmentation_models_pytorch  # noqa: F401
    except Exception:
        pass
    try:
        import monai  # noqa: F401
    except Exception:
        pass
else:
    print("Internet off - Relying on preinstalled dependencies (no pip installs).")



## === cell 1
import os
import random
from glob import glob

import numpy as np
import pandas as pd

try:
    import cv2  # type: ignore

    CV2_AVAILABLE = True
except Exception:
    cv2 = None
    CV2_AVAILABLE = False

try:
    from PIL import Image
except Exception as e:
    raise RuntimeError(
        "PIL (Pillow) is required for fallback image IO but is missing."
    ) from e

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.colors import ListedColormap
    import seaborn as sns
except Exception:
    plt = None
    Rectangle = None
    ListedColormap = None
    sns = None

try:
    from tqdm.notebook import tqdm
except Exception:
    from tqdm import tqdm

try:
    tqdm.pandas()
except Exception:
    pass

from sklearn.model_selection import StratifiedGroupKFold

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

try:
    import albumentations as A  # type: ignore
    from albumentations.pytorch import ToTensorV2  # type: ignore

    ALBU_AVAILABLE = True
except Exception:
    A = None
    ToTensorV2 = None
    ALBU_AVAILABLE = False

try:
    import segmentation_models_pytorch as smp  # type: ignore

    SMP_AVAILABLE = True
except Exception:
    smp = None
    SMP_AVAILABLE = False

MONAI_AVAILABLE = False


def _imread_gray_float01(path: str) -> np.ndarray:
    """Read a grayscale image as float32 in [0,1]. Uses cv2 if available; otherwise PIL."""
    if CV2_AVAILABLE:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"cv2.imread failed for: {path}")
        img = img.astype("float32")
    else:
        im = Image.open(path).convert("I")  # 32-bit signed integer pixels
        img = np.array(im, dtype=np.float32)

    mx = float(np.max(img)) if img.size else 0.0
    if mx > 0:
        img = img / mx
    return img.astype(np.float32)


def _resize_nearest(img: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Nearest neighbor resize for HxW or HxWxC."""
    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    if CV2_AVAILABLE:
        if img.ndim == 2:
            return cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        return cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    if img.ndim == 2:
        im = Image.fromarray(img)
        im = im.resize((out_w, out_h), resample=Image.NEAREST)
        return np.array(im)
    chs = []
    for c in range(img.shape[2]):
        im = Image.fromarray(img[..., c])
        im = im.resize((out_w, out_h), resample=Image.NEAREST)
        chs.append(np.array(im))
    return np.stack(chs, axis=-1)


class _ComposeFallback:
    """Minimal albumentations-like Compose that supports Resize + Normalize + ToTensor."""

    def __init__(self, resize_hw, mean, std, max_pixel_value=1.0, transpose_mask=True):
        self.resize_hw = resize_hw
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.max_pixel_value = float(max_pixel_value)
        self.transpose_mask = bool(transpose_mask)

    def __call__(self, image, mask=None):
        image = _resize_nearest(image, tuple(self.resize_hw)).astype(np.float32)
        if mask is not None:
            mask = _resize_nearest(mask, tuple(self.resize_hw)).astype(np.uint8)

        image = image / self.max_pixel_value
        image = (image - self.mean) / self.std

        if image.ndim == 2:
            image = image[..., None]
        image_t = torch.from_numpy(image.transpose(2, 0, 1)).float().contiguous()

        out = {"image": image_t}
        if mask is not None:
            if self.transpose_mask and mask.ndim == 3:
                mask_t = torch.from_numpy(mask.transpose(2, 0, 1)).long().contiguous()
            else:
                mask_t = torch.from_numpy(mask).long().contiguous()
            out["mask"] = mask_t
        return out


class _FallbackSMP:
    class Unet(nn.Module):
        def __init__(
            self, encoder_name=None, encoder_weights=None, in_channels=3, classes=3
        ):
            super().__init__()
            self.enc1 = nn.Sequential(
                nn.Conv2d(in_channels, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            self.pool1 = nn.MaxPool2d(2)
            self.enc2 = nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
            self.pool2 = nn.MaxPool2d(2)

            self.bottleneck = nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )

            self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
            self.dec2 = nn.Sequential(
                nn.Conv2d(128, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
            self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
            self.dec1 = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )

            self.out_conv = nn.Conv2d(32, classes, kernel_size=1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool1(e1))
            b = self.bottleneck(self.pool2(e2))

            d2 = self.up2(b)
            d2 = torch.cat([d2, e2], dim=1)
            d2 = self.dec2(d2)

            d1 = self.up1(d2)
            d1 = torch.cat([d1, e1], dim=1)
            d1 = self.dec1(d1)

            return self.out_conv(d1)

    class losses:
        class SoftBCEWithLogitsLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.bce = nn.BCEWithLogitsLoss()

            def forward(self, y_pred, y_true):
                return self.bce(y_pred, y_true)

        class DiceLoss(nn.Module):
            def __init__(self, mode="multilabel", eps=1e-6):
                super().__init__()
                self.eps = eps

            def forward(self, y_pred, y_true):
                y_pred = torch.sigmoid(y_pred)
                dims = (2, 3)
                intersection = (y_pred * y_true).sum(dims)
                union = y_pred.sum(dims) + y_true.sum(dims)
                dice = (2.0 * intersection + self.eps) / (union + self.eps)
                return 1.0 - dice.mean()


if not SMP_AVAILABLE:
    smp = _FallbackSMP()
    print(
        "segmentation_models_pytorch not found; using local fallback UNet + losses (score may differ)."
    )



## === cell 2
print(f"Number of available CPUs: {os.cpu_count()}")
print(f"Number of available GPUs: {torch.cuda.device_count()}")



## === cell 3
DIR_PATH = "/kaggle/input/uw-madison-gi-tract-image-segmentation/"

pd.set_option("display.max_colwidth", 400)

if ListedColormap is not None:
    CMAP1 = ListedColormap([[0, 0, 0, 0], [1, 0, 0, 1]])
    CMAP2 = ListedColormap([[0, 0, 0, 0], [0, 1, 0, 1]])
    CMAP3 = ListedColormap([[0, 0, 0, 0], [0, 0, 1, 1]])
else:
    CMAP1 = CMAP2 = CMAP3 = None

RANDOM_SEED = 0

IMAGE_NORMALIZE_MEAN = (0.485, 0.456, 0.406)
IMAGE_NORMALIZE_SD = (0.229, 0.224, 0.225)

IMAGE_RESIZE = [288, 288]

BATCH_SIZE_TRAIN = 32
BATCH_SIZE_VALID = BATCH_SIZE_TRAIN * 2
BATCH_SIZE_TEST = BATCH_SIZE_TRAIN * 2

DATA_LOADER_NUM_WORKERS = 4

NUM_CLASSES = 3
CLASS_NAMES = ["large_bowel", "small_bowel", "stomach"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 5

MODEL_PARAMS_FILE_NAME = "GIT-Seg-efficientnet-b1.pth"
MODEL_PARAMS_LOAD_FILE_PATH = (
    "/kaggle/input/git-seg/pytorch/default/2/GIT-Seg-efficientnet-b1.pth"
)

TRAIN_VALID_SPLIT = True
TEST_PREDICT = True

SAVE_TRAIN_VALID_MODEL = True
LOAD_MODEL_FOR_TEST_PREDICT = True



## === cell 4
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 5
data = pd.read_csv(DIR_PATH + "train.csv")
data.head()



## === cell 6
data_nonnaseg = data.loc[data.segmentation.notna(), :]
data_nonnaseg.head()



## === cell 7
data[["case", "day", "slice"]] = data["id"].str.extract(
    r"case(\d+)_day(\d+)_slice_(\d+)"
)
data




## === cell 8
def get_path_df(train=True):
    base = "train" if train else "test"
    paths = glob(os.path.join(DIR_PATH, base, "case*/case*_day*/scans/*.png"))
    path_df = pd.DataFrame(paths, columns=["image_path"])

    path_df[["case", "day", "slice", "slice_w", "slice_h", "px_w", "px_h"]] = (
        path_df.image_path.str.extract(
            r".*/case(\d+)_day(\d+)/scans/slice_(\d+)_(\d+)_(\d+)_([0-9]+(?:\.[0-9]+)?)_([0-9]+(?:\.[0-9]+)?)\.png"
        )
    )
    return path_df


path_df = get_path_df()



## === cell 9
data.info()



## === cell 10
path_df.info()



## === cell 11
data = data.merge(path_df, on=["case", "day", "slice"], how="left")
data



## === cell 12
data.info()



## === cell 13
data.px_w.unique(), data.px_h.unique()



## === cell 14
data.case.unique(), data.day.unique(), data.slice.unique(), data.slice_w.unique(), data.slice_h.unique()



## === cell 15
int_cols = ["case", "day", "slice", "slice_w", "slice_h"]
for c in int_cols:
    data[c] = pd.to_numeric(data[c], errors="coerce")
data[int_cols] = data[int_cols].fillna(0).astype(np.uint32)

float_cols = ["px_w", "px_h"]
for c in float_cols:
    data[c] = pd.to_numeric(data[c], errors="coerce")
data[float_cols] = data[float_cols].fillna(0).astype(np.float32)

data.info()




## === cell 16
def rle_decode(mask_rle, shape):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height,width) of array to return

    Competition pixels are numbered from top-to-bottom, then left-to-right.
    That corresponds to flattening in Fortran order (column-major) on (H, W).
    """
    s = np.asarray(mask_rle.split(), dtype=int)
    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")


def rle_encode(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted.
    Competition expects empty string for empty mask.

    IMPORTANT: encode using Fortran order (column-major) to match the competition.
    """
    img = (img > 0).astype(np.uint8)

    pixels = img.flatten(order="F")
    if pixels.max() == 0:
        return ""
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 17
def get_mask(id_, data_):
    data_subset_id = data_.loc[data_["id"] == id_]
    if len(data_subset_id) == 0:
        return None
    slice_dim = data_subset_id[["slice_h", "slice_w"]].iloc[0]
    shape = (int(slice_dim.slice_h), int(slice_dim.slice_w), 3)
    mask = np.zeros(shape, dtype=np.uint8)
    for i, class_ in enumerate(CLASS_NAMES):
        data_subset_class = data_subset_id[data_subset_id["class"] == class_]
        if len(data_subset_class) == 0:
            continue
        rle = data_subset_class.segmentation.squeeze()
        if not pd.isna(rle):
            mask[..., i] = rle_decode(rle, shape[:2])
    return mask




## === cell 18
example_path = path_df["image_path"].iloc[0] if len(path_df) else None
print("Example image path:", example_path)
print(
    "CV2_AVAILABLE:",
    CV2_AVAILABLE,
    "| ALBU_AVAILABLE:",
    ALBU_AVAILABLE,
    "| SMP_AVAILABLE:",
    SMP_AVAILABLE,
)




## === cell 19
def load_image(id_, data_):
    data_subset = data_.loc[data_.id == id_]
    if len(data_subset) == 0:
        raise KeyError(f"id not found in dataframe: {id_}")
    p = data_subset.image_path.iloc[0]
    if not isinstance(p, str) or (not os.path.exists(p)):
        raise FileNotFoundError(f"image_path missing for id={id_}: {p}")
    return _imread_gray_float01(p)




## === cell 20
def display_image(
    id_,
    data_,
    pred_mask=None,
    apply_CLAHE=False,
    show_orig_img=True,
    show_true_mask=True,
    show_pred_mask=False,
):
    if plt is None:
        print("Matplotlib not available; skipping display.")
        return

    img = load_image(id_, data_)
    img_u8 = (img * 255).astype(np.uint8)

    if apply_CLAHE and CV2_AVAILABLE:
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2, 2))
        img_u8 = clahe.apply(img_u8)

    mask = get_mask(id_, data_)

    plt.figure(figsize=(9, 3))

    i = 1
    if show_orig_img:
        plt.subplot(1, 3, i)
        i += 1
        plt.imshow(img_u8, cmap="bone")
        plt.title(f"{id_} image")
        plt.axis("off")

    if show_true_mask and mask is not None:
        plt.subplot(1, 3, i)
        i += 1
        plt.imshow(img_u8, cmap="bone")
        plt.title("Image with true mask")
        if CMAP1 is not None:
            plt.imshow(mask[..., 0], cmap=CMAP1)
            plt.imshow(mask[..., 1], cmap=CMAP2)
            plt.imshow(mask[..., 2], cmap=CMAP3)
        plt.axis("off")

        if Rectangle is not None and CMAP1 is not None:
            handles = [
                Rectangle((0, 0), 1, 1, color=CMAP1(1.0)),
                Rectangle((0, 0), 1, 1, color=CMAP2(1.0)),
                Rectangle((0, 0), 1, 1, color=CMAP3(1.0)),
            ]
            labels = ["Large Bowel", "Small Bowel", "Stomach"]
            plt.legend(
                handles,
                labels,
                bbox_to_anchor=(1.0, -0.4),
                loc="lower right",
                borderaxespad=0.0,
            )

    if show_pred_mask and pred_mask is not None:
        plt.subplot(1, 3, i)
        plt.imshow(img_u8, cmap="bone")
        plt.title("Image with predicted mask")
        if CMAP1 is not None:
            plt.imshow(pred_mask[..., 0], cmap=CMAP1)
            plt.imshow(pred_mask[..., 1], cmap=CMAP2)
            plt.imshow(pred_mask[..., 2], cmap=CMAP3)
        plt.axis("off")

    plt.tight_layout()
    plt.show()




## === cell 21
some_id = data["id"].iloc[0]
display_image(some_id, data)



## === cell 22
display_image(some_id, data, apply_CLAHE=True)



## === cell 23
example_ids = data["id"].unique()[:3]
for _id in example_ids:
    display_image(_id, data, apply_CLAHE=True)



## === cell 24
for _id in example_ids:
    display_image(_id, data, apply_CLAHE=True)




## === cell 25
def display_multiple_slices(
    id_array, data_, apply_CLAHE=False, show_pred_mask=False, pred_mask_array=None
):
    if plt is None:
        print("Matplotlib not available; skipping display.")
        return

    l = len(id_array)
    if l == 0:
        print("No ids to display.")
        return
    rows = int(np.ceil(l / 5))
    max_cols = 5
    data_subset = data_.loc[data_.id.isin(id_array), :].copy()

    plt.figure(figsize=(max_cols * 3, rows * 3))

    for i in range(l):
        id_ = id_array[i]
        data_one = data_subset.loc[data_subset.id == id_]
        if len(data_one) == 0:
            continue
        p = data_one.image_path.iloc[0]
        if not isinstance(p, str) or (not os.path.exists(p)):
            continue

        img = _imread_gray_float01(p)
        if apply_CLAHE and CV2_AVAILABLE:
            clahe_local = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2, 2))
            img_u8 = (img * 255).astype(np.uint8)
            img_u8 = clahe_local.apply(img_u8)
            img_disp = img_u8
        else:
            img_disp = img

        if show_pred_mask and pred_mask_array is not None:
            mask = pred_mask_array[i]
        else:
            mask = get_mask(id_, data_)

        plt.subplot(rows, max_cols, i + 1)
        plt.imshow(img_disp, cmap="bone")
        plt.title(id_)
        if mask is not None and CMAP1 is not None:
            plt.imshow(mask[..., 0], cmap=CMAP1)
            plt.imshow(mask[..., 1], cmap=CMAP2)
            plt.imshow(mask[..., 2], cmap=CMAP3)
        plt.axis("off")

    plt.tight_layout()
    plt.show()




## === cell 26
display_multiple_slices(
    data.query("case == 123 and day == 20 and slice >= 63 and slice <= 70").id.unique(),
    data,
    apply_CLAHE=True,
)



## === cell 27
display_multiple_slices(
    data.query("case == 131 and day == 0 and slice > 55 and slice <= 70").id.unique(),
    data,
    apply_CLAHE=True,
)



## === cell 28
data.loc[data.segmentation.isna(), :].head()



## === cell 29
data.isna().sum()



## === cell 30
print(
    f"Num cases : {len(data.case.unique())}         Num unique days : {len(data.day.unique())}           Num unique slices : {len(data.slice.unique())}"
)



## === cell 31
count_df = (
    data[["id", "slice_w", "slice_h"]]
    .drop_duplicates()[["slice_w", "slice_h"]]
    .value_counts()
    .reset_index(name="count")
)
count_df["percent"] = count_df["count"] * 100 / sum(count_df["count"])
print(sum(count_df["count"]))
count_df



## === cell 32
count_df = (
    data[["id", "px_w", "px_h"]]
    .drop_duplicates()[["px_w", "px_h"]]
    .value_counts()
    .reset_index(name="count")
)
count_df["percent"] = count_df["count"] * 100 / sum(count_df["count"])
print(sum(count_df["count"]))
count_df



## === cell 33
if sns is not None and plt is not None:
    day_dist = (
        data[["case", "day"]]
        .drop_duplicates()["case"]
        .value_counts()
        .reset_index(name="num_days")
    )

    sns.histplot(
        data=day_dist,
        x="num_days",
        bins=range(1, int(day_dist["num_days"].max()) + 1),
        discrete=True,
    )
    plt.xlabel("Number of Days per Case")
    plt.ylabel("Number of Cases")
    plt.title("Distribution of Days per Case")
    plt.show()



## === cell 34
if sns is not None and plt is not None:
    slice_dist = (
        data[["case", "day", "slice"]]
        .drop_duplicates()[["case", "day"]]
        .value_counts()
        .reset_index(name="num_slices")
    )

    sns.histplot(
        data=slice_dist,
        x="num_slices",
        bins=range(1, int(slice_dist["num_slices"].max()) + 1),
        discrete=True,
    )
    plt.xlabel("Number of slices per case-days")
    plt.ylabel("Number of specific case-days")
    plt.title("Distribution of slices per case-day")
    plt.show()



## === cell 35
slice_dist = (
    data[["case", "day", "slice"]]
    .drop_duplicates()[["case", "day"]]
    .value_counts()
    .reset_index(name="num_slices")
)
slice_dist.loc[slice_dist.num_slices == 80, :]



## === cell 36
case_day_slice_df = data[
    ["case", "day", "slice", "slice_w", "slice_h"]
].drop_duplicates()
case_day_slice_df.merge(case_day_slice_df, on=["case", "day"]).query(
    "(slice_w_x != slice_w_y) | (slice_h_x != slice_h_y)"
)



## === cell 37
case_day_slice_df = data[
    ["case", "day", "slice", "slice_w", "slice_h"]
].drop_duplicates()
case_day_slice_df.merge(case_day_slice_df, on=["case"]).query(
    "(slice_w_x != slice_w_y) | (slice_h_x != slice_h_y)"
)



## === cell 38
case_day_slice_df = data[["case", "day", "slice", "px_w", "px_h"]].drop_duplicates()
case_day_slice_df.merge(case_day_slice_df, on=["case", "day"]).query(
    "(px_w_x != px_w_y) | (px_h_x != px_h_y)"
)



## === cell 39
case_day_slice_df = data[["case", "day", "slice", "px_w", "px_h"]].drop_duplicates()
case_day_slice_df.merge(case_day_slice_df, on=["case"]).query(
    "(px_w_x != px_w_y) | (px_h_x != px_h_y)"
)



## === cell 40
num_missing_seg_masks = data.segmentation.isna().sum()
print(
    f"Missing Seg Mask \n count = {num_missing_seg_masks}\n percentage = {num_missing_seg_masks/len(data)*100}"
)



## === cell 41
data["class"].value_counts()



## === cell 42
if sns is not None and plt is not None:
    na_counts = (
        data.groupby("class")["segmentation"]
        .apply(lambda s: s.isna().sum())
        .reset_index(name="count")
    )
    na_counts["percent"] = (
        100 * na_counts["count"] / data.groupby("class")["segmentation"].size().values
    )

    sns.set_style("whitegrid")
    ax = sns.barplot(data=na_counts, x="class", y="percent")

    for i, row in na_counts.iterrows():
        ax.text(
            i,
            row["percent"] + 1,
            f"{row['percent']:.2f}% ({row['count']})",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.ylabel("Percentage")
    plt.xlabel("Segmentation Class")
    plt.title("Missing Segmentation Masks")
    plt.show()



## === cell 43
if sns is not None and plt is not None:
    case_day_seg_missing = (
        data[["case", "day", "class", "segmentation"]]
        .groupby(["case", "day", "class"])["segmentation"]
        .apply(lambda s: s.isna().sum())
        .reset_index(name="count")
        .sort_values(by="count", ascending=False)
    )

    sns.boxplot(
        data=case_day_seg_missing,
        x="class",
        y="count",
    )
    sns.stripplot(
        data=case_day_seg_missing,
        x="class",
        y="count",
        color="black",
        size=3,
        jitter=True,
        alpha=0.4,
    )
    plt.ylabel("Missing Mask Count")
    plt.xlabel("Segmentation Class")
    plt.title("Distribution of Missing Masks per Class (by Case-Day)")
    plt.show()



## === cell 44
for _id in data["id"].unique()[:4]:
    display_image(_id, data, apply_CLAHE=True)



## === cell 45
for _id in data["id"].unique()[4:8]:
    display_image(_id, data, apply_CLAHE=True)



## === cell 46
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
index_train, index_valid = next(
    sgkf.split(data.id, data.segmentation.isna(), data.case)
)



## === cell 47
len(index_train), len(index_valid)



## === cell 48
data_train = data.iloc[index_train, :]
data_valid = data.iloc[index_valid, :]



## === cell 49
data_train.head()



## === cell 50
data_valid.head()



## === cell 51
print(len(data_train.case.unique()), len(data_valid.case.unique()))



## === cell 52
data_train_sub = data_train.loc[data_train.case.isin(data_train.case.unique()[:11]), :]
data_valid_sub = data_valid.loc[data_valid.case.isin(data_valid.case.unique()[:2]), :]

print(
    len(data_train_sub),
    len(data_valid_sub),
    len(data_train_sub) / max(1, len(data_valid_sub)),
)



## === cell 53
missing_masks_train = data_train_sub.segmentation.isna().sum()
missing_masks_valid = data_valid_sub.segmentation.isna().sum()
print(missing_masks_train, missing_masks_train * 100 / len(data_train_sub))
print(missing_masks_valid, missing_masks_valid * 100 / len(data_valid_sub))



## === cell 54
na_counts_train = (
    data_train_sub.groupby("class")["segmentation"]
    .apply(lambda s: s.isna().sum())
    .reset_index(name="count")
)
na_counts_train["percent"] = (
    100
    * na_counts_train["count"]
    / data_train_sub.groupby("class")["segmentation"].size().values
)

na_counts_valid = (
    data_valid_sub.groupby("class")["segmentation"]
    .apply(lambda s: s.isna().sum())
    .reset_index(name="count")
)
na_counts_valid["percent"] = (
    100
    * na_counts_valid["count"]
    / data_valid_sub.groupby("class")["segmentation"].size().values
)

na_counts_train, na_counts_valid



## === cell 55
data_train_sub = data_train_sub.reset_index(drop=True)



## === cell 56
data_valid_sub = data_valid_sub.reset_index(drop=True)




## === cell 57
class GITractDataset(Dataset):
    def __init__(self, df, is_train=True, transforms=None):
        self.df = df
        self.id_ = df["id"].unique()
        self.is_train = is_train
        self.transforms = transforms

    def __len__(self):
        return len(self.id_)

    def __getitem__(self, idx):
        id_ = self.id_[idx]
        img = load_image(id_, self.df)
        img = np.tile(img[..., None], [1, 1, 3])

        if self.is_train:
            mask = get_mask(id_, self.df)
            if mask is None:
                data_sub = self.df.loc[self.df.id == id_]
                height = int(data_sub.slice_h.iloc[0])
                width = int(data_sub.slice_w.iloc[0])
                mask = np.zeros((height, width, NUM_CLASSES), dtype=np.uint8)

            if self.transforms:
                augmented = self.transforms(image=img, mask=mask)
                img = augmented["image"]
                mask = augmented["mask"]
            return img, mask, id_
        else:
            data_sub = self.df.loc[self.df.id == id_]
            height = int(data_sub.slice_h.iloc[0])
            width = int(data_sub.slice_w.iloc[0])
            if self.transforms:
                augmented = self.transforms(image=img)
                img = augmented["image"]
            return img, id_, height, width




## === cell 58
if ALBU_AVAILABLE:
    transform_train = A.Compose(
        [
            A.Resize(
                IMAGE_RESIZE[0],
                IMAGE_RESIZE[1],
                interpolation=cv2.INTER_LINEAR if CV2_AVAILABLE else 1,
                mask_interpolation=cv2.INTER_NEAREST if CV2_AVAILABLE else 0,
            ),
            A.Normalize(
                mean=IMAGE_NORMALIZE_MEAN, std=IMAGE_NORMALIZE_SD, max_pixel_value=1.0
            ),
            ToTensorV2(transpose_mask=True),
        ]
    )

    transform_valid = A.Compose(
        [
            A.Resize(
                IMAGE_RESIZE[0],
                IMAGE_RESIZE[1],
                interpolation=cv2.INTER_LINEAR if CV2_AVAILABLE else 1,
                mask_interpolation=cv2.INTER_NEAREST if CV2_AVAILABLE else 0,
            ),
            A.Normalize(
                mean=IMAGE_NORMALIZE_MEAN, std=IMAGE_NORMALIZE_SD, max_pixel_value=1.0
            ),
            ToTensorV2(transpose_mask=True),
        ]
    )
else:
    transform_train = _ComposeFallback(
        resize_hw=IMAGE_RESIZE,
        mean=IMAGE_NORMALIZE_MEAN,
        std=IMAGE_NORMALIZE_SD,
        max_pixel_value=1.0,
        transpose_mask=True,
    )
    transform_valid = _ComposeFallback(
        resize_hw=IMAGE_RESIZE,
        mean=IMAGE_NORMALIZE_MEAN,
        std=IMAGE_NORMALIZE_SD,
        max_pixel_value=1.0,
        transpose_mask=True,
    )



## === cell 59
dataset_train = GITractDataset(data_train_sub, transforms=transform_train)
dataset_valid = GITractDataset(data_valid_sub, transforms=transform_valid)

dataloader_train = DataLoader(
    dataset_train,
    batch_size=BATCH_SIZE_TRAIN,
    shuffle=True,
    num_workers=DATA_LOADER_NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)
dataloader_valid = DataLoader(
    dataset_valid,
    batch_size=BATCH_SIZE_VALID,
    shuffle=False,
    num_workers=DATA_LOADER_NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)



## === cell 60
dataset = next(iter(dataloader_train))
img, mask, id_ = dataset
print(img.shape, mask.shape, len(id_))



## === cell 61
idx = 0
np.max(img[idx].numpy()), np.min(img[idx].numpy())



## === cell 62
type(img[idx].numpy()[0, 0, 0]), type(mask[idx].numpy()[0, 0, 0])




## === cell 63
def display_dataset(
    dataset_batch,
    display_orig=False,
    num_images=None,
    denormalize=False,
    apply_CLAHE=False,
):
    if plt is None:
        print("Matplotlib not available; skipping display.")
        return

    img_arr, mask_arr, id_arr = dataset_batch
    if num_images is None:
        num_images = len(img_arr)
    max_cols = 5

    if display_orig:
        num_images = 5
        rows = 2
        plt.figure(figsize=(max_cols * 3, rows * 3))
        ids_shown = []
    else:
        rows = int(np.ceil(num_images / max_cols))
        plt.figure(figsize=(max_cols * 3, rows * 3))

    for idx in range(num_images):
        img, mask, id_ = img_arr[idx], mask_arr[idx], id_arr[idx]
        img = img.permute(1, 2, 0)
        if denormalize:
            img = img * torch.tensor(IMAGE_NORMALIZE_SD) + torch.tensor(
                IMAGE_NORMALIZE_MEAN
            )
            img = img.clamp(0, 1)
        img = img.cpu().numpy()
        img = (img * 255).astype(np.uint8)

        mask = mask.permute(1, 2, 0).cpu().numpy()

        plt.subplot(rows, max_cols, idx + 1)
        plt.imshow(img[:, :, 0], cmap="bone")
        plt.title(f"{idx} : {id_}")

        if CMAP1 is not None:
            plt.imshow(mask[..., 0], cmap=CMAP1)
            plt.imshow(mask[..., 1], cmap=CMAP2)
            plt.imshow(mask[..., 2], cmap=CMAP3)
        plt.axis("off")

        if display_orig:
            ids_shown.append(id_)

    plt.tight_layout()
    plt.show()




## === cell 64
display_dataset(dataset, num_images=5, denormalize=True)



## === cell 65
display_dataset(dataset, display_orig=True, denormalize=True, apply_CLAHE=True)



## === cell 66
smp_encoder_weights = (
    None if TEST_PREDICT and LOAD_MODEL_FOR_TEST_PREDICT else "imagenet"
)
model = smp.Unet(
    encoder_name="efficientnet-b1",
    encoder_weights=smp_encoder_weights,
    in_channels=3,
    classes=NUM_CLASSES,
)
model.to(DEVICE)



## === cell 67
optimizer = optim.Adam(model.parameters(), lr=1e-3)



## === cell 68
dice_loss = smp.losses.DiceLoss(mode="multilabel")
BCE_loss = smp.losses.SoftBCEWithLogitsLoss()


def loss_fn(y_pred, y_true, loss_wt=0.5):
    return dice_loss(y_pred, y_true) * loss_wt + BCE_loss(y_pred, y_true) * (
        1 - loss_wt
    )




## === cell 69
class DiceScoreCustom:
    def __init__(self, num_classes, eps=1e-6):
        self.num_classes = num_classes
        self.eps = eps
        self.reset()

    def reset(self):
        self.dice_sum = 0.0
        self.image_count = 0
        self.organ_dice_sum = torch.zeros(self.num_classes)
        self.organ_count = torch.zeros(self.num_classes)

    def update(self, preds, targets):
        I = (targets & preds).sum((2, 3))
        U = (targets | preds).sum((2, 3))
        dice = (2 * I) / (U + I + self.eps)
        non_empty = U > 0
        organ_counts = non_empty.sum(dim=1)
        dice_per_image = dice.sum(dim=1) / organ_counts.clamp(min=1)

        self.dice_sum += dice_per_image.sum().item()
        self.image_count += dice_per_image.numel()
        self.organ_dice_sum += dice.sum(dim=0).detach().cpu()
        self.organ_count += non_empty.sum(dim=0).detach().cpu()

    def compute(self):
        overall = (
            torch.tensor(self.dice_sum / self.image_count)
            if self.image_count > 0
            else torch.tensor(0.0)
        )
        per_organ = torch.where(
            self.organ_count > 0,
            self.organ_dice_sum / self.organ_count,
            torch.tensor(0.0),
        )
        return overall, per_organ




## === cell 70
class HausdorffDistanceCustom:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.h3d_sum = 0.0
        self.image3d_count = 0
        self.organ_h3d_sum = np.zeros(self.num_classes)
        self.organ_count_sum = np.zeros(self.num_classes)

    def update(self, preds, targets):
        return

    def compute(self):
        overall = 0.0
        per_organ = np.zeros(self.num_classes)
        return overall, per_organ




## === cell 71
dice_score_obj = DiceScoreCustom(num_classes=NUM_CLASSES)
hausdorff_obj = HausdorffDistanceCustom(num_classes=NUM_CLASSES)




## === cell 72
def one_epoch_train(epoch):
    model.train()
    running_loss = 0.0

    loop = tqdm(dataloader_train, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for data_batch in loop:
        imgs, masks, ids = data_batch
        imgs, masks = imgs.to(DEVICE, dtype=torch.float), masks.to(
            DEVICE, dtype=torch.float
        )
        optimizer.zero_grad()
        pred_masks = model(imgs)

        loss = loss_fn(pred_masks, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(dataloader_train)
    return avg_loss




## === cell 73
slices80_casedays = set(
    data_valid[["case", "day", "slice"]]
    .drop_duplicates()
    .value_counts(["case", "day"])
    .loc[lambda s: s == 80]
    .index
)




## === cell 74
def one_epoch_valid():
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        for data_batch in dataloader_valid:
            imgs, masks, ids = data_batch
            imgs, masks = imgs.to(DEVICE, dtype=torch.float), masks.to(
                DEVICE, dtype=torch.float
            )
            pred_masks = model(imgs)
            loss = loss_fn(pred_masks, masks)
            running_loss += loss.item()

            pred_masks_bin = (torch.sigmoid(pred_masks) > 0.5).int()
            masks_bin = masks.int()
            dice_score_obj.update(pred_masks_bin, masks_bin)

        avg_loss = running_loss / len(dataloader_valid)
        epoch_dice_score = dice_score_obj.compute()
        dice_score_obj.reset()
        epoch_hausdorff = hausdorff_obj.compute()
        hausdorff_obj.reset()

    return avg_loss, epoch_dice_score, epoch_hausdorff




## === cell 75
if TRAIN_VALID_SPLIT:
    for epoch in range(EPOCHS):
        loss_train = one_epoch_train(epoch)
        loss_valid, dice_score, hausdorff = one_epoch_valid()
        dice_overall, dice_per_organ = dice_score
        hausdorff_overall, hausdorff_per_organ = hausdorff
        combined_metric = 0.4 * dice_overall + 0.6 * (1 - float(hausdorff_overall))
        print(
            f"Epoch {epoch+1} | "
            f"Train Loss: {loss_train:.3f} | Valid Loss: {loss_valid:.3f} | "
            f"Combined metric: {combined_metric:.3f} | "
            f"Dice: {dice_overall:.3f} (LB {dice_per_organ[0]:.3f}, SB {dice_per_organ[1]:.3f}, S {dice_per_organ[2]:.3f}) | "
            f"Hausdorff: {hausdorff_overall:.3f}"
        )



## === cell 76
if TRAIN_VALID_SPLIT and SAVE_TRAIN_VALID_MODEL:
    torch.save(model.state_dict(), MODEL_PARAMS_FILE_NAME)
    print("Saved trained model to:", os.path.abspath(MODEL_PARAMS_FILE_NAME))



## === cell 77
if TEST_PREDICT:
    load_path = None
    if os.path.exists(MODEL_PARAMS_FILE_NAME):
        load_path = MODEL_PARAMS_FILE_NAME
    elif LOAD_MODEL_FOR_TEST_PREDICT and os.path.exists(MODEL_PARAMS_LOAD_FILE_PATH):
        load_path = MODEL_PARAMS_LOAD_FILE_PATH

    if load_path is not None:
        state = torch.load(load_path, map_location=DEVICE)
        try:
            model.load_state_dict(state)
        except Exception:
            model.load_state_dict(state, strict=False)
        print(f"Loaded model weights from: {load_path}")
    else:
        print(
            f"No checkpoint found at {MODEL_PARAMS_FILE_NAME} or {MODEL_PARAMS_LOAD_FILE_PATH}.\nProceeding with current model weights."
        )

    model.eval()

    sample_sub = pd.read_csv(DIR_PATH + "sample_submission.csv")
    data_test = pd.read_csv(DIR_PATH + "test.csv")

    data_test[["case", "day", "slice"]] = data_test["id"].str.extract(
        r"case(\d+)_day(\d+)_slice_(\d+)"
    )
    path_df_test = get_path_df(train=False)
    data_test = data_test.merge(path_df_test, on=["case", "day", "slice"], how="left")

    int_cols = ["case", "day", "slice", "slice_w", "slice_h"]
    for c in int_cols:
        data_test[c] = pd.to_numeric(data_test[c], errors="coerce")
    data_test[int_cols] = data_test[int_cols].fillna(0).astype(np.uint32)

    float_cols = ["px_w", "px_h"]
    for c in float_cols:
        data_test[c] = pd.to_numeric(data_test[c], errors="coerce")
    data_test[float_cols] = data_test[float_cols].fillna(0).astype(np.float32)

    if data_test["image_path"].isna().any():
        missing = data_test.loc[data_test["image_path"].isna(), "id"].head(5).tolist()
        raise RuntimeError(
            f"Missing image_path after merge for some test ids, e.g.: {missing}"
        )

    if ALBU_AVAILABLE:
        transform_test = A.Compose(
            [
                A.Resize(
                    IMAGE_RESIZE[0],
                    IMAGE_RESIZE[1],
                    interpolation=cv2.INTER_LINEAR if CV2_AVAILABLE else 1,
                ),
                A.Normalize(
                    mean=IMAGE_NORMALIZE_MEAN,
                    std=IMAGE_NORMALIZE_SD,
                    max_pixel_value=1.0,
                ),
                ToTensorV2(transpose_mask=False),
            ]
        )
    else:
        transform_test = _ComposeFallback(
            resize_hw=IMAGE_RESIZE,
            mean=IMAGE_NORMALIZE_MEAN,
            std=IMAGE_NORMALIZE_SD,
            max_pixel_value=1.0,
            transpose_mask=False,
        )

    dataset_test = GITractDataset(data_test, is_train=False, transforms=transform_test)
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=BATCH_SIZE_TEST,
        shuffle=False,
        num_workers=DATA_LOADER_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )



## === cell 78
if TEST_PREDICT:
    pred_map = {}

    with torch.no_grad():
        for imgs, ids, heights, widths in dataloader_test:
            imgs = imgs.to(DEVICE, dtype=torch.float)
            pred_masks = model(imgs)
            pred_masks = (torch.sigmoid(pred_masks) > 0.5).int()
            pred_masks = pred_masks.permute(0, 2, 3, 1).cpu().numpy()

            for mask, id_, h, w in zip(pred_masks, ids, heights, widths):
                mask_orig_size = _resize_nearest(
                    mask.astype(np.uint8), (int(h), int(w))
                ).astype(np.uint8)

                for chid, cls in enumerate(CLASS_NAMES):
                    pred_map[(id_, cls)] = rle_encode(mask_orig_size[..., chid])

    out = sample_sub.copy()
    out["predicted"] = [
        pred_map.get((rid, rcls), "") for rid, rcls in zip(out["id"], out["class"])
    ]

    out = (
        out[["id", "class", "predicted"]]
        .sort_values(["id", "class"])
        .reset_index(drop=True)
    )

    out.to_csv("submission.csv", index=False)

    print(out.head())
    print("Saved submission.csv with shape:", out.shape)
    print("Saved to:", os.path.abspath("submission.csv"))
    assert list(out.columns) == ["id", "class", "predicted"]
    assert len(out) == 20400
