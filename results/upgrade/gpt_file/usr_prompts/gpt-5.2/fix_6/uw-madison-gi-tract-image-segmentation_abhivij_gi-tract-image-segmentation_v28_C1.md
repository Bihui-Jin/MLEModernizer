# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

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
    print("Internet appears ON, but skipping pip installs to keep environment stable.")
else:
    print("Internet off - Relying on preinstalled dependencies")



## === cell 1
import numpy as np
import pandas as pd

import os
import random
import re
import sys
import time

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
import seaborn as sns

from glob import glob

from tqdm.auto import tqdm

tqdm.pandas()

from sklearn.model_selection import StratifiedGroupKFold

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

try:
    from monai.metrics.utils import get_mask_edges, get_surface_distance  # type: ignore

    _HAS_MONAI = True
except Exception as e:
    _HAS_MONAI = False
    get_mask_edges, get_surface_distance = None, None
    print(f"MONAI unavailable (continuing without it): {type(e).__name__}: {e}")



## === cell 2
print(f"Number of available CPUs: {os.cpu_count()}")
print(f"Number of available GPUs: {torch.cuda.device_count()}")



## === cell 3
pass



## === cell 4
DIR_PATH = "/kaggle/input/uw-madison-gi-tract-image-segmentation/"

pd.set_option("display.max_colwidth", 400)

CMAP1 = ListedColormap([[0, 0, 0, 0], [1, 0, 0, 1]])  # black transparent, red opaque
CMAP2 = ListedColormap([[0, 0, 0, 0], [0, 1, 0, 1]])  # black transparent, green opaque
CMAP3 = ListedColormap([[0, 0, 0, 0], [0, 0, 1, 1]])  # black transparent, blue opaque

RANDOM_SEED = 0

IMAGE_NORMALIZE_MEAN = (0.485, 0.456, 0.406)
IMAGE_NORMALIZE_SD = (0.229, 0.224, 0.225)

IMAGE_RESIZE = [256, 256]

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
    "/kaggle/input/git-seg/pytorch/256x256/1/GIT-Seg-256x256-efficientnet-b1.pth"
)

TRAIN_VALID_SPLIT = False
TEST_PREDICT = True

SAVE_TRAIN_VALID_MODEL = False
LOAD_MODEL_FOR_TEST_PREDICT = True

SAVE_MASKS = False
LOAD_SAVED_MASKS = True

MASK_DATASET_ROOT = "/kaggle/input/git-seg-mask/"



## === cell 5
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)



## === cell 6
pass



## === cell 7
data = pd.read_csv(DIR_PATH + "train.csv")
data.head()



## === cell 8
data_nonnaseg = data.loc[data.segmentation.notna(), :]
data_nonnaseg.head()



## === cell 9
data[["case", "day", "slice"]] = data["id"].str.extract(
    r"case(\d+)_day(\d+)_slice_(\d+)"
)
data




## === cell 10
def get_path_df(train=True):
    if train:
        paths = glob(DIR_PATH + "train/*/*/*/*")
    else:
        paths = glob(DIR_PATH + "test/*/*/*/*")
    path_df = pd.DataFrame(paths, columns=["image_path"])
    path_df[["case", "day", "slice", "slice_w", "slice_h", "px_w", "px_h"]] = (
        path_df.image_path.str.extract(
            r".*/case(\d+)_day(\d+)/scans/slice_(\d+)_(\d+)_(\d+)_(\d+\.\d+)_(\d+\.\d+)\.png"
        )
    )

    return path_df


path_df = get_path_df()



## === cell 11
data.info()



## === cell 12
path_df.info()



## === cell 13
pass



## === cell 14
data = data.merge(path_df, on=["case", "day", "slice"])
data



## === cell 15
data.info()



## === cell 16
data.px_w.unique(), data.px_h.unique()



## === cell 17
data.case.unique(), data.day.unique(), data.slice.unique(), data.slice_w.unique(), data.slice_h.unique()



## === cell 18
int_cols = ["case", "day", "slice", "slice_w", "slice_h"]
data[int_cols] = data[int_cols].astype(np.uint32)

float_cols = ["px_w", "px_h"]
data[float_cols] = data[float_cols].astype(np.float32)

data.info()



## === cell 19
pass




## === cell 20
def rle_decode(mask_rle, shape):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    IMPORTANT: competition expects pixels numbered top-to-bottom then left-to-right,
    which corresponds to flatten(order='F').
    """
    if (
        mask_rle is None
        or (isinstance(mask_rle, float) and np.isnan(mask_rle))
        or mask_rle == ""
    ):
        return np.zeros(shape, dtype=np.uint8)

    s = np.asarray(mask_rle.split(), dtype=np.int64)
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
    Returns run length as string formatted

    IMPORTANT: encode in flatten(order='F') to match evaluation pixel order.
    """
    if img is None:
        return ""
    img = (img > 0).astype(np.uint8)
    pixels = img.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 21
pass




## === cell 22
def dict_size(d):
    size = sys.getsizeof(d)  # dict container itself
    for k, v in d.items():
        size += sys.getsizeof(k) + sys.getsizeof(v)
    return size


id_to_impath = dict(
    data[["id", "image_path"]]
    .drop_duplicates("id")
    .set_index("id")["image_path"]
    .to_dict()
)
print("id_to_impath size:", dict_size(id_to_impath) / (1024 * 1024), "MB")

id_dicts = {"impath": id_to_impath}

id_to_shape = dict(
    data[["id", "slice_h", "slice_w"]]
    .drop_duplicates("id")
    .set_index("id")[["slice_h", "slice_w"]]
    .apply(tuple, axis=1)
    .to_dict()
)
idclass_to_rle = {
    (id_, class_): seg
    for id_, class_, seg in zip(data.id, data["class"], data.segmentation)
    if pd.notna(seg)
}
id_dicts["shape"] = id_to_shape
id_dicts["rle"] = idclass_to_rle

print("id_to_shape size:", dict_size(id_to_shape) / (1024 * 1024), "MB")
print("idclass_to_rle size:", dict_size(idclass_to_rle) / (1024 * 1024), "MB")



## === cell 23
pass




## === cell 24
def get_mask(id_, id_dicts):
    """
    id_dicts : dict of id_mapping dicts - allowed keys : impath, shape, rle
    """
    if LOAD_SAVED_MASKS:
        try:
            id_to_impath_local = id_dicts["impath"]
            mask_path = MASK_DATASET_ROOT + os.path.relpath(
                id_to_impath_local[id_], DIR_PATH
            )
            mask_path = os.path.splitext(mask_path)[0] + ".npy"
            if os.path.exists(mask_path):
                return np.load(mask_path)
        except Exception:
            pass  # fall back below

    id_to_shape_local, idclass_to_rle_local = id_dicts["shape"], id_dicts["rle"]
    h, w = id_to_shape_local[id_]
    shape = (int(h), int(w), 3)
    mask = np.zeros(shape, dtype=np.uint8)
    for i, class_ in enumerate(CLASS_NAMES):
        rle = idclass_to_rle_local.get((id_, class_))
        if rle:
            mask[..., i] = rle_decode(rle, shape[:2])
    return mask




## === cell 25
full_image_file_path = (
    DIR_PATH + "train/case123/case123_day20/scans/slice_0065_266_266_1.50_1.50.png"
)
img = cv2.imread(full_image_file_path, cv2.IMREAD_UNCHANGED)

if img is None:
    print("Skipping visualization: example file not found:", full_image_file_path)
else:
    print(img.shape)
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Gray")
    plt.axis("off")
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(img, cmap="bone")
    plt.title("Bone")
    plt.axis("off")
    plt.colorbar()
    plt.tight_layout()
    plt.show()



## === cell 26
img0 = cv2.imread(full_image_file_path, cv2.IMREAD_UNCHANGED)
if img0 is None:
    print("Skipping CLAHE demo: example file not found:", full_image_file_path)
else:
    img = img0.astype("float32")
    img_norm = img.copy()
    mx = np.max(img)
    if mx > 0:
        img_norm /= mx
    img_norm_u8 = (img_norm * 255).astype(np.uint8)

    clahe1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
    clahe3 = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2, 2))

    res1 = clahe1.apply(img_norm_u8)
    res2 = clahe2.apply(img_norm_u8)
    res3 = clahe3.apply(img_norm_u8)

    plt.figure(figsize=(20, 4))
    for i, (title, im) in enumerate(
        zip(
            [
                "Original",
                "Normalized",
                "CLAHE clip=2 grid=8x8",
                "CLAHE clip=2 grid=2x2",
                "CLAHE clip=1 grid=2x2",
            ],
            [img, img_norm_u8, res1, res2, res3],
        )
    ):
        plt.subplot(1, 5, i + 1)
        plt.imshow(im, cmap="bone")
        plt.title(title)
        plt.colorbar()
        plt.axis("off")
    plt.tight_layout()
    plt.show()



## === cell 27
pass



## === cell 28
pass




## === cell 29
def load_image(id_, id_to_impath):
    path = id_to_impath.get(id_)
    if path is None:
        raise KeyError(f"Image path not found for id={id_}")
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"cv2.imread returned None for path={path}")
    img = img.astype("float32")  # convert from original 16-bit
    mx = np.max(img)
    if mx > 0:
        img /= mx
    return img




## === cell 30
pass




## === cell 31
def display_image(
    id_,
    id_dicts,
    pred_mask=None,
    apply_CLAHE=False,
    show_orig_img=True,
    show_true_mask=True,
    show_pred_mask=False,
):

    img = load_image(id_, id_dicts["impath"])
    img_u8 = (img * 255).astype(np.uint8)  # 0-255 range required for CLAHE.
    if apply_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2, 2))
        img_u8 = clahe.apply(img_u8)

    mask = get_mask(id_, id_dicts)

    plt.figure(figsize=(9, 3))

    i = 1
    if show_orig_img:
        plt.subplot(1, 3, i)
        i += 1
        plt.imshow(img_u8, cmap="bone")
        plt.title(f"{id_} image")
        plt.axis("off")

    if show_true_mask:
        plt.subplot(1, 3, i)
        i += 1
        plt.imshow(img_u8, cmap="bone")
        plt.title("Image with true mask")
        plt.imshow(mask[..., 0], cmap=CMAP1)
        plt.imshow(mask[..., 1], cmap=CMAP2)
        plt.imshow(mask[..., 2], cmap=CMAP3)

        handles = [
            Rectangle((0, 0), 1, 1, color=CMAP1(1.0)),
            Rectangle((0, 0), 1, 1, color=CMAP2(1.0)),
            Rectangle((0, 0), 1, 1, color=CMAP3(1.0)),
        ]
        labels = ["Large Bowel", "Small Bowel", "Stomach"]
        plt.axis("off")
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
        plt.imshow(pred_mask[..., 0], cmap=CMAP1)
        plt.imshow(pred_mask[..., 1], cmap=CMAP2)
        plt.imshow(pred_mask[..., 2], cmap=CMAP3)

        handles = [
            Rectangle((0, 0), 1, 1, color=CMAP1(1.0)),
            Rectangle((0, 0), 1, 1, color=CMAP2(1.0)),
            Rectangle((0, 0), 1, 1, color=CMAP3(1.0)),
        ]
        labels = ["Large Bowel", "Small Bowel", "Stomach"]
        plt.axis("off")
        plt.legend(
            handles,
            labels,
            bbox_to_anchor=(1.0, -0.4),
            loc="lower right",
            borderaxespad=0.0,
        )

    plt.tight_layout()
    plt.show()




## === cell 32
_example_id = "case131_day0_slice_0066"
if _example_id in id_dicts["impath"]:
    display_image(_example_id, id_dicts)
else:
    print("Skipping display_image; id not found:", _example_id)



## === cell 33
_example_id = "case131_day0_slice_0066"
if _example_id in id_dicts["impath"]:
    display_image(_example_id, id_dicts, apply_CLAHE=True)
else:
    print("Skipping display_image; id not found:", _example_id)



## === cell 34
pass



## === cell 35
_example_id = "case123_day20_slice_0065"
if _example_id in id_dicts["impath"]:
    display_image(_example_id, id_dicts, apply_CLAHE=True)
else:
    print("Skipping display_image; id not found:", _example_id)



## === cell 36
_example_id = "case123_day20_slice_0001"
if _example_id in id_dicts["impath"]:
    display_image(_example_id, id_dicts, apply_CLAHE=True)
else:
    print("Skipping display_image; id not found:", _example_id)



## === cell 37
pass



## === cell 38
pass




## === cell 39
def display_multiple_slices(
    id_array, id_dicts, apply_CLAHE=False, show_pred_mask=False, pred_mask_array=None
):
    """
    id_array : an array of ids like case123_day20_slice_0001
    id_dicts : dict of id_mapping dicts - allowed keys : impath, shape, rle
    """
    l = len(id_array)
    rows = np.ceil(l / 5).astype(int)
    max_cols = 5

    plt.figure(figsize=(max_cols * 3, rows * 3))

    for i in range(l):
        id_ = id_array[i]
        if id_ not in id_dicts["impath"]:
            continue

        img = cv2.imread(id_dicts["impath"][id_], cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        img = img.astype("float32")
        mx = np.max(img)
        if mx > 0:
            img /= mx

        if apply_CLAHE:
            clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2, 2))
            img_u8 = (img * 255).astype(np.uint8)
            img_u8 = clahe.apply(img_u8)
            img_show = img_u8
        else:
            img_show = img

        if show_pred_mask and pred_mask_array is not None:
            mask = pred_mask_array[i]
        else:
            mask = get_mask(id_, id_dicts)

        plt.subplot(rows, max_cols, i + 1)
        plt.imshow(img_show, cmap="bone")
        plt.title(id_)
        plt.imshow(mask[..., 0], cmap=CMAP1)
        plt.imshow(mask[..., 1], cmap=CMAP2)
        plt.imshow(mask[..., 2], cmap=CMAP3)
        plt.axis("off")

        if i == 0:
            handles = [
                Rectangle((0, 0), 1, 1, color=CMAP1(1.0)),
                Rectangle((0, 0), 1, 1, color=CMAP2(1.0)),
                Rectangle((0, 0), 1, 1, color=CMAP3(1.0)),
            ]
            labels = ["Large Bowel", "Small Bowel", "Stomach"]
            plt.legend(
                handles,
                labels,
                bbox_to_anchor=(0.0, 1.5),
                loc="upper left",
                borderaxespad=0.0,
            )

    plt.tight_layout()
    plt.show()




## === cell 40
_ids = data.query(
    "case == 123 and day == 20 and slice >= 63 and slice <= 70"
).id.unique()
if len(_ids):
    display_multiple_slices(_ids, id_dicts, apply_CLAHE=True)
else:
    print("No ids found for visualization query in cell 42")



## === cell 41
_ids = data.query("case == 131 and day == 0 and slice > 55 and slice <= 70").id.unique()
if len(_ids):
    display_multiple_slices(_ids, id_dicts, apply_CLAHE=True)
else:
    print("No ids found for visualization query in cell 43")



## === cell 42
pass



## === cell 43
data.loc[data.segmentation.isna(), :].head()



## === cell 44
data.isna().sum()



## === cell 45
pass



## === cell 46
print(
    f"Num cases : {len(data.case.unique())}         Num unique days : {len(data.day.unique())}           Num unique slices : {len(data.slice.unique())}"
)



## === cell 47
count_df = (
    data[["id", "slice_w", "slice_h"]]
    .drop_duplicates()[["slice_w", "slice_h"]]
    .value_counts()
    .reset_index(name="count")
)
count_df["percent"] = count_df["count"] * 100 / sum(count_df["count"])
print(sum(count_df["count"]))
count_df



## === cell 48
count_df = (
    data[["id", "px_w", "px_h"]]
    .drop_duplicates()[["px_w", "px_h"]]
    .value_counts()
    .reset_index(name="count")
)
count_df["percent"] = count_df["count"] * 100 / sum(count_df["count"])
print(sum(count_df["count"]))
count_df



## === cell 49
pass



## === cell 50
day_dist = (
    data[["case", "day"]]
    .drop_duplicates()["case"]
    .value_counts()
    .reset_index(name="num_days")
)
display(day_dist.head())



## === cell 51
pass



## === cell 52
slice_dist = (
    data[["case", "day", "slice"]]
    .drop_duplicates()[["case", "day"]]
    .value_counts()
    .reset_index(name="num_slices")
)
display(slice_dist.head())



## === cell 53
pass



## === cell 54
slice_dist.loc[slice_dist.num_slices == 80, :].head()



## === cell 55
case_day_slice_df = data[
    ["case", "day", "slice", "slice_w", "slice_h"]
].drop_duplicates()
case_day_slice_df.merge(case_day_slice_df, on=["case", "day"]).query(
    "(slice_w_x != slice_w_y) | (slice_h_x != slice_h_y)"
).head()



## === cell 56
pass



## === cell 57
case_day_slice_df = data[
    ["case", "day", "slice", "slice_w", "slice_h"]
].drop_duplicates()
case_day_slice_df.merge(case_day_slice_df, on=["case"]).query(
    "(slice_w_x != slice_w_y) | (slice_h_x != slice_h_y)"
).head()



## === cell 58
pass



## === cell 59
case_day_slice_df = data[["case", "day", "slice", "px_w", "px_h"]].drop_duplicates()
case_day_slice_df.merge(case_day_slice_df, on=["case", "day"]).query(
    "(px_w_x != px_w_y) | (px_h_x != px_h_y)"
).head()



## === cell 60
case_day_slice_df = data[["case", "day", "slice", "px_w", "px_h"]].drop_duplicates()
case_day_slice_df.merge(case_day_slice_df, on=["case"]).query(
    "(px_w_x != px_w_y) | (px_h_x != px_h_y)"
).head()



## === cell 61
pass



## === cell 62
pass



## === cell 63
num_missing_seg_masks = data.segmentation.isna().sum()
print(
    f"Missing Seg Mask \n count = {num_missing_seg_masks}\n percentage = {num_missing_seg_masks/len(data)*100:.2f}"
)



## === cell 64
pass



## === cell 65
data["class"].value_counts()



## === cell 66
pass



## === cell 67
na_counts = (
    data.groupby("class")["segmentation"]
    .apply(lambda s: s.isna().sum())
    .reset_index(name="count")
)
na_counts["percent"] = (
    100 * na_counts["count"] / data.groupby("class")["segmentation"].size().values
)
na_counts.head()



## === cell 68
case_day_seg_missing = (
    data[["case", "day", "class", "segmentation"]]
    .groupby(["case", "day", "class"])["segmentation"]
    .apply(lambda s: s.isna().sum())
    .reset_index(name="count")
    .sort_values(by="count", ascending=False)
)
case_day_seg_missing.head()



## === cell 69
pass



## === cell 70
pass



## === cell 71
for _example_id in [
    "case43_day26_slice_0057",
    "case43_day26_slice_0058",
    "case43_day26_slice_0121",
    "case43_day26_slice_0122",
]:
    if _example_id in id_dicts["impath"]:
        display_image(_example_id, id_dicts, apply_CLAHE=True)
    else:
        print("Skipping display_image; id not found:", _example_id)



## === cell 72
pass



## === cell 73
pass



## === cell 74
for _example_id in [
    "case117_day15_slice_0009",
    "case117_day15_slice_0010",
    "case117_day15_slice_0065",
    "case117_day15_slice_0066",
]:
    if _example_id in id_dicts["impath"]:
        display_image(_example_id, id_dicts, apply_CLAHE=True)
    else:
        print("Skipping display_image; id not found:", _example_id)



## === cell 75
pass



## === cell 76
pass



## === cell 77
pass




## === cell 78
def save_mask(id_, id_dicts):
    mask = get_mask(id_, id_dicts)
    image_path = id_dicts["impath"][id_]
    rel_path = os.path.relpath(image_path, DIR_PATH)
    mask_path = os.path.splitext(rel_path)[0] + ".npy"
    mask_dir = mask_path.rsplit("/", 1)[0]
    os.makedirs(mask_dir, exist_ok=True)
    np.save(mask_path, mask)




## === cell 79
if SAVE_MASKS:
    for id_ in tqdm(data[["id"]].drop_duplicates()["id"].values):
        save_mask(id_, id_dicts)
else:
    print("SAVE_MASKS is False; skipping mask saving.")



## === cell 80
pass



## === cell 81
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
index_train, index_valid = next(
    sgkf.split(data.id, data.segmentation.isna(), data.case)
)



## === cell 82
len(index_train), len(index_valid)



## === cell 83
data_train = data.iloc[index_train, :]
data_valid = data.iloc[index_valid, :]



## === cell 84
data_train.head()



## === cell 85
data_valid.head()



## === cell 86
pass



## === cell 87
print(len(data_train.case.unique()), len(data_valid.case.unique()))



## === cell 88
data_train_sub = data_train.loc[data_train.case.isin(data_train.case.unique()[:11]), :]
data_valid_sub = data_valid.loc[data_valid.case.isin(data_valid.case.unique()[:2]), :]

print(
    len(data_train_sub),
    len(data_valid_sub),
    len(data_train_sub) / max(1, len(data_valid_sub)),
)



## === cell 89
missing_masks_train = data_train_sub.segmentation.isna().sum()
missing_masks_valid = data_valid_sub.segmentation.isna().sum()
print(missing_masks_train, missing_masks_train * 100 / len(data_train_sub))
print(missing_masks_valid, missing_masks_valid * 100 / len(data_valid_sub))



## === cell 90
pass



## === cell 91
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
display(na_counts_train.head())

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
display(na_counts_valid.head())



## === cell 92
pass



## === cell 93
data_train_sub = data_train_sub.reset_index(drop=True)



## === cell 94
data_valid_sub = data_valid_sub.reset_index(drop=True)



## === cell 95
pass



## === cell 96
pass




## === cell 97
class GITractDataset(Dataset):
    def __init__(
        self, df, is_test=False, transforms=None, load_saved_masks=LOAD_SAVED_MASKS
    ):
        self.df = df.copy()
        self.is_test = is_test
        self.transforms = transforms

        self.df["id"] = self.df["id"].astype(str)

        if "segmentation" not in self.df.columns:
            self.df["segmentation"] = np.nan

        self.id_ = self.df[["id"]].drop_duplicates()["id"].values

        self.id_to_impath = (
            self.df[["id", "image_path"]]
            .drop_duplicates("id")
            .set_index("id")["image_path"]
            .to_dict()
        )
        self.id_to_shape = (
            self.df[["id", "slice_h", "slice_w"]]
            .drop_duplicates("id")
            .set_index("id")[["slice_h", "slice_w"]]
            .apply(tuple, axis=1)
            .to_dict()
        )

        self.idclass_to_rle = {
            (id_, class_): seg
            for id_, class_, seg in zip(
                self.df["id"].values,
                self.df["class"].values,
                self.df["segmentation"].values,
            )
            if pd.notna(seg)
        }

        self.id_dicts = {
            "impath": self.id_to_impath,
            "shape": self.id_to_shape,
            "rle": self.idclass_to_rle,
        }

    def __len__(self):
        return len(self.id_)

    def __getitem__(self, idx):
        id_ = self.id_[idx]
        img = load_image(id_, self.id_dicts["impath"])
        img = np.repeat(img[..., None], 3, axis=2)

        if not self.is_test:
            mask = get_mask(id_, self.id_dicts)
            if self.transforms:
                augmented = self.transforms(image=img, mask=mask)
                img = augmented["image"]
                mask = augmented["mask"]
            return img, mask, id_
        else:
            h, w = self.id_dicts["shape"][id_]
            if self.transforms:
                augmented = self.transforms(image=img)
                img = augmented["image"]
            return img, id_, int(h), int(w)




## === cell 98
pass



## === cell 99
transform_train = A.Compose(
    [
        A.Resize(
            IMAGE_RESIZE[0],
            IMAGE_RESIZE[1],
            interpolation=cv2.INTER_NEAREST,
            mask_interpolation=cv2.INTER_NEAREST,
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
            interpolation=cv2.INTER_NEAREST,
            mask_interpolation=cv2.INTER_NEAREST,
        ),
        A.Normalize(
            mean=IMAGE_NORMALIZE_MEAN, std=IMAGE_NORMALIZE_SD, max_pixel_value=1.0
        ),
        ToTensorV2(transpose_mask=True),
    ]
)



## === cell 100
dataset_train = GITractDataset(data_train, transforms=transform_train)
dataset_valid = GITractDataset(data_valid, transforms=transform_valid)

dataloader_train = DataLoader(
    dataset_train,
    batch_size=BATCH_SIZE_TRAIN,
    shuffle=True,
    num_workers=DATA_LOADER_NUM_WORKERS,
    pin_memory=True,
)
dataloader_valid = DataLoader(
    dataset_valid,
    batch_size=BATCH_SIZE_VALID,
    shuffle=False,
    num_workers=DATA_LOADER_NUM_WORKERS,
    pin_memory=True,
)



## === cell 101
try:
    dataset_batch = next(iter(dataloader_train))
    img, mask, id_ = dataset_batch
    print(img.shape, mask.shape, len(id_))
except Exception as e:
    print("DataLoader sanity-check failed:", type(e).__name__, e)



## === cell 102
if "img" in globals() and img is not None and hasattr(img, "shape"):
    idx = min(0, img.shape[0] - 1)
    print(np.max(img[idx].cpu().numpy()), np.min(img[idx].cpu().numpy()))



## === cell 103
if "img" in globals() and "mask" in globals() and img is not None and mask is not None:
    idx = min(0, img.shape[0] - 1)
    print(type(img[idx].cpu().numpy()[0, 0, 0]), type(mask[idx].cpu().numpy()[0, 0, 0]))




## === cell 104
def display_dataset(
    dataset, display_orig=False, num_images=None, denormalize=False, apply_CLAHE=False
):
    """
    dataset : batch tuple (img_arr, mask_arr, id_arr)
    """
    img_arr, mask_arr, id_arr = dataset
    if num_images is None:
        num_images = len(img_arr)
    max_cols = 5

    if display_orig:
        num_images = min(5, len(img_arr))
        rows = 2
        plt.figure(figsize=(max_cols * 3, rows * 3))
        ids_shown = list()
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
        img_u8 = (img * 255).astype(np.uint8)

        if apply_CLAHE:
            clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2, 2))
            for ch in range(3):
                img_u8[:, :, ch] = clahe.apply(img_u8[:, :, ch])

        mask = mask.permute(1, 2, 0).cpu().numpy()

        plt.subplot(rows, max_cols, idx + 1)
        plt.imshow(img_u8[:, :, 0], cmap="bone")
        plt.title(f"{idx} : {id_}")

        plt.imshow(mask[..., 0], cmap=CMAP1)
        plt.imshow(mask[..., 1], cmap=CMAP2)
        plt.imshow(mask[..., 2], cmap=CMAP3)
        plt.axis("off")

        if idx == 0:
            handles = [
                Rectangle((0, 0), 1, 1, color=CMAP1(1.0)),
                Rectangle((0, 0), 1, 1, color=CMAP2(1.0)),
                Rectangle((0, 0), 1, 1, color=CMAP3(1.0)),
            ]
            labels = ["Large Bowel", "Small Bowel", "Stomach"]
            plt.legend(
                handles,
                labels,
                bbox_to_anchor=(0.0, 1.5),
                loc="upper left",
                borderaxespad=0.0,
            )

        if display_orig:
            ids_shown.append(id_)

    plt.tight_layout()
    plt.show()




## === cell 105
if "dataset_batch" in globals():
    display_dataset(dataset_batch, num_images=5, denormalize=True)
else:
    print("Skipping display_dataset; no batch available")



## === cell 106
if "dataset_batch" in globals():
    display_dataset(
        dataset_batch,
        display_orig=False,
        denormalize=True,
        apply_CLAHE=True,
        num_images=5,
    )
else:
    print("Skipping display_dataset; no batch available")



## === cell 107
pass




## === cell 108
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


class UNetSmall(nn.Module):
    def __init__(self, in_channels=3, classes=3, base_ch=32):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base_ch * 2, base_ch * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base_ch * 4, base_ch * 8)

        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = DoubleConv(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = DoubleConv(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = DoubleConv(base_ch * 2, base_ch)

        self.head = nn.Conv2d(base_ch, classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.head(d1)


if TRAIN_VALID_SPLIT or TEST_PREDICT:
    model = UNetSmall(in_channels=3, classes=NUM_CLASSES, base_ch=32)
    model.to(DEVICE)



## === cell 109
pass



## === cell 110
if TRAIN_VALID_SPLIT or TEST_PREDICT:
    optimizer = optim.Adam(model.parameters(), lr=1e-3)



## === cell 111
pass



## === cell 112
bce_logits = nn.BCEWithLogitsLoss(reduction="mean")


def dice_loss_multilabel(logits, targets, eps=1e-6):
    """
    logits: (B,C,H,W), targets: (B,C,H,W) float {0,1}
    """
    probs = torch.sigmoid(logits)
    probs = probs.contiguous()
    targets = targets.contiguous()
    dims = (0, 2, 3)
    intersection = (probs * targets).sum(dims)
    denom = probs.sum(dims) + targets.sum(dims)
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def loss_fn(y_pred, y_true, loss_wt=0.5):
    return dice_loss_multilabel(y_pred, y_true) * loss_wt + bce_logits(
        y_pred, y_true
    ) * (1 - loss_wt)




## === cell 113
pass



## === cell 114
pass




## === cell 115
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
        """
        preds, targets: (B, C, H, W) binary {0,1} tensors
        Skip organs where both pred & target are empty.
        """
        I = (targets & preds).sum((2, 3))
        U = (targets | preds).sum((2, 3))
        dice = (2 * I) / (U + I + self.eps)
        non_empty = U > 0  # [B, C]

        organ_counts = non_empty.sum(dim=1)  # [B]
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




## === cell 116
pass




## === cell 117
class HausdorffDistanceCustom:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.h3d_sum = 0.0
        self.image3d_count = 0
        self.organ_h3d_sum = np.zeros(self.num_classes)
        self.organ_count_sum = np.zeros(self.num_classes)

    def _compute_hausdorff_per_organ(self, preds, targets):
        """
        preds and targets : (Depth, Height, Width) binary {0,1} numpy arrays
        If MONAI isn't available, return 0 for equal arrays else 1 (bounded), so code runs.
        """
        if np.all(preds == targets):
            return 0.0

        if not _HAS_MONAI:
            return 1.0

        (edges_preds, edges_targets) = get_mask_edges(preds, targets)
        surface_distance = get_surface_distance(
            edges_preds, edges_targets, distance_metric="euclidean"
        )

        if surface_distance.shape == (0,):
            return 0.0
        dist = float(surface_distance.max())
        max_dist = float(np.sqrt(np.sum((np.array(preds.shape) - 1) ** 2)))
        if dist > max_dist:
            return 1.0
        return dist / max_dist

    def update(self, preds, targets):
        """
        preds and targets : (Channel, Depth, Height, Width) binary {0,1} numpy arrays
        """
        U = (targets | preds).sum((1, 2, 3))  # [C]
        hausdorff = np.array(
            [
                self._compute_hausdorff_per_organ(preds[i, ...], targets[i, ...])
                for i in range(NUM_CLASSES)
            ]
        )
        non_empty = U > 0
        organ_count = int(non_empty.sum())
        if organ_count != 0:
            hausdorff_per_3dimage = float(hausdorff.sum() / organ_count)
            self.h3d_sum += hausdorff_per_3dimage
            self.image3d_count += 1
        self.organ_h3d_sum += hausdorff
        self.organ_count_sum += non_empty

    def compute(self):
        overall = self.h3d_sum / max(1, self.image3d_count)
        per_organ = np.divide(self.organ_h3d_sum, np.maximum(1, self.organ_count_sum))
        return overall, per_organ




## === cell 118
dice_score_obj = DiceScoreCustom(num_classes=NUM_CLASSES)
hausdorff_obj = HausdorffDistanceCustom(num_classes=NUM_CLASSES)



## === cell 119
pass




## === cell 120
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




## === cell 121
slices80_casedays = set(
    data_valid[["case", "day", "slice"]]
    .drop_duplicates()
    .value_counts(["case", "day"])
    .loc[lambda s: s == 80]
    .index
)




## === cell 122
def one_epoch_valid():
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        pred_masks_dict, masks_dict = {}, {}
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

            for p, m, id_ in zip(pred_masks_bin, masks_bin, ids):
                match = re.match(r"case(\d+)_day(\d+)_slice_(\d+)", str(id_))
                if match:
                    caseid, dayid, sliceid = map(int, match.groups())
                else:
                    continue

                casedayid = (caseid, dayid)

                pred_masks_dict.setdefault(casedayid, []).append((sliceid, p))
                masks_dict.setdefault(casedayid, []).append((sliceid, m))

                if (len(pred_masks_dict[casedayid]) == 144) or (
                    casedayid in slices80_casedays
                    and len(pred_masks_dict[casedayid]) == 80
                ):
                    pred_masks_sorted = [
                        pp.cpu().numpy()
                        for sid, pp in sorted(
                            pred_masks_dict[casedayid], key=lambda x: x[0]
                        )
                    ]
                    masks_sorted = [
                        mm.cpu().numpy()
                        for sid, mm in sorted(masks_dict[casedayid], key=lambda x: x[0])
                    ]

                    pred_masks_volume = np.stack(pred_masks_sorted, axis=1)
                    masks_volume = np.stack(masks_sorted, axis=1)

                    hausdorff_obj.update(pred_masks_volume, masks_volume)

                    del pred_masks_dict[casedayid], masks_dict[casedayid]

        avg_loss = running_loss / len(dataloader_valid)
        epoch_dice_score = dice_score_obj.compute()
        dice_score_obj.reset()

        epoch_hausdorff = hausdorff_obj.compute()
        hausdorff_obj.reset()

    return avg_loss, epoch_dice_score, epoch_hausdorff




## === cell 123
if TRAIN_VALID_SPLIT:
    for epoch in range(EPOCHS):
        loss_train = one_epoch_train(epoch)
        loss_valid, dice_score, hausdorff = one_epoch_valid()
        dice_overall, dice_per_organ = dice_score
        hausdorff_overall, hausdorff_per_organ = hausdorff
        combined_metric = 0.4 * dice_overall + 0.6 * (1 - hausdorff_overall)
        print(
            f"Epoch {epoch+1} | "
            f"Train Loss: {loss_train:.3f} | Valid Loss: {loss_valid:.3f} | "
            f"Combined metric: {combined_metric:.3f} | "
            f"Dice: {dice_overall:.3f} (LB {dice_per_organ[0]:.3f}, SB {dice_per_organ[1]:.3f}, S {dice_per_organ[2]:.3f}) | "
            f"Hausdorff: {hausdorff_overall:.3f} (LB {hausdorff_per_organ[0]:.3f}, SB {hausdorff_per_organ[1]:.3f}, S {hausdorff_per_organ[2]:.3f})"
        )



## === cell 124
if TRAIN_VALID_SPLIT and SAVE_TRAIN_VALID_MODEL:
    torch.save(model.state_dict(), MODEL_PARAMS_FILE_NAME)



## === cell 125
pass




## === cell 126
def _find_checkpoint(preferred_path: str) -> str | None:
    if preferred_path and os.path.exists(preferred_path):
        return preferred_path

    candidates = []
    fname = os.path.basename(preferred_path) if preferred_path else None
    search_roots = [
        "/kaggle/input",
        "/kaggle/working",
    ]
    for root in search_roots:
        if not os.path.exists(root):
            continue
        if fname:
            candidates.extend(glob(os.path.join(root, "**", fname), recursive=True))

    return candidates[0] if candidates else None


def _load_checkpoint_if_compatible(model: torch.nn.Module, ckpt_path: str) -> bool:
    """
    Change is directly score-relevant: avoids silently using random weights when checkpoint
    belongs to a different architecture (current root cause of ~0.003 score).
    We only accept loading if a high fraction of keys match with identical shapes.
    """
    try:
        state = torch.load(ckpt_path, map_location="cpu")
        if (
            isinstance(state, dict)
            and "state_dict" in state
            and isinstance(state["state_dict"], dict)
        ):
            state = state["state_dict"]
        if not isinstance(state, dict):
            return False

        model_sd = model.state_dict()
        matched = 0
        for k, v in state.items():
            if k in model_sd and hasattr(v, "shape") and v.shape == model_sd[k].shape:
                matched += 1

        match_ratio = matched / max(1, len(model_sd))
        print(
            f"Checkpoint compatibility: matched {matched}/{len(model_sd)} keys (ratio={match_ratio:.2f})"
        )

        if match_ratio < 0.80:
            print("Checkpoint appears incompatible with UNetSmall; will not load it.")
            return False

        missing, unexpected = model.load_state_dict(state, strict=False)
        print("Loaded checkpoint into UNetSmall with strict=False.")
        if missing:
            print("Missing keys (truncated):", missing[:5], "...")
        if unexpected:
            print("Unexpected keys (truncated):", unexpected[:5], "...")
        return True
    except Exception as e:
        print("WARNING: failed to load checkpoint:", type(e).__name__, e)
        return False


if TEST_PREDICT:
    loaded_ok = False
    if LOAD_MODEL_FOR_TEST_PREDICT:
        ckpt = _find_checkpoint(MODEL_PARAMS_LOAD_FILE_PATH)
        if ckpt is None:
            print(
                "WARNING: checkpoint not found. Will train UNetSmall briefly to avoid random-weight submission."
            )
        else:
            print("Found checkpoint candidate:", ckpt)
            loaded_ok = _load_checkpoint_if_compatible(model, ckpt)

    if not loaded_ok:
        print(
            "Fallback: enabling TRAIN_VALID_SPLIT to train UNetSmall for inference weights."
        )
        TRAIN_VALID_SPLIT = True
        for epoch in range(EPOCHS):
            loss_train = one_epoch_train(epoch)
            loss_valid, dice_score, hausdorff = one_epoch_valid()
            dice_overall, dice_per_organ = dice_score
            hausdorff_overall, hausdorff_per_organ = hausdorff
            combined_metric = 0.4 * dice_overall + 0.6 * (1 - hausdorff_overall)
            print(
                f"[Fallback Train] Epoch {epoch+1} | "
                f"Train Loss: {loss_train:.3f} | Valid Loss: {loss_valid:.3f} | "
                f"Combined metric: {combined_metric:.3f}"
            )

    model.eval()

    data_test = pd.read_csv(DIR_PATH + "sample_submission.csv")
    test_set_hidden = not bool(len(data_test))
    if test_set_hidden:
        data_test = data_valid.copy()
    else:
        data_test["id"] = data_test["id"].astype(str)
        data_test[["case", "day", "slice"]] = data_test["id"].str.extract(
            r"case(\d+)_day(\d+)_slice_(\d+)"
        )
        for c in ["case", "day", "slice"]:
            data_test[c] = pd.to_numeric(data_test[c], errors="coerce")
        data_test = data_test.dropna(subset=["case", "day", "slice"]).copy()

        path_df_test = get_path_df(train=False)
        for c in ["case", "day", "slice", "slice_w", "slice_h"]:
            path_df_test[c] = pd.to_numeric(path_df_test[c], errors="coerce")
        for c in ["px_w", "px_h"]:
            path_df_test[c] = pd.to_numeric(path_df_test[c], errors="coerce")

        data_test = data_test.merge(
            path_df_test, on=["case", "day", "slice"], how="left"
        )

        before = len(data_test)
        data_test = data_test.dropna(subset=["image_path", "slice_w", "slice_h"]).copy()
        after = len(data_test)
        if after != before:
            print(
                f"WARNING: dropped {before-after} test rows due to missing image paths after merge."
            )

        int_cols = ["case", "day", "slice", "slice_w", "slice_h"]
        data_test[int_cols] = data_test[int_cols].astype(np.uint32)

        float_cols = ["px_w", "px_h"]
        data_test[float_cols] = data_test[float_cols].astype(np.float32)

    transform_test = A.Compose(
        [
            A.Resize(IMAGE_RESIZE[0], IMAGE_RESIZE[1], interpolation=cv2.INTER_NEAREST),
            A.Normalize(
                mean=IMAGE_NORMALIZE_MEAN, std=IMAGE_NORMALIZE_SD, max_pixel_value=1.0
            ),
            ToTensorV2(transpose_mask=False),
        ]
    )
    dataset_test = GITractDataset(data_test, is_test=True, transforms=transform_test)
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=BATCH_SIZE_TEST,
        shuffle=False,
        num_workers=DATA_LOADER_NUM_WORKERS,
    )



## === cell 127
if TEST_PREDICT:
    test_ids, test_class, test_pred_RLE = [], [], []

    with torch.no_grad():
        for imgs, ids, heights, widths in tqdm(dataloader_test, desc="Predicting"):
            imgs = imgs.to(DEVICE, dtype=torch.float)
            pred_masks = model(imgs)
            pred_masks = (torch.sigmoid(pred_masks) > 0.5).int()
            pred_masks = pred_masks.permute(0, 2, 3, 1).cpu().numpy()  # [B, H, W, C]

            for mask, id_, h, w in zip(pred_masks, ids, heights, widths):
                mask_orig_size = cv2.resize(
                    mask, dsize=(int(w), int(h)), interpolation=cv2.INTER_NEAREST
                )

                rles = [
                    rle_encode(mask_orig_size[..., chid].astype(np.uint8))
                    for chid in range(NUM_CLASSES)
                ]

                test_ids.extend([str(id_)] * NUM_CLASSES)
                test_class.extend(CLASS_NAMES)
                test_pred_RLE.extend(rles)

    submission_df = pd.DataFrame(
        {"id": test_ids, "class": test_class, "predicted": test_pred_RLE}
    )

    submission_df = submission_df[["id", "class", "predicted"]]

    sample_sub = pd.read_csv(DIR_PATH + "sample_submission.csv")[["id", "class"]]
    sample_sub["id"] = sample_sub["id"].astype(str)
    submission_df["id"] = submission_df["id"].astype(str)

    submission_df = sample_sub.merge(
        submission_df, on=["id", "class"], how="left", validate="1:1"
    )
    submission_df["predicted"] = submission_df["predicted"].fillna("")
    submission_df.to_csv("submission.csv", index=False)

    print("Wrote submission.csv with shape:", submission_df.shape)
    print(submission_df.head())
