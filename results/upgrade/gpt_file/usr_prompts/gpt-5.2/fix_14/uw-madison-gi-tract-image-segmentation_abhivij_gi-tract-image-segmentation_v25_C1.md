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

0.8175250973297868

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the internet-dependent installs and make the script robust to missing optional libraries (monai, segmentation_models_pytorch, albumentations) so it can run end-to-end in the Kaggle offline environment. I fix the dataset/mask loading failure by disabling the external pre-saved mask dependency and decoding masks directly from `train.csv` RLE when training/validating (and keep test inference mask-free). I also fix broken/optional visualization cells so they don’t crash (missing files/ids/clahe variables), and correct the model-weight loading path by falling back gracefully if the checkpoint is unavailable. Finally, I ensure test inference always builds `dataloader_test` and writes a valid `submission.csv` with the required `id,class,predicted` columns.'
- What this solution (achieved 0.0) has done: 'I (1) fix the pandas `query` crashes in the visualization cells by avoiding `astype(int)` inside `DataFrame.query` (it treats `int` as an undefined variable), making those cells non-blocking. Then (2) I ensure test inference actually uses `test.csv` (not `sample_submission.csv`) so the predictions align to the exact required `(id, class)` rows and avoid format/misalignment issues that can yield a near-zero score. Finally (3) I add a strict reindex against `sample_submission.csv` right before writing `submission.csv` to guarantee correct row order and completeness while keeping the model/prediction logic unchanged.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with an invalid RLE encoding (the competition expects column-major/Fortran-order flattening), which makes the decoded masks essentially wrong even if the model predicts reasonable blobs. I make the smallest possible fix by correcting `rle_encode`/`rle_decode` to use the required pixel order, without changing the model, training loop, or thresholding semantics. I also ensure empty masks encode to an empty string (common expectation for this competition) to avoid edge-case parsing issues. These changes should move the score up substantially toward your target while keeping everything else identical.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still most consistent with the RLE pixel-order being wrong for this competition, which makes decoded masks effectively nonsense even if predictions are reasonable. I make the smallest targeted fix by switching both `rle_encode` and `rle_decode` to the required column-major (Fortran) convention and ensuring the round-trip matches the Kaggle expectation. I also add a tiny safety check to force predicted masks to be contiguous uint8 before encoding (no semantic change, just avoids edge-case encoding bugs). Everything else (model, transforms, thresholding, submission alignment to `sample_submission.csv`) stays the same.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still most consistent with a submission-format mismatch rather than model quality: this competition’s `id` column in `sample_submission.csv`/`test.csv` is numeric (e.g., `0..`), but your code is writing string slice-ids like `case110_day12_slice_0001`, which not align and effectively evaluates as empty/incorrect. I make the smallest fix by building test inference on the exact `test.csv` rows (numeric `id` + `class`) and mapping each row to the correct image path via (`case`,`day`,`slice`) extracted from `image_path`, while keeping the model, thresholding, and RLE logic unchanged. I also keep the final strict reindex against `sample_submission.csv` to guarantee correct ordering and completeness. These changes should move the score sharply upward toward your target without altering core training/inference semantics.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still most consistent with the submission masks being evaluated against the wrong images: the competition `test.csv` `id` values already correspond to specific slices, but the current code reassigns them by sorting scan paths and zipping to submission ids, which can silently permute predictions and collapse the score. I make the smallest fix by extracting the true slice-level key (`case/day/slice`) directly from `test.csv` `id`, joining to `path_df_test` on those parsed fields, and using that to load the correct image per submission row—without changing the model, transforms, threshold (0.5), or RLE logic. I also ensure the `case/day/slice` columns are parsed as integers consistently on both sides to avoid merge mismatches. The final submission still be strictly reindexed to `sample_submission.csv` for correct order/completeness.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still most consistent with a silent test-image/id mismatch: `GITractDataset` builds `self.id_ = df.id.unique()` and then looks up `image_path` via `dict(zip(df.id, df.image_path))`, but in test you pass `data_test_images` that has one row per slice id (good) while the submission expects predictions for every `(id,class)` row (3 per slice), and the current pipeline relies on `sample_submission.csv` ids being the same strings as slice-ids (they are), but `pred_by_id[str(id_)]` can still miss due to `id_` being a `numpy`/`torch` scalar converted to a different string form. I make the smallest fix by forcing ids to be plain Python strings consistently at dataset construction and at prediction-time keys, so lookup into `pred_by_id` always hits. I also make sure the test dataset is built from the full `test.csv` (all rows), but still only loads each slice once by de-duplicating internally in the dataset while retaining a stable string id—this preserves the model/inference logic and only fixes alignment. Finally, I keep your existing Fortran-order RLE encode/decode and strict reindexing to `sample_submission.csv` unchanged.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the submission not matching the competition’s expected `test.csv` row order/contents (or missing predictions for many `(id,class)` rows), even if the model runs. I make the smallest fix to ensure predictions are generated per unique slice, then expanded back to *exactly* the 20,400 `(id,class)` rows from `test.csv`, and finally reindexed to `sample_submission.csv` to guarantee perfect alignment. I also force `id` values coming from the DataLoader to become plain Python strings (decoding bytes if needed) so dictionary lookups never miss silently. Core model/threshold/RLE logic stays unchanged.'
- What this solution (achieved 0.0) has done: 'Your 0.0 is still most consistent with an ID-to-image mismatch at test-time (predictions being produced for the wrong slice rows, or not being found when expanding back to the 20,400 `(id,class)` rows), rather than true model quality. I make one minimal, targeted change: build the test dataset from `test.csv` but ensure the dataset’s `id_` ordering exactly follows the first occurrence order in `test.csv` (not `df.id.unique()` which can reorder), and use those exact ids as keys for `pred_by_id`. I also add a hard assertion that every `test.csv` slice id got exactly one prediction (so we fail loudly instead of silently producing mostly-empty masks). This keeps the model, thresholding (0.5), resizing, and RLE logic unchanged while fixing the most common “valid submission but scores 0.0” cause.'
- What this solution (achieved 0.0) has done: 'I fix the execution blockers that prevent a submission from being written: the missing checkpoint currently raises an exception, and because that cell fails the test DataLoader is never created (causing the later `NameError`). The smallest safe change is to allow inference to proceed even when the external `.pth` isn’t available by disabling weight-loading (keeping the same model forward/threshold/RLE logic), so the pipeline always produces `submission.csv`. I also make the test DataLoader construction unconditional inside `TEST_PREDICT` so downstream cells never depend on a previously-failed cell. These changes are primarily correctness/stability; they should move you from “no valid submission” to a valid submission (score likely low without weights, but nonzero if predictions aren’t empty).'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the model not actually using trained weights at inference (so predictions are essentially random/empty), even though the submission format/ordering now looks correct. The smallest score-moving change toward your target is to make checkpoint loading robust to common Kaggle formats: many `.pth` files are saved as `{"state_dict": ...}` or with `"model."/"module."` prefixes, so `model.load_state_dict(state)` silently fails to load meaningful weights or throws, leaving you with an untrained model. I add a minimal `load_state_dict_flexible()` helper that extracts the real state dict, strips common prefixes, and loads with `strict=False` while reporting missing/unexpected keys; everything else (model, transforms, threshold=0.5, RLE encoding, submission alignment) stays the same. This should move the score up substantially from 0.0 and closer to your target, without changing evaluation semantics.'

# 9. Code solution

## === cell 0
import os, socket, sys, re, time, random
import numpy as np
import pandas as pd


def internet_on(host="8.8.8.8", port=53, timeout=2):
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except Exception:
        return False


print("Internet available:", internet_on())
print(
    "Running in Kaggle-like environment; will not attempt pip installs from within script."
)



## === cell 1
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
import seaborn as sns
from glob import glob

try:
    from tqdm.notebook import tqdm
except Exception:
    from tqdm import tqdm

try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception as e:
    StratifiedGroupKFold = None
    print("Warning: sklearn not available; TRAIN_VALID_SPLIT path may not work.", e)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    _HAS_ALB = True
except Exception as e:
    A = None
    ToTensorV2 = None
    _HAS_ALB = False
    print("Warning: albumentations not available; using minimal torch transforms.", e)

try:
    import segmentation_models_pytorch as smp

    _HAS_SMP = True
except Exception as e:
    smp = None
    _HAS_SMP = False
    print(
        "Warning: segmentation_models_pytorch not available; using fallback UNet-like model.",
        e,
    )

try:
    from monai.metrics.utils import get_mask_edges, get_surface_distance

    _HAS_MONAI = True
except Exception as e:
    get_mask_edges = None
    get_surface_distance = None
    _HAS_MONAI = False
    print(
        "Warning: monai not available; Hausdorff metric will be skipped if TRAIN_VALID_SPLIT.",
        e,
    )



## === cell 2
pass



## === cell 3
print(f"Number of available CPUs: {os.cpu_count()}")
print(f"Number of available GPUs: {torch.cuda.device_count()}")



## === cell 4
pass



## === cell 5
DIR_PATH = "/kaggle/input/uw-madison-gi-tract-image-segmentation/"

pd.set_option("display.max_colwidth", 400)

CMAP1 = ListedColormap([[0, 0, 0, 0], [1, 0, 0, 1]])  # black transparent, red opaque
CMAP2 = ListedColormap([[0, 0, 0, 0], [0, 1, 0, 1]])  # black transparent, green opaque
CMAP3 = ListedColormap([[0, 0, 0, 0], [0, 0, 1, 1]])  # black transparent, blue opaque

RANDOM_SEED = 0

IMAGE_NORMALIZE_MEAN = (0.485, 0.456, 0.406)
IMAGE_NORMALIZE_SD = (0.229, 0.224, 0.225)

IMAGE_RESIZE = [224, 224]

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
    "/kaggle/input/git-seg/pytorch/224x224/1/GIT-Seg-224x224-efficientnet-b1.pth"
)

TRAIN_VALID_SPLIT = False
TEST_PREDICT = True

SAVE_TRAIN_VALID_MODEL = False

LOAD_MODEL_FOR_TEST_PREDICT = True  # keep user intent; we'll auto-disable if missing.

SAVE_MASKS = False
LOAD_SAVED_MASKS = False

MASK_DATASET_ROOT = "/kaggle/input/git-seg-mask/"



## === cell 6
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)



## === cell 7
pass



## === cell 8
data = pd.read_csv(DIR_PATH + "train.csv")
data.head()



## === cell 9
data_nonnaseg = data.loc[data.segmentation.notna(), :]
data_nonnaseg.head()



## === cell 10
data[["case", "day", "slice"]] = data["id"].str.extract(
    r"case(\d+)_day(\d+)_slice_(\d+)"
)
data




## === cell 11
def get_path_df(train=True):
    if train:
        paths = glob(DIR_PATH + "train/*/*/scans/*.png")
    else:
        paths = glob(DIR_PATH + "test/*/*/scans/*.png")

    path_df = pd.DataFrame(paths, columns=["image_path"])
    path_df[["case", "day", "slice", "slice_w", "slice_h", "px_w", "px_h"]] = (
        path_df.image_path.str.extract(
            r".*/case(\d+)_day(\d+)/scans/slice_(\d+)_(\d+)_(\d+)_(\d+\.\d+)_(\d+\.\d+)\.png"
        )
    )
    return path_df


path_df = get_path_df()



## === cell 12
data.info()



## === cell 13
path_df.info()



## === cell 14
pass



## === cell 15
data = data.merge(path_df, on=["case", "day", "slice"])
data



## === cell 16
data.info()



## === cell 17
data.px_w.unique(), data.px_h.unique()



## === cell 18
data.case.unique(), data.day.unique(), data.slice.unique(), data.slice_w.unique(), data.slice_h.unique()



## === cell 19
int_cols = ["case", "day", "slice", "slice_w", "slice_h"]
data[int_cols] = data[int_cols].astype(np.uint32)

float_cols = ["px_w", "px_h"]
data[float_cols] = data[float_cols].astype(np.float32)

data.info()



## === cell 20
pass




## === cell 21
def rle_decode(mask_rle, shape):
    """
    mask_rle: run-length as string formatted (start length), 1-indexed
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    if mask_rle is None:
        return np.zeros(shape, dtype=np.uint8)
    mask_rle = str(mask_rle)
    if mask_rle.strip() == "":
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
    Returns run length as string formatted (empty string for empty mask)
    """
    if img is None:
        return ""
    img = (img > 0).astype(np.uint8)

    pixels = img.flatten(order="F")
    if pixels.sum() == 0:
        return ""

    pixels = np.concatenate([[0], pixels, [0]]).astype(np.uint8)
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(int(x)) for x in runs)




## === cell 22
pass




## === cell 23
def dict_size(d):
    size = sys.getsizeof(d)
    for k, v in d.items():
        size += sys.getsizeof(k) + sys.getsizeof(v)
    return size


id_to_impath = dict(zip(data.id, data.image_path))
print("id_to_impath size:", dict_size(id_to_impath) / (1024 * 1024), "MB")

id_dicts = {"impath": id_to_impath}

id_to_shape = dict(zip(data["id"], zip(data["slice_h"], data["slice_w"])))
idclass_to_rle = {
    (id_, class_): seg
    for id_, class_, seg in zip(data.id, data["class"], data.segmentation)
    if pd.notna(seg)
}
print("id_to_shape size:", dict_size(id_to_shape) / (1024 * 1024), "MB")
print("idclass_to_rle size:", dict_size(idclass_to_rle) / (1024 * 1024), "MB")

id_dicts["shape"] = id_to_shape
id_dicts["rle"] = idclass_to_rle



## === cell 24
pass




## === cell 25
def get_mask(id_, id_dicts):
    """
    id_dicts: dict with keys impath, shape, rle
    Returns mask (H,W,3) uint8
    """
    id_to_shape, idclass_to_rle = id_dicts["shape"], id_dicts["rle"]
    h, w = id_to_shape[id_]
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    for i, class_ in enumerate(CLASS_NAMES):
        rle = idclass_to_rle.get((id_, class_))
        if rle:
            mask[..., i] = rle_decode(rle, (h, w))
    return mask




## === cell 26
full_image_file_path = (
    DIR_PATH + "train/case123/case123_day20/scans/slice_0065_266_266_1.50_1.50.png"
)
img = cv2.imread(full_image_file_path, cv2.IMREAD_UNCHANGED)
if img is None:
    print("Example image not found at:", full_image_file_path)
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



## === cell 27
if cv2.imread(full_image_file_path, cv2.IMREAD_UNCHANGED) is None:
    print("Skipping CLAHE demo; example image missing.")
else:
    img = cv2.imread(full_image_file_path, cv2.IMREAD_UNCHANGED).astype("float32")
    img_norm = img.copy()
    mx = np.max(img_norm)
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



## === cell 28
pass



## === cell 29
pass




## === cell 30
def load_image(id_, id_to_impath):
    path = id_to_impath.get(id_)
    if path is None:
        raise KeyError(f"id not found in id_to_impath: {id_}")
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to read image for id {id_} at path: {path}")
    img = img.astype("float32")
    mx = np.max(img)
    if mx > 0:
        img /= mx
    return img




## === cell 31
pass




## === cell 32
def display_image(
    id_,
    id_dicts,
    pred_mask=None,
    apply_CLAHE=False,
    show_orig_img=True,
    show_true_mask=True,
    show_pred_mask=False,
):
    if id_ not in id_dicts["impath"]:
        print("ID not found:", id_)
        return

    img = load_image(id_, id_dicts["impath"])
    img_u8 = (img * 255).astype(np.uint8)
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




## === cell 33
example_id = data.id.iloc[0]
display_image(example_id, id_dicts)



## === cell 34
display_image(example_id, id_dicts, apply_CLAHE=True)



## === cell 35
pass



## === cell 36
print("Example id for display:", example_id)



## === cell 37
print("Example id for display:", example_id)



## === cell 38
pass



## === cell 39
pass




## === cell 40
def display_multiple_slices(
    id_array, id_dicts, apply_CLAHE=False, show_pred_mask=False, pred_mask_array=None
):
    l = len(id_array)
    if l == 0:
        print("No ids to display.")
        return

    rows = int(np.ceil(l / 5))
    max_cols = 5
    plt.figure(figsize=(max_cols * 3, rows * 3))

    clahe_local = (
        cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2, 2)) if apply_CLAHE else None
    )

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
            img_u8 = (img * 255).astype(np.uint8)
            img_u8 = clahe_local.apply(img_u8)
        else:
            img_u8 = (img * 255).astype(np.uint8)

        if show_pred_mask and pred_mask_array is not None:
            mask = pred_mask_array[i]
        else:
            mask = get_mask(id_, id_dicts)

        plt.subplot(rows, max_cols, i + 1)
        plt.imshow(img_u8, cmap="bone")
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




## === cell 41
_min_case = int(data["case"].astype(np.int32).min())
_min_day = int(data["day"].astype(np.int32).min())
display_multiple_slices(
    data.loc[
        (data["case"].astype(np.int32) == _min_case)
        & (data["day"].astype(np.int32) == _min_day),
        "id",
    ].unique()[:10],
    id_dicts,
    apply_CLAHE=True,
)



## === cell 42
display_multiple_slices(
    data.loc[
        (data["case"].astype(np.int32) == _min_case)
        & (data["day"].astype(np.int32) == _min_day),
        "id",
    ].unique()[10:20],
    id_dicts,
    apply_CLAHE=True,
)



## === cell 43
pass



## === cell 44
data.loc[data.segmentation.isna(), :].head()



## === cell 45
data.isna().sum()



## === cell 46
pass



## === cell 47
print(
    f"Num cases : {len(data.case.unique())} "
    f"Num unique days : {len(data.day.unique())}   "
    f"Num unique slices : {len(data.slice.unique())}"
)



## === cell 48
count_df = (
    data[["id", "slice_w", "slice_h"]]
    .drop_duplicates()[["slice_w", "slice_h"]]
    .value_counts()
    .reset_index(name="count")
)
count_df["percent"] = count_df["count"] * 100 / sum(count_df["count"])
print(sum(count_df["count"]))
count_df.head()



## === cell 49
count_df = (
    data[["id", "px_w", "px_h"]]
    .drop_duplicates()[["px_w", "px_h"]]
    .value_counts()
    .reset_index(name="count")
)
count_df["percent"] = count_df["count"] * 100 / sum(count_df["count"])
print(sum(count_df["count"]))
count_df.head()



## === cell 50
pass



## === cell 51
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



## === cell 52
pass



## === cell 53
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



## === cell 54
pass



## === cell 55
slice_dist.loc[slice_dist.num_slices == 80, :].head()



## === cell 56
case_day_slice_df = data[
    ["case", "day", "slice", "slice_w", "slice_h"]
].drop_duplicates()
case_day_slice_df.merge(case_day_slice_df, on=["case", "day"]).query(
    "(slice_w_x != slice_w_y) | (slice_h_x != slice_h_y)"
).head()



## === cell 57
pass



## === cell 58
case_day_slice_df = data[
    ["case", "day", "slice", "slice_w", "slice_h"]
].drop_duplicates()
case_day_slice_df.merge(case_day_slice_df, on=["case"]).query(
    "(slice_w_x != slice_w_y) | (slice_h_x != slice_h_y)"
).head()



## === cell 59
pass



## === cell 60
case_day_slice_df = data[["case", "day", "slice", "px_w", "px_h"]].drop_duplicates()
case_day_slice_df.merge(case_day_slice_df, on=["case", "day"]).query(
    "(px_w_x != px_w_y) | (px_h_x != px_h_y)"
).head()



## === cell 61
case_day_slice_df = data[["case", "day", "slice", "px_w", "px_h"]].drop_duplicates()
case_day_slice_df.merge(case_day_slice_df, on=["case"]).query(
    "(px_w_x != px_w_y) | (px_h_x != px_h_y)"
).head()



## === cell 62
pass



## === cell 63
pass



## === cell 64
num_missing_seg_masks = data.segmentation.isna().sum()
print(
    f"Missing Seg Mask \n count = {num_missing_seg_masks}\n percentage = {num_missing_seg_masks/len(data)*100:.2f}"
)



## === cell 65
pass



## === cell 66
data["class"].value_counts()



## === cell 67
pass



## === cell 68
na_counts = (
    data.groupby("class")["segmentation"]
    .apply(lambda s: s.isna().sum())
    .reset_index(name="count")
)
na_counts["percent"] = (
    100 * na_counts["count"] / data.groupby("class")["segmentation"].size().values
)
ax = sns.barplot(
    data=na_counts, x="class", y="percent", palette=[CMAP1(1.0), CMAP2(1.0), CMAP3(1.0)]
)
plt.ylabel("Percentage")
plt.xlabel("Segmentation Class")
plt.title("Missing Segmentation Masks")
plt.show()



## === cell 69
case_day_seg_missing = (
    data[["case", "day", "class", "segmentation"]]
    .groupby(["case", "day", "class"])["segmentation"]
    .apply(lambda s: s.isna().sum())
    .reset_index(name="count")
    .sort_values(by="count", ascending=False)
)
case_day_seg_missing.head()



## === cell 70
pass



## === cell 71
pass



## === cell 72
display_multiple_slices(data.id.unique()[:10], id_dicts, apply_CLAHE=True)



## === cell 73
pass



## === cell 74
pass



## === cell 75
display_multiple_slices(data.id.unique()[10:20], id_dicts, apply_CLAHE=True)



## === cell 76
pass



## === cell 77
pass



## === cell 78
pass




## === cell 79
def save_mask(id_, id_dicts):
    mask = get_mask(id_, id_dicts)
    image_path = id_dicts["impath"][id_]
    rel_path = os.path.relpath(image_path, DIR_PATH)
    mask_path = os.path.splitext(rel_path)[0] + ".npy"
    mask_dir = mask_path.rsplit("/", 1)[0]
    os.makedirs(mask_dir, exist_ok=True)
    np.save(mask_path, mask)




## === cell 80
if SAVE_MASKS:
    for id_ in tqdm(data.id.unique()):
        save_mask(id_, id_dicts)



## === cell 81
pass



## === cell 82
if StratifiedGroupKFold is None:
    TRAIN_VALID_SPLIT = False
    index_train = np.arange(len(data))
    index_valid = np.array([], dtype=int)
else:
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    index_train, index_valid = next(
        sgkf.split(data.id, data.segmentation.isna(), data.case)
    )



## === cell 83
len(index_train), len(index_valid)



## === cell 84
data_train = data.iloc[index_train, :]
data_valid = (
    data.iloc[index_valid, :] if len(index_valid) else data.iloc[index_train[:0], :]
)



## === cell 85
data_train.head()



## === cell 86
data_valid.head()



## === cell 87
pass



## === cell 88
print(len(data_train.case.unique()), len(data_valid.case.unique()))



## === cell 89
data_train_sub = data_train.loc[data_train.case.isin(data_train.case.unique()[:11]), :]
data_valid_sub = data_valid.loc[data_valid.case.isin(data_valid.case.unique()[:2]), :]
print(len(data_train_sub), len(data_valid_sub))



## === cell 90
missing_masks_train = data_train_sub.segmentation.isna().sum()
missing_masks_valid = data_valid_sub.segmentation.isna().sum()
print(missing_masks_train, missing_masks_train * 100 / len(data_train_sub))
if len(data_valid_sub):
    print(missing_masks_valid, missing_masks_valid * 100 / len(data_valid_sub))



## === cell 91
pass



## === cell 92
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
na_counts_train.head()



## === cell 93
pass



## === cell 94
data_train_sub = data_train_sub.reset_index(drop=True)



## === cell 95
data_valid_sub = data_valid_sub.reset_index(drop=True)



## === cell 96
pass



## === cell 97
pass




## === cell 98
class GITractDataset(Dataset):
    def __init__(
        self, df, is_test=False, transforms=None, load_saved_masks=LOAD_SAVED_MASKS
    ):
        df = df.copy()
        df["id"] = df["id"].astype(str)

        self.id_ = df["id"].drop_duplicates(keep="first").tolist()

        self.is_test = is_test
        self.transforms = transforms

        id_to_impath = dict(zip(df.id, df.image_path))
        self.id_dicts = {"impath": id_to_impath}

        id_to_shape = dict(zip(df.id, zip(df.slice_h, df.slice_w)))
        self.id_dicts["shape"] = id_to_shape

        if (not is_test) and (not load_saved_masks):
            idclass_to_rle = {
                (id_, class_): seg
                for id_, class_, seg in zip(df.id, df["class"], df.segmentation)
                if pd.notna(seg)
            }
        else:
            idclass_to_rle = None

        self.id_dicts["rle"] = idclass_to_rle

    def __len__(self):
        return len(self.id_)

    def __getitem__(self, idx):
        id_ = self.id_[idx]
        img = load_image(id_, self.id_dicts["impath"])
        img = np.repeat(img[..., None], 3, axis=2)

        if not self.is_test:
            h, w = self.id_dicts["shape"][id_]
            mask = np.zeros((h, w, 3), dtype=np.uint8)
            for i, class_ in enumerate(CLASS_NAMES):
                rle = (
                    None
                    if self.id_dicts["rle"] is None
                    else self.id_dicts["rle"].get((id_, class_))
                )
                if rle:
                    mask[..., i] = rle_decode(rle, (h, w))

            if self.transforms:
                out = self.transforms(image=img, mask=mask)
                img, mask = out["image"], out["mask"]
            return img, mask, id_
        else:
            h, w = self.id_dicts["shape"][id_]
            if self.transforms:
                out = self.transforms(image=img)
                img = out["image"]
            return img, id_, h, w




## === cell 99
class _SimpleResizeNormalizeToTensor:
    def __init__(self, size_hw, mean, std):
        self.h, self.w = size_hw
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    def __call__(self, image, mask=None):
        img = cv2.resize(
            image, (self.w, self.h), interpolation=cv2.INTER_NEAREST
        ).astype(np.float32)
        img = (img - self.mean) / self.std
        img_t = torch.from_numpy(img.transpose(2, 0, 1)).float()
        if mask is None:
            return {"image": img_t}
        m = cv2.resize(mask, (self.w, self.h), interpolation=cv2.INTER_NEAREST).astype(
            np.float32
        )
        mask_t = torch.from_numpy(m.transpose(2, 0, 1)).float()
        return {"image": img_t, "mask": mask_t}




## === cell 100
if _HAS_ALB:
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
else:
    transform_train = _SimpleResizeNormalizeToTensor(
        IMAGE_RESIZE, IMAGE_NORMALIZE_MEAN, IMAGE_NORMALIZE_SD
    )
    transform_valid = _SimpleResizeNormalizeToTensor(
        IMAGE_RESIZE, IMAGE_NORMALIZE_MEAN, IMAGE_NORMALIZE_SD
    )



## === cell 101
dataset_train = GITractDataset(
    data_train, transforms=transform_train, load_saved_masks=LOAD_SAVED_MASKS
)
dataset_valid = (
    GITractDataset(
        data_valid, transforms=transform_valid, load_saved_masks=LOAD_SAVED_MASKS
    )
    if len(data_valid)
    else None
)

dataloader_train = DataLoader(
    dataset_train,
    batch_size=BATCH_SIZE_TRAIN,
    shuffle=True,
    num_workers=DATA_LOADER_NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)
dataloader_valid = None
if dataset_valid is not None:
    dataloader_valid = DataLoader(
        dataset_valid,
        batch_size=BATCH_SIZE_VALID,
        shuffle=False,
        num_workers=DATA_LOADER_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )



## === cell 102
dataset = next(iter(dataloader_train))
img, mask, id_ = dataset
print(img.shape, mask.shape, len(id_))



## === cell 103
idx = 0
print(float(torch.max(img[idx])), float(torch.min(img[idx])))



## === cell 104
print(type(img[idx][0, 0, 0].item()), type(mask[idx][0, 0, 0].item()))




## === cell 105
def display_dataset(dataset_batch, num_images=5, denormalize=False, apply_CLAHE=False):
    img_arr, mask_arr, id_arr = dataset_batch
    num_images = min(num_images, len(img_arr))
    max_cols = 5
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

    plt.tight_layout()
    plt.show()




## === cell 106
display_dataset(dataset, num_images=5, denormalize=True)



## === cell 107
display_dataset(dataset, num_images=5, denormalize=True, apply_CLAHE=True)



## === cell 108
pass




## === cell 109
class _TinyUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(2)
        self.bott = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
        )
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Conv2d(16, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bott(self.pool2(e2))
        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)


if TRAIN_VALID_SPLIT or TEST_PREDICT:
    if _HAS_SMP:
        smp_encoder_weights = "imagenet"
        model = smp.Unet(
            encoder_name="efficientnet-b1",
            encoder_weights=smp_encoder_weights,
            in_channels=3,
            classes=NUM_CLASSES,
        )
    else:
        model = _TinyUNet(in_ch=3, out_ch=NUM_CLASSES)
    model.to(DEVICE)



## === cell 110
pass



## === cell 111
if TRAIN_VALID_SPLIT or TEST_PREDICT:
    optimizer = optim.Adam(model.parameters(), lr=1e-3)



## === cell 112
pass



## === cell 113
if _HAS_SMP:
    dice_loss = smp.losses.DiceLoss(mode="multilabel")
    BCE_loss = smp.losses.SoftBCEWithLogitsLoss()

    def loss_fn(y_pred, y_true, loss_wt=0.5):
        return dice_loss(y_pred, y_true) * loss_wt + BCE_loss(y_pred, y_true) * (
            1 - loss_wt
        )

else:
    bce = nn.BCEWithLogitsLoss()

    def _soft_dice_loss(logits, targets, eps=1e-6):
        probs = torch.sigmoid(logits)
        num = 2 * (probs * targets).sum(dim=(2, 3))
        den = (probs + targets).sum(dim=(2, 3)) + eps
        return 1 - (num / den).mean()

    def loss_fn(y_pred, y_true, loss_wt=0.5):
        return _soft_dice_loss(y_pred, y_true) * loss_wt + bce(y_pred, y_true) * (
            1 - loss_wt
        )




## === cell 114
pass



## === cell 115
pass




## === cell 116
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




## === cell 117
pass




## === cell 118
class HausdorffDistanceCustom:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.h3d_sum = 0.0
        self.image3d_count = 0
        self.organ_h3d_sum = np.zeros(self.num_classes, dtype=np.float64)
        self.organ_count_sum = np.zeros(self.num_classes, dtype=np.float64)

    def _compute_hausdorff_per_organ(self, preds, targets):
        if not _HAS_MONAI:
            return np.nan
        if np.all(preds == targets):
            return 0.0
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
        if not _HAS_MONAI:
            return
        U = (targets | preds).sum((1, 2, 3))
        hausdorff = np.array(
            [
                self._compute_hausdorff_per_organ(preds[i, ...], targets[i, ...])
                for i in range(NUM_CLASSES)
            ]
        )
        non_empty = U > 0
        organ_count = int(non_empty.sum())
        if organ_count != 0:
            hausdorff_per_3dimage = float(np.nansum(hausdorff) / organ_count)
            self.h3d_sum += hausdorff_per_3dimage
            self.image3d_count += 1
        self.organ_h3d_sum += np.nan_to_num(hausdorff, nan=0.0)
        self.organ_count_sum += non_empty.astype(np.float64)

    def compute(self):
        if not _HAS_MONAI or self.image3d_count == 0:
            return 0.0, np.zeros(self.num_classes, dtype=np.float64)
        overall = self.h3d_sum / self.image3d_count
        per_organ = np.divide(self.organ_h3d_sum, np.maximum(self.organ_count_sum, 1.0))
        return overall, per_organ




## === cell 119
dice_score_obj = DiceScoreCustom(num_classes=NUM_CLASSES)
hausdorff_obj = HausdorffDistanceCustom(num_classes=NUM_CLASSES)



## === cell 120
pass




## === cell 121
def one_epoch_train(epoch):
    model.train()
    running_loss = 0.0
    loop = tqdm(dataloader_train, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for imgs, masks, ids in loop:
        imgs = imgs.to(DEVICE, dtype=torch.float)
        masks = masks.to(DEVICE, dtype=torch.float)
        optimizer.zero_grad()
        pred_masks = model(imgs)
        loss = loss_fn(pred_masks, masks)
        loss.backward()
        optimizer.step()
        running_loss += float(loss.item())
        loop.set_postfix(loss=float(loss.item()))
    return running_loss / max(len(dataloader_train), 1)




## === cell 122
slices80_casedays = set()
if len(data_valid):
    slices80_casedays = set(
        data_valid[["case", "day", "slice"]]
        .drop_duplicates()
        .value_counts(["case", "day"])
        .loc[lambda s: s == 80]
        .index
    )




## === cell 123
def one_epoch_valid():
    if dataloader_valid is None:
        return None
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        pred_masks_dict, masks_dict = {}, {}
        for imgs, masks, ids in dataloader_valid:
            imgs = imgs.to(DEVICE, dtype=torch.float)
            masks = masks.to(DEVICE, dtype=torch.float)
            pred_masks = model(imgs)
            loss = loss_fn(pred_masks, masks)
            running_loss += float(loss.item())

            pred_bin = (torch.sigmoid(pred_masks) > 0.5).int()
            masks_bin = masks.int()
            dice_score_obj.update(pred_bin, masks_bin)

            if _HAS_MONAI:
                for p, m, id_ in zip(pred_bin, masks_bin, ids):
                    match = re.match(r"case(\d+)_day(\d+)_slice_(\d+)", id_)
                    if not match:
                        continue
                    caseid, dayid, sliceid = map(int, match.groups())
                    casedayid = (caseid, dayid)
                    pred_masks_dict.setdefault(casedayid, []).append((sliceid, p))
                    masks_dict.setdefault(casedayid, []).append((sliceid, m))

                    if (len(pred_masks_dict[casedayid]) == 144) or (
                        casedayid in slices80_casedays
                        and len(pred_masks_dict[casedayid]) == 80
                    ):
                        pred_sorted = [
                            pp.cpu().numpy()
                            for sid, pp in sorted(
                                pred_masks_dict[casedayid], key=lambda x: x[0]
                            )
                        ]
                        mask_sorted = [
                            mm.cpu().numpy()
                            for sid, mm in sorted(
                                masks_dict[casedayid], key=lambda x: x[0]
                            )
                        ]
                        pred_vol = np.stack(pred_sorted, axis=1)
                        mask_vol = np.stack(mask_sorted, axis=1)
                        hausdorff_obj.update(pred_vol, mask_vol)
                        del pred_masks_dict[casedayid], masks_dict[casedayid]

        avg_loss = running_loss / max(len(dataloader_valid), 1)
        dice_score = dice_score_obj.compute()
        dice_score_obj.reset()

        hausdorff = hausdorff_obj.compute()
        hausdorff_obj.reset()

    return avg_loss, dice_score, hausdorff




## === cell 124
if TRAIN_VALID_SPLIT:
    for epoch in range(EPOCHS):
        loss_train = one_epoch_train(epoch)
        valid_out = one_epoch_valid()
        if valid_out is None:
            print(f"Epoch {epoch+1} | Train Loss: {loss_train:.3f}")
            continue
        loss_valid, dice_score, hausdorff = valid_out
        dice_overall, dice_per_organ = dice_score
        hausdorff_overall, hausdorff_per_organ = hausdorff
        combined_metric = 0.4 * float(dice_overall) + 0.6 * (
            1 - float(hausdorff_overall)
        )
        print(
            f"Epoch {epoch+1} | "
            f"Train Loss: {loss_train:.3f} | Valid Loss: {loss_valid:.3f} | "
            f"Combined metric: {combined_metric:.3f} | "
            f"Dice: {float(dice_overall):.3f} (LB {float(dice_per_organ[0]):.3f}, SB {float(dice_per_organ[1]):.3f}, S {float(dice_per_organ[2]):.3f}) | "
            f"Hausdorff: {float(hausdorff_overall):.3f}"
        )



## === cell 125
if TRAIN_VALID_SPLIT and SAVE_TRAIN_VALID_MODEL:
    torch.save(model.state_dict(), MODEL_PARAMS_FILE_NAME)



## === cell 126
pass



## === cell 127
if TEST_PREDICT:

    def _find_checkpoint(preferred_path: str) -> str | None:
        candidates = []
        if preferred_path:
            candidates.append(preferred_path)

        candidates += [
            "/kaggle/input/git-seg/GIT-Seg-224x224-efficientnet-b1.pth",
            "/kaggle/input/git-seg/GIT-Seg-efficientnet-b1.pth",
            "/kaggle/input/git-seg/pytorch/224x224/1/GIT-Seg-224x224-efficientnet-b1.pth",
            "/kaggle/input/git-seg/pytorch/224x224/1/GIT-Seg-efficientnet-b1.pth",
        ]

        for p in candidates:
            if p and os.path.exists(p):
                return p

        for root, _, files in os.walk("/kaggle/input"):
            for fn in files:
                if fn.endswith(".pth") and (
                    "efficientnet-b1" in fn.lower() or "git-seg" in fn.lower()
                ):
                    p = os.path.join(root, fn)
                    if preferred_path and os.path.basename(p) == os.path.basename(
                        preferred_path
                    ):
                        return p
                    candidates.append(p)

        for p in candidates:
            if p and os.path.exists(p):
                return p
        return None

    def load_state_dict_flexible(model: torch.nn.Module, ckpt_obj, verbose=True):
        if isinstance(ckpt_obj, dict):
            for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
                if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                    ckpt_obj = ckpt_obj[k]
                    break

        if not isinstance(ckpt_obj, dict):
            raise TypeError(f"Unsupported checkpoint type: {type(ckpt_obj)}")

        if any(isinstance(v, dict) for v in ckpt_obj.values()) and not any(
            isinstance(v, torch.Tensor) for v in ckpt_obj.values()
        ):
            dict_candidates = {k: v for k, v in ckpt_obj.items() if isinstance(v, dict)}
            if dict_candidates:
                best_k = max(dict_candidates, key=lambda kk: len(dict_candidates[kk]))
                ckpt_obj = dict_candidates[best_k]

        state_dict = ckpt_obj

        keys = list(state_dict.keys())
        if len(keys) == 0:
            raise ValueError("Empty state_dict in checkpoint.")

        if all(k.startswith("module.") for k in keys):
            state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}
        if all(k.startswith("model.") for k in keys):
            state_dict = {k[len("model.") :]: v for k, v in state_dict.items()}

        model_sd = model.state_dict()
        model_keys = set(model_sd.keys())

        if not (set(state_dict.keys()) & model_keys):
            for pref in ["net.", "encoder.", "unet.", "segmentation_model."]:
                if any(k.startswith(pref) for k in state_dict.keys()):
                    stripped = {k[len(pref) :]: v for k, v in state_dict.items()}
                    if set(stripped.keys()) & model_keys:
                        state_dict = stripped
                        break

        compatible = 0
        total = 0
        for k, v in state_dict.items():
            if k in model_sd and hasattr(v, "shape") and v.shape == model_sd[k].shape:
                compatible += 1
            total += 1

        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        if verbose:
            print(
                f"Checkpoint loaded with strict=False. Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}"
            )
            print(f"Compatible tensors by name+shape: {compatible}/{max(total,1)}")

        if (
            compatible < 50
        ):  # conservative sanity check; typical checkpoints have hundreds+
            print(
                "WARNING: Very few compatible tensors were loaded. "
                "This likely means the model is effectively untrained and may score near 0.0."
            )

        return missing, unexpected, compatible

    ckpt_path = _find_checkpoint(MODEL_PARAMS_LOAD_FILE_PATH)

    if LOAD_MODEL_FOR_TEST_PREDICT:
        if ckpt_path is None:
            print(
                "WARNING: LOAD_MODEL_FOR_TEST_PREDICT=True but no checkpoint was found. "
                "Continuing without loading weights to allow submission generation."
            )
        else:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            try:
                _, _, compatible = load_state_dict_flexible(model, ckpt, verbose=True)
                print("Loaded model weights from:", ckpt_path)
                if compatible < 50:
                    print(
                        "WARNING: Checkpoint loading seems ineffective; submission may still score poorly."
                    )
            except Exception as e:
                print(
                    "WARNING: Found checkpoint but failed to load it flexibly; continuing without weights. Error:",
                    repr(e),
                )

    model.eval()

    data_test = pd.read_csv(DIR_PATH + "test.csv")  # 20400 rows (id,class)
    sample_sub = pd.read_csv(DIR_PATH + "sample_submission.csv")

    data_test["id"] = data_test["id"].astype(str)
    sample_sub["id"] = sample_sub["id"].astype(str)

    data_test[["case", "day", "slice"]] = data_test["id"].str.extract(
        r"case(\d+)_day(\d+)_slice_(\d+)"
    )
    data_test[["case", "day", "slice"]] = data_test[["case", "day", "slice"]].astype(
        np.uint32
    )

    path_df_test = get_path_df(train=False)
    path_df_test[["case", "day", "slice"]] = path_df_test[
        ["case", "day", "slice"]
    ].astype(np.uint32)
    path_df_test[["slice_w", "slice_h"]] = path_df_test[["slice_w", "slice_h"]].astype(
        np.uint32
    )

    test_slices = data_test.drop_duplicates("id")[["id", "case", "day", "slice"]].copy()
    test_slices = test_slices.merge(
        path_df_test[["case", "day", "slice", "image_path", "slice_h", "slice_w"]],
        on=["case", "day", "slice"],
        how="left",
        validate="one_to_one",
    )
    if test_slices["image_path"].isna().any():
        missing = (
            test_slices.loc[test_slices["image_path"].isna(), "id"].head(5).tolist()
        )
        raise RuntimeError(
            f"Some test slice ids could not be mapped to image paths (showing up to 5): {missing}"
        )

    data_test_images = test_slices[["id", "image_path", "slice_h", "slice_w"]].copy()

    if _HAS_ALB:
        transform_test = A.Compose(
            [
                A.Resize(
                    IMAGE_RESIZE[0], IMAGE_RESIZE[1], interpolation=cv2.INTER_NEAREST
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
        transform_test = _SimpleResizeNormalizeToTensor(
            IMAGE_RESIZE, IMAGE_NORMALIZE_MEAN, IMAGE_NORMALIZE_SD
        )

    dataset_test = GITractDataset(
        data_test_images,
        is_test=True,
        transforms=transform_test,
        load_saved_masks=LOAD_SAVED_MASKS,
    )
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=BATCH_SIZE_TEST,
        shuffle=False,
        num_workers=DATA_LOADER_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )



## === cell 128
if TEST_PREDICT:

    def _to_py_str(x):
        if isinstance(x, (bytes, bytearray)):
            return x.decode("utf-8")
        if isinstance(x, np.ndarray):
            if x.shape == ():
                return str(x.item())
            if x.size == 1:
                return str(x.reshape(-1)[0].item())
            return str(x.tolist())
        if torch.is_tensor(x):
            if x.ndim == 0:
                return str(x.item())
            if x.numel() == 1:
                return str(x.view(-1)[0].item())
            return str(x.detach().cpu().tolist())
        return str(x)

    pred_by_id = {}

    with torch.no_grad():
        for imgs, ids, heights, widths in tqdm(dataloader_test, desc="Predicting"):
            imgs = imgs.to(DEVICE, dtype=torch.float)
            pred_masks = model(imgs)
            pred_masks = (torch.sigmoid(pred_masks) > 0.5).int()
            pred_masks = pred_masks.permute(0, 2, 3, 1).cpu().numpy()  # [B,H,W,C]

            for mask, id_, h, w in zip(pred_masks, ids, heights, widths):
                id_key = _to_py_str(id_)
                mask_orig_size = cv2.resize(
                    mask.astype(np.uint8),
                    dsize=(int(w), int(h)),
                    interpolation=cv2.INTER_NEAREST,
                )
                mask_orig_size = np.ascontiguousarray(mask_orig_size, dtype=np.uint8)
                rles = [
                    rle_encode(mask_orig_size[..., chid]) for chid in range(NUM_CLASSES)
                ]
                pred_by_id[id_key] = dict(zip(CLASS_NAMES, rles))

    expected_ids = set(data_test["id"].astype(str).unique().tolist())
    got_ids = set(pred_by_id.keys())
    missing_ids = sorted(list(expected_ids - got_ids))
    if len(missing_ids) > 0:
        raise RuntimeError(
            f"Missing predictions for {len(missing_ids)} slice ids (showing up to 5): {missing_ids[:5]}"
        )

    out_rows = []
    for sub_id, cls in zip(data_test["id"].tolist(), data_test["class"].tolist()):
        rle = pred_by_id.get(str(sub_id), {}).get(cls, "")
        out_rows.append((sub_id, cls, rle))
    submission_df = pd.DataFrame(out_rows, columns=["id", "class", "predicted"])

    submission_df = sample_sub[["id", "class"]].merge(
        submission_df, on=["id", "class"], how="left", validate="one_to_one"
    )
    if submission_df["predicted"].isna().any():
        submission_df["predicted"] = submission_df["predicted"].fillna("")

    non_empty = int((submission_df["predicted"].astype(str).str.len() > 0).sum())
    print("Non-empty predicted RLE rows:", non_empty, "/", len(submission_df))
    if non_empty < 50:
        raise RuntimeError(
            "Submission appears to be almost entirely empty; this typically yields ~0.0. "
            "Checkpoint likely didn't load correctly."
        )

    submission_df.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", submission_df.shape)
    print(submission_df.head())

## --- ERROR in cell 128, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/3127175937.py in <cell line: 0>()
     64     print("Non-empty predicted RLE rows:", non_empty, "/", len(submission_df))
     65     if non_empty < 50:
---> 66         raise RuntimeError(
     67             "Submission appears to be almost entirely empty; this typically yields ~0.0. "
     68             "Checkpoint likely didn't load correctly."

RuntimeError: Submission appears to be almost entirely empty; this typically yields ~0.0. Checkpoint likely didn't load correctly.
