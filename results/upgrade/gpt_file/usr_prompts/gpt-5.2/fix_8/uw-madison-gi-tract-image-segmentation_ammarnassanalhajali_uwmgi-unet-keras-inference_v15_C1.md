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

0.4484978484787068

# 6. Current score

0.0044

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0029) has done: 'I fix the TensorFlow/protobuf import crash by forcing TensorFlow to use the python protobuf implementation before importing TF, and by using a safer import order. Then I fix the missing pretrained model path by falling back to a minimal U-Net-like model with the same 128×128×3 input and 3-channel sigmoid output so the notebook can run end-to-end even without external model files. Finally, I fix the DataGenerator/test prediction length mismatch (it was dropping the last partial batch) and ensure the RLE encoding uses the required column-major (Fortran) order so the submission format is valid and scores move toward the target instead of being penalized by wrong encoding.'
- What this solution (achieved 0.0) has done: 'I fix the protobuf/TensorFlow crash by forcing a compatible protobuf runtime mode before importing TensorFlow and, if needed, downgrading protobuf in-notebook to a version that matches TF’s expected API (this is the root cause of `MessageFactory.GetPrototype` failures). Then I correct the submission id parsing so slice filenames are built correctly (the current `slice` extraction is wrong), which currently causes missing/incorrect image-path joins and effectively random/empty predictions (driving the very low score). Finally, I make RLE decoding/encoding consistent with the competition’s column-major convention (decode currently reshapes in C-order), which prevents metric penalties from malformed masks; these are minimal, logic-preserving fixes aimed at moving the score up toward the target.'
- What this solution (achieved 0.0) has done: 'I fix the root cause of the “PyDataset has length 0” error by ensuring `df_train` is never empty after the path-merge step (the current `slice` parsing is wrong and breaks the join to PNG files). I correct `slice` extraction to keep the `slice_XXXX` token intact and build the join key robustly using `os.path.dirname(path)` rather than brittle string splits. These are minimal, logic-preserving fixes that unblock inference and produce a non-empty generator, which should raise the score from 0.0 toward the target simply by generating valid, correctly-aligned predictions and RLEs. The rest of the model/prediction/RLE logic is kept the same.'
- What this solution (achieved 0.0) has done: 'I fix the root cause of the empty merged dataframe by correcting how `path_partial` is constructed so it matches the actual PNG filename stem (`slice_XXXX`) inside each case/day scans folder. This make `df_train` non-empty, which in turn fixes the downstream “PyDataset has length 0” error during `model.predict`. I also make the width/height extraction robust by parsing them from the actual PNG filename (not from `path_partial`) to avoid silent mis-parsing. These changes are minimal and keep the model/inference/RLE logic the same, but should move the score up from 0.0 by producing valid, correctly-aligned masks and a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'I fix the root cause of the empty `df_train` (and the downstream “PyDataset has length 0”) by correcting the PNG filename parsing and the `slice` join key: the code currently swaps width/height indices and builds a `path_partial` that can silently fail to match real scan filenames. I make the merge robust by extracting the actual `slice_XXXX` token from each PNG path and joining on a stable key (`case/day/slice`) instead of brittle string concatenations, while keeping the overall data flow and model inference logic unchanged. I also ensure `df_train` is constructed in the correct 1-row-per-slice order and that the prediction generator always has non-zero length. These are execution-unblocking fixes and should move the score up from 0.0 toward the target by producing correctly-aligned, valid RLE masks.'
- What this solution (achieved 0.0044) has done: 'I fix the root cause of the empty merge by correctly parsing `id` into the real `slice_XXXX` token (it was using only the numeric part, so it never matched PNG filenames) and by joining using a robust `case/day/slice` key extracted consistently from both the CSV ids and the discovered PNG paths. Then I keep the existing inference + RLE pipeline intact, only adjusting the dataframe construction so `df_train` is built directly from the merged rows (instead of relying on `[::3]` ordering assumptions that can silently break). These changes are execution-unblocking and should raise the score from 0.0 toward the target simply by producing non-empty, correctly-aligned predictions and a valid `submission.csv`. No model/loss/thresholding logic is changed.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings

warnings.filterwarnings("ignore")

import gc
from glob import glob

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

try:
    import google.protobuf as _pb
    from packaging import version as _version

    _pb_ver = getattr(_pb, "__version__", "0.0.0")
    if _version.parse(_pb_ver) >= _version.parse("4.21.0"):
        import sys, subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf<4.21"]
        )
except Exception:
    pass

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

tf.random.set_seed(42)
np.random.seed(42)



## === cell 1
BATCH_SIZE = 16
EPOCHS = 30
n_splits = 5
fold_selected = 1  # 1..5



## === cell 2
df = pd.read_csv(
    "../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv"
)
DEBUG = False
if df.shape[0] == 0:
    DEBUG = True
if DEBUG:
    df = pd.read_csv("../input/uw-madison-gi-tract-image-segmentation/train.csv")
    df.pop("segmentation")
    df["predicted"] = ""



## === cell 3
df.rename(columns={"class": "class_name"}, inplace=True)
df["case"] = df["id"].apply(lambda x: int(x.split("_")[0].replace("case", "")))
df["day"] = df["id"].apply(lambda x: int(x.split("_")[1].replace("day", "")))
df["slice"] = df["id"].apply(lambda x: "_".join(x.split("_")[2:4]))  # 'slice_0001'

if DEBUG:
    TRAIN_DIR = "../input/uw-madison-gi-tract-image-segmentation/train"
else:
    TRAIN_DIR = "../input/uw-madison-gi-tract-image-segmentation/test"

all_images = glob(os.path.join(TRAIN_DIR, "**", "*.png"), recursive=True)
if len(all_images) == 0:
    raise FileNotFoundError(f"No PNGs found under {TRAIN_DIR}")


def _parse_png_meta(p: str):
    """
    Returns: case(int), day(int), slice_token(str like 'slice_0001'), width(int), height(int), path(str)
    based on directory structure and filename: slice_0001_266_266_1.50_1.50.png
    """
    parts = p.replace("\\", "/").split("/")
    case_dir = parts[-4]  # case101
    day_dir = parts[-3]  # case101_day20

    fname = os.path.basename(p)[:-4]
    fparts = fname.split("_")
    if len(fparts) < 6:
        raise ValueError(f"Unexpected PNG filename format: {os.path.basename(p)}")

    slice_token = "_".join(fparts[:2])  # slice_0001
    width = int(float(fparts[2]))
    height = int(float(fparts[3]))

    case = int(case_dir.replace("case", ""))
    day = int(day_dir.split("_day")[-1])
    return case, day, slice_token, width, height, p


png_meta = []
for p in all_images:
    try:
        png_meta.append(_parse_png_meta(p))
    except Exception:
        continue

tmp_df = pd.DataFrame(
    png_meta, columns=["case", "day", "slice", "width", "height", "path"]
)

df = df.merge(tmp_df, on=["case", "day", "slice"], how="inner")

if df.shape[0] == 0:
    raise RuntimeError(
        "Merge between IDs and PNG paths produced 0 rows.\n"
        f"TRAIN_DIR={TRAIN_DIR}\n"
        f"Example df slices: {df[['id']].head(3).to_dict(orient='records')}\n"
        f"Example png slices: {tmp_df[['case','day','slice']].head(3).to_dict(orient='records')}\n"
        "Check id parsing (slice token) and folder selection."
    )

del png_meta, tmp_df, all_images
df.head(5)



## === cell 4
df = df.sort_values(["id", "class_name"]).reset_index(drop=True)

df_wide = df.pivot_table(
    index=["id", "case", "day", "slice", "width", "height", "path"],
    columns="class_name",
    values="predicted",
    aggfunc="first",
    fill_value="",
).reset_index()

for col in ["large_bowel", "small_bowel", "stomach"]:
    if col not in df_wide.columns:
        df_wide[col] = ""

df_train = df_wide[
    [
        "id",
        "path",
        "case",
        "day",
        "slice",
        "width",
        "height",
        "large_bowel",
        "small_bowel",
        "stomach",
    ]
].copy()

del df, df_wide
df_train.reset_index(inplace=True, drop=True)
df_train.fillna("", inplace=True)
df_train.head(5)



## === cell 5
print(df_train.shape)
if DEBUG:
    df_train = df_train.sample(frac=0.05, random_state=42).reset_index(drop=True)
print(df_train.shape)

if df_train.shape[0] == 0:
    raise RuntimeError(
        "After merging ids to PNG paths, df_train is empty. "
        "This indicates an id->slice->filename join issue."
    )



## === cell 6
gc.collect()




## === cell 7
def rle_encode(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted.

    Competition expects pixels numbered top-to-bottom then left-to-right,
    which corresponds to flatten(order='F') for a (H,W) array.
    """
    if img.ndim != 2:
        img = img.squeeze()
    pixels = img.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(int(x)) for x in runs)


def rle_decode(mask_rle, shape, color=1):
    """
    mask_rle: run-length as string formatted (start length), can be '' for empty mask
    shape: (height,width,channels) of array to return
    Returns numpy array, 1 - mask, 0 - background

    Decode matches encode's column-major convention via order='F'.
    """
    h, w, c = shape
    if (
        mask_rle is None
        or mask_rle == ""
        or (isinstance(mask_rle, float) and np.isnan(mask_rle))
    ):
        return np.zeros((h, w, c), dtype=np.float32)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths

    img = np.zeros((h * w, c), dtype=np.float32)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = color

    return img.reshape((h, w, c), order="F")


def build_masks(labels, input_shape, colors=True):
    height, width = input_shape
    if colors:
        mask = np.zeros((height, width, 3), dtype=np.float32)
        for label in labels:
            mask += rle_decode(label, shape=(height, width, 3), color=np.random.rand(3))
    else:
        mask = np.zeros((height, width, 1), dtype=np.float32)
        for label in labels:
            mask += rle_decode(label, shape=(height, width, 1))
    mask = mask.clip(0, 1)
    return mask




## === cell 8
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size=BATCH_SIZE, subset="train", shuffle=False):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.shuffle = shuffle
        self.subset = subset
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        batch_indexes = self.indexes[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        cur_bs = len(batch_indexes)

        X = np.empty((cur_bs, 128, 128, 3), dtype=np.float32)
        if self.subset == "train":
            y = np.empty((cur_bs, 128, 128, 3), dtype=np.float32)

        for i, row_idx in enumerate(batch_indexes):
            img_path = self.df.loc[row_idx, "path"]
            w = int(self.df.loc[row_idx, "width"])
            h = int(self.df.loc[row_idx, "height"])

            img = self.__load_grayscale(img_path)  # (128,128,1)
            img3 = np.repeat(img, 3, axis=-1)  # (128,128,3)
            X[i] = img3

            if self.subset == "train":
                for k, j in zip([0, 1, 2], ["large_bowel", "small_bowel", "stomach"]):
                    rles = self.df.loc[row_idx, j]
                    masks = rle_decode(rles, shape=(h, w, 1))  # (h,w,1)
                    masks = cv2.resize(
                        masks, (128, 128), interpolation=cv2.INTER_NEAREST
                    )
                    if masks.ndim == 2:
                        masks = masks[..., None]
                    y[i, :, :, k] = masks[:, :, 0]

        if self.subset == "train":
            return X, y
        return X

    def __load_grayscale(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32)
        img = img / 255.0
        img = np.expand_dims(img, axis=-1)
        return img




## === cell 9
gc.collect()



## === cell 10
from tensorflow.keras.losses import binary_crossentropy


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


def dice_loss(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2.0 * K.sum(intersection) + smooth) / (
        K.sum(y_true_f) + K.sum(y_pred_f) + smooth
    )
    return 1.0 - score


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(tf.cast(y_true, tf.float32), y_pred) + 0.5 * dice_loss(
        tf.cast(y_true, tf.float32), y_pred
    )


class FixedDropout(tf.keras.layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        symbolic_shape = K.shape(inputs)
        noise_shape = [
            symbolic_shape[axis] if shape is None else shape
            for axis, shape in enumerate(self.noise_shape)
        ]
        return tuple(noise_shape)




## === cell 11
gc.collect()



## === cell 12
custom_objects = {
    "FixedDropout": FixedDropout,
    "dice_coef": dice_coef,
    "iou_coef": iou_coef,
    "bce_dice_loss": bce_dice_loss,
}

model_path = "../input/uwmgi-unet-keras/model.h5"


def build_fallback_unet(input_shape=(128, 128, 3), n_classes=3):
    inputs = keras.layers.Input(shape=input_shape)

    c1 = keras.layers.Conv2D(16, 3, activation="relu", padding="same")(inputs)
    c1 = keras.layers.Conv2D(16, 3, activation="relu", padding="same")(c1)
    p1 = keras.layers.MaxPooling2D()(c1)

    c2 = keras.layers.Conv2D(32, 3, activation="relu", padding="same")(p1)
    c2 = keras.layers.Conv2D(32, 3, activation="relu", padding="same")(c2)
    p2 = keras.layers.MaxPooling2D()(c2)

    b = keras.layers.Conv2D(64, 3, activation="relu", padding="same")(p2)
    b = keras.layers.Conv2D(64, 3, activation="relu", padding="same")(b)

    u2 = keras.layers.UpSampling2D()(b)
    u2 = keras.layers.Concatenate()([u2, c2])
    c3 = keras.layers.Conv2D(32, 3, activation="relu", padding="same")(u2)
    c3 = keras.layers.Conv2D(32, 3, activation="relu", padding="same")(c3)

    u1 = keras.layers.UpSampling2D()(c3)
    u1 = keras.layers.Concatenate()([u1, c1])
    c4 = keras.layers.Conv2D(16, 3, activation="relu", padding="same")(u1)
    c4 = keras.layers.Conv2D(16, 3, activation="relu", padding="same")(c4)

    outputs = keras.layers.Conv2D(n_classes, 1, activation="sigmoid", padding="same")(
        c4
    )
    m = keras.Model(inputs, outputs)
    return m


if os.path.exists(model_path):
    model = load_model(model_path, custom_objects=custom_objects)
else:
    model = build_fallback_unet((128, 128, 3), 3)

gc.collect()



## === cell 13
pred_batches = DataGenerator(df_train, batch_size=1, subset="test", shuffle=False)
if len(pred_batches) == 0:
    raise RuntimeError(
        "Prediction generator length is 0; df_train is unexpectedly empty."
    )
gc.collect()
LOGITS = model.predict(pred_batches, verbose=1)
gc.collect()



## === cell 14
len(LOGITS)



## === cell 15
lbs, sbs, sts = [], [], []
for index in tqdm(range(df_train.shape[0]), total=df_train.shape[0]):
    h = int(df_train.iloc[index]["height"])
    w = int(df_train.iloc[index]["width"])

    pred0 = cv2.resize(LOGITS[index, :, :, 0], (w, h), interpolation=cv2.INTER_NEAREST)
    pred1 = cv2.resize(LOGITS[index, :, :, 1], (w, h), interpolation=cv2.INTER_NEAREST)
    pred2 = cv2.resize(LOGITS[index, :, :, 2], (w, h), interpolation=cv2.INTER_NEAREST)

    pred_arr = (pred0 >= 0.5).astype("uint8")
    lbs.append(rle_encode(pred_arr))

    pred_arr = (pred1 >= 0.5).astype("uint8")
    sbs.append(rle_encode(pred_arr))

    pred_arr = (pred2 >= 0.5).astype("uint8")
    sts.append(rle_encode(pred_arr))

del LOGITS
gc.collect()



## === cell 16
df_ids = df_train[["id"]].copy()
gc.collect()



## === cell 17
ids, classes, rles = [], [], []
for i, (_, row) in enumerate(tqdm(df_ids.iterrows(), total=df_ids.shape[0])):
    ids.extend([row["id"]] * 3)
    classes.extend(["large_bowel", "small_bowel", "stomach"])
    rles.extend([lbs[i], sbs[i], sts[i]])



## === cell 18
sub = pd.DataFrame({"id": ids, "class": classes, "predicted": rles})

sample = pd.read_csv(
    "../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv"
)
submission = sample.merge(sub, on=["id", "class"], how="left", suffixes=("", "_pred"))

if "predicted_pred" in submission.columns:
    submission["predicted"] = submission["predicted_pred"].fillna("")
    submission = submission[["id", "class", "predicted"]]
else:
    submission["predicted"] = submission["predicted"].fillna("")

submission.to_csv("submission.csv", index=False)



## === cell 19
submission.head()
