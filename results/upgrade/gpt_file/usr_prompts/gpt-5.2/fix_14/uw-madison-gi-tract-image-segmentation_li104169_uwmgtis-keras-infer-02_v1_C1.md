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

# 5. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault(
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_C_DESCRIPTORS", "1"
)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.pop("TF_USE_LEGACY_KERAS", None)

import warnings

warnings.filterwarnings("ignore")

from glob import glob
from pathlib import Path
from time import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import cv2

from joblib import dump, load

from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    Activation,
    MaxPool2D,
    Conv2DTranspose,
    Concatenate,
    Input,
    Dropout,
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib as mpl
from matplotlib.patches import Rectangle

pd.set_option("display.max_columns", 200)
pd.set_option("display.max_colwidth", 200)

try:
    from IPython.display import display
except Exception:

    def display(x):
        print(x)


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


print("TF version:", tf.__version__)



## === cell 1
repertory = "/kaggle/input/"

DIR = repertory + "uw-madison-gi-tract-image-segmentation/"
TRAIN_DIR = DIR + "train"
TEST_DIR = DIR + "test"
train_csv = DIR + "train.csv"
test_csv = DIR + "test.csv"
sample_sub = DIR + "sample_submission.csv"

df_train = pd.read_csv(train_csv)
df_train.head(10)




## === cell 2
class CFG:
    BATCH_SIZE = 64
    img_size = (256, 256, 3)  # keep as in original
    n_fold = 5
    fold_selected = 1
    epochs = 100
    seed = 42
    steps_per_epoch_train = None
    steps_per_epoch_val = None


seed_everything(CFG.seed)



## === cell 3
_SCAN_CACHE = {}  # subset -> dict(partial_key->full_path)


def _build_scan_path_map(root_dir):
    m = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if not fn.endswith(".png"):
                continue
            p = os.path.join(dirpath, fn)
            key = p.rsplit("_", 4)[0]
            m[key] = p
    return m


def preprocessing(df, subset="train"):
    df = df.copy()

    if "class" in df.columns:
        df["class"] = df["class"].astype(str)

    parts = df["id"].astype(str).str.split("_", expand=True)
    df["case"] = parts[0].str.replace("case", "", regex=False).astype(np.int16)
    df["day"] = parts[1].str.replace("day", "", regex=False).astype(np.int16)
    df["slice"] = parts[3]  # keep as string as in original usage

    root = TRAIN_DIR if subset == "train" else TEST_DIR
    if subset not in _SCAN_CACHE:
        _SCAN_CACHE[subset] = _build_scan_path_map(root)

    base = root
    df["path_partial"] = (
        base
        + "/case"
        + df["case"].astype(str)
        + "/case"
        + df["case"].astype(str)
        + "_day"
        + df["day"].astype(str)
        + "/scans/slice_"
        + df["slice"].astype(str)
    )

    m = _SCAN_CACHE[subset]
    df["path"] = df["path_partial"].map(m)
    df = df.drop(columns=["path_partial"])

    stem = df["path"].astype(str).str[:-4]
    t = stem.str.rsplit("_", n=4, expand=True)
    df["width"] = t[1].astype(np.int16)
    df["height"] = t[2].astype(np.int16)
    df["px_spacing_h"] = t[3].astype(np.float32)
    df["px_spacing_w"] = t[4].astype(np.float32)

    return df




## === cell 4
train_df = preprocessing(df_train, subset="train")
train_df.head()




## === cell 5
def rle_decode(mask_rle, shape):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    if mask_rle is None:
        return np.zeros(shape[:2], dtype=np.uint8)
    if isinstance(mask_rle, float) and np.isnan(mask_rle):
        return np.zeros(shape[:2], dtype=np.uint8)
    if isinstance(mask_rle, str) and mask_rle.strip() == "":
        return np.zeros(shape[:2], dtype=np.uint8)

    s = np.asarray(mask_rle.split(), dtype=np.int64)
    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    img_starts = starts
    img_ends = ends
    img[img_starts] = 1
    end_mask = img_ends < img.size
    img[img_ends[end_mask]] ^= 1
    img = np.cumsum(img, dtype=np.uint8)
    return img.reshape(shape[0], shape[1])


def rle_encode(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    """
    pixels = img.reshape(-1).astype(np.uint8, copy=False)
    if pixels.size == 0:
        return ""
    padded = np.empty(pixels.size + 2, dtype=np.uint8)
    padded[0] = 0
    padded[-1] = 0
    padded[1:-1] = pixels
    changes = np.nonzero(padded[1:] != padded[:-1])[0] + 1
    if changes.size == 0:
        return ""
    changes[1::2] -= changes[::2]
    return " ".join(map(str, changes))




## === cell 6
def id2mask(id_):
    itrain_df = train_df[train_df["id"] == id_]
    wh = itrain_df[["height", "width"]].iloc[0]
    shape = (wh.height, wh.width, 3)
    mask = np.zeros(shape, dtype=np.uint8)
    for i, class_ in enumerate(["large_bowel", "small_bowel", "stomach"]):
        ctrain_df = itrain_df[itrain_df["class"] == class_]
        rle = ctrain_df.segmentation.squeeze() if len(ctrain_df) else ""
        if (
            len(ctrain_df)
            and not (isinstance(rle, float) and np.isnan(rle))
            and str(rle).strip() != ""
        ):
            mask[..., i] = rle_decode(rle, shape[:2])
    return mask


def rgb2gray(mask):
    pad_mask = np.pad(mask, pad_width=[(0, 0), (0, 0), (1, 0)])
    gray_mask = pad_mask.argmax(-1)
    return gray_mask


def gray2rgb(mask):
    rgb_mask = tf.keras.utils.to_categorical(mask, num_classes=4)
    return rgb_mask[..., 1:].astype(mask.dtype)




## === cell 7
def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = img.astype("float32")  # original is uint16
    denom = img.max() - img.min()
    if denom < 1e-6:
        img = np.zeros_like(img, dtype=np.uint8)
    else:
        img = (img - img.min()) / denom * 255.0
        img = img.astype("uint8")
    return img


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




## === cell 8
def restructure(df, subset="train"):
    df_out = pd.DataFrame({"id": df["id"][::3]})

    if subset == "train":
        df_out["large_bowel"] = df["segmentation"][::3].values
        df_out["small_bowel"] = df["segmentation"][1::3].values
        df_out["stomach"] = df["segmentation"][2::3].values

    df_out["path"] = df["path"][::3].values
    df_out["case"] = df["case"][::3].values
    df_out["day"] = df["day"][::3].values
    df_out["slice"] = df["slice"][::3].values
    df_out["width"] = df["width"][::3].values
    df_out["height"] = df["height"][::3].values

    df_out = df_out.reset_index(drop=True)
    df_out = df_out.fillna("")

    if subset == "train":
        df_out["count"] = np.sum(df_out.iloc[:, 1:4] != "", axis=1).values

    display(df_out.sample(5))
    return df_out




## === cell 9
DF_train = restructure(train_df, subset="train")




## === cell 10
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        df,
        batch_size=CFG.BATCH_SIZE,
        subset="train",
        shuffle=False,
        img_shape=CFG.img_size,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.shuffle = shuffle
        self.subset = subset
        self.batch_size = batch_size
        self.img_shape = img_shape
        self._H, self._W, self._C = img_shape
        self._dsize = (self._W, self._H)  # cv2 uses (width,height)

        self._paths = self.df["path"].to_numpy()
        self._widths = (
            self.df["width"].to_numpy() if "width" in self.df.columns else None
        )
        self._heights = (
            self.df["height"].to_numpy() if "height" in self.df.columns else None
        )
        self._ids = self.df["id"].to_numpy() if "id" in self.df.columns else None
        self._classes = (
            self.df["class"].to_numpy() if "class" in self.df.columns else None
        )

        if subset == "train":
            self._rle_lb = self.df["large_bowel"].to_numpy()
            self._rle_sb = self.df["small_bowel"].to_numpy()
            self._rle_st = self.df["stomach"].to_numpy()

        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        current_bs = len(indexes)

        X = np.empty((current_bs, self._H, self._W, self._C), dtype=np.float32)
        y = np.empty((current_bs, self._H, self._W, self._C), dtype=np.float32)

        if self.subset != "train":
            id_ = [None] * current_bs
            heights = [None] * current_bs
            widths = [None] * current_bs
            classes = [None] * current_bs

        for i, idx in enumerate(indexes):
            img_path = self._paths[idx]

            w = int(self._widths[idx])
            h = int(self._heights[idx])

            img = self.__load_grayscale(img_path)
            X[i] = img

            if self.subset == "train":
                rles0 = self._rle_lb[idx]
                rles1 = self._rle_sb[idx]
                rles2 = self._rle_st[idx]
                for k, rles in enumerate((rles0, rles1, rles2)):
                    mask = rle_decode(rles, shape=(h, w))
                    mask = cv2.resize(
                        mask, (self._W, self._H), interpolation=cv2.INTER_NEAREST
                    )
                    y[i, :, :, k] = mask.astype(np.float32)
            else:
                id_[i] = self._ids[idx]
                heights[i] = h
                widths[i] = w
                classes[i] = self._classes[idx]

        if self.subset == "train":
            return X, y
        else:
            return X, id_, widths, heights, classes

    def __load_grayscale(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
        img = cv2.resize(img, self._dsize, interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32, copy=False)
        mn = float(img.min())
        mx = float(img.max())
        denom = mx - mn
        if denom < 1e-6:
            img.fill(0.0)
        else:
            img = (img - mn) / denom  # scale to [0,1]
        img = img[..., None]  # (H,W,1)
        if self._C == 3:
            img = np.repeat(img, 3, axis=-1)
        return img




## === cell 11
def dice_coef(y_true, y_pred, smooth=1e-6):
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




## === cell 12
def conv_block(input, num_filters, batchnorm):
    x = Conv2D(num_filters, kernel_size=(3, 3), padding="same")(input)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(num_filters, kernel_size=(3, 3), padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def encoder_block(input, num_filters, dropout=False, batchnorm=True):
    x = conv_block(input, num_filters, batchnorm)
    p = MaxPool2D((2, 2))(x)
    if dropout:
        p = Dropout(0.3)(p)
    return x, p


def decoder_block(input, skip_features, num_filters, dropout=False, batchnorm=True):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    if dropout:
        x = Dropout(0.3)(x)
    x = conv_block(x, num_filters, batchnorm)
    return x


def build_unet(input_shape, dropout=False, batchnorm=True, activation="sigmoid"):
    inputs = Input(shape=input_shape)

    s0, p0 = encoder_block(inputs, 32, dropout, batchnorm)
    s1, p1 = encoder_block(p0, 64, dropout, batchnorm)
    s2, p2 = encoder_block(p1, 128, dropout, batchnorm)
    s3, p3 = encoder_block(p2, 256, dropout, batchnorm)
    s4, p4 = encoder_block(p3, 512, dropout, batchnorm)
    s5, p5 = encoder_block(p4, 1024, dropout, batchnorm)

    b1 = conv_block(p5, 2048, batchnorm)

    d0 = decoder_block(b1, s5, 1024, dropout, batchnorm)
    d1 = decoder_block(d0, s4, 512, dropout, batchnorm)
    d2 = decoder_block(d1, s3, 256, dropout, batchnorm)
    d3 = decoder_block(d2, s2, 128, dropout, batchnorm)
    d4 = decoder_block(d3, s1, 64, dropout, batchnorm)
    d5 = decoder_block(d4, s0, 32, dropout, batchnorm)

    outputs = Conv2D(3, 1, padding="same", activation=activation)(d5)

    model = Model(inputs, outputs, name="U-Net")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=bce_dice_loss,
        metrics=[iou_coef, dice_coef],
    )
    return model


def plot_train(model):
    losses = (
        model
        if isinstance(model, pd.DataFrame)
        else pd.DataFrame(model.history.history)
    )
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(losses["loss"].index, losses["loss"], label="Train_Loss")
    plt.plot(losses["val_loss"].index, losses["val_loss"], label="Val_loss")
    plt.title("LOSS")
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(losses["dice_coef"].index, losses["dice_coef"], label="Train_dice_coef")
    plt.plot(
        losses["val_dice_coef"].index, losses["val_dice_coef"], label="Val_dice_coef"
    )
    plt.title("DICE")
    plt.xlabel("Epoch")
    plt.ylabel("dice_coef")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(losses["iou_coef"].index, losses["iou_coef"], label="Train_iou_coef")
    plt.plot(losses["val_iou_coef"].index, losses["val_iou_coef"], label="Val_iou_coef")
    plt.title("IOU")
    plt.xlabel("Epoch")
    plt.ylabel("iou_coef")
    plt.legend()
    plt.show()


def fit_model(
    model,
    model_name,
    train_dataset,
    validation_dataset,
    model_path_override=None,
    score_path_override=None,
):
    model_path = model_path_override or (models_path + str(model_name) + ".h5")
    score_path = score_path_override or (
        results_path + "score_" + str(model_name) + ".joblib"
    )

    if model_path is not None and os.path.isfile(model_path):
        model = load_model(
            model_path,
            custom_objects={
                "bce_dice_loss": bce_dice_loss,
                "iou_coef": iou_coef,
                "dice_coef": dice_coef,
            },
        )
        if score_path is not None and os.path.isfile(score_path):
            results = load(score_path)
            plot_train(results)
        else:
            results = None
    else:
        assert (
            train_dataset is not None and validation_dataset is not None
        ), "Training requested but datasets are None. Provide generators or ensure weights exist."
        early_stop = EarlyStopping(monitor="val_loss", patience=5)
        model.fit(
            train_dataset,
            epochs=CFG.epochs,
            validation_data=validation_dataset,
            callbacks=[early_stop],
        )

        results = pd.DataFrame(model.history.history)
        plot_train(model)

        dump(
            results,
            results_path + "score_" + str(model_name) + ".joblib",
            compress=True,
        )
        model.save(models_path + str(model_name) + ".h5")

    return model, results




## === cell 13
path_load_infer = repertory + "uwmgtis-keras-train-02/"
models_path = path_load_infer
results_path = path_load_infer

input_shape = CFG.img_size
model = build_unet(input_shape, dropout=True, batchnorm=True)

candidate_model_files = ["U-net.h5", "U-Net.h5", "unet.h5", "model.h5"]
candidate_score_files = [
    "score_U-net.joblib",
    "score_U-Net.joblib",
    "score_unet.joblib",
    "score_model.joblib",
]


def find_first_existing(base_dir, filenames):
    if base_dir is not None:
        for fn in filenames:
            p = os.path.join(base_dir, fn)
            if os.path.isfile(p):
                return p
        for fn in filenames:
            hits = glob(os.path.join(base_dir, "**", fn), recursive=True)
            hits = [h for h in hits if os.path.isfile(h)]
            if len(hits):
                hits.sort()
                return hits[0]
    return None


def find_any_h5_in_kaggle_inputs():
    roots = [
        "/kaggle/input/uwmgtis-keras-train-02",
        "/kaggle/input/uw-madison-gi-tract-image-segmentation",
    ]
    hits = []
    for r in roots:
        if os.path.isdir(r):
            hits.extend(glob(os.path.join(r, "**", "*.h5"), recursive=True))
    hits = [h for h in hits if os.path.isfile(h)]
    hits.sort()
    return hits[0] if hits else None


found_model_path = find_first_existing(models_path, candidate_model_files)
found_score_path = find_first_existing(results_path, candidate_score_files)

if found_model_path is None:
    found_model_path = find_any_h5_in_kaggle_inputs()

have_any_weights = found_model_path is not None
print(
    "Found pretrained:",
    have_any_weights,
    "| model:",
    found_model_path,
    "| scores:",
    found_score_path,
)



## === cell 14
if not have_any_weights:
    fold = KFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
    idxs = np.arange(len(DF_train))
    tr_idx, va_idx = list(fold.split(idxs))[CFG.fold_selected]

    train_slice_df = DF_train.iloc[tr_idx].reset_index(drop=True)
    val_slice_df = DF_train.iloc[va_idx].reset_index(drop=True)

    train_dataset = DataGenerator(
        train_slice_df, batch_size=CFG.BATCH_SIZE, subset="train", shuffle=True
    )
    validation_dataset = DataGenerator(
        val_slice_df, batch_size=CFG.BATCH_SIZE, subset="train", shuffle=False
    )

    if CFG.steps_per_epoch_train is not None:
        train_dataset = train_dataset
    if CFG.steps_per_epoch_val is not None:
        validation_dataset = validation_dataset

    models_path = "/kaggle/working/"
    results_path = "/kaggle/working/"

    model, results = fit_model(
        model,
        "U-net",
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        model_path_override=os.path.join(models_path, "U-net.h5"),
        score_path_override=os.path.join(results_path, "score_U-net.joblib"),
    )
else:
    model, results = fit_model(
        model,
        "U-net",
        train_dataset=None,
        validation_dataset=None,
        model_path_override=found_model_path,
        score_path_override=found_score_path,
    )



## === cell 15
sub_df = pd.read_csv(sample_sub)

if not len(sub_df):
    debug = True
    sub_df = pd.read_csv(train_csv)
    test_df = preprocessing(df_train, subset="train")
    test_df = test_df[: 1000 * 3]
else:
    debug = False
    test_df = preprocessing(sub_df, subset="test")

assert (
    "class" in test_df.columns
), "preprocessing() must preserve 'class' for test inference"
test_df.head(5)




## === cell 16
def _load_batch_grayscale(paths, dsize, out_channels=3):
    bs = len(paths)
    H, W = dsize[1], dsize[0]
    X = np.empty((bs, H, W, out_channels), dtype=np.float32)
    for i, p in enumerate(paths):
        img = cv2.imread(p, cv2.IMREAD_ANYDEPTH)
        img = cv2.resize(img, dsize, interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32, copy=False)
        mn = float(img.min())
        mx = float(img.max())
        denom = mx - mn
        if denom < 1e-6:
            img.fill(0.0)
        else:
            img = (img - mn) / denom
        img = img[..., None]
        if out_channels == 3:
            img = np.repeat(img, 3, axis=-1)
        X[i] = img
    return X


def infer(DF, model, batch_size=CFG.BATCH_SIZE):
    uniq = DF.drop_duplicates("id", keep="first").reset_index(drop=True)

    paths = uniq["path"].to_numpy()
    widths = uniq["width"].to_numpy()
    heights = uniq["height"].to_numpy()
    ids = uniq["id"].to_numpy()

    n = len(uniq)
    pred_rle = [None] * (n * 3)
    pred_ids = [None] * (n * 3)
    pred_classes = [None] * (n * 3)

    class_order = ("large_bowel", "small_bowel", "stomach")
    out_i = 0

    H, W, C = CFG.img_size
    dsize = (W, H)

    for start in tqdm(
        range(0, n, batch_size), total=(n + batch_size - 1) // batch_size
    ):
        end = min(start + batch_size, n)
        X = _load_batch_grayscale(paths[start:end], dsize=dsize, out_channels=C)
        preds = model.predict(X, verbose=0)  # (bs, 256,256,3)

        bs = end - start
        for j in range(bs):
            w = int(widths[start + j])
            h = int(heights[start + j])

            p0 = cv2.resize(preds[j, :, :, 0], (w, h), interpolation=cv2.INTER_NEAREST)
            p1 = cv2.resize(preds[j, :, :, 1], (w, h), interpolation=cv2.INTER_NEAREST)
            p2 = cv2.resize(preds[j, :, :, 2], (w, h), interpolation=cv2.INTER_NEAREST)

            b0 = (p0 > 0.5).astype(np.uint8, copy=False)
            b1 = (p1 > 0.5).astype(np.uint8, copy=False)
            b2 = (p2 > 0.5).astype(np.uint8, copy=False)

            _id = ids[start + j]
            pred_ids[out_i] = _id
            pred_classes[out_i] = class_order[0]
            pred_rle[out_i] = rle_encode(b0)
            out_i += 1

            pred_ids[out_i] = _id
            pred_classes[out_i] = class_order[1]
            pred_rle[out_i] = rle_encode(b1)
            out_i += 1

            pred_ids[out_i] = _id
            pred_classes[out_i] = class_order[2]
            pred_rle[out_i] = rle_encode(b2)
            out_i += 1

    return pred_rle, pred_ids, pred_classes




## === cell 17
CFG.BATCH_SIZE = 64

pred_rle, pred_ids, pred_classes = infer(test_df, model, batch_size=CFG.BATCH_SIZE)

print("Pred rows:", len(pred_rle), "Expected:", len(test_df))



## === cell 18
submission = pd.DataFrame(
    {"id": pred_ids, "class": pred_classes, "predicted": pred_rle}
)

sub_df = pd.read_csv(sample_sub).copy()
sub_df = sub_df.drop(columns=["predicted"])
sub_df = sub_df.merge(submission, on=["id", "class"], how="left")
sub_df["predicted"] = sub_df["predicted"].fillna("")
sub_df = sub_df[["id", "class", "predicted"]]

sample_order = pd.read_csv(sample_sub)[["id", "class"]]
sub_df = sample_order.merge(sub_df, on=["id", "class"], how="left")
sub_df["predicted"] = sub_df["predicted"].fillna("")
sub_df = sub_df[["id", "class", "predicted"]]

assert list(sub_df.columns) == ["id", "class", "predicted"]
assert len(sub_df) == len(pd.read_csv(sample_sub))

sub_df.to_csv("submission.csv", index=False)

display(sub_df.head())
print("Saved submission.csv with shape:", sub_df.shape)
print("Null predicted:", sub_df["predicted"].isna().sum())
print("Empty predicted:", (sub_df["predicted"].astype(str).str.len() == 0).sum())
