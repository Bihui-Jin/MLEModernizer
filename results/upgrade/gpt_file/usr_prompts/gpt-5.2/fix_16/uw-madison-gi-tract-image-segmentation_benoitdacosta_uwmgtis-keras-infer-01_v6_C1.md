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
import warnings

warnings.filterwarnings("ignore")

from time import time
import os
from glob import glob
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import dump, load
from tqdm.auto import tqdm

from sklearn.model_selection import GroupKFold

import cv2

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
from tensorflow.keras.callbacks import EarlyStopping

pd.set_option("display.max_columns", 200)
pd.set_option("display.max_colwidth", 200)


def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)




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
    img_size = (
        128,
        128,
        1,
    )  # grayscale channel; model already uses input_shape from this
    n_fold = 5
    fold_selected = 1
    epochs = 100
    seed = 42
    steps_per_epoch_train = None
    steps_per_epoch_val = None


seed_everything(CFG.seed)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(min(8, os.cpu_count() or 2))
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass




## === cell 3
def _build_scans_index(base_dir: str, subset: str):
    safe_base = str(base_dir).replace("/", "_").replace("\\", "_").strip("_")
    cache_path = os.path.join(
        "/kaggle/working", f"scans_index_{subset}_{safe_base}.parquet"
    )
    if os.path.isfile(cache_path):
        return pd.read_parquet(cache_path)

    files = glob(os.path.join(base_dir, "case*/case*_day*/scans/slice_*.png"))
    if len(files) == 0:
        idx = pd.DataFrame(
            columns=[
                "case",
                "day",
                "slice_num",
                "path",
                "width",
                "height",
                "px_spacing_w",
                "px_spacing_h",
            ]
        )
        idx.to_parquet(cache_path, index=False)
        return idx

    rel = pd.Series(files, dtype="string")
    parts = rel.str.replace("\\", "/", regex=False).str.split("/", expand=True)
    case_str = parts.iloc[:, -4]
    day_str = parts.iloc[:, -3]
    fn = parts.iloc[:, -1]

    case = case_str.str.replace("case", "", regex=False).astype("int32")
    day = day_str.str.split("_day", expand=True)[1].astype("int32")

    stem = fn.str.replace(".png", "", regex=False)
    toks = stem.str.split("_", expand=True)
    slice_num = toks[1].astype("int32")
    w = toks[2].astype("int32")
    h = toks[3].astype("int32")
    wsp = toks[4].astype("float32")
    hsp = toks[5].astype("float32")

    idx = pd.DataFrame(
        {
            "case": case.values,
            "day": day.values,
            "slice_num": slice_num.values,
            "path": rel.values,
            "width": w.values,
            "height": h.values,
            "px_spacing_w": wsp.values,
            "px_spacing_h": hsp.values,
        }
    )
    idx.to_parquet(cache_path, index=False)
    return idx


def preprocessing(df, subset="train"):
    cols_sig = "_".join(df.columns.tolist())
    cache_path = os.path.join(
        "/kaggle/working",
        f"preprocessed_{subset}_{len(df)}_{hash(cols_sig) & 0xffffffff}.parquet",
    )
    if os.path.isfile(cache_path):
        return pd.read_parquet(cache_path)

    df = df.copy()

    parts = df["id"].str.split("_", expand=True)
    df["case"] = parts[0].str.replace("case", "", regex=False).astype(np.int32)
    df["day"] = parts[1].str.replace("day", "", regex=False).astype(np.int32)

    slice_str = parts[3]
    df["slice"] = slice_str + ".png"
    df["slice_prefix"] = "slice_" + slice_str
    df["slice_num"] = slice_str.astype(np.int32)

    base_dir = TRAIN_DIR if subset == "train" else TEST_DIR
    idx = _build_scans_index(base_dir, subset=subset)

    df = df.merge(
        idx,
        on=["case", "day", "slice_num"],
        how="left",
        copy=False,
        validate="many_to_one",
    )

    for col, default in [
        ("path", None),
        ("width", 0),
        ("height", 0),
        ("px_spacing_h", np.nan),
        ("px_spacing_w", np.nan),
    ]:
        if col not in df.columns:
            df[col] = default

    df.to_parquet(cache_path, index=False)
    return df




## === cell 4
train_df = preprocessing(df_train, subset="train")
train_df.head()




## === cell 5
def rle_decode(mask_rle, shape):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height, width) of array to return
    """
    if (
        mask_rle is None
        or mask_rle == ""
        or (isinstance(mask_rle, float) and np.isnan(mask_rle))
    ):
        return np.zeros(shape, dtype=np.uint8)
    s = np.asarray(mask_rle.split(), dtype=np.int64)
    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    diff = np.zeros(img.size + 1, dtype=np.int32)
    np.add.at(diff, starts, 1)
    np.add.at(diff, ends, -1)
    img[:] = (np.cumsum(diff[:-1]) > 0).astype(np.uint8)
    return img.reshape(shape)


def rle_encode(img):
    pixels = img.reshape(-1, order="C").astype(np.uint8)
    if pixels.size == 0:
        return ""
    pad = np.empty(pixels.size + 2, dtype=np.uint8)
    pad[0] = 0
    pad[-1] = 0
    pad[1:-1] = pixels
    runs = np.flatnonzero(pad[1:] != pad[:-1]).astype(np.int64) + 1
    if runs.size == 0:
        return ""
    runs[1::2] -= runs[::2]
    return " ".join(runs.astype(str))




## === cell 6
def restructure(df, subset="train"):
    df_out = pd.DataFrame({"id": df["id"][::3].values})

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

    df_out = df_out.reset_index(drop=True).fillna("")
    if subset == "train":
        df_out["count"] = np.sum(df_out.iloc[:, 1:4] != "", axis=1).values
    return df_out


DF_train = restructure(train_df, subset="train")
DF_train.head()




## === cell 7
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        df,
        batch_size=CFG.BATCH_SIZE,
        subset="train",
        shuffle=False,
        img_shape=CFG.img_size,
        precomputed_masks=None,  # np.ndarray [N,H,W,3] for subset=="train"
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.shuffle = shuffle
        self.subset = subset
        self.batch_size = int(batch_size)
        self.img_shape = img_shape
        self.precomputed_masks = precomputed_masks

        self._path = self.df["path"].values
        self._width = self.df["width"].astype(np.int32).values
        self._height = self.df["height"].astype(np.int32).values
        self._id = self.df["id"].values
        if self.subset == "train":
            self._rle_lb = self.df["large_bowel"].values
            self._rle_sb = self.df["small_bowel"].values
            self._rle_st = self.df["stomach"].values
        else:
            self._class = self.df["class"].values

        self.indexes = np.arange(len(self.df), dtype=np.int32)
        self._dsize_hw = (int(self.img_shape[0]), int(self.img_shape[1]))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        batch_indexes = self.indexes[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        bs = int(len(batch_indexes))

        X = np.empty(
            (bs, self.img_shape[0], self.img_shape[1], self.img_shape[2]),
            dtype=np.float32,
        )

        if self.subset == "train":
            if self.precomputed_masks is None:
                y = np.zeros(
                    (bs, self.img_shape[0], self.img_shape[1], 3), dtype=np.float32
                )
            else:
                y = self.precomputed_masks[batch_indexes].astype(np.float32, copy=False)

        if self.subset != "train":
            id_, heights, widths, classes = [], [], [], []

        for i, df_idx in enumerate(batch_indexes):
            img_path = self._path[df_idx]
            w = int(self._width[df_idx])
            h = int(self._height[df_idx])

            X[i] = self._load_cached(img_path, self._dsize_hw)

            if self.subset == "train" and self.precomputed_masks is None:
                rles = (
                    self._rle_lb[df_idx],
                    self._rle_sb[df_idx],
                    self._rle_st[df_idx],
                )
                for k, r in enumerate(rles):
                    m = rle_decode(r, shape=(h, w))
                    m = cv2.resize(m, self._dsize_hw, interpolation=cv2.INTER_NEAREST)
                    y[i, :, :, k] = m
            elif self.subset != "train":
                id_.append(self._id[df_idx])
                heights.append(h)
                widths.append(w)
                classes.append(self._class[df_idx])

        if self.subset == "train":
            return X, y
        else:
            return X, id_, widths, heights, classes

    @staticmethod
    @lru_cache(maxsize=50000)
    def _load_cached(img_path, dsize_hw):
        img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
        img = cv2.resize(img, dsize_hw, interpolation=cv2.INTER_AREA)
        img = img.astype("float32", copy=False)
        mn = float(img.min())
        mx = float(img.max())
        denom = mx - mn
        if denom > 0:
            img = (img - mn) / denom
        else:
            img = img * 0.0
        img = np.expand_dims(img, axis=-1).astype(np.float32, copy=False)
        return img




## === cell 8
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(tf.cast(y_pred, tf.float32))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def iou_coef(y_true, y_pred, smooth=1.0):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


def dice_loss(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(tf.cast(y_pred, tf.float32))
    intersection = y_true_f * y_pred_f
    score = (2.0 * K.sum(intersection) + smooth) / (
        K.sum(y_true_f) + K.sum(y_pred_f) + smooth
    )
    return 1.0 - score


def bce_dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    return binary_crossentropy(y_true, y_pred) + 0.5 * dice_loss(y_true, y_pred)




## === cell 9
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

    s1, p1 = encoder_block(inputs, 64, dropout, batchnorm)
    s2, p2 = encoder_block(p1, 128, dropout, batchnorm)
    s3, p3 = encoder_block(p2, 256, dropout, batchnorm)
    s4, p4 = encoder_block(p3, 512, dropout, batchnorm)

    b1 = conv_block(p4, 1024, batchnorm)

    d1 = decoder_block(b1, s4, 512, dropout, batchnorm)
    d2 = decoder_block(d1, s3, 256, dropout, batchnorm)
    d3 = decoder_block(d2, s2, 128, dropout, batchnorm)
    d4 = decoder_block(d3, s1, 64, dropout, batchnorm)

    outputs = Conv2D(3, 1, padding="same", activation=activation)(d4)

    model = Model(inputs, outputs, name="U-Net")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=bce_dice_loss,
        metrics=[iou_coef, dice_coef],
    )
    return model


def plot_train(model_or_df):
    losses = (
        model_or_df
        if isinstance(model_or_df, pd.DataFrame)
        else pd.DataFrame(model_or_df.history.history)
    )
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(losses["loss"].index, losses["loss"], label="Train_Loss")
    if "val_loss" in losses:
        plt.plot(losses["val_loss"].index, losses["val_loss"], label="Val_loss")
    plt.title("LOSS")
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.legend()

    plt.subplot(1, 3, 2)
    if "dice_coef" in losses:
        plt.plot(
            losses["dice_coef"].index, losses["dice_coef"], label="Train_dice_coef"
        )
    if "val_dice_coef" in losses:
        plt.plot(
            losses["val_dice_coef"].index,
            losses["val_dice_coef"],
            label="Val_dice_coef",
        )
    plt.title("DICE")
    plt.xlabel("Epoch")
    plt.ylabel("dice_coef")
    plt.legend()

    plt.subplot(1, 3, 3)
    if "iou_coef" in losses:
        plt.plot(losses["iou_coef"].index, losses["iou_coef"], label="Train_iou_coef")
    if "val_iou_coef" in losses:
        plt.plot(
            losses["val_iou_coef"].index, losses["val_iou_coef"], label="Val_iou_coef"
        )
    plt.title("IOU")
    plt.xlabel("Epoch")
    plt.ylabel("iou_coef")
    plt.legend()
    plt.show()




## === cell 10
path_load_infer = repertory + "uwmgtis-keras-train-01/"
models_path = path_load_infer
results_path = path_load_infer

os.makedirs(models_path, exist_ok=True)
os.makedirs(results_path, exist_ok=True)


def fit_model(model, model_name, train_sequence, validation_sequence):
    model_file = os.path.join(models_path, f"{model_name}.h5")
    results_file = os.path.join(results_path, f"score_{model_name}.joblib")

    if os.path.isfile(model_file):
        model = load_model(
            model_file,
            custom_objects={
                "bce_dice_loss": bce_dice_loss,
                "iou_coef": iou_coef,
                "dice_coef": dice_coef,
            },
        )
        if os.path.isfile(results_file):
            results = load(results_file)
        else:
            results = pd.DataFrame()
    else:
        early_stop = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )

        model.fit(
            train_sequence,
            epochs=CFG.epochs,
            validation_data=validation_sequence,
            callbacks=[early_stop],
            verbose=2,
        )
        results = pd.DataFrame(model.history.history)
        dump(results, results_file, compress=True)
        model.save(model_file)

    return model, results




## === cell 11
def _precompute_masks(df, img_shape, cache_path=None):
    if cache_path is not None and os.path.isfile(cache_path):
        return np.load(cache_path, allow_pickle=False)["masks"]

    n = len(df)
    H, W = img_shape[0], img_shape[1]
    width = df["width"].astype(np.int32).values
    height = df["height"].astype(np.int32).values
    rle_lb = df["large_bowel"].values
    rle_sb = df["small_bowel"].values
    rle_st = df["stomach"].values

    masks = np.empty((n, H, W, 3), dtype=np.uint8)
    for i in tqdm(range(n), desc="Precomputing masks", leave=False):
        w = int(width[i])
        h = int(height[i])

        m0 = rle_decode(rle_lb[i], shape=(h, w))
        m0 = cv2.resize(m0, (W, H), interpolation=cv2.INTER_NEAREST)

        m1 = rle_decode(rle_sb[i], shape=(h, w))
        m1 = cv2.resize(m1, (W, H), interpolation=cv2.INTER_NEAREST)

        m2 = rle_decode(rle_st[i], shape=(h, w))
        m2 = cv2.resize(m2, (W, H), interpolation=cv2.INTER_NEAREST)

        masks[i, :, :, 0] = m0
        masks[i, :, :, 1] = m1
        masks[i, :, :, 2] = m2

    if cache_path is not None:
        np.savez_compressed(cache_path, masks=masks)
    return masks


def _make_tf_dataset(df, masks_uint8, batch_size, shuffle, img_shape):
    paths = df["path"].astype(str).values
    H, W = int(img_shape[0]), int(img_shape[1])

    masks_f = masks_uint8.astype(np.float32, copy=False)

    ds = tf.data.Dataset.from_tensor_slices((paths, masks_f))
    if shuffle:
        buf = int(min(len(paths), 8192))
        ds = ds.shuffle(buffer_size=buf, seed=CFG.seed, reshuffle_each_iteration=True)

    def _load_tf(path, y):
        img_bytes = tf.io.read_file(path)
        x = tf.io.decode_png(img_bytes, channels=1, dtype=tf.uint16)  # [H0,W0,1]
        x = tf.image.resize(x, [H, W], method=tf.image.ResizeMethod.AREA)
        x = tf.cast(x, tf.float32)
        mn = tf.reduce_min(x)
        mx = tf.reduce_max(x)
        denom = mx - mn
        x = tf.cond(denom > 0.0, lambda: (x - mn) / denom, lambda: tf.zeros_like(x))
        x = tf.ensure_shape(x, [H, W, 1])
        y = tf.ensure_shape(tf.cast(y, tf.float32), [H, W, 3])
        return x, y

    autotune = tf.data.AUTOTUNE
    ds = ds.map(_load_tf, num_parallel_calls=autotune, deterministic=True)
    ds = ds.cache()  # caches decoded+resized tensors (RAM) for epoch-to-epoch speed
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(autotune)
    return ds


split_cache = os.path.join(results_path, f"split_idx_fold{CFG.fold_selected}.npz")
if os.path.isfile(split_cache):
    tmp = np.load(split_cache)
    train_idx = tmp["train_idx"]
    val_idx = tmp["val_idx"]
else:
    gkf = GroupKFold(n_splits=CFG.n_fold)
    groups = DF_train["case"].values
    for f, (tr_i, va_i) in enumerate(
        gkf.split(DF_train, DF_train["count"].values, groups=groups)
    ):
        if f == CFG.fold_selected:
            train_idx, val_idx = tr_i, va_i
            break
    np.savez_compressed(split_cache, train_idx=train_idx, val_idx=val_idx)

DF_tr = DF_train.iloc[train_idx].reset_index(drop=True)
DF_va = DF_train.iloc[val_idx].reset_index(drop=True)

tr_masks = _precompute_masks(
    DF_tr, CFG.img_size, cache_path=os.path.join(results_path, "tr_masks_128.npz")
)
va_masks = _precompute_masks(
    DF_va, CFG.img_size, cache_path=os.path.join(results_path, "va_masks_128.npz")
)

train_gen = _make_tf_dataset(
    DF_tr, tr_masks, batch_size=CFG.BATCH_SIZE, shuffle=True, img_shape=CFG.img_size
)
val_gen = _make_tf_dataset(
    DF_va, va_masks, batch_size=CFG.BATCH_SIZE, shuffle=False, img_shape=CFG.img_size
)

input_shape = CFG.img_size
model = build_unet(input_shape, dropout=True, batchnorm=True)

model, results = fit_model(model, "U-net", train_gen, val_gen)
if isinstance(results, pd.DataFrame) and len(results):
    plot_train(results)




## === cell 12
sub_df = pd.read_csv(sample_sub)
test_base = pd.read_csv(test_csv)

test_df_raw = preprocessing(test_base, subset="test")
test_df_img = restructure(test_df_raw, subset="test")
test_df = test_base.merge(
    test_df_img[["id", "path", "case", "day", "slice", "width", "height"]],
    on="id",
    how="left",
)
test_df.head(5)




## === cell 13
def infer(DF, model, batch_size=CFG.BATCH_SIZE):
    class_to_k = {"large_bowel": 0, "small_bowel": 1, "stomach": 2}
    H_in, W_in = int(CFG.img_size[0]), int(CFG.img_size[1])

    g = DF.groupby("id", sort=False, observed=True)

    uniq_ids = g.size().index.astype("string").to_numpy()
    first = g.nth(0).reset_index()  # one row per id (keeps original order)
    paths = first["path"].astype(str).to_numpy()
    widths = first["width"].astype(np.int32).to_numpy()
    heights = first["height"].astype(np.int32).to_numpy()

    ds = tf.data.Dataset.from_tensor_slices(paths)

    def _load_tf(path):
        img_bytes = tf.io.read_file(path)
        x = tf.io.decode_png(img_bytes, channels=1, dtype=tf.uint16)
        x = tf.image.resize(x, [H_in, W_in], method=tf.image.ResizeMethod.AREA)
        x = tf.cast(x, tf.float32)
        mn = tf.reduce_min(x)
        mx = tf.reduce_max(x)
        denom = mx - mn
        x = tf.cond(denom > 0.0, lambda: (x - mn) / denom, lambda: tf.zeros_like(x))
        x = tf.ensure_shape(x, [H_in, W_in, 1])
        return x

    autotune = tf.data.AUTOTUNE
    ds = ds.map(_load_tf, num_parallel_calls=autotune, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(autotune)

    preds_all = np.empty((len(uniq_ids), H_in, W_in, 3), dtype=np.float32)
    seen = 0
    for X in tqdm(ds, total=int(np.ceil(len(uniq_ids) / batch_size)), leave=False):
        bs = int(X.shape[0])
        preds_all[seen : seen + bs] = model.predict_on_batch(X)
        seen += bs

    pred_rle = []
    pred_ids = []
    pred_classes = []

    ids_full = DF["id"].astype("string").to_numpy()
    classes_full = DF["class"].astype("string").to_numpy()

    id_to_i = {str(_id): i for i, _id in enumerate(uniq_ids)}

    for _id, cls in tqdm(zip(ids_full, classes_full), total=len(DF), leave=False):
        i = id_to_i[str(_id)]
        w = int(widths[i])
        h = int(heights[i])
        k = class_to_k[str(cls)]

        pred_img = preds_all[i, :, :, k]
        if (h, w) != (H_in, W_in):
            pred_img = cv2.resize(pred_img, (w, h), interpolation=cv2.INTER_NEAREST)
        pred_bin = (pred_img > 0.5).astype(np.uint8)

        pred_ids.append(str(_id))
        pred_classes.append(str(cls))
        pred_rle.append(rle_encode(pred_bin))

    return pred_rle, pred_ids, pred_classes


CFG.BATCH_SIZE = 32
pred_rle, pred_ids, pred_classes = infer(test_df, model, batch_size=CFG.BATCH_SIZE)




## === cell 14
submission = pd.DataFrame(
    {"id": pred_ids, "class": pred_classes, "predicted": pred_rle}
)

sub_df = pd.read_csv(sample_sub)
sub_df = sub_df.drop(columns=["predicted"])
sub_df = sub_df.merge(submission, on=["id", "class"], how="left")
sub_df["predicted"] = sub_df["predicted"].fillna("")

sub_df.to_csv("submission.csv", index=False)

sub_df.head()
