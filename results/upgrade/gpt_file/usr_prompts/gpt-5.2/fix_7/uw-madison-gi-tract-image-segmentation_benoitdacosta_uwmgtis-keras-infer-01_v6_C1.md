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

from joblib import dump, load, Parallel, delayed
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




## === cell 3
@lru_cache(maxsize=None)
def _scans_dir(subset: str, case: int, day: int) -> str:
    base_dir = TRAIN_DIR if subset == "train" else TEST_DIR
    return os.path.join(base_dir, f"case{case}", f"case{case}_day{day}", "scans")


def _build_scan_file_table(subset: str) -> pd.DataFrame:
    base_dir = TRAIN_DIR if subset == "train" else TEST_DIR
    rows = []

    try:
        with os.scandir(base_dir) as it_case:
            for case_entry in it_case:
                if not case_entry.is_dir():
                    continue
                name_case = case_entry.name
                if not name_case.startswith("case"):
                    continue
                try:
                    case = int(name_case[4:])
                except Exception:
                    continue

                case_dir = case_entry.path
                with os.scandir(case_dir) as it_day:
                    for day_entry in it_day:
                        if not day_entry.is_dir():
                            continue
                        name_day = day_entry.name  # caseXXX_dayYY
                        if "_day" not in name_day:
                            continue
                        try:
                            day = int(name_day.split("_day", 1)[1])
                        except Exception:
                            continue

                        scans_dir = os.path.join(day_entry.path, "scans")
                        if not os.path.isdir(scans_dir):
                            continue

                        with os.scandir(scans_dir) as it_scan:
                            for f in it_scan:
                                if not f.is_file():
                                    continue
                                fn = f.name
                                if not (
                                    fn.startswith("slice_") and fn.endswith(".png")
                                ):
                                    continue
                                parts = fn.split("_", 2)
                                if len(parts) < 2:
                                    continue
                                slice_prefix = parts[0] + "_" + parts[1]
                                rows.append((case, day, slice_prefix, f.path))
    except FileNotFoundError:
        pass

    return pd.DataFrame(rows, columns=["case", "day", "slice_prefix", "path"])


@lru_cache(maxsize=4)
def _scan_file_table_cached(subset: str) -> pd.DataFrame:
    cache_dir = "/kaggle/working"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"scan_file_table_{subset}.parquet")
    if os.path.isfile(cache_path):
        return pd.read_parquet(cache_path)
    tbl = _build_scan_file_table(subset)
    try:
        tbl.to_parquet(cache_path, index=False)
    except Exception:
        tbl.to_csv(cache_path.replace(".parquet", ".csv"), index=False)
    return tbl


def preprocessing(df, subset="train"):
    df = df.copy()

    parts = df["id"].str.split("_", expand=True)
    df["case"] = parts[0].str.replace("case", "", regex=False).astype(int)
    df["day"] = parts[1].str.replace("day", "", regex=False).astype(int)
    df["slice"] = parts[3]  # includes ".png" suffix

    df["slice_prefix"] = "slice_" + df["slice"].str.replace(".png", "", regex=False)

    scan_tbl = _scan_file_table_cached(subset)
    df = df.merge(scan_tbl, on=["case", "day", "slice_prefix"], how="left")

    stem = df["path"].astype(str).str[:-4]
    tokens = stem.str.rsplit("_", n=4, expand=True)
    df["width"] = tokens[1].astype(int)
    df["height"] = tokens[2].astype(int)
    df["px_spacing_h"] = tokens[3].astype(float)
    df["px_spacing_w"] = tokens[4].astype(float)
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
    runs = np.flatnonzero(pad[1:] != pad[:-1]) + 1
    runs[1::2] -= runs[::2]
    return " ".join(map(str, runs))




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
        self.batch_size = batch_size
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

        self.indexes = np.arange(len(self.df))
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
        bs = len(batch_indexes)

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

        id_, heights, widths, classes = [], [], [], []

        for i, df_idx in enumerate(batch_indexes):
            img_path = self._path[df_idx]
            w = int(self._width[df_idx])
            h = int(self._height[df_idx])

            img = self.__load_grayscale(img_path)
            X[i] = img

            if self.subset == "train" and self.precomputed_masks is None:
                rles = (
                    self._rle_lb[df_idx],
                    self._rle_sb[df_idx],
                    self._rle_st[df_idx],
                )
                for k, r in enumerate(rles):
                    m = rle_decode(r, shape=(h, w)).astype(np.uint8)
                    m = cv2.resize(
                        m, self.img_shape[0:2], interpolation=cv2.INTER_NEAREST
                    )
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

    def __load_grayscale(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
        dsize = self.img_shape[0:2]
        img = cv2.resize(img, dsize, interpolation=cv2.INTER_AREA)
        img = img.astype("float32")
        mn = img.min()
        mx = img.max()
        denom = mx - mn
        if denom > 0:
            img = (img - mn) / denom
        else:
            img = img * 0.0
        img = np.expand_dims(img, axis=-1).astype(np.float32)
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


def fit_model(model, model_name, train_dataset, validation_dataset):
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
            train_dataset,
            epochs=CFG.epochs,
            validation_data=validation_dataset,
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

    def _one(i):
        w = int(width[i])
        h = int(height[i])

        m0 = rle_decode(rle_lb[i], shape=(h, w))
        m0 = cv2.resize(m0, (W, H), interpolation=cv2.INTER_NEAREST)

        m1 = rle_decode(rle_sb[i], shape=(h, w))
        m1 = cv2.resize(m1, (W, H), interpolation=cv2.INTER_NEAREST)

        m2 = rle_decode(rle_st[i], shape=(h, w))
        m2 = cv2.resize(m2, (W, H), interpolation=cv2.INTER_NEAREST)

        out = np.stack([m0, m1, m2], axis=-1).astype(np.uint8, copy=False)
        return out

    n_jobs = min(8, (os.cpu_count() or 2))
    masks_list = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_one)(i) for i in tqdm(range(n), desc="Precomputing masks", leave=False)
    )
    masks = np.stack(masks_list, axis=0)

    if cache_path is not None:
        np.savez_compressed(cache_path, masks=masks)
    return masks


gkf = GroupKFold(n_splits=5)
groups = DF_train["case"].values
train_idx, val_idx = next(gkf.split(DF_train, DF_train["count"].values, groups=groups))

DF_tr = DF_train.iloc[train_idx].reset_index(drop=True)
DF_va = DF_train.iloc[val_idx].reset_index(drop=True)

tr_masks = _precompute_masks(
    DF_tr, CFG.img_size, cache_path=os.path.join(results_path, "tr_masks_128.npz")
)
va_masks = _precompute_masks(
    DF_va, CFG.img_size, cache_path=os.path.join(results_path, "va_masks_128.npz")
)

train_gen = DataGenerator(
    DF_tr,
    batch_size=CFG.BATCH_SIZE,
    subset="train",
    shuffle=True,
    img_shape=CFG.img_size,
    precomputed_masks=tr_masks,
)
val_gen = DataGenerator(
    DF_va,
    batch_size=CFG.BATCH_SIZE,
    subset="train",
    shuffle=False,
    img_shape=CFG.img_size,
    precomputed_masks=va_masks,
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
    pred_rle = []
    pred_ids = []
    pred_classes = []

    DF_batch = DataGenerator(
        DF, batch_size=batch_size, subset="test", shuffle=False, img_shape=CFG.img_size
    )
    class_to_k = {"large_bowel": 0, "small_bowel": 1, "stomach": 2}

    resize_map_cache = {}
    H_in, W_in = CFG.img_size[0], CFG.img_size[1]

    for b in tqdm(range(len(DF_batch)), total=len(DF_batch), leave=False):
        img, id_, widths, heights, classes = DF_batch[b]
        preds = model.predict_on_batch(img)  # [bs, H, W, 3]

        ks = np.fromiter(
            (class_to_k[c] for c in classes), dtype=np.int64, count=len(classes)
        )
        pred_chan = preds[np.arange(len(classes)), :, :, ks]  # [bs, H, W]

        for j in range(len(id_)):
            w = int(widths[j])
            h = int(heights[j])
            key = (w, h)
            if key not in resize_map_cache:
                map1, map2 = cv2.initInterpolation(
                    (W_in, H_in), (w, h), cv2.INTER_NEAREST
                )
                resize_map_cache[key] = (map1, map2)
            map1, map2 = resize_map_cache[key]

            pred_img = cv2.remap(
                pred_chan[j], map1, map2, interpolation=cv2.INTER_NEAREST
            )
            pred_img = (pred_img > 0.5).astype(np.uint8)

            pred_ids.append(id_[j])
            pred_classes.append(classes[j])
            pred_rle.append(rle_encode(pred_img))

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
