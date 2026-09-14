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
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import warnings

warnings.filterwarnings("ignore")

import google.protobuf  # noqa: F401

import gc
from glob import glob

import numpy as np
import pandas as pd
import cv2

from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

tf.random.set_seed(0)
np.random.seed(0)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(min(8, os.cpu_count() or 8))
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

print("TF:", tf.__version__)



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
    df = df.drop(columns=["segmentation"])
    df["predicted"] = ""

df.head()



## === cell 3
df = df.rename(columns={"class": "class_name"})

parts = df["id"].str.split("_", expand=True)
df["case"] = parts[0].str.replace("case", "", regex=False).astype(np.int32)
df["day"] = parts[1].str.replace("day", "", regex=False).astype(np.int32)
df["slice"] = parts[3].astype(str)

if DEBUG:
    IMG_DIR = "../input/uw-madison-gi-tract-image-segmentation/train"
else:
    IMG_DIR = "../input/uw-madison-gi-tract-image-segmentation/test"

all_images = glob(os.path.join(IMG_DIR, "**", "*.png"), recursive=True)
if len(all_images) == 0:
    raise FileNotFoundError(f"No .png images found under {IMG_DIR}")

base_dir = all_images[0].rsplit("/", 4)[0]

df["path_partial"] = (
    base_dir
    + "/case"
    + df["case"].astype(str)
    + "/case"
    + df["case"].astype(str)
    + "_day"
    + df["day"].astype(str)
    + "/scans/slice_"
    + df["slice"].astype(str)
)

img_partials = [str(p.rsplit("_", 4)[0]) for p in all_images]
partial_to_path = dict(zip(img_partials, all_images))
df["path"] = df["path_partial"].map(partial_to_path)
df = df.drop(columns=["path_partial"])

missing = df["path"].isna().sum()
if missing:
    raise FileNotFoundError(
        f"{missing} rows could not be matched to an image path. Check path parsing."
    )

stem = df["path"].str.replace(".png", "", regex=False)
tail = stem.str.rsplit("_", n=4, expand=True)
df["width"] = tail[1].astype(np.int32)
df["height"] = tail[2].astype(np.int32)

df.head()



## === cell 4
df_img = pd.DataFrame({"id": df["id"][::3].values})
df_img["path"] = df["path"][::3].values
df_img["predicted"] = df["predicted"][::3].values if "predicted" in df.columns else ""
df_img["case"] = df["case"][::3].values
df_img["day"] = df["day"][::3].values
df_img["slice"] = df["slice"][::3].values
df_img["width"] = df["width"][::3].values
df_img["height"] = df["height"][::3].values

del df
df_img = df_img.reset_index(drop=True)
df_img = df_img.fillna("")
df_img.head()



## === cell 5
print(df_img.shape)
if DEBUG:
    df_img = df_img.sample(frac=0.05, random_state=0).reset_index(drop=True)
print(df_img.shape)
gc.collect()




## === cell 6
def rle_encode(img: np.ndarray) -> str:
    img = img.astype(np.uint8, copy=False)
    pixels = img.ravel(order="F")
    if pixels.size == 0:
        return ""
    p = np.empty(pixels.size + 2, dtype=np.uint8)
    p[0] = 0
    p[1:-1] = pixels
    p[-1] = 0
    changes = np.flatnonzero(p[1:] != p[:-1]) + 1
    if changes.size == 0:
        return ""
    runs = changes.astype(np.int64, copy=False)
    runs[1::2] -= runs[::2]
    return " ".join(runs.astype(str).tolist())


def rle_decode(mask_rle: str, shape, color=1) -> np.ndarray:
    if isinstance(shape, (tuple, list)) and len(shape) == 2:
        h, w = int(shape[0]), int(shape[1])
        c = 1
        out_shape = (h, w)
        want_3d = False
    else:
        h, w, c = int(shape[0]), int(shape[1]), int(shape[2])
        out_shape = (h, w, c)
        want_3d = True

    if (
        mask_rle is None
        or mask_rle == ""
        or (isinstance(mask_rle, float) and np.isnan(mask_rle))
    ):
        return np.zeros(out_shape, dtype=np.float32)

    s = mask_rle.split()
    starts = np.asarray(s[0::2], dtype=np.int64) - 1
    lengths = np.asarray(s[1::2], dtype=np.int64)
    ends = starts + lengths

    n = h * w
    diff = np.zeros(n + 1, dtype=np.int32)
    np.add.at(diff, starts, 1)
    np.add.at(diff, ends, -1)
    flat = (np.cumsum(diff[:-1]) > 0).astype(np.float32) * float(color)

    if c == 1:
        img = flat.reshape((h, w), order="F")
        if want_3d:
            return img[:, :, None]
        return img
    img = np.repeat(flat[:, None], c, axis=1).reshape((h, w, c), order="F")
    return img




## === cell 7
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size=BATCH_SIZE, subset="train", shuffle=False):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.shuffle = shuffle
        self.subset = subset
        self.batch_size = int(batch_size)

        self.paths = self.df["path"].values
        self.widths = self.df["width"].values.astype(np.int32)
        self.heights = self.df["height"].values.astype(np.int32)

        if self.subset == "train":
            self.rle_lb = (
                self.df["large_bowel"].values
                if "large_bowel" in self.df.columns
                else np.array([""] * len(self.df), dtype=object)
            )
            self.rle_sb = (
                self.df["small_bowel"].values
                if "small_bowel" in self.df.columns
                else np.array([""] * len(self.df), dtype=object)
            )
            self.rle_st = (
                self.df["stomach"].values
                if "stomach" in self.df.columns
                else np.array([""] * len(self.df), dtype=object)
            )

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

        X = np.empty((cur_bs, 128, 128, 1), dtype=np.float32)

        if self.subset == "train":
            y = np.empty((cur_bs, 128, 128, 3), dtype=np.float32)

        for i, row_idx in enumerate(batch_indexes):
            img_path = self.paths[row_idx]
            w = int(self.widths[row_idx])
            h = int(self.heights[row_idx])

            img = self.__load_grayscale(img_path)
            X[i] = img

            if self.subset == "train":
                rles = (
                    self.rle_lb[row_idx],
                    self.rle_sb[row_idx],
                    self.rle_st[row_idx],
                )
                for k, rle in enumerate(rles):
                    masks = rle_decode(rle, shape=(h, w, 1))  # (h,w,1)
                    masks2d = masks[:, :, 0]
                    masks2d = cv2.resize(
                        masks2d, (128, 128), interpolation=cv2.INTER_NEAREST
                    )
                    y[i, :, :, k] = masks2d

        if self.subset == "train":
            return X, y
        return X

    def __load_grayscale(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)
        return img




## === cell 8
from tensorflow.keras.losses import binary_crossentropy


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(tf.cast(y_pred, tf.float32))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def iou_coef(y_true, y_pred, smooth=1):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
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
custom_objects = {
    "dice_coef": dice_coef,
    "iou_coef": iou_coef,
    "bce_dice_loss": bce_dice_loss,
}

MODEL_PATH = "../input/uwmgi-unet-keras/model.h5"


def build_simple_unet(input_shape=(128, 128, 1), n_classes=3):
    inputs = keras.Input(shape=input_shape)

    c1 = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    c1 = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(c1)
    p1 = keras.layers.MaxPooling2D()(c1)

    c2 = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(p1)
    c2 = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(c2)
    p2 = keras.layers.MaxPooling2D()(c2)

    c3 = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(p2)
    c3 = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(c3)

    u2 = keras.layers.UpSampling2D()(c3)
    u2 = keras.layers.Concatenate()([u2, c2])
    c4 = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(u2)
    c4 = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(c4)

    u1 = keras.layers.UpSampling2D()(c4)
    u1 = keras.layers.Concatenate()([u1, c1])
    c5 = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(u1)
    c5 = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(c5)

    outputs = keras.layers.Conv2D(n_classes, 1, padding="same", activation="sigmoid")(
        c5
    )
    model = keras.Model(inputs, outputs)
    return model


if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(
        MODEL_PATH, custom_objects=custom_objects, compile=False
    )
    print("Loaded model:", MODEL_PATH)
else:
    model = build_simple_unet()
    print(
        f"WARNING: model file not found at {MODEL_PATH}. Will train the fallback model on train.csv."
    )

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss=bce_dice_loss)

gc.collect()



## === cell 10
if not os.path.exists(MODEL_PATH):
    train_df = pd.read_csv("../input/uw-madison-gi-tract-image-segmentation/train.csv")
    train_df = train_df.rename(columns={"class": "class_name"})

    train_piv = (
        train_df.pivot_table(
            index="id", columns="class_name", values="segmentation", aggfunc="first"
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for col in ["large_bowel", "small_bowel", "stomach"]:
        if col not in train_piv.columns:
            train_piv[col] = ""

    parts = train_piv["id"].str.split("_", expand=True)
    train_piv["case"] = parts[0].str.replace("case", "", regex=False).astype(np.int32)
    train_piv["day"] = parts[1].str.replace("day", "", regex=False).astype(np.int32)
    train_piv["slice"] = parts[3].astype(str)

    TRAIN_DIR = "../input/uw-madison-gi-tract-image-segmentation/train"
    train_images = glob(os.path.join(TRAIN_DIR, "**", "*.png"), recursive=True)
    if len(train_images) == 0:
        raise FileNotFoundError(f"No .png images found under {TRAIN_DIR}")

    train_base_dir = train_images[0].rsplit("/", 4)[0]

    train_piv["path_partial"] = (
        train_base_dir
        + "/case"
        + train_piv["case"].astype(str)
        + "/case"
        + train_piv["case"].astype(str)
        + "_day"
        + train_piv["day"].astype(str)
        + "/scans/slice_"
        + train_piv["slice"].astype(str)
    )

    train_img_partials = [str(p.rsplit("_", 4)[0]) for p in train_images]
    train_partial_to_path = dict(zip(train_img_partials, train_images))
    train_piv["path"] = train_piv["path_partial"].map(train_partial_to_path)
    train_piv = train_piv.drop(columns=["path_partial"]).reset_index(drop=True)

    miss = train_piv["path"].isna().sum()
    if miss:
        train_piv = train_piv.dropna(subset=["path"]).reset_index(drop=True)

    stem = train_piv["path"].str.replace(".png", "", regex=False)
    tail = stem.str.rsplit("_", n=4, expand=True)
    train_piv["width"] = tail[1].astype(np.int32)
    train_piv["height"] = tail[2].astype(np.int32)

    cases = np.array(sorted(train_piv["case"].unique()))
    rng = np.random.default_rng(0)
    rng.shuffle(cases)
    n_val_cases = max(1, int(0.1 * len(cases)))
    val_cases = set(cases[:n_val_cases])

    trn_df = train_piv[~train_piv["case"].isin(val_cases)].reset_index(drop=True)
    val_df = train_piv[train_piv["case"].isin(val_cases)].reset_index(drop=True)

    max_train = 2500
    max_val = 400
    if len(trn_df) > max_train:
        trn_df = trn_df.sample(n=max_train, random_state=0).reset_index(drop=True)
    if len(val_df) > max_val:
        val_df = val_df.sample(n=max_val, random_state=0).reset_index(drop=True)

    class CachedTrainDataGenerator(tf.keras.utils.Sequence):
        def __init__(self, df, batch_size=BATCH_SIZE, shuffle=False):
            super().__init__()
            self.df = df.reset_index(drop=True)
            self.batch_size = int(batch_size)
            self.shuffle = bool(shuffle)

            self.paths = self.df["path"].values
            self.widths = self.df["width"].values.astype(np.int32)
            self.heights = self.df["height"].values.astype(np.int32)
            self.rle_lb = self.df["large_bowel"].values
            self.rle_sb = self.df["small_bowel"].values
            self.rle_st = self.df["stomach"].values

            n = len(self.df)
            self.X_cache = np.empty((n, 128, 128, 1), dtype=np.float32)
            self.y_cache = np.empty((n, 128, 128, 3), dtype=np.float32)

            for i in tqdm(range(n), desc="Caching train data", total=n):
                img = cv2.imread(self.paths[i], cv2.IMREAD_ANYDEPTH)
                if img is None:
                    raise FileNotFoundError(f"Failed to read image: {self.paths[i]}")
                img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
                img = img.astype(np.float32) / 255.0
                self.X_cache[i, :, :, 0] = img

                h = int(self.heights[i])
                w = int(self.widths[i])
                rles = (self.rle_lb[i], self.rle_sb[i], self.rle_st[i])
                for k, rle in enumerate(rles):
                    m = rle_decode(rle, shape=(h, w, 1))[:, :, 0]
                    m = cv2.resize(m, (128, 128), interpolation=cv2.INTER_NEAREST)
                    self.y_cache[i, :, :, k] = m

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
            return self.X_cache[batch_indexes], self.y_cache[batch_indexes]

    train_gen = CachedTrainDataGenerator(trn_df, batch_size=BATCH_SIZE, shuffle=True)
    val_gen = (
        CachedTrainDataGenerator(val_df, batch_size=BATCH_SIZE, shuffle=False)
        if len(val_df)
        else None
    )

    history = model.fit(
        train_gen,
        validation_data=val_gen if val_gen is not None else None,
        epochs=EPOCHS,
        verbose=2,
    )
    del (
        train_df,
        train_piv,
        train_images,
        trn_df,
        val_df,
        train_gen,
        val_gen,
        history,
    )
    gc.collect()



## === cell 11
paths = df_img["path"].to_numpy(dtype=object, copy=False)


def _load_img_tf(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_png(img_bytes, channels=1, dtype=tf.uint16)
    img = tf.image.resize(img, [128, 128], method=tf.image.ResizeMethod.AREA)
    img = tf.cast(img, tf.float32) / 255.0
    return img  # (128,128,1)


AUTOTUNE = tf.data.AUTOTUNE
PRED_BATCH = 32

pred_ds = (
    tf.data.Dataset.from_tensor_slices(paths)
    .map(_load_img_tf, num_parallel_calls=AUTOTUNE, deterministic=True)
    .batch(PRED_BATCH, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

gc.collect()

thr = 0.5
n = len(df_img)
widths_all = df_img["width"].to_numpy(dtype=np.int32, copy=False)
heights_all = df_img["height"].to_numpy(dtype=np.int32, copy=False)

lbs = [""] * n
sbs = [""] * n
sts = [""] * n

_resize_buf_cache = {}  # (h,w) -> (buf0, buf1, buf2) float32 arrays

start = 0
for Xb in tqdm(pred_ds, total=int(np.ceil(n / PRED_BATCH))):
    probs_b = model.predict_on_batch(Xb)  # (bs,128,128,3)
    probs_b = np.asarray(probs_b)
    bs = probs_b.shape[0]
    end = min(start + bs, n)

    h_batch = heights_all[start:end]
    w_batch = widths_all[start:end]

    for j in range(end - start):
        h = int(h_batch[j])
        w = int(w_batch[j])

        key = (h, w)
        bufs = _resize_buf_cache.get(key)
        if bufs is None:
            bufs = (
                np.empty((h, w), dtype=np.float32),
                np.empty((h, w), dtype=np.float32),
                np.empty((h, w), dtype=np.float32),
            )
            _resize_buf_cache[key] = bufs

        p0 = cv2.resize(
            probs_b[j, :, :, 0], (w, h), interpolation=cv2.INTER_NEAREST, dst=bufs[0]
        )
        p1 = cv2.resize(
            probs_b[j, :, :, 1], (w, h), interpolation=cv2.INTER_NEAREST, dst=bufs[1]
        )
        p2 = cv2.resize(
            probs_b[j, :, :, 2], (w, h), interpolation=cv2.INTER_NEAREST, dst=bufs[2]
        )

        m0 = (p0 >= thr).astype(np.uint8, copy=False)
        m1 = (p1 >= thr).astype(np.uint8, copy=False)
        m2 = (p2 >= thr).astype(np.uint8, copy=False)

        idx = start + j
        lbs[idx] = rle_encode(m0)
        sbs[idx] = rle_encode(m1)
        sts[idx] = rle_encode(m2)

    start += bs
    del Xb, probs_b
    if (start // PRED_BATCH) % 80 == 0:
        gc.collect()

gc.collect()
print(len(lbs), len(sbs), len(sts))



## === cell 12
sample_sub = pd.read_csv(
    "../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv"
)

id_to_idx = pd.Series(np.arange(len(df_img), dtype=np.int32), index=df_img["id"].values)
idx = sample_sub["id"].map(id_to_idx).astype("Int32")  # pandas nullable int

empty_rle = ""
pred_arr = np.full(len(sample_sub), empty_rle, dtype=object)

valid_mask = idx.notna().values
valid_pos = np.flatnonzero(valid_mask)
idx_valid = idx[valid_mask].astype(np.int32).values
cls_valid = sample_sub.loc[valid_mask, "class"].values

lbs_a = np.asarray(lbs, dtype=object)
sbs_a = np.asarray(sbs, dtype=object)
sts_a = np.asarray(sts, dtype=object)

m0 = cls_valid == "large_bowel"
m1 = cls_valid == "small_bowel"
m2 = ~(m0 | m1)  # stomach

pred_arr[valid_pos[m0]] = lbs_a[idx_valid[m0]]
pred_arr[valid_pos[m1]] = sbs_a[idx_valid[m1]]
pred_arr[valid_pos[m2]] = sts_a[idx_valid[m2]]

sample_sub["predicted"] = pred_arr
sample_sub.to_csv("submission.csv", index=False)
print(sample_sub.head())
print("Wrote submission.csv with shape:", sample_sub.shape)



## === cell 13
sub = pd.read_csv("submission.csv")
assert list(sub.columns) == ["id", "class", "predicted"]
assert sub.shape[0] == 20400
print(sub["predicted"].isna().sum(), "NaN predicted values")
print("Done.")
