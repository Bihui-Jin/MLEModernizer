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
import os, random, gc, warnings

warnings.filterwarnings("ignore")

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

random.seed(42)
import numpy as np

np.random.seed(42)



## === cell 1
import pandas as pd
import cv2
from tqdm import tqdm
from functools import lru_cache


def _safe_import_tf():
    """
    Fix for Kaggle images where TF + protobuf can crash with:
    AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
    We force python protobuf via env vars above and also clear protobuf modules on retry.
    """
    try:
        import tensorflow as tf  # noqa: F401

        return tf
    except Exception:
        import sys

        for k in list(sys.modules.keys()):
            if k.startswith("google.protobuf"):
                sys.modules.pop(k, None)
        import tensorflow as tf  # noqa: F401

        return tf


tf = _safe_import_tf()
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

try:
    cv2.setNumThreads(0)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass



## === cell 2
BATCH_SIZE = 16
EPOCHS = 30
n_splits = 5
fold_selected = 1  # 1..5



## === cell 3
df = pd.read_csv(
    "../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv"
)
DEBUG = False
if df.shape[0] == 0:
    DEBUG = True
if DEBUG == True:
    df = pd.read_csv("../input/uw-madison-gi-tract-image-segmentation/train.csv")
    df.pop("segmentation")
    df["predicted"] = ""



## === cell 4
df.rename(columns={"class": "class_name"}, inplace=True)

parts = df["id"].str.split("_", expand=True)
df["case"] = parts[0].str.replace("case", "", regex=False).astype(np.int32)
df["day"] = parts[1].str.replace("day", "", regex=False).astype(np.int32)
df["slice"] = parts[3]  # like '0000.png'

if DEBUG:
    TRAIN_DIR = "../input/uw-madison-gi-tract-image-segmentation/train"
else:
    TRAIN_DIR = "../input/uw-madison-gi-tract-image-segmentation/test"


@lru_cache(maxsize=None)
def _build_slice_map(case_int: int, day_int: int):
    scans_dir = os.path.join(
        TRAIN_DIR, f"case{case_int}", f"case{case_int}_day{day_int}", "scans"
    )
    try:
        files = os.listdir(scans_dir)
    except FileNotFoundError:
        return {}, scans_dir

    m = {}
    for fn in files:
        if not fn.startswith("slice_") or not fn.endswith(".png"):
            continue
        parts = fn[:-4].split("_")
        if len(parts) < 4:
            continue
        idx = parts[1]
        m[idx] = fn
    return m, scans_dir


def _get_path_w_h(case_int: int, day_int: int, slice_png: str):
    slice_idx = slice_png[:-4]
    m, scans_dir = _build_slice_map(int(case_int), int(day_int))
    fn = m.get(slice_idx, "")
    if fn:
        parts = fn[:-4].split("_")
        w = int(parts[2])
        h = int(parts[3])
        return os.path.join(scans_dir, fn), w, h
    return "", 0, 0


paths_w_h = [
    _get_path_w_h(c, d, s)
    for c, d, s in zip(df["case"].values, df["day"].values, df["slice"].values)
]
df["path"] = [p for (p, _, _) in paths_w_h]
df["width"] = np.array([w for (_, w, _) in paths_w_h], dtype=np.int32)
df["height"] = np.array([h for (_, _, h) in paths_w_h], dtype=np.int32)

del paths_w_h, parts
df.head(5)



## === cell 5
df_test = df[
    ["id", "class_name", "path", "case", "day", "slice", "width", "height"]
].copy()
del df
df_test.reset_index(inplace=True, drop=True)
df_test.fillna("", inplace=True)
df_test.head(5)



## === cell 6
print(df_test.shape)
if DEBUG:
    df_test = df_test.sample(frac=0.05).reset_index(drop=True)
print(df_test.shape)



## === cell 7
gc.collect()




## === cell 8
def rle_encode(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted.
    IMPORTANT: competition expects flattening in column-major order (top-to-bottom, then left-to-right),
    which corresponds to Fortran order flatten on (H,W).
    """
    if img.ndim != 2:
        img = img.squeeze()
    pixels = img.reshape(-1, order="F")
    if pixels.dtype != np.uint8:
        pixels = pixels.astype(np.uint8, copy=False)

    padded = np.empty(pixels.size + 2, dtype=np.uint8)
    padded[0] = 0
    padded[-1] = 0
    padded[1:-1] = pixels

    changes = np.flatnonzero(padded[1:] != padded[:-1]) + 1
    if changes.size == 0:
        return ""
    changes[1::2] -= changes[::2]
    return " ".join(map(str, changes.tolist()))


def rle_decode(mask_rle, shape, color=1):
    """
    mask_rle: run-length as string formated (start length)
    shape: (height,width,channels) of array to return
    Returns numpy array, 1 - mask, 0 - background

    Bug fix: always return 3D array when shape has channels (even if channels==1),
    so downstream code can safely index [:,:,0].
    """
    h, w, c = int(shape[0]), int(shape[1]), int(shape[2])

    if mask_rle is None or mask_rle == "":
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
        mask = np.zeros((height, width, 3))
        for label in labels:
            mask += rle_decode(label, shape=(height, width, 3), color=np.random.rand(3))
    else:
        mask = np.zeros((height, width, 1))
        for label in labels:
            mask += rle_decode(label, shape=(height, width, 1))
    mask = mask.clip(0, 1)
    return mask




## === cell 9
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size=BATCH_SIZE, subset="train", shuffle=False):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.shuffle = shuffle
        self.subset = subset
        self.batch_size = batch_size

        self._cache = {} if subset != "train" else None

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        X = np.empty((self.batch_size, 128, 128, 3), dtype=np.float32)
        y = np.zeros((self.batch_size, 128, 128, 3), dtype=np.float32)

        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        df_b = self.df.iloc[indexes]

        for i, (img_path, w, h) in enumerate(
            zip(df_b["path"].values, df_b["width"].values, df_b["height"].values)
        ):
            img = self.__load_grayscale(img_path)  # (128,128,3)
            X[i] = img

            if self.subset == "train":
                for k, j in zip([0, 1, 2], ["large_bowel", "small_bowel", "stomach"]):
                    rles = df_b[j].values[i]
                    masks = rle_decode(
                        rles, shape=(int(h), int(w), 1)
                    )  # always (h,w,1)
                    masks = cv2.resize(
                        masks, (128, 128), interpolation=cv2.INTER_NEAREST
                    )
                    y[i, :, :, k] = masks[:, :, 0] if masks.ndim == 3 else masks

        if self.subset == "train":
            return X, y
        else:
            return X

    def __load_grayscale(self, img_path):
        if self._cache is not None:
            hit = self._cache.get(img_path)
            if hit is not None:
                return hit

        if not img_path:
            img = np.zeros((128, 128), dtype=np.float32)
        else:
            img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
            if img is None:
                img = np.zeros((128, 128), dtype=np.float32)
            else:
                img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA).astype(
                    np.float32, copy=False
                )
                mx = float(img.max())
                if mx > 0.0:
                    img = img / mx

        img = img[:, :, None]  # (128,128,1)
        img = np.repeat(img, 3, axis=-1)  # (128,128,3)

        if self._cache is not None:
            self._cache[img_path] = img
        return img




## === cell 10
gc.collect()



## === cell 11
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
    y_true = tf.cast(y_true, tf.float32)
    return binary_crossentropy(y_true, y_pred) + 0.5 * dice_loss(y_true, y_pred)


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




## === cell 12
gc.collect()



## === cell 13
custom_objects = {
    "FixedDropout": FixedDropout,
    "dice_coef": dice_coef,
    "iou_coef": iou_coef,
    "bce_dice_loss": bce_dice_loss,
}

model_path = "../input/uwmgi-unet-keras/model.h5"
if os.path.exists(model_path):
    model = load_model(model_path, custom_objects=custom_objects)
else:
    inputs = keras.Input(shape=(128, 128, 3))
    x = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = keras.layers.Conv2D(8, 3, padding="same", activation="relu")(x)
    outputs = keras.layers.Conv2D(3, 1, padding="same", activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)

    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=bce_dice_loss)

    train_df = pd.read_csv("../input/uw-madison-gi-tract-image-segmentation/train.csv")
    train_df.rename(columns={"class": "class_name"}, inplace=True)

    parts = train_df["id"].str.split("_", expand=True)
    train_df["case"] = parts[0].str.replace("case", "", regex=False).astype(np.int32)
    train_df["day"] = parts[1].str.replace("day", "", regex=False).astype(np.int32)
    train_df["slice"] = parts[3]

    TRAIN_IMG_DIR = "../input/uw-madison-gi-tract-image-segmentation/train"

    @lru_cache(maxsize=None)
    def _build_slice_map_train(case_int: int, day_int: int):
        scans_dir = os.path.join(
            TRAIN_IMG_DIR, f"case{case_int}", f"case{case_int}_day{day_int}", "scans"
        )
        try:
            files = os.listdir(scans_dir)
        except FileNotFoundError:
            return {}, scans_dir
        m = {}
        for fn in files:
            if not fn.startswith("slice_") or not fn.endswith(".png"):
                continue
            parts2 = fn[:-4].split("_")
            if len(parts2) < 4:
                continue
            idx = parts2[1]
            m[idx] = fn
        return m, scans_dir

    def _get_path_w_h_train(case_int: int, day_int: int, slice_png: str):
        slice_idx = slice_png[:-4]
        m, scans_dir = _build_slice_map_train(int(case_int), int(day_int))
        fn = m.get(slice_idx, "")
        if fn:
            parts2 = fn[:-4].split("_")
            w = int(parts2[2])
            h = int(parts2[3])
            return os.path.join(scans_dir, fn), w, h
        return "", 0, 0

    paths_w_h = [
        _get_path_w_h_train(c, d, s)
        for c, d, s in zip(
            train_df["case"].values, train_df["day"].values, train_df["slice"].values
        )
    ]
    train_df["path"] = [p for (p, _, _) in paths_w_h]
    train_df["width"] = np.array([w for (_, w, _) in paths_w_h], dtype=np.int32)
    train_df["height"] = np.array([h for (_, _, h) in paths_w_h], dtype=np.int32)
    del paths_w_h, parts

    pivot = (
        train_df.pivot_table(
            index=["id", "path", "width", "height", "case", "day", "slice"],
            columns="class_name",
            values="segmentation",
            aggfunc="first",
        )
        .reset_index()
        .fillna("")
    )

    unique_cases = np.array(sorted(pivot["case"].unique()))
    rng = np.random.RandomState(42)
    rng.shuffle(unique_cases)
    cut = int(0.9 * len(unique_cases))
    train_cases = set(unique_cases[:cut])
    val_cases = set(unique_cases[cut:])

    trn = pivot[pivot["case"].isin(train_cases)].reset_index(drop=True)
    val = pivot[pivot["case"].isin(val_cases)].reset_index(drop=True)

    trn_gen = DataGenerator(trn, batch_size=BATCH_SIZE, subset="train", shuffle=True)
    val_gen = DataGenerator(val, batch_size=BATCH_SIZE, subset="train", shuffle=False)

    steps_per_epoch = max(1, len(trn_gen))
    validation_steps = max(1, len(val_gen))

    model.fit(
        trn_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=2,
    )

gc.collect()



## === cell 14
pred_batches = DataGenerator(df_test, batch_size=16, subset="test", shuffle=False)
gc.collect()

try:
    pred_ds = tf.data.Dataset.from_generator(
        lambda: pred_batches,
        output_signature=tf.TensorSpec(shape=(None, 128, 128, 3), dtype=tf.float32),
    ).prefetch(tf.data.AUTOTUNE)
    PROBS = model.predict(pred_ds, verbose=1)
except Exception:
    PROBS = model.predict(pred_batches, verbose=1)

gc.collect()



## === cell 15
n_full = (len(df_test) // pred_batches.batch_size) * pred_batches.batch_size
if n_full < len(df_test):
    tail_df = df_test.iloc[n_full:].reset_index(drop=True)
    tail_gen = DataGenerator(tail_df, batch_size=1, subset="test", shuffle=False)
    tail_probs = model.predict(tail_gen, verbose=0)
    PROBS = np.concatenate([PROBS[:n_full], tail_probs], axis=0)
else:
    PROBS = PROBS[: len(df_test)]

len(PROBS), len(df_test)



## === cell 16
class_to_ch = {"large_bowel": 0, "small_bowel": 1, "stomach": 2}

heights = df_test["height"].to_numpy(np.int32, copy=False)
widths = df_test["width"].to_numpy(np.int32, copy=False)
classes = df_test["class_name"].to_numpy(copy=False)

rles = []
for i in tqdm(range(df_test.shape[0]), total=df_test.shape[0]):
    h = int(heights[i])
    w = int(widths[i])
    dsize = (w, h)

    ch = class_to_ch.get(classes[i], 0)

    pred_resized = cv2.resize(
        PROBS[i, :, :, ch], dsize, interpolation=cv2.INTER_NEAREST
    )
    pred_bin = (pred_resized >= 0.5).astype("uint8")
    rles.append(rle_encode(pred_bin))

gc.collect()



## === cell 17
sub = pd.DataFrame(
    {
        "id": df_test["id"].values,
        "class": df_test["class_name"].values,
        "predicted": rles,
    }
)

sample = pd.read_csv(
    "../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv"
)
sub = sample[["id", "class"]].merge(sub, on=["id", "class"], how="left")
sub["predicted"] = sub["predicted"].fillna("")

sub.to_csv("submission.csv", index=False)



## === cell 18
sub.head()



## === cell 19
print("Wrote submission.csv with shape:", sub.shape)
print("Null predicted:", sub["predicted"].isna().sum())
print("Empty predicted:", (sub["predicted"] == "").sum())
