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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import warnings

warnings.filterwarnings("ignore")




## === cell 1
import pandas as pd
import numpy as np
import gc
import cv2
from glob import glob
from tqdm import tqdm

import tensorflow as tf  # type: ignore
from tensorflow import keras
from tensorflow.keras import backend as K

print("TensorFlow version:", tf.__version__)
print("Eager execution:", tf.executing_eagerly())

np.random.seed(0)
try:
    tf.random.set_seed(0)
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
sample_sub = pd.read_csv(
    "../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv"
)
test_df = pd.read_csv("../input/uw-madison-gi-tract-image-segmentation/test.csv")
train_csv = pd.read_csv("../input/uw-madison-gi-tract-image-segmentation/train.csv")

print(
    "sample_sub:",
    sample_sub.shape,
    "test_df:",
    test_df.shape,
    "train_csv:",
    train_csv.shape,
)




## === cell 4
def add_meta_and_paths(df_in, base_dir):
    df = df_in.copy()
    df.rename(columns={"class": "class_name"}, inplace=True)

    parts = df["id"].str.split("_", expand=True)
    df["case"] = parts[0].str.replace("case", "", regex=False).astype(np.int32)
    df["day"] = parts[1].str.replace("day", "", regex=False).astype(np.int32)
    df["slice"] = parts[3].astype(str)

    case_str = "case" + df["case"].astype(str)
    day_str = "day" + df["day"].astype(str)

    scan_dir = base_dir + "/" + case_str + "/" + case_str + "_" + day_str + "/scans/"
    df["scan_dir"] = scan_dir
    df["scan_glob"] = df["scan_dir"] + "slice_" + df["slice"] + "_*.png"

    def _resolve_one(pat):
        m = glob(pat)
        if len(m) != 1:
            return (
                None if len(m) == 0 else m[0]
            )  # deterministic fallback if duplicates exist
        return m[0]

    df["path"] = df["scan_glob"].map(_resolve_one)

    if df["path"].isna().any():
        missing = df[df["path"].isna()].head(5)
        raise FileNotFoundError(
            f"Some image paths could not be resolved. Examples:\n{missing[['id','scan_glob']]}"
        )

    stem = df["path"].str.rsplit("/", n=1, expand=True)[1].str[:-4]
    wh = stem.str.rsplit("_", n=4, expand=True)
    df["width"] = wh[1].astype(np.int32)
    df["height"] = wh[2].astype(np.int32)

    df.drop(columns=["scan_dir", "scan_glob"], inplace=True)
    return df




## === cell 5
test_df2 = add_meta_and_paths(
    test_df, "../input/uw-madison-gi-tract-image-segmentation/test"
)

train_df2 = add_meta_and_paths(
    train_csv, "../input/uw-madison-gi-tract-image-segmentation/train"
)

train_pivot = train_df2.pivot_table(
    index=["id", "path", "case", "day", "slice", "width", "height"],
    columns="class_name",
    values="segmentation",
    aggfunc="first",
).reset_index()
for c in ["large_bowel", "small_bowel", "stomach"]:
    if c not in train_pivot.columns:
        train_pivot[c] = ""
train_pivot[["large_bowel", "small_bowel", "stomach"]] = train_pivot[
    ["large_bowel", "small_bowel", "stomach"]
].fillna("")

print("train_pivot:", train_pivot.shape, "test_df2:", test_df2.shape)




## === cell 6
test_unique = test_df2.drop_duplicates(subset=["id"], keep="first").reset_index(
    drop=True
)
print("test_unique:", test_unique.shape)

gc.collect()




## === cell 7
def rle_encode(img):
    """
    img: numpy array, 1 - mask, 0 - background (2D)
    Returns run length as string formatted.
    """
    if img is None:
        return ""
    pixels = (img > 0).astype(np.uint8).flatten(order="F")
    p = np.concatenate(([0], pixels, [0]))
    changes = np.flatnonzero(p[1:] != p[:-1]) + 1
    if changes.size == 0:
        return ""
    changes[1::2] -= changes[::2]
    return " ".join(map(str, changes))


def rle_decode(mask_rle, shape, color=1):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height,width)
    Returns numpy array (H, W) with values {0, color}
    """
    h, w = shape
    if mask_rle is None or mask_rle == "":
        return np.zeros((h, w), dtype=np.float32)

    s = mask_rle.split()
    starts = np.asarray(s[0::2], dtype=np.int64) - 1
    lengths = np.asarray(s[1::2], dtype=np.int64)
    ends = starts + lengths

    img = np.zeros((h * w,), dtype=np.float32)
    if starts.size == 0:
        return img.reshape((h, w), order="F")

    boundaries = np.zeros((h * w + 1,), dtype=np.int32)
    np.add.at(boundaries, starts, 1)
    np.add.at(boundaries, ends, -1)
    run_mask = np.cumsum(boundaries[:-1]) > 0
    img[run_mask] = float(color)

    return img.reshape((h, w), order="F")




## === cell 8
MASK_CACHE_PATH = "/kaggle/working/masks_128x128_float32.mmap"


def build_or_load_mask_cache(df):
    n = len(df)
    if os.path.exists(MASK_CACHE_PATH):
        mm = np.memmap(
            MASK_CACHE_PATH, mode="r", dtype=np.float32, shape=(n, 128, 128, 3)
        )
        return mm

    mm = np.memmap(MASK_CACHE_PATH, mode="w+", dtype=np.float32, shape=(n, 128, 128, 3))

    cols = ["large_bowel", "small_bowel", "stomach"]
    heights = df["height"].astype(np.int32).values
    widths = df["width"].astype(np.int32).values
    rle_arrays = [df[c].astype(str).values for c in cols]

    for i in tqdm(range(n), total=n):
        h = int(heights[i])
        w = int(widths[i])
        for k in range(3):
            rles = rle_arrays[k][i]
            mask2d = rle_decode(rles, shape=(h, w))
            mask2d = cv2.resize(mask2d, (128, 128), interpolation=cv2.INTER_NEAREST)
            mm[i, :, :, k] = mask2d

    mm.flush()
    mm = np.memmap(MASK_CACHE_PATH, mode="r", dtype=np.float32, shape=(n, 128, 128, 3))
    return mm




## === cell 9
IMG_CACHE_PATH = "/kaggle/working/images_128x128_float32.mmap"
_IMG_CACHE = None
_IMG_CACHE_INDEX = None


def build_or_load_image_cache(paths):
    global _IMG_CACHE, _IMG_CACHE_INDEX
    uniq_paths = pd.Index(paths).unique().tolist()
    uniq_paths.sort()
    n = len(uniq_paths)

    index = {p: i for i, p in enumerate(uniq_paths)}
    _IMG_CACHE_INDEX = index

    if os.path.exists(IMG_CACHE_PATH):
        mm = np.memmap(
            IMG_CACHE_PATH, mode="r", dtype=np.float32, shape=(n, 128, 128, 1)
        )
        _IMG_CACHE = mm
        return mm, index

    mm = np.memmap(IMG_CACHE_PATH, mode="w+", dtype=np.float32, shape=(n, 128, 128, 1))
    for p, i in tqdm(index.items(), total=n):
        img = cv2.imread(p, cv2.IMREAD_ANYDEPTH)
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)
        mm[i] = img
    mm.flush()
    mm = np.memmap(IMG_CACHE_PATH, mode="r", dtype=np.float32, shape=(n, 128, 128, 1))
    _IMG_CACHE = mm
    return mm, index


def make_tf_dataset(
    df,
    subset="train",
    batch_size=BATCH_SIZE,
    shuffle=False,
    mask_cache=None,
    img_cache=None,
    img_index=None,
):
    paths = df["path"].astype(str).values
    if img_cache is None or img_index is None:
        raise ValueError(
            "img_cache and img_index must be provided for this optimized pipeline."
        )

    img_ids = np.fromiter(
        (img_index[p] for p in paths), dtype=np.int32, count=len(paths)
    )
    ds = tf.data.Dataset.from_tensor_slices(
        (np.arange(len(paths), dtype=np.int32), img_ids)
    )

    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=0, reshuffle_each_iteration=True)

    def _load_x(row_idx, img_idx):
        def _py_read(i):
            return img_cache[int(i)]

        x = tf.numpy_function(_py_read, [img_idx], tf.float32)
        x.set_shape((128, 128, 1))
        return row_idx, x

    ds = ds.map(_load_x, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)

    if subset == "train":
        if mask_cache is None:
            raise ValueError(
                "mask_cache must be provided for train subset in this pipeline."
            )

        def _load_y(row_idx, x):
            def _py_mask(i):
                return mask_cache[int(i)]

            y = tf.numpy_function(_py_mask, [row_idx], tf.float32)
            y.set_shape((128, 128, 3))
            return x, y

        ds = ds.map(_load_y, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    else:
        ds = ds.map(
            lambda row_idx, x: x,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds




## === cell 10
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
def conv_block(x, filters, dropout=0.0):
    x = keras.layers.Conv2D(filters, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(filters, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    if dropout and dropout > 0:
        x = FixedDropout(dropout)(x)
    return x


def build_unet(input_shape=(128, 128, 1), n_classes=3):
    inputs = keras.Input(shape=input_shape)

    c1 = conv_block(inputs, 32, dropout=0.0)
    p1 = keras.layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, 64, dropout=0.0)
    p2 = keras.layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, 128, dropout=0.0)
    p3 = keras.layers.MaxPooling2D()(c3)

    bn = conv_block(p3, 256, dropout=0.0)

    u3 = keras.layers.UpSampling2D()(bn)
    u3 = keras.layers.Concatenate()([u3, c3])
    c6 = conv_block(u3, 128, dropout=0.0)

    u2 = keras.layers.UpSampling2D()(c6)
    u2 = keras.layers.Concatenate()([u2, c2])
    c7 = conv_block(u2, 64, dropout=0.0)

    u1 = keras.layers.UpSampling2D()(c7)
    u1 = keras.layers.Concatenate()([u1, c1])
    c8 = conv_block(u1, 32, dropout=0.0)

    outputs = keras.layers.Conv2D(n_classes, 1, activation="sigmoid")(c8)
    model = keras.Model(inputs, outputs)
    return model


model = build_unet()
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=bce_dice_loss,
    metrics=[dice_coef, iou_coef],
)

model.summary()
gc.collect()




## === cell 12
unique_cases = train_pivot["case"].unique()
unique_cases = np.sort(unique_cases)

fold_idx = (fold_selected - 1) % n_splits
case_folds = np.array_split(unique_cases, n_splits)
val_cases = set(case_folds[fold_idx].tolist())

train_rows = train_pivot[~train_pivot["case"].isin(val_cases)].reset_index(drop=True)
val_rows = train_pivot[train_pivot["case"].isin(val_cases)].reset_index(drop=True)

print("train_rows:", train_rows.shape, "val_rows:", val_rows.shape)

all_paths = np.concatenate(
    [
        train_rows["path"].astype(str).values,
        val_rows["path"].astype(str).values,
        test_unique["path"].astype(str).values,
    ]
)
img_cache, img_index = build_or_load_image_cache(all_paths)

train_mask_cache = build_or_load_mask_cache(train_rows)
val_mask_cache_path = "/kaggle/working/masks_val_128x128_float32.mmap"
if not os.path.exists(val_mask_cache_path):
    old = MASK_CACHE_PATH
    MASK_CACHE_PATH = val_mask_cache_path
    val_mask_cache = build_or_load_mask_cache(val_rows)
    MASK_CACHE_PATH = old
else:
    val_mask_cache = np.memmap(
        val_mask_cache_path,
        mode="r",
        dtype=np.float32,
        shape=(len(val_rows), 128, 128, 3),
    )

train_gen = make_tf_dataset(
    train_rows,
    batch_size=BATCH_SIZE,
    subset="train",
    shuffle=True,
    mask_cache=train_mask_cache,
    img_cache=img_cache,
    img_index=img_index,
)
val_gen = make_tf_dataset(
    val_rows,
    batch_size=BATCH_SIZE,
    subset="train",
    shuffle=False,
    mask_cache=val_mask_cache,
    img_cache=img_cache,
    img_index=img_index,
)

gc.collect()




## === cell 13
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    verbose=2,
)

gc.collect()




## === cell 14
pred_gen = make_tf_dataset(
    test_unique,
    batch_size=32,
    subset="test",
    shuffle=False,
    img_cache=img_cache,
    img_index=img_index,
)

LOGITS = model.predict(
    pred_gen,
    verbose=1,
)

print("LOGITS:", LOGITS.shape)
gc.collect()




## === cell 15
pred_map = {}  # (id, class) -> rle

ids = test_unique["id"].values
hs = test_unique["height"].astype(np.int32).values
ws = test_unique["width"].astype(np.int32).values
classes = ["large_bowel", "small_bowel", "stomach"]

for index in tqdm(range(len(test_unique)), total=len(test_unique)):
    img_id = ids[index]
    h = int(hs[index])
    w = int(ws[index])
    dsize = (w, h)

    for k, cls in enumerate(classes):
        pred2d = LOGITS[index, :, :, k]
        pred2d = cv2.resize(pred2d, dsize, interpolation=cv2.INTER_NEAREST)
        pred_arr = (pred2d > 0.5).astype("uint8")
        pred_map[(img_id, cls)] = rle_encode(pred_arr)

del LOGITS
gc.collect()

sub = sample_sub.copy()
sub["predicted"] = [
    pred_map.get((rid, rcls), "")
    for rid, rcls in zip(sub["id"].values, sub["class"].values)
]

assert sub.shape[0] == sample_sub.shape[0]
sub["predicted"] = sub["predicted"].fillna("")

sub.to_csv("submission.csv", index=False)
print(sub.shape)
print(sub.head())
print("Wrote submission.csv")
