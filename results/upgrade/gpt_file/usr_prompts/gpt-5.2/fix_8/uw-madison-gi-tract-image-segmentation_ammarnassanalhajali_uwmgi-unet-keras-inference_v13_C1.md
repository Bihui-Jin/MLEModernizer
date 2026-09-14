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
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PYTHONHASHSEED"] = "42"

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

random.seed(42)

import numpy as np

np.random.seed(42)

import pandas as pd
import cv2
from glob import glob

try:
    from google.protobuf.internal import api_implementation

    try:
        api_implementation._SetType("python")
    except Exception:
        pass
except Exception:
    pass

import tensorflow as tf

tf.random.set_seed(42)

try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

print("TF version:", tf.__version__)



## === cell 1
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold

from tensorflow.keras import backend as K



## === cell 2
BATCH_SIZE = 16
EPOCHS = 30
n_splits = 5
fold_selected = 1  # 1..5

IMG_SIZE = 128
DATA_DIR = "../input/uw-madison-gi-tract-image-segmentation"



## === cell 3
sub_df = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
train_csv = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))

sub_df.rename(columns={"class": "class_name"}, inplace=True)
test_df.rename(columns={"class": "class_name"}, inplace=True)
train_csv.rename(columns={"class": "class_name"}, inplace=True)


def add_id_cols(df_):
    df_ = df_.copy()
    parts = df_["id"].str.split("_", expand=True)
    df_["case"] = parts[0].str.replace("case", "", regex=False).astype(np.int32)
    df_["day"] = parts[1].str.replace("day", "", regex=False).astype(np.int32)
    df_["slice"] = parts[3]
    return df_


test_df = add_id_cols(test_df)
train_csv = add_id_cols(train_csv)

print("train_csv:", train_csv.shape, "test_df:", test_df.shape, "sub_df:", sub_df.shape)




## === cell 4
def _scan_png_index(base_dir):
    pattern = os.path.join(base_dir, "case*", "case*_day*", "scans", "*.png")
    paths = glob(pattern)
    rows = []
    for p in paths:
        p2 = p.replace("\\", "/")
        parts = p2.split("/")
        if len(parts) < 4:
            continue
        case_dir = parts[-4]
        day_dir = parts[-3]
        if not (case_dir.startswith("case") and "_day" in day_dir):
            continue
        try:
            case = int(case_dir.replace("case", ""))
            day = int(day_dir.split("_day")[1])
        except Exception:
            continue

        fn = os.path.basename(p2)
        stem = fn[:-4]
        rparts = stem.rsplit("_", 4)
        if len(rparts) != 5:
            continue
        slice_prefix = rparts[0]  # "slice_XXXX"
        try:
            width = int(rparts[1])
            height = int(rparts[2])
        except Exception:
            continue
        sl = slice_prefix.replace("slice_", "")
        rows.append((case, day, sl, p, width, height))

    idx = pd.DataFrame(
        rows, columns=["case", "day", "slice", "path", "width", "height"]
    )
    idx = idx.drop_duplicates(
        subset=["case", "day", "slice"], keep="first"
    ).reset_index(drop=True)
    return idx


def build_slice_df(base_dir, df_ids):
    idx = _scan_png_index(base_dir)
    df_tmp = df_ids.merge(idx, on=["case", "day", "slice"], how="left")
    df_tmp = df_tmp.dropna(subset=["path"]).reset_index(drop=True)
    df_tmp["width"] = df_tmp["width"].astype(np.int32)
    df_tmp["height"] = df_tmp["height"].astype(np.int32)
    return df_tmp


train_base = os.path.join(DATA_DIR, "train")
test_base = os.path.join(DATA_DIR, "test")

train_slice_ids = (
    train_csv[["id", "case", "day", "slice"]].drop_duplicates().reset_index(drop=True)
)
test_slice_ids = (
    test_df[["id", "case", "day", "slice"]].drop_duplicates().reset_index(drop=True)
)

train_slices = build_slice_df(train_base, train_slice_ids)
test_slices = build_slice_df(test_base, test_slice_ids)

print("train_slices:", train_slices.shape, "test_slices:", test_slices.shape)
train_slices.head(3)



## === cell 5
train_pivot = train_csv.pivot_table(
    index="id", columns="class_name", values="segmentation", aggfunc="first"
).reset_index()
for c in ["large_bowel", "small_bowel", "stomach"]:
    if c not in train_pivot.columns:
        train_pivot[c] = ""
train_pivot[["large_bowel", "small_bowel", "stomach"]] = train_pivot[
    ["large_bowel", "small_bowel", "stomach"]
].fillna("")

train_slices = train_slices.merge(train_pivot, on="id", how="left")
train_slices[["large_bowel", "small_bowel", "stomach"]] = train_slices[
    ["large_bowel", "small_bowel", "stomach"]
].fillna("")

print("train_slices with masks:", train_slices.shape)
train_slices.head(3)



## === cell 6
gc.collect()




## === cell 7
def rle_encode(img):
    """
    Competition expects pixels numbered top-to-bottom, then left-to-right,
    which corresponds to Fortran order flattening.
    img: numpy array (H,W), 1 - mask, 0 - background
    Returns run length as string formatted
    """
    pixels = img.flatten(order="F")
    pixels = pixels.astype(np.uint8, copy=False)
    pixels = np.concatenate(([0], pixels, [0]))
    changes = np.flatnonzero(pixels[1:] != pixels[:-1]) + 1
    if changes.size == 0:
        return ""
    changes[1::2] -= changes[::2]
    return " ".join(map(str, changes.tolist()))


def rle_decode(mask_rle, shape):
    """
    Decode to a 2D mask (H,W) in Fortran order to match the competition format.
    """
    h, w = shape
    if (
        mask_rle is None
        or mask_rle == ""
        or (isinstance(mask_rle, float) and np.isnan(mask_rle))
    ):
        return np.zeros((h, w), dtype=np.float32)

    s = mask_rle.split()
    starts = np.asarray(s[0:][::2], dtype=np.int64) - 1
    lengths = np.asarray(s[1:][::2], dtype=np.int64)
    if starts.size == 0:
        return np.zeros((h, w), dtype=np.float32)
    ends = starts + lengths

    img = np.zeros(h * w, dtype=np.float32)
    img[starts] += 1.0
    valid_ends = ends[ends < img.size]
    img[valid_ends] -= 1.0
    img = (np.cumsum(img) > 0).astype(np.float32, copy=False)
    return img.reshape((h, w), order="F")




## === cell 8
CACHE_DIR = "./cache_masks_128"
os.makedirs(CACHE_DIR, exist_ok=True)


def _mask_cache_path(id_str):
    return os.path.join(CACHE_DIR, f"{id_str}.npy")


def precompute_train_masks(df):
    ids = df["id"].values.astype(str)
    heights = df["height"].values.astype(np.int32)
    widths = df["width"].values.astype(np.int32)

    missing_mask = np.fromiter(
        (not os.path.exists(_mask_cache_path(rid)) for rid in ids),
        dtype=bool,
        count=len(ids),
    )
    if not missing_mask.any():
        print("Mask cache: all present")
        return

    miss_idx = np.flatnonzero(missing_mask)
    print(f"Mask cache: building {len(miss_idx)}/{len(df)} masks (one-time)")

    lb_col = df["large_bowel"].values
    sb_col = df["small_bowel"].values
    st_col = df["stomach"].values

    for i in tqdm(miss_idx, total=len(miss_idx)):
        h = int(heights[i])
        w = int(widths[i])

        m0 = rle_decode(lb_col[i], shape=(h, w))
        m1 = rle_decode(sb_col[i], shape=(h, w))
        m2 = rle_decode(st_col[i], shape=(h, w))

        y0 = cv2.resize(m0, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        y1 = cv2.resize(m1, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        y2 = cv2.resize(m2, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        y = np.stack([y0, y1, y2], axis=-1).astype(np.float32, copy=False)
        np.save(_mask_cache_path(ids[i]), y, allow_pickle=False)


precompute_train_masks(train_slices)
gc.collect()




## === cell 9
def make_dataset(df, batch_size, subset="train", shuffle=False):
    paths = df["path"].values.astype(str)
    ids = df["id"].values.astype(str)

    if subset == "train":
        mask_paths = np.array([_mask_cache_path(rid) for rid in ids], dtype=object)

    if subset == "train":
        ds = tf.data.Dataset.from_tensor_slices((paths, mask_paths))
    else:
        ds = tf.data.Dataset.from_tensor_slices(paths)

    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=42, reshuffle_each_iteration=True)

    def _load_img_tf(p):
        img_bytes = tf.io.read_file(p)
        img = tf.image.decode_png(img_bytes, channels=1, dtype=tf.uint8)
        img = tf.image.resize(
            img,
            [IMG_SIZE, IMG_SIZE],
            method=tf.image.ResizeMethod.AREA,
            antialias=False,
        )
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.repeat(img, repeats=3, axis=-1)
        img.set_shape((IMG_SIZE, IMG_SIZE, 3))
        return img

    def _load_mask_npy(mp):
        def _np_load(x):
            return np.load(x.decode("utf-8"), allow_pickle=False).astype(
                np.float32, copy=False
            )

        mask = tf.numpy_function(_np_load, [mp], Tout=tf.float32)
        mask.set_shape((IMG_SIZE, IMG_SIZE, 3))
        return mask

    if subset == "train":

        def _map_fn(p, mp):
            x = _load_img_tf(p)
            y = _load_mask_npy(mp)
            return x, y

        ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    else:

        def _map_fn(p):
            x = _load_img_tf(p)
            return x

        ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)

    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
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
    y_pred = tf.cast(y_pred, tf.float32)
    return binary_crossentropy(y_true, y_pred) + 0.5 * dice_loss(y_true, y_pred)




## === cell 11
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    Concatenate,
)
from tensorflow.keras.models import Model


def build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 3), n_classes=3):
    inputs = Input(shape=input_shape)

    c1 = Conv2D(32, 3, activation="relu", padding="same")(inputs)
    c1 = Conv2D(32, 3, activation="relu", padding="same")(c1)
    p1 = MaxPooling2D()(c1)

    c2 = Conv2D(64, 3, activation="relu", padding="same")(p1)
    c2 = Conv2D(64, 3, activation="relu", padding="same")(c2)
    p2 = MaxPooling2D()(c2)

    c3 = Conv2D(128, 3, activation="relu", padding="same")(p2)
    c3 = Conv2D(128, 3, activation="relu", padding="same")(c3)
    p3 = MaxPooling2D()(c3)

    bn = Conv2D(256, 3, activation="relu", padding="same")(p3)
    bn = Conv2D(256, 3, activation="relu", padding="same")(bn)

    u3 = UpSampling2D()(bn)
    u3 = Conv2D(128, 2, activation="relu", padding="same")(u3)
    m3 = Concatenate()([u3, c3])
    c6 = Conv2D(128, 3, activation="relu", padding="same")(m3)
    c6 = Conv2D(128, 3, activation="relu", padding="same")(c6)

    u2 = UpSampling2D()(c6)
    u2 = Conv2D(64, 2, activation="relu", padding="same")(u2)
    m2 = Concatenate()([u2, c2])
    c7 = Conv2D(64, 3, activation="relu", padding="same")(m2)
    c7 = Conv2D(64, 3, activation="relu", padding="same")(c7)

    u1 = UpSampling2D()(c7)
    u1 = Conv2D(32, 2, activation="relu", padding="same")(u1)
    m1 = Concatenate()([u1, c1])
    c8 = Conv2D(32, 3, activation="relu", padding="same")(m1)
    c8 = Conv2D(32, 3, activation="relu", padding="same")(c8)

    outputs = Conv2D(n_classes, 1, activation="sigmoid", padding="same")(c8)
    return Model(inputs, outputs)


model = build_unet()
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=bce_dice_loss,
    metrics=[dice_coef, iou_coef],
)
model.summary()



## === cell 12
cases = train_slices[["case"]].copy()
cases["case_mod"] = cases["case"] % 10  # cheap strat label

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
folds = list(skf.split(train_slices, cases["case_mod"].values))
tr_idx, va_idx = folds[fold_selected - 1]

train_df = train_slices.iloc[tr_idx].reset_index(drop=True)
valid_df = train_slices.iloc[va_idx].reset_index(drop=True)

print("train/valid:", train_df.shape, valid_df.shape)

precompute_train_masks(train_df)
precompute_train_masks(valid_df)

train_ds = make_dataset(train_df, batch_size=BATCH_SIZE, subset="train", shuffle=True)
valid_ds = make_dataset(valid_df, batch_size=BATCH_SIZE, subset="train", shuffle=False)



## === cell 13
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    verbose=2,
)

gc.collect()



## === cell 14
PRED_BATCH = 16
test_ds = make_dataset(test_slices, batch_size=PRED_BATCH, subset="test", shuffle=False)
LOGITS = model.predict(test_ds, verbose=1)
LOGITS = LOGITS[: len(test_slices)]
print("LOGITS:", LOGITS.shape)
gc.collect()



## === cell 15
ids_arr = test_slices["id"].values
h_arr = test_slices["height"].values.astype(np.int32)
w_arr = test_slices["width"].values.astype(np.int32)

resize = cv2.resize
INTERP = cv2.INTER_NEAREST
thr = 0.5

lbs, sbs, sts = [], [], []
append_lb = lbs.append
append_sb = sbs.append
append_st = sts.append

for index in tqdm(range(len(test_slices))):
    h = int(h_arr[index])
    w = int(w_arr[index])

    p = LOGITS[index]
    pred0 = resize(p[:, :, 0], (w, h), interpolation=INTERP)
    pred1 = resize(p[:, :, 1], (w, h), interpolation=INTERP)
    pred2 = resize(p[:, :, 2], (w, h), interpolation=INTERP)

    append_lb(rle_encode((pred0 >= thr).astype(np.uint8, copy=False)))
    append_sb(rle_encode((pred1 >= thr).astype(np.uint8, copy=False)))
    append_st(rle_encode((pred2 >= thr).astype(np.uint8, copy=False)))

del LOGITS
gc.collect()



## === cell 16
pred_df = pd.DataFrame(
    {
        "id": np.repeat(ids_arr, 3),
        "class_name": np.tile(
            np.array(["large_bowel", "small_bowel", "stomach"]), len(ids_arr)
        ),
        "predicted": np.concatenate(
            [
                np.array(lbs, dtype=object),
                np.array(sbs, dtype=object),
                np.array(sts, dtype=object),
            ]
        )
        .reshape(3, -1)
        .T.reshape(-1),
    }
)

sub = sub_df.merge(pred_df, on=["id", "class_name"], how="left")
sub["predicted"] = sub["predicted"].fillna("")
sub.rename(columns={"class_name": "class"}, inplace=True)

sub = sub[["id", "class", "predicted"]]
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
sub.head()
