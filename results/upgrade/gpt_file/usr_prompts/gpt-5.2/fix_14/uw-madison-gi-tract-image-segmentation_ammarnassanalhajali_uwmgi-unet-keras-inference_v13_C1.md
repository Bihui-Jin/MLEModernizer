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
import re


def _parse_scan_filename(fn):
    if not fn.endswith(".png"):
        return None
    stem = fn[:-4]
    parts = stem.split("_")
    if len(parts) < 6 or parts[0] != "slice":
        return None
    slice_part = parts[1]
    try:
        width = int(parts[2])
        height = int(parts[3])
    except Exception:
        width, height = None, None
    return slice_part, width, height


def _build_scans_dir(base_dir, case_num, day_num):
    return os.path.join(
        base_dir, f"case{case_num}", f"case{case_num}_day{day_num}", "scans"
    )


_SLICE_RE = re.compile(
    r"case(?P<case>\d+)/case(?P=case)_day(?P<day>\d+)/scans/slice_(?P<slice>\d+)_(?P<w>\d+)_(?P<h>\d+)_.*\.png$"
)


def _index_scans_fast(base_dir):
    pat = os.path.join(base_dir, "case*", "case*_day*", "scans", "slice_*.png")
    files = glob(pat)
    rows = []
    for p in files:
        m = _SLICE_RE.search(p.replace("\\", "/"))
        if not m:
            continue
        rows.append(
            (
                int(m.group("case")),
                int(m.group("day")),
                m.group("slice"),
                p,
                int(m.group("w")),
                int(m.group("h")),
            )
        )
    if not rows:
        return pd.DataFrame(columns=["case", "day", "slice", "path", "width", "height"])
    df = pd.DataFrame(rows, columns=["case", "day", "slice", "path", "width", "height"])
    df["case"] = df["case"].astype(np.int32)
    df["day"] = df["day"].astype(np.int32)
    df["slice"] = df["slice"].astype(str)
    df["width"] = df["width"].astype(np.int32)
    df["height"] = df["height"].astype(np.int32)
    df = df.drop_duplicates(subset=["case", "day", "slice"], keep="first")
    return df


def build_slice_df_fast(base_dir, df_ids):
    df_ids = df_ids[["id", "case", "day", "slice"]].copy()
    df_ids["case"] = df_ids["case"].astype(np.int32)
    df_ids["day"] = df_ids["day"].astype(np.int32)
    df_ids["slice"] = df_ids["slice"].astype(str)

    scan_index = _index_scans_fast(base_dir)
    out = df_ids.merge(scan_index, on=["case", "day", "slice"], how="inner")
    return out.reset_index(drop=True)


train_base = os.path.join(DATA_DIR, "train")
test_base = os.path.join(DATA_DIR, "test")

train_slice_ids = (
    train_csv[["id", "case", "day", "slice"]].drop_duplicates().reset_index(drop=True)
)
test_slice_ids = (
    test_df[["id", "case", "day", "slice"]].drop_duplicates().reset_index(drop=True)
)

train_slices = build_slice_df_fast(train_base, train_slice_ids)
test_slices = build_slice_df_fast(test_base, test_slice_ids)

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
    pixels = img.flatten(order="F")
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
    return " ".join(map(str, changes))


def rle_decode(mask_rle, shape):
    h, w = shape
    if (
        mask_rle is None
        or mask_rle == ""
        or (isinstance(mask_rle, float) and np.isnan(mask_rle))
    ):
        return np.zeros((h, w), dtype=np.float32)
    s = mask_rle.split()
    if not s:
        return np.zeros((h, w), dtype=np.float32)

    starts = np.asarray(s[0::2], dtype=np.int64) - 1
    lengths = np.asarray(s[1::2], dtype=np.int64)
    if starts.size == 0:
        return np.zeros((h, w), dtype=np.float32)

    n = h * w
    ends = starts + lengths
    starts = np.clip(starts, 0, n)
    ends = np.clip(ends, 0, n)

    diff = np.zeros(n + 1, dtype=np.int16)
    np.add.at(diff, starts, 1)
    np.add.at(diff, ends, -1)
    flat = (np.cumsum(diff[:-1]) > 0).astype(np.float32, copy=False)
    return flat.reshape((h, w), order="F")




## === cell 8
CACHE_DIR = "./cache_masks_128"
os.makedirs(CACHE_DIR, exist_ok=True)

MMAP_PATH = os.path.join(CACHE_DIR, f"train_masks_fold{fold_selected}_{IMG_SIZE}.dat")
IDMAP_PATH = os.path.join(
    CACHE_DIR, f"train_masks_fold{fold_selected}_{IMG_SIZE}_ids.npy"
)


def _build_id_to_index(ids):
    return {rid: i for i, rid in enumerate(ids)}


from concurrent.futures import ProcessPoolExecutor, as_completed


def _mask_worker_chunk(args):
    start_i, end_i, heights, widths, lb_col, sb_col, st_col = args
    out_rows = []
    for i in range(start_i, end_i):
        h = int(heights[i])
        w = int(widths[i])
        m0 = rle_decode(lb_col[i], shape=(h, w))
        m1 = rle_decode(sb_col[i], shape=(h, w))
        m2 = rle_decode(st_col[i], shape=(h, w))

        y0 = cv2.resize(m0, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        y1 = cv2.resize(m1, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        y2 = cv2.resize(m2, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        out = np.empty((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        out[:, :, 0] = (y0 > 0.5).astype(np.uint8, copy=False)
        out[:, :, 1] = (y1 > 0.5).astype(np.uint8, copy=False)
        out[:, :, 2] = (y2 > 0.5).astype(np.uint8, copy=False)
        out_rows.append((i, out))
    return out_rows


def precompute_train_masks_memmap(df_all):
    ids = df_all["id"].values.astype(str)
    heights = df_all["height"].values.astype(np.int32)
    widths = df_all["width"].values.astype(np.int32)

    if os.path.exists(MMAP_PATH) and os.path.exists(IDMAP_PATH):
        try:
            saved_ids = np.load(IDMAP_PATH, allow_pickle=False).astype(str)
            if saved_ids.shape[0] == ids.shape[0] and np.all(saved_ids == ids):
                print("Mask memmap cache: present and matching")
                return
        except Exception:
            pass

    print(f"Mask memmap cache: building {len(df_all)} masks (fold-local, one-time)")
    np.save(IDMAP_PATH, ids, allow_pickle=False)

    mm = np.memmap(
        MMAP_PATH,
        mode="w+",
        dtype=np.uint8,
        shape=(len(df_all), IMG_SIZE, IMG_SIZE, 3),
    )

    lb_col = df_all["large_bowel"].values
    sb_col = df_all["small_bowel"].values
    st_col = df_all["stomach"].values

    max_workers = min(4, os.cpu_count() or 1)
    chunk = 128  # fewer, bigger tasks => far less IPC/overhead

    tasks = []
    for start in range(0, len(df_all), chunk):
        end = min(len(df_all), start + chunk)
        tasks.append((start, end, heights, widths, lb_col, sb_col, st_col))

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_mask_worker_chunk, t) for t in tasks]
        for fu in as_completed(futures):
            rows = fu.result()
            for j, arr in rows:
                mm[j] = arr

    mm.flush()
    del mm




## === cell 9
_MASK_MM = None
_MASK_SHAPE = None
_MASK_PATH = None
_MASK_DTYPE = None


def _ensure_mask_memmap_open(masks_memmap_path, masks_shape, masks_dtype):
    global _MASK_MM, _MASK_SHAPE, _MASK_PATH, _MASK_DTYPE
    if (
        _MASK_MM is None
        or _MASK_SHAPE != masks_shape
        or _MASK_PATH != masks_memmap_path
        or _MASK_DTYPE != masks_dtype
    ):
        _MASK_PATH = masks_memmap_path
        _MASK_SHAPE = masks_shape
        _MASK_DTYPE = masks_dtype
        _MASK_MM = np.memmap(_MASK_PATH, mode="r", dtype=_MASK_DTYPE, shape=_MASK_SHAPE)


IMG_CACHE_DIR = "./cache_images_128"
os.makedirs(IMG_CACHE_DIR, exist_ok=True)


def make_dataset(
    df,
    batch_size,
    subset="train",
    shuffle=False,
    id_to_index=None,
    masks_memmap_path=None,
    masks_shape=None,
    masks_dtype=None,
    cache_in_memory=False,
    cache_images_to_disk=False,
    cache_tag="",
):
    paths = df["path"].values.astype(str)
    ids = df["id"].values.astype(str)

    if subset == "train":
        if id_to_index is None:
            raise ValueError("id_to_index required for train dataset")
        if masks_memmap_path is None or masks_shape is None or masks_dtype is None:
            raise ValueError(
                "masks_memmap_path/masks_shape/masks_dtype required for train dataset"
            )
        mask_idx = np.fromiter(
            (id_to_index[rid] for rid in ids), dtype=np.int32, count=len(ids)
        )
        ds = tf.data.Dataset.from_tensor_slices((paths, mask_idx))
    else:
        ds = tf.data.Dataset.from_tensor_slices(paths)

    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=42, reshuffle_each_iteration=True)

    def _load_img_tf(p):
        img_bytes = tf.io.read_file(p)
        img = tf.io.decode_png(img_bytes, channels=1)  # uint8
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

    if subset == "train":
        _ensure_mask_memmap_open(masks_memmap_path, masks_shape, masks_dtype)

        def _read_mask_np(mi_np):
            mi = int(mi_np)
            return np.array(_MASK_MM[mi], copy=False)

        def _map_fn(p, mi):
            x = _load_img_tf(p)
            y = tf.numpy_function(_read_mask_np, [mi], Tout=tf.uint8)
            y = tf.ensure_shape(y, (IMG_SIZE, IMG_SIZE, 3))
            y = tf.cast(y, tf.float32)
            return x, y

        ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)

        if cache_images_to_disk and cache_tag:
            ds = ds.cache(os.path.join(IMG_CACHE_DIR, f"train_{cache_tag}.cache"))
    else:

        def _map_fn(p):
            x = _load_img_tf(p)
            return x

        ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
        if cache_images_to_disk and cache_tag:
            ds = ds.cache(os.path.join(IMG_CACHE_DIR, f"test_{cache_tag}.cache"))

    if cache_in_memory:
        ds = ds.cache()

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

fold_masks_df = pd.concat([train_df, valid_df], axis=0, ignore_index=True)
precompute_train_masks_memmap(fold_masks_df)
gc.collect()

all_ids = np.load(IDMAP_PATH, allow_pickle=False).astype(str)
id_to_index = _build_id_to_index(all_ids)

masks_shape = (len(all_ids), IMG_SIZE, IMG_SIZE, 3)

train_ds = make_dataset(
    train_df,
    batch_size=BATCH_SIZE,
    subset="train",
    shuffle=True,
    id_to_index=id_to_index,
    masks_memmap_path=MMAP_PATH,
    masks_shape=masks_shape,
    masks_dtype=np.uint8,
    cache_in_memory=False,
    cache_images_to_disk=True,
    cache_tag=f"fold{fold_selected}_train",
)
valid_ds = make_dataset(
    valid_df,
    batch_size=BATCH_SIZE,
    subset="train",
    shuffle=False,
    id_to_index=id_to_index,
    masks_memmap_path=MMAP_PATH,
    masks_shape=masks_shape,
    masks_dtype=np.uint8,
    cache_in_memory=True,  # validation repeats each epoch; safe to cache in memory
)



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
test_ds = make_dataset(
    test_slices,
    batch_size=PRED_BATCH,
    subset="test",
    shuffle=False,
    cache_images_to_disk=True,
    cache_tag=f"fold{fold_selected}_test",
)

ids_arr = test_slices["id"].values.astype(str)
h_arr = test_slices["height"].values.astype(np.int32)
w_arr = test_slices["width"].values.astype(np.int32)

thr = 0.5

lbs = []
sbs = []
sts = []

offset = 0
for xb in tqdm(test_ds, total=(len(test_slices) + PRED_BATCH - 1) // PRED_BATCH):
    pb = model.predict_on_batch(xb)  # (bs, 128, 128, 3), float32
    bs = pb.shape[0]
    for k in range(bs):
        i = offset + k
        h = int(h_arr[i])
        w = int(w_arr[i])
        p = pb[k]

        pred0 = cv2.resize(p[:, :, 0], (w, h), interpolation=cv2.INTER_NEAREST)
        pred1 = cv2.resize(p[:, :, 1], (w, h), interpolation=cv2.INTER_NEAREST)
        pred2 = cv2.resize(p[:, :, 2], (w, h), interpolation=cv2.INTER_NEAREST)

        lbs.append(rle_encode((pred0 >= thr).astype(np.uint8, copy=False)))
        sbs.append(rle_encode((pred1 >= thr).astype(np.uint8, copy=False)))
        sts.append(rle_encode((pred2 >= thr).astype(np.uint8, copy=False)))
    offset += bs

gc.collect()

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
