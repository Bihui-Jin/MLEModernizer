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


def build_slice_df_fast(base_dir, df_ids):
    df_ids = df_ids[["id", "case", "day", "slice"]].copy()
    df_ids["case"] = df_ids["case"].astype(np.int32)
    df_ids["day"] = df_ids["day"].astype(np.int32)
    df_ids["slice"] = df_ids["slice"].astype(str)

    cd_to_map = {}
    paths = np.empty(len(df_ids), dtype=object)
    widths = np.empty(len(df_ids), dtype=np.int32)
    heights = np.empty(len(df_ids), dtype=np.int32)

    for i, (case, day, sl) in enumerate(
        zip(df_ids["case"].values, df_ids["day"].values, df_ids["slice"].values)
    ):
        key = (int(case), int(day))
        if key not in cd_to_map:
            scans_dir = os.path.join(
                base_dir, f"case{key[0]}", f"case{key[0]}_day{key[1]}", "scans"
            )
            files = glob(os.path.join(scans_dir, "slice_*.png"))
            m = {}
            for p in files:
                mm = _SLICE_RE.search(p.replace("\\", "/"))
                if not mm:
                    continue
                m[mm.group("slice")] = (p, int(mm.group("w")), int(mm.group("h")))
            cd_to_map[key] = m

        rec = cd_to_map[key].get(str(sl))
        if rec is None:
            paths[i] = None
            widths[i] = 0
            heights[i] = 0
        else:
            paths[i], widths[i], heights[i] = rec

    out = df_ids.copy()
    out["path"] = paths
    out["width"] = widths
    out["height"] = heights
    out = out[out["path"].notna()].reset_index(drop=True)
    return out


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
    return " ".join(changes.astype(str).tolist())


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

MMAP_MASK_PATH = os.path.join(CACHE_DIR, f"train_masks_all_{IMG_SIZE}.dat")
IDMAP_PATH = os.path.join(CACHE_DIR, f"train_masks_all_{IMG_SIZE}_ids.npy")

IMG_CACHE_DIR = "./cache_images_128"
os.makedirs(IMG_CACHE_DIR, exist_ok=True)
MMAP_IMG_PATH = os.path.join(IMG_CACHE_DIR, f"train_imgs_all_{IMG_SIZE}.dat")
IMIDMAP_PATH = os.path.join(IMG_CACHE_DIR, f"train_imgs_all_{IMG_SIZE}_ids.npy")


def _build_id_to_index(ids):
    return {rid: i for i, rid in enumerate(ids)}


def precompute_all_memmaps(df_all):
    ids = df_all["id"].values.astype(str)
    heights = df_all["height"].values.astype(np.int32)
    widths = df_all["width"].values.astype(np.int32)
    paths = df_all["path"].values.astype(str)

    need_masks = True
    if os.path.exists(MMAP_MASK_PATH) and os.path.exists(IDMAP_PATH):
        try:
            saved_ids = np.load(IDMAP_PATH, allow_pickle=False).astype(str)
            if saved_ids.shape[0] == ids.shape[0] and np.all(saved_ids == ids):
                need_masks = False
                print("Mask memmap cache: present and matching")
        except Exception:
            pass

    need_imgs = True
    if os.path.exists(MMAP_IMG_PATH) and os.path.exists(IMIDMAP_PATH):
        try:
            saved_ids2 = np.load(IMIDMAP_PATH, allow_pickle=False).astype(str)
            if saved_ids2.shape[0] == ids.shape[0] and np.all(saved_ids2 == ids):
                need_imgs = False
                print("Image memmap cache: present and matching")
        except Exception:
            pass

    if not need_masks and not need_imgs:
        return

    if need_masks:
        print(f"Mask memmap cache: building {len(df_all)} masks (global, one-time)")
        np.save(IDMAP_PATH, ids, allow_pickle=False)
        mm = np.memmap(
            MMAP_MASK_PATH,
            mode="w+",
            dtype=np.uint8,
            shape=(len(df_all), IMG_SIZE, IMG_SIZE, 3),
        )
        lb_col = df_all["large_bowel"].values
        sb_col = df_all["small_bowel"].values
        st_col = df_all["stomach"].values

        for i in tqdm(range(len(df_all)), desc="Masks->memmap"):
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
            mm[i] = out

        mm.flush()
        del mm

    if need_imgs:
        print(f"Image memmap cache: building {len(df_all)} images (global, one-time)")
        np.save(IMIDMAP_PATH, ids, allow_pickle=False)
        mmx = np.memmap(
            MMAP_IMG_PATH,
            mode="w+",
            dtype=np.float32,
            shape=(len(df_all), IMG_SIZE, IMG_SIZE, 3),
        )

        for i in tqdm(range(len(df_all)), desc="Images->memmap"):
            p = paths[i]
            im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if im is None:
                raise RuntimeError(f"Failed to read image: {p}")
            if im.ndim == 3:
                im = im[:, :, 0]
            im = cv2.resize(im, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            x = im.astype(np.float32) / 255.0
            out = np.empty((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
            out[:, :, 0] = x
            out[:, :, 1] = x
            out[:, :, 2] = x
            mmx[i] = out

        mmx.flush()
        del mmx




## === cell 9
_TRAIN_IMG_MM = None
_TRAIN_MASK_MM = None
_TRAIN_MM_N = None
_TRAIN_IMG_T = None
_TRAIN_MASK_T = None


def _open_train_memmaps_as_tensors(n):
    global _TRAIN_IMG_MM, _TRAIN_MASK_MM, _TRAIN_MM_N, _TRAIN_IMG_T, _TRAIN_MASK_T
    if _TRAIN_MM_N == n and _TRAIN_IMG_T is not None and _TRAIN_MASK_T is not None:
        return
    _TRAIN_MM_N = n

    _TRAIN_IMG_MM = np.memmap(
        MMAP_IMG_PATH, mode="r", dtype=np.float32, shape=(n, IMG_SIZE, IMG_SIZE, 3)
    )
    _TRAIN_MASK_MM = np.memmap(
        MMAP_MASK_PATH, mode="r", dtype=np.uint8, shape=(n, IMG_SIZE, IMG_SIZE, 3)
    )

    _TRAIN_IMG_T = tf.convert_to_tensor(np.asarray(_TRAIN_IMG_MM), dtype=tf.float32)
    _TRAIN_MASK_T = tf.cast(
        tf.convert_to_tensor(np.asarray(_TRAIN_MASK_MM)), tf.float32
    )


def make_dataset_from_memmaps(
    df, batch_size, subset="train", shuffle=False, id_to_index=None
):
    ids = df["id"].values.astype(str)
    if id_to_index is None:
        raise ValueError("id_to_index required")
    idx = np.fromiter((id_to_index[rid] for rid in ids), dtype=np.int32, count=len(ids))

    ds = tf.data.Dataset.from_tensor_slices(idx)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(idx), seed=42, reshuffle_each_iteration=True)

    def _map_fn(i):
        x = tf.gather(_TRAIN_IMG_T, i)
        y = tf.gather(_TRAIN_MASK_T, i)
        x = tf.ensure_shape(x, (IMG_SIZE, IMG_SIZE, 3))
        y = tf.ensure_shape(y, (IMG_SIZE, IMG_SIZE, 3))
        return x, y

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds


def make_test_dataset_png(paths, batch_size, cache_images_to_disk=False, cache_tag=""):
    paths = paths.astype(str)
    ds = tf.data.Dataset.from_tensor_slices(paths)

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

    ds = ds.map(_load_img_tf, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    if cache_images_to_disk and cache_tag:
        ds = ds.cache(os.path.join(IMG_CACHE_DIR, f"test_{cache_tag}.cache"))
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

precompute_all_memmaps(train_slices)
gc.collect()

all_ids = np.load(IDMAP_PATH, allow_pickle=False).astype(str)
id_to_index = _build_id_to_index(all_ids)

_open_train_memmaps_as_tensors(len(all_ids))

train_ds = make_dataset_from_memmaps(
    train_df,
    batch_size=BATCH_SIZE,
    subset="train",
    shuffle=True,
    id_to_index=id_to_index,
)
valid_ds = make_dataset_from_memmaps(
    valid_df,
    batch_size=BATCH_SIZE,
    subset="train",
    shuffle=False,
    id_to_index=id_to_index,
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
test_ds = make_test_dataset_png(
    test_slices["path"].values,
    batch_size=PRED_BATCH,
    cache_images_to_disk=True,
    cache_tag=f"fold{fold_selected}_test",
)

ids_arr = test_slices["id"].values.astype(str)
h_arr = test_slices["height"].values.astype(np.int32)
w_arr = test_slices["width"].values.astype(np.int32)

thr = 0.5

lbs = [""] * len(test_slices)
sbs = [""] * len(test_slices)
sts = [""] * len(test_slices)

offset = 0
for xb in tqdm(
    test_ds, total=(len(test_slices) + PRED_BATCH - 1) // PRED_BATCH, desc="Predict+RLE"
):
    pb = model.predict_on_batch(xb)  # (bs, IMG_SIZE, IMG_SIZE, 3) float32
    bs = pb.shape[0]
    for j in range(bs):
        i = offset + j
        h = int(h_arr[i])
        w = int(w_arr[i])
        p = pb[j]
        pred0 = cv2.resize(p[:, :, 0], (w, h), interpolation=cv2.INTER_NEAREST)
        pred1 = cv2.resize(p[:, :, 1], (w, h), interpolation=cv2.INTER_NEAREST)
        pred2 = cv2.resize(p[:, :, 2], (w, h), interpolation=cv2.INTER_NEAREST)

        lbs[i] = rle_encode((pred0 >= thr).astype(np.uint8, copy=False))
        sbs[i] = rle_encode((pred1 >= thr).astype(np.uint8, copy=False))
        sts[i] = rle_encode((pred2 >= thr).astype(np.uint8, copy=False))
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
