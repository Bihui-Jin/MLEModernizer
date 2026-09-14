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
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

from concurrent.futures import ThreadPoolExecutor

_CPU_WORKERS = max(1, min(8, (os.cpu_count() or 2)))



## === cell 1
BATCH_SIZE = 16
EPOCHS = 30
n_splits = 5
fold_selected = 1  # 1..5



## === cell 2
df = pd.read_csv("../input/uw-madison-gi-tract-image-segmentation/train.csv")
DEBUG = True

df.rename(columns={"class": "class_name"}, inplace=True)

id_split = df["id"].str.split("_", expand=True)
df["case"] = id_split[0].str.replace("case", "", regex=False).astype(np.int32)
df["day"] = id_split[1].str.replace("day", "", regex=False).astype(np.int32)
df["slice"] = (id_split[2] + "_" + id_split[3]).astype(str)  # 'slice_0001'

TRAIN_DIR = "../input/uw-madison-gi-tract-image-segmentation/train"


def _resolve_scan_png_path(
    train_or_test_dir: str, case: int, day: int, slice_token: str
) -> str:
    pat = (
        f"{train_or_test_dir}/case{case}/case{case}_day{day}/scans/"
        f"{slice_token}_*.png"
    )
    hits = glob(pat)
    if not hits:
        return ""
    hits.sort()
    return hits[0]


def _parse_png_meta_from_path(p: str):
    """
    Returns: width(int), height(int) from filename: slice_0001_266_266_1.50_1.50.png
    """
    fname = os.path.basename(p)[:-4]
    fparts = fname.split("_")
    if len(fparts) < 6:
        raise ValueError(f"Unexpected PNG filename format: {os.path.basename(p)}")
    width = int(float(fparts[2]))
    height = int(float(fparts[3]))
    return width, height


def _build_scan_index(base_dir: str, case: int, day: int) -> dict:
    scans_dir = f"{base_dir}/case{case}/case{case}_day{day}/scans"
    if not os.path.isdir(scans_dir):
        return {}
    m = {}
    try:
        for fn in os.listdir(scans_dir):
            if not fn.endswith(".png"):
                continue
            parts = fn.split("_")
            if len(parts) < 6:
                continue
            token = parts[0] + "_" + parts[1]
            p = os.path.join(scans_dir, fn)
            prev = m.get(token)
            if prev is None or fn < os.path.basename(prev):
                m[token] = p
    except FileNotFoundError:
        return {}
    return m


def _resolve_paths_and_meta(
    df_uniq: pd.DataFrame, base_dir: str, desc: str
) -> pd.DataFrame:
    pairs = (
        df_uniq[["case", "day"]].drop_duplicates().itertuples(index=False, name=None)
    )
    idx = {}
    for case, day in tqdm(list(pairs), desc=f"Indexing {desc} scan dirs (case/day)"):
        idx[(int(case), int(day))] = _build_scan_index(base_dir, int(case), int(day))

    n = len(df_uniq)
    paths = np.empty(n, dtype=object)
    widths = np.empty(n, dtype=np.int32)
    heights = np.empty(n, dtype=np.int32)

    cases = df_uniq["case"].astype(np.int32).values
    days = df_uniq["day"].astype(np.int32).values
    slices = df_uniq["slice"].values

    for i in tqdm(range(n), desc=f"Resolving {desc} PNG paths (indexed)"):
        key = (int(cases[i]), int(days[i]))
        p = idx.get(key, {}).get(slices[i], "")
        if not p:
            paths[i] = ""
            widths[i] = -1
            heights[i] = -1
            continue
        try:
            w, h = _parse_png_meta_from_path(p)
        except Exception:
            p = ""
            w, h = -1, -1
        paths[i] = p
        widths[i] = w
        heights[i] = h

    out = df_uniq.copy()
    out["path"] = paths
    out["width"] = widths
    out["height"] = heights
    return out


uniq = df[["id", "case", "day", "slice"]].drop_duplicates().reset_index(drop=True)
uniq = _resolve_paths_and_meta(uniq, TRAIN_DIR, desc="train")

uniq = uniq[uniq["path"] != ""].copy()
if uniq.shape[0] == 0:
    raise FileNotFoundError(
        f"No PNGs resolved under {TRAIN_DIR}. Check directory structure."
    )

df = df.merge(uniq[["id", "path", "width", "height"]], on="id", how="inner")

del uniq, id_split
gc.collect()
df.head(5)



## === cell 3
df = df.sort_values(["id", "class_name"]).reset_index(drop=True)

df_wide = df.pivot_table(
    index=["id", "case", "day", "slice", "width", "height", "path"],
    columns="class_name",
    values="segmentation",
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



## === cell 4
print(df_train.shape)

df_train = df_train.sample(frac=0.30, random_state=42).reset_index(drop=True)
print(df_train.shape)

if df_train.shape[0] == 0:
    raise RuntimeError(
        "After resolving ids to PNG paths, df_train is empty. "
        "This indicates an id->slice->filename join issue."
    )



## === cell 5
gc.collect()




## === cell 6
def rle_encode(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted.

    Competition expects pixels numbered top-to-bottom then left-to-right,
    which corresponds to flatten(order='F') for a (H,W) array.
    """
    if img.ndim != 2:
        img = img.squeeze()
    if img.size == 0:
        return ""
    pixels = img.flatten(order="F")
    if pixels.max(initial=0) == 0:
        return ""
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(map(str, runs.astype(np.int64)))


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
    starts = np.asarray(s[0::2], dtype=np.int64) - 1
    lengths = np.asarray(s[1::2], dtype=np.int64)
    ends = starts + lengths

    mark = np.zeros(h * w + 1, dtype=np.int32)
    np.add.at(mark, starts, 1)
    np.add.at(mark, ends, -1)
    m = (np.cumsum(mark[:-1]) > 0).astype(np.float32)

    if c == 1:
        out = m.reshape((h, w, 1), order="F")
        if color != 1:
            out *= float(color)
        return out
    else:
        out = np.zeros((h * w, c), dtype=np.float32)
        out[:] = m[:, None] * np.asarray(color, dtype=np.float32)[None, :]
        return out.reshape((h, w, c), order="F")


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




## === cell 7
_MASK_CACHE_128 = {}
_IMG_CACHE_128 = {}


def _get_mask_128_cached(rle: str, h: int, w: int) -> np.ndarray:
    key = (h, w, rle)
    m = _MASK_CACHE_128.get(key)
    if m is not None:
        return m
    m_full = rle_decode(rle, shape=(h, w, 1))  # float32 (h,w,1)
    m_128 = cv2.resize(m_full, (128, 128), interpolation=cv2.INTER_NEAREST)
    if m_128.ndim == 2:
        m_128 = m_128[..., None]
    m_128 = m_128[:, :, 0].astype(np.float32, copy=False)  # (128,128)
    _MASK_CACHE_128[key] = m_128
    return m_128


def _get_img_128_cached(path: str) -> np.ndarray:
    img = _IMG_CACHE_128.get(path)
    if img is not None:
        return img
    im = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    if im is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    im = cv2.resize(im, (128, 128), interpolation=cv2.INTER_AREA)
    im = (im.astype(np.float32) / 255.0)[..., None]  # (128,128,1)
    img3 = np.repeat(im, 3, axis=-1)  # (128,128,3)
    _IMG_CACHE_128[path] = img3
    return img3


def _precompute_masks_128(df_in: pd.DataFrame) -> np.ndarray:
    n = len(df_in)
    y = np.zeros((n, 128, 128, 3), dtype=np.float32)
    heights = df_in["height"].astype(np.int32).values
    widths = df_in["width"].astype(np.int32).values
    lb = df_in["large_bowel"].values
    sb = df_in["small_bowel"].values
    st = df_in["stomach"].values

    def _worker(i: int):
        h = int(heights[i])
        w = int(widths[i])
        return (
            i,
            _get_mask_128_cached(lb[i], h, w),
            _get_mask_128_cached(sb[i], h, w),
            _get_mask_128_cached(st[i], h, w),
        )

    with ThreadPoolExecutor(max_workers=_CPU_WORKERS) as ex:
        for i, m0, m1, m2 in tqdm(
            ex.map(_worker, range(n)),
            total=n,
            desc="Precomputing train masks (128x128)",
        ):
            y[i, :, :, 0] = m0
            y[i, :, :, 1] = m1
            y[i, :, :, 2] = m2
    return y


def _precompute_images_128(paths: np.ndarray) -> np.ndarray:
    n = len(paths)
    X = np.empty((n, 128, 128, 3), dtype=np.float32)

    def _worker(i: int):
        return i, _get_img_128_cached(paths[i])

    with ThreadPoolExecutor(max_workers=_CPU_WORKERS) as ex:
        for i, img in tqdm(
            ex.map(_worker, range(n)),
            total=n,
            desc="Precomputing images (128x128)",
        ):
            X[i] = img
    return X


gc.collect()



## === cell 8
gc.collect()



## === cell 9
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




## === cell 10
gc.collect()



## === cell 11
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

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss=bce_dice_loss)

gc.collect()



## === cell 12
train_paths = df_train["path"].values
X_train = _precompute_images_128(train_paths)
y_train = _precompute_masks_128(df_train)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(
    buffer_size=len(df_train), seed=42, reshuffle_each_iteration=True
)
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

model.fit(train_ds, epochs=EPOCHS, verbose=1)

del X_train, y_train, train_ds, train_paths
gc.collect()



## === cell 13
df_test = pd.read_csv(
    "../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv"
)
df_test.rename(columns={"class": "class_name"}, inplace=True)

id_split = df_test["id"].str.split("_", expand=True)
df_test["case"] = id_split[0].str.replace("case", "", regex=False).astype(np.int32)
df_test["day"] = id_split[1].str.replace("day", "", regex=False).astype(np.int32)
df_test["slice"] = (id_split[2] + "_" + id_split[3]).astype(str)

TEST_DIR = "../input/uw-madison-gi-tract-image-segmentation/test"

uniq = df_test[["id", "case", "day", "slice"]].drop_duplicates().reset_index(drop=True)
uniq = _resolve_paths_and_meta(uniq, TEST_DIR, desc="test")

uniq = uniq[uniq["path"] != ""].copy()
if uniq.shape[0] == 0:
    raise FileNotFoundError(
        f"No PNGs resolved under {TEST_DIR}. Check directory structure."
    )

df_test = df_test.merge(uniq[["id", "path", "width", "height"]], on="id", how="inner")
if df_test.shape[0] == 0:
    raise RuntimeError("Test merge produced 0 rows; check parsing/join keys.")

df_test = df_test.sort_values(["id", "class_name"]).reset_index(drop=True)

df_test_wide = df_test.pivot_table(
    index=["id", "case", "day", "slice", "width", "height", "path"],
    columns="class_name",
    values="predicted",
    aggfunc="first",
    fill_value="",
).reset_index()

for col in ["large_bowel", "small_bowel", "stomach"]:
    if col not in df_test_wide.columns:
        df_test_wide[col] = ""

df_infer = df_test_wide[
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
df_infer.reset_index(drop=True, inplace=True)
df_infer.fillna("", inplace=True)

del df_test, df_test_wide, uniq, id_split
gc.collect()

df_infer.head(3)



## === cell 14
X_test = _precompute_images_128(df_infer["path"].values)
test_ds = (
    tf.data.Dataset.from_tensor_slices(X_test).batch(32).prefetch(tf.data.AUTOTUNE)
)

gc.collect()
LOGITS = model.predict(test_ds, verbose=1)
gc.collect()

del test_ds, X_test
gc.collect()



## === cell 15
len(LOGITS)



## === cell 16
lbs, sbs, sts = [], [], []
heights = df_infer["height"].astype(np.int32).values
widths = df_infer["width"].astype(np.int32).values

resize = cv2.resize
INTERP = cv2.INTER_NEAREST
thr = 0.5
enc = rle_encode

for index in tqdm(range(df_infer.shape[0]), total=df_infer.shape[0]):
    h = int(heights[index])
    w = int(widths[index])

    pred = LOGITS[index]
    pred0 = resize(pred[:, :, 0], (w, h), interpolation=INTERP)
    pred1 = resize(pred[:, :, 1], (w, h), interpolation=INTERP)
    pred2 = resize(pred[:, :, 2], (w, h), interpolation=INTERP)

    lbs.append(enc((pred0 >= thr).astype("uint8", copy=False)))
    sbs.append(enc((pred1 >= thr).astype("uint8", copy=False)))
    sts.append(enc((pred2 >= thr).astype("uint8", copy=False)))

del LOGITS
gc.collect()



## === cell 17
df_ids = df_infer[["id"]].copy()
gc.collect()



## === cell 18
id_arr = df_ids["id"].values
ids = np.repeat(id_arr, 3)
classes = np.tile(
    np.array(["large_bowel", "small_bowel", "stomach"], dtype=object), len(id_arr)
)
rles = np.empty(len(ids), dtype=object)
rles[0::3] = lbs
rles[1::3] = sbs
rles[2::3] = sts



## === cell 19
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



## === cell 20
submission.head()
