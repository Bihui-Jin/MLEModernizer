# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

# 5. Target score

0.4505897321783137

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.00459) has done: 'I fix the TensorFlow import crash by forcing the pure-Python protobuf implementation early, which resolves the `MessageFactory.GetPrototype` error in many Kaggle images. Then I remove the missing external model dependency (`../input/uwmgi-unet-keras/model.h5`) by replacing it with a tiny, deterministic fallback Keras model that preserves the same I/O semantics (128×128×3 → 128×128×3 sigmoid) so the pipeline runs end-to-end. Finally, I fix the submission-length mismatch by generating predictions for the full `sample_submission` rows (not every 3rd row) and by writing RLEs aligned exactly to the sample ordering, producing a valid `submission.csv`.'
- What this solution (achieved 0.00378) has done: 'I fix the TensorFlow/protobuf crash by importing TensorFlow only after forcing the pure-Python protobuf implementation and by defensively retrying the import with a clean protobuf state. Then, to increase score toward your target (your current 0.00459 is far below 0.4506), I minimally change the fallback behavior so the model is trained on the provided `train.csv` masks (same 128×128×3 → 128×128×3 sigmoid semantics, same loss) and then used for test inference, instead of using random untrained weights. Finally, I keep submission generation aligned exactly to `sample_submission.csv` and ensure RLE encoding is correct and stable.'

# 9. Code solution

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



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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


def _build_global_scan_index(root_dir: str):
    idx = {}  # (case_int, day_int) -> (scans_dir, {slice_idx: (filename, w, h)})
    if not os.path.isdir(root_dir):
        return idx
    for case_name in os.listdir(root_dir):
        if not case_name.startswith("case"):
            continue
        case_path = os.path.join(root_dir, case_name)
        if not os.path.isdir(case_path):
            continue
        try:
            case_int = int(case_name.replace("case", ""))
        except Exception:
            continue

        for day_name in os.listdir(case_path):
            if not day_name.startswith(f"{case_name}_day"):
                continue
            try:
                day_int = int(day_name.split("_day", 1)[1])
            except Exception:
                continue

            scans_dir = os.path.join(case_path, day_name, "scans")
            if not os.path.isdir(scans_dir):
                continue

            m = {}
            for fn in os.listdir(scans_dir):
                if (not fn.startswith("slice_")) or (not fn.endswith(".png")):
                    continue
                stem = fn[:-4]
                p = stem.split("_")
                if len(p) < 4:
                    continue
                slice_idx = p[1]
                try:
                    w = int(p[2])
                    h = int(p[3])
                except Exception:
                    continue
                m[slice_idx] = (fn, w, h)
            idx[(case_int, day_int)] = (scans_dir, m)
    return idx


_GLOBAL_SCAN_INDEX = _build_global_scan_index(TRAIN_DIR)


def _get_path_w_h(case_int: int, day_int: int, slice_png: str):
    slice_idx = slice_png[:-4]
    key = (int(case_int), int(day_int))
    v = _GLOBAL_SCAN_INDEX.get(key)
    if v is None:
        return "", 0, 0
    scans_dir, m = v
    hit = m.get(slice_idx)
    if hit is None:
        return "", 0, 0
    fn, w, h = hit
    return os.path.join(scans_dir, fn), w, h


case_arr = df["case"].to_numpy(np.int32, copy=False)
day_arr = df["day"].to_numpy(np.int32, copy=False)
slice_arr = df["slice"].to_numpy(copy=False)

paths = np.empty(len(df), dtype=object)
widths = np.empty(len(df), dtype=np.int32)
heights = np.empty(len(df), dtype=np.int32)

for i in range(len(df)):
    p, w, h = _get_path_w_h(case_arr[i], day_arr[i], slice_arr[i])
    paths[i] = p
    widths[i] = w
    heights[i] = h

df["path"] = paths
df["width"] = widths
df["height"] = heights

del parts, case_arr, day_arr, slice_arr, paths, widths, heights
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

        self._paths = self.df["path"].to_numpy(copy=False)
        self._widths = self.df["width"].to_numpy(np.int32, copy=False)
        self._heights = self.df["height"].to_numpy(np.int32, copy=False)

        if subset == "train":
            self._rle_large = self.df["large_bowel"].to_numpy(copy=False)
            self._rle_small = self.df["small_bowel"].to_numpy(copy=False)
            self._rle_stom = self.df["stomach"].to_numpy(copy=False)

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

        idxs = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        for i, ridx in enumerate(idxs):
            img_path = self._paths[ridx]
            w = int(self._widths[ridx])
            h = int(self._heights[ridx])

            img = self.__load_grayscale(img_path)  # (128,128,3)
            X[i] = img

            if self.subset == "train":
                for k, rles in enumerate(
                    (self._rle_large[ridx], self._rle_small[ridx], self._rle_stom[ridx])
                ):
                    masks = rle_decode(rles, shape=(h, w, 1))  # always (h,w,1)
                    masks = cv2.resize(
                        masks, (128, 128), interpolation=cv2.INTER_NEAREST
                    )
                    y[i, :, :, k] = masks[:, :, 0]

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
                    img *= 1.0 / mx

        img3 = np.empty((128, 128, 3), dtype=np.float32)
        img3[:, :, 0] = img
        img3[:, :, 1] = img
        img3[:, :, 2] = img

        if self._cache is not None:
            self._cache[img_path] = img3
        return img3




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

    _TRAIN_GLOBAL_SCAN_INDEX = _build_global_scan_index(TRAIN_IMG_DIR)

    def _get_path_w_h_train(case_int: int, day_int: int, slice_png: str):
        slice_idx = slice_png[:-4]
        key = (int(case_int), int(day_int))
        v = _TRAIN_GLOBAL_SCAN_INDEX.get(key)
        if v is None:
            return "", 0, 0
        scans_dir, m = v
        hit = m.get(slice_idx)
        if hit is None:
            return "", 0, 0
        fn, w, h = hit
        return os.path.join(scans_dir, fn), w, h

    case_arr = train_df["case"].to_numpy(np.int32, copy=False)
    day_arr = train_df["day"].to_numpy(np.int32, copy=False)
    slice_arr = train_df["slice"].to_numpy(copy=False)

    paths = np.empty(len(train_df), dtype=object)
    widths = np.empty(len(train_df), dtype=np.int32)
    heights = np.empty(len(train_df), dtype=np.int32)
    for i in range(len(train_df)):
        p, w, h = _get_path_w_h_train(case_arr[i], day_arr[i], slice_arr[i])
        paths[i] = p
        widths[i] = w
        heights[i] = h

    train_df["path"] = paths
    train_df["width"] = widths
    train_df["height"] = heights
    del paths, widths, heights, parts, case_arr, day_arr, slice_arr

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



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
error                                     Traceback (most recent call last)
/tmp/ipykernel_11/4086858345.py in <cell line: 0>()
     90     validation_steps = max(1, len(val_gen))
     91 
---> 92     model.fit(
     93         trn_gen,
     94         validation_data=val_gen,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_11/1538948734.py in __getitem__(self, index)
     50                 ):
     51                     masks = rle_decode(rles, shape=(h, w, 1))  # always (h,w,1)
---> 52                     masks = cv2.resize(
     53                         masks, (128, 128), interpolation=cv2.INTER_NEAREST
     54                     )

error: OpenCV(4.12.0) /io/opencv/modules/imgproc/src/resize.cpp:4208: error: (-215:Assertion failed) !ssize.empty() in function 'resize'


## === cell 14
pred_batches = DataGenerator(df_test, batch_size=16, subset="test", shuffle=False)
gc.collect()

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
rles_append = rles.append
for i in tqdm(range(df_test.shape[0]), total=df_test.shape[0]):
    h = int(heights[i])
    w = int(widths[i])
    ch = class_to_ch.get(classes[i], 0)

    pred_resized = cv2.resize(
        PROBS[i, :, :, ch], (w, h), interpolation=cv2.INTER_NEAREST
    )
    pred_bin = (pred_resized >= 0.5).astype(np.uint8, copy=False)
    rles_append(rle_encode(pred_bin))

gc.collect()



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
error                                     Traceback (most recent call last)
/tmp/ipykernel_11/344862983.py in <cell line: 0>()
     13     ch = class_to_ch.get(classes[i], 0)
     14 
---> 15     pred_resized = cv2.resize(
     16         PROBS[i, :, :, ch], (w, h), interpolation=cv2.INTER_NEAREST
     17     )

error: OpenCV(4.12.0) /io/opencv/modules/imgproc/src/resize.cpp:4211: error: (-215:Assertion failed) inv_scale_x > 0 in function 'resize'


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



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2045423720.py in <cell line: 0>()
----> 1 sub = pd.DataFrame(
      2     {
      3         "id": df_test["id"].values,
      4         "class": df_test["class_name"].values,
      5         "predicted": rles,

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __init__(self, data, index, columns, dtype, copy)
    776         elif isinstance(data, dict):
    777             # GH#38939 de facto copy defaults to False only in non-dict cases
--> 778             mgr = dict_to_mgr(data, index, columns, dtype=dtype, copy=copy, typ=manager)
    779         elif isinstance(data, ma.MaskedArray):
    780             from numpy.ma import mrecords

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in dict_to_mgr(data, index, columns, dtype, typ, copy)
    501             arrays = [x.copy() if hasattr(x, "dtype") else x for x in arrays]
    502 
--> 503     return arrays_to_mgr(arrays, columns, index, dtype=dtype, typ=typ, consolidate=copy)
    504 
    505 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in arrays_to_mgr(arrays, columns, index, dtype, verify_integrity, typ, consolidate)
    112         # figure out the index, if necessary
    113         if index is None:
--> 114             index = _extract_index(arrays)
    115         else:
    116             index = ensure_index(index)

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in _extract_index(data)
    675         lengths = list(set(raw_lengths))
    676         if len(lengths) > 1:
--> 677             raise ValueError("All arrays must be of the same length")
    678 
    679         if have_dicts:

ValueError: All arrays must be of the same length

## === cell 18
sub.head()



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3518946013.py in <cell line: 0>()
----> 1 sub.head()
      2 

NameError: name 'sub' is not defined

## === cell 19
print("Wrote submission.csv with shape:", sub.shape)
print("Null predicted:", sub["predicted"].isna().sum())
print("Empty predicted:", (sub["predicted"] == "").sum())

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1996318216.py in <cell line: 0>()
----> 1 print("Wrote submission.csv with shape:", sub.shape)
      2 print("Null predicted:", sub["predicted"].isna().sum())
      3 print("Empty predicted:", (sub["predicted"] == "").sum())

NameError: name 'sub' is not defined
