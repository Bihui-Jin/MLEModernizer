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

0.7877486177900668

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the environment-breaking import issue in the first cell by removing legacy standalone `keras` imports and using `tensorflow.keras` consistently, which also resolves the `pd`/`tf` `NameError`s caused by the first cell failing early. I update model loading to work with Keras 3 by switching from `load_model()` on a SavedModel directory to `keras.layers.TFSMLayer`, preserving the same inference semantics. I also harden RLE encode/decode and the data generator to handle empty masks and the final incomplete batch, preventing runtime errors and ensuring a complete `submission.csv` is written with the exact required columns. These changes are correctness/stability focused and should yield a valid submission file end-to-end.'
- What this solution (achieved 0.0) has done: 'I fix the environment-breaking protobuf/TensorFlow import error by forcing the pure-Python protobuf implementation before importing TensorFlow, which prevents the `MessageFactory` AttributeError. Then I fix the missing model issue by dynamically locating the SavedModel directory inside `../input/` (falling back to an all-empty-mask submission if no model is present), so `model` is always defined and the notebook completes. Finally, I keep the same inference + thresholding + RLE logic, but make the grayscale normalization correct for 16-bit PNGs to avoid near-all-ones inputs that collapse predictions and yield a 0.0 score, improving toward the target while staying within the original semantics.'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow/protobuf crash that stops the very first import by forcing a compatible protobuf implementation and (if needed) downgrading protobuf in-notebook before importing TensorFlow. Then I make sure the inference model loads robustly in Kaggle by discovering the SavedModel directory and wrapping it with `keras.layers.TFSMLayer`, keeping the same prediction/threshold/RLE semantics. Finally, I harden the test-loop to avoid accidental skipping/duplication issues and guarantee the produced `submission.csv` exactly matches `sample_submission.csv`’s row ordering and required columns, which should lift the score from the current 0.0 (broken pipeline) toward the target.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

try:
    import tensorflow as tf  # noqa: F401
except Exception as e:
    msg = str(e)
    if (
        ("MessageFactory" in msg)
        or ("GetPrototype" in msg)
        or ("protobuf" in msg.lower())
    ):
        import sys
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf<4"]
        )
        for m in list(sys.modules.keys()):
            if m.startswith("google.protobuf") or m.startswith("tensorflow"):
                sys.modules.pop(m, None)
        import tensorflow as tf  # noqa: F401
    else:
        raise

from glob import glob
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import cv2
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)




## === cell 2
def rle_decode(mask_rle, shape, color=1):
    """
    Decode RLE into a mask of `shape`.
    Supports empty strings (returns all zeros).
    """
    if (
        mask_rle is None
        or (isinstance(mask_rle, float) and np.isnan(mask_rle))
        or str(mask_rle).strip() == ""
    ):
        return np.zeros(shape, dtype=np.float32)

    s = np.array(str(mask_rle).split(), dtype=int)
    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths

    if len(shape) == 3:
        h, w, d = shape
        img = np.zeros((h * w, d), dtype=np.float32)
    else:
        h, w = shape
        img = np.zeros((h * w,), dtype=np.float32)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = color

    return img.reshape(shape)


def rle_encode(arr):
    """
    Encode a binary mask to RLE.
    Handles empty/all-zero masks safely.
    """
    if arr is None:
        return ""
    arr = np.asarray(arr)
    if arr.size == 0:
        return ""
    pixels = arr.reshape(-1).astype(np.uint8)

    if pixels.max() == 0:
        return ""

    pixels = np.concatenate([[0], pixels, [0]])
    changes = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs = changes[::2]
    ends = changes[1::2]
    lengths = ends - runs
    rle = np.column_stack((runs, lengths)).reshape(-1)
    return " ".join(map(str, rle))


def open_gray16(_path, normalize=True, to_rgb=False):
    """Helper to open 16-bit grayscale PNGs."""
    img = cv2.imread(_path, cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {_path}")
    if normalize:
        denom = 65535.0 if img.dtype == np.uint16 else 255.0
        img = img.astype(np.float32) / denom
    if to_rgb:
        img = np.tile(np.expand_dims(img, axis=-1), 3)
    return img




## === cell 3
def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def mean_iou(y_true, y_pred):
    yt0 = y_true[:, :, :, 0]
    yp0 = K.cast(y_pred[:, :, :, 0] > 0.5, "float32")
    inter = tf.math.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 1)))
    union = tf.math.count_nonzero(tf.add(yt0, yp0))
    iou = tf.where(tf.equal(union, 0), 1.0, tf.cast(inter / union, "float32"))
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




## === cell 4
def find_savedmodel_dir(search_root="../input", max_depth=7):
    pattern = os.path.join(search_root, *["*"] * max_depth, "saved_model.pb")
    candidates = glob(pattern)
    if not candidates:
        pattern2 = os.path.join(search_root, *["*"] * max_depth, "saved_model.pbtxt")
        candidates = glob(pattern2)
    if not candidates:
        return None
    candidates = sorted(
        set(os.path.dirname(p) for p in candidates), key=lambda x: (len(x), x)
    )
    return candidates[0]


MODEL_DIR = "../input/unet-model/model1-20220702T171713Z-001/model1"
if not (
    os.path.isdir(MODEL_DIR)
    and (
        os.path.exists(os.path.join(MODEL_DIR, "saved_model.pb"))
        or os.path.exists(os.path.join(MODEL_DIR, "saved_model.pbtxt"))
    )
):
    discovered = find_savedmodel_dir("../input", max_depth=8)
    if discovered is not None:
        MODEL_DIR = discovered
    else:
        MODEL_DIR = None


def build_infer_model_from_savedmodel(model_dir):
    try:
        sigs = tf.saved_model.load(model_dir).signatures
        endpoint_name = (
            "serving_default" if "serving_default" in sigs else list(sigs.keys())[0]
        )
    except Exception:
        endpoint_name = "serving_default"

    layer = keras.layers.TFSMLayer(model_dir, call_endpoint=endpoint_name)
    inp = keras.Input(shape=(128, 128, 3), name="input")
    out = layer(inp)
    if isinstance(out, dict):
        out = out[list(out.keys())[0]]
    return keras.Model(inputs=inp, outputs=out, name="unet_infer")


def build_dummy_zero_model():
    inp = keras.Input(shape=(128, 128, 3), name="input")
    out = keras.layers.Lambda(
        lambda x: tf.zeros((tf.shape(x)[0], 128, 128, 3), dtype=tf.float32)
    )(inp)
    return keras.Model(inp, out, name="dummy_zero_model")


try:
    if MODEL_DIR is None:
        raise FileNotFoundError("No SavedModel found under ../input")
    model = build_infer_model_from_savedmodel(MODEL_DIR)
except Exception as e:
    warnings.warn(
        f"Could not load SavedModel for inference ({e}). Using dummy zero model."
    )
    model = build_dummy_zero_model()



## === cell 5
df1 = pd.read_csv("../input/uw-madison-gi-tract-image-segmentation/train.csv")
submission_template = pd.read_csv(
    "../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv"
)

DEBUG = False




## === cell 6
def preprocessing(df, subset="train"):
    df = df.copy()
    df["case"] = df["id"].apply(lambda x: int(x.split("_")[0].replace("case", "")))
    df["day"] = df["id"].apply(lambda x: int(x.split("_")[1].replace("day", "")))
    df["slice"] = df["id"].apply(lambda x: x.split("_")[3])

    if (subset == "train") or (DEBUG):
        DIR = "../input/uw-madison-gi-tract-image-segmentation/train"
    else:
        DIR = "../input/uw-madison-gi-tract-image-segmentation/test"

    all_images = glob(os.path.join(DIR, "**", "*.png"), recursive=True)
    if len(all_images) == 0:
        raise FileNotFoundError(f"No PNG images found under {DIR}")

    x = all_images[0].rsplit("/", 4)[0]

    path_partial_list = []
    for i in range(df.shape[0]):
        path_partial_list.append(
            os.path.join(
                x,
                "case" + str(df["case"].values[i]),
                "case"
                + str(df["case"].values[i])
                + "_"
                + "day"
                + str(df["day"].values[i]),
                "scans",
                "slice_" + str(df["slice"].values[i]),
            )
        )
    df["path_partial"] = path_partial_list

    partials = [str(p.rsplit("_", 4)[0]) for p in all_images]
    tmp_df = pd.DataFrame({"path_partial": partials, "path": all_images})

    df = df.merge(tmp_df, on="path_partial", how="left").drop(columns=["path_partial"])
    if df["path"].isna().any():
        missing = df[df["path"].isna()][["id", "case", "day", "slice"]].head(5)
        raise RuntimeError(
            f"Failed to match some ids to image paths. Example missing rows:\n{missing}"
        )

    df["filename"] = df["path"].apply(lambda x: x.split("/")[-1])
    df["unique_filename"] = df.apply(
        lambda row: f"{row.case}_{row.day}_{row.filename}", axis=1
    )

    df["width"] = df["path"].apply(lambda x: int(x[:-4].rsplit("_", 4)[1]))
    df["height"] = df["path"].apply(lambda x: int(x[:-4].rsplit("_", 4)[2]))
    df["px_spacing_h"] = df["path"].apply(lambda x: float(x[:-4].rsplit("_", 4)[3]))
    df["px_spacing_w"] = df["path"].apply(lambda x: float(x[:-4].rsplit("_", 4)[4]))
    return df




## === cell 7
test_df = preprocessing(submission_template[["id", "class"]].copy(), subset="test")
train_df = preprocessing(df1, subset="train")




## === cell 8
def segment(df, subset="train"):
    df_out = pd.DataFrame({"id": df["id"][::3]})

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
    df_out["px_spacing_h"] = df["px_spacing_h"][::3].values
    df_out["px_spacing_w"] = df["px_spacing_w"][::3].values
    df_out["filename"] = df["filename"][::3].values
    df_out["unique_filename"] = df["unique_filename"][::3].values

    df_out.reset_index(inplace=True, drop=True)
    df_out.fillna("", inplace=True)
    if subset == "train":
        df_out["count"] = np.sum(df_out.iloc[:, 1:4] != "", axis=1).values
    return df_out


train_df = segment(train_df, subset="train")
test_seg_df = segment(test_df, subset="test")



## === cell 9
BATCH_SIZE = 32
im_height = 128
im_width = 128




## === cell 10
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size=BATCH_SIZE, subset="train", shuffle=False):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.shuffle = shuffle
        self.subset = subset
        self.batch_size = batch_size
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

        X = np.empty((bs, im_height, im_width, 3), dtype=np.float32)
        if self.subset == "train":
            y = np.empty((bs, im_height, im_width, 3), dtype=np.float32)

        ids = list(self.df["id"].iloc[batch_indexes])

        for i, img_path in enumerate(self.df["path"].iloc[batch_indexes]):
            w = int(self.df["width"].iloc[batch_indexes[i]])
            h = int(self.df["height"].iloc[batch_indexes[i]])

            img = self.__load_grayscale(img_path)  # (128,128,1)
            X[i] = img  # broadcast to (128,128,3)

            if self.subset == "train":
                for k, j in enumerate(["large_bowel", "small_bowel", "stomach"]):
                    rles = self.df[j].iloc[batch_indexes[i]]
                    mask = rle_decode(rles, shape=(h, w, 1))
                    mask = cv2.resize(
                        mask, (im_width, im_height), interpolation=cv2.INTER_NEAREST
                    )
                    y[i, :, :, k] = mask[:, :, 0] if mask.ndim == 3 else mask

        if self.subset == "train":
            return X, y
        else:
            return X, ids

    def __load_grayscale(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img = cv2.resize(img, (im_width, im_height), interpolation=cv2.INTER_AREA)
        denom = 65535.0 if img.dtype == np.uint16 else 255.0
        img = img.astype(np.float32) / denom
        img = np.expand_dims(img, axis=-1)
        return img




## === cell 11
val_generator = DataGenerator(
    test_seg_df, batch_size=BATCH_SIZE, subset="test", shuffle=False
)



## === cell 12
id_to_meta = test_seg_df.set_index("id")[["width", "height"]]

pred_map = {}  # (id, class) -> rle

num_batches = len(val_generator)
for i in range(num_batches):
    X, id_batch = val_generator[i]
    preds = model.predict(X, verbose=0)

    preds = np.asarray(preds)
    if preds.ndim == 3:
        preds = np.expand_dims(preds, axis=-1)
    if preds.shape[-1] != 3:
        raise RuntimeError(
            f"Unexpected model output shape: {preds.shape}, expected last dim=3"
        )

    for j, _id in enumerate(id_batch):
        if _id not in id_to_meta.index:
            continue

        w = int(id_to_meta.loc[_id, "width"])
        h = int(id_to_meta.loc[_id, "height"])

        for k, cls in enumerate(("large_bowel", "small_bowel", "stomach")):
            pred_img = cv2.resize(
                preds[j, :, :, k], (w, h), interpolation=cv2.INTER_NEAREST
            )
            pred_img = (pred_img > 0.5).astype("uint8")
            pred_map[(_id, cls)] = rle_encode(pred_img)

submission_df = submission_template.copy()
submission_df["predicted"] = [
    pred_map.get((row_id, row_cls), "")
    for row_id, row_cls in zip(
        submission_df["id"].values, submission_df["class"].values
    )
]

submission_df.to_csv("submission.csv", index=False)
print(submission_df.shape)
print(submission_df.head())
print("Wrote submission.csv")
