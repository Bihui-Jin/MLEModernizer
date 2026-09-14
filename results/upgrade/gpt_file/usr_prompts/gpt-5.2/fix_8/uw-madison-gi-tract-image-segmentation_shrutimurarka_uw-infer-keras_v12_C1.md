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

0.8099985307671078

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the environment-breaking import issue by removing the mixed standalone `keras` imports and using only `tensorflow.keras`, which avoids the protobuf `MessageFactory.GetPrototype` crash. I also make the model-loading robust: if the external pretrained model file is missing (as in your run), the script fall back to a valid all-empty-mask submission so it always produces `submission.csv` end-to-end. Next, I fix missing imports/undefined names (`pd`, `tf`, `K`, `DEBUG`) and correct a couple of logic bugs that would break execution (bad mislabeled-data filter using `|` instead of `&`, unsafe RLE encoder for empty masks, and generator shape broadcasting). Finally, I keep the inference/core pipeline the same when the model exists, producing properly ordered `id,class,predicted` rows aligned to `sample_submission.csv`.'
- What this solution (achieved 0.0) has done: 'I fix the environment-breaking protobuf/TensorFlow import crash by forcing the pure-Python protobuf implementation *before* importing TensorFlow (this is the root cause of the `MessageFactory.GetPrototype` error in Kaggle for some TF/protobuf combinations). I keep the model/inference logic unchanged, but make the image normalization consistent (16-bit PNGs should be scaled by 65535, not 255) to avoid systematically wrong inputs that can drive the score to ~0 even when the model loads. I also make the path-root detection OS-agnostic (use `os.path` instead of hardcoded `/`) to prevent silent path mapping failures across environments. The script still always write a valid `submission.csv` with the correct columns and row order.'
- What this solution (achieved 0.0) has done: 'We fix the TensorFlow/protobuf crash that prevents the notebook from running by forcing a compatible protobuf runtime before importing TensorFlow and by switching to the pure-TF Keras API consistently. Then we make the model path lookup robust (search common Kaggle input locations) so the pretrained `.h5` is actually found when present, which is necessary to move the score from 0.0 toward the target (empty-mask fallback guarantees 0.0). Finally, we keep the inference pipeline the same but ensure the submission rows exactly match `sample_submission.csv` order and that RLE encoding is correct and stable for both empty and non-empty masks.'
- What this solution (achieved 0.0) has done: 'You’re crashing at TensorFlow import due to the known TF/protobuf binary incompatibility in some Kaggle images; the minimal reliable fix is to force the pure-Python protobuf backend before any TensorFlow import and also disable the C++ implementation explicitly. Next, your image-path root detection can silently pick the wrong root (because it walks up fixed parent counts), so I make it deterministic by using the known TRAIN_DIR/TEST_DIR as the root directly, which prevents missing-path merges and ensures inference runs. Finally, to move the score off 0.0 toward your target, I keep the same model/inference logic but make the model-path resolver robust enough to actually find the pretrained `.h5` in common Kaggle input locations; if it still isn’t present, the script still produce a valid empty-mask `submission.csv`.'
- What this solution (achieved 0.0) has done: 'We fix the TensorFlow/protobuf import crash causing the run to stop in cell 1 by forcing the pure-Python protobuf backend early and (critically) ensuring TensorFlow doesn’t use the C++ protobuf implementation. Then we add a small, safe fallback so if TensorFlow still fails to import in this environment, the pipeline still generate a valid `submission.csv` (so you never get a runtime 0.0 due to no file). Finally, we keep the model/inference logic unchanged when TensorFlow loads and the `.h5` exists, so the score can move up from 0.0 toward the target when the pretrained model is available.'
- What this solution (achieved 0.0) has done: 'We fix the TensorFlow/protobuf import crash by setting the additional environment variable `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` *before* any TensorFlow import (your current variables weren’t sufficient for this Kaggle image). Then we keep the model/inference pipeline unchanged, but make model-path discovery a bit more robust by also searching for any `.h5` if the exact basename isn’t found (this is necessary to move the score off 0.0, since the empty-mask fallback guarantees 0.0). Finally, we ensure the submission is written in exactly the `sample_submission.csv` row order with correct columns, so Kaggle accepts it.'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow/protobuf import crash that currently stops the notebook in cell 1 by forcing the pure-Python protobuf backend before any TensorFlow-related import and by avoiding problematic protobuf version pinning env vars that trigger the `MessageFactory.GetPrototype` failure in Kaggle. Then I keep your core inference pipeline unchanged, but make it robust so it can still create a valid `submission.csv` even if TensorFlow or the model cannot be loaded. Finally, to move the score up from 0.0 toward the target, I keep your model discovery logic but ensure TensorFlow import succeeds in the Kaggle image so the pretrained `.h5` can actually be used when present.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_USE_C"] = "0"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import random
import warnings
from glob import glob

import cv2
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)



## === cell 1
TF_AVAILABLE = True
try:
    import tensorflow as tf
    from tensorflow.keras import backend as K
    from tensorflow.keras.losses import binary_crossentropy

    tf.random.set_seed(SEED)
except Exception as e:
    TF_AVAILABLE = False
    tf = None
    K = None
    binary_crossentropy = None
    print("WARNING: TensorFlow failed to import; will write empty submission.csv.")
    print(f"TensorFlow import error: {type(e).__name__}: {e}")



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
DATA_DIR = "../input/uw-madison-gi-tract-image-segmentation"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

MODEL_PATH = "../input/unet-model/model_v2.h5"




## === cell 3
def rle_decode(mask_rle, shape, color=1):
    """
    Decode run-length encoding into mask.
    shape can be (h, w) or (h, w, d).
    """
    if (
        mask_rle is None
        or mask_rle == ""
        or (isinstance(mask_rle, float) and np.isnan(mask_rle))
    ):
        return np.zeros(shape, dtype=np.float32)

    s = np.array(mask_rle.split(), dtype=int)
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
    Encode binary mask to RLE, Kaggle-style.
    Handles empty/all-zero masks safely.
    Note: GI tract competition expects Fortran-order flattening.
    """
    if arr is None:
        return ""
    arr = np.asarray(arr)
    if arr.size == 0:
        return ""
    arr = (arr > 0).astype(np.uint8)

    if arr.sum() == 0:
        return ""

    pixels = arr.T.reshape(-1)  # flatten(order='F') for 2D
    pixels = np.concatenate([[0], pixels, [0]])
    changes = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs = changes.copy()
    runs[1::2] = runs[1::2] - runs[::2]
    return " ".join(str(x) for x in runs)


def open_gray16(_path, normalize=True, to_rgb=False):
    img = cv2.imread(_path, cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {_path}")
    if normalize:
        img = img.astype(np.float32) / 65535.0
    if to_rgb:
        img = np.tile(img[..., None], (1, 1, 3))
    return img




## === cell 4
if TF_AVAILABLE:

    def iou_coef(y_true, y_pred, smooth=1):
        intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
        union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
        iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
        return iou

    def dice_coef(y_true, y_pred, smooth=1):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2.0 * intersection + smooth) / (
            K.sum(y_true_f) + K.sum(y_pred_f) + smooth
        )

    def mean_iou(y_true, y_pred):
        yt0 = y_true[:, :, :, 0]
        yp0 = K.cast(y_pred[:, :, :, 0] > 0.5, "float32")
        inter = tf.math.count_nonzero(
            tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 1))
        )
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
        return binary_crossentropy(
            tf.cast(y_true, tf.float32), y_pred
        ) + 0.5 * dice_loss(tf.cast(y_true, tf.float32), y_pred)




## === cell 5
model = None
MODEL_AVAILABLE = False

if TF_AVAILABLE:
    from tensorflow import keras

    class FixedDropout(keras.layers.Dropout):
        def _get_noise_shape(self, inputs):
            if self.noise_shape is None:
                return self.noise_shape
            symbolic_shape = K.shape(inputs)
            noise_shape = [
                symbolic_shape[axis] if shape is None else shape
                for axis, shape in enumerate(self.noise_shape)
            ]
            return tuple(noise_shape)

    custom_objects = {
        "FixedDropout": FixedDropout,
        "dice_coef": dice_coef,
        "iou_coef": iou_coef,
        "bce_dice_loss": bce_dice_loss,
    }

    def _resolve_model_path(initial_path: str) -> str:
        """
        Robust model discovery in Kaggle:
        - keep initial_path if it exists
        - else search ../input and /kaggle/input recursively for the exact basename
        - else (last resort) search for any .h5 and prefer ones containing 'model'/'unet'
        """
        if initial_path and os.path.exists(initial_path):
            return initial_path

        base_name = os.path.basename(initial_path) if initial_path else "model_v2.h5"
        roots = ["../input", "/kaggle/input", "/kaggle/data/input"]

        candidates = []
        for root in roots:
            candidates.extend(glob(os.path.join(root, "**", base_name), recursive=True))
        for p in candidates:
            if os.path.basename(p) == base_name and os.path.exists(p):
                return p

        h5s = []
        for root in roots:
            h5s.extend(glob(os.path.join(root, "**", "*.h5"), recursive=True))
        h5s = [p for p in h5s if os.path.exists(p)]
        if h5s:

            def score_path(p):
                bn = os.path.basename(p).lower()
                sc = 0
                if "unet" in bn:
                    sc += 2
                if "model" in bn:
                    sc += 1
                if "v2" in bn:
                    sc += 1
                return (sc, -len(p))

            h5s_sorted = sorted(h5s, key=score_path, reverse=True)
            return h5s_sorted[0]

        return initial_path

    MODEL_PATH = _resolve_model_path(MODEL_PATH)

    if MODEL_PATH and os.path.exists(MODEL_PATH):
        model = keras.models.load_model(
            MODEL_PATH, custom_objects=custom_objects, compile=False
        )
        MODEL_AVAILABLE = True

print(
    f"TF_AVAILABLE={TF_AVAILABLE}, MODEL_AVAILABLE={MODEL_AVAILABLE}, MODEL_PATH='{MODEL_PATH}'"
)



## === cell 6
df1 = pd.read_csv(TRAIN_CSV)
print(df1.head())



## === cell 7
test_df = pd.read_csv(SAMPLE_SUB)

DEBUG = False
if len(test_df) == 0:
    DEBUG = True
    test_df = df1.iloc[:300, :][["id", "class"]].copy()
    test_df["predicted"] = ""
else:
    DEBUG = False

submission1 = test_df.copy()
print(test_df.head())




## === cell 8
def preprocessing(df, subset="train"):
    df = df.copy()
    df["case"] = df["id"].apply(lambda x: int(x.split("_")[0].replace("case", "")))
    df["day"] = df["id"].apply(lambda x: int(x.split("_")[1].replace("day", "")))
    df["slice"] = df["id"].apply(lambda x: x.split("_")[3])

    if (subset == "train") or (DEBUG):
        DIR = TRAIN_DIR
    else:
        DIR = TEST_DIR

    all_images = glob(os.path.join(DIR, "**", "*.png"), recursive=True)
    if len(all_images) == 0:
        raise FileNotFoundError(f"No PNG images found under {DIR}")

    root = DIR

    path_partial_list = []
    for i in range(0, df.shape[0]):
        path_partial_list.append(
            os.path.join(
                root,
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
        missing = df[df["path"].isna()].head(5)[["id", "case", "day", "slice"]]
        raise RuntimeError(
            f"Failed to map some ids to image paths. Examples:\n{missing}"
        )

    df["filename"] = df["path"].apply(lambda x: os.path.basename(x))
    df["unique_filename"] = df.apply(
        lambda row: str(row.case) + "_" + str(row.day) + "_" + str(row.filename), axis=1
    )

    df["width"] = df["path"].apply(lambda x: int(x[:-4].rsplit("_", 4)[1]))
    df["height"] = df["path"].apply(lambda x: int(x[:-4].rsplit("_", 4)[2]))
    df["px_spacing_h"] = df["path"].apply(lambda x: float(x[:-4].rsplit("_", 4)[3]))
    df["px_spacing_w"] = df["path"].apply(lambda x: float(x[:-4].rsplit("_", 4)[4]))

    return df




## === cell 9
test_df = preprocessing(test_df, subset="test")
print(test_df.shape)
test_df.head()



## === cell 10
train_df = preprocessing(df1, subset="train")
train_df.head()



## === cell 11
train_df = train_df[(train_df["case"] != 7) & (train_df["case"] != 0)].reset_index(
    drop=True
)
train_df = train_df[(train_df["case"] != 81) & (train_df["case"] != 30)].reset_index(
    drop=True
)




## === cell 12
def segment(df, subset="train"):
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



## === cell 13
BATCH_SIZE = 32
im_height = 128
im_width = 128



## === cell 14
if TF_AVAILABLE:

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
            current_bs = len(batch_indexes)

            X = np.empty((current_bs, im_height, im_width, 3), dtype=np.float32)

            ids = list(self.df["id"].iloc[batch_indexes])

            if self.subset == "train":
                y = np.empty((current_bs, im_height, im_width, 3), dtype=np.float32)

            for i, df_idx in enumerate(batch_indexes):
                img_path = self.df["path"].iloc[df_idx]
                w = int(self.df["width"].iloc[df_idx])
                h = int(self.df["height"].iloc[df_idx])

                img = self.__load_grayscale(img_path)  # (128,128,1)
                X[i] = np.tile(img, (1, 1, 3))

                if self.subset == "train":
                    for k, j in enumerate(["large_bowel", "small_bowel", "stomach"]):
                        rles = self.df[j].iloc[df_idx]
                        mask = rle_decode(rles, shape=(h, w, 1))
                        mask = cv2.resize(
                            mask, (im_width, im_height), interpolation=cv2.INTER_NEAREST
                        )
                        if mask.ndim == 2:
                            mask = mask[..., None]
                        y[i, :, :, k] = mask[:, :, 0]

            if self.subset == "train":
                return X, y
            else:
                return X, ids

        def __load_grayscale(self, img_path):
            img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
            if img is None:
                raise FileNotFoundError(f"Could not read image: {img_path}")
            img = cv2.resize(img, (im_width, im_height), interpolation=cv2.INTER_AREA)

            if img.dtype == np.uint16:
                img = img.astype(np.float32) / 65535.0
            else:
                img = img.astype(np.float32)
                mx = float(np.max(img)) if img.size else 1.0
                if mx > 0:
                    img /= mx

            img = np.expand_dims(img, axis=-1)
            return img




## === cell 15
if TF_AVAILABLE:
    val_generator = DataGenerator(
        test_seg_df, batch_size=BATCH_SIZE, subset="test", shuffle=False
    )
    print(len(val_generator), BATCH_SIZE)
else:
    val_generator = None



## === cell 16
if (not TF_AVAILABLE) or (not MODEL_AVAILABLE):
    submission_df = submission1.copy()
    submission_df["predicted"] = ""
    submission_df.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with empty predictions (TF/model unavailable).")
else:
    ids = []
    classs = []
    predics = []

    num_batches = len(val_generator)
    for i in range(num_batches):
        X, batch_ids = val_generator[i]
        preds = model.predict(X, verbose=0)  # (bs,128,128,3)

        for j in range(len(batch_ids)):
            if batch_ids[j] in ids:
                continue

            row = test_seg_df.loc[test_seg_df["id"] == batch_ids[j]].iloc[0]
            w = int(row["width"])
            h = int(row["height"])

            ids.extend((batch_ids[j], batch_ids[j], batch_ids[j]))
            classs.extend(("large_bowel", "small_bowel", "stomach"))

            for k in range(3):
                pred_img = cv2.resize(
                    preds[j, :, :, k], (w, h), interpolation=cv2.INTER_NEAREST
                )
                pred_img = (pred_img > 0.5).astype(np.uint8)
                predics.append(rle_encode(pred_img))

    pred_df = pd.DataFrame({"id": ids, "class": classs, "predicted": predics})

    submission_df = submission1[["id", "class"]].merge(
        pred_df, on=["id", "class"], how="left"
    )
    submission_df["predicted"] = submission_df["predicted"].fillna("")
    submission_df.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with model predictions.")



## === cell 17
submission_df = pd.read_csv("submission.csv")
print(submission_df.shape)
print(submission_df.head())
print(submission_df.isna().sum())
print("submission.csv saved")
