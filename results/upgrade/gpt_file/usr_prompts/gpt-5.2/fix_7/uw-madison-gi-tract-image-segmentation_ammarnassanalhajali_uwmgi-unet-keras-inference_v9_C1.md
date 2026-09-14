# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.4735557113523095

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the TensorFlow import crash by forcing the pure‑Python protobuf implementation before importing TF, which resolves the `MessageFactory.GetPrototype` error in Kaggle’s environment. Then I fix the missing model file by loading the UWMGI model from Kaggle’s dataset folder if present, otherwise fall back to a safe baseline (all-empty masks) so a valid submission is always produced. I also fix the generator length/prediction alignment bug (your `__len__` used `floor`, which drops the last partial batch and causes `IndexError` later) by switching to `ceil` and iterating only over available predictions. Finally, I ensure RLE encoding matches the competition’s column-major (Fortran) order and produce `submission.csv` with the exact required columns.'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow/protobuf crash by pinning a compatible protobuf runtime via `pip` before importing TensorFlow (this is the direct cause of the `MessageFactory.GetPrototype` error in Kaggle). I also make the environment/data-path handling robust by auto-selecting the correct `/kaggle/input/...` base path instead of using `../input/...`, which commonly breaks depending on notebook working directory. These changes are execution/stability fixes and keep the model/generator/inference/RLE core logic identical, so score should move from 0.0 (crash) to a real non-zero score once the pretrained model can load and predict. Finally, I keep the submission writing unchanged but ensure it always produces a valid `submission.csv` even if model loading fails.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with generating an all-empty (or near-all-empty) submission due to the pretrained model not loading and/or overly aggressive post-processing (`np.round` on probabilities). To move the score upward toward the 0.473 target with minimal changes, I (1) make model discovery more reliable by also searching inside the already-mounted competition dataset directory and preferring the first successfully loaded model, and (2) replace `np.round` with a small, safer probability threshold (0.5) while keeping the same logits→resize→binary→RLE pipeline (same semantics, just correct binarization). I also ensure prediction count aligns exactly with `df_train` by using `model.predict(..., steps=len(generator))` and slicing, preventing silent truncation/misalignment that can degrade the submission validity/quality. These changes keep your architecture/training untouched (still pure inference) and only address issues that directly drive the 0.0 score.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the model never loading (so you submit all-empty masks) or with silently producing unusable predictions due to custom object deserialization issues. To move the score upward toward the 0.4736 target with minimal changes, I (1) make model loading robust by correctly handling models saved with custom losses/metrics (load with `safe_mode=False` when needed, and include a `custom_objects` mapping for common names), and (2) ensure inference uses a stable sigmoid conversion if the model outputs logits (without changing the architecture or pipeline). I also keep your RLE and submission alignment logic intact, but add a tiny safeguard to guarantee predicted arrays are within [0,1] before thresholding, preventing degenerate all-zeros from numeric issues. These are execution/compatibility fixes aimed specifically at getting non-empty, valid segmentations rather than changing your approach.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with a structurally valid submission that contains effectively all-empty masks because no pretrained model is actually available/loaded, so the model inference never produces non-trivial RLEs. To move the score upward toward the 0.4736 target with minimal risk, I (1) stop relying on an external model file and instead use your existing input pipeline to build a tiny “leak-free baseline” by estimating per-class average masks from the provided `train.csv` RLEs at native resolution, then (2) apply those averaged priors to each test slice and RLE-encode them. This preserves your core feature extraction and RLE semantics (same decode/resize/encode logic), produces non-empty, valid segmentations, and typically yields a meaningful non-zero score without changing architecture/training loops. I also fix a key bug in `rle_decode`: it currently reshapes in C-order, which is inconsistent with the required Fortran-order convention used by Kaggle RLE, and that mismatch can collapse any learned/estimated masks toward garbage/empty.'

# 9. Code solution

## === cell 0
import os, random, gc, warnings, sys, subprocess

warnings.filterwarnings("ignore")

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PYTHONHASHSEED"] = "42"
random.seed(42)


def _ensure_protobuf_compat():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver

        if pb_ver is None:
            return
        major = int(pb_ver.split(".")[0])
        if major >= 5:
            raise RuntimeError(f"protobuf {pb_ver} too new for this TF build")
    except Exception:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==3.20.3"]
        )


_ensure_protobuf_compat()
gc.collect()



## === cell 1
import pandas as pd
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm

import tensorflow as tf

K = tf.keras.backend
from tensorflow.keras.models import load_model



## === cell 2
BATCH_SIZE = 16
EPOCHS = 30
n_splits = 5
fold_selected = 1  # 1..5



## === cell 3
DATA_ROOT_CANDIDATES = [
    "/kaggle/input/uw-madison-gi-tract-image-segmentation",
    "../input/uw-madison-gi-tract-image-segmentation",
]
DATA_ROOT = None
for p in DATA_ROOT_CANDIDATES:
    if os.path.exists(p):
        DATA_ROOT = p
        break
if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate uw-madison-gi-tract-image-segmentation dataset folder."
    )

df = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))
DEBUG = False
if df.shape[0] == 0:
    DEBUG = True
if DEBUG:
    df = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"))
    df.pop("segmentation")
    df["predicted"] = ""



## === cell 4
df.rename(columns={"class": "class_name"}, inplace=True)
df["case"] = df["id"].apply(lambda x: int(x.split("_")[0].replace("case", "")))
df["day"] = df["id"].apply(lambda x: int(x.split("_")[1].replace("day", "")))
df["slice"] = df["id"].apply(lambda x: x.split("_")[3])

if DEBUG:
    TRAIN_DIR = os.path.join(DATA_ROOT, "train")
else:
    TRAIN_DIR = os.path.join(DATA_ROOT, "test")

all_train_images = glob(os.path.join(TRAIN_DIR, "**", "*.png"), recursive=True)
if len(all_train_images) == 0:
    raise FileNotFoundError(f"No PNG images found under: {TRAIN_DIR}")

x = all_train_images[0].rsplit("/", 4)[0]  # .../train or .../test

path_partial_list = []
for i in range(0, df.shape[0]):
    path_partial_list.append(
        os.path.join(
            x,
            "case" + str(df["case"].values[i]),
            "case" + str(df["case"].values[i]) + "_" + "day" + str(df["day"].values[i]),
            "scans",
            "slice_" + str(df["slice"].values[i]),
        )
    )
df["path_partial"] = path_partial_list

path_partial_list = []
for i in range(0, len(all_train_images)):
    path_partial_list.append(str(all_train_images[i].rsplit("_", 4)[0]))

tmp_df = pd.DataFrame()
tmp_df["path_partial"] = path_partial_list
tmp_df["path"] = all_train_images

df = df.merge(tmp_df, on="path_partial").drop(columns=["path_partial"])
df["width"] = df["path"].apply(lambda x: int(x[:-4].rsplit("_", 4)[1]))
df["height"] = df["path"].apply(lambda x: int(x[:-4].rsplit("_", 4)[2]))
del x, path_partial_list, tmp_df
df.head(5)



## === cell 5
df_train = pd.DataFrame({"id": df["id"][::3]})
df_train["path"] = df["path"][::3].values
df_train["predicted"] = df["predicted"][::3].values
df_train["case"] = df["case"][::3].values
df_train["day"] = df["day"][::3].values
df_train["slice"] = df["slice"][::3].values
df_train["width"] = df["width"][::3].values
df_train["height"] = df["height"][::3].values

del df
df_train.reset_index(inplace=True, drop=True)
df_train.fillna("", inplace=True)
df_train.head(5)



## === cell 6
print(df_train.shape)
if DEBUG:
    df_train = df_train.sample(frac=0.05, random_state=42).reset_index(drop=True)
print(df_train.shape)



## === cell 7
gc.collect()




## === cell 8
def rle_encode(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted.

    Kaggle UWMGI expects RLE over pixels numbered top-to-bottom then left-to-right,
    which corresponds to flattening in Fortran order ('F') for a (H,W) array.
    """
    if img is None:
        return ""
    img = img.astype(np.uint8)
    if img.ndim != 2:
        img = img.squeeze()
    pixels = img.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape, color=1):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height,width,channels) of array to return
    Returns numpy array, 1 - mask, 0 - background

    CHANGE (score-relevant correctness): decode must be consistent with Kaggle's
    Fortran-order flattening used by rle_encode, otherwise masks are transposed/scrambled
    and may become effectively empty after resizing/thresholding.
    """
    if (
        mask_rle is None
        or mask_rle == ""
        or (isinstance(mask_rle, float) and np.isnan(mask_rle))
    ):
        return np.zeros(shape, dtype=np.float32)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths

    h, w, c = shape
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
        self.df = df
        self.shuffle = shuffle
        self.subset = subset
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.df))
        batch_ids = self.indexes[start:end]
        bs = len(batch_ids)

        X = np.empty((bs, 128, 128, 3), dtype=np.float32)

        if self.subset == "train":
            y = np.empty((bs, 128, 128, 3), dtype=np.float32)

        for i, row_idx in enumerate(batch_ids):
            img_path = self.df["path"].iloc[row_idx]
            w = int(self.df["width"].iloc[row_idx])
            h = int(self.df["height"].iloc[row_idx])

            img = self.__load_grayscale(img_path)  # (128,128,3)
            X[i] = img

            if self.subset == "train":
                for k, j in zip([0, 1, 2], ["large_bowel", "small_bowel", "stomach"]):
                    rles = self.df[j].iloc[row_idx]
                    masks = rle_decode(rles, shape=(h, w, 1))
                    masks = cv2.resize(
                        masks, (128, 128), interpolation=cv2.INTER_NEAREST
                    )
                    y[i, :, :, k] = masks[:, :, 0]

        if self.subset == "train":
            return X, y
        else:
            return X

    def __load_grayscale(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32)

        if img.max() > 255.0:
            img = img / 65535.0
        else:
            img = img / 255.0

        img = np.expand_dims(img, axis=-1)  # (128,128,1)
        img = np.repeat(img, 3, axis=-1)  # (128,128,3)
        return img




## === cell 10
gc.collect()



## === cell 11
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




## === cell 12
gc.collect()



## === cell 13
custom_objects = {
    "dice_coef": dice_coef,
    "iou_coef": iou_coef,
    "dice_loss": dice_loss,
    "bce_dice_loss": bce_dice_loss,
    "binary_crossentropy": binary_crossentropy,
}

candidate_model_paths = [
    "../input/uwmgi-unet-keras/model.h5",
    "../input/uwmgi-unet-keras/model.keras",
    "/kaggle/input/uwmgi-unet-keras/model.h5",
    "/kaggle/input/uwmgi-unet-keras/model.keras",
]

candidate_model_paths += glob(os.path.join(DATA_ROOT, "**", "model.h5"), recursive=True)
candidate_model_paths += glob(
    os.path.join(DATA_ROOT, "**", "model.keras"), recursive=True
)

candidate_model_paths += glob("/kaggle/input/**/model.h5", recursive=True)
candidate_model_paths += glob("/kaggle/input/**/*.h5", recursive=True)
candidate_model_paths += glob("/kaggle/input/**/model.keras", recursive=True)
candidate_model_paths += glob("../input/**/model.h5", recursive=True)
candidate_model_paths += glob("../input/**/*.h5", recursive=True)
candidate_model_paths += glob("../input/**/model.keras", recursive=True)

candidate_model_paths = list(dict.fromkeys(candidate_model_paths))

model = None
loaded_from = None
for p in candidate_model_paths:
    if os.path.exists(p):
        try:
            model = load_model(p, custom_objects=custom_objects, compile=False)
            loaded_from = p
            break
        except TypeError:
            try:
                model = load_model(
                    p, custom_objects=custom_objects, compile=False, safe_mode=False
                )
                loaded_from = p
                break
            except Exception as e2:
                print(f"Failed to load model at {p} with safe_mode=False: {e2}")
        except Exception as e:
            print(f"Failed to load model at {p}: {e}")

if model is not None:
    print(f"Loaded model from: {loaded_from}")
else:
    print(
        "WARNING: No pretrained model file found. Will use a train-prior mask baseline (non-empty) to avoid 0.0 score."
    )
gc.collect()



## === cell 14
pred_batches = DataGenerator(df_train, batch_size=16, subset="test", shuffle=False)
gc.collect()

if model is not None:
    LOGITS = model.predict(pred_batches, steps=len(pred_batches), verbose=1)
    LOGITS = LOGITS[: len(df_train)]
else:
    LOGITS = None
gc.collect()




## === cell 15
def _build_average_mask_priors(data_root, out_hw=(128, 128), max_groups=800):
    train_csv = os.path.join(data_root, "train.csv")
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Missing train.csv at {train_csv}")
    tr = pd.read_csv(train_csv)

    tr["case"] = tr["id"].apply(lambda x: int(x.split("_")[0].replace("case", "")))
    tr["day"] = tr["id"].apply(lambda x: int(x.split("_")[1].replace("day", "")))
    tr["slice"] = tr["id"].apply(lambda x: x.split("_")[3])

    ids_unique = tr["id"].unique()
    if max_groups is not None and len(ids_unique) > max_groups:
        rng = np.random.default_rng(42)
        ids_unique = rng.choice(ids_unique, size=max_groups, replace=False)

    train_dir = os.path.join(data_root, "train")
    all_png = glob(os.path.join(train_dir, "**", "*.png"), recursive=True)
    if len(all_png) == 0:
        raise FileNotFoundError(f"No train PNGs found under {train_dir}")

    base = all_png[0].rsplit("/", 4)[0]
    wanted_partials = []
    for _id in ids_unique:
        _case = int(_id.split("_")[0].replace("case", ""))
        _day = int(_id.split("_")[1].replace("day", ""))
        _slice = _id.split("_")[3]
        wanted_partials.append(
            os.path.join(
                base,
                f"case{_case}",
                f"case{_case}_day{_day}",
                "scans",
                f"slice_{_slice}",
            )
        )
    wanted_partials = set(wanted_partials)

    partials = [p.rsplit("_", 4)[0] for p in all_png]
    tmp = pd.DataFrame({"path_partial": partials, "path": all_png})
    tmp = tmp[tmp["path_partial"].isin(wanted_partials)].drop_duplicates("path_partial")

    H0, W0 = out_hw
    priors_sum = {
        "large_bowel": np.zeros((H0, W0), dtype=np.float64),
        "small_bowel": np.zeros((H0, W0), dtype=np.float64),
        "stomach": np.zeros((H0, W0), dtype=np.float64),
    }
    priors_n = 0

    tr_sub = tr[tr["id"].isin(ids_unique)].copy()
    tr_pivot = tr_sub.pivot_table(
        index="id", columns="class", values="segmentation", aggfunc="first"
    )
    tr_pivot = tr_pivot.reset_index()

    for _, row in tqdm(
        tr_pivot.iterrows(), total=len(tr_pivot), desc="Building priors"
    ):
        _id = row["id"]
        _case = int(_id.split("_")[0].replace("case", ""))
        _day = int(_id.split("_")[1].replace("day", ""))
        _slice = _id.split("_")[3]
        pp = os.path.join(
            base,
            f"case{_case}",
            f"case{_case}_day{_day}",
            "scans",
            f"slice_{_slice}",
        )
        m = tmp[tmp["path_partial"] == pp]
        if m.shape[0] == 0:
            continue
        img_path = m["path"].values[0]
        w = int(img_path[:-4].rsplit("_", 4)[1])
        h = int(img_path[:-4].rsplit("_", 4)[2])

        for cls in ["large_bowel", "small_bowel", "stomach"]:
            rle = row.get(cls, "")
            mask = rle_decode(rle, shape=(h, w, 1))[:, :, 0]
            mask = cv2.resize(mask, (W0, H0), interpolation=cv2.INTER_NEAREST)
            priors_sum[cls] += mask

        priors_n += 1

    if priors_n == 0:
        return {k: np.zeros((H0, W0), dtype=np.float32) for k in priors_sum.keys()}

    priors_mean = {
        k: (priors_sum[k] / priors_n).astype(np.float32) for k in priors_sum.keys()
    }
    return priors_mean


if LOGITS is None:
    priors = _build_average_mask_priors(DATA_ROOT, out_hw=(128, 128), max_groups=800)
else:
    priors = None

gc.collect()



## === cell 16
if LOGITS is None:
    n_pred = df_train.shape[0]
else:
    n_pred = int(LOGITS.shape[0])

n_use = min(df_train.shape[0], n_pred)
lbs, sbs, sts = [], [], []

THRESH = 0.5

if LOGITS is None:
    for index in tqdm(range(n_use), total=n_use, desc="Predicting (priors)"):
        h = int(df_train.iloc[index]["height"])
        w = int(df_train.iloc[index]["width"])

        pred0 = cv2.resize(
            priors["large_bowel"], (w, h), interpolation=cv2.INTER_NEAREST
        )
        pred1 = cv2.resize(
            priors["small_bowel"], (w, h), interpolation=cv2.INTER_NEAREST
        )
        pred2 = cv2.resize(priors["stomach"], (w, h), interpolation=cv2.INTER_NEAREST)

        lbs.append(rle_encode((pred0 >= THRESH).astype("uint8")))
        sbs.append(rle_encode((pred1 >= THRESH).astype("uint8")))
        sts.append(rle_encode((pred2 >= THRESH).astype("uint8")))
else:
    logits_min = float(np.nanmin(LOGITS))
    logits_max = float(np.nanmax(LOGITS))
    use_sigmoid = (logits_min < -0.05) or (logits_max > 1.05)
    if use_sigmoid:
        LOGITS = 1.0 / (1.0 + np.exp(-np.clip(LOGITS, -30, 30)))

    LOGITS = np.nan_to_num(LOGITS, nan=0.0, posinf=1.0, neginf=0.0)
    LOGITS = np.clip(LOGITS, 0.0, 1.0)

    for index in tqdm(range(n_use), total=n_use, desc="Predicting (model)"):
        h = int(df_train.iloc[index]["height"])
        w = int(df_train.iloc[index]["width"])

        pred0 = cv2.resize(
            LOGITS[index, :, :, 0], (w, h), interpolation=cv2.INTER_NEAREST
        )
        pred1 = cv2.resize(
            LOGITS[index, :, :, 1], (w, h), interpolation=cv2.INTER_NEAREST
        )
        pred2 = cv2.resize(
            LOGITS[index, :, :, 2], (w, h), interpolation=cv2.INTER_NEAREST
        )

        lbs.append(rle_encode((pred0 >= THRESH).astype("uint8")))
        sbs.append(rle_encode((pred1 >= THRESH).astype("uint8")))
        sts.append(rle_encode((pred2 >= THRESH).astype("uint8")))

del LOGITS
gc.collect()



## === cell 17
df_ids = df_train[["id"]].copy()
gc.collect()



## === cell 18
ids, classes, rles = [], [], []
for i in tqdm(range(n_use), total=n_use):
    _id = df_ids.iloc[i]["id"]
    ids.extend([_id, _id, _id])
    classes.extend(["large_bowel", "small_bowel", "stomach"])
    rles.extend([lbs[i], sbs[i], sts[i]])



## === cell 19
sub = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))
pred_df = pd.DataFrame({"id": ids, "class": classes, "predicted": rles})

sub = sub.merge(pred_df, on=["id", "class"], how="left", suffixes=("", "_pred"))
sub["predicted"] = sub["predicted_pred"].fillna("")
sub = sub[["id", "class", "predicted"]]

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)

sub.head()
