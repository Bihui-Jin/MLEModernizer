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

0.513932718393051

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the protobuf/TensorFlow import crash by forcing the pure-Python protobuf implementation before importing TensorFlow, which avoids the `MessageFactory.GetPrototype` error seen in Kaggle environments. Then I fix the missing model file issue by loading the model from the correct competition input directory if present, and if not, fall back to producing a valid (empty-mask) submission so the notebook always yields `submission.csv`. I also make inference robust by ensuring `model.predict(...)` returns one prediction per slice (matching `len(df_slices)`), and I guard the RLE/merge logic so list-length mismatches cannot crash submission creation. These changes preserve the existing core logic (pretrained model inference + RLE encoding) and focus on end-to-end correctness and producing a valid submission.'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow/protobuf crash by pinning the pure-Python protobuf implementation early and also forcing the slower-but-compatible python protobuf path via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2` before importing TensorFlow. Then I fix an inference logic bug where `DataGenerator.__len__` drops the last (partial) batch, causing `model.predict` to return too few predictions and cascading into empty/incorrect submissions (score 0.0); this is corrected by using `ceil` and allocating the final batch size properly. Finally, I ensure the test generator is built from unique slices (as intended) and the submission mapping remains aligned to `sample_submission.csv`, producing a valid `submission.csv` deterministically.'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow/protobuf crash by ensuring the pure-Python protobuf implementation is enforced *before* TensorFlow is imported and by falling back to a safe import path if the first import attempt fails. Then I fix the RLE encoding orientation to the competition’s required column-major (top-to-bottom then left-to-right) order; the current row-major flattening can produce effectively wrong masks and a near-0 score even when the model predicts meaningful segmentations. Finally, I keep your inference/generator/core pipeline intact, but add a small safety guard to binarize predictions with a 0.5 threshold (instead of rounding after resize) to avoid calibration quirks while preserving semantics and producing a valid `submission.csv` end-to-end.'
- What this solution (achieved 0.0) has done: 'You’re crashing before any model inference because the TensorFlow import workaround isn’t sufficient for this Kaggle image: the protobuf runtime is still picking a C++/incompatible path, triggering `MessageFactory.GetPrototype` during TensorFlow import. I fix this by forcing the pure-Python protobuf implementation and *also* pre-importing `google.protobuf` early (before TensorFlow) with a clean-module fallback, which is the minimal change that unblocks execution. After that, I keep your generator/inference/RLE logic intact, only adding a small robustness check to ensure the submission always has exactly the sample_submission rows even if slice prediction mapping is shorter (score-neutral but prevents invalid submissions). The script then run end-to-end and always write `submission.csv`.'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow import crash by avoiding the incompatible protobuf runtime path: instead of forcing the pure-Python protobuf (which is what triggers the `MessageFactory.GetPrototype` AttributeError in this environment), we remove those protobuf environment overrides and, as a minimal compatibility fallback, set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp` before importing TensorFlow. This unblocks execution so the model (if present) can run inference and produce non-empty masks rather than a 0.0 score from an empty submission. I keep your generator/inference/RLE core logic intact, only adjusting the import/bootstrap cell and keeping deterministic seeds. The script still always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow import crash by forcing protobuf to use the pure-Python implementation (the current `cpp` setting fails because `_message` isn’t available in this environment) and importing TensorFlow only after that environment is set. This unblocks the rest of the pipeline so `tf`, Keras, the generator, and custom loss/metrics are defined correctly and no longer cascade into `NameError`s. I also keep your existing inference + RLE logic intact, only adding a safe “empty submission” fallback if no `.h5` model is found so the script always writes a valid `submission.csv` (but when a model exists, it produce non-empty predictions and improve score from 0.0 toward the target). All paths and the submission format remain unchanged.'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow/protobuf import crash by changing the protobuf environment override to use the C++ implementation (which is the compatible path in this Kaggle image) and ensuring the env vars are set before importing TensorFlow. Then I keep your inference/generator/RLE pipeline intact, but make the model-file search a bit more robust for this competition’s typical public model datasets so you don’t silently fall back to an all-empty submission (which yields ~0.0). Finally, I keep the existing Fortran-order RLE encoding (correct for this competition) and ensure we always write a valid `submission.csv` with exactly the sample submission rows.'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow/protobuf import crash by forcing protobuf to use the pure-Python implementation (since the C++ `_message` extension is missing in this environment) before importing TensorFlow. Then I keep your existing inference + generator + RLE logic intact, only adding a small robustness guard so the code always produces a valid `submission.csv` even if no `.h5` model exists in `../input` (in that case it fall back to an all-empty submission, matching your intended behavior). This make the notebook run end-to-end and (when a model is found) produce non-empty masks with correct Fortran-order RLE encoding, moving the score up from 0.0 toward your target. No changes are made to the model architecture/training approach—only import/bootstrap and execution stability.'
- What this solution (achieved 0.0) has done: 'I fix the crash happening before any inference by removing the protobuf environment overrides that trigger the `MessageFactory.GetPrototype` error in this Kaggle TensorFlow build, and instead import TensorFlow in a clean, default environment. Then I keep your existing generator/inference/RLE pipeline intact, but add a small safety check to ensure the predicted tensor has 3 channels (matching the three classes) before indexing, preventing silent shape bugs that can lead to empty/invalid masks (and a 0.0 score). Finally, I keep the submission alignment to `sample_submission.csv` unchanged and always write a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation *before* importing TensorFlow, which addresses the `MessageFactory.GetPrototype` error that currently prevents any inference and leads to a 0.0 score. I also make the slice-id parsing robust (your `slice` extraction index is wrong for this competition’s ids) so image paths merge correctly and you don’t silently end up with missing/empty predictions. Finally, I keep your model loading, generator, inference, thresholding, and RLE logic the same, only adding a minimal safety check to ensure the submission always has exactly the sample submission rows and writes `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PYTHONHASHSEED"] = "42"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import random
import numpy as np

random.seed(42)
np.random.seed(42)

import google.protobuf  # noqa: F401

import tensorflow as tf

tf.random.set_seed(42)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import gc
import cv2
from glob import glob
from tqdm import tqdm

from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model



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
df["case"] = df["id"].apply(lambda x: int(x.split("_")[0].replace("case", "")))
df["day"] = df["id"].apply(lambda x: int(x.split("_")[1].replace("day", "")))

df["slice"] = df["id"].apply(lambda x: x.split("_")[2].replace("slice_", ""))

if DEBUG:
    TRAIN_DIR = "../input/uw-madison-gi-tract-image-segmentation/train"
else:
    TRAIN_DIR = "../input/uw-madison-gi-tract-image-segmentation/test"

all_train_images = glob(os.path.join(TRAIN_DIR, "**", "*.png"), recursive=True)
if len(all_train_images) == 0:
    raise FileNotFoundError(f"No .png images found under {TRAIN_DIR}")

x = all_train_images[0].rsplit("/", 4)[0]

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
df_slices = pd.DataFrame({"id": df["id"][::3]})
df_slices["path"] = df["path"][::3].values
df_slices["predicted"] = df["predicted"][::3].values
df_slices["case"] = df["case"][::3].values
df_slices["day"] = df["day"][::3].values
df_slices["slice"] = df["slice"][::3].values
df_slices["width"] = df["width"][::3].values
df_slices["height"] = df["height"][::3].values

del df
df_slices.reset_index(inplace=True, drop=True)
df_slices.fillna("", inplace=True)
df_slices.head(5)



## === cell 6
print(df_slices.shape)
if DEBUG:
    df_slices = df_slices.sample(frac=0.05, random_state=42).reset_index(drop=True)
print(df_slices.shape)



## === cell 7
gc.collect()




## === cell 8
def rle_encode(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted.

    IMPORTANT: Competition expects pixels numbered top-to-bottom then left-to-right,
    which corresponds to Fortran/column-major order flattening.
    """
    img = img.astype(np.uint8)
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

    img = np.zeros((shape[0] * shape[1], shape[2]), dtype=np.float32)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = color
    img = img.reshape((shape[0], shape[1], shape[2]), order="F")
    return img


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
import math


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size=BATCH_SIZE, subset="train", shuffle=False):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.shuffle = shuffle
        self.subset = subset
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        return int(math.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        batch_indexes = self.indexes[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        bs = len(batch_indexes)

        X = np.empty((bs, 128, 128, 1), dtype=np.float32)

        if self.subset == "train":
            y = np.empty((bs, 128, 128, 3), dtype=np.float32)

        for i, idx in enumerate(batch_indexes):
            img_path = self.df["path"].iloc[idx]
            w = int(self.df["width"].iloc[idx])
            h = int(self.df["height"].iloc[idx])

            img = self.__load_grayscale(img_path)
            X[i] = img

            if self.subset == "train":
                for k, j in zip([0, 1, 2], ["large_bowel", "small_bowel", "stomach"]):
                    rles = self.df[j].iloc[idx]
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
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)
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
    "bce_dice_loss": bce_dice_loss,
}

candidate_model_paths = [
    "../input/uwmgi-unet-keras/model.h5",  # original (may not exist)
    "../input/uw-madison-gi-tract-image-segmentation/model.h5",
]
try:
    candidate_model_paths += glob("../input/**/model.h5", recursive=True)
    candidate_model_paths += glob("../input/**/best*.h5", recursive=True)
    candidate_model_paths += glob("../input/**/*.h5", recursive=True)
except Exception:
    pass

seen = set()
candidate_model_paths = [
    p for p in candidate_model_paths if not (p in seen or seen.add(p))
]

model = None
model_path_used = None
for p in candidate_model_paths:
    if p and os.path.exists(p):
        model_path_used = p
        break

if model_path_used is not None:
    model = load_model(model_path_used, custom_objects=custom_objects, compile=False)

print("Model path used:", model_path_used)
gc.collect()



## === cell 14
if model is None:
    LOGITS = None
    print("WARNING: No model file found. Will create an all-empty submission.")
else:
    pred_batches = DataGenerator(df_slices, batch_size=1, subset="test", shuffle=False)
    gc.collect()
    LOGITS = model.predict(pred_batches, verbose=1)
    gc.collect()

    if LOGITS.ndim == 3:
        LOGITS = np.expand_dims(LOGITS, axis=-1)
    if LOGITS.shape[-1] == 1:
        LOGITS = np.repeat(LOGITS, 3, axis=-1)



## === cell 15
lbs, sbs, sts = [], [], []

if LOGITS is None:
    lbs = [""] * len(df_slices)
    sbs = [""] * len(df_slices)
    sts = [""] * len(df_slices)
else:
    n_pred = int(LOGITS.shape[0])
    n_need = int(len(df_slices))
    if n_pred != n_need:
        n = min(n_pred, n_need)
        LOGITS = LOGITS[:n]
        df_slices = df_slices.iloc[:n].reset_index(drop=True)
        print(f"WARNING: Prediction/df_slices length mismatch; truncated to {n}")

    for index in tqdm(range(len(df_slices)), total=len(df_slices)):
        root_shape = (
            int(df_slices.iloc[index]["height"]),
            int(df_slices.iloc[index]["width"]),
        )

        target_size = (root_shape[1], root_shape[0])

        pred_arr = cv2.resize(
            LOGITS[index, :, :, 0], target_size, interpolation=cv2.INTER_NEAREST
        )
        pred_arr = (pred_arr >= 0.5).astype("uint8")
        lbs.append(rle_encode(pred_arr))

        pred_arr = cv2.resize(
            LOGITS[index, :, :, 1], target_size, interpolation=cv2.INTER_NEAREST
        )
        pred_arr = (pred_arr >= 0.5).astype("uint8")
        sbs.append(rle_encode(pred_arr))

        pred_arr = cv2.resize(
            LOGITS[index, :, :, 2], target_size, interpolation=cv2.INTER_NEAREST
        )
        pred_arr = (pred_arr >= 0.5).astype("uint8")
        sts.append(rle_encode(pred_arr))

    del LOGITS
    gc.collect()



## === cell 16
df_slices = df_slices[["id"]].copy()
gc.collect()



## === cell 17
slice_pred_map = {}
n = len(df_slices)
if not (len(lbs) == len(sbs) == len(sts) == n):
    m = min(n, len(lbs), len(sbs), len(sts))
    df_slices = df_slices.iloc[:m].reset_index(drop=True)
    lbs, sbs, sts = lbs[:m], sbs[:m], sts[:m]
    n = m
    print(f"WARNING: Truncated outputs to {n} due to length mismatch.")

for i in range(n):
    slice_pred_map[df_slices.iloc[i]["id"]] = {
        "large_bowel": lbs[i],
        "small_bowel": sbs[i],
        "stomach": sts[i],
    }

sub = pd.read_csv(
    "../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv"
)

sub["predicted"] = sub.apply(
    lambda r: slice_pred_map.get(r["id"], {}).get(r["class"], ""), axis=1
)

sub = sub[["id", "class", "predicted"]]

sample = pd.read_csv(
    "../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv"
)[["id", "class"]]
sub = sample.merge(sub, on=["id", "class"], how="left")
sub["predicted"] = sub["predicted"].fillna("")

print(sub.shape)
sub.head()



## === cell 18
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with", len(sub), "rows")



## === cell 19
sub.head()
