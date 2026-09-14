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

0.4484978484787068

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0029) has done: 'I fix the TensorFlow/protobuf import crash by forcing TensorFlow to use the python protobuf implementation before importing TF, and by using a safer import order. Then I fix the missing pretrained model path by falling back to a minimal U-Net-like model with the same 128×128×3 input and 3-channel sigmoid output so the notebook can run end-to-end even without external model files. Finally, I fix the DataGenerator/test prediction length mismatch (it was dropping the last partial batch) and ensure the RLE encoding uses the required column-major (Fortran) order so the submission format is valid and scores move toward the target instead of being penalized by wrong encoding.'
- What this solution (achieved 0.0) has done: 'I fix the protobuf/TensorFlow crash by forcing a compatible protobuf runtime mode before importing TensorFlow and, if needed, downgrading protobuf in-notebook to a version that matches TF’s expected API (this is the root cause of `MessageFactory.GetPrototype` failures). Then I correct the submission id parsing so slice filenames are built correctly (the current `slice` extraction is wrong), which currently causes missing/incorrect image-path joins and effectively random/empty predictions (driving the very low score). Finally, I make RLE decoding/encoding consistent with the competition’s column-major convention (decode currently reshapes in C-order), which prevents metric penalties from malformed masks; these are minimal, logic-preserving fixes aimed at moving the score up toward the target.'

# 9. Code solution

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



## === cell 1
BATCH_SIZE = 16
EPOCHS = 30
n_splits = 5
fold_selected = 1  # 1..5



## === cell 2
df = pd.read_csv(
    "../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv"
)
DEBUG = False
if df.shape[0] == 0:
    DEBUG = True
if DEBUG:
    df = pd.read_csv("../input/uw-madison-gi-tract-image-segmentation/train.csv")
    df.pop("segmentation")
    df["predicted"] = ""



## === cell 3
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
    raise FileNotFoundError(f"No PNGs found under {TRAIN_DIR}")

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
df["width"] = df["path"].apply(lambda p: int(p[:-4].rsplit("_", 4)[1]))
df["height"] = df["path"].apply(lambda p: int(p[:-4].rsplit("_", 4)[2]))

del x, path_partial_list, tmp_df
df.head(5)



## === cell 4
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



## === cell 5
print(df_train.shape)
if DEBUG:
    df_train = df_train.sample(frac=0.05).reset_index(drop=True)
print(df_train.shape)



## === cell 6
gc.collect()




## === cell 7
def rle_encode(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted.

    Competition expects pixels numbered top-to-bottom then left-to-right,
    which corresponds to flatten(order='F') for a (H,W) array.
    """
    if img.ndim != 2:
        img = img.squeeze()
    pixels = img.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(int(x)) for x in runs)


def rle_decode(mask_rle, shape, color=1):
    """
    mask_rle: run-length as string formatted (start length), can be '' for empty mask
    shape: (height,width,channels) of array to return
    Returns numpy array, 1 - mask, 0 - background

    --- Fix 3: Decode must match encode's column-major convention.
    We fill a flat array then reshape with order='F'.
    """
    h, w, c = shape
    if (
        mask_rle is None
        or mask_rle == ""
        or (isinstance(mask_rle, float) and np.isnan(mask_rle))
    ):
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
        mask = np.zeros((height, width, 3), dtype=np.float32)
        for label in labels:
            mask += rle_decode(label, shape=(height, width, 3), color=np.random.rand(3))
    else:
        mask = np.zeros((height, width, 1), dtype=np.float32)
        for label in labels:
            mask += rle_decode(label, shape=(height, width, 1))
    mask = mask.clip(0, 1)
    return mask




## === cell 8
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size=BATCH_SIZE, subset="train", shuffle=False):
        super().__init__()
        self.df = df.reset_index(drop=True)
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
        batch_indexes = self.indexes[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        cur_bs = len(batch_indexes)

        X = np.empty((cur_bs, 128, 128, 3), dtype=np.float32)
        if self.subset == "train":
            y = np.empty((cur_bs, 128, 128, 3), dtype=np.float32)

        for i, row_idx in enumerate(batch_indexes):
            img_path = self.df.loc[row_idx, "path"]
            w = int(self.df.loc[row_idx, "width"])
            h = int(self.df.loc[row_idx, "height"])

            img = self.__load_grayscale(img_path)  # (128,128,1)
            img3 = np.repeat(img, 3, axis=-1)  # (128,128,3)
            X[i] = img3

            if self.subset == "train":
                for k, j in zip([0, 1, 2], ["large_bowel", "small_bowel", "stomach"]):
                    rles = self.df.loc[row_idx, j]
                    masks = rle_decode(rles, shape=(h, w, 1))  # (h,w,1)
                    masks = cv2.resize(
                        masks, (128, 128), interpolation=cv2.INTER_NEAREST
                    )
                    if masks.ndim == 2:
                        masks = masks[..., None]
                    y[i, :, :, k] = masks[:, :, 0]

        if self.subset == "train":
            return X, y
        return X

    def __load_grayscale(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32)
        img = img / 255.0
        img = np.expand_dims(img, axis=-1)
        return img




## === cell 9
gc.collect()



## === cell 10
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




## === cell 11
gc.collect()



## === cell 12
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

gc.collect()



## === cell 13
pred_batches = DataGenerator(df_train, batch_size=1, subset="test", shuffle=False)
gc.collect()
LOGITS = model.predict(pred_batches, verbose=1)
gc.collect()



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2792429212.py in <cell line: 0>()
      1 pred_batches = DataGenerator(df_train, batch_size=1, subset="test", shuffle=False)
      2 gc.collect()
----> 3 LOGITS = model.predict(pred_batches, verbose=1)
      4 gc.collect()
      5 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/trainers/data_adapters/py_dataset_adapter.py in get_tf_dataset(self)
    293             ]
    294             if len(batches) == 0:
--> 295                 raise ValueError("The PyDataset has length 0")
    296             self._output_signature = data_adapter_utils.get_tensor_spec(batches)
    297 

ValueError: The PyDataset has length 0

## === cell 14
len(LOGITS)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/571340957.py in <cell line: 0>()
----> 1 len(LOGITS)
      2 

NameError: name 'LOGITS' is not defined

## === cell 15
lbs, sbs, sts = [], [], []
for index in tqdm(range(df_train.shape[0]), total=df_train.shape[0]):
    h = int(df_train.iloc[index]["height"])
    w = int(df_train.iloc[index]["width"])

    pred0 = cv2.resize(LOGITS[index, :, :, 0], (w, h), interpolation=cv2.INTER_NEAREST)
    pred1 = cv2.resize(LOGITS[index, :, :, 1], (w, h), interpolation=cv2.INTER_NEAREST)
    pred2 = cv2.resize(LOGITS[index, :, :, 2], (w, h), interpolation=cv2.INTER_NEAREST)

    pred_arr = (pred0 >= 0.5).astype("uint8")
    lbs.append(rle_encode(pred_arr))

    pred_arr = (pred1 >= 0.5).astype("uint8")
    sbs.append(rle_encode(pred_arr))

    pred_arr = (pred2 >= 0.5).astype("uint8")
    sts.append(rle_encode(pred_arr))

del LOGITS
gc.collect()



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1585381678.py in <cell line: 0>()
     17     sts.append(rle_encode(pred_arr))
     18 
---> 19 del LOGITS
     20 gc.collect()
     21 

NameError: name 'LOGITS' is not defined

## === cell 16
df_ids = df_train[["id"]].copy()
gc.collect()



## === cell 17
ids, classes, rles = [], [], []
for i, (_, row) in enumerate(tqdm(df_ids.iterrows(), total=df_ids.shape[0])):
    ids.extend([row["id"]] * 3)
    classes.extend(["large_bowel", "small_bowel", "stomach"])
    rles.extend([lbs[i], sbs[i], sts[i]])



## === cell 18
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



## === cell 19
submission.head()
