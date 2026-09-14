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

0.4093368204087838

# 6. Current score

0.00391

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.00391) has done: 'I fix the TensorFlow import crash by forcing the pure-Python protobuf backend early (this is a common Kaggle/runtime incompatibility causing the `MessageFactory.GetPrototype` error). Then I remove the hard dependency on a missing external model file by training the same U-Net-style segmentation model inside the notebook (keeping the same loss/metrics and data pipeline semantics), so predictions can be produced end-to-end. Finally, I fix submission-building bugs (list length mismatch, missing `class` column) by generating predictions directly aligned to `test.csv`/`sample_submission.csv` row order and writing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os, random, gc, warnings

warnings.filterwarnings("ignore")

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PYTHONHASHSEED"] = "42"
random.seed(42)

import numpy as np

np.random.seed(42)

import pandas as pd
import cv2
from glob import glob
import tensorflow as tf

tf.random.set_seed(42)

print("TF version:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
    df_["case"] = df_["id"].apply(lambda x: int(x.split("_")[0].replace("case", "")))
    df_["day"] = df_["id"].apply(lambda x: int(x.split("_")[1].replace("day", "")))
    df_["slice"] = df_["id"].apply(lambda x: x.split("_")[3])
    return df_


test_df = add_id_cols(test_df)
train_csv = add_id_cols(train_csv)

print("train_csv:", train_csv.shape, "test_df:", test_df.shape, "sub_df:", sub_df.shape)




## === cell 4
def build_slice_df(base_dir, df_ids):
    scans_dir = os.path.join(base_dir, "scans")
    all_images = glob(os.path.join(base_dir, "**", "scans", "*.png"), recursive=True)
    if len(all_images) == 0:
        raise FileNotFoundError(f"No png images found under {base_dir}")

    prefix = all_images[0].rsplit("/", 4)[0]

    path_partial = []
    for i in range(df_ids.shape[0]):
        path_partial.append(
            os.path.join(
                prefix,
                f"case{df_ids['case'].values[i]}",
                f"case{df_ids['case'].values[i]}_day{df_ids['day'].values[i]}",
                "scans",
                f"slice_{df_ids['slice'].values[i]}",
            )
        )
    df_tmp = df_ids.copy()
    df_tmp["path_partial"] = path_partial

    img_partials = [str(p.rsplit("_", 4)[0]) for p in all_images]
    img_df = pd.DataFrame({"path_partial": img_partials, "path": all_images})

    out = df_tmp.merge(img_df, on="path_partial", how="left").drop(
        columns=["path_partial"]
    )
    if out["path"].isna().any():
        out = out.dropna(subset=["path"]).reset_index(drop=True)

    out["width"] = out["path"].apply(lambda x: int(x[:-4].rsplit("_", 4)[1]))
    out["height"] = out["path"].apply(lambda x: int(x[:-4].rsplit("_", 4)[2]))
    return out


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
    img: numpy array (H,W), 1 - mask, 0 - background
    Returns run length as string formatted
    """
    pixels = img.flatten(order="C")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape, color=1):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height,width,channels) of array to return
    Returns numpy array with given channels.
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
    return img.reshape(shape)




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
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        bs = len(indexes)

        X = np.empty((bs, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

        if self.subset == "train":
            y = np.empty((bs, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

        for i, row_idx in enumerate(indexes):
            img_path = self.df.loc[row_idx, "path"]
            w = int(self.df.loc[row_idx, "width"])
            h = int(self.df.loc[row_idx, "height"])

            img = self.__load_grayscale(img_path)  # (IMG_SIZE,IMG_SIZE,1)
            img = np.repeat(img, 3, axis=-1)  # (IMG_SIZE,IMG_SIZE,3)
            X[i] = img

            if self.subset == "train":
                for k, j in zip([0, 1, 2], ["large_bowel", "small_bowel", "stomach"]):
                    rles = self.df.loc[row_idx, j]
                    masks = rle_decode(rles, shape=(h, w, 1))
                    masks = cv2.resize(
                        masks, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST
                    )
                    y[i, :, :, k] = masks[:, :, 0]

        if self.subset == "train":
            return X, y
        else:
            return X

    def __load_grayscale(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)
        return img




## === cell 9
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




## === cell 10
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



## === cell 11
cases = train_slices[["case"]].copy()
cases["case_mod"] = cases["case"] % 10  # cheap strat label

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
folds = list(skf.split(train_slices, cases["case_mod"].values))
tr_idx, va_idx = folds[fold_selected - 1]

train_df = train_slices.iloc[tr_idx].reset_index(drop=True)
valid_df = train_slices.iloc[va_idx].reset_index(drop=True)

print("train/valid:", train_df.shape, valid_df.shape)

train_gen = DataGenerator(train_df, batch_size=BATCH_SIZE, subset="train", shuffle=True)
valid_gen = DataGenerator(
    valid_df, batch_size=BATCH_SIZE, subset="train", shuffle=False
)



## === cell 12
history = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=EPOCHS,
    verbose=2,
)

gc.collect()



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/2557126397.py in <cell line: 0>()
      1 # Train end-to-end so we can generate predictions (replaces missing ../input/uwmgi-unet-keras/model.h5).
      2 # No early stopping; fixed epochs as originally specified.
----> 3 history = model.fit(
      4     train_gen,
      5     validation_data=valid_gen,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_11/2837703151.py in __getitem__(self, index)
     41                         masks, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST
     42                     )
---> 43                     y[i, :, :, k] = masks[:, :, 0]
     44 
     45         if self.subset == "train":

IndexError: too many indices for array: array is 2-dimensional, but 3 were indexed

## === cell 13
pred_gen = DataGenerator(test_slices, batch_size=1, subset="test", shuffle=False)
LOGITS = model.predict(pred_gen, verbose=1)
LOGITS = LOGITS[: len(test_slices)]
print("LOGITS:", LOGITS.shape)
gc.collect()



## === cell 14
lbs, sbs, sts = [], [], []
for index in tqdm(range(len(test_slices))):
    h = int(test_slices.loc[index, "height"])
    w = int(test_slices.loc[index, "width"])

    pred0 = cv2.resize(LOGITS[index, :, :, 0], (w, h), interpolation=cv2.INTER_NEAREST)
    pred1 = cv2.resize(LOGITS[index, :, :, 1], (w, h), interpolation=cv2.INTER_NEAREST)
    pred2 = cv2.resize(LOGITS[index, :, :, 2], (w, h), interpolation=cv2.INTER_NEAREST)

    lbs.append(rle_encode((pred0 >= 0.5).astype("uint8")))
    sbs.append(rle_encode((pred1 >= 0.5).astype("uint8")))
    sts.append(rle_encode((pred2 >= 0.5).astype("uint8")))

del LOGITS
gc.collect()



## === cell 15
pred_map = {}
for i in range(len(test_slices)):
    rid = test_slices.loc[i, "id"]
    pred_map[(rid, "large_bowel")] = lbs[i]
    pred_map[(rid, "small_bowel")] = sbs[i]
    pred_map[(rid, "stomach")] = sts[i]

sub = sub_df.copy()
sub["predicted"] = [
    pred_map.get((rid, cname), "")
    for rid, cname in zip(sub["id"].values, sub["class_name"].values)
]
sub.rename(columns={"class_name": "class"}, inplace=True)

sub = sub[["id", "class", "predicted"]]
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
sub.head()



## === cell 16
sub.tail()
