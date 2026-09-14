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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import warnings

warnings.filterwarnings("ignore")

from time import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob
from joblib import dump, load
from tqdm import tqdm

from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    Activation,
    MaxPool2D,
    Conv2DTranspose,
    Concatenate,
    Input,
    Dropout,
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping

import cv2

pd.set_option("display.max_columns", 200)
pd.set_option("display.max_colwidth", 200)

os.environ.setdefault("PYTHONHASHSEED", "42")
np.random.seed(42)
tf.random.set_seed(42)




## === cell 1
repertory = "/kaggle/input/"

DIR = repertory + "uw-madison-gi-tract-image-segmentation/"
TRAIN_DIR = DIR + "train"
TEST_DIR = DIR + "test"
train_csv = DIR + "train.csv"
sample_sub = DIR + "sample_submission.csv"
test_csv = DIR + "test.csv"

df_train = pd.read_csv(train_csv)
df_train.head(10)




## === cell 2
class CFG:
    BATCH_SIZE = 64
    img_size = (256, 256, 1)
    n_fold = 5
    fold_selected = 1
    epochs = 100
    seed = 42
    steps_per_epoch_train = None
    steps_per_epoch_val = None




## === cell 3
def preprocessing(df, subset="train"):
    df = df.copy()

    parts = df["id"].str.split("_", expand=True)
    df["case"] = parts[0].str.replace("case", "", regex=False).astype(np.int32)
    df["day"] = parts[1].str.replace("day", "", regex=False).astype(np.int32)
    df["slice"] = parts[3]  # keep as string to match filenames

    base_dir = TRAIN_DIR if subset == "train" else TEST_DIR
    all_images = glob(os.path.join(base_dir, "**", "*.png"), recursive=True)

    partials = [p.rsplit("_", 4)[0] for p in all_images]
    path_map = dict(zip(partials, all_images))

    x = all_images[0].rsplit("/", 4)[0] if len(all_images) else base_dir
    df["path_partial"] = (
        x
        + "/case"
        + df["case"].astype(str)
        + "/case"
        + df["case"].astype(str)
        + "_day"
        + df["day"].astype(str)
        + "/scans/slice_"
        + df["slice"].astype(str)
    )

    df["path"] = df["path_partial"].map(path_map)
    df = df.drop(columns=["path_partial"])

    p = df["path"].astype("string")
    stem = p.str.replace(".png", "", regex=False)
    spl = stem.str.rsplit("_", n=4, expand=True)
    df["width"] = pd.to_numeric(spl[1], errors="coerce")
    df["height"] = pd.to_numeric(spl[2], errors="coerce")
    df["px_spacing_h"] = pd.to_numeric(spl[3], errors="coerce")
    df["px_spacing_w"] = pd.to_numeric(spl[4], errors="coerce")

    df = df.dropna(subset=["path", "width", "height"]).reset_index(drop=True)
    df["width"] = df["width"].astype(int)
    df["height"] = df["height"].astype(int)

    return df




## === cell 4
train_df = preprocessing(df_train, subset="train")
train_df.head()




## === cell 5
def rle_decode(mask_rle, shape):
    if (
        mask_rle is None
        or mask_rle == ""
        or (isinstance(mask_rle, float) and np.isnan(mask_rle))
    ):
        return np.zeros(shape, dtype=np.uint8)

    s = np.asarray(mask_rle.split(), dtype=np.int64)
    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


def rle_encode(img):
    if img is None:
        return ""
    pixels = np.asarray(img, dtype=np.uint8).ravel(order="C")
    if pixels.size == 0:
        return ""
    padded = np.empty(pixels.size + 2, dtype=np.uint8)
    padded[0] = 0
    padded[-1] = 0
    padded[1:-1] = pixels
    runs = np.flatnonzero(padded[1:] != padded[:-1]) + 1
    runs[1::2] -= runs[::2]
    return " ".join(map(str, runs.tolist()))




## === cell 6
def restructure(df, subset="train"):
    df = df.reset_index(drop=True)

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

    df_out = df_out.fillna("").reset_index(drop=True)
    if subset == "train":
        df_out["count"] = np.sum(df_out.iloc[:, 1:4] != "", axis=1).values

    return df_out




## === cell 7
DF_train = restructure(train_df, subset="train")
DF_train.head()




## === cell 8
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        df,
        batch_size=CFG.BATCH_SIZE,
        subset="train",
        shuffle=False,
        img_shape=CFG.img_size,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.shuffle = shuffle
        self.subset = subset
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()

        self._path = self.df["path"].to_numpy()
        self._width = self.df["width"].to_numpy()
        self._height = self.df["height"].to_numpy()
        self._id = self.df["id"].to_numpy()
        if self.subset == "train":
            self._rle_lb = self.df["large_bowel"].to_numpy()
            self._rle_sb = self.df["small_bowel"].to_numpy()
            self._rle_st = self.df["stomach"].to_numpy()

    def __len__(self):
        if self.subset == "train":
            return int(np.floor(len(self.df) / self.batch_size))
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.df))
        batch_ids = self.indexes[start:end]
        bs = len(batch_ids)

        X = np.empty(
            (bs, self.img_shape[0], self.img_shape[1], self.img_shape[2]),
            dtype=np.float32,
        )

        if self.subset == "train":
            y = np.empty(
                (bs, self.img_shape[0], self.img_shape[1], 3), dtype=np.float32
            )
        else:
            id_, heights, widths = [], [], []

        for i, row_idx in enumerate(batch_ids):
            img_path = self._path[row_idx]
            w = int(self._width[row_idx])
            h = int(self._height[row_idx])

            img = self.__load_grayscale(img_path)
            X[i] = img

            if self.subset == "train":
                for k, rles in enumerate(
                    (
                        self._rle_lb[row_idx],
                        self._rle_sb[row_idx],
                        self._rle_st[row_idx],
                    )
                ):
                    mask = rle_decode(rles, shape=(h, w))
                    mask = cv2.resize(
                        mask, self.img_shape[0:2], interpolation=cv2.INTER_NEAREST
                    )
                    y[i, :, :, k] = mask.astype(np.float32)
            else:
                id_.append(self._id[row_idx])
                heights.append(h)
                widths.append(w)

        if self.subset == "train":
            return X, y
        else:
            return X, id_, widths, heights

    def __load_grayscale(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
        if img is None:
            img = np.zeros(self.img_shape[0:2], dtype=np.uint16)

        img = cv2.resize(img, self.img_shape[0:2], interpolation=cv2.INTER_AREA)
        img = img.astype("float32")
        denom = img.max() - img.min()
        if denom > 0:
            img = (img - img.min()) / denom
        else:
            img = img * 0.0
        img = np.expand_dims(img, axis=-1)
        return img




## === cell 9
def dice_coef(y_true, y_pred, smooth=1e-6):
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




## === cell 10
def conv_block(input, num_filters, batchnorm):
    x = Conv2D(num_filters, kernel_size=(3, 3), padding="same")(input)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(num_filters, kernel_size=(3, 3), padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def encoder_block(input, num_filters, dropout=False, batchnorm=True):
    x = conv_block(input, num_filters, batchnorm)
    p = MaxPool2D((2, 2))(x)
    if dropout:
        p = Dropout(0.3)(p)
    return x, p


def decoder_block(input, skip_features, num_filters, dropout=False, batchnorm=True):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    if dropout:
        x = Dropout(0.3)(x)
    x = conv_block(x, num_filters, batchnorm)
    return x


def build_unet(input_shape, dropout=False, batchnorm=True, activation="sigmoid"):
    inputs = Input(shape=input_shape)

    s0, p0 = encoder_block(inputs, 32, dropout, batchnorm)
    s1, p1 = encoder_block(p0, 64, dropout, batchnorm)
    s2, p2 = encoder_block(p1, 128, dropout, batchnorm)
    s3, p3 = encoder_block(p2, 256, dropout, batchnorm)
    s4, p4 = encoder_block(p3, 512, dropout, batchnorm)
    s5, p5 = encoder_block(p4, 1024, dropout, batchnorm)

    b1 = conv_block(p5, 2048, batchnorm)

    d0 = decoder_block(b1, s5, 1024, dropout, batchnorm)
    d1 = decoder_block(d0, s4, 512, dropout, batchnorm)
    d2 = decoder_block(d1, s3, 256, dropout, batchnorm)
    d3 = decoder_block(d2, s2, 128, dropout, batchnorm)
    d4 = decoder_block(d3, s1, 64, dropout, batchnorm)
    d5 = decoder_block(d4, s0, 32, dropout, batchnorm)

    outputs = Conv2D(3, 1, padding="same", activation=activation)(d5)

    model = Model(inputs, outputs, name="U-Net")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=bce_dice_loss,
        metrics=[iou_coef, dice_coef],
    )
    return model




## === cell 11
def fit_model(model, model_name, train_dataset, validation_dataset):
    if os.path.isfile(models_path + str(model_name) + ".h5"):
        model = load_model(
            models_path + str(model_name) + ".h5",
            custom_objects={
                "bce_dice_loss": bce_dice_loss,
                "iou_coef": iou_coef,
                "dice_coef": dice_coef,
            },
        )
        results = load(results_path + "score_" + str(model_name) + ".joblib")
    else:
        if train_dataset is None or validation_dataset is None:
            raise ValueError(
                "No pretrained model found, but train/validation datasets were not provided. "
                "Provide datasets or ensure the pretrained model path is correct."
            )
        early_stop = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
        model.fit(
            train_dataset,
            epochs=CFG.epochs,
            validation_data=validation_dataset,
            callbacks=[early_stop],
        )

        results = pd.DataFrame(model.history.history)
        dump(
            results,
            results_path + "score_" + str(model_name) + ".joblib",
            compress=True,
        )
        model.save(models_path + str(model_name) + ".h5")

    return model, results




## === cell 12
path_load_infer = repertory + "uwmgtis-keras-train-02/"
models_path = path_load_infer
results_path = path_load_infer




## === cell 13
input_shape = CFG.img_size
model = build_unet(input_shape, dropout=True, batchnorm=True)




## === cell 14
need_train = not os.path.isfile(models_path + "U-net.h5")
if need_train:
    kf = KFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
    splits = list(kf.split(DF_train))
    tr_idx, va_idx = splits[CFG.fold_selected]
    tr_df = DF_train.iloc[tr_idx].reset_index(drop=True)
    va_df = DF_train.iloc[va_idx].reset_index(drop=True)

    train_gen = DataGenerator(
        tr_df,
        batch_size=CFG.BATCH_SIZE,
        subset="train",
        shuffle=True,
        img_shape=CFG.img_size,
    )
    val_gen = DataGenerator(
        va_df,
        batch_size=CFG.BATCH_SIZE,
        subset="train",
        shuffle=False,
        img_shape=CFG.img_size,
    )
else:
    train_gen = None
    val_gen = None




## === cell 15
model, results = fit_model(model, "U-net", train_gen, val_gen)




## === cell 16
sub_df = pd.read_csv(sample_sub)
debug = False

df_test = pd.read_csv(test_csv)
test_df = preprocessing(df_test, subset="test")
test_df.head()




## === cell 17
def infer(test_df, model, batch_size=CFG.BATCH_SIZE):
    pred_rle = []
    pred_ids = []
    pred_classes = []

    img_df = test_df.drop_duplicates(subset=["id"]).reset_index(drop=True)

    gen = DataGenerator(
        img_df,
        batch_size=batch_size,
        subset="test",
        shuffle=False,
        img_shape=CFG.img_size,
    )

    classes = ("large_bowel", "small_bowel", "stomach")

    for imgs, ids, widths, heights in tqdm(gen, total=len(gen)):
        preds = model.predict(imgs, verbose=0)  # (B, 256, 256, 3)

        for j in range(len(ids)):
            wj = int(widths[j])
            hj = int(heights[j])
            for k, cls in enumerate(classes):
                prob = preds[j, :, :, k]
                pred_img = cv2.resize(prob, (wj, hj), interpolation=cv2.INTER_NEAREST)
                pred_bin = (pred_img > 0.5).astype(np.uint8)

                pred_ids.append(ids[j])
                pred_classes.append(cls)
                pred_rle.append(rle_encode(pred_bin))

    return pred_rle, pred_ids, pred_classes




## === cell 18
CFG.BATCH_SIZE = 4
pred_rle, pred_ids, pred_classes = infer(test_df, model, batch_size=CFG.BATCH_SIZE)




## === cell 19
submission = pd.DataFrame(
    {"id": pred_ids, "class": pred_classes, "predicted": pred_rle}
)

sub_df = pd.read_csv(sample_sub)
sub_df = sub_df.drop(columns=["predicted"]).merge(
    submission, on=["id", "class"], how="left"
)
sub_df["predicted"] = sub_df["predicted"].fillna("")

sub_df.to_csv("submission.csv", index=False)
sub_df.head()
