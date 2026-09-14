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

0.7308678564746782

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import warnings

warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob
import cv2

from tqdm import tqdm
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
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
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

pd.set_option("display.max_columns", 200)
pd.set_option("display.max_colwidth", 200)




## === cell 1
repertory = "/kaggle/input/"

DIR = repertory + "uw-madison-gi-tract-image-segmentation/"
TRAIN_DIR = DIR + "train"
TEST_DIR = DIR + "test"
train_csv = DIR + "train.csv"
sample_sub = DIR + "sample_submission.csv"

df_train = pd.read_csv(train_csv)
df_train.head(10)




## === cell 2
class CFG:
    BATCH_SIZE = 64
    img_size = (256, 256, 1)  # grayscale input
    n_fold = 5
    fold_selected = 1
    epochs = 100
    seed = 42
    steps_per_epoch_train = None
    steps_per_epoch_val = None




## === cell 3
def preprocessing(df, subset="train"):
    df["case"] = df["id"].apply(lambda x: int(x.split("_")[0].replace("case", "")))
    df["day"] = df["id"].apply(lambda x: int(x.split("_")[1].replace("day", "")))
    df["slice"] = df["id"].apply(lambda x: x.split("_")[3])
    if subset == "train":
        all_images = glob(os.path.join(TRAIN_DIR, "**", "*.png"), recursive=True)
        base_path = all_images[0].rsplit("/", 4)[0]  # .../train
    else:
        all_images = glob(os.path.join(TEST_DIR, "**", "*.png"), recursive=True)
        base_path = all_images[0].rsplit("/", 4)[0]  # .../test

    path_partial_list = []
    for i in range(df.shape[0]):
        path_partial_list.append(
            os.path.join(
                base_path,
                f"case{df['case'].values[i]}",
                f"case{df['case'].values[i]}_day{df['day'].values[i]}",
                "scans",
                f"slice_{df['slice'].values[i]}",
            )
        )
    df["path_partial"] = path_partial_list

    full_paths = []
    partials = []
    for img_path in all_images:
        partials.append(str(img_path.rsplit("_", 4)[0]))
        full_paths.append(img_path)
    tmp_df = pd.DataFrame({"path_partial": partials, "path": full_paths})
    df = pd.merge(df, tmp_df, on="path_partial").drop(columns=["path_partial"])

    df["width"] = df["path"].apply(lambda x: int(x[:-4].rsplit("_", 4)[1]))
    df["height"] = df["path"].apply(lambda x: int(x[:-4].rsplit("_", 4)[2]))
    df["px_spacing_h"] = df["path"].apply(lambda x: float(x[:-4].rsplit("_", 4)[3]))
    df["px_spacing_w"] = df["path"].apply(lambda x: float(x[:-4].rsplit("_", 4)[4]))

    return df




## === cell 4
train_df = preprocessing(df_train, subset="train")
train_df.head()




## === cell 5
def rle_decode(mask_rle, shape):
    s = np.asarray(mask_rle.split(), dtype=int)
    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


def rle_encode(img):
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 6
def id2mask(id_):
    itrain_df = train_df[train_df["id"] == id_]
    wh = itrain_df[["height", "width"]].iloc[0]
    shape = (wh.height, wh.width, 3)
    mask = np.zeros(shape, dtype=np.uint8)
    for i, class_ in enumerate(["large_bowel", "small_bowel", "stomach"]):
        ctrain_df = itrain_df[itrain_df["class"] == class_]
        rle = ctrain_df.segmentation.squeeze()
        if len(ctrain_df) and not pd.isna(rle):
            mask[..., i] = rle_decode(rle, shape[:2])
    return mask


def rgb2gray(mask):
    pad_mask = np.pad(mask, pad_width=[(0, 0), (0, 0), (1, 0)])
    gray_mask = pad_mask.argmax(-1)
    return gray_mask


def gray2rgb(mask):
    rgb_mask = to_categorical(mask, num_classes=4)
    return rgb_mask[..., 1:].astype(mask.dtype)




## === cell 7
def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = img.astype("float32")
    img = (img - img.min()) / (img.max() - img.min()) * 255.0
    img = img.astype("uint8")
    return img


def show_img(img, mask=None):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    plt.imshow(img, cmap="bone")
    if mask is not None:
        plt.imshow(mask, alpha=0.5)
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=c)
            for c in [(0.667, 0.0, 0.0), (0.0, 0.667, 0.0), (0.0, 0.0, 0.667)]
        ]
        labels = ["Large Bowel", "Small Bowel", "Stomach"]
        plt.legend(handles, labels)
    plt.axis("off")




## === cell 8
def restructure(df, subset="train"):
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
    df_out = df_out.reset_index(drop=True).fillna("")
    if subset == "train":
        df_out["count"] = np.sum(df_out.iloc[:, 1:4] != "", axis=1).values
    return df_out




## === cell 9
DF_train = restructure(train_df, subset="train")




## === cell 10
class DataGenerator(tf.keras.utils.Sequence):
    """
    Optimized generator:
    - Loads and normalises the image once per sample.
    - Decodes each class mask only once and resizes it in a single call.
    - Keeps the same output shapes, preserving training semantics.
    """

    def __init__(
        self,
        df,
        batch_size=CFG.BATCH_SIZE,
        subset="train",
        shuffle=False,
        img_shape=CFG.img_size,
    ):
        self.df = df
        self.batch_size = batch_size
        self.subset = subset
        self.shuffle = shuffle
        self.img_shape = img_shape
        self.indexes = np.arange(len(df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        batch_idxs = self.indexes[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        X = np.empty((self.batch_size, *self.img_shape), dtype=np.float32)
        y = np.empty((self.batch_size, *self.img_shape[:2], 3), dtype=np.float32)
        ids, heights, widths, classes = [], [], [], []

        for i, idx in enumerate(batch_idxs):
            img_path = self.df["path"].iloc[idx]
            w = self.df["width"].iloc[idx]
            h = self.df["height"].iloc[idx]

            img = self.__load_grayscale(img_path)
            X[i] = img

            if self.subset == "train":
                masks = []
                for cls in ["large_bowel", "small_bowel", "stomach"]:
                    rle = self.df[cls].iloc[idx]
                    if rle:
                        mask = rle_decode(rle, (h, w, 1))
                    else:
                        mask = np.zeros((h, w, 1), dtype=np.uint8)
                    mask_resized = cv2.resize(mask, self.img_shape[:2])
                    masks.append(mask_resized.squeeze())
                y[i] = np.stack(masks, axis=-1)
            else:
                ids.append(self.df["id"].iloc[idx])
                heights.append(h)
                widths.append(w)
                classes.append(self.df["class"].iloc[idx])

        if self.subset == "train":
            return X, y
        else:
            return X, ids, widths, heights, classes

    def __load_grayscale(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
        img = cv2.resize(img, self.img_shape[:2])
        img = img.astype("float32")
        img = (img - img.min()) / (img.max() - img.min())
        img = np.expand_dims(img, axis=-1)
        return img




## === cell 11
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
    return tf.keras.losses.binary_crossentropy(
        tf.cast(y_true, tf.float32), y_pred
    ) + 0.5 * dice_loss(tf.cast(y_true, tf.float32), y_pred)




## === cell 12
def conv_block(input_tensor, num_filters, batchnorm):
    x = Conv2D(num_filters, (3, 3), padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def encoder_block(input_tensor, num_filters, dropout=False, batchnorm=True):
    x = conv_block(input_tensor, num_filters, batchnorm)
    p = MaxPool2D((2, 2))(x)
    if dropout:
        p = Dropout(0.3)(p)
    return x, p


def decoder_block(
    input_tensor, skip_tensor, num_filters, dropout=False, batchnorm=True
):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input_tensor)
    x = Concatenate()([x, skip_tensor])
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


def plot_train(history):
    df = pd.DataFrame(history.history)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(df["loss"], label="Train loss")
    plt.plot(df["val_loss"], label="Val loss")
    plt.title("Loss")
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(df["dice_coef"], label="Train Dice")
    plt.plot(df["val_dice_coef"], label="Val Dice")
    plt.title("Dice")
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(df["iou_coef"], label="Train IoU")
    plt.plot(df["val_iou_coef"], label="Val IoU")
    plt.title("IoU")
    plt.legend()
    plt.show()


def fit_model(
    model, model_name, train_dataset, validation_dataset, models_path, results_path
):
    os.makedirs(models_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    weight_file = os.path.join(models_path, f"{model_name}.h5")
    result_file = os.path.join(results_path, f"score_{model_name}.joblib")
    if os.path.isfile(weight_file):
        model = load_model(
            weight_file,
            custom_objects={
                "bce_dice_loss": bce_dice_loss,
                "iou_coef": iou_coef,
                "dice_coef": dice_coef,
            },
        )
        results = load(result_file)
        plot_train(results)
    else:
        early_stop = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
        model.fit(
            train_dataset,
            epochs=CFG.epochs,
            validation_data=validation_dataset,
            callbacks=[early_stop],
            workers=4,
            use_multiprocessing=True,
        )
        results = pd.DataFrame(model.history.history)
        plot_train(model)
        dump(results, result_file, compress=True)
        model.save(weight_file)
    return model, results




## === cell 13
path_load_infer = repertory + "uwmgtis-keras-train-02/"
models_path = path_load_infer
results_path = path_load_infer




## === cell 14
input_shape = CFG.img_size
model = build_unet(input_shape, dropout=True, batchnorm=True)




## === cell 15
pretrained_path = os.path.join(models_path, "U-net.h5")
if os.path.isfile(pretrained_path):
    model = load_model(
        pretrained_path,
        custom_objects={
            "bce_dice_loss": bce_dice_loss,
            "iou_coef": iou_coef,
            "dice_coef": dice_coef,
        },
    )
    results = None
else:
    train_df_split, val_df_split = train_test_split(
        DF_train, test_size=0.2, random_state=CFG.seed, stratify=DF_train["case"]
    )
    train_gen = DataGenerator(
        train_df_split,
        batch_size=CFG.BATCH_SIZE,
        subset="train",
        shuffle=True,
        img_shape=CFG.img_size,
    )
    val_gen = DataGenerator(
        val_df_split,
        batch_size=CFG.BATCH_SIZE,
        subset="train",
        shuffle=False,
        img_shape=CFG.img_size,
    )
    model, results = fit_model(
        model,
        "U-net",
        train_gen,
        val_gen,
        models_path=models_path,
        results_path=results_path,
    )




## === cell 16
sub_df = pd.read_csv(sample_sub)
if len(sub_df) == 0:
    debug = True
    sub_df = pd.read_csv(train_csv)
    test_df = preprocessing(df_train, subset="train")
    test_df = test_df[: 1000 * 3]
else:
    debug = False
    test_df = preprocessing(sub_df, subset="test")
test_df.head(5)




## === cell 17
def infer(DF, model, batch_size=CFG.BATCH_SIZE):
    pred_rle, pred_ids, pred_classes = [], [], []
    generator = DataGenerator(DF, batch_size=batch_size, subset="test", shuffle=False)
    for img, ids, widths, heights, classes in tqdm(generator):
        preds = model.predict(img, verbose=0)
        for j in range(batch_size):
            cls = classes[j]
            k = 0 if cls == "large_bowel" else 1 if cls == "small_bowel" else 2
            pred_img = cv2.resize(
                preds[j, :, :, k],
                (widths[j], heights[j]),
                interpolation=cv2.INTER_NEAREST,
            )
            pred_img = (pred_img > 0.5).astype("uint8")
            pred_ids.append(ids[j])
            pred_classes.append(cls)
            pred_rle.append(rle_encode(pred_img))
    return pred_rle, pred_ids, pred_classes




## === cell 18
CFG.BATCH_SIZE = 3
pred_rle, pred_ids, pred_classes = infer(test_df, model, batch_size=CFG.BATCH_SIZE)




## === cell 19
submission = pd.DataFrame(
    {"id": pred_ids, "class": pred_classes, "predicted": pred_rle}
)

if debug:
    base = pd.read_csv(train_csv).drop(columns=["segmentation"])
else:
    base = pd.read_csv(sample_sub).drop(columns=["predicted"])
sub_df = base.merge(submission, on=["id", "class"])
sub_df.to_csv("submission.csv", index=False)

submission.sample(10)
