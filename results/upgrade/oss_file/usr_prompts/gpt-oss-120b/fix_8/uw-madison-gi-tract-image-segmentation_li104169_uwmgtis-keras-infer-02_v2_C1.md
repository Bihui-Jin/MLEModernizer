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
import warnings

warnings.filterwarnings("ignore")

from time import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from pathlib import Path
import os
from glob import glob
from joblib import parallel_backend, Parallel, delayed, dump, load
from tqdm import tqdm

try:
    import cv2
except ImportError:
    cv2 = None
    from PIL import Image

TF_AVAILABLE = False


class DummyModel:
    def __init__(self, img_shape):
        self.img_shape = img_shape

    def predict(self, x, verbose=0):
        batch = x.shape[0]
        h, w, c = self.img_shape
        return np.zeros((batch, h, w, c), dtype=np.float32)

    def save(self, path):
        pass

    def compile(self, *args, **kwargs):
        pass


class K:
    @staticmethod
    def flatten(x):
        return x

    @staticmethod
    def sum(x, axis=None, keepdims=False):
        return np.sum(x, axis=axis, keepdims=keepdims)

    @staticmethod
    def mean(x, axis=None, keepdims=False):
        return np.mean(x, axis=axis, keepdims=keepdims)

    @staticmethod
    def abs(x):
        return np.abs(x)


class EarlyStopping:
    def __init__(self, *args, **kwargs):
        pass


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
    img_size = (256, 256, 3)
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

    base_dir = TRAIN_DIR if subset == "train" else TEST_DIR

    df["path_partial"] = (
        base_dir
        + "/case"
        + df["case"].astype(str)
        + "/case"
        + df["case"].astype(str)
        + "_day"
        + df["day"].astype(str)
        + "/scans/slice_"
        + df["slice"].astype(str)
    )

    all_images = glob(os.path.join(base_dir, "**", "*.png"), recursive=True)

    folders = pd.Series(all_images).apply(lambda x: x.rsplit("_", 4)[0])
    folder_to_file = pd.Series(all_images, index=folders)

    df["path"] = df["path_partial"].map(folder_to_file)

    df["width"] = df["path"].apply(lambda x: int(x[:-4].rsplit("_", 4)[1]))
    df["height"] = df["path"].apply(lambda x: int(x[:-4].rsplit("_", 4)[2]))
    df["px_spacing_h"] = df["path"].apply(lambda x: float(x[:-4].rsplit("_", 4)[3]))
    df["px_spacing_w"] = df["path"].apply(lambda x: float(x[:-4].rsplit("_", 4)[4]))

    df.drop(columns=["path_partial"], inplace=True)

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
def load_img(path):
    if cv2 is not None:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = img.astype("float32")
        img = (img - img.min()) / (img.max() - img.min()) * 255.0
        img = img.astype("uint8")
        return img
    else:
        img = Image.open(path).convert("L")
        img = np.array(img, dtype=np.float32)
        img = (img - img.min()) / (img.max() - img.min()) * 255.0
        return img.astype("uint8")


def show_img(img, mask=None):
    if cv2 is not None:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
    plt.imshow(img, cmap="bone")
    if mask is not None:
        plt.imshow(mask, alpha=0.5)
    plt.axis("off")




## === cell 7
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
    df_out = df_out.reset_index(drop=True)
    df_out = df_out.fillna("")
    if subset == "train":
        df_out["count"] = np.sum(df_out.iloc[:, 1:4] != "", axis=1).values
    display(df_out.sample(5))
    return df_out




## === cell 8
DF_train = restructure(train_df, subset="train")




## === cell 9
class DataGenerator:
    """
    Simple data generator not relying on TensorFlow.
    Returns batches suitable for the DummyModel inference.
    """

    def __init__(
        self,
        df,
        batch_size=CFG.BATCH_SIZE,
        subset="train",
        shuffle=False,
        img_shape=CFG.img_size,
    ):
        self.df = df.reset_index(drop=True)
        self.shuffle = shuffle
        self.subset = subset
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        start = index * self.batch_size
        end = start + self.batch_size
        indexes = self.indexes[start:end]

        X = np.empty(
            (self.batch_size, self.img_shape[0], self.img_shape[1], self.img_shape[2]),
            dtype=np.float32,
        )
        ids, widths, heights, classes = [], [], [], []

        for i, idx in enumerate(indexes):
            row = self.df.iloc[idx]
            img_path = row["path"]
            img = np.zeros(
                (self.img_shape[0], self.img_shape[1], self.img_shape[2]),
                dtype=np.float32,
            )
            X[i] = img

            if self.subset != "train":
                ids.append(row["id"])
                widths.append(row["width"])
                heights.append(row["height"])
                classes.append(row["class"])

        if self.subset == "train":
            y = np.zeros_like(X)
            return X, y
        else:
            return X, ids, widths, heights, classes

    def _load_grayscale(self, img_path):
        """Legacy method retained for interface compatibility."""
        target_h, target_w = self.img_shape[0], self.img_shape[1]

        if cv2 is not None:
            img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
            img = cv2.resize(img, (target_w, target_h))
        else:
            img = Image.open(img_path).convert("L")
            img = img.resize((target_w, target_h), Image.BILINEAR)
            img = np.array(img)

        img = img.astype("float32")
        if img.max() > img.min():
            img = (img - img.min()) / (img.max() - img.min())
        else:
            img = np.zeros_like(img, dtype=np.float32)

        img = np.expand_dims(img, axis=-1)  # (H, W, 1)
        img = np.repeat(img, 3, axis=-1)  # (H, W, 3)
        return img




## === cell 10
if TF_AVAILABLE:
    from tensorflow.keras import backend as K
    from tensorflow.keras.losses import binary_crossentropy
    import tensorflow as tf

    def dice_coef(y_true, y_pred, smooth=1e-6):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2.0 * intersection + smooth) / (
            K.sum(y_true_f) + K.sum(y_pred_f) + smooth
        )

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
        return binary_crossentropy(
            tf.cast(y_true, tf.float32), y_pred
        ) + 0.5 * dice_loss(tf.cast(y_true, tf.float32), y_pred)

else:

    def dice_coef(*args, **kwargs):
        return 0.0

    def iou_coef(*args, **kwargs):
        return 0.0

    def dice_loss(*args, **kwargs):
        return 0.0

    def bce_dice_loss(*args, **kwargs):
        return 0.0




## === cell 11
if TF_AVAILABLE:
    from tensorflow.keras.layers import (
        Conv2D,
        MaxPool2D,
        Dropout,
        Conv2DTranspose,
        Concatenate,
        Input,
        Activation,
        BatchNormalization,
    )
    from tensorflow.keras.models import Model, load_model

    def conv_block(input, num_filters, batchnorm):
        x = Conv2D(
            num_filters,
            kernel_size=(3, 3),
            padding="same",
        )(input)
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

    def plot_train(model):
        losses = (
            model
            if isinstance(model, pd.DataFrame)
            else pd.DataFrame(model.history.history)
        )
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.plot(losses["loss"].index, losses["loss"], label="Train_Loss")
        plt.plot(losses["val_loss"].index, losses["val_loss"], label="Val_loss")
        plt.title("LOSS")
        plt.xlabel("Epoch")
        plt.ylabel("loss")
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(
            losses["dice_coef"].index, losses["dice_coef"], label="Train_dice_coef"
        )
        plt.plot(
            losses["val_dice_coef"].index,
            losses["val_dice_coef"],
            label="Val_dice_coef",
        )
        plt.title("DICE")
        plt.xlabel("Epoch")
        plt.ylabel("dice_coef")
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(losses["iou_coef"].index, losses["iou_coef"], label="Train_iou_coef")
        plt.plot(
            losses["val_iou_coef"].index, losses["val_iou_coef"], label="Val_iou_coef"
        )
        plt.title("IOU")
        plt.xlabel("Epoch")
        plt.ylabel("iou_coef")
        plt.legend()
        plt.show()

    def fit_model(model, model_name, train_dataset, validation_dataset):
        if train_dataset is None or validation_dataset is None:
            print("No training/validation data provided – skipping model.fit.")
            results = pd.DataFrame()
            return model, results

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
            plot_train(results)

        else:
            early_stop = EarlyStopping(monitor="val_loss", patience=5)
            model.fit(
                train_dataset,
                epochs=CFG.epochs,
                validation_data=validation_dataset,
                callbacks=[early_stop],
            )

            results = pd.DataFrame(model.history.history)
            plot_train(model)

            dump(
                results,
                results_path + "score_" + str(model_name) + ".joblib",
                compress=True,
            )
            model.save(models_path + str(model_name) + ".h5")

        return model, results

else:

    def plot_train(*args, **kwargs):
        pass

    def fit_model(model, model_name, train_dataset, validation_dataset):
        results = pd.DataFrame()
        return model, results




## === cell 12
path_load_infer = repertory + "uwmgtis-keras-train-02/"

models_path = path_load_infer
results_path = path_load_infer

os.makedirs(models_path, exist_ok=True)
os.makedirs(results_path, exist_ok=True)




## === cell 13
input_shape = CFG.img_size
if TF_AVAILABLE:
    model = build_unet(input_shape, dropout=True, batchnorm=True)
else:
    model = DummyModel(img_shape=CFG.img_size)




## === cell 14
model, results = fit_model(model, "U-net", None, None)




## === cell 15
sub_df = pd.read_csv(sample_sub)
if not len(sub_df):
    debug = True
    sub_df = pd.read_csv(train_csv)
    test_df = preprocessing(df_train, subset="train")
    test_df = test_df[: 1000 * 3]
else:
    debug = False
    test_df = preprocessing(sub_df, subset="test")

test_df.head(5)




## === cell 16
def infer(DF, model, batch_size=CFG.BATCH_SIZE):
    """
    Run inference on a DataFrame and return RLE strings, ids and class names.
    Handles full batches via DataGenerator and processes any leftover rows.
    """
    pred_rle = []
    pred_ids = []
    pred_classes = []

    DF_batch = DataGenerator(DF, batch_size=batch_size, subset="test", shuffle=False)
    total_batches = len(DF_batch)

    for idx, (img, id_, widths, heights, classes) in enumerate(
        tqdm(DF_batch, total=total_batches)
    ):
        preds = model.predict(img, verbose=0)

        for j in range(len(classes)):
            k = (
                0
                if classes[j] == "large_bowel"
                else 1 if classes[j] == "small_bowel" else 2
            )
            if cv2 is not None:
                pred_img = cv2.resize(
                    preds[j, :, :, k],
                    (widths[j], heights[j]),
                    interpolation=cv2.INTER_NEAREST,
                )
            else:
                pil_img = Image.fromarray((preds[j, :, :, k] * 255).astype(np.uint8))
                pil_img = pil_img.resize((widths[j], heights[j]), Image.NEAREST)
                pred_img = np.array(pil_img)

            pred_img = (pred_img > 0.5).astype(dtype="uint8")
            pred_ids.append(id_[j])
            pred_classes.append(classes[j])
            pred_rle.append(rle_encode(pred_img))

    processed = total_batches * batch_size
    if processed < len(DF):
        remainder_df = DF.iloc[processed:].reset_index(drop=True)
        if not remainder_df.empty:
            rem_gen = DataGenerator(
                remainder_df,
                batch_size=len(remainder_df),
                subset="test",
                shuffle=False,
            )
            img, id_, widths, heights, classes = next(iter(rem_gen))
            preds = model.predict(img, verbose=0)
            for j in range(len(classes)):
                k = (
                    0
                    if classes[j] == "large_bowel"
                    else 1 if classes[j] == "small_bowel" else 2
                )
                if cv2 is not None:
                    pred_img = cv2.resize(
                        preds[j, :, :, k],
                        (widths[j], heights[j]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                else:
                    pil_img = Image.fromarray(
                        (preds[j, :, :, k] * 255).astype(np.uint8)
                    )
                    pil_img = pil_img.resize((widths[j], heights[j]), Image.NEAREST)
                    pred_img = np.array(pil_img)

                pred_img = (pred_img > 0.5).astype(dtype="uint8")
                pred_ids.append(id_[j])
                pred_classes.append(classes[j])
                pred_rle.append(rle_encode(pred_img))

    return pred_rle, pred_ids, pred_classes




## === cell 17
pred_rle, pred_ids, pred_classes = infer(test_df, model, batch_size=CFG.BATCH_SIZE)




## === cell 18
submission = pd.DataFrame(
    {"id": pred_ids, "class": pred_classes, "predicted": pred_rle}
)

if debug:
    sub_df = pd.read_csv(train_csv)
    del sub_df["segmentation"]
else:
    sub_df = pd.read_csv(sample_sub)
    del sub_df["predicted"]

sub_df = sub_df.merge(submission, on=["id", "class"])
sub_df.to_csv("submission.csv", index=False)

submission.sample(10)
