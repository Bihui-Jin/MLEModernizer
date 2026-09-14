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

0.0210097797292989

# 6. Current score

0.00474

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.00407) has done: 'I fix the TensorFlow import crash by forcing the Keras backend to use the legacy protobuf implementation before importing TF, which avoids the `MessageFactory.GetPrototype` error in Kaggle images. I also fix the missing model file issue by building the same kind of small U-Net model in-code (so the pipeline can run end-to-end without external datasets) and skipping weight loading when the file is not present. Next, I fix the prediction length mismatch by ensuring the generator yields exactly `len(df_img)` items (including the last partial batch) and by predicting in the same order/length as `df_img`. Finally, I always write a valid `submission.csv` with correct columns/row count, falling back to empty masks if anything is missing, so you get a valid submission and a non-error score.'
- What this solution (achieved 0.00474) has done: 'I fix the TensorFlow/protobuf crash by enforcing the pure-Python protobuf runtime before any TensorFlow import and by also forcing TF to use the Python implementation (this is the root cause of the `MessageFactory.GetPrototype` error). Then I make the pipeline actually learn (instead of using a random fallback model) by training the same small U-Net on the provided `train.csv` masks and images, keeping the same architecture/loss and only adding a minimal, deterministic train/validation split and a short training call. Finally, I keep the existing RLE encoding/submission assembly logic but ensure all required columns exist and paths are resolved robustly so the notebook runs end-to-end and writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import warnings

warnings.filterwarnings("ignore")

import gc
from glob import glob

import numpy as np
import pandas as pd
import cv2

from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

tf.random.set_seed(0)
np.random.seed(0)

print("TF:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
    df = df.drop(columns=["segmentation"])
    df["predicted"] = ""

df.head()



## === cell 3
df = df.rename(columns={"class": "class_name"})

df["case"] = df["id"].apply(lambda x: int(x.split("_")[0].replace("case", "")))
df["day"] = df["id"].apply(lambda x: int(x.split("_")[1].replace("day", "")))
df["slice"] = df["id"].apply(lambda x: x.split("_")[3])

if DEBUG:
    IMG_DIR = "../input/uw-madison-gi-tract-image-segmentation/train"
else:
    IMG_DIR = "../input/uw-madison-gi-tract-image-segmentation/test"

all_images = glob(os.path.join(IMG_DIR, "**", "*.png"), recursive=True)
if len(all_images) == 0:
    raise FileNotFoundError(f"No .png images found under {IMG_DIR}")

base_dir = all_images[0].rsplit("/", 4)[0]

path_partial_list = []
for i in range(df.shape[0]):
    path_partial_list.append(
        os.path.join(
            base_dir,
            f"case{df['case'].values[i]}",
            f"case{df['case'].values[i]}_day{df['day'].values[i]}",
            "scans",
            f"slice_{df['slice'].values[i]}",
        )
    )
df["path_partial"] = path_partial_list

tmp_df = pd.DataFrame(
    {
        "path_partial": [str(p.rsplit("_", 4)[0]) for p in all_images],
        "path": all_images,
    }
)

df = df.merge(tmp_df, on="path_partial", how="left").drop(columns=["path_partial"])

missing = df["path"].isna().sum()
if missing:
    raise FileNotFoundError(
        f"{missing} rows could not be matched to an image path. Check path parsing."
    )

df["width"] = df["path"].apply(lambda x: int(x[:-4].rsplit("_", 4)[1]))
df["height"] = df["path"].apply(lambda x: int(x[:-4].rsplit("_", 4)[2]))

df.head()



## === cell 4
df_img = pd.DataFrame({"id": df["id"][::3].values})
df_img["path"] = df["path"][::3].values
df_img["predicted"] = df["predicted"][::3].values if "predicted" in df.columns else ""
df_img["case"] = df["case"][::3].values
df_img["day"] = df["day"][::3].values
df_img["slice"] = df["slice"][::3].values
df_img["width"] = df["width"][::3].values
df_img["height"] = df["height"][::3].values

del df
df_img = df_img.reset_index(drop=True)
df_img = df_img.fillna("")
df_img.head()



## === cell 5
print(df_img.shape)
if DEBUG:
    df_img = df_img.sample(frac=0.05, random_state=0).reset_index(drop=True)
print(df_img.shape)
gc.collect()




## === cell 6
def rle_encode(img: np.ndarray) -> str:
    """
    img: 2D numpy array, 1 - mask, 0 - background
    Returns run length as string formatted.
    Kaggle expects 1-indexed runs in column-major order (top-to-bottom, then left-to-right),
    which corresponds to flattening in Fortran order.
    """
    img = img.astype(np.uint8)
    pixels = img.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle: str, shape, color=1) -> np.ndarray:
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height, width, channels)
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
    return img.reshape(shape, order="F")




## === cell 7
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size=BATCH_SIZE, subset="train", shuffle=False):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.shuffle = shuffle
        self.subset = subset
        self.batch_size = int(batch_size)
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

        X = np.empty((cur_bs, 128, 128, 1), dtype=np.float32)
        y = np.empty((cur_bs, 128, 128, 3), dtype=np.float32)

        for i, row_idx in enumerate(batch_indexes):
            img_path = self.df.loc[row_idx, "path"]
            w = int(self.df.loc[row_idx, "width"])
            h = int(self.df.loc[row_idx, "height"])

            img = self.__load_grayscale(img_path)
            X[i] = img

            if self.subset == "train":
                for k, j in zip([0, 1, 2], ["large_bowel", "small_bowel", "stomach"]):
                    rles = self.df.loc[row_idx, j] if j in self.df.columns else ""
                    masks = rle_decode(rles, shape=(h, w, 1))
                    masks = cv2.resize(
                        masks, (128, 128), interpolation=cv2.INTER_NEAREST
                    )
                    y[i, :, :, k] = masks[:, :, 0]

        if self.subset == "train":
            return X, y
        return X

    def __load_grayscale(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)
        return img




## === cell 8
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




## === cell 9
custom_objects = {
    "dice_coef": dice_coef,
    "iou_coef": iou_coef,
    "bce_dice_loss": bce_dice_loss,
}

MODEL_PATH = "../input/uwmgi-unet-keras/model.h5"


def build_simple_unet(input_shape=(128, 128, 1), n_classes=3):
    inputs = keras.Input(shape=input_shape)

    c1 = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    c1 = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(c1)
    p1 = keras.layers.MaxPooling2D()(c1)

    c2 = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(p1)
    c2 = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(c2)
    p2 = keras.layers.MaxPooling2D()(c2)

    c3 = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(p2)
    c3 = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(c3)

    u2 = keras.layers.UpSampling2D()(c3)
    u2 = keras.layers.Concatenate()([u2, c2])
    c4 = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(u2)
    c4 = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(c4)

    u1 = keras.layers.UpSampling2D()(c4)
    u1 = keras.layers.Concatenate()([u1, c1])
    c5 = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(u1)
    c5 = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(c5)

    outputs = keras.layers.Conv2D(n_classes, 1, padding="same", activation="sigmoid")(
        c5
    )
    model = keras.Model(inputs, outputs)
    return model


if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(
        MODEL_PATH, custom_objects=custom_objects, compile=False
    )
    print("Loaded model:", MODEL_PATH)
else:
    model = build_simple_unet()
    print(
        f"WARNING: model file not found at {MODEL_PATH}. Will train the fallback model on train.csv."
    )

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss=bce_dice_loss)

gc.collect()



## === cell 10
if not os.path.exists(MODEL_PATH):
    train_df = pd.read_csv("../input/uw-madison-gi-tract-image-segmentation/train.csv")
    train_df = train_df.rename(columns={"class": "class_name"})

    train_piv = (
        train_df.pivot_table(
            index="id", columns="class_name", values="segmentation", aggfunc="first"
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for col in ["large_bowel", "small_bowel", "stomach"]:
        if col not in train_piv.columns:
            train_piv[col] = ""

    train_piv["case"] = train_piv["id"].apply(
        lambda x: int(x.split("_")[0].replace("case", ""))
    )
    train_piv["day"] = train_piv["id"].apply(
        lambda x: int(x.split("_")[1].replace("day", ""))
    )
    train_piv["slice"] = train_piv["id"].apply(lambda x: x.split("_")[3])

    TRAIN_DIR = "../input/uw-madison-gi-tract-image-segmentation/train"
    train_images = glob(os.path.join(TRAIN_DIR, "**", "*.png"), recursive=True)
    if len(train_images) == 0:
        raise FileNotFoundError(f"No .png images found under {TRAIN_DIR}")

    train_base_dir = train_images[0].rsplit("/", 4)[0]
    train_piv["path_partial"] = train_piv.apply(
        lambda r: os.path.join(
            train_base_dir,
            f"case{int(r['case'])}",
            f"case{int(r['case'])}_day{int(r['day'])}",
            "scans",
            f"slice_{r['slice']}",
        ),
        axis=1,
    )
    train_tmp = pd.DataFrame(
        {
            "path_partial": [str(p.rsplit("_", 4)[0]) for p in train_images],
            "path": train_images,
        }
    )
    train_piv = (
        train_piv.merge(train_tmp, on="path_partial", how="left")
        .drop(columns=["path_partial"])
        .reset_index(drop=True)
    )
    miss = train_piv["path"].isna().sum()
    if miss:
        train_piv = train_piv.dropna(subset=["path"]).reset_index(drop=True)

    train_piv["width"] = train_piv["path"].apply(
        lambda x: int(x[:-4].rsplit("_", 4)[1])
    )
    train_piv["height"] = train_piv["path"].apply(
        lambda x: int(x[:-4].rsplit("_", 4)[2])
    )

    cases = np.array(sorted(train_piv["case"].unique()))
    rng = np.random.default_rng(0)
    rng.shuffle(cases)
    n_val_cases = max(1, int(0.1 * len(cases)))
    val_cases = set(cases[:n_val_cases])

    trn_df = train_piv[~train_piv["case"].isin(val_cases)].reset_index(drop=True)
    val_df = train_piv[train_piv["case"].isin(val_cases)].reset_index(drop=True)

    max_train = 2500
    max_val = 400
    if len(trn_df) > max_train:
        trn_df = trn_df.sample(n=max_train, random_state=0).reset_index(drop=True)
    if len(val_df) > max_val:
        val_df = val_df.sample(n=max_val, random_state=0).reset_index(drop=True)

    train_gen = DataGenerator(
        trn_df, batch_size=BATCH_SIZE, subset="train", shuffle=True
    )
    val_gen = DataGenerator(
        val_df, batch_size=BATCH_SIZE, subset="train", shuffle=False
    )

    history = model.fit(
        train_gen,
        validation_data=val_gen if len(val_df) else None,
        epochs=EPOCHS,
        verbose=2,
    )
    del (
        train_df,
        train_piv,
        train_images,
        train_tmp,
        trn_df,
        val_df,
        train_gen,
        val_gen,
        history,
    )
    gc.collect()



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/3545673728.py in <cell line: 0>()
     94 
     95     # Train for the configured EPOCHS (no early stopping).
---> 96     history = model.fit(
     97         train_gen,
     98         validation_data=val_gen if len(val_df) else None,

/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
     68             # To get the full stack trace, call:
     69             # `tf.debugging.disable_traceback_filtering()`
---> 70             raise e.with_traceback(filtered_tb) from None
     71         finally:
     72             del filtered_tb

/tmp/ipykernel_11/2443990020.py in __getitem__(self, index)
     40                         masks, (128, 128), interpolation=cv2.INTER_NEAREST
     41                     )
---> 42                     y[i, :, :, k] = masks[:, :, 0]
     43 
     44         if self.subset == "train":

IndexError: too many indices for array: array is 2-dimensional, but 3 were indexed

## === cell 11
pred_batches = DataGenerator(df_img, batch_size=8, subset="test", shuffle=False)
gc.collect()

LOGITS = model.predict(pred_batches, verbose=1)
gc.collect()

LOGITS = LOGITS[: len(df_img)]
print("LOGITS shape:", LOGITS.shape)



## === cell 12
lbs, sbs, sts = [], [], []

thr = 0.5

for index in tqdm(range(len(df_img)), total=len(df_img)):
    h = int(df_img.iloc[index]["height"])
    w = int(df_img.iloc[index]["width"])

    pred0 = cv2.resize(LOGITS[index, :, :, 0], (w, h), interpolation=cv2.INTER_NEAREST)
    pred1 = cv2.resize(LOGITS[index, :, :, 1], (w, h), interpolation=cv2.INTER_NEAREST)
    pred2 = cv2.resize(LOGITS[index, :, :, 2], (w, h), interpolation=cv2.INTER_NEAREST)

    pred0 = (pred0 >= thr).astype("uint8")
    pred1 = (pred1 >= thr).astype("uint8")
    pred2 = (pred2 >= thr).astype("uint8")

    lbs.append(rle_encode(pred0))
    sbs.append(rle_encode(pred1))
    sts.append(rle_encode(pred2))

del LOGITS
gc.collect()

print(len(lbs), len(sbs), len(sts))



## === cell 13
sample_sub = pd.read_csv(
    "../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv"
)

id_to_idx = {id_: i for i, id_ in enumerate(df_img["id"].values)}

predicted = []
empty_rle = ""  # empty mask allowed

for _, row in tqdm(sample_sub.iterrows(), total=len(sample_sub)):
    idx = id_to_idx.get(row["id"], None)
    if idx is None or idx >= len(lbs):
        predicted.append(empty_rle)
        continue

    if row["class"] == "large_bowel":
        predicted.append(lbs[idx])
    elif row["class"] == "small_bowel":
        predicted.append(sbs[idx])
    else:  # stomach
        predicted.append(sts[idx])

sample_sub["predicted"] = predicted
sample_sub.to_csv("submission.csv", index=False)
print(sample_sub.head())
print("Wrote submission.csv with shape:", sample_sub.shape)



## === cell 14
sub = pd.read_csv("submission.csv")
assert list(sub.columns) == ["id", "class", "predicted"]
assert sub.shape[0] == 20400
print(sub["predicted"].isna().sum(), "NaN predicted values")
print("Done.")
