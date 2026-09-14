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

0.4705569229227646

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the runtime failures by (1) removing the conflicting standalone `keras` imports that trigger the protobuf `MessageFactory.GetPrototype` error and standardizing everything on `tf.keras`, (2) fixing the data generator’s image channel/shape bugs so inference runs, and (3) replacing the deprecated `predict_generator` call with `model.predict`. I also make the model-loading robust by using the correct `tf.keras.models.load_model` symbol and falling back safely if the external model file isn’t present (so a submission is always produced). Finally, I fix the resize argument order and ensure the RLE encoding uses the required column-wise (Fortran-order) flattening so the submission is valid and scores correctly rather than near-zero due to an encoding mismatch.'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow/protobuf import crash that prevents the notebook from running by enforcing the pure-TensorFlow Keras path and setting a safe protobuf implementation before TensorFlow loads. I also correct a key logic bug where the test dataframe never has the per-class columns needed by the generator (so masks were silently wrong/empty), by building `df_train` directly from `test.csv` / `sample_submission.csv` and then attaching paths and metadata consistently. Finally, I keep your model/inference/RLE core logic intact, but ensure prediction length matches the full test set (no dropped remainder) so the submission aligns exactly and scores above 0 instead of effectively empty/misaligned predictions.'

# 9. Code solution

## === cell 0
import os, random, gc, warnings

warnings.filterwarnings("ignore")

os.environ["PYTHONHASHSEED"] = "42"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

random.seed(42)



## === cell 1
import pandas as pd
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
BATCH_SIZE = 16
EPOCHS = 30
n_splits = 5
fold_selected = 1  # 1..5



## === cell 3
BASE_PATH = "../input/uw-madison-gi-tract-image-segmentation"
DEBUG = False

test_df = pd.read_csv(f"{BASE_PATH}/test.csv")
sample_sub = pd.read_csv(f"{BASE_PATH}/sample_submission.csv")

df_train = test_df[["id"]].drop_duplicates().reset_index(drop=True)

df_train["case"] = df_train["id"].apply(
    lambda x: int(x.split("_")[0].replace("case", ""))
)
df_train["day"] = df_train["id"].apply(
    lambda x: int(x.split("_")[1].replace("day", ""))
)
df_train["slice"] = df_train["id"].apply(lambda x: x.split("_")[3])

TRAIN_DIR = f"{BASE_PATH}/test"

all_test_images = glob(os.path.join(TRAIN_DIR, "**", "*.png"), recursive=True)
if len(all_test_images) == 0:
    raise RuntimeError(f"No png images found under {TRAIN_DIR}")

x = all_test_images[0].rsplit("/", 4)[0]  # ../input/.../test

df_train["path_partial"] = [
    os.path.join(
        x,
        "case" + str(c),
        "case" + str(c) + "_" + "day" + str(d),
        "scans",
        "slice_" + str(s),
    )
    for c, d, s in zip(
        df_train["case"].values, df_train["day"].values, df_train["slice"].values
    )
]

tmp_df = pd.DataFrame(
    {
        "path_partial": [str(p.rsplit("_", 4)[0]) for p in all_test_images],
        "path": all_test_images,
    }
)

df_train = df_train.merge(tmp_df, on="path_partial", how="left").drop(
    columns=["path_partial"]
)
missing = df_train["path"].isna().sum()
if missing:
    raise RuntimeError(
        f"Failed to match {missing} test ids to image paths. Check id parsing/path building."
    )

df_train["width"] = df_train["path"].apply(lambda p: int(p[:-4].rsplit("_", 4)[1]))
df_train["height"] = df_train["path"].apply(lambda p: int(p[:-4].rsplit("_", 4)[2]))

df_train.fillna("", inplace=True)
df_train.head(5)



## === cell 4
print(df_train.shape)
if DEBUG:
    df_train = df_train.sample(frac=0.05, random_state=42).reset_index(drop=True)
print(df_train.shape)



## === cell 5
gc.collect()




## === cell 6
def rle_encode(img):
    """
    img: 2D numpy array, 1 - mask, 0 - background
    Returns run length as string formatted.
    """
    img = img.astype(np.uint8)
    pixels = img.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape, color=1):
    """
    mask_rle: run-length string (start length)
    shape: (height, width, channels)
    Returns numpy array with given shape.
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
        mask = np.zeros((height, width, 3), dtype=np.float32)
        for label in labels:
            mask += rle_decode(label, shape=(height, width, 3), color=np.random.rand(3))
    else:
        mask = np.zeros((height, width, 1), dtype=np.float32)
        for label in labels:
            mask += rle_decode(label, shape=(height, width, 1))
    mask = mask.clip(0, 1)
    return mask




## === cell 7
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size=BATCH_SIZE, subset="train", shuffle=False):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.shuffle = shuffle
        self.subset = subset
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        if self.subset == "test":
            return int(np.ceil(len(self.df) / self.batch_size))
        return int(np.floor(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.df))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        batch_indexes = self.indexes[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        bs = len(batch_indexes)

        X = np.empty((bs, 128, 128, 3), dtype=np.float32)
        y = np.empty((bs, 128, 128, 3), dtype=np.float32)

        for i, idx in enumerate(batch_indexes):
            img_path = self.df["path"].iloc[idx]
            w = int(self.df["width"].iloc[idx])
            h = int(self.df["height"].iloc[idx])

            img = self.__load_grayscale(img_path)  # (128,128,1)
            img3 = np.repeat(img, 3, axis=-1)  # (128,128,3)
            X[i] = img3

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
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)
        return img




## === cell 8
gc.collect()



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
    return binary_crossentropy(y_true, y_pred) + 0.5 * dice_loss(y_true, y_pred)


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




## === cell 10
gc.collect()



## === cell 11
custom_objects = {
    "FixedDropout": FixedDropout,
    "dice_coef": dice_coef,
    "iou_coef": iou_coef,
    "bce_dice_loss": bce_dice_loss,
}

MODEL_PATH = "../input/uwmgi-unet-keras/model.h5"
model = None
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
else:
    model = None

gc.collect()



## === cell 12
pred_batches = DataGenerator(df_train, batch_size=1, subset="test", shuffle=False)
gc.collect()

if model is not None:
    LOGITS = model.predict(pred_batches, verbose=1)
else:
    LOGITS = np.zeros((len(df_train), 128, 128, 3), dtype=np.float32)

if LOGITS.shape[0] != len(df_train):
    LOGITS = LOGITS[: len(df_train)]

gc.collect()



## === cell 13
len(LOGITS)



## === cell 14
lbs, sbs, sts = [], [], []
for index, row in tqdm(df_train.iterrows(), total=df_train.shape[0]):
    h = int(df_train.iloc[index]["height"])
    w = int(df_train.iloc[index]["width"])

    pred0 = cv2.resize(LOGITS[index, :, :, 0], (w, h), interpolation=cv2.INTER_NEAREST)
    pred1 = cv2.resize(LOGITS[index, :, :, 1], (w, h), interpolation=cv2.INTER_NEAREST)
    pred2 = cv2.resize(LOGITS[index, :, :, 2], (w, h), interpolation=cv2.INTER_NEAREST)

    pred0 = (pred0 >= 0.5).astype("uint8")
    pred1 = (pred1 >= 0.5).astype("uint8")
    pred2 = (pred2 >= 0.5).astype("uint8")

    lbs.append(rle_encode(pred0))
    sbs.append(rle_encode(pred1))
    sts.append(rle_encode(pred2))

del LOGITS
gc.collect()



## === cell 15
df_train = df_train[["id"]].copy()
gc.collect()



## === cell 16
ids, classes, rles = [], [], []
for index, row in tqdm(df_train.iterrows(), total=df_train.shape[0]):
    ids.extend([row["id"]] * 3)
    classes.extend(["large_bowel", "small_bowel", "stomach"])
    rles.extend([lbs[index], sbs[index], sts[index]])



## === cell 17
sub = pd.DataFrame({"id": ids, "class": classes, "predicted": rles})
sub.to_csv("submission.csv", index=False)

assert sub.shape[0] == 3 * df_train.shape[0]
assert set(sub.columns) == {"id", "class", "predicted"}
print(sub.head())



## === cell 18
sub.head()
