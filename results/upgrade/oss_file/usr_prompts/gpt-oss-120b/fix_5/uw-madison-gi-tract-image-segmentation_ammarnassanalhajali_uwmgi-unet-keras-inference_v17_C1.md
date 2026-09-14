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

- What this solution (achieved 0.0) has done: 'I guard TensorFlow/Keras imports, skip model loading if unavailable, and fall back to a zero‑mask prediction so the pipeline runs end‑to‑end and produces a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.0) has done: 'I fixed the missing imports, replaced the fragile path‑building/merge logic with a reliable per‑row lookup for the image files, added safe handling for width/height extraction, and provided a fallback `binary_crossentropy` when TensorFlow isn’t available. These changes resolve the NameError and KeyError issues, allowing the pipeline to run end‑to‑end and produce a correct `submission.csv` while keeping the original modeling logic intact.'
- What this solution (achieved 0.0) has done: 'I bypass TensorFlow import to avoid the protobuf AttributeError and add a safety check when resizing predictions so that zero height/width values default to the model’s 128 × 128 size. This keeps the original pipeline logic, ensures a valid `submission.csv` is written, and moves the score from 0 toward the target without altering core modeling.'

# 9. Code solution

## === cell 0
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os
import gc
import cv2
from tqdm import tqdm
from glob import glob

tf = None
keras = None
K = None
load_model = None

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import backend as K
    from tensorflow.keras.models import load_model
    from tensorflow.keras.losses import binary_crossentropy
except Exception as e:
    tf = None
    keras = None
    K = None
    load_model = None

    def binary_crossentropy(y_true, y_pred):
        return np.mean(
            -(y_true * np.log(y_pred + 1e-7) + (1 - y_true) * np.log(1 - y_pred + 1e-7))
        )




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
    df.pop("segmentation")
    df["predicted"] = ""

df.rename(columns={"class": "class_name"}, inplace=True)
df["case"] = df["id"].apply(lambda x: int(x.split("_")[0].replace("case", "")))
df["day"] = df["id"].apply(lambda x: int(x.split("_")[1].replace("day", "")))
df["slice"] = df["id"].apply(lambda x: x.split("_")[3])

TRAIN_DIR = (
    "../input/uw-madison-gi-tract-image-segmentation/test"
    if not DEBUG
    else "../input/uw-madison-gi-tract-image-segmentation/train"
)


def locate_image(row):
    folder = os.path.join(
        TRAIN_DIR,
        f"case{row['case']}",
        f"case{row['case']}_day{row['day']}",
        "scans",
    )
    candidates = glob(os.path.join(folder, "*.png"))
    return candidates[0] if candidates else ""


df["path"] = df.apply(locate_image, axis=1)


def parse_dim(p):
    if not p:
        return 0, 0
    name = os.path.basename(p)[:-4]  # strip .png
    parts = name.rsplit("_", 4)
    if len(parts) >= 3:
        try:
            w = int(parts[-4])
            h = int(parts[-3])
            return w, h
        except:
            pass
    return 0, 0


df[["width", "height"]] = df["path"].apply(lambda p: pd.Series(parse_dim(p)))



## === cell 3
df_train = pd.DataFrame(
    {
        "id": df["id"][::3].reset_index(drop=True),
        "path": df["path"][::3].values,
        "predicted": df["predicted"][::3].values,
        "case": df["case"][::3].values,
        "day": df["day"][::3].values,
        "slice": df["slice"][::3].values,
        "width": df["width"][::3].values,
        "height": df["height"][::3].values,
    }
)
df_train.fillna("", inplace=True)
df_train.reset_index(inplace=True, drop=True)



## === cell 4
print(df_train.shape)
if DEBUG:
    df_train = df_train.sample(frac=0.05).reset_index(drop=True)
print(df_train.shape)



## === cell 5
gc.collect()




## === cell 6
def rle_encode(img):
    """Encode binary mask to RLE string."""
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape, color=1):
    """Decode RLE string to binary mask."""
    if not mask_rle:
        return np.zeros(shape, dtype=np.float32)
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.float32)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = color
    return img.reshape((shape[0], shape[1]))




## === cell 7
class DataGenerator(tf.keras.utils.Sequence if tf else object):
    def __init__(self, df, batch_size=BATCH_SIZE, subset="train", shuffle=False):
        self.df = df
        self.batch_size = batch_size
        self.subset = subset
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        X = np.empty((self.batch_size, 128, 128, 3), dtype=np.float32)
        y = np.empty((self.batch_size, 128, 128, 3), dtype=np.float32)
        batch_idxs = self.indexes[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        for i, idx in enumerate(batch_idxs):
            w = self.df["width"].iloc[idx]
            h = self.df["height"].iloc[idx]
            img = self.__load_grayscale(self.df["path"].iloc[idx])
            X[i] = img
            if self.subset == "train":
                for k, organ in enumerate(["large_bowel", "small_bowel", "stomach"]):
                    rle = self.df[organ].iloc[idx]
                    mask = rle_decode(rle, shape=(h, w))
                    mask = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_NEAREST)
                    y[i, :, :, k] = mask
        return (X, y) if self.subset == "train" else X

    def __load_grayscale(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
        img = cv2.resize(img, (128, 128))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)
        img = np.repeat(img, 3, axis=-1)  # make 3 channels
        return img




## === cell 8
if K is not None:

    def dice_coef(y_true, y_pred, smooth=1):
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


if keras is not None:

    class FixedDropout(keras.layers.Dropout):
        def _get_noise_shape(self, inputs):
            if self.noise_shape is None:
                return self.noise_shape
            symbolic_shape = K.shape(inputs)
            return tuple(
                symbolic_shape[axis] if shape is None else shape
                for axis, shape in enumerate(self.noise_shape)
            )

else:
    FixedDropout = None



## === cell 9
custom_objects = {}
if FixedDropout is not None:
    custom_objects.update(
        {
            "FixedDropout": FixedDropout,
            "dice_coef": dice_coef,
            "iou_coef": iou_coef,
            "bce_dice_loss": bce_dice_loss,
        }
    )
if load_model is not None:
    try:
        model = load_model(
            "../input/uwmgi-unet-keras/model.h5", custom_objects=custom_objects
        )
    except Exception as e:
        print("Model load failed:", e)
        model = None
else:
    model = None
gc.collect()



## === cell 10
if model is not None:
    pred_batches = DataGenerator(df_train, batch_size=1, subset="test", shuffle=False)
    LOGITS = model.predict(pred_batches, verbose=1)
else:
    LOGITS = np.zeros((len(df_train), 128, 128, 3), dtype=np.float32)

gc.collect()



## === cell 11
print("Logits shape:", LOGITS.shape)



## === cell 12
lbs, sbs, sts = [], [], []
for idx in tqdm(range(df_train.shape[0]), total=df_train.shape[0]):
    root_shape = (df_train.iloc[idx]["height"], df_train.iloc[idx]["width"])
    if root_shape[0] == 0 or root_shape[1] == 0:
        target_h, target_w = 128, 128
    else:
        target_h, target_w = root_shape
    for channel, lst in zip([0, 1, 2], [lbs, sbs, sts]):
        pred_arr = np.round(
            cv2.resize(
                LOGITS[idx, :, :, channel],
                (target_w, target_h),
                interpolation=cv2.INTER_NEAREST,
            )
        ).astype("uint8")
        lst.append(rle_encode(pred_arr))

del LOGITS
gc.collect()



## === cell 13
submission_ids = []
submission_classes = []
submission_rles = []
for idx, row in df_train.iterrows():
    submission_ids.extend([row["id"]] * 3)
    submission_classes.extend(["large_bowel", "small_bowel", "stomach"])
    submission_rles.extend([lbs[idx], sbs[idx], sts[idx]])

df_sub = pd.DataFrame(
    {"id": submission_ids, "class": submission_classes, "predicted": submission_rles}
)

assert len(df_sub) == len(submission_ids) == len(submission_rles)

df_sub.to_csv("submission.csv", index=False)
print("Saved submission.csv with", df_sub.shape[0], "rows.")
