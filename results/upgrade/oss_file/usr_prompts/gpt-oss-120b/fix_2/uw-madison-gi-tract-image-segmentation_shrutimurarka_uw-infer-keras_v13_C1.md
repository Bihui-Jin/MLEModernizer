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

0.8099985307671078

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'The fix imports only TensorFlow/Keras (removing the conflicting standalone keras import), adds the missing `pandas`, `numpy` and backend imports, replaces the failing model load with a safe fallback dummy model, removes the `cv2` dependency by using Pillow and TensorFlow resizing, and ensures all variables (`df1`, `test_df`, `submission1`, etc.) are defined before use. These changes unblock the pipeline and generate a correctly‑formatted `submission.csv` while keeping the original logic unchanged.'

# 9. Code solution

## === cell 0
import os
import glob
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Sequence

warnings.filterwarnings("ignore")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def rle_decode(mask_rle, shape, color=1):
    s = np.array(mask_rle.split(), dtype=int)

    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths

    if len(shape) == 3:
        h, w, d = shape
        img = np.zeros((h * w, d), dtype=np.float32)
    else:
        h, w = shape
        img = np.zeros((h * w,), dtype=np.float32)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = color

    return img.reshape(shape)


def rle_encode(arr):
    arr = arr.reshape(-1)
    indexes = (np.where(arr[1:] != arr[:-1])[0]) + 1
    final = []
    one = indexes[0:-1:2]
    two = indexes[1::2]
    for start, end in zip(one, two):
        final.append(start)
        final.append(end - start)
    return " ".join(map(str, final))


def open_gray16(_path, normalize=True, to_rgb=False):
    img = Image.open(_path)
    img = np.array(img).astype(np.float32)
    if normalize:
        img = img / 65535.0
    if to_rgb:
        img = np.stack([img] * 3, axis=-1)
    else:
        img = np.expand_dims(img, -1)
    return img




## === cell 2
def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def mean_iou(y_true, y_pred):
    yt0 = y_true[:, :, :, 0]
    yp0 = K.cast(y_pred[:, :, :, 0] > 0.5, "float32")
    inter = tf.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 1)))
    union = tf.count_nonzero(tf.add(yt0, yp0))
    iou = tf.where(tf.equal(union, 0), 1.0, tf.cast(inter / union, "float32"))
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




## === cell 3
custom_objects = {
    "dice_coef": dice_coef,
    "iou_coef": iou_coef,
    "bce_dice_loss": bce_dice_loss,
}

model_path = "../input/unet-model/file/file/model_32"

try:
    model = load_model(model_path, custom_objects=custom_objects)
except Exception as e:
    inputs = tf.keras.Input(shape=(128, 128, 3))
    x = tf.keras.layers.Conv2D(
        3,
        (1, 1),
        activation="sigmoid",
        kernel_initializer="zeros",
        bias_initializer="zeros",
    )(inputs)
    model = tf.keras.Model(inputs, x)



## === cell 4
df1 = pd.read_csv("../input/uw-madison-gi-tract-image-segmentation/train.csv")
print(df1.head())



## === cell 5
test_df = pd.read_csv(
    "../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv"
)
if len(test_df) == 0:
    DEBUG = True
    test_df = df1.iloc[:300, :][["id", "class"]].copy()
    test_df["predicted"] = ""
else:
    DEBUG = False

submission1 = test_df.copy()
print(test_df.head())




## === cell 6
def preprocessing(df, subset="train"):
    df["case"] = df["id"].apply(lambda x: int(x.split("_")[0].replace("case", "")))
    df["day"] = df["id"].apply(lambda x: int(x.split("_")[1].replace("day", "")))
    df["slice"] = df["id"].apply(lambda x: x.split("_")[3])

    DIR = (
        "../input/uw-madison-gi-tract-image-segmentation/train"
        if (subset == "train") or DEBUG
        else "../input/uw-madison-gi-tract-image-segmentation/test"
    )

    all_images = glob.glob(os.path.join(DIR, "**", "*.png"), recursive=True)
    x = all_images[0].rsplit("/", 4)[0]

    path_partial_list = []
    for i in range(df.shape[0]):
        path_partial_list.append(
            os.path.join(
                x,
                f"case{df['case'].values[i]}",
                f"case{df['case'].values[i]}_day{df['day'].values[i]}",
                "scans",
                f"slice_{df['slice'].values[i]}",
            )
        )
    df["path_partial"] = path_partial_list

    tmp = []
    for p in all_images:
        tmp.append(p.rsplit("_", 4)[0])
    tmp_df = pd.DataFrame({"path_partial": tmp, "path": all_images})
    df = df.merge(tmp_df, on="path_partial", how="left").drop(columns=["path_partial"])

    df["filename"] = df["path"].apply(lambda x: x.split("/")[-1])
    df["unique_filename"] = df.apply(
        lambda row: f"{row.case}_{row.day}_{row.filename}", axis=1
    )

    df["width"] = df["path"].apply(lambda x: int(x[:-4].rsplit("_", 4)[1]))
    df["height"] = df["path"].apply(lambda x: int(x[:-4].rsplit("_", 4)[2]))
    df["px_spacing_h"] = df["path"].apply(lambda x: float(x[:-4].rsplit("_", 4)[3]))
    df["px_spacing_w"] = df["path"].apply(lambda x: float(x[:-4].rsplit("_", 4)[4][:4]))
    return df




## === cell 7
test_df = preprocessing(test_df, subset="test")
print(test_df.shape)
print(test_df.head())



## === cell 8
train_df = preprocessing(df1, subset="train")
print(train_df.shape)



## === cell 9
train_df = train_df[(train_df["case"] != 7) | (train_df["case"] != 0)].reset_index(
    drop=True
)
train_df = train_df[(train_df["case"] != 81) | (train_df["case"] != 30)].reset_index(
    drop=True
)




## === cell 10
def segment(df, subset="train"):
    df_out = pd.DataFrame({"id": df["id"][::3].reset_index(drop=True)})

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
    df_out["px_spacing_h"] = df["px_spacing_h"][::3].values
    df_out["px_spacing_w"] = df["px_spacing_w"][::3].values
    df_out["filename"] = df["filename"][::3].values
    df_out["unique_filename"] = df["unique_filename"][::3].values

    df_out.reset_index(inplace=True, drop=True)
    df_out.fillna("", inplace=True)
    if subset == "train":
        df_out["count"] = np.sum(df_out.iloc[:, 1:4] != "", axis=1).values
    return df_out


train_df = segment(train_df, subset="train")
print(train_df.head())



## === cell 11
BATCH_SIZE = 32
im_height = 128
im_width = 128




## === cell 12
class DataGenerator(Sequence):
    def __init__(self, df, batch_size=BATCH_SIZE, subset="train", shuffle=False):
        self.df = df
        self.shuffle = shuffle
        self.subset = subset
        self.batch_size = batch_size
        self.indexes = np.arange(len(df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        batch_idx = self.indexes[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        X = np.empty((len(batch_idx), im_height, im_width, 3), dtype=np.float32)
        y = np.empty((len(batch_idx), im_height, im_width, 3), dtype=np.float32)

        for i, idx in enumerate(batch_idx):
            img_path = self.df["path"].iloc[idx]
            w = self.df["width"].iloc[idx]
            h = self.df["height"].iloc[idx]
            img = self.__load_grayscale(img_path)  # (128,128,1)
            X[i] = np.repeat(img, 3, axis=-1)  # broadcast to 3 channels

            if self.subset == "train":
                for k, cls in enumerate(["large_bowel", "small_bowel", "stomach"]):
                    rle = self.df[cls].iloc[idx]
                    mask = rle_decode(rle, shape=(h, w, 1))
                    mask_resized = tf.image.resize(
                        mask, (im_height, im_width), method="nearest"
                    ).numpy()
                    y[i, :, :, k] = mask_resized.squeeze()

        if self.subset == "train":
            return X, y
        else:
            return X, self.df["id"].iloc[batch_idx].tolist()

    def __load_grayscale(self, img_path):
        img = Image.open(img_path)
        img = img.resize((im_width, im_height), Image.BILINEAR)
        img = np.array(img).astype(np.float32) / 255.0
        if img.ndim == 2:
            img = np.expand_dims(img, -1)
        return img




## === cell 13
val_generator = DataGenerator(
    test_df, batch_size=BATCH_SIZE, subset="test", shuffle=False
)



## === cell 14
print("Validation steps:", len(val_generator))
print("Batch size:", BATCH_SIZE)



## === cell 15
ids = []
classs = []
predics = []

num_batches = len(val_generator)
for i in range(num_batches):
    X, batch_ids = val_generator[i]
    preds = model.predict(X, verbose=0)  # (batch,128,128,3)

    for j, cur_id in enumerate(batch_ids):
        if cur_id not in ids:
            w = int(test_df["width"][test_df["id"] == cur_id].unique())
            h = int(test_df["height"][test_df["id"] == cur_id].unique())

            ids.extend([cur_id, cur_id, cur_id])
            classs.extend(["large_bowel", "small_bowel", "stomach"])

            for k in range(3):
                pred_mask = (
                    tf.image.resize(preds[j, :, :, k : k + 1], (h, w), method="nearest")
                    .numpy()
                    .squeeze()
                )
                pred_mask = (pred_mask > 0.5).astype(np.uint8)
                predics.append(rle_encode(pred_mask))



## === cell 16
submission_df = pd.DataFrame({"id": ids, "class": classs, "predicted": predics})
print("Submission shape:", submission_df.shape)

if submission1.shape[0] > 0:
    del submission1["predicted"]
    submission_df = submission1.merge(submission_df, on=["id", "class"])

submission_df.to_csv("submission.csv", index=False)
print("Saved submission.csv")



## === cell 17
print(submission_df.head(10))



## === cell 18
print(submission_df.sample(10).head())
