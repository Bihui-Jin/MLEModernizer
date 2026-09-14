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

0.3428370399198315

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.00242) has done: 'I fix the TensorFlow import crash by forcing the runtime to use the bundled pure-Python protobuf implementation (this resolves the `MessageFactory.GetPrototype` error in many Kaggle images). Then I remove the hard dependency on a missing external dataset model file by training the same loaded-in-script UNet-style pipeline end-to-end on the provided `train.csv` (keeping the generator/loss/resize/RLE semantics intact) and using that trained model for test predictions. Finally, I ensure the submission has exactly the same rows/order as `sample_submission.csv` by building predictions per `id` and class and writing `submission.csv` with the required columns.'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow/protobuf crash by using a compatible import strategy (try `tf.compat.v1` first, and fall back to the env var workaround), so the notebook runs reliably. Then I fix the `IndexError` in the generator by making `rle_decode` return a consistent 2D mask and handling the resize/channel dimensions correctly, which unblocks training. Finally, I correct the RLE encoding order to match the competition’s expected pixel ordering (top-to-bottom then left-to-right is column-major/Fortran order), which is a minimal but critical change that should move the score substantially toward the target without changing the core UNet/training loop.'

# 9. Code solution

## === cell 0
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import warnings

warnings.filterwarnings("ignore")



## === cell 1
import pandas as pd
import numpy as np
import gc
import cv2
from glob import glob
from tqdm import tqdm

try:
    import tensorflow.compat.v1 as tf  # type: ignore

    tf.disable_v2_behavior()
    from tensorflow import keras
    from tensorflow.keras import backend as K
except Exception:
    import tensorflow as tf  # type: ignore
    from tensorflow import keras
    from tensorflow.keras import backend as K

print("TensorFlow version:", tf.__version__)



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
sample_sub = pd.read_csv(
    "../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv"
)
test_df = pd.read_csv("../input/uw-madison-gi-tract-image-segmentation/test.csv")
train_csv = pd.read_csv("../input/uw-madison-gi-tract-image-segmentation/train.csv")

print(
    "sample_sub:",
    sample_sub.shape,
    "test_df:",
    test_df.shape,
    "train_csv:",
    train_csv.shape,
)




## === cell 4
def add_meta_and_paths(df_in, base_dir):
    df = df_in.copy()
    df.rename(columns={"class": "class_name"}, inplace=True)

    df["case"] = df["id"].apply(lambda x: int(x.split("_")[0].replace("case", "")))
    df["day"] = df["id"].apply(lambda x: int(x.split("_")[1].replace("day", "")))
    df["slice"] = df["id"].apply(lambda x: x.split("_")[3])

    all_images = glob(os.path.join(base_dir, "**", "*.png"), recursive=True)
    if len(all_images) == 0:
        raise FileNotFoundError(f"No png images found under: {base_dir}")

    x = all_images[0].rsplit("/", 4)[0]  # base folder path before caseXXX/...

    path_partial_list = []
    for i in range(df.shape[0]):
        path_partial_list.append(
            os.path.join(
                x,
                "case" + str(df["case"].values[i]),
                "case"
                + str(df["case"].values[i])
                + "_"
                + "day"
                + str(df["day"].values[i]),
                "scans",
                "slice_" + str(df["slice"].values[i]),
            )
        )
    df["path_partial"] = path_partial_list

    img_partial = [str(p.rsplit("_", 4)[0]) for p in all_images]
    tmp_df = pd.DataFrame({"path_partial": img_partial, "path": all_images})

    df = df.merge(tmp_df, on="path_partial", how="left").drop(columns=["path_partial"])
    if df["path"].isna().any():
        missing = df[df["path"].isna()].head(5)
        raise FileNotFoundError(
            f"Some image paths could not be resolved. Examples:\n{missing}"
        )

    df["width"] = df["path"].apply(lambda x: int(x[:-4].rsplit("_", 4)[1]))
    df["height"] = df["path"].apply(lambda x: int(x[:-4].rsplit("_", 4)[2]))
    return df




## === cell 5
test_df2 = add_meta_and_paths(
    test_df, "../input/uw-madison-gi-tract-image-segmentation/test"
)

train_df2 = add_meta_and_paths(
    train_csv, "../input/uw-madison-gi-tract-image-segmentation/train"
)

train_pivot = train_df2.pivot_table(
    index=["id", "path", "case", "day", "slice", "width", "height"],
    columns="class_name",
    values="segmentation",
    aggfunc="first",
).reset_index()
for c in ["large_bowel", "small_bowel", "stomach"]:
    if c not in train_pivot.columns:
        train_pivot[c] = ""
train_pivot[["large_bowel", "small_bowel", "stomach"]] = train_pivot[
    ["large_bowel", "small_bowel", "stomach"]
].fillna("")

print("train_pivot:", train_pivot.shape, "test_df2:", test_df2.shape)



## === cell 6
test_unique = test_df2.drop_duplicates(subset=["id"], keep="first").reset_index(
    drop=True
)
print("test_unique:", test_unique.shape)

gc.collect()




## === cell 7
def rle_encode(img):
    """
    img: numpy array, 1 - mask, 0 - background (2D)
    Returns run length as string formatted.
    """
    if img is None:
        return ""
    img = (img > 0).astype(np.uint8)
    pixels = img.flatten(order="F")  # competition expects column-major order
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape, color=1):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height,width)
    Returns numpy array (H, W) with values {0, color}
    """
    h, w = shape
    if mask_rle is None or mask_rle == "":
        return np.zeros((h, w), dtype=np.float32)
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros((h * w,), dtype=np.float32)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = color
    return img.reshape((h, w), order="F")




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

        X = np.empty((bs, 128, 128, 1), dtype=np.float32)
        y = None
        if self.subset == "train":
            y = np.empty((bs, 128, 128, 3), dtype=np.float32)

        for i, idx in enumerate(indexes):
            img_path = self.df["path"].iloc[idx]
            w = int(self.df["width"].iloc[idx])
            h = int(self.df["height"].iloc[idx])

            img = self.__load_grayscale(img_path)
            X[i] = img

            if self.subset == "train":
                for k, j in zip([0, 1, 2], ["large_bowel", "small_bowel", "stomach"]):
                    rles = self.df[j].iloc[idx]
                    mask2d = rle_decode(rles, shape=(h, w))
                    mask2d = cv2.resize(
                        mask2d, (128, 128), interpolation=cv2.INTER_NEAREST
                    )
                    y[i, :, :, k] = mask2d

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
def conv_block(x, filters, dropout=0.0):
    x = keras.layers.Conv2D(filters, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(filters, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    if dropout and dropout > 0:
        x = FixedDropout(dropout)(x)
    return x


def build_unet(input_shape=(128, 128, 1), n_classes=3):
    inputs = keras.Input(shape=input_shape)

    c1 = conv_block(inputs, 32, dropout=0.0)
    p1 = keras.layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, 64, dropout=0.0)
    p2 = keras.layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, 128, dropout=0.0)
    p3 = keras.layers.MaxPooling2D()(c3)

    bn = conv_block(p3, 256, dropout=0.0)

    u3 = keras.layers.UpSampling2D()(bn)
    u3 = keras.layers.Concatenate()([u3, c3])
    c6 = conv_block(u3, 128, dropout=0.0)

    u2 = keras.layers.UpSampling2D()(c6)
    u2 = keras.layers.Concatenate()([u2, c2])
    c7 = conv_block(u2, 64, dropout=0.0)

    u1 = keras.layers.UpSampling2D()(c7)
    u1 = keras.layers.Concatenate()([u1, c1])
    c8 = conv_block(u1, 32, dropout=0.0)

    outputs = keras.layers.Conv2D(n_classes, 1, activation="sigmoid")(c8)
    model = keras.Model(inputs, outputs)
    return model


model = build_unet()
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=bce_dice_loss,
    metrics=[dice_coef, iou_coef],
)
model.summary()

gc.collect()



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/454710234.py in <cell line: 0>()
     48     metrics=[dice_coef, iou_coef],
     49 )
---> 50 model.summary()
     51 
     52 gc.collect()

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/tensor_shape.py in __truediv__(self, other)
    563       TypeError.
    564     """
--> 565     raise TypeError("unsupported operand type(s) for /: 'Dimension' and '{}', "
    566                     "please use // instead".format(type(other).__name__))
    567 

TypeError: unsupported operand type(s) for /: 'Dimension' and 'int', please use // instead

## === cell 11
unique_cases = train_pivot["case"].unique()
unique_cases = np.sort(unique_cases)

fold_idx = (fold_selected - 1) % n_splits
case_folds = np.array_split(unique_cases, n_splits)
val_cases = set(case_folds[fold_idx].tolist())

train_rows = train_pivot[~train_pivot["case"].isin(val_cases)].reset_index(drop=True)
val_rows = train_pivot[train_pivot["case"].isin(val_cases)].reset_index(drop=True)

print("train_rows:", train_rows.shape, "val_rows:", val_rows.shape)

train_gen = DataGenerator(
    train_rows, batch_size=BATCH_SIZE, subset="train", shuffle=True
)
val_gen = DataGenerator(val_rows, batch_size=BATCH_SIZE, subset="train", shuffle=False)

gc.collect()



## === cell 12
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    verbose=2,
)

gc.collect()



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/3807287760.py in <cell line: 0>()
----> 1 history = model.fit(
      2     train_gen,
      3     validation_data=val_gen,
      4     epochs=EPOCHS,
      5     verbose=2,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in __iter__(self)
    501         return iterator_ops.OwnedIterator(self)
    502     else:
--> 503       raise RuntimeError("`tf.data.Dataset` only supports Python-style "
    504                          "iteration in eager mode or within tf.function.")
    505 

RuntimeError: `tf.data.Dataset` only supports Python-style iteration in eager mode or within tf.function.

## === cell 13
pred_gen = DataGenerator(test_unique, batch_size=1, subset="test", shuffle=False)
LOGITS = model.predict(pred_gen, verbose=1)

print("LOGITS:", LOGITS.shape)
gc.collect()



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/4141877404.py in <cell line: 0>()
      1 pred_gen = DataGenerator(test_unique, batch_size=1, subset="test", shuffle=False)
----> 2 LOGITS = model.predict(pred_gen, verbose=1)
      3 
      4 print("LOGITS:", LOGITS.shape)
      5 gc.collect()

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in __iter__(self)
    501         return iterator_ops.OwnedIterator(self)
    502     else:
--> 503       raise RuntimeError("`tf.data.Dataset` only supports Python-style "
    504                          "iteration in eager mode or within tf.function.")
    505 

RuntimeError: `tf.data.Dataset` only supports Python-style iteration in eager mode or within tf.function.

## === cell 14
pred_map = {}  # (id, class) -> rle

for index, row in tqdm(test_unique.iterrows(), total=test_unique.shape[0]):
    img_id = row["id"]
    h = int(row["height"])
    w = int(row["width"])
    dsize = (w, h)

    for k, cls in enumerate(["large_bowel", "small_bowel", "stomach"]):
        pred_arr = (
            cv2.resize(LOGITS[index, :, :, k], dsize, interpolation=cv2.INTER_NEAREST)
            > 0.5
        ).astype("uint8")
        pred_map[(img_id, cls)] = rle_encode(pred_arr)

del LOGITS
gc.collect()



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/80821162.py in <cell line: 0>()
      9     for k, cls in enumerate(["large_bowel", "small_bowel", "stomach"]):
     10         pred_arr = (
---> 11             cv2.resize(LOGITS[index, :, :, k], dsize, interpolation=cv2.INTER_NEAREST)
     12             > 0.5
     13         ).astype("uint8")

NameError: name 'LOGITS' is not defined

## === cell 15
sub = sample_sub.copy()
sub["predicted"] = [
    pred_map.get((rid, rcls), "")
    for rid, rcls in zip(sub["id"].values, sub["class"].values)
]

assert sub.shape[0] == sample_sub.shape[0]
sub["predicted"] = sub["predicted"].fillna("")

sub.to_csv("submission.csv", index=False)
print(sub.shape)
print(sub.head())
print("Wrote submission.csv")
