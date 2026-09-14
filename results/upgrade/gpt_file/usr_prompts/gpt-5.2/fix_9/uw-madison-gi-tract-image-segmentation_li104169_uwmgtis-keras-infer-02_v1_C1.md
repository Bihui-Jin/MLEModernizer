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

0.7308678564746782

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the early import/runtime crash by removing the incompatible `tqdm.notebook` import that triggers a protobuf-related `MessageFactory.GetPrototype` error in this Kaggle environment, while keeping the rest of your stack unchanged. Then I fix the `EarlyStopping`/`cv2` `NameError`s by ensuring the needed symbols are available at call time (without changing your model/core logic). Finally, I make inference/submission robust: correct the grayscale generator’s channel dimension to match `img_size`, handle empty RLEs safely during training-mask decode, iterate through the final partial batch, and ensure the merged submission preserves all rows in `sample_submission.csv` and writes a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow import crash by forcing the pure-Python protobuf implementation *before* importing TensorFlow/Keras, which avoids the `MessageFactory.GetPrototype` error in this environment. I also remove mixed `keras` vs `tf.keras` imports (a common source of runtime incompatibilities) while keeping the exact same U-Net architecture, losses, and training semantics. Next, I make inference iterate over the full test set (your generator currently drops the last partial batch), ensuring the submission has exactly the same 20400 rows as `sample_submission.csv`. Finally, I keep the pretrained-weight loading behavior unchanged, but if weights are missing we still produce a valid `submission.csv` (all-empty masks) rather than crashing.'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation (and its required environment variable) before *any* TensorFlow import happens, which is the root cause of the `MessageFactory.GetPrototype` error. Then I ensure the script always produces a full-length (20400 rows) submission by robustly iterating through the entire `Sequence` (including the last partial batch) and by safely handling missing/empty RLEs during training-mask decode. Finally, I make pretrained-weight loading robust to common path/name mismatches in Kaggle inputs (still preferring your existing `.h5` if present), so you don’t silently fall back to an all-empty submission (which yields ~0 score) when weights exist under a slightly different filename.'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow/protobuf import crash that currently prevents any model from loading/running, by setting the necessary environment variables before TensorFlow is imported and by using a safe fallback that disables C++ protobuf if needed. Then I make the pretrained-weights loading path robust so the code actually finds and loads the provided `.h5` if it exists (otherwise you currently submit all-empty masks which scores ~0). Finally, I keep your U-Net, loss, and inference logic unchanged, only ensuring the inference loop covers the full test set and always writes a valid `submission.csv` with exactly the sample submission rows.'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf backend *before* any TensorFlow import and by adding a small safe fallback if the environment still tries to use the C++ implementation. Then I make pretrained-weight loading robust: if a `.h5` is found we load it directly (instead of relying on copying to a specific filename), preventing the current all-empty submission that yields a ~0 score. Finally, I keep your U-Net, losses, and inference logic the same but ensure inference always runs end-to-end and writes a valid `submission.csv` with exactly the `sample_submission.csv` row order.'
- What this solution (achieved 0.0) has done: 'I fix the current hard crash at import time by ensuring the protobuf/TensorFlow compatibility environment variables are set before any TensorFlow/protobuf code is imported, and by adding a safe fallback that forces the pure-Python protobuf backend if the first TensorFlow import fails. Then I make the pretrained-weight loading path more robust by searching common Kaggle input locations for a `.h5` file so you don’t silently fall back to an all-empty submission (which explains the 0.0 score). Finally, I keep your U-Net, loss, and inference logic the same while guaranteeing inference covers the full test set and always writes a valid `submission.csv` with exactly the sample submission’s 20400 rows.'
- What this solution (achieved 0.0) has done: 'I fix the import-time crash causing your pipeline to never reach training/inference (the protobuf/TensorFlow `MessageFactory.GetPrototype` issue) by forcing the safe pure‑python protobuf path *before* any TensorFlow/protobuf import and by avoiding the eager `tensorflow` import until after env is set. Then I fix the inference-time bug where `test_df` doesn’t have a `class` column (because `preprocessing()` doesn’t preserve it), which otherwise breaks `DataGenerator` for the test set and can lead to empty/invalid predictions. Finally, I keep your U-Net, losses, thresholding, and overall flow unchanged, but make weight loading deterministic (only load when found) so you don’t silently fall back to an all-empty submission (which explains the 0.0 score).'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow/protobuf import crash that stops the notebook at cell 0 by forcing the pure‑python protobuf backend and (if needed) downgrading protobuf usage via a safe environment flag before any TF import, with a guarded retry that clears partially-imported modules. Then I fix a logic bug in `preprocessing()` for the test set: it currently loses the `class` column during merge, which breaks `DataGenerator` inference and can lead to empty/invalid predictions (and a 0.0 score). Finally, I keep your U-Net, loss, thresholds, and inference semantics the same but make weight discovery/load deterministic and robust (only searching likely locations), so if weights exist they are actually used; otherwise we still emit a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import warnings

warnings.filterwarnings("ignore")

from glob import glob
from pathlib import Path
from time import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import cv2

from joblib import dump, load

from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold

try:
    import tensorflow as tf
except Exception as e:
    import sys

    for m in list(sys.modules.keys()):
        if m.startswith(("tensorflow", "google.protobuf", "keras")):
            try:
                del sys.modules[m]
            except Exception:
                pass
    import tensorflow as tf  # retry

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
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib as mpl
from matplotlib.patches import Rectangle

pd.set_option("display.max_columns", 200)
pd.set_option("display.max_colwidth", 200)

try:
    from IPython.display import display
except Exception:

    def display(x):
        print(x)


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
repertory = "/kaggle/input/"

DIR = repertory + "uw-madison-gi-tract-image-segmentation/"
TRAIN_DIR = DIR + "train"
TEST_DIR = DIR + "test"
train_csv = DIR + "train.csv"
test_csv = DIR + "test.csv"
sample_sub = DIR + "sample_submission.csv"

df_train = pd.read_csv(train_csv)
df_train.head(10)




## === cell 2
class CFG:
    BATCH_SIZE = 64
    img_size = (256, 256, 3)  # keep as in original
    n_fold = 5
    fold_selected = 1
    epochs = 100
    seed = 42
    steps_per_epoch_train = None
    steps_per_epoch_val = None


seed_everything(CFG.seed)




## === cell 3
def preprocessing(df, subset="train"):
    df = df.copy()

    if "class" in df.columns:
        df["class"] = df["class"].astype(str)

    df["case"] = df["id"].apply(lambda x: int(x.split("_")[0].replace("case", "")))
    df["day"] = df["id"].apply(lambda x: int(x.split("_")[1].replace("day", "")))
    df["slice"] = df["id"].apply(lambda x: x.split("_")[3])

    if subset == "train":
        all_images = glob(os.path.join(TRAIN_DIR, "**", "*.png"), recursive=True)
        x = all_images[0].rsplit("/", 4)[0]
    else:
        all_images = glob(os.path.join(TEST_DIR, "**", "*.png"), recursive=True)
        x = all_images[0].rsplit("/", 4)[0]

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

    path_partial_list = []
    for i in range(len(all_images)):
        path_partial_list.append(str(all_images[i].rsplit("_", 4)[0]))

    tmp_df = pd.DataFrame()
    tmp_df["path_partial"] = path_partial_list
    tmp_df["path"] = all_images

    df = pd.merge(df, tmp_df, on="path_partial", how="left").drop(
        columns=["path_partial"]
    )

    if "class_x" in df.columns:
        df["class"] = df["class_x"].astype(str)
        drop_cols = [c for c in ["class_x", "class_y"] if c in df.columns]
        df = df.drop(columns=drop_cols)

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
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    if mask_rle is None:
        return np.zeros(shape[:2], dtype=np.uint8)
    if isinstance(mask_rle, float) and np.isnan(mask_rle):
        return np.zeros(shape[:2], dtype=np.uint8)
    if isinstance(mask_rle, str) and mask_rle.strip() == "":
        return np.zeros(shape[:2], dtype=np.uint8)

    s = np.asarray(mask_rle.split(), dtype=int)
    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape[0], shape[1])


def rle_encode(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    """
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
        rle = ctrain_df.segmentation.squeeze() if len(ctrain_df) else ""
        if (
            len(ctrain_df)
            and not (isinstance(rle, float) and np.isnan(rle))
            and str(rle).strip() != ""
        ):
            mask[..., i] = rle_decode(rle, shape[:2])
    return mask


def rgb2gray(mask):
    pad_mask = np.pad(mask, pad_width=[(0, 0), (0, 0), (1, 0)])
    gray_mask = pad_mask.argmax(-1)
    return gray_mask


def gray2rgb(mask):
    rgb_mask = tf.keras.utils.to_categorical(mask, num_classes=4)
    return rgb_mask[..., 1:].astype(mask.dtype)




## === cell 7
def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = img.astype("float32")  # original is uint16
    denom = img.max() - img.min()
    if denom < 1e-6:
        img = np.zeros_like(img, dtype=np.uint8)
    else:
        img = (img - img.min()) / denom * 255.0
        img = img.astype("uint8")
    return img


def show_img(img, mask=None):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    plt.imshow(img, cmap="bone")

    if mask is not None:
        plt.imshow(mask, alpha=0.5)
        handles = [
            Rectangle((0, 0), 1, 1, color=_c)
            for _c in [(0.667, 0.0, 0.0), (0.0, 0.667, 0.0), (0.0, 0.0, 0.667)]
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

    df_out = df_out.reset_index(drop=True)
    df_out = df_out.fillna("")

    if subset == "train":
        df_out["count"] = np.sum(df_out.iloc[:, 1:4] != "", axis=1).values

    display(df_out.sample(5))
    return df_out




## === cell 9
DF_train = restructure(train_df, subset="train")




## === cell 10
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
        self.df = df
        self.shuffle = shuffle
        self.subset = subset
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.indexes = np.arange(len(df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        current_bs = len(indexes)

        X = np.empty(
            (current_bs, self.img_shape[0], self.img_shape[1], self.img_shape[2]),
            dtype=np.float32,
        )
        y = np.empty(
            (current_bs, self.img_shape[0], self.img_shape[1], self.img_shape[2]),
            dtype=np.float32,
        )

        id_, heights, widths, classes = [], [], [], []

        for i, img_path in enumerate(self.df["path"].iloc[indexes]):
            if self.subset != "train":
                id_.append(self.df["id"].iloc[indexes[i]])
                heights.append(int(self.df["height"].iloc[indexes[i]]))
                widths.append(int(self.df["width"].iloc[indexes[i]]))
                classes.append(self.df["class"].iloc[indexes[i]])

            w = int(self.df["width"].iloc[indexes[i]])
            h = int(self.df["height"].iloc[indexes[i]])

            img = self.__load_grayscale(img_path)
            X[i] = img

            if self.subset == "train":
                for k, j in enumerate(["large_bowel", "small_bowel", "stomach"]):
                    rles = self.df[j].iloc[indexes[i]]
                    mask = rle_decode(rles, shape=(h, w))
                    mask = cv2.resize(
                        mask, self.img_shape[0:2], interpolation=cv2.INTER_NEAREST
                    )
                    y[i, :, :, k] = mask.astype(np.float32)

        if self.subset == "train":
            return X, y
        else:
            return X, id_, widths, heights, classes

    def __load_grayscale(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
        dsize = self.img_shape[0:2]
        img = cv2.resize(img, dsize, interpolation=cv2.INTER_LINEAR)
        img = img.astype("float32")
        denom = img.max() - img.min()
        if denom < 1e-6:
            img = np.zeros_like(img, dtype=np.float32)
        else:
            img = (img - img.min()) / denom  # scale to [0,1]
        img = np.expand_dims(img, axis=-1)  # (H,W,1)
        if self.img_shape[2] == 3:
            img = np.repeat(img, 3, axis=-1)
        return img.astype(np.float32)




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
    return binary_crossentropy(tf.cast(y_true, tf.float32), y_pred) + 0.5 * dice_loss(
        tf.cast(y_true, tf.float32), y_pred
    )




## === cell 12
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
    plt.plot(losses["dice_coef"].index, losses["dice_coef"], label="Train_dice_coef")
    plt.plot(
        losses["val_dice_coef"].index, losses["val_dice_coef"], label="Val_dice_coef"
    )
    plt.title("DICE")
    plt.xlabel("Epoch")
    plt.ylabel("dice_coef")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(losses["iou_coef"].index, losses["iou_coef"], label="Train_iou_coef")
    plt.plot(losses["val_iou_coef"].index, losses["val_iou_coef"], label="Val_iou_coef")
    plt.title("IOU")
    plt.xlabel("Epoch")
    plt.ylabel("iou_coef")
    plt.legend()
    plt.show()


def fit_model(
    model,
    model_name,
    train_dataset,
    validation_dataset,
    model_path_override=None,
    score_path_override=None,
):
    model_path = model_path_override or (models_path + str(model_name) + ".h5")
    score_path = score_path_override or (
        results_path + "score_" + str(model_name) + ".joblib"
    )

    if model_path is not None and os.path.isfile(model_path):
        model = load_model(
            model_path,
            custom_objects={
                "bce_dice_loss": bce_dice_loss,
                "iou_coef": iou_coef,
                "dice_coef": dice_coef,
            },
        )
        if score_path is not None and os.path.isfile(score_path):
            results = load(score_path)
            plot_train(results)
        else:
            results = None
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




## === cell 13
path_load_infer = repertory + "uwmgtis-keras-train-02/"
models_path = path_load_infer
results_path = path_load_infer

input_shape = CFG.img_size
model = build_unet(input_shape, dropout=True, batchnorm=True)

candidate_model_files = ["U-net.h5", "U-Net.h5", "unet.h5", "model.h5"]
candidate_score_files = [
    "score_U-net.joblib",
    "score_U-Net.joblib",
    "score_unet.joblib",
    "score_model.joblib",
]


def find_first_existing(base_dir, filenames):
    if base_dir is not None:
        for fn in filenames:
            p = os.path.join(base_dir, fn)
            if os.path.isfile(p):
                return p
        for fn in filenames:
            hits = glob(os.path.join(base_dir, "**", fn), recursive=True)
            hits = [h for h in hits if os.path.isfile(h)]
            if len(hits):
                hits.sort()
                return hits[0]
    return None


def find_any_h5_in_kaggle_inputs():
    roots = [
        "/kaggle/input/uwmgtis-keras-train-02",
        "/kaggle/input/uw-madison-gi-tract-image-segmentation",
    ]
    hits = []
    for r in roots:
        if os.path.isdir(r):
            hits.extend(glob(os.path.join(r, "**", "*.h5"), recursive=True))
    hits = [h for h in hits if os.path.isfile(h)]
    hits.sort()
    return hits[0] if hits else None


found_model_path = find_first_existing(models_path, candidate_model_files)
found_score_path = find_first_existing(results_path, candidate_score_files)

if found_model_path is None:
    found_model_path = find_any_h5_in_kaggle_inputs()

have_any_weights = found_model_path is not None
print(
    "Found pretrained:",
    have_any_weights,
    "| model:",
    found_model_path,
    "| scores:",
    found_score_path,
)



## === cell 14
if have_any_weights:
    model, results = fit_model(
        model,
        "U-net",
        None,
        None,
        model_path_override=found_model_path,
        score_path_override=found_score_path,
    )
else:
    results = None



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

assert (
    "class" in test_df.columns
), "preprocessing() must preserve 'class' for test inference"
test_df.head(5)




## === cell 16
def infer(DF, model, batch_size=CFG.BATCH_SIZE):
    pred_rle = []
    pred_ids = []
    pred_classes = []

    DF_batch = DataGenerator(DF, batch_size=batch_size, subset="test", shuffle=False)

    for batch_idx in tqdm(range(len(DF_batch)), total=len(DF_batch)):
        img, id_, widths, heights, classes = DF_batch[batch_idx]
        preds = model.predict(img, verbose=0)

        for j in range(len(id_)):
            k = (
                0
                if classes[j] == "large_bowel"
                else 1 if classes[j] == "small_bowel" else 2
            )

            pred_img = cv2.resize(
                preds[j, :, :, k],
                (int(widths[j]), int(heights[j])),
                interpolation=cv2.INTER_NEAREST,
            )
            pred_img = (pred_img > 0.5).astype("uint8")

            pred_ids.append(id_[j])
            pred_classes.append(classes[j])
            pred_rle.append(rle_encode(pred_img))

    return pred_rle, pred_ids, pred_classes




## === cell 17
CFG.BATCH_SIZE = 3

if have_any_weights:
    pred_rle, pred_ids, pred_classes = infer(test_df, model, batch_size=CFG.BATCH_SIZE)
else:
    pred_ids = test_df["id"].tolist()
    pred_classes = test_df["class"].tolist()
    pred_rle = [""] * len(test_df)

print("Pred rows:", len(pred_rle), "Expected:", len(test_df))



## === cell 18
submission = pd.DataFrame(
    {"id": pred_ids, "class": pred_classes, "predicted": pred_rle}
)

sub_df = pd.read_csv(sample_sub).copy()
sub_df = sub_df.drop(columns=["predicted"])
sub_df = sub_df.merge(submission, on=["id", "class"], how="left")
sub_df["predicted"] = sub_df["predicted"].fillna("")
sub_df = sub_df[["id", "class", "predicted"]]

sample_order = pd.read_csv(sample_sub)[["id", "class"]]
sub_df = sample_order.merge(sub_df, on=["id", "class"], how="left")
sub_df["predicted"] = sub_df["predicted"].fillna("")
sub_df = sub_df[["id", "class", "predicted"]]

assert list(sub_df.columns) == ["id", "class", "predicted"]
assert len(sub_df) == len(pd.read_csv(sample_sub))

sub_df.to_csv("submission.csv", index=False)

display(sub_df.head())
print("Saved submission.csv with shape:", sub_df.shape)
print("Null predicted:", sub_df["predicted"].isna().sum())
print("Empty predicted:", (sub_df["predicted"].astype(str).str.len() == 0).sum())
