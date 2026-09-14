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

0.3344721353175812

# 6. Current score

0.07411

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.24608) has done: 'I fix the import/runtime failures by removing the conflicting `tensorflow`/`keras` mixed imports that trigger the protobuf `MessageFactory.GetPrototype` error, and by consistently using `tf.keras` APIs (including `load_model`). I also fix the data generator to output 3-channel images (your current code allocates `(…,128,128,3)` but loads only 1 channel), and correct dtype scaling to `float32`. For inference, I replace deprecated `predict_generator` with `model.predict`, ensure predictions align with the number of test images, and fix the `cv2.resize` argument order (OpenCV expects `(width,height)`), which was producing wrong shapes and downstream indexing issues. Finally, I generate the submission by starting from `sample_submission.csv` and filling `predicted` in the correct row order to guarantee a valid 20400-row submission CSV.'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow import crash by setting the protobuf implementation to the pure-Python fallback before importing `tensorflow`, which resolves the `MessageFactory.GetPrototype` error in Kaggle environments. Then I fix the missing pretrained model path by searching for an available `.h5/.keras` model under `../input` and load it; if none is found, the script fall back to producing a valid (empty-mask) submission rather than crashing, ensuring end-to-end execution. I also make the `DataGenerator` robust to missing/NaN class columns during test-time inference (so it won’t KeyError), while preserving the same preprocessing and inference logic. Finally, I always write a correctly formatted `submission.csv` (20400 rows, columns `id,class,predicted`) aligned to `sample_submission.csv`.'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow/protobuf crash by setting the protobuf Python implementation *and* forcing the pure-Python backend before TensorFlow is imported, plus adding a safe fallback to continue even if TF still cannot import (so you always get a valid `submission.csv`). I also fix an inference-length bug: your `DataGenerator.__len__` drops the remainder, causing fewer predictions than test images and leaving most rows blank, which can yield a near-zero score; switching to `ceil` and predicting without an explicit `steps` makes predictions cover the full test set. Finally, I keep your model, preprocessing, and RLE logic intact while making the submission mapping robust to ensure all 20400 rows are filled in the correct order.'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow/protobuf import crash that currently prevents any model inference by setting the additional protobuf/TensorFlow env flags that are needed in Kaggle’s TF+protobuf combination and ensuring they are applied before importing TensorFlow. If TensorFlow still cannot import, the script still complete and write a valid `submission.csv`, but with the TF import fixed it should load the pretrained model and generate non-empty predictions, improving the score from 0.0 toward your target. I keep your model loading, generator, resizing, thresholding (rounding), and submission mapping logic the same, only touching what’s necessary for runtime stability and end-to-end output. I also add a safe fallback to guarantee the `predicted` column is fully populated (never NaN) and the submission has exactly the sample submission’s row order.'
- What this solution (achieved 0.0) has done: 'I fix the TensorFlow/protobuf crash that currently prevents any model from running by setting the additional environment flags that are required in Kaggle’s TF+protobuf combination *before* importing TensorFlow. To keep end-to-end execution guaranteed, I also keep a safe fallback path that still writes a valid `submission.csv` even if TensorFlow cannot import or a model file cannot be found. I preserve your core inference logic (generator → `model.predict` → resize → `np.round` → RLE) and only make minimal robustness fixes around imports and submission completeness so you don’t end up with missing/NaN predictions that can tank the score. No architecture/training logic is changed.'
- What this solution (achieved 0.07411) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf backend earlier and adding the additional env flags commonly required in Kaggle’s TF+protobuf combos, so the model can actually load and run inference (instead of producing empty masks and scoring 0.0). I also add a safe, score-positive fallback: if TF still can’t import or no model is found, the script generate a simple non-empty segmentation (via image-threshold-based mask) rather than all-empty RLEs, which should move the score up toward your target while keeping the rest of your pipeline (paths, resizing, RLE, submission mapping) intact. Finally, I keep the submission aligned to `sample_submission.csv` and ensure every row has a valid string in `predicted`.'

# 9. Code solution

## === cell 0
import os, random
import numpy as np

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_enable_xla_devices=false")

seed = 42
random.seed(seed)
np.random.seed(seed)

TF_AVAILABLE = False
TF_IMPORT_ERROR = None
tf = None

try:
    import tensorflow as tf  # noqa: E402

    tf.random.set_seed(seed)
    TF_AVAILABLE = True
except Exception as e:
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = repr(e)
    tf = None
    print("WARNING: TensorFlow failed to import; will fall back to heuristic masks.")
    print("TF import error:", TF_IMPORT_ERROR)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import gc
import cv2
from glob import glob
from tqdm import tqdm
import math

if TF_AVAILABLE:
    from tensorflow.keras import backend as K
    from tensorflow.keras.models import load_model
    from tensorflow.keras.losses import binary_crossentropy



## === cell 2
BATCH_SIZE = 16
EPOCHS = 30
n_splits = 5
fold_selected = 1  # 1..5



## === cell 3
df = pd.read_csv(
    "../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv"
)
DEBUG = False
if df.shape[0] == 0:
    DEBUG = True
if DEBUG == True:
    df = pd.read_csv("../input/uw-madison-gi-tract-image-segmentation/train.csv")
    df.pop("segmentation")
    df["predicted"] = ""



## === cell 4
df.rename(columns={"class": "class_name"}, inplace=True)
df["case"] = df["id"].apply(lambda x: int(x.split("_")[0].replace("case", "")))
df["day"] = df["id"].apply(lambda x: int(x.split("_")[1].replace("day", "")))
df["slice"] = df["id"].apply(lambda x: x.split("_")[3])

if DEBUG:
    TRAIN_DIR = "../input/uw-madison-gi-tract-image-segmentation/train"
else:
    TRAIN_DIR = "../input/uw-madison-gi-tract-image-segmentation/test"

all_train_images = glob(os.path.join(TRAIN_DIR, "**", "*.png"), recursive=True)

if len(all_train_images) == 0:
    raise RuntimeError(f"No PNG images found under {TRAIN_DIR}. Check dataset path.")

x = all_train_images[0].rsplit("/", 4)[0]

path_partial_list = []
for i in range(0, df.shape[0]):
    path_partial_list.append(
        os.path.join(
            x,
            "case" + str(df["case"].values[i]),
            "case" + str(df["case"].values[i]) + "_" + "day" + str(df["day"].values[i]),
            "scans",
            "slice_" + str(df["slice"].values[i]),
        )
    )
df["path_partial"] = path_partial_list

path_partial_list = []
for i in range(0, len(all_train_images)):
    path_partial_list.append(str(all_train_images[i].rsplit("_", 4)[0]))

tmp_df = pd.DataFrame()
tmp_df["path_partial"] = path_partial_list
tmp_df["path"] = all_train_images

df = df.merge(tmp_df, on="path_partial").drop(columns=["path_partial"])
df["width"] = df["path"].apply(lambda x: int(x[:-4].rsplit("_", 4)[1]))
df["height"] = df["path"].apply(lambda x: int(x[:-4].rsplit("_", 4)[2]))

del x, path_partial_list, tmp_df
df.head(5)



## === cell 5
df_train = pd.DataFrame({"id": df["id"][::3]})
df_train["path"] = df["path"][::3].values
df_train["predicted"] = df["predicted"][::3].values
df_train["case"] = df["case"][::3].values
df_train["day"] = df["day"][::3].values
df_train["slice"] = df["slice"][::3].values
df_train["width"] = df["width"][::3].values
df_train["height"] = df["height"][::3].values

del df
df_train.reset_index(inplace=True, drop=True)
df_train.fillna("", inplace=True)
df_train.head(5)



## === cell 6
print(df_train.shape)
if DEBUG:
    df_train = df_train.sample(frac=0.05).reset_index(drop=True)
print(df_train.shape)



## === cell 7
gc.collect()




## === cell 8
def rle_encode(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted.
    """
    pixels = img.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape, color=1):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height,width,channels) of array to return
    Returns numpy array.
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




## === cell 9
if TF_AVAILABLE:

    class DataGenerator(tf.keras.utils.Sequence):
        def __init__(self, df, batch_size=BATCH_SIZE, subset="train", shuffle=False):
            super().__init__()
            self.df = df.reset_index(drop=True)
            self.shuffle = shuffle
            self.subset = subset
            self.batch_size = batch_size
            self.on_epoch_end()

        def __len__(self):
            return int(math.ceil(len(self.df) / self.batch_size))

        def on_epoch_end(self):
            self.indexes = np.arange(len(self.df))
            if self.shuffle:
                np.random.shuffle(self.indexes)

        def __getitem__(self, index):
            start = index * self.batch_size
            end = min((index + 1) * self.batch_size, len(self.df))
            indexes = self.indexes[start:end]
            cur_bs = len(indexes)

            X = np.empty((cur_bs, 128, 128, 3), dtype=np.float32)
            y = np.empty((cur_bs, 128, 128, 3), dtype=np.float32)

            for i, img_path in enumerate(self.df.loc[indexes, "path"].values):
                w = int(self.df.loc[indexes[i], "width"])
                h = int(self.df.loc[indexes[i], "height"])

                img = self.__load_grayscale_as_3ch(img_path)
                X[i] = img

                if self.subset == "train":
                    for k, j in zip(
                        [0, 1, 2], ["large_bowel", "small_bowel", "stomach"]
                    ):
                        rles = ""
                        if j in self.df.columns:
                            rles = self.df.loc[indexes[i], j]
                        masks = rle_decode(rles, shape=(h, w, 1))
                        masks = cv2.resize(
                            masks, (128, 128), interpolation=cv2.INTER_NEAREST
                        )
                        y[i, :, :, k] = masks[:, :, 0]

            if self.subset == "train":
                return X, y
            else:
                return X

        def __load_grayscale_as_3ch(self, img_path):
            img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
            if img is None:
                raise FileNotFoundError(f"Could not read image: {img_path}")
            img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=-1)
            img = np.repeat(img, 3, axis=-1)
            return img




## === cell 10
gc.collect()



## === cell 11
if TF_AVAILABLE:

    def dice_coef(y_true, y_pred, smooth=1.0):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2.0 * intersection + smooth) / (
            K.sum(y_true_f) + K.sum(y_pred_f) + smooth
        )

    def iou_coef(y_true, y_pred, smooth=1.0):
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




## === cell 12
gc.collect()

model = None
if TF_AVAILABLE:
    custom_objects = {
        "FixedDropout": FixedDropout,
        "dice_coef": dice_coef,
        "iou_coef": iou_coef,
        "bce_dice_loss": bce_dice_loss,
    }

    def find_first_model_file():
        candidates = []
        for pattern in [
            "../input/**/model.h5",
            "../input/**/*.h5",
            "../input/**/*.keras",
        ]:
            candidates.extend(glob(pattern, recursive=True))
        candidates_sorted = sorted(
            candidates, key=lambda p: (os.path.basename(p) != "model.h5", len(p), p)
        )
        return candidates_sorted[0] if candidates_sorted else None

    model_path = "../input/uwmgi-unet-keras/model.h5"
    if not os.path.exists(model_path):
        alt = find_first_model_file()
        if alt is not None:
            model_path = alt

    if os.path.exists(model_path):
        model = load_model(model_path, custom_objects=custom_objects, compile=False)
        print("Loaded model from:", model_path)
    else:
        print(
            "WARNING: No pretrained model file found under ../input. Will fall back to heuristic masks."
        )

gc.collect()



## === cell 13
LOGITS = None
if TF_AVAILABLE and model is not None:
    pred_batches = DataGenerator(df_train, batch_size=1, subset="test", shuffle=False)
    gc.collect()
    LOGITS = model.predict(pred_batches, verbose=1)
    gc.collect()
else:
    pred_batches = None



## === cell 14
if LOGITS is None:
    print("LOGITS: None (no model or TF unavailable). Using heuristic masks.")
else:
    print("LOGITS shape:", LOGITS.shape)
print("df_train shape:", df_train.shape)




## === cell 15
def heuristic_mask_from_path(img_path, root_h, root_w):
    img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
    if img is None:
        return np.zeros((root_h, root_w), dtype=np.uint8)
    if img.shape[0] != root_h or img.shape[1] != root_w:
        img = cv2.resize(img, (root_w, root_h), interpolation=cv2.INTER_AREA)
    img8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, m = cv2.threshold(img8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if m.mean() < 0.01:  # almost empty -> invert
        m = 1 - m
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    return m.astype(np.uint8)


n_pred = len(df_train) if LOGITS is None else min(len(LOGITS), len(df_train))

lbs, sbs, sts = [], [], []
for index in tqdm(range(n_pred), total=n_pred):
    root_h = int(df_train.iloc[index]["height"])
    root_w = int(df_train.iloc[index]["width"])

    if LOGITS is None:
        m = heuristic_mask_from_path(df_train.iloc[index]["path"], root_h, root_w)
        r = rle_encode(m)
        lbs.append(r)
        sbs.append(r)
        sts.append(r)
        continue

    pred0 = cv2.resize(
        LOGITS[index, :, :, 0], (root_w, root_h), interpolation=cv2.INTER_NEAREST
    )
    pred1 = cv2.resize(
        LOGITS[index, :, :, 1], (root_w, root_h), interpolation=cv2.INTER_NEAREST
    )
    pred2 = cv2.resize(
        LOGITS[index, :, :, 2], (root_w, root_h), interpolation=cv2.INTER_NEAREST
    )

    pred_arr = np.round(pred0).astype("uint8")
    lbs.append(rle_encode(pred_arr))

    pred_arr = np.round(pred1).astype("uint8")
    sbs.append(rle_encode(pred_arr))

    pred_arr = np.round(pred2).astype("uint8")
    sts.append(rle_encode(pred_arr))

del LOGITS
gc.collect()



## === cell 16
df_ids = df_train[["id"]].iloc[:n_pred].reset_index(drop=True)
gc.collect()



## === cell 17
sub = pd.read_csv(
    "../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv"
)

pred_map = {}
for i in range(n_pred):
    pred_map[(df_ids.loc[i, "id"], "large_bowel")] = lbs[i]
    pred_map[(df_ids.loc[i, "id"], "small_bowel")] = sbs[i]
    pred_map[(df_ids.loc[i, "id"], "stomach")] = sts[i]

sub["predicted"] = [
    pred_map.get((row_id, cls), "")
    for row_id, cls in zip(sub["id"].values, sub["class"].values)
]
sub["predicted"] = sub["predicted"].fillna("").astype(str)

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
