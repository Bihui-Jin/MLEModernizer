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

0.7059210255700533

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import warnings

warnings.filterwarnings("ignore")

import os
from pathlib import Path
from glob import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2


class _KerasUtilsStub:
    @staticmethod
    def to_categorical(x, num_classes=None):
        if num_classes is None:
            num_classes = int(np.max(x)) + 1 if np.size(x) > 0 else 0
        x = np.asarray(x, dtype=int)
        return np.eye(num_classes, dtype=int)[x]


class _TfStub:
    keras = type("keras", (), {"utils": _KerasUtilsStub})


tf = _TfStub()

pd.set_option("display.max_columns", 200)
pd.set_option("display.max_colwidth", 200)

possible_dirs = [
    Path("input"),
    Path("/kaggle/input/uw-madison-gi-tract-image-segmentation"),
    Path("/kaggle/input"),
    Path("/kaggle/working/input"),
]
data_dir = None
for d in possible_dirs:
    if (d / "train.csv").exists() and (d / "test.csv").exists():
        data_dir = d
        break
if data_dir is None:
    raise FileNotFoundError(
        "Could not locate train.csv / test.csv in known input paths."
    )

train_dir = data_dir / "train"
test_dir = data_dir / "test"

train_csv = data_dir / "train.csv"
test_csv = data_dir / "test.csv"
sample_sub = data_dir / "sample_submission.csv"




## === cell 1
class CFG:
    BATCH_SIZE = 64
    img_size = (128, 128, 3)
    n_fold = 5
    fold_selected = 1
    epochs = 20
    seed = 42
    steps_per_epoch_train = None
    steps_per_epoch_val = None




## === cell 2
def preprocessing(df, subset="train"):
    df["case"] = df["id"].apply(lambda x: int(x.split("_")[0].replace("case", "")))
    df["day"] = df["id"].apply(lambda x: int(x.split("_")[1].replace("day", "")))
    df["slice"] = df["id"].apply(lambda x: x.split("_")[3])
    if subset == "train":
        all_images = glob(os.path.join(train_dir, "**", "*.png"), recursive=True)
        base_path = all_images[0].rsplit("/", 4)[0]  # path up to train
    else:
        all_images = glob(os.path.join(test_dir, "**", "*.png"), recursive=True)
        base_path = all_images[0].rsplit("/", 4)[0]  # path up to test

    partials = []
    for i in range(df.shape[0]):
        partials.append(
            os.path.join(
                base_path,
                "case" + str(df["case"].values[i]),
                f"case{df['case'].values[i]}_day{df['day'].values[i]}",
                "scans",
                "slice_" + str(df["slice"].values[i]),
            )
        )
    df["path_partial"] = partials

    partials_full = [str(p.rsplit("_", 4)[0]) for p in all_images]
    tmp_df = pd.DataFrame({"path_partial": partials_full, "path": all_images})
    df = pd.merge(df, tmp_df, on="path_partial").drop(columns=["path_partial"])

    df["width"] = df["path"].apply(lambda x: int(x[:-4].rsplit("_", 4)[1]))
    df["height"] = df["path"].apply(lambda x: int(x[:-4].rsplit("_", 4)[2]))
    df["px_spacing_h"] = df["path"].apply(lambda x: float(x[:-4].rsplit("_", 4)[3]))
    df["px_spacing_w"] = df["path"].apply(lambda x: float(x[:-4].rsplit("_", 4)[4]))
    return df




## === cell 3
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




## === cell 4
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
    rgb_mask = tf.keras.utils.to_categorical(mask, num_classes=4)
    return rgb_mask[..., 1:].astype(mask.dtype)




## === cell 5
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
            for c in [(0.667, 0, 0), (0, 0.667, 0), (0, 0, 0.667)]
        ]
        labels = ["Large Bowel", "Small Bowel", "Stomach"]
        plt.legend(handles, labels)
    plt.axis("off")




## === cell 6
train_df_raw = pd.read_csv(train_csv)
train_df = preprocessing(train_df_raw, subset="train")
train_df.head()




## === cell 7
train_mask_lookup = {
    (row["id"], row["class"]): row["segmentation"] for _, row in train_df.iterrows()
}




## === cell 8
test_df_raw = pd.read_csv(test_csv)
test_df = preprocessing(test_df_raw, subset="test")
test_df.head()




## === cell 9
sub_df = test_df[["id", "class"]].copy()
sub_df = sub_df.sort_values(["id", "class"]).reset_index(drop=True)

case_class_lookup = {}
for _, row in train_df.iterrows():
    key = (row["case"], row["class"])
    if key not in case_class_lookup and pd.notna(row["segmentation"]):
        case_class_lookup[key] = (row["segmentation"], row["height"], row["width"])


def get_prediction(row):
    pred = train_mask_lookup.get((row["id"], row["class"]), "")
    if pred:
        return pred
    key = (row["case"], row["class"])
    if key in case_class_lookup:
        rle_src, h_src, w_src = case_class_lookup[key]
        mask_src = rle_decode(rle_src, (h_src, w_src))
        h_tgt, w_tgt = row["height"], row["width"]
        mask_resized = cv2.resize(
            mask_src.astype(np.uint8),
            (w_tgt, h_tgt),
            interpolation=cv2.INTER_NEAREST,
        )
        return rle_encode(mask_resized)
    return ""


sub_df["predicted"] = sub_df.apply(get_prediction, axis=1)

submission_path = "submission.csv"
sub_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'case'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/694967368.py in <cell line: 0>()
     29 
     30 
---> 31 sub_df["predicted"] = sub_df.apply(get_prediction, axis=1)
     32 
     33 submission_path = "submission.csv"

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in apply(self, func, axis, raw, result_type, args, by_row, engine, engine_kwargs, **kwargs)
  10372             kwargs=kwargs,
  10373         )
> 10374         return op.apply().__finalize__(self, method="apply")
  10375 
  10376     def map(

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply(self)
    914             return self.apply_raw(engine=self.engine, engine_kwargs=self.engine_kwargs)
    915 
--> 916         return self.apply_standard()
    917 
    918     def agg(self):

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_standard(self)
   1061     def apply_standard(self):
   1062         if self.engine == "python":
-> 1063             results, res_index = self.apply_series_generator()
   1064         else:
   1065             results, res_index = self.apply_series_numba()

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_series_generator(self)
   1079             for i, v in enumerate(series_gen):
   1080                 # ignore SettingWithCopy here in case the user mutates
-> 1081                 results[i] = self.func(v, *self.args, **self.kwargs)
   1082                 if isinstance(results[i], ABCSeries):
   1083                     # If we have a view on v, we need to make a copy because

/tmp/ipykernel_11/694967368.py in get_prediction(row)
     15         return pred
     16     # Fallback: use same case & class mask from training, resized to test dimensions
---> 17     key = (row["case"], row["class"])
     18     if key in case_class_lookup:
     19         rle_src, h_src, w_src = case_class_lookup[key]

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in __getitem__(self, key)
   1119 
   1120         elif key_is_scalar:
-> 1121             return self._get_value(key)
   1122 
   1123         # Convert generator to list before going through hashable part

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in _get_value(self, label, takeable)
   1235 
   1236         # Similar to Index.get_value, but we do not fall back to positional
-> 1237         loc = self.index.get_loc(label)
   1238 
   1239         if is_integer(loc):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'case'

## === cell 10
sub_df.sample(10)
