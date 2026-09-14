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

geopandas==0.14.4
google-api-python-client==2.177.0
imageio==2.37.0
imageio-ffmpeg==0.6.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
plotly==5.24.1
plotly-express==0.4.1
protobuf==6.33.0
requests==2.32.5
requests-oauthlib==2.0.0
requests-toolbelt==1.0.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
seaborn==0.12.2
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
tqdm==4.67.1

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

0.39764

# 6. Current score

0.51373

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.51373) has done: 'I fix the import/runtime crash first by removing incompatible dependencies (notably `tensorflow_addons` and `pandarallel`) that aren’t needed for the core logic and are causing the `MessageFactory.GetPrototype` error in this environment. Next, I ensure `os`/other modules are imported before they’re used and replace the notebook-only `display()` calls with a safe fallback so the script runs as a plain Kaggle notebook/script. Then I harden the file-path merge logic so it can’t crash when a partial identifier isn’t found and make sure every test row gets a valid `predicted` string (empty string for “no mask”) so a submission CSV is always produced. These changes keep the solution’s core “lookup RLE by image geometry/class and filter by slice index” logic intact while making it run end-to-end and write `submission.csv`.'
- What this solution (achieved 0.51373) has done: 'I fix the runtime crash in the imports by removing the `tensorflow_hub` import, which is triggering the protobuf `MessageFactory.GetPrototype` incompatibility in this environment; the core solution logic does not use TF Hub anywhere. I also make preprocessing robust for test-time by not dropping unmatched test rows (so every `id,class` gets a prediction string), while keeping the original training-side behavior. Finally, I keep submission generation identical but ensure it always writes a valid `submission.csv` with the correct columns and empty strings for missing masks. These changes are score-neutral to slightly worse (more empty predictions instead of silently dropping rows), moving the score gently toward your lower target without changing the fundamental “geometry/class RLE lookup + slice-range filter” logic.'
- What this solution (achieved 0.51373) has done: 'I fix the import-time crash by removing TensorFlow usage from this pipeline (it’s not used by the core “geometry/class RLE lookup + slice-range filter” logic, and it’s what’s triggering the protobuf `MessageFactory.GetPrototype` error here). I also keep the same preprocessing and submission logic but harden file-list handling (avoid indexing `globbed_file_list[0]` when empty) and ensure every `id,class` row gets a valid `predicted` string. These are runtime/stability fixes and should keep behavior essentially score-neutral (if anything, slightly worse due to safer empty predictions), which moves your current score (0.51373) closer toward the lower target band around 0.39764 without changing the approach. The script run end-to-end and write `submission.csv` with the required columns.'
- What this solution (achieved 0.51373) has done: 'I fix the import-time crash caused by TensorFlow/protobuf in this environment by fully skipping the TensorFlow import (it is not used anywhere in the pipeline) so the notebook runs end-to-end. I keep the existing “geometry/class → RLE lookup + slice-range filter” logic intact, and only make small stability fixes to ensure all required columns exist for both train/test preprocessing and that `predicted` is always a valid string. Because your current score (0.51373) is above the target (0.39764) and higher-is-better, these changes are intended to be score-neutral to slightly worse while primarily ensuring reliability. The script still write a valid `submission.csv` with columns `id,class,predicted`.'

# 9. Code solution

## === cell 0
print("\n... IMPORTS STARTING ...\n")

print("\n\tVERSION INFORMATION")
import os
import sys
import gc
import re
import math
import time
import json
import random
import warnings
from glob import glob

import numpy as np
import pandas as pd

TF_AVAILABLE = False
print("\t\t– TENSORFLOW IMPORT SKIPPED (not needed for this pipeline)")

print(f"\t\t– NUMPY VERSION: {np.__version__}")
print(f"\t\t– PANDAS VERSION: {pd.__version__}")

import sklearn

print(f"\t\t– SKLEARN VERSION: {sklearn.__version__}")

import cv2
import matplotlib
import matplotlib.pyplot as plt

print(f"\t\t– MATPLOTLIB VERSION: {matplotlib.__version__}")

try:
    from IPython.display import display  # type: ignore
except Exception:

    def display(x):
        print(x)


pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def seed_it_all(seed=7):
    """Attempt to be reproducible."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


seed_it_all(7)

print("\n\n... IMPORTS COMPLETE ...\n")




## === cell 1
print("\n... BASIC DATA SETUP STARTING ...\n\n")

DATA_DIR = "/kaggle/input/uw-madison-gi-tract-image-segmentation"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
train_df = pd.read_csv(TRAIN_CSV)

all_train_images = glob(os.path.join(TRAIN_DIR, "**", "*.png"), recursive=True)

print("\n... ORIGINAL TRAINING DATAFRAME (head)... \n")
display(train_df.head())

TEST_DIR = os.path.join(DATA_DIR, "test")
SS_CSV = os.path.join(DATA_DIR, "sample_submission.csv")
ss_df = pd.read_csv(SS_CSV)

all_test_images = glob(os.path.join(TEST_DIR, "**", "*.png"), recursive=True)

DEBUG = len(ss_df) == 0

if DEBUG:
    TEST_DIR = TRAIN_DIR
    all_test_images = all_train_images
    ss_df = train_df.iloc[:10].copy()
    ss_df = ss_df[["id", "class"]]
    ss_df["predicted"] = ""

print("\n\n\n... ORIGINAL SUBMISSION DATAFRAME (head)... \n")
display(ss_df.head())

SF2LF = {"lb": "Large Bowel", "sb": "Small Bowel", "st": "Stomach"}
LF2SF = {v: k for k, v in SF2LF.items()}
print(f"\n\n\n... ARE WE DEBUGGING: {DEBUG}... \n")

print("\n... BASIC DATA SETUP FINISHED ...\n\n")




## === cell 2
def get_filepath_from_partial_identifier(_ident, file_list):
    matches = [x for x in file_list if _ident in x]
    return matches[0] if len(matches) else None


def df_preprocessing(df, globbed_file_list, is_test=False):
    """Preprocessing steps applied to get column information + file path mapping."""
    df = df.copy()

    df["case_id_str"] = df["id"].apply(lambda x: x.split("_", 2)[0])
    df["case_id"] = df["id"].apply(
        lambda x: int(x.split("_", 2)[0].replace("case", ""))
    )

    df["day_num_str"] = df["id"].apply(lambda x: x.split("_", 2)[1])
    df["day_num"] = df["id"].apply(lambda x: int(x.split("_", 2)[1].replace("day", "")))

    df["slice_id"] = df["id"].apply(lambda x: x.split("_", 2)[2])

    if len(globbed_file_list) == 0:
        print("WARNING: globbed_file_list is empty; cannot map ids to scan file paths.")
        df["f_path"] = np.nan
    else:
        base = globbed_file_list[0].rsplit("/", 4)[0] + "/"
        df["_partial_ident"] = (
            base
            + df["case_id_str"]
            + "/"
            + df["case_id_str"]
            + "_"
            + df["day_num_str"]
            + "/scans/"
            + df["slice_id"]
        )

        _tmp_merge_df = pd.DataFrame(
            {
                "_partial_ident": [x.rsplit("_", 4)[0] for x in globbed_file_list],
                "f_path": globbed_file_list,
            }
        )

        df = df.merge(_tmp_merge_df, on="_partial_ident", how="left").drop(
            columns=["_partial_ident"]
        )

    missing = df["f_path"].isna().sum()
    if missing:
        msg = f"WARNING: {missing} rows could not be matched to scan files"
        if is_test:
            print(msg + "; keeping them (will submit empty masks for them).")
        else:
            print(msg + "; dropping them from train preprocessing.")
            df = df.dropna(subset=["f_path"]).reset_index(drop=True)

    for col in ["slice_h", "slice_w", "px_spacing_h", "px_spacing_w"]:
        if col not in df.columns:
            df[col] = np.nan

    has_path = df["f_path"].notna()
    if has_path.any():
        df.loc[has_path, "slice_h"] = df.loc[has_path, "f_path"].apply(
            lambda x: int(x[:-4].rsplit("_", 4)[1])
        )
        df.loc[has_path, "slice_w"] = df.loc[has_path, "f_path"].apply(
            lambda x: int(x[:-4].rsplit("_", 4)[2])
        )
        df.loc[has_path, "px_spacing_h"] = df.loc[has_path, "f_path"].apply(
            lambda x: float(x[:-4].rsplit("_", 4)[3])
        )
        df.loc[has_path, "px_spacing_w"] = df.loc[has_path, "f_path"].apply(
            lambda x: float(x[:-4].rsplit("_", 4)[4])
        )

    if not is_test:
        l_bowel_df = df[df["class"] == "large_bowel"][["id", "segmentation"]].rename(
            columns={"segmentation": "lb_seg_rle"}
        )
        s_bowel_df = df[df["class"] == "small_bowel"][["id", "segmentation"]].rename(
            columns={"segmentation": "sb_seg_rle"}
        )
        stomach_df = df[df["class"] == "stomach"][["id", "segmentation"]].rename(
            columns={"segmentation": "st_seg_rle"}
        )

        df = df.merge(l_bowel_df, on="id", how="left")
        df = df.merge(s_bowel_df, on="id", how="left")
        df = df.merge(stomach_df, on="id", how="left")

        df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)

        df["lb_seg_flag"] = df["lb_seg_rle"].apply(lambda x: not pd.isna(x))
        df["sb_seg_flag"] = df["sb_seg_rle"].apply(lambda x: not pd.isna(x))
        df["st_seg_flag"] = df["st_seg_rle"].apply(lambda x: not pd.isna(x))
        df["n_segs"] = (
            df["lb_seg_flag"].astype(int)
            + df["sb_seg_flag"].astype(int)
            + df["st_seg_flag"].astype(int)
        )
    else:
        if "n_segs" not in df.columns:
            df["n_segs"] = np.nan

    new_col_order = [
        "id",
        "f_path",
        "n_segs",
        "lb_seg_rle",
        "lb_seg_flag",
        "sb_seg_rle",
        "sb_seg_flag",
        "st_seg_rle",
        "st_seg_flag",
        "slice_h",
        "slice_w",
        "px_spacing_h",
        "px_spacing_w",
        "case_id_str",
        "case_id",
        "day_num_str",
        "day_num",
        "slice_id",
    ]
    if is_test:
        new_col_order.insert(1, "class")
    new_col_order = [_c for _c in new_col_order if _c in df.columns]
    df = df[new_col_order]

    return df


train_df = df_preprocessing(train_df, all_train_images, is_test=False)
ss_df = df_preprocessing(ss_df, all_test_images, is_test=True)

print("Processed train_df (head):")
display(train_df.head())
print("Processed ss_df (head):")
display(ss_df.head())




## === cell 3
a = train_df.query("n_segs == 3").slice_id.value_counts()
arr = []
for slice_ in a[a > 10].index:
    arr.append(int(slice_.split("_")[-1]))

arr = np.array(arr) if len(arr) else np.array([0])
print(
    "slice index min/max among frequent 3-seg slices:", int(arr.min()), int(arr.max())
)




## === cell 4
def rle_decode(mask_rle, shape, color=1):
    """Run-length decode (start length), returns mask in given shape."""
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


def rle_decode_top_to_bot_first(mask_rle, shape):
    """Decode to 2D mask where pixels numbered top->bottom then left->right."""
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape((shape[1], shape[0]), order="F").T


def rle_encode(img):
    """Encode binary mask to RLE string."""
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def open_gray16(_path, normalize=True, to_rgb=False):
    """Helper to open 16-bit grayscale PNG."""
    if normalize:
        img = cv2.imread(_path, cv2.IMREAD_ANYDEPTH) / 65535.0
    else:
        img = cv2.imread(_path, cv2.IMREAD_ANYDEPTH)

    if to_rgb:
        return np.tile(np.expand_dims(img, axis=-1), 3)
    return img




## === cell 5
slice_px_map = {}
tmp = (
    train_df[train_df.n_segs == 3]
    .groupby(["slice_h", "slice_w", "px_spacing_h", "px_spacing_w"])[
        ["lb_seg_rle", "sb_seg_rle", "st_seg_rle"]
    ]
    .first()
    .reset_index()
)

for _, row in tmp.iterrows():
    key_base = (
        f"{row['slice_h']}-{row['slice_w']}-{row['px_spacing_h']}-{row['px_spacing_w']}"
    )
    slice_px_map[f"{key_base}-large_bowel"] = row["lb_seg_rle"]
    slice_px_map[f"{key_base}-small_bowel"] = row["sb_seg_rle"]
    slice_px_map[f"{key_base}-stomach"] = row["st_seg_rle"]

print("slice_px_map size:", len(slice_px_map))




## === cell 6
ss_df["sli"] = ss_df["slice_id"].apply(lambda x: int(str(x).split("_")[-1]))

ss_df["ident"] = (
    ss_df["slice_h"].astype(str)
    + "-"
    + ss_df["slice_w"].astype(str)
    + "-"
    + ss_df["px_spacing_h"].astype(str)
    + "-"
    + ss_df["px_spacing_w"].astype(str)
    + "-"
    + ss_df["class"].astype(str)
)

ss_df["predicted"] = ss_df["ident"].map(slice_px_map)

ss_df["predicted"] = ss_df.apply(
    lambda x: x.predicted if (x.sli >= 52 and x.sli <= 111) else np.nan, axis=1
)

ss_df["predicted"] = ss_df["predicted"].fillna("")
ss_df["predicted"] = ss_df["predicted"].astype(str)




## === cell 7
sub_df = ss_df[["id", "class", "predicted"]].copy()

out_path = "submission.csv"
sub_df.to_csv(out_path, index=False)

print("Wrote:", out_path, "shape:", sub_df.shape)
display(sub_df.head(10))
print("Non-empty predicted rows:", int((sub_df["predicted"].str.len() > 0).sum()))
