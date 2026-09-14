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

0.39538

# 6. Current score

0.44309

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.44309) has done: 'I make the notebook run end-to-end by fixing the import/runtime blockers: remove/guard unavailable or incompatible imports (tensorflow_addons, pandarallel, kaggle_datasets) and ensure core modules like `os` are imported before first use. Then I eliminate `display()` calls (not guaranteed in script mode) and wrap the EDA/plotting cells so they don’t stop submission generation. Finally, I keep your original “nearest (slice_h, slice_w, pixel spacing) → copy RLE from train” submission logic, but ensure `predicted` is always a valid string (empty when missing) and write a proper `submission.csv` with the required columns.'
- What this solution (achieved 0.44309) has done: 'I fix the immediate runtime crash in the imports cell by removing/guarding the `tensorflow_hub` import, which is triggering a protobuf `MessageFactory.GetPrototype` incompatibility in this environment. I keep the rest of the pipeline and the core “(slice_h, slice_w, pixel spacing) → copy RLE from train” logic unchanged so the score behavior stays essentially the same (your current 0.44309 is already above the 0.39538 target band). I also make one minimal robustness tweak to ensure `predicted` is always a non-null string and that the submission columns exactly match the required format. The script run end-to-end and write `submission.csv`.'
- What this solution (achieved 0.44309) has done: 'I fix the runtime crash caused by importing `tensorflow_hub`, which is triggering a protobuf `MessageFactory.GetPrototype` AttributeError in this Kaggle environment. The fix is to fully guard/disable `tensorflow_hub` import (since it’s unused by the solution) so the notebook runs end-to-end reliably. I keep the core “(slice_h, slice_w, px_spacing_h, px_spacing_w, class) → copy RLE from train” submission logic unchanged to avoid unnecessary score changes (your current 0.44309 is already within ±10% of the 0.39538 target). I also ensure `predicted` is always a valid non-null string and that `submission.csv` is written with the exact required columns.'
- What this solution (achieved 0.44309) has done: 'I fix the import-time crash by proactively guarding against the protobuf `MessageFactory.GetPrototype` incompatibility (it can be triggered transitively by optional visualization/TF ecosystem imports), while keeping your actual submission logic unchanged to avoid unnecessary score movement (your 0.44309 is already within ±10% of the 0.39538 target). I also make the plotting stack optional so headless/script execution can’t fail before writing the CSV. Finally, I add a small safety fallback so that if any test image metadata fails to parse/match, the submission still contains valid empty-string RLEs and is always written as `submission.csv`.'
- What this solution (achieved 0.44309) has done: 'I fix the import-time crash (`MessageFactory.GetPrototype`) by proactively pinning protobuf to its pure-Python implementation before any TensorFlow-related imports occur, which avoids the incompatible C++ protobuf path that triggers this error in some Kaggle TF 2.18 images. I keep your existing “(slice_h, slice_w, px_spacing_h, px_spacing_w, class) → copy RLE from train” submission logic unchanged to keep the score behavior essentially the same (your current score is already within the ±10% target band). I also add a small safety guard so that if the crash still occurs for any reason, the script falls back to writing a valid all-empty submission.csv rather than failing to produce a file.'

# 9. Code solution

## === cell 0
print("\n... IMPORTS STARTING ...\n")

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

print("\n\tVERSION INFORMATION")
import sys
import gc
import json
import math
import time
import random
import warnings
from glob import glob

import numpy as np
import pandas as pd

import tensorflow as tf

tfhub = None
_tfhub_ver = "disabled (protobuf incompatibility in this environment)"

print(f"\t\t– PYTHON VERSION: {sys.version.split()[0]}")
print(f"\t\t– TENSORFLOW VERSION: {tf.__version__}")
print(f"\t\t– TENSORFLOW HUB VERSION: {_tfhub_ver}")
print(f"\t\t– NUMPY VERSION: {np.__version__}")
print(f"\t\t– PANDAS VERSION: {pd.__version__}")

try:
    import tensorflow_addons as tfa  # noqa: F401

    print(f"\t\t– TENSORFLOW ADDONS VERSION: {tfa.__version__}")
except Exception as e:
    tfa = None
    print(f"\t\t– TENSORFLOW ADDONS: not available/disabled ({type(e).__name__}: {e})")

try:
    from pandarallel import pandarallel  # noqa: F401

    pandarallel.initialize()
    print("\t\t– PANDARALLEL: initialized")
except Exception as e:
    pandarallel = None
    print(f"\t\t– PANDARALLEL: not available/disabled ({type(e).__name__}: {e})")

import sklearn  # noqa: F401
from sklearn.preprocessing import RobustScaler, PolynomialFeatures  # noqa: F401
from sklearn.model_selection import GroupKFold, StratifiedKFold  # noqa: F401

from scipy.spatial import cKDTree  # noqa: F401

HAVE_PLOT = True
try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib import rc

    rc("animation", html="jshtml")
    print(f"\t\t– MATPLOTLIB VERSION: {matplotlib.__version__}")
except Exception as e:
    HAVE_PLOT = False
    matplotlib = None
    plt = None
    Rectangle = None
    print(f"\t\t– MATPLOTLIB: disabled ({type(e).__name__}: {e})")

try:
    import plotly
    import plotly.express as px
    import plotly.graph_objects as go  # noqa: F401
    import plotly.io as pio

    print(pio.renderers)
except Exception as e:
    plotly = None
    px = None
    go = None
    pio = None
    print(f"\t\t– PLOTLY: disabled ({type(e).__name__}: {e})")

try:
    import seaborn as sns  # noqa: F401
except Exception as e:
    sns = None
    print(f"\t\t– SEABORN: disabled ({type(e).__name__}: {e})")

from PIL import Image, ImageEnhance  # noqa: F401
import cv2

pd.options.mode.chained_assignment = None


def seed_it_all(seed=7):
    """Attempt to be Reproducible."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


seed_it_all(7)

print("\n\n... IMPORTS COMPLETE ...\n")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
print(f"\n... ACCELERATOR SETUP STARTING ...\n")

try:
    TPU = tf.distribute.cluster_resolver.TPUClusterResolver()
except Exception:
    TPU = None

if TPU:
    print(f"\n... RUNNING ON TPU - {TPU.master()}...")
    tf.config.experimental_connect_to_cluster(TPU)
    tf.tpu.experimental.initialize_tpu_system(TPU)
    strategy = tf.distribute.experimental.TPUStrategy(TPU)
else:
    print(f"\n... RUNNING ON CPU/GPU ...")
    strategy = tf.distribute.get_strategy()

N_REPLICAS = strategy.num_replicas_in_sync
print(f"... # OF REPLICAS: {N_REPLICAS} ...\n")
print(f"\n... ACCELERATOR SETUP COMPLETED ...\n")



## === cell 2
print("\n... DATA ACCESS SETUP STARTED ...\n")

if TPU:
    try:
        from kaggle_datasets import KaggleDatasets

        DATA_DIR = KaggleDatasets().get_gcs_path(
            "uw-madison-gi-tract-image-segmentation"
        )
        save_locally = tf.saved_model.SaveOptions(
            experimental_io_device="/job:localhost"
        )
        load_locally = tf.saved_model.LoadOptions(
            experimental_io_device="/job:localhost"
        )
    except Exception as e:
        print(
            f"... KaggleDatasets unavailable on this runtime ({type(e).__name__}: {e}); using local /kaggle/input ..."
        )
        DATA_DIR = "/kaggle/input/uw-madison-gi-tract-image-segmentation"
        save_locally = None
        load_locally = None
else:
    DATA_DIR = "/kaggle/input/uw-madison-gi-tract-image-segmentation"
    save_locally = None
    load_locally = None

print(f"\n... DATA DIRECTORY PATH IS:\n\t--> {DATA_DIR}")

print(f"\n... IMMEDIATE CONTENTS OF DATA DIRECTORY IS:")
for file in tf.io.gfile.glob(os.path.join(DATA_DIR, "*")):
    print(f"\t--> {file}")

print("\n\n... DATA ACCESS SETUP COMPLETED ...\n")



## === cell 3
print(f"\n... XLA OPTIMIZATIONS STARTING ...\n")
print(f"\n... CONFIGURE JIT (JUST IN TIME) COMPILATION ...\n")
tf.config.optimizer.set_jit(True)
print(f"\n... XLA OPTIMIZATIONS COMPLETED ...\n")



## === cell 4
print("\n... BASIC DATA SETUP STARTING ...\n\n")

TRAIN_DIR = os.path.join(DATA_DIR, "train")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
train_df = pd.read_csv(TRAIN_CSV)

all_train_images = glob(os.path.join(TRAIN_DIR, "**", "*.png"), recursive=True)

print("\n... TRAINING DATAFRAME HEAD... \n")
print(train_df.head())

TEST_DIR = os.path.join(DATA_DIR, "test")
SS_CSV = os.path.join(DATA_DIR, "sample_submission.csv")
ss_df = pd.read_csv(SS_CSV)

all_test_images = glob(os.path.join(TEST_DIR, "**", "*.png"), recursive=True)

print("\n\n... SUBMISSION DATAFRAME HEAD... \n")
print(ss_df.head())

DEBUG = len(ss_df) == 0

if DEBUG:
    TEST_DIR = TRAIN_DIR
    all_test_images = all_train_images
    ss_df = train_df.iloc[:10].copy()
    ss_df = ss_df[["id", "class"]]
    ss_df["predicted"] = ""
    print("\n\n... DEBUG SUBMISSION DATAFRAME... \n")
    print(ss_df)

SF2LF = {"lb": "Large Bowel", "sb": "Small Bowel", "st": "Stomach"}
LF2SF = {v: k for k, v in SF2LF.items()}
print(f"\n\n... ARE WE DEBUGGING: {DEBUG}... \n")

print("\n... BASIC DATA SETUP FINISHED ...\n\n")




## === cell 5
def get_filepath_from_partial_identifier(_ident, file_list):
    return [x for x in file_list if _ident in x][0]


def df_preprocessing(df, globbed_file_list, is_test=False):
    """The preprocessing steps applied to get column information."""
    df = df.copy()

    df["case_id_str"] = df["id"].apply(lambda x: x.split("_", 2)[0])
    df["case_id"] = df["id"].apply(
        lambda x: int(x.split("_", 2)[0].replace("case", ""))
    )

    df["day_num_str"] = df["id"].apply(lambda x: x.split("_", 2)[1])
    df["day_num"] = df["id"].apply(lambda x: int(x.split("_", 2)[1].replace("day", "")))

    df["slice_id"] = df["id"].apply(lambda x: x.split("_", 2)[2])

    base_dir = globbed_file_list[0].rsplit("/", 4)[0]  # .../train or .../test
    df["_partial_ident"] = (
        base_dir
        + "/"
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

    df["slice_h"] = df["f_path"].apply(
        lambda x: int(x[:-4].rsplit("_", 4)[1]) if isinstance(x, str) else np.nan
    )
    df["slice_w"] = df["f_path"].apply(
        lambda x: int(x[:-4].rsplit("_", 4)[2]) if isinstance(x, str) else np.nan
    )

    df["px_spacing_h"] = df["f_path"].apply(
        lambda x: float(x[:-4].rsplit("_", 4)[3]) if isinstance(x, str) else np.nan
    )
    df["px_spacing_w"] = df["f_path"].apply(
        lambda x: float(x[:-4].rsplit("_", 4)[4]) if isinstance(x, str) else np.nan
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




## === cell 6
print("\n... UPDATING DATAFRAMES WITH ACCESSIBLE INFORMATION STARTED ...\n\n")

train_df = df_preprocessing(train_df, all_train_images, is_test=False)
print("\n... UPDATED TRAINING DATAFRAME HEAD... \n")
print(train_df.head())

ss_df = df_preprocessing(ss_df, all_test_images, is_test=True)
print("\n\n... UPDATED SUBMISSION DATAFRAME HEAD... \n")
print(ss_df.head())

print("\n... UPDATING DATAFRAMES WITH ACCESSIBLE INFORMATION FINISHED ...\n\n")




## === cell 7
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


def rle_decode_top_to_bot_first(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape((shape[1], shape[0]), order="F").T


def rle_encode(img):
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def flatten_l_o_l(nested_list):
    return [item for sublist in nested_list for item in sublist]


def load_json_to_dict(json_path):
    with open(json_path) as json_file:
        data = json.load(json_file)
    return data


def tf_load_png(img_path):
    return tf.image.decode_png(tf.io.read_file(img_path), channels=3)


def open_gray16(_path, normalize=True, to_rgb=False):
    if normalize:
        if to_rgb:
            return np.tile(
                np.expand_dims(
                    cv2.imread(_path, cv2.IMREAD_ANYDEPTH) / 65535.0, axis=-1
                ),
                3,
            )
        else:
            return cv2.imread(_path, cv2.IMREAD_ANYDEPTH) / 65535.0
    else:
        if to_rgb:
            return np.tile(
                np.expand_dims(cv2.imread(_path, cv2.IMREAD_ANYDEPTH), axis=-1), 3
            )
        else:
            return cv2.imread(_path, cv2.IMREAD_ANYDEPTH)




## === cell 8
def get_overlay(img_path, rle_strs, img_shape, _alpha=0.999, _beta=0.35, _gamma=0):
    _img = open_gray16(img_path, to_rgb=True)
    _img = ((_img - _img.min()) / (_img.max() - _img.min() + 1e-8)).astype(np.float32)
    _seg_rgb = np.stack(
        [
            (
                rle_decode(rle_str, shape=img_shape, color=1)
                if rle_str is not None
                else np.zeros(img_shape, dtype=np.float32)
            )
            for rle_str in rle_strs
        ],
        axis=-1,
    ).astype(np.float32)
    seg_overlay = cv2.addWeighted(
        src1=_img, alpha=_alpha, src2=_seg_rgb, beta=_beta, gamma=_gamma
    )
    return seg_overlay


def examine_id(
    ex_id,
    df=None,
    plot_overlay=True,
    print_meta=False,
    plot_grayscale=False,
    plot_binary_segmentation=False,
):
    if not HAVE_PLOT:
        print("Plotting disabled in this runtime; skipping visualization.")
        return

    if df is None:
        df = train_df

    print(f"\n... ID ({ex_id}) EXPLORATION STARTED ...\n\n")
    demo_ex = df[df.id == ex_id].squeeze()

    if print_meta:
        print(f"\n... META FOR ID=`{ex_id}` ...\n")
        print(demo_ex)

    if plot_grayscale:
        plt.figure(figsize=(12, 12))
        plt.imshow(open_gray16(demo_ex.f_path), cmap="gray")
        plt.title(f"Original Grayscale Image For ID: {demo_ex.id}", fontweight="bold")
        plt.axis(False)
        plt.show()

    if plot_binary_segmentation:
        plt.figure(figsize=(20, 10))
        for i, _seg_type in enumerate(["lb", "sb", "st"]):
            if pd.isna(demo_ex.get(f"{_seg_type}_seg_rle", np.nan)):
                continue
            plt.subplot(1, 3, i + 1)
            plt.imshow(
                rle_decode(
                    demo_ex[f"{_seg_type}_seg_rle"],
                    shape=(demo_ex.slice_w, demo_ex.slice_h),
                    color=1,
                )
            )
            plt.title(
                f"RLE Encoding For {SF2LF[_seg_type]} Segmentation", fontweight="bold"
            )
            plt.axis(False)
        plt.tight_layout()
        plt.show()

    if plot_overlay:
        _rle_strs = [
            (
                demo_ex[f"{_seg_type}_seg_rle"]
                if not pd.isna(demo_ex.get(f"{_seg_type}_seg_rle", np.nan))
                else None
            )
            for _seg_type in ["lb", "sb", "st"]
        ]
        seg_overlay = get_overlay(
            demo_ex.f_path, _rle_strs, img_shape=(demo_ex.slice_w, demo_ex.slice_h)
        )

        plt.figure(figsize=(12, 12))
        plt.imshow(seg_overlay)
        plt.title(f"Segmentation Overlay For ID: {demo_ex.id}", fontweight="bold")
        handles = [
            Rectangle((0, 0), 1, 1, color=_c)
            for _c in [(0.667, 0.0, 0.0), (0.0, 0.667, 0.0), (0.0, 0.0, 0.667)]
        ]
        labels = [
            "Large Bowel Segmentation Map",
            "Small Bowel Segmentation Map",
            "Stomach Segmentation Map",
        ]
        plt.legend(handles, labels)
        plt.axis(False)
        plt.show()

    print("\n\n... SINGLE ID EXPLORATION FINISHED ...\n\n")




## === cell 9
RUN_EDA = False

if RUN_EDA and HAVE_PLOT:
    print("\n... SINGLE ID EXPLORATION STARTED ...\n\n")
    DEMO_ID = "case123_day20_slice_0082"
    demo_ex = train_df[train_df.id == DEMO_ID].squeeze()
    print(demo_ex.to_frame())

    plt.figure(figsize=(12, 12))
    plt.imshow(open_gray16(demo_ex.f_path), cmap="gray")
    plt.title(f"Original Grayscale Image For ID: {demo_ex.id}", fontweight="bold")
    plt.axis(False)
    plt.show()

    print("\n\n... SINGLE ID EXPLORATION FINISHED ...\n\n")



## === cell 10
if RUN_EDA and HAVE_PLOT:
    N_TO_PLOT = 10
    for _id in (
        train_df[train_df.n_segs == 3]
        .groupby("case_id")["id"]
        .first()
        .sample(N_TO_PLOT)
    ):
        examine_id(_id)



## === cell 11
if RUN_EDA and (px is not None):

    def get_seg_combo_str(row):
        seg_str_list = []
        if row.get("lb_seg_flag", False):
            seg_str_list.append("Large Bowel")
        if row.get("sb_seg_flag", False):
            seg_str_list.append("Small Bowel")
        if row.get("st_seg_flag", False):
            seg_str_list.append("Stomach")
        return ", ".join(seg_str_list) if len(seg_str_list) > 0 else "No Mask"

    train_df["seg_combo_str"] = train_df.apply(get_seg_combo_str, axis=1)
    fig = px.histogram(
        train_df,
        train_df["n_segs"].astype(str),
        color="seg_combo_str",
        title="Number of Segmentation Masks Per Image",
        labels={
            "x": "Number of Segmentation Masks Per Image",
            "seg_combo_str": "Segmentation Masks Present",
        },
    )
    fig.show()



## === cell 12
if RUN_EDA and (px is not None):
    dd = train_df.drop_duplicates(subset=["slice_w", "slice_h"]).copy()
    dd["count"] = (
        train_df.groupby(["slice_w", "slice_h"])["id"]
        .transform("count")
        .iloc[dd.index]
        .values
    )
    dd["size_legend"] = (
        "(" + dd["slice_w"].astype(str) + "," + dd["slice_h"].astype(str) + ")"
    )
    fig = px.scatter(
        dd,
        x="slice_w",
        y="slice_h",
        size="count",
        color="size_legend",
        title="Bubble Chart Showing The Various Image Sizes",
        labels={
            "color": "Size Legend",
            "count": "Number Of Observations",
            "slice_h": "Image Slice Height (pixels)",
            "slice_w": "Image Slice Width (pixels)",
        },
        size_max=160,
    )
    fig.show()



## === cell 13
if RUN_EDA and (px is not None):
    dd = train_df.drop_duplicates(subset=["px_spacing_w", "px_spacing_h"]).copy()
    dd["count"] = (
        train_df.groupby(["px_spacing_w", "px_spacing_h"])["id"]
        .transform("count")
        .iloc[dd.index]
        .values
    )
    dd["px_legend"] = (
        "("
        + dd["px_spacing_w"].astype(str)
        + ","
        + dd["px_spacing_h"].astype(str)
        + ")"
    )
    fig = px.scatter(
        dd,
        x="px_spacing_w",
        y="px_spacing_h",
        size="count",
        color="px_legend",
        title="Bubble Chart Showing The Various Pixel Spacings",
        labels={
            "color": "Pixel Spacing Sets Legend",
            "count": "Number Of Observations",
            "px_spacing_h": "Pixel Spacing Height (mm)",
            "px_spacing_w": "Pixel Spacing Width (mm)",
        },
        size_max=160,
    )
    fig.show()



## === cell 14
if RUN_EDA and (px is not None):
    fig = px.histogram(
        train_df,
        train_df.case_id.astype(str),
        color="day_num_str",
        title="Distribution Of Images Per Case ID",
        labels={"x": "Case ID", "day_num_str": "The Day The Scan Took Place"},
        width=2000,
    )
    fig.show()



## === cell 15
print(
    "\n... BUILDING SUBMISSION VIA (slice_h, slice_w, px_spacing_h, px_spacing_w, class) LOOKUP ...\n"
)

try:
    slice_px_map = {}
    train_grp = (
        train_df[train_df.n_segs == 3]
        .groupby(["slice_h", "slice_w", "px_spacing_h", "px_spacing_w"])[
            ["lb_seg_rle", "sb_seg_rle", "st_seg_rle"]
        ]
        .first()
        .reset_index()
    )

    for _, row in train_grp.iterrows():
        slice_px_map[
            f"{row['slice_h']}-{row['slice_w']}-{row['px_spacing_h']}-{row['px_spacing_w']}-large_bowel"
        ] = row["lb_seg_rle"]
        slice_px_map[
            f"{row['slice_h']}-{row['slice_w']}-{row['px_spacing_h']}-{row['px_spacing_w']}-small_bowel"
        ] = row["sb_seg_rle"]
        slice_px_map[
            f"{row['slice_h']}-{row['slice_w']}-{row['px_spacing_h']}-{row['px_spacing_w']}-stomach"
        ] = row["st_seg_rle"]

    ss_df = ss_df.copy()

    for c in ["slice_h", "slice_w", "px_spacing_h", "px_spacing_w", "class"]:
        if c in ss_df.columns:
            ss_df[c] = ss_df[c].astype(object)

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

    ss_df["predicted"] = ss_df["predicted"].where(ss_df["predicted"].notna(), "")
    ss_df["predicted"] = ss_df["predicted"].astype(str)
    ss_df.loc[ss_df["predicted"].str.lower().isin(["nan", "none"]), "predicted"] = ""

    submission_df = ss_df[["id", "class", "predicted"]]
except Exception as e:
    print(
        f"Submission pipeline failed ({type(e).__name__}: {e}). Writing empty submission as fallback."
    )
    base_ss = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
    submission_df = base_ss[["id", "class", "predicted"]].copy()
    submission_df["predicted"] = ""

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"... WROTE SUBMISSION: {submission_path} | rows={len(submission_df)}")
print(submission_df.head(10))
print("\n... SUBMISSION GENERATION COMPLETE ...\n")
