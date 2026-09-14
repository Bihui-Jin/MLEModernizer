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

0.39538

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.44309) has done: 'I fix the import/runtime failures that prevent any submission from being written: remove/guard unavailable libraries (notably `tensorflow_addons` and `pandarallel`) and ensure required modules like `os` are imported before use. I also make the notebook Kaggle-safe by defining a `display()` fallback when not running in an IPython environment. Finally, I make the submission generation robust by filling missing RLEs with empty strings (valid “no mask” output) and ensuring `submission.csv` is always created with the required columns and row count.'
- What this solution (achieved 0.44309) has done: 'I fix the crash in the import cell caused by a protobuf incompatibility that happens when importing Plotly (it triggers an AttributeError in `google.protobuf`). The minimal safe fix is to avoid importing Plotly entirely, since it is not used anywhere in the training/inference/submission logic, keeping the core approach and resulting predictions unchanged. I also keep the existing guards for optional libraries and ensure `submission.csv` is always written with the correct columns and row count. No model/metric logic be changed, so the score should remain essentially the same (and thus still within the requested “minimal change” constraint).'
- What this solution (achieved 0.0) has done: 'I remove the import-time crash that prevents the notebook from running by avoiding TensorFlow Hub (it isn’t used anywhere in the training/inference/submission logic) and by adding a safe protobuf-related environment setting before any TensorFlow import. I keep the existing “slice_px_map lookup” submission logic intact, since that’s the core approach producing your current score, and only make minimal defensive tweaks (empty glob handling, stable root derivation) to avoid runtime errors on edge cases. These changes are expected to be score-neutral (or negligibly different) while ensuring the code runs end-to-end and always writes a valid `submission.csv`. No modeling/training semantics are changed.'
- What this solution (achieved 0.0) has done: 'I fix the import-time crash (`MessageFactory.GetPrototype`) by forcing protobuf to use the pure-Python implementation **before any protobuf-dependent imports** and by proactively removing the already-imported C++ protobuf module if present, then importing TensorFlow afterwards. I also make the RLE decode helper robust to empty/NaN strings (even though the main pipeline doesn’t call it) to prevent future runtime errors. The rest of the pipeline (the `slice_px_map` lookup-based submission logic that determines your score) is kept identical to preserve evaluation semantics, while ensuring `submission.csv` is always written with the exact required columns/row count.'
- What this solution (achieved 0.0) has done: 'I fix the import-time protobuf crash that currently stops execution before any submission is written by forcing the pure-Python protobuf implementation and preventing incompatible optional libraries from being imported. I keep the core “slice_px_map lookup” submission logic intact (which determines the score) and only add minimal defensive guards to ensure file discovery and dataframe merges don’t raise on edge cases. Finally, I ensure a valid `submission.csv` is always written with exactly the required columns and row count matching `sample_submission.csv`, with missing predictions filled as empty strings (valid no-mask RLE). These changes are intended to be score-neutral to slightly positive vs the current 0.0 (which is due to crashing), moving the score toward the target.'
- What this solution (achieved 0.0) has done: 'The crash happens before any training/inference because importing TensorFlow triggers a protobuf incompatibility (`MessageFactory.GetPrototype`) in this environment. I apply the minimal, Kaggle-safe fix by forcing the pure-Python protobuf implementation at the very top and explicitly removing any preloaded `google.protobuf` modules before importing TensorFlow, which prevents the AttributeError. I also keep optional/unneeded imports (e.g., plotly/tfhub/tfa/pandarallel) guarded so they can’t re-trigger protobuf issues. The rest of your pipeline (the `slice_px_map` lookup-based submission generation) is kept identical so the score should move from 0.0 (crash/no valid submission) back toward the expected baseline, and it always write a valid `submission.csv` with the required columns/row count.'
- What this solution (achieved 0.0) has done: 'The runtime crash happens before any submission is written because importing TensorFlow triggers an incompatible protobuf API (`MessageFactory.GetPrototype`) in this environment. I apply the minimal safe fix by forcing the pure-Python protobuf implementation *and* proactively removing/invalidating the C++ protobuf backend modules **before** any TensorFlow import, and as a fallback I run the pipeline without TensorFlow (it’s not used for this lookup-based submission logic anyway). I also keep the existing slice-to-RLE mapping logic unchanged to preserve evaluation semantics, and ensure the output `submission.csv` is always produced with exactly the required columns/row count. These changes are score-neutral relative to the intended baseline behavior (the score should move from 0.0 due to crashing toward your expected range).'
- What this solution (achieved 0.0) has done: 'I fix the crash in the import cell that prevents the notebook from running and writing a submission by *avoiding TensorFlow entirely* (it is not used anywhere in your actual slice-to-RLE lookup pipeline). This is the smallest change that unblocks execution and should restore a non-zero score (your current 0.0 is from not producing a valid submission due to the crash). I keep all feature extraction and the `slice_px_map` lookup logic identical, and only add minimal safety so the generated submission always matches `sample_submission.csv` row-for-row. The output be a valid `submission.csv` with the required columns and 20400 rows.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with producing mostly-empty (or mismatched) masks: the `slice_px_map` keys are built only from training slices with `n_segs==3`, and the float spacing strings must match *exactly* between train/test, so most test rows won’t find a key and become empty RLE. To move your score upward toward the 0.39538 target with minimal core-logic change, I (1) build the lookup map from all training slices that have each class present (not only `n_segs==3`), and (2) make the float spacing key stable by rounding both train and test spacing to a fixed precision before keying. This preserves your “lookup a memorized RLE by (H,W,spacing,class)” approach while drastically reducing accidental key misses. Submission writing stays identical and still guarantees a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 is consistent with the submission being effectively all-empty or badly aligned to the expected “id,class” rows due to preprocessing dropping many test rows when filepaths don’t merge (and then re-merging back can still leave most predictions empty). I make the preprocessing for test keep all rows (do not drop rows with missing `f_path`), and instead parse `slice_h/w/spacing` directly from the `id`’s corresponding PNG path via a fast lookup table; this preserves your same lookup-based “memorize RLE by (H,W,spacing,class)” core logic but prevents accidental row loss. I also make the key strings consistent by formatting the rounded spacings to a fixed number of decimals, reducing train/test key mismatches caused by float-to-string differences. These are minimal, score-positive fixes intended to move you up toward the 0.39538 target without changing the approach.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 indicates the submission is effectively all-empty (or nearly so), which is typically caused here by key mismatches between train/test spacing/size parsing and the lookup keys. I make two minimal, score-relevant fixes while preserving your same “memorize an RLE by (H,W,spacing,class) and look it up for test” core logic: (1) build the lookup map using the most common (mode) RLE per (H,W,spacing,class) instead of an arbitrary `.first()` (which often picks empty), and (2) ensure test rows without a matched PNG path still get a valid (but empty) prediction without breaking alignment, while keeping the exact sample_submission row order. These changes should move the score upward toward the target band by substantially reducing accidental empty outputs without changing the approach or introducing new modeling. The submission writing remains identical and always produces `submission.csv` with the required columns and 20400 rows.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 is consistent with producing an almost-all-empty (or invalidly encoded) submission rather than a truly “bad model”. The smallest score-positive fix that preserves your exact lookup-based approach is to ensure the RLE strings you copy from `train.csv` are always valid for the competition’s required ordering (“top-to-bottom, then left-to-right”): the training masks are encoded in a different convention, so copying them verbatim into `predicted` can score ~0. I minimally convert the memorized RLEs from train convention to the submission convention by decoding with the original orientation and re-encoding in the required order, while keeping your same `(H,W,spacing,class) -> memorized mask` logic and the same robust sample_submission alignment. This should move the score upward toward the target band without changing any modeling/training semantics (there is none here) and still writes a correct `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 is most consistent with a submission that decodes to (almost) no masks or invalid masks; the most score-relevant minimal fix is to ensure the RLE convention matches the competition’s required “top-to-bottom then left-to-right” ordering. I keep your core “memorize an RLE by (H,W,spacing,class) and look it up for test” approach intact, but (1) correct the train→submission RLE conversion (your current decode/shape usage is inconsistent and can silently produce wrong masks), and (2) make encoding explicitly column-major (Fortran order) to match the competition spec. These changes should move the score up from 0.0 toward your target without changing the overall method or introducing any modeling. The script still run end-to-end and always write a valid `submission.csv` with the exact required columns/row count.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 is almost certainly coming from an RLE convention mismatch: you’re copying RLEs from `train.csv` but converting them using the *wrong* decode orientation/shape, which can silently generate invalid/empty geometry after re-encoding and score near zero. I make the smallest core-preserving fix by decoding the training RLE using the competition’s required Fortran-order convention (`order="F"`) for the correct `(h,w)` shape, then re-encoding identically; this keeps your same `(slice_h,slice_w,spacing,class)->memorized RLE` lookup approach unchanged. I also add a tiny safety check so any malformed RLEs fall back to empty string instead of producing broken outputs. This should move the score upward toward (and likely into) your target band without changing the overall method.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with the submission being treated as invalid (or effectively empty) by the evaluator due to malformed RLE strings. The smallest score-positive fix that preserves your lookup-based core logic is to **sanitize every predicted RLE** (including those copied from train) so it’s guaranteed to be well-formed: sorted runs, positive lengths, and no overlaps, and to fall back to empty string if anything is off. I also ensure the RLE decode helper used in conversion always operates in the correct Kaggle Fortran-order and that the final `submission.csv` exactly matches `sample_submission.csv` rows and columns. These changes don’t change your approach (memorize-by-(H,W,spacing,class) and map onto test), but they should move the score up toward your target by preventing the scorer from zeroing out invalid masks.'

# 9. Code solution

## === cell 0
print("\n... IMPORTS STARTING ...\n")

import os
import sys

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

print("\n\tVERSION INFORMATION")

import gc
import re
import io
import math
import time
import json
import random
import shutil
import zipfile
import warnings
from glob import glob
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd

try:
    from IPython.display import display as _ip_display

    def display(x):
        return _ip_display(x)

except Exception:

    def display(x):
        if isinstance(x, pd.DataFrame):
            print(x.head())
        else:
            print(x)


tf = None
print("\t\t– TENSORFLOW: skipped (not used; avoids protobuf runtime crash)")

tfhub = None
print("\t\t– TENSORFLOW HUB: skipped (not used)")

try:
    import tensorflow_addons as tfa  # optional; not used downstream

    print(f"\t\t– TENSORFLOW ADDONS VERSION: {tfa.__version__}")
except Exception as e:
    tfa = None
    print(f"\t\t– TENSORFLOW ADDONS: not available ({type(e).__name__}: {e})")

import sklearn

print(f"\t\t– SKLEARN VERSION: {sklearn.__version__}")

from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.model_selection import GroupKFold, StratifiedKFold
from scipy.spatial import cKDTree

try:
    from pandarallel import pandarallel

    pandarallel.initialize()
    _PANDARALLEL_OK = True
except Exception:
    _PANDARALLEL_OK = False

import imageio
import urllib
import pickle
import ast
import gzip

import matplotlib

print(f"\t\t– MATPLOTLIB VERSION: {matplotlib.__version__}")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import matplotlib.patches as patches

plotly = None
pio = None
print("\t\t– PLOTLY: skipped (not used)")

import seaborn as sns
from PIL import Image, ImageEnhance
import cv2

try:
    from kaggle_datasets import KaggleDatasets
except Exception:
    KaggleDatasets = None


def seed_it_all(seed=7):
    """Attempt to be Reproducible"""
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

print("\n... ORIGINAL TRAINING DATAFRAME... \n")
display(train_df)

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

print("\n\n\n... ORIGINAL SUBMISSION DATAFRAME... \n")
display(ss_df)

SF2LF = {"lb": "Large Bowel", "sb": "Small Bowel", "st": "Stomach"}
LF2SF = {v: k for k, v in SF2LF.items()}
print(f"\n\n\n... ARE WE DEBUGGING: {DEBUG}... \n")

print("\n... BASIC DATA SETUP FINISHED ...\n\n")




## === cell 2
def get_filepath_from_partial_identifier(_ident, file_list):
    return [x for x in file_list if _ident in x][0]


def _build_sliceid_to_path(globbed_file_list):
    """
    Minimal, score-relevant robustness:
    Build a fast lookup from slice_id (the filename stem before '_{h}_{w}_{sh}_{sw}.png')
    to the full png path.
    """
    mp = {}
    for p in globbed_file_list:
        stem = os.path.basename(p)[:-4]
        slice_id = stem.rsplit("_", 4)[0]
        mp[slice_id] = p
    return mp


def df_preprocessing(df, globbed_file_list, is_test=False):
    """The preprocessing steps applied to get column information"""
    df = df.copy()

    df["case_id_str"] = df["id"].apply(lambda x: x.split("_", 2)[0])
    df["case_id"] = df["id"].apply(
        lambda x: int(x.split("_", 2)[0].replace("case", ""))
    )

    df["day_num_str"] = df["id"].apply(lambda x: x.split("_", 2)[1])
    df["day_num"] = df["id"].apply(lambda x: int(x.split("_", 2)[1].replace("day", "")))

    df["slice_id"] = df["id"].apply(lambda x: x.split("_", 2)[2])

    if len(globbed_file_list) == 0:
        raise RuntimeError(
            "No .png files found. Check DATA_DIR/TRAIN_DIR/TEST_DIR paths."
        )

    sliceid_to_path = _build_sliceid_to_path(globbed_file_list)
    df["f_path"] = df["slice_id"].map(sliceid_to_path)

    if not is_test:
        df = df[~df["f_path"].isna()].reset_index(drop=True)

    def _safe_int(tok):
        try:
            return int(tok)
        except Exception:
            return np.nan

    def _safe_float(tok):
        try:
            return float(tok)
        except Exception:
            return np.nan

    df["slice_h"] = df["f_path"].apply(
        lambda x: _safe_int(x[:-4].rsplit("_", 4)[1]) if isinstance(x, str) else np.nan
    )
    df["slice_w"] = df["f_path"].apply(
        lambda x: _safe_int(x[:-4].rsplit("_", 4)[2]) if isinstance(x, str) else np.nan
    )
    df["px_spacing_h"] = df["f_path"].apply(
        lambda x: (
            _safe_float(x[:-4].rsplit("_", 4)[3]) if isinstance(x, str) else np.nan
        )
    )
    df["px_spacing_w"] = df["f_path"].apply(
        lambda x: (
            _safe_float(x[:-4].rsplit("_", 4)[4]) if isinstance(x, str) else np.nan
        )
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


train_df = df_preprocessing(train_df, all_train_images)
ss_df = df_preprocessing(ss_df, all_test_images, is_test=True)

display(train_df.head())
display(ss_df.head())




## === cell 3
def rle_decode(mask_rle, shape, color=1):
    if mask_rle is None or (isinstance(mask_rle, float) and np.isnan(mask_rle)):
        return np.zeros(shape, dtype=np.float32)
    mask_rle = str(mask_rle).strip()
    if mask_rle == "":
        return np.zeros(shape, dtype=np.float32)

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


def rle_decode_kaggle(mask_rle, shape_hw):
    """
    Decode RLE (column-major/Fortran order) into a (H,W) mask.
    """
    if mask_rle is None or (isinstance(mask_rle, float) and np.isnan(mask_rle)):
        return np.zeros(shape_hw, dtype=np.uint8)
    mask_rle = str(mask_rle).strip()
    if mask_rle == "":
        return np.zeros(shape_hw, dtype=np.uint8)

    s = np.asarray(mask_rle.split(), dtype=int)
    if s.size % 2 != 0:
        return np.zeros(shape_hw, dtype=np.uint8)

    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths

    h, w = shape_hw
    img = np.zeros(h * w, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        if lo < 0:
            lo = 0
        if hi > img.size:
            hi = img.size
        if lo < hi:
            img[lo:hi] = 1

    return img.reshape((h, w), order="F")


def rle_encode(img):
    img = (img > 0).astype(np.uint8)
    pixels = img.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def rle_sanitize(mask_rle, shape_hw=None):
    if mask_rle is None or (isinstance(mask_rle, float) and np.isnan(mask_rle)):
        return ""
    s = str(mask_rle).strip()
    if s == "":
        return ""
    toks = s.split()
    if len(toks) % 2 != 0:
        return ""
    try:
        arr = np.asarray(toks, dtype=np.int64)
    except Exception:
        return ""
    starts = arr[0::2]
    lens = arr[1::2]
    if np.any(starts <= 0) or np.any(lens <= 0):
        return ""

    order = np.argsort(starts)
    starts = starts[order]
    lens = lens[order]
    ends = starts + lens - 1

    merged = []
    cur_s = int(starts[0])
    cur_e = int(ends[0])
    for s0, e0 in zip(starts[1:], ends[1:]):
        s0 = int(s0)
        e0 = int(e0)
        if s0 <= cur_e + 1:
            cur_e = max(cur_e, e0)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s0, e0
    merged.append((cur_s, cur_e))

    if shape_hw is not None:
        try:
            h, w = int(shape_hw[0]), int(shape_hw[1])
            n = h * w
        except Exception:
            n = None
        if n is not None and n > 0:
            clipped = []
            for s0, e0 in merged:
                s0 = max(1, s0)
                e0 = min(n, e0)
                if s0 <= e0:
                    clipped.append((s0, e0))
            merged = clipped

    out = []
    for s0, e0 in merged:
        out.append(str(s0))
        out.append(str(e0 - s0 + 1))
    return " ".join(out)


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




## === cell 4
_KEY_ROUND_DECIMALS = 3
train_df["px_spacing_h_k"] = train_df["px_spacing_h"].round(_KEY_ROUND_DECIMALS)
train_df["px_spacing_w_k"] = train_df["px_spacing_w"].round(_KEY_ROUND_DECIMALS)
ss_df["px_spacing_h_k"] = ss_df["px_spacing_h"].round(_KEY_ROUND_DECIMALS)
ss_df["px_spacing_w_k"] = ss_df["px_spacing_w"].round(_KEY_ROUND_DECIMALS)


def _fmt_k(x, d=_KEY_ROUND_DECIMALS):
    if pd.isna(x):
        return "nan"
    return f"{float(x):.{d}f}"


def _mode_non_empty(series):
    s = series.dropna().astype(str)
    s = s[s.str.len() > 0]
    if len(s) == 0:
        return np.nan
    vc = s.value_counts()
    return vc.index[0]


def _convert_train_rle_to_submission_rle(train_rle, h, w):
    """
    Score-relevant fix (minimal): ensure the RLE is valid and in Kaggle convention.
    We decode with (h,w) in Fortran order, then re-encode; finally sanitize.
    """
    if train_rle is None or (isinstance(train_rle, float) and np.isnan(train_rle)):
        return ""
    train_rle = str(train_rle).strip()
    if train_rle == "":
        return ""
    if pd.isna(h) or pd.isna(w):
        return ""
    h = int(h)
    w = int(w)

    try:
        m = rle_decode_kaggle(train_rle, (h, w))
        enc = rle_encode(m)
        return rle_sanitize(enc, (h, w))
    except Exception:
        return ""


slice_px_map = {}

grp_cols = ["slice_h", "slice_w", "px_spacing_h_k", "px_spacing_w_k"]
tmp = (
    train_df.groupby(grp_cols, dropna=False)[["lb_seg_rle", "sb_seg_rle", "st_seg_rle"]]
    .agg(
        {
            "lb_seg_rle": _mode_non_empty,
            "sb_seg_rle": _mode_non_empty,
            "st_seg_rle": _mode_non_empty,
        }
    )
    .reset_index()
)

for _, row in tmp.iterrows():
    if (
        pd.isna(row["slice_h"])
        or pd.isna(row["slice_w"])
        or pd.isna(row["px_spacing_h_k"])
        or pd.isna(row["px_spacing_w_k"])
    ):
        continue

    key_base = (
        f"{int(row['slice_h'])}-"
        f"{int(row['slice_w'])}-"
        f"{_fmt_k(row['px_spacing_h_k'])}-"
        f"{_fmt_k(row['px_spacing_w_k'])}"
    )

    h = int(row["slice_h"])
    w = int(row["slice_w"])

    lb = row.get("lb_seg_rle", np.nan)
    sb = row.get("sb_seg_rle", np.nan)
    st = row.get("st_seg_rle", np.nan)

    lb_c = _convert_train_rle_to_submission_rle(lb, h, w) if pd.notna(lb) else ""
    sb_c = _convert_train_rle_to_submission_rle(sb, h, w) if pd.notna(sb) else ""
    st_c = _convert_train_rle_to_submission_rle(st, h, w) if pd.notna(st) else ""

    if lb_c != "":
        slice_px_map[f"{key_base}-large_bowel"] = lb_c
    if sb_c != "":
        slice_px_map[f"{key_base}-small_bowel"] = sb_c
    if st_c != "":
        slice_px_map[f"{key_base}-stomach"] = st_c

print(f"Built slice_px_map with {len(slice_px_map):,} keys")




## === cell 5
ss_df["ident"] = (
    ss_df["slice_h"].fillna(-1).astype(int).astype(str)
    + "-"
    + ss_df["slice_w"].fillna(-1).astype(int).astype(str)
    + "-"
    + ss_df["px_spacing_h_k"].apply(_fmt_k).astype(str)
    + "-"
    + ss_df["px_spacing_w_k"].apply(_fmt_k).astype(str)
    + "-"
    + ss_df["class"].astype(str)
)

ss_df["predicted"] = ss_df["ident"].map(slice_px_map)
ss_df["predicted"] = ss_df["predicted"].fillna("")  # valid empty RLE

ss_df["predicted"] = [
    rle_sanitize(r, (h, w)) if (r != "" and pd.notna(h) and pd.notna(w)) else ""
    for r, h, w in zip(
        ss_df["predicted"].values, ss_df["slice_h"].values, ss_df["slice_w"].values
    )
]

submission_df = ss_df[["id", "class", "predicted"]].copy()

sample_path = os.path.join(DATA_DIR, "sample_submission.csv")
if os.path.exists(sample_path):
    sample_df = pd.read_csv(sample_path)
    submission_df = sample_df[["id", "class"]].merge(
        submission_df, on=["id", "class"], how="left"
    )
    submission_df["predicted"] = submission_df["predicted"].fillna("")

out_path = "submission.csv"
submission_df.to_csv(out_path, index=False)

print(f"Wrote {out_path} with shape {submission_df.shape}")
display(submission_df.head(10))
print("Empty RLE fraction:", (submission_df["predicted"] == "").mean())
print("Non-empty example (if any):")
nn = submission_df[submission_df["predicted"].str.len() > 0].head(1)
display(nn)
