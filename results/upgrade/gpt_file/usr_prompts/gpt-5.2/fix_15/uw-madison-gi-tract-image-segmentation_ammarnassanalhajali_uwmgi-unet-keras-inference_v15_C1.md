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

0.4484978484787068

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0029) has done: 'I fix the TensorFlow/protobuf import crash by forcing TensorFlow to use the python protobuf implementation before importing TF, and by using a safer import order. Then I fix the missing pretrained model path by falling back to a minimal U-Net-like model with the same 128×128×3 input and 3-channel sigmoid output so the notebook can run end-to-end even without external model files. Finally, I fix the DataGenerator/test prediction length mismatch (it was dropping the last partial batch) and ensure the RLE encoding uses the required column-major (Fortran) order so the submission format is valid and scores move toward the target instead of being penalized by wrong encoding.'
- What this solution (achieved 0.0) has done: 'I fix the protobuf/TensorFlow crash by forcing a compatible protobuf runtime mode before importing TensorFlow and, if needed, downgrading protobuf in-notebook to a version that matches TF’s expected API (this is the root cause of `MessageFactory.GetPrototype` failures). Then I correct the submission id parsing so slice filenames are built correctly (the current `slice` extraction is wrong), which currently causes missing/incorrect image-path joins and effectively random/empty predictions (driving the very low score). Finally, I make RLE decoding/encoding consistent with the competition’s column-major convention (decode currently reshapes in C-order), which prevents metric penalties from malformed masks; these are minimal, logic-preserving fixes aimed at moving the score up toward the target.'
- What this solution (achieved 0.0) has done: 'I fix the root cause of the “PyDataset has length 0” error by ensuring `df_train` is never empty after the path-merge step (the current `slice` parsing is wrong and breaks the join to PNG files). I correct `slice` extraction to keep the `slice_XXXX` token intact and build the join key robustly using `os.path.dirname(path)` rather than brittle string splits. These are minimal, logic-preserving fixes that unblock inference and produce a non-empty generator, which should raise the score from 0.0 toward the target simply by generating valid, correctly-aligned predictions and RLEs. The rest of the model/prediction/RLE logic is kept the same.'
- What this solution (achieved 0.0) has done: 'I fix the root cause of the empty merged dataframe by correcting how `path_partial` is constructed so it matches the actual PNG filename stem (`slice_XXXX`) inside each case/day scans folder. This make `df_train` non-empty, which in turn fixes the downstream “PyDataset has length 0” error during `model.predict`. I also make the width/height extraction robust by parsing them from the actual PNG filename (not from `path_partial`) to avoid silent mis-parsing. These changes are minimal and keep the model/inference/RLE logic the same, but should move the score up from 0.0 by producing valid, correctly-aligned masks and a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'I fix the root cause of the empty `df_train` (and the downstream “PyDataset has length 0”) by correcting the PNG filename parsing and the `slice` join key: the code currently swaps width/height indices and builds a `path_partial` that can silently fail to match real scan filenames. I make the merge robust by extracting the actual `slice_XXXX` token from each PNG path and joining on a stable key (`case/day/slice`) instead of brittle string concatenations, while keeping the overall data flow and model inference logic unchanged. I also ensure `df_train` is constructed in the correct 1-row-per-slice order and that the prediction generator always has non-zero length. These are execution-unblocking fixes and should move the score up from 0.0 toward the target by producing correctly-aligned, valid RLE masks.'
- What this solution (achieved 0.0044) has done: 'I fix the root cause of the empty merge by correctly parsing `id` into the real `slice_XXXX` token (it was using only the numeric part, so it never matched PNG filenames) and by joining using a robust `case/day/slice` key extracted consistently from both the CSV ids and the discovered PNG paths. Then I keep the existing inference + RLE pipeline intact, only adjusting the dataframe construction so `df_train` is built directly from the merged rows (instead of relying on `[::3]` ordering assumptions that can silently break). These changes are execution-unblocking and should raise the score from 0.0 toward the target simply by producing non-empty, correctly-aligned predictions and a valid `submission.csv`. No model/loss/thresholding logic is changed.'
- What this solution (achieved 0.2567) has done: 'Your current score is far below the target because the fallback U-Net is being used with random weights (no pretrained weights and no training), so predictions are essentially noise. To move the score toward the target with minimal core-logic disruption, I add a lightweight training step only when `DEBUG=True` (i.e., when train masks are available) and then run inference on the real test set to generate the submission. I keep the same model architecture, same loss (`bce_dice_loss`), and the same RLE encoding/thresholding; the only changes are (1) correctly loading `train.csv` segmentations in debug mode, (2) training the existing fallback model on a small subset, and (3) switching back to test for submission inference. This should materially increase the score toward your target while keeping changes focused and within the existing pipeline.'
- What this solution (achieved 0.0) has done: 'The timeout is dominated by three things: (1) expensive `glob()` directory scans per-slice when resolving PNG paths, (2) precomputing and holding *all* train/test images and masks in memory before building the `tf.data` pipelines, and (3) slow per-sample Python loops during RLE encoding. The optimized version keeps the exact same training loop/epochs/loss/model and evaluation semantics, but replaces per-slice `glob()` with a single indexed directory scan per case/day (much faster and equivalent), streams images/masks through `tf.data` instead of precomputing full arrays (same data, just produced lazily), and parallelizes the RLE+resize postprocess with a thread pool while keeping deterministic ordering. These changes reduce wall time substantially without changing the core algorithm or accuracy (only negligible float differences may occur due to equivalent resizing paths). File paths and overall structure are preserved.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings

warnings.filterwarnings("ignore")

import gc
from glob import glob

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

try:
    import google.protobuf as _pb
    from packaging import version as _version

    _pb_ver = getattr(_pb, "__version__", "0.0.0")
    if _version.parse(_pb_ver) >= _version.parse("4.21.0"):
        import sys, subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf<4.21"]
        )
except Exception:
    pass

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

tf.random.set_seed(42)
np.random.seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

from concurrent.futures import ThreadPoolExecutor

_CPU_WORKERS = max(1, min(8, (os.cpu_count() or 2)))



## === cell 1
BATCH_SIZE = 16
EPOCHS = 30
n_splits = 5
fold_selected = 1  # 1..5



## === cell 2
df = pd.read_csv("../input/uw-madison-gi-tract-image-segmentation/train.csv")
DEBUG = True

df.rename(columns={"class": "class_name"}, inplace=True)

id_split = df["id"].str.split("_", expand=True)
df["case"] = id_split[0].str.replace("case", "", regex=False).astype(np.int32)
df["day"] = id_split[1].str.replace("day", "", regex=False).astype(np.int32)
df["slice"] = (id_split[2] + "_" + id_split[3]).astype(str)  # 'slice_0001'

TRAIN_DIR = "../input/uw-madison-gi-tract-image-segmentation/train"


def _resolve_scan_png_path(
    train_or_test_dir: str, case: int, day: int, slice_token: str
) -> str:
    pat = (
        f"{train_or_test_dir}/case{case}/case{case}_day{day}/scans/"
        f"{slice_token}_*.png"
    )
    hits = glob(pat)
    if not hits:
        return ""
    hits.sort()
    return hits[0]


def _parse_png_meta_from_path(p: str):
    """
    Returns: width(int), height(int) from filename: slice_0001_266_266_1.50_1.50.png
    """
    fname = os.path.basename(p)[:-4]
    fparts = fname.split("_")
    if len(fparts) < 6:
        raise ValueError(f"Unexpected PNG filename format: {os.path.basename(p)}")
    width = int(float(fparts[2]))
    height = int(float(fparts[3]))
    return width, height


def _build_scan_index(base_dir: str, case: int, day: int) -> dict:
    scans_dir = f"{base_dir}/case{case}/case{case}_day{day}/scans"
    if not os.path.isdir(scans_dir):
        return {}
    m = {}
    try:
        for fn in os.listdir(scans_dir):
            if not fn.endswith(".png"):
                continue
            parts = fn.split("_")
            if len(parts) < 6:
                continue
            token = parts[0] + "_" + parts[1]
            p = os.path.join(scans_dir, fn)
            prev = m.get(token)
            if prev is None or fn < os.path.basename(prev):
                m[token] = p
    except FileNotFoundError:
        return {}
    return m


def _resolve_paths_and_meta(
    df_uniq: pd.DataFrame, base_dir: str, desc: str
) -> pd.DataFrame:
    pairs = (
        df_uniq[["case", "day"]].drop_duplicates().itertuples(index=False, name=None)
    )
    idx = {}
    for case, day in tqdm(list(pairs), desc=f"Indexing {desc} scan dirs (case/day)"):
        idx[(int(case), int(day))] = _build_scan_index(base_dir, int(case), int(day))

    n = len(df_uniq)
    paths = np.empty(n, dtype=object)
    widths = np.empty(n, dtype=np.int32)
    heights = np.empty(n, dtype=np.int32)

    cases = df_uniq["case"].astype(np.int32).values
    days = df_uniq["day"].astype(np.int32).values
    slices = df_uniq["slice"].values

    for i in tqdm(range(n), desc=f"Resolving {desc} PNG paths (indexed)"):
        key = (int(cases[i]), int(days[i]))
        p = idx.get(key, {}).get(slices[i], "")
        if not p:
            paths[i] = ""
            widths[i] = -1
            heights[i] = -1
            continue
        try:
            w, h = _parse_png_meta_from_path(p)
        except Exception:
            p = ""
            w, h = -1, -1
        paths[i] = p
        widths[i] = w
        heights[i] = h

    out = df_uniq.copy()
    out["path"] = paths
    out["width"] = widths
    out["height"] = heights
    return out


uniq = df[["id", "case", "day", "slice"]].drop_duplicates().reset_index(drop=True)
uniq = _resolve_paths_and_meta(uniq, TRAIN_DIR, desc="train")

uniq = uniq[uniq["path"] != ""].copy()
if uniq.shape[0] == 0:
    raise FileNotFoundError(
        f"No PNGs resolved under {TRAIN_DIR}. Check directory structure."
    )

df = df.merge(uniq[["id", "path", "width", "height"]], on="id", how="inner")

del uniq, id_split
gc.collect()
df.head(5)



## === cell 3
df = df.sort_values(["id", "class_name"]).reset_index(drop=True)

df_wide = df.pivot_table(
    index=["id", "case", "day", "slice", "width", "height", "path"],
    columns="class_name",
    values="segmentation",
    aggfunc="first",
    fill_value="",
).reset_index()

for col in ["large_bowel", "small_bowel", "stomach"]:
    if col not in df_wide.columns:
        df_wide[col] = ""

df_train = df_wide[
    [
        "id",
        "path",
        "case",
        "day",
        "slice",
        "width",
        "height",
        "large_bowel",
        "small_bowel",
        "stomach",
    ]
].copy()

del df, df_wide
df_train.reset_index(inplace=True, drop=True)
df_train.fillna("", inplace=True)
df_train.head(5)



## === cell 4
print(df_train.shape)

df_train = df_train.sample(frac=0.30, random_state=42).reset_index(drop=True)
print(df_train.shape)

if df_train.shape[0] == 0:
    raise RuntimeError(
        "After resolving ids to PNG paths, df_train is empty. "
        "This indicates an id->slice->filename join issue."
    )



## === cell 5
gc.collect()




## === cell 6
def rle_encode(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted.

    Competition expects pixels numbered top-to-bottom then left-to-right,
    which corresponds to flatten(order='F') for a (H,W) array.
    """
    if img.ndim != 2:
        img = img.squeeze()
    if img.size == 0:
        return ""
    pixels = img.flatten(order="F")
    if pixels.max(initial=0) == 0:
        return ""
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(map(str, runs.astype(np.int64)))


def rle_decode(mask_rle, shape, color=1):
    """
    mask_rle: run-length as string formatted (start length), can be '' for empty mask
    shape: (height,width,channels) of array to return
    Returns numpy array, 1 - mask, 0 - background

    Decode matches encode's column-major convention via order='F'.
    """
    h, w, c = shape
    if (
        mask_rle is None
        or mask_rle == ""
        or (isinstance(mask_rle, float) and np.isnan(mask_rle))
    ):
        return np.zeros((h, w, c), dtype=np.float32)

    s = mask_rle.split()
    starts = np.asarray(s[0::2], dtype=np.int64) - 1
    lengths = np.asarray(s[1::2], dtype=np.int64)
    ends = starts + lengths

    mark = np.zeros(h * w + 1, dtype=np.int32)
    np.add.at(mark, starts, 1)
    np.add.at(mark, ends, -1)
    m = (np.cumsum(mark[:-1]) > 0).astype(np.float32)

    if c == 1:
        out = m.reshape((h, w, 1), order="F")
        if color != 1:
            out *= float(color)
        return out
    else:
        out = np.zeros((h * w, c), dtype=np.float32)
        out[:] = m[:, None] * np.asarray(color, dtype=np.float32)[None, :]
        return out.reshape((h, w, c), order="F")


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
_MASK_CACHE_128 = {}
_IMG_CACHE_128 = {}


def _get_mask_128_cached(rle: str, h: int, w: int) -> np.ndarray:
    key = (h, w, rle)
    m = _MASK_CACHE_128.get(key)
    if m is not None:
        return m
    m_full = rle_decode(rle, shape=(h, w, 1))  # float32 (h,w,1)
    m_128 = cv2.resize(m_full, (128, 128), interpolation=cv2.INTER_NEAREST)
    if m_128.ndim == 2:
        m_128 = m_128[..., None]
    m_128 = m_128[:, :, 0].astype(np.float32, copy=False)  # (128,128)
    _MASK_CACHE_128[key] = m_128
    return m_128


def _get_img_128_cached(path: str) -> np.ndarray:
    img = _IMG_CACHE_128.get(path)
    if img is not None:
        return img
    im = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    if im is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    im = cv2.resize(im, (128, 128), interpolation=cv2.INTER_AREA)
    im = (im.astype(np.float32) / 255.0)[..., None]  # (128,128,1)
    img3 = np.repeat(im, 3, axis=-1)  # (128,128,3)
    _IMG_CACHE_128[path] = img3
    return img3


def _precompute_masks_128(df_in: pd.DataFrame) -> np.ndarray:
    n = len(df_in)
    y = np.zeros((n, 128, 128, 3), dtype=np.float32)
    heights = df_in["height"].astype(np.int32).values
    widths = df_in["width"].astype(np.int32).values
    lb = df_in["large_bowel"].values
    sb = df_in["small_bowel"].values
    st = df_in["stomach"].values

    def _worker(i: int):
        h = int(heights[i])
        w = int(widths[i])
        return (
            i,
            _get_mask_128_cached(lb[i], h, w),
            _get_mask_128_cached(sb[i], h, w),
            _get_mask_128_cached(st[i], h, w),
        )

    with ThreadPoolExecutor(max_workers=_CPU_WORKERS) as ex:
        for i, m0, m1, m2 in tqdm(
            ex.map(_worker, range(n)),
            total=n,
            desc="Precomputing train masks (128x128)",
        ):
            y[i, :, :, 0] = m0
            y[i, :, :, 1] = m1
            y[i, :, :, 2] = m2
    return y


def _precompute_images_128(paths: np.ndarray) -> np.ndarray:
    n = len(paths)
    X = np.empty((n, 128, 128, 3), dtype=np.float32)

    def _worker(i: int):
        return i, _get_img_128_cached(paths[i])

    with ThreadPoolExecutor(max_workers=_CPU_WORKERS) as ex:
        for i, img in tqdm(
            ex.map(_worker, range(n)),
            total=n,
            desc="Precomputing images (128x128)",
        ):
            X[i] = img
    return X


gc.collect()



## === cell 8
gc.collect()



## === cell 9
from tensorflow.keras.losses import binary_crossentropy


def dice_coef(y_true, y_pred, smooth=1):
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

model_path = "../input/uwmgi-unet-keras/model.h5"


def build_fallback_unet(input_shape=(128, 128, 3), n_classes=3):
    inputs = keras.layers.Input(shape=input_shape)

    c1 = keras.layers.Conv2D(16, 3, activation="relu", padding="same")(inputs)
    c1 = keras.layers.Conv2D(16, 3, activation="relu", padding="same")(c1)
    p1 = keras.layers.MaxPooling2D()(c1)

    c2 = keras.layers.Conv2D(32, 3, activation="relu", padding="same")(p1)
    c2 = keras.layers.Conv2D(32, 3, activation="relu", padding="same")(c2)
    p2 = keras.layers.MaxPooling2D()(c2)

    b = keras.layers.Conv2D(64, 3, activation="relu", padding="same")(p2)
    b = keras.layers.Conv2D(64, 3, activation="relu", padding="same")(b)

    u2 = keras.layers.UpSampling2D()(b)
    u2 = keras.layers.Concatenate()([u2, c2])
    c3 = keras.layers.Conv2D(32, 3, activation="relu", padding="same")(u2)
    c3 = keras.layers.Conv2D(32, 3, activation="relu", padding="same")(c3)

    u1 = keras.layers.UpSampling2D()(c3)
    u1 = keras.layers.Concatenate()([u1, c1])
    c4 = keras.layers.Conv2D(16, 3, activation="relu", padding="same")(u1)
    c4 = keras.layers.Conv2D(16, 3, activation="relu", padding="same")(c4)

    outputs = keras.layers.Conv2D(n_classes, 1, activation="sigmoid", padding="same")(
        c4
    )
    m = keras.Model(inputs, outputs)
    return m


if os.path.exists(model_path):
    model = load_model(model_path, custom_objects=custom_objects)
else:
    model = build_fallback_unet((128, 128, 3), 3)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss=bce_dice_loss)

gc.collect()




## === cell 12
def _make_train_dataset(df_in: pd.DataFrame, batch_size: int) -> tf.data.Dataset:
    paths = df_in["path"].values.astype(object)
    heights = df_in["height"].astype(np.int32).values
    widths = df_in["width"].astype(np.int32).values
    lb = df_in["large_bowel"].values.astype(object)
    sb = df_in["small_bowel"].values.astype(object)
    st = df_in["stomach"].values.astype(object)

    def _py_load(path_b, h_b, w_b, lb_b, sb_b, st_b):
        path = (
            path_b.decode() if isinstance(path_b, (bytes, bytearray)) else str(path_b)
        )
        h = int(h_b)
        w = int(w_b)
        img = _get_img_128_cached(path).astype(np.float32, copy=False)  # (128,128,3)
        m0 = _get_mask_128_cached(
            lb_b.decode() if isinstance(lb_b, (bytes, bytearray)) else str(lb_b), h, w
        )
        m1 = _get_mask_128_cached(
            sb_b.decode() if isinstance(sb_b, (bytes, bytearray)) else str(sb_b), h, w
        )
        m2 = _get_mask_128_cached(
            st_b.decode() if isinstance(st_b, (bytes, bytearray)) else str(st_b), h, w
        )
        y = np.stack([m0, m1, m2], axis=-1).astype(
            np.float32, copy=False
        )  # (128,128,3)
        return img, y

    def _tf_load(path, h, w, lb_r, sb_r, st_r):
        img, y = tf.py_function(
            _py_load,
            inp=[path, h, w, lb_r, sb_r, st_r],
            Tout=[tf.float32, tf.float32],
        )
        img.set_shape((128, 128, 3))
        y.set_shape((128, 128, 3))
        return img, y

    ds = tf.data.Dataset.from_tensor_slices((paths, heights, widths, lb, sb, st))
    ds = ds.shuffle(buffer_size=len(df_in), seed=42, reshuffle_each_iteration=True)
    ds = ds.map(_tf_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = _make_train_dataset(df_train, BATCH_SIZE)
model.fit(train_ds, epochs=EPOCHS, verbose=1)

del train_ds
gc.collect()



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
UnknownError                              Traceback (most recent call last)
/tmp/ipykernel_11/2156829481.py in <cell line: 0>()
     49 
     50 train_ds = _make_train_dataset(df_train, BATCH_SIZE)
---> 51 model.fit(train_ds, epochs=EPOCHS, verbose=1)
     52 
     53 del train_ds

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
     57       e.message += " name: " + name
     58     raise core._status_to_exception(e) from None
---> 59   except TypeError as e:
     60     keras_symbolic_tensors = [x for x in inputs if _is_keras_symbolic_tensor(x)]
     61     if keras_symbolic_tensors:

UnknownError: Graph execution error:

Detected at node EagerPyFunc defined at (most recent call last):
<stack traces unavailable>
Error in user-defined function passed to MapDataset:5 transformation with iterator: Iterator::Root::Prefetch::BatchV2::Map: FileNotFoundError: Could not read image: tf.Tensor(b'../input/uw-madison-gi-tract-image-segmentation/train/case90/case90_day29/scans/slice_0089_266_266_1.50_1.50.png', shape=(), dtype=string)
Traceback (most recent call last):

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 267, in __call__
    return func(device, token, args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 145, in __call__
    outputs = self._call(device, args)
              ^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 152, in _call
    ret = self._func(*args)
          ^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py", line 643, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_11/2156829481.py", line 18, in _py_load
    img = _get_img_128_cached(path).astype(np.float32, copy=False)  # (128,128,3)
          ^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_11/1638556494.py", line 25, in _get_img_128_cached
    raise FileNotFoundError(f"Could not read image: {path}")

FileNotFoundError: Could not read image: tf.Tensor(b'../input/uw-madison-gi-tract-image-segmentation/train/case90/case90_day29/scans/slice_0089_266_266_1.50_1.50.png', shape=(), dtype=string)


	 [[{{node EagerPyFunc}}]]
	 [[IteratorGetNext]] [Op:__inference_multi_step_on_iterator_4485]

## === cell 13
df_test = pd.read_csv(
    "../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv"
)
df_test.rename(columns={"class": "class_name"}, inplace=True)

id_split = df_test["id"].str.split("_", expand=True)
df_test["case"] = id_split[0].str.replace("case", "", regex=False).astype(np.int32)
df_test["day"] = id_split[1].str.replace("day", "", regex=False).astype(np.int32)
df_test["slice"] = (id_split[2] + "_" + id_split[3]).astype(str)

TEST_DIR = "../input/uw-madison-gi-tract-image-segmentation/test"

uniq = df_test[["id", "case", "day", "slice"]].drop_duplicates().reset_index(drop=True)
uniq = _resolve_paths_and_meta(uniq, TEST_DIR, desc="test")

uniq = uniq[uniq["path"] != ""].copy()
if uniq.shape[0] == 0:
    raise FileNotFoundError(
        f"No PNGs resolved under {TEST_DIR}. Check directory structure."
    )

df_test = df_test.merge(uniq[["id", "path", "width", "height"]], on="id", how="inner")
if df_test.shape[0] == 0:
    raise RuntimeError("Test merge produced 0 rows; check parsing/join keys.")

df_test = df_test.sort_values(["id", "class_name"]).reset_index(drop=True)

df_test_wide = df_test.pivot_table(
    index=["id", "case", "day", "slice", "width", "height", "path"],
    columns="class_name",
    values="predicted",
    aggfunc="first",
    fill_value="",
).reset_index()

for col in ["large_bowel", "small_bowel", "stomach"]:
    if col not in df_test_wide.columns:
        df_test_wide[col] = ""

df_infer = df_test_wide[
    [
        "id",
        "path",
        "case",
        "day",
        "slice",
        "width",
        "height",
        "large_bowel",
        "small_bowel",
        "stomach",
    ]
].copy()
df_infer.reset_index(drop=True, inplace=True)
df_infer.fillna("", inplace=True)

del df_test, df_test_wide, uniq, id_split
gc.collect()

df_infer.head(3)




## === cell 14
def _make_test_dataset(paths: np.ndarray, batch_size: int = 32) -> tf.data.Dataset:
    paths = paths.astype(object)

    def _py_img(path_b):
        path = (
            path_b.decode() if isinstance(path_b, (bytes, bytearray)) else str(path_b)
        )
        return _get_img_128_cached(path).astype(np.float32, copy=False)

    def _tf_img(path):
        img = tf.py_function(_py_img, inp=[path], Tout=tf.float32)
        img.set_shape((128, 128, 3))
        return img

    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(_tf_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


test_ds = _make_test_dataset(df_infer["path"].values, batch_size=32)
gc.collect()
LOGITS = model.predict(test_ds, verbose=1)
gc.collect()

del test_ds
gc.collect()



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
UnknownError                              Traceback (most recent call last)
/tmp/ipykernel_11/2704749226.py in <cell line: 0>()
     22 test_ds = _make_test_dataset(df_infer["path"].values, batch_size=32)
     23 gc.collect()
---> 24 LOGITS = model.predict(test_ds, verbose=1)
     25 gc.collect()
     26 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

UnknownError: {{function_node __wrapped__IteratorGetNext_output_types_1_device_/job:localhost/replica:0/task:0/device:CPU:0}} Error in user-defined function passed to MapDataset:12 transformation with iterator: Iterator::Root::Prefetch::BatchV2::Map: FileNotFoundError: Could not read image: tf.Tensor(b'../input/uw-madison-gi-tract-image-segmentation/test/case110/case110_day12/scans/slice_0001_360_310_1.50_1.50.png', shape=(), dtype=string)
Traceback (most recent call last):

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 267, in __call__
    return func(device, token, args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 145, in __call__
    outputs = self._call(device, args)
              ^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 152, in _call
    ret = self._func(*args)
          ^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py", line 643, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_11/2704749226.py", line 9, in _py_img
    return _get_img_128_cached(path).astype(np.float32, copy=False)
           ^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_11/1638556494.py", line 25, in _get_img_128_cached
    raise FileNotFoundError(f"Could not read image: {path}")

FileNotFoundError: Could not read image: tf.Tensor(b'../input/uw-madison-gi-tract-image-segmentation/test/case110/case110_day12/scans/slice_0001_360_310_1.50_1.50.png', shape=(), dtype=string)


	 [[{{node EagerPyFunc}}]] [Op:IteratorGetNext] name: 

## === cell 15
len(LOGITS)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/571340957.py in <cell line: 0>()
----> 1 len(LOGITS)
      2 

NameError: name 'LOGITS' is not defined

## === cell 16
n_inf = df_infer.shape[0]
lbs = np.empty(n_inf, dtype=object)
sbs = np.empty(n_inf, dtype=object)
sts = np.empty(n_inf, dtype=object)

heights = df_infer["height"].astype(np.int32).values
widths = df_infer["width"].astype(np.int32).values

resize = cv2.resize
INTERP = cv2.INTER_NEAREST
thr = 0.5
enc = rle_encode


def _post_worker(i: int):
    h = int(heights[i])
    w = int(widths[i])
    pred = LOGITS[i]
    pred0 = resize(pred[:, :, 0], (w, h), interpolation=INTERP)
    pred1 = resize(pred[:, :, 1], (w, h), interpolation=INTERP)
    pred2 = resize(pred[:, :, 2], (w, h), interpolation=INTERP)
    r0 = enc((pred0 >= thr).astype("uint8", copy=False))
    r1 = enc((pred1 >= thr).astype("uint8", copy=False))
    r2 = enc((pred2 >= thr).astype("uint8", copy=False))
    return i, r0, r1, r2


with ThreadPoolExecutor(max_workers=_CPU_WORKERS) as ex:
    for i, r0, r1, r2 in tqdm(
        ex.map(_post_worker, range(n_inf)),
        total=n_inf,
        desc="Postprocess + RLE",
    ):
        lbs[i] = r0
        sbs[i] = r1
        sts[i] = r2

del LOGITS
gc.collect()



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3637066733.py in <cell line: 0>()
     29 
     30 with ThreadPoolExecutor(max_workers=_CPU_WORKERS) as ex:
---> 31     for i, r0, r1, r2 in tqdm(
     32         ex.map(_post_worker, range(n_inf)),
     33         total=n_inf,

/usr/local/lib/python3.11/dist-packages/tqdm/std.py in __iter__(self)
   1179 
   1180         try:
-> 1181             for obj in iterable:
   1182                 yield obj
   1183                 # Update and possibly print the progressbar.

/usr/lib/python3.11/concurrent/futures/_base.py in result_iterator()
    617                     # Careful not to keep a reference to the popped future
    618                     if timeout is None:
--> 619                         yield _result_or_cancel(fs.pop())
    620                     else:
    621                         yield _result_or_cancel(fs.pop(), end_time - time.monotonic())

/usr/lib/python3.11/concurrent/futures/_base.py in _result_or_cancel(***failed resolving arguments***)
    315     try:
    316         try:
--> 317             return fut.result(timeout)
    318         finally:
    319             fut.cancel()

/usr/lib/python3.11/concurrent/futures/_base.py in result(self, timeout)
    447                     raise CancelledError()
    448                 elif self._state == FINISHED:
--> 449                     return self.__get_result()
    450 
    451                 self._condition.wait(timeout)

/usr/lib/python3.11/concurrent/futures/_base.py in __get_result(self)
    399         if self._exception:
    400             try:
--> 401                 raise self._exception
    402             finally:
    403                 # Break a reference cycle with the exception in self._exception

/usr/lib/python3.11/concurrent/futures/thread.py in run(self)
     56 
     57         try:
---> 58             result = self.fn(*self.args, **self.kwargs)
     59         except BaseException as exc:
     60             self.future.set_exception(exc)

/tmp/ipykernel_11/3637066733.py in _post_worker(i)
     18     h = int(heights[i])
     19     w = int(widths[i])
---> 20     pred = LOGITS[i]
     21     pred0 = resize(pred[:, :, 0], (w, h), interpolation=INTERP)
     22     pred1 = resize(pred[:, :, 1], (w, h), interpolation=INTERP)

NameError: name 'LOGITS' is not defined

## === cell 17
df_ids = df_infer[["id"]].copy()
gc.collect()



## === cell 18
id_arr = df_ids["id"].values
ids = np.repeat(id_arr, 3)
classes = np.tile(
    np.array(["large_bowel", "small_bowel", "stomach"], dtype=object), len(id_arr)
)
rles = np.empty(len(ids), dtype=object)
rles[0::3] = lbs
rles[1::3] = sbs
rles[2::3] = sts



## === cell 19
sub = pd.DataFrame({"id": ids, "class": classes, "predicted": rles})

sample = pd.read_csv(
    "../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv"
)
submission = sample.merge(sub, on=["id", "class"], how="left", suffixes=("", "_pred"))

if "predicted_pred" in submission.columns:
    submission["predicted"] = submission["predicted_pred"].fillna("")
    submission = submission[["id", "class", "predicted"]]
else:
    submission["predicted"] = submission["predicted"].fillna("")

submission.to_csv("submission.csv", index=False)



## === cell 20
submission.head()
