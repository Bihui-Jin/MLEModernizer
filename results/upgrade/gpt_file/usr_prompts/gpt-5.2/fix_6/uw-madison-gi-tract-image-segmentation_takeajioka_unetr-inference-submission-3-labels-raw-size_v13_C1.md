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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
nibabel==5.3.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
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

0.8428073683343751

# 6. Current score

0.15716

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the immediate runtime blocker by removing the dependency on `monai_weekly` (not available) and instead using the already-installed `monai` package for UNETR/inference. I also fix Pandas 2.x incompatibilities (`DataFrame.append` removal) by switching to list-accumulation + `pd.concat`, and correct JSON writing so `load_decathlon_datalist` reads a proper dict (not a quoted string). Finally, I make file/path handling robust (choose the correct dataset root, safe normalization when volume is constant, CPU fallback if CUDA is unavailable) and ensure a valid `submission.csv` with the required columns and ordering is always produced.'
- What this solution (achieved 0.0158) has done: 'I fix the end-to-end failures by removing the hard dependency on MONAI (not installed in this environment) and the missing external pretrained weights path, while keeping the same overall pipeline structure (build volumes → iterate per case/day → produce RLE per slice → merge into sample_submission). To move the score up from 0.0 toward the target, I generate a simple, valid binary segmentation per slice using robust intensity-thresholding (Otsu on the 8-bit normalized volume you already create), which is a minimal “model replacement” necessary only because the intended model cannot run here. I also correct a shape/axis issue in `segm_rle` (your volumes are `(Z,H,W)` but `segm_rle` expects `(C,Z,H,W)`) and make RLE encoding match the competition’s required ordering (column-major/Fortran order). Finally, the script always write `submission.csv` with the required `id,class,predicted` columns and correct row alignment to the provided `sample_submission.csv`.'
- What this solution (achieved 0.01536) has done: 'Your current score (0.0158) is far below the target (0.8428), so we need a real inference pipeline improvement, but still with minimal changes and preserving your existing “load per volume → predict per slice → RLE encode → merge into sample submission” core flow. The biggest issue is that you currently predict the same binary mask for all three classes, which is heavily penalized by the metric; we instead produce three different masks using simple intensity-based heuristics that are still lightweight and deterministic. Concretely, we (1) compute a per-slice body/ROI mask to suppress background, (2) use multiple Otsu thresholds (low/mid/high) within the ROI plus simple morphology to generate distinct candidate masks per class, and (3) add a tiny per-class minimum-area filter so empty predictions are preferred over obvious false positives. This keeps the same structure and RLE semantics, but should move the score substantially upward toward the target.'
- What this solution (achieved 0.15716) has done: 'The timeout is dominated by repeatedly loading every scan volume twice (once to write NIfTI in cell 8, again to predict in cell 20) plus extremely slow Python-level connected-components (`keep_largest_component`) and per-slice pandas groupby/iterrows in `segm_rle`. I remove the redundant NIfTI creation loop (it is unused by submission generation), cache each loaded volume path list per case/day, and speed up image loading using PIL’s `Image.open(...).getdata()`-based conversion with preallocation to avoid repeated list growth. I replace `keep_largest_component` with a provably equivalent connected-components implementation using `torchvision.ops.masks_to_boxes` is not applicable; instead I use `scipy` isn’t available, so I keep pure-numpy but make it linear-time with an explicit stack over only foreground pixels (coordinate list) and avoid scanning the full image repeatedly. Finally, I vectorize `segm_rle` by building slice-to-row indices once and avoid per-row `.loc` assignment/iterrows, preserving identical RLE encoding semantics.'

# 9. Code solution

## === cell 0
import os, glob, gc, json, re
import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(0)



## === cell 1
DATASET_FOLDER = "../input/uw-madison-gi-tract-image-segmentation"
sub_path = os.path.join(DATASET_FOLDER, "sample_submission.csv")
sub_df = pd.read_csv(sub_path)
sub = len(sub_df) > 0
sub



## === cell 2
if sub:
    df_train = pd.read_csv(os.path.join(DATASET_FOLDER, "test.csv"))
    folder = "test"
    df_train["predicted"] = ""
else:
    df_train = pd.read_csv(os.path.join(DATASET_FOLDER, "train.csv"))[:1000]
    df_train = df_train.rename(columns={"segmentation": "predicted"})
    folder = "train"

df_train.head()



## === cell 3
_id_pat = re.compile(r"case(\d+)_day(\d+)_slice_(\d+)")
m = df_train["id"].str.extract(_id_pat)
df_train["Case"] = m[0].astype(np.int32)
df_train["Day"] = m[1].astype(np.int32)
df_train["Slice"] = m[2]  # keep as string to preserve original grouping semantics
df_train.head()



## === cell 4
df_train_overview = (
    df_train.groupby(["Case", "Day"], sort=False)
    .size()
    .reset_index(name="Slices")
    .astype({"Case": np.int32, "Day": np.int32, "Slices": np.int32})
)
df_train_overview.head()



## === cell 5
from PIL import Image

_img_list_cache = {}


def _list_pngs(img_dir: str):
    key = img_dir
    v = _img_list_cache.get(key)
    if v is not None:
        return v
    imgs = glob.glob(os.path.join(img_dir, "*.png"))
    imgs.sort()
    if len(imgs) == 0:
        raise FileNotFoundError(f"No PNG slices found in: {img_dir}")
    _img_list_cache[key] = imgs
    return imgs


def load_image_volume(img_dir, quant=0.01):
    """
    Same logic/outputs as before: read all slices -> stack -> optional quantile clip
    -> min/max normalize to uint8.
    Speedups: avoid building a Python list of arrays; preallocate volume and fill.
    """
    imgs = _list_pngs(img_dir)
    with Image.open(imgs[0]) as im0:
        arr0 = np.asarray(im0, dtype=np.uint8)
    Z = len(imgs)
    H, W = arr0.shape
    vol_u8 = np.empty((Z, H, W), dtype=np.uint8)
    vol_u8[0] = arr0

    for i in range(1, Z):
        with Image.open(imgs[i]) as im:
            vol_u8[i] = np.asarray(im, dtype=np.uint8)

    vol = vol_u8.astype(np.float32, copy=False)

    if quant is not None and quant > 0:
        q_low, q_high = np.percentile(vol, [quant * 100, (1 - quant) * 100])
        vol = np.clip(vol, q_low, q_high, out=vol)

    v_min = float(vol.min())
    v_max = float(vol.max())
    denom = v_max - v_min
    if denom < 1e-8:
        out = np.zeros_like(vol_u8, dtype=np.uint8)
    else:
        vol = (vol - v_min) / denom
        vol = vol * 255.0
        out = vol.astype(np.uint8)

    return out




## === cell 6
pass



## === cell 7
pass



## === cell 8
pass



## === cell 9
pass



## === cell 10
pass



## === cell 11
pass



## === cell 12
pass



## === cell 13
pass




## === cell 14
def otsu_threshold_uint8(img2d_uint8: np.ndarray) -> int:
    """Compute Otsu threshold for a single uint8 image (1D or 2D) without external deps."""
    x = img2d_uint8.ravel()
    if x.size == 0:
        return 0
    hist = np.bincount(x, minlength=256).astype(np.float64)
    total = x.size
    prob = hist / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]

    denom = omega * (1.0 - omega)
    denom[denom == 0] = np.nan
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom

    t = int(np.nanargmax(sigma_b2))
    if not np.isfinite(t):
        t = 0
    return t




## === cell 15
sz = (80, 144, 192)




## === cell 16
def rle_decode(mask_rle, shape):
    if (
        mask_rle is None
        or (isinstance(mask_rle, float) and np.isnan(mask_rle))
        or str(mask_rle).strip() == ""
    ):
        return np.zeros(shape, dtype=np.uint8)
    s = str(mask_rle).split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")


def rle_encode(img):
    if img is None:
        return ""
    img = (img > 0).astype(np.uint8)
    if img.sum() == 0:
        return ""
    pixels = img.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 17
def segm_rle(segm, df_vol):
    """
    segm expected shape: (C, Z, H, W), where C aligns to sorted(df_vol['class'].unique()).
    """
    df_vol = df_vol.copy()
    df_vol["predicted"] = df_vol["predicted"].replace(np.nan, "")

    lbs = sorted(df_vol["class"].unique())
    lb_to_c = {lb: i for i, lb in enumerate(lbs)}

    pred = np.empty(len(df_vol), dtype=object)
    pred[:] = ""

    slice_idx = df_vol["Slice"].astype(np.int32).to_numpy() - 1
    cls_list = df_vol["class"].to_numpy()

    for i in range(len(df_vol)):
        c = lb_to_c[cls_list[i]]
        z = slice_idx[i]
        mask = segm[c, z]
        pred[i] = rle_encode((mask > 0.5).astype(np.uint8))

    df_vol.loc[:, "predicted"] = pred
    del segm
    gc.collect()
    return df_vol.loc[:, ["id", "class", "predicted"]].reset_index(drop=True)




## === cell 18
def simple_area_filter(mask2d: np.ndarray, min_area: int = 64) -> np.ndarray:
    if int(mask2d.sum()) < int(min_area):
        return np.zeros_like(mask2d, dtype=np.uint8)
    return mask2d.astype(np.uint8)


def binary_dilate(mask: np.ndarray, iters: int = 1) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    for _ in range(iters):
        p = np.pad(m, ((1, 1), (1, 1)), mode="constant", constant_values=0)
        m = (
            p[0:-2, 0:-2]
            | p[0:-2, 1:-1]
            | p[0:-2, 2:]
            | p[1:-1, 0:-2]
            | p[1:-1, 1:-1]
            | p[1:-1, 2:]
            | p[2:, 0:-2]
            | p[2:, 1:-1]
            | p[2:, 2:]
        ).astype(np.uint8)
    return m


def binary_erode(mask: np.ndarray, iters: int = 1) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    for _ in range(iters):
        p = np.pad(m, ((1, 1), (1, 1)), mode="constant", constant_values=0)
        m = (
            p[0:-2, 0:-2]
            & p[0:-2, 1:-1]
            & p[0:-2, 2:]
            & p[1:-1, 0:-2]
            & p[1:-1, 1:-1]
            & p[1:-1, 2:]
            & p[2:, 0:-2]
            & p[2:, 1:-1]
            & p[2:, 2:]
        ).astype(np.uint8)
    return m


def binary_close(mask: np.ndarray, iters: int = 1) -> np.ndarray:
    return binary_erode(binary_dilate(mask, iters=iters), iters=iters)


def binary_open(mask: np.ndarray, iters: int = 1) -> np.ndarray:
    return binary_dilate(binary_erode(mask, iters=iters), iters=iters)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    if m.sum() == 0:
        return m

    H, W = m.shape
    visited = np.zeros((H, W), dtype=np.uint8)

    ys, xs = np.nonzero(m)
    best_size = 0
    best_coords_y = None
    best_coords_x = None

    for start_y, start_x in zip(ys, xs):
        if visited[start_y, start_x]:
            continue

        stack_y = [int(start_y)]
        stack_x = [int(start_x)]
        visited[start_y, start_x] = 1

        coords_y = []
        coords_x = []
        while stack_y:
            cy = stack_y.pop()
            cx = stack_x.pop()
            coords_y.append(cy)
            coords_x.append(cx)

            ny = cy - 1
            if ny >= 0 and m[ny, cx] and not visited[ny, cx]:
                visited[ny, cx] = 1
                stack_y.append(ny)
                stack_x.append(cx)
            ny = cy + 1
            if ny < H and m[ny, cx] and not visited[ny, cx]:
                visited[ny, cx] = 1
                stack_y.append(ny)
                stack_x.append(cx)
            nx = cx - 1
            if nx >= 0 and m[cy, nx] and not visited[cy, nx]:
                visited[cy, nx] = 1
                stack_y.append(cy)
                stack_x.append(nx)
            nx = cx + 1
            if nx < W and m[cy, nx] and not visited[cy, nx]:
                visited[cy, nx] = 1
                stack_y.append(cy)
                stack_x.append(nx)

        comp_size = len(coords_y)
        if comp_size > best_size:
            best_size = comp_size
            best_coords_y = coords_y
            best_coords_x = coords_x

    out = np.zeros_like(m, dtype=np.uint8)
    if best_coords_y is not None:
        out[
            np.asarray(best_coords_y, dtype=np.int32),
            np.asarray(best_coords_x, dtype=np.int32),
        ] = 1
    return out


def predict_volume_to_segm(vol_zyx_uint8: np.ndarray, classes: list[str]) -> np.ndarray:
    """
    Same heuristic approach as provided; only relies on sped-up keep_largest_component.
    """
    Z, H, W = vol_zyx_uint8.shape
    C = len(classes)
    segm = np.zeros((C, Z, H, W), dtype=np.float32)

    min_area_map = {
        "large_bowel": max(96, (H * W) // 450),
        "small_bowel": max(64, (H * W) // 600),
        "stomach": max(96, (H * W) // 500),
    }

    for z in range(Z):
        sl = vol_zyx_uint8[z]

        low_thr = int(np.percentile(sl, 15))
        roi = (sl > low_thr).astype(np.uint8)
        roi = binary_close(roi, iters=2)
        roi = binary_open(roi, iters=1)
        roi = simple_area_filter(roi, min_area=max(512, (H * W) // 18))
        if roi.sum() == 0:
            continue

        t_global = otsu_threshold_uint8(sl)

        sl_roi_vals = sl[roi > 0]
        if sl_roi_vals.size > 128:
            t_roi = otsu_threshold_uint8(sl_roi_vals.astype(np.uint8))
        else:
            t_roi = t_global

        t_low = int(np.clip(0.70 * t_roi, 0, 255))
        t_mid = int(np.clip(1.00 * t_roi, 0, 255))
        t_high = int(np.clip(1.20 * t_roi, 0, 255))

        m_low = ((sl > t_low) & (roi > 0)).astype(np.uint8)
        m_mid = ((sl > t_mid) & (roi > 0)).astype(np.uint8)
        m_high = ((sl > t_high) & (roi > 0)).astype(np.uint8)

        m_low = binary_open(m_low, iters=1)
        m_low = binary_close(m_low, iters=2)

        m_mid = binary_open(m_mid, iters=1)
        m_mid = binary_close(m_mid, iters=1)

        m_high = binary_open(m_high, iters=1)
        m_high = binary_close(m_high, iters=1)

        for c, cls in enumerate(classes):
            if cls == "small_bowel":
                m = m_mid
            elif cls == "large_bowel":
                m = m_low
            elif cls == "stomach":
                m = m_high
            else:
                m = m_mid

            m = simple_area_filter(
                m, min_area=min_area_map.get(cls, max(64, (H * W) // 600))
            )

            if m.sum() > 0:
                m = keep_largest_component(m)

            segm[c, z] = m

    return segm




## === cell 19
pred_parts = []

for i in range(len(df_train_overview)):
    CASE = int(df_train_overview.loc[i, "Case"])
    DAY = int(df_train_overview.loc[i, "Day"])
    IMAGE_FOLDER = os.path.join(
        DATASET_FOLDER, folder, f"case{CASE}", f"case{CASE}_day{DAY}", "scans"
    )

    vol = load_image_volume(img_dir=IMAGE_FOLDER)  # (Z,H,W) uint8
    print("heuristic inputs:", tuple(vol.shape), "->", f"case{CASE}_day{DAY}")

    df_ = df_train[(df_train["Case"] == CASE) & (df_train["Day"] == DAY)]
    classes = sorted(df_["class"].unique())
    segm = predict_volume_to_segm(vol, classes=classes)  # (C,Z,H,W)

    pred_parts.append(segm_rle(segm, df_))

    del vol, segm, df_
    gc.collect()

pred_df = (
    pd.concat(pred_parts, axis=0, ignore_index=True)
    if pred_parts
    else pd.DataFrame(columns=["id", "class", "predicted"])
)
pred_df.head(), len(pred_df)



## === cell 20
df_train = df_train.drop(columns=["predicted"], errors="ignore").merge(
    pred_df, on=["id", "class"], how="left"
)
df_train["predicted"] = df_train["predicted"].fillna("")
df_train.head()



## === cell 21
pred_df.head()



## === cell 22
sub_template = pd.read_csv(os.path.join(DATASET_FOLDER, "sample_submission.csv"))
sub_template = sub_template.drop(columns=["predicted"], errors="ignore")
sub_df = sub_template.merge(
    df_train[["id", "class", "predicted"]], on=["id", "class"], how="left"
)
sub_df["predicted"] = sub_df["predicted"].fillna("")
sub_df.head()



## === cell 23
sub_df = sub_df.loc[:, ["id", "class", "predicted"]]
sub_df["predicted"] = sub_df["predicted"].astype(str).replace("nan", "")
sub_df.head()



## === cell 24
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)



## === cell 25
sub_df



## === cell 26
assert os.path.exists("submission.csv")
assert list(sub_df.columns) == ["id", "class", "predicted"]
assert len(sub_df) == len(
    pd.read_csv(os.path.join(DATASET_FOLDER, "sample_submission.csv"))
)
print("submission.csv looks valid.")
