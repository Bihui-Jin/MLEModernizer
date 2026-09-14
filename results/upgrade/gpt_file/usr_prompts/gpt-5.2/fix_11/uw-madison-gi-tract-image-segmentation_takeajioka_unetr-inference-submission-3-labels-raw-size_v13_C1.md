# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os, glob, gc, json, re
import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(0)

import torch

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False



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
from concurrent.futures import ThreadPoolExecutor

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


def _quantile_from_hist(hist: np.ndarray, q: float) -> int:
    if q <= 0:
        return 0
    if q >= 1:
        return 255
    cdf = np.cumsum(hist)
    total = cdf[-1]
    if total <= 0:
        return 0
    target = int(np.ceil(q * total))
    idx = int(np.searchsorted(cdf, target, side="left"))
    return int(np.clip(idx, 0, 255))


def load_image_volume(img_dir, quant=0.01, max_workers=8):
    """
    Same logic/outputs as before: read all slices -> stack -> optional quantile clip
    -> min/max normalize to uint8.
    Optimized: histogram-based quantiles + LUT transform avoids expensive float percentiles.
    """
    imgs = _list_pngs(img_dir)

    def _read_one(p):
        with Image.open(p) as im:
            return np.asarray(im, dtype=np.uint8)

    arr0 = _read_one(imgs[0])
    Z = len(imgs)
    H, W = arr0.shape
    vol_u8 = np.empty((Z, H, W), dtype=np.uint8)
    vol_u8[0] = arr0

    if Z > 1:
        n_workers = min(max_workers, Z - 1)
        if n_workers <= 1:
            for i in range(1, Z):
                vol_u8[i] = _read_one(imgs[i])
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                for i, arr in enumerate(ex.map(_read_one, imgs[1:]), start=1):
                    vol_u8[i] = arr

    if quant is not None and quant > 0:
        hist = np.bincount(vol_u8.ravel(), minlength=256).astype(np.int64)
        lo = _quantile_from_hist(hist, float(quant))
        hi = _quantile_from_hist(hist, float(1.0 - quant))
        if hi < lo:
            lo, hi = hi, lo
        if hi == lo:
            return np.zeros_like(vol_u8, dtype=np.uint8)

        lut = np.arange(256, dtype=np.int32)
        lut = np.clip(lut, lo, hi) - lo
        lut = (lut.astype(np.float32) * (255.0 / float(hi - lo))).astype(np.uint8)
        return lut[vol_u8]
    else:
        v_min = int(vol_u8.min())
        v_max = int(vol_u8.max())
        denom = v_max - v_min
        if denom <= 0:
            return np.zeros_like(vol_u8, dtype=np.uint8)
        lut = (np.clip(np.arange(256, dtype=np.int32), v_min, v_max) - v_min).astype(
            np.float32
        )
        lut = (lut * (255.0 / float(denom))).astype(np.uint8)
        return lut[vol_u8]




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




## === cell 14
sz = (80, 144, 192)




## === cell 15
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




## === cell 16
def simple_area_filter(mask2d: np.ndarray, min_area: int = 64) -> np.ndarray:
    if int(mask2d.sum()) < int(min_area):
        return np.zeros_like(mask2d, dtype=np.uint8)
    return (mask2d > 0).astype(np.uint8)


from numpy.lib.stride_tricks import sliding_window_view


def _max_filter3x3_u8(mask_u8: np.ndarray) -> np.ndarray:
    p = np.pad(mask_u8, ((1, 1), (1, 1)), mode="constant", constant_values=0)
    w = sliding_window_view(p, (3, 3))
    return w.max(axis=(-1, -2)).astype(np.uint8, copy=False)


def _min_filter3x3_u8(mask_u8: np.ndarray) -> np.ndarray:
    p = np.pad(mask_u8, ((1, 1), (1, 1)), mode="constant", constant_values=0)
    w = sliding_window_view(p, (3, 3))
    return w.min(axis=(-1, -2)).astype(np.uint8, copy=False)


def binary_dilate(mask: np.ndarray, iters: int = 1) -> np.ndarray:
    x = (mask > 0).astype(np.uint8, copy=False)
    for _ in range(int(iters)):
        x = _max_filter3x3_u8(x)
    return x


def binary_erode(mask: np.ndarray, iters: int = 1) -> np.ndarray:
    x = (mask > 0).astype(np.uint8, copy=False)
    for _ in range(int(iters)):
        x = _min_filter3x3_u8(x)
    return x


def binary_close(mask: np.ndarray, iters: int = 1) -> np.ndarray:
    return binary_erode(binary_dilate(mask, iters=iters), iters=iters)


def binary_open(mask: np.ndarray, iters: int = 1) -> np.ndarray:
    return binary_dilate(binary_erode(mask, iters=iters), iters=iters)


def _connected_components_2d_u8(
    mask2d_u8: np.ndarray, connectivity: int = 8
) -> np.ndarray:
    m = mask2d_u8 != 0
    H, W = m.shape
    labels = np.zeros((H, W), dtype=np.int32)
    if m.sum() == 0:
        return labels

    parent = np.zeros(H * W + 1, dtype=np.int32)
    rank = np.zeros(H * W + 1, dtype=np.uint8)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    next_label = 0
    for y in range(H):
        row_fg = m[y]
        for x in range(W):
            if not row_fg[x]:
                continue
            neigh_labels = []
            if y > 0:
                if m[y - 1, x]:
                    neigh_labels.append(labels[y - 1, x])
                if connectivity == 8:
                    if x > 0 and m[y - 1, x - 1]:
                        neigh_labels.append(labels[y - 1, x - 1])
                    if x + 1 < W and m[y - 1, x + 1]:
                        neigh_labels.append(labels[y - 1, x + 1])
            if x > 0 and m[y, x - 1]:
                neigh_labels.append(labels[y, x - 1])

            if not neigh_labels:
                next_label += 1
                labels[y, x] = next_label
                parent[next_label] = next_label
            else:
                nl = [v for v in neigh_labels if v != 0]
                if not nl:
                    next_label += 1
                    labels[y, x] = next_label
                    parent[next_label] = next_label
                else:
                    min_l = int(min(nl))
                    labels[y, x] = min_l
                    for other in nl:
                        if other != min_l:
                            union(min_l, int(other))

    if next_label == 1:
        return labels

    roots = np.arange(next_label + 1, dtype=np.int32)
    for i in range(1, next_label + 1):
        if parent[i] != 0:
            roots[i] = find(i)

    uniq = np.unique(roots[1 : next_label + 1])
    remap = np.zeros(next_label + 1, dtype=np.int32)
    remap[uniq] = np.arange(1, uniq.size + 1, dtype=np.int32)

    out = labels
    nz = out != 0
    out[nz] = remap[roots[out[nz]]]
    return out


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    m = (mask > 0).astype(np.uint8, copy=False)
    if int(m.sum()) == 0:
        return m
    labels = _connected_components_2d_u8(m, connectivity=8)
    max_label = int(labels.max())
    if max_label <= 1:
        return m
    cnt = np.bincount(labels.ravel(), minlength=max_label + 1)
    cnt[0] = 0
    best = int(cnt.argmax())
    return (labels == best).astype(np.uint8, copy=False)


_otsu_cache = {}


def _otsu_cached(arr_u8: np.ndarray) -> int:
    key = (arr_u8.shape[0], arr_u8.shape[1], hash(arr_u8.tobytes()))
    v = _otsu_cache.get(key)
    if v is not None:
        return v
    t = otsu_threshold_uint8(arr_u8)
    _otsu_cache[key] = t
    return t


def predict_volume_to_segm(vol_zyx_uint8: np.ndarray, classes: list[str]) -> np.ndarray:
    """
    Same heuristic approach/core flow; minimal, deterministic refinements to improve score:
    - Tighten ROI by keeping only the largest component (reduces background false positives).
    - Create more class-distinct masks:
        * large_bowel: band between low and mid thresholds (ring-like structures)
        * small_bowel: mid-threshold within ROI
        * stomach: high-threshold then slight dilation (stomach tends to be brighter/fluid)
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
        roi = keep_largest_component(roi)
        roi = binary_open(roi, iters=1)
        roi = simple_area_filter(roi, min_area=max(512, (H * W) // 18))
        if roi.sum() == 0:
            continue

        t_global = _otsu_cached(sl)

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

        m_band = (m_low & (1 - m_mid)).astype(np.uint8)  # for large_bowel
        m_band = binary_close(m_band, iters=1)

        m_stom = binary_dilate(m_high, iters=1)  # stomach slightly expanded
        m_stom = binary_close(m_stom, iters=1)

        for c, cls in enumerate(classes):
            if cls == "small_bowel":
                m = m_mid
            elif cls == "large_bowel":
                m = m_band
            elif cls == "stomach":
                m = m_stom
            else:
                m = m_mid

            m = simple_area_filter(
                m, min_area=min_area_map.get(cls, max(64, (H * W) // 600))
            )

            if m.sum() > 0:
                m = keep_largest_component(m)

            segm[c, z] = m

    return segm




## === cell 17
def segm_rle(segm, df_vol):
    """
    segm expected shape: (C, Z, H, W), where C aligns to sorted(df_vol['class'].unique()).
    """
    dfv = df_vol[["id", "class", "Slice"]].copy()

    lbs = sorted(dfv["class"].unique())
    lb_to_c = {lb: i for i, lb in enumerate(lbs)}

    slice_idx = dfv["Slice"].astype(np.int32).to_numpy() - 1
    cls_arr = dfv["class"].to_numpy()
    c_idx = np.fromiter(
        (lb_to_c[c] for c in cls_arr), dtype=np.int32, count=len(cls_arr)
    )

    pred = np.empty(len(dfv), dtype=object)
    for i in range(len(dfv)):
        pred[i] = rle_encode((segm[c_idx[i], slice_idx[i]] > 0.5).astype(np.uint8))

    dfv["predicted"] = pred
    del segm
    gc.collect()
    return dfv.loc[:, ["id", "class", "predicted"]].reset_index(drop=True)


pred_parts = []

for (CASE, DAY), df_ in df_train.groupby(["Case", "Day"], sort=False):
    CASE = int(CASE)
    DAY = int(DAY)
    IMAGE_FOLDER = os.path.join(
        DATASET_FOLDER, folder, f"case{CASE}", f"case{CASE}_day{DAY}", "scans"
    )

    vol = load_image_volume(img_dir=IMAGE_FOLDER)  # (Z,H,W) uint8
    print("heuristic inputs:", tuple(vol.shape), "->", f"case{CASE}_day{DAY}")

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



## === cell 18
df_train = df_train.drop(columns=["predicted"], errors="ignore").merge(
    pred_df, on=["id", "class"], how="left"
)
df_train["predicted"] = df_train["predicted"].fillna("")
df_train.head()



## === cell 19
pred_df.head()



## === cell 20
sub_template = pd.read_csv(os.path.join(DATASET_FOLDER, "sample_submission.csv"))
sub_template = sub_template.drop(columns=["predicted"], errors="ignore")
sub_df = sub_template.merge(
    df_train[["id", "class", "predicted"]], on=["id", "class"], how="left"
)
sub_df["predicted"] = sub_df["predicted"].fillna("")
sub_df.head()



## === cell 21
sub_df = sub_df.loc[:, ["id", "class", "predicted"]]
sub_df["predicted"] = sub_df["predicted"].astype(str).replace("nan", "")
sub_df.head()



## === cell 22
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)



## === cell 23
sub_df



## === cell 24
assert os.path.exists("submission.csv")
assert list(sub_df.columns) == ["id", "class", "predicted"]
assert len(sub_df) == len(
    pd.read_csv(os.path.join(DATASET_FOLDER, "sample_submission.csv"))
)
print("submission.csv looks valid.")
