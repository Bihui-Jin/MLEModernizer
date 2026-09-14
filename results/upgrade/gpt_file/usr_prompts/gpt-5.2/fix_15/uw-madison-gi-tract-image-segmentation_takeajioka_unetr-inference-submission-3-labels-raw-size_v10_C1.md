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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import glob, os, gc, json, math, time

try:
    from IPython.display import display  # type: ignore
except Exception:

    def display(x):
        print(x)


import random

random.seed(0)
np.random.seed(0)



## === cell 1
DATASET_FOLDER = "/kaggle/input/uw-madison-gi-tract-image-segmentation"

sub_df = pd.read_csv(os.path.join(DATASET_FOLDER, "sample_submission.csv"))
sub = True if len(sub_df) else False
print("sub:", sub, "sample_submission rows:", len(sub_df))



## === cell 2
if sub is True:
    path_csv = os.path.join(DATASET_FOLDER, "test.csv")
    df_train = pd.read_csv(path_csv)
    df_train["predicted"] = ""
    folder = "test"
else:
    path_csv = os.path.join(DATASET_FOLDER, "train.csv")
    df_train = pd.read_csv(path_csv)[:1000]
    df_train = df_train.rename(columns={"segmentation": "predicted"})
    folder = "train"
display(df_train.head())




## === cell 3
def extract_details(id_):
    id_fields = id_.split("_")
    case = id_fields[0].replace("case", "")
    day = id_fields[1].replace("day", "")
    slice_id = id_fields[3]
    return {
        "Case": int(case),
        "Day": int(day),
        "Slice": slice_id,
    }




## === cell 4
_id = df_train["id"].astype(str)
parts = _id.str.split("_", expand=True)
df_train["Case"] = parts[0].str.replace("case", "", regex=False).astype(np.int32)
df_train["Day"] = parts[1].str.replace("day", "", regex=False).astype(np.int32)
df_train["Slice"] = parts[3]
del _id, parts
gc.collect()
display(df_train.head())



## === cell 5
df_train_overview = (
    df_train.groupby(["Case", "Day"], sort=False)
    .size()
    .reset_index(name="Slices")
    .astype({"Case": np.int32, "Day": np.int32, "Slices": np.int32})
)
display(df_train_overview.head())



## === cell 6
from PIL import Image

_SCAN_LIST_CACHE = {}  # img_dir -> sorted list of png paths


def _get_scan_list(img_dir: str):
    imgs = _SCAN_LIST_CACHE.get(img_dir)
    if imgs is None:
        imgs = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        _SCAN_LIST_CACHE[img_dir] = imgs
    return imgs


def load_image_volume(img_dir, quant=0.01):
    imgs = _get_scan_list(img_dir)
    if len(imgs) == 0:
        return np.zeros((0, 0, 0), dtype=np.uint8)

    with Image.open(imgs[0]) as im0:
        im0 = im0.convert("L")
        w, h = im0.size
    z = len(imgs)

    vol = np.empty((z, h, w), dtype=np.uint16)

    for i, p in enumerate(imgs):
        with Image.open(p) as im:
            vol[i] = np.asarray(im.convert("L"), dtype=np.uint16)

    if quant:
        q_low, q_high = np.percentile(vol, [quant * 100, (1 - quant) * 100])
        vol = np.clip(vol, q_low, q_high)

    v_min = int(vol.min())
    v_max = int(vol.max())
    if v_max > v_min:
        vol_f = (vol.astype(np.float32) - float(v_min)) / float(v_max - v_min)
    else:
        vol_f = vol.astype(np.float32) * 0.0
    vol_u8 = (vol_f * 255.0).astype(np.uint8)
    return vol_u8




## === cell 7
import nibabel as nib



## === cell 8
from concurrent.futures import ThreadPoolExecutor, as_completed

df_train_overview["vol_path"] = ""
VOL_CACHE = {}  # key: vol_path (str) -> np.ndarray (Z,H,W) uint8

cases = df_train_overview["Case"].to_numpy(dtype=np.int32, copy=False)
days = df_train_overview["Day"].to_numpy(dtype=np.int32, copy=False)

image_folders = [
    os.path.join(
        DATASET_FOLDER,
        folder,
        f"case{int(c)}",
        f"case{int(c)}_day{int(d)}",
        "scans",
    )
    for c, d in zip(cases, days)
]
vol_paths = [f"./{int(c)}_{int(d)}_vol.nii.gz" for c, d in zip(cases, days)]
df_train_overview["vol_path"] = vol_paths

t0 = time.time()
max_workers = min(8, (os.cpu_count() or 4))  # safe cap for Kaggle CPU
print("Loading volumes with threads:", max_workers, "num volumes:", len(image_folders))

with ThreadPoolExecutor(max_workers=max_workers) as ex:
    futs = {
        ex.submit(load_image_volume, img_dir): vp
        for img_dir, vp in zip(image_folders, vol_paths)
    }
    for j, fut in enumerate(as_completed(futs), 1):
        vp = futs[fut]
        vol = fut.result()
        VOL_CACHE[vp] = vol
        if j <= 3 or j == len(vol_paths) or (j % 10 == 0):
            print(
                f"Loaded {j}/{len(vol_paths)} vol_path={vp} shape={tuple(vol.shape)} elapsed={time.time()-t0:.1f}s"
            )

gc.collect()



## === cell 9
df_train_overview



## === cell 10
test_data = []
for i in range(len(df_train_overview)):
    CASE = int(df_train_overview["Case"].iat[i])
    DAY = int(df_train_overview["Day"].iat[i])
    path = {"image": f"./{CASE}_{DAY}_vol.nii.gz"}
    test_data.append(path)



## === cell 11
data1 = {
    "description": "UWM",
    "labels": {
        "0": "background",
        "1": "large_bowel",
        "2": "small_bowel",
        "3": "stomach",
    },
    "test": test_data,
}

json_string = json.dumps(data1)
print(json_string)



## === cell 12
with open("json_data.json", "w") as outfile:
    json.dump(data1, outfile)



## === cell 13
import torch
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


class SimpleTestDataset(Dataset):
    def __init__(self, items, transform=None):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        x = dict(self.items[idx])
        if self.transform is not None:
            x = self.transform(x)
        return x


class LoadCachedVolumeAsTensor:
    def __init__(self, cache, keys=("image",)):
        self.cache = cache
        self.keys = keys

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            path = d[k]
            arr_u8 = self.cache.get(path, None)
            if arr_u8 is None:
                nii = nib.load(path)
                arr = nii.get_fdata().astype(np.float32)
            else:
                arr = arr_u8.astype(np.float32, copy=False)
            meta = {"filename_or_obj": path, "spatial_shape": arr.shape}
            d[k] = arr  # (Z,H,W) float32
            d[f"{k}_meta_dict"] = meta
        return d


class AddChannelFirst:
    def __init__(self, keys=("image",)):
        self.keys = keys

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            arr = d[k]  # (Z,H,W)
            d[k] = np.expand_dims(arr, 0)  # (C,Z,H,W) with C=1
        return d


class ToTensor:
    def __init__(self, keys=("image",)):
        self.keys = keys

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            d[k] = torch.from_numpy(d[k])
        return d


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


def sliding_window_inference(inputs, roi_size, sw_batch_size, predictor, overlap=0.8):
    return predictor(inputs)


class UNETRStub(torch.nn.Module):
    """
    Deterministic heuristic model (no-training).
    """

    def __init__(self, out_channels=3, thr=0.15, smooth_k=3):
        super().__init__()
        self.out_channels = out_channels
        self.thr = float(thr)
        self.smooth_k = int(smooth_k)

    def forward(self, x):
        b, _, z, h, w = x.shape
        im = x[:, 0:1]  # (B,1,Z,H,W)

        fg = torch.sigmoid((im - self.thr) * 2.0)  # (B,1,Z,H,W)

        yy = torch.linspace(0, 1, h, device=x.device).view(1, 1, 1, h, 1)
        xxg = torch.linspace(0, 1, w, device=x.device).view(1, 1, 1, 1, w)

        bottom = torch.clamp((yy - 0.45) / 0.55, 0.0, 1.0)  # towards bottom
        center = 1.0 - torch.clamp((xxg - 0.5).abs() / 0.5, 0.0, 1.0)  # towards middle

        lb_logit = 1.8 * fg + 0.8 * bottom + 0.3 * center - 1.2
        sb_logit = 1.7 * fg + 0.9 * bottom - 0.4 * center - 1.1
        st_logit = 1.9 * fg - 1.1 * bottom + 0.4 * center - 1.0

        out = torch.cat([lb_logit, sb_logit, st_logit], dim=1)  # (B,3,Z,H,W)

        if self.smooth_k >= 3 and self.smooth_k % 2 == 1:
            pad = self.smooth_k // 2
            out = torch.nn.functional.avg_pool3d(
                out, kernel_size=self.smooth_k, stride=1, padding=pad
            )

        return out.to(dtype=torch.float32)


print("Using UNETRStub (deterministic heuristic, MONAI unavailable).")



## === cell 14
sz = (80, 144, 192)



## === cell 15
test_transforms = Compose(
    [
        LoadCachedVolumeAsTensor(cache=VOL_CACHE, keys=("image",)),
        AddChannelFirst(keys=("image",)),
        ToTensor(keys=("image",)),
    ]
)



## === cell 16
model = UNETRStub(out_channels=3, thr=0.15, smooth_k=3).to(device)



## === cell 17
ckpt_path = "/kaggle/input/unetr-raw-size/best_metric_model.pth"
if os.path.exists(ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu")
    try:
        model.load_state_dict(state, strict=False)
        print("Loaded checkpoint:", ckpt_path)
    except Exception as e:
        print(
            "Checkpoint exists but couldn't be loaded into stub model; continuing. Error:",
            repr(e),
        )
    del state
    gc.collect()
else:
    print("Checkpoint not found; continuing with stub model:", ckpt_path)




## === cell 18
def rle_decode(mask_rle, shape):
    """
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


def rle_encode(img):
    """
    Kaggle GI-Tract RLE is defined in column-major order (top-to-bottom, then left-to-right).
    Ensure correct ordering by encoding the transposed mask (H,W)->(W,H) before flatten.
    """
    if img.dtype != np.uint8:
        img = img.astype(np.uint8, copy=False)

    pixels = img.T.reshape(-1)
    pixels = np.concatenate(
        (np.array([0], dtype=pixels.dtype), pixels, np.array([0], dtype=pixels.dtype))
    )
    runs = np.flatnonzero(pixels[1:] != pixels[:-1]) + 1
    runs[1::2] -= runs[::2]
    if runs.size == 0:
        return ""
    return " ".join(map(str, runs.tolist()))




## === cell 19
CLASS_TO_CH = {"large_bowel": 0, "small_bowel": 1, "stomach": 2}


def _binary_closing2d_torch(
    mask_u8: np.ndarray, k: int = 5, iters: int = 1
) -> np.ndarray:
    if mask_u8.size == 0:
        return mask_u8
    if k < 3 or k % 2 == 0:
        return mask_u8
    iters = max(1, int(iters))
    pad = k // 2
    t = torch.from_numpy((mask_u8 > 0).astype(np.float32, copy=False)).view(
        1, 1, *mask_u8.shape
    )
    for _ in range(iters):
        t = torch.nn.functional.max_pool2d(t, kernel_size=k, stride=1, padding=pad)
        t = -torch.nn.functional.max_pool2d(-t, kernel_size=k, stride=1, padding=pad)
    out = (t[0, 0] > 0.5).to(torch.uint8).numpy()
    return out


def _keep_largest_component2d(mask_u8: np.ndarray) -> np.ndarray:
    if mask_u8.size == 0:
        return mask_u8
    m = (mask_u8 > 0).astype(np.uint8, copy=False)
    if m.max() == 0:
        return m

    H, W = m.shape
    labels = np.zeros((H, W), dtype=np.int32)

    parent = np.zeros(H * W + 1, dtype=np.int32)
    rank = np.zeros(H * W + 1, dtype=np.uint8)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int):
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    next_label = 0

    for i in range(H):
        row = m[i]
        lab_row = labels[i]
        for j in range(W):
            if row[j] == 0:
                continue
            neigh = []
            if i > 0:
                if labels[i - 1, j] > 0:
                    neigh.append(labels[i - 1, j])
                if j > 0 and labels[i - 1, j - 1] > 0:
                    neigh.append(labels[i - 1, j - 1])
                if j + 1 < W and labels[i - 1, j + 1] > 0:
                    neigh.append(labels[i - 1, j + 1])
            if j > 0 and lab_row[j - 1] > 0:
                neigh.append(lab_row[j - 1])

            if not neigh:
                next_label += 1
                lab = next_label
                parent[lab] = lab
                lab_row[j] = lab
            else:
                lab = int(min(neigh))
                lab_row[j] = lab
                for nb in neigh:
                    if nb != lab:
                        union(lab, int(nb))

    flat = labels.reshape(-1)
    nz = flat > 0
    flat_nz = flat[nz]
    roots = np.empty_like(flat_nz)
    for idx in range(flat_nz.size):
        roots[idx] = find(int(flat_nz[idx]))

    uniq_roots, inv = np.unique(roots, return_inverse=True)
    counts = np.bincount(inv)
    best_root = uniq_roots[int(np.argmax(counts))]

    out = np.zeros((H, W), dtype=np.uint8)
    coords = np.flatnonzero(flat > 0)
    for p in coords:
        if find(int(flat[p])) == int(best_root):
            flat[p] = -1  # mark
    out.reshape(-1)[flat.reshape(-1) == -1] = 1
    return out


def segm_rle(segm, df_vol):
    df_vol_local = df_vol.loc[:, ["id", "class", "Slice"]].copy()
    df_vol_local = df_vol_local.replace(np.nan, "")

    slice_num = pd.to_numeric(df_vol_local["Slice"], errors="coerce")
    df_sorted = (
        df_vol_local.assign(_slice_num=slice_num)
        .sort_values(["_slice_num", "Slice"], kind="mergesort")
        .reset_index(drop=False)  # keep original index in column "index"
    )
    orig_index = df_sorted["index"].to_numpy()

    segm = segm.astype(np.float32, copy=False)  # (3,Z,H,W)
    z_dim = int(segm.shape[1])
    n_rows = int(len(df_sorted))
    use_positional = n_rows == z_dim

    thr = np.array([0.05, 0.05, 0.05], dtype=np.float32)
    any_fg = np.any(segm > thr[:, None, None, None], axis=0)  # (Z,H,W)
    cls = np.argmax(segm, axis=0).astype(np.int16)  # (Z,H,W)
    win_logit = np.max(segm, axis=0)
    conf_gate = any_fg & (win_logit > 0.02)

    classes_sorted = df_sorted["class"].astype(str).to_numpy()
    slice_num_sorted = df_sorted["_slice_num"].to_numpy()
    slice_str_sorted = df_sorted["Slice"].astype(str).to_numpy()

    if use_positional:
        z_idx = np.arange(n_rows, dtype=np.int32)
    else:
        z_idx = slice_num_sorted.astype(np.float64) - 1.0
        bad = ~np.isfinite(z_idx)
        if bad.any():
            bi = np.flatnonzero(bad)
            tmp = np.empty(bi.size, dtype=np.int32)
            for j, pos in enumerate(bi):
                try:
                    tmp[j] = int(slice_str_sorted[pos]) - 1
                except Exception:
                    tmp[j] = int(pos)
            z_idx[bad] = tmp.astype(np.float64)
        z_idx = np.clip(z_idx.astype(np.int32), 0, max(0, z_dim - 1))

    preds_sorted = np.empty(n_rows, dtype=object)
    preds_sorted[:] = ""

    unique_z = np.unique(z_idx)
    rle_by_z = {int(z): ["", "", ""] for z in unique_z}

    for z in unique_z:
        zz = int(z)
        gate_z = conf_gate[zz]
        cls_z = cls[zz]
        for ch in (0, 1, 2):
            mask = ((cls_z == ch) & gate_z).astype(np.uint8, copy=False)
            if mask.any():
                mask = _binary_closing2d_torch(mask, k=5, iters=1)
                mask = _keep_largest_component2d(mask)
                rle_by_z[zz][ch] = rle_encode(mask) if mask.any() else ""
            else:
                rle_by_z[zz][ch] = ""

    for pos in range(n_rows):
        ch = CLASS_TO_CH.get(classes_sorted[pos], None)
        if ch is None:
            preds_sorted[pos] = ""
        else:
            preds_sorted[pos] = rle_by_z[int(z_idx[pos])][ch]

    out_df = df_vol.loc[:, ["id", "class"]].copy()
    out_df["predicted"] = ""

    orig_index_int = orig_index.astype(np.int64, copy=False)
    order = np.argsort(orig_index_int, kind="mergesort")
    inv_pos = np.empty_like(order)
    inv_pos[order] = np.arange(n_rows, dtype=np.int64)
    out_df["predicted"] = preds_sorted[inv_pos]

    return out_df




## === cell 20
with open("./json_data.json", "r") as f:
    datasets = json.load(f)

datalist = datasets.get("test", [])
test_ds = SimpleTestDataset(datalist, transform=test_transforms)
print("Num volumes:", len(test_ds))




## === cell 21
def _collate(batch):
    return batch[0]


test_loader = DataLoader(
    test_ds,
    batch_size=1,
    shuffle=False,
    num_workers=min(2, (os.cpu_count() or 2)),  # small, avoids overhead
    pin_memory=torch.cuda.is_available(),
    collate_fn=_collate,
    persistent_workers=False,
)

df_by_case_day = {k: g for k, g in df_train.groupby(["Case", "Day"], sort=False)}
cd_from_path = {
    vp: (int(c), int(d))
    for vp, c, d in zip(
        df_train_overview["vol_path"].to_numpy(),
        df_train_overview["Case"].to_numpy(),
        df_train_overview["Day"].to_numpy(),
    )
}

model.eval()

pred_parts = []
t0 = time.time()

for i, x_data in enumerate(test_loader):
    vol_path = x_data["image_meta_dict"]["filename_or_obj"]
    CASE, DAY = cd_from_path[vol_path]

    img = x_data["image"]  # (1,Z,H,W) torch float32 on CPU
    mean = img.mean()
    std = img.std(unbiased=False)
    img = (img - mean) / (std + 1e-6)

    inputs = img.unsqueeze(0).to(device, non_blocking=True)  # (B=1,C=1,Z,H,W)

    if i < 3 or (i % 10 == 0) or (i == len(df_train_overview) - 1):
        print(
            "Infer",
            CASE,
            DAY,
            "inputs:",
            tuple(inputs.shape),
            "elapsed:",
            f"{time.time()-t0:.1f}s",
        )
    with torch.no_grad():
        output = sliding_window_inference(inputs, sz, 1, model, overlap=0.8)
        output = output.detach().cpu()

    segm = np.squeeze(output.numpy(), axis=0)  # (3,Z,H,W)
    df_ = df_by_case_day[(CASE, DAY)]
    pred_parts.append(segm_rle(segm, df_))

    del inputs, output, segm, df_, x_data

pred_df = (
    pd.concat(pred_parts, axis=0, ignore_index=True)
    if len(pred_parts)
    else pd.DataFrame(columns=["id", "class", "predicted"])
)
display(pred_df.head())
print("pred_df shape:", pred_df.shape, "total elapsed:", f"{time.time()-t0:.1f}s")
gc.collect()



## === cell 22
df_train = df_train.copy()
display(df_train.head())



## === cell 23
display(pred_df.head())



## === cell 24
sub_df = pd.read_csv(os.path.join(DATASET_FOLDER, "sample_submission.csv"))



## === cell 25
sub_df = sub_df.drop(columns=["predicted"], errors="ignore")
sub_df = sub_df.merge(pred_df, on=["id", "class"], how="left")
sub_df["predicted"] = sub_df["predicted"].fillna("")
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)



## === cell 26
display(sub_df.head())
print("Non-empty predictions:", (sub_df["predicted"].astype(str).str.len() > 0).sum())
print("submission.csv exists:", os.path.exists("submission.csv"))
print(
    "submission.csv size (bytes):",
    os.path.getsize("submission.csv") if os.path.exists("submission.csv") else -1,
)
