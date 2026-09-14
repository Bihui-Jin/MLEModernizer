# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.11

# 2. Installed packages

geopandas==0.14.4
ipywidgets==8.1.5
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (135 lines)
            sample_submission.csv (2 lines)
            sample_submission.csv.zip (215 Bytes)
            test.zip (3.0 GB)
            train.zip (15.5 GB)
            test/
                a/
                    mask.png (40.7 kB)
                    surface_volume/
                        06.tif (79.8 MB)
                        01.tif (79.8 MB)
                        ... and 63 other files
                test/
            train/
                1/
                    inklabels.png (92.6 kB)
                    inklabels_rle.csv (2 lines)
                    ... and 2 other files
                    surface_volume/
                        42.tif (103.6 MB)
                        45.tif (103.6 MB)
                        ... and 63 other files
                2/
                    inklabels.png (294.3 kB)
                    inklabels_rle.csv (2 lines)
                    ... and 2 other files
                    surface_volume/
                        10.tif (281.9 MB)
                        17.tif (281.9 MB)
                        ... and 63 other files
                train/
            vesuvius-challenge-ink-detection/
                description.md (135 lines)
                sample_submission.csv (2 lines)
                ... and 3 other files
                test/
                    a/
                        mask.png (40.7 kB)
                        surface_volume/
                            ... (max depth reached)
                    test/
                train/
                    1/
                        inklabels.png (92.6 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    2/
                        inklabels.png (294.3 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    train/
                vesuvius-challenge-ink-detection/
        input/
            description.md (135 lines)
            sample_submission.csv (2 lines)
            sample_submission.csv.zip (215 Bytes)
            test.zip (3.0 GB)
            train.zip (15.5 GB)
            test/
                a/
                    mask.png (40.7 kB)
                    surface_volume/
                        06.tif (79.8 MB)
                        01.tif (79.8 MB)
                        ... and 63 other files
                test/
                    a/
                        mask.png (40.7 kB)
                        surface_volume/
                            ... (max depth reached)
                    test/
            train/
                1/
                    inklabels.png (92.6 kB)
                    inklabels_rle.csv (2 lines)
                    ... and 2 other files
                    surface_volume/
                        42.tif (103.6 MB)
                        45.tif (103.6 MB)
                        ... and 63 other files
                2/
                    inklabels.png (294.3 kB)
                    inklabels_rle.csv (2 lines)
                    ... and 2 other files
                    surface_volume/
                        10.tif (281.9 MB)
                        17.tif (281.9 MB)
                        ... and 63 other files
                train/
                    1/
                        inklabels.png (92.6 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    2/
                        inklabels.png (294.3 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    train/
            vesuvius-challenge-ink-detection/
                description.md (135 lines)
                sample_submission.csv (2 lines)
                ... and 3 other files
                test/
                    a/
                        mask.png (40.7 kB)
                        surface_volume/
                            ... (max depth reached)
                    test/
                train/
                    1/
                        inklabels.png (92.6 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    2/
                        inklabels.png (294.3 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    train/
                vesuvius-challenge-ink-detection/
        working/
            vesuvius-challenge-ink-detection/
                description.md (135 lines)
                sample_submission.csv (2 lines)
                ... and 3 other files
                test/
                    a/
                        mask.png (40.7 kB)
                        surface_volume/
                            ... (max depth reached)
                    test/
                train/
                    1/
                        inklabels.png (92.6 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    2/
                        inklabels.png (294.3 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    train/
                vesuvius-challenge-ink-detection/
```

-> data/sample_submission.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> data/train/1/inklabels_rle.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> data/train/2/inklabels_rle.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> data/vesuvius-challenge-ink-detection/sample_submission.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> data/vesuvius-challenge-ink-detection/train/1/inklabels_rle.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> data/vesuvius-challenge-ink-detection/train/2/inklabels_rle.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> input/sample_submission.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd
import PIL.Image as Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from tqdm import tqdm

_PREFIX_CANDIDATES = [
    "/kaggle/input/vesuvius-challenge-ink-detection/",
    "/kaggle/input/vesuvius-challenge/",
    "/kaggle/data/vesuvius-challenge-ink-detection/",
    "/kaggle/data/vesuvius-challenge/",
]
DATA_ROOT = next(
    (p for p in _PREFIX_CANDIDATES if os.path.exists(p)), _PREFIX_CANDIDATES[0]
)

TRAIN_ROOT = os.path.join(DATA_ROOT, "train")
TEST_ROOT = os.path.join(DATA_ROOT, "test")

BUFFER = 30  # Buffer size in x and y direction
Z_START = 27  # First slice in the z direction to use
Z_DIM = 10  # Number of slices in the z direction
TRAINING_STEPS = 20000
LEARNING_RATE = 0.05
BATCH_SIZE = 24
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = True

TRAIN_FRAGMENT = "1"
PREFIX = os.path.join(TRAIN_ROOT, TRAIN_FRAGMENT) + "/"

rect = (1100, 3500, 700, 950)



## === cell 1
mask = np.array(Image.open(PREFIX + "mask.png").convert("1"))
label = (
    torch.from_numpy(np.array(Image.open(PREFIX + "inklabels.png")))
    .gt(0)
    .float()
    .to(DEVICE)
)



## === cell 2
images = [
    np.array(Image.open(filename), dtype=np.float32) / 65535.0
    for filename in tqdm(
        sorted(glob.glob(PREFIX + "surface_volume/*.tif"))[Z_START : Z_START + Z_DIM],
        desc="Loading train stack",
        leave=False,
    )
]
image_stack = torch.stack(
    [torch.from_numpy(image) for image in images], dim=0
).contiguous()




## === cell 3
class SubvolumeIndexDataset(data.Dataset):
    def __init__(self, n: int):
        self.n = int(n)

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return int(index)


_STACK_DEV_CACHE = {}
_PAD_UNFOLD_CACHE = {}
_FLAT_UNFOLD_CACHE = {}


def _cache_key(t: torch.Tensor):
    return (int(t.data_ptr()), tuple(t.shape), str(t.dtype), str(t.device))


def _get_stack_dev(image_stack_cpu: torch.Tensor) -> torch.Tensor:
    if DEVICE.type != "cuda":
        return image_stack_cpu
    key = _cache_key(image_stack_cpu) + (str(DEVICE),)
    cached = _STACK_DEV_CACHE.get(key)
    if cached is None or cached.device != DEVICE:
        cached = image_stack_cpu.to(DEVICE, non_blocking=True)
        _STACK_DEV_CACHE[key] = cached
    return cached


def _get_unfold_view(stack_cpu: torch.Tensor):
    """
    Returns:
      stack_dev: (Z, H+2B, W+2B) on DEVICE
      unfold_view: (Z, H, W, P, P) view on DEVICE (no copy)
    """
    P = BUFFER * 2 + 1
    stack_dev = _get_stack_dev(stack_cpu)
    key = (_cache_key(stack_cpu), str(DEVICE), BUFFER)
    cached = _PAD_UNFOLD_CACHE.get(key)
    if cached is None:
        padded = torch.nn.functional.pad(
            stack_dev, (BUFFER, BUFFER, BUFFER, BUFFER), mode="constant", value=0.0
        )
        unfold = padded.unfold(1, P, 1).unfold(2, P, 1)  # (Z, H, W, P, P)
        _PAD_UNFOLD_CACHE[key] = (padded, unfold)
        cached = _PAD_UNFOLD_CACHE[key]
    _, unfold = cached
    return stack_dev, unfold


def _get_flat_unfold(stack_cpu: torch.Tensor):
    """
    Performance-critical: create a flattened view (Z, H*W, P*P) so we can extract
    a batch of patches via gather with a single contiguous allocation for the batch.

    Semantics are identical to:
      patches = unfold[:, ys, xs, :, :]  # (Z,B,P,P)
    because linear index i = ys*W + xs selects the same (y,x) position.
    """
    P = BUFFER * 2 + 1
    key = (_cache_key(stack_cpu), str(DEVICE), BUFFER, "flat_unfold")
    cached = _FLAT_UNFOLD_CACHE.get(key)
    if cached is not None:
        return cached  # (flat, H, W, P)
    _, unfold = _get_unfold_view(stack_cpu)  # (Z,H,W,P,P)
    Z, H, W, _, _ = unfold.shape
    flat = unfold.reshape(Z, H * W, P * P)  # view (no copy)
    _FLAT_UNFOLD_CACHE[key] = (flat, H, W, P)
    return flat, H, W, P


def _extract_subvolumes_from_coords(
    flat_unfold: torch.Tensor, H: int, W: int, P: int, coords_yx: torch.Tensor
):
    """
    coords_yx: (B,2) on DEVICE, long
    Returns subvolumes: (B,1,Z,P,P) float on DEVICE
    """
    ys = coords_yx[:, 0]
    xs = coords_yx[:, 1]
    lin = ys * W + xs  # (B,)
    gathered = flat_unfold.gather(
        1, lin.view(1, -1, 1).expand(flat_unfold.shape[0], -1, flat_unfold.shape[2])
    )
    sub = gathered.permute(1, 0, 2).contiguous().view(-1, 1, flat_unfold.shape[0], P, P)
    return sub


def make_collate_unfold(
    stack_cpu: torch.Tensor, label_dev: torch.Tensor, pixels_cpu_t: torch.Tensor
):
    """
    pixels_cpu_t: (N,2) torch.long on CPU, contiguous.
    Uses precomputed unfold view to extract patches.
    """
    flat, H, W, P = _get_flat_unfold(stack_cpu)

    def collate(batch):
        idxs = torch.as_tensor(batch, dtype=torch.long)
        coords = pixels_cpu_t.index_select(0, idxs).to(
            device=DEVICE, non_blocking=True
        )  # (B,2)
        sub = _extract_subvolumes_from_coords(flat, H, W, P, coords)
        ink = label_dev[coords[:, 0], coords[:, 1]].view(-1, 1)
        return sub, ink

    return collate


model = nn.Sequential(
    nn.Conv3d(1, 16, 3, 1, 1),
    nn.MaxPool3d(2, 2),
    nn.Conv3d(16, 32, 3, 1, 1),
    nn.MaxPool3d(2, 2),
    nn.Conv3d(32, 64, 3, 1, 1),
    nn.MaxPool3d(2, 2),
    nn.Flatten(start_dim=1),
    nn.LazyLinear(128),
    nn.ReLU(),
    nn.LazyLinear(1),
    nn.Sigmoid(),
).to(DEVICE)



## === cell 4
print("Generating pixel lists...")
not_border = np.zeros(mask.shape, dtype=bool)
not_border[BUFFER : mask.shape[0] - BUFFER, BUFFER : mask.shape[1] - BUFFER] = True
arr_mask = np.array(mask).astype(bool) & not_border

inside_rect = np.zeros(mask.shape, dtype=bool) & arr_mask
inside_rect[rect[1] : rect[1] + rect[3] + 1, rect[0] : rect[0] + rect[2] + 1] = True

outside_rect = np.ones(mask.shape, dtype=bool) & arr_mask
outside_rect[rect[1] : rect[1] + rect[3] + 1, rect[0] : rect[0] + rect[2] + 1] = False

pixels_inside_rect = np.argwhere(inside_rect)
pixels_outside_rect = np.argwhere(outside_rect)

pixels_outside_t_cpu = torch.from_numpy(
    pixels_outside_rect.astype(np.int64, copy=False)
).contiguous()
pixels_inside_t_cpu = torch.from_numpy(
    pixels_inside_rect.astype(np.int64, copy=False)
).contiguous()

P_train = BUFFER * 2 + 1
stack_dev = _get_stack_dev(image_stack)  # (Z,H,W) on DEVICE (or CPU if no CUDA)
_padded_train = torch.nn.functional.pad(
    stack_dev, (BUFFER, BUFFER, BUFFER, BUFFER), mode="constant", value=0.0
)  # (Z, H+2B, W+2B)

flat_train, H_train, W_train = None, int(stack_dev.shape[1]), int(stack_dev.shape[2])


def _extract_subvolumes_from_coords(
    flat_unfold_unused: torch.Tensor,
    H_unused: int,
    W_unused: int,
    P_unused: int,
    coords_yx: torch.Tensor,
):
    """
    coords_yx: (B,2) on DEVICE, long
    Returns subvolumes: (B,1,Z,P,P) float on DEVICE
    Semantics match the original unfold-based extraction.
    """
    ys = coords_yx[:, 0].to(dtype=torch.long)
    xs = coords_yx[:, 1].to(dtype=torch.long)

    ys0 = ys
    xs0 = xs

    patches = []
    for y, x in zip(ys0.tolist(), xs0.tolist()):
        patch = _padded_train[:, y : y + P_train, x : x + P_train]  # (Z,P,P)
        patches.append(patch)
    sub = torch.stack(patches, dim=0).unsqueeze(1).contiguous()  # (B,1,Z,P,P)
    return sub


pixels_outside_t = pixels_outside_t_cpu.to(
    device=DEVICE, dtype=torch.long, non_blocking=True
)
pixels_inside_t = pixels_inside_t_cpu.to(
    device=DEVICE, dtype=torch.long, non_blocking=True
)

criterion = nn.BCELoss()
optimizer = optim.ASGD(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.5, total_iters=4)

print("Training...")
model.train()

N_out = pixels_outside_t.shape[0]

batch_coords = torch.empty((BATCH_SIZE, 2), device=DEVICE, dtype=torch.long)

for _ in tqdm(range(TRAINING_STEPS), total=TRAINING_STEPS, desc="Train", leave=False):
    idxs = torch.randint(0, N_out, (BATCH_SIZE,), device=DEVICE, dtype=torch.long)
    coords = pixels_outside_t.index_select(0, idxs)  # (B,2)

    subvolumes = _extract_subvolumes_from_coords(
        flat_train, H_train, W_train, P_train, coords
    )
    inklabels = label[coords[:, 0], coords[:, 1]].view(-1, 1)

    optimizer.zero_grad(set_to_none=True)
    outputs = model(subvolumes)
    loss = criterion(outputs, inklabels)
    loss.backward()
    optimizer.step()
    scheduler.step()


## === cell 5
output = torch.zeros_like(label).float()

model.eval()
with torch.no_grad():
    N_in = pixels_inside_t.shape[0]
    idx = 0
    while idx < N_in:
        end = min(idx + BATCH_SIZE, N_in)
        coords = pixels_inside_t[idx:end]
        subvolumes = _extract_subvolumes_from_coords(
            flat_train, H_train, W_train, P_train, coords
        )
        preds = model(subvolumes).view(-1)
        output[coords[:, 0], coords[:, 1]] = preds
        idx = end



## === cell 6
THRESHOLD = 0.4




## === cell 7
def rle_binary_mask(mask2d_bool: np.ndarray) -> str:
    pixels = mask2d_bool.astype(np.uint8).ravel(order="C")
    pixels = np.concatenate(([0], pixels, [0]))
    changes = np.flatnonzero(pixels[1:] != pixels[:-1]) + 1
    if changes.size == 0:
        return ""
    runs = changes.copy()
    runs[1::2] -= runs[::2]
    return " ".join(map(str, runs.tolist()))


_TIF_CACHE = {}


def load_image_stack(fragment_dir: str) -> torch.Tensor:
    sv_dir = os.path.join(fragment_dir, "surface_volume")
    if sv_dir not in _TIF_CACHE:
        _TIF_CACHE[sv_dir] = sorted(glob.glob(os.path.join(sv_dir, "*.tif")))
    files = _TIF_CACHE[sv_dir][Z_START : Z_START + Z_DIM]
    imgs = [np.array(Image.open(fn), dtype=np.float32) / 65535.0 for fn in files]
    stack = torch.stack(
        [torch.from_numpy(im) for im in imgs], dim=0
    ).contiguous()  # CPU
    return stack


def predict_fragment(fragment_id: str) -> str:
    fragment_dir = os.path.join(TEST_ROOT, fragment_id)
    frag_mask = np.array(
        Image.open(os.path.join(fragment_dir, "mask.png")).convert("1")
    ).astype(bool)

    not_border_local = np.zeros(frag_mask.shape, dtype=bool)
    not_border_local[
        BUFFER : frag_mask.shape[0] - BUFFER, BUFFER : frag_mask.shape[1] - BUFFER
    ] = True
    valid_pixels_mask = frag_mask & not_border_local

    pixels = np.argwhere(valid_pixels_mask)
    if pixels.size == 0:
        return ""

    pixels_t_cpu = torch.from_numpy(pixels.astype(np.int64, copy=False)).contiguous()

    stack_cpu = load_image_stack(fragment_dir)
    flat, H, W, P = _get_flat_unfold(stack_cpu)

    pixels_t = pixels_t_cpu.to(device=DEVICE, dtype=torch.long, non_blocking=True)
    prob = torch.zeros(
        (frag_mask.shape[0], frag_mask.shape[1]), device=DEVICE, dtype=torch.float32
    )

    model.eval()
    with torch.no_grad():
        N = pixels_t.shape[0]
        idx = 0
        while idx < N:
            end = min(idx + BATCH_SIZE, N)
            coords = pixels_t[idx:end]
            subvolumes = _extract_subvolumes_from_coords(flat, H, W, P, coords)
            preds = model(subvolumes).view(-1)
            prob[coords[:, 0], coords[:, 1]] = preds
            idx = end

    bin_mask = (prob > THRESHOLD).detach().cpu().numpy().astype(bool)
    bin_mask &= frag_mask
    return rle_binary_mask(bin_mask)


sample_path = os.path.join(DATA_ROOT, "sample_submission.csv")
sample = pd.read_csv(sample_path)

preds = []
for fid in tqdm(sample["Id"].tolist(), desc="Predicting test fragments"):
    preds.append(predict_fragment(str(fid)))

sub = pd.DataFrame({"Id": sample["Id"], "Predicted": preds})
sub.to_csv("submission.csv", index=False)
print(sub.head())
print(
    "Wrote submission.csv with", len(sub), "rows to", os.path.abspath("submission.csv")
)

## --- ERROR in cell 7, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mOutOfMemoryError[0m                          Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/4262195155.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     74[0m [0mpreds[0m [0;34m=[0m [0;34m[[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m     75[0m [0;32mfor[0m [0mfid[0m [0;32min[0m [0mtqdm[0m[0;34m([0m[0msample[0m[0;34m[[0m[0;34m"Id"[0m[0;34m][0m[0;34m.[0m[0mtolist[0m[0;34m([0m[0;34m)[0m[0;34m,[0m [0mdesc[0m[0;34m=[0m[0;34m"Predicting test fragments"[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 76[0;31m     [0mpreds[0m[0;34m.[0m[0mappend[0m[0;34m([0m[0mpredict_fragment[0m[0;34m([0m[0mstr[0m[0;34m([0m[0mfid[0m[0;34m)[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     77[0m [0;34m[0m[0m
[1;32m     78[0m [0msub[0m [0;34m=[0m [0mpd[0m[0;34m.[0m[0mDataFrame[0m[0;34m([0m[0;34m{[0m[0;34m"Id"[0m[0;34m:[0m [0msample[0m[0;34m[[0m[0;34m"Id"[0m[0;34m][0m[0;34m,[0m [0;34m"Predicted"[0m[0;34m:[0m [0mpreds[0m[0;34m}[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/4262195155.py[0m in [0;36mpredict_fragment[0;34m(fragment_id)[0m
[1;32m     45[0m     [0mstack_cpu[0m [0;34m=[0m [0mload_image_stack[0m[0;34m([0m[0mfragment_dir[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     46[0m     [0;31m# Runtime fix: use the same fast gather-based extraction for inference.[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 47[0;31m     [0mflat[0m[0;34m,[0m [0mH[0m[0;34m,[0m [0mW[0m[0;34m,[0m [0mP[0m [0;34m=[0m [0m_get_flat_unfold[0m[0;34m([0m[0mstack_cpu[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     48[0m [0;34m[0m[0m
[1;32m     49[0m     [0mpixels_t[0m [0;34m=[0m [0mpixels_t_cpu[0m[0;34m.[0m[0mto[0m[0;34m([0m[0mdevice[0m[0;34m=[0m[0mDEVICE[0m[0;34m,[0m [0mdtype[0m[0;34m=[0m[0mtorch[0m[0;34m.[0m[0mlong[0m[0;34m,[0m [0mnon_blocking[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/1081115243.py[0m in [0;36m_get_flat_unfold[0;34m(stack_cpu)[0m
[1;32m     70[0m     [0m_[0m[0;34m,[0m [0munfold[0m [0;34m=[0m [0m_get_unfold_view[0m[0;34m([0m[0mstack_cpu[0m[0;34m)[0m  [0;31m# (Z,H,W,P,P)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     71[0m     [0mZ[0m[0;34m,[0m [0mH[0m[0;34m,[0m [0mW[0m[0;34m,[0m [0m_[0m[0;34m,[0m [0m_[0m [0;34m=[0m [0munfold[0m[0;34m.[0m[0mshape[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 72[0;31m     [0mflat[0m [0;34m=[0m [0munfold[0m[0;34m.[0m[0mreshape[0m[0;34m([0m[0mZ[0m[0;34m,[0m [0mH[0m [0;34m*[0m [0mW[0m[0;34m,[0m [0mP[0m [0;34m*[0m [0mP[0m[0;34m)[0m  [0;31m# view (no copy)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     73[0m     [0m_FLAT_UNFOLD_CACHE[0m[0;34m[[0m[0mkey[0m[0;34m][0m [0;34m=[0m [0;34m([0m[0mflat[0m[0;34m,[0m [0mH[0m[0;34m,[0m [0mW[0m[0;34m,[0m [0mP[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     74[0m     [0;32mreturn[0m [0mflat[0m[0;34m,[0m [0mH[0m[0;34m,[0m [0mW[0m[0;34m,[0m [0mP[0m[0;34m[0m[0;34m[0m[0m

[0;31mOutOfMemoryError[0m: CUDA out of memory. Tried to allocate 5534.17 GiB. GPU 0 has a total capacity of 47.53 GiB of which 39.46 GiB is free. Process 2783073 has 8.06 GiB memory in use. Of the allocated memory 7.74 GiB is allocated by PyTorch, and 6.64 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
