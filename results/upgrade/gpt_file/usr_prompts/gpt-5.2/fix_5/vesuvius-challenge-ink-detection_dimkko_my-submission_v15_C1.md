# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Overview
Detect the presence of ink from 3d x-ray scans of detached fragments of ancient papyrus scrolls.

## Metric
We evaluate how well your output image matches our reference image using a modified version of the [Sørensen--Dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient), where instead of using the F1 score, we are using the F0.5 score. The F0.5 score is given by:

$$
\frac{\left(1+\beta^2\right) p r}{\beta^2 p+r} \text { where } p=\frac{t p}{t p+f p}, r=\frac{t p}{t p+f n}, \beta=0.5
$$

The F0.5 score weights precision higher than recall, which improves the ability to form coherent characters out of detected ink areas.

In order to reduce the submission file size, our metric uses run-length encoding on the pixel values. Instead of submitting an exhaustive list of indices for your segmentation, you will submit pairs of values that contain a start position and a run length. E.g. '1 3' implies starting at pixel 1 and running a total of 3 pixels (1,2,3).

Note that, at the time of encoding, the output should be binary, with 0 indicating "no ink" and 1 indicating "ink".

The competition format requires a space delimited list of pairs. For example, '1 3 10 5' implies pixels 1,2,3,10,11,12,13,14 are to be included in the mask. The metric checks that the pairs are sorted, positive, and the decoded pixel values are not duplicated. The pixels are numbered from left to right, then top to bottom: 1 is pixel (1,1), 2 is pixel (1,2), etc.

Your output should be a single file, **submission.csv**, with this run-length encoded information. This should have a header with two columns, `Id` and `Predicted`, and with one row for every directory under **test/**. For example:

```
Id,Predicted
a,1 1 5 1 etc.
b,10 20 etc.
```

For a real-world example of what these files look like, see `inklabels_rce.csv` in the data directories, which have been generated with [this script](https://gist.github.com/janpaul123/ca3477c1db6de4346affca37e0e3d5b0).


## Data
- **[train/test]/[fragment_id]/surface_volume/[image_id].tif** slices from the 3d x-ray [surface volume](https://scrollprize.org/tutorial1#3-surface-volumes). Each file contains a greyscale slice in the z-direction. Each fragment contains 65 slices. Combined this image stack gives us `width * height * 65` number of voxels per fragment. You can expect two fragments in the hidden test set, which together are roughly the same size as a single training fragment. The sample slices available to download in the test folders are simply copied from training fragment one, but when you submit your notebook they will be substituted with the real test data.
- **[train/test]/[fragment_id]/mask.png** --- a binary mask of which pixels contain data.
- **train/[fragment_id]/inklabels.png** --- a binary mask of the ink vs no-ink labels.
- **train/[fragment_id]/inklabels_rle.csv** --- a run-length-encoded version of the labels, generated using [this script](https://gist.github.com/janpaul123/ca3477c1db6de4346affca37e0e3d5b0). This is the same format as you should make your submission in.
- **train/[fragment_id]/ir.png** --- the infrared photo on which the binary mask is based.
- **sample_submission.csv**, an example of a submission file in the correct format. You need to output the following file in the home directory: **submission.csv**.

# 2. Python version

3.11

# 3. Installed packages

catboost==1.2.8
geopandas==0.14.4
ipywidgets==8.1.5
joblib==1.5.2
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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
sklearn-pandas==2.2.0
tifffile==2025.6.11
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

# 5. Target score

0.069465470841898

# 6. Current score

0.14407

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.14407) has done: 'Most of the timeout comes from the extremely large training loop (1000 epochs over ~1200 patches with a heavy U-Net) and from repeatedly rebuilding large CPU tensors during inference tiling. I keep the exact same model, loss, optimizer, dataset sampling, and inference semantics, but remove avoidable overhead by (1) compiling the model with `torch.compile` (same math, faster execution), (2) enabling cuDNN autotuning (doesn’t change results in eval and preserves overall training semantics; determinism remains controlled by your seed settings), (3) preallocating/pinning inference buffers and using non-blocking H2D transfers efficiently, and (4) eliminating redundant conversions/copies in stack loading and tiling. These are all constant-factor speedups that don’t change the algorithm or the produced masks (beyond negligible FP differences).'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from concurrent.futures import ThreadPoolExecutor
import os

output_path = "./output/"
if not os.path.exists(output_path):
    os.makedirs(output_path)



## === cell 1
import cv2
import random
import torch
import torch.nn as nn
import torch.optim as optim
import glob
from PIL import Image
import torch.utils.data as data
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import gc
import tifffile

gc.collect()



## === cell 2
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed



## === cell 3
device = "cuda" if torch.cuda.is_available() else "cpu"
device




## === cell 4
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




## === cell 5
learning_rate = 0.0001
batch_size = 50
cube_size = 32
epochs = 1000
seed = 42
stack_count = 30
seed_everything(seed)
kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
block_size = cube_size

DATA_ROOT = "/kaggle/input/vesuvius-challenge-ink-detection"
TRAIN_ROOT = os.path.join(DATA_ROOT, "train")
TEST_ROOT = os.path.join(DATA_ROOT, "test")

assert os.path.isdir(TRAIN_ROOT), f"TRAIN_ROOT not found: {TRAIN_ROOT}"
assert os.path.isdir(TEST_ROOT), f"TEST_ROOT not found: {TEST_ROOT}"

_USE_COMPILE = True
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True




## === cell 6
class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttentionBlock, self).__init__()

        self.query_conv = nn.Conv2d(
            in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1
        )
        self.key_conv = nn.Conv2d(
            in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1
        )
        self.value_conv = nn.Conv2d(
            in_channels=in_channels, out_channels=in_channels, kernel_size=1
        )

        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, H, W = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, H * W)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(batch_size, -1, H * W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        out = self.gamma * out + x
        return out




## === cell 7
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=kernel_size, padding=padding
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out




## === cell 8
class R2U_Block(nn.Module):
    def __init__(self, in_channels, hidden_channels, t=2):
        super(R2U_Block, self).__init__()
        self.t = t
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size=3, padding=1
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size=3, padding=1
        )
        self.conv4 = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size=3, padding=1
        )
        self.rnn = nn.LSTM(
            hidden_channels, hidden_channels, num_layers=2, batch_first=True
        )

    def forward(self, x):
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        inp = h
        for _ in range(self.t):
            h = self.relu(self.conv3(h) + inp)
            inp = h
        h, _ = self.rnn(h)
        h = self.relu(self.conv4(h))
        h = x + h
        return h




## === cell 9
class UNET(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, stride=2)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, stride=2)

        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(1024)
        self.relu5 = nn.ReLU()

        self.upconv6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv6 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.relu6 = nn.ReLU()

        self.upconv7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.relu7 = nn.ReLU()

        self.upconv8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(128)
        self.relu8 = nn.ReLU()

        self.upconv9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(64)
        self.relu9 = nn.ReLU()

        self.conv10 = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x1 = x.clone()

        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x2 = x.clone()

        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x3 = x.clone()

        x = self.pool3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x4 = x.clone()

        x = self.pool4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)

        x = self.upconv6(x)
        x = torch.cat([x4, x], dim=1)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)

        x = self.upconv7(x)
        x = torch.cat([x3, x], dim=1)
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu7(x)

        x = self.upconv8(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu8(x)

        x = self.upconv9(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv9(x)
        x = self.bn9(x)
        x = self.relu9(x)

        x = self.conv10(x)
        x = self.sigmoid(x)
        return x




## === cell 10
model_CNN = UNET(stack_count).to(device)
optimizer = optim.Adam(model_CNN.parameters(), lr=learning_rate)
criterion = nn.BCELoss()

if _USE_COMPILE:
    try:
        model_CNN = torch.compile(model_CNN, mode="reduce-overhead")
    except Exception as e:
        print("torch.compile unavailable/failed, continuing eager. Error:", repr(e))

gc.collect()



## === cell 11
_STACK_CACHE_DIR = os.path.join(output_path, "stack_cache_sc30_cs32")
os.makedirs(_STACK_CACHE_DIR, exist_ok=True)


def _read_tif_gray(path):
    arr = tifffile.imread(path)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def _first_n_tif_paths(surface_dir, n):
    paths = []
    for i in range(1, n + 1):
        p = os.path.join(surface_dir, f"{i:02d}.tif")
        if os.path.exists(p):
            paths.append(p)
        else:
            files = sorted(glob.glob(os.path.join(surface_dir, "*.tif")))
            return files[:n]
    return paths


def _cache_key_for_surface(surface_dir, stack_count):
    safe = surface_dir.strip("/").replace("/", "__")
    return os.path.join(_STACK_CACHE_DIR, f"{safe}__sc{stack_count}.npy")


def _load_stack(surface_dir, stack_count):
    cache_path = _cache_key_for_surface(surface_dir, stack_count)
    if os.path.exists(cache_path):
        return np.load(cache_path, mmap_mode="r")  # (S,H,W) float32

    files = _first_n_tif_paths(surface_dir, stack_count)
    if len(files) == 0:
        raise FileNotFoundError(f"No tif files in {surface_dir}")

    first = _read_tif_gray(files[0])
    h, w = first.shape[:2]
    stack = np.empty((len(files), h, w), dtype=np.float32)

    for idx, f in enumerate(files):
        sl = _read_tif_gray(f).astype(np.float32, copy=False)
        sl = cv2.filter2D(sl, -1, kernel)

        sl_min = float(sl.min())
        sl_max = float(sl.max())
        if sl_max > sl_min:
            sl8 = ((sl - sl_min) * (255.0 / (sl_max - sl_min))).astype(np.uint8)
        else:
            sl8 = np.zeros(sl.shape, dtype=np.uint8)

        sl8 = cv2.equalizeHist(sl8)
        sl8 = cv2.medianBlur(sl8, 3)

        stack[idx] = sl8.astype(np.float32) * (1.0 / 255.0)

    np.save(cache_path, stack)
    return stack  # (S,H,W) float32 in [0,1]


def process_volume_data(path, stack_count, cube_size, model_CNN, device, batch_size):
    stack_np = _load_stack(path, stack_count=stack_count)  # (S,H,W) float32

    stack_t = torch.from_numpy(stack_np)  # shares memory if not memmap; ok either way

    H, W = int(stack_t.shape[1]), int(stack_t.shape[2])
    pad_h = (cube_size - (H % cube_size)) % cube_size
    pad_w = (cube_size - (W % cube_size)) % cube_size

    image = F.pad(stack_t, (0, pad_w, 0, pad_h, 0, 0))  # (S,Hp,Wp)
    Hp, Wp = int(image.shape[1]), int(image.shape[2])

    tiles = image.unfold(1, cube_size, cube_size).unfold(2, cube_size, cube_size)
    nH, nW = int(tiles.shape[1]), int(tiles.shape[2])
    tiles = (
        tiles.permute(1, 2, 0, 3, 4)
        .contiguous()
        .view(nH * nW, stack_count, cube_size, cube_size)
    )

    preds = torch.empty((nH * nW, 1, cube_size, cube_size), dtype=torch.float32)
    if torch.cuda.is_available():
        preds = preds.pin_memory()

    model_CNN.eval()
    with torch.no_grad():
        for start in range(0, tiles.shape[0], batch_size):
            batch_cpu = tiles[start : start + batch_size]
            if torch.cuda.is_available():
                batch = batch_cpu.pin_memory().to(device, non_blocking=True)
            else:
                batch = batch_cpu.to(device)
            out = model_CNN(batch)  # (B,1,cs,cs)
            preds[start : start + out.shape[0]].copy_(
                out.detach().to("cpu"), non_blocking=False
            )

    preds = (
        preds.view(nH, nW, 1, cube_size, cube_size).permute(2, 0, 3, 1, 4).contiguous()
    )
    pred_full = preds.view(1, nH * cube_size, nW * cube_size)
    return pred_full[:, :H, :W]  # (1,H,W)


def rle_encode(mask: np.ndarray) -> str:
    pixels = mask.flatten(order="C")
    pixels = np.concatenate([[0], pixels.astype(np.uint8, copy=False), [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


class VesuviusPatchDataset(Dataset):
    def __init__(
        self,
        fragment_ids,
        cube_size=32,
        stack_count=30,
        pos_frac=0.5,
        patches_per_fragment=600,
    ):
        self.fragment_ids = [str(x) for x in fragment_ids]
        self.cube_size = cube_size
        self.stack_count = stack_count
        self.pos_frac = float(pos_frac)
        self.patches_per_fragment = int(patches_per_fragment)

        self.items = []  # list of (fid, y, x)
        self._cache = {}

        for fid in self.fragment_ids:
            frag_dir = os.path.join(TRAIN_ROOT, fid)
            surface_dir = os.path.join(frag_dir, "surface_volume")
            label_path = os.path.join(frag_dir, "inklabels.png")
            mask_path = os.path.join(frag_dir, "mask.png")
            if not (
                os.path.isdir(surface_dir)
                and os.path.isfile(label_path)
                and os.path.isfile(mask_path)
            ):
                continue

            stack = _load_stack(surface_dir, stack_count=self.stack_count)  # (S,H,W)
            lbl = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            msk = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if lbl is None or msk is None:
                continue
            lbl = (lbl > 0).astype(np.uint8)
            msk = (msk > 0).astype(np.uint8)

            self._cache[fid] = (stack, lbl, msk)
            H, W = lbl.shape

            valid = msk.copy()
            valid[:cube_size, :] = 0
            valid[-cube_size:, :] = 0
            valid[:, :cube_size] = 0
            valid[:, -cube_size:] = 0

            pos = np.argwhere((lbl == 1) & (valid == 1))
            neg = np.argwhere((lbl == 0) & (valid == 1))

            n_total = self.patches_per_fragment
            n_pos = int(n_total * self.pos_frac)
            n_neg = n_total - n_pos

            if len(pos) == 0:
                n_pos = 0
                n_neg = n_total

            if len(neg) == 0:
                n_neg = 0
                n_pos = n_total

            if n_pos > 0:
                idx = np.random.choice(
                    len(pos), size=min(n_pos, len(pos)), replace=len(pos) < n_pos
                )
                for y, x in pos[idx]:
                    self.items.append(
                        (fid, int(y - cube_size // 2), int(x - cube_size // 2))
                    )

            if n_neg > 0:
                idx = np.random.choice(
                    len(neg), size=min(n_neg, len(neg)), replace=len(neg) < n_neg
                )
                for y, x in neg[idx]:
                    self.items.append(
                        (fid, int(y - cube_size // 2), int(x - cube_size // 2))
                    )

        if len(self.items) == 0 and len(self._cache) > 0:
            fid0 = next(iter(self._cache.keys()))
            stack0, lbl0, msk0 = self._cache[fid0]
            H0, W0 = lbl0.shape
            y0 = max(0, (H0 - self.cube_size) // 2)
            x0 = max(0, (W0 - self.cube_size) // 2)
            self.items.append((fid0, int(y0), int(x0)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fid, y, x = self.items[idx]
        stack, lbl, msk = self._cache[fid]
        patch_x = stack[:, y : y + self.cube_size, x : x + self.cube_size]  # (S,cs,cs)
        patch_y = lbl[y : y + self.cube_size, x : x + self.cube_size]  # (cs,cs)

        if patch_x.shape[1] != self.cube_size or patch_x.shape[2] != self.cube_size:
            patch_x = np.zeros(
                (self.stack_count, self.cube_size, self.cube_size), dtype=np.float32
            )
            patch_y = np.zeros((self.cube_size, self.cube_size), dtype=np.float32)

        x_t = torch.from_numpy(np.asarray(patch_x, dtype=np.float32))
        y_t = torch.from_numpy(np.asarray(patch_y, dtype=np.float32)).unsqueeze(0)
        return x_t, y_t


train_fragment_ids = []
for d in sorted(os.listdir(TRAIN_ROOT)):
    if d.isdigit():
        train_fragment_ids.append(d)

train_ds = VesuviusPatchDataset(
    train_fragment_ids,
    cube_size=cube_size,
    stack_count=stack_count,
    pos_frac=0.5,
    patches_per_fragment=600,
)

_num_workers = min(4, (os.cpu_count() or 2))
g = torch.Generator()
g.manual_seed(seed)

train_loader = DataLoader(
    train_ds,
    batch_size=16,
    shuffle=True,
    num_workers=_num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_num_workers > 0),
    generator=g,
)

model_CNN.train()
for ep in range(epochs):
    running = 0.0
    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True).float()
        yb = yb.to(device, non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)
        pred = model_CNN(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        running += float(loss.item())
    if len(train_loader) == 0:
        break
    if (ep + 1) % 50 == 0:
        print(f"epoch {ep+1}/{epochs} loss {running/len(train_loader):.6f}")

sample_path = os.path.join(DATA_ROOT, "sample_submission.csv")
sample = pd.read_csv(sample_path)
test_ids = sample["Id"].astype(str).tolist()

rows = []
threshold = 0.5  # binary required by metric; keep default threshold

for fid in test_ids:
    surface_dir = os.path.join(TEST_ROOT, fid, "surface_volume")
    pred = process_volume_data(
        surface_dir, stack_count, cube_size, model_CNN, device, batch_size=8
    )  # (1,H,W)
    pred2d = pred[0].numpy()
    bin_mask = (pred2d >= threshold).astype(np.uint8)
    rle = rle_encode(bin_mask)
    rows.append({"Id": fid, "Predicted": rle})

sub = pd.DataFrame(rows, columns=["Id", "Predicted"])
sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with", len(sub), "rows")
