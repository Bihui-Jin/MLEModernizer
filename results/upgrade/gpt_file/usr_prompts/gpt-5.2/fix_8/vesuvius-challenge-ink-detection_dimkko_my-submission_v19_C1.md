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

0.0619493384633419

# 6. Current score

0.00022

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.00022) has done: 'Main bottlenecks are (1) repeatedly loading/processing huge TIFF stacks and (2) doing expensive per-block Python overhead during training/inference. The refactor keeps the same UNET, BCELoss, optimizer, epochs, cube/block logic, and thresholding, but makes data loading and preprocessing faster via parallel slice decode/preprocess, and reduces GPU/CPU transfer overhead by using a pinned-memory DataLoader over precomputed blocks. It also avoids recomputing block tensors every epoch by caching the already-tiled training tensors once per fragment (same exact values), and uses vectorized RLE string building to remove Python-join overhead. These changes are runtime-only and preserve evaluation semantics (same inputs to the model and same postprocessing), with only negligible floating-point differences possible from parallel slice preparation order (values are identical).'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

output_path = "./output/"
os.makedirs(output_path, exist_ok=True)



## === cell 1
import cv2
import gc
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import tifffile

cv2.setNumThreads(max(1, os.cpu_count() or 1))

gc.collect()



## === cell 2
from sklearn.model_selection import train_test_split



## === cell 3
device = "cuda" if torch.cuda.is_available() else "cpu"
device




## === cell 4
def seed_everything(seed: int):
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
batch_size = 8  # unchanged from provided script
cube_size = 32
epochs = 3  # unchanged from provided script
seed = 42
stack_count = 30
seed_everything(seed)

kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
block_size = cube_size

DATA_ROOT = "/kaggle/input/vesuvius-challenge-ink-detection"
if not os.path.exists(DATA_ROOT):
    DATA_ROOT = "/kaggle/data/vesuvius-challenge-ink-detection"

TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass




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
        x1 = x

        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x2 = x

        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x3 = x

        x = self.pool3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x4 = x

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



## === cell 11
from concurrent.futures import ThreadPoolExecutor


def _prep_slice(img2d: np.ndarray) -> np.ndarray:
    img = img2d.astype(np.float32, copy=False)
    img = cv2.filter2D(img, -1, kernel)
    img = np.clip(img, 0, None)
    img8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img8 = cv2.equalizeHist(img8)
    img8 = cv2.medianBlur(img8, 3)
    return img8.astype(np.float32) / 255.0


def _read_volume_stack(surface_dir: str, stack_count: int) -> np.ndarray:
    """
    Reads first stack_count tif slices, returns float32 array [C,H,W] in [0,1].
    """
    tif_paths = sorted(glob.glob(os.path.join(surface_dir, "*.tif")))
    if len(tif_paths) == 0:
        raise FileNotFoundError(f"No .tif files in {surface_dir}")
    tif_paths = tif_paths[:stack_count]

    first = tifffile.imread(tif_paths[0])
    if first.ndim == 3:
        first = first[..., 0]
    H, W = first.shape

    vol = np.empty((len(tif_paths), H, W), dtype=np.float32)
    vol[0] = _prep_slice(first)

    def _load_and_prep(path: str) -> np.ndarray:
        img = tifffile.imread(path)
        if img.ndim == 3:
            img = img[..., 0]
        return _prep_slice(img)

    if len(tif_paths) > 1:
        max_workers = min(8, (os.cpu_count() or 4))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for i, arr in enumerate(ex.map(_load_and_prep, tif_paths[1:]), start=1):
                vol[i] = arr

    return vol


def _read_mask_png(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return (m > 0).astype(np.uint8)


def _read_inklabels_png(path: str) -> np.ndarray:
    y = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if y is None:
        raise FileNotFoundError(path)
    return (y > 0).astype(np.uint8)


def _make_blocks(vol: np.ndarray, cube_size: int):
    """
    vol: [C,H,W]
    returns blocks tensor [N,C,cube,cube] and padding info.
    """
    C, H, W = vol.shape
    pad_h = (cube_size - (H % cube_size)) % cube_size
    pad_w = (cube_size - (W % cube_size)) % cube_size
    vol_t = torch.from_numpy(vol)  # [C,H,W]
    vol_t = F.pad(vol_t, (0, pad_w, 0, pad_h, 0, 0))
    _, Hp, Wp = vol_t.shape

    tiles = vol_t.unfold(1, cube_size, cube_size).unfold(2, cube_size, cube_size)
    blocks = tiles.permute(1, 2, 0, 3, 4).contiguous().view(-1, C, cube_size, cube_size)
    return blocks, (pad_h, pad_w), (Hp, Wp)


def _make_label_blocks(lbl: np.ndarray, cube_size: int, pad_h: int, pad_w: int):
    y = torch.from_numpy(lbl.astype(np.float32)).unsqueeze(0)  # [1,H,W]
    y = F.pad(y, (0, pad_w, 0, pad_h, 0, 0))  # [1,Hp,Wp]
    tiles = y.unfold(1, cube_size, cube_size).unfold(2, cube_size, cube_size)
    blocks = tiles.permute(1, 2, 0, 3, 4).contiguous().view(-1, 1, cube_size, cube_size)
    return blocks  # [N,1,cube,cube]


from torch.utils.data import TensorDataset, DataLoader


def process_volume_data_from_vol(
    vol: np.ndarray, cube_size: int, model_CNN, device, batch_size
):
    """
    Returns probability map tensor [1,Hp,Wp] from an already-preprocessed vol [C,H,W].
    """
    blocks, (pad_h, pad_w), (Hp, Wp) = _make_blocks(vol, cube_size)

    model_CNN.eval()
    ds = TensorDataset(blocks)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )

    outs = []
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device, non_blocking=(device == "cuda")).float()
            out = model_CNN(x).detach().cpu()  # [B,1,cube,cube]
            outs.append(out)
    out_all = torch.cat(outs, dim=0)  # [N,1,cube,cube]

    n_h = Hp // cube_size
    n_w = Wp // cube_size
    pred_full = (
        out_all.view(n_h, n_w, 1, cube_size, cube_size)
        .permute(2, 0, 3, 1, 4)
        .contiguous()
        .view(1, Hp, Wp)
    )
    return pred_full, (pad_h, pad_w)


def process_volume_data(path, stack_count, cube_size, model_CNN, device, batch_size):
    vol = _read_volume_stack(path, stack_count)  # [C,H,W]
    return process_volume_data_from_vol(vol, cube_size, model_CNN, device, batch_size)




## === cell 12
def list_fragments(root_dir: str):
    frags = []
    for d in sorted(os.listdir(root_dir)):
        frag_dir = os.path.join(root_dir, d)
        if not os.path.isdir(frag_dir):
            continue
        surface_dir = os.path.join(frag_dir, "surface_volume")
        if not os.path.isdir(surface_dir):
            continue
        if len(glob.glob(os.path.join(surface_dir, "*.tif"))) == 0:
            continue
        frags.append(d)
    return frags


train_fragments = list_fragments(TRAIN_DIR)
assert len(train_fragments) > 0, "No training fragments found"

cached_train = {}
for frag in train_fragments:
    frag_dir = os.path.join(TRAIN_DIR, frag)
    surface_dir = os.path.join(frag_dir, "surface_volume")
    label_path = os.path.join(frag_dir, "inklabels.png")

    vol = _read_volume_stack(surface_dir, stack_count)  # preprocessed [C,H,W]
    lbl = _read_inklabels_png(label_path)

    blocks, (pad_h, pad_w), _ = _make_blocks(vol, cube_size)
    y_blocks = _make_label_blocks(lbl, cube_size, pad_h, pad_w)

    cached_train[frag] = (blocks, y_blocks)
    gc.collect()

model_CNN.train()
for epoch in range(epochs):
    epoch_losses = []
    for frag in train_fragments:
        blocks, y_blocks = cached_train[frag]

        idx = torch.randperm(blocks.shape[0])
        blocks_shuf = blocks[idx]
        y_blocks_shuf = y_blocks[idx]

        ds = TensorDataset(blocks_shuf, y_blocks_shuf)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,  # already shuffled above
            num_workers=0,
            pin_memory=(device == "cuda"),
            drop_last=False,
        )

        for x, y in loader:
            x = x.to(device, non_blocking=(device == "cuda")).float()
            y = y.to(device, non_blocking=(device == "cuda")).float()

            optimizer.zero_grad(set_to_none=True)
            out = model_CNN(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))

        del idx, blocks_shuf, y_blocks_shuf, ds, loader
        gc.collect()

    print(f"epoch {epoch+1}/{epochs} loss={np.mean(epoch_losses):.6f}")




## === cell 13
def rle_encode(mask: np.ndarray) -> str:
    """
    mask: 2D uint8 {0,1}, row-major as per competition (left->right, top->bottom).
    Returns space-delimited start/length pairs with 1-based indexing.
    """
    pixels = mask.reshape(-1, order="C").astype(np.uint8, copy=False)
    if pixels.size == 0:
        return ""
    padded = np.empty(pixels.size + 2, dtype=np.uint8)
    padded[0] = 0
    padded[-1] = 0
    padded[1:-1] = pixels
    changes = np.flatnonzero(padded[1:] != padded[:-1]) + 1
    if changes.size == 0:
        return ""
    runs = changes[0::2]
    ends = changes[1::2]
    lengths = ends - runs
    out = np.empty(runs.size * 2, dtype=np.int64)
    out[0::2] = runs
    out[1::2] = lengths
    return " ".join(map(str, out.tolist()))


thr = 0.5

sample_path = os.path.join(DATA_ROOT, "sample_submission.csv")
if not os.path.exists(sample_path):
    sample_path = "/kaggle/input/sample_submission.csv"
sample_sub = pd.read_csv(sample_path)
expected_ids = sample_sub["Id"].tolist()

test_fragments = list_fragments(TEST_DIR)
assert len(test_fragments) > 0, "No test fragments found"

cached_test_vol = {}
cached_test_mask = {}
for frag in test_fragments:
    frag_dir = os.path.join(TEST_DIR, frag)
    surface_dir = os.path.join(frag_dir, "surface_volume")
    mask_path = os.path.join(frag_dir, "mask.png")
    cached_test_vol[frag] = _read_volume_stack(surface_dir, stack_count)
    cached_test_mask[frag] = _read_mask_png(mask_path)

pred_map_by_id = {}
for frag in tqdm(test_fragments, desc="Predict test"):
    vol = cached_test_vol[frag]
    m = cached_test_mask[frag]

    prob_map, _ = process_volume_data_from_vol(
        vol, cube_size, model_CNN, device, batch_size
    )
    prob = prob_map.squeeze(0).numpy()

    H, W = m.shape
    prob = prob[:H, :W]

    pred = (prob > thr).astype(np.uint8)
    pred = pred * m
    pred_map_by_id[frag] = rle_encode(pred)

rows = []
for frag_id in expected_ids:
    rows.append({"Id": frag_id, "Predicted": pred_map_by_id.get(frag_id, "")})

sub = pd.DataFrame(rows)[["Id", "Predicted"]]
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
