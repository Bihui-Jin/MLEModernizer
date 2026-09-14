# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

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
import numpy as np
import glob
from PIL import Image
import torch.utils.data as data
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from ipywidgets import interact, fixed
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from scipy.ndimage import morphology
from tqdm.notebook import tqdm_notebook
from torch.utils.data import random_split
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import gc
from torch.utils.data import TensorDataset, SubsetRandomSampler
from torchvision.transforms import Normalize
import tifffile
import torch.nn.functional as F

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
device = f"cuda" if torch.cuda.is_available() else "cpu"
device




## === cell 4
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




## === cell 5
learning_rate = 0.0001
batch_size = 50
cube_size = 32
epochs = 6  # FIX: was 1000 (won't finish within time). Keep core logic; just a practical runtime fix.
seed = 42
stack_count = 30
seed_everything(seed)
kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
block_size = cube_size

DATA_ROOT = "/kaggle/input/vesuvius-challenge-ink-detection"
if not os.path.exists(DATA_ROOT):
    DATA_ROOT = "/kaggle/input"  # fallback for alternate mount layouts

TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")




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
        for i in range(self.t):
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
gc.collect()




## === cell 11
def _list_tif_slices(surface_volume_dir):
    fns = sorted(glob.glob(os.path.join(surface_volume_dir, "*.tif")))
    return fns


def _read_tif_gray16(path):
    img = tifffile.imread(path)
    if img.ndim == 3:
        img = img[..., 0]
    img = img.astype(np.float32)
    return img


def _preprocess_slice(slice_img, kernel):
    x = slice_img
    x = cv2.filter2D(x, -1, kernel)
    x8 = np.clip(x / 65535.0 * 255.0, 0, 255).astype(np.uint8)
    x8 = cv2.equalizeHist(x8)
    x8 = cv2.medianBlur(x8, 3)
    x = x8.astype(np.float32) / 255.0
    return x


def process_volume_data(path, stack_count, cube_size, model_CNN, device, batch_size):
    slice_paths = _list_tif_slices(path)
    if len(slice_paths) == 0:
        raise FileNotFoundError(f"No .tif slices found in {path}")

    first = _read_tif_gray16(slice_paths[0])
    H, W = first.shape[:2]
    stack_3 = torch.zeros((stack_count, H, W), dtype=torch.float32)

    for idx, fn in enumerate(tqdm_notebook(slice_paths[:stack_count])):
        sl = _read_tif_gray16(fn)
        sl = _preprocess_slice(sl, kernel)
        stack_3[idx] = torch.from_numpy(sl)
        gc.collect()

    padding = (
        cube_size - stack_3.shape[1] % cube_size,
        cube_size - stack_3.shape[2] % cube_size,
    )
    image = F.pad(stack_3, (0, padding[1], 0, padding[0], 0, 0))

    blocks_stack_3 = []
    for i in tqdm_notebook(range(0, image.shape[1], cube_size)):
        for j in range(0, image.shape[2], cube_size):
            block = image[:, i : i + cube_size, j : j + cube_size].unsqueeze(0)
            blocks_stack_3.append(block)

    blocks_stack = torch.cat(blocks_stack_3, dim=0)
    blocks_stack = torch.split(blocks_stack, batch_size, dim=0)

    y_start = -cube_size
    i = 0
    stack_shape = (1, stack_3.shape[1] + padding[0], stack_3.shape[2] + padding[1])
    out_full = torch.zeros(stack_shape, dtype=torch.float32)

    model_CNN.eval()
    with torch.no_grad():
        for images in tqdm_notebook(blocks_stack):
            images = images.to(device).float()
            outputs = model_CNN(images)
            for cube in outputs.to("cpu"):
                x_start = (i * cube_size) % stack_shape[2]
                if x_start == 0:
                    y_start += cube_size
                out_full[
                    :,
                    y_start : y_start + cube.shape[1],
                    x_start : x_start + cube.shape[2],
                ] += cube[0]
                i += 1
    return out_full




## === cell 12


def rle_encode(mask: np.ndarray) -> str:
    """
    mask: 2D boolean/0-1 array. Pixels numbered left-to-right, top-to-bottom, 1-indexed.
    Returns space-delimited start length pairs.
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    pixels = mask.flatten(order="C")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


class VesuviusPatchDataset(Dataset):
    def __init__(
        self, fragment_dirs, stack_count, cube_size, max_patches_per_fragment=4000
    ):
        self.samples = []
        self.stack_count = stack_count
        self.cube_size = cube_size

        for fdir in fragment_dirs:
            vol_dir = os.path.join(fdir, "surface_volume")
            mask_path = os.path.join(fdir, "mask.png")
            ink_path = os.path.join(fdir, "inklabels.png")

            slice_paths = _list_tif_slices(vol_dir)
            if len(slice_paths) < stack_count:
                continue

            m = np.array(Image.open(mask_path).convert("L")) > 0
            y = np.array(Image.open(ink_path).convert("L")) > 0
            y = (y & m).astype(np.uint8)

            H, W = y.shape
            ys, xs = np.where(m)
            if len(xs) == 0:
                continue

            rng = np.random.default_rng(seed)
            idxs = rng.choice(
                len(xs), size=min(max_patches_per_fragment, len(xs)), replace=False
            )

            for k in idxs:
                cy, cx = int(ys[k]), int(xs[k])
                y0 = cy - cube_size // 2
                x0 = cx - cube_size // 2
                y0 = max(0, min(H - cube_size, y0))
                x0 = max(0, min(W - cube_size, x0))
                self.samples.append(
                    (
                        slice_paths[:stack_count],
                        y0,
                        x0,
                        y0 + cube_size,
                        x0 + cube_size,
                        y0,
                        x0,
                        fdir,
                    )
                )

            gc.collect()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        slice_paths, y0, x0, y1, x1, ly0, lx0, fdir = self.samples[idx]

        y_full = (
            np.array(Image.open(os.path.join(fdir, "inklabels.png")).convert("L")) > 0
        )
        m_full = np.array(Image.open(os.path.join(fdir, "mask.png")).convert("L")) > 0
        y_patch = (
            y_full[ly0 : ly0 + self.cube_size, lx0 : lx0 + self.cube_size]
            & m_full[ly0 : ly0 + self.cube_size, lx0 : lx0 + self.cube_size]
        ).astype(np.float32)

        x_stack = np.zeros(
            (self.stack_count, self.cube_size, self.cube_size), dtype=np.float32
        )
        for c in range(self.stack_count):
            sl = _read_tif_gray16(slice_paths[c])
            sl = _preprocess_slice(sl, kernel)
            x_stack[c] = sl[y0:y1, x0:x1]

        x = torch.from_numpy(x_stack)  # (C,H,W)
        y = torch.from_numpy(y_patch).unsqueeze(0)  # (1,H,W)
        return x, y


def train_model(model, train_loader, device, epochs, lr):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for ep in range(epochs):
        running = 0.0
        for xb, yb in tqdm(
            train_loader, desc=f"train epoch {ep+1}/{epochs}", leave=False
        ):
            xb = xb.to(device).float()
            yb = yb.to(device).float()

            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu())
        gc.collect()
    return model


def predict_fragment(
    model, fragment_dir, stack_count, cube_size, batch_size, device, thr=0.5
):
    vol_dir = os.path.join(fragment_dir, "surface_volume")
    mask_path = os.path.join(fragment_dir, "mask.png")

    slice_paths = _list_tif_slices(vol_dir)
    if len(slice_paths) < stack_count:
        raise RuntimeError(f"Not enough slices in {vol_dir}: {len(slice_paths)}")

    m = np.array(Image.open(mask_path).convert("L")) > 0
    H, W = m.shape

    stack = np.zeros((stack_count, H, W), dtype=np.float32)
    for c in range(stack_count):
        sl = _read_tif_gray16(slice_paths[c])
        stack[c] = _preprocess_slice(sl, kernel)

    pad_h = (cube_size - H % cube_size) % cube_size
    pad_w = (cube_size - W % cube_size) % cube_size
    stack_t = torch.from_numpy(stack)
    stack_t = F.pad(stack_t, (0, pad_w, 0, pad_h, 0, 0))  # (C,Hp,Wp)
    Hp, Wp = stack_t.shape[1], stack_t.shape[2]

    patches = []
    coords = []
    for y0 in range(0, Hp, cube_size):
        for x0 in range(0, Wp, cube_size):
            patches.append(
                stack_t[:, y0 : y0 + cube_size, x0 : x0 + cube_size].unsqueeze(0)
            )
            coords.append((y0, x0))
    patches = torch.cat(patches, dim=0)

    model.eval()
    pred_full = torch.zeros((Hp, Wp), dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, patches.shape[0], batch_size):
            xb = patches[i : i + batch_size].to(device).float()
            out = model(xb).detach().cpu()  # (B,1,32,32)
            out = out[:, 0]
            for j in range(out.shape[0]):
                y0, x0 = coords[i + j]
                pred_full[y0 : y0 + cube_size, x0 : x0 + cube_size] = out[j]

    pred_full = pred_full[:H, :W].numpy()
    pred_bin = (pred_full >= thr) & m
    return pred_bin.astype(np.uint8)


train_fragment_dirs = []
for frag in sorted(os.listdir(TRAIN_DIR)):
    fdir = os.path.join(TRAIN_DIR, frag)
    if os.path.isdir(fdir) and os.path.exists(os.path.join(fdir, "inklabels.png")):
        train_fragment_dirs.append(fdir)

train_ds = VesuviusPatchDataset(
    fragment_dirs=train_fragment_dirs,
    stack_count=stack_count,
    cube_size=cube_size,
    max_patches_per_fragment=3000,
)
train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

model_CNN = train_model(
    model_CNN, train_loader, device=device, epochs=epochs, lr=learning_rate
)

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
ids = sample_sub["Id"].tolist()

rows = []
for fid in ids:
    fdir = os.path.join(TEST_DIR, str(fid))
    if not os.path.exists(fdir):
        alt = os.path.join(
            DATA_ROOT, "vesuvius-challenge-ink-detection", "test", str(fid)
        )
        if os.path.exists(alt):
            fdir = alt
        else:
            raise FileNotFoundError(
                f"Test fragment dir not found for Id={fid} at {fdir}"
            )

    pred_mask = predict_fragment(
        model_CNN,
        fdir,
        stack_count=stack_count,
        cube_size=cube_size,
        batch_size=batch_size,
        device=device,
        thr=0.5,
    )
    rle = rle_encode(pred_mask)
    rows.append({"Id": fid, "Predicted": rle})
    gc.collect()

sub = pd.DataFrame(rows)
sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
