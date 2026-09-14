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
batch_size = 50  # размер батча
cube_size = 32  # нарезка изображений на cube_size х cube_size
epochs = 1000
seed = 42
stack_count = 30  # количество срезов для модели
seed_everything(seed)
kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
block_size = cube_size




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

        b, c, hgt, wdt = h.shape
        seq = h.permute(0, 2, 3, 1).contiguous().view(b, hgt * wdt, c)  # (B, HW, C)
        seq, _ = self.rnn(seq)
        h2 = seq.view(b, hgt, wdt, c).permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        h2 = self.relu(self.conv4(h2))
        h2 = x + h2
        return h2




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
def _read_tif_gray(path_tif: str) -> np.ndarray:
    """Read grayscale TIFF reliably in Kaggle; cv2.imread can fail for 16-bit TIFF."""
    img = tifffile.imread(path_tif)
    if img.ndim == 3:
        img = img[..., 0]
    return img


def _equalize_like_16bit_to_float01(x16: np.ndarray) -> np.ndarray:
    """
    Bugfix: cv2.equalizeHist supports only CV_8UC1, but our slices are often 16-bit.
    Minimal replacement: apply CLAHE on a downscaled 8-bit view, then return float32 in [0,1].
    """
    x16 = np.clip(x16, 0, 65535).astype(np.uint16)
    x8 = (x16 / 257).astype(np.uint8)  # 0..65535 -> 0..255
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    x8e = clahe.apply(x8)
    return x8e.astype(np.float32) / 255.0


def _select_center_slices(file_list, stack_count):
    """
    Score-relevant, minimal change: using centered slices tends to be more informative than
    always taking the first slices, while keeping the same stack_count and preprocessing.
    """
    n = len(file_list)
    if n <= stack_count:
        return file_list
    start = max(0, (n - stack_count) // 2)
    return file_list[start : start + stack_count]


def process_volume_data(path, stack_count, cube_size, model_CNN, device, batch_size):
    model_CNN.eval()

    file_list = sorted(glob.glob(os.path.join(path, "*.tif")))
    if len(file_list) == 0:
        raise FileNotFoundError(f"No .tif files found in {path}")

    file_list = _select_center_slices(file_list, stack_count)

    first = _read_tif_gray(file_list[0])
    stack_3 = torch.zeros(
        (stack_count, first.shape[0], first.shape[1]), dtype=torch.float32
    )

    for idx, filename in enumerate(tqdm(file_list[:stack_count], desc="Read slices")):
        slice_data = _read_tif_gray(filename).astype(np.float32)

        slice_data = cv2.filter2D(slice_data, -1, kernel)
        slice_data = _equalize_like_16bit_to_float01(slice_data * 1.0)

        slice_u8 = np.clip(slice_data * 255.0, 0, 255).astype(np.uint8)
        slice_u8 = cv2.medianBlur(slice_u8, 3)
        slice_data = slice_u8.astype(np.float32) / 255.0

        stack_3[idx, :, :] = torch.from_numpy(slice_data)
        gc.collect()

    padding = (
        (cube_size - stack_3.shape[1] % cube_size) % cube_size,
        (cube_size - stack_3.shape[2] % cube_size) % cube_size,
    )

    image = F.pad(stack_3, (0, padding[1], 0, padding[0], 0, 0))
    blocks_stack_3 = []
    for i in tqdm(range(0, image.shape[1], block_size), desc="Make blocks y"):
        for j in range(0, image.shape[2], block_size):
            block = image[:, i : i + block_size, j : j + block_size]
            block = block.unsqueeze(0)
            blocks_stack_3.append(block)

    blocks_stack = torch.cat(blocks_stack_3, dim=0)
    blocks_stack = torch.split(blocks_stack, batch_size, dim=0)

    out_shape = (
        1,
        stack_3.shape[1] + padding[0],
        stack_3.shape[2] + padding[1],
    )  # (1,H,W)
    stack_out = torch.zeros(out_shape, dtype=torch.float32)
    stack_cnt = torch.zeros(out_shape, dtype=torch.float32)

    y_start = -cube_size
    i = 0
    with torch.no_grad():
        for images in tqdm(blocks_stack, desc="Predict blocks"):
            images = images.to(device).float()
            outputs = model_CNN(images)  # (B,1,H,W)

            for cube in outputs.to("cpu"):
                x_start = (i * cube_size) % out_shape[2]
                if x_start == 0:
                    y_start += cube_size

                h, w = cube.shape[1], cube.shape[2]  # cube is (1,H,W)
                stack_out[:, y_start : y_start + h, x_start : x_start + w] += cube
                stack_cnt[:, y_start : y_start + h, x_start : x_start + w] += 1.0
                i += 1

    stack_cnt = stack_cnt.clamp_min(1.0)
    stack_out = stack_out / stack_cnt
    stack_out = stack_out[:, : stack_3.shape[1], : stack_3.shape[2]]
    return stack_out




## === cell 11
def rle_encode(mask: np.ndarray) -> str:
    """
    Kaggle Vesuvius expects 1-indexed RLE, flattened row-major (left-to-right, top-to-bottom).
    mask: 2D uint8/bool array with 1 for ink, 0 for background
    """
    pixels = mask.flatten(order="C").astype(np.uint8)
    padded = np.concatenate([[0], pixels, [0]])
    changes = np.where(padded[1:] != padded[:-1])[0] + 1
    runs = changes[::2]
    lengths = changes[1::2] - runs
    if len(runs) == 0:
        return ""
    rle = " ".join(str(x) for pair in zip(runs, lengths) for x in pair)
    return rle


def fbeta_from_counts(tp: float, fp: float, fn: float, beta: float = 0.5) -> float:
    b2 = beta * beta
    denom = (1 + b2) * tp + b2 * fn + fp
    if denom <= 0:
        return 0.0
    return ((1 + b2) * tp) / denom


def remove_small_components(mask01: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 1:
        return mask01
    mask01 = mask01.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask01, connectivity=8
    )
    if num_labels <= 1:
        return mask01
    out = np.zeros_like(mask01)
    for lab in range(1, num_labels):
        area = stats[lab, cv2.CC_STAT_AREA]
        if area >= min_size:
            out[labels == lab] = 1
    return out




## === cell 12
DATA_ROOT_CANDIDATES = [
    "/kaggle/input/vesuvius-challenge-ink-detection",
    "/kaggle/data/vesuvius-challenge-ink-detection",
    "/kaggle/input",
    "/kaggle/data",
]
DATA_ROOT = None
for cand in DATA_ROOT_CANDIDATES:
    if os.path.exists(os.path.join(cand, "test")) and os.path.exists(
        os.path.join(cand, "sample_submission.csv")
    ):
        DATA_ROOT = cand
        break
if DATA_ROOT is None:
    DATA_ROOT = "/kaggle/input/vesuvius-challenge-ink-detection"

test_dir = os.path.join(DATA_ROOT, "test")
train_dir = os.path.join(DATA_ROOT, "train")
sample_path = os.path.join(DATA_ROOT, "sample_submission.csv")
sample_sub = pd.read_csv(sample_path)

all_dirs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]


def _is_valid_fragment_id(d: str) -> bool:
    vol_dir = os.path.join(test_dir, d, "surface_volume")
    return os.path.isdir(vol_dir) and (
        len(glob.glob(os.path.join(vol_dir, "*.tif"))) > 0
    )


test_ids = sorted([d for d in all_dirs if _is_valid_fragment_id(d)])

if len(test_ids) == 0:
    raise FileNotFoundError(
        f"No valid fragment folders with surface_volume/*.tif found under: {test_dir}. "
        f"Found dirs: {all_dirs[:20]}"
    )


def discover_train_fragments(train_dir: str):
    if not os.path.isdir(train_dir):
        return []
    out = []
    for d in sorted(os.listdir(train_dir)):
        frag_dir = os.path.join(train_dir, d)
        if not os.path.isdir(frag_dir):
            continue
        vol_dir = os.path.join(frag_dir, "surface_volume")
        ink_path = os.path.join(frag_dir, "inklabels.png")
        mask_path = os.path.join(frag_dir, "mask.png")
        if (
            os.path.isdir(vol_dir)
            and os.path.exists(ink_path)
            and os.path.exists(mask_path)
        ):
            if len(glob.glob(os.path.join(vol_dir, "*.tif"))) > 0:
                out.append(d)
    return out


def load_mask_png(path: str) -> np.ndarray:
    m = np.array(Image.open(path).convert("L"))
    return (m > 0).astype(np.uint8)




## === cell 13
model_CNN = UNET(stack_count).to(device)

possible_weight_paths = [
    "/kaggle/input/model-2/model_CNN_1000_epoch_30_stack.pth",
    "/kaggle/input/model_CNN_1000_epoch_30_stack.pth",
    "/kaggle/working/model_CNN_1000_epoch_30_stack.pth",
    "./model_CNN_1000_epoch_30_stack.pth",
]

loaded = False
load_errors = []
for p in possible_weight_paths:
    if os.path.exists(p):
        try:
            state_dict = torch.load(p, map_location=device, weights_only=True)
            model_CNN.load_state_dict(state_dict, strict=True)
            loaded = True
            break
        except TypeError:
            try:
                state_dict = torch.load(p, map_location=device)
                model_CNN.load_state_dict(state_dict, strict=True)
                loaded = True
                break
            except Exception as e:
                load_errors.append((p, repr(e)))
        except Exception as e:
            load_errors.append((p, repr(e)))


def make_train_patches_from_fragment(frag_id: str):
    vol_path = os.path.join(train_dir, frag_id, "surface_volume")
    file_list = sorted(glob.glob(os.path.join(vol_path, "*.tif")))
    if len(file_list) == 0:
        return None

    file_list = _select_center_slices(file_list, stack_count)

    first = _read_tif_gray(file_list[0])
    stack_3 = torch.zeros(
        (stack_count, first.shape[0], first.shape[1]), dtype=torch.float32
    )
    for idx, filename in enumerate(file_list[:stack_count]):
        slice_data = _read_tif_gray(filename).astype(np.float32)
        slice_data = cv2.filter2D(slice_data, -1, kernel)
        slice_data = _equalize_like_16bit_to_float01(slice_data * 1.0)
        slice_u8 = np.clip(slice_data * 255.0, 0, 255).astype(np.uint8)
        slice_u8 = cv2.medianBlur(slice_u8, 3)
        slice_data = slice_u8.astype(np.float32) / 255.0
        stack_3[idx, :, :] = torch.from_numpy(slice_data)

    gt = load_mask_png(os.path.join(train_dir, frag_id, "inklabels.png"))
    valid = load_mask_png(os.path.join(train_dir, frag_id, "mask.png"))
    if gt.shape != (stack_3.shape[1], stack_3.shape[2]):
        return None

    padding = (
        (cube_size - stack_3.shape[1] % cube_size) % cube_size,
        (cube_size - stack_3.shape[2] % cube_size) % cube_size,
    )
    image = F.pad(stack_3, (0, padding[1], 0, padding[0], 0, 0))
    gt_t = torch.from_numpy(gt.astype(np.float32))
    valid_t = torch.from_numpy(valid.astype(np.float32))
    gt_t = F.pad(gt_t.unsqueeze(0), (0, padding[1], 0, padding[0])).squeeze(0)
    valid_t = F.pad(valid_t.unsqueeze(0), (0, padding[1], 0, padding[0])).squeeze(0)

    X_blocks, Y_blocks, V_blocks = [], [], []
    for i in range(0, image.shape[1], block_size):
        for j in range(0, image.shape[2], block_size):
            blk = image[:, i : i + block_size, j : j + block_size]  # (C,H,W)
            yb = gt_t[i : i + block_size, j : j + block_size]  # (H,W)
            vb = valid_t[i : i + block_size, j : j + block_size]  # (H,W)
            X_blocks.append(blk.unsqueeze(0))
            Y_blocks.append(yb.unsqueeze(0).unsqueeze(0))  # (1,1,H,W)
            V_blocks.append(vb.unsqueeze(0).unsqueeze(0))  # (1,1,H,W)

    X = torch.cat(X_blocks, dim=0)  # (N,C,H,W)
    Y = torch.cat(Y_blocks, dim=0)  # (N,1,H,W)
    V = torch.cat(V_blocks, dim=0)  # (N,1,H,W)
    return X, Y, V


class StreamingPatchDataset(Dataset):
    def __init__(self, train_ids):
        self.items = []
        self.cache = {}
        for frag_id in train_ids:
            packs = make_train_patches_from_fragment(frag_id)
            if packs is None:
                continue
            X, Y, V = packs
            keep = V.view(V.shape[0], -1).sum(dim=1) > 0
            idxs = torch.nonzero(keep, as_tuple=False).view(-1).tolist()
            if len(idxs) == 0:
                continue
            self.cache[frag_id] = (X, Y, V)
            for k in idxs:
                self.items.append((frag_id, k))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        frag_id, k = self.items[i]
        X, Y, V = self.cache[frag_id]
        return X[k], Y[k], V[k]


def train_if_needed(model_CNN, device):
    train_ids = discover_train_fragments(train_dir)
    if len(train_ids) == 0:
        return False

    ds = StreamingPatchDataset(train_ids)
    if len(ds) == 0:
        return False

    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False
    )

    model_CNN.train()
    optimizer = optim.Adam(model_CNN.parameters(), lr=learning_rate)
    bce = nn.BCELoss(reduction="none")

    train_epochs = 3
    for ep in range(train_epochs):
        for xb, yb, vb in tqdm(dl, desc=f"Train (fallback) ep {ep+1}/{train_epochs}"):
            xb = xb.to(device).float()
            yb = yb.to(device).float()
            vb = vb.to(device).float()

            optimizer.zero_grad(set_to_none=True)
            out = model_CNN(xb)
            loss_map = bce(out, yb) * vb
            denom = vb.sum().clamp_min(1.0)
            loss = loss_map.sum() / denom
            loss.backward()
            optimizer.step()
        gc.collect()

    model_CNN.eval()
    try:
        torch.save(
            model_CNN.state_dict(), "/kaggle/working/model_CNN_1000_epoch_30_stack.pth"
        )
    except Exception:
        pass
    return True


if not loaded:
    _trained = train_if_needed(model_CNN, device)
    loaded = _trained

if (not loaded) and (len(load_errors) > 0):
    raise RuntimeError(f"Found weight files but failed to load: {load_errors}")

model_CNN.eval()
gc.collect()
loaded




## === cell 14
def compute_best_thr_and_min_size(model_CNN, device):
    train_ids = discover_train_fragments(train_dir)

    if len(train_ids) == 0:
        return 0.85, 1

    thr_grid = np.round(np.arange(0.70, 0.96, 0.02), 2)  # 0.70..0.94
    min_size_grid = [1, 8, 16, 32]  # tiny cleanup only

    best = (-1.0, 0.85, 1)  # (score, thr, min_size)

    for frag_id in tqdm(train_ids, desc="Tune thr on train"):
        vol_path = os.path.join(train_dir, frag_id, "surface_volume")
        pred_map = process_volume_data(
            vol_path, stack_count, cube_size, model_CNN, device, batch_size
        )
        pred_map = pred_map.squeeze(0).numpy()

        gt = load_mask_png(os.path.join(train_dir, frag_id, "inklabels.png"))
        valid = load_mask_png(os.path.join(train_dir, frag_id, "mask.png"))

        if pred_map.shape != gt.shape:
            continue

        p = pred_map[valid > 0].reshape(-1)
        y = gt[valid > 0].reshape(-1)

        order = np.argsort(-p)  # descending
        p_sorted = p[order]
        y_sorted = y[order].astype(np.uint8)

        total_pos = float(y_sorted.sum())

        tp_cum = np.cumsum(y_sorted, dtype=np.float64)
        fp_cum = np.cumsum(1 - y_sorted, dtype=np.float64)

        for thr in thr_grid:
            k = np.searchsorted(-p_sorted, -thr, side="right")  # number selected
            if k <= 0:
                tp = 0.0
                fp = 0.0
            else:
                tp = float(tp_cum[k - 1])
                fp = float(fp_cum[k - 1])
            fn = total_pos - tp

            base_score = fbeta_from_counts(tp, fp, fn, beta=0.5)

            if base_score >= best[0] - 0.01:
                for min_size in min_size_grid:
                    pred_mask = (pred_map >= thr).astype(np.uint8)
                    if min_size > 1:
                        pred_mask = remove_small_components(
                            pred_mask, min_size=min_size
                        )
                    pred_mask = (pred_mask * valid).astype(np.uint8)

                    pm = pred_mask[valid > 0].reshape(-1)
                    tp2 = float(((pm == 1) & (y == 1)).sum())
                    fp2 = float(((pm == 1) & (y == 0)).sum())
                    fn2 = float(((pm == 0) & (y == 1)).sum())
                    sc = fbeta_from_counts(tp2, fp2, fn2, beta=0.5)
                    if sc > best[0]:
                        best = (sc, float(thr), int(min_size))

    if best[0] < 0:
        return 0.85, 1
    return best[1], best[2]


thr, min_comp_size = compute_best_thr_and_min_size(model_CNN, device)
thr, min_comp_size



## === cell 15
pred_rows = []

for frag_id in tqdm(test_ids, desc="Fragments"):
    vol_path = os.path.join(test_dir, frag_id, "surface_volume")
    pred_map = process_volume_data(
        vol_path, stack_count, cube_size, model_CNN, device, batch_size
    )
    pred_map = pred_map.squeeze(0).numpy()

    mask_path = os.path.join(test_dir, frag_id, "mask.png")
    if os.path.exists(mask_path):
        valid_mask = np.array(Image.open(mask_path).convert("L"))
        valid_mask = (valid_mask > 0).astype(np.uint8)
        if valid_mask.shape == pred_map.shape:
            pred_map = pred_map * valid_mask

    pred_mask = (pred_map >= thr).astype(np.uint8)

    if min_comp_size > 1:
        pred_mask = remove_small_components(pred_mask, min_size=min_comp_size)

    pred_rows.append({"Id": frag_id, "Predicted": rle_encode(pred_mask)})

submission_pred = pd.DataFrame(pred_rows)

submission = sample_sub[["Id"]].merge(submission_pred, on="Id", how="left")
submission["Predicted"] = submission["Predicted"].fillna("")

output_path = "submission.csv"
submission.to_csv(output_path, index=False)
output_path
