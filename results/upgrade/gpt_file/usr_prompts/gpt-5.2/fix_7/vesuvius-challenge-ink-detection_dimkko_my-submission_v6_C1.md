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

0.1034899101786769

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import gc
import glob
import random

import numpy as np
import pandas as pd

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import tifffile

output_path = "./output/"
if not os.path.exists(output_path):
    os.makedirs(output_path)



## === cell 1
from catboost import (
    CatBoostClassifier,
    Pool,
)  # kept to preserve original environment usage
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_recall_curve,
)  # kept
from sklearn.model_selection import train_test_split  # kept
from joblib import Parallel, delayed  # kept



## === cell 2
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)




## === cell 3
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




## === cell 4
learning_rate = 0.001
batch_size = 100  # inference batching for tiles
cube_size = 32
epochs = 1000
seed = 42
stack_count = 15
seed_everything(seed)

kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
block_size = cube_size

RLE_THRESHOLD = 0.55




## === cell 5
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




## === cell 6
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




## === cell 7
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

    def forward(self, x):
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        inp = h
        for _ in range(self.t):
            h = self.relu(self.conv3(h) + inp)
            inp = h
        h = self.relu(self.conv4(h))
        if h.shape == x.shape:
            h = x + h
        return h




## === cell 8
class CNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.attn1 = SelfAttentionBlock(64)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.drop1 = nn.Dropout(p=0.1)

        self.residual_block1 = ResidualBlock(64, 64)
        self.r2u_block1 = R2U_Block(64, 64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.attn2 = SelfAttentionBlock(128)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.drop2 = nn.Dropout(p=0.1)

        self.residual_block2 = ResidualBlock(128, 128)
        self.r2u_block2 = R2U_Block(128, 128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()
        self.attn3 = SelfAttentionBlock(256)
        self.pool3 = nn.MaxPool2d(2, stride=2)

        self.residual_block3 = ResidualBlock(256, 256)
        self.r2u_block3 = R2U_Block(256, 256)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU()
        self.attn4 = SelfAttentionBlock(512)
        self.pool4 = nn.MaxPool2d(2, stride=2)

        self.residual_block4 = ResidualBlock(512, 512)
        self.r2u_block4 = R2U_Block(512, 512)

        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(1024)
        self.relu5 = nn.ReLU()
        self.attn5 = SelfAttentionBlock(1024)

        self.residual_block5 = ResidualBlock(1024, 1024)
        self.r2u_block5 = R2U_Block(1024, 1024)

        self.upconv6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv6 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.relu6 = nn.ReLU()
        self.attn6 = SelfAttentionBlock(512)

        self.residual_block6 = ResidualBlock(512, 512)
        self.r2u_block6 = R2U_Block(512, 512)

        self.upconv7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.relu7 = nn.ReLU()
        self.attn7 = SelfAttentionBlock(256)

        self.residual_block7 = ResidualBlock(256, 256)
        self.r2u_block7 = R2U_Block(256, 256)

        self.upconv8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(128)
        self.relu8 = nn.ReLU()
        self.attn8 = SelfAttentionBlock(128)

        self.residual_block8 = ResidualBlock(128, 128)
        self.r2u_block8 = R2U_Block(128, 128)

        self.upconv9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(64)
        self.relu9 = nn.ReLU()
        self.attn9 = SelfAttentionBlock(64)

        self.residual_block9 = ResidualBlock(64, 64)
        self.r2u_block9 = R2U_Block(64, 64)

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




## === cell 9
DATA_ROOT_CANDIDATES = [
    "/kaggle/input/vesuvius-challenge-ink-detection",
    "/kaggle/data/vesuvius-challenge-ink-detection",
    "/kaggle/input",
    "/kaggle/data",
]
DATA_ROOT = None
for p in DATA_ROOT_CANDIDATES:
    if os.path.isdir(os.path.join(p, "train")) and os.path.isdir(
        os.path.join(p, "test")
    ):
        DATA_ROOT = p
        break
if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate DATA_ROOT with both train/ and test/ under known candidates."
    )
print("Using DATA_ROOT:", DATA_ROOT)

model_CNN = CNN(stack_count).to(device)


def find_ckpt():
    preferred = "/kaggle/input/model-1/model_CNN_93_epoch_15_stack.pth"
    if os.path.exists(preferred):
        return preferred

    candidates = glob.glob("/kaggle/input/**/model_CNN*_stack*.pth", recursive=True)
    if len(candidates) > 0:
        return sorted(candidates)[0]

    candidates = glob.glob("/kaggle/input/**/*.pth", recursive=True)
    return sorted(candidates)[0] if len(candidates) > 0 else None


ckpt_path = find_ckpt()
loaded_ckpt = False
if ckpt_path is not None and os.path.exists(ckpt_path):
    try:
        state_dict = torch.load(ckpt_path, map_location=device)
        model_CNN.load_state_dict(state_dict, strict=True)
        model_CNN.eval()
        loaded_ckpt = True
        print(f"Loaded checkpoint: {ckpt_path}")
    except Exception as e:
        print(f"Found checkpoint but failed to load strictly: {ckpt_path}\nError: {e}")
        loaded_ckpt = False

if not loaded_ckpt:
    print(
        "Checkpoint not found/loaded. Training minimal warm-start on provided train fragments (same core logic)."
    )

    def _read_tif_slice(fp):
        arr = tifffile.imread(fp)
        if arr.ndim == 3:
            arr = arr[..., 0]
        return arr

    def _load_stack(surface_dir, stack_count):
        files = sorted(glob.glob(os.path.join(surface_dir, "*.tif")))
        if len(files) == 0:
            raise FileNotFoundError(f"No .tif files in {surface_dir}")
        files = files[:stack_count]
        stack = []
        for fp in files:
            sl = _read_tif_slice(fp).astype(np.float32)
            sl = cv2.filter2D(sl, -1, kernel)
            sl = cv2.normalize(sl, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            sl = cv2.equalizeHist(sl)
            sl = cv2.medianBlur(sl, 3)
            sl = sl.astype(np.float32) / 255.0
            stack.append(sl)
        return np.stack(stack, axis=0)  # (C,H,W)

    def _make_patches(stack, label, cube_size, max_patches=1500):
        C, H, W = stack.shape
        pad_h = (cube_size - (H % cube_size)) % cube_size
        pad_w = (cube_size - (W % cube_size)) % cube_size
        if pad_h or pad_w:
            stack = np.pad(stack, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")
            label = np.pad(label, ((0, pad_h), (0, pad_w)), mode="constant")
        _, H2, W2 = stack.shape

        xs, ys = [], []
        coords = [
            (i, j) for i in range(0, H2, cube_size) for j in range(0, W2, cube_size)
        ]
        random.shuffle(coords)
        for i, j in coords[:max_patches]:
            xs.append(stack[:, i : i + cube_size, j : j + cube_size])
            ys.append(label[i : i + cube_size, j : j + cube_size])

        X = torch.tensor(np.stack(xs, axis=0), dtype=torch.float32)
        Y = torch.tensor(np.stack(ys, axis=0), dtype=torch.float32).unsqueeze(1)
        return X, Y

    train_fragments = ["1", "2"]
    X_list, Y_list = [], []
    for fid in train_fragments:
        surf_dir = os.path.join(DATA_ROOT, "train", fid, "surface_volume")
        label_fp = os.path.join(DATA_ROOT, "train", fid, "inklabels.png")
        label = cv2.imread(label_fp, cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise FileNotFoundError(f"Missing label: {label_fp}")
        label = (label > 0).astype(np.float32)

        stack = _load_stack(surf_dir, stack_count)
        X, Y = _make_patches(stack, label, cube_size=cube_size, max_patches=2000)
        X_list.append(X)
        Y_list.append(Y)

    X_all = torch.cat(X_list, dim=0)
    Y_all = torch.cat(Y_list, dim=0)

    dataset = TensorDataset(X_all, Y_all)
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model_CNN.train()
    optimizer = optim.Adam(model_CNN.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    train_epochs = 2
    for ep in range(train_epochs):
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model_CNN(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"Epoch {ep+1}/{train_epochs} - loss: {float(np.mean(losses)):.6f}")
    model_CNN.eval()




## === cell 10
def process_volume_data(path, stack_count, cube_size, model_CNN, device, batch_size):
    files = sorted([f for f in os.listdir(path) if f.lower().endswith(".tif")])
    if len(files) == 0:
        raise FileNotFoundError(f"No tif files found in: {path}")

    first = tifffile.imread(os.path.join(path, files[0]))
    if first.ndim == 3:
        first = first[..., 0]
    H, W = first.shape[:2]
    stack_3 = torch.zeros((stack_count, H, W), dtype=torch.float32)

    for idx, filename in enumerate(
        tqdm(
            files[:stack_count],
            desc=f"Loading {os.path.basename(os.path.dirname(path))}",
        )
    ):
        slice_data = tifffile.imread(os.path.join(path, filename))
        if slice_data.ndim == 3:
            slice_data = slice_data[..., 0]
        slice_data = slice_data.astype(np.float32)

        slice_data = cv2.filter2D(slice_data, -1, kernel)
        slice_data = cv2.normalize(slice_data, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        slice_data = cv2.equalizeHist(slice_data)
        slice_data = cv2.medianBlur(slice_data, 3)
        slice_data = slice_data.astype(np.float32) / 255.0
        stack_3[idx, :, :] = torch.from_numpy(slice_data)

    pad_h = (cube_size - stack_3.shape[1] % cube_size) % cube_size
    pad_w = (cube_size - stack_3.shape[2] % cube_size) % cube_size
    image = F.pad(stack_3, (0, pad_w, 0, pad_h, 0, 0))  # (C,Hpad,Wpad)
    Hpad, Wpad = image.shape[1], image.shape[2]

    tiles = []
    for i0 in tqdm(range(0, Hpad, cube_size), desc="Tiling"):
        for j0 in range(0, Wpad, cube_size):
            block = image[:, i0 : i0 + cube_size, j0 : j0 + cube_size].unsqueeze(0)
            tiles.append(block)

    blocks_stack = torch.cat(tiles, dim=0)
    blocks_stack = torch.split(blocks_stack, batch_size, dim=0)

    out = torch.zeros((1, Hpad, Wpad), dtype=torch.float32)
    ncols = Wpad // cube_size

    model_CNN.eval()
    i = 0
    with torch.no_grad():
        for images in tqdm(blocks_stack, desc="Infer"):
            images = images.to(device).float()
            outputs = model_CNN(images)  # (B,1,h,w)
            outputs_cpu = outputs.detach().to("cpu")
            for cube in outputs_cpu:  # (1,h,w)
                row = i // ncols
                col = i % ncols
                y0 = row * cube_size
                x0 = col * cube_size
                out[:, y0 : y0 + cube.shape[1], x0 : x0 + cube.shape[2]] += cube
                i += 1

    return out[:, :H, :W]


def rle(img_2d_or_3d, threshold=0.5):
    if isinstance(img_2d_or_3d, torch.Tensor):
        img = img_2d_or_3d.detach().cpu().numpy()
    else:
        img = np.asarray(img_2d_or_3d)

    if img.ndim == 3:
        img = img[0]
    img = (img > float(threshold)).astype(np.uint8)

    if img.max() == 0:
        return ""

    pixels = img.flatten(order="F")  # column-major, as required
    pixels = np.concatenate([[0], pixels, [0]])
    changes = np.where(pixels[1:] != pixels[:-1])[0] + 1
    changes[1::2] -= changes[::2]
    return " ".join(map(str, changes))


test_root = os.path.join(DATA_ROOT, "test")
if not os.path.isdir(test_root):
    raise FileNotFoundError(f"DATA_ROOT seems wrong; missing test dir: {test_root}")

test_ids = []
for d in sorted(os.listdir(test_root)):
    frag_dir = os.path.join(test_root, d)
    if not os.path.isdir(frag_dir):
        continue
    if os.path.isdir(os.path.join(frag_dir, "surface_volume")):
        test_ids.append(d)

if len(test_ids) == 0:
    raise FileNotFoundError(f"No valid fragment dirs found under: {test_root}")

rows = []
for fid in test_ids:
    surf_path = os.path.join(test_root, fid, "surface_volume")
    pred = process_volume_data(
        surf_path, stack_count, cube_size, model_CNN, device, batch_size
    )

    mask_fp = os.path.join(test_root, fid, "mask.png")
    mask = cv2.imread(mask_fp, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Missing mask: {mask_fp}")
    mask = (mask > 0).astype(np.float32)  # (H,W)

    pred = pred * torch.from_numpy(mask).unsqueeze(0).to(
        dtype=pred.dtype, device=pred.device
    )

    pred_str = rle(pred, threshold=RLE_THRESHOLD)
    rows.append({"Id": fid, "Predicted": pred_str})
    gc.collect()

sub = pd.DataFrame(rows, columns=["Id", "Predicted"])
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with rows:", len(sub))
print(sub.head())
