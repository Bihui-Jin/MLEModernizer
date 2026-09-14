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

0.0969398873979281

# 6. Current score

0.15126

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.15008) has done: 'I fix the OpenCV crash by making histogram equalization work with the TIFF data type: `cv2.equalizeHist` only supports 8-bit single-channel images, so we safely normalize each slice to uint8 before equalizing, then continue the same filtering/normalization pipeline. I also add a small safety guard so if a fragment is missing or has no tif files, it yields an empty prediction string rather than crashing, ensuring a valid `submission.csv` is always written. These are execution/correctness fixes and keep the model/training/inference logic the same, only changing preprocessing to be valid in this environment. The script run end-to-end and write `submission.csv` to both the working directory and `./output/`.'
- What this solution (achieved 0.15126) has done: 'Your current score (0.15008) is higher than the target (0.09694), so to move closer we should slightly reduce performance rather than improve it. The most score-influential minimal knob here (without changing model/training core logic) is the final probability threshold used to binarize the mask before RLE; increasing it generally increase precision and reduce recall, often lowering overall F0.5 when you’re already overconfident/overpredicting. I add a tiny, deterministic threshold calibration step using the existing training fragments (no new model, no new loss), selecting a threshold that maximizes F0.5 on a small held-out set of patches, then apply that threshold at test-time. This keeps your architecture, training loop, preprocessing, and inference the same; it only replaces the fixed `0.6` with a data-driven threshold intended to shift the score downward toward the target.'

# 9. Code solution

## === cell 0
import os
import random
import glob
import gc
from pathlib import Path

import numpy as np
import pandas as pd

output_path = "./output/"
os.makedirs(output_path, exist_ok=True)



## === cell 1
import cv2
import tifffile

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm

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
def seed_everything(seed: int):
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
epochs = 1000
seed = 42
stack_count = 30
seed_everything(seed)
kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
block_size = cube_size

DATA_ROOT = "/kaggle/input/vesuvius-challenge-ink-detection"
if not os.path.exists(DATA_ROOT):
    DATA_ROOT = "/kaggle/input"  # safe fallback, but expected path above exists in Kaggle dataset

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
def _sorted_tif_paths(surface_volume_dir: str):
    paths = glob.glob(os.path.join(surface_volume_dir, "*.tif"))

    def _key(p):
        s = Path(p).stem
        try:
            return int(s)
        except Exception:
            return s

    return sorted(paths, key=_key)


def _to_uint8_for_equalize(img: np.ndarray) -> np.ndarray:
    if img.ndim != 2:
        raise ValueError(f"Expected single-channel 2D image, got shape {img.shape}")
    if img.dtype == np.uint8:
        return img
    img_f = img.astype(np.float32)
    mn = float(img_f.min())
    mx = float(img_f.max())
    if mx <= mn:
        return np.zeros_like(img_f, dtype=np.uint8)
    img_u8 = (255.0 * (img_f - mn) / (mx - mn)).clip(0, 255).astype(np.uint8)
    return img_u8


def load_stack_tensor(surface_volume_dir: str, stack_count: int):
    tif_paths = _sorted_tif_paths(surface_volume_dir)[:stack_count]
    if len(tif_paths) == 0:
        raise FileNotFoundError(f"No .tif found in {surface_volume_dir}")
    first = tifffile.imread(tif_paths[0])
    if first.ndim == 3:
        first = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    H, W = first.shape[:2]
    stack = np.zeros((stack_count, H, W), dtype=np.float32)
    for i, p in enumerate(tif_paths):
        img = tifffile.imread(p)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.filter2D(img, -1, kernel)

        img_u8 = _to_uint8_for_equalize(img)
        img_u8 = cv2.equalizeHist(img_u8)
        img_u8 = cv2.medianBlur(img_u8, 3)

        img_f = img_u8.astype(np.float32) / 255.0
        stack[i] = img_f
    return torch.from_numpy(stack)  # (C,H,W)


def extract_patches(
    stack_chw: torch.Tensor,
    label_hw: torch.Tensor,
    patch: int,
    max_patches: int,
    rng: np.random.RandomState,
):
    C, H, W = stack_chw.shape
    ys = rng.randint(0, H - patch + 1, size=max_patches)
    xs = rng.randint(0, W - patch + 1, size=max_patches)
    X = torch.zeros((max_patches, C, patch, patch), dtype=torch.float32)
    y = torch.zeros((max_patches, 1, patch, patch), dtype=torch.float32)
    for i, (yy, xx) in enumerate(zip(ys, xs)):
        X[i] = stack_chw[:, yy : yy + patch, xx : xx + patch]
        y[i, 0] = label_hw[yy : yy + patch, xx : xx + patch]
    return X, y


model_CNN = UNET(stack_count).to(device)

pretrained_path = "/kaggle/input/model-2/model_CNN_1000_epoch_30_stack.pth"
loaded = False
if os.path.exists(pretrained_path):
    state_dict = torch.load(pretrained_path, map_location="cpu")
    model_CNN.load_state_dict(state_dict)
    loaded = True

if not loaded:
    train_fragment_ids = sorted(
        [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
    )
    train_fragment_ids = [fid for fid in train_fragment_ids if fid.isdigit()]
    rng = np.random.RandomState(seed)

    X_list, y_list = [], []
    for fid in train_fragment_ids:
        frag_dir = os.path.join(TRAIN_DIR, fid)
        vol_dir = os.path.join(frag_dir, "surface_volume")
        label_path = os.path.join(frag_dir, "inklabels.png")
        if not (os.path.exists(vol_dir) and os.path.exists(label_path)):
            continue
        stack = load_stack_tensor(vol_dir, stack_count=stack_count)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label = (label > 0).astype(np.float32)
        label = torch.from_numpy(label)
        Xp, yp = extract_patches(
            stack, label, patch=cube_size, max_patches=256, rng=rng
        )
        X_list.append(Xp)
        y_list.append(yp)

    X_train = torch.cat(X_list, dim=0)
    y_train = torch.cat(y_list, dim=0)

    ds = torch.utils.data.TensorDataset(X_train, y_train)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False
    )

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model_CNN.parameters(), lr=learning_rate)

    model_CNN.train()
    train_epochs = min(epochs, 8)
    for ep in range(train_epochs):
        running = 0.0
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model_CNN(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
        _ = running / len(ds)

    model_CNN.eval()

gc.collect()




## === cell 11
def process_volume_data(path, stack_count, cube_size, model_CNN, device, batch_size):
    stack_3 = load_stack_tensor(path, stack_count=stack_count)  # (C,H,W)
    C, H, W = stack_3.shape

    pad_h = (cube_size - (H % cube_size)) % cube_size
    pad_w = (cube_size - (W % cube_size)) % cube_size

    image = F.pad(stack_3, (0, pad_w, 0, pad_h, 0, 0))  # pad W then H
    _, Hp, Wp = image.shape

    blocks = []
    coords = []
    for y in range(0, Hp, cube_size):
        for x in range(0, Wp, cube_size):
            block = image[:, y : y + cube_size, x : x + cube_size].unsqueeze(0)
            blocks.append(block)
            coords.append((y, x))

    blocks_stack = torch.cat(blocks, dim=0)  # (N,C,ps,ps)

    out_full = torch.zeros((1, Hp, Wp), dtype=torch.float32)

    model_CNN.eval()
    with torch.no_grad():
        for start in range(0, blocks_stack.shape[0], batch_size):
            end = min(start + batch_size, blocks_stack.shape[0])
            imgs = blocks_stack[start:end].to(device).float()
            outs = model_CNN(imgs).to("cpu")  # (B,1,ps,ps)
            for b in range(outs.shape[0]):
                y, x = coords[start + b]
                out_full[:, y : y + cube_size, x : x + cube_size] = outs[b]

    out_full = out_full[:, :H, :W]
    return out_full  # (1,H,W)




## === cell 12
def rle_encode(mask: np.ndarray) -> str:
    pixels = mask.flatten(order="C")  # left-to-right, then top-to-bottom
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def fbeta_from_counts(tp: int, fp: int, fn: int, beta: float = 0.5) -> float:
    if tp == 0:
        return 0.0
    b2 = beta * beta
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    denom = b2 * p + r
    if denom <= 0:
        return 0.0
    return (1.0 + b2) * p * r / denom


def calibrate_threshold_on_train(
    model_CNN,
    device,
    train_dir: str,
    stack_count: int,
    cube_size: int,
    seed: int,
    max_frags: int = 2,
    patches_per_frag: int = 128,
):
    frag_ids = sorted(
        [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    )
    frag_ids = [fid for fid in frag_ids if fid.isdigit()][:max_frags]
    rng = np.random.RandomState(seed + 123)

    probs = []
    gts = []

    model_CNN.eval()
    with torch.no_grad():
        for fid in frag_ids:
            frag_dir = os.path.join(train_dir, fid)
            vol_dir = os.path.join(frag_dir, "surface_volume")
            label_path = os.path.join(frag_dir, "inklabels.png")
            if not (os.path.exists(vol_dir) and os.path.exists(label_path)):
                continue

            stack = load_stack_tensor(vol_dir, stack_count=stack_count)  # (C,H,W)
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            label = (label > 0).astype(np.uint8)

            H, W = label.shape
            if H < cube_size or W < cube_size:
                continue

            ys = rng.randint(0, H - cube_size + 1, size=patches_per_frag)
            xs = rng.randint(0, W - cube_size + 1, size=patches_per_frag)

            for yy, xx in zip(ys, xs):
                xb = (
                    stack[:, yy : yy + cube_size, xx : xx + cube_size]
                    .unsqueeze(0)
                    .to(device)
                    .float()
                )
                pred = (
                    model_CNN(xb).squeeze(0).squeeze(0).detach().to("cpu").numpy()
                )  # (ps,ps)
                gt = label[yy : yy + cube_size, xx : xx + cube_size]  # (ps,ps)

                probs.append(pred.reshape(-1))
                gts.append(gt.reshape(-1))

    if len(probs) == 0:
        return 0.6  # fallback to original behavior

    probs = np.concatenate(probs).astype(np.float32)
    gts = np.concatenate(gts).astype(np.uint8)

    thr_grid = np.linspace(0.55, 0.90, 15, dtype=np.float32)

    best_thr = 0.6
    best_score = -1.0
    for thr in thr_grid:
        pred = (probs > thr).astype(np.uint8)
        tp = int(((pred == 1) & (gts == 1)).sum())
        fp = int(((pred == 1) & (gts == 0)).sum())
        fn = int(((pred == 0) & (gts == 1)).sum())
        s = fbeta_from_counts(tp, fp, fn, beta=0.5)

        if (s > best_score + 1e-12) or (
            abs(s - best_score) <= 1e-12 and thr > best_thr
        ):
            best_score = s
            best_thr = float(thr)

    return best_thr


sample_df = pd.read_csv(SAMPLE_SUB_PATH)
test_ids = sample_df["Id"].tolist()

cal_thr = calibrate_threshold_on_train(
    model_CNN=model_CNN,
    device=device,
    train_dir=TRAIN_DIR,
    stack_count=stack_count,
    cube_size=cube_size,
    seed=seed,
    max_frags=2,
    patches_per_frag=128,
)
print(f"[INFO] Using calibrated threshold: {cal_thr:.4f} (was fixed 0.6)")

pred_strings = []
for fid in test_ids:
    frag_dir = os.path.join(TEST_DIR, str(fid))
    vol_dir = os.path.join(frag_dir, "surface_volume")
    mask_path = os.path.join(frag_dir, "mask.png")

    try:
        pred_prob = process_volume_data(
            vol_dir, stack_count, cube_size, model_CNN, device, batch_size
        )  # (1,H,W)
        pred_prob = pred_prob.squeeze(0).numpy()
    except Exception as e:
        print(f"[WARN] Failed on fragment {fid} at {vol_dir}: {repr(e)}")
        pred_strings.append("")
        continue

    if os.path.exists(mask_path):
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        m = (m > 0).astype(np.uint8)
    else:
        m = np.ones_like(pred_prob, dtype=np.uint8)

    pred_bin = ((pred_prob > cal_thr).astype(np.uint8) * m).astype(np.uint8)
    pred_strings.append(rle_encode(pred_bin))

sub = pd.DataFrame({"Id": test_ids, "Predicted": pred_strings})
sub.to_csv("submission.csv", index=False)
sub.to_csv(os.path.join(output_path, "submission.csv"), index=False)
print(sub.head())
print("Wrote submission.csv")
