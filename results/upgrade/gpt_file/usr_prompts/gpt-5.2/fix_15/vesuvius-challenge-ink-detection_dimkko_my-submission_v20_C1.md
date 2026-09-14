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

0.0690302087378954

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.14243) has done: 'I fix the two hard failures preventing an end-to-end run: the missing pretrained model path and the bogus attempt to read a non-existent submission from another dataset. To keep the core model logic intact while ensuring a valid `submission.csv`, the notebook (1) try to load the pretrained weights if present, otherwise proceed with the randomly initialized UNET (still producing a valid file), and (2) actually run inference on the real `test/` fragments and write the required RLE format. I also fix the TIFF/PNG reading in `process_volume_data` (it currently uses `cv2.imread` on `.tif` and normalizes incorrectly), and add a correct binary threshold + RLE encoder so the submission matches the competition format. These changes are execution- and format-critical; they don’t change the architecture or training loop (there is no training here).'

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

        self.proj = None
        if in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = x if self.proj is None else self.proj(x)
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
        self.rnn = nn.Identity()

    def forward(self, x):
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        inp = h
        for i in range(self.t):
            h = self.relu(self.conv3(h) + inp)
            inp = h
        h = self.rnn(h)
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
model_CNN = UNET(stack_count).to(device).float()

candidate_weight_paths = [
    "/kaggle/input/model-2/model_CNN_1000_epoch_30_stack.pth",  # original external dataset path
    "/kaggle/input/model-2/model_CNN_1000_epoch_30_stack.pt",
    "/kaggle/input/model-2/model.pth",
    "/kaggle/input/vesuvius-challenge-ink-detection/model_CNN_1000_epoch_30_stack.pth",
    "/kaggle/input/vesuvius-challenge-ink-detection/model_CNN_1000_epoch_30_stack.pt",
    "/kaggle/data/vesuvius-challenge-ink-detection/model_CNN_1000_epoch_30_stack.pth",
    "/kaggle/data/vesuvius-challenge-ink-detection/model_CNN_1000_epoch_30_stack.pt",
]

weight_path = next((p for p in candidate_weight_paths if os.path.exists(p)), None)

HAS_PRETRAINED = weight_path is not None

if HAS_PRETRAINED:
    state_dict = torch.load(weight_path, map_location="cpu")
    model_CNN.load_state_dict(state_dict)
    print(f"Loaded weights: {weight_path}")
else:
    print(
        "WARNING: pretrained weights not found; will use safe empty-mask fallback at inference to avoid massive false positives under F0.5."
    )

model_CNN.eval()
gc.collect()




## === cell 11
def _read_grayscale_any(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".tif", ".tiff"]:
        img = tifffile.imread(path)
        if img.ndim == 3:
            img = img[..., 0]
        return img
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img


def _numeric_slice_sort_key(fp: str) -> int:
    stem = os.path.splitext(os.path.basename(fp))[0]
    try:
        return int(stem)
    except Exception:
        digits = "".join([c for c in stem if c.isdigit()])
        if digits == "":
            return 10**9
        return int(digits)


def process_volume_data(
    path, stack_count, cube_size, model_CNN, device, batch_size, data_mask_np=None
):
    files = glob.glob(os.path.join(path, "*.tif"))
    if len(files) == 0:
        files = glob.glob(os.path.join(path, "*.*"))
    files = sorted(files, key=_numeric_slice_sort_key)

    if len(files) == 0:
        raise FileNotFoundError(f"No slice files found in {path}")

    first = _read_grayscale_any(files[0])
    if first is None:
        raise FileNotFoundError(f"Failed to read first slice: {files[0]}")
    H, W = first.shape[:2]

    stack_3 = torch.zeros((stack_count, H, W), dtype=torch.float32)

    use_n = min(len(files), stack_count)
    last_slice_tensor = None

    for idx in range(use_n):
        fp = files[idx]
        slice_data = _read_grayscale_any(fp)
        if slice_data is None:
            raise FileNotFoundError(f"Failed to read slice: {fp}")
        slice_data = slice_data.astype(np.float32)

        slice_data = cv2.filter2D(slice_data, -1, kernel)
        slice_data = cv2.normalize(slice_data, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        slice_data = cv2.equalizeHist(slice_data)
        slice_data = cv2.medianBlur(slice_data, 3)
        slice_data = slice_data.astype(np.float32) / 255.0

        if data_mask_np is not None:
            slice_data = slice_data * data_mask_np

        last_slice_tensor = torch.from_numpy(slice_data)
        stack_3[idx, :, :] = last_slice_tensor

    if use_n < stack_count:
        if last_slice_tensor is None:
            last_slice_tensor = torch.zeros((H, W), dtype=torch.float32)
        for idx in range(use_n, stack_count):
            stack_3[idx, :, :] = last_slice_tensor

    pad_h = (cube_size - (H % cube_size)) % cube_size
    pad_w = (cube_size - (W % cube_size)) % cube_size

    image = F.pad(stack_3, (0, pad_w, 0, pad_h))  # (C, H, W)
    padded_H, padded_W = image.shape[1], image.shape[2]

    blocks = []
    for i in range(0, padded_H, cube_size):
        for j in range(0, padded_W, cube_size):
            block = image[:, i : i + cube_size, j : j + cube_size].unsqueeze(0)
            blocks.append(block)

    blocks_stack = torch.cat(blocks, dim=0)
    blocks_stack = torch.split(blocks_stack, batch_size, dim=0)

    out_full = torch.zeros((1, padded_H, padded_W), dtype=torch.float32)
    patch_idx = 0
    patches_per_row = padded_W // cube_size

    with torch.no_grad():
        for images in blocks_stack:
            images = images.to(device).float()
            outputs = model_CNN(images)  # (B, 1, h, w)

            for cube in outputs.to("cpu"):
                row = patch_idx // patches_per_row
                col = patch_idx % patches_per_row
                y_start = row * cube_size
                x_start = col * cube_size
                out_full[
                    :,
                    y_start : y_start + cube.shape[1],
                    x_start : x_start + cube.shape[2],
                ] = cube[0]
                patch_idx += 1

    out_full = out_full[:, :H, :W]
    return out_full




## === cell 12
DATA_ROOT = "/kaggle/input/vesuvius-challenge-ink-detection"
if not os.path.exists(DATA_ROOT):
    alt = "/kaggle/data/vesuvius-challenge-ink-detection"
    if os.path.exists(alt):
        DATA_ROOT = alt

TEST_DIR = os.path.join(DATA_ROOT, "test")
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)


def _is_valid_fragment_dir(base_dir: str, frag_id: str) -> bool:
    frag_dir = os.path.join(base_dir, frag_id)
    if not os.path.isdir(frag_dir):
        return False
    if not os.path.isdir(os.path.join(frag_dir, "surface_volume")):
        return False
    if not os.path.isfile(os.path.join(frag_dir, "mask.png")):
        return False
    return True


test_ids = []
if os.path.isdir(TEST_DIR):
    for d in sorted(os.listdir(TEST_DIR)):
        if _is_valid_fragment_dir(TEST_DIR, d):
            test_ids.append(d)


def rle_encode(mask: np.ndarray) -> str:
    if mask is None:
        return ""
    mask = mask.astype(np.uint8)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")

    pixels = mask.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    changes = np.where(pixels[1:] != pixels[:-1])[0] + 1
    if changes.size == 0:
        return ""
    runs = changes.copy()
    runs[1::2] = runs[1::2] - runs[::2]
    return " ".join(map(str, runs.tolist()))


def load_mask_png(fragment_dir):
    mp = os.path.join(fragment_dir, "mask.png")
    m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Missing mask.png at {mp}")
    return (m > 0).astype(np.uint8)


def load_inklabels_png(fragment_dir):
    lp = os.path.join(fragment_dir, "inklabels.png")
    m = cv2.imread(lp, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Missing inklabels.png at {lp}")
    return (m > 0).astype(np.uint8)


def fbeta_score_binary(
    y_true: np.ndarray, y_pred: np.ndarray, beta: float = 0.5
) -> float:
    y_true = y_true.astype(np.uint8).ravel()
    y_pred = y_pred.astype(np.uint8).ravel()
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    if tp == 0:
        return 0.0
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    b2 = beta * beta
    denom = b2 * p + r
    return float(((1 + b2) * p * r / denom) if denom > 0 else 0.0)


def calibrate_threshold(train_ids, candidate_thresholds):
    scores = {t: [] for t in candidate_thresholds}
    used = 0
    for fid in train_ids:
        frag_dir = os.path.join(TRAIN_DIR, str(fid))
        vol_dir = os.path.join(frag_dir, "surface_volume")
        if not (os.path.isdir(frag_dir) and os.path.isdir(vol_dir)):
            continue
        try:
            data_mask = load_mask_png(frag_dir).astype(np.uint8)

            pred = (
                process_volume_data(
                    vol_dir,
                    stack_count,
                    cube_size,
                    model_CNN,
                    device,
                    batch_size,
                    data_mask_np=data_mask.astype(np.float32),
                )
                .squeeze(0)
                .numpy()
            )

            y_true = load_inklabels_png(frag_dir).astype(np.uint8)

            valid = data_mask.astype(bool)
            y_true_v = y_true[valid].astype(np.uint8)
            pred_v = pred[valid].astype(np.float32)

            for t in candidate_thresholds:
                y_pred_v = (pred_v >= t).astype(np.uint8)
                scores[t].append(fbeta_score_binary(y_true_v, y_pred_v, beta=0.5))
            used += 1
        except Exception as e:
            print(f"Threshold calibration skipped fragment {fid} due to: {e}")
            continue

    if used == 0:
        return None, None

    mean_scores = {t: float(np.mean(v)) if len(v) else 0.0 for t, v in scores.items()}
    best_t = max(mean_scores, key=mean_scores.get)
    return best_t, mean_scores


def remove_small_components(
    bin_mask: np.ndarray, min_area: int, valid_mask: np.ndarray | None = None
) -> np.ndarray:
    if min_area <= 1:
        return bin_mask
    m = bin_mask.astype(np.uint8)
    if valid_mask is not None:
        m = (m * valid_mask.astype(np.uint8)).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num_labels <= 1:
        return m
    keep = np.zeros_like(m, dtype=np.uint8)
    for lab in range(1, num_labels):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area >= min_area:
            keep[labels == lab] = 1

    if valid_mask is not None:
        keep = (keep * valid_mask.astype(np.uint8)).astype(np.uint8)
    return keep


train_ids = []
if os.path.isdir(TRAIN_DIR):
    for d in sorted(os.listdir(TRAIN_DIR)):
        if d.isdigit():
            train_ids.append(d)

train_ids_for_calib = train_ids[:2]

candidate_thresholds = [0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97]

best_thresh, thresh_scores = (None, None)
if HAS_PRETRAINED:
    best_thresh, thresh_scores = calibrate_threshold(
        train_ids_for_calib, candidate_thresholds
    )

THRESH = 0.94
if best_thresh is not None:
    THRESH = float(best_thresh)
    print("Calibrated THRESH =", THRESH, "scores =", thresh_scores)
else:
    print("Using default THRESH =", THRESH, "(calibration unavailable or disabled)")

MIN_COMPONENT_AREA = 20

want_ids = test_ids if len(test_ids) > 0 else sample_sub["Id"].astype(str).tolist()
have_pred = {}

for fid in want_ids:
    frag_dir = os.path.join(TEST_DIR, str(fid))
    vol_dir = os.path.join(frag_dir, "surface_volume")
    try:
        if not (
            os.path.isdir(vol_dir)
            and os.path.isfile(os.path.join(frag_dir, "mask.png"))
        ):
            have_pred[str(fid)] = ""
            continue

        data_mask = load_mask_png(frag_dir).astype(np.uint8)

        if not HAS_PRETRAINED:
            bin_mask = np.zeros_like(data_mask, dtype=np.uint8)
            have_pred[str(fid)] = rle_encode(bin_mask)
            continue

        pred = process_volume_data(
            vol_dir,
            stack_count,
            cube_size,
            model_CNN,
            device,
            batch_size,
            data_mask_np=data_mask.astype(np.float32),
        )  # (1,H,W)
        pred = pred.squeeze(0).numpy()

        pred = pred * data_mask.astype(pred.dtype)

        bin_mask = (pred >= THRESH).astype(np.uint8)
        bin_mask = (bin_mask * data_mask).astype(np.uint8)
        bin_mask = remove_small_components(
            bin_mask, min_area=MIN_COMPONENT_AREA, valid_mask=data_mask
        )
        bin_mask = (bin_mask * data_mask).astype(np.uint8)

        have_pred[str(fid)] = rle_encode(bin_mask)
    except Exception as e:
        print(f"Inference failed for test Id={fid} due to: {e}")
        have_pred[str(fid)] = ""

sub_df = pd.DataFrame(
    [{"Id": str(i), "Predicted": have_pred.get(str(i), "")} for i in want_ids],
    columns=["Id", "Predicted"],
)
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)



## === cell 13
sub_df.head()
