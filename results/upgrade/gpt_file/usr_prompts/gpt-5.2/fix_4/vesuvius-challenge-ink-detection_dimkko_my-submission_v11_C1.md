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
    random.seed(seed)  # фиксируем генератор случайных чисел
    os.environ["PYTHONHASHSEED"] = str(seed)  # фиксируем заполнения хешей
    np.random.seed(seed)  # фиксируем генератор случайных чисел numpy
    torch.manual_seed(seed)  # фиксируем генератор случайных чисел pytorch
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # фиксируем генератор случайных чисел для GPU
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = (
        True  # выбираем только детерминированные алгоритмы (для сверток)
    )
    torch.backends.cudnn.benchmark = False  # фиксируем алгоритм вычисления сверток




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
        x1 = x.clone()  # сохраним для skip-связи

        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x2 = x.clone()  # сохраним для skip-связи

        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x3 = x.clone()  # сохраним для skip-связи

        x = self.pool3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x4 = x.clone()  # сохраним для skip-связи

        x = self.pool4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)

        x = self.upconv6(x)
        x = torch.cat([x4, x], dim=1)  # skip-связь
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)

        x = self.upconv7(x)
        x = torch.cat([x3, x], dim=1)  # skip-связь
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu7(x)

        x = self.upconv8(x)
        x = torch.cat([x2, x], dim=1)  # skip-связь
        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu8(x)

        x = self.upconv9(x)
        x = torch.cat([x1, x], dim=1)  # skip-связь
        x = self.conv9(x)
        x = self.bn9(x)
        x = self.relu9(x)

        x = self.conv10(x)
        x = self.sigmoid(x)

        return x




## === cell 10
DATA_ROOT_CANDIDATES = [
    "/kaggle/input/vesuvius-challenge-ink-detection",
    "/kaggle/data/vesuvius-challenge-ink-detection",
    "/kaggle/input",
    "/kaggle/data",
]


def find_data_root():
    for root in DATA_ROOT_CANDIDATES:
        if os.path.isdir(os.path.join(root, "train")) and os.path.isdir(
            os.path.join(root, "test")
        ):
            return root
    for root in DATA_ROOT_CANDIDATES:
        if os.path.isdir(root):
            for dirpath, dirnames, _ in os.walk(root):
                if "train" in dirnames and "test" in dirnames:
                    return dirpath
    raise FileNotFoundError(
        "Could not locate competition data root containing train/ and test/."
    )


DATA_ROOT = find_data_root()
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")

model_CNN = UNET(stack_count).to(device)

checkpoint_path = "/kaggle/input/model-2/model_CNN_1000_epoch_30_stack.pth"
loaded = False
if os.path.exists(checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location=device)
    model_CNN.load_state_dict(state_dict)
    loaded = True

gc.collect()
loaded, DATA_ROOT



## === cell 11


def _read_tif_gray(path):
    arr = tifffile.imread(path)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) * (1.0 / 65535.0)
    arr = arr.astype(np.float32, copy=False)
    maxv = float(arr.max()) if arr.size else 1.0
    if maxv > 1.5:
        arr /= 65535.0
    elif maxv != 0:
        arr /= maxv
    return arr


def _preprocess_slice_to_float01(path):
    slice_data = _read_tif_gray(path)  # float32
    s8 = np.clip(slice_data * 255.0, 0, 255).astype(np.uint8, copy=False)
    s8 = cv2.filter2D(s8, -1, kernel)
    s8 = cv2.equalizeHist(s8)
    s8 = cv2.medianBlur(s8, 3)
    return s8.astype(np.float32) * (1.0 / 255.0)


def process_volume_data(path, stack_count, cube_size, model_CNN, device, batch_size):
    file_list = sorted(glob.glob(os.path.join(path, "*.tif")))
    if len(file_list) == 0:
        raise FileNotFoundError(f"No .tif files found in: {path}")
    file_list = file_list[:stack_count]

    max_workers = min(8, (os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        vol_slices = list(ex.map(_preprocess_slice_to_float01, file_list))

    vol = np.stack(vol_slices, axis=0)  # [C,H,W] float32
    stack_3 = torch.from_numpy(vol)  # CPU float32

    pad_h = (cube_size - stack_3.shape[1] % cube_size) % cube_size
    pad_w = (cube_size - stack_3.shape[2] % cube_size) % cube_size

    image = F.pad(stack_3, (0, pad_w, 0, pad_h, 0, 0))  # [C,Hp,Wp]
    C, Hp, Wp = image.shape

    tiles = (
        image.unfold(1, cube_size, cube_size)
        .unfold(2, cube_size, cube_size)
        .permute(1, 2, 0, 3, 4)
        .contiguous()
        .view(-1, C, cube_size, cube_size)
    )  # [N,C,cs,cs] on CPU

    if torch.cuda.is_available():
        tiles = tiles.pin_memory()

    out_tiles = torch.empty(
        (tiles.shape[0], 1, cube_size, cube_size), dtype=torch.float32, device="cpu"
    )

    model_CNN.eval()
    use_cuda = device == "cuda"
    stream = torch.cuda.Stream() if use_cuda else None

    with torch.inference_mode():
        for start in range(0, tiles.shape[0], batch_size):
            end = min(start + batch_size, tiles.shape[0])
            batch_cpu = tiles[start:end]

            if use_cuda:
                with torch.cuda.stream(stream):
                    batch = batch_cpu.to(device, non_blocking=True)
                torch.cuda.current_stream().wait_stream(stream)
                out = model_CNN(batch)
                out_tiles[start:end].copy_(out.detach().to("cpu", non_blocking=False))
            else:
                out = model_CNN(batch_cpu)
                out_tiles[start:end].copy_(out)

    ny = Hp // cube_size
    nx = Wp // cube_size
    out_full = (
        out_tiles.view(ny, nx, 1, cube_size, cube_size)
        .permute(2, 0, 3, 1, 4)
        .contiguous()
        .view(1, Hp, Wp)
    )

    return out_full[:, : stack_3.shape[1], : stack_3.shape[2]]


def load_mask_png(path):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return (m > 0).astype(np.uint8)


def load_inklabels_png(path):
    y = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if y is None:
        raise FileNotFoundError(path)
    return (y > 0).astype(np.uint8)


def rle_encode(mask):
    pixels = mask.flatten(order="C")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


class PatchDataset(Dataset):
    def __init__(self, volume, labels, valid_mask, cube_size):
        self.volume = volume  # torch [C,H,W]
        self.labels = labels  # torch [1,H,W]
        self.valid_mask = valid_mask  # torch [1,H,W]
        self.cube = cube_size

        C, H, W = self.volume.shape
        self.H = H
        self.W = W

        self.coords = []
        step = cube_size  # keep original semantics close to inference tiling
        for y in range(0, H - cube_size + 1, step):
            for x in range(0, W - cube_size + 1, step):
                vm = self.valid_mask[0, y : y + cube_size, x : x + cube_size]
                if vm.mean().item() > 0.1:
                    self.coords.append((y, x))

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        y, x = self.coords[idx]
        X = self.volume[:, y : y + self.cube, x : x + self.cube]
        Y = self.labels[:, y : y + self.cube, x : x + self.cube]
        return X, Y


def build_train_data(fragment_id):
    frag_dir = os.path.join(TRAIN_DIR, str(fragment_id))
    vol_dir = os.path.join(frag_dir, "surface_volume")
    mask_path = os.path.join(frag_dir, "mask.png")
    label_path = os.path.join(frag_dir, "inklabels.png")

    mask2d = load_mask_png(mask_path)
    label2d = load_inklabels_png(label_path)

    file_list = sorted(glob.glob(os.path.join(vol_dir, "*.tif")))[:stack_count]
    if len(file_list) < stack_count:
        raise RuntimeError(f"Not enough slices in {vol_dir}: {len(file_list)}")

    max_workers = min(8, (os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        vol_slices = list(ex.map(_preprocess_slice_to_float01, file_list))
    vol = np.stack(vol_slices, axis=0)  # [C,H,W]

    volume = torch.from_numpy(vol).float()
    labels = torch.from_numpy(label2d[None, ...].astype(np.float32))
    valid = torch.from_numpy(mask2d[None, ...].astype(np.float32))
    return volume, labels, valid


def train_if_needed(model):
    if loaded:
        return model

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    train_entries = [
        d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))
    ]
    volumes = [int(d) for d in train_entries if d.isdigit()]
    if len(volumes) == 0:
        raise RuntimeError("No training fragments found.")

    datasets = []
    for fid in sorted(volumes):
        vol, lab, msk = build_train_data(fid)
        datasets.append(PatchDataset(vol, lab, msk, cube_size))
    full_ds = data.ConcatDataset(datasets)

    num_workers = min(4, (os.cpu_count() or 2))
    loader = DataLoader(
        full_ds,
        batch_size=16,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    train_epochs = max(3, min(15, epochs))
    for ep in range(train_epochs):
        for X, Y in loader:
            X = X.to(device, non_blocking=True)
            Y = Y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(X)
            loss = criterion(pred, Y)
            loss.backward()
            optimizer.step()

    return model


model_CNN = train_if_needed(model_CNN)

test_ids = sorted(
    [d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))]
)

rows = []
for tid in test_ids:
    frag_dir = os.path.join(TEST_DIR, tid)
    vol_dir = os.path.join(frag_dir, "surface_volume")
    mask_path = os.path.join(frag_dir, "mask.png")

    pred_map = process_volume_data(
        vol_dir, stack_count, cube_size, model_CNN, device, batch_size
    )  # [1,H,W]
    pred_map = pred_map.squeeze(0).numpy()

    mask2d = load_mask_png(mask_path)
    pred_map = pred_map * mask2d

    bin_mask = (pred_map > 0.5).astype(np.uint8)
    rle = rle_encode(bin_mask)
    rows.append({"Id": tid, "Predicted": rle})

sub = pd.DataFrame(rows, columns=["Id", "Predicted"])
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with", len(sub), "rows")
