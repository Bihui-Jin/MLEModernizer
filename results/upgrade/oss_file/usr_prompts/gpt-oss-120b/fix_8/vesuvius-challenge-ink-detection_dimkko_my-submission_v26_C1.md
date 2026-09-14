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
import os
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
import gc
from torch.utils.data import TensorDataset, SubsetRandomSampler
from torchvision.transforms import Normalize
import tifffile
import pandas as pd

gc.collect()




## === cell 1
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed




## === cell 2
device = "cuda" if torch.cuda.is_available() else "cpu"
device




## === cell 3
def seed_everything(seed):
    random.seed(seed)  # фиксируем генератор случайных чисел
    os.environ["PYTHONHASHSEED"] = str(seed)  # фиксируем заполнения хешей
    np.random.seed(seed)  # фиксируем генератор случайных чисел numpy
    torch.manual_seed(seed)  # фиксируем генератор случайных чисел pytorch
    torch.cuda.manual_seed(seed)  # фиксируем генератор случайных чисел для GPU
    torch.backends.cudnn.deterministic = (
        True  # выбираем только детерминированные алгоритмы (для сверток)
    )
    torch.backends.cudnn.benchmark = False  # фиксируем алгоритм вычисления сверток


learning_rate = 0.0001
batch_size = 128  # increased batch size to reduce loop overhead
cube_size = 128  # нарезка изображений на cube_size х cube_size
epochs = 1000
seed = 42
stack_count = 32  # количество срезов для модели
seed_everything(seed)
kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
block_size = cube_size




## === cell 4
class UNET2(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(1024)

        self.conv6 = nn.Conv2d(1024, 2048, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(2048)

        self.conv8 = nn.Conv2d(2048, 4096, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(4096)

        self.upconv10 = nn.ConvTranspose2d(4096, 2048, kernel_size=2, stride=2)
        self.conv10 = nn.Conv2d(4096, 2048, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm2d(2048)

        self.upconv11 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.conv11 = nn.Conv2d(2048, 1024, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(1024)

        self.upconv12 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv12 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(512)

        self.upconv13 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv13 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn13 = nn.BatchNorm2d(256)

        self.upconv14 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv14 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn14 = nn.BatchNorm2d(128)

        self.upconv15 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv15 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn15 = nn.BatchNorm2d(64)

        self.conv16 = nn.Conv2d(64, 1, kernel_size=1)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x1 = x.clone()
        x = self.pool(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x2 = x.clone()
        x = self.pool(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x3 = x.clone()
        x = self.pool(x)
        x3_3 = x.clone()

        x = self.relu(self.bn4(self.conv4(x)))
        x4 = x.clone()
        x = self.pool(x)

        x = self.relu(self.bn5(self.conv5(x)))
        x5 = x.clone()
        x = self.pool(x)

        x = self.relu(self.bn6(self.conv6(x)))
        x6 = x.clone()
        x = self.pool(x)

        x = self.relu(self.bn8(self.conv8(x)))

        x = self.upconv10(x)
        x = torch.cat([x6, x], dim=1)
        x = self.conv10(x)
        x = self.bn10(x)
        x = self.relu(x)

        x = self.upconv11(x)
        x = torch.cat([x5, x], dim=1)
        x = self.conv11(x)
        x = self.bn11(x)
        x = self.relu(x)

        x = self.upconv12(x)
        x = torch.cat([x4, x], dim=1)
        x = self.conv12(x)
        x = self.bn12(x)
        x = self.relu(x)

        x = self.upconv13(x)
        x = torch.cat([x3, x], dim=1)
        x = self.conv13(x)
        x = self.bn13(x)
        x = self.relu(x)

        x = self.upconv14(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv14(x)
        x = self.bn14(x)
        x = self.relu(x)

        x = self.upconv15(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv15(x)
        x = self.bn15(x)
        x = self.relu(x)

        x = self.conv16(x)
        x = self.sigmoid(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()




## === cell 5
try:
    state_dict = torch.load(
        "/kaggle/input/model-2/model_Unet_stack_3_2D.pth", map_location=device
    )
    model_CNN = UNET2(stack_count)
    model_CNN.load_state_dict(state_dict)
except FileNotFoundError:
    print("Pretrained model not found – using randomly initialized UNet.")
    model_CNN = UNET2(stack_count)
    model_CNN._init_weight()
model_CNN.to(device)
model_CNN.eval()
gc.collect()




## === cell 6
def _load_slice(slice_path):
    """Fast loading of a single TIFF slice using tifffile."""
    try:
        img = tifffile.imread(slice_path)
        if img.ndim == 3:  # some TIFFs may have a channel axis
            img = img[..., 0]
        img = img.astype(np.float32) / 255.0
    except Exception:
        img = np.zeros((0, 0), dtype=np.float32)
    return img


def process_volume_data(path, stack_count, cube_size, model_CNN, device, batch_size):
    """
    Process a volume located at `path`. Returns a prediction mask (torch tensor).
    """
    if not os.path.isdir(path) or len(os.listdir(path)) == 0:
        return torch.zeros((1, cube_size, cube_size), dtype=torch.float32)

    first_file = sorted(os.listdir(path))[0]
    sample_img = tifffile.imread(os.path.join(path, first_file))
    if sample_img is None:
        return torch.zeros((1, cube_size, cube_size), dtype=torch.float32)
    if sample_img.ndim == 3:
        sample_img = sample_img[..., 0]
    orig_h, orig_w = sample_img.shape

    stack_np = np.empty((stack_count, orig_h, orig_w), dtype=np.float32)

    slice_files = sorted(os.listdir(path))[:stack_count]
    slice_paths = [os.path.join(path, f) for f in slice_files]

    loaded_slices = Parallel(n_jobs=4, backend="threading")(
        delayed(_load_slice)(p) for p in slice_paths
    )
    for idx, arr in enumerate(loaded_slices):
        stack_np[idx] = arr

    pad_h = (cube_size - orig_h % cube_size) % cube_size
    pad_w = (cube_size - orig_w % cube_size) % cube_size
    if pad_h > 0 or pad_w > 0:
        stack_np = np.pad(
            stack_np,
            ((0, 0), (0, pad_h), (0, pad_w)),
            mode="constant",
            constant_values=0,
        )

    padded_h = orig_h + pad_h
    padded_w = orig_w + pad_w

    stack_tensor = torch.from_numpy(stack_np)  # shape (C, H, W)

    out_volume = torch.zeros((padded_h, padded_w), dtype=torch.float32)

    v_blocks = padded_h // cube_size
    h_blocks = padded_w // cube_size

    model_CNN.eval()
    with torch.no_grad():
        batch_blocks = []
        coords = []  # list of (y_start, x_start) for each block in the batch
        for vi in range(v_blocks):
            y_start = vi * cube_size
            for hi in range(h_blocks):
                x_start = hi * cube_size
                block = stack_tensor[
                    :, y_start : y_start + cube_size, x_start : x_start + cube_size
                ]
                batch_blocks.append(block.unsqueeze(0))  # (1, C, H, W)
                coords.append((y_start, x_start))

                if len(batch_blocks) == batch_size or (
                    vi == v_blocks - 1 and hi == h_blocks - 1
                ):
                    batch_tensor = torch.cat(batch_blocks).to(device)  # (B, C, H, W)
                    outputs = model_CNN(batch_tensor)  # (B, 1, H, W)
                    outputs = outputs.squeeze(1).cpu()  # (B, H, W)

                    for (y, x), out_cube in zip(coords, outputs):
                        out_volume[y : y + cube_size, x : x + cube_size] += out_cube

                    batch_blocks.clear()
                    coords.clear()

    out_volume = out_volume[:orig_h, :orig_w]
    return out_volume.unsqueeze(0)  # shape (1, H, W)


def rle(img):
    """
    Run‑length encode a 2‑D binary mask (numpy array).
    Pixels are ordered row‑major, 1‑indexed as required by the competition.
    """
    mask = (img >= 0.5).astype(np.uint8)
    padded = np.pad(mask.flatten(order="C"), (1, 1), mode="constant", constant_values=0)
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    lengths = ends - starts
    return starts.astype(np.uint32), lengths.astype(np.uint32)


base_input = "/kaggle/input" if os.path.isdir("/kaggle/input") else "./input"
test_root = os.path.join(base_input, "vesuvius-challenge-ink-detection", "test")

fragment_ids = [
    d for d in os.listdir(test_root) if os.path.isdir(os.path.join(test_root, d))
]

submission_lines = ["Id,Predicted"]
for fid in sorted(fragment_ids):
    try:
        vol_path = os.path.join(test_root, fid, "surface_volume")
        preds = process_volume_data(
            vol_path, stack_count, cube_size, model_CNN, device, batch_size
        )[0]
        starts, lengths = rle(preds.numpy())
        if len(starts) == 0:
            rle_str = ""
        else:
            rle_str = " ".join(map(str, sum(zip(starts, lengths), ())))
        submission_lines.append(f"{fid},{rle_str}")
    except Exception as e:
        print(f"Error processing fragment {fid}: {e}")
        submission_lines.append(f"{fid},")

with open("submission.csv", "w") as f:
    f.write("\n".join(submission_lines))

print("Submission file written to submission.csv")




## === cell 7
pd.read_csv("submission.csv")
