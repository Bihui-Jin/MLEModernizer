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

0.0968707289588988

# 6. Current score

0.03109

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.08095) has done: 'I remove the hard dependency on the missing external model file by training the same UNET architecture locally on the provided training fragments, so the notebook runs end-to-end in this Kaggle environment. I also fix the data loading bugs in `process_volume_data` (it was using `cv2.imread` on `.tif` and scaling incorrectly) and make the padding logic correct when the image size is already divisible by `cube_size`. Finally, I generate `submission.csv` in the required `Id,Predicted` RLE format by running inference on each test fragment and encoding the binary mask (restricted to the fragment `mask.png`) so Kaggle accepts the file.'
- What this solution (achieved 0.01105) has done: 'To move your score up toward the 0.0969 target with minimal risk, I’m keeping your exact UNET/training loop/data extraction intact and only adjusting prediction post-processing to better match the F0.5 metric (precision-weighted). Concretely: (1) use the correct RLE flattening order (column-major) expected by this competition, which can materially improve the measured score without changing the model, and (2) slightly increase the binarization threshold to favor precision and improve F0.5. I’m also setting the model to eval mode during inference explicitly (already effectively done, but kept for safety) and leaving all paths unchanged so the notebook still writes a valid `submission.csv`.'
- What this solution (achieved 0.01526) has done: 'I remove the biggest CPU bottlenecks without touching the model, loss, training loop semantics, feature extraction steps, or thresholds logic: (1) avoid Python list-of-blocks construction and inner Python loops when tiling/untiling by using `torch.unfold`/reshape and batched writes, (2) speed up `.tif` reading + per-slice OpenCV preprocessing by using a thread pool (same operations, just parallelized), (3) cut redundant disk I/O by computing the train-fragment prediction maps once and reusing them for all threshold candidates, and (4) use DataLoader worker processes + pinned memory for faster training transfers. These changes are provably equivalent (same pixels go through the same preprocessing and the same model; only the batching/assembly strategy and caching differ), and they directly target the timeout causes (I/O + Python-loop overhead).'
- What this solution (achieved 0.00316) has done: 'Your current score (0.01526) is far below the target (0.09687), so we should make small changes that legitimately improve the F0.5 score without altering your UNET/training loop or feature extraction. The biggest likely issue is that the model is trained on non-overlapping 32×32 tiles but inferred over the full image with hard tile boundaries, which creates seams and hurts precision; we can fix this by using overlapped tiling at inference and averaging overlaps (same model, same inputs, just better stitching). Because F0.5 favors precision, we keep your threshold-tuning but compute train/test predictions using the same overlapped inference so the selected threshold matches test-time behavior. Finally, we apply the provided `mask.png` during threshold tuning (not just after) to avoid learning a threshold influenced by invalid regions.'
- What this solution (achieved 0.01763) has done: 'Your current score (0.00316) is far below the target (0.09687), so we should make a small, high-impact correction that preserves your model/training core logic. The biggest likely culprit is a mismatch between the intended training “stack_count=30 slices” and what your loaders actually read: both `process_volume_data` and `load_fragment_train` take the *first* 30 `.tif` files, which are not guaranteed to be the most informative mid-slices; using the centered 30 slices is a common baseline improvement in this competition without changing the model. I also ensure the `.tif` file ordering is numerically correct (e.g., `1.tif, 2.tif, ..., 10.tif` instead of lexicographic) so slice selection is stable and meaningful. These are minimal changes that keep your UNET, loss, training loop, tiling, overlap inference, and threshold tuning intact, but should substantially improve the learned signal and therefore move the score toward the target.'
- What this solution (achieved 0.03109) has done: 'I keep your UNET, training loop, loss, and tiling/inference logic unchanged, and only adjust the threshold selection to better match the competition’s *global* F0.5 behavior. Right now the threshold is tuned by averaging per-fragment F0.5, which can pick a suboptimal threshold because Kaggle effectively scores over the union of pixels; I instead maximize F0.5 computed on the concatenated masked pixels from both train fragments (same predictions, same metric, just aggregated correctly). To avoid “grid miss” while staying minimal, I expand the threshold grid slightly around your current range so we can land nearer the true optimum without changing the model. This should move your score upward toward the 0.0969 target while preserving core semantics and still producing a valid `submission.csv`.'

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
batch_size = 50  # размер батча
cube_size = 32  # нарезка изображений на cube_size х cube_size
epochs = 6  # FIX: original 1000 would exceed time; minimal workable training to produce a valid submission end-to-end.
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
        b, c, hgt, wdt = h.shape
        h_seq = h.permute(0, 2, 3, 1).contiguous().view(b, hgt * wdt, c)
        h_seq, _ = self.rnn(h_seq)
        h2 = h_seq.view(b, hgt, wdt, c).permute(0, 3, 1, 2).contiguous()
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
data_root = "/kaggle/input/vesuvius-challenge-ink-detection"

train_root = os.path.join(data_root, "train")
test_root = os.path.join(data_root, "test")

model_CNN = UNET(stack_count).to(device)
gc.collect()




## === cell 11
def _read_tif_gray(path_tif):
    arr = tifffile.imread(path_tif)
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = arr.astype(np.float32)
    maxv = float(np.max(arr)) if np.max(arr) > 0 else 1.0
    arr = arr / maxv
    return arr


def _preprocess_slice_from_path(fn: str) -> np.ndarray:
    sl = _read_tif_gray(fn)
    sl_u8 = (np.clip(sl, 0, 1) * 255).astype(np.uint8)
    sl_u8 = cv2.filter2D(sl_u8, -1, kernel)
    sl_u8 = cv2.equalizeHist(sl_u8)
    sl_u8 = cv2.medianBlur(sl_u8, 3)
    return sl_u8.astype(np.float32) / 255.0


def _sorted_tif_files(vol_dir: str):
    files = glob.glob(os.path.join(vol_dir, "*.tif"))

    def _key(p):
        b = os.path.basename(p)
        stem = os.path.splitext(b)[0]
        try:
            return int(stem)
        except ValueError:
            return stem

    return sorted(files, key=_key)


def _select_center_slices(files, stack_count: int):
    if len(files) <= stack_count:
        return files
    start = (len(files) - stack_count) // 2
    return files[start : start + stack_count]


def process_volume_data(
    path, stack_count, cube_size, model_CNN, device, batch_size, stride=None
):
    files_all = _sorted_tif_files(path)
    if len(files_all) == 0:
        raise FileNotFoundError(f"No .tif files found in {path}")
    files = _select_center_slices(files_all, stack_count)

    first = _read_tif_gray(files[0])
    H, W = first.shape

    max_workers = min(8, (os.cpu_count() or 2))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        slices = list(ex.map(_preprocess_slice_from_path, files))

    stack_3 = torch.from_numpy(np.stack(slices, axis=0)).to(torch.float32)  # (S,H,W)

    if stride is None:
        stride = cube_size  # default keeps previous semantics

    pad_h = (stride - (H % stride)) % stride
    pad_w = (stride - (W % stride)) % stride
    Hp = H + pad_h
    Wp = W + pad_w
    if Hp < cube_size:
        pad_h += cube_size - Hp
        Hp = H + pad_h
    if Wp < cube_size:
        pad_w += cube_size - Wp
        Wp = W + pad_w

    image = F.pad(stack_3, (0, pad_w, 0, pad_h, 0, 0))  # (S,Hp,Wp)
    Hp, Wp = image.shape[1], image.shape[2]

    ny = (Hp - cube_size) // stride + 1
    nx = (Wp - cube_size) // stride + 1

    patches = (
        image.unfold(1, cube_size, stride)
        .unfold(2, cube_size, stride)
        .permute(1, 2, 0, 3, 4)
        .contiguous()
        .view(ny * nx, stack_count, cube_size, cube_size)
    )

    model_CNN.eval()
    out_patches = torch.empty((ny * nx, 1, cube_size, cube_size), dtype=torch.float32)

    with torch.no_grad():
        for start in range(0, patches.shape[0], batch_size):
            end = min(start + batch_size, patches.shape[0])
            xb = patches[start:end].to(device, non_blocking=True).float()
            out = model_CNN(xb).detach().to("cpu")
            out_patches[start:end].copy_(out)

    pred_sum = torch.zeros((Hp, Wp), dtype=torch.float32)
    pred_cnt = torch.zeros((Hp, Wp), dtype=torch.float32)

    idx = 0
    for iy in range(ny):
        y0 = iy * stride
        y1 = y0 + cube_size
        for ix in range(nx):
            x0 = ix * stride
            x1 = x0 + cube_size
            pred_sum[y0:y1, x0:x1] += out_patches[idx, 0]
            pred_cnt[y0:y1, x0:x1] += 1.0
            idx += 1

    full_map = (pred_sum / torch.clamp(pred_cnt, min=1.0)).unsqueeze(0)  # (1,Hp,Wp)
    return full_map[:, :H, :W]


def rle_encode(mask: np.ndarray) -> str:
    mask = (mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        return ""
    pixels = mask.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(int(x)) for x in runs)


def fbeta_from_binary(
    y_true_bin: np.ndarray, y_pred_bin: np.ndarray, beta: float = 0.5
) -> float:
    y_true_bin = (y_true_bin > 0).astype(np.uint8).ravel()
    y_pred_bin = (y_pred_bin > 0).astype(np.uint8).ravel()
    tp = int(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
    fp = int(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
    fn = int(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))
    if tp == 0:
        return 0.0
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    b2 = beta * beta
    denom = b2 * p + r
    if denom == 0:
        return 0.0
    return (1 + b2) * p * r / denom


class PatchDataset(Dataset):
    def __init__(
        self,
        stack: torch.Tensor,
        label: torch.Tensor,
        valid_mask: torch.Tensor,
        cube_size: int,
    ):
        self.stack = stack
        self.label = label
        self.valid_mask = valid_mask
        self.cube = cube_size
        H, W = label.shape
        coords = []
        for y in range(0, H - cube_size + 1, cube_size):
            for x in range(0, W - cube_size + 1, cube_size):
                if valid_mask[y : y + cube_size, x : x + cube_size].sum() > 0:
                    coords.append((y, x))
        self.coords = coords

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        y, x = self.coords[idx]
        img = self.stack[:, y : y + self.cube, x : x + self.cube]
        msk = self.label[y : y + self.cube, x : x + self.cube].unsqueeze(0)
        return img, msk


def load_fragment_train(fragment_id: str, stack_count: int):
    frag_dir = os.path.join(train_root, str(fragment_id))
    vol_dir = os.path.join(frag_dir, "surface_volume")
    files_all = _sorted_tif_files(vol_dir)
    files = _select_center_slices(files_all, stack_count)

    first = _read_tif_gray(files[0])
    H, W = first.shape

    max_workers = min(8, (os.cpu_count() or 2))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        slices = list(ex.map(_preprocess_slice_from_path, files))
    stack = torch.from_numpy(np.stack(slices, axis=0)).to(torch.float32)  # (S,H,W)

    ink = cv2.imread(os.path.join(frag_dir, "inklabels.png"), cv2.IMREAD_GRAYSCALE)
    ink = (ink > 0).astype(np.float32)
    ink = torch.from_numpy(ink)

    m = cv2.imread(os.path.join(frag_dir, "mask.png"), cv2.IMREAD_GRAYSCALE)
    m = (m > 0).astype(np.float32)
    m = torch.from_numpy(m)

    pad_h = (cube_size - (H % cube_size)) % cube_size
    pad_w = (cube_size - (W % cube_size)) % cube_size
    stack = F.pad(stack, (0, pad_w, 0, pad_h, 0, 0))
    ink = F.pad(ink, (0, pad_w, 0, pad_h))
    m = F.pad(m, (0, pad_w, 0, pad_h))
    return stack, ink, m, (H, W)


model_CNN.train()
criterion = nn.BCELoss()
optimizer = optim.Adam(model_CNN.parameters(), lr=learning_rate)

all_loaders = []
for fid in ["1", "2"]:
    stack, ink, m, _ = load_fragment_train(fid, stack_count)
    ds = PatchDataset(stack, ink, m, cube_size)
    if len(ds) == 0:
        continue

    nw = min(4, (os.cpu_count() or 2))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(nw > 0),
        prefetch_factor=2 if nw > 0 else None,
    )
    all_loaders.append(loader)

for ep in range(epochs):
    ep_loss = 0.0
    n_batches = 0
    for loader in all_loaders:
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model_CNN(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            ep_loss += float(loss.detach().cpu().item())
            n_batches += 1
    if n_batches > 0:
        print(f"epoch {ep+1}/{epochs} loss={ep_loss/n_batches:.5f}")

model_CNN.eval()
gc.collect()

thr_grid = np.round(np.arange(0.55, 0.901, 0.025), 3).tolist()
best_thr = 0.70
best_f05 = -1.0

inference_stride = (
    cube_size // 2
)  # overlap (16px) to reduce seams; inference-only change
train_cache = {}
for fid in ["1", "2"]:
    frag_dir = os.path.join(train_root, str(fid))
    vol_dir = os.path.join(frag_dir, "surface_volume")
    pred_map = process_volume_data(
        vol_dir,
        stack_count,
        cube_size,
        model_CNN,
        device,
        batch_size,
        stride=inference_stride,
    )[0].numpy()
    ink = cv2.imread(os.path.join(frag_dir, "inklabels.png"), cv2.IMREAD_GRAYSCALE)
    y_true = (ink > 0).astype(np.uint8)
    m = cv2.imread(os.path.join(frag_dir, "mask.png"), cv2.IMREAD_GRAYSCALE)
    valid = (m > 0).astype(np.uint8)
    train_cache[fid] = (pred_map, y_true, valid)

for thr in thr_grid:
    y_true_all = []
    y_pred_all = []
    for fid in ["1", "2"]:
        pred_map, y_true, valid = train_cache[fid]
        y_true_all.append(y_true[valid > 0].ravel())
        y_pred_all.append((pred_map[valid > 0] >= thr).astype(np.uint8).ravel())
    y_true_all = (
        np.concatenate(y_true_all, axis=0)
        if len(y_true_all)
        else np.array([], dtype=np.uint8)
    )
    y_pred_all = (
        np.concatenate(y_pred_all, axis=0)
        if len(y_pred_all)
        else np.array([], dtype=np.uint8)
    )

    global_f05 = (
        fbeta_from_binary(y_true_all, y_pred_all, beta=0.5) if y_true_all.size else 0.0
    )
    if global_f05 > best_f05:
        best_f05 = global_f05
        best_thr = float(thr)

print(
    f"Chosen threshold (GLOBAL masked F0.5 on train): thr={best_thr:.3f}, f0.5={best_f05:.6f}"
)

sample_path = os.path.join(data_root, "sample_submission.csv")
sub = pd.read_csv(sample_path)

pred_strings = []
for frag_id in sub["Id"].astype(str).tolist():
    frag_dir = os.path.join(test_root, frag_id)
    vol_dir = os.path.join(frag_dir, "surface_volume")

    pred_map = process_volume_data(
        vol_dir,
        stack_count,
        cube_size,
        model_CNN,
        device,
        batch_size,
        stride=inference_stride,
    )[0].numpy()

    m = cv2.imread(os.path.join(frag_dir, "mask.png"), cv2.IMREAD_GRAYSCALE)
    valid = (m > 0).astype(np.uint8)

    bin_mask = ((pred_map >= best_thr).astype(np.uint8) * valid).astype(np.uint8)
    pred_strings.append(rle_encode(bin_mask))

sub["Predicted"] = pred_strings
output_path = "submission.csv"
sub.to_csv(output_path, index=False)
print("Wrote", output_path, "rows=", len(sub))
print(sub.head())
