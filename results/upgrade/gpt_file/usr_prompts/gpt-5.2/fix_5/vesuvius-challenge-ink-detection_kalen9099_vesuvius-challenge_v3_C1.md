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

geopandas==0.14.4
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
protobuf==6.33.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
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

0.0395623179948312

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.15335) has done: 'I make the smallest changes needed to (1) ensure your inference thresholding is applied to probabilities (sigmoid) rather than raw logits (currently `.gt(0.4)` is on logits, which badly miscalibrates the binary mask and usually hurts F0.5), and (2) fix the RLE flattening order to match the competition’s pixel numbering (left-to-right then top-to-bottom), which is row-major (`order="C"`) rather than column-major (`order="F"`). These two fixes keep your model/patching core logic identical while making the submission encoding and binarization consistent with the metric expectations, which should move the score upward from “not yielded”/invalid toward your target. I also add a tiny safety cast so the predicted mask is strictly 0/1 uint8 before encoding, avoiding any accidental non-binary values.'

# 9. Code solution

## === cell 0
import os
import csv
import math
import glob
import gc
import json
import random
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import PIL.Image as Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as thd
from torch.nn import functional as F
from torchvision import transforms

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from tqdm import tqdm
from sklearn.metrics import fbeta_score
from sklearn.exceptions import UndefinedMetricWarning




## === cell 1
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        channels = int(out_channels / 2)
        if in_channels > out_channels:
            channels = int(in_channels / 2)

        layers = [
            nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        ]

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSampling, self).__init__()
        self.maxpool_to_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_to_conv(x)


class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSampling, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels + int(in_channels / 2), out_channels)

    def forward(self, inputs1, inputs2):
        inputs1 = self.up(inputs1)
        outputs = torch.cat([inputs1, inputs2], dim=1)
        outputs = self.conv(outputs)
        return outputs


class LastConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LastConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        output = self.conv(x)
        return output


class InkDetector(nn.Module):
    def __init__(self, in_channels=66):
        super(InkDetector, self).__init__()
        self.in_channels = in_channels

        self.inputs = DoubleConv(
            in_channels,
            64,
        )
        self.down_1 = DownSampling(64, 128)
        self.down_2 = DownSampling(128, 256)

        self.up_2 = UpSampling(256, 128)
        self.up_3 = UpSampling(128, 64)
        self.outputs = LastConv(64, 1)

    def forward(self, x):
        x1 = self.inputs(x)
        x2 = self.down_1(x1)
        x3 = self.down_2(x2)

        x6 = self.up_2(x3, x2)
        x7 = self.up_3(x6, x1)
        x = self.outputs(x7)
        return x




## === cell 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE



## === cell 3
model = InkDetector().to(DEVICE)



## === cell 4
LEARNING_RATE = 1e-5
read = True  # keep original behavior intention
TRAIN_RUN = False

WEIGHT_PATH = "/kaggle/input/kalen9099/32.pt"


def _find_any_checkpoint() -> Optional[str]:
    if os.path.exists(WEIGHT_PATH):
        return WEIGHT_PATH
    candidates = []
    for p in glob.glob("/kaggle/input/**/*.pt", recursive=True):
        try:
            sz = os.path.getsize(p)
        except OSError:
            continue
        if sz > 10_000:
            candidates.append((sz, p))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


ckpt_path = _find_any_checkpoint()
if read and ckpt_path is not None:
    try:
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {ckpt_path}")
        if missing or unexpected:
            print(
                f"Non-strict load. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}"
            )
    except Exception as e:
        print(f"Checkpoint load failed ({ckpt_path}): {e}")
        read = False
else:
    print("No checkpoint found; proceeding without pretrained weights.")
    read = False



## === cell 5
BUFFER = 32
if TRAIN_RUN:
    index = 1
    while index <= 3:
        print("Running dataset:", index)

        base_path = "/kaggle/input/vesuvius-challenge/"
        testloader_PATH = base_path + "train/"
        train_name = os.listdir(testloader_PATH + str(index) + "/surface_volume")
        lable_name = os.listdir(testloader_PATH + str(index))  # 讀取所有檔案名稱
        print(testloader_PATH, str(index))

        train_PATH = []
        lable_PATH = []
        mask_PATH = []
        for i in range(1):  # 所有檔案路徑 only dataset1
            for j in range(65):
                train_PATH.append(
                    testloader_PATH + str(index) + "/surface_volume/" + train_name[j]
                )
            lable_PATH.append(testloader_PATH + str(index) + "/inklabels.png")
            mask_PATH.append(testloader_PATH + str(index) + "/mask.png")

        train_dataset = []

        transform = transforms.Compose([transforms.ToTensor()])

        for i in tqdm(range(65)):  # 讀取資料
            mid = cv2.imread(train_PATH[i], cv2.IMREAD_GRAYSCALE)  # numpy數據
            train_dataset.append(mid)

        lable_dataset = cv2.imread(lable_PATH[0], cv2.IMREAD_GRAYSCALE)
        mask_dataset = cv2.imread(mask_PATH[0], cv2.IMREAD_GRAYSCALE)

        del train_PATH
        del lable_PATH
        del mask_PATH

        for cut in range(20):
            print("Running Cut:", cut + 1)

            all_fragment = []
            BUFFER = 32
            epoch = 1000  # 隨機切塊成n塊
            transform = transforms.Compose([transforms.ToTensor()])
            NP255 = np.ones((BUFFER * 2, BUFFER * 2)) * 255

            for n in tqdm(range(epoch)):
                x = random.randint(BUFFER, len(train_dataset[0]) - BUFFER)
                y = random.randint(BUFFER, len(train_dataset[0][0]) - BUFFER)
                while not (
                    0.7 * BUFFER * BUFFER
                    > (
                        lable_dataset[x - BUFFER : x + BUFFER, y - BUFFER : y + BUFFER]
                        == NP255
                    ).sum()
                    > 0.3 * BUFFER * BUFFER
                ):
                    x = random.randint(BUFFER, len(train_dataset[0]) - BUFFER)
                    y = random.randint(BUFFER, len(train_dataset[0][0]) - BUFFER)

                temp = torch.zeros((len(train_dataset) + 1, BUFFER * 2, BUFFER * 2))
                mid = []
                for a in range(len(train_dataset)):
                    one = np.zeros((BUFFER * 2, BUFFER * 2))
                    one[0 : BUFFER * 2, 0 : BUFFER * 2] = train_dataset[a][
                        x - BUFFER : x + BUFFER, y - BUFFER : y + BUFFER
                    ]
                    if a == 0:
                        temp[0] = transform(np.array(one.astype("uint8")))
                    temp[a + 1] = transform(np.array(one.astype("uint8")))
                mid.append(temp)
                temp = np.zeros((BUFFER * 2, BUFFER * 2))
                temp[0 : BUFFER * 2, 0 : BUFFER * 2] = lable_dataset[
                    x - BUFFER : x + BUFFER, y - BUFFER : y + BUFFER
                ]
                mid.append(transform(np.array(temp.astype("uint8"))))
                all_fragment.append(mid)

            BATCH_SIZE = 10
            train_loader = thd.DataLoader(
                all_fragment, batch_size=BATCH_SIZE, shuffle=False
            )

            epoch = 10000
            LEARNING_RATE = 1e-5
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=LEARNING_RATE,
                betas=(0.9, 0.999),
                eps=1e-09,
                weight_decay=0,
                amsgrad=False,
            )
            TRAINING_STEPS = len(train_loader)
            print("TRAINING_STEPS:", TRAINING_STEPS)

            fbeta_save = []
            loss_save = []
            for j in range(epoch):
                criterion = nn.BCEWithLogitsLoss()
                optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer, max_lr=LEARNING_RATE, total_steps=TRAINING_STEPS
                )
                model.train()
                running_loss = 0.0
                running_accuracy = 0.0
                running_fbeta = 0.0
                denom = 0
                pbar = tqdm(enumerate(train_loader), total=TRAINING_STEPS)
                for i, (subvolumes, inklabels) in pbar:
                    if i >= TRAINING_STEPS:
                        break
                    optimizer.zero_grad()
                    outputs = model(subvolumes.to(torch.float).to(DEVICE))
                    loss = criterion(outputs.float(), inklabels.float().to(DEVICE))
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    pred_ink = outputs.detach().sigmoid().gt(0.4).cpu().int()
                    accuracy = (pred_ink == inklabels).sum().float() / (
                        outputs.size(0)
                        * outputs.size(1)
                        * outputs.size(2)
                        * outputs.size(3)
                    )
                    running_fbeta += fbeta_score(
                        inklabels.view(-1).numpy(), pred_ink.view(-1).numpy(), beta=0.5
                    )
                    running_accuracy += accuracy.item()
                    running_loss += loss.item()
                    denom += 1
                    pbar.set_postfix(
                        {
                            "Loss": running_loss / denom,
                            "Accuracy": running_accuracy / denom,
                            "F0.5": running_fbeta / denom,
                            "Epoch": j,
                        }
                    )
                    if (i + 1) % TRAINING_STEPS == 0:
                        fbeta_save.append(running_fbeta / denom)
                        loss_save.append(running_loss / denom)
                        running_loss = 0.0
                        running_accuracy = 0.0
                        running_fbeta = 0.0
                        denom = 0
                        torch.save(model.state_dict(), WEIGHT_PATH)

        index = index + 1
        del all_fragment
        del lable_dataset
        del mask_dataset
        del subvolumes
        del inklabels



## === cell 6
warnings.simplefilter("ignore", UndefinedMetricWarning)



## === cell 7
if TRAIN_RUN:
    with torch.no_grad():
        pbar = tqdm(enumerate(train_loader), total=TRAINING_STEPS)
        for i, (subvolumes, inklabels) in pbar:
            if i < 20:
                outputs = model(subvolumes.to(torch.float).to(DEVICE))
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
                ax1.set_title("subvolumes")
                ax1.imshow(subvolumes[i][0], cmap="gray")
                ax2.set_title("outputs")
                ax2.imshow(outputs[i][0].cpu(), cmap="gray")
                ax4.set_title("inklabels")
                ax4.imshow(inklabels[i][0], cmap="gray")
                a = torch.empty((BUFFER * 2, BUFFER * 2))
                for j in range(BUFFER * 2):
                    for k in range(BUFFER * 2):
                        a[j][k] = outputs[i][0][j][k].gt(0.4)
                ax3.set_title("outputs to 0/1")
                ax3.imshow(a, cmap="gray")
                plt.show()

    del train_dataset
    del train_loader
    del all_fragment
    gc.collect()



## === cell 8
pass



## === cell 9
DATASET_ROOT = "/kaggle/input/vesuvius-challenge-ink-detection"
SAMPLE_SUB_PATH = f"{DATASET_ROOT}/sample_submission.csv"

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_ids = sample_sub["Id"].tolist()

base_path = DATASET_ROOT
testloader_PATH = base_path + "/test/"

print("Using test path:", testloader_PATH)
print("Test Ids:", test_ids)



## === cell 10
transform = transforms.Compose([transforms.ToTensor()])
BUFFER = 32




## === cell 11
def load_fragment_stack(fragment_dir: str) -> List[np.ndarray]:
    sv_dir = os.path.join(fragment_dir, "surface_volume")
    slice_paths = sorted(glob.glob(os.path.join(sv_dir, "*.tif")))
    if len(slice_paths) == 0:
        raise FileNotFoundError(f"No .tif slices found in {sv_dir}")
    stack = []
    for p in slice_paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read image: {p}")
        stack.append(img)
    return stack


def make_patches_from_stack(
    stack: List[np.ndarray], buffer_size: int
) -> Tuple[List[torch.Tensor], int, int, int, int]:
    height = stack[0].shape[0]
    width = stack[0].shape[1]
    iter_h = math.ceil(height / (buffer_size * 2))
    iter_w = math.ceil(width / (buffer_size * 2))

    all_fragment = []
    x = 0
    y = 0
    for i in range(iter_h):
        y = 0
        for j in range(iter_w):
            temp = torch.empty((len(stack) + 1, buffer_size * 2, buffer_size * 2))
            for a in range(len(stack)):
                temp_pad = torch.zeros(buffer_size * 2, buffer_size * 2)
                x2 = min(x + 2 * buffer_size, height)
                y2 = min(y + 2 * buffer_size, width)
                patch_np = stack[a][x:x2, y:y2]
                patch_t = transform(np.array(patch_np.astype("uint8"))).squeeze(0)
                temp_pad[: patch_t.shape[0], : patch_t.shape[1]] = patch_t
                if a == 0:
                    temp[a] = temp_pad
                temp[a + 1] = temp_pad
            all_fragment.append(temp)
            y += buffer_size * 2
        x += buffer_size * 2

    return all_fragment, height, width, iter_h, iter_w




## === cell 12
BATCH_SIZE = 1




## === cell 13
def infer_mask_for_fragment(
    fragment_id: str, threshold: float = 0.4
) -> Tuple[torch.Tensor, np.ndarray]:
    fragment_dir = os.path.join(testloader_PATH, fragment_id)
    stack = load_fragment_stack(fragment_dir)

    mask_path = os.path.join(fragment_dir, "mask.png")
    mask_dataset = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_dataset is None:
        raise FileNotFoundError(f"mask.png not found/readable: {mask_path}")

    all_fragment, height, width, iter_h, iter_w = make_patches_from_stack(stack, BUFFER)
    test_loader = thd.DataLoader(all_fragment, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()

    all_a = torch.empty(BUFFER * 2 * iter_h, BUFFER * 2 * iter_w, dtype=torch.float32)

    m = 0
    n = 0
    with torch.no_grad():
        for subvolumes in tqdm(
            test_loader,
            total=len(test_loader),
            desc=f"Infer {fragment_id}",
            leave=False,
        ):
            outputs = model(subvolumes.to(torch.float32).to(DEVICE))
            probs = outputs.sigmoid()

            a = (
                probs[0, 0, : BUFFER * 2, : BUFFER * 2]
                .gt(threshold)
                .to(torch.float32)
                .cpu()
            )

            x = m * BUFFER * 2
            y = n * BUFFER * 2
            all_a[x : x + BUFFER * 2, y : y + BUFFER * 2] = a

            if n + 1 < iter_w:
                n += 1
            else:
                n = 0
                m += 1
            if m == iter_h:
                break

    all_a = all_a[:height, :width]
    mask01 = (mask_dataset > 0).astype(np.uint8)

    all_masked_a = all_a.numpy().astype(np.uint8) * mask01

    del stack, all_fragment, test_loader
    gc.collect()
    return torch.from_numpy(all_masked_a), mask01




## === cell 14
def rle(output: torch.Tensor) -> str:
    mask = output.detach().cpu().numpy().astype(np.uint8)
    if mask.size == 0:
        return ""

    flat = mask.flatten(order="C")

    padded = np.pad(flat, (1, 1), mode="constant", constant_values=0)
    changes = np.where(padded[1:] != padded[:-1])[0]
    if changes.size == 0:
        return ""
    runs = changes[::2]
    ends = changes[1::2]
    lengths = ends - runs
    starts_1idx = runs + 1
    if starts_1idx.size == 0:
        return ""
    rle_pairs = np.column_stack([starts_1idx, lengths]).reshape(-1)
    return " ".join(map(str, rle_pairs.tolist()))




## === cell 15
VIS = False



## === cell 16
pass



## === cell 17
INFER_THRESHOLD = 0.85

submission = defaultdict(list)

for fid in test_ids:
    pred_mask, _mask01 = infer_mask_for_fragment(fid, threshold=INFER_THRESHOLD)

    if VIS:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.set_title(f"{fid}: pred_mask")
        ax1.imshow(pred_mask.numpy(), cmap="gray")
        ax2.set_title(f"{fid}: mask")
        ax2.imshow(_mask01, cmap="gray")
        plt.show()

    submission["Id"].append(fid)
    submission["Predicted"].append(rle(pred_mask))

    del pred_mask, _mask01
    gc.collect()



## === cell 18
sub_df = pd.DataFrame.from_dict(submission)
sub_df = sample_sub[["Id"]].merge(sub_df, on="Id", how="left")
sub_df["Predicted"] = sub_df["Predicted"].fillna("")
sub_path = "/kaggle/working/submission.csv"
sub_df.to_csv(sub_path, index=False)
print("Wrote:", sub_path)
print(sub_df.head())
