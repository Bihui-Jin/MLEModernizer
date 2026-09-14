# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 1
import os
import csv
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torchvision
import random
import matplotlib.pyplot as plt
import pandas as pd 
import torch.optim as optim
import math
import cv2
import tensorflow as tf
import numpy.ma as ma
from torch.nn import functional as F

from sklearn.metrics import fbeta_score
import torch.utils.data as thd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from torchvision import datasets
import gc
import glob
import json
from collections import defaultdict
import multiprocessing as mp
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple
import warnings

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import PIL.Image as Image
from sklearn.exceptions import UndefinedMetricWarning
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as thd
from tqdm import tqdm

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
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
            nn.ReLU(True)
        ]

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)

class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSampling, self).__init__()
        self.maxpool_to_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_to_conv(x)

class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSampling, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels + int(in_channels / 2), out_channels)

    def forward(self, inputs1, inputs2):
        
        inputs1 = self.up(inputs1)
        outputs = torch.cat([inputs1, inputs2], dim=1)
        outputs = self.conv(outputs)
        return outputs

class LastConv(nn.Module):
    def __init__(self, in_channels, out_channels ):
        super(LastConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1 )
    def forward(self, x):
        output = self.conv(x)
        return output
    
class InkDetector(nn.Module):
    def __init__(self, in_channels=66):
        super(InkDetector, self).__init__()
        self.in_channels = in_channels
        
        self.inputs = DoubleConv(in_channels, 64,)
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

## === cell 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## === cell 4
model = InkDetector().to(DEVICE)

## === cell 5
LEARNING_RATE = 1e-5
read = True # To avoid re-running when saving the notebook
TRAIN_RUN = False

WEIGHT_PATH = '/kaggle/input/kalen9099/32.pt'
if read:
    model_weights = torch.load(WEIGHT_PATH)
    model.load_state_dict(model_weights) #TRAINING_STEPS

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/540476744.py in <cell line: 0>()
      5 WEIGHT_PATH = '/kaggle/input/kalen9099/32.pt'
      6 if read:
----> 7     model_weights = torch.load(WEIGHT_PATH)
      8     model.load_state_dict(model_weights) #TRAINING_STEPS

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in load(f, map_location, pickle_module, weights_only, mmap, **pickle_load_args)
   1423         pickle_load_args["encoding"] = "utf-8"
   1424 
-> 1425     with _open_file_like(f, "rb") as opened_file:
   1426         if _is_zipfile(opened_file):
   1427             # The zipfile reader is going to advance the current file position.

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in _open_file_like(name_or_buffer, mode)
    749 def _open_file_like(name_or_buffer, mode):
    750     if _is_path(name_or_buffer):
--> 751         return _open_file(name_or_buffer, mode)
    752     else:
    753         if "w" in mode:

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in __init__(self, name, mode)
    730 class _open_file(_opener):
    731     def __init__(self, name, mode):
--> 732         super().__init__(open(name, mode))
    733 
    734     def __exit__(self, *args):

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/kalen9099/32.pt'

## === cell 6
BUFFER = 32
if TRAIN_RUN:
    index = 1
    while(index <= 3):
        print("Running dataset:", index)
        
        base_path = "/kaggle/input/vesuvius-challenge/"
        testloader_PATH = base_path + "train/"
        train_name = os.listdir(testloader_PATH+str(index)+'/surface_volume')
        lable_name = os.listdir(testloader_PATH+str(index)) #讀取所有檔案名稱
        print(testloader_PATH,str(index))
        
        train_PATH = []
        lable_PATH = []
        mask_PATH = []
        for i in range(1):#所有檔案路徑 only dataset1
            for j in range(65):
                train_PATH.append(testloader_PATH+str(index)+'/surface_volume/'+train_name[j])
            lable_PATH.append(testloader_PATH+str(index)+'/inklabels.png')
            mask_PATH.append(testloader_PATH+str(index)+'/mask.png')
          
        train_dataset = []

        transform = transforms.Compose([transforms.ToTensor()])
        
        for i in tqdm(range(65)): #讀取資料
            mid = cv2.imread(train_PATH[i],cv2.IMREAD_GRAYSCALE)  # numpy數據
            train_dataset.append(mid)
            
        lable_dataset = cv2.imread(lable_PATH[0],cv2.IMREAD_GRAYSCALE)
        mask_dataset = cv2.imread(mask_PATH[0],cv2.IMREAD_GRAYSCALE)


        del train_PATH
        del lable_PATH
        del mask_PATH

      
        for cut in range(20):
            print("Running Cut:", cut+1)

            all_fragment = []
            BUFFER = 32
            epoch = 1000 #隨機切塊成n塊
            transform = transforms.Compose([transforms.ToTensor()])
            NP255 = np.ones((BUFFER*2,BUFFER*2))*255

            for n in tqdm(range(epoch)):
                x = random.randint(BUFFER, len(train_dataset[0]) - BUFFER)
                y = random.randint(BUFFER, len(train_dataset[0][0]) - BUFFER)
                while not (0.7*BUFFER*BUFFER > (lable_dataset[x-BUFFER:x+BUFFER,y-BUFFER:y+BUFFER] == NP255).sum() > 0.3*BUFFER*BUFFER) :
                    x = random.randint(BUFFER, len(train_dataset[0]) - BUFFER)
                    y = random.randint(BUFFER, len(train_dataset[0][0]) - BUFFER)

                temp = torch.zeros((len(train_dataset)+1,BUFFER*2,BUFFER*2))
                mid = []
                for a in range(len(train_dataset)):
                    one = np.zeros((BUFFER*2,BUFFER*2))
                    one[0:BUFFER*2,0:BUFFER*2] = train_dataset[a][x-BUFFER:x+BUFFER,y-BUFFER:y+BUFFER]
                    if a == 0:
                        temp[0] = transform(np.array(one.astype('uint8')))
                    temp[a+1] = transform(np.array(one.astype('uint8')))
                mid.append(temp)
                temp = np.zeros((BUFFER*2,BUFFER*2))
                temp[0:BUFFER*2,0:BUFFER*2] = lable_dataset[x-BUFFER:x+BUFFER,y-BUFFER:y+BUFFER]
                mid.append(transform(np.array(temp.astype('uint8'))))
                all_fragment.append(mid)

            BATCH_SIZE = 10
            train_loader = thd.DataLoader(all_fragment, batch_size=BATCH_SIZE, shuffle=False)

            epoch = 10000
            LEARNING_RATE = 1e-5
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-09, weight_decay=0, amsgrad=False)
            TRAINING_STEPS = len(train_loader)
            print('TRAINING_STEPS:', TRAINING_STEPS)

            fbeta_save = []
            loss_save = []
            for j in range(epoch):
                criterion = nn.BCEWithLogitsLoss()
                optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
                scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, total_steps=TRAINING_STEPS)
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
                    accuracy = (pred_ink == inklabels).sum().float()/(outputs.size(0)*outputs.size(1)*outputs.size(2)*outputs.size(3))
                    running_fbeta += fbeta_score(inklabels.view(-1).numpy(), pred_ink.view(-1).numpy(), beta=0.5)
                    running_accuracy += accuracy.item()
                    running_loss += loss.item()
                    denom += 1
                    pbar.set_postfix({"Loss": running_loss / denom, "Accuracy": running_accuracy / denom, "F0.5": running_fbeta / denom, "Epoch": j})
                    if (i + 1) % TRAINING_STEPS == 0:
                        fbeta_save.append(running_fbeta / denom)
                        loss_save.append(running_loss / denom)
                        running_loss = 0.
                        running_accuracy = 0.
                        running_fbeta = 0.
                        denom = 0
                        torch.save(model.state_dict(), WEIGHT_PATH)

        index = index + 1
        del all_fragment
        del lable_dataset
        del mask_dataset
        del subvolumes
        del inklabels

## === cell 7
warnings.simplefilter('ignore', UndefinedMetricWarning)

## === cell 8
if TRAIN_RUN:
    with torch.no_grad():
        pbar = tqdm(enumerate(train_loader), total=TRAINING_STEPS)
        for i,(subvolumes, inklabels) in pbar:
            if i <20:
                outputs = model(subvolumes.to(torch.float).to(DEVICE))
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4)
                ax1.set_title("subvolumes")
                ax1.imshow(subvolumes[i][0], cmap = "gray")
                ax2.set_title("outputs")
                ax2.imshow(outputs[i][0].cpu(), cmap = "gray")
                ax4.set_title("inklabels")
                ax4.imshow(inklabels[i][0], cmap = "gray")
                a = torch.empty((BUFFER*2,BUFFER*2))
                for j in range(BUFFER*2):
                    for k in range(BUFFER*2):
                        a[j][k] = outputs[i][0][j][k].gt(0.4)
                ax3.set_title("outputs to 0/1")
                ax3.imshow(a, cmap = "gray")
                plt.show()

    del train_dataset
    del train_loader
    del all_fragment
    gc.collect()

## === cell 10
base_path = "/kaggle/input/vesuvius-challenge/"
test_dir = "a"
testloader_PATH = base_path + "test/"
test_name = os.listdir(testloader_PATH+test_dir+'/surface_volume')
lable_name = os.listdir(testloader_PATH+test_dir) #讀取所有檔案名稱
print(testloader_PATH+test_dir)

test_PATH = []
mask_PATH = []
for i in range(1):#所有檔案路徑 only dataset1
    for j in range(65):
        test_PATH.append(testloader_PATH+test_dir+'/surface_volume/'+test_name[j])
    mask_PATH.append(testloader_PATH+test_dir+'/mask.png')
    
test_dataset = []

np.set_printoptions(precision=2) #控制小數精度

for i in tqdm(range(65)): #讀取資料
    mid = cv2.imread(test_PATH[i],cv2.IMREAD_GRAYSCALE)  # numpy數據
    test_dataset.append(mid)

mask_dataset = cv2.imread(mask_PATH[0],cv2.IMREAD_GRAYSCALE)

test_PATH = None
mask_PATH = None
test_name = None
lable_name = None
mid = None
del test_PATH
del mask_PATH
del test_name
del lable_name
del mid
gc.collect()

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2370417335.py in <cell line: 0>()
      4 test_dir = "a"
      5 testloader_PATH = base_path + "test/"
----> 6 test_name = os.listdir(testloader_PATH+test_dir+'/surface_volume')
      7 lable_name = os.listdir(testloader_PATH+test_dir) #讀取所有檔案名稱
      8 print(testloader_PATH+test_dir)

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/vesuvius-challenge/test/a/surface_volume'

## === cell 11
transform = transforms.Compose([transforms.ToTensor()])
BUFFER = 32
height = len(mask_dataset)
width = len(mask_dataset[0])
print("height:", height, " width:", width)

iter_h = math.ceil(height / (BUFFER * 2))
iter_w = math.ceil(width / (BUFFER * 2))
print("iter_h:", iter_h, " iter_w:", iter_w)
print("Buffer:",BUFFER)

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2284515748.py in <cell line: 0>()
      1 transform = transforms.Compose([transforms.ToTensor()])
      2 BUFFER = 32
----> 3 height = len(mask_dataset)
      4 width = len(mask_dataset[0])
      5 print("height:", height, " width:", width)

NameError: name 'mask_dataset' is not defined

## === cell 12
temp_pad = torch.zeros(BUFFER*2,BUFFER*2)
mid = []
all_fragment = []
x = 0
y = 0
tp = 0
print(len(test_dataset))

one = []
for i in tqdm(range(iter_h)):
    for j in range(iter_w):
        temp = torch.empty((len(test_dataset)+1,BUFFER*2,BUFFER*2))
        for a in range(len(test_dataset)):
            temp_pad = torch.zeros(BUFFER*2,BUFFER*2)
            if y+(2*BUFFER)-1 > width and x+(2*BUFFER)-1 > height:
                temp_pad[0:height-x, 0:width-y] = torch.from_numpy(test_dataset[a][x:height, y:width])
                tp = 1
            elif y+(2*BUFFER)-1 > width:
                temp_pad[0:(2*BUFFER), 0:width-y] = transform(np.array(test_dataset[a][x:x+(2*BUFFER), y:width]))
                tp = 2
            elif x+(2*BUFFER)-1 > height:
                temp_pad[0:height-x, 0:(2*BUFFER)] = transform(np.array(test_dataset[a][x:height, y:y+(2*BUFFER)]))
                tp = 3
            else:
                temp_pad = transform(np.array(test_dataset[a][x:x+(2*BUFFER), y:y+(2*BUFFER)]))
                tp = 4
            if a == 0:
                temp[a] = temp_pad
            temp[a+1] = temp_pad
        y = y + (BUFFER * 2)
        if tp == 1:
            y = 0
            x = 0
        elif tp == 2:
            y = 0
        all_fragment.append(temp)
    x = x + BUFFER*2
    temp = None
    temp = None
    del temp
    del temp_pad
    gc.collect()
    
test_dataset = None
del test_dataset
gc.collect()

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/464111095.py in <cell line: 0>()
      6 y = 0
      7 tp = 0
----> 8 print(len(test_dataset))
      9 
     10 one = []

NameError: name 'test_dataset' is not defined

## === cell 13
BATCH_SIZE = 1
test_loader = thd.DataLoader(all_fragment, batch_size=BATCH_SIZE, shuffle=False)
TEST_STEPS = len(test_loader)
print('TEST_STEPS:', TEST_STEPS)

## === cell 14
x = 0
y = 0
m = 0
n = 0
all_a = torch.empty(BUFFER*2*iter_h,BUFFER*2*iter_w)

with torch.no_grad():
    for i,subvolumes in enumerate(tqdm(test_loader)):
        outputs = model(subvolumes.to(torch.float).to(DEVICE))
        a = torch.empty(BUFFER*2,BUFFER*2)
        a = outputs[0][0][0:BUFFER*2, 0:BUFFER*2].gt(0.4)*1
        x =  m * BUFFER * 2 
        y =  n * BUFFER * 2 
        all_a[x:x+(BUFFER*2), y:y+(BUFFER*2)] = a[0:BUFFER*2, 0:BUFFER*2]
        if n + 1 < iter_w:
            n = n + 1
        elif n + 1 == iter_w:
            n = 0
            m = m + 1
        
        if m  == iter_h:
            print("break")
            break
            

test_loader = None
all_fragment = None
a =  None
del test_loader
del all_fragment
del a
gc.collect()

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3532197131.py in <cell line: 0>()
      3 m = 0
      4 n = 0
----> 5 all_a = torch.empty(BUFFER*2*iter_h,BUFFER*2*iter_w)
      6 
      7 with torch.no_grad():

NameError: name 'iter_h' is not defined

## === cell 15
all_a = all_a[0:height, 0:width]
all_masked_a = all_a * mask_dataset

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2216286572.py in <cell line: 0>()
----> 1 all_a = all_a[0:height, 0:width]
      2 all_masked_a = all_a * mask_dataset

NameError: name 'all_a' is not defined

## === cell 16
fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(15, 15))
ax1.set_title("all_a")
ax1.imshow(all_a, cmap = "gray")
ax2.set_title("mask")
ax2.imshow(mask_dataset, cmap = "gray")
ax3.set_title("all_masked_a")
ax3.imshow(all_masked_a, cmap = "gray")

plt.show()

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/867274275.py in <cell line: 0>()
      1 fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(15, 15))
      2 ax1.set_title("all_a")
----> 3 ax1.imshow(all_a, cmap = "gray")
      4 ax2.set_title("mask")
      5 ax2.imshow(mask_dataset, cmap = "gray")

NameError: name 'all_a' is not defined

## === cell 17
def rle(output):
    output = output.view(-1)
    flat_img = np.where(output > 0.4, 1, 0).astype(np.uint8)
    temp = np.insert(flat_img,0,0)
    temp = temp[0:len(temp)-1]
    starts = np.array((flat_img[:-1] == 1) & (temp[:-1] == 0))
    ends = np.array((flat_img[:-1] == 1) & (flat_img[1:] == 0))
    starts_ix = np.where(starts)[0]
    ends_ix = np.where(ends)[0]
    lengths = ends_ix - starts_ix
    output = None
    flat_img = None
    temp = None
    starts = None
    ends = None
    ends_ix = None
    del output
    del flat_img
    del temp
    del starts
    del ends
    del ends_ix
    return " ".join(map(str, sum(zip(starts_ix, lengths), ())))

## === cell 18
submission = defaultdict(list)
submission["Id"].append("a")
submission["Predicted"].append(rle(all_masked_a))
submission["Id"].append("b")
submission["Predicted"].append(rle(all_masked_a))
mask_dataset = None
all_a = None
all_masked_a = None
outputs = None
del mask_dataset
del all_a
del all_masked_a
del outputs
gc.collect()

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3083339038.py in <cell line: 0>()
      1 submission = defaultdict(list)
      2 submission["Id"].append("a")
----> 3 submission["Predicted"].append(rle(all_masked_a))
      4 submission["Id"].append("b")
      5 submission["Predicted"].append(rle(all_masked_a))

NameError: name 'all_masked_a' is not defined

## === cell 19
pd.DataFrame.from_dict(submission).to_csv("/kaggle/working/submission.csv", index=False)
pd.DataFrame.from_dict(submission)

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/763773700.py in <cell line: 0>()
----> 1 pd.DataFrame.from_dict(submission).to_csv("/kaggle/working/submission.csv", index=False)
      2 pd.DataFrame.from_dict(submission)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in from_dict(cls, data, orient, dtype, columns)
   1915 
   1916         if orient != "tight":
-> 1917             return cls(data, index=index, columns=columns, dtype=dtype)
   1918         else:
   1919             realdata = data["data"]

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __init__(self, data, index, columns, dtype, copy)
    776         elif isinstance(data, dict):
    777             # GH#38939 de facto copy defaults to False only in non-dict cases
--> 778             mgr = dict_to_mgr(data, index, columns, dtype=dtype, copy=copy, typ=manager)
    779         elif isinstance(data, ma.MaskedArray):
    780             from numpy.ma import mrecords

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in dict_to_mgr(data, index, columns, dtype, typ, copy)
    501             arrays = [x.copy() if hasattr(x, "dtype") else x for x in arrays]
    502 
--> 503     return arrays_to_mgr(arrays, columns, index, dtype=dtype, typ=typ, consolidate=copy)
    504 
    505 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in arrays_to_mgr(arrays, columns, index, dtype, verify_integrity, typ, consolidate)
    112         # figure out the index, if necessary
    113         if index is None:
--> 114             index = _extract_index(arrays)
    115         else:
    116             index = ensure_index(index)

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in _extract_index(data)
    675         lengths = list(set(raw_lengths))
    676         if len(lengths) > 1:
--> 677             raise ValueError("All arrays must be of the same length")
    678 
    679         if have_dicts:

ValueError: All arrays must be of the same length
