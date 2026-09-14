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

matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-image==0.25.2
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

0.1206642791156834

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
class AverageMeter(object):
    def __init__(self):
        self.sum = 0
        self.n = 0

    def update(self, x, n=1):
        self.sum += float(x)
        self.n += n

    def reset(self):
        self.sum = 0
        self.n = 0

    def get_value(self):
        if self.n:
            return self.sum / self.n
        return 0

## === cell 1
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_model_parameters(model):
    total_parameters = 0
    for layer in list(model.parameters()):
        layer_parameter = 1
        for l in list(layer.size()):
            layer_parameter *= l
        total_parameters += layer_parameter
    return total_parameters


def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        out = F.relu6(x + 3., self.inplace) / 6.
        return out * x


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SqueezeBlock(nn.Module):
    def __init__(self, exp_size, divide=4):
        super(SqueezeBlock, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(exp_size, exp_size // divide),
            nn.ReLU(inplace=True),
            nn.Linear(exp_size // divide, exp_size),
            h_sigmoid()
        )

    def forward(self, x):
        batch, channels, height, width = x.size()
        out = F.avg_pool2d(x, kernel_size=[height, width]).view(batch, -1)
        out = self.dense(out)
        out = out.view(batch, channels, 1, 1)

        return out * x


class MobileBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size, stride, nonLinear, SE, exp_size):
        super(MobileBlock, self).__init__()
        self.out_channels = out_channels
        self.nonLinear = nonLinear
        self.SE = SE
        padding = (kernal_size - 1) // 2

        self.use_connect = stride == 1 and in_channels == out_channels

        if self.nonLinear == "RE":
            activation = nn.ReLU
        else:
            activation = h_swish

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, exp_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(exp_size),
            activation(inplace=True)
        )
        self.depth_conv = nn.Sequential(
            nn.Conv2d(exp_size, exp_size, kernel_size=kernal_size, stride=stride, padding=padding, groups=exp_size),
            nn.BatchNorm2d(exp_size),
        )

        if self.SE:
            self.squeeze_block = SqueezeBlock(exp_size)

        self.point_conv = nn.Sequential(
            nn.Conv2d(exp_size, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            activation(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.depth_conv(out)

        if self.SE:
            out = self.squeeze_block(out)

        out = self.point_conv(out)

        if self.use_connect:
            return x + out
        else:
            return out


class MobileNetV3(nn.Module):
    def __init__(self, model_mode="LARGE", num_classes=1000, multiplier=1.0, dropout_rate=0.0):
        super(MobileNetV3, self).__init__()
        self.num_classes = num_classes

        if model_mode == "LARGE":

            layers = [
                [16, 16, 3, 1, "RE", False, 16],
                [16, 24, 3, 1, "RE", False, 64],
                [24, 24, 3, 1, "RE", False, 72],
                [24, 40, 5, 1, "RE", True, 72],
                [40, 40, 5, 1, "RE", True, 120],

                [40, 40, 5, 1, "RE", True, 120],
                [40, 80, 3, 1, "HS", False, 240],
                [80, 80, 3, 1, "HS", False, 200],
                [80, 80, 3, 1, "HS", False, 184],
                [80, 80, 3, 1, "HS", False, 184],

                [80, 112, 3, 1, "HS", True, 480],
                [112, 112, 3, 1, "HS", True, 672],
                [112, 160, 5, 1, "HS", True, 672],
                [160, 160, 5, 1, "HS", True, 672],
                [160, 160, 5, 1, "HS", True, 960],
            ]
            init_conv_out = _make_divisible(16 * multiplier)
            print(f"init_conv_out={init_conv_out}")
            self.init_conv = nn.Sequential(
                nn.Conv2d(in_channels=RandomOpt().Z_DIM, out_channels=init_conv_out, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(init_conv_out),
                h_swish(inplace=True),
            )

            self.block = []
            for in_channels, out_channels, kernal_size, stride, nonlinear, se, exp_size in layers:
                in_channels = _make_divisible(in_channels * multiplier)
                out_channels = _make_divisible(out_channels * multiplier)
                exp_size = _make_divisible(exp_size * multiplier)
                self.block.append(MobileBlock(in_channels, out_channels, kernal_size, stride, nonlinear, se, exp_size))
            self.block = nn.Sequential(*self.block)

            out_conv1_in = _make_divisible(160 * multiplier)
            out_conv1_out = _make_divisible(960 * multiplier)
            self.out_conv1 = nn.Sequential(
                nn.Conv2d(out_conv1_in, out_conv1_out, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_conv1_out),
                h_swish(inplace=True),
            )

            out_conv2_in = _make_divisible(960 * multiplier)
            out_conv2_out = _make_divisible(1280 * multiplier)
            self.out_conv2 = nn.Sequential(
                nn.Conv2d(out_conv2_in, out_conv2_out, kernel_size=1, stride=1),
                h_swish(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Conv2d(out_conv2_out, self.num_classes, kernel_size=1, stride=1),
            )

        elif model_mode == "SMALL":

            layers = [
                [16, 16, 3, 1, "RE", True, 16],
                [16, 24, 3, 1, "RE", False, 72],
                [24, 24, 3, 1, "RE", False, 88],
                [24, 40, 5, 1, "RE", True, 96],
                [40, 40, 5, 1, "RE", True, 240],
                [40, 40, 5, 1, "RE", True, 240],
                [40, 48, 5, 1, "HS", True, 120],
                [48, 48, 5, 1, "HS", True, 144],
                [48, 96, 5, 1, "HS", True, 288],
                [96, 96, 5, 1, "HS", True, 576],
                [96, 96, 5, 1, "HS", True, 576],
            ]

            init_conv_out = _make_divisible(16 * multiplier)
            self.init_conv = nn.Sequential(
                nn.Conv2d(in_channels=RandomOpt().Z_DIM, out_channels=init_conv_out, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(init_conv_out),
                h_swish(inplace=True),
            )

            self.block = []
            for in_channels, out_channels, kernal_size, stride, nonlinear, se, exp_size in layers:
                in_channels = _make_divisible(in_channels * multiplier)
                out_channels = _make_divisible(out_channels * multiplier)
                exp_size = _make_divisible(exp_size * multiplier)
                self.block.append(MobileBlock(in_channels, out_channels, kernal_size, stride, nonlinear, se, exp_size))
            self.block = nn.Sequential(*self.block)

            out_conv1_in = _make_divisible(96 * multiplier)
            out_conv1_out = _make_divisible(576 * multiplier)
            self.out_conv1 = nn.Sequential(
                nn.Conv2d(out_conv1_in, out_conv1_out, kernel_size=1, stride=1),
                SqueezeBlock(out_conv1_out),
                nn.BatchNorm2d(out_conv1_out),
                h_swish(inplace=True),
            )

            out_conv2_in = _make_divisible(576 * multiplier)
            out_conv2_out = _make_divisible(1280 * multiplier)
            self.out_conv2 = nn.Sequential(
                nn.Conv2d(out_conv2_in, out_conv2_out, kernel_size=1, stride=1),
                h_swish(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Conv2d(out_conv2_out, self.num_classes, kernel_size=1, stride=1),
            )


        self.my_out_conv2 = nn.Sequential(
            nn.ConvTranspose2d(out_conv2_in, self.num_classes, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)

        )
        self.apply(_weights_init)

    def forward(self, x):
        out = self.init_conv(x)
        out = self.block(out)
        out = self.out_conv1(out)
        batch, channels, height, width = out.size()
        out = self.my_out_conv2(out)

        return out

import argparse

def get_args():


    parser = argparse.ArgumentParser("parameters")

    parser.add_argument("--dataset-mode", type=str, default="IMAGENET", help="(example: CIFAR10, CIFAR100, IMAGENET), (default: IMAGENET)")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs, (default: 100)")
    parser.add_argument("--batch-size", type=int, default=512, help="number of batch size, (default, 512)")
    parser.add_argument("--learning-rate", type=float, default=1e-1, help="learning_rate, (default: 1e-1)")
    parser.add_argument("--dropout", type=float, default=0.8, help="dropout rate, not implemented yet, (default: 0.8)")
    parser.add_argument('--model-mode', type=str, default="LARGE", help="(example: LARGE, SMALL), (default: LARGE)")
    parser.add_argument("--load-pretrained", type=bool, default=False, help="(default: False)")
    parser.add_argument('--evaluate', type=bool, default=False, help="Testing time: True, (default: False)")
    parser.add_argument('--multiplier', type=float, default=1.0, help="(default: 1.0)")
    parser.add_argument('--print-interval', type=int, default=5, help="training information and evaluation information output frequency, (default: 5)")
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--distributed', type=bool, default=False)


    args = parser.parse_args(args=[])

    return args

## === cell 2
import numpy as np
import torch.utils.data as data
import os
import PIL.Image as Image
from tqdm import tqdm
import glob
import torch.nn as nn
from torch import optim
import torch




class RandomOpt():
    def __init__(self):
        self.SHARED_HEIGHT = 4096  # Height to resize all papyri RESIZE圖片高度 最後要提交的檔案要resize回來
        self.BUFFER = 64  # Half-size of papyrus patches we'll use as model inputs 圖片的X/2 Y/2 的值
        self.Z_DIM = 24 # Number of slices in the z direction. Max value is 64 - Z_START
        self.Z_START = 8  # Offset of slices in the z direction
        self.DATA_DIR = "/kaggle/input/vesuvius-challenge-ink-detection"
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.merge_img = True


def resize(img, SHARED_HEIGHT=RandomOpt().SHARED_HEIGHT):
    current_width, current_height = img.size
    aspect_ratio = current_width / current_height
    new_width = int(SHARED_HEIGHT * aspect_ratio)
    new_size = (new_width, SHARED_HEIGHT)
    img = img.resize(new_size)
    return img


def load_mask(split, index, DATA_DIR=RandomOpt().DATA_DIR):
    img = Image.open(f"{DATA_DIR}/{split}/{index}/mask.png").convert('1')
    img = resize(img)
    return torch.from_numpy(np.array(img))

def load_labels(split, index, DATA_DIR=RandomOpt().DATA_DIR):
    img = Image.open(f"{DATA_DIR}/{split}/{index}/inklabels.png")
    img = resize(img)
    return torch.from_numpy(np.array(img)).gt(0).float()

def load_volume(split, index, DATA_DIR=RandomOpt().DATA_DIR, Z_START=RandomOpt().Z_START, Z_DIM=RandomOpt().Z_DIM):
    z_slices = []


    
    z_slices_fnames = sorted(glob.glob(f"{DATA_DIR}/{split}/{index}/surface_volume/*.tif"))[Z_START:Z_START + Z_DIM]

    for z, filename in tqdm(enumerate(z_slices_fnames),desc='load_volume->'+f"{DATA_DIR}/{split}/{index}/surface_volume"):
        img = Image.open(filename)
        img = torch.from_numpy(np.array(resize(img), dtype="float32"))
        if z ==0:
            z_slices =img
        elif z==1:
            z_slices = torch.stack((z_slices,img),dim=0)
        else:
            z_slices=torch.vstack((z_slices,torch.unsqueeze(img,0)))
        del img
        gc.collect()
    return z_slices

    return tmp


def sample_random_location(shape, BUFFER=RandomOpt().BUFFER):
    a=BUFFER
    random_train_x = (shape[0] - BUFFER - 1 - a)*torch.rand(1)+a
    random_train_y = (shape[1] - BUFFER - 1 - a)*torch.rand(1)+a
    random_train_location = torch.stack([random_train_x, random_train_y])
    return random_train_location

def is_in_masked_zone(location, mask):
    return mask[location[0].long(), location[1].long()]

def is_in_val_zone(location, val_location, val_zone_size, BUFFER=RandomOpt().BUFFER):
    x = location[0]
    y = location[1]
    x_match = val_location[0] - BUFFER <= x <= val_location[0] + val_zone_size[0] + BUFFER
    y_match = val_location[1] - BUFFER <= y <= val_location[1] + val_zone_size[1] + BUFFER
    return x_match and y_match

class RandomPatchLocDataset(data.Dataset):
    def __init__(self, mask, val_location, val_zone_size):
        self.mask = mask
        self.val_location = val_location
        self.val_zone_size = val_zone_size
        self.sample_random_location_train = lambda x: sample_random_location(mask.shape)
        self.is_in_mask_train = lambda x: is_in_masked_zone(x, mask)

    def is_proper_train_location(self, location):
        return not is_in_val_zone(location, self.val_location, self.val_zone_size) and self.is_in_mask_train(location)

    def __len__(self):
        return 1280

    def __getitem__(self, index):
        loc = self.sample_random_location_train(0)
        while not self.is_proper_train_location(loc):
            loc = self.sample_random_location_train(0)
        return loc.int().squeeze(1)

## === cell 3
import matplotlib.pyplot as plt
from skimage.transform import resize as resize_ski
import PIL.Image as Image
import gc




class ModelOpt:
    def __init__(self):
        self.GPU_ID = '0'
        self.Z_DIM = RandomOpt().Z_DIM
        self.BUFFER = RandomOpt().BUFFER
        self.SEED = 0
        self.BATCH_SIZE = 20 #每次輸入的立方體取樣個數  input data shape=  [self.BATCH_SIZE, self.Z_DIM ,self.BUFFER*2 , self.BUFFER*2]
        self.LEARNING_RATE =1e-4
        self.TRAINING_EPOCH = 30
        self.LOG_DIR = r'/kaggle/working/'
        self.LOAD_VOLUME = [1,2,3,1,2,3] #讀取幾類樣本
        self.VAL_LOC = (1300, 1000) # 取VAL圖片的起始點
        self.VAL_SIZE = (300, 7000) # VAL圖片的大小  高:1300~1600, 寬:1000~8000
        self.merge_img = True
        self.LOAD_VOLUME = np.unique(self.LOAD_VOLUME) if self.merge_img else self.LOAD_VOLUME #要合併圖片的話 每種讀一次就好
        print(f"self.merge_img = {self.merge_img}  and self.LOAD_VOLUME={self.LOAD_VOLUME}")


class RandomPatchModel():
    def __init__(self,compute_predictions_map_flag=False,opt = ModelOpt()):
        self.opt = opt
        self._setup_all()
        self.compute_predictions_map_flag=compute_predictions_map_flag
        self.net = MobileNetV3(model_mode="SMALL", num_classes=1, multiplier=args.multiplier,
                               dropout_rate=args.dropout).to(self.device)

    def load_data(self,LOAD_VOLUME=ModelOpt().LOAD_VOLUME):
        print(f'load_data with LOAD_VOLUME={LOAD_VOLUME}')
        if not self.compute_predictions_map_flag:
            self.volume_list = [load_volume('train', i) for i in LOAD_VOLUME]
            print(f"volume_list len={len(self.volume_list) } and shape of values = {self.volume_list[0].shape} ")
            self.volume = torch.cat(self.volume_list, dim=2)
            print(f"Finial volume Shape = {self.volume.shape }")
            self.opt.VAL_LOC = (round(self.volume.shape[1] / 2), 0)  # 取VAL圖片的起始點
            self.opt.VAL_SIZE = (300, round(self.volume.shape[2])-1)  # VAL圖片的大小  高:, 寬:
            self.mask_list = [load_mask('train', i) for i in LOAD_VOLUME]
            print(f"mask_list len={len(self.mask_list) } and shape of values = {self.mask_list[0].shape} ")
            self.mask = torch.cat(self.mask_list, dim=1)
            print(f"mask Shape = {self.mask.shape}")
            self.labels_list = [load_labels('train', i) for i in LOAD_VOLUME]
            print(f"labels_list len={len(self.labels_list) } and shape of values = {self.labels_list[0].shape} ")
            self.labels = torch.cat(self.labels_list, dim=1)
            print(f"labels Shape = {self.labels.shape }")

            self.loc_datast = RandomPatchLocDataset(self.mask, val_location= self.opt.VAL_LOC, val_zone_size= self.opt.VAL_SIZE)
            self.loc_loader = data.DataLoader(self.loc_datast, batch_size= self.opt.BATCH_SIZE)
            self.val_loc = []
            for x in range( self.opt.VAL_LOC[0],  self.opt.VAL_LOC[0] +  self.opt.VAL_SIZE[0],  self.opt.BUFFER): # VAL圖片 每個圖片的中心點
                for y in range( self.opt.VAL_LOC[1],  self.opt.VAL_LOC[1] +  self.opt.VAL_SIZE[1],  self.opt.BUFFER):
                    if is_in_masked_zone([torch.tensor(x),torch.tensor(y)], self.mask): # VAL圖片是否在MASK之內
                        self.val_loc.append([[x, y]])
            print(f"\n======> Num of  Patches for Val: {len(self.val_loc)}")


    def _setup_all(self):
        np.random.seed(self.opt.SEED)
        torch.manual_seed(self.opt.SEED)
        torch.cuda.manual_seed_all(self.opt.SEED)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        self.device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"self.device={self.device}")
        self.log_dir = self.opt.LOG_DIR
        self.ckpt = os.path.join(self.log_dir)

    def get_subvolume(self, batch_loc, volume, labels):
        subvolume = []
        label = []
        for l in batch_loc:
            x = l[0]
            y = l[1]
            sv = volume[:, x - self.opt.BUFFER:x + self.opt.BUFFER, y - self.opt.BUFFER:y + self.opt.BUFFER]
            sv = sv / 65535.
            subvolume.append(sv)
            if labels is not None:
                lb = labels[x - self.opt.BUFFER:x + self.opt.BUFFER, y - self.opt.BUFFER:y + self.opt.BUFFER]
                lb = lb.unsqueeze(0)
                label.append(lb)
        subvolume = torch.stack(subvolume)
        if labels is not None:
            label = torch.stack(label)
        return subvolume, label

    def augment_train_data(self, subvolume, label):
        return subvolume, label
    def train_loop(self,merge_img=ModelOpt().merge_img):
        print("=====> Begin training")
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.opt.LEARNING_RATE)
        self.net.train()

        best_val_loss = 100
        best_val_acc = 0
        meter = AverageMeter()

        if merge_img: #一次讀多種img合併
            self.load_data()  # 重新定義資料集
            for epoch in range(self.opt.TRAINING_EPOCH):
                bar = tqdm(enumerate(self.loc_loader), total=len(self.loc_datast) / self.opt.BATCH_SIZE)
                bar.set_description_str(f"Epoch: {epoch}")
                for i, loc in bar:
                    subvolume, label = self.get_subvolume(loc, self.volume, self.labels)  # 取出樣本以及他的labels
                    loss = self._train_step(subvolume.to(self.device), label)
                    meter.update(loss)
                    bar.set_postfix_str(f"Avg loss: {np.round(meter.get_value(), 3)}")

                val_loss, val_acc = self.validataion_loop()
                print(f"======> Val Loss:{np.round(val_loss, 3)} | Val Acc:{np.round(val_acc, 3)} ")
                if val_loss < best_val_loss and val_acc > best_val_acc:
                    torch.save(self.net.state_dict(), os.path.join(self.ckpt, "best.pt"))
                    print("======> Save best val model")

                    best_val_loss = val_loss
                    best_val_acc = val_acc

            del self.loc_datast
            del self.loc_loader
            del self.labels
            del self.labels_list
            del self.mask
            del self.mask_list
            del self.volume
            del self.volume_list
            del self.val_loc
            gc.collect()
        else:
            for dataset_count in self.opt.LOAD_VOLUME:
                self.load_data(LOAD_VOLUME=[dataset_count]) # 重新定義資料集
                for epoch in range(self.opt.TRAINING_EPOCH):
                    bar = tqdm(enumerate(self.loc_loader), total=len(self.loc_datast) / self.opt.BATCH_SIZE)
                    bar.set_description_str(f"Epoch: {epoch}")
                    for i, loc in bar:
                        subvolume, label = self.get_subvolume(loc, self.volume, self.labels) #取出樣本以及他的labels
                        loss = self._train_step(subvolume.to(self.device), label)
                        meter.update(loss)
                        bar.set_postfix_str(f"Avg loss: {np.round(meter.get_value(),3)}")

                    val_loss, val_acc = self.validataion_loop()
                    print(f"======> Val Loss:{np.round(val_loss,3)} | Val Acc:{np.round(val_acc,3)} ")
                    if val_loss < best_val_loss and val_acc > best_val_acc:
                        torch.save(self.net.state_dict(), os.path.join(self.ckpt, "best.pt"))
                        print("======> Save best val model")

                        best_val_loss = val_loss
                        best_val_acc = val_acc

                del self.loc_datast
                del self.loc_loader
                del self.labels
                del self.labels_list
                del self.mask
                del self.mask_list
                del self.volume
                del self.volume_list
                del self.val_loc
                gc.collect()

    def _train_step(self, subvolume, label):
        self.optimizer.zero_grad()
        outputs = self.net(subvolume)
        loss = self.criterion(outputs, label.to(self.device))
        loss.backward()
        self.optimizer.step()
        return loss

    def validataion_loop(self,only_val=False):
        meter_loss = AverageMeter()
        meter_acc = AverageMeter()
        self.net.eval()

        if only_val:
            print(f"validataion_loop  only_val")
            self.load_data(LOAD_VOLUME=[1,2,3])

            for loc in self.val_loc:
                subvolume, label = self.get_subvolume(loc, self.volume, self.labels)
                outputs = self.net(subvolume.to(self.device))
                loss = self.criterion(outputs, label.to(self.device))
                meter_loss.update(loss)
                pred = torch.sigmoid(outputs) > 0.5
                meter_acc.update(
                    (pred == label.to(self.device)).sum(),
                    int(torch.prod(torch.tensor(label.shape)))
                )

            del self.loc_datast
            del self.loc_loader
            del self.labels
            del self.labels_list
            del self.mask
            del self.mask_list
            del self.volume
            del self.volume_list
            del self.val_loc
            gc.collect()

        else:
            for loc in self.val_loc:
                subvolume, label = self.get_subvolume(loc, self.volume, self.labels)
                outputs = self.net(subvolume.to(self.device))
                loss = self.criterion(outputs, label.to(self.device))
                meter_loss.update(loss)
                pred = torch.sigmoid(outputs) > 0.5
                meter_acc.update(
                    (pred == label.to(self.device)).sum(),
                    int(torch.prod(torch.tensor(label.shape)))
                )
            self.net.train()
        return meter_loss.get_value(), meter_acc.get_value()

    def validataion_figplt(self):

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        rect = plt.Rectangle((self.opt.VAL_LOC[1], self.opt.VAL_LOC[0]), self.opt.VAL_SIZE[1], self.opt.VAL_SIZE[0], fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
        font = {'color': 'red',
                'size': 20,
                'family': 'Times New Roman'}
        plt.text(0.1, 0.1, "validataion Area", fontdict=font)
        plt.imshow(self.volume[0])
        plt.show()

    def load_best_ckpt(self,Test=False):
        if Test:
            self.net.load_state_dict(torch.load("/kaggle/input/load-best-test/best.pt"))
        else:
            self.net.load_state_dict(torch.load(os.path.join(self.ckpt, "best.pt")))



def compute_predictions_map(split, index):
    print(f"======> Load data from {split}/{index}")
    test_volume = load_volume(split=split, index=index)
    print(f"======> load_mask from {split}/{index}")
    test_mask = load_mask(split=split, index=index)
    print(f"======> Volume shape: {test_volume.shape}")
    test_locations = []
    BUFFER = model.opt.BUFFER
    stride = BUFFER // 2

    for x in range(BUFFER, test_volume.shape[1] - BUFFER, stride):
        for y in range(BUFFER, test_volume.shape[2] - BUFFER, stride):
            if is_in_masked_zone([torch.tensor(x),torch.tensor(y)], test_mask):
                test_locations.append((x, y))
    print(f"======> {len(test_locations)} test locations (after filtering by mask)")

    predictions_map = torch.zeros((1, 1, test_volume.shape[1], test_volume.shape[2]))
    predictions_map_counts = torch.zeros((1, 1, test_volume.shape[1], test_volume.shape[2]))
    print(f"======> Compute predictions")

    with torch.no_grad():
        bar = tqdm(test_locations,desc='compute_predictions_map->'+f"{DATA_DIR}/{split}/{index}:")
        for loc in bar:
            subvolume, label = model.get_subvolume([loc], test_volume, None)
            outputs = model.net(subvolume.to(model.device))
            pred = torch.sigmoid(outputs)
            predictions_map[:, :, loc[0] - BUFFER : loc[0] + BUFFER, loc[1] - BUFFER : loc[1] + BUFFER] += pred.cpu()
            predictions_map_counts[:, :, loc[0] - BUFFER : loc[0] + BUFFER, loc[1] - BUFFER : loc[1] + BUFFER] += 1
            del subvolume
            del outputs
            del pred
            gc.collect()


    predictions_map /= (predictions_map_counts + 1e-7)
    return predictions_map

def rle(predictions_map, threshold):
    flat_img = predictions_map.flatten()
    flat_img = np.where(flat_img > threshold, 1, 0).astype(np.uint8)

    starts = np.array((flat_img[:-1] == 0) & (flat_img[1:] == 1))
    ends = np.array((flat_img[:-1] == 1) & (flat_img[1:] == 0))
    starts_ix = np.where(starts)[0] + 2
    ends_ix = np.where(ends)[0] + 2
    lengths = ends_ix - starts_ix
    return " ".join(map(str, sum(zip(starts_ix, lengths), ())))





if __name__ == '__main__':
    Image.MAX_IMAGE_PIXELS = None
    torch.cuda.empty_cache()
    args = get_args()
    DATA_DIR = RandomOpt().DATA_DIR
    threshold_a = 0.10
    threshold_b = 0.10
    

    
    model = RandomPatchModel(compute_predictions_map_flag=False)
    model.train_loop()

    model.load_best_ckpt()
    model.net.eval()
    model.criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    loss, acc = model.validataion_loop(only_val=True)
    print(f"Val loss: {np.round(loss, 3)} | Val acc: {np.round(acc, 3)}")


    predictions_map_a = compute_predictions_map(split="test", index="a")
    plt.imshow(predictions_map_a.squeeze() > threshold_a, cmap='gray')
    original_size_a = Image.open(DATA_DIR + "/test/a/mask.png").size
    print(f"original_size_a={original_size_a}")
    print("Do predictions_map_a")
    predictions_map_a = resize_ski(predictions_map_a.squeeze(), original_size_a).squeeze()
    print("Do rle_a")
    rle_a = rle(predictions_map_a, threshold=threshold_a)
    del predictions_map_a
    del original_size_a
    gc.collect()

    predictions_map_b = compute_predictions_map(split="test", index="b")
    plt.imshow(predictions_map_b.squeeze() > threshold_b, cmap='gray')
    original_size_b = Image.open(DATA_DIR + "/test/b/mask.png").size
    print(f"original_size_b={original_size_b}")
    print("Do predictions_map_a")
    predictions_map_b = resize_ski(predictions_map_b.squeeze(), original_size_b).squeeze()
    print("Do rle_b")
    rle_b = rle(predictions_map_b, threshold=threshold_b)
    del predictions_map_b
    del original_size_b
    gc.collect()
    
    print("Id,Predicted\na," + rle_a + "\nb," + rle_b, file=open('/kaggle/working/submission.csv', 'w'))


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2634068806.py in <cell line: 0>()
    359     model = RandomPatchModel(compute_predictions_map_flag=False)
    360 # #     print(summary(model.net, (RandomOpt().Z_DIM, 128,128)))
--> 361     model.train_loop()
    362 
    363     # Validation

/tmp/ipykernel_11/2634068806.py in train_loop(self, merge_img)
    128 
    129         if merge_img: #一次讀多種img合併
--> 130             self.load_data()  # 重新定義資料集
    131 #             print(f"Shape of Train data = {self.volume.shape}")
    132 #             print(f"Shape of Train label = {self.label.shape}")

/tmp/ipykernel_11/2634068806.py in load_data(self, LOAD_VOLUME)
     50             print(f"volume_list len={len(self.volume_list) } and shape of values = {self.volume_list[0].shape} ")
     51             # Here volume: [Z_DIM, SHARED_HEIGHT, W_V1 + W_V2 + ...]
---> 52             self.volume = torch.cat(self.volume_list, dim=2)
     53             print(f"Finial volume Shape = {self.volume.shape }")
     54             self.opt.VAL_LOC = (round(self.volume.shape[1] / 2), 0)  # 取VAL圖片的起始點

TypeError: expected Tensor as element 2 in argument 0, but got list
