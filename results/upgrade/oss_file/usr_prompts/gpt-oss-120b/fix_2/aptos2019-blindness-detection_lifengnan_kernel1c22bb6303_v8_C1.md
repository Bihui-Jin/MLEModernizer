# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
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
sklearn-pandas==2.2.0
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
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.5027069774014674

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import argparse
import time
from datetime import timedelta
import shutil
from sklearn import metrics
from PIL import Image, ImageOps
import torch.utils.data as data
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision.transforms as transforms
import csv
import torch.nn.functional as F
import math
from functools import partial
import random
import numbers




## === cell 1
class GroupRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):
        w, h = img_group[0].size
        th, tw = self.size
        out_images = []
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        for img in img_group:
            assert img.size[0] == w and img.size[1] == h
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))
        return out_images


class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupRandomHorizontalFlip(object):
    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img_group, is_flow=False):
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            if self.is_flow:
                for i in range(0, len(ret), 2):
                    ret[i] = ImageOps.invert(ret[i])
            return ret
        else:
            return img_group


class GroupRandomVerticalFlip(object):
    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img_group):
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_TOP_BOTTOM) for img in img_group]
            if self.is_flow:
                for i in range(1, len(ret), 2):
                    ret[i] = ImageOps.invert(ret[i])
            return ret
        else:
            return img_group


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.size()[0] // len(self.mean))
        rep_std = self.std * (tensor.size()[0] // len(self.std))
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)
        return tensor


class GroupScale(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = transforms.Resize(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupOverSample(object):
    def __init__(self, crop_size, scale_size=None):
        self.crop_size = (
            crop_size if not isinstance(crop_size, int) else (crop_size, crop_size)
        )
        if scale_size is not None:
            self.scale_worker = GroupScale(scale_size)
        else:
            self.scale_worker = None

    def __call__(self, img_group):
        if self.scale_worker is not None:
            img_group = self.scale_worker(img_group)
        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size
        offsets = GroupMultiScaleCrop.fill_fix_offset(
            False, image_w, image_h, crop_w, crop_h
        )
        oversample_group = []
        for o_w, o_h in offsets:
            normal_group = []
            flip_group = []
            for i, img in enumerate(img_group):
                crop = img.crop((o_w, o_h, o_w + crop_w, o_h + crop_h))
                normal_group.append(crop)
                flip_crop = crop.copy().transpose(Image.FLIP_LEFT_RIGHT)
                if img.mode == "L" and i % 2 == 0:
                    flip_group.append(ImageOps.invert(flip_crop))
                else:
                    flip_group.append(flip_crop)
            oversample_group.extend(normal_group)
            oversample_group.extend(flip_group)
        return oversample_group


class GroupMultiScaleCrop(object):
    def __init__(
        self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True
    ):
        self.scales = scales if scales is not None else [1, 0.875, 0.75, 0.66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = (
            input_size if not isinstance(input_size, int) else [input_size, input_size]
        )
        self.interpolation = Image.BILINEAR

    def __call__(self, img_group):
        im_size = img_group[0].size
        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [
            img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
            for img in img_group
        ]
        ret_img_group = [
            img.resize((self.input_size[0], self.input_size[1]), self.interpolation)
            for img in crop_img_group
        ]
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [
            self.input_size[1] if abs(x - self.input_size[1]) < 3 else x
            for x in crop_sizes
        ]
        crop_w = [
            self.input_size[0] if abs(x - self.input_size[0]) < 3 else x
            for x in crop_sizes
        ]
        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))
        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(
                image_w, image_h, crop_pair[0], crop_pair[1]
            )
        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(
            self.more_fix_crop, image_w, image_h, crop_w, crop_h
        )
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) / 4
        h_step = (image_h - crop_h) / 4
        ret = [
            (0, 0),
            (4 * w_step, 0),
            (0, 4 * h_step),
            (4 * w_step, 4 * h_step),
            (2 * w_step, 2 * h_step),
        ]
        if more_fix_crop:
            ret.extend(
                [
                    (0, 2 * h_step),
                    (4 * w_step, 2 * h_step),
                    (2 * w_step, 4 * h_step),
                    (2 * w_step, 0),
                    (1 * w_step, 1 * h_step),
                    (3 * w_step, 1 * h_step),
                    (1 * w_step, 3 * h_step),
                    (3 * w_step, 3 * h_step),
                ]
            )
        return ret


class GroupRandomResizedCrop(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_group):
        for attempt in range(10):
            area = img_group[0].size[0] * img_group[0].size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3.0 / 4, 4.0 / 3)
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))
            if random.random() < 0.5:
                w, h = h, w
            if w <= img_group[0].size[0] and h <= img_group[0].size[1]:
                x1 = random.randint(0, img_group[0].size[0] - w)
                y1 = random.randint(0, img_group[0].size[1] - h)
                found = True
                break
        else:
            found = False
            x1 = y1 = 0
        if found:
            out_group = []
            for img in img_group:
                img = img.crop((x1, y1, x1 + w, y1 + h))
                out_group.append(img.resize((self.size, self.size), self.interpolation))
            return out_group
        else:
            scale = GroupScale(self.size, interpolation=self.interpolation)
            crop = GroupRandomCrop(self.size)
            return crop(scale(img_group))


class GroupColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.worker = transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupRandomRotate90(object):
    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img_group):
        v = random.random()
        if v < 0.5:
            ret = [img.rotate(90) for img in img_group]
            if self.is_flow:
                ret2 = []
                for i in range(0, len(ret), 2):
                    ret2.append(ret[i + 1])
                    ret2.append(ret[i])
                ret = ret2
            return ret
        else:
            return img_group


class Stack(object):
    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img_group):
        if not self.is_flow:
            return np.stack(img_group, axis=0)
        else:
            cat_img = []
            for i in range(0, len(img_group), 2):
                cat_img.append(np.stack(img_group[i : i + 2], axis=2))
            return np.stack(cat_img, axis=0)


class ToTorchFormatTensor(object):
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        img = torch.from_numpy(pic).permute(3, 0, 1, 2).contiguous()
        return img.float().div(255) if self.div else img.float()




## === cell 2
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_dict = {
    "34": [resnet34, "A"],
    "50": [resnet50, "B"],
    "101": [resnet101, "B"],
    "152": [resnet152, "B"],
}
weights = "../input/weights/blind_resnet101_best_73.30567080549184.pth"

model = model_dict["101"][0](
    sample_size=112,
    sample_duration=16,
    shortcut_type=model_dict["101"][1],
    num_classes=5,
    dropout=0,
)

model = torch.nn.DataParallel(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

print("Validating...")

Transforms = {
    "val": transforms.Compose(
        [
            GroupScale(128),
            GroupCenterCrop(112),
            Stack(is_flow=False),
            ToTorchFormatTensor(div=True),
            GroupNormalize(
                mean=[110.63666788 / 255.0, 103.16065604 / 255.0, 96.29023126 / 255.0],
                std=[38.7568578 / 255.0, 37.88248729 / 255.0, 40.02898126 / 255.0],
            ),
        ]
    )
}

if os.path.exists(weights):
    ckpt = torch.load(weights, map_location=device)
    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2747840745.py in <cell line: 0>()
      2 
      3 model_dict = {
----> 4     "34": [resnet34, "A"],
      5     "50": [resnet50, "B"],
      6     "101": [resnet101, "B"],

NameError: name 'resnet34' is not defined

## === cell 3
imgs_dict = {}
final = []
imgs = pd.read_csv(
    os.path.join("../input/aptos2019-blindness-detection", "test.csv"), usecols=[0]
)
img_group = []
num = imgs.shape[0]
sub = []

for i in range(num):
    final.append(imgs.loc[i][0])
    img_name = final[i] + ".png"
    img_path = os.path.join(
        "../input/aptos2019-blindness-detection/test_images/", img_name
    )
    img_group.append(Image.open(img_path).convert("RGB"))
    transform_val = Transforms["val"]
    img = transform_val(img_group)
    img_group = []  # reset for next iteration
    img = img.view(-1, 3, 112, 112)
    input_var = Variable(img.to(device))
    with torch.no_grad():
        output = model(input_var)
    _, pred = output.topk(1, dim=1, largest=True, sorted=True)
    pred_1 = pred.t()
    sub.append(str(pred_1.item()))

print(f"Generated predictions for {len(sub)} images")
sample = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")
sample["diagnosis"] = sub
submission_path = "submission.csv"
sample.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1637286297.py in <cell line: 0>()
     15     )
     16     img_group.append(Image.open(img_path).convert("RGB"))
---> 17     transform_val = Transforms["val"]
     18     img = transform_val(img_group)
     19     img_group = []  # reset for next iteration

NameError: name 'Transforms' is not defined
