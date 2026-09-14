# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

3.8

# 3. Installed packages

numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.8747400261442159

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I (1) remove the failing TensorBoard import path by forcing the no-op `SummaryWriter` so the notebook imports cleanly, and (2) fix the missing checkpoint crash by making inference robust when the pretrained `.pkl` is not present (it still run end-to-end and write a valid `submission.csv`). To keep the original core model/inference semantics intact, I won’t change the network architecture or preprocessing; I only add a safe fallback weight initialization and load with `strict=False` to tolerate minor key mismatches if a checkpoint is found. Finally, I ensure the test image path works for both `/kaggle/input/...` and the provided `/kaggle/data/...` mirror, so file I/O doesn’t break in different Kaggle layouts.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is coming from using randomly initialized weights because the checkpoint path doesn’t exist in your environment; that produce near-random predictions and kappa ~0. The smallest score-improving change that preserves your exact model/inference logic is to (1) load an ImageNet-pretrained DenseNet201 backbone (no architecture change) and (2) add a deterministic “ordinal prior” fallback that uses the training label distribution to pick a constant class if the checkpoint is missing (this is legitimate, uses only train.csv labels, and usually beat random toward your target). If a checkpoint is found, behavior stays the same as your current script (we still load it and predict as before). Submission writing and paths stay unchanged, and the script still runs end-to-end under 600s.'
- What this solution (achieved -0.00717) has done: 'Your 0.0 score is because the code usually can’t find/load the trained `.pkl` checkpoint, so it falls back to a constant label (or effectively untrained behavior), which yields near-random/weak kappa. To move the score toward your target with minimal semantic change, I (1) make checkpoint discovery more reliable by also searching `/kaggle/working` and common dataset subfolders, and (2) if the checkpoint is still missing, switch the fallback from a constant median label to a **label-distribution prior** (sampling diagnoses according to `train.csv` frequencies) which is still legitimate and usually improves kappa over a constant without changing the model/inference logic. I also ensure DenseNet201 is instantiated with the current torchvision API while keeping ImageNet-pretrained weights (same intent as your current `pretrained=True`). Submission format/path stays identical (`/kaggle/working/submission.csv`).'
- What this solution (achieved 0.0) has done: 'Your current negative kappa comes from the missing checkpoint path, so the script uses a random label-distribution sampler; that produces weak/near-random agreement. To move the score toward your target with minimal semantic change, I keep your model, preprocessing, and inference exactly the same when a checkpoint exists, but make the no-checkpoint fallback *deterministic* and *slightly informative*: use the ImageNet-pretrained DenseNet201 backbone (already in your code) and then output the single most frequent class from `train.csv` (mode), which typically beats random on QWK. I also make the test-image path resolution consistent across all dataset classes and keep the submission writing identical (`/kaggle/working/submission.csv`). These changes are small, run fast, and should move score upward toward the target without altering the core trained-checkpoint behavior.'

# 9. Code solution

## === cell 0
from __future__ import print_function, absolute_import

import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np
import random
from PIL import Image
import cv2
import csv
import math

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from torch.optim import lr_scheduler

import torchvision
import torchvision.transforms as transforms

from sklearn.metrics import f1_score
from tqdm import tqdm

import pandas as pd


class SummaryWriter:  # noqa: N801
    def __init__(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def add_image(self, *args, **kwargs):
        pass

    def add_histogram(self, *args, **kwargs):
        pass

    def close(self):
        pass


def seed_everything(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(0)



## === cell 1
name_file = "../input/aptos2019-blindness-detection/test.csv"
if not os.path.exists(name_file):
    alt = "/kaggle/data/aptos2019-blindness-detection/test.csv"
    if os.path.exists(alt):
        name_file = alt

csv_file = csv.reader(open(name_file, "r"))
content = []
for line in csv_file:
    content.append(line[0] + ".png")
content = content[1:]




## === cell 2
class Baseline_single(nn.Module):
    def __init__(self, num_classes, loss_type="single BCE", **kwargs):
        super(Baseline_single, self).__init__()
        self.loss_type = loss_type

        try:
            weights = torchvision.models.DenseNet201_Weights.IMAGENET1K_V1
            resnet50 = torchvision.models.densenet201(weights=weights)
        except Exception:
            resnet50 = torchvision.models.densenet201(pretrained=True)

        self.base = nn.Sequential(*list(resnet50.children())[:-1])
        self.feature_dim = 1920
        if self.loss_type == "single BCE":
            self.ap = nn.AdaptiveAvgPool2d(1)
            self.classifiers = nn.Linear(
                in_features=self.feature_dim, out_features=num_classes
            )
            self.sigmoid = nn.Sigmoid()
            self.dropout = nn.Dropout(0.5)
            self.cal_score = nn.Linear(in_features=num_classes, out_features=1)

    def freeze_base(self):
        for p in self.base.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, x1):
        x = self.base(x1)
        if self.loss_type == "single BCE":
            x = self.ap(x)
            x = self.dropout(x)
            x = x.view(x.size(0), -1)
            ys = self.classifiers(x)
        return ys




## === cell 3
def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


def change_size(image):
    b = cv2.threshold(image, 15, 255, cv2.THRESH_BINARY)
    binary_image = b[1]
    binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    print(binary_image.shape)

    x = binary_image.shape[0]
    print("高度x=", x)
    y = binary_image.shape[1]
    print("宽度y=", y)
    edges_x = []
    edges_y = []

    for i in range(x):
        for j in range(y):
            if binary_image[i][j] == 255:
                edges_x.append(i)
                edges_y.append(j)

    left = min(edges_x)
    right = max(edges_x)
    width = right - left

    bottom = min(edges_y)
    top = max(edges_y)
    height = top - bottom

    pre1_picture = image[left : left + width, bottom : bottom + height]
    return pre1_picture


def crop_image1(img, tol=7):
    mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]


def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


def load_ben_color(image, sigmaX=10):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (492, 492))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image


def findCircle(image):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_img = image[:, :, 0]
    s_img = image[:, :, 1]
    v_img = image[:, :, 2]
    height, width = v_img.shape
    mask_v_a = cv2.adaptiveThreshold(
        v_img,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        int(max(height, width) / 16) * 2 + 1,
        1,
    )

    ratio = 128 / min(height, width)
    msk = cv2.resize(
        mask_v_a,
        (int(width * ratio), int(height * ratio)),
        interpolation=cv2.INTER_CUBIC,
    )
    h, w = msk.shape
    msk_expand = np.zeros((3 * h, 3 * w), np.uint8)
    msk_expand[h : 2 * h, w : 2 * w] = msk
    long_edge = max(h, w)
    r0 = round(0.3 * long_edge)
    r1 = round(0.7 * long_edge)
    circles = cv2.HoughCircles(
        msk_expand,
        cv2.HOUGH_GRADIENT,
        1,
        90,
        param1=50,
        param2=5,
        minRadius=r0,
        maxRadius=r1,
    )

    if circles is None:
        c_x = width / 2
        c_y = height / 2
        radius = 0.55 * max(height, width)
    else:
        circles = np.uint16(np.around(circles))
        c_x = (circles[0, 0, 0] - w) / ratio
        c_y = (circles[0, 0, 1] - h) / ratio
        radius = circles[0, 0, 2] / ratio
    return c_x, c_y, radius


def circleCrop(c_x, c_y, radius, height, width):
    if math.floor(radius + c_y) > height:
        y0 = max(math.ceil(c_y - radius), 0)
        y1 = height
        if math.floor(radius + c_x) > width:
            x1 = width
        else:
            x1 = math.floor(radius + c_x)
        if math.floor(c_x - radius < 0):
            x0 = 0
        else:
            x0 = math.floor(c_x - radius)
    elif math.ceil(c_y - radius) < 0:
        y0 = 0
        y1 = min(math.floor(c_y + radius), height)
        if math.floor(radius + c_x) > width:
            x1 = width
        else:
            x1 = math.floor(radius + c_x)
        if math.floor(c_x - radius < 0):
            x0 = 0
        else:
            x0 = math.floor(c_x - radius)
    else:
        y0 = math.ceil(c_y - radius)
        y1 = math.floor(c_y + radius)
        x0 = math.ceil(c_x - radius)
        x1 = math.floor(c_x + radius)

    return x0, x1, y0, y1


def trimFundus(image):
    c_x, c_y, radius = findCircle(image)
    height = image.shape[0]
    width = image.shape[1]
    x0, x1, y0, y1 = circleCrop(c_x, c_y, radius, height, width)
    trimed = image[y0:y1, x0:x1, :]
    return trimed


def load_ben_yuan(image, sigmaX=10):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (512, 512))
    return image


PARAM = 92


def Radius_Reduction(img, PARAM):
    h, w, c = img.shape
    Frame = np.zeros((h, w, c), dtype=np.uint8)
    cv2.circle(
        Frame,
        (int(math.floor(w / 2)), int(math.floor(h / 2))),
        int(math.floor((h * PARAM) / float(2 * 100))),
        (255, 255, 255),
        -1,
    )
    Frame1 = cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)
    img1 = cv2.bitwise_and(img, img, mask=Frame1)
    return img1


def info_image(im):
    cy = im.shape[0] // 2
    midline = im[cy, :]
    midline = np.where(midline > midline.mean() / 3)[0]
    if len(midline) > im.shape[1] // 2:
        x_start, x_end = np.min(midline), np.max(midline)
    else:
        x_start, x_end = im.shape[1] // 10, 9 * im.shape[1] // 10
    cx = (x_start + x_end) / 2
    r = (x_end - x_start) / 2
    return cx, cy, r


def resize_image(im, img_size, augmentation=False):
    cx, cy, r = info_image(im)
    scaling = img_size / (2 * r)
    rotation = 0
    if augmentation:
        scaling *= 1 + 0.3 * (np.random.rand() - 0.5)
        rotation = 360 * np.random.rand()
    M = cv2.getRotationMatrix2D((cx, cy), rotation, scaling)
    M[0, 2] -= cx - img_size / 2
    M[1, 2] -= cy - img_size / 2
    return cv2.warpAffine(im, M, (img_size, img_size))


def subtract_median_bg_image(im):
    k = np.max(im.shape) // 20 * 2 + 1
    bg = cv2.medianBlur(im, k)
    return cv2.addWeighted(im, 4, bg, -4, 128)


def subtract_gaussian_bg_image(im):
    bg = cv2.GaussianBlur(im, (0, 0), 10)
    return cv2.addWeighted(im, 4, bg, -4, 128)


def open_img(fn, size):
    image = cv2.imread(fn)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = resize_image(image, size)
    image = subtract_gaussian_bg_image(image)
    image = Radius_Reduction(image, PARAM)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (512, 512))
    return image


def get_preds(arr):
    mask = arr == 0
    return np.clip(np.where(mask.any(1), mask.argmax(1), 5) - 1, 0, 4)


cnt_t = 0


def _resolve_test_image_path(fn: str) -> str:
    p1 = "/kaggle/input/aptos2019-blindness-detection/test_images/" + fn
    if os.path.exists(p1):
        return p1
    p2 = "/kaggle/data/aptos2019-blindness-detection/test_images/" + fn
    return p2


class eye_dataset(Dataset):
    """docstring for data"""

    def __init__(self, txt_path, transform=None):
        imgs = []
        for img in txt_path:
            imgs.append(img)
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        fn = self.imgs[index]
        img = cv2.imread(_resolve_test_image_path(fn))
        img_copy = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_copy = img.copy()
        try:
            img = trimFundus(img)
            img = cv2.resize(img, (512, 512))
        except Exception as e:
            print(e)
            img = crop_image_from_gray(img_copy)
            img = cv2.resize(img, (512, 512))
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, fn[:-4]

    def __len__(self):
        return len(self.imgs)


class eye_dataset_circle(Dataset):
    """docstring for data"""

    def __init__(self, txt_path, transform=None):
        imgs = []
        for img in txt_path:
            imgs.append(img)
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        fn = self.imgs[index]
        path = _resolve_test_image_path(fn)
        img = open_img(path, 530)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, fn[:-4]

    def __len__(self):
        return len(self.imgs)


class eye_dataset_orl(Dataset):
    """docstring for data"""

    def __init__(self, txt_path, transform=None):
        imgs = []
        for img in txt_path:
            imgs.append(img)
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        fn = self.imgs[index]
        img = cv2.imread(_resolve_test_image_path(fn))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = crop_image_from_gray(img)
        img = cv2.resize(img, (512, 512))
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, fn[:-4]

    def __len__(self):
        return len(self.imgs)




## === cell 4
def _find_checkpoint(preferred_path):
    if os.path.exists(preferred_path):
        return preferred_path

    target_name = os.path.basename(preferred_path)
    base_dirs = [
        "/kaggle/input",
        "/kaggle/data",
        "/kaggle/working",
    ]

    matches = []
    for base in base_dirs:
        if not os.path.exists(base):
            continue
        for root, _, files in os.walk(base):
            if target_name in files:
                matches.append(os.path.join(root, target_name))

    quick_candidates = [
        os.path.join("/kaggle/input", "temp-file", target_name),
        os.path.join("/kaggle/data", "temp-file", target_name),
        os.path.join("/kaggle/input", "aptos2019-blindness-detection", target_name),
        os.path.join("/kaggle/data", "aptos2019-blindness-detection", target_name),
    ]
    for p in quick_candidates:
        if os.path.exists(p):
            matches.append(p)

    if len(matches) > 0:
        matches = list(dict.fromkeys(matches))
        matches.sort(key=lambda p: os.path.getsize(p), reverse=True)
        return matches[0]
    return None


def _get_train_mode_label():
    train_csv = "../input/aptos2019-blindness-detection/train.csv"
    if not os.path.exists(train_csv):
        alt = "/kaggle/data/aptos2019-blindness-detection/train.csv"
        if os.path.exists(alt):
            train_csv = alt
    df = pd.read_csv(train_csv)
    vc = df["diagnosis"].value_counts()
    if len(vc) == 0:
        return 0
    return int(vc.idxmax())


if __name__ == "__main__":

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(0)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    transform2 = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    name_file = "../input/aptos2019-blindness-detection/test.csv"
    if not os.path.exists(name_file):
        alt = "/kaggle/data/aptos2019-blindness-detection/test.csv"
        if os.path.exists(alt):
            name_file = alt

    csv_file = csv.reader(open(name_file, "r"))
    content = []
    for line in csv_file:
        content.append(line[0] + ".png")
    content = content[1:]

    test_data = eye_dataset_orl(content, transform2)
    net = Baseline_single(num_classes=5)
    if use_gpu:
        net = net.cuda()

    preferred_ckpt = "/kaggle/input/temp-file/model_yuan512_dense201_00001_adam_combine_orl_bce_maxest.pkl"
    ckpt_path = _find_checkpoint(preferred_ckpt)

    fallback_label = None
    if ckpt_path is None:
        print(
            f"WARNING: Checkpoint not found at '{preferred_ckpt}' (or under /kaggle/input,/kaggle/data,/kaggle/working). "
            "Falling back to deterministic train-label mode prediction."
        )
        fallback_label = _get_train_mode_label()
        print(f"Using fallback constant diagnosis={fallback_label}")
    else:
        state = torch.load(ckpt_path, map_location=("cuda" if use_gpu else "cpu"))
        if (
            isinstance(state, dict)
            and "state_dict" in state
            and isinstance(state["state_dict"], dict)
        ):
            state = state["state_dict"]
        cleaned = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            cleaned[nk] = v
        net.load_state_dict(cleaned, strict=False)
        print(f"Loaded checkpoint: {ckpt_path}")

    dataloader_test = DataLoader(
        test_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=use_gpu
    )

    sub_path = "/kaggle/working/submission.csv"
    with open(sub_path, "w", newline="") as f:
        f_csv = csv.writer(f)
        f_csv.writerow(["id_code", "diagnosis"])

    with torch.no_grad():
        net.eval()
        for _, item in tqdm(enumerate(dataloader_test), total=len(dataloader_test)):
            data, name = item
            if fallback_label is not None:
                pred_int = int(fallback_label)
            else:
                if use_gpu:
                    data = data.cuda(non_blocking=True)
                out = net(data)
                predicted = get_preds((torch.sigmoid(out) > 0.5).cpu().numpy())
                pred_int = int(predicted[0])

            row = [str(name[0]), str(pred_int)]
            with open(sub_path, "a", newline="") as f:
                f_csv = csv.writer(f)
                f_csv.writerow(row)

    df_sub = pd.read_csv(sub_path)
    print(df_sub.diagnosis.value_counts(dropna=False))
    print(f"Wrote: {sub_path} (rows={len(df_sub)})")
