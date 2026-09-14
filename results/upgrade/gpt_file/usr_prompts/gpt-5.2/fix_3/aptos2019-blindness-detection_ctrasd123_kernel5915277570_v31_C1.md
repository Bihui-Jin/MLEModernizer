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

geopandas==0.14.4
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

0.8423085504026899

# 6. Current score

0.69006

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0031) has done: 'I remove the hard dependency on `tensorboardX` (it isn’t installed) by falling back to PyTorch’s built-in `SummaryWriter` or a no-op writer, which unblocks imports without changing the model. Then I fix inference so it can run without the missing external pretrained `.pkl` by training the same DenseNet201-based model on `train.csv` images from the provided dataset and using that checkpoint for test predictions. I also fix a couple of runtime issues in the model forward (unnecessary `torch.tensor(x)` copies that can break on GPU) and make prediction writing deterministic and submission-format-correct. The core architecture and prediction post-processing (`sigmoid` + `get_preds`) are preserved; the changes are only to make the pipeline runnable end-to-end and produce `submission.csv`.'
- What this solution (achieved 0.69006) has done: 'I fix the import crash in the first cell by avoiding TensorBoard/TensorFlow-dependent writers entirely (they aren’t needed for training/inference here). Then I correct the model definition bug where DenseNet’s classifier was accidentally removed (the current `children()[:-1]` truncation breaks the forward pass), while keeping the same DenseNet201 backbone and training loop. Finally, I keep the submission writing logic but add a stable prediction mapping from logits to 0–4 classes (argmax over the 5 outputs) to better match the quadratic kappa label space and avoid the near-random `sigmoid`+thresholding behavior that’s causing the very low score. These changes are minimal, unblock end-to-end execution, and should substantially move the score toward the target.'

# 9. Code solution

## === cell 0
import os
import sys
import time
import math
import random
import datetime
import argparse
import os.path as osp
import csv

import numpy as np
import pandas as pd

import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler

import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm


class SummaryWriter:
    def __init__(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def add_histogram(self, *args, **kwargs):
        pass

    def add_image(self, *args, **kwargs):
        pass

    def close(self):
        pass


def seed_everything(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(0)



## === cell 1
name_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
csv_file = csv.reader(open(name_file, "r"))
content = []
for i, line in enumerate(csv_file):
    if i == 0:
        continue
    content.append(line[0] + ".png")

print("Num test images from CSV:", len(content))
print("First few ids:", content[:5])




## === cell 2
class Baseline_single(nn.Module):
    def __init__(self, num_classes, loss_type="single BCE", **kwargs):
        super(Baseline_single, self).__init__()
        self.loss_type = loss_type

        base = torchvision.models.densenet201(weights=None)
        self.base = base.features  # feature extractor
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
        x = F.relu(x, inplace=True)
        if self.loss_type == "single BCE":
            x = self.ap(x)
            x = self.dropout(x)
            x = x.view(x.size(0), -1)
            ys = self.classifiers(x)
        return ys




## === cell 3
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
        img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
        img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
        img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
        return np.stack([img1, img2, img3], axis=-1)


def findCircle(image):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_img = hsv_img[:, :, 2]
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
        x1 = width if math.floor(radius + c_x) > width else math.floor(radius + c_x)
        x0 = 0 if math.floor(c_x - radius < 0) else math.floor(c_x - radius)
    elif math.ceil(c_y - radius) < 0:
        y0 = 0
        y1 = min(math.floor(c_y + radius), height)
        x1 = width if math.floor(radius + c_x) > width else math.floor(radius + c_x)
        x0 = 0 if math.floor(c_x - radius < 0) else math.floor(c_x - radius)
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


def subtract_gaussian_bg_image(im):
    bg = cv2.GaussianBlur(im, (0, 0), 10)
    return cv2.addWeighted(im, 4, bg, -4, 128)


def open_img(fn, size):
    image = cv2.imread(fn)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {fn}")
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


class eye_dataset_circle(Dataset):
    def __init__(self, txt_path, root_dir, transform=None):
        self.imgs = list(txt_path)
        self.transform = transform
        self.root_dir = root_dir

    def __getitem__(self, index):
        fn = self.imgs[index]
        img = open_img(osp.join(self.root_dir, fn), 530)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, fn[:-4]

    def __len__(self):
        return len(self.imgs)


class eye_dataset_circle_train(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __getitem__(self, index):
        id_code = self.df.loc[index, "id_code"]
        label = int(self.df.loc[index, "diagnosis"])
        fn = id_code + ".png"
        img = open_img(osp.join(self.root_dir, fn), 530)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.df)




## === cell 4
if __name__ == "__main__":
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    if use_gpu:
        cudnn.benchmark = True
    else:
        print("Currently using CPU (GPU is highly recommended)")

    transform2 = transforms.Compose([transforms.ToTensor()])

    data_root = "/kaggle/input/aptos2019-blindness-detection"
    train_csv_path = osp.join(data_root, "train.csv")
    test_csv_path = osp.join(data_root, "test.csv")
    train_img_dir = osp.join(data_root, "train_images")
    test_img_dir = osp.join(data_root, "test_images")

    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    net = Baseline_single(num_classes=5).to(device)

    train_data = eye_dataset_circle_train(
        train_df, root_dir=train_img_dir, transform=transform2
    )
    train_loader = DataLoader(
        train_data,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        pin_memory=use_gpu,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    net.train()
    epoch_loss = 0.0
    n_seen = 0
    for data, target in tqdm(train_loader, desc="Training (1 epoch)"):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        out = net(data)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        bs = data.size(0)
        epoch_loss += loss.item() * bs
        n_seen += bs

    print("Train loss:", epoch_loss / max(1, n_seen))

    test_content = (test_df["id_code"].astype(str) + ".png").tolist()
    test_data = eye_dataset_circle(
        test_content, root_dir=test_img_dir, transform=transform2
    )
    dataloader_test = DataLoader(
        test_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=use_gpu
    )

    sub_path = "/kaggle/working/submission.csv"
    with open(sub_path, "w", newline="") as f:
        f_csv = csv.writer(f)
        f_csv.writerow(["id_code", "diagnosis"])

        net.eval()
        with torch.no_grad():
            for _, item in tqdm(
                enumerate(dataloader_test),
                total=len(dataloader_test),
                desc="Predicting",
            ):
                data, name = item
                data = data.to(device, non_blocking=True)
                out = net(data)

                predicted = torch.argmax(out, dim=1).detach().cpu().numpy()
                f_csv.writerow([str(name[0]), int(predicted[0])])

    vc = pd.read_csv(sub_path).diagnosis.value_counts()
    print("Wrote:", sub_path)
    print(vc)
