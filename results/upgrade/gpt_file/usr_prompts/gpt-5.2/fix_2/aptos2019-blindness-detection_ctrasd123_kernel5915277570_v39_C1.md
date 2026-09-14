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

0.8338470714182296

# 6. Current score

0.42872

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.42872) has done: 'I remove the hard dependency on `tensorboardX` (it isn’t installed) by switching to PyTorch’s built-in `SummaryWriter` when available, falling back to a no-op writer otherwise. I also fix the inference pipeline so it no longer crashes on a missing external pretrained `.pkl` file by training the same DenseNet201-based model on the provided `train.csv`/`train_images` and then running prediction on `test_images`. To keep the core logic intact, I preserve the same model class and ordinal-BCE style prediction (`sigmoid` + `get_preds`), only adding the minimal training code needed to generate weights. Finally, I ensure a valid `/kaggle/working/submission.csv` is written with the required `id_code,diagnosis` columns and correct row alignment.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import os.path as osp
import sys
import time
import datetime
import math
import random
import csv

from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter  # noqa: F401
except Exception:

    class SummaryWriter:  # minimal no-op fallback
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def add_histogram(self, *args, **kwargs):
            pass

        def close(self):
            pass


SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DATA_ROOT = "/kaggle/input/aptos2019-blindness-detection"
TRAIN_CSV = osp.join(DATA_ROOT, "train.csv")
TEST_CSV = osp.join(DATA_ROOT, "test.csv")
TRAIN_IMG_DIR = osp.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = osp.join(DATA_ROOT, "test_images")
SUB_PATH = "/kaggle/working/submission.csv"



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/tensorboard/compat/__init__.py in tf()
     41     try:
---> 42         from tensorboard.compat import notf  # noqa: F401
     43     except ImportError:

ImportError: cannot import name 'notf' from 'tensorboard.compat' (/usr/local/lib/python3.11/dist-packages/tensorboard/compat/__init__.py)

During handling of the above exception, another exception occurred:

AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
test_df = pd.read_csv(TEST_CSV)
test_files = (test_df["id_code"].astype(str) + ".png").tolist()

train_df = pd.read_csv(TRAIN_CSV)
train_files = (train_df["id_code"].astype(str) + ".png").tolist()
train_labels = train_df["diagnosis"].astype(int).tolist()

print("train:", len(train_files), "test:", len(test_files))




## === cell 2
class Baseline_single(nn.Module):
    def __init__(self, num_classes, loss_type="single BCE", **kwargs):
        super(Baseline_single, self).__init__()
        self.loss_type = loss_type
        resnet50 = torchvision.models.densenet201(pretrained=False)
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
def crop_image_from_gray(img, tol=7):
    if img is None:
        return img
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
        img = np.stack([img1, img2, img3], axis=-1)
        return img
    return img


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
    height, width = image.shape[0], image.shape[1]
    x0, x1, y0, y1 = circleCrop(c_x, c_y, radius, height, width)
    trimed = image[y0:y1, x0:x1, :]
    return trimed


PARAM = 92


def Radius_Reduction(img, PARAM):
    h, w, c = img.shape
    frame = np.zeros((h, w, c), dtype=np.uint8)
    cv2.circle(
        frame,
        (int(math.floor(w / 2)), int(math.floor(h / 2))),
        int(math.floor((h * PARAM) / float(2 * 100))),
        (255, 255, 255),
        -1,
    )
    frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img1 = cv2.bitwise_and(img, img, mask=frame1)
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
        raise FileNotFoundError(fn)
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
    def __init__(self, filenames, labels=None, transform=None, is_train=False):
        self.filenames = list(filenames)
        self.labels = None if labels is None else list(labels)
        self.transform = transform
        self.is_train = is_train

    def __getitem__(self, index):
        fn = self.filenames[index]
        if self.is_train:
            path = osp.join(TRAIN_IMG_DIR, fn)
        else:
            path = osp.join(TEST_IMG_DIR, fn)

        try:
            img = open_img(path, 530)
        except Exception:
            img = cv2.imread(path)
            if img is None:
                raise FileNotFoundError(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = crop_image_from_gray(img)
            img = cv2.resize(img, (512, 512))

        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        img_id = fn[:-4]
        if self.labels is None:
            return img, img_id
        else:
            return img, int(self.labels[index])

    def __len__(self):
        return len(self.filenames)




## === cell 4
if __name__ == "__main__":
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    if use_gpu:
        cudnn.benchmark = True

    transform2 = transforms.Compose(
        [
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
        ]
    )

    def encode_ordinal(y, num_classes=5):
        y = int(y)
        return torch.tensor(
            [1.0 if k < y else 0.0 for k in range(num_classes)], dtype=torch.float32
        )

    class OrdinalBCELoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.bce = nn.BCEWithLogitsLoss()

        def forward(self, logits, targets):
            return self.bce(logits, targets)

    train_ds = eye_dataset_circle(
        train_files, labels=train_labels, transform=transform2, is_train=True
    )
    test_ds = eye_dataset_circle(
        test_files, labels=None, transform=transform2, is_train=False
    )

    train_loader = DataLoader(
        train_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=use_gpu
    )
    test_loader = DataLoader(
        test_ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=use_gpu
    )

    net = Baseline_single(num_classes=5).to(device)

    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    criterion = OrdinalBCELoss()

    net.train()
    epochs = 2  # keep minimal to fit 600s while still producing non-random predictions
    for ep in range(epochs):
        running = 0.0
        for imgs, y in tqdm(train_loader, desc=f"train epoch {ep+1}/{epochs}"):
            imgs = imgs.to(device, non_blocking=True)
            targets = torch.stack([encode_ordinal(v, 5) for v in y]).to(
                device, non_blocking=True
            )

            optimizer.zero_grad(set_to_none=True)
            logits = net(imgs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0)
        print(f"epoch {ep+1} loss: {running/len(train_ds):.6f}")

    net.eval()
    ids = []
    preds = []
    with torch.no_grad():
        for imgs, img_id in tqdm(test_loader, desc="infer"):
            imgs = imgs.to(device, non_blocking=True)
            out = net(imgs)
            pred = get_preds((torch.sigmoid(out) > 0.5).cpu().numpy())
            ids.extend(list(img_id))
            preds.extend(list(pred.astype(int)))

    sub = pd.DataFrame({"id_code": ids, "diagnosis": preds})
    sub = test_df[["id_code"]].merge(sub, on="id_code", how="left")
    sub["diagnosis"] = sub["diagnosis"].fillna(0).astype(int)
    sub.to_csv(SUB_PATH, index=False)
    print("Wrote:", SUB_PATH, "shape:", sub.shape)
    print(sub["diagnosis"].value_counts().sort_index())
